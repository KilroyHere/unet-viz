/* ==========================================================================
   U-Net Segmentation Demo
   Models: MobileNetV2 (encoder feature extraction)
           DeepLab v3+ ADE20K (150-class semantic segmentation)
           BodyPix v2 (body-part segmentation)
   All run entirely in-browser via TensorFlow.js.
   ========================================================================== */

/* eslint-disable no-undef */

// currentPalette is built dynamically from the DeepLab legend after each
// inference, so it never goes stale and works with any DeepLab base model.
// Format: [{ name: string, color: [r, g, b] }, ...]  (index 0 = background)
let currentPalette = [];

// Build palette from the `legend` object DeepLab returns with every segment() call.
// ADE20K returns { 'class name': [r, g, b] } arrays.
// Pascal VOC returns { 'class name': 'rgb(r, g, b)' } strings.
// This function handles both formats.
function buildPaletteFromLegend(legend) {
  const palette  = [{ name: 'background', color: [0, 0, 0] }];
  const seenKeys = new Set(['0,0,0']);

  Object.entries(legend).forEach(([name, colorVal]) => {
    let r, g, b;
    if (Array.isArray(colorVal)) {
      // ADE20K format: [r, g, b]
      [r, g, b] = colorVal;
    } else if (typeof colorVal === 'string') {
      // Pascal VOC format: "rgb(r, g, b)"
      const m = colorVal.match(/(\d+)[,\s]+(\d+)[,\s]+(\d+)/);
      if (!m) return;
      [r, g, b] = [+m[1], +m[2], +m[3]];
    } else {
      return;
    }
    const key = `${r},${g},${b}`;
    if (!seenKeys.has(key)) {
      seenKeys.add(key);
      palette.push({ name, color: [r, g, b] });
    }
  });
  return palette;
}

// Convert RGBA segmentation map → per-pixel palette index.
function paletteRgbaToIdx(rgba, w, h, palette) {
  const colorToIdx = new Map();
  palette.forEach((cls, i) => {
    colorToIdx.set(`${cls.color[0]},${cls.color[1]},${cls.color[2]}`, i);
  });
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const key = `${rgba[i * 4]},${rgba[i * 4 + 1]},${rgba[i * 4 + 2]}`;
    out[i] = colorToIdx.get(key) ?? 0;
  }
  return out;
}

// ============================================================
// BODYPIX 24-PART PALETTE
// Order MUST match BodyPix's internal part indices exactly:
//   0 left_face  1 right_face
//   2 left_upper_arm_front   3 left_upper_arm_back
//   4 right_upper_arm_front  5 right_upper_arm_back
//   6 left_lower_arm_front   7 left_lower_arm_back
//   8 right_lower_arm_front  9 right_lower_arm_back
//  10 left_hand             11 right_hand
//  12 torso_front           13 torso_back
//  14 left_upper_leg_front  15 left_upper_leg_back
//  16 right_upper_leg_front 17 right_upper_leg_back
//  18 left_lower_leg_front  19 left_lower_leg_back
//  20 right_lower_leg_front 21 right_lower_leg_back
//  22 left_foot             23 right_foot
// ============================================================
const BODY_PARTS = [
  { name: 'Left Face',          color: [110, 64,  170] }, //  0
  { name: 'Right Face',         color: [143, 61,  178] }, //  1
  { name: 'Left Upper Arm F',   color: [178, 60,  178] }, //  2
  { name: 'Left Upper Arm B',   color: [210, 62,  167] }, //  3
  { name: 'Right Upper Arm F',  color: [238, 67,  149] }, //  4
  { name: 'Right Upper Arm B',  color: [255, 78,  125] }, //  5
  { name: 'Left Lower Arm F',   color: [255, 94,  99 ] }, //  6
  { name: 'Left Lower Arm B',   color: [255, 115, 75 ] }, //  7
  { name: 'Right Lower Arm F',  color: [255, 140, 56 ] }, //  8
  { name: 'Right Lower Arm B',  color: [239, 166, 47 ] }, //  9
  { name: 'Left Hand',          color: [217, 193, 49 ] }, // 10
  { name: 'Right Hand',         color: [194, 219, 64 ] }, // 11
  { name: 'Torso Front',        color: [175, 240, 91 ] }, // 12
  { name: 'Torso Back',         color: [135, 245, 87 ] }, // 13
  { name: 'Left Upper Leg F',   color: [96,  247, 96 ] }, // 14
  { name: 'Left Upper Leg B',   color: [64,  243, 115] }, // 15
  { name: 'Right Upper Leg F',  color: [40,  234, 141] }, // 16
  { name: 'Right Upper Leg B',  color: [28,  219, 169] }, // 17
  { name: 'Left Lower Leg F',   color: [26,  199, 194] }, // 18
  { name: 'Left Lower Leg B',   color: [33,  176, 213] }, // 19
  { name: 'Right Lower Leg F',  color: [47,  150, 224] }, // 20
  { name: 'Right Lower Leg B',  color: [65,  125, 224] }, // 21
  { name: 'Left Foot',          color: [84,  101, 214] }, // 22
  { name: 'Right Foot',         color: [99,  81,  195] }, // 23
];

// ============================================================
// U-NET STAGE DEFINITIONS
// displaySize: CSS pixel size for the stage canvas thumbnail.
// Sizes shrink toward the bottleneck to visualise spatial compression.
// ============================================================
const UNET_STAGES = [
  // --- Encoder (left, blue) ---
  { id: 'enc-0', side: 'enc', depth: 0, displaySize: 88,
    label: 'Input',      res: '384×384×3',
    desc: 'The raw RGB image. Every pixel (384×384 = 147,456 of them) carries its original colour. No information has been lost yet.' },
  { id: 'enc-1', side: 'enc', depth: 1, displaySize: 70,
    label: 'Enc 1',      res: '192×192×32',
    desc: 'First strided convolution (stride 2). Resolution halves to 192×192. 32 feature channels detect oriented edges and colour gradients — almost identical to classic Gabor filters.' },
  { id: 'enc-2', side: 'enc', depth: 2, displaySize: 56,
    label: 'Enc 2',      res: '96×96×96',
    desc: 'Resolution halves again to 96×96; 96 channels now detect textures and simple patterns assembled from the edges below.' },
  { id: 'enc-3', side: 'enc', depth: 3, displaySize: 44,
    label: 'Enc 3',      res: '48×48×192',
    desc: 'Down to 48×48. 192 channels encode object-part detectors: eyes, wheels, fur patches. Each "pixel" here represents a 8×8 region of the original image.' },
  // --- Bottleneck (purple) ---
  { id: 'btn',   side: 'btn', depth: 4, displaySize: 36,
    label: 'Bottleneck', res: '24×24×320',
    desc: 'Most compressed point: 24×24 feature maps, 320 channels. Each cell covers ~16×16 pixels of the input. The model has distilled abstract semantic concepts — but spatial detail is nearly gone. This is the "U" base.' },
  // --- Decoder (right, warm) ---
  { id: 'dec-3', side: 'dec', depth: 3, displaySize: 44,
    label: 'Dec 3',      res: '48×48×192',
    desc: 'First 2× upsample from bottleneck. Skip connection from Enc 3 is concatenated here — the decoder receives the fine-grained 48×48 encoder features to recover edge sharpness.' },
  { id: 'dec-2', side: 'dec', depth: 2, displaySize: 56,
    label: 'Dec 2',      res: '96×96×96',
    desc: 'Second upsample + skip from Enc 2. Texture-level information is restored. Segment boundaries begin to sharpen.' },
  { id: 'dec-1', side: 'dec', depth: 1, displaySize: 70,
    label: 'Dec 1',      res: '192×192×32',
    desc: 'Third upsample + skip from Enc 1. Fine edge information recovered. Class boundaries are now nearly pixel-accurate.' },
  { id: 'dec-0', side: 'dec', depth: 0, displaySize: 88,
    label: 'Output Mask',res: '384×384×150',
    desc: 'Full-resolution segmentation map. Each of the 384×384 pixels carries a class probability vector; argmax gives the final label. 150 scene classes for ADE20K, 24 body parts for BodyPix.' },
];

// Which encoder↔decoder pairs to connect with skip arcs (by stage id)
// Listed outermost→innermost; arc heights are inverted so outermost = tallest arc
const SKIP_PAIRS = [
  { enc: 'enc-1', dec: 'dec-1', color: '#4a9eed', label: '192×192 features' },
  { enc: 'enc-2', dec: 'dec-2', color: '#9b72d4', label: '96×96 features'   },
  { enc: 'enc-3', dec: 'dec-3', color: '#e06070', label: '48×48 features'   },
];

// ============================================================
// STATE
// ============================================================
let currentMode    = 'deeplab';
let mobilenetModel = null;      // tf.GraphModel – for encoder feature extraction
let deeplabModel   = null;      // deeplab.SemanticSegmentation
let bodypixModel   = null;      // bodyPix.BodyPix
let encoderNodes   = [];        // MobileNetV2 intermediate node names (4 enc + bottleneck)

let lastSegIdx     = null;      // Int32Array|Uint8Array, class index per pixel (-1=bg for bodypix)
let lastSegW       = 0;
let lastSegH       = 0;
let lastEncFeatures = null;     // Array of {data, shape} from MobileNetV2

let blendAlpha     = 0.7;
let isolatedClass  = null;      // number | null
let hideBg         = false;
let showSkip       = true;      // for the explainer toggle
let activeStageId  = null;      // for keyboard nav

let webcamStream   = null;
let webcamRunning  = false;

// ============================================================
// DOM REFS
// ============================================================
const $ = id => document.getElementById(id);
const inputCanvas  = $('input-canvas');
const outputCanvas = $('output-canvas');
const inputCtx     = inputCanvas.getContext('2d');
const outputCtx    = outputCanvas.getContext('2d');
const webcamEl     = $('webcam');

// BOOT is deferred to end of file so all const declarations are initialised first.

// ============================================================
// MODEL LOADING
// ============================================================
const MOBILENET_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json';

async function loadMobileNet() {
  if (mobilenetModel) return;
  $('load-detail').textContent = 'Loading MobileNetV2 encoder backbone…';
  mobilenetModel = await tf.loadGraphModel(MOBILENET_URL, {
    onProgress: f => setLoadStep(`MobileNetV2 · ${(f * 100).toFixed(0)}%`, 5 + f * 45),
  });
  // Discover Relu6 nodes; pick 4 evenly-spaced ones → enc-1, enc-2, enc-3, btn
  const allNodes = Object.values(mobilenetModel.executor.graph.nodes);
  const relus = allNodes
    .filter(n => /relu/i.test(n.op) || /relu/i.test(n.name))
    .map(n => n.name)
    .filter(n => /Relu6|relu6|Relu$|relu$/i.test(n));

  const NPICK = 4;
  encoderNodes = [];
  const seen = new Set();
  for (let i = 0; i < NPICK && relus.length > 0; i++) {
    const idx = Math.round((i / (NPICK - 1)) * (relus.length - 1));
    if (!seen.has(idx)) { seen.add(idx); encoderNodes.push(relus[idx]); }
  }
  setLoadStep('MobileNetV2 ready.', 50);
}

async function loadDeepLab() {
  if (deeplabModel) return;
  $('load-detail').textContent = 'Loading DeepLab v3+ ADE20K (150 classes)…';
  setLoadStep('DeepLab v3+ ADE20K downloading…', 55);
  deeplabModel = await deeplab.load({
    base: 'ade20k',
    quantizationBytes: 2,
  });
  setLoadStep('DeepLab ready.', 100);
}

async function loadBodyPix() {
  if (bodypixModel) return;
  setLoadStep('Loading BodyPix…', 0);
  setStatus('', 'Loading BodyPix…');
  bodypixModel = await bodyPix.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    multiplier: 0.75,
    quantBytes: 2,
  });
  setLoadStep('BodyPix ready.', 100);
}

// ============================================================
// MODE TOGGLE
// ============================================================
$('mode-deeplab').addEventListener('click', () => switchMode('deeplab'));
$('mode-bodypix').addEventListener('click', () => switchMode('bodypix'));

async function switchMode(mode) {
  if (currentMode === mode) return;
  currentMode = mode;
  $('mode-deeplab').classList.toggle('active', mode === 'deeplab');
  $('mode-bodypix').classList.toggle('active', mode === 'bodypix');

  // Load BodyPix model first (if needed) so inference fires immediately after selectPreset
  if (mode === 'bodypix' && !bodypixModel) {
    $('loading').classList.remove('hidden');
    $('load-fill').style.width = '0%';
    await loadBodyPix();
    $('loading').classList.add('hidden');
  }

  // Swap preset thumbnails to match the new mode, then run inference
  await buildPresets();
  await selectPreset(0);
}

// ============================================================
// INFERENCE
// ============================================================
async function runInference() {
  if (!mobilenetModel) return;

  setStatus('', 'Running…');
  animateArrows(true);
  const t0 = performance.now();

  // --- Encoder features (always from MobileNetV2) ---
  lastEncFeatures = await getEncoderFeatures();

  // --- Segmentation ---
  if (currentMode === 'deeplab') {
    if (!deeplabModel) { setStatus('error', 'DeepLab not loaded'); return; }
    const result = await deeplabModel.segment(inputCanvas);
    // Build per-inference palette from the legend, then convert RGBA → indices
    currentPalette = buildPaletteFromLegend(result.legend);
    lastSegIdx = paletteRgbaToIdx(result.segmentationMap, result.width, result.height, currentPalette);
    lastSegW = result.width;
    lastSegH = result.height;
  } else {
    if (!bodypixModel) { setStatus('error', 'BodyPix not loaded'); return; }
    const seg = await bodypixModel.segmentPersonParts(inputCanvas, {
      internalResolution: 'medium',
      segmentationThreshold: 0.6,
      maxDetections: 5,
    });
    lastSegIdx = new Int32Array(seg.data); // -1 = background
    lastSegW = seg.width;
    lastSegH = seg.height;
  }

  const dt = performance.now() - t0;
  animateArrows(false);
  $('pred-time').textContent = `${dt.toFixed(0)} ms`;
  setStatus('ready', `${currentMode === 'deeplab' ? 'ADE20K' : 'BodyPix'} · ${tf.getBackend()} · ${dt.toFixed(0)} ms`);

  drawUNetStages();
  drawSegOutput();
  buildLegend();
  updateSkipExplainer();
}


// ============================================================
// ENCODER FEATURE EXTRACTION (MobileNetV2 intermediate activations)
// ============================================================
async function getEncoderFeatures() {
  const features = [];
  if (!mobilenetModel || encoderNodes.length === 0) return features;

  // Resize input to 224×224 for MobileNetV2
  const tmp = document.createElement('canvas');
  tmp.width = tmp.height = 224;
  tmp.getContext('2d').drawImage(inputCanvas, 0, 0, 224, 224);

  const outputs = tf.tidy(() => {
    const img = tf.browser.fromPixels(tmp).toFloat().div(127.5).sub(1).expandDims(0);
    const outs = mobilenetModel.execute(img, encoderNodes);
    return (Array.isArray(outs) ? outs : [outs]).map(t => ({
      data: t.dataSync(),
      shape: t.shape.slice(1), // [H, W, C]
    }));
  });

  return outputs;
}

// ============================================================
// U-NET DIAGRAM — BUILD HTML STRUCTURE
// ============================================================
function buildUNetDiagram() {
  const container = $('unet-stages');
  container.innerHTML = '';

  UNET_STAGES.forEach((stage, i) => {
    // Arrow between stages
    if (i > 0) {
      const arrow = document.createElement('div');
      arrow.className = 'unet-arrow';
      arrow.id = `arrow-${i}`;
      arrow.textContent = '→';
      container.appendChild(arrow);
    }

    // Stage card
    const div = document.createElement('div');
    div.className = `unet-stage ${stage.side}-stage`;
    div.id = `stage-${stage.id}`;
    div.style.setProperty('--display-size', stage.displaySize + 'px');
    div.innerHTML = `
      <canvas class="stage-canvas" id="canvas-${stage.id}" width="64" height="64"></canvas>
      <div class="stage-name">${stage.label}</div>
      <div class="stage-res">${stage.res}</div>
    `;
    div.addEventListener('click', () => {
      selectStage(stage.id);
      $('sdp-name').textContent = stage.label + ' — ' + stage.res;
      $('sdp-body').textContent = stage.desc;
    });
    container.appendChild(div);
  });

  // Draw skip arcs after the layout settles
  requestAnimationFrame(() => requestAnimationFrame(drawSkipConnections));
}

// ============================================================
// U-NET DIAGRAM — DRAW STAGE CANVASES AFTER INFERENCE
// ============================================================
function drawUNetStages() {
  const encStageIds = ['enc-0', 'enc-1', 'enc-2', 'enc-3', 'btn'];

  // Encoder: draw input image on enc-0, feature heatmaps on enc-1..btn
  const enc0 = $('canvas-enc-0');
  if (enc0) {
    enc0.width  = inputCanvas.width;
    enc0.height = inputCanvas.height;
    enc0.getContext('2d').drawImage(inputCanvas, 0, 0, enc0.width, enc0.height);
  }

  for (let i = 0; i < lastEncFeatures.length; i++) {
    const id = encStageIds[i + 1]; // enc-1, enc-2, enc-3, btn (+1 because enc-0 = raw input)
    const cv = $(`canvas-${id}`);
    if (!cv || !lastEncFeatures[i]) continue;
    const feat = lastEncFeatures[i];
    const [H, W, C] = feat.shape;
    cv.width = W; cv.height = H;
    drawTopFilterHeatmap(cv.getContext('2d'), feat.data, H, W, C);
  }

  // Decoder: show progressive refinement across each upsample stage.
  // Each stage's "effective resolution" is much lower than its display size,
  // simulating what the decoder actually sees before skip connections restore detail.
  // effectiveRes = how many distinct spatial blocks are visible (lower = blockier = earlier decoder).
  const isBodyPix = (currentMode === 'bodypix');
  const decStages = [
    { id: 'dec-3', displayRes: 48,  effectiveRes: 6  }, // bottleneck upsampled ×2 — still very coarse
    { id: 'dec-2', displayRes: 96,  effectiveRes: 18 }, // one skip merged — moderate detail
    { id: 'dec-1', displayRes: 192, effectiveRes: 56 }, // two skips merged — mostly sharp
    { id: 'dec-0', displayRes: 384, effectiveRes: 384}, // full output — pixel-perfect
  ];

  for (const { id, displayRes, effectiveRes } of decStages) {
    const cv = $(`canvas-${id}`);
    if (!cv) continue;
    cv.width = cv.height = displayRes;
    drawDecoderStage(cv.getContext('2d'), lastSegIdx, lastSegW, lastSegH,
                     displayRes, effectiveRes, isBodyPix);
  }
}

// Draw the single most-activated filter channel at a stage, using the heat colormap
function drawTopFilterHeatmap(ctx, data, H, W, C) {
  // Find the channel with max mean activation
  let bestChan = 0, bestMean = -Infinity;
  const HW = H * W;
  for (let c = 0; c < C; c++) {
    let s = 0;
    for (let p = 0; p < HW; p++) s += data[p * C + c];
    const m = s / HW;
    if (m > bestMean) { bestMean = m; bestChan = c; }
  }
  // Find max for normalization
  let maxV = 0;
  for (let p = 0; p < HW; p++) {
    const v = data[p * C + bestChan];
    if (v > maxV) maxV = v;
  }
  const imgData = ctx.createImageData(W, H);
  const m = Math.max(1e-6, maxV);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const v = data[(y * W + x) * C + bestChan] / m;
    const t = Math.max(0, Math.min(1, v));
    const [r, g, b] = heatColor(t);
    const idx = (y * W + x) * 4;
    imgData.data[idx]     = r;
    imgData.data[idx + 1] = g;
    imgData.data[idx + 2] = b;
    imgData.data[idx + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

// Draw a decoder stage canvas showing the segmentation as it would appear at
// that point in the decoder pipeline.
//
// displayRes  — canvas pixel size (e.g. 48, 96, 192, 384)
// effectiveRes — how many unique spatial samples we draw from (controls blockiness).
//               Low values (6) = early decoder: coarse blobs, no boundaries.
//               High values (384) = final output: pixel-sharp.
//
// This correctly illustrates that without skip connections the decoder can only
// produce coarse class regions, and each skip merges finer spatial detail.
function drawDecoderStage(ctx, segIdx, segW, segH, displayRes, effectiveRes, isBodyPix) {
  if (!segIdx) return;
  const classes = isBodyPix ? BODY_PARTS : currentPalette;

  // Step 1 — build a tiny effectiveRes×effectiveRes label grid by majority-voting
  // within each cell (more accurate than simple nearest-neighbour).
  const eff = Math.min(effectiveRes, displayRes);
  const cellW = segW / eff;
  const cellH = segH / eff;
  const labelGrid = new Int32Array(eff * eff);

  for (let gy = 0; gy < eff; gy++) {
    for (let gx = 0; gx < eff; gx++) {
      // Sample a few pixels from this cell and take the most common class
      const votes = new Map();
      const x0 = Math.floor(gx * cellW), x1 = Math.min(segW, Math.ceil((gx + 1) * cellW));
      const y0 = Math.floor(gy * cellH), y1 = Math.min(segH, Math.ceil((gy + 1) * cellH));
      for (let sy = y0; sy < y1; sy += Math.max(1, Math.floor(cellH / 3))) {
        for (let sx = x0; sx < x1; sx += Math.max(1, Math.floor(cellW / 3))) {
          const c = segIdx[sy * segW + sx];
          votes.set(c, (votes.get(c) || 0) + 1);
        }
      }
      let best = 0, bestV = -1;
      votes.forEach((v, k) => { if (v > bestV) { bestV = v; best = k; } });
      labelGrid[gy * eff + gx] = best;
    }
  }

  // Step 2 — paint the displayRes canvas by nearest-neighbour upscaling from labelGrid.
  // This gives the characteristic "blocky" look of early decoder stages.
  const imgData = ctx.createImageData(displayRes, displayRes);
  // Opacity is slightly reduced for early stages to hint at uncertainty.
  const certainty = Math.min(1, effectiveRes / displayRes + 0.15); // 0.28 for dec-3 → 1 for output

  for (let y = 0; y < displayRes; y++) {
    for (let x = 0; x < displayRes; x++) {
      const gx = Math.min(eff - 1, Math.floor(x * eff / displayRes));
      const gy = Math.min(eff - 1, Math.floor(y * eff / displayRes));
      const cls = labelGrid[gy * eff + gx];

      let r = 20, g = 20, b = 20, a = Math.round(certainty * 220);
      if (isBodyPix) {
        if (cls >= 0 && cls < classes.length) { [r, g, b] = classes[cls].color; }
        else { a = Math.round(certainty * 50); }
      } else {
        if (cls > 0 && cls < classes.length) { [r, g, b] = classes[cls].color; }
        else { r = 20; g = 20; b = 20; a = Math.round(certainty * 160); }
      }

      const idx = (y * displayRes + x) * 4;
      imgData.data[idx]     = r;
      imgData.data[idx + 1] = g;
      imgData.data[idx + 2] = b;
      imgData.data[idx + 3] = a;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

// Thin wrapper still used by the skip explainer
function drawResampledSeg(ctx, segIdx, segW, segH, targetRes, isBodyPix) {
  drawDecoderStage(ctx, segIdx, segW, segH, targetRes, targetRes, isBodyPix);
}

// Black→orange→yellow heat colormap (same as CNN demo)
function heatColor(t) {
  const a = [12, 12, 16], b = [217, 82, 38], c = [255, 235, 153];
  if (t < 0.55) {
    const u = t / 0.55;
    return [a[0] + (b[0]-a[0])*u, a[1] + (b[1]-a[1])*u, a[2] + (b[2]-a[2])*u];
  }
  const u = (t - 0.55) / 0.45;
  return [b[0] + (c[0]-b[0])*u, b[1] + (c[1]-b[1])*u, b[2] + (c[2]-b[2])*u];
}

// ============================================================
// SKIP CONNECTION SVG
// ============================================================
function drawSkipConnections() {
  const svg  = $('skip-svg');
  const wrap = $('unet-wrap');
  if (!svg || !wrap) return;
  const wrapRect = wrap.getBoundingClientRect();
  if (wrapRect.width === 0) return;

  svg.setAttribute('width',  wrapRect.width);
  svg.setAttribute('height', wrapRect.height);
  svg.style.width  = wrapRect.width  + 'px';
  svg.style.height = wrapRect.height + 'px';
  svg.innerHTML = '';

  SKIP_PAIRS.forEach((pair, pairIdx) => {
    const { enc, dec, color, label } = pair;
    const encEl = $(`stage-${enc}`);
    const decEl = $(`stage-${dec}`);
    if (!encEl || !decEl) return;

    const encR = encEl.getBoundingClientRect();
    const decR = decEl.getBoundingClientRect();

    const x1 = encR.left + encR.width  / 2 - wrapRect.left;
    const x2 = decR.left + decR.width  / 2 - wrapRect.left;
    const y   = encR.top  - wrapRect.top + 8;
    const midX = (x1 + x2) / 2;
    // Outermost pair (enc-1↔dec-1, pairIdx=0) gets the tallest arc;
    // innermost (enc-3↔dec-3, pairIdx=2) gets the shortest.
    const arcH = 24 + (SKIP_PAIRS.length - 1 - pairIdx) * 20; // [24, 44, 64]

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', `M ${x1} ${y} Q ${midX} ${y - arcH} ${x2} ${y}`);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', color);
    path.setAttribute('stroke-width', '2');
    path.setAttribute('stroke-dasharray', '5 3');
    path.setAttribute('class', `skip-path${showSkip ? '' : ' inactive'}`);
    path.setAttribute('id', `skip-path-${pairIdx}`);
    svg.appendChild(path);

    // Resolution label at arc apex
    const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    txt.setAttribute('x', midX);
    txt.setAttribute('y', y - arcH - 4);
    txt.setAttribute('text-anchor', 'middle');
    txt.setAttribute('font-size', '9');
    txt.setAttribute('font-family', 'IBM Plex Mono, monospace');
    txt.setAttribute('fill', color);
    txt.setAttribute('opacity', showSkip ? '1' : '0.2');
    txt.textContent = label;
    svg.appendChild(txt);
  });
}

// Redraw skip arcs after window resize
window.addEventListener('resize', () => {
  requestAnimationFrame(drawSkipConnections);
});

// Arrow flowing animation during inference
function animateArrows(on) {
  document.querySelectorAll('.unet-arrow').forEach(a => a.classList.toggle('flowing', on));
}

// ============================================================
// STAGE KEYBOARD NAV + SELECTION
// ============================================================
function selectStage(id) {
  activeStageId = id;
  document.querySelectorAll('.unet-stage').forEach(el => {
    el.classList.toggle('active-stage', el.id === `stage-${id}`);
  });
}

document.addEventListener('keydown', e => {
  if (document.activeElement && document.activeElement.tagName === 'INPUT') return;
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  e.preventDefault();
  const ids = UNET_STAGES.map(s => s.id);
  let idx = ids.indexOf(activeStageId);
  if (idx < 0) idx = 0;
  const next = e.key === 'ArrowLeft' ? Math.max(0, idx - 1) : Math.min(ids.length - 1, idx + 1);
  if (next !== idx) {
    const stage = UNET_STAGES[next];
    selectStage(stage.id);
    $('sdp-name').textContent = stage.label + ' — ' + stage.res;
    $('sdp-body').textContent = stage.desc;
  }
});

// ============================================================
// SEGMENTATION OUTPUT CANVAS
// ============================================================
function drawSegOutput() {
  if (!lastSegIdx) return;
  const W = outputCanvas.width, H = outputCanvas.height;
  const isBodyPix = (currentMode === 'bodypix');
  const classes = isBodyPix ? BODY_PARTS : currentPalette;

  // Draw input image as base
  outputCtx.drawImage(inputCanvas, 0, 0, W, H);

  // Build coloured mask at segmentation resolution
  const imgData = outputCtx.createImageData(lastSegW, lastSegH);
  for (let i = 0; i < lastSegW * lastSegH; i++) {
    const cls = lastSegIdx[i];
    let r = 0, g = 0, b = 0, a = 0;

    if (isBodyPix) {
      if (cls >= 0 && cls < BODY_PARTS.length) {
        [r, g, b] = BODY_PARTS[cls].color;
        a = hideBg ? 255 : 230;
        if (isolatedClass !== null && cls !== isolatedClass) { r = 100; g = 100; b = 100; a = 80; }
      } else {
        a = hideBg ? 0 : 0; // background always hidden in BodyPix
      }
    } else {
      if (cls === 0) {
        a = hideBg ? 0 : Math.round(blendAlpha * 60); r = 0; g = 0; b = 0;
      } else if (cls < currentPalette.length) {
        [r, g, b] = currentPalette[cls].color;
        a = Math.round(blendAlpha * 230);
        if (isolatedClass !== null && cls !== isolatedClass) { r = 120; g = 120; b = 120; a = 60; }
      }
    }

    imgData.data[i * 4]     = r;
    imgData.data[i * 4 + 1] = g;
    imgData.data[i * 4 + 2] = b;
    imgData.data[i * 4 + 3] = a;
  }

  // Upscale the mask to output canvas size
  const tmp = document.createElement('canvas');
  tmp.width = lastSegW; tmp.height = lastSegH;
  tmp.getContext('2d').putImageData(imgData, 0, 0);

  outputCtx.imageSmoothingEnabled = true;
  outputCtx.imageSmoothingQuality = 'high';
  outputCtx.drawImage(tmp, 0, 0, W, H);
}

// Blend slider
$('blend-slider').addEventListener('input', e => {
  blendAlpha = parseFloat(e.target.value);
  drawSegOutput();
});

// Hide background toggle
let hideBgOn = false;
$('hide-bg-btn').addEventListener('click', () => {
  hideBgOn = !hideBgOn;
  hideBg = hideBgOn;
  $('hide-bg-btn').classList.toggle('active-warm', hideBgOn);
  drawSegOutput();
});

// Click-to-inspect pixel
$('output-canvas').addEventListener('mousemove', e => {
  if (!lastSegIdx) return;
  const rect = outputCanvas.getBoundingClientRect();
  const px = Math.floor((e.clientX - rect.left) / rect.width  * lastSegW);
  const py = Math.floor((e.clientY - rect.top)  / rect.height * lastSegH);
  const cls = lastSegIdx[py * lastSegW + px];
  const isBodyPix = (currentMode === 'bodypix');

  const tooltip  = $('pixel-tooltip');
  const swatch   = $('pt-swatch');
  const label    = $('pt-label');

  let name = '—', color = [80, 80, 80];
  if (isBodyPix) {
    if (cls >= 0 && cls < BODY_PARTS.length) { name = BODY_PARTS[cls].name; color = BODY_PARTS[cls].color; }
    else name = 'background';
  } else {
    if (cls >= 0 && cls < currentPalette.length) { name = currentPalette[cls].name; color = currentPalette[cls].color; }
  }

  swatch.style.background = `rgb(${color[0]},${color[1]},${color[2]})`;
  label.textContent = name;
  tooltip.style.left = e.clientX + 'px';
  tooltip.style.top  = e.clientY + 'px';
  tooltip.classList.remove('hidden');
});
$('output-canvas').addEventListener('mouseleave', () => {
  $('pixel-tooltip').classList.add('hidden');
});

// Click to isolate/un-isolate a class
$('output-canvas').addEventListener('click', e => {
  if (!lastSegIdx) return;
  const rect = outputCanvas.getBoundingClientRect();
  const px  = Math.floor((e.clientX - rect.left) / rect.width  * lastSegW);
  const py  = Math.floor((e.clientY - rect.top)  / rect.height * lastSegH);
  const cls = lastSegIdx[py * lastSegW + px];
  if (cls < 0) { isolatedClass = null; }
  else if (isolatedClass === cls) { isolatedClass = null; }
  else { isolatedClass = cls; }
  drawSegOutput();
  buildLegend();
});

// ============================================================
// CLASS LEGEND
// ============================================================
function buildLegend() {
  const list = $('legend-list');
  if (!lastSegIdx) return;
  const isBodyPix = (currentMode === 'bodypix');
  const classes   = isBodyPix ? BODY_PARTS : currentPalette;
  const N = lastSegW * lastSegH;

  // Count pixels per class
  const counts = new Map();
  for (let i = 0; i < N; i++) {
    const c = lastSegIdx[i];
    if (c < 0) continue;
    counts.set(c, (counts.get(c) || 0) + 1);
  }

  // Sort by pixel count descending; skip background (class 0 in VOC)
  const sorted = [...counts.entries()]
    .filter(([c]) => isBodyPix ? c >= 0 : c > 0)
    .sort((a, b) => b[1] - a[1]);

  if (sorted.length === 0) {
    list.innerHTML = '<div class="legend-empty">No objects detected</div>';
    return;
  }

  const maxCount = sorted[0][1];
  list.innerHTML = '';

  sorted.forEach(([classIdx, count]) => {
    const cls = classes[classIdx];
    if (!cls) return;
    const pct = (count / N * 100).toFixed(1);
    const [r, g, b] = cls.color;
    const isIsolated = (isolatedClass === classIdx);
    const isDimmed   = (isolatedClass !== null && !isIsolated);

    const item = document.createElement('div');
    item.className = `legend-item${isIsolated ? ' isolated' : ''}${isDimmed ? ' dimmed' : ''}`;
    item.innerHTML = `
      <div class="li-swatch" style="background:rgb(${r},${g},${b})"></div>
      <div class="li-name">${cls.name}</div>
      <div class="li-pct">${pct}%</div>
      <div class="li-bar-wrap">
        <div class="li-bar" style="width:${(count/maxCount*100).toFixed(1)}%;background:rgb(${r},${g},${b})"></div>
      </div>
    `;
    item.addEventListener('click', () => {
      isolatedClass = (isolatedClass === classIdx) ? null : classIdx;
      drawSegOutput();
      buildLegend();
    });
    list.appendChild(item);
  });
}

$('reset-classes-btn').addEventListener('click', () => {
  isolatedClass = null;
  drawSegOutput();
  buildLegend();
});

// ============================================================
// SKIP CONNECTION EXPLAINER
// ============================================================
function updateSkipExplainer() {
  if (!lastEncFeatures || !lastSegIdx) return;

  // Left: encoder features at level 1 (largest available feature map, most detail)
  // Draw to a temp canvas at native resolution, then scale up to 200×200
  const encCanvas = $('skip-enc');
  encCanvas.width = encCanvas.height = 200;
  const feat = lastEncFeatures[1] || lastEncFeatures[0] || lastEncFeatures[lastEncFeatures.length - 1];
  if (feat) {
    const [H, W, C] = feat.shape;
    const tmp = document.createElement('canvas');
    tmp.width = W; tmp.height = H;
    drawTopFilterHeatmap(tmp.getContext('2d'), feat.data, H, W, C);
    const ctx = encCanvas.getContext('2d');
    ctx.imageSmoothingEnabled = false; // nearest-neighbour → keeps blocky look
    ctx.drawImage(tmp, 0, 0, 200, 200);
  }

  // Right: either blurry (no skip) or intermediate-quality with skip
  redrawSkipExplainerRight();

  // Final: full segmentation at 200×200
  const finalCanvas = $('skip-final');
  finalCanvas.width = finalCanvas.height = 200;
  drawDecoderStage(finalCanvas.getContext('2d'), lastSegIdx, lastSegW, lastSegH,
                   200, 200, currentMode === 'bodypix');
}

function redrawSkipExplainerRight() {
  const canvas = $('skip-right');
  canvas.width = canvas.height = 200;
  const ctx = canvas.getContext('2d');

  if (!showSkip) {
    // Simulate "no skip": bottleneck-only (very coarse 10×10 block grid)
    drawDecoderStage(ctx, lastSegIdx, lastSegW, lastSegH, 200, 10, currentMode === 'bodypix');
    // Dark vignette to reinforce "blurry / lost detail" idea
    ctx.fillStyle = 'rgba(0,0,0,0.22)';
    ctx.fillRect(0, 0, 200, 200);
    $('skip-right-label').textContent = 'Without skip connections';
    $('skip-right-foot').textContent  = 'Decoder bottleneck only: coarse, blurry boundaries';
  } else {
    // With skip: 56-block effective resolution (mid-decoder quality, noticeable improvement)
    drawDecoderStage(ctx, lastSegIdx, lastSegW, lastSegH, 200, 56, currentMode === 'bodypix');
    $('skip-right-label').textContent = 'With skip connections';
    $('skip-right-foot').textContent  = 'Skip features restore spatial detail at each decoder stage';
  }
}

$('toggle-skip-btn').addEventListener('click', () => {
  showSkip = !showSkip;
  $('toggle-skip-btn').textContent = showSkip ? 'Toggle skip ON/OFF' : 'Toggle skip ON/OFF ← (currently OFF)';
  // Update skip path opacity in SVG
  document.querySelectorAll('.skip-path').forEach(p => p.classList.toggle('inactive', !showSkip));
  document.querySelectorAll('.skip-svg text').forEach(t => t.setAttribute('opacity', showSkip ? '0.8' : '0.2'));
  redrawSkipExplainerRight();
});

// ============================================================
// INPUT IMAGE HANDLING
// ============================================================
// Preset images — actual photos served from same origin (no CORS issues).
// ADE20K presets: diverse indoor/outdoor scenes with many semantic classes.
const PRESETS_ADE20K = [
  { name: 'ADE20K demo',  url: 'images/ade20k_official.jpg',  label: 'ADE20K · official demo image' },
  { name: 'bedroom',      url: 'images/bedroom.jpg',          label: 'indoor · bedroom' },
  { name: 'kitchen',      url: 'images/kitchen.jpg',          label: 'indoor · kitchen' },
  { name: 'park',         url: 'images/park.jpg',             label: 'outdoor · park' },
  { name: 'street',       url: 'images/street.jpg',           label: 'outdoor · street crowd' },
];

// BodyPix presets: person-focused photos that show body-part segmentation clearly.
const PRESETS_BODYPIX = [
  { name: 'lecturer',  url: 'images/bodypix_2.jpg', label: 'BodyPix · lecturer' },
  { name: 'group',     url: 'images/bodypix_1.jpg', label: 'BodyPix · group of friends' },
];

// Active preset list — switches with mode
function activePresets() {
  return currentMode === 'bodypix' ? PRESETS_BODYPIX : PRESETS_ADE20K;
}

// Load an image URL onto a canvas, cropping to a square from the centre.
function loadImageToCanvas(url, ctx, w, h) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const s  = Math.min(img.naturalWidth, img.naturalHeight);
      const sx = (img.naturalWidth  - s) / 2;
      const sy = (img.naturalHeight - s) / 2;
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(img, sx, sy, s, s, 0, 0, w, h);
      resolve();
    };
    img.onerror = () => resolve();
    img.src = url;
  });
}

let buildPresetsReady = Promise.resolve(); // resolved immediately if buildPresets hasn't run

function buildPresets() {
  const row = $('preset-row');
  row.innerHTML = '';
  const presets = activePresets();
  const promises = presets.map((p, i) => {
    const btn = document.createElement('button');
    btn.title = p.name; btn.dataset.idx = i;
    const cv = document.createElement('canvas');
    cv.width = cv.height = 56;
    btn.appendChild(cv);
    btn.addEventListener('click', () => selectPreset(i));
    row.appendChild(btn);
    return loadImageToCanvas(p.url, cv.getContext('2d'), 56, 56);
  });
  buildPresetsReady = Promise.all(promises);
  return buildPresetsReady;
}

async function selectPreset(idx) {
  document.querySelectorAll('#preset-row button').forEach((b, i) => b.classList.toggle('active', i === idx));
  stopWebcam();
  const p = activePresets()[idx];
  await loadImageToCanvas(p.url, inputCtx, 384, 384);
  $('input-meta').textContent = p.label;
  await runInference();
}

$('btn-upload').addEventListener('click', () => $('file-input').click());
$('file-input').addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  stopWebcam();
  document.querySelectorAll('#preset-row button').forEach(b => b.classList.remove('active'));
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = async () => {
    const s = Math.min(img.width, img.height);
    inputCtx.clearRect(0, 0, 384, 384);
    inputCtx.drawImage(img, (img.width-s)/2, (img.height-s)/2, s, s, 0, 0, 384, 384);
    $('input-meta').textContent = `upload · ${file.name.slice(0, 20)}`;
    URL.revokeObjectURL(url);
    await runInference();
  };
  img.src = url;
});

$('btn-webcam').addEventListener('click', toggleWebcam);

async function toggleWebcam() {
  if (webcamRunning) { stopWebcam(); return; }
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 384, height: 384 }, audio: false });
    webcamEl.srcObject = webcamStream;
    await webcamEl.play();
    webcamRunning = true;
    $('btn-webcam').textContent = '◉ stop';
    $('btn-webcam').classList.add('active-warm');
    $('input-meta').textContent = 'webcam · live';
    document.querySelectorAll('#preset-row button').forEach(b => b.classList.remove('active'));
    let lastInfer = 0;
    const loop = async () => {
      if (!webcamRunning) return;
      const vw = webcamEl.videoWidth || 384, vh = webcamEl.videoHeight || 384;
      const s = Math.min(vw, vh);
      inputCtx.drawImage(webcamEl, (vw-s)/2, (vh-s)/2, s, s, 0, 0, 384, 384);
      const now = performance.now();
      if (now - lastInfer > 400) { lastInfer = now; await runInference(); }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err); setStatus('error', 'Webcam denied');
  }
}
function stopWebcam() {
  webcamRunning = false;
  if (webcamStream) { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }
  $('btn-webcam').textContent = '◉ webcam';
  $('btn-webcam').classList.remove('active-warm');
}

// ============================================================
// GUIDED TOUR
// ============================================================
const TOUR_STEPS = [
  {
    stageId: 'enc-0',
    title: '① The Input — 384×384 pixels',
    body: 'This is the raw image before anything happens. Every pixel carries its RGB colour. Segmentation must assign a label to ALL 147,456 pixels — that requires very different architecture than simple classification.',
  },
  {
    stageId: 'enc-2',
    title: '② Encoder — compressing features',
    body: 'Each encoder stage halves the resolution and doubles the channels. The model is trading spatial precision for semantic richness. By the bottleneck, it "knows" what objects are present but has nearly forgotten where they are exactly.',
  },
  {
    stageId: 'btn',
    title: '③ The Bottleneck — most compressed',
    body: 'Only 24×24 spatial positions remain. Each "pixel" here represents a 16×16 patch of the original image. The model has distilled abstract object identities. Without skip connections the decoder would produce extremely blurry boundaries.',
  },
  {
    stageId: 'dec-3',
    title: '④ Skip connections — the key innovation',
    body: 'Look at the curved arcs above the diagram. Each skip connection carries high-resolution encoder features directly to the corresponding decoder stage. This is what makes U-Net recover fine boundaries that the bottleneck lost.',
  },
  {
    stageId: 'dec-1',
    title: '⑤ Decoder — recovering spatial detail',
    body: 'Each decoder stage upsamples (×2) and merges the skip connection. The segmentation boundary sharpens at each step. Try clicking "Toggle skip ON/OFF" in the explainer section to see exactly what skip connections contribute.',
  },
  {
    stageId: 'dec-0',
    title: '⑥ Output mask — every pixel labelled',
    body: 'Full 384×384 resolution. Every pixel has one of 21 Pascal VOC class labels (or 24 body parts in BodyPix mode). Click any pixel in the output panel to see its class. Click a class in the legend to isolate it.',
  },
];

let tourActive = false, tourStep = 0;
const tourTooltip = $('tour-tooltip');
const tourBtn     = $('tour-btn');

$('tour-btn').addEventListener('click', () => { if (tourActive) stopTour(); else startTour(); });
$('tt-prev').addEventListener('click', () => { if (tourStep > 0) { tourStep--; showTourStep(); } });
$('tt-next').addEventListener('click', () => { if (tourStep < TOUR_STEPS.length - 1) { tourStep++; showTourStep(); } else stopTour(); });
$('tt-close').addEventListener('click', stopTour);

function startTour() {
  tourActive = true; tourStep = 0;
  tourBtn.textContent = '◼ stop tour'; tourBtn.classList.add('active');
  showTourStep();
}
function stopTour() {
  tourActive = false;
  tourBtn.textContent = '▶ guided tour'; tourBtn.classList.remove('active');
  tourTooltip.classList.add('hidden');
}
function showTourStep() {
  if (!tourActive) return;
  const step = TOUR_STEPS[tourStep];
  $('tt-title').textContent = step.title;
  $('tt-body').textContent  = step.body;
  $('tt-progress').textContent = `step ${tourStep + 1} of ${TOUR_STEPS.length}`;
  $('tt-prev').disabled = (tourStep === 0);
  $('tt-next').textContent = tourStep === TOUR_STEPS.length - 1 ? '✓ finish' : 'next →';
  tourTooltip.classList.remove('hidden');
  if (step.stageId) {
    selectStage(step.stageId);
    const stage = UNET_STAGES.find(s => s.id === step.stageId);
    if (stage) { $('sdp-name').textContent = stage.label + ' — ' + stage.res; $('sdp-body').textContent = stage.desc; }
  }
}

// ============================================================
// UTILITIES
// ============================================================
function setLoadStep(msg, pct) {
  $('load-step').textContent = msg;
  $('load-fill').style.width = (pct ?? 0) + '%';
}
function setStatus(kind, text) {
  const el = $('status');
  el.classList.remove('ready', 'error');
  if (kind) el.classList.add(kind);
  $('status-text').textContent = text;
}

// ============================================================
// BOOT  (placed last so every const / function is initialised)
// ============================================================
(async function main() {
  buildUNetDiagram();
  buildPresets(); // fire-and-forget — thumbnails render while models download

  // Wait for TF.js to pick a backend, with a hard timeout fallback.
  const raceResult = await Promise.race([
    tf.ready().then(() => 'tf-ready'),
    new Promise(r => setTimeout(() => r('timeout'), 5000)),
  ]);
  console.log('[boot] tf race:', raceResult, '| backend:', tf.getBackend());

  // Load MobileNetV2 (encoder features) + DeepLab in parallel
  await Promise.all([loadMobileNet(), loadDeepLab()]);

  setStatus('ready', `Ready · ${tf.getBackend()}`);
  setTimeout(() => $('loading').classList.add('hidden'), 300);

  // Make sure preset thumbnails are done before we copy one to the input canvas
  await buildPresetsReady;

  // Load default preset and run inference
  await selectPreset(0);

  // Auto-highlight the Input stage so the description panel isn't empty
  selectStage('enc-0');
  $('sdp-name').textContent = `${UNET_STAGES[0].label} — ${UNET_STAGES[0].res}`;
  $('sdp-body').textContent = UNET_STAGES[0].desc;
})().catch(err => console.error('[boot] FATAL:', err));
