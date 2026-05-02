## U-Net Visualization Demo

This demo is designed to teach how encoder-decoder segmentation systems work by tying architecture, model behavior, and predictions together in one workflow. I chose a U-Net-style explanation structure with explicit encoder, bottleneck, decoder, and skip stages so users can see where spatial information is compressed and where it is recovered.

My model choices were intentional. I use MobileNetV2 as the shared encoder because it is lightweight and fast enough for in-browser interaction, while still producing meaningful intermediate features. For task heads, I included DeepLab on ADE20K and BodyPix to show two different output spaces: broad scene semantics versus fine-grained human-part segmentation. This lets users compare how similar architecture principles support different segmentation objectives.

I iterated primarily on architecture communication and feature interpretability. Early versions focused only on final masks; later versions added stage-wise visualizations, skip-connection overlays, and a decoder resolution ladder to make multi-scale reconstruction explicit. I then added channel exploration for encoder and bottleneck stages so users can inspect which feature channels are most active and connect representation depth to visual patterns.

The final demo is effective because each feature maps to a conceptual question: what each stage represents, why skip connections matter, how confidence and entropy relate to boundaries, and how changing the decoder changes label semantics. The goal is not just to show segmentation output, but to make the full model pipeline legible.
