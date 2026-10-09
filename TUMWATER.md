# arbvis — project brief

## Initial prompt

<!-- tumwater:prompt:start -->
# arbvis

Visualize arbitrary binary files in a way that makes structure visible at a glance. arbvis lays bytes out along a [Hilbert curve](https://en.wikipedia.org/wiki/Hilbert_curve) and colors them by value range. Null regions, ASCII text, compressed payloads, and section boundaries all produce recognizable visual signatures. The default 2D mode renders a zoomable image (one pixel per byte); the [3D mode](#3d-mode) lifts the same idea into a volume you can fly through, using opacity to reveal the cube's interior.

**For ML model weights**, use [**modelweightvis**](https://github.com/znation/modelweightvis), built on top of arbvis. arbvis renders `.safetensors` / `.gguf` / `.bin` checkpoints as raw bytes; modelweightvis adds tensor-format parsing, an architectural layout that stacks transformer blocks at each tensor's natural element shape, MoE expert-vs-expert diffs, finetune auto-detection, and dtype-aware coloring. Architecturally, modelweightvis is a thin crate that registers tensor-aware plugins and hooks against arbvis's registry — see [Relationship to modelweightvis](#relationship-to-modelweightvis) below.
<!-- tumwater:prompt:end -->

## Status

<!-- tumwater:status:start -->
_No status yet. The readme loop keeps this section up to date._
<!-- tumwater:status:end -->
