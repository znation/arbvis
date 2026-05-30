# arbvis

Visualize ML model weights and arbitrary binary files as 2D images that make structure visible at a glance. For `.safetensors`, `.gguf`, and PyTorch `.bin`/`.pth`/`.pt` checkpoints, arbvis renders each tensor at its natural element shape and stacks transformer blocks so corresponding weights line up across layers. For anything else, it falls back to a global [Hilbert curve](https://en.wikipedia.org/wiki/Hilbert_curve) layout that maps one pixel per byte — null regions, ASCII text, compressed payloads, and section boundaries all produce recognizable visual signatures.

## Quick start: visualize a model from the Hub

```sh
arbvis hf://meta-llama/Llama-3.2-1B --tiles ./out
# then open out/index.html in a browser
```

`hf://` inputs are fetched directly — no manual download. The output is a [Leaflet.js](https://leafletjs.com/) tile pyramid you can zoom across; at maximum zoom, one pixel is one tensor element.

In the architectural layout, each tensor is rendered at its 2D element shape. Transformer blocks stack vertically and corresponding sub-tensors (e.g. `q_proj` across every layer) are pixel-aligned, so block-to-block changes — quantization steps, finetune deltas, dead heads — line up as horizontal bands.

## What you see

### Layouts

- **Architectural** — selected automatically when every input is safetensors with transformer-style tensor names. 1 px = 1 element; blocks stack vertically; sub-tensors align across the stack.
- **Byte-Hilbert** — 1 px = 1 byte along a Hilbert curve over the concatenated input bytes. Used for non-safetensors inputs, or anywhere structural metadata is absent.

Override with `--layout auto|arch|hilbert`. `arch` forces structure-aware (falls back to Hilbert if no input is safetensors); `hilbert` forces the legacy global-Hilbert layout for regression checks or for non-tensor data.

### Byte colors

Raw bytes are colored by range (based on [Stairwell's approach](https://stairwell.com/blog/hilbert-curves-visualizing-binary-files-with-color-and-patterns/)):

| Value | Color |
|-------|-------|
| `0x00` | Black |
| `0x01`–`0x1F` | Green (control characters) |
| `0x20`–`0x7E` | Blue (printable ASCII) |
| `0x7F`–`0xFE` | Red (high bytes) |
| `0xFF` | White |

### Diff colors

In `--diff` and `--moe-diff` modes, each pixel encodes a signed delta: **black** for identical, **green** for values that grew, **red** for values that shrank, **white** for non-finite results. Per-tensor RMS normalization (default) keeps the contrast stable across tensors of wildly different scale.

## Supported input formats

- **safetensors** (`.safetensors`) — single file or sharded index
- **GGUF** (`.gguf`) — quantized weights are dequantized for diffing
- **PyTorch pickle** (`.bin` / `.pth` / `.pt`) — parsed without invoking `__reduce__`/`find_class`, so loading untrusted headers is safe
- **Plain binary** — any unrecognized file is rendered byte-for-byte
- **JSON / JSONL** — structure-aware in diff mode (see below)

Diffs match tensors by canonical name across formats, so e.g. a GGUF checkpoint diffs cleanly against the corresponding safetensors release.

## Comparing two models: `--diff`

```sh
arbvis --diff hf://meta-llama/Llama-3.2-1B hf://meta-llama/Llama-3.2-1B-Instruct --tiles ./out
```

Per-tensor element-wise diff between two checkpoints (local files, directories, or `hf://` URLs).

### Diff metric (`--diff-metric`)

- `rms` (default) — per-tensor RMS-normalized signed delta. Stable across tensors of different scale.
- `abs-log` — absolute delta on a log brightness scale. Honest about raw magnitudes.
- `exact` — ternary: identical bytes → black; any change → full saturation.

### Finetune mode (`--finetune` / `--no-finetune`)

When both arguments are `hf://` model URLs, arbvis auto-detects whether the second is declared as a finetune of the first via the HF model card (`base_model` + `base_model_relation`). In finetune mode, tensors present only on the base side render as grey crosshatch (informational); anything new on the finetune side or with a mismatched shape aborts the run. Pass `--finetune` to force the relation on, `--no-finetune` to force it off.

### JSON / JSONL structure-aware diff

When both `--diff` inputs have a `.json` or `.jsonl` extension, arbvis aligns them by structure (object keys, array elements, value boundaries) before computing byte deltas, so a single-key insertion near the top of a file doesn't smear every following byte across the canvas.

## MoE expert-vs-expert matrix: `--moe-diff`

```sh
arbvis --moe-diff hf://mistralai/Mixtral-8x7B-v0.1 --tiles ./out
```

Renders an N×N grid showing the element-wise diff of every expert pair within a single MoE model. Each cell stacks `gate_proj`, `up_proj`, and `down_proj` horizontally; only the upper triangle and diagonal are drawn (the raw diff is antisymmetric).

Supports HF-style per-expert safetensors layouts: Mixtral, Qwen3-MoE, OLMoE, and DeepSeek routed experts. GGUF fused-expert tensors are not yet supported.

## Output destinations

### Tiled viewer (`--tiles DIR`, recommended)

```sh
arbvis file1.bin file2.bin --tiles ./out
```

Generates a Leaflet pyramid (`out/tiles/{z}/{x}/{y}.{ext}` plus `out/index.html`). Advantages over single-image mode:

- Full resolution at every zoom level (1 px = 1 byte / 1 element at max zoom).
- Vector file/tensor boundaries — sharp at every scale, not baked into pixels.
- No size limit — works on files of any size; lower zoom levels are averaged.
- HTML labels positioned at each region's area-weighted centroid.

### Single image and window

```sh
arbvis /bin/ls                      # open a display window
arbvis /bin/ls --output out.png     # write a PNG
cat /dev/urandom | head -c 65536 | arbvis   # read from stdin
```

With no output flag, arbvis opens a display window (press ESC to close). With `--output`, it writes a single PNG. Both are capped at 4096×4096 — larger inputs are subsampled, so use `--tiles` when detail matters.

![arbvis screenshot](arbvis.png)

*Byte-Hilbert single-image mode: multiple unrelated files (images, parquet, mp3, an SSH key) concatenated and rendered together — each file's content signature is immediately distinguishable.*

### HF Hub output

Both `--output` and `--tiles` accept `hf://` URLs and upload directly to the Hub:

```sh
arbvis model.safetensors --output  hf://datasets/me/vis/model.png
arbvis hf://meta-llama/Llama-3.2-1B --tiles hf://datasets/me/vis/llama
```

Note: `--tiles hf://…` uploads `tiles/`, `index.html`, and `labels.json` to the target repo, but the Hub won't render `index.html` on its own. Use `--space` for a working URL.

### Deploy a viewable Space (`--space`)

```sh
arbvis hf://meta-llama/Llama-3.2-1B --space me/llama-vis
```

Renders the tile pyramid and deploys a Docker Space that serves the Leaflet viewer. Tiles live in an auto-created sibling bucket repo (`me/llama-vis_bucket`); the Space itself is stateless and just proxies them.

### Tile format (`--tile-format`)

`avif` (default) — ~30-50% smaller over the wire and supported in every modern browser. Leaf tiles are encoded near-lossless (each pixel is one source byte); pyramid tiles are lossy at quality 85.

`png` — universal fallback for byte-for-byte regression checks or audiences without AVIF support.

## Working with the Hub

`hf://` URLs work as both input and output. Forms accepted:

```
hf://owner/repo[@rev][/path]                     # model (default), optional revision
hf://models/owner/repo[@rev][/path]              # explicit model
hf://datasets/owner/repo[@rev][/path]
hf://spaces/owner/repo[@rev][/path]
hf://buckets/owner/bucket[/path]                 # no revision concept
```

Whole-repo URLs (no `/path`) expand to every file in the repo. Single-file URLs fetch just that file.

### Streaming (`--stream`)

By default, `hf://` inputs are downloaded to the local hf-hub cache before rendering, and tile output is staged on local disk before upload. `--stream` flips both: input bytes are range-fetched per tile, and tiles are pushed to the Hub as they are produced. The disk-backed default is faster and more recoverable; use `--stream` only when input or output data won't fit on local disk.

### Xet xorb visualization (`--show-xet-xorbs`)

```sh
arbvis hf://meta-llama/Llama-3.2-1B --show-xet-xorbs --tiles ./out
```

For xet-backed Hub files, colors each region by the xorb (content-addressed chunk) it was reconstructed from: hue encodes xorb ID, intensity encodes the underlying byte. Useful for seeing how a model's weights are partitioned across the CAS.

## Other useful flags

- `--title TEXT` — title shown in the viewer info panel (defaults to "arbvis" or "arbvis diff").
- `-l, --file-list FILE` — read input paths from `FILE`, one per line; `-` reads from stdin.
- `--regen-html DIR` — rebuild `index.html` for an existing tile directory without re-rendering tiles. Useful after editing the viewer template.
- `--space OWNER/REPO --tiles LOCAL_DIR` (with no input files) — re-deploy an already-rendered tile directory to a Space without re-rendering.

```sh
arbvis --regen-html ./out
arbvis --space me/llama-vis --tiles ./out
```

## Building

Requires Rust (stable).

```sh
cargo build --release
./target/release/arbvis <file> --tiles ./output
```

Or install into your `PATH`:

```sh
cargo install --path .
```

## Credits

Color scheme inspired by [Stairwell's binary visualization post](https://stairwell.com/blog/hilbert-curves-visualizing-binary-files-with-color-and-patterns/). Built on [clap](https://crates.io/crates/clap) (CLI), [image](https://crates.io/crates/image) + [png](https://crates.io/crates/png) + rav1e (tile encoding), [fast_hilbert](https://crates.io/crates/fast_hilbert) (curve mapping), [hf-hub](https://crates.io/crates/hf-hub) + [xet-core-structures](https://crates.io/crates/xet-core-structures) (Hub I/O), [candle-core](https://crates.io/crates/candle-core) (GGUF + PyTorch pickle), [minifb](https://crates.io/crates/minifb) (window display), and [Leaflet.js](https://leafletjs.com/) (the viewer).
