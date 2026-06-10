# arbvis

Visualize arbitrary binary files as 2D images that make structure visible at a glance. arbvis lays bytes out along a [Hilbert curve](https://en.wikipedia.org/wiki/Hilbert_curve) — one pixel per byte — and colors them by value range. Null regions, ASCII text, compressed payloads, and section boundaries all produce recognizable visual signatures.

**For ML model weights**, use [**modelweightvis**](https://github.com/znation/modelweightvis), built on top of arbvis. arbvis renders `.safetensors` / `.gguf` / `.bin` checkpoints as raw bytes; modelweightvis adds tensor-format parsing, an architectural layout that stacks transformer blocks at each tensor's natural element shape, MoE expert-vs-expert diffs, finetune auto-detection, and dtype-aware coloring. Architecturally, modelweightvis is a thin crate that registers tensor-aware plugins and hooks against arbvis's registry — see [Relationship to modelweightvis](#relationship-to-modelweightvis) below.

## Quick start

```sh
arbvis /bin/ls --output ls.png
```

Renders `/bin/ls` as a single Hilbert-curve PNG. With no `--output`, arbvis opens a display window. For zoomable tiles:

```sh
arbvis /tmp/foo.bin --tiles ./out
# then open out/index.html in a browser
```

The output is a [Leaflet.js](https://leafletjs.com/) tile pyramid you can zoom across; at maximum zoom, one pixel is one byte.

## What you see

### Byte-Hilbert layout

1 px = 1 byte along a Hilbert curve over the concatenated input bytes. The curve preserves locality: nearby bytes in the file end up nearby in the image, so contiguous regions (a string table, a compressed payload, an embedded image) appear as coherent blobs rather than scattered noise.

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

In `--diff` mode, each pixel encodes the byte-wise difference between the two inputs. Identical bytes render as black; the larger the delta, the brighter the pixel.

## Supported input formats

- **Plain binary** — anything not specifically detected is rendered byte-for-byte.
- **JSON / JSONL** — structure-aware in diff mode (see below).

Anything else — `.safetensors`, `.gguf`, PyTorch `.bin` — is rendered as plain bytes here. For tensor-format awareness use [modelweightvis](https://github.com/znation/modelweightvis).

## Comparing two files: `--diff`

```sh
arbvis --diff a.bin b.bin --tiles ./out
arbvis --diff hf://owner/repo/a.json hf://owner/repo/b.json --output diff.png
```

Plain-byte diff aligns the two inputs at offset 0 and computes per-byte deltas. Whole directories work too — each file pairs up by name across the two roots.

### JSON / JSONL structure-aware diff

When both `--diff` inputs have a `.json` or `.jsonl` extension, arbvis aligns them by structure (object keys, array elements, value boundaries) before computing byte deltas, so a single-key insertion near the top of a file doesn't smear every following byte across the canvas.

## Output destinations

### Tiled viewer (`--tiles DIR`, recommended)

```sh
arbvis file1.bin file2.bin --tiles ./out
```

Generates a Leaflet pyramid (`out/tiles/{z}/{x}/{y}.{ext}` plus `out/index.html`). Advantages over single-image mode:

- Full resolution at every zoom level (1 px = 1 byte at max zoom).
- Vector file boundaries — sharp at every scale, not baked into pixels.
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
arbvis file.bin --output hf://datasets/me/vis/file.png
arbvis dir/ --tiles hf://datasets/me/vis/dir
```

Note: `--tiles hf://…` uploads `tiles/`, `index.html`, and `labels.json` to the target repo, but the Hub won't render `index.html` on its own. Use `--space` for a working URL.

### Deploy a viewable Space (`--space`)

```sh
arbvis hf://datasets/owner/dataset --space me/dataset-vis
```

Renders the tile pyramid and deploys a Docker Space that serves the Leaflet viewer. Tiles live in an auto-created sibling bucket repo (`me/dataset-vis_bucket`); the Space itself is stateless and just proxies them.

### Tile format (`--tile-format`)

`avif` (default) — ~30–50% smaller over the wire and supported in every modern browser. Leaf tiles are encoded near-lossless (each pixel is one source byte); pyramid tiles are lossy at quality 85.

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

By default, `hf://` inputs are downloaded to the local HF cache (via the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)) before rendering, and tile output is staged on local disk before upload. `--stream` flips both: input bytes are range-fetched per tile, and tiles are pushed to the Hub as they are produced. The disk-backed default is faster and more recoverable; use `--stream` only when input or output data won't fit on local disk.

### Xet xorb visualization (`--show-xet-xorbs`)

```sh
arbvis hf://datasets/owner/dataset --show-xet-xorbs --tiles ./out
```

For xet-backed Hub files, colors each region by the xorb (content-addressed chunk) it was reconstructed from: hue encodes xorb ID, intensity encodes the underlying byte. Useful for seeing how a file is partitioned across the CAS.

modelweightvis layers a dtype-aware element coloring on top of the same xorb hue for `.safetensors` / `.gguf` inputs; arbvis covers the generic byte path.

## Other useful flags

- `--title TEXT` — title shown in the viewer info panel (defaults to `"arbvis"` or `"arbvis diff"`).
- `-l, --file-list FILE` — read input paths from `FILE`, one per line; `-` reads from stdin.
- `--regen-html DIR` — rebuild `index.html` for an existing tile directory without re-rendering tiles. Useful after editing the viewer template.
- `--space OWNER/REPO --tiles LOCAL_DIR` (with no input files) — re-deploy an already-rendered tile directory to a Space without re-rendering.

```sh
arbvis --regen-html ./out
arbvis --space me/vis --tiles ./out
```

## Relationship to modelweightvis

arbvis is the byte-only foundation: Hilbert layout, byte coloring, JSON-aware diff, Hub I/O, tile pyramid, Space deploy, xet xorb path, streaming. It has no knowledge of tensors, model formats, or transformer architecture — `.safetensors` and `.gguf` get the same byte-Hilbert treatment as any other binary.

[modelweightvis](https://github.com/znation/modelweightvis) is a separate crate that extends arbvis through its plugin / hook surface (no fork, no patch): `FormatPlugin` impls parse `.safetensors` / `.gguf` / pickle headers and stuff `ModelInfo` into each source's extension map; `LayoutPlugin` impls add the architectural transformer layout and the MoE summary / CKA panel layouts; `DiffSourceBuilder` adds tensor-aware diffing; option-slot hooks (`MoeSummaryPrep`, `MoeCkaPrep`, `RepoDiffPrep`, `FinetuneDetect`, `SingleImageArchHook`, `PrepareSourcesExtension`) tap CLI dispatch points. The `modelweightvis` binary builds an `arbvis::Registry::with_defaults()`, calls `modelweightvis::register_all(&mut registry)`, and hands off to `arbvis::run`. Same renderer, same Hub I/O, same tile pyramid — just with the tensor-aware plugins registered.

**Which to use:**
- **arbvis** — for non-model binaries (any file format), JSON/JSONL diffs, plain-byte diffs, the xet xorb path on arbitrary content. Smaller dependency footprint (no `candle-core` / `regex` / `zip` / `half`).
- **modelweightvis** — for `.safetensors` / `.gguf` / `.bin` model checkpoints, architectural transformer layout, `--moe-summary` / `--moe-cka` / `--probe`, `--diff-metric`, `--finetune` / `--no-finetune`, `--layout`. Inherits arbvis's full CLI surface (`--tiles`, `--space`, `--stream`, `--show-xet-xorbs`, `--regen-html`, etc.) — no need to use both binaries.

## Building

Requires Rust (stable) and the official Hugging Face [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) on `$PATH` (install via `pip install -U huggingface_hub`, `brew install huggingface-cli`, or `curl -LsSf https://hf.co/cli/install.sh | bash`). arbvis shells out to `hf` for every Hub download / upload / sync.

```sh
cargo build --release
./target/release/arbvis <file> --tiles ./output
```

Or install into your `PATH`:

```sh
cargo install --path .
```

For modelweightvis, see the [standalone modelweightvis repo](https://github.com/znation/modelweightvis) — it depends on arbvis via a pinned git revision and inherits arbvis's full CLI surface.

## Credits

Color scheme inspired by [Stairwell's binary visualization post](https://stairwell.com/blog/hilbert-curves-visualizing-binary-files-with-color-and-patterns/). Built on [clap](https://crates.io/crates/clap) (CLI), [image](https://crates.io/crates/image) + [png](https://crates.io/crates/png) + rav1e (tile encoding), [fast_hilbert](https://crates.io/crates/fast_hilbert) (curve mapping), the official Hugging Face [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) (Hub I/O) + [xet-core-structures](https://crates.io/crates/xet-core-structures) (per-tile xet decode), [minifb](https://crates.io/crates/minifb) (window display), and [Leaflet.js](https://leafletjs.com/) (the viewer).
