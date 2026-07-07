# arbvis

Visualize arbitrary binary files in a way that makes structure visible at a glance. arbvis lays bytes out along a [Hilbert curve](https://en.wikipedia.org/wiki/Hilbert_curve) and colors them by value range. Null regions, ASCII text, compressed payloads, and section boundaries all produce recognizable visual signatures. The default 2D mode renders a zoomable image (one pixel per byte); the [3D mode](#3d-mode) lifts the same idea into a volume you can fly through, using opacity to reveal the cube's interior.

**For ML model weights**, use [**modelweightvis**](https://github.com/znation/modelweightvis), built on top of arbvis. arbvis renders `.safetensors` / `.gguf` / `.bin` checkpoints as raw bytes; modelweightvis adds tensor-format parsing, an architectural layout that stacks transformer blocks at each tensor's natural element shape, MoE expert-vs-expert diffs, finetune auto-detection, and dtype-aware coloring. Architecturally, modelweightvis is a thin crate that registers tensor-aware plugins and hooks against arbvis's registry — see [Relationship to modelweightvis](#relationship-to-modelweightvis) below.

## Quick start

```sh
arbvis /tmp/foo.bin --out ./out
# then serve ./out over HTTP and open index.html in a browser
```

The output is a [Leaflet.js](https://leafletjs.com/) tile pyramid you can zoom across; at maximum zoom, one pixel is one byte. Add `--3d` for the volume viewer:

```sh
arbvis /tmp/foo.bin --3d --out ./out3d
```

To publish either as a live, shareable visualization, swap `--out` for `--space`:

```sh
arbvis hf://datasets/owner/dataset --space me/dataset-vis
```

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

## 3D mode (`--3d`)

`--3d` lays the bytes along a **3D Hilbert curve** inside a cube — the natural generalization of the 2D layout — and emits a self-contained [Three.js](https://threejs.org/) viewer bundle (`index.html`, `volume.bin`, `bricks.bin`, `pagetable.bin`, `meta.json`). It deploys as a Hugging Face Space exactly like 2D (`--space`).

```sh
arbvis model.safetensors --3d --out ./out3d      # local bundle (serve over HTTP)
arbvis hf://datasets/owner/dataset --3d --space me/vis-3d   # deploy a Space
```

Where 2D color is fully opaque, **3D uses opacity to encode density** so you can see *through* the cube to its internal structure instead of just an opaque shell. The viewer is a GPU ray-march of a bounded voxel grid: color encodes the mean byte value (the same [byte-color scheme](#byte-colors) as 2D); opacity comes from an adjustable, log-style **transfer function**. Render and download cost depend on the grid resolution, *not* the input size, so a multi-GB file renders as smoothly as a small one.

- **Opacity source** — *Activity* (default: mean byte "brightness", so null/padding regions fade to transparent and real data stands out) or *Density* (how many bytes fall in each voxel).
- **Volume opacity / Volume contrast / Threshold** sliders tune the transfer function.

**Controls:** drag to rotate · right-drag to pan · scroll to zoom (the camera auto-frames the occupied region on load). Hover or click a region to identify the tensor under the cursor (structured layouts).

**Grid resolution (`--grid N`)** — the voxel cube side, a power of two in `2–512` (default `256`). Higher is more detailed but a larger download (≈ `N³ · 4` bytes; `128³` ≈ 8 MB, `256³` ≈ 64 MB, `512³` ≈ 512 MB).

**Volume resolution (`--volume-res N`)** — build the sparse brick pool at this virtual side (power of two, above `--grid`) instead of the dense grid, so the *volume* can exceed `--grid` for sparse data. Only occupied bricks are stored and streamed into a bounded GPU cache on demand (ray-guided, GigaVoxels-style), so VRAM tracks the visible working set, not the total data size. `0` (default) keeps the volume at `--grid`.

Like the 2D viewer, the 3D bundle loads Three.js from a CDN and fetches its data over HTTP — open `index.html` through a web server, not a `file://` URL.

### Not yet implemented

The 3D mode is scoped for incremental delivery. Shipped so far: **ray-guided
bricked sparse-voxel streaming** for the volume (GigaVoxels-style page table +
brick pool with empty-space skipping and a bounded on-demand GPU cache), so the
ray-marched volume can drill past a single bounded grid. Still deferred:

- **3D file-boundary overlays** and an **interactive transfer-function editor** with a density histogram.

## Supported input formats

- **Plain binary** — anything not specifically detected is rendered byte-for-byte.
- **JSON / JSONL** — structure-aware in diff mode (see below).

Anything else — `.safetensors`, `.gguf`, PyTorch `.bin` — is rendered as plain bytes here. For tensor-format awareness use [modelweightvis](https://github.com/znation/modelweightvis).

## Comparing two files: `--diff`

```sh
arbvis --diff a.bin b.bin --out ./out
arbvis --diff hf://owner/repo/a.json hf://owner/repo/b.json --out hf://datasets/me/vis/diff
```

Plain-byte diff aligns the two inputs at offset 0 and computes per-byte deltas. Whole directories work too — each file pairs up by name across the two roots.

### JSON / JSONL structure-aware diff

When both `--diff` inputs have a `.json` or `.jsonl` extension, arbvis aligns them by structure (object keys, array elements, value boundaries) before computing byte deltas, so a single-key insertion near the top of a file doesn't smear every following byte across the canvas.

## Output destinations

arbvis writes a self-contained web-viewer **bundle**. There are two ways to get one:

- `--out DIR` — write the bundle to a local directory (or an `hf://` URL).
- `--space NAMESPACE/REPO` — render the bundle and deploy a live Hugging Face Space.

`--out` and `--space` work the same in both 2D and `--3d`.

### Local bundle (`--out DIR`)

```sh
arbvis file1.bin file2.bin --out ./out
# serve it locally, e.g.:  python3 -m http.server -d ./out
```

In 2D this generates a Leaflet pyramid (`out/tiles/{z}/{x}/{y}.{ext}`, `out/index.html`, `out/labels.json`):

- Full resolution at every zoom level (1 px = 1 byte at max zoom).
- Vector file boundaries — sharp at every scale, not baked into pixels.
- No size limit — works on files of any size; lower zoom levels are averaged.
- HTML labels positioned at each region's area-weighted centroid.

In `--3d` it generates the volume bundle (`index.html`, `volume.bin`, `bricks.bin`, `pagetable.bin`, `meta.json`) — see [3D mode](#3d-mode). Either bundle loads its rendering library from a CDN and fetches its data over HTTP, so open `index.html` through a web server, not a `file://` URL.

![arbvis screenshot](arbvis.png)

*Multiple unrelated files (images, parquet, mp3, an SSH key) concatenated and rendered together — each file's content signature is immediately distinguishable.*

### HF Hub output

`--out` accepts an `hf://` URL and uploads the bundle directly to the Hub:

```sh
arbvis dir/ --out hf://datasets/me/vis/dir
```

Note: `--out hf://…` uploads the bundle files to the target repo, but the Hub won't render `index.html` on its own. Use `--space` for a working URL.

### Deploy a viewable Space (`--space`)

```sh
arbvis hf://datasets/owner/dataset --space me/dataset-vis
arbvis hf://datasets/owner/dataset --3d --space me/dataset-vis-3d
```

Renders the bundle and deploys a Docker Space that serves the viewer. The bundle data lives in an auto-created sibling bucket repo (`me/dataset-vis_bucket`); the Space itself is stateless and just proxies it.

### Tile format (`--tile-format`, 2D only)

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
arbvis hf://datasets/owner/dataset --show-xet-xorbs --out ./out
```

For xet-backed Hub files, colors each region by the xorb (content-addressed chunk) it was reconstructed from: hue encodes xorb ID, intensity encodes the underlying byte. Useful for seeing how a file is partitioned across the CAS.

modelweightvis layers a dtype-aware element coloring on top of the same xorb hue for `.safetensors` / `.gguf` inputs; arbvis covers the generic byte path.

## Other useful flags

- `--title TEXT` — title shown in the viewer info panel (defaults to `"arbvis"` or `"arbvis diff"`).
- `-l, --file-list FILE` — read input paths from `FILE`, one per line; `-` reads from stdin.
- `--regen-html DIR` — rebuild `index.html` for an existing bundle directory without re-rendering (2D or, with `--3d`, the volume bundle). Useful after editing the viewer template.
- `--space OWNER/REPO --out LOCAL_DIR` (with no input files) — re-deploy an already-rendered bundle to a Space without re-rendering. Add `--3d` to re-deploy a volume bundle.

```sh
arbvis --regen-html ./out
arbvis --space me/vis --out ./out
```

## Relationship to modelweightvis

arbvis is the byte-only foundation: Hilbert layout, byte coloring, JSON-aware diff, Hub I/O, tile pyramid, Space deploy, xet xorb path, streaming. It has no knowledge of tensors, model formats, or transformer architecture — `.safetensors` and `.gguf` get the same byte-Hilbert treatment as any other binary.

[modelweightvis](https://github.com/znation/modelweightvis) is a separate crate that extends arbvis through its generic plugin surface (no fork, no patch) — it's one *specialization* of arbvis, and a new one (for any structured binary format) plugs in the same way: `FormatPlugin` impls parse `.safetensors` / `.gguf` / pickle headers and stuff `ModelInfo` into each source's extension map; `LayoutPlugin` impls add the architectural transformer layout and the MoE summary / CKA panel layouts; `SourceProvider` impls turn an invocation (`--moe`, a repo-level or directory `--diff`) into render sources; a layout-keyed `LeafLoader`/`LeafRenderer` pair draws the arch layout; `DiffSourceBuilder` adds tensor-aware file-pair diffing; `PrepareSourcesExtension` fetches sidecar config. The `modelweightvis` binary builds an `arbvis::Registry::with_defaults()`, calls `modelweightvis::register_all(&mut registry, &args)`, and hands off to `arbvis::run`. Same renderer, same Hub I/O, same tile pyramid — just with the tensor-aware plugins registered.

**Which to use:**
- **arbvis** — for non-model binaries (any file format), JSON/JSONL diffs, plain-byte diffs, the xet xorb path on arbitrary content. Smaller dependency footprint (no `candle-core` / `regex` / `zip` / `half`).
- **modelweightvis** — for `.safetensors` / `.gguf` / `.bin` model checkpoints, architectural transformer layout, `--moe-summary` / `--moe-cka` / `--probe`, `--diff-metric`, `--finetune` / `--no-finetune`, `--layout`. Inherits arbvis's full CLI surface (`--out`, `--3d`, `--space`, `--stream`, `--show-xet-xorbs`, `--regen-html`, etc.) — no need to use both binaries.

## Building

Requires Rust (stable) and the official Hugging Face [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) on `$PATH` (install via `pip install -U huggingface_hub`, `brew install huggingface-cli`, or `curl -LsSf https://hf.co/cli/install.sh | bash`). arbvis shells out to `hf` for every Hub download / upload / sync.

```sh
cargo build --release
./target/release/arbvis <file> --out ./output
```

Or install into your `PATH`:

```sh
cargo install --path .
```

For modelweightvis, see the [standalone modelweightvis repo](https://github.com/znation/modelweightvis) — it depends on arbvis via a pinned git revision and inherits arbvis's full CLI surface.

## Credits

Color scheme inspired by [Stairwell's binary visualization post](https://stairwell.com/blog/hilbert-curves-visualizing-binary-files-with-color-and-patterns/). Built on [clap](https://crates.io/crates/clap) (CLI), [image](https://crates.io/crates/image) + [png](https://crates.io/crates/png) + rav1e (tile encoding), [fast_hilbert](https://crates.io/crates/fast_hilbert) (2D curve mapping; the 3D curve is a hand-rolled Skilling transform), the official Hugging Face [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) (Hub I/O) + [xet-core-structures](https://crates.io/crates/xet-core-structures) (per-tile xet decode), [Leaflet.js](https://leafletjs.com/) (the 2D viewer), and [Three.js](https://threejs.org/) (the 3D viewer).
