# arbvis

![arbvis screenshot](arbvis.png)

Visualize binary files as [Hilbert curve](https://en.wikipedia.org/wiki/Hilbert_curve) plots. Each byte is mapped to a color and placed along a Hilbert curve, making structural patterns in the file visually apparent — null regions, ASCII text, compressed/encrypted data, and boundaries between file sections all produce recognizable visual signatures.

## Quick start

```sh
arbvis huge_file.bin --tiles ./output
# then open output/index.html in a browser
```

## Tiled viewer (recommended)

The primary way to use arbvis is the `--tiles` mode, which generates a [Leaflet.js](https://leafletjs.com/) tile pyramid you can open in any browser:

```sh
arbvis file1.bin file2.bin --tiles ./output
```

This produces:
- `output/tiles/{z}/{x}/{y}.png` — XYZ tiles at multiple zoom levels
- `output/index.html` — ready-to-use Leaflet viewer

The tiled viewer offers significant advantages over the single-image mode:

- **Full resolution at every scale** — the highest zoom level preserves one pixel per byte with no downsampling; lower zoom levels are averaged so you can zoom out smoothly across arbitrarily large files
- **Vector file boundaries** — borders between files are drawn as crisp vector polylines that remain sharp at any zoom level, rather than being baked into raster pixels
- **Accurate file labels** — labels are positioned at the area-weighted centroid of each file's actual data region and rendered as HTML, so they're always legible regardless of zoom
- **No size limit** — works on files of any size; the tiled pyramid approach avoids the memory constraints of single-image rendering

## Color scheme

Bytes are colored by range (based on [Stairwell's approach](https://stairwell.com/blog/hilbert-curves-visualizing-binary-files-with-color-and-patterns/)):

| Value | Color |
|-------|-------|
| `0x00` | Black |
| `0x01`–`0x1F` | Green (control characters) |
| `0x20`–`0x7E` | Blue (printable ASCII) |
| `0x7F`–`0xFE` | Red (high bytes) |
| `0xFF` | White |

## Single-image mode

For quick one-off inspection, arbvis can display a window or write a PNG directly:

```sh
# Display in a window
arbvis /bin/ls

# Write to a file
arbvis /bin/ls --output out.png

# Multiple files (boundaries marked in black)
arbvis file1.bin file2.bin --output out.png

# Pipe from stdin
cat /dev/urandom | head -c 65536 | arbvis
```

If no files are given, reads from stdin. For large files the image is subsampled to fit a 4096×4096 canvas, so detail is lost — use `--tiles` instead.

## Building

Requires Rust (stable).

```sh
cargo build --release
./target/release/arbvis <file> --tiles ./output
```

## Dependencies

- [`clap`](https://crates.io/crates/clap) — CLI argument parsing
- [`fast_hilbert`](https://crates.io/crates/fast_hilbert) — Hilbert curve index-to-coordinate mapping
- [`image`](https://crates.io/crates/image) — Image construction and tile encoding
- [`show-image`](https://crates.io/crates/show-image) — Windowed display (single-image mode)
