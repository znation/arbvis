# arbvis

Visualize binary files as [Hilbert curve](https://en.wikipedia.org/wiki/Hilbert_curve) plots. Each byte is mapped to a color and placed along a Hilbert curve, making structural patterns in the file visually apparent — null regions, ASCII text, compressed/encrypted data, and boundaries between file sections all produce recognizable visual signatures.

## Color scheme

Bytes are colored by range (based on [Stairwell's approach](https://stairwell.com/resources/hilbert-curves-visualizing-binary-files-with-color-and-patterns/)):

| Value | Color |
|-------|-------|
| `0x00` | Black |
| `0x01`–`0x1F` | Green (control characters) |
| `0x20`–`0x7E` | Blue (printable ASCII) |
| `0x7F`–`0xFE` | Red (high bytes) |
| `0xFF` | White |

## Usage

```
arbvis [FILES]...
```

If no files are given, reads from stdin.

```sh
# Visualize a single file
arbvis /bin/ls

# Visualize multiple files (boundaries marked in black)
arbvis file1.bin file2.bin

# Pipe from stdin
cat /dev/urandom | head -c 65536 | arbvis
```

When multiple files are provided, pixels on the border between file regions are drawn black, making the boundary between files visible.

## Building

Requires Rust (stable).

```sh
cargo build --release
./target/release/arbvis <file>
```

## Dependencies

- [`clap`](https://crates.io/crates/clap) — CLI argument parsing
- [`fast_hilbert`](https://crates.io/crates/fast_hilbert) — Hilbert curve index-to-coordinate mapping
- [`image`](https://crates.io/crates/image) — Image construction
- [`show-image`](https://crates.io/crates/show-image) — Windowed image display
