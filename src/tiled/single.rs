//! Single-image PNG export (`--png FILE`).
//!
//! Renders the entire 2D Hilbert canvas as one indexed PNG — one pixel per
//! byte, same byte→color scheme as the tile pyramid — instead of a Leaflet
//! tile bundle. Geometry mirrors `build_tile_plan` (`src/tiled/mod.rs`) and
//! the pixel placement mirrors `render_leaf_tile_from_buf`
//! (`src/tiled/leaf.rs`): Hilbert square `sq` sits at x-offset `sq * height`,
//! so the PNG matches the viewer's layout exactly.
//!
//! The PNG is 8-bit indexed with a 256-entry palette built from
//! `color::build_pixel_lut()` in LUT order, so a pixel's palette index is the
//! byte value it renders. Bytes beyond the input total stay at index 0
//! (black), the same background the pyramid uses.
//!
//! Input bytes are read through the existing `Data` handles (`load_source_data`:
//! mmap for local files, range fetches for HTTP), one bounded chunk at a time,
//! so a multi-GB file never gets slurped into RAM and mmap pages stay warm.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::color::build_pixel_lut;
use crate::data::{load_source_data, Data, Source};
use crate::geometry::hilbert_to_xy_u64;
use crate::tiled::leaf::TILE_LOG2;

/// Bytes fetched from a source per `fetch_range` call. Large enough to keep
/// mmap/HTTP overhead negligible, small enough to bound peak RAM.
const CHUNK_BYTES: u64 = 1 << 20;

/// Canvas geometry for the single-image render — the same derivation
/// `build_tile_plan` performs (see `src/tiled/mod.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SingleGeom {
    /// Vertical Hilbert order: the image is `2^kh` pixels tall.
    pub kh: u8,
    /// Horizontal Hilbert order: the image is `2^kw` pixels wide.
    pub kw: u8,
    /// Image width in pixels (`2^kw`).
    pub width: u32,
    /// Image height in pixels (`2^kh`).
    pub height: u32,
    /// Bytes per Hilbert square (`height²`).
    pub square_pixels: u64,
}

/// Derive the single-image canvas geometry from the byte `total`.
///
/// `s` starts at `2 * TILE_LOG2` (the smallest canvas a tile pyramid would
/// build) and grows until `2^s >= total`; the image is `(1<<kw) × (1<<kh)`.
pub fn single_geometry(total: u64) -> SingleGeom {
    let mut s = 2 * TILE_LOG2 as u32;
    while (1u64 << s) < total {
        s += 1;
    }
    let kh = s / 2;
    let kw = s.div_ceil(2);
    let height = 1u32 << kh;
    let width = 1u32 << kw;
    SingleGeom {
        kh: kh as u8,
        kw: kw as u8,
        width,
        height,
        square_pixels: (height as u64) * (height as u64),
    }
}

/// Encode `pixels` (one palette index per pixel, row-major) as an 8-bit
/// indexed PNG whose palette is the full 256-entry `build_pixel_lut()` in LUT
/// order — so index *i* renders the same color the tile pyramid gives byte
/// value *i*.
fn encode_indexed_single_png(
    width: u32,
    height: u32,
    pixels: &[u8],
) -> anyhow::Result<Vec<u8>> {
    let lut = build_pixel_lut();
    let palette: Vec<u8> = lut.iter().flat_map(|c| c.0).collect();
    debug_assert_eq!(palette.len(), 256 * 3);

    let mut out: Vec<u8> = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Indexed);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_palette(&palette);
        // Same fast-compression tradeoff as the leaf tiles: the indexed
        // stream is Hilbert-local and compresses to within ~0.4% of zlib
        // level 6 while encoding far faster.
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder.write_header().map_err(|e: png::EncodingError| anyhow::anyhow!(e.to_string()))?;
        writer
            .write_image_data(pixels)
            .map_err(|e: png::EncodingError| anyhow::anyhow!(e.to_string()))?;
    }
    Ok(out)
}

/// Map Hilbert byte index `i` to its image pixel (row-major offset).
///
/// Square `sq` sits at x-offset `sq * height`, matching the pyramid's
/// per-square layout (`sq_off` / `xy2h_u64` in `src/tiled/leaf.rs`).
#[inline]
fn pixel_offset(i: u64, geom: &SingleGeom) -> usize {
    let sq = i / geom.square_pixels;
    let local = i % geom.square_pixels;
    let (lx, ly) = hilbert_to_xy_u64(local, geom.kh);
    let x = sq * geom.height as u64 + lx as u64;
    let y = ly as u64;
    (y * geom.width as u64 + x) as usize
}

/// Render all `sources` (concatenated, `total` bytes) into one indexed PNG
/// written to `out`. Reads through the mmap / range-fetch `Data` path in
/// `CHUNK_BYTES` chunks; writes to a `.tmp` sibling and renames on success.
pub async fn render_single_png(
    sources: &[Source],
    total: u64,
    out: &Path,
) -> anyhow::Result<()> {
    let geom = single_geometry(total);
    let mut pixels = vec![0u8; geom.width as usize * geom.height as usize];

    // Open every source up front (mmap for local, lightweight handle for
    // HTTP) — mirrors `build_tile_plan`'s `load_source_data` loop.
    let source_data: Vec<Data> = sources
        .iter()
        .map(load_source_data)
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Walk byte indices sequentially across source boundaries so mmap pages
    // stay warm and each source sees one ascending range-fetch sequence.
    let mut cumulative: Vec<u64> = Vec::with_capacity(sources.len());
    let mut off = 0u64;
    for s in sources {
        cumulative.push(off);
        off += s.byte_size;
    }

    let mut pos = 0u64;
    while pos < total {
        let src_idx = cumulative.partition_point(|&c| c <= pos) - 1;
        let src_end = if src_idx + 1 < cumulative.len() {
            cumulative[src_idx + 1]
        } else {
            total
        };
        let chunk_len = (src_end - pos).min(CHUNK_BYTES) as usize;
        let buf = source_data[src_idx]
            .fetch_range(pos - cumulative[src_idx], chunk_len)
            .await
            .with_context(|| format!("reading byte range at {} of source {src_idx}", pos))?;
        for (k, &b) in buf.iter().enumerate() {
            pixels[pixel_offset(pos + k as u64, &geom)] = b;
        }
        pos += chunk_len as u64;
    }

    let png = encode_indexed_single_png(geom.width, geom.height, &pixels)?;

    // Write to a temp sibling, then rename, so a crash never leaves a
    // half-written PNG at the requested path.
    let tmp = out.with_extension("png.tmp");
    fs::write(&tmp, &png)
        .with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, out).with_context(|| format!("renaming into {}", out.display()))?;
    log::info!(
        "wrote {} ({}x{}, {} bytes of input)",
        out.display(),
        geom.width,
        geom.height,
        total
    );
    Ok(())
}

/// Resolve the output path for `--png FILE`.
///
/// With `--out DIR` (local directory), `FILE` is placed inside `DIR`;
/// otherwise `FILE` is used as given. `hf://` destinations are rejected:
/// a single PNG is a local file, not a bundle to upload.
pub fn png_output_path(png: &Path, out: Option<&Path>) -> anyhow::Result<PathBuf> {
    let Some(out) = out else {
        return Ok(png.to_path_buf());
    };
    if out.to_string_lossy().starts_with("hf://") {
        anyhow::bail!(
            "--png writes a single local file; --out must be a local directory, got {out:?}"
        );
    }
    Ok(out.join(png.file_name().ok_or_else(|| {
        anyhow::anyhow!("--png path has no file name: {}", png.display())
    })?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Source;
    use std::io::Write;

    /// Write a temp file: `fill` bytes of 0x00 followed by a 0x41 run.
    fn temp_input(zeros: usize, a_run: usize) -> anyhow::Result<(PathBuf, PathBuf)> {
        let mut buf = vec![0u8; zeros];
        buf.extend(std::iter::repeat_n(0x41u8, a_run));
        let mut f = tempfile::NamedTempFile::new()?;
        f.write_all(&buf)?;
        let path = f.into_temp_path().keep()?;
        Ok((path.clone(), path))
    }

    fn file_source(path: &Path, len: u64) -> Source {
        use crate::data::SourceKind;
        Source {
            file_idx: 0,
            kind: SourceKind::File(path.to_path_buf()),
            byte_size: len,
            name_override: None,
            xet_terms: None,
            extensions: Default::default(),
        }
    }

    #[test]
    fn geometry_matches_tile_plan_derivation() {
        // 300 bytes: s stays at 2*TILE_LOG2 = 18, so kh=kw=9: one 512×512 square.
        let g = single_geometry(300);
        assert_eq!(
            g,
            SingleGeom {
                kh: 9,
                kw: 9,
                width: 512,
                height: 512,
                square_pixels: 262_144,
            }
        );
        // 300_000 bytes: s grows 18 → 19, kh=9, kw=10: two 512×512 squares wide.
        let g = single_geometry(300_000);
        assert_eq!(
            g,
            SingleGeom {
                kh: 9,
                kw: 10,
                width: 1024,
                height: 512,
                square_pixels: 262_144,
            }
        );
        // Small files still get the minimum pyramid-sized canvas (512×512).
        let g = single_geometry(1);
        assert_eq!((g.width, g.height, g.square_pixels), (512, 512, 262_144));
    }

    #[test]
    fn pixel_offset_matches_pyramid_square_layout() {
        let g = single_geometry(300_000);
        // Square 0 starts at x=0; square 1 at x=height. Index square_pixels
        // (first byte of square 1) must land in square 1's column band, row 0.
        let off = pixel_offset(g.square_pixels, &g);
        let x = off % g.width as usize;
        let y = off / g.width as usize;
        assert_eq!(x, g.height as usize, "square 1 starts at x = height");
        assert_eq!(y, 0, "Hilbert index 0 of a square maps to y = 0");
        // Within square 0, index i and pixel_offset(i) agree with a direct
        // hilbert_to_xy lookup.
        for i in [0u64, 1, 2, 3, 42, 255] {
            let (lx, ly) = hilbert_to_xy_u64(i, g.kh);
            assert_eq!(pixel_offset(i, &g), ly as usize * g.width as usize + lx as usize);
        }
    }

    #[tokio::test]
    async fn renders_known_buffer_as_indexed_png() {
        let (_keep, path) = temp_input(300, 100).unwrap();
        let total = 400;
        let sources = vec![file_source(&path, total)];

        let out_dir = tempfile::tempdir().unwrap();
        let out = out_dir.path().join("out.png");
        render_single_png(&sources, total, &out).await.unwrap();

        // Decode and check the frame.
        let f = std::fs::File::open(&out).unwrap();
        let dec = png::Decoder::new(std::io::BufReader::new(f));
        let mut reader = dec.read_info().unwrap();
        let geom = single_geometry(total);
        assert_eq!(
            (reader.info().width, reader.info().height),
            (geom.width, geom.height)
        );
        assert_eq!(reader.info().color_type, png::ColorType::Indexed);
        let mut bytes = vec![0u8; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut bytes).unwrap();

        let lut = build_pixel_lut();
        // Every byte index i in 0..total must land on its Hilbert position
        // with the exact LUT color for its byte value.
        for i in 0..total {
            let b = if i < 300 { 0x00 } else { 0x41 };
            let off = pixel_offset(i, &geom);
            assert_eq!(
                bytes[off], b,
                "pixel at byte index {i} holds the wrong palette index"
            );
        }
        // Spot-check the palette itself: index 0 is black, 0x41 is the
        // ASCII-blue band color.
        let png_bytes = std::fs::read(&out).unwrap();
        let raw_pal = extract_plte(&png_bytes).unwrap();
        assert_eq!(&raw_pal[0..3], &[0, 0, 0]);
        assert_eq!(raw_pal[0x41 * 3..0x41 * 3 + 3], lut[0x41].0[..]);
        assert_eq!(raw_pal.len() % 3, 0);
    }

    /// Pull the PLTE chunk out of a PNG file (test-only helper).
    fn extract_plte(png: &[u8]) -> Option<&[u8]> {
        let mut pos = 8;
        while pos + 8 <= png.len() {
            let len = u32::from_be_bytes(png[pos..pos + 4].try_into().ok()?) as usize;
            if &png[pos + 4..pos + 8] == b"PLTE" {
                return png.get(pos + 8..pos + 8 + len);
            }
            pos += 12 + len;
        }
        None
    }

    #[test]
    fn output_path_rejects_hf_out() {
        assert!(png_output_path(Path::new("a.png"), Some(Path::new("hf://me/repo"))).is_err());
        assert_eq!(
            png_output_path(Path::new("a.png"), None).unwrap(),
            PathBuf::from("a.png")
        );
        assert_eq!(
            png_output_path(Path::new("a.png"), Some(Path::new("/tmp/o"))).unwrap(),
            PathBuf::from("/tmp/o/a.png")
        );
    }
}
