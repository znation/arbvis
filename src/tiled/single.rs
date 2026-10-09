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
//! an in-memory snapshot for local files, range fetches for HTTP) and processed
//! in `CHUNK_BYTES` chunks, so remote reads stay incremental. For local files,
//! though, the snapshot is the whole file (`std::fs::read`), so peak RAM scales
//! with the input size; only remote sources read incrementally.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::color::build_pixel_lut;
use crate::data::{load_source_data, Data, Source};
use crate::geometry::hilbert_to_xy_u64;
use crate::layout::hilbert::hilbert_canvas;
use crate::tiled::leaf::{local_curve_to_xy, tile_curve_frame, TILE, TILE_LOG2};

/// Bytes fetched from a source per `fetch_range` call. Large enough to keep
/// per-chunk fetch overhead negligible, small enough to bound peak RAM.
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

/// Derive the single-image canvas geometry from the byte `total`, via
/// [`hilbert_canvas`]: `s` starts at
/// `2 * TILE_LOG2` (the smallest canvas a tile pyramid would build) and grows
/// until `2^s >= total`; the image is `(1<<kw) × (1<<kh)`.
pub fn single_geometry(total: u64) -> SingleGeom {
    let g = hilbert_canvas(total);
    SingleGeom {
        kh: g.kh,
        kw: g.kw,
        width: g.width,
        height: g.height,
        square_pixels: g.square_pixels,
    }
}

/// Encode `pixels` (one palette index per pixel, row-major) as an 8-bit
/// indexed PNG whose palette is the full 256-entry `build_pixel_lut()` in LUT
/// order — so index *i* renders the same color the tile pyramid gives byte
/// value *i*.
fn encode_indexed_single_png(width: u32, height: u32, pixels: &[u8]) -> anyhow::Result<Vec<u8>> {
    let lut = build_pixel_lut();
    let palette: Vec<u8> = lut.iter().flat_map(|c| c.0).collect();
    debug_assert_eq!(palette.len(), 256 * 3);

    crate::tiled::leaf::encode_png(width, height, pixels, Some(&palette))
        .map_err(anyhow::Error::msg)
}

/// Encode an RGB image as a truecolor 8-bit PNG. Diff output mixes
/// signed-delta LUT colors, one-sided-source tints, and crosshatch fills, so
/// its palette is not bounded at 256 entries — indexed PNG won't do.
fn encode_rgb_png(img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> anyhow::Result<Vec<u8>> {
    let (width, height) = img.dimensions();
    crate::tiled::leaf::encode_png(width, height, img.as_raw(), None).map_err(anyhow::Error::msg)
}

/// Scatter state for one leaf-size tile (TILE×TILE pixels) of the canvas.
///
/// Bytes arrive in Hilbert order, and each leaf tile covers one contiguous
/// run of Hilbert indices, so the render loop advances this state once per
/// 256 KiB instead of paying a full O(kh) Hilbert decode plus div/mod per
/// byte. Within the run, the pixel for curve position `v` comes from the
/// tile's precomputed Hilbert frame (see `tile_curve_frame` in
/// `src/tiled/leaf.rs`) applied to `local_curve_to_xy()[v]`.
struct TileScatter {
    /// Hilbert index of the tile's first byte.
    base: u64,
    /// Hilbert index one past the tile's last byte.
    end: u64,
    /// Raster offset of the tile's top-left pixel.
    origin: usize,
    /// The tile's Hilbert frame: `(swap, cx, cy)`. A pixel `(px, py)` sits at
    /// curve offset `LOCAL_CURVE_LUT[(b << TILE_LOG2) | a]` where
    /// `(a, b) = (py ^ cy, px ^ cx)` when swapped, else `(px ^ cx, py ^ cy)`.
    frame: (bool, u32, u32),
}

/// Scatter state for the tile containing Hilbert index `i`.
fn tile_scatter(i: u64, geom: &SingleGeom) -> TileScatter {
    let tile_area = TILE as u64 * TILE as u64;
    let sq = i / geom.square_pixels;
    let local = i % geom.square_pixels;
    let tile_order = geom.kh - TILE_LOG2;
    // Tiles fill the square in curve order: coarse tile index `c` starts at
    // Hilbert offset `c * tile_area` within the square.
    let c = local / tile_area;
    let (ltx, lty) = hilbert_to_xy_u64(c, tile_order);
    let sq_base = sq * geom.square_pixels;
    TileScatter {
        base: sq_base + c * tile_area,
        end: sq_base + (c + 1) * tile_area,
        origin: ((sq * geom.height as u64 + ltx as u64 * TILE as u64)
            + lty as u64 * TILE as u64 * geom.width as u64) as usize,
        frame: tile_curve_frame(ltx, lty, geom.kh),
    }
}

/// Raster offset of the byte at Hilbert index `i`, advancing `tile` when `i`
/// crosses into the next tile. Equivalent to `pixel_offset(i, geom)`.
#[inline]
fn scatter_offset(tile: &mut TileScatter, i: u64, geom: &SingleGeom) -> usize {
    if i >= tile.end {
        *tile = tile_scatter(i, geom);
    }
    let (swap, cx, cy) = tile.frame;
    let packed = local_curve_to_xy()[(i - tile.base) as usize];
    // Unpack the identity-frame pixel (px_id, py_id) for this curve position,
    // then invert the frame transform to reach the tile's real pixel.
    let (px_id, py_id) = (packed & (TILE - 1), packed >> TILE_LOG2);
    let (px, py) = if swap {
        (py_id ^ cx, px_id ^ cy)
    } else {
        (px_id ^ cx, py_id ^ cy)
    };
    tile.origin + py as usize * geom.width as usize + px as usize
}

/// Render all `sources` (concatenated, `total` bytes) into one indexed PNG
/// written to `out`. Reads through the snapshot / range-fetch `Data` path in
/// `CHUNK_BYTES` chunks; writes to a `.tmp` sibling and renames on success.
/// Open every source and build the cumulative-offset table used to locate a
/// byte position's owning source.
///
/// Returns the opened `Data` handles (in-memory snapshot for local, a
/// lightweight handle for HTTP — mirrors `build_tile_plan`'s `load_source_data`
/// loop) paired with
/// `cumulative[i]`, the concatenated offset at which source `i` begins.
fn open_sources(sources: &[Source]) -> anyhow::Result<(Vec<Data>, Vec<u64>)> {
    let source_data: Vec<Data> = sources
        .iter()
        .map(load_source_data)
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut cumulative = Vec::with_capacity(sources.len());
    let mut off = 0u64;
    for s in sources {
        cumulative.push(off);
        off += s.byte_size;
    }
    Ok((source_data, cumulative))
}

/// Source index, source-local offset, and length of the byte range this
/// pipeline reads for the file position `pos`, capped at [`CHUNK_BYTES`] and
/// at the owning source's end. Pure helper so tests can pin boundary
/// behavior (source switches, final partial chunk) without I/O.
fn chunk_span(cumulative: &[u64], total: u64, pos: u64) -> ChunkSpan {
    let src = cumulative.partition_point(|&c| c <= pos) - 1;
    let src_end = if src + 1 < cumulative.len() {
        cumulative[src + 1]
    } else {
        total
    };
    ChunkSpan {
        src,
        offset: pos - cumulative[src],
        len: (src_end - pos).min(CHUNK_BYTES) as usize,
    }
}

/// See [`chunk_span`].
struct ChunkSpan {
    src: usize,
    offset: u64,
    len: usize,
}

pub async fn render_single_png(sources: &[Source], total: u64, out: &Path) -> anyhow::Result<()> {
    let geom = single_geometry(total);
    let mut pixels = vec![0u8; geom.width as usize * geom.height as usize];

    let (source_data, cumulative) = open_sources(sources)?;

    // Walk byte positions sequentially across source boundaries, pipelined one
    // chunk ahead: while the Hilbert scatter of the current chunk runs (pure
    // CPU), the next chunk's fetch is already in flight. On remote sources this
    // overlaps the HTTP round-trip with the scatter instead of serializing
    // them; on local sources the fetch is an in-memory byte-snapshot copy, so a
    // one-chunk read-ahead is effectively free. Fetch order stays ascending,
    // so each source still sees one ascending range-fetch sequence.

    let mut pos = 0u64;
    let first = chunk_span(&cumulative, total, pos);
    let mut buf = source_data[first.src]
        .fetch_range(first.offset, first.len)
        .await
        .with_context(|| format!("reading byte range at {} of source {}", pos, first.src))?;
    while pos < total {
        let next_pos = pos + buf.len() as u64;
        // Kick off the next chunk's fetch before scattering this one, so the
        // round-trip overlaps the CPU work.
        let mut next = None;
        let mut next_fut: Option<
            std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<Vec<u8>>> + '_>>,
        > = None;
        if next_pos < total {
            let c = chunk_span(&cumulative, total, next_pos);
            let data = &source_data[c.src];
            next_fut = Some(Box::pin(async move {
                data.fetch_range(c.offset, c.len).await.with_context(|| {
                    format!("reading byte range at {} of source {}", next_pos, c.src)
                })
            }));
            next = Some(c);
        }
        let mut tile = tile_scatter(pos, &geom);
        for (k, &b) in buf.iter().enumerate() {
            pixels[scatter_offset(&mut tile, pos + k as u64, &geom)] = b;
        }
        if let Some(f) = next_fut {
            buf = f.await?;
        }
        debug_assert!(next.is_none_or(|c| buf.len() == c.len));
        pos = next_pos;
    }

    let png = encode_indexed_single_png(geom.width, geom.height, &pixels)?;

    // Write to a temp sibling, then rename, so a crash never leaves a
    // half-written PNG at the requested path.
    let tmp = out.with_extension("png.tmp");
    fs::write(&tmp, &png).with_context(|| format!("writing {}", tmp.display()))?;
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

/// Render all `sources` (concatenated, `total` bytes) in diff mode into one
/// truecolor RGB PNG written to `out`.
///
/// The tile grid is identical to plain mode (`single_geometry`), but each
/// TILE×TILE tile is loaded with the pyramid's `load_tile_bytes` (Hilbert
/// order, per-source ranges) and rendered with the pyramid's
/// `render_leaf_tile_diff` — signed-delta LUT plus crosshatch fills and
/// one-sided-source tints from [`crate::tiled::diff_leaf_mode`] — then blitted
/// at raster `(tx*TILE, ty*TILE)`. Output is truecolor (see
/// [`encode_rgb_png`]); write goes to a `.tmp` sibling, then rename.
pub async fn render_single_diff_png(
    sources: &[Source],
    total: u64,
    out: &Path,
) -> anyhow::Result<()> {
    use crate::tiled::leaf::{load_tile_bytes, render_leaf_tile_diff, TileFormat};
    use crate::tiled::{diff_leaf_mode, LeafMode};

    let geom = single_geometry(total);
    let LeafMode::Diff {
        pixel_lut,
        plain_lut,
        fills,
        tints,
    } = diff_leaf_mode(sources)
    else {
        unreachable!("diff_leaf_mode always returns LeafMode::Diff")
    };

    let (source_data, cumulative) = open_sources(sources)?;

    let height_tiles = geom.height / TILE;
    let width_tiles = geom.width / TILE;
    let mut img = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::new(geom.width, geom.height);
    for ty in 0..height_tiles {
        for tx in 0..width_tiles {
            let tile_buf = load_tile_bytes(
                tx,
                ty,
                geom.kh,
                height_tiles,
                geom.square_pixels,
                total,
                &source_data,
                &cumulative,
            )
            .await?;
            let (tile_img, _) = render_leaf_tile_diff(
                tx,
                ty,
                geom.kh,
                height_tiles,
                geom.square_pixels,
                total,
                &tile_buf,
                &pixel_lut,
                &plain_lut,
                &fills,
                &tints,
                TileFormat::Png,
            )
            .map_err(|e| anyhow::anyhow!(e))?;
            blit_tile(&mut img, &tile_img, tx, ty, geom.width);
        }
    }

    let png = encode_rgb_png(&img)?;
    let tmp = out.with_extension("png.tmp");
    fs::write(&tmp, &png).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, out).with_context(|| format!("renaming into {}", out.display()))?;
    log::info!(
        "wrote {} ({}x{}, diff mode, {} bytes of input)",
        out.display(),
        geom.width,
        geom.height,
        total
    );
    Ok(())
}

/// Copy a TILE×TILE tile image into the full canvas at raster
/// `(tx*TILE, ty*TILE)`.
fn blit_tile(
    img: &mut image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    tile_img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    tx: u32,
    ty: u32,
    canvas_width: u32,
) {
    let raw = tile_img.as_raw();
    let dst: &mut [u8] = img.as_mut();
    for py in 0..TILE as usize {
        let y = ty as usize * TILE as usize + py;
        let x = tx as usize * TILE as usize;
        let dst_start = (y * canvas_width as usize + x) * 3;
        let src_start = py * TILE as usize * 3;
        dst[dst_start..dst_start + TILE as usize * 3]
            .copy_from_slice(&raw[src_start..src_start + TILE as usize * 3]);
    }
}

/// Resolve the output path for `--png FILE`.
///
/// With `--out DIR` (local directory), `FILE` is placed inside `DIR`;
/// otherwise `FILE` is used as given. `hf://` destinations are rejected:
/// a single PNG is a local file, not a bundle to upload.
pub fn png_output_path(png: &Path, out: Option<&Path>) -> anyhow::Result<PathBuf> {
    // Fail before the render if the target exists and is a directory —
    // otherwise the PNG write fails late, at rename time, with a bare
    // "renaming into <path>" and no hint about what is wrong.
    let resolved = if let Some(out) = out {
        if out.to_string_lossy().starts_with("hf://") {
            anyhow::bail!(
                "--png writes a single local file; --out must be a local directory, got {out:?}"
            );
        }
        out.join(
            png.file_name()
                .ok_or_else(|| anyhow::anyhow!("--png path has no file name: {}", png.display()))?,
        )
    } else {
        png.to_path_buf()
    };
    if resolved.is_dir() {
        anyhow::bail!(
            "--png {} is a directory; pass the PNG file path to write (e.g. {})",
            resolved.display(),
            resolved.join("out.png").display()
        );
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Source;
    use std::io::Write;

    /// chunk_span reproduces the original inline chunk selection: chunks cap
    /// at CHUNK_BYTES and at the owning source's end, and the source switch
    /// resolves via the cumulative-offset partition point.
    #[test]
    fn chunk_span_matches_source_boundaries() {
        // Sources: 5, 3, 0-length-free rest (total 10).
        let cumulative = [0u64, 5, 8];
        let total = 12u64;
        assert_eq!(chunk_span(&cumulative, total, 0).len, 5); // capped by source end
        assert_eq!(chunk_span(&cumulative, total, 0).src, 0);
        assert_eq!(chunk_span(&cumulative, total, 4).offset, 4);
        let c = chunk_span(&cumulative, total, 5);
        assert_eq!((c.src, c.offset, c.len), (1, 0, 3));
        let c = chunk_span(&cumulative, total, 8);
        assert_eq!((c.src, c.offset, c.len), (2, 0, 4)); // final source runs to total
        let c = chunk_span(&cumulative, total, 12 - 1);
        assert_eq!((c.src, c.offset, c.len), (2, 3, 1));
    }

    /// A single large source chunks at CHUNK_BYTES and the last chunk is
    /// partial.
    #[test]
    fn chunk_span_caps_at_chunk_bytes() {
        let cumulative = [0u64];
        let total = CHUNK_BYTES + 7;
        assert_eq!(chunk_span(&cumulative, total, 0).len, CHUNK_BYTES as usize);
        let last = chunk_span(&cumulative, total, CHUNK_BYTES);
        assert_eq!((last.src, last.offset, last.len), (0, CHUNK_BYTES, 7));
    }

    /// Reference implementation (full O(kh) Hilbert decode plus div/mod per
    /// byte); the equivalence oracle for [`TileScatter`], which the render
    /// loop actually uses.
    #[inline]
    fn pixel_offset(i: u64, geom: &SingleGeom) -> usize {
        let sq = i / geom.square_pixels;
        let local = i % geom.square_pixels;
        let (lx, ly) = hilbert_to_xy_u64(local, geom.kh);
        let x = sq * geom.height as u64 + lx as u64;
        let y = ly as u64;
        (y * geom.width as u64 + x) as usize
    }

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
            assert_eq!(
                pixel_offset(i, &g),
                ly as usize * g.width as usize + lx as usize
            );
        }
    }

    #[test]
    #[ignore = "timing harness: cargo test --release --lib -- --ignored bench_scatter"]
    fn bench_scatter() {
        let g = single_geometry(1u64 << 44);
        let n = 1u64 << 24;
        let mut sink = 0usize;
        let t0 = std::time::Instant::now();
        let mut tile = tile_scatter(0, &g);
        for i in 0..n {
            sink = sink.wrapping_add(scatter_offset(&mut tile, i, &g));
        }
        let t1 = std::time::Instant::now();
        for i in 0..n {
            sink = sink.wrapping_add(pixel_offset(i, &g));
        }
        let t2 = std::time::Instant::now();
        eprintln!(
            "sink={sink} scatter={:?} ({:.1} ns/byte) pixel_offset={:?} ({:.1} ns/byte)",
            t1 - t0,
            (t1 - t0).as_nanos() as f64 / n as f64,
            t2 - t1,
            (t2 - t1).as_nanos() as f64 / n as f64,
        );
    }

    #[test]
    fn scatter_matches_pixel_offset() {
        // Canvas wider than tall (odd s → two squares per row) and taller
        // than one leaf tile in both axes, so every code path exercises:
        // tile transitions within a square, square transitions, and the
        // framed (non-identity) Hilbert frames most tiles land on.
        let g = single_geometry((1u64 << 18) * 3 + 7); // kh=9, kw=10, 4 tiles
        let mut tile = tile_scatter(0, &g);
        for i in 0..((1u64 << 18) * 3 + 7) {
            assert_eq!(
                scatter_offset(&mut tile, i, &g),
                pixel_offset(i, &g),
                "Hilbert index {i}"
            );
        }
        // A canvas with kh > TILE_LOG2: nested tiles inside one square.
        let g = single_geometry(1u64 << 21); // kh=10, kw=10, 16 tiles
        let mut tile = tile_scatter(0, &g);
        for i in 0..(1u64 << 21) {
            assert_eq!(
                scatter_offset(&mut tile, i, &g),
                pixel_offset(i, &g),
                "Hilbert index {i}"
            );
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
    fn output_path_rejects_directory_target() {
        let dir = tempfile::tempdir().unwrap();
        // --png <existing directory> (no --out): the path itself is the target.
        let err = png_output_path(Path::new(dir.path()), None).unwrap_err();
        assert!(err.to_string().contains("is a directory"), "{err}");
        // --png a.png --out <dir containing a directory named a.png>.
        let clash = dir.path().join("a.png");
        std::fs::create_dir(&clash).unwrap();
        let err = png_output_path(Path::new("a.png"), Some(dir.path())).unwrap_err();
        assert!(err.to_string().contains("is a directory"), "{err}");
        // A non-existing resolved path is still accepted.
        assert!(png_output_path(Path::new("a.png"), Some(&dir.path().join("o"))).is_ok());
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

    /// Fabricated diff-mode source list: a 300-byte whole-file diff pair
    /// (identical first 200 bytes, diverging tail), a 100-byte
    /// `UnmatchedRegion` (crosshatch fill), and a 50-byte `OneSidedRange`
    /// (tinted plain bytes). Total canvas: 450 bytes → 512×512, one tile.
    fn diff_sources() -> anyhow::Result<(Vec<Source>, u64)> {
        let mut orig = vec![0x00u8; 200];
        orig.extend(std::iter::repeat_n(0x07u8, 100));
        let mut mod_ = vec![0x00u8; 200];
        mod_.extend(std::iter::repeat_n(0xFFu8, 100));
        let (o_path, m_path) = temp_input_pair(&orig, &mod_)?;
        use crate::data::{DiffFill, SourceKind};
        let mk = |kind: SourceKind, byte_size: u64, name: &str| Source {
            file_idx: 0,
            kind,
            byte_size,
            name_override: Some(name.to_string()),
            xet_terms: None,
            extensions: Default::default(),
        };
        let sources = vec![
            mk(
                SourceKind::Diff {
                    original: o_path,
                    modified: m_path,
                },
                300,
                "pair",
            ),
            mk(
                SourceKind::UnmatchedRegion {
                    fill: DiffFill::Red,
                },
                100,
                "unmatched",
            ),
            mk(
                SourceKind::OneSidedRange {
                    data: std::sync::Arc::new(Data::Owned(vec![0x41; 50])),
                    start: 0,
                    fill: DiffFill::Green,
                },
                50,
                "inserted",
            ),
        ];
        let total: u64 = sources.iter().map(|s| s.byte_size).sum();
        Ok((sources, total))
    }

    fn temp_input_pair(orig: &[u8], mod_: &[u8]) -> anyhow::Result<(PathBuf, PathBuf)> {
        let o = tempfile::NamedTempFile::new()?;
        o.as_file().write_all(orig)?;
        let m = tempfile::NamedTempFile::new()?;
        m.as_file().write_all(mod_)?;
        Ok((o.into_temp_path().keep()?, m.into_temp_path().keep()?))
    }

    #[tokio::test]
    async fn diff_png_matches_reference_tiles() {
        let (sources, total) = diff_sources().unwrap();
        let out_dir = tempfile::tempdir().unwrap();
        let out = out_dir.path().join("diff.png");
        render_single_diff_png(&sources, total, &out).await.unwrap();

        // Decode with the `png` crate: must be truecolor RGB (diff output
        // mixes signed-delta LUT colors, tints, and crosshatch fills, so the
        // palette is not 256 entries).
        let file = std::io::BufReader::new(std::fs::File::open(&out).unwrap());
        let mut decoder = png::Decoder::new(file);
        decoder.set_transformations(png::Transformations::empty());
        let mut reader = decoder.read_info().unwrap();
        assert_eq!(reader.info().width, 512);
        assert_eq!(reader.info().height, 512);
        assert_eq!(reader.info().color_type, png::ColorType::Rgb);
        let mut buf = vec![0u8; reader.output_buffer_size().expect("buffer size known")];
        let frame = reader.next_frame(&mut buf).unwrap();
        assert_eq!(frame.width, 512);
        assert_eq!(frame.height, 512);

        // Reference: render the same sources tile-by-tile through the
        // pyramid's loader + diff renderer and blit each TILE×TILE result at
        // (tx*TILE, ty*TILE) with an independent put_pixel loop.
        use crate::tiled::leaf::{load_tile_bytes, render_leaf_tile_diff, TileFormat};
        use crate::tiled::{diff_leaf_mode, LeafMode};
        let LeafMode::Diff {
            pixel_lut,
            plain_lut,
            fills,
            tints,
        } = diff_leaf_mode(&sources)
        else {
            unreachable!()
        };
        let (source_data, cumulative) = open_sources(&sources).unwrap();
        let geom = single_geometry(total);
        let (ht, wt) = (geom.height / TILE, geom.width / TILE);
        let mut reference =
            image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::new(geom.width, geom.height);
        for ty in 0..ht {
            for tx in 0..wt {
                let tile_buf = load_tile_bytes(
                    tx,
                    ty,
                    geom.kh,
                    ht,
                    geom.square_pixels,
                    total,
                    &source_data,
                    &cumulative,
                )
                .await
                .unwrap();
                let (tile_img, _) = render_leaf_tile_diff(
                    tx,
                    ty,
                    geom.kh,
                    ht,
                    geom.square_pixels,
                    total,
                    &tile_buf,
                    &pixel_lut,
                    &plain_lut,
                    &fills,
                    &tints,
                    TileFormat::Png,
                )
                .map_err(|e| anyhow::anyhow!(e))
                .unwrap();
                for py in 0..TILE {
                    for px in 0..TILE {
                        reference.put_pixel(tx * TILE + px, ty * TILE + py, tile_img[(px, py)]);
                    }
                }
            }
        }

        assert_eq!(frame.color_type, png::ColorType::Rgb);
        let decoded: &[[u8; 3]] =
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const [u8; 3], 512 * 512) };
        for (i, px) in decoded.iter().enumerate() {
            let (x, y) = ((i % 512) as u32, (i / 512) as u32);
            let ref_px = reference[(x, y)].0;
            assert_eq!(px, &ref_px, "pixel ({x},{y})");
        }
    }

    #[tokio::test]
    async fn diff_png_matches_reference_tiles_multi_square() {
        // kw > kh case: TILE is 512, so total must exceed 2^18 = 262144 to
        // force s=19: kh=9 (height 512, one tile row) and kw=10 (width 1024,
        // two tile columns) — two squares side by side, the geometry the
        // single-square reference test never reaches.
        const PAIR_BYTES: usize = 262_200;
        let orig = vec![0x10u8; PAIR_BYTES];
        let mod_ = (0u32..PAIR_BYTES as u32)
            .map(|i| (i % 251) as u8)
            .collect::<Vec<u8>>();
        let (o_path, m_path) = temp_input_pair(&orig, &mod_).unwrap();
        use crate::data::{Source, SourceKind};
        let sources = vec![Source {
            file_idx: 0,
            kind: SourceKind::Diff {
                original: o_path,
                modified: m_path,
            },
            byte_size: PAIR_BYTES as u64,
            name_override: Some("pair".to_string()),
            xet_terms: None,
            extensions: Default::default(),
        }];
        let total = PAIR_BYTES as u64;
        let out_dir = tempfile::tempdir().unwrap();
        let out = out_dir.path().join("diff.png");
        render_single_diff_png(&sources, total, &out).await.unwrap();

        let file = std::io::BufReader::new(std::fs::File::open(&out).unwrap());
        let mut decoder = png::Decoder::new(file);
        decoder.set_transformations(png::Transformations::empty());
        let mut reader = decoder.read_info().unwrap();
        assert_eq!(reader.info().width, 1024);
        assert_eq!(reader.info().height, 512);
        let mut buf = vec![0u8; reader.output_buffer_size().expect("buffer size known")];
        let frame = reader.next_frame(&mut buf).unwrap();
        assert_eq!(frame.width, 1024);
        assert_eq!(frame.height, 512);

        use crate::tiled::leaf::{load_tile_bytes, render_leaf_tile_diff, TileFormat};
        use crate::tiled::{diff_leaf_mode, LeafMode};
        let LeafMode::Diff {
            pixel_lut,
            plain_lut,
            fills,
            tints,
        } = diff_leaf_mode(&sources)
        else {
            unreachable!()
        };
        let (source_data, cumulative) = open_sources(&sources).unwrap();
        let geom = single_geometry(total);
        assert!(
            geom.kw > geom.kh,
            "test must exercise the multi-square case"
        );
        let (ht, wt) = (geom.height / TILE, geom.width / TILE);
        let mut reference =
            image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::new(geom.width, geom.height);
        for ty in 0..ht {
            for tx in 0..wt {
                let tile_buf = load_tile_bytes(
                    tx,
                    ty,
                    geom.kh,
                    ht,
                    geom.square_pixels,
                    total,
                    &source_data,
                    &cumulative,
                )
                .await
                .unwrap();
                let (tile_img, _) = render_leaf_tile_diff(
                    tx,
                    ty,
                    geom.kh,
                    ht,
                    geom.square_pixels,
                    total,
                    &tile_buf,
                    &pixel_lut,
                    &plain_lut,
                    &fills,
                    &tints,
                    TileFormat::Png,
                )
                .map_err(|e| anyhow::anyhow!(e))
                .unwrap();
                for py in 0..TILE {
                    for px in 0..TILE {
                        reference.put_pixel(tx * TILE + px, ty * TILE + py, tile_img[(px, py)]);
                    }
                }
            }
        }
        for y in 0..geom.height {
            for x in 0..geom.width {
                let i = (y * geom.width + x) as usize;
                let px = &buf[i * 3..i * 3 + 3];
                assert_eq!(px, &reference[(x, y)].0, "pixel ({x},{y})");
            }
        }
    }

    #[tokio::test]
    async fn diff_leaf_mode_matches_build_tile_plan() {
        let (sources, total) = diff_sources().unwrap();
        // Source isn't Clone; build the same fabricated list twice so the
        // plan path and the helper get independent instances.
        let (helper_sources, _helper_total) = diff_sources().unwrap();
        let plan = crate::tiled::build_tile_plan(
            sources,
            total,
            true,
            false,
            crate::layout::LayoutMode::Hilbert,
            &crate::registry::Registry::with_defaults(),
        )
        .await
        .unwrap();
        use crate::tiled::{diff_leaf_mode, LeafMode};
        let LeafMode::Diff {
            pixel_lut,
            plain_lut,
            fills,
            tints,
        } = diff_leaf_mode(&helper_sources)
        else {
            unreachable!()
        };
        let LeafMode::Diff {
            pixel_lut: plan_lut,
            plain_lut: plan_plain,
            fills: plan_fills,
            tints: plan_tints,
        } = plan.mode
        else {
            panic!("expected diff mode plan")
        };
        assert_eq!(&*pixel_lut, &*plan_lut);
        assert_eq!(&*plain_lut, &*plan_plain);
        assert_eq!(&*fills, &*plan_fills);
        assert_eq!(&*tints, &*plan_tints);
        // And the fabricated lists are what the diff renderer expects:
        // sorted by start, covering the fabricated sources only.
        assert_eq!(
            fills.as_ref(),
            &[(300u64, 400u64, crate::data::DiffFill::Red)]
        );
        assert_eq!(
            tints.as_ref(),
            &[(400u64, 450u64, crate::data::DiffFill::Green)]
        );
    }
}
