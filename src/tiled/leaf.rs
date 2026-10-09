//! Leaf-tile rendering and encoding: the highest-resolution tile level, where
//! one pixel is one byte, in plain and diff variants, as AVIF or PNG.

use std::io::Cursor;

use image::codecs::avif::AvifEncoder;
use image::{ImageEncoder, ImageFormat, Rgb};
use rustc_hash::FxHashMap;

use crate::data::{Data, DiffFill};

/// Edge length in pixels of one leaf tile (so a tile covers `TILE²` bytes in
/// plain mode). Every tile pyramid geometry derives from this constant.
pub const TILE: u32 = 512;
/// `log2(TILE)` — the curve order of one tile's Hilbert frame.
pub const TILE_LOG2: u8 = TILE.trailing_zeros() as u8;
/// Bytes covered by one leaf tile (`TILE²`); the size of a plain-mode
/// `tile_buf`.
pub const TILE_PIXELS: usize = (TILE as usize) * (TILE as usize);
const TILE_AREA: u64 = TILE_PIXELS as u64;

type TileResult = Result<(image::ImageBuffer<Rgb<u8>, Vec<u8>>, Vec<u8>), String>;

/// On-disk format for an encoded tile.
///
/// Three options, chosen per-tile by `mod.rs`:
///
/// - `IndexedPng`: 8-bit indexed-color PNG with the tile's unique RGB values
///   as a ≤256-entry palette. Lossless and ~2-4× smaller than truecolor PNG
///   for Plain mode leaves (where every pixel comes from a fixed
///   256-color LUT). Falls back to truecolor PNG if a tile happens to need
///   more than 256 colors.
/// - `Avif`: AV1 still-image. `quality=100` is near-lossless; lower values
///   are lossy. AV1 isn't tuned for palette content, so we only use AVIF for
///   pyramid (downsampled, continuous-tone) tiles and Xet-mode leaf tiles
///   (which can exceed 256 colors).
/// - `Png`: 24-bit truecolor PNG. Universal fallback; the regression baseline.
#[derive(Clone, Copy, Debug)]
pub enum TileFormat {
    IndexedPng,
    Avif { quality: u8, speed: u8 },
    Png,
}

impl TileFormat {
    /// File extension used in tile paths and the Leaflet URL template.
    /// `IndexedPng` is still a `.png` on disk — the indexed-vs-truecolor
    /// difference is internal to PNG.
    pub fn extension(&self) -> &'static str {
        match self {
            TileFormat::IndexedPng | TileFormat::Png => "png",
            TileFormat::Avif { .. } => "avif",
        }
    }
}

/// Encode one in-memory RGB image to the chosen on-disk format, returning the
/// raw bytes (caller writes them to disk or uploads). The `image` argument is
/// returned alongside so the streaming pyramid accumulator can keep using it
/// without a copy.
pub fn encode_tile(
    img: image::ImageBuffer<Rgb<u8>, Vec<u8>>,
    fmt: TileFormat,
) -> Result<(image::ImageBuffer<Rgb<u8>, Vec<u8>>, Vec<u8>), String> {
    let bytes = match fmt {
        TileFormat::IndexedPng => match encode_indexed_png(&img)? {
            Some(b) => b,
            // Tile exceeded the 256-color palette budget — fall back to
            // truecolor PNG so we never crash. For Plain mode this
            // shouldn't happen; for Xet mode the caller should pick AVIF.
            None => encode_truecolor_png(&img)?,
        },
        TileFormat::Avif { quality, speed } => {
            let mut out: Vec<u8> = Vec::new();
            let enc = AvifEncoder::new_with_speed_quality(&mut out, speed, quality);
            enc.write_image(
                img.as_raw(),
                img.width(),
                img.height(),
                image::ExtendedColorType::Rgb8,
            )
            .map_err(|e| e.to_string())?;
            out
        }
        TileFormat::Png => encode_truecolor_png(&img)?,
    };
    Ok((img, bytes))
}

fn encode_truecolor_png(img: &image::ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor::new(Vec::new());
    img.write_to(&mut cursor, ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(cursor.into_inner())
}

/// Build and drive a `png::Encoder` for `pixels`, returning the completed
/// PNG byte stream. The crate's three direct `png::` call sites (indexed
/// leaf tiles in `encode_indexed_png`, and both single-canvas encoders in
/// `src/tiled/single.rs`) all use fast compression; when `indexed_palette`
/// is `Some`, the stream is an 8-bit `Indexed` image with that palette,
/// otherwise 8-bit `Rgb`.
pub(crate) fn encode_png(
    width: u32,
    height: u32,
    pixels: &[u8],
    indexed_palette: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        match indexed_palette {
            Some(palette) => {
                encoder.set_color(png::ColorType::Indexed);
                encoder.set_depth(png::BitDepth::Eight);
                encoder.set_palette(palette);
            }
            None => {
                encoder.set_color(png::ColorType::Rgb);
                encoder.set_depth(png::BitDepth::Eight);
            }
        }
        // Use the fdeflate fast path rather than the crate default (zlib level
        // 6 via flate2). The indexed pixel stream is highly structured (Hilbert
        // locality), so it compresses to within +0.4% of level 6 here while
        // encoding far faster — DEFLATE is a measurable slice of the leaf phase.
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(pixels).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// Try to encode `img` as an 8-bit indexed-color PNG, returning `Ok(None)` if
/// the tile uses more than 256 distinct RGB values (caller falls back).
///
/// Builds the palette on the fly from the pixels actually present — works for
/// any RGB content without needing the source LUT. The forward pass is O(N)
/// with an `FxHashMap<[u8;3], u8>` lookup per pixel.
fn encode_indexed_png(
    img: &image::ImageBuffer<Rgb<u8>, Vec<u8>>,
) -> Result<Option<Vec<u8>>, String> {
    let pixel_count = (img.width() as usize) * (img.height() as usize);
    let mut palette: Vec<[u8; 3]> = Vec::with_capacity(64);
    let mut idx_of: FxHashMap<[u8; 3], u8> = FxHashMap::default();
    idx_of.reserve(64);
    let mut indexed: Vec<u8> = Vec::with_capacity(pixel_count);

    let raw = img.as_raw();
    let mut i = 0;
    while i < raw.len() {
        let rgb = [raw[i], raw[i + 1], raw[i + 2]];
        let idx = match idx_of.get(&rgb) {
            Some(&v) => v,
            None => {
                if palette.len() >= 256 {
                    return Ok(None);
                }
                let v = palette.len() as u8;
                idx_of.insert(rgb, v);
                palette.push(rgb);
                v
            }
        };
        indexed.push(idx);
        i += 3;
    }

    let palette_bytes: Vec<u8> = palette.iter().flatten().copied().collect();

    encode_png(img.width(), img.height(), &indexed, Some(&palette_bytes)).map(Some)
}

/// Compute the starting Hilbert byte index for tile `(tx, ty)`.
pub fn tile_pixel_start(tx: u32, ty: u32, kh: u8, height_tiles: u32, square_pixels: u64) -> u64 {
    let sq = (tx / height_tiles) as u64;
    let sq_off = sq * square_pixels;
    let local_tx = tx % height_tiles;
    let tile_order = kh - TILE_LOG2;
    let base = xy2h_u64(local_tx as u64, ty as u64, tile_order) * TILE_AREA;
    sq_off + base
}

/// Async load stage: populate a `TILE×TILE`-byte tile buffer for tile `(tx, ty)`.
///
/// Walks the per-tile Hilbert byte range across source boundaries and issues
/// one async `fetch_range` per source overlap (≤ 2 in practice). Local sources
/// (`Data::Mapped` / `Data::Owned`) resolve via an in-memory copy of the
/// snapshot; HTTP
/// sources await an actual HTTP request, throttled by
/// [`crate::throttle::Throttle::global`].
pub async fn load_tile_bytes(
    tx: u32,
    ty: u32,
    kh: u8,
    height_tiles: u32,
    square_pixels: u64,
    total: u64,
    source_data: &[Data],
    cumulative_offsets: &[u64],
) -> anyhow::Result<Box<[u8; TILE_PIXELS]>> {
    let mut tile_buf = Box::new([0u8; TILE_PIXELS]);
    let tile_pixel_start = tile_pixel_start(tx, ty, kh, height_tiles, square_pixels);
    let readable_end = (tile_pixel_start + TILE_AREA).min(total);
    if tile_pixel_start >= readable_end {
        return Ok(tile_buf);
    }

    let mut pos = tile_pixel_start;
    let mut buf_off = 0usize;
    while pos < readable_end {
        let src_idx = cumulative_offsets.partition_point(|&c| c <= pos) - 1;
        let data = &source_data[src_idx];
        let src_end = if src_idx + 1 < cumulative_offsets.len() {
            cumulative_offsets[src_idx + 1]
        } else {
            total
        };
        let chunk_end = readable_end.min(src_end);
        let chunk_len = (chunk_end - pos) as usize;
        let local_off = pos - cumulative_offsets[src_idx];
        let fetched = data.fetch_range(local_off, chunk_len).await?;
        tile_buf[buf_off..buf_off + chunk_len].copy_from_slice(&fetched);
        pos = chunk_end;
        buf_off += chunk_len;
    }
    Ok(tile_buf)
}

/// CPU-only render from a pre-filled tile buffer.
/// Row-major LUT of the order-`TILE_LOG2` Hilbert index for tile-local pixel
/// coordinates: `LOCAL_CURVE_LUT[(py << TILE_LOG2) | px] == xy2h(px, py,
/// TILE_LOG2)`. Built once per process (~1 MiB) so the per-pixel render loops
/// replace a full order-`kh` Hilbert computation with one table lookup.
static LOCAL_CURVE_LUT: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();

fn local_curve_lut() -> &'static [u32] {
    LOCAL_CURVE_LUT.get_or_init(|| {
        (0u32..TILE_AREA as u32)
            .map(|i| xy2h_u64((i & (TILE - 1)) as u64, (i >> TILE_LOG2) as u64, TILE_LOG2) as u32)
            .collect()
    })
}

/// The per-tile Hilbert frame.
///
/// The order-`kh` Hilbert sub-curve covering leaf tile `(tx, ty)` is the
/// order-`TILE_LOG2` curve under one of eight dihedral transforms (coordinate
/// swap + per-axis complement), determined by the tile's position in the
/// coarse curve. Returns `(swap, cx, cy)` such that, for every tile-local
/// pixel `(px, py)`, with `base` the tile's starting curve index:
///
/// ```text
/// xy2h(tx*TILE + px, ty*TILE + py, kh) - base
///     == LOCAL_CURVE_LUT[frame(px, py)]
/// ```
/// where `frame(px, py)` swaps the coordinates when `swap` and XORs them with
/// `(cx, cy)` respectively (complement within a TILE-sized axis is XOR with
/// `TILE - 1`).
///
/// Computed by running the classic bitwise Hilbert rotation over the coarse
/// levels only (side length ≥ TILE): each level's quadrant bits are constant
/// across the tile, so the composition acting on the low `TILE_LOG2` bits is
/// a fixed affine map — swap parity `m` plus complements folded into the
/// constants. O(kh) per tile, replacing an O(kh) computation per pixel.
pub(super) fn tile_curve_frame(tx: u32, ty: u32, kh: u8) -> (bool, u32, u32) {
    debug_assert!(kh >= TILE_LOG2, "leaf tiles need kh ≥ TILE_LOG2");
    let mut x = (tx as u64) << TILE_LOG2;
    let mut y = (ty as u64) << TILE_LOG2;
    let mut m = 0u32;
    let mut side = 1u64 << (kh - 1);
    let tile_side = TILE as u64;
    while side >= tile_side {
        let rx = (x & side) > 0;
        let ry = (y & side) > 0;
        if !ry {
            if rx {
                // Complement the remaining (lower) coordinates; bits above the
                // current level are already consumed and never read again.
                x = !x;
                y = !y;
            }
            std::mem::swap(&mut x, &mut y);
            m += 1;
        }
        side >>= 1;
    }
    let mask = (TILE - 1) as u64;
    ((m & 1) == 1, (x & mask) as u32, (y & mask) as u32)
}

/// Inverse of the row-major `LOCAL_CURVE_LUT`: for curve position `v` within
/// a tile (identity frame), the packed raster index `(py << TILE_LOG2) | px`
/// of the pixel sitting at that curve position. Built once per process so
/// curve-order render loops can scatter colors to their raster coordinates
/// with one lookup.
static LOCAL_CURVE_TO_XY: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();

pub(super) fn local_curve_to_xy() -> &'static [u32] {
    LOCAL_CURVE_TO_XY.get_or_init(|| {
        let lut = local_curve_lut();
        let mut inv = vec![0u32; TILE_AREA as usize];
        for (i, &v) in lut.iter().enumerate() {
            inv[v as usize] = i as u32;
        }
        inv
    })
}

/// Curve offset of tile-local pixel `(px, py)` within its tile's byte range
/// (i.e. the `tile_buf` index), via the tile's precomputed Hilbert frame.
#[inline]
fn tile_local_curve_idx(frame: (bool, u32, u32), px: u32, py: u32) -> u64 {
    let (swap, cx, cy) = frame;
    let (a, b) = if swap {
        (py ^ cy, px ^ cx)
    } else {
        (px ^ cx, py ^ cy)
    };
    local_curve_lut()[((b << TILE_LOG2) | a) as usize] as u64
}

/// Render one plain-mode leaf tile from a pre-loaded `tile_buf` of
/// `TILE_PIXELS` curve-ordered bytes, mapping each byte through `pixel_lut`
/// and encoding to `fmt`. Bytes beyond `total` (the final partial tile) render
/// black. This is the bytes-per-pixel counterpart of
/// [`render_leaf_tile_xet_from_buf`].
pub fn render_leaf_tile_from_buf(
    tx: u32,
    ty: u32,
    kh: u8,
    height_tiles: u32,
    square_pixels: u64,
    total: u64,
    tile_buf: &[u8; TILE_PIXELS],
    pixel_lut: &[Rgb<u8>; 256],
    fmt: TileFormat,
) -> TileResult {
    let tile_pixel_start = tile_pixel_start(tx, ty, kh, height_tiles, square_pixels);
    let frame = tile_curve_frame(tx % height_tiles, ty, kh);

    let mut img = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(TILE, TILE);
    for py in 0..TILE {
        for px in 0..TILE {
            let local_idx = tile_local_curve_idx(frame, px, py);
            let pixel_idx = tile_pixel_start + local_idx;
            let color = if pixel_idx < total {
                pixel_lut[tile_buf[local_idx as usize] as usize]
            } else {
                Rgb([0u8, 0, 0])
            };
            img.put_pixel(px, py, color);
        }
    }
    encode_tile(img, fmt)
}

/// Whether a tile's pixel-screen position falls on a crosshatch stripe.
///
/// The pattern is two diagonals (`/` and `\`) of period `CROSSHATCH_PERIOD`,
/// each `CROSSHATCH_STRIPE_WIDTH` pixels wide. Their intersection produces the
/// visual "##" crosshatch. Tied to absolute (px, py) so the pattern doesn't
/// shift between tiles.
const CROSSHATCH_PERIOD: u32 = 8;
const CROSSHATCH_STRIPE_WIDTH: u32 = 2;
#[inline]
fn is_crosshatch_stripe(px: u32, py: u32) -> bool {
    let a = (px + py) % CROSSHATCH_PERIOD;
    let b = (px + (CROSSHATCH_PERIOD - py % CROSSHATCH_PERIOD)) % CROSSHATCH_PERIOD;
    a < CROSSHATCH_STRIPE_WIDTH || b < CROSSHATCH_STRIPE_WIDTH
}

/// Render a diff-mode leaf tile.
///
/// Three pixel paths, in priority order:
/// 1. `fills` — byte positions belonging to an `UnmatchedRegion` source.
///    Painted with a crosshatch pattern from `DiffFill::colors()`; the tile
///    buffer byte at this position is ignored.
/// 2. `tints` — byte positions belonging to a `OneSidedRange` source (JSON /
///    JSONL structure-aware diff). The tile buffer byte is a real file byte;
///    it's looked up via `plain_lut` then blended 50/50 with the fill's
///    crosshatch base color so the side of origin is visible while the byte
///    content stays legible.
/// 3. Aligned diff region. Byte is a signed delta; looked up via `pixel_lut`
///    (which the caller built with `build_diff_signed_lut`).
pub fn render_leaf_tile_diff(
    tx: u32,
    ty: u32,
    kh: u8,
    height_tiles: u32,
    square_pixels: u64,
    total: u64,
    tile_buf: &[u8; TILE_PIXELS],
    pixel_lut: &[Rgb<u8>; 256],
    plain_lut: &[Rgb<u8>; 256],
    fills: &[(u64, u64, DiffFill)],
    tints: &[(u64, u64, DiffFill)],
    fmt: TileFormat,
) -> TileResult {
    let tile_pixel_start = tile_pixel_start(tx, ty, kh, height_tiles, square_pixels);

    // Local view of the fills overlapping this tile. Avoids scanning the full
    // (potentially thousands of) fills list per pixel.
    let first_range = fills.partition_point(|r| r.1 <= tile_pixel_start);
    let first_tint = tints.partition_point(|r| r.1 <= tile_pixel_start);

    let (swap, cx, cy) = tile_curve_frame(tx % height_tiles, ty, kh);
    let xy_lut = local_curve_to_xy();
    let mut img = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(TILE, TILE);
    // Visit pixels in curve order: `pixel_idx` then increases strictly
    // monotonically, and since `fills`/`tints` are sorted by start and
    // non-overlapping, a forward-advancing cursor per list finds each pixel's
    // range in O(1) amortized instead of rescanning the local list per pixel.
    let mut fill_cur = first_range;
    let mut tint_cur = first_tint;
    for curve in 0..TILE_AREA {
        let pixel_idx = tile_pixel_start + curve;
        // Scatter target: unpack the identity-frame pixel at this curve
        // position, then undo this tile's frame (XOR-with-constant plus an
        // optional coordinate swap — each its own inverse) to raster coords.
        let packed = xy_lut[curve as usize];
        let (a, b) = (packed & (TILE - 1), packed >> TILE_LOG2);
        let (px, py) = if swap {
            (b ^ cx, a ^ cy)
        } else {
            (a ^ cx, b ^ cy)
        };
        let color = if pixel_idx >= total {
            Rgb([0u8, 0, 0])
        } else {
            while fill_cur < fills.len() && fills[fill_cur].1 <= pixel_idx {
                fill_cur += 1;
            }
            let byte = tile_buf[curve as usize];
            if fill_cur < fills.len() && fills[fill_cur].0 <= pixel_idx {
                let (stripe, base_c) = fills[fill_cur].2.colors();
                if is_crosshatch_stripe(px, py) {
                    stripe
                } else {
                    base_c
                }
            } else {
                while tint_cur < tints.len() && tints[tint_cur].1 <= pixel_idx {
                    tint_cur += 1;
                }
                if tint_cur < tints.len() && tints[tint_cur].0 <= pixel_idx {
                    blend_with_tint(plain_lut[byte as usize], tints[tint_cur].2)
                } else {
                    pixel_lut[byte as usize]
                }
            }
        };
        img.put_pixel(px, py, color);
    }
    encode_tile(img, fmt)
}

/// 50/50 blend of a byte-LUT color with a `DiffFill`'s base crosshatch color.
/// Keeps both the byte content (modulates luminance) and the side-of-origin
/// (the tint dominates the hue).
#[inline]
fn blend_with_tint(c: Rgb<u8>, fill: DiffFill) -> Rgb<u8> {
    let (_stripe, base_c) = fill.colors();
    Rgb([
        ((c[0] as u16 + base_c[0] as u16) / 2) as u8,
        ((c[1] as u16 + base_c[1] as u16) / 2) as u8,
        ((c[2] as u16 + base_c[2] as u16) / 2) as u8,
    ])
}

/// CPU-only render in xet/xorb mode from a pre-filled tile buffer.
///
/// Each pixel's byte is read from `tile_buf`. Its absolute file offset is
/// looked up in `xorb_ranges` to find its xorb's Tableau-20 color index. The
/// final RGB color is the Tableau color scaled per-channel by `byte / 255.0`.
pub fn render_leaf_tile_xet_from_buf(
    tx: u32,
    ty: u32,
    kh: u8,
    height_tiles: u32,
    square_pixels: u64,
    total: u64,
    tile_buf: &[u8; TILE_PIXELS],
    pixel_lut: &[Rgb<u8>; 256],
    xorb_ranges: &[(u64, u64, u8)],
    tableau: &[Rgb<u8>; 20],
    fmt: TileFormat,
) -> TileResult {
    let tile_pixel_start = tile_pixel_start(tx, ty, kh, height_tiles, square_pixels);

    let (swap, cx, cy) = tile_curve_frame(tx % height_tiles, ty, kh);
    let xy_lut = local_curve_to_xy();
    let mut img = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(TILE, TILE);
    // Visit pixels in curve order: `pixel_idx` then increases strictly
    // monotonically, and since `xorb_ranges` is sorted by start and
    // non-overlapping, a forward-advancing cursor finds each pixel's range in
    // O(1) amortized instead of a log-n binary search per pixel. Seed it past
    // every range that ends at or before the tile start (ends are sorted too,
    // since the ranges are sorted by start and non-overlapping).
    let mut cur = xorb_ranges.partition_point(|r| r.1 <= tile_pixel_start);
    for curve in 0..TILE_AREA {
        let pixel_idx = tile_pixel_start + curve;
        // Scatter target: unpack the identity-frame pixel at this curve
        // position, then undo this tile's frame (XOR-with-constant plus an
        // optional coordinate swap — each its own inverse) to raster coords.
        let packed = xy_lut[curve as usize];
        let (a, b) = (packed & (TILE - 1), packed >> TILE_LOG2);
        let (px, py) = if swap {
            (b ^ cx, a ^ cy)
        } else {
            (a ^ cx, b ^ cy)
        };
        let color = if pixel_idx >= total {
            Rgb([0u8, 0, 0])
        } else {
            while cur < xorb_ranges.len() && xorb_ranges[cur].1 <= pixel_idx {
                cur += 1;
            }
            let byte = tile_buf[curve as usize];
            if cur < xorb_ranges.len() && xorb_ranges[cur].0 <= pixel_idx {
                let t = tableau[xorb_ranges[cur].2 as usize];
                let scale = byte as u16;
                Rgb([
                    ((t[0] as u16 * scale + 127) / 255) as u8,
                    ((t[1] as u16 * scale + 127) / 255) as u8,
                    ((t[2] as u16 * scale + 127) / 255) as u8,
                ])
            } else {
                pixel_lut[byte as usize]
            }
        };
        img.put_pixel(px, py, color);
    }
    encode_tile(img, fmt)
}

/// x,y → Hilbert index using u64 intermediate arithmetic.
/// Supports curve orders up to 32 (files up to ~4 EiB).
fn xy2h_u64(x: u64, y: u64, order: u8) -> u64 {
    use fast_hilbert::xy2h;
    assert!(
        x <= u32::MAX as u64 && y <= u32::MAX as u64,
        "xy2h coordinates overflow u32"
    );
    xy2h::<u32>(x as u32, y as u32, order) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xy2h_u64_roundtrip_small() {
        let h = xy2h_u64(3, 4, 8);
        let (x, y) = crate::geometry::hilbert_to_xy_u64(h, 8);
        assert_eq!((x, y), (3, 4));
    }

    /// End-to-end: `render_leaf_tile_from_buf` must produce the same pixels
    /// as a direct reference implementation that computes every pixel's
    /// Hilbert index with `xy2h_u64`.
    #[test]
    fn render_leaf_tile_matches_xy2h_reference() {
        let kh = 13u8;
        let height_tiles = 1u32 << (kh - TILE_LOG2);
        let square_pixels = 1u64 << (2 * kh as u32);
        let total = square_pixels * 3;
        let mut tile_buf = Box::new([0u8; TILE_PIXELS]);
        for (i, b) in tile_buf.iter_mut().enumerate() {
            *b = (i * 2654435761 % 256) as u8;
        }
        let pixel_lut: [Rgb<u8>; 256] =
            std::array::from_fn(|i| Rgb([(i * 7) as u8, (i * 13) as u8, (i * 29) as u8]));
        for &(tx, ty) in &[(0u32, 0u32), (1, 2), (3, 3), (7, 5)] {
            let (img, _) = render_leaf_tile_from_buf(
                tx,
                ty,
                kh,
                height_tiles,
                square_pixels,
                total,
                &tile_buf,
                &pixel_lut,
                TileFormat::Png,
            )
            .unwrap();
            let sq = (tx / height_tiles) as u64;
            let sq_off = sq * square_pixels;
            let local_tx = tx % height_tiles;
            let base = xy2h_u64(local_tx as u64, ty as u64, kh - TILE_LOG2) * TILE_AREA;
            for py in 0..TILE {
                for px in 0..TILE {
                    let lx = ((local_tx as u64) << TILE_LOG2) | px as u64;
                    let ly = ((ty as u64) << TILE_LOG2) | py as u64;
                    let local_idx = xy2h_u64(lx, ly, kh);
                    let pixel_idx = sq_off + local_idx;
                    let expected = if pixel_idx < total {
                        pixel_lut[tile_buf[(local_idx - base) as usize] as usize]
                    } else {
                        Rgb([0u8, 0, 0])
                    };
                    assert_eq!(
                        img.get_pixel(px, py),
                        &expected,
                        "tile ({tx},{ty}) pixel ({px},{py})"
                    );
                }
            }
        }
    }

    /// `render_leaf_tile_xet_from_buf` must reproduce the per-pixel
    /// binary-search coloring it replaced: tableau-scaled colors inside xorb
    /// ranges, the plain LUT in the gaps between ranges, and black past
    /// `total`. The reference here is an independent per-pixel linear find,
    /// not the forward cursor under test.
    #[test]
    fn render_leaf_tile_xet_matches_binary_search_reference() {
        let kh = 13u8;
        let height_tiles = 1u32 << (kh - TILE_LOG2);
        let square_pixels = 1u64 << (2 * kh as u32);
        let total = square_pixels * 3;
        let mut tile_buf = Box::new([0u8; TILE_PIXELS]);
        for (i, b) in tile_buf.iter_mut().enumerate() {
            *b = (i * 2654435761 % 256) as u8;
        }
        let pixel_lut: [Rgb<u8>; 256] =
            std::array::from_fn(|i| Rgb([(i * 7) as u8, (i * 13) as u8, (i * 29) as u8]));
        let tableau: [Rgb<u8>; 20] = std::array::from_fn(|i| {
            Rgb([(i * 11 + 3) as u8, (i * 5 + 90) as u8, (i * 31 + 40) as u8])
        });
        // Sorted by start, non-overlapping, with gaps between the ranges so
        // both the Some and None branches get exercised inside each tile.
        let xorb_ranges: Vec<(u64, u64, u8)> = (0..20u64)
            .map(|k| (k * 700_000 + 100, k * 700_000 + 700_000, k as u8 % 20))
            .collect();
        for &(tx, ty) in &[(0u32, 0u32), (1, 2), (3, 3), (7, 5)] {
            let (img, _) = render_leaf_tile_xet_from_buf(
                tx,
                ty,
                kh,
                height_tiles,
                square_pixels,
                total,
                &tile_buf,
                &pixel_lut,
                &xorb_ranges,
                &tableau,
                TileFormat::Png,
            )
            .unwrap();
            let sq = (tx / height_tiles) as u64;
            let sq_off = sq * square_pixels;
            let local_tx = tx % height_tiles;
            let base = xy2h_u64(local_tx as u64, ty as u64, kh - TILE_LOG2) * TILE_AREA;
            for py in 0..TILE {
                for px in 0..TILE {
                    let lx = ((local_tx as u64) << TILE_LOG2) | px as u64;
                    let ly = ((ty as u64) << TILE_LOG2) | py as u64;
                    let local_idx = xy2h_u64(lx, ly, kh);
                    let pixel_idx = sq_off + local_idx;
                    let byte = tile_buf[(local_idx - base) as usize];
                    let expected = if pixel_idx >= total {
                        Rgb([0u8, 0, 0])
                    } else {
                        let idx = xorb_ranges
                            .iter()
                            .find(|&&(s, e, _)| s <= pixel_idx && pixel_idx < e)
                            .map(|&(_, _, c)| c);
                        match idx {
                            Some(c) => {
                                let t = tableau[c as usize];
                                let scale = byte as u16;
                                Rgb([
                                    ((t[0] as u16 * scale + 127) / 255) as u8,
                                    ((t[1] as u16 * scale + 127) / 255) as u8,
                                    ((t[2] as u16 * scale + 127) / 255) as u8,
                                ])
                            }
                            None => pixel_lut[byte as usize],
                        }
                    };
                    assert_eq!(
                        img.get_pixel(px, py),
                        &expected,
                        "tile ({tx},{ty}) pixel ({px},{py})"
                    );
                }
            }
        }
    }

    /// `tile_curve_frame` + `tile_local_curve_idx` must reproduce
    /// `xy2h(lx, ly, kh) - base` exactly: the frame/LUT fast path is only
    /// valid if it is the identity on the real Hilbert indexing.
    #[test]
    fn tile_curve_frame_identity() {
        let check_tile = |tx: u32, ty: u32, kh: u8, pixels: &[(u32, u32)]| {
            let tile_order = kh - TILE_LOG2;
            let base = xy2h_u64(tx as u64, ty as u64, tile_order) * TILE_AREA;
            let frame = tile_curve_frame(tx, ty, kh);
            for &(px, py) in pixels {
                let lx = ((tx as u64) << TILE_LOG2) | px as u64;
                let ly = ((ty as u64) << TILE_LOG2) | py as u64;
                let expected = xy2h_u64(lx, ly, kh) - base;
                assert_eq!(
                    tile_local_curve_idx(frame, px, py),
                    expected,
                    "tile ({tx},{ty}) kh={kh} pixel ({px},{py})"
                );
            }
        };
        // All pixels of the four kh=10 tiles (covers one orientation state).
        let all_pixels: Vec<(u32, u32)> = (0..TILE)
            .flat_map(|px| (0..TILE).map(move |py| (px, py)))
            .collect();
        for ty in 0..2u32 {
            for tx in 0..2u32 {
                check_tile(tx, ty, 10, &all_pixels);
            }
        }
        // Sampled pixels across tiles of several orders — spans all eight
        // dihedral states (swap parity and complements vary with the tile's
        // coarse position and the order's parity).
        let samples: Vec<(u32, u32)> = [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (511, 511),
            (300, 17),
            (256, 256),
            (7, 509),
            (510, 1),
            (100, 400),
            (3, 3),
            (511, 0),
        ]
        .to_vec();
        for kh in [11u8, 12, 13, 16, 21, 24] {
            let t = kh - TILE_LOG2;
            let tiles = 1u64 << (2 * t);
            let step = (tiles / 512).max(1);
            for tile_i in (0..tiles).step_by(step as usize) {
                let tx = (tile_i % (1 << t)) as u32;
                let ty = (tile_i / (1 << t)) as u32;
                check_tile(tx, ty, kh, &samples);
            }
        }
    }

    /// The diff renderer must match a naive per-pixel reference exactly —
    /// only the image matters, not the pixel visit order. The fills and tints
    /// below span tile boundaries, sit before the first tile, cover whole
    /// tiles, and end past `total`.
    #[test]
    fn render_leaf_tile_diff_matches_reference() {
        let kh = 11u8; // 4×4 tile grid, covers several frame orientations
        let square_pixels = 1u64 << (2 * kh);
        let total = square_pixels * 3 / 4;
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let tile_buf: [u8; TILE_PIXELS] = core::array::from_fn(|_| next() as u8);
        let pixel_lut: [Rgb<u8>; 256] =
            core::array::from_fn(|i| Rgb([i as u8, (i * 3) as u8, (i * 7) as u8]));
        let plain_lut: [Rgb<u8>; 256] =
            core::array::from_fn(|i| Rgb([(i * 5) as u8, i as u8, (i * 11) as u8]));
        let fills: Vec<(u64, u64, DiffFill)> = [
            (0, 100_000, DiffFill::Grey),
            (total / 2, total / 2 + 7, DiffFill::Green),
            (
                square_pixels / 2,
                square_pixels / 2 + 300_000,
                DiffFill::Red,
            ),
            (total - 5, total + 50, DiffFill::Red),
            (square_pixels + 1, square_pixels + 2, DiffFill::Green),
        ]
        .into_iter()
        .collect();
        let tints: Vec<(u64, u64, DiffFill)> = [
            (50_000, 150_000, DiffFill::Green),
            (2 * square_pixels, 3 * square_pixels, DiffFill::Grey),
        ]
        .into_iter()
        .collect();
        let check_tile = |tx: u32, ty: u32, kh: u8| {
            let height_tiles = 1u32 << (kh - TILE_LOG2);
            let square_pixels = 1u64 << (2 * kh);
            let img = render_leaf_tile_diff(
                tx,
                ty,
                kh,
                height_tiles,
                square_pixels,
                total,
                &tile_buf,
                &pixel_lut,
                &plain_lut,
                &fills,
                &tints,
                TileFormat::Png,
            )
            .unwrap()
            .0;
            let tile_order = kh - TILE_LOG2;
            for py in 0..TILE {
                for px in 0..TILE {
                    let local_idx =
                        tile_local_curve_idx(tile_curve_frame(tx % height_tiles, ty, kh), px, py);
                    let pixel_idx = (tx as u64 / height_tiles as u64) * square_pixels
                        + xy2h_u64(tx as u64 % height_tiles as u64, ty as u64, tile_order)
                            * TILE_AREA
                        + local_idx;
                    let expected = if pixel_idx >= total {
                        Rgb([0u8, 0, 0])
                    } else {
                        let byte = tile_buf[local_idx as usize];
                        if let Some(&(_, _, f)) =
                            fills.iter().find(|r| pixel_idx >= r.0 && pixel_idx < r.1)
                        {
                            let (stripe, base_c) = f.colors();
                            if is_crosshatch_stripe(px, py) {
                                stripe
                            } else {
                                base_c
                            }
                        } else if let Some(&(_, _, t)) =
                            tints.iter().find(|r| pixel_idx >= r.0 && pixel_idx < r.1)
                        {
                            blend_with_tint(plain_lut[byte as usize], t)
                        } else {
                            pixel_lut[byte as usize]
                        }
                    };
                    assert_eq!(
                        img.get_pixel(px, py),
                        &expected,
                        "diff tile ({tx},{ty}) pixel ({px},{py})"
                    );
                }
            }
        };
        // All tiles of the kh=11 grid, then sampled tiles of higher orders —
        // spans all eight dihedral frame states.
        for ty in 0..4u32 {
            for tx in 0..4u32 {
                check_tile(tx, ty, 11);
            }
        }
        for kh in [12u8, 13, 16, 21] {
            let t = kh - TILE_LOG2;
            let tiles = 1u64 << (2 * t);
            let step = (tiles / 64).max(1);
            for tile_i in (0..tiles).step_by(step as usize) {
                let tx = (tile_i % (1 << t)) as u32;
                let ty = (tile_i / (1 << t)) as u32;
                check_tile(tx, ty, kh);
            }
        }
    }
}
