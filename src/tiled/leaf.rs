use std::io::Cursor;

use image::codecs::avif::AvifEncoder;
use image::{ImageEncoder, ImageFormat, Rgb};
use rustc_hash::FxHashMap;

use crate::data::{Data, DiffFill};

pub const TILE: u32 = 512;
pub const TILE_LOG2: u8 = TILE.trailing_zeros() as u8;
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

    let mut out: Vec<u8> = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, img.width(), img.height());
        encoder.set_color(png::ColorType::Indexed);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_palette(&palette_bytes[..]);
        // Use the fdeflate fast path rather than the crate default (zlib level
        // 6 via flate2). The indexed pixel stream is highly structured (Hilbert
        // locality), so it compresses to within +0.4% of level 6 here while
        // encoding far faster — DEFLATE is a measurable slice of the leaf phase.
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer
            .write_image_data(&indexed)
            .map_err(|e| e.to_string())?;
    }
    Ok(Some(out))
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
/// (`Data::Mapped` / `Data::Owned`) resolve via a memcpy off the mmap; HTTP
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
fn tile_curve_frame(tx: u32, ty: u32, kh: u8) -> (bool, u32, u32) {
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
    let sq = (tx / height_tiles) as u64;
    let sq_off = sq * square_pixels;
    let local_tx = tx % height_tiles;
    let tile_order = kh - TILE_LOG2;
    let base = xy2h_u64(local_tx as u64, ty as u64, tile_order) * TILE_AREA;
    let frame = tile_curve_frame(local_tx, ty, kh);

    let mut img = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(TILE, TILE);
    for py in 0..TILE {
        for px in 0..TILE {
            let local_idx = tile_local_curve_idx(frame, px, py);
            let pixel_idx = sq_off + base + local_idx;
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
    let sq = (tx / height_tiles) as u64;
    let sq_off = sq * square_pixels;
    let local_tx = tx % height_tiles;
    let tile_order = kh - TILE_LOG2;
    let base = xy2h_u64(local_tx as u64, ty as u64, tile_order) * TILE_AREA;
    let tile_pixel_start = sq_off + base;
    let tile_pixel_end = (tile_pixel_start + TILE_AREA).min(total);

    // Local view of the fills overlapping this tile. Avoids scanning the full
    // (potentially thousands of) fills list per pixel.
    let first_range = fills.partition_point(|r| r.1 <= tile_pixel_start);
    let local_fills: Vec<(u64, u64, DiffFill)> = fills[first_range..]
        .iter()
        .take_while(|r| r.0 < tile_pixel_end)
        .copied()
        .collect();

    let first_tint = tints.partition_point(|r| r.1 <= tile_pixel_start);
    let local_tints: Vec<(u64, u64, DiffFill)> = tints[first_tint..]
        .iter()
        .take_while(|r| r.0 < tile_pixel_end)
        .copied()
        .collect();

    let frame = tile_curve_frame(local_tx, ty, kh);
    let mut img = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(TILE, TILE);
    for py in 0..TILE {
        for px in 0..TILE {
            let local_idx = tile_local_curve_idx(frame, px, py);
            let pixel_idx = sq_off + base + local_idx;
            let color = if pixel_idx >= total {
                Rgb([0u8, 0, 0])
            } else {
                let byte = tile_buf[local_idx as usize];
                let mut fill: Option<DiffFill> = None;
                for &(start, end, f) in &local_fills {
                    if pixel_idx >= start && pixel_idx < end {
                        fill = Some(f);
                        break;
                    }
                }
                if let Some(f) = fill {
                    let (stripe, base_c) = f.colors();
                    if is_crosshatch_stripe(px, py) {
                        stripe
                    } else {
                        base_c
                    }
                } else {
                    let mut tint: Option<DiffFill> = None;
                    for &(start, end, f) in &local_tints {
                        if pixel_idx >= start && pixel_idx < end {
                            tint = Some(f);
                            break;
                        }
                    }
                    match tint {
                        Some(t) => blend_with_tint(plain_lut[byte as usize], t),
                        None => pixel_lut[byte as usize],
                    }
                }
            };
            img.put_pixel(px, py, color);
        }
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
    let sq = (tx / height_tiles) as u64;
    let sq_off = sq * square_pixels;
    let local_tx = tx % height_tiles;
    let tile_order = kh - TILE_LOG2;
    let base = xy2h_u64(local_tx as u64, ty as u64, tile_order) * TILE_AREA;

    let frame = tile_curve_frame(local_tx, ty, kh);
    let mut img = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(TILE, TILE);
    for py in 0..TILE {
        for px in 0..TILE {
            let local_idx = tile_local_curve_idx(frame, px, py);
            let pixel_idx = sq_off + base + local_idx;
            let color = if pixel_idx < total {
                let byte = tile_buf[local_idx as usize];
                match xorb_color_idx(xorb_ranges, pixel_idx) {
                    Some(idx) => {
                        let t = tableau[idx as usize];
                        let scale = byte as u16;
                        Rgb([
                            ((t[0] as u16 * scale + 127) / 255) as u8,
                            ((t[1] as u16 * scale + 127) / 255) as u8,
                            ((t[2] as u16 * scale + 127) / 255) as u8,
                        ])
                    }
                    None => pixel_lut[byte as usize],
                }
            } else {
                Rgb([0u8, 0, 0])
            };
            img.put_pixel(px, py, color);
        }
    }
    encode_tile(img, fmt)
}

fn xorb_color_idx(ranges: &[(u64, u64, u8)], pixel_idx: u64) -> Option<u8> {
    if ranges.is_empty() {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = ranges.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (s, e, c) = ranges[mid];
        if pixel_idx < s {
            hi = mid;
        } else if pixel_idx >= e {
            lo = mid + 1;
        } else {
            return Some(c);
        }
    }
    None
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
}
