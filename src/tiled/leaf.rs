use std::collections::HashMap;
use std::io::Cursor;

use image::codecs::avif::AvifEncoder;
use image::{ImageEncoder, ImageFormat, Rgb};

use crate::data::Data;
use crate::safetensors::DiffFill;

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
///   for Plain/Dtype mode leaves (where every pixel comes from a fixed
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
            // truecolor PNG so we never crash. For Plain/Dtype mode this
            // shouldn't happen; for Xet mode the caller should pick AVIF.
            None => encode_truecolor_png(&img)?,
        },
        TileFormat::Avif { quality, speed } => {
            let mut out: Vec<u8> = Vec::new();
            let enc = AvifEncoder::new_with_speed_quality(&mut out, speed, quality);
            enc.write_image(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgb8)
                .map_err(|e| e.to_string())?;
            out
        }
        TileFormat::Png => encode_truecolor_png(&img)?,
    };
    Ok((img, bytes))
}

fn encode_truecolor_png(img: &image::ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor::new(Vec::new());
    img.write_to(&mut cursor, ImageFormat::Png).map_err(|e| e.to_string())?;
    Ok(cursor.into_inner())
}

/// Try to encode `img` as an 8-bit indexed-color PNG, returning `Ok(None)` if
/// the tile uses more than 256 distinct RGB values (caller falls back).
///
/// Builds the palette on the fly from the pixels actually present — works for
/// any RGB content without needing the source LUT. The forward pass is O(N)
/// with a `HashMap<[u8;3], u8>` lookup per pixel.
fn encode_indexed_png(img: &image::ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<Vec<u8>>, String> {
    let pixel_count = (img.width() as usize) * (img.height() as usize);
    let mut palette: Vec<[u8; 3]> = Vec::with_capacity(64);
    let mut idx_of: HashMap<[u8; 3], u8> = HashMap::with_capacity(64);
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
        // Default compression. The `png` crate uses zlib level 6 by default;
        // we leave it alone — the indexed pixel stream is already highly
        // structured (Hilbert curve = strong locality) so DEFLATE wins
        // most of its compression off the predictable byte sequence rather
        // than the compression level knob.
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(&indexed).map_err(|e| e.to_string())?;
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

    let mut img = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(TILE, TILE);
    for py in 0..TILE {
        let ly = ty * TILE + py;
        for px in 0..TILE {
            let lx = local_tx * TILE + px;
            let local_idx = xy2h_u64(lx as u64, ly as u64, kh);
            let pixel_idx = sq_off + local_idx;
            let color = if pixel_idx < total {
                pixel_lut[tile_buf[(local_idx - base) as usize] as usize]
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

/// Render a diff-mode leaf tile. Same as `render_leaf_tile_from_buf` but with
/// an overlay: byte positions inside any `fills` range are painted with a
/// crosshatch pattern instead of going through the byte LUT. Used for tensors
/// / files that exist on only one side of the diff (and so have no per-byte
/// signal to encode).
pub fn render_leaf_tile_diff(
    tx: u32,
    ty: u32,
    kh: u8,
    height_tiles: u32,
    square_pixels: u64,
    total: u64,
    tile_buf: &[u8; TILE_PIXELS],
    pixel_lut: &[Rgb<u8>; 256],
    fills: &[(u64, u64, DiffFill)],
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

    let mut img = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(TILE, TILE);
    for py in 0..TILE {
        let ly = ty * TILE + py;
        for px in 0..TILE {
            let lx = local_tx * TILE + px;
            let local_idx = xy2h_u64(lx as u64, ly as u64, kh);
            let pixel_idx = sq_off + local_idx;
            let color = if pixel_idx >= total {
                Rgb([0u8, 0, 0])
            } else {
                let mut fill: Option<DiffFill> = None;
                for &(start, end, f) in &local_fills {
                    if pixel_idx >= start && pixel_idx < end { fill = Some(f); break; }
                }
                match fill {
                    Some(f) => {
                        let (stripe, base_c) = f.colors();
                        if is_crosshatch_stripe(px, py) { stripe } else { base_c }
                    }
                    None => pixel_lut[tile_buf[(local_idx - base) as usize] as usize],
                }
            };
            img.put_pixel(px, py, color);
        }
    }
    encode_tile(img, fmt)
}

/// Render one `TILE×TILE` leaf tile using position-based dtype coloring (safetensors mode).
///
/// Does not read file bytes — color is determined entirely by byte position via
/// `color_ranges`, a sorted list of `(start, end, color)` entries from
/// `safetensors::build_color_ranges`.
pub fn render_leaf_tile_dtype(
    tx: u32,
    ty: u32,
    kh: u8,
    height_tiles: u32,
    square_pixels: u64,
    total: u64,
    color_ranges: &[(u64, u64, image::Rgb<u8>)],
    fmt: TileFormat,
) -> TileResult {
    let sq = (tx / height_tiles) as u64;
    let sq_off = sq * square_pixels;
    let local_tx = tx % height_tiles;

    let tile_order = kh - TILE_LOG2;
    let base = xy2h_u64(local_tx as u64, ty as u64, tile_order) * TILE_AREA;
    let tile_pixel_start = sq_off + base;
    let tile_pixel_end = (tile_pixel_start + TILE_AREA).min(total);

    let first_range = color_ranges.partition_point(|r| r.1 <= tile_pixel_start);
    let local_ranges: Vec<(u64, u64, image::Rgb<u8>)> = color_ranges[first_range..]
        .iter()
        .take_while(|r| r.0 < tile_pixel_end)
        .copied()
        .collect();

    let mut img = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::new(TILE, TILE);
    for py in 0..TILE {
        let ly = ty * TILE + py;
        for px in 0..TILE {
            let lx = local_tx * TILE + px;
            let local_idx = xy2h_u64(lx as u64, ly as u64, kh);
            let pixel_idx = sq_off + local_idx;
            let color = if pixel_idx < total {
                let mut found = image::Rgb([0u8, 0, 0]);
                for &(start, end, c) in &local_ranges {
                    if pixel_idx >= start && pixel_idx < end {
                        found = c;
                        break;
                    }
                }
                found
            } else {
                image::Rgb([0u8, 0, 0])
            };
            img.put_pixel(px, py, color);
        }
    }
    encode_tile(img, fmt)
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

    let mut img = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(TILE, TILE);
    for py in 0..TILE {
        let ly = ty * TILE + py;
        for px in 0..TILE {
            let lx = local_tx * TILE + px;
            let local_idx = xy2h_u64(lx as u64, ly as u64, kh);
            let pixel_idx = sq_off + local_idx;
            let color = if pixel_idx < total {
                let byte = tile_buf[(local_idx - base) as usize];
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
}
