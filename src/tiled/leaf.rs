use std::io::Cursor;

use image::{ImageFormat, Rgb};

use crate::data::Data;

type TileResult = Result<(image::ImageBuffer<Rgb<u8>, Vec<u8>>, Vec<u8>), String>;

fn encode_png(img: image::ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<(image::ImageBuffer<Rgb<u8>, Vec<u8>>, Vec<u8>), String> {
    let mut cursor = Cursor::new(Vec::new());
    img.write_to(&mut cursor, ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok((img, cursor.into_inner()))
}

/// Render one 256×256 leaf tile, returning (pixels, png_bytes).
///
/// Each tile at the highest zoom level covers a 256×256-pixel region that
/// corresponds to a contiguous Hilbert sub-curve of exactly 65536 bytes.
/// For local sources, byte access is zero-copy via `Deref`. For remote
/// `Data::Http` sources, each source segment is fetched via a single HTTP
/// range request.
///
/// Uses u64 for Hilbert indices to support files > 16 GiB.
pub fn render_leaf_tile(
    tx: u32,
    ty: u32,
    kh: u8,
    height_tiles: u32,
    square_pixels: u64,
    total: u64,
    source_data: &[Data],
    cumulative_offsets: &[u64],
    pixel_lut: &[Rgb<u8>; 256],
) -> TileResult {
    const TILE: u32 = 256;
    const TILE_PIXELS: usize = (TILE as usize) * (TILE as usize);
    const TILE_AREA: u64 = TILE_PIXELS as u64;

    let sq = (tx / height_tiles) as u64;
    let sq_off = sq * square_pixels;
    let local_tx = tx % height_tiles;

    let tile_order = kh - 8;
    let base = xy2h_u64(local_tx as u64, ty as u64, tile_order) * TILE_AREA;
    let tile_pixel_start = sq_off + base;

    let mut tile_buf = [0u8; TILE_PIXELS];
    let readable_end = (tile_pixel_start + TILE_AREA).min(total);
    if tile_pixel_start < readable_end {
        let mut pos = tile_pixel_start;
        let mut buf_off = 0usize;
        while pos < readable_end {
            let src_idx = cumulative_offsets.partition_point(|&c| c <= pos) - 1;
            let data = &source_data[src_idx];
            // Source end: next source's start, or total for the last source.
            let src_end = if src_idx + 1 < cumulative_offsets.len() {
                cumulative_offsets[src_idx + 1]
            } else {
                total
            };
            let chunk_end = readable_end.min(src_end);
            let chunk_len = (chunk_end - pos) as usize;
            let local_off = pos - cumulative_offsets[src_idx];
            let fetched = data.fetch_range(local_off, chunk_len)
                .map_err(|e| e.to_string())?;
            tile_buf[buf_off..buf_off + chunk_len].copy_from_slice(&fetched);
            pos = chunk_end;
            buf_off += chunk_len;
        }
    }

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
    encode_png(img)
}


/// Render one 256×256 leaf tile using position-based dtype coloring (safetensors mode).
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
) -> TileResult {
    const TILE: u32 = 256;
    const TILE_PIXELS: usize = (TILE as usize) * (TILE as usize);
    const TILE_AREA: u64 = TILE_PIXELS as u64;

    let sq = (tx / height_tiles) as u64;
    let sq_off = sq * square_pixels;
    let local_tx = tx % height_tiles;

    let tile_order = kh - 8;
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
    encode_png(img)
}

/// Render one 256×256 leaf tile using xorb coloring.
///
/// Each pixel's byte is read like in `render_leaf_tile`. Then the byte's
/// absolute file offset is looked up in `xorb_ranges` to find its xorb's
/// Tableau-20 color index. The final RGB color is the Tableau color scaled
/// per-channel by `byte / 255.0` (so byte=0 → black, byte=255 → full xorb
/// color, mid-range bytes → dimmer versions of the xorb hue).
///
/// Bytes outside any xorb range fall back to the default `pixel_lut`.
pub fn render_leaf_tile_xet(
    tx: u32,
    ty: u32,
    kh: u8,
    height_tiles: u32,
    square_pixels: u64,
    total: u64,
    source_data: &[Data],
    cumulative_offsets: &[u64],
    pixel_lut: &[Rgb<u8>; 256],
    xorb_ranges: &[(u64, u64, u8)],
    tableau: &[Rgb<u8>; 20],
) -> TileResult {
    const TILE: u32 = 256;
    const TILE_PIXELS: usize = (TILE as usize) * (TILE as usize);
    const TILE_AREA: u64 = TILE_PIXELS as u64;

    let sq = (tx / height_tiles) as u64;
    let sq_off = sq * square_pixels;
    let local_tx = tx % height_tiles;

    let tile_order = kh - 8;
    let base = xy2h_u64(local_tx as u64, ty as u64, tile_order) * TILE_AREA;
    let tile_pixel_start = sq_off + base;

    let mut tile_buf = [0u8; TILE_PIXELS];
    let readable_end = (tile_pixel_start + TILE_AREA).min(total);
    if tile_pixel_start < readable_end {
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
            let fetched = data.fetch_range(local_off, chunk_len)
                .map_err(|e| e.to_string())?;
            tile_buf[buf_off..buf_off + chunk_len].copy_from_slice(&fetched);
            pos = chunk_end;
            buf_off += chunk_len;
        }
    }

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
                        let scale = byte as u16; // 0..=255
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
    encode_png(img)
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

/// Render one 256×256 leaf tile in sorted mode from pre-built per-source histograms.
pub fn render_leaf_tile_sorted(
    tx: u32,
    ty: u32,
    kh: u8,
    height_tiles: u32,
    square_pixels: u64,
    total: u64,
    histograms: &[([u64; 257], u64)],
    pixel_lut: &[Rgb<u8>; 256],
) -> TileResult {
    const TILE: u32 = 256;
    const TILE_PIXELS: usize = (TILE as usize) * (TILE as usize);
    const TILE_AREA: u64 = TILE_PIXELS as u64;

    let sq = (tx / height_tiles) as u64;
    let sq_off = sq * square_pixels;
    let local_tx = tx % height_tiles;

    let tile_order = kh - 8;
    let base = xy2h_u64(local_tx as u64, ty as u64, tile_order) * TILE_AREA;
    let tile_pixel_start = sq_off + base;
    let tile_pixel_end = (tile_pixel_start + TILE_AREA).min(total);

    let mut tile_buf = [0u8; TILE_PIXELS];

    if tile_pixel_start < tile_pixel_end {
        for (prefix, src_offset) in histograms {
            let src_end = src_offset + prefix[256];
            if *src_offset >= tile_pixel_end || src_end <= tile_pixel_start {
                continue;
            }
            for v in 0usize..256 {
                let range_start = src_offset + prefix[v];
                let range_end = src_offset + prefix[v + 1];
                if range_end <= tile_pixel_start || range_start >= tile_pixel_end {
                    continue;
                }
                let fill_start = range_start.max(tile_pixel_start);
                let fill_end = range_end.min(tile_pixel_end);
                let buf_start = (fill_start - tile_pixel_start) as usize;
                let buf_end = (fill_end - tile_pixel_start) as usize;
                tile_buf[buf_start..buf_end].fill(v as u8);
            }
        }
    }

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
    encode_png(img)
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
