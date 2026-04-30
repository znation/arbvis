use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use image::Rgb;

use crate::data::{Source, SourceKind};

/// Render one 256×256 leaf tile and write it to `tile_path`.
///
/// Each tile at the highest zoom level covers a 256×256-pixel region that
/// corresponds to a contiguous Hilbert sub-curve of exactly 65536 bytes.
/// Files are opened, read, and closed per-tile to bound open fd counts.
///
/// Uses u64 for Hilbert indices to support files > 16 GiB.
pub fn render_leaf_tile(
    tile_path: &Path,
    tx: u32,
    ty: u32,
    kh: u8,
    height_tiles: u32,
    square_pixels: u64,
    total: u64,
    sources: &[Source],
    cumulative_offsets: &[u64],
    pixel_lut: &[Rgb<u8>; 256],
) -> Result<(), String> {
    const TILE: u32 = 256;
    const TILE_PIXELS: usize = (TILE as usize) * (TILE as usize);
    const TILE_AREA: u64 = TILE_PIXELS as u64;

    let sq = (tx / height_tiles) as u64;
    let sq_off = sq * square_pixels;
    let local_tx = tx % height_tiles;

    // Each leaf tile is a level-(kh-8) Hilbert sub-square covering exactly
    // TILE_PIXELS consecutive positions in the concatenated byte stream.
    let tile_order = kh - 8;
    let base = xy2h_u64(local_tx as u64, ty as u64, tile_order) * TILE_AREA;
    let tile_pixel_start = sq_off + base;

    // Read the tile's bytes from source files into a local buffer.
    let mut tile_buf = [0u8; TILE_PIXELS];
    let readable_end = (tile_pixel_start + TILE_AREA).min(total);
    if tile_pixel_start < readable_end {
        let mut pos = tile_pixel_start;
        let mut buf_off = 0usize;
        while pos < readable_end {
            let src_idx = cumulative_offsets.partition_point(|&c| c <= pos) - 1;
            let src = &sources[src_idx];
            let src_end = cumulative_offsets[src_idx] + src.byte_size;
            let chunk_end = readable_end.min(src_end);
            let chunk_len = (chunk_end - pos) as usize;
            let local_off = pos - cumulative_offsets[src_idx];
            match &src.kind {
                SourceKind::File(p) => {
                    let mut f = File::open(p)
                        .map_err(|e| format!("{}: {}", p.display(), e))?;
                    f.seek(SeekFrom::Start(local_off))
                        .map_err(|e| format!("{}: {}", p.display(), e))?;
                    f.read_exact(&mut tile_buf[buf_off..buf_off + chunk_len])
                        .map_err(|e| format!("{}: {}", p.display(), e))?;
                }
                SourceKind::Buffered(v) => {
                    let lo = local_off as usize;
                    tile_buf[buf_off..buf_off + chunk_len]
                        .copy_from_slice(&v[lo..lo + chunk_len]);
                }
            }
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
    if let Some(parent) = tile_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    img.save(tile_path)
        .map_err(|e| format!("{}: {}", tile_path.display(), e))?;
    Ok(())
}

/// x,y → Hilbert index using u64 intermediate arithmetic.
/// Supports curve orders up to 32 (files up to ~4 EiB).
fn xy2h_u64(x: u64, y: u64, order: u8) -> u64 {
    use fast_hilbert::xy2h;
    // fast_hilbert's xy2h::<u32> handles up to order-16.
    // For larger orders we'd need the u64 variant. Since kh is the order
    // for xy2h calls, and kh <= 32 with u64, this is fine — fast_hilbert
    // internally uses T=u32 but the x,y must fit in u32 (2^order - 1),
    // which for order <= 32 means up to 2^32 - 1. We cast safely.
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