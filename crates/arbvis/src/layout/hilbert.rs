//! Thin wrapper around the legacy byte-Hilbert geometry. Mirrors what
//! `tiled::mod::build_tile_plan` was computing inline. Owning this in a
//! struct lets the renderer take `&Layout` everywhere instead of threading
//! `(kh, height_tiles, square_pixels, total)` through ten function signatures.

use crate::tiled::leaf::{TILE, TILE_LOG2};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct HilbertLayout {
    pub kh: u8,
    pub width_tiles: u32,
    pub height_tiles: u32,
    pub world_w: u32,
    pub height: u32,
    pub max_zoom: u32,
    pub total_tiles: u64,
    pub square_pixels: u64,
    /// Total bytes the curve covers. Pixels with index `>= total` paint black.
    pub total: u64,
}

impl HilbertLayout {
    /// Compute the canvas dimensions for `total` bytes laid out via a
    /// 1px-per-byte square-tiled Hilbert curve, matching the formula in the
    /// previous `build_tile_plan`.
    pub fn from_total(total: u64) -> Self {
        let mut s = 2 * TILE_LOG2 as u32;
        while (1u64 << s) < total.max(1) {
            s += 1;
        }
        let kh = (s / 2) as u8;
        let kw = s.div_ceil(2) as u8;
        let height = 1u32 << kh;
        let width = 1u32 << kw;
        let tile_size = TILE;
        let max_zoom = kh as u32 - TILE_LOG2 as u32;
        let width_tiles = width / tile_size;
        let height_tiles = height / tile_size;
        let world_w = TILE << (kw as u32 - kh as u32);
        let square_pixels: u64 = (height as u64) * (height as u64);
        Self {
            kh,
            width_tiles,
            height_tiles,
            world_w,
            height,
            max_zoom,
            total_tiles: width_tiles as u64 * height_tiles as u64,
            square_pixels,
            total,
        }
    }
}
