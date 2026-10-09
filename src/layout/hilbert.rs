//! Thin wrapper around the legacy byte-Hilbert geometry. Mirrors what
//! `tiled::mod::build_tile_plan` was computing inline. Owning this in a
//! struct lets the renderer take `&Layout` everywhere instead of threading
//! `(kh, height_tiles, square_pixels, total)` through ten function signatures.

use crate::tiled::leaf::{TILE, TILE_LOG2};

/// The Hilbert canvas for `total` bytes at 1 px/byte: the smallest power-of-two
/// canvas with side `2^s` (starting at `2^(2·TILE_LOG2)`) split into a
/// `(1<<kw) × (1<<kh)` rectangle with `kw = s.div_ceil(2)`, `kh = s / 2`.
/// `square_pixels` is the side of one Hilbert square (`height²`).
/// Shared by [`HilbertLayout::from_total`], `tiled::build_tile_plan`, and
/// `tiled::single_geometry`, which each derived this inline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanvasGeom {
    pub kw: u8,
    pub kh: u8,
    pub width: u32,
    pub height: u32,
    pub square_pixels: u64,
}

/// Compute the [`CanvasGeom`] for `total` bytes. A zero total yields the
/// smallest canvas (`s = 2·TILE_LOG2`), since `1 << s` always covers it.
pub fn hilbert_canvas(total: u64) -> CanvasGeom {
    let mut s = 2 * TILE_LOG2 as u32;
    while (1u64 << s) < total {
        s += 1;
    }
    let kh = s / 2;
    let kw = s.div_ceil(2);
    let height = 1u32 << kh;
    let width = 1u32 << kw;
    CanvasGeom {
        kh: kh as u8,
        kw: kw as u8,
        width,
        height,
        square_pixels: (height as u64) * (height as u64),
    }
}

/// Geometry knobs for the byte-Hilbert canvas: the curve order, tile grid
/// dimensions, world size, and byte budget. Computed once by
/// [`HilbertLayout::from_total`] and consumed by the tile pipeline.
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
        let g = hilbert_canvas(total);
        let CanvasGeom {
            kw,
            kh,
            width,
            height,
            square_pixels,
            ..
        } = g;
        let tile_size = TILE;
        let max_zoom = kh as u32 - TILE_LOG2 as u32;
        let width_tiles = width / tile_size;
        let height_tiles = height / tile_size;
        let world_w = TILE << (kw as u32 - kh as u32);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smallest_canvas_fits_default_square() {
        // Anything up to the initial 2^18 canvas keeps the base 512x512
        // single-tile square: s stays at 2*TILE_LOG2 = 18.
        for total in [0u64, 1, 2, 262_143, 262_144] {
            let l = HilbertLayout::from_total(total);
            assert_eq!(l.kh, 9);
            assert_eq!(l.width_tiles, 1);
            assert_eq!(l.height_tiles, 1);
            assert_eq!(l.height, 512);
            assert_eq!(l.world_w, 512);
            assert_eq!(l.max_zoom, 0);
            assert_eq!(l.total_tiles, 1);
            assert_eq!(l.square_pixels, 512 * 512);
            assert_eq!(l.total, total);
        }
    }

    #[test]
    fn exact_power_of_two_boundary_grows_canvas() {
        // One byte past the base square forces s to 19: an odd exponent,
        // so the curve is one tile wider than it is tall.
        let l = HilbertLayout::from_total(262_145);
        assert_eq!(l.kh, 9);
        assert_eq!(l.height, 512);
        assert_eq!(l.width_tiles, 2);
        assert_eq!(l.height_tiles, 1);
        assert_eq!(l.total_tiles, 2);
        assert_eq!(l.world_w, 1024);
        assert_eq!(l.max_zoom, 0);
        assert_eq!(l.square_pixels, 512 * 512);
    }

    #[test]
    fn even_exponent_keeps_square_and_grows_zoom() {
        // 2^20 bytes: s = 20, a perfect square canvas two tiles per side.
        let l = HilbertLayout::from_total(1 << 20);
        assert_eq!(l.kh, 10);
        assert_eq!(l.width_tiles, 2);
        assert_eq!(l.height_tiles, 2);
        assert_eq!(l.height, 1024);
        assert_eq!(l.world_w, 512);
        assert_eq!(l.max_zoom, 1);
        assert_eq!(l.total_tiles, 4);
        assert_eq!(l.square_pixels, 1024 * 1024);
    }

    #[test]
    fn total_field_round_trips() {
        for total in [1u64, 500_000, 1 << 40] {
            assert_eq!(HilbertLayout::from_total(total).total, total);
        }
    }
}
