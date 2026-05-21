//! Architectural-layout leaf tile renderer.
//!
//! Coexists with the byte-Hilbert renderers in `super::leaf`: same tile size,
//! same encoded output, same pyramid accumulator — only the per-pixel mapping
//! differs. Each pixel is one *element* of one tensor (1 px = 1 element),
//! decoded via the dtype's natural stride.
//!
//! Fetch policy: one coalesced byte-range request per (tensor, tile)
//! intersection, spanning `[first_row, last_row)` of the column slice
//! `[col_first, col_last)`. This includes the inter-row gaps (`cols -
//! region_width` extra bytes per row) — the bandwidth waste is bounded and
//! beats issuing 100s of small per-row requests over HTTP. For local mmap
//! sources the gap is a free `memcpy` slice.

use image::Rgb;

use crate::data::Data;
use crate::layout::arch::ArchLayout;
use crate::layout::render::{
    diff_element_color, plain_element_color, xet_dtype_element_color, xet_element_color,
    PADDING_RGB,
};
use crate::layout::TileRegion;
use crate::safetensors::DiffMetric;
use crate::tiled::leaf::{encode_tile, TileFormat, TILE};

type TileResult = Result<(image::ImageBuffer<Rgb<u8>, Vec<u8>>, Vec<u8>), String>;

/// Bytes loaded for one tile in architectural mode: one buffer per region.
#[derive(Default)]
pub struct LoadedArchTile {
    pub regions: Vec<(TileRegion, Vec<u8>)>,
}

/// Compute the absolute byte span of `region` inside the source. Includes
/// inter-row gaps so that one HTTP range covers all rows of the region.
fn region_byte_span(r: &TileRegion) -> (u64, usize) {
    let elem = r.dtype.element_size() as u64;
    let stride = r.tensor_cols * elem;
    let first = r.tensor_byte_start + r.row_first * stride + r.col_first * elem;
    // Last byte: end of element at (row_last-1, col_last-1).
    let last =
        r.tensor_byte_start + (r.row_last_exclusive - 1) * stride + r.col_last_exclusive * elem;
    debug_assert!(last >= first);
    (first, (last - first) as usize)
}

/// Async load stage for architectural mode: fetch one coalesced byte range
/// per region in this tile.
pub async fn load_arch_tile_regions(
    tx: u32,
    ty: u32,
    layout: &ArchLayout,
    source_data: &[Data],
    cumulative_offsets: &[u64],
) -> anyhow::Result<LoadedArchTile> {
    let mut out = LoadedArchTile::default();
    let regions = layout.regions_in_tile(tx, ty);
    for region in regions {
        // `tensor_byte_start` is absolute across the concatenated source
        // stream; subtract the source's cumulative offset to get a local
        // file offset before issuing the per-source fetch.
        let src_off = cumulative_offsets
            .get(region.source_idx)
            .copied()
            .unwrap_or(0);
        let (abs_start, len) = region_byte_span(&region);
        let local_off = abs_start - src_off;
        let bytes = source_data[region.source_idx]
            .fetch_range(local_off, len)
            .await?;
        out.regions.push((region, bytes));
    }
    let _ = (tx, ty); // keep params for symmetry with the byte-mode signature
    Ok(out)
}

/// Iterate every pixel in `region`'s tile-local rectangle, calling
/// `paint(rel_x, rel_y, elem_idx)` once per pixel. `elem_idx` is the offset
/// (in *elements*, not bytes) into `bytes` where this pixel's element starts.
#[inline]
fn iter_region_pixels(region: &TileRegion, mut paint: impl FnMut(u32, u32, usize)) {
    let cols = region.tensor_cols;
    for py in region.tile_y0..region.tile_y1 {
        let dy = (py - region.tile_y0) as u64;
        let row_idx = dy * (region.col_last_exclusive - region.col_first);
        let _ = row_idx; // not actually used; we use cols-based stride below
        for px in region.tile_x0..region.tile_x1 {
            let dx = (px - region.tile_x0) as u64;
            // Element offset from the start of the fetched buffer:
            //   (dr * cols + dc) where dr = dy, dc = (col_first + dx) - col_first = dx.
            // BUT we also have to account for col_first ≠ 0 — the buffer
            // starts at element (row_first, col_first). Element (row_first+dy,
            // col_first+dx) lives at offset (dy*cols + dx + 0) elements past
            // the buffer start: dy*cols accounts for the gap, dx for the col
            // within the row.
            let elem_off = (dy * cols + dx) as usize;
            paint(px, py, elem_off);
        }
    }
}

fn blank_tile() -> image::ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(TILE, TILE);
    for p in img.pixels_mut() {
        *p = PADDING_RGB;
    }
    img
}

/// Plain-mode (single source, byte-value coloring via pixel_lut).
pub fn render_arch_tile_plain(
    tile: &LoadedArchTile,
    pixel_lut: &[Rgb<u8>; 256],
    fmt: TileFormat,
) -> TileResult {
    let mut img = blank_tile();
    for (region, bytes) in &tile.regions {
        let dtype = region.dtype;
        iter_region_pixels(region, |px, py, elem_off| {
            let color = plain_element_color(dtype, bytes, elem_off, pixel_lut);
            img.put_pixel(px, py, color);
        });
    }
    encode_tile(img, fmt)
}

/// Dtype-mode (single source, position-based dtype color, no byte read).
///
/// For architectural mode this is just "paint each region with its dtype
/// color flat" — there are no inter-tensor bytes, the gaps are padding.
pub fn render_arch_tile_dtype(tile: &LoadedArchTile, fmt: TileFormat) -> TileResult {
    let mut img = blank_tile();
    for (region, _bytes) in &tile.regions {
        let color = region.dtype.to_color();
        for py in region.tile_y0..region.tile_y1 {
            for px in region.tile_x0..region.tile_x1 {
                img.put_pixel(px, py, color);
            }
        }
    }
    encode_tile(img, fmt)
}

/// Diff-mode. Each region carries paired byte ranges — fetched in
/// architectural mode via twin layouts on the two sources. For v1, we
/// piggyback on the existing `TensorDiff` source kind so the underlying
/// `bytes` is already a difference-encoded byte stream (output of
/// `diff_to_u8`). Reads still go through `pixel_lut`.
pub fn render_arch_tile_diff(
    tile: &LoadedArchTile,
    pixel_lut: &[Rgb<u8>; 256],
    fmt: TileFormat,
) -> TileResult {
    let mut img = blank_tile();
    for (region, bytes) in &tile.regions {
        let dtype = region.dtype;
        iter_region_pixels(region, |px, py, elem_off| {
            // `TensorDiff` produces one byte per *element pair* (not per
            // element of the source dtype). So when this region comes from a
            // TensorDiff source, the per-pixel byte index is `elem_off` —
            // which is already correct because `region.dtype.element_size()`
            // for a diff buffer is 1.
            //
            // For non-diff regions inside a diff run (e.g. unmatched
            // crosshatch fills handled via DiffFill), we still want the LUT
            // path: take the proxy byte and route through the diff pixel LUT
            // (which is identical to plain when the byte is 127).
            let elem = dtype.element_size();
            let byte = if elem == 1 {
                bytes.get(elem_off).copied().unwrap_or(127)
            } else {
                // diff_to_u8 always produces u8 output; if a non-diff region
                // sneaks in here just paint via the plain-element proxy.
                let plain = plain_element_color(dtype, bytes, elem_off, pixel_lut);
                img.put_pixel(px, py, plain);
                return;
            };
            img.put_pixel(px, py, pixel_lut[byte as usize]);
        });
    }
    encode_tile(img, fmt)
}

/// Xet (plain) mode — byte intensity × xorb tableau color.
pub fn render_arch_tile_xet(
    tile: &LoadedArchTile,
    pixel_lut: &[Rgb<u8>; 256],
    xorb_ranges: &[(u64, u64, u8)],
    tableau: &[Rgb<u8>; 20],
    fmt: TileFormat,
) -> TileResult {
    let mut img = blank_tile();
    for (region, bytes) in &tile.regions {
        let dtype = region.dtype;
        let tbs = region.tensor_byte_start
            + region.row_first * region.tensor_cols * dtype.element_size() as u64
            + region.col_first * dtype.element_size() as u64;
        iter_region_pixels(region, |px, py, elem_off| {
            let color =
                xet_element_color(dtype, bytes, elem_off, tbs, xorb_ranges, tableau, pixel_lut);
            img.put_pixel(px, py, color);
        });
    }
    encode_tile(img, fmt)
}

/// Combined xet+safetensors mode — dtype hue × xorb tableau hue, modulated
/// by per-element intensity proxy.
pub fn render_arch_tile_xet_dtype(
    tile: &LoadedArchTile,
    xorb_ranges: &[(u64, u64, u8)],
    tableau: &[Rgb<u8>; 20],
    fmt: TileFormat,
) -> TileResult {
    let mut img = blank_tile();
    for (region, bytes) in &tile.regions {
        let dtype = region.dtype;
        let tbs = region.tensor_byte_start
            + region.row_first * region.tensor_cols * dtype.element_size() as u64
            + region.col_first * dtype.element_size() as u64;
        iter_region_pixels(region, |px, py, elem_off| {
            let color = xet_dtype_element_color(dtype, bytes, elem_off, tbs, xorb_ranges, tableau);
            img.put_pixel(px, py, color);
        });
    }
    encode_tile(img, fmt)
}

/// Element-aware diff render when paired byte ranges are available on both
/// sides. Currently unused — the v1 wiring routes diff through TensorDiff
/// sources, which already produce per-element bytes. Kept exported for the
/// follow-up that pairs two layouts directly.
#[allow(dead_code)]
pub fn render_arch_tile_diff_paired(
    tile_a: &LoadedArchTile,
    tile_b: &LoadedArchTile,
    metric: DiffMetric,
    pixel_lut: &[Rgb<u8>; 256],
    fmt: TileFormat,
) -> TileResult {
    let mut img = blank_tile();
    // Pair regions by tensor_id; assume parallel layouts (same canvas, same
    // tensor placement). Mismatches fall back to padding.
    let mut by_id_b: std::collections::HashMap<usize, &(TileRegion, Vec<u8>)> =
        std::collections::HashMap::new();
    for r in &tile_b.regions {
        by_id_b.insert(r.0.tensor_id, r);
    }

    for (region_a, bytes_a) in &tile_a.regions {
        let Some((region_b, bytes_b)) = by_id_b.get(&region_a.tensor_id) else {
            continue;
        };
        let dtype = region_a.dtype;
        let dtype_b = region_b.dtype;
        // Per-tensor scale is unknown at this layer in v1; pass 0 → RMS path
        // falls back to RMS_FLOOR.
        let scale_orig = 0.0f32;
        iter_region_pixels(region_a, |px, py, elem_off| {
            let color = diff_element_color(
                dtype, bytes_a, elem_off, dtype_b, bytes_b, elem_off, metric, scale_orig, pixel_lut,
            );
            img.put_pixel(px, py, color);
        });
    }
    encode_tile(img, fmt)
}
