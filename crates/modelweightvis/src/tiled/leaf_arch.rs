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

use arbvis::{encode_tile, Data, DiffMetric, TileFormat, TILE};

use crate::format::ElementStride;
use crate::layout::arch::ArchLayout;
use crate::layout::render::{
    diff_element_color, plain_element_color, xet_dtype_element_color, xet_element_color,
    PADDING_RGB,
};
use crate::layout::TileRegion;

type TileResult = Result<(image::ImageBuffer<Rgb<u8>, Vec<u8>>, Vec<u8>), String>;

/// Bytes loaded for one tile in architectural mode: one buffer per region.
#[derive(Default)]
pub struct LoadedArchTile {
    /// Per-region tuple: `(region, fetched_bytes, leading_elem_offset)`.
    /// `leading_elem_offset` is the element index inside `fetched_bytes`
    /// where this region's first painted pixel lives — always 0 for fixed-
    /// stride dtypes, non-zero for block-quantised tensors whose `col_first`
    /// doesn't fall on a block boundary.
    pub regions: Vec<(TileRegion, Vec<u8>, usize)>,
}

/// Compute the absolute byte span of `region` inside the source plus the
/// leading element offset inside the fetched buffer (non-zero only for
/// block-quantised dtypes whose `col_first` doesn't fall on a block
/// boundary). Includes inter-row gaps so that one HTTP range covers all
/// rows of the region.
fn region_byte_span(r: &TileRegion) -> (u64, usize, usize) {
    match r.dtype.stride() {
        ElementStride::Fixed(bpe) => {
            let elem = bpe as u64;
            let stride = r.tensor_cols * elem;
            let first = r.tensor_byte_start + r.row_first * stride + r.col_first * elem;
            // Last byte: end of element at (row_last-1, col_last-1).
            let last = r.tensor_byte_start
                + (r.row_last_exclusive - 1) * stride
                + r.col_last_exclusive * elem;
            debug_assert!(last >= first);
            (first, (last - first) as usize, 0)
        }
        ElementStride::Block {
            block_bytes,
            block_elements,
        } => {
            let be = block_elements as u64;
            let bb = block_bytes as u64;
            // Snap col_first down and col_last up to the enclosing block
            // boundaries so each row in the fetched buffer starts at a
            // block boundary (cols is a multiple of block_elements per
            // GGUF convention).
            let col_first_aligned = (r.col_first / be) * be;
            let col_last_aligned = r.col_last_exclusive.div_ceil(be) * be;
            let bytes_per_row = r.dtype.stride().bytes_per_row(r.tensor_cols);
            let first =
                r.tensor_byte_start + r.row_first * bytes_per_row + (col_first_aligned / be) * bb;
            let last = r.tensor_byte_start
                + (r.row_last_exclusive - 1) * bytes_per_row
                + (col_last_aligned / be) * bb;
            debug_assert!(last >= first);
            let leading_elems = (r.col_first - col_first_aligned) as usize;
            (first, (last - first) as usize, leading_elems)
        }
        // Packed-int dtypes: snap to packed-slot boundaries the same way
        // Block does to block boundaries. Each slot holds `elems_per_slot`
        // elements; `cols` is assumed to be a multiple of `elems_per_slot`
        // (true for canonical AWQ/GPTQ layouts with group_size ≥ elems_per_slot).
        ElementStride::Packed {
            bits,
            pack_dtype_bytes,
            ..
        } => {
            if bits == 0 {
                return (r.tensor_byte_start, 0, 0);
            }
            let elems_per_slot = ((pack_dtype_bytes as u64) * 8) / bits as u64;
            if elems_per_slot == 0 {
                return (r.tensor_byte_start, 0, 0);
            }
            let slot_bytes = pack_dtype_bytes as u64;
            let col_first_aligned = (r.col_first / elems_per_slot) * elems_per_slot;
            let col_last_aligned = r.col_last_exclusive.div_ceil(elems_per_slot) * elems_per_slot;
            let bytes_per_row = r.dtype.stride().bytes_per_row(r.tensor_cols);
            let first = r.tensor_byte_start
                + r.row_first * bytes_per_row
                + (col_first_aligned / elems_per_slot) * slot_bytes;
            let last = r.tensor_byte_start
                + (r.row_last_exclusive - 1) * bytes_per_row
                + (col_last_aligned / elems_per_slot) * slot_bytes;
            debug_assert!(last >= first);
            let leading_elems = (r.col_first - col_first_aligned) as usize;
            (first, (last - first) as usize, leading_elems)
        }
    }
}

/// Async load stage for architectural mode: fetch one coalesced byte range
/// per region in this tile.
pub async fn load_arch_tile_regions(
    zoom: u32,
    tx: u32,
    ty: u32,
    layout: &ArchLayout,
    source_data: &[Data],
    cumulative_offsets: &[u64],
) -> anyhow::Result<LoadedArchTile> {
    let mut out = LoadedArchTile::default();
    let regions = layout.regions_in_tile(zoom, tx, ty);
    for region in regions {
        // `tensor_byte_start` is absolute across the concatenated source
        // stream; subtract the source's cumulative offset to get a local
        // file offset before issuing the per-source fetch.
        let src_off = cumulative_offsets
            .get(region.source_idx)
            .copied()
            .unwrap_or(0);
        let (abs_start, len, leading) = region_byte_span(&region);
        let local_off = abs_start - src_off;
        let bytes = source_data[region.source_idx]
            .fetch_range(local_off, len)
            .await?;
        out.regions.push((region, bytes, leading));
    }
    let _ = (tx, ty); // keep params for symmetry with the byte-mode signature
    Ok(out)
}

/// Iterate every pixel in `region`'s tile-local rectangle, calling
/// `paint(rel_x, rel_y, elem_idx)` once per pixel. `elem_idx` is the offset
/// (in *elements*, not bytes) into the fetched `bytes` where this pixel's
/// element starts. `leading` is the element offset of the region's
/// `col_first` from the buffer start (zero for fixed-stride dtypes; the
/// distance from a block boundary for block-quantised dtypes).
///
/// The tensor is drawn at a display footprint that may differ from its element
/// grid (shrunk/enlarged, and × `2^(zoom-max_zoom)` at deeper zooms). Each
/// output pixel maps to an element by
/// `element = floor((samp + delta_px) * tensor_dim / footprint_dim)`:
///   * footprint > element grid (enlarge): consecutive pixels repeat an element
///     (replication).
///   * footprint < element grid (shrink): each pixel jumps several elements
///     (nearest-element subsample).
///   * equal (`scale == 1`, overview): the exact 1px=1element path.
///
/// The buffer is indexed `leading + row_rel * tensor_cols + col_rel`, where
/// `row_rel`/`col_rel` are element offsets from the region's `row_first`/
/// `col_first` anchor — identical to the 1:1 contract, just with the anchor and
/// per-pixel element index resampled.
#[inline]
fn iter_region_pixels(region: &TileRegion, leading: usize, mut paint: impl FnMut(u32, u32, usize)) {
    let cols = region.tensor_cols.max(1);
    let rows = region.tensor_rows.max(1);
    let fw = region.footprint_w.max(1);
    let fh = region.footprint_h.max(1);
    for py in region.tile_y0..region.tile_y1 {
        let dy = (py - region.tile_y0) as u64;
        let er = (region.samp_y0 + dy) * rows / fh; // absolute element row
        let row_rel = er - region.row_first;
        let row_base = leading + (row_rel * cols) as usize;
        for px in region.tile_x0..region.tile_x1 {
            let dx = (px - region.tile_x0) as u64;
            let ec = (region.samp_x0 + dx) * cols / fw; // absolute element col
            let col_rel = ec - region.col_first;
            paint(px, py, row_base + col_rel as usize);
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
    for (region, bytes, leading) in &tile.regions {
        let dtype = region.dtype;
        iter_region_pixels(region, *leading, |px, py, elem_off| {
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
    for (region, _bytes, _leading) in &tile.regions {
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
    for (region, bytes, leading) in &tile.regions {
        let dtype = region.dtype;
        iter_region_pixels(region, *leading, |px, py, elem_off| {
            // `TensorDiff` produces one byte per *element pair* (not per
            // element of the source dtype). For diff buffers `dtype` is U8
            // and `stride` is `Fixed(1)`, so the per-pixel byte index is
            // exactly `elem_off`. Non-diff regions inside a diff run fall
            // through to the plain-element path.
            let stride = dtype.stride();
            let byte = match stride {
                ElementStride::Fixed(1) => bytes.get(elem_off).copied().unwrap_or(127),
                _ => {
                    let plain = plain_element_color(dtype, bytes, elem_off, pixel_lut);
                    img.put_pixel(px, py, plain);
                    return;
                }
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
    for (region, bytes, leading) in &tile.regions {
        let dtype = region.dtype;
        // xet xorb coloring keys off absolute byte position. For fixed-stride
        // dtypes the byte address of element K is at a known offset; for
        // block-quantised dtypes we approximate by the block's start byte —
        // every element within one block shares the block's xorb hue.
        let tbs = region.tensor_byte_start
            + region.row_first * region.tensor_cols * dtype.element_size() as u64
            + region.col_first * dtype.element_size() as u64;
        iter_region_pixels(region, *leading, |px, py, elem_off| {
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
    for (region, bytes, leading) in &tile.regions {
        let dtype = region.dtype;
        let tbs = region.tensor_byte_start
            + region.row_first * region.tensor_cols * dtype.element_size() as u64
            + region.col_first * dtype.element_size() as u64;
        iter_region_pixels(region, *leading, |px, py, elem_off| {
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
    let mut by_id_b: std::collections::HashMap<usize, &(TileRegion, Vec<u8>, usize)> =
        std::collections::HashMap::new();
    for r in &tile_b.regions {
        by_id_b.insert(r.0.tensor_id, r);
    }

    for (region_a, bytes_a, leading_a) in &tile_a.regions {
        let Some((_region_b, bytes_b, leading_b)) = by_id_b.get(&region_a.tensor_id) else {
            continue;
        };
        let dtype = region_a.dtype;
        let dtype_b = _region_b.dtype;
        // Per-tensor scale is unknown at this layer in v1; pass 0 → RMS path
        // falls back to RMS_FLOOR.
        let scale_orig = 0.0f32;
        let leading_b = *leading_b;
        iter_region_pixels(region_a, *leading_a, |px, py, elem_off| {
            // Symmetric layouts give the same element offset on both sides
            // for fixed-stride dtypes; for block-stride we additionally
            // offset by side B's `leading` minus side A's so the matched
            // element pairs line up.
            let mod_off = elem_off + leading_b - *leading_a;
            let color = diff_element_color(
                dtype, bytes_a, elem_off, dtype_b, bytes_b, mod_off, metric, scale_orig, pixel_lut,
            );
            img.put_pixel(px, py, color);
        });
    }
    encode_tile(img, fmt)
}

#[cfg(test)]
mod region_byte_span_tests {
    use super::*;
    use crate::format::Dtype;

    fn region(dtype: Dtype, cols: u64, rf: u64, rl: u64, cf: u64, cl: u64, tbs: u64) -> TileRegion {
        TileRegion {
            source_idx: 0,
            tensor_id: 0,
            dtype,
            tensor_rows: 4096,
            tensor_cols: cols,
            row_first: rf,
            row_last_exclusive: rl,
            col_first: cf,
            col_last_exclusive: cl,
            tensor_byte_start: tbs,
            footprint_w: 1,
            footprint_h: 1,
            samp_x0: 0,
            samp_y0: 0,
            tile_x0: 0,
            tile_y0: 0,
            tile_x1: 1,
            tile_y1: 1,
        }
    }

    #[test]
    fn fixed_stride_exact_span() {
        // F32 (4 bytes), 100 cols, rows [2,4), cols [3,7), start 1000.
        let r = region(Dtype::F32, 100, 2, 4, 3, 7, 1000);
        let (first, len, leading) = region_byte_span(&r);
        let elem = 4u64;
        let stride = 100 * elem;
        assert_eq!(first, 1000 + 2 * stride + 3 * elem);
        // Last byte = end of element (row_last-1, col_last-1) == (row_last-1)*stride + col_last*elem.
        assert_eq!(len as u64, (1000 + 3 * stride + 7 * elem) - first);
        assert_eq!(leading, 0, "fixed stride has no leading block offset");
    }

    #[test]
    fn block_stride_snaps_to_block_boundary_and_reports_leading() {
        // Q4_0 is block-quantised; cols must be a multiple of block_elements.
        let (be, bb) = match Dtype::Q4_0.stride() {
            ElementStride::Block {
                block_elements,
                block_bytes,
            } => (block_elements as u64, block_bytes as u64),
            other => panic!("Q4_0 should be block stride, got {other:?}"),
        };
        let cols = 4 * be;
        let cf = be + 1; // deliberately NOT on a block boundary
        let cl = 2 * be + be / 2 + 1; // ends mid-block in a later block
        let r = region(Dtype::Q4_0, cols, 1, 3, cf, cl, 4096);
        let (first, len, leading) = region_byte_span(&r);

        let bytes_per_row = (cols / be) * bb;
        let cf_aligned = (cf / be) * be;
        let cl_aligned = cl.div_ceil(be) * be;
        assert_eq!(first, 4096 + bytes_per_row + (cf_aligned / be) * bb);
        let expected_last = 4096 + (3 - 1) * bytes_per_row + (cl_aligned / be) * bb;
        assert_eq!(len as u64, expected_last - first);
        // `first` lands exactly on a block boundary; leading is the element
        // distance from there to col_first.
        assert_eq!(leading as u64, cf - cf_aligned);
    }

    #[test]
    #[allow(clippy::identity_op)] // (rl-1) kept for structural parity with region_byte_span
    fn packed_stride_snaps_to_slot_boundary_and_reports_leading() {
        // Int4Packed packs 8 elements per int32 slot.
        let (eps, slot_bytes) = match Dtype::Int4Packed.stride() {
            ElementStride::Packed {
                bits,
                pack_dtype_bytes,
                ..
            } => (
                (pack_dtype_bytes as u64 * 8) / bits as u64,
                pack_dtype_bytes as u64,
            ),
            other => panic!("Int4Packed should be packed stride, got {other:?}"),
        };
        let cols = 4 * eps;
        let cf = eps + 2; // not on a slot boundary
        let cl = 2 * eps + 3;
        let r = region(Dtype::Int4Packed, cols, 0, 2, cf, cl, 0);
        let (first, len, leading) = region_byte_span(&r);

        let bytes_per_row = (cols / eps) * slot_bytes;
        let cf_aligned = (cf / eps) * eps;
        let cl_aligned = cl.div_ceil(eps) * eps;
        assert_eq!(first, (cf_aligned / eps) * slot_bytes);
        let expected_last = (2 - 1) * bytes_per_row + (cl_aligned / eps) * slot_bytes;
        assert_eq!(len as u64, expected_last - first);
        assert_eq!(leading as u64, cf - cf_aligned);
    }
}
