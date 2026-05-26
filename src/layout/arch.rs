//! Architectural (structure-aware) layout for safetensors checkpoints.
//!
//! Every tensor is placed at its natural 2D element shape, 1 pixel per
//! element. Transformer blocks (`{prefix}.layers.{N}.{sub_path}`) share a
//! single canonical arrangement: each block draws every sub-tensor in the
//! same relative position. Blocks are arranged in a grid of `cols` columns
//! chosen to keep the overall canvas near-square (see [`pick_column_count`]);
//! within a column, consecutive layers stay pixel-aligned column-for-column,
//! so q_proj at row 0 aligns with q_proj at row 1 etc. Across columns the
//! alignment is broken but the visualization is no longer absurdly tall.
//! Top-level tensors (embed_tokens, lm_head, norms) sit above and below the
//! block grid, centred horizontally.
//!
//! The output canvas is queryable per tile via [`ArchLayout::regions_in_tile`].

use std::collections::BTreeMap;

use crate::data::{Source, SourceKind, SourceMeta};
use crate::format::{Dtype, TensorMeta};
use crate::layout::bin_pack::{align_up, pack, Slot};
use crate::layout::name_tree::{self, LayerSlot};
use crate::layout::TileRegion;
use crate::tiled::leaf::TILE;

/// 8-px gutter between tensor slots inside a layer and between layer rows.
/// Keeps boundaries visually distinct without dominating the canvas.
const PAD: u32 = 8;

/// Maximum canvas width before bin-packing wraps a layer to a new shelf.
/// Sized to comfortably fit a llama-style block (concatenated q/k/v/o/MLP)
/// in one row while still pushing wider blocks to multi-line layouts.
const MAX_LAYER_WIDTH: u32 = 65_536;

/// One placed tensor in the architectural canvas.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PlacedTensor {
    pub source_idx: usize,
    pub tensor_id: usize,
    pub name: String,
    pub dtype: Dtype,
    pub tensor_byte_start: u64,
    /// `element_shape` = (rows, cols).
    pub tensor_rows: u64,
    pub tensor_cols: u64,
    /// Top-left of the tensor's element-pixel rectangle on the canvas.
    pub canvas_x: u32,
    pub canvas_y: u32,
    /// Hue used for entity labelling.
    pub hue: u16,
    /// Stable id of the layer this tensor belongs to: `None` for top-level
    /// singletons; `Some(layer_idx)` for transformer-block tensors.
    pub layer_idx: Option<u32>,
}

/// One transformer block's bounding rectangle on the canvas. Drawn as a
/// single layer-granularity overlay polygon.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LayerBounds {
    pub layer_idx: u32,
    pub canvas_x: u32,
    pub canvas_y: u32,
    pub width: u32,
    pub height: u32,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct ArchLayout {
    pub width: u32,
    pub height: u32,
    pub width_tiles: u32,
    pub height_tiles: u32,
    pub total_tiles: u64,
    pub max_zoom: u32,
    pub tensors: Vec<PlacedTensor>,
    pub layer_bounds: Vec<LayerBounds>,
    /// Architecture description from any `config.json` we found
    /// (e.g. `"LlamaForCausalLM (32 layers, hidden=4096)"`). Empty when no
    /// sidecar config was loaded.
    pub architecture: String,
    /// Tensor placements sorted by canvas_y then canvas_x — for fast
    /// `regions_in_tile` overlap queries via binary search.
    sorted_idx: Vec<usize>,
}

impl ArchLayout {
    /// Build an architectural layout. Returns `None` if the inputs can't be
    /// shape-mapped (e.g. zero tensors after merging sources).
    ///
    /// `metas` (parallel to `sources`, possibly shorter or empty) carries
    /// opportunistically-loaded `config.json` / `model.safetensors.index.json`
    /// data. When present these are used to (1) extend the layer stack to
    /// `num_hidden_layers` so partial-shard loads still produce a stable
    /// layout, (2) extend the canonical sub-path set with names from the
    /// index, and (3) record an architecture string for downstream display.
    pub fn try_build(
        sources: &[Source],
        cumulative_offsets: &[u64],
        metas: &[SourceMeta],
    ) -> Option<Self> {
        // Collect (source_idx, tensor_idx_in_source, tensor) tuples. Skip
        // UnmatchedRegion sources for now — they're already drawn via the
        // existing `DiffFill` crosshatch overlay path; the new layout slots
        // those into padding instead.
        let mut all: Vec<(usize, &TensorMeta, u64)> = Vec::new();
        for (sidx, s) in sources.iter().enumerate() {
            if matches!(s.kind, SourceKind::UnmatchedRegion { .. }) {
                continue;
            }
            let Some(st) = s.model_info.as_ref() else {
                continue;
            };
            let off = cumulative_offsets.get(sidx).copied().unwrap_or(0);
            for t in &st.tensors {
                all.push((sidx, t, off));
            }
        }
        if all.is_empty() {
            return None;
        }

        // Pick the first non-empty config across sources. Most multi-shard
        // checkpoints have one shared config.json next to all shards, so
        // we take any one — a discrepancy across them would surface as
        // mismatched tensor counts anyway.
        let pinned_config = metas.iter().find_map(|m| m.config.as_ref());
        let architecture = pinned_config.map(|c| c.summary()).unwrap_or_default();

        // Classify every tensor by name.
        let names: Vec<&str> = all.iter().map(|(_, t, _)| t.name.as_str()).collect();
        let profile = name_tree::classify(&names);

        // Group: each `layer_idx` -> { sub_path -> (source_idx, tensor, abs_byte_start) }.
        // Top-level singletons collect into `top_level`.
        let mut blocks: BTreeMap<u32, BTreeMap<String, (usize, &TensorMeta, u64)>> =
            BTreeMap::new();
        let mut top_level: Vec<(usize, &TensorMeta, u64)> = Vec::new();

        for ((sidx, t, base_off), slot) in all.iter().zip(profile.slots.iter()) {
            match slot {
                LayerSlot::Block { idx, sub_path } => {
                    blocks
                        .entry(*idx)
                        .or_default()
                        .insert(sub_path.clone(), (*sidx, *t, *base_off));
                }
                LayerSlot::TopLevel { .. } | LayerSlot::Generic { .. } => {
                    top_level.push((*sidx, *t, *base_off));
                }
            }
        }

        // If config.json gives us a definitive `num_hidden_layers`, validate
        // against the observed max and extend the canonical stack. Missing
        // layers stay empty in `blocks` and render as a row of padding (the
        // canonical slot positions are reserved but no tensors fill them),
        // which keeps the layout stable across partial-shard loads and diff
        // pairs that loaded different shard subsets.
        let observed_max = blocks.keys().copied().max();
        if let (Some(c), Some(observed)) = (pinned_config, observed_max) {
            if let Some(n) = c.num_hidden_layers {
                if n > 0 && n != observed + 1 {
                    if n > observed + 1 {
                        log::info!(
                            "arch layout: config.json reports {n} layers but only {} were loaded — extending layout to cover all {n}",
                            observed + 1,
                        );
                        for missing in 0..n {
                            blocks.entry(missing).or_default();
                        }
                    } else {
                        log::warn!(
                            "arch layout: config.json reports {n} layers but {} were observed in the input — keeping the larger of the two",
                            observed + 1,
                        );
                    }
                }
            }
        }

        // Compute the canonical layer arrangement: the union of every block's
        // sub_paths. For diff alignment, every layer slot is identical even
        // if some layers are missing a particular sub-tensor (renders as
        // padding in that slot).
        let mut canonical_subpaths: Vec<String> = {
            use std::collections::BTreeSet;
            let mut set: BTreeSet<String> = BTreeSet::new();
            for sub in blocks.values() {
                for k in sub.keys() {
                    set.insert(k.clone());
                }
            }
            // Also seed from any safetensors.index.json: tensor names listed
            // there but not loaded (different shard) become canonical slots
            // too. They have no dtype/shape so they only reserve a slot in
            // every layer; the actual rendered region is padding.
            for m in metas {
                if let Some(idx) = m.index.as_ref() {
                    for name in idx.weight_map.keys() {
                        if let Some(sub) = extract_block_sub_path(name) {
                            set.insert(sub);
                        }
                    }
                }
            }
            set.into_iter().collect()
        };
        // Sort sub-paths by a transformer-aware key so the canonical
        // arrangement reads attention-first, MLP-second, norms-last.
        canonical_subpaths.sort_by_key(|s| sub_path_order_key(s));

        // For each sub-path, the slot size is the max (rows, cols) across all
        // layers — guarantees every layer's slot fits. Round up to a 16-px
        // alignment so adjacent layers' grid lines stay aligned even when
        // dimensions differ subtly.
        let canon_slots: Vec<(String, Slot)> = canonical_subpaths
            .iter()
            .map(|sp| {
                let mut max_w: u64 = 1;
                let mut max_h: u64 = 1;
                for sub in blocks.values() {
                    if let Some((_, t, _)) = sub.get(sp) {
                        let (r, c) = t.element_shape();
                        if r > max_h {
                            max_h = r;
                        }
                        if c > max_w {
                            max_w = c;
                        }
                    }
                }
                let slot = Slot {
                    width: align_up(max_w.min(u32::MAX as u64) as u32, 16),
                    height: align_up(max_h.min(u32::MAX as u64) as u32, 16),
                };
                (sp.clone(), slot)
            })
            .collect();

        // Pack the canonical slots into one shelf (single layer's layout).
        let (placements, layer_w, layer_h) = pack(
            &canon_slots.iter().map(|(_, s)| *s).collect::<Vec<_>>(),
            MAX_LAYER_WIDTH,
            PAD,
        );

        // Pick a column count for the transformer-block grid so the canvas
        // ends up roughly square. `n_blocks == 0` (degenerate, no layers) and
        // `n_blocks == 1` collapse to a single-column layout.
        let n_blocks = blocks.len() as u32;
        let cols = pick_column_count(n_blocks, layer_w, layer_h, PAD);
        let grid_w = if cols == 0 {
            0
        } else {
            cols.saturating_mul(layer_w)
                .saturating_add(cols.saturating_sub(1).saturating_mul(PAD))
        };

        // Decide the canvas width: max of the grid width and the widest
        // top-level tensor.
        let top_widths: Vec<u32> = top_level
            .iter()
            .map(|(_, t, _)| {
                let (_, c) = t.element_shape();
                c.min(u32::MAX as u64) as u32
            })
            .collect();
        let canvas_w = grid_w
            .max(top_widths.iter().copied().max().unwrap_or(0))
            .max(1);

        // Lay tensors out top-to-bottom:
        //   1. top-level tensors classified as "input-side" (embedding-like names): centred at top
        //   2. transformer blocks in layer-index order (using the canonical arrangement)
        //   3. top-level tensors classified as "output-side" (lm_head / final-norm-like): centred at bottom
        let (top_inputs, top_outputs): (Vec<_>, Vec<_>) = top_level
            .iter()
            .copied()
            .partition(|(_, t, _)| is_input_side_name(&t.name));

        let mut tensors: Vec<PlacedTensor> = Vec::new();
        let mut layer_bounds: Vec<LayerBounds> = Vec::new();
        let mut cursor_y: u32 = 0;

        // 1. Input-side top-levels.
        for (sidx, t, base_off) in &top_inputs {
            let (rows, cols) = t.element_shape();
            let w = cols.min(u32::MAX as u64) as u32;
            let h = rows.min(u32::MAX as u64) as u32;
            let cx = canvas_w.saturating_sub(w) / 2;
            tensors.push(PlacedTensor {
                source_idx: *sidx,
                tensor_id: 0, // filled in below
                name: t.name.clone(),
                dtype: t.dtype,
                tensor_byte_start: base_off + t.file_start,
                tensor_rows: rows,
                tensor_cols: cols,
                canvas_x: cx,
                canvas_y: cursor_y,
                hue: name_hue_short(&t.name),
                layer_idx: None,
            });
            cursor_y = cursor_y.saturating_add(h).saturating_add(PAD);
        }

        // 2. Transformer blocks arranged in a `cols`-wide grid. The grid is
        // centred horizontally inside `canvas_w` so a wider top-level tensor
        // (e.g. lm_head spanning more cols than the block grid) doesn't push
        // the grid off-centre.
        let grid_x_offset = canvas_w.saturating_sub(grid_w) / 2;
        let grid_y0 = cursor_y;
        let grid_rows = if cols == 0 {
            0
        } else {
            n_blocks.div_ceil(cols)
        };
        for (block_pos, (idx, sub_map)) in blocks.iter().enumerate() {
            let pos = block_pos as u32;
            let (col, row) = if cols == 0 {
                (0, 0)
            } else {
                (pos % cols, pos / cols)
            };
            let block_x =
                grid_x_offset.saturating_add(col.saturating_mul(layer_w.saturating_add(PAD)));
            let block_y = grid_y0.saturating_add(row.saturating_mul(layer_h.saturating_add(PAD)));
            for ((sp, _), pl) in canon_slots.iter().zip(placements.iter()) {
                if let Some((sidx, t, base_off)) = sub_map.get(sp) {
                    let (rows, tcols) = t.element_shape();
                    let cx = block_x.saturating_add(pl.x);
                    let cy = block_y.saturating_add(pl.y);
                    tensors.push(PlacedTensor {
                        source_idx: *sidx,
                        tensor_id: 0,
                        name: t.name.clone(),
                        dtype: t.dtype,
                        tensor_byte_start: base_off + t.file_start,
                        tensor_rows: rows,
                        tensor_cols: tcols,
                        canvas_x: cx,
                        canvas_y: cy,
                        hue: name_hue_short(sp),
                        layer_idx: Some(*idx),
                    });
                }
            }
            layer_bounds.push(LayerBounds {
                layer_idx: *idx,
                canvas_x: block_x,
                canvas_y: block_y,
                width: layer_w,
                height: layer_h,
            });
        }
        if grid_rows > 0 {
            cursor_y =
                grid_y0.saturating_add(grid_rows.saturating_mul(layer_h.saturating_add(PAD)));
        }

        // 3. Output-side top-levels.
        for (sidx, t, base_off) in &top_outputs {
            let (rows, cols) = t.element_shape();
            let w = cols.min(u32::MAX as u64) as u32;
            let h = rows.min(u32::MAX as u64) as u32;
            let cx = canvas_w.saturating_sub(w) / 2;
            tensors.push(PlacedTensor {
                source_idx: *sidx,
                tensor_id: 0,
                name: t.name.clone(),
                dtype: t.dtype,
                tensor_byte_start: base_off + t.file_start,
                tensor_rows: rows,
                tensor_cols: cols,
                canvas_x: cx,
                canvas_y: cursor_y,
                hue: name_hue_short(&t.name),
                layer_idx: None,
            });
            cursor_y = cursor_y.saturating_add(h).saturating_add(PAD);
        }

        // Assign tensor_ids in canvas order.
        for (i, t) in tensors.iter_mut().enumerate() {
            t.tensor_id = i;
        }

        // Round canvas dimensions UP to tile-size multiples so the tile grid
        // covers all of the content rectangle…
        let raw_h = cursor_y.saturating_sub(PAD);
        let raw_canvas_h = align_up(raw_h.max(1), TILE);
        let raw_canvas_w = align_up(canvas_w, TILE);
        let raw_width_tiles = (raw_canvas_w / TILE).max(1);
        let raw_height_tiles = (raw_canvas_h / TILE).max(1);

        // …then pad both tile counts UP to powers of two. `PyramidAccumulator`
        // only emits a parent tile when all 4 of its children have contributed,
        // so any boundary cell whose 2×2 quad isn't fully populated stalls the
        // cascade — including, eventually, the zoom-0 root. Power-of-two grids
        // halve cleanly all the way down to (1, k) or (k, 1) tiles at zoom 0.
        // The padding cells render as PADDING_RGB (no tensor placements
        // intersect them), so the wasted bytes are tiny on disk.
        let width_tiles = next_pow2(raw_width_tiles);
        let height_tiles = next_pow2(raw_height_tiles);
        let canvas_w = width_tiles * TILE;
        let canvas_h = height_tiles * TILE;

        // Pyramid bottoms out when the *smaller* dimension hits 1 tile —
        // halving any further would produce fractional counts and the same
        // count==4 stall.  At zoom 0 the layout is
        //   (width_tiles / 2^max_zoom) × (height_tiles / 2^max_zoom)
        // tiles, exactly one of which equals 1.
        let max_zoom = (width_tiles.min(height_tiles).max(1) as f64).log2().round() as u32;

        let mut sorted_idx: Vec<usize> = (0..tensors.len()).collect();
        sorted_idx.sort_by_key(|&i| {
            let t = &tensors[i];
            (t.canvas_y, t.canvas_x)
        });

        Some(Self {
            width: canvas_w,
            height: canvas_h,
            width_tiles,
            height_tiles,
            total_tiles: width_tiles as u64 * height_tiles as u64,
            max_zoom,
            tensors,
            layer_bounds,
            architecture,
            sorted_idx,
        })
    }

    /// All tensor regions that overlap the tile at `(tx, ty)`.
    ///
    /// O(n) scan — fine because the architectural canvas typically holds
    /// O(hundreds) of tensors and tile rendering is the dominant cost
    /// downstream anyway. If this ever becomes a hot path, swap for an
    /// interval-tree on `canvas_y`.
    pub fn regions_in_tile(&self, tx: u32, ty: u32) -> Vec<TileRegion> {
        let tile_x0 = tx * TILE;
        let tile_y0 = ty * TILE;
        let tile_x1 = tile_x0 + TILE;
        let tile_y1 = tile_y0 + TILE;

        let mut out = Vec::new();
        for &i in &self.sorted_idx {
            let t = &self.tensors[i];
            let tw = t.tensor_cols.min(u32::MAX as u64) as u32;
            let th = t.tensor_rows.min(u32::MAX as u64) as u32;
            let tx0 = t.canvas_x;
            let ty0 = t.canvas_y;
            let tx1 = tx0.saturating_add(tw);
            let ty1 = ty0.saturating_add(th);

            // Early skip — once a tensor's top edge is past the tile bottom,
            // every subsequent tensor in sorted order is too (we sorted by y).
            if ty0 >= tile_y1 {
                break;
            }
            if tx1 <= tile_x0 || tx0 >= tile_x1 {
                continue;
            }
            if ty1 <= tile_y0 {
                continue;
            }

            let ix0 = tx0.max(tile_x0);
            let iy0 = ty0.max(tile_y0);
            let ix1 = tx1.min(tile_x1);
            let iy1 = ty1.min(tile_y1);

            let col_first = (ix0 - tx0) as u64;
            let col_last = (ix1 - tx0) as u64;
            let row_first = (iy0 - ty0) as u64;
            let row_last = (iy1 - ty0) as u64;

            out.push(TileRegion {
                source_idx: t.source_idx,
                tensor_id: t.tensor_id,
                dtype: t.dtype,
                tensor_rows: t.tensor_rows,
                tensor_cols: t.tensor_cols,
                row_first,
                row_last_exclusive: row_last,
                col_first,
                col_last_exclusive: col_last,
                tensor_byte_start: t.tensor_byte_start,
                tile_x0: ix0 - tile_x0,
                tile_y0: iy0 - tile_y0,
                tile_x1: ix1 - tile_x0,
                tile_y1: iy1 - tile_y0,
            });
        }
        out
    }
}

/// Order key for sub-paths within a layer: attention, then MLP, then norms,
/// then "other". Within attention, q/k/v/o; within MLP, gate/up/down.
fn sub_path_order_key(s: &str) -> (u8, u8, String) {
    let lower = s.to_lowercase();
    let group: u8 = if lower.contains("attn") || lower.contains("attention") {
        0
    } else if lower.contains("mlp")
        || lower.contains("feed_forward")
        || lower.contains("gate_proj")
        || lower.contains("up_proj")
        || lower.contains("down_proj")
    {
        1
    } else if lower.contains("norm") || lower.contains("ln_") {
        2
    } else {
        3
    };
    let sub_order: u8 = if lower.contains("q_proj") || lower.contains("query") {
        0
    } else if lower.contains("k_proj") || lower.contains("key") {
        1
    } else if lower.contains("v_proj") || lower.contains("value") {
        2
    } else if lower.contains("o_proj") || lower.contains("output") {
        3
    } else if lower.contains("gate_proj") {
        4
    } else if lower.contains("up_proj") {
        5
    } else if lower.contains("down_proj") {
        6
    } else {
        7
    };
    (group, sub_order, s.to_string())
}

fn is_input_side_name(name: &str) -> bool {
    let l = name.to_lowercase();
    l.contains("embed") || l.contains("wte") || l.contains("wpe")
}

/// Stable hue derived from a tensor's name (or sub-path). Different from
/// `geometry::name_hue` only in that it's u16 in [0, 360) and used for the
/// architectural layout's per-tensor entity hue.
/// Extract the in-layer sub-path for tensor names that match the
/// transformer-block pattern (e.g. `"model.layers.7.q_proj.weight"`
/// → `Some("q_proj.weight")`). Used to seed the canonical sub-path set
/// from `model.safetensors.index.json` entries that didn't get loaded.
fn extract_block_sub_path(name: &str) -> Option<String> {
    let caps = name_tree::block_regex_for_arch().captures(name)?;
    Some(caps.get(3)?.as_str().to_string())
}

/// Smallest power of two ≥ `n`. Returns 1 for `n == 0`.
fn next_pow2(n: u32) -> u32 {
    if n <= 1 {
        return 1;
    }
    1u32 << (32 - (n - 1).leading_zeros())
}

/// Pick a column count for arranging `n` transformer blocks in a grid so the
/// total grid width and height land as close to 1:1 as possible. Returns 0
/// when `n == 0` (no blocks to place) and 1 when `n == 1`.
///
/// Each candidate `c ∈ 1..=n` is scored by the absolute log-ratio of grid
/// width (`c * layer_w + (c-1) * gutter`) to grid height
/// (`ceil(n/c) * (layer_h + gutter)`). Ties broken in favour of the smaller
/// `c` so small models don't get fragmented across many narrow columns.
fn pick_column_count(n: u32, layer_w: u32, layer_h: u32, gutter: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    if n == 1 || layer_w == 0 || layer_h == 0 {
        return 1;
    }
    let mut best_c: u32 = 1;
    let mut best_score = f64::INFINITY;
    for c in 1..=n {
        let rows = n.div_ceil(c);
        let total_w =
            (c as u64) * (layer_w as u64) + (c.saturating_sub(1) as u64) * (gutter as u64);
        let total_h = (rows as u64) * (layer_h as u64 + gutter as u64);
        if total_w == 0 || total_h == 0 {
            continue;
        }
        let score = (total_w as f64 / total_h as f64).log2().abs();
        if score < best_score {
            best_score = score;
            best_c = c;
        }
    }
    best_c
}

fn name_hue_short(name: &str) -> u16 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    name.hash(&mut h);
    (h.finish() % 360) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_pow2_basics() {
        assert_eq!(next_pow2(0), 1);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(4), 4);
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(17), 32);
        assert_eq!(next_pow2(100), 128);
    }

    #[test]
    fn pick_column_count_degenerate_and_trivial() {
        // No blocks → 0 columns.
        assert_eq!(pick_column_count(0, 1000, 1000, PAD), 0);
        // Single block → 1 column (no grid to balance).
        assert_eq!(pick_column_count(1, 1000, 1000, PAD), 1);
        // Defensive: zero-sized blocks fall back to a single column rather
        // than dividing by zero.
        assert_eq!(pick_column_count(5, 0, 100, PAD), 1);
        assert_eq!(pick_column_count(5, 100, 0, PAD), 1);
    }

    #[test]
    fn pick_column_count_two_square_blocks_prefers_two_cols() {
        // Two identically-shaped blocks: 1×2 stack is taller than wide
        // (height ≈ 2*W), 2×1 row is wider than tall (width ≈ 2*W); both
        // sit equally far from square *without* gutters, but the gutter row
        // tips it toward 2 columns.
        assert_eq!(pick_column_count(2, 1000, 1000, PAD), 2);
    }

    #[test]
    fn pick_column_count_tall_many_blocks() {
        // 70 transformer blocks where each block is much wider than tall.
        // 1 column would be 1× : 70× tall (catastrophically vertical);
        // ideal is around sqrt(70 * layer_h / layer_w) = sqrt(70 * 0.2) ≈ 3.7,
        // so 4 columns gets us a near-square grid.
        let c = pick_column_count(70, 10_000, 2_000, PAD);
        assert_eq!(
            c, 4,
            "70 blocks of 10000×2000 should pick 4 columns to balance aspect; got {c}",
        );
    }

    #[test]
    fn pick_column_count_wide_blocks_stays_at_one_col() {
        // When each block is itself much wider than tall, even four of them
        // stack into a roughly-square canvas in a single column — adding more
        // columns would create a strip far wider than tall.
        let c = pick_column_count(4, 4_000, 1_000, PAD);
        assert_eq!(c, 1, "wide blocks already balance horizontally; got {c}");
    }

    /// Build a synthetic tensor for testing. `name`, `shape`.
    fn mk_t(name: &str, shape: Vec<u64>) -> TensorMeta {
        let elem_size = 4u64; // f32
        let n: u64 = shape.iter().product();
        TensorMeta {
            name: name.to_string(),
            dtype: Dtype::F32,
            shape,
            file_start: 0,
            file_end: n * elem_size,
            packed_sidecars: None,
        }
    }

    #[test]
    fn element_shape_3d_collapses_last() {
        let t = mk_t("conv1d.weight", vec![6144, 1, 4]);
        let (r, c) = t.element_shape();
        assert_eq!((r, c), (6144, 4));
    }

    #[test]
    fn element_shape_1d_is_strip() {
        let t = mk_t("norm.weight", vec![4096]);
        let (r, c) = t.element_shape();
        assert_eq!((r, c), (1, 4096));
    }

    #[test]
    fn sub_path_order_attention_before_mlp() {
        let attn = sub_path_order_key("self_attn.q_proj.weight");
        let mlp = sub_path_order_key("mlp.gate_proj.weight");
        assert!(attn < mlp);
    }

    #[test]
    fn sub_path_order_qkvo() {
        let q = sub_path_order_key("self_attn.q_proj.weight");
        let k = sub_path_order_key("self_attn.k_proj.weight");
        let v = sub_path_order_key("self_attn.v_proj.weight");
        let o = sub_path_order_key("self_attn.o_proj.weight");
        assert!(q < k && k < v && v < o);
    }

    #[test]
    fn extract_block_sub_path_strips_prefix_and_index() {
        assert_eq!(
            extract_block_sub_path("model.layers.7.self_attn.q_proj.weight"),
            Some("self_attn.q_proj.weight".to_string()),
        );
        assert_eq!(
            extract_block_sub_path("transformer.h.0.attn.c_attn.weight"),
            Some("attn.c_attn.weight".to_string()),
        );
        assert_eq!(extract_block_sub_path("lm_head.weight"), None);
    }

    /// Build a `Source` whose model header reports the listed tensors with
    /// sequential byte offsets. The Source's `kind` is `Buffered(empty)`
    /// only so we don't need a real file on disk; `ArchLayout::try_build`
    /// actually looks only at `model_info`.
    fn synthetic_source(tensors: Vec<TensorMeta>) -> Source {
        let total: u64 = tensors.iter().map(|t| t.file_end - t.file_start).sum();
        Source {
            file_idx: 0,
            kind: SourceKind::Buffered(Vec::new()),
            byte_size: total,
            model_info: Some(crate::format::ModelInfo {
                format: crate::format::SourceFormat::Safetensors,
                tensors,
                color_ranges: Vec::new(),
            }),
            name_override: None,
            xet_terms: None,
        }
    }

    #[test]
    fn config_extends_layer_stack_for_partial_shard() {
        // Simulate loading 4 of 8 transformer layers (shards split half/half).
        // Without config: layout shows 4 layer-bounds. With config that
        // declares 8 hidden layers, the layout should expose 8.
        let mut tensors: Vec<TensorMeta> = Vec::new();
        let mut off: u64 = 1024;
        for i in 0..4u64 {
            let n_elem = 64u64;
            let bytes = n_elem * 4;
            tensors.push(TensorMeta {
                name: format!("model.layers.{i}.self_attn.q_proj.weight"),
                dtype: Dtype::F32,
                shape: vec![8, 8],
                file_start: off,
                file_end: off + bytes,
                packed_sidecars: None,
            });
            off += bytes;
        }
        let source = synthetic_source(tensors);
        let cumulative = vec![0u64];

        // No config: 4 layer bounds.
        let no_config = ArchLayout::try_build(&[source], &cumulative, &[]).unwrap();
        assert_eq!(no_config.layer_bounds.len(), 4);
        assert!(no_config.architecture.is_empty());

        // With config (rebuild source since it was moved): 8 layer bounds.
        let mut tensors2: Vec<TensorMeta> = Vec::new();
        let mut off2: u64 = 1024;
        for i in 0..4u64 {
            let n_elem = 64u64;
            let bytes = n_elem * 4;
            tensors2.push(TensorMeta {
                name: format!("model.layers.{i}.self_attn.q_proj.weight"),
                dtype: Dtype::F32,
                shape: vec![8, 8],
                file_start: off2,
                file_end: off2 + bytes,
                packed_sidecars: None,
            });
            off2 += bytes;
        }
        let source2 = synthetic_source(tensors2);
        let config = crate::layout::model_config::ModelConfig {
            architectures: vec!["LlamaForCausalLM".to_string()],
            num_hidden_layers: Some(8),
            hidden_size: Some(8),
            ..Default::default()
        };
        let meta = SourceMeta {
            config: Some(config),
            index: None,
        };
        let with_config = ArchLayout::try_build(&[source2], &cumulative, &[meta]).unwrap();
        assert_eq!(with_config.layer_bounds.len(), 8);
        assert!(with_config.architecture.contains("LlamaForCausalLM"));
        assert!(with_config.architecture.contains("8 layers"));
    }

    /// `PyramidAccumulator` only emits a parent tile once all 4 children
    /// arrive, so canvas tile counts have to be powers of two on each axis
    /// AND `max_zoom = log2(min(w_p2, h_p2))` for the pyramid to drain all
    /// the way down to zoom 0. Without this, the leaflet viewer asks for
    /// `tiles/0/0/0.avif`, gets a 404, and renders a blank canvas — which is
    /// exactly the regression this test guards.
    #[test]
    fn canvas_dimensions_are_power_of_two() {
        // A non-square stack: 11 transformer layers, hidden=384, intermediate=1280.
        // That gives raw tile counts in the high-tens / low-hundreds — neither
        // a power of two on its own.
        let mut tensors: Vec<TensorMeta> = Vec::new();
        let mut off: u64 = 1024;
        for i in 0..11u64 {
            for (sub, shape) in [
                ("self_attn.q_proj.weight", vec![384, 384]),
                ("self_attn.k_proj.weight", vec![384, 384]),
                ("self_attn.v_proj.weight", vec![384, 384]),
                ("self_attn.o_proj.weight", vec![384, 384]),
                ("mlp.gate_proj.weight", vec![1280, 384]),
                ("mlp.up_proj.weight", vec![1280, 384]),
                ("mlp.down_proj.weight", vec![384, 1280]),
                ("input_layernorm.weight", vec![384]),
                ("post_attention_layernorm.weight", vec![384]),
            ] {
                let n: u64 = shape.iter().product();
                let bytes = n * 4;
                tensors.push(TensorMeta {
                    name: format!("model.layers.{i}.{sub}"),
                    dtype: Dtype::F32,
                    shape,
                    file_start: off,
                    file_end: off + bytes,
                    packed_sidecars: None,
                });
                off += bytes;
            }
        }
        let source = synthetic_source(tensors);
        let cumulative = vec![0u64];
        let layout = ArchLayout::try_build(&[source], &cumulative, &[]).unwrap();

        assert!(
            layout.width_tiles.is_power_of_two(),
            "width_tiles {} must be a power of two for the pyramid to drain",
            layout.width_tiles,
        );
        assert!(
            layout.height_tiles.is_power_of_two(),
            "height_tiles {} must be a power of two for the pyramid to drain",
            layout.height_tiles,
        );
        let smaller = layout.width_tiles.min(layout.height_tiles);
        assert_eq!(
            layout.max_zoom,
            smaller.trailing_zeros(),
            "max_zoom should be log2 of the smaller tile dim so 2^max_zoom divides both",
        );
        // Sanity: zoom 0 must have at least one tile in each dim, and exactly
        // one of the two should be exactly 1 (the smaller axis collapsed).
        let zoom0_w = layout.width_tiles >> layout.max_zoom;
        let zoom0_h = layout.height_tiles >> layout.max_zoom;
        assert!(zoom0_w >= 1 && zoom0_h >= 1);
        assert!(
            zoom0_w == 1 || zoom0_h == 1,
            "zoom 0 grid should be 1xN or Nx1, got {zoom0_w}x{zoom0_h}",
        );
    }

    /// With the multi-column grid, layers should arrange themselves into N×M
    /// blocks so that within a column corresponding sub-tensors stay
    /// pixel-aligned across rows (q_proj-in-layer-0 shares an x with
    /// q_proj-in-layer-cols, etc.). This is the alignment property the module
    /// comment promises — guards against regressions in the column/row math.
    #[test]
    fn multi_column_grid_preserves_within_column_alignment() {
        // 30 layers — same sub-tensors per layer (matches SmolLM2-135M's per-layer
        // shape). Picker chooses cols so the canvas trends toward near-square.
        let mut tensors: Vec<TensorMeta> = Vec::new();
        let mut off: u64 = 1024;
        for i in 0..30u64 {
            for (sub, shape) in [
                ("self_attn.q_proj.weight", vec![576, 576]),
                ("self_attn.k_proj.weight", vec![192, 576]),
                ("self_attn.v_proj.weight", vec![192, 576]),
                ("self_attn.o_proj.weight", vec![576, 576]),
                ("mlp.gate_proj.weight", vec![1536, 576]),
                ("mlp.up_proj.weight", vec![1536, 576]),
                ("mlp.down_proj.weight", vec![576, 1536]),
                ("input_layernorm.weight", vec![576]),
                ("post_attention_layernorm.weight", vec![576]),
            ] {
                let n: u64 = shape.iter().product();
                let bytes = n * 2; // BF16 is 2 bytes/elem
                tensors.push(TensorMeta {
                    name: format!("model.layers.{i}.{sub}"),
                    dtype: Dtype::BF16,
                    shape,
                    file_start: off,
                    file_end: off + bytes,
                    packed_sidecars: None,
                });
                off += bytes;
            }
        }
        let source = synthetic_source(tensors);
        let cumulative = vec![0u64];
        let layout = ArchLayout::try_build(&[source], &cumulative, &[]).unwrap();

        // Picker should have spread the 30 layers across more than one column —
        // a single column for 30 nearly-square blocks would be ~30× taller than
        // wide, which is the bug we're fixing.
        assert_eq!(layout.layer_bounds.len(), 30);
        let unique_xs: std::collections::BTreeSet<u32> =
            layout.layer_bounds.iter().map(|b| b.canvas_x).collect();
        let unique_ys: std::collections::BTreeSet<u32> =
            layout.layer_bounds.iter().map(|b| b.canvas_y).collect();
        let cols = unique_xs.len();
        let rows = unique_ys.len();
        assert!(
            cols > 1,
            "30 layers should spread across multiple columns; got {cols}",
        );
        assert!(cols * rows >= 30, "{cols}x{rows} can't hold 30 layers");

        // Within each column, every layer's q_proj must share an x-coordinate.
        // Equivalently, blocks at the same column index must have the same
        // canvas_x.
        let mut x_by_col: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
        for b in &layout.layer_bounds {
            x_by_col.entry(b.canvas_x).or_default().push(b.canvas_y);
        }
        // Every column should have a consistent x, by construction (each
        // canvas_x is itself a "column key"). The check that matters: layers
        // sharing canvas_x should ALSO have grid-spaced canvas_y values (i.e.
        // they're stacked rows in the same column).
        let row_pitch = layout
            .layer_bounds
            .iter()
            .map(|b| b.height)
            .next()
            .expect("at least one layer")
            + PAD;
        for (col_x, ys) in &x_by_col {
            let mut sorted_ys = ys.clone();
            sorted_ys.sort();
            for w in sorted_ys.windows(2) {
                assert_eq!(
                    w[1] - w[0],
                    row_pitch,
                    "layers in column at x={col_x} must be row-aligned at pitch {row_pitch}; got {} between {} and {}",
                    w[1] - w[0],
                    w[0],
                    w[1],
                );
            }
        }
    }
}
