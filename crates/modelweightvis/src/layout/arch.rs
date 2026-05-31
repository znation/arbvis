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

use arbvis::{name_hue, Source, SourceKind, TILE};

use crate::data::SourceMeta;
use crate::format::{Dtype, TensorMeta};
use crate::layout::bin_pack::{align_up, pack, Slot};
use crate::layout::name_tree::{self, LayerSlot};
use crate::layout::TileRegion;

/// 8-px gutter between tensor slots inside a layer and between layer rows.
/// Keeps boundaries visually distinct without dominating the canvas.
const PAD: u32 = 8;

/// Maximum canvas width before bin-packing wraps a layer to a new shelf.
/// Sized to comfortably fit a llama-style block (concatenated q/k/v/o/MLP)
/// in one row while still pushing wider blocks to multi-line layouts.
const MAX_LAYER_WIDTH: u32 = 65_536;

/// Upper bound on a tensor's longer display axis, in overview pixels. Any
/// tensor whose longest element axis exceeds this is shrunk (`scale < 1`) so it
/// can't dominate the canvas; the lost detail is recovered by the variable-
/// depth detail tiles (`ArchLayout::detail_depth`).
const CAP_HI: u32 = 2048;

/// Minimum display thickness, in overview pixels, that a tensor's *shorter*
/// axis is enlarged toward (`scale > 1`) so a thin 1×N vector reads as visible
/// data rather than a 1px sliver lost in the gutter. Bounded by `CAP_HI` on the
/// long axis, so an extreme aspect ratio caps the enlargement rather than
/// blowing the footprint up.
const MIN_THICK: u32 = 6;

/// Safety cap on how many extra (deeper-than-overview) zoom levels the
/// variable-depth pyramid will generate. Each level is a 2× finer sampling, so
/// `D_MAX = 12` resolves up to a 4096× shrink back to ≥1px/element — far beyond
/// any realistic vocab axis (e.g. a 512k-token embedding shrunk to `CAP_HI`
/// needs only 8 levels). Per-tensor inclusion ([`detail_depth_for_scale`]) means
/// the deepest levels carry only the genuinely-most-shrunk tensors, so this
/// generous cap doesn't inflate tile counts for tensors that resolve sooner.
const D_MAX: u32 = 12;

/// Number of extra zoom levels (beyond the overview leaf) at which a tensor of
/// display `scale` still carries finer detail — i.e. how deep the variable-depth
/// pyramid must go before that tensor reaches ≥1 display px per element. `0` for
/// tensors at or above 1:1 (`scale >= 1`). Capped at [`D_MAX`] as a safety bound.
///
/// Used both for the layout-wide [`ArchLayout::detail_depth`] (the max over all
/// tensors) and, per tensor, to decide at which detail levels a tensor should be
/// rendered — a tensor is only included at level `k` while `k <= its depth`, so a
/// mildly-shrunk matrix isn't redundantly re-rendered (as pure replication) at
/// the deep levels a vocab embedding needs.
pub fn detail_depth_for_scale(scale: f32) -> u32 {
    if !scale.is_finite() || scale <= 0.0 || scale >= 1.0 {
        return 0;
    }
    ((1.0 / scale).log2().ceil() as u32).clamp(1, D_MAX)
}

/// Uniform display scale (display px per element, linear) for a tensor of
/// element shape `(rows, cols)`. Preserves the true 2D aspect ratio (same scale
/// on both axes) while decoupling the on-canvas footprint from the element
/// count:
///   * longest axis `> CAP_HI` → `scale = CAP_HI / longest` (shrink).
///   * otherwise enlarge toward `MIN_THICK` on the shorter axis, but never push
///     the longer axis past `CAP_HI`.
///
/// Mid-sized square-ish tensors land at `scale == 1`.
fn pick_scale(rows: u64, cols: u64) -> f32 {
    let long = rows.max(cols).max(1) as f32;
    let short = rows.min(cols).max(1) as f32;
    let cap = CAP_HI as f32 / long; // <1 shrinks; >1 is the enlarge ceiling
    if long >= CAP_HI as f32 {
        return cap; // shrink so the long axis lands at CAP_HI
    }
    // long < CAP_HI: enlarge thin tensors toward MIN_THICK thickness, but the
    // long axis must stay within CAP_HI, so clamp by `cap` (which is > 1 here).
    let thick = (MIN_THICK as f32 / short).max(1.0);
    thick.min(cap)
}

/// Display footprint `(width, height)` in overview pixels for element shape
/// `(rows, cols)` at scale `s`. Each axis is at least 1px.
fn disp_dims(rows: u64, cols: u64, s: f32) -> (u32, u32) {
    let w = ((cols as f32 * s).round() as u64).max(1);
    let h = ((rows as f32 * s).round() as u64).max(1);
    (w.min(u32::MAX as u64) as u32, h.min(u32::MAX as u64) as u32)
}

/// One placed tensor in the architectural canvas.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PlacedTensor {
    pub source_idx: usize,
    pub tensor_id: usize,
    pub name: String,
    pub dtype: Dtype,
    pub tensor_byte_start: u64,
    /// `element_shape` = (rows, cols). The true element grid; drives byte
    /// mapping. May differ from the on-canvas footprint (`disp_w`/`disp_h`).
    pub tensor_rows: u64,
    pub tensor_cols: u64,
    /// On-canvas display footprint, in overview pixels. `disp = round(element *
    /// scale)`. Shrunk (`scale < 1`) for huge tensors, enlarged (`scale > 1`)
    /// for thin vectors. This is what the layout packs and what the viewer
    /// draws at the overview zoom.
    pub disp_w: u32,
    pub disp_h: u32,
    /// Uniform display scale (display px per element). See [`pick_scale`].
    pub scale: f32,
    /// Top-left of the tensor's display-footprint rectangle on the canvas.
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
    /// Extra zoom levels (beyond `max_zoom`) the variable-depth pyramid carries
    /// genuine source-resolution detail for, so shrunk tensors can be resolved
    /// to individual elements. `0` when no tensor was shrunk. See [`pick_scale`]
    /// and [`D_MAX`].
    pub detail_depth: u32,
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
            let Some(st) = s.extensions.get::<crate::format::ModelInfo>() else {
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

        // For each sub-path, the slot size is the max *display footprint*
        // across all layers — guarantees every layer's slot fits. Each tensor's
        // footprint is its element shape scaled by `pick_scale` (preserving
        // aspect). Round up to a 16-px alignment so adjacent layers' grid lines
        // stay aligned even when dimensions differ subtly.
        let canon_slots: Vec<(String, Slot)> = canonical_subpaths
            .iter()
            .map(|sp| {
                let mut max_w: u32 = 1;
                let mut max_h: u32 = 1;
                for sub in blocks.values() {
                    if let Some((_, t, _)) = sub.get(sp) {
                        let (r, c) = t.element_shape();
                        let (dw, dh) = disp_dims(r, c, pick_scale(r, c));
                        max_w = max_w.max(dw);
                        max_h = max_h.max(dh);
                    }
                }
                let slot = Slot {
                    width: align_up(max_w, 16),
                    height: align_up(max_h, 16),
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
        // top-level tensor *display footprint* (a very-wide tensor is shrunk by
        // `pick_scale` so its footprint width stays bounded by `CAP_HI`).
        let top_widths: Vec<u32> = top_level
            .iter()
            .map(|(_, t, _)| {
                let (rows, c) = t.element_shape();
                disp_dims(rows, c, pick_scale(rows, c)).0
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
            cursor_y = place_top_level(&mut tensors, canvas_w, cursor_y, *sidx, t, *base_off);
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
                    let s = pick_scale(rows, tcols);
                    let (dw, dh) = disp_dims(rows, tcols, s);
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
                        disp_w: dw,
                        disp_h: dh,
                        scale: s,
                        canvas_x: cx,
                        canvas_y: cy,
                        hue: name_hue(sp),
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
            cursor_y = place_top_level(&mut tensors, canvas_w, cursor_y, *sidx, t, *base_off);
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

        // How many extra zoom levels the variable-depth pyramid needs: the max
        // over all tensors of their individual detail depth (the most-shrunk
        // tensor drives it). `0` when nothing was shrunk.
        let detail_depth = tensors
            .iter()
            .map(|t| detail_depth_for_scale(t.scale))
            .max()
            .unwrap_or(0);

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
            detail_depth,
            tensors,
            layer_bounds,
            architecture,
            sorted_idx,
        })
    }

    /// All tensor regions that overlap the tile at `(zoom, tx, ty)`.
    ///
    /// `zoom >= max_zoom`: the overview leaf is at `max_zoom` (footprint drawn
    /// 1:1 with `disp_w`/`disp_h`); each deeper level multiplies the whole
    /// canvas — and every tensor's footprint — by `f = 2^(zoom - max_zoom)`, so
    /// shrunk tensors reveal finer element detail the farther you zoom. The
    /// pixel→element map per painted pixel is
    /// `element = floor((samp + delta) * tensor_dim / footprint_dim)`.
    ///
    /// O(n) scan — fine because the architectural canvas typically holds
    /// O(hundreds) of tensors and tile rendering is the dominant cost
    /// downstream anyway. If this ever becomes a hot path, swap for an
    /// interval-tree on `canvas_y`.
    pub fn regions_in_tile(&self, zoom: u32, tx: u32, ty: u32) -> Vec<TileRegion> {
        let f = 1u64 << zoom.saturating_sub(self.max_zoom);
        let tile_x0 = tx as u64 * TILE as u64;
        let tile_y0 = ty as u64 * TILE as u64;
        let tile_x1 = tile_x0 + TILE as u64;
        let tile_y1 = tile_y0 + TILE as u64;

        let mut out = Vec::new();
        for &i in &self.sorted_idx {
            let t = &self.tensors[i];
            // Footprint and origin at this zoom level (whole canvas × f).
            let fw = t.disp_w as u64 * f;
            let fh = t.disp_h as u64 * f;
            let tx0 = t.canvas_x as u64 * f;
            let ty0 = t.canvas_y as u64 * f;
            let tx1 = tx0 + fw;
            let ty1 = ty0 + fh;

            // Early skip — once a tensor's top edge is past the tile bottom,
            // every subsequent tensor in sorted order is too (we sorted by y;
            // the uniform × f preserves that order).
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

            // Display-pixel offsets of the painted rect within the footprint.
            let samp_x0 = ix0 - tx0;
            let samp_y0 = iy0 - ty0;
            let paint_w = ix1 - ix0;
            let paint_h = iy1 - iy0;

            // Element bounding box covered by the painted pixels: floor at the
            // first painted pixel, +1 past the last sampled element.
            let cols = t.tensor_cols.max(1);
            let rows = t.tensor_rows.max(1);
            let col_first = samp_x0 * cols / fw;
            let col_last = (samp_x0 + paint_w - 1) * cols / fw + 1;
            let row_first = samp_y0 * rows / fh;
            let row_last = (samp_y0 + paint_h - 1) * rows / fh + 1;

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
                footprint_w: fw,
                footprint_h: fh,
                samp_x0,
                samp_y0,
                tile_x0: (ix0 - tile_x0) as u32,
                tile_y0: (iy0 - tile_y0) as u32,
                tile_x1: (ix1 - tile_x0) as u32,
                tile_y1: (iy1 - tile_y0) as u32,
            });
        }
        out
    }

    /// Build an architectural canvas of per-MoE-layer N×N expert-vs-expert
    /// matrices. Triggered by [`crate::layout::select_layout`] when any
    /// source carries a `MoeCell` tag (set by
    /// [`crate::data::prepare_moe_diff_sources`]).
    ///
    /// Inside each cell the three weights `{gate_proj, up_proj, down_proj}`
    /// sit side-by-side at a uniform per-weight footprint sized so the
    /// longest tensor axis lands near [`MOE_SUB_AXIS_TARGET`]. Lower-triangle
    /// cells (`j < i`) are skipped — the raw diff is antisymmetric, so
    /// they'd just mirror the upper triangle.
    ///
    /// Returns `None` if no source carries an MoE tag (let the caller fall
    /// through to [`ArchLayout::try_build`] or hilbert).
    pub fn try_build_moe_diff(sources: &[Source], cumulative_offsets: &[u64]) -> Option<Self> {
        use crate::format::moe::ExpertWeight;

        // (layer, i, j, weight) → (source_idx, &TensorMeta, base_off)
        let mut cells: BTreeMap<(u32, u32, u32, ExpertWeight), (usize, &TensorMeta, u64)> =
            BTreeMap::new();
        for (sidx, s) in sources.iter().enumerate() {
            let Some(cell) = s.extensions.get::<crate::data::MoeCell>().copied() else {
                continue;
            };
            let Some(st) = s.extensions.get::<crate::format::ModelInfo>() else {
                continue;
            };
            let Some(t) = st.tensors.first() else {
                continue;
            };
            let off = cumulative_offsets.get(sidx).copied().unwrap_or(0);
            cells.insert((cell.layer, cell.i, cell.j, cell.weight), (sidx, t, off));
        }
        if cells.is_empty() {
            return None;
        }

        // Distinct layer indices, ascending.
        let layer_ids: Vec<u32> = {
            let set: std::collections::BTreeSet<u32> = cells.keys().map(|(l, ..)| *l).collect();
            set.into_iter().collect()
        };

        // Per-layer: determine N (max expert idx + 1) and a uniform per-weight
        // sub-cell footprint, large enough to hold any of the three weights.
        struct LayerGeom {
            n_experts: u32,
            sub_w: u32, // per-weight footprint width
            sub_h: u32, // per-weight footprint height
            scale: f32, // uniform scale applied to every per-weight tensor
            cell_w: u32,
            cell_h: u32,
        }
        let geoms: BTreeMap<u32, LayerGeom> = layer_ids
            .iter()
            .map(|&l| {
                let mut max_axis: u64 = 1;
                let mut max_rows: u64 = 1;
                let mut max_cols: u64 = 1;
                let mut n_experts: u32 = 0;
                for ((cl, i, j, _w), (_, t, _)) in cells.range(
                    (l, 0u32, 0u32, ExpertWeight::GateProj)
                        ..(l + 1, 0u32, 0u32, ExpertWeight::GateProj),
                ) {
                    debug_assert_eq!(*cl, l);
                    let (rows, cols) = t.element_shape();
                    max_rows = max_rows.max(rows);
                    max_cols = max_cols.max(cols);
                    max_axis = max_axis.max(rows.max(cols));
                    n_experts = n_experts.max(*i + 1).max(*j + 1);
                }
                // Per-weight scale: shrink so the long axis lands at
                // MOE_SUB_AXIS_TARGET. Never up-scale (sub_axis_target is
                // smaller than every realistic expert dim).
                let scale = (MOE_SUB_AXIS_TARGET as f32 / max_axis as f32).min(1.0);
                let (sub_w, sub_h) = disp_dims(max_rows, max_cols, scale);
                let sub_w = align_up(sub_w, 4);
                let sub_h = align_up(sub_h, 4);
                let cell_w = 3 * sub_w + 2 * MOE_INNER_PAD;
                let cell_h = sub_h;
                (
                    l,
                    LayerGeom {
                        n_experts,
                        sub_w,
                        sub_h,
                        scale,
                        cell_w,
                        cell_h,
                    },
                )
            })
            .collect();

        // Canvas-wide tensors and layer bounds. One matrix per layer, stacked
        // vertically with MOE_LAYER_GAP between.
        let mut tensors: Vec<PlacedTensor> = Vec::new();
        let mut layer_bounds: Vec<LayerBounds> = Vec::new();
        let mut cursor_y: u32 = 0;
        let mut canvas_w: u32 = 1;

        for &l in &layer_ids {
            let g = &geoms[&l];
            let n = g.n_experts;
            if n == 0 {
                continue;
            }
            let matrix_w = n
                .saturating_mul(g.cell_w)
                .saturating_add(n.saturating_sub(1).saturating_mul(MOE_CELL_PAD));
            let matrix_h = n
                .saturating_mul(g.cell_h)
                .saturating_add(n.saturating_sub(1).saturating_mul(MOE_CELL_PAD));
            canvas_w = canvas_w.max(matrix_w);
            let matrix_x = 0u32;
            let matrix_y = cursor_y;

            for i in 0..n {
                for j in i..n {
                    let cell_x = matrix_x
                        .saturating_add(j.saturating_mul(g.cell_w.saturating_add(MOE_CELL_PAD)));
                    let cell_y = matrix_y
                        .saturating_add(i.saturating_mul(g.cell_h.saturating_add(MOE_CELL_PAD)));
                    for (w_idx, weight) in [
                        ExpertWeight::GateProj,
                        ExpertWeight::UpProj,
                        ExpertWeight::DownProj,
                    ]
                    .iter()
                    .enumerate()
                    {
                        let Some((sidx, t, base_off)) = cells.get(&(l, i, j, *weight)) else {
                            continue;
                        };
                        let (rows, cols) = t.element_shape();
                        let (dw, dh) = disp_dims(rows, cols, g.scale);
                        // Centre each tensor inside its sub_w × sub_h sub-cell
                        // (shapes may differ slightly between gate/up vs down).
                        let sub_x =
                            cell_x.saturating_add((w_idx as u32) * (g.sub_w + MOE_INNER_PAD));
                        let sub_y = cell_y;
                        let inset_x = g.sub_w.saturating_sub(dw) / 2;
                        let inset_y = g.sub_h.saturating_sub(dh) / 2;
                        tensors.push(PlacedTensor {
                            source_idx: *sidx,
                            tensor_id: 0,
                            name: t.name.clone(),
                            dtype: t.dtype,
                            tensor_byte_start: base_off + t.file_start,
                            tensor_rows: rows,
                            tensor_cols: cols,
                            disp_w: dw,
                            disp_h: dh,
                            scale: g.scale,
                            canvas_x: sub_x.saturating_add(inset_x),
                            canvas_y: sub_y.saturating_add(inset_y),
                            hue: name_hue(weight.label()),
                            layer_idx: Some(l),
                        });
                    }
                }
            }

            layer_bounds.push(LayerBounds {
                layer_idx: l,
                canvas_x: matrix_x,
                canvas_y: matrix_y,
                width: matrix_w,
                height: matrix_h,
            });
            cursor_y = matrix_y
                .saturating_add(matrix_h)
                .saturating_add(MOE_LAYER_GAP);
        }

        if tensors.is_empty() {
            return None;
        }

        // Assign tensor_ids in canvas order.
        for (i, t) in tensors.iter_mut().enumerate() {
            t.tensor_id = i;
        }

        let raw_h = cursor_y.saturating_sub(MOE_LAYER_GAP).max(1);
        let raw_canvas_h = align_up(raw_h, TILE);
        let raw_canvas_w = align_up(canvas_w, TILE);
        let raw_width_tiles = (raw_canvas_w / TILE).max(1);
        let raw_height_tiles = (raw_canvas_h / TILE).max(1);
        let width_tiles = next_pow2(raw_width_tiles);
        let height_tiles = next_pow2(raw_height_tiles);
        let canvas_w = width_tiles * TILE;
        let canvas_h = height_tiles * TILE;
        let max_zoom = (width_tiles.min(height_tiles).max(1) as f64).log2().round() as u32;

        let detail_depth = tensors
            .iter()
            .map(|t| detail_depth_for_scale(t.scale))
            .max()
            .unwrap_or(0);

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
            detail_depth,
            tensors,
            layer_bounds,
            architecture: format!(
                "MoE expert diff ({} layer(s), {} expert(s)/layer)",
                layer_ids.len(),
                geoms.values().map(|g| g.n_experts).max().unwrap_or(0),
            ),
            sorted_idx,
        })
    }
}

/// Target longest-axis size, in overview pixels, for a single per-weight
/// tensor inside an MoE-diff cell. Picked so a 64-expert × 64-expert matrix
/// of 3-weight cells lands in a comfortable few-thousand-pixel canvas at the
/// overview zoom; the variable-depth pyramid recovers element-level detail
/// on zoom-in.
const MOE_SUB_AXIS_TARGET: u32 = 64;
/// Gutter between the three per-weight sub-cells (gate/up/down) inside one
/// matrix cell.
const MOE_INNER_PAD: u32 = 4;
/// Gutter between adjacent matrix cells.
const MOE_CELL_PAD: u32 = 8;
/// Gap between stacked per-layer matrices.
const MOE_LAYER_GAP: u32 = 32;

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

/// Place one top-level tensor centred horizontally at `cursor_y`, at its
/// display footprint (element shape scaled by [`pick_scale`], preserving the
/// true 2D aspect — no row re-wrapping). Returns the advanced cursor.
fn place_top_level(
    tensors: &mut Vec<PlacedTensor>,
    canvas_w: u32,
    cursor_y: u32,
    sidx: usize,
    t: &TensorMeta,
    base_off: u64,
) -> u32 {
    let (rows, cols) = t.element_shape();
    let s = pick_scale(rows, cols);
    let (dw, dh) = disp_dims(rows, cols, s);
    let center_offset = canvas_w.saturating_sub(dw) / 2;
    tensors.push(PlacedTensor {
        source_idx: sidx,
        tensor_id: 0, // filled in by the caller in canvas order
        name: t.name.clone(),
        dtype: t.dtype,
        tensor_byte_start: base_off + t.file_start,
        tensor_rows: rows,
        tensor_cols: cols,
        disp_w: dw,
        disp_h: dh,
        scale: s,
        canvas_x: center_offset,
        canvas_y: cursor_y,
        hue: name_hue(&t.name),
        layer_idx: None,
    });
    cursor_y.saturating_add(dh).saturating_add(PAD)
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

#[cfg(test)]
mod tests {
    use super::*;
    use arbvis::Extensions;

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
    /// actually looks only at the source's `ModelInfo` extension.
    fn synthetic_source(tensors: Vec<TensorMeta>) -> Source {
        let total: u64 = tensors.iter().map(|t| t.file_end - t.file_start).sum();
        let mut extensions = Extensions::default();
        extensions.insert(crate::format::ModelInfo {
            format: crate::format::SourceFormat::Safetensors,
            tensors,
            color_ranges: Vec::new(),
        });
        Source {
            file_idx: 0,
            kind: SourceKind::Buffered(Vec::new()),
            byte_size: total,
            name_override: None,
            xet_terms: None,
            extensions,
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

    #[test]
    fn pick_scale_shrinks_very_long_axis_to_cap() {
        // A vocab-sized embedding [152000, 1024] (very tall) and its transpose
        // [1024, 152000] (very wide) both shrink so their *longest* display axis
        // lands at CAP_HI — preserving the true 2D aspect (no re-wrap).
        for (rows, cols) in [(152_000u64, 1024u64), (1024, 152_000)] {
            let s = pick_scale(rows, cols);
            assert!(s < 1.0, "{rows}x{cols} should shrink (scale {s} < 1)");
            let (dw, dh) = disp_dims(rows, cols, s);
            assert!(
                dw.max(dh) <= CAP_HI + 1,
                "longest display axis {} should be ~CAP_HI {CAP_HI}",
                dw.max(dh),
            );
            // Aspect ratio is preserved (within rounding): footprint stays a
            // strip, just a small one — not reshaped toward square.
            let want = cols as f64 / rows as f64;
            let got = dw as f64 / dh as f64;
            assert!(
                (want.log2() - got.log2()).abs() < 0.05,
                "aspect not preserved: want {want}, got {got}",
            );
        }
    }

    #[test]
    fn pick_scale_enlarges_thin_vectors() {
        // A 1×576 norm/bias vector renders 1px tall at scale 1; it should be
        // enlarged so its short axis thickens past 1px, while the long axis
        // stays bounded by CAP_HI.
        let s = pick_scale(1, 576);
        assert!(s > 1.0, "thin vector should enlarge (scale {s} > 1)");
        let (dw, dh) = disp_dims(1, 576, s);
        assert!(dh >= 2, "short axis should thicken past 1px (got {dh})");
        assert!(dw <= CAP_HI, "long axis must stay within CAP_HI (got {dw})");
    }

    #[test]
    fn pick_scale_leaves_midsized_unchanged() {
        // Square-ish tensors well within [CAP_LO-ish, CAP_HI] keep scale 1.
        for (rows, cols) in [(1024u64, 1024u64), (576, 576), (1536, 576), (576, 1536)] {
            assert_eq!(
                pick_scale(rows, cols),
                1.0,
                "{rows}x{cols} should keep scale 1",
            );
        }
    }

    /// A vocab-sized embedding is placed as a single shrunk rectangle that
    /// preserves the true element dims (no re-wrap, no body/tail split), with a
    /// bounded display footprint, and the layout reports a positive
    /// `detail_depth` so the variable-depth pyramid will recover its detail.
    #[test]
    fn embedding_shrinks_to_single_bounded_rect() {
        const VOCAB: u64 = 152_000;
        const HIDDEN: u64 = 1024;
        let mut tensors: Vec<TensorMeta> = Vec::new();
        let embed_start: u64 = 4096;
        tensors.push(TensorMeta {
            name: "model.embed_tokens.weight".to_string(),
            dtype: Dtype::F32,
            shape: vec![VOCAB, HIDDEN],
            file_start: embed_start,
            file_end: embed_start + VOCAB * HIDDEN * 4,
            packed_sidecars: None,
        });
        let mut off = embed_start + VOCAB * HIDDEN * 4;
        for i in 0..4u64 {
            for sub in ["self_attn.q_proj.weight", "mlp.gate_proj.weight"] {
                let bytes = HIDDEN * HIDDEN * 4;
                tensors.push(TensorMeta {
                    name: format!("model.layers.{i}.{sub}"),
                    dtype: Dtype::F32,
                    shape: vec![HIDDEN, HIDDEN],
                    file_start: off,
                    file_end: off + bytes,
                    packed_sidecars: None,
                });
                off += bytes;
            }
        }
        let source = synthetic_source(tensors);
        let layout = ArchLayout::try_build(&[source], &[0], &[]).unwrap();

        let chunks: Vec<&PlacedTensor> = layout
            .tensors
            .iter()
            .filter(|t| t.name == "model.embed_tokens.weight")
            .collect();
        assert_eq!(
            chunks.len(),
            1,
            "embedding is one rectangle (no re-wrap split)"
        );
        let e = chunks[0];
        // True element dims preserved; byte start is the tensor start.
        assert_eq!((e.tensor_rows, e.tensor_cols), (VOCAB, HIDDEN));
        assert_eq!(e.tensor_byte_start, embed_start);
        // Shrunk: footprint bounded, scale < 1.
        assert!(
            e.scale < 1.0,
            "embedding should be shrunk (scale {})",
            e.scale
        );
        assert!(
            e.disp_w.max(e.disp_h) <= CAP_HI + 1,
            "footprint {}x{} should be bounded by CAP_HI",
            e.disp_w,
            e.disp_h,
        );
        // Canvas no longer dominated by the 152k strip.
        assert!(
            layout.height < VOCAB as u32,
            "canvas height {} should be well under {VOCAB}",
            layout.height,
        );
        // Variable-depth detail levels were requested for the shrunk tensor.
        assert!(
            layout.detail_depth > 0,
            "shrinking should request detail levels; got {}",
            layout.detail_depth,
        );
    }

    /// With no tensor shrunk (all within CAP_HI), `detail_depth` is 0 — the
    /// viewer stays single-layer, no detail tiles are emitted.
    #[test]
    fn no_shrink_means_no_detail_depth() {
        let mut tensors: Vec<TensorMeta> = Vec::new();
        let mut off: u64 = 1024;
        for i in 0..4u64 {
            for sub in ["self_attn.q_proj.weight", "mlp.gate_proj.weight"] {
                let bytes = 512 * 512 * 4;
                tensors.push(TensorMeta {
                    name: format!("model.layers.{i}.{sub}"),
                    dtype: Dtype::F32,
                    shape: vec![512, 512],
                    file_start: off,
                    file_end: off + bytes,
                    packed_sidecars: None,
                });
                off += bytes;
            }
        }
        let source = synthetic_source(tensors);
        let layout = ArchLayout::try_build(&[source], &[0], &[]).unwrap();
        assert_eq!(layout.detail_depth, 0);
    }

    #[test]
    fn detail_depth_for_scale_bounds() {
        // Not shrunk → no detail levels.
        assert_eq!(detail_depth_for_scale(1.0), 0);
        assert_eq!(detail_depth_for_scale(1.5), 0);
        // Shrunk → ceil(log2(1/scale)) levels.
        assert_eq!(detail_depth_for_scale(0.5), 1); // 2× shrink
        assert_eq!(detail_depth_for_scale(0.25), 2); // 4× shrink
        assert_eq!(detail_depth_for_scale(0.1), 4); // 10× → ceil(log2 10)=4
                                                    // Degenerate inputs are guarded.
        assert_eq!(detail_depth_for_scale(0.0), 0);
        assert_eq!(detail_depth_for_scale(f32::NAN), 0);
        // Never exceeds the safety cap.
        assert!(detail_depth_for_scale(1e-9) <= D_MAX);
    }

    /// A vocab embedding plus several moderately-shrunk weight matrices: the
    /// layout-wide `detail_depth` is driven by the most-shrunk tensor (the
    /// embedding), while each matrix needs only its own (shallower) depth — the
    /// property `detail_coords` relies on to avoid re-rendering matrices at the
    /// deep levels only the embedding needs.
    #[test]
    fn detail_depth_driven_by_most_shrunk_tensor() {
        const VOCAB: u64 = 200_000;
        const HIDDEN: u64 = 4096; // > CAP_HI → matrices are shrunk too
        let mut tensors: Vec<TensorMeta> = Vec::new();
        let mut off: u64 = 0;
        let push = |tensors: &mut Vec<TensorMeta>, off: &mut u64, name: &str, shape: Vec<u64>| {
            let n: u64 = shape.iter().product();
            tensors.push(TensorMeta {
                name: name.to_string(),
                dtype: Dtype::F32,
                shape,
                file_start: *off,
                file_end: *off + n * 4,
                packed_sidecars: None,
            });
            *off += n * 4;
        };
        push(
            &mut tensors,
            &mut off,
            "model.embed_tokens.weight",
            vec![VOCAB, HIDDEN],
        );
        for i in 0..4u64 {
            push(
                &mut tensors,
                &mut off,
                &format!("model.layers.{i}.self_attn.q_proj.weight"),
                vec![HIDDEN, HIDDEN],
            );
            push(
                &mut tensors,
                &mut off,
                &format!("model.layers.{i}.mlp.gate_proj.weight"),
                vec![HIDDEN, HIDDEN],
            );
        }
        let source = synthetic_source(tensors);
        let layout = ArchLayout::try_build(&[source], &[0], &[]).unwrap();

        let embed = layout
            .tensors
            .iter()
            .find(|t| t.name.contains("embed_tokens"))
            .unwrap();
        let matrix = layout
            .tensors
            .iter()
            .find(|t| t.name.contains("q_proj"))
            .unwrap();

        // 4096×4096 matrix → scale 0.5 → resolves in 1 detail level.
        assert!(matrix.scale < 1.0 && matrix.scale >= 0.49);
        assert_eq!(detail_depth_for_scale(matrix.scale), 1);
        // The embedding is far more shrunk and needs more levels.
        assert!(detail_depth_for_scale(embed.scale) > 1);
        // Layout-wide depth == the embedding's (the max).
        assert_eq!(layout.detail_depth, detail_depth_for_scale(embed.scale));
    }
}
