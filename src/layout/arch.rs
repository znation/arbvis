//! Architectural (structure-aware) layout for safetensors checkpoints.
//!
//! Every tensor is placed at its natural 2D element shape, 1 pixel per
//! element. Transformer blocks (`{prefix}.layers.{N}.{sub_path}`) share a
//! single canonical arrangement: each block draws every sub-tensor in the
//! same relative position so layer-N and layer-N+1 are pixel-aligned column
//! for column. Top-level tensors (embed_tokens, lm_head, norms) sit above
//! and below the block stack.
//!
//! The output canvas is queryable per tile via [`ArchLayout::regions_in_tile`].

use std::collections::BTreeMap;

use crate::data::{Source, SourceKind, SourceMeta};
use crate::layout::bin_pack::{align_up, pack, Slot};
use crate::layout::name_tree::{self, LayerSlot};
use crate::layout::TileRegion;
use crate::safetensors::{Dtype, TensorMeta};
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
            let Some(st) = s.safetensors.as_ref() else {
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

        // Decide the canvas width: max of the block-row width and the widest
        // top-level tensor.
        let top_widths: Vec<u32> = top_level
            .iter()
            .map(|(_, t, _)| {
                let (_, c) = t.element_shape();
                c.min(u32::MAX as u64) as u32
            })
            .collect();
        let canvas_w = layer_w
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

        // 2. Transformer blocks.
        for (idx, sub_map) in &blocks {
            let block_y = cursor_y;
            // Centre the block row inside the canvas width.
            let block_x_offset = canvas_w.saturating_sub(layer_w) / 2;
            for ((sp, _), pl) in canon_slots.iter().zip(placements.iter()) {
                if let Some((sidx, t, base_off)) = sub_map.get(sp) {
                    let (rows, cols) = t.element_shape();
                    let w = cols.min(u32::MAX as u64) as u32;
                    let h = rows.min(u32::MAX as u64) as u32;
                    // Within the slot, anchor at top-left (alignment matters
                    // more than centring — corresponding sub-tensors across
                    // layers stay column-aligned).
                    let _ = (w, h, pl);
                    let cx = block_x_offset.saturating_add(pl.x);
                    let cy = block_y.saturating_add(pl.y);
                    tensors.push(PlacedTensor {
                        source_idx: *sidx,
                        tensor_id: 0,
                        name: t.name.clone(),
                        dtype: t.dtype,
                        tensor_byte_start: base_off + t.file_start,
                        tensor_rows: rows,
                        tensor_cols: cols,
                        canvas_x: cx,
                        canvas_y: cy,
                        hue: name_hue_short(sp),
                        layer_idx: Some(*idx),
                    });
                }
            }
            layer_bounds.push(LayerBounds {
                layer_idx: *idx,
                canvas_x: block_x_offset,
                canvas_y: block_y,
                width: layer_w,
                height: layer_h,
            });
            cursor_y = block_y.saturating_add(layer_h).saturating_add(PAD);
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
        // covers everything.
        let raw_h = cursor_y.saturating_sub(PAD);
        let canvas_h = align_up(raw_h.max(1), TILE);
        let canvas_w = align_up(canvas_w, TILE);
        let width_tiles = canvas_w / TILE;
        let height_tiles = canvas_h / TILE;

        // Pyramid zoom levels: smallest k so 2^k >= max(width_tiles, height_tiles).
        let max_dim_tiles = width_tiles.max(height_tiles).max(1);
        let max_zoom = ((max_dim_tiles - 1).max(1) as f64).log2().ceil() as u32;

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
    Some(caps.get(2)?.as_str().to_string())
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

    /// Build a `Source` whose safetensors header reports the listed
    /// tensors, with sequential byte offsets. The Source's `kind` is
    /// `UnmatchedRegion` only so we don't need a real file on disk;
    /// `ArchLayout::try_build` actually looks only at `safetensors`.
    fn synthetic_source(tensors: Vec<TensorMeta>) -> Source {
        let total: u64 = tensors.iter().map(|t| t.file_end - t.file_start).sum();
        Source {
            file_idx: 0,
            kind: SourceKind::Buffered(Vec::new()),
            byte_size: total,
            safetensors: Some(crate::data::SafetensorsInfo {
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
}
