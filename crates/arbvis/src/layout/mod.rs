//! Mapping from canvas pixel coordinates to tensor element coordinates.
//!
//! Two built-in layouts, both exposed through the [`LayoutShape`] trait:
//!
//! - [`hilbert::HilbertLayout`] preserves the legacy byte-Hilbert behaviour:
//!   one continuous space-filling curve over the concatenated byte stream of
//!   every source. 1 px = 1 byte. Used for non-safetensors inputs and when
//!   the user passes `--layout hilbert`.
//!
//! - [`arch::ArchLayout`] is the structure-aware mode: each tensor occupies a
//!   rectangle of its natural 2D element shape (1 px = 1 element);
//!   transformer blocks are stacked vertically and pixel-aligned across the
//!   stack. Used when every source is safetensors and tensor names look
//!   transformer-style, unless overridden.

pub mod arch;
pub mod bin_pack;
pub mod hilbert;
pub mod model_config;
pub mod name_tree;
pub mod render;

use std::any::Any;

use crate::data::{Source, SourceMeta};
use crate::format::Dtype;
use crate::tiled::html::FileEntity;
use crate::tiled::leaf::TILE;

/// One contiguous (row-major) element range of one tensor that overlaps a
/// tile. The tile renderer fetches `byte_start..byte_end` from the source
/// and decodes elements at the natural dtype stride.
///
/// The tensor is drawn at a per-tensor display footprint that may differ from
/// its element grid (see [`crate::layout::arch::PlacedTensor`]): shrunk for
/// huge tensors, enlarged for thin vectors. At a given zoom level the footprint
/// is additionally multiplied by `2^(zoom - max_zoom)`. The renderer maps each
/// painted output pixel back to an element via the footprint→element ratio:
/// `element = floor((samp_off + delta_px) * tensor_dim / footprint_dim)`. The
/// `row_first`/`col_first` element bounds are the floor at the first painted
/// pixel, so they anchor both the byte fetch and the per-pixel offset.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TileRegion {
    /// Index into the source list this region's bytes come from.
    pub source_idx: usize,
    /// Index into the global tensor list (across all sources, in canvas order).
    pub tensor_id: usize,
    pub dtype: Dtype,
    /// Tensor row × col element-grid dimensions, as `TensorMeta::element_shape`.
    pub tensor_rows: u64,
    pub tensor_cols: u64,
    /// First/last row of the tensor element grid that overlap this tile.
    pub row_first: u64,
    pub row_last_exclusive: u64,
    /// First/last column of the tensor element grid that overlap this tile.
    pub col_first: u64,
    pub col_last_exclusive: u64,
    /// Byte offset within the source where this tensor's element data starts.
    pub tensor_byte_start: u64,
    /// Display footprint of the whole tensor at this tile's zoom level, in
    /// pixels (`disp_w/h * 2^(zoom - max_zoom)`). Drives the pixel→element map.
    pub footprint_w: u64,
    pub footprint_h: u64,
    /// Display-pixel offset, within the tensor's footprint, of the first
    /// painted pixel (`tile_x0`/`tile_y0`).
    pub samp_x0: u64,
    pub samp_y0: u64,
    /// Canvas pixel rectangle this region paints into, in *tile-local* (px, py)
    /// coordinates `[x0, x1) × [y0, y1)`.
    pub tile_x0: u32,
    pub tile_y0: u32,
    pub tile_x1: u32,
    pub tile_y1: u32,
}

/// User-facing layout selection.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LayoutMode {
    /// Auto-select: architectural if every source is safetensors AND
    /// transformer-style names are detected; otherwise hilbert.
    #[default]
    Auto,
    /// Force architectural (structure-aware) layout. Falls back to hilbert if
    /// no source is safetensors.
    Arch,
    /// Force the legacy global-Hilbert layout.
    Hilbert,
}

/// Behaviour every concrete layout implementation provides.
///
/// Used as a trait object (`Arc<dyn LayoutShape>`) by the tile pipeline so
/// the plan stage and the load/render workers can operate uniformly on any
/// layout. Variant-specific data (the placed-tensor list for arch, etc.) is
/// reached via [`LayoutShape::as_any`] + `Any::downcast_ref` from the few
/// call sites that need it (the arch single-image renderer, the arch
/// detail-tile pass, and the `ArchRegionsLoader` itself).
///
/// `id` doubles as the lookup key for the matching `LeafLoader`/`LeafRenderer`
/// pair in `LeafRegistry`: a layout that returns `"arch"` here dispatches to
/// the `"arch"` loader and renderer at tile time.
/// Canvas geometry every layout populates. The tile pipeline reads this
/// once to size the leaflet world, decide pyramid depth, and (for the
/// Hilbert path) walk the curve.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct CanvasGeom {
    /// Hilbert order in y (`height = 1 << kh`). Zero for non-Hilbert layouts
    /// — those paths don't index into the curve.
    pub kh: u8,
    pub width_tiles: u32,
    pub height_tiles: u32,
    /// World extent in leaflet coordinates at zoom 0 (one of the two axes
    /// always collapses to one tile).
    pub world_w: u32,
    pub world_h: u32,
    pub width: u32,
    pub height: u32,
    pub max_zoom: u32,
    pub total_tiles: u64,
    /// Pixel count of one Hilbert "square" (`height * height` for Hilbert,
    /// `1` for arch — the arch path doesn't divide by this value).
    pub square_pixels: u64,
    /// Total pixels the canvas covers (the Hilbert curve's byte budget /
    /// the arch canvas's pixel rectangle).
    pub total: u64,
}

pub trait LayoutShape: Send + Sync {
    /// Short stable identifier; also names the `LeafLoader`/`LeafRenderer`
    /// pair this layout dispatches to (`"hilbert-bytes"`, `"arch"`).
    fn id(&self) -> &'static str;
    /// Canvas dimensions + Hilbert geometry knobs.
    fn canvas_geom(&self) -> CanvasGeom;
    /// Number of extra zoom levels rendered as sparse detail tiles past
    /// `max_zoom`. Defaults to 0 for layouts that don't need per-tensor detail
    /// (Hilbert); arch overrides.
    fn detail_depth(&self) -> u32 {
        0
    }
    /// `true` when the tile pipeline should fetch a contiguous byte buffer
    /// per tile (Hilbert), `false` when it should fetch per-tensor regions
    /// (arch). Drives the [`crate::tiled::leaf_renderer::LeafTile`] variant.
    fn is_byte_layout(&self) -> bool {
        false
    }
    /// Layout-supplied list of overlay entities (per-tensor rectangles for
    /// arch; `None` to fall back to the generic `geometry::file_rects` path
    /// for Hilbert).
    fn layout_entities(&self) -> Option<Vec<FileEntity>> {
        None
    }
    /// Sparse tile coords this layout wants rendered at the given detail
    /// zoom (`max_zoom + 1..=max_zoom + detail_depth`). Empty for layouts
    /// without per-tensor detail levels.
    fn detail_coords(&self, _zoom: u32) -> Vec<(u32, u32)> {
        Vec::new()
    }
    /// Escape hatch for the few call sites that need the concrete layout
    /// type — implementations return `self`. Used by the layout's matching
    /// `LeafLoader` impl to recover its concrete struct.
    fn as_any(&self) -> &dyn Any;
}

impl LayoutShape for hilbert::HilbertLayout {
    fn id(&self) -> &'static str {
        "hilbert-bytes"
    }
    fn canvas_geom(&self) -> CanvasGeom {
        // Hilbert always has the wider axis on `world_w`, the shorter axis
        // collapsed to `TILE`. `width` is derived from `width_tiles * TILE`.
        let width = self.width_tiles * TILE;
        CanvasGeom {
            kh: self.kh,
            width_tiles: self.width_tiles,
            height_tiles: self.height_tiles,
            world_w: self.world_w,
            world_h: TILE,
            width,
            height: self.height,
            max_zoom: self.max_zoom,
            total_tiles: self.total_tiles,
            square_pixels: self.square_pixels,
            total: self.total,
        }
    }
    fn is_byte_layout(&self) -> bool {
        true
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl LayoutShape for arch::ArchLayout {
    fn id(&self) -> &'static str {
        "arch"
    }
    fn canvas_geom(&self) -> CanvasGeom {
        // `world_w`/`world_h` are the geographic extents at zoom 0 in
        // leaflet's coordinate system. At leaf zoom (`max_zoom`) the tile
        // grid is `width_tiles × height_tiles`; halving each step down to
        // zoom 0 leaves `width_tiles / 2^max_zoom × height_tiles / 2^max_zoom`
        // tiles, each TILE px wide in geo space.
        let two_pow_mz = 1u32 << self.max_zoom;
        let world_w = (self.width_tiles / two_pow_mz.max(1)).max(1) * TILE;
        let world_h = (self.height_tiles / two_pow_mz.max(1)).max(1) * TILE;
        CanvasGeom {
            // arch ignores `kh`/`square_pixels`; populate with safe values
            // that won't divide by zero if accidentally read.
            kh: 0,
            width_tiles: self.width_tiles,
            height_tiles: self.height_tiles,
            world_w,
            world_h,
            width: self.width,
            height: self.height,
            max_zoom: self.max_zoom,
            total_tiles: self.total_tiles,
            square_pixels: 1,
            total: self.width as u64 * self.height as u64,
        }
    }
    fn detail_depth(&self) -> u32 {
        self.detail_depth
    }
    fn layout_entities(&self) -> Option<Vec<FileEntity>> {
        let mut ents: Vec<FileEntity> = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            // Overlay rectangles use the on-canvas *display* footprint, not
            // the element grid, so labels/segments line up with what's drawn.
            let w = t.disp_w;
            let h = t.disp_h;
            let x0 = t.canvas_x;
            let y0 = t.canvas_y;
            let x1 = x0.saturating_add(w);
            let y1 = y0.saturating_add(h);
            let segments = vec![
                (x0, y0, x1, y0),
                (x1, y0, x1, y1),
                (x0, y1, x1, y1),
                (x0, y0, x0, y1),
            ];
            let cx = x0 + (x1 - x0) / 2;
            let cy = y0 + (y1 - y0) / 2;
            ents.push(FileEntity {
                name: t.name.clone(),
                pixel_x: cx,
                pixel_y: cy,
                hue: t.hue,
                byte_size: t
                    .tensor_rows
                    .saturating_mul(t.tensor_cols)
                    .saturating_mul(t.dtype.element_size() as u64),
                bbox: (x0, y0, x1, y1),
                segments,
            });
        }
        Some(ents)
    }
    fn detail_coords(&self, zoom: u32) -> Vec<(u32, u32)> {
        // Tile coords that any shrunk tensor's footprint overlaps at this
        // detail zoom. Deduped across tensors. Lifted from the prior
        // free function in `tiled::mod.rs`.
        use std::collections::BTreeSet;
        let level = zoom.saturating_sub(self.max_zoom);
        let f = 1u64 << level;
        let t_sz = TILE as u64;
        let mut set: BTreeSet<(u32, u32)> = BTreeSet::new();
        for t in &self.tensors {
            if arch::detail_depth_for_scale(t.scale) < level {
                continue;
            }
            let x0 = t.canvas_x as u64 * f;
            let y0 = t.canvas_y as u64 * f;
            let x1 = x0 + t.disp_w as u64 * f;
            let y1 = y0 + t.disp_h as u64 * f;
            let tx0 = (x0 / t_sz) as u32;
            let ty0 = (y0 / t_sz) as u32;
            let tx1 = ((x1 - 1) / t_sz) as u32;
            let ty1 = ((y1 - 1) / t_sz) as u32;
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    set.insert((tx, ty));
                }
            }
        }
        set.into_iter().collect()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Layout plugins
//
// `select_layout` (below) iterates `registry.layouts` in descending priority
// and returns the first plugin that's `applicable` and whose `build` returns
// `Some`. The three built-in plugins cover the existing layout selection
// tree: MoE-diff at priority 200, regular arch at 100, byte-Hilbert at the
// `i32::MIN` floor.
// ---------------------------------------------------------------------------

/// Byte-Hilbert plugin — always applies, always builds. The floor of the
/// priority stack.
pub struct HilbertLayoutPlugin;

impl crate::registry::LayoutPlugin for HilbertLayoutPlugin {
    fn id(&self) -> &'static str {
        "hilbert-bytes"
    }
    fn priority(&self) -> i32 {
        i32::MIN
    }
    fn applicable(&self, _ctx: &crate::registry::LayoutBuildCtx<'_>) -> bool {
        true
    }
    fn build(&self, ctx: &crate::registry::LayoutBuildCtx<'_>) -> Option<Box<dyn LayoutShape>> {
        Some(Box::new(hilbert::HilbertLayout::from_total(
            ctx.total_bytes,
        )))
    }
}

/// Architectural plugin — applies when sources carry safetensors metadata
/// and `--layout` doesn't force hilbert. Build returns `None` if no
/// transformer-style structure is detectable.
pub struct ArchLayoutPlugin;

impl ArchLayoutPlugin {
    /// In non-diff mode every source must be safetensors (otherwise the user
    /// has explicitly mixed in non-tensor inputs they'd expect to see). In
    /// diff mode it's enough that any source carries safetensors info: the
    /// typical case is a model-repo diff where the tensor sources are the
    /// point and tokenizer/config diffs are incidental.
    fn eligible(ctx: &crate::registry::LayoutBuildCtx<'_>) -> bool {
        if matches!(ctx.mode, LayoutMode::Hilbert) {
            return false;
        }
        let all = !ctx.sources.is_empty()
            && ctx
                .sources
                .iter()
                .all(|s| s.extensions.get::<crate::format::ModelInfo>().is_some());
        let any = ctx
            .sources
            .iter()
            .any(|s| s.extensions.get::<crate::format::ModelInfo>().is_some());
        if ctx.diff_mode {
            any
        } else {
            all
        }
    }
}

impl crate::registry::LayoutPlugin for ArchLayoutPlugin {
    fn id(&self) -> &'static str {
        "arch"
    }
    fn priority(&self) -> i32 {
        100
    }
    fn applicable(&self, ctx: &crate::registry::LayoutBuildCtx<'_>) -> bool {
        Self::eligible(ctx)
    }
    fn build(&self, ctx: &crate::registry::LayoutBuildCtx<'_>) -> Option<Box<dyn LayoutShape>> {
        let arch = arch::ArchLayout::try_build(ctx.sources, ctx.cumulative_offsets, ctx.metas)?;
        // Diff-mode info note: surface tensor sources that don't carry
        // safetensors info (e.g. tokenizer.json file diffs) — they won't
        // appear on the arch canvas.
        if ctx.diff_mode {
            let all = !ctx.sources.is_empty()
                && ctx
                    .sources
                    .iter()
                    .all(|s| s.extensions.get::<crate::format::ModelInfo>().is_some());
            if !all {
                let skipped = ctx
                    .sources
                    .iter()
                    .filter(|s| s.extensions.get::<crate::format::ModelInfo>().is_none())
                    .count();
                log::info!(
                    "arch layout: {skipped} non-safetensors diff source(s) will not appear on the arch canvas (file-level diffs are only rendered in --layout hilbert)"
                );
            }
        }
        Some(Box::new(arch))
    }
}

/// MoE-diff plugin — applies when any source carries a `MoeCell` tag (only
/// emitted by [`crate::data::prepare_moe_diff_sources`], so this fork can't
/// collide with a normal arch run).
pub struct MoeDiffLayoutPlugin;

impl crate::registry::LayoutPlugin for MoeDiffLayoutPlugin {
    fn id(&self) -> &'static str {
        "moe-diff"
    }
    fn priority(&self) -> i32 {
        200
    }
    fn applicable(&self, ctx: &crate::registry::LayoutBuildCtx<'_>) -> bool {
        if matches!(ctx.mode, LayoutMode::Hilbert) {
            return false;
        }
        ctx.sources
            .iter()
            .any(|s| s.extensions.get::<crate::data::MoeCell>().is_some())
    }
    fn build(&self, ctx: &crate::registry::LayoutBuildCtx<'_>) -> Option<Box<dyn LayoutShape>> {
        arch::ArchLayout::try_build_moe_diff(ctx.sources, ctx.cumulative_offsets)
            .map(|l| Box::new(l) as Box<dyn LayoutShape>)
    }
}

/// Build the layout for the given sources. Returns a byte-Hilbert layout for
/// non-architectural runs.
///
/// `metas` is an optional list parallel to `sources` carrying opportunistic
/// `config.json` / `model.safetensors.index.json` data. When non-empty it
/// lets the architectural layout: (1) validate the inferred layer count and
/// extend the canonical layer stack to `num_hidden_layers` when only a
/// subset of shards was loaded; (2) extend the canonical sub-path set with
/// tensor names listed in the index but not loaded. Pass `&[]` to opt out.
///
/// `diff_mode` relaxes the all-safetensors gate (see `ArchLayoutPlugin`).
///
/// Dispatch iterates `registry.layouts` in descending priority. The
/// `i32::MIN` `HilbertLayoutPlugin` floor always builds, so iteration always
/// terminates with a layout.
pub fn select_layout(
    sources: &[Source],
    cumulative_offsets: &[u64],
    total_bytes: u64,
    mode: LayoutMode,
    metas: &[SourceMeta],
    diff_mode: bool,
    registry: &crate::registry::Registry,
) -> Box<dyn LayoutShape> {
    let ctx = crate::registry::LayoutBuildCtx {
        sources,
        cumulative_offsets,
        total_bytes,
        mode,
        metas,
        diff_mode,
    };

    // Snapshot plugins ordered by descending priority. A handful of plugins,
    // sorted once per call — cheap. `Reverse` (not `-p.priority()`) because
    // the byte-Hilbert floor sits at `i32::MIN` and negating it overflows.
    let mut sorted: Vec<&std::sync::Arc<dyn crate::registry::LayoutPlugin>> =
        registry.layouts.iter().collect();
    sorted.sort_by_key(|p| std::cmp::Reverse(p.priority()));

    let mut chosen: Option<Box<dyn LayoutShape>> = None;
    let mut moe_applicable_but_failed = false;
    let mut arch_was_applicable = false;
    for plugin in &sorted {
        if !plugin.applicable(&ctx) {
            continue;
        }
        let pid = plugin.id();
        match plugin.build(&ctx) {
            Some(layout) => {
                chosen = Some(layout);
                break;
            }
            None => {
                if pid == "moe-diff" {
                    moe_applicable_but_failed = true;
                }
                if pid == "arch" {
                    arch_was_applicable = true;
                }
            }
        }
    }
    let chosen = chosen
        .expect("registry.layouts must include a floor plugin (HilbertLayoutPlugin at i32::MIN)");

    // Diagnostic logs preserve the original `select_layout` messages.
    if moe_applicable_but_failed {
        log::warn!("moe-diff sources present but layout build failed; falling back to hilbert");
    }
    if matches!(mode, LayoutMode::Arch) && chosen.id() != "arch" {
        if arch_was_applicable {
            log::warn!(
                "--layout arch requested but no recognisable structure; falling back to hilbert"
            );
        } else {
            log::warn!(
                "--layout arch requested but no input carries safetensors data; falling back to hilbert"
            );
        }
    }

    chosen
}
