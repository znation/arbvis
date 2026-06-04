//! Layout selection + the byte-only built-in layout.
//!
//! arbvis ships exactly one layout: [`hilbert::HilbertLayout`], a global
//! Hilbert curve over the concatenated byte stream of every source
//! (1 px = 1 byte). Everything else — `ArchLayout`, MoE-diff layout,
//! per-tensor regions, dtype-aware element decoding — lives in
//! `modelweightvis::layout` and registers via the `LayoutPlugin` trait.

pub mod hilbert;

use std::any::Any;

use crate::data::Source;
use crate::tiled::html::FileEntity;
use crate::tiled::leaf::TILE;

// `TileRegion` (the per-tensor tile-rectangle descriptor) lives in
// `modelweightvis::layout::TileRegion` along with the arch layout that
// produces it.

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

// `impl LayoutShape for ArchLayout` lives in `modelweightvis::layout::arch`
// with the `ArchLayout` struct itself.

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

// `ArchLayoutPlugin` and `MoeDiffLayoutPlugin` live in `modelweightvis::layout`
// (step 12e). The arbvis default registry no longer wires them up.

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
    diff_mode: bool,
    registry: &crate::registry::Registry,
) -> Box<dyn LayoutShape> {
    let ctx = crate::registry::LayoutBuildCtx {
        sources,
        cumulative_offsets,
        total_bytes,
        mode,
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
