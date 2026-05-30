//! Tile-loader and tile-renderer plugin surface.
//!
//! `LeafTile` describes one tile in terms of what data it needs (raw bytes,
//! per-tensor regions, or just padding). Both pipeline stages — load and
//! render — look up an implementation by `LeafTile::renderer_id` instead of
//! branching on layout type. Today the registry has two built-in pairs
//! (`"hilbert-bytes"` and `"arch"`); once `modelweightvis` is its own crate
//! it will ship `"arch"` from there and register it on the shared registry.
//!
//! Loaders and renderers share the same id so a `LeafTile` resolves both
//! halves of the pipeline in one lookup.

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::BoxFuture;

use crate::data::Data;
use crate::layout::arch::ArchLayout;
use crate::layout::LayoutShape;

use super::leaf::TileFormat;
use super::{EncodedTile, LeafMode, LoadedTile};

/// Per-tile descriptor used to pick a loader+renderer pair.
///
/// Today the variant is uniform across one plan (every tile in a Hilbert
/// plan is `Bytes`; every tile in an arch plan is `Regions`). The per-tile
/// callsite is structured so future layouts can return a mix — e.g. mostly
/// `Regions` with `Padding` for fully-empty tiles — without touching the
/// dispatch.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum LeafTile {
    Bytes { renderer_id: &'static str },
    Regions { renderer_id: &'static str },
    Padding,
}

impl LeafTile {
    /// `None` for `Padding`; the caller paints the padding color directly
    /// without resolving a renderer.
    #[allow(dead_code)]
    pub fn renderer_id(&self) -> Option<&'static str> {
        match self {
            LeafTile::Bytes { renderer_id } | LeafTile::Regions { renderer_id } => {
                Some(renderer_id)
            }
            LeafTile::Padding => None,
        }
    }
}

/// Inputs a loader needs to read bytes (or per-tensor regions) for one tile.
///
/// All fields are borrows so the call site can build a `LoadCtx` per
/// `(tx, ty)` inside the worker loop without copying its captured arcs.
pub struct LoadCtx<'a> {
    pub tx: u32,
    pub ty: u32,
    /// Pyramid zoom of the current pass. Hilbert ignores; arch uses it to
    /// scale the per-tensor display footprint.
    pub zoom: u32,
    pub kh: u8,
    pub height_tiles: u32,
    pub square_pixels: u64,
    pub total: u64,
    pub mode: &'a LeafMode,
    pub layout: &'a dyn LayoutShape,
    pub source_data: &'a [Data],
    pub cumulative_offsets: &'a [u64],
}

/// Inputs a renderer needs beyond the loaded tile.
///
/// Byte-Hilbert renderers consume the geometry fields (`kh`, `height_tiles`,
/// `square_pixels`, `total`); architectural renderers ignore them. Kept as
/// one shared struct so the trait signature is identical across renderers.
pub struct RenderCtx<'a> {
    pub mode: &'a LeafMode,
    pub fmt: TileFormat,
    pub kh: u8,
    pub height_tiles: u32,
    pub square_pixels: u64,
    pub total: u64,
}

/// One leaf-tile load strategy. Implementers are registered under a string id
/// in a [`LeafRegistry`] alongside the matching [`LeafRenderer`].
pub trait LeafLoader: Send + Sync {
    fn id(&self) -> &'static str;

    /// Whether this loader will perform I/O for the given context. The
    /// pipeline uses this to decide whether to acquire an HTTP throttle
    /// permit and record success on `Ok`. Byte-Hilbert returns `false` for
    /// `LeafMode::Dtype` (positional-only, no bytes needed); arch always
    /// fetches.
    fn needs_io(&self, ctx: &LoadCtx<'_>) -> bool;

    fn load<'a>(&'a self, ctx: &'a LoadCtx<'a>) -> BoxFuture<'a, anyhow::Result<LoadedTile>>;
}

/// One leaf-tile rendering strategy. Implementers are registered under a
/// string id in a [`LeafRegistry`] alongside the matching [`LeafLoader`].
pub trait LeafRenderer: Send + Sync {
    fn id(&self) -> &'static str;
    fn render(&self, tile: LoadedTile, ctx: &RenderCtx<'_>) -> Result<EncodedTile, String>;
}

/// `id` → `(loader, renderer)` lookup used by the tile pipeline. `Clone` is a
/// cheap `Arc` map clone so each worker can hold its own handle.
#[derive(Default, Clone)]
pub struct LeafRegistry {
    loaders: HashMap<&'static str, Arc<dyn LeafLoader>>,
    renderers: HashMap<&'static str, Arc<dyn LeafRenderer>>,
}

impl LeafRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_loader(&mut self, l: Arc<dyn LeafLoader>) {
        self.loaders.insert(l.id(), l);
    }

    pub fn register_renderer(&mut self, r: Arc<dyn LeafRenderer>) {
        self.renderers.insert(r.id(), r);
    }

    pub fn loader(&self, id: &str) -> Option<Arc<dyn LeafLoader>> {
        self.loaders.get(id).cloned()
    }

    pub fn renderer(&self, id: &str) -> Option<Arc<dyn LeafRenderer>> {
        self.renderers.get(id).cloned()
    }

    /// Registry pre-populated with the two built-in loader+renderer pairs
    /// (`"hilbert-bytes"`, `"arch"`). Both ship from arbvis today; once
    /// `modelweightvis` is split out it will own `"arch"` and register it
    /// onto a registry the binary constructs.
    pub fn with_defaults() -> Self {
        let mut r = Self::new();
        r.register_loader(Arc::new(HilbertBytesLoader));
        r.register_loader(Arc::new(ArchRegionsLoader));
        r.register_renderer(Arc::new(HilbertBytesRenderer));
        r.register_renderer(Arc::new(ArchRegionsRenderer));
        r
    }
}

/// Byte-Hilbert leaf loader; thin wrapper over [`super::leaf::load_tile_bytes`].
pub struct HilbertBytesLoader;

impl LeafLoader for HilbertBytesLoader {
    fn id(&self) -> &'static str {
        "hilbert-bytes"
    }

    fn needs_io(&self, ctx: &LoadCtx<'_>) -> bool {
        // `LeafMode::Dtype` colors purely from positional dtype ranges, so the
        // Hilbert tile buffer is unused — skip the byte fetch entirely (and
        // the throttle permit it would consume).
        ctx.mode.needs_bytes()
    }

    fn load<'a>(&'a self, ctx: &'a LoadCtx<'a>) -> BoxFuture<'a, anyhow::Result<LoadedTile>> {
        Box::pin(async move {
            if !ctx.mode.needs_bytes() {
                return Ok(LoadedTile {
                    tx: ctx.tx,
                    ty: ctx.ty,
                    tile_buf: None,
                    arch_tile: None,
                });
            }
            let buf = super::leaf::load_tile_bytes(
                ctx.tx,
                ctx.ty,
                ctx.kh,
                ctx.height_tiles,
                ctx.square_pixels,
                ctx.total,
                ctx.source_data,
                ctx.cumulative_offsets,
            )
            .await?;
            Ok(LoadedTile {
                tx: ctx.tx,
                ty: ctx.ty,
                tile_buf: Some(buf),
                arch_tile: None,
            })
        })
    }
}

/// Architectural leaf loader; thin wrapper over
/// [`super::leaf_arch::load_arch_tile_regions`].
pub struct ArchRegionsLoader;

impl LeafLoader for ArchRegionsLoader {
    fn id(&self) -> &'static str {
        "arch"
    }

    fn needs_io(&self, _ctx: &LoadCtx<'_>) -> bool {
        // Arch always fetches at least one tensor region per tile — the
        // renderer needs them regardless of `LeafMode`.
        true
    }

    fn load<'a>(&'a self, ctx: &'a LoadCtx<'a>) -> BoxFuture<'a, anyhow::Result<LoadedTile>> {
        Box::pin(async move {
            // The arch loader is only registered against the `"arch"` layout
            // id, and the plan-build site routes by id; a layout reaching here
            // that isn't an `ArchLayout` is a registry/plan bug, not a runtime
            // case.
            let arch = ctx
                .layout
                .as_any()
                .downcast_ref::<ArchLayout>()
                .expect("ArchRegionsLoader dispatched against non-ArchLayout");
            let at = super::leaf_arch::load_arch_tile_regions(
                ctx.zoom,
                ctx.tx,
                ctx.ty,
                arch,
                ctx.source_data,
                ctx.cumulative_offsets,
            )
            .await?;
            Ok(LoadedTile {
                tx: ctx.tx,
                ty: ctx.ty,
                tile_buf: None,
                arch_tile: Some(at),
            })
        })
    }
}

/// Byte-Hilbert leaf renderer; thin wrapper over [`super::render_one`].
pub struct HilbertBytesRenderer;

impl LeafRenderer for HilbertBytesRenderer {
    fn id(&self) -> &'static str {
        "hilbert-bytes"
    }

    fn render(&self, tile: LoadedTile, ctx: &RenderCtx<'_>) -> Result<EncodedTile, String> {
        super::render_one(
            tile,
            ctx.mode,
            ctx.kh,
            ctx.height_tiles,
            ctx.square_pixels,
            ctx.total,
            ctx.fmt,
        )
    }
}

/// Architectural leaf renderer; thin wrapper over [`super::render_one_arch`].
pub struct ArchRegionsRenderer;

impl LeafRenderer for ArchRegionsRenderer {
    fn id(&self) -> &'static str {
        "arch"
    }

    fn render(&self, tile: LoadedTile, ctx: &RenderCtx<'_>) -> Result<EncodedTile, String> {
        super::render_one_arch(tile, ctx.mode, ctx.fmt)
    }
}
