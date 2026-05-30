//! Tile-renderer plugin surface (step 3 of the arbvis/modelweightvis split).
//!
//! `LeafTile` describes one tile in terms of what data it needs (raw bytes,
//! per-tensor regions, or just padding). The render stage looks up a
//! [`LeafRenderer`] by `LeafTile::renderer_id`, instead of branching on
//! `Layout::is_architectural()`. Today the registry has two built-ins
//! (`"hilbert-bytes"`, `"arch"`); once `modelweightvis` is its own crate it
//! will ship `"arch"` from there and register it on the shared registry.
//!
//! The load stage still keys off `Layout::is_architectural()` for now —
//! lifting that branch is a follow-up step that adds `Layout::describe_tile`
//! and routes load through the same `LeafTile` descriptor.

use std::collections::HashMap;
use std::sync::Arc;

use super::leaf::TileFormat;
use super::{EncodedTile, LeafMode, LoadedTile};

/// Per-tile descriptor used to pick a renderer.
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

/// One leaf-tile rendering strategy. Implementers are registered under a
/// string id in a [`LeafRendererRegistry`].
pub trait LeafRenderer: Send + Sync {
    fn id(&self) -> &'static str;
    fn render(&self, tile: LoadedTile, ctx: &RenderCtx<'_>) -> Result<EncodedTile, String>;
}

/// `id` → `LeafRenderer` lookup used by the tile pipeline. `Clone` is a
/// cheap `Arc` map clone so each worker can hold its own handle.
#[derive(Default, Clone)]
pub struct LeafRendererRegistry {
    map: HashMap<&'static str, Arc<dyn LeafRenderer>>,
}

impl LeafRendererRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, r: Arc<dyn LeafRenderer>) {
        self.map.insert(r.id(), r);
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn LeafRenderer>> {
        self.map.get(id).cloned()
    }

    /// Registry pre-populated with the two built-in renderers
    /// (`"hilbert-bytes"`, `"arch"`). Both ship from arbvis today; once
    /// `modelweightvis` is split out it will own `"arch"` and register it
    /// onto a registry the binary constructs.
    pub fn with_defaults() -> Self {
        let mut r = Self::new();
        r.register(Arc::new(HilbertBytesRenderer));
        r.register(Arc::new(ArchRegionsRenderer));
        r
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
