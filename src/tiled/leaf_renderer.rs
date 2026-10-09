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
pub enum LeafTile {
    Bytes { renderer_id: &'static str },
    Regions { renderer_id: &'static str },
    Padding,
}

impl LeafTile {
    /// `None` for `Padding`; the caller paints the padding color directly
    /// without resolving a renderer.
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
    /// permit and record success on `Ok`. A positional-only loader (one that
    /// colors purely from tile coordinates, no source bytes) returns `false`
    /// to skip the fetch and the permit it would consume.
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
    /// Empty registry: the built-in byte-floor renderer registers itself via
    /// [`LeafRegistry::with_defaults`] callers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `l` under its own [`LeafLoader::id`], replacing any previous
    /// registration with the same id.
    pub fn register_loader(&mut self, l: Arc<dyn LeafLoader>) {
        self.loaders.insert(l.id(), l);
    }

    /// Register `r` under its own [`LeafRenderer::id`], replacing any previous
    /// registration with the same id.
    pub fn register_renderer(&mut self, r: Arc<dyn LeafRenderer>) {
        self.renderers.insert(r.id(), r);
    }

    /// Look up a loader by the id it was registered under.
    pub fn loader(&self, id: &str) -> Option<Arc<dyn LeafLoader>> {
        self.loaders.get(id).cloned()
    }

    /// Look up a renderer by the id it was registered under.
    pub fn renderer(&self, id: &str) -> Option<Arc<dyn LeafRenderer>> {
        self.renderers.get(id).cloned()
    }

    /// Registry pre-populated with the two built-in loader+renderer pairs
    /// Registry pre-populated with arbvis's own `"hilbert-bytes"`
    /// loader+renderer pair. The `"arch"` pair (still defined in this file)
    /// is registered by `modelweightvis::register_all` so the arbvis binary
    /// stays byte-only
    /// and the modelweightvis binary picks up tensor-aware rendering.
    pub fn with_defaults() -> Self {
        let mut r = Self::new();
        r.register_loader(Arc::new(HilbertBytesLoader));
        r.register_renderer(Arc::new(HilbertBytesRenderer));
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
        // Every byte-Hilbert mode reads source bytes into the tile buffer, so
        // this is effectively always true today; routed through `needs_bytes`
        // so a future positional-only mode can opt out of the fetch (and the
        // throttle permit it would consume) without touching this loader.
        ctx.mode.needs_bytes()
    }

    fn load<'a>(&'a self, ctx: &'a LoadCtx<'a>) -> BoxFuture<'a, anyhow::Result<LoadedTile>> {
        Box::pin(async move {
            if !ctx.mode.needs_bytes() {
                return Ok(LoadedTile {
                    tx: ctx.tx,
                    ty: ctx.ty,
                    tile_buf: None,
                    extra: None,
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
                extra: None,
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

// `ArchRegionsLoader` and `ArchRegionsRenderer` live in `modelweightvis::leaf`.
// The arbvis default `LeafRegistry` no longer wires them up.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use image::Rgb;

    use super::*;
    use crate::layout::hilbert::HilbertLayout;
    use crate::tiled::LeafMode;

    // `Padding` tiles bypass dispatch entirely — the caller paints the
    // padding color directly — while the data-carrying variants expose the
    // id the pipeline resolves a loader+renderer pair under.
    #[test]
    fn renderer_id_by_variant() {
        assert_eq!(
            LeafTile::Bytes {
                renderer_id: "hilbert-bytes"
            }
            .renderer_id(),
            Some("hilbert-bytes")
        );
        assert_eq!(
            LeafTile::Regions {
                renderer_id: "arch"
            }
            .renderer_id(),
            Some("arch")
        );
        assert_eq!(LeafTile::Padding.renderer_id(), None);
    }

    // A stub loader/renderer standing in for a downstream plugin (e.g.
    // modelweightvis's `"arch"` pair).
    struct MockLeaf;

    impl LeafLoader for MockLeaf {
        fn id(&self) -> &'static str {
            "mock"
        }
        fn needs_io(&self, _ctx: &LoadCtx<'_>) -> bool {
            false
        }
        fn load<'a>(&'a self, _ctx: &'a LoadCtx<'a>) -> BoxFuture<'a, anyhow::Result<LoadedTile>> {
            unimplemented!()
        }
    }

    impl LeafRenderer for MockLeaf {
        fn id(&self) -> &'static str {
            "mock"
        }
        fn render(&self, _tile: LoadedTile, _ctx: &RenderCtx<'_>) -> Result<EncodedTile, String> {
            unimplemented!()
        }
    }

    #[test]
    fn register_lookup_and_miss() {
        let mut reg = LeafRegistry::new();
        assert!(reg.loader("mock").is_none());
        assert!(reg.renderer("mock").is_none());
        reg.register_loader(Arc::new(MockLeaf));
        reg.register_renderer(Arc::new(MockLeaf));
        assert_eq!(reg.loader("mock").unwrap().id(), "mock");
        assert_eq!(reg.renderer("mock").unwrap().id(), "mock");
        // An unrelated id is a miss on both halves.
        assert!(reg.loader("other").is_none());
        assert!(reg.renderer("other").is_none());
    }

    // Registration is id-keyed: a re-register under the same id replaces the
    // earlier entry (a downstream overrides a built-in pair this way).
    #[test]
    fn re_register_replaces() {
        let mut reg = LeafRegistry::new();
        reg.register_loader(Arc::new(MockLeaf));
        reg.register_loader(Arc::new(MockLeaf));
        reg.register_renderer(Arc::new(MockLeaf));
        reg.register_renderer(Arc::new(MockLeaf));
        assert_eq!(reg.loader("mock").unwrap().id(), "mock");
        assert_eq!(reg.renderer("mock").unwrap().id(), "mock");
    }

    #[test]
    fn clone_shares_entries() {
        let mut reg = LeafRegistry::new();
        reg.register_loader(Arc::new(MockLeaf));
        reg.register_renderer(Arc::new(MockLeaf));
        let clone = reg.clone();
        assert!(clone.loader("mock").is_some());
        assert!(clone.renderer("mock").is_some());
    }

    // The default registry wires the `"hilbert-bytes"` pair (and nothing else).
    #[test]
    fn with_defaults_has_hilbert_bytes_pair() {
        let reg = LeafRegistry::with_defaults();
        assert_eq!(reg.loader("hilbert-bytes").unwrap().id(), "hilbert-bytes");
        assert_eq!(reg.renderer("hilbert-bytes").unwrap().id(), "hilbert-bytes");
        assert!(reg.loader("arch").is_none());
        assert!(reg.renderer("arch").is_none());
    }

    fn plain_mode() -> LeafMode {
        LeafMode::Plain {
            pixel_lut: Arc::new(std::array::from_fn(|_| Rgb([0, 0, 0]))),
        }
    }

    fn load_ctx<'a>(mode: &'a LeafMode, layout: &'a HilbertLayout) -> LoadCtx<'a> {
        LoadCtx {
            tx: 0,
            ty: 0,
            zoom: 0,
            kh: 0,
            height_tiles: 1,
            square_pixels: 256,
            total: 256,
            mode,
            layout,
            source_data: &[],
            cumulative_offsets: &[],
        }
    }

    // The byte-Hilbert loader needs I/O in every mode that exists today, so
    // the pipeline always fetches source bytes and takes a throttle permit.
    #[test]
    fn hilbert_bytes_loader_needs_io() {
        let mode = plain_mode();
        let layout = HilbertLayout::from_total(256);
        let ctx = load_ctx(&mode, &layout);
        assert!(HilbertBytesLoader.needs_io(&ctx));
    }

    // The built-in loader resolves its id and the built-in renderer resolves
    // its id — dispatch keys off the same string on both pipeline halves.
    #[test]
    fn builtin_ids_match() {
        assert_eq!(HilbertBytesLoader.id(), "hilbert-bytes");
        assert_eq!(HilbertBytesRenderer.id(), "hilbert-bytes");
    }
}
