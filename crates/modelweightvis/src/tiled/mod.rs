//! Architectural `LeafLoader` + `LeafRenderer` impls plus the moved
//! `leaf_arch` submodule (per-tile tensor-region loader + dtype-aware
//! renderers).

pub mod leaf_arch;

use futures::future::BoxFuture;

use arbvis::{EncodedTile, LeafLoader, LeafMode, LeafRenderer, LoadCtx, LoadedTile, RenderCtx};

use crate::layout::arch::ArchLayout;
use crate::tiled::leaf_arch::{
    load_arch_tile_regions, render_arch_tile_diff, render_arch_tile_dtype, render_arch_tile_plain,
    render_arch_tile_xet, render_arch_tile_xet_dtype, LoadedArchTile,
};

/// Architectural leaf loader: fetches one coalesced byte range per tensor
/// region intersecting the tile. Thin wrapper over arbvis's
/// `load_arch_tile_regions`.
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
            let at = load_arch_tile_regions(
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
                extra: Some(Box::new(at)),
            })
        })
    }
}

/// Architectural leaf renderer: dispatches the `LeafMode` to the matching
/// `render_arch_tile_*` function. The arch-side equivalent of
/// `arbvis::tiled::render_one` (which stays in arbvis for byte-Hilbert).
pub struct ArchRegionsRenderer;

impl LeafRenderer for ArchRegionsRenderer {
    fn id(&self) -> &'static str {
        "arch"
    }

    fn render(&self, tile: LoadedTile, ctx: &RenderCtx<'_>) -> Result<EncodedTile, String> {
        let LoadedTile {
            tx,
            ty,
            tile_buf: _,
            extra,
        } = tile;
        // Recover the `LoadedArchTile` the loader produced; if the payload
        // is missing or the wrong type, render an empty arch tile (rare —
        // only happens when registry wiring is wrong).
        let at = extra
            .and_then(|e| e.downcast::<LoadedArchTile>().ok())
            .map(|b| *b)
            .unwrap_or_default();
        let fmt = ctx.fmt;
        let (image, bytes) = match ctx.mode {
            LeafMode::Plain { pixel_lut } => render_arch_tile_plain(&at, pixel_lut, fmt)?,
            LeafMode::Xet {
                pixel_lut,
                xorb_ranges,
                tableau,
            } => render_arch_tile_xet(&at, pixel_lut, xorb_ranges, tableau, fmt)?,
            LeafMode::Dtype { .. } => render_arch_tile_dtype(&at, fmt)?,
            LeafMode::Diff {
                pixel_lut,
                fills: _,
                plain_lut: _,
                tints: _,
            } => render_arch_tile_diff(&at, pixel_lut, fmt)?,
            LeafMode::XetDtype {
                xorb_ranges,
                tableau,
                dtype_ranges: _,
            } => render_arch_tile_xet_dtype(&at, xorb_ranges, tableau, fmt)?,
        };
        Ok(EncodedTile {
            tx,
            ty,
            image,
            bytes,
        })
    }
}
