//! 2D tile pipeline: per-tile rendering, pyramid averaging, and encoding,
//! plus the `LoadedTile`/`EncodedTile` types shared along the way.

pub mod html;
pub mod leaf;
pub mod leaf_renderer;
pub mod pyramid_accum;
pub mod single;
pub mod streaming;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_channel::{bounded, Receiver, Sender};

use indicatif::ProgressBar;

use crate::color::{build_diff_signed_lut, build_pixel_lut};
use crate::data::{load_source_data, Data, DiffFill, Source, SourceKind};
use crate::geometry::{file_rects, hilbert_to_xy_u64, name_hue, outer_segments, rects_centroid};
use crate::layout::hilbert::{hilbert_canvas, CanvasGeom};
use crate::layout::{select_layout, LayoutMode, LayoutShape};
use crate::progress::{counter_style, multi, queue_style, status_style};
use crate::throttle::{Throttle, MAX_FETCH_WORKERS};
use crate::tiled::html::FileEntity;
use crate::tiled::leaf::{
    render_leaf_tile_diff, render_leaf_tile_from_buf, render_leaf_tile_xet_from_buf, TileFormat,
    TILE, TILE_PIXELS,
};
use crate::tiled::leaf_renderer::{LeafRegistry, LeafTile, LoadCtx, RenderCtx};
use crate::tiled::pyramid_accum::{write_tile_file, LocalFileSink, PyramidAccumulator};
use crate::xet::{XorbMap, TABLEAU_20};

/// Channel capacity for the fetch→process queue, per CPU core. Keeps memory
/// bounded — each in-flight tile holds a `TILE_PIXELS`-byte buffer plus a
/// `3 * TILE_PIXELS`-byte RGB pixel buffer. At `TILE = 512` that's 256 KiB +
/// 768 KiB = 1 MiB per tile, so a 16-CPU machine caps at ~16 × 2 × 1 MiB ≈
/// 32 MiB.
const CHANNEL_CAPACITY_PER_CPU: usize = 2;

fn channel_cap() -> usize {
    std::thread::available_parallelism().map_or(8, |n| n.get() * CHANNEL_CAPACITY_PER_CPU)
}

fn num_cpus_for_processing() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// Seven stacked indicatif bars covering the four pipeline stages. The monitor
/// task in [`drive_pipeline`] refreshes the throttle line, the three queue
/// lines, and the fetched/rendered counters every 500 ms; the writer stage
/// increments `written` directly.
///
/// All bars share a single `MultiProgress` so they redraw atomically. When
/// stderr is not a TTY (non-interactive runs) construction returns `None`
/// and no bars are drawn.
struct PipelineProgress {
    throttle: ProgressBar,
    coord_q: ProgressBar,
    loaded: ProgressBar,
    loaded_q: ProgressBar,
    rendered: ProgressBar,
    encoded_q: ProgressBar,
    written: ProgressBar,
}

impl PipelineProgress {
    fn new(total_tiles: u64, queue_cap: usize, throttle_max: usize) -> Self {
        let m = multi();
        let add = |bar: ProgressBar| m.add(bar);

        // Throttle bar: pos = current AIMD `active_limit`, len = `max_workers`
        // ceiling (128). The message refreshes from the monitor task every
        // 500 ms with current in-flight count.
        let throttle = add(ProgressBar::new(throttle_max as u64))
            .with_style(status_style())
            .with_message("HTTP throttle: 0/0 (in flight: 0)");
        // Queue bars: pos = current depth, len = channel capacity.
        let coord_q = add(ProgressBar::new(queue_cap as u64))
            .with_style(queue_style())
            .with_message("tile coord queue");
        // Counter bars: pos = tiles completed at this stage, len = total tiles.
        let loaded = add(ProgressBar::new(total_tiles))
            .with_style(counter_style())
            .with_message("tiles loaded");
        let loaded_q = add(ProgressBar::new(queue_cap as u64))
            .with_style(queue_style())
            .with_message("load → render queue");
        let rendered = add(ProgressBar::new(total_tiles))
            .with_style(counter_style())
            .with_message("tiles rendered");
        let encoded_q = add(ProgressBar::new(queue_cap as u64))
            .with_style(queue_style())
            .with_message("render → write queue");
        let written = add(ProgressBar::new(total_tiles))
            .with_style(counter_style())
            .with_message("tiles written");

        // 100 ms tick keeps the spinner alive and ETA fresh even when a stage
        // is briefly idle (e.g. waiting on a slow HTTP response).
        for pb in [
            &throttle, &coord_q, &loaded, &loaded_q, &rendered, &encoded_q, &written,
        ] {
            pb.enable_steady_tick(Duration::from_millis(100));
        }

        Self {
            throttle,
            coord_q,
            loaded,
            loaded_q,
            rendered,
            encoded_q,
            written,
        }
    }

    fn finish_all(&self) {
        // `finish_and_clear` removes each bar from the global `MultiProgress`
        // (vs. `finish`, which leaves it visible). The global multi keeps
        // running for subsequent phases (pyramid build, upload), so we want
        // pipeline bars gone once the pipeline ends.
        for pb in [
            &self.throttle,
            &self.coord_q,
            &self.loaded,
            &self.loaded_q,
            &self.rendered,
            &self.encoded_q,
            &self.written,
        ] {
            pb.finish_and_clear();
        }
    }
}

/// Which tiles a pipeline pass should render. The overview pass walks the dense
/// leaf grid (streamed, so a huge canvas doesn't materialise a giant coord
/// vec); detail passes render a sparse, explicitly-listed set of tiles.
///
/// `pub(super)` because `tiled::streaming` constructs `Dense` variants directly.
pub(super) enum TileCoords {
    Dense { width_tiles: u32, height_tiles: u32 },
    Sparse(Vec<(u32, u32)>),
}

impl TileCoords {
    fn len(&self) -> u64 {
        match self {
            TileCoords::Dense {
                width_tiles,
                height_tiles,
            } => *width_tiles as u64 * *height_tiles as u64,
            TileCoords::Sparse(v) => v.len() as u64,
        }
    }
}

/// Per-tile data flowing through the pipeline after the load stage.
///
/// `pub(super)` because the `leaf_renderer` submodule's `LeafRenderer` impls
/// consume one of these and dispatch to the right `render_one*` function.
pub struct LoadedTile {
    pub tx: u32,
    pub ty: u32,
    /// `Some` for the byte-Hilbert loader; a fixed 256 KiB buffer painted
    /// 1 byte → 1 pixel by the byte-LUT renderer.
    pub tile_buf: Option<Box<[u8; TILE_PIXELS]>>,
    /// Opaque per-loader payload. Custom `LeafLoader` implementations
    /// (e.g. modelweightvis's `ArchRegionsLoader`) stuff their per-tile
    /// data here; the matching `LeafRenderer` downcasts to recover it.
    /// `None` means the renderer expects byte-LUT mode only.
    pub extra: Option<Box<dyn std::any::Any + Send + Sync>>,
}

/// A fully rendered leaf tile: its pyramid coordinates, decoded RGB pixels,
/// and the encoded PNG bytes handed to the tile sink.
pub struct EncodedTile {
    pub tx: u32,
    pub ty: u32,
    pub image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    pub bytes: Vec<u8>,
}

/// Which leaf render to run.
///
/// `pub(super)` because `TilePlan::mode` and `derive_leaf_format` (also
/// `pub(super)`) expose this type to the `tiled::streaming` submodule.
#[derive(Clone)]
pub enum LeafMode {
    Plain {
        pixel_lut: Arc<[image::Rgb<u8>; 256]>,
    },
    Xet {
        pixel_lut: Arc<[image::Rgb<u8>; 256]>,
        xorb_ranges: Arc<Vec<(u64, u64, u8)>>,
        tableau: Arc<[image::Rgb<u8>; 20]>,
    },
    /// Diff mode: byte → color via the signed-diff LUT, *plus* a crosshatch
    /// overlay for byte ranges that map to `UnmatchedRegion` sources (tensors
    /// or files that exist on only one side). `fills` is sorted by start
    /// offset, non-overlapping. `tints` carries byte ranges from
    /// `OneSidedRange` sources (JSON / JSONL structure-aware diff): those
    /// bytes are real file bytes and are rendered via the plain LUT, blended
    /// 50/50 with the fill color so the side of origin is still legible.
    Diff {
        pixel_lut: Arc<[image::Rgb<u8>; 256]>,
        plain_lut: Arc<[image::Rgb<u8>; 256]>,
        fills: Arc<Vec<(u64, u64, DiffFill)>>,
        tints: Arc<Vec<(u64, u64, DiffFill)>>,
    },
}

impl LeafMode {
    /// Whether the fetch stage needs to read bytes for this mode.
    fn needs_bytes(&self) -> bool {
        matches!(
            self,
            LeafMode::Plain { .. } | LeafMode::Xet { .. } | LeafMode::Diff { .. }
        )
    }

    /// Whether this mode produces leaves with ≤256 distinct colors, making
    /// indexed-PNG the smallest lossless option. Plain mode draws from a
    /// fixed 256-entry LUT.
    /// Xet mode multiplies the byte LUT by 20 Tableau colors per xorb, which
    /// can exceed 256 distinct colors in a single tile — indexed-PNG would
    /// fall back to truecolor in that case, so we route Xet through AVIF
    /// instead where the encoder can win on the high-color content.
    /// Diff mode adds at most 6 crosshatch colors (3 fills × 2 shades) on top
    /// of the 256-entry diff LUT — usually still ≤256 distinct colors per
    /// tile in practice, and the encoder falls back to truecolor when it
    /// isn't.
    fn is_palette_safe(&self) -> bool {
        matches!(self, LeafMode::Plain { .. } | LeafMode::Diff { .. })
    }
}

mod scenes;
use scenes::{partition_scenes, SceneGroup};

mod regen;
pub use regen::regen_html;

mod pipeline;
pub use pipeline::run_tiles;

/// Shared geometry / entity / mode computation for both `run_tiles` and
/// `run_tiles_hf_streaming`. Holds everything needed to drive the pipeline.
///
/// `pub(super)` (and its fields are `pub(super)`) because `tiled::streaming`
/// reads `mode`, `max_zoom`, `world_w`/`world_h`, `entities`, etc. on the plan
/// it gets back from `build_tile_plan`.
pub(super) struct TilePlan {
    kh: u8,
    pub(super) width_tiles: u32,
    pub(super) height_tiles: u32,
    pub(super) world_w: u32,
    pub(super) world_h: u32,
    pub(super) height: u32,
    pub(super) width: u32,
    pub(super) max_zoom: u32,
    /// Extra zoom levels carrying variable-depth detail (0 for Hilbert / no
    /// shrunk tensors). Mirrors `ArchLayout::detail_depth`.
    pub(super) detail_depth: u32,
    pub(super) total_tiles: u64,
    square_pixels: u64,
    total: u64,
    pub(super) mode: LeafMode,
    source_data: Arc<Vec<Data>>,
    cumulative_offsets: Arc<Vec<u64>>,
    pub(super) entities: Vec<FileEntity>,
    layout: Arc<dyn LayoutShape>,
    /// Loader+renderer registry consulted by the load and render stages.
    /// Constructed with the two built-in pairs (`"hilbert-bytes"`, `"arch"`);
    /// future plugin wiring will let callers extend it before plan construction.
    leaf: Arc<LeafRegistry>,
    /// Per-plan tile descriptor; today uniform across every tile in the plan
    /// (one variant per layout). See [`leaf_renderer::LeafTile`].
    leaf_tile: LeafTile,
}

/// Build the diff-mode leaf mode for `sources` (in canvas order): the
/// signed-delta LUT as `pixel_lut`, the plain byte LUT as `plain_lut`, plus
/// crosshatch `fills` from `UnmatchedRegion` sources and `tints` from
/// `OneSidedRange` sources. Their byte_size already accounts for their canvas
/// footprint; each source's cumulative offset + size becomes a range. Sources
/// are listed in canvas order, so each list is already sorted by start.
/// Shared by `build_tile_plan` and the single-image diff PNG renderer.
pub(super) fn diff_leaf_mode(sources: &[Source]) -> LeafMode {
    let mut fills = Vec::new();
    let mut tints = Vec::new();
    let mut cumulative = 0u64;
    for source in sources {
        match &source.kind {
            SourceKind::UnmatchedRegion { fill } if source.byte_size > 0 => {
                fills.push((cumulative, cumulative + source.byte_size, *fill));
            }
            SourceKind::OneSidedRange { fill, .. } if source.byte_size > 0 => {
                tints.push((cumulative, cumulative + source.byte_size, *fill));
            }
            _ => {}
        }
        cumulative += source.byte_size;
    }
    LeafMode::Diff {
        pixel_lut: Arc::new(build_diff_signed_lut()),
        plain_lut: Arc::new(build_pixel_lut()),
        fills: Arc::new(fills),
        tints: Arc::new(tints),
    }
}

pub(super) async fn build_tile_plan(
    sources: Vec<Source>,
    total: u64,
    diff_mode: bool,
    show_xet_xorbs: bool,
    layout_mode: LayoutMode,
    registry: &crate::registry::Registry,
) -> anyhow::Result<TilePlan> {
    // Hilbert geometry derived from the byte total — also drives the
    // generic file-rects entity path (`file_rects` reads `total_pixels`,
    // `square_pixels`, `num_squares`, `height`, `kh`). For arch the trait's
    // `canvas_geom`/`layout_entities` override these downstream.
    let g = hilbert_canvas(total);
    let CanvasGeom {
        kh,
        kw,
        width,
        height,
        square_pixels,
    } = g;
    let total_pixels: u64 = width as u64 * height as u64;
    let num_squares = 1u32 << (kw - kh);

    let pixel_lut = Arc::new(if diff_mode {
        build_diff_signed_lut()
    } else {
        build_pixel_lut()
    });

    let mut cumulative_offsets: Vec<u64> = Vec::with_capacity(sources.len());
    {
        let mut off = 0u64;
        for s in &sources {
            cumulative_offsets.push(off);
            off += s.byte_size;
        }
    }

    // Open all source Data handles. `load_source_data` is sync (mmap for
    // local, lightweight handle clone for HTTP/LazyDiff) so a plain loop is
    // fine.
    let source_data: Vec<Data> = {
        let mut v = Vec::with_capacity(sources.len());
        for s in &sources {
            v.push(load_source_data(s)?);
        }
        v
    };

    // The xorb_map drives leaf coloring (LeafMode::Xet) — only build it when
    // the user explicitly asked for xorb coloring.
    let xorb_map = if show_xet_xorbs {
        XorbMap::build(
            sources
                .iter()
                .zip(cumulative_offsets.iter())
                .map(|(s, &off)| (s.xet_terms.as_deref(), off)),
        )
    } else {
        XorbMap {
            global_ranges: Vec::new(),
        }
    };
    let xet_mode = !xorb_map.is_empty();
    let tableau: [image::Rgb<u8>; 20] = {
        let mut arr = [image::Rgb([0u8, 0, 0]); 20];
        for (i, c) in TABLEAU_20.iter().enumerate() {
            arr[i] = image::Rgb(*c);
        }
        arr
    };

    // Per-element overlays (per-region color ranges + per-region entity
    // labels) for structure-aware specializations live downstream now —
    // they go through a layout plugin's own loader/renderer. arbvis
    // byte-Hilbert renders the pure byte-LUT.

    // File-level entity overlay (per-source rect + label, no tensor
    // awareness). modelweightvis's arch layout supplies its own per-tensor
    // entities via `LayoutShape::layout_entities`; arbvis byte-Hilbert
    // sticks to one entity per source file.
    let mut entities: Vec<FileEntity> = Vec::new();
    {
        let mut cumulative: u64 = 0;
        for source in &sources {
            let name = source.name();
            let data_start = cumulative;
            let data_end = cumulative + source.byte_size;
            let rects = file_rects(
                data_start,
                data_end,
                total_pixels,
                square_pixels,
                num_squares,
                height,
                kh as u8,
            );
            let (pixel_x, pixel_y) = rects_centroid(&rects).unwrap_or_else(|| {
                let mid = data_start + (data_end - data_start) / 2;
                let sq = mid / square_pixels;
                let (lx, ly) = hilbert_to_xy_u64(mid % square_pixels, kh as u8);
                (sq as u32 * height + lx, ly)
            });
            let hue = name_hue(&name);
            let segments = outer_segments(&rects);
            let bbox = rects
                .first()
                .map(|&first| {
                    rects
                        .iter()
                        .skip(1)
                        .fold(first, |(x0, y0, x1, y1), &(rx0, ry0, rx1, ry1)| {
                            (x0.min(rx0), y0.min(ry0), x1.max(rx1), y1.max(ry1))
                        })
                })
                .unwrap_or((0, 0, 0, 0));
            entities.push(FileEntity {
                name,
                pixel_x,
                pixel_y,
                hue,
                byte_size: data_end - data_start,
                bbox,
                segments,
            });
            cumulative += source.byte_size;
        }
    }

    // arbvis byte-Hilbert renders one of three modes: xet (xorb-colored),
    // diff (signed-delta LUT + crosshatch), or plain byte-LUT. Structure-aware
    // specializations don't reach this path — they ship their own layout plugin
    // with a matching leaf loader/renderer.
    let mode = if xet_mode {
        LeafMode::Xet {
            pixel_lut: pixel_lut.clone(),
            xorb_ranges: Arc::new(xorb_map.global_ranges),
            tableau: Arc::new(tableau),
        }
    } else if diff_mode {
        let LeafMode::Diff {
            pixel_lut,
            plain_lut,
            fills,
            tints,
        } = diff_leaf_mode(&sources)
        else {
            unreachable!("diff_leaf_mode always returns LeafMode::Diff")
        };
        LeafMode::Diff {
            pixel_lut,
            plain_lut,
            fills,
            tints,
        }
    } else {
        LeafMode::Plain {
            pixel_lut: pixel_lut.clone(),
        }
    };

    // Sidecar metadata (config.json / model.safetensors.index.json) is
    // model-side; if needed, a `FormatPlugin` populates `Source.extensions`
    // and the arch layout plugin reads from there. arbvis itself only
    // dispatches the layout.
    let layout = select_layout(
        &sources,
        &cumulative_offsets,
        total,
        layout_mode,
        diff_mode,
        registry,
    )?;

    // Today the per-plan LeafRegistry is the registry's own. Clone here so
    // the plan owns its arc.
    let leaf_registry = registry.leaf.clone();

    // Canvas geometry + overlay entities are now layout-supplied: arch
    // returns per-tensor rects from `layout_entities`; Hilbert falls back to
    // the generic file-rects path computed above. This keeps `tiled/` free
    // of concrete arch references — modelweightvis owns the arch layout but
    // its trait impl populates everything the pipeline reads.
    let geom = layout.canvas_geom();
    let (
        kh_out,
        width_tiles_out,
        height_tiles_out,
        world_w_out,
        world_h_out,
        height_out,
        width_out,
        max_zoom_out,
        total_tiles_out,
        square_pixels_out,
        total_out,
    ) = (
        geom.kh,
        geom.width_tiles,
        geom.height_tiles,
        geom.world_w,
        geom.world_h,
        geom.height,
        geom.width,
        geom.max_zoom,
        geom.total_tiles,
        geom.square_pixels,
        geom.total,
    );
    let entities = layout.layout_entities().unwrap_or(entities);

    let detail_depth = layout.detail_depth();

    // `is_byte_layout()` distinguishes the Hilbert byte-stream pipeline
    // (`LeafTile::Bytes`, fixed 256 KiB tile buffer) from per-tensor region
    // pipelines (`LeafTile::Regions`, variable number of small fetches).
    // `layout.id()` doubles as the registry key for the matching loader+
    // renderer pair.
    let leaf_tile = if layout.is_byte_layout() {
        LeafTile::Bytes {
            renderer_id: layout.id(),
        }
    } else {
        LeafTile::Regions {
            renderer_id: layout.id(),
        }
    };

    Ok(TilePlan {
        kh: kh_out,
        width_tiles: width_tiles_out,
        height_tiles: height_tiles_out,
        world_w: world_w_out,
        world_h: world_h_out,
        height: height_out,
        width: width_out,
        max_zoom: max_zoom_out,
        detail_depth,
        total_tiles: total_tiles_out,
        square_pixels: square_pixels_out,
        total: total_out,
        mode,
        source_data: Arc::new(source_data),
        cumulative_offsets: Arc::new(cumulative_offsets),
        entities,
        layout: Arc::from(layout),
        leaf: Arc::new(leaf_registry),
        leaf_tile,
    })
}

/// Drive the four-stage tile pipeline:
///   coord enumerator → N load workers (throttled when HTTP) → num_cpus render
///   workers → write closure (caller-supplied).
///
/// The caller's `on_tile` closure is invoked sequentially (the pipeline keeps
/// a single write task draining the encoded-tile channel) so it can mutate
/// shared state freely.
pub(super) async fn drive_pipeline<W>(
    plan: &TilePlan,
    leaf_format: TileFormat,
    zoom: u32,
    coords: TileCoords,
    mut on_tile: W,
) -> anyhow::Result<()>
where
    W: FnMut(EncodedTile) -> anyhow::Result<()> + Send,
{
    let cap = channel_cap();
    let (coord_tx, coord_rx): (Sender<(u32, u32)>, Receiver<(u32, u32)>) = bounded(cap);
    let (loaded_tx, loaded_rx): (Sender<LoadedTile>, Receiver<LoadedTile>) = bounded(cap);
    let (encoded_tx, encoded_rx): (Sender<EncodedTile>, Receiver<EncodedTile>) = bounded(cap);

    // Bars are added to the global `progress::multi()`; in non-TTY runs that
    // draws to a hidden target, so all updates here are no-ops but the rest
    // of the pipeline code stays branchless.
    let progress = Arc::new(PipelineProgress::new(coords.len(), cap, MAX_FETCH_WORKERS));
    let loaded_count = Arc::new(AtomicU64::new(0));
    let rendered_count = Arc::new(AtomicU64::new(0));
    let shutdown = Arc::new(AtomicBool::new(false));

    // Monitor task: poll throttle state + channel lengths + counters every
    // 500 ms and update the UI.
    let monitor_handle = {
        let progress = progress.clone();
        let coord_rx = coord_rx.clone();
        let loaded_rx = loaded_rx.clone();
        let encoded_rx = encoded_rx.clone();
        let loaded_count = loaded_count.clone();
        let rendered_count = rendered_count.clone();
        let shutdown = shutdown.clone();
        tokio::spawn(async move {
            loop {
                let throttle = Throttle::global();
                progress
                    .throttle
                    .set_position(throttle.active_limit() as u64);
                progress.throttle.set_message(format!(
                    "HTTP throttle: {}/{} (in flight: {})",
                    throttle.active_limit(),
                    throttle.max_workers(),
                    throttle.in_flight(),
                ));
                progress.coord_q.set_position(coord_rx.len() as u64);
                progress.loaded_q.set_position(loaded_rx.len() as u64);
                progress.encoded_q.set_position(encoded_rx.len() as u64);
                progress
                    .loaded
                    .set_position(loaded_count.load(Ordering::Relaxed));
                progress
                    .rendered
                    .set_position(rendered_count.load(Ordering::Relaxed));
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        })
    };

    // Stage 1: coord enumerator. Drives whatever tile set this pass covers —
    // the dense overview grid (streamed), or a sparse set of detail tiles.
    let coord_task = tokio::spawn(async move {
        match coords {
            TileCoords::Dense {
                width_tiles,
                height_tiles,
            } => {
                for ty in 0..height_tiles {
                    for tx in 0..width_tiles {
                        if coord_tx.send((tx, ty)).await.is_err() {
                            return; // downstream closed
                        }
                    }
                }
            }
            TileCoords::Sparse(v) => {
                for (tx, ty) in v {
                    if coord_tx.send((tx, ty)).await.is_err() {
                        return;
                    }
                }
            }
        }
        // closing coord_tx (drop on scope end) signals load workers to drain.
    });

    // Stage 2: load workers. Spawn up to MAX_FETCH_WORKERS. When any source
    // is remote (`Data::Http`/`LazyDiff`), each load acquires the AIMD HTTP
    // throttle before reading source bytes; workers above `active_limit` park
    // on the throttle's Notify. When every source is local (mmap or in-memory
    // — typical after `materialize_http_sources`), the throttle is bypassed
    // so 128-way mmap parallelism isn't capped at the throttle's initial
    // 4-way limit.
    //
    // Dispatch is via the `LeafLoader` registry: the plan's `leaf_tile`
    // descriptor names a renderer id, the registry resolves it once before
    // spawning the load worker pool, and each worker clones the resulting
    // `Arc<dyn _>` into its own task. Mirrors the render-stage dispatch
    // immediately below.
    let any_remote_source = plan.source_data.iter().any(|d| !d.is_local());
    let loader = plan
        .leaf_tile
        .renderer_id()
        .and_then(|id| plan.leaf.loader(id))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no leaf loader registered for tile descriptor {:?}",
                plan.leaf_tile
            )
        })?;
    let mut load_handles = Vec::new();
    for _ in 0..MAX_FETCH_WORKERS {
        let coord_rx = coord_rx.clone();
        let loaded_tx = loaded_tx.clone();
        let source_data = plan.source_data.clone();
        let cumulative_offsets = plan.cumulative_offsets.clone();
        let loaded_count = loaded_count.clone();
        let kh = plan.kh;
        let height_tiles = plan.height_tiles;
        let square_pixels = plan.square_pixels;
        let total = plan.total;
        let layout = plan.layout.clone();
        let mode = plan.mode.clone();
        let loader = loader.clone();
        load_handles.push(tokio::spawn(async move {
            while let Ok((tx, ty)) = coord_rx.recv().await {
                let ctx = LoadCtx {
                    tx,
                    ty,
                    zoom,
                    kh,
                    height_tiles,
                    square_pixels,
                    total,
                    mode: &mode,
                    layout: layout.as_ref(),
                    source_data: &source_data,
                    cumulative_offsets: &cumulative_offsets,
                };
                // Throttle only when the loader will actually do I/O: a
                // positional-only loader can skip byte fetches entirely, and
                // we don't want to hold a permit (or call `record_success`)
                // for a no-op load.
                let do_io = loader.needs_io(&ctx);
                let permit = if any_remote_source && do_io {
                    Some(Throttle::global().acquire().await)
                } else {
                    None
                };
                let result = loader.load(&ctx).await;
                drop(permit);
                let loaded_tile = match result {
                    Ok(t) => {
                        if any_remote_source && do_io {
                            Throttle::global().record_success();
                        }
                        t
                    }
                    Err(e) => {
                        // Fatal: the throttle's per-call retry already
                        // covered transient HTTP issues; anything reaching
                        // here is a permanent failure. `{e:?}` (anyhow's
                        // Debug) prints the full caused-by chain plus the
                        // captured backtrace (RUST_BACKTRACE is set on by
                        // main), so the user sees where it originated and
                        // what wrapped it — not just the topmost context.
                        log::error!("leaf load `{}` ({tx},{ty}) failed:\n{e:?}", loader.id());
                        // Close the coord channel so the other 127 workers
                        // see Err once the ~20-entry buffer drains, instead
                        // of grinding through tens of thousands more tiles
                        // after a fatal error.
                        coord_rx.close();
                        return Err::<(), anyhow::Error>(e);
                    }
                };
                loaded_count.fetch_add(1, Ordering::Relaxed);
                if loaded_tx.send(loaded_tile).await.is_err() {
                    break;
                }
            }
            Ok(())
        }));
    }
    drop(loaded_tx); // close when all load workers exit
                     // Keep one clone of `coord_rx` alive so the writer (stage 4) can `close()`
                     // it on a fatal error to cascade shutdown upstream. Without this clone
                     // we'd have dropped every receiver here and have no handle to close on.
    let coord_rx_for_writer = coord_rx.clone();
    drop(coord_rx);

    // Stage 3: render workers (= num_cpus). Each pulls a LoadedTile, runs
    // the CPU-bound pixel math + PNG encode inside `spawn_blocking`, and
    // sends the encoded result to the write channel.
    //
    // Dispatch is via the `LeafRenderer` registry: the plan's `leaf_tile`
    // descriptor names a renderer id, the registry resolves it once before
    // the worker loop, and each worker clones the resulting `Arc<dyn _>`
    // into its blocking task. Replaces the previous `if is_arch_local`
    // branch — same call paths underneath, just routed by id.
    let renderer = plan
        .leaf_tile
        .renderer_id()
        .and_then(|id| plan.leaf.renderer(id))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no leaf renderer registered for tile descriptor {:?}",
                plan.leaf_tile
            )
        })?;
    let num_proc = num_cpus_for_processing();
    let mut process_handles = Vec::new();
    for _ in 0..num_proc {
        let loaded_rx = loaded_rx.clone();
        let encoded_tx = encoded_tx.clone();
        let rendered_count = rendered_count.clone();
        let mode = plan.mode.clone();
        let kh = plan.kh;
        let height_tiles = plan.height_tiles;
        let square_pixels = plan.square_pixels;
        let total = plan.total;
        let renderer = renderer.clone();
        process_handles.push(tokio::spawn(async move {
            while let Ok(tile) = loaded_rx.recv().await {
                let mode = mode.clone();
                let renderer = renderer.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let ctx = RenderCtx {
                        mode: &mode,
                        fmt: leaf_format,
                        kh,
                        height_tiles,
                        square_pixels,
                        total,
                    };
                    renderer.render(tile, &ctx)
                })
                .await;
                let encoded = match result {
                    Ok(Ok(e)) => e,
                    Ok(Err(e)) => {
                        // `render_one` returns `Result<_, String>` so the
                        // best we can do is wrap as anyhow + log the chain.
                        let e = anyhow::anyhow!("render_one: {e}");
                        log::error!("render worker failed:\n{e:?}");
                        loaded_rx.close();
                        return Err::<(), anyhow::Error>(e);
                    }
                    Err(e) => {
                        // tokio JoinError — typically a panic in render_one.
                        // Promote the panic message via anyhow so the chain
                        // logs cleanly.
                        let e = anyhow::anyhow!("render join failure: {e}");
                        log::error!("render worker join failed:\n{e:?}");
                        loaded_rx.close();
                        return Err(e);
                    }
                };
                rendered_count.fetch_add(1, Ordering::Relaxed);
                if encoded_tx.send(encoded).await.is_err() {
                    break;
                }
            }
            Ok(())
        }));
    }
    drop(encoded_tx);
    drop(loaded_rx);

    // Stage 4: writer (in this task). Drain the encoded channel.
    let mut writer_err: Option<anyhow::Error> = None;
    while let Ok(tile) = encoded_rx.recv().await {
        if let Err(e) = on_tile(tile) {
            // Log the writer error in the rich form, then cascade shutdown
            // by closing upstream channels so the other stages stop fast
            // instead of producing tens of thousands more encoded tiles.
            log::error!("tile writer failed:\n{e:?}");
            encoded_rx.close();
            coord_rx_for_writer.close();
            writer_err = Some(e);
            break;
        }
        progress.written.inc(1);
    }

    // Stop the monitor before awaiting stage handles so the bars don't keep
    // ticking after work finishes.
    shutdown.store(true, Ordering::Relaxed);
    let _ = monitor_handle.await;

    // Surface the first error from any stage. Writer error wins (it triggers
    // the cascade); otherwise pick the first worker error.
    let _ = coord_task.await;
    let mut first_err: Option<anyhow::Error> = writer_err;
    for h in load_handles {
        match h.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                first_err.get_or_insert(e);
            }
            // A tokio JoinError means the worker task panicked or was
            // cancelled. That must not be swallowed: a panicking loader
            // silently drops every tile it was responsible for, and treating
            // the run as successful would leave a truncated pyramid behind
            // with exit code 0.
            Err(join) => {
                let e = anyhow::anyhow!("load worker task failed: {join}");
                log::error!("{e:?}");
                first_err.get_or_insert(e);
            }
        }
    }
    for h in process_handles {
        match h.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                first_err.get_or_insert(e);
            }
            // Same reasoning as the load stage: render workers run most of
            // their work inside `spawn_blocking` (whose panics are already
            // surfaced above), but a panic outside that call would otherwise
            // vanish here.
            Err(join) => {
                let e = anyhow::anyhow!("render worker task failed: {join}");
                log::error!("{e:?}");
                first_err.get_or_insert(e);
            }
        }
    }

    progress.finish_all();

    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Render the variable-depth detail levels (`max_zoom+1 ..= max_zoom+detail_depth`).
///
/// Each level is rendered directly from source over a sparse set of tiles (the
/// shrunk tensors' footprints) — no pyramid accumulation, so this is a no-op for
/// Hilbert layouts and for arch layouts where nothing was shrunk. `write_tile`
/// persists one encoded tile at the given zoom (local file or Hub upload).
///
/// Each level reads the shrunk tensors' bytes again, but sources are
/// materialised to local files before tiling (`data::materialize_http_sources`),
/// so these are mmap memcpys served from the page cache — no HTTP, no throttle.
/// The only repeated work is per-element decode, bounded by the (sparse) detail
/// tile count, so accumulating levels into a sparse mini-pyramid isn't worth the
/// quad-alignment complexity it would add.
pub(super) async fn render_detail_levels<F>(
    plan: &TilePlan,
    leaf_format: TileFormat,
    write_tile: &F,
) -> anyhow::Result<()>
where
    F: Fn(&EncodedTile, u32) -> anyhow::Result<()> + Sync,
{
    let detail_depth = plan.layout.detail_depth();
    if detail_depth == 0 {
        return Ok(());
    }
    let max_zoom = plan.layout.canvas_geom().max_zoom;
    // Detail tiles are an enhancement layer: where they're missing the viewer
    // falls back to upsampling the base overview (transparent errorTileUrl). So
    // a detail-pass failure is logged and ends detail rendering, but is NOT
    // propagated — the already-complete overview output (and, for the HF path,
    // the whole staged upload) must not be discarded over one bad detail tile.
    for k in 1..=detail_depth {
        let zoom = max_zoom + k;
        let coords = plan.layout.detail_coords(zoom);
        if coords.is_empty() {
            continue;
        }
        log::info!(
            "Rendering {} detail tiles at zoom {zoom} (+{k})...",
            coords.len()
        );
        if let Err(e) = drive_pipeline(
            plan,
            leaf_format,
            zoom,
            TileCoords::Sparse(coords),
            |t: EncodedTile| write_tile(&t, zoom),
        )
        .await
        {
            log::warn!(
                "detail level {zoom} (+{k}) failed ({e:#}); skipping remaining detail levels — the viewer will upsample the overview in those regions"
            );
            break;
        }
    }
    Ok(())
}

pub(super) fn render_one(
    tile: LoadedTile,
    mode: &LeafMode,
    kh: u8,
    height_tiles: u32,
    square_pixels: u64,
    total: u64,
    fmt: TileFormat,
) -> Result<EncodedTile, String> {
    let LoadedTile {
        tx,
        ty,
        tile_buf,
        extra: _,
    } = tile;
    let (image, bytes) = match mode {
        LeafMode::Plain { pixel_lut } => {
            let buf = tile_buf.as_deref().expect("plain mode needs tile_buf");
            render_leaf_tile_from_buf(
                tx,
                ty,
                kh,
                height_tiles,
                square_pixels,
                total,
                buf,
                pixel_lut,
                fmt,
            )?
        }
        LeafMode::Xet {
            pixel_lut,
            xorb_ranges,
            tableau,
        } => {
            let buf = tile_buf.as_deref().expect("xet mode needs tile_buf");
            render_leaf_tile_xet_from_buf(
                tx,
                ty,
                kh,
                height_tiles,
                square_pixels,
                total,
                buf,
                pixel_lut,
                xorb_ranges,
                tableau,
                fmt,
            )?
        }
        LeafMode::Diff {
            pixel_lut,
            plain_lut,
            fills,
            tints,
        } => {
            let buf = tile_buf.as_deref().expect("diff mode needs tile_buf");
            render_leaf_tile_diff(
                tx,
                ty,
                kh,
                height_tiles,
                square_pixels,
                total,
                buf,
                pixel_lut,
                plain_lut,
                fills,
                tints,
                fmt,
            )?
        }
    };
    Ok(EncodedTile {
        tx,
        ty,
        image,
        bytes,
    })
}

// `render_one_arch` lives in `modelweightvis::leaf::ArchRegionsRenderer::render`.
// arbvis no longer needs the architectural render dispatch.

/// Pick the actual leaf tile format given the user's request and the render
/// mode. When the user asked for AVIF and the mode produces ≤256 distinct
/// colors per tile (Plain), indexed-PNG beats lossless AVIF
/// substantially (AV1 isn't tuned for palette content). Xet-mode tiles can
/// exceed 256 colors so they stay on AVIF; truecolor PNG passes through
/// unchanged.
pub(super) fn derive_leaf_format(user_choice: TileFormat, mode: &LeafMode) -> TileFormat {
    match (user_choice, mode.is_palette_safe()) {
        (TileFormat::Avif { .. }, true) => TileFormat::IndexedPng,
        (fmt, _) => fmt,
    }
}

// Streaming tile output (`run_tiles_hf_streaming`) lives in the
// [`streaming`](crate::tiled::streaming) submodule. It uses the same plan and
// pipeline helpers above; keeping it in its own file makes the "off-by-default"
// path easy to find and easy to delete if it's ever superseded.

#[cfg(test)]
mod pipeline_join_tests {
    use super::*;
    use crate::tiled::leaf_renderer::{LeafLoader, LeafRenderer};
    use futures::future::BoxFuture;

    // A loader that panics, standing in for any bug deep inside a plugin's
    // load path (slice index, unwraps on malformed data, ...). tokio catches
    // the unwind as a JoinError; the pipeline must surface it, not drop it.
    struct PanickingLoader;

    impl LeafLoader for PanickingLoader {
        fn id(&self) -> &'static str {
            "panic-test"
        }
        fn needs_io(&self, _ctx: &LoadCtx<'_>) -> bool {
            false
        }
        fn load<'a>(&'a self, _ctx: &LoadCtx<'a>) -> BoxFuture<'a, anyhow::Result<LoadedTile>> {
            Box::pin(async {
                panic!("loader panic injected by test");
            })
        }
    }

    // Never reached (the loader panics first) but required to register the
    // pair under the id the plan's leaf_tile names.
    struct NeverRenderer;

    impl LeafRenderer for NeverRenderer {
        fn id(&self) -> &'static str {
            "panic-test"
        }
        fn render(&self, _tile: LoadedTile, _ctx: &RenderCtx<'_>) -> Result<EncodedTile, String> {
            Err("never renderer must not be reached".to_string())
        }
    }

    #[tokio::test]
    async fn panicking_load_worker_fails_the_pipeline() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.bin");
        std::fs::write(&input, vec![0u8; 64]).unwrap();
        let total = 64;

        let plan = build_tile_plan(
            vec![Source {
                file_idx: 0,
                kind: SourceKind::File(input),
                byte_size: total,
                name_override: None,
                xet_terms: None,
                extensions: Default::default(),
            }],
            total,
            false,
            false,
            crate::layout::LayoutMode::Auto,
            &crate::registry::Registry::with_defaults(),
        )
        .await
        .unwrap();

        let mut reg = LeafRegistry::new();
        reg.register_loader(Arc::new(PanickingLoader));
        reg.register_renderer(Arc::new(NeverRenderer));
        let plan = TilePlan {
            leaf: Arc::new(reg),
            leaf_tile: LeafTile::Bytes {
                renderer_id: "panic-test",
            },
            ..plan
        };

        let err = drive_pipeline(
            &plan,
            TileFormat::IndexedPng,
            0,
            TileCoords::Dense {
                width_tiles: 1,
                height_tiles: 1,
            },
            |_tile| Ok(()),
        )
        .await
        .expect_err("a panicking load worker must fail the pipeline");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("load worker task failed"),
            "unexpected message: {msg}"
        );
    }
}

#[cfg(test)]
mod scene_tests {
    #[test]
    fn regen_html_missing_labels_json_says_the_dir_must_be_a_viewer_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let err = crate::tiled::regen_html(dir.path(), &crate::registry::Branding::default())
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("labels.json"), "unexpected message: {msg}");
        assert!(msg.contains("viewer bundle"), "unexpected message: {msg}");
        assert!(msg.contains("--3d"), "unexpected message: {msg}");
    }
}
