pub mod html;
pub mod leaf;
pub mod leaf_renderer;
pub mod pyramid_accum;
pub mod streaming;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_channel::{bounded, Receiver, Sender};

use indicatif::ProgressBar;

use crate::color::{build_diff_signed_lut, build_pixel_lut};
use crate::data::{load_source_data, Data, DiffFill, SceneTag, Source, SourceKind};
use crate::geometry::{file_rects, hilbert_to_xy_u64, name_hue, outer_segments, rects_centroid};
use crate::layout::{select_layout, LayoutMode, LayoutShape};
use crate::progress::{counter_style, multi, queue_style, status_style};
use crate::throttle::{Throttle, MAX_FETCH_WORKERS};
use crate::tiled::html::FileEntity;
use crate::tiled::leaf::{
    render_leaf_tile_diff, render_leaf_tile_dtype, render_leaf_tile_from_buf,
    render_leaf_tile_xet_dtype_from_buf, render_leaf_tile_xet_from_buf, TileFormat, TILE,
    TILE_LOG2, TILE_PIXELS,
};
use crate::tiled::leaf_renderer::{LeafRegistry, LeafTile, LoadCtx, RenderCtx};
use crate::tiled::pyramid_accum::{LocalFileSink, PyramidAccumulator};
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
    Dtype {
        ranges: Arc<Vec<(u64, u64, image::Rgb<u8>)>>,
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
    /// Combined xet + safetensors: blend dtype hue with xorb tableau hue,
    /// modulated by byte intensity. Produces a single image where both tensor
    /// boundaries and xorb boundaries are visible.
    XetDtype {
        xorb_ranges: Arc<Vec<(u64, u64, u8)>>,
        tableau: Arc<[image::Rgb<u8>; 20]>,
        dtype_ranges: Arc<Vec<(u64, u64, image::Rgb<u8>)>>,
    },
}

impl LeafMode {
    /// Whether the fetch stage needs to read bytes for this mode.
    fn needs_bytes(&self) -> bool {
        matches!(
            self,
            LeafMode::Plain { .. }
                | LeafMode::Xet { .. }
                | LeafMode::Diff { .. }
                | LeafMode::XetDtype { .. }
        )
    }

    /// Whether this mode produces leaves with ≤256 distinct colors, making
    /// indexed-PNG the smallest lossless option. Plain mode draws from a
    /// fixed 256-entry LUT; Dtype mode uses a short list of dtype colors.
    /// Xet mode multiplies the byte LUT by 20 Tableau colors per xorb, which
    /// can exceed 256 distinct colors in a single tile — indexed-PNG would
    /// fall back to truecolor in that case, so we route Xet through AVIF
    /// instead where the encoder can win on the high-color content.
    /// Diff mode adds at most 6 crosshatch colors (3 fills × 2 shades) on top
    /// of the 256-entry diff LUT — usually still ≤256 distinct colors per
    /// tile in practice, and the encoder falls back to truecolor when it
    /// isn't. XetDtype crosses dtype hues with tableau hues per xorb, so it's
    /// high-color.
    fn is_palette_safe(&self) -> bool {
        matches!(
            self,
            LeafMode::Plain { .. } | LeafMode::Dtype { .. } | LeafMode::Diff { .. }
        )
    }
}

/// Return the extension of the first regular file found in `dir`.
fn sniff_ext_in(dir: &std::path::Path) -> Option<String> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
        })
}

/// Return the extension of any tile under `tiles/{zoom}/<x>/<y>.<ext>`.
fn sniff_ext_for_zoom(tiles_dir: &std::path::Path, zoom: u32) -> Option<String> {
    let zoom_dir = tiles_dir.join(format!("{zoom}"));
    let x_dir = std::fs::read_dir(&zoom_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .find(|e| e.path().is_dir())?;
    sniff_ext_in(&x_dir.path())
}

/// Parse one `labels.json` file entry back into a [`html::FileEntity`].
fn file_entity_from_json(v: &serde_json::Value) -> html::FileEntity {
    let name = v["name"].as_str().unwrap_or("").to_string();
    let pixel_x = v["x"].as_u64().unwrap_or(0) as u32;
    let pixel_y = v["y"].as_u64().unwrap_or(0) as u32;
    let hue = v["hue"].as_u64().unwrap_or(0) as u16;
    let byte_size = v["size"].as_u64().unwrap_or(0);
    let bbox = {
        if let Some(b) = v["bbox"].as_array() {
            let g = |i: usize| b.get(i).and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            (g(0), g(1), g(2), g(3))
        } else {
            (0, 0, 0, 0)
        }
    };
    let segments = if let Some(segs) = v["segs"].as_array() {
        segs.iter()
            .filter_map(|s| {
                let arr = s.as_array()?;
                let g = |i: usize| arr.get(i)?.as_u64().map(|x| x as u32);
                Some((g(0)?, g(1)?, g(2)?, g(3)?))
            })
            .collect()
    } else {
        vec![]
    };
    html::FileEntity {
        name,
        pixel_x,
        pixel_y,
        hue,
        byte_size,
        bbox,
        segments,
    }
}

/// Rebuild a multi-scene viewer from the scene-keyed `labels.json` shape
/// (`{ "scenes": [...] }`). All geometry is persisted per scene, so unlike the
/// single-scene path this needs no tile-directory scan. Returns `Ok(false)`
/// when `labels.json` isn't the scenes shape, so the caller falls through to
/// the legacy single-pyramid regen.
fn regen_html_multi(
    tile_dir: &Path,
    parsed: &serde_json::Value,
    branding: &crate::registry::Branding,
) -> anyhow::Result<bool> {
    let Some(scenes_json) = parsed.get("scenes").and_then(|v| v.as_array()) else {
        return Ok(false);
    };
    let u32_field =
        |s: &serde_json::Value, k: &str| s.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let str_field = |s: &serde_json::Value, k: &str, dflt: &str| {
        s.get(k)
            .and_then(|v| v.as_str())
            .unwrap_or(dflt)
            .to_string()
    };
    let scenes: Vec<html::SceneView> = scenes_json
        .iter()
        .map(|s| {
            let entities = s
                .get("files")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(file_entity_from_json).collect())
                .unwrap_or_default();
            let key = s.get("key").and_then(|v| v.as_str()).unwrap_or("");
            html::SceneView {
                key: (!key.is_empty()).then(|| key.to_string()),
                label: str_field(s, "label", ""),
                order: u32_field(s, "order"),
                world_w: u32_field(s, "world_w"),
                world_h: u32_field(s, "world_h"),
                max_zoom: u32_field(s, "max_zoom"),
                detail_depth: u32_field(s, "detail_depth"),
                height: u32_field(s, "height"),
                width: u32_field(s, "width"),
                leaf_ext: str_field(s, "leaf_ext", "png"),
                pyramid_ext: str_field(s, "pyramid_ext", "png"),
                entities,
            }
        })
        .collect();
    html::write_leaflet_html_multi(tile_dir, &scenes, &branding.name, &[], branding)?;
    log::info!(
        "Regenerated index.html ({} scenes) in {}",
        scenes.len(),
        tile_dir.display()
    );
    Ok(true)
}

/// Regenerate `index.html` for an existing tiles directory without re-rendering tiles.
pub fn regen_html(tile_dir: &Path, branding: &crate::registry::Branding) -> anyhow::Result<()> {
    let tiles_dir = tile_dir.join("tiles");

    // Read labels.json first. Newer outputs persist `max_zoom`/`detail_depth` so
    // we can tell the dense overview levels apart from the sparse variable-depth
    // detail levels — without that, the deepest *detail* zoom dir would be
    // mistaken for the overview leaf and corrupt every derived dimension.
    let labels_path = tile_dir.join("labels.json");
    let json_str = std::fs::read_to_string(&labels_path)
        .with_context(|| format!("cannot read {}", labels_path.display()))?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str)?;
    // Multi-scene outputs carry per-scene geometry in labels.json and live
    // under `tiles/<key>/…`, so they regenerate without any dir scan.
    if regen_html_multi(tile_dir, &parsed, branding)? {
        return Ok(());
    }
    let detail_depth = parsed
        .get("detail_depth")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let values: Vec<serde_json::Value> = match &parsed {
        serde_json::Value::Array(a) => a.clone(),
        serde_json::Value::Object(o) => o
            .get("files")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default(),
        _ => anyhow::bail!("labels.json: unexpected JSON shape (expected array or object)"),
    };

    // Overview leaf zoom: prefer the persisted value; else fall back to the
    // deepest zoom dir minus any detail levels (and minus 0 for legacy outputs
    // that predate the persisted fields).
    let deepest = std::fs::read_dir(&tiles_dir)
        .with_context(|| format!("cannot read {}", tiles_dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_string_lossy().parse::<u32>().ok())
        .max()
        .ok_or_else(|| anyhow::anyhow!("no zoom levels found in {}", tiles_dir.display()))?;
    let max_zoom = parsed
        .get("max_zoom")
        .and_then(|v| v.as_u64())
        .map(|m| m as u32)
        .unwrap_or_else(|| deepest.saturating_sub(detail_depth));

    let zoom_dir = tiles_dir.join(format!("{max_zoom}"));
    let width_tiles = std::fs::read_dir(&zoom_dir)
        .with_context(|| format!("cannot read {}", zoom_dir.display()))?
        .filter(|e| e.as_ref().map(|e| e.path().is_dir()).unwrap_or(false))
        .count() as u32;
    let first_x = std::fs::read_dir(&zoom_dir)?
        .filter_map(|e| e.ok())
        .find(|e| e.path().is_dir())
        .ok_or_else(|| anyhow::anyhow!("no x-dirs found at zoom {max_zoom}"))?;
    let first_x_path = first_x.path();
    let height_tiles = std::fs::read_dir(&first_x_path)?.count() as u32;
    // Sniff the tile extensions from existing files. Leaf zoom (max_zoom) and
    // the pyramid levels can use different formats — e.g. indexed-PNG leaves
    // with lossy-AVIF pyramid — so we sniff each one independently.
    let leaf_ext = sniff_ext_in(&first_x_path).unwrap_or_else(|| "png".to_string());
    let pyramid_ext = if max_zoom > 0 {
        sniff_ext_for_zoom(&tiles_dir, max_zoom - 1).unwrap_or_else(|| leaf_ext.clone())
    } else {
        leaf_ext.clone()
    };
    let height = height_tiles * TILE;
    // Unified Hilbert / arch bounds formula. At leaf zoom `max_zoom` the canvas
    // covers `width_tiles × height_tiles` tiles; the leaflet view at zoom 0
    // covers `width_tiles / 2^max_zoom × height_tiles / 2^max_zoom` tiles —
    // exactly one of which is 1 by construction (Hilbert: kh == max_zoom + 8;
    // arch: max_zoom = log2(min(w_p2, h_p2))). Multiplying back up by TILE
    // gives geo extents. For square Hilbert this collapses to `world_h = TILE,
    // world_w = TILE * 2^(kw-kh)` — the historical formula.
    let two_pow_mz = 1u32 << max_zoom;
    let world_w = (width_tiles / two_pow_mz.max(1)).max(1) * TILE;
    let world_h = (height_tiles / two_pow_mz.max(1)).max(1) * TILE;

    let entities: Vec<html::FileEntity> = values.iter().map(file_entity_from_json).collect();

    let width = width_tiles * TILE;
    html::write_leaflet_html(
        tile_dir,
        world_w,
        world_h,
        max_zoom,
        detail_depth,
        height,
        width,
        TILE,
        &entities,
        &branding.name,
        &[],
        &leaf_ext,
        &pyramid_ext,
        branding,
    )?;
    log::info!(
        "Regenerated index.html in {} (zoom 0–{max_zoom}, +{detail_depth} detail, {width_tiles}×{height_tiles} tiles, height={height})",
        tile_dir.display()
    );
    Ok(())
}

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
    let mut s = 2 * TILE_LOG2 as u32;
    while (1u64 << s) < total {
        s += 1;
    }
    let kh = s / 2;
    let kw = s.div_ceil(2);
    let height = 1u32 << kh;
    let width = 1u32 << kw;
    let square_pixels: u64 = (height as u64) * (height as u64);
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

    // Dtype-aware overlay (per-tensor color ranges + per-tensor entity
    // labels for safetensors in byte-Hilbert mode) lives in
    // `modelweightvis` now. arbvis byte-Hilbert renders pure byte-LUT;
    // tensor-aware visualisation goes through the arch layout instead.
    let dtype_mode = false;
    let combined_dtype_ranges: Vec<(u64, u64, image::Rgb<u8>)> = vec![];

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

    // For diff mode: collect crosshatch fills from any UnmatchedRegion
    // sources and tinted ranges from OneSidedRange sources. Their byte_size
    // already accounts for their canvas footprint; we just translate each
    // source's cumulative offset + size into a range. Sources are listed in
    // canvas order, so each resulting list is already sorted by start.
    let (diff_fills, diff_tints): (Vec<(u64, u64, DiffFill)>, Vec<(u64, u64, DiffFill)>) =
        if diff_mode {
            let mut fills = Vec::new();
            let mut tints = Vec::new();
            let mut cumulative = 0u64;
            for source in &sources {
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
            (fills, tints)
        } else {
            (Vec::new(), Vec::new())
        };

    // `XetDtype` and `Dtype` modes mix tensor color ranges with the byte/xet
    // overlay. arbvis byte-only has no tensor info, so we drop those modes
    // here. modelweightvis re-enables them when its `FormatPlugin` populates
    // `Source.extensions` and the arch layout takes over.
    let _ = (dtype_mode, combined_dtype_ranges);
    let mode = if xet_mode {
        LeafMode::Xet {
            pixel_lut: pixel_lut.clone(),
            xorb_ranges: Arc::new(xorb_map.global_ranges),
            tableau: Arc::new(tableau),
        }
    } else if diff_mode {
        LeafMode::Diff {
            pixel_lut: pixel_lut.clone(),
            plain_lut: Arc::new(build_pixel_lut()),
            fills: Arc::new(diff_fills),
            tints: Arc::new(diff_tints),
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
    );

    // Today the per-plan LeafRegistry is the registry's own; in step 12 once
    // tensor renderers move to modelweightvis it'll be the only path to get
    // the `"arch"` renderer registered. Clone here so the plan owns its arc.
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
                // Throttle only when the loader will actually do I/O: the
                // Hilbert loader skips byte fetches in `LeafMode::Dtype`, and
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
        if let Ok(Err(e)) = h.await {
            first_err.get_or_insert(e);
        }
    }
    for h in process_handles {
        if let Ok(Err(e)) = h.await {
            first_err.get_or_insert(e);
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
        LeafMode::Dtype { ranges } => {
            render_leaf_tile_dtype(tx, ty, kh, height_tiles, square_pixels, total, ranges, fmt)?
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
        LeafMode::XetDtype {
            xorb_ranges,
            tableau,
            dtype_ranges,
        } => {
            let buf = tile_buf.as_deref().expect("xet+dtype mode needs tile_buf");
            render_leaf_tile_xet_dtype_from_buf(
                tx,
                ty,
                kh,
                height_tiles,
                square_pixels,
                total,
                buf,
                xorb_ranges,
                tableau,
                dtype_ranges,
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

// `render_one_arch` lives in `modelweightvis::leaf::ArchRegionsRenderer::render`
// (step 12e). arbvis no longer needs the architectural render dispatch.

/// Pick the actual leaf tile format given the user's request and the render
/// mode. When the user asked for AVIF and the mode produces ≤256 distinct
/// colors per tile (Plain / Dtype), indexed-PNG beats lossless AVIF
/// substantially (AV1 isn't tuned for palette content). Xet-mode tiles can
/// exceed 256 colors so they stay on AVIF; truecolor PNG passes through
/// unchanged.
pub(super) fn derive_leaf_format(user_choice: TileFormat, mode: &LeafMode) -> TileFormat {
    match (user_choice, mode.is_palette_safe()) {
        (TileFormat::Avif { .. }, true) => TileFormat::IndexedPng,
        (fmt, _) => fmt,
    }
}

/// Run the tiled/pyramidal output pipeline to a local directory.
///
/// `leaf_format` controls how each leaf tile is encoded; the actual format
/// may be upgraded to `IndexedPng` when the render mode produces ≤256-color
/// tiles (see [`derive_leaf_format`]). `pyramid_format` controls the
/// downsampled levels (default AVIF q≈85, since averaged pixels already
/// smudge the palette). The pyramid is built in memory by the streaming
/// accumulator as leaves complete, so we never re-decode tiles back from
/// disk — which is also why this code path doesn't need a decoder for AVIF
/// tiles.
/// One scene's worth of sources, peeled off `Vec<Source>` by [`partition_scenes`].
pub(super) struct SceneGroup {
    /// `Some(key)` → tiles go under `tiles/<key>/`; `None` → legacy lone scene.
    pub(super) key: Option<String>,
    pub(super) label: String,
    pub(super) order: u32,
    pub(super) sources: Vec<Source>,
    pub(super) total: u64,
}

/// Group sources into scenes by their [`SceneTag`]. With no tags present, the
/// whole input is one implicit default scene (`key: None`) carrying the
/// caller's original `total` — preserving the exact legacy single-pyramid path.
/// With tags present, sources are bucketed by `key` (first-seen order, then
/// sorted by `order`); each scene's `total` is the sum of its source sizes.
pub(super) fn partition_scenes(sources: Vec<Source>, total: u64) -> Vec<SceneGroup> {
    let any_tagged = sources
        .iter()
        .any(|s| s.extensions.get::<SceneTag>().is_some());
    if !any_tagged {
        return vec![SceneGroup {
            key: None,
            label: String::new(),
            order: 0,
            sources,
            total,
        }];
    }

    let mut key_order: Vec<String> = Vec::new();
    let mut buckets: HashMap<String, (String, u32, Vec<Source>)> = HashMap::new();
    for s in sources {
        let (key, label, order) = match s.extensions.get::<SceneTag>() {
            Some(t) => (t.key.clone(), t.label.clone(), t.order),
            // Untagged source in an otherwise-tagged run: bucket it into a
            // sensible default scene rather than dropping it.
            None => ("main".to_string(), "Main".to_string(), u32::MAX),
        };
        let entry = buckets.entry(key.clone()).or_insert_with(|| {
            key_order.push(key.clone());
            (label, order, Vec::new())
        });
        entry.2.push(s);
    }

    let mut groups: Vec<SceneGroup> = key_order
        .into_iter()
        .map(|k| {
            let (label, order, srcs) = buckets.remove(&k).unwrap();
            let total = srcs.iter().map(|s| s.byte_size).sum();
            SceneGroup {
                key: Some(k),
                label,
                order,
                sources: srcs,
                total,
            }
        })
        .collect();
    groups.sort_by_key(|g| g.order);
    groups
}

pub async fn run_tiles(
    sources: Vec<Source>,
    total: u64,
    tile_dir: PathBuf,
    diff_mode: bool,
    title: &str,
    inputs: &[String],
    show_xet_xorbs: bool,
    leaf_format: TileFormat,
    pyramid_format: TileFormat,
    layout_mode: LayoutMode,
    registry: &crate::registry::Registry,
) -> anyhow::Result<()> {
    let scenes = partition_scenes(sources, total);
    let mut views: Vec<html::SceneView> = Vec::with_capacity(scenes.len());
    for group in scenes {
        let subdir = match &group.key {
            Some(k) => format!("tiles/{k}"),
            None => "tiles".to_string(),
        };
        let view = render_scene_to_disk(
            &tile_dir,
            &subdir,
            group,
            diff_mode,
            show_xet_xorbs,
            leaf_format,
            pyramid_format,
            layout_mode,
            registry,
        )
        .await?;
        views.push(view);
    }

    log::info!("Writing HTML viewer...");
    // The lone implicit scene takes the legacy single-layer viewer verbatim
    // (byte-identical output); anything tagged gets the multi-scene switcher.
    if views.len() == 1 && views[0].key.is_none() {
        let v = &views[0];
        html::write_leaflet_html(
            &tile_dir,
            v.world_w,
            v.world_h,
            v.max_zoom,
            v.detail_depth,
            v.height,
            v.width,
            TILE,
            &v.entities,
            title,
            inputs,
            &v.leaf_ext,
            &v.pyramid_ext,
            &registry.branding,
        )?;
    } else {
        html::write_leaflet_html_multi(&tile_dir, &views, title, inputs, &registry.branding)?;
    }

    log::info!("Tiled output written to {}", tile_dir.display());
    Ok(())
}

/// Render one scene's full pyramid (overview + detail levels) to disk under
/// `<tile_dir>/<subdir>/…` and return its [`html::SceneView`]. `subdir` is
/// `"tiles"` for the legacy lone scene or `"tiles/<key>"` for a named scene —
/// it's the only thing that distinguishes the two paths on disk.
#[allow(clippy::too_many_arguments)]
async fn render_scene_to_disk(
    tile_dir: &Path,
    subdir: &str,
    group: SceneGroup,
    diff_mode: bool,
    show_xet_xorbs: bool,
    leaf_format: TileFormat,
    pyramid_format: TileFormat,
    layout_mode: LayoutMode,
    registry: &crate::registry::Registry,
) -> anyhow::Result<html::SceneView> {
    let SceneGroup {
        key,
        label,
        order,
        sources,
        total,
    } = group;

    let plan = build_tile_plan(
        sources,
        total,
        diff_mode,
        show_xet_xorbs,
        layout_mode,
        registry,
    )
    .await?;

    let leaf_format = derive_leaf_format(leaf_format, &plan.mode);
    let max_zoom = plan.max_zoom;
    let total_tiles = plan.total_tiles;
    let leaf_ext = leaf_format.extension();
    let pyramid_ext = pyramid_format.extension();

    std::fs::create_dir_all(tile_dir.join(format!("{subdir}/{max_zoom}")))?;

    log::info!(
        "Rendering {} leaf tiles for {} ({} leaf / {} pyramid)...",
        total_tiles,
        subdir,
        leaf_ext,
        pyramid_ext
    );

    let sink = Arc::new(LocalFileSink {
        root: tile_dir.to_path_buf(),
    });
    let pyramid_path_fn: Arc<dyn Fn(u32, u32, u32) -> String + Send + Sync> = {
        let ext = pyramid_ext.to_string();
        let subdir = subdir.to_string();
        Arc::new(move |z, x, y| format!("{subdir}/{z}/{x}/{y}.{ext}"))
    };
    let pyramid = Arc::new(PyramidAccumulator::new(
        TILE,
        max_zoom,
        sink.clone(),
        pyramid_path_fn,
        pyramid_format,
    ));

    let tile_dir_for_write = tile_dir.to_path_buf();
    let subdir_for_write = subdir.to_string();
    let pyramid_for_write = pyramid.clone();
    drive_pipeline(
        &plan,
        leaf_format,
        max_zoom,
        TileCoords::Dense {
            width_tiles: plan.width_tiles,
            height_tiles: plan.height_tiles,
        },
        move |t: EncodedTile| {
            let path = tile_dir_for_write.join(format!(
                "{subdir_for_write}/{max_zoom}/{}/{}.{leaf_ext}",
                t.tx, t.ty
            ));
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating tile dir {}", parent.display()))?;
            }
            std::fs::write(&path, &t.bytes)
                .with_context(|| format!("writing tile {}", path.display()))?;
            pyramid_for_write.contribute(max_zoom, t.tx, t.ty, &t.image);
            Ok(())
        },
    )
    .await?;

    log::info!("Draining pyramid encode tasks...");
    pyramid.drain().await;
    drop(pyramid);

    // Variable-depth detail tiles: sparse deeper levels rendered directly from
    // source over the shrunk tensors' footprints (no pyramid accumulation).
    let detail_dir = tile_dir.to_path_buf();
    let subdir_for_detail = subdir.to_string();
    render_detail_levels(&plan, leaf_format, &move |t: &EncodedTile, z| {
        let path = detail_dir.join(format!(
            "{subdir_for_detail}/{z}/{}/{}.{leaf_ext}",
            t.tx, t.ty
        ));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &t.bytes)?;
        Ok(())
    })
    .await?;

    Ok(html::SceneView {
        key,
        label,
        order,
        world_w: plan.world_w,
        world_h: plan.world_h,
        max_zoom,
        detail_depth: plan.detail_depth,
        height: plan.height,
        width: plan.width,
        leaf_ext: leaf_ext.to_string(),
        pyramid_ext: pyramid_ext.to_string(),
        entities: plan.entities,
    })
}

// Streaming tile output (`run_tiles_hf_streaming`) lives in the
// [`streaming`](crate::tiled::streaming) submodule. It uses the same plan and
// pipeline helpers above; keeping it in its own file makes the "off-by-default"
// path easy to find and easy to delete if it's ever superseded.

#[cfg(test)]
mod scene_tests {
    use super::partition_scenes;
    use crate::data::{Extensions, SceneTag, Source, SourceKind};

    fn src(byte_size: u64, scene: Option<(&str, u32)>) -> Source {
        let mut extensions = Extensions::default();
        if let Some((key, order)) = scene {
            extensions.insert(SceneTag {
                key: key.to_string(),
                label: key.to_string(),
                order,
            });
        }
        Source {
            file_idx: 0,
            kind: SourceKind::Buffered(Vec::new()),
            byte_size,
            name_override: Some("t".to_string()),
            xet_terms: None,
            extensions,
        }
    }

    #[test]
    fn untagged_sources_form_one_default_scene_with_original_total() {
        let groups = partition_scenes(vec![src(10, None), src(20, None)], 999);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].key.is_none());
        // The lone default scene keeps the caller's `total` verbatim so the
        // legacy single-pyramid path is byte-for-byte unchanged.
        assert_eq!(groups[0].total, 999);
        assert_eq!(groups[0].sources.len(), 2);
    }

    #[test]
    fn tagged_sources_split_into_ordered_scenes_with_summed_totals() {
        // Deliberately interleaved and out-of-order to exercise grouping + sort.
        let groups = partition_scenes(
            vec![
                src(3, Some(("cka", 1))),
                src(10, Some(("summary", 0))),
                src(5, Some(("cka", 1))),
                src(20, Some(("summary", 0))),
            ],
            0,
        );
        assert_eq!(groups.len(), 2);
        // Sorted by `order`: summary (0) before cka (1).
        assert_eq!(groups[0].key.as_deref(), Some("summary"));
        assert_eq!(groups[0].total, 30);
        assert_eq!(groups[0].sources.len(), 2);
        assert_eq!(groups[1].key.as_deref(), Some("cka"));
        assert_eq!(groups[1].total, 8);
        assert_eq!(groups[1].sources.len(), 2);
    }
}
