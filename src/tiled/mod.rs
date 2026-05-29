pub mod html;
pub mod leaf;
pub mod leaf_arch;
pub mod pyramid_accum;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_channel::{bounded, Receiver, Sender};

use indicatif::ProgressBar;

use crate::color::{build_diff_signed_lut, build_pixel_lut};
use crate::data::{load_source_data, Data, Source, SourceKind};
use crate::format::DiffFill;
use crate::geometry::{file_rects, hilbert_to_xy_u64, name_hue, outer_segments, rects_centroid};
use crate::hf_upload::HfTileSink;
use crate::hf_url::HfOutputSpec;
use crate::layout::{select_layout, Layout, LayoutMode};
use crate::progress::{counter_style, multi, queue_style, status_style};
use crate::throttle::{Throttle, MAX_FETCH_WORKERS};
use crate::tiled::html::{generate_leaflet_content, FileEntity};
use crate::tiled::leaf::{
    load_tile_bytes, render_leaf_tile_diff, render_leaf_tile_dtype, render_leaf_tile_from_buf,
    render_leaf_tile_xet_dtype_from_buf, render_leaf_tile_xet_from_buf, TileFormat, TILE,
    TILE_LOG2, TILE_PIXELS,
};
use crate::tiled::leaf_arch::{
    load_arch_tile_regions, render_arch_tile_diff, render_arch_tile_dtype, render_arch_tile_plain,
    render_arch_tile_xet, render_arch_tile_xet_dtype, LoadedArchTile,
};
use crate::tiled::pyramid_accum::{LocalFileSink, PyramidAccumulator, TileSink};
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
enum TileCoords {
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
struct LoadedTile {
    tx: u32,
    ty: u32,
    /// `Some` when the mode needs raw byte data AND the layout is the legacy
    /// Hilbert curve (one fixed 256 KiB buffer per tile); `None` for dtype
    /// mode or architectural mode (which uses `arch_tile` instead).
    tile_buf: Option<Box<[u8; TILE_PIXELS]>>,
    /// `Some` when the layout is architectural; carries per-region byte
    /// slices for the (typically O(tens)) tensors that intersect this tile.
    arch_tile: Option<LoadedArchTile>,
}

struct EncodedTile {
    tx: u32,
    ty: u32,
    image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    bytes: Vec<u8>,
}

/// Which leaf render to run.
#[derive(Clone)]
enum LeafMode {
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

/// Regenerate `index.html` for an existing tiles directory without re-rendering tiles.
pub fn regen_html(tile_dir: &Path) -> anyhow::Result<()> {
    let tiles_dir = tile_dir.join("tiles");

    // Read labels.json first. Newer outputs persist `max_zoom`/`detail_depth` so
    // we can tell the dense overview levels apart from the sparse variable-depth
    // detail levels — without that, the deepest *detail* zoom dir would be
    // mistaken for the overview leaf and corrupt every derived dimension.
    let labels_path = tile_dir.join("labels.json");
    let json_str = std::fs::read_to_string(&labels_path)
        .with_context(|| format!("cannot read {}", labels_path.display()))?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str)?;
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

    let entities: Vec<html::FileEntity> = values
        .into_iter()
        .map(|v| {
            let name = v["name"].as_str().unwrap_or("").to_string();
            let pixel_x = v["x"].as_u64().unwrap_or(0) as u32;
            let pixel_y = v["y"].as_u64().unwrap_or(0) as u32;
            let hue = v["hue"].as_u64().unwrap_or(0) as u16;
            let byte_size = v["size"].as_u64().unwrap_or(0);
            let bbox = {
                let b = v["bbox"].as_array();
                if let Some(b) = b {
                    let g = |i: usize| b.get(i).and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                    (g(0), g(1), g(2), g(3))
                } else {
                    (0, 0, 0, 0)
                }
            };
            let segments = {
                if let Some(segs) = v["segs"].as_array() {
                    segs.iter()
                        .filter_map(|s| {
                            let arr = s.as_array()?;
                            let g = |i: usize| arr.get(i)?.as_u64().map(|x| x as u32);
                            Some((g(0)?, g(1)?, g(2)?, g(3)?))
                        })
                        .collect()
                } else {
                    vec![]
                }
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
        })
        .collect();

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
        "arbvis",
        &[],
        &leaf_ext,
        &pyramid_ext,
    )?;
    log::info!(
        "Regenerated index.html in {} (zoom 0–{max_zoom}, +{detail_depth} detail, {width_tiles}×{height_tiles} tiles, height={height})",
        tile_dir.display()
    );
    Ok(())
}

/// Shared geometry / entity / mode computation for both `run_tiles` and
/// `run_tiles_hf_streaming`. Holds everything needed to drive the pipeline.
struct TilePlan {
    kh: u8,
    width_tiles: u32,
    height_tiles: u32,
    world_w: u32,
    world_h: u32,
    height: u32,
    width: u32,
    max_zoom: u32,
    /// Extra zoom levels carrying variable-depth detail (0 for Hilbert / no
    /// shrunk tensors). Mirrors `ArchLayout::detail_depth`.
    detail_depth: u32,
    total_tiles: u64,
    square_pixels: u64,
    total: u64,
    mode: LeafMode,
    source_data: Arc<Vec<Data>>,
    cumulative_offsets: Arc<Vec<u64>>,
    entities: Vec<FileEntity>,
    layout: Arc<Layout>,
}

async fn build_tile_plan(
    sources: Vec<Source>,
    total: u64,
    diff_mode: bool,
    show_xet_xorbs: bool,
    layout_mode: LayoutMode,
) -> anyhow::Result<TilePlan> {
    let mut s = 2 * TILE_LOG2 as u32;
    while (1u64 << s) < total {
        s += 1;
    }
    let kh = s / 2;
    let kw = s.div_ceil(2);
    let height = 1u32 << kh;
    let width = 1u32 << kw;
    let tile_size = TILE;
    let max_zoom = kh - TILE_LOG2 as u32;
    let width_tiles = width / tile_size;
    let height_tiles = height / tile_size;
    let world_w = TILE << (kw - kh);
    // Hilbert canvases are always width≥height (kw ≥ kh), so the smaller axis
    // collapses to a single TILE in geo space — i.e. world_h == TILE.
    let world_h: u32 = TILE;
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

    let has_safetensors = !diff_mode && sources.iter().any(|s| s.model_info.is_some());
    let dtype_mode = has_safetensors && !xet_mode;
    let combined_dtype_ranges: Vec<(u64, u64, image::Rgb<u8>)> = if has_safetensors {
        let mut ranges = Vec::new();
        let mut cumulative: u64 = 0;
        for source in &sources {
            if let Some(st) = &source.model_info {
                for &(start, end, color) in &st.color_ranges {
                    ranges.push((cumulative + start, cumulative + end, color));
                }
            }
            cumulative += source.byte_size;
        }
        ranges
    } else {
        vec![]
    };

    let mut entities: Vec<FileEntity> = Vec::new();
    {
        let mut cumulative: u64 = 0;
        for source in &sources {
            if let Some(st) = &source.model_info {
                for tensor in &st.tensors {
                    let t_start = cumulative + tensor.file_start;
                    let t_end = cumulative + tensor.file_end;
                    if t_end <= t_start {
                        continue;
                    }
                    let rects = file_rects(
                        t_start,
                        t_end,
                        total_pixels,
                        square_pixels,
                        num_squares,
                        height,
                        kh as u8,
                    );
                    let (pixel_x, pixel_y) = rects_centroid(&rects).unwrap_or_else(|| {
                        let mid = t_start + (t_end - t_start) / 2;
                        let sq = mid / square_pixels;
                        let (lx, ly) = hilbert_to_xy_u64(mid % square_pixels, kh as u8);
                        (sq as u32 * height + lx, ly)
                    });
                    let name = tensor.label();
                    let hue = name_hue(&tensor.name);
                    let segments = outer_segments(&rects);
                    let bbox = rects
                        .first()
                        .map(|&first| {
                            rects.iter().skip(1).fold(
                                first,
                                |(x0, y0, x1, y1), &(rx0, ry0, rx1, ry1)| {
                                    (x0.min(rx0), y0.min(ry0), x1.max(rx1), y1.max(ry1))
                                },
                            )
                        })
                        .unwrap_or((0, 0, 0, 0));
                    entities.push(FileEntity {
                        name,
                        pixel_x,
                        pixel_y,
                        hue,
                        byte_size: t_end - t_start,
                        bbox,
                        segments,
                    });
                }
            } else {
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
                        rects.iter().skip(1).fold(
                            first,
                            |(x0, y0, x1, y1), &(rx0, ry0, rx1, ry1)| {
                                (x0.min(rx0), y0.min(ry0), x1.max(rx1), y1.max(ry1))
                            },
                        )
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
            }
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

    let mode = if xet_mode && has_safetensors {
        LeafMode::XetDtype {
            xorb_ranges: Arc::new(xorb_map.global_ranges),
            tableau: Arc::new(tableau),
            dtype_ranges: Arc::new(combined_dtype_ranges),
        }
    } else if xet_mode {
        LeafMode::Xet {
            pixel_lut: pixel_lut.clone(),
            xorb_ranges: Arc::new(xorb_map.global_ranges),
            tableau: Arc::new(tableau),
        }
    } else if dtype_mode {
        LeafMode::Dtype {
            ranges: Arc::new(combined_dtype_ranges),
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

    // Opportunistically fetch config.json / model.safetensors.index.json
    // for each source. These are advisory — `select_layout` falls back to
    // pure tensor-name inference when the sidecars are absent.
    let metas = if matches!(layout_mode, LayoutMode::Hilbert) {
        Vec::new()
    } else {
        crate::data::load_meta_for_sources(&sources).await
    };
    if let Some(arch_summary) = metas
        .iter()
        .find_map(|m| m.config.as_ref().map(|c| c.summary()))
    {
        log::info!("model config: {arch_summary}");
    }

    // Construct the layout. For Hilbert (the legacy path) we already have
    // every field — we wrap them so downstream code can take `&Layout`.
    // For Architectural we let the layout module rebuild the grid, then
    // override `width_tiles`, `height_tiles`, `max_zoom`, `total_tiles`,
    // and `world_w`/`height` with the layout-derived values.
    let layout = select_layout(
        &sources,
        &cumulative_offsets,
        total,
        layout_mode,
        &metas,
        diff_mode,
    );

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
        arch_entities,
    ) = match &layout {
        Layout::HilbertGlobal(_) => (
            kh as u8,
            width_tiles,
            height_tiles,
            world_w,
            world_h,
            height,
            width,
            max_zoom,
            width_tiles as u64 * height_tiles as u64,
            square_pixels,
            total,
            None,
        ),
        Layout::Architectural(a) => {
            // Build per-tensor entities directly from the architectural layout.
            // Skip Hilbert-decomposition: each entity is a single rect.
            let mut ents: Vec<FileEntity> = Vec::with_capacity(a.tensors.len());
            for t in &a.tensors {
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
            // For architectural the `kh`/`square_pixels`/`total` Hilbert
            // fields are unused at render time; populate them with safe
            // values that won't divide by zero if accidentally read.
            //
            // `world_w`/`world_h` are the geographic extents at zoom 0 in
            // leaflet's coordinate system. At leaf zoom (`max_zoom`) the tile
            // grid is `width_tiles × height_tiles`; halving each step down to
            // zoom 0 leaves `width_tiles / 2^max_zoom × height_tiles / 2^max_zoom`
            // tiles, each TILE px wide in geo space. `try_build` chose
            // `max_zoom = log2(min(w_p2, h_p2))` so exactly one of the two
            // ratios collapses to 1 — that's the Hilbert-style "fix the smaller
            // axis at TILE, scale the other" convention generalised to either
            // aspect.
            let two_pow_mz = 1u32 << a.max_zoom;
            let arch_world_w = (a.width_tiles / two_pow_mz.max(1)).max(1) * TILE;
            let arch_world_h = (a.height_tiles / two_pow_mz.max(1)).max(1) * TILE;
            (
                /* kh */ 0u8,
                a.width_tiles,
                a.height_tiles,
                arch_world_w,
                arch_world_h,
                a.height,
                a.width,
                a.max_zoom,
                a.total_tiles,
                /* square_pixels */ 1u64,
                /* total */ a.width as u64 * a.height as u64,
                Some(ents),
            )
        }
    };

    let entities = arch_entities.unwrap_or(entities);

    let detail_depth = match &layout {
        Layout::Architectural(a) => a.detail_depth,
        Layout::HilbertGlobal(_) => 0,
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
        layout: Arc::new(layout),
    })
}

/// Drive the four-stage tile pipeline:
///   coord enumerator → N load workers (throttled when HTTP) → num_cpus render
///   workers → write closure (caller-supplied).
///
/// The caller's `on_tile` closure is invoked sequentially (the pipeline keeps
/// a single write task draining the encoded-tile channel) so it can mutate
/// shared state freely.
async fn drive_pipeline<W>(
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
    let progress = Arc::new(PipelineProgress::new(
        coords.len() as u64,
        cap,
        MAX_FETCH_WORKERS,
    ));
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
    let needs_bytes = plan.mode.needs_bytes();
    let any_remote_source = plan.source_data.iter().any(|d| !d.is_local());
    let is_arch = plan.layout.is_architectural();
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
        let zoom = zoom;
        load_handles.push(tokio::spawn(async move {
            while let Ok((tx, ty)) = coord_rx.recv().await {
                let (tile_buf, arch_tile) = if is_arch {
                    // Architectural mode: fetch one coalesced range per
                    // tensor intersecting this tile.
                    let permit = if any_remote_source {
                        Some(Throttle::global().acquire().await)
                    } else {
                        None
                    };
                    let arch_layout = match layout.as_ref() {
                        Layout::Architectural(a) => a,
                        _ => unreachable!("is_arch && layout != Architectural"),
                    };
                    let result = load_arch_tile_regions(
                        zoom,
                        tx,
                        ty,
                        arch_layout,
                        &source_data,
                        &cumulative_offsets,
                    )
                    .await;
                    drop(permit);
                    match result {
                        Ok(at) => {
                            if any_remote_source {
                                Throttle::global().record_success();
                            }
                            (None, Some(at))
                        }
                        Err(e) => {
                            log::error!("load_arch_tile_regions({tx},{ty}) failed:\n{e:?}");
                            coord_rx.close();
                            return Err::<(), anyhow::Error>(e);
                        }
                    }
                } else if needs_bytes {
                    let permit = if any_remote_source {
                        Some(Throttle::global().acquire().await)
                    } else {
                        None
                    };
                    let result = load_tile_bytes(
                        tx,
                        ty,
                        kh,
                        height_tiles,
                        square_pixels,
                        total,
                        &source_data,
                        &cumulative_offsets,
                    )
                    .await;
                    drop(permit);
                    match result {
                        Ok(buf) => {
                            if any_remote_source {
                                Throttle::global().record_success();
                            }
                            (Some(buf), None)
                        }
                        Err(e) => {
                            // Fatal: the throttle's per-call retry already
                            // covered transient HTTP issues; anything reaching
                            // here is a permanent failure. `{e:?}` (anyhow's
                            // Debug) prints the full caused-by chain plus the
                            // captured backtrace (RUST_BACKTRACE is set on by
                            // main), so the user sees where it originated and
                            // what wrapped it — not just the topmost context.
                            log::error!("load_tile_bytes({tx},{ty}) failed:\n{e:?}");
                            // Close the coord channel so the other 127
                            // workers see Err once the ~20-entry buffer
                            // drains, instead of grinding through tens of
                            // thousands more tiles after a fatal error.
                            coord_rx.close();
                            return Err::<(), anyhow::Error>(e);
                        }
                    }
                } else {
                    (None, None)
                };
                loaded_count.fetch_add(1, Ordering::Relaxed);
                if loaded_tx
                    .send(LoadedTile {
                        tx,
                        ty,
                        tile_buf,
                        arch_tile,
                    })
                    .await
                    .is_err()
                {
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
        let is_arch_local = is_arch;
        process_handles.push(tokio::spawn(async move {
            while let Ok(tile) = loaded_rx.recv().await {
                let mode = mode.clone();
                let result = tokio::task::spawn_blocking(move || {
                    if is_arch_local {
                        render_one_arch(tile, &mode, leaf_format)
                    } else {
                        render_one(
                            tile,
                            &mode,
                            kh,
                            height_tiles,
                            square_pixels,
                            total,
                            leaf_format,
                        )
                    }
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

/// Tile coords (at zoom `zoom = max_zoom + k`) covered by the footprints of the
/// tensors that still carry genuine detail at *this* level — i.e. those whose
/// own `detail_depth_for_scale(scale) >= k`. A tensor is dropped from the level
/// once it has resolved to ≥1px/element, so a mildly-shrunk matrix (which
/// resolves at k=1) is NOT re-rendered as pure replication at the deeper levels
/// a vocab embedding needs — without this guard a model with many shrunk
/// tensors (hidden_size > CAP_HI) would emit millions of redundant detail tiles.
/// Deduped across tensors. Non-shrunk tensors that fall in a selected tile still
/// render (replicated), so each detail tile carries the complete scene at higher
/// resolution and overlays the base layer seamlessly.
fn detail_coords(layout: &crate::layout::arch::ArchLayout, zoom: u32) -> Vec<(u32, u32)> {
    use std::collections::BTreeSet;
    let level = zoom.saturating_sub(layout.max_zoom); // 1-based detail level
    let f = 1u64 << level;
    let t_sz = TILE as u64;
    let mut set: BTreeSet<(u32, u32)> = BTreeSet::new();
    for t in &layout.tensors {
        if crate::layout::arch::detail_depth_for_scale(t.scale) < level {
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
async fn render_detail_levels<F>(
    plan: &TilePlan,
    leaf_format: TileFormat,
    write_tile: &F,
) -> anyhow::Result<()>
where
    F: Fn(&EncodedTile, u32) -> anyhow::Result<()> + Sync,
{
    let arch = match plan.layout.as_ref() {
        Layout::Architectural(a) => a,
        _ => return Ok(()),
    };
    if arch.detail_depth == 0 {
        return Ok(());
    }
    // Detail tiles are an enhancement layer: where they're missing the viewer
    // falls back to upsampling the base overview (transparent errorTileUrl). So
    // a detail-pass failure is logged and ends detail rendering, but is NOT
    // propagated — the already-complete overview output (and, for the HF path,
    // the whole staged upload) must not be discarded over one bad detail tile.
    for k in 1..=arch.detail_depth {
        let zoom = arch.max_zoom + k;
        let coords = detail_coords(arch, zoom);
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

fn render_one(
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
        arch_tile: _,
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

/// Architectural-layout render dispatch. Same set of `LeafMode` variants;
/// each routes to a `leaf_arch::render_arch_tile_*` rather than the
/// byte-Hilbert renderer.
fn render_one_arch(
    tile: LoadedTile,
    mode: &LeafMode,
    fmt: TileFormat,
) -> Result<EncodedTile, String> {
    let LoadedTile {
        tx,
        ty,
        tile_buf: _,
        arch_tile,
    } = tile;
    let at = arch_tile.unwrap_or_default();
    let (image, bytes) = match mode {
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

/// Pick the actual leaf tile format given the user's request and the render
/// mode. When the user asked for AVIF and the mode produces ≤256 distinct
/// colors per tile (Plain / Dtype), indexed-PNG beats lossless AVIF
/// substantially (AV1 isn't tuned for palette content). Xet-mode tiles can
/// exceed 256 colors so they stay on AVIF; truecolor PNG passes through
/// unchanged.
fn derive_leaf_format(user_choice: TileFormat, mode: &LeafMode) -> TileFormat {
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
) -> anyhow::Result<()> {
    let plan = build_tile_plan(sources, total, diff_mode, show_xet_xorbs, layout_mode).await?;

    let leaf_format = derive_leaf_format(leaf_format, &plan.mode);
    let max_zoom = plan.max_zoom;
    let world_w = plan.world_w;
    let world_h = plan.world_h;
    let height = plan.height;
    let width = plan.width;
    let total_tiles = plan.total_tiles;
    let leaf_ext = leaf_format.extension();
    let pyramid_ext = pyramid_format.extension();

    std::fs::create_dir_all(tile_dir.join(format!("tiles/{max_zoom}")))?;

    log::info!(
        "Rendering {} leaf tiles ({} leaf / {} pyramid)...",
        total_tiles,
        leaf_ext,
        pyramid_ext
    );

    let sink = Arc::new(LocalFileSink {
        root: tile_dir.clone(),
    });
    let pyramid_path_fn: Arc<dyn Fn(u32, u32, u32) -> String + Send + Sync> = {
        let ext = pyramid_ext.to_string();
        Arc::new(move |z, x, y| format!("tiles/{z}/{x}/{y}.{ext}"))
    };
    let pyramid = Arc::new(PyramidAccumulator::new(
        TILE,
        max_zoom,
        sink.clone(),
        pyramid_path_fn,
        pyramid_format,
    ));

    let tile_dir_for_write = tile_dir.clone();
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
            let path =
                tile_dir_for_write.join(format!("tiles/{max_zoom}/{}/{}.{leaf_ext}", t.tx, t.ty));
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
    let detail_dir = tile_dir.clone();
    render_detail_levels(&plan, leaf_format, &move |t: &EncodedTile, z| {
        let path = detail_dir.join(format!("tiles/{z}/{}/{}.{leaf_ext}", t.tx, t.ty));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &t.bytes)?;
        Ok(())
    })
    .await?;

    log::info!("Writing HTML viewer...");
    html::write_leaflet_html(
        &tile_dir,
        world_w,
        world_h,
        max_zoom,
        plan.detail_depth,
        height,
        width,
        TILE,
        &plan.entities,
        title,
        inputs,
        leaf_ext,
        pyramid_ext,
    )?;

    log::info!("Tiled output written to {}", tile_dir.display());
    Ok(())
}

/// Run the tiled/pyramidal output pipeline, streaming tiles directly to HuggingFace Hub.
pub async fn run_tiles_hf_streaming(
    sources: Vec<Source>,
    total: u64,
    hf_out: &HfOutputSpec,
    diff_mode: bool,
    title: &str,
    inputs: &[String],
    show_xet_xorbs: bool,
    leaf_format: TileFormat,
    pyramid_format: TileFormat,
    layout_mode: LayoutMode,
) -> anyhow::Result<Vec<u8>> {
    crate::hf_url::require_token()?;
    let client = crate::hf_url::client()?;

    let plan = build_tile_plan(sources, total, diff_mode, show_xet_xorbs, layout_mode).await?;
    let leaf_format = derive_leaf_format(leaf_format, &plan.mode);
    let tile_size = TILE;
    let max_zoom = plan.max_zoom;
    let world_w = plan.world_w;
    let world_h = plan.world_h;
    let height = plan.height;
    let width = plan.width;
    let total_tiles = plan.total_tiles;
    let leaf_ext = leaf_format.extension();
    let pyramid_ext = pyramid_format.extension();

    let sink = Arc::new(HfTileSink::new(client, hf_out.clone())?);
    let pyramid_path_fn: Arc<dyn Fn(u32, u32, u32) -> String + Send + Sync> = {
        let hf_out = hf_out.clone();
        let ext = pyramid_ext.to_string();
        Arc::new(move |z, x, y| hf_out.tile_repo_path(z, x, y, &ext))
    };
    let pyramid = Arc::new(PyramidAccumulator::new(
        tile_size,
        max_zoom,
        sink.clone(),
        pyramid_path_fn,
        pyramid_format,
    ));

    log::info!(
        "Rendering and uploading {} leaf tiles ({} leaf / {} pyramid)...",
        total_tiles,
        leaf_ext,
        pyramid_ext
    );

    let sink_for_write = sink.clone();
    let pyramid_for_write = pyramid.clone();
    let hf_out_for_write = hf_out.clone();
    drive_pipeline(
        &plan,
        leaf_format,
        max_zoom,
        TileCoords::Dense {
            width_tiles: plan.width_tiles,
            height_tiles: plan.height_tiles,
        },
        move |t: EncodedTile| {
            let repo_path = hf_out_for_write.tile_repo_path(max_zoom, t.tx, t.ty, leaf_ext);
            sink_for_write.upload_tile(repo_path, t.bytes)?;
            pyramid_for_write.contribute(max_zoom, t.tx, t.ty, &t.image);
            Ok(())
        },
    )
    .await?;

    // Await any in-flight pyramid encode/upload tasks before commit so every
    // staged file is on disk by the time hf-hub takes the snapshot.
    pyramid.drain().await;

    // Variable-depth detail tiles (sparse deeper levels, no accumulation).
    let detail_sink = sink.clone();
    let detail_hf_out = hf_out.clone();
    render_detail_levels(&plan, leaf_format, &move |t: &EncodedTile, z| {
        let repo_path = detail_hf_out.tile_repo_path(z, t.tx, t.ty, leaf_ext);
        detail_sink.upload_tile(repo_path, t.bytes.clone())
    })
    .await?;

    log::info!("Uploading index.html and labels.json...");
    let (html_bytes, labels_bytes) = generate_leaflet_content(
        world_w,
        world_h,
        max_zoom,
        plan.detail_depth,
        height,
        width,
        TILE,
        &plan.entities,
        title,
        inputs,
        leaf_ext,
        pyramid_ext,
    );
    sink.upload_tile(hf_out.index_html_path(), html_bytes.clone())?;
    sink.upload_tile(hf_out.labels_json_path(), labels_bytes)?;

    drop(pyramid);

    log::info!("Creating HF Hub commit...");
    Arc::try_unwrap(sink)
        .map_err(|_| anyhow::anyhow!("unexpected extra Arc reference to tile sink"))?
        .commit("Add arbvis visualization tiles")
        .await?;

    log::info!(
        "Streaming output committed to hf://{}/{}",
        hf_out.repo_id,
        hf_out.path_prefix
    );
    Ok(html_bytes)
}
