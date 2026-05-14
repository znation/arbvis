pub mod html;
pub mod leaf;
pub mod pyramid;
pub mod pyramid_accum;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_channel::{bounded, Receiver, Sender};

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget};

use crate::color::{build_diff_signed_lut, build_pixel_lut};
use crate::data::{load_source_data, Data, Source};
use crate::geometry::{file_rects, hilbert_to_xy_u64, name_hue, outer_segments, rects_centroid};
use crate::hf_upload::HfTileSink;
use crate::hf_url::HfOutputSpec;
use crate::progress::{counter_style, queue_style, status_style};
use crate::throttle::{Throttle, MAX_FETCH_WORKERS};
use crate::tiled::html::{FileEntity, generate_leaflet_content};
use crate::tiled::leaf::{
    load_tile_bytes, render_leaf_tile_dtype, render_leaf_tile_from_buf,
    render_leaf_tile_xet_from_buf, TILE, TILE_LOG2, TILE_PIXELS,
};
use crate::tiled::pyramid::build_pyramid;
use crate::tiled::pyramid_accum::{PyramidAccumulator, TileSink};
use crate::xet::{TABLEAU_20, XorbMap};

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
    multi: MultiProgress,
    throttle: ProgressBar,
    coord_q: ProgressBar,
    loaded: ProgressBar,
    loaded_q: ProgressBar,
    rendered: ProgressBar,
    encoded_q: ProgressBar,
    written: ProgressBar,
}

impl PipelineProgress {
    fn new(total_tiles: u64, queue_cap: usize, throttle_max: usize) -> Option<Self> {
        if !std::io::stderr().is_terminal() {
            return None;
        }
        let multi = MultiProgress::with_draw_target(ProgressDrawTarget::stderr());
        let add = |bar: ProgressBar| multi.add(bar);

        // Throttle bar: pos = current AIMD `active_limit`, len = `max_workers`
        // ceiling (128). The message refreshes from the monitor task every
        // 500 ms with current in-flight count.
        let throttle = add(ProgressBar::new(throttle_max as u64))
            .with_style(status_style())
            .with_message("HTTP workers: 0/0 (in flight: 0)");
        // Queue bars: pos = current depth, len = channel capacity.
        let coord_q = add(ProgressBar::new(queue_cap as u64))
            .with_style(queue_style())
            .with_message("tile coord queue");
        // Counter bars: pos = tiles completed at this stage, len = total tiles.
        let loaded = add(ProgressBar::new(total_tiles))
            .with_style(counter_style())
            .with_message("tile bytes loaded");
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
        for pb in [&throttle, &coord_q, &loaded, &loaded_q, &rendered, &encoded_q, &written] {
            pb.enable_steady_tick(Duration::from_millis(100));
        }

        Some(Self {
            multi,
            throttle,
            coord_q,
            loaded,
            loaded_q,
            rendered,
            encoded_q,
            written,
        })
    }

    fn finish_all(&self) {
        for pb in [
            &self.throttle,
            &self.coord_q,
            &self.loaded,
            &self.loaded_q,
            &self.rendered,
            &self.encoded_q,
            &self.written,
        ] {
            pb.finish();
        }
        // Drop the MultiProgress via clear to release the terminal so any
        // subsequent log lines render cleanly.
        let _ = self.multi.clear();
    }
}

/// Per-tile data flowing through the pipeline after the load stage.
struct LoadedTile {
    tx: u32,
    ty: u32,
    /// `Some` when the mode needs raw byte data (plain or xet); `None` for
    /// dtype mode which doesn't read bytes.
    tile_buf: Option<Box<[u8; TILE_PIXELS]>>,
}

struct EncodedTile {
    tx: u32,
    ty: u32,
    image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    png_bytes: Vec<u8>,
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
}

impl LeafMode {
    /// Whether the fetch stage needs to read bytes for this mode.
    fn needs_bytes(&self) -> bool {
        matches!(self, LeafMode::Plain { .. } | LeafMode::Xet { .. })
    }
}

/// Regenerate `index.html` for an existing tiles directory without re-rendering tiles.
pub fn regen_html(tile_dir: &PathBuf) -> anyhow::Result<()> {
    let tiles_dir = tile_dir.join("tiles");

    let max_zoom = std::fs::read_dir(&tiles_dir)
        .with_context(|| format!("cannot read {}", tiles_dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_string_lossy().parse::<u32>().ok())
        .max()
        .ok_or_else(|| anyhow::anyhow!("no zoom levels found in {}", tiles_dir.display()))?;

    let zoom_dir = tiles_dir.join(format!("{max_zoom}"));
    let width_tiles = std::fs::read_dir(&zoom_dir)
        .with_context(|| format!("cannot read {}", zoom_dir.display()))?
        .filter(|e| e.as_ref().map(|e| e.path().is_dir()).unwrap_or(false))
        .count() as u32;
    let height_tiles = {
        let first_x = std::fs::read_dir(&zoom_dir)?
            .filter_map(|e| e.ok())
            .find(|e| e.path().is_dir())
            .ok_or_else(|| anyhow::anyhow!("no x-dirs found at zoom {max_zoom}"))?;
        std::fs::read_dir(first_x.path())?.count() as u32
    };
    let height = height_tiles * TILE;
    let world_w = (width_tiles / height_tiles.max(1)) * TILE;

    let labels_path = tile_dir.join("labels.json");
    let json_str = std::fs::read_to_string(&labels_path)
        .with_context(|| format!("cannot read {}", labels_path.display()))?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str)?;
    let (values, chunks_for_regen): (Vec<serde_json::Value>, Vec<(u32, u32, u32, u32)>) = match parsed {
        serde_json::Value::Array(a) => (a, Vec::new()),
        serde_json::Value::Object(ref o) => {
            let files = o.get("files").and_then(|v| v.as_array()).cloned().unwrap_or_default();
            let chunks = o.get("chunks").and_then(|v| v.as_array()).map(|arr| {
                arr.iter().filter_map(|s| {
                    let a = s.as_array()?;
                    let g = |i: usize| a.get(i)?.as_u64().map(|x| x as u32);
                    Some((g(0)?, g(1)?, g(2)?, g(3)?))
                }).collect()
            }).unwrap_or_default();
            (files, chunks)
        }
        _ => anyhow::bail!("labels.json: unexpected JSON shape (expected array or object)"),
    };
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
            html::FileEntity { name, pixel_x, pixel_y, hue, byte_size, bbox, segments }
        })
        .collect();

    html::write_leaflet_html(tile_dir, world_w, max_zoom, height, TILE, &entities, "arbvis", &[], &chunks_for_regen)?;
    log::info!(
        "Regenerated index.html in {} (zoom 0–{max_zoom}, {width_tiles}×{height_tiles} tiles, height={height})",
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
    height: u32,
    max_zoom: u32,
    total_tiles: u64,
    square_pixels: u64,
    total: u64,
    mode: LeafMode,
    source_data: Arc<Vec<Data>>,
    cumulative_offsets: Arc<Vec<u64>>,
    entities: Vec<FileEntity>,
    chunk_segments: Vec<(u32, u32, u32, u32)>,
}

async fn build_tile_plan(
    sources: Vec<Source>,
    total: u64,
    diff_mode: bool,
    show_xet_xorbs: bool,
    show_xet_chunks: bool,
) -> anyhow::Result<TilePlan> {
    let mut s = 2 * TILE_LOG2 as u32;
    while (1u64 << s) < total {
        s += 1;
    }
    let kh = s / 2;
    let kw = (s + 1) / 2;
    let height = 1u32 << kh;
    let width = 1u32 << kw;
    let tile_size = TILE;
    let max_zoom = kh - TILE_LOG2 as u32;
    let width_tiles = width / tile_size;
    let height_tiles = height / tile_size;
    let world_w = TILE << (kw - kh);
    let square_pixels: u64 = (height as u64) * (height as u64);
    let total_pixels: u64 = width as u64 * height as u64;
    let num_squares = 1u32 << (kw - kh);

    let pixel_lut = Arc::new(if diff_mode { build_diff_signed_lut() } else { build_pixel_lut() });

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

    // The xorb_map drives leaf coloring (LeafMode::Xet). Only build it when
    // the user explicitly asked for xorb coloring — otherwise `--show-xet-chunks`
    // alone would silently swap dtype/plain coloring for xorb tinting just
    // because xet_terms got populated for the chunk overlay.
    let xorb_map = if show_xet_xorbs {
        XorbMap::build(
            sources.iter().zip(cumulative_offsets.iter()).map(|(s, &off)| {
                (s.xet_terms.as_deref(), off)
            }),
        )
    } else {
        XorbMap { global_ranges: Vec::new() }
    };
    let xet_mode = !xorb_map.is_empty();
    let tableau: [image::Rgb<u8>; 20] = {
        let mut arr = [image::Rgb([0u8, 0, 0]); 20];
        for (i, c) in TABLEAU_20.iter().enumerate() {
            arr[i] = image::Rgb(*c);
        }
        arr
    };

    let dtype_mode = !diff_mode && !xet_mode && sources.iter().any(|s| s.safetensors.is_some());
    let combined_dtype_ranges: Vec<(u64, u64, image::Rgb<u8>)> = if dtype_mode {
        let mut ranges = Vec::new();
        let mut cumulative: u64 = 0;
        for source in &sources {
            if let Some(st) = &source.safetensors {
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

    let chunk_segments: Vec<(u32, u32, u32, u32)> = if show_xet_chunks {
        let mut segs = Vec::new();
        for (src, &off) in sources.iter().zip(cumulative_offsets.iter()) {
            let Some(terms) = src.xet_terms.as_deref() else { continue };
            for t in terms {
                if t.byte_len == 0 { continue; }
                let start = off + t.file_offset;
                let end = start + t.byte_len;
                let rects = file_rects(start, end, total_pixels, square_pixels, num_squares, height, kh as u8);
                segs.extend(outer_segments(&rects));
            }
        }
        segs
    } else {
        Vec::new()
    };

    let mut entities: Vec<FileEntity> = Vec::new();
    {
        let mut cumulative: u64 = 0;
        for source in &sources {
            if let Some(st) = &source.safetensors {
                for tensor in &st.tensors {
                    let t_start = cumulative + tensor.file_start;
                    let t_end = cumulative + tensor.file_end;
                    if t_end <= t_start { continue; }
                    let rects = file_rects(t_start, t_end, total_pixels, square_pixels, num_squares, height, kh as u8);
                    let (pixel_x, pixel_y) = rects_centroid(&rects).unwrap_or_else(|| {
                        let mid = t_start + (t_end - t_start) / 2;
                        let sq = mid / square_pixels;
                        let (lx, ly) = hilbert_to_xy_u64(mid % square_pixels, kh as u8);
                        (sq as u32 * height + lx, ly)
                    });
                    let name = tensor.label();
                    let hue = name_hue(&tensor.name);
                    let segments = outer_segments(&rects);
                    let bbox = rects.first().map(|&first| {
                        rects.iter().skip(1).fold(first, |(x0,y0,x1,y1), &(rx0,ry0,rx1,ry1)| {
                            (x0.min(rx0), y0.min(ry0), x1.max(rx1), y1.max(ry1))
                        })
                    }).unwrap_or((0,0,0,0));
                    entities.push(FileEntity { name, pixel_x, pixel_y, hue, byte_size: t_end - t_start, bbox, segments });
                }
            } else {
                let name = source.name();
                let data_start = cumulative;
                let data_end = cumulative + source.byte_size;
                let rects = file_rects(data_start, data_end, total_pixels, square_pixels, num_squares, height, kh as u8);
                let (pixel_x, pixel_y) = rects_centroid(&rects).unwrap_or_else(|| {
                    let mid = data_start + (data_end - data_start) / 2;
                    let sq = mid / square_pixels;
                    let (lx, ly) = hilbert_to_xy_u64(mid % square_pixels, kh as u8);
                    (sq as u32 * height + lx, ly)
                });
                let hue = name_hue(&name);
                let segments = outer_segments(&rects);
                let bbox = rects.first().map(|&first| {
                    rects.iter().skip(1).fold(first, |(x0,y0,x1,y1), &(rx0,ry0,rx1,ry1)| {
                        (x0.min(rx0), y0.min(ry0), x1.max(rx1), y1.max(ry1))
                    })
                }).unwrap_or((0,0,0,0));
                entities.push(FileEntity { name, pixel_x, pixel_y, hue, byte_size: data_end - data_start, bbox, segments });
            }
            cumulative += source.byte_size;
        }
    }

    let mode = if xet_mode {
        LeafMode::Xet {
            pixel_lut: pixel_lut.clone(),
            xorb_ranges: Arc::new(xorb_map.global_ranges),
            tableau: Arc::new(tableau),
        }
    } else if dtype_mode {
        LeafMode::Dtype { ranges: Arc::new(combined_dtype_ranges) }
    } else {
        LeafMode::Plain { pixel_lut: pixel_lut.clone() }
    };

    Ok(TilePlan {
        kh: kh as u8,
        width_tiles,
        height_tiles,
        world_w,
        height,
        max_zoom,
        total_tiles: width_tiles as u64 * height_tiles as u64,
        square_pixels,
        total,
        mode,
        source_data: Arc::new(source_data),
        cumulative_offsets: Arc::new(cumulative_offsets),
        entities,
        chunk_segments,
    })
}

/// Drive the four-stage tile pipeline:
///   coord enumerator → N load workers (throttled when HTTP) → num_cpus render
///   workers → write closure (caller-supplied).
///
/// The caller's `on_tile` closure is invoked sequentially (the pipeline keeps
/// a single write task draining the encoded-tile channel) so it can mutate
/// shared state freely.
async fn drive_pipeline<W>(plan: &TilePlan, mut on_tile: W) -> anyhow::Result<()>
where
    W: FnMut(EncodedTile) -> anyhow::Result<()> + Send,
{
    let cap = channel_cap();
    let (coord_tx, coord_rx): (Sender<(u32, u32)>, Receiver<(u32, u32)>) = bounded(cap);
    let (loaded_tx, loaded_rx): (Sender<LoadedTile>, Receiver<LoadedTile>) = bounded(cap);
    let (encoded_tx, encoded_rx): (Sender<EncodedTile>, Receiver<EncodedTile>) = bounded(cap);

    let width_tiles = plan.width_tiles;
    let height_tiles = plan.height_tiles;

    let progress: Option<Arc<PipelineProgress>> =
        PipelineProgress::new(plan.total_tiles, cap, MAX_FETCH_WORKERS).map(Arc::new);
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
                if let Some(prog) = progress.as_ref() {
                    let throttle = Throttle::global();
                    prog.throttle.set_position(throttle.active_limit() as u64);
                    prog.throttle.set_message(format!(
                        "HTTP workers: {}/{} (in flight: {})",
                        throttle.active_limit(),
                        throttle.max_workers(),
                        throttle.in_flight(),
                    ));
                    prog.coord_q.set_position(coord_rx.len() as u64);
                    prog.loaded_q.set_position(loaded_rx.len() as u64);
                    prog.encoded_q.set_position(encoded_rx.len() as u64);
                    prog.loaded.set_position(loaded_count.load(Ordering::Relaxed));
                    prog.rendered.set_position(rendered_count.load(Ordering::Relaxed));
                }
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        })
    };

    // Stage 1: coord enumerator.
    let coord_task = tokio::spawn(async move {
        for ty in 0..height_tiles {
            for tx in 0..width_tiles {
                if coord_tx.send((tx, ty)).await.is_err() {
                    return; // downstream closed
                }
            }
        }
        // closing coord_tx (drop on scope end) signals load workers to drain.
    });

    // Stage 2: load workers. Spawn up to MAX_FETCH_WORKERS; each acquires a
    // throttle permit before reading source bytes (which is HTTP for
    // `Data::Http`, mmap for `Data::Mapped`/`Owned`). Workers above the
    // throttle's `active_limit` park on the throttle's Notify, not on the
    // channel.
    let needs_bytes = plan.mode.needs_bytes();
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
        load_handles.push(tokio::spawn(async move {
            while let Ok((tx, ty)) = coord_rx.recv().await {
                let tile_buf = if needs_bytes {
                    let throttle = Throttle::global();
                    let permit = throttle.acquire().await;
                    let result = load_tile_bytes(
                        tx, ty, kh, height_tiles, square_pixels, total,
                        &source_data, &cumulative_offsets,
                    ).await;
                    drop(permit);
                    match result {
                        Ok(buf) => {
                            throttle.record_success();
                            Some(buf)
                        }
                        Err(e) => {
                            // The throttle's per-call retry already covers
                            // transient HTTP issues; anything reaching here
                            // is a permanent failure or unrecoverable error.
                            log::error!("load_tile_bytes({tx},{ty}) failed: {e}");
                            return Err::<(), anyhow::Error>(e);
                        }
                    }
                } else {
                    None
                };
                loaded_count.fetch_add(1, Ordering::Relaxed);
                if loaded_tx.send(LoadedTile { tx, ty, tile_buf }).await.is_err() {
                    break;
                }
            }
            Ok(())
        }));
    }
    drop(loaded_tx); // close when all load workers exit
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
        process_handles.push(tokio::spawn(async move {
            while let Ok(tile) = loaded_rx.recv().await {
                let mode = mode.clone();
                let result = tokio::task::spawn_blocking(move || {
                    render_one(tile, &mode, kh, height_tiles, square_pixels, total)
                })
                .await;
                let encoded = match result {
                    Ok(Ok(e)) => e,
                    Ok(Err(e)) => return Err::<(), anyhow::Error>(anyhow::anyhow!("{e}")),
                    Err(e) => return Err(anyhow::anyhow!("render join failure: {e}")),
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
    while let Ok(tile) = encoded_rx.recv().await {
        on_tile(tile)?;
        if let Some(prog) = progress.as_ref() {
            prog.written.inc(1);
        }
    }

    // Stop the monitor before awaiting stage handles so the bars don't keep
    // ticking after work finishes.
    shutdown.store(true, Ordering::Relaxed);
    let _ = monitor_handle.await;

    // Surface the first error from any stage.
    let _ = coord_task.await;
    for h in load_handles {
        if let Ok(Err(e)) = h.await { return Err(e); }
    }
    for h in process_handles {
        if let Ok(Err(e)) = h.await { return Err(e); }
    }

    if let Some(prog) = progress.as_ref() {
        prog.finish_all();
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
) -> Result<EncodedTile, String> {
    let LoadedTile { tx, ty, tile_buf } = tile;
    let (image, png_bytes) = match mode {
        LeafMode::Plain { pixel_lut } => {
            let buf = tile_buf.as_deref().expect("plain mode needs tile_buf");
            render_leaf_tile_from_buf(tx, ty, kh, height_tiles, square_pixels, total, buf, pixel_lut)?
        }
        LeafMode::Xet { pixel_lut, xorb_ranges, tableau } => {
            let buf = tile_buf.as_deref().expect("xet mode needs tile_buf");
            render_leaf_tile_xet_from_buf(
                tx, ty, kh, height_tiles, square_pixels, total,
                buf, pixel_lut, xorb_ranges, tableau,
            )?
        }
        LeafMode::Dtype { ranges } => {
            render_leaf_tile_dtype(tx, ty, kh, height_tiles, square_pixels, total, ranges)?
        }
    };
    Ok(EncodedTile { tx, ty, image, png_bytes })
}

/// Run the tiled/pyramidal output pipeline to a local directory.
pub async fn run_tiles(
    sources: Vec<Source>,
    total: u64,
    tile_dir: PathBuf,
    diff_mode: bool,
    title: &str,
    inputs: &[String],
    show_xet_xorbs: bool,
    show_xet_chunks: bool,
) -> anyhow::Result<()> {
    let plan = build_tile_plan(sources, total, diff_mode, show_xet_xorbs, show_xet_chunks).await?;

    let max_zoom = plan.max_zoom;
    let tile_size = TILE;
    let width_tiles = plan.width_tiles;
    let height_tiles = plan.height_tiles;
    let world_w = plan.world_w;
    let height = plan.height;
    let total_tiles = plan.total_tiles;

    std::fs::create_dir_all(tile_dir.join(format!("tiles/{max_zoom}")))?;

    log::info!("Rendering {} leaf tiles...", total_tiles);

    let tile_dir_for_write = tile_dir.clone();
    drive_pipeline(&plan, move |t: EncodedTile| {
        let path = tile_dir_for_write.join(format!("tiles/{max_zoom}/{}/{}.png", t.tx, t.ty));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating tile dir {}", parent.display()))?;
        }
        std::fs::write(&path, &t.png_bytes)
            .with_context(|| format!("writing tile {}", path.display()))?;
        Ok(())
    })
    .await?;

    log::info!("Building tile pyramid ({} zoom levels)...", max_zoom);
    let tiles_path = tile_dir.join("tiles");
    tokio::task::spawn_blocking(move || {
        build_pyramid(&tiles_path, tile_size, max_zoom, width_tiles, height_tiles)
    })
    .await
    .map_err(|e| anyhow::anyhow!("pyramid join failure: {e}"))??;

    log::info!("Writing HTML viewer...");
    html::write_leaflet_html(&tile_dir, world_w, max_zoom, height, TILE, &plan.entities, title, inputs, &plan.chunk_segments)?;

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
    show_xet_chunks: bool,
) -> anyhow::Result<Vec<u8>> {
    crate::hf_url::require_token()?;
    let client = crate::hf_url::client()?;

    let plan = build_tile_plan(sources, total, diff_mode, show_xet_xorbs, show_xet_chunks).await?;
    let tile_size = TILE;
    let max_zoom = plan.max_zoom;
    let world_w = plan.world_w;
    let height = plan.height;
    let total_tiles = plan.total_tiles;

    let sink = Arc::new(HfTileSink::new(client, hf_out.clone())?);
    let pyramid = Arc::new(PyramidAccumulator::new(tile_size, max_zoom, sink.clone(), Arc::new(hf_out.clone())));

    log::info!("Rendering and uploading {} leaf tiles...", total_tiles);

    let sink_for_write = sink.clone();
    let pyramid_for_write = pyramid.clone();
    let hf_out_for_write = hf_out.clone();
    drive_pipeline(&plan, move |t: EncodedTile| {
        let repo_path = hf_out_for_write.tile_repo_path(max_zoom, t.tx, t.ty);
        sink_for_write.upload_tile(repo_path, t.png_bytes)?;
        pyramid_for_write.contribute(max_zoom, t.tx, t.ty, &t.image);
        Ok(())
    })
    .await?;

    // Await any in-flight pyramid encode/upload tasks before commit so every
    // staged file is on disk by the time hf-hub takes the snapshot.
    pyramid.drain().await;

    log::info!("Uploading index.html and labels.json...");
    let (html_bytes, labels_bytes) = generate_leaflet_content(world_w, max_zoom, height, TILE, &plan.entities, title, inputs, &plan.chunk_segments);
    sink.upload_tile(hf_out.index_html_path(), html_bytes.clone())?;
    sink.upload_tile(hf_out.labels_json_path(), labels_bytes)?;

    drop(pyramid);

    log::info!("Creating HF Hub commit...");
    Arc::try_unwrap(sink)
        .map_err(|_| anyhow::anyhow!("unexpected extra Arc reference to tile sink"))?
        .commit("Add arbvis visualization tiles")
        .await?;

    log::info!("Streaming output committed to hf://{}/{}", hf_out.repo_id, hf_out.path_prefix);
    Ok(html_bytes)
}

