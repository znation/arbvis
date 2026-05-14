pub mod html;
pub mod leaf;
pub mod pyramid;
pub mod pyramid_accum;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_channel::{bounded, Receiver, Sender};

use indicatif::{ProgressBar, ProgressStyle};

use crate::color::{build_diff_signed_lut, build_pixel_lut};
use crate::data::{load_source_data, Data, Histogram, Source};
use crate::geometry::{file_rects, hilbert_to_xy_u64, name_hue, outer_segments, rects_centroid};
use crate::hf_upload::HfTileSink;
use crate::hf_url::HfOutputSpec;
use crate::throttle::{Throttle, MAX_FETCH_WORKERS};
use crate::tiled::html::{FileEntity, generate_leaflet_content};
use crate::tiled::leaf::{
    fetch_tile_bytes, render_leaf_tile_dtype, render_leaf_tile_from_buf, render_leaf_tile_sorted,
    render_leaf_tile_xet_from_buf, TILE_PIXELS,
};
use crate::tiled::pyramid::build_pyramid;
use crate::tiled::pyramid_accum::{PyramidAccumulator, TileSink};
use crate::xet::{TABLEAU_20, XorbMap};

/// Channel capacity for the fetch→process queue, per CPU core. Keeps memory
/// bounded — each in-flight tile holds a 64 KiB byte buffer plus a 192 KiB
/// RGB pixel buffer, so a 16-CPU machine caps at ~16 × 2 × 256 KiB ≈ 8 MiB.
const CHANNEL_CAPACITY_PER_CPU: usize = 2;

fn channel_cap() -> usize {
    std::thread::available_parallelism().map_or(8, |n| n.get() * CHANNEL_CAPACITY_PER_CPU)
}

fn num_cpus_for_processing() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// Per-tile data flowing through the pipeline.
struct FetchedTile {
    tx: u32,
    ty: u32,
    /// `Some` when the mode needs raw byte data (plain or xet); `None` for
    /// dtype/sort modes that don't read bytes.
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
    Sort {
        pixel_lut: Arc<[image::Rgb<u8>; 256]>,
        histograms: Arc<Vec<([u64; 257], u64)>>,
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
    let height = height_tiles * 256;
    let world_w = (width_tiles / height_tiles.max(1)) * 256;

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

    html::write_leaflet_html(tile_dir, world_w, max_zoom, height, &entities, "arbvis", &[], &chunks_for_regen)?;
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
    /// Indicates the fetch stage is a no-op for this mode (dtype/sort).
    histograms_for_sort: Option<Arc<Vec<([u64; 257], u64)>>>,
}

async fn build_tile_plan(
    sources: Vec<Source>,
    total: u64,
    sort: bool,
    diff_mode: bool,
    show_xet_chunks: bool,
) -> anyhow::Result<TilePlan> {
    if sort && sources.iter().any(|s| s.safetensors.is_some()) {
        anyhow::bail!(
            "--sort is incompatible with safetensors files: sort reorders bytes by value, \
             which destroys positional dtype information"
        );
    }

    let mut s = 16u32;
    while (1u64 << s) < total {
        s += 1;
    }
    let kh = s / 2;
    let kw = (s + 1) / 2;
    let height = 1u32 << kh;
    let width = 1u32 << kw;
    let tile_size = 256u32;
    let max_zoom = kh - 8;
    let width_tiles = width / tile_size;
    let height_tiles = height / tile_size;
    let world_w = 256u32 << (kw - kh);
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

    // For the sort path: stream each source to build histograms.
    let histograms: Vec<([u64; 257], u64)> = if sort {
        let sort_pb: Option<Arc<ProgressBar>> = if std::io::stderr().is_terminal() {
            let pb = ProgressBar::new(total);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} scanning ({eta})",
                )
                .unwrap()
                .progress_chars("=>-"),
            );
            pb.enable_steady_tick(Duration::from_millis(100));
            Some(Arc::new(pb))
        } else {
            None
        };
        // Histograms are built sequentially under async to avoid contention on
        // the progress bar Arc; each Histogram::build is mostly I/O anyway.
        let mut result = Vec::with_capacity(sources.len());
        for (s, &off) in sources.iter().zip(cumulative_offsets.iter()) {
            let hist = Histogram::build(s, sort_pb.as_deref()).await?;
            result.push((hist.prefix_sums(), off));
        }
        if let Some(ref pb) = sort_pb {
            pb.finish();
        }
        result
    } else {
        vec![]
    };

    // For non-sort: open all source Data handles. `load_source_data` is sync
    // (mmap for local, lightweight handle clone for HTTP/LazyDiff) so a plain
    // loop is fine.
    let source_data: Vec<Data> = if !sort {
        let mut v = Vec::with_capacity(sources.len());
        for s in &sources {
            v.push(load_source_data(s)?);
        }
        v
    } else {
        vec![]
    };

    let xorb_map = XorbMap::build(
        sources.iter().zip(cumulative_offsets.iter()).map(|(s, &off)| {
            (s.xet_terms.as_deref(), off)
        }),
    );
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

    let histograms_arc = if sort { Some(Arc::new(histograms)) } else { None };

    let mode = if xet_mode {
        LeafMode::Xet {
            pixel_lut: pixel_lut.clone(),
            xorb_ranges: Arc::new(xorb_map.global_ranges),
            tableau: Arc::new(tableau),
        }
    } else if dtype_mode {
        LeafMode::Dtype { ranges: Arc::new(combined_dtype_ranges) }
    } else if sort {
        LeafMode::Sort {
            pixel_lut: pixel_lut.clone(),
            histograms: histograms_arc.clone().unwrap(),
        }
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
        histograms_for_sort: histograms_arc,
    })
}

/// Drive the three-stage tile pipeline:
///   coord enumerator → N fetch workers (throttled) → num_cpus process workers
///   → write closure (caller-supplied).
///
/// The caller's `on_tile` closure is invoked sequentially (the pipeline keeps
/// a single write task draining the encoded-tile channel) so it can mutate
/// shared state freely.
async fn drive_pipeline<W>(plan: &TilePlan, pb: Option<Arc<ProgressBar>>, mut on_tile: W) -> anyhow::Result<()>
where
    W: FnMut(EncodedTile) -> anyhow::Result<()> + Send,
{
    let cap = channel_cap();
    let (coord_tx, coord_rx): (Sender<(u32, u32)>, Receiver<(u32, u32)>) = bounded(cap);
    let (fetched_tx, fetched_rx): (Sender<FetchedTile>, Receiver<FetchedTile>) = bounded(cap);
    let (encoded_tx, encoded_rx): (Sender<EncodedTile>, Receiver<EncodedTile>) = bounded(cap);

    let width_tiles = plan.width_tiles;
    let height_tiles = plan.height_tiles;

    // Stage 1: coord enumerator.
    let coord_task = tokio::spawn(async move {
        for ty in 0..height_tiles {
            for tx in 0..width_tiles {
                if coord_tx.send((tx, ty)).await.is_err() {
                    return; // downstream closed
                }
            }
        }
        // closing coord_tx (drop on scope end) signals fetch workers to drain.
    });

    // Stage 2: fetch workers. Spawn up to MAX_FETCH_WORKERS; each acquires a
    // throttle permit before issuing HTTP. Workers above the throttle's
    // `active_limit` park on the throttle's Notify, not on the channel.
    let needs_bytes = plan.mode.needs_bytes();
    let mut fetch_handles = Vec::new();
    for _ in 0..MAX_FETCH_WORKERS {
        let coord_rx = coord_rx.clone();
        let fetched_tx = fetched_tx.clone();
        let source_data = plan.source_data.clone();
        let cumulative_offsets = plan.cumulative_offsets.clone();
        let kh = plan.kh;
        let height_tiles = plan.height_tiles;
        let square_pixels = plan.square_pixels;
        let total = plan.total;
        fetch_handles.push(tokio::spawn(async move {
            while let Ok((tx, ty)) = coord_rx.recv().await {
                let tile_buf = if needs_bytes {
                    let throttle = Throttle::global();
                    let permit = throttle.acquire().await;
                    let result = fetch_tile_bytes(
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
                            log::error!("fetch_tile_bytes({tx},{ty}) failed: {e}");
                            return Err::<(), anyhow::Error>(e);
                        }
                    }
                } else {
                    None
                };
                if fetched_tx.send(FetchedTile { tx, ty, tile_buf }).await.is_err() {
                    break;
                }
            }
            Ok(())
        }));
    }
    drop(fetched_tx); // close when all fetch workers exit
    drop(coord_rx);

    // Stage 3: process workers (= num_cpus). Each pulls a FetchedTile, runs
    // the CPU-bound render+PNG encode inside `spawn_blocking`, and sends the
    // encoded result to the write channel.
    let num_proc = num_cpus_for_processing();
    let mut process_handles = Vec::new();
    for _ in 0..num_proc {
        let fetched_rx = fetched_rx.clone();
        let encoded_tx = encoded_tx.clone();
        let mode = plan.mode.clone();
        let kh = plan.kh;
        let height_tiles = plan.height_tiles;
        let square_pixels = plan.square_pixels;
        let total = plan.total;
        process_handles.push(tokio::spawn(async move {
            while let Ok(tile) = fetched_rx.recv().await {
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
                if encoded_tx.send(encoded).await.is_err() {
                    break;
                }
            }
            Ok(())
        }));
    }
    drop(encoded_tx);
    drop(fetched_rx);

    // Stage 4: writer (in this task). Drain the encoded channel.
    while let Ok(tile) = encoded_rx.recv().await {
        on_tile(tile)?;
        if let Some(ref pb) = pb {
            pb.inc(1);
        }
    }

    // Surface the first error from any stage.
    let _ = coord_task.await;
    for h in fetch_handles {
        if let Ok(Err(e)) = h.await { return Err(e); }
    }
    for h in process_handles {
        if let Ok(Err(e)) = h.await { return Err(e); }
    }

    Ok(())
}

fn render_one(
    tile: FetchedTile,
    mode: &LeafMode,
    kh: u8,
    height_tiles: u32,
    square_pixels: u64,
    total: u64,
) -> Result<EncodedTile, String> {
    let FetchedTile { tx, ty, tile_buf } = tile;
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
        LeafMode::Sort { pixel_lut, histograms } => {
            render_leaf_tile_sorted(tx, ty, kh, height_tiles, square_pixels, total, histograms, pixel_lut)?
        }
    };
    Ok(EncodedTile { tx, ty, image, png_bytes })
}

/// Run the tiled/pyramidal output pipeline to a local directory.
pub async fn run_tiles(
    sources: Vec<Source>,
    total: u64,
    tile_dir: PathBuf,
    sort: bool,
    diff_mode: bool,
    title: &str,
    inputs: &[String],
    show_xet_chunks: bool,
) -> anyhow::Result<()> {
    let plan = build_tile_plan(sources, total, sort, diff_mode, show_xet_chunks).await?;

    let max_zoom = plan.max_zoom;
    let tile_size = 256u32;
    let width_tiles = plan.width_tiles;
    let height_tiles = plan.height_tiles;
    let world_w = plan.world_w;
    let height = plan.height;
    let total_tiles = plan.total_tiles;

    let _ = plan.histograms_for_sort.clone(); // alive for lifetime of plan

    std::fs::create_dir_all(tile_dir.join(format!("tiles/{max_zoom}")))?;

    log::info!("Rendering {} leaf tiles...", total_tiles);
    let pb: Option<Arc<ProgressBar>> = if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new(total_tiles);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tiles ({eta})",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.enable_steady_tick(Duration::from_millis(100));
        Some(Arc::new(pb))
    } else {
        None
    };

    let tile_dir_for_write = tile_dir.clone();
    drive_pipeline(&plan, pb.clone(), move |t: EncodedTile| {
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

    if let Some(ref pb) = pb {
        pb.finish();
    }

    log::info!("Building tile pyramid ({} zoom levels)...", max_zoom);
    let tiles_path = tile_dir.join("tiles");
    tokio::task::spawn_blocking(move || {
        build_pyramid(&tiles_path, tile_size, max_zoom, width_tiles, height_tiles)
    })
    .await
    .map_err(|e| anyhow::anyhow!("pyramid join failure: {e}"))??;

    log::info!("Writing HTML viewer...");
    html::write_leaflet_html(&tile_dir, world_w, max_zoom, height, &plan.entities, title, inputs, &plan.chunk_segments)?;

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
    show_xet_chunks: bool,
) -> anyhow::Result<Vec<u8>> {
    crate::hf_url::require_token()?;
    let client = crate::hf_url::client()?;

    let plan = build_tile_plan(sources, total, false, diff_mode, show_xet_chunks).await?;
    let tile_size = 256u32;
    let max_zoom = plan.max_zoom;
    let world_w = plan.world_w;
    let height = plan.height;
    let total_tiles = plan.total_tiles;

    let sink = Arc::new(HfTileSink::new(client, hf_out.clone())?);
    let pyramid = Arc::new(PyramidAccumulator::new(tile_size, max_zoom, sink.clone(), Arc::new(hf_out.clone())));

    log::info!("Rendering and uploading {} leaf tiles...", total_tiles);
    let pb: Option<Arc<ProgressBar>> = if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new(total_tiles);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tiles ({eta})",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.enable_steady_tick(Duration::from_millis(100));
        Some(Arc::new(pb))
    } else {
        None
    };

    let sink_for_write = sink.clone();
    let pyramid_for_write = pyramid.clone();
    let hf_out_for_write = hf_out.clone();
    drive_pipeline(&plan, pb.clone(), move |t: EncodedTile| {
        let repo_path = hf_out_for_write.tile_repo_path(max_zoom, t.tx, t.ty);
        sink_for_write.upload_tile(repo_path, t.png_bytes)?;
        pyramid_for_write.contribute(max_zoom, t.tx, t.ty, &t.image);
        Ok(())
    })
    .await?;

    if let Some(ref pb) = pb {
        pb.finish();
    }

    log::info!("Uploading index.html and labels.json...");
    let (html_bytes, labels_bytes) = generate_leaflet_content(world_w, max_zoom, height, &plan.entities, title, inputs, &plan.chunk_segments);
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

