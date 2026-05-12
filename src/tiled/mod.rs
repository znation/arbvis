pub mod html;
pub mod leaf;
pub mod pyramid;
pub mod pyramid_accum;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;

use image;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::color::{build_diff_signed_lut, build_pixel_lut};
use crate::data::{load_source_data, Data, Histogram, Source};
use crate::geometry::{file_rects, hilbert_to_xy_u64, name_hue, outer_segments, rects_centroid};
use crate::hf_upload::{HfXetSession, HfXetTileSink};
use crate::hf_url::HfOutputSpec;
use crate::tiled::html::{FileEntity, generate_leaflet_content};
use crate::tiled::leaf::{render_leaf_tile, render_leaf_tile_dtype, render_leaf_tile_sorted};
use crate::tiled::pyramid::build_pyramid;
use crate::tiled::pyramid_accum::PyramidAccumulator;

/// Regenerate `index.html` for an existing tiles directory without re-rendering tiles.
///
/// Infers `max_zoom`, `height`, and `world_w` from the tile directory structure,
/// then re-reads `labels.json` and calls `write_leaflet_html` with the new template.
pub fn regen_html(tile_dir: &PathBuf) -> anyhow::Result<()> {
    let tiles_dir = tile_dir.join("tiles");

    // Find max zoom level.
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

    // Parse labels.json into FileEntity objects.
    let labels_path = tile_dir.join("labels.json");
    let json_str = std::fs::read_to_string(&labels_path)
        .with_context(|| format!("cannot read {}", labels_path.display()))?;
    let values: Vec<serde_json::Value> = serde_json::from_str(&json_str)?;
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

    html::write_leaflet_html(tile_dir, world_w, max_zoom, height, &entities)?;
    log::info!(
        "Regenerated index.html in {} (zoom 0–{max_zoom}, {width_tiles}×{height_tiles} tiles, height={height})",
        tile_dir.display()
    );
    Ok(())
}

/// Run the tiled/pyramidal output pipeline.
pub fn run_tiles(
    sources: Vec<Source>,
    total: u64,
    tile_dir: PathBuf,
    sort: bool,
    diff_mode: bool,
) -> anyhow::Result<()> {
    // --sort is incompatible with safetensors dtype coloring.
    if sort && sources.iter().any(|s| s.safetensors.is_some()) {
        anyhow::bail!(
            "--sort is incompatible with safetensors files: sort reorders bytes by value, \
             which destroys positional dtype information"
        );
    }

    // Find s = ceil(log2(total)), minimum 16 so the image is at least 256×256.
    // Split into kh = floor(s/2) (height) and kw = ceil(s/2) (width).
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

    let pixel_lut = if diff_mode { build_diff_signed_lut() } else { build_pixel_lut() };


    // Build cumulative byte-start offsets.
    let mut cumulative_offsets: Vec<u64> = Vec::with_capacity(sources.len());
    {
        let mut off = 0u64;
        for s in &sources {
            cumulative_offsets.push(off);
            off += s.byte_size;
        }
    }

    // For the sort path: stream each source to build a 256-entry histogram
    // (O(1) extra memory), then render tiles directly from the histogram.
    // For the non-sort path: mmap all sources upfront (all tiles need random access).
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
        let result = sources
            .par_iter()
            .zip(cumulative_offsets.par_iter())
            .map(|(s, &off)| -> anyhow::Result<([u64; 257], u64)> {
                let hist = Histogram::build(s, sort_pb.as_deref())?;
                Ok((hist.prefix_sums(), off))
            })
            .collect::<anyhow::Result<_>>()?;
        if let Some(ref pb) = sort_pb {
            pb.finish();
        }
        result
    } else {
        vec![]
    };
    let local_agent = Arc::new(ureq::AgentBuilder::new().build());
    let source_data: Vec<Data> = if !sort {
        sources
            .par_iter()
            .map(|s| load_source_data(s, &local_agent, None))
            .collect::<anyhow::Result<_>>()?
    } else {
        vec![]
    };

    // Build combined dtype color ranges for the whole byte stream (safetensors mode).
    // Dtype coloring is only used when NOT in diff mode — in diff mode we keep the
    // black→magenta diff LUT but still generate per-tensor entity overlays.
    let dtype_mode = !diff_mode && sources.iter().any(|s| s.safetensors.is_some());
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

    // Pre-compute entity metadata.
    // For safetensors sources: one entity per tensor.
    // For regular sources: one entity per source.
    let mut entities: Vec<FileEntity> = Vec::new();
    {
        let mut cumulative: u64 = 0;
        for source in &sources {
            if let Some(st) = &source.safetensors {
                // Emit one entity per tensor (boundary + label).
                for tensor in &st.tensors {
                    let t_start = cumulative + tensor.file_start;
                    let t_end = cumulative + tensor.file_end;
                    if t_end <= t_start {
                        continue;
                    }
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
                    let bbox = if let Some(first) = rects.first() {
                        rects.iter().skip(1).fold(*first, |(x0, y0, x1, y1), &(rx0, ry0, rx1, ry1)| {
                            (x0.min(rx0), y0.min(ry0), x1.max(rx1), y1.max(ry1))
                        })
                    } else {
                        (0, 0, 0, 0)
                    };
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
                let bbox = if let Some(first) = rects.first() {
                    rects
                        .iter()
                        .skip(1)
                        .fold(*first, |(x0, y0, x1, y1), &(rx0, ry0, rx1, ry1)| {
                            (x0.min(rx0), y0.min(ry0), x1.max(rx1), y1.max(ry1))
                        })
                } else {
                    (0, 0, 0, 0)
                };
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

    // Create leaf tiles directory.
    std::fs::create_dir_all(tile_dir.join(format!("tiles/{max_zoom}")))?;

    let total_tiles = width_tiles as u64 * height_tiles as u64;
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
    let first_err = (0..total_tiles).into_par_iter().find_map_any(|i| {
        let tx = (i % width_tiles as u64) as u32;
        let ty = (i / width_tiles as u64) as u32;
        let result = if dtype_mode {
            render_leaf_tile_dtype(
                tx, ty, kh as u8, height_tiles, square_pixels, total,
                &combined_dtype_ranges,
            )
        } else if sort {
            render_leaf_tile_sorted(
                tx, ty, kh as u8, height_tiles, square_pixels, total,
                &histograms, &pixel_lut,
            )
        } else {
            render_leaf_tile(
                tx, ty, kh as u8, height_tiles, square_pixels, total,
                &source_data, &cumulative_offsets, &pixel_lut,
            )
        };
        if let Some(ref pb) = pb {
            pb.inc(1);
        }
        match result {
            Err(e) => Some(e),
            Ok((_, png_bytes)) => {
                let path = tile_dir.join(format!("tiles/{max_zoom}/{tx}/{ty}.png"));
                if let Some(parent) = path.parent() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        return Some(e.to_string());
                    }
                }
                std::fs::write(&path, &png_bytes)
                    .map_err(|e| format!("{}: {}", path.display(), e))
                    .err()
            }
        }
    });

    if let Some(ref pb) = pb {
        pb.finish();
    }
    if let Some(e) = first_err {
        return Err(anyhow::anyhow!("{e}"));
    }

    log::info!("Building tile pyramid ({} zoom levels)...", max_zoom);
    build_pyramid(
        &tile_dir.join("tiles"),
        tile_size,
        max_zoom,
        width_tiles,
        height_tiles,
    )?;

    log::info!("Writing HTML viewer...");
    html::write_leaflet_html(&tile_dir, world_w, max_zoom, height, &entities)?;

    log::info!("Tiled output written to {}", tile_dir.display());
    Ok(())
}

/// Run the tiled/pyramidal output pipeline, streaming tiles directly to HuggingFace Hub
/// via LFS pre-upload + single commit. Zero local disk required.
pub fn run_tiles_hf_streaming(
    sources: Vec<Source>,
    total: u64,
    hf_out: &HfOutputSpec,
    diff_mode: bool,
) -> anyhow::Result<()> {
    let token = crate::hf_url::get_token()
        .context("HF token required for hf:// output; set HF_TOKEN or run `huggingface-cli login`")?;

    // Geometry setup — identical to run_tiles().
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

    let pixel_lut = if diff_mode { build_diff_signed_lut() } else { build_pixel_lut() };

    let mut cumulative_offsets: Vec<u64> = Vec::with_capacity(sources.len());
    {
        let mut off = 0u64;
        for src in &sources {
            cumulative_offsets.push(off);
            off += src.byte_size;
        }
    }

    let local_agent = Arc::new(ureq::AgentBuilder::new().build());
    let arc_token = Arc::new(token.clone());
    let source_data: Vec<Data> = sources
        .par_iter()
        .map(|src| load_source_data(src, &local_agent, Some(&arc_token)))
        .collect::<anyhow::Result<_>>()?;

    let dtype_mode = !diff_mode && sources.iter().any(|s| s.safetensors.is_some());
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

    // Pre-compute entity metadata.
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

    let session = Arc::new(HfXetSession::new(hf_out, token)?);
    let sink = Arc::new(HfXetTileSink(session.clone()));
    let pyramid = Arc::new(PyramidAccumulator::new(tile_size, max_zoom, sink, Arc::new(hf_out.clone())));

    let total_tiles = width_tiles as u64 * height_tiles as u64;
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

    let first_err: Option<String> = (0..total_tiles).into_par_iter().find_map_any(|i| {
        let tx = (i % width_tiles as u64) as u32;
        let ty = (i / width_tiles as u64) as u32;
        let result = if dtype_mode {
            render_leaf_tile_dtype(tx, ty, kh as u8, height_tiles, square_pixels, total, &combined_dtype_ranges)
        } else {
            render_leaf_tile(tx, ty, kh as u8, height_tiles, square_pixels, total, &source_data, &cumulative_offsets, &pixel_lut)
        };
        if let Some(ref pb) = pb { pb.inc(1); }
        match result {
            Err(e) => Some(e),
            Ok((img, png_bytes)) => {
                let repo_path = hf_out.tile_repo_path(max_zoom, tx, ty);
                if let Err(e) = session.upload_file(repo_path, png_bytes) {
                    return Some(e.to_string());
                }
                pyramid.contribute(max_zoom, tx, ty, &img);
                None
            }
        }
    });

    if let Some(ref pb) = pb { pb.finish(); }
    if let Some(e) = first_err {
        return Err(anyhow::anyhow!("{e}"));
    }

    // Upload index.html and labels.json before committing.
    log::info!("Uploading index.html and labels.json...");
    let (html_bytes, labels_bytes) = generate_leaflet_content(world_w, max_zoom, height, &entities);
    session.upload_file(hf_out.index_html_path(), html_bytes)?;
    session.upload_file(hf_out.labels_json_path(), labels_bytes)?;

    // Drop the pyramid Arc so session has only one reference remaining.
    drop(pyramid);

    log::info!("Creating HF Hub commit...");
    Arc::try_unwrap(session)
        .map_err(|_| anyhow::anyhow!("unexpected extra Arc reference to upload session"))?
        .commit("Add arbvis visualization tiles")?;

    log::info!("Streaming output committed to hf://{}/{}", hf_out.repo_id, hf_out.path_prefix);
    Ok(())
}