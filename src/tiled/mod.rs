pub mod html;
pub mod leaf;
pub mod pyramid;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use image::{self, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::color::{build_diff_pixel_lut, build_diff_position_lut, build_pixel_lut};
use crate::data::{load_source_data, Data, Histogram, Source};
use crate::geometry::{file_rects, hilbert_to_xy_u64, name_hue, outer_segments, rects_centroid};
use crate::tiled::html::FileEntity;
use crate::tiled::leaf::{
    render_leaf_tile, render_leaf_tile_diff_positional, render_leaf_tile_dtype,
    render_leaf_tile_sorted,
};
use crate::tiled::pyramid::build_pyramid;

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

    let pixel_lut = if diff_mode { build_diff_pixel_lut() } else { build_pixel_lut() };

    // Positional diff: every source has a position_hint (only true for safetensors diffs).
    // Falls back to the single-LUT path for plain binary diffs (position_hint is None).
    let positional_diff_mode = diff_mode
        && !sort
        && !sources.is_empty()
        && sources.iter().all(|s| s.position_hint.is_some());

    let per_source_luts: Vec<[Rgb<u8>; 256]> = if positional_diff_mode {
        sources
            .iter()
            .map(|s| build_diff_position_lut(s.position_hint.unwrap_or(0.0)))
            .collect()
    } else {
        vec![]
    };

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
    let source_data: Vec<Data> = if !sort {
        sources
            .par_iter()
            .map(|s| load_source_data(s))
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
                let rects = file_rects(
                    cumulative,
                    cumulative + source.byte_size,
                    total_pixels,
                    square_pixels,
                    num_squares,
                    height,
                    kh as u8,
                );
                let (pixel_x, pixel_y) = rects_centroid(&rects).unwrap_or_else(|| {
                    let mid = cumulative + source.byte_size / 2;
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
                    byte_size: source.byte_size,
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
        let path = tile_dir.join(format!("tiles/{max_zoom}/{tx}/{ty}.png"));
        let result = if dtype_mode {
            render_leaf_tile_dtype(
                &path,
                tx,
                ty,
                kh as u8,
                height_tiles,
                square_pixels,
                total,
                &combined_dtype_ranges,
            )
        } else if sort {
            render_leaf_tile_sorted(
                &path,
                tx,
                ty,
                kh as u8,
                height_tiles,
                square_pixels,
                total,
                &histograms,
                &pixel_lut,
            )
        } else if positional_diff_mode {
            render_leaf_tile_diff_positional(
                &path,
                tx,
                ty,
                kh as u8,
                height_tiles,
                square_pixels,
                total,
                &source_data,
                &cumulative_offsets,
                &per_source_luts,
            )
        } else {
            render_leaf_tile(
                &path,
                tx,
                ty,
                kh as u8,
                height_tiles,
                square_pixels,
                total,
                &source_data,
                &cumulative_offsets,
                &pixel_lut,
            )
        };
        if let Some(ref pb) = pb {
            pb.inc(1);
        }
        result.err()
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