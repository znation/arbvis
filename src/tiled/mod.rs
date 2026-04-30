pub mod html;
pub mod leaf;
pub mod pyramid;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::color::build_pixel_lut;
use crate::data::Source;
use crate::geometry::{file_rects, hilbert_to_xy_u64, name_hue, outer_segments, rects_centroid};
use crate::tiled::html::FileEntity;
use crate::tiled::leaf::render_leaf_tile;
use crate::tiled::pyramid::build_pyramid;

/// Run the tiled/pyramidal output pipeline.
pub fn run_tiles(
    sources: Vec<Source>,
    total: u64,
    tile_dir: PathBuf,
) -> anyhow::Result<()> {
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

    let pixel_lut = build_pixel_lut();

    // Build cumulative byte-start offsets.
    let mut cumulative_offsets: Vec<u64> = Vec::with_capacity(sources.len());
    {
        let mut off = 0u64;
        for s in &sources {
            cumulative_offsets.push(off);
            off += s.byte_size;
        }
    }

    // Pre-compute per-file entity metadata.
    let mut entities: Vec<FileEntity> = Vec::new();
    {
        let mut cumulative: u64 = 0;
        for source in &sources {
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
            cumulative += source.byte_size;
        }
    }

    // Create leaf tiles directory.
    std::fs::create_dir_all(tile_dir.join(format!("tiles/{max_zoom}")))?;

    let total_tiles = width_tiles as u64 * height_tiles as u64;
    let pb: Option<Arc<ProgressBar>> = if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new(total_tiles);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tiles ({eta})",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        Some(Arc::new(pb))
    } else {
        None
    };

    // Render all leaf tiles in parallel.
    let first_err = (0..total_tiles).into_par_iter().find_map_any(|i| {
        let tx = (i % width_tiles as u64) as u32;
        let ty = (i / width_tiles as u64) as u32;
        let path = tile_dir.join(format!("tiles/{max_zoom}/{tx}/{ty}.png"));
        let result = render_leaf_tile(
            &path,
            tx,
            ty,
            kh as u8,
            height_tiles,
            square_pixels,
            total,
            &sources,
            &cumulative_offsets,
            &pixel_lut,
        );
        if let Some(ref pb) = pb {
            pb.inc(1);
        }
        result.err()
    });

    if let Some(ref pb) = pb {
        pb.finish_and_clear();
    }
    if let Some(e) = first_err {
        return Err(anyhow::anyhow!("{e}"));
    }

    eprintln!("Building pyramid…");
    build_pyramid(
        &tile_dir.join("tiles"),
        tile_size,
        max_zoom,
        width_tiles,
        height_tiles,
    )?;

    html::write_leaflet_html(&tile_dir, world_w, max_zoom, height, &entities)?;

    eprintln!("Tiled output written to {}", tile_dir.display());
    Ok(())
}