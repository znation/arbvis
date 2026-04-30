use std::io::IsTerminal;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use image::Rgb;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

/// Build the lower-resolution zoom pyramid from the leaf tiles.
///
/// For each zoom level z from (max_zoom-1) down to 0, averages 2×2 child tiles
/// into a single parent tile.
pub fn build_pyramid(
    tiles_dir: &Path,
    tile_size: u32,
    max_zoom: u32,
    width_tiles: u32,
    height_tiles: u32,
) -> anyhow::Result<()> {
    if max_zoom == 0 {
        return Ok(());
    }

    let total_pyramid_tiles: u64 = (0..max_zoom)
        .map(|z| {
            let levels_from_max = max_zoom - z;
            let nx = (width_tiles >> levels_from_max).max(1) as u64;
            let ny = (height_tiles >> levels_from_max).max(1) as u64;
            nx * ny
        })
        .sum();
    let pb: Option<Arc<ProgressBar>> = if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new(total_pyramid_tiles);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tiles ({eta})",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        pb.enable_steady_tick(Duration::from_millis(100));
        Some(Arc::new(pb))
    } else {
        None
    };

    let half = tile_size / 2;
    for z in (0..max_zoom).rev() {
        let child_z = z + 1;
        let levels_from_max = max_zoom - z;
        let parent_nx = (width_tiles >> levels_from_max).max(1) as usize;
        let parent_ny = (height_tiles >> levels_from_max).max(1) as usize;
        let pb = pb.clone();

        let first_err = (0..parent_ny * parent_nx)
            .into_par_iter()
            .find_map_any(|i| {
                let y = i / parent_nx;
                let x = i % parent_nx;
                let result = (|| -> anyhow::Result<()> {
                    let mut parent: image::ImageBuffer<Rgb<u8>, Vec<u8>> =
                        image::ImageBuffer::new(tile_size, tile_size);
                    let mut has_data = false;
                    for dy in 0..2usize {
                        for dx in 0..2usize {
                            let cx = 2 * x + dx;
                            let cy = 2 * y + dy;
                            let child_path =
                                tiles_dir.join(format!("{child_z}/{cx}/{cy}.png"));
                            if !child_path.exists() {
                                continue;
                            }
                            let child = image::open(&child_path)?.to_rgb8();
                            has_data = true;
                            let raw = child.as_raw();
                            let row_stride = child.width() as usize * 3;
                            for py in 0..half as usize {
                                for px in 0..half as usize {
                                    let mut r = 0u32;
                                    let mut g = 0u32;
                                    let mut b = 0u32;
                                    let mut count = 0u32;
                                    for sy in 0..2usize {
                                        for sx in 0..2usize {
                                            let sx_ = px * 2 + sx;
                                            let sy_ = py * 2 + sy;
                                            if sx_ < child.width() as usize
                                                && sy_ < child.height() as usize
                                            {
                                                let off = sy_ * row_stride + sx_ * 3;
                                                r += raw[off] as u32;
                                                g += raw[off + 1] as u32;
                                                b += raw[off + 2] as u32;
                                                count += 1;
                                            }
                                        }
                                    }
                                    if count > 0 {
                                        let out_x = dx as u32 * half + px as u32;
                                        let out_y = dy as u32 * half + py as u32;
                                        parent.put_pixel(
                                            out_x,
                                            out_y,
                                            Rgb([
                                                (r / count) as u8,
                                                (g / count) as u8,
                                                (b / count) as u8,
                                            ]),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    if has_data {
                        let parent_path = tiles_dir.join(format!("{z}/{x}/{y}.png"));
                        if let Some(p) = parent_path.parent() {
                            std::fs::create_dir_all(p)?;
                        }
                        parent.save(&parent_path)?;
                    }
                    Ok(())
                })();
                if let Some(ref pb) = pb {
                    pb.inc(1);
                }
                result.err()
            });

        if let Some(e) = first_err {
            return Err(e);
        }
    }
    if let Some(pb) = pb {
        pb.finish_and_clear();
    }
    Ok(())
}