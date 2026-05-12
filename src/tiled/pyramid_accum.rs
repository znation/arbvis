use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use image::{ImageFormat, Rgb};

use crate::hf_url::HfOutputSpec;

/// Per-tile accumulator: running RGB sums (4 × u32 per pixel, averaged at encode time)
/// and a count of how many of the 4 child tiles have contributed.
struct TileAcc {
    sums: Vec<u32>,
    count: u8,
}

/// Streaming pyramid accumulator that builds parent/ancestor tiles in memory as
/// leaf tiles complete, uploading each parent to the provided sink immediately
/// without storing anything locally.
///
/// Thread-safe: designed for concurrent calls from rayon workers.
pub struct PyramidAccumulator<S: TileSink> {
    pending: Mutex<HashMap<(u32, u32, u32), Box<TileAcc>>>,
    tile_size: u32,
    #[allow(dead_code)]
    max_zoom: u32,
    sink: Arc<S>,
    spec: Arc<HfOutputSpec>,
}

/// Accepts encoded tile bytes and uploads them.
pub trait TileSink: Send + Sync {
    fn upload_tile(&self, repo_path: String, png_bytes: Vec<u8>) -> anyhow::Result<()>;
}

impl<S: TileSink> PyramidAccumulator<S> {
    pub fn new(tile_size: u32, max_zoom: u32, sink: Arc<S>, spec: Arc<HfOutputSpec>) -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            tile_size,
            max_zoom,
            sink,
            spec,
        }
    }

    /// Called after a tile at `(zoom, x, y)` is rendered. The tile's pixels are
    /// accumulated into the parent tile. When all 4 children of a parent arrive,
    /// the parent is encoded, uploaded, and its own contribution propagated upward.
    pub fn contribute(
        &self,
        zoom: u32,
        x: u32,
        y: u32,
        pixels: &image::ImageBuffer<Rgb<u8>, Vec<u8>>,
    ) {
        if zoom == 0 {
            return;
        }
        let parent_z = zoom - 1;
        let parent_x = x / 2;
        let parent_y = y / 2;
        let quad_x = (x % 2) as usize;
        let quad_y = (y % 2) as usize;
        let half = self.tile_size as usize / 2;
        let ts = self.tile_size as usize;

        let completed = {
            let mut pending = self.pending.lock().unwrap();
            let acc = pending
                .entry((parent_z, parent_x, parent_y))
                .or_insert_with(|| {
                    Box::new(TileAcc {
                        sums: vec![0u32; ts * ts * 3],
                        count: 0,
                    })
                });

            // Accumulate this child's 2×2-downscaled pixels into the parent quadrant.
            for py in 0..half {
                for px in 0..half {
                    let out_x = quad_x * half + px;
                    let out_y = quad_y * half + py;
                    let out_off = (out_y * ts + out_x) * 3;
                    for sy in 0..2usize {
                        for sx in 0..2usize {
                            let p = pixels.get_pixel((px * 2 + sx) as u32, (py * 2 + sy) as u32);
                            acc.sums[out_off] += p[0] as u32;
                            acc.sums[out_off + 1] += p[1] as u32;
                            acc.sums[out_off + 2] += p[2] as u32;
                        }
                    }
                }
            }
            acc.count += 1;

            if acc.count == 4 {
                pending.remove(&(parent_z, parent_x, parent_y))
            } else {
                None
            }
        };
        // Lock is released here; encode + upload without holding it.

        if let Some(acc) = completed {
            let ts32 = self.tile_size;
            let mut img = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(ts32, ts32);
            for (i, pixel) in img.pixels_mut().enumerate() {
                *pixel = Rgb([
                    (acc.sums[i * 3] / 4) as u8,
                    (acc.sums[i * 3 + 1] / 4) as u8,
                    (acc.sums[i * 3 + 2] / 4) as u8,
                ]);
            }

            let mut cursor = Cursor::new(Vec::new());
            if let Err(e) = img.write_to(&mut cursor, ImageFormat::Png) {
                eprintln!("pyramid: PNG encode error at zoom {parent_z} ({parent_x},{parent_y}): {e}");
                return;
            }
            let png_bytes = cursor.into_inner();
            let repo_path = self.spec.tile_repo_path(parent_z, parent_x, parent_y);
            if let Err(e) = self.sink.upload_tile(repo_path, png_bytes) {
                eprintln!("pyramid: upload error at zoom {parent_z} ({parent_x},{parent_y}): {e}");
                return;
            }

            // Propagate upward.
            self.contribute(parent_z, parent_x, parent_y, &img);
        }
    }
}
