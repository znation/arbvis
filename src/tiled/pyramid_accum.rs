use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use image::Rgb;
use tokio::task::JoinHandle;

use crate::tiled::leaf::{encode_tile, TileFormat};

/// Per-tile accumulator: running RGB sums (4 × u32 per pixel, averaged at encode time)
/// and a count of how many of the 4 child tiles have contributed.
struct TileAcc {
    sums: Vec<u32>,
    count: u8,
}

/// Streaming pyramid accumulator that builds parent/ancestor tiles in memory as
/// leaf tiles complete, encoding + dispatching each parent to the provided
/// sink immediately. Used by both the local-disk and HF-streaming output
/// paths; the latter avoids decoding tiles back from disk during pyramid
/// build, which matters for AVIF (no pure-Rust AVIF decoder in our dep set).
///
/// Thread-safe and async-aware: when a parent's 4 children have all
/// contributed, the encode+upload work is offloaded to
/// [`tokio::task::spawn_blocking`] so the calling thread can return immediately.
/// Outstanding tasks are tracked in `outstanding` and drained by
/// [`Self::drain`] before commit.
pub struct PyramidAccumulator<S: TileSink> {
    pending: Mutex<HashMap<(u32, u32, u32), Box<TileAcc>>>,
    outstanding: Mutex<Vec<JoinHandle<()>>>,
    tile_size: u32,
    #[allow(dead_code)]
    max_zoom: u32,
    sink: Arc<S>,
    /// Maps `(zoom, x, y)` → destination path string. The sink interprets it
    /// (HF repo path, local filesystem path, …).
    path_fn: Arc<dyn Fn(u32, u32, u32) -> String + Send + Sync>,
    /// Format used for pyramid (non-leaf) tiles. Leaf tiles are encoded in
    /// the leaf render stage and don't pass through this encoder.
    pyramid_format: TileFormat,
}

/// Accepts encoded tile bytes and persists them.
pub trait TileSink: Send + Sync + 'static {
    fn upload_tile(&self, path: String, bytes: Vec<u8>) -> anyhow::Result<()>;
}

/// Writes encoded tile bytes to a local filesystem path, creating parent
/// directories as needed. Used by the local `run_tiles` output path.
pub struct LocalFileSink {
    pub root: PathBuf,
}

impl TileSink for LocalFileSink {
    fn upload_tile(&self, path: String, bytes: Vec<u8>) -> anyhow::Result<()> {
        let full = self.root.join(path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&full, &bytes)?;
        Ok(())
    }
}

impl<S: TileSink> PyramidAccumulator<S> {
    pub fn new(
        tile_size: u32,
        max_zoom: u32,
        sink: Arc<S>,
        path_fn: Arc<dyn Fn(u32, u32, u32) -> String + Send + Sync>,
        pyramid_format: TileFormat,
    ) -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            outstanding: Mutex::new(Vec::new()),
            tile_size,
            max_zoom,
            sink,
            path_fn,
            pyramid_format,
        }
    }

    /// Called after a tile at `(zoom, x, y)` is rendered. The tile's pixels are
    /// accumulated into the parent tile. When all 4 children of a parent
    /// arrive, the encode + upload (and the recursive contribute upward) is
    /// dispatched to `spawn_blocking` so the writer task is never blocked on
    /// encoding or disk I/O.
    pub fn contribute(
        self: &Arc<Self>,
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

        // Downscale this child's pixels into a local quadrant buffer *outside*
        // the lock. This 2×2 box-filter (half·half·4 reads) is the expensive
        // part, and each child writes a disjoint quadrant of the parent, so it
        // needs no synchronisation. Only the cheap write-back + count bump below
        // run under the global `pending` lock, which 10 render workers contend.
        let mut quad = vec![0u32; half * half * 3];
        for py in 0..half {
            for px in 0..half {
                let q_off = (py * half + px) * 3;
                for sy in 0..2usize {
                    for sx in 0..2usize {
                        let p = pixels.get_pixel((px * 2 + sx) as u32, (py * 2 + sy) as u32);
                        quad[q_off] += p[0] as u32;
                        quad[q_off + 1] += p[1] as u32;
                        quad[q_off + 2] += p[2] as u32;
                    }
                }
            }
        }

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

            // Copy the precomputed quadrant into the parent's disjoint region,
            // one contiguous row at a time. Quadrants never overlap, so a plain
            // copy is equivalent to the previous `+=` into a zeroed buffer.
            for py in 0..half {
                let out_y = quad_y * half + py;
                let dst = (out_y * ts + quad_x * half) * 3;
                let src = py * half * 3;
                acc.sums[dst..dst + half * 3].copy_from_slice(&quad[src..src + half * 3]);
            }
            acc.count += 1;

            if acc.count == 4 {
                pending.remove(&(parent_z, parent_x, parent_y))
            } else {
                None
            }
        };
        // Lock released here; encode + upload happens off-thread.

        let Some(acc) = completed else { return };

        let me = Arc::clone(self);
        let ts32 = self.tile_size;
        let fmt = self.pyramid_format;
        let handle = tokio::task::spawn_blocking(move || {
            let mut img = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(ts32, ts32);
            for (i, pixel) in img.pixels_mut().enumerate() {
                *pixel = Rgb([
                    (acc.sums[i * 3] / 4) as u8,
                    (acc.sums[i * 3 + 1] / 4) as u8,
                    (acc.sums[i * 3 + 2] / 4) as u8,
                ]);
            }

            let (img, bytes) = match encode_tile(img, fmt) {
                Ok(v) => v,
                Err(e) => {
                    log::error!(
                        "pyramid: encode error at zoom {parent_z} ({parent_x},{parent_y}): {e}"
                    );
                    return;
                }
            };
            let path = (me.path_fn)(parent_z, parent_x, parent_y);
            if let Err(e) = me.sink.upload_tile(path, bytes) {
                log::error!("pyramid: write error at zoom {parent_z} ({parent_x},{parent_y}): {e}");
                return;
            }

            // Propagate upward — may recursively trigger another spawn_blocking
            // when this parent's parent reaches count=4.
            me.contribute(parent_z, parent_x, parent_y, &img);
        });
        self.outstanding.lock().unwrap().push(handle);
    }

    /// Await all outstanding encode+upload tasks. Call this before
    /// `HfTileSink::commit` so every staged file is on disk by commit time.
    ///
    /// The set may grow while we drain (a task may spawn another for its own
    /// parent), so we loop until the outstanding list is empty.
    pub async fn drain(&self) {
        loop {
            let handles = {
                let mut g = self.outstanding.lock().unwrap();
                std::mem::take(&mut *g)
            };
            if handles.is_empty() {
                return;
            }
            for h in handles {
                let _ = h.await;
            }
        }
    }
}
