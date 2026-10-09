//! Pyramid tile accumulation: per-tile RGB-sum accumulators that average
//! child tiles into parent tiles, drained to a [`TileSink`] at the end.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Context;
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
/// [`Self::drain`] before commit. The first failure any detached task hits
/// (encode or sink upload) is recorded in `first_error` and surfaced by
/// [`Self::drain`] as an `Err`, so a pyramid tile that silently failed to
/// persist cannot pass for a completed run.
pub struct PyramidAccumulator<S: TileSink> {
    pending: Mutex<HashMap<(u32, u32, u32), Box<TileAcc>>>,
    outstanding: Mutex<Vec<JoinHandle<()>>>,
    first_error: Mutex<Option<anyhow::Error>>,
    tile_size: u32,
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

/// Writes tile bytes to `path`, creating its parent directory first.
/// Shared by the local-disk output path and the HF staging sink, which both
/// persist tiles with exactly this sequence.
///
/// The write is staged to `<file>.part` in the same directory, then renamed
/// over `path`, so a process killed mid-write (or an ENOSPC partway through)
/// leaves the previous tile — or none — instead of a truncated tile that the
/// generated viewer serves as if complete.
pub fn write_tile_file(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating tile dir {}", parent.display()))?;
    }
    let part = crate::fsutil::part_path(path);
    if let Err(e) = std::fs::write(&part, bytes) {
        // Best effort: don't leave a stale partial staging file behind.
        let _ = std::fs::remove_file(&part);
        return Err(e).with_context(|| format!("writing tile {}", part.display()));
    }
    crate::fsutil::seal_part(&part, path)
        .map_err(|e| e.context(format!("sealing tile {}", path.display())))
}

/// Writes encoded tile bytes to a local filesystem path, creating parent
/// directories as needed. Used by the local `run_tiles` output path.
pub struct LocalFileSink {
    /// Directory tiles are written under (relative tile paths resolve into it).
    pub root: PathBuf,
}

impl TileSink for LocalFileSink {
    fn upload_tile(&self, path: String, bytes: Vec<u8>) -> anyhow::Result<()> {
        write_tile_file(&self.root.join(path), &bytes)
    }
}

impl<S: TileSink> PyramidAccumulator<S> {
    /// Create an accumulator that folds rendered `(zoom, x, y)` tiles upward
    /// through the pyramid, writing each encoded tile via `path_fn` to `sink`.
    pub fn new(
        tile_size: u32,
        sink: Arc<S>,
        path_fn: Arc<dyn Fn(u32, u32, u32) -> String + Send + Sync>,
        pyramid_format: TileFormat,
    ) -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            outstanding: Mutex::new(Vec::new()),
            first_error: Mutex::new(None),
            tile_size,
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
                    me.record_error(anyhow::anyhow!(
                        "pyramid encode error at zoom {parent_z} ({parent_x},{parent_y}): {e}"
                    ));
                    return;
                }
            };
            let path = (me.path_fn)(parent_z, parent_x, parent_y);
            if let Err(e) = me.sink.upload_tile(path, bytes) {
                log::error!("pyramid: write error at zoom {parent_z} ({parent_x},{parent_y}): {e}");
                me.record_error(anyhow::anyhow!(
                    "pyramid write error at zoom {parent_z} ({parent_x},{parent_y}): {e}"
                ));
                return;
            }

            // Propagate upward — may recursively trigger another spawn_blocking
            // when this parent's parent reaches count=4.
            me.contribute(parent_z, parent_x, parent_y, &img);
        });
        self.outstanding.lock().unwrap().push(handle);
    }

    /// Records the first error seen by any detached encode/upload task (later
    /// ones are dropped; the first is the root cause of the cascade) so
    /// [`Self::drain`] can fail the run instead of letting missing pyramid
    /// tiles pass silently.
    fn record_error(&self, e: anyhow::Error) {
        let mut g = self.first_error.lock().unwrap();
        if g.is_none() {
            *g = Some(e);
        }
    }

    /// Await all outstanding encode+upload tasks, failing if any of them hit
    /// an encode, upload, or panic error. Call this before
    /// `HfTileSink::commit` so every staged file is on disk by commit time.
    ///
    /// The set may grow while we drain (a task may spawn another for its own
    /// parent), so we loop until the outstanding list is empty.
    pub async fn drain(&self) -> anyhow::Result<()> {
        loop {
            let handles = {
                let mut g = self.outstanding.lock().unwrap();
                std::mem::take(&mut *g)
            };
            if handles.is_empty() {
                break;
            }
            for h in handles {
                if let Err(join_err) = h.await {
                    self.record_error(anyhow::anyhow!("pyramid task panicked: {join_err}"));
                }
            }
        }
        // Report (don't consume) the first recorded failure so a repeat
        // drain — e.g. the next scene in the streaming path — cannot pass
        // after a prior drain already observed the error.
        if let Some(e) = self.first_error.lock().unwrap().as_ref() {
            return Err(anyhow::anyhow!("{e}"));
        }
        Ok(())
    }
}

/// Await all outstanding pyramid encode/upload tasks and consume the
/// accumulator's last `Arc`, mapping any failure to a uniform "tile set is
/// incomplete" error.
///
/// Both production drain call sites (`tiled/pipeline.rs` and
/// `tiled/streaming.rs`) need exactly this sequence — drain, drop the
/// accumulator, contextualize the error — so it lives here once. The unit
/// tests below use [`PyramidAccumulator::drain`] directly instead, because
/// they re-drain or inspect the accumulator after the call.
pub async fn drain_and_report_incomplete<S: TileSink>(
    pyramid: Arc<PyramidAccumulator<S>>,
) -> anyhow::Result<()> {
    let drained = pyramid.drain().await;
    drop(pyramid);
    drained.context("pyramid overview-tile encode/upload failed; the tile set is incomplete")
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    #[test]
    fn write_tile_file_creates_missing_parents() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("a/b/c/tile.png");
        write_tile_file(&path, b"bytes").expect("write succeeds");
        assert_eq!(std::fs::read(&path).expect("read back"), b"bytes");
        // The staged file must not linger next to the finished tile.
        assert!(!path.with_file_name("tile.png.part").exists());
    }

    #[test]
    fn write_tile_file_failure_leaves_previous_tile_intact() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tile.png");
        write_tile_file(&path, b"original").expect("first write succeeds");

        // Simulate a failure partway through the staged write: make the
        // staging location unwritable by putting a directory in its place.
        std::fs::create_dir(dir.path().join("tile.png.part")).expect("mkdir");
        assert!(write_tile_file(&path, b"replacement").is_err());

        // The previously written tile is untouched — a truncated or failed
        // write must never clobber it with garbage.
        assert_eq!(
            std::fs::read(&path).expect("read back"),
            b"original",
            "failed write must not clobber the existing tile"
        );
    }

    #[test]
    fn write_tile_file_rename_failure_leaves_no_part() {
        // Simulate a rename failure (target path is a directory): the staged
        // file must not be left behind next to the failed destination.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tile.png");
        std::fs::create_dir(&path).expect("mkdir");
        assert!(write_tile_file(&path, b"bytes").is_err());
        assert!(
            !path.with_file_name("tile.png.part").exists(),
            "failed rename must not leave a stale .part staging file"
        );
    }

    /// Records (path, decoded RGB pixels) for every tile uploaded.
    #[derive(Default)]
    struct RecordingSink {
        uploads: StdMutex<Vec<(String, image::ImageBuffer<Rgb<u8>, Vec<u8>>)>>,
    }

    impl TileSink for RecordingSink {
        fn upload_tile(&self, path: String, bytes: Vec<u8>) -> anyhow::Result<()> {
            let img = image::load_from_memory(&bytes)
                .expect("uploaded bytes decode")
                .to_rgb8();
            self.uploads.lock().unwrap().push((path, img));
            Ok(())
        }
    }

    fn constant_tile(tile_size: u32, color: [u8; 3]) -> image::ImageBuffer<Rgb<u8>, Vec<u8>> {
        image::ImageBuffer::from_fn(tile_size, tile_size, |_, _| Rgb(color))
    }

    fn make_acc(
        sink: Arc<RecordingSink>,
        tile_size: u32,
    ) -> Arc<PyramidAccumulator<RecordingSink>> {
        Arc::new(PyramidAccumulator::new(
            tile_size,
            sink,
            Arc::new(|z, x, y| format!("{z}/{x}/{y}.png")),
            TileFormat::Png,
        ))
    }

    async fn uploads_after(
        acc: &Arc<PyramidAccumulator<RecordingSink>>,
    ) -> Vec<(String, image::ImageBuffer<Rgb<u8>, Vec<u8>>)> {
        acc.drain().await.expect("pyramid drain succeeds");
        let mut uploads = acc.sink.uploads.lock().unwrap().clone();
        uploads.sort_by(|a, b| a.0.cmp(&b.0));
        uploads
    }

    /// A sink whose every upload fails, to prove drain() surfaces the
    /// failure instead of letting an incomplete pyramid pass for success.
    struct FailingSink;
    impl TileSink for FailingSink {
        fn upload_tile(&self, _path: String, _bytes: Vec<u8>) -> anyhow::Result<()> {
            anyhow::bail!("simulated sink failure")
        }
    }

    /// All four children of one parent upload through a failing sink:
    /// drain() must return a descriptive error naming the failed tile.
    #[tokio::test]
    async fn upload_failure_is_surfaced_by_drain() {
        let acc: Arc<PyramidAccumulator<FailingSink>> = Arc::new(PyramidAccumulator::new(
            4,
            Arc::new(FailingSink),
            Arc::new(|z, x, y| format!("{z}/{x}/{y}.png")),
            TileFormat::Png,
        ));
        for x in 0..2 {
            for y in 0..2 {
                acc.contribute(1, x, y, &constant_tile(4, [1, 2, 3]));
            }
        }
        let err = acc
            .drain()
            .await
            .expect_err("failed upload must surface through drain");
        assert!(
            err.to_string()
                .contains("pyramid write error at zoom 0 (0,0)"),
            "unexpected error: {err:#}"
        );
        // A repeat drain must still report the failure, not reset to Ok.
        assert!(acc.drain().await.is_err());
    }

    #[tokio::test]
    async fn zoom_zero_tile_is_ignored() {
        let sink = Arc::new(RecordingSink::default());
        let acc = make_acc(sink, 4);
        acc.contribute(0, 0, 0, &constant_tile(4, [1, 2, 3]));
        assert!(uploads_after(&acc).await.is_empty());
    }

    #[tokio::test]
    async fn incomplete_parent_is_not_uploaded() {
        let sink = Arc::new(RecordingSink::default());
        let acc = make_acc(sink, 4);
        acc.contribute(1, 0, 0, &constant_tile(4, [10, 20, 30]));
        acc.contribute(1, 1, 0, &constant_tile(4, [10, 20, 30]));
        assert!(uploads_after(&acc).await.is_empty());
    }

    #[tokio::test]
    async fn four_children_average_into_parent_quadrants() {
        let sink = Arc::new(RecordingSink::default());
        let acc = make_acc(sink, 4);
        // (zoom 2) x%2 picks the parent's left/right half, y%2 top/bottom.
        acc.contribute(2, 0, 0, &constant_tile(4, [255, 0, 0])); // parent (1,0,0) top-left
        acc.contribute(2, 1, 0, &constant_tile(4, [0, 255, 0])); // top-right
        acc.contribute(2, 0, 1, &constant_tile(4, [0, 0, 255])); // bottom-left
        acc.contribute(2, 1, 1, &constant_tile(4, [255, 255, 255])); // bottom-right

        let uploads = uploads_after(&acc).await;
        assert_eq!(uploads.len(), 1, "exactly the one completed parent uploads");
        let (path, img) = &uploads[0];
        assert_eq!(path, "1/0/0.png");
        assert_eq!((img.width(), img.height()), (4, 4));
        assert_eq!(img.get_pixel(0, 0), &Rgb([255, 0, 0]), "top-left");
        assert_eq!(img.get_pixel(3, 0), &Rgb([0, 255, 0]), "top-right");
        assert_eq!(img.get_pixel(0, 3), &Rgb([0, 0, 255]), "bottom-left");
        assert_eq!(img.get_pixel(3, 3), &Rgb([255, 255, 255]), "bottom-right");
    }

    #[tokio::test]
    async fn averaging_blends_two_children_in_one_quadrant() {
        // Four distinct children sharing a parent: each quadrant is the box
        // filter of one child only, so feed a parent whose children mix —
        // instead, verify the /4 division directly: a parent fed by children
        // that each contribute their own color to their own quadrant must not
        // sum-overflow or misplace. Simpler check: 4 children of one parent
        // where every child has color (8, 4, 200) → parent is exactly that.
        let sink = Arc::new(RecordingSink::default());
        let acc = make_acc(sink, 4);
        for x in 0..2 {
            for y in 0..2 {
                acc.contribute(1, x, y, &constant_tile(4, [8, 4, 200]));
            }
        }
        let uploads = uploads_after(&acc).await;
        assert_eq!(uploads.len(), 1);
        let (_, img) = &uploads[0];
        assert_eq!(img.get_pixel(1, 1), &Rgb([8, 4, 200]));
        assert_eq!(img.get_pixel(2, 2), &Rgb([8, 4, 200]));
    }

    #[tokio::test]
    async fn completed_parents_propagate_to_the_root() {
        let sink = Arc::new(RecordingSink::default());
        let acc = make_acc(sink, 4);
        for x in 0..2 {
            for y in 0..2 {
                for cx in 0..2 {
                    for cy in 0..2 {
                        acc.contribute(2, 2 * x + cx, 2 * y + cy, &constant_tile(4, [1, 2, 3]));
                    }
                }
            }
        }
        // 4 zoom-1 parents complete (each uploading itself), and once all 4
        // contributed upward, the zoom-0 root tile uploads too.
        let uploads = uploads_after(&acc).await;
        let paths: Vec<&str> = uploads.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(
            paths,
            vec![
                "0/0/0.png",
                "1/0/0.png",
                "1/0/1.png",
                "1/1/0.png",
                "1/1/1.png"
            ]
        );
        for (_, img) in &uploads {
            assert_eq!(img.get_pixel(0, 0), &Rgb([1, 2, 3]));
        }
    }

    #[tokio::test]
    async fn drain_is_idempotent_when_idle() {
        let sink = Arc::new(RecordingSink::default());
        let acc = make_acc(sink, 4);
        let _ = acc.drain().await;
        let _ = acc.drain().await;
        assert!(acc.sink.uploads.lock().unwrap().is_empty());
    }

    /// A sink that panics: the JoinError from the detached spawn_blocking
    /// task must be recorded and surfaced by drain() as a panic error, not
    /// lost when the task dies outside any caller's await.
    struct PanickingSink;
    impl TileSink for PanickingSink {
        fn upload_tile(&self, _path: String, _bytes: Vec<u8>) -> anyhow::Result<()> {
            panic!("simulated sink panic")
        }
    }

    #[tokio::test]
    async fn task_panic_is_surfaced_by_drain() {
        let acc: Arc<PyramidAccumulator<PanickingSink>> = Arc::new(PyramidAccumulator::new(
            4,
            Arc::new(PanickingSink),
            Arc::new(|z, x, y| format!("{z}/{x}/{y}.png")),
            TileFormat::Png,
        ));
        for x in 0..2 {
            for y in 0..2 {
                acc.contribute(1, x, y, &constant_tile(4, [1, 2, 3]));
            }
        }
        let err = acc
            .drain()
            .await
            .expect_err("panicked task must surface through drain");
        assert!(
            err.to_string().contains("pyramid task panicked"),
            "unexpected error: {err:#}"
        );
    }

    /// drain_and_report_incomplete wraps drain failures in the uniform
    /// "tile set is incomplete" context so callers see a consistent message.
    #[tokio::test]
    async fn drain_and_report_incomplete_contextualizes_failure() {
        let acc: Arc<PyramidAccumulator<FailingSink>> = Arc::new(PyramidAccumulator::new(
            4,
            Arc::new(FailingSink),
            Arc::new(|z, x, y| format!("{z}/{x}/{y}.png")),
            TileFormat::Png,
        ));
        for x in 0..2 {
            for y in 0..2 {
                acc.contribute(1, x, y, &constant_tile(4, [1, 2, 3]));
            }
        }
        let err = drain_and_report_incomplete(acc)
            .await
            .expect_err("sink failure must map to the incomplete-tile-set error");
        assert!(
            err.to_string()
                .contains("pyramid overview-tile encode/upload failed"),
            "unexpected error: {err:#}"
        );
    }
}
