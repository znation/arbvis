use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Context;
use hf_hub::buckets::BucketUpload;
use hf_hub::progress::{ProgressEvent, ProgressHandler, UploadEvent};
use hf_hub::repository::CommitOperation;
use hf_hub::HFClient;
use indicatif::ProgressBar;
use tempfile::TempDir;

use crate::hf_url::{self, HfOutputSpec, RepoKind};
use crate::progress::{counter_style, multi, status_style};
use crate::throttle::with_throttle;
use crate::tiled::pyramid_accum::TileSink;

/// Three-bar indicatif progress display fed by hf-hub upload events.
///
/// Bar layout:
/// - status line: current phase (`uploading` / `committing` / `done`) + file count
/// - logical bytes: `bytes_completed / total_bytes` from `UploadEvent::Progress`
/// - transfer bytes: post-dedup network bytes actually sent (often ≪ logical bytes)
///
/// `total_bytes` is unknown until the `Start` event arrives, so the bars are
/// constructed with placeholder lengths and set on `Start`.
struct UploadProgressBars {
    status: ProgressBar,
    logical: ProgressBar,
    transfer: ProgressBar,
}

impl UploadProgressBars {
    /// Build the bars and attach them to the global `MultiProgress`. The
    /// hidden draw target handles non-interactive runs automatically; we no
    /// longer need an Option here.
    fn new() -> Self {
        let m = multi();
        let status = m.add(ProgressBar::new(0).with_style(status_style()));
        let logical = m.add(ProgressBar::new(0).with_style(counter_style()));
        let transfer = m.add(ProgressBar::new(0).with_style(counter_style()));
        status.set_message("hf upload: waiting for Start");
        logical.set_message("logical bytes");
        transfer.set_message("transfer bytes (post-dedup)");
        for pb in [&status, &logical, &transfer] {
            pb.enable_steady_tick(Duration::from_millis(100));
        }
        Self {
            status,
            logical,
            transfer,
        }
    }

    fn finish(&self) {
        // Clear from the global multi so the upload bars don't linger after
        // the run completes.
        self.status.finish_and_clear();
        self.logical.finish_and_clear();
        self.transfer.finish_and_clear();
    }
}

/// Adapter from `hf_hub::progress::ProgressHandler` to `UploadProgressBars`.
///
/// `ProgressHandler::on_progress` requires `&self` (no `&mut`) so the bars are
/// held behind a `Mutex` even though indicatif's bars are themselves `Sync` —
/// the lock keeps the three updates consistent within one event.
struct UploadProgressLogger {
    bars: Mutex<UploadProgressBars>,
}

impl UploadProgressLogger {
    fn new(bars: UploadProgressBars) -> Self {
        Self {
            bars: Mutex::new(bars),
        }
    }

    /// Stop the spinners and release the terminal lines. Safe to call when no
    /// bars were created (non-TTY runs).
    fn finish_bars(&self) {
        self.bars
            .lock()
            .expect("upload progress mutex poisoned")
            .finish();
    }
}

impl ProgressHandler for UploadProgressLogger {
    fn on_progress(&self, event: &ProgressEvent) {
        let guard = self.bars.lock().expect("upload progress mutex poisoned");
        let bars = &*guard;
        match event {
            ProgressEvent::Upload(UploadEvent::Start {
                total_files,
                total_bytes,
            }) => {
                bars.logical.set_length(*total_bytes);
                bars.transfer.set_length(*total_bytes);
                bars.status.set_message(format!(
                    "hf upload: {total_files} files, {total_bytes} bytes (uploading)",
                ));
            }
            ProgressEvent::Upload(UploadEvent::Progress {
                bytes_completed,
                total_bytes,
                transfer_bytes_completed,
                transfer_bytes,
                ..
            }) => {
                bars.logical.set_length(*total_bytes);
                bars.logical.set_position(*bytes_completed);
                bars.transfer.set_length(*transfer_bytes);
                bars.transfer.set_position(*transfer_bytes_completed);
            }
            ProgressEvent::Upload(UploadEvent::Committing) => {
                bars.status
                    .set_message("hf upload: committing (server-side)");
            }
            ProgressEvent::Upload(UploadEvent::Complete) => {
                bars.status.set_message("hf upload: complete");
            }
            ProgressEvent::Download(_) => {}
        }
    }
}

/// Sink for streaming tile output to the Hub.
///
/// Both code paths stage tiles to a `TempDir` as they are rendered, then hand
/// the disk-backed paths to hf-hub at commit time. This bounds steady-state RAM
/// to O(in-flight tiles) regardless of pyramid size; the bytes for already-
/// rendered tiles live only on local disk.
///
/// Why disk staging rather than upload-as-you-go: hf-hub 1.0.0-rc.1 exposes
/// neither a public "upload bytes now, reference by hash at commit time" seam
/// for git-backed repos (its `xet_upload` is `pub(crate)`) nor a per-file
/// streaming entry point for buckets (`upload_files` takes a `Vec<BucketUpload>`
/// of on-disk paths). To go below disk too — pyramids larger than local free
/// disk — needs an upstream hf-hub feature; until then the tempdir is the floor.
pub struct HfTileSink {
    client: HFClient,
    spec: HfOutputSpec,
    tempdir: TempDir,
    /// Tiles staged to `tempdir`, indexed by repo path. Recorded in a single
    /// mutex so the commit/upload step can iterate without re-walking the
    /// directory (and so an empty render is observable as `staged.is_empty()`).
    staged: Mutex<Vec<StagedTile>>,
}

struct StagedTile {
    repo_path: String,
    local_path: PathBuf,
}

impl HfTileSink {
    pub fn new(client: HFClient, spec: HfOutputSpec) -> anyhow::Result<Self> {
        let tempdir = tempfile::Builder::new()
            .prefix("arbvis-tiles-")
            .tempdir()
            .context("creating tile staging tempdir")?;
        Ok(Self {
            client,
            spec,
            tempdir,
            staged: Mutex::new(Vec::new()),
        })
    }

    /// Finalize: push everything to the Hub in one commit (or bucket upload).
    pub async fn commit(self, summary: &str) -> anyhow::Result<()> {
        let staged = self.staged.into_inner().expect("tile sink mutex poisoned");

        if staged.is_empty() {
            log::info!(
                "No tiles staged; skipping commit to hf://{}",
                self.spec.repo_id
            );
            return Ok(());
        }

        let (owner, name) = hf_url::split_owner_name(&self.spec.repo_id)?;
        log::info!(
            "Committing {} files to hf://{}",
            staged.len(),
            self.spec.repo_id
        );

        let logger = std::sync::Arc::new(UploadProgressLogger::new(UploadProgressBars::new()));

        match self.spec.kind {
            RepoKind::Bucket => {
                let uploads: Vec<BucketUpload> = staged
                    .into_iter()
                    .map(|t| BucketUpload::new(t.local_path, t.repo_path))
                    .collect();
                with_throttle(&format!("bucket upload {}", self.spec.repo_id), || async {
                    self.client
                        .bucket(owner, name)
                        .upload_files()
                        .files(uploads.clone())
                        .progress(logger.clone())
                        .send()
                        .await
                })
                .await?;
            }
            kind => {
                let ops: Vec<CommitOperation> = staged
                    .into_iter()
                    .map(|t| CommitOperation::add_file(t.repo_path, t.local_path))
                    .collect();
                let revision = self.spec.revision.clone();
                let message = summary.to_string();
                let label = format!("create_commit {}", self.spec.repo_id);
                with_throttle(&label, || async {
                    match kind {
                        RepoKind::Model => self
                            .client
                            .model(owner, name)
                            .create_commit()
                            .operations(ops.clone())
                            .commit_message(message.clone())
                            .revision(revision.clone())
                            .progress(logger.clone())
                            .send()
                            .await
                            .map(|_| ()),
                        RepoKind::Dataset => self
                            .client
                            .dataset(owner, name)
                            .create_commit()
                            .operations(ops.clone())
                            .commit_message(message.clone())
                            .revision(revision.clone())
                            .progress(logger.clone())
                            .send()
                            .await
                            .map(|_| ()),
                        RepoKind::Space => self
                            .client
                            .space(owner, name)
                            .create_commit()
                            .operations(ops.clone())
                            .commit_message(message.clone())
                            .revision(revision.clone())
                            .progress(logger.clone())
                            .send()
                            .await
                            .map(|_| ()),
                        RepoKind::Bucket => unreachable!(),
                    }
                })
                .await?;
            }
        }

        logger.finish_bars();
        // tempdir drops here — staged tile files are removed from disk.
        drop(self.tempdir);
        Ok(())
    }
}

impl TileSink for HfTileSink {
    fn upload_tile(&self, repo_path: String, png_bytes: Vec<u8>) -> anyhow::Result<()> {
        let local_path = self.tempdir.path().join(&repo_path);
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating tile dir {}", parent.display()))?;
        }
        std::fs::write(&local_path, &png_bytes)
            .with_context(|| format!("writing tile {}", local_path.display()))?;
        self.staged
            .lock()
            .expect("tile sink mutex poisoned")
            .push(StagedTile {
                repo_path,
                local_path,
            });
        Ok(())
    }
}
