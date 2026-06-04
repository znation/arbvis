use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Context;
use tempfile::TempDir;

use crate::hf_cli;
use crate::hf_url::{HfOutputSpec, RepoKind};
use crate::throttle::with_throttle;
use crate::tiled::pyramid_accum::TileSink;

/// Sink for streaming tile output to the Hub.
///
/// Tiles are staged to a `TempDir` as they're rendered, then handed to the
/// `hf` CLI at commit time. This bounds steady-state RAM to O(in-flight
/// tiles) regardless of pyramid size; the bytes for already-rendered tiles
/// live only on local disk. The CLI takes filesystem paths, so the tempdir
/// is the floor on local-disk usage — there's no in-memory upload seam to
/// remove it. Going below disk for pyramids larger than free disk would
/// require an upstream `hf` feature.
pub struct HfTileSink {
    spec: HfOutputSpec,
    tempdir: TempDir,
    /// Tiles staged to `tempdir`, recorded so commit can report counts
    /// (the CLI uploads the whole directory and doesn't need this list,
    /// but `is_empty()` is the cheap "did we render anything" check).
    staged: Mutex<Vec<StagedTile>>,
}

struct StagedTile {
    #[allow(dead_code)]
    repo_path: String,
    #[allow(dead_code)]
    local_path: PathBuf,
}

impl HfTileSink {
    pub fn new(spec: HfOutputSpec) -> anyhow::Result<Self> {
        let tempdir = tempfile::Builder::new()
            .prefix("arbvis-tiles-")
            .tempdir()
            .context("creating tile staging tempdir")?;
        Ok(Self {
            spec,
            tempdir,
            staged: Mutex::new(Vec::new()),
        })
    }

    /// Finalize: push everything to the Hub in one upload.
    ///
    /// Repos go through `hf upload-large-folder`, which batches commits
    /// internally (resumable on retry). Buckets go through `hf sync ...
    /// --delete`, which mirrors the prior `BucketSyncDirection::Upload` +
    /// `delete=true` semantics. Either one inherits its progress UX to
    /// the user's terminal via the helper's stderr forwarding.
    pub async fn commit(self, summary: &str) -> anyhow::Result<()> {
        let staged = self.staged.into_inner().expect("tile sink mutex poisoned");

        if staged.is_empty() {
            log::info!(
                "No tiles staged; skipping commit to hf://{}",
                self.spec.repo_id
            );
            return Ok(());
        }

        log::info!(
            "Uploading {} files to hf://{} ...",
            staged.len(),
            self.spec.repo_id,
        );

        let local_dir = self.tempdir.path().to_string_lossy().into_owned();

        match self.spec.kind {
            RepoKind::Bucket => {
                let dest = if self.spec.path_prefix.is_empty() {
                    format!("hf://buckets/{}", self.spec.repo_id)
                } else {
                    format!(
                        "hf://buckets/{}/{}",
                        self.spec.repo_id, self.spec.path_prefix
                    )
                };
                let label = format!("hf sync {local_dir} {dest}");
                with_throttle(&label, || async {
                    hf_cli::run_hf(["sync", local_dir.as_str(), dest.as_str(), "--delete"]).await
                })
                .await
                .with_context(|| format!("syncing tiles to {dest}"))?;
            }
            kind => {
                let repo_type = kind.cli_repo_type()?;
                let label = format!(
                    "hf upload-large-folder {} <- {}",
                    self.spec.repo_id, local_dir
                );
                let repo_id = self.spec.repo_id.clone();
                let revision = self.spec.revision.clone();
                // Tiles are staged at `tempdir/<prefix>/tiles/.../` (the path
                // prefix is baked into each `tile_repo_path` and joined onto
                // tempdir at staging time), so syncing the tempdir root puts
                // them at the right in-repo paths without a per-file argument.
                let _ = summary;
                with_throttle(&label, || async {
                    hf_cli::run_hf([
                        "upload-large-folder",
                        "--repo-type",
                        repo_type,
                        "--revision",
                        revision.as_str(),
                        repo_id.as_str(),
                        local_dir.as_str(),
                    ])
                    .await
                })
                .await
                .with_context(|| {
                    format!("uploading tiles to hf://{repo_id} via hf upload-large-folder")
                })?;
            }
        }

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
