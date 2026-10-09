//! Tile streaming to the Hub: [`crate::tiled::pyramid_accum::TileSink`]
//! implementation that stages tiles to a temp dir and uploads them at commit.

use std::sync::Mutex;

use anyhow::Context;
use tempfile::TempDir;

use crate::hf_cli;
use crate::hf_url::{bucket_url, HfOutputSpec, RepoKind};
use crate::throttle::with_throttle;
use crate::tiled::pyramid_accum::{write_tile_file, TileSink};

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
    staged: Mutex<usize>,
}

impl HfTileSink {
    /// Create a sink staging tiles into a fresh `arbvis-tiles-*` tempdir; the
    /// upload/commit happens later against `spec`.
    pub fn new(spec: HfOutputSpec) -> anyhow::Result<Self> {
        let tempdir = tempfile::Builder::new()
            .prefix("arbvis-tiles-")
            .tempdir()
            .context("creating tile staging tempdir")?;
        Ok(Self {
            spec,
            tempdir,
            staged: Mutex::new(0),
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
        let staged_count = self.staged.into_inner().expect("tile sink mutex poisoned");

        if staged_count == 0 {
            log::info!(
                "No tiles staged; skipping commit to hf://{}",
                self.spec.repo_id
            );
            return Ok(());
        }

        log::info!(
            "Uploading {staged_count} files to hf://{} ...",
            self.spec.repo_id,
        );

        let local_dir = self.tempdir.path().to_string_lossy().into_owned();

        match self.spec.kind {
            RepoKind::Bucket => {
                let dest = bucket_url(&self.spec.repo_id, &self.spec.path_prefix);
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
                // prefix is baked into each `tile_repo_path_in` and joined onto
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
        write_tile_file(&local_path, &png_bytes)?;
        *self.staged.lock().expect("tile sink mutex poisoned") += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_spec() -> HfOutputSpec {
        HfOutputSpec {
            repo_id: "test/repo".to_string(),
            kind: RepoKind::Model,
            revision: "main".to_string(),
            path_prefix: String::new(),
        }
    }

    #[test]
    fn upload_tile_stages_bytes_and_counts() {
        let sink = HfTileSink::new(test_spec()).expect("sink");
        sink.upload_tile("tiles/0/0_0.png".to_string(), b"png".to_vec())
            .expect("staging succeeds");
        // The staged file lands at the repo path joined onto the tempdir,
        // with its content intact and no leftover .part staging file.
        let staged_path = sink.tempdir.path().join("tiles/0/0_0.png");
        assert_eq!(
            std::fs::read(&staged_path).expect("read back staged tile"),
            b"png"
        );
        assert!(!staged_path.with_file_name("0_0.png.part").exists());
        assert_eq!(*sink.staged.lock().unwrap(), 1);

        // A second tile (and a second tile in the same directory) stages
        // independently and keeps the count in sync.
        sink.upload_tile("tiles/0/0_1.png".to_string(), b"two".to_vec())
            .expect("second staging succeeds");
        assert_eq!(
            std::fs::read(sink.tempdir.path().join("tiles/0/0_1.png")).expect("read back"),
            b"two"
        );
        assert_eq!(*sink.staged.lock().unwrap(), 2);
    }
    #[test]
    fn upload_tile_overwrites_same_repo_path() {
        let sink = HfTileSink::new(test_spec()).expect("sink");
        sink.upload_tile("tiles/t.png".to_string(), b"old".to_vec())
            .expect("first write");
        sink.upload_tile("tiles/t.png".to_string(), b"new".to_vec())
            .expect("second write");
        assert_eq!(
            std::fs::read(sink.tempdir.path().join("tiles/t.png")).expect("read back"),
            b"new"
        );
        assert_eq!(*sink.staged.lock().unwrap(), 2);
    }

    #[tokio::test]
    async fn commit_without_staged_tiles_skips_upload() {
        let sink = HfTileSink::new(test_spec()).expect("sink");
        let tempdir_path = sink.tempdir.path().to_path_buf();
        // Nothing was staged, so commit must return without invoking the
        // `hf` CLI (which would fail offline) and still clean up the
        // staging tempdir.
        sink.commit("empty run").await.expect("empty commit ok");
        assert!(!tempdir_path.exists(), "tempdir must be cleaned up");
    }

    use crate::hf_cli::tests::{FakeHfBinGuard, ENV_LOCK};

    /// Script asserting it was invoked as `hf sync <localdir> <dest> --delete`,
    /// that `<localdir>` actually holds the staged tile, and exiting 0.
    fn bucket_sync_script(expect_dest: &str) -> String {
        format!(
            "#!/bin/sh\n\
if [ \"$1\" = sync ] && [ \"$4\" = --delete ] && [ \"$3\" = {expect_dest} ] && \
[ -f \"$2/tiles/0/0_0.png\" ]; then exit 0; fi\n\
printf 'unexpected argv: %s\\n' \"$*\" >&2; exit 9\n",
            expect_dest = expect_dest
        )
    }

    #[tokio::test]
    async fn commit_bucket_uploads_with_sync_delete() {
        let _env = ENV_LOCK.lock().await;
        let _guard =
            FakeHfBinGuard::with_script(&bucket_sync_script("hf://buckets/test/repo/tiles"));
        let spec = HfOutputSpec {
            repo_id: "test/repo".to_string(),
            kind: RepoKind::Bucket,
            revision: "main".to_string(),
            path_prefix: "tiles".to_string(),
        };
        let sink = HfTileSink::new(spec).expect("sink");
        sink.upload_tile("tiles/0/0_0.png".to_string(), b"png".to_vec())
            .expect("staging succeeds");
        let tempdir_path = sink.tempdir.path().to_path_buf();
        sink.commit("summary ignored").await.expect("bucket commit");
        assert!(!tempdir_path.exists(), "tempdir must be cleaned up");
    }

    #[tokio::test]
    async fn commit_repo_uploads_large_folder() {
        let _env = ENV_LOCK.lock().await;
        // Argv must be: upload-large-folder --repo-type <type> --revision
        // <rev> <repo_id> <localdir>, with the staged tile under <localdir>.
        let _guard = FakeHfBinGuard::with_script(
            "#!/bin/sh\n\
if [ \"$1\" = upload-large-folder ] && [ \"$2\" = --repo-type ] \
&& [ \"$3\" = dataset ] && [ \"$4\" = --revision ] && [ \"$5\" = rev-7 ] \
&& [ \"$6\" = test/repo ] && [ -f \"$7/tiles/0/0_0.png\" ]; then exit 0; fi\n\
printf 'unexpected argv: %s\\n' \"$*\" >&2; exit 9\n",
        );
        let spec = HfOutputSpec {
            repo_id: "test/repo".to_string(),
            kind: RepoKind::Dataset,
            revision: "rev-7".to_string(),
            path_prefix: String::new(),
        };
        let sink = HfTileSink::new(spec).expect("sink");
        sink.upload_tile("tiles/0/0_0.png".to_string(), b"png".to_vec())
            .expect("staging succeeds");
        sink.commit("summary ignored").await.expect("repo commit");
    }

    #[tokio::test]
    async fn commit_propagates_cli_failure_with_context() {
        let _env = ENV_LOCK.lock().await;
        let _guard = FakeHfBinGuard::with_script(
            bucket_sync_script("hf://buckets/test/repo/tiles")
                .replace("exit 0; fi", "exit 7; fi")
                .as_str(),
        );
        let spec = HfOutputSpec {
            repo_id: "test/repo".to_string(),
            kind: RepoKind::Bucket,
            revision: "main".to_string(),
            path_prefix: "tiles".to_string(),
        };
        let sink = HfTileSink::new(spec).expect("sink");
        sink.upload_tile("tiles/0/0_0.png".to_string(), b"png".to_vec())
            .expect("staging succeeds");
        let err = sink
            .commit("summary")
            .await
            .expect_err("failed sync must error");
        let msg = err.to_string();
        assert!(
            msg.contains("syncing tiles to hf://buckets/test/repo/tiles"),
            "error should carry the sync context, got: {msg}"
        );
        // The CLI's own exit status lives one level down the context chain.
        let cause = err.root_cause().to_string();
        assert!(
            cause.contains("exit status: 7"),
            "error should carry the CLI exit status, got: {cause}"
        );
    }
}
