use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Context;
use hf_hub::buckets::BucketUpload;
use hf_hub::repository::CommitOperation;
use hf_hub::HFClientSync;
use tempfile::TempDir;

use crate::hf_url::HfOutputSpec;
use crate::tiled::pyramid_accum::TileSink;

/// Sink for streaming tile output to the Hub.
///
/// - For git-backed repos (model / dataset / space): each tile becomes an in-memory
///   `CommitOperation::Add` accumulated under a mutex; finalised in a single
///   `create_commit()` call. No tile bytes touch local disk.
/// - For buckets: each tile is written into a `TempDir` because hf-hub's bucket
///   upload API requires on-disk source paths to drive its xet upload + batch
///   register flow. The tempdir is deleted on `commit()` return.
pub struct HfTileSink {
    client: HFClientSync,
    spec: HfOutputSpec,
    state: SinkState,
}

enum SinkState {
    Repo { ops: Mutex<Vec<CommitOperation>> },
    Bucket { tempdir: TempDir },
}

impl HfTileSink {
    pub fn new(client: HFClientSync, spec: HfOutputSpec) -> anyhow::Result<Self> {
        let state = if spec.repo_type_str == "bucket" {
            SinkState::Bucket { tempdir: tempfile::tempdir().context("creating tile tempdir")? }
        } else {
            SinkState::Repo { ops: Mutex::new(Vec::new()) }
        };
        Ok(Self { client, spec, state })
    }

    /// Finalize: push everything to the Hub in one commit (or bucket sync).
    pub fn commit(self, summary: &str) -> anyhow::Result<()> {
        let (owner, name) = split_owner_name(&self.spec.repo_id)?;
        match self.state {
            SinkState::Repo { ops } => {
                let ops = ops.into_inner().unwrap();
                log::info!("Committing {} files to hf://{}", ops.len(), self.spec.repo_id);
                match self.spec.repo_type_str {
                    "model" => {
                        self.client.model(owner, name)
                            .create_commit()
                            .operations(ops)
                            .commit_message(summary.to_string())
                            .revision(self.spec.revision.clone())
                            .send()?;
                    }
                    "dataset" => {
                        self.client.dataset(owner, name)
                            .create_commit()
                            .operations(ops)
                            .commit_message(summary.to_string())
                            .revision(self.spec.revision.clone())
                            .send()?;
                    }
                    "space" => {
                        self.client.space(owner, name)
                            .create_commit()
                            .operations(ops)
                            .commit_message(summary.to_string())
                            .revision(self.spec.revision.clone())
                            .send()?;
                    }
                    other => anyhow::bail!("unknown repo type {other:?}"),
                }
            }
            SinkState::Bucket { tempdir } => {
                let bucket = self.client.bucket(owner, name);
                let entries = collect_bucket_entries(tempdir.path(), tempdir.path())?;
                log::info!("Uploading {} files to bucket {}", entries.len(), self.spec.repo_id);
                bucket.upload_files().files(entries).send()?;
            }
        }
        Ok(())
    }
}

impl TileSink for HfTileSink {
    fn upload_tile(&self, repo_path: String, png_bytes: Vec<u8>) -> anyhow::Result<()> {
        match &self.state {
            SinkState::Repo { ops } => {
                ops.lock().unwrap().push(CommitOperation::add_bytes(repo_path, png_bytes));
                Ok(())
            }
            SinkState::Bucket { tempdir } => {
                let path = tempdir.path().join(&repo_path);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&path, &png_bytes)?;
                Ok(())
            }
        }
    }
}

fn split_owner_name(repo_id: &str) -> anyhow::Result<(&str, &str)> {
    let slash = repo_id
        .find('/')
        .with_context(|| format!("expected owner/name, got {repo_id:?}"))?;
    Ok((&repo_id[..slash], &repo_id[slash + 1..]))
}

fn collect_bucket_entries(root: &std::path::Path, dir: &std::path::Path) -> anyhow::Result<Vec<BucketUpload>> {
    let mut out = Vec::new();
    walk_files(root, dir, &mut out)?;
    Ok(out)
}

fn walk_files(root: &std::path::Path, dir: &std::path::Path, out: &mut Vec<BucketUpload>) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_files(root, &path, out)?;
        } else if path.is_file() {
            let rel = path
                .strip_prefix(root)
                .with_context(|| format!("path outside root: {}", path.display()))?
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            out.push(BucketUpload::new(PathBuf::from(&path), rel));
        }
    }
    Ok(())
}
