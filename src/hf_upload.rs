use std::sync::{Arc, Mutex};

use anyhow::Context;
use xet::xet_session::{HeaderMap, HeaderValue, Sha256Policy, XetSessionBuilder, header};

use crate::hf_url::HfOutputSpec;
use crate::tiled::pyramid_accum::TileSink;

struct UploadedFile {
    repo_path: String,
    // XET hash for buckets (from xet_info.hash); SHA-256 for git-backed repos (from xet_info.sha256).
    commit_hash: String,
    size: u64,
}

/// Uploads files to Hugging Face Hub via the Xet CAS protocol.
///
/// For git-backed repos (model/dataset/space) this creates a single Hub commit.
/// For bucket repos this calls the `/api/buckets/{id}/batch` endpoint instead.
///
/// Thread-safe: `upload_tile` can be called concurrently from rayon workers.
pub struct HfXetSession {
    spec: Arc<HfOutputSpec>,
    token: String,
    commit: Arc<xet::xet_session::XetUploadCommit>,
    files: Mutex<Vec<UploadedFile>>,
}

impl HfXetSession {
    pub fn new(spec: &HfOutputSpec, token: String) -> anyhow::Result<Self> {
        // Buckets have no git revision: their token URL omits the revision segment.
        let refresh_url = if spec.repo_type_str == "bucket" {
            format!("{}/api/buckets/{}/xet-write-token", spec.endpoint, spec.repo_id)
        } else {
            format!(
                "{}/api/{}s/{}/xet-write-token/{}",
                spec.endpoint, spec.repo_type_str, spec.repo_id, spec.revision
            )
        };
        let mut hub_headers = HeaderMap::new();
        hub_headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}"))
                .context("invalid token characters")?,
        );

        let session = XetSessionBuilder::new().build().map_err(|e| anyhow::anyhow!("{e}"))?;
        let commit = session
            .new_upload_commit()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_token_refresh_url(refresh_url, hub_headers)
            .build_blocking()
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        Ok(Self {
            spec: Arc::new(spec.clone()),
            token,
            commit: Arc::new(commit),
            files: Mutex::new(Vec::new()),
        })
    }

    /// Upload one file's bytes to Xet CAS. Records the hash for the final Hub commit/batch.
    pub fn upload_file(&self, repo_path: String, bytes: Vec<u8>) -> anyhow::Result<()> {
        let size = bytes.len() as u64;
        // Buckets index by XET hash; git-backed repos need SHA-256 for the lfsFile commit.
        let sha256_policy = if self.spec.repo_type_str == "bucket" {
            Sha256Policy::Skip
        } else {
            Sha256Policy::Compute
        };
        let handle = self.commit
            .upload_bytes_blocking(bytes, sha256_policy, Some(repo_path.clone()))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let meta = handle
            .finalize_ingestion_blocking()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let commit_hash = if self.spec.repo_type_str == "bucket" {
            meta.xet_info.hash.clone()
        } else {
            meta.xet_info.sha256
                .with_context(|| format!("Xet did not compute SHA-256 for {repo_path}"))?
        };

        self.files.lock().unwrap().push(UploadedFile { repo_path, commit_hash, size });
        Ok(())
    }

    /// Finalize CAS commit and push all uploaded files to the Hub.
    pub fn commit(self, summary: &str) -> anyhow::Result<()> {
        let commit = Arc::try_unwrap(self.commit)
            .unwrap_or_else(|a| (*a).clone());
        commit.commit_blocking().map_err(|e| anyhow::anyhow!("{e}"))?;

        let files = self.files.into_inner().unwrap();
        let agent = ureq::AgentBuilder::new().build();

        if self.spec.repo_type_str == "bucket" {
            // Bucket batch API: no header line, each entry is {"type":"addFile",...}
            let batch_url = format!(
                "{}/api/buckets/{}/batch",
                self.spec.endpoint, self.spec.repo_id
            );
            let mut body: Vec<u8> = Vec::new();
            for f in &files {
                let entry = serde_json::json!({
                    "type": "addFile",
                    "path": f.repo_path,
                    "xetHash": f.commit_hash,
                });
                serde_json::to_writer(&mut body, &entry)?;
                body.push(b'\n');
            }
            let resp = agent
                .post(&batch_url)
                .set("Authorization", &format!("Bearer {}", self.token))
                .set("Content-Type", "application/x-ndjson")
                .send_bytes(&body)
                .with_context(|| format!("Bucket batch failed at {batch_url}"))?;
            if resp.status() >= 400 {
                anyhow::bail!("Bucket batch returned HTTP {}", resp.status());
            }
        } else {
            // Git-backed Hub commit: ndjson with header + lfsFile entries.
            let commit_url = format!(
                "{}/api/{}s/{}/commit/{}",
                self.spec.endpoint, self.spec.repo_type_str, self.spec.repo_id, self.spec.revision
            );
            let mut body: Vec<u8> = Vec::new();
            let header_obj = serde_json::json!({"key": "header", "value": {"summary": summary, "description": ""}});
            serde_json::to_writer(&mut body, &header_obj)?;
            body.push(b'\n');
            for f in &files {
                let file_obj = serde_json::json!({
                    "key": "lfsFile",
                    "value": {
                        "path": f.repo_path,
                        "algo": "sha256",
                        "oid": f.commit_hash,
                        "size": f.size,
                    }
                });
                serde_json::to_writer(&mut body, &file_obj)?;
                body.push(b'\n');
            }
            let resp = agent
                .post(&commit_url)
                .set("Authorization", &format!("Bearer {}", self.token))
                .set("Content-Type", "application/x-ndjson")
                .send_bytes(&body)
                .with_context(|| format!("Hub commit failed at {commit_url}"))?;
            if resp.status() >= 400 {
                anyhow::bail!("Hub commit returned HTTP {}", resp.status());
            }
        }
        Ok(())
    }
}

/// `TileSink` adapter that routes each tile through `HfXetSession::upload_file`.
pub struct HfXetTileSink(pub Arc<HfXetSession>);

impl TileSink for HfXetTileSink {
    fn upload_tile(&self, repo_path: String, png_bytes: Vec<u8>) -> anyhow::Result<()> {
        self.0.upload_file(repo_path, png_bytes)
    }
}
