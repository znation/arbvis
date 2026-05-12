use std::sync::{Arc, Mutex};

use anyhow::Context;
use xet::xet_session::{HeaderMap, HeaderValue, Sha256Policy, XetSessionBuilder, header};

use crate::hf_url::HfOutputSpec;
use crate::tiled::pyramid_accum::TileSink;

struct UploadedFile {
    repo_path: String,
    sha256: String,
    size: u64,
}

/// Uploads files to Hugging Face Hub via the Xet CAS protocol with a single Hub commit.
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
        let refresh_url = format!(
            "{}/api/{}s/{}/xet-write-token/{}",
            spec.endpoint, spec.repo_type_str, spec.repo_id, spec.revision
        );
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

    /// Upload one file's bytes to Xet CAS. Records the SHA-256 for the final Hub commit.
    pub fn upload_file(&self, repo_path: String, bytes: Vec<u8>) -> anyhow::Result<()> {
        let size = bytes.len() as u64;
        let handle = self.commit
            .upload_bytes_blocking(bytes, Sha256Policy::Compute, Some(repo_path.clone()))
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Finalize ingestion to get the SHA-256 immediately (before commit).
        let meta = handle
            .finalize_ingestion_blocking()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let sha256 = meta.xet_info.sha256
            .with_context(|| format!("Xet did not compute SHA-256 for {repo_path}"))?;

        self.files.lock().unwrap().push(UploadedFile { repo_path, sha256, size });
        Ok(())
    }

    /// Finalize CAS commit and create a single Hub repo commit referencing all uploaded files.
    pub fn commit(self, summary: &str) -> anyhow::Result<()> {
        // Finalize the Xet CAS commit (waits for all in-flight uploads).
        let commit = Arc::try_unwrap(self.commit)
            .unwrap_or_else(|a| (*a).clone());
        commit.commit_blocking().map_err(|e| anyhow::anyhow!("{e}"))?;

        let files = self.files.into_inner().unwrap();

        // Build Hub commit body as ndjson.
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
                    "oid": f.sha256,
                    "size": f.size,
                }
            });
            serde_json::to_writer(&mut body, &file_obj)?;
            body.push(b'\n');
        }

        let agent = ureq::AgentBuilder::new().build();
        let resp = agent
            .post(&commit_url)
            .set("Authorization", &format!("Bearer {}", self.token))
            .set("Content-Type", "application/x-ndjson")
            .send_bytes(&body)
            .with_context(|| format!("Hub commit failed at {commit_url}"))?;

        if resp.status() >= 400 {
            anyhow::bail!("Hub commit returned HTTP {}", resp.status());
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
