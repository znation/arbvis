use std::path::{Path, PathBuf};

use anyhow::Context;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};

/// A remote HF file that can be range-requested over HTTP without a full download.
pub struct RemoteFileSpec {
    pub cdn_url: String,
    pub size: u64,
    #[allow(dead_code)]
    pub token: Option<String>,
}

/// Parsed destination for Xet-streaming output.
#[derive(Clone)]
pub struct HfOutputSpec {
    pub endpoint: String,
    pub repo_id: String,
    pub repo_type_str: &'static str,  // "model", "dataset", or "space"
    pub revision: String,
    pub path_prefix: String,
}

impl HfOutputSpec {
    /// Path in repo for a tile at zoom z, column x, row y.
    pub fn tile_repo_path(&self, z: u32, x: u32, y: u32) -> String {
        let p = &self.path_prefix;
        if p.is_empty() { format!("tiles/{z}/{x}/{y}.png") }
        else { format!("{p}/tiles/{z}/{x}/{y}.png") }
    }
    pub fn index_html_path(&self) -> String {
        let p = &self.path_prefix;
        if p.is_empty() { "index.html".to_string() }
        else { format!("{p}/index.html") }
    }
    pub fn labels_json_path(&self) -> String {
        let p = &self.path_prefix;
        if p.is_empty() { "labels.json".to_string() }
        else { format!("{p}/labels.json") }
    }
}

struct HfUrl {
    repo_type: RepoType,
    repo_id: String,
    revision: String,
    path_in_repo: String,
}

/// Parse an `hf://` URL into its components.
///
/// Supported forms:
///   hf://{owner}/{repo}[@{rev}]/{path}          → model (default)
///   hf://models/{owner}/{repo}[@{rev}]/{path}   → model
///   hf://datasets/{owner}/{repo}[@{rev}]/{path} → dataset
///   hf://spaces/{owner}/{repo}[@{rev}]/{path}   → space
///   hf://buckets/{owner}/{bucket}[@{rev}]/{path} → dataset (Xet bucket, best-effort)
fn parse(raw: &str) -> anyhow::Result<HfUrl> {
    let rest = raw
        .strip_prefix("hf://")
        .ok_or_else(|| anyhow::anyhow!("expected hf:// prefix, got {raw:?}"))?;

    if rest.is_empty() {
        anyhow::bail!("empty hf:// URL");
    }

    let segs: Vec<&str> = rest.split('/').collect();

    // Consume an optional repo-type prefix.
    let (repo_type, segs) = match segs.first().copied() {
        Some("models") => (RepoType::Model, &segs[1..]),
        Some("datasets") => (RepoType::Dataset, &segs[1..]),
        Some("spaces") => (RepoType::Space, &segs[1..]),
        // Xet-backed buckets don't map cleanly to a Hub repo type;
        // download via the Dataset API as a best-effort fallback.
        Some("buckets") => (RepoType::Dataset, &segs[1..]),
        _ => (RepoType::Model, &segs[..]),
    };

    if segs.len() < 3 {
        anyhow::bail!(
            "hf:// URL must have the form hf://[type/]owner/repo[@@rev]/path, got {raw:?}"
        );
    }

    let owner = segs[0];

    // The repo segment may include an optional `@revision` suffix.
    let (repo_name, revision) = if let Some(at) = segs[1].find('@') {
        (&segs[1][..at], segs[1][at + 1..].to_string())
    } else {
        (segs[1], "main".to_string())
    };

    let repo_id = format!("{owner}/{repo_name}");
    let path_in_repo = segs[2..].join("/");

    if path_in_repo.is_empty() {
        anyhow::bail!("hf:// URL must include a file path after the repo, got {raw:?}");
    }

    Ok(HfUrl {
        repo_type,
        repo_id,
        revision,
        path_in_repo,
    })
}

/// Build an `ApiBuilder` with token from `HF_TOKEN` env var (if set),
/// falling back to the token file at `~/.cache/huggingface/token`.
fn make_api_builder() -> ApiBuilder {
    let mut builder = ApiBuilder::new();
    if let Ok(token) = std::env::var("HF_TOKEN") {
        builder = builder.with_token(Some(token));
    }
    builder
}

/// If `path` starts with `hf://`, download the file and return its local
/// cache path. Otherwise returns `path` unchanged.
///
/// The first download shows a live progress bar via indicatif. Subsequent
/// calls for the same file return instantly from `~/.cache/huggingface/hub/`.
pub fn resolve(path: &Path) -> anyhow::Result<PathBuf> {
    let s = path.to_string_lossy();
    if !s.starts_with("hf://") {
        return Ok(path.to_path_buf());
    }

    let hf = parse(&s).with_context(|| format!("invalid hf:// URL: {s:?}"))?;
    log::info!("Fetching {} from {} ...", hf.path_in_repo, hf.repo_id);

    let api = make_api_builder()
        .build()
        .context("failed to initialise HF API client")?;

    let repo = api.repo(Repo::with_revision(
        hf.repo_id.clone(),
        hf.repo_type,
        hf.revision,
    ));

    let local = repo
        .get(&hf.path_in_repo)
        .with_context(|| format!("failed to fetch hf://{}/{}", hf.repo_id, hf.path_in_repo))?;

    log::info!("Cached at {}", local.display());
    Ok(local)
}

/// Parse an `hf://` output URL and upload a single local file to the target repo.
pub fn upload_file_to(hf_url_str: &str, local: &Path) -> anyhow::Result<()> {
    let hf = parse(hf_url_str)?;
    crate::deploy::upload_file(local, &hf.repo_id, repo_type_name(hf.repo_type), &hf.path_in_repo)
}

/// Parse an `hf://` output URL and upload a local directory tree to the target repo.
pub fn upload_dir_to(hf_url_str: &str, local_dir: &Path) -> anyhow::Result<()> {
    let hf = parse(hf_url_str)?;
    crate::deploy::upload_dir(local_dir, &hf.repo_id, repo_type_name(hf.repo_type), &hf.path_in_repo)
}

pub fn repo_type_name(t: RepoType) -> &'static str {
    match t {
        RepoType::Model => "model",
        RepoType::Dataset => "dataset",
        RepoType::Space => "space",
    }
}

/// Return the HF token from `$HF_TOKEN` env var or `~/.cache/huggingface/token`.
pub fn get_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        if !t.is_empty() { return Some(t); }
    }
    let home = std::env::var("HOME").ok()?;
    let path = std::path::PathBuf::from(home).join(".cache/huggingface/token");
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Resolve an `hf://` path to a CDN URL + file size without downloading the file.
///
/// Uses a HEAD request to get `Content-Length`. The returned `cdn_url` supports
/// `Range: bytes=N-M` requests, enabling per-tile streaming.
pub fn resolve_to_http(path: &Path) -> anyhow::Result<RemoteFileSpec> {
    let s = path.to_string_lossy();
    let hf = parse(&s).with_context(|| format!("invalid hf:// URL: {s:?}"))?;

    let api = make_api_builder()
        .build()
        .context("failed to initialise HF API client")?;
    let repo = api.repo(Repo::with_revision(
        hf.repo_id.clone(),
        hf.repo_type,
        hf.revision.clone(),
    ));
    let cdn_url = repo.url(&hf.path_in_repo);

    let token = get_token();
    let agent = ureq::AgentBuilder::new().build();
    let mut req = agent.head(&cdn_url);
    if let Some(ref t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = req.call()
        .with_context(|| format!("HEAD request failed for {cdn_url}"))?;
    let size = resp
        .header("content-length")
        .and_then(|v| v.parse::<u64>().ok())
        .with_context(|| format!("missing Content-Length for {cdn_url}"))?;

    log::info!("Remote file {}: {} bytes", hf.path_in_repo, size);
    Ok(RemoteFileSpec { cdn_url, size, token })
}

/// Parse an `hf://` output URL into an `HfOutputSpec`.
pub fn parse_hf_output(hf_url_str: &str) -> anyhow::Result<HfOutputSpec> {
    let hf = parse(hf_url_str)
        .with_context(|| format!("invalid hf:// output URL: {hf_url_str:?}"))?;
    Ok(HfOutputSpec {
        endpoint: "https://huggingface.co".to_string(),
        repo_id: hf.repo_id,
        repo_type_str: repo_type_name(hf.repo_type),
        revision: hf.revision,
        path_prefix: hf.path_in_repo,
    })
}
