use std::path::{Path, PathBuf};

use anyhow::Context;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};

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

fn repo_type_name(t: RepoType) -> &'static str {
    match t {
        RepoType::Model => "model",
        RepoType::Dataset => "dataset",
        RepoType::Space => "space",
    }
}
