use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use hf_hub::{
    HFClientSync, HFRepositorySync, RepoTypeDataset, RepoTypeModel, RepoTypeSpace,
};

/// Typed repo handle bundling the kind, revision, and the in-process client.
///
/// Carries the type marker as an enum so `Data::Http` can dispatch range
/// downloads against the right `HFRepositorySync<T>` without leaking generics
/// through the rest of the codebase.
#[derive(Clone)]
pub enum RemoteRepo {
    Model(HFRepositorySync<RepoTypeModel>),
    Dataset(HFRepositorySync<RepoTypeDataset>),
    Space(HFRepositorySync<RepoTypeSpace>),
}

impl RemoteRepo {
    pub fn fetch_range(&self, filename: &str, revision: &str, range: std::ops::Range<u64>) -> anyhow::Result<Vec<u8>> {
        let bytes = match self {
            RemoteRepo::Model(r) => r.download_file_to_bytes().filename(filename).revision(revision).range(range).send(),
            RemoteRepo::Dataset(r) => r.download_file_to_bytes().filename(filename).revision(revision).range(range).send(),
            RemoteRepo::Space(r) => r.download_file_to_bytes().filename(filename).revision(revision).range(range).send(),
        }?;
        Ok(bytes.to_vec())
    }
}

/// A remote HF file accessed via range requests without a full download.
#[derive(Clone)]
pub struct RemoteFileSpec {
    pub repo: RemoteRepo,
    pub filename: Arc<String>,
    pub revision: Arc<String>,
    pub size: u64,
}

/// Parsed destination for streaming output (Hub repo or bucket).
#[derive(Clone)]
pub struct HfOutputSpec {
    pub repo_id: String,
    pub repo_type_str: &'static str, // "model", "dataset", "space", or "bucket"
    pub revision: String,
    pub path_prefix: String,
}

impl HfOutputSpec {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParsedRepoKind {
    Model,
    Dataset,
    Space,
    Bucket,
}

impl ParsedRepoKind {
    fn as_str(self) -> &'static str {
        match self {
            ParsedRepoKind::Model => "model",
            ParsedRepoKind::Dataset => "dataset",
            ParsedRepoKind::Space => "space",
            ParsedRepoKind::Bucket => "bucket",
        }
    }
}

struct HfUrl {
    kind: ParsedRepoKind,
    repo_id: String,
    revision: String,
    path_in_repo: String,
}

/// Parse an `hf://` URL into its components.
///
/// Supported forms:
///   hf://{owner}/{repo}[@{rev}]            → model (default), repo-level
///   hf://{owner}/{repo}[@{rev}]/{path}     → model (default), single file
///   hf://models/{owner}/{repo}[@{rev}][/{path}]   → model
///   hf://datasets/{owner}/{repo}[@{rev}][/{path}] → dataset
///   hf://spaces/{owner}/{repo}[@{rev}][/{path}]   → space
///   hf://buckets/{owner}/{bucket}[/{path}]         → bucket (Xet; no revision concept)
fn parse(raw: &str) -> anyhow::Result<HfUrl> {
    let rest = raw
        .strip_prefix("hf://")
        .ok_or_else(|| anyhow::anyhow!("expected hf:// prefix, got {raw:?}"))?;

    if rest.is_empty() {
        anyhow::bail!("empty hf:// URL");
    }

    let segs: Vec<&str> = rest.split('/').collect();

    let (kind, segs) = match segs.first().copied() {
        Some("models") => (ParsedRepoKind::Model, &segs[1..]),
        Some("datasets") => (ParsedRepoKind::Dataset, &segs[1..]),
        Some("spaces") => (ParsedRepoKind::Space, &segs[1..]),
        Some("buckets") => (ParsedRepoKind::Bucket, &segs[1..]),
        _ => (ParsedRepoKind::Model, &segs[..]),
    };

    if segs.len() < 2 {
        anyhow::bail!(
            "hf:// URL must have the form hf://[type/]owner/repo[@@rev][/path], got {raw:?}"
        );
    }

    let owner = segs[0];

    let (repo_name, revision) = if let Some(at) = segs[1].find('@') {
        (&segs[1][..at], segs[1][at + 1..].to_string())
    } else {
        (segs[1], "main".to_string())
    };

    let repo_id = format!("{owner}/{repo_name}");
    let path_in_repo = if segs.len() >= 3 { segs[2..].join("/") } else { String::new() };

    Ok(HfUrl { kind, repo_id, revision, path_in_repo })
}

/// Shared `HFClientSync` reused across all HF operations.
///
/// Reads token, endpoint, and cache config from the standard env vars (`HF_TOKEN`,
/// `HF_TOKEN_PATH`, `HF_ENDPOINT`, `HF_HOME`, `HF_HUB_CACHE`).
pub fn client() -> anyhow::Result<HFClientSync> {
    HFClientSync::new().context("failed to initialise HF client")
}

fn split_owner_name(repo_id: &str) -> anyhow::Result<(&str, &str)> {
    let slash = repo_id.find('/').with_context(|| format!("expected owner/name, got {repo_id:?}"))?;
    Ok((&repo_id[..slash], &repo_id[slash + 1..]))
}

fn make_remote_repo(client: &HFClientSync, hf: &HfUrl) -> anyhow::Result<RemoteRepo> {
    let (owner, name) = split_owner_name(&hf.repo_id)?;
    Ok(match hf.kind {
        ParsedRepoKind::Model => RemoteRepo::Model(client.model(owner, name)),
        // Buckets don't have a single read-side equivalent in the typed API; fall
        // back to a dataset handle (this preserves the prior 0.5 behavior).
        ParsedRepoKind::Dataset | ParsedRepoKind::Bucket => {
            RemoteRepo::Dataset(client.dataset(owner, name))
        }
        ParsedRepoKind::Space => RemoteRepo::Space(client.space(owner, name)),
    })
}

/// If `path` starts with `hf://`, download and return its local cache path.
/// For repo-level URLs (no file path), downloads all repo files and returns
/// the snapshot directory. Otherwise returns `path` unchanged.
pub fn resolve(path: &Path) -> anyhow::Result<PathBuf> {
    let s = path.to_string_lossy();
    if !s.starts_with("hf://") {
        return Ok(path.to_path_buf());
    }

    let hf = parse(&s).with_context(|| format!("invalid hf:// URL: {s:?}"))?;
    let cli = client()?;
    let repo = make_remote_repo(&cli, &hf)?;

    if hf.path_in_repo.is_empty() {
        log::info!("Resolving repo {} ...", hf.repo_id);
        let dir = match &repo {
            RemoteRepo::Model(r) => r.snapshot_download().revision(hf.revision.clone()).send(),
            RemoteRepo::Dataset(r) => r.snapshot_download().revision(hf.revision.clone()).send(),
            RemoteRepo::Space(r) => r.snapshot_download().revision(hf.revision.clone()).send(),
        }
        .with_context(|| format!("downloading {}", hf.repo_id))?;
        return Ok(dir);
    }

    log::info!("Fetching {} from {} ...", hf.path_in_repo, hf.repo_id);

    let local = match &repo {
        RemoteRepo::Model(r) => r.download_file().filename(hf.path_in_repo.clone()).revision(hf.revision.clone()).send(),
        RemoteRepo::Dataset(r) => r.download_file().filename(hf.path_in_repo.clone()).revision(hf.revision.clone()).send(),
        RemoteRepo::Space(r) => r.download_file().filename(hf.path_in_repo.clone()).revision(hf.revision.clone()).send(),
    }
    .with_context(|| format!("fetching hf://{}/{}", hf.repo_id, hf.path_in_repo))?;

    log::info!("Cached at {}", local.display());
    Ok(local)
}

/// Resolve an `hf://` path to a typed `RemoteFileSpec` without downloading.
///
/// Uses hf-hub's HEAD-based `get_file_metadata` to pick up the file size.
pub fn resolve_to_http(path: &Path) -> anyhow::Result<RemoteFileSpec> {
    let s = path.to_string_lossy();
    let hf = parse(&s).with_context(|| format!("invalid hf:// URL: {s:?}"))?;
    let cli = client()?;
    let repo = make_remote_repo(&cli, &hf)?;
    let meta = match &repo {
        RemoteRepo::Model(r) => r.get_file_metadata().filepath(hf.path_in_repo.clone()).revision(hf.revision.clone()).send(),
        RemoteRepo::Dataset(r) => r.get_file_metadata().filepath(hf.path_in_repo.clone()).revision(hf.revision.clone()).send(),
        RemoteRepo::Space(r) => r.get_file_metadata().filepath(hf.path_in_repo.clone()).revision(hf.revision.clone()).send(),
    }
    .with_context(|| format!("metadata for hf://{}/{}", hf.repo_id, hf.path_in_repo))?;

    log::info!("Remote file {}: {} bytes", hf.path_in_repo, meta.file_size);
    Ok(RemoteFileSpec {
        repo,
        filename: Arc::new(hf.path_in_repo),
        revision: Arc::new(hf.revision),
        size: meta.file_size,
    })
}

/// Returns true if the hf:// URL refers to an entire repo (no file path component).
pub fn is_repo_level(url_str: &str) -> bool {
    parse(url_str).map(|h| h.path_in_repo.is_empty()).unwrap_or(false)
}

/// List all files in a repo-level hf:// URL as `RemoteFileSpec`s without downloading.
pub fn list_repo_as_http_specs(url_str: &str) -> anyhow::Result<Vec<(String, RemoteFileSpec)>> {
    let hf = parse(url_str).with_context(|| format!("invalid hf:// URL: {url_str:?}"))?;
    let cli = client()?;
    let repo = make_remote_repo(&cli, &hf)?;

    let entries = match &repo {
        RemoteRepo::Model(r) => r.list_tree().revision(hf.revision.clone()).recursive(true).expand(true).send(),
        RemoteRepo::Dataset(r) => r.list_tree().revision(hf.revision.clone()).recursive(true).expand(true).send(),
        RemoteRepo::Space(r) => r.list_tree().revision(hf.revision.clone()).recursive(true).expand(true).send(),
    }
    .with_context(|| format!("listing files for {}", hf.repo_id))?;

    let mut specs = Vec::new();
    let revision = Arc::new(hf.revision);
    for entry in entries {
        if let hf_hub::repository::RepoTreeEntry::File { path, size, .. } = entry {
            log::info!("  {} — {} bytes", path, size);
            specs.push((
                path.clone(),
                RemoteFileSpec {
                    repo: repo.clone(),
                    filename: Arc::new(path),
                    revision: Arc::clone(&revision),
                    size,
                },
            ));
        }
    }

    if specs.is_empty() {
        anyhow::bail!("repo {} has no files", hf.repo_id);
    }
    Ok(specs)
}

/// Parse an `hf://` output URL and upload a single local file to the target repo.
pub fn upload_file_to(hf_url_str: &str, local: &Path) -> anyhow::Result<()> {
    let hf = parse(hf_url_str)?;
    crate::deploy::upload_file(local, &hf.repo_id, hf.kind.as_str(), &hf.path_in_repo)
}

/// Parse an `hf://` output URL and upload a local directory tree to the target repo.
pub fn upload_dir_to(hf_url_str: &str, local_dir: &Path) -> anyhow::Result<()> {
    let hf = parse(hf_url_str)?;
    crate::deploy::upload_dir(local_dir, &hf.repo_id, hf.kind.as_str(), &hf.path_in_repo)
}

/// Parse an `hf://` output URL into an `HfOutputSpec`.
pub fn parse_hf_output(hf_url_str: &str) -> anyhow::Result<HfOutputSpec> {
    let hf = parse(hf_url_str).with_context(|| format!("invalid hf:// output URL: {hf_url_str:?}"))?;
    Ok(HfOutputSpec {
        repo_id: hf.repo_id,
        repo_type_str: hf.kind.as_str(),
        revision: hf.revision,
        path_prefix: hf.path_in_repo,
    })
}
