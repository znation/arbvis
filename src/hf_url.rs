use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use futures::StreamExt;
use hf_hub::{
    HFBucket, HFClient, HFRepository, RepoTypeDataset, RepoTypeModel, RepoTypeSpace,
};

use crate::throttle::with_throttle;

/// Repo kind parsed from an `hf://` URL. Carried as a typed value rather than a
/// string so the four upload/download dispatch sites in this crate can match
/// exhaustively and the compiler enforces consistency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepoKind {
    Model,
    Dataset,
    Space,
    Bucket,
}

/// Typed repo handle bundling the kind, revision, and the in-process client.
///
/// Carries the type marker as an enum so `Data::Http` can dispatch range
/// downloads against the right `HFRepository<T>` without leaking generics
/// through the rest of the codebase.
///
/// Buckets are intentionally absent from this enum: the bucket HTTP API has no
/// range-read primitive in hf-hub 1.0, so any caller building a `RemoteRepo`
/// for read access is doing range I/O and bucket URLs are rejected upstream
/// (see [`make_remote_repo`]).
#[derive(Clone)]
pub enum RemoteRepo {
    Model(HFRepository<RepoTypeModel>),
    Dataset(HFRepository<RepoTypeDataset>),
    Space(HFRepository<RepoTypeSpace>),
}

impl RemoteRepo {
    pub async fn fetch_range(&self, filename: &str, revision: &str, range: std::ops::Range<u64>) -> anyhow::Result<Vec<u8>> {
        let label = format!("fetch_range {filename}");
        let bytes = with_throttle(&label, || async {
            match self {
                RemoteRepo::Model(r) => r.download_file_to_bytes().filename(filename).revision(revision).range(range.clone()).send().await,
                RemoteRepo::Dataset(r) => r.download_file_to_bytes().filename(filename).revision(revision).range(range.clone()).send().await,
                RemoteRepo::Space(r) => r.download_file_to_bytes().filename(filename).revision(revision).range(range.clone()).send().await,
            }
        }).await?;
        Ok(bytes.to_vec())
    }

    /// `models`, `datasets`, or `spaces` — the URL segment used in
    /// `/api/{api_segment}/{repo_id}/...` routes.
    pub fn api_segment(&self) -> &'static str {
        match self {
            RemoteRepo::Model(_) => "models",
            RemoteRepo::Dataset(_) => "datasets",
            RemoteRepo::Space(_) => "spaces",
        }
    }

    /// `owner/name` for the underlying repository.
    pub fn repo_id(&self) -> String {
        match self {
            RemoteRepo::Model(r) => format!("{}/{}", r.owner(), r.name()),
            RemoteRepo::Dataset(r) => format!("{}/{}", r.owner(), r.name()),
            RemoteRepo::Space(r) => format!("{}/{}", r.owner(), r.name()),
        }
    }
}

/// A remote HF file accessed via range requests without a full download.
#[derive(Clone)]
pub struct RemoteFileSpec {
    pub repo: RemoteRepo,
    pub filename: Arc<String>,
    pub revision: Arc<String>,
    pub size: u64,
    /// Xet Merkle hash, present iff this file is xet-backed.
    pub xet_hash: Option<String>,
}

/// Parsed destination for streaming output (Hub repo or bucket).
#[derive(Clone)]
pub struct HfOutputSpec {
    pub repo_id: String,
    pub kind: RepoKind,
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

#[derive(Debug)]
pub(crate) struct HfUrl {
    pub(crate) kind: RepoKind,
    pub(crate) repo_id: String,
    pub(crate) revision: String,
    pub(crate) path_in_repo: String,
}

/// Parse an `hf://` URL into its components.
///
/// Supported forms:
///   hf://{owner}/{repo}[@{rev}]            → model (default), repo-level
///   hf://{owner}/{repo}[@{rev}]/{path}     → model (default), single file
///   hf://models/{owner}/{repo}[@{rev}][/{path}]   → model
///   hf://datasets/{owner}/{repo}[@{rev}][/{path}] → dataset
///   hf://spaces/{owner}/{repo}[@{rev}][/{path}]   → space
///   hf://buckets/{owner}/{bucket}[/{path}]         → bucket (no revision concept)
///
/// Empty path segments — including a trailing slash — are stripped so that
/// `hf://owner/repo/path/` parses with `path_in_repo = "path"`, not `"path/"`.
pub(crate) fn parse(raw: &str) -> anyhow::Result<HfUrl> {
    let rest = raw
        .strip_prefix("hf://")
        .ok_or_else(|| anyhow::anyhow!("expected hf:// prefix, got {raw:?}"))?;

    if rest.is_empty() {
        anyhow::bail!("empty hf:// URL");
    }

    // Drop empty segments so `a//b` and trailing/leading slashes don't change parsing.
    let segs: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();

    let (kind, segs) = match segs.first().copied() {
        Some("models") => (RepoKind::Model, &segs[1..]),
        Some("datasets") => (RepoKind::Dataset, &segs[1..]),
        Some("spaces") => (RepoKind::Space, &segs[1..]),
        Some("buckets") => (RepoKind::Bucket, &segs[1..]),
        _ => (RepoKind::Model, &segs[..]),
    };

    if segs.len() < 2 {
        anyhow::bail!(
            "hf:// URL must have the form hf://[type/]owner/repo[@rev][/path], got {raw:?}"
        );
    }

    let owner = segs[0];

    let (repo_name, revision) = if let Some(at) = segs[1].find('@') {
        let rev = &segs[1][at + 1..];
        if rev.is_empty() {
            anyhow::bail!(
                "hf:// URL has an empty revision after '@': {raw:?}; either omit the '@' or specify a branch/commit"
            );
        }
        (&segs[1][..at], rev.to_string())
    } else {
        (segs[1], "main".to_string())
    };

    if owner.is_empty() || repo_name.is_empty() {
        anyhow::bail!("hf:// URL is missing owner or repo name: {raw:?}");
    }

    let repo_id = format!("{owner}/{repo_name}");
    let path_in_repo = if segs.len() >= 3 { segs[2..].join("/") } else { String::new() };

    Ok(HfUrl { kind, repo_id, revision, path_in_repo })
}

/// Shared `HFClient` reused across all HF operations.
///
/// Reads token, endpoint, and cache config from the standard env vars (`HF_TOKEN`,
/// `HF_TOKEN_PATH`, `HF_ENDPOINT`, `HF_HOME`, `HF_HUB_CACHE`).
pub fn client() -> anyhow::Result<HFClient> {
    HFClient::new().context("failed to initialise HF client")
}

/// The HF endpoint (mirrors hf-hub's resolution: `HF_ENDPOINT` env override,
/// else `https://huggingface.co`). Trailing slashes stripped.
pub fn endpoint() -> String {
    let raw = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    raw.trim_end_matches('/').to_string()
}

/// Resolve the HF auth token, returning `None` if no token is available.
///
/// hf-hub resolves the token internally and does not expose a public getter;
/// this mirrors its precedence so we can sign our own requests against the
/// xet endpoints (which hf-hub does not expose publicly).
///
/// Precedence: `HF_TOKEN` env → `HF_TOKEN_PATH` file → `$HF_HOME/token` file (with
/// `HF_HOME` defaulting to `~/.cache/huggingface`). Returns `None` if
/// `HF_HUB_DISABLE_IMPLICIT_TOKEN` is set.
pub fn read_token() -> Option<String> {
    if std::env::var("HF_HUB_DISABLE_IMPLICIT_TOKEN").is_ok_and(|v| !v.is_empty()) {
        return None;
    }
    if let Ok(v) = std::env::var("HF_TOKEN") {
        let t = v.trim();
        if !t.is_empty() {
            return Some(t.to_string());
        }
    }
    if let Ok(p) = std::env::var("HF_TOKEN_PATH") {
        if let Ok(s) = std::fs::read_to_string(&p) {
            let t = s.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    let hf_home = std::env::var("HF_HOME").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.cache/huggingface")
    });
    let token_file = PathBuf::from(&hf_home).join("token");
    if let Ok(s) = std::fs::read_to_string(&token_file) {
        let t = s.trim();
        if !t.is_empty() {
            return Some(t.to_string());
        }
    }
    None
}

/// Returns `Ok(())` if an HF token is resolvable, otherwise an error.
///
/// Used by CLI code that needs to fail with a useful message *before*
/// attempting a write operation that would otherwise fail with a confusing 401.
pub fn require_token() -> anyhow::Result<()> {
    if std::env::var("HF_HUB_DISABLE_IMPLICIT_TOKEN").is_ok_and(|v| !v.is_empty()) {
        anyhow::bail!(
            "HF_HUB_DISABLE_IMPLICIT_TOKEN is set; set HF_TOKEN explicitly or unset this var"
        );
    }
    if read_token().is_some() {
        return Ok(());
    }
    anyhow::bail!(
        "HF token required for hf:// output; set HF_TOKEN or run `hf auth login`"
    )
}

pub fn split_owner_name(repo_id: &str) -> anyhow::Result<(&str, &str)> {
    let slash = repo_id
        .find('/')
        .with_context(|| format!("expected owner/name, got {repo_id:?}"))?;
    Ok((&repo_id[..slash], &repo_id[slash + 1..]))
}

fn make_remote_repo(client: &HFClient, hf: &HfUrl) -> anyhow::Result<RemoteRepo> {
    let (owner, name) = split_owner_name(&hf.repo_id)?;
    match hf.kind {
        RepoKind::Model => Ok(RemoteRepo::Model(client.model(owner, name))),
        RepoKind::Dataset => Ok(RemoteRepo::Dataset(client.dataset(owner, name))),
        RepoKind::Space => Ok(RemoteRepo::Space(client.space(owner, name))),
        // hf-hub 1.0 exposes neither a range-read nor a bytes-stream API for
        // buckets — only whole-file `download_files` and HEAD-style metadata.
        // The Data::Http path requires range reads, so reject up front rather
        // than silently routing bucket URLs through dataset APIs.
        RepoKind::Bucket => anyhow::bail!(
            "bucket URLs do not support range/streaming reads (hf://buckets/{}/...). \
             Download the file first or use a model/dataset/space URL.",
            hf.repo_id
        ),
    }
}

fn make_bucket(client: &HFClient, hf: &HfUrl) -> anyhow::Result<HFBucket> {
    let (owner, name) = split_owner_name(&hf.repo_id)?;
    Ok(client.bucket(owner, name))
}

/// If `path` starts with `hf://`, download and return its local cache path.
/// For repo-level URLs (no file path), downloads all repo files and returns
/// the snapshot directory. Otherwise returns `path` unchanged.
pub async fn resolve(path: &Path) -> anyhow::Result<PathBuf> {
    let s = path.to_string_lossy();
    if !s.starts_with("hf://") {
        return Ok(path.to_path_buf());
    }

    let hf = parse(&s).with_context(|| format!("invalid hf:// URL: {s:?}"))?;
    let cli = client()?;

    if hf.kind == RepoKind::Bucket {
        return resolve_bucket(&cli, &hf).await;
    }

    let repo = make_remote_repo(&cli, &hf)?;

    if hf.path_in_repo.is_empty() {
        log::info!("Resolving repo {} ...", hf.repo_id);
        let dir = with_throttle(&format!("snapshot_download {}", hf.repo_id), || async {
            match &repo {
                RemoteRepo::Model(r) => r.snapshot_download().revision(hf.revision.clone()).send().await,
                RemoteRepo::Dataset(r) => r.snapshot_download().revision(hf.revision.clone()).send().await,
                RemoteRepo::Space(r) => r.snapshot_download().revision(hf.revision.clone()).send().await,
            }
        })
        .await
        .with_context(|| format!("downloading {}", hf.repo_id))?;
        return Ok(dir);
    }

    log::info!("Fetching {} from {} ...", hf.path_in_repo, hf.repo_id);

    let local = with_throttle(&format!("download_file {}", hf.path_in_repo), || async {
        match &repo {
            RemoteRepo::Model(r) => r.download_file().filename(hf.path_in_repo.clone()).revision(hf.revision.clone()).send().await,
            RemoteRepo::Dataset(r) => r.download_file().filename(hf.path_in_repo.clone()).revision(hf.revision.clone()).send().await,
            RemoteRepo::Space(r) => r.download_file().filename(hf.path_in_repo.clone()).revision(hf.revision.clone()).send().await,
        }
    })
    .await
    .with_context(|| format!("fetching hf://{}/{}", hf.repo_id, hf.path_in_repo))?;

    log::info!("Cached at {}", local.display());
    Ok(local)
}

/// Download a bucket file or full bucket tree to a fresh temp directory and return
/// the local path. Buckets have no in-cache `snapshot_download` equivalent, so the
/// caller is handed a tempdir whose lifetime is the same as the process; this
/// matches the existing semantics for repo-level downloads where the cache dir
/// outlives the call.
async fn resolve_bucket(cli: &HFClient, hf: &HfUrl) -> anyhow::Result<PathBuf> {
    use hf_hub::buckets::{BucketDownload, BucketTreeEntry};

    let bucket = make_bucket(cli, hf)?;
    let dest_root = tempfile::Builder::new()
        .prefix("arbvis-bucket-")
        .tempdir()
        .context("creating bucket download tempdir")?
        .keep();

    if hf.path_in_repo.is_empty() {
        log::info!("Resolving bucket {} ...", hf.repo_id);
        // list_tree returns a Stream of entries — collect under a single throttle permit.
        let entries: Vec<BucketTreeEntry> = with_throttle(
            &format!("bucket list_tree {}", hf.repo_id),
            || async {
                let stream = bucket.list_tree().recursive(true).send()?;
                futures::pin_mut!(stream);
                let mut out = Vec::new();
                while let Some(e) = stream.next().await {
                    out.push(e?);
                }
                Ok::<_, hf_hub::HFError>(out)
            },
        )
        .await
        .with_context(|| format!("listing bucket {}", hf.repo_id))?;
        let downloads: Vec<BucketDownload> = entries
            .into_iter()
            .filter_map(|e| match e {
                BucketTreeEntry::File { path, .. } => {
                    let local = dest_root.join(&path);
                    Some(BucketDownload::new(path, local))
                }
                BucketTreeEntry::Directory { .. } => None,
            })
            .collect();
        if downloads.is_empty() {
            anyhow::bail!("bucket {} has no files", hf.repo_id);
        }
        with_throttle(&format!("bucket download_files {}", hf.repo_id), || async {
            bucket.download_files().files(downloads.clone()).send().await
        })
        .await
        .with_context(|| format!("downloading bucket {}", hf.repo_id))?;
        return Ok(dest_root);
    }

    log::info!("Fetching {} from bucket {} ...", hf.path_in_repo, hf.repo_id);
    let local = dest_root.join(&hf.path_in_repo);
    with_throttle(&format!("bucket download_files {}", hf.path_in_repo), || async {
        bucket
            .download_files()
            .files(vec![BucketDownload::new(hf.path_in_repo.clone(), local.clone())])
            .send()
            .await
    })
    .await
    .with_context(|| format!("fetching hf://buckets/{}/{}", hf.repo_id, hf.path_in_repo))?;
    log::info!("Cached at {}", local.display());
    Ok(local)
}

/// Resolve an `hf://` path to a typed `RemoteFileSpec` without downloading.
///
/// Uses hf-hub's HEAD-based `get_file_metadata` to pick up the file size.
pub async fn resolve_to_http(path: &Path) -> anyhow::Result<RemoteFileSpec> {
    let s = path.to_string_lossy();
    let hf = parse(&s).with_context(|| format!("invalid hf:// URL: {s:?}"))?;
    let cli = client()?;
    let repo = make_remote_repo(&cli, &hf)?;
    let meta = with_throttle(&format!("get_file_metadata {}", hf.path_in_repo), || async {
        match &repo {
            RemoteRepo::Model(r) => r.get_file_metadata().filepath(hf.path_in_repo.clone()).revision(hf.revision.clone()).send().await,
            RemoteRepo::Dataset(r) => r.get_file_metadata().filepath(hf.path_in_repo.clone()).revision(hf.revision.clone()).send().await,
            RemoteRepo::Space(r) => r.get_file_metadata().filepath(hf.path_in_repo.clone()).revision(hf.revision.clone()).send().await,
        }
    })
    .await
    .with_context(|| format!("metadata for hf://{}/{}", hf.repo_id, hf.path_in_repo))?;

    log::info!("Remote file {}: {} bytes", hf.path_in_repo, meta.file_size);
    Ok(RemoteFileSpec {
        repo,
        filename: Arc::new(hf.path_in_repo),
        revision: Arc::new(hf.revision),
        size: meta.file_size,
        xet_hash: meta.xet_hash,
    })
}

/// Returns true if the hf:// URL refers to an entire repo (no file path component).
pub fn is_repo_level(url_str: &str) -> anyhow::Result<bool> {
    Ok(parse(url_str)?.path_in_repo.is_empty())
}

/// List all files in a repo-level hf:// URL as `RemoteFileSpec`s without downloading.
pub async fn list_repo_as_http_specs(url_str: &str) -> anyhow::Result<Vec<(String, RemoteFileSpec)>> {
    let hf = parse(url_str).with_context(|| format!("invalid hf:// URL: {url_str:?}"))?;
    let cli = client()?;
    let repo = make_remote_repo(&cli, &hf)?;

    // Collect from the streaming list_tree under one throttle permit.
    // Each variant has its own Stream type because the inferred impl Stream
    // depends on the typed RepoType, so the drain code is duplicated rather
    // than going through a generic closure.
    let entries: Vec<hf_hub::repository::RepoTreeEntry> = with_throttle(
        &format!("list_tree {}", hf.repo_id),
        || async {
            match &repo {
                RemoteRepo::Model(r) => {
                    let stream = r.list_tree().revision(hf.revision.clone()).recursive(true).expand(true).send()?;
                    futures::pin_mut!(stream);
                    let mut out = Vec::new();
                    while let Some(e) = stream.next().await { out.push(e?); }
                    Ok::<_, hf_hub::HFError>(out)
                }
                RemoteRepo::Dataset(r) => {
                    let stream = r.list_tree().revision(hf.revision.clone()).recursive(true).expand(true).send()?;
                    futures::pin_mut!(stream);
                    let mut out = Vec::new();
                    while let Some(e) = stream.next().await { out.push(e?); }
                    Ok::<_, hf_hub::HFError>(out)
                }
                RemoteRepo::Space(r) => {
                    let stream = r.list_tree().revision(hf.revision.clone()).recursive(true).expand(true).send()?;
                    futures::pin_mut!(stream);
                    let mut out = Vec::new();
                    while let Some(e) = stream.next().await { out.push(e?); }
                    Ok::<_, hf_hub::HFError>(out)
                }
            }
        },
    )
    .await
    .with_context(|| format!("listing files for {}", hf.repo_id))?;

    let mut specs = Vec::new();
    let revision = Arc::new(hf.revision);
    for entry in entries {
        if let hf_hub::repository::RepoTreeEntry::File { path, size, xet_hash, .. } = entry {
            log::info!("  {} — {} bytes", path, size);
            specs.push((
                path.clone(),
                RemoteFileSpec {
                    repo: repo.clone(),
                    filename: Arc::new(path),
                    revision: Arc::clone(&revision),
                    size,
                    xet_hash,
                },
            ));
        }
    }

    if specs.is_empty() {
        anyhow::bail!("repo {} has no files", hf.repo_id);
    }
    Ok(specs)
}

/// Parse an `hf://` output URL into an `HfOutputSpec`.
pub fn parse_hf_output(hf_url_str: &str) -> anyhow::Result<HfOutputSpec> {
    let hf = parse(hf_url_str).with_context(|| format!("invalid hf:// output URL: {hf_url_str:?}"))?;
    Ok(HfOutputSpec {
        repo_id: hf.repo_id,
        kind: hf.kind,
        revision: hf.revision,
        path_prefix: hf.path_in_repo,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> anyhow::Result<HfUrl> {
        parse(s)
    }

    #[test]
    fn parse_owner_repo_default_model() {
        let u = p("hf://alice/foo").unwrap();
        assert_eq!(u.kind, RepoKind::Model);
        assert_eq!(u.repo_id, "alice/foo");
        assert_eq!(u.revision, "main");
        assert_eq!(u.path_in_repo, "");
    }

    #[test]
    fn parse_owner_repo_with_revision() {
        let u = p("hf://alice/foo@dev").unwrap();
        assert_eq!(u.revision, "dev");
        assert_eq!(u.path_in_repo, "");
    }

    #[test]
    fn parse_owner_repo_with_file() {
        let u = p("hf://alice/foo/path/to/file.bin").unwrap();
        assert_eq!(u.repo_id, "alice/foo");
        assert_eq!(u.revision, "main");
        assert_eq!(u.path_in_repo, "path/to/file.bin");
    }

    #[test]
    fn parse_typed_prefixes() {
        assert_eq!(p("hf://models/a/b").unwrap().kind, RepoKind::Model);
        assert_eq!(p("hf://datasets/a/b").unwrap().kind, RepoKind::Dataset);
        assert_eq!(p("hf://spaces/a/b").unwrap().kind, RepoKind::Space);
        assert_eq!(p("hf://buckets/a/b").unwrap().kind, RepoKind::Bucket);
    }

    #[test]
    fn parse_dataset_with_revision_and_path() {
        let u = p("hf://datasets/alice/foo@v1/path/to/file").unwrap();
        assert_eq!(u.kind, RepoKind::Dataset);
        assert_eq!(u.repo_id, "alice/foo");
        assert_eq!(u.revision, "v1");
        assert_eq!(u.path_in_repo, "path/to/file");
    }

    #[test]
    fn parse_trailing_slash_strips_to_repo_level() {
        let u = p("hf://alice/foo/").unwrap();
        assert_eq!(u.path_in_repo, "");
    }

    #[test]
    fn parse_trailing_slash_on_path() {
        let u = p("hf://alice/foo/path/").unwrap();
        assert_eq!(u.path_in_repo, "path");
    }

    #[test]
    fn parse_duplicate_slashes_collapse() {
        let u = p("hf://alice/foo//path//file").unwrap();
        assert_eq!(u.path_in_repo, "path/file");
    }

    #[test]
    fn parse_empty_revision_rejected() {
        let err = p("hf://alice/foo@").unwrap_err().to_string();
        assert!(err.contains("empty revision"), "unexpected error: {err}");
    }

    #[test]
    fn parse_missing_prefix_rejected() {
        assert!(p("alice/foo").is_err());
        assert!(p("https://huggingface.co/alice/foo").is_err());
    }

    #[test]
    fn parse_empty_url_rejected() {
        assert!(p("hf://").is_err());
    }

    #[test]
    fn parse_missing_repo_rejected() {
        assert!(p("hf://alice").is_err());
        assert!(p("hf://datasets/alice").is_err());
    }

    #[test]
    fn is_repo_level_returns_error_on_bad_url() {
        assert!(is_repo_level("not-an-hf-url").is_err());
        assert!(is_repo_level("hf://a/b").unwrap());
        assert!(!is_repo_level("hf://a/b/file").unwrap());
    }

    #[test]
    fn bucket_url_parses_without_revision() {
        let u = p("hf://buckets/alice/foo/data/x").unwrap();
        assert_eq!(u.kind, RepoKind::Bucket);
        assert_eq!(u.repo_id, "alice/foo");
        assert_eq!(u.path_in_repo, "data/x");
    }
}
