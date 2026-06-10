use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use anyhow::Context;
use tokio::sync::Mutex as AsyncMutex;

use crate::hf_cli::{self, HfDownloadResult, HfTreeEntry};
use crate::throttle::with_throttle;

/// Repo kind parsed from an `hf://` URL. Carried as a typed value rather than a
/// string so the four upload/download dispatch sites in this crate can match
/// exhaustively and the compiler enforces consistency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RepoKind {
    Model,
    Dataset,
    Space,
    Bucket,
}

impl RepoKind {
    /// `models`, `datasets`, `spaces`, or `buckets` — the URL segment used in
    /// `/api/{api_segment}/{repo_id}/...` routes and the `hf {api_segment}`
    /// subcommand group.
    pub fn api_segment(self) -> &'static str {
        match self {
            RepoKind::Model => "models",
            RepoKind::Dataset => "datasets",
            RepoKind::Space => "spaces",
            RepoKind::Bucket => "buckets",
        }
    }

    /// Value to pass to `hf {download,upload} --type ...`. Only valid for
    /// model/dataset/space — buckets are addressed via the `hf buckets` /
    /// `hf sync` subcommand groups instead.
    pub fn cli_repo_type(self) -> anyhow::Result<&'static str> {
        match self {
            RepoKind::Model => Ok("model"),
            RepoKind::Dataset => Ok("dataset"),
            RepoKind::Space => Ok("space"),
            RepoKind::Bucket => anyhow::bail!(
                "buckets are addressed via `hf buckets` / `hf sync`, not the --type flag"
            ),
        }
    }
}

/// Repo handle for direct-HTTP read paths (`fetch_range`, xet CAS bypass).
///
/// Buckets are intentionally not constructable here: the bucket HTTP surface
/// has no public range-read primitive, so any caller building a `RemoteRepo`
/// for range I/O is rejected upstream (see [`make_remote_repo`]). The Hub I/O
/// that flows through the `hf` CLI uses the [`RepoKind`] + repo-id pair
/// directly and doesn't need this struct.
#[derive(Clone, Debug)]
pub struct RemoteRepo {
    kind: RepoKind,
    repo_id: String,
}

impl RemoteRepo {
    /// `owner/name` for the underlying repository.
    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    /// `models`, `datasets`, or `spaces` — the URL segment used in
    /// `/api/{api_segment}/{repo_id}/...` routes.
    pub fn api_segment(&self) -> &'static str {
        self.kind.api_segment()
    }

    /// Range-fetch `[range.start, range.end)` bytes from `filename` at
    /// `revision`. Direct HTTPS GET to the Hub's `/resolve/` URL with a
    /// `Range` header — `hf` CLI has no byte-range surface and tile
    /// rendering's `--stream` path needs the per-tile range read, so this
    /// stays direct-reqwest.
    pub async fn fetch_range(
        &self,
        filename: &str,
        revision: &str,
        range: std::ops::Range<u64>,
    ) -> anyhow::Result<Vec<u8>> {
        let label = format!("fetch_range {filename}");
        // The Hub `/resolve/` URL doesn't include the `models/` segment for
        // model repos — only `datasets/` and `spaces/` get a prefix. The
        // `/api/` URLs DO include `models/`, which is why `api_segment` here
        // would be wrong.
        let kind_prefix = match self.kind {
            RepoKind::Model => String::new(),
            RepoKind::Dataset => "datasets/".to_string(),
            RepoKind::Space => "spaces/".to_string(),
            // RemoteRepo can't be constructed with Bucket (rejected in
            // `make_remote_repo`), but match exhaustively to keep this honest.
            RepoKind::Bucket => unreachable!("RemoteRepo can't hold a bucket"),
        };
        let url = format!(
            "{}/{}{}/resolve/{}/{}",
            endpoint(),
            kind_prefix,
            self.repo_id,
            revision,
            filename,
        );
        // `Range: bytes=START-END` is inclusive on both sides; our `range.end`
        // is the exclusive Rust convention, so subtract 1 for the header.
        let header = format!("bytes={}-{}", range.start, range.end.saturating_sub(1));
        let bytes = with_throttle(&label, || async {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()?;
            let mut req = client.get(&url).header(reqwest::header::RANGE, &header);
            if let Some(tok) = read_token() {
                req = req.bearer_auth(tok);
            }
            let resp = req.send().await?;
            let resp = resp.error_for_status()?;
            let body = resp.bytes().await?;
            Ok::<_, reqwest::Error>(body)
        })
        .await
        .with_context(|| format!("range GET {url} {header}"))?;
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
    pub fn tile_repo_path(&self, z: u32, x: u32, y: u32, ext: &str) -> String {
        self.tile_repo_path_in(None, z, x, y, ext)
    }
    /// Scene-aware tile path: `[<prefix>/]tiles/[<scene>/]<z>/<x>/<y>.<ext>`.
    /// `scene = None` reproduces the legacy single-pyramid layout.
    pub fn tile_repo_path_in(
        &self,
        scene: Option<&str>,
        z: u32,
        x: u32,
        y: u32,
        ext: &str,
    ) -> String {
        let p = &self.path_prefix;
        let sub = match scene {
            Some(k) => format!("tiles/{k}"),
            None => "tiles".to_string(),
        };
        if p.is_empty() {
            format!("{sub}/{z}/{x}/{y}.{ext}")
        } else {
            format!("{p}/{sub}/{z}/{x}/{y}.{ext}")
        }
    }
    pub fn index_html_path(&self) -> String {
        let p = &self.path_prefix;
        if p.is_empty() {
            "index.html".to_string()
        } else {
            format!("{p}/index.html")
        }
    }
    pub fn labels_json_path(&self) -> String {
        let p = &self.path_prefix;
        if p.is_empty() {
            "labels.json".to_string()
        } else {
            format!("{p}/labels.json")
        }
    }
}

#[derive(Debug)]
pub struct HfUrl {
    pub kind: RepoKind,
    pub repo_id: String,
    pub revision: String,
    pub path_in_repo: String,
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
pub fn parse(raw: &str) -> anyhow::Result<HfUrl> {
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
    let path_in_repo = if segs.len() >= 3 {
        segs[2..].join("/")
    } else {
        String::new()
    };

    Ok(HfUrl {
        kind,
        repo_id,
        revision,
        path_in_repo,
    })
}

/// The HF endpoint (`HF_ENDPOINT` env override, else `https://huggingface.co`).
/// Trailing slashes stripped. Used by the direct-HTTP paths that bypass the
/// `hf` CLI (`fetch_range`, `fetch_model_card`, and `xet.rs`).
pub fn endpoint() -> String {
    let raw = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    raw.trim_end_matches('/').to_string()
}

/// Resolve the HF auth token, returning `None` if no token is available.
///
/// Mirrors the resolution order the `hf` CLI uses internally so the direct
/// HTTP paths (`fetch_range`, `fetch_model_card`, `xet.rs`) sign their
/// requests with the same token the CLI would.
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
    anyhow::bail!("HF token required for hf:// output; set HF_TOKEN or run `hf auth login`")
}

/// Fetch the HF Hub model card metadata for a repo (`/api/models/{repo_id}`).
///
/// The interpretation of fields like `cardData.base_model` /
/// `cardData.base_model_relation` is left to callers — model-specific logic
/// (e.g. finetune auto-detection) lives in
/// [`crate::finetune::detect_relation`], which the modelweightvis split will
/// own.
pub async fn fetch_model_card(repo_id: &str) -> anyhow::Result<serde_json::Value> {
    let url = format!("{}/api/models/{repo_id}", endpoint());
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("building reqwest client")?;
    let mut req = client.get(&url);
    if let Some(tok) = read_token() {
        req = req.bearer_auth(tok);
    }
    let resp = req.send().await.context("HF model_card request failed")?;
    let resp = resp
        .error_for_status()
        .context("HF model_card non-2xx status")?;
    let json: serde_json::Value = resp
        .json()
        .await
        .context("HF model_card JSON decode failed")?;
    Ok(json)
}

pub fn split_owner_name(repo_id: &str) -> anyhow::Result<(&str, &str)> {
    let slash = repo_id
        .find('/')
        .with_context(|| format!("expected owner/name, got {repo_id:?}"))?;
    Ok((&repo_id[..slash], &repo_id[slash + 1..]))
}

fn make_remote_repo(hf: &HfUrl) -> anyhow::Result<RemoteRepo> {
    match hf.kind {
        RepoKind::Model | RepoKind::Dataset | RepoKind::Space => Ok(RemoteRepo {
            kind: hf.kind,
            repo_id: hf.repo_id.clone(),
        }),
        // The bucket HTTP surface has no public range-read primitive; tile
        // rendering's `fetch_range` would have nothing to call. Reject up
        // front rather than silently routing through a different API.
        RepoKind::Bucket => anyhow::bail!(
            "bucket URLs do not support range/streaming reads (hf://buckets/{}/...). \
             Download the file first or use a model/dataset/space URL.",
            hf.repo_id
        ),
    }
}

/// Per-process cache of `hf {kind} list -R --json` (and `hf buckets ls -R
/// --json` for buckets) output, keyed by `(kind, repo_id, revision)`. The
/// CLI doesn't expose a HEAD-style "size of one file" query, so a single
/// metadata lookup requires a full tree listing; without this cache,
/// every `resolve_to_http` call paid for a fresh listing.
///
/// Bucket entries use the empty string for `revision` since buckets don't
/// have a revision concept.
fn listing_cache() -> &'static AsyncMutex<HashMap<(RepoKind, String, String), Arc<Vec<HfTreeEntry>>>>
{
    static CACHE: OnceLock<AsyncMutex<HashMap<(RepoKind, String, String), Arc<Vec<HfTreeEntry>>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| AsyncMutex::new(HashMap::new()))
}

/// Return the recursive listing for `(kind, repo_id, revision)`, populating
/// the per-process cache on first call.
async fn list_repo_entries(
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<Arc<Vec<HfTreeEntry>>> {
    let key = (kind, repo_id.to_string(), revision.to_string());
    {
        let cache = listing_cache().lock().await;
        if let Some(entries) = cache.get(&key) {
            return Ok(Arc::clone(entries));
        }
    }

    let label = format!("list_tree {} {repo_id}@{revision}", kind.api_segment());
    let entries: Vec<HfTreeEntry> = with_throttle(&label, || async {
        match kind {
            RepoKind::Bucket => {
                hf_cli::run_hf_json::<Vec<HfTreeEntry>, _, _>(["buckets", "ls", "-R", repo_id])
                    .await
            }
            _ => {
                hf_cli::run_hf_json::<Vec<HfTreeEntry>, _, _>([
                    kind.api_segment(),
                    "list",
                    "-R",
                    "--revision",
                    revision,
                    repo_id,
                ])
                .await
            }
        }
    })
    .await
    .with_context(|| format!("listing {} {repo_id}@{revision}", kind.api_segment()))?;

    let arc = Arc::new(entries);
    let mut cache = listing_cache().lock().await;
    cache.entry(key).or_insert_with(|| Arc::clone(&arc));
    Ok(arc)
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

    if hf.kind == RepoKind::Bucket {
        return resolve_bucket(&hf).await;
    }

    let repo_type = hf.kind.cli_repo_type()?;
    let label = if hf.path_in_repo.is_empty() {
        log::info!("Resolving repo {} ...", hf.repo_id);
        format!("hf download {}", hf.repo_id)
    } else {
        log::info!("Fetching {} from {} ...", hf.path_in_repo, hf.repo_id);
        format!("hf download {} {}", hf.repo_id, hf.path_in_repo)
    };

    let result = with_throttle(&label, || async {
        // `hf download <repo> [file]`: with `[file]` returns the file path,
        // without returns the snapshot directory. Either way `path` lands
        // under `~/.cache/huggingface/hub/...` (shared with any direct
        // `hf` invocations the user makes outside arbvis).
        let mut args = vec![
            "download".to_string(),
            "--type".to_string(),
            repo_type.to_string(),
            "--revision".to_string(),
            hf.revision.clone(),
            hf.repo_id.clone(),
        ];
        if !hf.path_in_repo.is_empty() {
            args.push(hf.path_in_repo.clone());
        }
        hf_cli::run_hf_json::<HfDownloadResult, _, _>(args.iter().map(String::as_str)).await
    })
    .await
    .with_context(|| format!("downloading hf://{}/{}", hf.repo_id, hf.path_in_repo))?;

    let local = PathBuf::from(result.path);
    log::info!("Cached at {}", local.display());
    Ok(local)
}

/// Download a bucket file or full bucket tree to a fresh temp directory and
/// return the local path. Buckets have no shared cache equivalent, so the
/// caller is handed a tempdir whose lifetime is the process — matching the
/// existing semantics where the cache dir outlives the call.
async fn resolve_bucket(hf: &HfUrl) -> anyhow::Result<PathBuf> {
    let dest_root = tempfile::Builder::new()
        .prefix("arbvis-bucket-")
        .tempdir()
        .context("creating bucket download tempdir")?
        .keep();

    let bucket_id = &hf.repo_id;
    if hf.path_in_repo.is_empty() {
        log::info!("Resolving bucket {} ...", bucket_id);
        let dest = dest_root.to_string_lossy().into_owned();
        let bucket_url = format!("hf://buckets/{bucket_id}");
        with_throttle(&format!("hf sync {bucket_id} -> {dest}"), || async {
            // `hf sync <source> <dest>` infers direction from argument order:
            // bucket source + local dest = download. The destination directory
            // already exists from `tempdir().keep()`, which is what `hf sync`
            // expects.
            hf_cli::run_hf(["sync", bucket_url.as_str(), dest.as_str()]).await
        })
        .await
        .with_context(|| format!("downloading bucket {bucket_id}"))?;
        return Ok(dest_root);
    }

    log::info!("Fetching {} from bucket {} ...", hf.path_in_repo, bucket_id);
    let local = dest_root.join(&hf.path_in_repo);
    if let Some(parent) = local.parent() {
        std::fs::create_dir_all(parent).context("creating bucket-file parent dir")?;
    }
    let src = format!("hf://buckets/{bucket_id}/{}", hf.path_in_repo);
    let dest = local.to_string_lossy().into_owned();
    with_throttle(&format!("hf buckets cp {src} {dest}"), || async {
        hf_cli::run_hf(["buckets", "cp", src.as_str(), dest.as_str()]).await
    })
    .await
    .with_context(|| format!("fetching hf://buckets/{bucket_id}/{}", hf.path_in_repo))?;
    log::info!("Cached at {}", local.display());
    Ok(local)
}

/// Resolve an `hf://` path to a typed `RemoteFileSpec` without downloading.
///
/// Backed by the per-process listing cache so this doesn't pay for a fresh
/// tree listing on every call.
pub async fn resolve_to_http(path: &Path) -> anyhow::Result<RemoteFileSpec> {
    let s = path.to_string_lossy();
    let hf = parse(&s).with_context(|| format!("invalid hf:// URL: {s:?}"))?;
    let repo = make_remote_repo(&hf)?;

    let entries = list_repo_entries(hf.kind, &hf.repo_id, &hf.revision).await?;
    let entry = entries
        .iter()
        .find(|e| e.path == hf.path_in_repo && e.is_file())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no file `{}` in {} {}@{} (or it's a directory)",
                hf.path_in_repo,
                hf.kind.api_segment(),
                hf.repo_id,
                hf.revision,
            )
        })?;

    let size = entry.size.unwrap_or(0);
    log::info!("Remote file {}: {} bytes", hf.path_in_repo, size);
    Ok(RemoteFileSpec {
        repo,
        filename: Arc::new(hf.path_in_repo),
        revision: Arc::new(hf.revision),
        size,
        xet_hash: entry.xet_hash.clone(),
    })
}

/// Returns true if `url_str` is a repo-level `hf://` URL (no file path
/// component).
///
/// A non-`hf://` input (e.g. a local path) returns `Ok(false)` — it isn't an
/// HF URL at all, so by definition it isn't repo-level. Only an actually
/// malformed `hf://` input is an error. This lets call sites use a single
/// `?` to route between the HTTP and local code paths without needing to
/// pre-gate on the prefix.
pub fn is_repo_level(url_str: &str) -> anyhow::Result<bool> {
    if !is_hf_url(url_str) {
        return Ok(false);
    }
    Ok(parse(url_str)?.path_in_repo.is_empty())
}

/// True iff `s` is an `hf://` URL. Centralises the prefix check so call sites
/// don't sprinkle `starts_with("hf://")` everywhere.
pub fn is_hf_url(s: &str) -> bool {
    s.starts_with("hf://")
}

/// Path-typed variant of [`is_hf_url`]: true iff `p`'s textual form starts
/// with `hf://`. Non-UTF-8 paths return `false` (they can't be hf:// URLs).
pub fn is_hf_path(p: &Path) -> bool {
    p.to_str().is_some_and(is_hf_url)
}

/// List all files in a repo-level hf:// URL as `RemoteFileSpec`s without downloading.
pub async fn list_repo_as_http_specs(
    url_str: &str,
) -> anyhow::Result<Vec<(String, RemoteFileSpec)>> {
    let hf = parse(url_str).with_context(|| format!("invalid hf:// URL: {url_str:?}"))?;
    let repo = make_remote_repo(&hf)?;

    let entries = list_repo_entries(hf.kind, &hf.repo_id, &hf.revision).await?;

    let revision = Arc::new(hf.revision);
    let mut specs = Vec::new();
    for entry in entries.iter() {
        if !entry.is_file() {
            continue;
        }
        let size = entry.size.unwrap_or(0);
        let path = entry.path.clone();
        log::info!("  {} — {} bytes", path, size);
        specs.push((
            path.clone(),
            RemoteFileSpec {
                repo: repo.clone(),
                filename: Arc::new(path),
                revision: Arc::clone(&revision),
                size,
                xet_hash: entry.xet_hash.clone(),
            },
        ));
    }

    if specs.is_empty() {
        anyhow::bail!("repo {} has no files", hf.repo_id);
    }
    Ok(specs)
}

/// Parse an `hf://` output URL into an `HfOutputSpec`.
pub fn parse_hf_output(hf_url_str: &str) -> anyhow::Result<HfOutputSpec> {
    let hf =
        parse(hf_url_str).with_context(|| format!("invalid hf:// output URL: {hf_url_str:?}"))?;
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
    fn is_repo_level_treats_non_hf_as_not_a_repo() {
        // A local path or other non-hf:// string is "not a repo URL" rather
        // than an error — the diff dispatcher in main.rs relies on this to
        // route local paths through the local code path.
        assert!(!is_repo_level("not-an-hf-url").unwrap());
        assert!(!is_repo_level("/tmp/foo.safetensors").unwrap());
        // Malformed hf:// inputs still error.
        assert!(is_repo_level("hf://").is_err());
        // Valid repo-level / file-level URLs return the expected bool.
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
