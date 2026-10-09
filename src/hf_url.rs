use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use anyhow::Context;
use tokio::sync::Mutex as AsyncMutex;

use serde::Deserialize;

use crate::hf_cli::{self, HfTreeEntry, HfTreeLfs};
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
/// for range I/O is rejected upstream (see `make_remote_repo`). The Hub I/O
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
        let label = format!("fetch_range {}", sanitize_log_text(filename));
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
            encode_url_path(&self.repo_id),
            encode_url_path(revision),
            encode_url_path(filename),
        );
        // `Range: bytes=START-END` is inclusive on both sides; our `range.end`
        // is the exclusive Rust convention, so subtract 1 for the header.
        let header = format!("bytes={}-{}", range.start, range.end.saturating_sub(1));
        let bytes = with_throttle(&label, || async {
            let mut req = authed_request(
                reqwest::Method::GET,
                &url,
                std::time::Duration::from_secs(60),
            )?;
            req = req.header(reqwest::header::RANGE, &header);
            let resp = req.send().await?;
            let resp = resp.error_for_status()?;
            let body = resp.bytes().await?;
            Ok::<_, reqwest::Error>(body)
        })
        .await
        .with_context(|| format!("range GET {} {header}", sanitize_log_text(&url)))?;
        validate_range_body(&label, &range, &bytes)?;
        Ok(bytes.to_vec())
    }
}

/// Reject a range-GET response whose body does not exactly cover the requested
/// range. A server that ignores or strips the `Range` header replies 200 with
/// the whole file, and a dropped connection can truncate the body; both pass
/// `error_for_status`. Callers slice the returned bytes into fixed-size buffers
/// (e.g. `tiled::leaf::load_tile_bytes`'s `copy_from_slice`), so an unvalidated
/// length turns a server fault into a mid-render panic with an opaque message
/// instead of a clean error naming the fetch.
fn validate_range_body(
    label: &str,
    range: &std::ops::Range<u64>,
    body: &[u8],
) -> anyhow::Result<()> {
    let want = (range.end - range.start) as usize;
    if body.len() == want {
        return Ok(());
    }
    let verdict = if body.len() > want {
        "server ignored the Range header (full-file body)"
    } else {
        "truncated response"
    };
    anyhow::bail!(
        "{}: range {}-{} returned {} bytes (expected {}): {verdict}",
        label,
        range.start,
        range.end.saturating_sub(1),
        body.len(),
        want,
    )
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

/// A parsed HF repo URL: the repo kind and id, the pinned revision, and the
/// file's path inside the repo.
#[derive(Debug)]
pub struct HfUrl {
    pub kind: RepoKind,
    pub repo_id: String,
    pub revision: String,
    pub path_in_repo: String,
}

/// Replace control characters (C0 + C1 + DEL) with `?` before a repo-derived
/// string — a filename from a tree listing, or a URL built from one — is
/// interpolated into a log line, throttle label, or anyhow context. These
/// strings render on the operator's terminal, and a hostile repo can pick
/// file names holding raw escape sequences (ESC [ … m recolors, OSC … BEL
/// rewrites the title / clipboard), so escape bytes must not survive.
pub fn sanitize_log_text(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_control() { '?' } else { c })
        .collect()
}

/// Parse an `hf://` URL into its components.
///
/// Supported forms:
///   `hf://{owner}/{repo}[@{rev}]`                  → model (default), repo-level
///   `hf://{owner}/{repo}[@{rev}]/{path}`           → model (default), single file
///   `hf://models/{owner}/{repo}[@{rev}][/{path}]`  → model
///   `hf://datasets/{owner}/{repo}[@{rev}][/{path}]` → dataset
///   `hf://spaces/{owner}/{repo}[@{rev}][/{path}]`  → space
///   `hf://buckets/{owner}/{bucket}[/{path}]`       → bucket (no revision concept)
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
        let path = segs[2..].join("/");
        // Reject `.`/`..` path segments: this path is later joined onto local
        // roots (e.g. the bucket-cache tempdir in `resolve_bucket`), and a
        // traversal segment would let a hostile URL escape that directory.
        // `.` and `..` are never meaningful in an HF repo path, so parsing
        // rejects them outright rather than hoping every join site checks.
        if segs[2..].iter().any(|seg| *seg == "." || *seg == "..") {
            anyhow::bail!("hf:// URL path segment `.` or `..` is not allowed: {raw:?}");
        }
        path
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
/// `hf` CLI (`fetch_range`, `fetch_model_card`, and `xet/mod.rs`).
pub fn endpoint() -> String {
    let raw = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    raw.trim_end_matches('/').to_string()
}

/// Build an HTTP request against `url` with a timeout and the HF token's
/// bearer auth already applied (`read_token()` is consulted here, so callers
/// must not add it again).
///
/// Returns `reqwest::Result` so both anyhow callers (`.context(...)` on the
/// returned error) and `reqwest::Error` closures (plain `?`) can consume it.
/// Deliberately *not* used by `fetch_tree_via_http`, which captures the token
/// once ahead of its pagination loop; calling this per page would re-read the
/// token source on every request.
pub(crate) fn authed_request(
    method: reqwest::Method,
    url: &str,
    timeout: std::time::Duration,
) -> reqwest::Result<reqwest::RequestBuilder> {
    let client = reqwest::Client::builder().timeout(timeout).build()?;
    let mut req = client.request(method, url);
    if let Some(tok) = read_token() {
        req = req.bearer_auth(tok);
    }
    Ok(req)
}

/// Read the HF auth token, returning `None` if no token is available.
///
/// Mirrors the resolution order the `hf` CLI uses internally so the direct
/// HTTP paths (`fetch_range`, `fetch_model_card`, `xet/mod.rs`) sign their
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
/// `crate::finetune::detect_relation`, which the modelweightvis split will
/// own.
pub async fn fetch_model_card(repo_id: &str) -> anyhow::Result<serde_json::Value> {
    let url = format!("{}/api/models/{}", endpoint(), encode_url_path(repo_id));
    let resp = authed_request(
        reqwest::Method::GET,
        &url,
        std::time::Duration::from_secs(10),
    )
    .context("building reqwest client")?
    .send()
    .await
    .context("HF model_card request failed")?;
    let resp = resp
        .error_for_status()
        .context("HF model_card non-2xx status")?;
    let json: serde_json::Value = resp
        .json()
        .await
        .context("HF model_card JSON decode failed")?;
    Ok(json)
}

/// Split an `owner/name` repo id at its single slash, erroring when the id
/// has none.
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

/// Hub tree-API entry (`GET /api/{kind}/{repo}/tree/{rev}?recursive=true`).
/// Field names follow the JSON API and are mapped into [`HfTreeEntry`].
#[derive(Debug, Deserialize)]
struct TreeApiEntry {
    #[serde(rename = "type")]
    entry_type: String,
    path: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    oid: Option<String>,
    #[serde(rename = "xetHash", default)]
    xet_hash: Option<String>,
    #[serde(default)]
    lfs: Option<TreeApiLfs>,
}

#[derive(Debug, Deserialize)]
struct TreeApiLfs {
    #[serde(default)]
    oid: Option<String>,
    #[serde(default)]
    size: Option<u64>,
    #[serde(rename = "pointerSize", default)]
    pointer_size: Option<u64>,
}

/// List a model/dataset/space repo's files via the Hub tree API.
///
/// Replaces the removed `hf {kind} list -R --json` CLI command — newer
/// huggingface_hub repurposed `hf {kind} list` into a Hub *search*, so the
/// per-repo file listing no longer exists on the CLI. The JSON tree API
/// returns the same per-file metadata arbvis needs (size, blob oid, LFS
/// sha256, xet hash) and paginates large repos via an RFC 5988 `Link`
/// header, which we follow to completion.
/// Upper bound on pages fetched from one tree-API listing. The Hub paginates
/// at 1,000 entries per page, so even multi-million-entry repos stay far
/// below this; hitting it means the `rel="next"` cursor is not advancing
/// (e.g. a broken mirror behind HF_ENDPOINT) and fetching would never end.
const MAX_TREE_PAGES: usize = 1_000;

async fn fetch_tree_via_http(
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<Vec<HfTreeEntry>> {
    fetch_tree_via_http_at(&endpoint(), kind, repo_id, revision).await
}

/// Same as [`fetch_tree_via_http`], against an explicit endpoint base —
/// the test seam that avoids mutating `HF_ENDPOINT` (which races with
/// parallel tests that read it).
async fn fetch_tree_via_http_at(
    base: &str,
    kind: RepoKind,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<Vec<HfTreeEntry>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;
    let token = read_token();
    let label = format!("list_tree {} {repo_id}@{revision}", kind.api_segment());

    let mut url = format!(
        "{}/api/{}/{}/tree/{}?recursive=true",
        base,
        kind.api_segment(),
        encode_url_path(repo_id),
        encode_url_path(revision),
    );

    let mut out: Vec<HfTreeEntry> = Vec::new();
    let mut pages = 0usize;
    loop {
        pages += 1;
        if pages > MAX_TREE_PAGES {
            anyhow::bail!(
                "listing {} {repo_id}@{revision} exceeded {MAX_TREE_PAGES} tree-API pages; \
                 the pagination cursor is not advancing — is HF_ENDPOINT pointing at a \
                 broken mirror?",
                kind.api_segment()
            );
        }
        let (page, next): (Vec<TreeApiEntry>, Option<String>) = with_throttle(&label, || {
            let client = &client;
            let token = token.as_deref();
            let url = url.clone();
            async move {
                let mut req = client.get(&url);
                if let Some(tok) = token {
                    req = req.bearer_auth(tok);
                }
                let resp = req.send().await?.error_for_status()?;
                let next = next_link(resp.headers().get(reqwest::header::LINK));
                let page = resp.json::<Vec<TreeApiEntry>>().await?;
                Ok::<_, reqwest::Error>((page, next))
            }
        })
        .await
        .with_context(|| {
            format!(
                "GET tree page for {} {repo_id}@{revision}",
                kind.api_segment()
            )
        })?;

        out.reserve(page.len());
        for e in page {
            // Directories carry no size; HfTreeEntry::is_file keys off `size`.
            let size = if e.entry_type == "directory" {
                None
            } else {
                e.size
            };
            out.push(HfTreeEntry {
                path: e.path,
                size,
                blob_id: e.oid,
                xet_hash: e.xet_hash,
                lfs: e.lfs.map(|l| HfTreeLfs {
                    sha256: l.oid,
                    size: l.size,
                    pointer_size: l.pointer_size,
                }),
            });
        }

        match next {
            Some(n) => url = n,
            None => break,
        }
    }
    Ok(out)
}

/// Extract the `rel="next"` target from an RFC 5988 `Link` header, if any.
/// The Hub uses this for cursor pagination of the tree API.
fn next_link(header: Option<&reqwest::header::HeaderValue>) -> Option<String> {
    let raw = header?.to_str().ok()?;
    for part in raw.split(',') {
        let mut segs = part.split(';');
        let target = segs.next()?.trim();
        let is_next = segs.any(|s| {
            let s = s.trim();
            s == "rel=\"next\"" || s == "rel=next"
        });
        if is_next {
            return Some(
                target
                    .trim_start_matches('<')
                    .trim_end_matches('>')
                    .to_string(),
            );
        }
    }
    None
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

    let entries: Vec<HfTreeEntry> = match kind {
        RepoKind::Bucket => {
            // Buckets keep their dedicated `hf buckets ls` listing (no tree API).
            let label = format!("list_tree buckets {repo_id}");
            with_throttle(&label, || async {
                hf_cli::run_hf_json::<Vec<HfTreeEntry>, _, _>(["buckets", "ls", "-R", repo_id])
                    .await
            })
            .await
            .with_context(|| format!("listing buckets {repo_id}"))?
        }
        _ => fetch_tree_via_http(kind, repo_id, revision)
            .await
            .with_context(|| format!("listing {} {repo_id}@{revision}", kind.api_segment()))?,
    };

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
    let path_disp = sanitize_log_text(&hf.path_in_repo);
    let label = if hf.path_in_repo.is_empty() {
        log::info!("Resolving repo {} ...", hf.repo_id);
        format!("hf download {}", hf.repo_id)
    } else {
        log::info!("Fetching {} from {} ...", path_disp, hf.repo_id);
        format!("hf download {} {}", hf.repo_id, path_disp)
    };

    let local = with_throttle(&label, || async {
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
        hf_cli::download(args.iter().map(String::as_str)).await
    })
    .await
    .with_context(|| format!("downloading hf://{}/{}", hf.repo_id, path_disp))?;

    log::info!("Cached at {}", sanitize_log_text(&local.to_string_lossy()));
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
        let src_url = bucket_url(bucket_id, "");
        with_throttle(&format!("hf sync {bucket_id} -> {dest}"), || async {
            // `hf sync <source> <dest>` infers direction from argument order:
            // bucket source + local dest = download. The destination directory
            // already exists from `tempdir().keep()`, which is what `hf sync`
            // expects.
            hf_cli::run_hf(["sync", src_url.as_str(), dest.as_str()]).await
        })
        .await
        .with_context(|| format!("downloading bucket {bucket_id}"))?;
        return Ok(dest_root);
    }

    let path_disp = sanitize_log_text(&hf.path_in_repo);
    log::info!("Fetching {} from bucket {} ...", path_disp, bucket_id);
    let local = dest_root.join(&hf.path_in_repo);
    if let Some(parent) = local.parent() {
        std::fs::create_dir_all(parent).context("creating bucket-file parent dir")?;
    }
    let src = bucket_url(bucket_id, &hf.path_in_repo);
    let dest = local.to_string_lossy().into_owned();
    with_throttle(
        &format!("hf buckets cp {src} {}", sanitize_log_text(&dest)),
        || async { hf_cli::run_hf(["buckets", "cp", src.as_str(), dest.as_str()]).await },
    )
    .await
    .with_context(|| format!("fetching hf://buckets/{bucket_id}/{}", path_disp))?;
    log::info!("Cached at {}", sanitize_log_text(&local.to_string_lossy()));
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
                sanitize_log_text(&hf.path_in_repo),
                hf.kind.api_segment(),
                hf.repo_id,
                hf.revision,
            )
        })?;

    let size = entry.size.unwrap_or(0);
    log::info!(
        "Remote file {}: {} bytes",
        sanitize_log_text(&hf.path_in_repo),
        size
    );
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

/// Build a bucket URL for `repo_id`, optionally rooted at `path`.
/// An empty `path` yields the bare bucket URL. Centralises the
/// `hf://buckets/{id}[/{path}]` form so call sites don't hand-format it.
pub fn bucket_url(repo_id: &str, path: &str) -> String {
    if path.is_empty() {
        format!("hf://buckets/{repo_id}")
    } else {
        format!("hf://buckets/{repo_id}/{path}")
    }
}

/// True iff `s` is an `hf://` URL. Centralises the prefix check so call sites
/// don't sprinkle `starts_with("hf://")` everywhere.
/// Convert an `hf://` URL to its Hub web URL, or `None` for non-hf paths and
/// bucket URLs (which have no web viewer page).
///
/// `hf://[type/]owner/repo[@rev][/path]` →
/// `{endpoint}/[type/]owner/repo/{blob|tree}/{rev}/{path}`, with `tree` when the
/// raw path ends in `/`. Parsing is delegated to [`parse`] so owner/repo/rev
/// splitting lives in one place; models intentionally omit the `models/`
/// segment, matching the Hub's web URLs. The host comes from [`endpoint`], so
/// `HF_ENDPOINT` applies here too.
pub fn web_url(raw: &str) -> Option<String> {
    let hf = parse(raw).ok()?;
    if hf.kind == RepoKind::Bucket {
        return None;
    }
    let kind_prefix = match hf.kind {
        RepoKind::Model => String::new(),
        RepoKind::Dataset => "datasets/".to_string(),
        RepoKind::Space => "spaces/".to_string(),
        RepoKind::Bucket => unreachable!("bucket returned above"),
    };
    let base = format!("{}/{kind_prefix}{}", endpoint(), hf.repo_id);
    if hf.path_in_repo.is_empty() {
        return Some(base);
    }
    let tree = raw
        .strip_prefix("hf://")
        .map(|rest| rest.ends_with('/'))
        .unwrap_or(false);
    let verb = if tree { "tree" } else { "blob" };
    Some(format!(
        "{base}/{verb}/{}/{path}",
        hf.revision,
        path = hf.path_in_repo
    ))
}

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
        log::info!("  {} — {} bytes", sanitize_log_text(&path), size);
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

/// Percent-encode a user-supplied identifier (`repo_id`, `revision`, `filename`,
/// `space_id`) for interpolation into a Hub URL path. Unreserved characters and
/// the segment-separating `/` pass through unchanged; everything else —
/// including `?`, `#`, `%`, whitespace, and control bytes — is percent-encoded,
/// so the identifier cannot alter the request's path, query, or fragment.
/// A segment that is exactly `.` or `..` gets its dots encoded, so the URL
/// parser cannot normalize it into path traversal (harmless interior dots in
/// real repo or file names keep their literal form).
pub fn encode_url_path(s: &str) -> String {
    s.split('/')
        .map(|seg| {
            if seg == "." {
                "%2E".to_string()
            } else if seg == ".." {
                "%2E.".to_string()
            } else {
                let mut out = String::with_capacity(seg.len());
                for &b in seg.as_bytes() {
                    match b {
                        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                            out.push(b as char)
                        }
                        _ => out.push_str(&format!("%{b:02X}")),
                    }
                }
                out
            }
        })
        .collect::<Vec<_>>()
        .join("/")
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

    #[tokio::test]
    async fn tree_pagination_stops_when_cursor_never_advances() {
        // Stub server that answers every request with one empty JSON page and
        // a `Link: rel="next"` header pointing back at itself — a
        // non-advancing pagination cursor, as served by a broken mirror
        // behind HF_ENDPOINT. The request body is read (and discarded) so
        // the client sees a well-formed request/response exchange.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let mut buf = [0u8; 4096];
                let _ = tokio::io::AsyncReadExt::read(&mut sock, &mut buf).await;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Link: <http://{addr}/api/models/stub/repo/tree/main>; rel=\"next\"\r\n\
                     Content-Length: 2\r\nConnection: close\r\n\r\n[]"
                );
                let _ = tokio::io::AsyncWriteExt::write_all(&mut sock, resp.as_bytes()).await;
            }
        });
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            fetch_tree_via_http_at(
                &format!("http://{addr}"),
                RepoKind::Model,
                "stub/repo",
                "main",
            ),
        )
        .await;
        let err = result
            .expect("tree listing must terminate, not loop forever")
            .expect_err("a non-advancing cursor must be an error, not a hang");
        assert!(
            err.to_string().contains("tree-API pages"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn validate_range_body_accepts_exact_body() {
        validate_range_body("fetch_range f.bin", &(10..14), &[1, 2, 3, 4]).unwrap();
        // An empty range must accept an empty body.
        validate_range_body("fetch_range f.bin", &(5..5), &[]).unwrap();
    }

    #[test]
    fn validate_range_body_rejects_truncated_body() {
        let err = validate_range_body("fetch_range f.bin", &(0..8), &[1, 2, 3]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("returned 3 bytes (expected 8)"), "{}", msg);
        assert!(msg.contains("truncated response"), "{}", msg);
    }

    #[test]
    fn validate_range_body_rejects_full_file_body_for_ignored_range() {
        let err =
            validate_range_body("fetch_range f.bin", &(1_000..1_004), &[0u8; 10]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("returned 10 bytes (expected 4)"), "{}", msg);
        assert!(msg.contains("ignored the Range header"), "{}", msg);
    }

    #[test]
    fn sanitize_log_text_strips_control_and_escape_bytes() {
        // ANSI color / OSC-title sequences a hostile repo can put in a file
        // name collapse to plain `?`s.
        assert_eq!(sanitize_log_text("\x1b[31mred\x1b[0m"), "?[31mred?[0m");
        assert_eq!(sanitize_log_text("\x1b]0;pwned\x07"), "?]0;pwned?");
        // Newlines / tabs / C1 / DEL cannot forge log-line boundaries either.
        assert_eq!(sanitize_log_text("a\nb\tc\u{85}d\u{7f}"), "a?b?c?d?");
    }

    #[test]
    fn sanitize_log_text_leaves_normal_names_untouched() {
        assert_eq!(sanitize_log_text("model.safetensors"), "model.safetensors");
        // Non-ASCII printable (CJK, emoji) is preserved.
        assert_eq!(sanitize_log_text("权重 🦙.bin"), "权重 🦙.bin");
    }

    #[test]
    fn encode_url_path_blocks_query_fragment_and_traversal_injection() {
        // `?` / `#` cannot start a query or fragment inside the URL path.
        assert_eq!(encode_url_path("main?x=1"), "main%3Fx%3D1");
        assert_eq!(encode_url_path("dev#frag"), "dev%23frag");
        // A literal `%` cannot forge an escape sequence.
        assert_eq!(encode_url_path("100%main"), "100%25main");
        // `.`/`..` segments are encoded, so `../` cannot escape the API path.
        assert_eq!(
            encode_url_path("../../api/models/o/r"),
            "%2E./%2E./api/models/o/r"
        );
        // Whitespace and control bytes are encoded too.
        assert_eq!(encode_url_path("a b\tc"), "a%20b%09c");
    }

    #[test]
    fn encode_url_path_leaves_legitimate_identifiers_untouched() {
        // Owner/repo slash, dots, and dashes survive; non-ASCII names are
        // percent-encoded as UTF-8, which the Hub decodes identically.
        assert_eq!(encode_url_path("alice/foo.bar"), "alice/foo.bar");
        assert_eq!(encode_url_path("refs/pr/1"), "refs/pr/1");
        assert_eq!(encode_url_path("权/重.bin"), "%E6%9D%83/%E9%87%8D.bin");
    }

    #[test]
    fn bucket_url_omits_empty_path() {
        assert_eq!(bucket_url("id", ""), "hf://buckets/id");
        assert_eq!(bucket_url("id", "tiles"), "hf://buckets/id/tiles");
    }

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
    fn parse_rejects_traversal_segments() {
        // `..` in the path would escape local roots when the path is joined
        // onto a download directory (e.g. resolve_bucket's tempdir).
        for url in [
            "hf://buckets/alice/repo/../../../etc/passwd",
            "hf://alice/repo/../secret",
            "hf://alice/repo/a/./b",
            "hf://alice/repo/..",
        ] {
            assert!(p(url).is_err(), "expected rejection of {url}");
        }
        // A segment merely containing dots is fine.
        assert_eq!(
            p("hf://alice/repo/file.model.bin").unwrap().path_in_repo,
            "file.model.bin"
        );
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
    fn web_url_bare_model_repo() {
        assert_eq!(
            web_url("hf://owner/repo"),
            Some("https://huggingface.co/owner/repo".to_string())
        );
    }

    #[test]
    fn web_url_bare_dataset_repo() {
        assert_eq!(
            web_url("hf://datasets/owner/repo"),
            Some("https://huggingface.co/datasets/owner/repo".to_string())
        );
    }

    #[test]
    fn web_url_file_in_model_repo() {
        assert_eq!(
            web_url("hf://owner/repo/model.safetensors"),
            Some("https://huggingface.co/owner/repo/blob/main/model.safetensors".to_string())
        );
    }

    #[test]
    fn web_url_file_in_dataset_repo() {
        assert_eq!(
            web_url("hf://datasets/owner/repo/data.safetensors"),
            Some(
                "https://huggingface.co/datasets/owner/repo/blob/main/data.safetensors".to_string()
            )
        );
    }

    #[test]
    fn web_url_directory_uses_tree_verb() {
        assert_eq!(
            web_url("hf://owner/repo/data/"),
            Some("https://huggingface.co/owner/repo/tree/main/data".to_string())
        );
    }

    #[test]
    fn web_url_non_hf_or_partial_returns_none() {
        assert_eq!(web_url("/local/path/file.safetensors"), None);
        assert_eq!(web_url("hf://owner"), None);
    }

    #[test]
    fn web_url_bucket_returns_none() {
        assert_eq!(web_url("hf://buckets/alice/foo/data/x"), None);
    }

    #[test]
    fn bucket_url_parses_without_revision() {
        let u = p("hf://buckets/alice/foo/data/x").unwrap();
        assert_eq!(u.kind, RepoKind::Bucket);
        assert_eq!(u.repo_id, "alice/foo");
        assert_eq!(u.path_in_repo, "data/x");
    }
}
