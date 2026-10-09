//! Xet protocol plumbing: wire types for the two Hub endpoints plus the
//! helpers that fetch a CAS token and a V2 reconstruction response.
//!
//! Split from `mod.rs` (found 2026-10-09); moved verbatim except for
//! visibility — the helpers are `pub(super)` so `mod.rs`'s public surface
//! (`reconstruction_for`, `XetReader`) is unchanged.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{anyhow, Context};
use serde::{de::DeserializeOwned, Deserialize};

use super::{http_client, XetTerm};
use crate::hf_url;
use crate::throttle::with_throttle;

/// JSON response from the Hub's `xet-read-token` endpoint: a short-lived CAS
/// bearer token plus the base URL of the CAS service it authenticates against.
#[derive(Deserialize)]
pub(super) struct XetReadTokenResponse {
    /// Bearer token for the CAS API; sent as `Authorization` on the
    /// reconstruction request.
    #[serde(rename = "accessToken")]
    pub(super) access_token: String,
    /// Base URL of the CAS service (e.g. `https://cas-bridge.xethub.hf.co`);
    /// trailing slashes are stripped before caching.
    #[serde(rename = "casUrl")]
    pub(super) cas_url: String,
}

/// JSON wire form of `ChunkRange`/`HttpRange`/`FileRange` from xet-client's
/// `cas_types::Range<Idx, Kind>` — the `_marker` field is `#[serde(skip)]`
/// so only `start`/`end` go on the wire.
#[derive(Deserialize, Clone, Copy)]
pub(super) struct WireRange<T> {
    /// First index or byte of the range; inclusive.
    pub(super) start: T,
    /// Last index or byte of the range; inclusivity depends on the context —
    /// chunk-index ranges are half-open `[start, end)`, packed-byte ranges
    /// are closed `[start, end]` (see `WireXorbRangeDescriptor`).
    pub(super) end: T,
}

/// One contiguous run of bytes in the reconstructed file, backed by a chunk
/// sequence from a single xorb. The reconstruction is the terms list in order.
#[derive(Deserialize)]
pub(super) struct ReconstructionTerm {
    /// Hash of the xorb holding this term's chunks (e.g. `"abc/123"`); keys
    /// the `xorbs` map on a V2 response.
    pub(super) hash: String,
    /// Number of bytes this term contributes to the file once unpacked.
    #[serde(rename = "unpacked_length")]
    pub(super) unpacked_length: u64,
    /// Chunk index `[start, end)` within the xorb. Captured so the reader can
    /// map a file-byte range to the exact chunks it spans (vs. approximating
    /// from term boundaries alone).
    pub(super) range: WireRange<u32>,
}

/// Per-xorb byte range fetch instructions: one signed URL covers some chunks,
/// described by `chunks` (chunk index range) and `bytes` (packed-byte range
/// inside the xorb, *inclusive end* — `HttpRange` semantics).
#[derive(Deserialize)]
pub(super) struct WireXorbRangeDescriptor {
    /// Chunk index range `[start, end)` inside the xorb to fetch.
    pub(super) chunks: WireRange<u32>,
    /// Packed-byte range inside the xorb, inclusive end (`HttpRange` semantics).
    pub(super) bytes: WireRange<u64>,
}

/// One entry of a V2 response's `xorbs` map: a signed URL that serves the
/// requested byte ranges of a single xorb, plus which ranges to fetch from it.
#[derive(Deserialize)]
pub(super) struct WireXorbMultiRangeFetch {
    /// Pre-signed GET URL for the xorb's chunk endpoint; valid without
    /// additional auth headers for its lifetime.
    pub(super) url: String,
    /// The disjoint byte ranges this URL covers, each as chunk indices plus
    /// packed-byte offsets.
    pub(super) ranges: Vec<WireXorbRangeDescriptor>,
}

#[derive(Deserialize)]
pub(super) struct ReconstructionResponse {
    /// The file's chunk terms in file order; concatenating their unpacked
    /// bytes (skipping zero-length terms) reproduces the file.
    pub(super) terms: Vec<ReconstructionTerm>,
    /// V2-only: per-xorb signed-URL fetch info. Absent on V1 responses; we
    /// require V2 for the direct-CAS reader path, so callers that need it
    /// should error if this field is missing.
    #[serde(default)]
    pub(super) xorbs: HashMap<String, Vec<WireXorbMultiRangeFetch>>,
}

/// A cached CAS credential: the service base URL and its bearer token, in
/// the normalized form used to build reconstruction URLs.
#[derive(Clone)]
pub(super) struct CasToken {
    /// CAS service base URL with trailing slashes stripped.
    pub(super) cas_url: String,
    /// Bearer token for CAS API requests.
    pub(super) access_token: String,
}

/// Per-process cache of CAS tokens, keyed by `(api_segment, repo_id, revision)`.
/// Tokens expire (the response includes an `exp` field) but for arbvis runs
/// they live well within the expiration window of a single visualization.
static CAS_TOKEN_CACHE: Mutex<Option<HashMap<(String, String, String), CasToken>>> =
    Mutex::new(None);

/// Drops the cached CAS token for `(api_segment, repo_id, revision)`, if any.
/// A no-op when the key is absent or the cache was never initialized; safe to
/// call before any token has been fetched.
pub(super) fn invalidate_cas_token_cache(api_segment: &str, repo_id: &str, revision: &str) {
    let key = (
        api_segment.to_string(),
        repo_id.to_string(),
        revision.to_string(),
    );
    let mut guard = CAS_TOKEN_CACHE.lock().unwrap();
    if let Some(cache) = guard.as_mut() {
        cache.remove(&key);
    }
}

/// Authenticated GET that runs under the global throttle, converts non-2xx
/// responses into reqwest errors via `error_for_status()`, and parses the
/// response body as JSON. `throttle_key` labels the request to the throttle;
/// `label` names it in the log line and in the error contexts
/// ("requesting {label} at {url}", "parsing {label} response from {url}").
async fn authed_get_json<T: DeserializeOwned>(
    throttle_key: &str,
    label: &str,
    url: &str,
    bearer: &str,
) -> anyhow::Result<T> {
    log::info!("Fetching {label}: {url}");
    let client = http_client();
    let resp = with_throttle(throttle_key, || async {
        client
            .get(url)
            .bearer_auth(bearer)
            .send()
            .await
            .and_then(|r| r.error_for_status())
    })
    .await
    .with_context(|| format!("requesting {label} at {url}"))?;
    resp.json::<T>()
        .await
        .with_context(|| format!("parsing {label} response from {url}"))
}

/// Everything needed to mint a CAS token for one repo/revision: the Hub
/// endpoint, the repo coordinates, and the Hub bearer token. Owns its strings
/// so callers (including tests, which point `endpoint` at a stub server) can
/// build it once and pass it around.
#[derive(Clone)]
pub(super) struct CasContext {
    /// Hub base URL (`hf_url::endpoint()` in production; `http://127.0.0.1` in
    /// tests). Trailing slashes must be stripped before storing.
    pub(super) endpoint: String,
    /// Repo kind's API segment (e.g. `models`).
    pub(super) api_segment: String,
    /// `owner/name` repo id.
    pub(super) repo_id: String,
    /// Revision (branch/tag/commit).
    pub(super) revision: String,
    /// Hub bearer token sent to the `xet-read-token` endpoint.
    pub(super) bearer: String,
}

/// Builds a [`CasContext`] from the production Hub endpoint and the user's HF
/// token. Errors when no HF token is available.
pub(super) fn cas_context(
    api_segment: &str,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<CasContext> {
    Ok(CasContext {
        endpoint: hf_url::endpoint(),
        api_segment: api_segment.to_string(),
        repo_id: repo_id.to_string(),
        revision: revision.to_string(),
        bearer: hf_url::read_token().ok_or_else(|| {
            anyhow!("HF token required for xet reconstruction; set HF_TOKEN or run `hf auth login`")
        })?,
    })
}

/// Fetches (or returns from `CAS_TOKEN_CACHE`) a CAS token for reading
/// `repo_id` at `revision` via the context's endpoint and bearer token. The
/// returned token's `cas_url` is trimmed of trailing slashes so URL joins are
/// safe.
pub(super) async fn fetch_cas_token_at(ctx: &CasContext) -> anyhow::Result<CasToken> {
    let key = (
        ctx.api_segment.clone(),
        ctx.repo_id.clone(),
        ctx.revision.clone(),
    );
    {
        let mut guard = CAS_TOKEN_CACHE.lock().unwrap();
        let cache = guard.get_or_insert_with(HashMap::new);
        if let Some(t) = cache.get(&key) {
            return Ok(t.clone());
        }
    }

    let url = format!(
        "{}/api/{}/{}/xet-read-token/{}",
        ctx.endpoint, ctx.api_segment, ctx.repo_id, ctx.revision,
    );
    // `error_for_status()` (inside `authed_get_json`) converts non-2xx into a
    // reqwest::Error carrying the status code so the throttle's classifier can
    // detect 429/5xx and retry. Response body detail is lost on error, but the
    // URL and status code are preserved.
    let parsed: XetReadTokenResponse = authed_get_json(
        &format!("xet-read-token {}", ctx.repo_id),
        "xet-read-token",
        &url,
        &ctx.bearer,
    )
    .await?;

    let token = CasToken {
        cas_url: parsed.cas_url.trim_end_matches('/').to_string(),
        access_token: parsed.access_token,
    };
    {
        let mut guard = CAS_TOKEN_CACHE.lock().unwrap();
        guard
            .get_or_insert_with(HashMap::new)
            .insert(key, token.clone());
    }
    Ok(token)
}

/// True when the error chain carries a reqwest 401/403 — i.e. the CAS service
/// rejected the bearer token (expired or revoked), not a network/parse
/// failure.
fn is_auth_error(e: &anyhow::Error) -> bool {
    use reqwest::StatusCode;
    e.chain().any(|c| {
        c.downcast_ref::<reqwest::Error>().is_some_and(|r| {
            matches!(
                r.status(),
                Some(StatusCode::UNAUTHORIZED) | Some(StatusCode::FORBIDDEN)
            )
        })
    })
}

/// Runs `op(cas_token)` with the cached CAS token; on a 401/403 from the CAS
/// service, invalidates the token cache, mints a fresh token, and retries
/// exactly once. Short-lived CAS tokens outlive a typical run, but a long-
/// running process (a large multi-source render) can cross an expiry boundary;
/// without this, the stale token stays cached and every later call fails with
/// an opaque 401 for the rest of the run.
async fn reconstruction_response_reminted(
    ctx: &CasContext,
    xet_hash_hex: &str,
) -> anyhow::Result<ReconstructionResponse> {
    let cas = fetch_cas_token_at(ctx).await?;
    match fetch_reconstruction_response(&cas, xet_hash_hex).await {
        Ok(resp) => Ok(resp),
        Err(e) if is_auth_error(&e) => {
            log::warn!(
                "CAS rejected the cached token (auth error); re-minting once for {}/{}/{}",
                ctx.api_segment,
                ctx.repo_id,
                ctx.revision
            );
            invalidate_cas_token_cache(&ctx.api_segment, &ctx.repo_id, &ctx.revision);
            let fresh = fetch_cas_token_at(ctx).await?;
            fetch_reconstruction_response(&fresh, xet_hash_hex).await
        }
        Err(e) => Err(e),
    }
}

/// Flattens the V2 reconstruction for `xet_hash_hex` into [`XetTerm`]s,
/// re-minting the CAS token once on a 401/403 (see
/// [`reconstruction_response_reminted`]). Used by the initial xet-terms path.
pub(super) async fn reconstruction_terms_at(
    ctx: &CasContext,
    xet_hash_hex: &str,
) -> anyhow::Result<Vec<XetTerm>> {
    let parsed = reconstruction_response_reminted(ctx, xet_hash_hex).await?;
    Ok(flatten_terms(parsed))
}

/// Fetches the full V2 reconstruction for `xet_hash_hex` (terms + signed-URL
/// descriptors), re-minting the CAS token once on a 401/403. Used by
/// `XetReader::new`.
pub(super) async fn reconstruction_response_at(
    ctx: &CasContext,
    xet_hash_hex: &str,
) -> anyhow::Result<ReconstructionResponse> {
    reconstruction_response_reminted(ctx, xet_hash_hex).await
}

/// Production wrapper: mints a CAS token via [`cas_context`] (Hub endpoint +
/// user's HF token). Kept for the URL-refresh path in `mod.rs`, which
/// invalidates the cache itself before calling.
pub(super) async fn fetch_cas_token(
    api_segment: &str,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<CasToken> {
    let ctx = cas_context(api_segment, repo_id, revision)?;
    fetch_cas_token_at(&ctx).await
}

/// GETs the V2 reconstruction for `xet_hash_hex` from `cas`'s CAS service,
/// authenticated with the CAS token's bearer token.
pub(super) async fn fetch_reconstruction_response(
    cas: &CasToken,
    xet_hash_hex: &str,
) -> anyhow::Result<ReconstructionResponse> {
    let url = format!("{}/v2/reconstructions/{}", cas.cas_url, xet_hash_hex);
    authed_get_json(
        &format!("reconstruction {xet_hash_hex}"),
        "reconstruction",
        &url,
        &cas.access_token,
    )
    .await
}

/// Fetches the reconstruction for `xet_hash_hex` and flattens it into
/// `XetTerm`s: one per non-empty term, with `file_offset` accumulated from
/// the `unpacked_length`s. Zero-length terms are skipped and contribute no
/// offset.
/// Flattens a V2 reconstruction into `XetTerm`s: one per non-empty term, with
/// `file_offset` accumulated from the `unpacked_length`s. Zero-length terms
/// are skipped and contribute no offset.
fn flatten_terms(parsed: ReconstructionResponse) -> Vec<XetTerm> {
    let mut offset: u64 = 0;
    let mut out = Vec::with_capacity(parsed.terms.len());
    for t in parsed.terms {
        if t.unpacked_length == 0 {
            continue;
        }
        out.push(XetTerm {
            file_offset: offset,
            byte_len: t.unpacked_length,
            xorb_hash: t.hash,
        });
        offset += t.unpacked_length;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn parses_v2_reconstruction_response_wire_fields() {
        // Field renames (`unpacked_length`, `accessToken`/`casUrl`) and the
        // inclusive-end `bytes` range are what the reader path depends on.
        let json = r#"{
            "terms": [
                {"hash": "abc/123", "unpacked_length": 64, "range": {"start": 0, "end": 2}},
                {"hash": "def/456", "unpacked_length": 10, "range": {"start": 2, "end": 5}}
            ],
            "xorbs": {
                "abc/123": [
                    {"url": "https://cas.example/chunk", "ranges": [
                        {"chunks": {"start": 0, "end": 2}, "bytes": {"start": 0, "end": 63}}
                    ]}
                ]
            }
        }"#;
        let resp: ReconstructionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.terms.len(), 2);
        assert_eq!(resp.terms[0].hash, "abc/123");
        assert_eq!(resp.terms[0].unpacked_length, 64);
        assert_eq!(resp.terms[0].range.start, 0);
        assert_eq!(resp.terms[0].range.end, 2);
        assert_eq!(resp.terms[1].range.end, 5);
        let fetches = resp.xorbs.get("abc/123").unwrap();
        assert_eq!(fetches.len(), 1);
        assert_eq!(fetches[0].url, "https://cas.example/chunk");
        let d = &fetches[0].ranges[0];
        assert_eq!((d.chunks.start, d.chunks.end), (0, 2));
        assert_eq!((d.bytes.start, d.bytes.end), (0, 63));
    }

    #[test]
    fn v1_response_without_xorbs_field_defaults_to_empty() {
        // V1 responses omit `xorbs`; `#[serde(default)]` must turn that into
        // an empty map rather than a parse error.
        let json =
            r#"{"terms": [{"hash": "h", "unpacked_length": 1, "range": {"start": 0, "end": 0}}]}"#;
        let resp: ReconstructionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.terms.len(), 1);
        assert!(resp.xorbs.is_empty());
    }

    #[test]
    fn wrong_field_names_rejected() {
        // Guards the serde renames: a response using camelCase or an unrenamed
        // length field must fail to parse, not silently produce zero values.
        for bad in [
            r#"{"terms":[{"hash":"h","unpackedLength":1,"range":{"start":0,"end":0}}]}"#,
            r#"{"terms":[{"hash":"h","unpacked_length":1,"range":{"start":0}}]}"#,
        ] {
            assert!(
                serde_json::from_str::<ReconstructionResponse>(bad).is_err(),
                "expected parse failure for {bad}"
            );
        }
        // Missing terms field entirely is also an error (no default on terms).
        assert!(serde_json::from_str::<ReconstructionResponse>(r#"{}"#).is_err());
    }

    #[test]
    fn invalidate_cas_token_cache_ignores_unknown_key_and_uninitialized_cache() {
        // Fresh process: cache never populated — invalidation must not panic
        // (no key inserted, no error surfaced).
        invalidate_cas_token_cache("api", "owner/repo", "main");
        // Unknown key on an initialized cache is a no-op removal.
        invalidate_cas_token_cache("api", "owner/other", "dev");
        invalidate_cas_token_cache("api", "owner/repo", "main");
    }

    // --- stale-CAS-token remint tests ----------------------------------------
    // A cached CAS bearer token can expire mid-run (they are short-lived); the
    // initial reconstruction request then 401s and, without a re-mint, every
    // later call in the process fails with the same opaque auth error. The
    // stub server below serves the two endpoints: `/xet-read-token/...`
    // (tokens tok1, tok2, ...) and `/v2/reconstructions/...` (401 first or
    // always, then the reconstruction JSON).

    struct StubCas {
        addr: std::net::SocketAddr,
        token_hits: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        recon_hits: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        recon_bearers: std::sync::Arc<Mutex<Vec<String>>>,
    }

    async fn spawn_cas_stub(recon_always_401: bool) -> StubCas {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let token_hits = Arc::new(AtomicUsize::new(0));
        let recon_hits = Arc::new(AtomicUsize::new(0));
        let recon_bearers = Arc::new(Mutex::new(Vec::new()));
        let th = token_hits.clone();
        let rh = recon_hits.clone();
        let rb = recon_bearers.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                // One request per connection (we answer Connection: close).
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    match sock.read(&mut chunk).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            buf.extend_from_slice(&chunk[..n]);
                            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                    }
                }
                let head = String::from_utf8_lossy(&buf).to_lowercase();
                let bearer = head
                    .lines()
                    .find(|l| l.starts_with("authorization:"))
                    .and_then(|l| l.split_whitespace().last())
                    .unwrap_or("")
                    .to_string();
                let path = head.lines().next().unwrap_or("");
                let resp = if path.contains("/xet-read-token/") {
                    let n = th.fetch_add(1, Ordering::Relaxed) + 1;
                    let body = format!(r#"{{"accessToken":"tok{n}","casUrl":"http://{addr}"}}"#);
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                } else if path.contains("/v2/reconstructions/") {
                    rb.lock().unwrap().push(bearer);
                    let hits = rh.fetch_add(1, Ordering::Relaxed) + 1;
                    if recon_always_401 || hits == 1 {
                        "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            .to_string()
                    } else {
                        let body = r#"{"terms":[{"hash":"x/1","unpacked_length":8,"range":{"start":0,"end":1}}],"xorbs":{}}"#;
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                    }
                } else {
                    "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        .to_string()
                };
                let _ = sock.write_all(resp.as_bytes()).await;
            }
        });
        StubCas {
            addr,
            token_hits,
            recon_hits,
            recon_bearers,
        }
    }

    fn stub_ctx(addr: std::net::SocketAddr, repo_id: &str) -> CasContext {
        CasContext {
            endpoint: format!("http://{addr}"),
            api_segment: "models".to_string(),
            repo_id: repo_id.to_string(),
            revision: "main".to_string(),
            bearer: "hub-token".to_string(),
        }
    }

    #[tokio::test]
    async fn stale_cached_cas_token_is_reminted_once_on_401() {
        invalidate_cas_token_cache("models", "stub/retry-ok", "main");
        let cas = spawn_cas_stub(false).await;
        let ctx = stub_ctx(cas.addr, "stub/retry-ok");
        let terms = reconstruction_terms_at(&ctx, "abc123").await.expect(
            "a 401 on the first reconstruction must trigger exactly one token re-mint and succeed",
        );
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].xorb_hash, "x/1");
        assert_eq!(terms[0].file_offset, 0);
        assert_eq!(terms[0].byte_len, 8);
        // The token endpoint was hit twice (original + re-mint) and the retry
        // carried the freshly minted token, not the rejected one.
        assert_eq!(cas.token_hits.load(Ordering::Relaxed), 2);
        assert_eq!(cas.recon_hits.load(Ordering::Relaxed), 2);
        assert_eq!(
            cas.recon_bearers.lock().unwrap().as_slice(),
            ["tok1", "tok2"]
        );
    }

    #[tokio::test]
    async fn persistent_auth_error_surfaces_after_one_remint() {
        invalidate_cas_token_cache("models", "stub/retry-fail", "main");
        let cas = spawn_cas_stub(true).await;
        let ctx = stub_ctx(cas.addr, "stub/retry-fail");
        let err = reconstruction_terms_at(&ctx, "abc123")
            .await
            .expect_err("a persistently rejecting CAS must surface an error, not loop");
        assert!(
            err.chain()
                .any(|c| c.downcast_ref::<reqwest::Error>().is_some()),
            "expected the reqwest auth error to surface, got: {err:#}"
        );
        // Bounded: exactly one re-mint, then give up.
        assert_eq!(cas.token_hits.load(Ordering::Relaxed), 2);
        assert_eq!(cas.recon_hits.load(Ordering::Relaxed), 2);
    }
}
