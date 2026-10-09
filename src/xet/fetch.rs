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

#[derive(Deserialize)]
pub(super) struct XetReadTokenResponse {
    #[serde(rename = "accessToken")]
    pub(super) access_token: String,
    #[serde(rename = "casUrl")]
    pub(super) cas_url: String,
}

/// JSON wire form of `ChunkRange`/`HttpRange`/`FileRange` from xet-client's
/// `cas_types::Range<Idx, Kind>` — the `_marker` field is `#[serde(skip)]`
/// so only `start`/`end` go on the wire.
#[derive(Deserialize, Clone, Copy)]
pub(super) struct WireRange<T> {
    pub(super) start: T,
    pub(super) end: T,
}

#[derive(Deserialize)]
pub(super) struct ReconstructionTerm {
    pub(super) hash: String,
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
    pub(super) chunks: WireRange<u32>,
    pub(super) bytes: WireRange<u64>,
}

#[derive(Deserialize)]
pub(super) struct WireXorbMultiRangeFetch {
    pub(super) url: String,
    pub(super) ranges: Vec<WireXorbRangeDescriptor>,
}

#[derive(Deserialize)]
pub(super) struct ReconstructionResponse {
    pub(super) terms: Vec<ReconstructionTerm>,
    /// V2-only: per-xorb signed-URL fetch info. Absent on V1 responses; we
    /// require V2 for the direct-CAS reader path, so callers that need it
    /// should error if this field is missing.
    #[serde(default)]
    pub(super) xorbs: HashMap<String, Vec<WireXorbMultiRangeFetch>>,
}

#[derive(Clone)]
pub(super) struct CasToken {
    pub(super) cas_url: String,
    pub(super) access_token: String,
}

/// Per-process cache of CAS tokens, keyed by `(api_segment, repo_id, revision)`.
/// Tokens expire (the response includes an `exp` field) but for arbvis runs
/// they live well within the expiration window of a single visualization.
static CAS_TOKEN_CACHE: Mutex<Option<HashMap<(String, String, String), CasToken>>> =
    Mutex::new(None);

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

pub(super) async fn fetch_cas_token(
    api_segment: &str,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<CasToken> {
    let key = (
        api_segment.to_string(),
        repo_id.to_string(),
        revision.to_string(),
    );
    {
        let mut guard = CAS_TOKEN_CACHE.lock().unwrap();
        let cache = guard.get_or_insert_with(HashMap::new);
        if let Some(t) = cache.get(&key) {
            return Ok(t.clone());
        }
    }

    let hf_token = hf_url::read_token().ok_or_else(|| {
        anyhow!("HF token required for xet reconstruction; set HF_TOKEN or run `hf auth login`")
    })?;

    let url = format!(
        "{}/api/{}/{}/xet-read-token/{}",
        hf_url::endpoint(),
        api_segment,
        repo_id,
        revision,
    );
    // `error_for_status()` (inside `authed_get_json`) converts non-2xx into a
    // reqwest::Error carrying the status code so the throttle's classifier can
    // detect 429/5xx and retry. Response body detail is lost on error, but the
    // URL and status code are preserved.
    let parsed: XetReadTokenResponse = authed_get_json(
        &format!("xet-read-token {repo_id}"),
        "xet-read-token",
        &url,
        &hf_token,
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

pub(super) async fn fetch_reconstruction_terms(
    cas: &CasToken,
    xet_hash_hex: &str,
) -> anyhow::Result<Vec<XetTerm>> {
    let parsed = fetch_reconstruction_response(cas, xet_hash_hex).await?;

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
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let json = r#"{"terms": [{"hash": "h", "unpacked_length": 1, "range": {"start": 0, "end": 0}}]}"#;
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
}
