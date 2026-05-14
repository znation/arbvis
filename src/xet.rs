//! Xet reconstruction-term fetcher and color tables.
//!
//! For xet-backed Hub files, this module fetches the per-file reconstruction
//! terms (which xorb each contiguous byte run came from, plus chunk index
//! ranges within that xorb) by talking directly to two documented Hub
//! endpoints:
//!
//!   1. `GET {endpoint}/api/{models|datasets|spaces}/{repo}/xet-read-token/{rev}`
//!      with the user's HF token → returns `{access_token, exp, cas_url}`.
//!   2. `GET {cas_url}/v2/reconstructions/{xet_hash_hex}` with the CAS token
//!      → returns `{offset_into_first_range, terms: [...], xorbs: {...}}`.
//!
//! hf-hub 1.0.0-rc.1 has these as `pub(crate)` internals only; the
//! reconstruction call is not exposed through `xet-client` either in a
//! blocking form. We avoid pulling the full async stack in by talking to the
//! two endpoints directly with `reqwest::blocking`.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context, anyhow};
use serde::Deserialize;

use crate::hf_url::{self, RemoteFileSpec};
use crate::throttle::with_throttle;

/// A flat byte-range view of one reconstruction term.
///
/// `terms` from the V2 reconstruction response are sequential and contiguous;
/// `file_offset` is the cumulative sum of preceding `byte_len`s, and
/// `xorb_hash` is the hex Merkle hash of the xorb the bytes came from.
#[derive(Clone, Debug)]
pub struct XetTerm {
    pub file_offset: u64,
    pub byte_len: u64,
    pub xorb_hash: String,
}

#[derive(Deserialize)]
struct XetReadTokenResponse {
    #[serde(rename = "accessToken")]
    access_token: String,
    #[serde(rename = "casUrl")]
    cas_url: String,
}

#[derive(Deserialize)]
struct ReconstructionTerm {
    hash: String,
    #[serde(rename = "unpacked_length")]
    unpacked_length: u64,
    // `range` (chunk index start/end within the xorb) is not used by the
    // visualization — chunk boundaries are derived from term boundaries.
}

#[derive(Deserialize)]
struct ReconstructionResponse {
    terms: Vec<ReconstructionTerm>,
}

#[derive(Clone)]
struct CasToken {
    cas_url: String,
    access_token: String,
}

/// Per-process cache of CAS tokens, keyed by `(api_segment, repo_id, revision)`.
/// Tokens expire (the response includes an `exp` field) but for arbvis runs
/// they live well within the expiration window of a single visualization.
static CAS_TOKEN_CACHE: Mutex<Option<HashMap<(String, String, String), CasToken>>> =
    Mutex::new(None);

fn fetch_cas_token(
    api_segment: &str,
    repo_id: &str,
    revision: &str,
) -> anyhow::Result<CasToken> {
    let key = (api_segment.to_string(), repo_id.to_string(), revision.to_string());
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
    log::info!("Requesting xet CAS token for {repo_id}@{revision}");
    let client = reqwest::blocking::Client::new();
    // `error_for_status()` converts non-2xx into a reqwest::Error carrying the
    // status code so the throttle's classifier can detect 429/5xx and retry.
    // Response body detail is lost on error, but the URL and status code are
    // preserved.
    let resp = with_throttle(&format!("xet-read-token {repo_id}"), || {
        client
            .get(&url)
            .bearer_auth(&hf_token)
            .send()
            .and_then(|r| r.error_for_status())
    })
    .with_context(|| format!("requesting xet-read-token at {url}"))?;
    let parsed: XetReadTokenResponse = resp
        .json()
        .with_context(|| format!("parsing xet-read-token response from {url}"))?;

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

fn fetch_reconstruction_terms(
    cas: &CasToken,
    xet_hash_hex: &str,
) -> anyhow::Result<Vec<XetTerm>> {
    let url = format!("{}/v2/reconstructions/{}", cas.cas_url, xet_hash_hex);
    log::info!("Fetching reconstruction terms: {url}");
    let client = reqwest::blocking::Client::new();
    let resp = with_throttle(&format!("reconstruction {xet_hash_hex}"), || {
        client
            .get(&url)
            .bearer_auth(&cas.access_token)
            .send()
            .and_then(|r| r.error_for_status())
    })
    .with_context(|| format!("requesting reconstruction at {url}"))?;
    let parsed: ReconstructionResponse = resp
        .json()
        .with_context(|| format!("parsing reconstruction response from {url}"))?;

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

/// Fetch reconstruction terms for a remote xet-backed file.
///
/// Returns `Ok(vec![])` if the file has no xet hash (e.g. plain LFS or
/// regular file). Errors only on real failures (network, auth, malformed
/// responses).
pub fn reconstruction_for(spec: &RemoteFileSpec) -> anyhow::Result<Vec<XetTerm>> {
    let Some(hash) = spec.xet_hash.as_deref() else {
        log::warn!(
            "{}: not xet-backed, skipping xet visualization for this source",
            spec.filename
        );
        return Ok(Vec::new());
    };
    let cas = fetch_cas_token(spec.repo.api_segment(), &spec.repo.repo_id(), &spec.revision)?;
    fetch_reconstruction_terms(&cas, hash)
}

/// Tableau-20 palette, https://vega.github.io/vega/docs/schemes/#tableau20
pub const TABLEAU_20: [[u8; 3]; 20] = [
    [0x4c, 0x78, 0xa8], // blue
    [0xf5, 0x8a, 0x3b], // orange
    [0x54, 0xa2, 0x4b], // green
    [0xee, 0x65, 0x5a], // red
    [0x72, 0xb7, 0xb2], // teal
    [0xee, 0xa6, 0xcb], // pink
    [0xa1, 0x71, 0x9c], // purple
    [0x9a, 0x70, 0x40], // brown
    [0xfb, 0xbf, 0x45], // yellow
    [0xba, 0xb0, 0xac], // grey
    [0x9e, 0xc9, 0xe2], // light blue
    [0xff, 0xbe, 0x7d], // light orange
    [0x88, 0xd2, 0x7a], // light green
    [0xff, 0x9d, 0x9a], // light red
    [0x8c, 0xd1, 0x7d], // light teal
    [0xfc, 0xcd, 0xe5], // light pink
    [0xb2, 0x79, 0xa2], // light purple
    [0xd1, 0xb9, 0x9b], // light brown
    [0xfb, 0xe2, 0x9a], // light yellow
    [0xcf, 0xcf, 0xcf], // light grey
];

/// Global xorb→color assignment, built from all sources' xet_terms in one pass.
///
/// `global_ranges` is the per-byte-range color table after shifting each
/// source's local file offsets by its cumulative offset in the concatenated
/// stream. Sorted by `start`, no overlaps.
pub struct XorbMap {
    pub global_ranges: Vec<(u64, u64, u8)>,
}

impl XorbMap {
    /// Build from per-source `(xet_terms, cumulative_offset)` pairs. Sources
    /// whose `xet_terms` is `None` or empty contribute nothing.
    pub fn build<'a, I>(per_source: I) -> Self
    where
        I: IntoIterator<Item = (Option<&'a [XetTerm]>, u64)>,
    {
        let mut by_hash: HashMap<String, u8> = HashMap::new();
        let mut next: usize = 0;
        let mut global_ranges = Vec::new();
        for (terms_opt, source_offset) in per_source {
            let Some(terms) = terms_opt else { continue };
            for t in terms {
                if t.byte_len == 0 {
                    continue;
                }
                let idx = if let Some(&i) = by_hash.get(&t.xorb_hash) {
                    i
                } else {
                    let i = (next % 20) as u8;
                    by_hash.insert(t.xorb_hash.clone(), i);
                    next += 1;
                    i
                };
                let start = source_offset + t.file_offset;
                let end = start + t.byte_len;
                global_ranges.push((start, end, idx));
            }
        }
        global_ranges.sort_by_key(|&(s, _, _)| s);
        XorbMap { global_ranges }
    }

    pub fn is_empty(&self) -> bool {
        self.global_ranges.is_empty()
    }

    /// Color index for a given absolute byte offset, or `None` if no term
    /// covers it. Binary search on `global_ranges`.
    pub fn color_idx_at(&self, pixel_idx: u64) -> Option<u8> {
        let ranges = &self.global_ranges;
        if ranges.is_empty() {
            return None;
        }
        let mut lo = 0usize;
        let mut hi = ranges.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let (s, e, c) = ranges[mid];
            if pixel_idx < s {
                hi = mid;
            } else if pixel_idx >= e {
                lo = mid + 1;
            } else {
                return Some(c);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(offset: u64, len: u64, hash: &str) -> XetTerm {
        XetTerm { file_offset: offset, byte_len: len, xorb_hash: hash.to_string() }
    }

    #[test]
    fn xorbmap_recycles_after_20_unique() {
        let mut terms = Vec::new();
        let mut off = 0u64;
        for i in 0..25 {
            terms.push(t(off, 1, &format!("xorb-{i}")));
            off += 1;
        }
        let m = XorbMap::build(std::iter::once((Some(&terms[..]), 0)));
        // 21st distinct xorb → wraps to color 0.
        assert_eq!(m.color_idx_at(20), Some(0));
        assert_eq!(m.color_idx_at(0), Some(0));
        // distinct first 20.
        assert_eq!(m.color_idx_at(19), Some(19));
    }

    #[test]
    fn xorbmap_shifts_by_source_offset() {
        let a = vec![t(0, 10, "A")];
        let b = vec![t(0, 5, "B")];
        let m = XorbMap::build(vec![
            (Some(&a[..]), 0),
            (Some(&b[..]), 10),
        ]);
        assert_eq!(m.color_idx_at(0), Some(0));
        assert_eq!(m.color_idx_at(9), Some(0));
        assert_eq!(m.color_idx_at(10), Some(1));
        assert_eq!(m.color_idx_at(14), Some(1));
        assert_eq!(m.color_idx_at(15), None);
    }

    #[test]
    fn xorbmap_shared_xorb_across_sources_gets_same_color() {
        let a = vec![t(0, 10, "shared")];
        let b = vec![t(0, 5, "shared")];
        let m = XorbMap::build(vec![
            (Some(&a[..]), 0),
            (Some(&b[..]), 10),
        ]);
        assert_eq!(m.color_idx_at(0), m.color_idx_at(10));
    }
}
