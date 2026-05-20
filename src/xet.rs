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
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, anyhow};
use lru::LruCache;
use serde::Deserialize;

use crate::hf_url::{self, RemoteFileSpec};
use crate::throttle::with_throttle;

/// Shared HTTP client used by every xet endpoint (token, reconstruction,
/// direct-CAS range fetches).
///
/// Two reasons to share one client instead of `reqwest::Client::new()` per call
/// site:
///
/// 1. **Connection pool reuse.** Each `Client` has its own pool. A diff across
///    `n` files spawns `n` `XetReader`s; one client per reader meant `n`
///    separate pools and no HTTP/2 multiplexing across readers. With one
///    shared client all xorb fetches against the same CAS host share live
///    connections.
/// 2. **Stuck-stream detection without penalising slow downloads.** A
///    fresh-`Client::new()` has no timeouts, so a hung HTTP/2 stream can hold
///    an AIMD throttle permit indefinitely. But a *total* request timeout
///    (`timeout(_)`) would kill a legitimately slow 60 MB descriptor download
///    just because the link is saturated by sibling fetches. `read_timeout`
///    is the right primitive: it fires only when no bytes have been received
///    for the configured window, so an actively-progressing download is
///    never cut off, while a connection that's silently stalled is killed
///    and the throttle retries.
pub fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            // Per-read timeout: only fires when the server stops sending bytes
            // for this long. A slow-but-progressing multi-MB descriptor
            // download is unaffected; a wedged HTTP/2 stream is killed in
            // well under the "minute-long stall" we used to see.
            .read_timeout(Duration::from_secs(30))
            // Tighter budget for opening the TCP+TLS connection. A failed
            // handshake should fail fast so the retry loop can pick a new
            // upstream / decorrelated-jitter window.
            .connect_timeout(Duration::from_secs(10))
            // Use rustls (default features) and let reqwest negotiate HTTP/2.
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .expect("reqwest client build")
    })
}

/// In-flight direct-CAS HTTP requests across all `XetReader`s.
///
/// These bypass the AIMD throttle (xet.rs `load_descriptor`), so the AIMD
/// counters miss them. The perf monitor reads this directly to see whether a
/// stall is really a stall or just CAS requests still in flight.
static CAS_INFLIGHT: AtomicUsize = AtomicUsize::new(0);
/// Cumulative completed CAS HTTP requests (success or error).
static CAS_COMPLETED: AtomicU64 = AtomicU64::new(0);
/// Cumulative bytes received from CAS HTTP responses.
static CAS_BYTES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug)]
pub struct CasStats {
    pub in_flight: usize,
    pub completed: u64,
    pub bytes: u64,
}

pub fn cas_stats() -> CasStats {
    CasStats {
        in_flight: CAS_INFLIGHT.load(Ordering::Relaxed),
        completed: CAS_COMPLETED.load(Ordering::Relaxed),
        bytes: CAS_BYTES.load(Ordering::Relaxed),
    }
}

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

/// JSON wire form of `ChunkRange`/`HttpRange`/`FileRange` from xet-client's
/// `cas_types::Range<Idx, Kind>` — the `_marker` field is `#[serde(skip)]`
/// so only `start`/`end` go on the wire.
#[derive(Deserialize, Clone, Copy)]
struct WireRange<T> {
    start: T,
    end: T,
}

#[derive(Deserialize)]
struct ReconstructionTerm {
    hash: String,
    #[serde(rename = "unpacked_length")]
    unpacked_length: u64,
    /// Chunk index `[start, end)` within the xorb. Captured so the reader can
    /// map a file-byte range to the exact chunks it spans (vs. approximating
    /// from term boundaries alone).
    range: WireRange<u32>,
}

/// Per-xorb byte range fetch instructions: one signed URL covers some chunks,
/// described by `chunks` (chunk index range) and `bytes` (packed-byte range
/// inside the xorb, *inclusive end* — `HttpRange` semantics).
#[derive(Deserialize)]
struct WireXorbRangeDescriptor {
    chunks: WireRange<u32>,
    bytes: WireRange<u64>,
}

#[derive(Deserialize)]
struct WireXorbMultiRangeFetch {
    url: String,
    ranges: Vec<WireXorbRangeDescriptor>,
}

#[derive(Deserialize)]
struct ReconstructionResponse {
    terms: Vec<ReconstructionTerm>,
    /// V2-only: per-xorb signed-URL fetch info. Absent on V1 responses; we
    /// require V2 for the direct-CAS reader path, so callers that need it
    /// should error if this field is missing.
    #[serde(default)]
    xorbs: HashMap<String, Vec<WireXorbMultiRangeFetch>>,
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

async fn fetch_cas_token(
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
    let client = http_client();
    // `error_for_status()` converts non-2xx into a reqwest::Error carrying the
    // status code so the throttle's classifier can detect 429/5xx and retry.
    // Response body detail is lost on error, but the URL and status code are
    // preserved.
    let resp = with_throttle(&format!("xet-read-token {repo_id}"), || async {
        client
            .get(&url)
            .bearer_auth(&hf_token)
            .send()
            .await
            .and_then(|r| r.error_for_status())
    })
    .await
    .with_context(|| format!("requesting xet-read-token at {url}"))?;
    let parsed: XetReadTokenResponse = resp
        .json()
        .await
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

async fn fetch_reconstruction_response(
    cas: &CasToken,
    xet_hash_hex: &str,
) -> anyhow::Result<ReconstructionResponse> {
    let url = format!("{}/v2/reconstructions/{}", cas.cas_url, xet_hash_hex);
    log::info!("Fetching reconstruction terms: {url}");
    let client = http_client();
    let resp = with_throttle(&format!("reconstruction {xet_hash_hex}"), || async {
        client
            .get(&url)
            .bearer_auth(&cas.access_token)
            .send()
            .await
            .and_then(|r| r.error_for_status())
    })
    .await
    .with_context(|| format!("requesting reconstruction at {url}"))?;
    resp.json::<ReconstructionResponse>()
        .await
        .with_context(|| format!("parsing reconstruction response from {url}"))
}

async fn fetch_reconstruction_terms(
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

/// Fetch reconstruction terms for a remote xet-backed file.
///
/// Returns `Ok(vec![])` if the file has no xet hash (e.g. plain LFS or
/// regular file). Errors only on real failures (network, auth, malformed
/// responses).
pub async fn reconstruction_for(spec: &RemoteFileSpec) -> anyhow::Result<Vec<XetTerm>> {
    let Some(hash) = spec.xet_hash.as_deref() else {
        log::warn!(
            "{}: not xet-backed, skipping xet visualization for this source",
            spec.filename
        );
        return Ok(Vec::new());
    };
    let cas = fetch_cas_token(spec.repo.api_segment(), &spec.repo.repo_id(), &spec.revision).await?;
    fetch_reconstruction_terms(&cas, hash).await
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

// ─── XetReader: direct-CAS byte fetcher ──────────────────────────────────────
//
// hf-hub's `xet_download_stream` rebuilds a `XetDownloadStreamGroup` per call,
// which dominates wall time for short range requests (see the comment on
// `data::materialize_http_sources`). When we know the file is xet-backed we
// can do the streaming ourselves: fetch the V2 reconstruction once, then for
// each byte range translate into (xorb, chunk-range) lookups and HTTP GET the
// pre-signed xorb URLs directly. Decoded chunk segments are cached so adjacent
// tiles (which the Hilbert curve makes spatially coherent) reuse the same
// decompression work.

/// One term from the V2 reconstruction response, in the form we keep at
/// runtime: cumulative file offset and exact chunk range within the xorb.
struct ReaderTerm {
    file_offset: u64,
    byte_len: u64,
    xorb_hash: String,
    chunk_start: u32,
    chunk_end: u32,
}

/// One signed-URL descriptor: a contiguous chunk range plus its packed-byte
/// range inside the xorb (HTTP Range header is `bytes=byte_start-byte_end`,
/// *inclusive end*).
#[derive(Clone)]
struct ReaderDescriptor {
    chunk_start: u32,
    chunk_end: u32,
    byte_start: u64,
    byte_end: u64,
    url: Arc<String>,
}

/// All descriptors for one xorb, sorted by `chunk_start` for binary search.
struct XorbInfo {
    descriptors: Vec<ReaderDescriptor>,
}

/// Decoded descriptor payload kept in the LRU. `chunk_byte_indices[i]` is the
/// start of chunk `chunk_start + i` within `data`, with a trailing entry equal
/// to `data.len()` — semantics of `xet_core_structures::xorb_object::deserialize_chunks`.
#[derive(Clone)]
struct DecodedDescriptor {
    data: Arc<Vec<u8>>,
    chunk_byte_indices: Arc<Vec<u32>>,
}

/// Cache key: `(xorb_hash, descriptor.byte_start)` uniquely identifies a
/// packed byte range within a xorb (sharing across files isn't exploited —
/// each `XetReader` has its own cache).
type CacheKey = (String, u64);

/// Byte-bounded LRU. `lru::LruCache` is capacity-bounded by entry *count*; we
/// pair it with a running byte total and evict until under budget.
struct DescriptorCache {
    lru: LruCache<CacheKey, DecodedDescriptor>,
    bytes: u64,
    budget: u64,
}

impl DescriptorCache {
    fn new(budget: u64) -> Self {
        // Capacity here is a hard upper bound on entry count; pick something
        // generous (1M entries) so the byte budget is the real limit.
        Self {
            lru: LruCache::new(NonZeroUsize::new(1_000_000).unwrap()),
            bytes: 0,
            budget,
        }
    }

    fn get(&mut self, key: &CacheKey) -> Option<DecodedDescriptor> {
        self.lru.get(key).map(|v| DecodedDescriptor {
            data: Arc::clone(&v.data),
            chunk_byte_indices: Arc::clone(&v.chunk_byte_indices),
        })
    }

    fn put(&mut self, key: CacheKey, value: DecodedDescriptor) {
        let added = value.data.len() as u64;
        if let Some(old) = self.lru.push(key, value) {
            self.bytes = self.bytes.saturating_sub(old.1.data.len() as u64);
        }
        self.bytes = self.bytes.saturating_add(added);
        while self.bytes > self.budget {
            match self.lru.pop_lru() {
                Some((_, v)) => self.bytes = self.bytes.saturating_sub(v.data.len() as u64),
                None => break,
            }
        }
    }
}

/// Direct-CAS byte fetcher for one xet-backed remote file.
///
/// Holds the V2 reconstruction (terms + signed-URL fetch info). The HTTP
/// client is process-shared via [`http_client`] so all readers reuse one
/// connection pool. `inflight` deduplicates concurrent fetches of the same
/// descriptor — RMS sampling and tile rendering both kick off many parallel
/// `fetch_range` calls whose ranges land in the same 50-65 MB descriptor, so
/// without dedup the same descriptor gets downloaded N times in parallel and
/// N-1 copies are discarded.
pub struct XetReader {
    terms: Vec<ReaderTerm>,          // sorted by file_offset
    xorbs: HashMap<String, XorbInfo>,
    file_size: u64,
    filename: Arc<String>,
    cache: Mutex<DescriptorCache>,
    inflight: Mutex<HashMap<CacheKey, Arc<tokio::sync::OnceCell<DecodedDescriptor>>>>,
}

/// Default per-reader cache budget — generous enough to keep hot regions of
/// a multi-GB safetensors warm across a tile-render sweep, but bounded so
/// many concurrent readers don't blow up RAM.
const DEFAULT_CACHE_BUDGET_BYTES: u64 = 512 * 1024 * 1024;

impl XetReader {
    /// Build a reader for a xet-backed remote file. Errors if the file has
    /// no xet hash (i.e. plain LFS / regular Hub file).
    pub async fn new(spec: &RemoteFileSpec) -> anyhow::Result<Arc<Self>> {
        let hash = spec
            .xet_hash
            .as_deref()
            .ok_or_else(|| anyhow!("{}: not xet-backed, cannot build XetReader", spec.filename))?;
        let cas = fetch_cas_token(spec.repo.api_segment(), &spec.repo.repo_id(), &spec.revision).await?;
        let raw = fetch_reconstruction_response(&cas, hash).await?;

        // Build the per-xorb descriptor lookup (sorted by chunk_start).
        let mut xorbs: HashMap<String, XorbInfo> = HashMap::with_capacity(raw.xorbs.len());
        for (xorb_hash, fetches) in raw.xorbs {
            let mut descriptors: Vec<ReaderDescriptor> = Vec::new();
            for fetch in fetches {
                let url = Arc::new(fetch.url);
                for desc in fetch.ranges {
                    descriptors.push(ReaderDescriptor {
                        chunk_start: desc.chunks.start,
                        chunk_end: desc.chunks.end,
                        byte_start: desc.bytes.start,
                        byte_end: desc.bytes.end,
                        url: Arc::clone(&url),
                    });
                }
            }
            descriptors.sort_by_key(|d| d.chunk_start);
            xorbs.insert(xorb_hash, XorbInfo { descriptors });
        }

        // Compute cumulative file offsets per term; verify each term's xorb
        // appears in `xorbs` (otherwise the fetch path would silently fail).
        let mut terms: Vec<ReaderTerm> = Vec::with_capacity(raw.terms.len());
        let mut offset: u64 = 0;
        for t in raw.terms {
            if t.unpacked_length == 0 {
                continue;
            }
            if !xorbs.contains_key(&t.hash) {
                anyhow::bail!(
                    "reconstruction for {}: term references xorb {} with no fetch info",
                    spec.filename,
                    t.hash
                );
            }
            terms.push(ReaderTerm {
                file_offset: offset,
                byte_len: t.unpacked_length,
                xorb_hash: t.hash,
                chunk_start: t.range.start,
                chunk_end: t.range.end,
            });
            offset += t.unpacked_length;
        }

        if offset != spec.size {
            log::warn!(
                "{}: reconstruction unpacked_length total {} disagrees with file size {}",
                spec.filename, offset, spec.size,
            );
        }

        Ok(Arc::new(Self {
            terms,
            xorbs,
            file_size: spec.size,
            filename: Arc::clone(&spec.filename),
            cache: Mutex::new(DescriptorCache::new(DEFAULT_CACHE_BUDGET_BYTES)),
            inflight: Mutex::new(HashMap::new()),
        }))
    }

    /// File-byte fetch. Walks terms overlapping `[start, start+len)`, then
    /// for each term walks the xorb descriptors overlapping its chunk range,
    /// fetching + decompressing (cached) and concatenating the requested
    /// slice into `out`.
    pub async fn fetch_range(&self, start: u64, len: usize) -> anyhow::Result<Vec<u8>> {
        let end = start.saturating_add(len as u64);
        if end > self.file_size {
            anyhow::bail!(
                "{}: range [{},{}) past file size {}",
                self.filename, start, end, self.file_size,
            );
        }
        let mut out = Vec::with_capacity(len);

        // Binary search for the first term overlapping `start`.
        let mut term_idx = self
            .terms
            .partition_point(|t| t.file_offset + t.byte_len <= start);

        while term_idx < self.terms.len() {
            let term = &self.terms[term_idx];
            if term.file_offset >= end {
                break;
            }
            self.append_term_range(term, start, end, &mut out).await?;
            term_idx += 1;
        }

        if out.len() != len {
            anyhow::bail!(
                "{}: fetch_range[{},{}) produced {} bytes (expected {})",
                self.filename, start, end, out.len(), len,
            );
        }
        Ok(out)
    }

    /// Append the slice of `term`'s data that overlaps `[req_start, req_end)`
    /// (in file-byte coordinates) to `out`.
    async fn append_term_range(
        &self,
        term: &ReaderTerm,
        req_start: u64,
        req_end: u64,
        out: &mut Vec<u8>,
    ) -> anyhow::Result<()> {
        let term_end = term.file_offset + term.byte_len;
        let need_start = req_start.max(term.file_offset);
        let need_end = req_end.min(term_end);
        // Bytes we want, in term-local coordinates `[0, term.byte_len)`.
        let mut local_lo = (need_start - term.file_offset) as usize;
        let local_hi = (need_end - term.file_offset) as usize;

        let xorb = self
            .xorbs
            .get(&term.xorb_hash)
            .ok_or_else(|| anyhow!("xorb {} missing from fetch map", term.xorb_hash))?;

        // Walk descriptors overlapping [term.chunk_start, term.chunk_end). The
        // descriptor list is sorted by chunk_start.
        let first_desc = xorb
            .descriptors
            .partition_point(|d| d.chunk_end <= term.chunk_start);

        // Cumulative byte length of the descriptor-prefix already consumed
        // from this term — used to translate `local_lo`/`local_hi` (term-local)
        // into descriptor-local byte offsets.
        let mut consumed_term_bytes: usize = 0;

        for desc in &xorb.descriptors[first_desc..] {
            if desc.chunk_start >= term.chunk_end {
                break;
            }
            // The slice of this descriptor's chunks that belongs to the term:
            // `[desc_lo_chunk, desc_hi_chunk)` is an absolute xorb chunk range.
            let desc_lo_chunk = desc.chunk_start.max(term.chunk_start);
            let desc_hi_chunk = desc.chunk_end.min(term.chunk_end);

            let decoded = self.load_descriptor(&term.xorb_hash, desc).await?;
            let indices = &decoded.chunk_byte_indices;
            let bytes = &decoded.data;

            // Translate to indices into `decoded.chunk_byte_indices` (which is
            // indexed from 0 = descriptor's first chunk).
            let lo_idx = (desc_lo_chunk - desc.chunk_start) as usize;
            let hi_idx = (desc_hi_chunk - desc.chunk_start) as usize;
            if hi_idx >= indices.len() {
                anyhow::bail!(
                    "descriptor {}@{}: chunk index {} out of bounds (have {})",
                    term.xorb_hash, desc.byte_start, hi_idx, indices.len(),
                );
            }
            let desc_byte_lo = indices[lo_idx] as usize;
            let desc_byte_hi = indices[hi_idx] as usize;
            let desc_byte_len = desc_byte_hi - desc_byte_lo;

            // This descriptor contributes `desc_byte_len` bytes to the term at
            // term-local offset `[consumed_term_bytes, consumed_term_bytes + desc_byte_len)`.
            let term_lo = consumed_term_bytes;
            let term_hi = consumed_term_bytes + desc_byte_len;

            if local_hi > term_lo && local_lo < term_hi {
                let copy_lo = local_lo.max(term_lo);
                let copy_hi = local_hi.min(term_hi);
                let inner_lo = desc_byte_lo + (copy_lo - term_lo);
                let inner_hi = desc_byte_lo + (copy_hi - term_lo);
                out.extend_from_slice(&bytes[inner_lo..inner_hi]);
                local_lo = copy_hi; // monotonically advances
                if local_lo >= local_hi {
                    return Ok(());
                }
            }

            consumed_term_bytes = term_hi;
        }
        Ok(())
    }

    /// Fetch (or cache-hit) a descriptor's decoded chunk segment.
    ///
    /// Concurrent calls for the same `(xorb_hash, descriptor.byte_start)` key
    /// share a single HTTP+decode pass via [`tokio::sync::OnceCell`]. Without
    /// this, 16-way `buffer_unordered` callers whose ranges all land in the
    /// same 50-65 MB descriptor would issue 16 parallel downloads and discard
    /// 15 of them — visible in the perf monitor as `cas in_flight=16` sitting
    /// at `0.0 req/s` for tens of seconds.
    async fn load_descriptor(
        &self,
        xorb_hash: &str,
        desc: &ReaderDescriptor,
    ) -> anyhow::Result<DecodedDescriptor> {
        let key = (xorb_hash.to_string(), desc.byte_start);
        if let Some(hit) = self.cache.lock().unwrap().get(&key) {
            return Ok(hit);
        }

        // Get-or-insert the in-flight cell. The first caller will run the
        // fetch closure; concurrent callers wait on the same OnceCell and
        // receive the result by clone.
        let cell = {
            let mut inflight = self.inflight.lock().unwrap();
            Arc::clone(
                inflight
                    .entry(key.clone())
                    .or_insert_with(|| Arc::new(tokio::sync::OnceCell::new())),
            )
        };

        // anyhow::Error isn't Clone, so OnceCell can't store a Result directly
        // — every awaiting caller would need the same error. Convert failures
        // to a String once and let each caller wrap it in an Err with the same
        // text. The first caller does the fetch; on success the value is also
        // installed in the LRU below.
        let decoded = cell
            .get_or_try_init(|| async {
                let res = self.fetch_and_decode_descriptor(xorb_hash, desc).await;
                match res {
                    Ok(d) => {
                        self.cache.lock().unwrap().put(key.clone(), d.clone());
                        Ok(d)
                    }
                    Err(e) => Err(format!("{e:#}")),
                }
            })
            .await
            .map_err(|e| anyhow!("{e}"))?
            .clone();

        // Drop the inflight entry now that the value is in the LRU. Future
        // callers will see the LRU hit. If a caller raced in between, they
        // will already have observed the OnceCell value via `get_or_try_init`,
        // so removal is safe.
        self.inflight.lock().unwrap().remove(&key);
        Ok(decoded)
    }

    /// Do the actual HTTP fetch + decode. Wrapped by `load_descriptor` which
    /// adds the cache + in-flight-dedup layers.
    async fn fetch_and_decode_descriptor(
        &self,
        xorb_hash: &str,
        desc: &ReaderDescriptor,
    ) -> anyhow::Result<DecodedDescriptor> {
        // HTTP Range header is INCLUSIVE on both ends (RFC 7233 §2.1) — match
        // `HttpRange` semantics from xet-client's wire format.
        //
        // Deliberately bypass `with_throttle` here: the tile-load worker that
        // ultimately drives this call already holds a `Throttle::global()`
        // permit. Acquiring a second one inside would deadlock when every
        // permit is held by a worker waiting on its nested fetch (and CAS
        // doesn't rate-limit the way Hub does, so the outer permit is enough
        // concurrency control). Network errors propagate; the load worker
        // surfaces them and aborts the pipeline.
        let range_header = format!("bytes={}-{}", desc.byte_start, desc.byte_end);
        CAS_INFLIGHT.fetch_add(1, Ordering::Relaxed);
        let bytes_res = async {
            let resp = http_client()
                .get(desc.url.as_str())
                .header(reqwest::header::RANGE, &range_header)
                .send()
                .await
                .and_then(|r| r.error_for_status())
                .with_context(|| format!("HTTP error fetching xorb range {} for {}", desc.byte_start, xorb_hash))?;
            resp.bytes()
                .await
                .with_context(|| format!("body read fetching xorb range {} for {}", desc.byte_start, xorb_hash))
        }.await;
        CAS_INFLIGHT.fetch_sub(1, Ordering::Relaxed);
        CAS_COMPLETED.fetch_add(1, Ordering::Relaxed);
        let bytes = bytes_res?;
        CAS_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);

        // Decode packed chunks → uncompressed bytes + per-chunk start offsets.
        let mut cursor = std::io::Cursor::new(bytes.as_ref());
        let (data, chunk_byte_indices) = xet_core_structures::xorb_object::deserialize_chunks(&mut cursor)
            .map_err(|e| anyhow!("decoding xorb {} bytes [{},{}]: {}", xorb_hash, desc.byte_start, desc.byte_end, e))?;

        Ok(DecodedDescriptor {
            data: Arc::new(data),
            chunk_byte_indices: Arc::new(chunk_byte_indices),
        })
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
