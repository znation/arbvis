use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use futures::future::BoxFuture;
use futures::stream::{self, StreamExt};
use image::Rgb;
use indicatif::ProgressBar;
use memmap2::Mmap;

use crate::progress::{counter_style, multi};
use crate::safetensors::{self, DiffFill, TensorMeta};
use crate::hf_url::{RemoteFileSpec, RemoteRepo};
use crate::xet::{self, XetReader, XetTerm};

/// Bounded concurrency for setup-time HTTP loops (xet reconstruction,
/// safetensors header fetches, non-safetensors diff downloads). The global
/// AIMD throttle still caps the *actual* in-flight count; this just lets the
/// runtime have enough simultaneous awaiting tasks to keep the throttle full.
const SETUP_FETCH_CONCURRENCY: usize = 16;

/// Build a one-shot progress bar attached to the global `MultiProgress` so
/// it interleaves cleanly with log output. Always returns `Some(...)`; the
/// non-TTY case is handled by the hidden draw target on the global multi.
/// `Option<ProgressBar>` is kept in the signature so existing call sites that
/// pattern-match it continue to compile.
fn setup_progress(label: &str, total: u64) -> Option<ProgressBar> {
    let pb = multi().add(ProgressBar::new(total))
        .with_style(counter_style())
        .with_message(label.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    Some(pb)
}

/// Async fetcher closure used by [`Data::LazyDiff`]. Captures its inputs by
/// `Arc` so the returned future is `'static` and can be sent across tasks.
pub type LazyFetcher =
    Arc<dyn Fn(u64, usize) -> BoxFuture<'static, anyhow::Result<Vec<u8>>> + Send + Sync>;

pub struct SafetensorsInfo {
    pub tensors: Vec<TensorMeta>,
    pub color_ranges: Vec<(u64, u64, Rgb<u8>)>,
}

/// Backing storage for a source's bytes.
pub enum Data {
    Mapped(Mmap),
    Owned(Vec<u8>),
    /// Remote file accessed via HF Hub range requests — never loaded locally.
    Http {
        repo: RemoteRepo,
        filename: Arc<String>,
        revision: Arc<String>,
    },
    /// Remote xet-backed file accessed via direct CAS range requests. Each
    /// `fetch_range` issues one or more HTTP GETs against signed xorb URLs
    /// and decompresses the resulting chunk segments locally, bypassing
    /// hf-hub's per-call xet stream-group rebuild. Decoded chunk segments
    /// are cached inside the reader for spatial-locality wins (adjacent
    /// tiles on the Hilbert curve share terms).
    Xet(Arc<XetReader>),
    /// Diff computed on demand per range — never stored in full.
    /// Async-only: the inner closure returns a future so it can issue HTTP
    /// range requests (and await them) without blocking the runtime.
    LazyDiff(LazyFetcher),
    /// Synthetic zero-filled backing for `SourceKind::UnmatchedRegion`. The
    /// renderer overrides bytes in these regions with a crosshatch pattern
    /// anyway, so the underlying bytes are irrelevant — but `fetch_range`
    /// must still return a buffer of the requested length.
    ZeroFill,
}

impl std::ops::Deref for Data {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Data::Mapped(m) => m,
            Data::Owned(v) => v,
            Data::Http { .. } => panic!("bug: use fetch_range() for remote HTTP Data, not Deref"),
            Data::Xet(_) => panic!("bug: use fetch_range() for Xet Data, not Deref"),
            Data::LazyDiff(_) => panic!("bug: use fetch_range() for LazyDiff Data, not Deref"),
            Data::ZeroFill => panic!("bug: use fetch_range() for ZeroFill Data, not Deref"),
        }
    }
}

impl Data {
    /// Return bytes `[start, start+len)` from this source.
    ///
    /// Async because `Http` and `LazyDiff` may issue HTTP range requests.
    /// Local variants (`Mapped`, `Owned`) resolve synchronously inside the
    /// future and incur a `Vec` allocation — the cost is dwarfed by the
    /// surrounding render work. Callers that only handle local data should
    /// use `Deref` for zero-copy slices.
    pub async fn fetch_range(&self, start: u64, len: usize) -> anyhow::Result<Vec<u8>> {
        match self {
            Data::Mapped(m) => Ok(m[start as usize..start as usize + len].to_vec()),
            Data::Owned(v) => Ok(v[start as usize..start as usize + len].to_vec()),
            Data::Http { repo, filename, revision } => {
                repo.fetch_range(filename, revision, start..start + len as u64).await
            }
            Data::Xet(reader) => reader.fetch_range(start, len).await,
            Data::LazyDiff(f) => f(start, len).await,
            Data::ZeroFill => Ok(vec![0u8; len]),
        }
    }

    /// Whether `fetch_range` resolves without issuing an HTTP request.
    ///
    /// `Http`, `Xet`, and `LazyDiff` may all hit the network. The tile load
    /// stage uses this to skip the AIMD HTTP throttle when nothing in flight
    /// could hit the Hub — otherwise mmap reads would be artificially capped
    /// at the throttle's initial 4-way concurrency.
    pub fn is_local(&self) -> bool {
        matches!(self, Data::Mapped(_) | Data::Owned(_) | Data::ZeroFill)
    }
}

/// How a source's bytes are stored.
pub enum SourceKind {
    Buffered(Vec<u8>),
    File(PathBuf),
    Diff { original: PathBuf, modified: PathBuf },
    /// Remote HF file, accessed via hf-hub range requests per tile.
    Http(RemoteFileSpec),
    /// Per-tensor diff computed lazily from two whole-file Data sources.
    /// `byte_size` output bytes are produced on demand. `metric` selects how
    /// per-element deltas are encoded; `scale_orig` carries any per-tensor
    /// statistic the metric needs (RMS of `orig` for `DiffMetric::Rms`),
    /// pre-computed at setup so the per-tile path stays pure-streaming.
    TensorDiff {
        orig: Arc<Data>,
        mod_: Arc<Data>,
        orig_start: u64,
        mod_start: u64,
        orig_dtype: safetensors::Dtype,
        mod_dtype: safetensors::Dtype,
        metric: safetensors::DiffMetric,
        scale_orig: f32,
    },
    /// A canvas region for a tensor / file that exists on only one side of a
    /// diff. The byte_size on the parent `Source` controls how much canvas
    /// space it takes; the underlying bytes are zero (the renderer paints a
    /// crosshatch pattern based on `fill` instead of using the byte LUT).
    UnmatchedRegion { fill: DiffFill },
}

/// Input specification: local file path or resolved remote HF file.
pub enum InputSpec {
    Local(PathBuf),
    Remote(RemoteFileSpec),
}

/// Metadata and storage descriptor for one input.
pub struct Source {
    pub file_idx: usize,
    pub kind: SourceKind,
    pub byte_size: u64,
    /// Populated for safetensors sources (including diff buffers derived from them).
    pub safetensors: Option<SafetensorsInfo>,
    /// Override the display name (used when kind is Buffered but has a real filename).
    pub name_override: Option<String>,
    /// Xet reconstruction terms for this source. `Some(vec)` when xet
    /// visualization was requested and the source has a xet hash; `None`
    /// when xet visualization is off; `Some(vec![])` when xet vis is on but
    /// this source isn't xet-backed.
    pub xet_terms: Option<Vec<XetTerm>>,
}

impl Source {
    /// Human-readable name for this source (file name or "stdin").
    pub fn name(&self) -> String {
        if let Some(ref n) = self.name_override {
            return n.clone();
        }
        match &self.kind {
            SourceKind::File(p) => p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.to_string_lossy().into_owned()),
            SourceKind::Buffered(_) => "stdin".to_string(),
            SourceKind::Diff { original, .. } => original
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| original.to_string_lossy().into_owned()),
            SourceKind::Http(spec) => spec.filename.as_str().to_string(),
            SourceKind::TensorDiff { .. } => {
                unreachable!("TensorDiff sources always have name_override set")
            }
            SourceKind::UnmatchedRegion { .. } => {
                unreachable!("UnmatchedRegion sources always have name_override set")
            }
        }
    }
}

/// Build sources and return total byte count.
///
/// Files are opened lazily (one at a time) to avoid exhausting OS fd limits.
/// Stdin is buffered into memory upfront since its size is unknown.
///
/// For .safetensors files: the header is parsed and attached as SafetensorsInfo
/// for dtype coloring. The file is kept as a single Source (one per file) so that
/// inter-tensor borders are not drawn and the Hilbert curve flows smoothly across
/// the whole file with color transitions only at tensor boundaries.
pub fn prepare_sources(files: &[PathBuf]) -> anyhow::Result<(Vec<Source>, u64)> {
    if files.is_empty() {
        log::info!("Reading stdin...");
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf)?;
        let len = buf.len() as u64;
        return Ok((
            vec![Source {
                file_idx: 0,
                kind: SourceKind::Buffered(buf),
                byte_size: len,
                safetensors: None,
                name_override: None,
                xet_terms: None,
            }],
            len,
        ));
    }

    // Expand any directory paths (e.g. from a repo-level hf:// download) into
    // their constituent files so they can be treated as individual sources.
    let expanded: Vec<PathBuf> = files.iter().flat_map(|p| {
        if p.is_dir() { collect_files_recursive(p) } else { vec![p.clone()] }
    }).collect();

    let mut sources = Vec::new();
    let mut total = 0u64;
    for path in &expanded {
        let size = match std::fs::metadata(path) {
            Ok(m) => m.len(),
            Err(e) => {
                log::warn!("{}: {} — skipping", path.display(), e);
                continue;
            }
        };

        let is_st = path.extension().and_then(|e| e.to_str()) == Some("safetensors");

        let safetensors_info = if is_st {
            match load_safetensors_info(path, size) {
                Ok(info) => Some(info),
                Err(e) => {
                    log::warn!("{}: failed to parse safetensors header: {} — treating as plain binary", path.display(), e);
                    None
                }
            }
        } else {
            None
        };

        total += size;
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::File(path.clone()),
            byte_size: size,
            safetensors: safetensors_info,
            name_override: None,
            xet_terms: None,
        });
    }
    Ok((sources, total))
}

/// Load a source's bytes for random access: mmaps file sources, clones buffered sources.
/// For diff sources, returns a LazyDiff that computes bytes on demand per tile.
/// For Http sources, returns a `Data::Http` handle that fetches byte ranges on demand.
pub fn load_source_data(s: &Source) -> anyhow::Result<Data> {
    match &s.kind {
        SourceKind::File(p) => {
            let f = File::open(p)?;
            Ok(Data::Mapped(unsafe { Mmap::map(&f) }?))
        }
        SourceKind::Buffered(v) => Ok(Data::Owned(v.clone())),
        SourceKind::Diff { original, modified } => {
            let f_o = File::open(original)?;
            let f_m = File::open(modified)?;
            let m_o = Arc::new(unsafe { Mmap::map(&f_o) }?);
            let m_m = Arc::new(unsafe { Mmap::map(&f_m) }?);
            Ok(Data::LazyDiff(Arc::new(move |start: u64, len: usize| {
                let m_o = Arc::clone(&m_o);
                let m_m = Arc::clone(&m_m);
                Box::pin(async move {
                    let a = &m_o[start as usize..start as usize + len];
                    let b = &m_m[start as usize..start as usize + len];
                    Ok(a.iter().zip(b.iter()).map(|(&a, &b)| {
                        let delta = b as i16 - a as i16;
                        let brightness = (delta.unsigned_abs() as f32 / 255.0 * 127.0).round() as u8;
                        if delta >= 0 { 127u8 + brightness } else { 127u8 - brightness }
                    }).collect())
                })
            })))
        }
        SourceKind::Http(spec) => Ok(Data::Http {
            repo: spec.repo.clone(),
            filename: Arc::clone(&spec.filename),
            revision: Arc::clone(&spec.revision),
        }),
        SourceKind::UnmatchedRegion { .. } => Ok(Data::ZeroFill),
        SourceKind::TensorDiff { orig, mod_, orig_start, mod_start, orig_dtype, mod_dtype, metric, scale_orig } => {
            let orig = Arc::clone(orig);
            let mod_ = Arc::clone(mod_);
            let orig_start = *orig_start;
            let mod_start = *mod_start;
            let orig_dtype = *orig_dtype;
            let mod_dtype = *mod_dtype;
            let metric = *metric;
            let scale_orig = *scale_orig;
            Ok(Data::LazyDiff(Arc::new(move |start: u64, len: usize| {
                let orig = Arc::clone(&orig);
                let mod_ = Arc::clone(&mod_);
                Box::pin(async move {
                    let orig_elem = orig_dtype.element_size() as u64;
                    let mod_elem = mod_dtype.element_size() as u64;
                    let ob = orig.fetch_range(orig_start + start * orig_elem, (len as u64 * orig_elem) as usize).await?;
                    let mb = mod_.fetch_range(mod_start + start * mod_elem, (len as u64 * mod_elem) as usize).await?;
                    Ok(orig_dtype.diff_to_u8(&ob, mod_dtype, &mb, metric, scale_orig))
                })
            })))
        }
    }
}

/// Build sources from a mixed list of local paths and remote HF file specs.
/// Remote specs are turned into `SourceKind::Http` entries (no download).
pub fn prepare_sources_from_specs(
    specs: &[InputSpec],
) -> anyhow::Result<(Vec<Source>, u64)> {
    if specs.is_empty() {
        log::info!("Reading stdin...");
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf)?;
        let len = buf.len() as u64;
        return Ok((
            vec![Source {
                file_idx: 0,
                kind: SourceKind::Buffered(buf),
                byte_size: len,
                safetensors: None,
                name_override: None,
                xet_terms: None,
            }],
            len,
        ));
    }

    let mut sources = Vec::new();
    let mut total = 0u64;

    for spec in specs {
        match spec {
            InputSpec::Local(path) => {
                let size = match std::fs::metadata(path) {
                    Ok(m) => m.len(),
                    Err(e) => {
                        log::warn!("{}: {} — skipping", path.display(), e);
                        continue;
                    }
                };
                let is_st = path.extension().and_then(|e| e.to_str()) == Some("safetensors");
                let safetensors_info = if is_st {
                    match load_safetensors_info(path, size) {
                        Ok(info) => Some(info),
                        Err(e) => {
                            log::warn!("{}: failed to parse safetensors header: {} — treating as plain binary", path.display(), e);
                            None
                        }
                    }
                } else {
                    None
                };
                total += size;
                sources.push(Source {
                    file_idx: sources.len(),
                    kind: SourceKind::File(path.clone()),
                    byte_size: size,
                    safetensors: safetensors_info,
                    name_override: None,
                    xet_terms: None,
                });
            }
            InputSpec::Remote(spec) => {
                let size = spec.size;
                total += size;
                sources.push(Source {
                    file_idx: sources.len(),
                    kind: SourceKind::Http(spec.clone()),
                    byte_size: size,
                    safetensors: None,
                    name_override: None,
                    xet_terms: None,
                });
            }
        }
    }

    Ok((sources, total))
}

/// Materialize every `SourceKind::Http` source as a local file via one
/// whole-file download per source, then swap each source to `SourceKind::File`.
///
/// Why: `Data::Http::fetch_range` ultimately hits hf-hub's xet streaming API
/// when the file is xet-backed. That code path has substantial *per-call*
/// setup cost — it fetches a fresh CAS token, builds a `XetDownloadStreamGroup`,
/// and opens a new stream — far more expensive than the ~65 KiB of payload
/// transferred for a single tile. With tens of thousands of tiles per file,
/// the per-call overhead dominates and the pipeline appears stalled.
///
/// One whole-file `download_file` per source amortises that overhead across
/// the entire file (which the renderer will read every byte of anyway). After
/// materialization, all tile reads are mmap'd `memcpy`s — no HTTP, no throttle.
///
/// `populate_xet_terms` must run *before* this so the xet term metadata is
/// captured from the still-remote `RemoteFileSpec`.
pub async fn materialize_http_sources(sources: &mut [Source]) -> anyhow::Result<()> {
    use crate::hf_url::RemoteRepo;
    use crate::throttle::with_throttle;

    // Snapshot (index, spec) for Http sources so the futures don't borrow `sources`.
    let jobs: Vec<(usize, RemoteFileSpec)> = sources
        .iter()
        .enumerate()
        .filter_map(|(i, s)| match &s.kind {
            SourceKind::Http(spec) => Some((i, spec.clone())),
            _ => None,
        })
        .collect();

    if jobs.is_empty() {
        return Ok(());
    }

    let pb = setup_progress("source files (downloading for xet view)", jobs.len() as u64);
    let pb_for_workers = pb.clone();

    let downloads: Vec<(usize, anyhow::Result<PathBuf>)> = stream::iter(jobs)
        .map(|(i, spec)| {
            let pb = pb_for_workers.clone();
            async move {
                let filename = (*spec.filename).clone();
                let revision = (*spec.revision).clone();
                let label = format!("download_file {filename}");
                let result = with_throttle(&label, || async {
                    match &spec.repo {
                        RemoteRepo::Model(r) => {
                            r.download_file()
                                .filename(filename.clone())
                                .revision(revision.clone())
                                .send()
                                .await
                        }
                        RemoteRepo::Dataset(r) => {
                            r.download_file()
                                .filename(filename.clone())
                                .revision(revision.clone())
                                .send()
                                .await
                        }
                        RemoteRepo::Space(r) => {
                            r.download_file()
                                .filename(filename.clone())
                                .revision(revision.clone())
                                .send()
                                .await
                        }
                    }
                })
                .await
                .map_err(anyhow::Error::from);
                if let Some(pb) = pb.as_ref() {
                    pb.inc(1);
                }
                (i, result)
            }
        })
        .buffer_unordered(SETUP_FETCH_CONCURRENCY)
        .collect()
        .await;

    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }

    for (i, r) in downloads {
        let path = r?;
        // Preserve display name + xet_terms; only the storage kind changes.
        let display = sources[i].name();
        sources[i].kind = SourceKind::File(path);
        if sources[i].name_override.is_none() {
            sources[i].name_override = Some(display);
        }
    }
    Ok(())
}

/// Fetch xet reconstruction terms for any HTTP-backed sources.
///
/// Each source gets `xet_terms = Some(vec)` — empty for sources without a xet
/// hash (local files, non-xet remote files), populated for xet-backed remote
/// sources. Errors from the xet endpoints propagate up.
pub async fn populate_xet_terms(sources: &mut [Source]) -> anyhow::Result<()> {
    // Fetch xet reconstructions concurrently — each Http source needs two
    // throttled HTTP round-trips (`xet-read-token` + `reconstructions/{hash}`)
    // and the global throttle caps the real concurrency, so let the runtime
    // have plenty of awaiting tasks.
    let pb = setup_progress("source files (xet reconstruction)", sources.len() as u64);
    let pb_for_close = pb.clone();

    // Decouple the per-source future from the borrow on `sources` by snapshotting
    // the index + http spec, then writing results back by index.
    let jobs: Vec<(usize, Option<RemoteFileSpec>)> = sources
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let spec = if let SourceKind::Http(spec) = &s.kind {
                Some(spec.clone())
            } else {
                None
            };
            (i, spec)
        })
        .collect();

    let pb_for_workers = pb.clone();
    let mut results: Vec<(usize, anyhow::Result<Vec<XetTerm>>)> = stream::iter(jobs)
        .map(|(i, maybe_spec)| {
            let pb = pb_for_workers.clone();
            async move {
                let terms = match maybe_spec {
                    Some(spec) => xet::reconstruction_for(&spec).await,
                    None => Ok(Vec::new()),
                };
                if let Some(pb) = pb.as_ref() {
                    pb.inc(1);
                }
                (i, terms)
            }
        })
        .buffer_unordered(SETUP_FETCH_CONCURRENCY)
        .collect()
        .await;

    if let Some(pb) = pb_for_close.as_ref() {
        pb.finish_and_clear();
    }

    // Stable order: write each result back to its original index. The first
    // error wins; later sources still get `xet_terms = None` so the caller
    // can distinguish "didn't fetch" from "fetched but empty".
    results.sort_by_key(|(i, _)| *i);
    for (i, r) in results {
        sources[i].xet_terms = Some(r?);
    }
    Ok(())
}

/// Read just the header of a safetensors file and return parsed metadata.
fn load_safetensors_info(path: &Path, file_size: u64) -> anyhow::Result<SafetensorsInfo> {
    // Read the first 8 bytes to get header_size, then read header_size more bytes.
    let mut f = File::open(path)?;
    let mut size_buf = [0u8; 8];
    f.read_exact(&mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf);
    if header_size > 100 * 1024 * 1024 {
        anyhow::bail!("header_size={} exceeds 100 MB safety limit", header_size);
    }
    let total_header = 8 + header_size as usize;
    let mut header_buf = vec![0u8; total_header];
    header_buf[..8].copy_from_slice(&size_buf);
    f.read_exact(&mut header_buf[8..])?;

    let (tensors, header_end) = safetensors::parse_header(&header_buf)?;
    let color_ranges = safetensors::build_color_ranges(&tensors, header_end, file_size);
    Ok(SafetensorsInfo { tensors, color_ranges })
}

/// Fetch and parse the safetensors header from any Data source.
/// For local sources (Mapped/Owned): zero-copy slice access.
/// For remote sources (Http): two range requests (8 bytes, then full header).
async fn fetch_safetensors_header(data: &Data) -> anyhow::Result<(Vec<safetensors::TensorMeta>, u64)> {
    let size_bytes = data.fetch_range(0, 8).await?;
    let header_size = u64::from_le_bytes(size_bytes[..8].try_into().unwrap());
    if header_size > 100 * 1024 * 1024 {
        anyhow::bail!("safetensors header_size={} exceeds 100 MB safety limit", header_size);
    }
    let total_header = 8 + header_size as usize;
    let header_bytes = data.fetch_range(0, total_header).await?;
    safetensors::parse_header(&header_bytes)
}

/// Recursively collect all files under `root`, sorted by path.
fn collect_files_recursive(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_recursive(root, &mut files);
    files.sort();
    files
}

fn collect_recursive(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            log::warn!("{}: {} — skipping", dir.display(), e);
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            files.push(path);
        } else if path.is_dir() {
            collect_recursive(&path, files);
        }
    }
}

/// Build diff sources from two files or two directories.
///
/// For files: both must be the same size (error if not), unless both are safetensors
/// (in which case a tensor-aligned diff buffer is computed).
/// For directories: files are matched by relative path; pairs with mismatched sizes
/// or no counterpart on the other side are skipped with a warning.
pub async fn prepare_diff_sources(
    original: &Path,
    modified: &Path,
    is_finetune: bool,
    metric: safetensors::DiffMetric,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let orig_is_file = original.is_file();
    let mod_is_file = modified.is_file();
    let orig_is_dir = original.is_dir();
    let mod_is_dir = modified.is_dir();

    let is_st = |p: &Path| -> bool {
        p.extension().and_then(|e| e.to_str()) == Some("safetensors")
    };

    if orig_is_file && mod_is_file {
        // Safetensors diff: expand into per-tensor diff Sources (one per matched pair).
        if is_st(original) && is_st(modified) {
            return build_safetensors_diff_sources(original, modified, is_finetune, metric).await;
        }

        let size_o = std::fs::metadata(original)?.len();
        let size_m = std::fs::metadata(modified)?.len();
        if size_o != size_m {
            anyhow::bail!(
                "--diff: file sizes differ ({} bytes vs {} bytes): {} vs {}",
                size_o,
                size_m,
                original.display(),
                modified.display()
            );
        }
        let source = Source {
            file_idx: 0,
            kind: SourceKind::Diff {
                original: original.to_path_buf(),
                modified: modified.to_path_buf(),
            },
            byte_size: size_o,
            safetensors: None,
            name_override: None,
            xet_terms: None,
        };
        return Ok((vec![source], size_o));
    }

    if orig_is_dir && mod_is_dir {
        let orig_files = collect_files_recursive(original);
        let mod_files = collect_files_recursive(modified);

        let mut sources = Vec::new();
        let mut total = 0u64;

        // Safetensors: match by tensor name across ALL .safetensors files on each side,
        // regardless of filename. This handles sharded vs unsharded and different naming
        // conventions (e.g. model.safetensors-00001-of-00001.safetensors vs model.safetensors).
        let orig_st: Vec<PathBuf> = orig_files.iter().filter(|p| is_st(p)).cloned().collect();
        let mod_st: Vec<PathBuf> = mod_files.iter().filter(|p| is_st(p)).cloned().collect();
        if !orig_st.is_empty() || !mod_st.is_empty() {
            match build_multi_safetensors_diff_sources(&orig_st, &mod_st, is_finetune, metric).await {
                Ok((mut tensor_sources, bytes)) => {
                    let base_idx = sources.len();
                    for s in &mut tensor_sources {
                        s.file_idx += base_idx;
                    }
                    sources.extend(tensor_sources);
                    total += bytes;
                }
                Err(e) => log::warn!("safetensors diff failed: {e} — skipping"),
            }
        }

        // Non-safetensors: match by relative path. Same-size pairs become a
        // byte diff; different-size or single-side files become crosshatched
        // unmatched regions so they remain visible.
        let orig_fill_kind = if is_finetune { DiffFill::Grey } else { DiffFill::Red };
        let orig_map: HashMap<PathBuf, PathBuf> = orig_files
            .iter()
            .filter(|p| !is_st(p))
            .filter_map(|p| {
                p.strip_prefix(original)
                    .ok()
                    .map(|rel| (rel.to_path_buf(), p.clone()))
            })
            .collect();
        let mod_map: HashMap<PathBuf, PathBuf> = mod_files
            .iter()
            .filter(|p| !is_st(p))
            .filter_map(|p| {
                p.strip_prefix(modified)
                    .ok()
                    .map(|rel| (rel.to_path_buf(), p.clone()))
            })
            .collect();

        let mut mod_only_keys: Vec<&PathBuf> = mod_map.keys()
            .filter(|k| !orig_map.contains_key(*k))
            .collect();
        mod_only_keys.sort();
        if is_finetune && !mod_only_keys.is_empty() {
            let names: Vec<String> = mod_only_keys.iter()
                .map(|rel| modified.join(rel).display().to_string())
                .collect();
            log::warn!(
                "--diff --finetune: modified side has {} file(s) with no counterpart on the \
                 original/base side — rendering as green crosshatch: {}",
                names.len(),
                names.join(", ")
            );
        }

        let mut sorted_keys: Vec<&PathBuf> = orig_map.keys().collect();
        sorted_keys.sort();

        for rel in sorted_keys {
            let orig_abs = &orig_map[rel];
            let size_o = match std::fs::metadata(orig_abs) {
                Ok(m) => m.len(),
                Err(e) => {
                    log::warn!("{}: {} — skipping", orig_abs.display(), e);
                    continue;
                }
            };
            match mod_map.get(rel) {
                None => {
                    if size_o == 0 { continue; }
                    sources.push(Source {
                        file_idx: sources.len(),
                        kind: SourceKind::UnmatchedRegion { fill: orig_fill_kind },
                        byte_size: size_o,
                        safetensors: None,
                        name_override: Some(format!("[only in original] {}", rel.display())),
                        xet_terms: None,
                    });
                    total += size_o;
                }
                Some(mod_abs) => {
                    let size_m = match std::fs::metadata(mod_abs) {
                        Ok(m) => m.len(),
                        Err(e) => {
                            log::warn!("{}: {} — skipping", mod_abs.display(), e);
                            continue;
                        }
                    };
                    if size_o != size_m {
                        // Size mismatch: render each side independently as
                        // unmatched so neither is hidden. In finetune mode
                        // this is technically a contract violation (the
                        // finetune carries bytes the base doesn't), but we
                        // warn rather than bail and still render — the
                        // green crosshatch makes the divergence visually
                        // obvious.
                        if is_finetune {
                            log::warn!(
                                "--diff --finetune: size mismatch ({} vs {} bytes) for {} — \
                                 rendering modified side as green crosshatch",
                                size_o, size_m, rel.display()
                            );
                        } else {
                            log::warn!(
                                "size mismatch ({} vs {} bytes) for {} — rendering each side as unmatched",
                                size_o, size_m, rel.display()
                            );
                        }
                        if size_o > 0 {
                            sources.push(Source {
                                file_idx: sources.len(),
                                kind: SourceKind::UnmatchedRegion { fill: orig_fill_kind },
                                byte_size: size_o,
                                safetensors: None,
                                name_override: Some(format!("[only in original] {}", rel.display())),
                                xet_terms: None,
                            });
                            total += size_o;
                        }
                        if size_m > 0 {
                            sources.push(Source {
                                file_idx: sources.len(),
                                kind: SourceKind::UnmatchedRegion { fill: DiffFill::Green },
                                byte_size: size_m,
                                safetensors: None,
                                name_override: Some(format!("[only in modified] {}", rel.display())),
                                xet_terms: None,
                            });
                            total += size_m;
                        }
                        continue;
                    }
                    sources.push(Source {
                        file_idx: sources.len(),
                        kind: SourceKind::Diff {
                            original: orig_abs.clone(),
                            modified: mod_abs.clone(),
                        },
                        byte_size: size_o,
                        safetensors: None,
                        name_override: None,
                        xet_terms: None,
                    });
                    total += size_o;
                }
            }
        }

        // mod-only files (non-finetune case — finetune bailed earlier).
        for rel in &mod_only_keys {
            let mod_abs = &mod_map[*rel];
            let size_m = match std::fs::metadata(mod_abs) {
                Ok(m) => m.len(),
                Err(e) => {
                    log::warn!("{}: {} — skipping", mod_abs.display(), e);
                    continue;
                }
            };
            if size_m == 0 { continue; }
            sources.push(Source {
                file_idx: sources.len(),
                kind: SourceKind::UnmatchedRegion { fill: DiffFill::Green },
                byte_size: size_m,
                safetensors: None,
                name_override: Some(format!("[only in modified] {}", rel.display())),
                xet_terms: None,
            });
            total += size_m;
        }

        if sources.is_empty() {
            anyhow::bail!("--diff: no matching file pairs found between the two directories");
        }
        return Ok((sources, total));
    }

    anyhow::bail!(
        "--diff: both arguments must be files or both must be directories (got {} and {})",
        if orig_is_file { "file" } else if orig_is_dir { "directory" } else { "missing path" },
        if mod_is_file { "file" } else if mod_is_dir { "directory" } else { "missing path" }
    );
}

/// Build diff sources from two repos listed as HTTP specs (no download).
///
/// Safetensors files are diffed lazily via range requests — no model weights
/// are downloaded to disk or held in RAM. Small non-safetensors files (≤16 MB)
/// are downloaded eagerly and binary-diffed; larger ones are skipped with a warning.
pub async fn prepare_diff_sources_from_http(
    orig_specs: &[(String, RemoteFileSpec)],
    mod_specs: &[(String, RemoteFileSpec)],
    is_finetune: bool,
    metric: safetensors::DiffMetric,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let is_st = |name: &str| name.ends_with(".safetensors");

    let orig_st: Vec<&(String, RemoteFileSpec)> = orig_specs.iter().filter(|(n, _)| is_st(n)).collect();
    let mod_st: Vec<&(String, RemoteFileSpec)>  = mod_specs.iter().filter(|(n, _)| is_st(n)).collect();

    let mut sources: Vec<Source> = Vec::new();
    let mut total = 0u64;

    // Safetensors diff — fully lazy, no download.
    if !orig_st.is_empty() || !mod_st.is_empty() {
        match build_multi_safetensors_diff_sources_from_http(&orig_st, &mod_st, is_finetune, metric).await {
            Ok((mut tensor_sources, bytes)) => {
                let base_idx = sources.len();
                for s in &mut tensor_sources { s.file_idx += base_idx; }
                sources.extend(tensor_sources);
                total += bytes;
            }
            Err(e) => log::warn!("safetensors diff failed: {e} — skipping"),
        }
    }

    // Non-safetensors files: match by filename. Same-size pairs become a byte
    // diff (downloaded if small); different-size or single-side files become
    // crosshatched unmatched regions so they remain visible. Large files are
    // sized but rendered as the orig-fill kind (we can't byte-diff something
    // we won't download).
    const MAX_EAGER_SIZE: u64 = 16 * 1024 * 1024;
    let orig_fill_kind = if is_finetune { DiffFill::Grey } else { DiffFill::Red };
    let orig_non: HashMap<&str, &RemoteFileSpec> =
        orig_specs.iter().filter(|(n, _)| !is_st(n)).map(|(n, s)| (n.as_str(), s)).collect();
    let mod_non: HashMap<&str, &RemoteFileSpec> =
        mod_specs.iter().filter(|(n, _)| !is_st(n)).map(|(n, s)| (n.as_str(), s)).collect();

    let mut mod_only_files: Vec<&str> = mod_non.keys().copied()
        .filter(|k| !orig_non.contains_key(k))
        .collect();
    mod_only_files.sort();
    if is_finetune && !mod_only_files.is_empty() {
        log::warn!(
            "--diff --finetune: modified side has {} file(s) with no counterpart on the \
             original/base side — rendering as green crosshatch: {}",
            mod_only_files.len(),
            mod_only_files.join(", ")
        );
    }

    let mut sorted: Vec<&str> = orig_non.keys().copied().collect();
    sorted.sort();

    // First pass (sync): partition into byte-diff jobs vs unmatched-region
    // sources. Diff jobs are downloaded in parallel afterwards.
    let mut diff_jobs: Vec<(String, RemoteFileSpec, RemoteFileSpec)> = Vec::new();
    let mut unmatched_orig: Vec<(String, u64, DiffFill)> = Vec::new();
    let mut unmatched_mod: Vec<(String, u64, DiffFill)> = Vec::new();
    for fname in sorted {
        let orig_spec = &orig_non[fname];
        let mod_spec = match mod_non.get(fname) {
            Some(s) => s,
            None => {
                if orig_spec.size > 0 {
                    unmatched_orig.push((fname.to_string(), orig_spec.size, orig_fill_kind));
                }
                continue;
            }
        };
        if orig_spec.size != mod_spec.size {
            if is_finetune {
                log::warn!(
                    "--diff --finetune: size mismatch for {fname} ({} vs {} bytes) — \
                     rendering modified side as green crosshatch",
                    orig_spec.size, mod_spec.size
                );
            } else {
                log::warn!(
                    "size mismatch for {fname} ({} vs {} bytes) — rendering each side as unmatched",
                    orig_spec.size, mod_spec.size
                );
            }
            if orig_spec.size > 0 {
                unmatched_orig.push((fname.to_string(), orig_spec.size, orig_fill_kind));
            }
            if mod_spec.size > 0 {
                unmatched_mod.push((fname.to_string(), mod_spec.size, DiffFill::Green));
            }
            continue;
        }
        if orig_spec.size > MAX_EAGER_SIZE {
            // Too large to byte-diff lazily; show the orig footprint as an
            // unmatched-region marker so the canvas still reflects the file's
            // existence. (Could be promoted to a tile-streamed lazy diff
            // later, but for now we just preserve visibility.)
            log::warn!(
                "{fname} exceeds {} MB — rendering as crosshatched region instead of byte diff",
                MAX_EAGER_SIZE / 1024 / 1024
            );
            unmatched_orig.push((fname.to_string(), orig_spec.size, orig_fill_kind));
            continue;
        }
        diff_jobs.push((fname.to_string(), (*orig_spec).clone(), (*mod_spec).clone()));
    }
    for fname in &mod_only_files {
        let spec = &mod_non[fname];
        if spec.size > 0 {
            unmatched_mod.push((fname.to_string(), spec.size, DiffFill::Green));
        }
    }

    let pb = setup_progress("file pairs (non-safetensors diff downloads)", diff_jobs.len() as u64);
    let pb_for_workers = pb.clone();
    let diffs: Vec<anyhow::Result<(String, Vec<u8>)>> = stream::iter(diff_jobs)
        .map(|(fname, orig_spec, mod_spec)| {
            let pb = pb_for_workers.clone();
            async move {
                let orig_data = Data::Http {
                    repo: orig_spec.repo.clone(),
                    filename: Arc::clone(&orig_spec.filename),
                    revision: Arc::clone(&orig_spec.revision),
                };
                let mod_data = Data::Http {
                    repo: mod_spec.repo.clone(),
                    filename: Arc::clone(&mod_spec.filename),
                    revision: Arc::clone(&mod_spec.revision),
                };
                let ob = orig_data.fetch_range(0, orig_spec.size as usize).await?;
                let mb = mod_data.fetch_range(0, mod_spec.size as usize).await?;
                let diff: Vec<u8> = ob.iter().zip(mb.iter()).map(|(&a, &b)| {
                    let delta = b as i16 - a as i16;
                    let brightness = (delta.unsigned_abs() as f32 / 255.0 * 127.0).round() as u8;
                    if delta >= 0 { 127u8 + brightness } else { 127u8 - brightness }
                }).collect();
                if let Some(pb) = pb.as_ref() {
                    pb.inc(1);
                }
                Ok((fname, diff))
            }
        })
        .buffer_unordered(SETUP_FETCH_CONCURRENCY)
        .collect()
        .await;
    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }
    // Re-sort by filename so the Source order is deterministic.
    let mut diffs: Vec<(String, Vec<u8>)> = diffs.into_iter().collect::<anyhow::Result<Vec<_>>>()?;
    diffs.sort_by(|a, b| a.0.cmp(&b.0));
    for (fname, diff) in diffs {
        let size = diff.len() as u64;
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::Buffered(diff),
            byte_size: size,
            safetensors: None,
            name_override: Some(fname),
            xet_terms: None,
        });
        total += size;
    }

    // Unmatched / oversize / size-mismatch files surface as crosshatched
    // regions so the user sees they exist even though no byte diff was
    // computed.
    unmatched_orig.sort();
    for (fname, size, fill) in unmatched_orig {
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill },
            byte_size: size,
            safetensors: None,
            name_override: Some(format!("[only in original] {fname}")),
            xet_terms: None,
        });
        total += size;
    }
    unmatched_mod.sort();
    for (fname, size, fill) in unmatched_mod {
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill },
            byte_size: size,
            safetensors: None,
            name_override: Some(format!("[only in modified] {fname}")),
            xet_terms: None,
        });
        total += size;
    }

    if sources.is_empty() {
        anyhow::bail!("--diff: no matching file pairs found between the two repos");
    }
    Ok((sources, total))
}

/// Strip the first `n` dot-delimited path components from `name`.
/// Returns the remainder after the n-th dot, or `None` if `name` has fewer than n+1 components.
fn strip_prefix_components(name: &str, n: usize) -> Option<&str> {
    let mut idx = 0;
    for _ in 0..n {
        idx += name[idx..].find('.')? + 1;
    }
    Some(&name[idx..])
}

/// Find (strip_orig, strip_mod) prefix depths that maximise unique 1-to-1 tensor name matches.
/// Returns (0, 0) if exact matching already produces matches.
fn find_strip_depths(
    orig_names: &[String],
    mod_names: &[String],
) -> (usize, usize) {
    // Count unique-suffix occurrences for a set of names stripped by `n` components.
    let unique_suffixes = |names: &[String], n: usize| -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for name in names {
            if let Some(s) = strip_prefix_components(name, n) {
                if !s.is_empty() {
                    *counts.entry(s.to_owned()).or_insert(0) += 1;
                }
            }
        }
        counts
    };

    let mut best = (0usize, 0usize, 0usize); // (strip_orig, strip_mod, match_count)
    for so in 0usize..=5 {
        let orig_counts = unique_suffixes(orig_names, so);
        for sm in 0usize..=5 {
            if so == 0 && sm == 0 { continue; }
            let mod_counts = unique_suffixes(mod_names, sm);
            let matches = orig_counts.iter()
                .filter(|(s, &oc)| oc == 1 && mod_counts.get(s.as_str()) == Some(&1))
                .count();
            if matches > best.2 {
                best = (so, sm, matches);
            }
        }
    }
    (best.0, best.1)
}

/// Result of tensor-name matching across the two sides of a diff.
pub struct TensorMatch {
    /// 1-to-1 matched pairs `(orig_full, mod_full)`, sorted by `orig_full`.
    pub pairs: Vec<(String, String)>,
    /// Tensor full names present only on the original side, sorted.
    pub orig_only: Vec<String>,
    /// Tensor full names present only on the modified side, sorted.
    pub mod_only: Vec<String>,
}

/// Match tensors under a fixed `(strip_o, strip_m)` strip pair: build the
/// stripped-suffix maps (with collisions blanked), then pair up unique 1-to-1
/// matches. Returns the matched `(orig_full, mod_full)` pairs only — caller
/// drives iteration and tracks the unmatched residual.
fn match_under_strip_depths(
    orig_names: &[String],
    mod_names: &[String],
    strip_o: usize,
    strip_m: usize,
) -> Vec<(String, String)> {
    let orig_by_stripped: HashMap<String, &str> = orig_names.iter()
        .filter_map(|n| {
            strip_prefix_components(n, strip_o)
                .filter(|s| !s.is_empty())
                .map(|s| (s.to_owned(), n.as_str()))
        })
        .fold(HashMap::new(), |mut acc, (stripped, full)| {
            acc.entry(stripped).and_modify(|v| *v = "").or_insert(full);
            acc
        });

    let mod_by_stripped: HashMap<String, &str> = mod_names.iter()
        .filter_map(|n| {
            strip_prefix_components(n, strip_m)
                .filter(|s| !s.is_empty())
                .map(|s| (s.to_owned(), n.as_str()))
        })
        .fold(HashMap::new(), |mut acc, (stripped, full)| {
            acc.entry(stripped).and_modify(|v| *v = "").or_insert(full);
            acc
        });

    let mut sorted_orig: Vec<&str> = orig_by_stripped.values().copied().filter(|s| !s.is_empty()).collect();
    sorted_orig.sort();

    let mut pairs = Vec::new();
    for orig_full in sorted_orig {
        let stripped = match strip_prefix_components(orig_full, strip_o) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        if let Some(&mod_full) = mod_by_stripped.get(stripped) {
            if !mod_full.is_empty() {
                pairs.push((orig_full.to_owned(), mod_full.to_owned()));
            }
        }
    }
    pairs
}

/// Find matched + unmatched tensor name groupings between two name sets.
///
/// **Multi-pass strip heuristic.** Real-world model files frequently mix
/// multiple wrapper-induced prefix nestings — e.g. GRaPE-2-Nano's language
/// tensors live under `model.language_model.language_model.language_model.*`
/// (matching the base's `model.language_model.*` at strip depths `(1, 3)`)
/// while its vision tensors live under `model.language_model.visual.*`
/// (matching the base's `model.visual.*` at strip depths `(2, 3)`). A single
/// `(strip_o, strip_m)` pair can't capture both — so we iterate:
///
/// 1. Pull out all exact-name matches first.
/// 2. Greedily pick the best `(strip_o, strip_m)` over the remaining
///    unmatched tensors, apply those matches, repeat.
/// 3. Stop when no further pair yields any matches.
///
/// Unmatched tensors are returned so callers can surface them (e.g. as
/// crosshatch fill) rather than silently dropping them.
fn find_matched_tensor_pairs(orig_names: &[String], mod_names: &[String]) -> TensorMatch {
    use std::collections::HashSet;
    let mut remaining_orig: HashSet<String> = orig_names.iter().cloned().collect();
    let mut remaining_mod:  HashSet<String> = mod_names.iter().cloned().collect();

    let mut pairs: Vec<(String, String)> = Vec::new();

    // Pass 0: exact-name overlap. Iterate the input order so the resulting
    // log line is deterministic on repeat runs.
    for name in orig_names {
        if remaining_orig.contains(name) && remaining_mod.contains(name) {
            remaining_orig.remove(name);
            remaining_mod.remove(name);
            pairs.push((name.clone(), name.clone()));
        }
    }
    if !pairs.is_empty() {
        log::debug!("safetensors diff: pass 0 exact match — {} pairs", pairs.len());
    }

    // Subsequent passes: greedy multi-pass strip search. Each pass picks the
    // (strip_o, strip_m) that yields the most matches over the *remaining*
    // unmatched tensors, applies those matches, and continues. Bounded by
    // the strip search range (find_strip_depths) and by termination once no
    // pair yields any matches.
    let mut pass = 0usize;
    loop {
        if remaining_orig.is_empty() || remaining_mod.is_empty() { break; }
        let orig_vec: Vec<String> = remaining_orig.iter().cloned().collect();
        let mod_vec:  Vec<String> = remaining_mod.iter().cloned().collect();
        let (strip_o, strip_m) = find_strip_depths(&orig_vec, &mod_vec);
        if strip_o == 0 && strip_m == 0 { break; }
        let new_pairs = match_under_strip_depths(&orig_vec, &mod_vec, strip_o, strip_m);
        if new_pairs.is_empty() { break; }
        pass += 1;
        log::info!(
            "safetensors diff: strip-match pass {}: stripping {} component(s) from original and {} from modified — {} new pair(s)",
            pass, strip_o, strip_m, new_pairs.len()
        );
        for (o, m) in &new_pairs {
            remaining_orig.remove(o);
            remaining_mod.remove(m);
        }
        pairs.extend(new_pairs);
    }

    let mut orig_only: Vec<String> = remaining_orig.into_iter().collect();
    let mut mod_only:  Vec<String> = remaining_mod.into_iter().collect();
    orig_only.sort();
    mod_only.sort();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    TensorMatch { pairs, orig_only, mod_only }
}

/// Sample each matched orig tensor's bytes and compute its RMS, used as the
/// per-tensor scale for `DiffMetric::Rms`. A 64 KB contiguous prefix is more
/// than enough for a stable estimate; for HTTP/Xet sources this is one extra
/// range fetch per matched tensor at setup, parallelised over the throttle.
async fn fetch_rms_estimates(
    paired_ok: &[(String, String)],
    orig_map: &HashMap<String, (usize, safetensors::TensorMeta)>,
    orig_data: &[Arc<Data>],
) -> anyhow::Result<Vec<f32>> {
    const SCALE_SAMPLE_BYTES: u64 = 64 * 1024;
    let pb = setup_progress("orig tensor RMS samples", paired_ok.len() as u64);
    let inputs: Vec<(usize, usize, u64, u64, safetensors::Dtype)> = paired_ok.iter()
        .enumerate()
        .map(|(idx, (orig_full, _))| {
            let (oi, orig_t) = &orig_map[orig_full];
            let elem = orig_t.dtype.element_size() as u64;
            let tensor_bytes = orig_t.file_end.saturating_sub(orig_t.file_start);
            let want = SCALE_SAMPLE_BYTES.min(tensor_bytes);
            // Align sample length down to a whole element so rms_from_buf
            // only sees complete values.
            let len = if elem > 0 { (want / elem) * elem } else { 0 };
            (idx, *oi, orig_t.file_start, len, orig_t.dtype)
        })
        .collect();
    let mut out: Vec<(usize, f32)> = stream::iter(inputs)
        .map(|(idx, oi, start, len, dtype)| {
            let d = Arc::clone(&orig_data[oi]);
            let pb = pb.clone();
            async move {
                let scale = if len == 0 { 0.0 } else {
                    match d.fetch_range(start, len as usize).await {
                        Ok(bytes) => safetensors::rms_from_buf(dtype, &bytes),
                        Err(e) => {
                            log::warn!("safetensors diff: orig RMS sample failed ({e}); using 0.0");
                            0.0
                        }
                    }
                };
                if let Some(pb) = pb.as_ref() { pb.inc(1); }
                (idx, scale)
            }
        })
        .buffer_unordered(SETUP_FETCH_CONCURRENCY)
        .collect()
        .await;
    if let Some(pb) = pb.as_ref() { pb.finish_and_clear(); }
    out.sort_by_key(|(i, _)| *i);
    Ok(out.into_iter().map(|(_, s)| s).collect())
}

/// Core tensor diff builder: given two parallel lists of whole-file Data sources,
/// build per-tensor TensorDiff Source entries without reading any tensor bytes.
async fn build_multi_safetensors_diff_sources_inner(
    orig_data: &[Arc<Data>],
    mod_data:  &[Arc<Data>],
    is_finetune: bool,
    metric: safetensors::DiffMetric,
) -> anyhow::Result<(Vec<Source>, u64)> {
    // Two HTTP range requests per file (8-byte preamble + variable header).
    // For sharded models this is dozens of files per side; serializing them
    // wastes the throttle. Fetch both sides concurrently with bounded
    // parallelism.
    let total = (orig_data.len() + mod_data.len()) as u64;
    let pb = setup_progress("source files (safetensors headers)", total);

    async fn fetch_all(
        data: &[Arc<Data>],
        side: &'static str,
        pb: &Option<ProgressBar>,
    ) -> anyhow::Result<Vec<(usize, Vec<safetensors::TensorMeta>)>> {
        let pb = pb.clone();
        let mut out: Vec<(usize, anyhow::Result<Vec<safetensors::TensorMeta>>)> =
            stream::iter(data.iter().enumerate())
                .map(|(i, d)| {
                    let pb = pb.clone();
                    let d = Arc::clone(d);
                    async move {
                        let r = fetch_safetensors_header(&d)
                            .await
                            .map(|(t, _)| t)
                            .with_context(|| format!("reading safetensors header for {side} file {i}"));
                        if let Some(pb) = pb.as_ref() {
                            pb.inc(1);
                        }
                        (i, r)
                    }
                })
                .buffer_unordered(SETUP_FETCH_CONCURRENCY)
                .collect()
                .await;
        out.sort_by_key(|(i, _)| *i);
        out.into_iter().map(|(i, r)| r.map(|t| (i, t))).collect()
    }

    let orig_headers = fetch_all(orig_data, "orig", &pb).await?;
    let mod_headers = fetch_all(mod_data, "mod", &pb).await?;
    if let Some(pb) = pb.as_ref() {
        pb.finish_and_clear();
    }

    let mut orig_map: HashMap<String, (usize, safetensors::TensorMeta)> = HashMap::new();
    for (i, tensors) in orig_headers {
        for t in tensors {
            orig_map.entry(t.name.clone()).or_insert((i, t));
        }
    }
    let mut mod_map: HashMap<String, (usize, safetensors::TensorMeta)> = HashMap::new();
    for (i, tensors) in mod_headers {
        for t in tensors {
            mod_map.entry(t.name.clone()).or_insert((i, t));
        }
    }

    let orig_names: Vec<String> = orig_map.keys().cloned().collect();
    let mod_names:  Vec<String> = mod_map.keys().cloned().collect();
    let TensorMatch { pairs, orig_only, mod_only } =
        find_matched_tensor_pairs(&orig_names, &mod_names);

    // Tensors present in both, but with incompatible shapes, can't be diffed
    // element-wise. Treat each side independently: in non-finetune mode both
    // sides surface as unmatched (red on orig, green on mod). In finetune
    // mode the modified side is an error (see below), so we fail fast.
    let mut shape_mismatch: Vec<(String, String)> = Vec::new();
    let mut paired_ok: Vec<(String, String)> = Vec::with_capacity(pairs.len());
    for (orig_full, mod_full) in pairs {
        let orig_t = &orig_map[&orig_full].1;
        let mod_t = &mod_map[&mod_full].1;
        if orig_t.shape != mod_t.shape {
            shape_mismatch.push((orig_full, mod_full));
        } else {
            paired_ok.push((orig_full, mod_full));
        }
    }

    // Finetune contract: every tensor the finetune ships should exist (with
    // the same shape) on the base side. Real-world models sometimes break
    // this — e.g. wrapper-saved finetunes that nest a vision tower under a
    // language_model prefix the base doesn't share — so we warn rather than
    // bail, and surface the offending tensors as green crosshatch via the
    // normal only-in-modified rendering below.
    if is_finetune {
        let mut mod_extras: Vec<String> = mod_only.clone();
        for (_, mod_full) in &shape_mismatch {
            mod_extras.push(mod_full.clone());
        }
        if !mod_extras.is_empty() {
            mod_extras.sort();
            log::warn!(
                "safetensors diff --finetune: modified side has {} tensor(s) not present \
                 (or with mismatched shape) on the original/base side — rendering as green \
                 crosshatch: {}",
                mod_extras.len(),
                mod_extras.join(", ")
            );
        }
    }

    // For DiffMetric::Rms we need a per-tensor scale (RMS of orig). Sample
    // up to RMS_SAMPLE_ELEMS elements per tensor via a single range fetch;
    // for HTTP sources this is one extra request per tensor at setup time,
    // for local mmap it's free. AbsLog and Exact don't need a scale.
    let scales: Vec<f32> = if matches!(metric, safetensors::DiffMetric::Rms) {
        fetch_rms_estimates(&paired_ok, &orig_map, orig_data).await?
    } else {
        vec![0.0; paired_ok.len()]
    };

    let mut sources: Vec<Source> = Vec::new();
    let mut total = 0u64;

    for ((orig_full, mod_full), scale_orig) in paired_ok.iter().zip(scales.iter()) {
        let (oi, orig_t) = &orig_map[orig_full];
        let (mi, mod_t)  = &mod_map[mod_full];

        let nelem: u64 = orig_t.shape.iter().product();
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::TensorDiff {
                orig: Arc::clone(&orig_data[*oi]),
                mod_: Arc::clone(&mod_data[*mi]),
                orig_start: orig_t.file_start,
                mod_start:  mod_t.file_start,
                orig_dtype: orig_t.dtype,
                mod_dtype:  mod_t.dtype,
                metric,
                scale_orig: *scale_orig,
            },
            byte_size: nelem,
            safetensors: None,
            name_override: Some(orig_t.label()),
            xet_terms: None,
        });
        total += nelem;
    }

    // Unmatched / shape-mismatched tensors become crosshatched canvas regions.
    // In finetune mode only orig-only entries can survive (mod-side errors
    // were already raised above), and they render as informational grey.
    let orig_fill = if is_finetune { DiffFill::Grey } else { DiffFill::Red };

    let mut orig_unmatched: Vec<&safetensors::TensorMeta> = Vec::new();
    for name in &orig_only {
        orig_unmatched.push(&orig_map[name].1);
    }
    for (orig_full, _) in &shape_mismatch {
        orig_unmatched.push(&orig_map[orig_full].1);
    }
    for t in orig_unmatched {
        let nelem: u64 = t.shape.iter().product();
        if nelem == 0 { continue; }
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill: orig_fill },
            byte_size: nelem,
            safetensors: None,
            name_override: Some(format!("[only in original] {}", t.label())),
            xet_terms: None,
        });
        total += nelem;
    }

    // mod-only tensors render as green crosshatch in both modes. In finetune
    // mode the warning above already flagged the contract violation; the
    // green crosshatch surfaces it visually too.
    for name in &mod_only {
        let t = &mod_map[name].1;
        let nelem: u64 = t.shape.iter().product();
        if nelem == 0 { continue; }
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill: DiffFill::Green },
            byte_size: nelem,
            safetensors: None,
            name_override: Some(format!("[only in modified] {}", t.label())),
            xet_terms: None,
        });
        total += nelem;
    }

    if !orig_only.is_empty() || !mod_only.is_empty() || !shape_mismatch.is_empty() {
        log::info!(
            "safetensors diff: {} matched, {} only in original, {} only in modified, {} shape-mismatch",
            sources.iter().filter(|s| matches!(s.kind, SourceKind::TensorDiff { .. })).count(),
            orig_only.len(),
            mod_only.len(),
            shape_mismatch.len()
        );
    }

    Ok((sources, total))
}

/// Build per-tensor diff Sources from multiple local .safetensors files on each side.
async fn build_multi_safetensors_diff_sources(
    orig_files: &[PathBuf],
    mod_files: &[PathBuf],
    is_finetune: bool,
    metric: safetensors::DiffMetric,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let open_arcs = |files: &[PathBuf]| -> anyhow::Result<Vec<Arc<Data>>> {
        files.iter().map(|p| {
            let f = File::open(p).with_context(|| format!("opening {}", p.display()))?;
            Ok(Arc::new(Data::Mapped(unsafe { Mmap::map(&f) }?)))
        }).collect()
    };
    let orig_data = open_arcs(orig_files)?;
    let mod_data  = open_arcs(mod_files)?;
    build_multi_safetensors_diff_sources_inner(&orig_data, &mod_data, is_finetune, metric).await
}

/// Build per-tensor diff Sources from multiple remote .safetensors files on each side.
/// Headers are fetched via range requests; tensor data is never downloaded.
///
/// Each xet-backed file gets a `Data::Xet` reader (one V2 reconstruction fetch
/// per file, then direct-CAS range fetches afterward — see `XetReader`).
/// Non-xet remote files fall back to `Data::Http` which routes through hf-hub.
async fn build_multi_safetensors_diff_sources_from_http(
    orig_specs: &[&(String, RemoteFileSpec)],
    mod_specs:  &[&(String, RemoteFileSpec)],
    is_finetune: bool,
    metric: safetensors::DiffMetric,
) -> anyhow::Result<(Vec<Source>, u64)> {
    async fn make_arcs(specs: Vec<RemoteFileSpec>) -> anyhow::Result<Vec<Arc<Data>>> {
        let total = specs.len() as u64;
        let pb = setup_progress("source files (xet reconstruction for diff)", total);
        let pb_for_workers = pb.clone();
        let mut out: Vec<(usize, anyhow::Result<Arc<Data>>)> = stream::iter(specs.into_iter().enumerate())
            .map(|(i, spec)| {
                let pb = pb_for_workers.clone();
                async move {
                    let r: anyhow::Result<Arc<Data>> = if spec.xet_hash.is_some() {
                        match XetReader::new(&spec).await {
                            Ok(reader) => Ok(Arc::new(Data::Xet(reader))),
                            Err(e) => {
                                log::warn!(
                                    "{}: XetReader build failed ({e}); falling back to hf-hub Data::Http",
                                    spec.filename,
                                );
                                Ok(Arc::new(Data::Http {
                                    repo: spec.repo.clone(),
                                    filename: Arc::clone(&spec.filename),
                                    revision: Arc::clone(&spec.revision),
                                }))
                            }
                        }
                    } else {
                        Ok(Arc::new(Data::Http {
                            repo: spec.repo.clone(),
                            filename: Arc::clone(&spec.filename),
                            revision: Arc::clone(&spec.revision),
                        }))
                    };
                    if let Some(pb) = pb.as_ref() {
                        pb.inc(1);
                    }
                    (i, r)
                }
            })
            .buffer_unordered(SETUP_FETCH_CONCURRENCY)
            .collect()
            .await;
        if let Some(pb) = pb.as_ref() {
            pb.finish_and_clear();
        }
        out.sort_by_key(|(i, _)| *i);
        out.into_iter().map(|(_, r)| r).collect::<anyhow::Result<Vec<_>>>()
    }
    let orig_data = make_arcs(orig_specs.iter().map(|(_, s)| s.clone()).collect()).await?;
    let mod_data  = make_arcs(mod_specs.iter().map(|(_, s)| s.clone()).collect()).await?;
    build_multi_safetensors_diff_sources_inner(&orig_data, &mod_data, is_finetune, metric).await
}

/// Build per-tensor diff Sources from two single .safetensors files.
async fn build_safetensors_diff_sources(
    original: &Path,
    modified: &Path,
    is_finetune: bool,
    metric: safetensors::DiffMetric,
) -> anyhow::Result<(Vec<Source>, u64)> {
    build_multi_safetensors_diff_sources(
        &[original.to_path_buf()],
        &[modified.to_path_buf()],
        is_finetune,
        metric,
    ).await
}
