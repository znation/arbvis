use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::stream::{self, StreamExt};
use indicatif::ProgressBar;
use memmap2::Mmap;

use crate::hf_url::{RemoteFileSpec, RemoteRepo};
use crate::progress::{counter_style, multi};
use crate::xet::{self, XetReader, XetTerm};

/// Per-element delta encoding for `--diff` / `--moe-diff`. The CLI exposes
/// this via the `--diff-metric` flag. Tensor-aware backends interpret the
/// value; arbvis core just plumbs it through `prepare_diff_sources` and
/// the registered hooks.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum DiffMetric {
    #[default]
    Rms,
    AbsLog,
    Exact,
}

/// Crosshatch fill color for `UnmatchedRegion` / `OneSidedRange` sources —
/// the diff path uses these to mark one-side-only spans visually.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiffFill {
    Grey,
    Red,
    Green,
}

impl DiffFill {
    /// `(stripe, base)` colors for the crosshatch pattern. `stripe` is the
    /// foreground diagonal line color; `base` is the fill behind it.
    pub fn colors(self) -> (image::Rgb<u8>, image::Rgb<u8>) {
        match self {
            DiffFill::Grey => (image::Rgb([80, 80, 80]), image::Rgb([160, 160, 160])),
            DiffFill::Red => (image::Rgb([120, 0, 0]), image::Rgb([220, 40, 40])),
            DiffFill::Green => (image::Rgb([0, 120, 0]), image::Rgb([40, 220, 40])),
        }
    }
}

/// Async fetcher closure used by [`Data::LazyDiff`]. Captures its inputs by
/// `Arc` so the returned future is `'static` and can be sent across tasks.
/// `CustomSource::open` impls in `modelweightvis::data` build these for
/// per-tensor diff buffers.
pub type LazyFetcher =
    Arc<dyn Fn(u64, usize) -> BoxFuture<'static, anyhow::Result<Vec<u8>>> + Send + Sync>;

/// Bounded concurrency for setup-time HTTP loops (xet reconstruction,
/// safetensors header fetches, non-safetensors diff downloads). The global
/// AIMD throttle still caps the *actual* in-flight count; this just lets the
/// runtime have enough simultaneous awaiting tasks to keep the throttle full.
const SETUP_FETCH_CONCURRENCY: usize = 16;

/// Seconds of zero-byte silence before [`download_specs_to_paths`] aborts an
/// in-flight `hf-hub` `download_file` and treats it as a transient timeout.
/// hf-hub's `reqwest::Client` is built without `.timeout()` / `.read_timeout()`
/// and its streaming write loop (`stream_response_to_file_with_progress`) has
/// no per-chunk timeout, so a CDN edge that silently stops sending bytes hangs
/// the future forever. A `ProgressHandler` stamps the latest event time and a
/// sibling watchdog cancels the future after this many seconds of silence.
///
/// Override with `ARBVIS_DOWNLOAD_STALL_SECS` if 30 s is too aggressive (very
/// slow links may legitimately go 30 s between TCP windows under heavy
/// congestion). The throttle's `Outcome::Timeout` retry budget
/// (`MAX_TIMEOUT_RETRIES = 5`) gates total wall-time before we give up.
const DEFAULT_DOWNLOAD_STALL_SECS: u64 = 30;

/// Build a one-shot progress bar attached to the global `MultiProgress` so
/// it interleaves cleanly with log output. Always returns `Some(...)`; the
/// non-TTY case is handled by the hidden draw target on the global multi.
/// `Option<ProgressBar>` is kept in the signature so existing call sites that
/// pattern-match it continue to compile.
fn setup_progress(label: &str, total: u64) -> Option<ProgressBar> {
    let pb = multi()
        .add(ProgressBar::new(total))
        .with_style(counter_style())
        .with_message(label.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    Some(pb)
}

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
    /// A windowed view onto another `Data`. `fetch_range(s, n)` resolves to
    /// `inner.fetch_range(base + s, n)`. Used by JSON / JSONL structure-aware
    /// diff so each one-sided structural span can expose its underlying bytes
    /// without re-mmapping the file.
    OffsetSlice {
        inner: Arc<Data>,
        base: u64,
    },
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
            Data::OffsetSlice { inner, base } => &inner[*base as usize..],
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
            Data::Http {
                repo,
                filename,
                revision,
            } => {
                repo.fetch_range(filename, revision, start..start + len as u64)
                    .await
            }
            Data::Xet(reader) => reader.fetch_range(start, len).await,
            Data::LazyDiff(f) => f(start, len).await,
            Data::ZeroFill => Ok(vec![0u8; len]),
            Data::OffsetSlice { inner, base } => {
                let inner = Arc::clone(inner);
                let base = *base;
                Box::pin(async move { inner.fetch_range(base + start, len).await }).await
            }
        }
    }

    /// Whether `fetch_range` resolves without issuing an HTTP request.
    ///
    /// `Http`, `Xet`, and `LazyDiff` may all hit the network. The tile load
    /// stage uses this to skip the AIMD HTTP throttle when nothing in flight
    /// could hit the Hub — otherwise mmap reads would be artificially capped
    /// at the throttle's initial 4-way concurrency.
    pub fn is_local(&self) -> bool {
        match self {
            Data::Mapped(_) | Data::Owned(_) | Data::ZeroFill => true,
            Data::OffsetSlice { inner, .. } => inner.is_local(),
            Data::Http { .. } | Data::Xet(_) | Data::LazyDiff(_) => false,
        }
    }
}

/// A `Source` variant supplied by a downstream crate / plugin.
///
/// Today the only impl is `TensorDiffSource` (per-tensor diff buffer, was
/// `SourceKind::TensorDiff`). When `modelweightvis` splits out it'll bring
/// its tensor-diff impls along; the arbvis core just dispatches by trait.
pub trait CustomSource: Send + Sync {
    /// Stable identifier for diagnostic logs and runtime predicates (e.g.
    /// "is this a tensor-diff source?"). Format: kebab-case.
    fn id(&self) -> &'static str;
    /// Byte size of the synthetic stream this source exposes. Drives canvas
    /// layout (Hilbert + arch both read it).
    #[allow(dead_code)]
    fn byte_size(&self) -> u64;
    /// Open the source for the render pipeline. Returns a `Data` handle the
    /// load stage can `fetch_range` against.
    fn open(&self) -> anyhow::Result<Data>;
}

/// How a source's bytes are stored.
#[non_exhaustive]
pub enum SourceKind {
    Buffered(Vec<u8>),
    File(PathBuf),
    Diff {
        original: PathBuf,
        modified: PathBuf,
    },
    /// Remote HF file, accessed via hf-hub range requests per tile.
    Http(RemoteFileSpec),
    /// A canvas region for a tensor / file that exists on only one side of a
    /// diff. The byte_size on the parent `Source` controls how much canvas
    /// space it takes; the underlying bytes are zero (the renderer paints a
    /// crosshatch pattern based on `fill` instead of using the byte LUT).
    UnmatchedRegion {
        fill: DiffFill,
    },
    /// Byte-for-byte signed diff over a sub-range of two whole-file Data
    /// sources. Identical rendering semantics to `SourceKind::Diff` (signed
    /// byte delta) but parameterised over (Arc<Data>, start_offset) so each
    /// structurally-aligned span emitted by the JSON / JSONL aligner becomes
    /// one Source over its byte range.
    RangeDiff {
        orig: Arc<Data>,
        mod_: Arc<Data>,
        orig_start: u64,
        mod_start: u64,
    },
    /// Bytes from a single side of the diff (insertion or deletion). The
    /// renderer paints the real bytes through the plain byte LUT and then
    /// blends `fill` over the top so the side of origin is clear while the
    /// content remains legible.
    OneSidedRange {
        data: Arc<Data>,
        start: u64,
        fill: DiffFill,
    },
    /// Source supplied by a [`CustomSource`] impl. The arbvis pipeline only
    /// touches its `open` / `byte_size` / `id`; everything else is up to the
    /// impl. Today this carries `TensorDiffSource` for per-tensor `--diff` /
    /// `--moe-diff` runs.
    Custom(Box<dyn CustomSource>),
}

pub enum InputSpec {
    Local(PathBuf),
    Remote(RemoteFileSpec),
}

/// Typed extension map for [`Source`]. Holds at most one value per
/// concrete type. Format/layout plugins use this to attach typed metadata
/// (e.g. `ModelInfo` from a `FormatPlugin`, `MoeCell` from MoE-diff prep)
/// without bolting extra fields onto `Source`.
#[derive(Default)]
pub struct Extensions {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Extensions {
    /// Insert a value keyed by its type. Replaces any prior value of the same
    /// type.
    pub fn insert<T: Any + Send + Sync>(&mut self, value: T) {
        self.map.insert(TypeId::of::<T>(), Box::new(value));
    }

    /// Lookup the value associated with `T`, if any. Plugin readers (step 8
    /// onwards) call this to fetch the typed metadata they care about.
    #[allow(dead_code)]
    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|v| v.downcast_ref::<T>())
    }
}

impl std::fmt::Debug for Extensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Don't try to render the boxed payloads — `dyn Any` doesn't expose a
        // useful representation. Surface the count for diagnostic dumps.
        f.debug_struct("Extensions")
            .field("type_count", &self.map.len())
            .finish()
    }
}

/// Metadata and storage descriptor for one input.
pub struct Source {
    pub file_idx: usize,
    pub kind: SourceKind,
    pub byte_size: u64,
    /// Override the display name (used when kind is Buffered but has a real filename).
    pub name_override: Option<String>,
    /// Xet reconstruction terms for this source. `Some(vec)` when xet
    /// visualization was requested and the source has a xet hash; `None`
    /// when xet visualization is off; `Some(vec![])` when xet vis is on but
    /// this source isn't xet-backed.
    pub xet_terms: Option<Vec<XetTerm>>,
    /// Typed per-source metadata that format and layout plugins consume.
    /// Today this carries `ModelInfo` (safetensors header parse) and
    /// `MoeCell` (the MoE-diff layout's per-cell tag); future format
    /// plugins in `modelweightvis` will stuff their own per-source data
    /// here.
    pub extensions: Extensions,
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
            SourceKind::Custom(cs) => {
                unreachable!(
                    "Custom source `{}` reached Source::name without a name_override",
                    cs.id()
                )
            }
            SourceKind::UnmatchedRegion { .. } => {
                unreachable!("UnmatchedRegion sources always have name_override set")
            }
            SourceKind::RangeDiff { .. } => {
                unreachable!("RangeDiff sources always have name_override set")
            }
            SourceKind::OneSidedRange { .. } => {
                unreachable!("OneSidedRange sources always have name_override set")
            }
        }
    }
}

/// Build sources and return total byte count.
///
/// Files are opened lazily (one at a time) to avoid exhausting OS fd limits.
/// Stdin is buffered into memory upfront since its size is unknown.
///
/// For .safetensors files: the header is parsed and attached as ModelInfo
/// for dtype coloring. The file is kept as a single Source (one per file) so that
/// inter-tensor borders are not drawn and the Hilbert curve flows smoothly across
/// the whole file with color transitions only at tensor boundaries.
pub fn prepare_sources(
    files: &[PathBuf],
    registry: &crate::registry::Registry,
) -> anyhow::Result<(Vec<Source>, u64)> {
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
                name_override: None,
                xet_terms: None,
                extensions: Extensions::default(),
            }],
            len,
        ));
    }

    // Expand any directory paths (e.g. from a repo-level hf:// download) into
    // their constituent files so they can be treated as individual sources.
    let expanded: Vec<PathBuf> = files
        .iter()
        .flat_map(|p| {
            if p.is_dir() {
                collect_files_recursive(p)
            } else {
                vec![p.clone()]
            }
        })
        .collect();

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

        // Ask each registered `FormatPlugin` whether it recognizes this
        // path; the first that does gets to populate the source's
        // typed-extensions map (e.g. with `ModelInfo`). arbvis itself
        // knows nothing format-specific.
        let mut extensions = Extensions::default();
        for plugin in &registry.formats {
            if plugin.detects_path(path) {
                if let Err(e) = plugin.populate_local(path, size, &mut extensions) {
                    log::warn!(
                        "{}: format plugin `{}` failed: {e} — treating as plain binary",
                        path.display(),
                        plugin.id()
                    );
                }
                break;
            }
        }

        total += size;
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::File(path.clone()),
            byte_size: size,
            name_override: None,
            xet_terms: None,
            extensions,
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
                    // Zero-pad reads beyond either side's length so that
                    // same-name files with different sizes can share one diff
                    // source. The longer side's tail diffs against zero.
                    let read_padded = |m: &Mmap| -> Vec<u8> {
                        let s = start as usize;
                        let mlen = m.len();
                        let mut buf = vec![0u8; len];
                        if s < mlen {
                            let take = (mlen - s).min(len);
                            buf[..take].copy_from_slice(&m[s..s + take]);
                        }
                        buf
                    };
                    let a = read_padded(&m_o);
                    let b = read_padded(&m_m);
                    Ok(a.iter()
                        .zip(b.iter())
                        .map(|(&a, &b)| {
                            let delta = b as i16 - a as i16;
                            let brightness =
                                (delta.unsigned_abs() as f32 / 255.0 * 127.0).round() as u8;
                            if delta >= 0 {
                                127u8 + brightness
                            } else {
                                127u8 - brightness
                            }
                        })
                        .collect())
                })
            })))
        }
        SourceKind::Http(spec) => Ok(Data::Http {
            repo: spec.repo.clone(),
            filename: Arc::clone(&spec.filename),
            revision: Arc::clone(&spec.revision),
        }),
        SourceKind::UnmatchedRegion { .. } => Ok(Data::ZeroFill),
        SourceKind::RangeDiff {
            orig,
            mod_,
            orig_start,
            mod_start,
        } => {
            let orig = Arc::clone(orig);
            let mod_ = Arc::clone(mod_);
            let orig_start = *orig_start;
            let mod_start = *mod_start;
            Ok(Data::LazyDiff(Arc::new(move |start: u64, len: usize| {
                let orig = Arc::clone(&orig);
                let mod_ = Arc::clone(&mod_);
                Box::pin(async move {
                    let a = orig.fetch_range(orig_start + start, len).await?;
                    let b = mod_.fetch_range(mod_start + start, len).await?;
                    Ok(a.iter()
                        .zip(b.iter())
                        .map(|(&a, &b)| {
                            let delta = b as i16 - a as i16;
                            let brightness =
                                (delta.unsigned_abs() as f32 / 255.0 * 127.0).round() as u8;
                            if delta >= 0 {
                                127u8 + brightness
                            } else {
                                127u8 - brightness
                            }
                        })
                        .collect())
                })
            })))
        }
        SourceKind::OneSidedRange { data, start, .. } => Ok(Data::OffsetSlice {
            inner: Arc::clone(data),
            base: *start,
        }),
        SourceKind::Custom(cs) => cs.open(),
    }
}

/// Build sources from a mixed list of local paths and remote HF file specs.
/// Remote specs are turned into `SourceKind::Http` entries (no download).
///
/// Async so the remote arm can call each [`crate::registry::FormatPlugin`]'s
/// `populate_remote` over a `Data::Http` handle — that's the only way
/// `ModelInfo` gets stuffed into `Source.extensions` when `--stream` /
/// `--show-xet-xorbs` keep the file remote. Without that population, the
/// arch layout's `applicable()` check returns false on every remote source
/// and arbvis silently falls back to byte-Hilbert.
pub async fn prepare_sources_from_specs(
    specs: &[InputSpec],
    registry: &crate::registry::Registry,
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
                name_override: None,
                xet_terms: None,
                extensions: Extensions::default(),
            }],
            len,
        ));
    }

    // Pre-expand `InputSpec::Local(dir)` into one `InputSpec::Local(file)` per
    // contained file, matching the behaviour of [`prepare_sources`]. Without
    // this, a local directory pushed into the specs (e.g. by `--stream
    // ./snapshots/llama-7b/`) would yield a single Source pointing at the
    // directory itself — load_source_data then mmaps the dir and fails.
    let expanded: Vec<InputSpec> = specs
        .iter()
        .flat_map(|spec| match spec {
            InputSpec::Local(p) if p.is_dir() => collect_files_recursive(p)
                .into_iter()
                .map(InputSpec::Local)
                .collect::<Vec<_>>(),
            InputSpec::Local(p) => vec![InputSpec::Local(p.clone())],
            InputSpec::Remote(s) => vec![InputSpec::Remote(s.clone())],
        })
        .collect();

    let mut sources = Vec::new();
    let mut total = 0u64;

    for spec in &expanded {
        match spec {
            InputSpec::Local(path) => {
                let size = match std::fs::metadata(path) {
                    Ok(m) => m.len(),
                    Err(e) => {
                        log::warn!("{}: {} — skipping", path.display(), e);
                        continue;
                    }
                };
                let mut extensions = Extensions::default();
                for plugin in &registry.formats {
                    if plugin.detects_path(path) {
                        if let Err(e) = plugin.populate_local(path, size, &mut extensions) {
                            log::warn!(
                                "{}: format plugin `{}` failed: {e} — treating as plain binary",
                                path.display(),
                                plugin.id()
                            );
                        }
                        break;
                    }
                }
                total += size;
                sources.push(Source {
                    file_idx: sources.len(),
                    kind: SourceKind::File(path.clone()),
                    byte_size: size,
                    name_override: None,
                    xet_terms: None,
                    extensions,
                });
            }
            InputSpec::Remote(spec) => {
                let size = spec.size;
                total += size;
                // Open a `Data::Http` handle pointing at the same remote file
                // the downstream loader/renderer will use. This costs nothing
                // up front — `Data::Http` is just (repo, filename, revision);
                // `populate_remote` is what issues the actual head-prefix
                // range fetch needed to parse the format header.
                let data = Data::Http {
                    repo: spec.repo.clone(),
                    filename: Arc::clone(&spec.filename),
                    revision: Arc::clone(&spec.revision),
                };
                let mut extensions = Extensions::default();
                let filename_path = Path::new(spec.filename.as_str());
                for plugin in &registry.formats {
                    if plugin.detects_path(filename_path) {
                        if let Err(e) = plugin.populate_remote(&data, size, &mut extensions).await {
                            // Non-fatal: arch layout falls back to byte-Hilbert,
                            // the same way it would for any source whose format
                            // plugin couldn't parse its header.
                            log::warn!(
                                "{}: format plugin `{}` (remote) failed: {e} — \
                                 treating as plain binary",
                                spec.filename,
                                plugin.id()
                            );
                        }
                        break;
                    }
                }
                sources.push(Source {
                    file_idx: sources.len(),
                    kind: SourceKind::Http(spec.clone()),
                    byte_size: size,
                    name_override: None,
                    xet_terms: None,
                    extensions,
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

    let indices: Vec<usize> = jobs.iter().map(|(i, _)| *i).collect();
    let specs: Vec<RemoteFileSpec> = jobs.into_iter().map(|(_, s)| s).collect();
    let paths = download_specs_to_paths(&specs, "source files (downloading for xet view)").await?;

    for (i, path) in indices.into_iter().zip(paths) {
        // Preserve display name + xet_terms; only the storage kind changes.
        let display = sources[i].name();
        sources[i].kind = SourceKind::File(path);
        if sources[i].name_override.is_none() {
            sources[i].name_override = Some(display);
        }
    }
    Ok(())
}

/// `ProgressHandler` that records the wall-clock time of the most recent
/// hf-hub progress event into a shared atomic. Paired with the watchdog future
/// inside [`download_specs_to_paths`] to detect silent body-stream stalls.
#[derive(Clone)]
struct StallSentinel {
    last_event_unix_ms: Arc<std::sync::atomic::AtomicI64>,
}

impl hf_hub::progress::ProgressHandler for StallSentinel {
    fn on_progress(&self, _event: &hf_hub::progress::ProgressEvent) {
        self.last_event_unix_ms
            .store(unix_now_ms(), std::sync::atomic::Ordering::Relaxed);
    }
}

fn unix_now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Per-attempt error from the closure passed to `with_throttle` inside
/// [`download_specs_to_paths`]. Wraps `hf-hub`'s `HFError` and adds a `Stalled`
/// variant that classifies as `Outcome::Timeout`, so the AIMD throttle's
/// existing retry path (`MAX_TIMEOUT_RETRIES`, backoff, scale-down) handles
/// connection stalls without any special-case logic above.
#[derive(Debug)]
enum DownloadAttemptError {
    Hf(hf_hub::HFError),
    Stalled { filename: String, idle_secs: u64 },
}

impl std::fmt::Display for DownloadAttemptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DownloadAttemptError::Hf(e) => write!(f, "{e}"),
            DownloadAttemptError::Stalled {
                filename,
                idle_secs,
            } => write!(
                f,
                "{filename}: no progress for {idle_secs}s — treating as transient timeout",
            ),
        }
    }
}

impl std::error::Error for DownloadAttemptError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DownloadAttemptError::Hf(e) => Some(e),
            DownloadAttemptError::Stalled { .. } => None,
        }
    }
}

impl crate::throttle::ErrorClassify for DownloadAttemptError {
    fn classify(&self) -> crate::throttle::Outcome {
        match self {
            DownloadAttemptError::Hf(e) => e.classify(),
            DownloadAttemptError::Stalled { .. } => crate::throttle::Outcome::Timeout,
        }
    }
}

/// Download a batch of [`RemoteFileSpec`]s to the local hf-hub cache and return
/// the local paths in the same order. Drives the AIMD throttle through
/// [`crate::throttle::with_throttle`] and reports progress via a one-shot
/// `setup_progress` bar.
///
/// Each attempt is wrapped in a stall watchdog: a [`StallSentinel`] passed as
/// the hf-hub progress handler stamps every event, and a sibling future
/// cancels the download if no event arrives for `DEFAULT_DOWNLOAD_STALL_SECS`
/// (`ARBVIS_DOWNLOAD_STALL_SECS` env override). Stalls surface as
/// [`DownloadAttemptError::Stalled`], which classifies as `Outcome::Timeout`
/// and goes through the throttle's normal retry/backoff path.
///
/// Shared by every disk-backed materialisation path:
/// [`materialize_http_sources`] (normal flow's `SourceKind::Http` swap) and
/// [`materialize_remote_arcs`] (the `Arc<Data>`s buried inside
/// `SourceKind::TensorDiff` for `--diff`/`--moe-diff`).
pub async fn download_specs_to_paths(
    specs: &[RemoteFileSpec],
    progress_label: &str,
) -> anyhow::Result<Vec<PathBuf>> {
    use crate::throttle::with_throttle;
    use std::sync::atomic::{AtomicI64, Ordering};

    let pb = setup_progress(progress_label, specs.len() as u64);
    let pb_for_workers = pb.clone();

    let stall_secs: u64 = std::env::var("ARBVIS_DOWNLOAD_STALL_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_DOWNLOAD_STALL_SECS);
    let stall_ms: i64 = (stall_secs as i64).saturating_mul(1000);

    let mut downloads: Vec<(usize, anyhow::Result<PathBuf>)> =
        stream::iter(specs.iter().cloned().enumerate())
            .map(|(i, spec)| {
                let pb = pb_for_workers.clone();
                async move {
                    let filename = (*spec.filename).clone();
                    let revision = (*spec.revision).clone();
                    let label = format!("download_file {filename}");
                    let result = with_throttle(&label, || {
                        let spec = spec.clone();
                        let filename = filename.clone();
                        let revision = revision.clone();
                        async move {
                            let last_event = Arc::new(AtomicI64::new(unix_now_ms()));
                            let sentinel = StallSentinel {
                                last_event_unix_ms: Arc::clone(&last_event),
                            };
                            let download_fut = async {
                                match &spec.repo {
                                    RemoteRepo::Model(r) => {
                                        r.download_file()
                                            .filename(filename.clone())
                                            .revision(revision.clone())
                                            .progress(sentinel)
                                            .send()
                                            .await
                                    }
                                    RemoteRepo::Dataset(r) => {
                                        r.download_file()
                                            .filename(filename.clone())
                                            .revision(revision.clone())
                                            .progress(sentinel)
                                            .send()
                                            .await
                                    }
                                    RemoteRepo::Space(r) => {
                                        r.download_file()
                                            .filename(filename.clone())
                                            .revision(revision.clone())
                                            .progress(sentinel)
                                            .send()
                                            .await
                                    }
                                }
                            };
                            let watchdog = async {
                                loop {
                                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                                    let idle_ms = unix_now_ms()
                                        .saturating_sub(last_event.load(Ordering::Relaxed));
                                    if idle_ms >= stall_ms {
                                        return idle_ms;
                                    }
                                }
                            };
                            tokio::select! {
                                r = download_fut => r.map_err(DownloadAttemptError::Hf),
                                idle_ms = watchdog => {
                                    let idle_secs = (idle_ms / 1000).max(0) as u64;
                                    log::warn!(
                                        "{filename}: download stalled (no bytes for {idle_secs}s); aborting attempt and retrying via throttle"
                                    );
                                    Err(DownloadAttemptError::Stalled {
                                        filename: filename.clone(),
                                        idle_secs,
                                    })
                                }
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

    downloads.sort_by_key(|(i, _)| *i);
    downloads
        .into_iter()
        .map(|(_, r)| r)
        .collect::<anyhow::Result<Vec<_>>>()
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

// ---------------------------------------------------------------------------
// Diff source builders
//
// Three built-in `DiffSourceBuilder` impls cover the file-pair diff cases.
// Directory-pair diffs stay inline in `prepare_diff_sources` for now —
// they'll move behind the trait when format detection migrates to
// `modelweightvis` (step 12).
// ---------------------------------------------------------------------------

fn is_json_path(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()),
        Some("json") | Some("jsonl")
    )
}

/// JSON / JSONL structure-aware diff. Applies when both paths have a
/// `.json` or `.jsonl` extension.
pub struct JsonDiffBuilder;

#[async_trait::async_trait]
impl crate::registry::DiffSourceBuilder for JsonDiffBuilder {
    fn id(&self) -> &'static str {
        "json"
    }
    fn priority(&self) -> i32 {
        200
    }
    async fn try_build(
        &self,
        ctx: &crate::registry::DiffBuildCtx<'_>,
    ) -> anyhow::Result<Option<(Vec<Source>, u64)>> {
        if !(is_json_path(ctx.original) && is_json_path(ctx.modified)) {
            return Ok(None);
        }
        let out =
            crate::json_diff::build_json_diff_sources(ctx.original, ctx.modified, ctx.is_finetune)
                .await?;
        Ok(Some(out))
    }
}

// `TensorDiffBuilder` lives in `modelweightvis::diff` (step 12e). The
// arbvis default registry no longer wires it up.

/// Plain-byte diff: builds one `SourceKind::Diff` source over a same-sized
/// pair. The floor of the builder priority stack — applies whenever the two
/// files exist and have the same size, and bails with an error if sizes
/// differ (matching the original `prepare_diff_sources` contract).
pub struct PlainBytesDiffBuilder;

#[async_trait::async_trait]
impl crate::registry::DiffSourceBuilder for PlainBytesDiffBuilder {
    fn id(&self) -> &'static str {
        "plain-bytes"
    }
    fn priority(&self) -> i32 {
        0
    }
    async fn try_build(
        &self,
        ctx: &crate::registry::DiffBuildCtx<'_>,
    ) -> anyhow::Result<Option<(Vec<Source>, u64)>> {
        let size_o = std::fs::metadata(ctx.original)?.len();
        let size_m = std::fs::metadata(ctx.modified)?.len();
        if size_o != size_m {
            anyhow::bail!(
                "--diff: file sizes differ ({} bytes vs {} bytes): {} vs {}",
                size_o,
                size_m,
                ctx.original.display(),
                ctx.modified.display()
            );
        }
        let source = Source {
            file_idx: 0,
            kind: SourceKind::Diff {
                original: ctx.original.to_path_buf(),
                modified: ctx.modified.to_path_buf(),
            },
            byte_size: size_o,
            name_override: None,
            xet_terms: None,
            extensions: Extensions::default(),
        };
        Ok(Some((vec![source], size_o)))
    }
}

/// Build diff sources from two files or two directories.
///
/// For files: dispatched through `registry.diffs` by descending priority —
/// `TensorDiffBuilder` (300) → `JsonDiffBuilder` (200) → `PlainBytesDiffBuilder`
/// (0). The plain-byte floor always builds for a same-sized pair, so
/// iteration terminates with a valid diff for any well-formed input.
///
/// For directories: files are matched by relative path; pairs with mismatched
/// sizes or no counterpart on the other side are skipped with a warning. The
/// safetensors sub-tree is matched by tensor name across both sides. This
/// branch is still inline pending step 12.
pub async fn prepare_diff_sources(
    original: &Path,
    modified: &Path,
    is_finetune: bool,
    metric: DiffMetric,
    registry: &crate::registry::Registry,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let orig_is_file = original.is_file();
    let mod_is_file = modified.is_file();
    let orig_is_dir = original.is_dir();
    let mod_is_dir = modified.is_dir();

    // Whether a file should be diffed via the tensor-aware (modelweightvis-
    // registered) path. arbvis itself doesn't know which extensions are
    // tensor formats — the `DirectoryTensorDiffPrep` hook says. When no
    // hook is registered, the directory branch falls through to pure
    // byte-diff for every file (no tensor matching).
    let is_tensor = |p: &Path| -> bool {
        registry
            .dir_tensor_diff
            .as_ref()
            .is_some_and(|h| h.is_tensor_file(p))
    };

    if orig_is_file && mod_is_file {
        let ctx = crate::registry::DiffBuildCtx {
            original,
            modified,
            is_finetune,
            metric,
        };
        let mut sorted: Vec<&Arc<dyn crate::registry::DiffSourceBuilder>> =
            registry.diffs.iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.priority()));
        for builder in &sorted {
            if let Some(out) = builder.try_build(&ctx).await? {
                return Ok(out);
            }
        }
        anyhow::bail!(
            "--diff: no registered builder handled the input pair ({} vs {})",
            original.display(),
            modified.display()
        );
    }

    if orig_is_dir && mod_is_dir {
        let orig_files = collect_files_recursive(original);
        let mod_files = collect_files_recursive(modified);

        let mut sources = Vec::new();
        let mut total = 0u64;

        // Tensor files: hand off to the registered `DirectoryTensorDiffPrep`
        // hook (modelweightvis-side). This handles sharded model layouts,
        // tensor-name matching across files, etc. arbvis core only knows
        // how to byte-diff files matched by relative path.
        let orig_tensor: Vec<PathBuf> = orig_files
            .iter()
            .filter(|p| is_tensor(p))
            .cloned()
            .collect();
        let mod_tensor: Vec<PathBuf> = mod_files.iter().filter(|p| is_tensor(p)).cloned().collect();
        if !orig_tensor.is_empty() || !mod_tensor.is_empty() {
            if let Some(hook) = registry.dir_tensor_diff.as_ref() {
                match hook
                    .prepare(&orig_tensor, &mod_tensor, is_finetune, metric)
                    .await
                {
                    Ok((mut tensor_sources, bytes)) => {
                        let base_idx = sources.len();
                        for s in &mut tensor_sources {
                            s.file_idx += base_idx;
                        }
                        sources.extend(tensor_sources);
                        total += bytes;
                    }
                    Err(e) => log::warn!("tensor-aware directory diff failed: {e} — skipping"),
                }
            }
            // If no hook, the tensor files fall through to the byte-diff
            // path below (matched by relative path like any other file).
        }

        // Non-tensor files (or all files when no `DirectoryTensorDiffPrep`
        // is registered): match by relative path. Same-size pairs become a
        // byte diff; different-size or single-side files become crosshatched
        // unmatched regions so they remain visible.
        let orig_fill_kind = if is_finetune {
            DiffFill::Grey
        } else {
            DiffFill::Red
        };
        let orig_map: HashMap<PathBuf, PathBuf> = orig_files
            .iter()
            .filter(|p| !is_tensor(p))
            .filter_map(|p| {
                p.strip_prefix(original)
                    .ok()
                    .map(|rel| (rel.to_path_buf(), p.clone()))
            })
            .collect();
        let mod_map: HashMap<PathBuf, PathBuf> = mod_files
            .iter()
            .filter(|p| !is_tensor(p))
            .filter_map(|p| {
                p.strip_prefix(modified)
                    .ok()
                    .map(|rel| (rel.to_path_buf(), p.clone()))
            })
            .collect();

        let mut mod_only_keys: Vec<&PathBuf> = mod_map
            .keys()
            .filter(|k| !orig_map.contains_key(*k))
            .collect();
        mod_only_keys.sort();
        if is_finetune && !mod_only_keys.is_empty() {
            let names: Vec<String> = mod_only_keys
                .iter()
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
                    if size_o == 0 {
                        continue;
                    }
                    sources.push(Source {
                        file_idx: sources.len(),
                        kind: SourceKind::UnmatchedRegion {
                            fill: orig_fill_kind,
                        },
                        byte_size: size_o,
                        name_override: Some(format!("[only in original] {}", rel.display())),
                        xet_terms: None,
                        extensions: Extensions::default(),
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
                        if is_finetune {
                            log::warn!(
                                "--diff --finetune: size mismatch ({} vs {} bytes) for {} — \
                                 byte-diffing with zero-padding on the shorter side",
                                size_o,
                                size_m,
                                rel.display()
                            );
                        } else {
                            log::warn!(
                                "size mismatch ({} vs {} bytes) for {} — byte-diffing with zero-padding",
                                size_o, size_m, rel.display()
                            );
                        }
                    }
                    let max_size = size_o.max(size_m);
                    if max_size == 0 {
                        continue;
                    }
                    sources.push(Source {
                        file_idx: sources.len(),
                        kind: SourceKind::Diff {
                            original: orig_abs.clone(),
                            modified: mod_abs.clone(),
                        },
                        byte_size: max_size,
                        name_override: None,
                        xet_terms: None,
                        extensions: Extensions::default(),
                    });
                    total += max_size;
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
            if size_m == 0 {
                continue;
            }
            sources.push(Source {
                file_idx: sources.len(),
                kind: SourceKind::UnmatchedRegion {
                    fill: DiffFill::Green,
                },
                byte_size: size_m,
                name_override: Some(format!("[only in modified] {}", rel.display())),
                xet_terms: None,
                extensions: Extensions::default(),
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
        if orig_is_file {
            "file"
        } else if orig_is_dir {
            "directory"
        } else {
            "missing path"
        },
        if mod_is_file {
            "file"
        } else if mod_is_dir {
            "directory"
        } else {
            "missing path"
        }
    );
}
