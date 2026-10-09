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

/// Crosshatch fill color for `UnmatchedRegion` / `OneSidedRange` sources —
/// the diff path uses these to mark one-side-only spans visually.
mod diff;

pub use diff::{
    byte_directory_diff, prepare_diff_sources, DiffFill, JsonDiffBuilder, PlainBytesDiffBuilder,
};

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

pub(super) fn collect_files_recursive(root: &Path) -> Vec<PathBuf> {
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

/// The backing storage for a file's bytes: a local memory map, an owned
/// buffer, or one of the remote readers (HTTP range requests against the
/// Hub, or the direct xet CAS decoder).
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
    /// and decompresses the resulting chunk segments locally. The `hf` CLI
    /// has no byte-range surface, so per-tile xet dedup wire-speedup goes
    /// through this direct decoder. Decoded chunk segments are cached
    /// inside the reader for spatial-locality wins (adjacent tiles on the
    /// Hilbert curve share terms).
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
    /// Remote HF file, accessed via direct HTTPS range requests per tile.
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
    /// impl. Today this carries `TensorDiffSource` for per-tensor `--diff`
    /// runs.
    Custom(Box<dyn CustomSource>),
}

/// One input to a render: a local filesystem path (expanded to the files
/// under it when it names a directory) or a remote Hub file spec.
pub enum InputSpec {
    Local(PathBuf),
    Remote(RemoteFileSpec),
}

/// Typed extension map for [`Source`]. Holds at most one value per
/// concrete type. Format/layout plugins use this to attach typed metadata
/// (e.g. `ModelInfo` from a `FormatPlugin`, or the per-panel tags the MoE
/// summary / CKA preps attach) without bolting extra fields onto `Source`.
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

    /// Lookup the value associated with `T`, if any. Plugin readers call this
    /// to fetch the typed metadata they care about.
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

/// Assigns a [`Source`] to a named *scene*. Sources sharing a `key` are
/// rendered into their own independent tile pyramid under `tiles/<key>/…`,
/// and the Leaflet viewer gets a base-layer switcher ("tabs") to toggle
/// between scenes. Sources with no `SceneTag` form a single implicit default
/// scene rendered to the legacy `tiles/…` path with single-layer HTML — so
/// every non-scene render (hilbert / arch / diff) is byte-for-byte unchanged.
///
/// Attached via [`Extensions::insert`] by a producer that wants more than one
/// lens in a single run (e.g. `modelweightvis --moe` emits a `"summary"` and a
/// `"cka"` scene). The tiler partitions on this tag *before* layout selection,
/// so each scene independently picks its own [`crate::layout::LayoutShape`].
#[derive(Clone, Debug)]
pub struct SceneTag {
    /// Path-safe slug used as the on-disk / repo subdirectory (`tiles/<key>/`)
    /// and the scene's stable identity in `labels.json`.
    pub key: String,
    /// Human-readable label shown on the viewer's tab / layer switcher.
    pub label: String,
    /// Tab ordering; the lowest-`order` scene is the default-active layer.
    pub order: u32,
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
    /// Today this carries `ModelInfo` (safetensors header parse) and the
    /// per-panel MoE summary / CKA tags; future format plugins in
    /// `modelweightvis` will stuff their own per-source data here.
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
    if sources.is_empty() {
        // Every explicitly named input failed to stat (typically a typo'd
        // path, each already logged as a warning above). Fail fast instead of
        // rendering a silent empty viewer. An empty *input list* is stdin and
        // handled above; an empty *result* from a non-empty list is an error.
        anyhow::bail!(
            "no readable input files: {} (see the skip warnings above)",
            files
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok((sources, total))
}

/// Map a byte-wise diff to the viewer's signed-delta color encoding: the
/// neutral 127 gray shifts brighter when `b > a` and darker when `b < a`,
/// scaled by the magnitude of the delta. Both sides must be the same length
/// (callers zero-pad shorter sides first).
fn diff_bytes_to_color(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter()
        .zip(b.iter())
        .map(|(&a, &b)| {
            let delta = b as i16 - a as i16;
            let brightness = (delta.unsigned_abs() as f32 / 255.0 * 127.0).round() as u8;
            if delta >= 0 {
                127u8 + brightness
            } else {
                127u8 - brightness
            }
        })
        .collect()
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
                    Ok(diff_bytes_to_color(&a, &b))
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
                    Ok(diff_bytes_to_color(&a, &b))
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
                                crate::hf_url::sanitize_log_text(spec.filename.as_str()),
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
/// Why: `Data::Http::fetch_range` issues a fresh HTTPS GET (with Range) per
/// tile. With tens of thousands of tiles per file, the per-call TLS + HTTP
/// setup dominates and the pipeline appears stalled.
///
/// One whole-file `hf download` per source amortises that overhead across
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

/// Download a batch of [`RemoteFileSpec`]s to the local HF cache and return
/// the local paths in the same order. Drives the AIMD throttle through
/// [`crate::throttle::with_throttle`] and reports progress via a one-shot
/// `setup_progress` bar.
///
/// Each download is a fresh `hf download <repo> <file>` subprocess. The CLI
/// applies its own `HF_HUB_DOWNLOAD_TIMEOUT` (default 10 s) to detect CDN
/// stalls, so we no longer need a custom watchdog around the call. Errors
/// classify via `HfCliError::ErrorClassify`, which feeds the throttle's
/// existing retry/backoff path.
///
/// Shared by every disk-backed materialisation path:
/// [`materialize_http_sources`] (normal flow's `SourceKind::Http` swap) and
/// [`materialize_remote_arcs`] (the `Arc<Data>`s buried inside
/// `SourceKind::TensorDiff` for `--diff`).
pub async fn download_specs_to_paths(
    specs: &[RemoteFileSpec],
    progress_label: &str,
) -> anyhow::Result<Vec<PathBuf>> {
    use crate::hf_cli;
    use crate::throttle::with_throttle;

    let pb = setup_progress(progress_label, specs.len() as u64);
    let pb_for_workers = pb.clone();

    let mut downloads: Vec<(usize, anyhow::Result<PathBuf>)> =
        stream::iter(specs.iter().cloned().enumerate())
            .map(|(i, spec)| {
                let pb = pb_for_workers.clone();
                async move {
                    let filename = (*spec.filename).clone();
                    let revision = (*spec.revision).clone();
                    let label = format!(
                        "hf download {}",
                        crate::hf_url::sanitize_log_text(&filename),
                    );
                    let repo_id = spec.repo.repo_id().to_string();
                    // Map the api_segment ("models"/"datasets"/"spaces") to the
                    // `--type` flag value the CLI expects.
                    let repo_type = match spec.repo.api_segment() {
                        "models" => "model",
                        "datasets" => "dataset",
                        "spaces" => "space",
                        other => {
                            return (
                                i,
                                Err(anyhow::anyhow!(
                                    "unexpected api_segment {other:?} on remote repo"
                                )),
                            );
                        }
                    };
                    let result: Result<PathBuf, anyhow::Error> = with_throttle(&label, || async {
                        hf_cli::download([
                            "download",
                            "--type",
                            repo_type,
                            "--revision",
                            revision.as_str(),
                            repo_id.as_str(),
                            filename.as_str(),
                        ])
                        .await
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

#[cfg(test)]
mod diff_bytes_to_color_tests {
    use super::diff_bytes_to_color;

    #[test]
    fn equal_bytes_stay_neutral_and_deltas_shift_signed() {
        // Identical bytes map to the neutral 127 gray.
        assert_eq!(diff_bytes_to_color(&[5, 200], &[5, 200]), [127, 127]);
        // Positive delta brightens, negative delta darkens, scaled by
        // magnitude: a full ±255 swing lands exactly 127±127.
        assert_eq!(diff_bytes_to_color(&[0], &[255]), [254]);
        assert_eq!(diff_bytes_to_color(&[255], &[0]), [0]);
        // Mid-magnitude delta: 128/255 of the way from 127 toward the pole.
        assert_eq!(diff_bytes_to_color(&[0], &[128]), [191]);
        assert_eq!(diff_bytes_to_color(&[128], &[0]), [63]);
    }
}

#[cfg(test)]
mod prepare_sources_tests {
    use super::prepare_sources;
    use crate::registry::Registry;
    use std::path::PathBuf;

    #[test]
    fn missing_paths_fail_instead_of_rendering_empty() {
        let registry = Registry::with_defaults();
        let err = match prepare_sources(
            &[
                PathBuf::from("no-such-file-1.bin"),
                PathBuf::from("no-such-file-2.bin"),
            ],
            &registry,
        ) {
            Err(e) => e,
            Ok(_) => panic!("all-missing input list should fail"),
        };
        assert!(
            err.to_string().contains("no readable input files"),
            "unexpected error: {err:#}"
        );
        assert!(err.to_string().contains("no-such-file-1.bin"));
    }

    #[test]
    fn one_readable_path_among_missing_still_renders() {
        let registry = Registry::with_defaults();
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.bin");
        std::fs::write(&good, b"hello").unwrap();
        let (sources, total) =
            prepare_sources(&[good, PathBuf::from("no-such-file.bin")], &registry).unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(total, 5);
    }
}
