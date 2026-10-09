//! Data sources: backing storage ([`Data`]), source construction and loading,
//! and the [`DiffFill`] crosshatch colors. The diff-source builders live in
//! `crate::data_diff`.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::future::BoxFuture;
use memmap2::Mmap;

use crate::hf_url::{RemoteFileSpec, RemoteRepo};
use crate::xet::{XetReader, XetTerm};

mod diff;

pub use diff::DiffFill;

// The diff-source builders and directory byte-diff walker live in
// `crate::data_diff`; re-exported here so `data::…` paths stay valid.
pub use crate::data_diff::{
    byte_directory_diff, prepare_diff_sources, JsonDiffBuilder, PlainBytesDiffBuilder,
};

/// Async fetcher closure used by [`Data::LazyDiff`]. Captures its inputs by
/// `Arc` so the returned future is `'static` and can be sent across tasks.
/// `CustomSource::open` impls in `modelweightvis::data` build these for
/// per-tensor diff buffers.
pub type LazyFetcher =
    Arc<dyn Fn(u64, usize) -> BoxFuture<'static, anyhow::Result<Vec<u8>>> + Send + Sync>;

/// Source construction and loading (prepare/load/materialize). The type
/// definitions below stay here; this re-export keeps the crate-internal call
/// sites (`data::prepare_sources`, `diff.rs`) and the `lib.rs` surface intact.
mod remote;
mod source;
pub use remote::{
    download_specs_to_paths, materialize_http_sources, populate_xet_terms,
    prepare_sources_from_specs,
};
pub use source::{collect_files_recursive, load_source_data, prepare_sources};

/// True when a path carries a `.json` or `.jsonl` extension. Shared by the
/// JSON diff-source builders (structure-aware diff applies only when both
/// sides of a pair qualify).
pub(crate) fn is_json_path(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()),
        Some("json") | Some("jsonl")
    )
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
            Data::Mapped(m) => slice_local(m, start, len),
            Data::Owned(v) => slice_local(v, start, len),
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

/// Slice a local byte buffer for `fetch_range`, turning an out-of-bounds
/// range into a descriptive error instead of a slice-index panic.
///
/// A source's `byte_size` is captured by an earlier stat/scan, so the file
/// can shrink (another writer truncating it, a swapped-in shorter file)
/// between that scan and the render. Without this guard the render dies
/// mid-run with a panic; with it, the caller gets a clean `anyhow` error
/// naming the requested range and the actual length.
fn slice_local(buf: &[u8], start: u64, len: usize) -> anyhow::Result<Vec<u8>> {
    let s = start as usize;
    let end = s
        .checked_add(len)
        .ok_or_else(|| anyhow::anyhow!("byte range [start {start}, len {len}) overflows usize"))?;
    if end > buf.len() {
        anyhow::bail!(
            "byte range [{s}, {end}) is out of bounds: source is {} bytes \
             (file may have shrunk since it was scanned)",
            buf.len()
        );
    }
    Ok(buf[s..end].to_vec())
}

/// A `Source` variant supplied by a downstream crate / plugin. arbvis core
/// ships no impls today; downstream crates (e.g. modelweightvis's tensor-diff
/// sources, formerly `SourceKind::TensorDiff`) supply them, and the core just
/// dispatches by trait.
pub trait CustomSource: Send + Sync {
    /// Stable identifier for diagnostic logs and runtime predicates (e.g.
    /// "is this a tensor-diff source?"). Format: kebab-case.
    fn id(&self) -> &'static str;
    /// Byte size of the synthetic stream this source exposes. arbvis's own
    /// pipeline never calls this today — canvas layout reads the
    /// `Source::byte_size` field, which the impl sets when it constructs the
    /// `Source` — but it stays part of the plugin contract for downstream.
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
    /// byte delta) but parameterised over `(Arc<Data>, start_offset)` so each
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
    /// touches `open` (and `id` for diagnostics); the canvas size comes from
    /// the `Source::byte_size` field the impl sets when it constructs the
    /// `Source`, and everything else is up to the impl. arbvis constructs no
    /// `Custom` sources today; downstream crates supply them (formerly
    /// `SourceKind::TensorDiff`) for tensor-aware `--diff` runs.
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

#[cfg(test)]
mod data_fetch_tests {
    use super::Data;
    use std::sync::Arc;

    fn owned(bytes: &[u8]) -> Data {
        Data::Owned(bytes.to_vec())
    }

    #[tokio::test]
    async fn fetch_range_slices_local_variants() {
        let data = owned(b"hello world");
        assert_eq!(data.fetch_range(0, 5).await.unwrap(), b"hello");
        assert_eq!(data.fetch_range(6, 5).await.unwrap(), b"world");
        assert_eq!(data.fetch_range(11, 0).await.unwrap(), b"");
    }

    #[tokio::test]
    async fn fetch_range_errors_on_out_of_bounds() {
        // A file that shrinks after the scan (another writer truncating it)
        // previously panicked with an opaque slice-index message, killing the
        // whole render; it must be a descriptive error instead.
        let data = owned(b"abc");
        let err = data.fetch_range(2, 5).await.unwrap_err().to_string();
        assert!(err.contains("out of bounds"), "unexpected: {err}");
        assert!(err.contains("shrunk"), "unexpected: {err}");
        // Exact-end ranges (start == len) stay valid, including zero-length.
        assert_eq!(data.fetch_range(3, 0).await.unwrap(), b"");
    }

    #[tokio::test]
    async fn fetch_range_rejects_len_overflow() {
        let data = owned(b"abc");
        let err = data
            .fetch_range(usize::MAX as u64, usize::MAX)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("overflows usize"), "unexpected: {err}");
    }

    #[tokio::test]
    async fn offset_slice_fetch_error_names_the_shifted_range() {
        // base + start past the inner data now surfaces the inner error
        // instead of panicking.
        let inner = Arc::new(owned(b"0123456789"));
        let view = Data::OffsetSlice { inner, base: 8 };
        assert_eq!(view.fetch_range(0, 2).await.unwrap(), b"89");
        let err = view.fetch_range(3, 2).await.unwrap_err().to_string();
        assert!(err.contains("out of bounds"), "unexpected: {err}");
    }

    #[tokio::test]
    async fn zero_fill_returns_requested_length_of_zeros() {
        let data = Data::ZeroFill;
        assert_eq!(data.fetch_range(0, 4).await.unwrap(), [0u8; 4]);
        assert_eq!(data.fetch_range(1_000_000, 2).await.unwrap(), [0u8; 2]);
    }

    #[tokio::test]
    async fn offset_slice_shifts_fetch_and_deref_into_inner() {
        let inner = Arc::new(owned(b"0123456789"));
        let view = Data::OffsetSlice {
            inner: Arc::clone(&inner),
            base: 3,
        };
        // fetch_range(s, n) resolves to inner.fetch_range(base + s, n).
        assert_eq!(view.fetch_range(0, 2).await.unwrap(), b"34");
        assert_eq!(view.fetch_range(4, 3).await.unwrap(), b"789");
        // Deref exposes the inner tail starting at base.
        assert_eq!(&*view, b"3456789");
    }

    #[tokio::test]
    async fn offset_slice_delegates_is_local_to_inner() {
        let inner = Arc::new(owned(b"abc"));
        let view = Data::OffsetSlice {
            inner: Arc::clone(&inner),
            base: 0,
        };
        assert!(view.is_local());
        assert!(view.is_local());
    }

    #[tokio::test]
    async fn lazy_diff_fetches_through_the_closure() {
        let data = Data::LazyDiff(Arc::new(move |start: u64, len: usize| {
            Box::pin(async move { Ok((start as u8..).take(len).collect::<Vec<u8>>()) })
                as futures::future::BoxFuture<'static, anyhow::Result<Vec<u8>>>
        }));
        assert!(!data.is_local(), "LazyDiff may hit the network");
        assert_eq!(data.fetch_range(10, 3).await.unwrap(), [10, 11, 12]);
    }

    #[test]
    fn is_local_matches_each_variant() {
        assert!(owned(b"").is_local());
        assert!(Data::ZeroFill.is_local());
    }
}

#[cfg(test)]
mod source_name_tests {
    use super::{Extensions, Source, SourceKind};

    fn source(kind: SourceKind) -> Source {
        Source {
            file_idx: 0,
            kind,
            byte_size: 0,
            name_override: None,
            xet_terms: None,
            extensions: Extensions::default(),
        }
    }

    #[test]
    fn name_varies_by_kind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.bin");
        std::fs::write(&path, b"x").unwrap();
        assert_eq!(source(SourceKind::File(path.clone())).name(), "model.bin");
        assert_eq!(source(SourceKind::Buffered(vec![])).name(), "stdin");
        assert_eq!(
            source(SourceKind::Diff {
                original: path.clone(),
                modified: dir.path().join("other.bin"),
            })
            .name(),
            "model.bin"
        );
    }

    #[test]
    fn name_override_wins_over_every_kind() {
        let mut s = source(SourceKind::Buffered(vec![1]));
        s.name_override = Some("real-name.bin".to_string());
        assert_eq!(s.name(), "real-name.bin");
    }

    #[test]
    fn extensions_store_one_typed_value_and_replace() {
        let mut ext = Extensions::default();
        ext.insert(7u32);
        ext.insert("tag".to_string());
        assert_eq!(*ext.get::<u32>().unwrap(), 7);
        assert_eq!(ext.get::<u64>(), None);
        // Same-type insert replaces the prior value.
        ext.insert(9u32);
        assert_eq!(*ext.get::<u32>().unwrap(), 9);
        // Debug only reports the count, never the payloads.
        let debug = format!("{ext:?}");
        assert!(debug.contains("type_count: 2"), "unexpected: {debug}");
    }
}
