//! Pluggable surface threaded through [`crate::run`].
//!
//! arbvis is the generic byte-only foundation: it owns the tile pipeline,
//! byte-Hilbert layout, JSON-diff, plain-byte diff, HF I/O, and Space deploy.
//! Everything specialization-specific (a domain's format parsing, structured
//! layouts, custom source preparation, custom tile
//! loaders/renderers) is registered against this surface by a downstream crate
//! — `modelweightvis` (tensor/model-weight viz) is one such specialization.
//!
//! The extension points come in three families:
//!   - **Vec slots** (`formats`, `layouts`, `diffs`, `providers`): multiple
//!     plugins, iterated by descending priority. The byte built-ins are always
//!     present; a downstream pushes higher-priority entries.
//!   - **Id-keyed maps** (`leaf`): a per-layout-id tile loader/renderer
//!     registered by a downstream layout plugin.
//!   - **Option slots** (`prepare_sources_extension`): a single optional tap
//!     point. When `None` the feature no-ops and the rest of arbvis still works.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;

use crate::data::{Data, Extensions, Source};
use crate::layout::{LayoutMode, LayoutShape};
use crate::tiled::leaf_renderer::LeafRegistry;

/// Format-aware metadata populator. `prepare_sources` /
/// `prepare_sources_from_specs` call each registered `FormatPlugin` for
/// every loaded source — first plugin with `detects_path()` true wins —
/// and lets it stuff format-specific data (e.g. `ModelInfo` for
/// safetensors) into `Source.extensions`. Plugin readers (e.g.
/// `ArchLayoutPlugin`) read from `s.extensions.get::<…>()`.
pub trait FormatPlugin: Send + Sync {
    fn id(&self) -> &'static str;
    /// Should this plugin handle `path`? File extension match.
    fn detects_path(&self, path: &Path) -> bool;
    /// Mmap `path` and parse its header; populate `exts` with whatever
    /// the plugin wants attached. Sync (called from `prepare_sources`).
    fn populate_local(
        &self,
        path: &Path,
        file_size: u64,
        exts: &mut Extensions,
    ) -> anyhow::Result<()>;
    /// Async variant for an already-open `Data` source (HTTP / mmap).
    /// Called from `prepare_sources_from_specs`.
    fn populate_remote<'a>(
        &'a self,
        data: &'a Data,
        byte_size: u64,
        exts: &'a mut Extensions,
    ) -> BoxFuture<'a, anyhow::Result<()>>;
}

/// Inputs every [`LayoutPlugin`] (2D) and [`VolumeShapePlugin`] (3D) sees when
/// deciding whether to apply and how to build.
pub struct LayoutBuildCtx<'a> {
    pub sources: &'a [Source],
    pub cumulative_offsets: &'a [u64],
    pub total_bytes: u64,
    pub mode: LayoutMode,
    pub diff_mode: bool,
    /// The `--3d` voxel cube side (a power of two). Only meaningful to a
    /// [`VolumeShapePlugin`]; the 2D tile path sets it to 0 and ignores it.
    pub grid_side: u32,
}

/// Builds a 2D layout for the given sources, when applicable.
pub trait LayoutPlugin: Send + Sync {
    fn id(&self) -> &'static str;
    fn priority(&self) -> i32;
    fn applicable(&self, ctx: &LayoutBuildCtx<'_>) -> bool;
    fn build(&self, ctx: &LayoutBuildCtx<'_>) -> Option<Box<dyn LayoutShape>>;
}

/// Builds a 3D volume layout for the given sources, when applicable. The 3D
/// analog of [`LayoutPlugin`]; selected by
/// [`crate::volume::select_volume_shape`] with the same descending-priority /
/// `i32::MIN`-floor rules. A downstream gates on [`LayoutBuildCtx::mode`] in
/// `applicable` exactly as the 2D arch layout does.
pub trait VolumeShapePlugin: Send + Sync {
    fn id(&self) -> &'static str;
    fn priority(&self) -> i32;
    fn applicable(&self, ctx: &LayoutBuildCtx<'_>) -> bool;
    fn build(&self, ctx: &LayoutBuildCtx<'_>) -> Option<Box<dyn crate::volume::VolumeShape>>;
}

/// Inputs every [`DiffSourceBuilder`] sees for a file-pair `--diff` run.
pub struct DiffBuildCtx<'a> {
    pub original: &'a Path,
    pub modified: &'a Path,
    pub is_finetune: bool,
}

/// Builds diff sources for a file-pair `--diff` run when the input pair
/// matches its shape.
#[async_trait]
pub trait DiffSourceBuilder: Send + Sync {
    #[allow(dead_code)]
    fn id(&self) -> &'static str;
    fn priority(&self) -> i32;
    async fn try_build(&self, ctx: &DiffBuildCtx<'_>)
        -> anyhow::Result<Option<(Vec<Source>, u64)>>;
}

/// Coarse output-destination shape — enough for a [`SourceProvider`] to gate
/// on without seeing the tempdir / upload internals of arbvis's `OutputDest`.
///
/// arbvis now has a single output shape: a deployable web-viewer bundle
/// directory (a 2D tile pyramid, or — under `--3d` — a volume bundle). The
/// native window and single-PNG destinations were removed; `Bundle` is the
/// lone variant a provider can see. Whether the run is 2D or 3D is carried
/// separately on [`SourceCtx::three_d`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DestKind {
    Bundle,
}

/// The two sides of a `--diff`, exactly as the user wrote them (a local path
/// or an `hf://` URL). Kept as an ordered pair — the diff LUT and crosshatch
/// direction depend on (original, modified) order — and deliberately neutral:
/// arbvis core no longer knows or cares whether the two sides are tensors.
pub struct DiffPair<'a> {
    pub original: &'a str,
    pub modified: &'a str,
}

/// The parsed, neutral description of what the user asked arbvis to render.
/// Built once in [`crate::run`] and passed to every [`SourceProvider`].
pub struct SourceCtx<'a> {
    /// Positional inputs (`FILES` + expanded `--file-list`), already collected.
    /// Empty means "read stdin" for the byte provider.
    pub inputs: &'a [PathBuf],
    /// Present iff `--diff ORIGINAL MODIFIED` was passed.
    pub diff: Option<DiffPair<'a>>,
    /// The resolved destination shape. Always [`DestKind::Bundle`] today; kept
    /// so a provider can gate on it as the surface grows.
    pub dest_kind: DestKind,
    /// `--3d`: render the volume/point-cloud bundle instead of the 2D tile
    /// pyramid. Providers still build the same byte `Source`s either way; this
    /// lets one opt out of a mode it can't represent (e.g. a 2D-only layout).
    pub three_d: bool,
    /// `--stream`.
    pub stream: bool,
    /// `--show-xet-xorbs`.
    pub show_xet_xorbs: bool,
    /// The registry, so a provider can reuse arbvis's byte machinery
    /// (`data::prepare_sources`, `data::byte_directory_diff`, the diff-builder
    /// cascade, `hf_url` helpers).
    pub registry: &'a Registry,
}

/// Per-provider render hints that [`crate::run`] folds into the final render
/// configuration. Everything else (layout mode, tile formats, the user's
/// `--title` override) comes from `Args`/`Registry`, not the provider.
#[derive(Clone, Debug, Default)]
pub struct RenderHints {
    /// Drives the diff LUT + crosshatch overlay and the layout `diff_mode` gate.
    pub diff_mode: bool,
    /// Title suffix used when the user didn't pass `--title` (e.g. `"diff"`,
    /// `"moe"`, or `""` for the plain byte path). Fed to arbvis's `default_title`.
    pub title_suffix: Cow<'static, str>,
    /// Whether to color regions by xet xorb id. The byte provider passes the
    /// user's `--show-xet-xorbs`; diff / multi-scene providers force `false`.
    pub show_xet_xorbs: bool,
    /// Display inputs for the viewer info panel — the provider is the authority
    /// on what it actually rendered (e.g. the two `--diff` paths, the input
    /// file list, or a single specialization target).
    pub inputs: Vec<String>,
}

/// Turns a parsed invocation into render sources — the single generic seam that
/// replaced the old per-feature option-slot hooks. arbvis's own byte renderer
/// and byte/JSON diff are built-in providers; a downstream specialization
/// registers higher-priority providers for its own input shapes (e.g. a
/// multi-scene view, a repo-level diff, a directory diff).
///
/// Priority-ordered exactly like [`LayoutPlugin`]: [`crate::run`] iterates
/// registered providers by descending priority and uses the first whose
/// [`applicable`](SourceProvider::applicable) returns true, which then commits
/// (no deferral — so `applicable` must be precise, not optimistic).
///
/// arbvis ships two built-ins: a byte-diff provider (priority 100, applies when
/// `--diff` is set) and a normal-bytes provider (priority `i32::MIN`, the floor
/// that always applies). The floor guarantees the iteration always terminates
/// with a chosen provider.
///
/// `?Send`: a downstream provider's `prepare` future may be non-`Send` (e.g. a
/// model forward pass holding `!Send` state across awaits). `run` only `.await`s
/// the chosen future locally, never spawning it across threads, so dropping the
/// `Send` bound is safe.
#[async_trait(?Send)]
pub trait SourceProvider: Send + Sync {
    /// Stable identifier for diagnostics. Format: kebab-case.
    fn id(&self) -> &'static str;
    /// Selection priority; higher wins. The built-in normal-bytes floor sits at
    /// `i32::MIN`.
    fn priority(&self) -> i32;
    /// Cheap, synchronous gate — inspect `ctx`'s shape only, no I/O.
    fn applicable(&self, ctx: &SourceCtx<'_>) -> bool;
    /// Build the render sources. Only the chosen provider's `prepare` runs.
    async fn prepare(&self, ctx: &SourceCtx<'_>)
        -> anyhow::Result<(Vec<Source>, u64, RenderHints)>;
}

/// Opportunistic post-prepare-sources enrichment hook. Runs after sources
/// have been built and `FormatPlugin::populate_{local,remote}` has stuffed
/// each source's own format-specific metadata into its `extensions`. The
/// impl sees the full source list at once so it can dedup cross-source work
/// (e.g. one `config.json` fetch per HF repo regardless of shard count).
/// arbvis no-ops when this hook isn't registered.
///
/// In practice the only caller is modelweightvis's
/// `SourceMetaSidecarHook`, which fetches `config.json` and
/// `model.safetensors.index.json` alongside each source and inserts a
/// `SourceMeta` into the extensions map for the arch layout to read back.
/// Other tensor-aware backends (e.g. an ONNX backend) could plug in here
/// the same way without arbvis growing any extra slots.
///
/// `?Send`: see [`SourceProvider`].
#[async_trait(?Send)]
pub trait PrepareSourcesExtension: Send + Sync {
    async fn enrich(&self, sources: &mut [Source]) -> anyhow::Result<()>;
}

/// Viewer branding: the tool name shown in the HTML title fallbacks
/// (`"{name}"` / `"{name} diff"` / `"{name} moe"`) and the repo URL used for
/// the title link + leaflet attribution.
///
/// Defaults to arbvis's own identity so the standalone `arbvis` binary keeps
/// its branding. A downstream crate (e.g. `modelweightvis`) overrides
/// [`Registry::branding`] to rebrand the viewer it generates.
#[derive(Clone, Debug)]
pub struct Branding {
    pub name: Cow<'static, str>,
    pub repo_url: Cow<'static, str>,
}

impl Branding {
    pub fn new(name: impl Into<Cow<'static, str>>, repo_url: impl Into<Cow<'static, str>>) -> Self {
        Self {
            name: name.into(),
            repo_url: repo_url.into(),
        }
    }
}

impl Default for Branding {
    fn default() -> Self {
        Self::new("arbvis", "https://github.com/znation/arbvis")
    }
}

/// Plugin slots threaded through [`crate::run`].
#[derive(Clone, Default)]
pub struct Registry {
    pub formats: Vec<Arc<dyn FormatPlugin>>,
    pub layouts: Vec<Arc<dyn LayoutPlugin>>,
    /// 3D (`--3d`) layout plugins, the volume analog of `layouts`. `run` picks
    /// the highest-priority applicable one via
    /// [`crate::volume::select_volume_shape`]; the `i32::MIN`
    /// `HilbertVolumePlugin` floor (installed by [`Registry::with_defaults`])
    /// always applies.
    pub volume_shapes: Vec<Arc<dyn VolumeShapePlugin>>,
    /// Id-keyed 3D voxel renderers, the volume analog of `leaf`. A structured
    /// `VolumeShape`'s entities dispatch here by `renderer_id`.
    pub voxel: crate::volume::VoxelRegistry,
    pub leaf: LeafRegistry,
    pub diffs: Vec<Arc<dyn DiffSourceBuilder>>,
    /// Priority-ordered source providers. `run` picks the highest-priority
    /// applicable one. See [`SourceProvider`]; populated with arbvis's two
    /// byte built-ins by [`Registry::with_defaults`].
    pub providers: Vec<Arc<dyn SourceProvider>>,
    /// Cross-source enrichment pass that runs once per render after every
    /// `Source` has been built. See [`PrepareSourcesExtension`].
    pub prepare_sources_extension: Option<Arc<dyn PrepareSourcesExtension>>,
    /// User-chosen layout strategy for this run. arbvis-only leaves the default
    /// ([`LayoutMode::Auto`]); a downstream maps its own `--layout` flag onto
    /// this (e.g. [`LayoutMode::Forced`] with one of its layout ids).
    pub layout_mode: LayoutMode,
    /// When `true`, a forced [`LayoutMode::Forced`] layout that can't build for
    /// the inputs becomes a hard error in [`crate::layout::select_layout`]
    /// instead of a warning + silent fallback. Downstreams that map a
    /// `--strict-layout` flag set this alongside `layout_mode`. No effect under
    /// [`LayoutMode::Auto`]/[`LayoutMode::Hilbert`] (neither can fall back).
    pub strict_layout: bool,
    /// Viewer branding (tool name + repo URL). See [`Branding`].
    pub branding: Branding,
}

impl Registry {
    /// Registry populated with arbvis's own (byte-only) built-ins:
    /// `HilbertLayoutPlugin`, the `JsonDiffBuilder`/`PlainBytesDiffBuilder`
    /// file-pair builders, the `ByteDiffProvider`/`NormalBytesProvider` source
    /// providers, and the `"hilbert-bytes"` leaf loader+renderer. A downstream
    /// extends these (e.g. `modelweightvis::register_all`).
    pub fn with_defaults() -> Self {
        Self {
            formats: Vec::new(),
            layouts: vec![Arc::new(crate::layout::HilbertLayoutPlugin)],
            volume_shapes: vec![Arc::new(crate::volume::HilbertVolumePlugin)],
            voxel: crate::volume::VoxelRegistry::with_defaults(),
            leaf: LeafRegistry::with_defaults(),
            diffs: vec![
                Arc::new(crate::data::JsonDiffBuilder),
                Arc::new(crate::data::PlainBytesDiffBuilder),
            ],
            providers: vec![
                Arc::new(crate::ByteDiffProvider),
                Arc::new(crate::NormalBytesProvider),
            ],
            prepare_sources_extension: None,
            layout_mode: LayoutMode::Auto,
            strict_layout: false,
            branding: Branding::default(),
        }
    }
}
