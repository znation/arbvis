//! Pluggable surface threaded through [`crate::run`].
//!
//! After step 12 + the heavy-dep relocation, arbvis is the byte-only
//! foundation: it owns the tile pipeline, byte-Hilbert layout, JSON-diff,
//! plain-byte diff, HF I/O, and Space deploy. Everything model-aware
//! (tensor-format parsing, architectural / MoE layouts, tensor-diff
//! source prep, finetune auto-detection, the arch single-image renderer,
//! the arch tile loader/renderer) lives in `modelweightvis` and registers
//! against this surface.
//!
//! The hooks are split into two families:
//!   - **Vec slots** (`formats`, `layouts`, `diffs`): multiple plugins,
//!     iterated/priority-ordered. Always present (empty by default).
//!   - **Option slots** (`moe`, `repo_diff`, `dir_tensor_diff`,
//!     `finetune_detect`, `single_image_arch`): single tap point per CLI
//!     dispatch. When `None`, the corresponding feature errors out (or
//!     defaults) and the rest of arbvis still works.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;

use crate::data::{Data, DiffMetric, Extensions, Source, SummaryStat};
use crate::hf_url::RemoteFileSpec;
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

/// Inputs every [`LayoutPlugin`] sees when deciding whether to apply and
/// how to build.
pub struct LayoutBuildCtx<'a> {
    pub sources: &'a [Source],
    pub cumulative_offsets: &'a [u64],
    pub total_bytes: u64,
    pub mode: LayoutMode,
    pub diff_mode: bool,
}

/// Builds a layout for the given sources, when applicable.
pub trait LayoutPlugin: Send + Sync {
    fn id(&self) -> &'static str;
    fn priority(&self) -> i32;
    fn applicable(&self, ctx: &LayoutBuildCtx<'_>) -> bool;
    fn build(&self, ctx: &LayoutBuildCtx<'_>) -> Option<Box<dyn LayoutShape>>;
}

/// Inputs every [`DiffSourceBuilder`] sees for a file-pair `--diff` run.
pub struct DiffBuildCtx<'a> {
    pub original: &'a Path,
    pub modified: &'a Path,
    pub is_finetune: bool,
    pub metric: DiffMetric,
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

/// Hooks the `--moe` CLI flag's source preparation. arbvis errors out when
/// the flag is passed but no `MoeScenesPrep` is registered (the byte-only
/// `arbvis` binary has none; `modelweightvis` registers the tensor-aware impl).
#[async_trait(?Send)]
pub trait MoeScenesPrep: Send + Sync {
    /// Build every MoE scene for `input` in a single pass — loading the
    /// model once — and return their `Source`s concatenated into one list.
    ///
    /// Each returned `Source` carries a [`crate::SceneTag`] assigning it to
    /// a scene (the canonical impl emits a `"summary"` scene of per-expert
    /// scalar heatmaps and a `"cka"` scene of N×N linear-CKA matrices); the
    /// tiler partitions on that tag into independent pyramids and the viewer
    /// renders a tab switcher.
    ///
    /// `stat` configures the summary scene's per-expert scalar; `sample` is
    /// the CKA scene's random-projection dimension (smaller = faster). When
    /// `probe.enabled`, the impl runs a routing-faithful forward and attaches
    /// a behavioral panel to each scene (routing frequency to summary,
    /// co-activation to CKA).
    ///
    /// `?Send`: the canonical impl in `modelweightvis::hooks` carries futures
    /// whose internal lifetimes confuse the rustc HRTB check that
    /// `#[async_trait]`'s `+ Send` bound triggers. The CLI dispatch never
    /// spawns this future across threads — only `.await`s it locally — so
    /// dropping the Send requirement is safe.
    async fn prepare(
        &self,
        input: &str,
        stat: SummaryStat,
        sample: u32,
        stream: bool,
        probe: &crate::ProbeOpts,
    ) -> anyhow::Result<(Vec<Source>, u64)>;
}

/// Hooks repo-level `--diff` (both args are `hf://owner/repo` URLs).
/// arbvis errors out when this case is hit and no `RepoDiffPrep` is
/// registered.
///
/// `?Send`: see [`MoeScenesPrep`].
#[async_trait(?Send)]
pub trait RepoDiffPrep: Send + Sync {
    async fn prepare(
        &self,
        orig_specs: &[(String, RemoteFileSpec)],
        mod_specs: &[(String, RemoteFileSpec)],
        is_finetune: bool,
        metric: DiffMetric,
        stream: bool,
    ) -> anyhow::Result<(Vec<Source>, u64)>;
}

/// Hooks the tensor-files-in-directory branch of `--diff <dir> <dir>`.
/// The non-tensor files are still handled by arbvis's generic byte-diff
/// + crosshatch path.
///
/// `?Send`: see [`MoeScenesPrep`].
#[async_trait(?Send)]
pub trait DirectoryTensorDiffPrep: Send + Sync {
    /// Which directory entries this preparer takes responsibility for.
    /// arbvis partitions directory contents on this predicate before
    /// invoking `prepare`.
    fn is_tensor_file(&self, p: &Path) -> bool;
    async fn prepare(
        &self,
        orig_files: &[PathBuf],
        mod_files: &[PathBuf],
        is_finetune: bool,
        metric: DiffMetric,
    ) -> anyhow::Result<(Vec<Source>, u64)>;
}

/// Hooks the HF "is X a finetune of Y" model-card lookup. arbvis defaults
/// to "not a finetune" when this hook isn't registered or returns `None`.
///
/// `?Send`: see [`MoeScenesPrep`].
#[async_trait(?Send)]
pub trait FinetuneDetect: Send + Sync {
    async fn detect(&self, orig_url: &str, mod_url: &str) -> Option<bool>;
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
/// `?Send`: see [`MoeScenesPrep`].
#[async_trait(?Send)]
pub trait PrepareSourcesExtension: Send + Sync {
    async fn enrich(&self, sources: &mut [Source]) -> anyhow::Result<()>;
}

/// Hooks the single-image arch render path. Invoked from
/// `single::run_single` when the chosen layout's id is `"arch"`. arbvis
/// falls back to byte-Hilbert if this hook isn't registered.
pub trait SingleImageArchHook: Send + Sync {
    fn render(
        &self,
        files: &[PathBuf],
        output: Option<PathBuf>,
        sources: &[Source],
        layout: &dyn LayoutShape,
    ) -> anyhow::Result<()>;
}

/// Plugin slots threaded through [`crate::run`].
#[derive(Clone, Default)]
pub struct Registry {
    pub formats: Vec<Arc<dyn FormatPlugin>>,
    pub layouts: Vec<Arc<dyn LayoutPlugin>>,
    pub leaf: LeafRegistry,
    pub diffs: Vec<Arc<dyn DiffSourceBuilder>>,
    pub moe: Option<Arc<dyn MoeScenesPrep>>,
    pub repo_diff: Option<Arc<dyn RepoDiffPrep>>,
    pub dir_tensor_diff: Option<Arc<dyn DirectoryTensorDiffPrep>>,
    pub finetune_detect: Option<Arc<dyn FinetuneDetect>>,
    pub single_image_arch: Option<Arc<dyn SingleImageArchHook>>,
    /// Cross-source enrichment pass that runs once per render after every
    /// `Source` has been built. See [`PrepareSourcesExtension`].
    pub prepare_sources_extension: Option<Arc<dyn PrepareSourcesExtension>>,
}

impl Registry {
    /// Registry populated with arbvis's own (byte-only) built-ins:
    /// `HilbertLayoutPlugin`, `JsonDiffBuilder`, `PlainBytesDiffBuilder`,
    /// and the `"hilbert-bytes"` leaf loader+renderer. All Option hooks
    /// start as `None` — `modelweightvis::register_all` populates them.
    pub fn with_defaults() -> Self {
        Self {
            formats: Vec::new(),
            layouts: vec![Arc::new(crate::layout::HilbertLayoutPlugin)],
            leaf: LeafRegistry::with_defaults(),
            diffs: vec![
                Arc::new(crate::data::JsonDiffBuilder),
                Arc::new(crate::data::PlainBytesDiffBuilder),
            ],
            moe: None,
            repo_diff: None,
            dir_tensor_diff: None,
            finetune_detect: None,
            single_image_arch: None,
            prepare_sources_extension: None,
        }
    }
}
