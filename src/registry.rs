//! Pluggable surface threaded through [`crate::run`].
//!
//! Step 5 of the arbvis/modelweightvis split. Today the trait slots are empty
//! and the existing hardcoded paths (`select_layout`, `prepare_diff_sources`,
//! per-format header parsers) still drive behaviour; later steps will move
//! that logic behind these traits so `modelweightvis` can register tensor
//! formats, the arch layout, and the tensor-diff source builder without
//! touching arbvis.
//!
//! The [`leaf`] field already carries the working `LeafRegistry` (loader+
//! renderer pairs registered in step 3/4a) — that one *is* consumed today.
//! The other three slots are placeholders.

use std::sync::Arc;

use crate::data::{Source, SourceMeta};
use crate::layout::{LayoutMode, LayoutShape};
use crate::tiled::leaf_renderer::LeafRegistry;

/// Parses format-specific headers (safetensors / GGUF / PyTorch pickle / …).
///
/// Consumed in step 12 when format detection moves into modelweightvis.
#[allow(dead_code)]
pub trait FormatPlugin: Send + Sync {
    fn id(&self) -> &'static str;
}

/// Inputs every [`LayoutPlugin`] sees when deciding whether to apply and
/// how to build.
pub struct LayoutBuildCtx<'a> {
    pub sources: &'a [Source],
    pub cumulative_offsets: &'a [u64],
    pub total_bytes: u64,
    pub mode: LayoutMode,
    pub metas: &'a [SourceMeta],
    pub diff_mode: bool,
}

/// Builds a layout for the given sources, when applicable.
///
/// Plugins are tried in descending `priority()` order; the first that returns
/// `true` from `applicable` and `Some(...)` from `build` wins. The byte-Hilbert
/// floor (`i32::MIN`) is guaranteed to always build, so iteration always
/// terminates with a layout.
pub trait LayoutPlugin: Send + Sync {
    /// Stable name (also matches the `LayoutShape::id` of the built layout
    /// when applicable). Used for diagnostic logs and registry lookup.
    fn id(&self) -> &'static str;
    /// Higher wins. Byte-Hilbert is the floor at `i32::MIN`; arch is 100;
    /// MoE-diff is 200.
    fn priority(&self) -> i32;
    /// Cheap check based on `ctx` — returns whether this plugin should be
    /// considered for the current run. Separate from `build` so a `--layout
    /// arch`-requested-but-no-tensors case can produce a more specific log.
    fn applicable(&self, ctx: &LayoutBuildCtx<'_>) -> bool;
    /// Actually build the layout. Returns `None` if applicable but the
    /// layout couldn't be constructed (e.g. arch is applicable but no
    /// transformer-style names were detected). The dispatcher then continues
    /// to the next plugin.
    fn build(&self, ctx: &LayoutBuildCtx<'_>) -> Option<Box<dyn LayoutShape>>;
}

/// Builds diff sources for a `--diff` run when the input pair matches its
/// shape (JSON / plain bytes / per-tensor / …).
///
/// Consumed in step 9 when `prepare_diff_sources` becomes a priority
/// iteration over registered builders.
#[allow(dead_code)]
pub trait DiffSourceBuilder: Send + Sync {
    fn id(&self) -> &'static str;
    fn priority(&self) -> i32;
}

/// Plugin slots threaded through [`crate::run`]. The binary `main()` (and
/// `modelweightvis::main` post-split) builds a registry, populates it with
/// its own plugins, and passes it in.
///
/// `Clone` is a cheap shallow clone — every slot is a `Vec<Arc<…>>` or the
/// `LeafRegistry`'s `Arc<HashMap<…>>`.
#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct Registry {
    pub formats: Vec<Arc<dyn FormatPlugin>>,
    pub layouts: Vec<Arc<dyn LayoutPlugin>>,
    /// Loader+renderer lookup for the tile pipeline. Already populated with
    /// built-in `"hilbert-bytes"` and `"arch"` pairs by `LeafRegistry::with_defaults`.
    pub leaf: LeafRegistry,
    pub diffs: Vec<Arc<dyn DiffSourceBuilder>>,
}

impl Registry {
    /// Registry populated with arbvis's own built-ins.
    ///
    /// Today: `leaf` carries the `"hilbert-bytes"` + `"arch"` loader/renderer
    /// pairs; `layouts` carries `HilbertLayoutPlugin` (the `i32::MIN` floor),
    /// `ArchLayoutPlugin`, and `MoeDiffLayoutPlugin`. The `"arch"` parts move
    /// to modelweightvis in step 12. `formats` and `diffs` remain empty until
    /// steps 9 and 12 populate them.
    pub fn with_defaults() -> Self {
        Self {
            formats: Vec::new(),
            layouts: vec![
                Arc::new(crate::layout::HilbertLayoutPlugin),
                Arc::new(crate::layout::ArchLayoutPlugin),
                Arc::new(crate::layout::MoeDiffLayoutPlugin),
            ],
            leaf: LeafRegistry::with_defaults(),
            diffs: Vec::new(),
        }
    }
}
