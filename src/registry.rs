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

use crate::tiled::leaf_renderer::LeafRegistry;

/// Parses format-specific headers (safetensors / GGUF / PyTorch pickle / …).
///
/// Consumed in step 12 when format detection moves into modelweightvis.
#[allow(dead_code)]
pub trait FormatPlugin: Send + Sync {
    fn id(&self) -> &'static str;
}

/// Builds a layout for the given sources, when applicable.
///
/// Consumed in step 6 when `select_layout` becomes a priority iteration over
/// registered plugins.
#[allow(dead_code)]
pub trait LayoutPlugin: Send + Sync {
    fn id(&self) -> &'static str;
    fn priority(&self) -> i32;
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
    /// Today only `leaf` is populated (`"hilbert-bytes"` byte-Hilbert pair).
    /// The `"arch"` leaf pair is also present until step 12 hands it off to
    /// modelweightvis. The other three slots will be populated as steps 6, 9,
    /// and 12 migrate their respective hardcoded paths behind the trait
    /// surface.
    pub fn with_defaults() -> Self {
        Self {
            formats: Vec::new(),
            layouts: Vec::new(),
            leaf: LeafRegistry::with_defaults(),
            diffs: Vec::new(),
        }
    }
}
