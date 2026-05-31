//! Tensor-format-aware visualization built on `arbvis`.
//!
//! Owns every model-aware piece of the workspace:
//! - `format/` — safetensors / GGUF / PyTorch-pickle header parsers, dtype
//!   tables, MoE expert decoding.
//! - `layout/` — architectural canvas, MoE-diff matrix layout, transformer
//!   name classification, dtype-aware element colorizers.
//! - `tiled/` — per-tensor tile load and the dtype-aware tile renderers.
//! - `data` — `TensorDiffSource`, MoE-diff source prep, multi-shard
//!   safetensors diff helpers, `MoeCell`, `SourceMeta`.
//! - `finetune` — HF model-card finetune auto-detection.
//! - Plugin impls (`ArchLayoutPlugin`, `MoeDiffLayoutPlugin`,
//!   `TensorDiffBuilder`, `ArchRegionsLoader`, `ArchRegionsRenderer`, the
//!   `FormatPlugin` family, plus the `MoeDiffPrep`/`RepoDiffPrep`/
//!   `DirectoryTensorDiffPrep`/`FinetuneDetect`/`SingleImageArchHook` hooks)
//!   are registered on a registry via [`register_all`].
//!
//! The `modelweightvis` binary builds `arbvis::Registry::with_defaults()`,
//! calls `register_all(&mut registry)`, and hands off to `arbvis::run`.

// Mid-relocation: many model-aware helpers (FormatPlugin impls,
// MoeDiffPrep/RepoDiffPrep/DirectoryTensorDiffPrep/FinetuneDetect/
// SingleImageArchHook trait impls, plus the data-side prep helpers
// they call) are defined but not yet wired into `register_all`. The
// dead_code allow comes off when the hook impls land.
#![allow(dead_code, clippy::too_many_arguments, clippy::type_complexity)]

mod data;
mod diff;
mod finetune;
mod format;
mod layout;
mod tiled;

pub use diff::TensorDiffBuilder;
pub use layout::{ArchLayoutPlugin, MoeDiffLayoutPlugin};
pub use tiled::{ArchRegionsLoader, ArchRegionsRenderer};

use std::sync::Arc;

use arbvis::Registry;

/// Register every tensor-aware plugin on `registry`.
pub fn register_all(registry: &mut Registry) {
    // Tensor-aware diff (.safetensors / .gguf file pairs) — priority 300 so
    // it wins over the JSON / plain-byte fallbacks for matching pairs.
    registry.diffs.push(Arc::new(TensorDiffBuilder));

    // Architectural + MoE-diff layouts. Priority 200 / 100; `select_layout`
    // sorts by `priority()` descending.
    registry.layouts.push(Arc::new(ArchLayoutPlugin));
    registry.layouts.push(Arc::new(MoeDiffLayoutPlugin));

    // Tile loader+renderer pair for the `"arch"` layout id.
    registry.leaf.register_loader(Arc::new(ArchRegionsLoader));
    registry
        .leaf
        .register_renderer(Arc::new(ArchRegionsRenderer));

    // Optional hooks (TODO): implement `TensorMoeDiffPrep`,
    // `TensorRepoDiffPrep`, `TensorDirectoryDiffPrep`,
    // `HfModelCardFinetuneDetect`, `ArchSingleImageHook`, and the
    // `SafetensorsFormatPlugin` / `GgufFormatPlugin` / `PickleFormatPlugin`
    // family on top of the moved-here `crate::data::*` helpers. Until
    // those wrappers exist, --moe-diff, repo-level --diff,
    // directory-safetensors --diff, single-image arch, and
    // FormatPlugin-driven `ModelInfo` population error cleanly via
    // arbvis's `Option`-slot guards.
}
