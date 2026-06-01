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

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

mod data;
mod diff;
mod finetune;
mod format;
mod format_plugin;
mod hooks;
mod layout;
mod single_arch;
mod tiled;

pub use diff::TensorDiffBuilder;
pub use format_plugin::{GgufFormatPlugin, PickleFormatPlugin, SafetensorsFormatPlugin};
pub use hooks::{
    ArchSingleImageHook, HfModelCardFinetuneDetect, TensorDirectoryDiffPrep, TensorMoeDiffPrep,
    TensorRepoDiffPrep,
};
pub use layout::{ArchLayoutPlugin, MoeDiffLayoutPlugin};
pub use tiled::{ArchRegionsLoader, ArchRegionsRenderer};

use std::sync::Arc;

use arbvis::Registry;

/// Register every tensor-aware plugin on `registry`.
///
/// Populates the four Vec slots (`formats`, `layouts`, `diffs`, plus
/// `leaf`'s loader+renderer maps) and every Option-slot hook
/// (`moe_diff`, `repo_diff`, `dir_tensor_diff`, `finetune_detect`,
/// `single_image_arch`). After this returns, arbvis::run handles every
/// CLI shape the pre-split single-crate `arbvis` did, including
/// `--moe-diff`, repo-level `--diff`, directory-safetensors `--diff`,
/// single-image arch, and FormatPlugin-driven `ModelInfo` population.
pub fn register_all(registry: &mut Registry) {
    // Per-format header parsers — first plugin that detects a path
    // wins. Stuff `ModelInfo` into `Source.extensions` so the arch
    // layout / arch tile loader / renderer pick it up downstream.
    registry.formats.push(Arc::new(SafetensorsFormatPlugin));
    registry.formats.push(Arc::new(GgufFormatPlugin));
    registry.formats.push(Arc::new(PickleFormatPlugin));

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

    // Option-slot hooks — each one taps a single CLI dispatch.
    registry.moe_diff = Some(Arc::new(TensorMoeDiffPrep));
    registry.repo_diff = Some(Arc::new(TensorRepoDiffPrep));
    registry.dir_tensor_diff = Some(Arc::new(TensorDirectoryDiffPrep));
    registry.finetune_detect = Some(Arc::new(HfModelCardFinetuneDetect));
    registry.single_image_arch = Some(Arc::new(ArchSingleImageHook));
}
