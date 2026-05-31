//! Tensor-format-aware visualization built on `arbvis`.
//!
//! Provides the model-aware plugin set (architectural + MoE-diff layouts,
//! tensor-diff source builder, arch leaf loader+renderer) layered on top
//! of `arbvis::Registry::with_defaults`. The binary calls `register_all`
//! before handing off to `arbvis::run`, which is what makes the
//! `modelweightvis` binary differ from the byte-only `arbvis` binary.
//!
//! The underlying types these plugins build (`arbvis::ArchLayout`,
//! `arbvis::TensorDiffSource`, the `format::*` parsers, etc.) still live
//! in arbvis as pub-exposed items. The full source relocation that would
//! actually lift the heavy deps (`candle-core`, `regex`, `zip`, `half`)
//! out of arbvis is deferred — see the workspace README.

mod diff;
mod layout;
mod leaf;

pub use diff::TensorDiffBuilder;
pub use layout::{ArchLayoutPlugin, MoeDiffLayoutPlugin};
pub use leaf::{ArchRegionsLoader, ArchRegionsRenderer};

use std::sync::Arc;

use arbvis::Registry;

/// Register every tensor-aware plugin on `registry`.
pub fn register_all(registry: &mut Registry) {
    // Tensor-aware diff (.safetensors / .gguf file pairs) — priority 300 so
    // it wins over the JSON / plain-byte fallbacks for matching pairs.
    registry.diffs.push(Arc::new(TensorDiffBuilder));

    // Architectural + MoE-diff layouts. Priority order is encoded on the
    // plugin (200 / 100); `select_layout` sorts by `priority()` descending.
    registry.layouts.push(Arc::new(ArchLayoutPlugin));
    registry.layouts.push(Arc::new(MoeDiffLayoutPlugin));

    // Tile loader+renderer pair for the `"arch"` layout id. The arch
    // layout dispatches to these via `LeafRegistry` lookup at tile time.
    registry.leaf.register_loader(Arc::new(ArchRegionsLoader));
    registry
        .leaf
        .register_renderer(Arc::new(ArchRegionsRenderer));
}
