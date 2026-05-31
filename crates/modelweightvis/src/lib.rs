//! Tensor-format-aware visualization built on `arbvis`.
//!
//! Today this crate is a thin "extra plugins" layer: `register_all` adds the
//! tensor-aware layouts (arch, MoE-diff), the tensor-diff source builder,
//! and the `"arch"` leaf loader+renderer pair on top of an
//! `arbvis::Registry::with_defaults()`. The plugin TYPES still live in the
//! arbvis crate (step 12e relocates them along with the heavy deps
//! `candle-core`, `regex`, `zip`, `half`).
//!
//! The modelweightvis binary calls this before handing off to
//! `arbvis::run`, which is how the byte-only `arbvis` binary and the
//! tensor-aware `modelweightvis` binary diverge.

use std::sync::Arc;

use arbvis::{
    ArchLayoutPlugin, ArchRegionsLoader, ArchRegionsRenderer, MoeDiffLayoutPlugin, Registry,
    TensorDiffBuilder,
};

/// Register every tensor-aware plugin on `registry`. Idempotent in the
/// "every-id-replaces" sense: re-registering an id replaces the prior
/// `LeafLoader`/`LeafRenderer` for that id, and re-pushing a
/// `LayoutPlugin`/`DiffSourceBuilder` just duplicates it (which the
/// priority loop tolerates, but you shouldn't do twice in the same run).
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
