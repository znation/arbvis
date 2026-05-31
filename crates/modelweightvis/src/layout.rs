//! Tensor-aware `LayoutPlugin` impls: regular arch + MoE-diff arch.
//!
//! These build `arbvis::ArchLayout` (which still lives in arbvis as a
//! pub-exposed type; step 12e's full source relocation is deferred). The
//! plugins themselves moved out so `arbvis` no longer references them — the
//! arbvis binary's default registry has no architectural layout and falls
//! back to byte-Hilbert for tensor files.

use arbvis::{
    ArchLayout, LayoutBuildCtx, LayoutMode, LayoutPlugin, LayoutShape, ModelInfo, MoeCell,
};

/// Architectural plugin — applies when sources carry safetensors metadata
/// and `--layout` doesn't force hilbert. Build returns `None` if no
/// transformer-style structure is detectable.
pub struct ArchLayoutPlugin;

impl ArchLayoutPlugin {
    /// In non-diff mode every source must be safetensors (otherwise the user
    /// has explicitly mixed in non-tensor inputs they'd expect to see). In
    /// diff mode it's enough that any source carries safetensors info: the
    /// typical case is a model-repo diff where the tensor sources are the
    /// point and tokenizer/config diffs are incidental.
    fn eligible(ctx: &LayoutBuildCtx<'_>) -> bool {
        if matches!(ctx.mode, LayoutMode::Hilbert) {
            return false;
        }
        let all = !ctx.sources.is_empty()
            && ctx
                .sources
                .iter()
                .all(|s| s.extensions.get::<ModelInfo>().is_some());
        let any = ctx
            .sources
            .iter()
            .any(|s| s.extensions.get::<ModelInfo>().is_some());
        if ctx.diff_mode {
            any
        } else {
            all
        }
    }
}

impl LayoutPlugin for ArchLayoutPlugin {
    fn id(&self) -> &'static str {
        "arch"
    }
    fn priority(&self) -> i32 {
        100
    }
    fn applicable(&self, ctx: &LayoutBuildCtx<'_>) -> bool {
        Self::eligible(ctx)
    }
    fn build(&self, ctx: &LayoutBuildCtx<'_>) -> Option<Box<dyn LayoutShape>> {
        let arch = ArchLayout::try_build(ctx.sources, ctx.cumulative_offsets, ctx.metas)?;
        // Diff-mode info note: surface tensor sources that don't carry
        // safetensors info (e.g. tokenizer.json file diffs) — they won't
        // appear on the arch canvas.
        if ctx.diff_mode {
            let all = !ctx.sources.is_empty()
                && ctx
                    .sources
                    .iter()
                    .all(|s| s.extensions.get::<ModelInfo>().is_some());
            if !all {
                let skipped = ctx
                    .sources
                    .iter()
                    .filter(|s| s.extensions.get::<ModelInfo>().is_none())
                    .count();
                log::info!(
                    "arch layout: {skipped} non-safetensors diff source(s) will not appear on the arch canvas (file-level diffs are only rendered in --layout hilbert)"
                );
            }
        }
        Some(Box::new(arch))
    }
}

/// MoE-diff plugin — applies when any source carries a `MoeCell` tag (only
/// emitted by the MoE-diff source preparation in arbvis, so this fork
/// can't collide with a normal arch run).
pub struct MoeDiffLayoutPlugin;

impl LayoutPlugin for MoeDiffLayoutPlugin {
    fn id(&self) -> &'static str {
        "moe-diff"
    }
    fn priority(&self) -> i32 {
        200
    }
    fn applicable(&self, ctx: &LayoutBuildCtx<'_>) -> bool {
        if matches!(ctx.mode, LayoutMode::Hilbert) {
            return false;
        }
        ctx.sources
            .iter()
            .any(|s| s.extensions.get::<MoeCell>().is_some())
    }
    fn build(&self, ctx: &LayoutBuildCtx<'_>) -> Option<Box<dyn LayoutShape>> {
        ArchLayout::try_build_moe_diff(ctx.sources, ctx.cumulative_offsets)
            .map(|l| Box::new(l) as Box<dyn LayoutShape>)
    }
}
