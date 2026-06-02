//! Tensor-aware CLI surface that flattens [`arbvis::Args`] and adds the
//! four model-side flags (`--moe-diff`, `--finetune` / `--no-finetune`,
//! `--diff-metric`, `--layout`).
//!
//! These four flags used to live on `arbvis::Args` directly, which meant
//! `arbvis --help` advertised them even though they only do anything when
//! a tensor-aware backend is registered (and the byte-only `arbvis`
//! binary errors out at runtime if the user actually passes them). After
//! the split they're owned here, so `arbvis --help` advertises only the
//! flags it can actually run, and `modelweightvis --help` advertises
//! both halves.
//!
//! Clap's `#[command(flatten)]` makes the two structs share a single
//! `Command`, so `--diff` (defined on `arbvis::Args`) and `--finetune`
//! (defined here) can reference each other in `requires =` / `conflicts_with`
//! constraints just as if they lived on the same struct. The cross-crate
//! visibility doesn't matter to clap — `arbvis::Args` derives `Parser` /
//! `FromArgMatches`, which is all the flatten machinery needs.

use std::path::PathBuf;

use arbvis::{DiffMetric, LayoutMode, ModelOpts};
use clap::{Parser, ValueEnum};

/// CLI mirror of [`arbvis::DiffMetric`]. Kept separate from the core type so
/// the clap derive's variant-doc strings and enum-discoverable help text
/// don't leak into the library API.
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
pub enum DiffMetricArg {
    /// Per-tensor RMS-normalized signed delta. Stable across tensors of
    /// different scale; default.
    #[default]
    Rms,
    /// Absolute delta on a log brightness scale. Honest about raw magnitudes.
    AbsLog,
    /// Ternary: identical bytes → black; any change → full saturation.
    Exact,
}

impl From<DiffMetricArg> for DiffMetric {
    fn from(a: DiffMetricArg) -> Self {
        match a {
            DiffMetricArg::Rms => DiffMetric::Rms,
            DiffMetricArg::AbsLog => DiffMetric::AbsLog,
            DiffMetricArg::Exact => DiffMetric::Exact,
        }
    }
}

/// CLI choice for layout strategy. Mirrors [`arbvis::LayoutMode`].
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
pub enum LayoutArg {
    /// Architectural layout if every input is safetensors with detectable
    /// structure; otherwise byte-Hilbert. Default.
    #[default]
    Auto,
    /// Force architectural (structure-aware) layout. Falls back to hilbert if
    /// no input is safetensors.
    Arch,
    /// Force the legacy global-Hilbert layout (1 px = 1 byte). Useful for
    /// non-safetensors inputs and regression-checking the old output.
    Hilbert,
}

impl From<LayoutArg> for LayoutMode {
    fn from(a: LayoutArg) -> Self {
        match a {
            LayoutArg::Auto => LayoutMode::Auto,
            LayoutArg::Arch => LayoutMode::Arch,
            LayoutArg::Hilbert => LayoutMode::Hilbert,
        }
    }
}

/// Tensor-aware visualization built on `arbvis`.
///
/// Flattens [`arbvis::Args`] (the byte-only CLI surface) and adds four
/// tensor-aware flags. Use [`Args::split`] to peel out the inner
/// `arbvis::Args` + `arbvis::ModelOpts` for the call into `arbvis::run`.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[command(flatten)]
    pub arbvis: arbvis::Args,

    /// Visualize an N×N expert-vs-expert diff matrix for each MoE layer of a single
    /// model. MODEL is a local path or hf:// URL. Each cell (i, j) shows the
    /// element-wise diff between expert i and expert j for `gate_proj`, `up_proj`,
    /// and `down_proj`, stacked horizontally; only the upper triangle + diagonal is
    /// rendered (the raw diff is antisymmetric). Currently supports HF-style
    /// per-expert safetensors (Mixtral / Qwen3-MoE / OLMoE / DeepSeek routed
    /// experts); GGUF fused-expert tensors are not yet supported.
    #[arg(
        long,
        value_name = "MODEL",
        conflicts_with_all = ["diff", "files", "file_list", "finetune", "no_finetune", "show_xet_xorbs"]
    )]
    pub moe_diff: Option<PathBuf>,

    /// Force-treat the second --diff argument as a finetune of the first.
    /// In finetune mode, tensors present only on the base side are rendered
    /// as crosshatched grey (informational); anything present only on the
    /// finetune side (or with a mismatched shape) is treated as a contract
    /// violation and aborts the run. Without --finetune / --no-finetune the
    /// relation is auto-detected from the HF model card (`base_model` +
    /// `base_model_relation`) when both args are `hf://` model URLs, and
    /// defaults to non-finetune otherwise.
    #[arg(long, requires = "diff", conflicts_with = "no_finetune")]
    pub finetune: bool,

    /// Force-treat the diff as NOT a finetune (overrides auto-detection).
    /// In this mode, tensors/files present only on one side render as red
    /// (original-only) or green (modified-only) crosshatch.
    #[arg(long = "no-finetune", requires = "diff", conflicts_with = "finetune")]
    pub no_finetune: bool,

    /// How per-element tensor deltas are encoded for visualization
    /// (applies to `--diff` and `--moe-diff`).
    #[arg(long, value_enum, default_value_t = DiffMetricArg::Rms)]
    pub diff_metric: DiffMetricArg,

    /// Layout strategy for arranging tensors on the canvas.
    ///
    /// `auto` (default): structure-aware layout when every input is
    /// safetensors and tensor names look transformer-style; otherwise the
    /// legacy global-Hilbert curve.
    ///
    /// `arch`: force structure-aware layout. Each tensor is rendered at its
    /// natural 2D element shape (1 px = 1 element); transformer blocks stack
    /// vertically with corresponding sub-tensors pixel-aligned across the
    /// stack. Falls back to hilbert if no input is safetensors.
    ///
    /// `hilbert`: force the legacy layout. 1 px = 1 byte along a global
    /// Hilbert curve over the concatenated source bytes. Reproduces the
    /// pre-architectural output for regression checks.
    #[arg(long, value_enum, default_value_t = LayoutArg::Auto)]
    pub layout: LayoutArg,
}

impl Args {
    /// Peel the flattened struct into the two halves `arbvis::run` takes:
    /// the byte-only `arbvis::Args` and the tensor-aware `ModelOpts`.
    pub fn split(self) -> (arbvis::Args, ModelOpts) {
        let opts = ModelOpts {
            moe_diff: self.moe_diff,
            finetune: self.finetune,
            no_finetune: self.no_finetune,
            diff_metric: self.diff_metric.into(),
            layout_mode: self.layout.into(),
        };
        (self.arbvis, opts)
    }
}
