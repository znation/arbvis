//! Format-agnostic tensor / model types.
//!
//! `TensorMeta`, `ModelInfo`, `DiffFill`, and `DiffMetric` are shared across
//! every model format (safetensors, GGUF, future). They carry no format-
//! specific state — `file_start`/`file_end` are absolute byte ranges in the
//! underlying file so the renderer can treat both formats identically.

use image::Rgb;

use super::dtype::Dtype;
use super::SourceFormat;

/// Crosshatched-fill kind for unmatched-tensor / unmatched-file regions in diff mode.
///
/// `Grey` is for the "expected" finetune case (tensor present in base but
/// absent from finetune — e.g. a base model's vision tower vs a text-only
/// finetune). `Red` and `Green` flag genuine structural divergence between two
/// otherwise-compatible files. Each region is filled with diagonal crosshatch
/// lines so it's instantly distinguishable from a real signed-diff color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiffFill {
    /// Tensor/file present in original only, finetune mode (informational).
    Grey,
    /// Tensor/file present in original only, non-finetune mode (divergence).
    Red,
    /// Tensor/file present in modified only, non-finetune mode (divergence).
    Green,
}

impl DiffFill {
    /// `(stripe, base)` colors for the crosshatch pattern. `stripe` is the
    /// foreground diagonal line color; `base` is the fill behind it.
    pub fn colors(self) -> (Rgb<u8>, Rgb<u8>) {
        match self {
            DiffFill::Grey => (Rgb([80, 80, 80]), Rgb([160, 160, 160])),
            DiffFill::Red => (Rgb([120, 0, 0]), Rgb([220, 40, 40])),
            DiffFill::Green => (Rgb([0, 120, 0]), Rgb([40, 220, 40])),
        }
    }
}

/// Selects how per-element diffs are encoded for visualization.
///
/// All three preserve the sign convention (green = grew, red = shrank,
/// black = no change, white = NaN/Inf in either side). They differ in how
/// brightness is computed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum DiffMetric {
    /// Per-tensor RMS-normalized signed delta:
    ///   `signed = clamp(delta / (K_RMS_SAT * rms(orig)), -1, 1)`
    ///
    /// Reads as "how many tensor-stddevs did this element move." Stable
    /// across tensors regardless of weight scale, and doesn't blow up on
    /// small base weights the way per-element `(m-o)/|o|` does. Requires a
    /// per-tensor scale (computed at setup via sampling). Default.
    #[default]
    Rms,
    /// Absolute delta on a log brightness scale, no normalization:
    ///   `signed = sign(delta) * clamp((log10(|delta|) - log10(ABS_LOG_MIN))
    ///             / (log10(ABS_LOG_MAX) - log10(ABS_LOG_MIN)), 0, 1)`
    ///
    /// Honest about raw magnitudes; no per-tensor pre-pass. Tensors with
    /// naturally larger weights look hotter even if untouched relative to
    /// their scale.
    AbsLog,
    /// Ternary: identical bytes → black, any change → full saturation in the
    /// direction of the change. Best diagnostic for LoRA-merge patterns —
    /// every untouched element is pitch black, touched elements glow.
    Exact,
}

/// Saturation threshold for `DiffMetric::Rms`: an element whose delta equals
/// `K_RMS_SAT * rms(orig)` paints at full brightness. 0.5 means "half a
/// tensor-stddev is fully saturated"; a typical LoRA-merge moves median
/// elements by ~0.005 stddevs (subtle), an aggressive full-finetune by ~0.05
/// stddevs (clearly visible), an uncorrelated init by ~1 stddev (saturated).
pub const K_RMS_SAT: f32 = 0.5;

/// Floor for `rms(orig)` in `DiffMetric::Rms`, used to avoid divide-by-zero
/// on all-zero tensors and to cap sensitivity on near-zero tensors.
pub const RMS_FLOOR: f32 = 1e-6;

/// Log-brightness range endpoints for `DiffMetric::AbsLog`. Deltas with
/// `|delta| < ABS_LOG_MIN` paint black; `|delta| >= ABS_LOG_MAX` saturate.
/// The span covers the typical range of useful bf16 finetune deltas.
pub const ABS_LOG_MIN: f32 = 1e-6;
pub const ABS_LOG_MAX: f32 = 1e-1;

/// Per-tensor metadata, format-agnostic.
///
/// For safetensors: built from the JSON header at file open. For GGUF: built
/// from the tensor info table. `file_start`/`file_end` are absolute byte
/// offsets into the underlying file; the renderer does not need to know which
/// format produced them.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Absolute byte positions in the file [start, end)
    pub file_start: u64,
    pub file_end: u64,
}

impl TensorMeta {
    /// 2D pixel-grid shape used by the architectural layout. Distinct from
    /// `shape`, which is the raw tensor shape: this collapses to exactly two
    /// dimensions so a tensor occupies a flat rectangle on the canvas.
    ///
    /// - 0-D (scalar) → `(1, 1)`
    /// - 1-D `(n)` → `(1, n)` (one-pixel-tall strip)
    /// - 2-D `(r, c)` → `(r, c)` (preserved)
    /// - ≥3-D `(a, b, c, …)` → `(a, b*c*…)` (last dims collapsed into the
    ///   column axis). The element index within the resulting rect uses
    ///   row-major order, which matches the byte order in the underlying
    ///   file: element `(row, col)` lives at the logical position
    ///   `row*cols + col`.
    pub fn element_shape(&self) -> (u64, u64) {
        match self.shape.len() {
            0 => (1, 1),
            1 => (1, self.shape[0]),
            2 => (self.shape[0], self.shape[1]),
            _ => {
                let rows = self.shape[0];
                let cols: u64 = self.shape[1..].iter().product();
                (rows, cols)
            }
        }
    }

    pub fn label(&self) -> String {
        let shape_str: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        format!(
            "{} [{}, {}]",
            self.name,
            self.dtype.label(),
            shape_str.join("×")
        )
    }
}

/// Format-aware metadata attached to a `Source` whose underlying file is a
/// recognised model format. The tensor list drives the architectural layout;
/// `color_ranges` drives the legacy Hilbert dtype-mode coloring.
///
/// `format` records which parser produced this — read by cross-format diff
/// matching to canonicalise tensor names before pairing.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    #[allow(dead_code)]
    pub format: SourceFormat,
    pub tensors: Vec<TensorMeta>,
    pub color_ranges: Vec<(u64, u64, Rgb<u8>)>,
}
