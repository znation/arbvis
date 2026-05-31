//! Format-agnostic tensor / model types.
//!
//! `TensorMeta`, `ModelInfo`, `DiffFill`, and `DiffMetric` are shared across
//! every model format (safetensors, GGUF, future). They carry no format-
//! specific state — `file_start`/`file_end` are absolute byte ranges in the
//! underlying file so the renderer can treat both formats identically.

use image::Rgb;

use super::dtype::Dtype;
use super::SourceFormat;

// DiffFill and DiffMetric moved to arbvis (byte-foundation). The
// `format::DiffMetric` / `format::DiffFill` names that the per-format
// parsers use are now re-exported from `format/mod.rs`.

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
    /// AWQ / GPTQ packed-int sidecar references. `Some(...)` for tensors
    /// whose [`Dtype`] is one of `IntNPacked`; `None` for plain / Block
    /// dtypes. The qweight byte range lives in `file_start`/`file_end`; the
    /// sidecar struct carries the parallel byte ranges and dtypes for the
    /// `scales` and `qzeros` tensors needed to dequantise.
    #[allow(dead_code)]
    pub packed_sidecars: Option<PackedSidecars>,
}

/// Byte ranges and dtypes for the `scales` / `qzeros` sidecar tensors that
/// accompany an AWQ/GPTQ-style packed-int `qweight` tensor in the same file.
/// Populated by [`crate::format::safetensors::fuse_packed_quant_triples`].
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PackedSidecars {
    pub scales_start: u64,
    pub scales_end: u64,
    pub scales_dtype: Dtype,
    /// `None` for AWQ symmetric quants without zero-points. `Some` for
    /// GPTQ / EXL2 / AWQ-with-qzeros.
    pub zeros_start: Option<u64>,
    pub zeros_end: Option<u64>,
    pub zeros_dtype: Dtype,
    /// Number of output columns in the unpacked tensor — needed when
    /// indexing per-element across rows.
    pub cols: u32,
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
