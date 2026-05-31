//! Format-agnostic dtype: plain (safetensors-style, fixed bytes/element) +
//! block-quantized (GGUF-style, fixed bytes/block, fixed elements/block).
//!
//! `TensorElementReader` is the single per-element decode path the renderer
//! and diff math both use. It hides whether an element comes from a plain
//! little-endian read or from a block dequant: callers ask for "element K"
//! and get an `f32`.

use std::borrow::Cow;

use candle_core::quantized::gguf_file::TensorInfo;
use candle_core::quantized::GgmlDType;
use candle_core::CpuStorage;
use image::Rgb;

use super::types::{ABS_LOG_MAX, ABS_LOG_MIN, K_RMS_SAT, RMS_FLOOR};
use super::DiffMetric;

/// How an element is addressed within a contiguous tensor byte buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementStride {
    /// Fixed bytes per element (every safetensors dtype and the unquantised
    /// GGUF dtypes F32/F16/BF16). Element `k` lives at byte offset
    /// `k * bytes_per_element`.
    Fixed(usize),
    /// Block-quantized: each block of `block_bytes` covers `block_elements`
    /// logical elements. Element `k` lives in block `k / block_elements` at
    /// in-block position `k % block_elements`; the block bytes start at
    /// `(k / block_elements) * block_bytes`.
    Block {
        block_bytes: usize,
        block_elements: usize,
    },
    /// AWQ/GPTQ-style bit-packed integers with out-of-band scales/zeros.
    ///
    /// `bits` low-bits-per-element are packed into a `pack_dtype_bytes`-wide
    /// integer (typically int32, so `pack_dtype_bytes = 4`). Every
    /// `group_size` consecutive elements share one scale (and, for GPTQ,
    /// one zero-point) drawn from sidecar tensors. The dequant formula is
    /// `(q - zero) * scale` for GPTQ, `q * scale` for AWQ.
    ///
    /// Unlike `Block`, the scale/zero data is NOT inline with the quants —
    /// it lives in separately-named tensors. The
    /// [`TensorElementReader::with_sidecars`] builder threads those buffers
    /// through. Element `k` resolves to packed slot
    /// `k / (pack_dtype_bytes * 8 / bits)` with intra-slot position
    /// `k % (pack_dtype_bytes * 8 / bits)`.
    Packed {
        bits: u8,
        pack_dtype_bytes: u8,
        group_size: u32,
    },
}

impl ElementStride {
    /// Bytes spanned by `n` consecutive elements starting from a block
    /// boundary. For fixed: `n * bytes_per_element`. For block: rounded up
    /// to the nearest whole block. For packed: rounded up to the nearest
    /// whole pack-slot (e.g. 8 elements for int4-in-int32).
    #[allow(dead_code)]
    pub fn bytes_for_elements(self, n: usize) -> usize {
        match self {
            ElementStride::Fixed(b) => n * b,
            ElementStride::Block {
                block_bytes,
                block_elements,
            } => {
                if block_elements == 0 {
                    0
                } else {
                    n.div_ceil(block_elements) * block_bytes
                }
            }
            ElementStride::Packed {
                bits,
                pack_dtype_bytes,
                ..
            } => {
                if bits == 0 {
                    0
                } else {
                    let elems_per_slot = (pack_dtype_bytes as usize * 8) / bits as usize;
                    if elems_per_slot == 0 {
                        0
                    } else {
                        n.div_ceil(elems_per_slot) * pack_dtype_bytes as usize
                    }
                }
            }
        }
    }

    /// Bytes spanned by one full row of `cols` elements in a row-major tensor.
    /// For fixed: `cols * bytes_per_element`. For block/packed: the whole-row
    /// stride, assuming `cols` is a multiple of the block/slot element count
    /// (true for canonical GGUF and AWQ/GPTQ layouts). Used to address the
    /// start of an arbitrary row without re-deriving the packing math.
    pub fn bytes_per_row(self, cols: u64) -> u64 {
        match self {
            ElementStride::Fixed(b) => cols * b as u64,
            ElementStride::Block {
                block_bytes,
                block_elements,
            } => {
                if block_elements == 0 {
                    0
                } else {
                    (cols / block_elements as u64) * block_bytes as u64
                }
            }
            ElementStride::Packed {
                bits,
                pack_dtype_bytes,
                ..
            } => {
                if bits == 0 {
                    return 0;
                }
                let elems_per_slot = (pack_dtype_bytes as u64 * 8) / bits as u64;
                cols.checked_div(elems_per_slot)
                    .map(|n| n * pack_dtype_bytes as u64)
                    .unwrap_or(0)
            }
        }
    }
}

/// Tensor element data type.
///
/// Covers safetensors plain dtypes (fixed bytes/element) plus the GGUF
/// block-quantized variants. `element_size()` returns 1 for quantized types
/// — callers that need real stride information must use `stride()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    // Plain types (safetensors + GGUF F32/F16/BF16).
    F64,
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    Unknown,
    // GGUF block-quantized.
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    // AWQ/GPTQ/EXL2-style bit-packed quants. The element bytes live
    // inline; the per-group `scales` and `zeros` tensors live as
    // sidecar tensors named alongside the quant tensor (see
    // [`crate::format::safetensors::fuse_packed_quant_triples`]).
    /// 4 bits per element, packed 8-per-int32 with f16/f32 scales (+ zeros).
    Int4Packed,
    /// 3 bits per element, packed ~10-per-int32 with f16 scales (+ zeros).
    /// Used by EXL2 mixed-precision layouts. The 4-bit case dominates the
    /// HF corpus today; 3-bit is held for future EXL2 / GPTQ detection.
    #[allow(dead_code)]
    Int3Packed,
    /// 8 bits per element, packed 4-per-int32 with f16 scales (+ zeros).
    /// Rarer on the Hub but reserved for future format detection.
    #[allow(dead_code)]
    Int8Packed,
}

impl Dtype {
    /// Parse a safetensors header dtype string (`"F32"`, `"BF16"`, …).
    /// Returns `Dtype::Unknown` for anything unrecognised — safetensors
    /// itself can ship custom dtype strings, so we don't bail on unknowns.
    pub fn from_safetensors_str(s: &str) -> Self {
        match s {
            "F64" => Dtype::F64,
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "F8_E4M3" | "F8_E4M3FN" | "F8_E4M3FNUZ" => Dtype::F8E4M3,
            "F8_E5M2" | "F8_E5M2FNUZ" => Dtype::F8E5M2,
            "I64" => Dtype::I64,
            "I32" => Dtype::I32,
            "I16" => Dtype::I16,
            "I8" => Dtype::I8,
            "U64" => Dtype::U64,
            "U32" => Dtype::U32,
            "U16" => Dtype::U16,
            "U8" => Dtype::U8,
            "BOOL" => Dtype::Bool,
            _ => Dtype::Unknown,
        }
    }

    /// Translate a candle-core `GgmlDType` (from a parsed GGUF tensor info)
    /// into the arbvis `Dtype`. Non-quantized GGUF dtypes (`F32`/`F16`/`BF16`)
    /// map onto the plain safetensors variants so the renderer treats them
    /// identically.
    pub fn from_ggml(ty: GgmlDType) -> Self {
        match ty {
            GgmlDType::F32 => Dtype::F32,
            GgmlDType::F16 => Dtype::F16,
            GgmlDType::BF16 => Dtype::BF16,
            GgmlDType::Q4_0 => Dtype::Q4_0,
            GgmlDType::Q4_1 => Dtype::Q4_1,
            GgmlDType::Q5_0 => Dtype::Q5_0,
            GgmlDType::Q5_1 => Dtype::Q5_1,
            GgmlDType::Q8_0 => Dtype::Q8_0,
            GgmlDType::Q8_1 => Dtype::Q8_1,
            GgmlDType::Q2K => Dtype::Q2K,
            GgmlDType::Q3K => Dtype::Q3K,
            GgmlDType::Q4K => Dtype::Q4K,
            GgmlDType::Q5K => Dtype::Q5K,
            GgmlDType::Q6K => Dtype::Q6K,
            GgmlDType::Q8K => Dtype::Q8K,
        }
    }

    /// Inverse of [`Dtype::from_ggml`] for types that have a candle-side
    /// representation. Returns `None` for safetensors-only dtypes like F8/I*.
    pub fn to_ggml(self) -> Option<GgmlDType> {
        Some(match self {
            Dtype::F32 => GgmlDType::F32,
            Dtype::F16 => GgmlDType::F16,
            Dtype::BF16 => GgmlDType::BF16,
            Dtype::Q4_0 => GgmlDType::Q4_0,
            Dtype::Q4_1 => GgmlDType::Q4_1,
            Dtype::Q5_0 => GgmlDType::Q5_0,
            Dtype::Q5_1 => GgmlDType::Q5_1,
            Dtype::Q8_0 => GgmlDType::Q8_0,
            Dtype::Q8_1 => GgmlDType::Q8_1,
            Dtype::Q2K => GgmlDType::Q2K,
            Dtype::Q3K => GgmlDType::Q3K,
            Dtype::Q4K => GgmlDType::Q4K,
            Dtype::Q5K => GgmlDType::Q5K,
            Dtype::Q6K => GgmlDType::Q6K,
            Dtype::Q8K => GgmlDType::Q8K,
            _ => return None,
        })
    }

    /// Bytes per logical element, or `1` for block-quantized dtypes.
    ///
    /// Kept for legacy call sites that compute byte ranges as
    /// `count * dtype.element_size()` and have no way to address blocks.
    /// New code should use [`Dtype::stride`].
    pub fn element_size(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::F8E4M3
            | Dtype::F8E5M2
            | Dtype::I8
            | Dtype::U8
            | Dtype::Bool
            | Dtype::Unknown => 1,
            // Block-quantised: every quantised dtype is sub-byte-per-element
            // (Q4_0 = 0.5625 bytes/elem). Return 1 so legacy callers that do
            // `n * element_size()` don't divide by zero; the real stride lives
            // in `Dtype::stride()`.
            Dtype::Q4_0
            | Dtype::Q4_1
            | Dtype::Q5_0
            | Dtype::Q5_1
            | Dtype::Q8_0
            | Dtype::Q8_1
            | Dtype::Q2K
            | Dtype::Q3K
            | Dtype::Q4K
            | Dtype::Q5K
            | Dtype::Q6K
            | Dtype::Q8K
            // Same rationale for packed-int dtypes — the real stride lives
            // in `Dtype::stride()`. (Int4Packed is 0.5 bytes/elem, etc.)
            | Dtype::Int4Packed
            | Dtype::Int3Packed
            | Dtype::Int8Packed => 1,
        }
    }

    /// Real stride information — `Fixed(n)` for plain dtypes, `Block { … }`
    /// for the GGUF quantised types, `Packed { … }` for AWQ/GPTQ-style
    /// bit-packed ints. AWQ/GPTQ canonical layout is int4 packed into int32
    /// with 128-element groups; sub-byte int3/int8 use the same packing
    /// dtype but different bit widths.
    pub fn stride(self) -> ElementStride {
        match self {
            Dtype::Int4Packed => ElementStride::Packed {
                bits: 4,
                pack_dtype_bytes: 4,
                group_size: 128,
            },
            Dtype::Int3Packed => ElementStride::Packed {
                bits: 3,
                pack_dtype_bytes: 4,
                group_size: 128,
            },
            Dtype::Int8Packed => ElementStride::Packed {
                bits: 8,
                pack_dtype_bytes: 4,
                group_size: 128,
            },
            _ => {
                if let Some(g) = self.to_ggml() {
                    // For F32/F16/BF16, type_size == 4/2/2 and block_size == 1,
                    // so this collapses to `Fixed(type_size)`.
                    let bytes = g.type_size();
                    let elems = g.block_size();
                    if elems <= 1 {
                        ElementStride::Fixed(bytes)
                    } else {
                        ElementStride::Block {
                            block_bytes: bytes,
                            block_elements: elems,
                        }
                    }
                } else {
                    ElementStride::Fixed(self.element_size())
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            Dtype::Q4_0
                | Dtype::Q4_1
                | Dtype::Q5_0
                | Dtype::Q5_1
                | Dtype::Q8_0
                | Dtype::Q8_1
                | Dtype::Q2K
                | Dtype::Q3K
                | Dtype::Q4K
                | Dtype::Q5K
                | Dtype::Q6K
                | Dtype::Q8K
                | Dtype::Int4Packed
                | Dtype::Int3Packed
                | Dtype::Int8Packed
        )
    }

    /// True for AWQ/GPTQ-style packed integers whose dequant requires
    /// out-of-band scales (and, for GPTQ, zero-points) — see
    /// [`ElementStride::Packed`].
    #[allow(dead_code)]
    pub fn is_packed(self) -> bool {
        matches!(
            self,
            Dtype::Int4Packed | Dtype::Int3Packed | Dtype::Int8Packed
        )
    }

    /// Display color for `build_color_ranges` and the architectural arch
    /// legend. Plain dtypes keep their original safetensors palette; each
    /// quantised type gets a distinct cool hue so a tile of mixed-precision
    /// GGUF tensors reads as a striated rainbow.
    pub fn to_color(self) -> Rgb<u8> {
        match self {
            Dtype::F32 => Rgb([255, 120, 50]),
            Dtype::F16 => Rgb([255, 210, 60]),
            Dtype::BF16 => Rgb([180, 255, 60]),
            Dtype::F64 => Rgb([255, 50, 50]),
            Dtype::F8E4M3 => Rgb([50, 220, 255]),
            Dtype::F8E5M2 => Rgb([50, 255, 200]),
            Dtype::I8 | Dtype::U8 => Rgb([60, 60, 255]),
            Dtype::I16 | Dtype::U16 => Rgb([130, 60, 255]),
            Dtype::I32 | Dtype::U32 | Dtype::I64 | Dtype::U64 | Dtype::Bool => Rgb([220, 60, 255]),
            Dtype::Unknown => Rgb([0, 0, 0]),
            // GGUF quantised palette. Stepped through teal→indigo so the
            // visual band scales with precision (Q2 darkest, Q8 brightest).
            Dtype::Q2K => Rgb([0, 80, 110]),
            Dtype::Q3K => Rgb([0, 110, 140]),
            Dtype::Q4_0 => Rgb([0, 140, 170]),
            Dtype::Q4_1 => Rgb([0, 160, 190]),
            Dtype::Q4K => Rgb([0, 180, 200]),
            Dtype::Q5_0 => Rgb([60, 170, 220]),
            Dtype::Q5_1 => Rgb([80, 180, 230]),
            Dtype::Q5K => Rgb([100, 190, 240]),
            Dtype::Q6K => Rgb([130, 180, 240]),
            Dtype::Q8_0 => Rgb([160, 170, 240]),
            Dtype::Q8_1 => Rgb([180, 160, 240]),
            Dtype::Q8K => Rgb([200, 150, 240]),
            // AWQ/GPTQ packed-int palette. Magenta family — visually distinct
            // from the cool GGUF range so a mixed AWQ + base diff reads
            // unambiguously. Stepped by bit width: int3 darkest, int8 brightest.
            Dtype::Int3Packed => Rgb([160, 40, 180]),
            Dtype::Int4Packed => Rgb([200, 60, 200]),
            Dtype::Int8Packed => Rgb([230, 100, 210]),
        }
    }

    /// Short string used in tensor labels (legend, tooltips). Matches the
    /// safetensors header convention for plain dtypes; for quantised dtypes
    /// uses the canonical GGML name.
    pub fn label(self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::F64 => "F64",
            Dtype::F8E4M3 => "F8E4M3",
            Dtype::F8E5M2 => "F8E5M2",
            Dtype::I8 => "I8",
            Dtype::U8 => "U8",
            Dtype::I16 => "I16",
            Dtype::U16 => "U16",
            Dtype::I32 => "I32",
            Dtype::U32 => "U32",
            Dtype::I64 => "I64",
            Dtype::U64 => "U64",
            Dtype::Bool => "BOOL",
            Dtype::Unknown => "?",
            Dtype::Q4_0 => "Q4_0",
            Dtype::Q4_1 => "Q4_1",
            Dtype::Q5_0 => "Q5_0",
            Dtype::Q5_1 => "Q5_1",
            Dtype::Q8_0 => "Q8_0",
            Dtype::Q8_1 => "Q8_1",
            Dtype::Q2K => "Q2K",
            Dtype::Q3K => "Q3K",
            Dtype::Q4K => "Q4K",
            Dtype::Q5K => "Q5K",
            Dtype::Q6K => "Q6K",
            Dtype::Q8K => "Q8K",
            Dtype::Int3Packed => "I3_pack",
            Dtype::Int4Packed => "I4_pack",
            Dtype::Int8Packed => "I8_pack",
        }
    }

    /// Compute the signed diff between matched elements, returning one u8
    /// per element pair.
    ///
    /// `self` is the dtype of the `orig` buffer; `mod_dtype` is the dtype of
    /// the `mod_` buffer (they may differ — e.g. a cross-format diff between
    /// a safetensors F16 base and a GGUF Q4_K finetune). `scale_orig` is the
    /// per-tensor scale (RMS of `orig`) used by `DiffMetric::Rms`.
    ///
    /// `orig_start_elem`/`mod_start_elem` let the caller supply a buffer
    /// that begins on a block boundary (for quantised tensors) but skip
    /// `orig_start_elem` elements into it before pairing. For safetensors
    /// fixed-stride buffers pass `0`.
    ///
    /// Encoding: 127 = no change, 128–254 = increased, 0–126 = decreased,
    /// 255 = non-finite.
    pub fn diff_to_u8(
        self,
        orig: &[u8],
        orig_start_elem: usize,
        mod_dtype: Dtype,
        mod_: &[u8],
        mod_start_elem: usize,
        metric: DiffMetric,
        scale_orig: f32,
        elem_count: usize,
    ) -> Vec<u8> {
        let rms_denom = (K_RMS_SAT * scale_orig.max(RMS_FLOOR)).max(f32::MIN_POSITIVE);
        let log_min = ABS_LOG_MIN.log10();
        let log_max = ABS_LOG_MAX.log10();
        let mut o_reader = TensorElementReader::new(self, orig);
        let mut m_reader = TensorElementReader::new(mod_dtype, mod_);
        let mut out = Vec::with_capacity(elem_count);
        for k in 0..elem_count {
            let o = o_reader.element(orig_start_elem + k);
            let m = m_reader.element(mod_start_elem + k);
            if !o.is_finite() || !m.is_finite() {
                out.push(255u8);
                continue;
            }
            let delta = m - o;
            let signed = match metric {
                DiffMetric::Rms => (delta / rms_denom).clamp(-1.0, 1.0),
                DiffMetric::AbsLog => {
                    let abs_d = delta.abs();
                    if abs_d <= ABS_LOG_MIN {
                        0.0
                    } else {
                        let norm =
                            ((abs_d.log10() - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
                        if delta >= 0.0 {
                            norm
                        } else {
                            -norm
                        }
                    }
                }
                DiffMetric::Exact => {
                    if delta == 0.0 {
                        0.0
                    } else if delta > 0.0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
            };
            let brightness = (signed.abs() * 127.0).round() as u8;
            let byte = if signed >= 0.0 {
                127u8.saturating_add(brightness)
            } else {
                127u8.saturating_sub(brightness)
            };
            out.push(byte);
        }
        out
    }
}

/// Decode one plain (non-quantized) element from a little-endian byte slice.
/// Panics for quantized dtypes — callers must use [`TensorElementReader`].
pub fn decode_element(dtype: Dtype, bytes: &[u8]) -> f32 {
    match dtype {
        Dtype::F32 => f32::from_le_bytes(bytes.try_into().unwrap()),
        Dtype::F16 => half::f16::from_le_bytes(bytes.try_into().unwrap()).to_f32(),
        Dtype::BF16 => half::bf16::from_le_bytes(bytes.try_into().unwrap()).to_f32(),
        Dtype::F64 => f64::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::I8 => (bytes[0] as i8) as f32,
        Dtype::U8 | Dtype::Bool => bytes[0] as f32,
        Dtype::I16 => i16::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::U16 => u16::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::I32 => i32::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::U32 => u32::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::I64 => i64::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::U64 => u64::from_le_bytes(bytes.try_into().unwrap()) as f32,
        Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::Unknown => bytes[0] as f32,
        // Quantized dtypes need block context — see TensorElementReader.
        Dtype::Q4_0
        | Dtype::Q4_1
        | Dtype::Q5_0
        | Dtype::Q5_1
        | Dtype::Q8_0
        | Dtype::Q8_1
        | Dtype::Q2K
        | Dtype::Q3K
        | Dtype::Q4K
        | Dtype::Q5K
        | Dtype::Q6K
        | Dtype::Q8K
        // Packed dtypes need sidecar context — see TensorElementReader::with_sidecars.
        | Dtype::Int4Packed
        | Dtype::Int3Packed
        | Dtype::Int8Packed => {
            panic!(
                "decode_element: quantized dtype {:?} needs TensorElementReader",
                dtype
            )
        }
    }
}

/// Decode one element from an AWQ/GPTQ-style packed-int tensor.
///
/// `qweight` is the row-major byte buffer of packed ints (typically int32);
/// `sc` carries the matching scales / qzeros byte buffers. Element `k` is
/// the logical (unpacked) element index — row-major across the unpacked
/// `[rows, cols]` shape, where `cols = sc.cols`.
///
/// Layout assumed (GPTQ / AWQ canonical):
///   - quant slots are arranged row-major across columns, so the slot
///     containing logical column `c` of row `r` lives at
///     `(r * (cols / elems_per_slot) + c / elems_per_slot)` packed ints
///   - within a slot, the `bits`-wide field for in-slot index `i` is at
///     bit positions `[i*bits, i*bits + bits)`
///   - scales are `[rows / group_size, cols]` (one scale per (group, col))
///   - qzeros mirror scales but bit-packed the same way as qweight
///
/// Returns NaN on any out-of-range access (the renderer treats that as a
/// "padding / partial fetch" pixel).
fn packed_element(
    qweight: &[u8],
    sc: PackedSidecarRefs<'_>,
    k: usize,
    bits: u8,
    pack_dtype_bytes: u8,
    group_size: usize,
) -> f32 {
    if bits == 0 || pack_dtype_bytes == 0 || sc.cols == 0 {
        return f32::NAN;
    }
    let bits = bits as usize;
    let slot_bytes = pack_dtype_bytes as usize;
    let elems_per_slot = (slot_bytes * 8) / bits;
    if elems_per_slot == 0 {
        return f32::NAN;
    }
    let cols = sc.cols as usize;
    if cols == 0 || !cols.is_multiple_of(elems_per_slot) {
        return f32::NAN;
    }
    let row = k / cols;
    let col = k % cols;
    let slots_per_row = cols / elems_per_slot;
    let slot_idx = row * slots_per_row + col / elems_per_slot;
    let in_slot = col % elems_per_slot;

    // Read packed int (little-endian) from qweight.
    let off = slot_idx * slot_bytes;
    if off + slot_bytes > qweight.len() {
        return f32::NAN;
    }
    let packed = read_u32_le(&qweight[off..off + slot_bytes]);
    let mask: u32 = if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let q = ((packed >> (in_slot * bits)) & mask) as i32;

    // Group index and scale lookup.
    let group_idx = row.checked_div(group_size).unwrap_or(0);
    let scale_elem_idx = group_idx * cols + col;
    let scale = read_scalar(sc.scales, sc.scales_dtype, scale_elem_idx);
    if !scale.is_finite() {
        return f32::NAN;
    }

    // Zero lookup. For AWQ asymmetric: zeros buffer is packed the same way
    // as qweight; for symmetric variants (no zeros) treat zero as 0.
    let zero = if let Some(zb) = sc.zeros {
        if sc.zeros_dtype == Dtype::F16 || sc.zeros_dtype == Dtype::F32 {
            // EXL2 stores zeros as floats; same indexing as scales.
            let z = read_scalar(zb, sc.zeros_dtype, scale_elem_idx);
            if !z.is_finite() {
                return f32::NAN;
            }
            z
        } else {
            // GPTQ/AWQ store zeros bit-packed identically to qweight.
            let zslot = group_idx * slots_per_row + col / elems_per_slot;
            let zoff = zslot * slot_bytes;
            if zoff + slot_bytes > zb.len() {
                return f32::NAN;
            }
            let zp = read_u32_le(&zb[zoff..zoff + slot_bytes]);
            let z = ((zp >> (in_slot * bits)) & mask) as i32;
            z as f32
        }
    } else {
        0.0
    };

    ((q as f32) - zero) * scale
}

fn read_u32_le(b: &[u8]) -> u32 {
    let mut v = 0u32;
    for (i, &x) in b.iter().take(4).enumerate() {
        v |= (x as u32) << (i * 8);
    }
    v
}

fn read_scalar(bytes: &[u8], dtype: Dtype, elem_idx: usize) -> f32 {
    let bpe = dtype.element_size();
    let off = elem_idx * bpe;
    if off + bpe > bytes.len() {
        return f32::NAN;
    }
    decode_element(dtype, &bytes[off..off + bpe])
}

/// Per-element f32 decode with a single-block dequant cache.
///
/// The renderer and diff math both iterate elements 0..N within a byte buffer
/// that starts at element 0 of a tensor. `element(k)` returns the f32 value
/// of element K. For plain dtypes this is a fixed-stride little-endian read.
/// For quantized dtypes the reader keeps the most recently dequantized block
/// in `cache`; sequential `element(k)` calls in the hot per-tile loop hit
/// cache on every call after the first within each block.
///
/// For [`ElementStride::Packed`] dtypes (AWQ / GPTQ) the reader additionally
/// needs the matching `scales` and `qzeros` byte buffers. Attach them via
/// [`TensorElementReader::with_sidecars`]; without sidecars, packed dtypes
/// decode to NaN (so the renderer paints them as a sentinel rather than
/// returning a wrong f32).
pub struct TensorElementReader<'a> {
    dtype: Dtype,
    bytes: &'a [u8],
    /// Cached `(block_index, dequantized_block_floats)`. Only populated for
    /// quantized dtypes.
    cache: Option<(usize, Vec<f32>)>,
    /// AWQ/GPTQ sidecars. `None` for non-packed dtypes; required (else
    /// `element` returns NaN) for `ElementStride::Packed`.
    sidecars: Option<PackedSidecarRefs<'a>>,
}

/// Borrowed sidecar buffers for one packed-int tensor. Attached to a
/// [`TensorElementReader`] via [`TensorElementReader::with_sidecars`].
///
/// Lifetime is tied to the reader so the caller can hold the fetched byte
/// buffers on the stack alongside the reader without lifetime gymnastics.
#[derive(Clone, Copy)]
pub struct PackedSidecarRefs<'a> {
    pub scales: &'a [u8],
    pub scales_dtype: Dtype,
    /// `None` for symmetric quants (e.g. AWQ without zero-points).
    pub zeros: Option<&'a [u8]>,
    pub zeros_dtype: Dtype,
    /// Number of output columns in the unpacked tensor — needed to map a
    /// logical element index to its (group_index, in_group_pos) pair when
    /// `group_size` doesn't evenly divide the row stride.
    pub cols: u32,
}

impl<'a> TensorElementReader<'a> {
    pub fn new(dtype: Dtype, bytes: &'a [u8]) -> Self {
        Self {
            dtype,
            bytes,
            cache: None,
            sidecars: None,
        }
    }

    /// Builder: attach AWQ/GPTQ sidecar tensors. Only used by callers
    /// rendering packed-int tensors; for plain / Block dtypes this is a
    /// no-op (the sidecars are simply unused).
    #[allow(dead_code)]
    pub fn with_sidecars(mut self, refs: PackedSidecarRefs<'a>) -> Self {
        self.sidecars = Some(refs);
        self
    }

    /// f32 value of element `k`. Returns NaN if `k` is out of range so the
    /// caller's `is_finite()` check paints the pixel as a NaN sentinel
    /// instead of panicking on a partial fetch.
    pub fn element(&mut self, k: usize) -> f32 {
        match self.dtype.stride() {
            ElementStride::Fixed(bpe) => {
                let off = k * bpe;
                if off + bpe > self.bytes.len() {
                    return f32::NAN;
                }
                decode_element(self.dtype, &self.bytes[off..off + bpe])
            }
            ElementStride::Block {
                block_bytes,
                block_elements,
            } => {
                if block_elements == 0 {
                    return f32::NAN;
                }
                let blk_idx = k / block_elements;
                let in_blk = k % block_elements;
                if !matches!(self.cache, Some((idx, _)) if idx == blk_idx) {
                    let start = blk_idx * block_bytes;
                    if start + block_bytes > self.bytes.len() {
                        return f32::NAN;
                    }
                    let g = self.dtype.to_ggml().expect("quantized dtype has GgmlDType");
                    // `from_data` panics on misalignment. Copy the block bytes
                    // into a fresh Vec<u8> — heap-aligned to 8 bytes, which
                    // covers every block struct's alignment (≤ 2 in practice).
                    let raw = self.bytes[start..start + block_bytes].to_vec();
                    let qt = g.from_data(Cow::Owned(raw));
                    match qt.dequantize(block_elements) {
                        Ok(CpuStorage::F32(v)) => {
                            self.cache = Some((blk_idx, v));
                        }
                        _ => return f32::NAN,
                    }
                }
                self.cache
                    .as_ref()
                    .map(|(_, v)| v[in_blk])
                    .unwrap_or(f32::NAN)
            }
            ElementStride::Packed {
                bits,
                pack_dtype_bytes,
                group_size,
            } => {
                let Some(sc) = self.sidecars else {
                    // Without sidecars, packed dtypes can't be dequantized;
                    // return NaN so the renderer paints a sentinel.
                    return f32::NAN;
                };
                packed_element(
                    self.bytes,
                    sc,
                    k,
                    bits,
                    pack_dtype_bytes,
                    group_size as usize,
                )
            }
        }
    }

    /// Estimate `rms = sqrt(mean(x²))` over the first `sample_elements`
    /// elements (or the whole buffer if shorter). Skips non-finite values.
    pub fn rms_estimate(&mut self, sample_elements: usize) -> f32 {
        let n = match self.dtype.stride() {
            ElementStride::Fixed(bpe) => self
                .bytes
                .len()
                .checked_div(bpe)
                .map(|m| sample_elements.min(m))
                .unwrap_or(0),
            ElementStride::Block {
                block_bytes,
                block_elements,
            } => self
                .bytes
                .len()
                .checked_div(block_bytes)
                .map(|nb| sample_elements.min(nb * block_elements))
                .unwrap_or(0),
            ElementStride::Packed {
                bits,
                pack_dtype_bytes,
                ..
            } => {
                if bits == 0 {
                    0
                } else {
                    let elems_per_slot = (pack_dtype_bytes as usize * 8) / bits as usize;
                    self.bytes
                        .len()
                        .checked_div(pack_dtype_bytes as usize)
                        .map(|slots| sample_elements.min(slots * elems_per_slot))
                        .unwrap_or(0)
                }
            }
        };
        if n == 0 {
            return 0.0;
        }
        let mut sum_sq = 0.0f64;
        let mut count = 0u64;
        for k in 0..n {
            let v = self.element(k);
            if v.is_finite() {
                sum_sq += (v as f64) * (v as f64);
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            (sum_sq / count as f64).sqrt() as f32
        }
    }
}

/// Estimate RMS over a contiguous tensor byte slice. Convenience wrapper
/// around [`TensorElementReader::rms_estimate`] for call sites that don't
/// already hold a reader.
///
/// `sample_elements` is the *logical* element count, not bytes — for plain
/// dtypes that's `bytes.len() / element_size`; for quantized it's
/// `(bytes.len() / block_bytes) * block_elements`.
pub fn rms_from_buf(dtype: Dtype, bytes: &[u8]) -> f32 {
    let mut reader = TensorElementReader::new(dtype, bytes);
    let n = match dtype.stride() {
        ElementStride::Fixed(bpe) => bytes.len().checked_div(bpe).unwrap_or(0),
        ElementStride::Block {
            block_bytes,
            block_elements,
        } => bytes
            .len()
            .checked_div(block_bytes)
            .map(|nb| nb * block_elements)
            .unwrap_or(0),
        ElementStride::Packed {
            bits,
            pack_dtype_bytes,
            ..
        } => {
            if bits == 0 {
                0
            } else {
                let elems_per_slot = (pack_dtype_bytes as usize * 8) / bits as usize;
                bytes
                    .len()
                    .checked_div(pack_dtype_bytes as usize)
                    .map(|slots| slots * elems_per_slot)
                    .unwrap_or(0)
            }
        }
    };
    reader.rms_estimate(n)
}

/// Convenience: given a GGUF `TensorInfo`, the in-file byte range
/// `[file_start, file_end)`. `file_start` is `tensor_data_offset +
/// info.offset`; `file_end` is `file_start + byte_size_for(info)`. Used by
/// the GGUF parser when translating into `TensorMeta`.
pub fn gguf_tensor_byte_range(info: &TensorInfo, tensor_data_offset: u64) -> (u64, u64) {
    let elem_count: usize = info.shape.elem_count();
    let g = info.ggml_dtype;
    let bytes = elem_count / g.block_size() * g.type_size();
    let start = tensor_data_offset + info.offset;
    (start, start + bytes as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_from_str_roundtrip() {
        assert!(matches!(Dtype::from_safetensors_str("BF16"), Dtype::BF16));
        assert!(matches!(Dtype::from_safetensors_str("F16"), Dtype::F16));
        assert!(matches!(
            Dtype::from_safetensors_str("F8_E4M3"),
            Dtype::F8E4M3
        ));
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn reader_plain_f32() {
        let bytes = f32_bytes(&[1.0, 2.0, -3.5]);
        let mut r = TensorElementReader::new(Dtype::F32, &bytes);
        assert_eq!(r.element(0), 1.0);
        assert_eq!(r.element(1), 2.0);
        assert_eq!(r.element(2), -3.5);
        assert!(r.element(3).is_nan());
    }

    #[test]
    fn reader_quantized_q8_0_does_not_crash_on_padded_block() {
        // Smoke-test the quantised dequant path: a syntactically valid Q8_0
        // block (1 f16 scale + 32 i8 quants = 34 bytes) goes through the
        // candle dequant kernel without panicking, returns finite values,
        // and out-of-range reads return NaN as the renderer expects.
        //
        // We don't assert specific dequant values because candle's
        // `GgmlDType::from_data` reads block scales/quants via a slice
        // reinterpretation whose results are correct in production (via
        // `BlockQ8_0::from_float`) but harder to construct from raw bytes
        // here without going through private struct fields. End-to-end
        // correctness is verified by the GGUF render path in the
        // verification plan.
        let bytes = vec![0u8; 34];
        let mut r = TensorElementReader::new(Dtype::Q8_0, &bytes);
        for k in 0..32 {
            let v = r.element(k);
            assert!(v.is_finite(), "elem {k}: got {v}, expected finite");
        }
        // Out-of-block read returns NaN sentinel.
        assert!(r.element(32).is_nan());
    }

    #[test]
    fn diff_rms_zero_delta_paints_black() {
        let o = f32_bytes(&[0.1, -0.2, 0.3]);
        let m = o.clone();
        let out = Dtype::F32.diff_to_u8(&o, 0, Dtype::F32, &m, 0, DiffMetric::Rms, 0.1, 3);
        assert_eq!(out, vec![127, 127, 127]);
    }

    #[test]
    fn diff_rms_half_stddev_saturates() {
        let rms: f32 = 0.04;
        let o = f32_bytes(&[0.1, 0.1]);
        let m = f32_bytes(&[0.1 + 0.5 * rms, 0.1 - 0.5 * rms]);
        let out = Dtype::F32.diff_to_u8(&o, 0, Dtype::F32, &m, 0, DiffMetric::Rms, rms, 2);
        assert_eq!(out, vec![254, 0]);
    }

    #[test]
    fn diff_non_finite_paints_white() {
        let o = f32_bytes(&[0.1, f32::NAN, 0.1]);
        let m = f32_bytes(&[0.1, 0.1, f32::INFINITY]);
        for metric in [DiffMetric::Rms, DiffMetric::AbsLog, DiffMetric::Exact] {
            let out = Dtype::F32.diff_to_u8(&o, 0, Dtype::F32, &m, 0, metric, 0.1, 3);
            assert_eq!(out[0], 127, "{metric:?} same value");
            assert_eq!(out[1], 255, "{metric:?} NaN in orig");
            assert_eq!(out[2], 255, "{metric:?} Inf in mod");
        }
    }

    #[test]
    fn rms_from_buf_basic() {
        let b = f32_bytes(&[1.0, -1.0, 1.0, -1.0]);
        let r = rms_from_buf(Dtype::F32, &b);
        assert!((r - 1.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn rms_from_buf_ignores_non_finite() {
        let b = f32_bytes(&[2.0, f32::NAN, -2.0, f32::INFINITY, 2.0]);
        let r = rms_from_buf(Dtype::F32, &b);
        assert!((r - 2.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn rms_from_buf_empty_is_zero() {
        let r = rms_from_buf(Dtype::F32, &[]);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn stride_fixed_for_plain() {
        assert_eq!(Dtype::F32.stride(), ElementStride::Fixed(4));
        assert_eq!(Dtype::F16.stride(), ElementStride::Fixed(2));
        assert_eq!(Dtype::I8.stride(), ElementStride::Fixed(1));
    }

    #[test]
    fn stride_packed_for_awq_dtypes() {
        match Dtype::Int4Packed.stride() {
            ElementStride::Packed {
                bits,
                pack_dtype_bytes,
                group_size,
            } => {
                assert_eq!(bits, 4);
                assert_eq!(pack_dtype_bytes, 4);
                assert_eq!(group_size, 128);
            }
            other => panic!("expected Packed for Int4Packed, got {other:?}"),
        }
    }

    #[test]
    fn packed_int4_dequant_known_values() {
        // Build a 2-row × 8-col tensor of int4 quants:
        //   row 0: [1, 2, 3, 4, 5, 6, 7, 8]   (one int32 slot = 0x87654321)
        //   row 1: [0, 1, 0, 2, 0, 3, 0, 4]   (one int32 slot = 0x40302010)
        // Two group_size=2 groups per row (well, group_size really applies on
        // rows for canonical layouts — for this micro test we set group_size = 2
        // so we exercise the group lookup). Scale = 1.0, zero = 0 ⇒ dequant
        // returns the int4 value as-is.
        //
        // We construct sidecars manually:
        //   scales: 2 groups × 8 cols, all = 1.0 (f32)
        //   zeros:  None (symmetric)
        let qweight: Vec<u8> = {
            let row0: u32 = 0x8765_4321;
            let row1: u32 = 0x4030_2010;
            let mut v = Vec::new();
            v.extend_from_slice(&row0.to_le_bytes());
            v.extend_from_slice(&row1.to_le_bytes());
            v
        };
        let scales: Vec<u8> = {
            let mut v = Vec::new();
            // 2 groups × 8 cols, but for this single-row-per-group layout
            // we just need group 0 (covers row 0) and group 1 (covers row 1).
            // Each group has 8 cols. So 16 scales total.
            for _ in 0..16 {
                v.extend_from_slice(&1.0f32.to_le_bytes());
            }
            v
        };
        let sc = PackedSidecarRefs {
            scales: &scales,
            scales_dtype: Dtype::F32,
            zeros: None,
            zeros_dtype: Dtype::Unknown,
            cols: 8,
        };

        let mut r = TensorElementReader::new(Dtype::Int4Packed, &qweight).with_sidecars(sc);
        // Override the stride's group_size via construction is not exposed,
        // but the stride() method returns group_size=128 for Int4Packed by
        // default. For our 2-row test we set the scales table to a full
        // 16-element layout (groups 0/1 × cols 0..8) so the group lookup at
        // row 0 picks scale group 0 = 1.0, row 1 picks group 0 again (since
        // row 1 / 128 = 0). Either way, scale=1.0.
        // Row 0 values:
        assert_eq!(r.element(0), 1.0);
        assert_eq!(r.element(1), 2.0);
        assert_eq!(r.element(7), 8.0);
        // Row 1 values:
        assert_eq!(r.element(8), 0.0);
        assert_eq!(r.element(9), 1.0);
        assert_eq!(r.element(15), 4.0);
    }

    #[test]
    fn packed_without_sidecars_returns_nan() {
        // Same packed buffer, no sidecars attached → NaN per element.
        let qweight: Vec<u8> = vec![0xFFu8; 4];
        let mut r = TensorElementReader::new(Dtype::Int4Packed, &qweight);
        for k in 0..8 {
            assert!(
                r.element(k).is_nan(),
                "k={k}: expected NaN without sidecars"
            );
        }
    }

    #[test]
    fn stride_block_for_quantized() {
        match Dtype::Q4_0.stride() {
            ElementStride::Block {
                block_bytes,
                block_elements,
            } => {
                assert_eq!(block_elements, 32);
                assert_eq!(block_bytes, 18);
            }
            other => panic!("expected Block, got {other:?}"),
        }
        match Dtype::Q4K.stride() {
            ElementStride::Block {
                block_bytes,
                block_elements,
            } => {
                assert_eq!(block_elements, 256);
                // Q4K is 144 bytes per 256-element block.
                assert_eq!(block_bytes, 144);
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }
}
