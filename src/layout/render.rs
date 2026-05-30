//! Per-element colorizers shared between the tiled and single-image paths.
//!
//! The architectural layout calls these once per output pixel, where each
//! pixel corresponds to one element of one tensor. The legacy byte-Hilbert
//! path doesn't use these — it operates per byte through a precomputed LUT.

use image::Rgb;

use crate::format::{DiffMetric, Dtype, TensorElementReader};

/// Neutral background colour for canvas pixels that fall outside every
/// tensor's rectangle in [`crate::layout::arch::ArchLayout`]. Not pure black
/// so the 2×2 pyramid average doesn't push a near-zero diff toward the
/// background and steal contrast from genuinely-zero diffs.
pub const PADDING_RGB: Rgb<u8> = Rgb([20, 20, 20]);

/// Map a single decoded f32 element to one of the 256-entry pixel LUT
/// indices used in legacy plain-byte mode. We pick the MSB of the dtype's
/// little-endian bytewise representation for float types (since that's the
/// byte the byte-mode renderer would have hit at element index 0), and the
/// raw value clamped to [0, 255] for ints.
#[inline]
pub fn element_to_byte_proxy(dtype: Dtype, raw: &[u8]) -> u8 {
    match dtype {
        // For little-endian floats, the high-order byte is at the end of the
        // slice — and that's where the sign bit + exponent live, which the
        // byte-LUT colouring is most responsive to.
        Dtype::F64 | Dtype::F32 | Dtype::F16 | Dtype::BF16 => *raw.last().unwrap_or(&0),
        // F8 dtypes are single-byte; everything in `raw` is the element.
        Dtype::F8E4M3 | Dtype::F8E5M2 => raw.first().copied().unwrap_or(0),
        // Integers: use the low byte directly so the LUT modulates by
        // magnitude in the typical case.
        Dtype::I64 | Dtype::U64 | Dtype::I32 | Dtype::U32 | Dtype::I16 | Dtype::U16 => {
            raw.first().copied().unwrap_or(0)
        }
        Dtype::I8 | Dtype::U8 | Dtype::Bool | Dtype::Unknown => raw.first().copied().unwrap_or(0),
        // Quantized dtypes shouldn't reach this function — callers route
        // through `TensorElementReader` for them. Take the first byte as a
        // best-effort fallback in case any path slips through.
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
        // Packed-int dtypes (AWQ/GPTQ) also reach this only by accident —
        // sidecar-aware dequant flows through `TensorElementReader::with_sidecars`.
        // First-byte fallback gives a plausible legacy hilbert hue.
        | Dtype::Int4Packed
        | Dtype::Int3Packed
        | Dtype::Int8Packed => raw.first().copied().unwrap_or(0),
    }
}

/// Element-aware colour for plain-mode visualisation.
///
/// For fixed-stride dtypes: reads `dtype.element_size()` bytes at
/// `elem_idx * elem_size` and proxies through the LUT. Byte-identical to
/// the pre-quantisation behaviour for safetensors paths.
///
/// For block-quantised dtypes: dequantises element `elem_idx` to f32 via
/// [`TensorElementReader`] and proxies via the f32's MSB (matching how the
/// equivalent F32 path lights the LUT).
#[inline]
pub fn plain_element_color(
    dtype: Dtype,
    bytes: &[u8],
    elem_idx: usize,
    pixel_lut: &[Rgb<u8>; 256],
) -> Rgb<u8> {
    use crate::format::ElementStride;
    match dtype.stride() {
        ElementStride::Fixed(elem) => {
            let start = elem_idx * elem;
            if start + elem > bytes.len() {
                return PADDING_RGB;
            }
            let raw = &bytes[start..start + elem];
            pixel_lut[element_to_byte_proxy(dtype, raw) as usize]
        }
        ElementStride::Block { .. } => {
            let mut reader = TensorElementReader::new(dtype, bytes);
            let v = reader.element(elem_idx);
            if !v.is_finite() {
                return PADDING_RGB;
            }
            // MSB of the f32 representation = sign + exponent bits. Same
            // bucket the byte-Hilbert renderer would pick if it saw the
            // dequantised value as a 4-byte LE f32.
            let byte = (v.to_bits() >> 24) as u8;
            pixel_lut[byte as usize]
        }
        // Packed-int dtypes need scales/zeros sidecar buffers which this
        // renderer entry point doesn't have. Paint as padding so the
        // viewer sees "we recognise the tensor but can't dequant per-element
        // here yet" rather than a misleading false-color signal.
        ElementStride::Packed { .. } => PADDING_RGB,
    }
}

/// Element-aware diff colour. Decodes one element from each side via the
/// shared [`TensorElementReader`] (so quantised tensors dequant on the fly),
/// runs the metric, and returns the resulting LUT colour.
#[inline]
pub fn diff_element_color(
    orig_dtype: Dtype,
    orig_bytes: &[u8],
    orig_idx: usize,
    mod_dtype: Dtype,
    mod_bytes: &[u8],
    mod_idx: usize,
    metric: DiffMetric,
    scale_orig: f32,
    pixel_lut: &[Rgb<u8>; 256],
) -> Rgb<u8> {
    let mut o_reader = TensorElementReader::new(orig_dtype, orig_bytes);
    let mut m_reader = TensorElementReader::new(mod_dtype, mod_bytes);
    let o = o_reader.element(orig_idx);
    let m = m_reader.element(mod_idx);
    if !o.is_finite() || !m.is_finite() {
        return pixel_lut[255];
    }
    let delta = m - o;
    let signed = match metric {
        DiffMetric::Rms => {
            use crate::format::{K_RMS_SAT, RMS_FLOOR};
            let rms_denom = (K_RMS_SAT * scale_orig.max(RMS_FLOOR)).max(f32::MIN_POSITIVE);
            (delta / rms_denom).clamp(-1.0, 1.0)
        }
        DiffMetric::AbsLog => {
            use crate::format::{ABS_LOG_MAX, ABS_LOG_MIN};
            let abs_d = delta.abs();
            if abs_d <= ABS_LOG_MIN {
                0.0
            } else {
                let log_min = ABS_LOG_MIN.log10();
                let log_max = ABS_LOG_MAX.log10();
                let norm = ((abs_d.log10() - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
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
    pixel_lut[byte as usize]
}

/// Look up an xorb colour for a byte offset, then blend it 50/50 with the
/// tensor's dtype hue and modulate by the per-element intensity proxy.
/// Mirrors `tiled::leaf::render_leaf_tile_xet_dtype_from_buf`.
#[inline]
pub fn xet_dtype_element_color(
    dtype: Dtype,
    bytes: &[u8],
    elem_idx: usize,
    tensor_byte_start: u64,
    xorb_ranges: &[(u64, u64, u8)],
    tableau: &[Rgb<u8>; 20],
) -> Rgb<u8> {
    let (byte, abs_byte_pos) =
        element_intensity_and_position(dtype, bytes, elem_idx, tensor_byte_start);
    let Some((byte, abs_byte_pos)) = byte.zip(Some(abs_byte_pos)) else {
        return PADDING_RGB;
    };
    let d = dtype.to_color();
    match xorb_color_idx(xorb_ranges, abs_byte_pos) {
        Some(idx) => {
            let t = tableau[idx as usize];
            let s = byte as u32;
            Rgb([
                (((d[0] as u32 + t[0] as u32) * s + 255) / 510) as u8,
                (((d[1] as u32 + t[1] as u32) * s + 255) / 510) as u8,
                (((d[2] as u32 + t[2] as u32) * s + 255) / 510) as u8,
            ])
        }
        None => {
            let s = byte as u16;
            Rgb([
                ((d[0] as u16 * s + 127) / 255) as u8,
                ((d[1] as u16 * s + 127) / 255) as u8,
                ((d[2] as u16 * s + 127) / 255) as u8,
            ])
        }
    }
}

/// Plain xet colour: byte intensity × xorb tableau hue (no dtype blend).
/// Mirrors `tiled::leaf::render_leaf_tile_xet_from_buf`.
#[inline]
pub fn xet_element_color(
    dtype: Dtype,
    bytes: &[u8],
    elem_idx: usize,
    tensor_byte_start: u64,
    xorb_ranges: &[(u64, u64, u8)],
    tableau: &[Rgb<u8>; 20],
    pixel_lut: &[Rgb<u8>; 256],
) -> Rgb<u8> {
    let (byte, abs_byte_pos) =
        element_intensity_and_position(dtype, bytes, elem_idx, tensor_byte_start);
    let Some(byte) = byte else {
        return PADDING_RGB;
    };
    match xorb_color_idx(xorb_ranges, abs_byte_pos) {
        Some(idx) => {
            let t = tableau[idx as usize];
            let scale = byte as u16;
            Rgb([
                ((t[0] as u16 * scale + 127) / 255) as u8,
                ((t[1] as u16 * scale + 127) / 255) as u8,
                ((t[2] as u16 * scale + 127) / 255) as u8,
            ])
        }
        None => pixel_lut[byte as usize],
    }
}

/// Decode element `elem_idx` into an intensity byte (for the byte-LUT path)
/// and a representative absolute byte position (for the xorb xorb lookup).
///
/// For fixed-stride dtypes the byte position is exact (the first byte of the
/// element). For block-quantised dtypes we return the block's start byte,
/// so every element within a block shares one xorb hue.
fn element_intensity_and_position(
    dtype: Dtype,
    bytes: &[u8],
    elem_idx: usize,
    tensor_byte_start: u64,
) -> (Option<u8>, u64) {
    use crate::format::ElementStride;
    match dtype.stride() {
        ElementStride::Fixed(elem) => {
            let start = elem_idx * elem;
            if start + elem > bytes.len() {
                return (None, tensor_byte_start);
            }
            let raw = &bytes[start..start + elem];
            (
                Some(element_to_byte_proxy(dtype, raw)),
                tensor_byte_start + start as u64,
            )
        }
        ElementStride::Block {
            block_bytes,
            block_elements,
        } => {
            let mut reader = TensorElementReader::new(dtype, bytes);
            let v = reader.element(elem_idx);
            if !v.is_finite() {
                return (None, tensor_byte_start);
            }
            let block_idx = elem_idx / block_elements.max(1);
            let abs = tensor_byte_start + (block_idx * block_bytes) as u64;
            (Some((v.to_bits() >> 24) as u8), abs)
        }
        // Packed-int dtypes: report the packed-slot byte position so the
        // xorb hue stays stable across the elements within a slot. We can't
        // dequant without sidecars, so the intensity byte is None (renders
        // as padding).
        ElementStride::Packed {
            bits,
            pack_dtype_bytes,
            ..
        } => {
            if bits == 0 {
                (None, tensor_byte_start)
            } else {
                let elems_per_slot = (pack_dtype_bytes as usize * 8) / bits as usize;
                let slot_idx = elem_idx.checked_div(elems_per_slot).unwrap_or(0);
                let abs = tensor_byte_start + (slot_idx * pack_dtype_bytes as usize) as u64;
                (None, abs)
            }
        }
    }
}

/// Binary-search a sorted-non-overlapping `(start, end, color_idx)` list for
/// the entry that contains `pos`.
fn xorb_color_idx(ranges: &[(u64, u64, u8)], pos: u64) -> Option<u8> {
    if ranges.is_empty() {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = ranges.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (s, e, c) = ranges[mid];
        if pos < s {
            hi = mid;
        } else if pos >= e {
            lo = mid + 1;
        } else {
            return Some(c);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_rgb_is_not_pure_black() {
        // Sanity: the 2×2 pyramid average mustn't paint padding identical to
        // an unrelated diff-of-zero region.
        assert_ne!(PADDING_RGB, Rgb([0u8, 0, 0]));
    }

    #[test]
    fn element_byte_proxy_f32_msb() {
        // Little-endian f32 1.0 → bytes [0x00, 0x00, 0x80, 0x3f]; MSB is 0x3f.
        let raw = 1.0f32.to_le_bytes();
        assert_eq!(element_to_byte_proxy(Dtype::F32, &raw), 0x3f);
    }
}
