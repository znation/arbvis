//! Per-element colorizers shared between the tiled and single-image paths.
//!
//! The architectural layout calls these once per output pixel, where each
//! pixel corresponds to one element of one tensor. The legacy byte-Hilbert
//! path doesn't use these — it operates per byte through a precomputed LUT.

use image::Rgb;

use crate::safetensors::{decode_element, DiffMetric, Dtype};

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
    }
}

/// Element-aware colour for plain-mode visualisation. Reads one element
/// from `bytes` at `[idx*elem_size, (idx+1)*elem_size)` and runs it through
/// `pixel_lut` via the byte-proxy mapping.
#[inline]
pub fn plain_element_color(
    dtype: Dtype,
    bytes: &[u8],
    elem_idx: usize,
    pixel_lut: &[Rgb<u8>; 256],
) -> Rgb<u8> {
    let elem = dtype.element_size();
    let start = elem_idx * elem;
    if start + elem > bytes.len() {
        return PADDING_RGB;
    }
    let raw = &bytes[start..start + elem];
    pixel_lut[element_to_byte_proxy(dtype, raw) as usize]
}

/// Element-aware diff colour. Decodes one element from each side, runs the
/// shared diff metric, and returns the resulting LUT colour.
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
    let orig_elem = orig_dtype.element_size();
    let mod_elem = mod_dtype.element_size();
    let os = orig_idx * orig_elem;
    let ms = mod_idx * mod_elem;
    if os + orig_elem > orig_bytes.len() || ms + mod_elem > mod_bytes.len() {
        return PADDING_RGB;
    }
    let o = decode_element(orig_dtype, &orig_bytes[os..os + orig_elem]);
    let m = decode_element(mod_dtype, &mod_bytes[ms..ms + mod_elem]);

    if !o.is_finite() || !m.is_finite() {
        return pixel_lut[255];
    }
    let delta = m - o;
    let signed = match metric {
        DiffMetric::Rms => {
            // Mirror the constants in safetensors.rs::diff_to_u8.
            const K_RMS_SAT: f32 = 0.5;
            const RMS_FLOOR: f32 = 1e-6;
            let rms_denom = (K_RMS_SAT * scale_orig.max(RMS_FLOOR)).max(f32::MIN_POSITIVE);
            (delta / rms_denom).clamp(-1.0, 1.0)
        }
        DiffMetric::AbsLog => {
            const ABS_LOG_MIN: f32 = 1e-6;
            const ABS_LOG_MAX: f32 = 1e-1;
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
    let elem = dtype.element_size();
    let start = elem_idx * elem;
    if start + elem > bytes.len() {
        return PADDING_RGB;
    }
    let raw = &bytes[start..start + elem];
    let byte = element_to_byte_proxy(dtype, raw);
    let d = dtype.to_color();
    let abs_byte_pos = tensor_byte_start + start as u64;
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
    let elem = dtype.element_size();
    let start = elem_idx * elem;
    if start + elem > bytes.len() {
        return PADDING_RGB;
    }
    let raw = &bytes[start..start + elem];
    let byte = element_to_byte_proxy(dtype, raw);
    let abs_byte_pos = tensor_byte_start + start as u64;
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
