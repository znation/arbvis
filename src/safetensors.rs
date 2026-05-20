use image::Rgb;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
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
}

impl Dtype {
    pub fn from_str(s: &str) -> Self {
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

    pub fn element_size(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::I8 | Dtype::U8 | Dtype::Bool | Dtype::Unknown => 1,
        }
    }

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
        }
    }

    /// Compute the signed diff between matched elements, returning one u8 per element pair.
    /// `self` is the dtype for `orig`; `mod_dtype` is the dtype for `mod_`.
    /// `scale_orig` is the per-tensor scale (RMS of `orig`); only used by `DiffMetric::Rms`.
    /// Encoding: 127 = no change, 128–254 = increased, 0–126 = decreased, 255 = non-finite.
    /// No intermediate Vec<f32> is allocated.
    pub fn diff_to_u8(
        self,
        orig: &[u8],
        mod_dtype: Dtype,
        mod_: &[u8],
        metric: DiffMetric,
        scale_orig: f32,
    ) -> Vec<u8> {
        let orig_elem = self.element_size();
        let mod_elem = mod_dtype.element_size();
        let rms_denom = (K_RMS_SAT * scale_orig.max(RMS_FLOOR)).max(f32::MIN_POSITIVE);
        let log_min = ABS_LOG_MIN.log10();
        let log_max = ABS_LOG_MAX.log10();
        orig.chunks_exact(orig_elem)
            .zip(mod_.chunks_exact(mod_elem))
            .map(|(oc, mc)| {
                let o = decode_element(self, oc);
                let m = decode_element(mod_dtype, mc);
                if !o.is_finite() || !m.is_finite() { return 255u8; }
                let delta = m - o;
                let signed = match metric {
                    DiffMetric::Rms => (delta / rms_denom).clamp(-1.0, 1.0),
                    DiffMetric::AbsLog => {
                        let abs_d = delta.abs();
                        if abs_d <= ABS_LOG_MIN {
                            0.0
                        } else {
                            let norm = ((abs_d.log10() - log_min) / (log_max - log_min))
                                .clamp(0.0, 1.0);
                            if delta >= 0.0 { norm } else { -norm }
                        }
                    }
                    DiffMetric::Exact => {
                        if delta == 0.0 { 0.0 }
                        else if delta > 0.0 { 1.0 }
                        else { -1.0 }
                    }
                };
                let brightness = (signed.abs() * 127.0).round() as u8;
                if signed >= 0.0 { 127u8.saturating_add(brightness) }
                else { 127u8.saturating_sub(brightness) }
            })
            .collect()
    }
}

/// Selects how per-element diffs are encoded for visualization.
///
/// All three preserve the sign convention (green = grew, red = shrank,
/// black = no change, white = NaN/Inf in either side). They differ in how
/// brightness is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffMetric {
    /// Per-tensor RMS-normalized signed delta:
    ///   `signed = clamp(delta / (K_RMS_SAT * rms(orig)), -1, 1)`
    ///
    /// Reads as "how many tensor-stddevs did this element move." Stable
    /// across tensors regardless of weight scale, and doesn't blow up on
    /// small base weights the way per-element `(m-o)/|o|` does. Requires a
    /// per-tensor scale (computed at setup via sampling). Default.
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

impl Default for DiffMetric {
    fn default() -> Self { DiffMetric::Rms }
}

/// Saturation threshold for `DiffMetric::Rms`: an element whose delta equals
/// `K_RMS_SAT * rms(orig)` paints at full brightness. 0.5 means "half a
/// tensor-stddev is fully saturated"; a typical LoRA-merge moves median
/// elements by ~0.005 stddevs (subtle), an aggressive full-finetune by ~0.05
/// stddevs (clearly visible), an uncorrelated init by ~1 stddev (saturated).
const K_RMS_SAT: f32 = 0.5;

/// Floor for `rms(orig)` in `DiffMetric::Rms`, used to avoid divide-by-zero
/// on all-zero tensors and to cap sensitivity on near-zero tensors.
const RMS_FLOOR: f32 = 1e-6;

/// Log-brightness range endpoints for `DiffMetric::AbsLog`. Deltas with
/// `|delta| < ABS_LOG_MIN` paint black; `|delta| >= ABS_LOG_MAX` saturate.
/// The span covers the typical range of useful bf16 finetune deltas.
const ABS_LOG_MIN: f32 = 1e-6;
const ABS_LOG_MAX: f32 = 1e-1;

/// Estimate RMS = `sqrt(mean(x²))` of a contiguous tensor slice. Skips
/// non-finite elements. Returns 0.0 for an empty buffer or for buffers with
/// no finite samples.
///
/// Pass the entire tensor for an exact RMS, or a contiguous sample slice for
/// an estimate — for most NN weight tensors a few thousand contiguous
/// elements give a stable estimate.
pub fn rms_from_buf(dtype: Dtype, bytes: &[u8]) -> f32 {
    let elem = dtype.element_size();
    if elem == 0 || bytes.is_empty() { return 0.0; }
    let mut sum_sq = 0.0f64;
    let mut count = 0u64;
    for chunk in bytes.chunks_exact(elem) {
        let v = decode_element(dtype, chunk);
        if v.is_finite() {
            sum_sq += (v as f64) * (v as f64);
            count += 1;
        }
    }
    if count == 0 { return 0.0; }
    (sum_sq / count as f64).sqrt() as f32
}

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
    pub fn label(&self) -> String {
        let shape_str: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        let dtype_str = match self.dtype {
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
        };
        format!("{} [{}, {}]", self.name, dtype_str, shape_str.join("×"))
    }
}

/// Decode a single element from a little-endian byte slice.
fn decode_element(dtype: Dtype, bytes: &[u8]) -> f32 {
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
    }
}

/// Parse a safetensors file's header from raw bytes.
///
/// Returns tensors sorted by file_start and the absolute end offset of the header.
pub fn parse_header(data: &[u8]) -> anyhow::Result<(Vec<TensorMeta>, u64)> {
    if data.len() < 8 {
        anyhow::bail!("safetensors: file too short to contain header size field");
    }
    let header_size = u64::from_le_bytes(data[..8].try_into().unwrap());
    let header_end = 8 + header_size;
    if header_end as usize > data.len() {
        anyhow::bail!(
            "safetensors: header_size={} exceeds file length={}",
            header_size,
            data.len()
        );
    }
    if header_size > 100 * 1024 * 1024 {
        anyhow::bail!("safetensors: header_size={} exceeds 100 MB safety limit", header_size);
    }

    let json_bytes = &data[8..header_end as usize];
    let root: serde_json::Value = serde_json::from_slice(json_bytes)
        .map_err(|e| anyhow::anyhow!("safetensors: invalid JSON header: {}", e))?;

    let obj = root
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("safetensors: header JSON is not an object"))?;

    let mut tensors = Vec::with_capacity(obj.len());

    for (name, val) in obj {
        if name == "__metadata__" {
            continue;
        }
        let tensor_obj = val
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("safetensors: tensor '{}' is not an object", name))?;

        let dtype_str = tensor_obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("safetensors: tensor '{}' missing 'dtype'", name))?;
        let dtype = Dtype::from_str(dtype_str);

        let shape: Vec<u64> = tensor_obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("safetensors: tensor '{}' missing 'shape'", name))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .ok_or_else(|| anyhow::anyhow!("safetensors: tensor '{}' shape dim is not u64", name))
            })
            .collect::<anyhow::Result<_>>()?;

        let offsets = tensor_obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .filter(|a| a.len() == 2)
            .ok_or_else(|| {
                anyhow::anyhow!("safetensors: tensor '{}' missing valid 'data_offsets'", name)
            })?;
        let rel_start = offsets[0]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("safetensors: tensor '{}' data_offsets[0] not u64", name))?;
        let rel_end = offsets[1]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("safetensors: tensor '{}' data_offsets[1] not u64", name))?;

        tensors.push(TensorMeta {
            name: name.clone(),
            dtype,
            shape,
            file_start: header_end + rel_start,
            file_end: header_end + rel_end,
        });
    }

    tensors.sort_by_key(|t| t.file_start);
    Ok((tensors, header_end))
}

/// Build a sorted list of (start, end, color) ranges covering the entire file.
///
/// The header region is gray, each tensor region gets its dtype color, gaps are black.
pub fn build_color_ranges(
    tensors: &[TensorMeta],
    header_end: u64,
    file_size: u64,
) -> Vec<(u64, u64, Rgb<u8>)> {
    let mut ranges: Vec<(u64, u64, Rgb<u8>)> = Vec::with_capacity(tensors.len() + 2);

    // Header region (gray).
    if header_end > 0 {
        ranges.push((0, header_end, Rgb([100, 100, 100])));
    }

    let mut pos = header_end;
    for t in tensors {
        // Gap between tensors (black).
        if t.file_start > pos {
            ranges.push((pos, t.file_start, Rgb([0, 0, 0])));
        }
        if t.file_end > t.file_start {
            ranges.push((t.file_start, t.file_end, t.dtype.to_color()));
        }
        pos = t.file_end;
    }
    // Trailing gap (black).
    if pos < file_size {
        ranges.push((pos, file_size, Rgb([0, 0, 0])));
    }

    ranges
}

/// Return the display color for a byte at `pos` within a file.
///
/// `ranges` must be the output of `build_color_ranges` (sorted by start, non-overlapping).
#[inline]
pub fn color_for_pos(pos: u64, ranges: &[(u64, u64, Rgb<u8>)]) -> Rgb<u8> {
    let idx = ranges.partition_point(|r| r.1 <= pos);
    if idx < ranges.len() && ranges[idx].0 <= pos {
        ranges[idx].2
    } else {
        Rgb([0, 0, 0])
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_from_str_roundtrip() {
        assert!(matches!(Dtype::from_str("BF16"), Dtype::BF16));
        assert!(matches!(Dtype::from_str("F16"), Dtype::F16));
        assert!(matches!(Dtype::from_str("F8_E4M3"), Dtype::F8E4M3));
    }

    #[test]
    fn color_for_pos_header_region() {
        let ranges = vec![
            (0u64, 100u64, Rgb([100u8, 100, 100])),
            (100u64, 200u64, Rgb([180u8, 255, 60])),
        ];
        assert_eq!(color_for_pos(0, &ranges), Rgb([100, 100, 100]));
        assert_eq!(color_for_pos(50, &ranges), Rgb([100, 100, 100]));
        assert_eq!(color_for_pos(100, &ranges), Rgb([180, 255, 60]));
        assert_eq!(color_for_pos(199, &ranges), Rgb([180, 255, 60]));
    }

    #[test]
    fn color_for_pos_out_of_range() {
        let ranges = vec![(0u64, 100u64, Rgb([255u8, 0, 0]))];
        assert_eq!(color_for_pos(200, &ranges), Rgb([0, 0, 0]));
    }

    /// Encode a slice of f32 values as little-endian bytes for diff tests.
    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn diff_rms_zero_delta_paints_black() {
        let o = f32_bytes(&[0.1, -0.2, 0.3]);
        let m = o.clone();
        let out = Dtype::F32.diff_to_u8(&o, Dtype::F32, &m, DiffMetric::Rms, 0.1);
        assert_eq!(out, vec![127, 127, 127]);
    }

    #[test]
    fn diff_rms_half_stddev_saturates() {
        // K_RMS_SAT = 0.5, so delta = 0.5 * rms saturates to brightness 127.
        let rms: f32 = 0.04;
        let o = f32_bytes(&[0.1, 0.1]);
        let m = f32_bytes(&[0.1 + 0.5 * rms, 0.1 - 0.5 * rms]);
        let out = Dtype::F32.diff_to_u8(&o, Dtype::F32, &m, DiffMetric::Rms, rms);
        assert_eq!(out, vec![254, 0]);
    }

    #[test]
    fn diff_rms_quarter_saturation_is_midbright() {
        // delta = 0.25 * K_RMS_SAT * rms → |signed| = 0.25 → brightness 32.
        let rms: f32 = 0.04;
        let k = 0.5_f32; // K_RMS_SAT
        let target = 0.25_f32;
        let d = target * k * rms;
        let o = f32_bytes(&[0.1, 0.1]);
        let m = f32_bytes(&[0.1 + d, 0.1 - d]);
        let out = Dtype::F32.diff_to_u8(&o, Dtype::F32, &m, DiffMetric::Rms, rms);
        let expected = (target * 127.0).round() as u8;
        assert_eq!(out[0], 127 + expected);
        assert_eq!(out[1], 127 - expected);
    }

    #[test]
    fn diff_rms_zero_scale_falls_back_to_floor() {
        // scale_orig == 0 must not panic and must still discriminate sign.
        let o = f32_bytes(&[0.0]);
        let m = f32_bytes(&[1.0]);
        let out = Dtype::F32.diff_to_u8(&o, Dtype::F32, &m, DiffMetric::Rms, 0.0);
        // 1.0 ≫ 0.5 * RMS_FLOOR, so this saturates green.
        assert_eq!(out, vec![254]);
    }

    #[test]
    fn diff_abs_log_below_floor_is_black() {
        // |delta| = 1e-7 < ABS_LOG_MIN (1e-6) → no signal.
        let o = f32_bytes(&[0.5, 0.5]);
        let m = f32_bytes(&[0.5 + 1e-7, 0.5 - 1e-7]);
        let out = Dtype::F32.diff_to_u8(&o, Dtype::F32, &m, DiffMetric::AbsLog, 0.0);
        assert_eq!(out, vec![127, 127]);
    }

    #[test]
    fn diff_abs_log_above_ceiling_saturates() {
        // |delta| = 1.0 ≫ ABS_LOG_MAX (1e-1) → full saturation.
        let o = f32_bytes(&[0.0, 0.0]);
        let m = f32_bytes(&[1.0, -1.0]);
        let out = Dtype::F32.diff_to_u8(&o, Dtype::F32, &m, DiffMetric::AbsLog, 0.0);
        assert_eq!(out, vec![254, 0]);
    }

    #[test]
    fn diff_abs_log_midpoint() {
        // |delta| = sqrt(min*max) = 1e-3.5 sits at the midpoint of the log span.
        let abs_d = 10f32.powf(-3.5);
        let o = f32_bytes(&[0.0]);
        let m = f32_bytes(&[abs_d]);
        let out = Dtype::F32.diff_to_u8(&o, Dtype::F32, &m, DiffMetric::AbsLog, 0.0);
        // 0.5 brightness = 64; expect 127 + 64 = 191. Allow ±1 for log rounding.
        assert!(out[0] >= 190 && out[0] <= 192, "got {}", out[0]);
    }

    #[test]
    fn diff_exact_is_ternary() {
        // Use deltas well above f32 mantissa precision at 0.1 (~1.5e-8) so the
        // representable difference is non-zero.
        let o = f32_bytes(&[0.1, 0.1, 0.1, 0.1]);
        let m = f32_bytes(&[0.1, 0.1 + 1e-4, 0.1 - 1e-4, 0.5]);
        let out = Dtype::F32.diff_to_u8(&o, Dtype::F32, &m, DiffMetric::Exact, 0.0);
        // Identical → 127; any positive delta → 254; any negative → 0.
        assert_eq!(out, vec![127, 254, 0, 254]);
    }

    #[test]
    fn diff_non_finite_paints_white() {
        let o = f32_bytes(&[0.1, f32::NAN, 0.1]);
        let m = f32_bytes(&[0.1, 0.1, f32::INFINITY]);
        for metric in [DiffMetric::Rms, DiffMetric::AbsLog, DiffMetric::Exact] {
            let out = Dtype::F32.diff_to_u8(&o, Dtype::F32, &m, metric, 0.1);
            assert_eq!(out[0], 127, "{metric:?} same value");
            assert_eq!(out[1], 255, "{metric:?} NaN in orig");
            assert_eq!(out[2], 255, "{metric:?} Inf in mod");
        }
    }

    #[test]
    fn rms_from_buf_basic() {
        // RMS of [1, -1, 1, -1] is 1.
        let b = f32_bytes(&[1.0, -1.0, 1.0, -1.0]);
        let r = rms_from_buf(Dtype::F32, &b);
        assert!((r - 1.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn rms_from_buf_ignores_non_finite() {
        let b = f32_bytes(&[2.0, f32::NAN, -2.0, f32::INFINITY, 2.0]);
        // Finite values: [2, -2, 2] → mean(x²)=4 → rms=2.
        let r = rms_from_buf(Dtype::F32, &b);
        assert!((r - 2.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn rms_from_buf_empty_is_zero() {
        let r = rms_from_buf(Dtype::F32, &[]);
        assert_eq!(r, 0.0);
    }
}
