use image::Rgb;

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

    /// Compute the signed-relative diff between matched elements, returning one u8 per element pair.
    /// `self` is the dtype for `orig`; `mod_dtype` is the dtype for `mod_`.
    /// Encoding: 127 = no change, 128–254 = increased, 0–126 = decreased, 255 = non-finite.
    /// No intermediate Vec<f32> is allocated.
    pub fn diff_to_u8(self, orig: &[u8], mod_dtype: Dtype, mod_: &[u8], epsilon: f32) -> Vec<u8> {
        let orig_elem = self.element_size();
        let mod_elem = mod_dtype.element_size();
        orig.chunks_exact(orig_elem)
            .zip(mod_.chunks_exact(mod_elem))
            .map(|(oc, mc)| {
                let o = decode_element(self, oc);
                let m = decode_element(mod_dtype, mc);
                if !o.is_finite() || !m.is_finite() { return 255u8; }
                let signed_rel = (m - o) / o.abs().max(epsilon);
                let brightness = (signed_rel.abs().sqrt().min(1.0) * 127.0).round() as u8;
                if signed_rel >= 0.0 { 127u8 + brightness } else { 127u8 - brightness }
            })
            .collect()
    }

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
}
