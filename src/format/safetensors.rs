//! Safetensors header parser.
//!
//! The format is:
//!   - 8 bytes: little-endian u64 `header_size`
//!   - `header_size` bytes: JSON object mapping tensor name → `{dtype, shape,
//!     data_offsets: [start, end]}` plus an optional `__metadata__` key
//!   - tensor data at byte offset `8 + header_size`
//!
//! `data_offsets` are relative to the end of the header; `TensorMeta`
//! stores absolute file offsets.

use image::Rgb;

use super::dtype::Dtype;
use super::types::TensorMeta;

/// Parse a safetensors file's header from raw bytes.
///
/// Returns tensors sorted by `file_start` and the absolute end offset of the
/// header region (= start of tensor data).
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
        anyhow::bail!(
            "safetensors: header_size={} exceeds 100 MB safety limit",
            header_size
        );
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
        let dtype = Dtype::from_safetensors_str(dtype_str);

        let shape: Vec<u64> = tensor_obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("safetensors: tensor '{}' missing 'shape'", name))?
            .iter()
            .map(|d| {
                d.as_u64().ok_or_else(|| {
                    anyhow::anyhow!("safetensors: tensor '{}' shape dim is not u64", name)
                })
            })
            .collect::<anyhow::Result<_>>()?;

        let offsets = tensor_obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .filter(|a| a.len() == 2)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "safetensors: tensor '{}' missing valid 'data_offsets'",
                    name
                )
            })?;
        let rel_start = offsets[0].as_u64().ok_or_else(|| {
            anyhow::anyhow!("safetensors: tensor '{}' data_offsets[0] not u64", name)
        })?;
        let rel_end = offsets[1].as_u64().ok_or_else(|| {
            anyhow::anyhow!("safetensors: tensor '{}' data_offsets[1] not u64", name)
        })?;

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

/// Build a sorted list of `(start, end, color)` ranges covering the entire
/// file. The header region is grey, each tensor region gets its dtype color,
/// gaps are black.
pub fn build_color_ranges(
    tensors: &[TensorMeta],
    header_end: u64,
    file_size: u64,
) -> Vec<(u64, u64, Rgb<u8>)> {
    let mut ranges: Vec<(u64, u64, Rgb<u8>)> = Vec::with_capacity(tensors.len() + 2);
    if header_end > 0 {
        ranges.push((0, header_end, Rgb([100, 100, 100])));
    }
    let mut pos = header_end;
    for t in tensors {
        if t.file_start > pos {
            ranges.push((pos, t.file_start, Rgb([0, 0, 0])));
        }
        if t.file_end > t.file_start {
            ranges.push((t.file_start, t.file_end, t.dtype.to_color()));
        }
        pos = t.file_end;
    }
    if pos < file_size {
        ranges.push((pos, file_size, Rgb([0, 0, 0])));
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_ranges_header_first() {
        let t = TensorMeta {
            name: "t".to_string(),
            dtype: Dtype::F32,
            shape: vec![4],
            file_start: 100,
            file_end: 200,
        };
        let r = build_color_ranges(&[t], 100, 200);
        assert_eq!(r[0], (0, 100, Rgb([100, 100, 100])));
        assert_eq!(r[1], (100, 200, Dtype::F32.to_color()));
    }
}
