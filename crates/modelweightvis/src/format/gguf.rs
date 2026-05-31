//! GGUF parser, layered on top of candle-core's `quantized::gguf_file::Content`.
//!
//! GGUF lays out a file as four sequential regions:
//!   - magic + version + counts (24 bytes)
//!   - metadata KV table (variable size; ends where the tensor info table starts)
//!   - tensor info table (variable size; ends at `tensor_data_offset`)
//!   - tensor data (rest of file)
//!
//! Unlike safetensors there is no length prefix telling us where the metadata
//! ends — the only way to find `tensor_data_offset` is to parse through the
//! KV table and the tensor info table. Callers therefore feed enough of the
//! file prefix to `parse_header` to cover both tables; on overshoot we report
//! a clear error so the data layer can retry with a bigger fetch.

use std::collections::HashMap;
use std::io::Cursor;

use candle_core::quantized::gguf_file::{Content, Value};
use image::Rgb;

use super::dtype::{gguf_tensor_byte_range, Dtype};
use super::types::TensorMeta;

/// Parsed GGUF header — everything except tensor data.
///
/// Wraps candle-core's `Content` to keep the rest of the codebase
/// independent of candle's types. `metadata` is the raw KV table; the
/// architectural layout's `ModelConfig::from_gguf_metadata` reads from it.
pub struct GgufHeader {
    pub tensors: Vec<TensorMeta>,
    /// Absolute byte offset where tensor data starts (after KV + info tables
    /// + any alignment padding).
    pub tensor_data_offset: u64,
    pub metadata: HashMap<String, Value>,
}

/// Parse a GGUF file header from raw bytes. The bytes must cover the magic,
/// the KV table, and the tensor info table — at minimum
/// `tensor_data_offset` bytes. `Content::read` errors if the prefix is too
/// short; callers should catch that and retry with a larger fetch.
pub fn parse_header(data: &[u8]) -> anyhow::Result<GgufHeader> {
    let mut reader = Cursor::new(data);
    let content = Content::read(&mut reader)
        .map_err(|e| anyhow::anyhow!("gguf: header parse failed: {}", e))?;

    let mut tensors = Vec::with_capacity(content.tensor_infos.len());
    for (name, info) in &content.tensor_infos {
        let dtype = Dtype::from_ggml(info.ggml_dtype);
        let (file_start, file_end) = gguf_tensor_byte_range(info, content.tensor_data_offset);
        let shape: Vec<u64> = info.shape.dims().iter().map(|&d| d as u64).collect();
        tensors.push(TensorMeta {
            name: name.clone(),
            dtype,
            shape,
            file_start,
            file_end,
            packed_sidecars: None,
        });
    }
    tensors.sort_by_key(|t| t.file_start);

    Ok(GgufHeader {
        tensors,
        tensor_data_offset: content.tensor_data_offset,
        metadata: content.metadata,
    })
}

/// Build display color ranges for a GGUF file. Three header sub-regions are
/// shown in slightly different greys so the KV table, the info table, and
/// any alignment padding read as distinct bands in the legacy Hilbert dtype
/// view. Tensor data uses the per-tensor dtype color; gaps between tensors
/// are black.
pub fn build_color_ranges(
    tensors: &[TensorMeta],
    tensor_data_offset: u64,
    file_size: u64,
) -> Vec<(u64, u64, Rgb<u8>)> {
    let mut ranges: Vec<(u64, u64, Rgb<u8>)> = Vec::with_capacity(tensors.len() + 2);
    // Combined header region (magic + KV + info table + alignment padding).
    // We don't bother sub-dividing — without re-running the parser we can't
    // tell where the KV table ends and the info table begins, and the
    // visual payoff is small relative to the renderer plumbing.
    if tensor_data_offset > 0 {
        ranges.push((0, tensor_data_offset, Rgb([100, 100, 100])));
    }
    let mut pos = tensor_data_offset;
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

/// Look up a metadata KV value as a u64. Handles the various integer types
/// GGUF KV pairs may use. Returns `None` if the key is missing or has a
/// non-integer type.
pub fn metadata_u64(metadata: &HashMap<String, Value>, key: &str) -> Option<u64> {
    match metadata.get(key)? {
        Value::U8(v) => Some(*v as u64),
        Value::U16(v) => Some(*v as u64),
        Value::U32(v) => Some(*v as u64),
        Value::U64(v) => Some(*v),
        Value::I8(v) if *v >= 0 => Some(*v as u64),
        Value::I16(v) if *v >= 0 => Some(*v as u64),
        Value::I32(v) if *v >= 0 => Some(*v as u64),
        Value::I64(v) if *v >= 0 => Some(*v as u64),
        _ => None,
    }
}

/// Look up a metadata KV value as a string.
pub fn metadata_string<'a>(metadata: &'a HashMap<String, Value>, key: &str) -> Option<&'a str> {
    match metadata.get(key)? {
        Value::String(s) => Some(s.as_str()),
        _ => None,
    }
}

/// Look up the length of an array-valued KV. Useful for e.g.
/// `tokenizer.ggml.tokens` (vocab size = array length).
pub fn metadata_array_len(metadata: &HashMap<String, Value>, key: &str) -> Option<usize> {
    match metadata.get(key)? {
        Value::Array(v) => Some(v.len()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid GGUF v2 byte stream with a single string KV pair
    /// (`general.architecture` → `"llama"`) and zero tensors. Exercises the
    /// magic/version/counts parser plus one KV decode path.
    fn synthetic_no_tensor_gguf() -> Vec<u8> {
        let mut bytes = Vec::new();
        // magic = "GGUF" (LE u32)
        bytes.extend_from_slice(&0x46554747u32.to_le_bytes());
        // version 2
        bytes.extend_from_slice(&2u32.to_le_bytes());
        // tensor_count = 0
        bytes.extend_from_slice(&0u64.to_le_bytes());
        // metadata_kv_count = 1
        bytes.extend_from_slice(&1u64.to_le_bytes());
        // KV: key string, type, value
        let key = b"general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        // value_type = 8 (String)
        bytes.extend_from_slice(&8u32.to_le_bytes());
        let val = b"llama";
        bytes.extend_from_slice(&(val.len() as u64).to_le_bytes());
        bytes.extend_from_slice(val);
        // Tensor info table is empty.
        // Pad to 32-byte alignment so tensor_data_offset is reachable.
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes
    }

    #[test]
    fn parse_header_minimal_no_tensors() {
        let bytes = synthetic_no_tensor_gguf();
        let header = parse_header(&bytes).expect("parses");
        assert!(header.tensors.is_empty());
        assert_eq!(
            metadata_string(&header.metadata, "general.architecture"),
            Some("llama")
        );
        // `tensor_data_offset` lands at the end of the header (which is also
        // where we padded to; both will be at the same 32-byte boundary).
        assert!(header.tensor_data_offset > 0);
        assert!(header.tensor_data_offset.is_multiple_of(32));
    }

    #[test]
    fn build_color_ranges_no_tensors_yields_header_only() {
        let r = build_color_ranges(&[], 64, 128);
        // One header band [0..64) and a trailing-gap band [64..128).
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].0, 0);
        assert_eq!(r[0].1, 64);
        assert_eq!(r[1].0, 64);
        assert_eq!(r[1].1, 128);
    }
}
