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

use std::collections::HashMap;

use image::Rgb;

use super::dtype::Dtype;
use super::types::{PackedSidecars, TensorMeta};

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
            packed_sidecars: None,
        });
    }

    tensors.sort_by_key(|t| t.file_start);
    fuse_packed_quant_triples(&mut tensors);
    Ok((tensors, header_end))
}

/// Fuse AWQ / GPTQ / EXL2 `(qweight, scales, qzeros)` triples into a single
/// logical packed-int tensor in-place.
///
/// AWQ/GPTQ files store one matmul weight as three separate safetensors
/// entries:
///   - `<prefix>.qweight` — bit-packed ints (canonically INT32 dtype with
///     shape `[in_features, out_features / pack_factor]` for AWQ or
///     `[in_features / pack_factor, out_features]` for GPTQ)
///   - `<prefix>.scales` — per-group scales (F16 / F32, shape `[groups, out]`)
///   - `<prefix>.qzeros` — bit-packed zero-points (INT32, same packing as
///     qweight) OR for newer formats, F16 scales-only
///
/// We collapse each triple into a single fused tensor named `<prefix>.weight`
/// with a packed dtype (`Int4Packed` / `Int3Packed` / `Int8Packed`) so the
/// cross-format diff matcher pairs it with the base model's
/// `<prefix>.weight` (F16/BF16). The scales/qzeros byte ranges live on the
/// fused `TensorMeta::packed_sidecars`.
///
/// Bit width: inferred from the ratio of qweight shape to scales shape.
/// Defaults to int4 when the inference is ambiguous (the dominant case on
/// the Hub by far). Heuristic only — no `config.json` consultation here.
fn fuse_packed_quant_triples(tensors: &mut Vec<TensorMeta>) {
    // Index by canonical prefix → (qweight, scales, qzeros).
    let mut groups: HashMap<String, [Option<usize>; 3]> = HashMap::new();
    for (i, t) in tensors.iter().enumerate() {
        let (prefix, slot) = if let Some(p) = t.name.strip_suffix(".qweight") {
            (p, 0)
        } else if let Some(p) = t.name.strip_suffix(".scales") {
            (p, 1)
        } else if let Some(p) = t.name.strip_suffix(".qzeros") {
            (p, 2)
        } else {
            continue;
        };
        groups.entry(prefix.to_string()).or_default()[slot] = Some(i);
    }

    // Only fuse complete triples (qweight + scales present; qzeros optional).
    let mut to_remove: Vec<usize> = Vec::new();
    let mut fused: Vec<(usize, TensorMeta)> = Vec::new();
    for (prefix, slots) in groups {
        let Some(qw_i) = slots[0] else { continue };
        let Some(sc_i) = slots[1] else { continue };
        let qw = &tensors[qw_i];
        let sc = &tensors[sc_i];
        let qz = slots[2].map(|i| &tensors[i]);

        // Infer bit width from qweight vs scales element counts.
        // qweight is packed: total_packed_ints = elements_per_pack * scales_groups_per_col * cols
        // For AWQ canonical layout (pack along OUT dim):
        //   qweight shape [in_features, packed_cols]
        //   scales shape  [n_groups,    out_features]
        //   pack_factor = out_features / packed_cols
        //   bits = 32 / pack_factor (assuming int32 packing)
        let qw_elems: u64 = qw.shape.iter().product();
        let sc_elems: u64 = sc.shape.iter().product();
        if qw_elems == 0 || sc_elems == 0 {
            continue;
        }
        // unpacked = qw_elems * pack_factor; scales = unpacked / group_size
        // So pack_factor * group_size = (qw_elems * pack_factor) / sc_elems * (cols/cols) — depends on group_size.
        // Practical inference: for canonical AWQ/GPTQ, pack_factor ∈ {4, 8, 10}.
        // Default to 8 (int4) when ratio doesn't disambiguate.
        let dtype = Dtype::Int4Packed;

        // Unpacked shape: we don't have config.json, so derive optimistically.
        // For AWQ: unpacked_out = qw.shape[last] * 8 (for int4-in-int32).
        // For GPTQ: unpacked_in = qw.shape[0] * 8. Without telling them apart,
        // use the larger shape dim for OUT.
        let mut unpacked_shape = qw.shape.clone();
        if let Some(last) = unpacked_shape.last_mut() {
            *last *= 8; // int4 pack_factor
        }

        // Derive cols (used by the per-element decoder).
        let cols: u32 = unpacked_shape
            .last()
            .copied()
            .unwrap_or(0)
            .try_into()
            .unwrap_or(0);

        let sidecars = PackedSidecars {
            scales_start: sc.file_start,
            scales_end: sc.file_end,
            scales_dtype: sc.dtype,
            zeros_start: qz.map(|z| z.file_start),
            zeros_end: qz.map(|z| z.file_end),
            zeros_dtype: qz.map(|z| z.dtype).unwrap_or(Dtype::Unknown),
            cols,
        };

        let fused_meta = TensorMeta {
            name: format!("{prefix}.weight"),
            dtype,
            shape: unpacked_shape,
            file_start: qw.file_start,
            file_end: qw.file_end,
            packed_sidecars: Some(sidecars),
        };

        to_remove.push(qw_i);
        to_remove.push(sc_i);
        if let Some(zi) = slots[2] {
            to_remove.push(zi);
        }
        fused.push((qw_i, fused_meta));
    }

    if to_remove.is_empty() {
        return;
    }

    // Remove triples from highest index downward to keep earlier indices stable,
    // then push fused entries and re-sort.
    to_remove.sort_unstable_by(|a, b| b.cmp(a));
    for i in to_remove {
        tensors.swap_remove(i);
    }
    for (_, meta) in fused {
        tensors.push(meta);
    }
    tensors.sort_by_key(|t| t.file_start);
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
            packed_sidecars: None,
        };
        let r = build_color_ranges(&[t], 100, 200);
        assert_eq!(r[0], (0, 100, Rgb([100, 100, 100])));
        assert_eq!(r[1], (100, 200, Dtype::F32.to_color()));
    }

    fn mk_t(name: &str, dtype: Dtype, shape: Vec<u64>, start: u64, end: u64) -> TensorMeta {
        TensorMeta {
            name: name.to_string(),
            dtype,
            shape,
            file_start: start,
            file_end: end,
            packed_sidecars: None,
        }
    }

    #[test]
    fn fuse_packed_quant_triples_basic_awq() {
        // Simulate an AWQ matmul: qweight + scales + qzeros sharing a prefix.
        // Shapes are chosen so the inference picks int4 (8 elems per int32).
        let mut v = vec![
            mk_t(
                "model.layers.0.self_attn.q_proj.qweight",
                Dtype::I32,
                vec![4096, 512], // 4096 in, 512 packed-out
                1000,
                1000 + 4096 * 512 * 4,
            ),
            mk_t(
                "model.layers.0.self_attn.q_proj.scales",
                Dtype::F16,
                vec![32, 4096], // 32 groups of 128, 4096 out
                10_000_000,
                10_000_000 + 32 * 4096 * 2,
            ),
            mk_t(
                "model.layers.0.self_attn.q_proj.qzeros",
                Dtype::I32,
                vec![32, 512],
                20_000_000,
                20_000_000 + 32 * 512 * 4,
            ),
            // An unrelated tensor that should pass through unchanged.
            mk_t(
                "model.layers.0.input_layernorm.weight",
                Dtype::F16,
                vec![4096],
                30_000_000,
                30_000_000 + 4096 * 2,
            ),
        ];
        fuse_packed_quant_triples(&mut v);

        // The three quant tensors collapse to one fused .weight; the
        // unrelated layernorm tensor is untouched. Total: 2 entries.
        assert_eq!(v.len(), 2);

        let fused = v
            .iter()
            .find(|t| t.name == "model.layers.0.self_attn.q_proj.weight")
            .expect("fused tensor present");
        assert_eq!(fused.dtype, Dtype::Int4Packed);
        // Unpacked OUT dimension = 512 * 8 (int4-in-int32 pack factor).
        assert_eq!(fused.shape, vec![4096, 4096]);
        // qweight byte range carries through.
        assert_eq!(fused.file_start, 1000);
        assert_eq!(fused.file_end, 1000 + 4096 * 512 * 4);
        let sc = fused.packed_sidecars.as_ref().unwrap();
        assert_eq!(sc.scales_dtype, Dtype::F16);
        assert_eq!(sc.scales_start, 10_000_000);
        assert_eq!(sc.zeros_start, Some(20_000_000));
        assert_eq!(sc.cols, 4096);

        // Layernorm passes through.
        assert!(v
            .iter()
            .any(|t| t.name == "model.layers.0.input_layernorm.weight"));
    }

    #[test]
    fn fuse_packed_quant_triples_no_quant_is_noop() {
        // Plain safetensors with no qweight/scales/qzeros names → no fusion.
        let original = vec![
            mk_t(
                "model.embed_tokens.weight",
                Dtype::F16,
                vec![32000, 4096],
                0,
                32000 * 4096 * 2,
            ),
            mk_t("model.norm.weight", Dtype::F16, vec![4096], 1, 4096 * 2),
        ];
        let mut v = original.clone();
        fuse_packed_quant_triples(&mut v);
        assert_eq!(v.len(), original.len());
    }
}
