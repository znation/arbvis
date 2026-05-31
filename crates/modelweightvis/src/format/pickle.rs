//! PyTorch pickle (`.bin` / `.pth` / `.pt`) header parser.
//!
//! Modern PyTorch (`torch.save` since 1.6) writes a zip archive containing:
//!   - one `*/data.pkl` entry: a pickle stream describing the `state_dict`
//!     — tensor name → `_rebuild_tensor_v2(storage, offset, size, stride, …)`
//!   - one `*/data/N` entry per storage: the raw little-endian element bytes
//!
//! `candle_core::pickle::read_pth_tensor_info` walks the pickle opcode stream
//! without invoking `__reduce__` / `find_class`, so loading untrusted headers
//! is safe — it never instantiates Python objects, just collects the
//! `(name, dtype, layout, storage_path, storage_size)` quintuples we need.
//!
//! What candle does *not* expose is the absolute byte offset of each
//! storage entry inside the file. We close that gap with the `zip` crate's
//! `ZipFile::data_start()`: open the same archive a second time, look up
//! each tensor's storage entry by name, and stitch the byte range together.
//!
//! Compression: PyTorch stores tensor data with `compression = Stored` (no
//! compression). We bail on compressed entries because their on-disk bytes
//! don't correspond to dequantizable little-endian elements.

use std::fs::File;
use std::path::Path;

use candle_core::pickle::read_pth_tensor_info;
use candle_core::DType;
use image::Rgb;
use zip::ZipArchive;

use super::dtype::Dtype;
use super::types::TensorMeta;

/// Parsed PyTorch pickle file header — every tensor's absolute byte range
/// in the file, plus the byte offset where the first tensor's data starts
/// (used by [`build_color_ranges`] to draw the "everything before tensor
/// data" header band).
pub struct PickleHeader {
    pub tensors: Vec<TensorMeta>,
    /// Byte offset of the earliest tensor's data — i.e. where the zip
    /// local file headers end and the first tensor's raw bytes begin.
    /// For files with no tensors this equals `file_size` (whole file is
    /// header).
    pub tensor_data_offset: u64,
}

/// Parse a PyTorch `.bin` / `.pth` / `.pt` header from a local file path.
///
/// Returns the per-tensor `TensorMeta` list (sorted by `file_start`) and the
/// offset where the first tensor's bytes start.
///
/// Errors if:
///   - the file isn't a valid zip archive
///   - the pickle stream uses constructs candle can't decode
///   - any matched storage entry is compressed (we'd need a decompressor
///     on the hot per-tile path, which arbvis avoids)
pub fn parse_header(path: &Path) -> anyhow::Result<PickleHeader> {
    let infos = read_pth_tensor_info(path, false, None)
        .map_err(|e| anyhow::anyhow!("pickle: pickle stream parse failed: {e}"))?;
    let mut zip = ZipArchive::new(File::open(path)?)
        .map_err(|e| anyhow::anyhow!("pickle: zip open failed: {e}"))?;

    let mut tensors = Vec::with_capacity(infos.len());
    for info in infos {
        let dtype = map_candle_dtype(info.dtype);
        let shape: Vec<u64> = info
            .layout
            .shape()
            .dims()
            .iter()
            .map(|&d| d as u64)
            .collect();
        let elem_bytes = info.dtype.size_in_bytes() as u64;
        let elem_count: u64 = info.layout.shape().elem_count() as u64;

        let entry = zip
            .by_name(&info.path)
            .map_err(|e| anyhow::anyhow!("pickle: storage entry '{}' missing: {e}", info.path))?;
        if entry.compression() != zip::CompressionMethod::Stored {
            anyhow::bail!(
                "pickle: storage entry '{}' is compressed ({:?}); arbvis only supports STORE",
                info.path,
                entry.compression()
            );
        }
        let storage_start = entry.data_start();
        // `layout.start_offset()` is already in bytes (`offset * dtype.size_in_bytes()`
        // in candle's `rebuild_args`), so don't multiply it again.
        let file_start = storage_start + info.layout.start_offset() as u64;
        let file_end = file_start + elem_count * elem_bytes;

        tensors.push(TensorMeta {
            name: info.name,
            dtype,
            shape,
            file_start,
            file_end,
            packed_sidecars: None,
        });
    }
    tensors.sort_by_key(|t| t.file_start);
    let tensor_data_offset = tensors.first().map(|t| t.file_start).unwrap_or(0);

    Ok(PickleHeader {
        tensors,
        tensor_data_offset,
    })
}

/// Translate a candle-core `DType` into the arbvis `Dtype`. PyTorch's
/// pickle path only ever emits a small set of storage classes (F32/F64/F16/
/// BF16/U8/I64 — see `rebuild_args` in candle-core's pickle.rs), but we
/// handle the full candle enum so the door stays open if candle widens it.
fn map_candle_dtype(d: DType) -> Dtype {
    match d {
        DType::F32 => Dtype::F32,
        DType::F64 => Dtype::F64,
        DType::F16 => Dtype::F16,
        DType::BF16 => Dtype::BF16,
        DType::U8 => Dtype::U8,
        DType::U32 => Dtype::U32,
        DType::I16 => Dtype::I16,
        DType::I32 => Dtype::I32,
        DType::I64 => Dtype::I64,
        DType::F8E4M3 => Dtype::F8E4M3,
        // Variants candle exposes for completeness but PyTorch never writes
        // via standard `torch.save`. Map them to `Unknown` so visualisation
        // falls back to "grey opaque" rather than panicking.
        _ => Dtype::Unknown,
    }
}

/// Build a sorted list of `(start, end, color)` ranges covering the entire
/// file. Everything before the first tensor (the zip central directory +
/// local file headers + the `data.pkl` blob) is rendered as a grey header
/// band; each tensor region gets its dtype color; gaps in between are black.
pub fn build_color_ranges(
    tensors: &[TensorMeta],
    tensor_data_offset: u64,
    file_size: u64,
) -> Vec<(u64, u64, Rgb<u8>)> {
    let mut ranges: Vec<(u64, u64, Rgb<u8>)> = Vec::with_capacity(tensors.len() + 2);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-build a minimal PyTorch zip archive on disk:
    ///   - one storage entry (8 f32 values, raw little-endian bytes)
    ///   - one `data.pkl` entry pointing at it as `weight` (state_dict layout)
    ///
    /// Mirrors what `torch.save({"weight": tensor})` produces, minus the
    /// torch-version metadata files. Verifies the parser end-to-end:
    /// pickle stream → tensor info → zip lookup → absolute byte range.
    fn write_minimal_pth(path: &Path) {
        use std::io::Write;
        use zip::write::{SimpleFileOptions, ZipWriter};
        let f = File::create(path).unwrap();
        let mut zip = ZipWriter::new(f);
        let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

        // Storage entry: 8 contiguous f32 values.
        zip.start_file("model/data/0", opts).unwrap();
        let values: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut bytes = Vec::with_capacity(32);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        zip.write_all(&bytes).unwrap();

        // Pickle stream describing `{"weight": _rebuild_tensor_v2(storage_0, …)}`.
        // Hand-assembled protocol-2 ops: this is the same opcode subset
        // `torch.save` emits for a simple state_dict.
        zip.start_file("model/data.pkl", opts).unwrap();
        let pkl = build_pickle();
        zip.write_all(&pkl).unwrap();
        zip.finish().unwrap();
    }

    /// Hand-rolled pickle bytestream:
    ///   PROTO 2 / EMPTY_DICT / MARK / BINUNICODE "weight" /
    ///   GLOBAL torch._utils._rebuild_tensor_v2 / MARK /
    ///       BINPERSID ("storage", FloatStorage, "0", "cpu", 8) /
    ///       BININT1 0 (start_offset) /
    ///       MARK BININT1 2 BININT1 4 TUPLE (size = (2,4)) /
    ///       MARK BININT1 4 BININT1 1 TUPLE (stride = (4,1)) /
    ///       NEWFALSE (requires_grad) / NONE (backward_hooks) /
    ///   TUPLE / REDUCE / SETITEMS / STOP
    ///
    /// candle's pickle reader recognises this as the standard rebuild form
    /// (`rebuild_args` in candle-core's pickle.rs). We use `BINUNICODE`
    /// (opcode `X`, u32-length-prefixed) rather than `SHORT_BINUNICODE`
    /// (opcode `0x8c`) because candle's reader doesn't decode the latter.
    fn build_pickle() -> Vec<u8> {
        let mut p = vec![
            0x80, // PROTO
            0x02, b'}', // EMPTY_DICT
            b'(', // MARK
        ];

        write_binunicode(&mut p, "weight");

        // Value: GLOBAL "torch._utils\n_rebuild_tensor_v2\n"
        p.push(b'c');
        p.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n");

        // Args tuple for REDUCE: MARK <args...> TUPLE
        p.push(b'('); // MARK

        // arg 0: storage = BINPERSID popping a tuple
        //   (b"storage", FloatStorage_class, "0", "cpu", 8)
        p.push(b'('); // MARK
        write_binunicode(&mut p, "storage");
        // GLOBAL torch FloatStorage
        p.push(b'c');
        p.extend_from_slice(b"torch\nFloatStorage\n");
        write_binunicode(&mut p, "0");
        write_binunicode(&mut p, "cpu");
        p.push(b'K'); // BININT1 8
        p.push(8);
        p.push(b't'); // TUPLE
        p.push(b'Q'); // BINPERSID

        // arg 1: storage_offset = 0
        p.push(b'K');
        p.push(0);

        // arg 2: size = (2, 4)
        p.push(b'('); // MARK
        p.push(b'K');
        p.push(2);
        p.push(b'K');
        p.push(4);
        p.push(b't'); // TUPLE

        // arg 3: stride = (4, 1)
        p.push(b'('); // MARK
        p.push(b'K');
        p.push(4);
        p.push(b'K');
        p.push(1);
        p.push(b't'); // TUPLE

        // arg 4: requires_grad = False
        p.push(0x89); // NEWFALSE

        // arg 5: backward_hooks = None
        p.push(b'N');

        p.push(b't'); // TUPLE
        p.push(b'R'); // REDUCE

        p.push(b'u'); // SETITEMS (consumes pairs back to MARK)
        p.push(b'.'); // STOP
        p
    }

    fn write_binunicode(p: &mut Vec<u8>, s: &str) {
        p.push(b'X'); // BINUNICODE (u32 LE length + UTF-8 bytes)
        let bytes = s.as_bytes();
        p.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        p.extend_from_slice(bytes);
    }

    #[test]
    fn parse_minimal_pth() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tiny.bin");
        write_minimal_pth(&path);

        let header = parse_header(&path).expect("parse should succeed");
        assert_eq!(header.tensors.len(), 1);
        let t = &header.tensors[0];
        assert_eq!(t.name, "weight");
        assert_eq!(t.dtype, Dtype::F32);
        assert_eq!(t.shape, vec![2, 4]);
        // 8 f32 elements at 4 bytes each = 32 bytes.
        assert_eq!(t.file_end - t.file_start, 32);
        // Byte range must fall inside the file.
        let file_size = std::fs::metadata(&path).unwrap().len();
        assert!(t.file_end <= file_size, "file_end past EOF");

        // Verify we can read the original little-endian f32s back from the
        // claimed byte range — proves we resolved the zip data_start correctly.
        let raw = std::fs::read(&path).unwrap();
        let slice = &raw[t.file_start as usize..t.file_end as usize];
        for k in 0..8 {
            let bytes: [u8; 4] = slice[k * 4..(k + 1) * 4].try_into().unwrap();
            let v = f32::from_le_bytes(bytes);
            assert_eq!(v, k as f32, "elem {k}");
        }
    }

    #[test]
    fn build_ranges_pickle() {
        let t = TensorMeta {
            name: "weight".to_string(),
            dtype: Dtype::F32,
            shape: vec![2, 4],
            file_start: 200,
            file_end: 232,
            packed_sidecars: None,
        };
        let r = build_color_ranges(&[t], 200, 400);
        assert_eq!(r[0], (0, 200, Rgb([100, 100, 100])));
        assert_eq!(r[1], (200, 232, Dtype::F32.to_color()));
        assert_eq!(r[2], (232, 400, Rgb([0, 0, 0])));
    }
}
