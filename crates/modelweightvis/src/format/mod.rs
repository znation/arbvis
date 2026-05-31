//! Multi-format model parsing.
//!
//! arbvis recognises three model file formats: `.safetensors` (HuggingFace's
//! reference format), `.gguf` (llama.cpp's quantised inference format), and
//! PyTorch pickle (`.bin` / `.pth` / `.pt`, the pre-safetensors HF legacy
//! format). Each one has its own header layout, tensor name convention, and
//! dtype universe, but downstream code only ever sees the format-agnostic
//! types re-exported here: [`TensorMeta`], [`ModelInfo`], [`Dtype`],
//! [`DiffMetric`], and [`DiffFill`].
//!
//! Format dispatch is via the [`SourceFormat`] enum. A `match` arm per
//! format keeps the hot per-tile decode path monomorphic; adding a fourth
//! format (MLX, ONNX, …) means adding one variant and one sibling module.

use std::path::Path;

use image::Rgb;

pub mod dtype;
pub mod gguf;
pub mod moe;
pub mod name_map;
pub mod pickle;
pub mod safetensors;
pub mod types;

pub use dtype::{rms_from_buf, Dtype, ElementStride, TensorElementReader};
pub use name_map::to_canonical;
// DiffMetric / DiffFill moved to arbvis; re-export so the many
// `format::DiffMetric` call sites below keep compiling.
pub use arbvis::DiffMetric;
pub use types::{ModelInfo, TensorMeta, ABS_LOG_MAX, ABS_LOG_MIN, K_RMS_SAT, RMS_FLOOR};

/// Which model file format produced a [`ModelInfo`].
///
/// `SourceFormat::from_path` / `from_name` recognises a file by extension;
/// unrecognised paths return `None`, signalling "treat as plain bytes" (no
/// per-tensor coloring, no architectural layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    Safetensors,
    Gguf,
    /// PyTorch pickle (`.bin` / `.pth` / `.pt`) — a zip archive containing
    /// a `data.pkl` opcode stream and one storage entry per tensor. Decoded
    /// safely (no Python execution) via `candle_core::pickle`.
    Pickle,
}

impl SourceFormat {
    /// Recognise a format from a local path's extension.
    pub fn from_path(p: &Path) -> Option<Self> {
        let ext = p.extension().and_then(|e| e.to_str())?;
        Self::from_extension(ext)
    }

    /// Recognise a format from a filename string (used for remote HF files
    /// where we only have a name, not a `Path`).
    pub fn from_name(name: &str) -> Option<Self> {
        let dot = name.rfind('.')?;
        Self::from_extension(&name[dot + 1..])
    }

    fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "safetensors" => Some(SourceFormat::Safetensors),
            "gguf" => Some(SourceFormat::Gguf),
            // PyTorch's three torch.save() extensions. `.bin` is also used
            // by other formats (HF tokenizer binaries, generic blobs); if
            // a non-pickle `.bin` slips in the parser will error cleanly
            // and the data layer falls back to "treat as plain bytes".
            "bin" | "pth" | "pt" => Some(SourceFormat::Pickle),
            _ => None,
        }
    }

    /// Render a tensor name into the cross-format canonical form used by the
    /// diff matcher. See [`name_map::to_canonical`].
    pub fn canonical_name(self, raw: &str) -> String {
        to_canonical(self, raw)
    }

    /// Build display color ranges over the whole file. Dispatches to the
    /// per-format builder. Currently called only from tests; the data
    /// layer's `load_model_info` reaches into the per-format builders
    /// directly to avoid an extra method call on the hot setup path.
    #[allow(dead_code)]
    pub fn build_color_ranges(
        self,
        tensors: &[TensorMeta],
        header_end: u64,
        file_size: u64,
    ) -> Vec<(u64, u64, Rgb<u8>)> {
        match self {
            SourceFormat::Safetensors => {
                safetensors::build_color_ranges(tensors, header_end, file_size)
            }
            SourceFormat::Gguf => gguf::build_color_ranges(tensors, header_end, file_size),
            SourceFormat::Pickle => pickle::build_color_ranges(tensors, header_end, file_size),
        }
    }
}

/// Return the display color for a byte at `pos` within a file.
///
/// `ranges` must be the output of [`SourceFormat::build_color_ranges`]
/// (sorted by start, non-overlapping).
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
    use std::path::Path;

    #[test]
    fn from_path_safetensors() {
        assert_eq!(
            SourceFormat::from_path(Path::new("model.safetensors")),
            Some(SourceFormat::Safetensors)
        );
        assert_eq!(
            SourceFormat::from_path(Path::new("/tmp/foo/bar.SAFETENSORS")),
            Some(SourceFormat::Safetensors)
        );
    }

    #[test]
    fn from_path_gguf() {
        assert_eq!(
            SourceFormat::from_path(Path::new("model.gguf")),
            Some(SourceFormat::Gguf)
        );
        assert_eq!(
            SourceFormat::from_path(Path::new("/tmp/model.GGUF")),
            Some(SourceFormat::Gguf)
        );
    }

    #[test]
    fn from_path_pickle() {
        assert_eq!(
            SourceFormat::from_path(Path::new("pytorch_model.bin")),
            Some(SourceFormat::Pickle)
        );
        assert_eq!(
            SourceFormat::from_path(Path::new("model.pth")),
            Some(SourceFormat::Pickle)
        );
        assert_eq!(
            SourceFormat::from_path(Path::new("/tmp/checkpoint.PT")),
            Some(SourceFormat::Pickle)
        );
    }

    #[test]
    fn from_path_unknown() {
        assert_eq!(SourceFormat::from_path(Path::new("/etc/hosts")), None);
        assert_eq!(SourceFormat::from_path(Path::new("foo")), None);
        assert_eq!(SourceFormat::from_path(Path::new("model.onnx")), None);
    }

    #[test]
    fn from_name_remote() {
        assert_eq!(
            SourceFormat::from_name("Qwen3-30B-A3B-Q4_K_M.gguf"),
            Some(SourceFormat::Gguf)
        );
        assert_eq!(
            SourceFormat::from_name("model-00001-of-00004.safetensors"),
            Some(SourceFormat::Safetensors)
        );
        assert_eq!(SourceFormat::from_name("config.json"), None);
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
