use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use image::Rgb;
use indicatif::ProgressBar;
use memmap2::Mmap;

use crate::safetensors::{self, TensorMeta};

pub struct SafetensorsInfo {
    pub tensors: Vec<TensorMeta>,
    pub color_ranges: Vec<(u64, u64, Rgb<u8>)>,
}

/// Backing storage for a source's bytes.
pub enum Data {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Data {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Data::Mapped(m) => m,
            Data::Owned(v) => v,
        }
    }
}

/// How a source's bytes are stored.
pub enum SourceKind {
    Buffered(Vec<u8>),
    File(PathBuf),
    Diff { original: PathBuf, modified: PathBuf },
}

/// Metadata and storage descriptor for one input.
pub struct Source {
    pub file_idx: usize,
    pub kind: SourceKind,
    pub byte_size: u64,
    /// Populated for safetensors sources (including diff buffers derived from them).
    pub safetensors: Option<SafetensorsInfo>,
    /// Override the display name (used when kind is Buffered but has a real filename).
    pub name_override: Option<String>,
    /// If set, only the bytes [start, end) within the file are used for this source.
    /// Used for per-tensor expansion so each tensor occupies its own Hilbert region.
    pub byte_range: Option<(u64, u64)>,
}

impl Source {
    /// Human-readable name for this source (file name or "stdin").
    pub fn name(&self) -> String {
        if let Some(ref n) = self.name_override {
            return n.clone();
        }
        match &self.kind {
            SourceKind::File(p) => p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.to_string_lossy().into_owned()),
            SourceKind::Buffered(_) => "stdin".to_string(),
            SourceKind::Diff { original, .. } => original
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| original.to_string_lossy().into_owned()),
        }
    }
}

/// Build sources and return total byte count.
///
/// Files are opened lazily (one at a time) to avoid exhausting OS fd limits.
/// Stdin is buffered into memory upfront since its size is unknown.
///
/// For .safetensors files (or when `format_safetensors` is true): the header is
/// parsed and the file is expanded into one Source per tensor (plus a `__header__`
/// Source for the JSON metadata region). Each tensor occupies its own contiguous
/// Hilbert region, eliminating the block-boundary artifacting that occurs when
/// 65536-byte tile boundaries cut through tensor data.
pub fn prepare_sources(files: &[PathBuf], format_safetensors: bool) -> anyhow::Result<(Vec<Source>, u64)> {
    if files.is_empty() {
        log::info!("Reading stdin...");
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf)?;
        let len = buf.len() as u64;
        return Ok((
            vec![Source {
                file_idx: 0,
                kind: SourceKind::Buffered(buf),
                byte_size: len,
                safetensors: None,
                name_override: None,
                byte_range: None,
            }],
            len,
        ));
    }

    let mut sources = Vec::new();
    let mut total = 0u64;
    for path in files.iter() {
        let size = match std::fs::metadata(path) {
            Ok(m) => m.len(),
            Err(e) => {
                eprintln!("warning: {}: {} — skipping", path.display(), e);
                continue;
            }
        };

        let is_st = format_safetensors
            || path.extension().and_then(|e| e.to_str()) == Some("safetensors");

        if is_st {
            match expand_safetensors_sources(path, size, &mut sources) {
                Ok(file_total) => { total += file_total; }
                Err(e) => {
                    eprintln!("warning: {}: failed to parse safetensors header: {} — treating as plain binary", path.display(), e);
                    total += size;
                    sources.push(Source {
                        file_idx: sources.len(),
                        kind: SourceKind::File(path.clone()),
                        byte_size: size,
                        safetensors: None,
                        name_override: None,
                        byte_range: None,
                    });
                }
            }
        } else {
            total += size;
            sources.push(Source {
                file_idx: sources.len(),
                kind: SourceKind::File(path.clone()),
                byte_size: size,
                safetensors: None,
                name_override: None,
                byte_range: None,
            });
        }
    }
    Ok((sources, total))
}

/// Expand a single .safetensors file into per-tensor Sources, appending them to `out`.
///
/// Emits one `__header__` Source covering the JSON metadata region, then one Source
/// per tensor sorted by file offset. Each Source gets local color_ranges so that
/// existing dtype-coloring code in the rendering pipeline works without modification.
///
/// Returns the file's total byte count (header + all tensor data).
fn expand_safetensors_sources(
    path: &Path,
    file_size: u64,
    out: &mut Vec<Source>,
) -> anyhow::Result<u64> {
    let st_info = load_safetensors_info(path, file_size)?;
    let header_end = st_info.tensors.first().map(|t| t.file_start).unwrap_or(file_size);

    // Header source (gray dtype coloring).
    let header_size = header_end;
    out.push(Source {
        file_idx: out.len(),
        kind: SourceKind::File(path.to_path_buf()),
        byte_size: header_size,
        safetensors: Some(SafetensorsInfo {
            tensors: vec![],
            color_ranges: vec![(0, header_size, image::Rgb([100, 100, 100]))],
        }),
        name_override: Some("__header__ (metadata)".to_string()),
        byte_range: Some((0, header_size)),
    });

    // Per-tensor sources.
    for tensor in &st_info.tensors {
        let tensor_size = tensor.file_end - tensor.file_start;
        let dtype_color = tensor.dtype.to_color();
        // Build a local TensorMeta with positions relative to this source (0-based).
        let local_tensor = safetensors::TensorMeta {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
            file_start: 0,
            file_end: tensor_size,
        };
        out.push(Source {
            file_idx: out.len(),
            kind: SourceKind::File(path.to_path_buf()),
            byte_size: tensor_size,
            safetensors: Some(SafetensorsInfo {
                tensors: vec![local_tensor],
                color_ranges: vec![(0, tensor_size, dtype_color)],
            }),
            name_override: Some(tensor.label()),
            byte_range: Some((tensor.file_start, tensor.file_end)),
        });
    }

    Ok(file_size)
}

/// Load a source's bytes for random access: mmaps file sources, clones buffered sources.
/// For diff sources, mmaps both files and returns `abs(modified - original)` as an owned vec.
/// When `byte_range` is set on a File source, only those bytes are returned.
pub fn load_source_data(s: &Source) -> anyhow::Result<Data> {
    match &s.kind {
        SourceKind::File(p) => {
            if let Some((start, end)) = s.byte_range {
                let mut f = File::open(p)?;
                f.seek(SeekFrom::Start(start))?;
                let mut buf = vec![0u8; (end - start) as usize];
                f.read_exact(&mut buf)?;
                Ok(Data::Owned(buf))
            } else {
                let f = File::open(p)?;
                Ok(Data::Mapped(unsafe { Mmap::map(&f) }?))
            }
        }
        SourceKind::Buffered(v) => Ok(Data::Owned(v.clone())),
        SourceKind::Diff { original, modified } => {
            let f_o = File::open(original)?;
            let f_m = File::open(modified)?;
            let m_o = unsafe { Mmap::map(&f_o) }?;
            let m_m = unsafe { Mmap::map(&f_m) }?;
            let diff: Vec<u8> = m_o
                .iter()
                .zip(m_m.iter())
                .map(|(&a, &b)| a.abs_diff(b))
                .collect();
            Ok(Data::Owned(diff))
        }
    }
}

/// Read just the header of a safetensors file and return parsed metadata.
fn load_safetensors_info(path: &Path, file_size: u64) -> anyhow::Result<SafetensorsInfo> {
    // Read the first 8 bytes to get header_size, then read header_size more bytes.
    let mut f = File::open(path)?;
    let mut size_buf = [0u8; 8];
    f.read_exact(&mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf);
    if header_size > 100 * 1024 * 1024 {
        anyhow::bail!("header_size={} exceeds 100 MB safety limit", header_size);
    }
    let total_header = 8 + header_size as usize;
    let mut header_buf = vec![0u8; total_header];
    header_buf[..8].copy_from_slice(&size_buf);
    f.read_exact(&mut header_buf[8..])?;

    let (tensors, header_end) = safetensors::parse_header(&header_buf)?;
    let color_ranges = safetensors::build_color_ranges(&tensors, header_end, file_size);
    Ok(SafetensorsInfo { tensors, color_ranges })
}

/// Byte-value frequency histogram for a source.
///
/// Enables O(1)-memory sorted rendering: rather than sorting bytes and storing
/// the result, the renderer derives the sorted layout from the 256-entry count
/// array alone (see `prefix_sums`).
pub struct Histogram(pub [u64; 256]);

impl Histogram {
    /// Build a histogram by streaming through the source.
    ///
    /// For file sources the file is read in 4 MB chunks so peak extra memory
    /// is bounded regardless of file size. An optional progress bar is
    /// incremented by the number of bytes processed in each chunk.
    pub fn build(s: &Source, pb: Option<&ProgressBar>) -> anyhow::Result<Self> {
        const CHUNK: usize = 4 * 1024 * 1024;
        let mut counts = [0u64; 256];

        match &s.kind {
            SourceKind::File(p) => {
                let mut f = File::open(p)?;
                if let Some((start, _)) = s.byte_range {
                    f.seek(SeekFrom::Start(start))?;
                }
                let mut remaining = s.byte_size;
                let mut buf = vec![0u8; CHUNK];
                while remaining > 0 {
                    let to_read = (remaining as usize).min(CHUNK);
                    let n = f.read(&mut buf[..to_read])?;
                    if n == 0 {
                        break;
                    }
                    for &b in &buf[..n] {
                        counts[b as usize] += 1;
                    }
                    remaining -= n as u64;
                    if let Some(pb) = pb {
                        pb.inc(n as u64);
                    }
                }
            }
            SourceKind::Buffered(v) => {
                for chunk in v.chunks(CHUNK) {
                    for &b in chunk {
                        counts[b as usize] += 1;
                    }
                    if let Some(pb) = pb {
                        pb.inc(chunk.len() as u64);
                    }
                }
            }
            SourceKind::Diff { original, modified } => {
                let mut f_o = File::open(original)?;
                let mut f_m = File::open(modified)?;
                let mut buf_o = vec![0u8; CHUNK];
                let mut buf_m = vec![0u8; CHUNK];
                loop {
                    let n = f_o.read(&mut buf_o)?;
                    if n == 0 {
                        break;
                    }
                    f_m.read_exact(&mut buf_m[..n])?;
                    for i in 0..n {
                        counts[buf_o[i].abs_diff(buf_m[i]) as usize] += 1;
                    }
                    if let Some(pb) = pb {
                        pb.inc(n as u64);
                    }
                }
            }
        }

        Ok(Histogram(counts))
    }

    /// Prefix sums: `prefix[v]` = number of bytes with value strictly less than `v`.
    /// `prefix[256]` = total byte count.
    pub fn prefix_sums(&self) -> [u64; 257] {
        let mut prefix = [0u64; 257];
        for i in 0..256 {
            prefix[i + 1] = prefix[i] + self.0[i];
        }
        prefix
    }
}

/// Recursively collect all files under `root`, sorted by path.
fn collect_files_recursive(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_recursive(root, &mut files);
    files.sort();
    files
}

fn collect_recursive(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("warning: {}: {} — skipping", dir.display(), e);
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            files.push(path);
        } else if path.is_dir() {
            collect_recursive(&path, files);
        }
    }
}

/// Build diff sources from two files or two directories.
///
/// For files: both must be the same size (error if not), unless both are safetensors
/// (in which case a tensor-aligned diff buffer is computed).
/// For directories: files are matched by relative path; pairs with mismatched sizes
/// or no counterpart on the other side are skipped with a warning.
pub fn prepare_diff_sources(
    original: &Path,
    modified: &Path,
    format_safetensors: bool,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let orig_is_file = original.is_file();
    let mod_is_file = modified.is_file();
    let orig_is_dir = original.is_dir();
    let mod_is_dir = modified.is_dir();

    let is_st = |p: &Path| -> bool {
        format_safetensors || p.extension().and_then(|e| e.to_str()) == Some("safetensors")
    };

    if orig_is_file && mod_is_file {
        // Safetensors diff: expand into per-tensor diff Sources (one per matched pair).
        if is_st(original) && is_st(modified) {
            return build_safetensors_diff_sources(original, modified);
        }

        let size_o = std::fs::metadata(original)?.len();
        let size_m = std::fs::metadata(modified)?.len();
        if size_o != size_m {
            anyhow::bail!(
                "--diff: file sizes differ ({} bytes vs {} bytes): {} vs {}",
                size_o,
                size_m,
                original.display(),
                modified.display()
            );
        }
        let source = Source {
            file_idx: 0,
            kind: SourceKind::Diff {
                original: original.to_path_buf(),
                modified: modified.to_path_buf(),
            },
            byte_size: size_o,
            safetensors: None,
            name_override: None,
            byte_range: None,
        };
        return Ok((vec![source], size_o));
    }

    if orig_is_dir && mod_is_dir {
        let orig_files = collect_files_recursive(original);
        let mod_files = collect_files_recursive(modified);

        // Build relative-path → absolute-path maps for each side.
        let orig_map: HashMap<PathBuf, PathBuf> = orig_files
            .iter()
            .filter_map(|p| {
                p.strip_prefix(original)
                    .ok()
                    .map(|rel| (rel.to_path_buf(), p.clone()))
            })
            .collect();
        let mod_map: HashMap<PathBuf, PathBuf> = mod_files
            .iter()
            .filter_map(|p| {
                p.strip_prefix(modified)
                    .ok()
                    .map(|rel| (rel.to_path_buf(), p.clone()))
            })
            .collect();

        // Warn about files only in modified.
        for rel in mod_map.keys() {
            if !orig_map.contains_key(rel) {
                eprintln!(
                    "warning: {} has no counterpart in original — skipping",
                    modified.join(rel).display()
                );
            }
        }

        let mut sources = Vec::new();
        let mut total = 0u64;
        let mut sorted_keys: Vec<&PathBuf> = orig_map.keys().collect();
        sorted_keys.sort();

        for rel in sorted_keys {
            let orig_abs = &orig_map[rel];
            match mod_map.get(rel) {
                None => {
                    eprintln!(
                        "warning: {} has no counterpart in modified — skipping",
                        orig_abs.display()
                    );
                }
                Some(mod_abs) => {
                    let size_o = match std::fs::metadata(orig_abs) {
                        Ok(m) => m.len(),
                        Err(e) => {
                            eprintln!("warning: {}: {} — skipping", orig_abs.display(), e);
                            continue;
                        }
                    };
                    let size_m = match std::fs::metadata(mod_abs) {
                        Ok(m) => m.len(),
                        Err(e) => {
                            eprintln!("warning: {}: {} — skipping", mod_abs.display(), e);
                            continue;
                        }
                    };
                    if size_o != size_m {
                        eprintln!(
                            "warning: size mismatch ({} vs {} bytes) for {} — skipping",
                            size_o,
                            size_m,
                            rel.display()
                        );
                        continue;
                    }
                    sources.push(Source {
                        file_idx: sources.len(),
                        kind: SourceKind::Diff {
                            original: orig_abs.clone(),
                            modified: mod_abs.clone(),
                        },
                        byte_size: size_o,
                        safetensors: None,
                        name_override: None,
                        byte_range: None,
                    });
                    total += size_o;
                }
            }
        }

        if sources.is_empty() {
            anyhow::bail!("--diff: no matching file pairs found between the two directories");
        }
        return Ok((sources, total));
    }

    anyhow::bail!(
        "--diff: both arguments must be files or both must be directories (got {} and {})",
        if orig_is_file { "file" } else if orig_is_dir { "directory" } else { "missing path" },
        if mod_is_file { "file" } else if mod_is_dir { "directory" } else { "missing path" }
    );
}

/// Build per-tensor diff Sources from two .safetensors files.
///
/// Each matched tensor pair (same name, same dtype+shape) becomes one Buffered Source
/// containing the normalized log-scale float32 diff as u8 values. This gives each
/// tensor its own contiguous Hilbert region and its own label in the viewer.
fn build_safetensors_diff_sources(
    original: &Path,
    modified: &Path,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let f_o = File::open(original)?;
    let f_m = File::open(modified)?;
    let m_o = unsafe { Mmap::map(&f_o) }?;
    let m_m = unsafe { Mmap::map(&f_m) }?;

    let (orig_tensors, _) = safetensors::parse_header(&m_o)?;
    let (mod_tensors, _) = safetensors::parse_header(&m_m)?;

    let mod_map: HashMap<&str, &safetensors::TensorMeta> =
        mod_tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    // Warn about tensors only in modified.
    for t in &mod_tensors {
        if !orig_tensors.iter().any(|o| o.name == t.name) {
            eprintln!("warning: safetensors diff: tensor '{}' only in modified — skipping", t.name);
        }
    }

    // Pass 1: compute all per-element diffs and find the global max for normalization.
    struct TensorWork<'a> {
        orig: &'a safetensors::TensorMeta,
        diffs: Vec<f32>,
    }
    let mut work: Vec<TensorWork> = Vec::new();
    let mut global_max: f32 = 0.0;

    for orig_t in &orig_tensors {
        let mod_t = match mod_map.get(orig_t.name.as_str()) {
            Some(t) => t,
            None => {
                eprintln!("warning: safetensors diff: tensor '{}' only in original — skipping", orig_t.name);
                continue;
            }
        };
        if orig_t.dtype != mod_t.dtype || orig_t.shape != mod_t.shape {
            eprintln!("warning: safetensors diff: tensor '{}' dtype/shape mismatch — skipping", orig_t.name);
            continue;
        }
        let orig_bytes = &m_o[orig_t.file_start as usize..orig_t.file_end as usize];
        let mod_bytes = &m_m[mod_t.file_start as usize..mod_t.file_end as usize];
        let orig_vals = orig_t.dtype.decode_to_f32(orig_bytes);
        let mod_vals = mod_t.dtype.decode_to_f32(mod_bytes);
        let diffs: Vec<f32> = orig_vals.iter().zip(mod_vals.iter())
            .map(|(&a, &b)| (a - b).abs()).collect();
        for &d in &diffs {
            if d.is_finite() && d > global_max { global_max = d; }
        }
        work.push(TensorWork { orig: orig_t, diffs });
    }

    if work.is_empty() {
        anyhow::bail!("--diff: no matching tensor pairs found between the two safetensors files");
    }

    // Pass 2: normalize with log scale and build per-tensor Buffered Sources.
    let log_max = if global_max > 0.0 { (global_max as f64 + 1.0).ln() } else { 1.0 };
    let mut sources: Vec<Source> = Vec::new();
    let mut total = 0u64;

    for tw in work {
        let t = tw.orig;
        let es = t.dtype.element_size();
        let buf_size = tw.diffs.len() * es;
        let mut buf = vec![0u8; buf_size];
        for (i, &diff) in tw.diffs.iter().enumerate() {
            let byte_val = if diff.is_finite() {
                let normalized = (diff as f64 + 1.0).ln() / log_max;
                (normalized * 255.0).round().clamp(0.0, 255.0) as u8
            } else {
                255
            };
            let start = i * es;
            let end = (start + es).min(buf_size);
            buf[start..end].fill(byte_val);
        }
        let byte_size = buf_size as u64;
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::Buffered(buf),
            byte_size,
            safetensors: None,
            name_override: Some(t.label()),
            byte_range: None,
        });
        total += byte_size;
    }

    Ok((sources, total))
}
