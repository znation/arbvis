use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
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
    /// Bytes of invisible zero-padding prepended to this source's buffer to align
    /// its data start to a Hilbert quadrant boundary.  Zero for non-diff sources.
    /// The actual tensor data begins at `cumulative_offset + leading_gap`.
    pub leading_gap: u64,
    /// Offset within this source's buffer where actual data ends (exclusive).
    /// Bytes in `[0, leading_gap)` and `[data_end, byte_size)` are alignment padding
    /// and should be colored differently from real data bytes.
    /// Equals `byte_size` for non-padded sources.
    pub data_end: u64,
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
/// parsed and attached as SafetensorsInfo for dtype coloring. The file is kept as
/// a single Source (one per file) so that inter-tensor borders are not drawn and
/// the Hilbert curve flows smoothly across the whole file with color transitions
/// only at tensor boundaries.
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
                leading_gap: 0,
                data_end: len,
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

        let safetensors_info = if is_st {
            match load_safetensors_info(path, size) {
                Ok(info) => Some(info),
                Err(e) => {
                    eprintln!("warning: {}: failed to parse safetensors header: {} — treating as plain binary", path.display(), e);
                    None
                }
            }
        } else {
            None
        };

        total += size;
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::File(path.clone()),
            byte_size: size,
            safetensors: safetensors_info,
            name_override: None,
            leading_gap: 0,
            data_end: size,
        });
    }
    Ok((sources, total))
}

/// Load a source's bytes for random access: mmaps file sources, clones buffered sources.
/// For diff sources, mmaps both files and returns `abs(modified - original)` as an owned vec.
pub fn load_source_data(s: &Source) -> anyhow::Result<Data> {
    match &s.kind {
        SourceKind::File(p) => {
            let f = File::open(p)?;
            Ok(Data::Mapped(unsafe { Mmap::map(&f) }?))
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
            leading_gap: 0,
            data_end: size_o,
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
                    if is_st(orig_abs) && is_st(mod_abs) {
                        match build_safetensors_diff_sources(orig_abs, mod_abs) {
                            Ok((mut tensor_sources, bytes)) => {
                                let base_idx = sources.len();
                                for s in &mut tensor_sources {
                                    s.file_idx += base_idx;
                                }
                                sources.extend(tensor_sources);
                                total += bytes;
                            }
                            Err(e) => eprintln!(
                                "warning: {}: safetensors diff failed: {e} — skipping",
                                rel.display()
                            ),
                        }
                        continue;
                    }
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
                        leading_gap: 0,
                        data_end: size_o,
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

/// Return the smallest power of 4 that is >= n, minimum 1.
///
/// Powers of 4 align with Hilbert curve quadrant boundaries, so padding each
/// tensor's diff buffer to the next power of 4 makes the tensor occupy exactly
/// one complete Hilbert sub-square (a perfect square region in the 2D image).
fn next_power_of_2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p = p.saturating_mul(2);
    }
    p
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

    // Single pass: compute per-element relative diffs and build per-tensor sources.
    //
    // Each element is normalized by its own original magnitude:
    //   relative_diff = |orig - mod| / max(|orig|, ε)
    // so the brightness encodes *how much the weight changed relative to what it was*:
    //   black  → identical (0% change)
    //   dim    → small relative change (e.g. 1–5%)
    //   bright → large relative change (e.g. 50%+)
    //   max    → diff ≥ original magnitude (100%+ change)
    //
    // A sqrt scale maps the [0, 1] relative range to [0, 255] so that small-but-real
    // changes (e.g. 1%) are visible (byte ≈ 16) rather than crushed near zero.
    // Values > 1 are clamped to max brightness.
    const EPSILON: f32 = 1e-6;
    let mut sources: Vec<Source> = Vec::new();
    let mut total = 0u64;
    for orig_t in &orig_tensors {
        let mod_t = match mod_map.get(orig_t.name.as_str()) {
            Some(t) => t,
            None => {
                eprintln!("warning: safetensors diff: tensor '{}' only in original — skipping", orig_t.name);
                continue;
            }
        };
        if orig_t.shape != mod_t.shape {
            eprintln!("warning: safetensors diff: tensor '{}' shape mismatch {:?} vs {:?} — skipping",
                orig_t.name, orig_t.shape, mod_t.shape);
            continue;
        }
        let orig_bytes = &m_o[orig_t.file_start as usize..orig_t.file_end as usize];
        let mod_bytes  = &m_m[mod_t.file_start  as usize..mod_t.file_end  as usize];
        let orig_vals = orig_t.dtype.decode_to_f32(orig_bytes);
        let mod_vals  = mod_t.dtype.decode_to_f32(mod_bytes);

        let t = orig_t;
        let buf: Vec<u8> = orig_vals.iter().zip(mod_vals.iter()).map(|(&o, &m)| {
            if !o.is_finite() || !m.is_finite() { return 255u8; }
            let rel = (o - m).abs() / o.abs().max(EPSILON);
            // sqrt scale: 1% change → byte≈16, 25% → 128, 100%+ → 255
            (rel.sqrt().min(1.0) * 255.0).round() as u8
        }).collect();
        // Pad element count to the next power of 4 so this tensor fits in exactly
        // one Hilbert sub-quadrant (a perfect square region).
        let padded_size = next_power_of_2(buf.len()).max(1) as u64;

        // Align the tensor's START to a padded_size boundary so that
        // decompose_hilbert produces a single rectangle for the data region.
        // If the current cumulative offset isn't already aligned, prepend enough
        // zero bytes to reach the next aligned position.
        let leading_gap = {
            let rem = total % padded_size;
            if rem == 0 { 0u64 } else { padded_size - rem }
        };

        // Build: [leading_gap zeros] [element diffs, zero-padded to padded_size]
        let data_end = leading_gap + buf.len() as u64;
        let mut full_buf = vec![0u8; leading_gap as usize];
        full_buf.extend_from_slice(&buf);
        full_buf.resize((leading_gap + padded_size) as usize, 0u8);

        let byte_size = full_buf.len() as u64;
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::Buffered(full_buf),
            byte_size,
            safetensors: None,
            name_override: Some(t.label()),
            leading_gap,
            data_end,
        });
        total += byte_size;
    }

    Ok((sources, total))
}

#[cfg(test)]
mod tests {
    use super::next_power_of_2;

    #[test]
    fn next_power_of_2_values() {
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(4), 4);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(16), 16);
        assert_eq!(next_power_of_2(17), 32);
        assert_eq!(next_power_of_2(64), 64);
        assert_eq!(next_power_of_2(65), 128);
        assert_eq!(next_power_of_2(1_000_000), 1_048_576); // 2^20
    }
}
