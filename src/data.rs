use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::Rgb;
use indicatif::ProgressBar;
use memmap2::Mmap;

use crate::safetensors::{self, TensorMeta};
use crate::hf_url::RemoteFileSpec;

pub struct SafetensorsInfo {
    pub tensors: Vec<TensorMeta>,
    pub color_ranges: Vec<(u64, u64, Rgb<u8>)>,
}

/// Backing storage for a source's bytes.
pub enum Data {
    Mapped(Mmap),
    Owned(Vec<u8>),
    /// Remote file accessed via HTTP range requests — never loaded locally.
    Http {
        url: Arc<String>,
        token: Option<Arc<String>>,
        agent: Arc<ureq::Agent>,
    },
}

impl std::ops::Deref for Data {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Data::Mapped(m) => m,
            Data::Owned(v) => v,
            Data::Http { .. } => panic!("bug: use fetch_range() for remote HTTP Data, not Deref"),
        }
    }
}

impl Data {
    /// Return bytes `[start, start+len)` from this source.
    /// For local sources: zero-copy borrow. For Http: one HTTP range request.
    pub fn fetch_range(&self, start: u64, len: usize) -> anyhow::Result<Cow<'_, [u8]>> {
        match self {
            Data::Mapped(m) => {
                Ok(Cow::Borrowed(&m[start as usize..start as usize + len]))
            }
            Data::Owned(v) => {
                Ok(Cow::Borrowed(&v[start as usize..start as usize + len]))
            }
            Data::Http { url, token, agent } => {
                use std::io::Read as _;
                let range = format!("bytes={start}-{}", start + len as u64 - 1);
                let mut req = agent.get(url.as_str()).set("Range", &range);
                if let Some(t) = token {
                    req = req.set("Authorization", &format!("Bearer {t}"));
                }
                let resp = req.call()
                    .map_err(|e| anyhow::anyhow!("range request failed: {e}"))?;
                let mut bytes = Vec::with_capacity(len);
                resp.into_reader().read_to_end(&mut bytes)
                    .map_err(|e| anyhow::anyhow!("range response read failed: {e}"))?;
                anyhow::ensure!(bytes.len() == len,
                    "range request returned {} bytes, expected {}", bytes.len(), len);
                Ok(Cow::Owned(bytes))
            }
        }
    }
}

/// How a source's bytes are stored.
pub enum SourceKind {
    Buffered(Vec<u8>),
    File(PathBuf),
    Diff { original: PathBuf, modified: PathBuf },
    /// Remote HF file, accessed via HTTP range requests per tile.
    Http { cdn_url: String },
}

/// Input specification: local file path or resolved remote HF file.
pub enum InputSpec {
    Local(PathBuf),
    Remote(RemoteFileSpec),
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
            SourceKind::Http { cdn_url } => cdn_url
                .rsplit('/')
                .next()
                .unwrap_or(cdn_url)
                .to_string(),
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
            }],
            len,
        ));
    }

    // Expand any directory paths (e.g. from a repo-level hf:// download) into
    // their constituent files so they can be treated as individual sources.
    let expanded: Vec<PathBuf> = files.iter().flat_map(|p| {
        if p.is_dir() { collect_files_recursive(p) } else { vec![p.clone()] }
    }).collect();

    let mut sources = Vec::new();
    let mut total = 0u64;
    for path in &expanded {
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
        });
    }
    Ok((sources, total))
}

/// Load a source's bytes for random access: mmaps file sources, clones buffered sources.
/// For diff sources, mmaps both files and returns the signed diff encoding used by the
/// diff LUT: 127 = no change, 128–254 = byte grew (green), 0–126 = byte shrank (red).
/// For Http sources, returns a `Data::Http` handle that fetches byte ranges on demand.
pub fn load_source_data(
    s: &Source,
    agent: &Arc<ureq::Agent>,
    token: Option<&Arc<String>>,
) -> anyhow::Result<Data> {
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
                .map(|(&a, &b)| {
                    let delta = b as i16 - a as i16; // positive = byte grew
                    let brightness = (delta.unsigned_abs() as f32 / 255.0 * 127.0).round() as u8;
                    if delta >= 0 { 127u8 + brightness } else { 127u8 - brightness }
                })
                .collect();
            Ok(Data::Owned(diff))
        }
        SourceKind::Http { cdn_url } => Ok(Data::Http {
            url: Arc::new(cdn_url.clone()),
            token: token.cloned(),
            agent: agent.clone(),
        }),
    }
}

/// Build sources from a mixed list of local paths and remote HF file specs.
/// Remote specs are turned into `SourceKind::Http` entries (no download).
pub fn prepare_sources_from_specs(
    specs: &[InputSpec],
    format_safetensors: bool,
) -> anyhow::Result<(Vec<Source>, u64)> {
    if specs.is_empty() {
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
            }],
            len,
        ));
    }

    let mut sources = Vec::new();
    let mut total = 0u64;

    for spec in specs {
        match spec {
            InputSpec::Local(path) => {
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
                });
            }
            InputSpec::Remote(spec) => {
                total += spec.size;
                sources.push(Source {
                    file_idx: sources.len(),
                    kind: SourceKind::Http { cdn_url: spec.cdn_url.clone() },
                    byte_size: spec.size,
                    safetensors: None,
                    name_override: None,
                });
            }
        }
    }

    Ok((sources, total))
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
            SourceKind::Http { .. } => {
                anyhow::bail!(
                    "--sort is not supported for remote hf:// inputs: \
                     building a histogram requires streaming the entire file over HTTP"
                );
            }
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
            // Signed relative change: positive = weight grew, negative = weight shrank.
            // Encoded as a u8 centred at 127:
            //   127         → no change (black)
            //   128..=254   → increased (red), brightness ∝ sqrt(rel)
            //   0..=126     → decreased (cyan), brightness ∝ sqrt(rel)
            //   255         → non-finite (reserved above)
            let signed_rel = (m - o) / o.abs().max(EPSILON);
            let brightness = (signed_rel.abs().sqrt().min(1.0) * 127.0).round() as u8;
            if signed_rel >= 0.0 { 127u8 + brightness } else { 127u8 - brightness }
        }).collect();
        let byte_size = buf.len() as u64;
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::Buffered(buf),
            byte_size,
            safetensors: None,
            name_override: Some(t.label()),
        });
        total += byte_size;
    }

    Ok((sources, total))
}

