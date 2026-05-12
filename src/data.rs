use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
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

        let mut sources = Vec::new();
        let mut total = 0u64;

        // Safetensors: match by tensor name across ALL .safetensors files on each side,
        // regardless of filename. This handles sharded vs unsharded and different naming
        // conventions (e.g. model.safetensors-00001-of-00001.safetensors vs model.safetensors).
        let orig_st: Vec<PathBuf> = orig_files.iter().filter(|p| is_st(p)).cloned().collect();
        let mod_st: Vec<PathBuf> = mod_files.iter().filter(|p| is_st(p)).cloned().collect();
        if !orig_st.is_empty() || !mod_st.is_empty() {
            if orig_st.is_empty() {
                eprintln!("warning: original has no .safetensors files — skipping model weight diff");
            } else if mod_st.is_empty() {
                eprintln!("warning: modified has no .safetensors files — skipping model weight diff");
            } else {
                match build_multi_safetensors_diff_sources(&orig_st, &mod_st) {
                    Ok((mut tensor_sources, bytes)) => {
                        let base_idx = sources.len();
                        for s in &mut tensor_sources {
                            s.file_idx += base_idx;
                        }
                        sources.extend(tensor_sources);
                        total += bytes;
                    }
                    Err(e) => eprintln!("warning: safetensors diff failed: {e} — skipping"),
                }
            }
        }

        // Non-safetensors: match by relative path, require equal sizes.
        let orig_map: HashMap<PathBuf, PathBuf> = orig_files
            .iter()
            .filter(|p| !is_st(p))
            .filter_map(|p| {
                p.strip_prefix(original)
                    .ok()
                    .map(|rel| (rel.to_path_buf(), p.clone()))
            })
            .collect();
        let mod_map: HashMap<PathBuf, PathBuf> = mod_files
            .iter()
            .filter(|p| !is_st(p))
            .filter_map(|p| {
                p.strip_prefix(modified)
                    .ok()
                    .map(|rel| (rel.to_path_buf(), p.clone()))
            })
            .collect();

        for rel in mod_map.keys() {
            if !orig_map.contains_key(rel) {
                eprintln!(
                    "warning: {} has no counterpart in original — skipping",
                    modified.join(rel).display()
                );
            }
        }

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

/// Strip the first `n` dot-delimited path components from `name`.
/// Returns the remainder after the n-th dot, or `None` if `name` has fewer than n+1 components.
fn strip_prefix_components(name: &str, n: usize) -> Option<&str> {
    let mut idx = 0;
    for _ in 0..n {
        idx += name[idx..].find('.')? + 1;
    }
    Some(&name[idx..])
}

/// Find (strip_orig, strip_mod) prefix depths that maximise unique 1-to-1 tensor name matches.
/// Returns (0, 0) if exact matching already produces matches.
fn find_strip_depths(
    orig_names: &[String],
    mod_names: &[String],
) -> (usize, usize) {
    // Count unique-suffix occurrences for a set of names stripped by `n` components.
    let unique_suffixes = |names: &[String], n: usize| -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for name in names {
            if let Some(s) = strip_prefix_components(name, n) {
                if !s.is_empty() {
                    *counts.entry(s.to_owned()).or_insert(0) += 1;
                }
            }
        }
        counts
    };

    let mut best = (0usize, 0usize, 0usize); // (strip_orig, strip_mod, match_count)
    for so in 0usize..=5 {
        let orig_counts = unique_suffixes(orig_names, so);
        for sm in 0usize..=5 {
            if so == 0 && sm == 0 { continue; }
            let mod_counts = unique_suffixes(mod_names, sm);
            let matches = orig_counts.iter()
                .filter(|(s, &oc)| oc == 1 && mod_counts.get(s.as_str()) == Some(&1))
                .count();
            if matches > best.2 {
                best = (so, sm, matches);
            }
        }
    }
    (best.0, best.1)
}

/// Build per-tensor diff Sources from multiple .safetensors files on each side.
///
/// Tensors are matched by name across ALL files on each side, so sharded models
/// (model.safetensors-00001-of-00002.safetensors, …) and single-file models
/// (model.safetensors) are compared correctly even when filenames differ.
///
/// When no tensors match by exact name (e.g. because a fine-tuned model wraps
/// the backbone under a deeper module path), the function automatically searches
/// for a prefix-strip depth that maximises unique 1-to-1 tensor matches.
fn build_multi_safetensors_diff_sources(
    orig_files: &[PathBuf],
    mod_files: &[PathBuf],
) -> anyhow::Result<(Vec<Source>, u64)> {
    let open_mmaps = |files: &[PathBuf]| -> anyhow::Result<Vec<Mmap>> {
        files.iter().map(|p| {
            let f = File::open(p).with_context(|| format!("opening {}", p.display()))?;
            Ok(unsafe { Mmap::map(&f) }?)
        }).collect()
    };
    let orig_mmaps = open_mmaps(orig_files)?;
    let mod_mmaps  = open_mmaps(mod_files)?;

    // Build tensor maps: full name → (mmap_index, TensorMeta).
    let mut orig_map: HashMap<String, (usize, safetensors::TensorMeta)> = HashMap::new();
    for (i, mmap) in orig_mmaps.iter().enumerate() {
        let (tensors, _) = safetensors::parse_header(mmap)?;
        for t in tensors {
            orig_map.entry(t.name.clone()).or_insert((i, t));
        }
    }
    let mut mod_map: HashMap<String, (usize, safetensors::TensorMeta)> = HashMap::new();
    for (i, mmap) in mod_mmaps.iter().enumerate() {
        let (tensors, _) = safetensors::parse_header(mmap)?;
        for t in tensors {
            mod_map.entry(t.name.clone()).or_insert((i, t));
        }
    }

    let orig_names: Vec<String> = orig_map.keys().cloned().collect();
    let mod_names: Vec<String> = mod_map.keys().cloned().collect();

    // Determine how many leading name components to strip on each side.
    // (0, 0) means exact names match; non-zero means the backbone was re-nested
    // (e.g. model.layers.* in original vs model.lm.model.layers.* in modified).
    let exact_overlap = orig_names.iter().any(|n| mod_map.contains_key(n.as_str()));
    let (strip_o, strip_m) = if exact_overlap {
        (0, 0)
    } else {
        let depths = find_strip_depths(&orig_names, &mod_names);
        if depths != (0, 0) {
            log::info!(
                "safetensors diff: no exact tensor name overlap; \
                 stripping {} prefix component(s) from original and {} from modified",
                depths.0, depths.1
            );
        }
        depths
    };

    // Build lookup map keyed by stripped name → full orig name.
    // (When strip_o == 0 this is just the identity.)
    let orig_by_stripped: HashMap<String, &str> = orig_map.keys()
        .filter_map(|n| {
            strip_prefix_components(n, strip_o)
                .filter(|s| !s.is_empty())
                .map(|s| (s.to_owned(), n.as_str()))
        })
        .fold(HashMap::new(), |mut acc, (stripped, full)| {
            // Keep only unique stripped suffixes to avoid false matches.
            acc.entry(stripped).and_modify(|v| *v = "").or_insert(full);
            acc
        });

    let mod_by_stripped: HashMap<String, &str> = mod_map.keys()
        .filter_map(|n| {
            strip_prefix_components(n, strip_m)
                .filter(|s| !s.is_empty())
                .map(|s| (s.to_owned(), n.as_str()))
        })
        .fold(HashMap::new(), |mut acc, (stripped, full)| {
            acc.entry(stripped).and_modify(|v| *v = "").or_insert(full);
            acc
        });

    // Warn about tensors only in modified (by stripped suffix).
    for stripped in mod_by_stripped.keys() {
        if !orig_by_stripped.contains_key(stripped.as_str()) {
            // Only warn when mod full name isn't in orig_by_stripped — too noisy for large mismatches.
            // Skip per-tensor warnings when strip depths differ; a summary suffices.
            if strip_o == 0 && strip_m == 0 {
                if let Some(&full) = mod_by_stripped.get(stripped.as_str()) {
                    if !full.is_empty() {
                        eprintln!("warning: safetensors diff: tensor '{full}' only in modified — skipping");
                    }
                }
            }
        }
    }

    const EPSILON: f32 = 1e-6;
    let mut sources: Vec<Source> = Vec::new();
    let mut total = 0u64;

    let mut sorted_orig: Vec<&str> = orig_by_stripped.values().copied().filter(|s| !s.is_empty()).collect();
    sorted_orig.sort();

    for orig_full in sorted_orig {
        let stripped = match strip_prefix_components(orig_full, strip_o) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        let mod_full = match mod_by_stripped.get(stripped) {
            Some(s) if !s.is_empty() => *s,
            _ => {
                if strip_o == 0 && strip_m == 0 {
                    eprintln!("warning: safetensors diff: tensor '{orig_full}' only in original — skipping");
                }
                continue;
            }
        };

        let (oi, orig_t) = &orig_map[orig_full];
        let (mi, mod_t) = &mod_map[mod_full];

        if orig_t.shape != mod_t.shape {
            eprintln!(
                "warning: safetensors diff: tensor '{orig_full}' shape mismatch {:?} vs {:?} — skipping",
                orig_t.shape, mod_t.shape
            );
            continue;
        }
        let orig_bytes = &orig_mmaps[*oi][orig_t.file_start as usize..orig_t.file_end as usize];
        let mod_bytes  = &mod_mmaps[*mi][mod_t.file_start  as usize..mod_t.file_end  as usize];
        let orig_vals = orig_t.dtype.decode_to_f32(orig_bytes);
        let mod_vals  = mod_t.dtype.decode_to_f32(mod_bytes);

        let buf: Vec<u8> = orig_vals.iter().zip(mod_vals.iter()).map(|(&o, &m)| {
            if !o.is_finite() || !m.is_finite() { return 255u8; }
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
            name_override: Some(orig_t.label()),
        });
        total += byte_size;
    }

    Ok((sources, total))
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

