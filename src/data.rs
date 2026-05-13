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
    /// Diff computed on demand per range — never stored in full.
    LazyDiff(Arc<dyn Fn(u64, usize) -> anyhow::Result<Vec<u8>> + Send + Sync>),
}

impl std::ops::Deref for Data {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Data::Mapped(m) => m,
            Data::Owned(v) => v,
            Data::Http { .. } => panic!("bug: use fetch_range() for remote HTTP Data, not Deref"),
            Data::LazyDiff(_) => panic!("bug: use fetch_range() for LazyDiff Data, not Deref"),
        }
    }
}

impl Data {
    /// Return bytes `[start, start+len)` from this source.
    /// For local sources: zero-copy borrow. For Http/LazyDiff: computed on demand.
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
            Data::LazyDiff(f) => Ok(Cow::Owned(f(start, len)?)),
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
    /// Per-tensor diff computed lazily from two whole-file Data sources.
    /// `byte_size` (= nelem) output bytes are produced on demand.
    TensorDiff {
        orig: Arc<Data>,
        mod_: Arc<Data>,
        orig_start: u64,
        mod_start: u64,
        orig_dtype: safetensors::Dtype,
        mod_dtype: safetensors::Dtype,
        nelem: u64,
    },
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
            SourceKind::TensorDiff { .. } => {
                unreachable!("TensorDiff sources always have name_override set")
            }
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
/// For diff sources, returns a LazyDiff that computes bytes on demand per tile.
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
            let m_o = Arc::new(unsafe { Mmap::map(&f_o) }?);
            let m_m = Arc::new(unsafe { Mmap::map(&f_m) }?);
            Ok(Data::LazyDiff(Arc::new(move |start: u64, len: usize| {
                let a = &m_o[start as usize..start as usize + len];
                let b = &m_m[start as usize..start as usize + len];
                Ok(a.iter().zip(b.iter()).map(|(&a, &b)| {
                    let delta = b as i16 - a as i16;
                    let brightness = (delta.unsigned_abs() as f32 / 255.0 * 127.0).round() as u8;
                    if delta >= 0 { 127u8 + brightness } else { 127u8 - brightness }
                }).collect())
            })))
        }
        SourceKind::Http { cdn_url } => Ok(Data::Http {
            url: Arc::new(cdn_url.clone()),
            token: token.cloned(),
            agent: agent.clone(),
        }),
        SourceKind::TensorDiff { orig, mod_, orig_start, mod_start, orig_dtype, mod_dtype, .. } => {
            let orig = Arc::clone(orig);
            let mod_ = Arc::clone(mod_);
            let orig_start = *orig_start;
            let mod_start = *mod_start;
            let orig_dtype = *orig_dtype;
            let mod_dtype = *mod_dtype;
            const EPSILON: f32 = 1e-6;
            Ok(Data::LazyDiff(Arc::new(move |start: u64, len: usize| {
                let orig_elem = orig_dtype.element_size() as u64;
                let mod_elem = mod_dtype.element_size() as u64;
                let ob = orig.fetch_range(orig_start + start * orig_elem, (len as u64 * orig_elem) as usize)?;
                let mb = mod_.fetch_range(mod_start + start * mod_elem, (len as u64 * mod_elem) as usize)?;
                Ok(orig_dtype.diff_to_u8(&ob, mod_dtype, &mb, EPSILON))
            })))
        }
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

/// Fetch and parse the safetensors header from any Data source.
/// For local sources (Mapped/Owned): zero-copy slice access.
/// For remote sources (Http): two range requests (8 bytes, then full header).
fn fetch_safetensors_header(data: &Data) -> anyhow::Result<(Vec<safetensors::TensorMeta>, u64)> {
    let size_bytes = data.fetch_range(0, 8)?;
    let header_size = u64::from_le_bytes(size_bytes.as_ref()[..8].try_into().unwrap());
    if header_size > 100 * 1024 * 1024 {
        anyhow::bail!("safetensors header_size={} exceeds 100 MB safety limit", header_size);
    }
    let total_header = 8 + header_size as usize;
    let header_bytes = data.fetch_range(0, total_header)?;
    safetensors::parse_header(header_bytes.as_ref())
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
            SourceKind::TensorDiff { orig, mod_, orig_start, mod_start, orig_dtype, mod_dtype, nelem } => {
                const CHUNK_ELEMS: u64 = 512 * 1024;
                const EPSILON: f32 = 1e-6;
                let orig_elem = orig_dtype.element_size() as u64;
                let mod_elem = mod_dtype.element_size() as u64;
                let mut elem = 0u64;
                while elem < *nelem {
                    let batch = CHUNK_ELEMS.min(*nelem - elem);
                    let ob = orig.fetch_range(
                        orig_start + elem * orig_elem,
                        (batch * orig_elem) as usize,
                    )?;
                    let mb = mod_.fetch_range(
                        mod_start + elem * mod_elem,
                        (batch * mod_elem) as usize,
                    )?;
                    for b in orig_dtype.diff_to_u8(&ob, *mod_dtype, &mb, EPSILON) {
                        counts[b as usize] += 1;
                    }
                    if let Some(pb) = pb {
                        pb.inc(batch);
                    }
                    elem += batch;
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

/// Build diff sources from two repos listed as HTTP specs (no download).
///
/// Safetensors files are diffed lazily via HTTP range requests — no model weights
/// are downloaded to disk or held in RAM. Small non-safetensors files (≤16 MB)
/// are downloaded eagerly and binary-diffed; larger ones are skipped with a warning.
pub fn prepare_diff_sources_from_http(
    orig_specs: &[(String, RemoteFileSpec)],
    mod_specs: &[(String, RemoteFileSpec)],
    agent: Arc<ureq::Agent>,
    token: Option<Arc<String>>,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let is_st = |name: &str| name.ends_with(".safetensors");

    let orig_st: Vec<&(String, RemoteFileSpec)> = orig_specs.iter().filter(|(n, _)| is_st(n)).collect();
    let mod_st: Vec<&(String, RemoteFileSpec)>  = mod_specs.iter().filter(|(n, _)| is_st(n)).collect();

    let mut sources: Vec<Source> = Vec::new();
    let mut total = 0u64;

    // Safetensors diff — fully lazy, no download.
    if !orig_st.is_empty() || !mod_st.is_empty() {
        if orig_st.is_empty() {
            eprintln!("warning: original has no .safetensors files — skipping model weight diff");
        } else if mod_st.is_empty() {
            eprintln!("warning: modified has no .safetensors files — skipping model weight diff");
        } else {
            match build_multi_safetensors_diff_sources_from_http(&orig_st, &mod_st, &agent, token.as_ref()) {
                Ok((mut tensor_sources, bytes)) => {
                    let base_idx = sources.len();
                    for s in &mut tensor_sources { s.file_idx += base_idx; }
                    sources.extend(tensor_sources);
                    total += bytes;
                }
                Err(e) => eprintln!("warning: safetensors diff failed: {e} — skipping"),
            }
        }
    }

    // Non-safetensors files: match by filename, download if small.
    const MAX_EAGER_SIZE: u64 = 16 * 1024 * 1024;
    let orig_non: HashMap<&str, &RemoteFileSpec> =
        orig_specs.iter().filter(|(n, _)| !is_st(n)).map(|(n, s)| (n.as_str(), s)).collect();
    let mod_non: HashMap<&str, &RemoteFileSpec> =
        mod_specs.iter().filter(|(n, _)| !is_st(n)).map(|(n, s)| (n.as_str(), s)).collect();

    for (fname, _) in &mod_non {
        if !orig_non.contains_key(fname) {
            eprintln!("warning: {fname} only in modified — skipping");
        }
    }

    let mut sorted: Vec<&str> = orig_non.keys().copied().collect();
    sorted.sort();

    for fname in sorted {
        let orig_spec = &orig_non[fname];
        let mod_spec = match mod_non.get(fname) {
            Some(s) => s,
            None => { eprintln!("warning: {fname} only in original — skipping"); continue; }
        };
        if orig_spec.size != mod_spec.size {
            eprintln!("warning: size mismatch for {fname} ({} vs {} bytes) — skipping",
                orig_spec.size, mod_spec.size);
            continue;
        }
        if orig_spec.size > MAX_EAGER_SIZE {
            eprintln!("warning: {fname} exceeds {} MB — skipping non-safetensors large file",
                MAX_EAGER_SIZE / 1024 / 1024);
            continue;
        }
        let orig_data = Data::Http {
            url: Arc::new(orig_spec.cdn_url.clone()),
            token: token.clone(),
            agent: agent.clone(),
        };
        let mod_data = Data::Http {
            url: Arc::new(mod_spec.cdn_url.clone()),
            token: token.clone(),
            agent: agent.clone(),
        };
        let ob = orig_data.fetch_range(0, orig_spec.size as usize)?;
        let mb = mod_data.fetch_range(0, mod_spec.size as usize)?;
        let diff: Vec<u8> = ob.iter().zip(mb.iter()).map(|(&a, &b)| {
            let delta = b as i16 - a as i16;
            let brightness = (delta.unsigned_abs() as f32 / 255.0 * 127.0).round() as u8;
            if delta >= 0 { 127u8 + brightness } else { 127u8 - brightness }
        }).collect();
        let size = diff.len() as u64;
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::Buffered(diff),
            byte_size: size,
            safetensors: None,
            name_override: Some(fname.to_string()),
        });
        total += size;
    }

    if sources.is_empty() {
        anyhow::bail!("--diff: no matching file pairs found between the two repos");
    }
    Ok((sources, total))
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

/// Find 1-to-1 matched tensor name pairs between two name sets.
/// Applies the prefix-strip heuristic when no exact name overlap exists.
/// Emits warnings for unmatched tensors when strip depths are both zero.
/// Returns (orig_full_name, mod_full_name) pairs sorted by orig name.
fn find_matched_tensor_pairs(orig_names: &[String], mod_names: &[String]) -> Vec<(String, String)> {
    let mod_set: std::collections::HashSet<&str> = mod_names.iter().map(|s| s.as_str()).collect();
    let exact_overlap = orig_names.iter().any(|n| mod_set.contains(n.as_str()));
    let (strip_o, strip_m) = if exact_overlap {
        (0, 0)
    } else {
        let depths = find_strip_depths(orig_names, mod_names);
        if depths != (0, 0) {
            log::info!(
                "safetensors diff: no exact tensor name overlap; \
                 stripping {} prefix component(s) from original and {} from modified",
                depths.0, depths.1
            );
        }
        depths
    };

    let orig_by_stripped: HashMap<String, &str> = orig_names.iter()
        .filter_map(|n| {
            strip_prefix_components(n, strip_o)
                .filter(|s| !s.is_empty())
                .map(|s| (s.to_owned(), n.as_str()))
        })
        .fold(HashMap::new(), |mut acc, (stripped, full)| {
            acc.entry(stripped).and_modify(|v| *v = "").or_insert(full);
            acc
        });

    let mod_by_stripped: HashMap<String, &str> = mod_names.iter()
        .filter_map(|n| {
            strip_prefix_components(n, strip_m)
                .filter(|s| !s.is_empty())
                .map(|s| (s.to_owned(), n.as_str()))
        })
        .fold(HashMap::new(), |mut acc, (stripped, full)| {
            acc.entry(stripped).and_modify(|v| *v = "").or_insert(full);
            acc
        });

    if strip_o == 0 && strip_m == 0 {
        for (stripped, &full) in &mod_by_stripped {
            if !orig_by_stripped.contains_key(stripped) && !full.is_empty() {
                eprintln!("warning: safetensors diff: tensor '{full}' only in modified — skipping");
            }
        }
    }

    let mut sorted_orig: Vec<&str> = orig_by_stripped.values().copied().filter(|s| !s.is_empty()).collect();
    sorted_orig.sort();

    let mut pairs = Vec::new();
    for orig_full in sorted_orig {
        let stripped = match strip_prefix_components(orig_full, strip_o) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        match mod_by_stripped.get(stripped) {
            Some(s) if !s.is_empty() => {
                pairs.push((orig_full.to_owned(), (*s).to_owned()));
            }
            _ => {
                if strip_o == 0 && strip_m == 0 {
                    eprintln!("warning: safetensors diff: tensor '{orig_full}' only in original — skipping");
                }
            }
        }
    }
    pairs
}

/// Core tensor diff builder: given two parallel lists of whole-file Data sources,
/// build per-tensor TensorDiff Source entries without reading any tensor bytes.
fn build_multi_safetensors_diff_sources_inner(
    orig_data: &[Arc<Data>],
    mod_data:  &[Arc<Data>],
) -> anyhow::Result<(Vec<Source>, u64)> {
    // Build tensor maps: full name → (data_index, TensorMeta).
    let mut orig_map: HashMap<String, (usize, safetensors::TensorMeta)> = HashMap::new();
    for (i, data) in orig_data.iter().enumerate() {
        let (tensors, _) = fetch_safetensors_header(data)
            .with_context(|| format!("reading safetensors header for orig file {i}"))?;
        for t in tensors {
            orig_map.entry(t.name.clone()).or_insert((i, t));
        }
    }
    let mut mod_map: HashMap<String, (usize, safetensors::TensorMeta)> = HashMap::new();
    for (i, data) in mod_data.iter().enumerate() {
        let (tensors, _) = fetch_safetensors_header(data)
            .with_context(|| format!("reading safetensors header for mod file {i}"))?;
        for t in tensors {
            mod_map.entry(t.name.clone()).or_insert((i, t));
        }
    }

    let orig_names: Vec<String> = orig_map.keys().cloned().collect();
    let mod_names:  Vec<String> = mod_map.keys().cloned().collect();
    let pairs = find_matched_tensor_pairs(&orig_names, &mod_names);

    let mut sources: Vec<Source> = Vec::new();
    let mut total = 0u64;

    for (orig_full, mod_full) in pairs {
        let (oi, orig_t) = &orig_map[&orig_full];
        let (mi, mod_t)  = &mod_map[&mod_full];

        if orig_t.shape != mod_t.shape {
            eprintln!(
                "warning: safetensors diff: tensor '{}' shape mismatch {:?} vs {:?} — skipping",
                orig_full, orig_t.shape, mod_t.shape
            );
            continue;
        }

        let nelem: u64 = orig_t.shape.iter().product();
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::TensorDiff {
                orig: Arc::clone(&orig_data[*oi]),
                mod_: Arc::clone(&mod_data[*mi]),
                orig_start: orig_t.file_start,
                mod_start:  mod_t.file_start,
                orig_dtype: orig_t.dtype,
                mod_dtype:  mod_t.dtype,
                nelem,
            },
            byte_size: nelem,
            safetensors: None,
            name_override: Some(orig_t.label()),
        });
        total += nelem;
    }

    Ok((sources, total))
}

/// Build per-tensor diff Sources from multiple local .safetensors files on each side.
fn build_multi_safetensors_diff_sources(
    orig_files: &[PathBuf],
    mod_files: &[PathBuf],
) -> anyhow::Result<(Vec<Source>, u64)> {
    let open_arcs = |files: &[PathBuf]| -> anyhow::Result<Vec<Arc<Data>>> {
        files.iter().map(|p| {
            let f = File::open(p).with_context(|| format!("opening {}", p.display()))?;
            Ok(Arc::new(Data::Mapped(unsafe { Mmap::map(&f) }?)))
        }).collect()
    };
    let orig_data = open_arcs(orig_files)?;
    let mod_data  = open_arcs(mod_files)?;
    build_multi_safetensors_diff_sources_inner(&orig_data, &mod_data)
}

/// Build per-tensor diff Sources from multiple remote .safetensors files on each side.
/// Headers are fetched via HTTP range requests; tensor data is never downloaded.
fn build_multi_safetensors_diff_sources_from_http(
    orig_specs: &[&(String, RemoteFileSpec)],
    mod_specs:  &[&(String, RemoteFileSpec)],
    agent: &Arc<ureq::Agent>,
    token: Option<&Arc<String>>,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let make_arcs = |specs: &[&(String, RemoteFileSpec)]| -> Vec<Arc<Data>> {
        specs.iter().map(|(_, spec)| {
            Arc::new(Data::Http {
                url:   Arc::new(spec.cdn_url.clone()),
                token: token.cloned(),
                agent: agent.clone(),
            })
        }).collect()
    };
    let orig_data = make_arcs(orig_specs);
    let mod_data  = make_arcs(mod_specs);
    build_multi_safetensors_diff_sources_inner(&orig_data, &mod_data)
}

/// Build per-tensor diff Sources from two single .safetensors files.
fn build_safetensors_diff_sources(
    original: &Path,
    modified: &Path,
) -> anyhow::Result<(Vec<Source>, u64)> {
    build_multi_safetensors_diff_sources(
        &[original.to_path_buf()],
        &[modified.to_path_buf()],
    )
}
