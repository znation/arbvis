use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use indicatif::ProgressBar;
use memmap2::Mmap;

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
}

impl Source {
    /// Human-readable name for this source (file name or "stdin").
    pub fn name(&self) -> String {
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
pub fn prepare_sources(files: &[PathBuf]) -> anyhow::Result<(Vec<Source>, u64)> {
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
            }],
            len,
        ));
    }

    let mut sources = Vec::with_capacity(files.len());
    let mut total = 0u64;
    for (i, path) in files.iter().enumerate() {
        let size = match std::fs::metadata(path) {
            Ok(m) => m.len(),
            Err(e) => {
                eprintln!("warning: {}: {} — skipping", path.display(), e);
                continue;
            }
        };
        total += size;
        sources.push(Source {
            file_idx: i,
            kind: SourceKind::File(path.clone()),
            byte_size: size,
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
                let mut buf = vec![0u8; CHUNK];
                loop {
                    let n = f.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    for &b in &buf[..n] {
                        counts[b as usize] += 1;
                    }
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
/// For files: both must be the same size (error if not).
/// For directories: files are matched by relative path; pairs with mismatched sizes
/// or no counterpart on the other side are skipped with a warning.
pub fn prepare_diff_sources(
    original: &Path,
    modified: &Path,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let orig_is_file = original.is_file();
    let mod_is_file = modified.is_file();
    let orig_is_dir = original.is_dir();
    let mod_is_dir = modified.is_dir();

    if orig_is_file && mod_is_file {
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
