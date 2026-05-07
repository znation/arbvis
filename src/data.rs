use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;

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
        }
    }
}

/// Build sources and return total byte count.
///
/// Files are opened lazily (one at a time) to avoid exhausting OS fd limits.
/// Stdin is buffered into memory upfront since its size is unknown.
pub fn prepare_sources(
    files: &[PathBuf],
) -> anyhow::Result<(Vec<Source>, u64)> {
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

/// Load a source's bytes, optionally sorting them by value.
///
/// For file sources: mmaps the file (sort=false) or reads and sorts it
/// (sort=true). For buffered sources (stdin): clones the buffer, sorting
/// if requested.
pub fn load_source_data(s: &Source, sort: bool) -> anyhow::Result<Data> {
    match &s.kind {
        SourceKind::File(p) => {
            if sort {
                let mut bytes = std::fs::read(p)?;
                bytes.sort_unstable();
                Ok(Data::Owned(bytes))
            } else {
                let f = File::open(p)?;
                Ok(Data::Mapped(unsafe { Mmap::map(&f) }?))
            }
        }
        SourceKind::Buffered(v) => {
            if sort {
                let mut bytes = v.clone();
                bytes.sort_unstable();
                Ok(Data::Owned(bytes))
            } else {
                Ok(Data::Owned(v.clone()))
            }
        }
    }
}