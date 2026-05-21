//! JSON / JSONL structure-aware diff.
//!
//! Entry point: [`build_json_diff_sources`]. Given two file paths whose
//! extensions are `.json` or `.jsonl`, parse both, align by structure, and
//! produce a `Vec<Source>` for the rendering pipeline.
//!
//! The aligner pads at the finest structural granularity (object keys, array
//! elements, primitive value prefixes) so a one-byte insertion near the top
//! of a file doesn't smear every following byte across the canvas. See
//! [`align`] for algorithm details.

pub mod align;
pub mod parse;
pub mod source;

use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::data::{Data, Source, SourceKind};
use crate::safetensors::DiffFill;

use align::{align_documents, coalesce, AlignmentSpan};
use parse::parse;

/// Hard cap on the number of post-coalesce spans we'll emit for a single
/// diff. Beyond this, the per-source cumulative-offsets array bloats memory
/// and the renderer slows down without proportional value. The caller falls
/// back to a whole-file byte diff with a `log::warn!` when exceeded.
const MAX_SPANS: usize = 1_000_000;

/// Top-level entry. Detects `.jsonl` (line-delimited) vs `.json` (single
/// document) and dispatches.
pub async fn build_json_diff_sources(
    original: &Path,
    modified: &Path,
    is_finetune: bool,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let orig_bytes = std::fs::read(original)
        .with_context(format!("reading {}", original.display()))?;
    let mod_bytes = std::fs::read(modified)
        .with_context(format!("reading {}", modified.display()))?;

    let orig_label = filename_label(original);
    let mod_label = filename_label(modified);

    let is_jsonl = matches!(
        original.extension().and_then(|e| e.to_str()),
        Some("jsonl")
    ) || matches!(
        modified.extension().and_then(|e| e.to_str()),
        Some("jsonl")
    );

    // Mmap each file for the per-tile RangeDiff / OneSidedRange path. Even
    // though we already have the bytes in `orig_bytes`/`mod_bytes`, the
    // existing diff machinery expects `Arc<Data>` handles, and reusing the
    // mmap path keeps the lazy-fetch semantics for free.
    let orig_data = open_mmap_data(original)?;
    let mod_data = open_mmap_data(modified)?;

    let spans = if is_jsonl {
        align_jsonl(&orig_bytes, &mod_bytes)
    } else {
        match (parse(&orig_bytes), parse(&mod_bytes)) {
            (Ok(o), Ok(m)) => align_documents(&o, &m, 0, 0),
            (orig_r, mod_r) => {
                log_parse_failures(original, &orig_r, modified, &mod_r);
                return fallback_byte_diff(original, modified, &orig_bytes, &mod_bytes, is_finetune, orig_data, mod_data);
            }
        }
    };

    if spans.len() > MAX_SPANS {
        log::warn!(
            "json-diff: {} post-coalesce spans for {} vs {} exceeds cap ({}) — \
             falling back to whole-file byte diff",
            spans.len(),
            original.display(),
            modified.display(),
            MAX_SPANS
        );
        return fallback_byte_diff(original, modified, &orig_bytes, &mod_bytes, is_finetune, orig_data, mod_data);
    }

    let (sources, total) = source::spans_to_sources(
        &spans,
        orig_data,
        mod_data,
        is_finetune,
        &orig_label,
        &mod_label,
    );

    log::info!(
        "json-diff: {} → {} ({} sources, {} bytes)",
        original.display(),
        modified.display(),
        sources.len(),
        total
    );
    Ok((sources, total))
}

fn align_jsonl(orig: &[u8], mod_: &[u8]) -> Vec<AlignmentSpan> {
    let orig_lines = split_lines(orig);
    let mod_lines = split_lines(mod_);

    // Hash each line by content (siphash via DefaultHasher) for LCS pairing.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let hash = |s: &[u8]| -> u64 {
        let mut h = DefaultHasher::new();
        s.hash(&mut h);
        h.finish()
    };
    let cap = align::MAX_LCS_ELEMENTS;
    let pairs = if orig_lines.len() <= cap && mod_lines.len() <= cap {
        let o_hashes: Vec<u64> = orig_lines.iter().map(|r| hash(&orig[r.start as usize..r.end as usize])).collect();
        let m_hashes: Vec<u64> = mod_lines.iter().map(|r| hash(&mod_[r.start as usize..r.end as usize])).collect();
        align::lcs_pairs(&o_hashes, &m_hashes)
    } else {
        log::warn!(
            "json-diff: jsonl file has {} / {} lines — exceeding LCS cap ({}), \
             falling back to index-based line matching",
            orig_lines.len(), mod_lines.len(), cap
        );
        let len = orig_lines.len().min(mod_lines.len());
        (0..len).map(|i| (i, i)).collect()
    };

    let mut out = Vec::new();
    let mut oi = 0usize;
    let mut mi = 0usize;
    for &(po, pm) in &pairs {
        while oi < po {
            out.push(AlignmentSpan::OrigOnly { orig: orig_lines[oi].clone() });
            oi += 1;
        }
        while mi < pm {
            out.push(AlignmentSpan::ModOnly { mod_: mod_lines[mi].clone() });
            mi += 1;
        }
        let or = orig_lines[oi].clone();
        let mr = mod_lines[mi].clone();
        // If line bytes are identical, emit a single Aligned span over both
        // (cheap, common). Otherwise, try a structural parse + align of the
        // two lines and translate the resulting span byte coordinates into
        // whole-file space.
        let o_slice = &orig[or.start as usize..or.end as usize];
        let m_slice = &mod_[mr.start as usize..mr.end as usize];
        if o_slice == m_slice {
            out.push(AlignmentSpan::Aligned { orig: or, mod_: mr });
        } else {
            // Strip a trailing newline before parsing (the parser is strict).
            let (o_payload, o_nl) = strip_trailing_newline(o_slice);
            let (m_payload, m_nl) = strip_trailing_newline(m_slice);
            match (parse(o_payload), parse(m_payload)) {
                (Ok(od), Ok(md)) => {
                    let line_spans = align_documents(&od, &md, or.start, mr.start);
                    out.extend(line_spans);
                    // Align the trailing newline byte(s) explicitly.
                    if o_nl > 0 || m_nl > 0 {
                        align::align_bytes_pub(
                            or.end - o_nl..or.end,
                            mr.end - m_nl..mr.end,
                            0, 0,
                            &mut out,
                        );
                    }
                }
                _ => {
                    // Fall back to byte-level align over the two lines.
                    align::align_bytes_pub(or.clone(), mr.clone(), 0, 0, &mut out);
                }
            }
        }
        oi += 1;
        mi += 1;
    }
    while oi < orig_lines.len() {
        out.push(AlignmentSpan::OrigOnly { orig: orig_lines[oi].clone() });
        oi += 1;
    }
    while mi < mod_lines.len() {
        out.push(AlignmentSpan::ModOnly { mod_: mod_lines[mi].clone() });
        mi += 1;
    }
    coalesce(out)
}

fn split_lines(src: &[u8]) -> Vec<std::ops::Range<u64>> {
    let mut out = Vec::new();
    let mut start = 0usize;
    for (i, &b) in src.iter().enumerate() {
        if b == b'\n' {
            out.push(start as u64..(i + 1) as u64);
            start = i + 1;
        }
    }
    if start < src.len() {
        out.push(start as u64..src.len() as u64);
    }
    out
}

fn strip_trailing_newline(s: &[u8]) -> (&[u8], u64) {
    if let Some((&last, rest)) = s.split_last() {
        if last == b'\n' {
            if let Some((&prev, body)) = rest.split_last() {
                if prev == b'\r' {
                    return (body, 2);
                }
            }
            return (rest, 1);
        }
    }
    (s, 0)
}

fn open_mmap_data(path: &Path) -> anyhow::Result<Arc<Data>> {
    let f = std::fs::File::open(path)
        .with_context(format!("opening {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&f)? };
    Ok(Arc::new(Data::Mapped(mmap)))
}

fn filename_label(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string_lossy().into_owned())
}

fn log_parse_failures(
    orig_path: &Path,
    orig_r: &Result<parse::Document, parse::ParseError>,
    mod_path: &Path,
    mod_r: &Result<parse::Document, parse::ParseError>,
) {
    if let Err(e) = orig_r {
        log::warn!(
            "json-diff: failed to parse {} ({}) — falling back to byte diff",
            orig_path.display(), e
        );
    }
    if let Err(e) = mod_r {
        log::warn!(
            "json-diff: failed to parse {} ({}) — falling back to byte diff",
            mod_path.display(), e
        );
    }
}

/// Same fallback shape as the directory-diff path: same-size → single byte
/// Diff Source; size mismatch → two crosshatched UnmatchedRegion Sources.
fn fallback_byte_diff(
    orig_path: &Path,
    mod_path: &Path,
    orig_bytes: &[u8],
    mod_bytes: &[u8],
    is_finetune: bool,
    _orig_data: Arc<Data>,
    _mod_data: Arc<Data>,
) -> anyhow::Result<(Vec<Source>, u64)> {
    let orig_fill = if is_finetune { DiffFill::Grey } else { DiffFill::Red };
    if orig_bytes.len() == mod_bytes.len() {
        let size = orig_bytes.len() as u64;
        let source = Source {
            file_idx: 0,
            kind: SourceKind::Diff {
                original: orig_path.to_path_buf(),
                modified: mod_path.to_path_buf(),
            },
            byte_size: size,
            safetensors: None,
            name_override: None,
            xet_terms: None,
        };
        return Ok((vec![source], size));
    }
    let mut sources = Vec::new();
    let mut total = 0u64;
    if !orig_bytes.is_empty() {
        let size = orig_bytes.len() as u64;
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill: orig_fill },
            byte_size: size,
            safetensors: None,
            name_override: Some(format!("[only in original] {}", filename_label(orig_path))),
            xet_terms: None,
        });
        total += size;
    }
    if !mod_bytes.is_empty() {
        let size = mod_bytes.len() as u64;
        sources.push(Source {
            file_idx: sources.len(),
            kind: SourceKind::UnmatchedRegion { fill: DiffFill::Green },
            byte_size: size,
            safetensors: None,
            name_override: Some(format!("[only in modified] {}", filename_label(mod_path))),
            xet_terms: None,
        });
        total += size;
    }
    Ok((sources, total))
}

/// Small wrapper trait so call sites can chain `.with_context(format!(...))`
/// without pulling in `anyhow::Context` everywhere.
trait WithContextStr<T> {
    fn with_context(self, msg: String) -> anyhow::Result<T>;
}
impl<T, E: Into<anyhow::Error>> WithContextStr<T> for Result<T, E> {
    fn with_context(self, msg: String) -> anyhow::Result<T> {
        self.map_err(|e| {
            let e: anyhow::Error = e.into();
            e.context(msg)
        })
    }
}
