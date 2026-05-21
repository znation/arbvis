//! Translate a sequence of `AlignmentSpan`s into a `Vec<Source>` that slots
//! into the existing canvas layout.

use std::sync::Arc;

use crate::data::{Data, Source, SourceKind};
use crate::safetensors::DiffFill;

use super::align::AlignmentSpan;

/// Convert a coalesced span sequence into Sources. `orig_data` and `mod_data`
/// are the shared whole-file Data handles each `RangeDiff` / `OneSidedRange`
/// references via `Arc::clone`.
///
/// Color choice for one-sided spans follows the directory-diff convention:
/// `is_finetune` → Grey for orig-only, Green for mod-only.
/// Otherwise → Red for orig-only, Green for mod-only.
pub fn spans_to_sources(
    spans: &[AlignmentSpan],
    orig_data: Arc<Data>,
    mod_data: Arc<Data>,
    is_finetune: bool,
    orig_label: &str,
    mod_label: &str,
) -> (Vec<Source>, u64) {
    let orig_fill = if is_finetune {
        DiffFill::Grey
    } else {
        DiffFill::Red
    };

    let mut sources: Vec<Source> = Vec::with_capacity(spans.len() * 2);
    let mut total: u64 = 0;

    for span in spans {
        match span {
            AlignmentSpan::Aligned { orig, mod_ } => {
                let o_len = orig.end.saturating_sub(orig.start);
                let m_len = mod_.end.saturating_sub(mod_.start);
                if o_len == 0 && m_len == 0 {
                    continue;
                }
                let common = o_len.min(m_len);
                if common > 0 {
                    let idx = sources.len();
                    sources.push(Source {
                        file_idx: idx,
                        kind: SourceKind::RangeDiff {
                            orig: Arc::clone(&orig_data),
                            mod_: Arc::clone(&mod_data),
                            orig_start: orig.start,
                            mod_start: mod_.start,
                        },
                        byte_size: common,
                        safetensors: None,
                        name_override: Some(format!(
                            "diff @ orig:[{}, {}) vs mod:[{}, {})",
                            orig.start,
                            orig.start + common,
                            mod_.start,
                            mod_.start + common
                        )),
                        xet_terms: None,
                    });
                    total += common;
                }
                // Length-mismatched tail. We treat the surplus as a one-sided
                // structural region rendered on the same canvas with a tinted
                // overlay so the user can read the inserted/deleted bytes.
                if o_len > common {
                    let len = o_len - common;
                    let idx = sources.len();
                    sources.push(Source {
                        file_idx: idx,
                        kind: SourceKind::OneSidedRange {
                            data: Arc::clone(&orig_data),
                            start: orig.start + common,
                            fill: orig_fill,
                        },
                        byte_size: len,
                        safetensors: None,
                        name_override: Some(format!(
                            "[only in original] {} @ [{}, {})",
                            orig_label,
                            orig.start + common,
                            orig.end
                        )),
                        xet_terms: None,
                    });
                    total += len;
                }
                if m_len > common {
                    let len = m_len - common;
                    let idx = sources.len();
                    sources.push(Source {
                        file_idx: idx,
                        kind: SourceKind::OneSidedRange {
                            data: Arc::clone(&mod_data),
                            start: mod_.start + common,
                            fill: DiffFill::Green,
                        },
                        byte_size: len,
                        safetensors: None,
                        name_override: Some(format!(
                            "[only in modified] {} @ [{}, {})",
                            mod_label,
                            mod_.start + common,
                            mod_.end
                        )),
                        xet_terms: None,
                    });
                    total += len;
                }
            }
            AlignmentSpan::OrigOnly { orig } => {
                let len = orig.end.saturating_sub(orig.start);
                if len == 0 {
                    continue;
                }
                let idx = sources.len();
                sources.push(Source {
                    file_idx: idx,
                    kind: SourceKind::OneSidedRange {
                        data: Arc::clone(&orig_data),
                        start: orig.start,
                        fill: orig_fill,
                    },
                    byte_size: len,
                    safetensors: None,
                    name_override: Some(format!(
                        "[only in original] {} @ [{}, {})",
                        orig_label, orig.start, orig.end
                    )),
                    xet_terms: None,
                });
                total += len;
            }
            AlignmentSpan::ModOnly { mod_ } => {
                let len = mod_.end.saturating_sub(mod_.start);
                if len == 0 {
                    continue;
                }
                let idx = sources.len();
                sources.push(Source {
                    file_idx: idx,
                    kind: SourceKind::OneSidedRange {
                        data: Arc::clone(&mod_data),
                        start: mod_.start,
                        fill: DiffFill::Green,
                    },
                    byte_size: len,
                    safetensors: None,
                    name_override: Some(format!(
                        "[only in modified] {} @ [{}, {})",
                        mod_label, mod_.start, mod_.end
                    )),
                    xet_terms: None,
                });
                total += len;
            }
        }
    }

    (sources, total)
}
