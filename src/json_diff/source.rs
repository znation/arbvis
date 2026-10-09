//! Translate a sequence of `AlignmentSpan`s into a `Vec<Source>` that slots
//! into the existing canvas layout.

use std::sync::Arc;

use crate::data::{Data, DiffFill, Extensions, Source, SourceKind};

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
    let orig_fill = DiffFill::orig_only(is_finetune);

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
                        name_override: Some(format!(
                            "diff @ orig:[{}, {}) vs mod:[{}, {})",
                            orig.start,
                            orig.start + common,
                            mod_.start,
                            mod_.start + common
                        )),
                        xet_terms: None,
                        extensions: Extensions::default(),
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
                        name_override: Some(format!(
                            "[only in original] {} @ [{}, {})",
                            orig_label,
                            orig.start + common,
                            orig.end
                        )),
                        xet_terms: None,
                        extensions: Extensions::default(),
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
                        name_override: Some(format!(
                            "[only in modified] {} @ [{}, {})",
                            mod_label,
                            mod_.start + common,
                            mod_.end
                        )),
                        xet_terms: None,
                        extensions: Extensions::default(),
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
                    name_override: Some(format!(
                        "[only in original] {} @ [{}, {})",
                        orig_label, orig.start, orig.end
                    )),
                    xet_terms: None,
                    extensions: Extensions::default(),
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
                    name_override: Some(format!(
                        "[only in modified] {} @ [{}, {})",
                        mod_label, mod_.start, mod_.end
                    )),
                    xet_terms: None,
                    extensions: Extensions::default(),
                });
                total += len;
            }
        }
    }

    (sources, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Arc<Data> {
        Arc::new(Data::Owned(vec![0u8; 16]))
    }

    fn one_sided(s: &Source) -> (u64, DiffFill) {
        match s.kind {
            SourceKind::OneSidedRange { start, fill, .. } => (start, fill),
            _ => panic!("expected OneSidedRange"),
        }
    }

    #[test]
    fn aligned_equal_lengths_become_one_range_diff() {
        let (sources, total) = spans_to_sources(
            &[AlignmentSpan::Aligned {
                orig: 4..10,
                mod_: 2..8,
            }],
            data(),
            data(),
            false,
            "o",
            "m",
        );
        assert_eq!(sources.len(), 1);
        assert_eq!(total, 6);
        assert_eq!(sources[0].byte_size, 6);
        match &sources[0].kind {
            SourceKind::RangeDiff {
                orig_start,
                mod_start,
                ..
            } => {
                assert_eq!(*orig_start, 4);
                assert_eq!(*mod_start, 2);
            }
            _ => panic!("expected RangeDiff"),
        }
        assert!(sources[0]
            .name_override
            .as_deref()
            .unwrap()
            .contains("diff @ orig:[4, 10) vs mod:[2, 8)"));
    }

    #[test]
    fn aligned_mismatched_lengths_emit_tail_one_sided_sources() {
        // orig 5 bytes vs mod 2 bytes: 2 common + 3 orig-only tail.
        let orig = data();
        let (sources, total) = spans_to_sources(
            &[AlignmentSpan::Aligned {
                orig: 0..5,
                mod_: 8..10,
            }],
            Arc::clone(&orig),
            data(),
            false,
            "orig.json",
            "mod.json",
        );
        assert_eq!(sources.len(), 2);
        assert_eq!(total, 5);
        assert_eq!(sources[0].byte_size, 2);
        let (start, fill) = one_sided(&sources[1]);
        assert_eq!(start, 2); // orig.start + common
        assert_eq!(fill, DiffFill::Red);
        assert_eq!(sources[1].byte_size, 3);
        assert!(sources[1]
            .name_override
            .as_deref()
            .unwrap()
            .contains("[only in original] orig.json @ [2, 5)"));
    }

    #[test]
    fn finetune_uses_grey_for_orig_only() {
        let (sources, _) = spans_to_sources(
            &[AlignmentSpan::Aligned {
                orig: 0..4,
                mod_: 0..2,
            }],
            data(),
            data(),
            true,
            "o",
            "m",
        );
        let (_, fill) = one_sided(&sources[1]);
        assert_eq!(fill, DiffFill::Grey);
    }

    #[test]
    fn empty_aligned_span_is_skipped() {
        let (sources, total) = spans_to_sources(
            &[
                AlignmentSpan::Aligned {
                    orig: 3..3,
                    mod_: 7..7,
                },
                AlignmentSpan::OrigOnly { orig: 0..0 },
                AlignmentSpan::ModOnly { mod_: 1..1 },
            ],
            data(),
            data(),
            false,
            "o",
            "m",
        );
        assert!(sources.is_empty());
        assert_eq!(total, 0);
    }

    #[test]
    fn orig_only_and_mod_only_use_directory_diff_colors() {
        let (sources, total) = spans_to_sources(
            &[
                AlignmentSpan::OrigOnly { orig: 0..3 },
                AlignmentSpan::ModOnly { mod_: 4..9 },
            ],
            data(),
            data(),
            false,
            "orig.json",
            "mod.json",
        );
        assert_eq!(sources.len(), 2);
        assert_eq!(total, 8);
        let (start, fill) = one_sided(&sources[0]);
        assert_eq!((start, fill), (0, DiffFill::Red));
        assert_eq!(sources[0].byte_size, 3);
        assert!(sources[0]
            .name_override
            .as_deref()
            .unwrap()
            .contains("[only in original] orig.json @ [0, 3)"));
        let (start, fill) = one_sided(&sources[1]);
        assert_eq!((start, fill), (4, DiffFill::Green));
        assert_eq!(sources[1].byte_size, 5);
        assert!(sources[1]
            .name_override
            .as_deref()
            .unwrap()
            .contains("[only in modified] mod.json @ [4, 9)"));
    }

    #[test]
    fn file_idx_is_sequential_and_matches_position() {
        let (sources, _) = spans_to_sources(
            &[
                AlignmentSpan::Aligned {
                    orig: 0..2,
                    mod_: 0..4,
                },
                AlignmentSpan::OrigOnly { orig: 5..6 },
            ],
            data(),
            data(),
            false,
            "o",
            "m",
        );
        for (i, s) in sources.iter().enumerate() {
            assert_eq!(s.file_idx, i);
        }
        assert_eq!(sources.len(), 3);
    }
}
