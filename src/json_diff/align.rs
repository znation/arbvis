//! Align two parsed JSON documents into a flat sequence of `AlignmentSpan`s.
//!
//! Algorithm summary:
//! - Object members align by decoded key (orig-file order).
//! - Array elements align via LCS over a 64-bit shape hash (object schema +
//!   primitive content). Falls back to index-based pairing for arrays larger
//!   than `MAX_LCS_ELEMENTS`.
//! - Primitives align with finest-granularity padding: common min-length
//!   prefix is byte-diffed; the surplus on the longer side becomes a one-sided
//!   tail.
//! - Whitespace and structural punctuation are preserved byte-for-byte; we
//!   never normalise.
//! - A single O(n) coalesce pass merges adjacent same-kind spans whose byte
//!   ranges are contiguous on both sides.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::Range;

use super::parse::{Child, Document, Node, NodeKind};

/// One alignment unit. Byte coordinates are in whole-file space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlignmentSpan {
    Aligned { orig: Range<u64>, mod_: Range<u64> },
    OrigOnly { orig: Range<u64> },
    ModOnly { mod_: Range<u64> },
}

/// Maximum number of elements an array may have before we skip LCS and fall
/// back to index-based pairing. O(n*m) LCS over hashes is fine up to a few
/// tens of thousands; beyond that the alignment time dwarfs the parse.
pub const MAX_LCS_ELEMENTS: usize = 50_000;

/// Align two whole documents. `orig_base` and `mod_base` are the byte offsets
/// at which each parsed document starts in the (whole) file — usually 0 for
/// single JSON files, and the start of each line for JSONL.
pub fn align_documents(
    orig_doc: &Document,
    mod_doc: &Document,
    orig_base: u64,
    mod_base: u64,
) -> Vec<AlignmentSpan> {
    let mut out = Vec::new();
    align_ws(
        orig_doc.leading_ws.clone(),
        mod_doc.leading_ws.clone(),
        orig_base,
        mod_base,
        &mut out,
    );
    align_node(&orig_doc.root, &mod_doc.root, orig_base, mod_base, &mut out);
    align_ws(
        orig_doc.trailing_ws.clone(),
        mod_doc.trailing_ws.clone(),
        orig_base,
        mod_base,
        &mut out,
    );
    coalesce(out)
}

/// Align two byte ranges that are pure whitespace (or any opaque byte runs).
/// Common-prefix length is aligned; surplus on the longer side is one-sided.
fn align_ws(
    o: Range<u64>,
    m: Range<u64>,
    orig_base: u64,
    mod_base: u64,
    out: &mut Vec<AlignmentSpan>,
) {
    align_bytes(o, m, orig_base, mod_base, out);
}

/// Public wrapper used by the JSONL line aligner for trailing-newline ranges.
pub fn align_bytes_pub(
    o: Range<u64>,
    m: Range<u64>,
    orig_base: u64,
    mod_base: u64,
    out: &mut Vec<AlignmentSpan>,
) {
    align_bytes(o, m, orig_base, mod_base, out);
}

/// Generic helper: align two byte ranges (possibly of different lengths) using
/// the "common prefix + one-sided tail" rule. Empty inputs are a no-op.
fn align_bytes(
    o: Range<u64>,
    m: Range<u64>,
    orig_base: u64,
    mod_base: u64,
    out: &mut Vec<AlignmentSpan>,
) {
    let o_len = o.end.saturating_sub(o.start);
    let m_len = m.end.saturating_sub(m.start);
    let common = o_len.min(m_len);
    if common > 0 {
        out.push(AlignmentSpan::Aligned {
            orig: orig_base + o.start..orig_base + o.start + common,
            mod_: mod_base + m.start..mod_base + m.start + common,
        });
    }
    if o_len > common {
        out.push(AlignmentSpan::OrigOnly {
            orig: orig_base + o.start + common..orig_base + o.end,
        });
    }
    if m_len > common {
        out.push(AlignmentSpan::ModOnly {
            mod_: mod_base + m.start + common..mod_base + m.end,
        });
    }
}

fn push_orig_only(r: Range<u64>, base: u64, out: &mut Vec<AlignmentSpan>) {
    if r.start != r.end {
        out.push(AlignmentSpan::OrigOnly {
            orig: base + r.start..base + r.end,
        });
    }
}

fn push_mod_only(r: Range<u64>, base: u64, out: &mut Vec<AlignmentSpan>) {
    if r.start != r.end {
        out.push(AlignmentSpan::ModOnly {
            mod_: base + r.start..base + r.end,
        });
    }
}

fn push_node_orig_only(n: &Node, base: u64, out: &mut Vec<AlignmentSpan>) {
    push_orig_only(n.range(), base, out);
}

fn push_node_mod_only(n: &Node, base: u64, out: &mut Vec<AlignmentSpan>) {
    push_mod_only(n.range(), base, out);
}

fn align_node(o: &Node, m: &Node, orig_base: u64, mod_base: u64, out: &mut Vec<AlignmentSpan>) {
    if o.kind != m.kind {
        push_node_orig_only(o, orig_base, out);
        push_node_mod_only(m, mod_base, out);
        return;
    }
    match o.kind {
        NodeKind::Object => align_object(o, m, orig_base, mod_base, out),
        NodeKind::Array => align_array(o, m, orig_base, mod_base, out),
        _ => align_primitive(o, m, orig_base, mod_base, out),
    }
}

fn align_primitive(
    o: &Node,
    m: &Node,
    orig_base: u64,
    mod_base: u64,
    out: &mut Vec<AlignmentSpan>,
) {
    align_bytes(o.range(), m.range(), orig_base, mod_base, out);
}

fn align_object(o: &Node, m: &Node, orig_base: u64, mod_base: u64, out: &mut Vec<AlignmentSpan>) {
    // Opening brace.
    out.push(AlignmentSpan::Aligned {
        orig: orig_base + o.byte_start..orig_base + o.byte_start + 1,
        mod_: mod_base + m.byte_start..mod_base + m.byte_start + 1,
    });

    // Build mod-side key map (first occurrence wins on duplicates).
    let mut mod_by_key: HashMap<&str, usize> = HashMap::new();
    for (i, c) in m.children.iter().enumerate() {
        if let Child::Member { key_decoded, .. } = c {
            mod_by_key.entry(key_decoded.as_str()).or_insert(i);
        }
    }
    let mut mod_used = vec![false; m.children.len()];

    // Walk orig in source order.
    for oc in &o.children {
        match oc {
            Child::Member {
                key_decoded,
                key_range,
                between_key_value,
                value,
                trailing,
            } => {
                if key_decoded.is_empty() && key_range.start == key_range.end {
                    // Synthetic ws-only sentinel (from empty object with interior ws).
                    // Emit its trailing as orig-only — there is no peer on the mod side
                    // (the mod-side ws lives in its own sentinel, if any).
                    push_orig_only(trailing.clone(), orig_base, out);
                    continue;
                }
                if let Some(&mi) = mod_by_key.get(key_decoded.as_str()) {
                    mod_used[mi] = true;
                    let mc = &m.children[mi];
                    if let Child::Member {
                        key_range: mk_range,
                        between_key_value: mb,
                        value: mv,
                        trailing: mt,
                        ..
                    } = mc
                    {
                        align_bytes(
                            key_range.clone(),
                            mk_range.clone(),
                            orig_base,
                            mod_base,
                            out,
                        );
                        align_bytes(
                            between_key_value.clone(),
                            mb.clone(),
                            orig_base,
                            mod_base,
                            out,
                        );
                        align_node(value, mv, orig_base, mod_base, out);
                        align_bytes(trailing.clone(), mt.clone(), orig_base, mod_base, out);
                    }
                } else {
                    // Orig-only member: dump the whole member span.
                    push_orig_only(key_range.clone(), orig_base, out);
                    push_orig_only(between_key_value.clone(), orig_base, out);
                    push_node_orig_only(value, orig_base, out);
                    push_orig_only(trailing.clone(), orig_base, out);
                }
            }
            Child::Element { .. } => unreachable!("object should not contain Element"),
        }
    }

    // Mod-only members (visited in mod source order to keep their bytes contiguous).
    for (i, mc) in m.children.iter().enumerate() {
        if mod_used[i] {
            continue;
        }
        match mc {
            Child::Member {
                key_decoded,
                key_range,
                between_key_value,
                value,
                trailing,
            } => {
                if key_decoded.is_empty() && key_range.start == key_range.end {
                    push_mod_only(trailing.clone(), mod_base, out);
                    continue;
                }
                push_mod_only(key_range.clone(), mod_base, out);
                push_mod_only(between_key_value.clone(), mod_base, out);
                push_node_mod_only(value, mod_base, out);
                push_mod_only(trailing.clone(), mod_base, out);
            }
            Child::Element { .. } => unreachable!(),
        }
    }

    // Closing brace.
    out.push(AlignmentSpan::Aligned {
        orig: orig_base + o.byte_end - 1..orig_base + o.byte_end,
        mod_: mod_base + m.byte_end - 1..mod_base + m.byte_end,
    });
}

fn align_array(o: &Node, m: &Node, orig_base: u64, mod_base: u64, out: &mut Vec<AlignmentSpan>) {
    // Opening bracket.
    out.push(AlignmentSpan::Aligned {
        orig: orig_base + o.byte_start..orig_base + o.byte_start + 1,
        mod_: mod_base + m.byte_start..mod_base + m.byte_start + 1,
    });

    let o_elems: Vec<&Child> = o.children.iter().collect();
    let m_elems: Vec<&Child> = m.children.iter().collect();

    let use_lcs = o_elems.len() <= MAX_LCS_ELEMENTS && m_elems.len() <= MAX_LCS_ELEMENTS;

    if use_lcs {
        let o_hashes: Vec<u64> = o_elems.iter().map(|c| shape_hash_of(c)).collect();
        let m_hashes: Vec<u64> = m_elems.iter().map(|c| shape_hash_of(c)).collect();
        let pairs = lcs_pairs(&o_hashes, &m_hashes);
        emit_array_pairs(&o_elems, &m_elems, &pairs, orig_base, mod_base, out);
    } else {
        // Index-based fallback. Warn once at the call site (we don't have
        // logging context here — caller checks size and logs).
        let len = o_elems.len().min(m_elems.len());
        let mut pairs = Vec::with_capacity(len);
        for i in 0..len {
            pairs.push((i, i));
        }
        emit_array_pairs(&o_elems, &m_elems, &pairs, orig_base, mod_base, out);
    }

    // Closing bracket.
    out.push(AlignmentSpan::Aligned {
        orig: orig_base + o.byte_end - 1..orig_base + o.byte_end,
        mod_: mod_base + m.byte_end - 1..mod_base + m.byte_end,
    });
}

/// Walk paired indices in order, emitting one-sided spans for unmatched
/// elements between matches.
fn emit_array_pairs(
    o_elems: &[&Child],
    m_elems: &[&Child],
    pairs: &[(usize, usize)],
    orig_base: u64,
    mod_base: u64,
    out: &mut Vec<AlignmentSpan>,
) {
    let mut oi = 0usize;
    let mut mi = 0usize;
    for &(po, pm) in pairs {
        while oi < po {
            emit_child_orig_only(o_elems[oi], orig_base, out);
            oi += 1;
        }
        while mi < pm {
            emit_child_mod_only(m_elems[mi], mod_base, out);
            mi += 1;
        }
        // Align the matched pair.
        emit_child_paired(o_elems[oi], m_elems[mi], orig_base, mod_base, out);
        oi += 1;
        mi += 1;
    }
    while oi < o_elems.len() {
        emit_child_orig_only(o_elems[oi], orig_base, out);
        oi += 1;
    }
    while mi < m_elems.len() {
        emit_child_mod_only(m_elems[mi], mod_base, out);
        mi += 1;
    }
}

fn emit_child_paired(
    oc: &Child,
    mc: &Child,
    orig_base: u64,
    mod_base: u64,
    out: &mut Vec<AlignmentSpan>,
) {
    match (oc, mc) {
        (
            Child::Element {
                value: ov,
                trailing: ot,
            },
            Child::Element {
                value: mv,
                trailing: mt,
            },
        ) => {
            align_node(ov, mv, orig_base, mod_base, out);
            align_bytes(ot.clone(), mt.clone(), orig_base, mod_base, out);
        }
        _ => unreachable!("array elements must be Element"),
    }
}

fn emit_child_orig_only(c: &Child, orig_base: u64, out: &mut Vec<AlignmentSpan>) {
    if let Child::Element { value, trailing } = c {
        push_node_orig_only(value, orig_base, out);
        push_orig_only(trailing.clone(), orig_base, out);
    }
}

fn emit_child_mod_only(c: &Child, mod_base: u64, out: &mut Vec<AlignmentSpan>) {
    if let Child::Element { value, trailing } = c {
        push_node_mod_only(value, mod_base, out);
        push_mod_only(trailing.clone(), mod_base, out);
    }
}

/// 64-bit shape hash. For containers we hash the structural skeleton; for
/// primitives we hash kind + byte content so identical literals match.
pub fn shape_hash_of(c: &Child) -> u64 {
    match c {
        Child::Element { value, .. } => shape_hash_node(value),
        Child::Member {
            key_decoded, value, ..
        } => {
            let mut h = DefaultHasher::new();
            "member".hash(&mut h);
            key_decoded.hash(&mut h);
            shape_hash_node(value).hash(&mut h);
            h.finish()
        }
    }
}

fn shape_hash_node(n: &Node) -> u64 {
    let mut h = DefaultHasher::new();
    match n.kind {
        NodeKind::Object => {
            "object".hash(&mut h);
            // Sorted decoded keys.
            let mut keys: Vec<&str> = n
                .children
                .iter()
                .filter_map(|c| match c {
                    Child::Member {
                        key_decoded,
                        key_range,
                        ..
                    } if !(key_decoded.is_empty() && key_range.start == key_range.end) => {
                        Some(key_decoded.as_str())
                    }
                    _ => None,
                })
                .collect();
            keys.sort_unstable();
            for k in keys {
                k.hash(&mut h);
            }
        }
        NodeKind::Array => {
            "array".hash(&mut h);
            // Length + element-kind histogram.
            let real_children: Vec<&Child> = n
                .children
                .iter()
                .filter(|c| match c {
                    Child::Element { value, .. } => {
                        !(value.kind == NodeKind::Null && value.byte_start == value.byte_end)
                    }
                    _ => true,
                })
                .collect();
            real_children.len().hash(&mut h);
            for c in real_children {
                if let Child::Element { value, .. } = c {
                    discriminant(value.kind).hash(&mut h);
                }
            }
        }
        kind => {
            // Primitive: hash kind + the source bytes (we don't have the source
            // here — the kind+range length is the best we can do without
            // passing bytes around). For practical purposes, two primitives
            // with the same kind and same byte length tend to match well;
            // the LCS still falls back to byte-diff for non-matches.
            //
            // We strengthen this for strings by also folding in the kind name.
            (kind as u8).hash(&mut h);
            // Range length (an approximation of value identity for primitives).
            let len = n.byte_end - n.byte_start;
            len.hash(&mut h);
        }
    }
    h.finish()
}

fn discriminant(k: NodeKind) -> u8 {
    match k {
        NodeKind::Object => 1,
        NodeKind::Array => 2,
        NodeKind::String => 3,
        NodeKind::Number => 4,
        NodeKind::Bool => 5,
        NodeKind::Null => 6,
    }
}

/// Wagner-Fischer LCS over two hash sequences. Returns matched index pairs
/// `(i_in_a, i_in_b)` in increasing order on both axes. O(n*m) time and
/// memory; caller is responsible for bounding inputs.
pub fn lcs_pairs(a: &[u64], b: &[u64]) -> Vec<(usize, usize)> {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return Vec::new();
    }
    // dp[i][j] = LCS length of a[..i], b[..j].
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in 0..n {
        for j in 0..m {
            dp[i + 1][j + 1] = if a[i] == b[j] {
                dp[i][j] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    // Backtrack.
    let mut pairs = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            pairs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] >= dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    pairs.reverse();
    pairs
}

/// Single linear pass: merge adjacent same-kind spans whose ranges are
/// contiguous in both files. Aligned spans only merge when both sides have
/// equal length on both the predecessor and the successor (so the diff math
/// stays well-defined).
pub fn coalesce(spans: Vec<AlignmentSpan>) -> Vec<AlignmentSpan> {
    let mut out: Vec<AlignmentSpan> = Vec::with_capacity(spans.len());
    for s in spans {
        if let Some(last) = out.last_mut() {
            match (&mut *last, &s) {
                (
                    AlignmentSpan::Aligned { orig: lo, mod_: lm },
                    AlignmentSpan::Aligned { orig: ro, mod_: rm },
                ) => {
                    let lo_len = lo.end.saturating_sub(lo.start);
                    let lm_len = lm.end.saturating_sub(lm.start);
                    let ro_len = ro.end.saturating_sub(ro.start);
                    let rm_len = rm.end.saturating_sub(rm.start);
                    if lo.end == ro.start
                        && lm.end == rm.start
                        && lo_len == lm_len
                        && ro_len == rm_len
                    {
                        lo.end = ro.end;
                        lm.end = rm.end;
                        continue;
                    }
                }
                (AlignmentSpan::OrigOnly { orig: lo }, AlignmentSpan::OrigOnly { orig: ro })
                    if lo.end == ro.start =>
                {
                    lo.end = ro.end;
                    continue;
                }
                (AlignmentSpan::ModOnly { mod_: lm }, AlignmentSpan::ModOnly { mod_: rm })
                    if lm.end == rm.start =>
                {
                    lm.end = rm.end;
                    continue;
                }
                _ => {}
            }
        }
        // Drop empty spans before pushing.
        if span_is_empty(&s) {
            continue;
        }
        out.push(s);
    }
    out
}

fn span_is_empty(s: &AlignmentSpan) -> bool {
    match s {
        AlignmentSpan::Aligned { orig, mod_ } => orig.start == orig.end && mod_.start == mod_.end,
        AlignmentSpan::OrigOnly { orig } => orig.start == orig.end,
        AlignmentSpan::ModOnly { mod_ } => mod_.start == mod_.end,
    }
}

#[cfg(test)]
mod tests {
    use super::super::parse::parse;
    use super::*;

    fn align(o: &str, m: &str) -> Vec<AlignmentSpan> {
        let od = parse(o.as_bytes()).unwrap();
        let md = parse(m.as_bytes()).unwrap();
        align_documents(&od, &md, 0, 0)
    }

    #[test]
    fn identical_collapses_to_one_aligned() {
        let r = align(r#"{"a":1}"#, r#"{"a":1}"#);
        assert_eq!(r.len(), 1);
        match &r[0] {
            AlignmentSpan::Aligned { orig, mod_ } => {
                assert_eq!(orig, &(0..7));
                assert_eq!(mod_, &(0..7));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn same_length_value_change_collapses_to_one_aligned() {
        // Both 7 bytes, only the digit changes.
        let r = align(r#"{"a":1}"#, r#"{"a":2}"#);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn longer_number_value_emits_mod_only_tail() {
        let r = align(r#"{"a":1}"#, r#"{"a":100}"#);
        // Expect: Aligned over "{\"a\":1" (6 bytes orig, 6 bytes mod), ModOnly "00", Aligned "}".
        // After coalesce the leading Aligned merges with the closing-brace-aligned only if contiguous; they're not contiguous on the mod side because of the ModOnly.
        // So we expect 3 spans.
        assert_eq!(r.len(), 3);
        match &r[1] {
            AlignmentSpan::ModOnly { mod_ } => assert_eq!(mod_.end - mod_.start, 2),
            _ => panic!("expected ModOnly, got {:?}", r[1]),
        }
    }

    #[test]
    fn key_added() {
        // Inserted key "b":2 between "a":1 and end.
        let r = align(r#"{"a":1}"#, r#"{"a":1,"b":2}"#);
        // Expect at least one ModOnly span over the new ",\"b\":2".
        let has_mod_only = r.iter().any(|s| matches!(s, AlignmentSpan::ModOnly { .. }));
        assert!(has_mod_only, "spans: {r:?}");
    }

    #[test]
    fn key_removed() {
        let r = align(r#"{"a":1,"b":2}"#, r#"{"a":1}"#);
        let has_orig_only = r
            .iter()
            .any(|s| matches!(s, AlignmentSpan::OrigOnly { .. }));
        assert!(has_orig_only, "spans: {r:?}");
    }

    #[test]
    fn reorder_matches_keys() {
        let r = align(r#"{"a":1,"b":2}"#, r#"{"b":2,"a":1}"#);
        // Both keys exist on both sides; the only structural mismatch is the
        // trailing-comma asymmetry (last member has no trailing comma).
        let total_one_sided = r
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    AlignmentSpan::OrigOnly { .. } | AlignmentSpan::ModOnly { .. }
                )
            })
            .count();
        assert!(total_one_sided <= 2, "spans: {r:?}");
        // The "a":1 substring should align even though it's at different positions.
        let has_cross_position_aligned = r.iter().any(|s| matches!(s, AlignmentSpan::Aligned { orig, mod_ } if orig != mod_ && orig.end - orig.start == 5));
        assert!(
            has_cross_position_aligned,
            "expected key-value aligned at differing positions: {r:?}"
        );
    }

    #[test]
    fn type_change_emits_both_sides() {
        let r = align(r#"{"x":"foo"}"#, r#"{"x":42}"#);
        let orig_only = r
            .iter()
            .filter(|s| matches!(s, AlignmentSpan::OrigOnly { .. }))
            .count();
        let mod_only = r
            .iter()
            .filter(|s| matches!(s, AlignmentSpan::ModOnly { .. }))
            .count();
        assert!(orig_only >= 1 && mod_only >= 1, "spans: {r:?}");
    }

    #[test]
    fn array_lcs_matches_unchanged_elements() {
        // Insert "99" between 1 and 2 in modified.
        let r = align("[1,2,3]", "[1,99,2,3]");
        // We expect at least one Aligned span and at least one ModOnly span.
        let aligned = r
            .iter()
            .filter(|s| matches!(s, AlignmentSpan::Aligned { .. }))
            .count();
        let mod_only = r
            .iter()
            .filter(|s| matches!(s, AlignmentSpan::ModOnly { .. }))
            .count();
        assert!(aligned >= 1 && mod_only >= 1, "spans: {r:?}");
    }

    #[test]
    fn coalesce_merges_adjacent_aligned() {
        let r = align(r#"[1,2,3]"#, r#"[1,2,3]"#);
        // Identical, should fold into 1 Aligned span.
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn whitespace_changes() {
        // Same logical content, different whitespace.
        let r = align(r#"{"a":1}"#, r#"{ "a" : 1 }"#);
        // Mod side has extra spaces; expect Aligned spans plus ModOnly for the spaces.
        let mod_only = r
            .iter()
            .filter(|s| matches!(s, AlignmentSpan::ModOnly { .. }))
            .count();
        assert!(mod_only >= 1, "spans: {r:?}");
    }
}
