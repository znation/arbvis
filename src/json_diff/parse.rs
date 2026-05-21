//! Hand-rolled byte-position-preserving JSON parser.
//!
//! The output is a `Document` whose tree covers every byte of the input
//! exactly once (concatenating all emitted ranges in document order reproduces
//! the input). That invariant is what lets the diff aligner translate
//! structural matches back into byte ranges without losing whitespace, commas,
//! or any other source-level detail.
//!
//! Strict RFC 8259: no comments, no trailing commas. Errors carry the byte
//! offset that triggered the failure so callers can log a useful message and
//! fall back to byte-level diffing.

use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Object,
    Array,
    String,
    Number,
    Bool,
    Null,
}

#[derive(Debug)]
pub struct Node {
    pub kind: NodeKind,
    pub byte_start: u64,
    pub byte_end: u64,
    pub children: Vec<Child>,
}

impl Node {
    pub fn range(&self) -> Range<u64> {
        self.byte_start..self.byte_end
    }
}

#[derive(Debug)]
pub enum Child {
    /// Object member: a quoted key, the colon-and-whitespace separator, the
    /// value subtree, and any trailing comma + whitespace before the next key.
    Member {
        key_range: Range<u64>,
        key_decoded: String,
        between_key_value: Range<u64>,
        value: Box<Node>,
        trailing: Range<u64>,
    },
    /// Array element: a value subtree and any trailing comma + whitespace.
    Element {
        value: Box<Node>,
        trailing: Range<u64>,
    },
}

#[derive(Debug)]
pub struct Document {
    pub leading_ws: Range<u64>,
    pub root: Node,
    pub trailing_ws: Range<u64>,
}

#[derive(Debug)]
pub struct ParseError {
    pub byte_offset: u64,
    pub msg: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "at byte {}: {}", self.byte_offset, self.msg)
    }
}

impl std::error::Error for ParseError {}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a [u8]) -> Self {
        Parser { src, pos: 0 }
    }

    fn err(&self, msg: impl Into<String>) -> ParseError {
        ParseError {
            byte_offset: self.pos as u64,
            msg: msg.into(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn expect(&mut self, b: u8, what: &str) -> Result<(), ParseError> {
        match self.peek() {
            Some(x) if x == b => {
                self.pos += 1;
                Ok(())
            }
            Some(other) => {
                Err(self.err(format!("expected {what} (0x{b:02x}), found 0x{other:02x}")))
            }
            None => Err(self.err(format!("expected {what} (0x{b:02x}), found EOF"))),
        }
    }

    /// Consume RFC 8259 whitespace (' ', '\t', '\n', '\r'). Returns the byte
    /// range that was skipped.
    fn skip_ws(&mut self) -> Range<u64> {
        let start = self.pos as u64;
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
        start..self.pos as u64
    }

    fn parse_value(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => {
                let (_decoded, end) = self.parse_string_at(start)?;
                Ok(Node {
                    kind: NodeKind::String,
                    byte_start: start as u64,
                    byte_end: end as u64,
                    children: vec![],
                })
            }
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(b) if b == b'-' || b.is_ascii_digit() => self.parse_number(),
            Some(b) => Err(self.err(format!("unexpected byte 0x{b:02x} at start of value"))),
            None => Err(self.err("unexpected EOF where value expected")),
        }
    }

    fn parse_object(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.expect(b'{', "'{'")?;
        let mut children: Vec<Child> = Vec::new();

        // Empty object: optional whitespace then '}'.
        let ws_after_open = self.skip_ws();
        if let Some(b'}') = self.peek() {
            self.pos += 1;
            // Fold the open-brace whitespace into a final synthetic trailing
            // range on the (zero) members. Simpler: hang it as a single
            // "trailing" Member-less Element? No: we represent the empty
            // object as zero children and let the alignment treat the
            // interior whitespace as part of the parent's surrounding bytes.
            // To preserve the round-trip invariant, encode the interior ws as
            // a synthetic Element whose value has zero length.
            if !ws_after_open.is_empty() {
                // Emit a sentinel Member with empty key_range / value spanning the
                // whitespace via the `trailing` field. This is a zero-info entry
                // for the aligner but keeps the round-trip whole.
                children.push(Child::Member {
                    key_range: ws_after_open.start..ws_after_open.start,
                    key_decoded: String::new(),
                    between_key_value: ws_after_open.start..ws_after_open.start,
                    value: Box::new(Node {
                        kind: NodeKind::Null,
                        byte_start: ws_after_open.start,
                        byte_end: ws_after_open.start,
                        children: vec![],
                    }),
                    trailing: ws_after_open.clone(),
                });
            }
            return Ok(Node {
                kind: NodeKind::Object,
                byte_start: start as u64,
                byte_end: self.pos as u64,
                children,
            });
        }

        // Non-empty: parse member, then optional comma + member, until '}'.
        // Roll the pre-first-key whitespace into the first member's key_range
        // prefix would break key decoding; instead extend the opening brace's
        // notion of "before first key" by attaching it as trailing of a synthetic
        // entry only if non-empty AND there are no real members. For non-empty
        // objects we keep ws_after_open as the prefix of `between_key_value` for
        // the first key — except `between_key_value` runs between key and value,
        // not before key. So we attach `ws_after_open` to a hidden field on the
        // first member: extend its key_range start backwards by the ws length.
        // Simpler and round-trip-safe: prefix the first member's effective key
        // range with the whitespace, but still decode only the quoted key.
        let mut leading_member_ws_start = ws_after_open.start;
        loop {
            let key_start = self.pos;
            if self.peek() != Some(b'"') {
                return Err(self.err("expected '\"' to start object key"));
            }
            let (key_decoded, key_end_excl) = self.parse_string_at(key_start)?;
            let key_range = (leading_member_ws_start)..(key_end_excl as u64);

            let between_start = self.pos;
            self.skip_ws();
            self.expect(b':', "':'")?;
            self.skip_ws();
            let between_end = self.pos as u64;
            let between_key_value = between_start as u64..between_end;

            let value = self.parse_value()?;
            let trailing_start = self.pos;
            self.skip_ws();
            // Optional comma.
            let saw_comma = if self.peek() == Some(b',') {
                self.pos += 1;
                self.skip_ws();
                true
            } else {
                false
            };
            let trailing = trailing_start as u64..self.pos as u64;

            children.push(Child::Member {
                key_range,
                key_decoded,
                between_key_value,
                value: Box::new(value),
                trailing,
            });

            if saw_comma {
                // Next iteration's key has no leading-ws prefix (it was rolled
                // into the previous member's `trailing`).
                leading_member_ws_start = self.pos as u64;
                continue;
            }

            // No comma → expect '}'.
            self.expect(b'}', "'}' to close object")?;
            return Ok(Node {
                kind: NodeKind::Object,
                byte_start: start as u64,
                byte_end: self.pos as u64,
                children,
            });
        }
    }

    fn parse_array(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        self.expect(b'[', "'['")?;
        let mut children: Vec<Child> = Vec::new();

        let ws_after_open = self.skip_ws();
        if let Some(b']') = self.peek() {
            self.pos += 1;
            if !ws_after_open.is_empty() {
                // Same trick as empty object: encode interior whitespace as a
                // sentinel zero-length element with trailing = ws.
                children.push(Child::Element {
                    value: Box::new(Node {
                        kind: NodeKind::Null,
                        byte_start: ws_after_open.start,
                        byte_end: ws_after_open.start,
                        children: vec![],
                    }),
                    trailing: ws_after_open.clone(),
                });
            }
            return Ok(Node {
                kind: NodeKind::Array,
                byte_start: start as u64,
                byte_end: self.pos as u64,
                children,
            });
        }

        // Non-empty: first element absorbs the leading whitespace by extending
        // its value start backwards. Easier and round-trip-safe: prepend a
        // hidden whitespace child only if non-empty AND the first element
        // wouldn't include it otherwise. Since values don't carry surrounding
        // whitespace, we instead carry the leading ws as part of the parent's
        // pseudo-Element trailing for index -1. But we have no index -1.
        //
        // Simplest correct approach: extend the first element's `value`
        // byte_start back to cover the whitespace. The aligner doesn't decode
        // bytes inside a primitive node — it just diffs ranges — and the
        // structural matching by index doesn't care that the first element's
        // range is slightly larger than just its literal. We do this by
        // pushing a synthetic "ws-only" Element with kind=Null and a zero-byte
        // value, but with `trailing` covering the ws. This keeps the
        // round-trip whole while leaving real elements untouched.
        if !ws_after_open.is_empty() {
            children.push(Child::Element {
                value: Box::new(Node {
                    kind: NodeKind::Null,
                    byte_start: ws_after_open.start,
                    byte_end: ws_after_open.start,
                    children: vec![],
                }),
                trailing: ws_after_open.clone(),
            });
        }
        loop {
            let value = self.parse_value()?;
            let trailing_start = self.pos;
            self.skip_ws();
            let saw_comma = if self.peek() == Some(b',') {
                self.pos += 1;
                self.skip_ws();
                true
            } else {
                false
            };
            let trailing = trailing_start as u64..self.pos as u64;
            children.push(Child::Element {
                value: Box::new(value),
                trailing,
            });

            if saw_comma {
                continue;
            }
            self.expect(b']', "']' to close array")?;
            return Ok(Node {
                kind: NodeKind::Array,
                byte_start: start as u64,
                byte_end: self.pos as u64,
                children,
            });
        }
    }

    /// Parse a JSON string starting at `start_pos` (which must be the opening
    /// quote). Returns the decoded string and the exclusive end position
    /// (one past the closing quote). Self.pos is advanced to that end.
    fn parse_string_at(&mut self, start_pos: usize) -> Result<(String, usize), ParseError> {
        debug_assert_eq!(self.src.get(start_pos), Some(&b'"'));
        self.pos = start_pos + 1;
        let mut out = String::new();
        loop {
            match self.advance() {
                None => {
                    return Err(ParseError {
                        byte_offset: self.pos as u64,
                        msg: "unterminated string".into(),
                    })
                }
                Some(b'"') => return Ok((out, self.pos)),
                Some(b'\\') => {
                    match self.advance() {
                        None => {
                            return Err(ParseError {
                                byte_offset: self.pos as u64,
                                msg: "EOF in escape".into(),
                            })
                        }
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'b') => out.push('\u{0008}'),
                        Some(b'f') => out.push('\u{000C}'),
                        Some(b'n') => out.push('\n'),
                        Some(b'r') => out.push('\r'),
                        Some(b't') => out.push('\t'),
                        Some(b'u') => {
                            let cp = self.parse_hex4()?;
                            // Surrogate pair handling: if cp is a high surrogate, expect \uXXXX low surrogate.
                            if (0xD800..=0xDBFF).contains(&cp) {
                                if self.advance() != Some(b'\\') || self.advance() != Some(b'u') {
                                    return Err(
                                        self.err("expected low surrogate after high surrogate")
                                    );
                                }
                                let lo = self.parse_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&lo) {
                                    return Err(self.err("invalid low surrogate"));
                                }
                                let codepoint = 0x10000
                                    + (((cp - 0xD800) as u32) << 10)
                                    + ((lo - 0xDC00) as u32);
                                if let Some(c) = char::from_u32(codepoint) {
                                    out.push(c);
                                } else {
                                    return Err(self.err("invalid surrogate pair"));
                                }
                            } else if (0xDC00..=0xDFFF).contains(&cp) {
                                return Err(self.err("unexpected low surrogate"));
                            } else if let Some(c) = char::from_u32(cp as u32) {
                                out.push(c);
                            } else {
                                return Err(self.err("invalid \\u codepoint"));
                            }
                        }
                        Some(b) => return Err(self.err(format!("invalid escape \\0x{b:02x}"))),
                    }
                }
                Some(b) if b < 0x20 => {
                    return Err(self.err(format!("unescaped control byte 0x{b:02x} in string")));
                }
                Some(b) => {
                    // Append byte as-is. We don't validate UTF-8 here; the
                    // decoded String may carry invalid UTF-8 only if the input
                    // contained invalid sequences, in which case we accept by
                    // appending the byte raw via push_str on a from_utf8_lossy
                    // — simpler: push the byte by treating as a char only if
                    // <=0x7F, else accumulate raw bytes via a Vec.
                    //
                    // Pragmatic: we only need decoded keys for object matching.
                    // Use push() for ASCII; for multi-byte UTF-8 collect bytes
                    // and rely on input being valid.
                    if b < 0x80 {
                        out.push(b as char);
                    } else {
                        // Multi-byte start: figure out length, copy raw bytes,
                        // and try to interpret. If decoding fails, we substitute
                        // the replacement char so matching still works.
                        let extra = if b & 0xE0 == 0xC0 {
                            1
                        } else if b & 0xF0 == 0xE0 {
                            2
                        } else if b & 0xF8 == 0xF0 {
                            3
                        } else {
                            return Err(self.err("invalid UTF-8 lead byte in string"));
                        };
                        let mut buf = vec![b];
                        for _ in 0..extra {
                            match self.advance() {
                                Some(c) if c & 0xC0 == 0x80 => buf.push(c),
                                _ => return Err(self.err("invalid UTF-8 continuation in string")),
                            }
                        }
                        match std::str::from_utf8(&buf) {
                            Ok(s) => out.push_str(s),
                            Err(_) => out.push(char::REPLACEMENT_CHARACTER),
                        }
                    }
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, ParseError> {
        let mut acc: u16 = 0;
        for _ in 0..4 {
            let b = self.advance().ok_or_else(|| ParseError {
                byte_offset: self.pos as u64,
                msg: "EOF in \\uXXXX".into(),
            })?;
            let d: u16 = match b {
                b'0'..=b'9' => (b - b'0') as u16,
                b'a'..=b'f' => (b - b'a' + 10) as u16,
                b'A'..=b'F' => (b - b'A' + 10) as u16,
                _ => return Err(self.err(format!("invalid hex digit 0x{b:02x} in \\uXXXX"))),
            };
            acc = (acc << 4) | d;
        }
        Ok(acc)
    }

    fn parse_number(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        // Integer part: 0 | [1-9][0-9]*
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
            }
            Some(b) if (b'1'..=b'9').contains(&b) => {
                self.pos += 1;
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
            }
            _ => return Err(self.err("invalid number: missing integer part")),
        }
        // Fraction part.
        if self.peek() == Some(b'.') {
            self.pos += 1;
            let frac_start = self.pos;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.pos == frac_start {
                return Err(self.err("invalid number: empty fraction"));
            }
        }
        // Exponent part.
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            let exp_start = self.pos;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.pos == exp_start {
                return Err(self.err("invalid number: empty exponent"));
            }
        }
        Ok(Node {
            kind: NodeKind::Number,
            byte_start: start as u64,
            byte_end: self.pos as u64,
            children: vec![],
        })
    }

    fn parse_bool(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        let rem = &self.src[self.pos..];
        if rem.starts_with(b"true") {
            self.pos += 4;
            Ok(Node {
                kind: NodeKind::Bool,
                byte_start: start as u64,
                byte_end: self.pos as u64,
                children: vec![],
            })
        } else if rem.starts_with(b"false") {
            self.pos += 5;
            Ok(Node {
                kind: NodeKind::Bool,
                byte_start: start as u64,
                byte_end: self.pos as u64,
                children: vec![],
            })
        } else {
            Err(self.err("invalid bool literal"))
        }
    }

    fn parse_null(&mut self) -> Result<Node, ParseError> {
        let start = self.pos;
        if self.src[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(Node {
                kind: NodeKind::Null,
                byte_start: start as u64,
                byte_end: self.pos as u64,
                children: vec![],
            })
        } else {
            Err(self.err("invalid null literal"))
        }
    }
}

/// Parse a complete JSON document. Strict RFC 8259 (no comments, no trailing
/// commas).
pub fn parse(src: &[u8]) -> Result<Document, ParseError> {
    let mut p = Parser::new(src);
    let leading = p.skip_ws();
    let root = p.parse_value()?;
    let trailing = p.skip_ws();
    if p.pos != src.len() {
        return Err(ParseError {
            byte_offset: p.pos as u64,
            msg: "trailing data after root value".into(),
        });
    }
    Ok(Document {
        leading_ws: leading,
        root,
        trailing_ws: trailing,
    })
}

/// Walk a Document's tree and emit every leaf byte range in document order.
/// Used by the round-trip test: the concatenation must equal the input.
#[cfg(test)]
pub fn collect_ranges(doc: &Document) -> Vec<Range<u64>> {
    let mut out = Vec::new();
    if !doc.leading_ws.is_empty() {
        out.push(doc.leading_ws.clone());
    }
    walk_node(&doc.root, &mut out);
    if !doc.trailing_ws.is_empty() {
        out.push(doc.trailing_ws.clone());
    }
    out
}

#[cfg(test)]
fn walk_node(n: &Node, out: &mut Vec<Range<u64>>) {
    if n.children.is_empty() {
        if n.byte_start != n.byte_end {
            out.push(n.range());
        }
        return;
    }
    // Container: emit '{' or '[', then walk children, then '}' or ']'.
    let open = n.byte_start..n.byte_start + 1;
    if !open.is_empty() && (n.kind == NodeKind::Object || n.kind == NodeKind::Array) {
        out.push(open);
    }
    for c in &n.children {
        match c {
            Child::Member {
                key_range,
                between_key_value,
                value,
                trailing,
                ..
            } => {
                if !key_range.is_empty() {
                    out.push(key_range.clone());
                }
                if !between_key_value.is_empty() {
                    out.push(between_key_value.clone());
                }
                walk_node(value, out);
                if !trailing.is_empty() {
                    out.push(trailing.clone());
                }
            }
            Child::Element { value, trailing } => {
                walk_node(value, out);
                if !trailing.is_empty() {
                    out.push(trailing.clone());
                }
            }
        }
    }
    let close = n.byte_end - 1..n.byte_end;
    if n.kind == NodeKind::Object || n.kind == NodeKind::Array {
        out.push(close);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(s: &str) {
        let doc = parse(s.as_bytes()).expect("parse failed");
        let ranges = collect_ranges(&doc);
        let mut concat = Vec::new();
        for r in &ranges {
            concat.extend_from_slice(&s.as_bytes()[r.start as usize..r.end as usize]);
        }
        assert_eq!(
            concat.as_slice(),
            s.as_bytes(),
            "round-trip mismatch for: {s:?}\nranges: {ranges:?}"
        );
    }

    #[test]
    fn empty_object() {
        rt("{}");
    }
    #[test]
    fn empty_array() {
        rt("[]");
    }
    #[test]
    fn primitive_number() {
        rt("42");
        rt("-1.5e10");
    }
    #[test]
    fn primitive_bool() {
        rt("true");
        rt("false");
    }
    #[test]
    fn primitive_null() {
        rt("null");
    }
    #[test]
    fn primitive_string() {
        rt("\"hello\"");
    }
    #[test]
    fn string_escapes() {
        rt(r#""a\"b""#);
        rt(r#""\\""#);
        rt(r#""é""#);
    }
    #[test]
    fn unicode_keys() {
        rt(r#"{"日":1}"#);
    }
    #[test]
    fn nested() {
        rt(r#"{"a":{"b":[1,2,3]}}"#);
    }
    #[test]
    fn whitespace_runs() {
        rt("{ \"a\"  :  1 ,\n  \"b\" : [\n  1,\n  2\n] }");
    }
    #[test]
    fn leading_and_trailing_ws() {
        rt("  {\"a\":1}\n");
    }
    #[test]
    fn error_position() {
        let e = parse(b"{\"a\":}").unwrap_err();
        // Error should be at the '}' position (byte 5).
        assert!(
            e.byte_offset >= 4 && e.byte_offset <= 6,
            "got offset {}",
            e.byte_offset
        );
    }
    #[test]
    fn error_trailing_data() {
        let e = parse(b"42 garbage").unwrap_err();
        assert!(e.msg.contains("trailing"));
    }

    #[test]
    fn key_decoding() {
        // "{\"aé\":1}" with é as UTF-8 0xC3 0xA9.
        let doc = parse(b"{\"a\xc3\xa9\":1}").unwrap();
        match &doc.root.children[0] {
            Child::Member { key_decoded, .. } => assert_eq!(key_decoded, "aé"),
            _ => panic!("expected Member"),
        }
    }

    #[test]
    fn round_trip_many() {
        let inputs = [
            "{}",
            "[]",
            "[1]",
            "[1,2]",
            "{\"a\":[1,{\"b\":null}]}",
            "[ 1 , 2 , 3 ]",
            "{ }",
            "[ ]",
            "\"\"",
            "0",
            "1.0",
            "1e10",
            "-0.5e-3",
        ];
        for s in inputs {
            rt(s);
        }
    }
}
