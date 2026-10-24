//! WHATWG HTML tokenizer (spec 13.2.5).
//!
//! Input preprocessing (CR/CRLF -> LF, BOM strip) happens in [`Tokenizer::new`].
//! The tree builder drives the tokenizer one token at a time and switches its
//! content-model state ([`Tokenizer::set_state`]) for RCDATA/RAWTEXT/script/
//! PLAINTEXT, and sets [`Tokenizer::cdata_allowed`] from the adjusted current
//! node before each token (CDATA sections only exist in foreign content).

use std::borrow::Cow;
use std::collections::VecDeque;

use crate::atom::{AttrName, TagName};
use crate::entities;
use crate::token::{DoctypeToken, Tag, Token};
use crate::tree::{ParseError, ParseErrorCode, SourceLoc};

/// Tokenizer states (WHATWG 13.2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum State {
    Data,
    Rcdata,
    Rawtext,
    ScriptData,
    Plaintext,
    TagOpen,
    EndTagOpen,
    TagName,
    RcdataLessThanSign,
    RcdataEndTagOpen,
    RcdataEndTagName,
    RawtextLessThanSign,
    RawtextEndTagOpen,
    RawtextEndTagName,
    ScriptDataLessThanSign,
    ScriptDataEndTagOpen,
    ScriptDataEndTagName,
    ScriptDataEscapeStart,
    ScriptDataEscapeStartDash,
    ScriptDataEscaped,
    ScriptDataEscapedDash,
    ScriptDataEscapedDashDash,
    ScriptDataEscapedLessThanSign,
    ScriptDataEscapedEndTagOpen,
    ScriptDataEscapedEndTagName,
    ScriptDataDoubleEscapeStart,
    ScriptDataDoubleEscaped,
    ScriptDataDoubleEscapedDash,
    ScriptDataDoubleEscapedDashDash,
    ScriptDataDoubleEscapedLessThanSign,
    ScriptDataDoubleEscapeEnd,
    BeforeAttributeName,
    AttributeName,
    AfterAttributeName,
    BeforeAttributeValue,
    AttributeValueDoubleQuoted,
    AttributeValueSingleQuoted,
    AttributeValueUnquoted,
    AfterAttributeValueQuoted,
    SelfClosingStartTag,
    BogusComment,
    PiOpen,
    PiTarget,
    PiAfterTarget,
    PiData,
    PiQuestionable,
    MarkupDeclarationOpen,
    CommentStart,
    CommentStartDash,
    Comment,
    CommentLessThanSign,
    CommentLessThanSignBang,
    CommentLessThanSignBangDash,
    CommentLessThanSignBangDashDash,
    CommentEndDash,
    CommentEnd,
    CommentEndBang,
    Doctype,
    BeforeDoctypeName,
    DoctypeName,
    AfterDoctypeName,
    AfterDoctypePublicKeyword,
    BeforeDoctypePublicIdentifier,
    DoctypePublicIdentifierDoubleQuoted,
    DoctypePublicIdentifierSingleQuoted,
    AfterDoctypePublicIdentifier,
    BetweenDoctypePublicAndSystemIdentifiers,
    AfterDoctypeSystemKeyword,
    BeforeDoctypeSystemIdentifier,
    DoctypeSystemIdentifierDoubleQuoted,
    DoctypeSystemIdentifierSingleQuoted,
    AfterDoctypeSystemIdentifier,
    BogusDoctype,
    CdataSection,
    CdataSectionBracket,
    CdataSectionEnd,
    CharacterReference,
    NamedCharacterReference,
    NumericCharacterReference,
    HexadecimalCharacterReferenceStart,
    DecimalCharacterReferenceStart,
    HexadecimalCharacterReference,
    DecimalCharacterReference,
    NumericCharacterReferenceEnd,
}

const REPLACEMENT: char = '\u{FFFD}';

/// Control-character reference remapping (spec table, Windows-1252).
fn map_control(n: u32) -> u32 {
    match n {
        0x80 => 0x20AC,
        0x82 => 0x201A,
        0x83 => 0x0192,
        0x84 => 0x201E,
        0x85 => 0x2026,
        0x86 => 0x2020,
        0x87 => 0x2021,
        0x88 => 0x02C6,
        0x89 => 0x2030,
        0x8A => 0x0160,
        0x8B => 0x2039,
        0x8C => 0x0152,
        0x8E => 0x017D,
        0x91 => 0x2018,
        0x92 => 0x2019,
        0x93 => 0x201C,
        0x94 => 0x201D,
        0x95 => 0x2022,
        0x96 => 0x2013,
        0x97 => 0x2014,
        0x98 => 0x02DC,
        0x99 => 0x2122,
        0x9A => 0x0161,
        0x9B => 0x203A,
        0x9C => 0x0153,
        0x9E => 0x017E,
        0x9F => 0x0178,
        _ => n,
    }
}

fn is_noncharacter(n: u32) -> bool {
    (0xFDD0..=0xFDEF).contains(&n) || n <= 0x10FFFF && (n & 0xFFFE == 0xFFFE)
}

/// Length of the longest named character reference table entry matching
/// `input[start..]`. The candidate is the ASCII alphanumeric run starting
/// at `start`, optionally extended by one following `;` (maximal munch per
/// WHATWG 13.2.5.73). Returns 0 when nothing matches.
///
/// This equals the old incremental scan (which stopped at the first byte
/// that was neither a table entry nor a prefix of one): any table entry
/// that is a prefix of the run has all of its own prefixes also be
/// entry-prefixes, so the incremental scan never stops before the longest
/// entry prefix, and it records exactly the longest entry it passes.
fn longest_entity_match(input: &str, start: usize) -> usize {
    let bytes = input.as_bytes();
    let n = bytes.len();
    let mut j = start;
    while j < n && bytes[j].is_ascii_alphanumeric() {
        j += 1;
    }
    let mut end = j;
    if j < n && bytes[j] == b';' {
        end = j + 1;
    }
    // Longest name in the table is 31 bytes; anything longer cannot match.
    const MAX_ENTITY_LEN: usize = 32;
    let mut k = (end - start).min(MAX_ENTITY_LEN);
    while k > 0 {
        if entities::lookup(&input[start..start + k]).is_some() {
            return k;
        }
        k -= 1;
    }
    0
}

/// Longest named character reference match at the current read cursor.
/// The match region (ASCII alphanumerics plus an optional `;`) may span
/// `document.write` insertion segments, so up to 33 bytes are materialized
/// — [`longest_entity_match`] only inspects the first 32 — and the
/// single-segment matcher is reused on them.
fn longest_entity_match_at(segs: &[InputSeg<'_>], head: usize, pos: usize) -> usize {
    let mut buf = [0u8; 33];
    let mut len = 0usize;
    let mut seg = head;
    let mut pos = pos;
    // Alphanumeric run.
    while len < buf.len() {
        let bytes = segs[seg].buf.as_bytes();
        while pos < bytes.len() && len < buf.len() {
            let b = bytes[pos];
            if !b.is_ascii_alphanumeric() {
                break;
            }
            buf[len] = b;
            len += 1;
            pos += 1;
        }
        if pos < bytes.len() || seg == 0 || len >= buf.len() {
            break;
        }
        seg -= 1;
        pos = segs[seg].off;
    }
    // Optional `;` immediately after the run.
    let (mut sseg, mut spos) = (seg, pos);
    if spos >= segs[sseg].buf.len() && sseg > 0 {
        sseg -= 1;
        spos = segs[sseg].off;
    }
    if len < buf.len()
        && let Some(b';') = segs[sseg].buf.as_bytes().get(spos).copied()
    {
        buf[len] = b';';
        len += 1;
    }
    // The buffer holds ASCII by construction.
    let s = std::str::from_utf8(&buf[..len]).expect("entity scan is ASCII-only");
    longest_entity_match(s, 0)
}
/// Accumulate one digit into a numeric character reference code without
/// ever overflowing: any value that exceeds the maximum Unicode scalar
/// value saturates at `0x110000`, which the numeric-character-reference-end
/// step then maps to U+FFFD (a parse error per the spec).
fn add_char_ref_digit(code: u32, digit: u32, base: u32) -> u32 {
    match code.checked_mul(base).and_then(|c| c.checked_add(digit)) {
        Some(v) if v <= 0x10FFFF => v,
        _ => 0x110000,
    }
}

/// A processing instruction token under construction (WHATWG 13.2.5.72+).
/// The target accumulates in the temporary buffer per the spec; it is
/// moved here once the target is complete and data collection begins.
#[derive(Debug, Clone, Default)]
struct PiToken {
    target: String,
    data: String,
}

/// What stopped a [`Tokenizer::char_run`]: a stop byte (not consumed) or
/// end of input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunEnd {
    Stop(u8),
    Eof,
}

/// Advance `line`/`col` over an ASCII-only chunk (no newlines counted as
/// anything but `\n`; one column per byte since ASCII bytes are chars).
fn advance_ascii(line: &mut u32, col: &mut u32, chunk: &[u8]) {
    let mut c = *col;
    for &b in chunk {
        if b == b'\n' {
            *line += 1;
            c = 1;
        } else {
            c += 1;
        }
    }
    *col = c;
}

/// A byte position in the segmented input stream: offset `off` into
/// `segs[seg]`. Always on a char boundary. Snapshots of the read cursor
/// serve as range markers (character references, tag-name scans); they
/// stay valid for the whole token scan because segments are never removed
/// or reordered — [`Tokenizer::insert_input`] only pushes new ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cursor {
    seg: usize,
    off: usize,
}

/// One chunk of the input stream. `off` is the byte offset at which the
/// cursor entered this segment: 0 for fresh segments, the parked read
/// position for a segment resumed after an insertion was consumed. It is
/// written when the segment is pushed or parked and never mutated
/// afterwards, so range slices can recover each segment's lower bound.
struct InputSeg<'a> {
    buf: Cow<'a, str>,
    off: usize,
}

pub(crate) struct Tokenizer<'a> {
    /// Input stream segments in reverse read order: the back is the read
    /// head. Index 0 is the preprocessed base input (one leading BOM
    /// stripped, CR/CRLF normalized to LF), borrowed from the caller in
    /// the common case; [`Tokenizer::insert_input`] pushes owned
    /// `document.write` insertion segments here, so inserting text never
    /// copies or moves the base input. N small writes cost O(total
    /// inserted bytes), not O(total input) each.
    ///
    /// Consumed segments are retained (never removed or reordered): a
    /// [`Cursor`] recorded mid-scan keeps denoting the same bytes, and
    /// [`Tokenizer::unconsume`] can step back into the previous segment.
    /// Retained memory is bounded by the total `document.write` payload.
    segs: Vec<InputSeg<'a>>,
    /// Index into [`Tokenizer::segs`] of the segment currently being read;
    /// decreases as segments are exhausted.
    head: usize,
    /// Byte offset of the next unconsumed character in `segs[head]`,
    /// always on a char boundary. The invariant is `pos < len` whenever
    /// input remains; only `(head, pos) == (0, len)` denotes EOF.
    pos: usize,
    state: State,
    return_state: State,
    current_tag: Option<Tag>,
    tag_is_start: bool,
    /// True while the attribute being built is a duplicate (value discarded).
    current_attr_dropped: bool,
    /// Scratch buffer for the tag/attribute name under construction.
    /// Tag-name and attribute-name states push lowercased characters here
    /// instead of allocating a `String` per name; the completed name is
    /// interned into the tag/attribute slot ([`emit_tag`],
    /// [`finish_attribute_name`]) and the buffer cleared (retaining
    /// capacity) for the next name. Well-known names therefore cost zero
    /// allocation; unknown names keep one allocation, same as before.
    name_buf: String,
    current_comment: String,
    current_doctype: DoctypeToken,
    /// Processing instruction token under construction (WHATWG 13.2.5.72+).
    current_pi: Option<PiToken>,
    temp_buf: String,
    char_ref_code: u32,
    /// Whether the current numeric character reference consumed its
    /// terminating `;` (drives `missing-semicolon-after-character-reference`).
    char_ref_had_semi: bool,
    last_start_tag: TagName,
    /// Set by the tree builder before each token: false while the adjusted
    /// current node is in the HTML namespace.
    pub cdata_allowed: bool,
    /// Pending tokens with their source locations, popped in lockstep by
    /// [`Tokenizer::next_token`].
    queue: VecDeque<(Token, SourceLoc)>,
    /// Pending character data for the current text run. The character
    /// states ([`Tokenizer::char_run`]), character-reference flushes, and
    /// the `<`-led fallback emitters all append here instead of emitting
    /// one token per character; [`Tokenizer::next_token`] drains the
    /// whole run as a single [`Token::Text`] before the main queue.
    /// Anything queued afterwards is newer, so draining first preserves
    /// token order (a run is always older than queued tokens: `step` only
    /// runs when both are empty).
    text: String,
    /// Source location of the first character in [`Tokenizer::text`]: a
    /// text token keeps its first character's location.
    text_start: SourceLoc,
    eof_emitted: bool,
    /// 1-based line number of the next character to consume.
    line: u32,
    /// 1-based column (in chars, not bytes) of the next character to
    /// consume.
    col: u32,
    /// Position of the most recently consumed character.
    last_char_loc: SourceLoc,
    /// Position of the `<` that began the current tag/comment/doctype
    /// token. Only set at the `<` arms of the text states; a `<` consumed
    /// anywhere else (comment data, double-escaped script) never begins a
    /// token.
    token_start: SourceLoc,
    /// Position of the `&` that began the current character reference.
    entity_start: SourceLoc,
    /// Cursor of the `&` that began the current character reference,
    /// for flushing the literal source text on parse errors.
    amp_pos: Cursor,
    /// Location of the token most recently returned by `next_token`.
    token_loc: SourceLoc,
    /// Tokenizer parse errors (WHATWG 13.2.2), drained by the tree builder
    /// after each token via [`Tokenizer::take_errors`].
    errors: Vec<ParseError>,
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        // Input stream preprocessing: strip one leading BOM, CR/CRLF -> LF.
        // The common case (no CR) borrows the caller's buffer: no copy.
        let stripped = input.strip_prefix('\u{FEFF}').unwrap_or(input);
        let input = if stripped.contains('\r') {
            Cow::Owned(stripped.replace("\r\n", "\n").replace('\r', "\n"))
        } else {
            Cow::Borrowed(stripped)
        };
        Tokenizer {
            segs: vec![InputSeg { buf: input, off: 0 }],
            head: 0,
            pos: 0,
            state: State::Data,
            return_state: State::Data,
            current_tag: None,
            tag_is_start: true,
            current_attr_dropped: false,
            name_buf: String::new(),
            current_comment: String::new(),
            current_doctype: DoctypeToken::default(),
            current_pi: None,
            temp_buf: String::new(),
            char_ref_code: 0,
            char_ref_had_semi: false,
            // Empty unknown name: matches no end tag (the scratch name is
            // never empty in the end-tag-name states, which always consume
            // at least one character).
            last_start_tag: TagName::from(""),
            cdata_allowed: false,
            queue: VecDeque::new(),
            text: String::new(),
            text_start: SourceLoc { line: 1, col: 1 },
            eof_emitted: false,
            line: 1,
            col: 1,
            last_char_loc: SourceLoc { line: 1, col: 1 },
            token_start: SourceLoc { line: 1, col: 1 },
            entity_start: SourceLoc { line: 1, col: 1 },
            amp_pos: Cursor { seg: 0, off: 0 },
            token_loc: SourceLoc { line: 1, col: 1 },
            errors: Vec::new(),
        }
    }

    /// WHATWG `document.write` insertion point: push `text` as a new
    /// segment read before the next unconsumed character, so the
    /// tokenizer picks it up before the original remaining input.
    ///
    /// CR/CRLF are normalized to LF exactly like input preprocessing
    /// (13.2.3.1). This is O(inserted text): the base input is never
    /// copied or moved, so thousands of small writes stay linear instead
    /// of quadratic. Nested writes nest naturally: each insertion lands
    /// before everything still unread, including earlier insertions'
    /// remainders.
    ///
    /// Segments above `head` are already fully consumed (the cursor only
    /// moves forward), so they are dropped first: otherwise a later
    /// insertion would descend into and replay them.
    pub(crate) fn insert_input(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.segs.truncate(self.head + 1);
        let normalized: String = if text.contains('\r') {
            text.replace("\r\n", "\n").replace('\r', "\n")
        } else {
            text.to_owned()
        };
        // Park the current segment's read position as its re-entry offset,
        // then push the new text in front of everything unread: it is
        // consumed before the current segment's remainder and every later
        // segment, exactly "just before the next input character".
        self.segs[self.head].off = self.pos;
        self.segs.push(InputSeg {
            buf: Cow::Owned(normalized),
            off: 0,
        });
        self.head = self.segs.len() - 1;
        self.pos = 0;
    }

    /// The current read cursor.
    fn cursor(&self) -> Cursor {
        Cursor {
            seg: self.head,
            off: self.pos,
        }
    }

    /// Move the read cursor, restoring the read invariant (a cursor at a
    /// segment end denotes the next segment's entry).
    fn set_cursor(&mut self, c: Cursor) {
        self.head = c.seg;
        self.pos = c.off;
        self.normalize_cursor();
    }

    /// Restore the read invariant after the cursor advanced: it never sits
    /// at a segment end while a later segment remains.
    fn normalize_cursor(&mut self) {
        while self.head > 0 && self.pos >= self.segs[self.head].buf.len() {
            self.head -= 1;
            self.pos = self.segs[self.head].off;
        }
    }

    /// Advance `c` by `n` bytes, restoring the read invariant. `n` must
    /// stay within the input available from `c` (callers only advance over
    /// bytes they already observed, e.g. entity names).
    fn advance_cursor(&self, mut c: Cursor, mut n: usize) -> Cursor {
        while n > 0 {
            let len = self.segs[c.seg].buf.len();
            if c.off >= len {
                debug_assert!(c.seg > 0, "advance_cursor ran past EOF");
                c.seg -= 1;
                c.off = self.segs[c.seg].off;
                continue;
            }
            let take = n.min(len - c.off);
            c.off += take;
            n -= take;
        }
        while c.seg > 0 && c.off >= self.segs[c.seg].buf.len() {
            c.seg -= 1;
            c.off = self.segs[c.seg].off;
        }
        c
    }

    /// The byte at `c`, or `None` at EOF.
    fn byte_at(&self, c: Cursor) -> Option<u8> {
        self.segs[c.seg].buf.as_bytes().get(c.off).copied()
    }

    /// Feed each `&str` piece of the input range `[from, to)` to `f`, in
    /// order. Ranges never run backwards; segment entry offsets (see
    /// [`InputSeg::off`]) recover each segment's lower bound, so ranges
    /// spanning `document.write` insertions come out whole.
    fn with_range_pieces(segs: &[InputSeg<'_>], from: Cursor, to: Cursor, mut f: impl FnMut(&str)) {
        debug_assert!(from.seg >= to.seg, "input range runs backwards");
        if from.seg == to.seg {
            f(&segs[from.seg].buf[from.off..to.off]);
            return;
        }
        f(&segs[from.seg].buf[from.off..]);
        for i in (to.seg + 1..from.seg).rev() {
            let s = &segs[i];
            f(&s.buf[s.off..]);
        }
        let t = &segs[to.seg];
        f(&t.buf[t.off..to.off]);
    }

    /// Append the input range `[from, to)` to `out`.
    fn push_range(segs: &[InputSeg<'_>], from: Cursor, to: Cursor, out: &mut String) {
        Self::with_range_pieces(segs, from, to, |piece| out.push_str(piece));
    }

    /// Switch content-model state (RCDATA / RAWTEXT / script data / PLAINTEXT).
    pub fn set_state(&mut self, state: State) {
        self.state = state;
    }

    /// Seed the "last start tag" name (WHATWG "appropriate end tag" hook).
    ///
    /// `#[doc(hidden)]` test support for the html5lib tokenizer
    /// conformance harness (`crates/bowser_html/tests/tokenizer.rs`),
    /// which drives the tokenizer with the fixture's `lastStartTag`.
    /// Not public API: the struct itself is `pub(crate)`.
    #[doc(hidden)]
    pub fn set_last_start_tag(&mut self, tag: &str) {
        self.last_start_tag = TagName::from(tag);
    }

    fn peek_byte(&self) -> Option<u8> {
        self.segs[self.head].buf.as_bytes().get(self.pos).copied()
    }

    fn peek(&self) -> Option<char> {
        let b = self.peek_byte()?;
        if b < 0x80 {
            Some(b as char)
        } else {
            // `pos` is always on a char boundary, the input is valid
            // UTF-8, and characters never span segments (insertions land
            // on char boundaries).
            self.segs[self.head].buf[self.pos..].chars().next()
        }
    }

    fn consume(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.advance_pos(c);
        // The character lies wholly inside the current segment (see
        // `peek`); only a segment end needs normalizing afterwards.
        self.pos += c.len_utf8();
        self.normalize_cursor();
        Some(c)
    }

    /// Record the consumption of `c` at the current position for
    /// line/column tracking. Input preprocessing already normalized
    /// CR/CRLF to LF, so only '\n' advances the line.
    fn advance_pos(&mut self, c: char) {
        // 1-based line/column, counting characters (not bytes).
        self.last_char_loc = SourceLoc {
            line: self.line,
            col: self.col,
        };
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
    }

    /// Skip `n` already-matched ASCII input bytes (after `peek_literal`),
    /// keeping line/column tracking in sync with `consume()`. May span
    /// segments.
    fn skip(&mut self, n: usize) {
        for _ in 0..n {
            let c = self.segs[self.head].buf.as_bytes()[self.pos] as char;
            self.advance_pos(c);
            self.pos += 1;
            self.normalize_cursor();
        }
    }

    /// Undo the most recent `consume()`: step back one character and
    /// restore the line/column tracking to that character's position.
    /// Only valid immediately after the `consume()` being undone. May
    /// step back into the previous segment, which is retained for exactly
    /// this (see [`Tokenizer::segs`]).
    fn unconsume(&mut self) {
        if self.pos > self.segs[self.head].off {
            // Step back over UTF-8 continuation bytes to the character start.
            let bytes = self.segs[self.head].buf.as_bytes();
            let mut p = self.pos - 1;
            while bytes[p] & 0xC0 == 0x80 {
                p -= 1;
            }
            self.pos = p;
        } else {
            // The undone `consume()` took the final character of the
            // previous segment: it was fully consumed, so the character
            // ends at its buffer end.
            let prev = self.head + 1;
            let bytes = self.segs[prev].buf.as_bytes();
            let mut p = bytes.len() - 1;
            while bytes[p] & 0xC0 == 0x80 {
                p -= 1;
            }
            self.head = prev;
            self.pos = p;
        }
        self.line = self.last_char_loc.line;
        self.col = self.last_char_loc.col;
    }

    /// Match `want` at the current position without consuming; may span
    /// segments.
    fn peek_bytes(&self, want: &[u8], ascii_case_insensitive: bool) -> bool {
        let mut seg = self.head;
        let mut pos = self.pos;
        for &w in want {
            loop {
                let bytes = self.segs[seg].buf.as_bytes();
                if pos < bytes.len() {
                    let eq = if ascii_case_insensitive {
                        bytes[pos].eq_ignore_ascii_case(&w)
                    } else {
                        bytes[pos] == w
                    };
                    if !eq {
                        return false;
                    }
                    pos += 1;
                    break;
                }
                if seg == 0 {
                    return false;
                }
                seg -= 1;
                pos = self.segs[seg].off;
            }
        }
        true
    }

    /// Case-insensitive match of the ASCII string `s` at the current
    /// position, no consumption.
    fn peek_literal(&self, s: &str) -> bool {
        self.peek_bytes(s.as_bytes(), true)
    }

    /// Case-sensitive `[CDATA[` prefix check. Unlike `DOCTYPE`/`PUBLIC`/
    /// `SYSTEM` (matched ASCII case-insensitively per WHATWG), the CDATA
    /// open sequence matches case-sensitively.
    fn peek_cdata(&self) -> bool {
        self.peek_bytes(b"[CDATA[", false)
    }

    /// Record a tokenizer parse error (WHATWG 13.2.2) at the most recently
    /// consumed character's location. Call after consuming the offending
    /// character so the location points at it.
    fn tokenizer_error(&mut self, code: ParseErrorCode) {
        self.errors.push(ParseError {
            code,
            loc: Some(self.last_char_loc),
        });
    }

    /// Drain recorded tokenizer errors. The tree builder calls this after
    /// each token so tokenizer errors land in source order relative to
    /// tree-construction errors.
    pub(crate) fn take_errors(&mut self) -> Vec<ParseError> {
        std::mem::take(&mut self.errors)
    }

    fn emit(&mut self, tok: Token) {
        let loc = SourceLoc {
            line: self.line,
            col: self.col,
        };
        self.queue.push_back((tok, loc));
    }

    fn emit_char(&mut self, c: char) {
        // The common case: the character was just consumed, so its position
        // is `last_char_loc`. Callers that re-emit an earlier `<` use
        // `emit_from_token_start` instead. Appends to the pending text run;
        // `next_token` emits the run as one `Token::Text`.
        let loc = self.last_char_loc;
        if self.text.is_empty() {
            self.text_start = loc;
        }
        self.text.push(c);
    }

    /// Emit a `<`-led fallback sequence (`</`, `<!`, ...) whose characters
    /// are adjacent in the source starting at `token_start`: each char
    /// advances one column (no line break can intervene between adjacent
    /// input characters that are not newlines). Appends to the pending
    /// text run; only the run's start location is retained.
    fn emit_from_token_start(&mut self, s: &str) {
        if self.text.is_empty() {
            self.text_start = self.token_start;
        }
        self.text.push_str(s);
    }

    /// Append `s` to the pending text run, stamping the run's start from
    /// `start` when the run was empty.
    fn push_text(&mut self, s: &str, start: SourceLoc) {
        if s.is_empty() {
            return;
        }
        if self.text.is_empty() {
            self.text_start = start;
        }
        self.text.push_str(s);
    }

    fn emit_tag(&mut self) {
        let mut tag = self.current_tag.take().expect("no current tag");
        // The tag name was interned into `tag.name` when the tag-name
        // state completed (`finish_tag_name`); attribute parsing reuses
        // the scratch buffer afterwards.
        let is_start = self.tag_is_start;
        if is_start {
            self.last_start_tag = tag.name.clone();
        } else {
            tag.attrs.clear();
            tag.self_closing = false;
        }
        self.queue.push_back((
            if is_start {
                Token::StartTag(tag)
            } else {
                Token::EndTag(tag)
            },
            self.token_start,
        ));
    }

    fn emit_comment(&mut self) {
        let data = std::mem::take(&mut self.current_comment);
        self.queue
            .push_back((Token::Comment(data), self.token_start));
    }

    /// "Convert the temporary buffer to a comment" (WHATWG 13.2.5.72+):
    /// the comment's data is `?` followed by `buf` (the temporary
    /// buffer's contents). Reconsume the current character in the bogus
    /// comment state (the state is switched without consuming).
    fn convert_pi_to_comment(&mut self, buf: String) {
        let mut data = String::with_capacity(buf.len() + 1);
        data.push('?');
        data.push_str(&buf);
        self.current_comment = data;
        self.state = State::BogusComment;
    }

    fn emit_pi(&mut self) {
        let pi = self
            .current_pi
            .take()
            .expect("no current processing instruction");
        self.queue.push_back((
            Token::ProcessingInstruction {
                target: pi.target,
                data: pi.data,
            },
            self.token_start,
        ));
    }

    fn pi_data_mut(&mut self) -> &mut String {
        &mut self
            .current_pi
            .as_mut()
            .expect("no current processing instruction")
            .data
    }

    fn emit_doctype(&mut self) {
        let d = std::mem::take(&mut self.current_doctype);
        if d.force_quirks {
            // Every quirks-forcing path through the DOCTYPE states is a
            // parse error (the WHATWG `missing-doctype-name` /
            // `abrupt-doctype-public-identifier` /
            // `abrupt-doctype-system-identifier` / `eof-in-doctype` family,
            // collapsed into one code).
            self.errors.push(ParseError {
                code: ParseErrorCode::InvalidDoctype,
                loc: Some(self.token_start),
            });
        }
        self.queue.push_back((Token::Doctype(d), self.token_start));
    }

    fn start_tag(&mut self, is_start: bool) {
        self.name_buf.clear();
        self.current_tag = Some(Tag {
            name: TagName::default(),
            attrs: Vec::new(),
            self_closing: false,
        });
        self.tag_is_start = is_start;
        self.current_attr_dropped = false;
    }

    fn tag_mut(&mut self) -> &mut Tag {
        self.current_tag.as_mut().expect("no current tag")
    }

    /// Called when the tag name is complete (leaving the tag-name
    /// states): intern the scratch buffer into the current tag's name and
    /// release the scratch for attribute names. Well-known names become
    /// pure atoms (zero allocation, scratch keeps its capacity); unknown
    /// names copy once onto the name's sidecar, the same single
    /// allocation as the old per-tag `String`.
    fn finish_tag_name(&mut self) {
        let name = TagName::intern(self.name_buf.as_str());
        self.name_buf.clear();
        self.tag_mut().name = name;
    }

    fn start_attribute(&mut self, name: &str) {
        // The name accumulates in the scratch buffer (lowercased by the
        // caller states); the attribute entry is pushed by
        // `finish_attribute_name` once the name is complete and
        // de-duplicated.
        self.name_buf.clear();
        self.name_buf.push_str(name);
        self.current_attr_dropped = false;
    }

    fn attr_name_push(&mut self, c: char) {
        self.name_buf.push(c);
    }

    fn attr_value_push(&mut self, c: char) {
        if self.current_attr_dropped {
            return;
        }
        if let Some((_, value)) = self.tag_mut().attrs.last_mut() {
            value.push(c);
        }
    }

    /// Called when an attribute name is complete: intern it out of the
    /// scratch buffer and drop it if duplicated.
    fn finish_attribute_name(&mut self) {
        let name = AttrName::intern(self.name_buf.as_str());
        self.name_buf.clear();
        let tag = self.tag_mut();
        // `Name` equality is exact (atoms for well-known names, sidecar
        // strings for the rest), so this is the same duplicate check as
        // before, now on interned names.
        if tag.attrs.iter().any(|(n, _)| n == &name) {
            self.current_attr_dropped = true;
        } else {
            tag.attrs.push((name, String::new()));
        }
    }

    /// "Flush code points consumed as a character reference": emit the
    /// literal source text `s` as character data, or append it to the
    /// current attribute value when the reference appeared inside one.
    /// Non-attribute characters join the pending text run (stamped from
    /// `entity_start`, the `&` position), matching the old temporary-buffer
    /// behavior.
    fn flush_str(&mut self, s: &str) {
        let in_attr = matches!(
            self.return_state,
            State::AttributeValueDoubleQuoted
                | State::AttributeValueSingleQuoted
                | State::AttributeValueUnquoted
        );
        if in_attr {
            if !self.current_attr_dropped
                && let Some(tag) = self.current_tag.as_mut()
                && let Some((_, value)) = tag.attrs.last_mut()
            {
                value.push_str(s);
            }
        } else {
            let start = self.entity_start;
            self.push_text(s, start);
        }
    }

    /// Flush the literal source text of the current character reference
    /// (`&`, `&#`, `&#x`, `&#38`, ...), from the `amp_pos` cursor
    /// snapshot to the current read cursor, on the parse-error paths.
    /// The input is segmented (`segs`), so the range may span segment
    /// boundaries and is collected with `push_range` before flushing.
    /// The range is always ASCII: the numeric states only consume ASCII
    /// digits/`x`/uppercase-X before bailing, and the named scan only
    /// matches ASCII alphanumerics.
    fn flush_range(&mut self) {
        // Copy the byte range first: `flush_str` mutates `self` while the
        // range is read from `self.segs`.
        let start = self.amp_pos;
        let end = self.cursor();
        let mut text = String::new();
        Self::push_range(&self.segs, start, end, &mut text);
        self.flush_str(&text);
    }

    /// Append one [`Token::Text`] run of "plain" bytes: ASCII bytes not in
    /// `stops`. The byte range is pushed verbatim into the pending text
    /// run ([`Tokenizer::text`]) — no per-character decoding — while
    /// line/column tracking advances per character (columns count
    /// characters, not bytes). Stops before the first stop byte without
    /// consuming it, or at EOF. This is the hot path for text content: it
    /// replaces the per-character `step()` dispatch with a tight loop
    /// that emits no tokens at all until the run ends.
    fn char_run<const N: usize>(&mut self, stops: [u8; N]) -> RunEnd {
        let mut line = self.line;
        let mut col = self.col;
        let start_loc = SourceLoc { line, col };
        loop {
            self.normalize_cursor();
            let seg = self.head;
            let seg_start = self.pos;
            if seg_start >= self.segs[seg].buf.len() {
                debug_assert!(seg == 0, "normalize_cursor left a stale cursor");
                self.line = line;
                self.col = col;
                return RunEnd::Eof;
            }
            // Scan one segment. Disjoint field borrows: `segs` is read
            // while `text` / `text_start` are written directly.
            let segs = &self.segs;
            let bytes = segs[seg].buf.as_bytes();
            let mut pos = seg_start;
            let mut stop = None;
            while pos < bytes.len() {
                let b = bytes[pos];
                if b >= 0x80 {
                    // Non-ASCII: bytes are copied verbatim (the input is
                    // valid UTF-8 and `pos` stays on char boundaries, which
                    // never split across segments); only the length is
                    // needed for column accounting.
                    pos += if b < 0xE0 {
                        2
                    } else if b < 0xF0 {
                        3
                    } else {
                        4
                    };
                    col += 1;
                    continue;
                }
                if stops.contains(&b) {
                    stop = Some(b);
                    break;
                }
                pos += 1;
                if b == b'\n' {
                    line += 1;
                    col = 1;
                } else {
                    col += 1;
                }
            }
            // The `segs` borrow ends here (NLL).
            if pos > seg_start {
                let text = &mut self.text;
                if text.is_empty() {
                    self.text_start = start_loc;
                }
                // The run length is known: reserve once instead of
                // growing geometrically.
                text.reserve(pos - seg_start);
                text.push_str(&self.segs[seg].buf[seg_start..pos]);
            }
            self.pos = pos;
            if let Some(b) = stop {
                self.line = line;
                self.col = col;
                return RunEnd::Stop(b);
            }
            // Segment exhausted without hitting a stop byte: the loop head
            // normalizes into the next segment, or reports EOF.
        }
    }

    /// Fail-closed handler for a `char_run` stop byte that no match arm
    /// covers. `char_run` only ever stops on bytes in its stop set, so this
    /// is unreachable unless a future edit widens a stop set without adding
    /// an arm; consuming the byte as literal text keeps the tokenizer total
    /// instead of panicking. (Stop bytes are always ASCII: bytes >= 0x80
    /// are consumed as characters inside `char_run`.)
    #[cold]
    fn char_run_unexpected(&mut self, byte: u8) {
        // Fail-closed in all builds: consume the byte as literal text.
        self.consume();
        self.emit_char(byte as char);
    }

    /// Pop the pending text run as a single [`Token::Text`], if any.
    /// Returns `None` (leaving an empty buffer) once the run is drained.
    fn pop_text(&mut self) -> Option<(Token, SourceLoc)> {
        if self.text.is_empty() {
            return None;
        }
        // Move the run out; the buffer is left empty (no capacity is
        // retained: runs vary wildly in size, and holding on to a large
        // block's capacity would make every later small run pay for it).
        let run = std::mem::take(&mut self.text);
        let loc = self.text_start;
        Some((Token::Text(run), loc))
    }

    /// Cursor of the first byte at/after the read cursor for which
    /// `is_plain` is false, plus the number of bytes scanned. Only scans
    /// ASCII bytes: any byte >= 0x80 ends the run. Used for tag names,
    /// attribute names/values and comment data, where the run is appended
    /// to a buffer rather than emitted as tokens. May span segments.
    fn scan_while(&self, mut is_plain: impl FnMut(u8) -> bool) -> (Cursor, usize) {
        let mut seg = self.head;
        let mut pos = self.pos;
        let mut len = 0usize;
        loop {
            let bytes = self.segs[seg].buf.as_bytes();
            while pos < bytes.len() {
                let b = bytes[pos];
                if b >= 0x80 || !is_plain(b) {
                    return (Cursor { seg, off: pos }, len);
                }
                pos += 1;
                len += 1;
            }
            if seg == 0 {
                return (Cursor { seg, off: pos }, len);
            }
            seg -= 1;
            pos = self.segs[seg].off;
        }
    }

    /// Append the input range `[from, to)` to `out`, advancing `line`/`col`
    /// over it. For ASCII-only [`scan_while`](Self::scan_while) ranges: one
    /// column per byte, `\n` starting a new line.
    fn push_range_ascii(
        segs: &[InputSeg<'_>],
        from: Cursor,
        to: Cursor,
        line: &mut u32,
        col: &mut u32,
        out: &mut String,
    ) {
        let mut c = *col;
        Self::with_range_pieces(segs, from, to, |piece| {
            out.push_str(piece);
            advance_ascii(line, &mut c, piece.as_bytes());
        });
        *col = c;
    }

    fn is_appropriate_end_tag(&self) -> bool {
        // The in-progress end-tag name lives in the scratch buffer (see
        // `start_text_end_tag` / `end_tag_name_step`); both sides are
        // ASCII-lowercased, so this is an exact comparison.
        self.last_start_tag.as_str() == self.name_buf.as_str()
    }

    /// A tentative end tag that is not appropriate: emit a U+003C
    /// LESS-THAN SIGN character, a U+002F SOLIDUS character, and the
    /// temporary buffer's characters (in the order they were added, source
    /// case preserved) into the pending text run. Reconsume the current
    /// character in `text_state`.
    /// (WHATWG 13.2.5.11 RCDATA, 13.2.5.14 RAWTEXT, and the script data
    /// end tag name states, "Anything else".)
    fn emit_inappropriate_end_tag(&mut self, text_state: State) {
        self.current_tag = None;
        self.emit_from_token_start("</");
        // The buffered name characters sit in the source immediately
        // after `</`; only the run's start location is retained.
        let buf = std::mem::take(&mut self.temp_buf);
        let start = self.token_start;
        self.push_text(&buf, start);
        self.state = text_state;
    }

    /// Begin an end tag token from one of the RCDATA/RAWTEXT/script-data
    /// end tag open states: append the current input character to the
    /// temporary buffer (the spec's "ASCII alpha" entry), then move to
    /// the end tag name state.
    fn start_text_end_tag(&mut self, c: char, name_state: State) {
        self.start_tag(false);
        self.temp_buf.push(c);
        self.name_buf.push(c.to_ascii_lowercase());
        self.consume();
        self.state = name_state;
    }

    /// Shared RCDATA / RAWTEXT / script-data end tag name state.
    /// The temporary buffer accumulates the current input character
    /// (source case preserved); the tag name accumulates the lowercased
    /// character.
    fn end_tag_name_step(&mut self, text_state: State) {
        match self.peek() {
            Some(c) if c.is_ascii_uppercase() => {
                self.consume();
                self.temp_buf.push(c);
                self.name_buf.push(c.to_ascii_lowercase());
            }
            Some(c) if c.is_ascii_lowercase() => {
                self.consume();
                self.temp_buf.push(c);
                self.name_buf.push(c);
            }
            Some(c) if c.is_ascii_whitespace() => {
                if self.is_appropriate_end_tag() {
                    self.consume();
                    self.finish_tag_name();
                    self.state = State::BeforeAttributeName;
                } else {
                    self.emit_inappropriate_end_tag(text_state);
                }
            }
            Some('/') => {
                if self.is_appropriate_end_tag() {
                    self.consume();
                    self.finish_tag_name();
                    self.state = State::SelfClosingStartTag;
                } else {
                    self.emit_inappropriate_end_tag(text_state);
                }
            }
            Some('>') => {
                if self.is_appropriate_end_tag() {
                    self.consume();
                    self.finish_tag_name();
                    self.state = State::Data;
                    self.emit_tag();
                } else {
                    self.emit_inappropriate_end_tag(text_state);
                }
            }
            _ => {
                // EOF has no dedicated entry in the spec: it falls through
                // to "anything else", which emits `</` plus the buffered
                // name characters and reconsumes in the text state (which
                // then emits EOF).
                self.emit_inappropriate_end_tag(text_state);
            }
        }
    }

    /// Consume a character reference (named or numeric), flushing either the
    /// decoded characters or the literal source text on parse errors.
    fn character_reference_step(&mut self) {
        match self.state {
            State::CharacterReference => match self.peek_byte() {
                Some(b) if b.is_ascii_alphanumeric() => {
                    self.state = State::NamedCharacterReference;
                }
                Some(b'#') => {
                    self.consume();
                    self.state = State::NumericCharacterReference;
                }
                _ => {
                    self.flush_range();
                    self.state = self.return_state;
                }
            },
            State::NamedCharacterReference => {
                let start = self.cursor();
                let best = longest_entity_match_at(&self.segs, self.head, self.pos);
                let cps: Option<&[u32]> = if best > 0 {
                    let end = self.advance_cursor(start, best);
                    let mut name = String::with_capacity(best);
                    Self::push_range(&self.segs, start, end, &mut name);
                    entities::lookup(&name)
                } else {
                    None
                };
                let in_attr = matches!(
                    self.return_state,
                    State::AttributeValueDoubleQuoted
                        | State::AttributeValueSingleQuoted
                        | State::AttributeValueUnquoted
                );
                // Consume the matched name (ASCII: no line breaks possible).
                self.set_cursor(self.advance_cursor(start, best));
                self.col += best as u32;
                if best > 0 {
                    self.last_char_loc = SourceLoc {
                        line: self.line,
                        col: self.col - 1,
                    };
                }
                let followed_by_eq_or_alnum = match self.peek_byte() {
                    Some(b'=') => true,
                    Some(b) => b.is_ascii_alphanumeric(),
                    None => false,
                };
                let ends_with_semi =
                    best > 0 && self.byte_at(self.advance_cursor(start, best - 1)) == Some(b';');
                match cps {
                    Some(_) if in_attr && !ends_with_semi && followed_by_eq_or_alnum => {
                        // Ambiguous ampersand in attribute: literal `&` +
                        // the matched name. No parse error here.
                        self.flush_range();
                    }
                    Some(cps) => {
                        if !ends_with_semi {
                            self.tokenizer_error(
                                ParseErrorCode::MissingSemicolonAfterCharacterReference,
                            );
                        }
                        let mut decoded = String::new();
                        for &cp in cps {
                            decoded.push(char::from_u32(cp).unwrap_or(REPLACEMENT));
                        }
                        self.flush_str(&decoded);
                    }
                    None => {
                        // No named reference matched: the spec moves to the
                        // ambiguous ampersand state, which reports
                        // `unknown-named-character-reference` only when its
                        // next character is `;`. A bare `&` followed by
                        // anything else never reaches this state and stays
                        // error-free.
                        let mut j = start;
                        while self.byte_at(j).is_some_and(|b| b.is_ascii_alphanumeric()) {
                            j = self.advance_cursor(j, 1);
                        }
                        if self.byte_at(j) == Some(b';') {
                            self.tokenizer_error(ParseErrorCode::UnknownNamedCharacterReference);
                        }
                        self.flush_range();
                    }
                }
                self.state = self.return_state;
            }
            State::NumericCharacterReference => {
                let had_char = self.peek_byte().is_some();
                match self.consume() {
                    Some('x' | 'X') => {
                        self.char_ref_code = 0;
                        self.char_ref_had_semi = false;
                        self.state = State::HexadecimalCharacterReferenceStart;
                    }
                    _ => {
                        // Not x/X: reconsume in the decimal start state.
                        if had_char {
                            self.unconsume();
                        }
                        self.char_ref_code = 0;
                        self.char_ref_had_semi = false;
                        self.state = State::DecimalCharacterReferenceStart;
                    }
                }
            }
            _ => {}
        }
    }

    /// Handle one input character (or EOF) in the current state.
    fn step(&mut self) {
        match self.state {
            // ---- character reference states ----
            State::CharacterReference
            | State::NamedCharacterReference
            | State::NumericCharacterReference => self.character_reference_step(),

            State::HexadecimalCharacterReferenceStart => match self.peek_byte() {
                Some(b) if b.is_ascii_hexdigit() => {
                    self.state = State::HexadecimalCharacterReference;
                }
                _ => {
                    self.tokenizer_error(
                        ParseErrorCode::AbsenceOfDigitsInNumericCharacterReference,
                    );
                    self.flush_range();
                    self.state = self.return_state;
                }
            },

            State::DecimalCharacterReferenceStart => match self.peek_byte() {
                Some(b) if b.is_ascii_digit() => {
                    self.state = State::DecimalCharacterReference;
                }
                _ => {
                    self.tokenizer_error(
                        ParseErrorCode::AbsenceOfDigitsInNumericCharacterReference,
                    );
                    self.flush_range();
                    self.state = self.return_state;
                }
            },

            State::HexadecimalCharacterReference => match self.consume() {
                Some(c) if c.is_ascii_hexdigit() => {
                    let d = c.to_digit(16).unwrap_or(0);
                    self.char_ref_code = add_char_ref_digit(self.char_ref_code, d, 16);
                }
                Some(';') => {
                    self.char_ref_had_semi = true;
                    self.state = State::NumericCharacterReferenceEnd;
                }
                Some(_) => {
                    self.unconsume();
                    self.state = State::NumericCharacterReferenceEnd;
                }
                None => self.state = State::NumericCharacterReferenceEnd,
            },

            State::DecimalCharacterReference => match self.consume() {
                Some(c) if c.is_ascii_digit() => {
                    let d = c.to_digit(10).unwrap_or(0);
                    self.char_ref_code = add_char_ref_digit(self.char_ref_code, d, 10);
                }
                Some(';') => {
                    self.char_ref_had_semi = true;
                    self.state = State::NumericCharacterReferenceEnd;
                }
                Some(_) => {
                    self.unconsume();
                    self.state = State::NumericCharacterReferenceEnd;
                }
                None => self.state = State::NumericCharacterReferenceEnd,
            },

            State::NumericCharacterReferenceEnd => {
                let n = self.char_ref_code;
                if !self.char_ref_had_semi {
                    self.tokenizer_error(ParseErrorCode::MissingSemicolonAfterCharacterReference);
                }
                let mapped = if n == 0 {
                    self.tokenizer_error(ParseErrorCode::NullCharacterReference);
                    0xFFFD
                } else if n > 0x10FFFF {
                    self.tokenizer_error(ParseErrorCode::CharacterReferenceOutsideUnicodeRange);
                    0xFFFD
                } else if (0xD800..0xE000).contains(&n) {
                    self.tokenizer_error(ParseErrorCode::SurrogateCharacterReference);
                    0xFFFD
                } else if is_noncharacter(n) {
                    self.tokenizer_error(ParseErrorCode::NoncharacterCharacterReference);
                    n
                } else if n == 0x0D
                    || (0x01..=0x08).contains(&n)
                    || n == 0x0B
                    || (0x0E..=0x1F).contains(&n)
                    || (0x7F..=0x9F).contains(&n)
                {
                    self.tokenizer_error(ParseErrorCode::ControlCharacterReference);
                    map_control(n)
                } else {
                    n
                };
                let c = char::from_u32(mapped).unwrap_or(REPLACEMENT);
                let mut decoded = [0u8; 4];
                let s = c.encode_utf8(&mut decoded);
                self.flush_str(s);
                self.state = self.return_state;
            }

            // ---- data-ish states ----
            State::Data => match self.char_run(*b"<&\0") {
                RunEnd::Stop(b'<') => {
                    self.token_start = SourceLoc {
                        line: self.line,
                        col: self.col,
                    };
                    self.consume();
                    self.state = State::TagOpen;
                }
                RunEnd::Stop(b'&') => {
                    self.amp_pos = self.cursor();
                    self.entity_start = SourceLoc {
                        line: self.line,
                        col: self.col,
                    };
                    self.consume();
                    self.return_state = State::Data;
                    self.state = State::CharacterReference;
                }
                RunEnd::Stop(0) => {
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                    self.emit_char('\0');
                }
                RunEnd::Stop(b) => self.char_run_unexpected(b),
                RunEnd::Eof => self.emit(Token::Eof),
            },

            State::Rcdata => match self.char_run(*b"<&\0") {
                RunEnd::Stop(b'<') => {
                    self.token_start = SourceLoc {
                        line: self.line,
                        col: self.col,
                    };
                    self.consume();
                    self.state = State::RcdataLessThanSign;
                }
                RunEnd::Stop(b'&') => {
                    self.amp_pos = self.cursor();
                    self.entity_start = SourceLoc {
                        line: self.line,
                        col: self.col,
                    };
                    self.consume();
                    self.return_state = State::Rcdata;
                    self.state = State::CharacterReference;
                }
                RunEnd::Stop(0) => {
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                    self.emit_char(REPLACEMENT);
                }
                RunEnd::Stop(b) => self.char_run_unexpected(b),
                RunEnd::Eof => self.emit(Token::Eof),
            },

            State::Rawtext => match self.char_run(*b"<\0") {
                RunEnd::Stop(b'<') => {
                    self.token_start = SourceLoc {
                        line: self.line,
                        col: self.col,
                    };
                    self.consume();
                    self.state = State::RawtextLessThanSign;
                }
                RunEnd::Stop(0) => {
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                    self.emit_char(REPLACEMENT);
                }
                RunEnd::Stop(b) => self.char_run_unexpected(b),
                RunEnd::Eof => self.emit(Token::Eof),
            },

            State::ScriptData => match self.char_run(*b"<\0") {
                RunEnd::Stop(b'<') => {
                    self.token_start = SourceLoc {
                        line: self.line,
                        col: self.col,
                    };
                    self.consume();
                    self.state = State::ScriptDataLessThanSign;
                }
                RunEnd::Stop(0) => {
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                    self.emit_char(REPLACEMENT);
                }
                RunEnd::Stop(b) => self.char_run_unexpected(b),
                RunEnd::Eof => self.emit(Token::Eof),
            },

            State::Plaintext => match self.char_run(*b"\0") {
                RunEnd::Stop(0) => {
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                    self.emit_char(REPLACEMENT);
                }
                RunEnd::Stop(b) => self.char_run_unexpected(b),
                RunEnd::Eof => self.emit(Token::Eof),
            },

            // ---- tag states ----
            State::TagOpen => match self.peek() {
                Some('!') => {
                    self.consume();
                    self.state = State::MarkupDeclarationOpen;
                }
                Some('/') => {
                    self.consume();
                    self.state = State::EndTagOpen;
                }
                Some(c) if c.is_ascii_alphabetic() => {
                    self.start_tag(true);
                    self.state = State::TagName;
                }
                Some('?') => {
                    // Consumed: `?` opens a processing instruction; the PI
                    // states decide whether it degrades to a bogus comment.
                    self.consume();
                    self.temp_buf.clear();
                    self.state = State::PiOpen;
                }
                Some(_) => {
                    // `invalid-first-character-of-tag-name`: emit `<`,
                    // reconsume in data.
                    self.tokenizer_error(ParseErrorCode::InvalidFirstCharacterOfTagName);
                    self.emit_char('<');
                    self.state = State::Data;
                }
                None => {
                    // `eof-before-tag-name`.
                    self.tokenizer_error(ParseErrorCode::EofBeforeTagName);
                    self.emit_char('<');
                    self.emit(Token::Eof);
                }
            },

            State::EndTagOpen => match self.peek() {
                Some(c) if c.is_ascii_alphabetic() => {
                    self.start_tag(false);
                    self.state = State::TagName;
                }
                Some('>') => {
                    // `missing-end-tag-name`.
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::MissingEndTagName);
                    self.state = State::Data;
                }
                Some(_) => {
                    // `invalid-first-character-of-tag-name`: bogus comment.
                    self.tokenizer_error(ParseErrorCode::InvalidFirstCharacterOfTagName);
                    self.current_comment.clear();
                    self.state = State::BogusComment;
                }
                None => {
                    // `eof-before-tag-name`.
                    self.tokenizer_error(ParseErrorCode::EofBeforeTagName);
                    self.emit_from_token_start("</");
                    self.emit(Token::Eof);
                }
            },

            State::TagName => {
                // Bulk-append tag name characters that need no case folding:
                // anything except ASCII whitespace, `/`, `>`, NUL, ASCII
                // uppercase and non-ASCII (all handled below).
                let start = self.cursor();
                let (end, len) = self.scan_while(|b| {
                    !b.is_ascii_uppercase()
                        && b != 0
                        && b != b'/'
                        && b != b'>'
                        && !b.is_ascii_whitespace()
                });
                if len > 0 {
                    Self::push_range(&self.segs, start, end, &mut self.name_buf);
                    // No newlines or multibyte chars in the run: one column
                    // per byte.
                    self.col += len as u32;
                    self.set_cursor(end);
                }
                match self.peek_byte() {
                    Some(b) if b.is_ascii_whitespace() => {
                        self.consume();
                        self.finish_tag_name();
                        self.state = State::BeforeAttributeName;
                    }
                    Some(b'/') => {
                        self.consume();
                        self.finish_tag_name();
                        self.state = State::SelfClosingStartTag;
                    }
                    Some(b'>') => {
                        self.consume();
                        self.finish_tag_name();
                        self.state = State::Data;
                        self.emit_tag();
                    }
                    Some(b) if b.is_ascii_uppercase() => {
                        self.consume();
                        self.name_buf.push((b as char).to_ascii_lowercase());
                    }
                    Some(0) => {
                        self.consume();
                        self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                        self.name_buf.push(REPLACEMENT);
                    }
                    Some(_) => {
                        // Non-ASCII name character: append as-is.
                        let c = self.consume().unwrap();
                        self.name_buf.push(c);
                    }
                    None => {
                        self.tokenizer_error(ParseErrorCode::EofInTag);
                        self.emit(Token::Eof);
                    }
                }
            }

            State::RcdataLessThanSign => match self.peek() {
                Some('/') => {
                    self.consume();
                    self.temp_buf.clear();
                    self.state = State::RcdataEndTagOpen;
                }
                _ => {
                    self.emit_char('<');
                    if self.peek().is_none() {
                        self.emit(Token::Eof);
                    } else {
                        self.state = State::Rcdata;
                    }
                }
            },

            State::RcdataEndTagOpen => match self.peek() {
                Some(c) if c.is_ascii_alphabetic() => {
                    self.start_text_end_tag(c, State::RcdataEndTagName);
                }
                _ => {
                    if self.peek().is_none() {
                        self.emit_from_token_start("</");
                        self.emit(Token::Eof);
                    } else {
                        self.emit_from_token_start("</");
                        self.state = State::Rcdata;
                    }
                }
            },

            State::RcdataEndTagName => self.end_tag_name_step(State::Rcdata),

            State::RawtextLessThanSign => match self.peek() {
                Some('/') => {
                    self.consume();
                    self.temp_buf.clear();
                    self.state = State::RawtextEndTagOpen;
                }
                _ => {
                    self.emit_char('<');
                    if self.peek().is_none() {
                        self.emit(Token::Eof);
                    } else {
                        self.state = State::Rawtext;
                    }
                }
            },

            State::RawtextEndTagOpen => match self.peek() {
                Some(c) if c.is_ascii_alphabetic() => {
                    self.start_text_end_tag(c, State::RawtextEndTagName);
                }
                _ => {
                    if self.peek().is_none() {
                        self.emit_from_token_start("</");
                        self.emit(Token::Eof);
                    } else {
                        self.emit_from_token_start("</");
                        self.state = State::Rawtext;
                    }
                }
            },

            State::RawtextEndTagName => self.end_tag_name_step(State::Rawtext),

            State::ScriptDataLessThanSign => match self.peek() {
                Some('/') => {
                    self.consume();
                    self.temp_buf.clear();
                    self.state = State::ScriptDataEndTagOpen;
                }
                Some('!') => {
                    self.consume();
                    self.emit_from_token_start("<!");
                    self.state = State::ScriptDataEscapeStart;
                }
                _ => {
                    self.emit_char('<');
                    if self.peek().is_none() {
                        self.emit(Token::Eof);
                    } else {
                        self.state = State::ScriptData;
                    }
                }
            },

            State::ScriptDataEndTagOpen => match self.peek() {
                Some(c) if c.is_ascii_alphabetic() => {
                    self.start_text_end_tag(c, State::ScriptDataEndTagName);
                }
                _ => {
                    if self.peek().is_none() {
                        self.emit_from_token_start("</");
                        self.emit(Token::Eof);
                    } else {
                        self.emit_from_token_start("</");
                        self.state = State::ScriptData;
                    }
                }
            },

            State::ScriptDataEndTagName => self.end_tag_name_step(State::ScriptData),

            State::ScriptDataEscapeStart => match self.peek() {
                Some('-') => {
                    self.consume();
                    self.emit_char('-');
                    self.state = State::ScriptDataEscapeStartDash;
                }
                _ => {
                    if self.peek().is_none() {
                        self.emit(Token::Eof);
                    } else {
                        self.state = State::ScriptData;
                    }
                }
            },

            State::ScriptDataEscapeStartDash => match self.peek() {
                Some('-') => {
                    self.consume();
                    self.emit_char('-');
                    self.state = State::ScriptDataEscapedDashDash;
                }
                _ => {
                    if self.peek().is_none() {
                        self.emit(Token::Eof);
                    } else {
                        self.state = State::ScriptData;
                    }
                }
            },

            State::ScriptDataEscaped => match self.char_run(*b"-<\0") {
                RunEnd::Stop(b'-') => {
                    self.consume();
                    self.emit_char('-');
                    self.state = State::ScriptDataEscapedDash;
                }
                RunEnd::Stop(b'<') => {
                    self.consume();
                    self.state = State::ScriptDataEscapedLessThanSign;
                }
                RunEnd::Stop(0) => {
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                    self.emit_char(REPLACEMENT);
                }
                RunEnd::Stop(b) => self.char_run_unexpected(b),
                RunEnd::Eof => self.emit(Token::Eof),
            },

            // NOTE: the dash states below intentionally do NOT use char_run:
            // their "anything else" arm switches state (to escaped), so
            // every byte must be dispatched individually.
            State::ScriptDataEscapedDash => match self.peek() {
                Some('-') => {
                    self.consume();
                    self.emit_char('-');
                    self.state = State::ScriptDataEscapedDashDash;
                }
                Some('<') => {
                    self.consume();
                    self.state = State::ScriptDataEscapedLessThanSign;
                }
                Some('\0') => {
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                    self.emit_char(REPLACEMENT);
                    self.state = State::ScriptDataEscaped;
                }
                Some(c) => {
                    self.consume();
                    self.emit_char(c);
                    self.state = State::ScriptDataEscaped;
                }
                None => self.emit(Token::Eof),
            },

            State::ScriptDataEscapedDashDash => match self.peek() {
                Some('-') => {
                    self.consume();
                    self.emit_char('-');
                }
                Some('<') => {
                    self.consume();
                    self.state = State::ScriptDataEscapedLessThanSign;
                }
                Some('>') => {
                    self.consume();
                    self.emit_char('>');
                    self.state = State::ScriptData;
                }
                Some('\0') => {
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                    self.emit_char(REPLACEMENT);
                    self.state = State::ScriptDataEscaped;
                }
                Some(c) => {
                    self.consume();
                    self.emit_char(c);
                    self.state = State::ScriptDataEscaped;
                }
                None => self.emit(Token::Eof),
            },

            State::ScriptDataEscapedLessThanSign => match self.peek() {
                Some('/') => {
                    self.consume();
                    self.temp_buf.clear();
                    self.state = State::ScriptDataEscapedEndTagOpen;
                }
                Some(c) if c.is_ascii_alphabetic() => {
                    self.temp_buf.clear();
                    self.emit_char('<');
                    self.state = State::ScriptDataDoubleEscapeStart;
                }
                _ => {
                    self.emit_char('<');
                    if self.peek().is_none() {
                        self.emit(Token::Eof);
                    } else {
                        self.state = State::ScriptDataEscaped;
                    }
                }
            },

            State::ScriptDataEscapedEndTagOpen => match self.peek() {
                Some(c) if c.is_ascii_alphabetic() => {
                    self.start_text_end_tag(c, State::ScriptDataEscapedEndTagName);
                }
                _ => {
                    if self.peek().is_none() {
                        self.emit_from_token_start("</");
                        self.emit(Token::Eof);
                    } else {
                        self.emit_from_token_start("</");
                        self.state = State::ScriptDataEscaped;
                    }
                }
            },

            State::ScriptDataEscapedEndTagName => self.end_tag_name_step(State::ScriptDataEscaped),

            State::ScriptDataDoubleEscapeStart => match self.peek() {
                Some(c) if matches!(c, '\t' | '\n' | '\x0C' | ' ' | '/' | '>') => {
                    self.consume();
                    self.emit_char(c);
                    if self.temp_buf == "script" {
                        self.state = State::ScriptDataDoubleEscaped;
                    } else {
                        self.state = State::ScriptDataEscaped;
                    }
                }
                Some(c) if c.is_ascii_uppercase() => {
                    self.consume();
                    self.temp_buf.push(c.to_ascii_lowercase());
                    self.emit_char(c);
                }
                Some(c) if c.is_ascii_lowercase() => {
                    self.consume();
                    self.temp_buf.push(c);
                    self.emit_char(c);
                }
                _ => {
                    if self.peek().is_none() {
                        self.emit(Token::Eof);
                    } else {
                        self.state = State::ScriptDataEscaped;
                    }
                }
            },

            State::ScriptDataDoubleEscaped => match self.char_run(*b"-<\0") {
                RunEnd::Stop(b'-') => {
                    self.consume();
                    self.emit_char('-');
                    self.state = State::ScriptDataDoubleEscapedDash;
                }
                RunEnd::Stop(b'<') => {
                    self.consume();
                    self.emit_char('<');
                    self.state = State::ScriptDataDoubleEscapedLessThanSign;
                }
                RunEnd::Stop(0) => {
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                    self.emit_char(REPLACEMENT);
                }
                RunEnd::Stop(b) => self.char_run_unexpected(b),
                RunEnd::Eof => self.emit(Token::Eof),
            },

            State::ScriptDataDoubleEscapedDash => match self.peek() {
                Some('-') => {
                    self.consume();
                    self.emit_char('-');
                    self.state = State::ScriptDataDoubleEscapedDashDash;
                }
                Some('<') => {
                    self.consume();
                    self.emit_char('<');
                    self.state = State::ScriptDataDoubleEscapedLessThanSign;
                }
                Some('\0') => {
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                    self.emit_char(REPLACEMENT);
                    self.state = State::ScriptDataDoubleEscaped;
                }
                Some(c) => {
                    self.consume();
                    self.emit_char(c);
                    self.state = State::ScriptDataDoubleEscaped;
                }
                None => self.emit(Token::Eof),
            },

            State::ScriptDataDoubleEscapedDashDash => match self.peek() {
                Some('-') => {
                    self.consume();
                    self.emit_char('-');
                }
                Some('<') => {
                    self.consume();
                    self.emit_char('<');
                    self.state = State::ScriptDataDoubleEscapedLessThanSign;
                }
                Some('>') => {
                    self.consume();
                    self.emit_char('>');
                    self.state = State::ScriptData;
                }
                Some('\0') => {
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                    self.emit_char(REPLACEMENT);
                    self.state = State::ScriptDataDoubleEscaped;
                }
                Some(c) => {
                    self.consume();
                    self.emit_char(c);
                    self.state = State::ScriptDataDoubleEscaped;
                }
                None => self.emit(Token::Eof),
            },

            State::ScriptDataDoubleEscapedLessThanSign => match self.peek() {
                Some('/') => {
                    self.consume();
                    self.emit_char('/');
                    self.temp_buf.clear();
                    self.state = State::ScriptDataDoubleEscapeEnd;
                }
                _ => {
                    if self.peek().is_none() {
                        self.emit(Token::Eof);
                    } else {
                        self.state = State::ScriptDataDoubleEscaped;
                    }
                }
            },

            State::ScriptDataDoubleEscapeEnd => match self.peek() {
                Some(c) if matches!(c, '\t' | '\n' | '\x0C' | ' ' | '/' | '>') => {
                    self.consume();
                    self.emit_char(c);
                    if self.temp_buf == "script" {
                        self.state = State::ScriptDataEscaped;
                    } else {
                        self.state = State::ScriptDataDoubleEscaped;
                    }
                }
                Some(c) if c.is_ascii_uppercase() => {
                    self.consume();
                    self.temp_buf.push(c.to_ascii_lowercase());
                    self.emit_char(c);
                }
                Some(c) if c.is_ascii_lowercase() => {
                    self.consume();
                    self.temp_buf.push(c);
                    self.emit_char(c);
                }
                _ => {
                    if self.peek().is_none() {
                        self.emit(Token::Eof);
                    } else {
                        self.state = State::ScriptDataDoubleEscaped;
                    }
                }
            },

            // ---- attribute states ----
            State::BeforeAttributeName => match self.peek_byte() {
                Some(b) if b.is_ascii_whitespace() => {
                    self.consume();
                }
                Some(b'/') | Some(b'>') => {
                    self.state = State::AfterAttributeName;
                }
                None => {
                    // `eof-in-tag`.
                    self.tokenizer_error(ParseErrorCode::EofInTag);
                    self.emit(Token::Eof);
                }
                Some(b'=') => {
                    // `unexpected-equals-sign-before-attribute-name`.
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::UnexpectedEqualsSignBeforeAttributeName);
                    self.start_attribute("=");
                    self.state = State::AttributeName;
                }
                Some(_) => {
                    // Start the attribute and let the attribute name state
                    // consume (and bulk-scan) its first character.
                    self.start_attribute("");
                    self.state = State::AttributeName;
                }
            },

            State::AttributeName => {
                // Bulk-append name characters needing no case folding.
                let start = self.cursor();
                let (end, len) = self.scan_while(|b| {
                    !b.is_ascii_uppercase()
                        && b != 0
                        && b != b'/'
                        && b != b'>'
                        && b != b'='
                        && b != b'"'
                        && b != b'\''
                        && b != b'<'
                        && !b.is_ascii_whitespace()
                });
                if len > 0 {
                    Self::push_range(&self.segs, start, end, &mut self.name_buf);
                    self.col += len as u32;
                    self.set_cursor(end);
                }
                match self.peek_byte() {
                    Some(b) if b.is_ascii_whitespace() => {
                        self.consume();
                        self.finish_attribute_name();
                        self.state = State::AfterAttributeName;
                    }
                    Some(b'/') => {
                        self.consume();
                        self.finish_attribute_name();
                        self.state = State::SelfClosingStartTag;
                    }
                    Some(b'>') => {
                        self.consume();
                        self.finish_attribute_name();
                        self.state = State::Data;
                        self.emit_tag();
                    }
                    Some(b'=') => {
                        self.consume();
                        self.finish_attribute_name();
                        self.state = State::BeforeAttributeValue;
                    }
                    Some(b) if b.is_ascii_uppercase() => {
                        self.consume();
                        self.attr_name_push((b as char).to_ascii_lowercase());
                    }
                    Some(0) => {
                        self.consume();
                        self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                        self.attr_name_push(REPLACEMENT);
                    }
                    Some(b'"' | b'\'' | b'<') => {
                        // `unexpected-character-in-attribute-name`.
                        let c = self.consume().unwrap();
                        self.tokenizer_error(ParseErrorCode::UnexpectedCharacterInAttributeName);
                        self.attr_name_push(c);
                    }
                    Some(_) => {
                        let c = self.consume().unwrap();
                        self.attr_name_push(c);
                    }
                    None => {
                        // `eof-in-tag`.
                        self.tokenizer_error(ParseErrorCode::EofInTag);
                        self.emit(Token::Eof);
                    }
                }
            }

            State::AfterAttributeName => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                }
                Some('/') => {
                    self.consume();
                    self.state = State::SelfClosingStartTag;
                }
                Some('=') => {
                    self.consume();
                    self.state = State::BeforeAttributeValue;
                }
                Some('>') => {
                    self.consume();
                    self.state = State::Data;
                    self.emit_tag();
                }
                Some(_) => {
                    // Anything else: start a new attribute with no parse
                    // error (13.2.5.34; `missing-whitespace-between-
                    // attributes` belongs only to the 13.2.5.39
                    // after-attribute-name switch below).
                    self.start_attribute("");
                    self.state = State::AttributeName;
                }
                None => {
                    // `eof-in-tag`.
                    self.tokenizer_error(ParseErrorCode::EofInTag);
                    self.emit(Token::Eof);
                }
            },

            State::BeforeAttributeValue => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                }
                Some('"') => {
                    self.consume();
                    self.state = State::AttributeValueDoubleQuoted;
                }
                Some('\'') => {
                    self.consume();
                    self.state = State::AttributeValueSingleQuoted;
                }
                Some('>') => {
                    // `missing-attribute-value`.
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::MissingAttributeValue);
                    self.state = State::Data;
                    self.emit_tag();
                }
                Some(_) => {
                    self.state = State::AttributeValueUnquoted;
                }
                None => {
                    // `eof-in-tag`.
                    self.tokenizer_error(ParseErrorCode::EofInTag);
                    self.emit(Token::Eof);
                }
            },

            State::AttributeValueDoubleQuoted => {
                // Bulk-append value bytes up to `"`, `&`, NUL or non-ASCII.
                let start = self.cursor();
                let (end, len) = self.scan_while(|b| b != b'"' && b != b'&' && b != 0);
                if len > 0 {
                    if !self.current_attr_dropped
                        && let Some(tag) = self.current_tag.as_mut()
                        && let Some((_, value)) = tag.attrs.last_mut()
                    {
                        Self::push_range(&self.segs, start, end, value);
                    }
                    Self::with_range_pieces(&self.segs, start, end, |piece| {
                        advance_ascii(&mut self.line, &mut self.col, piece.as_bytes());
                    });
                    self.set_cursor(end);
                }
                match self.peek_byte() {
                    Some(b'"') => {
                        self.consume();
                        self.state = State::AfterAttributeValueQuoted;
                    }
                    Some(b'&') => {
                        self.amp_pos = self.cursor();
                        self.entity_start = SourceLoc {
                            line: self.line,
                            col: self.col,
                        };
                        self.consume();
                        self.return_state = State::AttributeValueDoubleQuoted;
                        self.state = State::CharacterReference;
                    }
                    Some(0) => {
                        self.consume();
                        self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                        self.attr_value_push(REPLACEMENT);
                    }
                    Some(_) => {
                        let c = self.consume().unwrap();
                        self.attr_value_push(c);
                    }
                    None => {
                        // `eof-in-tag`.
                        self.tokenizer_error(ParseErrorCode::EofInTag);
                        self.emit(Token::Eof);
                    }
                }
            }

            State::AttributeValueSingleQuoted => {
                let start = self.cursor();
                let (end, len) = self.scan_while(|b| b != b'\'' && b != b'&' && b != 0);
                if len > 0 {
                    if !self.current_attr_dropped
                        && let Some(tag) = self.current_tag.as_mut()
                        && let Some((_, value)) = tag.attrs.last_mut()
                    {
                        Self::push_range(&self.segs, start, end, value);
                    }
                    Self::with_range_pieces(&self.segs, start, end, |piece| {
                        advance_ascii(&mut self.line, &mut self.col, piece.as_bytes());
                    });
                    self.set_cursor(end);
                }
                match self.peek_byte() {
                    Some(b'\'') => {
                        self.consume();
                        self.state = State::AfterAttributeValueQuoted;
                    }
                    Some(b'&') => {
                        self.amp_pos = self.cursor();
                        self.entity_start = SourceLoc {
                            line: self.line,
                            col: self.col,
                        };
                        self.consume();
                        self.return_state = State::AttributeValueSingleQuoted;
                        self.state = State::CharacterReference;
                    }
                    Some(0) => {
                        self.consume();
                        self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                        self.attr_value_push(REPLACEMENT);
                    }
                    Some(_) => {
                        let c = self.consume().unwrap();
                        self.attr_value_push(c);
                    }
                    None => {
                        // `eof-in-tag`.
                        self.tokenizer_error(ParseErrorCode::EofInTag);
                        self.emit(Token::Eof);
                    }
                }
            }

            State::AttributeValueUnquoted => {
                let start = self.cursor();
                let (end, len) = self.scan_while(|b| {
                    b != b'&'
                        && b != b'>'
                        && b != 0
                        && b != b'"'
                        && b != b'\''
                        && b != b'<'
                        && b != b'='
                        && b != b'`'
                        && !b.is_ascii_whitespace()
                });
                if len > 0 {
                    if !self.current_attr_dropped
                        && let Some(tag) = self.current_tag.as_mut()
                        && let Some((_, value)) = tag.attrs.last_mut()
                    {
                        Self::push_range(&self.segs, start, end, value);
                    }
                    Self::with_range_pieces(&self.segs, start, end, |piece| {
                        advance_ascii(&mut self.line, &mut self.col, piece.as_bytes());
                    });
                    self.set_cursor(end);
                }
                match self.peek_byte() {
                    Some(b) if b.is_ascii_whitespace() => {
                        self.consume();
                        self.state = State::BeforeAttributeName;
                    }
                    Some(b'&') => {
                        self.amp_pos = self.cursor();
                        self.entity_start = SourceLoc {
                            line: self.line,
                            col: self.col,
                        };
                        self.consume();
                        self.return_state = State::AttributeValueUnquoted;
                        self.state = State::CharacterReference;
                    }
                    Some(b'>') => {
                        self.consume();
                        self.state = State::Data;
                        self.emit_tag();
                    }
                    Some(0) => {
                        self.consume();
                        self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                        self.attr_value_push(REPLACEMENT);
                    }
                    Some(b'"' | b'\'' | b'<' | b'=' | b'`') => {
                        // `unexpected-character-in-unquoted-attribute-value`.
                        let c = self.consume().unwrap();
                        self.tokenizer_error(
                            ParseErrorCode::UnexpectedCharacterInUnquotedAttributeValue,
                        );
                        self.attr_value_push(c);
                    }
                    Some(_) => {
                        let c = self.consume().unwrap();
                        self.attr_value_push(c);
                    }
                    None => {
                        // `eof-in-tag`.
                        self.tokenizer_error(ParseErrorCode::EofInTag);
                        self.emit(Token::Eof);
                    }
                }
            }

            State::AfterAttributeValueQuoted => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                    self.state = State::BeforeAttributeName;
                }
                Some('/') => {
                    self.consume();
                    self.state = State::SelfClosingStartTag;
                }
                Some('>') => {
                    self.consume();
                    self.state = State::Data;
                    self.emit_tag();
                }
                Some(_) => {
                    // `missing-whitespace-between-attributes`.
                    self.tokenizer_error(ParseErrorCode::MissingWhitespaceBetweenAttributes);
                    self.state = State::BeforeAttributeName;
                }
                None => {
                    // `eof-in-tag`.
                    self.tokenizer_error(ParseErrorCode::EofInTag);
                    self.emit(Token::Eof);
                }
            },

            State::SelfClosingStartTag => match self.peek() {
                Some('>') => {
                    self.consume();
                    if let Some(tag) = self.current_tag.as_mut() {
                        tag.self_closing = true;
                    }
                    self.state = State::Data;
                    self.emit_tag();
                }
                Some(_) => {
                    // `unexpected-solidus-in-tag`.
                    self.tokenizer_error(ParseErrorCode::UnexpectedSolidusInTag);
                    self.state = State::BeforeAttributeName;
                }
                None => {
                    // `eof-in-tag`.
                    self.tokenizer_error(ParseErrorCode::EofInTag);
                    self.emit(Token::Eof);
                }
            },

            State::BogusComment => {
                // Bulk-append up to `>` or NUL.
                let start = self.cursor();
                let (end, len) = self.scan_while(|b| b != b'>' && b != 0);
                if len > 0 {
                    Self::push_range_ascii(
                        &self.segs,
                        start,
                        end,
                        &mut self.line,
                        &mut self.col,
                        &mut self.current_comment,
                    );
                    self.set_cursor(end);
                }
                match self.peek_byte() {
                    Some(b'>') => {
                        self.consume();
                        self.state = State::Data;
                        self.emit_comment();
                    }
                    Some(0) => {
                        self.consume();
                        self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                        self.current_comment.push(REPLACEMENT);
                    }
                    Some(_) => {
                        let c = self.consume().unwrap();
                        self.current_comment.push(c);
                    }
                    None => {
                        self.emit_comment();
                        self.emit(Token::Eof);
                    }
                }
            }

            // ---- processing instruction states (WHATWG 13.2.5.72-76) ----
            State::PiOpen => match self.peek() {
                // ASCII alpha or `_` begins the target; reconsume it in
                // the target state.
                Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                    self.state = State::PiTarget;
                }
                Some(_) => {
                    // invalid-first-character-of-processing-instruction-target:
                    // degrade to a bogus comment.
                    let buf = std::mem::take(&mut self.temp_buf);
                    self.convert_pi_to_comment(buf);
                }
                None => self.emit(Token::Eof),
            },

            State::PiTarget => match self.peek() {
                Some(c) if c.is_ascii_whitespace() || c == '?' || c == '>' => {
                    let target = std::mem::take(&mut self.temp_buf);
                    if target.eq_ignore_ascii_case("xml")
                        || target.eq_ignore_ascii_case("xml-stylesheet")
                    {
                        // disallowed-processing-instruction-target: degrade
                        // to a bogus comment.
                        self.convert_pi_to_comment(target);
                    } else {
                        self.current_pi = Some(PiToken {
                            target,
                            data: String::new(),
                        });
                        self.state = State::PiAfterTarget;
                    }
                    // Reconsume the terminator in the next state.
                }
                Some(c) if c.is_ascii_alphanumeric() || c == '-' || c == '_' => {
                    self.consume();
                    self.temp_buf.push(c);
                }
                Some(_) => {
                    // invalid-processing-instruction-target: degrade to a
                    // bogus comment.
                    let buf = std::mem::take(&mut self.temp_buf);
                    self.convert_pi_to_comment(buf);
                }
                None => self.emit(Token::Eof),
            },

            State::PiAfterTarget => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                }
                Some(_) => {
                    // Reconsume in the data state.
                    self.state = State::PiData;
                }
                None => self.emit(Token::Eof),
            },

            State::PiData => {
                // Bulk-append up to `?`, `>` or non-ASCII.
                let start = self.cursor();
                let (end, len) = self.scan_while(|b| b != b'?' && b != b'>');
                if len > 0 {
                    Self::push_range_ascii(
                        &self.segs,
                        start,
                        end,
                        &mut self.line,
                        &mut self.col,
                        &mut self
                            .current_pi
                            .as_mut()
                            .expect("no current processing instruction")
                            .data,
                    );
                    self.set_cursor(end);
                }
                match self.peek_byte() {
                    Some(b'?') => {
                        self.consume();
                        self.state = State::PiQuestionable;
                    }
                    Some(b'>') => {
                        self.consume();
                        self.state = State::Data;
                        self.emit_pi();
                    }
                    Some(_) => {
                        let c = self.consume().unwrap();
                        self.pi_data_mut().push(c);
                    }
                    None => self.emit(Token::Eof),
                }
            }

            State::PiQuestionable => match self.peek() {
                Some('>') => {
                    self.consume();
                    self.state = State::Data;
                    self.emit_pi();
                }
                Some(_) => {
                    // The `?` joins the data; reconsume in the data state.
                    self.pi_data_mut().push('?');
                    self.state = State::PiData;
                }
                None => self.emit(Token::Eof),
            },

            // ---- markup declaration ----
            State::MarkupDeclarationOpen => {
                if self.peek_literal("--") {
                    self.skip(2);
                    self.current_comment.clear();
                    self.state = State::CommentStart;
                } else if self.peek_literal("doctype") {
                    self.skip(7);
                    self.state = State::Doctype;
                } else if self.peek_cdata() {
                    // `[CDATA[` matches ASCII case-sensitively (unlike
                    // `DOCTYPE` above).
                    if self.cdata_allowed {
                        self.skip(7);
                        self.state = State::CdataSection;
                    } else {
                        // `cdata-in-html-content`: bogus comment.
                        self.tokenizer_error(ParseErrorCode::CdataInHtmlContent);
                        self.current_comment.clear();
                        self.state = State::BogusComment;
                    }
                } else {
                    // Anything else (including EOF): `incorrectly-opened-
                    // comment`, then a bogus comment. BogusComment handles
                    // EOF by emitting the comment + EOF.
                    if self.peek().is_some() {
                        self.tokenizer_error(ParseErrorCode::IncorrectlyOpenedComment);
                    }
                    self.current_comment.clear();
                    self.state = State::BogusComment;
                }
            }

            State::CommentStart => match self.peek() {
                Some('-') => {
                    self.consume();
                    self.state = State::CommentStartDash;
                }
                Some('>') => {
                    // `abrupt-closing-of-empty-comment`.
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::AbruptClosingOfEmptyComment);
                    self.state = State::Data;
                    self.emit_comment();
                }
                Some(_) => {
                    self.state = State::Comment;
                }
                None => {
                    // `eof-in-comment`.
                    self.tokenizer_error(ParseErrorCode::EofInComment);
                    self.emit_comment();
                    self.emit(Token::Eof);
                }
            },

            State::CommentStartDash => match self.peek() {
                Some('-') => {
                    self.consume();
                    self.state = State::CommentEnd;
                }
                Some('>') => {
                    // `abrupt-closing-of-empty-comment`.
                    self.consume();
                    self.tokenizer_error(ParseErrorCode::AbruptClosingOfEmptyComment);
                    self.state = State::Data;
                    self.emit_comment();
                }
                Some(_) => {
                    self.current_comment.push('-');
                    self.state = State::Comment;
                }
                None => {
                    // `eof-in-comment`.
                    self.tokenizer_error(ParseErrorCode::EofInComment);
                    self.emit_comment();
                    self.emit(Token::Eof);
                }
            },

            State::Comment => {
                // Bulk-append up to `<`, `-`, NUL or non-ASCII.
                let start = self.cursor();
                let (end, len) = self.scan_while(|b| b != b'<' && b != b'-' && b != 0);
                if len > 0 {
                    Self::push_range_ascii(
                        &self.segs,
                        start,
                        end,
                        &mut self.line,
                        &mut self.col,
                        &mut self.current_comment,
                    );
                    self.set_cursor(end);
                }
                match self.peek_byte() {
                    Some(b'<') => {
                        self.consume();
                        self.current_comment.push('<');
                        self.state = State::CommentLessThanSign;
                    }
                    Some(b'-') => {
                        self.consume();
                        self.state = State::CommentEndDash;
                    }
                    Some(0) => {
                        self.consume();
                        self.tokenizer_error(ParseErrorCode::UnexpectedNullCharacter);
                        self.current_comment.push(REPLACEMENT);
                    }
                    Some(_) => {
                        let c = self.consume().unwrap();
                        self.current_comment.push(c);
                    }
                    None => {
                        // `eof-in-comment`.
                        self.tokenizer_error(ParseErrorCode::EofInComment);
                        self.emit_comment();
                        self.emit(Token::Eof);
                    }
                }
            }

            State::CommentLessThanSign => match self.peek() {
                Some('!') => {
                    self.consume();
                    self.current_comment.push('!');
                    self.state = State::CommentLessThanSignBang;
                }
                Some('<') => {
                    self.consume();
                    self.current_comment.push('<');
                }
                Some(_) => {
                    self.state = State::Comment;
                }
                None => {
                    // `eof-in-comment`.
                    self.tokenizer_error(ParseErrorCode::EofInComment);
                    self.emit_comment();
                    self.emit(Token::Eof);
                }
            },

            State::CommentLessThanSignBang => match self.peek() {
                Some('-') => {
                    self.consume();
                    self.state = State::CommentLessThanSignBangDash;
                }
                Some(_) => {
                    self.state = State::Comment;
                }
                None => {
                    // `eof-in-comment`.
                    self.tokenizer_error(ParseErrorCode::EofInComment);
                    self.emit_comment();
                    self.emit(Token::Eof);
                }
            },

            State::CommentLessThanSignBangDash => match self.peek() {
                Some('-') => {
                    self.consume();
                    self.state = State::CommentLessThanSignBangDashDash;
                }
                Some(_) => {
                    self.state = State::CommentEndDash;
                }
                None => {
                    // `eof-in-comment`.
                    self.tokenizer_error(ParseErrorCode::EofInComment);
                    self.emit_comment();
                    self.emit(Token::Eof);
                }
            },

            State::CommentLessThanSignBangDashDash => {
                self.state = State::CommentEnd;
                if self.peek().is_none() {
                    // `eof-in-comment`.
                    self.tokenizer_error(ParseErrorCode::EofInComment);
                    self.emit_comment();
                    self.emit(Token::Eof);
                }
            }

            State::CommentEndDash => match self.peek() {
                Some('-') => {
                    self.consume();
                    self.state = State::CommentEnd;
                }
                Some(_) => {
                    self.current_comment.push('-');
                    self.state = State::Comment;
                }
                None => {
                    // `eof-in-comment`.
                    self.tokenizer_error(ParseErrorCode::EofInComment);
                    self.emit_comment();
                    self.emit(Token::Eof);
                }
            },

            State::CommentEnd => match self.peek() {
                Some('>') => {
                    self.consume();
                    self.state = State::Data;
                    self.emit_comment();
                }
                Some('!') => {
                    self.consume();
                    self.state = State::CommentEndBang;
                }
                Some('-') => {
                    self.consume();
                    self.current_comment.push('-');
                }
                Some(_) => {
                    self.current_comment.push_str("--");
                    self.state = State::Comment;
                }
                None => {
                    // `eof-in-comment`.
                    self.tokenizer_error(ParseErrorCode::EofInComment);
                    self.emit_comment();
                    self.emit(Token::Eof);
                }
            },

            State::CommentEndBang => match self.peek() {
                Some('-') => {
                    self.consume();
                    self.current_comment.push_str("--!");
                    self.state = State::CommentEndDash;
                }
                Some('>') => {
                    self.consume();
                    self.state = State::Data;
                    self.emit_comment();
                }
                Some(_) => {
                    self.current_comment.push_str("--!");
                    self.state = State::Comment;
                }
                None => {
                    // `eof-in-comment`.
                    self.tokenizer_error(ParseErrorCode::EofInComment);
                    self.emit_comment();
                    self.emit(Token::Eof);
                }
            },

            // ---- DOCTYPE states ----
            State::Doctype => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                    self.state = State::BeforeDoctypeName;
                }
                Some('>') => {
                    self.state = State::BeforeDoctypeName;
                }
                Some(_) => {
                    self.state = State::BeforeDoctypeName;
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            State::BeforeDoctypeName => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                }
                Some(c) if c.is_ascii_uppercase() => {
                    self.consume();
                    self.current_doctype.name = Some(c.to_ascii_lowercase().to_string());
                    self.state = State::DoctypeName;
                }
                Some('\0') => {
                    self.consume();
                    self.current_doctype.name = Some(REPLACEMENT.to_string());
                    self.state = State::DoctypeName;
                }
                Some('>') => {
                    self.consume();
                    self.current_doctype.force_quirks = true;
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some(c) => {
                    self.consume();
                    self.current_doctype.name = Some(c.to_string());
                    self.state = State::DoctypeName;
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            State::DoctypeName => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                    self.state = State::AfterDoctypeName;
                }
                Some('>') => {
                    self.consume();
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some(c) if c.is_ascii_uppercase() => {
                    self.consume();
                    if let Some(n) = self.current_doctype.name.as_mut() {
                        n.push(c.to_ascii_lowercase());
                    }
                }
                Some('\0') => {
                    self.consume();
                    if let Some(n) = self.current_doctype.name.as_mut() {
                        n.push(REPLACEMENT);
                    }
                }
                Some(c) => {
                    self.consume();
                    if let Some(n) = self.current_doctype.name.as_mut() {
                        n.push(c);
                    } else {
                        self.current_doctype.name = Some(c.to_string());
                    }
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            State::AfterDoctypeName => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                }
                Some('>') => {
                    self.consume();
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some(_) => {
                    if self.peek_literal("public") {
                        self.skip(6);
                        self.state = State::AfterDoctypePublicKeyword;
                    } else if self.peek_literal("system") {
                        self.skip(6);
                        self.state = State::AfterDoctypeSystemKeyword;
                    } else {
                        self.current_doctype.force_quirks = true;
                        self.state = State::BogusDoctype;
                    }
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            // WHATWG 13.2.5.54 "after DOCTYPE public keyword state".
            State::AfterDoctypePublicKeyword => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                    self.state = State::BeforeDoctypePublicIdentifier;
                }
                Some('"') => {
                    self.consume();
                    self.current_doctype.public_id = Some(String::new());
                    self.state = State::DoctypePublicIdentifierDoubleQuoted;
                }
                Some('\'') => {
                    self.consume();
                    self.current_doctype.public_id = Some(String::new());
                    self.state = State::DoctypePublicIdentifierSingleQuoted;
                }
                Some('>') => {
                    self.consume();
                    self.current_doctype.force_quirks = true;
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some(_) => {
                    self.current_doctype.force_quirks = true;
                    self.state = State::BogusDoctype;
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            State::BeforeDoctypePublicIdentifier => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                }
                Some('"') => {
                    self.consume();
                    self.current_doctype.public_id = Some(String::new());
                    self.state = State::DoctypePublicIdentifierDoubleQuoted;
                }
                Some('\'') => {
                    self.consume();
                    self.current_doctype.public_id = Some(String::new());
                    self.state = State::DoctypePublicIdentifierSingleQuoted;
                }
                Some('>') => {
                    self.consume();
                    self.current_doctype.force_quirks = true;
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some(_) => {
                    self.current_doctype.force_quirks = true;
                    self.state = State::BogusDoctype;
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            State::DoctypePublicIdentifierDoubleQuoted => match self.peek() {
                Some('"') => {
                    self.consume();
                    self.state = State::AfterDoctypePublicIdentifier;
                }
                Some('\0') => {
                    self.consume();
                    if let Some(p) = self.current_doctype.public_id.as_mut() {
                        p.push(REPLACEMENT);
                    }
                }
                Some('>') => {
                    self.consume();
                    self.current_doctype.force_quirks = true;
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some(c) => {
                    self.consume();
                    if let Some(p) = self.current_doctype.public_id.as_mut() {
                        p.push(c);
                    }
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            State::DoctypePublicIdentifierSingleQuoted => match self.peek() {
                Some('\'') => {
                    self.consume();
                    self.state = State::AfterDoctypePublicIdentifier;
                }
                Some('\0') => {
                    self.consume();
                    if let Some(p) = self.current_doctype.public_id.as_mut() {
                        p.push(REPLACEMENT);
                    }
                }
                Some('>') => {
                    self.consume();
                    self.current_doctype.force_quirks = true;
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some(c) => {
                    self.consume();
                    if let Some(p) = self.current_doctype.public_id.as_mut() {
                        p.push(c);
                    }
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            State::AfterDoctypePublicIdentifier => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                    self.state = State::BetweenDoctypePublicAndSystemIdentifiers;
                }
                Some('>') => {
                    self.consume();
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some('"') => {
                    self.consume();
                    self.current_doctype.system_id = Some(String::new());
                    self.state = State::DoctypeSystemIdentifierDoubleQuoted;
                }
                Some('\'') => {
                    self.consume();
                    self.current_doctype.system_id = Some(String::new());
                    self.state = State::DoctypeSystemIdentifierSingleQuoted;
                }
                Some(_) => {
                    self.current_doctype.force_quirks = true;
                    self.state = State::BogusDoctype;
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            State::BetweenDoctypePublicAndSystemIdentifiers => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                }
                Some('>') => {
                    self.consume();
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some('"') => {
                    self.consume();
                    self.current_doctype.system_id = Some(String::new());
                    self.state = State::DoctypeSystemIdentifierDoubleQuoted;
                }
                Some('\'') => {
                    self.consume();
                    self.current_doctype.system_id = Some(String::new());
                    self.state = State::DoctypeSystemIdentifierSingleQuoted;
                }
                Some(_) => {
                    self.current_doctype.force_quirks = true;
                    self.state = State::BogusDoctype;
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            // WHATWG 13.2.5.58 "after DOCTYPE system keyword state".
            State::AfterDoctypeSystemKeyword => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                    self.state = State::BeforeDoctypeSystemIdentifier;
                }
                Some('"') => {
                    self.consume();
                    self.current_doctype.system_id = Some(String::new());
                    self.state = State::DoctypeSystemIdentifierDoubleQuoted;
                }
                Some('\'') => {
                    self.consume();
                    self.current_doctype.system_id = Some(String::new());
                    self.state = State::DoctypeSystemIdentifierSingleQuoted;
                }
                Some('>') => {
                    self.consume();
                    self.current_doctype.force_quirks = true;
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some(_) => {
                    self.current_doctype.force_quirks = true;
                    self.state = State::BogusDoctype;
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            State::BeforeDoctypeSystemIdentifier => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                }
                Some('"') => {
                    self.consume();
                    self.current_doctype.system_id = Some(String::new());
                    self.state = State::DoctypeSystemIdentifierDoubleQuoted;
                }
                Some('\'') => {
                    self.consume();
                    self.current_doctype.system_id = Some(String::new());
                    self.state = State::DoctypeSystemIdentifierSingleQuoted;
                }
                Some('>') => {
                    // WHATWG 13.2.5.59: `>` with the system identifier
                    // still missing ends the token cleanly: no parse
                    // error, no force-quirks.
                    self.consume();
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some(_) => {
                    self.current_doctype.force_quirks = true;
                    self.state = State::BogusDoctype;
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            State::DoctypeSystemIdentifierDoubleQuoted => match self.peek() {
                Some('"') => {
                    self.consume();
                    self.state = State::AfterDoctypeSystemIdentifier;
                }
                Some('\0') => {
                    self.consume();
                    if let Some(s) = self.current_doctype.system_id.as_mut() {
                        s.push(REPLACEMENT);
                    }
                }
                Some('>') => {
                    self.consume();
                    self.current_doctype.force_quirks = true;
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some(c) => {
                    self.consume();
                    if let Some(s) = self.current_doctype.system_id.as_mut() {
                        s.push(c);
                    }
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            State::DoctypeSystemIdentifierSingleQuoted => match self.peek() {
                Some('\'') => {
                    self.consume();
                    self.state = State::AfterDoctypeSystemIdentifier;
                }
                Some('\0') => {
                    self.consume();
                    if let Some(s) = self.current_doctype.system_id.as_mut() {
                        s.push(REPLACEMENT);
                    }
                }
                Some('>') => {
                    self.consume();
                    self.current_doctype.force_quirks = true;
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some(c) => {
                    self.consume();
                    if let Some(s) = self.current_doctype.system_id.as_mut() {
                        s.push(c);
                    }
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            State::AfterDoctypeSystemIdentifier => match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.consume();
                }
                Some('>') => {
                    self.consume();
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some(_) => {
                    self.state = State::BogusDoctype;
                }
                None => {
                    self.current_doctype.force_quirks = true;
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            State::BogusDoctype => match self.peek() {
                Some('>') => {
                    self.consume();
                    self.state = State::Data;
                    self.emit_doctype();
                }
                Some(_) => {
                    self.consume();
                }
                None => {
                    self.emit_doctype();
                    self.emit(Token::Eof);
                }
            },

            // ---- CDATA ----
            State::CdataSection => match self.char_run(*b"]") {
                RunEnd::Stop(b']') => {
                    self.consume();
                    self.state = State::CdataSectionBracket;
                }
                RunEnd::Stop(b) => self.char_run_unexpected(b),
                RunEnd::Eof => self.emit(Token::Eof),
            },

            State::CdataSectionBracket => match self.peek() {
                Some(']') => {
                    self.consume();
                    self.state = State::CdataSectionEnd;
                }
                Some(_) => {
                    self.emit_char(']');
                    self.state = State::CdataSection;
                }
                None => {
                    self.emit_char(']');
                    self.emit(Token::Eof);
                }
            },

            State::CdataSectionEnd => match self.peek() {
                Some(']') => {
                    self.consume();
                    self.emit_char(']');
                }
                Some('>') => {
                    self.consume();
                    self.state = State::Data;
                }
                Some(_) => {
                    self.emit_char(']');
                    self.emit_char(']');
                    self.state = State::CdataSection;
                }
                None => {
                    self.emit_char(']');
                    self.emit_char(']');
                    self.emit(Token::Eof);
                }
            },
        }
    }

    /// Get the next token, running the state machine until one is emitted.
    pub fn next_token(&mut self) -> Token {
        loop {
            if let Some((tok, loc)) = self.queue.pop_front() {
                if tok == Token::Eof {
                    self.eof_emitted = true;
                }
                // Text accumulated before this token becomes one Text
                // token; runs are always older than queued tokens. The
                // run is only drained here (not on every step) so entity
                // fallbacks and other multi-step text paths merge into a
                // single token.
                if let Some((ttok, tloc)) = self.pop_text() {
                    self.queue.push_front((tok, loc));
                    self.token_loc = tloc;
                    return ttok;
                }
                self.token_loc = loc;
                return tok;
            }
            if self.eof_emitted {
                // After EOF: drain any final text, then report EOF.
                if let Some((tok, loc)) = self.pop_text() {
                    self.token_loc = loc;
                    return tok;
                }
                self.token_loc = SourceLoc {
                    line: self.line,
                    col: self.col,
                };
                return Token::Eof;
            }
            self.step();
        }
    }

    /// Source location of the token most recently returned by
    /// [`Tokenizer::next_token`]: the `<` for tags/comments/doctypes, the
    /// first character for text tokens, the end of input for EOF.
    /// Lines and columns are 1-based; columns count characters, not bytes.
    pub fn token_loc(&self) -> SourceLoc {
        self.token_loc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenize(input: &str) -> Vec<Token> {
        let mut t = Tokenizer::new(input);
        let mut out = Vec::new();
        loop {
            let tok = t.next_token();
            let done = tok == Token::Eof;
            out.push(tok);
            if done {
                break;
            }
        }
        out
    }

    fn text(s: &str) -> Vec<Token> {
        vec![Token::Text(s.to_string())]
    }

    #[test]
    fn new_borrows_input_without_preprocessing() {
        // No BOM, no CR: the tokenizer must work on the caller's buffer
        // with no copy (pointer equality proves zero-copy).
        let input = "a".repeat(4096);
        let t = Tokenizer::new(&input);
        assert!(matches!(t.segs[0].buf, Cow::Borrowed(_)));
        assert_eq!(t.segs[0].buf.as_ptr(), input.as_ptr());
        assert_eq!(t.segs[0].buf.len(), input.len());
    }

    #[test]
    fn new_copies_only_for_cr_normalization() {
        // CR/CRLF normalization rewrites the input, so this path owns.
        let t = Tokenizer::new("a\r\nb\rc");
        assert!(matches!(t.segs[0].buf, Cow::Owned(_)));
        assert_eq!(&*t.segs[0].buf, "a\nb\nc");
        // Tokenizing the rewritten input still works.
        let mut toks = tokenize("a\r\nb");
        toks.pop(); // EOF
        assert_eq!(toks, text("a\nb"));
    }

    #[test]
    fn data_text_and_entities() {
        let mut toks = tokenize("a&amp;b");
        toks.pop(); // EOF
        // The `&amp;` expansion merges with the adjacent runs.
        assert_eq!(toks, text("a&b"));
    }

    #[test]
    fn entity_without_semicolon() {
        // &ampx -> "&" "&x"? No: maximal munch "&amp" then literal "x".
        let mut toks = tokenize("&ampx");
        toks.pop();
        assert_eq!(toks, text("&x"));
    }

    #[test]
    fn entity_missing_semicolon_before_alnum() {
        // &ampla -> "&" "la" (legacy `&amp` match; no-semicolon is only a
        // parse error, and the ambiguous-ampersand rule is attribute-only).
        let mut toks = tokenize("&ampla");
        toks.pop();
        assert_eq!(toks, text("&la"));
    }

    #[test]
    fn ambiguous_ampersand_in_attribute_is_literal() {
        // &amp followed by alnum inside an attribute value: literal text.
        let toks = tokenize("<div title=\"&ampla\">");
        match &toks[0] {
            Token::StartTag(t) => {
                assert_eq!(
                    t.attrs,
                    vec![(AttrName::from("title"), "&ampla".to_string())]
                );
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn unknown_entity_is_literal() {
        let mut toks = tokenize("&foo;bar");
        toks.pop();
        assert_eq!(toks, text("&foo;bar"));
    }

    #[test]
    fn numeric_references() {
        let mut toks = tokenize("&#38;&#x26;&#65;");
        toks.pop();
        assert_eq!(toks, text("&&A"));
    }

    #[test]
    fn numeric_edge_cases() {
        // NUL -> FFFD; out of range -> FFFD; surrogate -> FFFD
        let mut toks = tokenize("&#0;&#x110000;&#xD800;");
        toks.pop();
        assert_eq!(toks, text("\u{FFFD}\u{FFFD}\u{FFFD}"));
    }

    #[test]
    fn numeric_control_mapping() {
        // &#128; -> U+20AC (Windows-1252 remap)
        let mut toks = tokenize("&#128;&#x9F;");
        toks.pop();
        assert_eq!(toks, text("\u{20AC}\u{178}"));
    }

    #[test]
    fn numeric_reference_huge_hex_no_overflow() {
        // `&#xFFFFFFFFFFFFFFFFFFFFFFFF;` must not panic on overflow in
        // debug builds (or wrap in release): any value above 0x10FFFF
        // resolves to U+FFFD.
        let mut toks = tokenize("&#xFFFFFFFFFFFFFFFFFFFFFFFF;");
        toks.pop();
        assert_eq!(toks, text("\u{FFFD}"));
    }

    #[test]
    fn numeric_reference_huge_decimal_no_overflow() {
        // Same guarantee for the decimal accumulator.
        let mut toks = tokenize("&#99999999999999999999999999999;");
        toks.pop();
        assert_eq!(toks, text("\u{FFFD}"));
    }

    #[test]
    fn numeric_missing_semicolon() {
        let mut toks = tokenize("&#38x");
        toks.pop();
        assert_eq!(toks, text("&x"));
    }

    #[test]
    fn bare_ampersand() {
        let mut toks = tokenize("a & b &;c");
        toks.pop();
        assert_eq!(toks, text("a & b &;c"));
    }

    #[test]
    fn longest_match_wins() {
        // &notin; vs &not; : "&notin;" -> U+2209; "&notit;" -> U+00AC + "it;"
        let mut toks = tokenize("&notin;&notit;");
        toks.pop();
        assert_eq!(toks, text("\u{2209}\u{00AC}it;"));
    }

    #[test]
    fn double_encoded() {
        // &amp;amp; -> "&amp;"
        let mut toks = tokenize("&amp;amp;");
        toks.pop();
        assert_eq!(toks, text("&amp;"));
    }

    #[test]
    fn tag_with_attributes() {
        let toks = tokenize("<a href=\"x\" title='y' disabled data-z=1>");
        match &toks[0] {
            Token::StartTag(t) => {
                assert_eq!(t.name, "a");
                assert_eq!(
                    t.attrs,
                    vec![
                        (AttrName::from("href"), "x".to_string()),
                        (AttrName::from("title"), "y".to_string()),
                        (AttrName::from("disabled"), "".to_string()),
                        (AttrName::from("data-z"), "1".to_string()),
                    ]
                );
                assert!(!t.self_closing);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn uppercase_tag_and_attr_names_lowercased() {
        let toks = tokenize("<DIV CLASS=\"x\">");
        match &toks[0] {
            Token::StartTag(t) => {
                assert_eq!(t.name, "div");
                assert_eq!(t.attrs, vec![(AttrName::from("class"), "x".to_string())]);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn self_closing_and_end_tag() {
        let toks = tokenize("<br/><hr />");
        match &toks[0] {
            Token::StartTag(t) => assert!(t.self_closing),
            other => panic!("unexpected {other:?}"),
        }
        match &toks[1] {
            Token::StartTag(t) => assert!(t.self_closing),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn duplicate_attributes_dropped() {
        let toks = tokenize("<div a=\"1\" a=\"2\" b>");
        match &toks[0] {
            Token::StartTag(t) => {
                assert_eq!(
                    t.attrs,
                    vec![
                        (AttrName::from("a"), "1".to_string()),
                        (AttrName::from("b"), "".to_string())
                    ]
                );
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn comment_states() {
        let toks = tokenize("<!-- hi --><!--x--><!---->");
        assert_eq!(
            toks,
            vec![
                Token::Comment(" hi ".to_string()),
                Token::Comment("x".to_string()),
                Token::Comment(String::new()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn bogus_comment() {
        let toks = tokenize("<?xml foo?><! not a comment>");
        assert_eq!(
            toks,
            vec![
                Token::Comment("?xml foo?".to_string()),
                Token::Comment(" not a comment".to_string()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn pi_basic() {
        let toks = tokenize("<?target some data?>");
        assert_eq!(
            toks,
            vec![
                Token::ProcessingInstruction {
                    target: "target".to_string(),
                    data: "some data".to_string(),
                },
                Token::Eof,
            ]
        );
    }

    #[test]
    fn pi_empty_data() {
        // `>` and `?` both terminate the target; data may be empty.
        let toks = tokenize("<?a?><?b>");
        assert_eq!(
            toks,
            vec![
                Token::ProcessingInstruction {
                    target: "a".to_string(),
                    data: String::new(),
                },
                Token::ProcessingInstruction {
                    target: "b".to_string(),
                    data: String::new(),
                },
                Token::Eof,
            ]
        );
    }

    #[test]
    fn pi_questionable_data() {
        // `?` not followed by `>` joins the data.
        let toks = tokenize("<?a b?c?>");
        assert_eq!(
            toks,
            vec![
                Token::ProcessingInstruction {
                    target: "a".to_string(),
                    data: "b?c".to_string(),
                },
                Token::Eof,
            ]
        );
    }

    #[test]
    fn pi_target_case_preserved() {
        // Only the xml/xml-stylesheet comparison is ASCII
        // case-insensitive; the target keeps its source case.
        let toks = tokenize("<?Foo Bar?>");
        assert_eq!(
            toks,
            vec![
                Token::ProcessingInstruction {
                    target: "Foo".to_string(),
                    data: "Bar".to_string(),
                },
                Token::Eof,
            ]
        );
    }

    #[test]
    fn pi_disallowed_targets_degrade() {
        // `xml` and `xml-stylesheet` (ASCII case-insensitive) become bogus
        // comments whose data starts with `?`.
        let toks = tokenize("<?XML v?><?xml-stylesheet href=\"x\"?>");
        assert_eq!(
            toks,
            vec![
                Token::Comment("?XML v?".to_string()),
                Token::Comment("?xml-stylesheet href=\"x\"?".to_string()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn pi_bad_first_char_degrades() {
        // Only ASCII alpha or `_` may start a target.
        let toks = tokenize("<?1a?><? foo?><?>");
        assert_eq!(
            toks,
            vec![
                Token::Comment("?1a?".to_string()),
                Token::Comment("? foo?".to_string()),
                Token::Comment("?".to_string()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn pi_bad_target_char_degrades() {
        let toks = tokenize("<?a!b?>");
        assert_eq!(toks, vec![Token::Comment("?a!b?".to_string()), Token::Eof,]);
    }

    #[test]
    fn pi_eof_emits_only_eof() {
        // An unterminated PI never emits a partial token.
        for input in ["<?abc", "<?abc ", "<?abc def", "<?abc def?"] {
            assert_eq!(tokenize(input), vec![Token::Eof], "input: {input}");
        }
    }

    #[test]
    fn doctype_basic() {
        let toks = tokenize("<!DOCTYPE html>");
        match &toks[0] {
            Token::Doctype(d) => {
                assert_eq!(d.name.as_deref(), Some("html"));
                assert!(!d.force_quirks);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn doctype_public_system() {
        let toks = tokenize(
            "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01//EN\" \"http://www.w3.org/TR/html4/strict.dtd\">",
        );
        match &toks[0] {
            Token::Doctype(d) => {
                assert_eq!(d.name.as_deref(), Some("html"));
                assert_eq!(d.public_id.as_deref(), Some("-//W3C//DTD HTML 4.01//EN"));
                assert_eq!(
                    d.system_id.as_deref(),
                    Some("http://www.w3.org/TR/html4/strict.dtd")
                );
                assert!(!d.force_quirks);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn doctype_bogus_forces_quirks() {
        let toks = tokenize("<!DOCTYPE html foo bar>");
        match &toks[0] {
            Token::Doctype(d) => assert!(d.force_quirks),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn crlf_preprocessing() {
        let toks = tokenize("a\r\nb\rc");
        let mut toks = toks;
        toks.pop();
        assert_eq!(toks, text("a\nb\nc"));
    }

    #[test]
    fn rcdata_end_tag_matching() {
        // In RCDATA, `</title>` matching last_start_tag emits an end tag;
        // a `<` not followed by `/` is character data.
        // (A partial tag at EOF is dropped: no `<` is emitted for `<c`.)
        let mut t = Tokenizer::new("a</title>b<c");
        t.set_state(State::Rcdata);
        t.last_start_tag = TagName::from("title");
        let mut out = Vec::new();
        loop {
            let tok = t.next_token();
            let done = tok == Token::Eof;
            out.push(tok);
            if done {
                break;
            }
        }
        assert_eq!(
            out,
            vec![
                Token::Text("a".to_string()),
                Token::EndTag(Tag {
                    name: TagName::from("title"),
                    attrs: vec![],
                    self_closing: false,
                }),
                Token::Text("b".to_string()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn rcdata_inappropriate_end_tag_is_text() {
        // WHATWG 13.2.5.11 RCDATA end tag name state, "Anything else": an
        // inappropriate end tag emits `<`, `/`, and the temporary buffer's
        // characters (source case preserved), then reconsumes.
        let mut t = Tokenizer::new("</div>");
        t.set_state(State::Rcdata);
        t.last_start_tag = TagName::from("title");
        let mut out = Vec::new();
        loop {
            let tok = t.next_token();
            let done = tok == Token::Eof;
            out.push(tok);
            if done {
                break;
            }
        }
        let mut expect = text("</div>");
        expect.push(Token::Eof);
        assert_eq!(out, expect);
    }

    #[test]
    fn rcdata_inappropriate_end_tag_preserves_source_case() {
        // The temporary buffer gets the current input character, not the
        // lowercased tag-name character: `</DIV>` re-emits as `</DIV>`.
        let mut t = Tokenizer::new("</DIV>");
        t.set_state(State::Rcdata);
        t.last_start_tag = TagName::from("title");
        let mut out = Vec::new();
        loop {
            let tok = t.next_token();
            let done = tok == Token::Eof;
            out.push(tok);
            if done {
                break;
            }
        }
        let mut expect = text("</DIV>");
        expect.push(Token::Eof);
        assert_eq!(out, expect);
    }

    #[test]
    fn script_double_escaped() {
        // <script><!--<script></script>--></script> : inner </script> must not close
        let mut t = Tokenizer::new("<!--<script></script>--></script>");
        t.set_state(State::ScriptData);
        t.last_start_tag = TagName::from("script");
        let mut out = Vec::new();
        loop {
            let tok = t.next_token();
            let done = tok == Token::Eof;
            out.push(tok);
            if done {
                break;
            }
        }
        // Everything up to the final </script> is character data.
        let mut expect = text("<!--<script></script>-->");
        expect.push(Token::EndTag(Tag {
            name: TagName::from("script"),
            attrs: vec![],
            self_closing: false,
        }));
        expect.push(Token::Eof);
        assert_eq!(out, expect);
    }

    #[test]
    fn plaintext_consumes_rest() {
        let mut t = Tokenizer::new("a<b>c&d;");
        t.set_state(State::Plaintext);
        let mut out = Vec::new();
        loop {
            let tok = t.next_token();
            let done = tok == Token::Eof;
            out.push(tok);
            if done {
                break;
            }
        }
        let mut expect = text("a<b>c&d;");
        expect.push(Token::Eof);
        assert_eq!(out, expect);
    }

    #[test]
    fn long_text_run_is_single_token() {
        // Batching: a long run of character data is one `Token::Text`,
        // not one token per character.
        let input = "lorem ipsum dolor sit amet ".repeat(2000);
        let mut t = Tokenizer::new(&input);
        let mut count = 0;
        loop {
            let tok = t.next_token();
            if tok == Token::Eof {
                break;
            }
            count += 1;
            assert!(
                matches!(&tok, Token::Text(s) if s == &input),
                "expected one text token, got {tok:?}"
            );
        }
        assert_eq!(count, 1);
    }

    #[test]
    fn text_run_stops_at_markup() {
        // Runs are maximal but still split at tag/entity boundaries that
        // produce other tokens; adjacent character data merges.
        let toks = tokenize("ab&lt;cd<b>ef");
        let toks: Vec<Token> = toks.into_iter().filter(|t| *t != Token::Eof).collect();
        assert_eq!(
            toks,
            vec![
                Token::Text("ab<cd".to_string()),
                Token::StartTag(Tag {
                    name: TagName::from("b"),
                    attrs: vec![],
                    self_closing: false,
                }),
                Token::Text("ef".to_string()),
            ]
        );
    }

    #[test]
    fn charref_in_attribute() {
        let toks = tokenize("<a href=\"?x=1&amp;y=2\">");
        match &toks[0] {
            Token::StartTag(t) => {
                assert_eq!(
                    t.attrs,
                    vec![(AttrName::from("href"), "?x=1&y=2".to_string())]
                );
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn ambiguous_ampersand_in_attribute() {
        // &amp= in attribute stays literal (no semicolon + '=' follows)
        let toks = tokenize("<a href=\"&amp=x\">");
        match &toks[0] {
            Token::StartTag(t) => {
                assert_eq!(
                    t.attrs,
                    vec![(AttrName::from("href"), "&amp=x".to_string())]
                );
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn nul_handling() {
        // NUL in data passes through; tree construction ignores it.
        let mut toks = tokenize("a\0b");
        toks.pop();
        assert_eq!(toks, text("a\0b"));
        // NUL in RCDATA becomes FFFD.
        let mut t = Tokenizer::new("\0");
        t.set_state(State::Rcdata);
        assert_eq!(t.next_token(), Token::Text(REPLACEMENT.to_string()));
    }

    #[test]
    fn entity_table_max_name_length() {
        // `longest_entity_match` caps its search at 32 bytes; the table
        // must never outgrow that.
        let max = crate::entities::NAMED_REFERENCES
            .iter()
            .map(|(n, _)| n.len())
            .max()
            .unwrap();
        assert!(max <= 32, "longest entity name is {max} bytes");
    }

    #[test]
    fn longest_entity_match_matches_reference_scan() {
        // Reference implementation: the original incremental WHATWG scan
        // (longest table entry, stopping at the first byte that is neither
        // an entry nor a prefix of one). `longest_entity_match` must agree
        // with it on every input.
        fn is_prefix_ref(prefix: &str) -> bool {
            let t = crate::entities::NAMED_REFERENCES;
            let i = t.partition_point(|(n, _)| *n < prefix);
            i < t.len() && t[i].0.starts_with(prefix)
        }
        fn reference(input: &str, start: usize) -> usize {
            let bytes = input.as_bytes();
            let mut scanned = String::from("&");
            let mut best = 0usize;
            let mut j = start;
            loop {
                match bytes.get(j).copied() {
                    Some(b';') => {
                        scanned.push(';');
                        if crate::entities::lookup(&scanned[1..]).is_some() {
                            best = scanned.len() - 1;
                        }
                        break;
                    }
                    Some(b) if b.is_ascii_alphanumeric() => {
                        scanned.push(b as char);
                        if crate::entities::lookup(&scanned[1..]).is_some() {
                            best = scanned.len() - 1;
                        }
                        if !is_prefix_ref(&scanned[1..]) {
                            break;
                        }
                    }
                    _ => break,
                }
                j += 1;
            }
            best
        }

        let mut cases: Vec<String> = vec![
            "&".to_string(),
            "&;".to_string(),
            "&a".to_string(),
            "&a;".to_string(),
            "&amp".to_string(),
            "&amp;".to_string(),
            "&ampx".to_string(),
            "&ampx;".to_string(),
            "&ampla".to_string(),
            "&notit;".to_string(),
            "&notin;".to_string(),
            "&CounterClockwiseContourIntegral;".to_string(),
            "&CounterClockwiseContourIntegral".to_string(),
            "&aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;".to_string(),
            "&AMP;".to_string(),
            "&AMP".to_string(),
            "&1".to_string(),
            "&#".to_string(),
        ];
        for (name, _) in crate::entities::NAMED_REFERENCES {
            cases.push(format!("&{name}"));
            cases.push(format!("&{name}tail"));
            cases.push(format!("&{name}x;"));
            // Every strict prefix, with and without `;`.
            for i in 1..name.len() {
                cases.push(format!("&{}", &name[..i]));
                cases.push(format!("&{};", &name[..i]));
            }
        }
        for c in &cases {
            assert_eq!(
                longest_entity_match(c, 1),
                reference(c, 1),
                "mismatch for {c:?}"
            );
        }
    }

    // -- character reference parse errors (WHATWG 13.2.5.77-84) -----------

    /// Tokenize `input`, returning the tokens (sans EOF) and the recorded
    /// parse error codes in order.
    fn tokenize_with_errors(input: &str) -> (Vec<Token>, Vec<ParseErrorCode>) {
        let mut t = Tokenizer::new(input);
        let mut out = Vec::new();
        loop {
            let tok = t.next_token();
            let done = tok == Token::Eof;
            out.push(tok);
            if done {
                break;
            }
        }
        out.pop(); // EOF
        let codes = t.take_errors().into_iter().map(|e| e.code).collect();
        (out, codes)
    }

    fn text_of(toks: &[Token]) -> String {
        toks.iter()
            .map(|t| match t {
                Token::Text(s) => s.as_str(),
                other => panic!("expected text token, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn char_ref_null() {
        let (toks, codes) = tokenize_with_errors("&#0;");
        assert_eq!(text_of(&toks), "�");
        assert_eq!(codes, vec![ParseErrorCode::NullCharacterReference]);
    }

    #[test]
    fn char_ref_outside_unicode_range() {
        let (toks, codes) = tokenize_with_errors("&#x110000;");
        assert_eq!(text_of(&toks), "�");
        assert_eq!(
            codes,
            vec![ParseErrorCode::CharacterReferenceOutsideUnicodeRange]
        );
    }

    #[test]
    fn char_ref_surrogate() {
        let (toks, codes) = tokenize_with_errors("&#xD800;");
        assert_eq!(text_of(&toks), "�");
        assert_eq!(codes, vec![ParseErrorCode::SurrogateCharacterReference]);
    }

    #[test]
    fn char_ref_noncharacter() {
        // Noncharacters decode to themselves; still a parse error.
        let (toks, codes) = tokenize_with_errors("&#xFDD0;");
        assert_eq!(text_of(&toks), "﷐");
        assert_eq!(codes, vec![ParseErrorCode::NoncharacterCharacterReference]);
    }

    #[test]
    fn char_ref_control_remaps() {
        // `&#x80;` maps to U+20AC (spec table) and reports the error.
        let (toks, codes) = tokenize_with_errors("&#x80;");
        assert_eq!(text_of(&toks), "€");
        assert_eq!(codes, vec![ParseErrorCode::ControlCharacterReference]);
    }

    #[test]
    fn char_ref_missing_semicolon_numeric() {
        let (toks, codes) = tokenize_with_errors("&#65x");
        assert_eq!(text_of(&toks), "Ax");
        assert_eq!(
            codes,
            vec![ParseErrorCode::MissingSemicolonAfterCharacterReference]
        );
    }

    #[test]
    fn char_ref_missing_semicolon_named() {
        let (toks, codes) = tokenize_with_errors("&ampx");
        assert_eq!(text_of(&toks), "&x");
        assert_eq!(
            codes,
            vec![ParseErrorCode::MissingSemicolonAfterCharacterReference]
        );
    }

    #[test]
    fn char_ref_absence_of_digits() {
        for input in ["&#x;", "&#;", "&#xG;"] {
            let (toks, codes) = tokenize_with_errors(input);
            assert_eq!(text_of(&toks), input, "input {input:?}");
            assert_eq!(
                codes,
                vec![ParseErrorCode::AbsenceOfDigitsInNumericCharacterReference],
                "input {input:?}"
            );
        }
    }

    #[test]
    fn char_ref_unknown_named_with_semicolon() {
        // The ambiguous-ampersand path reaching `;`: parse error, and the
        // `;` is reconsumed as literal text.
        let (toks, codes) = tokenize_with_errors("&foo;");
        assert_eq!(text_of(&toks), "&foo;");
        assert_eq!(codes, vec![ParseErrorCode::UnknownNamedCharacterReference]);
    }

    #[test]
    fn char_ref_unknown_named_without_semicolon_is_quiet() {
        // No `;` on the ambiguous-ampersand path: no parse error.
        for input in ["&foo ", "&foo"] {
            let (toks, codes) = tokenize_with_errors(input);
            assert_eq!(text_of(&toks), input, "input {input:?}");
            assert!(codes.is_empty(), "input {input:?}: {codes:?}");
        }
    }

    #[test]
    fn char_ref_bare_ampersand_is_quiet() {
        // A bare `&` never enters the named-reference state: no error.
        let (toks, codes) = tokenize_with_errors("& ");
        assert_eq!(text_of(&toks), "& ");
        assert!(codes.is_empty());
    }

    #[test]
    fn non_ascii_tag_name_column_counts_chars_not_bytes() {
        // Contract (`Tokenizer::col` field doc): columns count characters,
        // not bytes. `é` is one char / two bytes in UTF-8; the text token
        // after `<dé>` must start at char column 5 (byte counting would
        // give 6).
        let mut t = Tokenizer::new("<dé>x");
        assert!(matches!(t.next_token(), Token::StartTag(_)));
        assert_eq!(t.token_loc().col, 1); // the `<`
        assert_eq!(t.next_token(), Token::Text("x".to_string()));
        assert_eq!(t.token_loc().col, 5); // `<`=1 d=2 é=3 `>`=4 x=5
    }
}
