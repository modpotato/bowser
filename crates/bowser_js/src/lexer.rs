//! ECMAScript (ES2024) lexer.
//!
//! Produces a flat token stream plus, separately, the spans of every line
//! terminator seen between tokens (for ASI). Comments and whitespace are
//! skipped; a `#!` hashbang line at the very start of the source is skipped.
//!
//! # Contract with the parser
//!
//! * [`lex`] / [`lex_with_trivia`] tokenize `/` **always** as [`TokenKind::Slash`];
//!   the lexer never guesses regex vs. division. When the grammar expects a
//!   regular expression at a slash, the parser calls [`lex_regex`] with the
//!   slash's span start and splices the returned [`TokenKind::RegexLiteral`]
//!   in, skipping every token whose span lies inside `[slash_start, regex_end)`.
//! * Because a regex body (e.g. `/"/`) is not tokenizable as ordinary JS,
//!   batch [`lex`] can fail on sources containing such regexes. The parser
//!   should drive the incremental [`Lexer`] instead: call [`Lexer::next_token`]
//!   in a loop; at a regex position call [`lex_regex`], synthesize the token,
//!   then [`Lexer::set_pos`] to the returned end offset and continue.
//!   [`Lexer::has_line_term_between`] answers ASI queries against the
//!   line-terminator spans collected so far.
//! * `new.target` lexes as one [`TokenKind::MetaPropertyNewTarget`] token
//!   only when the three parts are contiguous (`new.target`); with any trivia
//!   between them it lexes as `New`, `Dot`, `Target` and the parser combines.
//! * `target`, `let`, `static`, `of`, `get`, `set`, `async`, `from`, `as` lex
//!   as their keyword kinds unconditionally (they are contextual in the
//!   grammar; the parser decides when they act as identifiers).
//! * `BigInt` literals lex as [`TokenKind::BigInt`] holding the literal text
//!   without the trailing `n` (e.g. `"0xFF"`, `"1_000"`).
//! * Invalid escape sequences inside templates are kept verbatim in the cooked
//!   string (tagged templates permit them; cooked is `undefined` there, which
//!   the parser derives from the raw source if needed).
//! * `\uD800`-style lone surrogates in strings become U+FFFD (Rust strings are
//!   UTF-8); in identifiers they are an error.
//!
//! Identifiers follow UAX #31 ID_Start / ID_Continue for Unicode 15.1.0 (the
//! version ES2024 references); see [`id_tables`].

use crate::id_tables::{is_id_continue, is_id_start};
use crate::wtf8::Wtf8Buf;

/// Byte offsets into the source. `end` is exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

/// One lexical token kind. `String` payloads are decoded/cooked values.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    // Literals
    Numeric(f64),
    /// Legacy octal (`0755`) or legacy octal-like decimal (`08`, `078`):
    /// same numeric value as `Numeric`, but flagged so the parser can
    /// reject it in strict mode (spec Annex B.1.1). `0o17` and friends
    /// lex as plain `Numeric`.
    NumericOctal(f64),
    BigInt(String),
    /// String literal value as WTF-8: `\uD800` escapes produce lone
    /// surrogates, which `String` cannot hold.
    Str(Wtf8Buf),
    TemplateHead(Wtf8Buf),
    TemplateMiddle(Wtf8Buf),
    TemplateTail(Wtf8Buf),
    /// Raw source of the regex including slashes and flags, e.g. `"/ab+c/gi"`.
    /// Only produced by [`lex_regex`], never by [`lex`].
    RegexLiteral(String),
    Ident(String),
    PrivateIdent(String),
    // Reserved words
    Await,
    Break,
    Case,
    Catch,
    Class,
    Const,
    Continue,
    Debugger,
    Default,
    Delete,
    Do,
    Else,
    Enum,
    Export,
    Extends,
    False,
    Finally,
    For,
    Function,
    If,
    Import,
    In,
    Instanceof,
    New,
    Null,
    Return,
    Super,
    Switch,
    This,
    Throw,
    True,
    Try,
    Typeof,
    Var,
    Void,
    While,
    With,
    Yield,
    // Contextual keywords (emitted as keyword kinds unconditionally)
    Let,
    Static,
    Of,
    Get,
    Set,
    Async,
    From,
    As,
    Target,
    /// Contiguous `new.target` only; see module docs.
    MetaPropertyNewTarget,
    // Punctuators
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Dot,
    Semicolon,
    Comma,
    Colon,
    Question,
    Lt,
    Gt,
    LtEq,
    GtEq,
    EqEq,
    NotEq,
    EqEqEq,
    NotEqEq,
    Plus,
    Minus,
    Star,
    Percent,
    StarStar,
    PlusPlus,
    MinusMinus,
    LtLt,
    GtGt,
    GtGtGt,
    Amp,
    Pipe,
    Caret,
    Bang,
    Tilde,
    AmpAmp,
    PipePipe,
    QuestionQuestion,
    Eq,
    PlusEq,
    MinusEq,
    StarEq,
    PercentEq,
    StarStarEq,
    LtLtEq,
    GtGtEq,
    GtGtGtEq,
    AmpEq,
    PipeEq,
    CaretEq,
    AmpAmpEq,
    PipePipeEq,
    QuestionQuestionEq,
    Arrow,
    Ellipsis,
    OptChain,
    Slash,
    SlashEq,
    /// `@` — decorator marker (ES2022+). The parser does not implement
    /// decorators yet; the token exists so decorator syntax lexes.
    At,
    Eof,
}

/// A token with its source span.
#[derive(Clone, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

/// Lexical errors. Offsets are byte offsets into the source.
#[derive(Debug)]
pub enum LexError {
    UnexpectedChar(char, u32),
    UnterminatedString(u32),
    UnterminatedTemplate(u32),
    UnterminatedComment(u32),
    UnterminatedRegex(u32),
    InvalidNumber(u32),
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LexError::UnexpectedChar(c, off) => {
                write!(f, "unexpected character {c:?} at offset {off}")
            }
            LexError::UnterminatedString(off) => {
                write!(f, "unterminated string literal at offset {off}")
            }
            LexError::UnterminatedTemplate(off) => {
                write!(f, "unterminated template literal at offset {off}")
            }
            LexError::UnterminatedComment(off) => {
                write!(f, "unterminated comment at offset {off}")
            }
            LexError::UnterminatedRegex(off) => {
                write!(f, "unterminated regular expression at offset {off}")
            }
            LexError::InvalidNumber(off) => {
                write!(f, "invalid numeric literal at offset {off}")
            }
        }
    }
}

impl std::error::Error for LexError {}

/// Tokenize `src` into a flat token vector, terminated by [`TokenKind::Eof`].
#[cfg(test)]
pub fn lex(src: &str) -> Result<Vec<Token>, LexError> {
    lex_with_trivia(src).0
}

/// Like [`lex`], but also returns the spans of every line terminator found
/// between tokens (including ones inside block comments), for ASI.
/// On error, the spans collected before the failure are still returned.
#[cfg(test)]
pub fn lex_with_trivia(src: &str) -> (Result<Vec<Token>, LexError>, Vec<Span>) {
    let mut lx = Lexer::new(src);
    let mut toks = Vec::new();
    loop {
        match lx.next_token() {
            Ok(t) => {
                let done = t.kind == TokenKind::Eof;
                toks.push(t);
                if done {
                    break;
                }
            }
            Err(e) => return (Err(e), lx.line_terms),
        }
    }
    (Ok(toks), lx.line_terms)
}

/// Lex a regular expression literal starting at byte offset `start`
/// (which must be the `/`).
///
/// Returns the [`TokenKind::RegexLiteral`] (raw source including slashes and
/// flags) and the exclusive end offset. The parser is expected to call this
/// when the grammar calls for a regex at a [`TokenKind::Slash`] token, then
/// continue lexing after the returned offset.
pub fn lex_regex(src: &str, start: u32) -> Result<(TokenKind, u32), LexError> {
    let bytes = src.as_bytes();
    let mut pos = start as usize;
    match bytes.get(pos) {
        Some(b'/') => {}
        _ => {
            let c = src[pos..].chars().next().unwrap_or('\u{FFFD}');
            return Err(LexError::UnexpectedChar(c, start));
        }
    }
    pos += 1;
    let mut in_class = false;
    loop {
        if pos >= bytes.len() {
            return Err(LexError::UnterminatedRegex(start));
        }
        match bytes[pos] {
            b'\\' => {
                pos += 1;
                if pos >= bytes.len() {
                    return Err(LexError::UnterminatedRegex(start));
                }
                let c = src[pos..].chars().next().unwrap_or('\u{FFFD}');
                if matches!(c, '\r' | '\n' | '\u{2028}' | '\u{2029}') {
                    return Err(LexError::UnterminatedRegex(start));
                }
                pos += c.len_utf8();
            }
            b'[' => {
                in_class = true;
                pos += 1;
            }
            b']' => {
                in_class = false;
                pos += 1;
            }
            b'/' if !in_class => {
                pos += 1;
                break;
            }
            b'\r' | b'\n' => return Err(LexError::UnterminatedRegex(start)),
            _ => {
                let c = src[pos..].chars().next().unwrap_or('\u{FFFD}');
                if c == '\u{2028}' || c == '\u{2029}' {
                    return Err(LexError::UnterminatedRegex(start));
                }
                pos += c.len_utf8();
            }
        }
    }
    // Flags: identifier parts. Anything else ends the literal; the parser
    // validates the flag text itself.
    while pos < bytes.len() {
        let c = src[pos..].chars().next().unwrap_or('\u{FFFD}');
        if c.is_ascii_alphanumeric()
            || c == '$'
            || c == '_'
            || c == '\u{200C}'
            || c == '\u{200D}'
            || is_id_continue(c)
        {
            pos += c.len_utf8();
        } else {
            break;
        }
    }
    let end = pos as u32;
    Ok((
        TokenKind::RegexLiteral(src[start as usize..pos].to_string()),
        end,
    ))
}

/// Inline scratch buffer for short ASCII text (number literals). Stays on
/// the stack up to 64 bytes and spills to the heap for longer input, so
/// the common case allocates nothing.
struct NumBuf {
    stack: [u8; 64],
    len: usize,
    heap: Option<Vec<u8>>,
}

impl NumBuf {
    fn new() -> Self {
        NumBuf {
            stack: [0; 64],
            len: 0,
            heap: None,
        }
    }

    fn push(&mut self, b: u8) {
        if let Some(h) = &mut self.heap {
            h.push(b);
            return;
        }
        if self.len < self.stack.len() {
            self.stack[self.len] = b;
            self.len += 1;
        } else {
            let mut h = Vec::with_capacity(128);
            h.extend_from_slice(&self.stack[..self.len]);
            h.push(b);
            self.heap = Some(h);
        }
    }

    fn push_str(&mut self, s: &str) {
        for b in s.bytes() {
            self.push(b);
        }
    }

    fn as_str(&self) -> &str {
        if let Some(h) = &self.heap {
            std::str::from_utf8(h).expect("NumBuf holds ASCII")
        } else {
            std::str::from_utf8(&self.stack[..self.len]).expect("NumBuf holds ASCII")
        }
    }
}

/// Incremental lexer. The parser drives this token by token so it can
/// intervene for regex literals (see module docs).
#[derive(Debug)]
pub struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    brace_depth: u32,
    /// `(brace depth, template token start)` recorded at each open `${`;
    /// used to tell a template-closing `}` apart from an ordinary `}`.
    tmpl_stack: Vec<(u32, u32)>,
    line_terms: Vec<Span>,
    /// True when no token has been emitted since the last line terminator
    /// (or start of input); used for Annex B `-->` comments.
    line_start: bool,
}

impl<'a> Lexer<'a> {
    /// Create a lexer over `src`. A leading `#!` hashbang line is skipped.
    pub fn new(src: &'a str) -> Self {
        let mut lx = Lexer {
            src,
            bytes: src.as_bytes(),
            pos: 0,
            brace_depth: 0,
            tmpl_stack: Vec::new(),
            line_terms: Vec::new(),
            line_start: true,
        };
        if lx.bytes.starts_with(b"#!") {
            while lx.pos < lx.bytes.len() && !lx.is_line_term_at(lx.pos) {
                lx.pos += lx.char_at(lx.pos).len_utf8();
            }
        }
        lx
    }

    /// Reposition the lexer. Only supported use: jumping past a regex literal
    /// to an end offset returned by [`lex_regex`] for the slash token just
    /// produced. Template/brace state is left untouched, which is correct for
    /// that use because a regex body contains no line terminators.
    pub fn set_pos(&mut self, pos: u32) {
        self.pos = pos as usize;
    }

    /// True if a line-terminator span lies entirely after `prev_end` and at
    /// or before `cur_start`. Spans are recorded in ascending source order,
    /// so this is a binary search instead of a linear scan.
    pub fn has_line_term_between(&self, prev_end: u32, cur_start: u32) -> bool {
        let terms = &self.line_terms;
        let i = terms.partition_point(|s| s.start < prev_end);
        matches!(terms.get(i), Some(s) if s.end <= cur_start)
    }

    /// Lex the next token. At end of input returns [`TokenKind::Eof`]
    /// (repeatedly).
    pub fn next_token(&mut self) -> Result<Token, LexError> {
        self.skip_trivia()?;
        if self.pos >= self.bytes.len() {
            // An unclosed `${...}` at end of input is an unterminated template.
            if let Some(&(_, tstart)) = self.tmpl_stack.last() {
                return Err(LexError::UnterminatedTemplate(tstart));
            }
            let p = self.pos as u32;
            return Ok(Token {
                kind: TokenKind::Eof,
                span: Span { start: p, end: p },
            });
        }
        let start = self.pos as u32;
        let b = self.bytes[self.pos];
        self.line_start = false;
        match b {
            b'{' => {
                self.brace_depth += 1;
                Ok(self.take(start, 1, TokenKind::LBrace))
            }
            b'}' => self.lex_rbrace(start),
            b'(' => Ok(self.take(start, 1, TokenKind::LParen)),
            b')' => Ok(self.take(start, 1, TokenKind::RParen)),
            b'[' => Ok(self.take(start, 1, TokenKind::LBracket)),
            b']' => Ok(self.take(start, 1, TokenKind::RBracket)),
            b';' => Ok(self.take(start, 1, TokenKind::Semicolon)),
            b',' => Ok(self.take(start, 1, TokenKind::Comma)),
            b':' => Ok(self.take(start, 1, TokenKind::Colon)),
            b'~' => Ok(self.take(start, 1, TokenKind::Tilde)),
            b'@' => Ok(self.take(start, 1, TokenKind::At)),
            b'.' => {
                if self.at(b"...") {
                    Ok(self.take(start, 3, TokenKind::Ellipsis))
                } else if matches!(self.bytes.get(self.pos + 1), Some(b'0'..=b'9')) {
                    self.lex_number(start)
                } else {
                    Ok(self.take(start, 1, TokenKind::Dot))
                }
            }
            b'?' => {
                if self.at(b"??=") {
                    Ok(self.take(start, 3, TokenKind::QuestionQuestionEq))
                } else if self.at(b"??") {
                    Ok(self.take(start, 2, TokenKind::QuestionQuestion))
                } else if self.at(b"?.") {
                    // `?.` not followed by a digit; `a?.5:b` is `?` `.5` `:`.
                    if matches!(self.bytes.get(self.pos + 2), Some(b'0'..=b'9')) {
                        Ok(self.take(start, 1, TokenKind::Question))
                    } else {
                        Ok(self.take(start, 2, TokenKind::OptChain))
                    }
                } else {
                    Ok(self.take(start, 1, TokenKind::Question))
                }
            }
            b'=' => {
                if self.at(b"===") {
                    Ok(self.take(start, 3, TokenKind::EqEqEq))
                } else if self.at(b"==") {
                    Ok(self.take(start, 2, TokenKind::EqEq))
                } else if self.at(b"=>") {
                    Ok(self.take(start, 2, TokenKind::Arrow))
                } else {
                    Ok(self.take(start, 1, TokenKind::Eq))
                }
            }
            b'!' => {
                if self.at(b"!==") {
                    Ok(self.take(start, 3, TokenKind::NotEqEq))
                } else if self.at(b"!=") {
                    Ok(self.take(start, 2, TokenKind::NotEq))
                } else {
                    Ok(self.take(start, 1, TokenKind::Bang))
                }
            }
            b'<' => {
                if self.at(b"<<=") {
                    Ok(self.take(start, 3, TokenKind::LtLtEq))
                } else if self.at(b"<<") {
                    Ok(self.take(start, 2, TokenKind::LtLt))
                } else if self.at(b"<=") {
                    Ok(self.take(start, 2, TokenKind::LtEq))
                } else {
                    Ok(self.take(start, 1, TokenKind::Lt))
                }
            }
            b'>' => {
                if self.at(b">>>=") {
                    Ok(self.take(start, 4, TokenKind::GtGtGtEq))
                } else if self.at(b">>>") {
                    Ok(self.take(start, 3, TokenKind::GtGtGt))
                } else if self.at(b">>=") {
                    Ok(self.take(start, 3, TokenKind::GtGtEq))
                } else if self.at(b">>") {
                    Ok(self.take(start, 2, TokenKind::GtGt))
                } else if self.at(b">=") {
                    Ok(self.take(start, 2, TokenKind::GtEq))
                } else {
                    Ok(self.take(start, 1, TokenKind::Gt))
                }
            }
            b'+' => {
                if self.at(b"++") {
                    Ok(self.take(start, 2, TokenKind::PlusPlus))
                } else if self.at(b"+=") {
                    Ok(self.take(start, 2, TokenKind::PlusEq))
                } else {
                    Ok(self.take(start, 1, TokenKind::Plus))
                }
            }
            b'-' => {
                if self.at(b"--") {
                    Ok(self.take(start, 2, TokenKind::MinusMinus))
                } else if self.at(b"-=") {
                    Ok(self.take(start, 2, TokenKind::MinusEq))
                } else {
                    Ok(self.take(start, 1, TokenKind::Minus))
                }
            }
            b'*' => {
                if self.at(b"**=") {
                    Ok(self.take(start, 3, TokenKind::StarStarEq))
                } else if self.at(b"**") {
                    Ok(self.take(start, 2, TokenKind::StarStar))
                } else if self.at(b"*=") {
                    Ok(self.take(start, 2, TokenKind::StarEq))
                } else {
                    Ok(self.take(start, 1, TokenKind::Star))
                }
            }
            b'%' => {
                if self.at(b"%=") {
                    Ok(self.take(start, 2, TokenKind::PercentEq))
                } else {
                    Ok(self.take(start, 1, TokenKind::Percent))
                }
            }
            b'/' => {
                // `//` and `/*` are consumed by skip_trivia; `-->` never
                // reaches here. What remains is division or `/=`.
                if self.at(b"/=") {
                    Ok(self.take(start, 2, TokenKind::SlashEq))
                } else {
                    Ok(self.take(start, 1, TokenKind::Slash))
                }
            }
            b'&' => {
                if self.at(b"&&=") {
                    Ok(self.take(start, 3, TokenKind::AmpAmpEq))
                } else if self.at(b"&&") {
                    Ok(self.take(start, 2, TokenKind::AmpAmp))
                } else if self.at(b"&=") {
                    Ok(self.take(start, 2, TokenKind::AmpEq))
                } else {
                    Ok(self.take(start, 1, TokenKind::Amp))
                }
            }
            b'|' => {
                if self.at(b"||=") {
                    Ok(self.take(start, 3, TokenKind::PipePipeEq))
                } else if self.at(b"||") {
                    Ok(self.take(start, 2, TokenKind::PipePipe))
                } else if self.at(b"|=") {
                    Ok(self.take(start, 2, TokenKind::PipeEq))
                } else {
                    Ok(self.take(start, 1, TokenKind::Pipe))
                }
            }
            b'^' => {
                if self.at(b"^=") {
                    Ok(self.take(start, 2, TokenKind::CaretEq))
                } else {
                    Ok(self.take(start, 1, TokenKind::Caret))
                }
            }
            b'\'' | b'"' => self.lex_string(start),
            b'`' => self.lex_template(start),
            b'#' => self.lex_private_ident(start),
            b'\\' => self.lex_ident(start),
            _ => {
                let c = self.char_at(self.pos);
                if c.is_ascii_digit() {
                    self.lex_number(start)
                } else if is_id_start(c) || c == '$' || c == '_' {
                    self.lex_ident(start)
                } else {
                    Err(LexError::UnexpectedChar(c, start))
                }
            }
        }
    }

    // ---------- helpers ----------

    #[inline]
    fn at(&self, s: &[u8]) -> bool {
        self.bytes[self.pos..].starts_with(s)
    }

    #[inline]
    fn char_at(&self, pos: usize) -> char {
        self.src[pos..].chars().next().unwrap_or('\u{FFFD}')
    }

    #[inline]
    fn take(&mut self, start: u32, n: usize, kind: TokenKind) -> Token {
        self.pos += n;
        Token {
            kind,
            span: Span {
                start,
                end: self.pos as u32,
            },
        }
    }

    fn is_line_term_at(&self, pos: usize) -> bool {
        match self.bytes[pos] {
            b'\r' | b'\n' => true,
            _ => matches!(self.char_at(pos), '\u{2028}' | '\u{2029}'),
        }
    }

    /// Consume one line terminator (`\n`, `\r`, `\r\n`, `\u{2028}`, `\u{2029}`)
    /// at the current position.
    fn consume_newline(&mut self) {
        if self.bytes[self.pos] == b'\r' && self.bytes.get(self.pos + 1) == Some(&b'\n') {
            self.pos += 2;
        } else {
            // Advance by the full char width: `\u{2028}`/`\u{2029}` are
            // 3-byte chars, and landing mid-char panics on the next slice.
            self.pos += self.char_at(self.pos).len_utf8();
        }
    }

    fn push_line_term(&mut self, start: u32) {
        let end = self.pos as u32;
        self.line_terms.push(Span { start, end });
        self.line_start = true;
    }

    // ---------- trivia ----------

    /// Skip whitespace and comments, recording line-terminator spans.
    fn skip_trivia(&mut self) -> Result<(), LexError> {
        loop {
            if self.pos >= self.bytes.len() {
                return Ok(());
            }
            match self.bytes[self.pos] {
                b' ' | b'\t' | 0x0B | 0x0C => {
                    self.pos += 1;
                }
                b'\r' | b'\n' => {
                    let s = self.pos as u32;
                    self.consume_newline();
                    self.push_line_term(s);
                }
                b'/' => match self.bytes.get(self.pos + 1) {
                    Some(b'/') => {
                        self.pos += 2;
                        while self.pos < self.bytes.len() && !self.is_line_term_at(self.pos) {
                            self.pos += self.char_at(self.pos).len_utf8();
                        }
                    }
                    Some(b'*') => {
                        let cs = self.pos as u32;
                        self.pos += 2;
                        loop {
                            if self.pos >= self.bytes.len() {
                                return Err(LexError::UnterminatedComment(cs));
                            }
                            if self.bytes[self.pos] == b'*'
                                && self.bytes.get(self.pos + 1) == Some(&b'/')
                            {
                                self.pos += 2;
                                break;
                            }
                            if self.is_line_term_at(self.pos) {
                                // Line terminators inside block comments count
                                // for ASI, so record them.
                                let s = self.pos as u32;
                                self.consume_newline();
                                self.push_line_term(s);
                            } else {
                                self.pos += self.char_at(self.pos).len_utf8();
                            }
                        }
                    }
                    _ => return Ok(()),
                },
                // Annex B: `<!--` starts a single-line comment.
                b'<' if self.at(b"<!--") => {
                    self.pos += 4;
                    while self.pos < self.bytes.len() && !self.is_line_term_at(self.pos) {
                        self.pos += self.char_at(self.pos).len_utf8();
                    }
                }
                // Annex B: `-->` is a single-line comment only at line start.
                b'-' if self.line_start && self.at(b"-->") => {
                    self.pos += 3;
                    while self.pos < self.bytes.len() && !self.is_line_term_at(self.pos) {
                        self.pos += self.char_at(self.pos).len_utf8();
                    }
                }
                b if b < 0x80 => return Ok(()),
                _ => {
                    let c = self.char_at(self.pos);
                    if c == '\u{FEFF}'
                        || matches!(
                            c,
                            '\u{00A0}'
                                | '\u{1680}'
                                | '\u{202F}'
                                | '\u{205F}'
                                | '\u{3000}'
                                | '\u{2000}'..='\u{200A}'
                        )
                    {
                        self.pos += c.len_utf8();
                    } else if c == '\u{2028}' || c == '\u{2029}' {
                        let s = self.pos as u32;
                        self.pos += c.len_utf8();
                        self.push_line_term(s);
                    } else {
                        return Ok(());
                    }
                }
            }
        }
    }

    // ---------- braces and templates ----------

    fn lex_rbrace(&mut self, start: u32) -> Result<Token, LexError> {
        self.pos += 1;
        // A `}` at the recorded brace depth closes a `${...}` substitution
        // and resumes template lexing instead of producing an RBrace.
        if let Some(&(depth, _)) = self.tmpl_stack.last()
            && depth == self.brace_depth
        {
            self.tmpl_stack.pop();
            return self.template_parts(start, false);
        }
        if self.brace_depth > 0 {
            self.brace_depth -= 1;
        }
        Ok(Token {
            kind: TokenKind::RBrace,
            span: Span {
                start,
                end: self.pos as u32,
            },
        })
    }

    /// Lex a template starting at a backtick.
    fn lex_template(&mut self, start: u32) -> Result<Token, LexError> {
        self.pos += 1; // consume `
        self.template_parts(start, true)
    }

    /// Lex the cooked characters of a template (or template continuation)
    /// until the closing backtick or a `${`.
    fn template_parts(&mut self, start: u32, head: bool) -> Result<Token, LexError> {
        let mut cooked = Wtf8Buf::new();
        loop {
            if self.pos >= self.bytes.len() {
                return Err(LexError::UnterminatedTemplate(start));
            }
            match self.bytes[self.pos] {
                b'`' => {
                    self.pos += 1;
                    return Ok(Token {
                        kind: TokenKind::TemplateTail(cooked),
                        span: Span {
                            start,
                            end: self.pos as u32,
                        },
                    });
                }
                b'$' if self.bytes.get(self.pos + 1) == Some(&b'{') => {
                    self.pos += 2;
                    self.tmpl_stack.push((self.brace_depth, start));
                    let kind = if head {
                        TokenKind::TemplateHead(cooked)
                    } else {
                        TokenKind::TemplateMiddle(cooked)
                    };
                    return Ok(Token {
                        kind,
                        span: Span {
                            start,
                            end: self.pos as u32,
                        },
                    });
                }
                b'\\' => self.lex_escape(&mut cooked, start, true)?,
                b'\r' | b'\n' => {
                    // Line terminators inside a template are part of the token
                    // (normalized to \n), never ASI trivia.
                    cooked.push_char('\n');
                    self.consume_newline();
                }
                _ => {
                    let c = self.char_at(self.pos);
                    cooked.push_char(c);
                    self.pos += c.len_utf8();
                }
            }
        }
    }

    // ---------- strings ----------

    fn lex_string(&mut self, start: u32) -> Result<Token, LexError> {
        let quote = self.bytes[self.pos];
        // Fast path: scan for the closing quote. With no backslash, line
        // break, or non-ASCII byte, the cooked value is the source slice.
        let bytes = self.bytes;
        let mut end = self.pos + 1;
        while end < bytes.len() {
            let b = bytes[end];
            if b == quote {
                let s = Wtf8Buf::from(&self.src[self.pos + 1..end]);
                self.pos = end + 1;
                return Ok(Token {
                    kind: TokenKind::Str(s),
                    span: Span {
                        start,
                        end: self.pos as u32,
                    },
                });
            }
            if b == b'\\' || b == b'\r' || b == b'\n' || b >= 0x80 {
                break;
            }
            end += 1;
        }
        self.pos += 1;
        let mut s = Wtf8Buf::new();
        loop {
            if self.pos >= self.bytes.len() {
                return Err(LexError::UnterminatedString(start));
            }
            match self.bytes[self.pos] {
                b if b == quote => {
                    self.pos += 1;
                    break;
                }
                b'\\' => self.lex_escape(&mut s, start, false)?,
                b'\r' | b'\n' => return Err(LexError::UnterminatedString(start)),
                _ => {
                    let c = self.char_at(self.pos);
                    if c == '\u{2028}' || c == '\u{2029}' {
                        return Err(LexError::UnterminatedString(start));
                    }
                    s.push_char(c);
                    self.pos += c.len_utf8();
                }
            }
        }
        Ok(Token {
            kind: TokenKind::Str(s),
            span: Span {
                start,
                end: self.pos as u32,
            },
        })
    }

    /// Lex one escape sequence; the current position is at the backslash.
    /// In `template` mode (tagged templates may contain invalid escapes) an
    /// invalid escape is kept verbatim instead of erroring.
    fn lex_escape(
        &mut self,
        out: &mut Wtf8Buf,
        start: u32,
        template: bool,
    ) -> Result<(), LexError> {
        self.pos += 1; // consume `\`
        if self.pos >= self.bytes.len() {
            return Err(if template {
                LexError::UnterminatedTemplate(start)
            } else {
                LexError::UnterminatedString(start)
            });
        }
        match self.bytes[self.pos] {
            b'b' => {
                out.push_char('\u{8}');
                self.pos += 1;
            }
            b'f' => {
                out.push_char('\u{C}');
                self.pos += 1;
            }
            b'n' => {
                out.push_char('\n');
                self.pos += 1;
            }
            b'r' => {
                out.push_char('\r');
                self.pos += 1;
            }
            b't' => {
                out.push_char('\t');
                self.pos += 1;
            }
            b'v' => {
                out.push_char('\u{B}');
                self.pos += 1;
            }
            b'\'' | b'"' | b'\\' => {
                out.push_char(self.bytes[self.pos] as char);
                self.pos += 1;
            }
            // Line continuation: backslash + line terminator vanishes.
            b'\r' | b'\n' => self.consume_newline(),
            // Legacy octal escapes (Annex B).
            b'0'..=b'7' => {
                let first = self.bytes[self.pos] - b'0';
                self.pos += 1;
                let mut v = first as u32;
                // `\0` not followed by an octal digit is NUL; `\08` leaves
                // the `8` for normal processing.
                let extra = if first <= 3 { 2 } else { 1 };
                for _ in 0..extra {
                    match self.bytes.get(self.pos) {
                        Some(d @ b'0'..=b'7') => {
                            v = v * 8 + (d - b'0') as u32;
                            self.pos += 1;
                        }
                        _ => break,
                    }
                }
                // Octal escapes max out at 0o377: never a surrogate.
                out.push_unit(v as u16);
            }
            // `\8` and `\9` are lenient in engines (Annex-B-style).
            b'8' | b'9' => {
                out.push_char(self.bytes[self.pos] as char);
                self.pos += 1;
            }
            b'x' => {
                let h1 = self.hex_val_at(self.pos + 1);
                let h2 = self.hex_val_at(self.pos + 2);
                match (h1, h2) {
                    (Some(a), Some(b2)) => {
                        // Two hex digits: at most 0xFF, never a surrogate.
                        out.push_unit((a * 16 + b2) as u16);
                        self.pos += 3;
                    }
                    _ if template => {
                        // Keep the escape verbatim: only the backslash is
                        // consumed here; `x` is processed normally next.
                        out.push_char('\\');
                    }
                    _ => {
                        let off = if h1.is_none() {
                            self.pos + 1
                        } else {
                            self.pos + 2
                        };
                        return Err(LexError::UnexpectedChar(self.char_at(off), off as u32));
                    }
                }
            }
            b'u' => match self.unicode_escape(self.pos) {
                Ok((v, np)) => {
                    self.pos = np;
                    if v <= 0xFFFF {
                        // Includes lone surrogates: WTF-8 preserves them
                        // instead of replacing with U+FFFD.
                        out.push_unit(v as u16);
                    } else {
                        let v = v - 0x10000;
                        out.push_unit(0xD800 | (v >> 10) as u16);
                        out.push_unit(0xDC00 | (v & 0x3FF) as u16);
                    }
                }
                Err((c, off)) => {
                    if template {
                        // Keep the escape verbatim: only the backslash is
                        // consumed here; `u` is processed normally next.
                        out.push_char('\\');
                    } else {
                        return Err(LexError::UnexpectedChar(c, off));
                    }
                }
            },
            _ => {
                let c = self.char_at(self.pos);
                if c == '\u{2028}' || c == '\u{2029}' {
                    // Line continuation.
                    self.pos += c.len_utf8();
                } else {
                    out.push_char(c);
                    self.pos += c.len_utf8();
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn hex_val_at(&self, pos: usize) -> Option<u32> {
        match self.bytes.get(pos) {
            Some(b'0'..=b'9') => Some((self.bytes[pos] - b'0') as u32),
            Some(b'a'..=b'f') => Some((self.bytes[pos] - b'a' + 10) as u32),
            Some(b'A'..=b'F') => Some((self.bytes[pos] - b'A' + 10) as u32),
            _ => None,
        }
    }

    /// Scan `\uHHHH` or `\u{H...}` with `pos` at the `u`.
    /// Returns the value and the position past the escape, or the offending
    /// character and its offset.
    fn unicode_escape(&self, pos: usize) -> Result<(u32, usize), (char, u32)> {
        let err = |p: usize| -> (char, u32) {
            let c = if p < self.bytes.len() {
                self.char_at(p)
            } else {
                '\u{FFFD}'
            };
            (c, p as u32)
        };
        let mut p = pos + 1;
        if self.bytes.get(p) == Some(&b'{') {
            p += 1;
            let mut v: u32 = 0;
            let mut n = 0;
            while let Some(d) = self.hex_val_at(p) {
                v = v * 16 + d;
                n += 1;
                p += 1;
                if v > 0x10FFFF {
                    return Err(err(p));
                }
            }
            if n == 0 || self.bytes.get(p) != Some(&b'}') {
                return Err(err(p));
            }
            Ok((v, p + 1))
        } else {
            let mut v: u32 = 0;
            for _ in 0..4 {
                match self.hex_val_at(p) {
                    Some(d) => {
                        v = v * 16 + d;
                        p += 1;
                    }
                    None => return Err(err(p)),
                }
            }
            Ok((v, p))
        }
    }

    // ---------- numbers ----------

    /// Lex a numeric literal. Entry: at an ASCII digit, or at `.` followed by
    /// an ASCII digit (checked by the caller).
    fn lex_number(&mut self, start: u32) -> Result<Token, LexError> {
        // Separator-free digit text for `f64` parsing. Numbers are short,
        // so this stays on the stack unless separators make the literal
        // much longer than its digit count.
        let mut raw = NumBuf::new();
        let mut is_float = false;

        if self.bytes[self.pos] == b'.' {
            raw.push(b'.');
            self.pos += 1;
            let n = self.scan_dec_digits(&mut raw, start)?;
            debug_assert!(n >= 1);
            is_float = true;
        } else if self.bytes[self.pos] == b'0' {
            match self.bytes.get(self.pos + 1) {
                Some(b'x') | Some(b'X') => return self.lex_radix(start, 16),
                Some(b'b') | Some(b'B') => return self.lex_radix(start, 2),
                Some(b'o') | Some(b'O') => return self.lex_radix(start, 8),
                Some(b'0'..=b'9') => {
                    // Legacy octal (`0755` = 493) or legacy decimal-like
                    // (`078` = 78, `08` = 8): scan the whole digit run, then
                    // decide. Separators are rejected by engines here, and no
                    // `n` suffix is allowed (`0755n` is a SyntaxError).
                    let mut v_oct: f64 = 0.0;
                    let mut has89 = false;
                    raw.push(b'0');
                    self.pos += 1;
                    while let Some(d @ b'0'..=b'9') = self.bytes.get(self.pos) {
                        raw.push(*d);
                        if *d >= b'8' {
                            has89 = true;
                        } else {
                            v_oct = v_oct * 8.0 + (*d - b'0') as f64;
                        }
                        self.pos += 1;
                    }
                    let v = if has89 {
                        Self::parse_dec(raw.as_str(), start)?
                    } else {
                        v_oct
                    };
                    return Ok(self.num_tok(TokenKind::NumericOctal(v), start));
                }
                _ => {
                    raw.push(b'0');
                    self.pos += 1;
                    // `0_1` is an error in engines; lexing `0` then `_1` as an
                    // identifier reproduces that at parse time.
                    if self.bytes.get(self.pos) == Some(&b'n') {
                        self.pos += 1;
                        return Ok(self.bigint_tok(start));
                    }
                }
            }
        } else {
            raw.push(self.bytes[self.pos]);
            self.pos += 1;
            self.scan_dec_digits(&mut raw, start)?;
        }

        if self.bytes.get(self.pos) == Some(&b'.') {
            is_float = true;
            raw.push(b'.');
            self.pos += 1;
            self.scan_dec_digits(&mut raw, start)?;
        }
        if matches!(self.bytes.get(self.pos), Some(b'e') | Some(b'E')) {
            let save = self.pos;
            self.pos += 1;
            let mut exp = NumBuf::new();
            if matches!(self.bytes.get(self.pos), Some(b'+') | Some(b'-')) {
                exp.push(self.bytes[self.pos]);
                self.pos += 1;
            }
            if self.scan_dec_digits(&mut exp, start)? == 0 {
                // `1e` is `1` followed by identifier `e`.
                self.pos = save;
            } else {
                is_float = true;
                raw.push(b'e');
                raw.push_str(exp.as_str());
            }
        }
        if !is_float && self.bytes.get(self.pos) == Some(&b'n') {
            self.pos += 1;
            return Ok(self.bigint_tok(start));
        }
        let v = Self::parse_dec(raw.as_str(), start)?;
        Ok(self.num_tok(TokenKind::Numeric(v), start))
    }

    /// Lex `0x`/`0b`/`0o` (and capitals). Entry: at the `0`.
    fn lex_radix(&mut self, start: u32, radix: u32) -> Result<Token, LexError> {
        self.pos += 2;
        let mut v: f64 = 0.0;
        let mut any = false;
        loop {
            match self.bytes.get(self.pos) {
                Some(b'_') => {
                    if self.digit_val(self.pos + 1, radix).is_none() {
                        return Err(LexError::InvalidNumber(start));
                    }
                    self.pos += 1;
                }
                _ => match self.digit_val(self.pos, radix) {
                    Some(d) => {
                        v = v * radix as f64 + d as f64;
                        any = true;
                        self.pos += 1;
                    }
                    None => break,
                },
            }
        }
        if !any {
            return Err(LexError::InvalidNumber(start));
        }
        if self.bytes.get(self.pos) == Some(&b'n') {
            self.pos += 1;
            return Ok(self.bigint_tok(start));
        }
        Ok(self.num_tok(TokenKind::Numeric(v), start))
    }

    /// Scan ASCII decimal digits with `_` separators, appending the digits
    /// (not the separators) to `raw`. Returns the digit count. A separator
    /// not followed by a digit is an error.
    fn scan_dec_digits(&mut self, raw: &mut NumBuf, start: u32) -> Result<usize, LexError> {
        let mut n = 0;
        loop {
            match self.bytes.get(self.pos) {
                Some(b'0'..=b'9') => {
                    raw.push(self.bytes[self.pos]);
                    self.pos += 1;
                    n += 1;
                }
                Some(b'_') => {
                    if !matches!(self.bytes.get(self.pos + 1), Some(b'0'..=b'9')) {
                        return Err(LexError::InvalidNumber(start));
                    }
                    self.pos += 1;
                }
                _ => break,
            }
        }
        Ok(n)
    }

    fn digit_val(&self, pos: usize, radix: u32) -> Option<u32> {
        let b = *self.bytes.get(pos)?;
        let d = match b {
            b'0'..=b'9' => (b - b'0') as u32,
            b'a'..=b'f' => (b - b'a' + 10) as u32,
            b'A'..=b'F' => (b - b'A' + 10) as u32,
            _ => return None,
        };
        if d < radix { Some(d) } else { None }
    }

    fn parse_dec(raw: &str, start: u32) -> Result<f64, LexError> {
        raw.parse::<f64>()
            .map_err(|_| LexError::InvalidNumber(start))
    }

    fn num_tok(&self, kind: TokenKind, start: u32) -> Token {
        Token {
            kind,
            span: Span {
                start,
                end: self.pos as u32,
            },
        }
    }

    /// Build a BigInt token; the literal text excludes the trailing `n`.
    fn bigint_tok(&self, start: u32) -> Token {
        let end = self.pos as u32;
        let text = self.src[start as usize..(end - 1) as usize].to_string();
        Token {
            kind: TokenKind::BigInt(text),
            span: Span { start, end },
        }
    }

    // ---------- identifiers ----------

    /// Lex an identifier or keyword. Entry: at an IdentifierStart char, `$`,
    /// `_`, or a `\u` escape.
    fn lex_ident(&mut self, start: u32) -> Result<Token, LexError> {
        // Fast path: pure-ASCII identifier with no escapes. Scan the byte
        // run, then resolve the keyword from the source slice: one
        // allocation for a real identifier, none for a keyword.
        if self.bytes[self.pos] != b'\\' {
            let bytes = self.bytes;
            let mut end = self.pos;
            while end < bytes.len()
                && matches!(
                    bytes[end],
                    b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'$' | b'_'
                )
            {
                end += 1;
            }
            if end == bytes.len() || (bytes[end] != b'\\' && bytes[end] < 0x80) {
                let word = &self.src[self.pos..end];
                self.pos = end;
                let mut kind = keyword_or_ident(word);
                // Contiguous `new.target` lexes as one metaproperty token.
                if kind == TokenKind::New
                    && self.src[self.pos..].starts_with(".target")
                    && !self.ident_part_at(self.pos + 7)
                {
                    self.pos += 7;
                    kind = TokenKind::MetaPropertyNewTarget;
                }
                return Ok(Token {
                    kind,
                    span: Span {
                        start,
                        end: self.pos as u32,
                    },
                });
            }
            // `\` escape or non-ASCII: slow path below.
        }
        let mut name = String::new();
        if self.bytes[self.pos] == b'\\' {
            self.ident_escape(&mut name, true)?;
        } else {
            let c = self.char_at(self.pos);
            name.push(c);
            self.pos += c.len_utf8();
        }
        self.ident_continue(&mut name)?;
        let mut kind = keyword_or_ident(&name);
        // Contiguous `new.target` lexes as one metaproperty token.
        if kind == TokenKind::New
            && self.src[self.pos..].starts_with(".target")
            && !self.ident_part_at(self.pos + 7)
        {
            self.pos += 7;
            kind = TokenKind::MetaPropertyNewTarget;
        }
        Ok(Token {
            kind,
            span: Span {
                start,
                end: self.pos as u32,
            },
        })
    }

    fn lex_private_ident(&mut self, start: u32) -> Result<Token, LexError> {
        self.pos += 1; // consume `#`
        if self.pos >= self.bytes.len() {
            return Err(LexError::UnexpectedChar('#', start));
        }
        // Fast path: pure-ASCII private identifier with no escapes. The
        // first character must be `$`, `_`, or ASCII letter (a digit is an
        // error, caught by the slow path).
        if self.bytes[self.pos] != b'\\'
            && matches!(
                self.bytes[self.pos],
                b'a'..=b'z' | b'A'..=b'Z' | b'$' | b'_'
            )
        {
            let bytes = self.bytes;
            let mut end = self.pos;
            while end < bytes.len()
                && matches!(
                    bytes[end],
                    b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'$' | b'_'
                )
            {
                end += 1;
            }
            if end == bytes.len() || (bytes[end] != b'\\' && bytes[end] < 0x80) {
                let name = self.src[self.pos..end].to_string();
                self.pos = end;
                return Ok(Token {
                    kind: TokenKind::PrivateIdent(name),
                    span: Span {
                        start,
                        end: self.pos as u32,
                    },
                });
            }
        }
        let mut name = String::new();
        if self.bytes[self.pos] == b'\\' {
            self.ident_escape(&mut name, true)?;
        } else {
            let c = self.char_at(self.pos);
            if !(is_id_start(c) || c == '$' || c == '_') {
                return Err(LexError::UnexpectedChar(c, self.pos as u32));
            }
            name.push(c);
            self.pos += c.len_utf8();
        }
        self.ident_continue(&mut name)?;
        Ok(Token {
            kind: TokenKind::PrivateIdent(name),
            span: Span {
                start,
                end: self.pos as u32,
            },
        })
    }

    /// Scan IdentifierPart characters (including `\u` escapes) into `name`.
    fn ident_continue(&mut self, name: &mut String) -> Result<(), LexError> {
        loop {
            if self.pos >= self.bytes.len() {
                break;
            }
            let b = self.bytes[self.pos];
            if b == b'\\' {
                self.ident_escape(name, false)?;
            } else if b < 0x80 {
                if matches!(b, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'$' | b'_') {
                    name.push(b as char);
                    self.pos += 1;
                } else {
                    break;
                }
            } else {
                let c = self.char_at(self.pos);
                if is_id_continue(c) || c == '\u{200C}' || c == '\u{200D}' {
                    name.push(c);
                    self.pos += c.len_utf8();
                } else {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Decode a `\uXXXX` / `\u{...}` escape inside an identifier; the decoded
    /// character must be a valid start/part character.
    fn ident_escape(&mut self, out: &mut String, first: bool) -> Result<(), LexError> {
        let esc = self.pos as u32;
        if self.pos + 1 >= self.bytes.len() || self.bytes[self.pos + 1] != b'u' {
            return Err(LexError::UnexpectedChar('\u{FFFD}', esc));
        }
        match self.unicode_escape(self.pos + 1) {
            Ok((v, np)) => {
                // Lone surrogates are never identifier characters.
                let c = match char::from_u32(v) {
                    Some(c) => c,
                    None => return Err(LexError::UnexpectedChar('\\', esc)),
                };
                let ok = if first {
                    is_id_start(c) || c == '$' || c == '_'
                } else {
                    is_id_continue(c) || c == '$' || c == '_' || c == '\u{200C}' || c == '\u{200D}'
                };
                if !ok {
                    return Err(LexError::UnexpectedChar('\\', esc));
                }
                self.pos = np;
                out.push(c);
                Ok(())
            }
            Err((c, off)) => Err(LexError::UnexpectedChar(c, off)),
        }
    }

    /// True if the character at `pos` could continue an identifier
    /// (used for the `new.target` boundary check).
    fn ident_part_at(&self, pos: usize) -> bool {
        if pos >= self.bytes.len() {
            return false;
        }
        if self.bytes[pos] == b'\\' {
            return true; // `new.target\u0041...` must not combine
        }
        let c = self.char_at(pos);
        is_id_continue(c) || c == '$' || c == '\u{200C}' || c == '\u{200D}'
    }
}

/// Map a decoded identifier to its keyword kind, or [`TokenKind::Ident`].
fn keyword_or_ident(name: &str) -> TokenKind {
    match name {
        "await" => TokenKind::Await,
        "break" => TokenKind::Break,
        "case" => TokenKind::Case,
        "catch" => TokenKind::Catch,
        "class" => TokenKind::Class,
        "const" => TokenKind::Const,
        "continue" => TokenKind::Continue,
        "debugger" => TokenKind::Debugger,
        "default" => TokenKind::Default,
        "delete" => TokenKind::Delete,
        "do" => TokenKind::Do,
        "else" => TokenKind::Else,
        "enum" => TokenKind::Enum,
        "export" => TokenKind::Export,
        "extends" => TokenKind::Extends,
        "false" => TokenKind::False,
        "finally" => TokenKind::Finally,
        "for" => TokenKind::For,
        "function" => TokenKind::Function,
        "if" => TokenKind::If,
        "import" => TokenKind::Import,
        "in" => TokenKind::In,
        "instanceof" => TokenKind::Instanceof,
        "new" => TokenKind::New,
        "null" => TokenKind::Null,
        "return" => TokenKind::Return,
        "super" => TokenKind::Super,
        "switch" => TokenKind::Switch,
        "this" => TokenKind::This,
        "throw" => TokenKind::Throw,
        "true" => TokenKind::True,
        "try" => TokenKind::Try,
        "typeof" => TokenKind::Typeof,
        "var" => TokenKind::Var,
        "void" => TokenKind::Void,
        "while" => TokenKind::While,
        "with" => TokenKind::With,
        "yield" => TokenKind::Yield,
        "let" => TokenKind::Let,
        "static" => TokenKind::Static,
        "of" => TokenKind::Of,
        "get" => TokenKind::Get,
        "set" => TokenKind::Set,
        "async" => TokenKind::Async,
        "from" => TokenKind::From,
        "as" => TokenKind::As,
        "target" => TokenKind::Target,
        _ => TokenKind::Ident(name.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        lex(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    fn no_eof(src: &str) -> Vec<TokenKind> {
        let mut k = kinds(src);
        assert_eq!(k.pop(), Some(TokenKind::Eof));
        k
    }

    #[test]
    fn punctuator_torture() {
        assert_eq!(
            no_eof("a >>>= b"),
            vec![
                TokenKind::Ident("a".into()),
                TokenKind::GtGtGtEq,
                TokenKind::Ident("b".into())
            ]
        );
        assert_eq!(
            no_eof("x **= 2"),
            vec![
                TokenKind::Ident("x".into()),
                TokenKind::StarStarEq,
                TokenKind::Numeric(2.0)
            ]
        );
        assert_eq!(
            no_eof("x ** 2"),
            vec![
                TokenKind::Ident("x".into()),
                TokenKind::StarStar,
                TokenKind::Numeric(2.0)
            ]
        );
        assert_eq!(
            no_eof("a?.b ?? c ??= d &&= e ||= f"),
            vec![
                TokenKind::Ident("a".into()),
                TokenKind::OptChain,
                TokenKind::Ident("b".into()),
                TokenKind::QuestionQuestion,
                TokenKind::Ident("c".into()),
                TokenKind::QuestionQuestionEq,
                TokenKind::Ident("d".into()),
                TokenKind::AmpAmpEq,
                TokenKind::Ident("e".into()),
                TokenKind::PipePipeEq,
                TokenKind::Ident("f".into()),
            ]
        );
        assert_eq!(
            no_eof("f(...args)"),
            vec![
                TokenKind::Ident("f".into()),
                TokenKind::LParen,
                TokenKind::Ellipsis,
                TokenKind::Ident("args".into()),
                TokenKind::RParen,
            ]
        );
        assert_eq!(
            no_eof("() => x"),
            vec![
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::Arrow,
                TokenKind::Ident("x".into())
            ]
        );
        // `?.` followed by a digit is `?` then a number, not `?.`.
        assert_eq!(
            no_eof("a?.5:0"),
            vec![
                TokenKind::Ident("a".into()),
                TokenKind::Question,
                TokenKind::Numeric(0.5),
                TokenKind::Colon,
                TokenKind::Numeric(0.0),
            ]
        );
        assert_eq!(
            no_eof("a !== b; c <<= 1; d >>= 2; e &= 3;"),
            vec![
                TokenKind::Ident("a".into()),
                TokenKind::NotEqEq,
                TokenKind::Ident("b".into()),
                TokenKind::Semicolon,
                TokenKind::Ident("c".into()),
                TokenKind::LtLtEq,
                TokenKind::Numeric(1.0),
                TokenKind::Semicolon,
                TokenKind::Ident("d".into()),
                TokenKind::GtGtEq,
                TokenKind::Numeric(2.0),
                TokenKind::Semicolon,
                TokenKind::Ident("e".into()),
                TokenKind::AmpEq,
                TokenKind::Numeric(3.0),
                TokenKind::Semicolon,
            ]
        );
    }

    #[test]
    fn nested_templates() {
        assert_eq!(
            no_eof("`a${`b${c}d`}e`"),
            vec![
                TokenKind::TemplateHead("a".into()),
                TokenKind::TemplateHead("b".into()),
                TokenKind::Ident("c".into()),
                TokenKind::TemplateTail("d".into()),
                TokenKind::TemplateTail("e".into()),
            ]
        );
        assert_eq!(
            no_eof("`x${a}b${c}d`"),
            vec![
                TokenKind::TemplateHead("x".into()),
                TokenKind::Ident("a".into()),
                TokenKind::TemplateMiddle("b".into()),
                TokenKind::Ident("c".into()),
                TokenKind::TemplateTail("d".into()),
            ]
        );
        // Braces inside a substitution do not confuse the template tracker.
        assert_eq!(
            no_eof("`${ {a: 1} }`"),
            vec![
                TokenKind::TemplateHead("".into()),
                TokenKind::LBrace,
                TokenKind::Ident("a".into()),
                TokenKind::Colon,
                TokenKind::Numeric(1.0),
                TokenKind::RBrace,
                TokenKind::TemplateTail("".into()),
            ]
        );
        // Plain template with no substitutions is a single Tail.
        assert_eq!(no_eof("`hi`"), vec![TokenKind::TemplateTail("hi".into())]);
    }

    #[test]
    fn template_escapes_and_lines() {
        assert_eq!(
            no_eof("`a\\n\\t\\u{1F600}b`"),
            vec![TokenKind::TemplateTail("a\n\t\u{1F600}b".into())]
        );
        // CRLF inside a template normalizes to \n and is not ASI trivia.
        let (res, trivia) = lex_with_trivia("`a\r\nb`");
        assert!(res.is_ok());
        assert!(trivia.is_empty());
        assert_eq!(
            no_eof("`a\r\nb`"),
            vec![TokenKind::TemplateTail("a\nb".into())]
        );
        // Invalid escapes survive verbatim in templates (tagged templates).
        assert_eq!(
            no_eof("`\\unicode`"),
            vec![TokenKind::TemplateTail("\\unicode".into())]
        );
        assert_eq!(
            no_eof("`\\x4`"),
            vec![TokenKind::TemplateTail("\\x4".into())]
        );
    }

    #[test]
    fn strings() {
        assert_eq!(
            no_eof(r#"'a\nb\tc\\d\'e\"f'"#),
            vec![TokenKind::Str("a\nb\tc\\d'e\"f".into())]
        );
        assert_eq!(no_eof(r#""A""#), vec![TokenKind::Str("A".into())]);
        assert_eq!(
            no_eof(r"'\x41\u0042\u{1F600}'"),
            vec![TokenKind::Str("AB\u{1F600}".into())]
        );
        assert_eq!(no_eof(r"'\0'"), vec![TokenKind::Str("\0".into())]);
        assert_eq!(no_eof(r"'\08'"), vec![TokenKind::Str("\08".into())]);
        assert_eq!(no_eof(r"'\8'"), vec![TokenKind::Str("8".into())]);
        assert_eq!(no_eof(r"'\9'"), vec![TokenKind::Str("9".into())]);
        assert_eq!(no_eof(r"'\101'"), vec![TokenKind::Str("A".into())]);
        // Line continuation.
        assert_eq!(no_eof("'a\\\nb'"), vec![TokenKind::Str("ab".into())]);
        assert_eq!(no_eof("'a\\\r\nb'"), vec![TokenKind::Str("ab".into())]);
        // Errors.
        assert!(matches!(lex("'abc"), Err(LexError::UnterminatedString(0))));
        assert!(matches!(
            lex("'a\nb'"),
            Err(LexError::UnterminatedString(0))
        ));
        assert!(matches!(lex(r"'\x'"), Err(LexError::UnexpectedChar(_, _))));
        assert!(matches!(lex(r"'\x4'"), Err(LexError::UnexpectedChar(_, _))));
        assert!(matches!(lex(r"'\uz'"), Err(LexError::UnexpectedChar(_, _))));
        assert!(matches!(
            lex(r"'\u{110000}'"),
            Err(LexError::UnexpectedChar(_, _))
        ));
    }

    #[test]
    fn numbers() {
        let cases: &[(&str, f64)] = &[
            ("0", 0.0),
            ("123", 123.0),
            ("1_000", 1000.0),
            ("1_2_3", 123.0),
            ("0xFF", 255.0),
            ("0Xff", 255.0),
            ("0xF_F", 255.0),
            ("0b101", 5.0),
            ("0B1_0", 2.0),
            ("0o17", 15.0),
            ("0O7_7", 63.0),
            (".5", 0.5),
            ("5.", 5.0),
            ("5.e3", 5000.0),
            ("1e10", 1e10),
            ("1E-5", 1e-5),
            ("1e+3", 1000.0),
            ("1e1_0", 1e10),
            ("2.5", 2.5),
        ];
        for (src, want) in cases {
            assert_eq!(no_eof(src), vec![TokenKind::Numeric(*want)], "src={src}");
        }
        // Legacy octal and octal-like decimal lex as `NumericOctal` so the
        // parser can reject them in strict mode.
        for (src, want) in [
            ("0755", 493.0),
            ("00", 0.0),
            ("08", 8.0),
            ("09", 9.0),
            ("078", 78.0),
        ] {
            assert_eq!(
                no_eof(src),
                vec![TokenKind::NumericOctal(want)],
                "src={src}"
            );
        }
        // BigInts.
        assert_eq!(no_eof("10n"), vec![TokenKind::BigInt("10".into())]);
        assert_eq!(no_eof("0n"), vec![TokenKind::BigInt("0".into())]);
        assert_eq!(no_eof("0x1n"), vec![TokenKind::BigInt("0x1".into())]);
        assert_eq!(no_eof("0xFFn"), vec![TokenKind::BigInt("0xFF".into())]);
        assert_eq!(no_eof("0b101n"), vec![TokenKind::BigInt("0b101".into())]);
        assert_eq!(no_eof("0o17n"), vec![TokenKind::BigInt("0o17".into())]);
        assert_eq!(no_eof("1_000n"), vec![TokenKind::BigInt("1_000".into())]);
        // `1e` is `1` then identifier `e` (maximal munch backtracks).
        assert_eq!(
            no_eof("1e"),
            vec![TokenKind::Numeric(1.0), TokenKind::Ident("e".into())]
        );
        // `0_1` lexes as `0` and `_1` (engines reject it; the parser sees two tokens).
        assert_eq!(
            no_eof("0_1"),
            vec![TokenKind::Numeric(0.0), TokenKind::Ident("_1".into())]
        );
        // Malformed numbers.
        for src in ["0x", "0b", "0o", "0x_", "0b__1", "1__2", "1_", "1_ ", "0b2"] {
            assert!(
                matches!(lex(src), Err(LexError::InvalidNumber(0))),
                "src={src}"
            );
        }
    }

    #[test]
    fn identifiers_unicode() {
        assert_eq!(no_eof("é"), vec![TokenKind::Ident("é".into())]);
        assert_eq!(no_eof("中"), vec![TokenKind::Ident("中".into())]);
        assert_eq!(no_eof("℘x"), vec![TokenKind::Ident("℘x".into())]); // Other_ID_Start
        assert_eq!(no_eof("a·b"), vec![TokenKind::Ident("a·b".into())]); // Other_ID_Continue
        assert_eq!(no_eof("Ⅷ"), vec![TokenKind::Ident("Ⅷ".into())]); // Nl
        assert_eq!(
            no_eof("a\u{200D}b"),
            vec![TokenKind::Ident("a\u{200D}b".into())]
        ); // ZWJ
        assert_eq!(
            no_eof("$foo _bar2"),
            vec![
                TokenKind::Ident("$foo".into()),
                TokenKind::Ident("_bar2".into())
            ]
        );
        // U+30FB is ID_Continue only (15.1 fix); valid as a non-first char.
        assert_eq!(no_eof("a・"), vec![TokenKind::Ident("a・".into())]);
        assert_eq!(
            no_eof("\u{2EBF0}"),
            vec![TokenKind::Ident("\u{2EBF0}".into())]
        ); // CJK Ext I
    }

    #[test]
    fn identifier_escapes() {
        assert_eq!(no_eof("\\u0061"), vec![TokenKind::Ident("a".into())]);
        assert_eq!(no_eof("\\u{62}c"), vec![TokenKind::Ident("bc".into())]);
        // Escaped keywords are still keywords.
        assert_eq!(no_eof("cl\\u0061ss"), vec![TokenKind::Class]);
        assert_eq!(no_eof("\\u0069\\u0066"), vec![TokenKind::If]);
        // Private identifiers, including escapes.
        assert_eq!(no_eof("#foo"), vec![TokenKind::PrivateIdent("foo".into())]);
        assert_eq!(
            no_eof("#\\u0061bc"),
            vec![TokenKind::PrivateIdent("abc".into())]
        );
        assert!(matches!(lex("#1"), Err(LexError::UnexpectedChar('1', 1))));
        assert!(matches!(lex("#"), Err(LexError::UnexpectedChar('#', 0))));
        assert!(matches!(
            lex("\\uD800x"),
            Err(LexError::UnexpectedChar(_, _))
        ));
    }

    #[test]
    fn keywords() {
        let cases: &[(&str, TokenKind)] = &[
            ("await", TokenKind::Await),
            ("break", TokenKind::Break),
            ("case", TokenKind::Case),
            ("catch", TokenKind::Catch),
            ("class", TokenKind::Class),
            ("const", TokenKind::Const),
            ("continue", TokenKind::Continue),
            ("debugger", TokenKind::Debugger),
            ("default", TokenKind::Default),
            ("delete", TokenKind::Delete),
            ("do", TokenKind::Do),
            ("else", TokenKind::Else),
            ("enum", TokenKind::Enum),
            ("export", TokenKind::Export),
            ("extends", TokenKind::Extends),
            ("false", TokenKind::False),
            ("finally", TokenKind::Finally),
            ("for", TokenKind::For),
            ("function", TokenKind::Function),
            ("if", TokenKind::If),
            ("import", TokenKind::Import),
            ("in", TokenKind::In),
            ("instanceof", TokenKind::Instanceof),
            ("new", TokenKind::New),
            ("null", TokenKind::Null),
            ("return", TokenKind::Return),
            ("super", TokenKind::Super),
            ("switch", TokenKind::Switch),
            ("this", TokenKind::This),
            ("throw", TokenKind::Throw),
            ("true", TokenKind::True),
            ("try", TokenKind::Try),
            ("typeof", TokenKind::Typeof),
            ("var", TokenKind::Var),
            ("void", TokenKind::Void),
            ("while", TokenKind::While),
            ("with", TokenKind::With),
            ("yield", TokenKind::Yield),
            ("let", TokenKind::Let),
            ("static", TokenKind::Static),
            ("of", TokenKind::Of),
            ("get", TokenKind::Get),
            ("set", TokenKind::Set),
            ("async", TokenKind::Async),
            ("from", TokenKind::From),
            ("as", TokenKind::As),
            ("target", TokenKind::Target),
        ];
        for (src, want) in cases {
            assert_eq!(no_eof(src), vec![want.clone()], "src={src}");
        }
        // Non-keywords stay identifiers.
        assert_eq!(no_eof("package"), vec![TokenKind::Ident("package".into())]);
        assert_eq!(no_eof("foo"), vec![TokenKind::Ident("foo".into())]);
    }

    #[test]
    fn new_target() {
        assert_eq!(no_eof("new.target"), vec![TokenKind::MetaPropertyNewTarget]);
        assert_eq!(
            no_eof("new . target"),
            vec![TokenKind::New, TokenKind::Dot, TokenKind::Target]
        );
        assert_eq!(
            no_eof("new.targetx"),
            vec![
                TokenKind::New,
                TokenKind::Dot,
                TokenKind::Ident("targetx".into())
            ]
        );
        assert_eq!(
            no_eof("new.target()"),
            vec![
                TokenKind::MetaPropertyNewTarget,
                TokenKind::LParen,
                TokenKind::RParen
            ]
        );
    }

    #[test]
    fn comments_and_hashbang() {
        assert_eq!(
            no_eof("#!/usr/bin/env node\nlet x = 1;"),
            vec![
                TokenKind::Let,
                TokenKind::Ident("x".into()),
                TokenKind::Eq,
                TokenKind::Numeric(1.0),
                TokenKind::Semicolon,
            ]
        );
        assert_eq!(no_eof("// comment\nx"), vec![TokenKind::Ident("x".into())]);
        assert_eq!(
            no_eof("/* multi\nline */ x"),
            vec![TokenKind::Ident("x".into())]
        );
        // Annex B HTML-like comments.
        assert_eq!(no_eof("1<!--2"), vec![TokenKind::Numeric(1.0)]);
        assert_eq!(
            no_eof("x\n--> hidden\ny"),
            vec![TokenKind::Ident("x".into()), TokenKind::Ident("y".into())]
        );
        // `-->` mid-line is `--` then `>`.
        assert_eq!(
            no_eof("q-->0"),
            vec![
                TokenKind::Ident("q".into()),
                TokenKind::MinusMinus,
                TokenKind::Gt,
                TokenKind::Numeric(0.0),
            ]
        );
        // `<!--` needs all four characters contiguous.
        assert_eq!(
            no_eof("a < !--b"),
            vec![
                TokenKind::Ident("a".into()),
                TokenKind::Lt,
                TokenKind::Bang,
                TokenKind::MinusMinus,
                TokenKind::Ident("b".into()),
            ]
        );
        assert!(matches!(
            lex("/* abc"),
            Err(LexError::UnterminatedComment(0))
        ));
    }

    #[test]
    fn asi_trivia() {
        let (_, t) = lex_with_trivia("a\nb");
        assert_eq!(t, vec![Span { start: 1, end: 2 }]);
        let (_, t) = lex_with_trivia("a\r\nb");
        assert_eq!(t, vec![Span { start: 1, end: 3 }]);
        let (_, t) = lex_with_trivia("a\rb");
        assert_eq!(t, vec![Span { start: 1, end: 2 }]);
        let (_, t) = lex_with_trivia("a\u{2028}b"); // U+2028, explicit escape
        assert_eq!(t, vec![Span { start: 1, end: 4 }]);
        // Line terminators inside block comments count.
        let (_, t) = lex_with_trivia("a /*\n*/ b");
        assert_eq!(t, vec![Span { start: 4, end: 5 }]);
        let (_, t) = lex_with_trivia("a /* x */ b");
        assert!(t.is_empty());
        // No line terminator: no trivia.
        let (_, t) = lex_with_trivia("a + b");
        assert!(t.is_empty());
    }

    #[test]
    fn u2028_u2029_no_char_boundary_panic() {
        // U+2028 / U+2029 are 3-byte chars; the lexer must never land on a
        // non-char-boundary when consuming them as line terminators.
        let (_, t) = lex_with_trivia("a /*\u{2028}*/ b");
        assert_eq!(t, vec![Span { start: 4, end: 7 }]);
        let (_, t) = lex_with_trivia("a /*\u{2029}\u{2029}*/ b");
        assert_eq!(
            t,
            vec![Span { start: 4, end: 7 }, Span { start: 7, end: 10 }]
        );
        assert_eq!(
            no_eof("x\u{2029}y"),
            vec![TokenKind::Ident("x".into()), TokenKind::Ident("y".into())]
        );
        // U+2028 terminates a single-line comment and `<!--` comment.
        assert_eq!(no_eof("// c\u{2028}d"), vec![TokenKind::Ident("d".into())]);
        // U+2028 ends the `<!--` comment just like `\n` does.
        assert_eq!(
            no_eof("1<!--\u{2028}2"),
            vec![TokenKind::Numeric(1.0), TokenKind::Numeric(2.0)]
        );
        // U+2028 inside a template is just content, no panic.
        assert_eq!(no_eof("`a\u{2028}b`").len(), 1);
        // Mixed terminators in one block comment.
        let (_, t) = lex_with_trivia("/*\r\n\u{2028}\n\u{2029}*/x");
        assert_eq!(t.len(), 4);
        assert_eq!(
            no_eof("/*\r\n\u{2028}\n\u{2029}*/x"),
            vec![TokenKind::Ident("x".into())]
        );
    }

    #[test]
    fn regex_lexing() {
        let (kind, end) = lex_regex("/ab+c/gi", 0).unwrap();
        assert_eq!(kind, TokenKind::RegexLiteral("/ab+c/gi".into()));
        assert_eq!(end, 8);
        let (kind, end) = lex_regex("/[/]/", 0).unwrap();
        assert_eq!(kind, TokenKind::RegexLiteral("/[/]/".into()));
        assert_eq!(end, 5);
        let (kind, end) = lex_regex(r"/\//", 0).unwrap();
        assert_eq!(kind, TokenKind::RegexLiteral(r"/\//".into()));
        assert_eq!(end, 4);
        let (kind, end) = lex_regex("x = /a/; /b/", 4).unwrap();
        assert_eq!(kind, TokenKind::RegexLiteral("/a/".into()));
        assert_eq!(end, 7);
        assert!(matches!(
            lex_regex("/ab", 0),
            Err(LexError::UnterminatedRegex(0))
        ));
        assert!(matches!(
            lex_regex("/a\nb/", 0),
            Err(LexError::UnterminatedRegex(0))
        ));
        assert!(matches!(
            lex_regex("ab", 0),
            Err(LexError::UnexpectedChar('a', 0))
        ));
    }

    #[test]
    fn lexer_always_emits_slash() {
        // The batch lexer never guesses regex vs division.
        assert_eq!(
            no_eof("x = a / b / c;"),
            vec![
                TokenKind::Ident("x".into()),
                TokenKind::Eq,
                TokenKind::Ident("a".into()),
                TokenKind::Slash,
                TokenKind::Ident("b".into()),
                TokenKind::Slash,
                TokenKind::Ident("c".into()),
                TokenKind::Semicolon,
            ]
        );
    }

    #[test]
    fn incremental_regex_protocol() {
        // The documented parser protocol: drive Lexer incrementally, splice
        // regexes via lex_regex + set_pos.
        let src = "x = /ab+c/gi;";
        let mut lx = Lexer::new(src);
        assert_eq!(lx.next_token().unwrap().kind, TokenKind::Ident("x".into()));
        assert_eq!(lx.next_token().unwrap().kind, TokenKind::Eq);
        let slash = lx.next_token().unwrap();
        assert_eq!(slash.kind, TokenKind::Slash);
        let (kind, end) = lex_regex(src, slash.span.start).unwrap();
        assert_eq!(kind, TokenKind::RegexLiteral("/ab+c/gi".into()));
        lx.set_pos(end);
        assert_eq!(lx.next_token().unwrap().kind, TokenKind::Semicolon);
        assert_eq!(lx.next_token().unwrap().kind, TokenKind::Eof);
    }

    #[test]
    fn incremental_trivia_for_asi() {
        let mut lx = Lexer::new("a\nb");
        let a = lx.next_token().unwrap();
        assert_eq!(a.kind, TokenKind::Ident("a".into()));
        let b = lx.next_token().unwrap();
        assert_eq!(b.kind, TokenKind::Ident("b".into()));
        // The "\n" (span 1..2) sits between the two tokens.
        assert!(lx.has_line_term_between(a.span.end, b.span.start));
        assert!(!lx.has_line_term_between(0, a.span.end));
    }

    #[test]
    fn spans_and_eof() {
        let toks = lex("let x = 42;").unwrap();
        assert_eq!(toks[0].span, Span { start: 0, end: 3 });
        assert_eq!(toks[1].span, Span { start: 4, end: 5 });
        assert_eq!(toks[2].span, Span { start: 6, end: 7 });
        assert_eq!(toks[3].span, Span { start: 8, end: 10 });
        assert_eq!(toks[4].span, Span { start: 10, end: 11 });
        assert_eq!(toks[5].kind, TokenKind::Eof);
        assert_eq!(toks[5].span, Span { start: 11, end: 11 });
        let toks = lex("").unwrap();
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].span, Span { start: 0, end: 0 });
    }

    #[test]
    fn at_punctuator_for_decorators() {
        // `@` lexes as a punctuator (ES2022+ decorator marker), so
        // `@decorator class C {}` tokenizes even though the parser does not
        // implement decorators yet.
        assert_eq!(
            kinds("@decorator class C {}"),
            vec![
                TokenKind::At,
                TokenKind::Ident("decorator".into()),
                TokenKind::Class,
                TokenKind::Ident("C".into()),
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::Eof,
            ]
        );
        assert_eq!(lex("@").unwrap()[0].span, Span { start: 0, end: 1 });
    }

    #[test]
    fn misc_errors() {
        assert!(matches!(
            lex("`abc"),
            Err(LexError::UnterminatedTemplate(0))
        ));
        assert!(matches!(
            lex("`a${b"),
            Err(LexError::UnterminatedTemplate(0))
        ));
        // A lone hashbang is skipped, leaving just Eof.
        assert_eq!(kinds("#!"), vec![TokenKind::Eof]);
        // ...but `#!` anywhere else is an error.
        assert!(matches!(lex("x #!"), Err(LexError::UnexpectedChar('!', 3))));
    }

    #[test]
    fn error_display_includes_offset() {
        let e = LexError::InvalidNumber(42);
        assert!(e.to_string().contains("42"));
        let e = LexError::UnexpectedChar('@', 7);
        let s = e.to_string();
        assert!(s.contains("'@'") && s.contains('7'));
    }

    #[test]
    fn ident_fast_and_slow_paths_agree() {
        // ASCII fast path.
        assert_eq!(
            no_eof("alpha beta123 $x _y"),
            vec![
                TokenKind::Ident("alpha".into()),
                TokenKind::Ident("beta123".into()),
                TokenKind::Ident("$x".into()),
                TokenKind::Ident("_y".into()),
            ]
        );
        // Keywords still resolve through the fast path.
        assert_eq!(
            no_eof("if else while"),
            vec![TokenKind::If, TokenKind::Else, TokenKind::While]
        );
        // Escaped identifier: slow path.
        assert_eq!(no_eof("a\\u0041b"), vec![TokenKind::Ident("aAb".into())]);
        // Non-ASCII identifier: slow path.
        assert_eq!(no_eof("café"), vec![TokenKind::Ident("café".into())]);
        // `new.target` still combines through the fast path.
        assert_eq!(no_eof("new.target"), vec![TokenKind::MetaPropertyNewTarget]);
        // Private identifiers.
        assert_eq!(
            no_eof("#priv #a1"),
            vec![
                TokenKind::PrivateIdent("priv".into()),
                TokenKind::PrivateIdent("a1".into()),
            ]
        );
        assert!(lex("#1").is_err());
    }

    #[test]
    fn string_fast_and_slow_paths_agree() {
        assert_eq!(
            no_eof("'plain' \"also plain\""),
            vec![
                TokenKind::Str("plain".into()),
                TokenKind::Str("also plain".into()),
            ]
        );
        assert_eq!(
            no_eof("'a\\nb' 'tab\\there'"),
            vec![
                TokenKind::Str("a\nb".into()),
                TokenKind::Str("tab\there".into())
            ]
        );
        assert_eq!(no_eof("'caf\\u00e9'"), vec![TokenKind::Str("café".into())]);
    }

    #[test]
    fn number_separators_and_long_literals() {
        assert_eq!(no_eof("1_000"), vec![TokenKind::Numeric(1000.0)]);
        assert_eq!(no_eof("1_000.5e1_0"), vec![TokenKind::Numeric(1000.5e10)]);
        // Long literal exercises the NumBuf heap spillover (> 64 bytes).
        let long_int = "1".repeat(80);
        assert_eq!(
            no_eof(&long_int),
            vec![TokenKind::Numeric(long_int.parse::<f64>().unwrap())]
        );
        let long_sep = format!("{}_{}_{}", "1".repeat(30), "2".repeat(30), "3".repeat(30));
        let expect: f64 = format!("{}{}{}", "1".repeat(30), "2".repeat(30), "3".repeat(30))
            .parse()
            .unwrap();
        assert_eq!(no_eof(&long_sep), vec![TokenKind::Numeric(expect)]);
        assert_eq!(no_eof("0xFF"), vec![TokenKind::Numeric(255.0)]);
        assert_eq!(no_eof("10n"), vec![TokenKind::BigInt("10".into())]);
    }
}
