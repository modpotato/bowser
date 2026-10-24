//! CSS Syntax Module Level 3 tokenizer.
//!
//! Turns the raw stylesheet text into a flat token stream. Comments are
//! discarded (they never become tokens). The output always ends with a
//! single [`Token::Eof`].

/// A single lexical token.
///
/// Public because [`crate::CssValue::Tokens`] (the verbatim store for
/// custom-property values) exposes the raw stream; it is otherwise an
/// internal detail of the parser.
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Ident(String),
    Function(String),
    AtKeyword(String),
    /// `#name`; the bool is the "would start an identifier" flag.
    Hash(String, bool),
    String(String),
    BadString,
    Url(String),
    BadUrl,
    /// A numeric token. Kept as `f32` deliberately: integers past 2^24 lose
    /// exactness (`16777217` tokenizes as `16777216.0`), but widening to
    /// `f64` would ripple through `CssValue` (`Number`, `Length`,
    /// `Percentage`) into the cascade and matching layers, which are `f32`
    /// end to end. CSS values needing more than 24 bits of integer
    /// precision do not occur in practice, so the churn is not worth it.
    ///
    /// `integer` records whether the source lexeme was an `<integer>`
    /// (no fraction or exponent): the `f32` value alone cannot tell `2`
    /// from `2.0`, but the An+B microsyntax must accept the former and
    /// reject the latter, in every entry point (including the token-only
    /// stylesheet parser, which has no source text to rescan).
    Number {
        value: f32,
        integer: bool,
    },
    /// A dimension token; `integer` is the same `<integer>`-lexeme flag
    /// as on [`Token::Number`] (`2.0n` is not a valid An+B coefficient).
    Dimension {
        value: f32,
        integer: bool,
        unit: String,
    },
    Percentage(f32),
    Delim(char),
    Colon,
    Semicolon,
    Comma,
    LBracket,
    RBracket,
    LParen,
    RParen,
    LBrace,
    RBrace,
    Whitespace,
    Cdo,
    Cdc,
    Eof,
}

/// Tokenize `input` into a flat stream, ending with [`Token::Eof`].
pub(crate) fn tokenize(input: &str) -> Vec<Token> {
    let mut t = Tokenizer { input, pos: 0 };
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

/// Byte-offset cursor over the raw stylesheet text.
///
/// `pos` is a byte offset that always sits on a UTF-8 code-point
/// boundary: [`Tokenizer::bump`] advances by the consumed char's
/// `len_utf8()`, and the few manual `pos += n` sites only ever skip
/// ASCII chars (one byte each). `peek`/`peek_at` decode on demand, so
/// tokenizing never pre-collects a `Vec<char>` copy of the input.
struct Tokenizer<'a> {
    input: &'a str,
    /// Byte offset into [`Tokenizer::input`]; always a char boundary.
    pos: usize,
}

fn is_css_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0c')
}

/// Byte version of [`is_css_ws`]: the five CSS whitespace code points
/// are all ASCII, so whitespace runs can be scanned without decoding.
fn is_css_ws_byte(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0c)
}

fn is_name_start(c: char) -> bool {
    c.is_alphabetic() || c == '_' || (c as u32) >= 0x80
}

fn is_name_char(c: char) -> bool {
    is_name_start(c) || c.is_ascii_digit() || c == '-'
}

impl Tokenizer<'_> {
    /// The char at the cursor, if any. ASCII bytes decode without
    /// touching the UTF-8 decoder; anything else decodes one code point
    /// from the current boundary.
    fn peek(&self) -> Option<char> {
        debug_assert!(self.input.is_char_boundary(self.pos));
        let b = *self.input.as_bytes().get(self.pos)?;
        if b < 0x80 {
            Some(b as char)
        } else {
            // `pos` is a boundary, so this decodes exactly one char.
            self.input[self.pos..].chars().next()
        }
    }

    /// The char `off` positions past the cursor, if any.
    fn peek_at(&self, off: usize) -> Option<char> {
        debug_assert!(self.input.is_char_boundary(self.pos));
        let mut it = self.input[self.pos..].chars();
        for _ in 0..off {
            it.next()?;
        }
        it.next()
    }

    /// Consume the char at the cursor.
    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    /// Skip a run of CSS whitespace starting at the cursor. All five
    /// whitespace code points are ASCII, so this scans bytes directly
    /// instead of decoding char by char.
    fn eat_ws(&mut self) {
        let bytes = self.input.as_bytes();
        let mut i = self.pos;
        while i < bytes.len() && is_css_ws_byte(bytes[i]) {
            i += 1;
        }
        self.pos = i;
    }

    /// `\` at `pos + off` followed by something escapable (not newline/EOF).
    fn valid_escape_at(&self, off: usize) -> bool {
        if self.peek_at(off) != Some('\\') {
            return false;
        }
        !matches!(self.peek_at(off + 1), None | Some('\n' | '\r' | '\x0c'))
    }

    fn starts_ident_at(&self, off: usize) -> bool {
        match self.peek_at(off) {
            Some('-') => match self.peek_at(off + 1) {
                Some('-') => true,
                Some(c) if is_name_start(c) => true,
                _ => self.valid_escape_at(off + 1),
            },
            Some('\\') => self.valid_escape_at(off),
            Some(c) => is_name_start(c),
            None => false,
        }
    }

    fn starts_ident(&self) -> bool {
        self.starts_ident_at(0)
    }

    fn starts_number(&self) -> bool {
        match self.peek() {
            Some('+' | '-') => match self.peek_at(1) {
                Some(c) if c.is_ascii_digit() => true,
                Some('.') => matches!(self.peek_at(2), Some(c) if c.is_ascii_digit()),
                _ => false,
            },
            Some('.') => matches!(self.peek_at(1), Some(c) if c.is_ascii_digit()),
            Some(c) => c.is_ascii_digit(),
            None => false,
        }
    }

    fn skip_comment(&mut self) {
        // Byte scan: `/*` and `*/` are ASCII, so no decoding is needed.
        // A multi-byte char can never contain a 0x2A (`*`) or 0x2F (`/`)
        // byte, and every stop index lands on an ASCII char boundary.
        let bytes = self.input.as_bytes();
        while self.pos + 1 < bytes.len() && bytes[self.pos] == b'/' && bytes[self.pos + 1] == b'*' {
            let mut i = self.pos + 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            // No terminator: the comment eats the rest of the input.
            self.pos = if i + 1 < bytes.len() {
                i + 2
            } else {
                bytes.len()
            };
        }
    }

    /// Consume an escape; the current char must be `\` and [`Self::valid_escape_at`] must hold.
    ///
    /// The hex value accumulates arithmetically instead of via a scratch
    /// [`String`]: escapes allocate nothing.
    fn consume_escaped(&mut self) -> char {
        self.bump(); // the backslash
        let c = self.peek().unwrap_or('\u{FFFD}');
        if !c.is_ascii_hexdigit() {
            self.bump();
            return c;
        }
        let mut v: u32 = 0;
        for _ in 0..6 {
            match self.peek() {
                Some(h) if h.is_ascii_hexdigit() => {
                    // `to_digit` cannot fail: just checked `is_ascii_hexdigit`.
                    v = v * 16 + h.to_digit(16).unwrap_or(0);
                    self.bump();
                }
                _ => break,
            }
        }
        // One trailing whitespace is part of the escape.
        if self.peek() == Some('\r') && self.peek_at(1) == Some('\n') {
            self.pos += 2;
        } else if matches!(self.peek(), Some(c) if is_css_ws(c)) {
            self.pos += 1;
        }
        char::from_u32(v).unwrap_or('\u{FFFD}')
    }

    /// Consume a name. The common case — an ASCII run with no escapes —
    /// is sliced straight out of the input with a single exact-size
    /// allocation. Escapes and non-ASCII name chars take the slow path,
    /// which seeds its buffer with the already-scanned ASCII prefix.
    fn consume_name(&mut self) -> String {
        let start = self.pos;
        let bytes = self.input.as_bytes();
        let mut i = self.pos;
        while i < bytes.len() {
            let b = bytes[i];
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' {
                i += 1;
            } else {
                break;
            }
        }
        self.pos = i;
        // A backslash (possible escape) or a non-ASCII byte (possible
        // non-ASCII name char) needs the char-level slow path. Anything
        // else — a non-name char or EOF — ends the name here.
        let slow = matches!(bytes.get(i), Some(b'\\') | Some(0x80..=0xff));
        if !slow {
            return self.input[start..i].to_owned();
        }
        let mut s = self.input[start..i].to_owned();
        loop {
            match self.peek() {
                Some(c) if is_name_char(c) => {
                    s.push(c);
                    self.bump();
                }
                Some('\\') if self.valid_escape_at(0) => s.push(self.consume_escaped()),
                _ => break,
            }
        }
        s
    }

    /// The numeric value plus whether the lexeme was an `<integer>`
    /// (no `.` fraction and no exponent).
    ///
    /// The lexeme is all ASCII (`+ - . e E` and digits), so it is scanned
    /// as bytes and parsed straight from the input slice: no scratch
    /// [`String`] is built, so numeric tokens allocate nothing at all.
    fn consume_number(&mut self) -> (f32, bool) {
        let start = self.pos;
        let bytes = self.input.as_bytes();
        let mut i = self.pos;
        let mut integer = true;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i + 1 < bytes.len() && bytes[i] == b'.' && bytes[i + 1].is_ascii_digit() {
            integer = false;
            i += 2;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
            let save = i;
            let mut j = i + 1;
            if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
                j += 1;
            }
            let exp_digits = j;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > exp_digits {
                integer = false;
                i = j;
            } else {
                // Not a real exponent (`1e`, `1e+`): rewind so the `e`
                // lexes as a dimension unit instead.
                i = save;
            }
        }
        self.pos = i;
        // The slice holds exactly the chars the old scratch `String`
        // collected, in order, so the parsed value is identical.
        (self.input[start..i].parse::<f32>().unwrap_or(0.0), integer)
    }

    fn consume_numeric(&mut self) -> Token {
        let (value, integer) = self.consume_number();
        if self.starts_ident() {
            Token::Dimension {
                value,
                integer,
                unit: self.consume_name(),
            }
        } else if self.peek() == Some('%') {
            self.bump();
            Token::Percentage(value)
        } else {
            Token::Number { value, integer }
        }
    }

    /// Consume a string. The common case — no backslash, newline, or
    /// early EOF before the closing quote — is sliced straight out of
    /// the input with a single exact-size allocation. The scan only
    /// stops at ASCII bytes, so every stop index is a char boundary;
    /// a multi-byte char can never contain a stop byte. Escapes and
    /// line continuations take the slow path, seeded with the clean
    /// prefix already scanned.
    fn consume_string(&mut self) -> Token {
        let quote = self.bump().unwrap_or('"');
        let qb = quote as u8; // `"` or `'`; ASCII by construction
        let bytes = self.input.as_bytes();
        let mut i = self.pos;
        while i < bytes.len() {
            let b = bytes[i];
            if b == qb || b == b'\\' || b == b'\n' || b == b'\r' || b == 0x0c {
                break;
            }
            i += 1;
        }
        match bytes.get(i).copied() {
            Some(b) if b == qb => {
                let s = self.input[self.pos..i].to_owned();
                self.pos = i + 1;
                Token::String(s)
            }
            None => {
                let s = self.input[self.pos..].to_owned();
                self.pos = i;
                Token::String(s)
            }
            // A newline ends the string as a parse error; the newline
            // itself is not consumed.
            Some(b'\n') | Some(b'\r') | Some(0x0c) => {
                self.pos = i;
                Token::BadString
            }
            // The only remaining stop byte is the backslash.
            _ => {
                debug_assert_eq!(bytes[i], b'\\');
                let mut s = self.input[self.pos..i].to_owned();
                self.pos = i;
                loop {
                    match self.peek() {
                        None => return Token::String(s),
                        Some(c) if c == quote => {
                            self.bump();
                            return Token::String(s);
                        }
                        Some('\n' | '\r' | '\x0c') => return Token::BadString,
                        Some('\\') => match self.peek_at(1) {
                            None => {
                                self.bump();
                            }
                            Some('\n' | '\x0c') => {
                                self.pos += 2;
                            }
                            Some('\r') => {
                                self.pos += 2;
                                if self.peek() == Some('\n') {
                                    self.pos += 1;
                                }
                            }
                            _ => s.push(self.consume_escaped()),
                        },
                        Some(c) => {
                            s.push(c);
                            self.bump();
                        }
                    }
                }
            }
        }
    }

    fn consume_bad_url_remainder(&mut self) -> Token {
        loop {
            match self.peek() {
                None => return Token::BadUrl,
                Some(')') => {
                    self.bump();
                    return Token::BadUrl;
                }
                Some('\\') if self.valid_escape_at(0) => {
                    self.consume_escaped();
                }
                _ => {
                    self.bump();
                }
            }
        }
    }

    /// Called right after consuming `url(`.
    ///
    /// Reaching EOF before the closing `)` is a parse error: per CSS Syntax
    /// L3 the tokenizer flags it, but since this tokenizer carries no
    /// error channel, it yields [`Token::BadUrl`] instead of a [`Token::Url`]
    /// so the error is not silently lost — the declaration is dropped at
    /// value-parse time.
    fn consume_url(&mut self) -> Token {
        self.eat_ws();
        match self.peek() {
            // Quoted: this is really a function token; the string follows.
            Some('"' | '\'') => Token::Function("url".to_string()),
            _ => self.consume_unquoted_url(),
        }
    }

    /// Unquoted `url(...)` body. The common case — a run of non-special
    /// bytes with no escapes — is sliced straight out of the input with
    /// a single exact-size allocation. The scan only stops at ASCII
    /// bytes (or EOF), so every stop index is a char boundary. Any
    /// backslash punts to [`Self::consume_unquoted_url_slow`]: escapes
    /// are rare here and the slow path stays obviously correct.
    fn consume_unquoted_url(&mut self) -> Token {
        let start = self.pos;
        let bytes = self.input.as_bytes();
        let mut i = self.pos;
        let mut saw_backslash = false;
        while i < bytes.len() {
            let b = bytes[i];
            if b == b'\\' {
                saw_backslash = true;
                break;
            }
            // `b <= 0x20` covers space, \t, \n, \r, \x0c and every other
            // control; quotes, parens and 0x7f poison the URL. Multi-byte
            // chars are all >= 0x80 and never stop the scan early.
            if b <= 0x20 || b == 0x7f || b == b'"' || b == b'\'' || b == b'(' || b == b')' {
                break;
            }
            i += 1;
        }
        if saw_backslash {
            self.pos = start;
            return self.consume_unquoted_url_slow();
        }
        self.pos = i;
        match bytes.get(i).copied() {
            None => Token::BadUrl,
            Some(b')') => {
                self.pos = i + 1;
                Token::Url(self.input[start..i].to_owned())
            }
            Some(b) if is_css_ws_byte(b) => {
                let mut j = i + 1;
                while j < bytes.len() && is_css_ws_byte(bytes[j]) {
                    j += 1;
                }
                self.pos = j;
                match bytes.get(j).copied() {
                    None => Token::BadUrl,
                    Some(b')') => {
                        self.pos = j + 1;
                        Token::Url(self.input[start..i].to_owned())
                    }
                    _ => self.consume_bad_url_remainder(),
                }
            }
            // `"`, `'`, `(`, controls, 0x7f: poisoned URL.
            _ => self.consume_bad_url_remainder(),
        }
    }

    /// The pre-wave-65 unquoted-url loop, verbatim: char-by-char with
    /// escape decoding. Only reached when the fast scan saw a backslash.
    fn consume_unquoted_url_slow(&mut self) -> Token {
        let mut s = String::new();
        loop {
            match self.peek() {
                None => return Token::BadUrl,
                Some(')') => {
                    self.bump();
                    return Token::Url(s);
                }
                Some(c) if is_css_ws(c) => {
                    self.eat_ws();
                    match self.peek() {
                        None => return Token::BadUrl,
                        Some(')') => {
                            self.bump();
                            return Token::Url(s);
                        }
                        _ => return self.consume_bad_url_remainder(),
                    }
                }
                Some('"' | '\'' | '(') => return self.consume_bad_url_remainder(),
                Some(c) if (c as u32) < 0x20 || c == '\x7f' => {
                    return self.consume_bad_url_remainder();
                }
                Some('\\') if self.valid_escape_at(0) => s.push(self.consume_escaped()),
                Some('\\') => return self.consume_bad_url_remainder(),
                Some(c) => {
                    s.push(c);
                    self.bump();
                }
            }
        }
    }

    fn consume_ident_like(&mut self) -> Token {
        let name = self.consume_name();
        if self.peek() == Some('(') {
            self.bump();
            if name.eq_ignore_ascii_case("url") {
                self.consume_url()
            } else {
                Token::Function(name)
            }
        } else {
            Token::Ident(name)
        }
    }

    fn next_token(&mut self) -> Token {
        self.skip_comment();
        let c = match self.peek() {
            None => return Token::Eof,
            Some(c) => c,
        };
        match c {
            c if is_css_ws(c) => {
                self.eat_ws();
                Token::Whitespace
            }
            '"' | '\'' => self.consume_string(),
            '#' => {
                if matches!(self.peek_at(1), Some(nc) if is_name_char(nc))
                    || self.valid_escape_at(1)
                {
                    self.bump();
                    let id_flag = self.starts_ident();
                    Token::Hash(self.consume_name(), id_flag)
                } else {
                    self.bump();
                    Token::Delim('#')
                }
            }
            '(' => {
                self.bump();
                Token::LParen
            }
            ')' => {
                self.bump();
                Token::RParen
            }
            '[' => {
                self.bump();
                Token::LBracket
            }
            ']' => {
                self.bump();
                Token::RBracket
            }
            '{' => {
                self.bump();
                Token::LBrace
            }
            '}' => {
                self.bump();
                Token::RBrace
            }
            ',' => {
                self.bump();
                Token::Comma
            }
            ':' => {
                self.bump();
                Token::Colon
            }
            ';' => {
                self.bump();
                Token::Semicolon
            }
            '@' => {
                if self.starts_ident_at(1) {
                    self.bump();
                    Token::AtKeyword(self.consume_name())
                } else {
                    self.bump();
                    Token::Delim('@')
                }
            }
            '<' => {
                if self.peek_at(1) == Some('!')
                    && self.peek_at(2) == Some('-')
                    && self.peek_at(3) == Some('-')
                {
                    self.pos += 4;
                    Token::Cdo
                } else {
                    self.bump();
                    Token::Delim('<')
                }
            }
            '-' => {
                if self.starts_number() {
                    self.consume_numeric()
                } else if self.peek_at(1) == Some('-') && self.peek_at(2) == Some('>') {
                    self.pos += 3;
                    Token::Cdc
                } else if self.starts_ident() {
                    self.consume_ident_like()
                } else {
                    self.bump();
                    Token::Delim('-')
                }
            }
            '+' | '.' => {
                if self.starts_number() {
                    self.consume_numeric()
                } else {
                    self.bump();
                    Token::Delim(c)
                }
            }
            c if c.is_ascii_digit() => self.consume_numeric(),
            _ if self.starts_ident() => self.consume_ident_like(),
            _ => {
                self.bump();
                Token::Delim(c)
            }
        }
    }
}

/// Serialize a token stream back to CSS text.
///
/// Best-effort round-trip for values that were stored verbatim (see
/// [`crate::CssValue::Tokens`]): whitespace runs collapse to one space,
/// and numeric formatting follows `f32`'s `Display`, so this is not
/// byte-identical to the source. Function tokens serialize with their
/// argument tokens and closing paren; `Eof` serializes to nothing.
///
/// A single space is inserted between adjacent tokens wherever the CSS
/// Syntax 3 §5.4.1 serialization rules require it (e.g. `Ident("not")`
/// followed by `Function("media")`, which would otherwise re-tokenize
/// as the single ident `notmedia`), so the output re-parses to the
/// same token stream. Callers that strip whitespace before storing
/// tokens (notably `if()` conditions) depend on this.
pub fn serialize_tokens(toks: &[Token]) -> String {
    fn num(n: f32, integer: bool, out: &mut String) {
        if integer {
            out.push_str(&(n as i64).to_string());
        } else {
            out.push_str(&n.to_string());
        }
    }
    let mut out = String::new();
    let mut prev: Option<&Token> = None;
    for t in toks {
        if let Some(p) = prev
            && needs_separator(p, t)
        {
            out.push(' ');
        }
        match t {
            Token::Ident(s) | Token::AtKeyword(s) => out.push_str(s),
            Token::Function(s) => {
                out.push_str(s);
                out.push('(');
            }
            Token::Hash(s, _) => {
                out.push('#');
                out.push_str(s);
            }
            Token::String(s) => {
                out.push('"');
                out.push_str(&s.replace('"', "\\\""));
                out.push('"');
            }
            Token::BadString => out.push('"'),
            Token::Url(s) => {
                out.push_str("url(\"");
                out.push_str(&s.replace('"', "\\\""));
                out.push_str("\")");
            }
            Token::BadUrl => out.push_str("url("),
            Token::Number { value, integer } => num(*value, *integer, &mut out),
            Token::Dimension {
                value,
                integer,
                unit,
            } => {
                num(*value, *integer, &mut out);
                out.push_str(unit);
            }
            Token::Percentage(v) => {
                out.push_str(&v.to_string());
                out.push('%');
            }
            Token::Delim(c) => out.push(*c),
            Token::Colon => out.push(':'),
            Token::Semicolon => out.push(';'),
            Token::Comma => out.push(','),
            Token::LBracket => out.push('['),
            Token::RBracket => out.push(']'),
            Token::LParen => out.push('('),
            Token::RParen => out.push(')'),
            Token::LBrace => out.push('{'),
            Token::RBrace => out.push('}'),
            Token::Whitespace => out.push(' '),
            Token::Cdo => out.push_str("<!--"),
            Token::Cdc => out.push_str("-->"),
            Token::Eof => {}
        }
        prev = Some(t);
    }
    out
}

/// Whether serializing `prev` immediately followed by `next` needs a
/// separating space, per CSS Syntax 3 §5.4.1: without it the pair would
/// re-tokenize as one (or a different) token, breaking the `to_css`
/// round-trip invariant ("output re-parses to an equal tree").
///
/// Conservative extensions past the spec text: a number/percentage
/// followed by an ident-like or numeric token (`1 px` would re-tokenize
/// as the dimension `1px`), and a `-` delim followed by an ident-like
/// token (`a -b` would re-tokenize as the ident `a-b`), also get a
/// separator. A spurious space never changes the re-parsed token
/// stream (whitespace is insignificant between tokens), so erring on
/// the side of separating is safe.
fn needs_separator(prev: &Token, next: &Token) -> bool {
    // Right-hand tokens that would merge with a preceding ident-like
    // token into a single token when serialized adjacently.
    fn merges(next: &Token) -> bool {
        match next {
            Token::Ident(_) | Token::Function(_) | Token::Url(_) => true,
            Token::Delim('-') => true,
            Token::Number { .. } | Token::Percentage(_) | Token::Dimension { .. } => true,
            // The spec only requires this for a hash with the id flag
            // set; separating unconditionally is harmless.
            Token::Hash(_, _) => true,
            Token::Cdc => true,
            _ => false,
        }
    }
    let ident_like = matches!(
        prev,
        Token::Ident(_)
            | Token::AtKeyword(_)
            | Token::Hash(_, _)
            | Token::Dimension { .. }
            | Token::Number { .. }
            | Token::Percentage(_)
            | Token::Delim('-')
    );
    ident_like && merges(next)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(input: &str) -> Vec<Token> {
        tokenize(input)
    }

    #[test]
    fn empty_input_is_just_eof() {
        assert_eq!(toks(""), vec![Token::Eof]);
    }

    #[test]
    fn whitespace_runs_merge() {
        assert_eq!(toks("  \t\n"), vec![Token::Whitespace, Token::Eof]);
    }

    #[test]
    fn comments_are_skipped() {
        assert_eq!(
            toks("a/* hello */b"),
            vec![
                Token::Ident("a".into()),
                Token::Ident("b".into()),
                Token::Eof
            ]
        );
        assert_eq!(
            toks("a/**/b"),
            vec![
                Token::Ident("a".into()),
                Token::Ident("b".into()),
                Token::Eof
            ]
        );
        // Unterminated comment eats the rest.
        assert_eq!(
            toks("a/* never ends"),
            vec![Token::Ident("a".into()), Token::Eof]
        );
    }

    #[test]
    fn idents_and_functions() {
        assert_eq!(
            toks("-webkit-x foo2 _bar"),
            vec![
                Token::Ident("-webkit-x".into()),
                Token::Whitespace,
                Token::Ident("foo2".into()),
                Token::Whitespace,
                Token::Ident("_bar".into()),
                Token::Eof,
            ]
        );
        assert_eq!(
            toks("rgb("),
            vec![Token::Function("rgb".into()), Token::Eof]
        );
    }

    #[test]
    fn ident_escapes() {
        // `\26 B` is "&B" -> "B"? No: \26 is '&'. So "a\26b" == "a&b".
        assert_eq!(
            toks("a\\26 b"),
            vec![Token::Ident("a&b".into()), Token::Eof]
        );
        // Lone backslash before EOF is a delim.
        assert_eq!(toks("\\"), vec![Token::Delim('\\'), Token::Eof]);
    }

    #[test]
    fn strings() {
        assert_eq!(
            toks(r#""hello""#),
            vec![Token::String("hello".into()), Token::Eof]
        );
        assert_eq!(
            toks("'it\\'s'"),
            vec![Token::String("it's".into()), Token::Eof]
        );
        // Newline inside string -> bad string token; the rest of the line
        // tokenizes normally (the newline itself is not consumed).
        assert_eq!(
            toks("\"abc\ndef\""),
            vec![
                Token::BadString,
                Token::Whitespace,
                Token::Ident("def".into()),
                Token::String("".into()),
                Token::Eof,
            ]
        );
        // Unterminated string at EOF is still a string.
        assert_eq!(toks("\"abc"), vec![Token::String("abc".into()), Token::Eof]);
        // Line continuation: backslash + newline is dropped.
        assert_eq!(
            toks("\"a\\\nb\""),
            vec![Token::String("ab".into()), Token::Eof]
        );
    }

    #[test]
    fn numbers_dimensions_percentages() {
        // The `integer` flag records the `<integer>` vs `<number>` lexeme
        // distinction the An+B microsyntax needs.
        assert_eq!(
            toks("42"),
            vec![
                Token::Number {
                    value: 42.0,
                    integer: true
                },
                Token::Eof
            ]
        );
        assert_eq!(
            toks("42.0"),
            vec![
                Token::Number {
                    value: 42.0,
                    integer: false
                },
                Token::Eof
            ]
        );
        assert_eq!(
            toks("-4.5"),
            vec![
                Token::Number {
                    value: -4.5,
                    integer: false
                },
                Token::Eof
            ]
        );
        assert_eq!(
            toks("+.5"),
            vec![
                Token::Number {
                    value: 0.5,
                    integer: false
                },
                Token::Eof
            ]
        );
        assert_eq!(
            toks("1e3"),
            vec![
                Token::Number {
                    value: 1000.0,
                    integer: false
                },
                Token::Eof
            ]
        );
        assert_eq!(
            toks("1E-2"),
            vec![
                Token::Number {
                    value: 0.01,
                    integer: false
                },
                Token::Eof
            ]
        );
        // `1e` is a dimension with unit "e": the exponent needs digits.
        assert_eq!(
            toks("1e"),
            vec![
                Token::Dimension {
                    value: 1.0,
                    integer: true,
                    unit: "e".into()
                },
                Token::Eof
            ]
        );
        assert_eq!(
            toks("10px"),
            vec![
                Token::Dimension {
                    value: 10.0,
                    integer: true,
                    unit: "px".into()
                },
                Token::Eof
            ]
        );
        assert_eq!(
            toks("2.0n"),
            vec![
                Token::Dimension {
                    value: 2.0,
                    integer: false,
                    unit: "n".into()
                },
                Token::Eof
            ]
        );
        assert_eq!(toks("50%"), vec![Token::Percentage(50.0), Token::Eof]);
        // '.' alone is a delim, not a number.
        assert_eq!(toks("."), vec![Token::Delim('.'), Token::Eof]);
    }

    #[test]
    fn hash_tokens() {
        // `#id` is an ID hash.
        assert_eq!(
            toks("#id"),
            vec![Token::Hash("id".into(), true), Token::Eof]
        );
        // `#123` is unrestricted.
        assert_eq!(
            toks("#123"),
            vec![Token::Hash("123".into(), false), Token::Eof]
        );
        // `# ` is a delim.
        assert_eq!(
            toks("# "),
            vec![Token::Delim('#'), Token::Whitespace, Token::Eof]
        );
    }

    #[test]
    fn at_keywords() {
        assert_eq!(
            toks("@media"),
            vec![Token::AtKeyword("media".into()), Token::Eof]
        );
        assert_eq!(toks("@-x"), vec![Token::AtKeyword("-x".into()), Token::Eof]);
        assert_eq!(
            toks("@ 1"),
            vec![
                Token::Delim('@'),
                Token::Whitespace,
                Token::Number {
                    value: 1.0,
                    integer: true
                },
                Token::Eof
            ]
        );
    }

    #[test]
    fn urls() {
        assert_eq!(
            toks("url(foo.png)"),
            vec![Token::Url("foo.png".into()), Token::Eof]
        );
        assert_eq!(
            toks("url(  foo.png  )"),
            vec![Token::Url("foo.png".into()), Token::Eof]
        );
        // Quoted URL is a function + string.
        assert_eq!(
            toks("url(\"foo.png\")"),
            vec![
                Token::Function("url".into()),
                Token::String("foo.png".into()),
                Token::RParen,
                Token::Eof,
            ]
        );
        // Bad URL with a quote inside.
        assert_eq!(toks("url(a\"b)"), vec![Token::BadUrl, Token::Eof]);
        // Case-insensitive `URL(`.
        assert_eq!(toks("URL(x)"), vec![Token::Url("x".into()), Token::Eof]);
    }

    #[test]
    fn url_at_eof_is_bad_url() {
        // EOF before the closing `)` of an unquoted url() is a parse
        // error: a BadUrl token is produced, never `Url("")`.
        assert_eq!(toks("url("), vec![Token::BadUrl, Token::Eof]);
        assert_eq!(toks("url(foo"), vec![Token::BadUrl, Token::Eof]);
        assert_eq!(toks("url(  "), vec![Token::BadUrl, Token::Eof]);
        // But `url()` with its paren closed is still an (empty) Url.
        assert_eq!(toks("url()"), vec![Token::Url("".into()), Token::Eof]);
    }

    #[test]
    fn cdo_cdc() {
        assert_eq!(toks("<!--"), vec![Token::Cdo, Token::Eof]);
        assert_eq!(toks("-->"), vec![Token::Cdc, Token::Eof]);
        // `--x` is an ident, not CDC.
        assert_eq!(toks("--x"), vec![Token::Ident("--x".into()), Token::Eof]);
    }

    #[test]
    fn punctuation() {
        assert_eq!(
            toks("{}[]():;,"),
            vec![
                Token::LBrace,
                Token::RBrace,
                Token::LBracket,
                Token::RBracket,
                Token::LParen,
                Token::RParen,
                Token::Colon,
                Token::Semicolon,
                Token::Comma,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn serialize_tokens_round_trips() {
        // `toks` appends Eof, which serializes to nothing.
        let s = |input: &str| serialize_tokens(&toks(input));
        assert_eq!(s("16/9"), "16/9");
        assert_eq!(s("italic 12px/1.5 serif"), "italic 12px/1.5 serif");
        assert_eq!(s("var(--x, 1px)"), "var(--x, 1px)");
        assert_eq!(s("rgb(1, 2, 3)"), "rgb(1, 2, 3)");
        assert_eq!(s("#abc"), "#abc");
        assert_eq!(s("50%"), "50%");
        assert_eq!(s("1.5em"), "1.5em");
        assert_eq!(s("2fr"), "2fr");
    }

    #[test]
    fn serialize_tokens_inserts_required_separators() {
        // Whitespace-stripped token pairs that would re-tokenize as a
        // single (or different) token get one separating space, per CSS
        // Syntax 3 §5.4.1. This is what `if()` conditions rely on: the
        // math parser strips whitespace before storing them.
        let stripped = |input: &str| {
            toks(input)
                .into_iter()
                .filter(|t| !matches!(t, Token::Whitespace))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            serialize_tokens(&stripped("not media(width>100px)")),
            "not media(width>100px)"
        );
        assert_eq!(
            serialize_tokens(&stripped("not supports(display:grid)")),
            "not supports(display:grid)"
        );
        // Conservative extensions: `1 px` must not become `1px` (a
        // dimension), `a -b` must not become `a-b` (one ident).
        assert_eq!(serialize_tokens(&stripped("1 px")), "1 px");
        assert_eq!(serialize_tokens(&stripped("a -b")), "a -b");
        // Pairs that cannot merge stay separator-free.
        assert_eq!(serialize_tokens(&stripped("a,b")), "a,b");
        assert_eq!(
            serialize_tokens(&stripped("media(width>100px)")),
            "media(width>100px)"
        );
    }
}

#[cfg(test)]
mod differential {
    /// Differential oracle: the pre-rewrite tokenizer, kept verbatim
    /// (modulo renaming) so the property tests below can assert the new
    /// byte-offset implementation produces a byte-identical token stream.
    /// Test-only; compiled out of normal builds.
    mod old_impl {
        use super::super::Token;

        pub(super) fn old_tokenize(input: &str) -> Vec<Token> {
            let chars: Vec<char> = input.chars().collect();
            let mut t = OldTokenizer {
                chars: &chars,
                pos: 0,
            };
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

        struct OldTokenizer<'a> {
            chars: &'a [char],
            pos: usize,
        }

        fn is_css_ws(c: char) -> bool {
            matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0c')
        }

        fn is_name_start(c: char) -> bool {
            c.is_alphabetic() || c == '_' || (c as u32) >= 0x80
        }

        fn is_name_char(c: char) -> bool {
            is_name_start(c) || c.is_ascii_digit() || c == '-'
        }

        impl OldTokenizer<'_> {
            fn peek(&self) -> Option<char> {
                self.chars.get(self.pos).copied()
            }

            fn peek_at(&self, off: usize) -> Option<char> {
                self.chars.get(self.pos + off).copied()
            }

            fn bump(&mut self) -> Option<char> {
                let c = self.chars.get(self.pos).copied();
                if c.is_some() {
                    self.pos += 1;
                }
                c
            }

            fn eat_while(&mut self, f: fn(char) -> bool) {
                while matches!(self.peek(), Some(c) if f(c)) {
                    self.pos += 1;
                }
            }

            /// `\` at `pos + off` followed by something escapable (not newline/EOF).
            fn valid_escape_at(&self, off: usize) -> bool {
                if self.peek_at(off) != Some('\\') {
                    return false;
                }
                !matches!(self.peek_at(off + 1), None | Some('\n' | '\r' | '\x0c'))
            }

            fn starts_ident_at(&self, off: usize) -> bool {
                match self.peek_at(off) {
                    Some('-') => match self.peek_at(off + 1) {
                        Some('-') => true,
                        Some(c) if is_name_start(c) => true,
                        _ => self.valid_escape_at(off + 1),
                    },
                    Some('\\') => self.valid_escape_at(off),
                    Some(c) => is_name_start(c),
                    None => false,
                }
            }

            fn starts_ident(&self) -> bool {
                self.starts_ident_at(0)
            }

            fn starts_number(&self) -> bool {
                match self.peek() {
                    Some('+' | '-') => match self.peek_at(1) {
                        Some(c) if c.is_ascii_digit() => true,
                        Some('.') => matches!(self.peek_at(2), Some(c) if c.is_ascii_digit()),
                        _ => false,
                    },
                    Some('.') => matches!(self.peek_at(1), Some(c) if c.is_ascii_digit()),
                    Some(c) => c.is_ascii_digit(),
                    None => false,
                }
            }

            fn skip_comment(&mut self) {
                while self.peek() == Some('/') && self.peek_at(1) == Some('*') {
                    self.pos += 2;
                    loop {
                        match self.peek() {
                            None => return,
                            Some('*') if self.peek_at(1) == Some('/') => {
                                self.pos += 2;
                                break;
                            }
                            _ => self.pos += 1,
                        }
                    }
                }
            }

            /// Consume an escape; the current char must be `\` and [`Self::valid_escape_at`] must hold.
            fn consume_escaped(&mut self) -> char {
                self.bump(); // the backslash
                let c = self.peek().unwrap_or('\u{FFFD}');
                if !c.is_ascii_hexdigit() {
                    self.bump();
                    return c;
                }
                let mut hex = String::new();
                for _ in 0..6 {
                    match self.peek() {
                        Some(h) if h.is_ascii_hexdigit() => {
                            hex.push(h);
                            self.bump();
                        }
                        _ => break,
                    }
                }
                // One trailing whitespace is part of the escape.
                if self.peek() == Some('\r') && self.peek_at(1) == Some('\n') {
                    self.pos += 2;
                } else if matches!(self.peek(), Some(c) if is_css_ws(c)) {
                    self.pos += 1;
                }
                let v = u32::from_str_radix(&hex, 16).unwrap_or(0);
                char::from_u32(v).unwrap_or('\u{FFFD}')
            }

            fn consume_name(&mut self) -> String {
                let mut s = String::new();
                loop {
                    match self.peek() {
                        Some(c) if is_name_char(c) => {
                            s.push(c);
                            self.bump();
                        }
                        Some('\\') if self.valid_escape_at(0) => s.push(self.consume_escaped()),
                        _ => break,
                    }
                }
                s
            }

            /// The numeric value plus whether the lexeme was an `<integer>`
            /// (no `.` fraction and no exponent).
            fn consume_number(&mut self) -> (f32, bool) {
                let mut s = String::new();
                let mut integer = true;
                if matches!(self.peek(), Some('+' | '-')) {
                    s.push(self.bump().unwrap_or('+'));
                }
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    s.push(self.bump().unwrap_or('0'));
                }
                if self.peek() == Some('.')
                    && matches!(self.peek_at(1), Some(c) if c.is_ascii_digit())
                {
                    integer = false;
                    s.push('.');
                    self.bump();
                    while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                        s.push(self.bump().unwrap_or('0'));
                    }
                }
                if matches!(self.peek(), Some('e' | 'E')) {
                    let save = self.pos;
                    self.bump();
                    let mut exp = String::from("e");
                    if matches!(self.peek(), Some('+' | '-')) {
                        exp.push(self.bump().unwrap_or('+'));
                    }
                    let digits = exp.len();
                    while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                        exp.push(self.bump().unwrap_or('0'));
                    }
                    if exp.len() > digits {
                        integer = false;
                        s.push_str(&exp);
                    } else {
                        self.pos = save;
                    }
                }
                (s.parse::<f32>().unwrap_or(0.0), integer)
            }

            fn consume_numeric(&mut self) -> Token {
                let (value, integer) = self.consume_number();
                if self.starts_ident() {
                    Token::Dimension {
                        value,
                        integer,
                        unit: self.consume_name(),
                    }
                } else if self.peek() == Some('%') {
                    self.bump();
                    Token::Percentage(value)
                } else {
                    Token::Number { value, integer }
                }
            }

            fn consume_string(&mut self) -> Token {
                let quote = self.bump().unwrap_or('"');
                let mut s = String::new();
                loop {
                    match self.peek() {
                        None => return Token::String(s),
                        Some(c) if c == quote => {
                            self.bump();
                            return Token::String(s);
                        }
                        Some('\n' | '\r' | '\x0c') => return Token::BadString,
                        Some('\\') => match self.peek_at(1) {
                            None => {
                                self.bump();
                            }
                            Some('\n' | '\x0c') => {
                                self.pos += 2;
                            }
                            Some('\r') => {
                                self.pos += 2;
                                if self.peek() == Some('\n') {
                                    self.pos += 1;
                                }
                            }
                            _ => s.push(self.consume_escaped()),
                        },
                        Some(c) => {
                            s.push(c);
                            self.bump();
                        }
                    }
                }
            }

            fn consume_bad_url_remainder(&mut self) -> Token {
                loop {
                    match self.peek() {
                        None => return Token::BadUrl,
                        Some(')') => {
                            self.bump();
                            return Token::BadUrl;
                        }
                        Some('\\') if self.valid_escape_at(0) => {
                            self.consume_escaped();
                        }
                        _ => {
                            self.bump();
                        }
                    }
                }
            }

            /// Called right after consuming `url(`.
            ///
            /// Reaching EOF before the closing `)` is a parse error: per CSS Syntax
            /// L3 the tokenizer flags it, but since this tokenizer carries no
            /// error channel, it yields [`Token::BadUrl`] instead of a [`Token::Url`]
            /// so the error is not silently lost — the declaration is dropped at
            /// value-parse time.
            fn consume_url(&mut self) -> Token {
                self.eat_while(is_css_ws);
                match self.peek() {
                    // Quoted: this is really a function token; the string follows.
                    Some('"' | '\'') => Token::Function("url".to_string()),
                    _ => {
                        let mut s = String::new();
                        loop {
                            match self.peek() {
                                None => return Token::BadUrl,
                                Some(')') => {
                                    self.bump();
                                    return Token::Url(s);
                                }
                                Some(c) if is_css_ws(c) => {
                                    self.eat_while(is_css_ws);
                                    match self.peek() {
                                        None => return Token::BadUrl,
                                        Some(')') => {
                                            self.bump();
                                            return Token::Url(s);
                                        }
                                        _ => return self.consume_bad_url_remainder(),
                                    }
                                }
                                Some('"' | '\'' | '(') => return self.consume_bad_url_remainder(),
                                Some(c) if (c as u32) < 0x20 || c == '\x7f' => {
                                    return self.consume_bad_url_remainder();
                                }
                                Some('\\') if self.valid_escape_at(0) => {
                                    s.push(self.consume_escaped())
                                }
                                Some('\\') => return self.consume_bad_url_remainder(),
                                Some(c) => {
                                    s.push(c);
                                    self.bump();
                                }
                            }
                        }
                    }
                }
            }

            fn consume_ident_like(&mut self) -> Token {
                let name = self.consume_name();
                if self.peek() == Some('(') {
                    self.bump();
                    if name.eq_ignore_ascii_case("url") {
                        self.consume_url()
                    } else {
                        Token::Function(name)
                    }
                } else {
                    Token::Ident(name)
                }
            }

            fn next_token(&mut self) -> Token {
                self.skip_comment();
                let c = match self.peek() {
                    None => return Token::Eof,
                    Some(c) => c,
                };
                match c {
                    c if is_css_ws(c) => {
                        self.eat_while(is_css_ws);
                        Token::Whitespace
                    }
                    '"' | '\'' => self.consume_string(),
                    '#' => {
                        if matches!(self.peek_at(1), Some(nc) if is_name_char(nc))
                            || self.valid_escape_at(1)
                        {
                            self.bump();
                            let id_flag = self.starts_ident();
                            Token::Hash(self.consume_name(), id_flag)
                        } else {
                            self.bump();
                            Token::Delim('#')
                        }
                    }
                    '(' => {
                        self.bump();
                        Token::LParen
                    }
                    ')' => {
                        self.bump();
                        Token::RParen
                    }
                    '[' => {
                        self.bump();
                        Token::LBracket
                    }
                    ']' => {
                        self.bump();
                        Token::RBracket
                    }
                    '{' => {
                        self.bump();
                        Token::LBrace
                    }
                    '}' => {
                        self.bump();
                        Token::RBrace
                    }
                    ',' => {
                        self.bump();
                        Token::Comma
                    }
                    ':' => {
                        self.bump();
                        Token::Colon
                    }
                    ';' => {
                        self.bump();
                        Token::Semicolon
                    }
                    '@' => {
                        if self.starts_ident_at(1) {
                            self.bump();
                            Token::AtKeyword(self.consume_name())
                        } else {
                            self.bump();
                            Token::Delim('@')
                        }
                    }
                    '<' => {
                        if self.peek_at(1) == Some('!')
                            && self.peek_at(2) == Some('-')
                            && self.peek_at(3) == Some('-')
                        {
                            self.pos += 4;
                            Token::Cdo
                        } else {
                            self.bump();
                            Token::Delim('<')
                        }
                    }
                    '-' => {
                        if self.starts_number() {
                            self.consume_numeric()
                        } else if self.peek_at(1) == Some('-') && self.peek_at(2) == Some('>') {
                            self.pos += 3;
                            Token::Cdc
                        } else if self.starts_ident() {
                            self.consume_ident_like()
                        } else {
                            self.bump();
                            Token::Delim('-')
                        }
                    }
                    '+' | '.' => {
                        if self.starts_number() {
                            self.consume_numeric()
                        } else {
                            self.bump();
                            Token::Delim(c)
                        }
                    }
                    c if c.is_ascii_digit() => self.consume_numeric(),
                    _ if self.starts_ident() => self.consume_ident_like(),
                    _ => {
                        self.bump();
                        Token::Delim(c)
                    }
                }
            }
        }
    }

    use self::old_impl::old_tokenize;
    use super::tokenize;

    /// Deterministic xorshift64: std-only rule means no `rand` crate.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Fixed corpus: every existing unit-test input plus edge cases aimed
    /// at the rewrite's risk areas — multi-byte chars at each tokenizer
    /// decision point, escapes (including astral/surrogate/overlong-ish
    /// values and escapes at EOF), and comment/string/url boundaries.
    const CORPUS: &[&str] = &[
        "",
        " ",
        "  \t\n",
        "a",
        "a/* hello */b",
        "a/**/b",
        "a/* never ends",
        "-webkit-x foo2 _bar",
        "rgb(",
        "a\\26 b",
        "\\",
        "\"hello\"",
        "'it\\'s'",
        "\"abc\ndef\"",
        "\"abc",
        "\"a\\\nb\"",
        "\"a\\\r\nb\"",
        "\"a\\\rb\"",
        "\"a\\\x0cb\"",
        "42",
        "42.0",
        "-4.5",
        "+.5",
        "1e3",
        "1E-2",
        "1e",
        "10px",
        "2.0n",
        "50%",
        ".",
        "1e+",
        "1e-",
        "1E+5",
        "--1",
        "+-1",
        ".5.5",
        "#id",
        "#123",
        "# ",
        "#é",
        "#\\26 ",
        "@media",
        "@-x",
        "@ 1",
        "@é",
        "url(foo.png)",
        "url(  foo.png  )",
        "url(\"foo.png\")",
        "url(a\"b)",
        "URL(x)",
        "url(",
        "url(foo",
        "url(  ",
        "url()",
        "url(é)",
        "url(\\26 )",
        "url(a\\ b)",
        "<!--",
        "-->",
        "--x",
        "<!--x",
        "-->x",
        "<!-",
        "-",
        "{}[]():;,",
        "16/9",
        "italic 12px/1.5 serif",
        "var(--x, 1px)",
        "rgb(1, 2, 3)",
        "#abc",
        "1.5em",
        "2fr",
        // Multi-byte chars at every tokenizer decision point.
        "é",
        "é(",
        "@é",
        "#é",
        "1é",
        "ééé",
        "/* é */",
        "/*é",
        "-é",
        "--é",
        ".é",
        "+é",
        "<é",
        "é:",
        "\"é\"",
        "'é'",
        "aéb",
        "é1",
        "é-",
        "éé(",
        "éé ",
        "\\é",
        "a\\é b",
        "😀",
        "😀😀",
        "a😀b",
        "中 文",
        "\u{0301}",
        "e\u{0301}",
        "\u{feff}a",
        "\u{200b}",
        "\u{7f}",
        "\u{80}",
        "\u{10ffff}",
        // Escapes: hex lengths, astral, surrogates (→ U+FFFD), NUL,
        // escapes at EOF, escaped newlines.
        "\\26 ",
        "\\000026",
        "\\000026X",
        "\\41",
        "\\000041x",
        "\\0 ",
        "\\D800 ",
        "\\DFFF ",
        "\\110000 ",
        "\\10FFFF",
        "\\10FFFFF",
        "\\ ",
        "\\\n",
        "a\\",
        "\\\r\n",
        "\\\x0c",
        "a\\\nb",
        "\\41bc",
        "\\41 bc",
        "\"\\26 \"",
        "\"\\\\\"",
        "\"\\\"\"",
        "\"a\\",
        // Comments with escapes, nesting attempts, multibyte.
        "/* \\ */",
        "/***/",
        "/****/",
        "a/* /* */b",
        "/* \u{1f600} */x",
        // Hash / at-keyword / ident edges.
        "#\\31 23",
        "#-",
        "#--",
        "@-",
        "@--",
        "--",
        "---",
        "---->",
        // Number edges.
        "12e",
        "12.5e",
        ".e5",
        "+.e5",
        "1.2.3",
        "1e1e1",
        // URL edges.
        "url( a b )",
        "url(a b)",
        "url(\\)",
        "url(\\",
        "URL(  x  )",
        "uRl(x)",
        "url(\u{7f})",
        // CDO/CDC edges.
        "<!---",
        "<!---x",
        "--->",
        "-- -->",
        "< !--",
    ];

    #[test]
    fn old_and_new_agree_on_corpus() {
        for input in CORPUS {
            assert_eq!(
                old_tokenize(input),
                tokenize(input),
                "tokenizer mismatch on {input:?}"
            );
        }
    }

    /// Structured fragments for fuzz input generation: ASCII punctuation,
    /// alphanumerics, multi-byte chars (2/3/4-byte, combining, controls,
    /// astral), and CSS-shaped fragments (urls, escapes, comments).
    const PIECES: &[&str] = &[
        "{",
        "}",
        "[",
        "]",
        "(",
        ")",
        ":",
        ";",
        ",",
        ".",
        "#",
        "@",
        "<",
        ">",
        "+",
        "-",
        "*",
        "/",
        "!",
        "=",
        "\"",
        "'",
        "\\",
        "%",
        "&",
        "|",
        "~",
        "^",
        "$",
        "?",
        "_",
        " ",
        "\t",
        "\n",
        "\r",
        "\x0c",
        "a",
        "Z",
        "0",
        "9",
        "e",
        "E",
        "é",
        "ü",
        "ñ",
        "中",
        "日",
        "😀",
        "🎉",
        "\u{0301}",
        "Ω",
        "ß",
        "\u{7f}",
        "\u{80}",
        "\u{200b}",
        "\u{feff}",
        "\u{10ffff}",
        "url(",
        "rgb(",
        "@media",
        "<!--",
        "-->",
        "/*",
        "*/",
        "\\26 ",
        "\\41",
        "1e",
        "e10",
        ".5",
        "10px",
        "50%",
        "#id",
        "\"str",
        "'q'",
        "--",
        "->",
        "\\\n",
        "\\\r\n",
        "\\\x0c",
        "\\0 ",
        "\\D800 ",
        "\\110000 ",
        "\\10FFFF",
    ];

    /// Raw char soup: every interesting char class, for unstructured fuzz.
    const SOUP: &[char] = &[
        'a', 'Z', '0', '9', 'e', 'E', '-', '_', ' ', '\t', '\n', '\r', '\x0c', '{', '}', '[', ']',
        '(', ')', ':', ';', ',', '.', '#', '@', '<', '>', '+', '*', '/', '!', '=', '"', '\'', '\\',
        '%', '&', '|', '~', '^', '$', '?', 'é', 'ü', 'ñ', '中', '😀', '\u{0301}', 'Ω', 'ß',
        '\u{7f}', '\u{80}', '\u{200b}', '\u{feff}',
    ];

    #[test]
    fn old_and_new_agree_on_fuzz() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for i in 0..40_000 {
            let mut s = String::new();
            if i % 4 == 0 {
                // Unstructured char soup.
                for _ in 0..rng.below(48) {
                    s.push(SOUP[rng.below(SOUP.len())]);
                }
            } else {
                // Structured fragments.
                for _ in 0..rng.below(40) {
                    s.push_str(PIECES[rng.below(PIECES.len())]);
                }
            }
            assert_eq!(
                old_tokenize(&s),
                tokenize(&s),
                "tokenizer mismatch on {s:?}"
            );
        }
    }
}

/// Wave 65 perf builder (`p-token`): allocation/timing probes and
/// regression tests for the tokenizer's no-escape fast paths.
///
/// The tokenizer must produce a bit-identical [`Token`] stream to the
/// pre-wave-65 implementation; the 40k-case differential fuzz in
/// `differential` is the primary guard. The tests here pin the new
/// code paths directly (span-sliced names/strings/urls, slice-parsed
/// numbers, allocation-free escape decoding) and carry the before/after
/// timing harness on the ~171KB fixture.
#[cfg(test)]
mod w65_p_token_tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Deterministic xorshift64 (same construction as `differential::Rng`).
    struct W65Rng(u64);
    impl W65Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Deterministic ~171KB stylesheet exercising every token kind:
    /// idents, functions, hashes, strings, urls, numbers, dimensions,
    /// percentages, comments, at-rules, custom properties. The exact
    /// bytes are fixed by the seed, so before/after timings compare
    /// the same input.
    fn w65_fixture_css() -> String {
        let mut rng = W65Rng(0x9E3779B97F4A7C15 ^ 0x77);
        let mut css = String::new();
        let mut i = 0usize;
        while css.len() < 171 * 1024 {
            i += 1;
            if i.is_multiple_of(11) {
                css.push_str("@media (min-width: ");
                css.push_str(&(100 + rng.below(900)).to_string());
                css.push_str("px) {\n");
            }
            match rng.below(6) {
                0 => {
                    css.push_str(".cls-");
                    css.push_str(&i.to_string());
                }
                1 => {
                    css.push_str("#id-");
                    css.push_str(&i.to_string());
                }
                2 => css.push_str("div"),
                3 => {
                    css.push_str("ul > li.item-");
                    css.push_str(&i.to_string());
                    css.push_str(":hover");
                }
                4 => {
                    css.push_str("a[href=\"x\"][data-n=\"");
                    css.push_str(&i.to_string());
                    css.push_str("\"]");
                }
                _ => css.push_str("section article p"),
            }
            css.push_str(" {\n");
            // Long-named custom property, as in real stylesheets.
            css.push_str("  --theme-variable-");
            css.push_str(&i.to_string());
            css.push_str("-with-a-long-name: ");
            css.push_str(&i.to_string());
            css.push_str("px;\n");
            let ndecl = 4 + rng.below(5);
            for _ in 0..ndecl {
                css.push_str("  ");
                match rng.below(14) {
                    0 => {
                        css.push_str("color: #");
                        css.push_str(&format!("{:06x}", rng.next() & 0xFF_FFFF));
                        css.push(';');
                    }
                    1 => {
                        css.push_str("background-color: rgba(");
                        for _ in 0..3 {
                            css.push_str(&rng.below(256).to_string());
                            css.push_str(", ");
                        }
                        css.push_str("0.5);");
                    }
                    2 => {
                        css.push_str("margin: ");
                        css.push_str(&rng.below(40).to_string());
                        css.push_str("px ");
                        css.push_str(&rng.below(40).to_string());
                        css.push_str("px;");
                    }
                    3 => {
                        css.push_str("padding: ");
                        css.push_str(&rng.below(3).to_string());
                        css.push('.');
                        css.push_str(&rng.below(10).to_string());
                        css.push_str("em;");
                    }
                    4 => {
                        css.push_str("border: ");
                        css.push_str(&rng.below(9).to_string());
                        css.push_str("px solid black;");
                    }
                    5 => {
                        css.push_str("font-size: ");
                        css.push_str(&rng.below(200).to_string());
                        css.push_str("%;");
                    }
                    6 => {
                        css.push_str("font-family: \"Family ");
                        css.push_str(&i.to_string());
                        css.push_str("\", sans-serif;");
                    }
                    7 => {
                        css.push_str("width: calc(100% - ");
                        css.push_str(&rng.below(400).to_string());
                        css.push_str("px);");
                    }
                    8 => {
                        css.push_str("background-image: url(img-");
                        css.push_str(&i.to_string());
                        css.push_str(".png);");
                    }
                    9 => {
                        css.push_str("content: \"x");
                        css.push_str(&i.to_string());
                        css.push_str("y\";");
                    }
                    10 => {
                        css.push_str("opacity: 0.");
                        css.push_str(&rng.below(10).to_string());
                        css.push(';');
                    }
                    11 => {
                        css.push_str("z-index: ");
                        css.push_str(&rng.below(100).to_string());
                        css.push(';');
                    }
                    12 => {
                        css.push_str("line-height: ");
                        css.push_str(&rng.below(3).to_string());
                        css.push('.');
                        css.push_str(&rng.below(10).to_string());
                        css.push(';');
                    }
                    _ => {
                        css.push_str("transform: translate(");
                        css.push_str(&rng.below(400).to_string());
                        css.push_str("px, ");
                        css.push_str(&rng.below(400).to_string());
                        css.push_str("px);");
                    }
                }
                css.push('\n');
            }
            css.push_str("}\n");
            if i.is_multiple_of(11) {
                css.push_str("/* rule group ");
                css.push_str(&i.to_string());
                css.push_str(" */\n");
            }
            if i % 11 == 5 {
                css.push_str("}\n");
            }
        }
        css
    }

    fn w65_best_of(iters: u32, mut f: impl FnMut()) -> Duration {
        let mut best = Duration::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed());
        }
        best
    }

    #[test]
    fn w65_fixture_is_stable_size() {
        let css = w65_fixture_css();
        assert_eq!(css, w65_fixture_css(), "fixture must be deterministic");
        assert!(
            (171 * 1024..172 * 1024).contains(&css.len()),
            "fixture size drifted: {} bytes",
            css.len()
        );
    }

    /// Timing probe: raw lexer throughput on the fixture. Prints, does
    /// not assert on time (timing is for the wave report, not CI).
    #[test]
    fn w65_tokenize_fixture_timing() {
        let css = w65_fixture_css();
        let toks = tokenize(&css);
        assert!(toks.len() > 10_000, "fixture tokenized to {}", toks.len());
        let best = w65_best_of(5, || {
            std::hint::black_box(tokenize(&css));
        });
        eprintln!(
            "w65: tokenize {} bytes -> {} tokens: {:.1}ms best-of-5 (dev)",
            css.len(),
            toks.len(),
            best.as_secs_f64() * 1000.0
        );
    }

    /// Timing probe: full stylesheet parse on the fixture — the wave-64
    /// metric (64.6 -> 46.4ms dev); the lexer win stacks on top of it.
    #[test]
    fn w65_stylesheet_parse_timing() {
        let css = w65_fixture_css();
        let sheet = crate::parse_stylesheet(&css);
        assert!(!sheet.rules().is_empty());
        let best = w65_best_of(5, || {
            std::hint::black_box(crate::parse_stylesheet(&css));
        });
        eprintln!(
            "w65: parse_stylesheet {} bytes ({} rules): {:.1}ms best-of-5 (dev)",
            css.len(),
            sheet.rules().len(),
            best.as_secs_f64() * 1000.0
        );
    }

    /// Numbers parse from the input slice without a scratch `String`;
    /// values and the `<integer>`-lexeme flag must be bit-identical.
    #[test]
    fn w65_number_slice_parse() {
        let num = |input: &str| {
            let toks = tokenize(input);
            assert_eq!(toks.len(), 2, "one token + eof for {input:?}");
            toks.into_iter().next().unwrap()
        };
        assert_eq!(
            num("42"),
            Token::Number {
                value: 42.0,
                integer: true
            }
        );
        assert_eq!(
            num("42.0"),
            Token::Number {
                value: 42.0,
                integer: false
            }
        );
        assert_eq!(
            num("-4.5"),
            Token::Number {
                value: -4.5,
                integer: false
            }
        );
        assert_eq!(
            num("+.5"),
            Token::Number {
                value: 0.5,
                integer: false
            }
        );
        assert_eq!(
            num("1e3"),
            Token::Number {
                value: 1000.0,
                integer: false
            }
        );
        assert_eq!(
            num("1E-2"),
            Token::Number {
                value: 0.01,
                integer: false
            }
        );
        // `1e` / `1e+`: the exponent needs digits, so this is a dimension
        // with unit "e" (exponent backtrack must restore the cursor).
        assert_eq!(
            tokenize("1e"),
            vec![
                Token::Dimension {
                    value: 1.0,
                    integer: true,
                    unit: "e".into()
                },
                Token::Eof
            ]
        );
        assert_eq!(
            tokenize("1e+"),
            vec![
                Token::Dimension {
                    value: 1.0,
                    integer: true,
                    unit: "e".into()
                },
                Token::Delim('+'),
                Token::Eof
            ]
        );
        assert_eq!(
            num("16777217"),
            Token::Number {
                value: 16777216.0,
                integer: true
            }
        );
    }

    /// Name fast path (ASCII run, no escapes) vs slow path (escapes,
    /// non-ASCII): identical token text either way.
    #[test]
    fn w65_name_fast_and_slow_paths() {
        let ident = |input: &str| {
            let toks = tokenize(input);
            assert_eq!(toks.len(), 2, "one token + eof for {input:?}");
            toks.into_iter().next().unwrap()
        };
        // Pure ASCII: fast path.
        assert_eq!(
            ident("background-color"),
            Token::Ident("background-color".into())
        );
        assert_eq!(ident("-webkit-x"), Token::Ident("-webkit-x".into()));
        // Non-ASCII name chars: slow path, same text.
        assert_eq!(ident("école"), Token::Ident("école".into()));
        assert_eq!(ident("aéb"), Token::Ident("aéb".into()));
        assert_eq!(ident("abé"), Token::Ident("abé".into()));
        assert_eq!(ident("中"), Token::Ident("中".into()));
        // Escapes: slow path with decoded text. `\41bc` eats four hex
        // digits (U+41BC); a trailing space terminates the escape.
        assert_eq!(ident("a\\26 b"), Token::Ident("a&b".into()));
        assert_eq!(ident("\\41bc"), Token::Ident("䆼".into()));
        assert_eq!(ident("\\41 bc"), Token::Ident("Abc".into()));
        assert_eq!(ident("\\é"), Token::Ident("é".into()));
        // Backslash that is NOT a valid escape ends the name.
        assert_eq!(
            tokenize("a\\\nb"),
            vec![
                Token::Ident("a".into()),
                Token::Delim('\\'),
                Token::Whitespace,
                Token::Ident("b".into()),
                Token::Eof,
            ]
        );
        // Hash / at-keyword / dimension-unit go through the same name path.
        // `#\31 23`: the escape makes starts_ident true (any valid
        // escape counts), and decodes to '1', so the name is "123".
        assert_eq!(
            tokenize("#\\31 23"),
            vec![Token::Hash("123".into(), true), Token::Eof]
        );
        assert_eq!(
            tokenize("@m\\65 dia"),
            vec![Token::AtKeyword("media".into()), Token::Eof]
        );
        assert_eq!(
            tokenize("10p\\78"),
            vec![
                Token::Dimension {
                    value: 10.0,
                    integer: true,
                    unit: "px".into()
                },
                Token::Eof
            ]
        );
    }

    /// String fast path (no backslash/newline) vs slow path (escapes,
    /// continuations, multibyte).
    #[test]
    fn w65_string_fast_and_slow_paths() {
        let str_tok = |input: &str| {
            let toks = tokenize(input);
            assert_eq!(toks.len(), 2, "one token + eof for {input:?}");
            toks.into_iter().next().unwrap()
        };
        assert_eq!(str_tok("\"hello\""), Token::String("hello".into()));
        assert_eq!(str_tok("\"héllo→\""), Token::String("héllo→".into()));
        assert_eq!(str_tok("\"\""), Token::String("".into()));
        assert_eq!(str_tok("\"abc"), Token::String("abc".into()));
        assert_eq!(str_tok("\"a\\26 b\""), Token::String("a&b".into()));
        assert_eq!(str_tok("\"a\\\nb\""), Token::String("ab".into()));
        assert_eq!(str_tok("\"aé\\26 b\""), Token::String("aé&b".into()));
        // Bad strings: newline ends the token, newline not consumed.
        assert_eq!(
            tokenize("\"abc\ndef\""),
            vec![
                Token::BadString,
                Token::Whitespace,
                Token::Ident("def".into()),
                Token::String("".into()),
                Token::Eof,
            ]
        );
    }

    /// Unquoted-url fast path (no backslash) vs slow path (escapes).
    #[test]
    fn w65_url_fast_and_slow_paths() {
        let url_tok = |input: &str| {
            let toks = tokenize(input);
            assert_eq!(toks.len(), 2, "one token + eof for {input:?}");
            toks.into_iter().next().unwrap()
        };
        assert_eq!(url_tok("url(foo.png)"), Token::Url("foo.png".into()));
        assert_eq!(url_tok("url(  foo.png  )"), Token::Url("foo.png".into()));
        assert_eq!(url_tok("url(é)"), Token::Url("é".into()));
        assert_eq!(url_tok("url()"), Token::Url("".into()));
        // Escape inside the URL: slow path, decoded.
        assert_eq!(url_tok("url(\\26 )"), Token::Url("&".into()));
        assert_eq!(url_tok("url(a\\ b)"), Token::Url("a b".into()));
        // Poisoned URLs still go bad.
        assert_eq!(url_tok("url(a\"b)"), Token::BadUrl);
        assert_eq!(url_tok("url(a b)"), Token::BadUrl);
        assert_eq!(url_tok("url(\u{1})"), Token::BadUrl);
    }

    /// Escape decoding accumulates hex without a scratch `String`;
    /// astral, surrogate (→ U+FFFD), and NUL values must match.
    #[test]
    fn w65_escape_decoding() {
        let ident = |input: &str| match tokenize(input).into_iter().next().unwrap() {
            Token::Ident(s) => s,
            t => panic!("expected ident, got {t:?}"),
        };
        assert_eq!(ident("\\41"), "A");
        assert_eq!(ident("\\000041x"), "Ax");
        assert_eq!(ident("\\10FFFF"), "\u{10FFFF}");
        assert_eq!(ident("\\D800 "), "\u{FFFD}");
        // `\0` decodes to NUL itself here (`char::from_u32(0)` succeeds):
        // pre-existing behavior, kept bit-identical.
        assert_eq!(ident("\\0 "), "\0");
        assert_eq!(ident("\\110000 "), "\u{FFFD}");
    }

    /// Comment skipping over multibyte content and unterminated comments.
    #[test]
    fn w65_comment_byte_scan() {
        assert_eq!(
            tokenize("a/* \\ */b"),
            vec![
                Token::Ident("a".into()),
                Token::Ident("b".into()),
                Token::Eof
            ]
        );
        assert_eq!(
            tokenize("a/* \u{1f600} */b"),
            vec![
                Token::Ident("a".into()),
                Token::Ident("b".into()),
                Token::Eof
            ]
        );
        assert_eq!(
            tokenize("/***/x"),
            vec![Token::Ident("x".into()), Token::Eof]
        );
        assert_eq!(
            tokenize("a/* never ends"),
            vec![Token::Ident("a".into()), Token::Eof]
        );
    }
}
