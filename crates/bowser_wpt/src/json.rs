//! Minimal JSON parser, standard library only.
//!
//! Exists because this crate is not allowed dependencies (no serde). It is
//! deliberately small: it supports the full JSON grammar but optimizes for
//! readability, not speed. The WPT manifest is tens of megabytes; if parsing
//! ever shows up in a profile, replace the tree-building step with a
//! streaming scanner over the `items` object.

use std::fmt;

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Look up a key in an object value.
    pub(crate) fn get(&self, key: &str) -> Option<&Json> {
        if let Json::Obj(pairs) = self {
            pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        } else {
            None
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }

    pub(crate) fn as_arr(&self) -> Option<&[Json]> {
        if let Json::Arr(a) = self {
            Some(a)
        } else {
            None
        }
    }

    pub(crate) fn as_obj(&self) -> Option<&[(String, Json)]> {
        if let Json::Obj(o) = self {
            Some(o)
        } else {
            None
        }
    }
}

/// A JSON parse failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParseError {
    /// Byte offset in the input where the failure was detected.
    pub pos: usize,
    pub msg: &'static str,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JSON parse error at byte {}: {}", self.pos, self.msg)
    }
}

impl std::error::Error for ParseError {}

/// Parse a complete JSON document. Trailing garbage is an error.
pub(crate) fn parse(text: &str) -> Result<Json, ParseError> {
    let mut p = Parser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(p.err("trailing data after JSON value"));
    }
    Ok(v)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: &'static str) -> ParseError {
        ParseError { pos: self.pos, msg }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) {
        self.pos += 1;
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.bump();
        }
    }

    fn parse_value(&mut self) -> Result<Json, ParseError> {
        self.skip_ws();
        match self.peek() {
            Some(b'n') => self.parse_lit("null", Json::Null),
            Some(b't') => self.parse_lit("true", Json::Bool(true)),
            Some(b'f') => self.parse_lit("false", Json::Bool(false)),
            Some(b'"') => Ok(Json::Str(self.parse_string()?)),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            _ => Err(self.err("expected a JSON value")),
        }
    }

    fn parse_lit(&mut self, lit: &str, val: Json) -> Result<Json, ParseError> {
        if self.bytes[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            Ok(val)
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        debug_assert_eq!(self.peek(), Some(b'"'));
        self.bump(); // opening quote
        let mut out = String::new();
        loop {
            let b = self.peek().ok_or_else(|| self.err("unterminated string"))?;
            match b {
                b'"' => {
                    self.bump();
                    return Ok(out);
                }
                b'\\' => {
                    self.bump();
                    let e = self.peek().ok_or_else(|| self.err("unterminated escape"))?;
                    self.bump();
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\x08'),
                        b'f' => out.push('\x0C'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.parse_unicode_escape()?),
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                0x00..=0x1F => return Err(self.err("unescaped control character")),
                _ => {
                    // Consume one full UTF-8 code point.
                    let s = std::str::from_utf8(&self.bytes[self.pos..])
                        .map_err(|_| self.err("invalid UTF-8"))?;
                    let ch = s
                        .chars()
                        .next()
                        .ok_or_else(|| self.err("unterminated string"))?;
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let mut v: u32 = 0;
        for _ in 0..4 {
            let b = self
                .peek()
                .ok_or_else(|| self.err("truncated unicode escape"))?;
            let d = (b as char)
                .to_digit(16)
                .ok_or_else(|| self.err("bad hex digit"))?;
            v = v * 16 + d;
            self.bump();
        }
        Ok(v)
    }

    fn parse_unicode_escape(&mut self) -> Result<char, ParseError> {
        let hi = self.hex4()?;
        let cp = if (0xD800..0xDC00).contains(&hi) {
            // Surrogate pair.
            if self.peek() == Some(b'\\') {
                self.bump();
                if self.peek() == Some(b'u') {
                    self.bump();
                    let lo = self.hex4()?;
                    if (0xDC00..0xE000).contains(&lo) {
                        0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                    } else {
                        return Err(self.err("bad low surrogate"));
                    }
                } else {
                    return Err(self.err("bad low surrogate"));
                }
            } else {
                return Err(self.err("lone high surrogate"));
            }
        } else if (0xDC00..0xE000).contains(&hi) {
            return Err(self.err("lone low surrogate"));
        } else {
            hi
        };
        char::from_u32(cp).ok_or_else(|| self.err("invalid code point"))
    }

    fn parse_number(&mut self) -> Result<Json, ParseError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.bump();
        }
        match self.peek() {
            Some(b'0') => self.bump(),
            Some(c) if c.is_ascii_digit() => {
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.bump();
                }
            }
            _ => return Err(self.err("bad number")),
        }
        if self.peek() == Some(b'.') {
            self.bump();
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.err("bad number"));
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.bump();
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.bump();
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.bump();
            }
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.err("bad number"));
            }
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.bump();
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.err("bad number"))?;
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| self.err("bad number"))
    }

    fn parse_array(&mut self) -> Result<Json, ParseError> {
        debug_assert_eq!(self.peek(), Some(b'['));
        self.bump();
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.bump();
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.bump();
                }
                Some(b']') => {
                    self.bump();
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
    }

    fn parse_object(&mut self) -> Result<Json, ParseError> {
        debug_assert_eq!(self.peek(), Some(b'{'));
        self.bump();
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.bump();
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected string key in object"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(self.err("expected ':' in object"));
            }
            self.bump();
            let val = self.parse_value()?;
            pairs.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.bump();
                }
                Some(b'}') => {
                    self.bump();
                    return Ok(Json::Obj(pairs));
                }
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars() {
        assert_eq!(parse("null").unwrap(), Json::Null);
        assert_eq!(parse("true").unwrap(), Json::Bool(true));
        assert_eq!(parse("false").unwrap(), Json::Bool(false));
        assert_eq!(parse("42").unwrap(), Json::Num(42.0));
        assert_eq!(parse("-1.5e3").unwrap(), Json::Num(-1500.0));
        assert_eq!(parse("\"hi\"").unwrap(), Json::Str("hi".into()));
    }

    #[test]
    fn escapes_and_unicode() {
        assert_eq!(
            parse(r#""a\nb\t\"q\"\\""#).unwrap(),
            Json::Str("a\nb\t\"q\"\\".into())
        );
        assert_eq!(parse(r#""caf\u00e9""#).unwrap(), Json::Str("café".into()));
        assert_eq!(parse(r#""\uD83D\uDE00""#).unwrap(), Json::Str("😀".into()));
    }

    #[test]
    fn nested() {
        let v = parse(r#"{"a":[1,{"b":null}],"c":"x"}"#).unwrap();
        assert_eq!(v.get("c").and_then(Json::as_str), Some("x"));
        let a = v.get("a").and_then(Json::as_arr).unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(a[0], Json::Num(1.0));
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse("{").is_err());
        assert!(parse("[1,]").is_err());
        assert!(parse("nul").is_err());
        assert!(parse("1 2").is_err());
        assert!(parse("\"\\x\"").is_err());
        assert!(parse("\"bad\ttab\"").is_err());
    }
}
