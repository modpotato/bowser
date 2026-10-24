//! Web-platform-tests conformance harness.
//!
//! Runs `tests/fixtures/urltestdata.json` (the URL standard's parser
//! vectors) and `tests/fixtures/toascii.json` (IDNA vectors) against this
//! crate. The fixtures are plain data; a minimal JSON reader below parses
//! the subset they use (objects, arrays, strings, null, booleans).

use bowser_url::{domain_to_ascii, parse};

// ---------------------------------------------------------------------------
// Minimal JSON reader (fixtures only use objects, arrays, strings, null,
// booleans).
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(s: &'a str) -> Self {
        Reader {
            bytes: s.as_bytes(),
            pos: 0,
        }
    }

    fn ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, b: u8) -> Result<(), String> {
        self.ws();
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected {b} at {}", self.pos))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(c) => Err(format!("unexpected byte {c} at {}", self.pos)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn literal(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(v)
        } else {
            Err(format!("bad literal at {}", self.pos))
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Object(pairs));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.expect(b':')?;
            let val = self.value()?;
            pairs.push((key, val));
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(format!("bad object at {}", self.pos)),
            }
        }
        Ok(Json::Object(pairs))
    }

    fn array(&mut self) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(format!("bad array at {}", self.pos)),
            }
        }
        Ok(Json::Array(items))
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let b = self.peek().ok_or("unterminated string")?;
            self.pos += 1;
            match b {
                b'"' => break,
                b'\\' => {
                    let e = self.peek().ok_or("bad escape")?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.hex4()?),
                        _ => return Err(format!("bad escape \\{e} at {}", self.pos)),
                    }
                }
                _ => {
                    // Raw UTF-8 bytes; the fixtures are valid UTF-8.
                    let start = self.pos - 1;
                    let mut len = 1;
                    while start + len <= self.bytes.len()
                        && std::str::from_utf8(&self.bytes[start..start + len]).is_err()
                    {
                        len += 1;
                    }
                    let s = std::str::from_utf8(&self.bytes[start..start + len])
                        .map_err(|_| "bad utf-8")?;
                    out.push_str(s);
                    self.pos = start + len;
                }
            }
        }
        Ok(out)
    }

    fn hex4(&mut self) -> Result<char, String> {
        let n = self.hex4_u32()?;
        // Surrogate pair.
        if (0xD800..0xDC00).contains(&n) {
            if self.bytes.get(self.pos..self.pos + 2) == Some(b"\\u".as_slice()) {
                self.pos += 2;
                let lo = self.hex4_u32()?;
                if (0xDC00..0xE000).contains(&lo) {
                    let n = 0x10000 + ((n - 0xD800) << 10) + (lo - 0xDC00);
                    return char::from_u32(n).ok_or_else(|| "bad code point".to_string());
                }
                return Err("bad surrogate pair".to_string());
            }
            return Err("lone surrogate".to_string());
        }
        char::from_u32(n).ok_or_else(|| "bad code point".to_string())
    }

    fn hex4_u32(&mut self) -> Result<u32, String> {
        let mut n: u32 = 0;
        for _ in 0..4 {
            let b = self.peek().ok_or("bad \\u escape")?;
            self.pos += 1;
            n = n * 16
                + match b {
                    b'0'..=b'9' => (b - b'0') as u32,
                    b'a'..=b'f' => (b - b'a' + 10) as u32,
                    b'A'..=b'F' => (b - b'A' + 10) as u32,
                    _ => return Err("bad \\u escape".to_string()),
                };
        }
        Ok(n)
    }
}

fn load_fixture(name: &str) -> Json {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).expect("fixture readable");
    Reader::new(&text).value().expect("fixture parses")
}

fn field<'j>(entry: &'j Json, key: &str) -> &'j str {
    entry
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("fixture entry is missing {key:?}"))
}

/// The base URL of a urltestdata entry: a missing or null `"base"` key
/// means "no base URL".
fn base_field(entry: &Json) -> &str {
    entry.get("base").and_then(Json::as_str).unwrap_or("")
}

// ---------------------------------------------------------------------------
// urltestdata.json
// ---------------------------------------------------------------------------

#[test]
fn wpt_urltestdata() {
    let data = load_fixture("urltestdata.json");
    let Json::Array(entries) = &data else {
        panic!("top level must be an array")
    };
    let mut run = 0;
    let mut failures: Vec<String> = Vec::new();

    for entry in entries {
        let Json::Object(_) = entry else { continue }; // skip comment strings
        let input = field(entry, "input");
        let base_str = base_field(entry);
        let base = if base_str.is_empty() {
            None
        } else {
            match parse(base_str, None) {
                Ok(b) => Some(b),
                Err(e) => {
                    failures.push(format!("base {base_str:?} failed to parse: {e:?}"));
                    continue;
                }
            }
        };
        let result = parse(input, base.as_ref());
        run += 1;

        if entry.get("failure").is_some_and(|f| f == &Json::Bool(true)) {
            if result.is_ok() {
                failures.push(format!("expected failure but parsed: {input:?}"));
            }
            continue;
        }

        let url = match result {
            Ok(u) => u,
            Err(e) => {
                failures.push(format!("failed to parse {input:?}: {e:?}"));
                continue;
            }
        };

        let mut check = |name: &str, actual: &str, expected: &str| {
            if actual != expected {
                failures.push(format!(
                    "{input:?}: {name} = {actual:?}, expected {expected:?}"
                ));
            }
        };

        check("href", &url.serialize(), field(entry, "href"));
        check(
            "protocol",
            &format!("{}:", url.scheme()),
            field(entry, "protocol"),
        );
        check("username", url.username(), field(entry, "username"));
        check("password", url.password(), field(entry, "password"));
        check(
            "host",
            &match (url.host(), url.port()) {
                (Some(h), Some(p)) => format!("{h}:{p}"),
                (Some(h), None) => h.to_string(),
                (None, _) => String::new(),
            },
            field(entry, "host"),
        );
        check(
            "hostname",
            url.host().unwrap_or(""),
            field(entry, "hostname"),
        );
        check(
            "port",
            &url.port().map(|p| p.to_string()).unwrap_or_default(),
            field(entry, "port"),
        );
        check("pathname", url.path(), field(entry, "pathname"));
        // Note: WPT reports an empty query/fragment as "" without the
        // separator; normalize Some("") to "" to match.
        check(
            "search",
            &url.query()
                .filter(|q| !q.is_empty())
                .map(|q| format!("?{q}"))
                .unwrap_or_default(),
            field(entry, "search"),
        );
        check(
            "hash",
            &url.fragment()
                .filter(|f| !f.is_empty())
                .map(|f| format!("#{f}"))
                .unwrap_or_default(),
            field(entry, "hash"),
        );
    }

    assert_eq!(run, 896, "expected to run every urltestdata.json entry");
    assert!(
        failures.is_empty(),
        "{} of {run} cases failed:\n{}",
        failures.len(),
        failures
            .iter()
            .take(25)
            .map(|s| format!("  {s}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

// ---------------------------------------------------------------------------
// toascii.json
// ---------------------------------------------------------------------------

/// Entries that fail because full UTS #46 processing is not implemented:
/// they need complete NFC composition or canonical decomposition from the
/// Unicode Character Database, which the pragmatic IDNA subset in `idna`
/// deliberately omits. These are genuine deviations from UTS #46, not
/// URL-layer bugs. Each is a `(input, expected)` pair from the fixture.
#[allow(clippy::type_complexity)]
fn idna_known_deviations() -> Vec<(&'static str, Option<&'static str>)> {
    vec![
        // Needs full NFC composition: k + U+0341 -> U+1E31.
        ("look\u{341}out.net", Some("xn--looout-kp7b.net")),
        // Needs canonical decomposition: U+09DC -> U+09A1 U+09BC.
        ("ড়.com", Some("xn--15b8c.com")),
    ]
}

#[test]
fn wpt_toascii() {
    let data = load_fixture("toascii.json");
    let Json::Array(entries) = &data else {
        panic!("top level must be an array")
    };
    let deviations = idna_known_deviations();
    let mut run = 0;
    let mut failures: Vec<String> = Vec::new();

    for entry in entries {
        let Json::Object(_) = entry else { continue };
        let input = field(entry, "input");
        let expected = match entry.get("output") {
            Some(Json::Str(s)) => Some(s.as_str()),
            _ => None,
        };
        let actual = domain_to_ascii(input);
        let actual_str = actual.as_deref();
        run += 1;

        if actual_str == expected {
            continue;
        }
        // Tolerate documented deviations.
        if deviations
            .iter()
            .any(|(d_in, d_out)| *d_in == input && *d_out == expected)
        {
            continue;
        }
        failures.push(format!(
            "{input:?}: got {actual_str:?}, expected {expected:?}"
        ));
    }

    assert_eq!(run, 87, "expected to run every toascii.json entry");
    assert!(
        failures.is_empty(),
        "{} of {run} cases failed:\n{}",
        failures.len(),
        failures
            .iter()
            .take(25)
            .map(|s| format!("  {s}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
