//! BOM sniffing and the HTML Standard's prescan algorithm hooks.

use super::{Encoding, encoding_from_label};

/// WHATWG "BOM sniff": the encoding a byte order mark declares, if any.
#[must_use]
pub fn sniff_bom(bytes: &[u8]) -> Option<Encoding> {
    strip_bom(bytes).0
}

/// Split off a byte order mark: `(encoding, rest)`.
pub(crate) fn strip_bom(bytes: &[u8]) -> (Option<Encoding>, &[u8]) {
    if let [0xEF, 0xBB, 0xBF, rest @ ..] = bytes {
        (Some(Encoding::Utf8), rest)
    } else if let [0xFE, 0xFF, rest @ ..] = bytes {
        (Some(Encoding::Utf16Be), rest)
    } else if let [0xFF, 0xFE, rest @ ..] = bytes {
        (Some(Encoding::Utf16Le), rest)
    } else {
        (None, bytes)
    }
}

/// HTML Standard "prescan a byte stream to determine its encoding".
///
/// Scans `bytes` (callers typically pass the first 1024) for a `<meta>`
/// charset declaration. Returns the declared encoding, or `None` when the
/// prescan finds nothing (callers then fall back to a default, usually
/// UTF-8 or a locale default).
///
/// If the byte stream runs out mid-scan, the spec falls back to
/// [`xml_encoding`]; that fallback is applied here too.
#[must_use]
pub fn prescan_encoding(bytes: &[u8]) -> Option<Encoding> {
    Prescanner { bytes, pos: 0 }.run()
}

/// HTML Standard "get an XML encoding": sniff `<?xml ... encoding="..."?>`.
#[must_use]
pub fn xml_encoding(bytes: &[u8]) -> Option<Encoding> {
    let n = bytes.len();
    if n < 5 || bytes[..5] != [0x3C, 0x3F, 0x78, 0x6D, 0x6C] {
        return None; // not `<?xml`
    }
    let mut p = 5usize; // past `<?xml`
    let decl_end = bytes.iter().position(|&b| b == 0x3E)?;
    let needle = b"encoding";
    let mut found = None;
    let mut j = p;
    while j + needle.len() <= decl_end {
        if &bytes[j..j + needle.len()] == needle {
            found = Some(j);
            break;
        }
        j += 1;
    }
    p = found? + needle.len();
    while p < n && bytes[p] <= 0x20 {
        p += 1;
    }
    if p >= n || bytes[p] != 0x3D {
        return None;
    }
    p += 1;
    while p < n && bytes[p] <= 0x20 {
        p += 1;
    }
    if p >= n {
        return None;
    }
    let quote = bytes[p];
    if quote != 0x22 && quote != 0x27 {
        return None;
    }
    p += 1;
    let end = bytes[p..].iter().position(|&b| b == quote)? + p;
    let value = &bytes[p..end];
    if value.iter().any(|&b| b <= 0x20) {
        return None;
    }
    // Isomorphic decode: bytes < 0x80 map 1:1; anything else fails the label.
    if value.iter().any(|&b| b >= 0x80) {
        return None;
    }
    let label = core::str::from_utf8(value).ok()?;
    match encoding_from_label(label) {
        Some(Encoding::Utf16Be | Encoding::Utf16Le) => Some(Encoding::Utf8),
        other => other,
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, 0x09 | 0x0A | 0x0C | 0x0D | 0x20)
}

/// Extract a character encoding from a `<meta>` content attribute value,
/// per the HTML Standard ("the algorithm for extracting a character encoding
/// from a meta element"). Returns `None` when there is nothing usable.
fn extract_meta_charset(s: &str) -> Option<Encoding> {
    let b = s.as_bytes();
    let mut pos = 0usize;
    loop {
        // Find "charset" (ASCII case-insensitive) at/after pos.
        let mut found = None;
        let mut j = pos;
        while j + 7 <= b.len() {
            if b[j..j + 7].eq_ignore_ascii_case(b"charset") {
                found = Some(j);
                break;
            }
            j += 1;
        }
        let j = found?;
        pos = j + 7;
        while pos < b.len() && b[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= b.len() {
            return None;
        }
        if b[pos] != b'=' {
            continue; // next char is skipped by the fresh search below
        }
        pos += 1;
        while pos < b.len() && b[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= b.len() {
            return None;
        }
        let c = b[pos];
        if c == b'"' || c == b'\'' {
            let rest = &b[pos + 1..];
            let end = rest.iter().position(|&x| x == c)?;
            return encoding_from_label(core::str::from_utf8(&rest[..end]).ok()?);
        }
        let mut end = pos;
        while end < b.len() && !b[end].is_ascii_whitespace() && b[end] != b';' {
            end += 1;
        }
        return encoding_from_label(core::str::from_utf8(&b[pos..end]).ok()?);
    }
}

/// Abort signal: the prescan ran out of bytes (or hit the end condition).
struct Abort;

struct Prescanner<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Prescanner<'a> {
    fn byte(&self) -> Result<u8, Abort> {
        self.bytes.get(self.pos).copied().ok_or(Abort)
    }

    fn starts_with(&self, pat: &[u8]) -> bool {
        self.bytes
            .get(self.pos..)
            .is_some_and(|s| s.starts_with(pat))
    }

    fn run(mut self) -> Option<Encoding> {
        let result = self.scan_loop();
        match result {
            Ok(enc) => enc,
            // Out of bytes / end condition: fall back to XML sniffing.
            Err(Abort) => xml_encoding(self.bytes),
        }
    }

    /// The spec's loop; `Ok(Some)` is a definitive answer, `Err(Abort)` means
    /// out of bytes / end condition (caller applies the XML fallback).
    fn scan_loop(&mut self) -> Result<Option<Encoding>, Abort> {
        let n = self.bytes.len();
        loop {
            if self.pos >= n {
                // Ran out of bytes: abort (caller falls back to XML sniffing).
                return Err(Abort);
            }
            // UTF-16 `<?xml` sniff (spec loop branches).
            if self.starts_with(&[0x3C, 0x00, 0x3F, 0x00, 0x78, 0x00]) {
                return Ok(Some(Encoding::Utf16Le));
            }
            if self.starts_with(&[0x00, 0x3C, 0x00, 0x3F, 0x00, 0x78]) {
                return Ok(Some(Encoding::Utf16Be));
            }
            let b = self.bytes[self.pos];
            if b != 0x3C {
                self.pos += 1; // any other byte: next byte
                continue;
            }
            if self.starts_with(b"<!--") {
                // To the first 0x3E preceded by two 0x2D, after the 0x3C.
                let mut k = self.pos + 1;
                let mut found = None;
                while k < n {
                    if self.bytes[k] == 0x3E
                        && k >= 2
                        && self.bytes[k - 2] == 0x2D
                        && self.bytes[k - 1] == 0x2D
                    {
                        found = Some(k);
                        break;
                    }
                    k += 1;
                }
                self.pos = found.ok_or(Abort)?; // then: next byte
            } else if self.is_meta_open() {
                // `<meta` (ASCII case-insensitive) + space or slash.
                self.pos += 5; // point at the delimiter after "meta"
                if let Some(enc) = self.scan_meta()? {
                    return Ok(Some(enc));
                }
            } else if self.is_tag_open() {
                // `<`, optional `/`, then ASCII letter: skip attributes.
                while self.pos < n
                    && !matches!(
                        self.bytes[self.pos],
                        0x09 | 0x0A | 0x0C | 0x0D | 0x20 | 0x3E
                    )
                {
                    self.pos += 1;
                }
                while self.get_attribute()?.is_some() {}
            } else if self.starts_with(b"<!") || self.starts_with(b"</") || self.starts_with(b"<?")
            {
                // To the first 0x3E after the 0x3C.
                let mut k = self.pos + 1;
                while k < n && self.bytes[k] != 0x3E {
                    k += 1;
                }
                if k >= n {
                    return Err(Abort);
                }
                self.pos = k; // then: next byte
            } else {
                // `<` followed by anything else: next byte.
            }
            self.pos += 1; // next byte
        }
    }

    fn is_meta_open(&self) -> bool {
        let b = self.bytes;
        let p = self.pos;
        p + 6 <= b.len()
            && b[p] == 0x3C
            && matches!(b[p + 1], 0x4D | 0x6D)
            && matches!(b[p + 2], 0x45 | 0x65)
            && matches!(b[p + 3], 0x54 | 0x74)
            && matches!(b[p + 4], 0x41 | 0x61)
            && matches!(b[p + 5], 0x09 | 0x0A | 0x0C | 0x0D | 0x20 | 0x2F)
    }

    fn is_tag_open(&self) -> bool {
        let b = self.bytes;
        let p = self.pos;
        if p + 2 > b.len() || b[p] != 0x3C {
            return false;
        }
        let mut q = p + 1;
        if b[q] == 0x2F {
            q += 1;
        }
        q < b.len() && b[q].is_ascii_alphabetic()
    }

    /// The `<meta>` attribute scan. `Ok(Some(enc))` = definitive answer.
    fn scan_meta(&mut self) -> Result<Option<Encoding>, Abort> {
        #[derive(PartialEq)]
        enum Charset {
            Null,
            Failure,
            Enc(Encoding),
        }
        let mut seen: Vec<String> = Vec::new();
        let mut got_pragma = false;
        let mut need_pragma: Option<bool> = None;
        let mut charset = Charset::Null;
        while let Some((name, value)) = self.get_attribute()? {
            if seen.contains(&name) {
                continue;
            }
            seen.push(name.clone());
            match name.as_str() {
                "http-equiv" => {
                    if value == "content-type" {
                        got_pragma = true;
                    }
                }
                "content" => {
                    if charset == Charset::Null {
                        match extract_meta_charset(&value) {
                            Some(enc) => {
                                charset = Charset::Enc(enc);
                                need_pragma = Some(true);
                            }
                            None => {
                                need_pragma = Some(true);
                            }
                        }
                    }
                }
                "charset" => {
                    charset = match encoding_from_label(&value) {
                        Some(enc) => Charset::Enc(enc),
                        None => Charset::Failure,
                    };
                    need_pragma = Some(false);
                }
                _ => {}
            }
        }
        // Processing step.
        let need = match need_pragma {
            None => return Ok(None), // next byte
            Some(v) => v,
        };
        if need && !got_pragma {
            return Ok(None); // next byte
        }
        let enc = match charset {
            Charset::Null | Charset::Failure => return Ok(None), // next byte
            Charset::Enc(e) => e,
        };
        let enc = match enc {
            Encoding::Utf16Be | Encoding::Utf16Le => Encoding::Utf8,
            Encoding::XUserDefined => Encoding::Windows1252,
            e => e,
        };
        Ok(Some(enc))
    }

    /// "get an attribute": `Ok(None)` = there isn't one; `Err(Abort)` = out
    /// of bytes. Names and values are ASCII-lowercased per the spec.
    fn get_attribute(&mut self) -> Result<Option<(String, String)>, Abort> {
        // Skip whitespace and slashes.
        loop {
            let b = self.byte()?;
            if is_ws(b) || b == 0x2F {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.byte()? == 0x3E {
            return Ok(None);
        }
        let mut name = String::new();
        let mut value = String::new();
        // Attribute name.
        let mut has_value = false;
        loop {
            let b = self.byte()?;
            if b == 0x3D && !name.is_empty() {
                self.pos += 1;
                has_value = true;
                break;
            } else if is_ws(b) {
                break; // -> spaces step
            } else if b == 0x2F || b == 0x3E {
                return Ok(Some((name, String::new())));
            } else if b.is_ascii_uppercase() {
                name.push((b + 0x20) as char);
                self.pos += 1;
            } else {
                name.push(b as char);
                self.pos += 1;
            }
        }
        if !has_value {
            // Spaces step: skip whitespace, require '='.
            loop {
                let b = self.byte()?;
                if is_ws(b) {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.byte()? != 0x3D {
                return Ok(Some((name, String::new())));
            }
            self.pos += 1; // past '='
        }
        // Value step: skip whitespace.
        loop {
            let b = self.byte()?;
            if is_ws(b) {
                self.pos += 1;
            } else {
                break;
            }
        }
        let b = self.byte()?;
        if b == 0x22 || b == 0x27 {
            // Quoted value.
            let q = b;
            loop {
                self.pos += 1;
                let c = self.byte()?;
                if c == q {
                    self.pos += 1;
                    return Ok(Some((name, value)));
                } else if c.is_ascii_uppercase() {
                    value.push((c + 0x20) as char);
                } else {
                    value.push(c as char);
                }
            }
        }
        if b == 0x3E {
            return Ok(Some((name, String::new())));
        }
        // Unquoted value.
        loop {
            let c = self.byte()?;
            if is_ws(c) || c == 0x3E {
                return Ok(Some((name, value)));
            } else if c.is_ascii_uppercase() {
                value.push((c + 0x20) as char);
            } else {
                value.push(c as char);
            }
            self.pos += 1;
        }
    }
}
