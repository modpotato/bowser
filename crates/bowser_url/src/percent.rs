//! WHATWG percent encoding and decoding.
//!
//! See <https://url.spec.whatwg.org/#percent-encoded-bytes>.

/// A WHATWG percent-encode set.
///
/// Each set extends the previous one; see
/// <https://url.spec.whatwg.org/#percent-encoded-bytes>.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncodeSet {
    /// C0 controls and all code points greater than `~`.
    C0,
    /// [`EncodeSet::C0`] plus space, `"`, `<`, `>`, `` ` ``.
    Fragment,
    /// [`EncodeSet::C0`] plus space, `"`, `#`, `<`, `>`.
    Query,
    /// [`EncodeSet::Query`] plus `'`. Used for the query of special URLs.
    SpecialQuery,
    /// [`EncodeSet::Query`] plus `?`, `^`, `` ` ``, `{`, `}`.
    Path,
    /// [`EncodeSet::Path`] plus `/`, `:`, `;`, `=`, `@`, `[`, `\`, `]`, `|`.
    Userinfo,
}

/// Whether an ASCII byte is in `set`.
///
/// Bytes `>= 0x7f` are in every encode set, so callers test those
/// separately and only pass bytes `< 0x7f` here.
fn ascii_in_set(b: u8, set: EncodeSet) -> bool {
    debug_assert!(b < 0x7f);
    // C0 set: C0 controls (everything above '~' is handled by the caller).
    if b <= 0x1f {
        return true;
    }
    match set {
        EncodeSet::C0 => false,
        EncodeSet::Fragment => matches!(b, b' ' | b'"' | b'<' | b'>' | b'`'),
        EncodeSet::Query => matches!(b, b' ' | b'"' | b'#' | b'<' | b'>'),
        EncodeSet::SpecialQuery => matches!(b, b' ' | b'"' | b'#' | b'<' | b'>' | b'\''),
        EncodeSet::Path => matches!(
            b,
            b' ' | b'"' | b'#' | b'<' | b'>' | b'?' | b'^' | b'`' | b'{' | b'}'
        ),
        EncodeSet::Userinfo => matches!(
            b,
            b' ' | b'"'
                | b'#'
                | b'<'
                | b'>'
                | b'?'
                | b'^'
                | b'`'
                | b'{'
                | b'}'
                | b'/'
                | b':'
                | b';'
                | b'='
                | b'@'
                | b'['
                | b'\\'
                | b']'
                | b'|'
        ),
    }
}

/// Percent-encode `input` using `set`, appending the result to `out`.
///
/// This is the workhorse behind [`percent_encode`]. It scans for the first
/// byte that needs encoding and copies the clean prefix verbatim, so inputs
/// needing no encoding cost one byte-scan plus a copy instead of two
/// char-decoding passes, and callers building a larger string avoid an
/// intermediate allocation.
pub(crate) fn percent_encode_into(out: &mut String, input: &str, set: EncodeSet) {
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b >= 0x7f || ascii_in_set(b, set) {
            break;
        }
        i += 1;
    }
    out.push_str(&input[..i]);
    while i < bytes.len() {
        let b = bytes[i];
        if b >= 0x7f {
            // DEL and non-ASCII code points are in every encode set.
            if b == 0x7f {
                push_hex(out, b);
                i += 1;
            } else {
                // The input is valid UTF-8 and `i` is on a char boundary
                // (the scan above only stops at a lead byte), so `b` starts
                // a multibyte sequence: encode its bytes directly without
                // decoding the char.
                debug_assert!(b >= 0xc0);
                let len = utf8_len(b);
                for &b in &bytes[i..i + len] {
                    push_hex(out, b);
                }
                i += len;
            }
        } else if ascii_in_set(b, set) {
            push_hex(out, b);
            i += 1;
        } else {
            out.push(b as char);
            i += 1;
        }
    }
}

fn push_hex(out: &mut String, b: u8) {
    out.push('%');
    out.push(hex_digit(b >> 4));
    out.push(hex_digit(b & 0xf));
}

/// Percent-encode raw `bytes` using `set`, appending the result to `out`.
///
/// The byte-oriented twin of [`percent_encode_into`], for the URL
/// Standard's "percent-encode after encoding": the input was already
/// converted from text with a legacy encoding (e.g. Big5), so every byte
/// `>= 0x7f` is percent-encoded opaquely instead of being decoded as
/// UTF-8. Only used on the encoding-override path; the UTF-8 default
/// keeps the `&str` fast path above.
pub(crate) fn percent_encode_bytes_into(out: &mut String, bytes: &[u8], set: EncodeSet) {
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b >= 0x7f || ascii_in_set(b, set) {
            break;
        }
        i += 1;
    }
    for &b in &bytes[..i] {
        out.push(b as char);
    }
    for &b in &bytes[i..] {
        if b >= 0x7f || ascii_in_set(b, set) {
            push_hex(out, b);
        } else {
            out.push(b as char);
        }
    }
}

/// Length of the UTF-8 sequence starting with lead byte `b` (`>= 0x80`).
fn utf8_len(b: u8) -> usize {
    if b < 0xe0 {
        2
    } else if b < 0xf0 {
        3
    } else {
        4
    }
}

/// Percent-encode `input` using `set`.
///
/// Code points in the set are encoded as their UTF-8 bytes, each byte as
/// `%` followed by two uppercase hexadecimal digits. Code points outside
/// the set are appended unchanged.
pub fn percent_encode(input: &str, set: EncodeSet) -> String {
    let mut out = String::with_capacity(input.len());
    percent_encode_into(&mut out, input, set);
    out
}

pub(crate) fn hex_digit(n: u8) -> char {
    (b'0' + n + if n >= 10 { b'A' - b'0' - 10 } else { 0 }) as char
}

/// Percent-decode `input` into bytes.
///
/// A `%` followed by two ASCII hexadecimal digits decodes to that byte.
/// Any other code point contributes its UTF-8 encoding. A `%` not followed
/// by two hexadecimal digits is kept literally.
pub fn percent_decode(input: &str) -> Vec<u8> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && let (Some(h), Some(l)) = (
                hex_value(bytes.get(i + 1).copied()),
                hex_value(bytes.get(i + 2).copied()),
            )
        {
            out.push(h << 4 | l);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

fn hex_value(b: Option<u8>) -> Option<u8> {
    let b = b?;
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Decode percent-encoded bytes as UTF-8 without a BOM, replacing invalid
/// sequences with U+FFFD, as the URL standard's UTF-8 decode does.
pub fn utf8_decode_lossy(bytes: &[u8]) -> String {
    // Strip a single UTF-8 BOM if present.
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
    String::from_utf8_lossy(bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_sets() {
        // Space is not in the C0 control set, so it is left alone.
        assert_eq!(percent_encode("a b", EncodeSet::C0), "a b");
        assert_eq!(percent_encode("a b", EncodeSet::Query), "a%20b");
        assert_eq!(percent_encode("a\"b", EncodeSet::Fragment), "a%22b");
        assert_eq!(percent_encode("a#b", EncodeSet::Query), "a%23b");
        assert_eq!(percent_encode("a'b", EncodeSet::Query), "a'b");
        assert_eq!(percent_encode("a'b", EncodeSet::SpecialQuery), "a%27b");
        assert_eq!(percent_encode("a^b", EncodeSet::Path), "a%5Eb");
        assert_eq!(percent_encode("a^b", EncodeSet::Query), "a^b");
        assert_eq!(percent_encode("a|b", EncodeSet::Userinfo), "a%7Cb");
        assert_eq!(percent_encode("a|b", EncodeSet::Path), "a|b");
        assert_eq!(percent_encode("a`b", EncodeSet::Query), "a`b");
        assert_eq!(percent_encode("a`b", EncodeSet::Fragment), "a%60b");
    }

    #[test]
    fn encode_non_ascii() {
        assert_eq!(percent_encode("é", EncodeSet::C0), "%C3%A9");
        assert_eq!(percent_encode("é", EncodeSet::Fragment), "%C3%A9");
        assert_eq!(percent_encode("~", EncodeSet::C0), "~");
        assert_eq!(percent_encode("\u{7f}", EncodeSet::C0), "%7F");
    }

    #[test]
    fn decode_round_trip() {
        assert_eq!(percent_decode("a%20b"), b"a b");
        assert_eq!(percent_decode("a%2Fb"), b"a/b");
        assert_eq!(percent_decode("%"), b"%");
        assert_eq!(percent_decode("%2"), b"%2");
        assert_eq!(percent_decode("%zz"), b"%zz");
        assert_eq!(percent_decode("100%"), b"100%");
        assert_eq!(percent_decode("%C3%A9"), &[0xc3, 0xa9]);
    }

    #[test]
    fn passthrough_when_clean() {
        // Fast path must return an equal string.
        assert_eq!(
            percent_encode("abcXYZ019-._~", EncodeSet::Userinfo),
            "abcXYZ019-._~"
        );
    }
}
