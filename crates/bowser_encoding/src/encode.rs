//! Encoder implementations per the WHATWG Encoding Standard.
//!
//! Only encodings with a spec-defined encoder are supported here: UTF-8,
//! the legacy single-byte encodings, and Big5. The remaining encodings
//! (GB18030, Shift_JIS, EUC-JP, EUC-KR, UTF-16BE/LE, ISO-2022-JP,
//! replacement, x-user-defined) have no encoder in this crate yet;
//! [`encode`] returns `None` for them and the caller falls back to UTF-8
//! (the URL Standard's behavior when no encoding override applies).

use std::sync::OnceLock;

use super::Encoding;
use super::data;

/// Encode `input` into `encoding`'s bytes (WHATWG Encoding Standard,
/// "encode").
///
/// Returns `None` for encodings without an encoder in this crate; the
/// caller falls back to UTF-8.
///
/// Code points the encoding cannot represent are emitted as their
/// *percent-encoded* numeric character reference —
/// `%26%23<decimal>%3B` — exactly as the URL Standard's
/// "percent-encode after encoding" specifies (step 7.4 appends those
/// literal strings on encode failure; WPT `encoding/big5-encoder.html`:
/// `"\u2603"` encodes as `%26%239731%3B`). `%` is not in the
/// query/special-query percent-encode sets, so these bytes survive the
/// caller's percent-encoding unchanged.
pub fn encode(input: &str, encoding: Encoding) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len());
    if encode_into(input, encoding, &mut out) {
        Some(out)
    } else {
        None
    }
}

/// [`encode`] appending into `out` instead of allocating: returns `false`
/// (leaving `out` untouched) for encodings without an encoder, so the
/// caller can fall back to UTF-8 without a wasted allocation.
pub fn encode_into(input: &str, encoding: Encoding, out: &mut Vec<u8>) -> bool {
    match encoding {
        Encoding::Utf8 => {
            out.extend_from_slice(input.as_bytes());
            true
        }
        _ => {
            if let Some(table) = encoding.single_byte_table() {
                encode_single_byte(table, input, out);
                true
            } else if encoding == Encoding::Big5 {
                encode_big5(input, out);
                true
            } else {
                false
            }
        }
    }
}

/// Append the percent-encoded numeric character reference
/// `%26%23<cp>%3B` for a code point the encoding cannot represent (see
/// [`encode`]).
fn push_ncr(out: &mut Vec<u8>, cp: u32) {
    out.extend_from_slice(b"%26%23");
    let mut digits = [0u8; 10];
    let mut n = 0;
    let mut v = cp;
    // `cp` is never 0 here in practice, but handle it anyway.
    loop {
        digits[n] = b'0' + (v % 10) as u8;
        n += 1;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    for d in digits[..n].iter().rev() {
        out.push(*d);
    }
    out.extend_from_slice(b"%3B");
}

/// Legacy single-byte encoder: ASCII passes through; other code points
/// take the first table row holding them (the spec's "index pointer");
/// unmapped code points fall back to a numeric character reference.
fn encode_single_byte(table: &[u16; 128], input: &str, out: &mut Vec<u8>) {
    for ch in input.chars() {
        let cp = ch as u32;
        if cp < 0x80 {
            out.push(cp as u8);
            continue;
        }
        let byte = table
            .iter()
            .position(|&entry| u32::from(entry) == cp)
            .map(|i| 0x80 + i as u8);
        match byte {
            Some(b) => out.push(b),
            None => push_ncr(out, cp),
        }
    }
}

/// Big5 pointer for `cp`: the *highest* pointer whose index entry is
/// `cp` (duplicate-mapped code points prefer the highest pointer — WPT
/// `encoding/big5-encoder.html` maps `"\u2550"` to `%F9%F9`).
/// Keyed on the full code point, so astral duplicates whose low 16 bits
/// collide with a BMP entry resolve independently.
fn big5_pointer(cp: u32) -> Option<u32> {
    static REVERSE: OnceLock<Vec<(u32, u32)>> = OnceLock::new();
    let rev = REVERSE.get_or_init(|| {
        let mut v: Vec<(u32, u32)> = Vec::with_capacity(data::BIG5.len());
        for (pointer, &entry) in data::BIG5.iter().enumerate() {
            if entry != 0xFFFF_FFFF {
                v.push((entry, pointer as u32));
            }
        }
        // Stable sort: duplicates stay in ascending-pointer order, so
        // the sweep below keeps the highest pointer per code point.
        v.sort_by_key(|&(cp, _)| cp);
        let mut dedup: Vec<(u32, u32)> = Vec::with_capacity(v.len());
        for (cp, pointer) in v {
            if let Some(last) = dedup.last_mut()
                && last.0 == cp
            {
                last.1 = pointer;
                continue;
            }
            dedup.push((cp, pointer));
        }
        dedup
    });
    rev.binary_search_by_key(&cp, |&(cp, _)| cp)
        .ok()
        .map(|i| rev[i].1)
}

/// Big5 encoder (WHATWG Encoding Standard § big5 encoder): ASCII passes
/// through; otherwise the pointer's lead/trail bytes, excluding the
/// `lead < 0xA1` range the spec's encoder refuses (WPT: `"\u9EA6"` ->
/// `&#40614;`); unmapped code points fall back to a numeric character
/// reference.
fn encode_big5(input: &str, out: &mut Vec<u8>) {
    for ch in input.chars() {
        let cp = ch as u32;
        if cp < 0x80 {
            out.push(cp as u8);
            continue;
        }
        match big5_pointer(cp) {
            Some(pointer) => {
                let lead = pointer / 157 + 0x81;
                if lead < 0xA1 {
                    push_ncr(out, cp);
                    continue;
                }
                let trail = pointer % 157;
                let offset = if trail < 0x3F { 0x40 } else { 0x62 };
                out.push(lead as u8);
                out.push((trail + offset) as u8);
            }
            None => push_ncr(out, cp),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render encoded bytes the way `bowser_url`'s query percent-encoding
    /// would: `%` is not in the query set, so the pre-encoded NCRs pass
    /// through; other non-ASCII bytes become `%XX`; `@` (0x40) survives.
    fn q(bytes: &[u8]) -> String {
        let mut s = String::new();
        let mut i = 0;
        while i < bytes.len() {
            // Pass pre-encoded `%XX` triplets through untouched.
            if bytes[i] == b'%'
                && i + 2 < bytes.len()
                && bytes[i + 1].is_ascii_hexdigit()
                && bytes[i + 2].is_ascii_hexdigit()
            {
                s.push_str(std::str::from_utf8(&bytes[i..i + 3]).unwrap());
                i += 3;
                continue;
            }
            let b = bytes[i];
            if b.is_ascii_alphanumeric() || b == b'@' {
                s.push(b as char);
            } else {
                s.push_str(&format!("%{b:02X}"));
            }
            i += 1;
        }
        s
    }

    #[test]
    fn utf8_is_identity() {
        assert_eq!(encode("aéb", Encoding::Utf8).unwrap(), "aéb".as_bytes());
    }

    #[test]
    fn single_byte_windows1252() {
        // U+00E9 is 0xE9 in windows-1252; U+20AC is 0x80.
        assert_eq!(
            encode("é€", Encoding::Windows1252).unwrap(),
            vec![0xE9, 0x80]
        );
        // U+2603 is unmapped -> percent-encoded NCR.
        assert_eq!(
            encode("\u{2603}", Encoding::Windows1252).unwrap(),
            b"%26%239731%3B".to_vec()
        );
        // ASCII passes through untouched.
        assert_eq!(
            encode("az09", Encoding::Windows1252).unwrap(),
            b"az09".to_vec()
        );
    }

    #[test]
    fn big5_wpt_vectors() {
        // Vectors from WPT encoding/big5-encoder.html, rendered through
        // the query percent-encode set ('@' = 0x40 is query-allowed).
        let cases: &[(&str, &str)] = &[
            ("ab", "ab"),
            ("\u{3000}", "%A1@"),
            ("\u{20AC}", "%A3%E1"),
            ("\u{4E00}", "%A4@"),
            ("\u{79D4}", "%FE%FE"),
            ("\u{2550}", "%F9%F9"), // duplicate prefers highest pointer
            ("\u{9EA6}", "%26%2340614%3B"), // lead < 0xA1 excluded -> NCR
            ("\u{2603}", "%26%239731%3B"), // unmapped -> NCR
            ("\u{2626B}", "%26%23156267%3B"), // astral excluded -> NCR
            ("\u{203B5}", "%FDj"),  // astral duplicate, full-cp key
            ("\u{27607}", "%C8%A4"),
            ("\u{FFE2}", "%C8%CD"),
        ];
        for (input, expected) in cases {
            let bytes = encode(input, Encoding::Big5).unwrap();
            assert_eq!(q(&bytes), *expected, "input {input:?}");
        }
    }

    #[test]
    fn big5_raw_bytes() {
        // Byte-level spot checks (no percent-encoding involved).
        assert_eq!(
            encode("\u{3000}", Encoding::Big5).unwrap(),
            vec![0xA1, 0x40]
        );
        assert_eq!(encode("a", Encoding::Big5).unwrap(), vec![0x61]);
    }

    #[test]
    fn unsupported_encoding_returns_none() {
        assert!(encode("abc", Encoding::Gbk).is_none());
        assert!(encode("abc", Encoding::ShiftJis).is_none());
        assert!(encode("abc", Encoding::Utf16Le).is_none());
    }
}
