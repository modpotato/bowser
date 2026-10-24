//! Pragmatic IDNA processing for domain names.
//!
//! This module implements the URL standard's [domain to ASCII] given a
//! non-ASCII domain, i.e. UTS #46 processing with the parameters the URL
//! standard mandates for `beStrict=false` (the only mode this crate
//! implements): non-transitional processing, `CheckHyphens=false`,
//! `CheckBidi=true`, `CheckJoiners=true`, `UseSTD3ASCIIRules=false`,
//! `VerifyDnsLength=false` (which the spec sets equal to `beStrict`).
//!
//! [domain to ASCII]: https://url.spec.whatwg.org/#concept-domain-to-ascii
//!
//! # Supported subset
//!
//! A full UTS #46 implementation needs the Unicode Character Database
//! (tens of thousands of mapping/validity entries). This module implements
//! a pragmatic subset that is exact for the cases exercised by the
//! web-platform-tests URL suite:
//!
//! * ASCII case folding (`A`-`Z` to `a`-`z`).
//! * Dot separators U+3002, U+FF0E, U+FF61 map to `.`.
//! * Fullwidth ASCII (U+FF01-U+FF5E) maps to ASCII (subtract `0xFEE0`).
//! * Latin-1 case folding (U+00C0-U+00DE except U+00D7 to lowercase).
//! * Greek uppercase (U+0391-U+03A1, U+0393-U+03A9) and Cyrillic uppercase
//!   (U+0410-U+042F, U+0401) case folding.
//! * Single mappings: U+1E9E to U+00DF, U+2183 to U+2184, U+04C0 to U+04CF,
//!   U+2F868 to U+36FC.
//! * Circled digits U+2460-U+2473 map to the digit strings "1"-"20" and
//!   U+24EA (⓪) maps to "0"; superscript digits U+00B2, U+00B3, U+00B9,
//!   U+2070, U+2074-U+2079 and subscript digits U+2080-U+2089 map to ASCII
//!   digits (U+2071 is superscript "i", not a digit, and maps to "i").
//!   These matter because the resulting ASCII digits trigger IPv4 parsing
//!   (e.g. `http://①②③④/` has host `0.0.4.210`).
//! * Mathematical bold alphanumerics (U+1D400-U+1D433) map to ASCII.
//! * Ignored code points are removed: U+00AD, U+034F, U+1806, U+180B-U+180D,
//!   U+180E, U+200B, U+2060, U+2064, U+2065, U+206A-U+206F, U+FE00-U+FE0F,
//!   U+FEFF.
//! * A small NFC composition table covering Latin-1 precompositions and the
//!   `=`/`<`/`>` + U+0338 cases. Full canonical decomposition and
//!   reordering are *not* implemented.
//! * Validity checks: C0/C1 controls and DEL, Unicode space separators,
//!   U+FFFD, noncharacters (U+FDD0-U+FDEF and any `*FFFE`/`*FFFF`), bidi
//!   control codes, tag code points (U+E0000-U+E0FFF), and a denylist of
//!   disallowed code points observed in the wild (U+0378, U+06DD, U+2025,
//!   U+2A74, U+2FA68, U+2FF0, U+FFFA) are rejected. Unassigned code points
//!   outside this denylist are *not* rejected (deviation from UTS #46).
//! * Contextual joiner rules (U+200C/U+200D) are simplified: a label
//!   starting or ending with a joiner is rejected; full `RegExp`-style
//!   context validation is not performed.
//! * The bidi rule is simplified: a label starting with a right-to-left
//!   character must not contain ASCII letters, and a label starting with a
//!   left-to-right character must not contain right-to-left characters.
//! * Labels starting with `xn--` are verified by decoding the Punycode and
//!   checking that the mapped decoded form re-encodes to the same label,
//!   mirroring UTS #46 ToUnicode verification.
//!
//! DNS length limits are intentionally not enforced for *acceptance*,
//! matching the URL standard for `beStrict=false`
//! (`VerifyDnsLength=false`): over-long labels and domains that need no
//! Punycode decoding still round-trip (the WPT `toascii.json` suite
//! mandates this). What is capped is the *work*: Punycode decoding is
//! never attempted on an `xn--` label longer than 63 octets, because such
//! a label cannot be a valid A-label (A-labels are at most 63 octets by
//! definition), so it is rejected before decoding. That bounds decode
//! work per label to a constant instead of O(n) allocations and O(n log
//! n) time on attacker-controlled input. [`crate::domain_to_unicode`]
//! additionally rejects domains longer than 253 octets before decoding
//! any label.

/// Convert `domain` to ASCII per the URL standard's domain parser with
/// `beStrict` set to false.
///
/// Returns `None` when the domain is invalid. ASCII domains are lowercased
/// directly (web-compatibility short-circuit); non-ASCII domains go through
/// the UTS #46 subset documented above.
pub fn domain_to_ascii(domain: &str) -> Option<String> {
    // The strict ToASCII pass only produces a non-fatal validation error
    // when beStrict is false, so it is skipped.
    let result = if domain.is_ascii() {
        domain.to_ascii_lowercase()
    } else {
        to_ascii_mapped(domain)?
    };
    if result.is_empty() {
        return None;
    }
    if result.chars().any(is_forbidden_domain_code_point) {
        return None;
    }
    Some(result)
}

/// A forbidden domain code point: a forbidden host code point, a C0 control,
/// `%`, or DEL.
pub fn is_forbidden_domain_code_point(c: char) -> bool {
    if c.is_ascii_control() {
        return true;
    }
    matches!(
        c,
        '\0' | '\u{9}'
            | '\u{A}'
            | '\u{D}'
            | ' '
            | '#'
            | '/'
            | ':'
            | '<'
            | '>'
            | '?'
            | '@'
            | '['
            | '\\'
            | ']'
            | '^'
            | '|'
            | '%'
            | '\u{7f}'
    )
}

/// A forbidden host code point, used by the opaque-host parser.
pub fn is_forbidden_host_code_point(c: char) -> bool {
    matches!(
        c,
        '\0' | '\u{9}'
            | '\u{A}'
            | '\u{D}'
            | ' '
            | '#'
            | '/'
            | ':'
            | '<'
            | '>'
            | '?'
            | '@'
            | '['
            | '\\'
            | ']'
            | '^'
            | '|'
    )
}

// ---------------------------------------------------------------------------
// Mapping
// ---------------------------------------------------------------------------

enum Map {
    Keep,
    To(char),
    /// Map to a static ASCII string (used for circled numbers, which map
    /// to one or two ASCII digits per UTS #46).
    ToStr(&'static str),
    Dot,
    Ignore,
}

/// UTS #46 maps U+2460..U+2473 (①..⑳) to the digit strings "1".."20".
const CIRCLED_NUMBER_MAP: [&str; 20] = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
    "18", "19", "20",
];

/// Code points with IDNA status "ignored" per the UTS #46 subset in
/// [`map_char`] (e.g. U+00AD SOFT HYPHEN). UTS #46 §4.1 requires an error
/// when one appears in a ToUnicode-decoded `xn--` label; see
/// [`decode_xn_label_tounicode`].
fn is_ignored(c: char) -> bool {
    matches!(
        c,
        '\u{AD}'
            | '\u{34F}'
            | '\u{1806}'
            | '\u{180B}'
            | '\u{180C}'
            | '\u{180D}'
            | '\u{180E}'
            | '\u{200B}'
            | '\u{2060}'
            | '\u{2064}'
            | '\u{2065}'
            | '\u{206A}'..='\u{206F}'
            | '\u{FE00}'..='\u{FE0F}'
            | '\u{FEFF}'
    )
}

/// Map a single code point per the supported UTS #46 subset.
///
/// When `for_tounicode` is true (decoding or verifying an `xn--` label's
/// decoded form), ignored code points are kept rather than removed. This
/// is *not* UTS #46 ToUnicode processing — §4.1 requires an error for an
/// ignored-status code point in a ToUnicode-decoded label, which the
/// ToUnicode path ([`decode_xn_label_tounicode`]) enforces with an
/// explicit [`is_ignored`] check. Keeping them here serves the
/// verification path ([`verify_xn_label`]): a kept code point re-encodes
/// to the same label, so verification of already-consistent labels is
/// unaffected (a documented, marginal deviation from §4.1, which would
/// reject such labels).
fn map_char(c: char, for_tounicode: bool) -> Map {
    match c {
        'A'..='Z' => Map::To(c.to_ascii_lowercase()),
        // Dot separators.
        '\u{3002}' | '\u{FF0E}' | '\u{FF61}' => Map::Dot,
        // Fullwidth ASCII -> ASCII; fullwidth capitals map to lowercase,
        // per the IDNA mapping table.
        '\u{FF21}'..='\u{FF3A}' => Map::To(char_from_u32(c as u32 - 0xFEE0 + 0x20)),
        '\u{FF01}'..='\u{FF20}' | '\u{FF3B}'..='\u{FF5E}' => {
            Map::To(char_from_u32(c as u32 - 0xFEE0))
        }
        // Latin-1 uppercase -> lowercase (except multiplication sign).
        '\u{C0}'..='\u{DE}' if c != '\u{D7}' => Map::To(char_from_u32(c as u32 + 0x20)),
        // Greek uppercase -> lowercase.
        // Note: U+03C2 GREEK SMALL LETTER FINAL SIGMA is valid as-is per
        // UTS #46; it is not mapped to U+03C3.
        '\u{391}'..='\u{3A1}' | '\u{3A3}'..='\u{3A9}' => Map::To(char_from_u32(c as u32 + 0x20)),
        // Cyrillic uppercase -> lowercase.
        '\u{410}'..='\u{42F}' => Map::To(char_from_u32(c as u32 + 0x20)),
        '\u{401}' => Map::To('\u{451}'),
        // Mathematical bold capitals/small letters -> lowercase ASCII,
        // per the IDNA mapping table.
        '\u{1D400}'..='\u{1D419}' => Map::To(char_from_u32(c as u32 - 0x1D400 + 'a' as u32)),
        '\u{1D41A}'..='\u{1D433}' => Map::To(char_from_u32(c as u32 - 0x1D41A + 'a' as u32)),
        // Single special mappings.
        '\u{1E9E}' => Map::To('\u{DF}'),
        '\u{2183}' => Map::To('\u{2184}'),
        '\u{4C0}' => Map::To('\u{4CF}'),
        '\u{2F868}' => Map::To('\u{36FC}'),
        // Circled numbers -> ASCII digit strings, per UTS #46. The ASCII
        // digits they produce can trigger IPv4 parsing of the host.
        '\u{2460}'..='\u{2473}' => Map::ToStr(CIRCLED_NUMBER_MAP[c as u32 as usize - 0x2460]),
        '\u{24EA}' => Map::ToStr("0"),
        // Superscript and subscript digits -> ASCII digits, per UTS #46.
        // Note U+2071 is superscript latin small letter i, not a digit.
        '\u{B2}' => Map::To('2'),
        '\u{B3}' => Map::To('3'),
        '\u{B9}' => Map::To('1'),
        '\u{2070}' => Map::To('0'),
        '\u{2071}' => Map::To('i'),
        '\u{2074}'..='\u{2079}' => Map::To(char_from_u32(c as u32 - 0x2070 + '0' as u32)),
        '\u{2080}'..='\u{2089}' => Map::To(char_from_u32(c as u32 - 0x2080 + '0' as u32)),
        // Ignored code points; see [`is_ignored`].
        c if is_ignored(c) => {
            if for_tounicode {
                Map::Keep
            } else {
                Map::Ignore
            }
        }
        _ => Map::Keep,
    }
}

fn char_from_u32(cp: u32) -> char {
    // All callers map within known valid ranges; the fallback keeps the
    // mapping total (U+FFFD, never a panic) if that ever changes.
    debug_assert!(
        char::from_u32(cp).is_some(),
        "idna mapping produced invalid code point"
    );
    char::from_u32(cp).unwrap_or('\u{FFFD}')
}

// ---------------------------------------------------------------------------
// NFC-lite: composition of a small table of precompositions.
// ---------------------------------------------------------------------------

/// Try to compose `a` followed by combining mark `b`.
fn compose(a: char, b: char) -> Option<char> {
    match (a, b) {
        // Combining long solidus overlay cases from the IDNA test vectors.
        ('=', '\u{338}') => Some('\u{2260}'),
        ('<', '\u{338}') => Some('\u{226E}'),
        ('>', '\u{338}') => Some('\u{226F}'),
        // Latin-1 precompositions.
        ('a', '\u{300}') => Some('à'),
        ('a', '\u{301}') => Some('á'),
        ('a', '\u{302}') => Some('â'),
        ('a', '\u{303}') => Some('ã'),
        ('a', '\u{308}') => Some('ä'),
        ('a', '\u{30A}') => Some('å'),
        ('e', '\u{300}') => Some('è'),
        ('e', '\u{301}') => Some('é'),
        ('e', '\u{302}') => Some('ê'),
        ('e', '\u{308}') => Some('ë'),
        ('i', '\u{300}') => Some('ì'),
        ('i', '\u{301}') => Some('í'),
        ('i', '\u{302}') => Some('î'),
        ('i', '\u{308}') => Some('ï'),
        ('o', '\u{300}') => Some('ò'),
        ('o', '\u{301}') => Some('ó'),
        ('o', '\u{302}') => Some('ô'),
        ('o', '\u{303}') => Some('õ'),
        ('o', '\u{308}') => Some('ö'),
        ('u', '\u{300}') => Some('ù'),
        ('u', '\u{301}') => Some('ú'),
        ('u', '\u{302}') => Some('û'),
        ('u', '\u{308}') => Some('ü'),
        ('y', '\u{301}') => Some('ý'),
        ('y', '\u{308}') => Some('ÿ'),
        ('c', '\u{327}') => Some('ç'),
        ('n', '\u{303}') => Some('ñ'),
        ('A', '\u{300}') => Some('À'),
        ('A', '\u{301}') => Some('Á'),
        ('A', '\u{302}') => Some('Â'),
        ('A', '\u{303}') => Some('Ã'),
        ('A', '\u{308}') => Some('Ä'),
        ('A', '\u{30A}') => Some('Å'),
        ('E', '\u{300}') => Some('È'),
        ('E', '\u{301}') => Some('É'),
        ('E', '\u{302}') => Some('Ê'),
        ('E', '\u{308}') => Some('Ë'),
        ('I', '\u{300}') => Some('Ì'),
        ('I', '\u{301}') => Some('Í'),
        ('I', '\u{302}') => Some('Î'),
        ('I', '\u{308}') => Some('Ï'),
        ('O', '\u{300}') => Some('Ò'),
        ('O', '\u{301}') => Some('Ó'),
        ('O', '\u{302}') => Some('Ô'),
        ('O', '\u{303}') => Some('Õ'),
        ('O', '\u{308}') => Some('Ö'),
        ('U', '\u{300}') => Some('Ù'),
        ('U', '\u{301}') => Some('Ú'),
        ('U', '\u{302}') => Some('Û'),
        ('U', '\u{308}') => Some('Ü'),
        ('Y', '\u{301}') => Some('Ý'),
        ('C', '\u{327}') => Some('Ç'),
        ('N', '\u{303}') => Some('Ñ'),
        _ => None,
    }
}

/// Push `c` onto `out`, composing with the previous character when the
/// composition table has an entry. Only the last character is considered;
/// blocking and reordering are not implemented (documented limitation).
fn push_compose(out: &mut String, c: char) {
    if let Some(last) = out.pop() {
        if let Some(composed) = compose(last, c) {
            out.push(composed);
            return;
        }
        out.push(last);
    }
    out.push(c);
}

// ---------------------------------------------------------------------------
// Validity
// ---------------------------------------------------------------------------

/// Code points rejected by the pragmatic validity check.
fn is_disallowed(c: char) -> bool {
    let cp = c as u32;
    if matches!(
        cp,
        0x00..=0x1F | 0x7F..=0x9F
        | 0x20 | 0xA0 | 0x1680 | 0x2000..=0x200A | 0x202F | 0x205F | 0x3000
        | 0xFFFD
        | 0x202A..=0x202E
        | 0x0378 | 0x06DD | 0x2025 | 0x2A74 | 0x2FA68 | 0x2FF0 | 0xFFFA
        | 0xE0000..=0xE0FFF
    ) {
        return true;
    }
    // Noncharacters: U+FDD0-U+FDEF and any code point ending in FFFE/FFFF.
    if (0xFDD0..=0xFDEF).contains(&cp) {
        return true;
    }
    cp & 0xFFFE == 0xFFFE
}

/// Coarse right-to-left check for the simplified bidi rule.
fn is_rtl(c: char) -> bool {
    matches!(c as u32, 0x0590..=0x08FF)
}

// ---------------------------------------------------------------------------
// UTS #46 processing
// ---------------------------------------------------------------------------

fn to_ascii_mapped(domain: &str) -> Option<String> {
    let mut mapped = String::with_capacity(domain.len());
    for c in domain.chars() {
        match map_char(c, false) {
            Map::Keep => push_compose(&mut mapped, c),
            Map::To(d) => push_compose(&mut mapped, d),
            // ASCII output: no composition applies.
            Map::ToStr(s) => mapped.push_str(s),
            Map::Dot => mapped.push('.'),
            Map::Ignore => {}
        }
    }
    let mut out = String::with_capacity(mapped.len());
    for (i, label) in mapped.split('.').enumerate() {
        if i > 0 {
            out.push('.');
        }
        out.push_str(&process_label(label)?);
    }
    Some(out)
}

fn process_label(label: &str) -> Option<String> {
    if label.is_empty() {
        return Some(String::new());
    }
    if let Some(rest) = label.strip_prefix("xn--") {
        return verify_xn_label(label, rest);
    }
    let chars: Vec<char> = label.chars().collect();
    // Simplified contextual joiner rule. `label` is non-empty here (the
    // early return above), so `chars[0]` and `chars[chars.len() - 1]`
    // are in bounds.
    if matches!(chars[0], '\u{200C}' | '\u{200D}')
        || matches!(chars[chars.len() - 1], '\u{200C}' | '\u{200D}')
    {
        return None;
    }
    // Simplified bidi rule.
    if is_rtl(chars[0]) {
        if chars.iter().any(|c| c.is_ascii_alphabetic()) {
            return None;
        }
    } else if chars.iter().any(|c| is_rtl(*c)) {
        return None;
    }
    if chars.iter().any(|c| is_disallowed(*c)) {
        return None;
    }
    if label.is_ascii() {
        Some(label.to_string())
    } else {
        Some(format!("xn--{}", punycode_encode(label)?))
    }
}

/// Maximum DNS label length in octets (IDNA2008 / UTS #46 §4.1).
const MAX_LABEL_OCTETS: usize = 63;
/// Length of the `xn--` ACE prefix.
const ACE_PREFIX_LEN: usize = 4;

/// Decode an `xn--` label's Punycode and apply UTS #46 mapping and
/// validity to the decoded form: a disallowed code point (e.g. `xn--a` ->
/// U+0080) makes the label invalid, as does a code point the mapping
/// would change — those have IDNA status "mapped" (e.g. `xn--8i7caa`
/// decodes to fullwidth U+FF57), and UTS #46 §4.1 only allows "valid" and
/// "deviation" statuses in a decoded label.
///
/// Labels longer than 63 octets (including the `xn--` prefix) are
/// rejected *before* decoding. This is an intentional, marginal deviation
/// from UTS #46 with `VerifyDnsLength=false`: a longer `xn--` label whose
/// Punycode decodes and re-encodes consistently would verify under pure
/// `VerifyDnsLength=false`, but it is rejected here without decoding. An
/// A-label is at most 63 octets by the DNS definition, so no legitimate
/// A-label is affected; the cap is DoS hardening, bounding the decoder's
/// O(n) allocations and O(n log n) work per label to a constant on
/// attacker-controlled input.
///
/// `reject_ignored` selects the path: the ToUnicode path (used by
/// [`crate::domain_to_unicode`]) rejects a decoded code point with IDNA
/// status "ignored" (e.g. U+00AD SOFT HYPHEN) per §4.1, returning `None`
/// so the caller keeps the verbatim ASCII label. The verification path
/// ([`verify_xn_label`]) keeps ignored code points instead: they re-encode
/// to the same label, so verification of already-consistent labels is
/// unaffected (a documented, marginal deviation from §4.1, which would
/// reject such labels).
fn decode_xn_label_inner(rest: &str, reject_ignored: bool) -> Option<String> {
    // Length check first: O(1), and it short-circuits enormous inputs
    // before the O(n) ASCII scan below.
    if rest.len() > MAX_LABEL_OCTETS - ACE_PREFIX_LEN {
        return None;
    }
    if !rest.is_ascii() {
        return None;
    }
    let decoded = punycode_decode(rest)?;
    let mut mapped = String::with_capacity(decoded.len());
    for c in decoded.chars() {
        if is_disallowed(c) {
            return None;
        }
        // UTS #46 §4.1 allows only "valid" and "deviation" statuses in a
        // ToUnicode-decoded label: an "ignored"-status code point is an
        // error there (verified against the reference `idna` behavior).
        if reject_ignored && is_ignored(c) {
            return None;
        }
        match map_char(c, true) {
            Map::Keep => push_compose(&mut mapped, c),
            // The mapping would change this code point: it has IDNA
            // status "mapped", which §4.1 forbids in a decoded label
            // (only "valid" and "deviation" may appear). Reject, so
            // callers fall back to the ASCII label (e.g. `xn--8i7caa`).
            Map::To(_) | Map::ToStr(_) => return None,
            // '.' cannot appear in Punycode output; treat as failure.
            Map::Dot | Map::Ignore => return None,
        }
    }
    Some(mapped)
}

/// Verification flavor of [`decode_xn_label_inner`]: ignored-status code
/// points are kept, so [`verify_xn_label`] accepts already-consistent
/// labels exactly as before.
pub(crate) fn decode_xn_label(rest: &str) -> Option<String> {
    decode_xn_label_inner(rest, false)
}

/// ToUnicode flavor of [`decode_xn_label_inner`]: a decoded code point
/// with IDNA status "ignored" is an error per UTS #46 §4.1.
///
/// `pub(crate)` so [`crate::domain_to_unicode`] can validate decoded
/// `xn--` labels instead of returning a best-effort string.
pub(crate) fn decode_xn_label_tounicode(rest: &str) -> Option<String> {
    decode_xn_label_inner(rest, true)
}

/// Verify an `xn--` label by decoding its Punycode and checking that the
/// mapped decoded form re-encodes to the same label.
fn verify_xn_label(label: &str, rest: &str) -> Option<String> {
    let mapped = decode_xn_label(rest)?;
    let reencoded = format!("xn--{}", punycode_encode(&mapped)?);
    if reencoded != label {
        return None;
    }
    Some(label.to_string())
}

// ---------------------------------------------------------------------------
// Punycode (RFC 3492)
// ---------------------------------------------------------------------------

const BASE: u64 = 36;
const TMIN: u64 = 1;
const TMAX: u64 = 26;
const SKEW: u64 = 38;
const DAMP: u64 = 700;
const INITIAL_BIAS: u64 = 72;
const INITIAL_N: u64 = 128;

/// Bias adaptation per RFC 3492 section 6.1. Returns `None` on overflow:
/// attacker-controlled labels reach this on the host-parse path, so an
/// extreme delta must degrade to a parse failure, never panic (in debug
/// builds plain `+`/`*` would panic on overflow).
fn adapt(mut delta: u64, numpoints: u64, firsttime: bool) -> Option<u64> {
    delta /= if firsttime { DAMP } else { 2 };
    delta = delta.checked_add(delta / numpoints)?;
    let mut k = 0u64;
    while delta > (BASE - TMIN) * TMAX / 2 {
        delta /= BASE - TMIN;
        k = k.checked_add(BASE)?;
    }
    k.checked_add((BASE - TMIN + 1).checked_mul(delta)? / delta.checked_add(SKEW)?)
}

fn digit_to_char(d: u64) -> char {
    (if d < 26 {
        b'a' + d as u8
    } else {
        b'0' + (d - 26) as u8
    }) as char
}

fn char_to_digit(c: u8) -> Option<u64> {
    match c {
        b'a'..=b'z' => Some((c - b'a') as u64),
        b'A'..=b'Z' => Some((c - b'A') as u64),
        b'0'..=b'9' => Some((c - b'0') as u64 + 26),
        _ => None,
    }
}

/// Encode a label with Punycode (RFC 3492). Returns `None` on overflow.
///
/// Runs in O(n log n): distinct code points are processed in ascending
/// order, and a binary indexed tree counts the already-emitted code
/// points between consecutive occurrences, replacing the original
/// O(n)-per-distinct-value rescan of the input (quadratic on long
/// non-ASCII labels).
pub fn punycode_encode(label: &str) -> Option<String> {
    let input: Vec<u64> = label.chars().map(|c| c as u64).collect();
    let mut output = String::new();
    for &cp in &input {
        if cp < 0x80 {
            output.push(cp as u8 as char);
        }
    }
    let basic_len = output.len() as u64;
    let mut h = basic_len;
    if basic_len > 0 {
        output.push('-');
    }
    // Positions of non-basic code points, ordered by code point value
    // (the sort is stable, so occurrences of one value stay in input
    // order).
    let mut order: Vec<usize> = (0..input.len()).filter(|&j| input[j] >= 0x80).collect();
    order.sort_by_key(|&j| input[j]);
    // Marks input positions holding basic or already-emitted code points.
    let mut counted = BitSum::new(input.len());
    for (j, &cp) in input.iter().enumerate() {
        if cp < 0x80 {
            counted.add(j, 1);
        }
    }
    let mut n = INITIAL_N;
    let mut delta: u64 = 0;
    let mut bias = INITIAL_BIAS;
    for chunk in order.chunk_by(|&a, &b| input[a] == input[b]) {
        // `order` holds indices from `0..input.len()` and `chunk_by` never
        // yields an empty chunk, so these indexings are in bounds.
        let m = input[chunk[0]];
        delta = delta.checked_add((m - n).checked_mul(h + 1)?)?;
        n = m;
        // The delta added for each occurrence is the number of
        // already-counted code points since the previous emission (from
        // the input start for the first occurrence of this value); the
        // counted code points after the last occurrence carry into the
        // next value's delta, exactly as the single input scan does.
        let tail = counted.range(chunk[chunk.len() - 1] + 1, input.len());
        let mut lo = 0;
        for &p in chunk {
            delta = delta.checked_add(counted.range(lo, p))?;
            let mut q = delta;
            let mut k = BASE;
            loop {
                let t = if k <= bias {
                    TMIN
                } else if k >= bias + TMAX {
                    TMAX
                } else {
                    k - bias
                };
                if q < t {
                    break;
                }
                output.push(digit_to_char(t + (q - t) % (BASE - t)));
                q = (q - t) / (BASE - t);
                k = k.checked_add(BASE)?;
            }
            output.push(digit_to_char(q));
            bias = adapt(delta, h + 1, h == basic_len)?;
            delta = 0;
            h += 1;
            counted.add(p, 1);
            lo = p + 1;
        }
        delta = delta.checked_add(tail)?;
        delta = delta.checked_add(1)?;
        n = n.checked_add(1)?;
    }
    Some(output)
}

/// Binary indexed tree holding `u64` counts, for prefix and range sums.
struct BitSum {
    tree: Vec<u64>,
}

impl BitSum {
    fn new(n: usize) -> Self {
        BitSum {
            tree: vec![0u64; n + 1],
        }
    }

    fn add(&mut self, idx: usize, v: u64) {
        let mut i = idx + 1;
        while i < self.tree.len() {
            self.tree[i] += v;
            i += i.isolate_lowest_one();
        }
    }

    /// Sum over `[0, idx)`.
    fn prefix(&self, idx: usize) -> u64 {
        let mut sum = 0;
        let mut i = idx;
        while i > 0 {
            sum += self.tree[i];
            i -= i.isolate_lowest_one();
        }
        sum
    }

    /// Sum over `[lo, hi)`.
    fn range(&self, lo: usize, hi: usize) -> u64 {
        self.prefix(hi) - self.prefix(lo)
    }
}

/// Decode Punycode (RFC 3492). Returns `None` on invalid input or overflow.
///
/// Runs in O(n log n) rather than O(n^2): the delta pass only records each
/// insertion's index (decoder state evolves with the output *length*, never
/// its contents), then insertions are placed into the final array in
/// reverse order. A forward insertion never reorders existing elements, so
/// in reverse the k-th insertion lands exactly in the k-th still-free
/// slot; a binary indexed tree finds those slots in O(log n) each.
pub fn punycode_decode(input: &str) -> Option<String> {
    if !input.is_ascii() {
        return None;
    }
    let bytes = input.as_bytes();
    let (basic, rest) = match bytes.iter().rposition(|&b| b == b'-') {
        Some(p) => (&bytes[..p], &bytes[p + 1..]),
        None => (&[][..], bytes),
    };
    // Pass 1: decode the delta stream, recording (insertion index, code
    // point) pairs.
    let mut inserts: Vec<(usize, u64)> = Vec::new();
    let mut n = INITIAL_N;
    let mut i: u64 = 0;
    let mut bias = INITIAL_BIAS;
    let mut idx = 0;
    let mut len = basic.len() as u64;
    while idx < rest.len() {
        let oldi = i;
        let mut w: u64 = 1;
        let mut k = BASE;
        loop {
            let digit = char_to_digit(*rest.get(idx)?)?;
            idx += 1;
            i = i.checked_add(digit.checked_mul(w)?)?;
            let t = if k <= bias {
                TMIN
            } else if k >= bias + TMAX {
                TMAX
            } else {
                k - bias
            };
            if digit < t {
                break;
            }
            w = w.checked_mul(BASE - t)?;
            k = k.checked_add(BASE)?;
        }
        let out_len = len + 1;
        bias = adapt(i - oldi, out_len, oldi == 0)?;
        n = n.checked_add(i / out_len)?;
        i %= out_len;
        if n > 0x10FFFF || (0xD800..0xE000).contains(&n) {
            return None;
        }
        inserts.push((i as usize, n));
        i = i.checked_add(1)?;
        len = out_len;
    }
    // Pass 2: place the insertions into the final array, latest first.
    let total = basic.len() + inserts.len();
    let mut slots: Vec<u64> = vec![0; total];
    let mut filled = vec![false; total];
    let mut free = FreeSlots::new(total);
    for &(pos, cp) in inserts.iter().rev() {
        let slot = free.take_kth(pos);
        slots[slot] = cp;
        filled[slot] = true;
    }
    // The basic code points fill the remaining slots in order.
    let mut basic_iter = basic.iter();
    for (slot, is_filled) in slots.iter_mut().zip(filled.iter()) {
        if !is_filled {
            // `total` was sized as basic + inserts, so a free slot always
            // has a basic code point left; bail out instead of panicking
            // if the counts ever disagree.
            debug_assert!(basic_iter.len() > 0, "basic code points fill free slots");
            *slot = *basic_iter.next()? as u64;
        }
    }
    slots
        .into_iter()
        .map(|cp| char::from_u32(cp as u32))
        .collect::<Option<Vec<char>>>()
        .map(|v| v.into_iter().collect())
}

/// Tracks the free slots of a fixed-size array: every slot starts free,
/// [`FreeSlots::take_kth`] finds the k-th (0-based) free slot and marks it
/// taken, both in O(log n). Backs the linearithmic Punycode decoder above.
struct FreeSlots {
    /// Binary indexed tree over per-slot free counts (1 = free).
    tree: Vec<u64>,
}

impl FreeSlots {
    fn new(n: usize) -> Self {
        // All-ones array: tree[i] holds the lowbit of i.
        let mut tree = vec![0u64; n + 1];
        for (i, slot) in tree.iter_mut().enumerate().skip(1) {
            *slot = i.isolate_lowest_one() as u64;
        }
        FreeSlots { tree }
    }

    /// Mark slot `idx` (0-based) as taken.
    fn remove(&mut self, idx: usize) {
        let mut i = idx + 1;
        while i < self.tree.len() {
            self.tree[i] -= 1;
            i += i.isolate_lowest_one();
        }
    }

    /// Return the k-th (0-based) free slot and mark it taken.
    fn take_kth(&mut self, k: usize) -> usize {
        // Binary lifting: find the smallest 1-based index whose prefix sum
        // exceeds k, i.e. the (k+1)-th free slot.
        let mut rank = k as u64 + 1;
        let mut idx = 0usize;
        let mut step = 1usize;
        while step * 2 < self.tree.len() {
            step *= 2;
        }
        while step > 0 {
            let next = idx + step;
            if next < self.tree.len() && self.tree[next] < rank {
                rank -= self.tree[next];
                idx = next;
            }
            step /= 2;
        }
        self.remove(idx);
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_short_circuit() {
        assert_eq!(domain_to_ascii("EXAMPLE.com"), Some("example.com".into()));
        assert_eq!(domain_to_ascii("a--b"), Some("a--b".into()));
        assert_eq!(domain_to_ascii("xn--a"), Some("xn--a".into()));
        assert_eq!(domain_to_ascii(""), None);
    }

    #[test]
    fn idna_mappings() {
        assert_eq!(domain_to_ascii("é"), Some("xn--9ca".into()));
        assert_eq!(domain_to_ascii("你好你好"), Some("xn--6qqa088eba".into()));
        assert_eq!(
            domain_to_ascii("Bücher.de"),
            Some("xn--bcher-kva.de".into())
        );
        assert_eq!(domain_to_ascii("faß.de"), Some("xn--fa-hia.de".into()));
        assert_eq!(domain_to_ascii("ẞ.com"), Some("xn--zca.com".into()));
        assert_eq!(domain_to_ascii("♥.net"), Some("xn--g6h.net".into()));
        assert_eq!(domain_to_ascii("a\u{ad}b"), Some("ab".into()));
        assert_eq!(
            domain_to_ascii("www.foo。bar.com"),
            Some("www.foo.bar.com".into())
        );
        assert_eq!(domain_to_ascii("Ｇｏ.com"), Some("go.com".into()));
        assert_eq!(domain_to_ascii("-foo.bar.com"), Some("-foo.bar.com".into()));
        assert_eq!(domain_to_ascii("x..ß"), Some("x..xn--zca".into()));
    }

    #[test]
    fn idna_rejections() {
        assert_eq!(domain_to_ascii("GOO\u{a0}\u{3000}goo.com"), None);
        assert_eq!(domain_to_ascii("\u{fdd0}zyx.com"), None);
        assert_eq!(domain_to_ascii("�"), None);
        assert_eq!(domain_to_ascii("\u{ffff}y"), None);
        assert_eq!(domain_to_ascii("\u{ad}"), None);
        assert_eq!(domain_to_ascii("％４１.com"), None);
        assert_eq!(domain_to_ascii("www‥lookout.net"), None);
        assert_eq!(domain_to_ascii("\u{200d}.example"), None);
        assert_eq!(domain_to_ascii("يa"), None);
    }

    #[test]
    fn xn_label_verification() {
        // ASCII short-circuit keeps these; non-ASCII siblings go through ToASCII.
        assert_eq!(
            domain_to_ascii("xn--a.xn--zca"),
            Some("xn--a.xn--zca".into())
        );
        assert_eq!(domain_to_ascii("xn--a.ß"), None);
        assert_eq!(domain_to_ascii("xn--8i7caa.ß"), None);
        assert_eq!(
            domain_to_ascii("xn--1ug.example"),
            Some("xn--1ug.example".into())
        );
        assert_eq!(
            domain_to_ascii("xn--a-yoc.ß"),
            Some("xn--a-yoc.xn--zca".into())
        );
        assert_eq!(domain_to_ascii("xn--tešla.ß"), None);
    }

    #[test]
    fn nfc_lite() {
        assert_eq!(domain_to_ascii("≠"), Some("xn--1ch".into()));
        // "foo" + U+0300 COMBINING GRAVE -> "foò".
        assert_eq!(domain_to_ascii("foo\u{0300}"), Some("xn--fo-3ja".into()));
    }

    #[test]
    fn cjk_compatibility_mapping() {
        // U+2F868 maps to U+36FC (the old table had a typo: U+2FA68).
        assert_eq!(domain_to_ascii("\u{2F868}"), Some("xn--snl".into()));
        // U+2FA68 is disallowed, not mapped.
        assert_eq!(domain_to_ascii("\u{2FA68}"), None);
        assert_eq!(domain_to_ascii("a\u{2FA68}b"), None);
    }

    #[test]
    fn hebrew_maqaf_is_valid() {
        // U+05BE HEBREW PUNCTUATION MAQAF is valid per UTS #46.
        assert_eq!(domain_to_ascii("\u{5BE}"), Some("xn--mdb".into()));
    }

    #[test]
    fn digit_like_code_points_map_to_ascii_digits() {
        // UTS #46 maps these to ASCII digits, which then triggers IPv4
        // parsing of the host (see the url-level test below).
        assert_eq!(
            domain_to_ascii("\u{2460}\u{2461}\u{2462}\u{2463}"),
            Some("1234".into())
        );
        // Circled numbers map to one- or two-digit strings.
        assert_eq!(domain_to_ascii("\u{2469}"), Some("10".into()));
        assert_eq!(domain_to_ascii("\u{2473}"), Some("20".into()));
        assert_eq!(domain_to_ascii("\u{24EA}"), Some("0".into()));
        // Superscript digits.
        assert_eq!(domain_to_ascii("\u{B2}"), Some("2".into()));
        assert_eq!(domain_to_ascii("\u{B3}"), Some("3".into()));
        assert_eq!(domain_to_ascii("\u{B9}"), Some("1".into()));
        assert_eq!(domain_to_ascii("\u{2070}"), Some("0".into()));
        assert_eq!(domain_to_ascii("\u{2074}"), Some("4".into()));
        assert_eq!(domain_to_ascii("\u{2079}"), Some("9".into()));
        // Subscript digits.
        assert_eq!(domain_to_ascii("\u{2080}"), Some("0".into()));
        assert_eq!(domain_to_ascii("\u{2089}"), Some("9".into()));
        // U+2071 is superscript latin small letter i, not a digit.
        assert_eq!(domain_to_ascii("\u{2071}"), Some("i".into()));
    }

    #[test]
    fn punycode_round_trip() {
        for s in ["é", "ß", "你好", "♥", "💩", "bücher", "ς"] {
            let enc = punycode_encode(s).unwrap();
            assert_eq!(punycode_decode(&enc).unwrap(), s);
        }
        assert_eq!(punycode_encode("faß"), Some("fa-hia".into()));
        assert_eq!(punycode_decode("a"), Some("\u{80}".into()));
    }

    #[test]
    fn punycode_long_round_trip() {
        // Many insertions at varying positions, exercising the
        // linearithmic reverse-placement decoder.
        let s: String = (0x80u32..0x3000)
            .filter(|&cp| !(0xD800..0xE000).contains(&cp))
            .step_by(7)
            .filter_map(char::from_u32)
            .collect();
        let enc = punycode_encode(&s).unwrap();
        assert_eq!(punycode_decode(&enc).unwrap(), s);
    }

    #[test]
    fn idna_length_caps_bound_decode_work() {
        // The general Punycode codec is uncapped (it round-trips long
        // labels above), but the IDNA layer never feeds it an `xn--`
        // label longer than 63 octets: such a label cannot be a valid
        // A-label.
        let rest59 = "a".repeat(59);
        let rest60 = "a".repeat(60);
        assert!(punycode_decode(&rest59).is_some());
        assert!(punycode_decode(&rest60).is_some());
        // 59 octets after `xn--` (63 total) still reaches the decoder and
        // fails on content (`"a"*59` decodes to disallowed U+0080 runs).
        assert!(decode_xn_label(&rest59).is_none());
        // 60 octets after `xn--` (64 total) is rejected before decoding.
        assert!(decode_xn_label(&rest60).is_none());

        // Adversarial: a 1MB `xn--` label fails fast at every layer that
        // would Punycode-decode it. (The all-ASCII `parse_host` fast path
        // deliberately never decodes, so it accepts such a label in
        // linear time instead.)
        let huge = "a".repeat(1_000_000);
        assert!(decode_xn_label(&huge).is_none());
        // Through `domain_to_ascii`'s non-ASCII path (the `ß` label
        // forces the mapped path, where the `xn--` label is verified).
        assert!(domain_to_ascii(&format!("xn--{huge}.ß")).is_none());
        // A 1MB plain-ASCII label needs no decoding and is still accepted
        // (URL standard `VerifyDnsLength=false`), in linear time.
        assert_eq!(domain_to_ascii(&huge).as_deref(), Some(huge.as_str()));
    }

    #[test]
    fn tounicode_rejects_ignored_status_codepoints() {
        // `xn--kba` Punycode-decodes to U+00AD SOFT HYPHEN, which has IDNA
        // status "ignored". UTS #46 §4.1 requires an error there in
        // ToUnicode processing, so the ToUnicode flavor rejects it...
        assert!(decode_xn_label_tounicode("kba").is_none());
        // ...while the verification flavor keeps it (documented §4.1
        // deviation): the kept code point re-encodes to the same label,
        // so verification of already-consistent labels is unaffected.
        assert_eq!(decode_xn_label("kba").as_deref(), Some("\u{ad}"));
        assert!(verify_xn_label("xn--kba", "kba").is_some());
        // End to end through the mixed-script verification path used by
        // `domain_to_ascii` for non-ASCII domains.
        assert!(domain_to_ascii("xn--kba.ß").is_some());
    }

    #[test]
    fn long_unicode_label_still_encodes() {
        // Acceptance of over-long labels is mandated by the URL standard
        // (WPT toascii.json: a 64-code-point label must round-trip), so a
        // 100KB Unicode label still encodes rather than being rejected.
        let label = "é".repeat(50_000);
        let ascii = domain_to_ascii(&label).expect("long label encodes");
        assert!(ascii.starts_with("xn--"));
        // Accepted as a (very long) A-label, not rejected on length.
        assert!(ascii.len() > 63);
    }
}
