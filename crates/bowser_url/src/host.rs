//! Host parsing: domains, IPv4, IPv6, and opaque hosts.
//!
//! See <https://url.spec.whatwg.org/#host-parsing>.

use crate::idna::{domain_to_ascii, is_forbidden_host_code_point};
use crate::percent::{EncodeSet, percent_decode, percent_encode, utf8_decode_lossy};

/// A parsed host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Host {
    /// An ASCII domain (IDNA processing already applied).
    Domain(String),
    /// An IPv4 address as a 32-bit integer.
    Ipv4(u32),
    /// An IPv6 address as eight 16-bit pieces.
    Ipv6([u16; 8]),
    /// An opaque host (non-special schemes), percent-encoded with the C0 set.
    Opaque(String),
}

impl Host {
    /// Serialize the host as it appears in a URL string.
    pub fn serialize(&self) -> String {
        match self {
            Host::Domain(d) | Host::Opaque(d) => d.clone(),
            Host::Ipv4(a) => ipv4_serialize(*a),
            Host::Ipv6(p) => {
                let mut s = String::with_capacity(41);
                s.push('[');
                ipv6_serialize_into(&mut s, p);
                s.push(']');
                s
            }
        }
    }
}

/// Parse a host per the URL standard's host parser.
///
/// `is_special` is whether the URL's scheme is special. Returns `None` on
/// failure.
pub fn parse_host(input: &str, is_special: bool) -> Option<Host> {
    if input.starts_with('[') {
        if !input.ends_with(']') {
            return None;
        }
        // '[' and ']' are ASCII, so byte slicing is on char boundaries; both
        // guards held, so `input.len() >= 2` (no 1-char string both starts
        // with '[' and ends with ']'), and `1..len-1` is in bounds.
        return ipv6_parse(&input[1..input.len() - 1]).map(Host::Ipv6);
    }
    if !is_special {
        return opaque_host_parse(input).map(Host::Opaque);
    }
    if input.is_empty() {
        return None;
    }
    // A leading BOM is stripped by the UTF-8 decode on the slow path; do
    // the same up front so the fast path below agrees.
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    if input.is_empty() {
        return None;
    }
    if input.bytes().any(|b| b == b'%') {
        // A percent-encoded byte in the input is a non-fatal validation error.
        let domain = utf8_decode_lossy(&percent_decode(input));
        let ascii_domain = domain_to_ascii(&domain)?;
        return finish_domain(ascii_domain);
    }
    if input.is_ascii() {
        // Fast path: no percent-decoding to do. Lowercase and validate in a
        // single byte pass.
        let bytes = input.as_bytes();
        let mut lower = String::with_capacity(bytes.len());
        for &b in bytes {
            if is_forbidden_domain_byte(b) {
                return None;
            }
            lower.push(b.to_ascii_lowercase() as char);
        }
        // WHATWG URL §3.3 domain parser step 5: when beStrict is false and
        // the domain is ASCII, the lowercased domain is returned regardless
        // of ToASCII outcome, so no A-label verification happens here.
        // `xn--a` stays accepted (the spec's example `xn--8i7caa`, and WPT
        // toascii.json expects `"xn--a" -> "xn--a"`).
        return finish_domain(lower);
    }
    let ascii_domain = domain_to_ascii(input)?;
    finish_domain(ascii_domain)
}

/// Shared tail of [`parse_host`] for special schemes: parse as IPv4 when
/// the domain ends in a number, otherwise keep it as a domain.
fn finish_domain(ascii_domain: String) -> Option<Host> {
    if ascii_domain.is_empty() {
        return None;
    }
    if ends_in_number(&ascii_domain) {
        // Non-ASCII input reaching IPv4 is a non-fatal validation error.
        return ipv4_parse(&ascii_domain).map(Host::Ipv4);
    }
    Some(Host::Domain(ascii_domain))
}

/// Byte version of [`is_forbidden_domain_code_point`] for ASCII input.
fn is_forbidden_domain_byte(b: u8) -> bool {
    // is_ascii_control() or one of the listed code points; '%' included.
    b < 0x20
        || b == 0x7f
        || matches!(
            b,
            b' ' | b'#'
                | b'/'
                | b':'
                | b'<'
                | b'>'
                | b'?'
                | b'@'
                | b'['
                | b'\\'
                | b']'
                | b'^'
                | b'|'
                | b'%'
        )
}

/// The opaque-host parser: reject forbidden host code points, then UTF-8
/// percent-encode with the C0 set. The input is *not* percent-decoded.
pub fn opaque_host_parse(input: &str) -> Option<String> {
    if input.chars().any(is_forbidden_host_code_point) {
        return None;
    }
    Some(percent_encode(input, EncodeSet::C0))
}

// ---------------------------------------------------------------------------
// IPv4
// ---------------------------------------------------------------------------

/// Parse an IPv4 address per the URL standard. Returns the 32-bit value.
pub fn ipv4_parse(input: &str) -> Option<u32> {
    // Split into parts without allocating. A trailing empty part (from a
    // trailing '.') is dropped when there is more than one part, mirroring
    // the spec's split-then-pop.
    let s = if input.contains('.') {
        input.strip_suffix('.').unwrap_or(input)
    } else {
        input
    };
    let mut numbers = [0u64; 4];
    let mut count = 0usize;
    for part in s.split('.') {
        if count == 4 {
            return None;
        }
        let (n, _non_decimal) = ipv4_number_parse(part)?;
        // `count < 4` by the guard above. `split('.')` yields at least one
        // part and every iteration either returns or increments `count`,
        // so `count >= 1` after the loop: `count - 1` cannot underflow.
        numbers[count] = n;
        count += 1;
    }
    // Any part above 255 is a non-fatal validation error; any non-final
    // part above 255 is fatal.
    if numbers[..count - 1].iter().any(|&n| n > 255) {
        return None;
    }
    let last = numbers[count - 1];
    if last >= 256u64.pow(5 - count as u32) {
        return None;
    }
    let mut ipv4 = last;
    for (counter, &n) in numbers[..count - 1].iter().enumerate() {
        ipv4 += n * 256u64.pow(3 - counter as u32);
    }
    u32::try_from(ipv4).ok()
}

/// Parse one IPv4 number. Returns the value and whether it was non-decimal
/// (hexadecimal or octal), which is a non-fatal validation error.
fn ipv4_number_parse(input: &str) -> Option<(u64, bool)> {
    if input.is_empty() {
        return None;
    }
    let mut non_decimal = false;
    let mut radix = 10u32;
    let mut s = input;
    // "0x"/"0X"/"0" are ASCII, so these slices are on char boundaries;
    // `s.len() >= 2` keeps `2..` and `1..` in bounds.
    if s.len() >= 2 && (s.starts_with("0x") || s.starts_with("0X")) {
        non_decimal = true;
        s = &s[2..];
        radix = 16;
    } else if s.len() >= 2 && s.starts_with('0') {
        non_decimal = true;
        s = &s[1..];
        radix = 8;
    }
    if s.is_empty() {
        return Some((0, true));
    }
    let mut out: u64 = 0;
    for c in s.chars() {
        let d = c.to_digit(radix)? as u64;
        // The standard uses mathematical integers; saturate so the range
        // checks below still reject the value.
        out = out.saturating_mul(radix as u64).saturating_add(d);
    }
    Some((out, non_decimal))
}

/// The ends-in-a-number checker. `input` must be ASCII and non-empty.
pub fn ends_in_number(input: &str) -> bool {
    // Drop one trailing empty part, mirroring split('.') + pop.
    let s = input.strip_suffix('.').unwrap_or(input);
    let last = s.rsplit('.').next().unwrap_or("");
    if !last.is_empty() && last.bytes().all(|b| b.is_ascii_digit()) {
        return true;
    }
    ipv4_number_parse(last).is_some()
}

fn ipv4_serialize(address: u32) -> String {
    // "255.255.255.255" is the longest form: 15 bytes.
    let mut out = String::with_capacity(15);
    for (i, &octet) in address.to_be_bytes().iter().enumerate() {
        if i > 0 {
            out.push('.');
        }
        push_decimal_u8(&mut out, octet);
    }
    out
}

/// Append a `u8` as decimal digits, without `format!`.
fn push_decimal_u8(out: &mut String, v: u8) {
    if v >= 100 {
        out.push((b'0' + v / 100) as char);
    }
    if v >= 10 {
        out.push((b'0' + (v / 10) % 10) as char);
    }
    out.push((b'0' + v % 10) as char);
}

// ---------------------------------------------------------------------------
// IPv6
// ---------------------------------------------------------------------------

/// Parse an IPv6 address per the URL standard.
///
/// Operates on bytes: IPv6 input is ASCII (hex digits, `:`, `.`), and any
/// non-ASCII byte fails the digit/separator tests exactly like the
/// corresponding non-ASCII char would.
pub fn ipv6_parse(input: &str) -> Option<[u16; 8]> {
    let b = input.as_bytes();
    let mut address = [0u16; 8];
    let mut piece_index: usize = 0;
    let mut compress: Option<usize> = None;
    let mut pointer: usize = 0;

    let c = |p: usize| -> Option<u8> { b.get(p).copied() };

    // Step 5: leading "::".
    if c(pointer) == Some(b':') {
        if c(pointer + 1) != Some(b':') {
            return None;
        }
        pointer += 2;
        piece_index += 1;
        compress = Some(piece_index);
    }

    while c(pointer).is_some() {
        if piece_index == 8 {
            return None;
        }
        if c(pointer) == Some(b':') {
            if compress.is_some() {
                return None;
            }
            pointer += 1;
            piece_index += 1;
            compress = Some(piece_index);
            continue;
        }
        let mut value: u32 = 0;
        let mut length = 0;
        while length < 4 {
            match c(pointer) {
                Some(ch) if ch.is_ascii_hexdigit() => {
                    value = value * 16 + hex_val(ch) as u32;
                    pointer += 1;
                    length += 1;
                }
                _ => break,
            }
        }
        if c(pointer) == Some(b'.') {
            if length == 0 {
                return None;
            }
            pointer -= length;
            if piece_index > 6 {
                return None;
            }
            let mut numbers_seen = 0;
            while c(pointer).is_some() {
                let mut ipv4_piece: Option<u32> = None;
                if numbers_seen > 0 {
                    if c(pointer) == Some(b'.') && numbers_seen < 4 {
                        pointer += 1;
                    } else {
                        return None;
                    }
                }
                match c(pointer) {
                    Some(ch) if ch.is_ascii_digit() => {}
                    _ => return None,
                }
                while let Some(ch) = c(pointer) {
                    if !ch.is_ascii_digit() {
                        break;
                    }
                    let number = (ch - b'0') as u32;
                    match ipv4_piece {
                        None => ipv4_piece = Some(number),
                        Some(0) => return None,
                        Some(p) => {
                            let v = p * 10 + number;
                            if v > 255 {
                                return None;
                            }
                            ipv4_piece = Some(v);
                        }
                    }
                    pointer += 1;
                }
                let Some(piece) = ipv4_piece else {
                    // The digit check above guarantees at least one digit
                    // was consumed; bail out instead of panicking if the
                    // invariant ever breaks.
                    debug_assert!(false, "at least one digit seen");
                    return None;
                };
                // `piece_index <= 6` on entry (the `> 6` check above returned
                // otherwise). Each iteration writes at the current index and
                // increments at most once, and the loop ends after the
                // fourth number, so writes land at indices <= 7.
                debug_assert!(piece_index < 8, "ipv4 tail piece in bounds");
                address[piece_index] = address[piece_index] * 0x100 + piece as u16;
                numbers_seen += 1;
                if numbers_seen == 2 || numbers_seen == 4 {
                    piece_index += 1;
                }
            }
            if numbers_seen != 4 {
                return None;
            }
            break;
        } else if c(pointer) == Some(b':') {
            pointer += 1;
            c(pointer)?;
        } else if c(pointer).is_some() {
            return None;
        }
        // IPv6-piece-leading-zero is a non-fatal validation error.
        // `piece_index < 8` by the check at the top of the loop.
        debug_assert!(piece_index < 8, "ipv6 piece in bounds");
        address[piece_index] = value as u16;
        piece_index += 1;
    }

    if let Some(compress_at) = compress {
        // `compress_at <= piece_index <= 8`, so both swap indices stay
        // below 8: `compress_at + swaps - 1 == piece_index - 1 <= 7` and
        // `p` counts down from 7.
        let mut swaps = piece_index - compress_at;
        let mut p = 7;
        while p != 0 && swaps > 0 {
            address.swap(p, compress_at + swaps - 1);
            p -= 1;
            swaps -= 1;
        }
    } else if piece_index != 8 {
        return None;
    }
    Some(address)
}

/// Numeric value of an ASCII hex digit.
fn hex_val(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        _ => b - b'A' + 10,
    }
}

/// Serialize an IPv6 address per RFC 5952 (longest run of zero pieces of
/// length > 1 is compressed; ties go to the first run).
/// Append the RFC 5952 serialization of an IPv6 address to `out`: the
/// longest run of zero pieces of length > 1 is compressed, ties go to the
/// first run. No `format!` involved.
fn ipv6_serialize_into(out: &mut String, address: &[u16; 8]) {
    // Find the compressed piece index.
    let mut compress: Option<usize> = None;
    let mut best_start = 0;
    let mut best_len = 0;
    let mut cur_start = 0;
    let mut cur_len = 0;
    for (i, &piece) in address.iter().enumerate() {
        if piece == 0 {
            if cur_len == 0 {
                cur_start = i;
            }
            cur_len += 1;
            if cur_len > best_len {
                best_len = cur_len;
                best_start = cur_start;
            }
        } else {
            cur_len = 0;
        }
    }
    if best_len > 1 {
        compress = Some(best_start);
    }

    let mut ignore0 = false;
    for (i, &piece) in address.iter().enumerate() {
        if ignore0 && piece == 0 {
            continue;
        } else if ignore0 {
            ignore0 = false;
        }
        if Some(i) == compress {
            out.push_str(if i == 0 { "::" } else { ":" });
            ignore0 = true;
            continue;
        }
        push_hex_u16(out, piece);
        if i != 7 {
            out.push(':');
        }
    }
}

/// Append a `u16` as lowercase hexadecimal digits without leading zeros.
fn push_hex_u16(out: &mut String, v: u16) {
    let mut shift = 12;
    let mut started = false;
    while shift > 0 {
        let d = ((v >> shift) & 0xf) as u8;
        if d != 0 || started {
            started = true;
            out.push(hex_lower(d));
        }
        shift -= 4;
    }
    out.push(hex_lower((v & 0xf) as u8));
}

fn hex_lower(n: u8) -> char {
    (b'0' + n + if n >= 10 { b'a' - b'0' - 10 } else { 0 }) as char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipv4_cases() {
        assert_eq!(ipv4_parse("192.168.0.1"), Some(0xC0A80001));
        assert_eq!(ipv4_parse("192.0x00A80001"), Some(0xC0A80001));
        assert_eq!(ipv4_parse("0xc0.0250.01"), Some(0xC0A80001));
        assert_eq!(ipv4_parse("192.168.257"), Some(0xC0A80101));
        assert_eq!(ipv4_parse("999999999"), Some(0x3B9AC9FF));
        assert_eq!(ipv4_parse("1.2.3.4."), Some(0x01020304));
        assert_eq!(ipv4_parse("0x7f.0.0.1"), Some(0x7F000001));
        assert_eq!(ipv4_parse("0177.0.0.1"), Some(0x7F000001));
        assert_eq!(ipv4_parse("123"), Some(123));
        assert_eq!(ipv4_parse("256.256.256.256"), None);
        assert_eq!(ipv4_parse("1.2.3.09"), None);
        assert_eq!(ipv4_parse("1.2.3.4.5"), None);
        assert_eq!(ipv4_parse("1.2.3.256"), None);
        assert_eq!(ipv4_parse(""), None);
    }

    #[test]
    fn ipv6_cases() {
        assert_eq!(
            ipv6_parse("1:2:0:0:5:0:0:0"),
            Some([1, 2, 0, 0, 5, 0, 0, 0])
        );
        assert_eq!(ipv6_parse("::1"), Some([0, 0, 0, 0, 0, 0, 0, 1]));
        assert_eq!(ipv6_parse("::"), Some([0; 8]));
        assert_eq!(ipv6_parse("fe80::1"), Some([0xfe80, 0, 0, 0, 0, 0, 0, 1]));
        assert_eq!(
            ipv6_parse("::ffff:192.168.0.1"),
            Some([0, 0, 0, 0, 0, 0xffff, 0xc0a8, 1])
        );
        assert_eq!(ipv6_parse("1::2::3"), None);
        assert_eq!(ipv6_parse(":1"), None);
        assert_eq!(ipv6_parse("1:"), None);
        assert_eq!(ipv6_parse("12345::"), None);
        assert_eq!(ipv6_parse(""), None);
    }

    #[test]
    fn ipv6_serialization() {
        fn ipv6_serialize(address: &[u16; 8]) -> String {
            let mut out = String::new();
            ipv6_serialize_into(&mut out, address);
            out
        }
        assert_eq!(ipv6_serialize(&[1, 2, 0, 0, 5, 0, 0, 0]), "1:2:0:0:5::");
        assert_eq!(ipv6_serialize(&[0; 8]), "::");
        assert_eq!(ipv6_serialize(&[0, 0, 0, 0, 0, 0, 0, 1]), "::1");
        assert_eq!(ipv6_serialize(&[0xfe80, 0, 0, 0, 0, 0, 0, 1]), "fe80::1");
        // Ties compress the first run.
        assert_eq!(ipv6_serialize(&[1, 0, 0, 2, 0, 0, 3, 4]), "1::2:0:0:3:4");
    }

    #[test]
    fn host_parsing() {
        assert_eq!(
            parse_host("example.com", true),
            Some(Host::Domain("example.com".into()))
        );
        assert_eq!(
            parse_host("EXAMPLE.com", true),
            Some(Host::Domain("example.com".into()))
        );
        assert_eq!(parse_host("0x7f.0.0.1", true), Some(Host::Ipv4(0x7F000001)));
        assert_eq!(
            parse_host("[::1]", true),
            Some(Host::Ipv6([0, 0, 0, 0, 0, 0, 0, 1]))
        );
        assert_eq!(parse_host("[::1", true), None);
        assert_eq!(
            parse_host("faß.ExAmPlE", true),
            Some(Host::Domain("xn--fa-hia.example".into()))
        );
        // Opaque hosts are not lowercased, decoded, or IDNA-processed.
        assert_eq!(
            parse_host("H%4fSt", false),
            Some(Host::Opaque("H%4fSt".into()))
        );
        assert_eq!(
            parse_host("faß.ExAmPlE", false),
            Some(Host::Opaque("fa%C3%9F.ExAmPlE".into()))
        );
        assert_eq!(parse_host("a b", false), None);
        assert_eq!(parse_host("", true), None);
    }

    #[test]
    fn ascii_fast_path_accepts_xn_labels() {
        // WHATWG URL §3.3 domain parser step 5: when beStrict is false and
        // the domain is ASCII, the lowercased domain is returned regardless
        // of ToASCII outcome. `xn--a` decodes to disallowed U+0080, but it
        // is still accepted (the spec's example `xn--8i7caa`; WPT
        // toascii.json expects `"xn--a" -> "xn--a"`).
        assert_eq!(
            parse_host("xn--a", true),
            Some(Host::Domain("xn--a".into()))
        );
        assert_eq!(
            parse_host("xn--8i7caa", true),
            Some(Host::Domain("xn--8i7caa".into()))
        );
        assert_eq!(
            parse_host("xn--a.com", true),
            Some(Host::Domain("xn--a.com".into()))
        );
        assert_eq!(
            parse_host("sub.xn--a", true),
            Some(Host::Domain("sub.xn--a".into()))
        );
        // Case-insensitive: the label is lowercased, then accepted.
        assert_eq!(
            parse_host("XN--A.com", true),
            Some(Host::Domain("xn--a.com".into()))
        );
        // The non-ASCII path still verifies: `xn--a` decodes to disallowed
        // U+0080, so the whole domain is rejected.
        assert_eq!(parse_host("xn--a.é", true), None);
        // Non-ASCII input in the label itself also goes through
        // `domain_to_ascii` and fails verification.
        assert_eq!(parse_host("xn--tešla", true), None);
        // Valid `xn--` labels still pass, including WPT urltestdata hosts.
        assert_eq!(
            parse_host("xn--9ca", true),
            Some(Host::Domain("xn--9ca".into()))
        );
        assert_eq!(
            parse_host("xn--6qqa088eba", true),
            Some(Host::Domain("xn--6qqa088eba".into()))
        );
        assert_eq!(
            parse_host("a.b.c.xn--pokxncvks", true),
            Some(Host::Domain("a.b.c.xn--pokxncvks".into()))
        );
        // Bare `xn--` is not Punycode at all; it passes through unchanged.
        assert_eq!(parse_host("xn--", true), Some(Host::Domain("xn--".into())));
        // Leading/trailing hyphens are allowed: the URL standard uses
        // `CheckHyphens=false` (WPT toascii expects `-x` -> `-x`).
        assert_eq!(
            parse_host("-foo.com", true),
            Some(Host::Domain("-foo.com".into()))
        );
        assert_eq!(
            parse_host("foo-.com", true),
            Some(Host::Domain("foo-.com".into()))
        );
    }

    #[test]
    fn ascii_fast_path_skips_decoding_huge_labels() {
        // The all-ASCII fast path does no Punycode decoding: over-long
        // `xn--` labels are accepted in linear time (URL standard
        // `VerifyDnsLength=false`; WPT toascii.json expects long labels to
        // round-trip). The 63-octet pre-decode cap stays in
        // `decode_xn_label`, guarding `domain_to_unicode` and the
        // non-ASCII A-label verification path.
        let label64 = format!("xn--{}", "a".repeat(60));
        let label63 = format!("xn--{}", "a".repeat(59));
        assert_eq!(
            parse_host(&label64, true),
            Some(Host::Domain(label64.clone()))
        );
        assert_eq!(
            parse_host(&label63, true),
            Some(Host::Domain(label63.clone()))
        );
        let huge = format!("xn--{}", "a".repeat(1_000_000));
        assert_eq!(parse_host(&huge, true), Some(Host::Domain(huge)));
        // Sanity: short `xn--` labels are unaffected.
        assert_eq!(
            parse_host("xn--9ca", true),
            Some(Host::Domain("xn--9ca".into()))
        );
    }

    #[test]
    fn literal_xn_label_matches_percent_encoded() {
        // The all-ASCII fast path must agree with the percent-decoded
        // path, which runs `domain_to_ascii` without A-label verification.
        for (literal, encoded) in [
            ("xn--a", "xn--%61"),
            ("xn--8i7caa", "xn--8i7c%61%61"),
            ("xn--9ca", "xn--%39ca"),
            ("example.com", "ex%61mple.com"),
        ] {
            assert_eq!(parse_host(encoded, true), parse_host(literal, true));
            assert_eq!(
                parse_host(literal, true),
                Some(Host::Domain(literal.to_string()))
            );
        }
    }

    #[test]
    fn ends_in_number_cases() {
        assert!(ends_in_number("foo.09"));
        assert!(ends_in_number("foo.0x4"));
        assert!(!ends_in_number("foo.09.."));
        assert!(!ends_in_number("0x7f.0.0.0x7g"));
        assert!(ends_in_number("123"));
    }
}
