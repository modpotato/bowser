//! Decoder unit tests with hand-verified vectors.
//!
//! Multi-byte CJK vectors were cross-checked against the WHATWG index files
//! and `iconv`; UTF-8 edge cases follow the spec's "Best Practices for Using
//! U+FFFD" (maximal subparts).

use bowser_encoding::{Encoding, decode, decode_with, output_encoding};

fn dec(bytes: &[u8], enc: Encoding) -> (String, bool) {
    decode_with(bytes, enc).expect("supported encoding")
}

fn txt(bytes: &[u8], enc: Encoding) -> String {
    dec(bytes, enc).0
}

#[test]
fn utf8_valid() {
    assert_eq!(txt(b"hello", Encoding::Utf8), "hello");
    assert_eq!(
        txt("héllo wörld 😀".as_bytes(), Encoding::Utf8),
        "héllo wörld 😀"
    );
    assert_eq!(txt(b"\x00\x7f", Encoding::Utf8), "\0\x7f");
    assert!(!dec(b"abc", Encoding::Utf8).1);
}

#[test]
fn utf8_errors_follow_maximal_subparts() {
    // Overlong: E0 80 80 -> one FFFD per byte the spec consumes.
    assert_eq!(
        txt(b"\xe0\x80\x80", Encoding::Utf8),
        "\u{FFFD}\u{FFFD}\u{FFFD}"
    );
    // Surrogate half: ED A0 80 -> 3 x FFFD.
    assert_eq!(
        txt(b"\xed\xa0\x80", Encoding::Utf8),
        "\u{FFFD}\u{FFFD}\u{FFFD}"
    );
    // Truncated sequence at end -> exactly one FFFD.
    assert_eq!(txt(b"\xe1\x80", Encoding::Utf8), "\u{FFFD}");
    assert_eq!(txt(b"\xf0\x9f\x98", Encoding::Utf8), "\u{FFFD}");
    // Lone continuation bytes.
    assert_eq!(txt(b"\x80\xbf", Encoding::Utf8), "\u{FFFD}\u{FFFD}");
    // F0 28 8C 28 -> FFFD ( FFFD (.
    assert_eq!(
        txt(b"\xf0\x28\x8c\x28", Encoding::Utf8),
        "\u{FFFD}(\u{FFFD}("
    );
    // 0xC0/0xC1/0xF5-0xFF are never valid leads.
    assert_eq!(txt(b"\xc0\xaf", Encoding::Utf8), "\u{FFFD}\u{FFFD}");
    assert_eq!(
        txt(b"\xf5\x80\x80\x80", Encoding::Utf8),
        "\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}"
    );
    // E0 with bad second byte: E0 consumed alone, then the rest reprocessed.
    assert_eq!(txt(b"\xe0\x28\xa1", Encoding::Utf8), "\u{FFFD}(\u{FFFD}");
    // Error flag is set.
    assert!(dec(b"\xff", Encoding::Utf8).1);
    assert!(dec(b"a\xffb", Encoding::Utf8).0 == "a\u{FFFD}b");
}

#[test]
fn windows_1252_spot_checks() {
    let e = Encoding::Windows1252;
    assert_eq!(txt(b"\x80", e), "\u{20AC}"); // euro
    assert_eq!(txt(b"\x81", e), "\u{81}"); // control passthrough
    assert_eq!(txt(b"\x9d", e), "\u{9D}");
    assert_eq!(txt(b"\x9f", e), "\u{178}"); // Ÿ
    assert_eq!(txt(b"\xa0", e), "\u{a0}");
    assert_eq!(txt(b"\xff", e), "ÿ");
    assert_eq!(txt(b"caf\xe9", e), "café");
    assert!(!dec(b"abc", e).1);
}

#[test]
fn single_byte_spot_checks() {
    // IBM866: 0x80 -> U+0410 (А).
    assert_eq!(txt(b"\x80", Encoding::Ibm866), "\u{410}");
    // ISO-8859-7: 0xE1 -> U+03B1 (α).
    assert_eq!(txt(b"\xe1", Encoding::Iso8859_7), "\u{3B1}");
    // ISO-8859-5: 0xD0 -> U+0430 (а).
    assert_eq!(txt(b"\xd0", Encoding::Iso8859_5), "\u{430}");
    // KOI8-R: 0xE1 -> U+0410 (А).
    assert_eq!(txt(b"\xe1", Encoding::Koi8R), "\u{410}");
    // windows-1251: 0xC0 -> U+0410 (А).
    assert_eq!(txt(b"\xc0", Encoding::Windows1251), "\u{410}");
    // windows-1250: 0x9A -> U+0161 (š).
    assert_eq!(txt(b"\x9a", Encoding::Windows1250), "\u{161}");
    // macintosh: 0xDB -> U+20AC (euro in mac roman).
    assert_eq!(txt(b"\xdb", Encoding::Macintosh), "\u{20AC}");
    // windows-874: 0xA1 -> U+0E01 (ก).
    assert_eq!(txt(b"\xa1", Encoding::Windows874), "\u{E01}");
    // Unmapped byte -> FFFD with error (ISO-8859-6 0xA1 is unmapped).
    let (s, err) = dec(b"\xa1", Encoding::Iso8859_6);
    assert_eq!(s, "\u{FFFD}");
    assert!(err);
    // ASCII always maps to itself.
    for enc in [Encoding::Windows1252, Encoding::Macintosh, Encoding::Koi8U] {
        assert_eq!(txt(b"\x41\x7e", enc), "A~");
    }
}

#[test]
fn big5_decode() {
    let e = Encoding::Big5;
    assert_eq!(txt(b"\xa4\x40", e), "\u{4E00}"); // 一
    assert_eq!(txt(b"\xc0\xc6", e), "\u{66D9}"); // 曙
    // HKSCS two-code-point entries from the spec's table.
    assert_eq!(txt(b"\x88\x62", e), "\u{CA}\u{304}");
    assert_eq!(txt(b"\x88\x64", e), "\u{CA}\u{30C}");
    // 0x41 is a valid trail: 88 41 -> U+31C1.
    assert_eq!(txt(b"\x88\x41", e), "\u{31C1}");
    // Invalid trail byte is restored: FFFD then the byte itself.
    assert_eq!(txt(b"\x88\x20", e), "\u{FFFD} ");
    // Truncated lead at end -> one FFFD.
    assert_eq!(txt(b"\x88", e), "\u{FFFD}");
    // Bytes outside the lead range error.
    assert_eq!(txt(b"\x80", e), "\u{FFFD}");
    assert_eq!(txt(b"\xff", e), "\u{FFFD}");
    assert!(!dec(b"\xa4\x40", e).1);
    assert!(dec(b"\x88", e).1);
}

#[test]
fn gbk_and_gb18030_decode() {
    // GBK decode IS gb18030's decoder.
    for e in [Encoding::Gbk, Encoding::Gb18030] {
        assert_eq!(txt(b"\xd6\xd0", e), "\u{4E2D}"); // 中
        assert_eq!(txt(b"\xa1\xa1", e), "\u{3000}"); // ideographic space
        assert_eq!(txt(b"\x80", e), "\u{20AC}"); // euro
        assert_eq!(txt(b"\x81\x40", e), "\u{4E02}");
        assert_eq!(txt(b"abc", e), "abc");
        assert!(!dec(b"\xd6\xd0", e).1);
    }
    let e = Encoding::Gb18030;
    // Four-byte sequences via index gb18030 ranges.
    assert_eq!(txt(b"\x81\x30\x81\x30", e), "\u{80}");
    assert_eq!(txt(b"\x90\x30\x81\x30", e), "\u{10000}");
    assert_eq!(txt(b"\x81\x35\xf4\x37", e), "\u{E7C7}"); // pointer 7457
    // Four-byte pointer in the null gap 39420..189000 -> error.
    assert_eq!(txt(b"\x84\x39\xd7\x30", e), "\u{FFFD}");
    // Four-byte pointer above 1237575 -> error.
    assert_eq!(txt(b"\xfe\x39\xfe\x39", e), "\u{FFFD}");
    // Bad fourth byte: «second, third, byte» restored -> FFFD 0, then the
    // non-ASCII byte 0xFF is consumed by the state-1 error (not restored) ->
    // FFFD, then 9.
    assert_eq!(txt(b"\x81\x30\x81\xff\x39", e), "\u{FFFD}0\u{FFFD}9");
    // Bad second byte: «byte» restored -> FFFD then the byte itself.
    assert_eq!(txt(b"\x81\x20\x30", e), "\u{FFFD} 0");
    // Bad non-ASCII second byte: only ASCII bytes are restored to the
    // queue, so 81 FF is a single FFFD (the FF is consumed by the error).
    assert_eq!(txt(b"\x81\xff", e), "\u{FFFD}");
    assert_eq!(txt(b"\xfe\xff", e), "\u{FFFD}");
    // Truncated states at EOF -> single FFFD each.
    assert_eq!(txt(b"\x81", e), "\u{FFFD}");
    assert_eq!(txt(b"\x81\x30", e), "\u{FFFD}");
    assert_eq!(txt(b"\x81\x30\x81", e), "\u{FFFD}");
    assert!(dec(b"\x81", e).1);
}

#[test]
fn utf16_decode() {
    // LE: "A" + 😀 (surrogate pair).
    assert_eq!(
        txt(b"\x41\x00\x3d\xd8\x00\xde", Encoding::Utf16Le),
        "A\u{1F600}"
    );
    // BE: same.
    assert_eq!(
        txt(b"\x00\x41\xd8\x3d\xde\x00", Encoding::Utf16Be),
        "A\u{1F600}"
    );
    // Lone lead surrogate -> FFFD, next unit reprocessed.
    assert_eq!(txt(b"\x3d\xd8\x41\x00", Encoding::Utf16Le), "\u{FFFD}A");
    // Lead followed by lead: first errors, second pairs with trail.
    assert_eq!(
        txt(b"\x3d\xd8\x3d\xd8\x00\xde", Encoding::Utf16Le),
        "\u{FFFD}\u{1F600}"
    );
    // Lone trail surrogate -> FFFD.
    assert_eq!(txt(b"\x00\xde", Encoding::Utf16Le), "\u{FFFD}");
    // Odd trailing byte -> FFFD.
    assert_eq!(txt(b"\x41\x00\x42", Encoding::Utf16Le), "A\u{FFFD}");
    // Trailing lead surrogate at EOF -> FFFD.
    assert_eq!(txt(b"\x3d\xd8", Encoding::Utf16Le), "\u{FFFD}");
    assert!(!dec(b"\x41\x00", Encoding::Utf16Le).1);
    assert!(dec(b"\x00\xde", Encoding::Utf16Le).1);
}

#[test]
fn replacement_and_user_defined() {
    assert_eq!(txt(b"anything", Encoding::Replacement), "\u{FFFD}");
    assert_eq!(txt(b"", Encoding::Replacement), "");
    assert!(dec(b"x", Encoding::Replacement).1);
    assert!(!dec(b"", Encoding::Replacement).1);
    // x-user-defined: ASCII as-is, 0x80-0xFF -> U+F780+.
    assert_eq!(
        txt(b"\x41\x80\xff", Encoding::XUserDefined),
        "A\u{F780}\u{F7FF}"
    );
    assert!(!dec(b"\x80", Encoding::XUserDefined).1);
}

#[test]
fn iso_2022_jp_decode() {
    use bowser_encoding::encoding_from_label;
    let e = encoding_from_label("iso-2022-jp").expect("label resolves");
    assert_eq!(e, Encoding::Iso2022Jp);
    // JIS X 0208 escape: 0x24 0x22 is pointer 283 -> あ.
    assert_eq!(txt(b"\x1b$B\x24\x22", e), "あ");
    assert_eq!(txt("a\x1b$B\x24\x22\x1b(Bb".as_bytes(), e), "aあb");
    // Katakana and Roman escapes.
    assert_eq!(txt(b"\x1b(I\x21\x5f", e), "｡ﾟ");
    assert_eq!(txt(b"\x1b(J\x5c\x7e", e), "¥\u{203E}");
    // Repeated escape sequence -> error.
    assert_eq!(txt(b"\x1b$B\x1b$B", e), "\u{FFFD}");
    assert!(dec(b"\x1b$B\x1b$B", e).1);
    // Bad escape restores the byte.
    assert_eq!(txt(b"\x1bX", e), "\u{FFFD}X");
    // Pure ASCII is clean.
    assert!(!dec(b"hello", e).1);
}

#[test]
fn every_encoding_has_a_decoder() {
    // No encoding in the table may resolve to DecodeError::Unsupported:
    // per the spec, get-an-encoding must return a usable encoding.
    for e in Encoding::ALL {
        let r = decode_with(b"abc", e);
        assert!(r.is_ok(), "{} returned {:?}", e.name(), r);
        let r = decode(b"abc", e);
        assert!(r.is_ok(), "decode: {} returned {:?}", e.name(), r);
    }
}

#[test]
fn decode_applies_bom_and_output_rules() {
    // UTF-8 BOM overrides a lying fallback and is stripped.
    let r = decode(b"\xef\xbb\xbfhi", Encoding::Windows1252).unwrap();
    assert_eq!(r.encoding, Encoding::Utf8);
    assert_eq!(r.text, "hi");
    assert!(!r.had_errors);
    // UTF-16LE BOM.
    let r = decode(b"\xff\xfeA\x00", Encoding::Utf8).unwrap();
    assert_eq!(r.encoding, Encoding::Utf16Le);
    assert_eq!(r.text, "A");
    // UTF-16BE BOM.
    let r = decode(b"\xfe\xff\x00A", Encoding::Utf8).unwrap();
    assert_eq!(r.encoding, Encoding::Utf16Be);
    assert_eq!(r.text, "A");
    // No BOM: fallback used as-is (decode does not apply output encoding).
    let r = decode(b"\xe9", Encoding::Windows1252).unwrap();
    assert_eq!(r.encoding, Encoding::Windows1252);
    assert_eq!(r.text, "é");
    // output_encoding is the separate hook callers use for URL/form contexts.
    assert_eq!(output_encoding(Encoding::Replacement), Encoding::Utf8);
}
