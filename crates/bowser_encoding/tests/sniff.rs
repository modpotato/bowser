//! BOM sniffing, HTML prescan, and XML declaration fallback tests,
//! following the spec's "prescan a byte stream to determine its encoding".

use bowser_encoding::{Encoding, decode, prescan_encoding, sniff_bom, xml_encoding};

#[test]
fn bom_detection() {
    assert_eq!(sniff_bom(b"\xef\xbb\xbfabc"), Some(Encoding::Utf8));
    assert_eq!(sniff_bom(b"\xff\xfe\x41\x00"), Some(Encoding::Utf16Le));
    assert_eq!(sniff_bom(b"\xfe\xff\x00\x41"), Some(Encoding::Utf16Be));
    assert_eq!(sniff_bom(b"abc"), None);
    assert_eq!(sniff_bom(b""), None);
    assert_eq!(sniff_bom(b"\xef\xbb"), None);
    assert_eq!(sniff_bom(b"\xef\xbb\xbf"), Some(Encoding::Utf8));
}

#[test]
fn decode_strips_bom_and_overrides_fallback() {
    let r = decode(b"\xef\xbb\xbf<p>", Encoding::Windows1252).unwrap();
    assert_eq!(r.encoding, Encoding::Utf8);
    assert_eq!(r.text, "<p>");
    let r = decode(b"\xff\xfe<\x00", Encoding::Windows1252).unwrap();
    assert_eq!(r.encoding, Encoding::Utf16Le);
    assert_eq!(r.text, "<");
}

#[test]
fn prescan_meta_charset() {
    let cases: &[(&str, Encoding)] = &[
        (r#"<meta charset="utf-8">"#, Encoding::Utf8),
        (r#"<META CHARSET=UTF-8>"#, Encoding::Utf8),
        (r#"<meta charset='windows-1252'>"#, Encoding::Windows1252),
        (r#"<meta charset=big5>"#, Encoding::Big5),
        (r#"<meta charset = "gbk">"#, Encoding::Gbk),
        (
            "<meta\n\tcharset\n=\n\"iso-8859-7\"\n>",
            Encoding::Iso8859_7,
        ),
        (
            r#"<meta http-equiv="content-type" content="text/html; charset=euc-kr">"#,
            Encoding::EucKr,
        ),
        (
            r#"<meta content="text/html;charset=Shift_JIS" http-equiv=Content-Type>"#,
            Encoding::ShiftJis,
        ),
    ];
    for (html, expected) in cases {
        assert_eq!(prescan_encoding(html.as_bytes()), Some(*expected), "{html}");
    }
}

#[test]
fn prescan_applies_spec_substitutions() {
    // x-user-defined -> windows-1252; UTF-16 labels -> UTF-8.
    assert_eq!(
        prescan_encoding(br#"<meta charset="x-user-defined">"#),
        Some(Encoding::Windows1252)
    );
    assert_eq!(
        prescan_encoding(br#"<meta charset="utf-16le">"#),
        Some(Encoding::Utf8)
    );
    assert_eq!(
        prescan_encoding(br#"<meta charset="utf-16be">"#),
        Some(Encoding::Utf8)
    );
}

#[test]
fn prescan_skips_comments_and_non_meta() {
    // Meta inside a comment is invisible to the prescan.
    assert_eq!(
        prescan_encoding(br#"<!-- <meta charset="big5"> --><p>hi"#),
        None
    );
    // Bogus tags don't match; a later real meta does.
    assert_eq!(
        prescan_encoding(br#"<div charset="big5"><meta charset="gbk">"#),
        Some(Encoding::Gbk)
    );
    // `charset` on a non-meta element is ignored.
    assert_eq!(prescan_encoding(br#"<p charset="big5">"#,), None);
}

#[test]
fn prescan_first_meta_wins() {
    assert_eq!(
        prescan_encoding(br#"<meta charset="big5"><meta charset="gbk">"#),
        Some(Encoding::Big5)
    );
    // A meta with no usable charset doesn't block a later one.
    assert_eq!(
        prescan_encoding(br#"<meta name="x"><meta charset="gbk">"#),
        Some(Encoding::Gbk)
    );
    // Unknown charset label: skipped, scanning continues.
    assert_eq!(
        prescan_encoding(br#"<meta charset="bogus"><meta charset="gbk">"#),
        Some(Encoding::Gbk)
    );
}

#[test]
fn prescan_content_charset_extraction() {
    // content="...; charset=..." extraction rules.
    assert_eq!(
        prescan_encoding(
            br#"<meta http-equiv="content-type" content="text/html; charset = big5">"#
        ),
        Some(Encoding::Big5)
    );
    // No charset in content -> keep scanning.
    assert_eq!(
        prescan_encoding(br#"<meta http-equiv="content-type" content="text/html">"#),
        None
    );
    // http-equiv must be content-type.
    assert_eq!(
        prescan_encoding(br#"<meta http-equiv="refresh" content="0;url=x;charset=big5">"#),
        None
    );
    // Non-ASCII bytes around the charset break extraction.
    let mut html = br#"<meta http-equiv="content-type" content="text/html; charset="#.to_vec();
    html.extend_from_slice("big5".as_bytes());
    html.push(0xE9);
    html.extend_from_slice(br#"">"#);
    assert_eq!(prescan_encoding(&html), None);
}

#[test]
fn prescan_needs_pragma() {
    // <meta charset> works without http-equiv.
    assert_eq!(
        prescan_encoding(br#"<meta charset="big5">"#),
        Some(Encoding::Big5)
    );
    // <meta content="...charset=..."> without http-equiv is ignored...
    assert_eq!(
        prescan_encoding(br#"<meta content="text/html;charset=big5">"#),
        None
    );
    // ...even if a charset attribute is missing entirely.
    assert_eq!(
        prescan_encoding(br#"<meta name="description" content="x">"#),
        None
    );
}

#[test]
fn prescan_xml_fallback() {
    // Running out of bytes aborts the prescan -> XML declaration fallback.
    assert_eq!(
        prescan_encoding(br#"<?xml version="1.0" encoding="ISO-8859-2"?>"#),
        Some(Encoding::Iso8859_2)
    );
    // The fallback also fires after real content with no meta.
    assert_eq!(
        prescan_encoding(br#"<?xml version="1.0" encoding="gbk"?><p>hi"#),
        Some(Encoding::Gbk)
    );
    // Unparseable XML declaration -> None.
    assert_eq!(prescan_encoding(br#"<?xml version="1.0"?><p>hi"#), None);
}

#[test]
fn utf16_xml_declaration_sniff() {
    // `<\0?\0x\0m\0l\0` (UTF-16LE-encoded `<?xml`) -> UTF-16LE.
    let le: Vec<u8> = "<?xml"
        .encode_utf16()
        .flat_map(|u| u.to_le_bytes())
        .collect();
    assert_eq!(prescan_encoding(&le), Some(Encoding::Utf16Le));
    // UTF-16BE-encoded `<?xml` -> UTF-16BE.
    let be: Vec<u8> = "<?xml"
        .encode_utf16()
        .flat_map(|u| u.to_be_bytes())
        .collect();
    assert_eq!(prescan_encoding(&be), Some(Encoding::Utf16Be));
    // Incomplete prefix (only `<\0?`) -> None.
    assert_eq!(prescan_encoding(b"<\0?"), None);
}

#[test]
fn xml_encoding_direct() {
    assert_eq!(
        xml_encoding(br#"<?xml version="1.0" encoding="UTF-8"?>"#),
        Some(Encoding::Utf8)
    );
    assert_eq!(
        xml_encoding(br#"<?xml version='1.0' encoding='Shift_JIS'?>"#),
        Some(Encoding::ShiftJis)
    );
    // Spec substitutions apply: utf-16 -> utf-8.
    assert_eq!(
        xml_encoding(br#"<?xml encoding="utf-16"?>"#),
        Some(Encoding::Utf8)
    );
    assert_eq!(
        xml_encoding(br#"<?xml encoding="utf-16be"?>"#),
        Some(Encoding::Utf8)
    );
    // Unknown label -> None.
    assert_eq!(xml_encoding(br#"<?xml encoding="bogus"?>"#), None);
    // Not an XML declaration at all.
    assert_eq!(xml_encoding(b"<p>hi"), None);
    assert_eq!(xml_encoding(b""), None);
    // Unterminated declaration -> None.
    assert_eq!(xml_encoding(br#"<?xml encoding="utf-8""#), None);
}
