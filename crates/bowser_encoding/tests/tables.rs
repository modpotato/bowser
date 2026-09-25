//! Exhaustive cross-checks against the vendored WHATWG fixtures.
//!
//! `tests/fixtures/index-*.txt` are verbatim copies of the spec's index
//! files; `encodings.json` is the spec's label table. Every entry is decoded
//! and compared, so the generated tables in `data.rs` cannot drift.

use bowser_encoding::{Encoding, INDEX_VERSIONS, decode_with, encoding_from_label};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Parse an `index-*.txt` fixture into (pointer, code point) entries.
fn parse_index(name: &str) -> Vec<(u32, u32)> {
    let text = fs::read_to_string(fixture_dir().join(name)).expect("fixture readable");
    // The vendored fixture must be the exact bytes the table was built from.
    let ident_line = text
        .lines()
        .find(|l| l.starts_with("# Identifier: "))
        .expect("identifier line");
    let date_line = text
        .lines()
        .find(|l| l.starts_with("# Date: "))
        .expect("date line");
    let stem = name
        .strip_prefix("index-")
        .unwrap()
        .strip_suffix(".txt")
        .unwrap();
    let (vfile, ident, date) = INDEX_VERSIONS
        .iter()
        .find(|(f, _, _)| f == &format!("index-{stem}.txt"))
        .expect("index version recorded");
    assert_eq!(vfile, &format!("index-{stem}.txt"));
    assert_eq!(ident_line, &format!("# Identifier: {ident}"));
    assert_eq!(date_line, &format!("# Date: {date}"));
    text.lines()
        .filter(|l| {
            let t = l.trim_start();
            !t.is_empty() && !t.starts_with('#') && t.as_bytes()[0].is_ascii_digit()
        })
        .map(|l| {
            let mut parts = l.split_whitespace();
            let ptr: u32 = parts.next().unwrap().parse().unwrap();
            let cp =
                u32::from_str_radix(parts.next().unwrap().trim_start_matches("0x"), 16).unwrap();
            (ptr, cp)
        })
        .collect()
}

fn find_from(hay: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    hay[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| from + p)
}

/// Parse `encodings.json` into `(name, labels)` per encoding object.
fn parse_encodings_json(text: &str) -> Vec<(String, Vec<String>)> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(li) = find_from(b, i, b"\"labels\"") {
        let mut j = li + 8;
        while b[j] != b'[' {
            j += 1;
        }
        j += 1;
        let mut labels = Vec::new();
        loop {
            while b[j] != b'"' && b[j] != b']' {
                j += 1;
            }
            if b[j] == b']' {
                break;
            }
            j += 1;
            let start = j;
            while b[j] != b'"' {
                j += 1;
            }
            labels.push(String::from_utf8(b[start..j].to_vec()).unwrap());
            j += 1;
        }
        let ni = find_from(b, j, b"\"name\"").expect("name after labels");
        let mut k = ni + 6;
        while b[k] != b'"' {
            k += 1;
        }
        k += 1;
        let start = k;
        while b[k] != b'"' {
            k += 1;
        }
        out.push((String::from_utf8(b[start..k].to_vec()).unwrap(), labels));
        i = k;
    }
    out
}

#[test]
fn all_228_labels_resolve_to_the_right_encoding() {
    let text = fs::read_to_string(fixture_dir().join("encodings.json")).unwrap();
    let encodings = parse_encodings_json(&text);
    assert_eq!(encodings.len(), 40, "one object per encoding");
    let mut seen = HashMap::new();
    let mut total = 0;
    for (name, labels) in &encodings {
        assert!(!labels.is_empty(), "{name} has no labels");
        for label in labels {
            let enc = encoding_from_label(label)
                .unwrap_or_else(|| panic!("label {label:?} should resolve"));
            assert_eq!(
                enc.name(),
                name.to_lowercase(),
                "label {label:?} -> {:?}, expected {name}",
                enc
            );
            assert!(
                seen.insert(label.clone(), enc).is_none(),
                "dup label {label}"
            );
            total += 1;
        }
    }
    assert_eq!(total, 228, "expected 228 labels");
    // The spec's canonical names are exactly the lowercase json names.
    let mut ours: Vec<&str> = Encoding::ALL.iter().map(|e| e.name()).collect();
    ours.sort_unstable();
    let mut theirs: Vec<String> = encodings.iter().map(|(n, _)| n.to_lowercase()).collect();
    theirs.sort_unstable();
    assert_eq!(ours, theirs);
}

#[test]
fn unknown_labels_do_not_resolve() {
    for label in ["", "utf", "latin1x", "bogus", "utf-88", "windows1252x", " "] {
        assert_eq!(encoding_from_label(label), None, "{label:?}");
    }
    // ...but surrounding whitespace is trimmed per spec, so these DO resolve.
    assert_eq!(encoding_from_label("utf-8 "), Some(Encoding::Utf8));
    assert_eq!(
        encoding_from_label(" csiso2022jp"),
        Some(Encoding::Iso2022Jp)
    );
}

#[test]
fn label_lookup_is_case_and_space_insensitive() {
    assert_eq!(encoding_from_label("UTF-8"), Some(Encoding::Utf8));
    assert_eq!(encoding_from_label("  Utf-8\t"), Some(Encoding::Utf8));
    assert_eq!(
        encoding_from_label("WINDOWS-1252"),
        Some(Encoding::Windows1252)
    );
    assert_eq!(encoding_from_label("Shift_JIS"), Some(Encoding::ShiftJis));
    assert_eq!(encoding_from_label("gbk"), Some(Encoding::Gbk));
    assert_eq!(encoding_from_label("big5"), Some(Encoding::Big5));
}

/// All 27 single-byte index fixtures: every entry decodes to its code point.
#[test]
fn single_byte_tables_match_fixtures() {
    let stems = [
        "ibm866",
        "iso-8859-2",
        "iso-8859-3",
        "iso-8859-4",
        "iso-8859-5",
        "iso-8859-6",
        "iso-8859-7",
        "iso-8859-8",
        "iso-8859-10",
        "iso-8859-13",
        "iso-8859-14",
        "iso-8859-15",
        "iso-8859-16",
        "koi8-r",
        "koi8-u",
        "macintosh",
        "windows-874",
        "windows-1250",
        "windows-1251",
        "windows-1252",
        "windows-1253",
        "windows-1254",
        "windows-1255",
        "windows-1256",
        "windows-1257",
        "windows-1258",
        "x-mac-cyrillic",
    ];
    let mut total = 0;
    for stem in stems {
        let enc = encoding_from_label(stem).unwrap();
        let entries = parse_index(&format!("index-{stem}.txt"));
        // Some indexes are sparse: unmapped bytes are simply absent.
        let map: HashMap<u32, u32> = entries.into_iter().collect();
        for i in 0..128u32 {
            let (s, err) = decode_with(&[0x80 + i as u8], enc).unwrap();
            match map.get(&i) {
                Some(cp) => {
                    assert_eq!(
                        s,
                        char::from_u32(*cp).unwrap().to_string(),
                        "{stem} byte {:#x}",
                        0x80 + i
                    );
                    assert!(!err, "{stem} byte {:#x} flagged error", 0x80 + i);
                }
                None => {
                    assert_eq!(
                        s,
                        "\u{FFFD}",
                        "{stem} byte {:#x} should be unmapped",
                        0x80 + i
                    );
                    assert!(err, "{stem} byte {:#x} should flag error", 0x80 + i);
                }
            }
            total += 1;
        }
    }
    assert_eq!(total, 27 * 128);
    // iso-8859-8-i shares iso-8859-8's table per the standard.
    let a = decode_with(b"\xe0", Encoding::Iso8859_8).unwrap();
    let b = decode_with(b"\xe0", Encoding::Iso8859_8I).unwrap();
    assert_eq!(a, b);
}

/// Every Big5 fixture entry round-trips through the decoder.
#[test]
fn big5_table_matches_fixture() {
    let entries = parse_index("index-big5.txt");
    assert_eq!(entries.len(), 18590);
    let mut prev = None;
    for (ptr, cp) in &entries {
        if let Some(p) = prev {
            assert!(*ptr > p, "not sorted");
        }
        prev = Some(*ptr);
        let lead = 0x81 + (ptr / 157) as u8;
        let off = ptr % 157;
        let trail = if off < 63 {
            0x40 + off as u8
        } else {
            0x62 + off as u8
        };
        let (s, err) = decode_with(&[lead, trail], Encoding::Big5).unwrap();
        assert_eq!(
            s,
            char::from_u32(*cp).unwrap().to_string(),
            "big5 pointer {ptr}"
        );
        assert!(!err, "big5 pointer {ptr} flagged error");
    }
}

/// Every gb18030 two-byte fixture entry round-trips through the decoder.
#[test]
fn gb18030_two_byte_table_matches_fixture() {
    let entries = parse_index("index-gb18030.txt");
    assert_eq!(entries.len(), 23940);
    for (i, (ptr, cp)) in entries.iter().enumerate() {
        assert_eq!(*ptr as usize, i);
        let lead = 0x81 + (ptr / 190) as u8;
        let r = ptr % 190;
        let trail = if r < 63 {
            0x40 + r as u8
        } else {
            0x41 + r as u8
        };
        let (s, err) = decode_with(&[lead, trail], Encoding::Gb18030).unwrap();
        assert_eq!(
            s,
            char::from_u32(*cp).unwrap().to_string(),
            "gb18030 pointer {ptr}"
        );
        assert!(!err, "gb18030 pointer {ptr} flagged error");
    }
}

fn four_bytes(mut p: u32) -> [u8; 4] {
    let b = 0x30 + (p % 10) as u8;
    p /= 10;
    let t = 0x81 + (p % 126) as u8;
    p /= 126;
    let s = 0x30 + (p % 10) as u8;
    p /= 10;
    let f = 0x81 + p as u8;
    [f, s, t, b]
}

/// gb18030 ranges: every entry start decodes, interpolation works, dead
/// zones error, and the max pointer is honored.
#[test]
fn gb18030_ranges_match_fixture() {
    let entries: Vec<(u32, u32)> = parse_index("index-gb18030-ranges.txt");
    assert_eq!(entries.len(), 207);
    // Every entry start.
    for (ptr, cp) in &entries {
        let (s, err) = decode_with(&four_bytes(*ptr), Encoding::Gb18030).unwrap();
        assert_eq!(
            s,
            char::from_u32(*cp).unwrap().to_string(),
            "range pointer {ptr}"
        );
        assert!(!err);
    }
    // Interpolation between entries: 0 -> 0x0080, 36 -> 0x00A5, so 10 -> 0x008A.
    assert_eq!(
        decode_with(&four_bytes(10), Encoding::Gb18030).unwrap().0,
        "\u{8A}"
    );
    // Dead zone 39420..189000 -> error.
    for p in [39420, 50000, 100000, 188999] {
        let (s, err) = decode_with(&four_bytes(p), Encoding::Gb18030).unwrap();
        assert_eq!(s, "\u{FFFD}", "dead zone {p}");
        assert!(err, "dead zone {p} must flag error");
    }
    // 189000 -> U+10000 (the out-of-order tail entry).
    assert_eq!(
        decode_with(&four_bytes(189000), Encoding::Gb18030)
            .unwrap()
            .0,
        "\u{10000}"
    );
    // Above 1237575 -> error. 0xFE 0x39 0xFE 0x39 is pointer 1587599.
    let (s, err) = decode_with(&[0xFE, 0x39, 0xFE, 0x39], Encoding::Gb18030).unwrap();
    assert_eq!(s, "\u{FFFD}");
    assert!(err);
    // 1237575 itself is a valid pointer.
    let (s, err) = decode_with(&four_bytes(1237575), Encoding::Gb18030).unwrap();
    assert!(!err);
    assert_ne!(s, "\u{FFFD}");
}
