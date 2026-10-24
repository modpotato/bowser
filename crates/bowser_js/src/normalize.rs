//! Unicode normalization (UAX #15): NFC, NFD, NFKC, NFKD.
//!
//! The tables live in the generated sibling module [`tables`]
//! (`normalize_tables.rs`, vendored from Unicode 16.0.0 UCD). Hangul
//! syllables (U+AC00..U+D7A3) are handled algorithmically per UAX #15 and
//! have no table entries, exactly as in the UCD.
//!
//! Lone surrogates (0xD800..0xDFFF) surface as `u32` code points and pass
//! through untouched: no mapping mentions them and their combining class
//! is 0, which matches the spec's code-point treatment of ill-formed
//! UTF-16.

#[path = "normalize_tables.rs"]
mod tables;

/// A Unicode normalization form (the `form` argument of
/// `String.prototype.normalize`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Form {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
}

impl Form {
    /// Parse the `ToString`'d form argument. Anything else is a spec
    /// `RangeError`.
    pub(crate) fn from_name(name: &[u8]) -> Option<Form> {
        match name {
            b"NFC" => Some(Form::Nfc),
            b"NFD" => Some(Form::Nfd),
            b"NFKC" => Some(Form::Nfkc),
            b"NFKD" => Some(Form::Nfkd),
            _ => None,
        }
    }

    fn is_compat(self) -> bool {
        matches!(self, Form::Nfkc | Form::Nfkd)
    }

    fn is_composing(self) -> bool {
        matches!(self, Form::Nfc | Form::Nfkc)
    }
}

// Hangul algorithmic constants (UAX #15, section "Hangul Syllable
// Boundary Determination" / composition).
const SBASE: u32 = 0xAC00;
const LBASE: u32 = 0x1100;
const VBASE: u32 = 0x1161;
const TBASE: u32 = 0x11A7;
const LCOUNT: u32 = 19;
const VCOUNT: u32 = 21;
const TCOUNT: u32 = 28;
const NCOUNT: u32 = VCOUNT * TCOUNT; // 588
const SCOUNT: u32 = LCOUNT * NCOUNT; // 11172

/// Normalize a code-point sequence (`u32`; lone surrogates pass through)
/// into the given form, per UAX #15: full decomposition, canonical
/// ordering, then composition for the composed forms.
pub(crate) fn normalize_cps(input: &[u32], form: Form) -> Vec<u32> {
    let compat = form.is_compat();
    let mut buf = Vec::with_capacity(input.len());
    for &cp in input {
        decompose_into(cp, compat, &mut buf);
    }
    canonical_order(&mut buf);
    if form.is_composing() {
        compose_in_place(&mut buf);
    }
    buf
}

/// Recursive (full) decomposition of one code point into `out`.
fn decompose_into(cp: u32, compat: bool, out: &mut Vec<u32>) {
    // Hangul syllable: algorithmic decomposition to L, V, (T).
    if (SBASE..SBASE + SCOUNT).contains(&cp) {
        let s_index = cp - SBASE;
        out.push(LBASE + s_index / NCOUNT);
        out.push(VBASE + (s_index % NCOUNT) / TCOUNT);
        let t = TBASE + s_index % TCOUNT;
        if t != TBASE {
            out.push(t);
        }
        return;
    }
    if let Ok(i) = tables::DECOMP_INDEX.binary_search_by_key(&cp, |&(k, _)| k) {
        let packed = tables::DECOMP_INDEX[i].1;
        // Low bit marks a compatibility mapping: skipped unless the form
        // is NFKC/NFKD.
        if compat || packed & 1 == 0 {
            let mut p = (packed >> 1) as usize;
            let len = tables::DECOMP_DATA[p] as usize;
            p += 1;
            // Full (recursive) decomposition: mappings can target
            // characters that decompose further (e.g. U+1F82 → U+1F00 U+0342).
            for j in 0..len {
                decompose_into(tables::DECOMP_DATA[p + j], compat, out);
            }
            return;
        }
    }
    out.push(cp);
}

/// Canonical combining class; 0 for everything without a table entry
/// (including Hangul jamo and lone surrogates).
fn ccc(cp: u32) -> u8 {
    let ranges = tables::CCC_RANGES;
    let mut lo = 0usize;
    let mut hi = ranges.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (s, e, c) = ranges[mid];
        if cp < s {
            hi = mid;
        } else if cp > e {
            lo = mid + 1;
        } else {
            return c;
        }
    }
    0
}

/// Canonical ordering (UAX #15): stable-sort each maximal run of combining
/// marks by combining class.
fn canonical_order(buf: &mut [u32]) {
    let n = buf.len();
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && ccc(buf[j]) != 0 {
            j += 1;
        }
        if j - i > 1 {
            // `sort_by_key` is stable, as the ordering requires.
            buf[i..j].sort_by_key(|&c| ccc(c));
        }
        // Skip the starter at `j` (or run off the end).
        i = j + 1;
    }
}

/// Primary-composite lookup: table first, Hangul algorithmic composition
/// per UAX #15 (`L + V -> LV`, `LV + T -> LVT`).
fn compose_pair(a: u32, b: u32) -> Option<u32> {
    if (LBASE..LBASE + LCOUNT).contains(&a) && (VBASE..VBASE + VCOUNT).contains(&b) {
        return Some(SBASE + ((a - LBASE) * VCOUNT + (b - VBASE)) * TCOUNT);
    }
    if (SBASE..SBASE + SCOUNT).contains(&a)
        && (TBASE + 1..TBASE + TCOUNT).contains(&b)
        && (a - SBASE).is_multiple_of(TCOUNT)
    {
        return Some(a + (b - TBASE));
    }
    let key = ((a as u64) << 32) | b as u64;
    tables::COMPOSE_PAIRS
        .binary_search_by_key(&key, |&(k, _)| k)
        .ok()
        .map(|i| tables::COMPOSE_PAIRS[i].1)
}

/// Composition (UAX #15): walk the decomposed buffer, composing each
/// character with the last starter unless blocked by an intervening
/// character of equal or greater combining class.
fn compose_in_place(buf: &mut Vec<u32>) {
    let mut out: Vec<u32> = Vec::with_capacity(buf.len());
    for &c in buf.iter() {
        if let Some(lpos) = out.iter().rposition(|&x| ccc(x) == 0) {
            let blocked = out[lpos + 1..].iter().any(|&b| ccc(b) >= ccc(c));
            if !blocked && let Some(p) = compose_pair(out[lpos], c) {
                out[lpos] = p;
                continue;
            }
        }
        out.push(c);
    }
    *buf = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cps(s: &str) -> Vec<u32> {
        s.chars().map(|c| c as u32).collect()
    }

    fn to_string(cps: &[u32]) -> String {
        cps.iter()
            .map(|&c| char::from_u32(c).unwrap_or('\u{FFFD}'))
            .collect()
    }

    fn norm(s: &str, form: Form) -> String {
        to_string(&normalize_cps(&cps(s), form))
    }

    #[test]
    fn form_names() {
        assert_eq!(Form::from_name(b"NFC"), Some(Form::Nfc));
        assert_eq!(Form::from_name(b"NFD"), Some(Form::Nfd));
        assert_eq!(Form::from_name(b"NFKC"), Some(Form::Nfkc));
        assert_eq!(Form::from_name(b"NFKD"), Some(Form::Nfkd));
        assert_eq!(Form::from_name(b"nfc"), None);
        assert_eq!(Form::from_name(b""), None);
        assert_eq!(Form::from_name(b"NFC "), None);
    }

    #[test]
    fn basic_latin() {
        // U+00E9 precomposed vs decomposed.
        assert_eq!(norm("é", Form::Nfc), "é");
        assert_eq!(norm("é", Form::Nfd), "e\u{301}");
        assert_eq!(norm("e\u{301}", Form::Nfc), "é");
        assert_eq!(norm("e\u{301}", Form::Nfd), "e\u{301}");
        // Canonical ordering: U+0301 (ccc 230) sorts before U+0327 (ccc 202)? No:
        // 202 < 230, so cedilla comes first.
        assert_eq!(norm("e\u{301}\u{327}", Form::Nfd), "e\u{327}\u{301}");
        // After canonical ordering (0327 before 0301), e + 0327 composes to
        // U+0229; no composite exists for U+0229 + 0301.
        assert_eq!(norm("e\u{301}\u{327}", Form::Nfc), "\u{229}\u{301}");
    }

    #[test]
    fn compatibility() {
        // U+00A0 NO-BREAK SPACE -> U+0020 under NFKC/NFKD only.
        assert_eq!(norm(" ", Form::Nfc), " ");
        assert_eq!(norm(" ", Form::Nfkc), " ");
        assert_eq!(norm(" ", Form::Nfkd), " ");
        // U+2126 OHM SIGN: canonical singleton -> U+03A9 in all forms.
        assert_eq!(norm("Ω", Form::Nfc), "Ω");
        assert_eq!(norm("Ω", Form::Nfd), "Ω");
        // U+FB01 LATIN SMALL LIGATURE FI -> "fi" under NFKC/NFKD only.
        assert_eq!(norm("ﬁ", Form::Nfc), "ﬁ");
        assert_eq!(norm("ﬁ", Form::Nfkc), "fi");
    }

    #[test]
    fn hangul() {
        // U+AC01 (각) <-> U+1100 U+1161 U+11A8
        assert_eq!(norm("각", Form::Nfd), "각");
        assert_eq!(norm("각", Form::Nfc), "각");
        assert_eq!(norm("가", Form::Nfd), "가");
        assert_eq!(norm("가", Form::Nfc), "가");
        assert_eq!(norm("각", Form::Nfkc), "각");
        // Jamo already in canonical order compose left to right.
        assert_eq!(norm("가", Form::Nfd), "가");
    }

    #[test]
    fn singleton_and_blocked() {
        // U+0344 COMBINING GREEK DIALYTIKA TONOS decomposes canonically to
        // U+0308 U+0301, but is composition-excluded: NFC must not recompose it.
        assert_eq!(norm("̈́", Form::Nfd), "̈́");
        assert_eq!(norm("̈́", Form::Nfc), "̈́");
        assert_eq!(norm("̈́", Form::Nfc), "̈́");
        // Blocked composition: U+0061 U+0308 U+0301 — the 0301 cannot
        // compose with the starter across 0308 (same class 230).
        let s = "a\u{308}\u{301}";
        assert_eq!(norm(s, Form::Nfc), "ä\u{301}");
    }

    #[test]
    fn lone_surrogates_pass_through() {
        // Lone surrogates are not valid chars; they survive normalization.
        let input = [0xD800u32, 0x41, 0xDC00, 0xE9];
        for form in [Form::Nfc, Form::Nfd, Form::Nfkc, Form::Nfkd] {
            let out = normalize_cps(&input, form);
            assert_eq!(&out[0..3], &[0xD800, 0x41, 0xDC00]);
            match form {
                Form::Nfc | Form::Nfkc => assert_eq!(out[3], 0xE9),
                Form::Nfd | Form::Nfkd => assert_eq!(&out[3..], &[0x65, 0x301]),
            }
        }
    }

    /// The official Unicode NormalizationTest.txt vectors: for every line,
    /// `NFC(source) == NFC`, `NFD(source) == NFD`, etc. across all four forms.
    #[test]
    fn normalization_test_txt() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/NormalizationTest.txt"
        );
        let text = std::fs::read_to_string(path).expect("NormalizationTest.txt fixture");
        fn parse_seq(s: &str) -> Vec<u32> {
            s.split_whitespace()
                .map(|h| u32::from_str_radix(h, 16).unwrap())
                .collect()
        }
        let forms = [
            (Form::Nfc, 1usize),
            (Form::Nfd, 2usize),
            (Form::Nfkc, 3usize),
            (Form::Nfkd, 4usize),
        ];
        let mut cases = 0u32;
        let mut failures = 0u32;
        for line in text.lines() {
            // Strip the trailing "# ..." comment; fields are hex only.
            let line = match line.find('#') {
                Some(i) => &line[..i],
                None => line,
            }
            .trim();
            if line.is_empty() || line.starts_with('@') {
                continue;
            }
            let cols: Vec<&str> = line
                .split(';')
                .map(str::trim)
                .filter(|c| !c.is_empty())
                .collect();
            assert_eq!(cols.len(), 5, "bad line: {line}");
            let src = parse_seq(cols[0]);
            let expected: Vec<Vec<u32>> = cols[1..].iter().map(|c| parse_seq(c)).collect();
            for &(form, idx) in &forms {
                cases += 1;
                let got = normalize_cps(&src, form);
                if got != expected[idx - 1] {
                    failures += 1;
                    if failures <= 5 {
                        eprintln!(
                            "FAIL {form:?} src={} got={:X?} want={:X?}",
                            cols[0],
                            got,
                            expected[idx - 1]
                        );
                    }
                }
            }
        }
        eprintln!("NormalizationTest: {cases} checks, {failures} failures");
        assert!(cases > 70000, "too few cases parsed: {cases}");
        assert_eq!(failures, 0, "{failures} NormalizationTest failures");
    }
}
