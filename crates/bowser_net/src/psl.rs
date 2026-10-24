//! Public Suffix List (PSL) lookups over a compact embedded trie.
//!
//! The list is vendored from <https://publicsuffix.org/list/public_suffix_list.dat>
//! (see [`list_version`]) and compressed at build time into
//! `psl_data.bin` by `psl/gen_psl.py`: a reversed-label (TLD-first) trie
//! where each node records whether an exact, wildcard (`*.`), or exception
//! (`!`) rule ends there. Lookup implements the prevailing-rule algorithm:
//! candidates are examined longest-first, and at each candidate an exception
//! rule beats a wildcard rule beats an exact rule; with no match the `*`
//! rule makes the last label the public suffix.
//!
//! All functions take plain host strings. Hosts are matched ASCII
//! case-insensitively and a single trailing dot is ignored. Unicode (IDN)
//! labels never match: the trie stores rules IDNA-encoded to ASCII, so
//! callers must pass the ASCII (punycode) form — UTS-46 processing belongs
//! to `bowser_url`, not here.

use std::borrow::Cow;

const DATA: &[u8] = include_bytes!("../tests/fixtures/psl_data.bin");

const MAGIC: &[u8; 8] = b"BPSLv001";

const FLAG_EXACT: u8 = 0b001;
const FLAG_WILDCARD: u8 = 0b010;
const FLAG_EXCEPTION: u8 = 0b100;

/// Maximum number of labels in a host we will walk. DNS names are at most
/// 253 octets, so 128 labels always suffices for valid input; longer
/// inputs fall back to the `*` rule.
const MAX_LABELS: usize = 128;

/// Strip one trailing dot and ASCII-lowercase a host for canonical use
/// (cookie domains, comparisons). Returns a borrowed slice when the
/// input is already canonical.
pub fn normalize_host(host: &str) -> Cow<'_, str> {
    let h = host.strip_suffix('.').unwrap_or(host);
    if h.bytes().any(|b| b.is_ascii_uppercase()) {
        Cow::Owned(h.to_ascii_lowercase())
    } else {
        Cow::Borrowed(h)
    }
}

/// The `(version, commit)` of the vendored list, e.g.
/// `("2026-09-24_13-26-36_UTC", "a179a48...")`.
pub fn list_version() -> (&'static str, &'static str) {
    let mut pos = 0;
    if DATA.get(0..8) != Some(MAGIC.as_slice()) {
        return ("", "");
    }
    pos += 8;
    let version = read_len_prefixed(pos);
    pos += 1 + version.len();
    let commit = read_len_prefixed(pos);
    (version, commit)
}

fn read_len_prefixed(pos: usize) -> &'static str {
    let len = DATA.get(pos).copied().unwrap_or(0) as usize;
    let bytes = DATA.get(pos + 1..pos + 1 + len).unwrap_or(&[]);
    std::str::from_utf8(bytes).unwrap_or("")
}

/// Byte offset of the trie root node, or `None` if the header is corrupt.
fn trie_root() -> Option<usize> {
    if DATA.get(0..8) != Some(MAGIC.as_slice()) {
        return None;
    }
    let mut pos = 8;
    for _ in 0..2 {
        let len = DATA.get(pos).copied().unwrap_or(0) as usize;
        pos = pos.checked_add(1 + len)?;
    }
    pos = pos.checked_add(4)?; // rule count
    DATA.get(pos..).is_some().then_some(pos)
}

/// Find the child of the node at `node_off` whose label equals `label`
/// (ASCII case-insensitive). Returns the child node's offset.
fn find_child(node_off: usize, label: &str) -> Option<usize> {
    let count = u16::from_le_bytes([*DATA.get(node_off + 1)?, *DATA.get(node_off + 2)?]);
    let mut pos = node_off + 3;
    let want = label.as_bytes();
    for _ in 0..count {
        let label_len = *DATA.get(pos)? as usize;
        pos += 1;
        let child_label = DATA.get(pos..pos + label_len)?;
        pos += label_len;
        let subtree_len = u32::from_le_bytes(DATA.get(pos..pos + 4)?.try_into().ok()?) as usize;
        pos += 4;
        if child_label.eq_ignore_ascii_case(want) {
            return Some(pos);
        }
        pos = pos.checked_add(subtree_len)?;
    }
    None
}

/// The public suffix of `host`, returned as a slice of `host` (case
/// preserved). Never returns an empty slice for non-empty input; for `""`
/// it returns `""`.
pub fn public_suffix(host: &str) -> &str {
    walk(host).unwrap_or_else(|| last_label(host))
}

/// The registrable domain (eTLD+1): one label above the public suffix, or
/// `None` when `host` is itself a public suffix (or empty).
pub fn registrable_domain(host: &str) -> Option<&str> {
    let h = host.strip_suffix('.').unwrap_or(host);
    let ps = public_suffix(host);
    if ps.len() >= h.len() {
        return None;
    }
    // `ps` is a strict suffix of `h` starting at a label boundary; the
    // registrable domain starts one label earlier.
    let rest = &h[..h.len() - ps.len() - 1];
    let start = rest.rfind('.').map_or(0, |i| i + 1);
    Some(&h[start..])
}

/// True when `host` is itself a public suffix.
pub fn is_public_suffix(host: &str) -> bool {
    let h = host.strip_suffix('.').unwrap_or(host);
    public_suffix(host).len() == h.len()
}

/// Core walk: longest candidate first; at each candidate, exception >
/// wildcard > exact. Returns the byte range of the public suffix within
/// the dot-stripped host, or `None` on corrupt data / absurd input.
fn walk(host: &str) -> Option<&str> {
    let root = trie_root()?;
    let h = host.strip_suffix('.').unwrap_or(host);

    // Label byte ranges, left to right.
    let mut ranges = [(0usize, 0usize); MAX_LABELS];
    let mut n = 0usize;
    let mut start = 0usize;
    for (i, b) in h.bytes().enumerate() {
        if b == b'.' {
            if n >= MAX_LABELS {
                return None;
            }
            ranges[n] = (start, i);
            n += 1;
            start = i + 1;
        }
    }
    if n >= MAX_LABELS {
        return None;
    }
    ranges[n] = (start, h.len());
    n += 1;

    // Walk from the TLD down, recording the node at each depth.
    // depths[k] = (node offset, d) with d = number of labels consumed.
    let mut depths = [(0usize, 0usize); MAX_LABELS];
    let mut ndepth = 0usize;
    let mut node_off = root;
    for d in 1..=n {
        let (s, e) = ranges[n - d];
        let Some(child) = find_child(node_off, &h[s..e]) else {
            break;
        };
        depths[ndepth] = (child, d);
        ndepth += 1;
        node_off = child;
    }

    // Longest candidate first.
    for &(off, d) in depths[..ndepth].iter().rev() {
        let flags = *DATA.get(off)?;
        if flags & FLAG_EXCEPTION != 0 {
            let (s, _) = ranges[n - d + 1];
            return Some(&h[s..]);
        }
        if flags & FLAG_WILDCARD != 0 && d < n {
            let (s, _) = ranges[n - d - 1];
            return Some(&h[s..]);
        }
        if flags & FLAG_EXACT != 0 {
            let (s, _) = ranges[n - d];
            return Some(&h[s..]);
        }
    }
    // The "*" rule: the public suffix is the last label.
    let (s, _) = ranges[n - 1];
    Some(&h[s..])
}

/// Last label of the dot-stripped host (the `*` rule fallback).
fn last_label(host: &str) -> &str {
    let h = host.strip_suffix('.').unwrap_or(host);
    match h.rfind('.') {
        Some(i) => &h[i + 1..],
        None => h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_recorded() {
        let (version, commit) = list_version();
        assert_eq!(version, "2026-09-24_13-26-36_UTC");
        assert_eq!(commit, "a179a48c465e818cfd8d626691cb317985da87fb");
    }

    #[test]
    fn basic_suffixes() {
        assert_eq!(public_suffix("com"), "com");
        assert_eq!(public_suffix("example.com"), "com");
        assert_eq!(public_suffix("www.example.com"), "com");
        assert_eq!(public_suffix("a.b.c.d.example.com"), "com");
        assert_eq!(public_suffix("co.uk"), "co.uk");
        assert_eq!(public_suffix("example.co.uk"), "co.uk");
        assert_eq!(public_suffix("www.example.co.uk"), "co.uk");
        assert_eq!(public_suffix("foo.bar.baz"), "baz"); // "*" rule
        assert_eq!(public_suffix("localhost"), "localhost"); // "*" rule
    }

    #[test]
    fn wildcard_rules() {
        // *.ck
        assert_eq!(public_suffix("ck"), "ck");
        assert_eq!(public_suffix("b.ck"), "b.ck");
        assert_eq!(public_suffix("a.b.ck"), "b.ck");
        // *.yokohama.jp (no exact yokohama.jp rule)
        assert_eq!(public_suffix("yokohama.jp"), "jp");
        assert_eq!(public_suffix("x.yokohama.jp"), "x.yokohama.jp");
        assert_eq!(public_suffix("a.x.yokohama.jp"), "x.yokohama.jp");
        // *.sch.uk
        assert_eq!(public_suffix("sch.uk"), "uk");
        assert_eq!(public_suffix("essex.sch.uk"), "essex.sch.uk");
    }

    #[test]
    fn exception_rules() {
        // !www.ck
        assert_eq!(public_suffix("www.ck"), "ck");
        assert_eq!(public_suffix("foo.www.ck"), "ck");
        // !city.kawasaki.jp with *.kawasaki.jp
        assert_eq!(public_suffix("kawasaki.jp"), "jp");
        assert_eq!(public_suffix("city.kawasaki.jp"), "kawasaki.jp");
        assert_eq!(public_suffix("foo.city.kawasaki.jp"), "kawasaki.jp");
        assert_eq!(public_suffix("o.city.kawasaki.jp"), "kawasaki.jp");
        assert_eq!(public_suffix("other.kawasaki.jp"), "other.kawasaki.jp");
    }

    #[test]
    fn private_domains() {
        // appspot.com is an exact rule in the private section.
        assert_eq!(public_suffix("appspot.com"), "appspot.com");
        assert_eq!(public_suffix("foo.appspot.com"), "appspot.com");
    }

    #[test]
    fn idn_rules_match_punycode() {
        // 公司.cn is stored IDNA-encoded as xn--55qx5d.cn.
        assert_eq!(public_suffix("xn--55qx5d.cn"), "xn--55qx5d.cn");
        assert_eq!(public_suffix("www.xn--55qx5d.cn"), "xn--55qx5d.cn");
    }

    #[test]
    fn case_and_trailing_dot() {
        assert_eq!(public_suffix("EXAMPLE.COM"), "COM");
        assert_eq!(public_suffix("example.COM."), "COM");
        assert_eq!(public_suffix("WWW.Example.Co.UK"), "Co.UK");
        assert_eq!(public_suffix("B.CK"), "B.CK");
    }

    #[test]
    fn registrable_domains() {
        assert_eq!(registrable_domain("www.example.com"), Some("example.com"));
        assert_eq!(registrable_domain("example.com"), Some("example.com"));
        assert_eq!(registrable_domain("com"), None);
        assert_eq!(registrable_domain("example.co.uk"), Some("example.co.uk"));
        assert_eq!(
            registrable_domain("www.example.co.uk"),
            Some("example.co.uk")
        );
        assert_eq!(registrable_domain("co.uk"), None);
        assert_eq!(registrable_domain("b.ck"), None);
        assert_eq!(registrable_domain("a.b.ck"), Some("a.b.ck"));
        assert_eq!(registrable_domain("www.ck"), Some("www.ck"));
        assert_eq!(registrable_domain("foo.bar"), Some("foo.bar"));
        assert_eq!(registrable_domain(""), None);
    }

    #[test]
    fn public_suffix_predicate() {
        assert!(is_public_suffix("com"));
        assert!(is_public_suffix("co.uk"));
        assert!(is_public_suffix("b.ck"));
        assert!(is_public_suffix("ck"));
        // *.yokohama.jp makes subdomains public suffixes, but yokohama.jp
        // itself remains registrable.
        assert!(!is_public_suffix("yokohama.jp"));
        assert!(!is_public_suffix("example.com"));
        assert!(!is_public_suffix("www.example.co.uk"));
        assert!(!is_public_suffix("www.ck"));
        assert!(!is_public_suffix("city.kawasaki.jp"));
    }

    #[test]
    fn normalize() {
        assert_eq!(normalize_host("Example.COM."), "example.com");
        assert_eq!(normalize_host("example.com"), "example.com");
        assert_eq!(normalize_host("EXAMPLE.COM"), "example.com");
    }

    #[test]
    fn degenerate_inputs() {
        assert_eq!(public_suffix(""), "");
        assert_eq!(public_suffix("."), "");
        assert!(is_public_suffix(""));
        assert_eq!(registrable_domain("."), None);
        // Absurd label counts fall back to the "*" rule, never panic.
        let deep: String = (0..300).map(|_| "a.").collect::<String>() + "com";
        assert_eq!(public_suffix(&deep), "com");
    }
}
