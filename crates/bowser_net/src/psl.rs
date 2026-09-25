//! Public Suffix List (PSL) lookups over a compact embedded trie.
//!
//! The list is vendored from <https://publicsuffix.org/list/public_suffix_list.dat>
//! (see [`list_version`]) and compressed at build time into
//! `psl_data.bin` by `psl/gen_psl.py`: a reversed-label (TLD-first) trie
//! where each node records whether an exact, wildcard (`*.`), or exception
//! (`!`) rule ends there. Lookup implements the prevailing-rule algorithm:
//! first any exception rule among the candidates prevails (longest wins),
//! otherwise candidates are examined longest-first with wildcard beating
//! exact; with no match the `*` rule makes the last label the public suffix.
//!
//! All functions take plain host strings. Hosts are matched ASCII
//! case-insensitively and trailing dots (FQDN root markers) are ignored.
//! Unicode (IDN) labels never match: the trie stores rules IDNA-encoded to
//! ASCII, so callers must pass the ASCII (punycode) form — UTS-46
//! processing belongs to `bowser_url`, not here.

use std::borrow::Cow;

const DATA: &[u8] = include_bytes!("psl_data.bin");

const MAGIC: &[u8; 8] = b"BPSLv001";

const FLAG_EXACT: u8 = 0b001;
const FLAG_WILDCARD: u8 = 0b010;
const FLAG_EXCEPTION: u8 = 0b100;

/// Maximum number of labels in a host we will walk. DNS names are at most
/// 253 octets, so 128 labels always suffices for valid input; longer
/// inputs fall back to the `*` rule.
const MAX_LABELS: usize = 128;

/// Strip all trailing dots (FQDN root markers) from a host. `""` stays `""`.
fn strip_trailing_dots(host: &str) -> &str {
    host.trim_end_matches('.')
}

/// Strip ASCII whitespace and all trailing dots, then ASCII-lowercase a
/// host for canonical use (cookie domains, comparisons). Returns a
/// borrowed slice when the input is already canonical.
///
/// Callers should pre-trim their input; the trimming here is defensive so
/// a stray space or an extra root dot cannot poison label splitting.
pub fn normalize_host(host: &str) -> Cow<'_, str> {
    let h = strip_trailing_dots(host.trim_matches(|c: char| c.is_ascii_whitespace()));
    if h.bytes().any(|b| b.is_ascii_uppercase()) {
        Cow::Owned(h.to_ascii_lowercase())
    } else {
        Cow::Borrowed(h)
    }
}

/// The `(version, commit)` of the vendored list, e.g.
/// `("2026-09-24_13-26-36_UTC", "a179a48...")`.
// Introspection API: no in-tree callers yet, but the vendored-data contract
// includes version reporting (psl's own tests cover it), so it is kept
// rather than deleted now that the module is `pub(crate)`.
#[allow(dead_code)]
pub fn list_version() -> (&'static str, &'static str) {
    let mut pos = 0;
    if DATA.get(0..8) != Some(MAGIC.as_slice()) {
        return ("", "");
    }
    pos += 8;
    let version = read_len_prefixed(DATA, pos);
    pos += 1 + version.len();
    let commit = read_len_prefixed(DATA, pos);
    (version, commit)
}

#[allow(dead_code)] // only used by `list_version` (kept: see above)
fn read_len_prefixed(data: &[u8], pos: usize) -> &str {
    let len = data.get(pos).copied().unwrap_or(0) as usize;
    let bytes = data.get(pos + 1..pos + 1 + len).unwrap_or(&[]);
    std::str::from_utf8(bytes).unwrap_or("")
}

/// Byte slice starting at the trie root node, or `None` if the header is
/// corrupt. Taking the data as a parameter (rather than using `DATA`
/// directly) lets tests walk synthetic tries.
fn trie(data: &[u8]) -> Option<&[u8]> {
    if data.get(0..8) != Some(MAGIC.as_slice()) {
        return None;
    }
    let mut pos = 8;
    for _ in 0..2 {
        let len = data.get(pos).copied().unwrap_or(0) as usize;
        pos = pos.checked_add(1 + len)?;
    }
    pos = pos.checked_add(4)?; // rule count
    data.get(pos..)
}

/// Find the child of the node at `node_off` whose label equals `label`
/// (ASCII case-insensitive). Returns the child node's offset, relative to
/// the start of `trie`.
fn find_child(trie: &[u8], node_off: usize, label: &str) -> Option<usize> {
    let count = u16::from_le_bytes([*trie.get(node_off + 1)?, *trie.get(node_off + 2)?]);
    let mut pos = node_off + 3;
    let want = label.as_bytes();
    for _ in 0..count {
        let label_len = *trie.get(pos)? as usize;
        pos += 1;
        let child_label = trie.get(pos..pos + label_len)?;
        pos += label_len;
        let subtree_len = u32::from_le_bytes(trie.get(pos..pos + 4)?.try_into().ok()?) as usize;
        pos += 4;
        if child_label.eq_ignore_ascii_case(want) {
            return Some(pos);
        }
        pos = pos.checked_add(subtree_len)?;
    }
    None
}

/// The public suffix of `host`, returned as a slice of `host` (case
/// preserved). Never returns an empty slice for non-empty input, except
/// degenerate all-dots input (`"..."`), which is treated as empty; for
/// `""` it returns `""`.
pub fn public_suffix(host: &str) -> &str {
    walk(DATA, host).unwrap_or_else(|| last_label(host))
}

/// The registrable domain (eTLD+1): one label above the public suffix, or
/// `None` when `host` is itself a public suffix (or empty).
pub fn registrable_domain(host: &str) -> Option<&str> {
    let h = strip_trailing_dots(host);
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
    let h = strip_trailing_dots(host);
    public_suffix(host).len() == h.len()
}

/// Core walk over the embedded data. Returns the public suffix as a slice
/// of the dot-stripped host, or `None` on corrupt data / absurd input.
fn walk<'a>(data: &[u8], host: &'a str) -> Option<&'a str> {
    walk_trie(trie(data)?, host)
}

/// Core walk over a trie slice (the bytes starting at the root node).
/// Two passes over the candidates (the host and its parent domains,
/// TLD-first):
///
/// 1. exception pass: per the PSL spec an exception rule at *any* candidate
///    depth prevails over longer wildcard/exact matches, so scan all
///    depths first (longest wins); the public suffix is one label beyond
///    the exception rule.
/// 2. longest-first pass: wildcard beats exact at each candidate.
///
/// Returns the byte range of the public suffix within the dot-stripped
/// host, or `None` on corrupt data / absurd input.
fn walk_trie<'a>(trie: &[u8], host: &'a str) -> Option<&'a str> {
    let h = strip_trailing_dots(host);

    // Label byte ranges, left to right.
    let mut ranges_buf = [(0usize, 0usize); MAX_LABELS];
    let mut n = 0usize;
    let mut start = 0usize;
    for (i, b) in h.bytes().enumerate() {
        if b == b'.' {
            if n >= MAX_LABELS {
                return None;
            }
            ranges_buf[n] = (start, i);
            n += 1;
            start = i + 1;
        }
    }
    if n >= MAX_LABELS {
        return None;
    }
    ranges_buf[n] = (start, h.len());
    n += 1;
    // Only the first `n` entries hold valid ranges; slice so that a
    // `.get()` past the end fails instead of reading a stale entry.
    let ranges = &ranges_buf[..n];

    // Walk from the TLD down, recording the node at each depth.
    // depths[k] = (node offset, d) with d = number of labels consumed.
    let mut depths = [(0usize, 0usize); MAX_LABELS];
    let mut ndepth = 0usize;
    let mut node_off = 0; // the root node is at offset 0 of the trie slice
    for d in 1..=n {
        let (s, e) = ranges[n - d];
        let Some(child) = find_child(trie, node_off, &h[s..e]) else {
            break;
        };
        depths[ndepth] = (child, d);
        ndepth += 1;
        node_off = child;
    }

    // Pass 1: an exception rule at any candidate depth prevails (longest
    // wins); the public suffix starts one label beyond the exception rule.
    for &(off, d) in depths[..ndepth].iter().rev() {
        let flags = *trie.get(off)?;
        if flags & FLAG_EXCEPTION != 0 {
            // `n - d + 1` can exceed the valid label range on corrupt
            // input (e.g. an exception flag at depth 1); fail closed.
            let (s, _) = *ranges.get(n - d + 1)?;
            return Some(&h[s..]);
        }
    }
    // Pass 2: longest candidate first, wildcard beats exact.
    for &(off, d) in depths[..ndepth].iter().rev() {
        let flags = *trie.get(off)?;
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
    let h = strip_trailing_dots(host);
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

    #[test]
    fn trailing_dots() {
        assert_eq!(public_suffix("example.com.."), "com");
        assert_eq!(public_suffix("example.com..."), "com");
        assert_eq!(public_suffix("WWW.CK.."), "CK");
        assert_eq!(normalize_host("example.com..."), "example.com");
        assert_eq!(registrable_domain("example.com.."), Some("example.com"));
        assert!(is_public_suffix("co.uk.."));
        // All-dots input is treated as empty.
        assert_eq!(public_suffix("..."), "");
        assert_eq!(normalize_host("..."), "");
        assert_eq!(registrable_domain("..."), None);
    }

    #[test]
    fn normalize_trims_whitespace() {
        // Callers should pre-trim, but stray whitespace must not poison labels.
        assert_eq!(normalize_host(" example.com "), "example.com");
        assert_eq!(normalize_host("\tExample.COM.\n"), "example.com");
        assert_eq!(normalize_host("  ...  "), "");
    }

    /// Build a minimal trie (root node at offset 0) from
    /// `(flags, labels left-to-right)` rules, in the `psl_data.bin` format.
    #[cfg(test)]
    fn build_trie(rules: &[(u8, &[&str])]) -> Vec<u8> {
        #[derive(Clone)]
        struct Node {
            kids: Vec<(Vec<u8>, usize)>,
            flags: u8,
        }
        let mut nodes = vec![Node {
            kids: Vec::new(),
            flags: 0,
        }];
        for &(flags, labels) in rules {
            let mut idx = 0;
            for lab in labels.iter().rev() {
                let labb = lab.as_bytes();
                let next = match nodes[idx].kids.iter().find(|(l, _)| l == labb) {
                    Some(&(_, c)) => c,
                    None => {
                        nodes.push(Node {
                            kids: Vec::new(),
                            flags: 0,
                        });
                        let c = nodes.len() - 1;
                        nodes[idx].kids.push((labb.to_vec(), c));
                        c
                    }
                };
                idx = next;
            }
            nodes[idx].flags |= flags;
        }
        fn serialize(nodes: &[Node], idx: usize, out: &mut Vec<u8>) {
            let mut kids = nodes[idx].kids.clone();
            kids.sort_by(|a, b| a.0.cmp(&b.0));
            out.push(nodes[idx].flags);
            out.extend_from_slice(&(kids.len() as u16).to_le_bytes());
            for (lab, c) in &kids {
                let mut child = Vec::new();
                serialize(nodes, *c, &mut child);
                out.push(lab.len() as u8);
                out.extend_from_slice(lab);
                out.extend_from_slice(&(child.len() as u32).to_le_bytes());
                out.extend_from_slice(&child);
            }
        }
        let mut out = Vec::new();
        serialize(&nodes, 0, &mut out);
        out
    }

    #[test]
    fn exception_flag_at_depth1_fails_closed() {
        // Corrupt trie: EXCEPTION on the TLD node itself. `n - d + 1` is out
        // of range here, so the walk must fail closed (None) instead of
        // panicking or reading a stale label range.
        let t = build_trie(&[(FLAG_EXCEPTION, &["ck"])]);
        assert_eq!(walk_trie(&t, "a.ck"), None);
        // ... including at the maximum label count, where the old direct
        // indexing would panic outright.
        let deep: String = (0..127).map(|_| "a.").collect::<String>() + "a";
        let t2 = build_trie(&[(FLAG_EXCEPTION, &["a"])]);
        assert_eq!(walk_trie(&t2, &deep), None);
        // A well-formed exception still resolves one label beyond the rule.
        let t3 = build_trie(&[(FLAG_EXCEPTION, &["www", "ck"]), (FLAG_WILDCARD, &["ck"])]);
        assert_eq!(walk_trie(&t3, "a.www.ck"), Some("ck"));
        assert_eq!(walk_trie(&t3, "www.ck"), Some("ck"));
    }

    #[test]
    fn exception_priority_is_global() {
        // An exception at a *shorter* candidate must prevail over an exact
        // rule at a longer one: the old longest-first single pass returned
        // "x.y.b" here, but the spec gives the exception global priority.
        let t = build_trie(&[
            (FLAG_EXACT, &["x", "y", "b"]),
            (FLAG_EXCEPTION, &["y", "b"]),
        ]);
        assert_eq!(walk_trie(&t, "x.y.b"), Some("b"));
        // Sanity: without the exception, longest-first still applies.
        let t2 = build_trie(&[(FLAG_EXACT, &["x", "y", "b"])]);
        assert_eq!(walk_trie(&t2, "x.y.b"), Some("x.y.b"));
        // And the longest exception wins when several match.
        let t3 = build_trie(&[
            (FLAG_EXCEPTION, &["y", "b"]),
            (FLAG_EXCEPTION, &["z", "y", "b"]),
        ]);
        assert_eq!(walk_trie(&t3, "z.y.b"), Some("y.b"));
    }
}
