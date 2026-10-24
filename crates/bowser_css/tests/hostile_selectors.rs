//! Hostile selector shapes: pathological inputs from remote stylesheets
//! must degrade gracefully, never abort the process.
//!
//! Complex selectors are compound-capped at parse time
//! (`MAX_COMPLEX_COMPOUNDS` in `selector.rs`): a 10k-compound selector is
//! rejected during parsing instead of reaching the matcher, which bounds
//! `match_complex`'s per-call failed-state memo (compounds x nodes
//! entries) no matter how hostile the stylesheet. `:has()` relative
//! selectors keep their own tighter cap (`MAX_HAS_RELATIVE_COMPOUNDS`):
//! over-cap groups are dropped like invalid groups of the forgiving
//! list. Both the parser and the matcher are iterative, so even a
//! max-cap selector cannot overflow the native stack; these tests pin
//! that on the default 2 MiB test-thread stack, where one native frame
//! per compound would blow past the limit.

use bowser_css::{matches, parse_selector_list};
use bowser_dom::{Document, NodeId};

/// A chain of `depth` nested divs under the document root; returns the
/// outermost and the innermost div.
fn div_chain(depth: usize) -> (Document, NodeId, NodeId) {
    let mut doc = Document::new();
    let root = doc.root();
    let mut parent = root;
    for _ in 0..depth {
        let el = doc.create_element("div");
        doc.append_child(parent, el);
        parent = el;
    }
    let outer = doc.first_child(root).expect("chain has an outermost div");
    (doc, outer, parent)
}

/// `div:has(> div > div > ...)` with `compounds` div compounds.
fn child_chain_selector(compounds: usize) -> String {
    format!("div:has(> {})", vec!["div"; compounds].join(" > "))
}

/// `div:has(~ div ~ div ~ ...)` with `compounds` div compounds.
fn sibling_chain_selector(compounds: usize) -> String {
    format!("div:has(~ {})", vec!["div"; compounds].join(" ~ "))
}

/// Over-cap `:has()` relative selectors are rejected at parse time: the
/// group is dropped like an invalid group of the forgiving list, and
/// with no surviving group the selector is invalid. Parsing is
/// iterative, so even a 10k-compound argument is rejected without
/// touching the native stack — and quickly.
#[test]
fn over_cap_has_relative_selector_rejected_at_parse() {
    // At the cap the argument still parses.
    assert!(parse_selector_list(&child_chain_selector(16)).is_ok());
    for n in [17, 100, 10_000] {
        let start = std::time::Instant::now();
        assert!(
            parse_selector_list(&child_chain_selector(n)).is_err(),
            "{n}-compound :has() argument should be rejected at parse"
        );
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "rejecting a {n}-compound :has() argument took too long"
        );
    }
    // Sibling chains are capped the same way.
    assert!(parse_selector_list(&sibling_chain_selector(17)).is_err());
    assert!(parse_selector_list(&sibling_chain_selector(12_000)).is_err());
}

#[test]
fn max_cap_child_chain_has_miss_cannot_overflow_stack() {
    let (doc, outer, _) = div_chain(200);
    // 16 compounds (the parse cap) against a 200-deep chain, the last
    // unmatchable: misses, but must return `false` — the iterative
    // `RelFrame` stack keeps the search off the native stack.
    let mut parts = vec!["div"; 15];
    parts.push("zzz");
    let sel_str = format!("div:has(> {})", parts.join(" > "));
    let sel = parse_selector_list(&sel_str).unwrap();
    assert!(!matches(&sel[0], &doc, outer));
}

#[test]
fn max_cap_sibling_chain_has_hit_cannot_overflow_stack() {
    // 16 subsequent-sibling steps down one `:has()` argument: the old
    // `dfs_relative` recursed one native frame per matched compound
    // (~300B+ each), overflowing the 2 MiB test-thread stack. The
    // iterative search must return `true`. Siblings — not a deep chain —
    // keep fixture building O(n): every append lands directly under the
    // same parent, so no ancestor walk grows with depth. (The siblings
    // hang off a wrapper div: a `Document` accepts only one root
    // element, so they cannot be root-level siblings.)
    let mut doc = Document::new();
    let root = doc.root();
    let wrap = doc.create_element("div");
    doc.append_child(root, wrap);
    let anchor = doc.create_element("div");
    doc.append_child(wrap, anchor);
    for _ in 0..16 {
        let el = doc.create_element("div");
        doc.append_child(wrap, el);
    }
    let sel = parse_selector_list(&sibling_chain_selector(16)).unwrap();
    assert!(matches(&sel[0], &doc, anchor));
}

#[test]
fn max_cap_outer_selector_cannot_overflow_stack() {
    let (doc, _, inner) = div_chain(200);
    // 64 compounds (the parse cap) against a 200-deep chain, the
    // leftmost unmatchable: misses, but must return `false` — the
    // iterative `match_complex` Frame stack keeps the search off the
    // native stack.
    let sel_str = format!("zzz {}", vec!["div"; 63].join(" "));
    let sel = parse_selector_list(&sel_str).unwrap();
    assert!(!matches(&sel[0], &doc, inner));
}

#[test]
fn over_cap_outer_selector_rejected_at_parse() {
    // A 2000-compound outer selector never reaches the matcher: the
    // parse-time compound cap rejects it quickly, bounding
    // `match_complex`'s per-call failed-state memo (compounds x nodes
    // entries) no matter how hostile the stylesheet.
    let sel_str = format!("div {}", vec!["div"; 2_000].join(" "));
    let start = std::time::Instant::now();
    assert!(parse_selector_list(&sel_str).is_err());
    assert!(
        start.elapsed() < std::time::Duration::from_secs(10),
        "rejecting a 2000-compound outer selector took too long"
    );
}
