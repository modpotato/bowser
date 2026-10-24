//! Conformance tests for `:has()` relative-selector matching.
//!
//! The traversal itself is allocation-free (pinned by `has_alloc.rs`);
//! these tests pin the *semantics*: child vs descendant anchoring,
//! `+`/`~` sibling combinators (including intervening text nodes),
//! multi-compound relative selectors, forgiving relative-selector
//! lists, nested-`:has()` rejection, and anchor exclusion.

use bowser_css::{matches, parse_selector_list};
use bowser_dom::{Document, NodeId};

/// Fixture:
///
/// ```text
/// section#s
///   div#a
///     p#p1
///     span.x
///       em#e1
///   "text"
///   div#b
///     span
///     em#e2
///   div#c          (empty)
/// ```
///
/// The text node between `#a` and `#b` pins that sibling combinators
/// skip non-element siblings.
struct Fixture {
    doc: Document,
    a: NodeId,
    b: NodeId,
    c: NodeId,
    p1: NodeId,
    x: NodeId,
    s: NodeId,
}

fn fixture() -> Fixture {
    let mut doc = Document::new();
    let root = doc.root();
    let s = doc.create_element("section");
    doc.set_attribute(s, "id", "s");
    doc.append_child(root, s);

    let a = doc.create_element("div");
    doc.set_attribute(a, "id", "a");
    doc.append_child(s, a);
    let p1 = doc.create_element("p");
    doc.set_attribute(p1, "id", "p1");
    doc.append_child(a, p1);
    let x = doc.create_element("span");
    doc.set_attribute(x, "class", "x");
    doc.append_child(a, x);
    let e1 = doc.create_element("em");
    doc.set_attribute(e1, "id", "e1");
    doc.append_child(x, e1);

    let text = doc.create_text_node("hello");
    doc.append_child(s, text);

    let b = doc.create_element("div");
    doc.set_attribute(b, "id", "b");
    doc.append_child(s, b);
    let bspan = doc.create_element("span");
    doc.append_child(b, bspan);
    let e2 = doc.create_element("em");
    doc.set_attribute(e2, "id", "e2");
    doc.append_child(b, e2);

    let c = doc.create_element("div");
    doc.set_attribute(c, "id", "c");
    doc.append_child(s, c);

    Fixture {
        doc,
        a,
        b,
        c,
        p1,
        x,
        s,
    }
}

fn sel(css: &str) -> bowser_css::Selector {
    parse_selector_list(css).unwrap()[0].clone()
}

#[test]
fn has_child_vs_descendant_anchoring() {
    let f = fixture();
    // `> em`: only a *direct* child em counts (`#b` only).
    let gt_em = sel("div:has(> em)");
    assert!(matches(&gt_em, &f.doc, f.b));
    assert!(!matches(&gt_em, &f.doc, f.a));
    assert!(!matches(&gt_em, &f.doc, f.c));
    // Bare `em`: any descendant counts (`#a` via `#e1`, `#b` via `#e2`).
    let desc_em = sel("div:has(em)");
    assert!(matches(&desc_em, &f.doc, f.a));
    assert!(matches(&desc_em, &f.doc, f.b));
    assert!(!matches(&desc_em, &f.doc, f.c));
    // `> p` vs `p`: `#p1` is a direct child of `#a` only.
    assert!(matches(&sel("div:has(> p)"), &f.doc, f.a));
    assert!(!matches(&sel("div:has(> p)"), &f.doc, f.b));
}

#[test]
fn has_multi_compound_relative_selectors() {
    let f = fixture();
    // Child chain: `#a > span > em` only.
    let chain = sel("div:has(> span > em)");
    assert!(matches(&chain, &f.doc, f.a));
    assert!(!matches(&chain, &f.doc, f.b));
    assert!(!matches(&chain, &f.doc, f.c));
    // Mixed descendant/child: a span descendant that itself has an em
    // child — `#a` only (`#b`'s span has no em child).
    let mixed = sel("div:has(span > em)");
    assert!(matches(&mixed, &f.doc, f.a));
    assert!(!matches(&mixed, &f.doc, f.b));
    // Longer chain through the section: `section > div > span > em`.
    assert!(matches(&sel("section:has(> div > span > em)"), &f.doc, f.s));
    assert!(!matches(&sel("section:has(> div > p > em)"), &f.doc, f.s));
}

#[test]
fn has_sibling_combinators() {
    let f = fixture();
    // `+`: next *element* sibling — the text node between `#a` and
    // `#b` does not break adjacency.
    let plus_div = sel("div:has(+ div)");
    assert!(matches(&plus_div, &f.doc, f.a));
    assert!(matches(&plus_div, &f.doc, f.b));
    assert!(!matches(&plus_div, &f.doc, f.c));
    // `+` is immediate: `#a`'s next element sibling is `#b`, not `#c`.
    assert!(!matches(&sel("#a:has(+ #c)"), &f.doc, f.a));
    assert!(matches(&sel("#a:has(+ #b)"), &f.doc, f.a));
    // `~`: any later element sibling.
    let tilde_div = sel("div:has(~ div)");
    assert!(matches(&tilde_div, &f.doc, f.a));
    assert!(matches(&tilde_div, &f.doc, f.b));
    assert!(!matches(&tilde_div, &f.doc, f.c));
    assert!(matches(&sel("#a:has(~ #c)"), &f.doc, f.a));
    // `#p1 + span`: `#p1`'s next element sibling is `span.x`.
    assert!(matches(&sel("p:has(+ span)"), &f.doc, f.p1));
    assert!(!matches(&sel("p:has(+ em)"), &f.doc, f.p1));
    // Multi-compound with a sibling step: `#a + div` (`#b`) has a span
    // child; `#b + div` (`#c`) has no children.
    assert!(matches(&sel("#a:has(+ div > span)"), &f.doc, f.a));
    assert!(!matches(&sel("#b:has(+ div > span)"), &f.doc, f.b));
    // `~` multi-compound: some later div has a span child.
    assert!(matches(&sel("#a:has(~ div > span)"), &f.doc, f.a));
    assert!(!matches(&sel("#b:has(~ div > span)"), &f.doc, f.b));
}

#[test]
fn has_relative_selector_list() {
    let f = fixture();
    // Either branch may match: `#a` via `> .x`, `#b` via `> #e2`.
    let list = sel("div:has(> .x, > #e2)");
    assert!(matches(&list, &f.doc, f.a));
    assert!(matches(&list, &f.doc, f.b));
    assert!(!matches(&list, &f.doc, f.c));
    // Mixed combinators across branches.
    let mixed = sel("div:has(> p, ~ div)");
    assert!(matches(&mixed, &f.doc, f.a)); // `> p`
    assert!(matches(&mixed, &f.doc, f.b)); // `~ div` (`#c`)
    assert!(!matches(&mixed, &f.doc, f.c));
}

#[test]
fn has_forgiving_invalid_group() {
    let f = fixture();
    // An invalid group is dropped; the valid one still decides.
    let forgiving = sel("div:has(> span, >>>)");
    let plain = sel("div:has(> span)");
    for n in [f.a, f.b, f.c] {
        assert_eq!(
            matches(&forgiving, &f.doc, n),
            matches(&plain, &f.doc, n),
            "forgiving mismatch at {n:?}"
        );
    }
    assert!(matches(&forgiving, &f.doc, f.a));
    // No valid group at all: the whole selector is invalid.
    assert!(parse_selector_list("div:has(>>>)").is_err());
    assert!(parse_selector_list("div:has()").is_err());
}

#[test]
fn has_cannot_nest() {
    // Selectors 4 forbids `:has()` inside `:has()`'s argument.
    assert!(parse_selector_list("div:has(span:has(em))").is_err());
    assert!(parse_selector_list("div:has(:has(span))").is_err());
    // ...but the outer `:has()` itself still parses when valid.
    assert!(parse_selector_list("div:has(span)").is_ok());
}

#[test]
fn has_excludes_anchor() {
    let f = fixture();
    // The anchor is never its own `:has()` witness: `#a` has no div
    // descendants, `span.x` has no span descendants.
    assert!(!matches(&sel("div:has(div)"), &f.doc, f.a));
    assert!(!matches(&sel("span:has(span)"), &f.doc, f.x));
    // Descendants do count.
    assert!(matches(&sel("section:has(div)"), &f.doc, f.s));
    assert!(matches(&sel("div:has(span)"), &f.doc, f.a));
    // The anchor is excluded for every leading combinator: `#c` has no
    // children or siblings matching.
    assert!(!matches(&sel("div:has(> div)"), &f.doc, f.c));
    assert!(!matches(&sel("div:has(+ div)"), &f.doc, f.c));
}

#[test]
fn has_with_outer_compound() {
    let f = fixture();
    // `:has()` combines with the outer compound normally.
    assert!(matches(&sel("div#a:has(> span.x)"), &f.doc, f.a));
    assert!(!matches(&sel("div#b:has(> span.x)"), &f.doc, f.b));
    assert!(!matches(&sel("span#a:has(> span)"), &f.doc, f.a));
    // Chained `:has()`: `#a` has a span child with an em child.
    assert!(matches(&sel("div:has(> span):has(> p)"), &f.doc, f.a));
    assert!(!matches(&sel("div:has(> span):has(> p)"), &f.doc, f.b));
}
