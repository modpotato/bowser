//! Benchmark: the `:nth-child` family at scale.
//!
//! Builds a document with one wide parent (4000 `li` children) and styles
//! every child through a `Styler` over a stylesheet of `:nth-child` /
//! `:nth-last-child` / `:nth-of-type` rules. That exercises the cascade's
//! shared `SiblingCache`: the per-parent sibling index is built once and
//! every repeated query against the parent is O(1) amortized.
//!
//! For contrast, `naive` reimplements the old per-query sibling walk —
//! what the matcher did before memoization — for the same three index
//! computations. It is a strict lower bound on the old cost: it skips the
//! cascade entirely, so any gap understates the real win.
//!
//! Run: `cargo bench --bench nth_bench` (needs a release build for
//! meaningful numbers).

use std::time::Instant;

use bowser_css::{Styler, Viewport, default_ua_stylesheet, parse_stylesheet};
use bowser_dom::{Document, NodeId, NodeKind};

const N: usize = 4000;

fn build_doc() -> (Document, Vec<NodeId>) {
    let mut doc = Document::new();
    let root = doc.root();
    let html = doc.create_element("html");
    doc.append_child(root, html);
    let body = doc.create_element("body");
    doc.append_child(html, body);
    let div = doc.create_element("div");
    doc.append_child(body, div);
    let mut lis = Vec::with_capacity(N);
    for _ in 0..N {
        let li = doc.create_element("li");
        doc.append_child(div, li);
        lis.push(li);
    }
    (doc, lis)
}

fn is_element(doc: &Document, node: NodeId) -> bool {
    matches!(doc.kind(node), NodeKind::Element(_))
}

fn prev_element_sibling(doc: &Document, node: NodeId) -> Option<NodeId> {
    let mut s = doc.previous_sibling(node);
    while let Some(n) = s {
        if is_element(doc, n) {
            return Some(n);
        }
        s = doc.previous_sibling(n);
    }
    None
}

fn next_element_sibling(doc: &Document, node: NodeId) -> Option<NodeId> {
    let mut s = doc.next_sibling(node);
    while let Some(n) = s {
        if is_element(doc, n) {
            return Some(n);
        }
        s = doc.next_sibling(n);
    }
    None
}

/// The pre-memoization algorithm: one O(siblings) walk per query, so
/// matching `:nth-child` over all N children costs O(N^2). Mirrors the
/// three index computations the old matcher performed.
fn naive_indexes(doc: &Document, node: NodeId) -> (i32, i32, i32) {
    let mut index = 1;
    let mut s = prev_element_sibling(doc, node);
    while let Some(n) = s {
        index += 1;
        s = prev_element_sibling(doc, n);
    }
    let mut from_end = 1;
    let mut t = next_element_sibling(doc, node);
    while let Some(n) = t {
        from_end += 1;
        t = next_element_sibling(doc, n);
    }
    // `:nth-of-type`: same-tag walk (all children are `li` here).
    let mut of_type = 1;
    let mut u = prev_element_sibling(doc, node);
    while let Some(n) = u {
        of_type += 1;
        u = prev_element_sibling(doc, n);
    }
    (index, from_end, of_type)
}

fn main() {
    let (doc, lis) = build_doc();
    let sheet = parse_stylesheet(
        "li:nth-child(2n) { color: red; } \
         li:nth-last-child(3n+1) { color: blue; } \
         li:nth-of-type(2) { color: green; }",
    );
    let ua = default_ua_stylesheet();
    let viewport = Viewport { w: 800.0, h: 600.0 };
    let author = [&sheet];
    let styler = Styler::new(&doc, &author, &ua, viewport);

    // Warmup: styles every element once, building the sibling index.
    let mut sink = 0u64;
    for &li in &lis {
        sink += styler.style_for(li).get("color").is_some() as u64;
    }

    let rounds = 10u64;
    let start = Instant::now();
    for _ in 0..rounds {
        for &li in &lis {
            sink += styler.style_for(li).get("color").is_some() as u64;
        }
    }
    let styled = start.elapsed();
    let ns_per_element = styled.as_nanos() as f64 / (N as f64 * rounds as f64);

    let start = Instant::now();
    for _ in 0..rounds {
        for &li in &lis {
            let (a, b, c) = naive_indexes(&doc, li);
            sink += (a + b + c) as u64;
        }
    }
    let naive = start.elapsed();

    println!("sink: {sink} (ignore)");
    println!("elements: {N}, rounds: {rounds}");
    println!(
        "styler (memoized): {:.3}s total, {:.1} ns/element",
        styled.as_secs_f64(),
        ns_per_element
    );
    println!(
        "naive index walks alone: {:.3}s total ({:.1}x the full \
         memoized style pass, and the old matcher paid this *on top of* \
         the cascade)",
        naive.as_secs_f64(),
        naive.as_secs_f64() / styled.as_secs_f64()
    );
}
