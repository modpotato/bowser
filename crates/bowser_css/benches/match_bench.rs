//! Benchmark: selector matching throughput.
//!
//! Builds a synthetic document (~2300 elements with classes, ids,
//! attributes, text children) and a representative selector set, then
//! times `matches_with_state` over every element x selector pair.
//!
//! Run: `cargo bench --bench match_bench` (needs a release build for
//! meaningful numbers).

use std::time::Instant;

use bowser_css::{ElementState, matches_with_state, parse_selector_list};
use bowser_dom::{Document, NodeId, NodeKind};

const SELECTORS: &[&str] = &[
    "div",
    "DIV",
    ".card",
    ".active",
    "#item-10",
    "div.card",
    "div.card.active",
    "[data-role]",
    "[data-role=button]",
    "[class~=active]",
    "[class~=card]",
    "[lang|=en]",
    "[id^=item]",
    "[id$=x]",
    "[data-role*=but]",
    "[id=ITEM-20 i]",
    "[lang^=EN i]",
    "section .card",
    "section div.card.active",
    "div > p",
    "div > span",
    "p + p",
    "p ~ span",
    "li:nth-child(2n+1)",
    "li:nth-child(odd)",
    "li:nth-child(3)",
    "li:first-child",
    "li:last-child",
    "span:empty",
    "p:empty",
    "div:not(.hidden)",
    ":not(span)",
    ":root",
    "svg|rect",
    "*|*",
    "*",
    "div p a",
];

fn build_doc() -> (Document, Vec<NodeId>) {
    let mut doc = Document::new();
    let root = doc.root();
    let html = doc.create_element("html");
    doc.append_child(root, html);
    let body = doc.create_element("body");
    doc.append_child(html, body);
    let mut els = Vec::with_capacity(2400);
    els.push(html);
    els.push(body);
    for s in 0..50 {
        let sec = doc.create_element("section");
        doc.append_child(body, sec);
        doc.set_attribute(sec, "lang", "en-US");
        doc.set_attribute(sec, "id", &format!("sec-{s}"));
        els.push(sec);
        for i in 0..40 {
            let tag = match i % 5 {
                0 => "div",
                1 => "p",
                2 => "span",
                3 => "li",
                _ => "a",
            };
            let el = doc.create_element(tag);
            doc.append_child(sec, el);
            if i % 3 == 0 {
                doc.set_attribute(el, "class", "card active primary");
            } else if i % 3 == 1 {
                doc.set_attribute(el, "class", "card hidden");
            }
            if i % 4 == 0 {
                doc.set_attribute(el, "id", &format!("item-{}", s * 40 + i));
            }
            if i % 5 == 0 {
                doc.set_attribute(el, "data-role", "button");
            }
            if i % 2 == 0 {
                let t = doc.create_text_node("some text");
                doc.append_child(el, t);
            }
            els.push(el);
        }
        // An SVG island to exercise namespace selectors.
        let svg = doc.create_element_ns(bowser_dom::Namespace::Svg, "svg");
        doc.append_child(sec, svg);
        els.push(svg);
        for _ in 0..5 {
            let r = doc.create_element_ns(bowser_dom::Namespace::Svg, "rect");
            doc.append_child(svg, r);
            els.push(r);
        }
    }
    // Sanity: all collected nodes are elements.
    debug_assert!(
        els.iter()
            .all(|&n| matches!(doc.kind(n), NodeKind::Element(_)))
    );
    (doc, els)
}

fn main() {
    let (doc, els) = build_doc();
    let sels: Vec<_> = SELECTORS
        .iter()
        .map(|s| {
            let list = parse_selector_list(s).expect("bench selector must parse");
            assert_eq!(list.len(), 1, "{s}");
            list.into_iter().next().unwrap()
        })
        .collect();
    let state = ElementState::default();
    let pairs = els.len() * sels.len();
    println!(
        "elements: {}, selectors: {}, pairs/round: {}",
        els.len(),
        sels.len(),
        pairs
    );

    // Warmup.
    let mut sink: u64 = 0;
    for &el in &els {
        for sel in &sels {
            sink += matches_with_state(sel, &doc, el, &state) as u64;
        }
    }

    let rounds = 40;
    let start = Instant::now();
    for _ in 0..rounds {
        for &el in &els {
            for sel in &sels {
                sink += matches_with_state(sel, &doc, el, &state) as u64;
            }
        }
    }
    let dt = start.elapsed();
    let total_matches = pairs as u64 * rounds as u64;
    let ns_per = dt.as_nanos() as f64 / total_matches as f64;
    println!("sink: {sink} (ignore)");
    println!(
        "total: {:.3}s for {total_matches} matches",
        dt.as_secs_f64()
    );
    println!("ns/match: {ns_per:.1}");
    println!("matches/sec: {:.0}", 1e9 / ns_per);
}
