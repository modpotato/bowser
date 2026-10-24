//! Benchmark harness for DOM operations (std only, no deps).
//!
//! Times the crate's hot paths on a deterministic synthetic document:
//! parse + build, traversal (`descendants`, `walk`-heavy queries),
//! `query_selector_all`, `serialize`, `get_element_by_id`,
//! `get_elements_by_tag_name`, `text_content`, `clone_node` (deep),
//! attribute access, and bulk mutation (`append_child`).
//!
//! Run with: `cargo bench -p bowser_dom --bench dom_ops` (uses the
//! workspace bench profile). This is a `harness = false` bench so it
//! works on stable Rust with zero dependencies.

use bowser_dom::Document;
use std::time::Instant;

/// Deterministic xorshift64 PRNG.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const WORDS: &[&str] = &[
    "lorem",
    "ipsum",
    "dolor",
    "sit",
    "amet",
    "consectetur",
    "adipiscing",
];
const TAGS: &[&str] = &["div", "span", "p", "section", "article", "em", "strong"];

/// Build a deterministic synthetic document via the public mutation API.
fn build_doc(n: usize) -> Document {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut doc = Document::new();
    let root = doc.root();
    let html = doc.create_element("html");
    doc.append_child(root, html);
    let body = doc.create_element("body");
    doc.append_child(html, body);
    for i in 0..n {
        let t = TAGS[rng.below(TAGS.len() as u64) as usize];
        let el = doc.create_element(t);
        doc.set_attribute(el, "class", &format!("c{}", i % 50));
        doc.set_attribute(el, "data-x", "v");
        if i % 10 == 0 {
            doc.set_attribute(el, "id", &format!("n{i}"));
        }
        let txt = doc.create_text_node(&format!("{} {} ", WORDS[i % WORDS.len()], i));
        doc.append_child(el, txt);
        doc.append_child(body, el);
    }
    doc
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Time `f` over `runs` iterations, return median seconds.
fn bench<R>(name: &str, runs: usize, warmup: usize, mut f: impl FnMut() -> R) -> f64
where
    R: Sized,
{
    let mut sink = Vec::with_capacity(runs);
    for _ in 0..warmup {
        std::hint::black_box(f());
    }
    for _ in 0..runs {
        let start = Instant::now();
        let r = f();
        // Observe the result so the optimizer cannot eliminate the work.
        std::hint::black_box(&r);
        let dt = start.elapsed().as_secs_f64();
        sink.push((dt, std::mem::size_of_val(&r) as u64));
    }
    let med = median(sink.iter().map(|(dt, _)| *dt).collect());
    std::hint::black_box(&sink);
    println!("  {name:<32} median: {:>9.3} ms", med * 1000.0);
    med
}

fn main() {
    // ~5000 elements, ~10k text nodes, ~15010 arena slots.
    let doc = build_doc(5000);
    let root = doc.root();
    let body = doc.document_element().unwrap();
    println!("document nodes: (root id = {root:?})");

    bench("build_doc(5000)", 11, 2, || build_doc(5000));
    bench("descendants(root)", 21, 3, || doc.descendants(root));
    bench("descendants(body)", 21, 3, || doc.descendants(body));
    bench("query_selector_all(\"div.c7 span\")", 11, 2, || {
        doc.query_selector_all("div.c7 span")
    });
    bench("query_selector_all(\"div > .c7\")", 11, 2, || {
        doc.query_selector_all("div > .c7")
    });
    bench("query_selector_all(\"#n9990\")", 11, 2, || {
        doc.query_selector_all("#n9990")
    });
    bench("serialize(root)", 11, 2, || doc.serialize());
    bench("get_element_by_id(\"n9990\")", 21, 3, || {
        doc.get_element_by_id("n9990")
    });
    bench("get_elements_by_tag_name(\"div\")", 21, 3, || {
        doc.get_elements_by_tag_name("div")
    });
    bench("text_content(root)", 21, 3, || doc.text_content(root));

    // Mutation-heavy: append 5000 nodes to a fresh body each run.
    bench("append_child x5000", 21, 3, || {
        let mut d = Document::new();
        let b = d.create_element("body");
        d.append_child(d.root(), b);
        for i in 0..5000 {
            let el = d.create_element("div");
            d.set_attribute(el, "id", &format!("m{i}"));
            d.append_child(b, el);
        }
        d
    });

    // Deep clone of the body subtree (fresh identical doc each run:
    // `build_doc` is deterministic so ids line up with `body`).
    bench("clone_node(deep) html", 11, 2, || {
        let mut d = build_doc(5000);
        d.clone_node(d.document_element().unwrap(), true)
    });

    // Attribute access: repeated get/set on a hot element.
    // Attribute access: repeated get/set on a hot element.
    let target = doc.get_element_by_id("n10").unwrap();
    bench("get_attribute x100k", 11, 2, || {
        let mut acc = 0usize;
        for _ in 0..100_000 {
            acc += doc.get_attribute(target, "class").unwrap().len();
        }
        acc
    });
    bench("set_attribute x100k", 11, 2, || {
        let mut d = build_doc(5000);
        let t = d.get_element_by_id("n10").unwrap();
        for i in 0..100_000 {
            d.set_attribute(t, "data-k", &format!("v{i}"));
        }
        d
    });
}
