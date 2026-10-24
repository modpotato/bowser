//! Benchmark harness for `Document::parse_html` (std only, no deps).
//!
//! Builds one deterministic synthetic document (273.1 KiB, measured)
//! exercising the hot paths of tree construction — nested divs, tables
//! (foster parenting), lists, formatting elements (adoption agency),
//! comments, PIs, foreign SVG/MathML content, templates,
//! select/selectedcontent, entities, and script/style raw text — then times
//! `parse_html` over repeated runs and reports median wall time and
//! throughput.
//!
//! Run with: `cargo bench -p bowser_dom` (uses the workspace release
//! profile). This is a `harness = false` bench so it works on stable
//! Rust with zero dependencies.

use bowser_dom::Document;
use std::time::Instant;

/// Deterministic xorshift64 PRNG (std-only; the input must not depend on
/// any external RNG crate).
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
    "elit",
    "sed",
    "do",
    "eiusmod",
    "tempor",
    "αβγ",
    "🦊",
    "naïve",
];
const TAGS: &[&str] = &[
    "div", "span", "p", "section", "article", "em", "strong", "b", "i", "u", "s",
];

/// Build a deterministic 273.1 KiB HTML document (measured) hitting the
/// parser's hot paths.
fn build_input() -> Vec<u8> {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut out = String::new();
    out.push_str("<!DOCTYPE html><html><head><title>bench</title><style>.x{color:red}</style>");
    out.push_str("<script>if(a<b&&c>d){e(1)}</script></head><body>");
    // Nested formatting (adoption agency pressure) + deep-ish nesting.
    for i in 0..2800 {
        let t = TAGS[rng.below(TAGS.len() as u64) as usize];
        out.push('<');
        out.push_str(t);
        out.push_str(" class=\"c");
        out.push_str(&i.to_string());
        out.push_str("\" id=\"n");
        out.push_str(&i.to_string());
        out.push_str("\" data-x=\"v\">");
        if i % 7 == 0 {
            out.push_str("</b></i></u></s>");
        }
        for _ in 0..3 {
            out.push_str(WORDS[rng.below(WORDS.len() as u64) as usize]);
            out.push(' ');
        }
        if i % 11 == 0 {
            out.push_str("<!-- comment ");
            out.push_str(&i.to_string());
            out.push_str(" -->");
        }
        if i % 50 == 0 {
            // Foster parenting: stray text/table children.
            out.push_str("<table><tr><td>cell");
            out.push_str(&i.to_string());
            out.push_str("</td></tr>stray<tr><td>more</td></tr></table>");
        }
        if i % 100 == 0 {
            out.push_str("<svg><foreignObject><p>html-in-svg</p></foreignObject></svg>");
            out.push_str("<math><mi>x</mi><mo>+</mo><mn>1</mn></math>");
        }
        if i % 200 == 0 {
            out.push_str("<template><div>tmpl");
            out.push_str(&i.to_string());
            out.push_str("</div></template>");
        }
    }
    for _ in 0..2800 {
        out.push_str("</div>");
    }
    out.push_str("<select><option>one</option><option selected>two</option>");
    out.push_str("<button><selectedcontent></selectedcontent></button></select>");
    out.push_str("<ul>");
    for i in 0..2000 {
        out.push_str("<li>item ");
        out.push_str(&i.to_string());
        out.push_str(" &amp; &lt;entity&gt;</li>");
    }
    out.push_str("</ul><?pi target data?><p>tail");
    out.push_str("</body></html>");
    out.into_bytes()
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

fn main() {
    let input = build_input();
    println!(
        "input: {} bytes ({:.1} KiB)",
        input.len(),
        input.len() as f64 / 1024.0
    );

    // Warmup (page faults, allocator pools).
    for _ in 0..3 {
        let _ = Document::parse_html(&input);
    }

    let mut times = Vec::with_capacity(21);
    // Keep one document alive across iterations so the optimizer cannot
    // discard the parse; check a checksum-ish property per run.
    let mut sink_check: usize = 0;
    for _ in 0..21 {
        let start = Instant::now();
        let (doc, _quirks) = Document::parse_html(&input);
        let elapsed = start.elapsed();
        sink_check = sink_check.wrapping_add(doc.serialize().len());
        times.push(elapsed.as_secs_f64());
    }
    std::hint::black_box(sink_check);

    let med = median(times);
    let mib_s = (input.len() as f64 / (1024.0 * 1024.0)) / med;
    println!(
        "runs: 21, median: {:.3} ms ({:.2} MiB/s)",
        med * 1000.0,
        mib_s
    );
}
