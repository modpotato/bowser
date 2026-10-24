//! Hand-rolled benchmark for the layout engine (std only; no bench harness).
//!
//! Run: `cargo run --release --example bench_layout`
//! Cases can be selected with argv: blocks | inline | deep | ibs | all

use bowser_dom::Document;
use bowser_layout::{FixedFontMetrics, Size};
use std::time::Instant;

const VP: Size = Size { w: 800.0, h: 600.0 };
const CSS: &str = r#"
p { margin: 8px 4px; padding: 3px; font-size: 16px; line-height: 1.4; }
span.hi { font-size: 20px; color: red; }
div.box { width: 50%; margin: 5px auto; border-top-width: 1px; }
.ib { display: inline-block; margin: 2px; padding: 4px; }
b { font-size: 1.2em; }
"#;

fn gen_blocks(paras: usize, words: usize) -> String {
    let mut s = String::from("<html><body>");
    let para = "<p>The quick brown fox jumps over the lazy dog <span class=\"hi\">PACK</span> \
                my box with five dozen liquor jugs <b>bold words here</b> and more text.</p>";
    for _ in 0..paras {
        s.push_str(para);
    }
    let _ = words;
    s.push_str("</body></html>");
    s
}

fn gen_inline_text(paras: usize, words_per: usize) -> String {
    let mut s = String::from("<html><body>");
    for _ in 0..paras {
        s.push_str("<p>");
        for w in 0..words_per {
            if w % 7 == 0 {
                s.push_str("<span class=\"hi\">word</span> ");
            } else if w % 11 == 0 {
                s.push_str("<b>word</b> ");
            } else {
                s.push_str("word ");
            }
        }
        s.push_str("</p>");
    }
    s.push_str("</body></html>");
    s
}

fn gen_deep(depth: usize) -> String {
    let mut s = String::from("<html><body>");
    for _ in 0..depth {
        s.push_str("<div class=\"box\">text <span>deep</span>");
    }
    for _ in 0..depth {
        s.push_str("</div>");
    }
    s.push_str("</body></html>");
    s
}

fn gen_inline_blocks(count: usize, nest: usize) -> String {
    fn one(nest: usize) -> String {
        let mut s = String::from("<span class=\"ib\">text ");
        if nest > 0 {
            s.push_str(&one(nest - 1));
        }
        s.push_str(" tail</span>");
        s
    }
    let mut s = String::from("<html><body><p>");
    for _ in 0..count {
        s.push_str(&one(nest));
        s.push(' ');
    }
    s.push_str("</p></body></html>");
    s
}

fn bench(name: &str, html: &str, iters: usize) {
    let (doc, _) = Document::parse_html(html.as_bytes());
    let ss = bowser_css::parse_stylesheet(CSS);
    // count elements for per-element reporting
    let fonts = FixedFontMetrics;
    // warmup
    for _ in 0..3 {
        let _ = bowser_layout::layout(&doc, &[&ss], VP, &fonts);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let r = bowser_layout::layout(&doc, &[&ss], VP, &fonts);
        std::hint::black_box(r.rect.w);
    }
    let dt = t0.elapsed();
    let per = dt.as_secs_f64() / iters as f64 * 1e6;
    println!("{name:>14}: {per:10.1} us/iter over {iters} iters");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    let run = |name: &str| which == "all" || which == name;
    if run("blocks") {
        bench("blocks", &gen_blocks(2000, 0), 10);
    }
    if run("inline") {
        bench("inline", &gen_inline_text(400, 60), 10);
    }
    if run("deep") {
        bench("deep", &gen_deep(3000), 10);
    }
    if run("ibs") {
        bench("ibs", &gen_inline_blocks(400, 4), 10);
    }
}
