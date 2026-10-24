//! Benchmark harness for the layout engine (std only, no deps).
//!
//! Times the full `layout()` hot path (style already applied per call —
//! the engine resolves styles during layout) on deterministic synthetic
//! documents: block paragraphs, inline text runs, deep nesting,
//! inline-blocks, flexbox, and tables.
//!
//! Run with: `cargo bench -p bowser_layout` (uses the workspace bench
//! profile). This is a `harness = false` bench so it works on stable
//! Rust with zero dependencies.

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
.flex { display: flex; gap: 4px; }
.flex > div { flex: 1 1 auto; min-width: 0; }
table.t { border-collapse: collapse; width: 100%; }
table.t td { border: 1px solid black; padding: 2px 4px; }
"#;

fn gen_blocks(paras: usize) -> String {
    let mut s = String::from("<html><body>");
    let para = "<p>The quick brown fox jumps over the lazy dog <span class=\"hi\">PACK</span> \
                my box with five dozen liquor jugs <b>bold words here</b> and more text.</p>";
    for _ in 0..paras {
        s.push_str(para);
    }
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

fn gen_flex(rows: usize, per_row: usize) -> String {
    let mut s = String::from("<html><body>");
    for _ in 0..rows {
        s.push_str("<div class=\"flex\">");
        for i in 0..per_row {
            s.push_str(&format!("<div>item {i} with some text</div>"));
        }
        s.push_str("</div>");
    }
    s.push_str("</body></html>");
    s
}

fn gen_table(rows: usize, cols: usize) -> String {
    let mut s = String::from("<html><body><table class=\"t\">");
    for r in 0..rows {
        s.push_str("<tr>");
        for c in 0..cols {
            s.push_str(&format!("<td>cell {r}x{c}</td>"));
        }
        s.push_str("</tr>");
    }
    s.push_str("</table></body></html>");
    s
}

fn gen_table_fixed_rowspan(cells: usize, rowspan: usize) -> String {
    // Pathological input for `merge_fixed_redundant_rows` (CSS Tables 3
    // §3.3.2): one fixed-layout row of `cells` cells each spanning
    // `rowspan` tracks. Every implicit track past the first merges, so
    // the old per-track rescan cost dominates this bench.
    let mut s = String::from(
        "<html><body><table style=\"table-layout: fixed; width: 800px; \
         border-collapse: collapse;\"><tr>",
    );
    for c in 0..cells {
        s.push_str(&format!("<td rowspan=\"{rowspan}\">cell {c}</td>"));
    }
    s.push_str("</tr></table></body></html>");
    s
}

fn gen_table_fixed_colspan(rows: usize, colspan: usize) -> String {
    // Pathological input for `merge_fixed_redundant_columns` (CSS
    // Tables 3 §3.3.2): `rows` fixed-layout rows of one cell each,
    // every cell spanning `colspan` tracks. Every implicit track past
    // the first merges, so the old per-track rescan cost dominates this
    // bench.
    let mut s = String::from(
        "<html><body><table style=\"table-layout: fixed; width: 800px; \
         border-collapse: collapse;\">",
    );
    for r in 0..rows {
        s.push_str(&format!("<tr><td colspan=\"{colspan}\">cell {r}</td></tr>"));
    }
    s.push_str("</table></body></html>");
    s
}

fn gen_table_auto_colspan(rows: usize, colspan: usize) -> String {
    // Pathological input for `auto_column_minima`'s spanning-cell loop
    // (`distribute_spanning_min`, and the max-content counterpart
    // `distribute_spanning_max` in `auto_column_maxima`): `rows` rows,
    // one cell per row spanning `colspan` tracks, so every row runs the
    // spanning-cell distribution and the reused
    // `SpanningCandidateScratch` buffers (vis/cands/weights) dominate
    // the allocation profile. `table-layout: auto` is the default and
    // is deliberately left unset: the fixed branch never calls
    // `auto_column_minima`/`auto_column_maxima` — only the §17.5.2.2
    // automatic branch does — so a fixed-layout table could not
    // exercise the changed code. `merge_fixed_redundant_columns` is not
    // wired into the layout path, so the colspan spans survive grid
    // collection and reach the distribution loop intact.
    let mut s = String::from("<html><body><table style=\"border-collapse: collapse;\">");
    for r in 0..rows {
        s.push_str(&format!(
            "<tr><td colspan=\"{colspan}\">cell {r} with some content</td></tr>"
        ));
    }
    s.push_str("</table></body></html>");
    s
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

/// Time `layout()` over `runs` iterations, return median seconds.
fn bench(name: &str, html: &str, runs: usize, warmup: usize) -> f64 {
    let (doc, _) = Document::parse_html(html.as_bytes());
    let ss = bowser_css::parse_stylesheet(CSS);
    let fonts = FixedFontMetrics;
    for _ in 0..warmup {
        std::hint::black_box(bowser_layout::layout(&doc, &[&ss], VP, &fonts));
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let r = bowser_layout::layout(&doc, &[&ss], VP, &fonts);
        // Observe the output so the optimizer cannot eliminate the work.
        std::hint::black_box(r.rect.w);
        times.push(start.elapsed().as_secs_f64());
    }
    let med = median(times);
    println!("  {name:<28} median: {:>9.3} ms", med * 1000.0);
    med
}

/// Time the wave-51 subtree entry on one `<p>` of a large document:
/// the incremental path's per-edit cost, against the `blocks` case's
/// full-document cost on the same generated page.
fn bench_subtree(name: &str, html: &str, runs: usize, warmup: usize) {
    use bowser_layout::LayoutBox;
    use std::collections::HashMap;

    fn find(box_: &LayoutBox, node: bowser_dom::NodeId) -> Option<&LayoutBox> {
        if box_.node == Some(node) {
            return Some(box_);
        }
        box_.children.iter().find_map(|c| find(c, node))
    }
    fn find_parent(box_: &LayoutBox, node: bowser_dom::NodeId) -> Option<&LayoutBox> {
        if box_.children.iter().any(|c| c.node == Some(node)) {
            return Some(box_);
        }
        box_.children.iter().find_map(|c| find_parent(c, node))
    }

    let (doc, _) = Document::parse_html(html.as_bytes());
    let ss = bowser_css::parse_stylesheet(CSS);
    let fonts = FixedFontMetrics;
    let full = bowser_layout::layout(&doc, &[&ss], VP, &fonts);
    let p = doc
        .get_elements_by_tag_name("p")
        .item(1000)
        .expect("bench doc has 1000+ paragraphs");
    // The containing width the full pass offered the target: its
    // parent box's content width (body: no padding/border in the
    // bench stylesheet, so the border box is the content box).
    let parent = find_parent(&full, p).expect("p parent box");
    let containing_width = parent.rect.w;
    assert!(find(&full, p).is_some());
    let run_once = || {
        bowser_layout::layout_subtree_with_intrinsics_and_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &fonts,
            &HashMap::new(),
            (0.0, 0.0),
            &HashMap::new(),
            p,
            containing_width,
        )
        .expect("subtree layout")
    };
    for _ in 0..warmup {
        std::hint::black_box(run_once().rect.w);
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let r = run_once();
        std::hint::black_box(r.rect.w);
        times.push(start.elapsed().as_secs_f64());
    }
    let med = median(times);
    println!("  {name:<28} median: {:>9.3} ms", med * 1000.0);
}

fn main() {
    println!("layout benches (viewport 800x600, FixedFontMetrics):");
    // `BENCH_ONLY=<case>` runs a single case (blocks|inline|deep|iblocks|flex|table|table-fixed|
    // table_colspan_spanning_min|subtree): lets a `time`-style wrapper attribute CPU time per case.
    let only = std::env::var("BENCH_ONLY").ok();
    let run = |name: &str| only.as_deref().is_none_or(|o| o == name);
    if run("subtree") {
        bench_subtree("subtree(1p of 2000p)", &gen_blocks(2000), 7, 2);
    }
    if run("blocks") {
        bench("blocks(2000p)", &gen_blocks(2000), 7, 2);
    }
    if run("inline") {
        bench("inline(400p x 60w)", &gen_inline_text(400, 60), 7, 2);
    }
    if run("deep") {
        bench("deep(3000)", &gen_deep(3000), 7, 2);
    }
    if run("iblocks") {
        bench("inline-blocks(400 x n4)", &gen_inline_blocks(400, 4), 7, 2);
    }
    if run("flex") {
        bench("flex(50r x 20i)", &gen_flex(50, 20), 7, 2);
    }
    if run("table") {
        bench("table(200r x 8c)", &gen_table(200, 8), 7, 2);
    }
    if run("table-fixed") {
        bench(
            "table-fixed(400c x rs1000)",
            &gen_table_fixed_rowspan(400, 1000),
            7,
            2,
        );
        bench(
            "table-fixed(400r x cs1000)",
            &gen_table_fixed_colspan(400, 1000),
            7,
            2,
        );
    }
    if run("table_colspan_spanning_min") {
        // Automatic-layout table (`table-layout: auto` is the default,
        // left unset on purpose): takes the §17.5.2.2 branch, which
        // runs `auto_column_minima` / `auto_column_maxima` and their
        // spanning-cell distribution loops (`distribute_spanning_min` /
        // `distribute_spanning_max`). 200 cells each spanning 4
        // columns, so the per-cell scratch reuse (vis/cands/weights)
        // dominates the allocation profile. A fixed-layout table would
        // never reach these functions: the fixed branch distributes
        // width excess with `distribute_excess_width` only, and the
        // changed functions' other caller (`table_pref_width`) runs
        // only in shrink-to-fit contexts.
        bench(
            "table_colspan_spanning_min(200r x cs4)",
            &gen_table_auto_colspan(200, 4),
            7,
            2,
        );
    }
}
