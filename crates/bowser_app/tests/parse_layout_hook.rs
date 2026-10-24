//! End-to-end tests for the on-demand mid-parse layout wiring through the
//! app's full-page parse path. Parser-inserted scripts run via
//! [`ParserExecutor`] with the parse-phase document shim, so their geometry
//! getters (`offsetWidth`, `getBoundingClientRect`, ...) route through the
//! `ParseLayoutHook` in `bowser_js` — measured values, not the old zero
//! shim. The hook's per-script pass counter travels in
//! [`ParseOutcome::layout_passes`] for the instrumentation assertions below.
//!
//! This exercises the same path a real page load takes (tokenizer +
//! executor), not the `bowser_js` probe harness: the values and pass
//! counts asserted here come from the app's own wiring.

use std::collections::HashMap;

use bowser_app::{PageVm, ParseOutcome, ParserExecutor};
use bowser_dom::{Document, NodeId, NodeKind};

/// Rendered text of the body, skipping `<script>` subtrees (their source
/// text is not rendered output); `document.write` output is included.
fn body_text(document: &Document) -> String {
    let body = document.query_selector("body").unwrap().unwrap();
    fn walk(document: &Document, node: NodeId, out: &mut String) {
        match document.kind(node) {
            NodeKind::Text(text) => out.push_str(text),
            NodeKind::Element(data) if data.tag_name == "script" => {}
            _ => {
                let mut cursor = document.first_child(node);
                while let Some(child) = cursor {
                    walk(document, child, out);
                    cursor = document.next_sibling(child);
                }
            }
        }
    }
    let mut out = String::new();
    walk(document, body, &mut out);
    out
}

/// Drive `html` through the app's scripted full-page parse and return the
/// outcome (including per-script layout pass counts) and the body text.
fn run_full_page(html: &str) -> (ParseOutcome, String) {
    let mut page_vm = PageVm::new();
    let base = bowser_url::parse("https://example.com/", None).unwrap();
    let prefetch = HashMap::new();
    let (document, outcome) = {
        let mut executor = ParserExecutor::new(&mut page_vm, &base, &prefetch, None);
        let (document, _quirks) =
            Document::parse_html_with_executor(html.as_bytes(), true, &mut executor);
        let outcome = executor.outcome();
        (document, outcome)
    };
    (outcome, body_text(&document))
}

#[test]
fn mid_parse_geometry_reads_before_parse_completes() {
    // The script runs mid-parse — before the rest of the document (and
    // any post-parse layout) exists — yet geometry getters return the
    // hook's measured values: the div is 100x50 per its inline style,
    // and the rect width/height are exact.
    let html = concat!(
        "<div id=\"t\" style=\"width:100px;height:50px\"></div>",
        "<script>",
        "var el = document.getElementById('t');",
        "var r = el.getBoundingClientRect();",
        "document.write([el.offsetWidth, el.offsetHeight, r.width, r.height].join('|'));",
        "</script>",
        "<p>later</p>",
    );
    let (outcome, text) = run_full_page(html);
    assert_eq!(text, "100|50|100|50later");
    assert_eq!(outcome.layout_passes.len(), 1);
    assert_eq!(
        outcome.layout_passes[0], 1,
        "one partial layout pass for the repeat reads: the hook caches against the tree's mutation count"
    );
}

#[test]
fn mutation_mid_script_reruns_layout_pass() {
    // The classic forced-sync-layout pattern through the app's full-page
    // path: a style write between two reads bumps the tree's
    // `mutation_count`, so the second read re-runs the partial layout and
    // the per-script pass counter increases (1 -> 2).
    let html = concat!(
        "<div id=\"t\" style=\"width:100px;height:50px\"></div>",
        "<script>",
        "var el = document.getElementById('t');",
        "var before = el.offsetWidth;",
        "el.style.width = '200px';",
        "var after = el.offsetWidth;",
        "document.write([before, after].join('|'));",
        "</script>",
    );
    let (outcome, text) = run_full_page(html);
    assert_eq!(text, "100|200");
    assert_eq!(outcome.layout_passes.len(), 1);
    assert_eq!(
        outcome.layout_passes[0], 2,
        "the mutation between reads must force a second partial layout pass"
    );
}

#[test]
fn display_none_reports_honest_zero_mid_parse() {
    // No box, no geometry: `display: none` measures all zeros mid-parse,
    // exactly as the honest-zero contract documents.
    let html = concat!(
        "<div id=\"t\" style=\"display:none;width:100px\"></div>",
        "<script>document.write(document.getElementById('t').offsetWidth);</script>",
    );
    let (outcome, text) = run_full_page(html);
    assert_eq!(text, "0");
    assert_eq!(outcome.layout_passes.len(), 1);
    // A pass still ran (the read happened); it measured a box-less tree.
    assert_eq!(
        outcome.layout_passes[0], 1,
        "exactly one pass: the read ran, measuring a box-less tree"
    );
}

#[test]
fn script_without_geometry_reads_records_zero_passes() {
    // The counter is instrumentation, not page behavior: a script that
    // never reads geometry must not pay for (or report) a layout pass.
    let html = concat!(
        "<div id=\"t\" style=\"width:100px\"></div>",
        "<script>document.write(document.getElementById('t').id);</script>",
    );
    let (outcome, text) = run_full_page(html);
    assert_eq!(text, "t");
    assert_eq!(outcome.layout_passes, vec![0]);
}

#[test]
fn dom_mutation_between_scripts_invalidates_next_read() {
    // Parser-driven invalidation across scripts: the first script
    // mutates the DOM (appends an element), the second reads geometry.
    // The second script must see the new layout — the appended element
    // measures at its styled size, and its pass counter increases while
    // the mutating script (which reads no geometry) records zero.
    let html = concat!(
        "<div id=\"t\" style=\"width:100px;height:50px\"></div>",
        "<script>",
        "var d = document.createElement('div');",
        "d.id = 'n';",
        "d.style.width = '77px';",
        "d.style.height = '21px';",
        "document.body.appendChild(d);",
        "</script>",
        "<script>",
        "var el = document.getElementById('n');",
        "document.write([el.offsetWidth, el.offsetHeight].join('|'));",
        "</script>",
    );
    let (outcome, text) = run_full_page(html);
    assert_eq!(text, "77|21");
    assert_eq!(
        outcome.layout_passes,
        vec![0, 1],
        "the mutating script reads no geometry (0 passes); the reading script lays out the mutated tree (1 pass)"
    );
}
