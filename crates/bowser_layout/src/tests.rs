//! Unit tests for block and inline layout.
//!
//! Tests build tiny documents with `Document::parse_html`, style them with
//! small stylesheets parsed by the CSS implementation, and assert on the
//! resulting `LayoutBox` tree. The HTML parser always wraps content in
//! `<html><head></head><body>…`, and `<head>` is `display: none` per the UA
//! stylesheet, so `layout.children[0]` is the `<body>` box.

use super::{
    BoxType, FixedFontMetrics, FontFamilyId, FontMetrics, IntrinsicDims, LayoutBox, MarkerBox,
    Size, SnapAlign, TABLE_TEST_RESET_CSS, establishes_fixed_cb, layout, layout_with_intrinsics,
    layout_with_scroll, overflow_axes, scroll_containers, scroll_containers_with_offset,
    snap_areas_for_container,
};
use std::collections::HashMap;

const VP: Size = Size { w: 800.0, h: 600.0 };

/// Standards-mode input for the layout helpers: a doctype-less
/// fragment parses as quirks mode per spec, which would let the
/// quirks rules (e.g. §3.13: a table cell's `height` as border-box)
/// leak into tests whose expectations predate quirks plumbing and
/// match the real WPT files (which all carry `<!doctype html>`).
/// A caller-supplied doctype is left alone, so quirks fixtures keep
/// their mode.
///
/// Mode split, documented (w77fix finding 11): only the helpers in
/// this file that take an HTML string inject a doctype
/// (`layout_html`, `layout_html_markers`). The table helpers in
/// `layout_table.rs` (`layout_table_html`, `layout_table_doc`,
/// `layout_table_html_with`, and the module-local `layout_html`
/// helpers) and `layout_doc_with_intrinsics` /
/// `layout_html_with_moved_node` below do NOT inject: doctype-less
/// input there parses as full quirks mode, so the same fixture
/// gets standards mode in one helper and quirks in another.
/// Pass an explicit doctype in those helpers when the fixture's
/// mode matters; do not "fix" the split by injecting everywhere
/// without re-baselining the table fixtures, which were written
/// against quirks mode.
///
/// Comment-before-doctype edge: leading comments are skipped when
/// looking for a caller doctype. Without that, a fixture starting
/// with a comment followed by a quirks doctype would get a second,
/// prepended doctype shadowing it (the first doctype token wins)
/// and silently re-mode to standards.
fn standards_html(html: &str) -> String {
    let mut rest = html.trim_start();
    while let Some(after) = rest.strip_prefix("<!--") {
        match after.find("-->") {
            Some(end) => rest = after[end + 3..].trim_start(),
            None => break,
        }
    }
    if rest.to_ascii_lowercase().starts_with("<!doctype") {
        html.to_string()
    } else {
        format!("<!DOCTYPE html>{html}")
    }
}

fn layout_html(html: &str, css: &str) -> LayoutBox {
    let html = standards_html(html);
    let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
    let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
    layout(&doc, &[&ss], VP, &FixedFontMetrics)
}

/// Like [`layout_html`], but also returns the `list-style-position:
/// outside` markers, each tagged with its owning list item's DOM node.
fn layout_html_markers(html: &str, css: &str) -> (LayoutBox, Vec<MarkerBox>) {
    let html = standards_html(html);
    let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
    let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
    crate::layout_impl_with_markers(
        &doc,
        &[&ss],
        VP,
        &FixedFontMetrics,
        &HashMap::new(),
        (0.0, 0.0),
        &HashMap::new(),
    )
}

/// Mode split (see `standards_html` above): the document is
/// already parsed, so no doctype can be injected here — a doc
/// parsed from doctype-less input is full quirks mode. Callers
/// that need standards mode must include the doctype in the
/// input they parse.
fn layout_doc_with_intrinsics(
    doc: &bowser_dom::Document,
    css: &str,
    fonts: &dyn FontMetrics,
    intrinsics: &HashMap<bowser_dom::NodeId, IntrinsicDims>,
) -> LayoutBox {
    let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
    layout_with_intrinsics(doc, &[&ss], VP, fonts, intrinsics)
}

#[test]
fn standards_html_keeps_doctype_after_leading_comment() {
    // w77fix finding 11, comment-before-doctype edge: a quirks
    // doctype after a leading comment must survive untouched —
    // the old whitespace-only check prepended a second doctype,
    // shadowing it (first doctype token wins) and silently
    // re-moding the fixture to standards. Fail-before: the
    // quirks input below came back with a prepended doctype.
    let quirks = "<!-- note --><!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\"><table></table>";
    assert_eq!(standards_html(quirks), quirks);
    let quirks_ws = "  <!-- a --><!-- b --><!doctype html><table></table>";
    assert_eq!(standards_html(quirks_ws), quirks_ws);
    let plain = "<!-- note --><table></table>";
    assert_eq!(standards_html(plain), format!("<!DOCTYPE html>{plain}"));
    let plain_ws = "  <table></table>";
    assert_eq!(
        standards_html(plain_ws),
        format!("<!DOCTYPE html>{plain_ws}")
    );
}

/// Metrics where uppercase letters are twice as wide as lowercase, so a
/// `text-transform` visibly changes line breaking.
struct CaseMetrics;

impl FontMetrics for CaseMetrics {
    fn advance(&self, text: &str, font_size: f32, _family: FontFamilyId) -> f32 {
        text.chars()
            .map(|c| {
                let unit = if c.is_ascii_uppercase() { 1.0 } else { 0.5 };
                unit * font_size
            })
            .sum()
    }

    fn line_height(&self, font_size: f32, _family: FontFamilyId) -> f32 {
        1.2 * font_size
    }
}
/// First descendant element of `<body>` with the given tag name.
fn body_child_tag(doc: &bowser_dom::Document, tag: &str) -> bowser_dom::NodeId {
    let root = doc.document_element().expect("document element");
    let body = doc
        .children(root)
        .into_iter()
        .find(|&c| matches!(&doc.kind(c), bowser_dom::NodeKind::Element(d) if d.tag_name == "body"))
        .expect("body");
    doc.children(body)
        .into_iter()
        .find(|&c| matches!(&doc.kind(c), bowser_dom::NodeKind::Element(d) if d.tag_name == tag))
        .expect("tagged child")
}

/// The `<body>` box.
fn body(l: &LayoutBox) -> &LayoutBox {
    &l.children[0]
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.01
}

#[test]
fn block_children_stack_vertically() {
    let l = layout_html(
        r#"<div class="a"><div class="b"></div><div class="c"></div></div>"#,
        ".a { width: 400px; } .b { height: 10px; } .c { height: 20px; }",
    );
    let a = &body(&l).children[0];
    assert_eq!(a.box_type, BoxType::Block);
    assert!(approx(a.rect.w, 400.0));
    let b = &a.children[0];
    let c = &a.children[1];
    assert!(approx(b.rect.y, a.rect.y));
    assert!(approx(c.rect.y, b.rect.y + 10.0));
    assert!(approx(a.rect.h, 30.0));
}

#[test]
fn auto_width_fills_containing_block() {
    let l = layout_html(r#"<div class="a"></div>"#, ".a { height: 5px; }");
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 800.0));
}

#[test]
fn percentage_width_resolves_against_containing_block() {
    let l = layout_html(
        r#"<div class="a"><div class="b"></div></div>"#,
        ".a { width: 400px; } .b { width: 50%; height: 5px; }",
    );
    let b = &body(&l).children[0].children[0];
    assert!(approx(b.rect.w, 200.0));
}

#[test]
fn percentage_margin_resolves_against_containing_block_width() {
    let l = layout_html(
        r#"<div class="a"><div class="b"></div></div>"#,
        ".a { width: 400px; } .b { margin-left: 25%; height: 5px; }",
    );
    let b = &body(&l).children[0].children[0];
    assert!(approx(b.rect.x, 100.0));
    assert!(approx(b.rect.w, 300.0));
}

#[test]
fn auto_margins_center() {
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 200px; height: 5px; margin-left: auto; margin-right: auto; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.x, 300.0));
    assert!(approx(a.rect.w, 200.0));
}

#[test]
fn padding_and_border_add_to_border_box() {
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 100px; padding: 10px; border: 5px solid black; }",
    );
    let a = &body(&l).children[0];
    // 5 + 10 + 100 + 10 + 5
    assert!(approx(a.rect.w, 130.0));
    assert!(approx(a.rect.h, 30.0));
}

#[test]
fn height_percentage_needs_definite_containing_height() {
    let l = layout_html(
        r#"<div class="a"><div class="b"></div></div>"#,
        ".a { height: 200px; } .b { height: 50%; }",
    );
    let b = &body(&l).children[0].children[0];
    assert!(approx(b.rect.h, 100.0));

    // Auto-height parent: the percentage behaves as auto (zero here).
    let l = layout_html(
        r#"<div class="a"><div class="b"></div></div>"#,
        ".b { height: 50%; }",
    );
    let b = &body(&l).children[0].children[0];
    assert!(approx(b.rect.h, 0.0));
}

#[test]
fn sibling_margins_collapse_to_max() {
    let l = layout_html(
        r#"<div class="a"><div class="b"></div><div class="c"></div></div>"#,
        ".b { height: 10px; margin-bottom: 10px; } .c { height: 10px; margin-top: 30px; }",
    );
    let a = &body(&l).children[0];
    let b = &a.children[0];
    let c = &a.children[1];
    assert!(approx(c.rect.y - (b.rect.y + b.rect.h), 30.0));
}

#[test]
fn negative_margins_collapse_by_sum_rule() {
    let l = layout_html(
        r#"<div class="a"><div class="b"></div><div class="c"></div></div>"#,
        ".b { height: 10px; margin-bottom: -5px; } .c { height: 10px; margin-top: 10px; }",
    );
    let a = &body(&l).children[0];
    let b = &a.children[0];
    let c = &a.children[1];
    assert!(approx(c.rect.y - (b.rect.y + b.rect.h), 5.0));
}

#[test]
fn parent_and_first_child_margins_collapse() {
    let l = layout_html(
        r#"<div class="a"><div class="b"></div></div>"#,
        ".a { margin-top: 5px; } .b { height: 10px; margin-top: 15px; }",
    );
    let a = &body(&l).children[0];
    let b = &a.children[0];
    // max(5, 15) = 15 above the parent's border box; the child starts at
    // the parent's content top with no extra gap.
    assert!(approx(a.rect.y, 15.0));
    assert!(approx(b.rect.y, a.rect.y));
}

#[test]
fn parent_and_last_child_margins_collapse() {
    let l = layout_html(
        r#"<div class="a"><div class="b"></div></div><div class="c"></div>"#,
        ".a { margin-bottom: 5px; } .b { height: 10px; margin-bottom: 15px; } .c { height: 4px; margin-top: 7px; }",
    );
    let a = &body(&l).children[0];
    let c = &body(&l).children[1];
    // The parent's auto height excludes the collapsed-through margin…
    assert!(approx(a.rect.h, 10.0));
    // …but it still collapses with the next sibling: max(max(5,15),7) = 15.
    assert!(approx(c.rect.y - (a.rect.y + a.rect.h), 15.0));
}

#[test]
fn border_blocks_parent_child_collapse() {
    let l = layout_html(
        r#"<div class="a"><div class="b"></div></div>"#,
        ".a { margin-top: 5px; border-top-width: 1px; border-top-style: solid; } \
         .b { height: 10px; margin-top: 15px; }",
    );
    let a = &body(&l).children[0];
    let b = &a.children[0];
    assert!(approx(a.rect.y, 5.0));
    assert!(approx(b.rect.y, a.rect.y + 1.0 + 15.0));
}

#[test]
fn border_style_none_zeroes_used_width() {
    // `border-style` defaults to `none`, which forces the used border
    // width to zero (CSS 2.1 §8.5.3): a `border-top-width` alone does not
    // block parent/child margin collapse.
    let l = layout_html(
        r#"<div class="a"><div class="b"></div></div>"#,
        ".a { margin-top: 5px; border-top-width: 1px; } .b { height: 10px; margin-top: 15px; }",
    );
    let a = &body(&l).children[0];
    let b = &a.children[0];
    // Collapsed: max(5, 15) = 15; the 1px width is dropped.
    assert!(approx(a.rect.y, 15.0));
    assert!(approx(b.rect.y, 15.0));
}

#[test]
fn empty_block_margins_collapse_through() {
    let l = layout_html(
        r#"<div class="a"></div><div class="b"></div>"#,
        ".a { margin-top: 8px; margin-bottom: 12px; } .b { height: 5px; }",
    );
    let a = &body(&l).children[0];
    let b = &body(&l).children[1];
    assert!(approx(a.rect.h, 0.0));
    // An empty block's own margins collapse into each other: max(8, 12).
    assert!(approx(a.rect.y, 12.0));
    // …and then into the parent's top margin, so the adjoining set collapses
    // exactly once: nothing separates the empty block from its sibling.
    assert!(approx(b.rect.y, 12.0));
}

#[test]
fn empty_block_between_siblings_collapses_once() {
    let l = layout_html(
        r#"<div class="a"><div class="b"></div><div class="e"></div><div class="c"></div></div>"#,
        ".b { height: 10px; margin-bottom: 20px; } .e { margin-top: 8px; margin-bottom: 12px; } .c { height: 10px; margin-top: 30px; }",
    );
    let a = &body(&l).children[0];
    let b = &a.children[0];
    let e = &a.children[1];
    let c = &a.children[2];
    assert!(approx(e.rect.h, 0.0));
    // All four adjoining margins collapse once: max(20, 8, 12, 30) = 30.
    assert!(approx(c.rect.y - (b.rect.y + b.rect.h), 30.0));
}

#[test]
fn empty_block_first_child_collapses_with_parent_top() {
    let l = layout_html(
        r#"<div class="a"><div class="e"></div><div class="s"></div></div>"#,
        ".e { margin-top: 8px; margin-bottom: 12px; } .s { height: 10px; }",
    );
    let a = &body(&l).children[0];
    let e = &a.children[0];
    let s = &a.children[1];
    // The empty block's through-collapsed margin joins the parent's top
    // margin, so it is consumed there: no gap below the empty block.
    assert!(approx(e.rect.y, a.rect.y));
    assert!(approx(s.rect.y, e.rect.y));
    assert!(approx(a.rect.h, 10.0));
}

#[test]
fn empty_block_last_child_collapses_with_parent_bottom() {
    let l = layout_html(
        r#"<div class="a"><div class="s"></div><div class="e"></div></div><div class="d"></div>"#,
        ".s { height: 10px; } .e { margin-top: 8px; margin-bottom: 12px; } .d { height: 4px; margin-top: 7px; }",
    );
    let a = &body(&l).children[0];
    let s = &a.children[0];
    let e = &a.children[1];
    let d = &body(&l).children[1];
    // The empty block's through-collapsed margin joins the parent's bottom
    // margin instead of sitting above the empty block…
    assert!(approx(e.rect.y, s.rect.y + s.rect.h));
    assert!(approx(a.rect.h, 10.0));
    // …so it collapses with the next sibling: max(max(0, 12), 7) = 12.
    assert!(approx(d.rect.y - (a.rect.y + a.rect.h), 12.0));
}

#[test]
fn auto_height_contains_last_child_margin_box() {
    let l = layout_html(
        r#"<div class="a"><div class="b"></div></div>"#,
        ".a { padding-bottom: 10px; } .b { height: 20px; margin-bottom: 15px; }",
    );
    let a = &body(&l).children[0];
    // CSS 2.1 §10.6.3: the child's bottom margin cannot collapse through the
    // parent's padding, so it stays inside the auto height: 20 + 15 + 10.
    assert!(approx(a.rect.h, 45.0));
}

#[test]
fn empty_block_percentage_height_collapses_through() {
    // A percentage height against an indefinite containing-block height
    // behaves as auto, so the empty block still collapses through like the
    // auto-height case: the adjoining set collapses once.
    let l = layout_html(
        r#"<div class="a"><div class="b"></div><div class="e"></div><div class="c"></div></div>"#,
        ".b { height: 10px; margin-bottom: 20px; } .e { height: 50%; margin-top: 8px; margin-bottom: 12px; } .c { height: 10px; margin-top: 30px; }",
    );
    let a = &body(&l).children[0];
    let b = &a.children[0];
    let e = &a.children[1];
    let c = &a.children[2];
    assert!(approx(e.rect.h, 0.0));
    assert!(approx(c.rect.y - (b.rect.y + b.rect.h), 30.0));

    // …while against a definite height the percentage is real, the block is
    // not collapsible, and its margins place normally on both sides.
    let l = layout_html(
        r#"<div class="a"><div class="b"></div><div class="e"></div><div class="c"></div></div>"#,
        ".a { height: 200px; } .b { height: 10px; margin-bottom: 20px; } .e { height: 50%; margin-top: 8px; margin-bottom: 12px; } .c { height: 10px; margin-top: 30px; }",
    );
    let a = &body(&l).children[0];
    let b = &a.children[0];
    let e = &a.children[1];
    let c = &a.children[2];
    assert!(approx(e.rect.h, 100.0));
    // max(20, 8) + 100 + max(12, 30) = 150.
    assert!(approx(c.rect.y - (b.rect.y + b.rect.h), 150.0));
}

#[test]
fn display_none_skips_subtree() {
    let l = layout_html(
        r#"<div class="a"><div class="b"><div class="c"></div></div><div class="d"></div></div>"#,
        ".b { display: none; } .d { height: 7px; }",
    );
    let a = &body(&l).children[0];
    assert_eq!(a.children.len(), 1);
    assert!(approx(a.rect.h, 7.0));
}

#[test]
fn anonymous_blocks_wrap_mixed_content() {
    let l = layout_html(
        r#"<div class="a">hello <span>world</span><p>para</p></div>"#,
        ".a { font-size: 10px; } p { height: 20px; }",
    );
    let a = &body(&l).children[0];
    assert_eq!(a.children.len(), 2);
    assert_eq!(a.children[0].box_type, BoxType::AnonymousBlock);
    assert_eq!(a.children[1].box_type, BoxType::Block);
    // The anonymous block holds the text node and the span.
    assert_eq!(a.children[0].children.len(), 2);
    assert_eq!(a.children[0].children[0].box_type, BoxType::Text);
    assert_eq!(a.children[0].children[1].box_type, BoxType::Inline);
}

#[test]
fn no_anonymous_block_for_pure_inline_content() {
    let l = layout_html(
        r#"<div class="a">hello <span>world</span></div>"#,
        ".a { font-size: 10px; }",
    );
    let a = &body(&l).children[0];
    assert_eq!(a.children.len(), 2);
    assert_eq!(a.children[0].box_type, BoxType::Text);
    assert_eq!(a.children[1].box_type, BoxType::Inline);
}

#[test]
fn inline_wrapping_splits_lines() {
    // FixedFontMetrics: 6px per char at 10px font size; line height 12px.
    // "aaaa bbbb cccc" = 84px fits in 100px; adding " dddd" would be 114px.
    let l = layout_html(
        r#"<div class="a">aaaa bbbb cccc dddd</div>"#,
        ".a { width: 100px; font-size: 10px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 24.0));
    // One text box spanning two lines: its rect is the union.
    let t = &a.children[0];
    assert_eq!(t.box_type, BoxType::Text);
    assert!(approx(t.rect.w, 84.0));
    assert!(approx(t.rect.h, 24.0));
}

#[test]
fn normal_whitespace_collapses() {
    let l = layout_html(r#"<div class="a">x  y</div>"#, ".a { font-size: 10px; }");
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.text.as_deref(), Some("x y"));
    assert!(approx(t.rect.w, 18.0));
}

#[test]
fn pre_whitespace_is_preserved() {
    let l = layout_html(r#"<pre class="a">x  y</pre>"#, ".a { font-size: 10px; }");
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.text.as_deref(), Some("x  y"));
    assert!(approx(t.rect.w, 24.0));
}

#[test]
fn pre_newlines_force_breaks() {
    let l = layout_html("<pre class=\"a\">ab\ncd</pre>", ".a { font-size: 10px; }");
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 24.0));
}

#[test]
fn text_align_center_offsets_line() {
    let l = layout_html(
        r#"<div class="a">hi</div>"#,
        ".a { width: 100px; font-size: 10px; text-align: center; }",
    );
    let a = &body(&l).children[0];
    let t = &a.children[0];
    // "hi" is 12px wide; (100 - 12) / 2 = 44.
    assert!(approx(t.rect.x, a.rect.x + 44.0));
}

#[test]
fn font_size_lengths_percentages_and_keywords() {
    // % and em resolve against the parent font size.
    let l = layout_html(
        r#"<div class="a"><div class="b">x</div></div>"#,
        ".a { font-size: 20px; } .b { font-size: 150%; }",
    );
    let t = &body(&l).children[0].children[0].children[0];
    assert!(approx(t.rect.h, 36.0)); // 1.2 * 30

    let l = layout_html(
        r#"<div class="a"><div class="b">x</div></div>"#,
        ".a { font-size: 20px; } .b { font-size: 2em; }",
    );
    let t = &body(&l).children[0].children[0].children[0];
    assert!(approx(t.rect.h, 48.0)); // 1.2 * 40

    // Absolute keyword.
    let l = layout_html(r#"<div class="a">x</div>"#, ".a { font-size: large; }");
    let t = &body(&l).children[0].children[0];
    assert!(approx(t.rect.h, 1.2 * 18.0));

    // Relative keywords.
    let l = layout_html(
        r#"<div class="a"><div class="b">x</div></div>"#,
        ".a { font-size: 20px; } .b { font-size: larger; }",
    );
    let t = &body(&l).children[0].children[0].children[0];
    assert!(approx(t.rect.h, 1.2 * 24.0));
}

#[test]
fn rem_uses_root_font_size() {
    let l = layout_html(
        r#"<div class="a">x</div>"#,
        "html { font-size: 10px; } .a { font-size: 2rem; }",
    );
    let t = &body(&l).children[0].children[0];
    assert!(approx(t.rect.h, 1.2 * 20.0));
}

#[test]
fn viewport_units() {
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 50vw; height: 25vh; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 400.0));
    assert!(approx(a.rect.h, 150.0));
}

#[test]
fn inline_block_shrink_to_fit() {
    let l = layout_html(
        r#"<div class="a"><span class="b">hello</span> world</div>"#,
        ".a { width: 200px; font-size: 10px; } .b { display: inline-block; }",
    );
    let a = &body(&l).children[0];
    let ib = &a.children[0];
    assert_eq!(ib.box_type, BoxType::InlineBlock);
    assert!(approx(ib.rect.w, 30.0));
    assert!(approx(ib.rect.h, 12.0));
    // One line: 30 + 6 + 30 = 66px.
    assert!(approx(a.rect.h, 12.0));
}

#[test]
fn deep_dom_does_not_overflow_the_stack() {
    let depth = 5000;
    let mut html = String::new();
    for _ in 0..depth {
        html.push_str("<div>");
    }
    html.push('x');
    for _ in 0..depth {
        html.push_str("</div>");
    }
    let l = layout_html(&html, "div { font-size: 10px; }");
    // Walk to the bottom iteratively and check the text survived.
    let mut b = body(&l);
    for _ in 0..depth {
        b = &b.children[0];
    }
    assert_eq!(b.children[0].box_type, BoxType::Text);
}

#[test]
fn em_lengths_use_element_font_size() {
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { font-size: 10px; margin-left: 2em; width: 10em; height: 5px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.x, 20.0));
    assert!(approx(a.rect.w, 100.0));
}

#[test]
fn rem_falls_back_to_16px() {
    // No font-size on html: rem resolves against the 16px initial value.
    let l = layout_html(r#"<div class="a">x</div>"#, ".a { font-size: 2rem; }");
    let t = &body(&l).children[0].children[0];
    assert!(approx(t.rect.h, 1.2 * 32.0));
}

#[test]
fn small_font_size_keyword() {
    let l = layout_html(r#"<div class="a">x</div>"#, ".a { font-size: small; }");
    let t = &body(&l).children[0].children[0];
    assert!(approx(t.rect.h, 1.2 * 13.0));
}

#[test]
fn float_margin_does_not_collapse() {
    // Float margins never collapse (CSS 2.1 §9.5): the float keeps its own
    // margins and the in-flow sibling stacks as if the float were absent.
    let l = layout_html(
        r#"<div class="a"></div><div class="b"></div>"#,
        ".a { float: left; width: 50px; height: 10px; margin: 5px 0 15px 0; } .b { height: 5px; }",
    );
    let b = body(&l);
    let (a, sib) = (&b.children[0], &b.children[1]);
    let top = a.rect.y;
    // Margin-box top sits at the cursor; the top margin is not collapsed
    // away and the bottom margin does not move the sibling.
    assert!(approx(a.rect.y - 5.0, top - 5.0));
    assert!(approx(sib.rect.y, top - 5.0));
    assert!(approx(a.rect.y + a.rect.h + 15.0, top + 25.0));
}

#[test]
fn float_takes_no_space_in_flow() {
    // Floats take no space in normal flow: the following sibling stacks
    // as if the float were not there, and the float shrink-to-fits.
    let l = layout_html(
        r#"<div class="a">xx</div><div class="b"></div>"#,
        ".a { float: left; height: 5px; } .b { height: 7px; }",
    );
    let b = body(&l);
    let (a, sib) = (&b.children[0], &b.children[1]);
    assert_eq!(a.box_type, BoxType::Block);
    // Shrink-to-fit around "xx": 2 chars at 0.6 * 16px.
    assert!(approx(a.rect.w, 2.0 * 0.6 * 16.0));
    assert!(approx(a.rect.x, sib.rect.x));
    assert!(approx(sib.rect.y, a.rect.y));
}

// ---------------------------------------------------------------------------
// Floats (`float: left | right`, `clear`, BFC containment).
// ---------------------------------------------------------------------------

#[test]
fn left_floats_stack_per_side() {
    // Same-side floats sit beside each other while they fit; a float
    // that does not fit drops below the earlier ones (CSS 2.1 §9.5.1).
    let l = layout_html(
        r#"<div class="w"><div class="a"></div><div class="b"></div><div class="c"></div></div>"#,
        ".w { width: 250px; } .a { float: left; width: 100px; height: 20px; } \
         .b { float: left; width: 100px; height: 20px; } \
         .c { float: left; width: 100px; height: 20px; }",
    );
    let w = &body(&l).children[0];
    let (a, b2, c) = (&w.children[0], &w.children[1], &w.children[2]);
    // .w has no padding/border: content x is the border-box x.
    let cx = w.rect.x;
    assert!(approx(a.rect.x, cx));
    assert!(approx(a.rect.y, w.rect.y));
    assert!(approx(b2.rect.x, cx + 100.0));
    assert!(approx(b2.rect.y, a.rect.y));
    // Only 50px remain beside .a/.b: .c drops below them.
    assert!(approx(c.rect.x, cx));
    assert!(approx(c.rect.y, a.rect.y + 20.0));
}

#[test]
fn left_float_text_wraps() {
    // Text lines beside a left float are shortened and shifted right;
    // lines below the float use the full width again.
    let l = layout_html(
        r#"<div class="w"><div class="fl"></div><p class="t">aa bb cc dd ee ff gg hh ii jj kk ll mm</p></div>"#,
        ".w { width: 120px; font-size: 10px; } .fl { float: left; width: 50px; height: 30px; } \
         .t { margin: 0; }",
    );
    let w = &body(&l).children[0];
    let cx = w.rect.x;
    let fl = &w.children[0];
    assert!(approx(fl.rect.x, cx));
    assert!(approx(fl.rect.w, 50.0));
    let t = &w.children[1];
    let frags = &t.children[0].fragments;
    assert!(!frags.is_empty());
    // First line sits beside the float: shifted right by its width.
    assert!(approx(frags[0].rect.x, cx + 50.0));
    // The last line is below the float's bottom: full width again.
    let last = frags.last().unwrap();
    assert!(last.rect.y >= fl.rect.y + fl.rect.h);
    assert!(approx(last.rect.x, cx));
}

#[test]
fn right_float_sits_at_content_end() {
    // A right float hugs the content-box end; text lines beside it start
    // at the content x and stay left of the float.
    let l = layout_html(
        r#"<div class="w"><div class="fr"></div><p class="t">aa bb cc dd ee ff</p></div>"#,
        ".w { width: 200px; font-size: 10px; } .fr { float: right; width: 60px; height: 30px; } \
         .t { margin: 0; }",
    );
    let w = &body(&l).children[0];
    let cx = w.rect.x;
    let fr = &w.children[0];
    assert!(approx(fr.rect.x, cx + 200.0 - 60.0));
    assert!(approx(fr.rect.y, w.rect.y));
    let t = &w.children[1];
    let frags = &t.children[0].fragments;
    assert!(!frags.is_empty());
    assert!(approx(frags[0].rect.x, cx));
    for fg in frags.iter() {
        if fg.rect.y < fr.rect.y + fr.rect.h {
            assert!(fg.rect.x + fg.rect.w <= cx + 140.0 + 0.01);
        }
    }
}

#[test]
fn text_indent_first_line_fit_subtracts_right_float() {
    // CSS 2.1 §9.4.2/§16.1: floats shorten the line box first;
    // `text-indent` applies relative to the line-box edge. A non-zero
    // first-line indent ignores a *left* float (crbug 1014247) but must
    // still respect a *right* float's intrusion. The 350px atomic fits
    // from the indent position to the content end (350 <= 400 - 20) but
    // not to the float-shortened right edge (350 > 400 - 100 - 20), so
    // the first line must shift below the right float.
    let l = layout_html(
        r#"<div class="w"><div class="fr"></div><p class="t"><span class="ib"></span></p></div>"#,
        ".w { width: 400px; font-size: 10px; } .fr { float: right; width: 100px; height: 30px; } \
         .t { margin: 0; text-indent: 20px; } .ib { display: inline-block; width: 350px; height: 10px; }",
    );
    let w = &body(&l).children[0];
    let cx = w.rect.x;
    let fr = &w.children[0];
    assert!(approx(fr.rect.x, cx + 300.0), "fr.rect={:?}", fr.rect);
    let t = &w.children[1];
    let ib = &t.children[0];
    assert_eq!(ib.box_type, BoxType::InlineBlock);
    // The first line does not fit beside the float: it moves below it.
    assert!(ib.rect.y >= fr.rect.y + fr.rect.h, "ib.rect={:?}", ib.rect);
    // It is still the first line, so it starts at the indent position.
    assert!(approx(ib.rect.x, cx + 20.0));
}

#[test]
fn text_indent_first_line_fit_both_floats_keeps_left_exception() {
    // crbug 1014247: with a non-zero first-line `text-indent`, the line
    // starts at the indent position, so the shift-below-float fit check
    // measures from the indent (`avail_f + dx - indent`), not from the
    // float-narrowed band edge. With left (50) and right (100) floats in
    // a 400px block and a 20px indent, the 260px atomic fits from the
    // indent to the float-shortened right edge (400 - 100 - 20 = 280):
    // it must stay on the first line, not shift below the floats. A
    // formula that drops the left-intrusion re-add (`avail_f - indent`
    // = 230) wrongly shifts it.
    let l = layout_html(
        r#"<div class="w"><div class="fl"></div><div class="fr"></div><p class="t"><span class="ib"></span></p></div>"#,
        ".w { width: 400px; font-size: 10px; } \
         .fl { float: left; width: 50px; height: 30px; } \
         .fr { float: right; width: 100px; height: 30px; } \
         .t { margin: 0; text-indent: 20px; } .ib { display: inline-block; width: 260px; height: 10px; }",
    );
    let w = &body(&l).children[0];
    let cx = w.rect.x;
    let (fl, fr) = (&w.children[0], &w.children[1]);
    assert!(approx(fl.rect.w, 50.0));
    assert!(approx(fr.rect.x, cx + 300.0), "fr.rect={:?}", fr.rect);
    let t = &w.children[2];
    let ib = &t.children[0];
    assert_eq!(ib.box_type, BoxType::InlineBlock);
    // The atomic stays on the first line, beside the floats.
    assert!(ib.rect.y < fl.rect.y + fl.rect.h, "ib.rect={:?}", ib.rect);
    // It starts at the indent position ...
    assert!(approx(ib.rect.x, cx + 20.0), "ib.rect={:?}", ib.rect);
    // ... and never overlaps the right float.
    assert!(
        ib.rect.x + ib.rect.w <= fr.rect.x + 0.01,
        "ib.rect={:?}",
        ib.rect
    );
}

#[test]
fn justify_first_line_indent_adds_back_left_float() {
    // crbug 1014247 (justify mirror): with a non-zero first-line
    // `text-indent`, the line starts at the indent position, so the
    // `text-align: justify` leftover must be measured against the
    // break-time width (`avail_f + dx - indent`), not the raw
    // float-shortened width. Here the 200px block has a 50px left
    // float and a 20px indent: the first line (100 + 6 + 72 = 178px,
    // one gap) justifies against 180px, spreading 2px into the gap.
    // Measuring against `avail_f - indent` (130px) leaves a negative
    // leftover and justifies nothing.
    let l = layout_html(
        r#"<div class="w"><div class="fl"></div><p class="t"><span class="a"></span> <span class="b"></span> <span class="c"></span></p></div>"#,
        ".w { width: 200px; font-size: 10px; } \
         .fl { float: left; width: 50px; height: 30px; } \
         .t { margin: 0; text-align: justify; text-indent: 20px; } \
         .a { display: inline-block; width: 100px; height: 10px; } \
         .b { display: inline-block; width: 72px; height: 10px; } \
         .c { display: inline-block; width: 60px; height: 10px; }",
    );
    let w = &body(&l).children[0];
    let cx = w.rect.x;
    let fl = &w.children[0];
    assert!(approx(fl.rect.w, 50.0));
    let t = &w.children[1];
    let ibs: Vec<_> = t
        .children
        .iter()
        .filter(|c| c.box_type == BoxType::InlineBlock)
        .collect();
    assert_eq!(ibs.len(), 3);
    let b = ibs[1];
    // The second inline-block sits on the first line, beside the float,
    // with the 2px justified gap after the space:
    // cx + indent(20) + a(100) + space(6) + extra(2).
    assert!(b.rect.y < fl.rect.y + fl.rect.h, "b.rect={:?}", b.rect);
    assert!(approx(b.rect.x, cx + 128.0), "b.rect={:?}", b.rect);
}

#[test]
fn float_intruding_mid_line_narrows_line() {
    // A float whose top falls inside a line's band (not at its top) still
    // narrows that line: available width is tested over the line's full
    // `[y, y+height)` band, not just at the line top (CSS 2.1 §9.5).
    // f1 (left, 70x20) sits at the top; f2 (right, 70x20) cannot fit
    // beside it (70 > 120 - 70), so it drops to y=20 — mid-band of the
    // second text line ([12, 24)).
    let l = layout_html(
        r#"<div class="w"><div class="f1"></div><div class="f2"></div><p class="t">aa bb cc dd ee ff gg hh ii jj kk ll</p></div>"#,
        ".w { width: 120px; font-size: 10px; } \
         .f1 { float: left; width: 70px; height: 20px; } \
         .f2 { float: right; width: 70px; height: 20px; } \
         .t { margin: 0; }",
    );
    let w = &body(&l).children[0];
    let cx = w.rect.x;
    let (f1, f2) = (&w.children[0], &w.children[1]);
    assert!(approx(f1.rect.x, cx));
    assert!(approx(f1.rect.y, w.rect.y));
    assert!(approx(f2.rect.x, cx + 50.0));
    assert!(approx(f2.rect.y, w.rect.y + 20.0));
    let t = &w.children[2];
    let frags = &t.children[0].fragments;
    assert!(!frags.is_empty());
    // Per-line extents, grouping the word pieces by their line top.
    let line_extent = |y: f32| {
        let mut x0 = f32::INFINITY;
        let mut x1 = f32::NEG_INFINITY;
        for fg in frags.iter() {
            if approx(fg.rect.y, y) {
                x0 = x0.min(fg.rect.x);
                x1 = x1.max(fg.rect.x + fg.rect.w);
            }
        }
        (x0, x1)
    };
    // Line 1 sits beside the left float: shifted right, 50px wide.
    let (x0, x1) = line_extent(w.rect.y);
    assert!(approx(x0, cx + 70.0));
    assert!(x1 <= cx + 120.0 + 0.01);
    // Line 2's band overlaps both floats, leaving no room: it moves
    // just below f1 (y=20) instead of overlapping f2, and stays within
    // the 50px left of f2.
    let (x0, x1) = line_extent(w.rect.y + 20.0);
    assert!(x0.is_finite(), "a line lands at y=20");
    assert!(approx(x0, cx));
    assert!(x1 <= cx + 50.0 + 0.01);
    // Line 3 is below f1, still beside f2: starts at the content x.
    let (x0, _) = line_extent(w.rect.y + 32.0);
    assert!(x0.is_finite(), "a line lands at y=32");
    assert!(approx(x0, cx));
}

#[test]
fn clear_both_moves_below_floats() {
    // `clear: both` moves the box below the lowest float on either side.
    let l = layout_html(
        r#"<div class="a"></div><div class="b"></div><div class="c"></div>"#,
        ".a { float: left; width: 50px; height: 30px; } \
         .b { float: right; width: 50px; height: 20px; } \
         .c { clear: both; height: 10px; }",
    );
    let b = body(&l);
    let (a, c) = (&b.children[0], &b.children[2]);
    assert!(approx(c.rect.y, a.rect.y + 30.0));
}

#[test]
fn bfc_contains_floats() {
    // `overflow: hidden` establishes a BFC, which contains its floats:
    // the parent grows to the float's bottom (CSS 2.1 §10.6.7). Without
    // a BFC the float hangs out of the heightless parent.
    let l = layout_html(
        r#"<div class="w"><div class="fl"></div></div><div class="v"><div class="fl"></div></div>"#,
        ".fl { float: left; width: 50px; height: 30px; } .w { overflow: hidden; }",
    );
    let b = body(&l);
    let (w, v) = (&b.children[0], &b.children[1]);
    assert!(approx(w.rect.h, 30.0));
    assert!(approx(v.rect.h, 0.0));
}

#[test]
fn bfc_auto_width_min_width_clears_float() {
    // CSS 2.1 §9.5: a BFC root's border box must not overlap a float's
    // margin box. With `width: auto` the box shrinks to the space beside
    // the float, but `min-width` floors that shrink: a 900px min-width
    // beside a 200px float in a 500px container (300px free) cannot fit,
    // so the box clears below the float instead of overlapping it.
    let l = layout_html(
        r#"<div class="c"><div class="f"></div><div class="b"></div></div>"#,
        ".c { width: 500px; } \
         .f { float: left; width: 200px; height: 100px; } \
         .b { overflow: hidden; min-width: 900px; height: 50px; }",
    );
    let c = &body(&l).children[0];
    let (f, b) = (&c.children[0], &c.children[1]);
    // Cleared below the float's margin box: no overlap.
    assert!(approx(b.rect.y, f.rect.y + f.rect.h));
    assert!(b.rect.y >= f.rect.y + f.rect.h);
}

#[test]
fn bfc_auto_width_without_min_width_sits_beside_float() {
    // The companion case: without `min-width`, the auto-width BFC root
    // still shrinks to the space beside the float (no clearance).
    let l = layout_html(
        r#"<div class="c"><div class="f"></div><div class="b"></div></div>"#,
        ".c { width: 500px; } \
         .f { float: left; width: 200px; height: 100px; } \
         .b { overflow: hidden; height: 50px; }",
    );
    let c = &body(&l).children[0];
    let (f, b) = (&c.children[0], &c.children[1]);
    assert!(approx(b.rect.y, f.rect.y));
    assert!(approx(b.rect.x, f.rect.x + f.rect.w));
    assert!(approx(b.rect.w, 300.0));
}

#[test]
fn bfc_post_layout_shift_ignores_non_intruding_floats() {
    // CSS 2.1 §9.5: the post-layout re-test shift target only counts
    // floats that horizontally overlap the child's border box. F0 (left
    // 100x200) and B (right 50x10000) narrow the auto-width BFC to
    // x∈[100,350); A (left 350x100) cannot fit beside F0, drops to
    // y=200, and intrudes into the 300px-tall BFC's body. The BFC must
    // clear A only (y=300), not the tall non-intruding B (y=10000):
    // the old vertical-only target overshot to y=10000.
    //
    // B precedes A in source order on purpose. Under CSS 2.1 §9.5.1
    // rule 5's floated-box clause (a later float's outer top may not
    // be higher than any earlier float's outer top), a B placed after
    // A could not sit at the top at all: its top would be floored to
    // A's top (y=200), beside A at x=350, where it *would*
    // horizontally intrude into the BFC the top point-test leaves
    // 300px wide — a different scenario, and not the non-intruding
    // tall-float case this test pins (Chrome/spec place B at the top
    // only when it comes first, exactly as here). An earlier version
    // of this test had B after A at y=0; that geometry encoded the
    // pre-rule-5 placement and was wrong.
    let l = layout_html(
        r#"<div class="c"><div class="f0"></div><div class="b"></div><div class="a"></div><div class="bfc"><div class="inner"></div></div></div>"#,
        ".c { width: 400px; } \
         .f0 { float: left; width: 100px; height: 200px; } \
         .a { float: left; width: 350px; height: 100px; } \
         .b { float: right; width: 50px; height: 10000px; } \
         .bfc { overflow: hidden; } \
         .inner { height: 300px; }",
    );
    let c = &body(&l).children[0];
    let (f0, b, a, bfc) = (
        &c.children[0],
        &c.children[1],
        &c.children[2],
        &c.children[3],
    );
    // Lock the intended geometry: B spans the full height on the right
    // without horizontally touching the BFC; A drops below F0 into the
    // BFC's band.
    assert!(approx(f0.rect.y, c.rect.y));
    assert!(approx(a.rect.y, c.rect.y + 200.0));
    assert!(approx(b.rect.y, c.rect.y));
    assert!(approx(b.rect.x, c.rect.x + 350.0));
    assert!(approx(bfc.rect.x, c.rect.x + 100.0));
    assert!(approx(bfc.rect.w, 250.0));
    // The BFC clears A (y=300) but not the non-intruding B.
    assert!(approx(bfc.rect.y, c.rect.y + 300.0));
    assert!(approx(bfc.rect.h, 300.0));
}

#[test]
fn bfc_post_layout_shift_clears_float_staircase() {
    // CSS 2.1 §9.5: the post-layout re-test keeps shifting until no
    // float violates — it must not stop after a fixed number of passes.
    // F1 (100x60) narrows the auto-width BFC to x∈[100,400) at cy=0, so
    // the pre-layout point test leaves it in place; the 350px-wide left
    // floats F2..F5 form a staircase below ([60,200), [200,400),
    // [400,700), [700,950)), each intruding into the 100px-tall BFC's
    // body only after the previous shift. The old 3-pass cap stranded
    // the BFC at [700,800), overlapping F5; it must clear every step.
    let l = layout_html(
        r#"<div class="c"><div class="f1"></div><div class="f2"></div><div class="f3"></div><div class="f4"></div><div class="f5"></div><div class="bfc"><div class="inner"></div></div></div>"#,
        ".c { width: 400px; } \
         .f1 { float: left; width: 100px; height: 60px; } \
         .f2 { float: left; width: 350px; height: 140px; } \
         .f3 { float: left; width: 350px; height: 200px; } \
         .f4 { float: left; width: 350px; height: 300px; } \
         .f5 { float: left; width: 350px; height: 250px; } \
         .bfc { overflow: hidden; } \
         .inner { height: 100px; }",
    );
    let c = &body(&l).children[0];
    let bfc = &c.children[5];
    // Lock the intended geometry: the BFC sits beside F1 at the top,
    // and the floats form the staircase.
    let f1 = &c.children[0];
    assert!(approx(f1.rect.x, c.rect.x));
    assert!(approx(f1.rect.y, c.rect.y));
    assert!(approx(bfc.rect.x, c.rect.x + 100.0));
    assert!(approx(bfc.rect.w, 300.0));
    assert!(approx(bfc.rect.h, 100.0));
    // The BFC clears the deepest float (y=950): no §9.5 violation.
    assert!(approx(bfc.rect.y, c.rect.y + 950.0));
    let (bx0, bx1, by0, by1) = (
        bfc.rect.x,
        bfc.rect.x + bfc.rect.w,
        bfc.rect.y,
        bfc.rect.y + bfc.rect.h,
    );
    for f in c.children[0..5].iter() {
        let (fx0, fx1, fy0, fy1) = (f.rect.x, f.rect.x + f.rect.w, f.rect.y, f.rect.y + f.rect.h);
        assert!(
            !(fy0 < by1 && fy1 > by0 && fx0 < bx1 && fx1 > bx0),
            "float at y={} overlaps the BFC",
            f.rect.y
        );
    }
}

#[test]
fn many_floats_lay_out() {
    // B6: float placement borrows the float list instead of cloning it
    // per float (the O(F) clone per float made float-heavy documents
    // quadratic in allocator traffic). 400 floats lay out with correct
    // per-side stacking: 16 per 500px row at 30px each, then wrapping.
    let mut html = String::from("<div class=\"c\">");
    for _ in 0..400 {
        html.push_str("<div class=\"f\"></div>");
    }
    html.push_str("</div>");
    let l = layout_html(
        &html,
        ".c { width: 500px; } .f { float: left; width: 30px; height: 10px; }",
    );
    let c = &body(&l).children[0];
    assert_eq!(c.children.len(), 400);
    let cx = c.rect.x;
    let cy = c.rect.y;
    for (i, f) in c.children.iter().enumerate() {
        let i = i as f32;
        assert!(approx(f.rect.x, cx + (i % 16.0) * 30.0));
        assert!(approx(f.rect.y, cy + (i / 16.0).floor() * 10.0));
    }
}

#[test]
fn table_gets_table_box_flex_becomes_block_grid_gets_grid_box() {
    let l = layout_html(
        r#"<div class="a"></div><div class="b"></div><div class="c"></div><div class="e"><span class="d">x</span></div>"#,
        ".a { display: table; } .b { display: flex; } .c { display: grid; } .d { display: inline-flex; }",
    );
    let kids = &body(&l).children;
    assert_eq!(kids[0].box_type, BoxType::Table);
    assert_eq!(kids[1].box_type, BoxType::Block);
    assert_eq!(kids[2].box_type, BoxType::Grid);
    assert_eq!(kids[3].children[0].box_type, BoxType::InlineBlock);
}

#[test]
fn display_keywords_are_ascii_case_insensitive() {
    let l = layout_html(
        r#"<div class="a"></div><div class="b"></div><div class="c"></div><div class="e"><span class="d">x</span></div>"#,
        ".a { DISPLAY: TABLE; } .b { Display: Flex; } .c { display: GRID; } .d { Display: INLINE-FLEX; }",
    );
    let kids = &body(&l).children;
    assert_eq!(kids[0].box_type, BoxType::Table);
    assert_eq!(kids[1].box_type, BoxType::Block);
    assert_eq!(kids[2].box_type, BoxType::Grid);
    assert_eq!(kids[3].children[0].box_type, BoxType::InlineBlock);
}

// ---- Table layout ----

/// Navigate a laid-out table: body > table > row-group > row > cell.
/// (A caption, when present, is the table's first child, so the row
/// group is located by box type.)
fn table_cell(l: &LayoutBox, row: usize, col: usize) -> &LayoutBox {
    let table = &body(l).children[0];
    assert_eq!(table.box_type, BoxType::Table);
    let group = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::TableRowGroup)
        .expect("table row group");
    &group.children[row].children[col]
}

#[test]
fn table_2x2_grid_alignment() {
    let l = layout_html(
        "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>",
        "td { font-size: 10px; }",
    );
    let table = &body(&l).children[0];
    assert_eq!(table.box_type, BoxType::Table);
    // A block-level table with `width: auto` shrink-to-fits (CSS 2.1
    // §17.5.2.2) instead of filling the containing block: two 6px
    // columns, no table padding/border.
    assert!(approx(table.rect.w, 12.0));
    // Row groups / rows / cells have the right box types.
    assert_eq!(table.children[0].box_type, BoxType::TableRowGroup);
    assert_eq!(table.children[0].children[0].box_type, BoxType::TableRow);

    let c00 = table_cell(&l, 0, 0);
    let c01 = table_cell(&l, 0, 1);
    let c10 = table_cell(&l, 1, 0);
    let c11 = table_cell(&l, 1, 1);
    for c in [c00, c01, c10, c11] {
        assert_eq!(c.box_type, BoxType::TableCell);
    }
    // Columns align: same x within a column, no overlap across columns.
    assert!(approx(c00.rect.x, c10.rect.x));
    assert!(approx(c01.rect.x, c11.rect.x));
    assert!(c00.rect.x + c00.rect.w <= c01.rect.x + 0.01);
    // Rows align: same y within a row, no overlap across rows.
    assert!(approx(c00.rect.y, c01.rect.y));
    assert!(approx(c10.rect.y, c11.rect.y));
    assert!(c00.rect.y + c00.rect.h <= c10.rect.y + 0.01);
    // One line of 10px text is 12px tall; the row is the tallest cell.
    assert!(approx(c00.rect.h, 12.0));
    assert!(approx(c10.rect.y, c00.rect.y + 12.0));
    // Table height covers both rows (plus the table's own zero borders).
    assert!(approx(table.rect.h, 24.0));
}

#[test]
fn table_auto_column_widths_sum_to_table_width() {
    // "short" vs "a much longer cell": the second column gets more width,
    // and the columns tile the table's content width exactly.
    let l = layout_html(
        "<table><tr><td>short</td><td>a much longer cell</td></tr></table>",
        "td { font-size: 10px; }",
    );
    let table = &body(&l).children[0];
    let c0 = table_cell(&l, 0, 0);
    let c1 = table_cell(&l, 0, 1);
    assert!(c1.rect.w > c0.rect.w);
    // Columns tile the table width: last column's right edge meets the
    // table's right edge (no borders/padding/spacing here).
    assert!(approx(c1.rect.x + c1.rect.w, table.rect.x + table.rect.w));
    assert!(approx(c0.rect.x, table.rect.x));
}

#[test]
fn table_colspan_spans_columns() {
    let l = layout_html(
        "<table><tr><td>a</td><td>b</td></tr><tr><td colspan=\"2\">wide</td></tr></table>",
        "td { font-size: 10px; }",
    );
    let c00 = table_cell(&l, 0, 0);
    let c01 = table_cell(&l, 0, 1);
    let wide = table_cell(&l, 1, 0);
    // The colspan cell covers both columns exactly.
    assert!(approx(wide.rect.x, c00.rect.x));
    assert!(approx(wide.rect.w, c00.rect.w + c01.rect.w));
    assert!(approx(wide.rect.x + wide.rect.w, c01.rect.x + c01.rect.w));
}

#[test]
fn table_colspan_includes_border_spacing() {
    let l = layout_html(
        "<table><tr><td>a</td><td>b</td></tr><tr><td colspan=\"2\">wide</td></tr></table>",
        "table { border-spacing: 10px; } td { font-size: 10px; }",
    );
    let c00 = table_cell(&l, 0, 0);
    let c01 = table_cell(&l, 0, 1);
    let wide = table_cell(&l, 1, 0);
    // The colspan cell covers both columns plus the inter-column spacing.
    assert!(approx(wide.rect.x, c00.rect.x));
    assert!(approx(wide.rect.w, c00.rect.w + c01.rect.w + 10.0));
    assert!(approx(wide.rect.x + wide.rect.w, c01.rect.x + c01.rect.w));
}

#[test]
fn table_bare_cells_share_one_anonymous_row() {
    // `display: table-cell` boxes directly under a `display: table` box
    // (no HTML parser to imply the wrappers): consecutive bare cells
    // share one anonymous row per CSS Tables 3 §2.2.1 rule 2.1.
    let l = layout_html(
        r#"<div class="t"><div class="c">a</div><div class="c">b</div></div>"#,
        ".t { display: table; } .c { display: table-cell; font-size: 10px; }",
    );
    let table = &body(&l).children[0];
    assert_eq!(table.box_type, BoxType::Table);
    let rows: Vec<&LayoutBox> = table
        .children
        .iter()
        .filter(|c| c.box_type == BoxType::TableRow)
        .collect();
    assert_eq!(rows.len(), 1);
    let cells: Vec<&LayoutBox> = rows[0]
        .children
        .iter()
        .filter(|c| c.box_type == BoxType::TableCell)
        .collect();
    assert_eq!(cells.len(), 2);
    // Same row: same y, side by side.
    assert!(approx(cells[0].rect.y, cells[1].rect.y));
    assert!(cells[0].rect.x + cells[0].rect.w <= cells[1].rect.x + 0.01);
}

#[test]
fn table_bare_cells_in_row_group_share_one_anonymous_row() {
    let l = layout_html(
        r#"<div class="t"><div class="g"><div class="c">a</div><div class="c">b</div></div></div>"#,
        ".t { display: table; } .g { display: table-row-group; } \
         .c { display: table-cell; font-size: 10px; }",
    );
    let table = &body(&l).children[0];
    assert_eq!(table.box_type, BoxType::Table);
    let group = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::TableRowGroup)
        .expect("table row group");
    let rows: Vec<&LayoutBox> = group
        .children
        .iter()
        .filter(|c| c.box_type == BoxType::TableRow)
        .collect();
    assert_eq!(rows.len(), 1);
    let cells: Vec<&LayoutBox> = rows[0]
        .children
        .iter()
        .filter(|c| c.box_type == BoxType::TableCell)
        .collect();
    assert_eq!(cells.len(), 2);
    assert!(approx(cells[0].rect.y, cells[1].rect.y));
    assert!(cells[0].rect.x + cells[0].rect.w <= cells[1].rect.x + 0.01);
}

#[test]
fn table_border_collapse_does_not_panic() {
    // `border-collapse: collapse` must not crash, with or without cell
    // borders; the grid still aligns.
    for css in [
        "table { border-collapse: collapse; } td { font-size: 10px; }",
        "table { border-collapse: collapse; } td { border: 2px solid black; font-size: 10px; }",
        "table { border-collapse: separate; border-spacing: 5px; } td { font-size: 10px; }",
    ] {
        let l = layout_html(
            "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>",
            css,
        );
        let c00 = table_cell(&l, 0, 0);
        let c10 = table_cell(&l, 1, 0);
        assert!(approx(c00.rect.x, c10.rect.x));
    }
}

#[test]
fn table_separate_border_spacing_insets_cells() {
    let l = layout_html(
        "<table><tr><td>a</td><td>b</td></tr></table>",
        "table { border-spacing: 10px; } td { font-size: 10px; }",
    );
    let table = &body(&l).children[0];
    let c0 = table_cell(&l, 0, 0);
    let c1 = table_cell(&l, 0, 1);
    // 10px between the table edge and the first cell, and between cells.
    assert!(approx(c0.rect.x, table.rect.x + 10.0));
    assert!(approx(c1.rect.x, c0.rect.x + c0.rect.w + 10.0));
    assert!(approx(
        c1.rect.x + c1.rect.w + 10.0,
        table.rect.x + table.rect.w
    ));
}

#[test]
fn table_border_spacing_two_values() {
    // `border-spacing: <horizontal> <vertical>`: 10px between columns,
    // 20px between rows.
    let l = layout_html(
        "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>",
        "table { border-spacing: 10px 20px; } td { font-size: 10px; }",
    );
    let c00 = table_cell(&l, 0, 0);
    let c01 = table_cell(&l, 0, 1);
    let c10 = table_cell(&l, 1, 0);
    assert!(approx(c01.rect.x, c00.rect.x + c00.rect.w + 10.0));
    assert!(approx(c10.rect.y, c00.rect.y + c00.rect.h + 20.0));
}

#[test]
fn table_col_widths_pin_fixed_layout_columns() {
    let l = layout_html(
        "<table><colgroup><col class=\"a\"><col class=\"b\"></colgroup>\
         <tr><td>x</td><td>y</td></tr></table>",
        "table { table-layout: fixed; width: 500px; } \
         col.a { width: 100px; } col.b { width: 200px; } td { font-size: 10px; }",
    );
    let c00 = table_cell(&l, 0, 0);
    let c01 = table_cell(&l, 0, 1);
    // All columns are pinned but the table is wider than the pins, so the
    // 200px of excess distributes proportionally to the pinned widths
    // (CSS Tables 3 §3.9.3.1 rule 2): 100 + 200*100/300, 200 + 200*200/300.
    assert!(approx(c00.rect.w, 166.67));
    assert!(approx(c01.rect.w, 333.33));
}

#[test]
fn table_col_width_beats_first_row_cell_width() {
    // In fixed layout a `col` width takes precedence over the first-row
    // cell's width for that column.
    let l = layout_html(
        "<table><colgroup><col class=\"a\"><col></colgroup>\
         <tr><td class=\"w\">x</td><td>y</td></tr></table>",
        "table { table-layout: fixed; width: 500px; } \
         col.a { width: 100px; } td { font-size: 10px; } .w { width: 300px; }",
    );
    assert!(approx(table_cell(&l, 0, 0).rect.w, 100.0));
}

#[test]
fn table_colgroup_width_spreads_over_span_in_fixed_layout() {
    // A `colgroup` with no `col` children spans its `span` attribute;
    // the group's 120px width is one requirement for the whole span,
    // spread over those columns (60px each, not 120px each). Both columns
    // are pinned but the 500px table is wider, so the 380px of excess
    // distributes proportionally to the pinned widths (CSS Tables 3
    // §3.9.3.1 rule 2): 60 + 380/2 each.
    let l = layout_html(
        "<table><colgroup class=\"g\" span=\"2\"></colgroup>\
         <tr><td>x</td><td>y</td></tr></table>",
        "table { table-layout: fixed; width: 500px; } \
         colgroup.g { width: 120px; } td { font-size: 10px; }",
    );
    assert!(approx(table_cell(&l, 0, 0).rect.w, 250.0));
    assert!(approx(table_cell(&l, 0, 1).rect.w, 250.0));
}

#[test]
fn table_col_width_is_minimum_in_auto_layout() {
    // In automatic layout the `col` width is a floor, not an exact pin.
    let l = layout_html(
        "<table><colgroup><col class=\"a\"></colgroup><tr><td>x</td></tr></table>",
        "table { width: 500px; } col.a { width: 200px; } td { font-size: 10px; }",
    );
    assert!(table_cell(&l, 0, 0).rect.w >= 200.0 - 0.01);
}

#[test]
fn table_colspan_zero_spans_to_end_of_table() {
    let l = layout_html(
        "<table><colgroup><col><col><col></colgroup>\
         <tr><td>a</td><td colspan=\"0\">b</td></tr>\
         <tr><td>c</td><td>d</td><td>e</td></tr></table>",
        "td { font-size: 10px; }",
    );
    // b spans columns 1-2: it starts where d starts and is as wide as
    // d and e together.
    let b = table_cell(&l, 0, 1);
    let d = table_cell(&l, 1, 1);
    let e = table_cell(&l, 1, 2);
    assert!(approx(b.rect.x, d.rect.x));
    assert!(approx(b.rect.w, d.rect.w + e.rect.w));
    assert!(approx(b.rect.x + b.rect.w, e.rect.x + e.rect.w));
}

#[test]
fn table_specified_height_distributes_excess_to_rows() {
    // A specified table height taller than the natural row heights
    // grows every row: 100px over two 12px rows -> 50px each.
    let l = layout_html(
        "<table style=\"height: 100px;\"><tr><td>a</td></tr><tr><td>b</td></tr></table>",
        "td { font-size: 10px; }",
    );
    assert!(approx(table_cell(&l, 0, 0).rect.h, 50.0));
    assert!(approx(table_cell(&l, 1, 0).rect.h, 50.0));
}

/// WPT css/css-tables/absolute-tables-003.html: "When sizing and
/// positioning abspos tables, the specified width is obeyed when the
/// intrinsic width is smaller". Tables consume width/height logically:
/// in a vertical writing mode the table's inline size resolves from
/// `height` and its block size from `width` (so `width: 50px` bounds the
/// horizontal size, which shrink-to-fits to the 104px cell + 4px
/// border-spacing); the `right: 0` offset stays physical. Only the
/// table's own writing mode matters, not the containing block's.
#[test]
fn wpt_absolute_tables_003_vertical_writing_modes() {
    let html = r#"<div class="cb"><table class="table"><tr><td class="cell"><div></div></td></tr></table></div>"#.to_string()
        + r#"<div class="cb vertical"><table class="table horizontal"><tr><td class="cell"><div></div></td></tr></table></div>"#
        + r#"<div class="cb"><table class="table vertical"><tr><td class="cell"><div></div></td></tr></table></div>"#
        + r#"<div class="cb vertical"><table class="table vertical"><tr><td class="cell"><div></div></td></tr></table></div>"#;
    let l = layout_html(
        &html,
        ".cb { position: relative; height: 200px; width: 200px; } \
         .table { position: absolute; width: 50px; right: 0px; } \
         table { border-spacing: 2px; } \
         td { padding: 1px; } \
         .cell { border: 1px solid green; width: 100px; } \
         .cell > div { width: 20px; } \
         .vertical { writing-mode: vertical-lr; } \
         .horizontal { writing-mode: horizontal-tb; }",
    );
    // (expected table width, expected offset-x, expected cell width).
    let expected = [
        (50.0, 150.0, 46.0),
        (50.0, 150.0, 46.0),
        (108.0, 92.0, 104.0),
        (108.0, 92.0, 104.0),
    ];
    for (i, &(tw, tx, cw)) in expected.iter().enumerate() {
        let cb = &body(&l).children[i];
        let table = &cb.children[0];
        assert_eq!(table.box_type, BoxType::Table);
        assert!(
            approx(table.rect.w, tw),
            "subtest {i}: table width {} != {tw}",
            table.rect.w
        );
        assert!(
            approx(table.rect.x - cb.rect.x, tx),
            "subtest {i}: table offset-x {} != {tx}",
            table.rect.x - cb.rect.x
        );
        let group = table
            .children
            .iter()
            .find(|c| c.box_type == BoxType::TableRowGroup)
            .expect("table row group");
        let cell = &group.children[0].children[0];
        assert_eq!(cell.box_type, BoxType::TableCell);
        assert!(
            approx(cell.rect.w, cw),
            "subtest {i}: cell width {} != {cw}",
            cell.rect.w
        );
    }
}

/// WPT css/css-tables/absolute-tables-005.html: "When sizing and
/// positioning abspos tables, the specified height is obeyed when the
/// intrinsic height is smaller". In a vertical writing mode `height`
/// feeds the inline (vertical) axis exactly — 50px, with the 100px cell
/// overflowing — while the block (horizontal) axis shrink-to-fits to
/// the 4px cell + 4px border-spacing; `right: 0` stays physical.
#[test]
fn wpt_absolute_tables_005_vertical_writing_modes() {
    let html = r#"<div class="cb"><table class="table"><tr><td class="cell"><div></div></td></tr></table></div>"#.to_string()
        + r#"<div class="cb vertical"><table class="table horizontal"><tr><td class="cell"><div></div></td></tr></table></div>"#
        + r#"<div class="cb"><table class="table vertical"><tr><td class="cell"><div></div></td></tr></table></div>"#
        + r#"<div class="cb vertical"><table class="table vertical"><tr><td class="cell"><div></div></td></tr></table></div>"#;
    let l = layout_html(
        &html,
        ".cb { position: relative; height: 200px; width: 200px; } \
         .table { position: absolute; height: 50px; right: 0px; } \
         table { border-spacing: 2px; } \
         td { padding: 1px; } \
         .cell { border: 1px solid green; height: 100px; } \
         .cell > div { height: 20px; } \
         .vertical { writing-mode: vertical-lr; } \
         .horizontal { writing-mode: horizontal-tb; }",
    );
    // (expected table width, expected table height, expected offset-x).
    let expected = [
        (8.0, 108.0, 192.0),
        (8.0, 108.0, 192.0),
        (8.0, 50.0, 192.0),
        (8.0, 50.0, 192.0),
    ];
    for (i, &(tw, th, tx)) in expected.iter().enumerate() {
        let cb = &body(&l).children[i];
        let table = &cb.children[0];
        assert_eq!(table.box_type, BoxType::Table);
        assert!(
            approx(table.rect.w, tw),
            "subtest {i}: table width {} != {tw}",
            table.rect.w
        );
        assert!(
            approx(table.rect.h, th),
            "subtest {i}: table height {} != {th}",
            table.rect.h
        );
        assert!(
            approx(table.rect.x - cb.rect.x, tx),
            "subtest {i}: table offset-x {} != {tx}",
            table.rect.x - cb.rect.x
        );
    }
}

/// WPT css/css-tables/absolute-tables-002.html: "the intrinsic width is
/// obeyed when the intrinsic width is larger than specified width".
/// In a vertical writing mode the block (horizontal) content 204px
/// floors the `width:50px` block spec, giving 208 in all four subtests.
#[test]
fn wpt_absolute_tables_002_vertical_writing_modes() {
    let html = r#"<div class="cb"><table class="table"><tr><td class="cell"><div></div></td></tr></table></div>"#.to_string()
        + r#"<div class="cb vertical"><table class="table horizontal"><tr><td class="cell"><div></div></td></tr></table></div>"#
        + r#"<div class="cb"><table class="table vertical"><tr><td class="cell"><div></div></td></tr></table></div>"#
        + r#"<div class="cb vertical"><table class="table vertical"><tr><td class="cell"><div></div></td></tr></table></div>"#;
    let l = layout_html(
        &html,
        ".cb { position: relative; height: 200px; width: 200px; } \
         .table { position: absolute; width: 50px; right: 0px; } \
         table { border-spacing: 2px; } \
         td { padding: 1px; } \
         .cell { border: 1px solid green; width: 100px; } \
         .cell > div { width: 200px; } \
         .vertical { writing-mode: vertical-lr; } \
         .horizontal { writing-mode: horizontal-tb; }",
    );
    // (expected table width, expected offset-x); all four subtests expect 208 / -8.
    for i in 0..4 {
        let cb = &body(&l).children[i];
        let table = &cb.children[0];
        assert_eq!(table.box_type, BoxType::Table);
        assert!(
            approx(table.rect.w, 208.0),
            "subtest {i}: table width {} != 208",
            table.rect.w
        );
        assert!(
            approx(table.rect.x - cb.rect.x, -8.0),
            "subtest {i}: table offset-x {} != -8",
            table.rect.x - cb.rect.x
        );
    }
}

/// WPT css/css-tables/absolute-tables-004.html: "the intrinsic height is
/// obeyed when the intrinsic height is larger than specified height".
/// All four subtests expect 8x208 @ 192: the two vertical-table subtests
/// need vertical intrinsic measurement of the cell content (the 200px
/// `div` inside the `td`), which the column machinery now does via
/// `vertical_cell_min_content` / `vertical_cell_max_content` when
/// `TableGrid::vertical` is set (see `transpose_vwm_table` for the
/// remaining vertical follow-ups).
#[test]
fn wpt_absolute_tables_004() {
    // Mirrors the WPT file's whitespace (text nodes between the cell
    // and the div) to exercise the same box tree the harness sees.
    let html = r#"<div class="cb"><table class="table"><tr><td class="cell">
        <div></div>
      </td></tr></table></div>"#
        .to_string()
        + r#"<div class="cb vertical"><table class="table horizontal"><tr><td class="cell">
        <div></div>
      </td></tr></table></div>"#
        + r#"<div class="cb"><table class="table vertical"><tr><td class="cell">
        <div></div>
      </td></tr></table></div>"#
        + r#"<div class="cb vertical"><table class="table vertical"><tr><td class="cell">
        <div></div>
      </td></tr></table></div>"#;
    let l = layout_html(
        &html,
        ".cb { position: relative; height: 200px; width: 200px; } \
         .table { position: absolute; height: 50px; right: 0px; } \
         table { border-spacing: 2px; } \
         td { padding: 1px; } \
         .cell { border: 1px solid green; height: 100px; } \
         .cell > div { height: 200px; } \
         .vertical { writing-mode: vertical-lr; } \
         .horizontal { writing-mode: horizontal-tb; }",
    );
    // (expected table width, expected table height, expected offset-x):
    // all four WPT subtests expect 8x208 @ 192.
    let expected = [
        (8.0, 208.0, 192.0),
        (8.0, 208.0, 192.0),
        (8.0, 208.0, 192.0),
        (8.0, 208.0, 192.0),
    ];
    for (i, &(tw, th, tx)) in expected.iter().enumerate() {
        let cb = &body(&l).children[i];
        let table = &cb.children[0];
        assert_eq!(table.box_type, BoxType::Table);
        assert!(
            approx(table.rect.w, tw),
            "subtest {i}: table width {} != {tw}",
            table.rect.w
        );
        assert!(
            approx(table.rect.h, th),
            "subtest {i}: table height {} != {th}",
            table.rect.h
        );
        assert!(
            approx(table.rect.x - cb.rect.x, tx),
            "subtest {i}: table offset-x {} != {tx}",
            table.rect.x - cb.rect.x
        );
    }
}

/// Parse `html`, move the element with `id=child_id` from inside
/// `id=outer_id` into the table, and lay out the document.
/// Mode split (see `standards_html` above): this helper parses
/// directly without injecting a doctype, so doctype-less input is
/// full quirks mode. Pass an explicit doctype in `html` when the
/// fixture's mode matters.
fn layout_html_with_moved_node(html: &str, css: &str, outer_id: &str, child_id: &str) -> LayoutBox {
    let (mut doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
    let outer = doc.get_element_by_id(outer_id).expect("outer");
    let child = doc.get_element_by_id(child_id).expect("child");
    let table = doc.get_element_by_id("t").expect("table");
    doc.remove_child(outer, child);
    doc.append_child(table, child);
    let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
    layout(&doc, &[&ss], VP, &FixedFontMetrics)
}

#[test]
fn table_stray_block_child_is_laid_out_not_dropped() {
    // A non-row box directly under a table (what the HTML parser would
    // foster-parent, here moved by DOM surgery) is wrapped in an
    // anonymous table row and cell in place (CSS 2.1 §17.2.1), not
    // silently dropped and not treated as a caption.
    let l = layout_html_with_moved_node(
        "<div id=\"outer\"><table id=\"t\"><tr><td>x</td></tr></table>\
         <div id=\"s\">stray</div></div>",
        "td { font-size: 10px; } #s { font-size: 10px; }",
        "outer",
        "s",
    );
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    // No caption-like block child anymore.
    assert!(table.children.iter().all(|c| c.box_type != BoxType::Block));
    let row = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::TableRow)
        .expect("anonymous row");
    let cell = row
        .children
        .iter()
        .find(|c| c.box_type == BoxType::TableCell)
        .expect("anonymous cell");
    let stray = cell
        .children
        .iter()
        .find(|c| c.box_type == BoxType::Block)
        .expect("stray block kept");
    assert!(stray.rect.h > 0.0);
}

#[test]
fn table_stray_inline_child_is_wrapped_and_laid_out() {
    // An inline run directly under a table is wrapped in an anonymous
    // table row and cell (CSS 2.1 §17.2.1) and laid out, not dropped.
    let l = layout_html_with_moved_node(
        "<div id=\"outer\"><table id=\"t\"><tr><td>x</td></tr></table>\
         <span id=\"s\">hi</span></div>",
        "td { font-size: 10px; } #s { font-size: 10px; }",
        "outer",
        "s",
    );
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    let row = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::TableRow)
        .expect("anonymous row");
    let cell = row
        .children
        .iter()
        .find(|c| c.box_type == BoxType::TableCell)
        .expect("anonymous cell");
    assert!(cell.rect.h > 0.0);
}

#[test]
fn table_row_group_stray_text_becomes_anonymous_row() {
    // Stray text directly under a row group is wrapped in an anonymous
    // row (then an anonymous cell), not dropped.
    let (mut doc, _) = bowser_dom::Document::parse_html(
        "<div id=\"outer\"><table id=\"t\"><tbody id=\"g\"><tr><td>x</td></tr></tbody>\
         <span id=\"s\">hi</span></div>"
            .as_bytes(),
    );
    let outer = doc.get_element_by_id("outer").expect("outer");
    let group = doc.get_element_by_id("g").expect("group");
    let stray = doc.get_element_by_id("s").expect("stray");
    doc.remove_child(outer, stray);
    doc.append_child(group, stray);
    let ss = bowser_css::parse_stylesheet("td { font-size: 10px; } #s { font-size: 10px; }");
    let l = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    let group_box = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::TableRowGroup)
        .expect("row group");
    // Two rows: the real one and the anonymous one holding the text.
    assert_eq!(group_box.children.len(), 2);
    assert!(
        group_box
            .children
            .iter()
            .all(|c| c.box_type == BoxType::TableRow && c.rect.h > 0.0)
    );
    // The anonymous row holds the stray text in an anonymous cell.
    let anon_row = &group_box.children[1];
    assert_eq!(anon_row.children.len(), 1);
    assert_eq!(anon_row.children[0].box_type, BoxType::TableCell);
    assert!(anon_row.children[0].rect.h > 0.0);
}

#[test]
fn table_row_stray_text_becomes_anonymous_cell() {
    // Stray text directly under a row is wrapped in an anonymous cell,
    // not dropped.
    let (mut doc, _) = bowser_dom::Document::parse_html(
        "<div id=\"outer\"><table id=\"t\"><tr id=\"r\"><td>x</td></tr></table>\
         <span id=\"s\">hi</span></div>"
            .as_bytes(),
    );
    let outer = doc.get_element_by_id("outer").expect("outer");
    let row = doc.get_element_by_id("r").expect("row");
    let stray = doc.get_element_by_id("s").expect("stray");
    doc.remove_child(outer, stray);
    doc.append_child(row, stray);
    let ss = bowser_css::parse_stylesheet("td { font-size: 10px; } #s { font-size: 10px; }");
    let l = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    let group_box = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::TableRowGroup)
        .expect("row group");
    let row_box = &group_box.children[0];
    assert_eq!(row_box.box_type, BoxType::TableRow);
    // Two cells: the real td and the anonymous cell holding the text.
    assert_eq!(row_box.children.len(), 2);
    assert!(row_box.children.iter().any(|c| c.rect.h > 0.0));
}

#[test]
fn table_floated_child_is_positioned_not_dropped() {
    // A floated box directly under a table is laid out and placed, with
    // its whole subtree following the float root (the caption pushes the
    // float below y=0, so a stale subtree would be visibly left behind).
    let l = layout_html_with_moved_node(
        "<div id=\"outer\"><table id=\"t\"><caption>cap</caption>\
         <tr><td>x</td></tr></table><div id=\"s\">floatme</div></div>",
        "td { font-size: 10px; } caption { font-size: 10px; } \
         #s { float: left; width: 100px; font-size: 10px; }",
        "outer",
        "s",
    );
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    let float = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::Block && approx(c.rect.w, 100.0))
        .expect("float kept");
    assert!(float.rect.h > 0.0);
    assert!(float.rect.y > 0.0, "float sits below the caption");
    // Every descendant rect sits inside the floated box: the subtree
    // followed the float root instead of staying at the origin.
    fn check_inside(b: &LayoutBox, fx: f32, fy: f32) {
        for c in &b.children {
            assert!(
                c.rect.x >= fx - 0.01 && c.rect.y >= fy - 0.01,
                "descendant {c:?} outside float at ({fx}, {fy})"
            );
            check_inside(c, fx, fy);
        }
    }
    check_inside(float, float.rect.x, float.rect.y);
}

#[test]
fn table_row_height_from_tallest_cell() {
    let l = layout_html(
        "<table><tr><td class=\"tall\">x</td><td>y</td></tr></table>",
        "td { font-size: 10px; } .tall { height: 50px; }",
    );
    let c0 = table_cell(&l, 0, 0);
    let c1 = table_cell(&l, 0, 1);
    // Both cells stretch to the row height (50px from the tall cell).
    assert!(approx(c0.rect.h, 50.0));
    assert!(approx(c1.rect.h, 50.0));
    assert!(approx(c0.rect.y, c1.rect.y));
}

#[test]
fn table_rowspan_occupies_two_rows() {
    let l = layout_html(
        "<table><tr><td rowspan=\"2\">tall</td><td>a</td></tr><tr><td>b</td></tr></table>",
        "td { font-size: 10px; }",
    );
    let table = &body(&l).children[0];
    let tall = table_cell(&l, 0, 0);
    let a = table_cell(&l, 0, 1);
    let b = table_cell(&l, 1, 0);
    // The rowspan cell starts at the first row and covers both rows'
    // heights; the second row's cell sits below the first.
    assert!(approx(tall.rect.y, a.rect.y));
    assert!(approx(tall.rect.h, a.rect.h + b.rect.h));
    assert!(approx(b.rect.y, a.rect.y + a.rect.h));
    assert_eq!(table.children[0].children.len(), 2);
}

#[test]
fn table_row_and_group_rects_cover_cells() {
    let l = layout_html(
        "<table><tbody><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></tbody></table>",
        "td { font-size: 10px; }",
    );
    let table = &body(&l).children[0];
    let group = &table.children[0];
    assert_eq!(group.box_type, BoxType::TableRowGroup);
    let row0 = &group.children[0];
    let row1 = &group.children[1];
    assert_eq!(row0.box_type, BoxType::TableRow);
    let c00 = &row0.children[0];
    let c01 = &row0.children[1];
    let c10 = &row1.children[0];
    // Row rects cover their cells: same top as the first cell, the full
    // column run wide, one row tall.
    assert!(approx(row0.rect.x, c00.rect.x));
    assert!(approx(row0.rect.y, c00.rect.y));
    assert!(approx(row0.rect.h, 12.0));
    assert!(approx(row0.rect.x + row0.rect.w, c01.rect.x + c01.rect.w));
    // Second row sits directly below the first.
    assert!(approx(row1.rect.y, row0.rect.y + row0.rect.h));
    assert!(approx(c10.rect.y, row1.rect.y));
    // The row-group rect is the union of its rows.
    assert!(approx(group.rect.x, row0.rect.x));
    assert!(approx(group.rect.y, row0.rect.y));
    assert!(approx(group.rect.w, row0.rect.w));
    assert!(approx(group.rect.h, row0.rect.h + row1.rect.h));
}

#[test]
fn table_cell_margins_do_not_collapse_through() {
    // Table cells establish a BFC: the inner div's margin-top stays
    // inside the cell instead of collapsing through it.
    let l = layout_html(
        "<table><tr><td><div class=\"in\">x</div></td></tr></table>",
        "td { font-size: 10px; } .in { font-size: 10px; margin-top: 20px; }",
    );
    let cell = table_cell(&l, 0, 0);
    let inner = &cell.children[0];
    assert_eq!(inner.box_type, BoxType::Block);
    assert!(approx(inner.rect.y, cell.rect.y + 20.0));
    assert!(approx(cell.rect.h, 32.0));
}

#[test]
fn table_caption_side_top_and_bottom() {
    for (extra_css, on_top) in [
        ("caption { caption-side: top; }", true),
        ("caption { caption-side: bottom; }", false),
    ] {
        let l = layout_html(
            "<table><caption>cap</caption><tr><td>x</td></tr></table>",
            &format!("td {{ font-size: 10px; }} caption {{ font-size: 10px; }} {extra_css}"),
        );
        let table = &body(&l).children[0];
        let caption = table
            .children
            .iter()
            .find(|c| c.box_type == BoxType::Block)
            .expect("table caption");
        assert_eq!(caption.box_type, BoxType::Block);
        let cell = table_cell(&l, 0, 0);
        if on_top {
            assert!(caption.rect.y + caption.rect.h <= cell.rect.y + 0.01);
        } else {
            assert!(caption.rect.y >= cell.rect.y + cell.rect.h - 0.01);
        }
        // The table covers both the caption and the row.
        let bottom = if on_top {
            cell.rect.y + cell.rect.h
        } else {
            caption.rect.y + caption.rect.h
        };
        assert!(approx(table.rect.y + table.rect.h, bottom));
    }
}

#[test]
fn vertical_align_super_shifts_up() {
    // super shifts by -0.35 * line-height: at 10px font (12px lines) the
    // "y" text sits 4.2px above the baseline-aligned "x" text.
    let l = layout_html(
        r#"<div class="a"><span>x</span><span class="s">y</span></div>"#,
        ".a { font-size: 10px; } .s { vertical-align: super; }",
    );
    let a = &body(&l).children[0];
    let tx = &a.children[0].children[0];
    let ty = &a.children[1].children[0];
    assert!(ty.rect.y < tx.rect.y);
    assert!(approx(tx.rect.y - ty.rect.y, 4.2));
}

#[test]
fn nowrap_does_not_wrap() {
    let l = layout_html(
        r#"<div class="a">aaaa bbbb cccc</div>"#,
        ".a { width: 50px; font-size: 10px; white-space: nowrap; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 12.0));
}

#[test]
fn both_negative_margins_take_minimum() {
    let l = layout_html(
        r#"<div class="a"><div class="b"></div><div class="c"></div></div>"#,
        ".b { height: 10px; margin-bottom: -10px; } .c { height: 10px; margin-top: -5px; }",
    );
    let a = &body(&l).children[0];
    let b = &a.children[0];
    let c = &a.children[1];
    assert!(approx(c.rect.y - (b.rect.y + b.rect.h), -10.0));
}

#[test]
fn percentage_padding_against_containing_block_width() {
    // rect is the border box (padding is inside it); the 25%-of-400px
    // padding shows up as the content/text offset.
    let l = layout_html(
        r#"<div class="a"><div class="b">x</div></div>"#,
        ".a { width: 400px; } .b { padding-left: 25%; font-size: 10px; }",
    );
    let b = &body(&l).children[0].children[0];
    assert!(approx(b.rect.x, 0.0));
    assert!(approx(b.rect.w, 400.0));
    let t = &b.children[0];
    assert!(approx(t.rect.x, 100.0));
}

#[test]
fn single_auto_margin_pushes_right() {
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 200px; height: 5px; margin-left: auto; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.x, 600.0));
    assert!(approx(a.rect.w, 200.0));
}

#[test]
fn inline_block_explicit_width_is_not_shrunk() {
    let l = layout_html(
        r#"<div class="a"><span class="b">hi</span></div>"#,
        ".a { font-size: 10px; } .b { display: inline-block; width: 50px; height: 10px; }",
    );
    let ib = &body(&l).children[0].children[0];
    assert_eq!(ib.box_type, BoxType::InlineBlock);
    assert!(approx(ib.rect.w, 50.0));
}

#[test]
fn font_size_inherited_to_body_text() {
    let l = layout_html(r#"<div>x</div>"#, "html { font-size: 10px; }");
    let t = &body(&l).children[0].children[0];
    assert!(approx(t.rect.h, 12.0));
}

#[test]
fn height_auto_includes_padding() {
    let l = layout_html(
        r#"<div class="a">x</div>"#,
        ".a { font-size: 10px; padding: 10px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 32.0));
}

#[test]
fn line_height_number_multiplier() {
    let l = layout_html(
        r#"<div class="a">x</div>"#,
        ".a { font-size: 10px; line-height: 2; }",
    );
    let t = &body(&l).children[0].children[0];
    assert!(approx(t.rect.h, 20.0));
}

#[test]
fn inline_block_baseline_alignment() {
    // .c wraps to two lines, so its baseline (last line) sits 12px below
    // the single-line .b baseline; .b is pushed down 12px on their shared
    // line box.
    let l = layout_html(
        r#"<div class="a"><span class="b">x</span><span class="c">yy zz</span></div>"#,
        ".a { font-size: 10px; width: 300px; } .b { display: inline-block; } .c { display: inline-block; width: 12px; }",
    );
    let a = &body(&l).children[0];
    let b = &a.children[0];
    let c = &a.children[1];
    assert!(approx(a.rect.h, 24.0));
    assert!(approx(b.rect.y - c.rect.y, 12.0));
}

#[test]
fn text_node_becomes_text_box() {
    let l = layout_html(r#"<div class="a">hello</div>"#, ".a { font-size: 10px; }");
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.box_type, BoxType::Text);
    assert_eq!(t.text.as_deref(), Some("hello"));
}

#[test]
fn display_none_root_yields_empty_box() {
    let l = layout_html(r#"<div></div>"#, "html { display: none; }");
    assert_eq!(l.box_type, BoxType::Block);
    assert!(l.children.is_empty());
    assert_eq!(
        l.rect,
        super::Rect {
            x: 0.0,
            y: 0.0,
            w: 0.0,
            h: 0.0
        }
    );
}

#[test]
fn deep_inline_nesting_does_not_overflow_the_stack() {
    let depth = 5000;
    let mut html = String::from("<div>");
    for _ in 0..depth {
        html.push_str("<span>");
    }
    html.push('x');
    for _ in 0..depth {
        html.push_str("</span>");
    }
    html.push_str("</div>");
    let l = layout_html(&html, "span { font-size: 10px; }");
    let mut b = &body(&l).children[0];
    assert_eq!(b.box_type, BoxType::Block);
    b = &b.children[0];
    for _ in 0..depth {
        assert_eq!(b.box_type, BoxType::Inline);
        b = &b.children[0];
    }
    assert_eq!(b.box_type, BoxType::Text);
    assert_eq!(b.text.as_deref(), Some("x"));
}

#[test]
fn inline_block_with_block_children_wraps_inline_runs() {
    // The inline runs around the block child get anonymous-block wrappers
    // with normal inline layout (12px lines at 10px font), not zero-height
    // rects.
    let l = layout_html(
        r#"<div class="a"><span class="b">a<div class="c">b</div>c</span></div>"#,
        ".a { font-size: 10px; } .b { display: inline-block; } .c { height: 20px; }",
    );
    let ib = &body(&l).children[0].children[0];
    assert_eq!(ib.box_type, BoxType::InlineBlock);
    assert_eq!(ib.children.len(), 3);
    assert_eq!(ib.children[0].box_type, BoxType::AnonymousBlock);
    assert_eq!(ib.children[1].box_type, BoxType::Block);
    assert_eq!(ib.children[2].box_type, BoxType::AnonymousBlock);
    let ta = &ib.children[0].children[0];
    let tc = &ib.children[2].children[0];
    assert_eq!(ta.text.as_deref(), Some("a"));
    assert_eq!(tc.text.as_deref(), Some("c"));
    assert!(approx(ta.rect.h, 12.0));
    assert!(approx(tc.rect.h, 12.0));
    assert!(approx(ib.rect.h, 44.0));
}

#[test]
fn min_width_clamps_used_width() {
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 100px; min-width: 300px; height: 5px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 300.0));
}

#[test]
fn max_width_clamps_used_width() {
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 500px; max-width: 300px; height: 5px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 300.0));
}

#[test]
fn min_height_wins_over_max_height() {
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { height: 100px; min-height: 300px; max-height: 200px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 300.0));
}

#[test]
fn max_height_clamps_auto_height() {
    let l = layout_html(
        r#"<div class="a">x</div>"#,
        ".a { font-size: 10px; max-height: 5px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 5.0));
}

#[test]
fn min_height_applies_to_auto_height() {
    let l = layout_html(
        r#"<div class="a">x</div>"#,
        ".a { font-size: 10px; min-height: 40px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 40.0));
}

#[test]
fn border_box_subtracts_padding_and_border() {
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 100px; height: 100px; padding: 10px; border: 5px solid black; box-sizing: border-box; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 100.0));
    assert!(approx(a.rect.h, 100.0));
}

#[test]
fn border_box_with_min_width() {
    // min-width clamps the border-box width under border-box sizing.
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 100px; min-width: 300px; padding: 10px; box-sizing: border-box; height: 5px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 300.0));
}

#[test]
fn display_contents_generates_no_box() {
    let l = layout_html(
        r#"<div class="a"><span class="c">hello</span><div class="d"></div></div>"#,
        ".a { font-size: 10px; } .c { display: contents; font-size: 20px; } .d { height: 5px; }",
    );
    let a = &body(&l).children[0];
    // No Inline box for the span: its text hoists into the div (wrapped in
    // an anonymous block next to the block-level div.d).
    assert_eq!(a.children.len(), 2);
    assert_eq!(a.children[0].box_type, BoxType::AnonymousBlock);
    let t = &a.children[0].children[0];
    assert_eq!(t.box_type, BoxType::Text);
    assert_eq!(t.text.as_deref(), Some("hello"));
    // Inheritance passes through display:contents: 20px font, 24px line.
    assert!(approx(t.rect.h, 24.0));
    assert_eq!(a.children[1].box_type, BoxType::Block);
}

#[test]
fn display_contents_hoists_element_children() {
    let l = layout_html(
        r#"<div class="a"><div class="c"><div class="e"></div><div class="f"></div></div></div>"#,
        ".e { height: 3px; } .f { height: 4px; } .c { display: contents; }",
    );
    let a = &body(&l).children[0];
    assert_eq!(a.children.len(), 2);
    assert!(approx(a.children[0].rect.h, 3.0));
    assert!(approx(a.children[1].rect.h, 4.0));
    assert!(approx(a.rect.h, 7.0));
}

#[test]
fn vertical_align_is_not_inherited() {
    // The inner span specifies no vertical-align: its text stays on the
    // baseline even inside the superscript outer span, while the outer
    // span's own text shifts up 4.2px (0.35 * 12px line height).
    let l = layout_html(
        r#"<div class="a"><span class="s">x<span class="t">y</span></span></div>"#,
        ".a { font-size: 10px; } .s { vertical-align: super; }",
    );
    let a = &body(&l).children[0];
    let tx = &a.children[0].children[0];
    let ty = &a.children[0].children[1].children[0];
    assert_eq!(tx.text.as_deref(), Some("x"));
    assert_eq!(ty.text.as_deref(), Some("y"));
    assert!(approx(ty.rect.y - tx.rect.y, 4.2));
}

#[test]
fn inline_block_percentage_height_resolves() {
    // The container's definite specified height flows into the
    // inline-block pre-layout: 50% of 100px is 50px.
    let l = layout_html(
        r#"<div class="a"><span class="b"></span></div>"#,
        ".a { height: 100px; font-size: 10px; } .b { display: inline-block; height: 50%; }",
    );
    let ib = &body(&l).children[0].children[0];
    assert_eq!(ib.box_type, BoxType::InlineBlock);
    assert!(approx(ib.rect.h, 50.0));

    // Indefinite container height: the percentage still behaves as auto.
    let l = layout_html(
        r#"<div class="a"><span class="b"></span></div>"#,
        ".a { font-size: 10px; } .b { display: inline-block; height: 50%; }",
    );
    let ib = &body(&l).children[0].children[0];
    assert!(approx(ib.rect.h, 0.0));
}

#[test]
fn line_height_strut_floors_line_box() {
    // The block's own line-height acts as a strut: the line box is 40px
    // even though the child specifies a 12px line height.
    let l = layout_html(
        r#"<div class="a"><span class="b">x</span></div>"#,
        ".a { font-size: 10px; line-height: 40px; } .b { font-size: 10px; line-height: 12px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 40.0));
}

#[test]
fn empty_inline_block_line_gets_strut_height() {
    // A block containing only an empty inline-block still gets the strut:
    // one 12px line at 10px font, not a zero-height line box.
    let l = layout_html(
        r#"<div class="a"><span class="b"></span></div>"#,
        ".a { font-size: 10px; } .b { display: inline-block; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 12.0));
}

#[test]
fn text_indent_offsets_first_line() {
    // "aaaa bbbb" wraps at 50px; only the first line is indented 20px: the
    // union rect starts at the unindented second line but stretches 20px
    // wider (24 + 20) to cover the indented first line.
    let l = layout_html(
        r#"<div class="a">aaaa bbbb</div>"#,
        ".a { width: 50px; font-size: 10px; text-indent: 20px; }",
    );
    let a = &body(&l).children[0];
    let t = &a.children[0];
    assert!(approx(t.rect.x - a.rect.x, 0.0));
    assert!(approx(t.rect.w, 44.0));

    // Percentage indents resolve against the content width: 10% of 200px.
    let l = layout_html(
        r#"<div class="a">hi</div>"#,
        ".a { width: 200px; font-size: 10px; text-indent: 10%; }",
    );
    let a = &body(&l).children[0];
    let t = &a.children[0];
    assert!(approx(t.rect.x - a.rect.x, 20.0));
}

#[test]
fn float_negative_text_indent_no_shift() {
    // CSS 2.1 §16.1: a negative text-indent widens the first line's
    // usable width (`content_w - indent` = 130px here), and overlapping
    // a leading left float is allowed. The 90px atomic inline-block
    // fits in the widened width (90 <= 130) but not in the
    // float-narrowed band (90 > 120 - 50 = 70), so it must stay on the
    // first line at the indent position, not shift below the float —
    // only a float spanning the full band may displace the line
    // (CSS 2.1 §9.5).
    let l = layout_html(
        r#"<div class="w"><div class="fl"></div><p class="t"><span class="ib"></span></p></div>"#,
        ".w { width: 120px; font-size: 10px; } \
         .fl { float: left; width: 50px; height: 30px; } \
         .t { margin: 0; text-indent: -1em; } \
         .ib { display: inline-block; width: 90px; height: 10px; }",
    );
    let w = &body(&l).children[0];
    let cx = w.rect.x;
    let fl = &w.children[0];
    assert!(approx(fl.rect.w, 50.0));
    let t = &w.children[1];
    let ib = &t.children[0];
    assert_eq!(ib.box_type, BoxType::InlineBlock);
    // The first line is not shifted below the float's bottom.
    assert!(ib.rect.y < fl.rect.y + fl.rect.h);
    // It starts at the indent position: 10px left of the content x.
    assert!(approx(ib.rect.x, cx - 10.0));
}

#[test]
fn float_negative_text_indent_right_float_still_subtracts() {
    // CSS 2.1 §9.4.2/§16.1: a negative text-indent widens the first
    // line's usable width, but a right float's intrusion is still
    // subtracted (`avail_f + dx - indent`). In this 400px block the
    // 340px atomic fits from the -20px indent to the content end
    // (420px) but not to the float-shortened right edge (400 - 100 -
    // (-20) = 320px), so the first line must shift below the right
    // float rather than overlap it. A formula that ignores right-float
    // intrusion on indented first lines (`content_w - indent` = 420px)
    // wrongly keeps it on the first line.
    let l = layout_html(
        r#"<div class="w"><div class="fr"></div><p class="t"><span class="ib"></span></p></div>"#,
        ".w { width: 400px; font-size: 10px; } \
         .fr { float: right; width: 100px; height: 30px; } \
         .t { margin: 0; text-indent: -20px; } \
         .ib { display: inline-block; width: 340px; height: 10px; }",
    );
    let w = &body(&l).children[0];
    let cx = w.rect.x;
    let fr = &w.children[0];
    assert!(approx(fr.rect.x, cx + 300.0), "fr.rect={:?}", fr.rect);
    let t = &w.children[1];
    let ib = &t.children[0];
    assert_eq!(ib.box_type, BoxType::InlineBlock);
    // The first line does not fit beside the float: it moves below it
    // instead of overlapping it.
    assert!(ib.rect.y >= fr.rect.y + fr.rect.h, "ib.rect={:?}", ib.rect);
    // It is still the first line, so it starts at the indent position.
    assert!(approx(ib.rect.x, cx - 20.0), "ib.rect={:?}", ib.rect);
    // With nothing beside it now, it never crosses the float's band.
    assert!(ib.rect.x + ib.rect.w <= cx + 400.0 + 0.01);
}

#[test]
fn vertical_align_length_shifts_box() {
    // Positive lengths raise the box above the baseline.
    let l = layout_html(
        r#"<div class="a"><span>x</span><span class="s">y</span></div>"#,
        ".a { font-size: 10px; } .s { vertical-align: 5px; }",
    );
    let a = &body(&l).children[0];
    let tx = &a.children[0].children[0];
    let ty = &a.children[1].children[0];
    assert!(approx(tx.rect.y - ty.rect.y, 5.0));

    // Negative lengths lower it.
    let l = layout_html(
        r#"<div class="a"><span>x</span><span class="s">y</span></div>"#,
        ".a { font-size: 10px; } .s { vertical-align: -5px; }",
    );
    let a = &body(&l).children[0];
    let tx = &a.children[0].children[0];
    let ty = &a.children[1].children[0];
    assert!(approx(ty.rect.y - tx.rect.y, 5.0));
}

#[test]
fn text_align_start_end_follow_direction() {
    // "hi" is 12px wide in a 100px box: right-aligned starts at 88.
    let l = layout_html(
        r#"<div class="a">hi</div>"#,
        ".a { width: 100px; font-size: 10px; text-align: start; direction: rtl; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.children[0].rect.x - a.rect.x, 88.0));

    let l = layout_html(
        r#"<div class="a">hi</div>"#,
        ".a { width: 100px; font-size: 10px; text-align: end; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.children[0].rect.x - a.rect.x, 88.0));

    // start in LTR is left.
    let l = layout_html(
        r#"<div class="a">hi</div>"#,
        ".a { width: 100px; font-size: 10px; text-align: start; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.children[0].rect.x - a.rect.x, 0.0));
}

#[test]
fn plaintext_hebrew_first_paragraph_is_rtl_under_ltr() {
    // `unicode-bidi: plaintext` + `direction: ltr` + `text-align:
    // start`: a Hebrew-first paragraph's base direction is RTL
    // (UAX #9 P2/P3, first strong character), so `start` is the right
    // edge. "אבגדה" is 5 chars * 6px = 30px in a 200px box.
    let l = layout_html(
        r#"<div class="a">אבגדה</div>"#,
        ".a { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; }",
    );
    let a = &body(&l).children[0];
    assert!(
        approx(a.children[0].rect.x - a.rect.x, 170.0),
        "plaintext Hebrew-first x offset {}",
        a.children[0].rect.x - a.rect.x
    );
}

#[test]
fn plaintext_arabic_first_paragraph_is_rtl() {
    // Strong AL decides the same way as strong R (UAX #9 P2).
    let l = layout_html(
        r#"<div class="a">ابتثج</div>"#,
        ".a { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.children[0].rect.x - a.rect.x, 170.0));
}

#[test]
fn plaintext_first_strong_covers_all_rtl_scripts() {
    // The first-strong scan uses the full Unicode R/AL bidi-class
    // table, not Hebrew/Arabic block approximations: a paragraph
    // whose first strong letter is Syriac, Thaana, Nko, Samaritan,
    // Mandaic, Syriac Supplement, Arabic Extended-B, or Adlam is RTL
    // exactly like a Hebrew one. Each string is 5 chars * 6px = 30px
    // in a 200px box, so RTL `start` puts it at offset 170.
    let css = ".a { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; }";
    for (script, text) in [
        ("Syriac", "\u{0710}\u{0712}\u{0713}\u{0715}\u{0717}"),
        ("Thaana", "\u{0780}\u{0781}\u{0782}\u{0783}\u{0784}"),
        ("Nko", "\u{07CA}\u{07CB}\u{07CC}\u{07CD}\u{07CE}"),
        ("Samaritan", "\u{0800}\u{0801}\u{0802}\u{0803}\u{0804}"),
        ("Mandaic", "\u{0840}\u{0841}\u{0842}\u{0843}\u{0844}"),
        (
            "Syriac Supplement",
            "\u{0860}\u{0861}\u{0862}\u{0863}\u{0864}",
        ),
        (
            "Arabic Extended-B",
            "\u{0870}\u{0871}\u{0872}\u{0873}\u{0874}",
        ),
        ("Adlam", "\u{1E900}\u{1E901}\u{1E902}\u{1E903}\u{1E904}"),
    ] {
        let html = format!("<div class=\"a\">{text}</div>");
        let l = layout_html(&html, css);
        let a = &body(&l).children[0];
        assert!(
            approx(a.children[0].rect.x - a.rect.x, 170.0),
            "{script}-first x offset {}",
            a.children[0].rect.x - a.rect.x
        );
    }
}

#[test]
fn plaintext_strong_format_marks_decide_first_strong() {
    // UAX #9 P2 scans by bidi type: the strong format characters
    // count even though they are not alphabetic. RLM (U+200F, class
    // R) and ALM (U+061C, class AL) leading a Latin word make the
    // paragraph RTL — 6 chars * 6px = 36px, offset 164 — while LRM
    // (U+200E, strong L) leading a Hebrew word terminates the scan
    // as LTR, offset 0. (The alphabetic gate alone skipped all
    // three and let the following letter decide, flipping both.)
    let css = ".a { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; }";
    for (name, text, want) in [
        ("RLM-leading", "\u{200F}abcde", 164.0),
        ("ALM-leading", "\u{061C}abcde", 164.0),
        ("LRM-leading", "\u{200E}אבגדה", 0.0),
    ] {
        let html = format!("<div class=\"a\">{text}</div>");
        let l = layout_html(&html, css);
        let a = &body(&l).children[0];
        assert!(
            approx(a.children[0].rect.x - a.rect.x, want),
            "{name} x offset {} (want {want})",
            a.children[0].rect.x - a.rect.x
        );
    }
}

#[test]
fn plaintext_latin_first_paragraph_is_ltr() {
    // Latin-first paragraph: first strong is L, so `start` stays left
    // even though the block is in a plaintext scope.
    let l = layout_html(
        r#"<div class="a">abcde</div>"#,
        ".a { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.children[0].rect.x - a.rect.x, 0.0));
}

#[test]
fn plaintext_neutral_leading_paragraph_first_strong_decides() {
    // Digits and punctuation are weak/neutral: they do not decide the
    // paragraph direction — the Hebrew word after them does (RTL).
    // "-- 123 אבג" measures 12 + 6 + 18 + 6 + 18 = 60px.
    let l = layout_html(
        r#"<div class="a">-- 123 אבג</div>"#,
        ".a { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; }",
    );
    let a = &body(&l).children[0];
    assert!(
        approx(a.children[0].rect.x - a.rect.x, 140.0),
        "neutral-leading x offset {}",
        a.children[0].rect.x - a.rect.x
    );
}

#[test]
fn plaintext_not_set_direction_rules_as_before() {
    // Regression pin: the same Hebrew text without `plaintext` keeps
    // the `direction` behavior — LTR `start` is the left edge, and an
    // explicit `unicode-bidi: isolate` (parsed, no layout behavior
    // this wave) does not turn plaintext on either.
    for css in [
        ".a { width: 200px; font-size: 10px; text-align: start; direction: ltr; }",
        ".a { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: isolate; }",
    ] {
        let l = layout_html(r#"<div class="a">אבגדה</div>"#, css);
        let a = &body(&l).children[0];
        assert!(
            approx(a.children[0].rect.x - a.rect.x, 0.0),
            "non-plaintext x offset for {css}"
        );
    }
    // And under `direction: rtl` with no strong character at all, a
    // plaintext paragraph falls back to the block direction.
    let l = layout_html(
        r#"<div class="a">12345</div>"#,
        ".a { width: 200px; font-size: 10px; text-align: start; direction: rtl; unicode-bidi: plaintext; }",
    );
    let a = &body(&l).children[0];
    assert!(
        approx(a.children[0].rect.x - a.rect.x, 170.0),
        "no-strong fallback x offset {}",
        a.children[0].rect.x - a.rect.x
    );
}

#[test]
fn plaintext_table_styled_cell_resets_scope() {
    // `unicode-bidi: plaintext` is NOT inherited (CSS Writing Modes 3
    // §2.7): it sets the base direction of the bidi paragraphs for
    // which *that box* is the containing block. A styled table cell
    // derives its flag from its own `unicode-bidi` (the block
    // boundary) — the plaintext table's scope stops there, so the
    // Hebrew-first paragraph lays out LTR (`text-align: start`,
    // `direction: ltr`): text at the left edge, not the right (170px)
    // a plaintext paragraph would take. (The anonymous-cell half of
    // this rule is pinned at the forest level by
    // `anonymous_table_wrappers_reset_bidi_plaintext_scope`, since
    // `text-align: start` does not follow paragraph direction through
    // table-internal boxes and box geometry cannot discriminate it.)
    let css = "body { margin: 0; } \
         .t { display: table; width: 200px; table-layout: fixed; direction: ltr; \
              text-align: start; unicode-bidi: plaintext; \
              border-collapse: collapse; font-size: 10px; } \
         .r { display: table-row; } \
         .c { display: table-cell; padding: 0; }";
    let l = layout_html(
        "<div class=\"t\"><div class=\"r\"><div class=\"c\">אבגדה</div></div></div>",
        css,
    );
    let t = &body(&l).children[0];
    let cell = find_box_type(t, BoxType::TableCell).expect("cell box");
    let text = find_box_type(cell, BoxType::Text).expect("text box");
    assert!(
        approx(text.rect.x - cell.rect.x, 0.0),
        "styled cell: LTR paragraph at left, offset {}",
        text.rect.x - cell.rect.x
    );
}

#[test]
fn plaintext_flex_anonymous_item_does_not_inherit_scope() {
    // Same block-boundary rule for anonymous flex items: stray text
    // directly under a plaintext flex container is wrapped in an
    // anonymous block item whose paragraphs follow `direction`, not
    // the first strong character. `flex-direction: column` stretches
    // the item across the container, so the paragraph direction is
    // observable in the text's x offset.
    let l = layout_html(
        "<div id=\"c\">אבגדה</div>",
        "body { margin: 0; } \
         #c { display: flex; flex-direction: column; width: 200px; \
              direction: ltr; text-align: start; unicode-bidi: plaintext; \
              font-size: 10px; }",
    );
    let c = &body(&l).children[0];
    let item = find_box_type(c, BoxType::AnonymousBlock).expect("anonymous item");
    assert!(
        approx(item.rect.w, 200.0),
        "column flex item stretches across: {}",
        item.rect.w
    );
    let text = find_box_type(item, BoxType::Text).expect("text box");
    assert!(
        approx(text.rect.x - item.rect.x, 0.0),
        "anonymous flex item: plaintext must not leak in, offset {}",
        text.rect.x - item.rect.x
    );
}

#[test]
fn plaintext_forced_break_paragraphs_get_own_directions() {
    // Paragraphs split at forced breaks: in one `pre` block the
    // Latin-first line stays left while the Hebrew-first line after
    // the break goes right.
    let l = layout_html(
        "<div class=\"a\">abcde\nאבגדה</div>",
        ".a { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; white-space: pre; }",
    );
    let a = &body(&l).children[0];
    let frags = &a.children[0].fragments;
    assert_eq!(frags.len(), 2, "one fragment per pre line: {frags:?}");
    assert!(
        approx(frags[0].rect.x - a.rect.x, 0.0),
        "line 1 x {}",
        frags[0].rect.x
    );
    assert!(
        approx(frags[1].rect.x - a.rect.x, 170.0),
        "line 2 x {}",
        frags[1].rect.x
    );
}

#[test]
fn plaintext_prewrap_soft_wrap_keeps_one_paragraph_direction() {
    // UAX #9 P1: paragraphs split only at paragraph separators,
    // never at soft wraps. This Hebrew-first `pre-wrap` paragraph
    // soft-wraps twice in a 60px box (6px/char at 10px): the pieces
    // are "אבגדה " (36px), "abcde " (36px), "fghij" (30px). Lines 2
    // and 3 start with Latin words, but they are the SAME paragraph,
    // so all three lines keep the Hebrew-derived RTL direction and
    // right-align (`start` = right): offsets 24, 24, 30. (Splitting
    // paragraphs at `break_before` — which soft wraps also set —
    // flipped lines 2/3 to LTR, offset 0.)
    let l = layout_html(
        "<div class=\"a\">אבגדה abcde fghij</div>",
        ".a { width: 60px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; white-space: pre-wrap; }",
    );
    let a = &body(&l).children[0];
    let frags = &a.children[0].fragments;
    assert_eq!(frags.len(), 3, "one fragment per pre-wrap piece: {frags:?}");
    for (i, want) in [24.0, 24.0, 30.0].iter().enumerate() {
        assert!(
            approx(frags[i].rect.x - a.rect.x, *want),
            "wrapped line {} x {} (want {want})",
            i + 1,
            frags[i].rect.x - a.rect.x
        );
    }
}

#[test]
fn plaintext_prewrap_forced_break_starts_new_paragraph() {
    // The counterpart: a `\n` in `pre-wrap` text IS a paragraph
    // separator, so the Hebrew-first line after it derives its own
    // RTL direction while the Latin-first line before it stays left.
    let l = layout_html(
        "<div class=\"a\">abcde\nאבגדה</div>",
        ".a { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; white-space: pre-wrap; }",
    );
    let a = &body(&l).children[0];
    let frags = &a.children[0].fragments;
    assert_eq!(frags.len(), 2, "one fragment per pre-wrap line: {frags:?}");
    assert!(
        approx(frags[0].rect.x - a.rect.x, 0.0),
        "line 1 x {}",
        frags[0].rect.x
    );
    assert!(
        approx(frags[1].rect.x - a.rect.x, 170.0),
        "line 2 x {}",
        frags[1].rect.x
    );
}

#[test]
fn plaintext_does_not_cross_into_descendant_block() {
    // Chat-container shape (CSS Writing Modes 3 §2.7): `plaintext` on
    // the outer block sets the base direction only of paragraphs for
    // which the OUTER block is the containing block. `unicode-bidi`
    // is not inherited, so the inner block container computes
    // `normal`: its Hebrew paragraph resolves against the inner
    // block's own (inherited) `direction: ltr`, and `text-align:
    // start` is the LEFT edge — as in Chrome. (An earlier version of
    // this test pinned the sticky-scope behavior and expected the
    // inner block to right-align; that was the bug.)
    let l = layout_html(
        r#"<div class="o"><div>אבגדה</div><div>abcde</div></div>"#,
        ".o { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; }",
    );
    let o = &body(&l).children[0];
    let he = &o.children[0];
    let la = &o.children[1];
    assert!(
        approx(he.children[0].rect.x - he.rect.x, 0.0),
        "inner Hebrew x offset {}",
        he.children[0].rect.x - he.rect.x
    );
    assert!(approx(la.children[0].rect.x - la.rect.x, 0.0));
}

#[test]
fn plaintext_inner_block_with_own_plaintext_derives_own_direction() {
    // The descendant block is not cut off from plaintext — it just
    // needs its OWN `unicode-bidi: plaintext`. With it, its
    // Hebrew-first paragraph derives RTL exactly as a top-level
    // plaintext block does.
    let l = layout_html(
        r#"<div class="o"><div class="i">אבגדה</div></div>"#,
        ".o { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; } .i { unicode-bidi: plaintext; }",
    );
    let o = &body(&l).children[0];
    let he = &o.children[0];
    assert!(
        approx(he.children[0].rect.x - he.rect.x, 170.0),
        "inner own-plaintext Hebrew x offset {}",
        he.children[0].rect.x - he.rect.x
    );
}

#[test]
fn plaintext_explicit_normal_on_inner_block_resets_outer_scope() {
    // An explicit `unicode-bidi: normal` on the inner block states
    // what the initial value already means after the block boundary:
    // no plaintext paragraphs inside it, whatever the outer block
    // sets. Its Hebrew text follows `direction: ltr` (left).
    let l = layout_html(
        r#"<div class="o"><div class="i">אבגדה</div></div>"#,
        ".o { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; } .i { unicode-bidi: normal; }",
    );
    let o = &body(&l).children[0];
    let he = &o.children[0];
    assert!(
        approx(he.children[0].rect.x - he.rect.x, 0.0),
        "inner normal Hebrew x offset {}",
        he.children[0].rect.x - he.rect.x
    );
}

#[test]
fn plaintext_propagates_across_inline_and_anonymous_boxes() {
    // Inline boxes share the containing block, so plaintext still
    // governs text nested in a plain inline element of a plaintext
    // block.
    let l = layout_html(
        r#"<div class="a"><span>אבגדה</span></div>"#,
        ".a { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; }",
    );
    let a = &body(&l).children[0];
    let span = &a.children[0];
    assert!(
        approx(span.children[0].rect.x - a.rect.x, 170.0),
        "inline-nested Hebrew x offset {}",
        span.children[0].rect.x - a.rect.x
    );
    // Anonymous blocks likewise share the scope: a plaintext block's
    // own direct text, wrapped in an anonymous block because a block
    // sibling follows, still derives RTL.
    let l = layout_html(
        r#"<div class="a">אבגדה<div>abcde</div></div>"#,
        ".a { width: 200px; font-size: 10px; text-align: start; direction: ltr; unicode-bidi: plaintext; }",
    );
    let a = &body(&l).children[0];
    let anon = &a.children[0];
    assert!(
        approx(anon.children[0].rect.x - anon.rect.x, 170.0),
        "anonymous-wrapped Hebrew x offset {}",
        anon.children[0].rect.x - anon.rect.x
    );
}

#[test]
fn logical_margin_resolves_at_box_build() {
    // LTR: margin-inline-start -> left.
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { margin-inline-start: 10px; width: 100px; height: 5px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.x, 10.0));
    // RTL: margin-inline-start -> right. The containing block (body) is
    // LTR, so per CSS 2.1 §10.3.3 the over-constrained margin-right is
    // ignored (the element's own direction does not govern this): the
    // 100px block sits at x=0. (If it mapped to margin-left instead,
    // x would be 10.)
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { direction: rtl; margin-inline-start: 10px; width: 100px; height: 5px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.x, 0.0));
    // A physical longhand declared later still wins at box build.
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { margin-inline-start: 40px; margin-left: 10px; width: 100px; height: 5px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.x, 10.0));
}

#[test]
fn logical_padding_and_border_at_box_build() {
    // RTL: padding-inline-start -> right, border-inline-end-width ->
    // left: border box = 100 + 20 + 4.
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { direction: rtl; width: 100px; height: 5px; \
         padding-inline-start: 20px; border-inline-end-width: 4px; \
         border-inline-end-style: solid; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 124.0));
}

#[test]
fn logical_inset_positions_box_at_box_build() {
    // LTR: inset-inline-start -> left.
    let l = layout_html(
        r#"<div class="mid"><div class="abs"></div></div>"#,
        ".mid { position: relative; width: 400px; height: 300px; } \
         .abs { position: absolute; inset-inline-start: 50px; top: 30px; width: 100px; height: 60px; }",
    );
    let mid = &body(&l).children[0];
    let abs = &mid.children[0];
    assert!(approx(abs.rect.x - mid.rect.x, 50.0));
    assert!(approx(abs.rect.y - mid.rect.y, 30.0));
    // RTL: inset-inline-start -> right: the 100px box ends 50px from
    // the mid's right edge.
    let l = layout_html(
        r#"<div class="mid"><div class="abs"></div></div>"#,
        ".mid { position: relative; width: 400px; height: 300px; } \
         .abs { position: absolute; direction: rtl; inset-inline-start: 50px; top: 30px; width: 100px; height: 60px; }",
    );
    let mid = &body(&l).children[0];
    let abs = &mid.children[0];
    assert!(approx(mid.rect.x + 400.0 - (abs.rect.x + 100.0), 50.0));
}

#[test]
fn inline_size_sets_used_width_at_box_build() {
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { inline-size: 200px; height: 5px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 200.0));
    // A physical width declared later wins over inline-size.
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { inline-size: 200px; width: 150px; height: 5px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 150.0));
}

#[test]
fn vertical_align_top_without_baseline_anchors() {
    // A lone vertical-align: top fragment aligns to the top of the line
    // box (the 40px strut), instead of collapsing to the baseline.
    let l = layout_html(
        r#"<div class="a"><span class="b"></span></div>"#,
        ".a { font-size: 10px; line-height: 40px; } .b { display: inline-block; height: 10px; vertical-align: top; }",
    );
    let a = &body(&l).children[0];
    let ib = &a.children[0];
    assert!(approx(a.rect.h, 40.0));
    assert!(approx(ib.rect.y - a.rect.y, 0.0));
}

#[test]
fn vertical_align_middle_without_baseline_anchors() {
    // CSS 2.1 §10.8.1: `vertical-align: middle` aligns the box's vertical
    // midpoint to the baseline plus half the x-height (i.e. baseline −
    // x_height/2 in y-down coordinates), NOT to the line-box midpoint.
    // With the corrected split_line_height (half-leading + ascent):
    // font 10px/40px, (a,d)=(9.6,2.4), half-leading=14, baseline at 23.6
    // from line top; x_height = 0.5em = 5px, align point at 21.1; the 10px
    // fragment centers there: y = 16.1.
    let l = layout_html(
        r#"<div class="a"><span class="b"></span></div>"#,
        ".a { font-size: 10px; line-height: 40px; } .b { display: inline-block; height: 10px; vertical-align: middle; }",
    );
    let a = &body(&l).children[0];
    let ib = &a.children[0];
    assert!(approx(a.rect.h, 40.0));
    assert!(approx(ib.rect.y - a.rect.y, 16.1));
}

#[test]
fn deep_inline_nesting_rect_union_is_single_pass() {
    // Complexity guard: the inline/text rect union used to re-scan the
    // fragment list once per inline box (O(fragments x nesting depth)).
    // Rects are now accumulated bottom-up in a single pass. Instead of a
    // wall-clock bound (which false-fails on loaded machines), assert the
    // operation count directly: the union pass visits every fragment
    // exactly once (a quadratic re-scan would visit ~depth^2/2).
    let depth = 3000;
    let mut html = String::from(r#"<div class="a">"#);
    for _ in 0..depth {
        html.push_str("<span>x");
    }
    for _ in 0..depth {
        html.push_str("</span>");
    }
    html.push_str("</div>");
    let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
    let ss = bowser_css::parse_stylesheet(".a { font-size: 10px; white-space: nowrap; }");
    let ua = bowser_css::default_ua_stylesheet();
    let Some((mut forest, root)) =
        super::boxes::build_box_tree(&doc, &[&ss], &ua, VP.w, VP.h, &HashMap::new())
    else {
        panic!("no root box");
    };
    super::layout::layout_forest(&mut forest, root, &FixedFontMetrics);
    // 3000 text boxes, one fragment each.
    assert_eq!(forest.union_frag_visits, depth);
    let (l, _) = super::layout::convert_forest(&mut forest, root);
    let a = &body(&l).children[0];
    // Walk down to the innermost span, then its text.
    let mut b = &a.children[0];
    for _ in 1..depth {
        b = &b.children[1];
    }
    let text = &b.children[0];
    assert_eq!(text.box_type, BoxType::Text);
    // Each "x" advances 6px on one line; the innermost text is last.
    assert!(approx(text.rect.x - a.rect.x, (depth - 1) as f32 * 6.0));
    // The outermost span's rect covers every fragment.
    assert!(approx(a.children[0].rect.w, depth as f32 * 6.0));
}

#[test]
fn deeply_nested_inline_blocks_memoize_preferred_width() {
    // Perf guard: shrink-to-fit used to re-walk each inline-block's whole
    // subtree for preferred widths (O(n^2) for nesting); 1600 nested
    // inline-blocks took 822ms in release. Preferred widths are now
    // memoized per layout pass (0.10s at the same depth in release).
    // Bounded depth so this stays fast in debug too; the wall-clock
    // bound is a tripwire against dropping the memoization.
    let depth = 800;
    let mut html = String::new();
    for _ in 0..depth {
        html.push_str(r#"<span class="ib">"#);
    }
    html.push('x');
    for _ in 0..depth {
        html.push_str("</span>");
    }
    let start = std::time::Instant::now();
    let l = layout_html(&html, ".ib { display: inline-block; font-size: 10px; }");
    let elapsed = start.elapsed();
    let mut b = body(&l);
    for _ in 0..depth {
        b = &b.children[0];
        assert_eq!(b.box_type, BoxType::InlineBlock);
        // Shrink-to-fit around the single "x" (6px advance).
        assert!(approx(b.rect.w, 6.0));
    }
    assert_eq!(b.children[0].box_type, BoxType::Text);
    assert!(elapsed.as_secs() < 5);
}

#[test]
fn text_fragments_record_per_line_placement() {
    // FixedFontMetrics: 6px per char at 10px font size; line height 12px.
    let l = layout_html(
        r#"<div class="a">aaaa bbbb cccc dddd</div>"#,
        ".a { width: 100px; font-size: 10px; }",
    );
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.box_type, BoxType::Text);
    // One fragment per laid-out Frag: "aaaa", " ", "bbbb", " ", "cccc" on
    // line 1, "dddd" on line 2 (the line-breaking space is trimmed away).
    assert_eq!(t.fragments.len(), 6);
    let widths: Vec<f32> = t.fragments.iter().map(|f| f.rect.w).collect();
    for (w, want) in widths.iter().zip([24.0, 6.0, 24.0, 6.0, 24.0, 24.0]) {
        assert!(approx(*w, want));
    }
    // First five fragments on line 1, last on line 2: distinct rects and
    // baselines in absolute page coordinates.
    for f in &t.fragments[..5] {
        assert!(approx(f.rect.y, t.rect.y));
        assert!(approx(f.baseline, t.rect.y + 9.6));
    }
    let last = &t.fragments[5];
    assert!(approx(last.rect.y, t.rect.y + 12.0));
    assert!(approx(last.baseline, t.rect.y + 21.6));
}

#[test]
fn single_line_text_has_one_fragment() {
    let l = layout_html(r#"<div class="a">hello</div>"#, ".a { font-size: 10px; }");
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.fragments.len(), 1);
    assert!(approx(t.fragments[0].rect.w, 30.0));
    assert!(approx(t.fragments[0].rect.w, t.rect.w));
}

#[test]
fn non_text_boxes_have_no_fragments() {
    let l = layout_html(
        r#"<div class="a">x<p class="b">y</p></div>"#,
        ".a { font-size: 10px; } .b { height: 20px; }",
    );
    assert!(l.fragments.is_empty());
    let a = &body(&l).children[0];
    assert!(a.fragments.is_empty());
    // The anonymous wrapper holds the text; it is not a text box itself.
    let anon = &a.children[0];
    assert_eq!(anon.box_type, BoxType::AnonymousBlock);
    assert!(anon.fragments.is_empty());
    assert!(!anon.children[0].fragments.is_empty());
}

#[test]
fn node_back_pointers() {
    let l = layout_html(
        r#"<div class="a">x<p class="b">y</p>z</div>"#,
        ".a { font-size: 10px; } .b { height: 20px; }",
    );
    let a = &body(&l).children[0];
    assert!(a.node.is_some());
    // Anonymous boxes have no DOM node.
    let anon = &a.children[0];
    assert_eq!(anon.box_type, BoxType::AnonymousBlock);
    assert_eq!(anon.node, None);
    // Element and text boxes point back at distinct DOM nodes.
    let tx = &anon.children[0];
    assert_eq!(tx.box_type, BoxType::Text);
    assert!(tx.node.is_some());
    let p = &a.children[1];
    assert!(p.node.is_some());
    assert_ne!(a.node, p.node);
    assert_ne!(tx.node, p.node);
}

#[test]
fn text_transform_uppercase() {
    let l = layout_html(
        r#"<div class="a">hello world</div>"#,
        ".a { font-size: 10px; text-transform: uppercase; }",
    );
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.text.as_deref(), Some("HELLO WORLD"));
}

#[test]
fn text_transform_lowercase_and_capitalize() {
    let l = layout_html(
        r#"<div class="a">HeLLo WoRLD</div>"#,
        ".a { font-size: 10px; text-transform: lowercase; }",
    );
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.text.as_deref(), Some("hello world"));

    let l = layout_html(
        r#"<div class="a">hello world</div>"#,
        ".a { font-size: 10px; text-transform: capitalize; }",
    );
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.text.as_deref(), Some("Hello World"));
}

#[test]
fn text_transform_applies_before_line_breaking() {
    // CaseMetrics: uppercase chars are twice as wide as lowercase.
    // "ab cd" is 25px lowercase (fits in 30px) but 45px uppercase.
    let css = ".a { width: 30px; font-size: 10px; }";
    let (doc, _) = bowser_dom::Document::parse_html(r#"<div class="a">ab cd</div>"#.as_bytes());
    let l = layout_doc_with_intrinsics(&doc, css, &CaseMetrics, &HashMap::new());
    assert!(approx(body(&l).children[0].rect.h, 12.0));

    let css = ".a { width: 30px; font-size: 10px; text-transform: uppercase; }";
    let l = layout_doc_with_intrinsics(&doc, css, &CaseMetrics, &HashMap::new());
    let a = &body(&l).children[0];
    // The transform ran before breaking: the wider uppercase text wraps.
    assert!(approx(a.rect.h, 24.0));
    let t = &a.children[0];
    assert_eq!(t.text.as_deref(), Some("AB CD"));
    assert_eq!(t.fragments.len(), 2);
    assert!(approx(t.fragments[0].rect.w, 20.0));
    assert!(approx(t.fragments[1].rect.w, 20.0));
}

#[test]
fn intrinsics_size_replaced_elements() {
    let (doc, _) = bowser_dom::Document::parse_html(r#"<img class="i">"#.as_bytes());
    let img = body_child_tag(&doc, "img");
    let mut map = HashMap::new();
    map.insert(img, IntrinsicDims::from_size(100.0, 50.0));

    // Block-level replaced element takes its box size from the map.
    let l = layout_doc_with_intrinsics(&doc, ".i { display: block; }", &FixedFontMetrics, &map);
    let b = &body(&l).children[0];
    assert_eq!(b.box_type, BoxType::Block);
    assert!(approx(b.rect.w, 100.0));
    assert!(approx(b.rect.h, 50.0));
    // The node back-pointer is the map key.
    assert_eq!(b.node, Some(img));

    // CSS width/height override the intrinsic size.
    let l = layout_doc_with_intrinsics(
        &doc,
        ".i { display: block; width: 40px; height: 10px; }",
        &FixedFontMetrics,
        &map,
    );
    let b = &body(&l).children[0];
    assert!(approx(b.rect.w, 40.0));
    assert!(approx(b.rect.h, 10.0));
}

#[test]
fn intrinsics_missing_entry_is_zero() {
    let (doc, _) = bowser_dom::Document::parse_html(r#"<img class="i">"#.as_bytes());
    let l = layout_doc_with_intrinsics(
        &doc,
        ".i { display: block; }",
        &FixedFontMetrics,
        &HashMap::new(),
    );
    let b = &body(&l).children[0];
    assert!(approx(b.rect.w, 0.0));
    assert!(approx(b.rect.h, 0.0));
}

#[test]
fn intrinsics_size_inline_replaced_element() {
    // An inline <img> is an atomic inline-level box (modeled as
    // inline-block) and still takes its size from the map.
    let (doc, _) = bowser_dom::Document::parse_html(r#"<img class="i">"#.as_bytes());
    let img = body_child_tag(&doc, "img");
    let mut map = HashMap::new();
    map.insert(img, IntrinsicDims::from_size(100.0, 50.0));
    let l = layout_doc_with_intrinsics(&doc, "", &FixedFontMetrics, &map);
    let b = &body(&l).children[0];
    assert_eq!(b.box_type, BoxType::InlineBlock);
    assert!(approx(b.rect.w, 100.0));
    assert!(approx(b.rect.h, 50.0));
}

#[test]
fn prewrap_preserves_spaces_and_breaks_on_newlines() {
    // FixedFontMetrics: 6px per char at 10px font size.
    let l = layout_html(
        "<div class=\"a\">a  b\ncd</div>",
        ".a { white-space: pre-wrap; width: 200px; font-size: 10px; }",
    );
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.box_type, BoxType::Text);
    let text = t.text.as_deref().unwrap();
    assert_eq!(text, "a  b\ncd");
    // Two visual lines: "a  b" and "cd" (forced break on \n).
    assert_eq!(t.fragments.len(), 2);
    assert_eq!(&text[t.fragments[0].text_range.clone()], "a  b");
    assert_eq!(&text[t.fragments[1].text_range.clone()], "cd");
    // Spaces survive verbatim: the first fragment is 4 chars wide.
    assert!(approx(t.fragments[0].rect.w, 24.0));
    // The forced break starts a new line below the first.
    assert!(t.fragments[1].rect.y > t.fragments[0].rect.y);
}

#[test]
fn prewrap_soft_wraps_at_spaces() {
    let l = layout_html(
        "<div class=\"a\">aaa bbb ccc</div>",
        ".a { white-space: pre-wrap; width: 30px; font-size: 10px; }",
    );
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.box_type, BoxType::Text);
    let text = t.text.as_deref().unwrap();
    // "aaa " is 24px; "aaa bbb" (42px) does not fit in 30px, so each word
    // wraps onto its own line with its trailing space kept.
    assert_eq!(t.fragments.len(), 3);
    let slices: Vec<&str> = t
        .fragments
        .iter()
        .map(|f| &text[f.text_range.clone()])
        .collect();
    assert_eq!(slices, ["aaa ", "bbb ", "ccc"]);
    let ys: Vec<f32> = t.fragments.iter().map(|f| f.rect.y).collect();
    assert!(ys[0] < ys[1] && ys[1] < ys[2]);
}

#[test]
fn prewrap_in_flex_item() {
    let l = layout_html(
        "<div class=\"c\">a  b\ncd</div>",
        ".c { display: flex; white-space: pre-wrap; }",
    );
    // The text run is wrapped in an anonymous block flex item (flexbox
    // §4); the text node lives inside it.
    let w = &body(&l).children[0].children[0];
    assert_eq!(w.box_type, BoxType::AnonymousBlock);
    let t = &w.children[0];
    assert_eq!(t.box_type, BoxType::Text);
    let text = t.text.as_deref().unwrap();
    // The flex text breaker honors pre-wrap too: spaces preserved, \n
    // forces a break, ranges slice the transformed text.
    assert_eq!(t.fragments.len(), 2);
    assert_eq!(&text[t.fragments[0].text_range.clone()], "a  b");
    assert_eq!(&text[t.fragments[1].text_range.clone()], "cd");
    assert!(t.fragments[1].rect.y > t.fragments[0].rect.y);
}

#[test]
fn preline_in_flex_item() {
    // `min-width: 0` on the item defeats the automatic minimum size (the
    // full text advance), so the item shrinks to the 30px container and
    // the text soft-wraps.
    let l = layout_html(
        "<div class=\"c\"><span class=\"i\">a  b\ncd efg hij</span></div>",
        ".c { display: flex; white-space: pre-line; width: 30px; font-size: 10px; } \
         .i { min-width: 0px; }",
    );
    // Container > anonymous wrapper > inline span > text.
    let t = &body(&l).children[0].children[0].children[0].children[0];
    assert_eq!(t.box_type, BoxType::Text);
    let text = t.text.as_deref().unwrap();
    // The flex text breaker honors pre-line too: spaces collapse, \n
    // forces a break, and the remainder soft-wraps at spaces within the
    // 30px item width.
    assert_eq!(text, "a b\ncd efg hij");
    let slices: Vec<&str> = t
        .fragments
        .iter()
        .map(|f| &text[f.text_range.clone()])
        .collect();
    assert_eq!(slices, ["a b", "cd", "efg", "hij"]);
    let ys: Vec<f32> = t.fragments.iter().map(|f| f.rect.y).collect();
    assert!(ys[0] < ys[1] && ys[1] < ys[2] && ys[2] < ys[3]);
}

#[test]
fn preline_collapses_spaces_but_breaks_on_newlines() {
    // FixedFontMetrics: 6px per char at 10px font size.
    let l = layout_html(
        "<div class=\"a\">a  b\tc\ndd</div>",
        ".a { white-space: pre-line; width: 200px; font-size: 10px; }",
    );
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.box_type, BoxType::Text);
    let text = t.text.as_deref().unwrap();
    // Space/tab runs collapse to one space; the newline survives as a
    // forced break (unlike `pre`, this is not `Pre` anymore).
    assert_eq!(text, "a b c\ndd");
    assert_eq!(t.fragments.len(), 2);
    let slices: Vec<&str> = t
        .fragments
        .iter()
        .map(|f| &text[f.text_range.clone()])
        .collect();
    assert_eq!(slices, ["a b c", "dd"]);
    // The forced break starts a new line below the first.
    assert!(t.fragments[1].rect.y > t.fragments[0].rect.y);
}

#[test]
fn preline_soft_wraps_at_spaces() {
    let l = layout_html(
        "<div class=\"a\">aaa bbb ccc</div>",
        ".a { white-space: pre-line; width: 30px; font-size: 10px; }",
    );
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.box_type, BoxType::Text);
    let text = t.text.as_deref().unwrap();
    // 30px fits "aaa" (18px) but not "aaa bbb"; each word wraps onto its
    // own line, and the space at the wrap point is trimmed (unlike
    // pre-wrap, which keeps it).
    assert_eq!(t.fragments.len(), 3);
    let slices: Vec<&str> = t
        .fragments
        .iter()
        .map(|f| &text[f.text_range.clone()])
        .collect();
    assert_eq!(slices, ["aaa", "bbb", "ccc"]);
    let ys: Vec<f32> = t.fragments.iter().map(|f| f.rect.y).collect();
    assert!(ys[0] < ys[1] && ys[1] < ys[2]);
}

#[test]
fn preline_preserves_consecutive_newlines() {
    let l = layout_html(
        "<div class=\"a\">ab\n\ncd</div>",
        ".a { white-space: pre-line; width: 200px; font-size: 10px; }",
    );
    let a = &body(&l).children[0];
    let t = &a.children[0];
    let text = t.text.as_deref().unwrap();
    // Segment breaks are not collapsible for pre-line: each newline is a
    // preserved line feed (CSS text-3 §4.1.3), so `ab\n\ncd` is three
    // lines with an empty line in the middle.
    assert_eq!(text, "ab\n\ncd");
    assert_eq!(t.fragments.len(), 3);
    let slices: Vec<&str> = t
        .fragments
        .iter()
        .map(|f| &text[f.text_range.clone()])
        .collect();
    assert_eq!(slices, ["ab", "", "cd"]);
    let ys: Vec<f32> = t.fragments.iter().map(|f| f.rect.y).collect();
    assert!(ys[0] < ys[1] && ys[1] < ys[2]);
    // Three lines at 12px line height.
    assert!(approx(a.rect.h, 36.0));
}

#[test]
fn zero_width_space_is_a_break_opportunity() {
    // FixedFontMetrics: 6px per char at 10px font size. As a single
    // fragment "aaa\u{200b}bbb" is 42px and overflows a 30px block on one
    // line; U+200B is a wrapping-point marker (CSS text-3 §5), so the line
    // breaker may wrap after it.
    let l = layout_html(
        "<div class=\"a\">aaa\u{200b}bbb</div>",
        ".a { width: 30px; font-size: 10px; }",
    );
    let a = &body(&l).children[0];
    let t = &a.children[0];
    assert_eq!(t.box_type, BoxType::Text);
    let text = t.text.as_deref().unwrap();
    // "aaa", the zero-width marker, "bbb": the ranges tile the text.
    assert_eq!(t.fragments.len(), 3);
    let slices: Vec<&str> = t
        .fragments
        .iter()
        .map(|f| &text[f.text_range.clone()])
        .collect();
    assert_eq!(slices, ["aaa", "\u{200b}", "bbb"]);
    // The marker contributes no advance and leaves no visible gap.
    assert!(approx(t.fragments[1].rect.w, 0.0));
    // Two lines at 12px line height: the break lands after the marker.
    assert!(approx(a.rect.h, 24.0));
    assert!(t.fragments[2].rect.y > t.fragments[0].rect.y);
}

#[test]
fn text_ranges_partition_text() {
    let l = layout_html(
        "<div class=\"a\">aaaa bbbb</div>",
        ".a { width: 100px; font-size: 10px; }",
    );
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.box_type, BoxType::Text);
    let text = t.text.as_deref().unwrap();
    // "aaaa", " ", "bbbb": the ranges tile the text with no gaps.
    assert_eq!(t.fragments.len(), 3);
    let joined: String = t
        .fragments
        .iter()
        .map(|f| &text[f.text_range.clone()])
        .collect();
    assert_eq!(joined, text);
    assert_eq!(&text[t.fragments[0].text_range.clone()], "aaaa");
    assert_eq!(&text[t.fragments[1].text_range.clone()], " ");
    assert_eq!(&text[t.fragments[2].text_range.clone()], "bbbb");
}

#[test]
fn normal_span_wraps_inside_pre_wrap_block() {
    // The span is `white-space: normal` inside a `pre-wrap` block: its
    // spaces are break opportunities, so the text wraps to two lines
    // instead of rendering as one overflowing line.
    let l = layout_html(
        r#"<div class="c"><span class="s">aaa bbb ccc ddd eee fff ggg hhh</span></div>"#,
        ".c { white-space: pre-wrap; width: 100px; font-size: 10px; } .s { white-space: normal; }",
    );
    let c = &body(&l).children[0];
    // "aaa bbb ccc ddd" (90px) then "eee fff ggg hhh" (90px): two 12px lines.
    assert!(approx(c.rect.h, 24.0));
    let t = &c.children[0].children[0];
    assert!(t.fragments.iter().any(|f| f.rect.y > t.fragments[0].rect.y));
}

#[test]
fn pre_wrap_span_in_normal_block_preserves_spaces() {
    // Reverse direction: a `pre-wrap` span inside a `normal` block keeps
    // its spaces verbatim and breaks only at its own forced breaks.
    let l = layout_html(
        "<div class=\"c\"><span class=\"s\">aaa bbb\nccc</span></div>",
        ".c { white-space: normal; width: 100px; font-size: 10px; } .s { white-space: pre-wrap; }",
    );
    let c = &body(&l).children[0];
    // One forced break: two 12px lines.
    assert!(approx(c.rect.h, 24.0));
    let t = &c.children[0].children[0];
    // One fragment per pre-fit piece; the space survives (no trimming).
    assert_eq!(t.fragments.len(), 2);
    let text = t.text.as_deref().unwrap();
    assert_eq!(&text[t.fragments[0].text_range.clone()], "aaa bbb");
    assert_eq!(&text[t.fragments[1].text_range.clone()], "ccc");
}

#[test]
fn inline_border_box_excludes_margin() {
    // An inline box's border box covers border/padding/content; margins
    // offset placement but sit outside it (CSS 2.1 §8.6).
    let l = layout_html(
        r#"<div class="c"><span class="s">hi</span></div>"#,
        ".c { width: 200px; } .s { margin-left: 10px; padding-left: 5px; }",
    );
    let c = &body(&l).children[0];
    let s = &c.children[0];
    assert_eq!(s.box_type, BoxType::Inline);
    // Border box starts after the 10px margin; the 5px padding is inside.
    assert!(approx(s.rect.x, c.rect.x + 10.0));
    // "hi" at 16px (0.6 advance) = 19.2px + 5px padding.
    assert!(approx(s.rect.w, 19.2 + 5.0));
}

#[test]
fn nested_inline_box_propagates_spacing() {
    // A nested inline box's border/padding propagates to its ancestors:
    // unions are stored widened, so the outer box covers the inner box,
    // not just the shared fragments.
    let l = layout_html(
        r#"<div class="c"><em class="e"><strong class="s">hi</strong></em></div>"#,
        ".e { padding-left: 4px; } .s { padding-left: 6px; }",
    );
    let c = &body(&l).children[0];
    let e = &c.children[0];
    let s = &e.children[0];
    // Inner border box: text plus its own 6px padding.
    assert!(approx(s.rect.w, 19.2 + 6.0));
    // Outer border box: inner box plus the outer 4px padding.
    assert!(approx(e.rect.w, s.rect.w + 4.0));
    assert!(approx(e.rect.x, c.rect.x));
}

#[test]
fn column_cross_axis_auto_margins_center() {
    // Cross-axis auto margins absorb positive free space before
    // `align-self` (flexbox §8.3): two auto margins center the item.
    let l = layout_html(
        r#"<div class="c"><div class="i">x</div></div>"#,
        ".c { display: flex; flex-direction: column; width: 200px; } \
         .i { width: 100px; margin-left: auto; margin-right: auto; }",
    );
    let c = &body(&l).children[0];
    let i = &c.children[0];
    assert!(approx(i.rect.w, 100.0));
    assert!(approx(i.rect.x, c.rect.x + 50.0));
}

#[test]
fn whitespace_collapses_across_text_nodes() {
    // The trailing space of one text node and the leading space of the
    // next collapse to a single space (CSS text-3 §4.1.2).
    let l = layout_html(
        r#"<div class="c">a <span> b</span></div>"#,
        ".c { width: 200px; }",
    );
    let c = &body(&l).children[0];
    let t1 = &c.children[0];
    let t2 = &c.children[1].children[0];
    // "a", " ", "b" at 9.6px each = 28.8; an uncollapsed pair of spaces
    // would total 38.4.
    let w: f32 = t1
        .fragments
        .iter()
        .chain(t2.fragments.iter())
        .map(|f| f.rect.w)
        .sum();
    assert!(approx(w, 28.8));
}

#[test]
fn intrinsic_height_border_box_keeps_content_height() {
    // A replaced element with `height: auto` uses its intrinsic height as
    // the *content* height; `box-sizing: border-box` does not strip
    // padding/border from it (there is no specified height to reinterpret).
    let (doc, _) = bowser_dom::Document::parse_html(r#"<img class="i">"#.as_bytes());
    let img = body_child_tag(&doc, "img");
    let mut map = HashMap::new();
    map.insert(img, IntrinsicDims::from_size(100.0, 50.0));
    let l = layout_doc_with_intrinsics(
        &doc,
        ".i { display: block; box-sizing: border-box; padding-top: 10px; padding-bottom: 10px; }",
        &FixedFontMetrics,
        &map,
    );
    let b = &body(&l).children[0];
    // Content height stays 50 (intrinsic); the border box adds the padding.
    assert!(approx(b.rect.h, 70.0));
}

#[test]
fn nested_inline_block_shifts_apply_once() {
    // Deferred shifts (`record_shift` + one `apply_pending_shifts` pass)
    // move each subtree exactly once: a nested inline-block must land at
    // its parent's content origin, not shifted once per nesting level.
    let l = layout_html(
        r#"<div class="c"><div class="a"><div class="b">x</div></div></div>"#,
        ".a { display: inline-block; } .b { display: inline-block; width: 50px; height: 20px; }",
    );
    let c = &body(&l).children[0];
    let a = &c.children[0];
    let b = &a.children[0];
    assert_eq!(b.box_type, BoxType::InlineBlock);
    assert!(approx(b.rect.x, a.rect.x));
    assert!(approx(b.rect.y, a.rect.y));
    assert!(approx(b.rect.w, 50.0));
}

// ---------------------------------------------------------------------------
// Grid layout (`display: grid`).
// ---------------------------------------------------------------------------

/// The grid container: the first child of `<body>`.
fn grid_box(l: &LayoutBox) -> &LayoutBox {
    let g = &body(l).children[0];
    assert_eq!(g.box_type, BoxType::Grid);
    g
}

#[test]
fn grid_2x2_fixed_tracks() {
    let l = layout_html(
        r#"<div class="g"><div class="i"></div><div class="i"></div><div class="i"></div><div class="i"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px; } .i { height: 20px; }",
    );
    let g = grid_box(&l);
    let items = &g.children;
    assert_eq!(items.len(), 4);
    assert!(approx(items[0].rect.x, 0.0));
    assert!(approx(items[0].rect.y, 0.0));
    assert!(approx(items[1].rect.x, 100.0));
    assert!(approx(items[1].rect.y, 0.0));
    assert!(approx(items[2].rect.x, 0.0));
    assert!(approx(items[2].rect.y, 20.0));
    assert!(approx(items[3].rect.x, 100.0));
    assert!(approx(items[3].rect.y, 20.0));
    // Two implicit auto rows of 20px each.
    assert!(approx(g.rect.h, 40.0));
}

#[test]
fn grid_fr_columns_share_free_space() {
    let l = layout_html(
        r#"<div class="g"><div class="i"></div><div class="i"></div></div>"#,
        ".g { display: grid; grid-template-columns: 1fr 2fr; width: 300px; }",
    );
    let g = grid_box(&l);
    let items = &g.children;
    assert_eq!(items.len(), 2);
    assert!(approx(items[0].rect.w, 100.0));
    assert!(approx(items[1].rect.x, 100.0));
    assert!(approx(items[1].rect.w, 200.0));
}

#[test]
fn grid_gap_separates_tracks() {
    let l = layout_html(
        r#"<div class="g"><div class="i"></div><div class="i"></div><div class="i"></div><div class="i"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px; gap: 10px; } .i { height: 20px; }",
    );
    let g = grid_box(&l);
    let items = &g.children;
    assert!(approx(items[1].rect.x, 110.0));
    assert!(approx(items[2].rect.x, 0.0));
    assert!(approx(items[2].rect.y, 30.0));
    assert!(approx(items[3].rect.x, 110.0));
    assert!(approx(items[3].rect.y, 30.0));
    // Two 20px rows plus one 10px row gap.
    assert!(approx(g.rect.h, 50.0));
}

#[test]
fn grid_item_spans_two_columns() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div><div class="i"></div><div class="i"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px; gap: 10px; } .a { grid-column: 1 / 3; height: 20px; } .i { height: 20px; }",
    );
    let g = grid_box(&l);
    let items = &g.children;
    assert_eq!(items.len(), 3);
    // The spanning item covers both columns plus the gap between them.
    assert!(approx(items[0].rect.x, 0.0));
    assert!(approx(items[0].rect.w, 210.0));
    // Auto-placed items flow into the next row after it.
    assert!(approx(items[1].rect.x, 0.0));
    assert!(approx(items[1].rect.y, 30.0));
    assert!(approx(items[2].rect.x, 110.0));
    assert!(approx(items[2].rect.y, 30.0));
}

#[test]
fn grid_grows_implicit_rows() {
    let l = layout_html(
        r#"<div class="g"><div class="i"></div><div class="i"></div><div class="i"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px; } .i { height: 20px; }",
    );
    let g = grid_box(&l);
    let items = &g.children;
    assert_eq!(items.len(), 3);
    // The third item wraps into an implicit second row.
    assert!(approx(items[2].rect.x, 0.0));
    assert!(approx(items[2].rect.y, 20.0));
    assert!(approx(g.rect.h, 40.0));
}

#[test]
fn grid_percentage_row_track_behaves_as_auto_when_height_indefinite() {
    let l = layout_html(
        r#"<div class="g"><div class="i"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-template-rows: 50%; } \
         .i { height: 20px; }",
    );
    let g = grid_box(&l);
    // The 50% track cannot resolve against the indefinite container
    // height, so it behaves as `auto` and sizes to the item's content.
    assert!(approx(g.rect.h, 20.0));
    assert!(approx(g.children[0].rect.y, g.rect.y));
}

#[test]
fn grid_percentage_row_track_resolves_against_definite_height() {
    let l = layout_html(
        r#"<div class="g"><div class="i"></div><div class="j"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-template-rows: 50% 50%; \
           height: 100px; } .i { height: 20px; } .j { height: 20px; }",
    );
    let g = grid_box(&l);
    // Each 50% track is 50px against the definite 100px height.
    assert!(approx(g.rect.h, 100.0));
    assert!(approx(g.children[0].rect.y, g.rect.y));
    assert!(approx(g.children[1].rect.y, g.rect.y + 50.0));
}

#[test]
fn grid_auto_columns_none_stacks_items_vertically() {
    let l = layout_html(
        r#"<div class="g"><div class="i"></div><div class="i"></div></div>"#,
        ".g { display: grid; } .i { height: 20px; }",
    );
    let g = grid_box(&l);
    let items = &g.children;
    assert_eq!(items.len(), 2);
    // No explicit columns: each auto item wraps to its own implicit row.
    assert!(approx(items[0].rect.y, g.rect.y));
    assert!(approx(items[1].rect.y, g.rect.y + 20.0));
    assert!(approx(g.rect.h, 40.0));
}

#[test]
fn grid_item_alignment_uses_collapsed_top_margin() {
    let l = layout_html(
        r#"<div class="g"><div class="i"><div class="c"></div></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; } \
         .i { margin-top: 10px; } .c { margin-top: 30px; height: 20px; }",
    );
    let g = grid_box(&l);
    let item = &g.children[0];
    // The item's top margin collapses through its first child to 30px,
    // and alignment positions the border box below that collapsed margin
    // (not below the uncollapsed 10px).
    assert!(approx(item.rect.y, g.rect.y + 30.0));
}

#[test]
fn grid_minmax_fr_track_applies_minimum() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div><div class="b"></div></div>"#,
        ".g { display: grid; grid-template-columns: minmax(100px, 1fr) 200px; width: 800px; } \
         .a { height: 20px; } .b { height: 20px; }",
    );
    let g = grid_box(&l);
    let items = &g.children;
    assert_eq!(items.len(), 2);
    // Column 0: max(0 content, 100px min) + (800 - 100 - 200) free = 600px.
    assert!(approx(items[0].rect.w, 600.0));
    assert!(approx(items[1].rect.x, 600.0));
    assert!(approx(items[1].rect.w, 200.0));
}

#[test]
fn grid_minmax_min_larger_than_max_clamps_to_min() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: minmax(200px, 100px); width: 800px; } \
         .a { height: 20px; }",
    );
    let g = grid_box(&l);
    // The maximum is ignored; the track keeps its 200px minimum.
    assert!(approx(g.children[0].rect.w, 200.0));
}

#[test]
fn grid_minmax_content_bounds_size_to_content() {
    // `preferred_width` measures text (not fixed widths), so the
    // content-sized track below fits 5 chars at 0.6 * 16px = 48px.
    let l = layout_html(
        r#"<div class="g"><div class="a">hello</div></div>"#,
        ".g { display: grid; grid-template-columns: minmax(min-content, max-content) 1fr; width: 800px; } \
         .a { height: 20px; }",
    );
    let g = grid_box(&l);
    // min-content/max-content both measure content in this pass: 48px.
    assert!(approx(g.children[0].rect.w, 48.0));
    // The `1fr` track starts right after it.
    assert!(approx(
        g.children[0].rect.x + g.children[0].rect.w,
        g.rect.x + 48.0
    ));
}

#[test]
fn grid_minmax_maximum_caps_growth() {
    // 20 chars at 0.6 * 16px = 192px of content: the track grows from
    // its 50px minimum but stops at its 120px maximum (grid §12.4).
    let l = layout_html(
        r#"<div class="g"><div class="a">abcdefghijklmnopqrst</div></div>"#,
        ".g { display: grid; grid-template-columns: minmax(50px, 120px); width: 800px; } \
         .a { height: 20px; }",
    );
    let g = grid_box(&l);
    assert!(approx(g.children[0].rect.w, 120.0));
}

#[test]
fn grid_minmax_inside_repeat_expands() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div><div class="b"></div></div>"#,
        ".g { display: grid; grid-template-columns: repeat(2, minmax(50px, 1fr)); width: 800px; } \
         .a { height: 20px; } .b { height: 20px; }",
    );
    let g = grid_box(&l);
    let items = &g.children;
    assert_eq!(items.len(), 2);
    // Two minmax(50px, 1fr) tracks: 50 + (800 - 100) / 2 = 400px each.
    assert!(approx(items[0].rect.w, 400.0));
    assert!(approx(items[1].rect.x, 400.0));
    assert!(approx(items[1].rect.w, 400.0));
}

#[test]
fn grid_minmax_fr_minimum_is_invalid() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div><div class="b"></div></div>"#,
        ".g { display: grid; grid-template-columns: minmax(1fr, 2fr); } \
         .a { height: 20px; } .b { height: 20px; }",
    );
    let g = grid_box(&l);
    // An `fr` minimum is invalid: the declaration is dropped, so the
    // grid falls back to implicit auto tracks and the items stack.
    let items = &g.children;
    assert_eq!(items.len(), 2);
    assert!(approx(items[0].rect.y, g.rect.y));
    assert!(approx(items[1].rect.y, g.rect.y + 20.0));
}

#[test]
fn grid_justify_self_start_shrinks_to_fit() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; } \
         .a { justify-self: start; width: 50px; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = &g.children[0];
    assert!(approx(a.rect.x, g.rect.x));
    assert!(approx(a.rect.w, 50.0));
}

#[test]
fn grid_justify_self_end_aligns_to_area_end() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; } \
         .a { justify-self: end; width: 50px; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = &g.children[0];
    assert!(approx(a.rect.x, g.rect.x + 350.0));
    assert!(approx(a.rect.w, 50.0));
}

#[test]
fn grid_justify_self_center() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; } \
         .a { justify-self: center; width: 50px; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = &g.children[0];
    assert!(approx(a.rect.x, g.rect.x + 175.0));
    assert!(approx(a.rect.w, 50.0));
}

#[test]
fn grid_justify_self_auto_defers_to_container() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div><div class="b"></div></div>"#,
        ".g { display: grid; grid-template-columns: 200px 200px; justify-items: center; } \
         .a { justify-self: start; width: 50px; height: 20px; } \
         .b { justify-self: auto; width: 50px; height: 20px; }",
    );
    let g = grid_box(&l);
    let (a, b) = (&g.children[0], &g.children[1]);
    // `start` overrides the container's `center` ...
    assert!(approx(a.rect.x, g.rect.x));
    // ... while `auto` (the initial value) defers to it.
    assert!(approx(b.rect.x, g.rect.x + 200.0 + 75.0));
    assert!(approx(b.rect.w, 50.0));
}

#[test]
fn grid_align_self_end_positions_at_block_end() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-template-rows: 100px; height: 100px; } \
         .a { align-self: end; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = &g.children[0];
    assert!(approx(a.rect.y, g.rect.y + 80.0));
    assert!(approx(a.rect.h, 20.0));
}

#[test]
fn grid_align_self_center() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-template-rows: 100px; height: 100px; } \
         .a { align-self: center; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = &g.children[0];
    assert!(approx(a.rect.y, g.rect.y + 40.0));
}

#[test]
fn grid_align_self_auto_defers_to_container() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-template-rows: 100px; \
           height: 100px; align-items: center; } \
         .a { align-self: auto; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = &g.children[0];
    // `auto` defers to the container's `align-items: center`.
    assert!(approx(a.rect.y, g.rect.y + 40.0));
}

#[test]
fn grid_align_self_stretch_overrides_container_start() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-template-rows: 100px; \
           height: 100px; align-items: start; } \
         .a { align-self: stretch; }",
    );
    let g = grid_box(&l);
    let a = &g.children[0];
    // Explicit `stretch` grows the auto-height item to the 100px area
    // even though the container says `start`.
    assert!(approx(a.rect.y, g.rect.y));
    assert!(approx(a.rect.h, 100.0));
}

/// The item's grid-area box: the first (only) child of the grid.
fn grid_item(l: &LayoutBox) -> &LayoutBox {
    &grid_box(l).children[0]
}

// css-grid-1 §10.2: auto margins absorb free space before alignment.
// The shrink-to-fit item below is 50x20 in a 400x100 area.

#[test]
fn grid_auto_margins_center_item() {
    let l = layout_html(
        r#"<div class="g"><div class="a"><div class="k"></div></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; grid-template-rows: 100px; \
           justify-items: start; } \
         .a { margin: auto; } .k { width: 50px; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // (400 - 50) / 2 = 175 inline, (100 - 20) / 2 = 40 block. The auto
    // margins also beat the default `stretch` on the block axis: the
    // item keeps its 20px height instead of growing to 100px.
    assert!(approx(a.rect.x, g.rect.x + 175.0));
    assert!(approx(a.rect.y, g.rect.y + 40.0));
    assert!(approx(a.rect.w, 50.0));
    assert!(approx(a.rect.h, 20.0));
}

#[test]
fn grid_auto_margin_left_pushes_to_inline_end() {
    let l = layout_html(
        r#"<div class="g"><div class="a"><div class="k"></div></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; justify-items: start; } \
         .a { margin-left: auto; } .k { width: 50px; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // A single auto margin takes all the free space: 400 - 50 = 350.
    assert!(approx(a.rect.x, g.rect.x + 350.0));
    assert!(approx(a.rect.w, 50.0));
}

#[test]
fn grid_auto_margin_right_alone_keeps_item_at_start() {
    let l = layout_html(
        r#"<div class="g"><div class="a"><div class="k"></div></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; justify-items: start; } \
         .a { margin-right: auto; } .k { width: 50px; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // `margin-right: auto` absorbs the free space on the far side; the
    // border box stays at the area start.
    assert!(approx(a.rect.x, g.rect.x));
    assert!(approx(a.rect.w, 50.0));
}

#[test]
fn grid_auto_margin_top_pushes_to_block_end() {
    let l = layout_html(
        r#"<div class="g"><div class="a"><div class="k"></div></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; grid-template-rows: 100px; \
           align-items: start; } \
         .a { margin-top: auto; } .k { width: 50px; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // 100 - 20 = 80px of free space goes to the top auto margin.
    assert!(approx(a.rect.y, g.rect.y + 80.0));
    assert!(approx(a.rect.h, 20.0));
}

#[test]
fn grid_auto_margins_beat_alignment_on_same_axis() {
    let l = layout_html(
        r#"<div class="g"><div class="a"><div class="k"></div></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; grid-template-rows: 100px; \
           justify-items: start; align-items: start; } \
         .a { margin: auto; justify-self: end; align-self: end; } \
         .k { width: 50px; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // The auto margins win over `justify-self: end` / `align-self:
    // end`: the item centers instead of going to the area end.
    assert!(approx(a.rect.x, g.rect.x + 175.0));
    assert!(approx(a.rect.y, g.rect.y + 40.0));
}

#[test]
fn grid_auto_margin_one_axis_alignment_applies_on_other() {
    let l = layout_html(
        r#"<div class="g"><div class="a"><div class="k"></div></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; grid-template-rows: 100px; \
           justify-items: start; align-items: start; } \
         .a { margin-left: auto; align-self: center; } \
         .k { width: 50px; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // Inline: the auto margin absorbs everything (right-aligned).
    // Block: no auto margin, so `align-self: center` applies.
    assert!(approx(a.rect.x, g.rect.x + 350.0));
    assert!(approx(a.rect.y, g.rect.y + 40.0));
}

#[test]
fn grid_auto_margins_over_constrained_behave_as_zero() {
    let l = layout_html(
        r#"<div class="g"><div class="a"><div class="k"></div></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; grid-template-rows: 20px; \
           align-items: start; } \
         .a { margin-top: auto; margin-bottom: auto; } \
         .k { width: 50px; height: 50px; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // Negative free space (20 - 50): the auto margins become zero and
    // the item overflows at the area start instead of shifting up.
    assert!(approx(a.rect.y, g.rect.y));
    assert!(approx(a.rect.h, 50.0));
}

#[test]
fn grid_auto_margins_definite_width_not_double_counted() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; justify-items: start; } \
         .a { width: 50px; height: 20px; margin: auto; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // Definite widths resolve inline auto margins in the block path
    // (CSS 2.1 §10.3.3); the grid placement must not absorb again.
    assert!(approx(a.rect.x, g.rect.x + 175.0));
    assert!(approx(a.rect.w, 50.0));
}

#[test]
fn grid_auto_margins_definite_width_beat_end_alignment() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; } \
         .a { width: 50px; height: 20px; margin: auto; justify-self: end; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // Definite width: §10.3.3 splits the 350px of free space into 175px
    // auto margins, and `justify-self: end` is a no-op (css-grid-1
    // §10.2) — the item centers instead of landing at x=350.
    assert!(approx(a.rect.x, g.rect.x + 175.0));
    assert!(approx(a.rect.w, 50.0));
}

#[test]
fn grid_auto_margins_definite_width_beat_center_alignment() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; } \
         .a { width: 50px; height: 20px; margin: auto; justify-self: center; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // The auto margins absorb the free space before alignment runs, so
    // the explicit `center` changes nothing: still x=175.
    assert!(approx(a.rect.x, g.rect.x + 175.0));
    assert!(approx(a.rect.w, 50.0));
}

#[test]
fn grid_single_auto_margin_beats_end_alignment() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; } \
         .a { width: 50px; height: 20px; margin-left: auto; justify-self: end; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // The single auto margin takes all 350px of free space; the
    // `justify-self: end` shift must not apply on top of it.
    assert!(approx(a.rect.x, g.rect.x + 350.0));
    assert!(approx(a.rect.w, 50.0));
}

#[test]
fn grid_single_auto_margin_right_beats_end_alignment() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; } \
         .a { width: 50px; height: 20px; margin-right: auto; justify-self: end; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // `margin-right: auto` absorbs the free space on the far side, so
    // the border box stays at the area start despite `justify-self:
    // end`.
    assert!(approx(a.rect.x, g.rect.x));
    assert!(approx(a.rect.w, 50.0));
}

#[test]
fn grid_auto_margins_suppress_stretch_sizing() {
    let l = layout_html(
        r#"<div class="g"><div class="a"><div class="k"></div></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; grid-template-rows: 100px; } \
         .a { margin: auto; } .k { width: 50px; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // css-grid-1 §10.2: auto margins suppress the default `stretch`
    // sizing on the inline axis. The item shrink-to-fits (50px) and the
    // margins absorb the free space instead of the item filling the
    // 400px area; the block axis was already suppressed the same way.
    assert!(approx(a.rect.w, 50.0));
    assert!(approx(a.rect.x, g.rect.x + 175.0));
    assert!(approx(a.rect.h, 20.0));
    assert!(approx(a.rect.y, g.rect.y + 40.0));
}

#[test]
fn grid_auto_margins_suppress_explicit_stretch_sizing() {
    let l = layout_html(
        r#"<div class="g"><div class="a"><div class="k"></div></div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; } \
         .a { margin-left: auto; margin-right: auto; justify-self: stretch; } \
         .k { width: 50px; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = grid_item(&l);
    // An explicit `justify-self: stretch` is suppressed the same way:
    // the item keeps its shrink-to-fit width and centers.
    assert!(approx(a.rect.w, 50.0));
    assert!(approx(a.rect.x, g.rect.x + 175.0));
}

#[test]
fn grid_place_self_two_values() {
    // `place-self: <align-self> <justify-self>?` expands to both
    // longhands (CSS Box Alignment 3 §6.1).
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-template-rows: 100px; \
           width: 100px; height: 100px; } \
         .a { place-self: start end; width: 20px; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = &g.children[0];
    assert!(approx(a.rect.y, g.rect.y));
    assert!(approx(a.rect.x, g.rect.x + 80.0));
}

#[test]
fn grid_place_self_single_value_sets_both() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-template-rows: 100px; \
           width: 100px; height: 100px; } \
         .a { place-self: center; width: 20px; height: 20px; }",
    );
    let g = grid_box(&l);
    let a = &g.children[0];
    assert!(approx(a.rect.y, g.rect.y + 40.0));
    assert!(approx(a.rect.x, g.rect.x + 40.0));
}

#[test]
fn grid_align_self_baseline_stretches_not_container_value() {
    // `baseline` has no baseline-alignment engine behind it, so it
    // stretches; it must not silently inherit the container's
    // `align-items: start`.
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-template-rows: 100px; \
           height: 100px; align-items: start; } \
         .a { align-self: baseline; }",
    );
    let g = grid_box(&l);
    let a = &g.children[0];
    assert!(approx(a.rect.y, g.rect.y));
    assert!(approx(a.rect.h, 100.0));
}

#[test]
fn flex_align_self_self_start_overrides_stretch() {
    // `self-start` maps onto start for flex items (CSS Box Alignment
    // 3): it overrides the container's `align-items: stretch`.
    let l = layout_html(
        r#"<div class="c"><div class="i"></div></div>"#,
        ".c { display: flex; width: 200px; height: 100px; align-items: stretch; } \
         .i { width: 50px; height: 20px; align-self: self-start; }",
    );
    let c = &body(&l).children[0];
    let i = &c.children[0];
    assert!(approx(i.rect.y, c.rect.y));
    assert!(approx(i.rect.h, 20.0));
}

#[test]
fn flex_justify_content_left_right_follow_direction() {
    // CSS Box Alignment 3 §4.2: `left`/`right` are physical — `right`
    // packs at the physical right in a left-to-right container.
    let l = layout_html(
        r#"<div class="c"><div class="i"></div></div>"#,
        ".c { display: flex; width: 200px; justify-content: right; } \
         .i { width: 50px; height: 20px; }",
    );
    let c = &body(&l).children[0];
    let i = &c.children[0];
    assert!(approx(i.rect.x, c.rect.x + 150.0));
    // In a right-to-left container `left` packs at the main end,
    // which is the physical left.
    let l = layout_html(
        r#"<div class="c"><div class="i"></div></div>"#,
        ".c { display: flex; width: 200px; direction: rtl; justify-content: left; } \
         .i { width: 50px; height: 20px; }",
    );
    let c = &body(&l).children[0];
    let i = &c.children[0];
    assert!(approx(i.rect.x, c.rect.x));
}

#[test]
fn dir_attribute_hint_flips_flex_direction() {
    // The `dir` presentational hint maps to the CSS `direction`
    // property: an rtl container lays flex items right-to-left, so
    // the first item packs at the physical right.
    let l = layout_html(
        r#"<div class="c" dir="rtl"><div class="i"></div></div>"#,
        ".c { display: flex; width: 200px; } .i { width: 50px; height: 20px; }",
    );
    let c = &body(&l).children[0];
    let i = &c.children[0];
    assert!(approx(i.rect.x, c.rect.x + 150.0));
}

#[test]
fn w77g_inline_table_flex_item_blockifies_with_basis_and_margins() {
    // css-flexbox-1 §4 blockification: `display: inline-table` on a
    // flex item becomes a table, so the item itself (not a style-less
    // anonymous wrapper) carries `flex-basis` and margins.
    // Fail-before: the container's item was the wrapper (basis auto,
    // no margins), so the table laid out at its 80px content width
    // at the container's start edge.
    let l = layout_html(
        r#"<div class="c"><table class="t"><tr><td><div style="width:40px"></div></td><td><div style="width:40px"></div></td></tr></table><div class="s"></div></div>"#,
        "body { margin: 0; } .c { display: flex; width: 600px; } \
         .t { display: inline-table; flex: 0 0 200px; margin: 0 10px; } \
         .s { width: 50px; height: 20px; }",
    );
    let c = &body(&l).children[0];
    assert_eq!(
        c.children.len(),
        2,
        "want table item + sibling, got {}",
        c.children.len()
    );
    let t = &c.children[0];
    assert_eq!(t.box_type, BoxType::Table);
    assert!(approx(t.rect.w, 200.0), "table w={} != 200", t.rect.w);
    assert!(
        approx(t.rect.x, c.rect.x + 10.0),
        "table x={} != container + 10",
        t.rect.x
    );
    let s = &c.children[1];
    assert!(
        approx(s.rect.x, c.rect.x + 220.0),
        "sibling x={} != container + 220",
        s.rect.x
    );
}

#[test]
fn w77g_inline_table_flex_item_grows() {
    // The blockified table item's `flex-grow` reaches it too: basis
    // 100 with grow 1 fills the 600px container. Fail-before: the
    // wrapper's grow was 0, so the table stayed at 80px.
    let l = layout_html(
        r#"<div class="c"><table class="t"><tr><td><div style="width:40px"></div></td><td><div style="width:40px"></div></td></tr></table></div>"#,
        "body { margin: 0; } .c { display: flex; width: 600px; } \
         .t { display: inline-table; flex: 1 1 100px; }",
    );
    let c = &body(&l).children[0];
    assert_eq!(c.children.len(), 1);
    let t = &c.children[0];
    assert_eq!(t.box_type, BoxType::Table);
    assert!(approx(t.rect.w, 600.0), "table w={} != 600", t.rect.w);
}

#[test]
fn w77g_inline_table_flex_item_extracted_from_text_run() {
    // An inline-table inside a text run is its own flex item, not
    // part of the run's anonymous item: "hi" is 12px at 10px font,
    // so the table starts 12px in and still carries its 200px basis.
    // Fail-before: one wrapper item held both the text and the table.
    let l = layout_html(
        r#"<div class="c">hi<table class="t"><tr><td><div style="width:40px"></div></td><td><div style="width:40px"></div></td></tr></table></div>"#,
        "body { margin: 0; } .c { display: flex; width: 600px; font-size: 10px; } \
         .t { display: inline-table; flex: 0 0 200px; }",
    );
    let c = &body(&l).children[0];
    assert_eq!(
        c.children.len(),
        2,
        "want text item + table item, got {}",
        c.children.len()
    );
    assert_eq!(c.children[0].box_type, BoxType::AnonymousBlock);
    let t = &c.children[1];
    assert_eq!(t.box_type, BoxType::Table);
    assert!(approx(t.rect.w, 200.0), "table w={} != 200", t.rect.w);
    assert!(
        approx(t.rect.x, c.rect.x + 12.0),
        "table x={} != container + 12",
        t.rect.x
    );
}

#[test]
fn w77g_inline_table_outside_flex_unchanged() {
    // Blockification is a flex-item rule only: an inline-table in
    // ordinary block flow stays an `InlineBlock` box at its content
    // width (80px from the two 40px cells).
    let l = layout_html(
        r#"<div class="a"><table class="t"><tr><td><div style="width:40px"></div></td><td><div style="width:40px"></div></td></tr></table></div>"#,
        ".a { width: 400px; } .t { display: inline-table; }",
    );
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.box_type, BoxType::InlineBlock);
    assert!(approx(t.rect.w, 80.0), "table w={} != 80", t.rect.w);
}

#[test]
fn w77g_flex_text_run_still_wraps_as_anonymous_item() {
    // The wrapper path stays load-bearing for text: a lone text run
    // is still one anonymous-block item (30px for "hello" at 10px).
    let l = layout_html(
        r#"<div class="c">hello</div>"#,
        "body { margin: 0; } .c { display: flex; width: 200px; font-size: 10px; }",
    );
    let c = &body(&l).children[0];
    assert_eq!(c.children.len(), 1);
    let w = &c.children[0];
    assert_eq!(w.box_type, BoxType::AnonymousBlock);
    assert!(approx(w.rect.w, 30.0), "wrapper w={} != 30", w.rect.w);
}

#[test]
fn w77g_flex_inline_block_item_still_wrapped() {
    // Only inline-table blockifies here: an inline-block flex item
    // keeps the old wrapper shape (wrapper > inline-block at 60px),
    // so other inline items are untouched by this change.
    let l = layout_html(
        r#"<div class="c"><div class="i"></div></div>"#,
        "body { margin: 0; } .c { display: flex; width: 200px; } \
         .i { display: inline-block; width: 60px; height: 20px; }",
    );
    let c = &body(&l).children[0];
    assert_eq!(c.children.len(), 1);
    let w = &c.children[0];
    assert_eq!(w.box_type, BoxType::AnonymousBlock);
    let i = &w.children[0];
    assert_eq!(i.box_type, BoxType::InlineBlock);
    assert!(approx(i.rect.w, 60.0), "inline-block w={} != 60", i.rect.w);
}

#[test]
fn w77fix_inline_table_blockifies_through_both_triggers() {
    // w77fix finding 12, joint pin: inline-table blockification
    // runs through two triggers that must agree. Container A has
    // a block-level sibling, so the `needs_wrap` branch in
    // `boxes.rs` pass 3 promotes the table at box-build time;
    // container B has only a text run beside the table, so pass 3
    // does not fire (no block-level child) and
    // `wrap_flex_text_runs` promotes it at flex-layout time.
    // Both tables carry the same `flex: 0 0 200px` and 10px side
    // margins, so both items must land identically: a `Table`
    // box, 200px wide, 10px off the container's start edge, with
    // the sibling after the margin box. A regression in either
    // trigger breaks this one test.
    let table = r#"<table class="t"><tr><td><div style="width:40px"></div></td><td><div style="width:40px"></div></td></tr></table>"#;
    let css = "body { margin: 0; } .c { display: flex; width: 600px; font-size: 10px; } \
         .t { display: inline-table; flex: 0 0 200px; margin: 0 10px; } \
         .s { width: 50px; height: 20px; }";
    // Trigger one (boxes pass 3): block sibling present.
    let l = layout_html(
        &format!(r#"<div class="c">{table}<div class="s"></div></div>"#),
        css,
    );
    let c = &body(&l).children[0];
    assert_eq!(c.children.len(), 2, "A: want table item + sibling");
    let t = &c.children[0];
    assert_eq!(t.box_type, BoxType::Table, "A: table must promote");
    assert!(approx(t.rect.w, 200.0), "A: table w={} != 200", t.rect.w);
    assert!(
        approx(t.rect.x, c.rect.x + 10.0),
        "A: table x={} != container + 10",
        t.rect.x
    );
    assert!(
        approx(c.children[1].rect.x, c.rect.x + 220.0),
        "A: sibling x={} != container + 220",
        c.children[1].rect.x
    );
    // Trigger two (wrap_flex_text_runs): text run, no block child.
    let l = layout_html(&format!(r#"<div class="c">hi{table}</div>"#), css);
    let c = &body(&l).children[0];
    assert_eq!(c.children.len(), 2, "B: want text item + table item");
    assert_eq!(c.children[0].box_type, BoxType::AnonymousBlock);
    let t = &c.children[1];
    assert_eq!(t.box_type, BoxType::Table, "B: table must promote");
    assert!(approx(t.rect.w, 200.0), "B: table w={} != 200", t.rect.w);
    assert!(
        approx(t.rect.x, c.rect.x + 22.0),
        "B: table x={} != container + 22 (12 text + 10 margin)",
        t.rect.x
    );
}

#[test]
fn block_height_stretch_fills_containing_block() {
    // CSS Sizing 3 §5: `height: stretch` fills the containing-block
    // height; against an indefinite CB height it behaves as `auto`.
    let l = layout_html(
        r#"<div class="c"><div class="i"></div></div>"#,
        ".c { height: 100px; } .i { height: stretch; }",
    );
    let c = &body(&l).children[0];
    let i = &c.children[0];
    assert!(approx(i.rect.h, 100.0));
    // Indefinite containing-block height: behaves as `auto` (the
    // empty item collapses).
    let l = layout_html(
        r#"<div class="c"><div class="i"></div></div>"#,
        ".i { height: stretch; }",
    );
    let c = &body(&l).children[0];
    let i = &c.children[0];
    assert!(approx(i.rect.h, 0.0));
}

#[test]
fn grid_justify_content_center() {
    let l = layout_html(
        r#"<div class="g"><div class="i"></div><div class="i"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px; width: 300px; \
           justify-content: center; } .i { height: 20px; }",
    );
    let g = grid_box(&l);
    let items = &g.children;
    // 300 - 200 = 100px of leftover space, centered: tracks start at x=50.
    assert!(approx(items[0].rect.x, 50.0));
    assert!(approx(items[1].rect.x, 150.0));
}

#[test]
fn grid_justify_content_space_between() {
    let l = layout_html(
        r#"<div class="g"><div class="i"></div><div class="i"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px; width: 300px; \
           justify-content: space-between; } .i { height: 20px; }",
    );
    let g = grid_box(&l);
    let items = &g.children;
    // All 100px of leftover space goes between the two tracks.
    assert!(approx(items[0].rect.x, 0.0));
    assert!(approx(items[1].rect.x, 200.0));
}

#[test]
fn grid_align_content_center() {
    let l = layout_html(
        r#"<div class="g"><div class="i"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-template-rows: 50px 50px; \
           height: 200px; align-content: center; } .i { height: 20px; }",
    );
    let g = grid_box(&l);
    // 200 - 100 = 100px of leftover space, centered: rows start at y=50.
    assert!(approx(g.children[0].rect.y, g.rect.y + 50.0));
}

#[test]
fn grid_auto_flow_column_places_down_then_across() {
    let l = layout_html(
        r#"<div class="g"><div class="i"></div><div class="i"></div><div class="i"></div></div>"#,
        ".g { display: grid; grid-template-rows: 30px 30px; grid-auto-flow: column; } \
         .i { width: 40px; height: 10px; }",
    );
    let g = grid_box(&l);
    let items = &g.children;
    assert_eq!(items.len(), 3);
    // Column-major: fill column 0 top-to-bottom, then column 1.
    assert!(approx(items[0].rect.x, 0.0));
    assert!(approx(items[0].rect.y, g.rect.y));
    assert!(approx(items[1].rect.x, 0.0));
    assert!(approx(items[1].rect.y, g.rect.y + 30.0));
    // `normal` content-alignment stretches the content-sized tracks to
    // absorb the free space: the two 40px columns grow to 400px each,
    // so column 1 starts at x=400.
    assert!(approx(items[2].rect.x, 400.0));
    assert!(approx(items[2].rect.y, g.rect.y));
}

#[test]
fn grid_order_reorders_auto_placement() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div><div class="b"></div><div class="c"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px; } \
         .a, .b, .c { height: 20px; } .b { order: -1; }",
    );
    let g = grid_box(&l);
    // `.b` (order -1) takes the first cell despite being second in DOM.
    // `children` stays in DOM order; placement follows `order`.
    let (a, b, c) = (&g.children[0], &g.children[1], &g.children[2]);
    assert!(approx(b.rect.x, 0.0));
    assert!(approx(a.rect.x, 100.0));
    assert!(approx(c.rect.x, 0.0));
    assert!(approx(c.rect.y, g.rect.y + 20.0));
}

#[test]
fn grid_floated_child_becomes_grid_item() {
    let l = layout_html(
        r#"<div class="g"><div class="f"></div><div class="a"></div><div class="b"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px; } \
         .f { float: left; width: 50px; height: 10px; } \
         .a, .b { height: 20px; }",
    );
    let g = grid_box(&l);
    // css-grid-1 §5.1: `float` computes to none on grid items, so the
    // floated child occupies the first cell like any other item.
    let (f, a, b) = (&g.children[0], &g.children[1], &g.children[2]);
    assert!(approx(f.rect.x, g.rect.x));
    assert!(approx(f.rect.y, g.rect.y));
    assert!(approx(f.rect.w, 50.0));
    assert!(approx(a.rect.x, 100.0));
    assert!(approx(a.rect.y, g.rect.y));
    assert!(approx(b.rect.x, 0.0));
    assert!(approx(b.rect.y, g.rect.y + 20.0));
}

#[test]
fn grid_subgrid_reuses_parent_columns() {
    let l = layout_html(
        r#"<div class="g"><div class="sg"><div class="i"></div><div class="i"></div></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px; } \
         .sg { display: grid; grid-template-columns: subgrid; grid-column: 1 / 3; } \
         .i { height: 20px; }",
    );
    let g = grid_box(&l);
    let sg = &g.children[0];
    assert_eq!(sg.box_type, BoxType::Grid);
    let items = &sg.children;
    assert_eq!(items.len(), 2);
    // The subgrid inherits the parent's two 100px tracks.
    assert!(approx(items[1].rect.x - items[0].rect.x, 100.0));
}

#[test]
fn grid_stretched_item_relays_out_at_stretched_height() {
    let l = layout_html(
        r#"<div class="g"><div class="s"><div class="c"></div></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-template-rows: 100px; \
           height: 100px; } .c { height: 50%; }",
    );
    let g = grid_box(&l);
    let s = &g.children[0];
    let c = &s.children[0];
    // The stretched item grows to the 100px area; the re-layout lets the
    // percentage-height child resolve against the stretched size.
    assert!(approx(s.rect.h, 100.0));
    assert!(approx(c.rect.h, 50.0));
}

#[test]
fn grid_auto_columns_size_implicit_tracks() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div><div class="b"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-auto-columns: 50px; } \
         .a, .b { height: 20px; } .b { grid-column: 3; }",
    );
    let g = grid_box(&l);
    let (a, b) = (&g.children[0], &g.children[1]);
    assert!(approx(a.rect.x, 0.0));
    // Column 3 (zero-based 2) is implicit: 100 + 50 = 150.
    assert!(approx(b.rect.x, 150.0));
    assert!(approx(b.rect.w, 50.0));
}

#[test]
fn grid_auto_rows_size_implicit_tracks() {
    let l = layout_html(
        r#"<div class="g"><div class="a"></div><div class="b"></div><div class="c"></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-template-rows: 25px; \
           grid-auto-rows: 40px; } .a, .b, .c { height: 20px; }",
    );
    let g = grid_box(&l);
    let (a, b, c) = (&g.children[0], &g.children[1], &g.children[2]);
    // Row 0 is the explicit 25px track; rows 1+ are implicit 40px tracks.
    assert!(approx(a.rect.y, g.rect.y));
    assert!(approx(b.rect.y, g.rect.y + 25.0));
    assert!(approx(c.rect.y, g.rect.y + 65.0));
}

// ---------------------------------------------------------------------------
// Positioning (`position: relative | absolute | fixed`, offsets, z-index).
// ---------------------------------------------------------------------------

#[test]
fn relative_offsets_do_not_move_siblings() {
    // `position: relative` shifts the box visually but keeps its flow
    // footprint: the following sibling still stacks at the unshifted slot.
    let l = layout_html(
        r#"<div class="a"></div><div class="rel"></div><div class="b"></div>"#,
        ".a { height: 20px; } .rel { position: relative; left: 15px; top: 10px; height: 30px; } .b { height: 40px; }",
    );
    let b = body(&l);
    let (a, rel, sib) = (&b.children[0], &b.children[1], &b.children[2]);
    // Shifted by (15, 10) from the static slot under .a.
    assert!(approx(rel.rect.x, a.rect.x + 15.0));
    assert!(approx(rel.rect.y, a.rect.y + 20.0 + 10.0));
    // The sibling still starts at the unshifted position (y = 20 + 30).
    assert!(approx(sib.rect.y, a.rect.y + 50.0));
    // Body height is unaffected by the visual shift.
    assert!(approx(b.rect.h, 90.0));
}

#[test]
fn relative_left_top_win_over_right_bottom() {
    // Over-constrained relative offsets: `left` wins over `right` and
    // `top` wins over `bottom` in LTR.
    let l = layout_html(
        r#"<div class="a"></div><div class="rel"></div>"#,
        ".a { height: 20px; } .rel { position: relative; left: 5px; right: 20px; top: 4px; bottom: 8px; height: 10px; }",
    );
    let b = body(&l);
    let (a, rel) = (&b.children[0], &b.children[1]);
    assert!(approx(rel.rect.x, a.rect.x + 5.0));
    assert!(approx(rel.rect.y, a.rect.y + 20.0 + 4.0));
}

#[test]
fn relative_right_only_shifts_left() {
    // With only `right` set, the box shifts left by its value.
    let l = layout_html(
        r#"<div class="a"></div><div class="rel"></div>"#,
        ".a { height: 20px; } .rel { position: relative; right: 20px; bottom: 8px; height: 10px; }",
    );
    let b = body(&l);
    let (a, rel) = (&b.children[0], &b.children[1]);
    assert!(approx(rel.rect.x, a.rect.x - 20.0));
    assert!(approx(rel.rect.y, a.rect.y + 20.0 - 8.0));
}

#[test]
fn abspos_takes_no_space_in_flow() {
    // Out-of-flow boxes do not push siblings: .b stacks directly under .a
    // and the parent's height excludes the absolute box.
    let l = layout_html(
        r#"<div class="a"></div><div class="abs"></div><div class="b"></div>"#,
        ".a { height: 20px; } .abs { position: absolute; height: 100px; } .b { height: 30px; }",
    );
    let b = body(&l);
    let (a, abs, sib) = (&b.children[0], &b.children[1], &b.children[2]);
    assert!(approx(sib.rect.y, a.rect.y + 20.0));
    assert!(approx(b.rect.h, 50.0));
    // The absolute box stays in the tree (document order) for painting.
    assert_eq!(b.children.len(), 3);
    assert!(approx(abs.rect.h, 100.0));
}

#[test]
fn abspos_static_position_fallback() {
    // All offsets `auto`: the box keeps the static position flow would
    // have given it (under .a, at the content-box x origin).
    let l = layout_html(
        r#"<div class="a"></div><div class="abs">hi</div>"#,
        ".a { height: 20px; } .abs { position: absolute; }",
    );
    let b = body(&l);
    let (a, abs) = (&b.children[0], &b.children[1]);
    assert!(approx(abs.rect.x, b.rect.x));
    assert!(approx(abs.rect.y, a.rect.y + 20.0));
    // `width: auto` shrink-to-fits the "hi" text.
    assert!(approx(abs.rect.w, 2.0 * 0.6 * 16.0));
}

#[test]
fn abspos_offsets_against_nearest_positioned_ancestor() {
    // .outer is static, .mid is relative: offsets resolve against .mid's
    // padding box, not .outer's.
    let l = layout_html(
        r#"<div class="outer"><div class="mid"><div class="abs"></div></div></div>"#,
        ".outer { width: 500px; padding: 25px; } \
         .mid { position: relative; width: 400px; height: 300px; } \
         .abs { position: absolute; left: 0px; top: 0px; width: 10px; height: 10px; }",
    );
    let outer = &body(&l).children[0];
    let mid = &outer.children[0];
    let abs = &mid.children[0];
    // .mid's padding-box origin (no border/padding on .mid).
    assert!(approx(abs.rect.x, mid.rect.x));
    assert!(approx(abs.rect.y, mid.rect.y));
    // Distinct from .outer's padding-box origin (25px to the left/up).
    assert!(approx(outer.rect.x + 25.0, mid.rect.x));
}

#[test]
fn abspos_left_top_offsets() {
    let l = layout_html(
        r#"<div class="mid"><div class="abs"></div></div>"#,
        ".mid { position: relative; width: 400px; height: 300px; } \
         .abs { position: absolute; left: 50px; top: 30px; width: 100px; height: 60px; }",
    );
    let mid = &body(&l).children[0];
    let abs = &mid.children[0];
    assert!(approx(abs.rect.x, mid.rect.x + 50.0));
    assert!(approx(abs.rect.y, mid.rect.y + 30.0));
    assert!(approx(abs.rect.w, 100.0));
    assert!(approx(abs.rect.h, 60.0));
}

#[test]
fn abspos_right_bottom_offsets() {
    // `right`/`bottom` measure from the containing block's end edges.
    let l = layout_html(
        r#"<div class="mid"><div class="abs"></div></div>"#,
        ".mid { position: relative; width: 400px; height: 300px; } \
         .abs { position: absolute; right: 20px; bottom: 10px; width: 100px; height: 60px; }",
    );
    let mid = &body(&l).children[0];
    let abs = &mid.children[0];
    assert!(approx(abs.rect.x, mid.rect.x + 400.0 - 20.0 - 100.0));
    assert!(approx(abs.rect.y, mid.rect.y + 300.0 - 10.0 - 60.0));
}

#[test]
fn abspos_viewport_fallback_without_positioned_ancestor() {
    // No positioned ancestor: the initial containing block (viewport
    // origin) is used.
    let l = layout_html(
        r#"<div class="wrap"><div class="abs"></div></div>"#,
        ".wrap { width: 200px; margin-left: 100px; } \
         .abs { position: absolute; left: 10px; top: 5px; width: 50px; height: 50px; }",
    );
    let wrap = &body(&l).children[0];
    let abs = &wrap.children[0];
    assert!(approx(wrap.rect.x, body(&l).rect.x + 100.0));
    assert!(approx(abs.rect.x, 10.0));
    assert!(approx(abs.rect.y, 5.0));
}

#[test]
fn fixed_positions_against_viewport() {
    // `position: fixed` ignores even positioned ancestors.
    let l = layout_html(
        r#"<div class="mid"><div class="fix"></div></div>"#,
        ".mid { position: relative; width: 400px; margin-left: 100px; } \
         .fix { position: fixed; left: 10px; top: 5px; width: 50px; height: 50px; }",
    );
    let fix = &body(&l).children[0].children[0];
    assert!(approx(fix.rect.x, 10.0));
    assert!(approx(fix.rect.y, 5.0));
}

#[test]
fn abspos_diverts_to_non_positioned_fixed_cb_ancestor() {
    // A non-positioned transform/filter/will-change/perspective/contain
    // ancestor diverts an absolute box even when a positioned ancestor
    // sits above it (CSS Transforms 1 §3): the box resolves against the
    // trigger's padding box, not the positioned outer.
    for decl in [
        "transform: translateX(5px)",
        "filter: blur(1px)",
        "will-change: transform",
        "perspective: 500px",
        "contain: paint",
    ] {
        let css = format!(
            ".outer {{ position: relative; }} \
             .mid {{ margin-left: 100px; width: 200px; height: 50px; {decl}; }} \
             .abs {{ position: absolute; left: 10px; top: 5px; width: 20px; height: 20px; }}"
        );
        let l = layout_html(
            r#"<div class="outer"><div class="mid"><div class="abs"></div></div></div>"#,
            &css,
        );
        let abs = &body(&l).children[0].children[0].children[0];
        // Against the mid box (margin-left: 100); a missed diversion
        // would resolve against the outer at x = 10.
        assert!(approx(abs.rect.x, 110.0), "{decl}: x={}", abs.rect.x);
        assert!(approx(abs.rect.y, 5.0), "{decl}: y={}", abs.rect.y);
    }
}

#[test]
fn fixed_resolves_against_fixed_cb_trigger_ancestor() {
    // `position: fixed` resolves against the nearest ancestor with a
    // fixed-CB trigger instead of the viewport.
    for decl in [
        "filter: blur(1px)",
        "will-change: perspective",
        "perspective: 500px",
        "contain: layout",
    ] {
        let css = format!(
            ".mid {{ margin-left: 100px; margin-top: 50px; width: 200px; height: 200px; {decl}; }} \
             .fix {{ position: fixed; left: 10px; top: 5px; width: 20px; height: 20px; }}"
        );
        let l = layout_html(r#"<div class="mid"><div class="fix"></div></div>"#, &css);
        let fix = &body(&l).children[0].children[0];
        // Against the mid box; without the trigger it would be the
        // viewport (10, 5).
        assert!(approx(fix.rect.x, 110.0), "{decl}: x={}", fix.rect.x);
        assert!(approx(fix.rect.y, 55.0), "{decl}: y={}", fix.rect.y);
    }
}

#[test]
fn many_abspos_descendants_record_once() {
    // B5: `record_abs_item` dedups per node in O(1) via a flag on the
    // node; the old linear scan of all recorded items was O(n) per
    // record, quadratic overall. 1500 out-of-flow descendants are each
    // recorded exactly once and laid out against the positioned parent.
    let mut html = String::from("<div class=\"c\">");
    for _ in 0..1500 {
        html.push_str("<div class=\"a\"></div>");
    }
    html.push_str("</div>");
    let l = layout_html(
        &html,
        ".c { position: relative; width: 400px; height: 300px; } \
         .a { position: absolute; width: 10px; height: 10px; }",
    );
    let c = &body(&l).children[0];
    assert_eq!(c.children.len(), 1500);
    for a in &c.children {
        assert!(approx(a.rect.w, 10.0));
        assert!(approx(a.rect.h, 10.0));
    }
}

#[test]
fn nested_abspos_resolves_against_abspos_ancestor() {
    // An absolute box is itself positioned, so it is the containing
    // block for absolute descendants.
    let l = layout_html(
        r#"<div class="o"><div class="i"></div></div>"#,
        ".o { position: absolute; left: 100px; top: 80px; width: 200px; height: 200px; } \
         .i { position: absolute; left: 10px; top: 10px; width: 20px; height: 20px; }",
    );
    let o = &body(&l).children[0];
    let i = &o.children[0];
    assert!(approx(o.rect.x, 100.0));
    assert!(approx(o.rect.y, 80.0));
    assert!(approx(i.rect.x, 110.0));
    assert!(approx(i.rect.y, 90.0));
}

#[test]
fn abspos_inline_child_takes_no_space() {
    // An absolute span inside a line contributes no width: the line
    // holds just "abef" at one line height.
    let l = layout_html(
        r#"<div class="c">ab<span class="abs">cd</span>ef</div>"#,
        ".abs { position: absolute; left: 5px; top: 5px; }",
    );
    let c = &body(&l).children[0];
    assert!(approx(c.rect.h, 1.2 * 16.0));
    // children: [Text("ab"), Inline(span.abs), Text("ef")].
    let abs = &c.children[1];
    assert_eq!(abs.box_type, BoxType::Inline);
    // Positioned against the viewport (no positioned ancestor).
    assert!(approx(abs.rect.x, 5.0));
    assert!(approx(abs.rect.y, 5.0));
}

#[test]
fn abspos_auto_margins_compute_to_zero() {
    // Out-of-flow roots do not center with auto margins (CSS 2.1 §10.3.7).
    let l = layout_html(
        r#"<div class="mid"><div class="abs"></div></div>"#,
        ".mid { position: relative; width: 400px; } \
         .abs { position: absolute; left: 0px; top: 0px; width: 100px; margin: 0 auto; }",
    );
    let mid = &body(&l).children[0];
    let abs = &mid.children[0];
    assert!(approx(abs.rect.x, mid.rect.x));
}

// F1: with `left`/`right` both `auto`, `layout_position` consumes the
// recorded static position. The static position must be measured from
// the content-based width, not the fill-provisional one `resolve_metrics`
// produces: auto side margins split the remainder, centering the box.
#[test]
fn abspos_fit_content_auto_margins_center_on_measured_width() {
    // The in-flow spacer forces the block formatting-context path (an
    // abspos-only container takes the inline path, whose static
    // position is fragment-based).
    let l = layout_html(
        r#"<div class="mid"><div class="sp"></div><div class="abs">hi</div></div>"#,
        ".mid { position: relative; width: 400px; } .sp { height: 10px; } \
         .abs { position: absolute; font-size: 10px; width: fit-content; margin: 0 auto; }",
    );
    let mid = &body(&l).children[0];
    let abs = &mid.children[1];
    // "hi" at 10px: 2 chars * 0.6 * 10 = 12px; centered in 400px.
    // With the fill-provisional static position the box sat at x == 0.
    assert!(approx(abs.rect.w, 12.0), "width {}", abs.rect.w);
    assert!(
        approx(abs.rect.x, mid.rect.x + (400.0 - 12.0) / 2.0),
        "x {}",
        abs.rect.x
    );
}

#[test]
fn z_index_is_resolved_on_layout_boxes() {
    let l = layout_html(
        r#"<div class="a"></div><div class="b"></div>"#,
        ".a { position: relative; z-index: 5; } .b { position: absolute; z-index: -2; }",
    );
    let b = body(&l);
    assert_eq!(b.children[0].z_index, Some(5));
    assert_eq!(b.children[1].z_index, Some(-2));
    // Static boxes report no z-index.
    let l2 = layout_html(r#"<div class="c"></div>"#, ".c { z-index: 7; }");
    assert_eq!(body(&l2).children[0].z_index, None);
}

#[test]
fn abs_block_child_does_not_split_inline_run() {
    // An absolute block child is hoisted past the anonymous-block split:
    // the in-flow text around it forms one run on one line.
    let l = layout_html(
        r#"<div class="c">ab<div class="abs"></div>cd</div>"#,
        ".abs { position: absolute; left: 0px; top: 0px; }",
    );
    let c = &body(&l).children[0];
    assert_eq!(c.children.len(), 2);
    let anon = c
        .children
        .iter()
        .find(|b| b.box_type == BoxType::AnonymousBlock)
        .expect("single anonymous block");
    assert!(approx(anon.rect.h, 1.2 * 16.0));
    assert!(approx(c.rect.h, 1.2 * 16.0));
}

#[test]
fn margin_collapse_skips_abspos_first_child() {
    // The absolute first child's margins never collapse (CSS 2.1 §8.3.1):
    // .p's top margin collapses with the first in-flow child's
    // (max(5, 20) = 20), not the absolute child's 50px.
    let l = layout_html(
        r#"<div class="p"><div class="abs"></div><div class="inflow"></div></div>"#,
        ".p { margin-top: 5px; } .abs { position: absolute; margin-top: 50px; } \
         .inflow { margin-top: 20px; height: 10px; }",
    );
    let p = &body(&l).children[0];
    assert!(approx(p.rect.y, 20.0));
    let inflow = &p.children[1];
    assert!(approx(inflow.rect.y, p.rect.y));
}

/// Iterative sanity walk over a layout tree: every box origin and every
/// text-fragment coordinate/baseline is finite, and every width/height is
/// finite and nonnegative. Hostile inputs must degrade, never poison.
fn assert_layout_sane(l: &LayoutBox) {
    fn check_rect(rect: &super::Rect, what: &str) {
        assert!(
            rect.x.is_finite() && rect.y.is_finite(),
            "{what} has non-finite origin: {rect:?}"
        );
        assert!(
            rect.w.is_finite() && rect.h.is_finite() && rect.w >= 0.0 && rect.h >= 0.0,
            "{what} has bad size: {rect:?}"
        );
    }
    let mut stack = vec![l];
    while let Some(b) = stack.pop() {
        check_rect(&b.rect, "box");
        for f in &b.fragments {
            check_rect(&f.rect, "fragment");
            assert!(f.baseline.is_finite(), "non-finite baseline: {f:?}");
        }
        stack.extend(b.children.iter());
    }
}

#[test]
fn very_deep_dom_does_not_panic() {
    let depth = 10_000;
    let mut html = String::new();
    for _ in 0..depth {
        html.push_str("<div>");
    }
    html.push('x');
    for _ in 0..depth {
        html.push_str("</div>");
    }
    let l = layout_html(&html, "div { font-size: 10px; }");
    let mut b = body(&l);
    for _ in 0..depth {
        b = &b.children[0];
    }
    assert_eq!(b.children[0].box_type, BoxType::Text);
    assert_layout_sane(&l);
}

#[test]
fn zero_and_negative_sizes_stay_sane() {
    let l = layout_html(
        r#"<div class="a"><div class="b">x</div></div>"#,
        ".a { width: 0px; height: 0px; margin: -10px; } \
         .b { width: -20px; height: -5px; margin-top: -3px; font-size: 0px; line-height: 0; }",
    );
    assert_layout_sane(&l);
}

#[test]
fn huge_values_stay_finite_and_bounded() {
    let l = layout_html(
        r#"<div class="a"><div class="b">x</div></div>"#,
        ".a { width: 1e30px; margin: 1e20px; font-size: 1e30px; } \
         .b { width: 50%; height: 1e30px; padding: 1e25px; line-height: 1e30; }",
    );
    assert_layout_sane(&l);
    // Clamped to the documented ceiling, not infinity.
    let a = &body(&l).children[0];
    assert!(a.rect.w <= 1e15);
}

#[test]
fn nonfinite_viewport_stays_sane() {
    for vp in [
        Size {
            w: f32::NAN,
            h: 600.0,
        },
        Size {
            w: 800.0,
            h: f32::NAN,
        },
        Size {
            w: f32::INFINITY,
            h: f32::NEG_INFINITY,
        },
        Size {
            w: -100.0,
            h: -50.0,
        },
    ] {
        let (doc, _) = bowser_dom::Document::parse_html(b"<div>x</div>");
        let ss = bowser_css::parse_stylesheet("div { width: 50vw; }");
        let l = layout(&doc, &[&ss], vp, &FixedFontMetrics);
        assert_layout_sane(&l);
    }
}

#[test]
fn nonfinite_intrinsics_stay_sane() {
    let (doc, _) = bowser_dom::Document::parse_html(b"<img>");
    let img = body_child_tag(&doc, "img");
    for size in [
        IntrinsicDims {
            w: Some(f32::NAN),
            h: Some(10.0),
            ratio: None,
        },
        IntrinsicDims {
            w: Some(f32::INFINITY),
            h: Some(f32::INFINITY),
            ratio: None,
        },
        IntrinsicDims {
            w: Some(-30.0),
            h: Some(-20.0),
            ratio: None,
        },
        IntrinsicDims {
            w: Some(1e30),
            h: Some(1e30),
            ratio: None,
        },
    ] {
        let mut intrinsics = HashMap::new();
        intrinsics.insert(img, size);
        let l = layout_doc_with_intrinsics(&doc, "", &FixedFontMetrics, &intrinsics);
        assert_layout_sane(&l);
    }
}

#[test]
fn hostile_text_layout_stays_sane() {
    // Huge font + tiny viewport + deep inline nesting.
    let mut html = String::from("<div>");
    for _ in 0..200 {
        html.push_str("<span>");
    }
    html.push_str("hello world hello world");
    for _ in 0..200 {
        html.push_str("</span>");
    }
    html.push_str("</div>");
    let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
    let ss = bowser_css::parse_stylesheet("div { font-size: 1e20px; width: 1px; }");
    let l = layout(&doc, &[&ss], Size { w: 1.0, h: 1.0 }, &FixedFontMetrics);
    assert_layout_sane(&l);
}

// ---- harsh-review fix verification tests ----

#[test]
fn fixed_position_ignores_in_flow_ancestor_shifts() {
    // A fixed box is positioned against the viewport: the relative
    // offset of an in-flow ancestor must not shift it (issue 1).
    let l = layout_html(
        r#"<div class="rel"><div class="fix"></div></div>"#,
        ".rel { position: relative; top: 50px; } \
         .fix { position: fixed; top: 0px; left: 0px; width: 10px; height: 10px; }",
    );
    let rel = &body(&l).children[0];
    let fix = &rel.children[0];
    assert!(approx(rel.rect.y, 50.0));
    assert!(approx(fix.rect.x, 0.0));
    assert!(approx(fix.rect.y, 0.0));
}

#[test]
fn bfc_root_shifts_beside_intruding_float() {
    // A BFC root's border box must not overlap an intruding float: it
    // shifts beside the float and narrows to the remaining space
    // (issue 2).
    let l = layout_html(
        r#"<div class="wrap"><div class="fl"></div><div class="bfc"></div></div>"#,
        ".wrap { width: 400px; } \
         .fl { float: left; width: 100px; height: 60px; } \
         .bfc { overflow: hidden; height: 20px; }",
    );
    let wrap = &body(&l).children[0];
    let cx = wrap.rect.x;
    let bfc = &wrap.children[1];
    // Shifted past the float, narrowed to the space beside it.
    assert!(approx(bfc.rect.x, cx + 100.0));
    assert!(approx(bfc.rect.w, 300.0));
    // It fits beside the float, so no clearance below it.
    assert!(approx(bfc.rect.y, wrap.rect.y));
}

#[test]
fn bfc_specified_width_clears_below_float() {
    // A BFC root with a specified width that cannot fit beside an
    // intruding float clears below it instead of overlapping it
    // (issue 2: used width must fit beside floats).
    let l = layout_html(
        r#"<div class="wrap"><div class="fl"></div><div class="bfc"></div></div>"#,
        ".wrap { width: 400px; } \
         .fl { float: left; width: 100px; height: 60px; } \
         .bfc { overflow: hidden; width: 350px; height: 20px; }",
    );
    let wrap = &body(&l).children[0];
    let bfc = &wrap.children[1];
    // 350px does not fit in the 300px beside the float: moved below it.
    assert!(approx(bfc.rect.y, wrap.rect.y + 60.0));
    assert!(approx(bfc.rect.x, wrap.rect.x));
    assert!(approx(bfc.rect.w, 350.0));
}

#[test]
fn bfc_float_band_clamped_by_min_height() {
    // The float band for a BFC root uses the min/max-height-clamped
    // height, not the raw specified height: with `height: 100px;
    // min-height: 200px`, the band is 200px tall. `.fl2` sits below
    // `.fl1` (it cannot fit beside it) and only overlaps the
    // 100-200px extension band, where it is the widest intrusion.
    let l = layout_html(
        r#"<div class="wrap"><div class="fl1"></div><div class="fl2"></div><div class="bfc"></div></div>"#,
        ".wrap { width: 400px; } \
         .fl1 { float: left; width: 100px; height: 100px; } \
         .fl2 { float: left; width: 350px; height: 100px; } \
         .bfc { display: flex; height: 100px; min-height: 200px; }",
    );
    let wrap = &body(&l).children[0];
    let cx = wrap.rect.x;
    let fl2 = &wrap.children[1];
    // .fl2 dropped below .fl1: it spans y 100-200, x 0-350.
    assert!(approx(fl2.rect.y, wrap.rect.y + 100.0));
    assert!(approx(fl2.rect.w, 350.0));
    let bfc = &wrap.children[2];
    // Used height is clamped up by min-height.
    assert!(approx(bfc.rect.h, 200.0));
    // Worst intrusion over the clamped 200px band is 350px (.fl2), so
    // the BFC root narrows to the 50px beside it, not the 300px beside
    // .fl1. Without the clamp its border box would overlap .fl2.
    assert!(approx(bfc.rect.x, cx + 350.0));
    assert!(approx(bfc.rect.w, 50.0));
    assert!(approx(bfc.rect.y, wrap.rect.y));
}

#[test]
fn abspos_solves_width_from_left_and_right() {
    // `width: auto` with both `left` and `right` specified fills the
    // remaining space (CSS 2.1 §10.3.7); not shrink-to-fit (issue 3).
    let l = layout_html(
        r#"<div class="mid"><div class="abs">hi</div></div>"#,
        ".mid { position: relative; width: 400px; height: 300px; } \
         .abs { position: absolute; left: 50px; right: 70px; top: 0px; height: 10px; }",
    );
    let mid = &body(&l).children[0];
    let abs = &mid.children[0];
    assert!(approx(abs.rect.x, mid.rect.x + 50.0));
    assert!(approx(abs.rect.w, 400.0 - 50.0 - 70.0));
}

#[test]
fn abspos_solves_height_from_top_and_bottom() {
    // `height: auto` with both `top` and `bottom` specified fills the
    // remaining space (CSS 2.1 §10.6.4); not the content height
    // (issue 3).
    let l = layout_html(
        r#"<div class="mid"><div class="abs"></div></div>"#,
        ".mid { position: relative; width: 400px; height: 300px; } \
         .abs { position: absolute; top: 20px; bottom: 30px; left: 0px; width: 10px; }",
    );
    let mid = &body(&l).children[0];
    let abs = &mid.children[0];
    assert!(approx(abs.rect.y, mid.rect.y + 20.0));
    assert!(approx(abs.rect.h, 300.0 - 20.0 - 30.0));
}

#[test]
fn abspos_auto_width_shrink_to_fits_with_single_offset() {
    // Shrink-to-fit still applies when only one horizontal offset is
    // set (issue 3: the solver needs both `left` and `right`).
    let l = layout_html(
        r#"<div class="mid"><div class="abs">hi</div></div>"#,
        ".mid { position: relative; width: 400px; height: 300px; } \
         .abs { position: absolute; left: 50px; top: 0px; height: 10px; }",
    );
    let mid = &body(&l).children[0];
    let abs = &mid.children[0];
    assert!(approx(abs.rect.x, mid.rect.x + 50.0));
    // "hi" at 16px with 0.6 advance per char.
    assert!(approx(abs.rect.w, 2.0 * 0.6 * 16.0));
}

#[test]
fn table_colspan_min_distribution_counts_inter_column_spacing() {
    // The multi-column minimum counts inter-column spacing as part of
    // the current span before distributing excess: with
    // `border-spacing: 10px`, a 48px-minimum colspan=2 cell over
    // columns with minima 6 and 36 leaves no excess
    // (6 + 36 + 10 >= 48), so the minima stay 6 and 36 and the final
    // column widths keep that 1:6 ratio (issue 4).
    let l = layout_html(
        "<table><tr><td>a</td><td>aabbcc</td></tr>\
         <tr><td colspan=\"2\">aabbccdd</td></tr></table>",
        "table { border-spacing: 10px; } td { font-size: 10px; white-space: nowrap; }",
    );
    let c00 = table_cell(&l, 0, 0);
    let c01 = table_cell(&l, 0, 1);
    assert!(approx(c00.rect.w / c01.rect.w, 6.0 / 36.0));
}

#[test]
fn relative_percent_top_against_indefinite_height_is_auto() {
    // `top: 50%` with an auto-height parent computes to `auto` (no
    // offset); unitless lengths still resolve (issue 5, CSS 2.1
    // §10.6.4).
    let l = layout_html(
        r#"<div class="a"><div class="b"></div></div>"#,
        ".a { width: 400px; } .b { position: relative; top: 50%; height: 10px; }",
    );
    let a = &body(&l).children[0];
    let b = &a.children[0];
    assert!(approx(b.rect.y, a.rect.y));
}

#[test]
fn relative_percent_top_against_definite_height_resolves() {
    // With a definite parent height the percentage resolves normally
    // (issue 5).
    let l = layout_html(
        r#"<div class="a"><div class="b"></div></div>"#,
        ".a { width: 400px; height: 200px; } .b { position: relative; top: 50%; height: 10px; }",
    );
    let a = &body(&l).children[0];
    let b = &a.children[0];
    assert!(approx(b.rect.y, a.rect.y + 100.0));
}

#[test]
fn unbreakable_line_shifts_below_full_width_float() {
    // A float covering the full content width leaves no room: the
    // unbreakable (`nowrap`) line shifts below the float instead of
    // overlapping it (issue 6, CSS 2.1 §9.5).
    let l = layout_html(
        r#"<div class="w"><div class="fl"></div><p class="t">hello</p></div>"#,
        ".w { width: 200px; } .fl { float: left; width: 200px; height: 30px; } \
         .t { margin: 0; white-space: nowrap; }",
    );
    let w = &body(&l).children[0];
    let fl = &w.children[0];
    let t = &w.children[1];
    let frags = &t.children[0].fragments;
    assert!(!frags.is_empty());
    assert!(frags[0].rect.y >= fl.rect.y + fl.rect.h - 0.01);
}

#[test]
fn nowrap_line_shifts_below_narrowing_float() {
    // Floats narrow the line without covering it fully: a `nowrap`
    // line wider than the narrowed space still shifts below the float
    // instead of overlapping it (issue 6). With no float intrusion an
    // over-wide line overflows normally.
    let l = layout_html(
        r#"<div class="w"><div class="fl"></div><p class="t">aabbccddee</p></div>"#,
        ".w { width: 200px; } .fl { float: left; width: 150px; height: 30px; } \
         .t { margin: 0; white-space: nowrap; }",
    );
    let w = &body(&l).children[0];
    let fl = &w.children[0];
    let t = &w.children[1];
    let frags = &t.children[0].fragments;
    assert!(!frags.is_empty());
    // 10 chars at 16px = 96px > 50px narrowed space: below the float.
    assert!(frags[0].rect.y >= fl.rect.y + fl.rect.h - 0.01);
}

#[test]
fn flex_out_of_flow_first_child_does_not_disturb_item_margins() {
    // The `mt_place_done` reset in `layout_flex` must use the same
    // first-in-flow-block-child predicate as `compute_mt_place`: a
    // leading out-of-flow child is skipped, and the real first item's
    // margins lay out normally (issue 7).
    let l = layout_html(
        r#"<div class="flex"><div class="abs"></div><div class="item"><div class="inner"></div></div></div>"#,
        ".flex { display: flex; width: 400px; } \
         .abs { position: absolute; width: 10px; height: 10px; } \
         .item { flex-basis: 100px; } \
         .inner { margin-top: 50%; height: 10px; }",
    );
    let flexb = &body(&l).children[0];
    let item = flexb
        .children
        .iter()
        .find(|c| approx(c.rect.w, 100.0))
        .expect("flex item");
    // The item's first in-flow child has `margin-top: 50%` of the
    // flexed 100px width: the item starts 50px below content top.
    assert!(approx(item.rect.y, flexb.rect.y + 50.0));
}

#[test]
fn flex_wrap_reverse_align_start_goes_to_cross_start() {
    // With `wrap-reverse` the cross-start direction flips: `align-items:
    // flex-start` aligns items to the physical bottom of each line
    // (issue 8).
    let l = layout_html(
        r#"<div class="c"><div class="a"></div><div class="b"></div><div class="e"></div></div>"#,
        ".c { display: flex; flex-wrap: wrap-reverse; width: 250px; \
             align-items: flex-start; align-content: flex-start; } \
         .a { width: 100px; height: 20px; } \
         .b { width: 100px; height: 40px; } \
         .e { width: 100px; height: 10px; }",
    );
    let c = &body(&l).children[0];
    let (a, b, e) = (&c.children[0], &c.children[1], &c.children[2]);
    // Two lines: [a, b] then [e]. With wrap-reverse the first line
    // stacks at the physical bottom.
    assert!(approx(e.rect.y, c.rect.y));
    assert!(approx(b.rect.y, c.rect.y + 10.0));
    // The shorter item `a` is bottom-aligned with `b` (cross-start =
    // the line's physical bottom).
    assert!(approx(a.rect.y, b.rect.y + b.rect.h - a.rect.h));
    assert!(approx(a.rect.y, c.rect.y + 30.0));
}

// ===== regression tests: grid auto-placement, clearance, abspos grid
// static position, nested fragmentation =====

fn frag_pages(html: &str, css: &str, page_h: f32) -> Vec<super::PageFragment> {
    let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
    let ss = bowser_css::parse_stylesheet(css);
    super::layout_fragmented(&doc, &[&ss], VP, &FixedFontMetrics, page_h)
}

#[test]
fn grid_span_only_row_keeps_span() {
    // `grid-row: span 2` is not a definite position, but the span is
    // retained by auto-placement (css-grid-1 §8.3): the first item
    // occupies two 20px rows and the next auto item starts in row 3.
    let l = layout_html(
        r#"<div class="g"><div class="a">x</div><div class="b">y</div></div>"#,
        ".g { display: grid; grid-template-columns: 100px; grid-auto-rows: 20px; } \
         .a { grid-row: span 2; } \
         .b { height: 10px; }",
    );
    let g = &body(&l).children[0];
    let (a, b) = (&g.children[0], &g.children[1]);
    assert!(approx(a.rect.h, 40.0));
    assert!(approx(b.rect.y, a.rect.y + 40.0));
}

#[test]
fn grid_span_only_column_keeps_span() {
    // `grid-column: span 2` auto-places with a 2-column span.
    let l = layout_html(
        r#"<div class="g"><div class="a">x</div><div class="b">y</div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px 100px; } \
         .a { grid-column: span 2; height: 10px; } \
         .b { height: 10px; }",
    );
    let g = &body(&l).children[0];
    let (a, b) = (&g.children[0], &g.children[1]);
    assert!(approx(a.rect.x, g.rect.x));
    assert!(approx(a.rect.w, 200.0));
    assert!(approx(b.rect.x, g.rect.x + 200.0));
    assert!(approx(b.rect.y, a.rect.y));
}

#[test]
fn grid_definite_start_span_end() {
    // `grid-column: 2 / span 2` occupies columns 2 and 3; the auto item
    // takes the first free cell.
    let l = layout_html(
        r#"<div class="g"><div class="a">x</div><div class="b">y</div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px 100px; } \
         .a { grid-column: 2 / span 2; height: 10px; } \
         .b { height: 10px; }",
    );
    let g = &body(&l).children[0];
    let (a, b) = (&g.children[0], &g.children[1]);
    assert!(approx(a.rect.x, g.rect.x + 100.0));
    assert!(approx(a.rect.w, 200.0));
    assert!(approx(b.rect.x, g.rect.x));
}

#[test]
fn grid_negative_line_end() {
    // `grid-column: 3 / -1` is the third column of a 3-column grid.
    let l = layout_html(
        r#"<div class="g"><div class="a">x</div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px 100px; } \
         .a { grid-column: 3 / -1; height: 10px; }",
    );
    let g = &body(&l).children[0];
    let a = &g.children[0];
    assert!(approx(a.rect.x, g.rect.x + 200.0));
    assert!(approx(a.rect.w, 100.0));
}

#[test]
fn grid_definite_item_does_not_advance_cursor() {
    // Fully definite items are placed by spec step 1 and never move the
    // auto-placement cursor (css-grid-1 §8.5): the auto item still takes
    // the first free cell even though the definite item came first.
    let l = layout_html(
        r#"<div class="g"><div class="a">x</div><div class="b">y</div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px; } \
         .a { grid-column: 2; grid-row: 1; height: 10px; } \
         .b { height: 10px; }",
    );
    let g = &body(&l).children[0];
    let (a, b) = (&g.children[0], &g.children[1]);
    assert!(approx(a.rect.x, g.rect.x + 100.0));
    assert!(approx(b.rect.x, g.rect.x));
    assert!(approx(b.rect.y, a.rect.y));
}

#[test]
fn grid_spanning_auto_item_wraps_past_cursor() {
    // The cursor sits at column 2 after the first item; a `span 3` auto
    // item cannot fit beside it, so it wraps to a fresh row instead of
    // growing implicit columns mid-row (css-grid-1 §8.5).
    let l = layout_html(
        r#"<div class="g"><div class="a">x</div><div class="b">y</div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px 100px; } \
         .a { height: 10px; } \
         .b { grid-column: span 3; height: 10px; }",
    );
    let g = &body(&l).children[0];
    let (a, b) = (&g.children[0], &g.children[1]);
    assert!(approx(b.rect.x, g.rect.x));
    assert!(approx(b.rect.w, 300.0));
    assert!(approx(b.rect.y, a.rect.y + 10.0));
}

#[test]
fn grid_large_auto_span_grows_implicit_grid() {
    // The implicit column count is sized up front for the largest auto
    // column span (css-grid-1 §8.5 step 3): a `span 3` item in a 2-column
    // grid occupies three 100px columns on the first row instead of
    // wrapping.
    let l = layout_html(
        r#"<div class="g"><div class="a">x</div><div class="b">y</div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 100px; grid-auto-columns: 100px; } \
         .a { grid-column: span 3; height: 10px; } \
         .b { height: 10px; }",
    );
    let g = &body(&l).children[0];
    let (a, b) = (&g.children[0], &g.children[1]);
    assert!(approx(a.rect.x, g.rect.x));
    assert!(approx(a.rect.w, 300.0));
    assert!(approx(b.rect.y, a.rect.y + 10.0));
}

#[test]
fn clear_negative_margin_keeps_border_edge_below_float() {
    // Clearance is computed from the hypothetical top border edge
    // (CSS 2.1 §9.5.2): the negative margin folds into that edge, so the
    // border top still lands at or below the float bottom instead of
    // overlapping it. Asserted relative to the float because the
    // negative margin also collapses through the body's top (the
    // engine's uniform first-child through-collapse, same as without
    // `clear`), shifting both boxes up together.
    let l = layout_html(
        r#"<div class="f"></div><div class="c"></div>"#,
        ".f { float: left; width: 100px; height: 100px; } \
         .c { clear: both; margin-top: -30px; height: 10px; }",
    );
    let b = body(&l);
    let (f, c) = (&b.children[0], &b.children[1]);
    assert!(approx(c.rect.y, f.rect.y + f.rect.h));
}

#[test]
fn abspos_grid_static_position_honors_self_alignment() {
    // The static position of an abspos grid child is determined as if it
    // were the sole grid item in the content box (css-grid-1 §9.2):
    // `justify-items: center; align-items: center` centers the 50px box
    // in the 300x200 grid.
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; width: 300px; height: 200px; \
             grid-template-columns: 300px; justify-items: center; align-items: center; } \
         .a { position: absolute; width: 50px; height: 50px; }",
    );
    let g = &body(&l).children[0];
    let a = &g.children[0];
    assert!(approx(a.rect.x, g.rect.x + 125.0));
    assert!(approx(a.rect.y, g.rect.y + 75.0));
}

#[test]
fn abspos_grid_self_align_end_with_margin_is_border_box_origin() {
    // Regression: the static-position alignment for an abspos grid child
    // must produce the BORDER-box origin for every alignment arm. The End
    // arm previously computed the margin-box left edge
    // (static + cw - ml - bw - mr), misplacing the box left by exactly
    // the start margin when `justify-self: end` (or `align-self: end`)
    // combined with a nonzero start margin.
    let l = layout_html(
        r#"<div class="g"><div class="a"></div></div>"#,
        ".g { display: grid; width: 300px; height: 200px; \
             grid-template-columns: 300px; } \
         .a { position: absolute; width: 50px; height: 50px; \
              justify-self: end; align-self: end; \
              margin-left: 30px; margin-top: 30px; }",
    );
    let g = &body(&l).children[0];
    let a = &g.children[0];
    // margin box right-aligned to the content box: border-box origin x
    // = static_x + cw - mr - bw = 300 - 0 (mr) - 50 (bw) = 250;
    // the 30px start margin is part of the margin box and is NOT added.
    assert!(approx(a.rect.x, g.rect.x + 250.0));
    assert!(approx(a.rect.y, g.rect.y + 150.0));
}

#[test]
fn float_contained_by_nested_bfc() {
    // A float inside an `overflow: hidden` wrapper is contained by that
    // wrapper's BFC: the wrapper grows around it and a following
    // `clear: both` sibling starts right after the wrapper.
    let l = layout_html(
        r#"<div class="w"><div class="f"></div></div><div class="c"></div>"#,
        ".w { overflow: hidden; } .f { float: left; width: 100px; height: 80px; } \
         .c { clear: both; height: 10px; }",
    );
    let b = body(&l);
    let (w, c) = (&b.children[0], &b.children[1]);
    assert!(approx(w.rect.h, 80.0));
    assert!(approx(c.rect.y, w.rect.y + 80.0));
}

#[test]
fn frag_nested_avoid_moves_whole_without_extra_page() {
    // A nested `break-inside: avoid` box straddling a page boundary moves
    // whole to the next page; its lines must not count toward the outer
    // block's widows/orphans and shove the whole unit down a page.
    let pages = frag_pages(
        r#"<div class="outer"><div class="pad"></div><div class="avoid">x</div></div>"#,
        ".pad { height: 550px; } .avoid { break-inside: avoid; height: 100px; }",
        600.0,
    );
    assert_eq!(pages.len(), 2);
}

#[test]
fn frag_nested_break_before() {
    // A nested `break-before: page` starts a new page.
    let pages = frag_pages(
        r#"<div class="outer"><div class="pad"></div><div class="nb">x</div></div>"#,
        ".pad { height: 100px; } .nb { break-before: page; height: 50px; }",
        600.0,
    );
    assert_eq!(pages.len(), 2);
}

#[test]
fn frag_taller_than_page_avoid_fragments() {
    // A taller-than-page `break-inside: avoid` box cannot move whole; it
    // fragments instead of hanging the paginator.
    let pages = frag_pages(
        r#"<div class="tall">x</div>"#,
        ".tall { break-inside: avoid; height: 2000px; }",
        600.0,
    );
    assert_eq!(pages.len(), 5);
}

#[test]
fn nested_subgrid_lays_out() {
    // A subgrid inside a subgrid defers to the outermost grid's track
    // listing; it must lay out without panicking.
    let l = layout_html(
        r#"<div class="g"><div class="s"><div class="s2"><div class="i">x</div></div></div></div>"#,
        ".g { display: grid; grid-template-columns: 100px 200px; } \
         .s { display: grid; grid-template-columns: subgrid; grid-column: 1 / 3; } \
         .s2 { display: grid; grid-template-columns: subgrid; grid-column: 1 / 3; } \
         .i { height: 10px; }",
    );
    let g = &body(&l).children[0];
    assert_eq!(g.children.len(), 1);
}

/// Find the layout box built for `node`.
fn find_box(root: &LayoutBox, node: bowser_dom::NodeId) -> Option<&LayoutBox> {
    if root.node == Some(node) {
        return Some(root);
    }
    root.children.iter().find_map(|c| find_box(c, node))
}

#[test]
fn snap_areas_collect_in_flow_targets_with_scroll_margin() {
    // In-flow descendants with `scroll-snap-align != none` become snap
    // areas: border box expanded by `scroll-margin`, in the container's
    // content coordinates. Out-of-flow (`position: fixed`) boxes and
    // `align: none` boxes are excluded.
    let (doc, _) = bowser_dom::Document::parse_html(
        "<html><body><div id=\"scroller\"><div id=\"a\">x</div><div id=\"b\">y</div><div id=\"c\">z</div><div id=\"f\">w</div></div></body></html>"
            .as_bytes(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#scroller { overflow: scroll; height: 100px; width: 200px; } \
         #a { height: 50px; scroll-snap-align: start; scroll-margin-top: 10px; } \
         #b { height: 50px; scroll-snap-align: center end; } \
         #c { height: 50px; } \
         #f { position: fixed; height: 50px; scroll-snap-align: start; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let scroller = doc.get_element_by_id("scroller").expect("scroller");
    let areas = snap_areas_for_container(&root, scroller, &doc);
    assert_eq!(
        areas.len(),
        2,
        "#a and #b are snap targets; #c has align none, #f is fixed"
    );
    // A single `scroll-snap-align` keyword applies to both axes.
    // `#b` is `center end`: the first value is the block axis, the
    // second the inline axis, so in horizontal-tb `align_x` (inline)
    // is `End` and `align_y` (block) is `Center`.
    assert_eq!(areas[0].align_x, SnapAlign::Start);
    assert_eq!(areas[0].align_y, SnapAlign::Start);
    assert_eq!(areas[1].align_x, SnapAlign::End);
    assert_eq!(areas[1].align_y, SnapAlign::Center);
    // `#a` is the first in-flow child: its snap rect starts at the
    // container's content origin, expanded 10px upward by
    // `scroll-margin-top`. `#scroller` is `overflow: scroll`, so its
    // forced vertical scrollbar consumes SCROLLBAR_THICKNESS from the
    // content box and `#a` fills the narrowed width.
    assert!(approx(areas[0].rect.x, 0.0));
    assert!(approx(areas[0].rect.y, -10.0));
    assert!(approx(areas[0].rect.w, 200.0 - crate::SCROLLBAR_THICKNESS));
    assert!(approx(areas[0].rect.h, 60.0));
    // `#b` follows `#a` (50px tall): no scroll margin, so its rect is
    // its border box in content coordinates.
    assert!(approx(areas[1].rect.x, 0.0));
    assert!(approx(areas[1].rect.y, 50.0));
    assert!(approx(areas[1].rect.h, 50.0));
    // A node id from another document matches nothing: no areas.
    let (doc2, _) = bowser_dom::Document::parse_html(
        "<html><body><div id=\"x\">q</div></body></html>".as_bytes(),
    );
    let foreign = doc2.get_element_by_id("x").expect("x");
    assert!(snap_areas_for_container(&root, foreign, &doc).is_empty());
}

#[test]
fn snap_areas_includes_fixed_in_transformed() {
    // A snap-aligned fixed box under a non-`none` `transform` between
    // it and the scroll container takes its containing block from that
    // transformed ancestor (CSS Transforms 1 §3) and scrolls with the
    // container's content, so it is a snap target — unlike a
    // viewport-anchored fixed box, which is still skipped.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"s\"><div id=\"t\"><div id=\"f\">x</div></div><div id=\"v\">y</div></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } \
         #s { overflow-y: auto; width: 200px; height: 100px; } \
         #t { transform: translate(0, 0); height: 50px; } \
         #f { position: fixed; top: 20px; left: 0; width: 50px; height: 50px; scroll-snap-align: start; } \
         #v { position: fixed; top: 300px; left: 0; width: 50px; height: 50px; scroll-snap-align: start; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let s = doc.get_element_by_id("s").expect("s");
    let areas = snap_areas_for_container(&root, s, &doc);
    assert_eq!(
        areas.len(),
        1,
        "only the fixed-in-transformed box is a snap target; the viewport-anchored fixed box is skipped"
    );
    // A single `scroll-snap-align` keyword applies to both axes.
    assert_eq!(areas[0].align_x, SnapAlign::Start);
    assert_eq!(areas[0].align_y, SnapAlign::Start);
    // `#f` is laid out `top: 20px` against `#t`'s padding box at the
    // container's content origin.
    assert!(approx(areas[0].rect.x, 0.0));
    assert!(approx(areas[0].rect.y, 20.0));
    assert!(approx(areas[0].rect.w, 50.0));
    assert!(approx(areas[0].rect.h, 50.0));
}

#[test]
fn snap_areas_skips_fixed_when_container_itself_transformed() {
    // A transformed scroll container is not a fixed containing block
    // inside itself: its fixed child is laid out against the
    // container's padding box (the scrollport frame), which does not
    // move when the container scrolls, so the child is still skipped —
    // only a transformed box strictly below the container makes a fixed
    // descendant a snap target.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"s\"><div id=\"f\">x</div></div></body></html>".as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } \
         #s { overflow-y: auto; width: 200px; height: 100px; transform: translate(0, 0); } \
         #f { position: fixed; top: 20px; left: 0; width: 50px; height: 50px; scroll-snap-align: start; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let s = doc.get_element_by_id("s").expect("s");
    assert!(
        snap_areas_for_container(&root, s, &doc).is_empty(),
        "a fixed child of a transformed scroll container is anchored to the scrollport frame, not a snap target"
    );
}

#[test]
fn snap_areas_viewport_includes_abspos_against_icb() {
    // `:root { scroll-snap-type }` with `position: absolute` snap
    // points and nothing positioned: their containing block is the
    // ICB, which scrolls with the document, so for the viewport
    // scroller (the tree root box) they are legitimate snap areas.
    // Previously the walk seeded `cb_inside` from the static root
    // box and dropped them all, leaving the viewport's area list
    // empty (no snap events ever fired on root programmatic scroll).
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"a\">x</div><div id=\"b\">y</div></body></html>".as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } \
         #a { position: absolute; top: 10px; left: 0; width: 50px; height: 50px; scroll-snap-align: start; } \
         #b { position: absolute; top: 700px; left: 0; width: 50px; height: 50px; scroll-snap-align: start; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let html = doc.document_element().expect("html");
    assert_eq!(
        root.node,
        Some(html),
        "the tree root is the viewport container"
    );
    let areas = snap_areas_for_container(&root, html, &doc);
    assert_eq!(
        areas.len(),
        2,
        "abspos snap points against the ICB participate for the viewport scroller"
    );
    assert!(approx(areas[0].rect.y, 10.0));
    assert!(approx(areas[1].rect.y, 700.0));
}

#[test]
fn snap_areas_nested_static_scroller_excludes_abspos_against_icb() {
    // The same abspos boxes under a nested *static* scroller: their
    // containing block (the ICB) lies outside the scroller, so they
    // still do not participate — the viewport exception must not leak
    // into nested containers.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"s\"><div id=\"a\">x</div><div id=\"b\">y</div></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } \
         #s { overflow-y: auto; width: 200px; height: 100px; } \
         #a { position: absolute; top: 10px; left: 0; width: 50px; height: 50px; scroll-snap-align: start; } \
         #b { position: absolute; top: 700px; left: 0; width: 50px; height: 50px; scroll-snap-align: start; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let s = doc.get_element_by_id("s").expect("s");
    assert!(
        snap_areas_for_container(&root, s, &doc).is_empty(),
        "abspos boxes against the ICB are not snap areas of a nested static scroller"
    );
}

#[test]
fn scroll_containers_reports_nested_scroller_geometry() {
    // A nested `overflow: scroll` box inside a scrolled viewport: layout
    // reports the box as a scroll container — its border box as the
    // scrollport, content-minus-scrollport as the maximum — so the app
    // can build nested scrollers from real layout output.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"inner\"><div id=\"tall\">x</div></div></body></html>".as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#inner { overflow: scroll; height: 100px; width: 200px; } #tall { height: 400px; }",
    );
    let root = layout_with_scroll(&doc, &[&ss], VP, &FixedFontMetrics, (0.0, 50.0));
    let inner = doc.get_element_by_id("inner").expect("inner");
    let containers = scroll_containers(&root);
    assert_eq!(containers.len(), 1, "only #inner is a container");
    let c = &containers[0];
    assert_eq!(c.node, Some(inner));
    // The scrollport is the box's own border box, in page coordinates.
    let box_rect = find_box(&root, inner).expect("inner box").rect;
    assert_eq!(c.scrollport, box_rect);
    assert_eq!((c.scrollport.w, c.scrollport.h), (200.0, 100.0));
    // 400px of content in a 100px scrollport scrolls 300px vertically.
    assert_eq!(c.max, (0.0, 300.0));
    assert_eq!(c.axes, (true, true));
    assert_eq!(c.user_axes, (true, true));
    assert_eq!(c.depth, 2, "html(0) > body(1) > #inner(2)");
}

#[test]
fn overflow_clip_is_not_a_scroll_container() {
    // CSS Overflow 3 §2: `overflow: clip` clips like `hidden` but does
    // not establish a scroll container — no ScrollTree entry (hence no
    // scrollbar and no scroller for `scrollTop` writes to target).
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"clip\"><div id=\"tall\">x</div></div></body></html>".as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#clip { overflow: clip; height: 100px; width: 200px; } #tall { height: 400px; }",
    );
    let root = layout_with_scroll(&doc, &[&ss], VP, &FixedFontMetrics, (0.0, 0.0));
    let clip = doc.get_element_by_id("clip").expect("clip");
    let containers = scroll_containers(&root);
    assert!(
        containers.iter().all(|c| c.node != Some(clip)),
        "overflow: clip must not mint a scroll container even with overflowing content"
    );
    // The classification itself: neither a container nor user-scrollable.
    let box_ = find_box(&root, clip).expect("clip box");
    let style = box_.style.as_ref().expect("clip style");
    assert_eq!(overflow_axes(style), ((false, false), (false, false)));
    // A lone `clip` axis pairs with the other axis's `visible` per the
    // visible→auto rule, so it still forms a one-axis container there.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"mixed\"><div id=\"tall\">x</div></div></body></html>".as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#mixed { overflow-x: clip; height: 100px; width: 200px; } #tall { height: 400px; }",
    );
    let root = layout_with_scroll(&doc, &[&ss], VP, &FixedFontMetrics, (0.0, 0.0));
    let mixed = doc.get_element_by_id("mixed").expect("mixed");
    let box_ = find_box(&root, mixed).expect("mixed box");
    let style = box_.style.as_ref().expect("mixed style");
    assert_eq!(overflow_axes(style), ((false, true), (false, true)));
}

#[test]
fn scroll_containers_origin_is_zero_without_negative_overflow() {
    // All content sits at or right/below the scrollport origin: the new
    // `scroll_origin` defaults to (0,0) and `max` is computed exactly as
    // before.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"inner\"><div id=\"tall\">x</div></div></body></html>".as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#inner { overflow: scroll; height: 100px; width: 200px; } #tall { height: 400px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let inner = doc.get_element_by_id("inner").expect("inner");
    let c = scroll_containers(&root)
        .into_iter()
        .find(|c| c.node == Some(inner))
        .expect("inner container");
    assert_eq!(c.scroll_origin, (0.0, 0.0));
    assert_eq!(c.max, (0.0, 300.0));
}

#[test]
fn scroll_containers_origin_tracks_negative_overflow() {
    // A relatively positioned child shifted above/left of the scrollport
    // origin: `scroll_origin` reports the negative minimum offset, while
    // `max` still only measures the far edges.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"inner\"><div id=\"neg\">x</div><div id=\"tall\">x</div></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#inner { overflow: scroll; height: 100px; width: 200px; } \
         #neg { position: relative; left: -40px; top: -30px; height: 50px; } \
         #tall { height: 400px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let inner = doc.get_element_by_id("inner").expect("inner");
    let c = scroll_containers(&root)
        .into_iter()
        .find(|c| c.node == Some(inner))
        .expect("inner container");
    assert_eq!(c.scroll_origin, (-40.0, -30.0));
    // `relative` shifts the box visually but not its flow footprint: #tall
    // still starts 50px below the container top and runs 400px, so the far
    // edge (and `max`) is unchanged by the visual shift.
    assert_eq!(c.max, (0.0, 350.0));
}

#[test]
fn scroll_containers_nests_and_clips_extents() {
    // Nested containers: the inner container's overflowing content is
    // clipped by its own scrollport and must not extend the outer
    // container's range; order is preorder (parents before children).
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"a\"><div id=\"b\"><div id=\"c\">x</div></div></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#a { overflow-y: auto; height: 200px; } \
         #b { overflow-y: auto; height: 100px; } \
         #c { height: 500px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let containers = scroll_containers(&root);
    let a = doc.get_element_by_id("a").expect("a");
    let b = doc.get_element_by_id("b").expect("b");
    let names: Vec<_> = containers.iter().map(|c| c.node).collect();
    assert_eq!(names, vec![Some(a), Some(b)], "preorder");
    assert_eq!(
        containers[0].max,
        (0.0, 0.0),
        "b's overflow is clipped by b"
    );
    assert_eq!(containers[1].max, (0.0, 400.0));
    assert!(containers[0].depth < containers[1].depth);
}

#[test]
fn scroll_containers_fixed_descendant_does_not_extend_max() {
    // A `position: fixed` child is viewport-anchored: it never scrolls
    // with the container, so it must not extend the container's `max`.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"s\"><div id=\"short\">x</div><div id=\"f\">y</div></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#s { overflow: auto; width: 100px; height: 100px; } \
         #short { height: 50px; } \
         #f { position: fixed; top: 5000px; height: 50px; }",
    );
    let root = layout_with_scroll(&doc, &[&ss], VP, &FixedFontMetrics, (0.0, 0.0));
    let s = doc.get_element_by_id("s").expect("s");
    let c = scroll_containers(&root)
        .into_iter()
        .find(|c| c.node == Some(s))
        .expect("s container");
    assert_eq!(
        c.max,
        (0.0, 0.0),
        "fixed child is viewport-anchored, not s's scrollable overflow"
    );
}

#[test]
fn scroll_containers_abspos_outside_scroller_does_not_extend_max() {
    // An abspos child whose containing block is the ICB (outside the
    // static scroller) does not scroll with the scroller.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"s3\"><div id=\"a3\">z</div></div></body></html>".as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#s3 { overflow: auto; width: 100px; height: 100px; } \
         #a3 { position: absolute; top: 3000px; height: 50px; }",
    );
    let root = layout_with_scroll(&doc, &[&ss], VP, &FixedFontMetrics, (0.0, 0.0));
    let s3 = doc.get_element_by_id("s3").expect("s3");
    let c = scroll_containers(&root)
        .into_iter()
        .find(|c| c.node == Some(s3))
        .expect("s3 container");
    assert_eq!(
        c.max,
        (0.0, 0.0),
        "abspos child with CB outside the scroller does not scroll with it"
    );
}

#[test]
fn scroll_containers_abspos_against_scroller_extends_max() {
    // An abspos child of a *positioned* scroller takes its containing
    // block from the scroller itself, scrolls with it, and counts.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"s\"><div id=\"a\">z</div></div></body></html>".as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#s { position: relative; overflow: auto; width: 100px; height: 100px; } \
         #a { position: absolute; top: 3000px; height: 50px; }",
    );
    let root = layout_with_scroll(&doc, &[&ss], VP, &FixedFontMetrics, (0.0, 0.0));
    let s = doc.get_element_by_id("s").expect("s");
    let c = scroll_containers(&root)
        .into_iter()
        .find(|c| c.node == Some(s))
        .expect("s container");
    assert_eq!(
        c.max,
        (0.0, 2950.0),
        "abspos child with CB == the scroller scrolls with it"
    );
}

#[test]
fn scroll_containers_with_offset_translates_fixed_scrollport() {
    // A fixed scroll container's rect is viewport-anchored; with the
    // viewport scrolled, the reported scrollport must be in absolute
    // page coordinates (what `ScrollContainerInfo` documents).
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"tall\">x</div><div id=\"fx\"><div id=\"in\">y</div></div><div id=\"outer\"><div id=\"nested\"><div id=\"deep\">z</div></div></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#tall { height: 2000px; } \
         #fx { position: fixed; top: 10px; left: 10px; width: 100px; height: 100px; overflow: auto; } \
         #in { height: 500px; } \
         #outer { position: fixed; top: 5px; left: 300px; } \
         #nested { overflow: auto; width: 80px; height: 60px; } \
         #deep { height: 400px; }",
    );
    let root = layout_with_scroll(&doc, &[&ss], VP, &FixedFontMetrics, (0.0, 500.0));
    let fx = doc.get_element_by_id("fx").expect("fx");
    let nested = doc.get_element_by_id("nested").expect("nested");
    let containers = scroll_containers_with_offset(&root, (0.0, 500.0));
    let c = containers
        .iter()
        .find(|c| c.node == Some(fx))
        .expect("fx container");
    assert_eq!(
        (c.scrollport.x, c.scrollport.y),
        (10.0, 510.0),
        "fixed scrollport translated into page coordinates"
    );
    // A wheel at viewport (20, 20) addresses page (20, 520): inside the
    // translated scrollport, as the app's hit test computes it.
    assert!(
        20.0 >= c.scrollport.x
            && 20.0 < c.scrollport.x + c.scrollport.w
            && 520.0 >= c.scrollport.y
            && 520.0 < c.scrollport.y + c.scrollport.h,
        "hit test at page (20, 520) must land inside fx"
    );
    // `max` is translation-invariant: 500px of content, 100px scrollport.
    assert_eq!(c.max, (0.0, 400.0));
    // A static container inside a fixed subtree is viewport-anchored
    // too, even though it is not itself `position: fixed`.
    let n = containers
        .iter()
        .find(|c| c.node == Some(nested))
        .expect("nested container");
    assert_eq!(
        (n.scrollport.x, n.scrollport.y),
        (300.0, 505.0),
        "static container in a fixed subtree is viewport-anchored"
    );
    assert_eq!(n.max, (0.0, 340.0));
    // The zero-offset variant keeps the historical viewport-coords
    // behavior for fixed boxes.
    let c0 = scroll_containers(&root)
        .into_iter()
        .find(|c| c.node == Some(fx))
        .expect("fx container");
    assert_eq!((c0.scrollport.x, c0.scrollport.y), (10.0, 10.0));
}

#[test]
fn scroll_containers_with_offset_fixed_in_transformed_not_translated() {
    // A fixed box under a non-`none` `transform` is positioned against
    // the transformed ancestor (CSS Transforms 1 §3) and scrolls with
    // it: its rect is already in page coordinates, so
    // `scroll_containers_with_offset` must NOT translate it by the
    // viewport scroll offset. The same holds for a scroll container
    // nested inside the fixed subtree.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"t\"><div id=\"fx\"><div id=\"in\">y</div><div id=\"nested\"><div id=\"deep\">z</div></div></div></div><div id=\"tall\">x</div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } \
         #tall { height: 2000px; } \
         #t { transform: translate(10px, 20px); } \
         #fx { position: fixed; top: 10px; left: 10px; width: 100px; height: 100px; overflow: auto; } \
         #in { height: 500px; } \
         #nested { overflow: auto; width: 80px; height: 60px; } \
         #deep { height: 400px; }",
    );
    let root = layout_with_scroll(&doc, &[&ss], VP, &FixedFontMetrics, (0.0, 500.0));
    let fx = doc.get_element_by_id("fx").expect("fx");
    let nested = doc.get_element_by_id("nested").expect("nested");
    let containers = scroll_containers_with_offset(&root, (0.0, 500.0));
    let c = containers
        .iter()
        .find(|c| c.node == Some(fx))
        .expect("fx container");
    // Page coordinates already: laid out against `#t`'s padding box at
    // the page origin, not the viewport.
    assert_eq!(
        (c.scrollport.x, c.scrollport.y),
        (10.0, 10.0),
        "fixed-in-transformed scrollport is not translated"
    );
    // `max` is translation-invariant: `#in` contributes 500px of content
    // (400 past the 100px scrollport) and `#nested`'s own 60px box sits
    // below it (`#nested` is a confirmed nested container, so its
    // scrolled content does not count).
    assert_eq!(c.max, (0.0, 460.0));
    // A container inside the fixed-in-transformed subtree scrolls with
    // the transformed ancestor too: `#nested` sits below `#in` at
    // y=510 in page coordinates, untranslated.
    let n = containers
        .iter()
        .find(|c| c.node == Some(nested))
        .expect("nested container");
    assert_eq!(
        (n.scrollport.x, n.scrollport.y),
        (10.0, 510.0),
        "container in a fixed-in-transformed subtree is not viewport-anchored"
    );
    assert_eq!(n.max, (0.0, 340.0));
}

#[test]
fn scrollable_extent_includes_fixed_in_transformed() {
    // A fixed box whose containing block is a transformed ancestor
    // inside the scroller scrolls with the scroller, so it extends the
    // scroller's scrollable overflow range (unlike a viewport-anchored
    // fixed box, which never does).
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"s\"><div id=\"t\"><div id=\"f\">x</div></div></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } \
         #s { overflow-y: auto; width: 200px; height: 100px; } \
         #t { transform: translate(0, 0); height: 50px; } \
         #f { position: fixed; top: 300px; left: 0; width: 50px; height: 50px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let s = doc.get_element_by_id("s").expect("s");
    let c = scroll_containers(&root)
        .into_iter()
        .find(|c| c.node == Some(s))
        .expect("s container");
    // `#f` sits at y=300..350 against `#t`'s padding box; `#s` is 100px
    // tall, so the range extends 250px past the scrollport.
    assert_eq!(c.max, (0.0, 250.0));
}

#[test]
fn scrollable_extent_still_skips_viewport_anchored_fixed() {
    // Control: a plain fixed box (no transformed ancestor) never extends
    // a scroller's range.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"s\"><div id=\"t\"><div id=\"f\">x</div></div></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } \
         #s { overflow-y: auto; width: 200px; height: 100px; } \
         #t { height: 50px; } \
         #f { position: fixed; top: 300px; left: 0; width: 50px; height: 50px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let s = doc.get_element_by_id("s").expect("s");
    let c = scroll_containers(&root)
        .into_iter()
        .find(|c| c.node == Some(s))
        .expect("s container");
    assert_eq!(c.max, (0.0, 0.0));
}

#[test]
fn scrollable_extent_fixed_under_scroller_itself_does_not_extend() {
    // A fixed box whose containing block is the scroller itself is
    // anchored to the scrollport frame, which does not move when the
    // scroller scrolls: its overflow can never be revealed by scrolling,
    // so it must not extend the range (CSS Overflow 3: scrollable
    // overflow is content scrolling can bring into view). The
    // scroller's own transform makes it the fixed box's containing
    // block, but that is not a transformed ancestor *inside* the
    // scroller's content.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"s\"><div id=\"f\">x</div></div></body></html>".as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } \
         #s { overflow: auto; width: 100px; height: 100px; transform: translateZ(0); } \
         #f { position: fixed; left: 500px; top: 0; width: 50px; height: 50px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let s = doc.get_element_by_id("s").expect("s");
    let c = scroll_containers(&root)
        .into_iter()
        .find(|c| c.node == Some(s))
        .expect("s container");
    // `#f` reaches x=550, 450px past the 100px scrollport, but it is
    // pinned to the frame: `max` must stay zero.
    assert_eq!(c.max, (0.0, 0.0));
}

#[test]
fn scrollable_extent_skips_split_inline_wrapper() {
    // WPT `scrollable-overflow-relative-positioned-block-in-inline`:
    // a relatively positioned block nested in an inline element
    // contributes the same scrollable overflow as the same block as a
    // direct sibling: the split-inline wrapper's unshifted flow
    // footprint (bottom ~704) must not fold in — only the shifted
    // content (bottom ~404) counts. Asserted through the production
    // consumer `scroll_containers_with_offset`.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div class=\"container\" id=\"nested\"><img style=\"width:10px;height:200px\"><span><div class=\"shifted\"></div></span></div><div class=\"container\" id=\"sibling\"><img style=\"width:10px;height:200px\"><div class=\"shifted\"></div></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } \
         .container { width: 10px; height: 10px; overflow: auto; } \
         .shifted { width: 10px; height: 500px; position: relative; bottom: 300px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let nested = doc.get_element_by_id("nested").expect("nested");
    let sibling = doc.get_element_by_id("sibling").expect("sibling");
    let containers = scroll_containers_with_offset(&root, (0.0, 0.0));
    let n = containers
        .iter()
        .find(|c| c.node == Some(nested))
        .expect("nested container");
    let s = containers
        .iter()
        .find(|c| c.node == Some(sibling))
        .expect("sibling container");
    // The wrapper's unshifted footprint would report a far edge of
    // ~704 (max ~694) and a zero origin; the shifted content's far
    // edge is ~404 and its top is shifted above the scrollport.
    assert_eq!(n.max, s.max, "nested and sibling extents must agree");
    assert_eq!(
        n.scroll_origin, s.scroll_origin,
        "nested and sibling origins must agree"
    );
    assert!(approx(n.max.1, 393.84), "nested max {:?}", n.max);
    assert!(
        approx(n.scroll_origin.1, -96.16),
        "nested origin {:?}",
        n.scroll_origin
    );
    assert_eq!(n.max.0, 0.0);
}

#[test]
fn scrollable_extent_split_inline_wrapper_padding_spills() {
    // C10: the split-inline skip used to drop the wrapper's
    // contribution entirely, but the wrapper lays out with its own
    // padding/border: a `padding: 40px` span around a 10px block
    // paints a ~90px border box, and the bottom 40px of padding was
    // unreachable by scrolling when only the child's 10px counted.
    // The wrapper's edges on sides with a border/padding inset now
    // contribute; the unpadded case above is unchanged.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div class=\"container\" id=\"s\"><span class=\"pad\"><div class=\"kid\"></div></span></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } \
         .container { width: 100px; height: 50px; overflow: auto; } \
         .pad { padding: 40px; } .kid { height: 10px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let s = doc.get_element_by_id("s").expect("container");
    let containers = scroll_containers_with_offset(&root, (0.0, 0.0));
    let c = containers
        .iter()
        .find(|c| c.node == Some(s))
        .expect("container");
    // Wrapper bottom = 40 (top pad) + 10 (kid) + 40 (bottom pad) = 90
    // against a 50px scrollport: max.y = 40. The child alone would
    // report 0 (its 40..50 span fits the scrollport exactly).
    assert!(approx(c.max.1, 40.0), "padded max {:?}", c.max);
    assert_eq!(c.max.0, 0.0);
    assert_eq!(c.scroll_origin, (0.0, 0.0));

    // Unpadded control: same shape, no padding — the wrapper
    // contributes nothing and the 10px child fits, as before.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div class=\"container\" id=\"s\"><span><div class=\"kid\"></div></span></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } \
         .container { width: 100px; height: 50px; overflow: auto; } \
         .kid { height: 10px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let s = doc.get_element_by_id("s").expect("container");
    let containers = scroll_containers_with_offset(&root, (0.0, 0.0));
    let c = containers
        .iter()
        .find(|c| c.node == Some(s))
        .expect("container");
    assert_eq!(c.max, (0.0, 0.0), "unpadded control max {:?}", c.max);
}

#[test]
fn scrollable_extent_split_inline_wrapper_spill_transformed() {
    // C10, transformed branch: the spill edges are contributed after
    // the wrapper's accumulated transform, like every other box. The
    // wrapper (padding 40px around a 10px block, border-box bottom
    // at 90) is translated 20px down: its bottom padding edge lands
    // at 110 against a 50px scrollport, while the child alone reaches
    // only 40 + 10 + 20 = 70.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div class=\"container\" id=\"s\"><span class=\"pad\"><div class=\"kid\"></div></span></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } \
         .container { width: 100px; height: 50px; overflow: auto; } \
         .pad { padding: 40px; transform: translate(0, 20px); } .kid { height: 10px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let s = doc.get_element_by_id("s").expect("container");
    let containers = scroll_containers_with_offset(&root, (0.0, 0.0));
    let c = containers
        .iter()
        .find(|c| c.node == Some(s))
        .expect("container");
    assert!(approx(c.max.1, 60.0), "transformed spill max {:?}", c.max);
}

/// `overflow_axes` on the real cascade for `#t`, for the F9 conformance
/// cases.
fn axes_for(css: &str) -> ((bool, bool), (bool, bool)) {
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"t\">x</div></body></html>".as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(css);
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let t = doc.get_element_by_id("t").expect("t");
    let b = find_box(&root, t).expect("box");
    overflow_axes(b.style.as_ref().expect("style"))
}

#[test]
fn overflow_axes_visible_computes_to_auto_against_non_visible() {
    // CSS Overflow 3 §2: when one axis computes to `visible` and the
    // other does not, the `visible` axis computes to `auto`
    // (user-scrollable). Longhand-vs-longhand, both orders.
    assert_eq!(
        axes_for("#t { overflow-x: visible; overflow-y: scroll; }"),
        ((true, true), (true, true))
    );
    assert_eq!(
        axes_for("#t { overflow-y: scroll; overflow-x: visible; }"),
        ((true, true), (true, true))
    );
    // A non-`visible` longhand over a `visible` shorthand.
    assert_eq!(
        axes_for("#t { overflow: visible; overflow-x: hidden; }"),
        ((true, true), (false, true))
    );
    // A lone non-`visible` longhand pairs the other axis to `auto`.
    assert_eq!(
        axes_for("#t { overflow-x: scroll; }"),
        ((true, true), (true, true))
    );
    // `overflow: hidden` alone stays non-user-scrollable on both axes:
    // `hidden` never consumes a user scroll.
    assert_eq!(
        axes_for("#t { overflow: hidden; }"),
        ((true, true), (false, false))
    );
    // The cascade expands the `overflow` shorthand into longhands at
    // declaration time, so an explicit `visible` longhand over a
    // non-`visible` shorthand is observable: per CSS Overflow 3 §2 the
    // `visible` axis computes to `auto` (user-scrollable) while the
    // other axis keeps the shorthand value.
    assert_eq!(
        axes_for("#t { overflow: hidden; overflow-x: visible; }"),
        ((true, true), (true, false))
    );
    // Two-value shorthand: first value is `overflow-x`, second is
    // `overflow-y`; the `visible` axis pairs to `auto`.
    assert_eq!(
        axes_for("#t { overflow: hidden visible; }"),
        ((true, true), (false, true))
    );
    // A later shorthand overrides an earlier longhand.
    assert_eq!(
        axes_for("#t { overflow-x: visible; overflow: hidden; }"),
        ((true, true), (false, false))
    );
}

#[test]
fn scroll_containers_deep_tree_does_not_overflow_stack() {
    // Candidate collection used to recurse: a hostile deep DOM (the HTML
    // parser has no nesting cap) overflowed the stack. The walk is
    // iterative now, like `scrollable_extent`; mirrors
    // `very_deep_dom_does_not_panic`.
    //
    // The tree is deliberately `mem::forget` at the end: dropping a
    // 20k-deep `LayoutBox` recurses, which is the crate's documented
    // API-mandated gap (lib.rs "Stack safety"), not what this test
    // guards. The pre-fix recursive `collect` overflows the 2MiB test
    // stack at this depth; the iterative walk does not.
    let depth = 20_000;
    let mut html = String::from("<div id=\"outer\">");
    for _ in 0..depth {
        html.push_str("<div>");
    }
    html.push('x');
    for _ in 0..depth {
        html.push_str("</div>");
    }
    html.push_str("</div>");
    let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
    let ss = bowser_css::parse_stylesheet("#outer { overflow: auto; height: 100px; }");
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let outer = doc.get_element_by_id("outer").expect("outer");
    let containers = scroll_containers(&root);
    assert_eq!(containers.len(), 1, "only #outer is a container");
    assert_eq!(containers[0].node, Some(outer));
    std::mem::forget(root);
}

/// Collect `(depth, marker)` for every outside marker, in tree order,
/// joining markers to their principal boxes by DOM node.
fn markers_in(
    l: &LayoutBox,
    markers: &[MarkerBox],
    depth: usize,
    out: &mut Vec<(usize, MarkerBox)>,
) {
    if let Some(nid) = l.node
        && let Some(m) = markers.iter().find(|m| m.node == Some(nid))
    {
        out.push((depth, m.clone()));
    }
    for c in &l.children {
        markers_in(c, markers, depth + 1, out);
    }
}

fn all_markers(l: &LayoutBox, markers: &[MarkerBox]) -> Vec<(usize, MarkerBox)> {
    let mut out = Vec::new();
    markers_in(l, markers, 0, &mut out);
    out
}

/// The outside marker owned by `li`'s list item, if it has one.
fn marker_for<'a>(markers: &'a [MarkerBox], li: &LayoutBox) -> Option<&'a MarkerBox> {
    li.node
        .and_then(|nid| markers.iter().find(|m| m.node == Some(nid)))
}

#[test]
fn ul_li_renders_disc_marker() {
    let (l, markers) = layout_html_markers("<ul><li>item</li></ul>", "");
    let li = &body(&l).children[0].children[0];
    assert_eq!(li.box_type, BoxType::Block);
    let m = marker_for(&markers, li).expect("list item has a marker");
    assert_eq!(m.text, "•");
    // The marker hangs in the inline-start margin area: left of the
    // principal box (ul's 40px UA padding makes room).
    assert!(m.rect.x < li.rect.x, "marker left of li: {:?}", m.rect);
    assert!(m.rect.w > 0.0 && m.rect.h > 0.0);
    // Vertically aligned with the item's first line, not the page top.
    assert!(m.rect.y >= li.rect.y);
    assert!(m.rect.y < li.rect.y + li.rect.h);
    // No `::marker` rule: the marker inherits the list item's style.
    assert!(m.style.is_none());
    assert!(approx(m.font_size, 16.0));
}

#[test]
fn list_style_type_none_renders_no_marker() {
    let (l, markers) =
        layout_html_markers("<ul><li>item</li></ul>", "li { list-style-type: none; }");
    let li = &body(&l).children[0].children[0];
    assert!(
        marker_for(&markers, li).is_none(),
        "no marker for list-style-type: none"
    );
}

#[test]
fn list_style_type_keyword_is_ascii_case_insensitive() {
    let (l, markers) = layout_html_markers(
        "<ul><li>item</li></ul>",
        "li { list-style-type: UPPER-ROMAN; }",
    );
    let li = &body(&l).children[0].children[0];
    let m = marker_for(&markers, li).expect("list item has a marker");
    assert_eq!(m.text, "I.");
}

#[test]
fn marker_rule_colors_marker() {
    let (l, markers) = layout_html_markers(
        "<ul><li>item</li></ul>",
        "li::marker { color: red; font-size: 20px; }",
    );
    let li = &body(&l).children[0].children[0];
    let m = marker_for(&markers, li).expect("list item has a marker");
    let style = m.style.as_ref().expect("::marker rule applies");
    assert_eq!(
        style.get("color"),
        Some(&bowser_css::CssValue::Color(bowser_css::Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255
        })),
        "::marker color applies to the marker"
    );
    assert!(approx(m.font_size, 20.0));
}

#[test]
fn marker_inherits_li_color_without_rule() {
    // Without a `::marker` rule the marker keeps the list item's own
    // color via inheritance (style stays `None`; paint falls back to
    // the originating element's style).
    let (l, markers) = layout_html_markers("<ul><li>item</li></ul>", "li { color: blue; }");
    let li = &body(&l).children[0].children[0];
    let m = marker_for(&markers, li).expect("list item has a marker");
    assert!(m.style.is_none());
    assert_eq!(
        li.style.as_ref().and_then(|s| s.get("color")),
        Some(&bowser_css::CssValue::Color(bowser_css::Color {
            r: 0,
            g: 0,
            b: 255,
            a: 255
        }))
    );
}

#[test]
fn nested_ul_indents_markers_further() {
    let (l, markers) = layout_html_markers("<ul><li>outer<ul><li>inner</li></ul></li></ul>", "");
    let markers = all_markers(&l, &markers);
    assert_eq!(markers.len(), 2, "one marker per list item");
    let (outer_depth, outer) = &markers[0];
    let (inner_depth, inner) = &markers[1];
    assert!(inner_depth > outer_depth, "inner item nests deeper");
    assert!(
        inner.rect.x > outer.rect.x,
        "nested marker indents further: outer {:?} inner {:?}",
        outer.rect,
        inner.rect
    );
}

#[test]
fn ol_decimal_counting_increments() {
    let (l, markers) = layout_html_markers("<ol><li>a</li><li>b</li><li>c</li></ol>", "");
    let markers = all_markers(&l, &markers);
    let texts: Vec<&str> = markers.iter().map(|(_, m)| m.text.as_str()).collect();
    assert_eq!(texts, vec!["1.", "2.", "3."]);
}

#[test]
fn ol_start_and_li_value_shift_counting() {
    let (l, markers) = layout_html_markers(
        "<ol start=\"5\"><li>a</li><li value=\"9\">b</li><li>c</li></ol>",
        "",
    );
    let markers = all_markers(&l, &markers);
    let texts: Vec<&str> = markers.iter().map(|(_, m)| m.text.as_str()).collect();
    assert_eq!(texts, vec!["5.", "9.", "10."]);
}

#[test]
fn inside_marker_is_first_inline_box() {
    let (l, markers) = layout_html_markers(
        "<ul><li>item</li></ul>",
        "li { list-style-position: inside; }",
    );
    let li = &body(&l).children[0].children[0];
    // Inside markers flow with the content: no outside marker box.
    assert!(marker_for(&markers, li).is_none());
    let first = &li.children[0];
    assert_eq!(first.box_type, BoxType::Inline);
    let text = &first.children[0];
    assert_eq!(text.box_type, BoxType::Text);
    assert_eq!(text.text.as_deref(), Some("• "));
}

#[test]
fn counter_styles_render() {
    let (l, markers) = layout_html_markers(
        "<ol><li>a</li><li>b</li></ol>",
        "li { list-style-type: lower-roman; }",
    );
    let markers = all_markers(&l, &markers);
    let texts: Vec<&str> = markers.iter().map(|(_, m)| m.text.as_str()).collect();
    assert_eq!(texts, vec!["i.", "ii."]);
    let (l, markers) =
        layout_html_markers("<ul><li>a</li></ul>", "ul { list-style-type: square; }");
    let markers = all_markers(&l, &markers);
    assert_eq!(markers[0].1.text, "▪");
    // `list-style-type` inherits from the list to its items.
    let (l, markers) =
        layout_html_markers("<ul><li>a</li></ul>", "ul { list-style-type: circle; }");
    let markers = all_markers(&l, &markers);
    assert_eq!(markers[0].1.text, "○");
}

// ---------------------------------------------------------------------------
// v1 multi-column layout (column-count/width/gap)
// ---------------------------------------------------------------------------

#[test]
fn multicol_two_columns_with_gap() {
    let l = layout_html(
        r#"<div class="a"><div></div><div></div><div></div><div></div></div>"#,
        ".a { width: 400px; column-count: 2; column-gap: 20px; } .a div { height: 50px; }",
    );
    let a = &body(&l).children[0];
    // col_w = (400 - 20) / 2 = 190; four 50px children stack 200px tall,
    // so the balance height is 100px: the first two stay in column 1,
    // the last two move whole to column 2.
    let (b, c, d, e) = (
        &a.children[0],
        &a.children[1],
        &a.children[2],
        &a.children[3],
    );
    assert!(approx(b.rect.w, 190.0), "child width {b:?}");
    assert!(approx(b.rect.x, a.rect.x));
    assert!(approx(b.rect.y, a.rect.y));
    assert!(approx(c.rect.x, a.rect.x));
    assert!(approx(c.rect.y, a.rect.y + 50.0));
    assert!(approx(d.rect.x, a.rect.x + 210.0));
    assert!(approx(d.rect.y, a.rect.y));
    assert!(approx(e.rect.x, a.rect.x + 210.0));
    assert!(approx(e.rect.y, a.rect.y + 50.0));
    assert!(approx(a.rect.h, 100.0), "container height {a:?}");
}

#[test]
fn multicol_column_width_derives_count() {
    let l = layout_html(
        r#"<div class="a"><div></div><div></div></div>"#,
        ".a { width: 400px; column-width: 150px; } .a div { height: 50px; }",
    );
    let a = &body(&l).children[0];
    // gap = 1em = 16px; N = floor((400 + 16) / (150 + 16)) = 2;
    // col_w = (400 - 16) / 2 = 192. Two 50px children: balance height 50,
    // so the second child moves whole to column 2.
    let (b, c) = (&a.children[0], &a.children[1]);
    assert!(approx(b.rect.w, 192.0), "column width {b:?}");
    assert!(approx(b.rect.x, a.rect.x));
    assert!(approx(c.rect.x, a.rect.x + 208.0), "second child x {c:?}");
    assert!(approx(c.rect.y, a.rect.y));
    assert!(approx(a.rect.h, 50.0));
}

#[test]
fn multicol_count_one_is_block() {
    let l = layout_html(
        r#"<div class="a"><div></div><div></div></div>"#,
        ".a { width: 400px; column-count: 1; } .a div { height: 50px; }",
    );
    let a = &body(&l).children[0];
    let (b, c) = (&a.children[0], &a.children[1]);
    // column-count: 1 never establishes a multicol context: plain block
    // geometry, full content width, natural stacking.
    assert!(approx(b.rect.w, 400.0));
    assert!(approx(b.rect.x, a.rect.x));
    assert!(approx(b.rect.y, a.rect.y));
    assert!(approx(c.rect.x, a.rect.x));
    assert!(approx(c.rect.y, a.rect.y + 50.0));
    assert!(approx(a.rect.h, 100.0));
}

#[test]
fn multicol_break_inside_avoid_keeps_child_whole() {
    let l = layout_html(
        r#"<div class="a"><div></div><div class="c"></div><div></div></div>"#,
        ".a { width: 400px; column-count: 2; column-gap: 20px; } .a div { height: 60px; } .c { break-inside: avoid; }",
    );
    let a = &body(&l).children[0];
    let c = &a.children[1];
    // col_w = 190; total 180, balance height 90. The avoid child does not
    // fit the remaining 30px of column 1, so it moves whole to column 2
    // rather than fragmenting: one box, full 60px height.
    assert!(approx(c.rect.h, 60.0), "avoid child kept whole {c:?}");
    assert!(approx(c.rect.x, a.rect.x + 210.0));
    assert!(approx(c.rect.y, a.rect.y));
    // No spill: the third child stacks below it in column 2.
    let d = &a.children[2];
    assert!(approx(d.rect.x, a.rect.x + 210.0));
    assert!(approx(d.rect.y, a.rect.y + 60.0));
}

#[test]
fn multicol_nested_elements_flow_across_columns() {
    let l = layout_html(
        r#"<div class="a"><div class="b"><div></div><div></div></div><div class="d"></div></div>"#,
        ".a { width: 400px; column-count: 2; column-gap: 20px; } .b div { height: 40px; } .d { height: 40px; }",
    );
    let a = &body(&l).children[0];
    let b = &a.children[0];
    let d = &a.children[1];
    // .b is 80px tall, .d 40px; total 120, balance height 80: .b fills
    // column 1, .d moves whole to column 2. The nested grandchildren keep
    // their internal flow, laid out at the column width.
    assert!(approx(b.rect.w, 190.0));
    assert!(approx(b.rect.x, a.rect.x));
    let (c1, c2) = (&b.children[0], &b.children[1]);
    assert!(approx(c1.rect.w, 190.0));
    assert!(approx(c2.rect.y, c1.rect.y + 40.0));
    assert!(approx(d.rect.x, a.rect.x + 210.0), "nested sibling x {d:?}");
    assert!(approx(d.rect.y, a.rect.y));
    assert!(approx(a.rect.h, 80.0));
}

#[test]
fn multicol_inline_content_splits_lines_across_columns() {
    // 20 words of "aaa " at 16px (9.6px/char, FixedFontMetrics): 5 words
    // per 192px column -> 4 lines, 19.2px each, 76.8px total. Balance
    // height 38.4: lines 1-2 stay in column 1, lines 3-4 move to column 2.
    let words = "aaa ".repeat(20);
    let html = format!(r#"<div class="a">{words}</div>"#);
    let l = layout_html(&html, ".a { width: 400px; column-count: 2; }");
    let a = &body(&l).children[0];
    let text = &a.children[0];
    assert_eq!(text.box_type, BoxType::Text);
    // One fragment per word ("aaa" + space): 9 fragments per 5-word line.
    // Lines 1-2 sit in column 1, lines 3-4 in column 2.
    assert_eq!(text.fragments.len(), 36);
    assert!(approx(text.fragments[0].rect.x, a.rect.x));
    assert!(approx(text.fragments[9].rect.x, a.rect.x));
    // gap = 1em = 16px -> column 2 starts at a.x + 192 + 16.
    assert!(approx(text.fragments[18].rect.x, a.rect.x + 208.0));
    assert!(approx(text.fragments[27].rect.x, a.rect.x + 208.0));
    assert!(approx(text.fragments[18].rect.y, a.rect.y));
    assert!(approx(a.rect.h, 38.4), "container height {a:?}");
}

// ---------------------------------------------------------------------------
// inline-flex / inline-grid / inline-table as atomic inline-level boxes
// ---------------------------------------------------------------------------

/// The atomic inline-level box (the `InlineBlock` box whose
/// `StyleData::display` is `InlineFlex` / `InlineGrid` / `InlineTable`)
/// among `t`'s children.
fn atomic_inline(t: &LayoutBox) -> &LayoutBox {
    t.children
        .iter()
        .find(|b| b.box_type == BoxType::InlineBlock)
        .expect("atomic inline-level box")
}

/// Absolute baseline of the first fragment of the text box holding
/// `text` in `b`'s subtree.
fn text_baseline(b: &LayoutBox, text: &str) -> f32 {
    if b.box_type == BoxType::Text && b.text.as_deref() == Some(text) {
        return b.fragments.first().map(|f| f.baseline).unwrap_or(-1.0);
    }
    b.children
        .iter()
        .map(|k| text_baseline(k, text))
        .find(|v| *v >= 0.0)
        .unwrap_or(-1.0)
}

#[test]
fn inline_flex_lays_out_flex_items_in_shrink_to_fit_line_box() {
    let l = layout_html(
        r#"<div class="t">ab<span class="c"><div class="a"></div><div class="b"></div></span>cd</div>"#,
        ".c { display: inline-flex; } .a { width: 100px; height: 10px; } .b { width: 60px; height: 10px; }",
    );
    let t = &body(&l).children[0];
    let c = atomic_inline(t);
    // Shrink-to-fit: 100 + 60, not the 800px available width.
    assert!(approx(c.rect.w, 160.0), "c.w = {}", c.rect.w);
    // The flex algorithm ran: both items share one row.
    assert_eq!(c.children.len(), 2);
    assert!(approx(c.children[0].rect.x, c.rect.x));
    assert!(approx(c.children[0].rect.w, 100.0));
    assert!(approx(c.children[1].rect.x, c.rect.x + 100.0));
    assert!(approx(c.children[1].rect.w, 60.0));
    assert!(approx(c.children[1].rect.y, c.children[0].rect.y));
    // Atomic in the text line: placed right after "ab" (2 × 9.6px).
    assert!(approx(c.rect.x, t.rect.x + 19.2), "c.x = {}", c.rect.x);
}

#[test]
fn inline_grid_lays_out_grid_items_in_shrink_to_fit_line_box() {
    let l = layout_html(
        r#"<div class="t">ab<span class="c"><div class="a"></div><div class="b"></div></span>cd</div>"#,
        ".c { display: inline-grid; grid-template-columns: 100px 60px; } \
         .a { height: 10px; } .b { height: 10px; }",
    );
    let t = &body(&l).children[0];
    let c = atomic_inline(t);
    // Shrink-to-fit tracks: 100 + 60, not the available width.
    assert!(approx(c.rect.w, 160.0), "c.w = {}", c.rect.w);
    // The grid algorithm ran: items placed in their tracks.
    assert_eq!(c.children.len(), 2);
    assert!(approx(c.children[0].rect.x, c.rect.x));
    assert!(approx(c.children[0].rect.w, 100.0));
    assert!(approx(c.children[1].rect.x, c.rect.x + 100.0));
    assert!(approx(c.children[1].rect.w, 60.0));
    assert!(approx(c.children[1].rect.y, c.children[0].rect.y));
    assert!(approx(c.rect.x, t.rect.x + 19.2), "c.x = {}", c.rect.x);
}

#[test]
fn inline_table_lays_out_table_grid_in_shrink_to_fit_line_box() {
    let l = layout_html(
        r#"<div class="t">ab<span class="c"><span class="r"><span class="d">aa</span></span><span class="r"><span class="d">bb</span></span></span>cd</div>"#,
        ".c { display: inline-table; } .r { display: table-row; } .d { display: table-cell; }",
    );
    let t = &body(&l).children[0];
    let c = atomic_inline(t);
    // The table algorithm ran: two rows stack vertically (38.4px) in one
    // 19.2px column. Without table layout the cells would flow inline on a
    // single 19.2px line 38.4px wide.
    assert!(approx(c.rect.w, 19.2), "c.w = {}", c.rect.w);
    assert!(approx(c.rect.h, 38.4), "c.h = {}", c.rect.h);
    // Shrink-to-fit: the column content width, not the available width.
    assert!(c.rect.w < 800.0);
    assert!(approx(c.rect.x, t.rect.x + 19.2), "c.x = {}", c.rect.x);
}

#[test]
fn inline_flex_baseline_is_first_item_baseline() {
    // Block-level children become distinct flex items (adjacent inline
    // runs would merge into one anonymous item).
    let l = layout_html(
        r#"<div class="t">x<span class="c"><div class="a">Hi</div><div class="b">Yo</div></span></div>"#,
        ".c { display: inline-flex; flex-direction: column; }",
    );
    let t = &body(&l).children[0];
    let c = atomic_inline(t);
    // The surrounding "x" shares its baseline with the first item's
    // "Hi" (not with the container's bottom edge); "Yo" sits lower.
    assert!(approx(text_baseline(t, "x"), text_baseline(c, "Hi")));
    assert!(text_baseline(c, "Yo") > text_baseline(c, "Hi"));
}

#[test]
fn inline_grid_baseline_is_first_item_baseline() {
    let l = layout_html(
        r#"<div class="t">x<span class="c"><div class="a">Hi</div><div class="b">Yo</div></span></div>"#,
        ".c { display: inline-grid; grid-template-columns: 100px; }",
    );
    let t = &body(&l).children[0];
    let c = atomic_inline(t);
    assert!(approx(text_baseline(t, "x"), text_baseline(c, "Hi")));
    assert!(text_baseline(c, "Yo") > text_baseline(c, "Hi"));
}

#[test]
fn inline_table_baseline_is_first_row_baseline() {
    let l = layout_html(
        r#"<div class="t">x<span class="c"><span class="r"><span class="d">Hi</span></span><span class="r"><span class="d">Yo</span></span></span></div>"#,
        ".c { display: inline-table; } .r { display: table-row; } .d { display: table-cell; }",
    );
    let t = &body(&l).children[0];
    let c = atomic_inline(t);
    // CSS 2.1 §10.8.1: the baseline of an inline-table is the baseline
    // of its first row.
    assert!(approx(text_baseline(t, "x"), text_baseline(c, "Hi")));
    assert!(text_baseline(c, "Yo") > text_baseline(c, "Hi"));
}

#[test]
fn vertical_align_shifts_inline_level_container() {
    let html = r#"<div class="t">x<span class="c"><div class="a"></div></span></div>"#;
    let plain = layout_html(
        html,
        ".c { display: inline-grid; } .a { width: 50px; height: 10px; }",
    );
    let shifted = layout_html(
        html,
        ".c { display: inline-grid; vertical-align: 5px; } .a { width: 50px; height: 10px; }",
    );
    let cy = |l: &LayoutBox| atomic_inline(&body(l).children[0]).rect.y;
    // Positive lengths raise the box (CSS 2.1 §10.8.1).
    assert!(
        approx(cy(&shifted), cy(&plain) - 5.0),
        "plain y = {}, shifted y = {}",
        cy(&plain),
        cy(&shifted)
    );
}

#[test]
fn inline_level_box_breaks_line_with_surrounding_text() {
    let l = layout_html(
        r#"<div class="t">aaa <span class="c"><div class="a"></div><div class="b"></div></span> bbb</div>"#,
        ".t { width: 200px; } .c { display: inline-flex; } .a { width: 100px; height: 10px; } .b { width: 100px; height: 10px; }",
    );
    let t = &body(&l).children[0];
    let c = atomic_inline(t);
    // "aaa " is 38.4px; the 200px atomic box cannot fit beside it, so it
    // breaks onto the next line instead of overflowing it.
    assert!(
        c.rect.y > t.rect.y + 1.0,
        "c.y = {}, t.y = {}",
        c.rect.y,
        t.rect.y
    );
    assert!(approx(c.rect.x, t.rect.x), "c.x = {}", c.rect.x);
}

#[test]
fn aspect_ratio_derives_auto_height_from_width() {
    // 160 / (16/9) = 90.
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 160px; aspect-ratio: 16/9; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 90.0), "h = {}", a.rect.h);
}

#[test]
fn aspect_ratio_border_box_transfers_border_box() {
    // Under `border-box` the ratio maps the border box: 200 / 2 = 100
    // border height, minus 2x10px padding = 80 content height.
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 200px; box-sizing: border-box; padding: 10px; aspect-ratio: 2/1; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 100.0), "h = {}", a.rect.h);
}

#[test]
fn aspect_ratio_derives_auto_width_from_height() {
    // Float: auto width + definite height -> width = 100 * 2 = 200.
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { float: left; height: 100px; aspect-ratio: 2/1; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 200.0), "w = {}", a.rect.w);
}

#[test]
fn aspect_ratio_height_clamped_by_min_max() {
    // Ratio height 90, but min-height wins per spec.
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 160px; aspect-ratio: 16/9; min-height: 200px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 200.0), "h = {}", a.rect.h);
    // ...and max-height clamps from above.
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 160px; aspect-ratio: 16/9; max-height: 40px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 40.0), "h = {}", a.rect.h);
}

#[test]
fn aspect_ratio_ignored_when_both_sizes_definite() {
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 160px; height: 50px; aspect-ratio: 16/9; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 50.0), "h = {}", a.rect.h);
}

#[test]
fn aspect_ratio_auto_means_no_transfer() {
    let l = layout_html(
        r#"<div class="a"></div>"#,
        ".a { width: 160px; aspect-ratio: auto; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.h, 0.0), "h = {}", a.rect.h);
}

#[test]
fn aspect_ratio_height_is_definite_for_children() {
    // A ratio-derived height counts as definite: percentage heights of
    // children resolve against it (90 * 50% = 45).
    let l = layout_html(
        r#"<div class="a"><div class="b"></div></div>"#,
        ".a { width: 160px; aspect-ratio: 16/9; } .b { height: 50%; }",
    );
    let b = &body(&l).children[0].children[0];
    assert!(approx(b.rect.h, 45.0), "h = {}", b.rect.h);
}

#[test]
fn aspect_ratio_specified_ratio_wins_over_intrinsic() {
    // css-sizing-4 §3.1: a specified `<ratio>` wins over the intrinsic
    // ratio on replaced elements; only `auto <ratio>` prefers it.
    let (doc, _) = bowser_dom::Document::parse_html(r#"<img class="i">"#.as_bytes());
    let img = body_child_tag(&doc, "img");
    let mut map = HashMap::new();
    map.insert(img, IntrinsicDims::from_size(100.0, 100.0));
    // 1:1 intrinsic, but `aspect-ratio: 2/1` transfers 100 -> 50.
    let l = layout_doc_with_intrinsics(
        &doc,
        ".i { display: block; aspect-ratio: 2/1; }",
        &FixedFontMetrics,
        &map,
    );
    let b = &body(&l).children[0];
    assert!(approx(b.rect.w, 100.0), "w = {}", b.rect.w);
    assert!(approx(b.rect.h, 50.0), "h = {}", b.rect.h);
    // `auto 2/1` prefers the intrinsic ratio: height stays 100.
    let l = layout_doc_with_intrinsics(
        &doc,
        ".i { display: block; aspect-ratio: auto 2/1; }",
        &FixedFontMetrics,
        &map,
    );
    let b = &body(&l).children[0];
    assert!(approx(b.rect.h, 100.0), "h = {}", b.rect.h);
}

#[test]
fn scroll_containers_rtl_origin_anchors_min_edge() {
    // `direction: rtl` lays the 400px child out right-aligned, so it
    // overflows the 200px scrollport to the LEFT. The minimum x offset
    // is still set by that minimum content edge: offsets shift content
    // by `-offset` uniformly (see `ScrollTree::bake_paint_tree`), so
    // negative offsets reveal leftward overflow in both directions.
    // `scroll_origin` is (-200, 0) — not (0, 0) — and `max` is (0, 0):
    // clamping into `[origin, max]` = [-200, 0] reaches every content
    // pixel, with offset 0 showing the inline-start (right) edge per
    // CSS Overflow 3 §3.2.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"inner\"><div id=\"wide\">x</div></div></body></html>".as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#inner { direction: rtl; overflow-x: scroll; width: 200px; height: 100px; } #wide { width: 400px; height: 50px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let inner = doc.get_element_by_id("inner").expect("inner");
    let c = scroll_containers(&root)
        .into_iter()
        .find(|c| c.node == Some(inner))
        .expect("inner container");
    assert_eq!(c.scroll_origin, (-200.0, 0.0));
    assert_eq!(c.max, (0.0, 0.0));
}

#[test]
fn scroll_containers_nested_rtl_origin_max_and_clamp() {
    // Nested horizontal RTL scrolling: outer (300px, `overflow-x:
    // scroll`) > inner (200px, `overflow-x: scroll`) > 400px of content.
    // The inner's leftward overflow is clipped by its own scrollport, so
    // the outer's range stays [(0,0), (0,0)]; the inner reports
    // `scroll_origin` (-200, 0), `max` (0, 0). The clamp sweep mirrors
    // the app's degenerate-safe `[origin, max]` clamp: every requested
    // offset lands in range, and the range extremes reveal the extreme
    // content pixels, so nested RTL scrolling reaches the whole content.
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"outer\"><div id=\"inner\"><div id=\"wide\">x</div></div></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#outer { direction: rtl; overflow-x: scroll; width: 300px; height: 200px; } \
         #inner { direction: rtl; overflow-x: scroll; width: 200px; height: 100px; } \
         #wide { width: 400px; height: 50px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let outer = doc.get_element_by_id("outer").expect("outer");
    let inner = doc.get_element_by_id("inner").expect("inner");
    let wide = doc.get_element_by_id("wide").expect("wide");
    let containers = scroll_containers(&root);
    let names: Vec<_> = containers.iter().map(|c| c.node).collect();
    assert_eq!(names, vec![Some(outer), Some(inner)], "preorder");
    let (o, i) = (&containers[0], &containers[1]);
    assert_eq!(o.scroll_origin, (0.0, 0.0), "outer: no scrollable overflow");
    assert_eq!(
        o.max,
        (0.0, 0.0),
        "outer: inner's overflow is clipped by inner"
    );
    assert_eq!(i.scroll_origin, (-200.0, 0.0), "inner: min content edge");
    assert_eq!(i.max, (0.0, 0.0), "inner: nothing past the right edge");
    assert!(o.depth < i.depth);
    // Degenerate-safe clamp, mirroring `ScrollState::set_offset`: the
    // minimum is `origin.min(max)` so a crossed publish cannot panic.
    let clamp = |v: f32, origin: f32, max: f32| v.clamp(origin.min(max), max);
    for requested in [-300.0, -200.0, -199.5, -1.0, 0.0, 42.0, 100.0] {
        let got = clamp(requested, i.scroll_origin.0, i.max.0);
        assert!(
            (-200.0..=0.0).contains(&got),
            "requested {requested} clamped to {got}, outside [-200, 0]"
        );
    }
    // Range extremes reveal the extreme content pixels: at `origin` the
    // scrollport's left edge meets the content's left edge; at `max`
    // (offset 0) it shows the inline-start (right) edge. Offsets shift
    // content by `-offset`, so the visible content span is
    // `[scrollport.x + offset, scrollport.x + scrollport.w + offset]`.
    let content = find_box(&root, wide).expect("wide box").rect;
    let visible_at = |offset: f32| {
        (
            i.scrollport.x + offset,
            i.scrollport.x + i.scrollport.w + offset,
        )
    };
    let (lo0, lo1) = visible_at(i.scroll_origin.0);
    assert_eq!((lo0, lo1), (content.x, content.x + 200.0));
    let (hi0, hi1) = visible_at(i.max.0);
    assert_eq!((hi0, hi1), (content.x + 200.0, content.x + content.w));
}

#[test]
fn scroll_containers_rtl_inherited_direction() {
    // The container's *computed* `direction` drives RTL-ness: #inner sets
    // no `direction` itself but inherits `rtl` from #outer, so its
    // content lays out right-aligned and its origin is the negative
    // leftward overflow, exactly like an explicitly-rtl container.
    let (doc, _) = bowser_dom::Document::parse_html(
        "<html><body><div id=\"outer\"><div id=\"inner\"><div id=\"wide\">x</div></div></div></body></html>"
            .as_bytes(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#outer { direction: rtl; } \
         #inner { overflow-x: scroll; width: 200px; height: 100px; } \
         #wide { width: 400px; height: 50px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let inner = doc.get_element_by_id("inner").expect("inner");
    let inbox = find_box(&root, inner).expect("inner box");
    assert_eq!(
        inbox.style.as_ref().and_then(|s| s.get_str("direction")),
        Some("rtl"),
        "computed direction inherits through the cascade"
    );
    let c = scroll_containers(&root)
        .into_iter()
        .find(|c| c.node == Some(inner))
        .expect("inner container");
    assert_eq!(c.scroll_origin, (-200.0, 0.0));
    assert_eq!(c.max, (0.0, 0.0));
}

#[test]
fn scroll_containers_rtl_dual_overflow_origin_not_neg_max() {
    // Asymmetric dual overflow under `direction: rtl`: the abspos child
    // sticks 60px past the scrollport's right edge while the 400px block
    // child overflows 200px to the left. The origin stays anchored at
    // the minimum content edge (-200), *not* at `-max` (-60): `-max`
    // would strand the [-200, -60) subrange, whose offsets reveal real
    // content, while `max` still measures the right edge (60).
    let (doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"inner\"><div id=\"wide\">x</div><div id=\"r\">y</div></div></body></html>"
            .as_slice(),
    );
    let ss = bowser_css::parse_stylesheet(
        "#inner { direction: rtl; overflow-x: scroll; width: 200px; height: 100px; position: relative; } \
         #wide { width: 400px; height: 50px; } \
         #r { position: absolute; right: -60px; top: 0; width: 100px; height: 20px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let inner = doc.get_element_by_id("inner").expect("inner");
    let c = scroll_containers(&root)
        .into_iter()
        .find(|c| c.node == Some(inner))
        .expect("inner container");
    assert_eq!(c.scroll_origin, (-200.0, 0.0), "min content edge, not -max");
    assert_eq!(
        c.max,
        (60.0, 0.0),
        "right-edge overflow past the scrollport"
    );
}

#[test]
fn auto_width_table_min_includes_border_spacing() {
    // CSS 2.1 §17.5.2.2: the table MIN includes inter-column spacing.
    // Two 18px-minimum columns with `border-spacing: 10px` have a MIN of
    // 18 + 18 + 3*10 = 66, not 36. The 30px containing block binds below
    // that floor, so the table keeps the 66px floor instead of shrinking
    // to the bare 36px column sum (which would starve the spacing).
    let l = layout_html(
        r#"<div class="w"><table><tr><td>aaa</td><td>bbb</td></tr></table></div>"#,
        ".w { width: 30px; } table { border-spacing: 10px; } \
         td { font-size: 10px; white-space: nowrap; }",
    );
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    assert!(approx(table.rect.w, 66.0));
}

#[test]
fn auto_width_border_box_table_clamps_in_border_box_terms() {
    // `box-sizing: border-box` shrink-to-fit resolves in border-box terms:
    // the used border-box width is max(MIN_c + E, min(MAX_c + E, avail)).
    // One column, `border-spacing: 10px`: MIN_c = 18 + 20 = 38,
    // MAX_c = 54 + 20 = 74; with 5px table padding (E = 10) and a 60px
    // containing block the border box is min(74 + 10, 60) = 60 — not the
    // 70px the content-box-terms computation would produce.
    let l = layout_html(
        r#"<div class="w"><table><tr><td>aaa bbb ccc</td></tr></table></div>"#,
        ".w { width: 60px; } table { box-sizing: border-box; padding: 5px; \
         border-spacing: 10px; } td { font-size: 10px; }",
    );
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    assert!(approx(table.rect.w, 60.0));
}

#[test]
fn inline_table_shrink_to_fit_respects_caption_min() {
    // CSS 2.1 §17.5.2.2 CAPMIN is not just for block-level tables: an
    // `inline-table` (an `InlineBlock` box taking the generic shrink
    // branch) with `width: auto` sizes from max(columns, caption), not
    // columns alone. 30 a's at 10px give a 180px caption minimum over a
    // 12px column.
    let l = layout_html(
        &format!(
            "<div class=\"a\"><table class=\"t\"><caption>{}</caption>\
             <tr><td>hi</td></tr></table></div>",
            "a".repeat(30)
        ),
        ".a { width: 400px; } table.t { display: inline-table; } \
         td, caption { font-size: 10px; }",
    );
    let t = &body(&l).children[0].children[0];
    assert_eq!(t.box_type, BoxType::InlineBlock);
    assert!(approx(t.rect.w, 180.0));
}

#[test]
fn shrink_to_fit_includes_nested_child_padding_border_margin() {
    // The float shrink-to-fits around its block child's *margin* box:
    // "hello" is 30px at 10px font; the kid adds 20px padding + 4px
    // border + 10px margins = 64px total.
    let l = layout_html(
        r#"<div class="f"><div class="kid">hello</div></div>"#,
        ".f { float: left; font-size: 10px; } \
         .kid { padding: 0 10px; border: 2px solid black; margin: 0 5px; }",
    );
    let f = &body(&l).children[0];
    assert!(approx(f.rect.w, 64.0));
    // The kid fits exactly: 30px content + 24px padding/border.
    let kid = &f.children[0];
    assert!(approx(kid.rect.x, 5.0));
    assert!(approx(kid.rect.w, 54.0));
}

#[test]
fn inline_block_shrink_to_fit_includes_padded_inline_child() {
    // Inline-level children sum into one run; the padded span's margin
    // box ("hi" = 12px + 16px padding) sets the inline-block width.
    let l = layout_html(
        r#"<div class="ib"><span class="p">hi</span></div>"#,
        ".ib { display: inline-block; font-size: 10px; } .p { padding: 0 8px; }",
    );
    let ib = &body(&l).children[0];
    assert_eq!(ib.box_type, BoxType::InlineBlock);
    assert!(approx(ib.rect.w, 28.0));
}

#[test]
fn table_auto_layout_max_content_includes_nested_padding() {
    // Max-content column sizing: the cell's preferred width includes the
    // inner div's padding ("hello" = 30px + 20px padding = 50px column).
    let l = layout_html(
        r#"<table class="t"><tr><td><div class="inner">hello</div></td></tr></table>"#,
        ".t { font-size: 10px; border-collapse: collapse; } .inner { padding: 0 10px; }",
    );
    let t = &body(&l).children[0];
    assert_eq!(t.box_type, BoxType::Table);
    assert!(approx(t.rect.w, 50.0));
}

#[test]
fn flex_item_automatic_minimum_includes_nested_padding() {
    // The automatic minimum size is the item's min-content width, which
    // includes the nested padded child's padding: 30px text + 20px
    // padding floors the 20px container's shrink at 50px.
    let l = layout_html(
        r#"<div class="flex"><div class="item"><div class="pad">hello</div></div></div>"#,
        ".flex { display: flex; width: 20px; font-size: 10px; } .pad { padding: 0 10px; }",
    );
    let item = &body(&l).children[0].children[0];
    assert!(approx(item.rect.w, 50.0));
}

// ---------------------------------------------------------------------------
// `establishes_fixed_cb` truth table.
// ---------------------------------------------------------------------------

/// Computed style of `<body>`'s first `<div>` child under a one-rule
/// author stylesheet, for testing style-query helpers end to end
/// (parse → cascade → helper).
fn computed_style_for_decl(decl: &str) -> bowser_css::ComputedStyle {
    let (doc, _) = bowser_dom::Document::parse_html(b"<div></div>");
    let node = body_child_tag(&doc, "div");
    let css = format!("div {{ {decl} }}");
    let ss = bowser_css::parse_stylesheet(&css);
    let ua = bowser_css::default_ua_stylesheet();
    bowser_css::compute_style(
        &doc,
        node,
        &[&ss],
        &ua,
        &bowser_css::Viewport { w: VP.w, h: VP.h },
    )
}

#[test]
fn establishes_fixed_cb_truth_table() {
    let check = |decl: &str| establishes_fixed_cb(&computed_style_for_decl(decl));
    // The pre-existing trigger still counts.
    assert!(check("transform: rotate(45deg)"));
    assert!(!check("transform: none"));
    // will-change: only transform/filter/perspective establish a CB
    // (CSS Position 3 §2.1); stacking-context-only hints
    // (opacity/mix-blend-mode/isolation) and `contain` do not.
    assert!(check("will-change: transform"));
    assert!(!check("will-change: opacity"));
    assert!(check("will-change: filter"));
    assert!(check("will-change: perspective"));
    assert!(check("will-change: top, transform"));
    assert!(!check("will-change: contain"));
    assert!(!check("will-change: mix-blend-mode"));
    assert!(!check("will-change: isolation"));
    // will-change: the initial value and non-triggers do not.
    assert!(!check("will-change: auto"));
    assert!(!check("will-change: top"));
    assert!(!check("will-change: scroll-position"));
    assert!(!check("color: red"));
    // filter.
    assert!(check("filter: blur(5px)"));
    assert!(check("filter: brightness(1)"));
    assert!(check("filter: drop-shadow(2px 2px 2px black)"));
    assert!(check("filter: url(noise.png)"));
    assert!(!check("filter: none"));
    // perspective.
    assert!(check("perspective: 500px"));
    assert!(check("perspective: 0"));
    assert!(!check("perspective: none"));
    // contain: paint/layout/strict/content establish a CB;
    // size/style alone do not.
    for v in [
        "paint",
        "layout",
        "strict",
        "content",
        "size paint",
        "layout style",
        "size layout style paint",
    ] {
        assert!(check(&format!("contain: {v}")), "contain: {v}");
    }
    for v in ["none", "size", "style", "size style"] {
        assert!(!check(&format!("contain: {v}")), "contain: {v}");
    }
}

#[test]
fn scrollbar_gutter_stable_included_in_intrinsic_width() {
    // A shrink-to-fit scroll container with `scrollbar-gutter: stable`
    // includes the gutter reservation in its intrinsic size (Chrome
    // behavior): the float measures content (100) + gutter (12).
    let l = layout_html(
        "<div class=\"c\"><div style=\"width: 100px; height: 10px;\"></div></div>",
        ".c { float: left; overflow-y: auto; height: 50px; scrollbar-gutter: stable; }",
    );
    let c = &body(&l).children[0];
    assert!(
        approx(c.rect.w, 100.0 + crate::SCROLLBAR_THICKNESS),
        "float width {} != 112",
        c.rect.w
    );
}

#[test]
fn scrollbar_gutter_both_edges_included_in_intrinsic_width() {
    // `stable both-edges` mirrors the strip to both sides: the
    // intrinsic size includes 2 * SCROLLBAR_THICKNESS, symmetric under
    // either direction.
    let l = layout_html(
        "<div class=\"c\"><div style=\"width: 100px; height: 10px;\"></div></div>",
        ".c { float: left; overflow-y: auto; height: 50px; scrollbar-gutter: stable both-edges; }",
    );
    let c = &body(&l).children[0];
    assert!(
        approx(c.rect.w, 100.0 + 2.0 * crate::SCROLLBAR_THICKNESS),
        "float width {} != 124",
        c.rect.w
    );
}

#[test]
fn scrollbar_gutter_max_content_clamped_inside_max_width() {
    // The gutter reservation folds *inside* the `min-width` /
    // `max-width` clamp (like min-content): with a dominating
    // `max-width`, the max-content intrinsic is the clamped total, not
    // the clamped content plus the reservation on top. The outer float
    // shrink-to-fits around the scroller, so its used width exposes the
    // scroller's intrinsic contribution: 100 content + 12 gutter clamped
    // to 105, not 112.
    let l = layout_html(
        "<div class=\"wrap\"><div class=\"c\"><div style=\"width: 100px; height: 10px;\"></div></div></div>",
        ".wrap { float: left; } .c { overflow-y: auto; height: 50px; scrollbar-gutter: stable; max-width: 105px; }",
    );
    let wrap = &body(&l).children[0];
    assert!(
        approx(wrap.rect.w, 105.0),
        "float width {} != 105",
        wrap.rect.w
    );
}

#[test]
fn scrollbar_gutter_auto_not_included_in_intrinsic_width() {
    // `overflow-y: auto` without `stable` reserves nothing at intrinsic
    // time (a shown scrollbar only narrows the content box once layout
    // knows content overflows).
    let l = layout_html(
        "<div class=\"c\"><div style=\"width: 100px; height: 10px;\"></div></div>",
        ".c { float: left; overflow-y: auto; height: 50px; }",
    );
    let c = &body(&l).children[0];
    assert!(approx(c.rect.w, 100.0), "float width {} != 100", c.rect.w);
}

#[test]
fn forced_scrollbar_included_in_intrinsic_width() {
    // `overflow-y: scroll` forces the classic scrollbar, which always
    // consumes its thickness from the content box — so the intrinsic
    // size includes it even without `scrollbar-gutter: stable`.
    let l = layout_html(
        "<div class=\"c\"><div style=\"width: 100px; height: 10px;\"></div></div>",
        ".c { float: left; overflow-y: scroll; height: 50px; }",
    );
    let c = &body(&l).children[0];
    assert!(
        approx(c.rect.w, 100.0 + crate::SCROLLBAR_THICKNESS),
        "float width {} != 112",
        c.rect.w
    );
}

#[test]
fn scrollbar_gutter_stable_included_in_min_content_width() {
    // `width: min-content` on a scroll container sizes to the
    // min-content size, which includes the gutter reservation.
    let l = layout_html(
        "<div class=\"c\"><div style=\"width: 100px; height: 10px;\"></div></div>",
        ".c { width: min-content; overflow-y: auto; height: 50px; scrollbar-gutter: stable; }",
    );
    let c = &body(&l).children[0];
    assert!(
        approx(c.rect.w, 100.0 + crate::SCROLLBAR_THICKNESS),
        "min-content width {} != 112",
        c.rect.w
    );
}

#[test]
fn scrollbar_gutter_definite_width_not_double_counted() {
    // A definite specified width already contains the gutter (it is
    // reserved from that exact width at layout time): the intrinsic
    // contribution adds nothing on top.
    let l = layout_html(
        "<div class=\"c\"><div style=\"width: 100px; height: 10px;\"></div></div>",
        ".c { float: left; width: 200px; overflow-y: auto; height: 50px; scrollbar-gutter: stable; }",
    );
    let c = &body(&l).children[0];
    assert!(approx(c.rect.w, 200.0), "float width {} != 200", c.rect.w);
}

#[test]
fn scrollbar_gutter_stable_included_in_grid_intrinsic_width() {
    // The grid intrinsic arm flows through the same per-node
    // finalization: a shrink-to-fit grid scroller includes the gutter.
    let l = layout_html(
        "<div class=\"c\"><div style=\"width: 100px; height: 10px;\"></div></div>",
        ".c { float: left; display: grid; overflow-y: auto; height: 50px; scrollbar-gutter: stable; }",
    );
    let c = &body(&l).children[0];
    assert!(
        approx(c.rect.w, 100.0 + crate::SCROLLBAR_THICKNESS),
        "grid float width {} != 112",
        c.rect.w
    );
}

#[test]
fn scrollbar_gutter_stable_included_in_flex_intrinsic_width() {
    // Same for the flex intrinsic arm.
    let l = layout_html(
        "<div class=\"c\"><div style=\"width: 100px; height: 10px;\"></div></div>",
        ".c { float: left; display: flex; overflow-y: auto; height: 50px; scrollbar-gutter: stable; }",
    );
    let c = &body(&l).children[0];
    assert!(
        approx(c.rect.w, 100.0 + crate::SCROLLBAR_THICKNESS),
        "flex float width {} != 112",
        c.rect.w
    );
}

#[test]
fn scrollbar_gutter_stable_narrows_children_width() {
    // `scrollbar-gutter: stable` on a scroll container reserves
    // SCROLLBAR_THICKNESS for the (possibly absent) scrollbar: children
    // lay out against the narrowed content width.
    let l = layout_html(
        "<div><div style=\"width: 100%; height: 10px;\">x</div></div>",
        "div { overflow-y: auto; width: 200px; height: 100px; scrollbar-gutter: stable; }",
    );
    let kid = &body(&l).children[0].children[0];
    // Child at 100% width should be 200 - 12 = 188px.
    assert!(
        (kid.rect.w - (200.0 - crate::SCROLLBAR_THICKNESS)).abs() < 0.01,
        "kid width {} != 188",
        kid.rect.w
    );
}

#[test]
fn scrollbar_gutter_on_root_propagates_to_viewport() {
    // CSS Overflow 4 §2.2: `scrollbar-gutter` on the root element is
    // propagated to the viewport, so the gutter is reserved against
    // the initial containing block: the root's own border box shrinks
    // by the gutter and its children fill the shrunk content box —
    // root, body and content all end up the same width (WPT
    // `scrollbar-gutter-propagation-004.html` / `-005.html`). Before
    // the fix the reservation only narrowed the root's children, so
    // body came out one gutter narrower than the root.
    let l = layout_html(
        "<div style=\"width: 100%; height: 200vh;\"></div>",
        ":root { scrollbar-gutter: stable; overflow: auto; } \
         body, html { margin: 0; padding: 0; border: none; }",
    );
    let root_w = l.rect.w;
    let body = body(&l);
    let content = &body.children[0];
    assert!(
        approx(root_w, VP.w - crate::SCROLLBAR_THICKNESS),
        "root width {root_w} != viewport - gutter"
    );
    assert!(
        approx(body.rect.w, root_w),
        "body width {} != root width {root_w}",
        body.rect.w
    );
    assert!(
        approx(content.rect.w, root_w),
        "content width {} != root width {root_w}",
        content.rect.w
    );
}

#[test]
fn scrollbar_gutter_auto_does_not_narrow() {
    // `scrollbar-gutter: auto` (the initial) reserves nothing.
    let l = layout_html(
        "<div><div style=\"width: 100%; height: 10px;\">x</div></div>",
        "div { overflow-y: auto; width: 200px; height: 100px; }",
    );
    let kid = &body(&l).children[0].children[0];
    assert!(
        (kid.rect.w - 200.0).abs() < 0.01,
        "kid width {} != 200",
        kid.rect.w
    );
}

#[test]
fn viewport_rerun_on_propagated_scroll() {
    // Propagated used `overflow-y: scroll` with document overflow: the
    // app paints the classic scrollbar, so the viewport re-run narrows
    // the initial containing block by its thickness (the first pass
    // only propagates the stable gutter, never the forced strip).
    let l = layout_html(
        "<div style=\"width: 100%; height: 200vh;\"></div>",
        "html { overflow-y: scroll; } body, html { margin: 0; padding: 0; border: none; }",
    );
    assert!(
        approx(l.rect.w, VP.w - crate::SCROLLBAR_THICKNESS),
        "root width {} != viewport - scrollbar",
        l.rect.w
    );
}

#[test]
fn viewport_rerun_on_negative_scroll_origin() {
    // Negative scroll origin (`min_y < 0`, `max_y == 0`): the app still
    // paints (`max > min`), so the re-run fires even without positive
    // overflow.
    let l = layout_html(
        "<div style=\"height: 10px; margin-top: -100px;\"></div>",
        "html { overflow-y: auto; } body, html { margin: 0; padding: 0; border: none; }",
    );
    assert!(
        approx(l.rect.w, VP.w - crate::SCROLLBAR_THICKNESS),
        "root width {} != viewport - scrollbar",
        l.rect.w
    );
}

#[test]
fn scrollbar_gutter_stable_rtl_shifts_children_left_edge() {
    // CSS Overflow 4: the `stable` strip reserves at the inline-end
    // edge — the left edge under `direction: rtl` — so children shift
    // right by the strip and narrow by it, instead of laying out from
    // the left edge with the strip silently taken off the right.
    let html = "<div id=\"c\"><div style=\"width: 100%; height: 10px;\">x</div></div>";
    let no_gutter = layout_html(
        html,
        "body { margin: 0; } #c { direction: rtl; overflow-y: auto; width: 200px; height: 100px; }",
    );
    let c0 = &body(&no_gutter).children[0];
    let kid0 = &c0.children[0];
    assert!(approx(kid0.rect.x - c0.rect.x, 0.0), "no-gutter kid x");
    assert!(
        approx(kid0.rect.w, 200.0),
        "no-gutter kid w {}",
        kid0.rect.w
    );

    let l = layout_html(
        html,
        "body { margin: 0; } #c { direction: rtl; overflow-y: auto; width: 200px; height: 100px; scrollbar-gutter: stable; }",
    );
    let c = &body(&l).children[0];
    let kid = &c.children[0];
    assert!(
        approx(kid.rect.x - c.rect.x, crate::SCROLLBAR_THICKNESS),
        "RTL stable kid x offset {} != 12",
        kid.rect.x - c.rect.x
    );
    assert!(
        approx(kid.rect.x - kid0.rect.x, crate::SCROLLBAR_THICKNESS),
        "RTL stable shift vs no-gutter"
    );
    assert!(
        approx(kid.rect.w, 200.0 - crate::SCROLLBAR_THICKNESS),
        "RTL stable kid w {} != 188",
        kid.rect.w
    );
}

#[test]
fn scrollbar_gutter_stable_ltr_keeps_children_at_left_edge() {
    // LTR control: the strip is on the right, so the content origin
    // does not move — only the width narrows.
    let l = layout_html(
        "<div id=\"c\"><div style=\"width: 100%; height: 10px;\">x</div></div>",
        "body { margin: 0; } #c { overflow-y: auto; width: 200px; height: 100px; scrollbar-gutter: stable; }",
    );
    let c = &body(&l).children[0];
    let kid = &c.children[0];
    assert!(approx(kid.rect.x - c.rect.x, 0.0), "LTR stable kid x");
    assert!(
        approx(kid.rect.w, 200.0 - crate::SCROLLBAR_THICKNESS),
        "LTR stable kid w {} != 188",
        kid.rect.w
    );
}

#[test]
fn scrollbar_gutter_both_edges_rtl_shifts_and_narrows_both() {
    // `stable both-edges` under RTL: strips on both sides — children
    // shift by the left strip and narrow by both.
    let l = layout_html(
        "<div id=\"c\"><div style=\"width: 100%; height: 10px;\">x</div></div>",
        "body { margin: 0; } #c { direction: rtl; overflow-y: auto; width: 200px; height: 100px; scrollbar-gutter: stable both-edges; }",
    );
    let c = &body(&l).children[0];
    let kid = &c.children[0];
    assert!(
        approx(kid.rect.x - c.rect.x, crate::SCROLLBAR_THICKNESS),
        "both-edges RTL kid x offset {} != 12",
        kid.rect.x - c.rect.x
    );
    assert!(
        approx(kid.rect.w, 200.0 - 2.0 * crate::SCROLLBAR_THICKNESS),
        "both-edges RTL kid w {} != 176",
        kid.rect.w
    );
}

#[test]
fn scrollbar_gutter_stable_rtl_shifts_inline_content() {
    // The inline formatting path (direct `do_inline_layout` and the
    // resumed-inline path for inline-block children) takes the same
    // shifted content origin as the block path.
    // `text-align: left` isolates the gutter shift from RTL
    // `start` alignment (which would right-align the line anyway).
    let l = layout_html(
        "<div id=\"c\">hi</div>",
        "body { margin: 0; } #c { direction: rtl; text-align: left; overflow-y: auto; width: 200px; height: 100px; scrollbar-gutter: stable; font-size: 10px; }",
    );
    let c = &body(&l).children[0];
    let text = &c.children[0];
    assert!(
        approx(text.rect.x - c.rect.x, crate::SCROLLBAR_THICKNESS),
        "RTL stable text x offset {} != 12",
        text.rect.x - c.rect.x
    );

    let l = layout_html(
        "<div id=\"c\"><span style=\"display: inline-block; width: 10px; height: 10px;\"></span></div>",
        "body { margin: 0; } #c { direction: rtl; text-align: left; overflow-y: auto; width: 200px; height: 100px; scrollbar-gutter: stable; }",
    );
    let c = &body(&l).children[0];
    let ib = &c.children[0];
    assert!(
        approx(ib.rect.x - c.rect.x, crate::SCROLLBAR_THICKNESS),
        "RTL stable inline-block x offset {} != 12",
        ib.rect.x - c.rect.x
    );
}

#[test]
fn scrollbar_gutter_on_rtl_root_propagates_to_viewport_left() {
    // The viewport reservation shares `stable_gutter_amounts`: an RTL
    // root's strip is on the left, so the root's border box shifts by
    // it (and narrows by it) instead of narrowing from the right.
    let l = layout_html(
        "<div style=\"width: 100%; height: 200vh;\"></div>",
        ":root { scrollbar-gutter: stable; overflow: auto; direction: rtl; } \
         body, html { margin: 0; padding: 0; border: none; }",
    );
    assert!(
        approx(l.rect.x, crate::SCROLLBAR_THICKNESS),
        "RTL root x {} != 12",
        l.rect.x
    );
    assert!(
        approx(l.rect.w, VP.w - crate::SCROLLBAR_THICKNESS),
        "RTL root w {} != viewport - gutter",
        l.rect.w
    );
    let content = &body(&l).children[0];
    assert!(
        approx(content.rect.w, l.rect.w),
        "content w {} != root w {} (no double reservation)",
        content.rect.w,
        l.rect.w
    );
}

#[test]
fn forced_scrollbar_consumes_content_width() {
    // CSS Overflow 3 §2.2: `overflow: scroll` forces the vertical
    // classic scrollbar to display whether or not content overflows,
    // and a displayed classic scrollbar consumes its thickness from
    // the scroll container's content box. WPT
    // `scrollbar-gutter-root.html` emulates the engine's scrollbar
    // width exactly this way — the `offsetWidth` difference across a
    // forced scroller — so the child must come out one
    // SCROLLBAR_THICKNESS narrower than the container.
    let l = layout_html(
        "<div id=\"outer\"><div id=\"inner\">x</div></div>",
        "#outer { overflow: scroll; width: 100px; height: 100px; } \
         #inner { width: 100%; height: 200%; }",
    );
    let inner = &body(&l).children[0].children[0];
    assert!(
        approx(inner.rect.w, 100.0 - crate::SCROLLBAR_THICKNESS),
        "forced-scroller child width {} != 88",
        inner.rect.w
    );
}

#[test]
fn forced_scrollbar_does_not_stack_with_stable_gutter() {
    // The forced strip sits at the inline-end edge — the same edge a
    // shown scrollbar paints — so it never stacks with the
    // `scrollbar-gutter: stable` reservation: the per-side max wins
    // (a shown scrollbar paints exactly the reserved strip). WPT
    // `scrollbar-gutter-001.html` "overflow scroll, scrollbar-gutter
    // stable" still only expects the single strip.
    let l = layout_html(
        "<div><div style=\"width: 100%; height: 10px;\">x</div></div>",
        "div { overflow-y: scroll; width: 200px; height: 100px; scrollbar-gutter: stable; }",
    );
    let kid = &body(&l).children[0].children[0];
    assert!(
        approx(kid.rect.w, 200.0 - crate::SCROLLBAR_THICKNESS),
        "scroll+stable child width {} != 188 (must not double-subtract)",
        kid.rect.w
    );
}

#[test]
fn forced_scrollbar_rtl_consumes_inline_start() {
    // Under `direction: rtl` the forced vertical strip is on the
    // inline-start (left) edge: the child narrows by the thickness and
    // its origin shifts past the strip.
    let l = layout_html(
        "<div><div style=\"width: 100%; height: 10px;\">x</div></div>",
        "body { margin: 0; } \
         div { direction: rtl; overflow-y: scroll; width: 200px; height: 100px; }",
    );
    let container = &body(&l).children[0];
    let kid = &container.children[0];
    assert!(
        approx(kid.rect.w, 200.0 - crate::SCROLLBAR_THICKNESS),
        "rtl forced-scroller child width {} != 188",
        kid.rect.w
    );
    assert!(
        approx(kid.rect.x - container.rect.x, crate::SCROLLBAR_THICKNESS),
        "rtl forced-scroller child x offset {} != 12",
        kid.rect.x - container.rect.x
    );
}

#[test]
fn auto_scrollbar_consumes_content_width_on_real_overflow() {
    // CSS Overflow 3 §2.2: `overflow-y: auto` shows the classic
    // vertical scrollbar exactly when content overflows the padding
    // box, and the shown scrollbar consumes its thickness from the
    // content box — otherwise it paints over the last
    // SCROLLBAR_THICKNESS px of content. (The unconditional
    // `overflow-y: scroll` case is `forced_scrollbar_consumes_content_width`.)
    let l = layout_html(
        "<div id=\"outer\"><div id=\"inner\">x</div></div>",
        "#outer { overflow-y: auto; width: 100px; height: 100px; } \
         #inner { width: 100%; height: 200%; }",
    );
    let inner = &body(&l).children[0].children[0];
    assert!(
        approx(inner.rect.w, 100.0 - crate::SCROLLBAR_THICKNESS),
        "auto-scroller child width {} != 88",
        inner.rect.w
    );
}

#[test]
fn auto_scrollbar_no_narrowing_without_overflow() {
    // No vertical overflow → no scrollbar → the content width is
    // untouched.
    let l = layout_html(
        "<div id=\"outer\"><div id=\"inner\">x</div></div>",
        "#outer { overflow-y: auto; width: 100px; height: 100px; } \
         #inner { width: 100%; height: 50%; }",
    );
    let inner = &body(&l).children[0].children[0];
    assert!(
        approx(inner.rect.w, 100.0),
        "non-overflowing auto container child width {} != 100",
        inner.rect.w
    );
}

#[test]
fn auto_scrollbar_auto_height_grows_instead() {
    // With an auto height the box grows to fit its content: no
    // overflow, no scrollbar, no narrowing.
    let l = layout_html(
        "<div id=\"outer\"><div id=\"inner\">x</div></div>",
        "#outer { overflow-y: auto; width: 100px; } \
         #inner { width: 100%; height: 200px; }",
    );
    let outer = &body(&l).children[0];
    let inner = &outer.children[0];
    assert!(
        approx(outer.rect.h, 200.0),
        "auto-height container grew to {} != 200",
        outer.rect.h
    );
    assert!(
        approx(inner.rect.w, 100.0),
        "auto-height auto container child width {} != 100",
        inner.rect.w
    );
}

#[test]
fn auto_scrollbar_redrives_flex_container() {
    // The `overflow-y: auto` scrollbar re-drive applies to flex
    // containers too: a definite-height flex container with real
    // overflow shows the classic scrollbar, which consumes its
    // thickness from the content box — the item at `width: 100%` must
    // come out SCROLLBAR_THICKNESS narrower, not painted under the bar.
    let l = layout_html(
        "<div id=\"outer\"><div id=\"inner\">x</div></div>",
        "body { margin: 0; } \
         #outer { display: flex; overflow-y: auto; width: 100px; height: 100px; } \
         #inner { width: 100%; height: 200%; }",
    );
    let inner = &body(&l).children[0].children[0];
    assert!(
        approx(inner.rect.w, 100.0 - crate::SCROLLBAR_THICKNESS),
        "flex auto-scroller child width {} != 88",
        inner.rect.w
    );
}

#[test]
fn auto_scrollbar_redrives_grid_container() {
    // Same for grid containers: the item's used width reflects the
    // narrowed content box after the re-drive.
    let l = layout_html(
        "<div id=\"outer\"><div id=\"inner\">x</div></div>",
        "body { margin: 0; } \
         #outer { display: grid; overflow-y: auto; width: 100px; height: 100px; } \
         #inner { width: 100%; height: 200%; }",
    );
    let inner = &body(&l).children[0].children[0];
    assert!(
        approx(inner.rect.w, 100.0 - crate::SCROLLBAR_THICKNESS),
        "grid auto-scroller child width {} != 88",
        inner.rect.w
    );
}

#[test]
fn auto_scrollbar_redrive_refreshes_abspos_static_position() {
    // The re-drive re-lays out the subtree against the narrowed width:
    // an abspos child with auto offsets must take its static position
    // from the *second* pass, not the first pass's stale record. The
    // paragraph wraps to twice as many lines at the narrowed width
    // (two 5-char words fit per 110px line but only one per 98px
    // line), so a stale record would place the abspos box four lines
    // too high.
    let l = layout_html(
        "<div id=\"outer\"><div id=\"para\">aaaaa bbbbb ccccc ddddd eeeee fffff ggggg hhhhh</div><div id=\"abs\"></div></div>",
        "body { margin: 0; } \
         #outer { position: relative; overflow-y: auto; width: 110px; height: 40px; } \
         #abs { position: absolute; width: 10px; height: 10px; }",
    );
    let outer = &body(&l).children[0];
    let para = &outer.children[0];
    let abs = &outer.children[1];
    // Sanity: the re-drive really did rewrap the paragraph (8 lines,
    // not the first pass's 4).
    assert!(
        approx(para.rect.h, 8.0 * 19.2),
        "paragraph height {} != 153.6 (rewrap did not happen)",
        para.rect.h
    );
    // The abspos box keeps the static position flow would have given
    // it: the paragraph's border-box bottom in the final (re-driven)
    // layout.
    assert!(
        approx(abs.rect.y, para.rect.y + para.rect.h),
        "abspos static y {} != paragraph bottom {}",
        abs.rect.y,
        para.rect.y + para.rect.h
    );
}

#[test]
fn auto_scrollbar_does_not_stack_with_stable_gutter() {
    // The auto strip sits at the inline-end edge — the same edge a
    // shown scrollbar paints — so it never stacks with the
    // `scrollbar-gutter: stable` reservation: the per-side max wins.
    let l = layout_html(
        "<div><div style=\"width: 100%; height: 300px;\">x</div></div>",
        "div { overflow-y: auto; width: 200px; height: 100px; scrollbar-gutter: stable; }",
    );
    let kid = &body(&l).children[0].children[0];
    assert!(
        approx(kid.rect.w, 200.0 - crate::SCROLLBAR_THICKNESS),
        "auto+stable child width {} != 188 (must not double-subtract)",
        kid.rect.w
    );
}

#[test]
fn auto_scrollbar_rtl_consumes_inline_start() {
    // Under `direction: rtl` the shown vertical strip is on the
    // inline-start (left) edge: the child narrows by the thickness and
    // its origin shifts past the strip.
    let l = layout_html(
        "<div><div style=\"width: 100%; height: 300px;\">x</div></div>",
        "body { margin: 0; } \
         div { direction: rtl; overflow-y: auto; width: 200px; height: 100px; }",
    );
    let container = &body(&l).children[0];
    let kid = &container.children[0];
    assert!(
        approx(kid.rect.w, 200.0 - crate::SCROLLBAR_THICKNESS),
        "rtl auto-scroller child width {} != 188",
        kid.rect.w
    );
    assert!(
        approx(kid.rect.x - container.rect.x, crate::SCROLLBAR_THICKNESS),
        "rtl auto-scroller child x offset {} != 12",
        kid.rect.x - container.rect.x
    );
}

#[test]
fn auto_scrollbar_hidden_never_narrows() {
    // `overflow-y: hidden` can never show a scrollbar: overflowing
    // content must not narrow the content width.
    let l = layout_html(
        "<div><div style=\"width: 100%; height: 300px;\">x</div></div>",
        "div { overflow-y: hidden; width: 200px; height: 100px; }",
    );
    let kid = &body(&l).children[0].children[0];
    assert!(
        approx(kid.rect.w, 200.0),
        "hidden container child width {} != 200",
        kid.rect.w
    );
}

#[test]
fn auto_scrollbar_no_strip_under_vertical_writing_mode() {
    // Under a vertical writing mode the physical left/right strip
    // cannot express the inline-end edge: reserve nothing rather than
    // narrowing along the wrong axis.
    let l = layout_html(
        "<div><div style=\"width: 100%; height: 300px;\">x</div></div>",
        "div { writing-mode: vertical-rl; overflow-y: auto; width: 200px; height: 100px; }",
    );
    let kid = &body(&l).children[0].children[0];
    assert!(
        approx(kid.rect.w, 200.0),
        "vertical-rl auto container child width {} != 200",
        kid.rect.w
    );
}

#[test]
fn viewport_auto_overflow_narrows_initial_containing_block() {
    // `overflow-y: auto` propagated to the viewport with real document
    // overflow: the shown scrollbar consumes its thickness from the
    // initial containing block, so the root's border box narrows by
    // the strip.
    let l = layout_html(
        "<div style=\"width: 100%; height: 200vh;\"></div>",
        "html { overflow-y: auto; } body, html { margin: 0; padding: 0; border: none; }",
    );
    assert!(
        approx(l.rect.w, VP.w - crate::SCROLLBAR_THICKNESS),
        "root w {} != viewport - scrollbar",
        l.rect.w
    );
    let content = &body(&l).children[0];
    assert!(
        approx(content.rect.w, l.rect.w),
        "content w {} != root w {}",
        content.rect.w,
        l.rect.w
    );
}

#[test]
fn viewport_default_overflow_narrows_initial_containing_block() {
    // Both-`visible` overflow computes to `auto` on the viewport
    // itself: a plain overflowing document gets the same narrowing.
    let l = layout_html(
        "<div style=\"width: 100%; height: 200vh;\"></div>",
        "body, html { margin: 0; padding: 0; border: none; }",
    );
    assert!(
        approx(l.rect.w, VP.w - crate::SCROLLBAR_THICKNESS),
        "root w {} != viewport - scrollbar",
        l.rect.w
    );
}

#[test]
fn viewport_auto_overflow_no_narrowing_without_overflow() {
    // `overflow-y: auto` on the root with no document overflow: no
    // scrollbar, no narrowing.
    let l = layout_html(
        "<div style=\"width: 100%; height: 10px;\"></div>",
        "html { overflow-y: auto; } body, html { margin: 0; padding: 0; border: none; }",
    );
    assert!(
        approx(l.rect.w, VP.w),
        "root w {} != viewport width (no overflow)",
        l.rect.w
    );
}

#[test]
fn viewport_auto_overflow_rtl_shifts_initial_containing_block() {
    // Under root `direction: rtl` the viewport's vertical scrollbar
    // paints at the left edge: the ICB shifts past the strip.
    let l = layout_html(
        "<div style=\"width: 100%; height: 200vh;\"></div>",
        "html { overflow-y: auto; direction: rtl; } body, html { margin: 0; padding: 0; border: none; }",
    );
    assert!(
        approx(l.rect.x, crate::SCROLLBAR_THICKNESS),
        "RTL root x {} != 12",
        l.rect.x
    );
    assert!(
        approx(l.rect.w, VP.w - crate::SCROLLBAR_THICKNESS),
        "RTL root w {} != viewport - scrollbar",
        l.rect.w
    );
}

#[test]
fn viewport_auto_overflow_does_not_stack_with_stable_gutter() {
    // The propagated `scrollbar-gutter: stable` strip already reserves
    // the space the shown scrollbar paints: the auto re-run must not
    // subtract a second strip.
    let l = layout_html(
        "<div style=\"width: 100%; height: 200vh;\"></div>",
        ":root { scrollbar-gutter: stable; overflow-y: auto; } \
         body, html { margin: 0; padding: 0; border: none; }",
    );
    assert!(
        approx(l.rect.w, VP.w - crate::SCROLLBAR_THICKNESS),
        "root w {} != viewport - single strip (must not double-subtract)",
        l.rect.w
    );
}

#[test]
fn scrollbar_gutter_stable_reserves_for_hidden() {
    // CSS Overflow 3 §2: `overflow: hidden` clips but still
    // establishes a scroll container, and `scrollbar-gutter` applies
    // to scroll containers — so `stable` reserves the strip even
    // though no scrollbar can ever appear (WPT
    // `scrollbar-gutter-001.html` "overflow hidden, scrollbar-gutter
    // stable" expects the content narrower than the container).
    let l = layout_html(
        "<div><div style=\"width: 100%; height: 10px;\">x</div></div>",
        "div { overflow-y: hidden; width: 200px; height: 100px; scrollbar-gutter: stable; }",
    );
    let kid = &body(&l).children[0].children[0];
    assert!(
        approx(kid.rect.w, 200.0 - crate::SCROLLBAR_THICKNESS),
        "hidden+stable child width {} != 188",
        kid.rect.w
    );
}

#[test]
fn scrollbar_gutter_stable_both_edges_hidden() {
    // `stable both-edges` on `overflow: hidden`: a strip on each side,
    // shifting the child's origin past the inline-start one (WPT
    // `scrollbar-gutter-001.html` "overflow hidden, scrollbar-gutter
    // stable both-edges").
    let l = layout_html(
        "<div><div style=\"width: 100%; height: 10px;\">x</div></div>",
        "body { margin: 0; } \
         div { overflow-y: hidden; width: 200px; height: 100px; \
               scrollbar-gutter: stable both-edges; }",
    );
    let container = &body(&l).children[0];
    let kid = &container.children[0];
    assert!(
        approx(kid.rect.w, 200.0 - 2.0 * crate::SCROLLBAR_THICKNESS),
        "hidden+both-edges child width {} != 176",
        kid.rect.w
    );
    assert!(
        approx(kid.rect.x - container.rect.x, crate::SCROLLBAR_THICKNESS),
        "hidden+both-edges child x offset {} != 12",
        kid.rect.x - container.rect.x
    );
}

/// First descendant box (preorder, `box_` included) of the given type.
fn find_box_type(box_: &LayoutBox, bt: BoxType) -> Option<&LayoutBox> {
    if box_.box_type == bt {
        return Some(box_);
    }
    box_.children.iter().find_map(|c| find_box_type(c, bt))
}

#[test]
fn scrollbar_gutter_flex_rtl_stable_shifts_and_narrows() {
    // C5: the reservation was computed in the `layout_block` prologue
    // and then discarded for flex containers (the dispatch returned
    // before the block/inline paths that use it). RTL `stable`: the
    // item area shifts past the left strip and narrows by it. The
    // item is `width: 100%`, so it fills the reserved content width.
    let l = layout_html(
        "<div id=\"c\"><div class=\"a\"></div></div>",
        "body { margin: 0; } \
         #c { display: flex; direction: rtl; overflow-y: auto; width: 200px; \
              height: 100px; scrollbar-gutter: stable; } \
         .a { width: 100%; height: 10px; }",
    );
    let c = &body(&l).children[0];
    let a = &c.children[0];
    assert!(
        approx(a.rect.x - c.rect.x, crate::SCROLLBAR_THICKNESS),
        "flex RTL kid x offset {} != 12",
        a.rect.x - c.rect.x
    );
    assert!(
        approx(a.rect.w, 200.0 - crate::SCROLLBAR_THICKNESS),
        "flex RTL kid w {} != 188",
        a.rect.w
    );
}

#[test]
fn scrollbar_gutter_flex_ltr_stable_narrows_without_shift() {
    // C5, LTR arm: the strip is on the right, so the item area only
    // narrows (this half was missing for flex before the fix too).
    let l = layout_html(
        "<div id=\"c\"><div class=\"a\"></div></div>",
        "body { margin: 0; } \
         #c { display: flex; overflow-y: auto; width: 200px; height: 100px; \
              scrollbar-gutter: stable; } \
         .a { width: 100%; height: 10px; }",
    );
    let c = &body(&l).children[0];
    let a = &c.children[0];
    assert!(approx(a.rect.x - c.rect.x, 0.0), "flex LTR kid x");
    assert!(
        approx(a.rect.w, 200.0 - crate::SCROLLBAR_THICKNESS),
        "flex LTR kid w {} != 188",
        a.rect.w
    );
}

#[test]
fn scrollbar_gutter_grid_rtl_and_ltr_stable() {
    // C5, grid arm: a single `1fr` track sizes against the reserved
    // content width and starts at the reserved origin.
    let html = "<div id=\"c\"><div class=\"a\"></div></div>";
    let css = |dir: &str| {
        format!(
            "body {{ margin: 0; }} \
             #c {{ display: grid; direction: {dir}; overflow-y: auto; width: 200px; \
                  height: 100px; scrollbar-gutter: stable; \
                  grid-template-columns: 1fr; }} \
             .a {{ height: 10px; }}"
        )
    };
    let l = layout_html(html, &css("rtl"));
    let c = &body(&l).children[0];
    let a = &c.children[0];
    assert!(
        approx(a.rect.x - c.rect.x, crate::SCROLLBAR_THICKNESS),
        "grid RTL kid x offset {} != 12",
        a.rect.x - c.rect.x
    );
    assert!(
        approx(a.rect.w, 200.0 - crate::SCROLLBAR_THICKNESS),
        "grid RTL kid w {} != 188",
        a.rect.w
    );
    let l = layout_html(html, &css("ltr"));
    let c = &body(&l).children[0];
    let a = &c.children[0];
    assert!(approx(a.rect.x - c.rect.x, 0.0), "grid LTR kid x");
    assert!(
        approx(a.rect.w, 200.0 - crate::SCROLLBAR_THICKNESS),
        "grid LTR kid w {} != 188",
        a.rect.w
    );
}

#[test]
fn scrollbar_gutter_table_rtl_and_ltr_stable() {
    // C5, table arm: the single column resolves against the reserved
    // content width and starts at the reserved origin.
    let html = "<table id=\"c\"><tr><td>x</td></tr></table>";
    let css = |dir: &str| {
        format!(
            "body {{ margin: 0; }} \
             #c {{ direction: {dir}; overflow-y: auto; width: 200px; height: 100px; \
                  scrollbar-gutter: stable; border-collapse: collapse; }} \
             td {{ padding: 0; border: none; font-size: 10px; }}"
        )
    };
    let l = layout_html(html, &css("rtl"));
    let t = &body(&l).children[0];
    let cell = find_box_type(t, BoxType::TableCell).expect("cell box");
    assert!(
        approx(cell.rect.x - t.rect.x, crate::SCROLLBAR_THICKNESS),
        "table RTL cell x offset {} != 12",
        cell.rect.x - t.rect.x
    );
    assert!(
        approx(cell.rect.w, 200.0 - crate::SCROLLBAR_THICKNESS),
        "table RTL cell w {} != 188",
        cell.rect.w
    );
    let l = layout_html(html, &css("ltr"));
    let t = &body(&l).children[0];
    let cell = find_box_type(t, BoxType::TableCell).expect("cell box");
    assert!(approx(cell.rect.x - t.rect.x, 0.0), "table LTR cell x");
    assert!(
        approx(cell.rect.w, 200.0 - crate::SCROLLBAR_THICKNESS),
        "table LTR cell w {} != 188",
        cell.rect.w
    );
}

#[test]
fn scrollbar_gutter_table_float_uses_reserved_width() {
    // N4: a float directly under a `scrollbar-gutter: stable` table is
    // placed against the reserved content width, not the node's
    // unreserved metrics width. LTR: the strip is on the right, so a
    // right float must stop one gutter short of the table edge
    // instead of intruding into the strip. RTL: the strip is on the
    // left, so the float's containing area ends at the table edge
    // instead of overflowing past it. (`display: table` on divs, not
    // a `<table>` element, so the parser does not foster-parent the
    // float out.)
    let html = "<div class=\"t\"><div class=\"r\"><div class=\"c\">x</div></div>\
                <div class=\"f\">f</div></div>";
    let css = |dir: &str| {
        format!(
            "body {{ margin: 0; }} \
             .t {{ display: table; direction: {dir}; overflow-y: auto; width: 200px; height: 100px; \
                  scrollbar-gutter: stable; border-collapse: collapse; }} \
             .r {{ display: table-row; }} \
             .c {{ display: table-cell; padding: 0; border: none; font-size: 10px; }} \
             .f {{ float: right; width: 50px; height: 10px; font-size: 10px; }}"
        )
    };
    let float_box = |l: &LayoutBox| {
        let t = &body(l).children[0];
        let f = t
            .children
            .iter()
            .find(|c| c.box_type == BoxType::Block)
            .expect("float box");
        (t.rect.x, f.rect.x, f.rect.w)
    };
    let (tx, fx, fw) = float_box(&layout_html(html, &css("rtl")));
    assert!(
        approx(fx + fw - tx, 200.0),
        "RTL float right edge at the table edge: {}",
        fx + fw - tx
    );
    let (tx, fx, fw) = float_box(&layout_html(html, &css("ltr")));
    assert!(
        approx(fx + fw - tx, 200.0 - crate::SCROLLBAR_THICKNESS),
        "LTR float right edge clear of the strip: {}",
        fx + fw - tx
    );
}

#[test]
fn scrollbar_gutter_both_edges_narrows_both_sides() {
    // `stable both-edges` reserves on both inline sides.
    let l = layout_html(
        "<div><div style=\"width: 100%; height: 10px;\">x</div></div>",
        "div { overflow-y: auto; width: 200px; height: 100px; scrollbar-gutter: stable both-edges; }",
    );
    let kid = &body(&l).children[0].children[0];
    assert!(
        (kid.rect.w - (200.0 - 2.0 * crate::SCROLLBAR_THICKNESS)).abs() < 0.01,
        "kid width {} != 176",
        kid.rect.w
    );
}

// css-sizing-3 intrinsic widths. `FixedFontMetrics` advances 0.6em per
// char, so at `font-size: 10px` "hello world" is 66px max-content and
// 30px min-content (the longest word).

#[test]
fn width_fit_content_shrink_wraps_block() {
    let l = layout_html(
        r#"<div class="a">hello world</div>"#,
        ".a { font-size: 10px; width: fit-content; }",
    );
    let a = &body(&l).children[0];
    // Short enough for one line: fit-content == max-content, far under
    // the 800px fill an `auto` width would take.
    assert!(approx(a.rect.w, 66.0), "fit-content {}", a.rect.w);
}

#[test]
fn width_min_content_uses_longest_word() {
    let l = layout_html(
        r#"<div class="a">hello world</div>"#,
        ".a { font-size: 10px; width: min-content; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 30.0), "min-content {}", a.rect.w);
}

#[test]
fn width_max_content_uses_full_text() {
    let l = layout_html(
        r#"<div class="a">hello world</div>"#,
        ".a { font-size: 10px; width: max-content; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 66.0), "max-content {}", a.rect.w);
}

#[test]
fn width_fit_content_clamped_to_available() {
    // 20 words: max-content (1434px) exceeds the 800px viewport, so
    // min(max-content, max(min-content, available)) caps at available.
    let words = vec!["hello world"; 20].join(" ");
    let l = layout_html(
        &format!(r#"<div class="a">{words}</div>"#),
        ".a { font-size: 10px; width: fit-content; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 800.0), "fit-content {}", a.rect.w);
}

#[test]
fn width_fit_content_respects_min_max_clamps() {
    let l = layout_html(
        r#"<div class="a">hello world</div>"#,
        ".a { font-size: 10px; width: fit-content; max-width: 40px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 40.0), "max-width clamp {}", a.rect.w);
    // min-width wins when over-constrained (mirrors `clamp_w`).
    let l = layout_html(
        r#"<div class="a">hi</div>"#,
        ".a { font-size: 10px; width: fit-content; min-width: 200px; max-width: 100px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 200.0), "min-wins {}", a.rect.w);
}

#[test]
fn inline_size_fit_content_maps_to_width() {
    let l = layout_html(
        r#"<div class="a">hello world</div>"#,
        ".a { font-size: 10px; inline-size: fit-content; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 66.0), "inline-size {}", a.rect.w);
}

#[test]
fn float_width_min_content() {
    let l = layout_html(
        r#"<div class="a">hello world</div>"#,
        ".a { float: left; font-size: 10px; width: min-content; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 30.0), "float min-content {}", a.rect.w);
}

#[test]
fn table_width_fit_content_shrink_wraps_like_auto() {
    let html = r#"<table class="t"><tr><td>hi</td></tr></table>"#;
    let auto_l = layout_html(html, ".t { font-size: 10px; }");
    let fit_l = layout_html(html, ".t { font-size: 10px; width: fit-content; }");
    let max_l = layout_html(html, ".t { font-size: 10px; width: max-content; }");
    let auto_t = &body(&auto_l).children[0];
    let fit_t = &body(&fit_l).children[0];
    let max_t = &body(&max_l).children[0];
    // A block-level table already shrink-to-fits on `auto`;
    // `fit-content` takes the same path.
    assert!(
        approx(auto_t.rect.w, fit_t.rect.w),
        "auto {} fit {}",
        auto_t.rect.w,
        fit_t.rect.w
    );
    assert!(
        approx(max_t.rect.w, fit_t.rect.w),
        "max {} fit {}",
        max_t.rect.w,
        fit_t.rect.w
    );
    assert!(fit_t.rect.w < 800.0, "table {}", fit_t.rect.w);
}

#[test]
fn flex_item_width_fit_content() {
    let l = layout_html(
        r#"<div class="f"><div class="i">hello world</div></div>"#,
        ".f { display: flex; width: 500px; font-size: 10px; } .i { width: fit-content; }",
    );
    let f = &body(&l).children[0];
    let i = &f.children[0];
    // No flex-grow: the item keeps its fit-content (== max-content for
    // this short text) hypothetical main size.
    assert!(approx(i.rect.w, 66.0), "flex item {}", i.rect.w);
}

#[test]
fn flex_item_width_min_content() {
    let l = layout_html(
        r#"<div class="f"><div class="i">hello world</div></div>"#,
        ".f { display: flex; width: 500px; font-size: 10px; } .i { width: min-content; flex: none; }",
    );
    let f = &body(&l).children[0];
    let i = &f.children[0];
    assert!(approx(i.rect.w, 30.0), "flex item {}", i.rect.w);
}

#[test]
fn grid_item_width_min_content() {
    let l = layout_html(
        r#"<div class="g"><div class="i">hello world</div></div>"#,
        ".g { display: grid; grid-template-columns: 400px; font-size: 10px; } .i { width: min-content; }",
    );
    let g = &body(&l).children[0];
    let i = &g.children[0];
    // A non-`auto` width opts out of stretch; the item shrink-fits to
    // its min-content size.
    assert!(approx(i.rect.w, 30.0), "grid item {}", i.rect.w);
}

// F3: a BFC root with `width: fit-content` must measure against the
// narrowed available width beside floats, not the provisional width
// from its full-width layout. Eight "hello world"s: max-content 570px,
// which exceeds the 500px beside the 300px float but fits once
// re-measured against the narrowed width (fit-content clamps to the
// available 500px). With the fill-provisional fit width the box
// cleared below the float (y == 50).
#[test]
fn bfc_fit_content_child_sits_beside_float() {
    let words = ["hello world"; 8].join(" ");
    let l = layout_html(
        &format!(r#"<div class="c"><div class="fl"></div><div class="bfc">{words}</div></div>"#),
        ".c { width: 800px; } .fl { float: left; width: 300px; height: 50px; } \
         .bfc { overflow: hidden; font-size: 10px; width: fit-content; }",
    );
    let c = &body(&l).children[0];
    let bfc = &c.children[1];
    assert!(
        approx(bfc.rect.y, 0.0),
        "cleared below float: y {}",
        bfc.rect.y
    );
    assert!(approx(bfc.rect.x, 300.0), "x {}", bfc.rect.x);
    assert!(approx(bfc.rect.w, 500.0), "w {}", bfc.rect.w);
}

// F4: css-sizing-3 §4 — `min-width` accepts the content-based keywords
// too. `SpecLen::resolve` returns `None` for them, which the old clamp
// silently defaulted to 0, dropping the constraint.
#[test]
fn width_fit_content_min_width_max_content_keyword() {
    // 20 words: max-content is 1434px; fit-content alone would clamp to
    // the 800px available width.
    let words = vec!["hello world"; 20].join(" ");
    let l = layout_html(
        &format!(r#"<div class="a">{words}</div>"#),
        ".a { font-size: 10px; width: fit-content; min-width: max-content; }",
    );
    let a = &body(&l).children[0];
    assert!(
        approx(a.rect.w, 1434.0),
        "min-width: max-content {}",
        a.rect.w
    );
}

#[test]
fn width_fit_content_min_width_min_content_keyword() {
    // min-content (30px) cannot lift a 66px fit-content width: guards the
    // F4 clamp against over-applying (min-width must not shrink).
    let l = layout_html(
        r#"<div class="a">hello world</div>"#,
        ".a { font-size: 10px; width: fit-content; min-width: min-content; }",
    );
    let a = &body(&l).children[0];
    assert!(
        approx(a.rect.w, 66.0),
        "min-width: min-content {}",
        a.rect.w
    );
}

// F4 through the flex main-axis clamp: a row item's `min-width` keyword
// is measured against the item's available width (F5), not dropped.
#[test]
fn flex_item_min_width_max_content_keyword() {
    // `width: auto`, so the max-content size is content-driven (570px);
    // the 100px flex-basis is floored by the keyword min-width.
    let words = ["hello world"; 8].join(" ");
    let l = layout_html(
        &format!(r#"<div class="f"><div class="i">{words}</div></div>"#),
        ".f { display: flex; width: 500px; font-size: 10px; } \
         .i { flex-basis: 100px; min-width: max-content; }",
    );
    let f = &body(&l).children[0];
    let i = &f.children[0];
    assert!(approx(i.rect.w, 570.0), "flex item {}", i.rect.w);
}

// css-sizing-3 §5.2: a block-level box with a definite specified width
// contributes exactly that width to BOTH intrinsic sizes. The
// `width: fit-content` parent sits in a 100px-wide outer container, so
// `available` (100px) binds: with the fixed-width child feeding
// min-content the parent sizes to 200px, not 100px.
#[test]
fn fit_content_parent_min_content_includes_fixed_child_width() {
    let l = layout_html(
        r#"<div class="o"><div class="p"><div class="c">xxxx</div></div></div>"#,
        ".o { width: 100px; font-size: 10px; } .p { width: fit-content; } .c { width: 200px; }",
    );
    let p = &body(&l).children[0].children[0];
    assert!(approx(p.rect.w, 200.0), "fit-content parent {}", p.rect.w);
}

// CSS Values 4 §8.1.4: a `calc()` that syntactically contained a `%`
// leaf is cyclic against an indefinite main-size base, so `flex-basis`
// behaves as `content` — like a bare percentage — instead of resolving
// the folded constant.
#[test]
fn column_flex_basis_calc_with_pct_behaves_as_content() {
    // `calc(10px + 0%)` folds to 10px but had a `%` leaf: the empty item
    // measures its content (0px), not the 10px constant.
    let l = layout_html(
        r#"<div class="f"><div class="i"></div></div>"#,
        ".f { display: flex; flex-direction: column; width: 200px; } .i { flex-basis: calc(10px + 0%); }",
    );
    let i = &body(&l).children[0].children[0];
    assert!(approx(i.rect.h, 0.0), "flex item {}", i.rect.h);
    // `calc(50%)` resolves against a zero base (0px) without the gate;
    // with it the item measures its 30px content instead.
    let l = layout_html(
        r#"<div class="f"><div class="i"><div class="k"></div></div></div>"#,
        ".f { display: flex; flex-direction: column; width: 200px; } .i { flex-basis: calc(50%); } .k { height: 30px; }",
    );
    let i = &body(&l).children[0].children[0];
    assert!(approx(i.rect.h, 30.0), "flex item {}", i.rect.h);
}

// === w51 regression tests ===

fn all_table_rows<'a>(b: &'a LayoutBox, out: &mut Vec<&'a LayoutBox>) {
    if b.box_type == BoxType::TableRow {
        out.push(b);
    }
    for c in &b.children {
        all_table_rows(c, out);
    }
}

#[test]
fn visibility_collapse_row_group_collapses_visible_rows() {
    // WPT css/css-tables/visibility-collapse-row-group-001: collapsing
    // a row group removes the whole group as a unit (CSS 2.1 §17.5.5),
    // so a row that explicitly overrides back to `visibility: visible`
    // still takes no space.
    let l = layout_html(
        r#"<table class="t"><tbody class="g"><tr class="r"><td class="c">x</td></tr><tr class="r"><td class="c">y</td></tr></tbody><tr class="r3"><td class="c">z</td></tr></table>"#,
        ".t { border: 5px solid black; } .g { visibility: collapse; } .r { visibility: visible; } .c { font-size: 10px; padding: 0; }",
    );
    let t = &body(&l).children[0];
    let mut rows = Vec::new();
    all_table_rows(t, &mut rows);
    assert_eq!(rows.len(), 3);
    assert!(approx(rows[0].rect.h, 0.0), "row1 h = {}", rows[0].rect.h);
    assert!(approx(rows[1].rect.h, 0.0), "row2 h = {}", rows[1].rect.h);
    assert!(approx(rows[2].rect.h, 12.0), "row3 h = {}", rows[2].rect.h);
    // Only the uncollapsed row contributes: 12px + 2x5px border.
    assert!(approx(t.rect.h, 22.0), "table h = {}", t.rect.h);
}

#[test]
fn visibility_collapse_row_group_inherited_still_collapses() {
    // Control: rows without the override inherit `collapse` from the
    // group and collapse (unchanged behavior).
    let l = layout_html(
        r#"<table class="t"><tbody class="g"><tr><td class="c">x</td></tr></tbody><tr><td class="c">z</td></tr></table>"#,
        ".t { border: 5px solid black; } .g { visibility: collapse; } .c { font-size: 10px; padding: 0; }",
    );
    let t = &body(&l).children[0];
    let mut rows = Vec::new();
    all_table_rows(t, &mut rows);
    assert_eq!(rows.len(), 2);
    assert!(approx(rows[0].rect.h, 0.0), "row1 h = {}", rows[0].rect.h);
    assert!(approx(rows[1].rect.h, 12.0), "row2 h = {}", rows[1].rect.h);
    assert!(approx(t.rect.h, 22.0), "table h = {}", t.rect.h);
}

// === w51 regression tests ===

#[test]
fn scrollbar_gutter_vertical_writing_mode_reserves_nothing_horizontally() {
    // WPT css/css-overflow/scrollbar-gutter-vertical-rl-002 (reduced):
    // under a vertical writing mode the inline axis is vertical, so a
    // `scrollbar-gutter: stable` strip belongs at the inline-end edge
    // (top/bottom), not left/right. The physical left/right
    // reservation must not narrow the width nor shift the content
    // origin along the wrong axis.
    let l = layout_html(
        r#"<div class="line"><div class="c"><div class="k">x</div></div></div>"#,
        ".line { display: flex; } \
         .c { writing-mode: vertical-rl; block-size: 200px; inline-size: 200px; margin: 10px; \
              overflow-x: auto; scrollbar-gutter: stable; } \
         .k { inline-size: 100%; block-size: 200%; }",
    );
    let c = &body(&l).children[0].children[0];
    let k = &c.children[0];
    // No horizontal reservation: the kid lays out against the full
    // 200px width (block-size: 200% = 400) from the content origin.
    assert!(approx(k.rect.x, c.rect.x), "k.x = {}", k.rect.x);
    assert!(approx(k.rect.w, 400.0), "k.w = {}", k.rect.w);
}

#[test]
fn scrollbar_gutter_vertical_both_edges_reserves_nothing_horizontally() {
    // Both-edges variant: likewise no left/right reservation under a
    // vertical writing mode.
    let l = layout_html(
        r#"<div class="line"><div class="c"><div class="k">x</div></div></div>"#,
        ".line { display: flex; } \
         .c { writing-mode: vertical-rl; block-size: 200px; inline-size: 200px; margin: 10px; \
              overflow-x: auto; scrollbar-gutter: stable both-edges; } \
         .k { inline-size: 100%; block-size: 200%; }",
    );
    let c = &body(&l).children[0].children[0];
    let k = &c.children[0];
    assert!(approx(k.rect.x, c.rect.x), "k.x = {}", k.rect.x);
    assert!(approx(k.rect.w, 400.0), "k.w = {}", k.rect.w);
}

// === w52 replaced-element intrinsic contributions ===

// css-sizing-3 §5.2: a replaced box's max-content AND min-content
// contribution is its intrinsic width — including when an *ancestor*
// measures it (the main layout paths already honored this; the
// intrinsic measurement dispatches did not, contributing 0 for a
// childless replaced box with `width: auto`).

fn doc_with_img_intrinsic(
    html: &str,
    w: f32,
    h: f32,
) -> (
    bowser_dom::Document,
    HashMap<bowser_dom::NodeId, IntrinsicDims>,
) {
    let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
    // Recursive: the <img> is usually nested inside a container div.
    fn find_tag(
        doc: &bowser_dom::Document,
        n: bowser_dom::NodeId,
        tag: &str,
    ) -> Option<bowser_dom::NodeId> {
        if matches!(&doc.kind(n), bowser_dom::NodeKind::Element(d) if d.tag_name == tag) {
            return Some(n);
        }
        doc.children(n)
            .into_iter()
            .find_map(|c| find_tag(doc, c, tag))
    }
    let root = doc.document_element().expect("document element");
    let img = find_tag(&doc, root, "img").expect("img element");
    let mut map = HashMap::new();
    map.insert(img, IntrinsicDims::from_size(w, h));
    (doc, map)
}

#[test]
fn replaced_img_sizes_grid_auto_track() {
    // `grid-template-columns: auto` containing an <img> sizes the track
    // to the image's intrinsic width, not 0 (`inline-grid` shrink-wraps
    // so the track size is directly observable).
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="g"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".g { display: inline-grid; grid-template-columns: auto; }",
        &FixedFontMetrics,
        &map,
    );
    let g = &body(&l).children[0];
    assert!(approx(g.rect.w, 100.0), "grid {}", g.rect.w);
    assert!(
        approx(g.children[0].rect.w, 100.0),
        "track {}",
        g.children[0].rect.w
    );
}

#[test]
fn replaced_img_grid_auto_track_clamped_by_max_width() {
    // The intrinsic contribution is clamped by min/max-width (§4: min
    // wins when over-constrained), exactly like a specified width.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="g"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".g { display: inline-grid; grid-template-columns: auto; } .i { max-width: 60px; }",
        &FixedFontMetrics,
        &map,
    );
    let g = &body(&l).children[0];
    assert!(approx(g.rect.w, 60.0), "grid {}", g.rect.w);
    assert!(
        approx(g.children[0].rect.w, 60.0),
        "track {}",
        g.children[0].rect.w
    );
}

#[test]
fn replaced_img_flex_item_intrinsic_sum() {
    // A row flex container's max-content is the items' intrinsic sum:
    // the <img> contributes its intrinsic width.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="f"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".f { display: inline-flex; }",
        &FixedFontMetrics,
        &map,
    );
    let f = &body(&l).children[0];
    assert!(approx(f.rect.w, 100.0), "container {}", f.rect.w);
}

#[test]
fn replaced_img_table_cell_minimum() {
    // A replaced cell's min-content contribution is its intrinsic
    // width: the auto column sizes to it.
    let (doc, map) = doc_with_img_intrinsic(
        r#"<table><tr><td><img class="i"></td></tr></table>"#,
        100.0,
        50.0,
    );
    let l = layout_doc_with_intrinsics(
        &doc,
        "table { border-spacing: 0; } td { padding: 0; }",
        &FixedFontMetrics,
        &map,
    );
    let cell = table_cell(&l, 0, 0);
    assert!(approx(cell.rect.w, 100.0), "cell {}", cell.rect.w);
}

#[test]
fn replaced_img_block_width_auto_contributes_intrinsic() {
    // Block-level replaced with `width: auto`: its max-content
    // contribution is its intrinsic width, so a fit-content parent
    // sizes to it.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="p"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".p { width: fit-content; } .i { display: block; }",
        &FixedFontMetrics,
        &map,
    );
    let p = &body(&l).children[0];
    assert!(approx(p.rect.w, 100.0), "parent {}", p.rect.w);
}

#[test]
fn replaced_img_calc_width_with_pct_keeps_transfer_through() {
    // `width: calc(20px + 50%)` on a replaced element resolves against
    // the zero base (20px): the definite specified width wins over the
    // intrinsic width in intrinsic measurement, exactly as in
    // used-value layout.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="p"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".p { width: fit-content; } .i { display: block; width: calc(20px + 50%); }",
        &FixedFontMetrics,
        &map,
    );
    let p = &body(&l).children[0];
    assert!(approx(p.rect.w, 20.0), "parent {}", p.rect.w);
}

// CSS Values 4 §8.1.4: `min-width: calc(10px + 0%)` folds to 10px but
// had a `%` leaf, so against the indefinite main-size base of a column
// flex container it behaves as `auto` (0.0), not as a 10px floor. The
// floor would surface through `clamp_flex_main` when an ancestor
// intrinsically sizes the column container (via
// `flex_item_intrinsic_contribution`).
#[test]
fn column_flex_item_min_width_calc_with_pct_behaves_as_none() {
    let l = layout_html(
        r#"<div class="p"><div class="f"><div class="i"></div></div></div>"#,
        ".p { width: fit-content; } .f { display: flex; flex-direction: column; } \
         .i { min-width: calc(10px + 0%); }",
    );
    let p = &body(&l).children[0];
    assert!(approx(p.rect.w, 0.0), "parent {}", p.rect.w);
}

// --- wave-55: shared style / text plumbing ---

/// `Rc<ComputedStyle>` threading: identical sibling boxes share one
/// style allocation end to end (styler memo → arena `Node` →
/// `LayoutBox`), instead of each box deep-cloning the delta entries.
#[test]
fn style_rc_shared_between_identical_boxes() {
    use std::rc::Rc;
    let root = layout_html("<div></div><div></div>", "");
    let body = &root.children[0];
    assert_eq!(body.children.len(), 2, "two div boxes");
    let sa = body.children[0].style.as_ref().expect("div has style");
    let sb = body.children[1].style.as_ref().expect("div has style");
    assert!(
        Rc::ptr_eq(sa, sb),
        "identical boxes share one style Rc, not two clones"
    );
    // Values still read through the shared handle.
    assert_eq!(sa.get_str("display"), Some("block"));
}

/// `LayoutBox.text` is the shared `Arc<str>`: no `String`
/// materialization at the arena-to-tree boundary, and cloning the tree
/// shares the allocation instead of copying bytes.
#[test]
fn text_box_text_is_shared_arc() {
    use std::sync::Arc;
    let root = layout_html("<div>hello</div>", "");
    let text = root.children[0].children[0]
        .children
        .iter()
        .find(|b| b.box_type == BoxType::Text)
        .expect("text box");
    assert_eq!(text.text.as_deref(), Some("hello"));
    let dup = root.clone();
    let dtext = dup.children[0].children[0]
        .children
        .iter()
        .find(|b| b.box_type == BoxType::Text)
        .expect("text box");
    assert!(
        Arc::ptr_eq(text.text.as_ref().unwrap(), dtext.text.as_ref().unwrap()),
        "tree clone shares the text Arc"
    );
}

/// Size pins for the hot path: `LayoutBox` stays within its 128-byte
/// tripwire, the shared handles are pointer-sized, and `StyleData`
/// rides inline in the arena node (no per-box `Box`).
#[test]
fn hot_struct_sizes() {
    use std::mem::size_of;
    use std::rc::Rc;
    use std::sync::Arc;
    assert!(
        size_of::<LayoutBox>() <= 128,
        "LayoutBox = {}",
        size_of::<LayoutBox>()
    );
    assert_eq!(size_of::<Option<Arc<str>>>(), 16);
    assert_eq!(size_of::<Option<Rc<bowser_css::ComputedStyle>>>(), 8);
    // Pinned: `StyleData` rides inline in the arena node, so growth here
    // grows every arena node; grow deliberately.
    assert_eq!(size_of::<super::style::StyleData>(), 304);
    eprintln!(
        "sizes: LayoutBox={} StyleData={} ComputedStyle={}",
        size_of::<LayoutBox>(),
        size_of::<super::style::StyleData>(),
        size_of::<bowser_css::ComputedStyle>()
    );
}

#[test]
fn shadow_children_lay_out_as_host_children() {
    // A shadow host renders its shadow tree instead of its light-DOM
    // children (no `<slot>` support: light children are not projected).
    let (mut doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"host\"></div></body></html>".as_slice(),
    );
    let host = doc.get_element_by_id("host").expect("host");
    let frag = doc
        .attach_shadow_root(host, bowser_dom::ShadowRootMode::Open)
        .expect("attach");
    let kid = doc.create_element("span");
    doc.set_attribute(kid, "id", "kid");
    doc.append_child(frag, kid);
    let light = doc.create_element("em");
    doc.set_attribute(light, "id", "light");
    doc.append_child(host, light);
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } #kid { display: block; height: 30px; } #light { display: block; height: 99px; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    // The host's box parents the shadow child's box; the light-DOM
    // child gets no box at all.
    let mut stack = vec![&root];
    let mut host_box: Option<&LayoutBox> = None;
    let mut light_box = false;
    while let Some(b) = stack.pop() {
        if b.node == Some(host) {
            host_box = Some(b);
        }
        if b.node == Some(light) {
            light_box = true;
        }
        stack.extend(b.children.iter());
    }
    let host_box = host_box.expect("host box");
    assert!(
        host_box.children.iter().any(|b| b.node == Some(kid)),
        "shadow child lays out as the host's child"
    );
    assert!(
        !light_box,
        "light-DOM child of a shadow host renders nothing"
    );
    let kid_box = host_box
        .children
        .iter()
        .find(|b| b.node == Some(kid))
        .expect("kid box");
    assert_eq!(kid_box.rect.h, 30.0);
}

#[test]
fn snap_areas_retarget_shadow_descendants_to_host() {
    // A snap-aligned element inside a (closed) shadow root is
    // discovered as a snap area, but the reported target is the host.
    let (mut doc, _) = bowser_dom::Document::parse_html(
        b"<html><body><div id=\"sc\"><div id=\"host\"></div><div id=\"plain\">p</div></div></body></html>"
            .as_slice(),
    );
    let host = doc.get_element_by_id("host").expect("host");
    let plain = doc.get_element_by_id("plain").expect("plain");
    let frag = doc
        .attach_shadow_root(host, bowser_dom::ShadowRootMode::Closed)
        .expect("attach");
    let kid = doc.create_element("div");
    doc.set_attribute(kid, "id", "kid");
    doc.append_child(frag, kid);
    let ss = bowser_css::parse_stylesheet(
        "body { margin: 0; } \
         #sc { overflow-y: auto; height: 100px; } \
         #kid { height: 50px; scroll-snap-align: start; } \
         #plain { height: 50px; scroll-snap-align: start; }",
    );
    let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
    let sc = doc.get_element_by_id("sc").expect("sc");
    let areas = snap_areas_for_container(&root, sc, &doc);
    assert_eq!(areas.len(), 2, "shadow kid and light-DOM plain are areas");
    // Preorder: the shadow kid (inside #host) comes first.
    assert_eq!(areas[0].target, Some(host));
    assert_eq!(areas[1].target, Some(plain));
}

// === replaced-element sequential min/max transfer (CSS 2.1 §10.4/§10.7) ===

#[test]
fn replaced_img_min_width_transfers_height() {
    // `img{width:auto;min-width:200px}` intrinsic 100x50: the min-width
    // violation transfers through the intrinsic ratio -> 200x100.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="p"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".i { display: block; min-width: 200px; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 200.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 100.0), "h {}", img.rect.h);
}

#[test]
fn replaced_img_max_width_transfers_height() {
    // `img{max-width:60px}` intrinsic 100x50 -> 60x30.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="p"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".i { display: block; max-width: 60px; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 60.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 30.0), "h {}", img.rect.h);
}

#[test]
fn replaced_img_min_height_transfers_width() {
    // `img{height:auto;min-height:100px}` intrinsic 100x50: the
    // min-height violation transfers back into the auto width -> 200x100.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="p"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".i { display: block; min-height: 100px; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 200.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 100.0), "h {}", img.rect.h);
}

#[test]
fn replaced_img_max_height_transfers_width() {
    // `img{max-height:20px}` intrinsic 100x50 -> 40x20.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="p"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".i { display: block; max-height: 20px; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 40.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 20.0), "h {}", img.rect.h);
}

#[test]
fn replaced_img_min_wins_over_max_width() {
    // Over-constrained min/max-width: min wins (CSS 2.1 §10.4), then the
    // winning width transfers -> 200x100.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="p"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".i { display: block; min-width: 200px; max-width: 150px; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 200.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 100.0), "h {}", img.rect.h);
}

#[test]
fn replaced_img_min_width_border_box() {
    // Under `box-sizing: border-box` min/max-width bound the border box:
    // `min-width: 200px` with 10px borders -> content 190, height 95.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="p"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".i { display: block; box-sizing: border-box; border: 10px solid black; min-width: 200px; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 200.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 110.0), "h {}", img.rect.h);
}

#[test]
fn replaced_img_specified_width_transfers_height() {
    // CSS 2.1 §10.6.2: a specified width with auto height transfers
    // through the intrinsic ratio -> 200x100, not 200x50.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="p"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".i { display: block; width: 200px; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 200.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 100.0), "h {}", img.rect.h);
}

#[test]
fn replaced_img_specified_height_transfers_width() {
    // CSS 2.1 §10.3.2: an auto width with a definite height transfers
    // through the intrinsic ratio -> 200x100.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="p"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".i { display: block; height: 100px; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 200.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 100.0), "h {}", img.rect.h);
}

#[test]
fn replaced_img_flex_row_min_width_transfer() {
    // Flex §9.7 clamps the main size; the cross size re-derives from the
    // ratio (flexbox §9.4 step 5): min-width:200px on the item -> 200x100.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="f"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".f { display: flex; } .i { display: block; min-width: 200px; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 200.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 100.0), "h {}", img.rect.h);
}

#[test]
fn replaced_img_flex_row_stretch_keeps_ratio() {
    // Aspect-ratio arm takes precedence over stretch (flexbox §9.4 step
    // 5): a stretched replaced item keeps its hypothetical cross size
    // (200px main -> 100px cross), not the 300px line height.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="f"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".f { display: flex; height: 300px; } .i { display: block; width: 200px; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 200.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 100.0), "h {}", img.rect.h);
}

#[test]
fn replaced_img_flex_column_cross_fill_keeps_ratio() {
    // Column flex cross-fill must not distort a replaced item: with a
    // 100px main size the cross derives from the ratio -> 200x100, not
    // the 400px column width.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="f"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".f { display: flex; flex-direction: column; width: 400px; } .i { display: block; height: 100px; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 200.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 100.0), "h {}", img.rect.h);
}

#[test]
fn replaced_img_grid_stretch_keeps_ratio() {
    // Grid stretch must not distort a replaced item: the stretched width
    // (300px area) transfers through the ratio -> 300x150.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="g"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".g { display: grid; grid-template-columns: 300px; } .i { display: block; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 300.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 150.0), "h {}", img.rect.h);
}

#[test]
fn replaced_img_grid_align_stretch_keeps_ratio() {
    // `align-self: stretch` on a replaced grid item: the aspect-ratio
    // arm takes precedence — the item keeps its ratio-derived height
    // (150px from the 300px stretched width), not the 400px row.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="g"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".g { display: grid; grid-template-columns: 300px; grid-template-rows: 400px; } .i { display: block; align-self: stretch; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 300.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 150.0), "h {}", img.rect.h);
}

#[test]
fn replaced_img_abspos_min_width_transfer() {
    // Absolutely positioned replaced boxes go through the same block
    // height path: min-width:200px -> 200x100.
    let (doc, map) = doc_with_img_intrinsic(r#"<div class="p"><img class="i"></div>"#, 100.0, 50.0);
    let l = layout_doc_with_intrinsics(
        &doc,
        ".p { position: relative; } .i { position: absolute; min-width: 200px; }",
        &FixedFontMetrics,
        &map,
    );
    let img = &body(&l).children[0].children[0];
    assert!(approx(img.rect.w, 200.0), "w {}", img.rect.w);
    assert!(approx(img.rect.h, 100.0), "h {}", img.rect.h);
}

#[test]
fn table_pref_width_clamped_by_max_width() {
    // The intrinsic preferred-width arm is clamped by max-width
    // (mirroring the block convention): a max-content parent sizes to
    // the clamped width, not the unclamped content width.
    let l = layout_html(
        "<div style=\"width:max-content\"><table style=\"max-width:100px\">\
         <tr><td><div style=\"width:300px\">x</div></td></tr></table></div>",
        "td { font-size: 10px; }",
    );
    let div = &body(&l).children[0];
    assert!(approx(div.rect.w, 100.0), "div w {}", div.rect.w);
}

#[test]
fn table_shrink_to_fit_keeps_border_padding() {
    // Diagnostic for the generic shrink branch: a floated table's
    // shrink-to-fit content width is the content-box preferred width;
    // the table's own border/padding must not be subtracted twice.
    let l = layout_html(
        "<table style=\"float:left; border:10px solid; padding:5px;\">\
         <tr><td><div style=\"width:100px\">x</div></td></tr></table>",
        "td { font-size: 10px; }",
    );
    let table = &body(&l).children[0];
    assert_eq!(table.box_type, BoxType::Table);
    // Content 100 + border 20 + padding 10 = 130 border-box.
    assert!(approx(table.rect.w, 130.0), "table w {}", table.rect.w);
}

#[test]
fn table_empty_with_border_spacing_has_zero_size() {
    // WPT css/css-tables/tentative/table-minmax.html subtests 11-13: an
    // empty table draws no inter-column spacing.
    for html in [
        "<table style=\"border-spacing: 10px;\"></table>",
        "<table style=\"border-spacing: 10px;\"><tbody></tbody></table>",
    ] {
        let l = layout_html(html, "");
        let table = &body(&l).children[0];
        assert_eq!(table.box_type, BoxType::Table);
        assert!(
            approx(table.rect.w, 0.0),
            "table w {} for {html}",
            table.rect.w
        );
        assert!(
            approx(table.rect.h, 0.0),
            "table h {} for {html}",
            table.rect.h
        );
    }
    // A table with an empty row but no columns: the horizontal
    // zero-column guard zeroes the width; the row may keep height.
    let html = "<table style=\"border-spacing: 10px;\"><tbody><tr></tr></tbody></table>";
    let l = layout_html(html, "");
    let table = &body(&l).children[0];
    assert_eq!(table.box_type, BoxType::Table);
    assert!(
        approx(table.rect.w, 0.0),
        "table w {} for {html}",
        table.rect.w
    );
}

#[test]
fn table_colspan_mixed_pct_fixed() {
    // WPT css-tables/tentative/colspan-redistribution table 23:
    // an auto-width colspan=2 cell (208px min) over a 40% column and
    // a fixed 50px column. The 40% column takes 40% of the 200px
    // distributable (80px); the fixed column takes the rest (120px).
    let l = layout_html(
        "<table style=\"width:1px; border-spacing:8px;\">\
         <tr><td style=\"width:40%\"><div style=\"width:20px\">40%</div></td>\
         <td style=\"width:50px\"><div style=\"width:50px\">50</div></td>\
         <td style=\"width:100px\"><div style=\"width:20px\">20</div></td></tr>\
         <tr><td colspan=\"2\"><div style=\"width:208px\">208px min</div></td></tr></table>",
        "td { font-size: 10px; }",
    );
    let table = &body(&l).children[0];
    assert_eq!(table.box_type, BoxType::Table);
    assert!(approx(table.rect.w, 252.0), "table w {}", table.rect.w);
    assert!(
        approx(table_cell(&l, 0, 0).rect.w, 80.0),
        "c00 w {}",
        table_cell(&l, 0, 0).rect.w
    );
    assert!(
        approx(table_cell(&l, 0, 1).rect.w, 120.0),
        "c01 w {}",
        table_cell(&l, 0, 1).rect.w
    );
    assert!(
        approx(table_cell(&l, 0, 2).rect.w, 20.0),
        "c02 w {}",
        table_cell(&l, 0, 2).rect.w
    );
}

#[test]
fn place_self_center_centers_grid_item() {
    // `place-self: center` sets both align-self and justify-self.
    let l = layout_html(
        "<div style=\"display:grid; width:200px; height:200px;\"><div id=\"i\" style=\"place-self:center; width:50px; height:50px;\">x</div></div>",
        "div { font-size: 10px; }",
    );
    let grid = &body(&l).children[0];
    let item = &grid.children[0];
    // Centered: (200-50)/2 = 75.
    assert!(approx(item.rect.x, 75.0), "x {}", item.rect.x);
    assert!(approx(item.rect.y, 75.0), "y {}", item.rect.y);
}

#[test]
fn place_self_two_values_split_axes() {
    // `place-self: start end` → align-self:start, justify-self:end.
    let l = layout_html(
        "<div style=\"display:grid; width:200px; height:200px;\"><div id=\"i\" style=\"place-self:start end; width:50px; height:50px;\">x</div></div>",
        "div { font-size: 10px; }",
    );
    let grid = &body(&l).children[0];
    let item = &grid.children[0];
    // justify-self:end → x = 150; align-self:start → y = 0.
    assert!(approx(item.rect.x, 150.0), "x {}", item.rect.x);
    assert!(approx(item.rect.y, 0.0), "y {}", item.rect.y);
}

#[test]
fn grid_baseline_align_stretches() {
    // `align-self: baseline` on a grid item without baseline info
    // falls back to stretch (not the container's alignment).
    let l = layout_html(
        "<div style=\"display:grid; width:200px; height:200px;\"><div style=\"align-self:baseline; width:50px;\">x</div></div>",
        "div { font-size: 10px; }",
    );
    let grid = &body(&l).children[0];
    let item = &grid.children[0];
    // Stretched to the row height (200).
    assert!(approx(item.rect.h, 200.0), "h {}", item.rect.h);
}

#[test]
fn flex_self_start_aligns_to_start() {
    // `align-self: self-start` on a flex item aligns to the flex start.
    let l = layout_html(
        "<div style=\"display:flex; width:200px; height:200px; align-items:center;\"><div style=\"align-self:self-start; width:50px; height:50px;\">x</div></div>",
        "div { font-size: 10px; }",
    );
    let flex = &body(&l).children[0];
    let item = &flex.children[0];
    // self-start → y = 0 (not centered).
    assert!(approx(item.rect.y, 0.0), "y {}", item.rect.y);
}

#[test]
fn justify_content_left_packs_start_ltr() {
    // `justify-content: left` in LTR packs at the main start.
    let l = layout_html(
        "<div style=\"display:flex; width:200px; justify-content:left;\"><div style=\"width:50px; height:10px;\">x</div></div>",
        "div { font-size: 10px; }",
    );
    let flex = &body(&l).children[0];
    let item = &flex.children[0];
    assert!(approx(item.rect.x, 0.0), "x {}", item.rect.x);
}

#[test]
fn justify_content_right_packs_end_ltr() {
    // `justify-content: right` in LTR packs at the main end.
    let l = layout_html(
        "<div style=\"display:flex; width:200px; justify-content:right;\"><div style=\"width:50px; height:10px;\">x</div></div>",
        "div { font-size: 10px; }",
    );
    let flex = &body(&l).children[0];
    let item = &flex.children[0];
    assert!(approx(item.rect.x, 150.0), "x {}", item.rect.x);
}

#[test]
fn dir_rtl_flips_flex_justify_left() {
    // `dir="rtl"` maps to `direction: rtl`; `justify-content: left`
    // is physical, so it still packs at the physical left (x=0).
    // In RTL flexbox the main axis is reversed, so physical left
    // is the flex End.
    let l = layout_html(
        "<div dir=\"rtl\" style=\"display:flex; width:200px; justify-content:left;\"><div style=\"width:50px; height:10px;\">x</div></div>",
        "div { font-size: 10px; }",
    );
    let flex = &body(&l).children[0];
    let item = &flex.children[0];
    // Physical left → x = 0 (flex End in RTL).
    assert!(approx(item.rect.x, 0.0), "x {}", item.rect.x);
}

#[test]
fn height_stretch_fills_container() {
    // `height: stretch` fills the containing block's height.
    let l = layout_html(
        "<div style=\"height:200px;\"><div style=\"height:stretch;\">x</div></div>",
        "div { font-size: 10px; }",
    );
    let outer = &body(&l).children[0];
    let inner = &outer.children[0];
    assert!(approx(inner.rect.h, 200.0), "h {}", inner.rect.h);
}

#[test]
fn height_stretch_indefinite_behaves_as_auto() {
    // `height: stretch` with an indefinite CB height behaves as auto.
    let l = layout_html(
        "<div><div style=\"height:stretch; font-size:10px; line-height:10px;\">x</div></div>",
        "",
    );
    let outer = &body(&l).children[0];
    let inner = &outer.children[0];
    // Auto height: content height (10px line).
    assert!(approx(inner.rect.h, 10.0), "h {}", inner.rect.h);
}

/// H1: vertical-rl tables must mirror the block (horizontal) axis, not
/// the inline (vertical) axis. Two columns (60px and 30px inline) in one
/// row: the row spans the full block width, so cells sit at the same
/// inline positions as vertical-lr, but cell *contents* mirror
/// horizontally within the row — the 30px inner div sits at the right
/// (block-start) edge of its cell, not the left.
#[test]
fn vertical_rl_table_mirrors_block_axis() {
    let html = r#"<div class="cb"><table class="t"><tr><td class="c"><div class="inner">x</div></td><td class="c2"><div class="inner">y</div></td></tr></table></div>"#.to_string();
    for (wm, expected_inner_x) in [("vertical-lr", 0.0), ("vertical-rl", 30.0)] {
        let css = format!(
            ".cb {{ width: 400px; }} \
             .t {{ writing-mode: {wm}; border-spacing: 0; }} \
             table, td {{ border: 0; padding: 0; vertical-align: top; }} \
             .c > .inner {{ width: 60px; height: 10px; }} \
             .c2 > .inner {{ width: 30px; height: 10px; }}"
        );
        let l = layout_html(&html, &css);
        let table = &body(&l).children[0].children[0];
        assert_eq!(table.box_type, BoxType::Table);
        // Table is 60px (block) by 20px (inline) in both modes: the
        // columns' inline (vertical) extents are the inner divs'
        // definite heights (10px each), not their widths.
        assert!(approx(table.rect.w, 60.0), "{wm}: table w {}", table.rect.w);
        assert!(approx(table.rect.h, 20.0), "{wm}: table h {}", table.rect.h);
        let group = table
            .children
            .iter()
            .find(|c| c.box_type == BoxType::TableRowGroup)
            .expect("row group");
        let row = &group.children[0];
        let cell2 = &row.children[1];
        assert_eq!(cell2.box_type, BoxType::TableCell);
        // Second column: 60px block by 10px inline, at inline offset 10.
        assert!(approx(cell2.rect.w, 60.0), "{wm}: cell w {}", cell2.rect.w);
        assert!(approx(cell2.rect.h, 10.0), "{wm}: cell h {}", cell2.rect.h);
        let inner = &cell2.children[0];
        // The 30px inner div: full cell height is 10px; its horizontal
        // position mirrors for vertical-rl (block-start is the right).
        assert!(
            approx(inner.rect.x, expected_inner_x),
            "{wm}: inner x {} != {expected_inner_x}",
            inner.rect.x
        );
        assert!(approx(inner.rect.y, 10.0), "{wm}: inner y {}", inner.rect.y);
        assert!(approx(inner.rect.w, 30.0), "{wm}: inner w {}", inner.rect.w);
        assert!(approx(inner.rect.h, 10.0), "{wm}: inner h {}", inner.rect.h);
    }
}

/// H2: §3.10.6 second-pass resume must not double-transpose captions.
/// A vertical table with a cell carrying a specified block size and
/// percentage-height content triggers the measurement/second-pass
/// split; the caption is transposed once (measurement resume) and must
/// not be transposed again (placement resume). It must end in physical
/// space: its width is horizontal, its height vertical.
#[test]
fn vertical_table_second_pass_caption_stays_physical() {
    let html = r#"<div class="cb"><table class="t"><caption class="cap">c</caption><tr><td class="c"><div class="pct">x</div></td></tr></table></div>"#.to_string();
    let css = ".cb { width: 400px; } \
         .t { writing-mode: vertical-rl; border-spacing: 0; } \
         table, td { border: 0; padding: 0; } \
         .cap { width: 80px; height: 12px; } \
         .c { height: 50px; } \
         .c > .pct { height: 50%; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    let cap = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::Block)
        .expect("caption");
    // Physical space: width is horizontal (80px), height is vertical (12px).
    assert!(
        approx(cap.rect.w, 80.0),
        "caption w {} != 80 (double-transposed?)",
        cap.rect.w
    );
    assert!(
        approx(cap.rect.h, 12.0),
        "caption h {} != 12 (double-transposed?)",
        cap.rect.h
    );
}

/// H3: a vertical table's definite physical `width` must feed its
/// preferred width. `width: 200px` on a vertical table in a
/// max-content parent contributes 200px (not ignored).
#[test]
fn vertical_table_definite_width_feeds_pref_width() {
    let html = r#"<div class="cb"><table class="t"><tr><td><div class="inner">x</div></td></tr></table></div>"#.to_string();
    let css = ".cb { width: max-content; } \
         .t { writing-mode: vertical-rl; width: 200px; border-spacing: 0; } \
         table, td { border: 0; padding: 0; } \
         .inner { width: 60px; height: 10px; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    assert!(
        approx(table.rect.w, 200.0),
        "table w {} != 200",
        table.rect.w
    );
}

/// H3: a vertical table's `height` must not leak into its physical
/// preferred width. `height: 200px; width: auto` sizes from content,
/// not 200px.
#[test]
fn vertical_table_height_does_not_feed_pref_width() {
    let html = r#"<div class="cb"><table class="t"><tr><td><div class="inner">x</div></td></tr></table></div>"#.to_string();
    let css = ".cb { width: max-content; } \
         .t { writing-mode: vertical-rl; height: 200px; width: auto; border-spacing: 0; } \
         table, td { border: 0; padding: 0; } \
         .inner { width: 60px; height: 10px; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    // Physical preferred width comes from content (60px block), not the
    // 200px inline (height) spec.
    assert!(
        approx(table.rect.w, 60.0),
        "table w {} != 60 (height leaked into pref width?)",
        table.rect.w
    );
}

/// M3: a vertical table's inline margins come from the physical
/// top/bottom. `margin-top: 20px` pushes the table 20px down (inline).
#[test]
fn vertical_table_margin_top_pushes_inline() {
    let html = r#"<div class="cb"><table class="t"><tr><td><div class="inner">x</div></td></tr></table></div>"#.to_string();
    let css = ".cb { width: 400px; border: 1px solid black; } \
         .t { writing-mode: vertical-rl; margin-top: 20px; border-spacing: 0; } \
         table, td { border: 0; padding: 0; } \
         .inner { width: 60px; height: 10px; }";
    let l = layout_html(&html, css);
    let cb = &body(&l).children[0];
    let table = &cb.children[0];
    assert_eq!(table.box_type, BoxType::Table);
    // Inline is vertical: margin-top shifts the table down by 20px
    // (plus the 1px border preventing margin collapse).
    assert!(
        approx(table.rect.y - cb.rect.y, 21.0),
        "table y {} != 21",
        table.rect.y - cb.rect.y
    );
}

/// M4: a vertical table's block size (physical width) under
/// `border-box` strips the horizontal border/padding, not the vertical.
#[test]
fn vertical_table_border_box_strips_horizontal_bp() {
    let html = r#"<div class="cb"><table class="t"><tr><td><div class="inner">x</div></td></tr></table></div>"#.to_string();
    let css = ".cb { width: 400px; } \
         .t { writing-mode: vertical-rl; width: 100px; box-sizing: border-box; \
              border-left: 10px solid black; padding-right: 5px; border-spacing: 0; } \
         table, td { border: 0; padding: 0; } \
         .inner { width: 60px; height: 10px; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    // Border-box width is 100px (block axis = horizontal).
    assert!(
        approx(table.rect.w, 100.0),
        "table w {} != 100",
        table.rect.w
    );
}

/// M4: a vertical table's percentage block size resolves against the
/// containing-block width.
#[test]
fn vertical_table_pct_width_resolves_against_cb_width() {
    let html = r#"<div class="cb"><table class="t"><tr><td><div class="inner">x</div></td></tr></table></div>"#.to_string();
    let css = ".cb { width: 400px; } \
         .t { writing-mode: vertical-rl; width: 50%; border-spacing: 0; } \
         table, td { border: 0; padding: 0; } \
         .inner { width: 60px; height: 10px; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    // 50% of 400px CB width = 200px physical width.
    assert!(
        approx(table.rect.w, 200.0),
        "table w {} != 200",
        table.rect.w
    );
}

/// M5: a vertical table's percentage min-width resolves against the
/// containing-block width (not height).
#[test]
fn vertical_table_pct_min_width_resolves_against_cb_width() {
    let html = r#"<div class="cb"><table class="t"><tr><td><div class="inner">x</div></td></tr></table></div>"#.to_string();
    let css = ".cb { width: 400px; height: 100px; } \
         .t { writing-mode: vertical-rl; min-width: 50%; border-spacing: 0; } \
         table, td { border: 0; padding: 0; } \
         .inner { width: 60px; height: 10px; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    // 50% of 400px CB width = 200px min physical width.
    assert!(
        approx(table.rect.w, 200.0),
        "table w {} != 200",
        table.rect.w
    );
}

/// M6: `border-spacing: 10px 20px` on a vertical table gives 20px
/// inter-column gaps (columns run vertically = inline axis).
#[test]
fn vertical_table_border_spacing_transposed() {
    let html = r#"<div class="cb"><table class="t"><tr><td><div class="i1">x</div></td><td><div class="i2">y</div></td></tr></table></div>"#.to_string();
    let css = ".cb { width: 400px; } \
         .t { writing-mode: vertical-rl; border-spacing: 10px 20px; } \
         table, td { border: 0; padding: 0; } \
         .i1 { width: 60px; height: 10px; } \
         .i2 { width: 30px; height: 10px; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    let group = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::TableRowGroup)
        .expect("row group");
    let row = &group.children[0];
    let cell1 = &row.children[0];
    let cell2 = &row.children[1];
    // Columns are stacked vertically (inline): with transposed spacing,
    // the inline gaps use the 20px vertical spacing. The columns'
    // inline extents are the inner divs' definite heights (10px each).
    // Table inline size = 20 + 10 + 20 + 10 + 20 = 80.
    assert!(approx(table.rect.h, 80.0), "table h {} != 80", table.rect.h);
    // cell2's inline offset = 20 + 10 + 20 = 50.
    assert!(approx(cell2.rect.h, 10.0), "cell2 h {}", cell2.rect.h);
    // The gap between columns is 20px (vertical spacing), not 10px.
    let cell1_bottom = cell1.rect.y + cell1.rect.h;
    // Note: cell rects are in algorithm space pre-transpose? Check physical.
    // For vertical-rl, physical y = inline. cell1 at y=20 (after leading 20px).
    assert!(
        approx(cell2.rect.y - cell1_bottom, 20.0),
        "inter-column gap {} != 20",
        cell2.rect.y - cell1_bottom
    );
}

/// M7: a vertical table's block-end margin (physical left for
/// vertical-rl) feeds mb_place, spacing the next sibling.
#[test]
fn vertical_rl_table_block_end_margin_spaces_sibling() {
    let html = r#"<div class="cb"><table class="t"><tr><td><div class="inner">x</div></td></tr></table><div class="after">y</div></div>"#.to_string();
    let css = ".cb { width: 400px; } \
         .t { writing-mode: vertical-rl; margin-left: 25px; border-spacing: 0; } \
         table, td { border: 0; padding: 0; } \
         .inner { width: 60px; height: 10px; } \
         .after { height: 10px; }";
    let l = layout_html(&html, css);
    let cb = &body(&l).children[0];
    let table = &cb.children[0];
    let after = &cb.children[1];
    assert_eq!(table.box_type, BoxType::Table);
    // The 25px margin-left (block-end for vertical-rl) should space the
    // following sibling. Table height (inline) is 60; sibling starts at
    // 60 + 25 = 85 (if mb_place is used).
    let gap = after.rect.y - (table.rect.y + table.rect.h);
    assert!(
        approx(gap, 25.0),
        "sibling gap {} != 25 (block-end margin not used?)",
        gap
    );
}

/// L5: a vertical table's column (inline) widths include the cell's
/// vertical padding/border, not the horizontal.
#[test]
fn vertical_table_column_width_uses_vertical_cell_padding() {
    let html = r#"<div class="cb"><table class="t"><tr><td class="c"><div class="inner">x</div></td></tr></table></div>"#.to_string();
    let css = ".cb { width: 400px; } \
         .t { writing-mode: vertical-rl; border-spacing: 0; } \
         table { border: 0; padding: 0; } \
         .c { padding-top: 15px; padding-bottom: 5px; padding-left: 50px; border: 0; } \
         .inner { width: 60px; height: 10px; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    // Column inline width = content 10 (the inner div's vertical inline
    // extent is its definite height) + vertical padding 15 + 5 = 30.
    // (The 50px left padding is block-axis and must not inflate it.)
    assert!(approx(table.rect.h, 30.0), "table h {} != 30", table.rect.h);
}

/// WPT css/css-tables/absolute-tables-001.html (.tbl 5-8): a
/// vertical-lr abspos table's inline-axis spec is the physical
/// `height`, and per CSS 2.1 §10.5 `height: 50%` resolves against the
/// CB *height* (76 = 60 + 5 + 11), not the CB width (80).
#[test]
fn vertical_table_pct_inline_size_resolves_against_cb_height() {
    let html = r#"<div class="cb"><table class="t"><tr><td>x</td></tr></table></div>"#.to_string();
    let css = ".cb { position: relative; border: 5px solid black; height: 60px; width: 60px; padding: 5px 7px 11px 13px; } \
         .t { writing-mode: vertical-lr; position: absolute; left: 0; top: 0; width: 50%; height: 50%; border-spacing: 0; } \
         table, td { border: 0; padding: 0; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0].children[0];
    assert_eq!(table.box_type, BoxType::Table);
    // Physical height is the inline axis: 50% of the 76px CB height.
    assert!(
        approx(table.rect.h, 38.0),
        "table h {} != 38 (pct base was CB width)",
        table.rect.h
    );
    // Physical width is the block axis: 50% of the 80px CB width.
    assert!(approx(table.rect.w, 40.0), "table w {} != 40", table.rect.w);
}

/// WPT css/css-tables/caption-orthogonal-block-start-margin.html: the
/// table is vertical-rl, so its block-start edge is its physical right
/// edge and the caption is offset by its 40px physical right margin —
/// not its 7px top margin.
#[test]
fn vertical_rl_table_caption_uses_block_axis_margins() {
    let html = r#"<table class="t"><caption>c</caption><tr><td></td></tr></table>"#.to_string();
    let css = ".t { writing-mode: vertical-rl; border-spacing: 0; } \
         caption { margin: 7px 40px 3px 0; } \
         td { width: 50px; height: 30px; padding: 0; } \
         table, td { border: 0; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0];
    assert_eq!(table.box_type, BoxType::Table);
    let cap = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::Block)
        .expect("caption");
    let gap = (table.rect.x + table.rect.w) - (cap.rect.x + cap.rect.w);
    assert!(
        approx(gap, 40.0),
        "caption block-start gap {} != 40 (used top margin)",
        gap
    );
}

#[test]
fn table_specified_width_is_border_box() {
    // WPT css/css-tables/column-track-merging.html (td_auto_width):
    // a `table` element's specified `width` names the border box
    // (HTML rendering: `table { box-sizing: border-box }` in the UA
    // sheet), so a 400px table with 10px borders has a 400px border
    // box, not 420px.
    let l = layout_html(
        "<table class=\"t\"><caption>c</caption><tr><td colspan=\"10\">x</td><td></td></tr><tr><td colspan=\"10\">y</td><td></td></tr></table>",
        "table.t { border: 10px solid gray; border-spacing: 20px; width: 400px; } td { width: 50px; height: 50px; padding: 0; }",
    );
    let table = &body(&l).children[0];
    assert!(approx(table.rect.w, 400.0), "table.w={}", table.rect.w);
}

#[test]
fn track_merge_spanning_cell_does_not_constrain_merged_column() {
    // WPT css/css-tables/column-track-merging.html (td_auto_width):
    // the `colspan=10` cell's 50px specified width applied to the
    // 10-track span, so after merging it does not constrain the
    // single merged track: the track absorbs the table's excess
    // width (270px) while the truly single-column cell stays at 50px.
    let l = layout_html(
        "<table class=\"t\"><caption>c</caption><tr><td colspan=\"10\">x</td><td></td></tr><tr><td colspan=\"10\">y</td><td></td></tr></table>",
        "table.t { border: 10px solid gray; border-spacing: 20px; width: 400px; } td { width: 50px; height: 50px; padding: 0; }",
    );
    let wide = table_cell(&l, 0, 0);
    let single = table_cell(&l, 0, 1);
    assert!(approx(wide.rect.w, 270.0), "wide.w={}", wide.rect.w);
    assert!(approx(single.rect.w, 50.0), "single.w={}", single.rect.w);
}

#[test]
fn table_element_defaults_to_border_box_sizing() {
    // WPT css/css-tables/tentative/td-box-sizing-002.html: a `table`
    // element's `width`/`height` name the border box by default (UA
    // `table { box-sizing: border-box }`); an explicit
    // `box-sizing: content-box` opts back out. (A `div` with
    // `display: table` stays content-box by default.)
    let css = ".t { border-top-width: 20px; border-bottom-width: 40px; border-left-width: 20px; border-right-width: 40px; border-style: solid; border-color: gray; padding: 10px; border-spacing: 0; width: 100px; height: 100px; } td { padding: 0; }";
    let l = layout_html(
        "<table class=\"t\"><tbody><tr><td></td></tr></tbody></table>",
        css,
    );
    let table = &body(&l).children[0];
    assert!(approx(table.rect.w, 100.0), "table.w={}", table.rect.w);
    assert!(approx(table.rect.h, 100.0), "table.h={}", table.rect.h);
    let l = layout_html(
        "<table class=\"t\"><tbody><tr><td></td></tr></tbody></table>",
        &format!("{css} .t {{ box-sizing: content-box; }}"),
    );
    let table = &body(&l).children[0];
    assert!(approx(table.rect.w, 180.0), "table.w={}", table.rect.w);
    assert!(approx(table.rect.h, 180.0), "table.h={}", table.rect.h);
}

// ---- `::before` / `::after` generated content (see `crate::pseudo`) ----

/// First element with the given tag name in DFS order from `root`.
fn find_tag(doc: &bowser_dom::Document, root: bowser_dom::NodeId, tag: &str) -> bowser_dom::NodeId {
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if matches!(&doc.kind(n), bowser_dom::NodeKind::Element(d) if d.tag_name == tag) {
            return n;
        }
        let mut kids = doc.children(n);
        kids.reverse();
        stack.extend(kids);
    }
    panic!("no <{tag}> found");
}

/// The owned computed style of the first `<tag>` in `html`, cascaded
/// with `css`. The `Styler` itself cannot escape (it borrows the
/// document and sheets), but the finished style can.
fn styled_for(html: &str, css: &str, tag: &str) -> bowser_css::ComputedStyle {
    let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
    let ua = bowser_css::default_ua_stylesheet();
    let ss = bowser_css::parse_stylesheet(css);
    let binding = [&ss];
    let styler = bowser_css::Styler::new(
        &doc,
        &binding,
        &ua,
        bowser_css::Viewport { w: VP.w, h: VP.h },
    );
    let node = find_tag(&doc, doc.document_element().expect("root"), tag);
    styler.style_for(node)
}

/// A fresh [`Forest`] holding one block-level element box for `style`,
/// as `pseudo::attach` expects (element box built, no children yet).
fn pseudo_forest(style: &bowser_css::ComputedStyle) -> (super::boxes::Forest, usize) {
    let display = super::style::display_of(style);
    let data = super::style::StyleData::from_style(style, display, 16.0, 16.0, VP.w, VP.h);
    let mut forest = super::boxes::Forest {
        nodes: Vec::new(),
        root_font_size: 16.0,
        vp_w: VP.w,
        vp_h: VP.h,
        abs_items: Vec::new(),
        gutter_propagated_root: None,
        root_is_html: false,
        quirks_mode: bowser_dom::QuirksMode::NoQuirks,
        #[cfg(test)]
        union_frag_visits: 0,
    };
    let elem = forest.add_node(
        super::BoxType::Block,
        Some(std::rc::Rc::new(style.clone())),
        data,
        None,
        None,
    );
    (forest, elem)
}

#[test]
fn generated_content_gate_flags_before_after_rules() {
    let (doc, _) = bowser_dom::Document::parse_html(b"<div></div>");
    let ua = bowser_css::default_ua_stylesheet();
    let vp = bowser_css::Viewport { w: VP.w, h: VP.h };
    let with = bowser_css::parse_stylesheet("div::after { content: \"c\"; }");
    let with_before = bowser_css::parse_stylesheet("div::before { content: \"c\"; }");
    let without = bowser_css::parse_stylesheet("div { color: red; }");
    assert!(bowser_css::Styler::new(&doc, &[&with], &ua, vp).has_generated_content_rules());
    assert!(bowser_css::Styler::new(&doc, &[&with_before], &ua, vp).has_generated_content_rules());
    assert!(!bowser_css::Styler::new(&doc, &[&without], &ua, vp).has_generated_content_rules());
}

#[test]
fn pseudo_after_appends_inline_text_box() {
    // `div::after { content: "c" }`: the pseudo box appends after the
    // element's own children — an Inline wrapper carrying the pseudo
    // style, holding one Text("c") child. The pseudo style here is the
    // element's style cloned, exactly the merge shape
    // `Styler::pseudo_style_for` produces for a content-only rule
    // (display reset to `inline`).
    let style = styled_for(
        "<div class=\"a\">x</div>",
        "div::after { content: \"c\"; }",
        "div",
    );
    let (mut forest, elem) = pseudo_forest(&style);
    // One pre-existing child, so append order is observable.
    let text_data = super::style::StyleData::inherit(&forest.nodes[elem].data);
    let first = forest.add_node(
        super::BoxType::Text,
        None,
        text_data,
        Some(std::sync::Arc::from("x")),
        Some(elem),
    );
    super::pseudo::attach(
        &mut forest,
        elem,
        super::pseudo::PseudoKind::After,
        super::pseudo::PseudoContent {
            style: std::rc::Rc::new(style.clone()),
            text: "c".to_owned(),
        },
        &style,
    );
    let kids = &forest.nodes[elem].children;
    assert_eq!(kids.len(), 2);
    assert_eq!(kids[0], first);
    let pi = kids[1];
    assert_eq!(forest.nodes[pi].box_type, super::BoxType::Inline);
    assert!(forest.nodes[pi].style.is_some());
    let grand = &forest.nodes[pi].children;
    assert_eq!(grand.len(), 1);
    let t = &forest.nodes[grand[0]];
    assert_eq!(t.box_type, super::BoxType::Text);
    assert_eq!(t.text.as_deref(), Some("c"));
}

#[test]
fn pseudo_before_prepends_inline_text_box() {
    // `div::before { content: "c" }`: same shape, prepended.
    let style = styled_for(
        "<div class=\"a\">x</div>",
        "div::before { content: \"c\"; }",
        "div",
    );
    let (mut forest, elem) = pseudo_forest(&style);
    let text_data = super::style::StyleData::inherit(&forest.nodes[elem].data);
    let first = forest.add_node(
        super::BoxType::Text,
        None,
        text_data,
        Some(std::sync::Arc::from("x")),
        Some(elem),
    );
    super::pseudo::attach(
        &mut forest,
        elem,
        super::pseudo::PseudoKind::Before,
        super::pseudo::PseudoContent {
            style: std::rc::Rc::new(style.clone()),
            text: "c".to_owned(),
        },
        &style,
    );
    let kids = &forest.nodes[elem].children;
    assert_eq!(kids.len(), 2);
    let pi = kids[0];
    assert_eq!(kids[1], first);
    assert_eq!(forest.nodes[pi].box_type, super::BoxType::Inline);
    let grand = &forest.nodes[pi].children;
    assert_eq!(grand.len(), 1);
    assert_eq!(forest.nodes[grand[0]].text.as_deref(), Some("c"));
}

#[test]
fn pseudo_table_caption_display_maps_to_block() {
    // A pseudo whose cascaded `display` is `table-caption` maps to a
    // Block whose stored style keeps the raw `table-caption` string, so
    // `is_table_caption` collects it with no table-layout changes.
    // Pinned through the real pipeline (`Styler::pseudo_style_for`
    // over `BEFORE_AFTER_ATOMS`): the `display: table-caption` winner
    // must survive the pseudo's `inline` reset.
    let (doc, _) = bowser_dom::Document::parse_html(b"<table></table>");
    let ua = bowser_css::default_ua_stylesheet();
    let ss =
        bowser_css::parse_stylesheet("table::after { content: \"x\"; display: table-caption; }");
    let binding = [&ss];
    let styler = bowser_css::Styler::new(
        &doc,
        &binding,
        &ua,
        bowser_css::Viewport { w: VP.w, h: VP.h },
    );
    let table = find_tag(&doc, doc.document_element().expect("root"), "table");
    let origin = styler.style_for(table);
    let pseudo = styler
        .pseudo_style_for(table, "after", bowser_css::BEFORE_AFTER_ATOMS)
        .expect("::after style");
    assert_eq!(
        bowser_css::content_text(&pseudo),
        Some("x"),
        "content winner must apply through pseudo_style_for"
    );
    assert_eq!(
        pseudo.get_str_by_index(bowser_css::props::DISPLAY),
        Some("table-caption")
    );
    let (mut forest, elem) = pseudo_forest(&origin);
    super::pseudo::attach(
        &mut forest,
        elem,
        super::pseudo::PseudoKind::After,
        super::pseudo::PseudoContent {
            style: std::rc::Rc::new(pseudo),
            text: "x".to_owned(),
        },
        &origin,
    );
    let kids = &forest.nodes[elem].children;
    assert_eq!(kids.len(), 1);
    let pi = kids[0];
    assert_eq!(forest.nodes[pi].box_type, super::BoxType::Block);
    assert_eq!(
        forest.nodes[pi]
            .style
            .as_ref()
            .and_then(|s| s.get_str_by_index(bowser_css::props::DISPLAY)),
        Some("table-caption")
    );
}

#[test]
fn pseudo_display_none_suppresses_box() {
    // `display: none` on the pseudo style: no box at all.
    let origin = styled_for("<div class=\"a\"></div>", "", "div");
    let none_style = styled_for("<div class=\"a\"></div>", ".a { display: none; }", "div");
    let (mut forest, elem) = pseudo_forest(&origin);
    super::pseudo::attach(
        &mut forest,
        elem,
        super::pseudo::PseudoKind::After,
        super::pseudo::PseudoContent {
            style: std::rc::Rc::new(none_style),
            text: "c".to_owned(),
        },
        &origin,
    );
    assert!(forest.nodes[elem].children.is_empty());
}

#[test]
fn display_none_skips_pseudo_synthesis() {
    // `display: none` elements `continue` before the pseudo hook, so a
    // `::after` rule must not resurrect them. Exercises the hook
    // placement through the full pipeline (`query_pseudo_content` ->
    // `Styler::pseudo_style_for` -> `content_text`).
    let l = layout_html(
        "<div class=\"a\"></div><div class=\"b\"></div>",
        ".a { display: none; } .a::after { content: \"c\"; } .b { height: 5px; }",
    );
    let kids = &body(&l).children;
    assert_eq!(kids.len(), 1);
    assert_eq!(kids[0].box_type, BoxType::Block);
}

#[test]
fn generated_content_after_end_to_end_order() {
    // Full pipeline: `div::after { content: "c" }` synthesizes an Inline
    // pseudo box holding one Text("c") child, appended after the
    // element's own content. Exercises the real `query_pseudo_content`
    // -> `Styler::pseudo_style_for` + `content_text` path, not just the
    // `attach` shape.
    let l = layout_html("<div class=\"a\">x</div>", ".a::after { content: \"c\"; }");
    let div = &body(&l).children[0];
    assert_eq!(div.box_type, BoxType::Block);
    let kids = &div.children;
    assert_eq!(kids.len(), 2, "expected [text, pseudo], got {kids:?}");
    assert_eq!(kids[0].box_type, BoxType::Text);
    assert_eq!(kids[0].text.as_deref(), Some("x"));
    let pseudo = &kids[1];
    assert_eq!(pseudo.box_type, BoxType::Inline);
    assert!(pseudo.style.is_some(), "pseudo keeps its computed style");
    assert_eq!(pseudo.children.len(), 1);
    let t = &pseudo.children[0];
    assert_eq!(t.box_type, BoxType::Text);
    assert_eq!(t.text.as_deref(), Some("c"));
}

#[test]
fn generated_content_before_end_to_end_prepend() {
    // Same shape for `::before`, prepended before the element's own
    // content.
    let l = layout_html("<div class=\"a\">x</div>", ".a::before { content: \"b\"; }");
    let div = &body(&l).children[0];
    let kids = &div.children;
    assert_eq!(kids.len(), 2, "expected [pseudo, text], got {kids:?}");
    let pseudo = &kids[0];
    assert_eq!(pseudo.box_type, BoxType::Inline);
    assert_eq!(pseudo.children.len(), 1);
    assert_eq!(pseudo.children[0].box_type, BoxType::Text);
    assert_eq!(pseudo.children[0].text.as_deref(), Some("b"));
    assert_eq!(kids[1].box_type, BoxType::Text);
    assert_eq!(kids[1].text.as_deref(), Some("x"));
}

/// Query the real `::before`/`::after` for `tag` through the cascade
/// ([`Styler::pseudo_style_for_shared`] + `resolve_content_text`),
/// returning the pseudo content plus the originating element's
/// computed style.
fn pseudo_query(
    html: &str,
    css: &str,
    tag: &str,
    kind: super::pseudo::PseudoKind,
) -> (super::pseudo::PseudoContent, bowser_css::ComputedStyle) {
    let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
    let ua = bowser_css::default_ua_stylesheet();
    let ss = bowser_css::parse_stylesheet(css);
    let binding = [&ss];
    let styler = bowser_css::Styler::new(
        &doc,
        &binding,
        &ua,
        bowser_css::Viewport { w: VP.w, h: VP.h },
    );
    let node = find_tag(&doc, doc.document_element().expect("root"), tag);
    let content = super::pseudo::query_pseudo_content(&styler, &doc, node, kind, &[])
        .expect("pseudo content expected");
    (content, styler.style_for(node))
}

/// Attach `content` as `kind` to a fresh forest holding one element box
/// for `origin`, returning the forest and the pseudo wrapper's index.
fn pseudo_attach_unit(
    origin: &bowser_css::ComputedStyle,
    kind: super::pseudo::PseudoKind,
    content: super::pseudo::PseudoContent,
) -> (super::boxes::Forest, usize) {
    let (mut forest, elem) = pseudo_forest(origin);
    super::pseudo::attach(&mut forest, elem, kind, content, origin);
    let pi = *forest.nodes[elem].children.last().expect("pseudo attached");
    (forest, pi)
}

#[test]
fn generated_content_collapses_whitespace_normal() {
    // `div::before { content: "a  b" }` under `white-space: normal`:
    // generated text collapses like DOM text ("a b"), not verbatim.
    let l = layout_html(
        "<div class=\"a\">x</div>",
        ".a::before { content: \"a  b\"; }",
    );
    let div = &body(&l).children[0];
    let pseudo = &div.children[0];
    assert_eq!(pseudo.box_type, BoxType::Inline);
    assert_eq!(pseudo.children.len(), 1);
    assert_eq!(pseudo.children[0].text.as_deref(), Some("a b"));
}

#[test]
fn generated_content_pseudo_own_white_space_pre() {
    // `white-space: pre` on the pseudo itself preserves the runs.
    let l = layout_html(
        "<div class=\"a\">x</div>",
        ".a::before { content: \"a  b\"; white-space: pre; }",
    );
    let div = &body(&l).children[0];
    let pseudo = &div.children[0];
    assert_eq!(pseudo.box_type, BoxType::Inline);
    assert_eq!(pseudo.children.len(), 1);
    assert_eq!(pseudo.children[0].text.as_deref(), Some("a  b"));
}

#[test]
fn generated_content_pseudo_own_text_transform_uppercase() {
    // `text-transform: uppercase` on the pseudo applies to generated
    // text (the transform used at collapse time is the pseudo's own).
    let l = layout_html(
        "<div class=\"a\">x</div>",
        ".a::before { content: \"ab\"; text-transform: uppercase; }",
    );
    let div = &body(&l).children[0];
    let pseudo = &div.children[0];
    assert_eq!(pseudo.children.len(), 1);
    assert_eq!(pseudo.children[0].text.as_deref(), Some("AB"));
}

#[test]
fn pseudo_collapses_segment_break_to_space() {
    // A segment break in generated content collapses to one space
    // under `white-space: normal` (mirrors DOM text).
    let origin = styled_for("<div class=\"a\"></div>", "", "div");
    let (forest, pi) = pseudo_attach_unit(
        &origin,
        super::pseudo::PseudoKind::Before,
        super::pseudo::PseudoContent {
            style: std::rc::Rc::new(origin.clone()),
            text: "a\nb".to_owned(),
        },
    );
    let grand = &forest.nodes[pi].children;
    assert_eq!(grand.len(), 1);
    assert_eq!(forest.nodes[grand[0]].text.as_deref(), Some("a b"));
}

#[test]
fn pseudo_empty_content_keeps_childless_wrapper() {
    // `content: ""` collapses to nothing, but the Inline wrapper stays
    // (the spec's empty-inline-box), carrying the pseudo style.
    let l = layout_html("<div class=\"a\">x</div>", ".a::before { content: \"\"; }");
    let div = &body(&l).children[0];
    let pseudo = &div.children[0];
    assert_eq!(pseudo.box_type, BoxType::Inline);
    assert!(pseudo.style.is_some());
    assert!(pseudo.children.is_empty());
}

#[test]
fn pseudo_attr_resolves_from_origin_attributes() {
    // `attr()` reads the originating element's attribute.
    let (content, _) = pseudo_query(
        "<div data-label=\"hi\"></div>",
        "div::before { content: attr(data-label); }",
        "div",
        super::pseudo::PseudoKind::Before,
    );
    assert_eq!(content.text, "hi");
}

#[test]
fn pseudo_attr_fallback_and_missing() {
    // A missing attribute renders the fallback; with no fallback it
    // contributes the empty string (the box still generates).
    let (content, _) = pseudo_query(
        "<div></div>",
        "div::before { content: attr(data-missing, \"fb\"); }",
        "div",
        super::pseudo::PseudoKind::Before,
    );
    assert_eq!(content.text, "fb");
    let (content, _) = pseudo_query(
        "<div></div>",
        "div::before { content: attr(data-missing); }",
        "div",
        super::pseudo::PseudoKind::Before,
    );
    assert_eq!(content.text, "");
    // A multi-item fallback resolves item by item.
    let (content, _) = pseudo_query(
        "<div></div>",
        "div::before { content: attr(data-missing, \"a\" \"b\"); }",
        "div",
        super::pseudo::PseudoKind::Before,
    );
    assert_eq!(content.text, "ab");
}

#[test]
fn pseudo_mixed_content_concatenates() {
    // Strings and `attr()` interleave in one `<content-list>`.
    let (content, _) = pseudo_query(
        "<div data-x=\"1\"></div>",
        "div::after { content: \"[\" attr(data-x) \"]\"; }",
        "div",
        super::pseudo::PseudoKind::After,
    );
    assert_eq!(content.text, "[1]");
}

/// Every `<li>` in `html`, in document order.
fn all_tags(doc: &bowser_dom::Document, tag: &str) -> Vec<bowser_dom::NodeId> {
    let root = doc.document_element().expect("root");
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if matches!(&doc.kind(n), bowser_dom::NodeKind::Element(d) if d.tag_name == tag) {
            out.push(n);
        }
        let mut kids = doc.children(n);
        kids.reverse();
        stack.extend(kids);
    }
    out
}

#[test]
fn pseudo_counter_resolves_with_scoping() {
    // CSS Lists 3 counters: the `ol` resets, each `li` increments, and
    // the `li`'s own increment is visible to its `::before` — and to
    // the *next* `li` (sibling increments share the scope frame).
    let html = "<ol><li>a</li><li>b</li></ol>";
    let css = "ol { counter-reset: sec; } li { counter-increment: sec; } \
               li::before { content: counter(sec) \". \"; }";
    let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
    let ua = bowser_css::default_ua_stylesheet();
    let ss = bowser_css::parse_stylesheet(css);
    let binding = [&ss];
    let styler = bowser_css::Styler::new(
        &doc,
        &binding,
        &ua,
        bowser_css::Viewport { w: VP.w, h: VP.h },
    );
    assert!(styler.has_counter_adjustments());
    let ol = find_tag(&doc, doc.document_element().expect("root"), "ol");
    let lis = all_tags(&doc, "li");
    assert_eq!(lis.len(), 2);
    // Drive the scope stack in document order, like
    // `build_box_tree_rooted` does.
    let mut scopes = super::pseudo::CounterScopes::new();
    let ol_style = styler.style_for(ol);
    scopes.enter(&doc, ol, &ol_style);
    let texts: Vec<String> = lis
        .iter()
        .map(|&li| {
            let li_style = styler.style_for(li);
            let snapshot = scopes.enter(&doc, li, &li_style);
            super::pseudo::query_pseudo_content(
                &styler,
                &doc,
                li,
                super::pseudo::PseudoKind::Before,
                &snapshot,
            )
            .expect("::before content")
            .text
        })
        .collect();
    assert_eq!(texts, vec!["1. ".to_string(), "2. ".to_string()]);
}

#[test]
fn pseudo_counters_joins_nested_scopes() {
    // `counters()` joins every in-scope value, outermost first; the
    // inner `ol`'s reset nests inside (not clobbering) the outer one.
    let html = "<ol><li>a<ol><li>b</li></ol></li></ol>";
    let css = "ol { counter-reset: sec; } li { counter-increment: sec; } \
               li::before { content: counters(sec, \".\", decimal) \" \"; }";
    let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
    let ua = bowser_css::default_ua_stylesheet();
    let ss = bowser_css::parse_stylesheet(css);
    let binding = [&ss];
    let styler = bowser_css::Styler::new(
        &doc,
        &binding,
        &ua,
        bowser_css::Viewport { w: VP.w, h: VP.h },
    );
    // Document-order walk entering every element, exactly like the
    // box-construction traversal drives `CounterScopes::enter`.
    let mut stack = vec![doc.document_element().expect("root")];
    let mut order = Vec::new();
    while let Some(n) = stack.pop() {
        order.push(n);
        let mut kids = doc.children(n);
        kids.reverse();
        stack.extend(kids);
    }
    let mut scopes = super::pseudo::CounterScopes::new();
    let mut texts = Vec::new();
    for n in order {
        let is_li = matches!(
            &doc.kind(n),
            bowser_dom::NodeKind::Element(d) if d.tag_name == "li"
        );
        if !matches!(&doc.kind(n), bowser_dom::NodeKind::Element(_)) {
            continue;
        }
        let style = styler.style_for(n);
        let snapshot = scopes.enter(&doc, n, &style);
        if is_li {
            texts.push(
                super::pseudo::query_pseudo_content(
                    &styler,
                    &doc,
                    n,
                    super::pseudo::PseudoKind::Before,
                    &snapshot,
                )
                .expect("::before content")
                .text,
            );
        }
    }
    assert_eq!(texts, vec!["1 ".to_string(), "1.1 ".to_string()]);
}

#[test]
fn pseudo_counter_styles_render() {
    // Counter styles: `upper-roman`, and a missing counter reads as 0.
    let (content, _) = pseudo_query(
        "<div></div>",
        "div { counter-reset: n 3; } div::before { content: counter(n, upper-roman); }",
        "div",
        super::pseudo::PseudoKind::Before,
    );
    assert_eq!(content.text, "III");
    let (content, _) = pseudo_query(
        "<div></div>",
        "div::before { content: counter(nope); }",
        "div",
        super::pseudo::PseudoKind::Before,
    );
    assert_eq!(content.text, "0");
}

#[test]
fn pseudo_counter_end_to_end_in_box_tree() {
    // The resolved counter text reaches the synthesized text box.
    let l = layout_html(
        "<ol><li>x</li></ol>",
        "ol { counter-reset: n; } li { counter-increment: n; } li::before { content: counter(n) \". \"; }",
    );
    let ol = &body(&l).children[0];
    let li = &ol.children[0];
    // `::before` prepends: first child is the pseudo Inline wrapper.
    let pseudo = &li.children[0];
    assert_eq!(pseudo.box_type, BoxType::Inline);
    assert_eq!(pseudo.children.len(), 1);
    assert_eq!(pseudo.children[0].text.as_deref(), Some("1. "));
}

#[test]
fn pseudo_own_font_size_line_height_apply() {
    // `::before { font-size: 30px; line-height: 2 }` resolves against
    // the origin sizes instead of inheriting them wholesale.
    let (content, origin) = pseudo_query(
        "<div class=\"a\"></div>",
        ".a::before { content: \"x\"; font-size: 30px; line-height: 2; }",
        "div",
        super::pseudo::PseudoKind::Before,
    );
    let (forest, pi) = pseudo_attach_unit(&origin, super::pseudo::PseudoKind::Before, content);
    let data = &forest.nodes[pi].data;
    assert!(
        (data.font_size - 30.0).abs() < 0.01,
        "font_size = {}",
        data.font_size
    );
    assert!(
        (data.line_height - 60.0).abs() < 0.01,
        "line_height = {}",
        data.line_height
    );
}

#[test]
fn pseudo_own_white_space_direction_apply() {
    // `white-space` and `direction` winners on the pseudo reach its
    // StyleData (origin is ltr / normal here).
    let (content, origin) = pseudo_query(
        "<div class=\"a\"></div>",
        ".a::before { content: \"x\"; white-space: pre; direction: rtl; }",
        "div",
        super::pseudo::PseudoKind::Before,
    );
    assert!(
        !origin
            .get_str_by_index(bowser_css::props::DIRECTION)
            .is_some_and(|d| d.eq_ignore_ascii_case("rtl"))
    );
    let (forest, pi) = pseudo_attach_unit(&origin, super::pseudo::PseudoKind::Before, content);
    let data = &forest.nodes[pi].data;
    assert_eq!(data.white_space, super::style::WhiteSpace::Pre);
    assert!(data.rtl, "pseudo direction: rtl must win");
}

#[test]
fn generated_content_textarea_select_skip_pseudos() {
    // `textarea` and `select` are replaced: no `::before` box inside
    // them, while a `div` in the same document still gets one. The two
    // form controls lay out as childless inline-blocks inside one
    // anonymous block (their UA display is inline-level).
    let l = layout_html(
        "<div class=\"a\"></div><textarea></textarea><select></select>",
        ".a::before { content: \"x\"; } textarea::before { content: \"x\"; } select::before { content: \"x\"; }",
    );
    let kids = &body(&l).children;
    assert_eq!(kids.len(), 2, "expected [div, anon-block], got {kids:?}");
    let has_pseudo = |b: &super::LayoutBox| {
        b.children.iter().any(|c| {
            c.box_type == super::BoxType::Inline
                && c.children.iter().any(|g| g.text.as_deref() == Some("x"))
        })
    };
    assert!(has_pseudo(&kids[0]), "div keeps its ::before");
    let anon = &kids[1];
    assert_eq!(anon.box_type, super::BoxType::AnonymousBlock);
    assert_eq!(anon.children.len(), 2);
    for control in &anon.children {
        assert_eq!(control.box_type, super::BoxType::InlineBlock);
        assert!(
            control.children.is_empty(),
            "replaced form control must not synthesize ::before"
        );
    }
}

#[test]
fn generated_content_none_renders_no_box() {
    // `content: none` (and the `normal` initial) is not renderable:
    // the rule matches and the gate fires, but no box is synthesized.
    let l = layout_html(
        "<div class=\"a\">x</div>",
        ".a::before { content: \"y\"; } .a::after { content: none; }",
    );
    let div = &body(&l).children[0];
    let kids = &div.children;
    assert_eq!(
        kids.len(),
        2,
        "expected [before-pseudo, text], got {kids:?}"
    );
    assert_eq!(kids[0].box_type, BoxType::Inline);
    assert_eq!(kids[0].children[0].text.as_deref(), Some("y"));
    assert_eq!(kids[1].box_type, BoxType::Text);
    assert_eq!(kids[1].text.as_deref(), Some("x"));
}

#[test]
fn wpt_caption_table7_after_text_removes_strut_descent() {
    // WPT css/css-tables/tentative/caption.html table 7:
    // <caption style="height:80%"><div style="height:30px"></div></caption>
    // with `main caption > div { display: inline-block; }` and
    // `main caption > div:after { content: "c"; }`. Expected caption
    // height 30: the ::after text gives the empty inline-block a text
    // baseline, so the 3.84px strut descent no longer inflates the line
    // box (without the rule the caption measures 33.84).
    let l = layout_html(
        "<main><table style=\"width:200px;height:100px\"><caption style=\"height:80%\"><div style=\"height:30px\"></div></caption><tr><td>auto</td></tr></table></main>",
        "main caption > div { display: inline-block; } main caption > div:after { content: \"c\"; }",
    );
    let table = &body(&l).children[0];
    // Find the caption: the Block box inside the Table box (the table
    // element may produce a Block wrapper around the inner Table).
    fn find_table(b: &super::LayoutBox) -> Option<&super::LayoutBox> {
        if b.box_type == super::BoxType::Table {
            return Some(b);
        }
        b.children.iter().find_map(find_table)
    }
    let inner = find_table(table).expect("inner table box");
    let caption = inner
        .children
        .iter()
        .find(|c| c.box_type == super::BoxType::Block)
        .expect("caption box");
    assert!(
        approx(caption.rect.h, 30.0),
        "caption h {} != 30",
        caption.rect.h
    );
}

#[test]
fn colgroup_width_spreads_to_auto_cols() {
    // WPT css/css-tables/tentative/colgroup-col.html table 3: a
    // `colgroup`'s CSS width applies at full value to each covered
    // column without its own `col` width (100px, not 100/3), flooring
    // the minimum only (the maxima floor was removed — a plain
    // `<col>` floors the maximum, a group spread never does).
    let l = layout_html(
        "<table><colgroup span=\"4\" style=\"width:100px\"><col><col style=\"width:50px\"><col style=\"width:150px\"></colgroup><tr><td>cg</td><td>col1</td><td>col2</td><td><div style=\"width:50px\">x</div></td></tr></table>",
        "td { padding: 0; }",
    );
    let widths: Vec<f32> = (0..4).map(|c| table_cell(&l, 0, c).rect.w).collect();
    assert!(approx(widths[0], 100.0), "widths={widths:?}");
    assert!(approx(widths[1], 50.0), "widths={widths:?}");
    assert!(approx(widths[2], 150.0), "widths={widths:?}");
    assert!(approx(widths[3], 50.0), "widths={widths:?}");
}

#[test]
fn colgroup_floor_honored_in_interpolation_branch() {
    // The `<colgroup>` hard floor (100px) must survive the
    // interpolation branch, not just the excess branch: with a 150px
    // table the assignable width falls between the guesses, and the
    // group-floored column must land on its 100px minimum rather than
    // its 9.6px max-content (the floor is baked into guesses 3 and 4,
    // which also fixes the branch decision).
    let words = "word ".repeat(52);
    let html = format!(
        "<table style=\"width:150px;border-collapse:collapse\"><colgroup style=\"width:100px\"><col></colgroup><col><tr><td>x</td><td>{words}</td></tr></table>"
    );
    let l = layout_html(&html, "td { padding: 0; }");
    let w0 = table_cell(&l, 0, 0).rect.w;
    let w1 = table_cell(&l, 0, 1).rect.w;
    assert!(approx(w0, 100.0), "col0 {w0} != 100");
    assert!(approx(w1, 50.0), "col1 {w1} != 50");
}

#[test]
fn auto_column_maxima_min_width_not_double_padded() {
    // `cell_min_width_bb` is border-box: the cell padding must not be
    // added again when the min-width dominates the max-content
    // measure. 100px border-box min-width + 20px padding on col0 and
    // 100px of content on col1 in a 300px table split 150/150 (the
    // double-count gave col0 163.6).
    let l = layout_html(
        "<table style=\"width:300px;border-collapse:collapse\"><tr><td style=\"min-width:100px;padding:10px;box-sizing:border-box\">x</td><td><div style=\"width:100px\">y</div></td></tr></table>",
        "td { padding: 0; }",
    );
    let w0 = table_cell(&l, 0, 0).rect.w;
    let w1 = table_cell(&l, 0, 1).rect.w;
    assert!(approx(w0, 150.0), "col0 {w0} != 150");
    assert!(approx(w1, 150.0), "col1 {w1} != 150");
}

#[test]
fn merged_track_does_not_carry_span_percentage() {
    // Intrinsic percentages read the pre-merge span (same guard as
    // constrainedness): a `colspan=10` cell's `width:50%` applied to
    // the 10-track span, so after merging it must not contribute 50%
    // to its single merged track. Without the guard the bogus 50%
    // pins the track's guesses at 200px and diverts the whole excess
    // to the other column (200/200); with it the track is an ordinary
    // unconstrained track (max-content 200 from the resolved specified
    // width, kept by the content-measure sites) sharing the 400px
    // table's excess proportionally: 200 + 190.4*200/209.6 = 381.68,
    // 9.6 + 190.4*9.6/209.6 = 18.32.
    let l = layout_html(
        "<table style=\"width:400px\"><tr><td colspan=\"10\" style=\"width:50%\">x</td><td>y</td></tr></table>",
        "td { padding: 0; }",
    );
    let w0 = table_cell(&l, 0, 0).rect.w;
    let w1 = table_cell(&l, 0, 1).rect.w;
    assert!(approx(w0, 381.68), "col0 {w0} != 381.68");
    assert!(approx(w1, 18.32), "col1 {w1} != 18.32");
}

#[test]
fn colgroup_col_span_widths() {
    // WPT css/css-tables/tentative/colgroup-col.html table 4: `col`
    // `span` attributes inside a `colgroup` generate multiple columns,
    // each taking the `col`'s width (or the group's when the `col` has
    // none).
    let l = layout_html(
        "<table><colgroup span=\"3\" style=\"width:100px\"><col><col style=\"width:50px\" span=\"2\"></colgroup><colgroup style=\"width:66px\"><col span=\"2\"></colgroup><tr><td>a</td><td>b</td><td>c</td><td>d</td><td>e</td><td><div style=\"width:75px\">x</div></td></tr></table>",
        "td { padding: 0; }",
    );
    let widths: Vec<f32> = (0..6).map(|c| table_cell(&l, 0, c).rect.w).collect();
    let expected = [100.0, 50.0, 50.0, 66.0, 66.0, 75.0];
    for (i, (&got, &exp)) in widths.iter().zip(expected.iter()).enumerate() {
        assert!(approx(got, exp), "col {i}: got {got}, exp {exp}");
    }
}

/// First descendant element of the document with the given tag name
/// (depth-first, document order).
fn descendant_tag(doc: &bowser_dom::Document, tag: &str) -> bowser_dom::NodeId {
    fn walk(
        doc: &bowser_dom::Document,
        n: bowser_dom::NodeId,
        tag: &str,
    ) -> Option<bowser_dom::NodeId> {
        if matches!(&doc.kind(n), bowser_dom::NodeKind::Element(e) if e.tag_name == tag) {
            return Some(n);
        }
        doc.children(n).into_iter().find_map(|c| walk(doc, c, tag))
    }
    walk(doc, doc.document_element().expect("root"), tag).expect("tagged descendant")
}

#[test]
fn w77_replaced_flex_basis_max_content_ignores_definite_width() {
    // w77 task 2: a replaced flex item's content-keyword basis
    // measures its intrinsic natural size, not its definite width:
    // <img width:100px> + flex-basis:max-content over a 300px natural
    // size bases at 300, not 100. Block-level so the img is the flex
    // item itself (an inline img is wrapped in an anonymous item).
    let (doc, _) =
        bowser_dom::Document::parse_html(r#"<div class="c"><img class="i"></div>"#.as_bytes());
    let img = descendant_tag(&doc, "img");
    let mut map = HashMap::new();
    map.insert(img, IntrinsicDims::from_size(300.0, 150.0));
    let l = layout_doc_with_intrinsics(
        &doc,
        "body { margin: 0; } .c { display: flex; width: 800px; } \
         .i { display: block; width: 100px; flex-basis: max-content; flex-grow: 0; flex-shrink: 0; }",
        &FixedFontMetrics,
        &map,
    );
    let item = &body(&l).children[0].children[0];
    assert!(
        approx(item.rect.w, 300.0),
        "replaced max-content basis {w} != 300",
        w = item.rect.w
    );
}

#[test]
fn w77_replaced_flex_basis_min_content_ignores_definite_width() {
    // Same shape with flex-basis:min-content: a replaced box's
    // min-content is its intrinsic size, not the definite floor.
    let (doc, _) =
        bowser_dom::Document::parse_html(r#"<div class="c"><img class="i"></div>"#.as_bytes());
    let img = descendant_tag(&doc, "img");
    let mut map = HashMap::new();
    map.insert(img, IntrinsicDims::from_size(300.0, 150.0));
    let l = layout_doc_with_intrinsics(
        &doc,
        "body { margin: 0; } .c { display: flex; width: 800px; } \
         .i { display: block; width: 100px; flex-basis: min-content; flex-grow: 0; flex-shrink: 0; }",
        &FixedFontMetrics,
        &map,
    );
    let item = &body(&l).children[0].children[0];
    assert!(
        approx(item.rect.w, 300.0),
        "replaced min-content basis {w} != 300",
        w = item.rect.w
    );
}

#[test]
fn w77_replaced_flex_basis_border_box_strips_padding_border() {
    // The keyword basis is content-box (like every other
    // flex_basis_content_width result): under border-box, the
    // intrinsic 300px is stripped of padding/border for the basis
    // and re-added at used size, so the item still lands at 300
    // border-box — not 330 (raw intrinsic as content) or 100.
    let (doc, _) =
        bowser_dom::Document::parse_html(r#"<div class="c"><img class="i"></div>"#.as_bytes());
    let img = descendant_tag(&doc, "img");
    let mut map = HashMap::new();
    map.insert(img, IntrinsicDims::from_size(300.0, 150.0));
    let l = layout_doc_with_intrinsics(
        &doc,
        "body { margin: 0; } .c { display: flex; width: 800px; } \
         .i { display: block; width: 100px; box-sizing: border-box; padding: 10px; border: 5px solid black; \
              flex-basis: max-content; flex-grow: 0; flex-shrink: 0; }",
        &FixedFontMetrics,
        &map,
    );
    let item = &body(&l).children[0].children[0];
    assert!(
        approx(item.rect.w, 300.0),
        "border-box replaced basis {w} != 300",
        w = item.rect.w
    );
}

#[test]
fn w77_intrinsic_width_collapses_cross_node_space_seam() {
    // w77 task 3: line layout's collapse_adjacent_spaces zeroes one
    // of two adjacent spaces across a text-node/inline boundary, but
    // the max-content walk counted both. A shrink-to-fit inline-block
    // over <span>a </span> + text " b" must measure "a b" (18px at
    // 10px Fixed metrics), not "a  b" (24px).
    let l = layout_html(
        r#"<div class="a"><span>a </span> b</div>"#,
        ".a { display: inline-block; font-size: 10px; }",
    );
    let a = &body(&l).children[0];
    assert!(
        approx(a.rect.w, 18.0),
        "seam max-content {w} != 18",
        w = a.rect.w
    );
}

#[test]
fn w77_intrinsic_width_collapses_space_seam_between_inlines() {
    // Same seam with both sides inside inline wrappers: the second
    // span's leading space collapses against the first span's
    // trailing space, so the inline-block measures 18, not 24.
    let l = layout_html(
        r#"<div class="a"><span>a </span><span> b</span></div>"#,
        ".a { display: inline-block; font-size: 10px; }",
    );
    let a = &body(&l).children[0];
    assert!(
        approx(a.rect.w, 18.0),
        "inline-inline seam {w} != 18",
        w = a.rect.w
    );
}

#[test]
fn w77_intrinsic_width_single_sided_space_is_not_dropped() {
    // Control: only one side of the seam holds a space, so nothing
    // collapses — "a" + " b" is still 18, and the fix must not drop
    // a lone leading space (which would give 12).
    let l = layout_html(
        r#"<div class="a"><span>a</span> b</div>"#,
        ".a { display: inline-block; font-size: 10px; }",
    );
    let a = &body(&l).children[0];
    assert!(
        approx(a.rect.w, 18.0),
        "single-sided space {w} != 18",
        w = a.rect.w
    );
}

#[test]
fn w77_intrinsic_width_pre_seam_spaces_are_preserved() {
    // Control: under white-space:pre, seam spaces are preserved in
    // line layout too, so the max-content width keeps both: 24.
    let l = layout_html(
        r#"<div class="a"><span>a </span> b</div>"#,
        ".a { display: inline-block; font-size: 10px; white-space: pre; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 24.0), "pre seam {w} != 24", w = a.rect.w);
}

#[test]
fn w77fix_seam_tab_edge_collapses_like_space() {
    // w77fix finding 13: `seam_space_drop` only checks U+0020
    // edges, but a tab edge reaches it already normalized:
    // `boxes::collapse_ws` turns a trailing tab into U+0020 for
    // normal white-space at box-build time, so the seam drop
    // fires exactly as for a space. A shrink-to-fit inline-block
    // over <span>a\t</span> + " b" must measure 18, not 24.
    let l = layout_html(
        "<div class=\"a\"><span>a\t</span> b</div>",
        ".a { display: inline-block; font-size: 10px; }",
    );
    let a = &body(&l).children[0];
    assert!(approx(a.rect.w, 18.0), "tab seam {w} != 18", w = a.rect.w);
}

#[test]
fn w77fix_seam_newline_edge_collapses_like_space() {
    // Same as the tab case for a newline edge: normal
    // white-space collapses it to U+0020 at box-build time, so
    // the seam drop fires and the inline-block measures 18.
    let l = layout_html(
        "<div class=\"a\"><span>a\n</span> b</div>",
        ".a { display: inline-block; font-size: 10px; }",
    );
    let a = &body(&l).children[0];
    assert!(
        approx(a.rect.w, 18.0),
        "newline seam {w} != 18",
        w = a.rect.w
    );
}

#[test]
fn w77fix_seam_tab_edge_collapses_under_pre_line() {
    // Pre-line also normalizes tabs to U+0020 (`collapse_ws`
    // pre-line arm), so the tab seam still drops: 18, not 24.
    let l = layout_html(
        "<div class=\"a\"><span>a\t</span> b</div>",
        ".a { display: inline-block; font-size: 10px; white-space: pre-line; }",
    );
    let a = &body(&l).children[0];
    assert!(
        approx(a.rect.w, 18.0),
        "pre-line tab seam {w} != 18",
        w = a.rect.w
    );
}

#[test]
fn w77fix_seam_newline_edge_preserved_under_pre_line() {
    // Pre-line is the exception that proves the rule: a newline
    // is a preserved segment break there, NOT a collapsible
    // space, so `seam_space_drop`'s U+0020 check correctly does
    // not fire (line layout's `collapse_adjacent_spaces` never
    // sees an `is_space` fragment for it either). Both sides
    // count: advance("a\n") + advance(" b") = 24 at 10px Fixed
    // metrics. Treating \n as a droppable space would give 18.
    let l = layout_html(
        "<div class=\"a\"><span>a\n</span> b</div>",
        ".a { display: inline-block; font-size: 10px; white-space: pre-line; }",
    );
    let a = &body(&l).children[0];
    assert!(
        approx(a.rect.w, 24.0),
        "pre-line newline seam {w} != 24",
        w = a.rect.w
    );
}

#[test]
fn w77d_vertical_table_caption_margins_grow_block_size() {
    // WPT css/css-tables/caption-orthogonal-margin-in-table-block-axis:
    // a vertical-rl table's block axis is physical-horizontal, so the
    // caption's two 40px horizontal margins must grow the table's
    // physical width by 80px versus the no-margin table, in both the
    // same-writing-mode and the orthogonal-caption subtests. Before
    // the w77-d fix the growth was only 50px: the block engine
    // subtracted the margins from the caption's fill width (30px
    // extent collapsed to 0) and `place_table_rows` added them again.
    let probe = |cap_wm: &str, margin: &str| -> (f32, f32) {
        let html = r#"<table class="t"><caption>c</caption><tr><td></td></tr></table>"#.to_string();
        let css = format!(
            ".t {{ writing-mode: vertical-rl; border-spacing: 0; }} caption {{ writing-mode: {}; margin: {}; font-size: 10px; }} td {{ width: 50px; height: 50px; padding: 0; }} table, td {{ border: 0; }}",
            cap_wm, margin
        );
        let l = layout_html(&html, &css);
        let table = &body(&l).children[0];
        let cap = table
            .children
            .iter()
            .find(|c| c.box_type == BoxType::Block)
            .expect("caption");
        (table.rect.w, cap.rect.w)
    };
    for cap_wm in ["vertical-rl", "horizontal-tb"] {
        let (w_m, cap_m) = probe(cap_wm, "10px 40px");
        let (w_n, cap_n) = probe(cap_wm, "0");
        assert!(
            approx(w_m - w_n, 80.0),
            "{cap_wm}: margin growth {} != 80 ({} vs {})",
            w_m - w_n,
            w_m,
            w_n
        );
        assert!(
            approx(cap_m, cap_n),
            "{cap_wm}: caption extent {} != no-margin {}",
            cap_m,
            cap_n
        );
    }
}

#[test]
fn w77d_deferred_orthogonal_caption_rewraps_against_block_size() {
    // W77-d task 1: vertical-lr table, auto block size, a horizontal
    // caption whose text wraps. Rows give a 50px block size; the
    // caption must re-lay against 50, not the 30px inline content
    // width: 10 "aa" words wrap 3/line (48px tall, 50 wide), not
    // 2/line (60px tall, 30 wide). Fail-before: table 80, cap 30x60.
    let html = r#"<table class="t"><caption>aa aa aa aa aa aa aa aa aa aa</caption><tr><td></td></tr></table>"#.to_string();
    let css = ".t { writing-mode: vertical-lr; border-spacing: 0; } caption { font-size: 10px; writing-mode: horizontal-tb; } td { width: 50px; height: 30px; padding: 0; } table, td { border: 0; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0];
    let cap = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::Block)
        .expect("caption");
    assert!(approx(cap.rect.w, 50.0), "cap w {} != 50", cap.rect.w);
    assert!(approx(cap.rect.h, 48.0), "cap h {} != 48", cap.rect.h);
    assert!(
        approx(table.rect.w, 100.0),
        "table w {} != 100",
        table.rect.w
    );
    assert!(approx(table.rect.h, 30.0), "table h {} != 30", table.rect.h);
    assert!(
        approx(cap.rect.x, table.rect.x),
        "cap x {} != table x {}",
        cap.rect.x,
        table.rect.x
    );
}

#[test]
fn w77d_deferred_orthogonal_caption_vertical_rl_mirrors() {
    // Vertical-rl mirror of the rewrap test: the top caption sits at
    // the block-start (right) edge, flush with the table's right edge.
    let html = r#"<table class="t"><caption>aa aa aa aa aa aa aa aa aa aa</caption><tr><td></td></tr></table>"#.to_string();
    let css = ".t { writing-mode: vertical-rl; border-spacing: 0; } caption { font-size: 10px; writing-mode: horizontal-tb; } td { width: 50px; height: 30px; padding: 0; } table, td { border: 0; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0];
    let cap = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::Block)
        .expect("caption");
    assert!(approx(cap.rect.w, 50.0), "cap w {} != 50", cap.rect.w);
    assert!(approx(cap.rect.h, 48.0), "cap h {} != 48", cap.rect.h);
    assert!(
        approx(table.rect.w, 100.0),
        "table w {} != 100",
        table.rect.w
    );
    let gap = (table.rect.x + table.rect.w) - (cap.rect.x + cap.rect.w);
    assert!(approx(gap, 0.0), "block-start gap {} != 0", gap);
}

#[test]
fn w77d_deferred_orthogonal_caption_bottom_stacks_after_rows() {
    // Bottom deferred caption: it stacks after the rows toward the
    // block-end (right side for vertical-lr), not at the block-start.
    let html = r#"<table class="t"><caption>aa aa aa aa aa aa aa aa aa aa</caption><tr><td></td></tr></table>"#.to_string();
    let css = ".t { writing-mode: vertical-lr; border-spacing: 0; } caption { font-size: 10px; writing-mode: horizontal-tb; caption-side: bottom; } td { width: 50px; height: 30px; padding: 0; } table, td { border: 0; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0];
    let cap = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::Block)
        .expect("caption");
    assert!(approx(cap.rect.w, 50.0), "cap w {} != 50", cap.rect.w);
    assert!(approx(cap.rect.h, 48.0), "cap h {} != 48", cap.rect.h);
    assert!(
        approx(table.rect.w, 100.0),
        "table w {} != 100",
        table.rect.w
    );
    assert!(
        approx(cap.rect.x - table.rect.x, 50.0),
        "cap x rel {} != 50",
        cap.rect.x - table.rect.x
    );
}

#[test]
fn w77d_deferred_orthogonal_caption_fixed_margins_preserved() {
    // Deferred caption with fixed horizontal margins: the re-laid
    // border stays at the 50px block size (task 2 override at the
    // final push) and the 40px margins add on top: table 50 + 130.
    let html = r#"<table class="t"><caption>aa aa aa aa aa aa aa aa aa aa</caption><tr><td></td></tr></table>"#.to_string();
    let css = ".t { writing-mode: vertical-lr; border-spacing: 0; } caption { font-size: 10px; writing-mode: horizontal-tb; margin: 10px 40px; } td { width: 50px; height: 30px; padding: 0; } table, td { border: 0; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0];
    let cap = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::Block)
        .expect("caption");
    assert!(approx(cap.rect.w, 50.0), "cap w {} != 50", cap.rect.w);
    assert!(approx(cap.rect.h, 48.0), "cap h {} != 48", cap.rect.h);
    assert!(
        approx(table.rect.w, 180.0),
        "table w {} != 180",
        table.rect.w
    );
    assert!(
        approx(cap.rect.x - table.rect.x, 40.0),
        "cap x rel {} != 40",
        cap.rect.x - table.rect.x
    );
    assert!(
        approx(cap.rect.y - table.rect.y, 10.0),
        "cap y rel {} != 10",
        cap.rect.y - table.rect.y
    );
}

#[test]
fn w77d_deferred_orthogonal_caption_auto_margins_recenter_after_rewrap() {
    // W77-d task 1b: auto top/bottom margins distribute against the
    // re-laid caption height. Inline content 100, caption re-wraps
    // to 48 tall at the 50px block width, so free 52 splits 26/26.
    // Fail-before: cap 100x24 at y rel 38.
    let html = r#"<table class="t"><caption>aa aa aa aa aa aa aa aa aa aa</caption><tr><td></td></tr></table>"#.to_string();
    let css = ".t { writing-mode: vertical-lr; border-spacing: 0; } caption { font-size: 10px; writing-mode: horizontal-tb; margin: auto 0; } td { width: 50px; height: 100px; padding: 0; } table, td { border: 0; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0];
    let cap = table
        .children
        .iter()
        .find(|c| c.box_type == BoxType::Block)
        .expect("caption");
    assert!(approx(cap.rect.w, 50.0), "cap w {} != 50", cap.rect.w);
    assert!(approx(cap.rect.h, 48.0), "cap h {} != 48", cap.rect.h);
    assert!(
        approx(cap.rect.y - table.rect.y, 26.0),
        "cap y rel {} != 26",
        cap.rect.y - table.rect.y
    );
}

#[test]
fn w77d_deferred_caption_interleaves_with_nondeferred_in_dom_order() {
    // W77-d task 1 (merged stacking): an auto vertical-lr table with
    // two top captions in DOM order — a same-mode (non-deferred)
    // caption first, then a deferred orthogonal caption. First place
    // stacks the 30px same-mode caption + 50px rows (80); the
    // deferred caption then re-lays against that finalized 80 (4
    // words/line, 36 tall). The final placement must keep the
    // same-mode caption at offset 0 and put the deferred caption at
    // offset 30 (table 160), not stack the deferred caption first
    // and shift the same-mode one to 50.
    let html = r#"<table class="t"><caption style="writing-mode: vertical-rl;">v</caption><caption style="writing-mode: horizontal-tb;">aa aa aa aa aa aa aa aa aa aa</caption><tr><td></td></tr></table>"#.to_string();
    let css = ".t { writing-mode: vertical-lr; border-spacing: 0; } caption { font-size: 10px; } td { width: 50px; height: 30px; padding: 0; } table, td { border: 0; }";
    let l = layout_html(&html, css);
    let table = &body(&l).children[0];
    let caps: Vec<&LayoutBox> = table
        .children
        .iter()
        .filter(|c| c.box_type == BoxType::Block)
        .collect();
    assert_eq!(caps.len(), 2, "want 2 caption boxes, got {}", caps.len());
    assert!(
        approx(caps[0].rect.w, 30.0),
        "same-mode cap w {} != 30",
        caps[0].rect.w
    );
    assert!(
        approx(caps[0].rect.x, table.rect.x),
        "same-mode cap x {} != table x {}",
        caps[0].rect.x,
        table.rect.x
    );
    assert!(
        approx(caps[1].rect.w, 80.0),
        "deferred cap w {} != 80",
        caps[1].rect.w
    );
    assert!(
        approx(caps[1].rect.h, 36.0),
        "deferred cap h {} != 36",
        caps[1].rect.h
    );
    assert!(
        approx(caps[1].rect.x - table.rect.x, 30.0),
        "deferred cap x rel {} != 30",
        caps[1].rect.x - table.rect.x
    );
    assert!(
        approx(table.rect.w, 160.0),
        "table w {} != 160",
        table.rect.w
    );
}

// w77 (grids, mirroring the w76fix table arm): a definite-width
// grid item with a max-content basis measures its tracks (150 +
// 150 = 300), not its width (100). The shrink-to-fit flex
// container pins the base: its width is set from the item bases
// before any grid layout runs, so the grid-growth masking that
// makes a plain final-width assertion pass either way cannot hide
// a floored base here. Before the fix the container measured 100.
#[test]
fn w77_grid_basis_max_content_uses_track_sum() {
    let l = layout_html(
        r#"<div class="wrap"><div class="c"><div class="g"><div class="i"></div><div class="i"></div></div></div></div>"#,
        "body { margin: 0; } .wrap { float: left; } .c { display: flex; } \
         .g { display: grid; grid-template-columns: 150px 150px; width: 100px; \
              flex-basis: max-content; flex-grow: 0; flex-shrink: 0; } .i { height: 20px; }",
    );
    let c = &body(&l).children[0].children[0];
    assert!(
        approx(c.rect.w, 300.0),
        "shrink flex container w={}",
        c.rect.w
    );
}

// Control: ordinary (non-flex-basis) grid layout is unchanged.
// A definite-width grid still lays out at its width with its
// tracks overflowing (second item at x=150), and its ordinary
// preferred width still floors at the definite width: a float
// wrapper around it measures 100, not the 300 track sum.
#[test]
fn w77_grid_ordinary_layout_definite_width_unchanged() {
    let l = layout_html(
        r#"<div class="g"><div class="i"></div><div class="i"></div></div>"#,
        ".g { display: grid; grid-template-columns: 150px 150px; width: 100px; } \
         .i { height: 20px; }",
    );
    let g = grid_box(&l);
    assert!(approx(g.rect.w, 100.0), "ordinary grid w={}", g.rect.w);
    assert!(
        approx(g.children[1].rect.x, 150.0),
        "item2 x={}",
        g.children[1].rect.x
    );
    let l = layout_html(
        r#"<div class="wrap"><div class="g"><div class="i"></div><div class="i"></div></div></div>"#,
        ".wrap { float: left; } .g { display: grid; grid-template-columns: 150px 150px; width: 100px; } \
         .i { height: 20px; }",
    );
    let wrap = &body(&l).children[0];
    assert!(
        approx(wrap.rect.w, 100.0),
        "ordinary shrink wrap w={}",
        wrap.rect.w
    );
}
