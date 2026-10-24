//! Eager vs deferred `font` shorthand parity.
//!
//! The same `font:` shorthand can take two resolution paths: the eager
//! token-level parser (`parse_font_shorthand` in parse.rs) when the
//! declaration has no `var()`, and the deferred cascade-side expansion
//! (`expand_font_value` in cascade.rs) once `var()` substitution runs.
//! The two paths must agree on all six longhands (`font-style`,
//! `font-variant`, `font-weight`, `font-size`, `line-height`,
//! `font-family`) for every var-free value — including the `/`
//! line-height separator, system font keywords, authored-case family
//! names, and invalid inputs (both must reject the same values and leave
//! the same fallbacks in place).

use bowser_css::{Styler, default_ua_stylesheet, parse_stylesheet};
use bowser_dom::Document;

const LONGHANDS: [&str; 6] = [
    "font-style",
    "font-variant",
    "font-weight",
    "font-size",
    "line-height",
    "font-family",
];

/// Computed values of the six `font` longhands on a fresh div, as
/// `Debug` strings (deterministic and comparable).
fn longhands(css: &str) -> [String; 6] {
    let mut doc = Document::new();
    let root = doc.root();
    let html = doc.create_element("html");
    doc.append_child(root, html);
    let div = doc.create_element("div");
    doc.append_child(html, div);
    let sheet = parse_stylesheet(css);
    let ua = default_ua_stylesheet();
    let refs = [&sheet];
    let styler = Styler::new(
        &doc,
        &refs,
        &ua,
        bowser_css::Viewport { w: 800.0, h: 600.0 },
    );
    let s = styler.style_for(div);
    LONGHANDS.map(|p| format!("{:?}", s.get(p)))
}

/// `font: <v>` must compute exactly like `font: var(--f)` with
/// `--f: <v>` for every var-free shorthand value.
fn assert_parity(v: &str) {
    let eager = longhands(&format!("div {{ font: {v}; }}"));
    let deferred = longhands(&format!("div {{ --f: {v}; font: var(--f); }}"));
    assert_eq!(eager, deferred, "eager/deferred mismatch for `font: {v}`");
}

#[test]
fn font_eager_deferred_parity() {
    // Core shapes: prefix longhands, size, line-height, families.
    for v in [
        "12px serif",
        "italic bold 12px/1.5 serif",
        "oblique small-caps 600 1.2em/150% \"Times New Roman\", Georgia, serif",
        "normal normal normal 12px serif",
        "700 16px \"Font Name\", sans-serif",
        "italic 12px/1.4em Times New Roman, serif",
        "100 12px serif",
        "900 12px serif",
        "bolder lighter 12px serif", // second weight: invalid in both
        "12px/1.5 serif",
        "12px / 1.5 serif", // spaces around the slash
        "12px/1.5",
        "12px/2.0 serif",
        "12px/0 serif",
        "12px/120% serif",
        "12px/1.2em serif",
        "12px/normal serif",
        "12px/Normal serif", // line-height keyword case folds in both
        "12px/calc(1 + 1) serif",
        "calc(10px + 2px)/1.5 serif",
        "120%/1.5 serif",
        "12q serif",
        "12PX/1.5 SERIF",  // authored case: units fold, families keep case
        "12px Red, serif", // color-named family keeps authored case
        "12px/1.5 BLUE, serif",
        "12px \"a\" \"b\"",            // adjacent strings: invalid in both
        "12px , serif",                // leading comma: invalid in both
        "12px, serif",                 // leading comma, no space: invalid in both
        "12px/1.5, serif",             // comma right after /line-height: invalid in both
        "italic, 12px/1.5 serif",      // comma before the size: invalid in both
        "12px/1.5 serif,, sans-serif", // doubled comma: invalid in both
        "12px/1.5 Arial, serif",       // comma between families: valid in both
        "12px serif,",                 // trailing comma: invalid in both
        "italic/1.5 serif",            // missing size: invalid in both
        "12px//1.5 serif",             // double slash: invalid in both
        "small 12px serif",            // unknown prefix ident: invalid in both
        "12px 123",                    // numeric family: invalid in both
        "italic italic 12px serif",    // doubled style: invalid in both
        "bold 700 12px serif",         // doubled weight: invalid in both
        "12px",
        "serif",
        "caption",
        "CAPTION",            // system keyword case-insensitive in both
        "caption 12px serif", // keyword must be sole: invalid in both
    ] {
        assert_parity(v);
    }
}

#[test]
fn font_deferred_slash_through_var_positions() {
    // The `/` line-height separator reaches the deferred expansion from
    // several substitution positions; each must match the equivalent
    // fully-eager declaration.
    let eager = longhands("div { font: italic 12px/1.5 serif; }");
    for css in [
        // Slash inside the variable.
        "div { --f: italic 12px/1.5 serif; font: var(--f); }",
        // Slash between two variables at the token level.
        "div { --f: italic 12px; font: var(--f)/1.5 serif; }",
        // Slash inside a var() fallback.
        "div { font: var(--missing, italic 12px/1.5 serif); }",
        // Every component from its own variable.
        "div { --a: italic 12px; --b: 1.5; --c: serif; font: var(--a)/var(--b) var(--c); }",
        // Whole value spliced into a sibling list.
        "div { --f: 12px/1.5 serif; font: italic var(--f); }",
    ] {
        assert_eq!(longhands(css), eager, "slash position mismatch: {css}");
    }
}

#[test]
fn font_deferred_nested_var_comma_on_slash_side() {
    // Commas arriving through a nested `var()` on the `/`-side were
    // never top-level in the outer variable's tokens, so they stay
    // valid family separators: `12px/1.5 var(--g)` with
    // `--g: Arial, serif` is `font: 12px/1.5 Arial, serif`.
    let eager = longhands("div { font: 12px/1.5 Arial, serif; }");
    let deferred = longhands("div { --g: Arial, serif; --f: 12px/1.5 var(--g); font: var(--f); }");
    assert_eq!(eager, deferred, "nested var comma mismatch");
}

#[test]
fn font_deferred_comma_families_through_var() {
    // Comma-separated families (and commas spliced in by substitution)
    // must expand to the same family list as the eager parser.
    let eager = longhands("div { font: 12px/1.5 Arial, \"Times New Roman\", serif; }");
    for css in [
        "div { --f: 12px/1.5 Arial, \"Times New Roman\", serif; font: var(--f); }",
        "div { --f: Arial, \"Times New Roman\", serif; font: 12px/1.5 var(--f); }",
        "div { --f: Arial; font: 12px/1.5 var(--f), \"Times New Roman\", serif; }",
    ] {
        assert_eq!(longhands(css), eager, "comma family mismatch: {css}");
    }
}

#[test]
fn font_deferred_var_comma_leading_on_slash_side_dropped() {
    // `font: 12px/var(--g)` with `--g: 1.5, serif` substitutes
    // token-wise to `font: 12px/1.5, serif`: a leading comma in the
    // `<family-name>#` list, invalid at computed-value time, so the
    // declaration must be dropped — exactly like the eager parser.
    for (eager_css, deferred_css) in [
        (
            "div { font: 12px/1.5, serif; }",
            "div { --g: 1.5, serif; font: 12px/var(--g); }",
        ),
        // A keyword line-height substituted the same way is equally
        // invalid: `font: 12px/Arial, serif` has the same leading comma
        // in the family list.
        (
            "div { font: 12px/Arial, serif; }",
            "div { --g: Arial, serif; font: 12px/var(--g); }",
        ),
    ] {
        let eager = longhands(eager_css);
        let deferred = longhands(deferred_css);
        // Both paths drop the declaration: `font-size` falls back to
        // `medium`, `line-height` to `normal`.
        assert_eq!(
            eager[3], "Some(Keyword(\"medium\"))",
            "eager not dropped: {eager_css}"
        );
        assert_eq!(
            deferred[3], "Some(Keyword(\"medium\"))",
            "deferred not dropped: {deferred_css}"
        );
        assert_eq!(eager, deferred, "eager/deferred mismatch: {deferred_css}");
    }
}
