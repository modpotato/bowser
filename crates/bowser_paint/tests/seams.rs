//! Cross-crate seam guards for the deduplicated style/unit logic.
//!
//! Following the `executable_script_types_match_bindings` pattern in
//! `script.rs`: where several crates must agree on one mapping, a test
//! pins the shared behavior so a future edit to one side cannot
//! silently drift from the others.
//!
//! - `font-size` resolution: paint's display-list builder calls
//!   [`bowser_layout::resolve_font_size`] directly (the former local
//!   mirror is gone). This battery pins the shared mapping — UA keyword
//!   scale, relative units, clamp-to-zero — so any change to the
//!   canonical implementation is deliberate and visible to paint's owner.
//! - `LengthUnit -> px` factor table: the CSS math evaluator, layout's
//!   `SpecLen::resolve`/`resolve_font_size`, and paint's `resolve_length`
//!   all route through [`bowser_css::length_to_px`]. This battery pins
//!   the table itself.
//! - font-request extraction: `bowser_app`'s window font resolution and
//!   paint's own display-list font resolver both call
//!   `render::{css_font_family, css_font_weight, css_font_style}`. This
//!   battery pins that shared `(family, weight, style)` reading through
//!   the real cascade.

use bowser_css::{
    ComputedStyle, CssValue, LengthUnit, Viewport, compute_style, default_ua_stylesheet,
    length_to_px, parse_stylesheet,
};
use bowser_dom::Document;
use bowser_layout::resolve_font_size;
use bowser_paint::fonts::FontStyle;
use bowser_paint::render::{css_font_family, css_font_style, css_font_weight};

fn kw(s: &str) -> Option<CssValue> {
    Some(CssValue::Keyword(s.into()))
}

/// (value, parent_fs, root_fs, vp_w, vp_h, expected_px)
fn font_size_cases() -> Vec<(Option<CssValue>, f32, f32, f32, f32, f32)> {
    vec![
        // Missing inherits the parent size.
        (None, 14.0, 16.0, 800.0, 600.0, 14.0),
        // Absolute units.
        (
            Some(CssValue::Length(20.0, LengthUnit::Px)),
            14.0,
            16.0,
            800.0,
            600.0,
            20.0,
        ),
        // Font-relative units.
        (
            Some(CssValue::Length(2.0, LengthUnit::Em)),
            14.0,
            16.0,
            800.0,
            600.0,
            28.0,
        ),
        (
            Some(CssValue::Length(1.5, LengthUnit::Rem)),
            14.0,
            16.0,
            800.0,
            600.0,
            24.0,
        ),
        // Viewport units.
        (
            Some(CssValue::Length(10.0, LengthUnit::Vw)),
            14.0,
            16.0,
            800.0,
            600.0,
            80.0,
        ),
        (
            Some(CssValue::Length(10.0, LengthUnit::Vh)),
            14.0,
            16.0,
            800.0,
            600.0,
            60.0,
        ),
        // Percentages and unitless numbers.
        (
            Some(CssValue::Percentage(150.0)),
            14.0,
            16.0,
            800.0,
            600.0,
            21.0,
        ),
        (Some(CssValue::Number(11.0)), 14.0, 16.0, 800.0, 600.0, 11.0),
        // UA keyword scale.
        (kw("xx-small"), 14.0, 16.0, 800.0, 600.0, 9.0),
        (kw("x-small"), 14.0, 16.0, 800.0, 600.0, 10.0),
        (kw("small"), 14.0, 16.0, 800.0, 600.0, 13.0),
        (kw("medium"), 14.0, 16.0, 800.0, 600.0, 16.0),
        (kw("large"), 14.0, 16.0, 800.0, 600.0, 18.0),
        (kw("x-large"), 14.0, 16.0, 800.0, 600.0, 24.0),
        (kw("xx-large"), 14.0, 16.0, 800.0, 600.0, 32.0),
        // Relative and unknown keywords; keywords are case-insensitive.
        (kw("larger"), 10.0, 16.0, 800.0, 600.0, 12.0),
        (kw("smaller"), 12.0, 16.0, 800.0, 600.0, 10.0),
        (kw("LARGE"), 14.0, 16.0, 800.0, 600.0, 18.0),
        (kw("bogus"), 14.0, 16.0, 800.0, 600.0, 14.0),
        // Negative results clamp to zero.
        (
            Some(CssValue::Length(-5.0, LengthUnit::Px)),
            14.0,
            16.0,
            800.0,
            600.0,
            0.0,
        ),
        (
            Some(CssValue::Percentage(-50.0)),
            14.0,
            16.0,
            800.0,
            600.0,
            0.0,
        ),
    ]
}

#[test]
fn font_size_resolution_matches_layout_spec() {
    for (value, parent_fs, root_fs, vp_w, vp_h, expected) in font_size_cases() {
        let got = resolve_font_size(value.as_ref(), parent_fs, root_fs, vp_w, vp_h);
        assert!(
            (got - expected).abs() < 1e-4,
            "font-size {value:?} (parent {parent_fs}, root {root_fs}, viewport {vp_w}x{vp_h}): got {got}, expected {expected}"
        );
    }
}

#[test]
fn length_unit_factor_table_is_canonical() {
    // (unit, value, em_basis, root_fs, vp_w, vp_h, expected_px)
    let cases = [
        (LengthUnit::Px, 7.0, 14.0, 16.0, 800.0, 600.0, 7.0),
        (LengthUnit::Em, 2.0, 14.0, 16.0, 800.0, 600.0, 28.0),
        (LengthUnit::Rem, 1.5, 14.0, 16.0, 800.0, 600.0, 24.0),
        (LengthUnit::Vw, 10.0, 14.0, 16.0, 800.0, 600.0, 80.0),
        (LengthUnit::Vh, 10.0, 14.0, 16.0, 800.0, 600.0, 60.0),
        // Zero bases stay zero; negative values pass through unchanged
        // (clamping is each caller's job, not the table's).
        (LengthUnit::Em, 3.0, 0.0, 16.0, 800.0, 600.0, 0.0),
        (LengthUnit::Px, -4.0, 14.0, 16.0, 800.0, 600.0, -4.0),
    ];
    for (unit, value, em, root, vw, vh, expected) in cases {
        let got = length_to_px(value, &unit, em, root, vw, vh);
        assert!(
            (got - expected).abs() < 1e-4,
            "length_to_px({value}, {unit:?}): got {got}, expected {expected}"
        );
    }
}

/// The computed style of `<p style='...'>x</p>` through the real cascade.
/// Single-quoted attribute so values may contain double quotes.
fn styled(style_attr: &str) -> ComputedStyle {
    let html = format!(r#"<p style='{style_attr}'>x</p>"#);
    let (doc, _) = Document::parse_html(html.as_bytes());
    let p = doc
        .query_selector("p")
        .expect("selector parses")
        .expect("parsed <p>");
    let author = parse_stylesheet("");
    compute_style(
        &doc,
        p,
        &[&author],
        &default_ua_stylesheet(),
        &Viewport { w: 800.0, h: 600.0 },
    )
}

#[test]
fn font_request_extraction_matches_shared_spec() {
    // (style attribute, expected family, weight, style)
    let cases = [
        ("", "serif", 400, FontStyle::Normal),
        (
            "font-family: monospace",
            "monospace",
            400,
            FontStyle::Normal,
        ),
        (
            "font-family: \"Helvetica Neue\", Arial, sans-serif",
            "Helvetica Neue",
            400,
            FontStyle::Normal,
        ),
        ("font-weight: bold", "serif", 700, FontStyle::Normal),
        ("font-weight: normal", "serif", 400, FontStyle::Normal),
        ("font-weight: 100", "serif", 100, FontStyle::Normal),
        ("font-weight: 900", "serif", 900, FontStyle::Normal),
        // Per CSS Fonts 4, integer weights 1..=1000 are valid; only
        // out-of-range numerics are invalid and fall back to `normal`.
        ("font-weight: 1000", "serif", 1000, FontStyle::Normal),
        ("font-weight: 2500", "serif", 400, FontStyle::Normal),
        ("font-style: italic", "serif", 400, FontStyle::Italic),
        ("font-style: oblique", "serif", 400, FontStyle::Oblique),
        ("font-style: normal", "serif", 400, FontStyle::Normal),
        // The `font` shorthand expands to the longhands.
        (
            "font: italic bold 16px serif",
            "serif",
            700,
            FontStyle::Italic,
        ),
        (
            "font-family: Verdana; font-weight: 600; font-style: oblique",
            // Unquoted names parse as case-insensitive keywords, stored
            // lowercased; quoted names keep their case (see above).
            "verdana",
            600,
            FontStyle::Oblique,
        ),
    ];
    for (attr, family, weight, style) in cases {
        let s = styled(attr);
        assert_eq!(css_font_family(&s), family, "family for {attr:?}");
        assert_eq!(css_font_weight(&s), weight, "weight for {attr:?}");
        assert_eq!(css_font_style(&s), style, "style for {attr:?}");
    }
}
