//! `display: list-item` marker boxes (CSS Lists 3).
//!
//! The principal box of a list item stays a plain block box; the marker
//! is laid out separately:
//! - `list-style-position: outside` (the default): the marker hangs in
//!   the inline-start margin area. It is never a flow child — no flow
//!   pass sees it — and [`position_markers`] (run at the end of
//!   `layout_forest`, after the deferred-shift flush, so every rect is
//!   final) places it against the principal box's border box,
//!   baseline-aligned with the item's first line box when the item has
//!   inline content.
//! - `list-style-position: inside`: the marker becomes the first inline
//!   box of the principal box, so it flows with the content.
//!
//! Marker text comes from `list-style-type` (`disc`/`circle`/`square`/
//! `decimal` plus the other CSS counter styles; `none` suppresses the
//! marker). `list-style-image` is never loaded: per CSS 2.1 a missing
//! image falls back to the `list-style-type` marker, which is what this
//! engine always renders.
//!
//! `::marker` declarations come from [`Styler::marker_style_for`];
//! anything the pseudo-element does not set inherits from the list
//! item (the originating element), per CSS Lists 3.

use bowser_css::{ComputedStyle, Styler, props};
use bowser_dom::{Document, NodeId, NodeKind};
use std::rc::Rc;

use super::boxes::{Forest, MarkerLayout, apply_text_transform};
use super::style::{Display, StyleData, font_family_name, resolve_font_size, resolve_line_height};
use super::{BoxType, FontMetrics, Rect};

/// The marker spec for one list item, built at box-construction time.
pub(crate) struct MarkerSpec {
    /// Marker text ("•", "1.", …).
    pub text: String,
    /// True for `list-style-position: inside`.
    pub position_inside: bool,
    /// `::marker` computed style when a `::marker` rule matched
    /// (`None` = the marker inherits the list item's style).
    pub marker_style: Option<ComputedStyle>,
}

/// 1-based ordinal of the `li` among its `li` siblings, per the HTML
/// list-item numbering rules: an explicit `value` attribute wins; a
/// preceding sibling's `value` seeds the count that follows it; the
/// parent list's `start` attribute seeds the first item (default 1).
/// Non-`li` siblings never affect the count, so nested lists count
/// independently.
pub(crate) fn list_item_ordinal(doc: &Document, li: NodeId) -> i64 {
    let is_li = |n: NodeId| matches!(doc.kind(n), NodeKind::Element(d) if d.tag_name.eq_ignore_ascii_case("li"));
    let value_of = |n: NodeId| {
        doc.get_attribute(n, "value")
            .and_then(|s| s.trim().parse::<i64>().ok())
    };
    if let Some(v) = value_of(li) {
        return v;
    }
    // Walk back: the nearest preceding `li` with a `value` seeds the
    // count; `dist` is the number of list-item steps from the seed to
    // this item (with no `value` anywhere, the `start` attribute seeds
    // the first item and `dist` counts the preceding siblings).
    let mut dist: i64 = 0;
    let mut sib = doc.previous_sibling(li);
    while let Some(s) = sib {
        if is_li(s) {
            dist = dist.saturating_add(1);
            if let Some(v) = value_of(s) {
                return v.saturating_add(dist);
            }
        }
        sib = doc.previous_sibling(s);
    }
    let start = doc
        .parent(li)
        .and_then(|p| doc.get_attribute(p, "start"))
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or(1);
    start.saturating_add(dist)
}

/// Marker text for a (lowercased) `list-style-type`: `None` for `none`
/// (no marker). Alphabetic systems clamp to a minimum of 1; Armenian
/// and Georgian degrade to decimal (no glyph tables here); unknown
/// types degrade to the disc rendering.
pub(crate) fn marker_text(list_style_type: &str, ordinal: i64) -> Option<String> {
    // Zero-alloc ASCII case-insensitive matching instead of lowering the
    // input at the call site.
    let t = list_style_type;
    if t.eq_ignore_ascii_case("none") {
        None
    } else if t.eq_ignore_ascii_case("disc") {
        Some("•".to_string())
    } else if t.eq_ignore_ascii_case("circle") {
        Some("○".to_string())
    } else if t.eq_ignore_ascii_case("square") {
        Some("▪".to_string())
    } else if t.eq_ignore_ascii_case("decimal") {
        Some(format!("{ordinal}."))
    } else if t.eq_ignore_ascii_case("decimal-leading-zero") {
        Some(format!("{ordinal:02}."))
    } else if t.eq_ignore_ascii_case("lower-roman") {
        Some(format!("{}.", to_roman(ordinal.max(1)).to_lowercase()))
    } else if t.eq_ignore_ascii_case("upper-roman") {
        Some(format!("{}.", to_roman(ordinal.max(1))))
    } else if t.eq_ignore_ascii_case("lower-alpha") || t.eq_ignore_ascii_case("lower-latin") {
        Some(format!("{}.", to_alpha(ordinal.max(1)).to_lowercase()))
    } else if t.eq_ignore_ascii_case("upper-alpha") || t.eq_ignore_ascii_case("upper-latin") {
        Some(format!("{}.", to_alpha(ordinal.max(1))))
    } else if t.eq_ignore_ascii_case("lower-greek") {
        Some(format!("{}.", to_greek(ordinal.max(1))))
    } else if t.eq_ignore_ascii_case("armenian") || t.eq_ignore_ascii_case("georgian") {
        Some(format!("{ordinal}."))
    } else {
        Some("•".to_string())
    }
}

/// Uppercase Roman numeral for 1..=3999; anything else degrades to the
/// decimal rendering (also bounds the output: no "M" repetition attack
/// from a hostile `value` attribute).
fn to_roman(n: i64) -> String {
    if !(1..=3999).contains(&n) {
        return n.to_string();
    }
    const TABLE: &[(i64, &str)] = &[
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut n = n;
    let mut out = String::new();
    for &(v, s) in TABLE {
        while n >= v {
            out.push_str(s);
            n -= v;
        }
    }
    out
}

/// Uppercase alphabetic counter: 1 -> A, 26 -> Z, 27 -> AA, ….
fn to_alpha(n: i64) -> String {
    let mut n = n;
    let mut out = Vec::new();
    while n > 0 {
        n -= 1;
        out.push((b'A' + (n % 26) as u8) as char);
        n /= 26;
    }
    out.iter().rev().collect()
}

/// Lowercase Greek counter: 1 -> α, …, 24 -> ω, 25 -> αα, ….
fn to_greek(n: i64) -> String {
    const GREEK: &[char] = &[
        'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ',
        'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
    ];
    let mut n = n;
    let mut out = Vec::new();
    while n > 0 {
        n -= 1;
        out.push(GREEK[(n % 24) as usize]);
        n /= 24;
    }
    out.iter().rev().collect()
}

/// Build the marker spec for a list-item element: its computed
/// list-style properties plus any matching `::marker` style. `None`
/// when `list-style-type: none` suppresses the marker.
pub(crate) fn spec_for(
    doc: &Document,
    styler: &Styler,
    dom: NodeId,
    style: &ComputedStyle,
) -> Option<MarkerSpec> {
    let list_type = style
        .get_str_by_index(props::LIST_STYLE_TYPE)
        .unwrap_or("disc");
    let ordinal = list_item_ordinal(doc, dom);
    let text = marker_text(list_type, ordinal)?;
    let marker_style = styler.marker_style_for(dom);
    let position_inside = style
        .get_str_by_index(props::LIST_STYLE_POSITION)
        .is_some_and(|p| p.eq_ignore_ascii_case("inside"));
    Some(MarkerSpec {
        text,
        position_inside,
        marker_style,
    })
}

/// Attach the marker to the list-item box `idx` (already built with
/// its style and `StyleData`).
pub(crate) fn attach(
    forest: &mut Forest,
    idx: usize,
    spec: MarkerSpec,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    fonts: &dyn FontMetrics,
) {
    let li_fs = forest.nodes[idx].data.font_size;
    let marker_fs = match &spec.marker_style {
        Some(ms) => resolve_font_size(
            ms.get_by_index(props::FONT_SIZE),
            li_fs,
            root_fs,
            vp_w,
            vp_h,
        ),
        None => li_fs,
    };
    if spec.position_inside {
        attach_inside(forest, idx, spec, marker_fs, root_fs, vp_w, vp_h, fonts);
        return;
    }
    let rtl = forest.nodes[idx].data.rtl;
    forest.nodes[idx].marker = Some(Box::new(MarkerLayout {
        text: spec.text,
        font_size: marker_fs,
        style: spec.marker_style,
        rtl,
        rect: None,
    }));
}

/// `list-style-position: inside`: the marker becomes the first inline
/// box of the principal box — an inline wrapper carrying the
/// `::marker` style (so its color/font apply when a rule matched)
/// around the marker text plus a separating space.
// Mirrors `attach`'s argument set plus the resolved marker font
// size: the established shape for this module's attach path, hence
// the argument count.
#[allow(clippy::too_many_arguments)]
fn attach_inside(
    forest: &mut Forest,
    idx: usize,
    spec: MarkerSpec,
    marker_fs: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    fonts: &dyn FontMetrics,
) {
    let mut data = StyleData::inherit(&forest.nodes[idx].data);
    data.display = Display::Inline;
    data.font_size = marker_fs;
    data.line_height = match &spec.marker_style {
        Some(ms) => resolve_line_height(
            ms.get_by_index(props::LINE_HEIGHT),
            marker_fs,
            root_fs,
            vp_w,
            vp_h,
            fonts,
            font_family_name(ms),
        ),
        None => forest.nodes[idx].data.line_height,
    };
    let inline_idx = forest.add_node(
        BoxType::Inline,
        spec.marker_style.map(Rc::new),
        data,
        None,
        Some(idx),
    );
    let text_data = StyleData::inherit(&forest.nodes[inline_idx].data);
    let text =
        apply_text_transform(&format!("{} ", spec.text), text_data.text_transform).into_owned();
    forest.add_node(
        BoxType::Text,
        None,
        text_data,
        Some(std::sync::Arc::from(text)),
        Some(inline_idx),
    );
    // The marker must precede the item's own content: `add_node`
    // appends, so rotate the marker node to the front. `rotate_right`
    // shifts the span once, where remove+insert memmoved it twice.
    let children = &mut forest.nodes[idx].children;
    if let Some(pos) = children.iter().rposition(|&c| c == inline_idx) {
        children[..=pos].rotate_right(1);
    }
}

/// Place every outside marker: rects are final when this runs (called
/// at the end of `layout_forest`, after the deferred-shift flush), so
/// markers compute directly in absolute page coordinates.
///
/// The marker sits in the inline-start margin area (inline-end for
/// rtl), right-aligned against the principal box's border edge with a
/// quarter-em gap, baseline-aligned with the item's first line box
/// (top-aligned with the content box when the item has no line
/// boxes).
pub(crate) fn position_markers(forest: &mut Forest, fonts: &dyn FontMetrics) {
    for i in 0..forest.nodes.len() {
        if forest.nodes[i].marker.is_none() {
            continue;
        }
        let (text, fs, rtl, family) = {
            let m = forest.nodes[i].marker.as_ref().expect("marker checked");
            // The marker's own `::marker` style wins when one matched;
            // otherwise it inherits the list item's family.
            let family = match m.style.as_ref() {
                Some(style) => fonts.intern_family(super::layout::css_font_family_name(style)),
                None => super::layout::family_id_for(forest, i, fonts),
            };
            (m.text.clone(), m.font_size, m.rtl, family)
        };
        let n = &forest.nodes[i];
        let w = fonts.advance(&text, fs, family);
        let lh = fonts.line_height(fs, family);
        let gap = 0.25 * fs;
        let x = if rtl {
            n.rect.x + n.rect.w + gap
        } else {
            n.rect.x - gap - w
        };
        // The 0.8/0.2 ascent/descent split matches the inline engine's
        // placeholder metrics (see the crate docs).
        let y = match first_baseline(forest, i) {
            Some(baseline) => baseline - 0.8 * lh,
            None => n.rect.y + n.metrics.bt + n.metrics.pt,
        };
        forest.nodes[i]
            .marker
            .as_mut()
            .expect("marker checked")
            .rect = Some(Rect { x, y, w, h: lh });
    }
}

/// Baseline of the first laid-out line fragment in the subtree, in
/// absolute page coordinates (`None` when the subtree laid out no
/// text, e.g. an empty list item).
fn first_baseline(forest: &Forest, idx: usize) -> Option<f32> {
    let mut stack = vec![idx];
    while let Some(i) = stack.pop() {
        let node = &forest.nodes[i];
        if let Some(f) = node.fragments.first() {
            return Some(f.baseline);
        }
        stack.extend(node.children.iter().rev().copied());
    }
    None
}
