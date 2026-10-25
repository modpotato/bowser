//! The cascade: from matching declarations to computed styles.
//!
//! [`compute_style`] collects every declaration whose selector matches the
//! node (and whose `@media` matches the viewport), sorts them by
//! (origin/importance, specificity, source order), and applies the winners.
//! Inline `style=""` attributes count as author origin with the highest
//! non-important precedence. Custom properties (`--*`) cascade and inherit
//! like normal properties, and `var()` references are substituted against
//! them (case-sensitive names, with fallback and cycle handling). The
//! inherited-property set takes values from ancestors; anything else left
//! unspecified stays absent.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use bowser_dom::{Document, ElementData, Namespace, NodeId, NodeKind, QuirksMode};

use crate::cbprops::Filter;
use crate::matching::{
    ElementState, FxBuildHasher, FxHashMap, FxHashSet, MatchScratch, RuleIndex, SelectorRef,
    SiblingCache, SubjectAttrs, matches_cascade, matches_pseudo_element,
};
use crate::parse::{
    Declaration, Stylesheet, fold_aspect_ratio, parse_declarations, parse_slash_aware_value,
    parse_stylesheet, split_top_comma, split_top_slash, split_top_slash_font,
};
use crate::propatom::{PropertyAtom, PropertyName};
use crate::selector::{Combinator, PseudoClass, Selector, specificity};
use crate::token::Token;
use crate::value::MAX_MATH_DEPTH;
use crate::value::{
    CalcSizeBasis, ClipPath, Color, CssValue, IfBranch, LengthUnit, MathContext, MathExpr,
    MathValue, RepeatMode, Shadow, TransformOrigin, angle_unit, css_source_contains_var,
    parse_calc_size_basis, parse_value, parse_value_preserving_case, system_font_values,
};

/// Viewport dimensions in px, used for `@media` evaluation.
#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub w: f32,
    pub h: f32,
}

/// The computed style of one element: cascade winners with `var()`
/// resolved and inheritance applied.
///
/// Sparse storage (wave 50): a full-width slot array costs
/// `PROP_COUNT * 8 = 1480` heap bytes per uniquely-styled element, yet
/// most slots just hold the shared initial value (see [`initials`]).
/// Instead `delta` stores only the slots whose computed value deviates
/// from the initial value, as `(index, value)` pairs sorted by index:
/// a plain element deviates in a handful of slots (its UA `display`,
/// any declared or inherited non-initial values), so the heap cost is
/// ~16 bytes per deviating slot and zero allocation at all for an
/// all-initial element. The observable mapping is unchanged:
/// [`get_by_index`](ComputedStyle::get_by_index) binary-searches
/// `delta` and falls back to the shared initial value on a miss, which
/// is exactly what the old full array held in that slot. Indices are
/// never renumbered — a delta entry's `u8` is the same
/// [`COMPUTED_PROPS`] position the full array used. Custom properties
/// (`--*`) keep dynamic names and stay in a map. The map is boxed and
/// `None` when the element declares no custom properties.
#[derive(Clone, Debug)]
pub struct ComputedStyle {
    delta: Box<[(u8, Arc<CssValue>)]>,
    /// Custom properties (`--*`), boxed and `None` when empty: most
    /// elements declare none, so the common case costs one word.
    /// (`clippy::box_collection` is allowed: the box keeps
    /// `ComputedStyle` small for the overwhelmingly common empty
    /// case — an inline map would cost 48 bytes per style.)
    #[allow(clippy::box_collection)]
    custom: Option<Box<FxHashMap<String, CssValue>>>,
}

impl ComputedStyle {
    /// Build a style from a fully computed, initial-filled slot array
    /// (the scratch form `Styler::compute` fills): slots holding the
    /// shared initial `Arc` — or no value, which the initial-filled
    /// invariant means the property has no initial value — are dropped
    /// and re-materialize from [`initials`] on read; every other slot
    /// becomes one delta entry. Callers must pass initial-filled slots
    /// (every `None`-computed property already replaced by its initial
    /// value), as `compute` and `marker_style_for` guarantee.
    #[allow(clippy::box_collection)]
    fn from_slots(
        props: [Option<Arc<CssValue>>; PROP_COUNT],
        custom: Option<Box<FxHashMap<String, CssValue>>>,
    ) -> Self {
        // Exact-size the delta up front. `into_boxed_slice` calls
        // `shrink_to_fit`, so the old `Vec::new()` + push growth ramp
        // cost up to 3 allocations per styled element (grow, grow,
        // shrink); counting the deviating slots first allocates exactly
        // once — the boxed slice itself, which is unavoidable — and
        // zero times for an all-initial element.
        //
        // A thread-local pooled buffer cannot do better here, which is
        // why there isn't one: `into_boxed_slice` moves the buffer's
        // allocation into the style, so a pooled `Vec` would hand out a
        // zero-capacity buffer on every take — the pool could never
        // retain capacity. (Styling is recursive, so a pool would also
        // need reentrancy guards, all for no benefit.)
        let mut deviating = 0usize;
        for (idx, value) in props.iter().enumerate() {
            if let Some(v) = value
                && !arc_is_initial(idx as u8, v)
            {
                deviating += 1;
            }
        }
        let mut delta: Vec<(u8, Arc<CssValue>)> = Vec::with_capacity(deviating);
        for (idx, value) in props.into_iter().enumerate() {
            if let Some(v) = value
                && !arc_is_initial(idx as u8, &v)
            {
                delta.push((idx as u8, v));
            }
        }
        debug_assert_eq!(delta.len(), deviating);
        ComputedStyle {
            delta: delta.into_boxed_slice(),
            custom,
        }
    }

    /// Set one slot to `value` (`None`, or the shared initial value,
    /// clears the slot back to its initial fallback). Used by
    /// `marker_style_for`, which starts from a finished style and
    /// overwrites the few `::marker` properties. The delta stays
    /// sorted; the rebuild is O(deviating slots), not O(PROP_COUNT).
    fn set_slot(&mut self, idx: u8, value: Option<Arc<CssValue>>) {
        let mut delta: Vec<(u8, Arc<CssValue>)> = std::mem::take(&mut self.delta).into_vec();
        match delta.binary_search_by_key(&idx, |(i, _)| *i) {
            Ok(pos) => {
                // `None` always clears: `is_initial_form` is false for
                // `None` on a property whose initial is `Some` (that
                // distinction keeps a memo's unset slot apart from an
                // initial-filled one), but here `None` is the caller's
                // explicit "clear this slot", so it must remove the
                // entry instead of leaving the stale value in place.
                if value.is_none() || is_initial_form(idx, &value) {
                    delta.remove(pos);
                } else if let Some(v) = value {
                    delta[pos].1 = v;
                }
            }
            Err(pos) => {
                if !is_initial_form(idx, &value)
                    && let Some(v) = value
                {
                    delta.insert(pos, (idx, v));
                }
            }
        }
        self.delta = delta.into_boxed_slice();
    }

    /// Number of deviating slots actually stored (test/evidence hook:
    /// pins that the sparse representation stays sparse).
    #[cfg(test)]
    fn stored_slots(&self) -> usize {
        self.delta.len()
    }
}

/// Computed `scroll-snap-type` (CSS Scroll Snap 1): whether the box is a
/// snap container and on which axes, with the strictness folded in.
///
/// The logical axes resolve against the box's writing mode the way the
/// snap engine consumes them: in `horizontal-tb`, `block` snaps like
/// `y` and `inline` like `x`; in the vertical writing modes
/// (`vertical-rl` / `vertical-lr`) the inline axis is vertical and the
/// block axis horizontal, so they swap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapType {
    None,
    ProximityX,
    ProximityY,
    ProximityBoth,
    MandatoryX,
    MandatoryY,
    MandatoryBoth,
}

/// Computed `scroll-snap-stop` (CSS Scroll Snap 1): whether a fling may
/// pass this snap area (`Normal`) or must stop at it (`Always`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapStop {
    Normal,
    Always,
}

/// One axis of computed `scroll-snap-align` (CSS Scroll Snap 1): how the
/// snap area aligns to the snapport on that axis, or `None` when the
/// box is not a snap area on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapAlign {
    None,
    Start,
    End,
    Center,
}

impl ComputedStyle {
    /// The computed value of `property`, or `None` when unspecified and
    /// not inherited (or invalid at computed-value time).
    pub fn get(&self, property: &str) -> Option<&CssValue> {
        match PropertyAtom::lookup(property) {
            Some(atom) if atom.is_computed() => self.get_by_index(atom.index()),
            _ => self.custom.as_ref().and_then(|m| m.get(property)),
        }
    }

    /// The computed value of the property with const index `index` (one
    /// of [`props`]), or `None` when unspecified and not inherited.
    /// A binary search over the deviating slots (typically < 16 of
    /// 180): no string hash, no custom-property fallback. A miss means
    /// the slot holds the property's initial value (which is what the
    /// old full array stored there); a miss on a property with no
    /// initial value yields `None`, exactly as before. Every `props::*`
    /// index is `< COMPUTED_PROPS.len()`, so the fallback read cannot
    /// panic.
    pub fn get_by_index(&self, index: u8) -> Option<&CssValue> {
        match self.delta.binary_search_by_key(&index, |(i, _)| *i) {
            Ok(pos) => Some(&self.delta[pos].1),
            Err(_) => initial_ref(index),
        }
    }

    /// The value as a string slice for keyword/string values.
    pub fn get_str(&self, property: &str) -> Option<&str> {
        match self.get(property) {
            Some(CssValue::Keyword(s)) => Some(s.as_str()),
            Some(CssValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// [`get_str`](Self::get_str) by const property index: no hashing.
    pub fn get_str_by_index(&self, index: u8) -> Option<&str> {
        match self.get_by_index(index) {
            Some(CssValue::Keyword(s)) => Some(s.as_str()),
            Some(CssValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Computed `scroll-snap-type` (CSS Scroll Snap 1), or [`SnapType::None`]
    /// when unspecified. `block`/`inline` resolve to the physical axis the
    /// snap engine works in, against this box's own computed
    /// `writing-mode` (the same logical-to-physical resolution
    /// `apply_logical_mapping` applies to the box properties): in
    /// `horizontal-tb` `inline` is `x` and `block` is `y`; in
    /// `vertical-rl` / `vertical-lr` the inline axis runs vertically, so
    /// `inline` is `y` and `block` is `x`. The physical keywords (`x`,
    /// `y`, `both`) never swap. A lone axis keyword means `proximity`.
    pub fn scroll_snap_type(&self) -> SnapType {
        let axis_kw = match self.get_by_index(props::SCROLL_SNAP_TYPE) {
            Some(CssValue::List(items)) => match items.first() {
                Some(CssValue::Keyword(k)) => Some(k.as_str()),
                _ => None,
            },
            Some(CssValue::Keyword(k)) => Some(k.as_str()),
            _ => None,
        };
        let strict = match self.get_by_index(props::SCROLL_SNAP_TYPE) {
            Some(CssValue::List(items)) => items
                .get(1)
                .is_some_and(|v| matches!(v, CssValue::Keyword(k) if k == "mandatory")),
            _ => false,
        };
        // The physical axis a logical keyword names, per writing mode
        // (mirrors `writing_mode_of`'s treatment of the vertical modes).
        let vertical = matches!(
            self.get_str_by_index(props::WRITING_MODE),
            Some(k) if k.eq_ignore_ascii_case("vertical-rl")
                || k.eq_ignore_ascii_case("vertical-lr")
        );
        let (inline_axis, block_axis) = if vertical { ("y", "x") } else { ("x", "y") };
        let axis = match axis_kw {
            Some("inline") => Some(inline_axis),
            Some("block") => Some(block_axis),
            other => other,
        };
        match (axis, strict) {
            (Some("x"), true) => SnapType::MandatoryX,
            (Some("x"), _) => SnapType::ProximityX,
            (Some("y"), true) => SnapType::MandatoryY,
            (Some("y"), _) => SnapType::ProximityY,
            (Some("both"), true) => SnapType::MandatoryBoth,
            (Some("both"), _) => SnapType::ProximityBoth,
            _ => SnapType::None,
        }
    }

    /// Computed `scroll-snap-stop` (CSS Scroll Snap 1), or
    /// [`SnapStop::Normal`] when unspecified.
    pub fn scroll_snap_stop(&self) -> SnapStop {
        match self.get_by_index(props::SCROLL_SNAP_STOP) {
            Some(CssValue::Keyword(k)) if k == "always" => SnapStop::Always,
            _ => SnapStop::Normal,
        }
    }

    /// Computed `scroll-snap-align` (CSS Scroll Snap 1) as
    /// `(block, inline)`: the first keyword is the block axis, the second
    /// the inline axis. A single specified value repeats on both axes;
    /// unspecified or `none` reads as [`SnapAlign::None`] on both.
    pub fn scroll_snap_align(&self) -> (SnapAlign, SnapAlign) {
        fn one(v: &CssValue) -> SnapAlign {
            match v {
                CssValue::Keyword(k) => match k.as_str() {
                    "start" => SnapAlign::Start,
                    "end" => SnapAlign::End,
                    "center" => SnapAlign::Center,
                    _ => SnapAlign::None,
                },
                _ => SnapAlign::None,
            }
        }
        match self.get_by_index(props::SCROLL_SNAP_ALIGN) {
            Some(CssValue::List(items)) => {
                let a = items.first().map(one).unwrap_or(SnapAlign::None);
                let b = items.get(1).map(one).unwrap_or(a);
                (a, b)
            }
            Some(v) => {
                let a = one(v);
                (a, a)
            }
            None => (SnapAlign::None, SnapAlign::None),
        }
    }
}

/// Compute the style of `node`: `ua` is the user-agent stylesheet, `author`
/// the author stylesheets in document order.
///
/// One-shot wrapper around [`Styler`]: builds a fresh styler, styles the
/// node's ancestor chain root-first (so every ancestor's cascade and
/// inheritance results are memoized before the target is styled), styles
/// the target, and drops the memoization caches. The root-first pass is
/// the fast path; a direct cold [`Styler::style_for`] on a deep node also
/// works — [`Styler::finish_prop`] drives the ancestor climb iteratively,
/// so no caller needs to pre-warm the chain (B3), though root-first
/// styling stays linear while repeated one-shot calls re-walk the
/// ancestor chain each time. For styling many nodes of one document,
/// build a single [`Styler`] and call [`Styler::style_for`] instead.
///
/// Dynamic pseudo-classes match against [`ElementState::default()`]
/// here; for interactive state, build a [`Styler`] and use
/// [`Styler::set_element_state`].
pub fn compute_style(
    doc: &Document,
    node: NodeId,
    author: &[&Stylesheet],
    ua: &Stylesheet,
    viewport: &Viewport,
) -> ComputedStyle {
    let styler = Styler::new(doc, author, ua, *viewport);
    // Ancestor chain, root-first, routed through `style_parent` so a
    // shadow child's chain includes the host: the host gets styled
    // before shadow children, keeping the `parent_ready` warm path.
    // Iterative: no recursion per ancestor.
    let mut chain = Vec::new();
    let mut cursor = Some(node);
    while let Some(n) = cursor {
        chain.push(n);
        cursor = styler.style_parent(n);
    }
    let mut out = None;
    for n in chain.into_iter().rev() {
        let s = styler.style_for(n);
        if n == node {
            out = Some(s);
        }
    }
    out.expect("node is always in its own ancestor chain")
}

/// The UA stylesheet gives `table` its spec display value (`table`);
/// layout implements table layout for it (see `bowser_layout`).
const UA_CSS: &str = r#"
html, body, div, p, h1, h2, h3, h4, h5, h6, ul, ol,
blockquote, pre, hr, form { display: block; }
li { display: list-item; }
ul { list-style-type: disc; }
ol { list-style-type: decimal; }
table { display: table; }
table { border-color: gray; }
table { border-spacing: 2px; }
/* HTML rendering: a `table` element's `width`/`height` name the border
   box (unlike `display: table` on other elements, which stays
   content-box by default). Overridable by an explicit `box-sizing`. */
table { box-sizing: border-box; }
td, th { padding: 1px; }
/* WHATWG HTML rendering §15.3.8: row groups and direct table rows
   default to `vertical-align: middle`, and rows/cells inherit it, so
   a `tr` author rule (or a row/section `valign` hint, which beats
   this UA rule) propagates down to the cells. A direct `td, th {
   vertical-align: middle; }` rule would give the same answer in a
   plain table but block that inheritance. Unlike table cells,
   `display: table-cell` on non-table elements keeps the `baseline`
   initial value; WPT css-tables/tentative/baseline-td.html
   distinguishes the two. */
thead, tbody, tfoot, table > tr { vertical-align: middle; }
tr, td, th { vertical-align: inherit; }
th { font-weight: bolder; text-align: center; }
caption { text-align: center; }
thead, tbody, tfoot, tr { border-color: inherit; }
tr { display: table-row; }
td, th { display: table-cell; }
thead, tbody, tfoot { display: table-row-group; }
col { display: table-column; }
colgroup { display: table-column-group; }
caption { display: table-caption; }
h1 { font-size: 2em; margin: 0.67em 0; }
h2 { font-size: 1.5em; margin: 0.83em 0; }
h3 { font-size: 1.17em; margin: 1em 0; }
h4 { font-size: 1em; margin: 1.33em 0; }
h5 { font-size: 0.83em; margin: 1.67em 0; }
h6 { font-size: 0.67em; margin: 2.33em 0; }
p { margin: 1em 0; }
ul, ol { padding-left: 40px; }
span, a, b, i, em, strong, code { display: inline; }
head, script, style, title, meta, link { display: none; }
img { display: inline-block; }
button, input, select, textarea { display: inline-block; }
pre { white-space: pre; }
"#;

/// The default user-agent stylesheet (HTML display defaults).
pub fn default_ua_stylesheet() -> Stylesheet {
    parse_stylesheet(UA_CSS)
}

/// Which sides of a table's outer frame the `frame` attribute shows.
#[derive(Clone, Copy)]
struct FrameSides {
    top: bool,
    right: bool,
    bottom: bool,
    left: bool,
}

/// Valid `rules` attribute values (Blink's `HTMLTableElement` rules
/// enum; anything else is "unset" and maps nothing).
#[derive(Clone, Copy, PartialEq, Eq)]
enum TableRules {
    None,
    Groups,
    Rows,
    Cols,
    All,
}

/// Parse an HTML non-negative integer (HTML's "rules for parsing
/// non-negative integers"): skip leading whitespace and an optional
/// `+`, then take the digit run. `None` is a parse error.
fn html_nonneg_int(value: &str) -> Option<u32> {
    let s = value.trim_start();
    let s = s.strip_prefix('+').unwrap_or(s);
    let mut n: u64 = 0;
    let mut any = false;
    for c in s.chars() {
        match c.to_digit(10) {
            Some(d) => {
                any = true;
                n = n.saturating_mul(10).saturating_add(u64::from(d));
            }
            None => break,
        }
    }
    any.then(|| n.min(u64::from(u32::MAX)) as u32)
}

/// Blink's `ParseBorderWidthAttribute`: the `border` attribute's
/// width in px; a parse error (e.g. `border="yes"`) yields 1.
fn table_border_width(value: &str) -> u32 {
    html_nonneg_int(value).unwrap_or(1)
}

/// Blink's `GetBordersFromFrameAttributeValue`: the sides the
/// `frame` attribute shows, or `None` for an invalid value.
fn frame_sides(value: &str) -> Option<FrameSides> {
    let v = value.trim().to_ascii_lowercase();
    let (mut top, mut right, mut bottom, mut left) = (false, false, false, false);
    match v.as_str() {
        "void" => {}
        "above" => top = true,
        "below" => bottom = true,
        "hsides" => {
            top = true;
            bottom = true;
        }
        "vsides" => {
            left = true;
            right = true;
        }
        "lhs" => left = true,
        "rhs" => right = true,
        "box" | "border" => {
            top = true;
            right = true;
            bottom = true;
            left = true;
        }
        _ => return None,
    }
    Some(FrameSides {
        top,
        right,
        bottom,
        left,
    })
}

/// The `rules` attribute's value, or `None` when unset/invalid.
fn table_rules(value: &str) -> Option<TableRules> {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" => Some(TableRules::None),
        "groups" => Some(TableRules::Groups),
        "rows" => Some(TableRules::Rows),
        "cols" => Some(TableRules::Cols),
        "all" => Some(TableRules::All),
        _ => None,
    }
}

/// An HTML dimension value (`width`/`height` attributes): the
/// WHATWG HTML §2.3.4.4 dimension grammar in strict full-consumption
/// form — ASCII whitespace trimmed from both ends, then one or more
/// ASCII digits, an optional single fraction (`.` followed by one or
/// more digits), and an optional single `%` suffix; anything else
/// maps to nothing. That shape is what §2.3.4.4 itself requires of
/// the number (a leading digit, no sign, no exponent, a fraction
/// only with digits after the dot), tightened here to full
/// consumption because the result becomes a CSS declaration: the
/// lenient prefix form of the parsing algorithm would silently turn
/// `100abc` into `100px` and `100 px` into `100px`, which are not
/// dimensions at all. A trailing `%` makes a percentage, anything
/// else a px length.
fn html_dimension(value: &str) -> Option<String> {
    let s = value.trim_matches(|c| matches!(c, '\t' | '\n' | '\x0C' | '\r' | ' '));
    let b = s.as_bytes();
    let mut i = 0;
    let int_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == int_start {
        return None;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == frac_start {
            return None;
        }
    }
    let num_end = i;
    let is_pct = if i < b.len() && b[i] == b'%' {
        i += 1;
        true
    } else {
        false
    };
    if i != b.len() {
        return None;
    }
    let num: f32 = s[..num_end].parse().ok()?;
    if is_pct {
        Some(format!("{num}%"))
    } else {
        Some(format!("{num}px"))
    }
}

/// Append one declaration to a hint declaration block.
fn hint_decl(text: &mut String, prop: &str, val: &str) {
    text.push_str(prop);
    text.push_str(": ");
    text.push_str(val);
    text.push_str("; ");
}

/// Append a `border-{side}-{kind}` declaration for all four sides.
fn hint_border4(text: &mut String, kind: &str, val: &str) {
    for side in ["top", "right", "bottom", "left"] {
        hint_decl(text, &format!("border-{side}-{kind}"), val);
    }
}

/// HTML presentational hints for table markup, as a declaration list
/// in cascade order (later declarations win among hints). The model
/// is Blink's `HTMLTableElement` (`CollectStyleForPresentationAttribute`,
/// `AdditionalPresentationAttributeStyle`, `GetCellBorders`,
/// `AdditionalGroupStyle`) ordered the way the css-tables-3
/// Appendix A sheet orders its rules — frame width first, the
/// `border` width (when nonzero) after it, the `border` outset/gray
/// rule before the per-side frame styles — which is the ordering the
/// WPT `html-to-css-mapping-2.html` expectations cite:
///
/// - `border=N` (parse error → 1, so `border="yes"` is 1px) sets the
///   table's border widths; with no `frame` it also yields the
///   `outset` additional style (or `solid` when `bordercolor` is
///   present) and the appendix's gray border color.
/// - A valid `rules` enables `border-collapse: collapse` and, when
///   the table has no frame and no nonzero border, the `hidden`
///   additional style; with a valid `frame` too, the appendix's
///   rules rule is what paints the table's border color black (the
///   frame rule overrides its gray in that combination).
/// - `frame` sets width `thin` and, per the appendix's later rules,
///   style `outset` on the shown sides and `hidden` on the rest.
/// - Cells take Blink's shared cell style: 1px `inset` gray for a
///   nonzero border (`solid`, color `inherit`, with `bordercolor`),
///   overridden per `rules` (`all` → `solid`; `rows`/`cols` → solid
///   on that axis, `none` on the other; `none` → no border;
///   `groups` → nothing), plus the padding for an explicit
///   `cellpadding` attribute. Row/column groups take the
///   `rules=groups` borders.
///
/// The declarations are injected by [`Styler::candidates`] as
/// author-origin candidates at zero specificity ahead of every
/// author sheet, per the HTML standard's presentational-hints
/// cascade level: they beat UA declarations and lose to any author
/// declaration.
fn table_presentational_hints(doc: &Document, node: NodeId) -> Vec<Declaration> {
    let NodeKind::Element(data) = doc.kind(node) else {
        return Vec::new();
    };
    let tag = data.tag_name.as_str();
    let mut text = String::new();
    if tag.eq_ignore_ascii_case("table") {
        table_element_hints(doc, node, &mut text);
    } else if tag.eq_ignore_ascii_case("td") || tag.eq_ignore_ascii_case("th") {
        if let Some(table) = ancestor_table(doc, node) {
            cell_hints(doc, table, &mut text);
        }
        // The cell's own dimension attributes map unconditionally on
        // the element (WHATWG §15.3.8's `td`/`th` rule, like the
        // `tr`/section mappings above): only the table-derived
        // border/padding hints need an ancestor table.
        cell_dimension_hints(doc, node, &mut text);
    } else if tag.eq_ignore_ascii_case("tr") {
        // WHATWG HTML rendering § "Tables": the `tr` element's
        // `height` attribute maps to the `height` property (no
        // ignoring-zero rule, unlike `td`/`th` — see below).
        if let Some(v) = doc.get_attribute(node, "height")
            && let Some(d) = html_dimension(v)
        {
            hint_decl(&mut text, "height", &d);
        }
    } else if tag.eq_ignore_ascii_case("col") {
        col_width_hint(doc, node, &mut text);
    } else if tag.eq_ignore_ascii_case("colgroup") {
        if let Some(table) = ancestor_table(doc, node) {
            group_hints(doc, table, tag, &mut text);
        }
        col_width_hint(doc, node, &mut text);
    } else if (tag.eq_ignore_ascii_case("thead")
        || tag.eq_ignore_ascii_case("tbody")
        || tag.eq_ignore_ascii_case("tfoot"))
        && let Some(table) = ancestor_table(doc, node)
    {
        group_hints(doc, table, tag, &mut text);
    }
    // Section `height` hints, ungated on an ancestor table like the
    // `tr` height arm above (the height itself needs no table
    // attributes, unlike `group_hints`). WHATWG HTML rendering §
    // 15.3.8: the `thead`/`tbody`/`tfoot` `height` attribute maps to
    // the dimension property `height` — the plain mapping, NOT
    // "(ignoring zero)", so `height="0"` maps to `height: 0`,
    // matching the `tr` rule and unlike the `td`/`th` and `table`
    // rules where zero is ignored.
    if (tag.eq_ignore_ascii_case("thead")
        || tag.eq_ignore_ascii_case("tbody")
        || tag.eq_ignore_ascii_case("tfoot"))
        && let Some(v) = doc.get_attribute(node, "height")
        && let Some(d) = html_dimension(v)
    {
        hint_decl(&mut text, "height", &d);
    }
    // `valign` maps on the row/section/cell elements themselves. The tag
    // test is allocation-free (`eq_ignore_ascii_case`, like the checks
    // above): the old `tag.to_ascii_lowercase()` allocated a `String`
    // per element on every style pass.
    if (tag.eq_ignore_ascii_case("thead")
        || tag.eq_ignore_ascii_case("tbody")
        || tag.eq_ignore_ascii_case("tfoot")
        || tag.eq_ignore_ascii_case("tr")
        || tag.eq_ignore_ascii_case("td")
        || tag.eq_ignore_ascii_case("th"))
        && let Some(v) = doc.get_attribute(node, "valign")
    {
        // Only a valid `valign` value allocates (for the lowercase copy
        // handed to the declaration builder); anything else is free.
        let v = v.trim();
        if v.eq_ignore_ascii_case("top")
            || v.eq_ignore_ascii_case("middle")
            || v.eq_ignore_ascii_case("bottom")
            || v.eq_ignore_ascii_case("baseline")
        {
            hint_decl(&mut text, "vertical-align", &v.to_ascii_lowercase());
        }
    }
    if text.is_empty() {
        Vec::new()
    } else {
        parse_declarations(&text)
    }
}

/// HTML presentational hint for the `dir` attribute: `dir=ltr` /
/// `dir=rtl` map to the CSS `direction` property at the
/// presentational-hint cascade level (author origin, zero
/// specificity — beating UA declarations, losing to any author
/// declaration), so `<body dir="rtl">` actually flips direction.
/// `dir=auto` maps to `unicode-bidi: plaintext` per the HTML
/// standard, but that needs first-strong-character bidi resolution
/// the text engine does not have yet — documented follow-up, not
/// mapped here.
fn dir_presentational_hint(doc: &Document, node: NodeId) -> Vec<Declaration> {
    let Some(v) = doc.get_attribute(node, "dir") else {
        return Vec::new();
    };
    // Allocation-free miss path: only a recognized value parses.
    let v = v.trim();
    if v.eq_ignore_ascii_case("ltr") {
        parse_declarations("direction: ltr")
    } else if v.eq_ignore_ascii_case("rtl") {
        parse_declarations("direction: rtl")
    } else {
        Vec::new()
    }
}

/// The nearest ancestor `<table>` element of `node`, if any.
fn ancestor_table(doc: &Document, node: NodeId) -> Option<NodeId> {
    let mut cursor = doc.parent(node);
    while let Some(n) = cursor {
        if let NodeKind::Element(d) = doc.kind(n)
            && d.tag_name.eq_ignore_ascii_case("table")
        {
            return Some(n);
        }
        cursor = doc.parent(n);
    }
    None
}

/// A table's parsed mapping inputs, shared by the table, cell, and
/// group hint builders.
struct TableAttrs {
    /// `border` width in px (`None` when the attribute is absent).
    border_w: Option<u32>,
    rules: Option<TableRules>,
    frame: Option<FrameSides>,
    bordercolor: Option<String>,
}

fn table_attrs(doc: &Document, table: NodeId) -> TableAttrs {
    TableAttrs {
        border_w: doc.get_attribute(table, "border").map(table_border_width),
        rules: doc.get_attribute(table, "rules").and_then(table_rules),
        frame: doc.get_attribute(table, "frame").and_then(frame_sides),
        bordercolor: doc
            .get_attribute(table, "bordercolor")
            .filter(|v| !v.is_empty())
            .map(str::to_string),
    }
}

/// Hints for the `<table>` element itself (see
/// [`table_presentational_hints`] for the ordering).
fn table_element_hints(doc: &Document, node: NodeId, text: &mut String) {
    let attrs = table_attrs(doc, node);
    let border_w = attrs.border_w.unwrap_or(0);
    // Frame widths first; a nonzero `border` width overrides them
    // (appendix order). A zero `border` only supplies its width when
    // there is no frame to supply one.
    if attrs.frame.is_some() {
        hint_border4(text, "width", "thin");
    }
    if attrs.rules.is_some() {
        hint_decl(text, "border-collapse", "collapse");
    }
    // The appendix's rules border-color rule, as the test's comments
    // cite it: it takes effect in the frame+rules combination (case
    // 2), and the nonzero-`border` rule below overrides it with gray
    // whenever that rule applies (case 3).
    if attrs.frame.is_some() && attrs.rules.is_some() {
        hint_border4(text, "color", "black");
    }
    if attrs.border_w.is_some() && (border_w > 0 || attrs.frame.is_none()) {
        hint_border4(text, "width", &format!("{border_w}px"));
    }
    if border_w > 0 {
        // The nonzero-`border` rule (appendix `table[border=$w]`):
        // outset style and gray color on the table; the frame
        // per-side styles below override the style where the frame
        // hides a side. Blink's additional style is `solid` instead
        // of `outset` when `bordercolor` is present.
        let style = if attrs.bordercolor.is_some() {
            "solid"
        } else {
            "outset"
        };
        hint_border4(text, "style", style);
        hint_border4(text, "color", "gray");
    } else if attrs.frame.is_none() && attrs.rules.is_some() {
        // Blink's additional style: a valid `rules` with no frame and
        // no nonzero border hides the table's own borders.
        hint_border4(text, "style", "hidden");
    }
    if let Some(frame) = attrs.frame {
        // The appendix's frame style rules come last: `outset` on the
        // shown sides, `hidden` on the rest.
        for (side, shown) in [
            ("top", frame.top),
            ("right", frame.right),
            ("bottom", frame.bottom),
            ("left", frame.left),
        ] {
            hint_decl(
                text,
                &format!("border-{side}-style"),
                if shown { "outset" } else { "hidden" },
            );
        }
    }
    if let Some(color) = &attrs.bordercolor {
        hint_border4(text, "color", color);
    }
    if let Some(v) = doc.get_attribute(node, "cellspacing")
        && !v.is_empty()
        && let Some(n) = html_nonneg_int(v)
    {
        hint_decl(text, "border-spacing", &format!("{n}px"));
    }
    if let Some(v) = doc.get_attribute(node, "bgcolor")
        && !v.is_empty()
    {
        hint_decl(text, "background-color", v);
    }
    if let Some(v) = doc.get_attribute(node, "align") {
        match v.trim().to_ascii_lowercase().as_str() {
            "center" => {
                hint_decl(text, "margin-left", "auto");
                hint_decl(text, "margin-right", "auto");
            }
            "left" | "right" => hint_decl(text, "float", v.trim()),
            _ => {}
        }
    }
    if let Some(v) = doc.get_attribute(node, "width")
        && let Some(d) = html_dimension_nonzero(v)
    {
        // WHATWG HTML rendering § 15.3.8: the `table` element's
        // `width` attribute maps to the dimension property
        // (ignoring zero), so `width="0"` maps nothing.
        hint_decl(text, "width", &d);
    }
    if let Some(v) = doc.get_attribute(node, "height")
        && let Some(d) = html_dimension_nonzero(v)
    {
        // Same section: `table` `height` is also (ignoring zero).
        hint_decl(text, "height", &d);
    }
}

/// Presentational-hint mapping for `col` / `colgroup`: the `width`
/// attribute maps to the `width` property (WHATWG HTML rendering §
/// 15.3.8: "The `col` element's `width` attribute maps to the
/// dimension property `width`"). Unlike `table`, columns have no
/// `height` mapping. Note the spec wording here is the plain
/// dimension-property mapping, NOT "(ignoring zero)": `col`
/// `width="0"` maps to `width: 0`, unlike `table`/`td`/`th` where
/// zero is ignored. This is deliberate in the spec text (the
/// "(ignoring zero)" qualifier appears on the `table`, `td`, and
/// `th` rules but not on the `col`, section, or `tr` rules), so
/// `col_width_hint` intentionally keeps plain [`html_dimension`]
/// (the `colgroup` arm is the pre-existing extension of the `col`
/// rule, kept plain for consistency with `col`). Mirrors the `width`
/// arm of [`table_element_hints`] apart from that zero handling.
fn col_width_hint(doc: &Document, node: NodeId, text: &mut String) {
    if let Some(v) = doc.get_attribute(node, "width")
        && let Some(d) = html_dimension(v)
    {
        hint_decl(text, "width", &d);
    }
}

/// An HTML dimension value that is not zero: [`html_dimension`]
/// renders a zero value as `0%`/`0px`, and the HTML rendering rules
/// for `td`/`th` dimension attributes ignore zero values, so those
/// are dropped here instead of mapped.
fn html_dimension_nonzero(value: &str) -> Option<String> {
    let d = html_dimension(value)?;
    if d == "0%" || d == "0px" {
        None
    } else {
        Some(d)
    }
}

/// Hints for a cell (`td`/`th`) from its ancestor table's mapping
/// (Blink's shared cell style; see [`table_presentational_hints`]).
/// The cell's own dimension attributes live in
/// [`cell_dimension_hints`]: they map unconditionally, unlike these
/// table-derived hints.
fn cell_hints(doc: &Document, table: NodeId, text: &mut String) {
    let attrs = table_attrs(doc, table);
    let border_w = attrs.border_w.unwrap_or(0);
    let cell_color = if attrs.bordercolor.is_some() {
        "inherit"
    } else {
        "gray"
    };
    if border_w > 0 {
        hint_border4(text, "width", "1px");
        let style = if attrs.bordercolor.is_some() {
            "solid"
        } else {
            "inset"
        };
        hint_border4(text, "style", style);
        hint_border4(text, "color", cell_color);
    }
    match attrs.rules {
        Some(TableRules::All) => {
            hint_border4(text, "width", "1px");
            hint_border4(text, "style", "solid");
            hint_border4(text, "color", cell_color);
        }
        Some(TableRules::Rows) => {
            for side in ["top", "bottom"] {
                hint_decl(text, &format!("border-{side}-width"), "1px");
                hint_decl(text, &format!("border-{side}-style"), "solid");
                hint_decl(text, &format!("border-{side}-color"), cell_color);
            }
            for side in ["left", "right"] {
                hint_decl(text, &format!("border-{side}-style"), "none");
            }
        }
        Some(TableRules::Cols) => {
            for side in ["left", "right"] {
                hint_decl(text, &format!("border-{side}-width"), "1px");
                hint_decl(text, &format!("border-{side}-style"), "solid");
                hint_decl(text, &format!("border-{side}-color"), cell_color);
            }
            for side in ["top", "bottom"] {
                hint_decl(text, &format!("border-{side}-style"), "none");
            }
        }
        Some(TableRules::None) => {
            hint_border4(text, "style", "none");
        }
        // `groups` and unset rules leave the border-attribute cell
        // style alone (Blink's `kNoBorders`).
        Some(TableRules::Groups) | None => {}
    }
    // `cellpadding` when the attribute is explicitly present
    // (Blink's `padding_`): an empty value means the 1px default; a
    // non-empty unparseable value means 0. An absent attribute maps
    // nothing, so the UA `td { padding: 1px }` default applies; a
    // present attribute (even `cellpadding="0"`) emits a hint that
    // beats the UA default.
    if let Some(v) = doc.get_attribute(table, "cellpadding") {
        let padding = if v.is_empty() {
            1
        } else {
            html_loose_int(v).unwrap_or(0).max(0)
        };
        for side in ["top", "right", "bottom", "left"] {
            hint_decl(text, &format!("padding-{side}"), &format!("{padding}px"));
        }
    }
}

/// The cell's own dimension attributes, mapped unconditionally on the
/// `td`/`th` element (WHATWG HTML rendering §15.3.8): `width` and
/// `height` map to the `width`/`height` properties, ignoring zero
/// values, so `width="0"` leaves the computed width untouched (in
/// particular the UA default / table layout still applies, and CSS
/// `width: 0` still works). Called from the `td`/`th` arm of
/// [`table_presentational_hints`] regardless of an ancestor table;
/// only the table-derived border/padding hints in [`cell_hints`]
/// need one.
fn cell_dimension_hints(doc: &Document, node: NodeId, text: &mut String) {
    if let Some(v) = doc.get_attribute(node, "width")
        && let Some(d) = html_dimension_nonzero(v)
    {
        hint_decl(text, "width", &d);
    }
    if let Some(v) = doc.get_attribute(node, "height")
        && let Some(d) = html_dimension_nonzero(v)
    {
        hint_decl(text, "height", &d);
    }
}
/// and an optional sign, then a digit run; trailing junk is ignored.
fn html_loose_int(value: &str) -> Option<i64> {
    let s = value.trim_start();
    let (sign, s) = match s.strip_prefix('-') {
        Some(rest) => (-1i64, rest),
        None => (1i64, s.strip_prefix('+').unwrap_or(s)),
    };
    let mut n: i64 = 0;
    let mut any = false;
    for c in s.chars() {
        match c.to_digit(10) {
            Some(d) => {
                any = true;
                n = n.saturating_mul(10).saturating_add(i64::from(d));
            }
            None => break,
        }
    }
    any.then(|| sign * n)
}

/// Hints for a row group / column group from its ancestor table's
/// mapping (Blink's `AdditionalGroupStyle` for `rules=groups`, plus
/// the appendix's `bordercolor` inheritance).
fn group_hints(doc: &Document, table: NodeId, tag: &str, text: &mut String) {
    let attrs = table_attrs(doc, table);
    if attrs.bordercolor.is_some() {
        hint_border4(text, "color", "inherit");
    }
    if attrs.rules != Some(TableRules::Groups) {
        return;
    }
    let sides: &[&str] = if tag.eq_ignore_ascii_case("colgroup") {
        &["left", "right"]
    } else {
        &["top", "bottom"]
    };
    for side in sides {
        hint_decl(text, &format!("border-{side}-width"), "1px");
        hint_decl(text, &format!("border-{side}-style"), "solid");
    }
}

/// Properties CSS Lists 3 allows on `::marker` (font properties,
/// `color`, and a few text properties), intersected with the
/// properties this engine computes. Declarations of any other
/// property on a `::marker` rule are ignored by
/// [`Styler::marker_style_for`]. Atoms, so the marker cascade path
/// filters candidates with one `u8` compare each.
const MARKER_ATOMS: &[PropertyAtom] = &[
    PropertyAtom::COLOR,
    PropertyAtom::FONT_FAMILY,
    PropertyAtom::FONT_SIZE,
    PropertyAtom::FONT_STYLE,
    PropertyAtom::FONT_WEIGHT,
    PropertyAtom::FONT_VARIANT,
    PropertyAtom::LINE_HEIGHT,
    PropertyAtom::DIRECTION,
    PropertyAtom::UNICODE_BIDI,
    PropertyAtom::WHITE_SPACE,
    PropertyAtom::LETTER_SPACING,
    PropertyAtom::WORD_SPACING,
    PropertyAtom::TEXT_TRANSFORM,
];

/// Properties `::before`/`::after` rules may set in this engine: the
/// `::marker` set (the private `MARKER_ATOMS` above) plus `content` (which
/// `::marker` deliberately excludes — a marker's text comes from
/// `list-style-type`, never from `content`) plus `display` (so
/// `div::before { display: block }` clearfixes and
/// `table::after { display: table-caption }` honor the pseudo's own
/// display instead of always rendering inline; the pseudo base style
/// still resets `display` to its initial `inline`, so only an explicit
/// pseudo declaration overrides it). Exported for
/// generated-content box construction (see `bowser_layout`'s
/// `pseudo::query_pseudo_content`, which passes it as the
/// `allowed_atoms` of [`Styler::pseudo_style_for`]). The
/// `before_after_atoms_mirror_marker_atoms_plus_content_and_display`
/// test pins the relationship so the two lists cannot drift apart.
pub const BEFORE_AFTER_ATOMS: &[PropertyAtom] = &[
    PropertyAtom::CONTENT,
    PropertyAtom::DISPLAY,
    PropertyAtom::COLOR,
    PropertyAtom::FONT_FAMILY,
    PropertyAtom::FONT_SIZE,
    PropertyAtom::FONT_STYLE,
    PropertyAtom::FONT_WEIGHT,
    PropertyAtom::FONT_VARIANT,
    PropertyAtom::LINE_HEIGHT,
    PropertyAtom::DIRECTION,
    PropertyAtom::UNICODE_BIDI,
    PropertyAtom::WHITE_SPACE,
    PropertyAtom::LETTER_SPACING,
    PropertyAtom::WORD_SPACING,
    PropertyAtom::TEXT_TRANSFORM,
];

/// Declares the computed-property set once and derives both the
/// [`COMPUTED_PROPS`] name table and the [`props`] const-index module from
/// it, so the two can never drift apart. Each entry is
/// `CONST_NAME = "css-name"`; indices are assigned in declaration order.
macro_rules! computed_props {
    ($( $(#[$_meta:meta])* $name:ident = $css:literal ),* $(,)?) => {
        /// Properties resolved for every element: the inherited set plus the
        /// non-inherited properties a layout engine needs. Custom properties
        /// (`--*`) are added dynamically from the cascade.
        ///
        /// `pub(crate)` so the `propatom` tests can pin the generated atom
        /// table's leading run against this declaration order.
        pub(crate) const COMPUTED_PROPS: &[&str] = &[ $($css),* ];

        /// Const indices into [`ComputedStyle`]'s property slots, in
        /// [`COMPUTED_PROPS`] order. [`ComputedStyle::get_by_index`] with
        /// one of these skips the string-hash lookup that
        /// [`ComputedStyle::get`] pays on every call, which is what the
        /// hot layout/paint paths use.
        pub mod props {
            computed_props!(@idx 0u8; $( $name = $css, )*);
        }
    };
    (@idx $i:expr; $name:ident = $css:literal, $($rest:tt)*) => {
        #[allow(missing_docs)]
        pub const $name: u8 = $i;
        computed_props!(@idx ($i + 1); $($rest)*);
    };
    (@idx $i:expr;) => {};
}

computed_props! {
    // Inherited.
    COLOR = "color",
    FONT_FAMILY = "font-family",
    FONT_SIZE = "font-size",
    FONT_STYLE = "font-style",
    FONT_WEIGHT = "font-weight",
    FONT_VARIANT = "font-variant",
    LINE_HEIGHT = "line-height",
    TEXT_ALIGN = "text-align",
    TEXT_INDENT = "text-indent",
    TEXT_TRANSFORM = "text-transform",
    LETTER_SPACING = "letter-spacing",
    WORD_SPACING = "word-spacing",
    WHITE_SPACE = "white-space",
    VISIBILITY = "visibility",
    CURSOR = "cursor",
    DIRECTION = "direction",
    // CSS Writing Modes 3: `unicode-bidi` sits next to `direction`
    // (its bidi companion), but unlike `direction` it is NOT inherited.
    UNICODE_BIDI = "unicode-bidi",
    // CSS Writing Modes 3: inherited; selects the block/inline axes
    // that every logical property below resolves against.
    WRITING_MODE = "writing-mode",
    // Box / layout.
    DISPLAY = "display",
    POSITION = "position",
    TOP = "top",
    RIGHT = "right",
    BOTTOM = "bottom",
    LEFT = "left",
    // CSS Logical Properties 1: flow-relative insets. Resolved to the
    // physical `top`/`right`/`bottom`/`left` slots against the
    // element's own computed `direction`/`writing-mode` after the
    // cascade (see `apply_logical_mapping`).
    INSET_INLINE_START = "inset-inline-start",
    INSET_INLINE_END = "inset-inline-end",
    INSET_BLOCK_START = "inset-block-start",
    INSET_BLOCK_END = "inset-block-end",
    FLOAT = "float",
    CLEAR = "clear",
    WIDTH = "width",
    HEIGHT = "height",
    MIN_WIDTH = "min-width",
    MIN_HEIGHT = "min-height",
    MAX_WIDTH = "max-width",
    MAX_HEIGHT = "max-height",
    // CSS Logical Properties 1: `inline-size` maps to `width` in
    // horizontal writing modes and to `height` in vertical ones;
    // `block-size` maps to the other axis.
    INLINE_SIZE = "inline-size",
    BLOCK_SIZE = "block-size",
    MARGIN_TOP = "margin-top",
    MARGIN_RIGHT = "margin-right",
    MARGIN_BOTTOM = "margin-bottom",
    MARGIN_LEFT = "margin-left",
    // CSS Logical Properties 1: flow-relative margins, resolved to the
    // physical `margin-*` slots (see `apply_logical_mapping`).
    MARGIN_INLINE_START = "margin-inline-start",
    MARGIN_INLINE_END = "margin-inline-end",
    MARGIN_BLOCK_START = "margin-block-start",
    MARGIN_BLOCK_END = "margin-block-end",
    PADDING_TOP = "padding-top",
    PADDING_RIGHT = "padding-right",
    PADDING_BOTTOM = "padding-bottom",
    PADDING_LEFT = "padding-left",
    // CSS Logical Properties 1: flow-relative padding, resolved to the
    // physical `padding-*` slots (see `apply_logical_mapping`).
    PADDING_INLINE_START = "padding-inline-start",
    PADDING_INLINE_END = "padding-inline-end",
    PADDING_BLOCK_START = "padding-block-start",
    PADDING_BLOCK_END = "padding-block-end",
    BORDER_TOP_WIDTH = "border-top-width",
    BORDER_RIGHT_WIDTH = "border-right-width",
    BORDER_BOTTOM_WIDTH = "border-bottom-width",
    BORDER_LEFT_WIDTH = "border-left-width",
    BORDER_TOP_STYLE = "border-top-style",
    BORDER_RIGHT_STYLE = "border-right-style",
    BORDER_BOTTOM_STYLE = "border-bottom-style",
    BORDER_LEFT_STYLE = "border-left-style",
    BORDER_TOP_COLOR = "border-top-color",
    BORDER_RIGHT_COLOR = "border-right-color",
    BORDER_BOTTOM_COLOR = "border-bottom-color",
    BORDER_LEFT_COLOR = "border-left-color",
    // CSS Logical Properties 1: flow-relative borders, resolved to the
    // physical `border-*-width/style/color` slots against the
    // element's own computed `direction`/`writing-mode` (see
    // `apply_logical_mapping`).
    BORDER_INLINE_START_WIDTH = "border-inline-start-width",
    BORDER_INLINE_START_STYLE = "border-inline-start-style",
    BORDER_INLINE_START_COLOR = "border-inline-start-color",
    BORDER_INLINE_END_WIDTH = "border-inline-end-width",
    BORDER_INLINE_END_STYLE = "border-inline-end-style",
    BORDER_INLINE_END_COLOR = "border-inline-end-color",
    BORDER_BLOCK_START_WIDTH = "border-block-start-width",
    BORDER_BLOCK_START_STYLE = "border-block-start-style",
    BORDER_BLOCK_START_COLOR = "border-block-start-color",
    BORDER_BLOCK_END_WIDTH = "border-block-end-width",
    BORDER_BLOCK_END_STYLE = "border-block-end-style",
    BORDER_BLOCK_END_COLOR = "border-block-end-color",
    BORDER_TOP_LEFT_RADIUS = "border-top-left-radius",
    BORDER_TOP_RIGHT_RADIUS = "border-top-right-radius",
    BORDER_BOTTOM_RIGHT_RADIUS = "border-bottom-right-radius",
    BORDER_BOTTOM_LEFT_RADIUS = "border-bottom-left-radius",
    BOX_SIZING = "box-sizing",
    // Preferred aspect ratio (css-sizing-4 §3.1): not inherited; layout
    // transfers automatic sizes through it.
    ASPECT_RATIO = "aspect-ratio",
    OVERFLOW = "overflow",
    OVERFLOW_X = "overflow-x",
    OVERFLOW_Y = "overflow-y",
    // CSS Overflow 3 §2: flow-relative overflow longhands. Each keeps its
    // own computed value (flow-relative) and folds onto the physical
    // `overflow-x`/`overflow-y` slots after the cascade (see
    // `apply_logical_mapping`).
    OVERFLOW_BLOCK = "overflow-block",
    OVERFLOW_INLINE = "overflow-inline",
    OPACITY = "opacity",
    MIX_BLEND_MODE = "mix-blend-mode",
    // CSS Transforms 1: not inherited. Only the containing-block
    // effect is implemented (a non-`none` transform makes the box the
    // containing block for `position: fixed` descendants); visual
    // rendering of the transform itself is future work.
    TRANSFORM = "transform",
    // CSS Will Change 1 §3: not inherited. `auto` (the initial) is a
    // bare keyword; a hint list is a `List` of keywords. Naming a
    // stacking-context trigger makes the box the containing block for
    // fixed/absolute descendants.
    WILL_CHANGE = "will-change",
    // Filter Effects 1 §8: not inherited. `none` (the initial) or a
    // `Filter` function list; a non-`none` filter makes the box the
    // containing block for fixed/absolute descendants. Rendering the
    // filter itself is future work.
    FILTER = "filter",
    // CSS Transforms 2: not inherited. `none` (the initial) or a
    // `<length>`; a non-`none` perspective makes the box the containing
    // block for fixed/absolute descendants.
    PERSPECTIVE = "perspective",
    // CSS Contain 1 §2.1 / CSS Contain 2: not inherited. `none` (the
    // initial), `strict`, `content`, or a subset of `size`/`layout`/
    // `style`/`paint`; `paint`, `layout`, `strict`, and `content` make
    // the box the containing block for fixed/absolute descendants.
    CONTAIN = "contain",
    // CSS Transforms 1 §14: the pivot point `transform` rotates/scales
    // around. Parsed by `TransformOrigin` (see `crate::value`).
    TRANSFORM_ORIGIN = "transform-origin",
    Z_INDEX = "z-index",
    // Flexbox.
    FLEX_DIRECTION = "flex-direction",
    FLEX_WRAP = "flex-wrap",
    JUSTIFY_CONTENT = "justify-content",
    ALIGN_ITEMS = "align-items",
    ALIGN_CONTENT = "align-content",
    ALIGN_SELF = "align-self",
    FLEX_GROW = "flex-grow",
    FLEX_SHRINK = "flex-shrink",
    FLEX_BASIS = "flex-basis",
    ORDER = "order",
    // Background.
    BACKGROUND_COLOR = "background-color",
    BACKGROUND_IMAGE = "background-image",
    BACKGROUND_REPEAT = "background-repeat",
    BACKGROUND_ATTACHMENT = "background-attachment",
    BACKGROUND_POSITION = "background-position",
    // CSS Backgrounds 3 §7: not inherited. `none` (the initial) or a
    // comma-separated `<shadow>` list, parsed by `parse_shadow_value`
    // (parse.rs) into `CssValue::Shadows`.
    BOX_SHADOW = "box-shadow",
    // Text misc.
    TEXT_DECORATION = "text-decoration",
    TEXT_DECORATION_STYLE = "text-decoration-style",
    // CSS Text Decoration 3 §4.3: inherited (unlike `box-shadow`). `none`
    // (the initial) or a comma-separated `<shadow>` list without `inset`
    // or spread; shares the `CssValue::Shadows` representation.
    TEXT_SHADOW = "text-shadow",
    VERTICAL_ALIGN = "vertical-align",
    // Lists.
    LIST_STYLE_TYPE = "list-style-type",
    LIST_STYLE_POSITION = "list-style-position",
    LIST_STYLE_IMAGE = "list-style-image",
    // CSS Generated Content 3 §2: `content` applies to `::before`,
    // `::after`, and `::marker` (not inherited; the initial `normal`
    // computes to `none` on the pseudo-elements themselves, so no box
    // is generated).
    CONTENT = "content",
    // Tables.
    BORDER_COLLAPSE = "border-collapse",
    BORDER_SPACING = "border-spacing",
    TABLE_LAYOUT = "table-layout",
    CAPTION_SIDE = "caption-side",
    EMPTY_CELLS = "empty-cells",
    // Grid.
    GRID_TEMPLATE_COLUMNS = "grid-template-columns",
    GRID_TEMPLATE_ROWS = "grid-template-rows",
    GRID_AUTO_COLUMNS = "grid-auto-columns",
    GRID_AUTO_ROWS = "grid-auto-rows",
    GRID_COLUMN = "grid-column",
    GRID_ROW = "grid-row",
    GAP = "gap",
    GRID_GAP = "grid-gap",
    ROW_GAP = "row-gap",
    COLUMN_GAP = "column-gap",
    JUSTIFY_ITEMS = "justify-items",
    JUSTIFY_SELF = "justify-self",
    GRID_AUTO_FLOW = "grid-auto-flow",
    // Fragmentation (page/column breaks).
    BREAK_BEFORE = "break-before",
    BREAK_AFTER = "break-after",
    BREAK_INSIDE = "break-inside",
    WIDOWS = "widows",
    ORPHANS = "orphans",
    // Multi-column layout (CSS Multicol 1). `column-gap` shares its name
    // with the grid/flex gap longhand: the single property serves both,
    // with `normal` meaning 1em in a multicol container and 0 elsewhere
    // (resolved by layout, not the cascade).
    COLUMN_COUNT = "column-count",
    COLUMN_WIDTH = "column-width",
    COLUMNS = "columns",
    // `column-span` and `column-fill` are single-keyword longhands; the
    // `column-rule` shorthand expands eagerly at parse time into its
    // width/style/color longhands (see `expand_shorthand` in parse.rs),
    // with a deferred cascade-side expansion for `var()` values.
    COLUMN_SPAN = "column-span",
    COLUMN_FILL = "column-fill",
    COLUMN_RULE = "column-rule",
    COLUMN_RULE_WIDTH = "column-rule-width",
    COLUMN_RULE_STYLE = "column-rule-style",
    COLUMN_RULE_COLOR = "column-rule-color",
    // Scroll snap areas (CSS Scroll Snap 1): box-side shorthands expand
    // at parse time; the longhands below are the computed slots.
    SCROLL_MARGIN_TOP = "scroll-margin-top",
    SCROLL_MARGIN_RIGHT = "scroll-margin-right",
    SCROLL_MARGIN_BOTTOM = "scroll-margin-bottom",
    SCROLL_MARGIN_LEFT = "scroll-margin-left",
    SCROLL_PADDING_TOP = "scroll-padding-top",
    SCROLL_PADDING_RIGHT = "scroll-padding-right",
    SCROLL_PADDING_BOTTOM = "scroll-padding-bottom",
    SCROLL_PADDING_LEFT = "scroll-padding-left",
    // Scroll snap logical longhands (CSS Scroll Snap 1 via CSS Logical
    // Properties 1): each keeps its own computed value in its slot and
    // folds onto the physical `scroll-margin-*` / `scroll-padding-*`
    // slots in `logical_map()` (see `apply_logical_mapping`).
    SCROLL_MARGIN_INLINE_START = "scroll-margin-inline-start",
    SCROLL_MARGIN_INLINE_END = "scroll-margin-inline-end",
    SCROLL_MARGIN_BLOCK_START = "scroll-margin-block-start",
    SCROLL_MARGIN_BLOCK_END = "scroll-margin-block-end",
    SCROLL_PADDING_INLINE_START = "scroll-padding-inline-start",
    SCROLL_PADDING_INLINE_END = "scroll-padding-inline-end",
    SCROLL_PADDING_BLOCK_START = "scroll-padding-block-start",
    SCROLL_PADDING_BLOCK_END = "scroll-padding-block-end",
    // Overscroll behavior (CSS Overscroll Behavior 1): one shorthand,
    // two per-axis longhands.
    OVERSCROLL_BEHAVIOR_X = "overscroll-behavior-x",
    OVERSCROLL_BEHAVIOR_Y = "overscroll-behavior-y",
    // Scroll behavior (CSSOM View): `auto | smooth`, not inherited.
    SCROLL_BEHAVIOR = "scroll-behavior",
    // Scrollbar gutter (CSS Overflow 4): `auto | stable && both-edges?`,
    // not inherited. Reserves scrollbar space in the scrollport so
    // content never shifts when scrollbars appear/disappear.
    SCROLLBAR_GUTTER = "scrollbar-gutter",
    // `touch-action` (CSS Pointer Events 3): `auto | none |
    // manipulation | [ pan-x | pan-left | pan-right ]? [ pan-y | pan-up |
    // pan-down ]?`, not inherited. The touch pipeline reads the scroller
    // container's computed value to gate drag/fling panning per axis.
    TOUCH_ACTION = "touch-action",
    // Scroll snap (CSS Scroll Snap 1): `scroll-snap-type` on the scroll
    // container, `scroll-snap-align` / `scroll-snap-stop` on snap areas.
    // Plain longhands, not shorthands, so no parse-time expansion.
    SCROLL_SNAP_TYPE = "scroll-snap-type",
    SCROLL_SNAP_ALIGN = "scroll-snap-align",
    SCROLL_SNAP_STOP = "scroll-snap-stop",
    // `scroll-initial-target` (CSS Scroll Snap 2 §3.1.2): `none |
    // nearest`, not inherited. `overflow-clip-margin` (CSS Overflow 3
    // §3.2): `<visual-box> || <length>`, not inherited, applies to the
    // boxes `overflow` applies to.
    SCROLL_INITIAL_TARGET = "scroll-initial-target",
    OVERFLOW_CLIP_MARGIN = "overflow-clip-margin",
    // CSS Shapes 1 / css-clip-path-1 §4: not inherited. `none` (the
    // initial), a `<basic-shape>`, or a `url()` reference. The shapes
    // parse structurally (see `parse_basic_shape` in value.rs);
    // `bowser_paint` resolves them against the border box.
    CLIP_PATH = "clip-path",
    // Filter Effects 2 §2: not inherited. `none` (the initial) or a
    // `<filter-value-list>` — the same grammar as `filter`, so it
    // shares the `Filter` value and parser. Like `filter`, a non-`none`
    // value makes the box the containing block for fixed/absolute
    // descendants (see `establishes_fixed_cb` in cbprops.rs).
    BACKDROP_FILTER = "backdrop-filter",
    // CSS Box Alignment 3 §6.1: `place-self` shorthand for `align-self`
    // + `justify-self`. Expanded to its longhands at parse time (and
    // after `var()` substitution in the cascade), so it never occupies
    // a computed slot; it needs an atom for the deferred-shorthand
    // path (`is_shorthand_property` / `expand_deferred_shorthand`).
    PLACE_SELF = "place-self",
    // CSS Lists 3 §4: `counter-reset` / `counter-increment`. Not
    // inherited; they build the counter scopes that `counter()` /
    // `counters()` in `content` resolve against (see
    // `pseudo::resolve_content_text`). Deliberately absent from
    // `BEFORE_AFTER_ATOMS`: v1 scopes only the originating element's
    // adjustments (pseudo-element counters are future work).
    COUNTER_RESET = "counter-reset",
    COUNTER_INCREMENT = "counter-increment",
}

// `Styler` keys its resolved-value memo by `COMPUTED_PROPS` index as `u8`.
const _: () = assert!(COMPUTED_PROPS.len() <= 256);

/// Upper bound on the computed-property count, sizing the fixed winner
/// arrays. Bump when the property set grows past it (the assert in
/// `cascade_expanded` fails loudly).
// w67 sibling in-flight: the w67 prop additions (touch-action, content,
// …) pushed COMPUTED_PROPS past 182; the `<= 256` assert above (u8 memo
// keys) is the binding one. Mechanical bump so the tree compiles; the
// owning worker should confirm/keep it.
const MAX_COMPUTED_PROPS: usize = 256;

/// Exact computed-property count, for stack-allocated per-element
/// working arrays (see `Styler::compute`): no heap allocation for the
/// scratch itself, only for the boxed array the finished
/// [`ComputedStyle`] owns.
const PROP_COUNT: usize = COMPUTED_PROPS.len();

/// Initial values as process-wide shared `Arc`s, indexed by
/// [`COMPUTED_PROPS`] position. Most properties of most elements take
/// their initial value, so sharing one allocation per property removes a
/// `String` allocation per (element, unspecified property). The table
/// is also the fallback layer of the sparse style/memo storage: a slot
/// absent from a [`ComputedStyle`]'s delta (or a [`MemoSet`]'s entries)
/// reads as the entry here.
fn initials() -> &'static [Option<Arc<CssValue>>] {
    static INITIALS: std::sync::OnceLock<Box<[Option<Arc<CssValue>>]>> = std::sync::OnceLock::new();
    INITIALS.get_or_init(|| {
        COMPUTED_PROPS
            .iter()
            .map(|p| initial_value(p).map(Arc::new))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    })
}

/// The shared initial `Arc` for `idx` (cloned: one atomic bump).
fn initial_rc(idx: u8) -> Option<Arc<CssValue>> {
    initials()[idx as usize].clone()
}

/// The shared initial value for `idx` by reference (no atomic bump):
/// the fallback arm of [`ComputedStyle::get_by_index`]. The table is
/// process-wide and never mutated after initialization, so the
/// reference is `'static`.
fn initial_ref(idx: u8) -> Option<&'static CssValue> {
    initials()[idx as usize].as_deref()
}

/// Whether `value` is exactly the shared initial `Arc` for `idx`.
/// Pointer equality only: a distinct allocation that merely compares
/// equal to the initial value is still stored sparsely (storing is
/// always correct; omitting requires the identical `Arc`).
fn arc_is_initial(idx: u8, value: &Arc<CssValue>) -> bool {
    match &initials()[idx as usize] {
        Some(init) => Arc::ptr_eq(init, value),
        None => false,
    }
}

/// Whether `value` is the canonical fallback form for `idx` — what a
/// sparse reader materializes for an absent slot: the shared initial
/// `Arc`, or `None` when the property has no initial value. Sparse
/// writers omit exactly the values for which this holds, so `Known`
/// memo values that are unset-but-initial-`Some` (an inherited chain
/// that topped out) still store an explicit `None` entry and never get
/// confused with the initial value.
fn is_initial_form(idx: u8, value: &Option<Arc<CssValue>>) -> bool {
    match (value, &initials()[idx as usize]) {
        (None, None) => true,
        (Some(v), Some(_)) => arc_is_initial(idx, v),
        _ => false,
    }
}

/// Default font size in px (the `medium` keyword / UA default).
const DEFAULT_FONT_SIZE_PX: f32 = 16.0;

/// Index of `font-size` in [`COMPUTED_PROPS`], resolved once without
/// hardcoding the position.
fn font_size_prop_idx() -> u8 {
    static IDX: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *IDX.get_or_init(|| {
        COMPUTED_PROPS
            .iter()
            .position(|p| *p == "font-size")
            .expect("font-size is in COMPUTED_PROPS") as u8
    })
}

fn is_inherited(property: &str) -> bool {
    matches!(
        property,
        "color"
            | "font-family"
            | "font-size"
            | "font-style"
            | "font-weight"
            | "font-variant"
            | "line-height"
            | "text-align"
            | "text-indent"
            | "text-transform"
            | "letter-spacing"
            | "word-spacing"
            | "white-space"
            | "visibility"
            | "cursor"
            | "direction"
            // CSS Writing Modes 3 §2.4: `writing-mode` is inherited.
            | "writing-mode"
            // CSS 2.1 §17.5.1: `empty-cells` is inherited.
            | "empty-cells"
            // CSS 2.1 §17.5.1 / css-tables-3 inheritance: the table
            // border model properties are inherited.
            | "border-collapse"
            | "border-spacing"
            | "caption-side"
            // CSS Text Decoration 3 §4.3: `text-shadow` is inherited
            // (unlike `box-shadow`).
            | "text-shadow"
            | "list-style-type"
            | "widows"
            | "orphans"
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Origin {
    Ua,
    Author,
    Inline,
}

/// Cascade rank: higher wins. Inline `style=""` is author origin; its
/// non-important declarations beat author stylesheets, its `!important`
/// ones beat those too but still lose to UA `!important`.
fn rank(origin: Origin, important: bool) -> u8 {
    match (origin, important) {
        (Origin::Ua, false) => 0,
        (Origin::Author, false) => 1,
        (Origin::Inline, false) => 2,
        (Origin::Author, true) => 3,
        (Origin::Inline, true) => 4,
        (Origin::Ua, true) => 5,
    }
}

/// Origin class of a cascade rank for `revert`: inline style counts as
/// author origin (CSS Cascade 5), so only UA vs author matters.
fn revert_origin_class(rank: u8) -> u8 {
    match rank {
        0 | 5 => 0, // Ua
        _ => 1,     // Author (stylesheets and inline)
    }
}

#[derive(Clone)]
struct Candidate<'a> {
    rank: u8,
    spec: (u32, u32, u32),
    sheet: usize,
    order: usize,
    decl_idx: usize,
    // Borrowed from the stylesheet's declaration for sheet rules (the
    // hot path: no per-node clone of the property name or value);
    // owned for inline `style` attributes, whose parsed declarations
    // are temporaries. Well-known names are a 1-byte atom either way,
    // so the cascade below resolves the property index with no string
    // hashing. Cloning a `Cow::Borrowed` is a pointer copy, so the
    // downstream winners array no longer heap-clones values.
    property: Cow<'a, PropertyName>,
    value: Cow<'a, CssValue>,
}

/// Words of `known` bits in a [`MemoSet`]: one bit per computed
/// property, rounded up (180 properties -> 3 words).
const KNOWN_WORDS: usize = PROP_COUNT.div_ceil(64);

/// One node's memoized computed values, sparsely stored (wave 50).
///
/// The old form was a boxed `[MemoSlot; PROP_COUNT]` array: 16 bytes
/// per slot * 181 slots = 2896 heap bytes per node per style pass,
/// retained for every styled node. Most memoized values are the
/// initial-filled form (the element took the shared initial value, or
/// an inherited chain topped out at the initial value), so this set
/// stores only what deviates:
///
/// - `known`: one bit per property — "computed" vs "not computed yet"
///   (the old `MemoSlot::Unknown` distinction, for all slots at once).
/// - `entries`: the computed values that are *not* the canonical
///   initial fallback ([`is_initial_form`]), sorted by index. A known
///   slot with no entry reads as [`initial_rc`]; a known slot whose
///   computed value is unset (`None`) while its initial is `Some` —
///   e.g. an inherited property on a root whose ancestors yield
///   nothing — stores an explicit `(idx, None)` entry, so it can never
///   be confused with the initial value (`font-size` resolution relies
///   on that distinction: unset reads as "inherit the parent's px",
///   the initial `medium` would read as 16px).
///
/// Inline cost is 24 bytes of bits plus a `Vec` header (48 bytes per
/// slab slot, no heap at all until a deviating value is memoized),
/// versus 8 bytes inline plus 2864 heap bytes before.
#[derive(Clone, Default)]
struct MemoSet {
    known: [u64; KNOWN_WORDS],
    entries: Vec<(u8, Option<Arc<CssValue>>)>,
}

impl MemoSet {
    fn is_known(&self, idx: u8) -> bool {
        let i = idx as usize;
        self.known[i / 64] & (1u64 << (i % 64)) != 0
    }

    fn mark_known(&mut self, idx: u8) {
        let i = idx as usize;
        self.known[i / 64] |= 1u64 << (i % 64);
    }

    /// Every property computed (the old "all slots `Known`" probe the
    /// warm root-first path in `compute` checks on the parent).
    fn is_complete(&self) -> bool {
        for (w, word) in self.known.iter().enumerate() {
            let valid = if (w + 1) * 64 <= PROP_COUNT {
                u64::MAX
            } else {
                (1u64 << (PROP_COUNT % 64)) - 1
            };
            if word & valid != valid {
                return false;
            }
        }
        true
    }

    /// The memoized value of `idx`: the outer `None` is "not computed
    /// yet"; the inner value is the computed one (which may itself be
    /// unset/`None`), exactly the old `MemoSlot` read.
    fn get(&self, idx: u8) -> Option<Option<Arc<CssValue>>> {
        if !self.is_known(idx) {
            return None;
        }
        Some(match self.entries.binary_search_by_key(&idx, |(i, _)| *i) {
            Ok(pos) => self.entries[pos].1.clone(),
            Err(_) => initial_rc(idx),
        })
    }

    /// Memoize `value` for `idx`. Values in canonical initial form are
    /// recorded as known-with-no-entry (the reader re-materializes
    /// them from [`initials`]); anything else upserts a sorted entry.
    fn set(&mut self, idx: u8, value: Option<Arc<CssValue>>) {
        self.mark_known(idx);
        match self.entries.binary_search_by_key(&idx, |(i, _)| *i) {
            Ok(pos) => {
                if is_initial_form(idx, &value) {
                    self.entries.remove(pos);
                } else {
                    self.entries[pos].1 = value;
                }
            }
            Err(pos) => {
                if !is_initial_form(idx, &value) {
                    self.entries.insert(pos, (idx, value));
                }
            }
        }
    }

    /// Record every slot of a fully computed, initial-filled array
    /// (the batch write at the end of `Styler::compute`): all known,
    /// entries only where the value deviates from the initial form.
    fn fill_from_slots(&mut self, props: &[Option<Arc<CssValue>>]) {
        self.known = [0; KNOWN_WORDS];
        for idx in 0..PROP_COUNT {
            self.mark_known(idx as u8);
        }
        self.entries.clear();
        for (idx, value) in props.iter().enumerate() {
            if !is_initial_form(idx as u8, value) {
                self.entries.push((idx as u8, value.clone()));
            }
        }
    }

    /// Record a finished style's values (style sharing's adopt step):
    /// all known; the style's delta entries are exactly the deviating
    /// values, and every omitted slot reads back as its initial value,
    /// which is what the donor's full array held there.
    fn fill_from_style(&mut self, style: &ComputedStyle) {
        self.known = [0; KNOWN_WORDS];
        for idx in 0..PROP_COUNT {
            self.mark_known(idx as u8);
        }
        self.entries.clear();
        self.entries
            .extend(style.delta.iter().map(|(i, v)| (*i, Some(v.clone()))));
    }
}

/// Work item for the iterative `var()` substitution machine in
/// [`Styler::resolve`].
enum ResolveWork<'a> {
    /// Resolve `value`, pushing exactly one [`ResolveResult`] onto the
    /// result stack.
    Value(CssValue),
    /// Borrowed form of [`ResolveWork::Value`]: the caller's entry value
    /// and `env` values arrive by reference, so resolving them never
    /// clones the specified value's own structure. Composite shapes
    /// destructure by reference (their items re-enter the stack still
    /// borrowed); leaves materialize exactly once, on the result stack.
    ValueRef(&'a CssValue),
    /// A `var()` reference with its name already interned: `id` is the
    /// [`VarNameInterner`] id, `fallback` the owned fallback value. The
    /// borrowed fast path pushes this instead of cloning the name
    /// `String` into an owned [`Value`](ResolveWork::Value); the owned
    /// path pushes it for parser-produced `Var` nodes. One arm below
    /// runs the shared resolution tail for both: a single copy of the
    /// logic, no fork.
    VarRef { id: u32, fallback: Option<CssValue> },
    /// A custom-property reference finished: pop its substituted value,
    /// memoize it under `name`, drop `name` from the active set; on
    /// success the value stands (spliced when it is a list), otherwise
    /// resolve `fallback`. `name` is a [`VarNameInterner`] id, never a
    /// cloned `String`.
    PropTail {
        name: u32,
        fallback: Option<CssValue>,
    },
    /// Pop `n` results and combine them into a `List` (all must succeed);
    /// list-valued `var()` substitutions splice in item-wise.
    ListTail(usize),
    /// Pop `n` track results and combine them back into the
    /// [`CssValue::Repeat`] descriptor they came from (all must
    /// succeed). The combination rules mirror [`ResolveWork::ListTail`]
    /// — list-valued `var()` substitutions splice item-wise, comma
    /// groups stay nested — but the dedicated descriptor shape is
    /// preserved, never flattened into a bare track listing.
    RepeatTail { mode: RepeatMode, n: usize },
    /// A custom property's `/`-separated sides finished resolving: pop
    /// the two side results and push one [`ResolveResult::Slash`].
    SlashTail,
    /// A custom property's comma-separated groups finished resolving:
    /// pop the `n` group results and push one [`ResolveResult::Comma`].
    CommaTail(usize),
    /// Push `None`: unresolvable reference without a fallback.
    PushNone,
    /// A `var()` fallback finished resolving: pop its single result and
    /// splice a list-valued fallback token-wise (like [`ResolveWork`]'s
    /// `PropTail` does for a defined custom property) instead of leaving
    /// it re-wrapped in a nested `List`. CSS substitutes token-wise, so a
    /// fallback must behave exactly like the property value would; without
    /// the splice the deferred `font` expansion misreads the nested list
    /// as a comma group (e.g. `font: 12px/var(--x, 1.5 Times) New Roman`
    /// split "Times" and "New Roman" into two families).
    FallbackTail,
}

/// One resolved stack item in [`Styler::resolve`].
///
/// The [`Slash`](ResolveResult::Slash) and [`Comma`](ResolveResult::Comma)
/// payloads are `Rc`-shared with the [`var()` memo](MemoVar): a memoized
/// slash/comma substitution re-enters the stack as a pointer bump instead
/// of deep-cloning its group `Vec`s per `var()` reference, and memoizing
/// a freshly resolved one moves the `Rc` instead of cloning the payload
/// for the memo entry.
enum ResolveResult {
    /// A single resolved value (`None` = invalid at computed-value time).
    Single(Option<CssValue>),
    /// Items of a list-valued `var()` substitution. CSS substitutes
    /// custom properties token-wise, so when the enclosing value is a
    /// list these splice in flat instead of nesting; anywhere else they
    /// re-wrap into a `List`. (`None` = the substitution failed.)
    Splice(Option<Vec<CssValue>>),
    /// A `var()` substitution whose custom-property token stream held a
    /// top-level `/`: the resolved before/after item groups (`None` =
    /// the substitution failed). CSS substitutes token-wise, so when
    /// the enclosing value is a list the slash still splits the whole
    /// list: `font: var(--x) serif` with `--x: 12px/1.5` is
    /// `font: 12px/1.5 serif`. Kept distinct from [`Splice`] so the
    /// marker is never confused with a comma list (both nest as
    /// `List([List, List])`); a second top-level slash is invalid.
    Slash(SharedSlash),
    /// A `var()` substitution whose custom-property token stream held a
    /// top-level comma: the resolved comma groups (`None` = the
    /// substitution failed). Each group splices as one nested list so
    /// the deferred `font` expansion can tell `Arial, sans-serif` (two
    /// families) from `Times New Roman` (one family name); every other
    /// consumer normalizes back to the [`parse_value`] shape (see
    /// [`normalize_comma_groups`]).
    Comma(SharedComma),
}

impl ResolveResult {
    /// The plain value form, re-wrapping a splice into a `List`, a
    /// slash substitution into its `/`-separated marker form, and a
    /// comma substitution into nested groups.
    fn into_option(self) -> Option<CssValue> {
        match self {
            ResolveResult::Single(v) => v,
            ResolveResult::Splice(items) => items.map(CssValue::List),
            // The payload may be shared with the `var()` memo (see
            // [`MemoVar`]): unwrap moves when this result owns the only
            // strong reference, and clones the groups only when the
            // memo still holds one — the same single copy the old
            // probe-then-move paid, relocated to the consumption point.
            ResolveResult::Slash(s) => Rc::unwrap_or_clone(s).map(|(before, after)| {
                CssValue::List(vec![CssValue::List(before), CssValue::List(after)])
            }),
            ResolveResult::Comma(g) => Rc::unwrap_or_clone(g)
                .map(|groups| CssValue::List(groups.into_iter().map(CssValue::List).collect())),
        }
    }
}

/// `Rc`-shared payload of a slash-marked `var()` substitution: the
/// resolved before/after item groups (`None` = invalid at computed-value
/// time). One allocation serves the in-flight [`ResolveResult`], the
/// [`var()` memo](MemoVar) entry, and every later memo hit.
type SharedSlash = Rc<Option<(Vec<CssValue>, Vec<CssValue>)>>;
/// `Rc`-shared payload of a comma-marked `var()` substitution: the
/// resolved comma groups (`None` = invalid at computed-value time).
/// Shared exactly like [`SharedSlash`].
type SharedComma = Rc<Option<Vec<Vec<CssValue>>>>;

/// Interned custom-property names shared by all `var()` resolution on one
/// [`Styler`]. The resolved-value memo and the cycle-detection set key by
/// the `u32` id instead of cloning the name `String` on every lookup; the
/// one `String` clone per *distinct* name happens once, at first use.
/// Append-only for the styler's lifetime: names never change meaning, so
/// ids stay valid across [`Styler::set_element_state`] invalidation.
#[derive(Default)]
struct VarNameInterner {
    ids: FxHashMap<String, u32>,
    names: Vec<String>,
}

impl VarNameInterner {
    /// The id for `name`, interning it on first use.
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_owned());
        self.ids.insert(name.to_owned(), id);
        id
    }

    /// The name for an id returned by [`intern`](Self::intern).
    fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }
}

/// Memoized `var()` substitution for one custom property on one node. A
/// slash-marked substitution keeps its marker here so a later `var()`
/// on the same node still splices it correctly whether it stands alone
/// or sits inside a list (see [`ResolveResult::Slash`]).
///
/// The slash/comma payloads are `Rc`-shared: memoizing a freshly
/// resolved substitution moves an `Rc` instead of deep-cloning the
/// group `Vec`s, and a probe hands out a pointer bump instead of cloning
/// the payload per `var()` reference. The plain [`Value`](MemoVar::Value)
/// payload stays an owned `CssValue`: every use site needs an owned value
/// to embed (splicing/extending parent lists), so sharing it would only
/// relocate the one unavoidable clone from the probe to the use.
#[derive(Clone)]
enum MemoVar {
    /// Plain substituted value (`None` = invalid at computed-value time).
    Value(Option<CssValue>),
    /// The token stream held a top-level `/`: resolved before/after
    /// groups (`None` = invalid at computed-value time).
    Slash(SharedSlash),
    /// The token stream held a top-level comma: resolved comma groups.
    Comma(SharedComma),
}

/// One node's cascade winners, `var()` unresolved and deferred
/// `var()`-in-shorthand declarations substituted and expanded to
/// longhands. Compact and sorted by [`COMPUTED_PROPS`] position: only
/// specified properties are stored (typically ~10 of 160), so building
/// and dropping it avoids zeroing a 160-slot array per node. Borrowed
/// from the stylesheet's declarations for sheet rules; owned only for
/// deferred `var()`-in-shorthand expansions (rare).
/// Sorted cascade winners for one node: `(COMPUTED_PROPS index, value,
/// cascade sequence)`. The sequence is the winner's position in the
/// cascade-sorted candidate list (higher wins later in the cascade); it
/// lets the logical-to-physical fold (`apply_logical_mapping`) decide
/// between a logical longhand and its physical counterpart by cascade
/// order, the way browsers do.
type NodeWinners<'a> = Rc<Vec<(u8, Cow<'a, CssValue>, u32)>>;

/// Look up the winning specified value for a [`COMPUTED_PROPS`]
/// position in a [`NodeWinners`] vec (binary search; the vec is sorted
/// by index with unique indices).
fn winner_at<'w>(winners: &'w [(u8, Cow<'_, CssValue>, u32)], idx: u8) -> Option<&'w CssValue> {
    winners
        .binary_search_by_key(&idx, |(i, _, _)| *i)
        .ok()
        .map(|pos| winners[pos].1.as_ref())
}

/// The computed `writing-mode`: which axes the logical properties
/// resolve against. Only the modes the text engine can lay out are
/// accepted (`is_valid_for_property` rejects `sideways-*`), so the
/// fallback arm is unreachable for computed values but keeps the
/// mapping total.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WritingMode {
    HorizontalTb,
    VerticalRl,
    VerticalLr,
}

fn writing_mode_of(props: &[Option<Arc<CssValue>>], wm_idx: u8) -> WritingMode {
    match props[wm_idx as usize].as_deref() {
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("vertical-rl") => {
            WritingMode::VerticalRl
        }
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("vertical-lr") => {
            WritingMode::VerticalLr
        }
        _ => WritingMode::HorizontalTb,
    }
}

fn direction_is_rtl(props: &[Option<Arc<CssValue>>], dir_idx: u8) -> bool {
    matches!(props[dir_idx as usize].as_deref(), Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("rtl"))
}

/// Physical side index for a flow-relative (axis, start/end) under the
/// element's own writing mode and direction: 0=top, 1=right, 2=bottom,
/// 3=left. Matches the abstract-to-physical table in CSS Writing Modes
/// 3 §6.4: in vertical writing modes `direction: rtl` runs the inline
/// axis bottom-to-top, while the block axis (`vertical-rl` →
/// right-to-left) never depends on `direction`.
fn logical_side_idx(wm: WritingMode, rtl: bool, inline_axis: bool, start: bool) -> usize {
    match (wm, inline_axis, rtl, start) {
        // horizontal-tb: inline axis is horizontal; block-start is top.
        (WritingMode::HorizontalTb, true, false, true) => 3, // left
        (WritingMode::HorizontalTb, true, false, false) => 1, // right
        (WritingMode::HorizontalTb, true, true, true) => 1,  // right
        (WritingMode::HorizontalTb, true, true, false) => 3, // left
        (WritingMode::HorizontalTb, false, _, true) => 0,    // top
        (WritingMode::HorizontalTb, false, _, false) => 2,   // bottom
        // vertical-rl / vertical-lr: inline axis is vertical.
        (_, true, false, true) => 0,                     // top
        (_, true, false, false) => 2,                    // bottom
        (_, true, true, true) => 2,                      // bottom
        (_, true, true, false) => 0,                     // top
        (WritingMode::VerticalRl, false, _, true) => 1,  // right
        (WritingMode::VerticalRl, false, _, false) => 3, // left
        (WritingMode::VerticalLr, false, _, true) => 3,  // left
        (WritingMode::VerticalLr, false, _, false) => 1, // right
    }
}

/// Every side-based logical longhand, as
/// `(logical idx, physical `top`-side base idx, inline axis?, start?)`.
/// The physical target is `base + logical_side_idx(...)`: each group
/// (`margin-*`, `padding-*`, `scroll-margin-*`, `scroll-padding-*`,
/// `top/right/bottom/left`, `border-top-width/…`) lists its sides in
/// top/right/bottom/left order in [`COMPUTED_PROPS`].
///
/// All indices are resolved once (no per-node hash lookup); the
/// per-node fold is a linear scan with no allocation.
struct LogicalMap {
    /// (logical idx, physical `top`-side base idx, inline axis?, start?).
    sides: Vec<(u8, u8, bool, bool)>,
    writing_mode: u8,
    direction: u8,
    width: u8,
    height: u8,
    inline_size: u8,
    block_size: u8,
    /// `overflow-x`, `overflow-y`, `overflow-block`, `overflow-inline`:
    /// the flow-relative overflow longhands map to the physical axes
    /// wholesale (no start/end split), folded in `apply_logical_mapping`.
    overflow_x: u8,
    overflow_y: u8,
    overflow_block: u8,
    overflow_inline: u8,
}

fn logical_map() -> &'static LogicalMap {
    static MAP: std::sync::OnceLock<LogicalMap> = std::sync::OnceLock::new();
    MAP.get_or_init(|| {
        let idx = |p: &str| {
            PropertyAtom::lookup(p)
                .expect("logical property registered")
                .index()
        };
        // The fold's `base + logical_side_idx(...)` arithmetic assumes
        // each side group lists its sides contiguously in
        // top/right/bottom/left order in `COMPUTED_PROPS`; fail fast in
        // debug builds if that ever drifts.
        debug_assert_eq!(idx("margin-right"), idx("margin-top") + 1);
        debug_assert_eq!(idx("margin-bottom"), idx("margin-top") + 2);
        debug_assert_eq!(idx("margin-left"), idx("margin-top") + 3);
        debug_assert_eq!(idx("padding-right"), idx("padding-top") + 1);
        debug_assert_eq!(idx("padding-bottom"), idx("padding-top") + 2);
        debug_assert_eq!(idx("padding-left"), idx("padding-top") + 3);
        debug_assert_eq!(idx("scroll-margin-right"), idx("scroll-margin-top") + 1);
        debug_assert_eq!(idx("scroll-margin-bottom"), idx("scroll-margin-top") + 2);
        debug_assert_eq!(idx("scroll-margin-left"), idx("scroll-margin-top") + 3);
        debug_assert_eq!(idx("scroll-padding-right"), idx("scroll-padding-top") + 1);
        debug_assert_eq!(idx("scroll-padding-bottom"), idx("scroll-padding-top") + 2);
        debug_assert_eq!(idx("scroll-padding-left"), idx("scroll-padding-top") + 3);
        debug_assert_eq!(idx("right"), idx("top") + 1);
        debug_assert_eq!(idx("bottom"), idx("top") + 2);
        debug_assert_eq!(idx("left"), idx("top") + 3);
        debug_assert_eq!(idx("border-right-width"), idx("border-top-width") + 1);
        debug_assert_eq!(idx("border-bottom-width"), idx("border-top-width") + 2);
        debug_assert_eq!(idx("border-left-width"), idx("border-top-width") + 3);
        debug_assert_eq!(idx("border-right-style"), idx("border-top-style") + 1);
        debug_assert_eq!(idx("border-bottom-style"), idx("border-top-style") + 2);
        debug_assert_eq!(idx("border-left-style"), idx("border-top-style") + 3);
        debug_assert_eq!(idx("border-right-color"), idx("border-top-color") + 1);
        debug_assert_eq!(idx("border-bottom-color"), idx("border-top-color") + 2);
        debug_assert_eq!(idx("border-left-color"), idx("border-top-color") + 3);
        let mut sides = Vec::with_capacity(32);
        for start in [true, false] {
            let end = if start { "start" } else { "end" };
            for (axis, inline_axis) in [("inline", true), ("block", false)] {
                sides.push((
                    idx(&format!("margin-{axis}-{end}")),
                    idx("margin-top"),
                    inline_axis,
                    start,
                ));
                sides.push((
                    idx(&format!("padding-{axis}-{end}")),
                    idx("padding-top"),
                    inline_axis,
                    start,
                ));
                sides.push((
                    idx(&format!("scroll-margin-{axis}-{end}")),
                    idx("scroll-margin-top"),
                    inline_axis,
                    start,
                ));
                sides.push((
                    idx(&format!("scroll-padding-{axis}-{end}")),
                    idx("scroll-padding-top"),
                    inline_axis,
                    start,
                ));
                sides.push((
                    idx(&format!("inset-{axis}-{end}")),
                    idx("top"),
                    inline_axis,
                    start,
                ));
                for part in ["width", "style", "color"] {
                    sides.push((
                        idx(&format!("border-{axis}-{end}-{part}")),
                        idx(&format!("border-top-{part}")),
                        inline_axis,
                        start,
                    ));
                }
            }
        }
        LogicalMap {
            sides,
            writing_mode: idx("writing-mode"),
            direction: idx("direction"),
            width: idx("width"),
            height: idx("height"),
            inline_size: idx("inline-size"),
            block_size: idx("block-size"),
            overflow_x: idx("overflow-x"),
            overflow_y: idx("overflow-y"),
            overflow_block: idx("overflow-block"),
            overflow_inline: idx("overflow-inline"),
        }
    })
}

/// Fold logical longhands into their physical slots after the cascade.
///
/// Each logical property keeps its own computed value in its slot (per
/// CSS Logical Properties 1 the computed value stays flow-relative);
/// the physical slot it maps to under the element's own computed
/// `direction`/`writing-mode` receives a clone of that value. When
/// both the logical longhand and its physical counterpart are
/// specified, the declaration later in the cascade wins (tracked via
/// the winner sequence numbers); an unspecified logical property never
/// disturbs the physical slot. A declaration that is invalid at
/// computed-value time is treated as absent for the fold (its sequence
/// is demoted to `u32::MAX`), so a valid counterpart still wins.
///
/// `winners` is the node's [`NodeWinners`]; `props` is the node's
/// fully computed slot array (initial values filled in).
fn apply_logical_mapping(
    props: &mut [Option<Arc<CssValue>>],
    winners: &[(u8, Cow<'_, CssValue>, u32)],
    invalid: &[bool; 256],
) {
    // Cascade sequence per property index; `u32::MAX` = unspecified or
    // invalid at computed-value time. An invalid winner keeps its
    // (initial-valued) slot but must not disturb the fold: treating it
    // as absent lets the valid counterpart declaration win, matching
    // the "invalid at computed-value time behaves as unset" rule.
    let mut seq = [u32::MAX; 256];
    for (i, _, s) in winners {
        seq[*i as usize] = if invalid[*i as usize] { u32::MAX } else { *s };
    }
    let map = logical_map();
    let wm = writing_mode_of(props, map.writing_mode);
    let rtl = direction_is_rtl(props, map.direction);
    let horizontal = wm == WritingMode::HorizontalTb;

    // `inline-size`/`block-size` map to `width`/`height` by axis, not by
    // side: inline-size is the `width` in horizontal writing modes and
    // the `height` in vertical ones; block-size is the other.
    for (lidx, inline_axis) in [(map.inline_size, true), (map.block_size, false)] {
        let lseq = seq[lidx as usize];
        if lseq == u32::MAX {
            continue;
        }
        let pidx = if inline_axis == horizontal {
            map.width
        } else {
            map.height
        };
        if seq[pidx as usize] == u32::MAX || lseq > seq[pidx as usize] {
            let v = props[lidx as usize].clone();
            props[pidx as usize] = v;
        }
    }

    // `overflow-block`/`overflow-inline` (CSS Overflow 3) map to the
    // physical axes wholesale, with no start/end split: in horizontal
    // writing modes inline is the `x` axis and block the `y` axis; in
    // vertical modes they swap. `direction` never flips the mapping
    // (the block axis direction does not change which physical axis it
    // is). Same winner-sequence tiebreak as above.
    for (lidx, inline_axis) in [(map.overflow_inline, true), (map.overflow_block, false)] {
        let lseq = seq[lidx as usize];
        if lseq == u32::MAX {
            continue;
        }
        let pidx = if inline_axis == horizontal {
            map.overflow_x
        } else {
            map.overflow_y
        };
        if seq[pidx as usize] == u32::MAX || lseq > seq[pidx as usize] {
            let v = props[lidx as usize].clone();
            props[pidx as usize] = v;
        }
    }

    for &(lidx, base, inline_axis, start) in &map.sides {
        let lseq = seq[lidx as usize];
        if lseq == u32::MAX {
            continue;
        }
        let pidx = base + logical_side_idx(wm, rtl, inline_axis, start) as u8;
        if seq[pidx as usize] == u32::MAX || lseq > seq[pidx as usize] {
            let v = props[lidx as usize].clone();
            props[pidx as usize] = v;
        }
    }
}

/// Cross-axis computed-value adjustment for `overflow-x`/`overflow-y`
/// (CSS Overflow 3 §3): if one axis is `visible` and the other is
/// neither `visible` nor `clip`, the `visible` axis computes to `auto`.
/// `clip` is never adjusted and never triggers the adjustment, and
/// combinations of `hidden`/`scroll`/`auto` are untouched (WPT
/// `css-overflow/parsing/overflow-computed.html`).
///
/// Runs on the physical slots after `apply_logical_mapping` (so logical
/// `overflow-block`/`overflow-inline` values participate) and before the
/// memo batch write in `compute`. The same adjustment also runs inside
/// [`Styler::finish_prop`] when either axis is resolved on the cold
/// path (see `finish_overflow_prop`), so a value memoized by — or
/// inherited through — a cold single-axis resolution is adjusted too,
/// not just the values a full `compute` produces.
fn fixup_overflow_axes(props: &mut [Option<Arc<CssValue>>]) {
    fn kw(props: &[Option<Arc<CssValue>>], idx: u8) -> Option<&str> {
        match props[idx as usize].as_deref() {
            Some(CssValue::Keyword(k)) => Some(k.as_str()),
            _ => None,
        }
    }
    let (x, y) = match (kw(props, props::OVERFLOW_X), kw(props, props::OVERFLOW_Y)) {
        (Some(x), Some(y)) => (x.to_string(), y.to_string()),
        _ => return,
    };
    let nx = overflow_adjusted_keyword(&x, &y);
    let ny = overflow_adjusted_keyword(&y, &x);
    if nx != x {
        props[props::OVERFLOW_X as usize] = Some(Arc::new(CssValue::Keyword(nx.into())));
    }
    if ny != y {
        props[props::OVERFLOW_Y as usize] = Some(Arc::new(CssValue::Keyword(ny.into())));
    }
}

/// One axis's cross-axis adjustment (see [`fixup_overflow_axes`]): a
/// side's adjustment depends only on the other side's *original*
/// keyword, so both sides adjust against the pre-adjustment pair.
fn overflow_adjusted_keyword<'a>(v: &'a str, other: &str) -> &'a str {
    match v {
        "visible" if !matches!(other, "visible" | "clip") => "auto",
        v => v,
    }
}

/// The cross-axis adjustment applied to a pair of not-yet
/// initial-filled axis values (the `finish_prop` form): `None` reads
/// as the property's initial value for the adjustment, and a component
/// the adjustment does not change keeps its original form (`None`
/// stays `None`), so the memo's unset-vs-initial distinction is
/// preserved exactly as in [`MemoSet::set`].
fn adjust_overflow_pair(
    x: Option<Arc<CssValue>>,
    y: Option<Arc<CssValue>>,
) -> (Option<Arc<CssValue>>, Option<Arc<CssValue>>) {
    fn effective(value: &Option<Arc<CssValue>>, idx: u8) -> Option<&CssValue> {
        match value {
            Some(v) => Some(v.as_ref()),
            None => initial_ref(idx),
        }
    }
    let (x_kw, y_kw) = match (
        effective(&x, props::OVERFLOW_X),
        effective(&y, props::OVERFLOW_Y),
    ) {
        (Some(CssValue::Keyword(xk)), Some(CssValue::Keyword(yk))) => (xk.as_str(), yk.as_str()),
        _ => return (x, y),
    };
    let nx = overflow_adjusted_keyword(x_kw, y_kw);
    let ny = overflow_adjusted_keyword(y_kw, x_kw);
    let new_x = if nx != x_kw {
        Some(Arc::new(CssValue::Keyword(nx.into())))
    } else {
        x
    };
    let new_y = if ny != y_kw {
        Some(Arc::new(CssValue::Keyword(ny.into())))
    } else {
        y
    };
    (new_x, new_y)
}

/// One epoch-tagged memo entry: the style-pass epoch the value was
/// computed in, plus the value. Probes only trust entries whose epoch
/// matches the styler's current epoch (see [`Styler::epoch`]).
type EpochEntry<V> = (u64, V);

/// `Rc`-shared custom-property environment: one node's merged
/// custom properties (see [`Styler::env`]).
type EnvRc = Rc<FxHashMap<String, CssValue>>;

/// One occupied slot of a [`NodeSlab`]: the node the value was memoized
/// for, the style-pass epoch it was written in, and the value. The
/// stored [`NodeId`] — not just its arena index — gates every probe:
/// indexes are dense per document but collide across documents, so a
/// node from a different document must miss, exactly like the old
/// `HashMap<NodeId, _>` keyed on the full `NodeId`.
struct SlabSlot<V> {
    node: NodeId,
    epoch: u64,
    value: V,
}

/// Index-addressed per-node memo slab replacing the old
/// `HashMap<NodeId, _>` memos: one `Vec` probe replaces a hash probe
/// per styled node. Slots are addressed by [`NodeId::index()`] and grow
/// on demand; each occupied slot records the full `NodeId` it was
/// written for, so a `NodeId` from a different document (whose arena
/// index collides with a local node's) probes as a miss and its insert
/// re-tags the slot instead of corrupting a local entry.
///
/// `epoch` versions the slot for the style-pass epoch (see
/// [`Styler::epoch`]): probes pass the current epoch and treat older
/// slots as misses; stale slots are overwritten on next use, so a
/// [`Styler::set_element_state`] bump never walks the slabs. `states`
/// is the exception — element state must survive epoch bumps — so it
/// uses the `_untagged` methods, which gate on the `NodeId` only.
struct NodeSlab<V> {
    slots: Vec<Option<SlabSlot<V>>>,
}

impl<V> Default for NodeSlab<V> {
    fn default() -> Self {
        NodeSlab { slots: Vec::new() }
    }
}

impl<V> NodeSlab<V> {
    /// Borrow the value memoized for `node` in `epoch`: `None` when the
    /// slot is empty, holds another node's entry, or is stale.
    fn probe(&self, node: NodeId, epoch: u64) -> Option<&V> {
        self.slots
            .get(node.index())
            .and_then(|s| s.as_ref())
            .and_then(|s| (s.node == node && s.epoch == epoch).then_some(&s.value))
    }

    /// [`probe`](Self::probe) without the epoch gate, for memos that are
    /// never epoch-tagged (`states`).
    fn probe_untagged(&self, node: NodeId) -> Option<&V> {
        self.slots
            .get(node.index())
            .and_then(|s| s.as_ref())
            .and_then(|s| (s.node == node).then_some(&s.value))
    }

    /// Memoize `value` for `node` in `epoch`, growing the slab when
    /// `node`'s index is beyond its end. Overwrites whatever the slot
    /// held (empty, foreign, or stale).
    fn insert(&mut self, node: NodeId, epoch: u64, value: V) {
        let idx = node.index();
        if idx >= self.slots.len() {
            self.slots.resize_with(idx + 1, || None);
        }
        self.slots[idx] = Some(SlabSlot { node, epoch, value });
    }

    /// [`insert`](Self::insert) without epoch tagging.
    fn insert_untagged(&mut self, node: NodeId, value: V) {
        self.insert(node, 0, value);
    }

    /// Borrow the slot value for `node` in `epoch`, installing `fresh()`
    /// when the slot is empty, foreign, or stale. A stale slot is reset,
    /// never trusted — the same retag-and-clear the old `HashMap::entry`
    /// call sites did.
    fn slot_or_insert_with(
        &mut self,
        node: NodeId,
        epoch: u64,
        fresh: impl FnOnce() -> V,
    ) -> &mut V {
        let idx = node.index();
        if idx >= self.slots.len() {
            self.slots.resize_with(idx + 1, || None);
        }
        let slot = &mut self.slots[idx];
        if !matches!(slot, Some(s) if s.node == node && s.epoch == epoch) {
            *slot = Some(SlabSlot {
                node,
                epoch,
                value: fresh(),
            });
        }
        &mut slot.as_mut().unwrap().value
    }

    /// Drop every slot, keeping the allocation: the epoch-counter
    /// wrap-around path in [`Styler::set_element_state`], matching the
    /// old `HashMap::clear` on every memo.
    fn clear(&mut self) {
        self.slots.clear();
    }
}

/// Mix one `u64` into a sharing-key accumulator with a splitmix64-grade
/// avalanche: cheap, deterministic (no per-process seed, so keys are
/// stable across runs), and strong enough that accidental collisions
/// between distinct match contexts are not a practical concern.
fn share_mix(h: u64, v: u64) -> u64 {
    let mut x = h ^ v;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    x
}

/// Fold a string into a sharing-key accumulator: FNV-1a over the bytes,
/// then [`share_mix`].
fn share_str(h: u64, s: &str) -> u64 {
    let mut x = 0xcbf2_9ce4_8422_2325u64;
    for b in s.as_bytes() {
        x ^= *b as u64;
        x = x.wrapping_mul(0x0100_0000_01b3);
    }
    share_mix(h, x)
}

/// Discriminant for [`Namespace`], folded into sharing keys: type
/// selectors test the element's namespace, so it is match-relevant.
/// Custom namespaces (see [`Namespace::Other`]) share one discriminant:
/// selector prefixes only resolve to the built-in namespaces, so distinct
/// custom URIs are indistinguishable to matching.
fn share_ns_disc(ns: &Namespace) -> u64 {
    match ns {
        Namespace::Html => 0,
        Namespace::MathMl => 1,
        Namespace::Svg => 2,
        Namespace::Other(_) => 3,
    }
}

/// Which selector features occur anywhere in the styler's stylesheets,
/// scanned once at construction. The style-sharing key only captures
/// the context a selector can actually observe: with no structural
/// pseudo-classes anywhere, for example, sibling positions cannot
/// affect matching and are left out of the key, keeping it cheap.
#[derive(Clone, Copy, Debug, Default)]
struct ShareFlags {
    /// Some selector uses `+` or `~`: matching can inspect preceding
    /// siblings, so the key folds in their signatures.
    sibling: bool,
    /// Some selector uses `:nth-child`, `:first-child`, `:only-child`
    /// (or the `-of-type` / `-last-` variants), or `:disabled` /
    /// `:enabled` (whose first-`legend`-in-a-disabled-`fieldset`
    /// exception observes sibling position): the key carries the
    /// element's sibling position.
    structural: bool,
    /// Some selector uses `:empty`: the key carries the emptiness bit.
    empty: bool,
    /// Some selector uses `:hover`, `:active`, `:focus`, `:link`, or
    /// `:visited`: the key carries element-state bits.
    dynamic: bool,
    /// Some selector uses `:has()`: matching can inspect descendants,
    /// which no fixed-size key can capture — sharing is disabled.
    has: bool,
}

/// True when every selector in the UA + author stylesheets matches a
/// node based only on the node itself and its ancestors: no sibling
/// combinators (`+`, `~`) and no `:has()` anywhere (including inside
/// `:not()` / `:is()` / `:where()` arguments).
///
/// Under this condition an attribute mutation on one element can only
/// change selector matching for that element and its descendants —
/// never for siblings, ancestors, or anything else outside its
/// subtree. (Attribute writes change no node's attributes but the
/// element's own, and change no structure or element state; the only
/// selector constructs that observe other elements' attributes are
/// the sibling combinators — in either direction — and `:has()`.)
/// The incremental pipeline (`bowser_app::window::try_partial_layout`)
/// uses this to admit style-affecting attribute writes to the splice
/// path: the subtree pass re-matches the subtree from the current
/// DOM, and nothing outside it can have restyled.
///
/// Reuses the audited [`ShareFlags::scan`] walk, which already flags
/// exactly these two constructs.
pub fn selectors_are_subtree_local(ua: &Stylesheet, author: &[&Stylesheet]) -> bool {
    let flags = ShareFlags::scan(ua, author);
    !flags.sibling && !flags.has
}

impl ShareFlags {
    fn scan(ua: &Stylesheet, author: &[&Stylesheet]) -> Self {
        let mut flags = ShareFlags::default();
        for sheet in std::iter::once(ua).chain(author.iter().copied()) {
            for rule in sheet.rules() {
                for selector in &rule.selectors {
                    flags.scan_selector(selector);
                }
            }
        }
        flags
    }

    fn scan_selector(&mut self, selector: &Selector) {
        for (i, compound) in selector.compounds.iter().enumerate() {
            if i > 0
                && let Some(combinator) = selector.combinators.get(i - 1)
                && matches!(
                    combinator,
                    Combinator::NextSibling | Combinator::SubsequentSibling
                )
            {
                self.sibling = true;
            }
            for pseudo in &compound.pseudos {
                self.scan_pseudo(pseudo);
            }
        }
    }

    fn scan_pseudo(&mut self, pseudo: &PseudoClass) {
        match pseudo {
            PseudoClass::Has(_) => {
                self.has = true;
            }
            PseudoClass::Hover
            | PseudoClass::Active
            | PseudoClass::Focus
            | PseudoClass::Link
            | PseudoClass::Visited => {
                self.dynamic = true;
            }
            PseudoClass::NthChild(..)
            | PseudoClass::NthLastChild(..)
            | PseudoClass::NthOfType(..)
            | PseudoClass::NthLastOfType(..)
            | PseudoClass::FirstChild
            | PseudoClass::LastChild
            | PseudoClass::FirstOfType
            | PseudoClass::LastOfType
            | PseudoClass::OnlyChild
            | PseudoClass::OnlyOfType => {
                self.structural = true;
            }
            PseudoClass::Empty => {
                self.empty = true;
            }
            // `:disabled` / `:enabled` observe ancestor state: a
            // `fieldset[disabled]` (or `optgroup[disabled]`) disables
            // descendants except those in its *first* `legend` child
            // (see `in_first_legend` in matching.rs), so matching
            // depends on the legend's sibling position — context the
            // sharing key only captures via the structural positions.
            PseudoClass::Disabled | PseudoClass::Enabled => {
                self.structural = true;
            }
            // `:not()` / `:is()` / `:where()` arguments are matched
            // against the same element the compound is tested on (which
            // may itself be an ancestor or sibling of the subject), so
            // their combinators and pseudo-classes count identically.
            PseudoClass::Not(args) | PseudoClass::Is(args) | PseudoClass::Where(args) => {
                for arg in args {
                    self.scan_selector(arg);
                }
            }
            // Audited context-free: `:root` matches only the document
            // element, whose parent (the document node) already
            // distinguishes it in the parent-key chain; `:checked` is a
            // pure function of the element's own tag and attributes
            // (`type`/`checked`/`selected`), all folded into the subject
            // signature. Neither needs a flag.
            _ => {}
        }
    }
}

impl<'a> Styler<'a> {
    /// Stylesheet-level scan for `::before` / `::after` rules: the gate
    /// behind [`Styler::has_generated_content_rules`]. A selector
    /// targets a pseudo-element only from its rightmost compound (see
    /// [`matches_pseudo_element`]), so that is all the scan checks —
    /// one pass over every selector, once per document.
    fn scan_generated_content(ua: &Stylesheet, author: &[&Stylesheet]) -> bool {
        std::iter::once(ua)
            .chain(author.iter().copied())
            .any(|sheet| {
                sheet.rules().iter().any(|rule| {
                    rule.selectors.iter().any(|sel| {
                        sel.compounds.last().is_some_and(|c| {
                            matches!(c.pseudo_element.as_deref(), Some("before") | Some("after"))
                        })
                    })
                })
            })
    }

    /// Stylesheet-level scan for `counter-reset` / `counter-increment`
    /// declarations: the gate behind
    /// [`Styler::has_counter_adjustments`]. One pass over every
    /// declaration, once per document.
    fn scan_counter_adjustments(ua: &Stylesheet, author: &[&Stylesheet]) -> bool {
        std::iter::once(ua)
            .chain(author.iter().copied())
            .any(|sheet| {
                sheet.rules().iter().any(|rule| {
                    rule.declarations.iter().any(|decl| {
                        matches!(
                            decl.property.as_str(),
                            "counter-reset" | "counter-increment"
                        )
                    })
                })
            })
    }
}

/// Memoized per-parent sibling context for style-sharing keys, built
/// with one child walk per parent per epoch. Positions serve the
/// structural pseudo-classes; the running `prefix` hash folds every
/// preceding element sibling's signature (plus its own position and
/// preceding context, so nested `~` chains stay sound) for the sibling
/// combinators.
#[derive(Clone, Debug, Default)]
struct ShareSibCtx {
    epoch: u64,
    /// 1-based index among element siblings, in document order.
    index: FxHashMap<NodeId, u32>,
    /// 1-based index from the end among element siblings.
    rindex: FxHashMap<NodeId, u32>,
    /// 1-based index among same-type element siblings.
    type_index: FxHashMap<NodeId, u32>,
    /// 1-based index from the end among same-type element siblings.
    type_rindex: FxHashMap<NodeId, u32>,
    /// Prefix hash before each element child: the folded signatures of
    /// all preceding element siblings.
    prefix: FxHashMap<NodeId, u64>,
}

/// One `::marker` style-sharing cache entry: the shared style plus the
/// epoch it was computed in (hash-keyed entries never self-overwrite,
/// so the epoch is checked on probe and the whole cache is dropped on
/// every style-pass epoch bump).
struct MarkerMemoEntry {
    epoch: u64,
    style: Rc<ComputedStyle>,
}

/// Apply normalized pseudo-element winners over a base style: one
/// `set_slot` per winning slot. Winners are already sorted by slot
/// and carry the shared initial `Arc` for `initial` winners (which
/// `set_slot` turns back into a cleared slot, exactly like the old
/// `marker_style_for` loop), so this is that application loop factored
/// out for the memoized and unmemoized paths.
fn apply_pseudo_winners(
    mut style: ComputedStyle,
    winners: &[(u8, Arc<CssValue>)],
) -> ComputedStyle {
    for (idx, value) in winners {
        style.set_slot(*idx, Some(Arc::clone(value)));
    }
    style
}

/// Memo key for a `::marker` style: the style-pass epoch, the
/// originating computed style's identity, and the resolved winners.
/// See [`Styler::marker_style_for_shared`] for the soundness argument.
fn marker_memo_key(epoch: u64, base: &Rc<ComputedStyle>, winners: &[(u8, Arc<CssValue>)]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    h = share_mix(h, epoch);
    // Pointer identity: `style_for_shared` hands out the interned `Rc`,
    // so equal pointers mean byte-identical styles.
    h = share_mix(h, Rc::as_ptr(base) as usize as u64);
    h = share_mix(h, winners.len() as u64);
    for (idx, value) in winners {
        h = share_mix(h, *idx as u64);
        h = hash_css_value(h, value);
    }
    h
}

/// Memo key for a `::before` / `::after` style: the [`marker_memo_key`]
/// inputs plus the pseudo-element name. `::before` and `::after`
/// winners are computed separately per pseudo ([`Styler::pseudo_winners`]
/// filters rules by the pseudo name), so the name joins the key: equal
/// keys imply byte-identical pseudo styles by the same soundness
/// argument as [`Styler::marker_style_for_shared`].
fn pseudo_memo_key(
    epoch: u64,
    pseudo: &str,
    base: &Rc<ComputedStyle>,
    winners: &[(u8, Arc<CssValue>)],
) -> u64 {
    share_str(marker_memo_key(epoch, base, winners), pseudo)
}

/// Memo key for the cheap [`Styler::pseudo_style_for_shared`] probe:
/// the node's style-sharing key plus the pseudo-element name. Equal
/// sharing keys imply equal pseudo winners (see the soundness argument
/// on [`Styler::pseudo_style_for_shared`]), so this key needs no winner
/// hash — the expensive part the probe exists to skip.
///
/// Domain-separated from [`pseudo_memo_key`] (which folds the epoch
/// first; this folds a fixed tag), so an entry stored under one key
/// scheme can never be read through the other.
fn pseudo_share_memo_key(share_key: u64, pseudo: &str) -> u64 {
    share_str(share_mix(share_key, 0x9e37_79b9_7f4a_7c15), pseudo)
}

/// Fold a resolved [`CssValue`] into the marker-memo key hash.
/// Discriminant-first so variants never collide; floats hash by bits.
/// Exotic variants (math trees, shadows, transforms — unreachable for
/// the pseudo-allowed properties today) hash their `Debug` rendering,
/// which is deterministic within a build, rather than panicking on a
/// future variant.
fn hash_css_value(h: u64, value: &CssValue) -> u64 {
    match value {
        CssValue::Keyword(k) => share_str(share_mix(h, 0), k.as_str()),
        CssValue::Length(n, unit) => {
            let h = share_mix(share_mix(h, 1), n.to_bits() as u64);
            // `LengthUnit` is not `Copy`; map the variants explicitly.
            let u = match unit {
                LengthUnit::Px => 0,
                LengthUnit::Cm => 1,
                LengthUnit::Mm => 2,
                LengthUnit::In => 3,
                LengthUnit::Pt => 4,
                LengthUnit::Pc => 5,
                LengthUnit::Q => 6,
                LengthUnit::Em => 7,
                LengthUnit::Rem => 8,
                LengthUnit::Ex => 9,
                LengthUnit::Vw => 10,
                LengthUnit::Vh => 11,
                LengthUnit::Vmin => 12,
                LengthUnit::Vmax => 13,
            };
            share_mix(h, u)
        }
        CssValue::Percentage(n) => share_mix(share_mix(h, 2), n.to_bits() as u64),
        CssValue::Color(c) => share_mix(
            share_mix(h, 3),
            u32::from_le_bytes([c.r, c.g, c.b, c.a]) as u64,
        ),
        CssValue::String(s) => share_str(share_mix(h, 4), s),
        CssValue::Url(s) => share_str(share_mix(h, 5), s),
        CssValue::Number(n) => share_mix(share_mix(h, 6), n.to_bits() as u64),
        CssValue::List(items) => {
            let mut h = share_mix(share_mix(h, 7), items.len() as u64);
            for item in items {
                h = hash_css_value(h, item);
            }
            h
        }
        // Anything else cannot survive resolution for the
        // pseudo-allowed properties; hash the `Debug` form instead of
        // failing to compile on a new variant.
        other => share_str(share_mix(h, 8), &format!("{other:?}")),
    }
}

/// One style-sharing cache entry: everything [`Styler::compute`]
/// derives from the match-relevant inputs. An element whose sharing
/// key equals the donor's adopts this style outright. `env` is
/// value-equal for equal keys (custom-property environments are a pure
/// function of the ancestor key chain); sharing the `Rc` is safe
/// because [`Styler::custom_env`] clones-on-write through
/// `Rc::make_mut` before extending it.
struct ShareEntry {
    epoch: u64,
    style: Rc<ComputedStyle>,
    env: EnvRc,
}

/// The renderable text of a pseudo-element's computed `content` value:
/// `Some(text)` for a `<string>` value, `None` for `normal` / `none`
/// (an absent slot reads as the initial `normal`). Generated-content
/// box construction (`::before` / `::after`) uses this to decide
/// whether a pseudo box is synthesized and what text it holds.
pub fn content_text(style: &ComputedStyle) -> Option<&str> {
    match style.get_by_index(props::CONTENT) {
        Some(CssValue::String(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// The `(name, adjustment)` pairs of a computed `counter-reset` /
/// `counter-increment` value: `none` (the initial, read from an absent
/// slot) means no adjustments. `default` is the adjustment for a name
/// with no explicit integer: `0` for `counter-reset`, `1` for
/// `counter-increment` (CSS Lists 3 §4). Values that fail the
/// `[ <custom-ident> <integer>? ]+ | none` grammar never reach a
/// computed style; an unexpected shape degrades to no adjustments
/// rather than panicking. Used by `bowser_layout` to build the counter
/// scopes that `counter()` / `counters()` in `content` resolve against
/// (see `pseudo::resolve_content_text`).
pub fn counter_adjustments(value: Option<&CssValue>, default: i64) -> Vec<(String, i64)> {
    let items = match value {
        None => return Vec::new(),
        Some(CssValue::Keyword(k)) if k.as_str() == "none" => return Vec::new(),
        // A single `<custom-ident>` parses to a bare keyword (the
        // generic parser only wraps multi-item values in a list).
        Some(CssValue::Keyword(k)) => return vec![(k.as_str().to_owned(), default)],
        Some(CssValue::List(items)) => items,
        Some(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < items.len() {
        let name = match &items[i] {
            CssValue::Keyword(k) => k.as_str().to_owned(),
            _ => return Vec::new(),
        };
        i += 1;
        let delta = match items.get(i) {
            Some(CssValue::Number(n)) => {
                i += 1;
                // Saturate: grammar validation already rejected
                // non-integers, but a huge finite float would
                // overflow the cast.
                n.clamp(-9e18, 9e18) as i64
            }
            _ => default,
        };
        out.push((name, delta));
    }
    out
}

/// Batch style computer for one document.
///
/// A `Styler` is bound — by borrow — to exactly one document, one author
/// stylesheet set, one UA stylesheet, and one viewport. Its memoization
/// caches (cascade winners, custom-property environments, resolved values)
/// persist across [`style_for`](Self::style_for) calls, so styling a whole
/// document root-first is linear in its size: each element reuses its
/// ancestors' memoized results instead of re-walking the ancestor chain.
///
/// Because the document, stylesheets, and viewport are borrowed for the
/// `Styler`'s lifetime, memoized results can never leak across documents
/// or stylesheet/viewport configurations; build a new `Styler` per
/// document. [`compute_style`] is the one-shot equivalent for single
/// queries.
///
/// Dynamic pseudo-classes (`:hover`, `:active`, `:focus`, `:link`,
/// `:visited`) match against each node's [`ElementState`], set with
/// [`set_element_state`](Self::set_element_state); nodes without an
/// explicit state use [`ElementState::default()`] (all false), which is
/// exactly the old static behavior.
/// Per-node `(name id, case)` map for [`Styler::var_memo`]: the epoch-tagged
/// substituted value of each custom property on the node. The slash/comma
/// [`MemoVar`] payloads are `Rc`-shared, so cloning a memo entry (per
/// `var()` probe) is a pointer bump instead of a deep clone of the group
/// `Vec`s.
type VarMemoMap = FxHashMap<(u32, bool), EpochEntry<MemoVar>>;

pub struct Styler<'a> {
    doc: &'a Document,
    /// The document's quirks mode, read from [`Document::quirks_mode`]
    /// at construction (w77 quirks plumbing): inheritance breaks at
    /// the table boundary consult this, so style and layout branch on
    /// the same parsed mode.
    quirks_mode: QuirksMode,
    author: &'a [&'a Stylesheet],
    ua: &'a Stylesheet,
    viewport: Viewport,
    /// Rightmost-compound rule indexes, built once per stylesheet: the UA
    /// sheet first, then the author sheets in document order. Keeps
    /// candidate matching linear in the selectors that can possibly match
    /// an element instead of quadratic in rules x nodes.
    rule_indexes: Vec<RuleIndex>,
    /// Style-pass epoch, bumped by
    /// [`set_element_state`](Self::set_element_state). Every memo entry
    /// below is tagged with the epoch it was computed in; probes treat
    /// entries from an older epoch as misses. This makes state-change
    /// invalidation O(1) — one counter bump instead of walking the slabs
    /// — with identical observable behavior: every memo depends on
    /// element state (matching reads `states`, everything downstream
    /// reads matching), so a state change must retire all of them.
    /// Stale entries are overwritten on next use, so memory stays
    /// bounded by the styled-node set.
    epoch: Cell<u64>,
    /// Per-node interactive state for dynamic pseudo-class matching.
    /// Absent entries mean [`ElementState::default()`]. Index-addressed
    /// [`NodeSlab`]: a `Vec` probe replaces the hash probe, and the
    /// slab is never epoch-tagged (state must survive epoch bumps).
    states: RefCell<NodeSlab<ElementState>>,
    /// Sorted cascade candidates per node (before `var()` substitution).
    /// `Rc`-shared: memo hits hand out a pointer bump instead of cloning
    /// every candidate's property name and value. Epoch-tagged: see
    /// [`Styler::epoch`]. Index-addressed [`NodeSlab`].
    cands: RefCell<NodeSlab<Rc<Vec<Candidate<'a>>>>>,
    /// Cascade winners per node with deferred `var()`-in-shorthand
    /// declarations substituted and expanded to longhands,
    /// index-addressed by [`COMPUTED_PROPS`] position: a plain array
    /// read replaces the old per-property `String`-keyed hash lookup
    /// (one hashed lookup per property per element). `Rc`-shared like
    /// [`Styler::cands`]. Epoch-tagged: see [`Styler::epoch`].
    /// Index-addressed [`NodeSlab`].
    expanded: RefCell<NodeSlab<NodeWinners<'a>>>,
    /// Custom-property environment per node: ancestors' merged with own.
    /// `Rc`-shared: a node that declares no custom properties reuses its
    /// parent's map outright, and memo hits cost a pointer bump instead
    /// of cloning the whole merged map. Epoch-tagged: see
    /// [`Styler::epoch`]. Index-addressed [`NodeSlab`].
    env: RefCell<NodeSlab<EnvRc>>,
    /// Memoized resolved property values: a single slab probe per node
    /// replaces the old per-property `(NodeId, u8)` key hashing. Each
    /// node's [`MemoSet`] sparsely stores only the values deviating
    /// from the initial fallback (its known-bitset distinguishes
    /// unknown from known-none), instead of a boxed 180-slot array.
    /// Epoch-tagged: see [`Styler::epoch`].
    resolved: RefCell<NodeSlab<MemoSet>>,
    /// Memoized resolved custom-property values, keyed by node, interned
    /// property name, and identifier-case mode: the substituted value of
    /// `--name` on `node` (`None` = invalid at computed-value time).
    /// Makes `var()` chains linear instead of cubic per node. The `u32`
    /// is a [`VarNameInterner`] id, so memo probes never clone the name.
    /// The case flag is part of the key because the deferred `font`
    /// shorthand resolves with case preservation while every other
    /// property uses the lowercasing parse; the two parses must never
    /// share cache entries. Epoch-tagged: see [`Styler::epoch`]. The
    /// outer per-node level is an index-addressed [`NodeSlab`]; each
    /// node's inner `(name id, case)` map keeps its epoch-tagged
    /// entries, exactly like the old flat `(NodeId, u32, bool)` key.
    var_memo: RefCell<NodeSlab<VarMemoMap>>,
    /// Scratch cycle-detection set for [`Styler::resolve`], reused across
    /// calls instead of allocating a fresh set per resolution. Keyed by
    /// interned name id. Always empty between calls: `resolve` removes
    /// every id it inserts.
    resolve_active: RefCell<FxHashSet<u32>>,
    /// Custom-property name interner shared by all `var()` resolution.
    var_names: RefCell<VarNameInterner>,
    /// Memoized computed `font-size` in px per node. `em` units and
    /// `font-size`/`line-height` percentages in math expressions resolve
    /// against these; memoizing keeps ancestor walks linear.
    /// Epoch-tagged: see [`Styler::epoch`]. Index-addressed [`NodeSlab`].
    font_px: RefCell<NodeSlab<f32>>,
    /// Memoized finished [`ComputedStyle`] per node, epoch-tagged like
    /// the other memos. The style is `Rc`-shared: `style_for` on an
    /// already-styled node is a slab probe plus a pointer clone instead
    /// of recomputing (or deep-cloning) all properties, and
    /// `marker_style_for` reuses it for its base style instead of
    /// recomputing the element's whole style a second time.
    style_memo: RefCell<NodeSlab<Rc<ComputedStyle>>>,
    /// Memoized `::marker` pseudo-element styles, shared across list
    /// items. Hash-keyed by ([`marker_memo_key`]): the style-pass
    /// epoch, the originating element's computed-style identity, and
    /// the resolved `::marker` winners — list items matching the same
    /// `::marker` rules (with the same originating style) share one
    /// `Rc<ComputedStyle>` instead of each paying the pseudo cascade
    /// plus a base-style clone. Hash-keyed like `share_cache` (entries
    /// never self-overwrite), so it is dropped on every style-pass
    /// epoch bump; see `set_element_state`.
    marker_memo: RefCell<FxHashMap<u64, MarkerMemoEntry>>,
    /// Memoized `::before` / `::after` pseudo-element styles, shared
    /// across elements. Two key schemes share the map: the cheap probe
    /// keys by ([`pseudo_share_memo_key`]) — the node's style-sharing
    /// key plus the pseudo-element name (`"before"` / `"after"`) — and
    /// runs before winners are computed, so a hit skips the pseudo
    /// cascade entirely; the fallback keys by ([`pseudo_memo_key`]) —
    /// the style-pass epoch, the pseudo name, the originating style's
    /// identity, and the resolved winners — for nodes with no sharing
    /// key. Either way, elements with equal keys share one
    /// `Rc<ComputedStyle>`. The stored style is the post-reset style
    /// (the `display` / `content` / `unicode-bidi` resets are constant
    /// per pseudo, so memoizing after applying them is sound).
    /// Hash-keyed like `marker_memo` (entries never self-overwrite), so
    /// it is dropped on every style-pass epoch bump; see
    /// `set_element_state`.
    pseudo_memo: RefCell<FxHashMap<u64, MarkerMemoEntry>>,
    /// Nodes whose `font-size` is currently being computed by
    /// [`Styler::font_size_px`]. Computing a font size can re-enter
    /// `font_size_px` for the same node (`font_size_px_for` ->
    /// `resolved_prop` -> `compute_prop` -> `eval_math_deep` ->
    /// `math_context` -> `font_size_px`, e.g.
    /// `html{font-size:calc(1rem + 1px)}`), and the memo insert happens
    /// after return — without this set the call recurses until the stack
    /// overflows on remote-stylesheet input. Re-entrant calls fail closed
    /// with the initial font size.
    font_px_active: RefCell<FxHashSet<NodeId>>,
    /// Per-parent memoized sibling indexes for the structural
    /// pseudo-classes (`:nth-child` et al.). One cache for the styler's
    /// whole lifetime: the styler borrows a single `Document`, so the
    /// parent-keyed index stays valid and repeated `:nth-child` queries
    /// against the same parent are O(1) amortized instead of O(siblings)
    /// per query.
    sib_cache: SiblingCache,
    /// Scratch buffer for [`RuleIndex::candidates_into`], reused across
    /// every query instead of allocating a fresh `Vec` per element per
    /// stylesheet.
    refs_scratch: RefCell<Vec<SelectorRef>>,
    /// Scratch space for selector matching (failed-state memo set and
    /// backtracking stack), reused across every [`matches_cascade`] call
    /// in the style pass instead of allocating per match. Always empty
    /// between calls: each matcher takes the buffers on entry and hands
    /// them back on exit.
    match_scratch: RefCell<MatchScratch>,
    /// Selector-dependency flags scanned once at construction: they
    /// decide which context the style-sharing key must capture.
    share_flags: ShareFlags,
    /// Stylesheet-level gate for `::before` / `::after` generated
    /// content: true when any UA or author selector targets those
    /// pseudo-elements (scanned once at construction, like
    /// `share_flags`). Box construction checks
    /// [`Styler::has_generated_content_rules`] before running its
    /// per-element pseudo query, so documents without generated-content
    /// rules pay ~zero in that hot loop.
    generated_content_rules: bool,
    /// Stylesheet-level gate for CSS counter scoping: true when any UA
    /// or author declaration sets `counter-reset` / `counter-increment`
    /// (scanned once at construction, like `generated_content_rules`).
    /// Box construction threads counter scopes only then; otherwise
    /// every `counter()` reads as 0. (Inline `style` adjustments
    /// without any stylesheet counter rule fall back to an ancestor
    /// fold per query — see `pseudo::resolve_content_text`.)
    counter_adjustment_rules: bool,
    /// Style-sharing cache: finished styles keyed by the hash of the
    /// match-relevant inputs (tag, namespace, attributes, state,
    /// sibling/ancestor context). Elements with identical inputs adopt
    /// the cached style instead of re-running rule matching, cascade
    /// expansion, and per-property resolution. Epoch-tagged like the
    /// other memos; disabled outright when any selector uses `:has()`.
    /// Bounded by the distinct-key set; entries hold `Rc`s, never
    /// duplicated styles.
    share_cache: RefCell<FxHashMap<u64, ShareEntry>>,
    /// Memoized sharing keys per node. Epoch-tagged through the slab:
    /// keys embed element state, so a state change (epoch bump)
    /// recomputes them. Computed iteratively up the ancestor chain, so
    /// deep DOMs cannot overflow the stack.
    share_keys: RefCell<NodeSlab<u64>>,
    /// Sorted cascade candidates per style-sharing key: elements whose
    /// match-relevant inputs hash equal share one `Rc<Vec<Candidate>>`
    /// instead of each building (and memoizing per node) an identical
    /// vec. Epoch-tagged; cleared on every epoch bump like
    /// [`share_cache`](Self::share_cache) (hash-keyed entries never
    /// self-overwrite, so a stale generation would leak otherwise).
    /// Bounded by the distinct-key set; entries hold `Rc`s, never
    /// duplicated candidate lists.
    cand_shares: RefCell<FxHashMap<u64, EpochEntry<Rc<Vec<Candidate<'a>>>>>>,
    /// Per-parent sibling context for sharing keys (positions for the
    /// structural pseudo-classes, prefix hashes for `+`/`~`), built once
    /// per parent per epoch. Epoch-tagged: preceding-sibling signatures
    /// embed state when dynamic pseudo-classes are in play.
    share_sib: RefCell<FxHashMap<NodeId, Rc<ShareSibCtx>>>,
    /// Parsed inline `style=""` declarations keyed by the raw attribute
    /// string: identical inline styles (common in generated pages) parse
    /// once instead of once per element. A pure function of the string,
    /// so no epoch tagging is needed; bounded by the distinct style
    /// strings in the document.
    inline_cache: RefCell<FxHashMap<String, Rc<Vec<Declaration>>>>,
    /// Test-only: style-sharing lookups that hit / missed, so tests can
    /// assert the cache engages deterministically.
    #[cfg(test)]
    share_hits: Cell<u64>,
    /// Test-only: style-sharing lookups that missed.
    #[cfg(test)]
    share_misses: Cell<u64>,
    /// Test-only: inline `style` attributes parsed (cache misses), so
    /// tests can assert the parse cache engages.
    #[cfg(test)]
    inline_parses: Cell<u64>,
    /// Test-only: cross-node candidate-vec shares that hit, so tests can
    /// assert the key-based candidate cache engages deterministically.
    #[cfg(test)]
    cand_key_hits: Cell<u64>,
}

impl<'a> Styler<'a> {
    /// Build a styler for `doc`: `author` stylesheets in document order,
    /// `ua` the user-agent stylesheet, `viewport` for `@media` evaluation.
    pub fn new(
        doc: &'a Document,
        author: &'a [&'a Stylesheet],
        ua: &'a Stylesheet,
        viewport: Viewport,
    ) -> Self {
        // One index per stylesheet, UA first, matching the `sheet_idx`
        // numbering `candidates` uses below.
        let mut rule_indexes = Vec::with_capacity(author.len() + 1);
        rule_indexes.push(RuleIndex::build(ua));
        for sheet in author {
            rule_indexes.push(RuleIndex::build(sheet));
        }
        // Stylesheet-level generated-content gate (see the
        // `generated_content_rules` field docs): one selector scan per
        // document, ~zero per-element cost when no `::before` /
        // `::after` rules exist.
        let generated_content_rules = Self::scan_generated_content(ua, author);
        // Stylesheet-level counter gate (see the
        // `counter_adjustment_rules` field docs): one declaration scan
        // per document, ~zero per-element scope maintenance when no
        // `counter-reset` / `counter-increment` rule exists.
        let counter_adjustment_rules = Self::scan_counter_adjustments(ua, author);
        Styler {
            doc,
            quirks_mode: doc.quirks_mode(),
            author,
            ua,
            viewport,
            rule_indexes,
            epoch: Cell::new(0),
            states: RefCell::new(NodeSlab::default()),
            cands: RefCell::new(NodeSlab::default()),
            expanded: RefCell::new(NodeSlab::default()),
            env: RefCell::new(NodeSlab::default()),
            resolved: RefCell::new(NodeSlab::default()),
            var_memo: RefCell::new(NodeSlab::default()),
            resolve_active: RefCell::new(FxHashSet::default()),
            var_names: RefCell::new(VarNameInterner::default()),
            font_px: RefCell::new(NodeSlab::default()),
            style_memo: RefCell::new(NodeSlab::default()),
            marker_memo: RefCell::new(FxHashMap::default()),
            pseudo_memo: RefCell::new(FxHashMap::default()),
            font_px_active: RefCell::new(FxHashSet::default()),
            sib_cache: SiblingCache::default(),
            refs_scratch: RefCell::new(Vec::new()),
            match_scratch: RefCell::new(MatchScratch::new()),
            share_flags: ShareFlags::scan(ua, author),
            generated_content_rules,
            counter_adjustment_rules,
            share_cache: RefCell::new(FxHashMap::default()),
            share_keys: RefCell::new(NodeSlab::default()),
            cand_shares: RefCell::new(FxHashMap::default()),
            share_sib: RefCell::new(FxHashMap::default()),
            inline_cache: RefCell::new(FxHashMap::default()),
            #[cfg(test)]
            share_hits: Cell::new(0),
            #[cfg(test)]
            share_misses: Cell::new(0),
            #[cfg(test)]
            inline_parses: Cell::new(0),
            // Test-only: cross-node candidate-vec shares that hit, so
            // tests can assert the key-based candidate cache engages
            // deterministically.
            #[cfg(test)]
            cand_key_hits: Cell::new(0),
        }
    }

    /// The computed style of `node`. Style nodes root-first so each call
    /// reuses its ancestors' memoized cascade and inheritance results.
    /// The finished style is memoized per node per style-pass epoch: a
    /// repeat query is a slab probe plus a clone, not a recomputation.
    pub fn style_for(&self, node: NodeId) -> ComputedStyle {
        (*self.style_for_shared(node)).clone()
    }

    /// The memoized [`ComputedStyle`] of `node`, shared by reference
    /// count. Same memoization as [`style_for`](Self::style_for), but the
    /// caller shares the memoized `Rc` instead of deep-cloning the
    /// `delta` entries on every call — one refcount bump per element per
    /// style pass instead of one heap allocation plus `Arc` bumps per
    /// delta entry. Layout threads this `Rc` through its box arena into
    /// the finished layout-box tree.
    pub fn style_for_shared(&self, node: NodeId) -> Rc<ComputedStyle> {
        let epoch = self.epoch.get();
        let hit = self.style_memo.borrow().probe(node, epoch).cloned();
        if let Some(rc) = hit {
            return rc;
        }
        let rc = self.compute(node);
        self.style_memo
            .borrow_mut()
            .insert(node, epoch, Rc::clone(&rc));
        rc
    }

    /// Set the interactive [`ElementState`] of `node`, used when matching
    /// the dynamic pseudo-classes (`:hover`, `:active`, `:focus`, `:link`,
    /// `:visited`). The renderer calls this on hover/focus/activation
    /// changes and re-styles the affected node(s).
    ///
    /// Nodes without an explicit state use [`ElementState::default()`]
    /// (all false): static matching, exactly as before. Changing a
    /// node's state bumps the style-pass epoch, retiring every memoized
    /// cascade result at once — every memo depends on state, including
    /// the merged custom-property environments of descendants — so the
    /// next [`style_for`](Self::style_for) recomputes with the new
    /// state. O(1) for the slabs: no slab is walked or cleared; stale
    /// entries fail their epoch check on probe and are overwritten on
    /// next use. The style-sharing cache is the exception: it is keyed
    /// by hash rather than by node, so stale generations are never
    /// overwritten and would leak a full generation of
    /// `Rc<ComputedStyle>` + environment entries per bump — it is
    /// dropped on every bump and rebuilds on the next pass.
    pub fn set_element_state(&self, node: NodeId, state: ElementState) {
        let changed = match self.states.borrow().probe_untagged(node).copied() {
            Some(prev) => prev != state,
            None => state != ElementState::default(),
        };
        if !changed {
            return;
        }
        self.states.borrow_mut().insert_untagged(node, state);
        let next = self.epoch.get().wrapping_add(1);
        if next == 0 {
            // The epoch counter wrapped (2^64 state changes on one
            // styler): epoch-0 entries from the styler's birth would
            // falsely probe as current, so drop every stale entry.
            self.cands.borrow_mut().clear();
            self.expanded.borrow_mut().clear();
            self.env.borrow_mut().clear();
            self.resolved.borrow_mut().clear();
            self.var_memo.borrow_mut().clear();
            self.font_px.borrow_mut().clear();
            self.style_memo.borrow_mut().clear();
            // The style-sharing cache, sharing keys, and sibling
            // contexts are epoch-tagged too: epoch 0 would falsely
            // probe as current after the wrap. The inline-style parse
            // cache is state-independent and survives.
            self.share_cache.borrow_mut().clear();
            self.share_keys.borrow_mut().clear();
            self.share_sib.borrow_mut().clear();
            self.cand_shares.borrow_mut().clear();
        }
        self.epoch.set(next);
        // The style-sharing cache is hash-keyed, so stale generations
        // are never overwritten (unlike the node-keyed slabs, which
        // self-overwrite per node): without this, every state change
        // leaks a full generation of `Rc<ComputedStyle>` + `Rc<HashMap>`
        // entries. Drop it on every bump; it rebuilds on the next pass.
        // (`share_keys` / `share_sib` are node-keyed and epoch-tagged,
        // so they safely self-overwrite.)
        self.share_cache.borrow_mut().clear();
        self.cand_shares.borrow_mut().clear();
        // The `::marker` style memo is hash-keyed too: same leak
        // rationale as `share_cache` above.
        self.marker_memo.borrow_mut().clear();
        // The `::before` / `::after` style memo is hash-keyed too:
        // same leak rationale as `marker_memo` above.
        self.pseudo_memo.borrow_mut().clear();
        // The cross-anchor `:has()` failure memo depends on element
        // states (see `MatchScratch::has_failed`): the epoch bump
        // retires it with every other memoized cascade result, so
        // failures recorded under the old states can never shadow the
        // new pass.
        self.match_scratch.borrow_mut().clear_has_memo();
    }

    /// The computed style of `node`'s `::marker` pseudo-element, or
    /// `None` when no `::marker` rule matches `node`.
    ///
    /// Declarations come from rules whose rightmost compound targets
    /// `::marker` and whose remaining selector matches `node` (the
    /// originating element), cascaded by (origin/importance,
    /// specificity, source order) exactly like element styles; an
    /// element's inline `style` attribute never applies to its
    /// pseudo-elements. Only the properties CSS Lists 3 allows on
    /// `::marker` participate (see [`MARKER_ATOMS`]); `var()`
    /// references resolve against `node`'s custom-property
    /// environment, and a declaration that is invalid at
    /// computed-value time is dropped.
    ///
    /// Properties the pseudo-element does not set inherit from the
    /// originating element (the list item), so the returned style
    /// starts as a clone of `node`'s own computed style with the
    /// marker winners applied on top.
    ///
    /// The style is memoized and reference-counted (see
    /// [`marker_style_for_shared`](Self::marker_style_for_shared)):
    /// list items matching the same `::marker` rules share one
    /// `Rc<ComputedStyle>`; the owned return here keeps
    /// `MarkerSpec`'s type untouched.
    pub fn marker_style_for(&self, node: NodeId) -> Option<ComputedStyle> {
        self.marker_style_for_shared(node).map(|rc| (*rc).clone())
    }

    /// The memoized `::marker` style of `node`, shared by reference
    /// count: list items matching the same `::marker` rules (with the
    /// same originating computed style) share one
    /// `Rc<ComputedStyle>` instead of each paying the pseudo cascade
    /// plus a base-style clone. `None` when no `::marker` rule
    /// matches `node`.
    ///
    /// The memo key ([`marker_memo_key`]) folds the style-pass epoch,
    /// the originating style's identity, and the resolved `::marker`
    /// winners. The originating style comes from
    /// [`style_for_shared`](Self::style_for_shared), which is itself
    /// cross-element shared for identical siblings, so its pointer
    /// identity implies value identity — and the winners are hashed by
    /// value. Equal keys therefore imply byte-identical marker styles.
    /// `var()` winners resolve per node against that node's own
    /// custom-property environment *before* hashing, so differing
    /// environments cannot collide.
    pub fn marker_style_for_shared(&self, node: NodeId) -> Option<Rc<ComputedStyle>> {
        let epoch = self.epoch.get();
        let base = self.style_for_shared(node);
        let winners = self.pseudo_winners(node, "marker", MARKER_ATOMS)?;
        let key = marker_memo_key(epoch, &base, &winners);
        if let Some(entry) = self.marker_memo.borrow().get(&key)
            && entry.epoch == epoch
        {
            return Some(Rc::clone(&entry.style));
        }
        // `(*base).clone()` hands back an owned style, so the marker
        // winners overwrite its sparse slots in place (see `set_slot`).
        let style = apply_pseudo_winners((*base).clone(), &winners);
        let rc = Rc::new(style);
        self.marker_memo.borrow_mut().insert(
            key,
            MarkerMemoEntry {
                epoch,
                style: Rc::clone(&rc),
            },
        );
        Some(rc)
    }

    /// The computed style of `node`'s `::before` / `::after`
    /// pseudo-element, or `None` when no rule targets it.
    ///
    /// The generalized form of [`marker_style_for`](Self::marker_style_for):
    /// declarations come from rules whose rightmost compound targets
    /// `::pseudo` and whose remaining selector matches `node` (the
    /// originating element), cascaded exactly like element styles; an
    /// element's inline `style` attribute never applies to its
    /// pseudo-elements. Only `allowed_atoms` participate (pass
    /// [`BEFORE_AFTER_ATOMS`] for `::before`/`::after`); `var()`
    /// references resolve against `node`'s custom-property
    /// environment, and a declaration that is invalid at
    /// computed-value time is dropped.
    ///
    /// Unlike `::marker`, the pseudo base style resets `display` to
    /// its initial `inline`: `::before`/`::after` generate inline-level
    /// boxes by default, and the originating element's `display`
    /// (often `block` or `list-item`) must not leak into them. `display`
    /// is in the `::before`/`::after` allowed set, so an explicit
    /// pseudo `display` declaration still overrides the reset. The
    /// non-inherited `content` and `unicode-bidi` reset to their
    /// initials alongside `display`: the base style is the originating
    /// element's computed style, which may carry a specified `content`
    /// (non-inherited, so it does not apply to the pseudo) or
    /// `unicode-bidi`; without the reset, `div { content: "evil" }`
    /// plus `div::before { color: red }` would synthesize a phantom box
    /// from the origin's `content` (spec: the pseudo starts at
    /// `content: normal`, i.e. no box). `::marker` queries route through
    /// the shared marker memo instead.
    ///
    /// The style is memoized and reference-counted (see
    /// [`pseudo_style_for_shared`](Self::pseudo_style_for_shared)):
    /// elements matching the same pseudo rules share one
    /// `Rc<ComputedStyle>`; the owned return here keeps the historical
    /// signature untouched.
    pub fn pseudo_style_for(
        &self,
        node: NodeId,
        pseudo: &str,
        allowed_atoms: &[PropertyAtom],
    ) -> Option<ComputedStyle> {
        if pseudo == "marker" {
            // One computation path for `::marker`: the shared memo.
            // (The marker memo's allowed set is fixed to
            // `MARKER_ATOMS`.)
            return self.marker_style_for_shared(node).map(|rc| (*rc).clone());
        }
        // One computation path for `::before` / `::after`: the shared
        // memo. The owned return keeps the historical signature
        // untouched (see `pseudo_style_for_shared`).
        self.pseudo_style_for_shared(node, pseudo, allowed_atoms)
            .map(|rc| (*rc).clone())
    }

    /// The memoized `::before` / `::after` style of `node`, shared by
    /// reference count: elements whose style-sharing keys are equal
    /// (hence identical pseudo winners — see below) share one
    /// `Rc<ComputedStyle>`. `None` when no rule targets the pseudo.
    ///
    /// The memo is probed by the node's style-sharing key ([`Styler::share_key`],
    /// memoized per node per epoch) *before* winners are computed, so a
    /// hit skips the pseudo cascade entirely — no sheet scan, no winner
    /// sort, no per-node `var()` resolution, no base-style clone. Only
    /// on a miss (or when sharing is disabled and no key exists) does
    /// the winners path run, publishing under both keys so the next
    /// equal-key node hits cheaply.
    ///
    /// Soundness of the cheap key: [`Styler::share_key`] folds every
    /// match-relevant input (tag, attributes, state, sibling/ancestor
    /// context) plus the parent key chain, and returns `None` outright
    /// when any selector uses `:has()` or sibling/structural
    /// pseudo-classes. Equal keys therefore imply identical selector
    /// matching — including for `::pseudo` rules, which match with the
    /// same matcher over the same inputs — and identical
    /// custom-property environments (value-equal for equal keys: the
    /// environment is a pure function of the ancestor key chain, and
    /// [`Styler::share_adopt`] installs the donor's environment for
    /// adopted nodes). [`Styler::pseudo_winners`] is a pure function of
    /// exactly those inputs (candidate matching plus env-resolved
    /// `var()`), so equal share keys imply equal winners; the base
    /// style comes from [`Styler::style_for_shared`], value-identical
    /// for equal keys; and the `display` / `content` / `unicode-bidi`
    /// resets below are constant per pseudo. Equal keys therefore imply
    /// byte-identical pseudo styles. (The `None` outcome — no rule
    /// targets the pseudo — is likewise identical for equal keys; it is
    /// simply not memoized, as before.)
    ///
    /// The stored style is the post-reset style: the `display` /
    /// `content` / `unicode-bidi` resets below are constant per pseudo,
    /// so memoizing after applying them keeps one computation path for
    /// the memoized and unmemoized callers.
    pub fn pseudo_style_for_shared(
        &self,
        node: NodeId,
        pseudo: &str,
        allowed_atoms: &[PropertyAtom],
    ) -> Option<Rc<ComputedStyle>> {
        let epoch = self.epoch.get();
        let share_key = self.share_key(node, epoch);
        if let Some(share_key) = share_key {
            let key = pseudo_share_memo_key(share_key, pseudo);
            // No epoch check on the probe: the whole map is dropped on
            // every epoch bump (see `set_element_state`), so every
            // entry present is current by construction.
            if let Some(style) = self
                .pseudo_memo
                .borrow()
                .get(&key)
                .map(|e| Rc::clone(&e.style))
            {
                return Some(style);
            }
        }
        let style = self.pseudo_style_uncached(node, pseudo, allowed_atoms, epoch)?;
        // Publish under the cheap key too, so the next equal-key node
        // hits before computing winners. First publisher wins: equal
        // keys imply equal styles, so a present entry is already the
        // right one (same rationale as `share_publish`).
        if let Some(share_key) = share_key
            && !self
                .pseudo_memo
                .borrow()
                .contains_key(&pseudo_share_memo_key(share_key, pseudo))
        {
            self.pseudo_memo.borrow_mut().insert(
                pseudo_share_memo_key(share_key, pseudo),
                MarkerMemoEntry {
                    epoch,
                    style: Rc::clone(&style),
                },
            );
        }
        Some(style)
    }

    /// The winners-hash path of [`pseudo_style_for_shared`](Self::pseudo_style_for_shared):
    /// compute `node`'s pseudo style from its resolved winners,
    /// memoizing by ([`pseudo_memo_key`]) so nodes without a sharing
    /// key still share across identical winners.
    fn pseudo_style_uncached(
        &self,
        node: NodeId,
        pseudo: &str,
        allowed_atoms: &[PropertyAtom],
        epoch: u64,
    ) -> Option<Rc<ComputedStyle>> {
        let base = self.style_for_shared(node);
        let winners = self.pseudo_winners(node, pseudo, allowed_atoms)?;
        let key = pseudo_memo_key(epoch, pseudo, &base, &winners);
        // No epoch check on the probe: the whole map is dropped on
        // every epoch bump (see `set_element_state`), so every entry
        // present is current by construction.
        if let Some(entry) = self.pseudo_memo.borrow().get(&key) {
            return Some(Rc::clone(&entry.style));
        }
        // `(*base).clone()` hands back an owned style, so the resets
        // and pseudo winners overwrite its sparse slots in place (see
        // `set_slot`).
        let mut style = (*base).clone();
        style.set_slot(props::DISPLAY, initial_rc(props::DISPLAY));
        // `content` and `unicode-bidi` are non-inherited and do not
        // apply to elements, but the base clone above carries the
        // originating element's specified values: reset them to their
        // initials so they cannot leak into the pseudo style (see the
        // doc comment on [`pseudo_style_for`](Self::pseudo_style_for)).
        style.set_slot(props::CONTENT, initial_rc(props::CONTENT));
        style.set_slot(props::UNICODE_BIDI, initial_rc(props::UNICODE_BIDI));
        let rc = Rc::new(apply_pseudo_winners(style, &winners));
        self.pseudo_memo.borrow_mut().insert(
            key,
            MarkerMemoEntry {
                epoch,
                style: Rc::clone(&rc),
            },
        );
        Some(rc)
    }

    /// Winning declarations for `node`'s `pseudo` pseudo-element:
    /// matching rules' declarations filtered to `allowed_atoms`,
    /// last-wins per property, `var()` resolved against `node`'s
    /// custom-property environment, normalized to `(slot, value)`
    /// pairs sorted by slot. `None` when nothing targets the
    /// pseudo-element (or every winner is dropped), so callers skip
    /// the base-style clone entirely.
    ///
    /// CSS-wide keywords: on inherited properties, `inherit` — and
    /// `unset`, which acts as `inherit` there — mean "take the
    /// originating element's value", which is exactly the base style,
    /// so the winner is dropped. On non-inherited properties
    /// (`content`, `unicode-bidi`, `display`), `inherit` takes the
    /// originating element's computed value (read from the origin's
    /// style, since the pseudo base may have reset the slot) and
    /// `unset` acts as `initial` (the shared initial value).
    /// `initial` resets the slot to the shared initial value. A
    /// declaration invalid at computed-value time (e.g. unresolvable
    /// `var()`) is dropped, as in `compute`. Keywords are checked after
    /// `var()` substitution so a variable resolving to one behaves like
    /// the literal keyword.
    fn pseudo_winners(
        &self,
        node: NodeId,
        pseudo: &str,
        allowed_atoms: &[PropertyAtom],
    ) -> Option<Vec<(u8, Arc<CssValue>)>> {
        if !matches!(self.doc.kind(node), NodeKind::Element(_)) {
            return None;
        }
        let mut cands: Vec<Candidate<'_>> = Vec::new();
        self.collect_pseudo_sheet(
            self.ua,
            &self.rule_indexes[0],
            Origin::Ua,
            0,
            node,
            pseudo,
            &mut cands,
        );
        for (i, sheet) in self.author.iter().enumerate() {
            self.collect_pseudo_sheet(
                sheet,
                &self.rule_indexes[i + 1],
                Origin::Author,
                i + 1,
                node,
                pseudo,
                &mut cands,
            );
        }
        if cands.is_empty() {
            return None;
        }
        // Stable sort: (origin/importance, specificity, source order).
        cands.sort_by(|a, b| {
            (a.rank, a.spec, a.sheet, a.order, a.decl_idx)
                .cmp(&(b.rank, b.spec, b.sheet, b.order, b.decl_idx))
        });
        // Last candidate wins per property (sorted ascending).
        let mut declared: HashMap<PropertyAtom, &CssValue> = HashMap::new();
        for c in &cands {
            let atom = c.property.atom();
            if allowed_atoms.contains(&atom) {
                declared.insert(atom, &*c.value);
            }
        }
        if declared.is_empty() {
            return None;
        }
        let env = self.custom_env(node);
        let mut winners: Vec<(u8, Arc<CssValue>)> = Vec::with_capacity(declared.len());
        for (atom, value) in declared {
            // Invalid at computed-value time (e.g. unresolvable
            // `var()`): the declaration is dropped, as in `compute`.
            let Some(resolved) = self.resolve(node, value, &env) else {
                continue;
            };
            // Every allowed atom is a computed property, so the atom
            // index is the slot index directly.
            debug_assert!(atom.is_computed());
            let idx = atom.index();
            // CSS-wide keywords, checked after substitution so a
            // `var()` resolving to one behaves like the literal
            // keyword.
            if let CssValue::Keyword(k) = &resolved
                && matches!(k.as_str(), "inherit" | "unset")
            {
                let wide = k.as_str();
                if is_inherited(atom.as_str()) {
                    // `inherit` — and `unset`, which acts as `inherit`
                    // on inherited properties — take the originating
                    // element's computed value, which is exactly the
                    // base style the pseudo starts from: drop the
                    // winner.
                    continue;
                }
                if wide == "unset" {
                    // `unset` acts as `initial` on non-inherited
                    // properties: the shared initial `Arc`. Every
                    // pseudo-allowed property has an initial value.
                    let Some(initial) = initial_rc(idx) else {
                        continue;
                    };
                    winners.push((idx, initial));
                } else {
                    // `inherit` on a non-inherited property takes the
                    // originating element's computed value. The pseudo
                    // base style may have reset the slot (e.g.
                    // `content` to `normal`), so read the origin's
                    // value explicitly instead of keeping the reset.
                    let origin = self.style_for_shared(node);
                    let Some(v) = origin.get_by_index(idx) else {
                        continue;
                    };
                    winners.push((idx, Arc::new(v.clone())));
                }
                continue;
            }
            let value = if let CssValue::Keyword(k) = &resolved
                && k.as_str() == "initial"
            {
                // Back to the initial value: the shared initial `Arc`.
                // Every pseudo-allowed property has an initial value.
                initial_rc(idx).expect("pseudo-allowed property needs an initial value")
            } else {
                // CSS Generated Content 3 §2.2: adjacent `<string>`s
                // concatenate (`content: "a" "b"` computes to `"ab"`).
                let resolved = if atom == PropertyAtom::CONTENT {
                    fold_content_value(resolved)
                } else {
                    resolved
                };
                Arc::new(resolved)
            };
            winners.push((idx, value));
        }
        if winners.is_empty() {
            return None;
        }
        // Sorted by slot: deterministic application order and a stable
        // memo-key hash.
        winners.sort_by_key(|(idx, _)| *idx);
        Some(winners)
    }

    /// Whether any stylesheet (UA or author) carries a `::before` /
    /// `::after` rule. Box construction gates its per-element
    /// generated-content query on this so documents without such rules
    /// pay ~zero in that hot loop (mirroring how the marker path gates
    /// on `display: list-item`).
    pub fn has_generated_content_rules(&self) -> bool {
        self.generated_content_rules
    }

    /// Whether any UA or author declaration sets `counter-reset` /
    /// `counter-increment` (see [`Styler::scan_counter_adjustments`]).
    /// Box construction threads counter scopes only then; otherwise
    /// every `counter()` reads as 0 without per-element scope work.
    pub fn has_counter_adjustments(&self) -> bool {
        self.counter_adjustment_rules
    }

    /// Cascade candidates for `node`'s `pseudo` pseudo-element: every
    /// matching `::pseudo` declaration from one origin, before sorting.
    /// Mirrors [`collect_sheet`](Self::collect_sheet) but only keeps
    /// selectors whose rightmost compound targets `pseudo`, matched
    /// against the originating element with the pseudo-element itself
    /// stripped.
    #[allow(clippy::too_many_arguments)]
    fn collect_pseudo_sheet(
        &self,
        sheet: &'a Stylesheet,
        index: &RuleIndex,
        origin: Origin,
        sheet_idx: usize,
        node: NodeId,
        pseudo: &str,
        cands: &mut Vec<Candidate<'a>>,
    ) {
        let data = match self.doc.kind(node) {
            NodeKind::Element(d) => d,
            _ => return,
        };
        let states = self.states.borrow();
        let state_for = |n: NodeId| states.probe_untagged(n).copied().unwrap_or_default();
        let mut refs = self.refs_scratch.borrow_mut();
        index.candidates_into(data, &mut refs);
        let mut i = 0;
        while i < refs.len() {
            let rule_idx = refs[i].rule;
            let mut j = i;
            while j < refs.len() && refs[j].rule == rule_idx {
                j += 1;
            }
            let rule = &sheet.rules()[rule_idx];
            if let Some(mq) = &rule.media
                && !mq.matches(self.viewport.w, self.viewport.h)
            {
                i = j;
                continue;
            }
            // Specificity of the most specific matching `::pseudo`
            // selector (the pseudo-element counts as a type selector,
            // per `specificity`).
            let mut best: Option<(u32, u32, u32)> = None;
            for r in &refs[i..j] {
                let selector = &rule.selectors[r.selector];
                let mut scratch = self.match_scratch.borrow_mut();
                if matches_pseudo_element(
                    selector,
                    pseudo,
                    self.doc,
                    node,
                    &state_for,
                    Some(&self.sib_cache),
                    &mut scratch,
                ) {
                    let s = specificity(selector);
                    best = Some(best.map_or(s, |b| b.max(s)));
                }
            }
            if let Some(spec) = best {
                for (di, decl) in rule.declarations.iter().enumerate() {
                    cands.push(Candidate {
                        rank: rank(origin, decl.important),
                        spec,
                        sheet: sheet_idx,
                        order: rule.order,
                        decl_idx: di,
                        property: Cow::Borrowed(&decl.property),
                        value: Cow::Borrowed(&decl.value),
                    });
                }
            }
            i = j;
        }
    }
}

/// Outcome of computing one property's specified value in
/// [`Styler::compute_prop`]: everything except the ancestor climb.
/// [`Styler::finish_prop`] drives a `PropStep` to a final computed value,
/// climbing ancestors iteratively so a cold `style_for` on a 10k-deep node
/// cannot overflow the call stack (B3).
enum PropStep {
    /// The computed value; no inheritance walk needed. `invalid` is set
    /// when the node's own specified value was invalid at
    /// computed-value time (failed `var()` substitution, grammar
    /// validation, or math evaluation): the slot holds the initial
    /// value, but the logical-to-physical fold must treat the winner
    /// as absent so a valid counterpart declaration still wins. A
    /// `revert` that rolls back to nothing (`unset` semantics) is
    /// valid, not invalid, and keeps its cascade position — hence the
    /// explicit flag rather than a `Done(None)` check at the use site.
    Done {
        value: Option<CssValue>,
        invalid: bool,
    },
    /// Climb to the parent: the parent's computed value is this node's
    /// computed value. `explicit` marks climbs from an explicit `inherit`
    /// (or `unset` on an inherited property): if the whole climb yields
    /// nothing, those fall back to the property's initial value, while a
    /// plain unspecified/inherited climb stays unset (`None`).
    Climb { parent: NodeId, explicit: bool },
}

impl<'a> Styler<'a> {
    /// Normalize a tag name for same-type grouping, mirroring
    /// `same_type` in matching: ASCII-lowercased for HTML (parser output
    /// already is, so this is usually a no-op), exact elsewhere.
    fn share_norm_tag(data: &ElementData) -> String {
        let tag = data.tag_name.as_str();
        if data.namespace == Namespace::Html && tag.bytes().any(|b| b.is_ascii_uppercase()) {
            tag.to_ascii_lowercase()
        } else {
            tag.to_owned()
        }
    }

    /// The match-relevant signature of one element: tag, namespace, and
    /// every attribute (name, namespace, value) in document order, plus
    /// interactive-state bits when dynamic pseudo-classes are in play.
    /// Selector matching is a pure function of these inputs (plus the
    /// ancestor/sibling context the key captures separately), so equal
    /// signatures match identically.
    fn share_subject_sig(&self, data: &ElementData, node: NodeId, flags: ShareFlags) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        h = share_str(h, data.tag_name.as_str());
        h = share_mix(h, share_ns_disc(&data.namespace));
        // Document order: matching looks attributes up by name, so
        // order never affects a match, but hashing in document order
        // keeps the key exact no matter what the DOM carries.
        for attr in &data.attributes {
            h = share_str(h, attr.name.as_str());
            if let Some(ns) = attr.namespace {
                h = share_str(h, ns);
            }
            h = share_str(h, attr.value.as_str());
        }
        if flags.dynamic {
            h = share_mix(h, self.share_state_bits(node));
        }
        h
    }

    /// Interactive-state bits consulted by the dynamic pseudo-classes.
    fn share_state_bits(&self, node: NodeId) -> u64 {
        let state = self
            .states
            .borrow()
            .probe_untagged(node)
            .copied()
            .unwrap_or_default();
        (state.hover as u64)
            | ((state.active as u64) << 1)
            | ((state.focus as u64) << 2)
            | ((state.visited as u64) << 3)
    }

    /// Whether `node` matches `:empty`, mirroring the matcher's arm
    /// exactly (element children defeat it; non-empty text defeats it;
    /// comments and PIs are ignored). The walk is `first_child` /
    /// `next_sibling`, which visits exactly the nodes the matcher's
    /// `children_cursor` would.
    fn share_is_empty(&self, node: NodeId) -> bool {
        let mut cursor = self.doc.first_child(node);
        while let Some(child) = cursor {
            match self.doc.kind(child) {
                NodeKind::Element(_) => return false,
                NodeKind::Text(t) if !t.is_empty() => return false,
                _ => {}
            }
            cursor = self.doc.next_sibling(child);
        }
        true
    }

    /// Memoized per-parent sibling context for sharing keys: positions
    /// for the structural pseudo-classes and running prefix hashes for
    /// the sibling combinators. One child walk per parent per epoch.
    fn share_sib_ctx(&self, parent: NodeId, epoch: u64, flags: ShareFlags) -> Rc<ShareSibCtx> {
        if let Some(hit) = self.share_sib.borrow().get(&parent)
            && hit.epoch == epoch
        {
            return Rc::clone(hit);
        }
        let mut ctx = ShareSibCtx {
            epoch,
            ..ShareSibCtx::default()
        };
        let mut order = Vec::new();
        let mut cursor = self.doc.first_child(parent);
        while let Some(child) = cursor {
            if matches!(self.doc.kind(child), NodeKind::Element(_)) {
                order.push(child);
            }
            cursor = self.doc.next_sibling(child);
        }
        let n = order.len() as u32;
        // Forward pass: 1-based positions and same-type ranks.
        let mut type_seen: FxHashMap<(u64, String), u32> = FxHashMap::default();
        for (i, &child) in order.iter().enumerate() {
            let i = i as u32 + 1;
            ctx.index.insert(child, i);
            ctx.rindex.insert(child, n - i + 1);
            if let NodeKind::Element(d) = self.doc.kind(child) {
                let key = (share_ns_disc(&d.namespace), Self::share_norm_tag(d));
                let t = type_seen.entry(key).or_insert(0);
                *t += 1;
                ctx.type_index.insert(child, *t);
            }
        }
        // Backward pass: same-type ranks from the end.
        let mut type_left = type_seen;
        for &child in order.iter().rev() {
            if let NodeKind::Element(d) = self.doc.kind(child)
                && let Some(left) =
                    type_left.get_mut(&(share_ns_disc(&d.namespace), Self::share_norm_tag(d)))
            {
                ctx.type_rindex.insert(child, *left);
                *left -= 1;
            }
        }
        // Prefix hashes over preceding element siblings. Each step folds
        // the sibling's signature, state, and position plus its own
        // prefix, so nested sibling combinators (`p ~ q ~ span`) stay
        // sound: testing a preceding sibling against such a selector
        // re-examines that sibling's preceding context, which is exactly
        // what its stored prefix captures.
        if flags.sibling {
            let mut prefix = share_mix(0xcbf2_9ce4_8422_2325, 0x7369_626c_696e_6773);
            for &child in &order {
                ctx.prefix.insert(child, prefix);
                let mut m = match self.doc.kind(child) {
                    NodeKind::Element(d) => self.share_subject_sig(d, child, flags),
                    _ => 0,
                };
                if flags.structural {
                    m = share_mix(m, ctx.index.get(&child).copied().unwrap_or(0) as u64);
                    m = share_mix(m, ctx.rindex.get(&child).copied().unwrap_or(0) as u64);
                    m = share_mix(m, ctx.type_index.get(&child).copied().unwrap_or(0) as u64);
                    m = share_mix(m, ctx.type_rindex.get(&child).copied().unwrap_or(0) as u64);
                }
                m = share_mix(m, prefix);
                prefix = share_mix(prefix, m);
            }
        }
        let ctx = Rc::new(ctx);
        self.share_sib.borrow_mut().insert(parent, Rc::clone(&ctx));
        ctx
    }

    /// One node's sharing key: a hash of every match-relevant input.
    /// The parent key folds in recursively, so equal keys imply equal
    /// ancestor chains of match-relevant inputs — and therefore
    /// identical selector matching, identical custom-property
    /// environments, and identical inheritance inputs. Non-elements
    /// match no selectors; their style is pure inheritance, captured by
    /// the parent key plus a kind discriminant.
    fn share_key_one(&self, node: NodeId, parent_key: u64, epoch: u64, flags: ShareFlags) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        match self.doc.kind(node) {
            NodeKind::Element(d) => {
                h = share_mix(h, 1);
                h = share_mix(h, self.share_subject_sig(d, node, flags));
            }
            NodeKind::Document => {
                h = share_mix(h, 2);
                h = share_mix(h, 0);
            }
            NodeKind::DocumentType(_) => {
                h = share_mix(h, 2);
                h = share_mix(h, 1);
            }
            NodeKind::Text(_) => {
                h = share_mix(h, 2);
                h = share_mix(h, 2);
            }
            NodeKind::Comment(_) => {
                h = share_mix(h, 2);
                h = share_mix(h, 3);
            }
            NodeKind::ProcessingInstruction { .. } => {
                h = share_mix(h, 2);
                h = share_mix(h, 4);
            }
            NodeKind::DocumentFragment => {
                h = share_mix(h, 2);
                h = share_mix(h, 5);
            }
        }
        if flags.structural || flags.sibling {
            match self.doc.parent(node) {
                Some(p) => {
                    let ctx = self.share_sib_ctx(p, epoch, flags);
                    if flags.structural {
                        h = share_mix(h, ctx.index.get(&node).copied().unwrap_or(0) as u64);
                        h = share_mix(h, ctx.rindex.get(&node).copied().unwrap_or(0) as u64);
                        h = share_mix(h, ctx.type_index.get(&node).copied().unwrap_or(0) as u64);
                        h = share_mix(h, ctx.type_rindex.get(&node).copied().unwrap_or(0) as u64);
                    }
                    if flags.sibling {
                        h = share_mix(h, ctx.prefix.get(&node).copied().unwrap_or(0));
                    }
                }
                // The root has no siblings: fold explicit zeros so the
                // key shape matches the non-root path.
                None => {
                    if flags.structural {
                        h = share_mix(h, 0);
                        h = share_mix(h, 0);
                        h = share_mix(h, 0);
                        h = share_mix(h, 0);
                    }
                    if flags.sibling {
                        h = share_mix(h, 0);
                    }
                }
            }
        }
        if flags.empty {
            h = share_mix(h, self.share_is_empty(node) as u64);
        }
        // Always chained: inheritance and custom-property environments
        // flow from the parent even when no selector uses a combinator
        // (e.g. `div { color: red }` inherits into children), so equal
        // keys must imply equal parent keys.
        h = share_mix(h, parent_key);
        h
    }

    /// Sharing key for `node`: `None` when sharing is disabled — `:has()`
    /// in any stylesheet (matching can inspect descendants, which no
    /// fixed-size key captures), or any sibling combinator / structural
    /// pseudo-class (`:disabled`/`:enabled` included): those keys fold
    /// the full preceding-sibling prefix chain / position tuples, making
    /// every element's key unique per document (0% hit rate) while still
    /// paying the O(children) context builds and ancestor hashing, i.e.
    /// net-negative. Otherwise walks up to the nearest key-memoized
    /// ancestor and hashes back down iteratively, so deep DOMs cannot
    /// overflow the stack. The walk routes through
    /// [`Styler::style_parent`]: the custom-property environment flows
    /// across the shadow boundary from the host, so the key chain must
    /// fold the host's full context — otherwise two structurally
    /// identical shadow subtrees under hosts with different `--x`
    /// values would share one parent key and the cache could hand a
    /// shadow child the wrong donor env.
    fn share_key(&self, node: NodeId, epoch: u64) -> Option<u64> {
        let flags = self.share_flags;
        if flags.has || flags.sibling || flags.structural {
            return None;
        }
        let mut chain = Vec::new();
        let mut parent_key = None;
        let mut cursor = Some(node);
        while let Some(n) = cursor {
            if let Some(&k) = self.share_keys.borrow().probe(n, epoch) {
                parent_key = Some(k);
                break;
            }
            chain.push(n);
            cursor = self.style_parent(n);
        }
        // Invariant: `chain` is non-empty unless `node` itself probed.
        let mut pk = parent_key.unwrap_or(0x726f_6f74_5f70_6172);
        for n in chain.into_iter().rev() {
            let k = self.share_key_one(n, pk, epoch, flags);
            self.share_keys.borrow_mut().insert(n, epoch, k);
            pk = k;
        }
        Some(pk)
    }

    /// Probe the style-sharing cache: the finished style and custom
    /// environment adopted from an element with an equal sharing key.
    fn share_lookup(&self, node: NodeId, epoch: u64) -> Option<(Rc<ComputedStyle>, EnvRc)> {
        let key = self.share_key(node, epoch)?;
        let cache = self.share_cache.borrow();
        let entry = cache.get(&key)?;
        if entry.epoch != epoch {
            return None;
        }
        #[cfg(test)]
        self.share_hits.set(self.share_hits.get() + 1);
        Some((Rc::clone(&entry.style), Rc::clone(&entry.env)))
    }

    /// Adopt a shared style for `node`: populate this node's memos from
    /// the donor entry so descendants keep the warm root-first path.
    /// The resolved slots must read as fully computed (`parent_ready`
    /// in `compute`, plus direct `resolved_prop` / `font_size_px`
    /// probes); the env memo keeps descendants' `custom_env` from
    /// re-running this node's candidate match.
    fn share_adopt(&self, node: NodeId, epoch: u64, style: &ComputedStyle, env: &EnvRc) {
        {
            let mut resolved = self.resolved.borrow_mut();
            let slots = resolved.slot_or_insert_with(node, epoch, MemoSet::default);
            slots.fill_from_style(style);
        }
        self.env.borrow_mut().insert(node, epoch, Rc::clone(env));
    }

    /// Publish a freshly computed style for future identical elements.
    /// First donor wins: every donor with the same key computes an
    /// equal style, so overwriting would only churn `Rc`s.
    fn share_publish(&self, node: NodeId, epoch: u64, style: &Rc<ComputedStyle>, env: &EnvRc) {
        let Some(key) = self.share_key(node, epoch) else {
            return;
        };
        let mut cache = self.share_cache.borrow_mut();
        let current = cache.get(&key).is_some_and(|e| e.epoch == epoch);
        if !current {
            cache.insert(
                key,
                ShareEntry {
                    epoch,
                    style: Rc::clone(style),
                    env: Rc::clone(env),
                },
            );
        }
    }

    /// Parsed declarations for an inline `style` attribute value, cached
    /// by the raw attribute string: identical inline styles parse once
    /// instead of once per element. Candidates clone the small
    /// declaration values out of the cached vec (no re-tokenizing).
    fn inline_declarations(&self, style: &str) -> Rc<Vec<Declaration>> {
        if let Some(hit) = self.inline_cache.borrow().get(style) {
            return Rc::clone(hit);
        }
        let parsed = Rc::new(parse_declarations(style));
        #[cfg(test)]
        self.inline_parses.set(self.inline_parses.get() + 1);
        self.inline_cache
            .borrow_mut()
            .insert(style.to_owned(), Rc::clone(&parsed));
        parsed
    }

    /// The parent for style-inheritance purposes: the normal DOM parent,
    /// except a shadow-root fragment routes to its host element. The
    /// fragment is unlinked, so a plain `doc.parent` walk ends there and
    /// inherited properties would fall back to initial values; per spec
    /// shadow content inherits from the host.
    fn style_parent(&self, node: NodeId) -> Option<NodeId> {
        let parent = self.doc.parent(node)?;
        if self.doc.is_shadow_root(parent) {
            self.doc.shadow_host_for(parent)
        } else {
            Some(parent)
        }
    }

    fn compute(&self, node: NodeId) -> Rc<ComputedStyle> {
        let epoch = self.epoch.get();
        // Cross-element style sharing: an element whose match-relevant
        // inputs hash equal to an already-styled element's adopts the
        // cached style instead of re-running the cascade. See
        // `share_key` for why equal keys imply equal computed styles.
        if let Some((style, env)) = self.share_lookup(node, epoch) {
            self.share_adopt(node, epoch, &style, &env);
            return style;
        }
        #[cfg(test)]
        self.share_misses.set(self.share_misses.get() + 1);
        // Fetch this node's cascade winners (deferred `var()` shorthands
        // substituted and expanded) and custom-property environment once
        // and thread them through every property: cloning the memo maps
        // per property (80 per node) was the dominant cost of
        // whole-document styling.
        let raw = self.cascade_expanded(node);
        let env = self.custom_env(node);
        // Index-addressed value array: filling one slot per property
        // avoids the old per-property `String` key allocation and hash
        // insert (that insert was ~30% of whole-document styling).
        // Values are `Arc`-shared with the memo: most properties take
        // their initial value, which is one process-wide allocation.
        // The scratch is a stack array (`PROP_COUNT` words, never
        // heap-allocated); the finished `ComputedStyle` keeps only the
        // slots deviating from the initials (see `from_slots`), so the
        // per-element heap cost is proportional to how unusual the
        // element's style is, not to `PROP_COUNT`.
        let mut props: [Option<Arc<CssValue>>; PROP_COUNT] = std::array::from_fn(|_| None);
        let font_idx = font_size_prop_idx();
        // Root-first fast path: when the parent's memo slots are all
        // filled (the parent was styled before this node, as
        // `build_box_tree` and `compute_style` guarantee), inheritance
        // resolves by direct slot indexing. Every `compute_prop` climb
        // targets the direct parent, so on-demand `memo_get` for the
        // climbing property replaces ~80 per-property `memo_set` hash
        // inserts, the per-property `memo_get` climb probes, and the
        // climb frame bookkeeping. Only inherited properties without a
        // specified value ever climb (a small fixed set), so cloning
        // all 160 parent Arcs up front would waste ~200 atomic ops per
        // node on values never read. The batch write below stays the
        // single memoization point for this node.
        //
        // The borrow is released before the loop: `compute_prop` can
        // re-enter the memo through `font_size_px_for` ->
        // `resolved_prop` -> `memo_set`, which needs `borrow_mut`.
        // The parent's slots are only read (never written) during this
        // node's computation — the re-entrant write targets this
        // node's own slots — but the `RefCell` cannot prove that, so
        // each inherited read re-borrows briefly instead of holding
        // the guard across `compute_prop`.
        let parent = self.style_parent(node);
        let parent_ready = parent.is_some_and(|p| {
            let resolved = self.resolved.borrow();
            resolved
                .probe(p, self.epoch.get())
                .is_some_and(|slots| slots.is_complete())
        });
        // Merge join over the sorted winners: both the property loop
        // and `raw` are in index order, so one linear pass replaces 160
        // binary searches.
        let mut witer = raw.iter().peekable();
        // Per-index: whether the node's specified value was invalid at
        // computed-value time (see `PropStep::Done`). The
        // logical-to-physical fold below treats invalid winners as
        // absent so a valid counterpart declaration still wins.
        let mut invalid = [false; 256];
        for (idx, prop) in COMPUTED_PROPS.iter().enumerate() {
            let idx = idx as u8;
            let specified = match witer.peek() {
                Some((i, v, _)) if *i == idx => {
                    let v = v.as_ref();
                    witer.next();
                    Some(v)
                }
                _ => None,
            };
            // Every supported property has a computed value: when the
            // cascade and inheritance yield nothing, the initial value
            // fills in (custom properties stay conditional and are
            // exposed separately below).
            //
            // The old loop did a `memo_get` per property here, but the
            // only slot that can already be filled for this node is
            // `font-size` (via `font_size_px_for` -> `resolved_prop`
            // during an earlier property's math evaluation); every other
            // slot is still `Unknown`, so those lookups were pure
            // overhead. On the fast path the batch write below records
            // every value; off the fast path `finish_prop` still
            // memoizes every frame it touches.
            let cached = if idx == font_idx {
                self.memo_get(node, idx)
            } else {
                None
            };
            let v = match cached {
                Some(hit) => hit.or_else(|| initial_rc(idx)),
                None => match parent_ready {
                    true => {
                        let r = match self.compute_prop(node, prop, specified, &env) {
                            PropStep::Done {
                                value,
                                invalid: inv,
                            } => {
                                invalid[idx as usize] = inv;
                                value.map(Arc::new)
                            }
                            // Every climb targets the direct parent
                            // (`compute_prop`, `apply_css_wide`).
                            // `parent_ready` guarantees the slot is
                            // `Known`, so `memo_get` hits.
                            PropStep::Climb { parent, explicit } => {
                                // `parent_ready` guarantees the slot
                                // is `Known`, so the outer `Option`
                                // hits; the inner one is the value
                                // (possibly unset/`None`).
                                let inherited = self.memo_get(parent, idx).flatten();
                                if inherited.is_none() && explicit {
                                    // Explicit `inherit` with nothing
                                    // above: the initial value (the old
                                    // `parent_value().or_else(initial)`).
                                    initial_rc(idx)
                                } else {
                                    inherited
                                }
                            }
                        };
                        r.or_else(|| initial_rc(idx))
                    }
                    // Cold path (ancestors not styled yet): the generic
                    // iterative climb driver, unchanged.
                    false => {
                        let (r, inv) = self.finish_prop(node, idx, prop, &raw, &env);
                        invalid[idx as usize] = inv;
                        r.or_else(|| initial_rc(idx))
                    }
                },
            };
            props[idx as usize] = v;
        }

        // CSS Logical Properties 1: fold the flow-relative longhands
        // (`margin-inline-start`, `inset-block-end`, `inline-size`, …)
        // into their physical counterparts against this element's own
        // computed `direction`/`writing-mode`, so layout and paint read
        // correct physical values from the usual slots. The logical
        // slots keep their own computed values (per spec, the computed
        // value of a logical property stays flow-relative). Runs on
        // both the warm and cold paths above: `finish_prop` knows
        // nothing about the fold, so it must happen here, before the
        // memo batch write below.
        apply_logical_mapping(&mut props, &raw, &invalid);

        // CSS Overflow 3 §3: impossible `visible`/`clip` cross-axis
        // combinations adjust on the physical slots (see
        // `fixup_overflow_axes`). Runs before the memo batch write so
        // every consumer reads the adjusted values.
        fixup_overflow_axes(&mut props);

        // One memo borrow per node instead of two per property. The
        // entry is retagged with the current epoch: a stale entry from
        // before a state change is reset, never trusted. The sparse
        // batch fill records only the slots deviating from the initial
        // fallback (see [`MemoSet::fill_from_slots`]).
        {
            let epoch = self.epoch.get();
            let mut resolved = self.resolved.borrow_mut();
            let slots = resolved.slot_or_insert_with(node, epoch, MemoSet::default);
            slots.fill_from_slots(&props);
        }

        // Every custom property in scope is queryable too, except
        // guaranteed-invalid ones (they are unset, never literal).
        // Most elements declare no custom properties: skip the map
        // entirely when `env` is empty so the common case allocates
        // nothing; keep it boxed and `None` when it ends up empty so
        // the common style pays one word.
        let custom = if env.is_empty() {
            None
        } else {
            let mut custom = FxHashMap::with_capacity_and_hasher(env.len(), FxBuildHasher);
            for (name, value) in env.iter() {
                if is_guaranteed_invalid(value) {
                    continue;
                }
                if let Some(r) = self.resolve(node, value, &env)
                    && !custom.contains_key(name)
                {
                    custom.insert(name.clone(), r);
                }
            }
            (!custom.is_empty()).then(|| Box::new(custom))
        };
        // The full-width scratch collapses to the sparse delta: only
        // slots deviating from the shared initials allocate.
        let computed = ComputedStyle::from_slots(props, custom);
        // Publish the finished style (and this node's custom-property
        // environment) for future elements with identical
        // match-relevant inputs.
        let shared = Rc::new(computed);
        self.share_publish(node, epoch, &shared, &env);
        shared
    }

    /// Sorted cascade candidates for `node` (memoized): every matching
    /// declaration from every origin, ordered by (origin/importance,
    /// specificity, source order). Values still contain `var()` and
    /// deferred shorthand declarations are unexpanded. The memoized vec
    /// is `Rc`-shared, so repeat lookups clone a pointer, not the
    /// candidate list.
    fn candidates(&self, node: NodeId) -> Rc<Vec<Candidate<'a>>> {
        let epoch = self.epoch.get();
        if let Some(c) = self.cands.borrow().probe(node, epoch).cloned() {
            return c;
        }
        // Cross-node candidate sharing: elements whose style-sharing key
        // matches an already-built candidate vec adopt its `Rc` instead
        // of rebuilding an identical sorted vec. Equal keys imply
        // identical match-relevant inputs (tag, attributes, state,
        // sibling/ancestor context — the same inputs the style-sharing
        // cache trusts for whole computed styles), so the candidate vecs
        // are equal. `None` (sharing disabled: `:has()`, sibling
        // combinators, structural pseudo-classes) skips the key cache
        // exactly like the style-sharing cache does. The per-node `cands`
        // slab is still populated, so node-keyed probe semantics are
        // unchanged: only the allocation is shared.
        let key = self.share_key(node, epoch);
        if let Some(key) = key
            && let Some(c) = self
                .cand_shares
                .borrow()
                .get(&key)
                .filter(|(e, _)| *e == epoch)
                .map(|(_, c)| Rc::clone(c))
        {
            #[cfg(test)]
            self.cand_key_hits.set(self.cand_key_hits.get() + 1);
            self.cands.borrow_mut().insert(node, epoch, Rc::clone(&c));
            return c;
        }
        // The subject element's `id`/`class` values, computed once per
        // element: every candidate selector's rightmost compound reuses
        // them through `matches_cascade` instead of re-scanning the
        // attribute list per compound. `None` for non-elements (which
        // match nothing and return from `collect_sheet` up front).
        let pre = match self.doc.kind(node) {
            NodeKind::Element(d) => Some(SubjectAttrs::for_element(node, d)),
            _ => None,
        };
        let mut cands: Vec<Candidate<'a>> = Vec::new();
        self.collect_sheet(
            self.ua,
            &self.rule_indexes[0],
            Origin::Ua,
            0,
            node,
            pre,
            &mut cands,
        );
        for (i, sheet) in self.author.iter().enumerate() {
            self.collect_sheet(
                sheet,
                &self.rule_indexes[i + 1],
                Origin::Author,
                i + 1,
                node,
                pre,
                &mut cands,
            );
        }
        // HTML presentational hints (the table border/frame/rules
        // mapping, see `table_presentational_hints`, plus the `dir`
        // attribute mapping, see `dir_presentational_hint`): author
        // origin at zero specificity on sheet 0, so they sort ahead of
        // every author sheet — beating UA declarations, losing to any
        // author declaration, and keeping their relative order.
        let mut hints = table_presentational_hints(self.doc, node);
        hints.extend(dir_presentational_hint(self.doc, node));
        for (di, decl) in hints.iter().enumerate() {
            cands.push(Candidate {
                rank: rank(Origin::Author, decl.important),
                spec: (0, 0, 0),
                sheet: 0,
                order: 0,
                decl_idx: di,
                property: Cow::Owned(decl.property.clone()),
                value: Cow::Owned(decl.value.clone()),
            });
        }
        if let Some(style) = self.doc.get_attribute(node, "style") {
            // The parsed declarations are cached by attribute string
            // (`inline_declarations`): identical `style=""` attributes
            // parse once, and candidates clone the small declaration
            // values out of the cached vec instead of re-tokenizing.
            for (di, decl) in self.inline_declarations(style).iter().enumerate() {
                cands.push(Candidate {
                    rank: rank(Origin::Inline, decl.important),
                    spec: (0, 0, 0),
                    sheet: usize::MAX,
                    order: usize::MAX,
                    decl_idx: di,
                    property: Cow::Owned(decl.property.clone()),
                    value: Cow::Owned(decl.value.clone()),
                });
            }
        }
        // Stable sort: (origin/importance, specificity, source order).
        cands.sort_by(|a, b| {
            (a.rank, a.spec, a.sheet, a.order, a.decl_idx)
                .cmp(&(b.rank, b.spec, b.sheet, b.order, b.decl_idx))
        });
        let cands = Rc::new(cands);
        let epoch = self.epoch.get();
        self.cands
            .borrow_mut()
            .insert(node, epoch, Rc::clone(&cands));
        if let Some(key) = key {
            self.cand_shares
                .borrow_mut()
                .insert(key, (epoch, Rc::clone(&cands)));
        }
        cands
    }

    /// Cascade winners for `node`: winning declaration per property, with
    /// `var()` references still unresolved and deferred `var()`-in-
    /// Cascade winners for `node` with deferred `var()`-in-shorthand
    /// declarations substituted and expanded to longhands,
    /// index-addressed by [`COMPUTED_PROPS`] position.
    ///
    /// Deferred declarations arrive from the parser stored as-is under
    /// the shorthand property name (e.g. `border: var(--w) solid red`).
    /// Each one is resolved against the node's custom-property
    /// environment: a `var()` with no fallback and no set property makes
    /// the declaration invalid at computed-value time, so it is dropped;
    /// otherwise the same expansion the parser would have run is applied,
    /// emitting all longhands with omitted components reset to initial
    /// values. Expansion splices the longhands into the sorted candidate
    /// list at the shorthand's own position (inheriting its cascade rank,
    /// specificity, and source order), so shorthand-vs-longhand ordering
    /// is preserved exactly as if the parser had expanded them.
    ///
    /// The vec keeps last-write-wins per index, matching the old
    /// last-insert-wins map over the spliced candidate list. Unknown
    /// properties are dropped: every reader looks up
    /// [`COMPUTED_PROPS`] positions only.
    fn cascade_expanded(&self, node: NodeId) -> NodeWinners<'a> {
        if let Some(m) = self
            .expanded
            .borrow()
            .probe(node, self.epoch.get())
            .cloned()
        {
            return m;
        }
        let env = self.custom_env(node);
        let cands = self.candidates(node);
        // Single pass over the sorted candidates: a deferred shorthand is
        // replaced in place by its expanded longhands (or dropped when
        // unresolvable), exactly as the old remove/splice loop did, but
        // without cloning the candidate vec or memmoving it.
        //
        // Compact: only specified properties are stored. Winners
        // accumulate in a stack array indexed directly by
        // `COMPUTED_PROPS` position (O(1) last-wins per candidate); the
        // final vec is compacted from it in index order, already sorted
        // for binary-search lookup.
        //
        // Only computed-property atoms (`is_computed`) yield indices into
        // `COMPUTED_PROPS`, so indexing the array below is always in
        // bounds.
        const _: () = assert!(COMPUTED_PROPS.len() <= MAX_COMPUTED_PROPS);
        let mut slots: [Option<(Cow<'a, CssValue>, u32)>; MAX_COMPUTED_PROPS] =
            core::array::from_fn(|_| None);
        // Insert or replace the winner for `idx` (last-wins), recording
        // the candidate's cascade sequence so the logical-to-physical
        // fold can order a logical longhand against its physical
        // counterpart.
        let mut set_winner =
            |idx: u8, value: Cow<'a, CssValue>, seq: u32| slots[idx as usize] = Some((value, seq));
        for (seq, cand) in cands.iter().enumerate() {
            let seq = seq as u32;
            if is_shorthand_property(&cand.property) && value_contains_var(&cand.value) {
                // The `font` shorthand resolves with identifier case
                // preserved, so family names agree with the eager token
                // parser (`parse_font_shorthand`); every other shorthand
                // keeps the lowercasing parse its expansion was written
                // for.
                let preserve_case = cand.property.atom() == PropertyAtom::FONT;
                if let Some(substituted) =
                    self.resolve_inner(node, &cand.value, &env, preserve_case)
                {
                    // The `font` expansion parses comma groups itself
                    // (family boundaries); every other shorthand sees the
                    // `parse_value` shape its expansion was written for.
                    let substituted = if cand.property.atom() == PropertyAtom::FONT {
                        substituted
                    } else {
                        normalize_comma_groups(substituted)
                    };
                    for (property, value) in expand_deferred_shorthand(&cand.property, substituted)
                    {
                        let atom = property.atom();
                        if atom.is_computed() {
                            // Expanded longhands share their shorthand's
                            // cascade position: one declaration, one
                            // sequence number.
                            set_winner(atom.index(), Cow::Owned(value), seq);
                        }
                    }
                    // An unclassifiable substituted value expands to
                    // nothing: the declaration is invalid at
                    // computed-value time.
                }
                // Else: dropped (invalid at computed-value time).
                continue;
            }
            let atom = cand.property.atom();
            if atom.is_computed() {
                // `Cow::Borrowed` clones as a pointer copy: no heap clone
                // of the declaration's value.
                set_winner(atom.index(), cand.value.clone(), seq);
            }
        }
        // Compact the slot array to `(idx, value, seq)` triples, already
        // in index order; this is the only pass over the winners.
        let mut winners: Vec<(u8, Cow<'a, CssValue>, u32)> = Vec::with_capacity(slots.len());
        for (idx, slot) in slots.into_iter().enumerate() {
            if let Some((value, seq)) = slot {
                winners.push((idx as u8, value, seq));
            }
        }
        let winners = Rc::new(winners);
        self.expanded
            .borrow_mut()
            .insert(node, self.epoch.get(), Rc::clone(&winners));
        winners
    }

    /// Collect matching declarations from one stylesheet into `cands`.
    ///
    /// The sheet's [`RuleIndex`] pre-filters selectors by their rightmost
    /// compound's tag/id/classes: only selectors that can possibly match
    /// the element are tested with [`matches_cascade`], resolving each
    /// examined element's [`ElementState`] through the per-node state map
    /// (empty = static matching). `pre` is the subject element's
    /// precomputed `id`/`class` values ([`SubjectAttrs`]), computed once
    /// per element by `candidates()`; [`matches_cascade`] threads them to
    /// every candidate selector's rightmost compound. One `@media` check
    /// per rule; specificity is the most specific matching selector's,
    /// exactly as the old test-every-selector loop computed.
    // `pre` is the subject element's precomputed id/class pair (see
    // `candidates()`); the other seven are the pre-existing sheet context.
    #[allow(clippy::too_many_arguments)]
    fn collect_sheet(
        &self,
        sheet: &'a Stylesheet,
        index: &RuleIndex,
        origin: Origin,
        sheet_idx: usize,
        node: NodeId,
        pre: Option<SubjectAttrs<'_>>,
        cands: &mut Vec<Candidate<'a>>,
    ) {
        // Only elements can match selectors; non-elements contribute
        // nothing (the old loop's `matches` returned false for them).
        let data = match self.doc.kind(node) {
            NodeKind::Element(d) => d,
            _ => return,
        };
        let states = self.states.borrow();
        // Each examined element's interactive state is resolved per
        // element, so `.parent:hover .child` tests the *parent's* hover
        // state rather than the child's.
        let state_for = |n: NodeId| states.probe_untagged(n).copied().unwrap_or_default();
        // Sorted + deduplicated by (rule, selector): each rule's refs form
        // one contiguous run. `refs` is the styler's scratch buffer: no
        // allocation per element per sheet.
        let mut refs = self.refs_scratch.borrow_mut();
        index.candidates_into(data, &mut refs);
        let mut i = 0;
        while i < refs.len() {
            let rule_idx = refs[i].rule;
            let mut j = i;
            while j < refs.len() && refs[j].rule == rule_idx {
                j += 1;
            }
            let rule = &sheet.rules()[rule_idx];
            if let Some(mq) = &rule.media
                && !mq.matches(self.viewport.w, self.viewport.h)
            {
                i = j;
                continue;
            }
            // Specificity of the most specific matching selector.
            let mut best: Option<(u32, u32, u32)> = None;
            for r in &refs[i..j] {
                let selector = &rule.selectors[r.selector];
                // The styler's match scratch is reused across every
                // element: the memo set and backtrack stack amortize to
                // zero allocation per element after the first match.
                let mut scratch = self.match_scratch.borrow_mut();
                if matches_cascade(
                    selector,
                    self.doc,
                    node,
                    &state_for,
                    Some(&self.sib_cache),
                    &mut scratch,
                    pre,
                ) {
                    let s = specificity(selector);
                    best = Some(best.map_or(s, |b| b.max(s)));
                }
            }
            if let Some(spec) = best {
                for (di, decl) in rule.declarations.iter().enumerate() {
                    cands.push(Candidate {
                        rank: rank(origin, decl.important),
                        spec,
                        sheet: sheet_idx,
                        order: rule.order,
                        decl_idx: di,
                        property: Cow::Borrowed(&decl.property),
                        value: Cow::Borrowed(&decl.value),
                    });
                }
            }
            i = j;
        }
    }

    /// Custom properties in scope for `node`: ancestors' merged, own
    /// cascade winners on top. Values stay unresolved (`var()` is
    /// substituted lazily with cycle detection).
    ///
    /// CSS-wide keywords on custom properties compute per CSS Custom
    /// Properties L1, never literally: `initial` becomes the
    /// guaranteed-invalid value (a private marker, never exposed);
    /// `inherit` and `unset` take the parent's value (custom properties
    /// are all inherited) or guaranteed-invalid when no ancestor defines
    /// the property.
    ///
    /// Iterative: walks up to the nearest memoized ancestor, then merges
    /// back down. A recursive version would nest one frame per ancestor
    /// and overflow the stack on very deep DOMs.
    ///
    /// The walk routes through [`Styler::style_parent`] so the merged
    /// environment crosses the shadow boundary: a shadow child's
    /// environment picks up the host's custom properties, per the CSS
    /// spec's shadow-tree inheritance. The shadow-root fragment itself
    /// contributes no candidates (non-elements match nothing), so
    /// skipping it is a no-op.
    ///
    /// The returned map is `Rc`-shared: memo hits cost a pointer bump
    /// instead of cloning the whole merged map, and a node that declares
    /// no custom properties reuses its parent's map outright (structural
    /// sharing via [`Rc::make_mut`], which clones only when the node
    /// actually overrides something).
    fn custom_env(&self, node: NodeId) -> EnvRc {
        let epoch = self.epoch.get();
        if let Some(e) = self.env.borrow().probe(node, epoch).cloned() {
            return e;
        }
        let mut chain = Vec::new();
        let mut base: Option<EnvRc> = None;
        let mut cursor = Some(node);
        while let Some(n) = cursor {
            if let Some(e) = self.env.borrow().probe(n, epoch).cloned() {
                base = Some(e);
                break;
            }
            chain.push(n);
            cursor = self.style_parent(n);
        }
        let mut map = base.unwrap_or_default();
        for n in chain.into_iter().rev() {
            // Custom-property winners come straight from the sorted
            // candidate list: ascending rank order with last-wins per
            // name matches the old raw-map sweep, and `revert` needs
            // the same list anyway.
            let cands = self.candidates(n);
            // Resolve this node's winners against the ancestors' map
            // first (`map` still holds only ancestors here), then apply
            // them, so `inherit`/`unset`/`revert` read the parent's
            // value, never this node's own.
            let mut own: Vec<(String, CssValue)> = Vec::new();
            for c in cands.iter() {
                let k: &str = c.property.as_str();
                let v: &CssValue = &c.value;
                if !k.starts_with("--") {
                    continue;
                }
                // Custom properties store verbatim token streams now: a
                // stream that is exactly one CSS-wide keyword keeps its
                // wide meaning (`--x: initial` is guaranteed-invalid,
                // `--x: inherit` reads the parent, `--x: revert` rolls
                // back); any other stream stays verbatim for `var()`
                // substitution.
                let v = match &*custom_property_keyword(v) {
                    CssValue::Keyword(kw) if kw.eq_ignore_ascii_case("initial") => {
                        guaranteed_invalid_value()
                    }
                    CssValue::Keyword(kw)
                        if kw.eq_ignore_ascii_case("inherit")
                            || kw.eq_ignore_ascii_case("unset") =>
                    {
                        map.get(k).cloned().unwrap_or_else(guaranteed_invalid_value)
                    }
                    CssValue::Keyword(kw)
                        if kw.eq_ignore_ascii_case("revert")
                            || kw.eq_ignore_ascii_case("revert-layer") =>
                    {
                        Self::revert_custom_value(&cands, k, &map)
                    }
                    _ => v.clone(),
                };
                own.push((k.to_owned(), v));
            }
            if !own.is_empty() {
                Rc::make_mut(&mut map).extend(own);
            }
            self.env.borrow_mut().insert(n, epoch, Rc::clone(&map));
        }
        map
    }

    /// Lower-origin value for `revert`/`revert-layer` on a custom
    /// property: the lower origin's winning value (stored unresolved,
    /// like every environment entry). Nothing to roll back to behaves
    /// as `unset`: custom properties inherit, so the ancestor's value
    /// wins, else guaranteed-invalid.
    fn revert_custom_value(
        cands: &[Candidate<'a>],
        name: &str,
        ancestors: &FxHashMap<String, CssValue>,
    ) -> CssValue {
        let inherit_or_invalid = || {
            ancestors
                .get(name)
                .cloned()
                .unwrap_or_else(guaranteed_invalid_value)
        };
        match Self::lower_origin_winner(cands, name) {
            Some(lower) => match &*custom_property_keyword(&lower.value) {
                CssValue::Keyword(kw) if kw.eq_ignore_ascii_case("initial") => {
                    guaranteed_invalid_value()
                }
                CssValue::Keyword(kw)
                    if kw.eq_ignore_ascii_case("inherit")
                        || kw.eq_ignore_ascii_case("unset")
                        || kw.eq_ignore_ascii_case("revert")
                        || kw.eq_ignore_ascii_case("revert-layer") =>
                {
                    inherit_or_invalid()
                }
                _ => lower.value.clone().into_owned(),
            },
            None => inherit_or_invalid(),
        }
    }

    /// Substitute `var()` against `env`, memoized per (node, custom
    /// property). Returns `None` when a reference is unresolvable and has
    /// no fallback (or a cycle is detected): the declaration is invalid at
    /// computed-value time.
    ///
    /// Iterative: an explicit work stack replaces the recursion (one frame
    /// per `var()` chain link would nest arbitrarily deep on long custom-
    /// property chains). Cycle detection uses a hash set for O(1) checks.
    /// Each custom property's substituted value is memoized per node, so a
    /// chain of N references resolves in O(N) total instead of O(N^2). A
    /// list-valued custom property substitutes token-wise: it splices flat
    /// into an enclosing list, and re-wraps into a `List` elsewhere.
    fn resolve<'r>(
        &self,
        node: NodeId,
        value: &'r CssValue,
        env: &'r FxHashMap<String, CssValue>,
    ) -> Option<CssValue> {
        self.resolve_inner(node, value, env, false)
    }

    /// [`resolve`](Self::resolve) with identifier-case control. With
    /// `preserve_case`, a custom property's token stream parses with
    /// [`parse_value_preserving_case`], so identifier keywords keep
    /// their authored case. Used only for the deferred `font` shorthand,
    /// whose family names must agree with the eager token parser; every
    /// other property keeps the lowercasing parse (CSS keywords are
    /// matched case-insensitively against the lowercased form, and the
    /// memo key includes the flag so the two parses never share cache
    /// entries).
    fn resolve_inner<'r>(
        &self,
        node: NodeId,
        value: &'r CssValue,
        env: &'r FxHashMap<String, CssValue>,
        preserve_case: bool,
    ) -> Option<CssValue> {
        // The guaranteed-invalid marker is invalid at computed-value
        // time wherever it is substituted (including `var()` inside
        // math via `substitute_math_vars`): fail closed before the
        // work stack can return it literally.
        if is_guaranteed_invalid(value) {
            return None;
        }
        let mut work = vec![ResolveWork::ValueRef(value)];
        // Result stack: every `Value` work item pushes exactly one
        // `ResolveResult`; the combinators pop their inputs and push
        // one combined result.
        let mut results: Vec<ResolveResult> = Vec::new();
        // Custom properties currently being resolved (cycle detection):
        // the styler-wide scratch set, reused across calls instead of
        // allocating a fresh `HashSet<String>` per resolution. Keyed by
        // interned name id, so inserts never clone the name. `resolve`
        // is iterative and never re-enters itself (the work loop below
        // makes no `self.resolve` calls), so holding the borrow for the
        // whole call is safe. Cleared on entry as a fail-safe; every
        // insert is paired with a remove in `PropTail`, so the set is
        // empty again on exit.
        let mut active = self.resolve_active.borrow_mut();
        active.clear();
        while let Some(w) = work.pop() {
            match w {
                ResolveWork::ValueRef(v) => match v {
                    // Composite shapes destructure by reference: no clone
                    // of the entry/`env` value's own structure. Items
                    // re-enter the stack still borrowed; each leaf
                    // materializes exactly once, on the result stack.
                    CssValue::List(items) => {
                        let n = items.len();
                        work.push(ResolveWork::ListTail(n));
                        // Reversed so items resolve in order (LIFO stack).
                        for item in items.iter().rev() {
                            work.push(ResolveWork::ValueRef(item));
                        }
                    }
                    CssValue::Repeat { mode, tracks } => {
                        let n = tracks.len();
                        work.push(ResolveWork::RepeatTail { mode: *mode, n });
                        // Reversed so tracks resolve in order (LIFO stack).
                        for track in tracks.iter().rev() {
                            work.push(ResolveWork::ValueRef(track));
                        }
                    }
                    // A `Var` node needs only its interned name id and its
                    // fallback: interning here avoids cloning the name
                    // `String` per `var()` reference (the owned path below
                    // pushes the same `VarRef` for parser-produced `Var`
                    // nodes, which arrive already owned).
                    CssValue::Var(name, fallback) => {
                        let id = self.var_names.borrow_mut().intern(name);
                        work.push(ResolveWork::VarRef {
                            id,
                            fallback: fallback.as_deref().cloned(),
                        });
                    }
                    CssValue::Tokens(toks) => {
                        let slashed = if preserve_case {
                            split_top_slash_font(toks)
                        } else {
                            split_top_slash(toks, preserve_case)
                        };
                        match slashed {
                            Some((before, after)) => {
                                work.push(ResolveWork::SlashTail);
                                work.push(ResolveWork::Value(CssValue::List(after)));
                                work.push(ResolveWork::Value(CssValue::List(before)));
                            }
                            None => match split_top_comma(toks, preserve_case) {
                                Some(groups) => {
                                    work.push(ResolveWork::CommaTail(groups.len()));
                                    for group in groups.into_iter().rev() {
                                        work.push(ResolveWork::Value(CssValue::List(group)));
                                    }
                                }
                                None => {
                                    let parse = if preserve_case {
                                        parse_value_preserving_case
                                    } else {
                                        parse_value
                                    };
                                    match parse(toks) {
                                        Ok(parsed) => work.push(ResolveWork::Value(parsed)),
                                        Err(_) => results.push(ResolveResult::Single(None)),
                                    }
                                }
                            },
                        }
                    }
                    // A borrowed leaf: the single materialization point.
                    other => results.push(ResolveResult::Single(Some(other.clone()))),
                },
                ResolveWork::Value(v) => match v {
                    CssValue::Var(name, fallback) => {
                        let fallback = fallback.map(|b| *b);
                        // Intern the name: the memo key and the active set
                        // use the `u32` id, so neither clones the name
                        // `String` per lookup (one clone per distinct name,
                        // at first use, inside the interner).
                        let id = self.var_names.borrow_mut().intern(&name);
                        work.push(ResolveWork::VarRef { id, fallback });
                    }
                    CssValue::List(items) => {
                        let n = items.len();
                        work.push(ResolveWork::ListTail(n));
                        // Reversed so items resolve in order (LIFO stack).
                        for item in items.into_iter().rev() {
                            work.push(ResolveWork::Value(item));
                        }
                    }
                    // A `repeat(auto-fill|auto-fit, …)` descriptor: its
                    // tracks resolve exactly like a list's items (a
                    // `var()` in the track list substitutes here), but
                    // `RepeatTail` reassembles the dedicated descriptor
                    // shape instead of a `List`, so the cascade never
                    // confuses it with a bare track listing.
                    CssValue::Repeat { mode, tracks } => {
                        let n = tracks.len();
                        work.push(ResolveWork::RepeatTail { mode, n });
                        // Reversed so tracks resolve in order (LIFO stack).
                        for track in tracks.into_iter().rev() {
                            work.push(ResolveWork::Value(track));
                        }
                    }
                    // A custom property's verbatim token stream, parsed now
                    // that it is being substituted (or read as its own
                    // computed value). Parsing here — not at parse time —
                    // is what keeps the declaration token-preserving: a
                    // top-level `/` becomes the slash marker the deferred
                    // `font`/`border-radius`/`grid-column` / `grid-row`
                    // expansions understand, so e.g.
                    // `font: var(--f)/1.5` still sees its slash after
                    // substitution. Each side of the `/` resolves
                    // separately and the marker is re-attached by
                    // `SlashTail`, so a slash inside the variable still
                    // splits an enclosing list (`font: var(--x) serif`
                    // with `--x: 12px/1.5`) instead of flattening away.
                    // An unparseable stream (two top-level slashes, an
                    // empty side) is invalid at computed-value time.
                    CssValue::Tokens(toks) => {
                        // A `var()` nested inside another function — e.g.
                        // an opaque gradient body the parser kept verbatim
                        // (see `parse_function`'s gradient arm) — never
                        // becomes a `Var` node on re-parse, so substitute
                        // it here at the token level; the re-parse below
                        // then sees the substituted stream and yields a
                        // real `Gradient`. Unresolvable, cyclic, or
                        // hostile references are invalid at
                        // computed-value time. A top-level `var()` is
                        // left alone: it still flows through the `Var`
                        // arm via re-parse, keeping its memo, fallback,
                        // and splice semantics.
                        let toks = if tokens_contain_nested_var(&toks) {
                            let mut budget = MAX_SUBST_TOKENS;
                            match substitute_nested_vars(
                                &toks,
                                env,
                                &self.var_names,
                                &mut active,
                                true,
                                0,
                                &mut budget,
                            ) {
                                Some(substituted) => substituted,
                                None => {
                                    results.push(ResolveResult::Single(None));
                                    continue;
                                }
                            }
                        } else {
                            toks
                        };
                        // The `font` path keeps top-level commas visible
                        // (see `split_top_slash_font`) so the deferred
                        // expansion rejects leading/doubled commas exactly
                        // like the eager token parser; every other
                        // shorthand keeps the historical flat parse.
                        let slashed = if preserve_case {
                            split_top_slash_font(&toks)
                        } else {
                            split_top_slash(&toks, preserve_case)
                        };
                        match slashed {
                            Some((before, after)) => {
                                work.push(ResolveWork::SlashTail);
                                work.push(ResolveWork::Value(CssValue::List(after)));
                                work.push(ResolveWork::Value(CssValue::List(before)));
                            }
                            // No top-level `/`: a top-level comma keeps its
                            // groups distinct through substitution (see
                            // [`split_top_comma`]); otherwise parse as a
                            // plain value.
                            None => match split_top_comma(&toks, preserve_case) {
                                Some(groups) => {
                                    work.push(ResolveWork::CommaTail(groups.len()));
                                    for group in groups.into_iter().rev() {
                                        work.push(ResolveWork::Value(CssValue::List(group)));
                                    }
                                }
                                None => {
                                    let parse = if preserve_case {
                                        parse_value_preserving_case
                                    } else {
                                        parse_value
                                    };
                                    match parse(&toks) {
                                        Ok(parsed) => work.push(ResolveWork::Value(parsed)),
                                        Err(_) => results.push(ResolveResult::Single(None)),
                                    }
                                }
                            },
                        }
                    }
                    other => results.push(ResolveResult::Single(Some(other))),
                },
                ResolveWork::VarRef { id, fallback } => {
                    // Shared tail of `var()` resolution for one reference:
                    // resolve the fallback, or fail when there is none.
                    // Memoized slash/comma substitutions re-enter the
                    // result stack as `Rc` clones (pointer bumps); only
                    // the plain-value case clones its `CssValue` at the
                    // use site — the one unavoidable copy, since the
                    // result stack needs an owned value to embed (the old
                    // code paid the same copy inside the probe instead).
                    if active.contains(&id) {
                        // Cycle: the reference closes a dependency cycle,
                        // so every custom property in the cycle is
                        // guaranteed-invalid at computed-value time
                        // (css-variables-1 §2.3). The inner fallback
                        // must NOT rescue it: a var() whose reference
                        // closes a cycle is invalid at computed-value
                        // time, and its fallback only applies when the
                        // name is not part of a cycle.
                        work.push(ResolveWork::PushNone)
                    } else if let Some(cached) = self.var_memo_probe(node, id, preserve_case) {
                        // Move out of the probed entry: the plain value is
                        // cloned once inside `var_memo_probe`, and moving it
                        // here avoids a second deep clone; the `Rc`
                        // payloads stay single pointer bumps.
                        match cached {
                            MemoVar::Value(Some(v)) => results.push(ResolveResult::Single(Some(v))),
                            MemoVar::Value(None) => match fallback {
                                Some(f) => {
                                    work.push(ResolveWork::FallbackTail);
                                    work.push(ResolveWork::Value(f))
                                }
                                None => work.push(ResolveWork::PushNone),
                            },
                            // A memoized slash marker re-enters as a
                            // marker, never as a flat value.
                            MemoVar::Slash(slashed) => results.push(ResolveResult::Slash(slashed)),
                            // A memoized comma substitution re-enters
                            // with its groups intact.
                            MemoVar::Comma(groups) => results.push(ResolveResult::Comma(groups)),
                        }
                    } else if let Some(v) = {
                        // The environment is keyed by name: translate
                        // the id back for the lookup. The interner
                        // borrow ends before the insert below; the
                        // returned reference borrows `env` (not the
                        // interner), so the value re-enters the stack
                        // as `ValueRef` with no clone.
                        let names = self.var_names.borrow();
                        env.get(names.name(id))
                    } && !is_guaranteed_invalid(v)
                    {
                        active.insert(id);
                        work.push(ResolveWork::PropTail { name: id, fallback });
                        work.push(ResolveWork::ValueRef(v));
                    } else {
                        match fallback {
                            Some(f) => {
                                work.push(ResolveWork::FallbackTail);
                                work.push(ResolveWork::Value(f))
                            }
                            None => work.push(ResolveWork::PushNone),
                        }
                    }
                }
                ResolveWork::PropTail { name, fallback } => {
                    let child = results.pop().unwrap_or(ResolveResult::Single(None));
                    active.remove(&name);
                    // Memoize the custom property's own substituted value
                    // (before fallback): a cyclic/invalid property stays
                    // invalid, and each `var()` site applies its own
                    // fallback. A slash marker is memoized as a marker
                    // (see `MemoVar`) so a later in-list `var()` on the
                    // same node still splices it correctly.
                    match child {
                        ResolveResult::Slash(slashed) => {
                            // The `Rc` moves into the memo entry: no clone
                            // of the group `Vec`s, and the result keeps
                            // the original.
                            self.var_memo_insert(
                                node,
                                name,
                                preserve_case,
                                MemoVar::Slash(Rc::clone(&slashed)),
                            );
                            results.push(ResolveResult::Slash(slashed));
                        }
                        // A comma-separated substitution keeps its groups
                        // (see [`ResolveResult::Comma`]) so a later
                        // `var()` on the same node still splices them
                        // correctly whether standalone or in a list.
                        ResolveResult::Comma(groups) => {
                            if groups.is_some() {
                                self.var_memo_insert(
                                    node,
                                    name,
                                    preserve_case,
                                    MemoVar::Comma(Rc::clone(&groups)),
                                );
                                results.push(ResolveResult::Comma(groups));
                            } else {
                                match fallback {
                                    Some(f) => {
                                        work.push(ResolveWork::FallbackTail);
                                        work.push(ResolveWork::Value(f))
                                    }
                                    None => work.push(ResolveWork::PushNone),
                                }
                            }
                        }
                        other => {
                            let child = other.into_option();
                            // The plain value still needs one clone: the
                            // memo and the result below are two owned
                            // consumers of one owned value. (The
                            // slash/comma payloads above are `Rc`-shared,
                            // so cloning a memo entry is a pointer bump.)
                            self.var_memo_insert(
                                node,
                                name,
                                preserve_case,
                                MemoVar::Value(child.clone()),
                            );
                            match child {
                                // A list-valued substitution splices token-wise
                                // when the `var()` sits inside a list (CSS-correct);
                                // elsewhere it re-wraps into a `List` on the way
                                // out. Parser-built nested lists (e.g. the
                                // `/`-separated `border-radius` groups) never pass
                                // through here, so their nesting is preserved.
                                Some(CssValue::List(items)) => {
                                    results.push(ResolveResult::Splice(Some(items)))
                                }
                                Some(v) => results.push(ResolveResult::Single(Some(v))),
                                None => match fallback {
                                    Some(f) => {
                                        work.push(ResolveWork::FallbackTail);
                                        work.push(ResolveWork::Value(f))
                                    }
                                    None => work.push(ResolveWork::PushNone),
                                },
                            }
                        }
                    }
                }
                ResolveWork::ListTail(n) => {
                    combine_list_tail(&mut results, n, |items, slash| match slash {
                        Some((before, after)) => {
                            CssValue::List(vec![CssValue::List(before), CssValue::List(after)])
                        }
                        None => CssValue::List(items),
                    })
                }
                ResolveWork::RepeatTail { mode, n } => {
                    // Same combination rules as a list (see
                    // `combine_list_tail`), but the result keeps the
                    // dedicated `repeat()` descriptor shape.
                    combine_list_tail(&mut results, n, |items, slash| {
                        let tracks = match slash {
                            Some((before, after)) => {
                                vec![CssValue::List(before), CssValue::List(after)]
                            }
                            None => items,
                        };
                        CssValue::Repeat { mode, tracks }
                    })
                }
                ResolveWork::SlashTail => {
                    // The `/`-separated sides were pushed as two lists
                    // (before first, so it resolves first); pop them in
                    // reverse and re-attach the slash marker.
                    let after = results
                        .pop()
                        .map(ResolveResult::into_option)
                        .unwrap_or(None);
                    let before = results
                        .pop()
                        .map(ResolveResult::into_option)
                        .unwrap_or(None);
                    match (before, after) {
                        (Some(b), Some(a)) => results.push(ResolveResult::Slash(Rc::new(Some((
                            as_list(&b),
                            as_list(&a),
                        ))))),
                        _ => results.push(ResolveResult::Slash(Rc::new(None))),
                    }
                }
                ResolveWork::CommaTail(n) => {
                    // The comma groups were pushed as lists in reverse
                    // (so the first resolves first); pop them and keep
                    // source order.
                    let mut groups: Vec<Vec<CssValue>> = Vec::with_capacity(n);
                    let mut ok = true;
                    for _ in 0..n {
                        match results
                            .pop()
                            .map(ResolveResult::into_option)
                            .unwrap_or(None)
                        {
                            Some(v) => groups.push(as_list(&v).to_vec()),
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    groups.reverse();
                    results.push(ResolveResult::Comma(Rc::new(if ok {
                        Some(groups)
                    } else {
                        None
                    })));
                }
                ResolveWork::PushNone => results.push(ResolveResult::Single(None)),
                ResolveWork::FallbackTail => {
                    match results.pop().unwrap_or(ResolveResult::Single(None)) {
                        ResolveResult::Single(Some(CssValue::List(items))) => {
                            results.push(ResolveResult::Splice(Some(items)))
                        }
                        other => results.push(other),
                    }
                }
            }
        }
        debug_assert_eq!(results.len(), 1, "resolve left stray results");
        results
            .pop()
            .map(ResolveResult::into_option)
            .unwrap_or(None)
    }
}

/// Normalize comma-group nesting from `var()` substitution back to the
/// shape [`parse_value`] produces: single-item groups flatten, multi-item
/// groups stay nested. Substitution preserves every comma group as a
/// nested list (so the deferred `font` expansion sees family boundaries),
/// but every other consumer — longhand validators, the remaining
/// shorthand expansions — was written against the parser's shape, where
/// only multi-word groups nest. The exact two-group slash-marker shape is
/// left alone: it unambiguously marks `/` (see `parse_border_radius`).
/// A [`CssValue::Repeat`] descriptor is never flattened: its track list
/// is a plain `Vec`, not a comma group, so a single-track
/// `repeat(auto-fill, 100px)` stays structurally distinct from the invalid
/// bare `auto-fill 100px` listing through substitution.
fn normalize_comma_groups(value: CssValue) -> CssValue {
    match value {
        CssValue::List(items)
            if !(items.len() == 2 && items.iter().all(|i| matches!(i, CssValue::List(_)))) =>
        {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    // A single-item comma group never nests in parser
                    // output (`parse_value_group` returns the item
                    // directly), so one here is always from substitution.
                    CssValue::List(mut group) if group.len() == 1 => {
                        out.push(group.pop().expect("single-item group"));
                    }
                    // Recurse so comma groups nested inside a substituted
                    // `repeat()` track list (or any other nesting)
                    // normalize too; a `Repeat` keeps its descriptor
                    // shape and only its tracks recurse.
                    other => out.push(normalize_comma_groups(other)),
                }
            }
            CssValue::List(out)
        }
        CssValue::Repeat { mode, tracks } => CssValue::Repeat {
            mode,
            tracks: tracks.into_iter().map(normalize_comma_groups).collect(),
        },
        v => v,
    }
}

/// Fold `scroll-snap-type` to its canonical computed form (CSS Scroll
/// Snap 1): `proximity` is the default strictness, so a pair naming it
/// (`x proximity`, from the parser or a `var()` substitution) folds to
/// the bare axis keyword, and a single-item list folds to the keyword
/// itself. This is the form CSSOM serializes (`x`, not `x proximity`).
/// Pairs with `mandatory` and a lone `none` keep their shape; anything
/// else passes through for the grammar check to reject.
fn fold_scroll_snap_type(value: CssValue) -> CssValue {
    match value {
        CssValue::List(items) if items.len() == 1 => items.into_iter().next().expect("len 1"),
        CssValue::List(mut items)
            if items.len() == 2
                && matches!(&items[1], CssValue::Keyword(k) if k == "proximity")
                && matches!(&items[0], CssValue::Keyword(k) if matches!(k.as_str(), "x" | "y" | "block" | "inline" | "both")) =>
        {
            items.drain(..).next().expect("axis keyword")
        }
        other => other,
    }
}

/// Whether `v` contains a nested [`CssValue::List`] at any depth (inside
/// plain lists, `repeat()` track lists, or `minmax()` sides).
///
/// For `grid-template-columns`/`grid-template-rows` every nested list is
/// a comma group — the parser preserves comma-ness and `var()`
/// substitution nests each comma group — because commas never separate
/// grid tracks (CSS Grid §7.1) and eager integer-`repeat()` expansion is
/// flat. A `/`-separated slash marker cannot occur here: `/` never
/// appears in a track listing (a top-level `/` is a parse error, so the
/// declaration never reaches the cascade).
fn value_contains_nested_list(v: &CssValue) -> bool {
    match v {
        CssValue::List(items) => items
            .iter()
            .any(|i| matches!(i, CssValue::List(_)) || value_contains_nested_list(i)),
        CssValue::Repeat { tracks, .. } => tracks.iter().any(value_contains_nested_list),
        CssValue::MinMax(min, max) => {
            value_contains_nested_list(min) || value_contains_nested_list(max)
        }
        _ => false,
    }
}

/// Fold `grid-auto-flow`'s two-keyword form (`row dense`) into one
/// space-joined [`CssValue::Keyword`], source order preserved. The value
/// parser produces a `List` for two space-separated idents, but layout
/// reads this property through [`ComputedStyle::get_str`], which only
/// resolves a bare `Keyword`; folding keeps `dense` visible to layout's
/// whitespace scan. A list with a non-keyword item (or more than two
/// items) is left alone so the grammar check drops it.
fn fold_grid_auto_flow(value: CssValue) -> CssValue {
    match value {
        CssValue::List(items) if (1..=2).contains(&items.len()) => {
            let mut words = Vec::with_capacity(items.len());
            for item in &items {
                match item {
                    CssValue::Keyword(k) => words.push(k.as_str()),
                    _ => return CssValue::List(items),
                }
            }
            CssValue::Keyword(words.join(" ").into())
        }
        other => other,
    }
}

/// Fold a `box-shadow`/`text-shadow` value into [`CssValue::Shadows`]
/// (CSS Backgrounds 3 §7 / CSS Text Decoration 3 §4.3).
///
/// The token-level parser (`parse_shadow_value` in parse.rs) already
/// produces `Shadows`, but a `var()` substitution — or any other path
/// that bypassed the parser — arrives as nested `List` shapes: after
/// `normalize_comma_groups`, comma-separated shadows from substitution
/// are one nested `List` per group, while a single substituted shadow
/// is a flat component list. `Keyword("none")` (a `var()` substituted
/// to `none`) folds to the empty list. Valid shapes become `Shadows`;
/// anything else passes through untouched for the grammar check to
/// reject. `allow_inset` selects the `box-shadow` grammar (`inset?`
/// and a spread radius allowed) versus `text-shadow` (neither).
fn fold_shadow_value(value: CssValue, allow_inset: bool) -> CssValue {
    match value {
        CssValue::Shadows(_) => value,
        CssValue::Keyword(k) if k == "none" => CssValue::Shadows(Vec::new()),
        CssValue::List(items) => {
            // One group per nested list (comma-separated substitution),
            // or a single flat component list. Mixed shapes are not a
            // shadow list.
            let groups: Vec<&[CssValue]> = if items.iter().all(|i| matches!(i, CssValue::List(_))) {
                items
                    .iter()
                    .map(|i| match i {
                        CssValue::List(group) => group.as_slice(),
                        _ => unreachable!("all items are lists"),
                    })
                    .collect()
            } else if items.iter().all(|i| !matches!(i, CssValue::List(_))) {
                vec![items.as_slice()]
            } else {
                return CssValue::List(items);
            };
            let mut shadows = Vec::with_capacity(groups.len());
            for group in groups {
                match Shadow::from_components(group, allow_inset) {
                    Some(shadow) => shadows.push(shadow),
                    None => return CssValue::List(items),
                }
            }
            CssValue::Shadows(shadows)
        }
        _ => value,
    }
}

/// Fold a `clip-path` value into [`CssValue::ClipPath`] (css-clip-path-1
/// §4).
///
/// The token-level parser (`parse_clip_path_value` in parse.rs) already
/// produces `ClipPath`, but a `var()` substitution — or any other path
/// that bypassed the parser — arrives in generic shapes:
/// `Keyword("none")` (a `var()` substituted to `none`) folds to
/// [`ClipPath::None`]; `Url` (a substituted `url()`) folds to
/// [`ClipPath::Url`]; a `List` ending in a `<geometry-box>` keyword has
/// the box stripped (the reference box is always the border box).
/// Anything else passes through untouched for the grammar check to
/// reject.
fn fold_clip_path_value(value: CssValue) -> CssValue {
    match value {
        CssValue::ClipPath(_) => value,
        CssValue::Keyword(k) if k == "none" => CssValue::ClipPath(ClipPath::None),
        CssValue::Url(u) => CssValue::ClipPath(ClipPath::Url(u)),
        CssValue::List(items) => {
            // `[<basic-shape> || <geometry-box>]`: strip a trailing
            // geometry-box keyword.
            let items: &[CssValue] = match items.as_slice() {
                [rest @ .., CssValue::Keyword(gb)] if is_clip_geometry_box(gb) => rest,
                _ => &items,
            };
            match items {
                [CssValue::ClipPath(cp)] => CssValue::ClipPath(cp.clone()),
                _ => CssValue::List(items.to_vec()),
            }
        }
        _ => value,
    }
}

/// Whether `name` is a `<geometry-box>` keyword (css-clip-path-1 §4).
fn is_clip_geometry_box(name: &str) -> bool {
    matches!(
        name,
        "border-box"
            | "padding-box"
            | "content-box"
            | "margin-box"
            | "fill-box"
            | "stroke-box"
            | "view-box"
    )
}

/// Sizing properties whose grammar takes the `stretch` keyword (CSS
/// Sizing 3 §5) — and therefore the legacy `-webkit-fill-available` /
/// `-moz-available` aliases (css-sizing-4), which normalize to
/// `stretch` here at the sizing validation arms and in the sizing
/// fold below. Chrome rejects the aliases for alignment properties,
/// so the mapping must never live in the generic value parser.
const STRETCH_SIZING_PROPS: &[&str] = &[
    "width",
    "min-width",
    "height",
    "min-height",
    "max-width",
    "max-height",
    "flex-basis",
];

/// Whether `value` is a legacy fill-available alias that normalizes
/// to `stretch` in sizing grammars.
fn is_fill_available_alias(value: &CssValue) -> bool {
    matches!(
        value,
        CssValue::Keyword(k) if matches!(k.as_str(), "-webkit-fill-available" | "-moz-available")
    )
}

/// css-sizing-4 legacy aliases: `-webkit-fill-available` /
/// `-moz-available` compute to the `stretch` keyword on sizing
/// properties only. Applied in `compute_prop` before grammar
/// validation (like `fold_grid_auto_flow`); alignment properties
/// never fold, so the aliases stay invalid there.
fn fold_fill_available(value: CssValue) -> CssValue {
    match value {
        CssValue::Keyword(k)
            if matches!(k.as_str(), "-webkit-fill-available" | "-moz-available") =>
        {
            CssValue::Keyword("stretch".into())
        }
        other => other,
    }
}

/// CSS Generated Content 3 §2.2: `content` takes a `<content-list>`,
/// and adjacent `<string>`s concatenate (`content: "a" "b"` computes
/// to `"ab"`). Fold a whitespace-separated all-string list into one
/// `String` before grammar validation so the validator and
/// `content_text` see the canonical shape; a mixed list (strings
/// beside `attr()` / `counter()` / `counters()`) keeps its items but
/// still concatenates adjacent `<string>` runs, so `"a" "b" attr(x)`
/// computes to `"ab"` followed by the `attr()` reference. Anything
/// else (`url()` / numbers / nested lists) folds to itself and stays
/// invalid.
fn fold_content_value(value: CssValue) -> CssValue {
    match value {
        CssValue::List(items) if !items.is_empty() => {
            if items.iter().all(|i| matches!(i, CssValue::String(_))) {
                let mut out = String::new();
                for item in items {
                    if let CssValue::String(s) = item {
                        out.push_str(&s);
                    }
                }
                CssValue::String(out)
            } else {
                // Mixed `<content-list>`: concatenate adjacent
                // `<string>` runs in place, keeping `attr()` /
                // `counter()` items for element-scoped resolution at
                // box-synthesis time (see
                // `pseudo::resolve_content_text`).
                let mut out: Vec<CssValue> = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        CssValue::String(s) => match out.last_mut() {
                            Some(CssValue::String(prev)) => prev.push_str(&s),
                            _ => out.push(CssValue::String(s)),
                        },
                        other => out.push(other),
                    }
                }
                CssValue::List(out)
            }
        }
        other => other,
    }
}

/// The initial value of a supported property, used by the `initial` and
/// `unset` CSS-wide keywords. Values follow the CSS specifications;
/// `color`'s initial is UA-defined per spec (black is the common default).
/// A custom property's verbatim token stream, parsed just far enough to
/// recognize a bare keyword. Custom properties store raw
/// [`CssValue::Tokens`] now, but the environment builders still need to
/// treat `--x: initial` (etc.) as CSS-wide keywords; a stream that
/// parses to a keyword is returned parsed, anything else is returned
/// unchanged for `var()` substitution.
fn custom_property_keyword(v: &CssValue) -> Cow<'_, CssValue> {
    match v {
        CssValue::Tokens(toks) => match parse_slash_aware_value(toks) {
            Some(kw @ CssValue::Keyword(_)) => Cow::Owned(kw),
            _ => Cow::Borrowed(v),
        },
        _ => Cow::Borrowed(v),
    }
}

/// Marker for the CSS guaranteed-invalid value inside custom-property
/// environments (see [`Styler::custom_env`]). A `Keyword` with a NUL byte,/// which can never arise from parsing (the tokenizer never produces NUL
/// in identifiers/keywords), so it cannot collide with author input. It
/// never escapes the environment: [`Styler::resolve`] maps it to `None`
/// (invalid at computed-value time), and computed-style exposure skips
/// it.
fn guaranteed_invalid_value() -> CssValue {
    CssValue::Keyword("\u{0}guaranteed-invalid".into())
}

/// Whether a custom-property environment value is the
/// guaranteed-invalid marker.
/// Radians of a token stream that is exactly one angle dimension
/// (`90deg`, `100grad`, `1.57rad`, `0.25turn`), ignoring surrounding
/// whitespace. `None` for anything else. Used by the math-`var()`
/// substitution angle pre-pass (see
/// [`Styler::angle_leaf_of_custom_prop`]).
fn single_angle_radians(toks: &[Token]) -> Option<f32> {
    let non_ws: Vec<&Token> = toks
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .collect();
    match non_ws.as_slice() {
        [Token::Dimension { value: n, unit, .. }] => angle_unit(unit).map(|u| u.to_radians(*n)),
        _ => None,
    }
}

/// Split a `var(<name>)` / `var(<name>, <fallback>)` token stream into
/// the property name and the raw fallback tokens (if any). `None` when
/// the stream is not exactly one `var()` reference. Mirrors the
/// name/fallback split in [`substitute_var_ref`]; used by the
/// math-`var()` substitution angle pre-pass.
fn var_ref_parts(toks: &[Token]) -> Option<(&str, Option<&[Token]>)> {
    let [Token::Function(f), rest @ ..] = toks else {
        return None;
    };
    if !f.eq_ignore_ascii_case("var") {
        return None;
    }
    let body = rest.strip_suffix(&[Token::RParen])?;
    // Split the name from the fallback at the first top-level comma.
    let mut comma = None;
    let mut d = 0u32;
    for (i, t) in body.iter().enumerate() {
        match t {
            Token::Function(_) | Token::LParen | Token::LBracket => d += 1,
            Token::RParen | Token::RBracket => d = d.saturating_sub(1),
            Token::Comma if d == 0 => {
                comma = Some(i);
                break;
            }
            _ => {}
        }
    }
    let (name_toks, fallback_toks) = match comma {
        Some(k) => (&body[..k], Some(&body[k + 1..])),
        None => (body, None),
    };
    let name = match name_toks
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .collect::<Vec<_>>()
        .as_slice()
    {
        [Token::Ident(n)] if n.starts_with("--") => n.as_str(),
        _ => return None,
    };
    Some((name, fallback_toks))
}

fn is_guaranteed_invalid(v: &CssValue) -> bool {
    matches!(v, CssValue::Keyword(kw) if kw == "\u{0}guaranteed-invalid")
}

fn initial_value(property: &str) -> Option<CssValue> {
    let kw = |s: &str| CssValue::Keyword(s.into());
    let px0 = CssValue::Length(0.0, LengthUnit::Px);
    let v = match property {
        "color" => CssValue::Color(Color {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        }),
        "font-family" => kw("serif"),
        "font-size" => kw("medium"),
        "font-style" | "font-variant" | "font-weight" => kw("normal"),
        "line-height" => kw("normal"),
        "text-align" => kw("start"),
        "text-indent" => px0,
        "text-transform" => kw("none"),
        "letter-spacing" | "word-spacing" => kw("normal"),
        "white-space" => kw("normal"),
        "visibility" => kw("visible"),
        "cursor" => kw("auto"),
        "direction" => kw("ltr"),
        // CSS Writing Modes 3: initial `normal` (not inherited).
        "unicode-bidi" => kw("normal"),
        // CSS Writing Modes 3: the initial writing mode is horizontal,
        // top-to-bottom.
        "writing-mode" => kw("horizontal-tb"),
        "display" => kw("inline"),
        "position" => kw("static"),
        "top" | "right" | "bottom" | "left" => kw("auto"),
        // CSS Logical Properties 1: flow-relative insets share the
        // physical inset grammar (`<length-percentage> | auto`).
        "inset-inline-start" | "inset-inline-end" | "inset-block-start" | "inset-block-end" => {
            kw("auto")
        }
        "float" | "clear" => kw("none"),
        "width" | "height" | "min-width" | "min-height" => kw("auto"),
        // CSS Logical Properties 1: `inline-size`/`block-size` share the
        // `width`/`height` grammar (`<length-percentage> | auto`).
        "inline-size" | "block-size" => kw("auto"),
        "max-width" | "max-height" => kw("none"),
        "margin-top" | "margin-right" | "margin-bottom" | "margin-left" => px0,
        // CSS Logical Properties 1: flow-relative margins share the
        // physical margin grammar (`<length-percentage> | auto`); they
        // are folded into the physical slots after the cascade (see
        // `apply_logical_mapping`), so their initial values only fill
        // their own slots.
        "margin-inline-start" | "margin-inline-end" | "margin-block-start" | "margin-block-end" => {
            px0
        }
        "padding-top" | "padding-right" | "padding-bottom" | "padding-left" => px0,
        "padding-inline-start"
        | "padding-inline-end"
        | "padding-block-start"
        | "padding-block-end" => px0,
        "border-top-width"
        | "border-right-width"
        | "border-bottom-width"
        | "border-left-width"
        | "border-inline-start-width"
        | "border-inline-end-width"
        | "border-block-start-width"
        | "border-block-end-width" => kw("medium"),
        "border-top-style"
        | "border-right-style"
        | "border-bottom-style"
        | "border-left-style"
        | "border-inline-start-style"
        | "border-inline-end-style"
        | "border-block-start-style"
        | "border-block-end-style" => kw("none"),
        "border-top-color"
        | "border-right-color"
        | "border-bottom-color"
        | "border-left-color"
        | "border-inline-start-color"
        | "border-inline-end-color"
        | "border-block-start-color"
        | "border-block-end-color" => kw("currentcolor"),
        "border-top-left-radius"
        | "border-top-right-radius"
        | "border-bottom-right-radius"
        | "border-bottom-left-radius" => px0,
        "box-sizing" => kw("content-box"),
        // css-sizing-4 §3.1: no preferred ratio by default.
        "aspect-ratio" => kw("auto"),
        "overflow" | "overflow-x" | "overflow-y" | "overflow-block" | "overflow-inline" => {
            kw("visible")
        }
        "opacity" => CssValue::Number(1.0),
        "mix-blend-mode" => kw("normal"),
        "transform" => kw("none"),
        // CSS Will Change 1: no hint by default.
        "will-change" => kw("auto"),
        // Filter Effects 1: no filter by default.
        "filter" => CssValue::Filter(Filter::None),
        "clip-path" => CssValue::ClipPath(ClipPath::None),
        "backdrop-filter" => CssValue::Filter(Filter::None),
        // CSS Backgrounds 3 §7 / CSS Text Decoration 3 §4.3: no shadow
        // by default (`none` computes to the empty layer list).
        "box-shadow" | "text-shadow" => CssValue::Shadows(Vec::new()),
        // CSS Transforms 2: no perspective by default.
        "perspective" => kw("none"),
        // CSS Contain 1: no containment by default.
        "contain" => kw("none"),
        // CSS Transforms 1 §14: initial `50% 50%` (matches
        // `TransformOrigin::initial`).
        "transform-origin" => {
            CssValue::List(vec![CssValue::Percentage(50.0), CssValue::Percentage(50.0)])
        }
        "z-index" => kw("auto"),
        "flex-direction" => kw("row"),
        "flex-wrap" => kw("nowrap"),
        "justify-content" => kw("normal"),
        "align-items" | "align-content" => kw("normal"),
        "align-self" => kw("auto"),
        "flex-grow" | "order" => CssValue::Number(0.0),
        "flex-shrink" => CssValue::Number(1.0),
        "flex-basis" => kw("auto"),
        "background-color" => CssValue::Color(Color {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        }),
        "background-image" => kw("none"),
        "background-attachment" => kw("scroll"),
        "background-repeat" => kw("repeat"),
        "background-position" => {
            CssValue::List(vec![CssValue::Percentage(0.0), CssValue::Percentage(0.0)])
        }
        "text-decoration" => kw("none"),
        "text-decoration-style" => kw("solid"),
        "vertical-align" => kw("baseline"),
        "list-style-type" => kw("disc"),
        "list-style-position" => kw("outside"),
        "list-style-image" => kw("none"),
        // CSS Generated Content 3 §2: `content` is not inherited; the
        // initial `normal` means "no generated box" on the
        // pseudo-elements.
        "content" => kw("normal"),
        // CSS Lists 3 §4: `counter-reset` / `counter-increment` are
        // not inherited; initial `none` (no adjustments).
        "counter-reset" | "counter-increment" => kw("none"),
        "border-collapse" => kw("separate"),
        "border-spacing" => px0,
        "table-layout" => kw("auto"),
        "caption-side" => kw("top"),
        "empty-cells" => kw("show"),
        "grid-template-columns" | "grid-template-rows" => kw("none"),
        "grid-auto-columns" | "grid-auto-rows" => kw("auto"),
        "grid-column" | "grid-row" => kw("auto"),
        "gap" | "grid-gap" | "row-gap" => px0,
        // `column-gap` is shared with multi-column layout, where the
        // initial `normal` computes to 1em (grid/flex treat `normal` as
        // 0; see `bowser_layout`'s gap readers).
        "column-gap" => kw("normal"),
        "justify-items" => kw("normal"),
        // `auto` (the initial value) defers to the container's
        // `justify-items`; layout resolves it per grid item.
        "justify-self" => kw("auto"),
        // The `place-self` shorthand never occupies a computed slot
        // (it expands to `align-self`/`justify-self` at parse time and
        // after `var()` substitution), but every atom-table entry
        // needs an initial value for the sparse-style fallback: `auto`
        // mirrors both longhands' initial.
        "place-self" => kw("auto"),
        // Sparse row-major placement; layout only honors the
        // dense/sparse distinction (column flow places row-major).
        "grid-auto-flow" => kw("row"),
        // CSS Fragmentation 3: break hints start unset (`auto`); the
        // line-box minimums default to 2.
        "break-before" | "break-after" | "break-inside" => kw("auto"),
        "widows" | "orphans" => CssValue::Number(2.0),
        // CSS Multicol 1: both `auto` (a single column); the shorthand
        // expands eagerly at parse time, so its own slot only serves
        // deferred `var()` substitution bookkeeping.
        "column-count" | "column-width" => kw("auto"),
        "columns" => CssValue::List(vec![kw("auto"), kw("auto")]),
        // CSS Multicol 1: a spanner starts unspanned and columns start
        // balanced. `column-rule-width` mirrors border-width's initial
        // (the `medium` keyword, resolved by layout), `column-rule-style`
        // mirrors border-style's (`none`), and `column-rule-color`
        // mirrors border-color's (`currentcolor`). The `column-rule`
        // shorthand slot only serves deferred `var()` substitution
        // bookkeeping (like `columns`).
        "column-span" => kw("none"),
        "column-fill" => kw("balance"),
        "column-rule" => CssValue::List(vec![kw("medium"), kw("none"), kw("currentcolor")]),
        "column-rule-width" => kw("medium"),
        "column-rule-style" => kw("none"),
        "column-rule-color" => kw("currentcolor"),
        // CSS Scroll Snap 1: scroll margins start at zero; scroll padding
        // starts at `auto` (no inset).
        "scroll-margin-top"
        | "scroll-margin-right"
        | "scroll-margin-bottom"
        | "scroll-margin-left"
        | "scroll-margin-inline-start"
        | "scroll-margin-inline-end"
        | "scroll-margin-block-start"
        | "scroll-margin-block-end" => px0,
        "scroll-padding-top"
        | "scroll-padding-right"
        | "scroll-padding-bottom"
        | "scroll-padding-left"
        | "scroll-padding-inline-start"
        | "scroll-padding-inline-end"
        | "scroll-padding-block-start"
        | "scroll-padding-block-end" => kw("auto"),
        // CSS Overscroll Behavior 1: the default lets scrolls chain.
        "overscroll-behavior-x" | "overscroll-behavior-y" => kw("auto"),
        // CSSOM View: programmatic scrolls are instant unless the
        // scroller opts into smooth.
        "scroll-behavior" => kw("auto"),
        // CSS Overflow 4: no reserved gutter space unless the scroller
        // opts into `stable`.
        "scrollbar-gutter" => kw("auto"),
        // CSS Pointer Events 3: touch input may pan both axes unless the
        // scroller opts out.
        "touch-action" => kw("auto"),
        // CSS Scroll Snap 1: no snapping unless the scroller opts in;
        // snap areas align `none` (not snap areas); a fling may pass
        // intermediate snap areas (`normal`).
        "scroll-snap-type" => kw("none"),
        "scroll-snap-align" => kw("none"),
        "scroll-snap-stop" => kw("normal"),
        // CSS Scroll Snap 2 §3.1.2: no initial snap target.
        "scroll-initial-target" => kw("none"),
        // CSS Overflow 3 §3.2: the clip margin starts at zero.
        "overflow-clip-margin" => px0,
        _ => return None,
    };
    Some(v)
}

/// One component of a border-radius longhand: a `<length-percentage>`,
/// with unitless zero accepted as a length.
fn is_radius_component(value: &CssValue) -> bool {
    matches!(
        value,
        CssValue::Length(_, _)
            | CssValue::Percentage(_)
            | CssValue::Math(_)
            | CssValue::MathDeferred { .. }
    ) || matches!(value, CssValue::Number(n) if *n == 0.0)
}

/// Whether a substituted value matches `property`'s grammar. A mismatch
/// makes the declaration invalid at computed-value time (the property is
/// treated as unset). Properties not listed here accept any value.
/// Whether a computed length/percentage/number is strictly negative.
/// Padding and border widths cannot be negative (invalid at
/// computed-value time). Deferred math comes in two forms (see
/// [`CssValue::MathDeferred`]): the affine form carries true
/// coefficients whose sign is genuinely unknown until layout plugs in
/// the base, but the frozen base-0 form (`pct == 0.0`) is a constant —
/// its sign is known here, so `min(-5px, 20%)` (which freezes to
/// `px: -5.0`) is rejected exactly like a literal `-5px` (CSS 2.1
/// §8.4). This also closes the pre-existing affine hole
/// `calc(-5px + 0%)`: spec-correct, since its percentage folded to
/// zero and the value is unconditionally negative.
fn is_negative_length(value: &CssValue) -> bool {
    match value {
        CssValue::Length(n, _) | CssValue::Percentage(n) | CssValue::Number(n) => *n < 0.0,
        CssValue::MathDeferred { pct, px, .. } if *pct == 0.0 => *px < 0.0,
        _ => false,
    }
}

/// Whether a math value is provably non-negative for every non-negative
/// percentage base. CSS Values 4 range checking (`<length [0,∞]>`) rejects
/// a percentage-bearing `calc()` unless its sign is statically known: the
/// base a `%` resolves against is a non-negative length, so
/// `calc(10px + 5%)` can never go negative, while `calc(100% - 50px)` can
/// and is invalid even though the sign is unknowable until layout.
fn math_value_non_negative(value: &CssValue) -> bool {
    match value {
        CssValue::Math(expr) => math_expr_non_negative(expr),
        // Deferred percentage math is affine in the layout base
        // (`value = pct * base + px`, `base >= 0`): non-negative for all
        // bases iff both coefficients are. The frozen base-0 form
        // (`pct == 0.0`) is a constant, so this is just its sign.
        CssValue::MathDeferred { pct, px, .. } => *px >= 0.0 && *pct >= 0.0,
        _ => true,
    }
}

/// Whether a `calc()`/`min()`/`max()`/`clamp()` tree is provably
/// non-negative for every non-negative percentage base. `var()` leaves
/// substitute before evaluation, so an unsubstituted tree defers to the
/// computed-value re-check (B8) on the substituted form; constant
/// (percentage-free) math keeps its existing evaluation path. Otherwise
/// the tree must be affine in the percentage (`value = pct * base + px`,
/// recovered with unit font/viewport scales — every length unit resolves
/// with a positive scale, so the coefficient signs are exact) with both
/// coefficients non-negative.
fn math_expr_non_negative(expr: &MathExpr) -> bool {
    if expr.contains_var() {
        return true;
    }
    if !expr.contains_percentage() {
        return true;
    }
    match expr.affine_coefficients(&MathContext::new(1.0, 1.0, 1.0, 1.0)) {
        Some((pct, px)) => px >= 0.0 && pct >= 0.0,
        // Non-affine percentage math (`min()`/`max()`/`clamp()` over
        // `%`): the sign is unknowable until layout resolves the base.
        None => false,
    }
}

/// Maximum absolute grid line number accepted for `grid-row` /
/// `grid-column` at computed-value time. CSS grid line numbers are
/// unbounded `<integer>`s in the grammar, but no real grid needs more
/// than this, and a hostile `grid-row: 1/1000000` would otherwise force
/// layout to allocate a ~1e12-cell placement grid. Layout clamps to
/// this bound defensively as well.
pub const MAX_GRID_LINE: i32 = 10_000;

/// Whether a raw number is a usable grid line: CSS requires `<integer>`
/// and 0 is never a valid line (grid §8.3).
fn is_grid_line_number(n: f32) -> bool {
    n.fract() == 0.0 && n != 0.0 && n.abs() <= MAX_GRID_LINE as f32
}

/// Whether a raw number is a usable `span <n>`: a positive integer. The
/// span's allocation impact is bounded separately in layout.
fn is_grid_span_number(n: f32) -> bool {
    n.fract() == 0.0 && n >= 1.0
}

/// CSS-wide keywords: valid alone on any property, including shorthands
/// (which expand to longhands carrying the keyword; the cascade
/// intercepts it per longhand before grammar validation).
pub(crate) fn is_css_wide_keyword(value: &CssValue) -> bool {
    matches!(value, CssValue::Keyword(k) if matches!(k.as_str(), "inherit" | "initial" | "unset" | "revert" | "revert-layer"))
}

/// Whether `value` is one `<content-list>` item: a `<string>`, an
/// `attr()` reference, or a `counter()` / `counters()` reference (see
/// the `content` grammar arm of [`is_valid_for_property`]).
fn is_content_item(value: &CssValue) -> bool {
    matches!(
        value,
        CssValue::String(_) | CssValue::Attr { .. } | CssValue::Counter { .. }
    )
}

/// Whether `value` matches the `counter-reset` / `counter-increment`
/// grammar `[ <custom-ident> <integer>? ]+ | none` (CSS Lists 3 §4).
/// The token parser lowercases idents into [`CssValue::Keyword`] and
/// leaves numbers raw, so this checks the pair structure: one or more
/// name-then-optional-integer pairs. A name that is a CSS-wide keyword
/// or `none` is rejected (the cascade intercepts CSS-wide keywords
/// before grammar validation; `none` alone is the no-adjustment form).
fn is_counter_value(value: &CssValue) -> bool {
    match value {
        CssValue::Keyword(k) => {
            // `none` alone is the no-adjustment form; any other single
            // ident is a one-pair `[ <custom-ident> <integer>? ]`.
            k.as_str() == "none" || !is_css_wide_keyword(value)
        }
        CssValue::List(items) => {
            if items.is_empty() {
                return false;
            }
            let mut i = 0;
            while i < items.len() {
                match &items[i] {
                    CssValue::Keyword(k)
                        if !is_css_wide_keyword(&items[i]) && k.as_str() != "none" =>
                    {
                        i += 1;
                    }
                    _ => return false,
                }
                // Optional integer adjustment (a non-integer or
                // non-finite number drops the declaration).
                if let Some(CssValue::Number(n)) = items.get(i) {
                    if n.fract() != 0.0 || !n.is_finite() {
                        return false;
                    }
                    i += 1;
                }
            }
            true
        }
        _ => false,
    }
}

/// Whether `value` is valid for `property`'s grammar. Used by the
/// cascade and by the parse-phase DOM shim's `style` setters (invalid
/// declarations are dropped, per CSSOM).
pub fn is_valid_for_property(property: &str, value: &CssValue) -> bool {
    let is_kw = |value: &CssValue, words: &[&str]| matches!(value, CssValue::Keyword(k) if words.contains(&k.as_str()));
    // Legacy `-webkit-fill-available`/`-moz-available` (css-sizing-4):
    // valid only in sizing grammars (Chrome rejects them for
    // alignment), so the alias→`stretch` mapping lives here at the
    // sizing validation, never in the generic value parser. Normalize
    // before the grammar check so the sizing arms below see `stretch`;
    // alignment arms (and `display`'s whitelist) keep rejecting the
    // raw aliases. Direct callers (the CSSOM shim's `CSS.supports`,
    // tests) get the same sizing grammar the cascade computes.
    let stretched;
    let value = if STRETCH_SIZING_PROPS.contains(&property) && is_fill_available_alias(value) {
        stretched = CssValue::Keyword("stretch".into());
        &stretched
    } else {
        value
    };
    // A math expression is syntactically valid anywhere a length or
    // percentage is: whether it computes to a usable value is decided
    // when `compute_prop` evaluates it (invalid math is invalid at
    // computed-value time and drops the declaration). This includes
    // layout-deferred percentage math (`MathDeferred`), which validates
    // as the length/percentages it stands in for.
    let is_length_or_pct = |value: &CssValue| {
        matches!(
            value,
            CssValue::Length(_, _)
                | CssValue::Percentage(_)
                | CssValue::Math(_)
                | CssValue::MathDeferred { .. }
                // `calc-size()` computes to a length (CSS Values 4 §7);
                // whether its basis is valid for the property is decided
                // when the cascade evaluates it.
                | CssValue::CalcSize { .. }
        )
        // CSS-wide rule: unitless zero is a valid <length>.
        || matches!(value, CssValue::Number(n) if *n == 0.0)
    };
    match property {
        "color"
        | "background-color"
        | "border-top-color"
        | "border-right-color"
        | "border-bottom-color"
        | "border-left-color"
        | "border-inline-start-color"
        | "border-inline-end-color"
        | "border-block-start-color"
        | "border-block-end-color"
        | "column-rule-color" => {
            matches!(value, CssValue::Color(_)) || is_kw(value, &["currentcolor"])
        }
        "margin-top" | "margin-right" | "margin-bottom" | "margin-left" => {
            is_length_or_pct(value) || is_kw(value, &["auto"])
        }
        // CSS Logical Properties 1: flow-relative margins take the same
        // grammar as the physical ones (`<length-percentage> | auto`).
        "margin-inline-start" | "margin-inline-end" | "margin-block-start" | "margin-block-end" => {
            is_length_or_pct(value) || is_kw(value, &["auto"])
        }
        "padding-top" | "padding-right" | "padding-bottom" | "padding-left" => {
            // Negative padding is invalid at computed-value time: a
            // literal negative fails here, a negative math result fails
            // the post-evaluation re-check in `compute_prop`.
            is_length_or_pct(value) && !is_negative_length(value)
        }
        // CSS Logical Properties 1: flow-relative padding takes the same
        // grammar as the physical longhands (non-negative
        // `<length-percentage>`).
        "padding-inline-start"
        | "padding-inline-end"
        | "padding-block-start"
        | "padding-block-end" => is_length_or_pct(value) && !is_negative_length(value),
        // `text-indent` accepts negative lengths (hanging indent), so it
        // is not grouped with padding.
        "text-indent" => is_length_or_pct(value),
        "border-top-width"
        | "border-right-width"
        | "border-bottom-width"
        | "border-left-width"
        | "border-inline-start-width"
        | "border-inline-end-width"
        | "border-block-start-width"
        | "border-block-end-width"
        | "column-rule-width" => {
            // `calc()` evaluates after validation; unitless zero is a
            // valid `<length>` (CSS-wide rule). Percentages are NOT valid
            // border/column-rule widths, so `is_length_or_pct` is not used
            // here, and a percentage-bearing `calc()` (which evaluates to
            // `MathDeferred`) fails the post-evaluation re-check in
            // `compute_prop`. Negative widths are invalid at
            // computed-value time.
            (matches!(value, CssValue::Length(_, _) | CssValue::Math(_))
                || matches!(value, CssValue::Number(n) if *n == 0.0)
                || is_kw(value, &["thin", "medium", "thick"]))
                && !is_negative_length(value)
        }
        "border-top-style"
        | "border-right-style"
        | "border-bottom-style"
        | "border-left-style"
        | "border-inline-start-style"
        | "border-inline-end-style"
        | "border-block-start-style"
        | "border-block-end-style"
        | "column-rule-style" => is_kw(
            value,
            &[
                "none", "hidden", "dotted", "dashed", "solid", "double", "groove", "ridge",
                "inset", "outset",
            ],
        ),
        "border-top-left-radius"
        | "border-top-right-radius"
        | "border-bottom-right-radius"
        | "border-bottom-left-radius" => {
            // `<length-percentage>`, or `[<h> <v>]` from the `/`-separated
            // radius shorthand. Unitless zero is a valid length.
            match value {
                CssValue::List(items) => items.len() == 2 && items.iter().all(is_radius_component),
                _ => is_radius_component(value),
            }
        }
        "display" => is_kw(
            value,
            &[
                "inline",
                "block",
                "inline-block",
                "none",
                "contents",
                "flow-root",
                "flex",
                "inline-flex",
                "grid",
                "inline-grid",
                "table",
                "inline-table",
                "table-row",
                "table-cell",
                "table-row-group",
                "table-header-group",
                "table-footer-group",
                "table-column",
                "table-column-group",
                "table-caption",
                "list-item",
            ],
        ),
        "font-size" => {
            is_length_or_pct(value)
                || is_kw(
                    value,
                    &[
                        "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large",
                        "larger", "smaller",
                    ],
                )
        }
        "font-weight" => {
            is_kw(value, &["normal", "bold", "bolder", "lighter"])
                || matches!(value, CssValue::Number(n) if n.fract() == 0.0 && matches!(*n as i32, 1..=1000))
                // `calc()` evaluates after validation; admit it here.
                || matches!(value, CssValue::Math(_) | CssValue::MathDeferred { .. })
        }
        "font-style" => is_kw(value, &["normal", "italic", "oblique"]),
        "font-variant" => is_kw(value, &["normal", "small-caps"]),
        "font-family" => matches!(
            value,
            CssValue::String(_) | CssValue::Keyword(_) | CssValue::List(_)
        ),
        "line-height" => {
            matches!(value, CssValue::Number(_))
                || is_length_or_pct(value)
                || is_kw(value, &["normal"])
        }
        // CSS Sizing 3 §2: all four of `width`/`min-width` and
        // `height`/`min-height` take `auto | <length-percentage> |
        // min-content | max-content | fit-content(<length-percentage>)`.
        // Only the *bare* `fit-content` keyword is recognized here —
        // the functional form `fit-content(<length-percentage>)` is not
        // parsed yet and is dropped as invalid (pre-existing gap; do
        // not copy this grammar to new properties without parsing the
        // function). CSS Sizing 3 §5 adds `stretch` (fill the
        // containing block) to the same four; layout resolves it at
        // used-value time (`bowser_layout`'s `specified_height`).
        "width" | "min-width" | "height" | "min-height" => {
            is_length_or_pct(value)
                || is_kw(
                    value,
                    &[
                        "auto",
                        "fit-content",
                        "min-content",
                        "max-content",
                        "stretch",
                    ],
                )
        },
        // CSS Logical Properties 1: `inline-size`/`block-size` take the
        // same grammar as `width`/`height` (including the intrinsic
        // keywords on the inline axis); they are folded into the
        // `width`/`height` slots after the cascade (see
        // `apply_logical_mapping`).
        "inline-size" => {
            is_length_or_pct(value)
                || is_kw(value, &["auto", "fit-content", "min-content", "max-content"])
        }
        "block-size" => {
            is_length_or_pct(value)
                || is_kw(value, &["auto", "fit-content", "min-content", "max-content"])
        }
        // CSS Flexbox 1 §9.2: `flex-basis: content | <'width'>`, so it
        // takes the `content` keyword plus the full `width` grammar
        // (including the intrinsic keywords and `stretch`, per CSS
        // Sizing 3 §5; bare `fit-content` keyword only — see the note
        // on `width` above). Layout treats `content` as content-based
        // sizing (see `bowser_layout`'s `spec_len_of`, which maps it to
        // `SpecLen::Auto`).
        "flex-basis" => {
            is_length_or_pct(value)
                || is_kw(
                    value,
                    &[
                        "auto",
                        "content",
                        "fit-content",
                        "min-content",
                        "max-content",
                        "stretch",
                    ],
                )
        }
        // CSS Sizing 3 §2: `max-width`/`max-height` take `none |
        // <length-percentage> | min-content | max-content |
        // fit-content(<length-percentage>)`, plus `stretch` per CSS
        // Sizing 3 §5 (fill the containing block; bare `fit-content`
        // keyword only — see the note on `width` above).
        "max-width" | "max-height" => {
            is_length_or_pct(value)
                || is_kw(
                    value,
                    &["none", "fit-content", "min-content", "max-content", "stretch"],
                )
        }
        "top" | "right" | "bottom" | "left" => is_length_or_pct(value) || is_kw(value, &["auto"]),
        // CSS Logical Properties 1: flow-relative insets take the same
        // grammar as the physical offsets.
        "inset-inline-start" | "inset-inline-end" | "inset-block-start" | "inset-block-end" => {
            is_length_or_pct(value) || is_kw(value, &["auto"])
        }
        // CSS Writing Modes 3: the supported writing modes. `sideways-*`
        // need glyph-orientation support the text engine does not have
        // yet, so they are invalid here rather than silently accepted.
        "writing-mode" => is_kw(value, &["horizontal-tb", "vertical-rl", "vertical-lr"]),
        // `calc()` evaluates after validation; admit it here.
        "opacity" => matches!(
            value,
            CssValue::Number(_)
                | CssValue::Percentage(_)
                | CssValue::Math(_)
                | CssValue::MathDeferred { .. }
        ),
        "mix-blend-mode" => is_kw(
            value,
            &[
                "normal", "multiply", "screen", "overlay", "darken", "lighten",
            ],
        ),
        // CSS Transforms 1 §8: `none | <transform-list>`. `none` parses
        // to a keyword; each function in a (space-separated) list parses
        // to `CssValue::Transform` (see `parse_transform_function`), so a
        // list arrives as `Transform` or `List` of `Transform`s. Unknown
        // functions fail at parse time and never reach this check.
        "transform" => {
            is_kw(value, &["none"])
                || matches!(value, CssValue::Transform(_))
                || matches!(value, CssValue::List(items) if !items.is_empty() && items.iter().all(|i| matches!(i, CssValue::Transform(_))))
        }
        // CSS Will Change 1 §3: `auto | <custom-ident>#`. `auto` stands
        // alone; anything else is a non-empty comma-separated ident
        // list (the value parser lowercases idents and splits top-level
        // commas into a `List`). `none` is not a valid value. Any ident
        // is accepted here (property names, custom idents); the
        // containing-block helper only inspects stacking-context
        // triggers.
        "will-change" => match value {
            // `auto` or any ident (`none` is not a valid value).
            CssValue::Keyword(k) => k != "none",
            CssValue::List(items) => {
                !items.is_empty()
                    && items
                        .iter()
                        .all(|i| matches!(i, CssValue::Keyword(k) if k != "auto"))
            }
            _ => false,
        },
        // Filter Effects 1 §8: `none | <filter-value-list>`. The parser
        // produces `Filter` for function lists; a `var()` substituted to
        // `none` arrives as a keyword.
        "filter" => matches!(value, CssValue::Filter(_)) || is_kw(value, &["none"]),
        // Filter Effects 2 §2: `none | <filter-value-list>`, the same
        // grammar as `filter` (shared `Filter` value and parser).
        "backdrop-filter" => matches!(value, CssValue::Filter(_)) || is_kw(value, &["none"]),
        // css-clip-path-1 §4: `none | <basic-shape> | url()`. The
        // token-level parser and `fold_clip_path_value` (which runs
        // before this check on both validation passes) produce
        // `ClipPath`; a `var()` substituted to `none` folds to
        // `ClipPath::None`.
        "clip-path" => matches!(value, CssValue::ClipPath(_)) || is_kw(value, &["none"]),
        // CSS Backgrounds 3 §7 / CSS Text Decoration 3 §4.3:
        // `none | <shadow>#`. The token-level parser and
        // `fold_shadow_value` (which runs before this check on both
        // validation passes) produce `Shadows`; a `var()` substituted to
        // `none` folds to the empty list. Every layer satisfies
        // `Shadow::components_valid` (lengths, non-negative blur, at most
        // one color, `inset`/spread only on `box-shadow`).
        "box-shadow" | "text-shadow" => {
            matches!(value, CssValue::Shadows(shadows) if shadows.iter().all(Shadow::components_valid))
        }
        // CSS Transforms 2: `none | <length>`. Negative lengths are
        // invalid at computed-value time; unitless zero is a valid
        // `<length>` (CSS-wide rule). Percentages are not `<length>`.
        "perspective" => {
            is_kw(value, &["none"])
                || ((matches!(
                    value,
                    CssValue::Length(_, _) | CssValue::Math(_) | CssValue::MathDeferred { .. }
                ) || matches!(value, CssValue::Number(n) if *n == 0.0))
                    && !is_negative_length(value))
        }
        // CSS Contain 1 §2.1 / CSS Contain 2 §2: `none | strict |
        // content | [ size || layout || style || paint ]`. `none`,
        // `strict`, and `content` stand alone; the four containment
        // types combine order-independently with no duplicates.
        "contain" => match value {
            CssValue::Keyword(k) => {
                matches!(k.as_str(), "none" | "strict" | "content")
                    || matches!(k.as_str(), "size" | "layout" | "style" | "paint")
            }
            CssValue::List(items) => {
                if items.is_empty() || items.len() > 4 {
                    return false;
                }
                let mut seen: u8 = 0;
                items.iter().all(|i| match i {
                    CssValue::Keyword(k) => {
                        let bit = match k.as_str() {
                            "size" => 1,
                            "layout" => 2,
                            "style" => 4,
                            "paint" => 8,
                            _ => return false,
                        };
                        if seen & bit != 0 {
                            return false;
                        }
                        seen |= bit;
                        true
                    }
                    _ => false,
                })
            }
            _ => false,
        },
        // CSS Transforms 1 §14: a 2D `<position>` — one or two of the
        // keywords, lengths, or percentages (math validates as the
        // length/percentage it stands in for; see
        // `TransformOrigin::parse`).
        "transform-origin" => TransformOrigin::parse(value).is_some(),
        "z-index" | "order" | "flex-grow" | "flex-shrink" => {
            matches!(
                value,
                CssValue::Number(_) | CssValue::Math(_) | CssValue::MathDeferred { .. }
            ) || is_kw(value, &["auto"])
        }
        "grid-column" | "grid-row" => {
            // `<grid-line> [ / <grid-line> ]?` where `<grid-line>` is
            // `auto`, `<integer>` (nonzero, bounded), or `span <integer>`.
            // The `/`-separated form is stored as
            // `List([List(before), List(after)])`; a bare side may itself
            // be a one-element list from the value parser. Non-integer,
            // zero, or absurd line numbers are invalid at computed-value
            // time: the declaration is dropped.
            fn valid_side(v: &CssValue) -> bool {
                match v {
                    CssValue::Keyword(k) => k.eq_ignore_ascii_case("auto"),
                    CssValue::Number(n) => is_grid_line_number(*n),
                    CssValue::List(items) => match items.as_slice() {
                        [CssValue::Keyword(k), CssValue::Number(n)]
                            if k.eq_ignore_ascii_case("span") =>
                        {
                            is_grid_span_number(*n)
                        }
                        [single] => valid_side(single),
                        _ => false,
                    },
                    _ => false,
                }
            }
            match value {
                CssValue::List(items)
                    if items.len() == 2 && items.iter().all(|v| matches!(v, CssValue::List(_))) =>
                {
                    items.iter().all(valid_side)
                }
                single => valid_side(single),
            }
        }
        "grid-auto-flow" => {
            // `row | column | dense | <row|column> dense`, order-
            // independent (`dense row`, `dense column` also accepted) and
            // case-insensitive. The two-keyword form arrives folded into
            // one space-joined keyword by `compute_prop` (layout reads the
            // property through `get_str`, which only resolves a bare
            // `Keyword`), so validate the whitespace-separated words.
            fn is_axis(w: &str) -> bool {
                w.eq_ignore_ascii_case("row") || w.eq_ignore_ascii_case("column")
            }
            match value {
                CssValue::Keyword(k) => {
                    let mut words = k.split_whitespace();
                    match (words.next(), words.next(), words.next()) {
                        (Some(a), None, None) => is_axis(a) || a.eq_ignore_ascii_case("dense"),
                        (Some(a), Some(b), None) => {
                            (a.eq_ignore_ascii_case("dense") && is_axis(b))
                                || (b.eq_ignore_ascii_case("dense") && is_axis(a))
                        }
                        _ => false,
                    }
                }
                _ => false,
            }
        }
        "grid-template-columns" | "grid-template-rows" => {
            // `<track-list>` at computed-value time: `none` (the
            // initial), `subgrid` (resolved by layout), a single track
            // size, or a list of track sizes, `minmax()` bounds, and
            // `repeat(auto-fill|auto-fit, …)` descriptors.
            //
            // A bare leading `auto-fill`/`auto-fit` keyword (the invalid
            // `auto-fill 100px` listing) is not a track size, so it fails
            // here with no ordering-dependent pre-check: the valid
            // `repeat(auto-fill|auto-fit, …)` arrives as the dedicated
            // `Repeat` variant, which `normalize_comma_groups` never
            // flattens, so the two shapes can never be confused.
            /// Whether `v` contains a `repeat()` descriptor at any depth:
            /// per CSS Grid §7.2.3.2 an auto-repeat may not nest inside
            /// another `repeat()`. The parser rejects eager nesting
            /// (`contains_auto_repeat` in value.rs), but `var()`
            /// substitution can smuggle one in after parsing
            /// (`--x: repeat(auto-fill, 100px)` into
            /// `repeat(auto-fill, var(--x))`), so the validator re-checks
            /// here; a nested descriptor invalidates the whole listing.
            fn deep_contains_repeat(v: &CssValue) -> bool {
                match v {
                    CssValue::Repeat { .. } => true,
                    CssValue::List(items) => items.iter().any(deep_contains_repeat),
                    CssValue::MinMax(min, max) => {
                        deep_contains_repeat(min) || deep_contains_repeat(max)
                    }
                    _ => false,
                }
            }
            fn is_track(v: &CssValue) -> bool {
                match v {
                    CssValue::Length(_, _)
                    | CssValue::Percentage(_)
                    | CssValue::Fr(_)
                    | CssValue::Math(_)
                    | CssValue::MathDeferred { .. }
                    | CssValue::MinMax(_, _) => true,
                    // `repeat(auto-fill|auto-fit, …)`: the tracks
                    // validate like any other track listing, but a
                    // `repeat()` descriptor nested anywhere inside is
                    // invalid (see `deep_contains_repeat`).
                    CssValue::Repeat { tracks, .. } => {
                        !tracks.iter().any(deep_contains_repeat)
                            && tracks.iter().all(is_track)
                    }
                    // CSS-wide rule: unitless zero is a valid `<length>`.
                    CssValue::Number(n) => *n == 0.0,
                    CssValue::Keyword(k) => {
                        k.eq_ignore_ascii_case("auto")
                            || k.eq_ignore_ascii_case("min-content")
                            || k.eq_ignore_ascii_case("max-content")
                    }
                    // A nested list is always a comma group — from the
                    // parser (which preserves comma-ness) or `var()`
                    // substitution — and commas never separate grid
                    // tracks (CSS Grid §7.1), so the listing is invalid.
                    // (Eager integer `repeat()` expansion is flat, so it
                    // never produces a nested list.)
                    CssValue::List(_) => false,
                    _ => false,
                }
            }
            match value {
                CssValue::Keyword(k) => {
                    k.eq_ignore_ascii_case("none")
                        || k.eq_ignore_ascii_case("subgrid")
                        || k.eq_ignore_ascii_case("auto")
                        || k.eq_ignore_ascii_case("min-content")
                        || k.eq_ignore_ascii_case("max-content")
                }
                CssValue::List(items) => items.iter().all(is_track),
                // A single track size parses bare, not wrapped in a
                // list: `100px`, `1fr`, `minmax(…)`, `calc(…)`.
                // `repeat(auto-fill|auto-fit, …)` parses to the
                // dedicated `Repeat` variant, handled by `is_track`.
                single => is_track(single),
            }
        }
        "position" => is_kw(
            value,
            &["static", "relative", "absolute", "fixed", "sticky"],
        ),
        "float" => is_kw(value, &["left", "right", "none"]),
        "clear" => is_kw(value, &["none", "left", "right", "both"]),
        "overflow" => {
            // The shorthand takes 1-2 keywords (`<overflow-x>
            // <overflow-y>?`); a single keyword is also valid.
            // CSS Overflow 3 adds `clip`: clips without a scroll container.
            let words = ["visible", "hidden", "scroll", "auto", "clip"];
            is_kw(value, &words)
                || matches!(value, CssValue::List(items) if (1..=2).contains(&items.len()) && items.iter().all(|i| matches!(i, CssValue::Keyword(k) if words.contains(&k.as_str()))))
        }
        // CSS Overflow 3: the logical `overflow-block`/`overflow-inline`
        // longhands take the same grammar as the physical axes (they fold
        // onto `overflow-x`/`overflow-y` in `apply_logical_mapping`).
        "overflow-x" | "overflow-y" | "overflow-block" | "overflow-inline" => {
            is_kw(value, &["visible", "hidden", "scroll", "auto", "clip"])
        }
        "overscroll-behavior-x" | "overscroll-behavior-y" => {
            is_kw(value, &["auto", "contain", "none"])
        }
        // CSSOM View `scroll-behavior`: `auto | smooth`.
        "scroll-behavior" => is_kw(value, &["auto", "smooth"]),
        // CSS Scroll Snap 1 `scroll-snap-type`: `none` alone, or one axis
        // keyword (`x | y | block | inline | both`) optionally followed by
        // `mandatory | proximity` (a lone axis means `proximity`).
        "scroll-snap-type" => {
            let axes = ["x", "y", "block", "inline", "both"];
            match value {
                CssValue::Keyword(_) => is_kw(value, &["none"]) || is_kw(value, &axes),
                CssValue::List(items) => {
                    (1..=2).contains(&items.len())
                        && matches!(&items[0], CssValue::Keyword(k) if axes.contains(&k.as_str()))
                        && (items.len() == 1
                            || matches!(&items[1], CssValue::Keyword(k) if ["mandatory", "proximity"].contains(&k.as_str())))
                }
                _ => false,
            }
        }
        // CSS Scroll Snap 1 `scroll-snap-align`:
        // `[ none | start | end | center ]{1,2}` — `none` is a legal
        // list item, not just a standalone value: `none start` leaves
        // the block axis non-snapping while the inline axis snaps (a
        // single value repeats on both axes).
        "scroll-snap-align" => match value {
            CssValue::Keyword(_) => is_kw(value, &["none", "start", "end", "center"]),
            CssValue::List(items) => {
                (1..=2).contains(&items.len())
                    && items.iter().all(|i| {
                        matches!(i, CssValue::Keyword(k) if ["none", "start", "end", "center"].contains(&k.as_str()))
                    })
            }
            _ => false,
        },
        // CSS Scroll Snap 1 `scroll-snap-stop`: `normal | always`.
        "scroll-snap-stop" => is_kw(value, &["normal", "always"]),
        // `scroll-initial-target`: `none | nearest` (CSS Scroll Snap 2).
        "scroll-initial-target" => is_kw(value, &["none", "nearest"]),
        // `scrollbar-gutter`: `auto | stable || both-edges` (CSS
        // Overflow 4). `both-edges` requires `stable`.
        "scrollbar-gutter" => match value {
            CssValue::Keyword(k) => matches!(k.as_str(), "auto" | "stable"),
            CssValue::List(items) => {
                items.len() == 2
                    && items.iter().any(|i| matches!(i, CssValue::Keyword(k) if k == "stable"))
                    && items
                        .iter()
                        .any(|i| matches!(i, CssValue::Keyword(k) if k == "both-edges"))
            }
            _ => false,
        },
        // `touch-action`: `auto | none | manipulation | [ pan-x |
        // pan-left | pan-right ]? [ pan-y | pan-up | pan-down ]?` (CSS
        // Pointer Events 3 §5). Single keywords pass through; two-keyword
        // values pick at most one per axis group (`pan-x pan-y`,
        // `pan-left pan-up`, ...). `auto`, `none` and `manipulation`
        // never combine.
        "touch-action" => match value {
            CssValue::Keyword(k) => matches!(
                k.as_str(),
                "auto"
                    | "none"
                    | "pan-x"
                    | "pan-y"
                    | "pan-left"
                    | "pan-right"
                    | "pan-up"
                    | "pan-down"
                    | "manipulation"
            ),
            CssValue::List(items) => {
                let axis_of = |s: &str| match s {
                    "pan-x" | "pan-left" | "pan-right" => 0,
                    "pan-y" | "pan-up" | "pan-down" => 1,
                    _ => -1,
                };
                let mut axes: u8 = 0;
                for item in items {
                    let CssValue::Keyword(k) = item else {
                        return false;
                    };
                    let axis = axis_of(k.as_str());
                    if axis < 0 {
                        return false;
                    }
                    let bit = 1u8 << axis;
                    if axes & bit != 0 {
                        // Two keywords from the same axis group
                        // (`pan-x pan-left`): invalid.
                        return false;
                    }
                    axes |= bit;
                }
                // Exactly one keyword per axis group: `pan-x pan-y`,
                // `pan-left pan-down`, either order.
                axes == 0b11
            }
            _ => false,
        },
        // `overflow-clip-margin`: `<visual-box> || <length>` (CSS
        // Overflow 3). One or two components, in either order.
        "overflow-clip-margin" => {
            let is_box = |v: &CssValue| {
                matches!(v, CssValue::Keyword(k) if matches!(k.as_str(), "content-box" | "padding-box" | "border-box"))
            };
            let is_len = |v: &CssValue| {
                // `<length [0,∞]>` (CSS Overflow 3 §3.2): negative
                // lengths are invalid at computed-value time; a literal
                // negative fails here, a negative `calc()` result fails
                // the post-evaluation re-check in `compute_prop`. A
                // percentage-bearing `calc()` is invalid unless provably
                // non-negative for every non-negative base (CSS Values 4
                // range checking): `calc(100% - 50px)` can resolve
                // negative, so it is rejected, while `calc(10px + 5%)`
                // cannot. The deferred post-evaluation form
                // (`MathDeferred`) carries the same check — it is the
                // `<length-percentage>` the grammar stands in for.
                (matches!(v, CssValue::Length(_, _) | CssValue::Math(_))
                    || matches!(v, CssValue::Number(n) if *n == 0.0)
                    || matches!(v, CssValue::MathDeferred { .. }))
                    && !is_negative_length(v)
                    && math_value_non_negative(v)
            };
            match value {
                CssValue::Keyword(_) => is_box(value) || is_len(value),
                CssValue::List(items) => {
                    items.len() == 2
                        && ((is_box(&items[0]) && is_len(&items[1]))
                            || (is_len(&items[0]) && is_box(&items[1])))
                }
                _ => is_len(value),
            }
        }
        "scroll-margin-top"
        | "scroll-margin-right"
        | "scroll-margin-bottom"
        | "scroll-margin-left"
        | "scroll-margin-block-start"
        | "scroll-margin-block-end"
        | "scroll-margin-inline-start"
        | "scroll-margin-inline-end" => {
            // `<length>`: negative lengths are valid scroll margins,
            // percentages and `auto` are not (CSS Scroll Snap 1 §
            // scroll-margin); unitless zero is a valid `<length>`
            // (CSS-wide rule). `calc()` evaluates after validation; a
            // percentage-bearing `calc()` (which evaluates to
            // `MathDeferred`) fails here since percentages are invalid.
            matches!(value, CssValue::Length(_, _) | CssValue::Math(_))
                || matches!(value, CssValue::Number(n) if *n == 0.0)
        }
        "scroll-margin-block" | "scroll-margin-inline" => {
            // `<length>{1,2}`: the 1- and 2-value logical shorthands.
            matches!(value, CssValue::List(items) if (1..=2).contains(&items.len())
                && items.iter().all(|v| matches!(v, CssValue::Length(_, _) | CssValue::Math(_))
                    || matches!(v, CssValue::Number(n) if *n == 0.0)))
                || matches!(value, CssValue::Length(_, _) | CssValue::Math(_))
                || matches!(value, CssValue::Number(n) if *n == 0.0)
        }
        "scroll-margin" => {
            // `<length>{1,4}`: the physical shorthand.
            matches!(value, CssValue::List(items) if (1..=4).contains(&items.len())
                && items.iter().all(|v| matches!(v, CssValue::Length(_, _) | CssValue::Math(_))
                    || matches!(v, CssValue::Number(n) if *n == 0.0)))
                || matches!(value, CssValue::Length(_, _) | CssValue::Math(_))
                || matches!(value, CssValue::Number(n) if *n == 0.0)
        }
        "scroll-padding-top"
        | "scroll-padding-right"
        | "scroll-padding-bottom"
        | "scroll-padding-left"
        | "scroll-padding-block-start"
        | "scroll-padding-block-end"
        | "scroll-padding-inline-start"
        | "scroll-padding-inline-end" => {
            // `<length-percentage> | auto`, non-negative (percentages
            // resolve against the scroll container's dimensions in
            // layout). Negative padding is invalid at computed-value
            // time: a literal negative fails here, a negative math result
            // fails the post-evaluation re-check in `compute_prop`.
            (is_length_or_pct(value) || is_kw(value, &["auto"])) && !is_negative_length(value)
        }
        "scroll-padding-block" | "scroll-padding-inline" => {
            // `<length-percentage> | auto` `{1,2}`: the 1- and 2-value
            // logical shorthands, non-negative per component.
            let one = (is_length_or_pct(value) || is_kw(value, &["auto"]))
                && !is_negative_length(value);
            let list = matches!(value, CssValue::List(items) if (1..=2).contains(&items.len())
                && items.iter().all(|v| (is_length_or_pct(v) || is_kw(v, &["auto"]))
                    && !is_negative_length(v)));
            one || list
        }
        "scroll-padding" => {
            // `<length-percentage> | auto` `{1,4}`: the physical
            // shorthand, non-negative per component.
            let one = (is_length_or_pct(value) || is_kw(value, &["auto"]))
                && !is_negative_length(value);
            let list = matches!(value, CssValue::List(items) if (1..=4).contains(&items.len())
                && items.iter().all(|v| (is_length_or_pct(v) || is_kw(v, &["auto"]))
                    && !is_negative_length(v)));
            one || list
        }
        "white-space" => is_kw(
            value,
            &[
                "normal",
                "pre",
                "nowrap",
                "pre-wrap",
                "pre-line",
                "break-spaces",
            ],
        ),
        "text-align" => is_kw(
            value,
            &["left", "right", "center", "justify", "start", "end"],
        ),
        "text-transform" => is_kw(value, &["none", "capitalize", "uppercase", "lowercase"]),
        "text-decoration" => matches!(
            value,
            CssValue::Keyword(_) | CssValue::List(_) | CssValue::Color(_)
        ),
        "text-decoration-style" => is_kw(value, &["solid", "double", "dotted", "dashed", "wavy"]),
        "vertical-align" => {
            is_length_or_pct(value)
                || is_kw(
                    value,
                    &[
                        "baseline",
                        "sub",
                        "super",
                        "top",
                        "text-top",
                        "middle",
                        "bottom",
                        "text-bottom",
                    ],
                )
        }
        "visibility" => is_kw(value, &["visible", "hidden", "collapse"]),
        "cursor" => matches!(
            value,
            CssValue::Keyword(_) | CssValue::Url(_) | CssValue::List(_)
        ),
        "direction" => is_kw(value, &["ltr", "rtl"]),
        // CSS Writing Modes 3 §2.6: the six `unicode-bidi` keywords.
        "unicode-bidi" => is_kw(
            value,
            &[
                "normal",
                "embed",
                "isolate",
                "bidi-override",
                "isolate-override",
                "plaintext",
            ],
        ),
        "box-sizing" => is_kw(value, &["content-box", "border-box"]),
        // css-sizing-4 §3.1: `auto` (no preferred ratio) or a folded
        // `AspectRatio` (see `fold_aspect_ratio`); anything else
        // (e.g. a bare `16 / 9` list that never folded) is invalid.
        // `Math` is admitted so `calc()` evaluates before the fold
        // (`compute_prop` runs math first, then folds); a math value
        // that survives evaluation is rejected by the post-evaluation
        // check below.
        "aspect-ratio" => {
            matches!(value, CssValue::AspectRatio { .. } | CssValue::Math(_))
                || is_kw(value, &["auto"])
        }
        "background-image" | "list-style-image" => {
            // Gradients are `<image>`s (CSS Images 3 §3.1): they survive
            // as `CssValue::Gradient` for the paint backend to parse
            // (`bowser_paint::gradient::parse_gradient`).
            matches!(value, CssValue::Url(_) | CssValue::Gradient { .. })
                || is_kw(value, &["none"])
        }
        "background-repeat" => is_kw(
            value,
            &[
                "repeat",
                "no-repeat",
                "repeat-x",
                "repeat-y",
                "space",
                "round",
            ],
        ),
        "background-position" => {
            is_length_or_pct(value) || matches!(value, CssValue::Keyword(_) | CssValue::List(_))
        }
        "list-style-type" => matches!(value, CssValue::Keyword(_)),
        "list-style-position" => is_kw(value, &["inside", "outside"]),
        // CSS Generated Content 3 §2: `content: normal | none |
        // <content-list>`, where a `<content-list>` item is a
        // `<string>`, an `attr()` reference, or a `counter()` /
        // `counters()` reference (each parsed to its own `CssValue`
        // shape; see `parse_function`). Adjacent `<string>`s
        // concatenate at computed-value time (see
        // `fold_content_value`); the `attr()` / `counter()` items
        // resolve element-scoped at box-synthesis time (see
        // `pseudo::resolve_content_text`). `<url>` / `<image>` are
        // still future work and stay invalid.
        "content" => {
            is_kw(value, &["normal", "none"])
                || is_content_item(value)
                || matches!(value, CssValue::List(items)
                    if !items.is_empty() && items.iter().all(is_content_item))
        }
        // CSS Lists 3 §4: `counter-reset` / `counter-increment` take
        // `[ <custom-ident> <integer>? ]+ | none`.
        "counter-reset" | "counter-increment" => is_counter_value(value),
        "border-collapse" => is_kw(value, &["collapse", "separate"]),
        "border-spacing" => {
            // Unitless zero is a valid `<length>`; `calc()` evaluates
            // after validation. One or two lengths (CSS 2.1 §17.5.2:
            // horizontal then vertical; a single value sets both).
            // Percentages are not valid `border-spacing`: a `calc()`
            // with percentage leaves is invalid at computed-value time
            // (CSS Values and Units 4 §10.9/§10.11: math functions are
            // type-checked and simplified at computed-value time), like
            // `box-shadow` lengths (see `is_shadow_length`). Layout therefore only ever sees
            // lengths here and needs no `had_pct` cyclic-percentage
            // gate; huge values are clamped by `sane_len` in
            // `SpecLen::resolve`.
            let is_spacing_len = |value: &CssValue| {
                let ok = match value {
                    CssValue::Length(_, _) => true,
                    CssValue::Number(n) => *n == 0.0,
                    CssValue::Math(e) => !e.contains_percentage(),
                    CssValue::MathDeferred { had_pct, .. } => !had_pct,
                    _ => false,
                };
                ok && !is_negative_length(value)
            };
            is_spacing_len(value)
                || matches!(value, CssValue::List(items)
                    if (1..=2).contains(&items.len()) && items.iter().all(is_spacing_len))
        }
        "table-layout" => is_kw(value, &["auto", "fixed"]),
        "caption-side" => is_kw(value, &["top", "bottom"]),
        "empty-cells" => is_kw(value, &["show", "hide"]),
        "flex-direction" => is_kw(value, &["row", "row-reverse", "column", "column-reverse"]),
        "flex-wrap" => is_kw(value, &["nowrap", "wrap", "wrap-reverse"]),
        "justify-content" => is_kw(
            value,
            &[
                "normal",
                "flex-start",
                "flex-end",
                "start",
                "end",
                "left",
                "right",
                "center",
                "space-between",
                "space-around",
                "space-evenly",
            ],
        ),
        "align-items" => is_kw(
            value,
            &[
                "normal",
                "stretch",
                "flex-start",
                "flex-end",
                "start",
                "end",
                "center",
                "baseline",
            ],
        ),
        // `align-content` additionally takes the <content-distribution>
        // keywords (flexbox §9.5); `align-items` does not.
        "align-content" => is_kw(
            value,
            &[
                "normal",
                "stretch",
                "flex-start",
                "flex-end",
                "start",
                "end",
                "center",
                "baseline",
                "space-between",
                "space-around",
                "space-evenly",
            ],
        ),
        "align-self" => is_valid_self_align(value),
        // Same self-alignment keywords as `align-self`; layout reads the
        // computed value per grid item (falling back to `justify-items`
        // on `auto`).
        "justify-self" => is_valid_self_align(value),
        // CSS Box Alignment 3 §6.1: `place-self: <align-self>
        // <justify-self>?`; a single value sets both longhands. A leading
        // `safe`/`unsafe` binds to the following keyword as one longhand,
        // so `place-self: unsafe center` is align-self `unsafe center`
        // (not align-self `unsafe` + justify-self `center`).
        "place-self" => split_place_self(value).is_some(),
        "letter-spacing" | "word-spacing" => {
            // `<length>` only (CSS Text 3 §9); a bare percentage is
            // rejected, and so is a `calc()` with percentage leaves —
            // invalid at computed-value time (CSS Values 4 §8.1.4), even
            // when the percentage folds to zero — like `box-shadow`
            // lengths (see `is_shadow_length`). Unitless zero is a valid
            // `<length>` (CSS-wide rule); negative lengths stay valid
            // (tighter spacing is allowed).
            let ok_len = match value {
                CssValue::Length(_, _) => true,
                CssValue::Number(n) => *n == 0.0,
                CssValue::Math(e) => !e.contains_percentage(),
                CssValue::MathDeferred { had_pct, .. } => !had_pct,
                _ => false,
            };
            ok_len || is_kw(value, &["normal"])
        }
        "row-gap" | "column-gap" => {
            // `<length-percentage> | normal`; `fr` is not a valid gap
            // (grid §10.2), so reject it here instead of letting layout
            // silently resolve it to zero. Negative gaps are invalid at
            // computed-value time, like negative padding.
            (is_length_or_pct(value) || is_kw(value, &["normal"])) && !is_negative_length(value)
        }
        // CSS Fragmentation 3: `break-before`/`break-after` take the
        // full break keyword set; `break-inside` only the avoid family.
        "break-before" | "break-after" => is_kw(
            value,
            &[
                "auto",
                "avoid",
                "page",
                "left",
                "right",
                "recto",
                "verso",
                "avoid-page",
                "column",
                "avoid-column",
            ],
        ),
        "break-inside" => is_kw(value, &["auto", "avoid", "avoid-page", "avoid-column"]),
        // `widows`/`orphans` are `<integer>` >= 1 (CSS Fragmentation 3).
        "widows" | "orphans" => {
            matches!(value, CssValue::Number(n) if n.fract() == 0.0 && *n >= 1.0)
        }
        // CSS Multicol 1: `column-count` is `<integer> | auto`
        // (clamped to >= 1 here; layout ignores 0/negative the same
        // way), `column-width` is `<length> | auto` (percentages are
        // not valid; `calc()` is accepted and resolved by layout like
        // every other length property).
        "column-count" => {
            is_kw(value, &["auto"])
                || matches!(value, CssValue::Number(n) if n.fract() == 0.0 && *n >= 1.0)
        }
        "column-width" => {
            // Percentages are not valid (CSS Multicol 1 §3:
            // `<length> | auto`); a `calc()` with percentage leaves is
            // invalid at computed-value time (CSS Values and Units 4
            // §10.9/§10.11: math functions are type-checked and
            // simplified at computed-value time) —
            // `box-shadow` lengths (see `is_shadow_length`). Layout
            // therefore only ever sees lengths here and needs no
            // `had_pct` cyclic-percentage gate.
            let ok_len = match value {
                CssValue::Length(_, _) => true,
                CssValue::Number(n) => *n == 0.0,
                CssValue::Math(e) => !e.contains_percentage(),
                CssValue::MathDeferred { had_pct, .. } => !had_pct,
                _ => false,
            };
            (ok_len || is_kw(value, &["auto"])) && !is_negative_length(value)
        }
        // CSS Multicol 1: `column-span` is `none | all`, `column-fill`
        // is `balance | auto` (single-keyword longhands).
        "column-span" => is_kw(value, &["none", "all"]),
        "column-fill" => is_kw(value, &["balance", "auto"]),
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// Deferred-shorthand expansion
// ---------------------------------------------------------------------------
//
// The parser stores a shorthand whose value contains `var()` as-is under
// the shorthand property name; the cascade substitutes the variables and
// then runs the same expansion the parser would have. The functions below
// are faithful mirrors of parse.rs's private `expand_box`,
// `expand_border_sides`, `expand_column_rule`, `expand_border_radius`,
// `expand_background`, `expand_list_style`, and `expand_flex`
// (which this module cannot call directly): they run on the substituted
// value — no `Var` nodes remain — emit every longhand, and reset omitted
// components to their initial values. Keep them in sync with parse.rs.

/// Properties the parser defers (rather than expands) when their value
/// contains a `var()` reference. Atom match: one `u8` compare per arm
/// set, no string hashing. The arm set must stay identical to the old
/// string list (every arm has an atom: the shorthand extras in the
/// generated table cover the non-computed names).
fn is_shorthand_property(property: &PropertyName) -> bool {
    matches!(
        property.atom(),
        PropertyAtom::MARGIN
            | PropertyAtom::PADDING
            | PropertyAtom::BORDER_WIDTH
            | PropertyAtom::BORDER_STYLE
            | PropertyAtom::BORDER_COLOR
            | PropertyAtom::BORDER
            | PropertyAtom::BORDER_TOP
            | PropertyAtom::BORDER_RIGHT
            | PropertyAtom::BORDER_BOTTOM
            | PropertyAtom::BORDER_LEFT
            | PropertyAtom::BORDER_RADIUS
            | PropertyAtom::BACKGROUND
            | PropertyAtom::LIST_STYLE
            | PropertyAtom::FONT
            | PropertyAtom::FLEX
            | PropertyAtom::GAP
            | PropertyAtom::GRID_GAP
            | PropertyAtom::PLACE_SELF
            | PropertyAtom::OVERFLOW
            | PropertyAtom::OVERSCROLL_BEHAVIOR
            | PropertyAtom::SCROLL_MARGIN
            | PropertyAtom::SCROLL_PADDING
            | PropertyAtom::SCROLL_MARGIN_INLINE
            | PropertyAtom::SCROLL_MARGIN_BLOCK
            | PropertyAtom::SCROLL_PADDING_INLINE
            | PropertyAtom::SCROLL_PADDING_BLOCK
            | PropertyAtom::COLUMNS
            | PropertyAtom::COLUMN_RULE
            | PropertyAtom::MARGIN_INLINE
            | PropertyAtom::MARGIN_BLOCK
            | PropertyAtom::PADDING_INLINE
            | PropertyAtom::PADDING_BLOCK
            | PropertyAtom::INSET_INLINE
            | PropertyAtom::INSET_BLOCK
            | PropertyAtom::BORDER_INLINE_WIDTH
            | PropertyAtom::BORDER_INLINE_STYLE
            | PropertyAtom::BORDER_INLINE_COLOR
            | PropertyAtom::BORDER_BLOCK_WIDTH
            | PropertyAtom::BORDER_BLOCK_STYLE
            | PropertyAtom::BORDER_BLOCK_COLOR
    )
}

/// Maximum tokens one nested-`var()` substitution may emit. Bounds
/// hostile diamond-shaped custom-property chains (`--a: var(--b)
/// var(--b)` over N levels doubles the output per level); legitimate
/// values are orders of magnitude smaller.
const MAX_SUBST_TOKENS: usize = 65_536;

/// Maximum custom-property expansions in one
/// [`Styler::substitute_math_vars`] pass. Each expansion runs the full
/// [`Styler::resolve`] machine plus a re-parse, so the unit of work is
/// far costlier than a token-path token: 4096 bounds even a hostile
/// diamond (`--x: calc(var(--y) + var(--y))` per level doubles the
/// expansions at constant depth, so depth alone cannot catch it) to
/// well under a second, while no legitimate `calc()` comes close.
const MAX_MATH_VAR_EXPANSIONS: usize = 4096;

/// Maximum structural recursion depth in one
/// [`Styler::substitute_math_vars`] pass: pure [`MathExpr`] tree descent
/// (the `Add`/`Neg`/`Min`/… arms) with no `var()` substitution work.
/// Each parsed math tree is already nesting-capped at parse time, so
/// this is only a stack-safety backstop for trees composed by
/// substitution; it is deliberately far above any legitimate input and
/// far below stack exhaustion. The [`MAX_MATH_DEPTH`] counter, by
/// contrast, counts only real `var()` expansions (see
/// [`Styler::substitute_math_vars`]), so a calc-wrapped chain like
/// `--ci: calc(var(--c{i-1}) + 0px)` costs one unit per link, not two.
const MAX_MATH_SUBST_STRUCT_DEPTH: u32 = 1024;

/// Whether `toks` holds a `var()` function token nested inside another
/// function or paren/bracket group (depth > 0). A top-level `var()`
/// re-parses into a [`CssValue::Var`] the resolve machine substitutes
/// structurally; only a nested one needs token-level substitution (see
/// [`substitute_nested_vars`]).
fn tokens_contain_nested_var(toks: &[Token]) -> bool {
    let mut depth = 0u32;
    for t in toks {
        match t {
            Token::Function(name) => {
                if depth > 0 && name.eq_ignore_ascii_case("var") {
                    return true;
                }
                depth += 1;
            }
            Token::LParen | Token::LBracket => depth += 1,
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    false
}

/// Index of the token closing the function/paren/bracket opened at
/// `open` (`toks[open]` is `Function`/`LParen`/`LBracket`), or `None`
/// when unbalanced.
fn find_matching_close(toks: &[Token], open: usize) -> Option<usize> {
    let mut d = 1u32;
    let mut j = open + 1;
    while j < toks.len() {
        match &toks[j] {
            Token::Function(_) | Token::LParen | Token::LBracket => d += 1,
            Token::RParen | Token::RBracket => {
                d -= 1;
                if d == 0 {
                    return Some(j);
                }
            }
            _ => {}
        }
        j += 1;
    }
    None
}

/// Strip leading/trailing whitespace from a token slice.
fn trim_token_slice(toks: &[Token]) -> &[Token] {
    let mut s = toks;
    while let Some((first, rest)) = s.split_first()
        && matches!(first, Token::Whitespace)
    {
        s = rest;
    }
    while let Some((last, rest)) = s.split_last()
        && matches!(last, Token::Whitespace)
    {
        s = rest;
    }
    s
}

/// Substitute `var()` references nested inside other functions (or
/// paren/bracket groups) at the token level — e.g. inside an opaque
/// gradient body the parser kept verbatim (see `parse_function`'s
/// gradient arm). The re-parse in the `Tokens` arm turns a top-level
/// `var()` into a [`CssValue::Var`] the machine substitutes
/// structurally, but a nested one would freeze verbatim (or re-parse
/// back into `Tokens` and loop the work stack), so it is spliced here
/// against `env` before parsing.
///
/// Returns the substituted stream, or `None` when a reference is
/// unresolvable, cyclic, or hostile: invalid at computed-value time.
///
/// `nested_only` leaves a top-level `var()` (depth 0) untouched so it
/// still flows through the `Var` arm with its memo, fallback, and
/// splice semantics. Spliced-in custom-property streams and `var()`
/// fallbacks substitute at every depth: any `var()` they still held
/// would otherwise need another resolve trip, and the `active` set
/// below is trip-local, so a self-referential property would loop the
/// machine instead of failing closed.
///
/// Cycle detection reuses the resolve machine's `active` set of
/// interned custom-property ids. `depth` caps hostile reference chains
/// at [`MAX_MATH_DEPTH`]; `budget` spends one per emitted token and
/// caps hostile diamond shapes.
fn substitute_nested_vars(
    toks: &[Token],
    env: &FxHashMap<String, CssValue>,
    var_names: &RefCell<VarNameInterner>,
    active: &mut FxHashSet<u32>,
    nested_only: bool,
    depth: u32,
    budget: &mut usize,
) -> Option<Vec<Token>> {
    if depth > MAX_MATH_DEPTH {
        return None;
    }
    let mut out: Vec<Token> = Vec::with_capacity(toks.len());
    let mut i = 0;
    let mut rel = 0u32;
    while i < toks.len() {
        let is_top_var = nested_only
            && rel == 0
            && matches!(&toks[i], Token::Function(n) if n.eq_ignore_ascii_case("var"));
        if is_top_var {
            // A top-level `var()` — fallback included — is left verbatim
            // for the `Var` arm: eagerly substituting inside a fallback
            // that may never be used would wrongly invalidate the
            // declaration when the property *is* defined.
            let j = find_matching_close(toks, i)?;
            for t in &toks[i..=j] {
                if *budget == 0 {
                    return None;
                }
                *budget -= 1;
                out.push(t.clone());
            }
            i = j + 1;
            continue;
        }
        let is_var = match &toks[i] {
            Token::Function(name) => (!nested_only || rel > 0) && name.eq_ignore_ascii_case("var"),
            _ => false,
        };
        if is_var {
            let j = find_matching_close(toks, i)?;
            let replacement =
                substitute_var_ref(&toks[i + 1..j], env, var_names, active, depth, budget)?;
            out.extend(replacement);
            // The skipped span is a balanced function: it nets zero
            // depth, and its `Function` token was never counted, so
            // `rel` is unchanged.
            i = j + 1;
            continue;
        }
        match &toks[i] {
            Token::Function(_) | Token::LParen | Token::LBracket => rel += 1,
            Token::RParen | Token::RBracket => rel = rel.saturating_sub(1),
            _ => {}
        }
        if *budget == 0 {
            return None;
        }
        *budget -= 1;
        out.push(toks[i].clone());
        i += 1;
    }
    Some(out)
}

/// Resolve one `var()` reference — its already-extracted body tokens —
/// to replacement tokens: the named custom property's verbatim stream,
/// or the fallback's tokens, each recursively substituted (see
/// [`substitute_nested_vars`]). `None` = unresolvable, cyclic, or
/// hostile: invalid at computed-value time.
fn substitute_var_ref(
    body: &[Token],
    env: &FxHashMap<String, CssValue>,
    var_names: &RefCell<VarNameInterner>,
    active: &mut FxHashSet<u32>,
    depth: u32,
    budget: &mut usize,
) -> Option<Vec<Token>> {
    // Split the property name from the fallback at the first top-level
    // comma, mirroring `parse_var`.
    let mut comma = None;
    let mut d = 0u32;
    for (k, t) in body.iter().enumerate() {
        match t {
            Token::Function(_) | Token::LParen | Token::LBracket => d += 1,
            Token::RParen | Token::RBracket => d = d.saturating_sub(1),
            Token::Comma if d == 0 => {
                comma = Some(k);
                break;
            }
            _ => {}
        }
    }
    let (name_toks, fallback_toks) = match comma {
        Some(k) => (&body[..k], Some(&body[k + 1..])),
        None => (body, None),
    };
    let name: String = match name_toks
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .collect::<Vec<_>>()
        .as_slice()
    {
        [Token::Ident(n)] if n.starts_with("--") => n.clone(),
        _ => return None,
    };
    let id = var_names.borrow_mut().intern(&name);
    if !active.insert(id) {
        return None; // cyclic reference: invalid at computed-value time
    }
    let result = match env.get(name.as_str()) {
        Some(CssValue::Tokens(t)) => {
            substitute_nested_vars(t, env, var_names, active, false, depth + 1, budget)
        }
        // A missing or guaranteed-invalid property falls back; anything
        // non-verbatim cannot occur (custom properties store token
        // streams) and fails closed.
        _ => match fallback_toks {
            Some(ft) => {
                let ft = trim_token_slice(ft);
                if ft.is_empty() {
                    None
                } else {
                    substitute_nested_vars(ft, env, var_names, active, false, depth + 1, budget)
                }
            }
            None => None,
        },
    };
    active.remove(&id);
    result
}

/// Whether a parsed value contains a `var()` reference anywhere (including
/// nested inside lists and inside `calc()`/`min()`/`max()`/`clamp()`
/// math trees; a `Var` node itself always counts).
fn value_contains_var(value: &CssValue) -> bool {
    match value {
        CssValue::Var(_, _) => true,
        CssValue::Math(e) => e.contains_var(),
        CssValue::CalcSize { basis, calc } => {
            calc.contains_var()
                || matches!(
                    basis.as_ref(),
                    CalcSizeBasis::Math(e) if e.contains_var()
                )
                // A `var()` basis always needs substitution.
                || matches!(basis.as_ref(), CalcSizeBasis::Var { .. })
        }
        CssValue::List(items) => items.iter().any(value_contains_var),
        CssValue::Repeat { tracks, .. } => tracks.iter().any(value_contains_var),
        CssValue::Tokens(toks) => toks
            .iter()
            .any(|t| matches!(t, Token::Function(name) if name.eq_ignore_ascii_case("var"))),
        // A `Gradient` retains its source verbatim: a `var(` in it means
        // the cascade must defer substitution. The parser no longer
        // emits these (see `parse_function`'s gradient arm), so this is
        // defense-in-depth for values built elsewhere.
        CssValue::Gradient { source, .. } => css_source_contains_var(source),
        _ => false,
    }
}

/// Run the parser's expansion for a substituted deferred shorthand,
/// returning `(longhand, value)` pairs. `important` is already encoded in
/// the candidate's cascade rank.
fn expand_deferred_shorthand(
    property: &PropertyName,
    value: CssValue,
) -> Vec<(PropertyName, CssValue)> {
    // The `*_value` mirrors below still return `(String, CssValue)`
    // pairs; interning happens once at this boundary so the cascade hot
    // path above works purely on atoms.
    let pairs: Vec<(String, CssValue)> = match property.atom() {
        PropertyAtom::MARGIN
        | PropertyAtom::PADDING
        | PropertyAtom::BORDER_WIDTH
        | PropertyAtom::BORDER_STYLE
        | PropertyAtom::BORDER_COLOR
        | PropertyAtom::SCROLL_MARGIN
        | PropertyAtom::SCROLL_PADDING => expand_box(property.as_str(), value),
        PropertyAtom::MARGIN_INLINE
        | PropertyAtom::MARGIN_BLOCK
        | PropertyAtom::PADDING_INLINE
        | PropertyAtom::PADDING_BLOCK
        | PropertyAtom::SCROLL_MARGIN_INLINE
        | PropertyAtom::SCROLL_MARGIN_BLOCK
        | PropertyAtom::SCROLL_PADDING_INLINE
        | PropertyAtom::SCROLL_PADDING_BLOCK
        | PropertyAtom::INSET_INLINE
        | PropertyAtom::INSET_BLOCK
        | PropertyAtom::BORDER_INLINE_WIDTH
        | PropertyAtom::BORDER_INLINE_STYLE
        | PropertyAtom::BORDER_INLINE_COLOR
        | PropertyAtom::BORDER_BLOCK_WIDTH
        | PropertyAtom::BORDER_BLOCK_STYLE
        | PropertyAtom::BORDER_BLOCK_COLOR => expand_logical_sides_value(property.as_str(), &value),
        PropertyAtom::BORDER => expand_border_sides(&["border"], value),
        PropertyAtom::BORDER_TOP
        | PropertyAtom::BORDER_RIGHT
        | PropertyAtom::BORDER_BOTTOM
        | PropertyAtom::BORDER_LEFT => expand_border_sides(&[property.as_str()], value),
        PropertyAtom::BORDER_RADIUS => expand_border_radius(value),
        PropertyAtom::BACKGROUND => expand_background(value),
        PropertyAtom::LIST_STYLE => expand_list_style(value),
        PropertyAtom::FONT => expand_font_value(&value),
        PropertyAtom::FLEX => expand_flex_value(&value),
        PropertyAtom::GAP | PropertyAtom::GRID_GAP => expand_gap_value(&value),
        PropertyAtom::PLACE_SELF => expand_place_self_value(&value),
        PropertyAtom::OVERFLOW => expand_overflow_value(&value),
        PropertyAtom::OVERSCROLL_BEHAVIOR => expand_overscroll_behavior_value(&value),
        PropertyAtom::COLUMNS => expand_columns_value(&value),
        PropertyAtom::COLUMN_RULE => expand_column_rule_value(&value),
        _ => vec![(property.as_str().to_string(), value)],
    };
    pairs
        .into_iter()
        .map(|(name, v)| (PropertyName::from(name), v))
        .collect()
}

fn decl(property: &str, value: CssValue) -> (String, CssValue) {
    (property.to_string(), value)
}

/// Split a value into its whitespace/comma-separated items.
/// Combine the `n` resolved items of one list-like value back into a
/// single value: the shared combination rules behind
/// [`ResolveWork::ListTail`] and [`ResolveWork::RepeatTail`]. Results pop
/// in reverse and flip to source order (a slash-marked substitution
/// merges with the siblings on either side, so order matters); single
/// values push, list-valued `var()` substitutions splice in item-wise,
/// a top-level `/` splits into before/after groups (a second one is
/// invalid at computed-value time), and comma groups splice as one
/// nested list per group, preserving the group structure for the
/// deferred `font` expansion (every other consumer normalizes back via
/// [`normalize_comma_groups`]). The `wrap` closure builds the final
/// value from the combined items and the optional slash split, so lists
/// and `repeat()` descriptors share the rules while keeping their own
/// shapes.
fn combine_list_tail(
    results: &mut Vec<ResolveResult>,
    n: usize,
    wrap: impl FnOnce(Vec<CssValue>, Option<(Vec<CssValue>, Vec<CssValue>)>) -> CssValue,
) {
    let mut rev: Vec<ResolveResult> = Vec::with_capacity(n);
    let mut ok = true;
    for _ in 0..n {
        match results.pop().unwrap_or(ResolveResult::Single(None)) {
            ResolveResult::Single(None) | ResolveResult::Splice(None) => ok = false,
            // The `Rc` payloads report their own `None`: invalid results
            // were filtered above, so the arms below can unwrap.
            ResolveResult::Slash(s) if s.is_none() => ok = false,
            ResolveResult::Comma(g) if g.is_none() => ok = false,
            r => rev.push(r),
        }
    }
    rev.reverse();
    let mut items: Vec<CssValue> = Vec::new();
    // At most one top-level `/` may split the list: siblings before the
    // `var()` join the `before` group, siblings after join the `after`
    // group. A second `/` is invalid at computed-value time.
    let mut slash: Option<(Vec<CssValue>, Vec<CssValue>)> = None;
    for r in rev {
        match r {
            ResolveResult::Single(Some(v)) => match &mut slash {
                Some((_, after)) => after.push(v),
                None => items.push(v),
            },
            // Token-wise splice of a substituted list-valued `var()`.
            ResolveResult::Splice(Some(vs)) => match &mut slash {
                Some((_, after)) => after.extend(vs),
                None => items.extend(vs),
            },
            ResolveResult::Slash(s) => {
                // Invalid (`None`) results were filtered above. The
                // payload may be shared with the `var()` memo: unwrap
                // moves when uniquely owned, and clones the two group
                // `Vec`s only when the memo still holds a reference —
                // the same single copy the old probe-then-move paid.
                let Some((before, after)) = Rc::unwrap_or_clone(s) else {
                    unreachable!("invalid results filtered")
                };
                if slash.is_some() {
                    ok = false; // two top-level slashes
                    break;
                }
                let mut b = std::mem::take(&mut items);
                b.extend(before);
                slash = Some((b, after));
            }
            // A comma-separated substitution splices one nested list per
            // group, preserving the group structure for the deferred
            // `font` expansion (every other consumer normalizes back via
            // [`normalize_comma_groups`]).
            ResolveResult::Comma(g) => {
                let Some(groups) = Rc::unwrap_or_clone(g) else {
                    unreachable!("invalid results filtered")
                };
                match &mut slash {
                    Some((_, after)) => after.extend(groups.into_iter().map(CssValue::List)),
                    None => items.extend(groups.into_iter().map(CssValue::List)),
                }
            }
            // Invalid results were filtered above.
            ResolveResult::Single(None) | ResolveResult::Splice(None) => {
                unreachable!("invalid results filtered")
            }
        }
    }
    let value = if ok { Some(wrap(items, slash)) } else { None };
    results.push(ResolveResult::Single(value));
}

fn as_list(value: &CssValue) -> Vec<CssValue> {
    match value {
        CssValue::List(items) => items.clone(),
        _ => vec![value.clone()],
    }
}

/// `gap` / `grid-gap` (`<row> || <column>`) expand to `row-gap` and
/// `column-gap` after `var()` substitution.
fn expand_gap_value(value: &CssValue) -> Vec<(String, CssValue)> {
    let items = as_list(value);
    if items.len() > 2 {
        return Vec::new(); // invalid: drop the declaration
    }
    let row = items
        .first()
        .cloned()
        .unwrap_or(CssValue::Keyword("normal".into()));
    let col = items.get(1).cloned().unwrap_or_else(|| row.clone());
    vec![decl("row-gap", row), decl("column-gap", col)]
}

/// CSS Box Alignment 3 §6: `<overflow-position> = safe | unsafe`.
fn is_overflow_position_kw(kw: &str) -> bool {
    kw.eq_ignore_ascii_case("safe") || kw.eq_ignore_ascii_case("unsafe")
}

/// CSS Box Alignment 3 §6: `<self-position>`, the subset that may follow
/// an `<overflow-position>` (`baseline` may not: the grammar is
/// `auto | normal | stretch | <baseline-position> |
/// <overflow-position>? <self-position>`).
fn is_self_position_kw(kw: &str) -> bool {
    kw.eq_ignore_ascii_case("center")
        || kw.eq_ignore_ascii_case("start")
        || kw.eq_ignore_ascii_case("end")
        || kw.eq_ignore_ascii_case("self-start")
        || kw.eq_ignore_ascii_case("self-end")
        || kw.eq_ignore_ascii_case("flex-start")
        || kw.eq_ignore_ascii_case("flex-end")
}

/// Validity for `align-self` / `justify-self`: a lone keyword, or an
/// `<overflow-position>` + `<self-position>` pair (kept as a two-item
/// [`CssValue::List`] so layout can read the overflow behavior).
fn is_valid_self_align(value: &CssValue) -> bool {
    if let CssValue::List(items) = value
        && items.len() == 2
        && let (CssValue::Keyword(a), CssValue::Keyword(b)) = (&items[0], &items[1])
    {
        return is_overflow_position_kw(a) && is_self_position_kw(b);
    }
    if let CssValue::List(_) = value {
        return false;
    }
    is_kw_list(
        value,
        &[
            "auto",
            "normal",
            "stretch",
            "flex-start",
            "flex-end",
            "start",
            "end",
            "self-start",
            "self-end",
            "center",
            "baseline",
        ],
    )
}

/// Keyword-list membership for a [`CssValue`], the same check the
/// `is_kw` closure inside [`is_valid_for_property`] performs (that
/// closure is function-local, so helpers defined outside it use this).
fn is_kw_list(value: &CssValue, words: &[&str]) -> bool {
    matches!(value, CssValue::Keyword(k) if words.contains(&k.as_str()))
}

/// Split a `place-self` value into its `align-self` / `justify-self`
/// longhand values per CSS Box Alignment 3 §6.1:
/// `place-self: <align-self> <justify-self>?`. A leading `safe`/`unsafe`
/// binds to the *following* keyword as one longhand component, so
/// `place-self: unsafe center` yields align-self `unsafe center` (a
/// two-item list), not align-self `unsafe` + justify-self `center`.
/// Returns `None` when the value is outside the shorthand grammar.
pub(crate) fn split_place_self(value: &CssValue) -> Option<(CssValue, CssValue)> {
    let items = as_list(value);
    // Parse one longhand component: an overflow keyword consumes the
    // next token too; anything else is a lone keyword.
    fn take_component(items: &[CssValue]) -> Option<(CssValue, usize)> {
        let first = items.first()?;
        if let CssValue::Keyword(k) = first
            && is_overflow_position_kw(k)
        {
            let second = items.get(1)?;
            return Some((CssValue::List(vec![first.clone(), second.clone()]), 2));
        }
        Some((first.clone(), 1))
    }
    let (align, n) = take_component(&items)?;
    let rest = &items[n..];
    let justify = if rest.is_empty() {
        align.clone()
    } else {
        let (j, m) = take_component(rest)?;
        if m != rest.len() {
            return None;
        }
        j
    };
    if !is_valid_self_align(&align) || !is_valid_self_align(&justify) {
        return None;
    }
    Some((align, justify))
}

/// `place-self` (`<align-self> <justify-self>?`) expands to
/// `align-self` and `justify-self` after `var()` substitution,
/// mirroring the parser's `expand_place_self` in parse.rs. A lone
/// value sets both longhands; a component outside the longhand
/// grammar invalidates the whole shorthand (it must not partially
/// apply). A lone CSS-wide keyword still expands; the cascade
/// intercepts it per longhand.
fn expand_place_self_value(value: &CssValue) -> Vec<(String, CssValue)> {
    let items = as_list(value);
    let lone_wide = items.len() == 1 && is_css_wide_keyword(&items[0]);
    if lone_wide {
        let kw = items.into_iter().next().unwrap();
        return vec![decl("align-self", kw.clone()), decl("justify-self", kw)];
    }
    // The `place-self` validation arm encodes the 1-2-value grammar
    // against the two longhands, so one call checks both components.
    match split_place_self(value) {
        Some((align, justify)) => vec![decl("align-self", align), decl("justify-self", justify)],
        None => Vec::new(), // invalid: drop the declaration
    }
}

/// `columns` (`<column-width> || <column-count>`) expands to
/// `column-width` and `column-count` after `var()` substitution,
/// mirroring the parser's `expand_columns` in parse.rs.
fn expand_columns_value(value: &CssValue) -> Vec<(String, CssValue)> {
    match split_columns_items(value) {
        Some((width, count)) => vec![decl("column-width", width), decl("column-count", count)],
        None => Vec::new(), // invalid: drop the declaration
    }
}

/// Split a `columns` value into `(column-width, column-count)`.
/// Grammar (CSS Multicol 1 §3): `<column-width> || <column-count>`,
/// order-independent, each side `auto` when absent. A bare `auto` fills
/// the first empty side; two widths, two counts, or anything else is
/// invalid (`None`).
fn split_columns_items(value: &CssValue) -> Option<(CssValue, CssValue)> {
    let items = as_list(value);
    if items.is_empty() || items.len() > 2 {
        return None;
    }
    let auto = || CssValue::Keyword("auto".into());
    let mut width: Option<CssValue> = None;
    let mut count: Option<CssValue> = None;
    for item in &items {
        let is_count = matches!(item, CssValue::Number(n) if n.fract() == 0.0 && *n >= 1.0);
        let is_width = matches!(
            item,
            CssValue::Length(_, _) | CssValue::Math(_) | CssValue::MathDeferred { .. }
        ) || matches!(item, CssValue::Number(n) if *n == 0.0)
            || matches!(item, CssValue::Keyword(k) if k.eq_ignore_ascii_case("auto"));
        let is_auto = matches!(item, CssValue::Keyword(k) if k.eq_ignore_ascii_case("auto"));
        if is_auto {
            // `auto` fills whichever side is still empty (width first,
            // so `columns: auto 3` is width:auto count:3).
            if width.is_none() {
                width = Some(auto());
            } else if count.is_none() {
                count = Some(auto());
            } else {
                return None;
            }
        } else if is_count && count.is_none() {
            count = Some(item.clone());
        } else if is_width && width.is_none() {
            if matches!(item, CssValue::Length(n, _) if *n < 0.0) {
                return None;
            }
            width = Some(item.clone());
        } else {
            return None;
        }
    }
    Some((width.unwrap_or_else(auto), count.unwrap_or_else(auto)))
}

/// `overflow` (`<overflow-x> <overflow-y>?`) expands to `overflow-x` and
/// `overflow-y` after `var()` substitution, mirroring the parser's
/// `expand_overflow` in parse.rs.
fn expand_overflow_value(value: &CssValue) -> Vec<(String, CssValue)> {
    let items = as_list(value);
    if items.len() > 2 {
        return Vec::new(); // invalid: drop the declaration
    }
    // A component outside the longhand grammar invalidates the whole
    // shorthand: it must not partially apply. A lone CSS-wide keyword
    // still expands; the cascade intercepts it per longhand.
    let lone_wide = items.len() == 1 && is_css_wide_keyword(&items[0]);
    if !lone_wide && !items.iter().all(|i| is_valid_for_property("overflow-x", i)) {
        return Vec::new(); // invalid: drop the declaration
    }
    let x = items
        .first()
        .cloned()
        .unwrap_or(CssValue::Keyword("visible".into()));
    let y = items.get(1).cloned().unwrap_or_else(|| x.clone());
    vec![decl("overflow-x", x), decl("overflow-y", y)]
}

/// `overscroll-behavior` (`<overscroll-behavior-x>
/// <overscroll-behavior-y>?`) expands to `overscroll-behavior-x` and
/// `overscroll-behavior-y` after `var()` substitution, mirroring the
/// parser's `expand_overscroll_behavior` in parse.rs.
fn expand_overscroll_behavior_value(value: &CssValue) -> Vec<(String, CssValue)> {
    let items = as_list(value);
    if items.len() > 2 {
        return Vec::new(); // invalid: drop the declaration
    }
    // A component outside the longhand grammar invalidates the whole
    // shorthand: it must not partially apply. A lone CSS-wide keyword
    // still expands; the cascade intercepts it per longhand.
    let lone_wide = items.len() == 1 && is_css_wide_keyword(&items[0]);
    if !lone_wide
        && !items
            .iter()
            .all(|i| is_valid_for_property("overscroll-behavior-x", i))
    {
        return Vec::new(); // invalid: drop the declaration
    }
    let x = items
        .first()
        .cloned()
        .unwrap_or(CssValue::Keyword("auto".into()));
    let y = items.get(1).cloned().unwrap_or_else(|| x.clone());
    vec![
        decl("overscroll-behavior-x", x),
        decl("overscroll-behavior-y", y),
    ]
}

fn expand_box(property: &str, value: CssValue) -> Vec<(String, CssValue)> {
    let items = as_list(&value);
    let [top, right, bottom, left] = match items.len() {
        1 => [0, 0, 0, 0].map(|_| items[0].clone()),
        2 => [
            items[0].clone(),
            items[1].clone(),
            items[0].clone(),
            items[1].clone(),
        ],
        3 => [
            items[0].clone(),
            items[1].clone(),
            items[2].clone(),
            items[1].clone(),
        ],
        4 => [
            items[0].clone(),
            items[1].clone(),
            items[2].clone(),
            items[3].clone(),
        ],
        _ => return Vec::new(), // invalid: drop the declaration
    };
    // `border-width` -> `border-top-width`; `margin` -> `margin-top`.
    let longhand = |side: &str| {
        if let Some(rest) = property.strip_prefix("border-") {
            format!("border-{side}-{rest}")
        } else {
            format!("{property}-{side}")
        }
    };
    // A component outside the longhand grammar invalidates the whole
    // shorthand: it must not partially apply. A lone CSS-wide keyword
    // still expands; the cascade intercepts it per longhand.
    let probe = longhand("top");
    let lone_wide = items.len() == 1 && is_css_wide_keyword(&items[0]);
    if !lone_wide && !items.iter().all(|i| is_valid_for_property(&probe, i)) {
        return Vec::new(); // invalid: drop the declaration
    }
    ["top", "right", "bottom", "left"]
        .into_iter()
        .zip([top, right, bottom, left])
        .map(|(side, v)| decl(&longhand(side), v))
        .collect()
}

/// Expand a `var()`-deferred logical shorthand after substitution:
/// the cascade-side twin of `expand_logical_sides` (parse.rs).
fn expand_logical_sides_value(property: &str, value: &CssValue) -> Vec<(String, CssValue)> {
    let items = as_list(value);
    let [start, end] = match items.len() {
        1 => [items[0].clone(), items[0].clone()],
        2 => [items[0].clone(), items[1].clone()],
        _ => return Vec::new(), // invalid: drop the declaration
    };
    let longhand = |edge: &str| {
        if let Some(rest) = property.strip_prefix("border-") {
            // `border-inline-width` -> `border-inline-start-width`.
            let (axis, kind) = rest
                .rsplit_once('-')
                .expect("logical border shorthand shape");
            format!("border-{axis}-{edge}-{kind}")
        } else {
            // `margin-inline` -> `margin-inline-start`.
            format!("{property}-{edge}")
        }
    };
    // A component outside the longhand grammar invalidates the whole
    // shorthand: it must not partially apply (mirrors `expand_box`'s
    // probe). A lone CSS-wide keyword still expands; the cascade
    // intercepts it per longhand.
    let probe = longhand("start");
    let lone_wide = items.len() == 1 && is_css_wide_keyword(&items[0]);
    if !lone_wide && !items.iter().all(|i| is_valid_for_property(&probe, i)) {
        return Vec::new(); // invalid: drop the declaration
    }
    vec![decl(&longhand("start"), start), decl(&longhand("end"), end)]
}

/// `border-radius: [<length-percentage>]{1,4} [ / <length-percentage>{1,4} ]?`
/// expands to per-corner `border-<corner>-radius` longhands (`top-left`,
/// `top-right`, `bottom-right`, `bottom-left`). The parser stores the
/// `/`-separated form as `List([List(h..), List(v..)])`; a comma group
/// never nests this way, so nesting unambiguously marks `/`. Without `/`
/// each corner gets a single value; with `/` it gets a `[horizontal
/// vertical]` pair list. Invalid (bad components, wrong count) drops the
/// declaration.
fn expand_border_radius(value: CssValue) -> Vec<(String, CssValue)> {
    let (h, v) = match value {
        CssValue::List(mut groups)
            if groups.len() == 2 && groups.iter().all(|g| matches!(g, CssValue::List(_))) =>
        {
            let v = groups.pop().expect("two groups");
            let h = groups.pop().expect("two groups");
            (as_list(&h), Some(as_list(&v)))
        }
        other => (as_list(&other), None),
    };
    if !(1..=4).contains(&h.len()) || !h.iter().all(is_radius_component) {
        return Vec::new();
    }
    let v = match v {
        Some(vv) if (1..=4).contains(&vv.len()) && vv.iter().all(is_radius_component) => Some(vv),
        Some(_) => return Vec::new(),
        None => None,
    };
    // 1-4 values -> [top-left, top-right, bottom-right, bottom-left]:
    // [a], [a b], [a b c], [a b c d] -> [a a a a], [a b a b], [a b c b], [a b c d].
    let quad = |items: &[CssValue], i: usize| match items.len() {
        1 => items[0].clone(),
        2 => items[i % 2].clone(),
        3 => items[if i == 2 { 2 } else { i % 2 }].clone(),
        _ => items[i].clone(),
    };
    ["top-left", "top-right", "bottom-right", "bottom-left"]
        .into_iter()
        .enumerate()
        .map(|(i, corner)| {
            let hv = match &v {
                Some(vv) => CssValue::List(vec![quad(&h, i), quad(vv, i)]),
                None => quad(&h, i),
            };
            decl(&format!("border-{corner}-radius"), hv)
        })
        .collect()
}

fn is_border_style(kw: &str) -> bool {
    matches!(
        kw,
        "none"
            | "hidden"
            | "dotted"
            | "dashed"
            | "solid"
            | "double"
            | "groove"
            | "ridge"
            | "inset"
            | "outset"
    )
}

fn is_border_width(kw: &str) -> bool {
    matches!(kw, "thin" | "medium" | "thick")
}

/// `border: <width> <style> <color>` and `border-<side>: ...` in any order,
/// 1-3 components. `border` covers all four sides and expands to per-side
/// longhands (`border-top-width`, ...). Every longhand is emitted;
/// components absent from the shorthand reset to their initial values
/// (`medium`, `none`, `currentcolor`). Unclassifiable items (unknown
/// keywords) drop the whole declaration.
fn expand_border_sides(prefixes: &[&str], value: CssValue) -> Vec<(String, CssValue)> {
    let mut width = None;
    let mut style = None;
    let mut color = None;
    for item in as_list(&value) {
        match item {
            // Unitless zero is a valid `<length>` (CSS-wide rule), and a
            // `calc()` width evaluates in the cascade after expansion.
            CssValue::Length(_, _) | CssValue::Math(_) | CssValue::MathDeferred { .. } => {
                if width.is_some() {
                    return Vec::new();
                }
                width = Some(item);
            }
            CssValue::Number(0.0) => {
                if width.is_some() {
                    return Vec::new();
                }
                width = Some(item);
            }
            CssValue::Color(_) => {
                if color.is_some() {
                    return Vec::new();
                }
                color = Some(item);
            }
            CssValue::Keyword(ref kw) => {
                let k = kw.as_str();
                if k == "currentcolor" {
                    if color.is_some() {
                        return Vec::new();
                    }
                    color = Some(item);
                } else if is_border_width(k) {
                    if width.is_some() {
                        return Vec::new();
                    }
                    width = Some(item);
                } else if is_border_style(k) {
                    if style.is_some() {
                        return Vec::new();
                    }
                    style = Some(item);
                } else {
                    return Vec::new();
                }
            }
            _ => return Vec::new(),
        }
    }
    // Omitted components reset to initial values.
    let width = width.unwrap_or(CssValue::Keyword("medium".into()));
    let style = style.unwrap_or(CssValue::Keyword("none".into()));
    let color = color.unwrap_or(CssValue::Keyword("currentcolor".into()));
    let mut out = Vec::new();
    let push_side = |prefix: &str, out: &mut Vec<(String, CssValue)>| {
        out.push(decl(&format!("{prefix}-width"), width.clone()));
        out.push(decl(&format!("{prefix}-style"), style.clone()));
        out.push(decl(&format!("{prefix}-color"), color.clone()));
    };
    for prefix in prefixes {
        if *prefix == "border" {
            // All four sides: one longhand per side per component.
            for side in ["top", "right", "bottom", "left"] {
                push_side(&format!("border-{side}"), &mut out);
            }
        } else {
            push_side(prefix, &mut out);
        }
    }
    out
}

/// `column-rule: <column-rule-width> || <column-rule-style> ||
/// <column-rule-color>` in any order, 1-3 components (CSS Multicol 1
/// §4). Unlike `border`, there are no sides: the three components expand
/// to `column-rule-width`, `column-rule-style`, and `column-rule-color`
/// directly. Component classification matches `expand_border_sides`
/// (and parse.rs's `expand_column_rule`); omitted components reset to
/// their initial values (`medium`, `none`, `currentcolor`).
/// Unclassifiable items (unknown keywords) drop the whole declaration.
fn expand_column_rule_value(value: &CssValue) -> Vec<(String, CssValue)> {
    let mut width = None;
    let mut style = None;
    let mut color = None;
    for item in as_list(value) {
        match item {
            // Unitless zero is a valid `<length>` (CSS-wide rule), and a
            // `calc()` width evaluates in the cascade after expansion.
            CssValue::Length(_, _) | CssValue::Math(_) | CssValue::MathDeferred { .. } => {
                if width.is_some() {
                    return Vec::new();
                }
                width = Some(item);
            }
            CssValue::Number(0.0) => {
                if width.is_some() {
                    return Vec::new();
                }
                width = Some(item);
            }
            CssValue::Color(_) => {
                if color.is_some() {
                    return Vec::new();
                }
                color = Some(item);
            }
            CssValue::Keyword(ref kw) => {
                let k = kw.as_str();
                if k == "currentcolor" {
                    if color.is_some() {
                        return Vec::new();
                    }
                    color = Some(item);
                } else if is_border_width(k) {
                    if width.is_some() {
                        return Vec::new();
                    }
                    width = Some(item);
                } else if is_border_style(k) {
                    if style.is_some() {
                        return Vec::new();
                    }
                    style = Some(item);
                } else {
                    return Vec::new();
                }
            }
            _ => return Vec::new(),
        }
    }
    // Omitted components reset to initial values.
    let width = width.unwrap_or(CssValue::Keyword("medium".into()));
    let style = style.unwrap_or(CssValue::Keyword("none".into()));
    let color = color.unwrap_or(CssValue::Keyword("currentcolor".into()));
    vec![
        decl("column-rule-width", width),
        decl("column-rule-style", style),
        decl("column-rule-color", color),
    ]
}

/// `font: [style] [variant] [weight] size [/ line-height] family`, from
/// the substituted value form. All six longhands are always emitted;
/// omitted ones reset to `normal`.
///
/// The `/` between size and line-height has no [`CssValue`] form, so the
/// parser stores a `/`-separated deferred value as
/// `List([List(before), List(after)])` (see `parse_slash_aware_value` in
/// parse.rs): `before` holds `[style] [variant] [weight] size`, `after`
/// holds `line-height family...`. A `var()`-substituted comma family can
/// also nest this way, so the marker is only honored when `before` is
/// exactly `[style?] [variant?] [weight?] size` and `after` starts with a
/// valid line-height; anything else drops the declaration. The `after`
/// side's comma groups flatten one level (a substituted `--f: 12px/1.5
/// Arial, sans-serif` keeps both families). Line-height accepts a number,
/// length, percentage, keyword, or `calc()`-family `Math`, matching the
/// eager token parser (`parse_font_shorthand` in parse.rs): the eager
/// and deferred paths must agree. Family groups that are ident runs
/// (substituted multi-word names like `--f: Times New Roman`) join with
/// spaces into one family name.
/// Parsed `[style|variant|weight]* size` prefix of a `font` value: the
/// three optional longhands, the size, and the index of the first item
/// after the size.
type FontPrefix = (
    Option<CssValue>,
    Option<CssValue>,
    Option<CssValue>,
    CssValue,
    usize,
);

/// Parse `[style|variant|weight]* size` from the start of `items`.
/// Returns the prefix (`None` when invalid).
fn parse_font_prefix(items: &[CssValue]) -> Option<FontPrefix> {
    let mut i = 0;
    let mut style = None;
    let mut variant = None;
    let mut weight = None;
    while i < items.len() {
        match &items[i] {
            CssValue::Keyword(kw) => {
                let k = kw.to_ascii_lowercase();
                match k.as_str() {
                    "italic" | "oblique" => {
                        if style.is_some() {
                            return None;
                        }
                        style = Some(CssValue::Keyword(k.into()));
                    }
                    "small-caps" => {
                        if variant.is_some() {
                            return None;
                        }
                        variant = Some(CssValue::Keyword(k.into()));
                    }
                    "bold" | "bolder" | "lighter" => {
                        if weight.is_some() {
                            return None;
                        }
                        weight = Some(CssValue::Keyword(k.into()));
                    }
                    "normal" => {}
                    _ => break,
                }
                i += 1;
            }
            CssValue::Number(n)
                if n.fract() == 0.0
                    && matches!(
                        *n as i32,
                        100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900
                    ) =>
            {
                if weight.is_some() {
                    return None;
                }
                weight = Some(CssValue::Number(*n));
                i += 1;
            }
            _ => break,
        }
    }
    let size = match items.get(i) {
        Some(CssValue::Length(n, unit)) => CssValue::Length(*n, unit.clone()),
        Some(CssValue::Percentage(p)) => CssValue::Percentage(*p),
        // `calc()` survives `var()` substitution as `Math` and evaluates
        // in the cascade after expansion (matching the eager token-level
        // `font` parser, which admits math functions for size).
        Some(CssValue::Math(e)) => CssValue::Math(e.clone()),
        _ => return None,
    };
    i += 1;
    Some((style, variant, weight, size, i))
}

/// Build the `font-family` longhand from the items after `font-size`
/// (and optional `/ line-height`). Each item is either a loose word
/// (from a group-less substitution or a literal) or one comma group
/// (`CssValue::List`, from a substituted comma-separated variable).
/// Loose ident words merge with adjacent groups: substitution is
/// token-wise, so `font: 12px a var(--f)` with `--f: b, c` means
/// `font: 12px a b, c` (families "a b" and "c"). A `String` is always its
/// own family; ident words before one are invalid (`a "b"`), matching
/// the eager token parser.
///
/// Split one comma group's items on nested comma-group lists spliced in
/// by `var()` substitution. Token-wise, `font: 12px var(--f), serif`
/// with `--f: Arial, sans-serif` is `font: 12px Arial, sans-serif,
/// serif`: the substituted comma groups split the enclosing group, so
/// `[12px, List([Arial]), List([sans-serif])]` becomes the two groups
/// `[[12px, Arial], [sans-serif]]`. A nested list merges its items into
/// the current group and then opens a new group at the comma boundary,
/// exactly like a comma in the token stream would.
fn split_spliced_groups(items: &[CssValue]) -> Vec<Vec<CssValue>> {
    let mut groups: Vec<Vec<CssValue>> = vec![Vec::new()];
    for item in items {
        match item {
            CssValue::List(sub) => {
                groups
                    .last_mut()
                    .expect("at least one group")
                    .extend(sub.iter().cloned());
                groups.push(Vec::new());
            }
            other => groups
                .last_mut()
                .expect("at least one group")
                .push(other.clone()),
        }
    }
    // Drop the trailing empty group opened after the last splice; an
    // empty group anywhere else (or an empty splice) is invalid and left
    // for `font_families` to reject.
    if groups.last().is_some_and(Vec::is_empty) {
        groups.pop();
    }
    groups
}
fn font_families(rest: &[CssValue]) -> Option<CssValue> {
    let mut families: Vec<String> = Vec::new();
    // Ident words of the family currently being built.
    let mut pending: Vec<String> = Vec::new();
    // Previous comma group's words, held back so a following ident run
    // merges into it (an ident run between two groups joins both).
    let mut deferred: Option<Vec<String>> = None;
    // Whether the previous loose item completed a family: a quoted string
    // may not follow it without an intervening comma (`"a" "b"` is
    // invalid), mirroring the eager token parser.
    let mut last_was_family = false;
    // Push one comma group's words, merging an adjacent ident run.
    let push_group = |words: Vec<String>,
                      families: &mut Vec<String>,
                      pending: &mut Vec<String>,
                      deferred: &mut Option<Vec<String>>| {
        if pending.is_empty() {
            if let Some(d) = deferred.take() {
                families.push(d.join(" "));
            }
            *deferred = Some(words);
        } else if let Some(d) = deferred.as_mut() {
            // Ident run between two groups: join all three.
            d.extend(std::mem::take(pending));
            d.extend(words);
        } else {
            let mut w = std::mem::take(pending);
            w.extend(words);
            *deferred = Some(w);
        }
    };
    for item in rest {
        match item {
            CssValue::Keyword(k) => {
                pending.push(k.as_str().to_owned());
                last_was_family = false;
            }
            CssValue::String(s) => {
                // A quoted family must stand alone or follow a comma:
                // `a "b"`, `"a" "b"`, and a string after a comma group
                // are all invalid, exactly like the eager token parser.
                if !pending.is_empty()
                    || last_was_family
                    || !families.is_empty()
                    || deferred.is_some()
                {
                    return None;
                }
                families.push(s.clone());
                last_was_family = true;
            }
            CssValue::List(group) => {
                if group.is_empty() {
                    return None;
                }
                // Snapshot before the group runs: a comma-splice group is
                // adjacent to the preceding loose item (no comma between
                // them), so a loose top-level `"A"` followed by a group
                // containing `"B"` is `"A" "B"` adjacency — invalid like
                // the eager parser. A group after a genuine comma (`"A",
                // "B"`) always enters with the flag clear, so the
                // comma-boundary reset below still stands.
                let entry_last_was_family = last_was_family;
                let mut words: Vec<String> = Vec::new();
                // A second quoted string in one group (`"a" "b"` with no
                // intervening comma) is invalid, mirroring the top-level
                // `last_was_family` rejection; the comma-boundary reset of
                // `last_was_family` still happens after the group.
                let mut saw_string = false;
                for w in group {
                    match w {
                        CssValue::Keyword(k) => words.push(k.as_str().to_owned()),
                        CssValue::String(s) => {
                            // A string is a hard family boundary even
                            // inside a group (`"a" b` is two families);
                            // ident words before it are invalid, as is a
                            // string adjacent to a loose top-level string
                            // (`"A"` then a group holding `"B"`).
                            if !words.is_empty()
                                || !pending.is_empty()
                                || saw_string
                                || entry_last_was_family
                            {
                                return None;
                            }
                            if let Some(d) = deferred.take() {
                                families.push(d.join(" "));
                            }
                            families.push(s.clone());
                            saw_string = true;
                        }
                        _ => return None,
                    }
                }
                if !words.is_empty() {
                    push_group(words, &mut families, &mut pending, &mut deferred);
                }
                // A comma group is a comma boundary: a quoted family may
                // follow it (`"a", "b"`), like the eager parser.
                last_was_family = false;
            }
            _ => return None,
        }
    }
    if !pending.is_empty() {
        match deferred.as_mut() {
            Some(d) => {
                d.extend(pending);
                families.push(d.join(" "));
            }
            None => families.push(pending.join(" ")),
        }
    } else if let Some(d) = deferred.take() {
        families.push(d.join(" "));
    }
    if families.is_empty() {
        return None; // font requires a family
    }
    if families.len() == 1 {
        Some(CssValue::String(families.pop().expect("one family")))
    } else {
        Some(CssValue::List(
            families.into_iter().map(CssValue::String).collect(),
        ))
    }
}

/// Emit the six `font` longhands from parsed components. Omitted
/// style/variant/weight reset to `normal`.
fn font_longhands(
    style: Option<CssValue>,
    variant: Option<CssValue>,
    weight: Option<CssValue>,
    size: CssValue,
    line_height: CssValue,
    family: CssValue,
) -> Vec<(String, CssValue)> {
    let normal = || CssValue::Keyword("normal".into());
    vec![
        decl("font-style", style.unwrap_or_else(&normal)),
        decl("font-variant", variant.unwrap_or_else(&normal)),
        decl("font-weight", weight.unwrap_or_else(&normal)),
        decl("font-size", size),
        decl("line-height", line_height),
        decl("font-family", family),
    ]
}

/// Expand the `/`-separated `font` form: `before` holds
/// `[style|variant|weight]* size` and must end at the size; `after`'s
/// first word is the line-height and the rest are families. The
/// `/`-side may hold comma groups from substitution (e.g.
/// `--f: 12px/1.5 Arial, sans-serif`). Returns empty when invalid.
fn expand_font_slash(before: &[CssValue], after: &[CssValue]) -> Vec<(String, CssValue)> {
    let Some((style, variant, weight, size, i)) = parse_font_prefix(before) else {
        return Vec::new();
    };
    // In the `/` form `before` must end at the size: anything after it is
    // invalid (this also rejects comma-group values that merely look like
    // the marker, e.g. a substituted `font: var(--a), serif`).
    if i != before.len() {
        return Vec::new();
    }
    // Normalize the `/`-side to comma groups so the first word is the
    // line-height regardless of grouping.
    let groups: Vec<Vec<CssValue>> = after.iter().map(as_list).collect();
    // An empty group is a leading or doubled comma on the `/`-side,
    // kept visible by the font-path split (`split_top_slash_font`);
    // invalid, like the eager parser.
    if groups.is_empty() || groups.iter().any(Vec::is_empty) {
        return Vec::new();
    }
    // A comma-group `List` in the first `/`-side slot that holds only
    // the line-height while more items follow is a leading comma in
    // the family list: `font: 12px/var(--g)` with `--g: 1.5, serif`
    // substitutes token-wise to `font: 12px/1.5, serif`, which the
    // eager parser drops. The comma was never top-level before
    // substitution, so the parse-time guard in `split_top_slash_font`
    // cannot see it; mirror that guard here on the substituted shape
    // (`after = [List([1.5]), List([serif])]`). Shapes with a family
    // word after the line-height in the first group
    // (`after[0] = List([1.5, Arial])`) or a flat line-height followed
    // by comma groups (`after = [1.5, List([Arial]), List([serif])]`)
    // stay valid.
    if matches!(after[0], CssValue::List(_)) && as_list(&after[0]).len() < 2 && after.len() > 1 {
        return Vec::new();
    }
    // Line-height accepts what the eager parser accepts: a number,
    // length, percentage, keyword, or math function. Keywords are
    // lowercased to match the eager token parser (`parse_font_shorthand`
    // lowercases its line-height ident); the case-preserving substitution
    // parse may hand one over with authored case.
    let lh = match &groups[0][0] {
        CssValue::Number(n) => CssValue::Number(*n),
        CssValue::Length(n, unit) => CssValue::Length(*n, unit.clone()),
        CssValue::Percentage(p) => CssValue::Percentage(*p),
        CssValue::Keyword(k) => CssValue::Keyword(k.to_ascii_lowercase().into()),
        // `calc()` survives `var()` substitution as `Math` and
        // evaluates in the cascade after expansion.
        CssValue::Math(e) => CssValue::Math(e.clone()),
        _ => return Vec::new(),
    };
    // Families: the rest of the first group, then the remaining items
    // off the `/`-side. Flat words after the line-height accumulate
    // into one ident run (a multi-word family stays one family);
    // nested `List`s are comma boundaries and pass through. Nested
    // comma-group splices from `var()` substitution split their
    // enclosing group (see [`split_spliced_groups`]).
    let mut fitems: Vec<CssValue> = Vec::new();
    if groups[0].len() > 1 {
        fitems.extend(
            split_spliced_groups(&groups[0][1..])
                .into_iter()
                .map(CssValue::List),
        );
    }
    let mut run: Vec<CssValue> = Vec::new();
    for item in &after[1..] {
        match item {
            CssValue::List(sub) => {
                // Comma boundary: flush the pending ident run first.
                if !run.is_empty() {
                    fitems.extend(split_spliced_groups(&run).into_iter().map(CssValue::List));
                    run.clear();
                }
                fitems.extend(split_spliced_groups(sub).into_iter().map(CssValue::List));
            }
            other => run.push(other.clone()),
        }
    }
    if !run.is_empty() {
        fitems.extend(split_spliced_groups(&run).into_iter().map(CssValue::List));
    }
    let Some(family) = font_families(&fitems) else {
        return Vec::new(); // font requires a family
    };
    font_longhands(style, variant, weight, size, lh, family)
}

/// Expand a comma-separated `font` substitution standing alone
/// (`font: var(--f)` with `--f: 12px arial, sans-serif`): the first group
/// holds `[style|variant|weight]* size`, the rest are families. Returns
/// empty when invalid.
fn expand_font_comma(groups: &[Vec<CssValue>]) -> Vec<(String, CssValue)> {
    if groups.is_empty() {
        return Vec::new();
    }
    let Some((style, variant, weight, size, i)) = parse_font_prefix(&groups[0]) else {
        return Vec::new();
    };
    // The first family must start in group 0: a comma right after the
    // size (`--f: 12px, serif`) is a leading comma in the family list,
    // invalid per css-values-4 §2 (like `example(,second)`), matching
    // the eager parser's leading-comma rejection.
    if i == groups[0].len() && groups.len() > 1 {
        return Vec::new();
    }
    // Nested comma-group splices from `var()` substitution split their
    // enclosing group (see [`split_spliced_groups`]).
    let mut fitems: Vec<CssValue> = Vec::new();
    if i < groups[0].len() {
        fitems.extend(
            split_spliced_groups(&groups[0][i..])
                .into_iter()
                .map(CssValue::List),
        );
    }
    for g in &groups[1..] {
        fitems.extend(split_spliced_groups(g).into_iter().map(CssValue::List));
    }
    let Some(family) = font_families(&fitems) else {
        return Vec::new(); // font requires a family
    };
    let normal = CssValue::Keyword("normal".into());
    font_longhands(style, variant, weight, size, normal, family)
}

fn expand_font_value(value: &CssValue) -> Vec<(String, CssValue)> {
    let items = as_list(value);
    // System font keywords (`font: var(--x)` with `--x: caption`): per
    // CSS Fonts 3 §3.2 the keyword must be the *sole* value and computes
    // to the UA's font for that purpose (`system_font_values` in
    // value.rs). This mirrors the eager check in parse.rs's
    // `parse_font_shorthand`; keep them in sync. Anything alongside the
    // keyword falls through to the normal path, which rejects it.
    if items.len() == 1
        && let CssValue::Keyword(kw) = &items[0]
        && let Some(vals) = system_font_values(&kw.to_ascii_lowercase())
    {
        let [style, variant, weight, size, line_height, family] = vals;
        return font_longhands(
            Some(style),
            Some(variant),
            Some(weight),
            size,
            line_height,
            family,
        );
    }
    // Comma groups: every item is a nested list, either from a
    // substituted comma-separated variable or from the parse-time comma
    // split of the declaration itself (`font: var(--f), serif` keeps each
    // comma-separated part distinct). The two-group shape is ambiguous
    // with the `/` marker, so the slash reading is tried first (a
    // substituted `--f: 12px arial, sans-serif` nests the same way); any
    // other group count is commas only.
    if !items.is_empty() && items.iter().all(|g| matches!(g, CssValue::List(_))) {
        let groups: Vec<Vec<CssValue>> = items.iter().map(|g| as_list(g).to_vec()).collect();
        if groups.len() == 2 {
            let r = expand_font_slash(&groups[0], &groups[1]);
            if !r.is_empty() {
                return r;
            }
        }
        // Nested `/` marker: `font: var(--f), serif` with
        // `--f: 12px/1.5 Arial` substitutes to
        // `[[[12px], [1.5, Arial]], [serif]]`, hiding the slash
        // marker one level down where the top-level slash attempt
        // cannot see it. The marker is honored only in comma group 0
        // (the size portion): once past the first top-level comma the
        // parser is in family territory and a `/` there must not be
        // read as the size/line-height divider. Unwrap the marker and
        // run the slash path with the remaining groups as trailing
        // families; on failure fall through to the comma expansion
        // unchanged.
        if let Some(first) = groups.first()
            && let [CssValue::List(before), CssValue::List(after)] = first.as_slice()
            // No leading comma: when the `/`-side carries only the
            // line-height (`--f: 12px/1.5`), a sibling comma group
            // (`font: var(--f), serif`) would separate it from the first
            // family with no family words before the comma — invalid
            // like the eager parser. Fall through to the comma
            // expansion, which rejects it the same way.
            && !(after.len() == 1 && groups.len() > 1)
        {
            let mut slash_after: Vec<CssValue> = after.clone();
            slash_after.extend(groups[1..].iter().map(|g| CssValue::List(g.clone())));
            let r = expand_font_slash(before, &slash_after);
            if !r.is_empty() {
                return r;
            }
        }
        return expand_font_comma(&groups);
    }
    let Some((style, variant, weight, size, i)) = parse_font_prefix(&items) else {
        return Vec::new();
    };
    if i >= items.len() {
        return Vec::new(); // font requires a family
    }
    let Some(family) = font_families(&items[i..]) else {
        return Vec::new();
    };
    let normal = CssValue::Keyword("normal".into());
    font_longhands(style, variant, weight, size, normal, family)
}

/// `background: <color> <image> <repeat> <attachment> <position>` in any order.
/// All five longhands are always emitted; omitted components reset to
/// initial values (`transparent`, `none`, `repeat`, `scroll`, `0% 0%`).
fn expand_background(value: CssValue) -> Vec<(String, CssValue)> {
    let mut color = None;
    let mut image = None;
    let mut repeat = None;
    let mut attachment = None;
    let mut position: Vec<CssValue> = Vec::new();
    for item in as_list(&value) {
        match item {
            CssValue::Color(_) => {
                if color.is_some() {
                    return Vec::new();
                }
                color = Some(item);
            }
            CssValue::Url(_) | CssValue::Gradient { .. } => {
                if image.is_some() {
                    return Vec::new();
                }
                image = Some(item);
            }
            CssValue::Length(_, _) | CssValue::Percentage(_) => position.push(item),
            CssValue::Keyword(kw) => {
                let k = kw.as_str();
                if k == "currentcolor" {
                    if color.is_some() {
                        return Vec::new();
                    }
                    color = Some(CssValue::Keyword(kw));
                } else if matches!(
                    k,
                    "repeat" | "no-repeat" | "repeat-x" | "repeat-y" | "space" | "round"
                ) {
                    if repeat.is_some() {
                        return Vec::new();
                    }
                    repeat = Some(CssValue::Keyword(kw));
                } else if matches!(k, "scroll" | "fixed" | "local") {
                    if attachment.is_some() {
                        return Vec::new();
                    }
                    attachment = Some(CssValue::Keyword(kw));
                } else if matches!(k, "left" | "right" | "top" | "bottom" | "center") {
                    position.push(CssValue::Keyword(kw));
                } else if k == "none" {
                    if image.is_some() {
                        return Vec::new();
                    }
                    image = Some(CssValue::Keyword(kw));
                } else {
                    return Vec::new();
                }
            }
            _ => return Vec::new(),
        }
    }
    if position.len() > 2 {
        return Vec::new();
    }
    // Every longhand is emitted; omitted ones reset to initial values.
    let color = color.unwrap_or(CssValue::Color(Color {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    })); // transparent
    let image = image.unwrap_or(CssValue::Keyword("none".into()));
    let repeat = repeat.unwrap_or(CssValue::Keyword("repeat".into()));
    let attachment = attachment.unwrap_or(CssValue::Keyword("scroll".into()));
    let position = if position.is_empty() {
        CssValue::List(vec![CssValue::Percentage(0.0), CssValue::Percentage(0.0)])
    } else if position.len() == 1 {
        position.pop().expect("one position")
    } else {
        CssValue::List(position)
    };
    vec![
        decl("background-color", color),
        decl("background-image", image),
        decl("background-repeat", repeat),
        decl("background-attachment", attachment),
        decl("background-position", position),
    ]
}

/// `list-style: <type> <position> <image>` in any order. All three
/// longhands are always emitted; omitted components reset to initial
/// values (`disc`, `outside`, `none`).
fn expand_list_style(value: CssValue) -> Vec<(String, CssValue)> {
    fn is_type(kw: &str) -> bool {
        matches!(
            kw,
            "disc"
                | "circle"
                | "square"
                | "decimal"
                | "decimal-leading-zero"
                | "lower-roman"
                | "upper-roman"
                | "lower-greek"
                | "lower-latin"
                | "upper-latin"
                | "armenian"
                | "georgian"
                | "lower-alpha"
                | "upper-alpha"
                | "none"
        )
    }
    let mut style_type = None;
    let mut position = None;
    let mut image = None;
    for item in as_list(&value) {
        match item {
            CssValue::Url(_) => {
                if image.is_some() {
                    return Vec::new();
                }
                image = Some(item);
            }
            CssValue::Keyword(kw) => {
                let k = kw.as_str();
                if is_type(k) {
                    if style_type.is_some() {
                        return Vec::new();
                    }
                    style_type = Some(CssValue::Keyword(kw));
                } else if matches!(k, "inside" | "outside") {
                    if position.is_some() {
                        return Vec::new();
                    }
                    position = Some(CssValue::Keyword(kw));
                } else if k == "none" {
                    if image.is_some() {
                        return Vec::new();
                    }
                    image = Some(CssValue::Keyword(kw));
                } else {
                    return Vec::new();
                }
            }
            _ => return Vec::new(),
        }
    }
    // Every longhand is emitted; omitted ones reset to initial values.
    vec![
        decl(
            "list-style-type",
            style_type.unwrap_or(CssValue::Keyword("disc".into())),
        ),
        decl(
            "list-style-position",
            position.unwrap_or(CssValue::Keyword("outside".into())),
        ),
        decl(
            "list-style-image",
            image.unwrap_or(CssValue::Keyword("none".into())),
        ),
    ]
}

/// Cascade-side mirror of parse.rs's `expand_flex`: expands a substituted
/// `flex` shorthand value (`flex: none | [ <flex-grow> <flex-shrink>? ||
/// <flex-basis> ]`, CSS flexbox-1 §7.1.1) into its three longhands.
/// `flex: none` -> `0 0 auto`; a single number -> `<n> 1 0%`; a single
/// basis -> `1 1 <basis>`; otherwise an absent grow/shrink defaults to 1
/// and an absent basis to `0%`. Numbers are flex factors (a unitless zero
/// is a factor, never a basis); lengths/percentages and the
/// `auto`/`content`/`fit-content`/`min-content`/`max-content`/`stretch`
/// keywords are the basis (the full `flex-basis` longhand set, CSS
/// flexbox-1 §7.1 / CSS Sizing 3 §5). `calc()` components are
/// rejected — their type is not known at parse time. Unclassifiable
/// items expand to nothing (invalid at computed-value time). Keep in sync
/// with parse.rs's `expand_flex`.
fn expand_flex_value(value: &CssValue) -> Vec<(String, CssValue)> {
    let items = as_list(value);
    // `flex: none` -> `0 0 auto`.
    if items.len() == 1
        && let CssValue::Keyword(k) = &items[0]
        && k.eq_ignore_ascii_case("none")
    {
        return vec![
            decl("flex-grow", CssValue::Number(0.0)),
            decl("flex-shrink", CssValue::Number(0.0)),
            decl("flex-basis", CssValue::Keyword("auto".into())),
        ];
    }
    /// A `<number>`: a flex factor (`flex-grow` first, then `flex-shrink`).
    fn is_factor(v: &CssValue) -> bool {
        matches!(v, CssValue::Number(_))
    }
    /// A `<flex-basis>`: a width, or the `auto`/`content`/`fit-content`/
    /// `min-content`/`max-content`/`stretch` keywords — the same set
    /// the `flex-basis` longhand accepts (see `is_valid_value`).
    fn is_basis(v: &CssValue) -> bool {
        matches!(v, CssValue::Length(_, _) | CssValue::Percentage(_))
            || matches!(v, CssValue::Number(n) if *n == 0.0)
            || matches!(v, CssValue::Keyword(k) if {
                let k = k.to_ascii_lowercase();
                k == "auto"
                    || k == "content"
                    || k == "fit-content"
                    || k == "min-content"
                    || k == "max-content"
                    || k == "stretch"
            })
    }
    let mut grow: Option<CssValue> = None;
    let mut shrink: Option<CssValue> = None;
    let mut basis: Option<CssValue> = None;
    for item in &items {
        if is_factor(item) {
            if grow.is_none() {
                grow = Some(item.clone());
            } else if shrink.is_none() {
                shrink = Some(item.clone());
            } else {
                return Vec::new(); // three factors: invalid
            }
        } else if is_basis(item) {
            if basis.is_none() {
                basis = Some(item.clone());
            } else {
                return Vec::new(); // two bases: invalid
            }
        } else {
            return Vec::new(); // unclassifiable component: invalid
        }
    }
    if items.is_empty() || (grow.is_none() && basis.is_none()) {
        return Vec::new();
    }
    vec![
        decl("flex-grow", grow.unwrap_or(CssValue::Number(1.0))),
        decl("flex-shrink", shrink.unwrap_or(CssValue::Number(1.0))),
        decl("flex-basis", basis.unwrap_or(CssValue::Percentage(0.0))),
    ]
}
impl<'a> Styler<'a> {
    /// Read a memo slot: `None` when unknown (not computed yet, or the
    /// entry is from a previous style-pass epoch), `Some(v)` when
    /// computed (`v` itself may be `None` = unset). Sparse: a known
    /// slot with no entry reads as the shared initial value (see
    /// [`MemoSet`]).
    fn memo_get(&self, node: NodeId, idx: u8) -> Option<Option<Arc<CssValue>>> {
        let guard = self.resolved.borrow();
        let slots = guard.probe(node, self.epoch.get())?;
        slots.get(idx)
    }

    /// Write a memo slot, installing the node's [`MemoSet`] (no heap
    /// allocation until a deviating value is recorded) on first use. A
    /// stale entry from a previous epoch is reset and retagged, never
    /// trusted.
    fn memo_set(&self, node: NodeId, idx: u8, value: Option<Arc<CssValue>>) {
        let epoch = self.epoch.get();
        let mut resolved = self.resolved.borrow_mut();
        let slots = resolved.slot_or_insert_with(node, epoch, MemoSet::default);
        slots.set(idx, value);
    }

    /// Probe the per-node custom-property memo for `--name` (interned as
    /// `id`) on `node`: the substituted value, or `None` on a miss
    /// (absent, foreign, or stale entry). The probe's clone is the only
    /// copy: the slash/comma payloads are `Rc`-shared (pointer bumps),
    /// and the plain value moves out at the hit site with no second
    /// clone.
    fn var_memo_probe(&self, node: NodeId, id: u32, preserve_case: bool) -> Option<MemoVar> {
        let epoch = self.epoch.get();
        let var_memo = self.var_memo.borrow();
        let inner = var_memo.probe(node, epoch)?;
        let (e, v) = inner.get(&(id, preserve_case))?;
        (*e == epoch).then(|| v.clone())
    }

    /// Memoize a custom-property substitution for `node`: the inner
    /// `(name id, case)` map is created on first use per node per epoch
    /// (a stale map is dropped, never trusted).
    fn var_memo_insert(&self, node: NodeId, id: u32, preserve_case: bool, value: MemoVar) {
        let epoch = self.epoch.get();
        let mut var_memo = self.var_memo.borrow_mut();
        var_memo
            .slot_or_insert_with(node, epoch, FxHashMap::default)
            .insert((id, preserve_case), (epoch, value));
    }

    fn resolved_prop(&self, node: NodeId, idx: u8, prop: &str) -> Option<Arc<CssValue>> {
        if let Some(r) = self.memo_get(node, idx) {
            return r;
        }
        // Cold path (e.g. layout querying one property, or an ancestor
        // reached through inheritance): fetch this node's cascade winners
        // and environment, then resolve. The ancestor climb inside is
        // iterative (`finish_prop`), so a cold `style_for` on a 10k-deep
        // node cannot overflow the call stack (B3).
        let raw = self.cascade_expanded(node);
        let env = self.custom_env(node);
        let (result, _) = self.finish_prop(node, idx, prop, &raw, &env);
        self.memo_set(node, idx, result.clone());
        result
    }

    /// Drive one property to its computed value: run [`Styler::compute_prop`]
    /// for the specified value, then climb ancestors iteratively for
    /// [`PropStep::Climb`] (explicit `inherit`, or unspecified values on
    /// inherited properties). The climb uses an explicit frame list, not
    /// recursion, so a cold `style_for` on a node 10k deep cannot overflow
    /// the call stack (B3) — whether the climb comes from unspecified
    /// inherited properties or explicit `inherit` keywords at every level.
    ///
    /// Used by both the warm root-first `compute()` path and the cold
    /// `resolved_prop` path. Every node on the climb memoizes the value it
    /// computes, exactly as the old per-frame recursion did.
    ///
    /// The two `overflow` axes are the exception: their computed values
    /// depend on each other (see [`fixup_overflow_axes`]), so a single
    /// axis is never finished — or memoized — on its own; those calls
    /// route to [`Styler::finish_overflow_prop`], which finishes the
    /// pair together.
    fn finish_prop(
        &self,
        node: NodeId,
        idx: u8,
        prop: &str,
        winners: &[(u8, Cow<'a, CssValue>, u32)],
        env: &FxHashMap<String, CssValue>,
    ) -> (Option<Arc<CssValue>>, bool) {
        if idx == props::OVERFLOW_X || idx == props::OVERFLOW_Y {
            return self.finish_overflow_prop(node, idx);
        }
        self.finish_prop_raw(node, idx, prop, winners, env)
    }

    /// Finish both `overflow` axes of `node` together and return the
    /// adjusted value of the requested axis (B10).
    ///
    /// `winners`/`env` are not threaded through: every node on the
    /// climb re-fetches its own memoized winners and environment,
    /// which for the starting node are the very values its callers
    /// (`compute`, `resolved_prop`) pass to `finish_prop` — those
    /// callers obtained them from the same memoized accessors.
    ///
    /// Phase 1 walks up iteratively (B3: no recursion), computing both
    /// axes per node until both are `Done`, or an ancestor with both
    /// axes already memoized — memoized overflow values are always the
    /// adjusted pair, because every memo write for these axes goes
    /// through this function, the `compute` batch write (which runs
    /// `fixup_overflow_axes` first), or style sharing's adopt of a
    /// finished style. Phase 2 walks back down: each node's raw pair
    /// takes its climbing components from the adjusted pair above it,
    /// is adjusted as a pair via [`adjust_overflow_pair`], and is
    /// memoized (both axes) before the next node down consumes it.
    /// An explicit `inherit` whose adjusted component above is unset
    /// falls back to the initial value, mirroring `finish_prop_raw`.
    fn finish_overflow_prop(&self, node: NodeId, idx: u8) -> (Option<Arc<CssValue>>, bool) {
        let (x_idx, y_idx) = (props::OVERFLOW_X, props::OVERFLOW_Y);
        /// One axis of one climb frame: a finished raw value, or a
        /// climb with its explicit-`inherit` flag.
        enum Axis {
            Done(Option<CssValue>),
            Climb { parent: NodeId, explicit: bool },
        }
        struct Frame {
            node: NodeId,
            x: Axis,
            y: Axis,
            x_invalid: bool,
            y_invalid: bool,
        }
        fn axis_of(step: PropStep) -> (Axis, bool) {
            match step {
                PropStep::Done { value, invalid } => (Axis::Done(value), invalid),
                PropStep::Climb { parent, explicit } => (Axis::Climb { parent, explicit }, false),
            }
        }
        // Phase 1: walk up, collecting one frame per node.
        let mut frames: Vec<Frame> = Vec::new();
        let mut n = node;
        // The adjusted pair above the top frame: only consulted when
        // the top frame climbs and a memoized ancestor ended the walk.
        let mut above: (Option<Arc<CssValue>>, Option<Arc<CssValue>>) = (None, None);
        loop {
            // A memoized ancestor with both axes known supplies the
            // pair the top frame climbs to; it is already adjusted.
            if n != node
                && let (Some(x), Some(y)) = (self.memo_get(n, x_idx), self.memo_get(n, y_idx))
            {
                above = (x, y);
                break;
            }
            let raw = self.cascade_expanded(n);
            let env = self.custom_env(n);
            let (x, x_invalid) =
                axis_of(self.compute_prop(n, "overflow-x", winner_at(&raw, x_idx), &env));
            let (y, y_invalid) =
                axis_of(self.compute_prop(n, "overflow-y", winner_at(&raw, y_idx), &env));
            // Both climbs target the same direct parent (every
            // `compute_prop` climb does), so either one names it.
            let parent = match (&x, &y) {
                (Axis::Climb { parent, .. }, _) | (_, Axis::Climb { parent, .. }) => Some(*parent),
                _ => None,
            };
            let done = parent.is_none() && matches!(x, Axis::Done(_)) && matches!(y, Axis::Done(_));
            frames.push(Frame {
                node: n,
                x,
                y,
                x_invalid,
                y_invalid,
            });
            if done {
                break;
            }
            // A climb with no parent cannot occur (`compute_prop`
            // only climbs to an existing parent); fall back to the
            // frames collected rather than looping forever if the
            // document invariant is ever broken.
            let Some(p) = parent else { break };
            n = p;
        }
        // The requested node's own invalid flag (ancestors' flags
        // belong to their frames, not to this return value).
        let invalid = match frames.first() {
            Some(f) if idx == x_idx => f.x_invalid,
            Some(f) => f.y_invalid,
            None => false,
        };
        // Phase 2: walk back down, adjusting and memoizing each pair.
        let mut current = above;
        for frame in frames.into_iter().rev() {
            let raw_x = match frame.x {
                Axis::Done(v) => v.map(Arc::new),
                Axis::Climb { explicit, .. } => {
                    if current.0.is_none() && explicit {
                        initial_rc(x_idx)
                    } else {
                        current.0.clone()
                    }
                }
            };
            let raw_y = match frame.y {
                Axis::Done(v) => v.map(Arc::new),
                Axis::Climb { explicit, .. } => {
                    if current.1.is_none() && explicit {
                        initial_rc(y_idx)
                    } else {
                        current.1.clone()
                    }
                }
            };
            let adjusted = adjust_overflow_pair(raw_x, raw_y);
            self.memo_set(frame.node, x_idx, adjusted.0.clone());
            self.memo_set(frame.node, y_idx, adjusted.1.clone());
            current = adjusted;
        }
        // The last frame processed in phase 2 is the starting node.
        let value = if idx == x_idx { current.0 } else { current.1 };
        (value, invalid)
    }

    /// The single-property driver behind [`Styler::finish_prop`] for
    /// every property except the `overflow` axes (which need the pair
    /// driver above).
    fn finish_prop_raw(
        &self,
        node: NodeId,
        idx: u8,
        prop: &str,
        winners: &[(u8, Cow<'a, CssValue>, u32)],
        env: &FxHashMap<String, CssValue>,
    ) -> (Option<Arc<CssValue>>, bool) {
        // Phase 1: walk up, collecting (node, explicit) frames until a
        // `Done` or a memoized ancestor.
        let mut frames: Vec<(NodeId, bool)> = Vec::new();
        let mut n = node;
        let mut step = self.compute_prop(n, prop, winner_at(winners, idx), env);
        // Whether this node's own specified value was invalid at
        // computed-value time (the logical-to-physical fold needs it);
        // later steps belong to ancestors.
        let invalid = matches!(step, PropStep::Done { invalid: true, .. });
        // Top of the climb: the node whose step finished the walk, its
        // value, and whether the edge into it was an explicit `inherit`.
        //
        // Values are `Arc`-shared: the memo, the frames on a climb, and
        // the caller's style array all alias the same allocation instead
        // of cloning `String`s per copy.
        let (top, mut value, top_explicit) = loop {
            match step {
                PropStep::Done { value, .. } => break (n, value.map(Arc::new), false),
                PropStep::Climb { parent, explicit } => {
                    // Warm shortcut: a memoized ancestor ends the climb.
                    if let Some(r) = self.memo_get(parent, idx) {
                        break (n, r, explicit);
                    }
                    frames.push((n, explicit));
                    n = parent;
                    let praw = self.cascade_expanded(n);
                    let penv = self.custom_env(n);
                    step = self.compute_prop(n, prop, winner_at(&praw, idx), &penv);
                }
            }
        };
        if frames.is_empty() {
            // No climb happened: writing the single frame directly
            // avoids the chain `Vec` the old code allocated per
            // (element, property).
            if value.is_none() && top_explicit {
                value = initial_rc(idx);
            }
            self.memo_set(top, idx, value.clone());
            return (value, invalid);
        }
        // Phase 2: walk back down. Each frame inherits the value above
        // it; an explicit `inherit` with nothing above falls back to the
        // initial value (the old `apply_css_wide` did
        // `parent_value().or_else(initial)`).
        let mut chain: Vec<(NodeId, bool)> = Vec::with_capacity(frames.len() + 1);
        chain.push((top, top_explicit));
        chain.extend(frames.into_iter().rev());
        for (m, explicit) in chain {
            if value.is_none() && explicit {
                value = initial_rc(idx);
            }
            self.memo_set(m, idx, value.clone());
        }
        (value, invalid)
    }

    /// The computed value of one property: cascade winner, `var()`
    /// substituted, CSS-wide keywords applied, grammar-validated, math
    /// evaluated — everything except the ancestor climb, which returns as
    /// [`PropStep::Climb`] for [`Styler::finish_prop`] to drive
    /// iteratively (B3).
    fn compute_prop(
        &self,
        node: NodeId,
        prop: &str,
        specified: Option<&CssValue>,
        env: &FxHashMap<String, CssValue>,
    ) -> PropStep {
        // Custom properties inherit through the merged environment and are
        // exempt from CSS-wide keywords and grammar validation (they hold
        // arbitrary token streams until substituted).
        if prop.starts_with("--") {
            // Guaranteed-invalid custom properties are unset, never
            // exposed literally.
            return PropStep::Done {
                value: env
                    .get(prop)
                    .filter(|v| !is_guaranteed_invalid(v))
                    .and_then(|v| self.resolve(node, v, env)),
                invalid: false,
            };
        }
        // Whether this node's specified value was invalid at
        // computed-value time. Threaded out on `PropStep::Done` so the
        // logical-to-physical fold can ignore invalid winners (see the
        // `PropStep::Done` docs). Set only where invalidity is
        // determined below — never on the `revert`-with-no-lower-origin
        // path, which is valid (`unset` semantics) and keeps its
        // cascade position.
        let mut invalid = false;
        let mut value = match specified {
            Some(v) => match self.resolve(node, v, env) {
                Some(r) => Some(r),
                // Failed `var()` substitution is invalid at
                // computed-value time.
                None => {
                    invalid = true;
                    None
                }
            },
            None => None,
        };
        // CSS-wide keywords, after substitution (a `var()` may itself
        // substitute to one). `inherit` climbs via `PropStep::Climb` so
        // the ancestor walk stays iterative. `revert`/`revert-layer`
        // substitute the lower-origin winner and re-run the check, so a
        // chained keyword (e.g. the lower origin declared `inherit`)
        // resolves instead of leaking literally. At most two iterations:
        // `revert_value` never returns `revert` itself.
        loop {
            let wide_kw = match &value {
                Some(CssValue::Keyword(kw))
                    if matches!(
                        kw.as_str(),
                        "inherit" | "initial" | "unset" | "revert" | "revert-layer"
                    ) =>
                {
                    Some(kw.clone())
                }
                _ => None,
            };
            let Some(kw) = wide_kw else { break };
            // Without a layer model every declaration is unlayered, so
            // `revert-layer` rolls back to the previous origin exactly
            // like `revert` (CSS Cascade 5 §8.2).
            if kw == "revert" || kw == "revert-layer" {
                match self.revert_value(node, prop, env) {
                    Some(v) => value = Some(v),
                    // No lower origin declares the property: `revert`
                    // behaves as `unset`.
                    None => match self.apply_css_wide(node, prop, "unset") {
                        PropStep::Done { value: v, .. } => {
                            value = v;
                            break;
                        }
                        climb => return climb,
                    },
                }
            } else {
                match self.apply_css_wide(node, prop, &kw) {
                    PropStep::Done { value: v, .. } => {
                        value = v;
                        break;
                    }
                    climb => return climb,
                }
            }
        }
        // Post-substitution grammar check: a substituted value that does
        // not match the property's grammar is invalid at computed-value
        // time, so the property is treated as unset. Comma groups from
        // `var()` substitution normalize to the `parse_value` shape first
        // (custom properties keep the nested form in the environment so
        // a later `font` substitution still sees family boundaries). The
        // valid `repeat(auto-fill|auto-fit, …)` keeps its dedicated
        // `Repeat` variant through normalization, so the grammar check
        // below tells it apart from the invalid bare `auto-fill <track>`
        // listing with no ordering-dependent pre-check.
        //
        // Commas never separate grid tracks (CSS Grid §7.1): reject a
        // `grid-template-columns`/`grid-template-rows` value containing
        // a nested list *before* normalization flattens single-item
        // comma groups away. Every nested list here is a comma group
        // from `var()` substitution (see `value_contains_nested_list`);
        // a top-level comma in an eager declaration is already rejected
        // at parse time (see `has_top_level_comma`).
        if let Some(v) = &value
            && matches!(prop, "grid-template-columns" | "grid-template-rows")
            && value_contains_nested_list(v)
        {
            value = None;
            invalid = true;
        }
        value = value.map(normalize_comma_groups);
        // `grid-auto-flow`'s two-keyword form must read as one keyword
        // (see `fold_grid_auto_flow`); fold before grammar validation so
        // the validator sees the folded shape on both validation passes.
        if prop == "grid-auto-flow" {
            value = value.map(fold_grid_auto_flow);
        }
        // `scroll-snap-type`: fold a default-strictness pair (`x
        // proximity`) to the canonical axis keyword before grammar
        // validation, like `grid-auto-flow` above.
        if prop == "scroll-snap-type" {
            value = value.map(fold_scroll_snap_type);
        }
        // `aspect-ratio`'s `/`-separated marker shape (from the parser or
        // `var()` substitution) folds to the compact `AspectRatio`
        // computed value. Math evaluates before the fold: `calc()` is a
        // valid `<number>` (css-sizing-4 §3.1), so `calc(2)` must fold to
        // `2 / 1` here rather than fail the grammar check below (the
        // `aspect-ratio` validation arm admits `Math` for the same
        // reason). Fold before grammar validation, like
        // `grid-auto-flow` above.
        if prop == "aspect-ratio" {
            let had_value = value.is_some();
            value = self.eval_math_deep(node, prop, value, env);
            invalid |= had_value && value.is_none();
            value = value.map(fold_aspect_ratio);
        }
        // `box-shadow`/`text-shadow`: a `var()` substitution (or any
        // other path that bypassed the token-level parser) arrives as
        // nested `List` shapes; fold valid shadow lists into `Shadows`
        // so the grammar check and paint see one shape. Fold before
        // grammar validation, like `grid-auto-flow` above.
        if prop == "box-shadow" || prop == "text-shadow" {
            value = value.map(|v| fold_shadow_value(v, prop == "box-shadow"));
        }
        // `content` takes a `<content-list>` whose adjacent `<string>`s
        // concatenate (`content: "a" "b"` computes to `"ab"`; see
        // `fold_content_value`). Fold before grammar validation, like
        // `grid-auto-flow` above.
        if prop == "content" {
            value = value.map(fold_content_value);
        }
        // `clip-path`: a `var()` substitution (or any other path that
        // bypassed the token-level parser) arrives in generic shapes;
        // fold valid shapes into `ClipPath` so the grammar check and
        // paint see one shape. Fold before grammar validation, like
        // `box-shadow` above.
        if prop == "clip-path" {
            value = value.map(fold_clip_path_value);
        }
        // css-sizing-4 legacy aliases: `-webkit-fill-available` /
        // `-moz-available` compute to the `stretch` keyword on sizing
        // properties only (Chrome rejects them for alignment, so
        // alignment arms never fold and keep rejecting the aliases).
        // Fold before grammar validation, like `grid-auto-flow` above,
        // so layout's `spec_len_of` sees the canonical `stretch`
        // keyword (the TWR table-6 fix relies on this).
        if STRETCH_SIZING_PROPS.contains(&prop) {
            value = value.map(fold_fill_available);
        }
        if let Some(v) = &value
            && !is_valid_for_property(prop, v)
        {
            value = None;
            invalid = true;
        }
        // Math evaluation, item-wise through arbitrarily nested lists
        // (A1): `calc()`/`min()`/`max()`/`clamp()` compute against real
        // font and viewport metrics here, where custom-property
        // environments are available for `var()` substitution. Percentage
        // math on layout-dependent properties defers to layout (A2);
        // invalid math is invalid at computed-value time.
        let had_value = value.is_some();
        value = self.eval_math_deep(node, prop, value, env);
        invalid |= had_value && value.is_none();
        // CSS Scroll Snap 1: `scroll-padding-*` is non-negative. A
        // *literal* negative was rejected by the grammar check above
        // (invalid at computed-value time), but a math expression that
        // evaluates negative clamps to zero instead (CSS Values 4
        // range checking; WPT `scroll-padding-computed` expects
        // `calc(10px - 0.5em)` to compute to `0px`, not `auto`).
        // `is_negative_length` only fires on unconditionally-negative
        // results (a percentage-bearing `MathDeferred` keeps its sign
        // unknown until layout and is left alone).
        if matches!(
            prop,
            "scroll-padding-top"
                | "scroll-padding-right"
                | "scroll-padding-bottom"
                | "scroll-padding-left"
                | "scroll-padding-inline-start"
                | "scroll-padding-inline-end"
                | "scroll-padding-block-start"
                | "scroll-padding-block-end"
        ) && let Some(v) = &value
            && is_negative_length(v)
        {
            value = Some(CssValue::Length(0.0, LengthUnit::Px));
        }
        // CSS Tables 3 / CSS Values 4 range checking: `border-spacing`
        // is non-negative. A *literal* negative was rejected by the
        // grammar check above, but a math expression that evaluates
        // negative clamps to zero instead of dropping the whole value.
        // Unlike the `scroll-padding-*` clamp above this maps
        // component-wise: `border-spacing` takes one or two lengths
        // (horizontal then vertical), so a negative component becomes
        // `0px` and a valid component in the pair form is kept.
        // `is_negative_length` only fires on unconditionally-negative
        // results (a percentage-bearing `MathDeferred` keeps its sign
        // unknown until layout and is left alone).
        if prop == "border-spacing" {
            value = value.map(|v| match v {
                CssValue::List(items) => CssValue::List(
                    items
                        .into_iter()
                        .map(|item| {
                            if is_negative_length(&item) {
                                CssValue::Length(0.0, LengthUnit::Px)
                            } else {
                                item
                            }
                        })
                        .collect(),
                ),
                v if is_negative_length(&v) => CssValue::Length(0.0, LengthUnit::Px),
                v => v,
            });
        }
        // B8: re-validate after math evaluation: `width: calc(1 + 2)`
        // computes to a bare number, which the width grammar rejects.
        if let Some(v) = &value
            && !is_valid_for_property(prop, v)
        {
            value = None;
            invalid = true;
        }
        debug_assert!(!invalid || value.is_none());
        match value {
            Some(v) => PropStep::Done {
                value: Some(v),
                invalid,
            },
            // Unspecified, or invalid at computed-value time: inherit,
            // else leave unset (the document's initial value).
            // WHATWG Quirks §3.11 (text decoration must not propagate
            // into tables) as probed by WPT `table-quirks.html`: the
            // test wraps a table in `font-style: italic` and asserts
            // the cell computes `normal`. `text-decoration` itself is
            // not an inherited property in this engine (and paint
            // reads it from the text box's own style, never from an
            // ancestor), so the §3.11 half is already satisfied; the
            // observable half is the inherited `font-style`, which
            // stops at the `<table>` boundary in full quirks mode
            // only — the historical non-inheritance of font properties
            // into tables. Limited-quirks documents propagate (the
            // limited-quirks twin asserts `italic`), and an explicit
            // `inherit`/`unset` keyword still climbs (handled in
            // `apply_css_wide`, not here): this gate covers only the
            // unspecified-inheritance arm.
            None if is_inherited(prop) => {
                if self.quirks_mode == QuirksMode::Quirks
                    && prop == "font-style"
                    && matches!(
                        self.doc.kind(node),
                        NodeKind::Element(d) if d.tag_name == "table"
                    )
                {
                    PropStep::Done {
                        value: None,
                        invalid,
                    }
                } else {
                    match self.style_parent(node) {
                        Some(p) => PropStep::Climb {
                            parent: p,
                            explicit: false,
                        },
                        None => PropStep::Done {
                            value: None,
                            invalid,
                        },
                    }
                }
            }
            None => PropStep::Done {
                value: None,
                invalid,
            },
        }
    }

    /// Evaluate `Math` values in a computed value, traversing arbitrarily
    /// nested lists iteratively (A1: the old code only handled one list
    /// level). Each `Math` leaf evaluates via [`Styler::eval_math_for_prop`];
    /// invalid math fails the whole value (`None`).
    fn eval_math_deep(
        &self,
        node: NodeId,
        prop: &str,
        value: Option<CssValue>,
        env: &FxHashMap<String, CssValue>,
    ) -> Option<CssValue> {
        let value = value?;
        // Iterative post-order traversal: every `Value` pushes exactly one
        // result; `ListTail(n)` combines the top `n` results into a list,
        // `MinMaxTail` combines the top two into a `minmax()`.
        enum Work {
            Value(CssValue),
            ListTail(usize),
            MinMaxTail,
            RepeatTail { mode: RepeatMode, n: usize },
            ShadowTail { n: usize, insets: Vec<bool> },
            ClipPathTail { template: ClipPath, n: usize },
        }
        let mut work = vec![Work::Value(value)];
        let mut results: Vec<Option<CssValue>> = Vec::new();
        while let Some(w) = work.pop() {
            match w {
                Work::Value(CssValue::Math(expr)) => {
                    results.push(self.eval_math_for_prop(node, prop, &expr, env))
                }
                // `calc-size()`: definite bases evaluate like math (the
                // basis substitutes for the calc-sum's `size` leaves);
                // keyword bases survive for layout (see
                // `eval_calc_size_for_prop`).
                Work::Value(CssValue::CalcSize { basis, calc }) => {
                    results.push(self.eval_calc_size_for_prop(node, prop, &basis, &calc, env))
                }
                Work::Value(CssValue::List(items)) => {
                    let n = items.len();
                    work.push(Work::ListTail(n));
                    // Reversed so items evaluate in order (LIFO stack).
                    for item in items.into_iter().rev() {
                        work.push(Work::Value(item));
                    }
                }
                // Math inside `repeat(auto-fill|auto-fit, …)` tracks
                // evaluates like math anywhere else in the track
                // listing; the descriptor keeps its shape through
                // evaluation.
                Work::Value(CssValue::Repeat { mode, tracks }) => {
                    let n = tracks.len();
                    work.push(Work::RepeatTail { mode, n });
                    // Reversed so tracks evaluate in order (LIFO stack).
                    for track in tracks.into_iter().rev() {
                        work.push(Work::Value(track));
                    }
                }
                // Math inside `minmax()` sides evaluates like math
                // anywhere else in the track listing (e.g.
                // `minmax(10px, calc(50%))`); invalid math fails the
                // whole value.
                Work::Value(CssValue::MinMax(min, max)) => {
                    work.push(Work::MinMaxTail);
                    work.push(Work::Value(*max));
                    work.push(Work::Value(*min));
                }
                // Math inside `box-shadow`/`text-shadow` layers evaluates
                // like math anywhere else (e.g. `calc(1px + 1em)` offsets);
                // the layer keeps its shape through evaluation, and the
                // `inset` flags ride along on the tail.
                Work::Value(CssValue::Shadows(shadows)) => {
                    let insets: Vec<bool> = shadows.iter().map(|s| s.inset).collect();
                    let n = shadows.len();
                    work.push(Work::ShadowTail { n, insets });
                    // Reversed so layers evaluate in source order (LIFO
                    // stack); fields push last-evaluated-first.
                    for shadow in shadows.into_iter().rev() {
                        let Shadow {
                            offset_x,
                            offset_y,
                            blur,
                            spread,
                            color,
                            ..
                        } = shadow;
                        work.push(Work::Value(color));
                        work.push(Work::Value(spread));
                        work.push(Work::Value(blur));
                        work.push(Work::Value(offset_y));
                        work.push(Work::Value(offset_x));
                    }
                }
                // Math inside `clip-path` shapes evaluates like math
                // anywhere else (e.g. `inset(calc(10px + 5%))`); the
                // shape keeps its structure through evaluation, rebuilt
                // from the evaluated leaves by `ClipPath::with_leaves`.
                Work::Value(CssValue::ClipPath(cp)) => {
                    let leaves = cp.leaves();
                    let n = leaves.len();
                    work.push(Work::ClipPathTail { template: cp, n });
                    // Reversed so leaves evaluate in canonical order
                    // (LIFO stack).
                    for leaf in leaves.into_iter().rev() {
                        work.push(Work::Value(leaf));
                    }
                }
                Work::Value(v) => results.push(Some(v)),
                Work::MinMaxTail => {
                    let max = results.pop().unwrap_or(None);
                    let min = results.pop().unwrap_or(None);
                    results.push(match (min, max) {
                        (Some(min), Some(max)) => {
                            Some(CssValue::MinMax(Box::new(min), Box::new(max)))
                        }
                        _ => None,
                    });
                }
                Work::ListTail(n) => {
                    let mut items = Vec::with_capacity(n);
                    let mut ok = true;
                    for _ in 0..n {
                        match results.pop().unwrap_or(None) {
                            Some(v) => items.push(v),
                            None => ok = false,
                        }
                    }
                    items.reverse();
                    results.push(if ok {
                        Some(CssValue::List(items))
                    } else {
                        None
                    });
                }
                Work::RepeatTail { mode, n } => {
                    let mut tracks = Vec::with_capacity(n);
                    let mut ok = true;
                    for _ in 0..n {
                        match results.pop().unwrap_or(None) {
                            Some(v) => tracks.push(v),
                            None => ok = false,
                        }
                    }
                    tracks.reverse();
                    results.push(if ok {
                        Some(CssValue::Repeat { mode, tracks })
                    } else {
                        None
                    });
                }
                Work::ShadowTail { n, insets } => {
                    // Five fields per layer were pushed (offset_x,
                    // offset_y, blur, spread, color); they pop in reverse,
                    // so reversing restores per-layer source order.
                    let mut fields = Vec::with_capacity(n * 5);
                    let mut ok = true;
                    for _ in 0..n * 5 {
                        match results.pop().unwrap_or(None) {
                            Some(v) => fields.push(v),
                            None => ok = false,
                        }
                    }
                    fields.reverse();
                    let mut shadows = Vec::with_capacity(n);
                    let mut field_iter = fields.into_iter();
                    for inset in insets {
                        match (
                            field_iter.next(),
                            field_iter.next(),
                            field_iter.next(),
                            field_iter.next(),
                            field_iter.next(),
                        ) {
                            (
                                Some(offset_x),
                                Some(offset_y),
                                Some(blur),
                                Some(spread),
                                Some(color),
                            ) => {
                                shadows.push(Shadow {
                                    offset_x,
                                    offset_y,
                                    blur,
                                    spread,
                                    color,
                                    inset,
                                });
                            }
                            _ => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    results.push(if ok {
                        Some(CssValue::Shadows(shadows))
                    } else {
                        None
                    });
                }
                Work::ClipPathTail { template, n } => {
                    // `n` leaves were pushed in canonical order; they pop
                    // in reverse, so reversing restores the order.
                    let mut leaves = Vec::with_capacity(n);
                    let mut ok = true;
                    for _ in 0..n {
                        match results.pop().unwrap_or(None) {
                            Some(v) => leaves.push(v),
                            None => ok = false,
                        }
                    }
                    leaves.reverse();
                    results.push(if ok {
                        template.with_leaves(&leaves).map(CssValue::ClipPath)
                    } else {
                        None
                    });
                }
            }
        }
        debug_assert_eq!(results.len(), 1, "eval_math_deep left stray results");
        results.pop().unwrap_or(None)
    }

    /// Evaluate one `calc()`/`min()`/`max()`/`clamp()` computed value for
    /// `prop`: substitute `var()` leaves against the element's
    /// custom-property environment, then evaluate with real font and
    /// viewport metrics. Returns `None` when the math is invalid (type
    /// errors, division by zero, unresolvable variables, excessive
    /// nesting).
    ///
    /// A2 evaluation split: math containing `%` on a property whose
    /// percentage base is a containing-block size (only layout knows it)
    /// is not evaluated here. When the tree is affine in its percentages
    /// it defers as [`CssValue::MathDeferred`] coefficients
    /// (`value = pct / 100 * base + px`, plus a `had_pct` bit recording
    /// that the tree syntactically contained a percentage — needed for
    /// cyclic-ness resolution per CSS Values 4 §8.1.4) for layout's
    /// used-value pass. Non-affine percentage math (`min()`/`max()`/
    /// `clamp()` with `%`) cannot fold to coefficients, so it defers as
    /// `MathDeferred { pct: 0.0, px, had_pct: true, affine: false }` where `px` is the
    /// historical base-0 approximation; the `had_pct` bit keeps the
    /// syntactic percentage record alive for the same cyclic-ness gates,
    /// and resolution against any base yields exactly the old base-0
    /// value. `font-size`/`line-height` percentages resolve against font
    /// metrics, so they always evaluate in the cascade.
    fn eval_math_for_prop(
        &self,
        node: NodeId,
        prop: &str,
        expr: &MathExpr,
        env: &FxHashMap<String, CssValue>,
    ) -> Option<CssValue> {
        // Fresh cycle-detection set and expansion budget per evaluation
        // (`Vec::new()` allocates nothing until the first `var()` push,
        // so var-free math pays ~nothing).
        let mut seen = Vec::new();
        let mut budget = MAX_MATH_VAR_EXPANSIONS;
        let expr = self.substitute_math_vars(node, expr, env, 0, 0, &mut seen, &mut budget)?;
        if expr.contains_percentage() && !matches!(prop, "font-size" | "line-height") {
            let ctx = self.math_context(node, prop);
            if expr.is_affine_in_percentage() {
                // Affinity makes two evaluations sufficient: font/viewport
                // leaves contribute identically to both, so the percentage
                // coefficient is their difference and the constant is the
                // base-0 value. Evaluation failure (e.g. division by zero)
                // means invalid math.
                // `had_pct` is always true inside this branch (the gate above
                // requires `contains_percentage()`), but capture it explicitly
                // so the bit travels with the value instead of being implied.
                let had_pct = expr.contains_percentage();
                return expr
                    .affine_coefficients(&ctx)
                    .map(|(pct, px)| CssValue::MathDeferred {
                        pct,
                        px,
                        had_pct,
                        affine: true,
                    });
            }
            // Non-affine percentage math: no coefficient pair can capture
            // `min()`/`max()`/`clamp()`, so keep the historical base-0
            // approximation as the constant — but carry it as
            // `MathDeferred` with the `had_pct` bit instead of folding to
            // a plain `Length`. The syntactic percentage record must
            // survive evaluation: `width: max(10px, 0%)` is cyclic (per
            // CSS Values 4 §8.1.4) and behaves as `auto` against an
            // indefinite base, not as a definite 10px. `pct: 0.0` keeps
            // the used value identical to the old base-0 fold wherever
            // layout resolves the coefficients. Evaluation failure (e.g.
            // division by zero) means invalid math.
            return match expr.evaluate(&ctx)? {
                // Unreachable when a `%` leaf is present (a percentage
                // always evaluates to `Length`, and mixing it with a bare
                // number is invalid math), but keep the old fold shape
                // rather than inventing a value. An angle result is
                // invalid at computed-value time: no property takes
                // `<angle>` through `calc()`.
                MathValue::Number(n) => Some(CssValue::Number(n)),
                MathValue::Angle(_) => None,
                MathValue::Length(px) => Some(CssValue::MathDeferred {
                    pct: 0.0,
                    px,
                    had_pct: true,
                    affine: false,
                }),
            };
        }
        let ctx = self.math_context(node, prop);
        match expr.evaluate(&ctx)? {
            MathValue::Number(n) => Some(CssValue::Number(n)),
            MathValue::Length(px) => Some(CssValue::Length(px, LengthUnit::Px)),
            // No property takes `<angle>` through `calc()`: an angle
            // result is invalid at computed-value time. This is what
            // rejects `width: calc(90deg)` — and `opacity: calc(90deg)` —
            // for the right reason, by type checking rather than
            // incidentally.
            MathValue::Angle(_) => None,
        }
    }

    /// Evaluate a `calc-size(<basis>, <calc-sum>)` for `prop`. `var()`
    /// leaves in the basis math and in the calc-sum substitute first,
    /// and a `var()` in basis position substitutes against the custom
    /// property environment before basis validation (CSS Values 4:
    /// substitution precedes grammar checking). A definite basis
    /// ([`CalcSizeBasis::is_definite`]) substitutes for the calc-sum's
    /// `size` leaves and the resulting tree evaluates exactly like a
    /// `calc()` (including the layout-deferred percentage split). A
    /// keyword basis (`auto`, `max-content`, ...) needs content
    /// measurement the cascade cannot do, so the value survives as
    /// [`CssValue::CalcSize`] for layout, which reads the basis via
    /// [`CssValue::calc_size_basis`]. Returns `None` when the math is
    /// invalid (unresolvable `var()`, type errors, division by zero,
    /// excessive nesting).
    fn eval_calc_size_for_prop(
        &self,
        node: NodeId,
        prop: &str,
        basis: &CalcSizeBasis,
        calc: &MathExpr,
        env: &FxHashMap<String, CssValue>,
    ) -> Option<CssValue> {
        let mut seen = Vec::new();
        let mut budget = MAX_MATH_VAR_EXPANSIONS;
        let calc = self.substitute_math_vars(node, calc, env, 0, 0, &mut seen, &mut budget)?;
        let basis = match basis {
            CalcSizeBasis::Math(e) => {
                let mut seen = Vec::new();
                let mut budget = MAX_MATH_VAR_EXPANSIONS;
                CalcSizeBasis::Math(self.substitute_math_vars(
                    node,
                    e,
                    env,
                    0,
                    0,
                    &mut seen,
                    &mut budget,
                )?)
            }
            // A `var()` basis substitutes before basis validation: the
            // custom property's token stream is checked against the
            // basis grammar (or the `var()` fallback is, when the
            // property is undefined or not a valid basis).
            CalcSizeBasis::Var { name, fallback } => {
                self.substitute_basis_var(name, fallback, env, 0)?
            }
            // Keyword/length/percentage bases hold no `var()` leaves.
            b => b.clone(),
        };
        match basis.as_math_expr() {
            Some(base) => {
                let tree = calc.substitute_size(&base)?;
                self.eval_math_for_prop(node, prop, &tree, env)
            }
            // Keyword basis: keep the parsed form for layout.
            None => Some(CssValue::CalcSize {
                basis: Box::new(basis.clone()),
                calc: Box::new(calc.clone()),
            }),
        }
    }

    /// Substitute a `var()` reference in `calc-size()` basis position.
    /// The custom property's token stream is checked against the basis
    /// grammar (substitution precedes grammar checking, CSS Values 4);
    /// a nested `var()` recurses, depth-capped against adversarial
    /// `--a: var(--b)` chains. An undefined property, a value that is
    /// not a valid basis, or a cycle falls back to the `var()`
    /// fallback tokens (also basis-grammar-checked); `None` with no
    /// usable fallback invalidates the declaration, exactly like an
    /// unresolvable `var()` in math (see [`Styler::substitute_math_vars`]).
    fn substitute_basis_var(
        &self,
        name: &str,
        fallback: &Option<Vec<Token>>,
        env: &FxHashMap<String, CssValue>,
        depth: u32,
    ) -> Option<CalcSizeBasis> {
        if depth > MAX_MATH_DEPTH {
            return None;
        }
        // The `var()` fallback, checked against the basis grammar. An
        // empty fallback substitutes nothing, which is invalid for a
        // basis.
        let fallback_basis = |depth: u32| -> Option<CalcSizeBasis> {
            let toks = fallback.as_ref()?;
            if toks.is_empty() {
                return None;
            }
            // Custom property streams may carry whitespace the basis
            // parser does not accept; filter it like `parse_calc_size`
            // does.
            let refs: Vec<&Token> = toks
                .iter()
                .filter(|t| !matches!(t, Token::Whitespace))
                .collect();
            match parse_calc_size_basis(&refs, depth + 1) {
                Ok(CalcSizeBasis::Var { name, fallback }) => {
                    self.substitute_basis_var(&name, &fallback, env, depth + 1)
                }
                Ok(basis) => Some(basis),
                Err(_) => None,
            }
        };
        match env.get(name) {
            Some(CssValue::Tokens(toks)) => {
                let refs: Vec<&Token> = toks
                    .iter()
                    .filter(|t| !matches!(t, Token::Whitespace))
                    .collect();
                match parse_calc_size_basis(&refs, depth + 1) {
                    // A nested `var()` keeps substituting; if it fails,
                    // the outer fallback still applies (B4).
                    Ok(CalcSizeBasis::Var { name, fallback }) => self
                        .substitute_basis_var(&name, &fallback, env, depth + 1)
                        .or_else(|| fallback_basis(depth)),
                    Ok(basis) => Some(basis),
                    // Defined but not a valid basis: the `var()`
                    // fallback applies, exactly as for an undefined
                    // property.
                    Err(_) => fallback_basis(depth),
                }
            }
            // An already-parsed value (e.g. from inline style): map the
            // basis-shaped ones directly.
            Some(v) => Self::basis_of_value(v).or_else(|| fallback_basis(depth)),
            None => fallback_basis(depth),
        }
    }

    /// Map an already-parsed value to a `calc-size()` basis, if it is
    /// basis-shaped: a sizing keyword, a length, a percentage, or math.
    /// `None` for anything else (colors, arbitrary keywords, ...).
    fn basis_of_value(value: &CssValue) -> Option<CalcSizeBasis> {
        match value {
            CssValue::Length(n, unit) => Some(CalcSizeBasis::Length(*n, unit.clone())),
            CssValue::Percentage(p) => Some(CalcSizeBasis::Percentage(*p)),
            CssValue::Math(e) => Some(CalcSizeBasis::Math(e.clone())),
            // CSS-wide rule: unitless zero is a valid `<length>`.
            CssValue::Number(n) if *n == 0.0 => Some(CalcSizeBasis::Length(0.0, LengthUnit::Px)),
            CssValue::Keyword(kw) => match kw.to_ascii_lowercase().as_str() {
                "auto" => Some(CalcSizeBasis::Auto),
                "max-content" => Some(CalcSizeBasis::MaxContent),
                "min-content" => Some(CalcSizeBasis::MinContent),
                "fit-content" => Some(CalcSizeBasis::FitContent),
                "stretch" | "available" => Some(CalcSizeBasis::Stretch),
                _ => None,
            },
            _ => None,
        }
    }

    /// Substitute `var()` leaves in a math tree. A defined custom
    /// property resolves (with `var()` cycle handling from
    /// [`Styler::resolve`]) and must itself be a math leaf; otherwise the
    /// fallback is used.
    ///
    /// Two separate counters guard the recursion:
    ///
    /// * `depth` counts **actual `var()` substitutions** — one unit per
    ///   expanded `var()` leaf (the [`MathExpr::Var`] arm only). Pure
    ///   structural descent (`Add`, `Neg`, `Min`, …) does not spend it.
    ///   Capped at [`MAX_MATH_DEPTH`]: a `--ci: calc(var(--c{i-1}) +
    ///   0px)` chain costs one unit per link, so up to ~64 links resolve
    ///   and longer chains are invalid at computed-value time
    ///   (fail-closed).
    /// * `sdepth` counts **pure structural descent** with no substitution
    ///   work. Capped at [`MAX_MATH_SUBST_STRUCT_DEPTH`], a stack-safety
    ///   backstop far above parse-time nesting limits.
    ///
    /// True cycles are caught by `seen` before either counter matters,
    /// and diamond-shaped expansion blowup is bounded by `budget`.
    ///
    /// `seen` holds the interned ids of the custom properties on the
    /// current expansion path. `resolve()` re-parses a custom property's
    /// token stream without substituting `var()` leaves nested inside
    /// `calc()` (the borrowed-`Tokens` arm of the resolve machine just
    /// re-parses), so a calc-wrapped cycle (`--a: calc(var(--b))`, `--b:
    /// calc(var(--a))`) would otherwise recurse [`MAX_MATH_DEPTH`] deep
    /// — one full resolve plus re-parse per level — before failing, and
    /// `resolve()` clears its own `resolve_active` set on entry, so the
    /// cycle state cannot live there. This separate path-local set spans
    /// the re-entry instead, mirroring
    /// [`Styler::custom_prop_angle_radians`]. Ids are pushed before
    /// expanding and popped after, so the same property referenced twice
    /// in one `calc()` (`calc(var(--y) + var(--y))`) still resolves —
    /// only a true cycle fails, invalid at computed-value time.
    /// `budget` spends one per custom-property expansion and bounds the
    /// total substitution work for shapes a path-local set cannot catch
    /// (a diamond doubles the expansions per level at constant depth).
    /// Both are untouched when the tree holds no `var()` leaves, so the
    /// common acyclic case pays ~nothing (a `Vec::new()` never pushes).
    // 8 params is inherent to this recursive substitution pass: node/env
    // are the immutable context, depth/sdepth/seen/budget the traversal
    // state. Bundling them into a struct would churn every recursive
    // call site for no behavioral gain.
    #[allow(clippy::too_many_arguments)]
    fn substitute_math_vars(
        &self,
        node: NodeId,
        expr: &MathExpr,
        env: &FxHashMap<String, CssValue>,
        depth: u32,
        sdepth: u32,
        seen: &mut Vec<u32>,
        budget: &mut usize,
    ) -> Option<MathExpr> {
        if depth > MAX_MATH_DEPTH || sdepth > MAX_MATH_SUBST_STRUCT_DEPTH {
            return None;
        }
        let sd = sdepth + 1;
        match expr {
            MathExpr::Var(name, fallback) => {
                // A defined-but-invalid custom property
                // (`--x: var(--undefined)`) resolves to `None`: the `var()`
                // fallback applies (B4), exactly as for an undefined
                // property. A defined value that is not a math leaf (e.g.
                // a keyword) fails the expression: per CSS, substitution
                // already succeeded, so the fallback does not apply.
                //
                // Cycle detection spans the `resolve()` re-entry below: a
                // property already on the current expansion path is
                // cyclic, so the declaration is invalid at
                // computed-value time — reached here instead of after
                // `MAX_MATH_DEPTH` full resolve + re-parse rounds. The id
                // is popped after expansion, so a repeated reference on a
                // sibling branch is not a cycle.
                let id = self.var_names.borrow_mut().intern(name);
                if seen.contains(&id) {
                    return None; // cyclic reference: invalid at computed-value time
                }
                if *budget == 0 {
                    return None; // hostile expansion shape: invalid at computed-value time
                }
                *budget -= 1;
                seen.push(id);
                let replacement = if let Some(v) = env.get(name) {
                    // Angle pre-pass: a custom property holding a single
                    // angle dimension (`--x: 90deg`, possibly through
                    // `var()` indirection) folds to an `Angle` leaf here,
                    // because the generic `parse_value` used by `resolve()`
                    // rejects a bare `deg` dimension and would otherwise
                    // fail the whole substitution.
                    if let Some(angle) = self.angle_leaf_of_custom_prop(name, env) {
                        Some(angle)
                    } else {
                        match self.resolve(node, v, env) {
                            // Defined and itself a math leaf: substitute it.
                            Some(r) => Self::math_leaf_of(&r),
                            // Defined but invalid at computed-value time
                            // (`--x: var(--undefined)`): the `var()` fallback
                            // applies, as for an undefined property.
                            None => fallback.as_deref().cloned(),
                        }
                    }
                } else {
                    fallback.as_deref().cloned()
                };
                // One unit of `depth` per actual `var()` substitution:
                // the replacement tree's own structural descent spends
                // `sdepth`, not `depth`.
                let d = depth + 1;
                let result = match replacement {
                    // The replacement may itself hold `var()` leaves
                    // (e.g. `--x: var(--y)`); keep substituting.
                    Some(e) => self.substitute_math_vars(node, &e, env, d, sdepth, seen, budget),
                    None => None,
                };
                seen.pop();
                result
            }
            MathExpr::Number(_)
            | MathExpr::Angle(_)
            | MathExpr::Percentage(_)
            | MathExpr::Length(..) => Some(expr.clone()),
            // The `calc-size()` interpolation keyword is not a `var()`:
            // it passes substitution through untouched and is replaced
            // with the resolved basis afterwards (see
            // `eval_calc_size_for_prop`).
            MathExpr::Size => Some(MathExpr::Size),
            MathExpr::Neg(e) => self
                .substitute_math_vars(node, e, env, depth, sd, seen, budget)
                .map(|e| MathExpr::Neg(Box::new(e))),
            MathExpr::Add(a, b) => Some(MathExpr::Add(
                Box::new(self.substitute_math_vars(node, a, env, depth, sd, seen, budget)?),
                Box::new(self.substitute_math_vars(node, b, env, depth, sd, seen, budget)?),
            )),
            MathExpr::Sub(a, b) => Some(MathExpr::Sub(
                Box::new(self.substitute_math_vars(node, a, env, depth, sd, seen, budget)?),
                Box::new(self.substitute_math_vars(node, b, env, depth, sd, seen, budget)?),
            )),
            MathExpr::Mul(a, b) => Some(MathExpr::Mul(
                Box::new(self.substitute_math_vars(node, a, env, depth, sd, seen, budget)?),
                Box::new(self.substitute_math_vars(node, b, env, depth, sd, seen, budget)?),
            )),
            MathExpr::Div(a, b) => Some(MathExpr::Div(
                Box::new(self.substitute_math_vars(node, a, env, depth, sd, seen, budget)?),
                Box::new(self.substitute_math_vars(node, b, env, depth, sd, seen, budget)?),
            )),
            MathExpr::Min(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.substitute_math_vars(node, item, env, depth, sd, seen, budget)?);
                }
                Some(MathExpr::Min(out))
            }
            MathExpr::Max(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.substitute_math_vars(node, item, env, depth, sd, seen, budget)?);
                }
                Some(MathExpr::Max(out))
            }
            MathExpr::Clamp(a, b, c) => Some(MathExpr::Clamp(
                Box::new(self.substitute_math_vars(node, a, env, depth, sd, seen, budget)?),
                Box::new(self.substitute_math_vars(node, b, env, depth, sd, seen, budget)?),
                Box::new(self.substitute_math_vars(node, c, env, depth, sd, seen, budget)?),
            )),
            // Wave 55 (CSS Values 4 math functions): pure structural
            // recursion, mirroring the arms above. `var()` leaves inside
            // the new functions substitute exactly like the old ones;
            // `if()` conditions hold no `var()` leaves (opaque token
            // streams), so only branch values recurse.
            MathExpr::Abs(e) => self
                .substitute_math_vars(node, e, env, depth, sd, seen, budget)
                .map(|e| MathExpr::Abs(Box::new(e))),
            MathExpr::Sign(e) => self
                .substitute_math_vars(node, e, env, depth, sd, seen, budget)
                .map(|e| MathExpr::Sign(Box::new(e))),
            MathExpr::Sqrt(e) => self
                .substitute_math_vars(node, e, env, depth, sd, seen, budget)
                .map(|e| MathExpr::Sqrt(Box::new(e))),
            MathExpr::Exp(e) => self
                .substitute_math_vars(node, e, env, depth, sd, seen, budget)
                .map(|e| MathExpr::Exp(Box::new(e))),
            MathExpr::Sin(e) => self
                .substitute_math_vars(node, e, env, depth, sd, seen, budget)
                .map(|e| MathExpr::Sin(Box::new(e))),
            MathExpr::Cos(e) => self
                .substitute_math_vars(node, e, env, depth, sd, seen, budget)
                .map(|e| MathExpr::Cos(Box::new(e))),
            MathExpr::Tan(e) => self
                .substitute_math_vars(node, e, env, depth, sd, seen, budget)
                .map(|e| MathExpr::Tan(Box::new(e))),
            MathExpr::Asin(e) => self
                .substitute_math_vars(node, e, env, depth, sd, seen, budget)
                .map(|e| MathExpr::Asin(Box::new(e))),
            MathExpr::Acos(e) => self
                .substitute_math_vars(node, e, env, depth, sd, seen, budget)
                .map(|e| MathExpr::Acos(Box::new(e))),
            MathExpr::Atan(e) => self
                .substitute_math_vars(node, e, env, depth, sd, seen, budget)
                .map(|e| MathExpr::Atan(Box::new(e))),
            MathExpr::Mod(a, b) => Some(MathExpr::Mod(
                Box::new(self.substitute_math_vars(node, a, env, depth, sd, seen, budget)?),
                Box::new(self.substitute_math_vars(node, b, env, depth, sd, seen, budget)?),
            )),
            MathExpr::Rem(a, b) => Some(MathExpr::Rem(
                Box::new(self.substitute_math_vars(node, a, env, depth, sd, seen, budget)?),
                Box::new(self.substitute_math_vars(node, b, env, depth, sd, seen, budget)?),
            )),
            MathExpr::Pow(a, b) => Some(MathExpr::Pow(
                Box::new(self.substitute_math_vars(node, a, env, depth, sd, seen, budget)?),
                Box::new(self.substitute_math_vars(node, b, env, depth, sd, seen, budget)?),
            )),
            MathExpr::Atan2(a, b) => Some(MathExpr::Atan2(
                Box::new(self.substitute_math_vars(node, a, env, depth, sd, seen, budget)?),
                Box::new(self.substitute_math_vars(node, b, env, depth, sd, seen, budget)?),
            )),
            MathExpr::Round { strategy, a, b } => Some(MathExpr::Round {
                strategy: *strategy,
                a: Box::new(self.substitute_math_vars(node, a, env, depth, sd, seen, budget)?),
                b: match b {
                    Some(b) => Some(Box::new(
                        self.substitute_math_vars(node, b, env, depth, sd, seen, budget)?,
                    )),
                    None => None,
                },
            }),
            MathExpr::Log { a, b } => Some(MathExpr::Log {
                a: Box::new(self.substitute_math_vars(node, a, env, depth, sd, seen, budget)?),
                b: match b {
                    Some(b) => Some(Box::new(
                        self.substitute_math_vars(node, b, env, depth, sd, seen, budget)?,
                    )),
                    None => None,
                },
            }),
            MathExpr::Hypot(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.substitute_math_vars(node, item, env, depth, sd, seen, budget)?);
                }
                Some(MathExpr::Hypot(out))
            }
            MathExpr::Toggle(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.substitute_math_vars(node, item, env, depth, sd, seen, budget)?);
                }
                Some(MathExpr::Toggle(out))
            }
            MathExpr::If(branches) => {
                let mut out = Vec::with_capacity(branches.len());
                for branch in branches {
                    out.push(IfBranch {
                        condition: branch.condition.clone(),
                        value: self.substitute_math_vars(
                            node,
                            &branch.value,
                            env,
                            depth,
                            sd,
                            seen,
                            budget,
                        )?,
                    });
                }
                Some(MathExpr::If(out))
            }
        }
    }

    /// Convert a resolved value into a math-tree leaf, if it is one.
    fn math_leaf_of(value: &CssValue) -> Option<MathExpr> {
        match value {
            CssValue::Number(n) => Some(MathExpr::Number(*n)),
            CssValue::Percentage(p) => Some(MathExpr::Percentage(*p)),
            CssValue::Length(n, u) => Some(MathExpr::Length(*n, u.clone())),
            CssValue::Math(e) => Some(e.clone()),
            _ => None,
        }
    }

    /// Fold a custom property's value to an [`MathExpr::Angle`] leaf when
    /// it is a single angle dimension — directly (`--x: 90deg`) or
    /// through `var()` indirection (`--x: var(--y)` with `--y: 90deg`,
    /// or `--x: var(--undefined, 90deg)`). Returns `None` for anything
    /// else so the caller falls through to the normal [`Styler::resolve`]
    /// path.
    ///
    /// The generic [`parse_value`] used by `resolve()` rejects a bare
    /// `deg`/`grad`/`rad`/`turn` dimension, so without this pre-pass a
    /// `var()`-parameterized angle (`--x: 90deg; opacity: sin(var(--x))`)
    /// fails substitution and drops the declaration while the equivalent
    /// literal (`sin(90deg)`) computes.
    fn angle_leaf_of_custom_prop(
        &self,
        name: &str,
        env: &FxHashMap<String, CssValue>,
    ) -> Option<MathExpr> {
        let mut seen: Vec<u32> = Vec::new();
        self.custom_prop_angle_radians(name, env, 0, &mut seen)
            .map(MathExpr::Angle)
    }

    /// [`Styler::angle_leaf_of_custom_prop`] worker: radians of the
    /// custom property's angle-dimension value, following `var()`
    /// indirection. `seen` holds the interned names on the current
    /// indirection chain (cycle detection); `depth` caps adversarial
    /// `--a: var(--b)` chains at [`MAX_MATH_DEPTH`].
    fn custom_prop_angle_radians(
        &self,
        name: &str,
        env: &FxHashMap<String, CssValue>,
        depth: u32,
        seen: &mut Vec<u32>,
    ) -> Option<f32> {
        if depth > MAX_MATH_DEPTH {
            return None;
        }
        let value = env.get(name)?;
        // A guaranteed-invalid property is not an angle: fall through
        // to the `resolve()` path so the `var()` fallback applies (B4).
        if is_guaranteed_invalid(value) {
            return None;
        }
        match value {
            // Custom properties hold verbatim token streams.
            CssValue::Tokens(toks) => {
                if let Some(r) = single_angle_radians(toks) {
                    return Some(r);
                }
                // `var()` indirection inside the stream.
                let (target, fallback) = var_ref_parts(toks)?;
                let id = self.var_names.borrow_mut().intern(target);
                if seen.contains(&id) {
                    return None; // cyclic: not an angle via this path
                }
                seen.push(id);
                let r = self
                    .custom_prop_angle_radians(target, env, depth + 1, seen)
                    .or_else(|| fallback.and_then(single_angle_radians));
                seen.pop();
                r
            }
            // An already-parsed `var()` (e.g. from inline style).
            CssValue::Var(target, fallback) => {
                let id = self.var_names.borrow_mut().intern(target);
                if seen.contains(&id) {
                    return None;
                }
                seen.push(id);
                let r = self
                    .custom_prop_angle_radians(target, env, depth + 1, seen)
                    .or_else(|| {
                        fallback.as_ref().and_then(|fb| match fb.as_ref() {
                            CssValue::Tokens(toks) => single_angle_radians(toks),
                            _ => None,
                        })
                    });
                seen.pop();
                r
            }
            _ => None,
        }
    }

    /// The CSS root: the document element (`<html>` in a parsed
    /// document). This is what `rem` units and `:root` refer to — note
    /// [`Document::root`] is the *document node*, not the root element,
    /// so it must not be used for root-relative CSS semantics. Falls back
    /// to the document node when the document has no element yet.
    fn css_root(&self) -> NodeId {
        self.doc
            .document_element()
            .unwrap_or_else(|| self.doc.root())
    }

    /// Metrics a math expression on `prop` resolves against: `em` uses
    /// the element's computed font size (the parent's when computing
    /// `font-size` itself, per CSS), `rem` the root font size, `vw`/`vh`
    /// the viewport. Percentages resolve against the font size for
    /// `font-size` and `line-height`, which always evaluate here.
    ///
    /// A2 evaluation split: for every other property, percentage-bearing
    /// math never reaches this context — affine trees defer to layout as
    /// [`CssValue::MathDeferred`] coefficients
    /// (see [`Styler::eval_math_for_prop`]), and non-affine percentage
    /// math (`min()`/`max()`/`clamp()` with `%`) defers as
    /// `MathDeferred { pct: 0.0, px, had_pct: true, affine: false }`, where `px` is the
    /// historical base-0 approximation and the `had_pct` bit preserves
    /// the syntactic percentage for cyclic-ness resolution.
    fn math_context(&self, node: NodeId, prop: &str) -> MathContext {
        let root = self.css_root();
        let font_size = if prop == "font-size" {
            self.doc
                .parent(node)
                .map(|p| self.font_size_px(p))
                .unwrap_or(DEFAULT_FONT_SIZE_PX)
        } else {
            self.font_size_px(node)
        };
        // While computing the root's own `font-size`, `rem` resolves
        // against the initial size (CSS Values 4): asking for the root's
        // computed size here would re-enter `font_size_px(root)`
        // mid-computation, which never terminates (the memo insert
        // happens after return).
        let root_font_size = if prop == "font-size" && node == root {
            DEFAULT_FONT_SIZE_PX
        } else {
            self.font_size_px(root)
        };
        MathContext {
            font_size,
            root_font_size,
            viewport_width: self.viewport.w,
            viewport_height: self.viewport.h,
            percentage_base: match prop {
                "font-size" | "line-height" => font_size,
                _ => 0.0,
            },
        }
    }

    /// Computed `font-size` of `node` in px. Mirrors the absolute-keyword
    /// scale in `bowser_layout::style` (which owns final font
    /// resolution); memoized per node so ancestor walks stay linear.
    ///
    /// Iterative over the ancestor chain (B3): the old recursion
    /// overflowed the stack on a cold 10k-deep DOM.
    fn font_size_px(&self, node: NodeId) -> f32 {
        let epoch = self.epoch.get();
        if let Some(px) = self.font_px.borrow().probe(node, epoch).copied() {
            return px;
        }
        // Re-entry guard (CRITICAL): `font_size_px_for` -> `resolved_prop`
        // -> `compute_prop` -> `eval_math_deep` -> `math_context` can call
        // back into `font_size_px` for a node whose size is still being
        // computed, and the memo insert below happens after return — so an
        // unguarded cycle recurses until the stack overflows on
        // remote-stylesheet input. Fail closed with the initial size.
        if !self.font_px_active.borrow_mut().insert(node) {
            return DEFAULT_FONT_SIZE_PX;
        }
        // Collect the unmemoized ancestor chain, node first and rootward.
        let mut chain: Vec<NodeId> = Vec::new();
        let mut n = node;
        let mut parent_px = loop {
            if let Some(px) = self.font_px.borrow().probe(n, epoch).copied() {
                break px;
            }
            match self.style_parent(n) {
                Some(p) => {
                    chain.push(n);
                    n = p;
                }
                None => {
                    chain.push(n);
                    break DEFAULT_FONT_SIZE_PX;
                }
            }
        };
        // Walk back down (rootward first): each node's px derives from its
        // parent's, exactly as the old recursion did, memoizing each.
        for m in chain.into_iter().rev() {
            let px = self.font_size_px_for(m, parent_px);
            self.font_px.borrow_mut().insert(m, epoch, px);
            parent_px = px;
        }
        self.font_px_active.borrow_mut().remove(&node);
        parent_px
    }

    /// One node's computed `font-size` in px given its parent's.
    fn font_size_px_for(&self, node: NodeId, parent_fs: f32) -> f32 {
        match self
            .resolved_prop(node, font_size_prop_idx(), "font-size")
            .as_deref()
        {
            Some(CssValue::Length(v, unit)) => match unit {
                LengthUnit::Px => *v,
                LengthUnit::Cm => *v * 96.0 / 2.54,
                LengthUnit::Mm => *v * 96.0 / 25.4,
                LengthUnit::In => *v * 96.0,
                LengthUnit::Pt => *v * 96.0 / 72.0,
                LengthUnit::Pc => *v * 16.0,
                LengthUnit::Q => *v * 96.0 / 101.6,
                LengthUnit::Em => *v * parent_fs,
                // `ex` is the x-height; without font metrics, 0.5em is the
                // CSS-specified fallback.
                LengthUnit::Ex => *v * parent_fs * 0.5,
                // CSS Values 4: `rem` on the root element itself resolves
                // against the initial font size, never recursively (an
                // unguarded `font_size_px(root)` here re-enters forever
                // when the root's own font size is a `rem` length).
                LengthUnit::Rem => {
                    if node == self.css_root() {
                        *v * DEFAULT_FONT_SIZE_PX
                    } else {
                        *v * self.font_size_px(self.css_root())
                    }
                }
                LengthUnit::Vw => *v * self.viewport.w / 100.0,
                LengthUnit::Vh => *v * self.viewport.h / 100.0,
                LengthUnit::Vmin => *v * self.viewport.w.min(self.viewport.h) / 100.0,
                LengthUnit::Vmax => *v * self.viewport.w.max(self.viewport.h) / 100.0,
            },
            Some(CssValue::Percentage(p)) => parent_fs * *p / 100.0,
            Some(CssValue::Number(n)) => *n,
            Some(CssValue::Keyword(k)) => match k.to_ascii_lowercase().as_str() {
                // Absolute keywords use the conventional UA scale.
                "xx-small" => 9.0,
                "x-small" => 10.0,
                "small" => 13.0,
                "medium" => 16.0,
                "large" => 18.0,
                "x-large" => 24.0,
                "xx-large" => 32.0,
                "larger" => parent_fs * 1.2,
                "smaller" => parent_fs / 1.2,
                _ => parent_fs,
            },
            _ => parent_fs,
        }
        .max(0.0)
    }

    /// Apply a CSS-wide keyword (`inherit`, `initial`, `unset`) to
    /// `prop` on `node`. `inherit` (and `unset` on inherited properties)
    /// climbs via [`PropStep::Climb`] so [`Styler::finish_prop`] can drive
    /// the ancestor walk iteratively (B3); a climb that finds no value
    /// falls back to the initial value, as before.
    /// Winning candidate for `prop` from an origin class strictly below
    /// the overall winner's, from ascending-sorted candidates. The
    /// shared lookup behind `revert` (and unlayered `revert-layer`) for
    /// both regular and custom properties.
    fn lower_origin_winner<'x>(
        cands: &'x [Candidate<'a>],
        prop: &str,
    ) -> Option<&'x Candidate<'a>> {
        let mut winner_class: Option<u8> = None;
        for c in cands.iter().rev() {
            if c.property.as_str() != prop {
                continue;
            }
            let class = revert_origin_class(c.rank);
            match winner_class {
                None => winner_class = Some(class),
                Some(wc) => {
                    if class < wc {
                        return Some(c);
                    }
                }
            }
        }
        None
    }

    /// Lower-origin winner for `revert` (and unlayered `revert-layer`):
    /// the substituted value of the winning declaration for `prop`
    /// among candidates from origins strictly below the `revert`
    /// declaration's own origin. Returns `None` when no lower origin
    /// declares the property (the caller falls back to `unset`
    /// semantics), or when the lower-origin winner is itself
    /// `revert`/`revert-layer` — either literally or after `var()`
    /// substitution (a `revert` with no lower origin to roll back to
    /// behaves as `unset`, so this never chains or loops).
    fn revert_value(
        &self,
        node: NodeId,
        prop: &str,
        env: &FxHashMap<String, CssValue>,
    ) -> Option<CssValue> {
        let cands = self.candidates(node);
        let lower = Self::lower_origin_winner(&cands, prop)?;
        // A `revert` in the lowest declaring origin has nothing to roll
        // back to: `unset` semantics (handled by the caller). Check both
        // the raw value and the substituted result: a lower-origin
        // `var()` may itself substitute to `revert`, and returning that
        // would make the caller's keyword loop spin forever.
        if let CssValue::Keyword(kw) = &*lower.value
            && matches!(kw.as_str(), "revert" | "revert-layer")
        {
            return None;
        }
        let resolved = self.resolve(node, &lower.value, env)?;
        if let CssValue::Keyword(kw) = &resolved
            && matches!(kw.as_str(), "revert" | "revert-layer")
        {
            return None;
        }
        Some(resolved)
    }

    fn apply_css_wide(&self, node: NodeId, prop: &str, keyword: &str) -> PropStep {
        let climb_or_initial = || match self.style_parent(node) {
            // Explicit `inherit`: nothing above still yields the initial
            // value (the old `parent_value().or_else(initial)`).
            Some(p) => PropStep::Climb {
                parent: p,
                explicit: true,
            },
            None => PropStep::Done {
                value: initial_value(prop),
                invalid: false,
            },
        };
        match keyword {
            "inherit" => climb_or_initial(),
            "initial" => PropStep::Done {
                value: initial_value(prop),
                invalid: false,
            },
            // Inherited properties behave as `inherit`, the rest as
            // `initial`.
            "unset" => {
                if is_inherited(prop) {
                    climb_or_initial()
                } else {
                    PropStep::Done {
                        value: initial_value(prop),
                        invalid: false,
                    }
                }
            }
            _ => PropStep::Done {
                value: None,
                invalid: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{parse_declarations, parse_stylesheet};
    use crate::value::{AttrData, Color, CounterData, LengthUnit};

    fn ua() -> Stylesheet {
        default_ua_stylesheet()
    }

    fn vp() -> Viewport {
        Viewport { w: 800.0, h: 600.0 }
    }

    fn style(doc: &Document, node: NodeId, css: &str, viewport: &Viewport) -> ComputedStyle {
        let author = parse_stylesheet(css);
        compute_style(doc, node, &[&author], &ua(), viewport)
    }

    fn sample() -> (Document, NodeId, NodeId) {
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        doc.set_attribute(div, "id", "d");
        doc.set_attribute(div, "class", "a");
        let p = doc.create_element("p");
        doc.append_child(div, p);
        doc.set_attribute(p, "id", "p");
        (doc, div, p)
    }

    fn kw(s: &str) -> CssValue {
        CssValue::Keyword(s.into())
    }

    /// A `table > tbody > tr > td` document; `attrs` are the table's
    /// attributes. Returns (doc, table, td).
    fn table_doc(attrs: &[(&str, &str)]) -> (Document, NodeId, NodeId) {
        let mut doc = Document::new();
        let root = doc.root();
        let table = doc.create_element("table");
        doc.append_child(root, table);
        for (name, value) in attrs {
            doc.set_attribute(table, name, value);
        }
        let tbody = doc.create_element("tbody");
        doc.append_child(table, tbody);
        let tr = doc.create_element("tr");
        doc.append_child(tbody, tr);
        let td = doc.create_element("td");
        doc.append_child(tr, td);
        (doc, table, td)
    }

    fn gray() -> CssValue {
        CssValue::Color(Color {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        })
    }

    fn black() -> CssValue {
        CssValue::Color(Color {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        })
    }

    #[test]
    fn table_mapping_case1_rules_groups_border_zero() {
        // WPT html-to-css-mapping-2 case 1: 0px hidden gray on the table.
        let (doc, table, _td) = table_doc(&[("border", "0"), ("rules", "groups")]);
        let s = style(&doc, table, "", &vp());
        assert_eq!(
            s.get("border-left-width"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        assert_eq!(s.get_str("border-left-style"), Some("hidden"));
        assert_eq!(s.get("border-left-color"), Some(&gray()));
        assert_eq!(s.get_str("border-collapse"), Some("collapse"));
    }

    #[test]
    fn table_mapping_case2_frame_box_border_zero() {
        // Case 2: 1px (thin) outset black.
        let (doc, table, _td) =
            table_doc(&[("border", "0"), ("rules", "groups"), ("frame", "box")]);
        let s = style(&doc, table, "", &vp());
        assert_eq!(s.get("border-left-width"), Some(&kw("thin")));
        assert_eq!(s.get_str("border-left-style"), Some("outset"));
        assert_eq!(s.get("border-left-color"), Some(&black()));
    }

    #[test]
    fn table_mapping_case3_border_ten_frame_box() {
        // Case 3a: 10px outset gray on the table; 3b/3c: 1px inset
        // gray on the cell.
        let (doc, table, td) =
            table_doc(&[("border", "10"), ("rules", "groups"), ("frame", "box")]);
        let s = style(&doc, table, "", &vp());
        assert_eq!(
            s.get("border-left-width"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
        assert_eq!(s.get_str("border-left-style"), Some("outset"));
        assert_eq!(s.get("border-left-color"), Some(&gray()));
        let c = style(&doc, td, "", &vp());
        assert_eq!(
            c.get("border-top-width"),
            Some(&CssValue::Length(1.0, LengthUnit::Px))
        );
        assert_eq!(c.get_str("border-top-style"), Some("inset"));
        assert_eq!(c.get_str("border-left-style"), Some("inset"));
        assert_eq!(c.get("border-top-color"), Some(&gray()));
        // No `cellpadding` attribute: no padding hint, so the UA
        // default `td { padding: 1px }` (CSS 2.1 Appendix D) applies.
        assert_eq!(
            c.get("padding-top"),
            Some(&CssValue::Length(1.0, LengthUnit::Px))
        );
    }

    #[test]
    fn table_mapping_case4_border_yes_rules_all() {
        // Case 4: `border="yes"` is a parse error -> 1px outset gray;
        // the cell is 1px solid gray via rules=all.
        let (doc, table, td) = table_doc(&[("border", "yes"), ("rules", "all")]);
        let s = style(&doc, table, "", &vp());
        assert_eq!(
            s.get("border-left-width"),
            Some(&CssValue::Length(1.0, LengthUnit::Px))
        );
        assert_eq!(s.get_str("border-left-style"), Some("outset"));
        assert_eq!(s.get("border-left-color"), Some(&gray()));
        let c = style(&doc, td, "", &vp());
        assert_eq!(c.get_str("border-left-style"), Some("solid"));
        assert_eq!(c.get("border-left-color"), Some(&gray()));
    }

    #[test]
    fn table_mapping_case5_frame_hsides_inline_style() {
        // Case 5: frame=hsides hides the left style; the inline
        // border-right-style beats the frame's hidden on the right.
        let (mut doc, table, _td) = table_doc(&[("border", "10"), ("frame", "hsides")]);
        doc.set_attribute(table, "style", "border-right-style: solid");
        let s = style(&doc, table, "", &vp());
        assert_eq!(s.get_str("border-left-style"), Some("hidden"));
        assert_eq!(s.get_str("border-right-style"), Some("solid"));
        assert_eq!(
            s.get("border-right-width"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
        assert_eq!(s.get("border-right-color"), Some(&gray()));
    }

    #[test]
    fn table_hints_lose_to_author_rules() {
        // Presentational hints are author-origin at zero specificity:
        // any author declaration beats them, UA declarations do not.
        let (doc, table, _td) = table_doc(&[("border", "10")]);
        let s = style(&doc, table, "table { border-left-style: dotted; }", &vp());
        assert_eq!(s.get_str("border-left-style"), Some("dotted"));
        // The hint width survives where the author sheet is silent.
        assert_eq!(
            s.get("border-left-width"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
    }

    #[test]
    fn col_and_colgroup_width_attributes_map_to_width_hint() {
        // WHATWG HTML rendering § "Tables": the `width` attribute on
        // `col` and `colgroup` maps to the `width` presentational hint
        // (css/css-tables/tentative/colgroup-col.html tables 1-2).
        let mut doc = Document::new();
        let root = doc.root();
        let table = doc.create_element("table");
        doc.append_child(root, table);
        let colgroup = doc.create_element("colgroup");
        // w77g: `100px` is not a dimension under the strict
        // full-consumption grammar (`html_dimension`), so this
        // mapping case uses the bare number, which maps to the same
        // 100px hint as before.
        doc.set_attribute(colgroup, "width", "100");
        doc.append_child(table, colgroup);
        let col = doc.create_element("col");
        doc.set_attribute(col, "width", "50%");
        doc.append_child(colgroup, col);
        let cg = style(&doc, colgroup, "", &vp());
        assert_eq!(
            cg.get("width"),
            Some(&CssValue::Length(100.0, LengthUnit::Px))
        );
        let c = style(&doc, col, "", &vp());
        assert_eq!(c.get("width"), Some(&CssValue::Percentage(50.0)));
        // Author declarations still beat the hint.
        let cg = style(&doc, colgroup, "colgroup { width: 7px; }", &vp());
        assert_eq!(
            cg.get("width"),
            Some(&CssValue::Length(7.0, LengthUnit::Px))
        );
    }

    #[test]
    fn td_th_width_and_height_attributes_map_to_dimension_hints() {
        // WHATWG HTML rendering § "Tables": `td`/`th` `width` and
        // `height` map to the `width`/`height` properties (this is
        // what css/css-tables/fractional-percent-width.html needs:
        // `width="0.5%"` etc. on cells, not just on `col`).
        let (doc, _table, td) = table_doc(&[]);
        let mut doc = doc;
        doc.set_attribute(td, "width", "50%");
        let s = style(&doc, td, "", &vp());
        assert_eq!(s.get("width"), Some(&CssValue::Percentage(50.0)));
        let (doc, _table, td) = table_doc(&[]);
        let mut doc = doc;
        doc.set_attribute(td, "width", "120");
        let s = style(&doc, td, "", &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(120.0, LengthUnit::Px))
        );
        // Fractional percent, the WPT case: `width="0.5%"`.
        let (doc, _table, td) = table_doc(&[]);
        let mut doc = doc;
        doc.set_attribute(td, "width", "0.5%");
        let s = style(&doc, td, "", &vp());
        assert_eq!(s.get("width"), Some(&CssValue::Percentage(0.5)));
        // `height` maps the same way.
        let (doc, _table, td) = table_doc(&[]);
        let mut doc = doc;
        doc.set_attribute(td, "height", "40");
        let s = style(&doc, td, "", &vp());
        assert_eq!(
            s.get("height"),
            Some(&CssValue::Length(40.0, LengthUnit::Px))
        );
        // `th` maps identically to `td`.
        let mut doc = Document::new();
        let root = doc.root();
        let table = doc.create_element("table");
        doc.append_child(root, table);
        let tr = doc.create_element("tr");
        doc.append_child(table, tr);
        let th = doc.create_element("th");
        doc.set_attribute(th, "width", "25%");
        doc.append_child(tr, th);
        let s = style(&doc, th, "", &vp());
        assert_eq!(s.get("width"), Some(&CssValue::Percentage(25.0)));
    }

    #[test]
    fn td_width_and_height_zero_attributes_are_ignored() {
        // HTML rendering: the `td`/`th` dimension mappings ignore
        // zero values, so `width="0"` must leave the computed width
        // at the `auto` initial, not force a zero width.
        let (doc, _table, td) = table_doc(&[]);
        let mut doc = doc;
        doc.set_attribute(td, "width", "0");
        doc.set_attribute(td, "height", "0%");
        let s = style(&doc, td, "", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
        assert_eq!(s.get("height"), Some(&kw("auto")));
        // CSS `width: 0` still works alongside `width="0"`: the
        // dropped hint does not override an author declaration.
        let s = style(&doc, td, "td { width: 0px; height: 0px; }", &vp());
        assert_eq!(s.get("width"), Some(&CssValue::Length(0.0, LengthUnit::Px)));
        assert_eq!(
            s.get("height"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
    }

    #[test]
    fn tr_height_attribute_maps_to_height_hint() {
        // WHATWG HTML rendering § "Tables": `tr` `height` maps to
        // the `height` property.
        let mut doc = Document::new();
        let root = doc.root();
        let table = doc.create_element("table");
        doc.append_child(root, table);
        let tr = doc.create_element("tr");
        doc.set_attribute(tr, "height", "30");
        doc.append_child(table, tr);
        let s = style(&doc, tr, "", &vp());
        assert_eq!(
            s.get("height"),
            Some(&CssValue::Length(30.0, LengthUnit::Px))
        );
    }

    #[test]
    fn tr_height_zero_attribute_maps_to_zero() {
        // WHATWG HTML rendering § 15.3.8 gives `tr` `height` the
        // plain dimension-property mapping, NOT "(ignoring zero)"
        // (like the section rule, unlike `td`/`th`): `height="0"`
        // maps to `height: 0`.
        let mut doc = Document::new();
        let root = doc.root();
        let table = doc.create_element("table");
        doc.append_child(root, table);
        let tr = doc.create_element("tr");
        doc.set_attribute(tr, "height", "0");
        doc.append_child(table, tr);
        let s = style(&doc, tr, "", &vp());
        assert_eq!(
            s.get("height"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
    }

    #[test]
    fn table_width_and_height_zero_attributes_are_ignored() {
        // WHATWG HTML rendering § 15.3.8: `table` `width` and
        // `height` map to the dimension property (ignoring zero).
        // No existing test depended on a zero mapping (checked
        // before this change: only the `td` zero test existed).
        // Zero leaves the computed value at the `auto` initial.
        let (doc, table, _td) = table_doc(&[("width", "0"), ("height", "0%")]);
        let s = style(&doc, table, "", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
        assert_eq!(s.get("height"), Some(&kw("auto")));
        // CSS `width: 0` still works alongside `width="0"`: the
        // dropped hint does not override an author declaration.
        let s = style(&doc, table, "table { width: 0px; height: 0px; }", &vp());
        assert_eq!(s.get("width"), Some(&CssValue::Length(0.0, LengthUnit::Px)));
        assert_eq!(
            s.get("height"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // Nonzero values still map.
        let (doc, table, _td) = table_doc(&[("width", "50%"), ("height", "40")]);
        let s = style(&doc, table, "", &vp());
        assert_eq!(s.get("width"), Some(&CssValue::Percentage(50.0)));
        assert_eq!(
            s.get("height"),
            Some(&CssValue::Length(40.0, LengthUnit::Px))
        );
    }

    #[test]
    fn col_width_zero_attribute_maps_to_zero() {
        // WHATWG HTML rendering § 15.3.8 gives `col` `width` the
        // plain dimension-property mapping, NOT "(ignoring zero)"
        // (contrast `table`/`td`/`th` above): `width="0"` maps to
        // `width: 0` rather than being dropped.
        let mut doc = Document::new();
        let root = doc.root();
        let table = doc.create_element("table");
        doc.append_child(root, table);
        let col = doc.create_element("col");
        doc.set_attribute(col, "width", "0");
        doc.append_child(table, col);
        let s = style(&doc, col, "", &vp());
        assert_eq!(s.get("width"), Some(&CssValue::Length(0.0, LengthUnit::Px)));
        let mut doc = Document::new();
        let root = doc.root();
        let table = doc.create_element("table");
        doc.append_child(root, table);
        let col = doc.create_element("col");
        doc.set_attribute(col, "width", "0%");
        doc.append_child(table, col);
        let s = style(&doc, col, "", &vp());
        assert_eq!(s.get("width"), Some(&CssValue::Percentage(0.0)));
    }

    #[test]
    fn w77g_html_dimension_rejects_partial_and_junk_values() {
        // WHATWG HTML §2.3.4.4 dimension grammar in its strict
        // full-consumption form: ASCII digits, one optional fraction
        // (`.` followed by digits), one optional `%` suffix, with
        // ASCII whitespace trimmed from both ends — anything else
        // maps to nothing. Fail-before: the lenient prefix parser
        // mapped every rejected case below to a length (it stopped
        // at the first non-digit/dot character and ignored the rest).
        // `td` uses the nonzero mapping, so a rejected value leaves
        // `width` at its `auto` initial.
        for v in [
            "100abc", "100 px", "10.5.2", "100px", ".5", "+100", "-100", "1e3", "100%%", "100.",
            "", "abc", "10.5.2%",
        ] {
            let (mut doc, _table, td) = table_doc(&[]);
            doc.set_attribute(td, "width", v);
            let s = style(&doc, td, "", &vp());
            assert_eq!(s.get("width"), Some(&kw("auto")), "{v:?} must not map");
        }
        // Valid forms still map through the same nonzero path.
        for (v, want) in [
            ("100", CssValue::Length(100.0, LengthUnit::Px)),
            ("100%", CssValue::Percentage(100.0)),
            ("10.5", CssValue::Length(10.5, LengthUnit::Px)),
            ("0.5%", CssValue::Percentage(0.5)),
            (" 100 ", CssValue::Length(100.0, LengthUnit::Px)),
            ("100% ", CssValue::Percentage(100.0)),
            ("007", CssValue::Length(7.0, LengthUnit::Px)),
        ] {
            let (mut doc, _table, td) = table_doc(&[]);
            doc.set_attribute(td, "width", v);
            let s = style(&doc, td, "", &vp());
            assert_eq!(s.get("width"), Some(&want), "{v:?} must map to {want:?}");
        }
        // The plain mapping (`col`) rejects junk the same way and
        // still maps valid forms, including zero (plain, not nonzero).
        let mut doc = Document::new();
        let root = doc.root();
        let table = doc.create_element("table");
        doc.append_child(root, table);
        let col = doc.create_element("col");
        doc.set_attribute(col, "width", "100abc");
        doc.append_child(table, col);
        let s = style(&doc, col, "", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")), "col 100abc must not map");
        let mut doc = Document::new();
        let root = doc.root();
        let table = doc.create_element("table");
        doc.append_child(root, table);
        let col = doc.create_element("col");
        doc.set_attribute(col, "width", "50%");
        doc.append_child(table, col);
        let s = style(&doc, col, "", &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Percentage(50.0)),
            "col 50% must map"
        );
    }

    #[test]
    fn colgroup_width_zero_attribute_maps_to_zero() {
        // Like `col` (and unlike `table`/`td`/`th`), `colgroup`
        // `width` is the plain dimension-property mapping: `width="0"`
        // maps to `width: 0` rather than being dropped.
        let mut doc = Document::new();
        let root = doc.root();
        let table = doc.create_element("table");
        doc.append_child(root, table);
        let colgroup = doc.create_element("colgroup");
        doc.set_attribute(colgroup, "width", "0");
        doc.append_child(table, colgroup);
        let s = style(&doc, colgroup, "", &vp());
        assert_eq!(s.get("width"), Some(&CssValue::Length(0.0, LengthUnit::Px)));
        let mut doc = Document::new();
        let root = doc.root();
        let table = doc.create_element("table");
        doc.append_child(root, table);
        let colgroup = doc.create_element("colgroup");
        doc.set_attribute(colgroup, "width", "0%");
        doc.append_child(table, colgroup);
        let s = style(&doc, colgroup, "", &vp());
        assert_eq!(s.get("width"), Some(&CssValue::Percentage(0.0)));
    }

    #[test]
    fn section_height_attributes_map_to_height_hints() {
        // WHATWG HTML rendering § 15.3.8: `thead`/`tbody`/`tfoot`
        // `height` maps to the dimension property `height`.
        for tag in ["thead", "tbody", "tfoot"] {
            let mut doc = Document::new();
            let root = doc.root();
            let table = doc.create_element("table");
            doc.append_child(root, table);
            let section = doc.create_element(tag);
            doc.set_attribute(section, "height", "30");
            doc.append_child(table, section);
            let s = style(&doc, section, "", &vp());
            assert_eq!(
                s.get("height"),
                Some(&CssValue::Length(30.0, LengthUnit::Px)),
                "{tag}"
            );
        }
    }

    #[test]
    fn section_height_zero_attribute_maps_to_zero() {
        // Same section, plain mapping (not "(ignoring zero)"), like
        // `tr`: `height="0"` maps to `height: 0`, unlike `td`/`th`
        // and `table` where zero is ignored.
        let mut doc = Document::new();
        let root = doc.root();
        let table = doc.create_element("table");
        doc.append_child(root, table);
        let tbody = doc.create_element("tbody");
        doc.set_attribute(tbody, "height", "0");
        doc.append_child(table, tbody);
        let s = style(&doc, tbody, "", &vp());
        assert_eq!(
            s.get("height"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // Section hints still coexist with the group (rules) hints.
        let mut doc = Document::new();
        let root = doc.root();
        let table = doc.create_element("table");
        doc.set_attribute(table, "rules", "groups");
        doc.append_child(root, table);
        let thead = doc.create_element("thead");
        doc.set_attribute(thead, "height", "25%");
        doc.append_child(table, thead);
        let s = style(&doc, thead, "", &vp());
        assert_eq!(s.get("height"), Some(&CssValue::Percentage(25.0)));
        assert_eq!(
            s.get("border-top-width"),
            Some(&CssValue::Length(1.0, LengthUnit::Px))
        );
    }

    #[test]
    fn cell_dimension_hints_leave_table_cell_hints_unaffected() {
        // A cell with its own dimension attributes still gets the
        // table-derived border/padding hints unchanged.
        let (doc, _table, td) = table_doc(&[("border", "10"), ("cellpadding", "3")]);
        let mut doc = doc;
        doc.set_attribute(td, "width", "50%");
        let s = style(&doc, td, "", &vp());
        assert_eq!(
            s.get("border-left-width"),
            Some(&CssValue::Length(1.0, LengthUnit::Px))
        );
        assert_eq!(s.get_str("border-left-style"), Some("inset"));
        assert_eq!(s.get("border-left-color"), Some(&gray()));
        assert_eq!(
            s.get("padding-left"),
            Some(&CssValue::Length(3.0, LengthUnit::Px))
        );
        assert_eq!(s.get("width"), Some(&CssValue::Percentage(50.0)));
    }

    #[test]
    fn table_cellspacing_and_cellpadding_map() {
        let (doc, table, td) = table_doc(&[("cellspacing", "7"), ("cellpadding", "3")]);
        let s = style(&doc, table, "", &vp());
        assert_eq!(
            s.get("border-spacing"),
            Some(&CssValue::Length(7.0, LengthUnit::Px))
        );
        let c = style(&doc, td, "", &vp());
        assert_eq!(
            c.get("padding-left"),
            Some(&CssValue::Length(3.0, LengthUnit::Px))
        );
    }

    #[test]
    fn ua_table_border_spacing_and_cell_padding_defaults() {
        // CSS 2.1 Appendix D / HTML rendering: `table { border-spacing: 2px }`
        // and `td, th { padding: 1px }`.
        let (doc, table, td) = table_doc(&[]);
        let s = style(&doc, table, "", &vp());
        assert_eq!(
            s.get("border-spacing"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        let c = style(&doc, td, "", &vp());
        assert_eq!(
            c.get("padding-left"),
            Some(&CssValue::Length(1.0, LengthUnit::Px))
        );
        // Author declarations still beat the UA defaults.
        let s = style(&doc, table, "table { border-spacing: 5px; }", &vp());
        assert_eq!(
            s.get("border-spacing"),
            Some(&CssValue::Length(5.0, LengthUnit::Px))
        );
        // Presentational hints (cellspacing/cellpadding) sit above UA too.
        let (doc, table, td) = table_doc(&[("cellspacing", "7"), ("cellpadding", "3")]);
        let s = style(&doc, table, "", &vp());
        assert_eq!(
            s.get("border-spacing"),
            Some(&CssValue::Length(7.0, LengthUnit::Px))
        );
        let c = style(&doc, td, "", &vp());
        assert_eq!(
            c.get("padding-left"),
            Some(&CssValue::Length(3.0, LengthUnit::Px))
        );
        // `cellpadding="0"` must still emit a hint: it beats the UA
        // `td { padding: 1px }` default (regression: the hint used to
        // be skipped for a zero value).
        let (doc, _table, td) = table_doc(&[("cellpadding", "0")]);
        let c = style(&doc, td, "", &vp());
        assert_eq!(
            c.get("padding-left"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
    }

    #[test]
    fn border_spacing_rejects_negative_lengths() {
        assert!(!is_valid_for_property(
            "border-spacing",
            &CssValue::Length(-5.0, LengthUnit::Px)
        ));
        assert!(!is_valid_for_property(
            "border-spacing",
            &CssValue::List(vec![
                CssValue::Length(10.0, LengthUnit::Px),
                CssValue::Length(-5.0, LengthUnit::Px),
            ])
        ));
        assert!(is_valid_for_property(
            "border-spacing",
            &CssValue::Length(5.0, LengthUnit::Px)
        ));
        // A negative declaration is dropped at computed-value time:
        // the computed value stays the initial 0px.
        let (doc, div, _) = sample();
        let s = style(&doc, div, "div { border-spacing: -5px; }", &vp());
        assert_eq!(
            s.get("border-spacing"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
    }

    #[test]
    fn border_spacing_negative_math_clamps_per_component() {
        // A math expression that evaluates negative clamps to `0px`
        // component-wise (CSS Values 4 range checking), instead of the
        // B8 re-validation dropping the whole list and losing a valid
        // component: at a 40px font `calc(10px - 0.5em)` is `-10px`, so
        // the pair computes to `0px 10px`, not the initial `0px`.
        let (doc, div, _) = sample();
        let zero = CssValue::Length(0.0, LengthUnit::Px);
        let ten = CssValue::Length(10.0, LengthUnit::Px);
        let s = style(
            &doc,
            div,
            "div { font-size: 40px; border-spacing: calc(10px - 0.5em) 10px; }",
            &vp(),
        );
        assert_eq!(
            s.get("border-spacing"),
            Some(&CssValue::List(vec![zero.clone(), ten.clone()]))
        );
        // A single negative-evaluating calc clamps to `0px`.
        let s = style(
            &doc,
            div,
            "div { border-spacing: calc(5px - 20px); }",
            &vp(),
        );
        assert_eq!(s.get("border-spacing"), Some(&zero));
        // Both components negative: both clamp.
        let s = style(
            &doc,
            div,
            "div { border-spacing: calc(5px - 20px) calc(-1px); }",
            &vp(),
        );
        assert_eq!(
            s.get("border-spacing"),
            Some(&CssValue::List(vec![zero.clone(), zero]))
        );
        // A non-calc pair is unchanged.
        let s = style(&doc, div, "div { border-spacing: 3px 10px; }", &vp());
        assert_eq!(
            s.get("border-spacing"),
            Some(&CssValue::List(vec![
                CssValue::Length(3.0, LengthUnit::Px),
                ten
            ]))
        );
    }

    #[test]
    fn border_spacing_pct_calc_invalid() {
        // Percentages are not valid `border-spacing` (CSS 2.1 §17.5.2):
        // a bare percentage is rejected, and so is a `calc()` with
        // percentage leaves (CSS Values and Units 4 §10.9/§10.11) —
        // even when the
        // percentage folds to zero. The declaration is dropped and the
        // computed value stays the initial 0px.
        let (doc, div, _) = sample();
        let zero = CssValue::Length(0.0, LengthUnit::Px);
        for decl in [
            "div { border-spacing: 10%; }",
            "div { border-spacing: calc(5px + 10%); }",
            "div { border-spacing: calc(5px + 0%); }",
            "div { border-spacing: 3px 10%; }",
            "div { border-spacing: calc(5px + 10%) 3px; }",
        ] {
            let s = style(&doc, div, decl, &vp());
            assert_eq!(s.get("border-spacing"), Some(&zero), "{decl}");
        }
        // Pure-length calc stays valid.
        let s = style(&doc, div, "div { border-spacing: calc(5px + 5px); }", &vp());
        assert_eq!(
            s.get("border-spacing"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
    }

    #[test]
    fn column_width_pct_calc_invalid() {
        // Percentages are not valid `column-width` (CSS Multicol 1 §3:
        // `<length> | auto`): a bare percentage is rejected, and so is
        // a `calc()` with percentage leaves (CSS Values and Units 4
        // §10.9/§10.11) —
        // even when the percentage folds to zero. The declaration is
        // dropped and the computed value stays the initial `auto`.
        let (doc, div, _) = sample();
        for decl in [
            ".a { column-width: 50%; }",
            ".a { column-width: calc(100px + 10%); }",
            ".a { column-width: calc(100px + 0%); }",
        ] {
            let s = style(&doc, div, decl, &vp());
            assert_eq!(s.get("column-width"), Some(&kw("auto")), "{decl}");
        }
        // Pure-length calc stays valid.
        let s = style(&doc, div, ".a { column-width: calc(100px + 10px); }", &vp());
        assert_eq!(
            s.get("column-width"),
            Some(&CssValue::Length(110.0, LengthUnit::Px))
        );
    }

    #[test]
    fn letter_spacing_pct_calc_invalid() {
        // Percentages are not valid `letter-spacing` (CSS Text 3 §9:
        // `<length>` only): a bare percentage is rejected, and so is a
        // `calc()` with percentage leaves (CSS Values 4 §8.1.4) — even
        // when the percentage folds to zero, and even through the new
        // wave-55 math functions (`abs(10%)` still contains a
        // percentage). The declaration is dropped and the computed
        // value stays the initial `normal`.
        let (doc, div, _) = sample();
        for decl in [
            "div { letter-spacing: 10%; }",
            "div { letter-spacing: calc(2px + 10%); }",
            "div { letter-spacing: calc(2px + 0%); }",
            "div { letter-spacing: abs(10%); }",
        ] {
            let s = style(&doc, div, decl, &vp());
            assert_eq!(s.get("letter-spacing"), Some(&kw("normal")), "{decl}");
        }
        // Pure-length calc stays valid.
        let s = style(&doc, div, "div { letter-spacing: calc(2px + 4px); }", &vp());
        assert_eq!(
            s.get("letter-spacing"),
            Some(&CssValue::Length(6.0, LengthUnit::Px))
        );
        // The new math functions stay valid when percentage-free.
        let s = style(&doc, div, "div { letter-spacing: abs(2px); }", &vp());
        assert_eq!(
            s.get("letter-spacing"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        // Negative lengths stay valid (tighter spacing is allowed).
        let s = style(&doc, div, "div { letter-spacing: -1px; }", &vp());
        assert_eq!(
            s.get("letter-spacing"),
            Some(&CssValue::Length(-1.0, LengthUnit::Px))
        );
    }

    #[test]
    fn word_spacing_pct_calc_invalid() {
        // Percentages are not valid `word-spacing` (CSS Text 3 §9:
        // `<length>` only): a bare percentage is rejected, and so is a
        // `calc()` with percentage leaves (CSS Values 4 §8.1.4) — even
        // when the percentage folds to zero. The declaration is dropped
        // and the computed value stays the initial `normal`.
        let (doc, div, _) = sample();
        for decl in [
            "div { word-spacing: 10%; }",
            "div { word-spacing: calc(2px + 10%); }",
            "div { word-spacing: calc(2px + 0%); }",
        ] {
            let s = style(&doc, div, decl, &vp());
            assert_eq!(s.get("word-spacing"), Some(&kw("normal")), "{decl}");
        }
        // Pure-length calc stays valid.
        let s = style(&doc, div, "div { word-spacing: calc(2px + 4px); }", &vp());
        assert_eq!(
            s.get("word-spacing"),
            Some(&CssValue::Length(6.0, LengthUnit::Px))
        );
        // The new math functions reject percentage leaves here too.
        let s = style(&doc, div, "div { word-spacing: abs(10%); }", &vp());
        assert_eq!(s.get("word-spacing"), Some(&kw("normal")));
        let s = style(&doc, div, "div { word-spacing: abs(2px); }", &vp());
        assert_eq!(
            s.get("word-spacing"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
    }

    #[test]
    fn specificity_beats_source_order() {
        let (doc, div, _) = sample();
        // Later rule has lower specificity: #d still wins.
        let s = style(&doc, div, "#d { color: red; } .a { color: blue; }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // Equal specificity: later rule wins.
        let s = style(&doc, div, ".a { color: red; } .a { color: blue; }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );
    }

    #[test]
    fn cascade_winners_resolve_via_atoms() {
        // Winner selection runs on property atoms now: the atom index
        // must be the `props::*` slot index, and last-wins must hold.
        assert_eq!(PropertyAtom::COLOR.index(), props::COLOR);
        assert_eq!(PropertyAtom::MARGIN_TOP.index(), props::MARGIN_TOP);
        let (doc, div, _) = sample();
        let s = style(
            &doc,
            div,
            ".a { color: red; margin-top: 1px; } #d { color: blue; }",
            &vp(),
        );
        // Specificity winner visible through both the name and the
        // atom-indexed slot reads.
        assert_eq!(s.get("color"), s.get_by_index(PropertyAtom::COLOR.index()));
        assert_eq!(
            s.get_by_index(props::COLOR),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );
        // Lower-specificity longhand still lands in its own slot.
        assert_eq!(
            s.get_by_index(props::MARGIN_TOP),
            s.get("margin-top"),
            "atom-indexed slot must equal name lookup"
        );
    }

    // ---- wave 50: sparse computed-style storage ----

    #[test]
    fn sparse_style_stores_only_deviating_slots() {
        let (doc, div, p) = sample();
        // No author CSS: the div deviates from the initials only by
        // its UA `display: block`; everything else reads back through
        // the initial-value fallback.
        let s = style(&doc, div, "", &vp());
        assert_eq!(s.get_by_index(props::DISPLAY), Some(&kw("block")));
        assert!(
            s.stored_slots() <= 2,
            "plain div stored {} slots",
            s.stored_slots()
        );
        for name in COMPUTED_PROPS {
            assert!(s.get(name).is_some(), "no computed value for {name}");
            if *name != "display" {
                assert_eq!(
                    s.get(name),
                    initial_value(name).as_ref(),
                    "{name} must read as its initial value when not deviating"
                );
            }
        }
        // The paragraph adds UA `margin: 1em 0` (top/bottom deviate;
        // the zero left/right margins expand as `Number(0)` rather than
        // the initial `Length(0, Px)`, so they are stored too — still
        // a handful of slots out of 181).
        let s = style(&doc, p, "", &vp());
        assert!(
            s.stored_slots() <= 6,
            "plain p stored {} slots",
            s.stored_slots()
        );
        // (The zero left/right margins compute to `Number(0)` from the
        // shorthand expansion — the sparse layer returns them verbatim.)
        assert_eq!(
            s.get_by_index(props::MARGIN_LEFT),
            Some(&CssValue::Number(0.0))
        );
        // Declared and inherited deviations are stored and read back.
        let s = style(&doc, p, "div { color: red; } p { margin-top: 5px; }", &vp());
        assert_eq!(s.get("color"), Some(&red()));
        assert_eq!(
            s.get_by_index(props::MARGIN_TOP),
            Some(&CssValue::Length(5.0, LengthUnit::Px))
        );
        assert!(
            s.stored_slots() <= 6,
            "styled p stored {}",
            s.stored_slots()
        );
    }

    #[test]
    fn sparse_marker_style_overwrites_sparse_slots() {
        // `::marker` starts from the element's sparse style and
        // overwrites single slots, including back to the initial
        // value; the element's own style is untouched.
        let mut doc = Document::new();
        let root = doc.root();
        let ul = doc.create_element("ul");
        doc.append_child(root, ul);
        let li = doc.create_element("li");
        doc.append_child(ul, li);
        let author = parse_stylesheet("li { color: blue; } li::marker { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let marker = styler.marker_style_for(li).expect("marker style");
        assert_eq!(marker.get("color"), Some(&red()));
        assert_eq!(styler.style_for(li).get("color"), Some(&blue()));
        // Every non-marker slot of the marker style still reads.
        for name in COMPUTED_PROPS {
            assert!(marker.get(name).is_some(), "marker: no value for {name}");
        }
    }

    // ---- wave 67: `content` + pseudo-element style infrastructure ----

    #[test]
    fn content_property_is_registered() {
        // `content` is a computed property with initial `normal`.
        let atom = PropertyAtom::lookup("content").expect("content atom");
        assert!(atom.is_computed());
        assert_eq!(atom.index(), props::CONTENT);
        assert_eq!(initial_value("content"), Some(kw("normal")));
        // Not inherited: an unstyled child reads the initial value even
        // when the parent specifies `content`.
        let (doc, _div, p) = sample();
        let s = style(&doc, p, "div { content: \"x\"; }", &vp());
        assert_eq!(s.get("content"), Some(&kw("normal")));
    }

    #[test]
    fn content_grammar_validation() {
        assert!(is_valid_for_property("content", &kw("normal")));
        assert!(is_valid_for_property("content", &kw("none")));
        assert!(is_valid_for_property(
            "content",
            &CssValue::String("c".into())
        ));
        assert!(!is_valid_for_property("content", &kw("bogus")));
        assert!(!is_valid_for_property(
            "content",
            &CssValue::Length(1.0, LengthUnit::Px)
        ));
        assert!(!is_valid_for_property("content", &CssValue::Number(1.0)));
    }

    #[test]
    fn content_string_parses_to_css_string() {
        // `content: "c"` rides the generic value parser into
        // `CssValue::String` (no `content`-specific parse arm needed).
        let decls = parse_declarations("content: \"c\"");
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].property.as_str(), "content");
        assert_eq!(decls[0].value, CssValue::String("c".into()));
    }

    #[test]
    fn content_computes_to_specified_value() {
        // Pass-through: no computed-value transform for `content`.
        let (doc, div, _) = sample();
        let s = style(&doc, div, "div { content: \"x\"; }", &vp());
        assert_eq!(s.get("content"), Some(&CssValue::String("x".into())));
        let s = style(&doc, div, "div { content: none; }", &vp());
        assert_eq!(s.get("content"), Some(&kw("none")));
        // Invalid at computed-value time: falls back to the initial.
        let s = style(&doc, div, "div { content: 12px; }", &vp());
        assert_eq!(s.get("content"), Some(&kw("normal")));
    }

    #[test]
    fn content_text_reads_string_only() {
        let (doc, div, _) = sample();
        let s = style(&doc, div, "div { content: \"hi\"; }", &vp());
        assert_eq!(content_text(&s), Some("hi"));
        let s = style(&doc, div, "div { content: none; }", &vp());
        assert_eq!(content_text(&s), None);
        let s = style(&doc, div, "", &vp());
        assert_eq!(content_text(&s), None);
    }

    #[test]
    fn before_after_atoms_mirror_marker_atoms_plus_content_and_display() {
        // `BEFORE_AFTER_ATOMS` must stay exactly `MARKER_ATOMS` plus
        // `content` and `display`: the pseudo cascade relies on the
        // relationship.
        let mut got: Vec<&str> = BEFORE_AFTER_ATOMS.iter().map(|a| a.as_str()).collect();
        got.sort_unstable();
        let mut want: Vec<&str> = MARKER_ATOMS.iter().map(|a| a.as_str()).collect();
        want.push("content");
        want.push("display");
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn pseudo_style_for_resets_display_to_inline() {
        // `::before` on a `display: block` div: the pseudo base style
        // must not inherit the originating element's `display`, while
        // the pseudo winners still apply.
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let author = parse_stylesheet("div::before { content: \"x\"; color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        // Sanity: the div itself is `display: block` (UA stylesheet).
        assert_eq!(
            styler.style_for(div).get_str_by_index(props::DISPLAY),
            Some("block")
        );
        let pseudo = styler
            .pseudo_style_for(div, "before", BEFORE_AFTER_ATOMS)
            .expect("::before style");
        assert_eq!(content_text(&pseudo), Some("x"));
        assert_eq!(pseudo.get("color"), Some(&red()));
        assert_eq!(pseudo.get_str_by_index(props::DISPLAY), Some("inline"));
    }

    #[test]
    fn pseudo_style_for_resets_origin_content_and_unicode_bidi() {
        // The originating element's non-inherited `content` /
        // `unicode-bidi` must not leak into the pseudo style: the base
        // clone carries the element's computed values, so without the
        // reset `div { content: "evil" }` plus
        // `div::before { color: red }` would synthesize a phantom box
        // (spec: the pseudo starts at `content: normal`, i.e. no box,
        // and `unicode-bidi: normal`).
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let author = parse_stylesheet(
            "div { content: \"evil\"; unicode-bidi: isolate; } div::before { color: red; }",
        );
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        // Sanity: the origin really does carry the specified values.
        let origin = styler.style_for(div);
        assert_eq!(
            origin.get("content"),
            Some(&CssValue::String("evil".into()))
        );
        assert_eq!(origin.get("unicode-bidi"), Some(&kw("isolate")));
        let pseudo = styler
            .pseudo_style_for(div, "before", BEFORE_AFTER_ATOMS)
            .expect("::before style");
        assert_eq!(pseudo.get("color"), Some(&red()));
        assert_eq!(content_text(&pseudo), None, "no phantom box");
        assert_eq!(pseudo.get("content"), Some(&kw("normal")));
        assert_eq!(pseudo.get("unicode-bidi"), Some(&kw("normal")));
    }

    #[test]
    fn pseudo_unset_on_non_inherited_resets_to_initial() {
        // `div::before { content: unset }`: `unset` acts as `initial`
        // on non-inherited properties, so the pseudo computes
        // `content: normal` (no box) instead of keeping the origin's
        // specified content. Same for `unicode-bidi`.
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let author = parse_stylesheet(
            "div { content: \"evil\"; unicode-bidi: isolate; } div::before { content: unset; unicode-bidi: unset; }",
        );
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let pseudo = styler
            .pseudo_style_for(div, "before", BEFORE_AFTER_ATOMS)
            .expect("::before style");
        assert_eq!(content_text(&pseudo), None, "unset -> normal -> no box");
        assert_eq!(pseudo.get("content"), Some(&kw("normal")));
        assert_eq!(pseudo.get("unicode-bidi"), Some(&kw("normal")));
    }

    #[test]
    fn pseudo_inherit_on_non_inherited_takes_origin_value() {
        // `div::before { content: inherit }`: `inherit` on a
        // non-inherited property takes the originating element's
        // computed value, overriding the pseudo's `normal` reset.
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let author =
            parse_stylesheet("div { content: \"evil\"; } div::before { content: inherit; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let pseudo = styler
            .pseudo_style_for(div, "before", BEFORE_AFTER_ATOMS)
            .expect("::before style");
        assert_eq!(content_text(&pseudo), Some("evil"));
    }

    #[test]
    fn pseudo_inherit_on_inherited_keeps_origin_value() {
        // `inherit` on an inherited property still takes the
        // originating element's value (the winner is dropped and the
        // base style carries it).
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let author =
            parse_stylesheet("div { color: red; } div::before { color: inherit; content: \"x\"; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let pseudo = styler
            .pseudo_style_for(div, "before", BEFORE_AFTER_ATOMS)
            .expect("::before style");
        assert_eq!(pseudo.get("color"), Some(&red()));
        assert_eq!(content_text(&pseudo), Some("x"));
    }

    #[test]
    fn pseudo_display_declaration_overrides_inline_reset() {
        // `div::before { display: block }` (clearfix): the `display`
        // winner must survive the pseudo's `inline` reset instead of
        // being dropped as a disallowed property.
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let author = parse_stylesheet("div::before { content: \"x\"; display: block; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let pseudo = styler
            .pseudo_style_for(div, "before", BEFORE_AFTER_ATOMS)
            .expect("::before style");
        assert_eq!(content_text(&pseudo), Some("x"));
        assert_eq!(pseudo.get_str_by_index(props::DISPLAY), Some("block"));
    }

    #[test]
    fn content_multi_string_concatenates() {
        // CSS Generated Content 3 §2.2: adjacent `<string>`s
        // concatenate. Element cascade path:
        let (doc, div, _) = sample();
        let s = style(&doc, div, "div { content: \"a\" \"b\"; }", &vp());
        assert_eq!(s.get("content"), Some(&CssValue::String("ab".into())));
        // Pseudo path (`Styler::pseudo_style_for` + `content_text`):
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let author = parse_stylesheet("div::before { content: \"a\" \"b\"; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let pseudo = styler
            .pseudo_style_for(div, "before", BEFORE_AFTER_ATOMS)
            .expect("::before style");
        assert_eq!(content_text(&pseudo), Some("ab"));
        // Grammar: an all-string list is valid; any other list shape
        // (numbers, mixed, empty) is not.
        assert!(is_valid_for_property(
            "content",
            &CssValue::List(vec![
                CssValue::String("a".into()),
                CssValue::String("b".into())
            ])
        ));
        assert!(!is_valid_for_property(
            "content",
            &CssValue::List(vec![CssValue::String("a".into()), CssValue::Number(1.0)])
        ));
        assert!(!is_valid_for_property("content", &CssValue::List(vec![])));
    }

    #[test]
    fn content_attr_counter_grammar() {
        // `attr()` / `counter()` / `counters()` parse to dedicated
        // shapes and are valid `content`, bare and in mixed lists.
        let attr = || {
            CssValue::Attr(Box::new(AttrData {
                name: "data-x".into(),
                fallback: None,
            }))
        };
        let counter = || {
            CssValue::Counter(Box::new(CounterData {
                name: "n".into(),
                style: None,
                separator: None,
            }))
        };
        assert!(is_valid_for_property("content", &attr()));
        assert!(is_valid_for_property("content", &counter()));
        assert!(is_valid_for_property(
            "content",
            &CssValue::List(vec![
                CssValue::String("a".into()),
                attr(),
                CssValue::String("b".into()),
                counter(),
            ])
        ));
        // Non-content shapes still rejected.
        assert!(!is_valid_for_property(
            "content",
            &CssValue::List(vec![
                CssValue::String("a".into()),
                CssValue::Url("u".into())
            ])
        ));
        assert!(!is_valid_for_property(
            "content",
            &CssValue::Url("u".into())
        ));
    }

    #[test]
    fn content_mixed_list_folds_string_runs() {
        // Adjacent `<string>`s concatenate even beside `attr()` /
        // `counter()` items; the function items survive the fold for
        // element-scoped resolution at box-synthesis time.
        let attr = || {
            CssValue::Attr(Box::new(AttrData {
                name: "data-x".into(),
                fallback: None,
            }))
        };
        let folded = fold_content_value(CssValue::List(vec![
            CssValue::String("a".into()),
            CssValue::String("b".into()),
            attr(),
            CssValue::String("c".into()),
        ]));
        assert_eq!(
            folded,
            CssValue::List(vec![
                CssValue::String("ab".into()),
                attr(),
                CssValue::String("c".into()),
            ])
        );
        // End-to-end: the folded mixed list computes and stays valid.
        let (doc, div, _) = sample();
        let s = style(
            &doc,
            div,
            "div { content: \"a\" \"b\" attr(data-x); }",
            &vp(),
        );
        assert!(is_valid_for_property(
            "content",
            s.get("content").expect("content computes")
        ));
    }

    #[test]
    fn quirks_font_style_stops_at_table_boundary() {
        // WHATWG Quirks §3.11 as probed by WPT `table-quirks.html`:
        // a table wrapped in `font-style: italic` computes `normal`
        // for its cells in full quirks mode only; limited-quirks and
        // no-quirks documents propagate `italic` (the limited-quirks
        // twin asserts `italic`). Other inherited properties are
        // untouched: `color` still propagates into the table in all
        // modes, and an explicit `font-style` on the table itself
        // still wins in quirks mode.
        let body =
            "<div style=\"font-style:italic;color:red\"><table><tr><td>x</td></tr></table></div>";
        for (html, expected) in [
            (
                format!(
                    "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\"><html><body>{body}</body></html>"
                ),
                "normal",
            ),
            (
                format!("<!DOCTYPE html><html><body>{body}</body></html>"),
                "italic",
            ),
            (
                format!(
                    "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\"><html><body>{body}</body></html>"
                ),
                "italic",
            ),
        ] {
            let (doc, _) = Document::parse_html(html.as_bytes());
            let td = doc.query_selector("td").unwrap().expect("td");
            let s = compute_style(&doc, td, &[], &ua(), &vp());
            assert_eq!(s.get_str("font-style"), Some(expected));
            assert_eq!(s.get("color"), Some(&red()));
        }
        // Explicit style on the table wins even in quirks mode.
        let (doc, _) = Document::parse_html(
            b"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\"><html><body><div style=\"font-style:italic\"><table style=\"font-style:oblique\"><tr><td>x</td></tr></table></div></body></html>",
        );
        let td = doc.query_selector("td").unwrap().expect("td");
        let s = compute_style(&doc, td, &[], &ua(), &vp());
        assert_eq!(s.get_str("font-style"), Some("oblique"));
    }

    #[test]
    fn counter_properties_validate_and_compute() {
        // Initial `none`, not inherited.
        assert_eq!(initial_value("counter-reset"), Some(kw("none")));
        assert_eq!(initial_value("counter-increment"), Some(kw("none")));
        assert!(!is_inherited("counter-reset"));
        assert!(!is_inherited("counter-increment"));
        // Grammar: `[ <custom-ident> <integer>? ]+ | none`.
        assert!(is_valid_for_property("counter-reset", &kw("none")));
        let pair = |n: &str| CssValue::Keyword(n.into());
        assert!(is_valid_for_property(
            "counter-reset",
            &CssValue::List(vec![pair("chapter")])
        ));
        assert!(is_valid_for_property(
            "counter-increment",
            &CssValue::List(vec![pair("chapter"), CssValue::Number(2.0)])
        ));
        assert!(is_valid_for_property(
            "counter-reset",
            &CssValue::List(vec![
                pair("a"),
                CssValue::Number(1.0),
                pair("b"),
                pair("c"),
                CssValue::Number(-3.0),
            ])
        ));
        // Non-integer adjustments, bare numbers, and `none` as a name
        // are rejected.
        assert!(!is_valid_for_property(
            "counter-reset",
            &CssValue::List(vec![pair("a"), CssValue::Number(2.5)])
        ));
        assert!(!is_valid_for_property(
            "counter-reset",
            &CssValue::List(vec![CssValue::Number(1.0)])
        ));
        assert!(!is_valid_for_property(
            "counter-reset",
            &CssValue::List(vec![pair("a"), pair("none")])
        ));
        // A bare number is not a valid adjustment list.
        assert!(!is_valid_for_property(
            "counter-reset",
            &CssValue::Number(1.0)
        ));
        // Computed values ride through the cascade.
        let (doc, div, _) = sample();
        let s = style(&doc, div, "div { counter-reset: chapter 2; }", &vp());
        assert_eq!(
            counter_adjustments(s.get("counter-reset"), 0),
            vec![("chapter".to_string(), 2)]
        );
        let s = style(&doc, div, "div { counter-increment: chapter; }", &vp());
        assert_eq!(
            counter_adjustments(s.get("counter-increment"), 1),
            vec![("chapter".to_string(), 1)]
        );
        // Unspecified: initial `none` means no adjustments.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert!(counter_adjustments(s.get("counter-reset"), 0).is_empty());
    }

    #[test]
    fn pseudo_style_for_ignores_disallowed_properties() {
        // `margin-top` is in no pseudo's allowed set: the declaration
        // is dropped, not applied.
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let author = parse_stylesheet("div::after { content: \"x\"; margin-top: 5px; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let pseudo = styler
            .pseudo_style_for(div, "after", BEFORE_AFTER_ATOMS)
            .expect("::after style");
        assert_eq!(content_text(&pseudo), Some("x"));
        assert_eq!(
            pseudo.get("margin-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px)),
            "disallowed properties must not leak into the pseudo style"
        );
    }

    #[test]
    fn pseudo_style_for_marker_keeps_origin_display() {
        // `::marker` via the generalized entry: same memoized result as
        // `marker_style_for`, and no `display` reset (unlike
        // `::before`/`::after`).
        let mut doc = Document::new();
        let root = doc.root();
        let ul = doc.create_element("ul");
        doc.append_child(root, ul);
        let li = doc.create_element("li");
        doc.append_child(ul, li);
        let author = parse_stylesheet("li::marker { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let via_marker = styler.marker_style_for(li).expect("marker style");
        let via_pseudo = styler
            .pseudo_style_for(li, "marker", MARKER_ATOMS)
            .expect("pseudo marker style");
        assert_eq!(via_marker.get("color"), via_pseudo.get("color"));
        assert_eq!(
            via_pseudo.get_str_by_index(props::DISPLAY),
            styler.style_for(li).get_str_by_index(props::DISPLAY),
            "::marker must keep the originating element's display"
        );
    }

    #[test]
    fn marker_memo_shares_rc_across_identical_items() {
        // Two `<li>`s matching the same `::marker` rules share one
        // `Rc<ComputedStyle>` instead of each paying the pseudo cascade
        // plus a base-style clone.
        let mut doc = Document::new();
        let root = doc.root();
        let ul = doc.create_element("ul");
        doc.append_child(root, ul);
        let li1 = doc.create_element("li");
        doc.append_child(ul, li1);
        let li2 = doc.create_element("li");
        doc.append_child(ul, li2);
        let author = parse_stylesheet("li::marker { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let a = styler.marker_style_for_shared(li1).expect("marker style 1");
        let b = styler.marker_style_for_shared(li2).expect("marker style 2");
        assert!(
            Rc::ptr_eq(&a, &b),
            "identical items must share one marker Rc"
        );
        assert_eq!(a.get("color"), Some(&red()));
        // A repeat query for the same node hits the memo too.
        let a2 = styler
            .marker_style_for_shared(li1)
            .expect("marker style 1 again");
        assert!(Rc::ptr_eq(&a, &a2));
        // The owned entry point still works (clone of the shared Rc).
        let owned = styler.marker_style_for(li1).expect("owned marker style");
        assert_eq!(owned.get("color"), Some(&red()));
    }

    #[test]
    fn marker_memo_distinguishes_different_winners() {
        let mut doc = Document::new();
        let root = doc.root();
        let ul = doc.create_element("ul");
        doc.append_child(root, ul);
        let li1 = doc.create_element("li");
        doc.set_attribute(li1, "class", "a");
        doc.append_child(ul, li1);
        let li2 = doc.create_element("li");
        doc.set_attribute(li2, "class", "b");
        doc.append_child(ul, li2);
        let author = parse_stylesheet("li.a::marker { color: red; } li.b::marker { color: blue; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let a = styler.marker_style_for_shared(li1).expect("marker style a");
        let b = styler.marker_style_for_shared(li2).expect("marker style b");
        assert!(!Rc::ptr_eq(&a, &b), "different winners must not share");
        assert_eq!(a.get("color"), Some(&red()));
        assert_eq!(b.get("color"), Some(&blue()));
    }

    #[test]
    fn marker_memo_retires_on_epoch_bump() {
        let mut doc = Document::new();
        let root = doc.root();
        let ul = doc.create_element("ul");
        doc.append_child(root, ul);
        let li = doc.create_element("li");
        doc.append_child(ul, li);
        let author = parse_stylesheet("li::marker { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let a = styler.marker_style_for_shared(li).expect("marker style");
        // A state change bumps the style-pass epoch, retiring the memo.
        styler.set_element_state(
            li,
            ElementState {
                hover: true,
                ..Default::default()
            },
        );
        let b = styler
            .marker_style_for_shared(li)
            .expect("marker style after bump");
        assert!(
            !Rc::ptr_eq(&a, &b),
            "epoch bump must retire the marker memo"
        );
        assert_eq!(b.get("color"), Some(&red()));
    }

    #[test]
    fn pseudo_memo_shares_rc_across_identical_elements() {
        // Two `<div>`s matching the same `::before` rules share one
        // `Rc<ComputedStyle>` instead of each paying the pseudo cascade
        // plus a base-style clone.
        let mut doc = Document::new();
        let root = doc.root();
        // NOTE: a document node takes only one element child, so the
        // pair lives under a container div.
        let wrap = doc.create_element("div");
        doc.append_child(root, wrap);
        let div1 = doc.create_element("div");
        doc.append_child(wrap, div1);
        let div2 = doc.create_element("div");
        doc.append_child(wrap, div2);
        let author = parse_stylesheet("div::before { content: \"x\"; color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let a = styler
            .pseudo_style_for_shared(div1, "before", BEFORE_AFTER_ATOMS)
            .expect("before style 1");
        let b = styler
            .pseudo_style_for_shared(div2, "before", BEFORE_AFTER_ATOMS)
            .expect("before style 2");
        assert!(
            Rc::ptr_eq(&a, &b),
            "identical elements must share one ::before Rc"
        );
        assert_eq!(a.get("color"), Some(&red()));
        // A repeat query for the same node hits the memo too.
        let a2 = styler
            .pseudo_style_for_shared(div1, "before", BEFORE_AFTER_ATOMS)
            .expect("before style 1 again");
        assert!(Rc::ptr_eq(&a, &a2));
        // The owned entry point still works (clone of the shared Rc).
        let owned = styler
            .pseudo_style_for(div1, "before", BEFORE_AFTER_ATOMS)
            .expect("owned before style");
        assert_eq!(owned.get("color"), Some(&red()));
        assert_eq!(content_text(&owned), Some("x"));
    }

    #[test]
    fn pseudo_memo_share_key_probe_implies_equal_winners() {
        // Proof of the cheap pre-winners probe's soundness: two
        // elements with equal style-sharing keys share one `::before`
        // `Rc` *without* the second computing winners — and the shared
        // style is correct even when the winners resolve `var()`
        // against the custom-property environment (the second element
        // adopts the donor's env via `share_adopt`, so the resolved
        // winner is identical).
        let mut doc = Document::new();
        let root = doc.root();
        // NOTE: a document node takes only one element child, so the
        // pair lives under a container div carrying the custom
        // property both siblings inherit.
        let wrap = doc.create_element("div");
        doc.set_attribute(wrap, "style", "--c: red");
        doc.append_child(root, wrap);
        let div1 = doc.create_element("div");
        doc.append_child(wrap, div1);
        let div2 = doc.create_element("div");
        doc.append_child(wrap, div2);
        let author = parse_stylesheet("div div::before { content: \"x\"; color: var(--c); }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let epoch = styler.epoch.get();
        let k1 = styler.share_key(div1, epoch).expect("sharing enabled");
        let k2 = styler.share_key(div2, epoch).expect("sharing enabled");
        assert_eq!(k1, k2, "test setup: the siblings must share one key");
        let a = styler
            .pseudo_style_for_shared(div1, "before", BEFORE_AFTER_ATOMS)
            .expect("before style 1");
        // `var(--c)` resolved against the inherited environment.
        assert_eq!(a.get("color"), Some(&red()));
        let b = styler
            .pseudo_style_for_shared(div2, "before", BEFORE_AFTER_ATOMS)
            .expect("before style 2");
        assert!(
            Rc::ptr_eq(&a, &b),
            "equal share keys must share one ::before Rc via the cheap probe"
        );
        assert_eq!(b.get("color"), Some(&red()));
    }

    #[test]
    fn pseudo_memo_share_key_probe_distinguishes_env() {
        // The converse: siblings under *different* custom-property
        // environments have different sharing keys, so the cheap probe
        // cannot hand one sibling's `var()`-resolved style to the other.
        let mut doc = Document::new();
        let root = doc.root();
        let wrap1 = doc.create_element("div");
        doc.set_attribute(wrap1, "style", "--c: red");
        doc.append_child(root, wrap1);
        // NOTE: a document node takes only one element child, so the
        // second subtree hangs off the first wrapper.
        let wrap2 = doc.create_element("div");
        doc.set_attribute(wrap2, "style", "--c: blue");
        doc.append_child(wrap1, wrap2);
        let div1 = doc.create_element("div");
        doc.append_child(wrap1, div1);
        let div2 = doc.create_element("div");
        doc.append_child(wrap2, div2);
        let author = parse_stylesheet("div div::before { content: \"x\"; color: var(--c); }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let epoch = styler.epoch.get();
        let k1 = styler.share_key(div1, epoch).expect("sharing enabled");
        let k2 = styler.share_key(div2, epoch).expect("sharing enabled");
        assert_ne!(k1, k2, "test setup: different envs must key apart");
        let a = styler
            .pseudo_style_for_shared(div1, "before", BEFORE_AFTER_ATOMS)
            .expect("before style 1");
        let b = styler
            .pseudo_style_for_shared(div2, "before", BEFORE_AFTER_ATOMS)
            .expect("before style 2");
        assert!(!Rc::ptr_eq(&a, &b), "different envs must not share");
        assert_eq!(a.get("color"), Some(&red()));
        assert_eq!(b.get("color"), Some(&blue()));
    }

    #[test]
    fn pseudo_memo_distinguishes_before_vs_after() {
        // `::before` and `::after` winners are computed separately, so
        // the pseudo name joins the memo key: identical elements with
        // different per-pseudo rules must not share.
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let author =
            parse_stylesheet("div::before { content: \"b\"; } div::after { content: \"a\"; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let before = styler
            .pseudo_style_for_shared(div, "before", BEFORE_AFTER_ATOMS)
            .expect("before style");
        let after = styler
            .pseudo_style_for_shared(div, "after", BEFORE_AFTER_ATOMS)
            .expect("after style");
        assert!(
            !Rc::ptr_eq(&before, &after),
            "::before and ::after must not share a memo entry"
        );
        assert_eq!(content_text(&before), Some("b"));
        assert_eq!(content_text(&after), Some("a"));
    }

    #[test]
    fn pseudo_memo_distinguishes_different_winners() {
        let mut doc = Document::new();
        let root = doc.root();
        // NOTE: a document node takes only one element child, so the
        // pair lives under a container div.
        let wrap = doc.create_element("div");
        doc.append_child(root, wrap);
        let div1 = doc.create_element("div");
        doc.set_attribute(div1, "class", "a");
        doc.append_child(wrap, div1);
        let div2 = doc.create_element("div");
        doc.set_attribute(div2, "class", "b");
        doc.append_child(wrap, div2);
        let author = parse_stylesheet(
            "div.a::before { content: \"x\"; color: red; } div.b::before { content: \"x\"; color: blue; }",
        );
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let a = styler
            .pseudo_style_for_shared(div1, "before", BEFORE_AFTER_ATOMS)
            .expect("before style a");
        let b = styler
            .pseudo_style_for_shared(div2, "before", BEFORE_AFTER_ATOMS)
            .expect("before style b");
        assert!(!Rc::ptr_eq(&a, &b), "different winners must not share");
        assert_eq!(a.get("color"), Some(&red()));
        assert_eq!(b.get("color"), Some(&blue()));
    }

    #[test]
    fn pseudo_memo_retires_on_epoch_bump() {
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let author = parse_stylesheet("div::before { content: \"x\"; color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let a = styler
            .pseudo_style_for_shared(div, "before", BEFORE_AFTER_ATOMS)
            .expect("before style");
        // A state change bumps the style-pass epoch, retiring the memo.
        styler.set_element_state(
            div,
            ElementState {
                hover: true,
                ..Default::default()
            },
        );
        let b = styler
            .pseudo_style_for_shared(div, "before", BEFORE_AFTER_ATOMS)
            .expect("before style after bump");
        assert!(
            !Rc::ptr_eq(&a, &b),
            "epoch bump must retire the pseudo memo"
        );
        assert_eq!(b.get("color"), Some(&red()));
    }

    #[test]
    fn sparse_memo_distinguishes_unset_from_initial() {
        // An unset inherited property on a chain that tops out must
        // memoize as unset (`None`), not as the initial value:
        // `font_size_px_for` reads unset as "use the parent's px" and
        // the initial `medium` as 16px, so confusing the two would
        // break font-size inheritance. After a full style pass the
        // same slot reads as the initial-filled form instead.
        let (doc, div, p) = sample();
        let author = parse_stylesheet("");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let font_idx = font_size_prop_idx();
        assert_eq!(
            styler.resolved_prop(p, font_idx, "font-size"),
            None,
            "unset font-size must stay unset through the cold climb"
        );
        assert_eq!(styler.memo_get(p, font_idx), Some(None));
        let _ = styler.style_for(div);
        let _ = styler.style_for(p);
        assert_eq!(
            styler.memo_get(p, font_idx),
            Some(initial_rc(font_idx)),
            "after a full pass the slot is initial-filled"
        );
        // The parent-ready probe the warm path relies on.
        let resolved = styler.resolved.borrow();
        let set = resolved.probe(p, styler.epoch.get()).expect("memo present");
        assert!(set.is_complete());
    }

    #[test]
    fn sparse_storage_sizes() {
        // The style value itself stays one box header + one word, and
        // the per-node memo is a bitset plus a vec header — versus the
        // old 1464 heap bytes per style and 2928 heap bytes per memo
        // (185 slots * 8 / 16 bytes), now paid only per deviating
        // slot at 16 bytes each.
        assert_eq!(std::mem::size_of::<ComputedStyle>(), 24);
        assert!(std::mem::size_of::<MemoSet>() <= 48);
        assert_eq!(
            PROP_COUNT * std::mem::size_of::<Option<Arc<CssValue>>>(),
            1480
        );
    }

    #[test]
    fn set_slot_none_clears_to_initial() {
        // B9: `set_slot(idx, None)` must actually clear the slot. The
        // property used here has a `Some` initial (`visible`), which
        // is exactly the case the old code got wrong: it treated
        // `None` as "not the initial form" and kept the stale entry.
        let (doc, div, _) = sample();
        let mut s = style(&doc, div, "div { overflow-x: hidden; }", &vp());
        assert_eq!(s.get_by_index(props::OVERFLOW_X), Some(&kw("hidden")));
        assert!(
            s.delta
                .binary_search_by_key(&props::OVERFLOW_X, |(i, _)| *i)
                .is_ok(),
            "the declared value must be stored before clearing"
        );
        let before = s.stored_slots();
        s.set_slot(props::OVERFLOW_X, None);
        assert_eq!(s.get_by_index(props::OVERFLOW_X), Some(&kw("visible")));
        assert_eq!(s.get("overflow-x"), Some(&kw("visible")));
        assert!(
            s.delta
                .binary_search_by_key(&props::OVERFLOW_X, |(i, _)| *i)
                .is_err(),
            "the delta must no longer contain the cleared slot"
        );
        assert_eq!(s.stored_slots(), before - 1);
        // Clearing an already-clear slot is a no-op, not a panic.
        s.set_slot(props::OVERFLOW_X, None);
        assert_eq!(s.get_by_index(props::OVERFLOW_X), Some(&kw("visible")));
        assert_eq!(s.stored_slots(), before - 1);
    }

    #[test]
    fn sparse_synthetic_document_allocation_evidence() {
        // 512 distinct elements (distinct classes defeat style
        // sharing, so every element owns its style and memo): the old
        // representations would allocate 512 * 1448 bytes of style
        // arrays plus 512 * 2896 bytes of memo arrays (~2.2 MB); the
        // sparse forms allocate 16 bytes per deviating slot.
        const N: usize = 512;
        let mut doc = Document::new();
        let root = doc.root();
        let wrap = doc.create_element("div");
        doc.append_child(root, wrap);
        let mut nodes = Vec::new();
        for i in 0..N {
            let el = doc.create_element("div");
            doc.set_attribute(el, "class", &format!("c{i}"));
            doc.append_child(wrap, el);
            nodes.push(el);
        }
        let author = parse_stylesheet(".c7 { color: red; margin-top: 3px; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let _ = styler.style_for(wrap);
        let mut style_heap = 0usize;
        let mut total_slots = 0usize;
        for &n in &nodes {
            let s = styler.style_for(n);
            total_slots += s.stored_slots();
            style_heap += s.stored_slots() * 16;
            assert!(s.get("display").is_some());
        }
        assert!(
            total_slots <= N * 4,
            "{total_slots} deviating slots over {N} elements"
        );
        assert!(style_heap <= N * 64, "{style_heap} style heap bytes");
        // Per-node memo entries across the whole slab.
        let resolved = styler.resolved.borrow();
        let mut memo_entries = 0usize;
        let mut memo_nodes = 0usize;
        for slot in resolved.slots.iter().flatten() {
            memo_nodes += 1;
            memo_entries += slot.value.entries.len();
        }
        assert!(memo_nodes >= N);
        assert!(
            memo_entries <= (N + 2) * 4,
            "{memo_entries} memo entries over {memo_nodes} nodes"
        );
    }

    #[test]
    fn important_beats_specificity() {
        let (doc, div, _) = sample();
        let s = style(
            &doc,
            div,
            "#d { color: red; } .a { color: blue !important; }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );
        // !important vs !important: specificity decides again.
        let s = style(
            &doc,
            div,
            ".a { color: red !important; } #d { color: blue !important; }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );
    }

    #[test]
    fn author_beats_ua() {
        let (doc, div, _) = sample();
        // UA says div { display: block }; author overrides.
        let s = style(&doc, div, "div { display: inline; }", &vp());
        assert_eq!(s.get_str("display"), Some("inline"));
        // But UA !important beats author normal.
        let ua_important = parse_stylesheet("div { display: block !important; }");
        let author = parse_stylesheet("div { display: inline; }");
        let s = compute_style(&doc, div, &[&author], &ua_important, &vp());
        assert_eq!(s.get_str("display"), Some("block"));
    }

    #[test]
    fn media_query_gating() {
        let (doc, div, _) = sample();
        let css = "@media (max-width: 100px) { div { color: red; } } div { color: blue; }";
        let narrow = Viewport { w: 50.0, h: 600.0 };
        let s = style(&doc, div, css, &narrow);
        // Both rules match at 50px; later one (blue) wins on source order.
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );
        let css2 =
            "@media (max-width: 100px) { div { color: red !important; } } div { color: blue; }";
        let s = style(&doc, div, css2, &narrow);
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // Wide viewport: media rule dropped, blue wins.
        let s = style(&doc, div, css2, &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );
    }

    #[test]
    fn multiple_author_sheets_in_order() {
        let (doc, div, _) = sample();
        let a1 = parse_stylesheet("div { color: red; }");
        let a2 = parse_stylesheet("div { color: blue; }");
        let s = compute_style(&doc, div, &[&a1, &a2], &ua(), &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );
    }

    #[test]
    fn inline_style_precedence() {
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        doc.set_attribute(div, "style", "color: green;");
        // Inline beats author normal, even with higher specificity.
        let s = style(&doc, div, "#x { color: red; } div { color: blue; }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 128,
                b: 0,
                a: 255
            }))
        );
        // Author !important beats inline normal.
        let s = style(&doc, div, "div { color: red !important; }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // Inline !important beats author !important.
        doc.set_attribute(div, "style", "color: green !important;");
        let s = style(&doc, div, "div { color: red !important; }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 128,
                b: 0,
                a: 255
            }))
        );
    }

    #[test]
    fn inheritance() {
        let (doc, _div, p) = sample();
        let s = style(&doc, p, "div { color: red; font-size: 20px; }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(20.0, LengthUnit::Px))
        );
        // Non-inherited properties do not inherit (UA sets no padding on p);
        // unset computes to the initial 0.
        let s = style(&doc, p, "div { padding-top: 5px; display: inline; }", &vp());
        assert_eq!(
            s.get("padding-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // UA default still applies on the child itself.
        assert_eq!(s.get_str("display"), Some("block"));
        // Directly specified value beats inherited.
        let s = style(&doc, p, "div { color: red; } p { color: blue; }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );
    }

    #[test]
    fn var_resolution() {
        let (doc, _div, p) = sample();
        // Custom property inherits from :root-like ancestor.
        let s = style(
            &doc,
            p,
            "div { --main: red; } p { color: var(--main); }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // Own custom property wins over inherited.
        let s = style(
            &doc,
            p,
            "div { --main: red; } p { --main: blue; color: var(--main); }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );
        // Fallback when undefined.
        let s = style(&doc, p, "p { color: var(--missing, green); }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 128,
                b: 0,
                a: 255
            }))
        );
        // No fallback and undefined: declaration invalid at computed-value
        // time -> property unset -> initial black.
        let s = style(&doc, p, "p { color: var(--missing); }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // Case-sensitive names.
        let s = style(
            &doc,
            p,
            "div { --Main: red; } p { color: var(--main, blue); }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );
        let s = style(
            &doc,
            p,
            "div { --Main: red; } p { color: var(--Main); }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // Chained var().
        let s = style(
            &doc,
            p,
            "div { --a: var(--b); --b: purple; } p { color: var(--a); }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 128,
                g: 0,
                b: 128,
                a: 255
            }))
        );
        // Cycle: falls back, then unset without fallback.
        let s = style(
            &doc,
            p,
            "p { --a: var(--a); color: var(--a, orange); }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 165,
                b: 0,
                a: 255
            }))
        );
        let s = style(
            &doc,
            p,
            "p { --a: var(--b); --b: var(--a); color: var(--a); }",
            &vp(),
        );
        // Cyclic: invalid at computed-value time -> initial black.
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            }))
        );
    }

    #[test]
    fn var_in_longhand() {
        let (doc, div, _) = sample();
        let s = style(&doc, div, "div { --w: 2px; margin-top: var(--w); }", &vp());
        assert_eq!(
            s.get("margin-top"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        // var() inside a shorthand the parser defers: the cascade
        // substitutes the reference and expands the shorthand afterwards,
        // so the longhands resolve (omitted components reset to initial).
        let s = style(
            &doc,
            div,
            "div { --w: 2px; border: var(--w) solid red; }",
            &vp(),
        );
        assert_eq!(
            s.get("border-top-width"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        assert_eq!(s.get_str("border-left-style"), Some("solid"));
        assert_eq!(
            s.get("border-right-color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // An unresolvable var() with no fallback drops the declaration
        // (border-top-width computes to initial `medium`).
        let s = style(&doc, div, "div { border: var(--nope) solid red; }", &vp());
        assert_eq!(s.get("border-top-width"), Some(&kw("medium")));
    }

    #[test]
    fn flex_shorthand_reaches_longhands() {
        let (doc, div, _) = sample();
        // `flex: 1` -> `1 1 0%`.
        let s = style(&doc, div, "div { flex: 1; }", &vp());
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get("flex-basis"), Some(&CssValue::Percentage(0.0)));
        // `flex: none` -> `0 0 auto`.
        let s = style(&doc, div, "div { flex: none; }", &vp());
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(0.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(0.0)));
        assert_eq!(s.get_str("flex-basis"), Some("auto"));
        // Three-value form.
        let s = style(&doc, div, "div { flex: 2 1 30px; }", &vp());
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(2.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(1.0)));
        assert_eq!(
            s.get("flex-basis"),
            Some(&CssValue::Length(30.0, LengthUnit::Px))
        );
        // `flex-basis` accepts the `content` keyword end-to-end.
        let s = style(&doc, div, "div { flex-basis: content; }", &vp());
        assert_eq!(s.get_str("flex-basis"), Some("content"));
        // An invalid shorthand sets nothing (the declaration is dropped);
        // the longhands compute to their initial values.
        let s = style(&doc, div, "div { flex: bogus; }", &vp());
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(0.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(1.0)));
        // A longhand still beats the shorthand at lower precedence:
        // the shorthand expands at parse time, so source order decides.
        let s = style(&doc, div, "div { flex: 1; flex-grow: 3; }", &vp());
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(3.0)));
        // Deferred: `var()` in the shorthand is substituted by the cascade
        // before expansion.
        let s = style(&doc, div, "div { --g: 2; flex: var(--g); }", &vp());
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(2.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get("flex-basis"), Some(&CssValue::Percentage(0.0)));
    }

    #[test]
    fn flex_shorthand_intrinsic_basis_keywords() {
        let (doc, div, _) = sample();
        // Deferred path: `var()` in the shorthand is substituted by the
        // cascade before expansion, so these exercise
        // `expand_flex_value` (the plain path goes through parse.rs's
        // mirror instead). Per CSS flexbox-1 §7.1 the basis slot takes
        // the full `flex-basis` grammar, including the intrinsic
        // keywords and `stretch` (the longhand's accepted set).
        let deferred = |css: &str| style(&doc, div, css, &vp());
        // Required forms.
        let s = deferred("div { --b: 1 1 max-content; flex: var(--b); }");
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get_str("flex-basis"), Some("max-content"));
        let s = deferred("div { --b: 1 1 min-content; flex: var(--b); }");
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get_str("flex-basis"), Some("min-content"));
        let s = deferred("div { --b: 0 0 fit-content; flex: var(--b); }");
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(0.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(0.0)));
        assert_eq!(s.get_str("flex-basis"), Some("fit-content"));
        let s = deferred("div { --b: 2 1 max-content; flex: var(--b); }");
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(2.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get_str("flex-basis"), Some("max-content"));
        // Single intrinsic basis -> `1 1 <basis>`; `stretch` matches the
        // longhand; two-value factor + intrinsic basis; basis-first order.
        let s = deferred("div { --b: max-content; flex: var(--b); }");
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get_str("flex-basis"), Some("max-content"));
        let s = deferred("div { --b: 1 1 stretch; flex: var(--b); }");
        assert_eq!(s.get_str("flex-basis"), Some("stretch"));
        let s = deferred("div { --b: 2 max-content; flex: var(--b); }");
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(2.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get_str("flex-basis"), Some("max-content"));
        let s = deferred("div { --b: max-content 2 1; flex: var(--b); }");
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(2.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get_str("flex-basis"), Some("max-content"));
        // Two intrinsic bases are still invalid: nothing is set, the
        // longhands compute to their initial values.
        let s = deferred("div { --b: max-content min-content; flex: var(--b); }");
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(0.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get_str("flex-basis"), Some("auto"));
        // Regression: previously valid forms, plain and deferred.
        let s = deferred("div { --b: none; flex: var(--b); }");
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(0.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(0.0)));
        assert_eq!(s.get_str("flex-basis"), Some("auto"));
        let s = deferred("div { --b: 3; flex: var(--b); }");
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(3.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get("flex-basis"), Some(&CssValue::Percentage(0.0)));
        let s = deferred("div { --b: 100px; flex: var(--b); }");
        assert_eq!(
            s.get("flex-basis"),
            Some(&CssValue::Length(100.0, LengthUnit::Px))
        );
        let s = deferred("div { --b: 2 1; flex: var(--b); }");
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(2.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get("flex-basis"), Some(&CssValue::Percentage(0.0)));
        let s = deferred("div { --b: 2 30px; flex: var(--b); }");
        assert_eq!(
            s.get("flex-basis"),
            Some(&CssValue::Length(30.0, LengthUnit::Px))
        );
        let s = style(&doc, div, "div { flex: 2 1 auto; }", &vp());
        assert_eq!(s.get_str("flex-basis"), Some("auto"));
        let s = style(&doc, div, "div { flex: 2 1 content; }", &vp());
        assert_eq!(s.get_str("flex-basis"), Some("content"));
        let s = style(&doc, div, "div { flex: 1 1 50%; }", &vp());
        assert_eq!(s.get("flex-basis"), Some(&CssValue::Percentage(50.0)));
    }

    #[test]
    fn break_spaces_white_space_computes() {
        let (doc, div, _) = sample();
        // `break-spaces` is a valid `white-space` keyword: it must compute
        // to itself, not fall back to `normal`.
        let s = style(&doc, div, "div { white-space: break-spaces; }", &vp());
        assert_eq!(s.get_str("white-space"), Some("break-spaces"));
    }

    #[test]
    fn ua_defaults_on_parsed_document() {
        let (doc, _) = Document::parse_html(
            b"<!DOCTYPE html><html><head><title>t</title></head><body><div><p>hi <span>x</span></p><img><button></button><table><caption>c</caption><tbody><tr><td>x</td></tr></tbody></table><ul><li></li></ul></div></body></html>",
        );
        let by_id = |tag: &str| doc.get_elements_by_tag_name(tag).first().unwrap();
        let disp = |tag: &str| {
            let n = by_id(tag);
            compute_style(&doc, n, &[], &ua(), &vp())
                .get_str("display")
                .unwrap()
                .to_string()
        };
        assert_eq!(disp("div"), "block");
        assert_eq!(disp("p"), "block");
        assert_eq!(disp("span"), "inline");
        assert_eq!(disp("head"), "none");
        assert_eq!(disp("title"), "none");
        assert_eq!(disp("img"), "inline-block");
        assert_eq!(disp("button"), "inline-block");
        assert_eq!(disp("table"), "table");
        assert_eq!(disp("tr"), "table-row");
        assert_eq!(disp("td"), "table-cell");
        assert_eq!(disp("tbody"), "table-row-group");
        assert_eq!(disp("caption"), "table-caption");
        assert_eq!(disp("ul"), "block");
        assert_eq!(disp("li"), "list-item");
        // h1 sizing + p margins + ul padding from the UA sheet.
        let (doc2, _) = Document::parse_html(b"<h1>hi</h1><ul><li>x</li></ul>");
        let h1 = doc2.get_elements_by_tag_name("h1").first().unwrap();
        let s = compute_style(&doc2, h1, &[], &ua(), &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(2.0, LengthUnit::Em))
        );
        assert_eq!(
            s.get("margin-top"),
            Some(&CssValue::Length(0.67, LengthUnit::Em))
        );
        let ul = doc2.get_elements_by_tag_name("ul").first().unwrap();
        let s = compute_style(&doc2, ul, &[], &ua(), &vp());
        assert_eq!(
            s.get("padding-left"),
            Some(&CssValue::Length(40.0, LengthUnit::Px))
        );
    }

    #[test]
    fn get_str() {
        let (doc, div, _) = sample();
        let s = style(
            &doc,
            div,
            "div { display: inline; font-family: \"hi\"; width: 5px; }",
            &vp(),
        );
        assert_eq!(s.get_str("display"), Some("inline"));
        assert_eq!(s.get_str("font-family"), Some("hi"));
        assert_eq!(s.get_str("width"), None); // not a string
        assert_eq!(s.get_str("nonexistent"), None);
    }

    #[test]
    fn grid_auto_flow_cascades() {
        let flow = |css: &str| {
            let (doc, div, _) = sample();
            style(&doc, div, css, &vp())
                .get_str("grid-auto-flow")
                .map(str::to_string)
        };
        // Single keywords, and the two-keyword form folded into one
        // space-joined keyword so layout's `get_str` read sees it.
        assert_eq!(flow("div { grid-auto-flow: row; }").as_deref(), Some("row"));
        assert_eq!(
            flow("div { grid-auto-flow: column; }").as_deref(),
            Some("column")
        );
        assert_eq!(
            flow("div { grid-auto-flow: dense; }").as_deref(),
            Some("dense")
        );
        assert_eq!(
            flow("div { grid-auto-flow: row dense; }").as_deref(),
            Some("row dense")
        );
        assert_eq!(
            flow("div { grid-auto-flow: dense column; }").as_deref(),
            Some("dense column")
        );
        // Case-insensitive (keywords are folded at parse time).
        assert_eq!(
            flow("div { grid-auto-flow: DENSE; }").as_deref(),
            Some("dense")
        );
        // `initial` resolves to the registered initial value (`row`).
        assert_eq!(
            flow("div { grid-auto-flow: initial; }").as_deref(),
            Some("row")
        );
        // Invalid forms drop the declaration, falling back to the initial
        // value (`row`) — never a garbage value.
        assert_eq!(
            flow("div { grid-auto-flow: row column; }").as_deref(),
            Some("row")
        );
        assert_eq!(
            flow("div { grid-auto-flow: dense dense; }").as_deref(),
            Some("row")
        );
        assert_eq!(
            flow("div { grid-auto-flow: row dense column; }").as_deref(),
            Some("row")
        );
        assert_eq!(flow("div { grid-auto-flow: 5px; }").as_deref(), Some("row"));
        // Unset: falls back to the registered initial value (`row`).
        assert_eq!(flow("div { display: grid; }").as_deref(), Some("row"));
    }

    #[test]
    fn grid_template_rejects_bare_auto_fill_fit() {
        let cols = |css: &str| {
            let (doc, div, _) = sample();
            style(&doc, div, css, &vp())
                .get("grid-template-columns")
                .cloned()
        };
        let rows = |css: &str| {
            let (doc, div, _) = sample();
            style(&doc, div, css, &vp())
                .get("grid-template-rows")
                .cloned()
        };
        // Invalid: a bare `auto-fill`/`auto-fit` keyword is not a track
        // size. The declaration is dropped and the property falls back
        // to its initial value (`none`).
        assert_eq!(
            cols("div { grid-template-columns: auto-fill 100px; }"),
            Some(kw("none"))
        );
        assert_eq!(
            cols("div { grid-template-columns: auto-fit minmax(100px, 1fr); }"),
            Some(kw("none"))
        );
        assert_eq!(
            rows("div { grid-template-rows: auto-fill 100px; }"),
            Some(kw("none"))
        );
        // A bare keyword alone is not a track listing either.
        assert_eq!(
            cols("div { grid-template-columns: auto-fill; }"),
            Some(kw("none"))
        );
        // Case-insensitive: keywords are folded at parse time.
        assert_eq!(
            cols("div { grid-template-columns: AUTO-FIT 100px; }"),
            Some(kw("none"))
        );
        // Same invalid listing smuggled through `var()` substitution.
        assert_eq!(
            cols("div { --t: auto-fill 100px; grid-template-columns: var(--t); }"),
            Some(kw("none"))
        );
        // Valid: `repeat(auto-fill, …)` is still accepted. The
        // single-track form keeps its dedicated `Repeat` descriptor
        // through substitution and normalization, so it is never
        // confused with the invalid bare listing above.
        assert_eq!(
            cols("div { grid-template-columns: repeat(auto-fill, 100px); }"),
            Some(CssValue::Repeat {
                mode: RepeatMode::Fill,
                tracks: vec![CssValue::Length(100.0, LengthUnit::Px)],
            })
        );
        // A single `minmax()` track also keeps its descriptor shape
        // (no flattening into `[Keyword(mode), track]`).
        assert_eq!(
            cols("div { grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); }"),
            Some(CssValue::Repeat {
                mode: RepeatMode::Fit,
                tracks: vec![CssValue::MinMax(
                    Box::new(CssValue::Length(100.0, LengthUnit::Px)),
                    Box::new(CssValue::Fr(1.0)),
                )],
            })
        );
        // Multi-track `repeat()`: the descriptor carries every track.
        assert_eq!(
            cols("div { grid-template-columns: repeat(auto-fill, 100px 1fr); }"),
            Some(CssValue::Repeat {
                mode: RepeatMode::Fill,
                tracks: vec![CssValue::Length(100.0, LengthUnit::Px), CssValue::Fr(1.0),],
            })
        );
        // `var()` inside the track list substitutes before layout sees
        // the descriptor: the declaration stays a valid repeat.
        assert_eq!(
            cols("div { --t: 100px; grid-template-columns: repeat(auto-fill, var(--t)); }"),
            Some(CssValue::Repeat {
                mode: RepeatMode::Fill,
                tracks: vec![CssValue::Length(100.0, LengthUnit::Px)],
            })
        );
        // A comma-separated substitution inside the track list is
        // invalid: commas never separate grid tracks (CSS Grid §7.1),
        // so the whole declaration drops and the property falls back
        // to `none`.
        assert_eq!(
            cols("div { --t: 100px, 200px; grid-template-columns: repeat(auto-fill, var(--t)); }"),
            Some(kw("none"))
        );
        // The whole descriptor smuggled through `var()`: still a
        // `Repeat`, still valid.
        assert_eq!(
            cols("div { --t: repeat(auto-fit, 50px); grid-template-columns: var(--t); }"),
            Some(CssValue::Repeat {
                mode: RepeatMode::Fit,
                tracks: vec![CssValue::Length(50.0, LengthUnit::Px)],
            })
        );
        // `calc()` inside the track list still evaluates in the
        // cascade (math evaluation recurses into the descriptor).
        assert_eq!(
            cols("div { grid-template-columns: repeat(auto-fill, calc(50px + 50px)); }"),
            Some(CssValue::Repeat {
                mode: RepeatMode::Fill,
                tracks: vec![CssValue::Length(100.0, LengthUnit::Px)],
            })
        );
        // Plain track lists and integer `repeat()` are unaffected.
        assert_eq!(
            cols("div { grid-template-columns: 100px 1fr; }"),
            Some(CssValue::List(vec![
                CssValue::Length(100.0, LengthUnit::Px),
                CssValue::Fr(1.0),
            ]))
        );
        assert_eq!(
            cols("div { grid-template-columns: repeat(2, 100px); }"),
            Some(CssValue::List(vec![
                CssValue::Length(100.0, LengthUnit::Px),
                CssValue::Length(100.0, LengthUnit::Px),
            ]))
        );
        // `none`, `subgrid`, and single track sizes still accepted
        // (a single track parses bare, not wrapped in a list).
        assert_eq!(
            cols("div { grid-template-columns: subgrid; }"),
            Some(kw("subgrid"))
        );
        assert_eq!(
            cols("div { grid-template-columns: auto; }"),
            Some(kw("auto"))
        );
        assert_eq!(
            cols("div { grid-template-columns: 100px; }"),
            Some(CssValue::Length(100.0, LengthUnit::Px))
        );
        assert_eq!(
            cols("div { grid-template-columns: minmax(50px, 120px); }"),
            Some(CssValue::MinMax(
                Box::new(CssValue::Length(50.0, LengthUnit::Px)),
                Box::new(CssValue::Length(120.0, LengthUnit::Px)),
            ))
        );
        // Non-track values are rejected, falling back to `none`.
        assert_eq!(
            cols("div { grid-template-columns: red; }"),
            Some(kw("none"))
        );
        assert_eq!(
            cols("div { grid-template-columns: 100px red; }"),
            Some(kw("none"))
        );
    }

    #[test]
    fn grid_template_rejects_comma_track_lists() {
        let cols = |css: &str| {
            let (doc, div, _) = sample();
            style(&doc, div, css, &vp())
                .get("grid-template-columns")
                .cloned()
        };
        let rows = |css: &str| {
            let (doc, div, _) = sample();
            style(&doc, div, css, &vp())
                .get("grid-template-rows")
                .cloned()
        };
        // Commas never separate grid tracks (CSS Grid §7.1): a
        // comma-separated listing is invalid and the declaration drops,
        // falling back to the initial value (`none`). The parser
        // preserves comma-ness (single-item comma groups nest), so the
        // validator can tell `100px, 200px` from `100px 200px`.
        assert_eq!(
            cols("div { grid-template-columns: 100px, 200px; }"),
            Some(kw("none"))
        );
        assert_eq!(
            rows("div { grid-template-rows: 100px, 200px; }"),
            Some(kw("none"))
        );
        // A mixed listing with any comma group is invalid too.
        assert_eq!(
            cols("div { grid-template-columns: 100px 200px, 300px; }"),
            Some(kw("none"))
        );
        // The same invalid listing smuggled through `var()`
        // substitution is rejected the same way.
        assert_eq!(
            cols("div { --t: 100px, 200px; grid-template-columns: var(--t); }"),
            Some(kw("none"))
        );
        // Space-separated listings are unaffected: flat lists of track
        // sizes stay valid.
        assert_eq!(
            cols("div { grid-template-columns: 100px 200px; }"),
            Some(CssValue::List(vec![
                CssValue::Length(100.0, LengthUnit::Px),
                CssValue::Length(200.0, LengthUnit::Px),
            ]))
        );
    }

    #[test]
    fn grid_template_rejects_nested_repeat() {
        let cols = |css: &str| {
            let (doc, div, _) = sample();
            style(&doc, div, css, &vp())
                .get("grid-template-columns")
                .cloned()
        };
        // An auto-repeat may not nest inside another `repeat()` (CSS
        // Grid §7.2.3.2). The parser rejects eager nesting, but `var()`
        // substitution can smuggle a `Repeat` descriptor into a track
        // list after parsing; the validator rejects it and the property
        // falls back to `none`.
        assert_eq!(
            cols(
                "div { --x: repeat(auto-fill, 100px); grid-template-columns: repeat(auto-fill, var(--x)); }"
            ),
            Some(kw("none"))
        );
        // Nested through `minmax()` the same way.
        assert_eq!(
            cols(
                "div { --x: repeat(auto-fill, 100px); grid-template-columns: repeat(auto-fill, minmax(var(--x), 200px)); }"
            ),
            Some(kw("none"))
        );
        // Sanity: a plain auto-repeat (no nesting) stays valid.
        assert_eq!(
            cols("div { grid-template-columns: repeat(auto-fill, 100px); }"),
            Some(CssValue::Repeat {
                mode: RepeatMode::Fill,
                tracks: vec![CssValue::Length(100.0, LengthUnit::Px)],
            })
        );
    }

    #[test]
    fn custom_props_queryable() {
        let (doc, div, _) = sample();
        let s = style(&doc, div, "div { --gap: 4px; }", &vp());
        assert_eq!(s.get("--gap"), Some(&CssValue::Length(4.0, LengthUnit::Px)));
        // kw helper used for a keyword assertion.
        let s = style(&doc, div, "div { --mode: fast; }", &vp());
        assert_eq!(s.get("--mode"), Some(&kw("fast")));
    }

    #[test]
    fn no_author_sheets() {
        let (doc, div, _) = sample();
        let s = compute_style(&doc, div, &[], &ua(), &vp());
        assert_eq!(s.get_str("display"), Some("block"));
        // Unspecified everywhere: the initial value (black).
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            }))
        );
    }

    #[test]
    fn ua_pre_preserves_whitespace() {
        let (doc, _) = Document::parse_html(b"<pre>a  b\nc</pre>");
        let pre = doc.get_elements_by_tag_name("pre").first().unwrap();
        let s = compute_style(&doc, pre, &[], &ua(), &vp());
        assert_eq!(s.get_str("white-space"), Some("pre"));
        // The pre display:block UA rule still applies.
        assert_eq!(s.get_str("display"), Some("block"));
    }

    #[test]
    fn border_shorthand_reaches_per_side_longhands() {
        let (doc, div, _) = sample();
        // Width + style + color, any order, all four sides.
        let s = style(&doc, div, "div { border: solid black 5px; }", &vp());
        for side in ["top", "right", "bottom", "left"] {
            assert_eq!(
                s.get(&format!("border-{side}-width")),
                Some(&CssValue::Length(5.0, LengthUnit::Px)),
                "{side}"
            );
            assert_eq!(
                s.get_str(&format!("border-{side}-style")),
                Some("solid"),
                "{side}"
            );
            assert_eq!(
                s.get(&format!("border-{side}-color")),
                Some(&CssValue::Color(Color {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 255
                })),
                "{side}"
            );
        }
        // Partial shorthand: unspecified components reset to their initial
        // values (per the crate's documented design), not left unset.
        let s = style(&doc, div, "div { border: 2px dotted; }", &vp());
        assert_eq!(
            s.get("border-top-width"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        assert_eq!(s.get_str("border-left-style"), Some("dotted"));
        assert_eq!(s.get("border-right-color"), Some(&kw("currentcolor")));
        // A side-specific shorthand still works alongside.
        let s = style(
            &doc,
            div,
            "div { border: 1px solid red; border-top-width: 9px; }",
            &vp(),
        );
        assert_eq!(
            s.get("border-top-width"),
            Some(&CssValue::Length(9.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("border-bottom-width"),
            Some(&CssValue::Length(1.0, LengthUnit::Px))
        );
        assert_eq!(s.get_str("border-top-style"), Some("solid"));
    }

    #[test]
    fn border_radius_computes_corner_radii() {
        let (doc, div, _) = sample();
        // The shorthand cascades into all four corner longhands.
        let s = style(&doc, div, "div { border-radius: 6px; }", &vp());
        for corner in ["top-left", "top-right", "bottom-right", "bottom-left"] {
            assert_eq!(
                s.get(&format!("border-{corner}-radius")),
                Some(&CssValue::Length(6.0, LengthUnit::Px)),
                "{corner}"
            );
        }
        // Unset radius computes to its initial value (0).
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(
            s.get("border-top-left-radius"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // `/`-separated horizontal/vertical radii compute as pairs.
        let s = style(&doc, div, "div { border-radius: 10px / 5px; }", &vp());
        assert_eq!(
            s.get("border-top-left-radius"),
            Some(&CssValue::List(vec![
                CssValue::Length(10.0, LengthUnit::Px),
                CssValue::Length(5.0, LengthUnit::Px),
            ]))
        );
        // var()-in-shorthand defers expansion to the cascade.
        let s = style(
            &doc,
            div,
            "div { --r: 5px; border-radius: var(--r); }",
            &vp(),
        );
        assert_eq!(
            s.get("border-top-left-radius"),
            Some(&CssValue::Length(5.0, LengthUnit::Px))
        );
        // Radii are not inherited: a child does not pick up the parent's
        // (it computes the initial 0).
        let (doc, _div, p) = sample();
        let s = style(&doc, p, "div { border-radius: 6px; }", &vp());
        assert_eq!(
            s.get("border-top-left-radius"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
    }

    #[test]
    fn list_style_type_inherits_from_list() {
        let mut doc = Document::new();
        let root = doc.root();
        let ol = doc.create_element("ol");
        doc.append_child(root, ol);
        let li = doc.create_element("li");
        doc.append_child(ol, li);
        // Per CSS, list-style properties inherit: yes, so `<li>` picks up
        // the `<ol>`'s marker style.
        let s = style(&doc, li, "ol { list-style-type: decimal; }", &vp());
        assert_eq!(s.get_str("list-style-type"), Some("decimal"));
        // The `<ol>` itself computes it directly.
        let s = style(&doc, ol, "ol { list-style-type: decimal; }", &vp());
        assert_eq!(s.get_str("list-style-type"), Some("decimal"));
        // Unset anywhere: the `<li>` inherits the `<ol>`'s UA `decimal`.
        let s = style(&doc, li, "div { color: red; }", &vp());
        assert_eq!(s.get("list-style-type"), Some(&kw("decimal")));
    }

    #[test]
    fn styler_batch_matches_one_shot() {
        let (doc, div, p) = sample();
        let css = "div { color: red; border: 1px solid blue; } p { margin: 2px; }";
        let author = parse_stylesheet(css);
        let ua = ua();
        let vp = vp();
        let sheets = [&author];
        let styler = Styler::new(&doc, &sheets, &ua, vp);
        for node in [div, p] {
            let batched = styler.style_for(node);
            let oneshot = compute_style(&doc, node, &[&author], &ua, &vp);
            for prop in [
                "color",
                "display",
                "margin-top",
                "border-top-width",
                "border-left-style",
                "border-bottom-color",
            ] {
                assert_eq!(batched.get(prop), oneshot.get(prop), "{prop}");
            }
        }
    }

    /// Styling a whole deep document through one [`Styler`] must be linear
    /// in its size: each element reuses its ancestors' memoized cascade and
    /// inheritance results instead of re-walking the ancestor chain (which
    /// was quadratic: ~2ms per ancestor level per element in debug, i.e.
    /// hours at depth 5000).
    #[test]
    fn batch_styler_styles_deep_dom_in_linear_time() {
        let depth = 2000;
        let mut doc = Document::new();
        let root = doc.root();
        let html = doc.create_element("html");
        doc.append_child(root, html);
        let mut parent = html;
        let mut nodes = Vec::with_capacity(depth + 1);
        nodes.push(html);
        for _ in 0..depth {
            let d = doc.create_element("div");
            doc.append_child(parent, d);
            parent = d;
            nodes.push(d);
        }
        // font-size is inherited (not specified per div): exercises the
        // ancestor walk that used to be quadratic.
        let author = parse_stylesheet("html { font-size: 10px; }");
        let ua = ua();
        let vp = vp();
        let start = std::time::Instant::now();
        let sheets = [&author];
        let styler = Styler::new(&doc, &sheets, &ua, vp);
        // Root-first, like a layout pass.
        for &n in &nodes {
            let _ = styler.style_for(n);
        }
        let elapsed = start.elapsed();
        // Inheritance still resolved correctly at the bottom.
        let bottom = styler.style_for(parent);
        assert_eq!(
            bottom.get("font-size"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
        assert!(
            elapsed < std::time::Duration::from_secs(30),
            "2000-deep batch styling took {elapsed:?}"
        );
    }

    /// Deferred `var()`-in-shorthand: substitution, expansion to all
    /// longhands (omitted components reset to initial), cascade ordering
    /// against longhands, and `!important` preservation.
    #[test]
    fn var_shorthand_expansion_ordering() {
        let (doc, div, _) = sample();
        // All four sides get every component; omitted color resets to
        // currentcolor ( tested in border_shorthand_reaches_per_side_longhands).
        let s = style(
            &doc,
            div,
            "div { --w: 2px; border: var(--w) solid red; }",
            &vp(),
        );
        for side in ["top", "right", "bottom", "left"] {
            assert_eq!(
                s.get(&format!("border-{side}-width")),
                Some(&CssValue::Length(2.0, LengthUnit::Px)),
                "{side}"
            );
            assert_eq!(
                s.get_str(&format!("border-{side}-style")),
                Some("solid"),
                "{side}"
            );
        }
        // Later longhand beats the earlier shorthand.
        let s = style(
            &doc,
            div,
            "div { --w: 2px; border: var(--w) solid red; border-top-width: 9px; }",
            &vp(),
        );
        assert_eq!(
            s.get("border-top-width"),
            Some(&CssValue::Length(9.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("border-bottom-width"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        // Later shorthand beats the earlier longhand.
        let s = style(
            &doc,
            div,
            "div { --w: 2px; border-top-width: 9px; border: var(--w) solid red; }",
            &vp(),
        );
        assert_eq!(
            s.get("border-top-width"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        // !important on the shorthand is preserved through expansion.
        let s = style(
            &doc,
            div,
            "div { --w: 2px; border: var(--w) solid red !important; border-top-width: 9px; }",
            &vp(),
        );
        assert_eq!(
            s.get("border-top-width"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        // Other deferred shorthands expand too, resetting omitted
        // components to initial values.
        let s = style(&doc, div, "div { --c: red; background: var(--c); }", &vp());
        assert_eq!(
            s.get("background-color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        assert_eq!(s.get_str("background-image"), Some("none"));
        assert_eq!(s.get_str("background-repeat"), Some("repeat"));
        // Deferred margin expands per-side.
        let s = style(&doc, div, "div { --m: 1px 2px; margin: var(--m); }", &vp());
        assert_eq!(
            s.get("margin-top"),
            Some(&CssValue::Length(1.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("margin-right"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
    }

    /// Gradient `background-image`s survive the cascade: the grammar
    /// validator admits `CssValue::Gradient`, and the value reaches the
    /// computed style verbatim for the paint backend to parse.
    #[test]
    fn background_image_gradient_survives_cascade() {
        let (doc, div, _) = sample();
        let s = style(
            &doc,
            div,
            "div { background-image: linear-gradient(to right, red, blue); }",
            &vp(),
        );
        assert_eq!(
            s.get("background-image"),
            Some(&CssValue::Gradient {
                name: "linear-gradient".into(),
                source: "linear-gradient(to right, red, blue)".into(),
            })
        );
        // Via the `background` shorthand too.
        let s = style(
            &doc,
            div,
            "div { background: repeating-conic-gradient(red, blue) no-repeat; }",
            &vp(),
        );
        assert_eq!(
            s.get("background-image"),
            Some(&CssValue::Gradient {
                name: "repeating-conic-gradient".into(),
                source: "repeating-conic-gradient(red, blue)".into(),
            })
        );
        // An empty gradient body still drops the declaration: the
        // initial `none` wins.
        let s = style(
            &doc,
            div,
            "div { background-image: linear-gradient(); }",
            &vp(),
        );
        assert_eq!(s.get_str("background-image"), Some("none"));
    }

    /// A `var()` inside a gradient body substitutes at computed-value
    /// time: the parser defers the function as a token stream and the
    /// cascade substitutes the references, re-parsing into a real
    /// `Gradient` (previously the `var()` froze verbatim in `source`
    /// and paint's gradient parser choked on it).
    #[test]
    fn background_image_gradient_var_substitutes() {
        let (doc, div, _) = sample();
        let g = |source: &str| CssValue::Gradient {
            name: "linear-gradient".into(),
            source: source.into(),
        };
        // Basic substitution.
        let s = style(
            &doc,
            div,
            "div { --a: red; background-image: linear-gradient(var(--a), blue); }",
            &vp(),
        );
        assert_eq!(
            s.get("background-image"),
            Some(&g("linear-gradient(red, blue)"))
        );
        // Nested `var()`s, including in stop positions.
        let s = style(
            &doc,
            div,
            "div { --a: red; --p: 50%; background-image: linear-gradient(to right, var(--a) var(--p), blue); }",
            &vp(),
        );
        assert_eq!(
            s.get("background-image"),
            Some(&g("linear-gradient(to right, red 50%, blue)"))
        );
        // `var()` with a fallback.
        let s = style(
            &doc,
            div,
            "div { background-image: linear-gradient(var(--undef, green), blue); }",
            &vp(),
        );
        assert_eq!(
            s.get("background-image"),
            Some(&g("linear-gradient(green, blue)"))
        );
        // `var()` through another custom property.
        let s = style(
            &doc,
            div,
            "div { --a: red; --g: linear-gradient(var(--a), blue); background-image: var(--g); }",
            &vp(),
        );
        assert_eq!(
            s.get("background-image"),
            Some(&g("linear-gradient(red, blue)"))
        );
        // Via the `background` shorthand too.
        let s = style(
            &doc,
            div,
            "div { --a: red; background: linear-gradient(var(--a), blue) no-repeat; }",
            &vp(),
        );
        assert_eq!(
            s.get("background-image"),
            Some(&g("linear-gradient(red, blue)"))
        );
        // An unresolvable `var()` is invalid at computed-value time:
        // the declaration drops and the initial `none` wins.
        let s = style(
            &doc,
            div,
            "div { background-image: linear-gradient(var(--undef), blue); }",
            &vp(),
        );
        assert_eq!(s.get_str("background-image"), Some("none"));
        // A cyclic reference through the gradient fails closed instead
        // of looping.
        let s = style(
            &doc,
            div,
            "div { --a: linear-gradient(var(--a), blue); background-image: var(--a); }",
            &vp(),
        );
        assert_eq!(s.get_str("background-image"), Some("none"));
    }

    /// Token-level `var()` substitution edge cases (see
    /// [`substitute_nested_vars`]).
    #[test]
    fn substitute_nested_vars_edge_cases() {
        use crate::token::tokenize;
        fn toks(s: &str) -> Vec<Token> {
            let mut t = tokenize(s);
            t.pop(); // drop the trailing Eof
            t
        }
        fn env_of(pairs: &[(&str, &str)]) -> FxHashMap<String, CssValue> {
            pairs
                .iter()
                .map(|(k, v)| ((*k).to_string(), CssValue::Tokens(toks(v))))
                .collect()
        }
        fn subst(css: &str, env: &FxHashMap<String, CssValue>) -> Option<Vec<Token>> {
            let t = toks(css);
            let var_names = RefCell::new(VarNameInterner::default());
            let mut active = FxHashSet::default();
            let mut budget = MAX_SUBST_TOKENS;
            substitute_nested_vars(&t, env, &var_names, &mut active, true, 0, &mut budget)
        }
        let css = |out: &[Token]| crate::value::tokens_to_css(out);

        // Basic nested substitution.
        let env = env_of(&[("--a", "red")]);
        let out = subst("linear-gradient(var(--a), blue)", &env).expect("substitutes");
        assert_eq!(css(&out), "linear-gradient(red, blue)");

        // Fallback substitution.
        let out = subst("linear-gradient(var(--undef, green), blue)", &env).expect("fallback used");
        assert_eq!(css(&out), "linear-gradient(green, blue)");

        // Unresolvable with no fallback: invalid.
        assert!(subst("linear-gradient(var(--undef), blue)", &env).is_none());

        // A top-level `var()` — fallback included — is left verbatim
        // for the `Var` arm, even when the fallback holds a nested
        // `var()`: eager substitution could wrongly invalidate the
        // declaration when the outer property *is* defined.
        let input = toks("var(--x, linear-gradient(var(--u), blue))");
        let out = subst("var(--x, linear-gradient(var(--u), blue))", &env).expect("verbatim");
        assert_eq!(out, input);

        // A self-referential property fails closed instead of looping.
        let env = env_of(&[("--a", "linear-gradient(var(--a), blue)")]);
        assert!(subst("linear-gradient(var(--a), blue)", &env).is_none());

        // A mutually referential pair fails closed too.
        let env = env_of(&[
            ("--a", "linear-gradient(var(--b), blue)"),
            ("--b", "linear-gradient(var(--a), blue)"),
        ]);
        assert!(subst("linear-gradient(var(--a), blue)", &env).is_none());

        // A hostile 70-deep chain fails closed at MAX_MATH_DEPTH.
        let mut pairs: Vec<(String, String)> = Vec::new();
        for i in 1..70 {
            pairs.push((format!("--d{i}"), format!("var(--d{})", i + 1)));
        }
        pairs.push(("--d70".to_string(), "red".to_string()));
        let env: FxHashMap<String, CssValue> = pairs
            .iter()
            .map(|(k, v)| (k.clone(), CssValue::Tokens(toks(v))))
            .collect();
        assert!(subst("linear-gradient(var(--d1), blue)", &env).is_none());

        // A hostile diamond (`--d1: var(--d2) var(--d2)`, …) fails
        // closed on the token budget instead of blowing up.
        let mut pairs: Vec<(String, String)> = Vec::new();
        for i in 1..20 {
            pairs.push((
                format!("--e{i}"),
                format!("var(--e{}) var(--e{})", i + 1, i + 1),
            ));
        }
        pairs.push(("--e20".to_string(), "red".to_string()));
        let env: FxHashMap<String, CssValue> = pairs
            .iter()
            .map(|(k, v)| (k.clone(), CssValue::Tokens(toks(v))))
            .collect();
        assert!(subst("linear-gradient(var(--e1), blue)", &env).is_none());
    }

    /// CSS-wide keywords: `inherit`, `initial`, `unset`.
    #[test]
    fn css_wide_keywords() {
        let (doc, _div, p) = sample();
        // inherit on an inherited property: parent's computed value.
        let s = style(&doc, p, "div { color: red; } p { color: inherit; }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // inherit with nothing to inherit: the initial value.
        let s = style(&doc, p, "p { color: inherit; }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // initial: the property's initial value.
        let s = style(&doc, p, "div { color: red; } p { color: initial; }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        let s = style(&doc, p, "p { display: initial; }", &vp());
        assert_eq!(s.get_str("display"), Some("inline"));
        // unset on an inherited property behaves as inherit...
        let s = style(&doc, p, "div { color: red; } p { color: unset; }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // ...and as initial on a non-inherited one (author origin still
        // beats the UA sheet).
        let s = style(&doc, p, "p { display: unset; }", &vp());
        assert_eq!(s.get_str("display"), Some("inline"));
        // Explicit inherit works on non-inherited properties too.
        let s = style(
            &doc,
            p,
            "div { margin-top: 5px; } p { margin-top: inherit; }",
            &vp(),
        );
        assert_eq!(
            s.get("margin-top"),
            Some(&CssValue::Length(5.0, LengthUnit::Px))
        );
        // A var() substituting to a CSS-wide keyword applies as one.
        let s = style(
            &doc,
            p,
            "div { --d: inline; } p { display: var(--d); }",
            &vp(),
        );
        assert_eq!(s.get_str("display"), Some("inline"));
    }

    /// Post-substitution grammar validation: a substituted value that does
    /// not match the property's grammar is invalid at computed-value time.
    #[test]
    fn var_substitution_grammar_validation() {
        let (doc, div, _) = sample();
        // `1px solid` is not a valid margin-top: the declaration is
        // invalid at computed-value time, so margin-top computes to its
        // initial 0.
        let s = style(
            &doc,
            div,
            "div { --x: 1px solid; margin-top: var(--x); }",
            &vp(),
        );
        assert_eq!(
            s.get("margin-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // `red blue` is not a valid color: invalid at computed-value
        // time, so color computes to initial black.
        let s = style(&doc, div, "div { --x: red blue; color: var(--x); }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // Valid substitutions still pass.
        let s = style(&doc, div, "div { --x: 3px; margin-top: var(--x); }", &vp());
        assert_eq!(
            s.get("margin-top"),
            Some(&CssValue::Length(3.0, LengthUnit::Px))
        );
        // Validation also rejects garbage specified directly (display
        // computes to initial `inline`).
        let s = style(&doc, div, "div { display: banana; }", &vp());
        assert_eq!(s.get("display"), Some(&kw("inline")));
    }

    /// Long `var()` chains resolve in linear time (memoized per
    /// node/property, iterative, hash-set cycle detection); a cyclic
    /// chain is still invalid at computed-value time.
    #[test]
    fn var_chain_resolves_fast_and_cycles_fail() {
        let n = 400;
        let mut css = String::from("div { ");
        for i in 0..n {
            css.push_str(&format!("--a{i}: var(--a{}); ", i + 1));
        }
        css.push_str(&format!("--a{n}: 7px; width: var(--a0); }}"));
        let (doc, div, _) = sample();
        let start = std::time::Instant::now();
        let s = style(&doc, div, &css, &vp());
        let elapsed = start.elapsed();
        assert_eq!(s.get("width"), Some(&CssValue::Length(7.0, LengthUnit::Px)));
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "400-link var() chain took {elapsed:?}"
        );
        // A 400-link cycle is invalid at computed-value time (and does not
        // recurse the stack doing it).
        let mut css = String::from("div { ");
        for i in 0..n {
            css.push_str(&format!("--a{i}: var(--a{}); ", i + 1));
        }
        css.push_str("--a400: var(--a0); width: var(--a0); }");
        let s = style(&doc, div, &css, &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
    }

    /// w57: a cyclic `var()` inside `calc()` fails fast at the math
    /// substitution layer. The resolve machine only substitutes
    /// top-level `var()`s, so a calc-wrapped custom property re-parses
    /// to a math tree whose `var()` leaves come back here — without the
    /// path-local `seen` set, a 2-cycle recurses `MAX_MATH_DEPTH` deep
    /// doing a full resolve + re-parse per level. The second visit to a
    /// property on the expansion path now fails closed in milliseconds.
    /// The declaration is invalid at computed-value time either way.
    #[test]
    fn calc_var_cycle_fails_fast() {
        let (doc, div, _) = sample();
        let start = std::time::Instant::now();
        let s = style(
            &doc,
            div,
            "div { --a: calc(var(--b) + 1px); --b: calc(var(--a) + 1px); \
             width: calc(var(--a)); }",
            &vp(),
        );
        let elapsed = start.elapsed();
        assert_eq!(s.get("width"), Some(&kw("auto")));
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "2-cycle in calc() took {elapsed:?}"
        );
        // A self-cycle through calc() behaves the same.
        let s = style(
            &doc,
            div,
            "div { --a: calc(var(--a) + 1px); width: calc(var(--a)); }",
            &vp(),
        );
        assert_eq!(s.get("width"), Some(&kw("auto")));
        // A plain (non-calc) 2-cycle is still caught by the resolve
        // machine on the first expansion.
        let s = style(
            &doc,
            div,
            "div { --a: var(--b); --b: var(--a); width: calc(var(--a) + 1px); }",
            &vp(),
        );
        assert_eq!(s.get("width"), Some(&kw("auto")));
    }

    /// w57: a diamond `var()` tree through `calc()` must not blow up
    /// exponentially. Each level references the previous one twice, so
    /// naive re-substitution costs 2^n resolve + re-parse rounds at
    /// constant depth — the depth cap cannot catch it. The expansion
    /// budget fails the hostile shape closed in bounded time instead,
    /// mirroring the token path's budget.
    #[test]
    fn calc_var_diamond_no_exponential_blowup() {
        let (doc, div, _) = sample();
        // A shallow diamond that fits the budget resolves exactly.
        let s = style(
            &doc,
            div,
            "div { --d0: 10px; --d1: calc(var(--d0) + var(--d0)); \
             --d2: calc(var(--d1) + var(--d1)); width: calc(var(--d2)); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(40.0, LengthUnit::Px))
        );
        // A 25-level diamond (2^25 naive expansions) fails closed via
        // the budget in well under the bound — no exponential blowup.
        let n = 25u32;
        let mut css = String::from("div { --d0: 10px; ");
        for i in 1..=n {
            css.push_str(&format!(
                "--d{i}: calc(var(--d{}) + var(--d{})); ",
                i - 1,
                i - 1
            ));
        }
        css.push_str(&format!("width: calc(var(--d{n})); }}"));
        let start = std::time::Instant::now();
        let s = style(&doc, div, &css, &vp());
        let elapsed = start.elapsed();
        assert_eq!(s.get("width"), Some(&kw("auto")));
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "diamond var() tree of {n} levels took {elapsed:?}"
        );
    }

    /// w57: the math-path cycle guard is per-expansion-path, not
    /// global. A deep (but acyclic) `var()` chain through `calc()`
    /// still resolves, and the same property referenced twice in one
    /// `calc()` still works — the `seen` id is popped when its
    /// expansion finishes, so sibling references are not mistaken for
    /// cycles.
    #[test]
    fn calc_var_acyclic_chains_still_resolve() {
        let (doc, div, _) = sample();
        // The same var twice in one calc: 10px + 10px.
        let s = style(
            &doc,
            div,
            "div { --y: 10px; width: calc(var(--y) + var(--y)); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(20.0, LengthUnit::Px))
        );
        // A 20-link acyclic chain through calc() resolves. (Each link
        // costs one depth unit — one per `var()` expansion; the
        // enclosing `+` is pure structural descent and spends only the
        // separate structural budget — so this stays well under
        // MAX_MATH_DEPTH.)
        let n = 20u32;
        let mut css = String::from("div { --c0: 10px; ");
        for i in 1..n {
            css.push_str(&format!("--c{i}: calc(var(--c{}) + 0px); ", i - 1));
        }
        css.push_str(&format!("width: calc(var(--c{})); }}", n - 1));
        let s = style(&doc, div, &css, &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
    }

    /// w59: pin the `MAX_MATH_DEPTH` boundary for `var()` chains through
    /// `calc()`. The depth counter now spends one unit per actual
    /// `var()` expansion (pure structural descent through the
    /// enclosing `+` is guarded separately), so the practical
    /// acyclic-chain limit is the full 64, not ~32: a 64-wide chain
    /// (63 calc-wrapped links plus the top-level `var()`) resolves,
    /// while a 65-wide chain exceeds the budget and is invalid at
    /// computed-value time, failing closed (`width` falls back to
    /// `auto`). The over-limit case is styled twice to pin determinism
    /// — guarding against future memoization making the fail-closed
    /// path order-dependent.
    #[test]
    fn calc_var_chain_depth_boundary_pinned() {
        fn chain_css(n: u32) -> String {
            let mut css = String::from("div { --c0: 10px; ");
            for i in 1..n {
                css.push_str(&format!("--c{i}: calc(var(--c{}) + 0px); ", i - 1));
            }
            css.push_str(&format!("width: calc(var(--c{})); }}", n - 1));
            css
        }
        let (doc, div, _) = sample();
        // 64 wide: 64 `var()` expansions, deepest at depth 64 =
        // MAX_MATH_DEPTH. Resolves.
        let s = style(&doc, div, &chain_css(64), &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
        // 65 wide: the 65th expansion would sit at depth 65 >
        // MAX_MATH_DEPTH — fail-closed to `auto`, identically on
        // repeated styling.
        let css = chain_css(65);
        for _ in 0..2 {
            let s = style(&doc, div, &css, &vp());
            assert_eq!(s.get("width"), Some(&kw("auto")));
        }
    }

    /// Many distinct `var()` references resolve correctly in one pass:
    /// distinct names, chains, fallbacks, cycles, and list splicing all
    /// go through the interned-name memo and the reused cycle-detection
    /// set. Guards the perf fix against behavioral regressions.
    #[test]
    fn many_var_references_resolve_correctly() {
        let n = 200;
        let mut css = String::from("div { ");
        // 200 distinct custom properties, each referenced through a
        // one-link chain (`--r{i}: var(--w{i})` -> `--w{i}: var(--v{i})`).
        for i in 0..n {
            css.push_str(&format!("--v{i}: {}px; --w{i}: var(--v{i}); ", i + 1));
        }
        for i in 0..n {
            css.push_str(&format!("--r{i}: var(--w{i}); "));
        }
        // Fallbacks, an undefined name, and a 2-cycle.
        css.push_str("--fb: var(--undefined, 7px); ");
        css.push_str("--cyc-a: var(--cyc-b); --cyc-b: var(--cyc-a); ");
        css.push_str("--cyc-use: var(--cyc-a, 9px); ");
        css.push_str("width: var(--r0); }");
        let (doc, div, _) = sample();
        let start = std::time::Instant::now();
        let s = style(&doc, div, &css, &vp());
        let elapsed = start.elapsed();
        for i in 0..n {
            assert_eq!(
                s.get(&format!("--r{i}")),
                Some(&CssValue::Length(i as f32 + 1.0, LengthUnit::Px)),
                "--r{i}"
            );
        }
        assert_eq!(s.get("--fb"), Some(&CssValue::Length(7.0, LengthUnit::Px)));
        // The cycle is invalid at computed-value time, so the fallback
        // applies; without a fallback the property is unset.
        assert_eq!(
            s.get("--cyc-use"),
            Some(&CssValue::Length(9.0, LengthUnit::Px))
        );
        assert_eq!(s.get("--cyc-a"), None);
        assert_eq!(s.get("width"), Some(&CssValue::Length(1.0, LengthUnit::Px)));
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "200 var() references took {elapsed:?}"
        );
    }

    /// An inner fallback on a custom property that closes a dependency
    /// cycle does NOT rescue it: per css-variables-1 §2.3 every property
    /// in the cycle is guaranteed-invalid at computed-value time, so
    /// `var(--a, 5px)` inside the cycle is invalid and only the *outer*
    /// fallback applies: `width: var(--a, 7px)` computes to 7px, not 5px.
    /// The outer fallback (`var(--cyc-a, 9px)` with no inner fallback in
    /// the cycle) must keep working -- see
    /// `many_var_references_resolve_correctly`.
    #[test]
    fn var_cycle_closing_inner_fallback_does_not_rescue() {
        let (doc, div, _) = sample();
        let s = style(
            &doc,
            div,
            "div { --a: var(--b); --b: var(--a, 5px); width: var(--a, 7px); }",
            &vp(),
        );
        // --a and --b are in a cycle: both guaranteed-invalid, so the
        // inner `5px` fallback inside --b never applies, and width falls
        // back to its own outer fallback, 7px.
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(7.0, LengthUnit::Px)),
            "width was {:?}, inner cycle fallback leaked through",
            s.get("width")
        );
        // The cycled custom properties themselves are invalid: neither
        // resolves to a length.
        assert!(
            !matches!(s.get("--a"), Some(CssValue::Length(..))),
            "--a was {:?}, cycle should be invalid",
            s.get("--a")
        );
    }

    /// The name interner dedups (one id per distinct name, no per-lookup
    /// clones retained) and the scratch cycle-detection set is empty
    /// between resolutions (reused, not reallocated, and never leaking
    /// active names).
    #[test]
    fn var_interner_dedups_and_scratch_set_reused() {
        let (doc, div, _) = sample();
        let author = parse_stylesheet(
            "div { --a: 1px; --b: var(--a); color: var(--b); \
             --c: var(--b); background: var(--a) var(--b); }",
        );
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let _ = styler.style_for(div);
        // Distinct var()-referenced names interned exactly once: --a, --b.
        // (Definition names like `--c` are env keys, never memo keys.)
        assert_eq!(styler.var_names.borrow().names.len(), 2);
        let first = styler.var_names.borrow_mut().intern("--a");
        let second = styler.var_names.borrow_mut().intern("--a");
        assert_eq!(first, second);
        // The scratch set is empty after resolution (every insert paired
        // with a remove) and retains capacity for reuse.
        assert!(styler.resolve_active.borrow().is_empty());
        // Resolving again reuses the interner and the set: no new names,
        // still empty afterwards.
        let _ = styler.style_for(div);
        assert_eq!(styler.var_names.borrow().names.len(), 2);
        assert!(styler.resolve_active.borrow().is_empty());
    }

    fn deep_doc(depth: usize) -> (Document, NodeId) {
        let mut doc = Document::new();
        let root = doc.root();
        let html = doc.create_element("html");
        doc.append_child(root, html);
        let mut parent = html;
        for _ in 0..depth {
            let d = doc.create_element("div");
            doc.append_child(parent, d);
            parent = d;
        }
        (doc, parent)
    }

    /// A one-shot `compute_style` on a deep node goes root-first through a
    /// `Styler` instead of recursing per ancestor, so it cannot overflow
    /// the stack.
    #[test]
    fn one_shot_compute_style_deep_dom() {
        let (doc, deepest) = deep_doc(2000);
        let author = parse_stylesheet("html { font-size: 10px; }");
        let s = compute_style(&doc, deepest, &[&author], &ua(), &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
    }

    #[test]
    fn math_evaluates_end_to_end() {
        let (doc, div, _) = sample();
        // The required case: `calc(100px - 20px)` computes to `80px`.
        let s = style(&doc, div, "div { width: calc(100px - 20px); }", &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(80.0, LengthUnit::Px))
        );
        // Relative units resolve against real metrics: `1em` is the
        // element's font size, `2rem` the root's, `10vw` the viewport.
        // (A parsed document, so the root element is `<html>` at the
        // initial 16px; in `sample()` the div itself is the document
        // element, so `rem` would resolve against its own font size.)
        let (doc2, _) = Document::parse_html(b"<html><body><div></div></body></html>");
        let div2 = doc2.get_elements_by_tag_name("div").first().unwrap();
        let s = style(
            &doc2,
            div2,
            "div { font-size: 20px; width: calc(1em + 2rem + 10vw); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(20.0 + 32.0 + 80.0, LengthUnit::Px))
        );
        // `min()`/`max()`/`clamp()` evaluate too.
        let s = style(
            &doc,
            div,
            "div { width: min(100px, 20px + 30px); height: max(10px, 5px); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(50.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("height"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
        // Invalid math is invalid at computed-value time: the declaration
        // is dropped, so the width computes to its initial `auto`.
        let s = style(&doc, div, "div { width: calc(1px + 2); }", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
        let s = style(&doc, div, "div { width: calc(1px / 0); }", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
    }

    #[test]
    fn math_var_substitution() {
        let (doc, div, _) = sample();
        // `var()` inside `calc()` substitutes from the element's
        // custom-property environment before evaluation.
        let s = style(
            &doc,
            div,
            "div { --gap: 20px; width: calc(100px - var(--gap)); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(80.0, LengthUnit::Px))
        );
        // Fallbacks work when the custom property is undefined.
        let s = style(
            &doc,
            div,
            "div { width: calc(100px - var(--gap, 30px)); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(70.0, LengthUnit::Px))
        );
        // An undefined variable with no fallback drops the declaration
        // (width computes to initial `auto`).
        let s = style(&doc, div, "div { width: calc(100px - var(--gap)); }", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
        // A custom property holding math itself substitutes as a tree.
        let s = style(
            &doc,
            div,
            "div { --w: calc(10px + 5px); width: calc(var(--w) * 2); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(30.0, LengthUnit::Px))
        );
        // A non-length variable is a type error: dropped.
        let s = style(
            &doc,
            div,
            "div { --c: red; width: calc(var(--c) + 1px); }",
            &vp(),
        );
        assert_eq!(s.get("width"), Some(&kw("auto")));
    }

    #[test]
    fn calc_size_evaluates_end_to_end() {
        let (doc, div, _) = sample();
        // A definite length basis evaluates like `calc()`.
        let s = style(
            &doc,
            div,
            "div { width: calc-size(100px, size * 2); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(200.0, LengthUnit::Px))
        );
        let s = style(
            &doc,
            div,
            "div { width: calc-size(calc(10px + 20px), size + 5px); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(35.0, LengthUnit::Px))
        );
        // A percentage basis defers to layout as coefficients, like any
        // other percentage math: `50% + 10px`.
        let s = style(
            &doc,
            div,
            "div { width: calc-size(50%, size + 10px); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::MathDeferred {
                pct: 50.0,
                px: 10.0,
                had_pct: true,
                affine: true,
            })
        );
        // Keyword bases survive the cascade untouched for layout: the
        // fixed-layout table worker reads the basis through
        // `calc_size_basis()`.
        let s = style(&doc, div, "div { width: calc-size(auto, size); }", &vp());
        assert_eq!(
            s.get("width").and_then(|v| v.calc_size_basis()),
            Some(&CalcSizeBasis::Auto)
        );
        let s = style(
            &doc,
            div,
            "div { width: calc-size(max-content, 10px + size); }",
            &vp(),
        );
        assert_eq!(
            s.get("width").and_then(|v| v.calc_size_basis()),
            Some(&CalcSizeBasis::MaxContent)
        );
        // `var()` inside the calc-sum substitutes before evaluation.
        let s = style(
            &doc,
            div,
            "div { --pad: 5px; width: calc-size(100px, size + var(--pad)); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(105.0, LengthUnit::Px))
        );
        // Invalid math drops the declaration (width computes to `auto`).
        let s = style(
            &doc,
            div,
            "div { width: calc-size(100px, size + 2); }",
            &vp(),
        );
        assert_eq!(s.get("width"), Some(&kw("auto")));
        let s = style(&doc, div, "div { width: calc-size(auto); }", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
    }

    #[test]
    fn calc_size_var_basis_substitutes_before_validation() {
        let (doc, div, _) = sample();
        // `var()` in basis position substitutes before basis
        // validation (CSS Values 4): a defined `--x: 100px` evaluates.
        let s = style(
            &doc,
            div,
            "div { --x: 100px; width: calc-size(var(--x), size); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(100.0, LengthUnit::Px))
        );
        // A definite keyword-valued property still computes: the
        // substituted basis evaluates through the calc-sum.
        let s = style(
            &doc,
            div,
            "div { --x: 100px; width: calc-size(var(--x), size * 2); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(200.0, LengthUnit::Px))
        );
        // A keyword-valued property (`--x: max-content`) leaves the
        // keyword basis for layout: the substituted value survives the
        // cascade untouched.
        let s = style(
            &doc,
            div,
            "div { --x: max-content; width: calc-size(var(--x), size + 10px); }",
            &vp(),
        );
        assert_eq!(
            s.get("width").and_then(|v| v.calc_size_basis()),
            Some(&CalcSizeBasis::MaxContent)
        );
        // An undefined property uses the `var()` fallback, checked
        // against the basis grammar (a keyword fallback is valid).
        let s = style(
            &doc,
            div,
            "div { width: calc-size(var(--undefined, 50px), size * 2); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(100.0, LengthUnit::Px))
        );
        let s = style(
            &doc,
            div,
            "div { width: calc-size(var(--undefined, max-content), size); }",
            &vp(),
        );
        assert_eq!(
            s.get("width").and_then(|v| v.calc_size_basis()),
            Some(&CalcSizeBasis::MaxContent)
        );
        // A defined-but-invalid basis falls back too (B4).
        let s = style(
            &doc,
            div,
            "div { --x: red; width: calc-size(var(--x, 25px), size + 5px); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(30.0, LengthUnit::Px))
        );
        // Unresolvable with no fallback: invalid at computed-value
        // time, the declaration drops (width computes to `auto`).
        let s = style(
            &doc,
            div,
            "div { width: calc-size(var(--undefined), size); }",
            &vp(),
        );
        assert_eq!(s.get("width"), Some(&kw("auto")));
        // A nested `var()` chain substitutes through.
        let s = style(
            &doc,
            div,
            "div { --y: 40px; --x: var(--y); width: calc-size(var(--x), size); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(40.0, LengthUnit::Px))
        );
    }

    #[test]
    fn math_font_relative_units() {
        let (doc, div, _) = sample();
        // `font-size` math: `em` and `%` resolve against the parent.
        let s = style(&doc, div, "div { font-size: calc(1em + 4px); }", &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(20.0, LengthUnit::Px))
        );
        let s = style(&doc, div, "div { font-size: calc(50% + 8px); }", &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(16.0, LengthUnit::Px))
        );
        // `line-height` math keeps unitless numbers unitless.
        let s = style(
            &doc,
            div,
            "div { font-size: 20px; line-height: calc(2 * 1.5); }",
            &vp(),
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(3.0)));
        let s = style(
            &doc,
            div,
            "div { font-size: 20px; line-height: calc(1em + 10px); }",
            &vp(),
        );
        assert_eq!(
            s.get("line-height"),
            Some(&CssValue::Length(30.0, LengthUnit::Px))
        );
    }

    #[test]
    fn unitless_zero_is_a_valid_length() {
        let (doc, div, _) = sample();
        // CSS-wide rule: unitless 0 is a valid <length> for length
        // properties (previously dropped as invalid at computed-value time).
        let s = style(&doc, div, "div { min-width: 0; }", &vp());
        assert_eq!(
            s.get("min-width"),
            Some(&CssValue::Number(0.0)),
            "unitless zero must survive grammar validation"
        );
        // Non-zero unitless numbers are still invalid lengths.
        let s = style(&doc, div, "div { min-width: 5; }", &vp());
        assert!(
            !matches!(s.get("min-width"), Some(CssValue::Number(5.0))),
            "unitless non-zero must not validate as a length"
        );
    }

    // ---- A1: list-valued var() splices into the enclosing list ----

    #[test]
    fn var_list_splices_into_enclosing_list() {
        let (doc, div, _) = sample();
        // `--x` holds two tokens; `var(--x)` splices them flat into the
        // `background-position` list instead of nesting a sub-list (A1).
        let s = style(
            &doc,
            div,
            "div { --x: calc(10px + 5px) 20px; background-position: left var(--x); }",
            &vp(),
        );
        assert_eq!(
            s.get("background-position"),
            Some(&CssValue::List(vec![
                CssValue::Keyword("left".into()),
                CssValue::Length(15.0, LengthUnit::Px),
                CssValue::Length(20.0, LengthUnit::Px),
            ]))
        );
    }

    #[test]
    fn nested_var_splice_evaluates_math_at_depth() {
        let (doc, div, _) = sample();
        // Two levels of indirection: `--outer` splices `--inner`, which
        // itself splices; the math inside evaluates after splicing (A1).
        let s = style(
            &doc,
            div,
            "div { --inner: calc(1px + 1px) 2px; --outer: left var(--inner); \
             background-position: var(--outer) top; }",
            &vp(),
        );
        assert_eq!(
            s.get("background-position"),
            Some(&CssValue::List(vec![
                CssValue::Keyword("left".into()),
                CssValue::Length(2.0, LengthUnit::Px),
                CssValue::Length(2.0, LengthUnit::Px),
                CssValue::Keyword("top".into()),
            ]))
        );
    }

    #[test]
    fn font_family_var_splice_stays_flat() {
        let (doc, div, _) = sample();
        // `--a` is a two-item list; splicing it into `font-family` must not
        // leak a nested list or an unevaluated Math (A1).
        let s = style(
            &doc,
            div,
            "div { --a: Arial, calc(1px + 1px); font-family: var(--a), serif; }",
            &vp(),
        );
        let v = s.get("font-family").expect("font-family must compute");
        match v {
            CssValue::List(items) => {
                assert_eq!(items.len(), 3, "splice must be flat, got {v:?}");
                assert!(
                    !items
                        .iter()
                        .any(|i| matches!(i, CssValue::List(_) | CssValue::Math(_))),
                    "no nested list or raw math may leak, got {v:?}"
                );
                assert_eq!(items[1], CssValue::Length(2.0, LengthUnit::Px));
            }
            other => panic!("expected a flat list, got {other:?}"),
        }
    }

    // ---- A2: percentage math defers to layout ----

    #[test]
    fn percentage_calc_defers_with_coefficients() {
        let (doc, div, _) = sample();
        // `calc(50% - 20px)` is affine in `%`: the cascade must not fold
        // the percentage against 0; it emits MathDeferred coefficients
        // for layout's real containing-block base (A2).
        let s = style(&doc, div, "div { width: calc(50% - 20px); }", &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::MathDeferred {
                pct: 50.0,
                px: -20.0,
                had_pct: true,
                affine: true
            })
        );
    }

    #[test]
    fn zero_folded_percentage_still_marks_had_pct() {
        let (doc, div, _) = sample();
        // `calc(10px + 0%)` folds to pct 0.0 / px 10.0, but it
        // syntactically contained a percentage leaf: the `had_pct` bit
        // marks it so cyclic-ness resolution (CSS Values 4 §8.1.4) treats
        // the whole value as `auto` against an indefinite base instead
        // of wrongly honoring it as a definite 10px.
        let s = style(&doc, div, "div { width: calc(10px + 0%); }", &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::MathDeferred {
                pct: 0.0,
                px: 10.0,
                had_pct: true,
                affine: true
            })
        );
        // Same for a fully-canceling percentage pair.
        let s = style(&doc, div, "div { width: calc(50% - 50%); }", &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::MathDeferred {
                pct: 0.0,
                px: 0.0,
                had_pct: true,
                affine: true
            })
        );
        // A plain length stays definite: no deferral, no bit.
        let s = style(&doc, div, "div { width: 10px; }", &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
    }

    #[test]
    fn non_affine_percentage_math_keeps_syntactic_record() {
        let (doc, div, _) = sample();
        // `min()`/`max()`/`clamp()` with `%` are not affine, so no
        // coefficient pair can capture them — but the syntactic
        // percentage must still survive evaluation. `max(10px, 0%)`
        // used to fold to a definite `Length(10px)` (base-0
        // approximation); it now defers as `MathDeferred` with `had_pct`,
        // so layout's cyclic-ness gates (CSS Values 4 §8.1.4) treat it as
        // `auto` against an indefinite base instead of a definite 10px.
        // `pct` stays 0.0, so resolving the coefficients against any base
        // yields exactly the old base-0 value (10px here).
        let s = style(&doc, div, "div { width: max(10px, 0%); }", &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::MathDeferred {
                pct: 0.0,
                px: 10.0,
                had_pct: true,
                affine: false
            })
        );
        // `min(50px, 20%)` keeps its percentage record too (the base-0
        // value is 0px, but it is not a definite length).
        let s = style(&doc, div, "div { width: min(50px, 20%); }", &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::MathDeferred {
                pct: 0.0,
                px: 0.0,
                had_pct: true,
                affine: false
            })
        );
        // `clamp()` likewise (`clamp(10px, 50%, 100px)` is 10px at
        // base 0, but the percentage record survives).
        let s = style(&doc, div, "div { width: clamp(10px, 50%, 100px); }", &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::MathDeferred {
                pct: 0.0,
                px: 10.0,
                had_pct: true,
                affine: false
            })
        );
        // Percentage-free `min()`/`max()` still fold to plain lengths.
        let s = style(&doc, div, "div { width: max(10px, 5px); }", &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
        // `var()`-introduced percentages keep working: substitution runs
        // before the deferral gate, so the substituted `%` leaf marks
        // `had_pct` exactly like a literal one.
        let s = style(
            &doc,
            div,
            "div { --w: 0%; width: max(10px, var(--w)); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::MathDeferred {
                pct: 0.0,
                px: 10.0,
                had_pct: true,
                affine: false
            })
        );
        // `font-size`/`line-height` percentages still resolve against
        // font metrics in the cascade, even inside `max()`.
        let s = style(&doc, div, "div { font-size: max(10px, 50%); }", &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
    }

    #[test]
    fn percentage_free_calc_evaluates_in_cascade() {
        let (doc, div, _) = sample();
        // No percentages: the cascade still evaluates eagerly (A2).
        let s = style(&doc, div, "div { width: calc(100px + 20px); }", &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(120.0, LengthUnit::Px))
        );
    }

    #[test]
    fn font_size_percentage_math_evaluates_in_cascade() {
        let (doc, div, _) = sample();
        // `font-size`/`line-height` percentages resolve against the font
        // size in the cascade itself (A2): 50% of the 16px default.
        let s = style(&doc, div, "div { font-size: calc(50% + 8px); }", &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(16.0, LengthUnit::Px))
        );
    }

    // ---- B3: iterative ancestor climb ----

    #[test]
    fn deep_chain_inheritance_is_iterative() {
        let mut doc = Document::new();
        let root = doc.root();
        let top = doc.create_element("div");
        doc.append_child(root, top);
        doc.set_attribute(top, "id", "top");
        let mut parent = top;
        for _ in 0..9_999 {
            let child = doc.create_element("div");
            doc.append_child(parent, child);
            parent = child;
        }
        let deep = parent;
        let author = parse_stylesheet("#top { color: red; }");
        let author_refs = [&author];
        let ua_ss = ua();
        // Direct cold `style_for` on the deepest node: 10k inherited
        // climbs must not overflow the call stack (B3). `color` is set
        // only on #top, so every level below climbs.
        let styler = Styler::new(&doc, &author_refs, &ua_ss, vp());
        let s = styler.style_for(deep);
        assert!(
            matches!(s.get("color"), Some(CssValue::Color(_))),
            "deep node must inherit the Color, got {:?}",
            s.get("color")
        );
    }

    #[test]
    fn deep_chain_explicit_inherit_is_iterative() {
        let mut doc = Document::new();
        let root = doc.root();
        let top = doc.create_element("div");
        doc.append_child(root, top);
        doc.set_attribute(top, "id", "top");
        let mut parent = top;
        for _ in 0..9_999 {
            let child = doc.create_element("div");
            doc.append_child(parent, child);
            parent = child;
        }
        let deep = parent;
        // Explicit `inherit` at every level: the climb is iterative even
        // when each step comes from a CSS-wide keyword (B3).
        let author = parse_stylesheet("#top { color: red; } div { color: inherit; }");
        let author_refs = [&author];
        let ua_ss = ua();
        let styler = Styler::new(&doc, &author_refs, &ua_ss, vp());
        let s = styler.style_for(deep);
        assert!(
            matches!(s.get("color"), Some(CssValue::Color(_))),
            "deep node must inherit the Color, got {:?}",
            s.get("color")
        );
    }

    // ---- B4: invalid custom property falls back inside math ----

    #[test]
    fn math_var_fallback_on_invalid_custom_property() {
        let (doc, div, _) = sample();
        // `--x` is defined but invalid at computed-value time (it
        // references an undefined var); the `var()` fallback applies (B4).
        let s = style(
            &doc,
            div,
            "div { --x: var(--undefined); width: calc(var(--x, 5px) + 1px); }",
            &vp(),
        );
        assert_eq!(s.get("width"), Some(&CssValue::Length(6.0, LengthUnit::Px)));
    }

    #[test]
    fn math_var_fallback_on_undefined_custom_property() {
        let (doc, div, _) = sample();
        // Undefined property: the fallback applies, as before (B4).
        let s = style(
            &doc,
            div,
            "div { width: calc(var(--nope, 5px) + 1px); }",
            &vp(),
        );
        assert_eq!(s.get("width"), Some(&CssValue::Length(6.0, LengthUnit::Px)));
    }

    // ---- B5: calc() admitted for scalar properties ----

    #[test]
    fn calc_admitted_for_scalar_properties() {
        let (doc, div, _) = sample();
        let s = style(
            &doc,
            div,
            "div { opacity: calc(0.5 + 0.25); z-index: calc(1 + 2); order: calc(4 - 1); \
             flex-grow: calc(1 + 1); flex-shrink: calc(3 - 1); font-weight: calc(400 + 300); }",
            &vp(),
        );
        assert_eq!(s.get("opacity"), Some(&CssValue::Number(0.75)));
        assert_eq!(s.get("z-index"), Some(&CssValue::Number(3.0)));
        assert_eq!(s.get("order"), Some(&CssValue::Number(3.0)));
        assert_eq!(s.get("flex-grow"), Some(&CssValue::Number(2.0)));
        assert_eq!(s.get("flex-shrink"), Some(&CssValue::Number(2.0)));
        assert_eq!(s.get("font-weight"), Some(&CssValue::Number(700.0)));
    }

    #[test]
    fn calc_admitted_for_spacing_properties() {
        let (doc, div, _) = sample();
        let s = style(
            &doc,
            div,
            "div { letter-spacing: calc(1px + 1px); word-spacing: calc(2px + 2px); }",
            &vp(),
        );
        assert_eq!(
            s.get("letter-spacing"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("word-spacing"),
            Some(&CssValue::Length(4.0, LengthUnit::Px))
        );
    }

    // ---- B6: unitless zero admitted for length properties ----

    #[test]
    fn unitless_zero_for_border_and_spacing() {
        let (doc, div, _) = sample();
        let s = style(
            &doc,
            div,
            "div { border-right-width: 0; letter-spacing: 0; border-spacing: 0; \
             border-top-width: 0px; }",
            &vp(),
        );
        assert_eq!(
            s.get("border-right-width"),
            Some(&CssValue::Number(0.0)),
            "border-right-width: 0"
        );
        assert_eq!(
            s.get("letter-spacing"),
            Some(&CssValue::Number(0.0)),
            "letter-spacing: 0"
        );
        assert_eq!(
            s.get("border-spacing"),
            Some(&CssValue::Number(0.0)),
            "border-spacing: 0"
        );
        assert_eq!(
            s.get("border-top-width"),
            Some(&CssValue::Length(0.0, LengthUnit::Px)),
            "border-top-width: 0px still parses as a length"
        );
    }

    // ---- B7: border shorthand with zero / calc widths ----

    #[test]
    fn border_shorthand_zero_and_calc_width() {
        let (doc, div, _) = sample();
        let s = style(&doc, div, "div { border: 0 solid red; }", &vp());
        assert_eq!(
            s.get("border-top-width"),
            Some(&CssValue::Number(0.0)),
            "unitless zero is a width, not a style"
        );
        assert_eq!(s.get_str("border-top-style"), Some("solid"));
        let s = style(&doc, div, "div { border: calc(2px) solid red; }", &vp());
        // `calc(2px)` evaluates to a length in the cascade.
        assert_eq!(
            s.get("border-top-width"),
            Some(&CssValue::Length(2.0, LengthUnit::Px)),
            "calc width must classify as a width, not a style"
        );
        assert_eq!(s.get_str("border-top-style"), Some("solid"));
    }

    // ---- B8: invalid math drops the declaration ----

    #[test]
    fn invalid_math_drops_declaration() {
        let (doc, div, _) = sample();
        // `calc(1 + 2)` is unitless and invalid as a width: the whole
        // declaration is dropped (B8), so the width computes to its
        // initial `auto`.
        let s = style(&doc, div, "div { width: calc(1 + 2); }", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
    }

    // ---- C4 follow-up: <angle> keeps its type through math ----

    #[test]
    fn angle_math_rejected_for_angle_less_properties() {
        let (doc, div, _) = sample();
        // `calc(90deg)` evaluates to an <angle>, which no property takes
        // through calc(): the declarations are invalid at computed-value
        // time and drop, leaving the initial values in place.
        let s = style(&doc, div, "div { opacity: calc(90deg); }", &vp());
        assert_eq!(s.get("opacity"), Some(&CssValue::Number(1.0)));
        let s = style(&doc, div, "div { line-height: calc(90deg); }", &vp());
        assert_eq!(s.get("line-height"), Some(&kw("normal")));
        let s = style(&doc, div, "div { z-index: calc(90deg); }", &vp());
        assert_eq!(s.get("z-index"), Some(&kw("auto")));
        let s = style(&doc, div, "div { width: calc(90deg); }", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
    }

    #[test]
    fn angle_mixed_arithmetic_rejected() {
        let (doc, div, _) = sample();
        // Angle/number/length mixing is a type error (CSS Values 4
        // §10.2): the declarations drop.
        let s = style(&doc, div, "div { width: calc(90deg * 1px); }", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
        let s = style(&doc, div, "div { width: calc(1px / 90deg); }", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
        let s = style(&doc, div, "div { width: calc(90deg + 1); }", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
        // atan2() with mixed angle/number arguments is invalid.
        let s = style(&doc, div, "div { opacity: atan2(90deg, 1); }", &vp());
        assert_eq!(s.get("opacity"), Some(&CssValue::Number(1.0)));
        // atan2() returns an <angle>: not a valid opacity either.
        let s = style(&doc, div, "div { opacity: atan2(1, 1); }", &vp());
        assert_eq!(s.get("opacity"), Some(&CssValue::Number(1.0)));
    }

    #[test]
    fn var_substituted_angle_reaches_trig() {
        let (doc, div, _) = sample();
        // A custom property holding an angle dimension substitutes into
        // trig functions exactly like the literal: sin(90deg) = 1.
        let s = style(
            &doc,
            div,
            "div { --x: 90deg; opacity: sin(var(--x)); }",
            &vp(),
        );
        match s.get("opacity") {
            Some(CssValue::Number(n)) => {
                assert!((n - 1.0).abs() < 1e-5, "opacity was {n}")
            }
            other => panic!("expected Number opacity, got {other:?}"),
        }
        // Through var() indirection: cos(45deg) = 1/√2.
        let s = style(
            &doc,
            div,
            "div { --a: var(--b); --b: 45deg; opacity: cos(var(--a)); }",
            &vp(),
        );
        match s.get("opacity") {
            Some(CssValue::Number(n)) => {
                assert!(
                    (n - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-4,
                    "opacity was {n}"
                )
            }
            other => panic!("expected Number opacity, got {other:?}"),
        }
        // Via the var() fallback: sin(30deg) = 0.5.
        let s = style(
            &doc,
            div,
            "div { opacity: sin(var(--undefined, 30deg)); }",
            &vp(),
        );
        match s.get("opacity") {
            Some(CssValue::Number(n)) => {
                assert!((n - 0.5).abs() < 1e-5, "opacity was {n}")
            }
            other => panic!("expected Number opacity, got {other:?}"),
        }
        // An angle-typed var() still fails where a number is required:
        // the type distinction survives substitution.
        let s = style(
            &doc,
            div,
            "div { --x: 90deg; opacity: calc(var(--x)); }",
            &vp(),
        );
        assert_eq!(s.get("opacity"), Some(&CssValue::Number(1.0)));
    }

    // ---- w56 follow-up: deep var() chains keep angle typing ----

    /// The angle pre-pass (`Styler::angle_leaf_of_custom_prop` /
    /// `custom_prop_angle_radians`) follows `var()` chains beyond a
    /// single link: mixed units, calc() leaves, mid-chain fallbacks,
    /// type rejection, cycles, and over-depth chains all behave.
    #[test]
    fn var_chain_deep_angle_reaches_trig() {
        let (doc, div, _) = sample();
        // The expected values below must differ from opacity's initial
        // value (1.0): a test that passes with the angle pre-pass deleted
        // (leaving opacity at 1.0) would be vacuous and prove nothing
        // about the pre-pass. sin(0.5turn) = sin(π) = 0 and
        // sin(60deg) = √3/2 ≈ 0.8660 both sit well clear of 1.0.
        // 1. 5-link chain, mixed units: sin(0.5turn) = sin(π) ≈ 0.
        let s = style(
            &doc,
            div,
            "div { --a: var(--b); --b: var(--c); --c: var(--d); --d: var(--e); \
             --e: 0.5turn; opacity: sin(var(--a)); }",
            &vp(),
        );
        match s.get("opacity") {
            Some(CssValue::Number(n)) => {
                assert!(n.abs() < 1e-5, "opacity was {n}")
            }
            other => panic!("expected Number opacity, got {other:?}"),
        }
        // 2. calc() angle leaf at depth: sin(30deg + 30deg) = sin(60deg) ≈ 0.8660.
        let s = style(
            &doc,
            div,
            "div { --a: var(--b); --b: var(--c); --c: var(--d); --d: var(--e); \
             --e: calc(30deg + 30deg); opacity: sin(var(--a)); }",
            &vp(),
        );
        match s.get("opacity") {
            Some(CssValue::Number(n)) => {
                assert!((n - 0.8660254).abs() < 1e-4, "opacity was {n}")
            }
            other => panic!("expected Number opacity, got {other:?}"),
        }
        // 3. Mid-chain fallback: sin(30deg) = 0.5.
        let s = style(
            &doc,
            div,
            "div { --a: var(--b); --b: var(--undefined, 30deg); \
             opacity: sin(var(--a)); }",
            &vp(),
        );
        match s.get("opacity") {
            Some(CssValue::Number(n)) => {
                assert!((n - 0.5).abs() < 1e-5, "opacity was {n}")
            }
            other => panic!("expected Number opacity, got {other:?}"),
        }
        // 4. Angle rejected through a chain where a number is required:
        // `opacity: calc(var(--a))` drops, leaving the initial 1.0.
        let s = style(
            &doc,
            div,
            "div { --a: var(--b); --b: var(--c); --c: 90deg; \
             opacity: calc(var(--a)); }",
            &vp(),
        );
        assert_eq!(s.get("opacity"), Some(&CssValue::Number(1.0)));
        // 5. Cyclic chain: the var() is invalid at computed-value time,
        // the sin() argument drops, and opacity stays at initial 1.0
        // (no hang). The timing bound turns a hang regression into a
        // test failure instead of a stalled suite.
        let start = std::time::Instant::now();
        let s = style(
            &doc,
            div,
            "div { --a: var(--b); --b: var(--a); opacity: sin(var(--a)); }",
            &vp(),
        );
        assert_eq!(s.get("opacity"), Some(&CssValue::Number(1.0)));
        assert!(
            start.elapsed() < std::time::Duration::from_secs(5),
            "2-cycle var() chain did not terminate promptly"
        );
        // 6. Over-depth chain: 70 links exceed MAX_MATH_DEPTH=64, so the
        // var() drops gracefully (no stack blowup) and opacity stays 1.0.
        let n = 70;
        let mut css = String::from("div { ");
        for i in 0..n {
            css.push_str(&format!("--a{i}: var(--a{}); ", i + 1));
        }
        css.push_str(&format!("--a{n}: 90deg; opacity: sin(var(--a0)); }}"));
        let start = std::time::Instant::now();
        let s = style(&doc, div, &css, &vp());
        let elapsed = start.elapsed();
        assert_eq!(s.get("opacity"), Some(&CssValue::Number(1.0)));
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "70-link var() chain took {elapsed:?}"
        );
    }

    // ---- B11: background-attachment is computed ----

    #[test]
    fn background_shorthand_fixed_attachment() {
        let (doc, div, _) = sample();
        let s = style(&doc, div, "div { background: url(x) fixed; }", &vp());
        assert_eq!(s.get_str("background-attachment"), Some("fixed"));
    }

    // ---- B12: font shorthand with vw / calc ----

    #[test]
    fn font_shorthand_vw_and_calc_size() {
        let (doc, div, _) = sample();
        let s = style(&doc, div, "div { font: 2vw serif; }", &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(2.0, LengthUnit::Vw))
        );
        assert_eq!(s.get_str("font-family"), Some("serif"));
        // `calc()` size with a line-height.
        let s = style(
            &doc,
            div,
            "div { font: italic bold calc(16px + 2px)/1.5 serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(18.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-style"), Some("italic"));
        assert_eq!(s.get_str("font-weight"), Some("bold"));
    }

    #[test]
    fn deferred_font_system_keyword_via_var() {
        let (doc, div, _) = sample();
        // `font: var(--x)` with `--x: caption` (etc.): the system font
        // keyword expands to the UA metrics, mirroring the eager path
        // (CSS Fonts 3 §3.2).
        for kw in [
            "caption",
            "icon",
            "menu",
            "message-box",
            "small-caption",
            "status-bar",
        ] {
            let s = style(
                &doc,
                div,
                &format!("div {{ --x: {kw}; font: var(--x); }}"),
                &vp(),
            );
            let expected_size = if kw == "small-caption" { 12.0 } else { 13.0 };
            assert_eq!(
                s.get("font-size"),
                Some(&CssValue::Length(expected_size, LengthUnit::Px)),
                "--x: {kw}"
            );
            assert_eq!(s.get_str("font-family"), Some("sans-serif"), "--x: {kw}");
            assert_eq!(s.get_str("font-style"), Some("normal"), "--x: {kw}");
            assert_eq!(s.get_str("font-variant"), Some("normal"), "--x: {kw}");
            assert_eq!(s.get_str("font-weight"), Some("normal"), "--x: {kw}");
            assert_eq!(s.get_str("line-height"), Some("normal"), "--x: {kw}");
        }
        // Case-insensitive through substitution.
        let s = style(&doc, div, "div { --x: MENU; font: var(--x); }", &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(13.0, LengthUnit::Px))
        );
        // The keyword must be the sole value: a combination is dropped,
        // leaving the initial values in place.
        let base = style(&doc, div, "div { }", &vp());
        for bad in [
            "div { --x: caption 12px; font: var(--x); }",
            "div { --x: bold caption; font: var(--x); }",
            "div { --x: caption serif; font: var(--x); }",
        ] {
            let s = style(&doc, div, bad, &vp());
            assert_eq!(s.get("font-size"), base.get("font-size"), "{bad}");
            assert_eq!(
                s.get_str("font-family"),
                base.get_str("font-family"),
                "{bad}"
            );
        }
        // `!important` propagates through the deferred expansion by rank
        // inheritance: the longhands produced by the shorthand beat a later
        // normal longhand declaration.
        let s = style(
            &doc,
            div,
            "div { --x: caption; font: var(--x) !important; font-size: 99px; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(13.0, LengthUnit::Px)),
            "!important system-font shorthand must beat later normal longhand"
        );
        assert_eq!(s.get_str("font-family"), Some("sans-serif"));
        // Deferred sole-value rejection: two keywords in the variable must be
        // rejected, leaving the initial values in place.
        let s = style(
            &doc,
            div,
            "div { --x: caption caption; font: var(--x); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            base.get("font-size"),
            "--x: caption caption"
        );
        assert_eq!(
            s.get_str("font-family"),
            base.get_str("font-family"),
            "--x: caption caption"
        );
        // A `var()` fallback supplying the keyword is accepted.
        let s = style(&doc, div, "div { font: var(--missing, caption); }", &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(13.0, LengthUnit::Px)),
            "var(--missing, caption) fallback"
        );
        assert_eq!(
            s.get_str("font-family"),
            Some("sans-serif"),
            "var(--missing, caption) fallback"
        );
    }

    #[test]
    fn font_weight_rejects_fractional() {
        let (doc, div, _) = sample();
        // `font-weight` is inherited; a rejected declaration falls back
        // to the initial `normal` instead of applying.
        let normal = || Some(CssValue::Keyword("normal".into()));
        // Direct: `400.5` is not an integer weight.
        let s = style(&doc, div, "div { font-weight: 400.5; }", &vp());
        assert_eq!(s.get("font-weight"), normal().as_ref());
        // Through `var()`: the fractional value survives substitution
        // and must still be rejected at computed-value time.
        let s = style(
            &doc,
            div,
            "div { --w: 400.5; font-weight: var(--w); }",
            &vp(),
        );
        assert_eq!(s.get("font-weight"), normal().as_ref());
        // Integer boundaries: 1 and 1000 are accepted, 0 and 1001 are not.
        for (css, expected) in [
            ("1", Some(CssValue::Number(1.0))),
            ("1000", Some(CssValue::Number(1000.0))),
            ("400", Some(CssValue::Number(400.0))),
            ("0", normal()),
            ("1001", normal()),
            ("-1", normal()),
        ] {
            let s = style(&doc, div, &format!("div {{ font-weight: {css}; }}"), &vp());
            assert_eq!(
                s.get("font-weight"),
                expected.as_ref(),
                "font-weight: {css}"
            );
            let s = style(
                &doc,
                div,
                &format!("div {{ --w: {css}; font-weight: var(--w); }}"),
                &vp(),
            );
            assert_eq!(
                s.get("font-weight"),
                expected.as_ref(),
                "font-weight: var(--w) with --w: {css}"
            );
        }
    }

    #[test]
    fn deferred_font_shorthand_with_var_and_line_height() {
        let (doc, div, _) = sample();
        // `font: var(--f)/1.5 serif`: the `/` splits at the token level
        // before deferring, and the cascade re-attaches line-height on
        // substitution.
        let s = style(
            &doc,
            div,
            "div { --f: 12px; font: var(--f)/1.5 serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-family"), Some("serif"));
        assert_eq!(s.get_str("font-style"), Some("normal"));
        assert_eq!(s.get_str("font-weight"), Some("normal"));
        // All three of size, line-height, and family from `var()`.
        let s = style(
            &doc,
            div,
            "div { --a: italic bold 14px; --b: 2; --c: serif; font: var(--a)/var(--b) var(--c); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(14.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(2.0)));
        assert_eq!(s.get_str("font-family"), Some("serif"));
        assert_eq!(s.get_str("font-style"), Some("italic"));
        assert_eq!(s.get_str("font-weight"), Some("bold"));
    }

    #[test]
    fn deferred_font_shorthand_slash_in_var_fallback_expands() {
        let (doc, div, _) = sample();
        // The `/` is inside the `var()` fallback: it cannot be split at
        // the token level, but the fallback is kept as verbatim tokens so
        // substitution still sees the slash — `font: var(--f, 12px/1.5)
        // serif` is `font: 12px/1.5 serif`.
        let s = style(&doc, div, "div { font: var(--f, 12px/1.5) serif; }", &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-family"), Some("serif"));
        // A defined property wins over the fallback.
        let s = style(
            &doc,
            div,
            "div { --f: 20px; font: var(--f, 12px/1.5) serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(20.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&kw("normal")));
        assert_eq!(s.get_str("font-family"), Some("serif"));
    }

    #[test]
    fn deferred_font_shorthand_var_slash_fallback_with_siblings() {
        let (doc, div, _) = sample();
        // The slash arrives via the fallback while siblings sit on both
        // sides: the before/after groups merge token-wise.
        let s = style(
            &doc,
            div,
            "div { font: italic var(--f, bold 12px/1.5) serif; }",
            &vp(),
        );
        assert_eq!(s.get_str("font-style"), Some("italic"));
        assert_eq!(s.get_str("font-weight"), Some("bold"));
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-family"), Some("serif"));
        // Slash outside the `var()`, plain fallback inside.
        let s = style(&doc, div, "div { font: 16px/1.5 var(--f, serif); }", &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(16.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-family"), Some("serif"));
        // A second slash via the fallback is still two top-level
        // slashes: invalid at computed-value time.
        let s = style(
            &doc,
            div,
            "div { font: 16px/1.5 var(--f, 2/3) serif; }",
            &vp(),
        );
        assert_eq!(s.get("font-size"), Some(&kw("medium")));
    }

    #[test]
    fn deferred_font_shorthand_line_height_var_forms() {
        let (doc, div, _) = sample();
        // Plain `/ line-height`: the specified line-height survives as a
        // computed value on the element.
        let s = style(&doc, div, "div { font: 12px/1.5 serif; }", &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-family"), Some("serif"));
        // `line-height` as a `var()` reference after the `/`:
        // substitution runs through the deferred expansion before the
        // longhands apply.
        let s = style(
            &doc,
            div,
            "div { --lh: 1.5; font: 12px/var(--lh) serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-family"), Some("serif"));
        // Full prefix with a `var()` fallback line-height.
        let s = style(
            &doc,
            div,
            "div { font: italic bold 12px/var(--lh, 1.5) serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-style"), Some("italic"));
        assert_eq!(s.get_str("font-weight"), Some("bold"));
        assert_eq!(s.get_str("font-family"), Some("serif"));
        // A `var()` nested inside `calc()` in the line-height position:
        // the math tree still counts as containing `var()`, so the
        // declaration takes the deferred path and the substituted
        // line-height evaluates in the cascade after expansion.
        let s = style(
            &doc,
            div,
            "div { --lh: 1.5; font: 12px/calc(var(--lh) * 2) serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(3.0)));
        assert_eq!(s.get_str("font-family"), Some("serif"));
        // Same for a `var()` nested in `calc()` in the size position.
        let s = style(
            &doc,
            div,
            "div { --s: 12px; font: calc(var(--s))/1.5 serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-family"), Some("serif"));
        // Invalid: missing size.
        let s = style(&doc, div, "div { font: /1.5 serif; }", &vp());
        assert_eq!(s.get("font-size"), Some(&kw("medium")));
        assert_eq!(s.get("line-height"), Some(&kw("normal")));
        // Invalid: `/` without a line-height.
        let s = style(&doc, div, "div { font: 12px/ serif; }", &vp());
        assert_eq!(s.get("font-size"), Some(&kw("medium")));
        assert_eq!(s.get("line-height"), Some(&kw("normal")));
        // Invalid via `var()`: `/` with an unresolvable line-height and
        // no fallback drops the declaration.
        let s = style(&doc, div, "div { font: 12px/var(--nope) serif; }", &vp());
        assert_eq!(s.get("font-size"), Some(&kw("medium")));
        assert_eq!(s.get("line-height"), Some(&kw("normal")));
    }

    #[test]
    fn deferred_font_slash_var_comma_leading_line_height_dropped() {
        let (doc, div, _) = sample();
        // `font: 12px/var(--g)` with `--g: 1.5, serif` substitutes
        // token-wise to `font: 12px/1.5, serif`: a leading comma in
        // the family list, invalid like the eager parser, so the
        // declaration is dropped.
        let s = style(
            &doc,
            div,
            "div { --g: 1.5, serif; font: 12px/var(--g); }",
            &vp(),
        );
        assert_eq!(s.get("font-size"), Some(&kw("medium")));
        assert_eq!(s.get("line-height"), Some(&kw("normal")));
        // Same when the substituted line-height is a keyword: the
        // comma group still starts the family list with a comma even
        // though "Arial" would lowercase to a valid-looking keyword.
        let s = style(
            &doc,
            div,
            "div { --g: Arial, serif; font: 12px/var(--g); }",
            &vp(),
        );
        assert_eq!(s.get("font-size"), Some(&kw("medium")));
        assert_eq!(s.get("line-height"), Some(&kw("normal")));
        // A family word after the line-height in the first comma group
        // keeps the declaration valid.
        let s = style(
            &doc,
            div,
            "div { --f: 12px/1.5 Arial, serif; font: var(--f); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("serif".into()),
            ]))
        );
        // Nested-var commas after a flat line-height stay valid too.
        let s = style(
            &doc,
            div,
            "div { --g: Arial, serif; --f: 12px/1.5 var(--g); font: var(--f); }",
            &vp(),
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("serif".into()),
            ]))
        );
    }

    #[test]
    fn deferred_font_slash_keeps_multiword_family_together() {
        let (doc, div, _) = sample();
        let one_family = |s: &ComputedStyle| {
            assert_eq!(
                s.get("font-size"),
                Some(&CssValue::Length(12.0, LengthUnit::Px))
            );
            assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
            assert_eq!(
                s.get("font-family"),
                Some(&CssValue::String("Times New Roman".into()))
            );
        };
        // Eager baseline: one family.
        one_family(&style(
            &doc,
            div,
            "div { font: 12px/1.5 Times New Roman; }",
            &vp(),
        ));
        // Size from `var()`: the words after the `/` must still form one
        // family, not three single-word families.
        one_family(&style(
            &doc,
            div,
            "div { --s: 12px; font: var(--s)/1.5 Times New Roman; }",
            &vp(),
        ));
        // Family from `var()`.
        one_family(&style(
            &doc,
            div,
            "div { --f: Times New Roman; font: 12px/1.5 var(--f); }",
            &vp(),
        ));
        // Line-height from `var()`.
        one_family(&style(
            &doc,
            div,
            "div { --lh: 1.5; font: 12px/var(--lh) Times New Roman; }",
            &vp(),
        ));
        // Line-height from a `var()` *fallback* whose value runs past the
        // line-height into the family: the fallback splices token-wise
        // like a defined custom property, so "Times" merges with the
        // following loose words into one family instead of splitting.
        one_family(&style(
            &doc,
            div,
            "div { font: 12px/var(--missing, 1.5 Times) New Roman; }",
            &vp(),
        ));
        // Comma boundaries still split: two families, not one merged run.
        let s = style(
            &doc,
            div,
            "div { --f: Arial, sans-serif; font: 12px/1.5 var(--f); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("sans-serif".into()),
            ]))
        );
        // Nested `/` marker: `font: var(--f), serif` with
        // `--f: 12px/1.5 Times New Roman` keeps the multi-word family
        // together and appends the trailing comma group.
        let s = style(
            &doc,
            div,
            "div { --f: 12px/1.5 Times New Roman; font: var(--f), serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Times New Roman".into()),
                CssValue::String("serif".into()),
            ]))
        );
        // Adjacent quoted strings via `var()` stay invalid, like the eager
        // `font: 12px/1.5 "A" "B"`: the declaration is dropped.
        let s = style(
            &doc,
            div,
            "div { --f: \"A\" \"B\"; font: 12px/1.5 var(--f); }",
            &vp(),
        );
        assert_eq!(s.get("font-size"), Some(&kw("medium")));
        assert_eq!(s.get("font-family"), Some(&kw("serif")));
    }

    #[test]
    fn deferred_font_loose_string_before_splice_group_dropped() {
        let (doc, div, _) = sample();
        // `font: 12px "A" var(--f)` with `--f: "B", "C"` substitutes to
        // `12px "A" "B", "C"`: the splice group is adjacent to the loose
        // `"A"` (no comma between them), so the declaration is dropped
        // like the eager parser drops `font: 12px "A" "B", "C"`.
        let s = style(
            &doc,
            div,
            "div { --f: \"B\", \"C\"; font: 12px \"A\" var(--f); }",
            &vp(),
        );
        assert_eq!(s.get("font-size"), Some(&kw("medium")));
        assert_eq!(s.get("font-family"), Some(&kw("serif")));
    }

    #[test]
    fn deferred_font_splice_group_idents_after_loose_string_kept() {
        let (doc, div, _) = sample();
        // Only a *string* is rejected after a loose top-level string:
        // `font: 12px "A" var(--f)` with `--f: b, c` substitutes to
        // `12px "A" b, c`, three valid families.
        let s = style(
            &doc,
            div,
            "div { --f: b, c; font: 12px \"A\" var(--f); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("A".into()),
                CssValue::String("b".into()),
                CssValue::String("c".into()),
            ]))
        );
    }

    #[test]
    fn deferred_font_comma_separated_strings_via_var_kept() {
        let (doc, div, _) = sample();
        // `font: 12px var(--f)` with `--f: "A", "B"`: the comma boundary
        // resets adjacency, so two families result.
        let s = style(
            &doc,
            div,
            "div { --f: \"A\", \"B\"; font: 12px var(--f); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("A".into()),
                CssValue::String("B".into()),
            ]))
        );
    }

    #[test]
    fn deferred_font_shorthand_var_comma_and_slash_fallback() {
        let (doc, div, _) = sample();
        // Comma groups in the fallback keep their boundaries: two
        // families, not one merged "arial sans-serif".
        let s = style(
            &doc,
            div,
            "div { font: 12px var(--f, Arial, sans-serif); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("sans-serif".into()),
            ]))
        );
        // Slash and commas together in one fallback: `font: 12px/1.5
        // Arial, sans-serif serif`.
        let s = style(
            &doc,
            div,
            "div { font: var(--f, 12px/1.5 Arial, sans-serif) serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("sans-serif".into()),
                CssValue::String("serif".into()),
            ]))
        );
    }

    #[test]
    fn var_fallback_boundary_skips_nested_commas() {
        let (doc, div, _) = sample();
        // Commas inside a nested function do not start the fallback.
        let s = style(&doc, div, "div { color: var(--c, rgb(1, 2, 3)); }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 1,
                g: 2,
                b: 3,
                a: 255
            }))
        );
        // A nested `var()` fallback with its own comma still resolves.
        let s = style(&doc, div, "div { color: var(--a, var(--b, red)); }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // A slash fallback in a non-slash property stays invalid at
        // computed-value time (the declaration drops).
        let s = style(&doc, div, "div { margin-top: var(--m, 1px/2px); }", &vp());
        assert_eq!(
            s.get("margin-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // ... but a slash-aware shorthand accepts the same fallback.
        let s = style(
            &doc,
            div,
            "div { border-radius: var(--r, 1px/2px); }",
            &vp(),
        );
        let corner = CssValue::List(vec![
            CssValue::Length(1.0, LengthUnit::Px),
            CssValue::Length(2.0, LengthUnit::Px),
        ]);
        assert_eq!(s.get("border-top-left-radius"), Some(&corner));
        assert_eq!(s.get("border-bottom-right-radius"), Some(&corner));
    }

    #[test]
    fn custom_property_slash_survives_var_substitution() {
        let (doc, div, _) = sample();
        // `--r` is stored as verbatim tokens; substituting it into
        // `border-radius` must still see the `/` separating horizontal
        // from vertical radii.
        let s = style(
            &doc,
            div,
            "div { --r: 1px/2px; border-radius: var(--r); }",
            &vp(),
        );
        let corner = CssValue::List(vec![
            CssValue::Length(1.0, LengthUnit::Px),
            CssValue::Length(2.0, LengthUnit::Px),
        ]);
        for c in [
            "border-top-left-radius",
            "border-top-right-radius",
            "border-bottom-right-radius",
            "border-bottom-left-radius",
        ] {
            assert_eq!(s.get(c), Some(&corner), "{c}");
        }
    }

    #[test]
    fn resolve_value_ref_composite_paths() {
        // `ResolveWork::ValueRef` resolves the specified value (and
        // `env` values) by reference instead of cloning them into the
        // work stack. Every composite shape must produce exactly the
        // owned path's result.
        let (doc, div, _) = sample();
        let s = style(
            &doc,
            div,
            "div { --r: 1px/2px; --t: 1px solid red; --m: 4px 8px; \
              border-radius: var(--r); border-top: var(--t); margin: var(--m); \
              grid-template-columns: repeat(auto-fill, 10px 20px); }",
            &vp(),
        );
        // Borrowed `Tokens` via `var()`: the `/` still splits.
        let corner = CssValue::List(vec![
            CssValue::Length(1.0, LengthUnit::Px),
            CssValue::Length(2.0, LengthUnit::Px),
        ]);
        assert_eq!(s.get("border-top-left-radius"), Some(&corner));
        // Borrowed `Tokens` via `var()`: plain parse, no slash.
        assert_eq!(
            s.get("border-top-width"),
            Some(&CssValue::Length(1.0, LengthUnit::Px))
        );
        // Borrowed `Tokens` via `var()`: the custom property's own
        // value is a list-valued token stream.
        assert_eq!(
            s.get("margin-top"),
            Some(&CssValue::Length(4.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("margin-right"),
            Some(&CssValue::Length(8.0, LengthUnit::Px))
        );
        // Borrowed `Repeat` descriptor entry keeps its shape.
        assert_eq!(
            s.get("grid-template-columns"),
            Some(&CssValue::Repeat {
                mode: RepeatMode::Fill,
                tracks: vec![
                    CssValue::Length(10.0, LengthUnit::Px),
                    CssValue::Length(20.0, LengthUnit::Px),
                ],
            })
        );
    }

    #[test]
    fn custom_property_slash_inside_font_substitution() {
        let (doc, div, _) = sample();
        // The `/` lives inside the custom property this time: the
        // deferred font expansion must still find it after substitution.
        let s = style(
            &doc,
            div,
            "div { --f: italic 12px/1.5 serif; font: var(--f); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-style"), Some("italic"));
        assert_eq!(s.get_str("font-family"), Some("serif"));
    }

    #[test]
    fn deferred_font_shorthand_slash_var_spliced_in_list() {
        let (doc, div, _) = sample();
        // `font: var(--x) serif` with `--x: 12px/1.5`: substitution is
        // token-wise, so the `/` inside the custom property still
        // separates font-size from line-height even though the `var()`
        // sits inside a larger list.
        let s = style(
            &doc,
            div,
            "div { --x: 12px/1.5; font: var(--x) serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-family"), Some("serif"));
        // Siblings on both sides of the `var()` merge into the right
        // slash groups: `italic 12px/1.5 serif`.
        let s = style(
            &doc,
            div,
            "div { --x: 12px/1.5; font: italic var(--x) serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-style"), Some("italic"));
        assert_eq!(s.get_str("font-family"), Some("serif"));
        // A slash through a nested `var()` chain keeps its marker too.
        let s = style(
            &doc,
            div,
            "div { --y: 12px/1.5; --x: var(--y); font: var(--x) serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-family"), Some("serif"));
        // Two top-level slashes stay invalid at computed-value time.
        let s = style(
            &doc,
            div,
            "div { --a: 12px/1.5; --b: 2/3; font: var(--a) var(--b) serif; }",
            &vp(),
        );
        assert_eq!(s.get("font-size"), Some(&kw("medium")));
    }

    #[test]
    fn deferred_font_shorthand_var_comma_family_list() {
        let (doc, div, _) = sample();
        // A comma-separated family list inside the custom property keeps
        // its commas through substitution instead of being dropped.
        let s = style(
            &doc,
            div,
            "div { --f: Arial, sans-serif; font: 12px var(--f); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("sans-serif".into()),
            ]))
        );
        // Comma list combined with a slash in one variable: `12px/1.5
        // Arial, sans-serif`.
        let s = style(
            &doc,
            div,
            "div { --f: 12px/1.5 Arial, sans-serif; font: var(--f); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("sans-serif".into()),
            ]))
        );
    }

    #[test]
    fn deferred_font_shorthand_nested_slash_marker_with_comma_sibling() {
        let (doc, div, _) = sample();
        // `font: var(--f), serif` with `--f: 12px/1.5 Arial` substitutes
        // to `[[[12px], [1.5, Arial]], [serif]]`: the `/` marker is nested
        // one level down, invisible to the top-level slash attempt. The
        // nested marker must still be unwrapped, not dropped as invalid.
        let s = style(
            &doc,
            div,
            "div { --f: 12px/1.5 Arial; font: var(--f), serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("serif".into()),
            ]))
        );
        // Genuine two comma groups (no nested slash marker) still take
        // the comma path: `--f: 12px arial` is size + family, and the
        // sibling comma adds a second family.
        let s = style(
            &doc,
            div,
            "div { --f: 12px arial; font: var(--f), serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("line-height"),
            Some(&CssValue::Keyword("normal".into()))
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("arial".into()),
                CssValue::String("serif".into()),
            ]))
        );
        // Three comma groups: `font: var(--f), serif, sans-serif` with
        // `--f: 12px/1.5 Arial` substitutes to
        // `[[[12px], [1.5, Arial]], [serif], [sans-serif]]`. The nested
        // slash marker must be found by scanning the groups, not dropped
        // as an invalid comma expansion.
        let s = style(
            &doc,
            div,
            "div { --f: 12px/1.5 Arial; font: var(--f), serif, sans-serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("serif".into()),
                CssValue::String("sans-serif".into()),
            ]))
        );
    }

    #[test]
    fn font_nested_slash_marker_group_zero_only() {
        let (doc, div, _) = sample();
        // Eager (no `var()`): `12px/30px "Helvetica Neue", Arial,
        // sans-serif` still parses with line-height 30px and all three
        // families; quoted names keep their case, idents lowercase.
        let s = style(
            &doc,
            div,
            "div { font: italic 12px/30px \"Helvetica Neue\", Arial, sans-serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("line-height"),
            Some(&CssValue::Length(30.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Helvetica Neue".into()),
                CssValue::String("Arial".into()),
                CssValue::String("sans-serif".into()),
            ]))
        );
        assert_eq!(s.get_str("font-style"), Some("italic"));
        // Wave-A shape: `font: var(--f), serif` with
        // `--f: 12px/1.5 Arial` lands the `/` marker in comma group 0
        // (the size portion), where it is still honored, with the later
        // groups kept as trailing families.
        let s = style(
            &doc,
            div,
            "div { --f: 12px/1.5 Arial; font: var(--f), serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("serif".into()),
            ]))
        );
        // A `/` marker nested in a LATER comma group is a family
        // position, not the size/line-height divider: `16px serif`
        // parses as group 0, and the marker-shaped group 1 must not be
        // misread as size 12px / line-height 1.5. The declaration is
        // dropped and the longhands keep their initial values.
        let s = style(
            &doc,
            div,
            "div { --f: 12px/1.5 Arial; font: 16px serif, var(--f); }",
            &vp(),
        );
        assert_eq!(s.get("font-size"), Some(&kw("medium")));
        assert_eq!(s.get("line-height"), Some(&kw("normal")));
        assert_eq!(s.get("font-family"), Some(&kw("serif")));
    }

    #[test]
    fn font_eager_deferred_case_parity() {
        let (doc, div, _) = sample();
        // The whole-value `var()` form must compute exactly what the
        // eager literal computes, including family identifier case and
        // the size/line-height slash pair.
        let eager = style(
            &doc,
            div,
            "div { font: italic bold 12px/1.5 Arial, sans-serif; }",
            &vp(),
        );
        let deferred = style(
            &doc,
            div,
            "div { --x: italic bold 12px/1.5 Arial, sans-serif; font: var(--x); }",
            &vp(),
        );
        for prop in [
            "font-style",
            "font-weight",
            "font-size",
            "line-height",
            "font-family",
        ] {
            assert_eq!(eager.get(prop), deferred.get(prop), "{prop}");
        }
        assert_eq!(
            deferred.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("sans-serif".into()),
            ]))
        );
        assert_eq!(deferred.get("line-height"), Some(&CssValue::Number(1.5)));
    }

    #[test]
    fn font_eager_deferred_color_name_parity() {
        let (doc, div, _) = sample();
        // A color name in the family position: the eager token parser
        // reads `Red` straight from the tokens as a family name, so the
        // deferred path must hand the font expansion the raw identifier
        // instead of routing it through the value parser's color
        // canonicalization (`transparent`/`currentcolor` too).
        for family in ["Red", "red", "transparent", "currentcolor"] {
            let eager = style(&doc, div, &format!("div {{ font: 12px {family}; }}"), &vp());
            let deferred = style(
                &doc,
                div,
                &format!("div {{ --x: 12px {family}; font: var(--x); }}"),
                &vp(),
            );
            for prop in [
                "font-style",
                "font-weight",
                "font-size",
                "line-height",
                "font-family",
            ] {
                assert_eq!(eager.get(prop), deferred.get(prop), "{prop} {family}");
            }
            assert_eq!(
                deferred.get("font-family"),
                Some(&CssValue::String(family.into())),
                "{family}"
            );
        }
    }

    #[test]
    fn deferred_font_shorthand_var_multiword_family() {
        let (doc, div, _) = sample();
        // Quoted multi-word family name.
        let s = style(
            &doc,
            div,
            "div { --f: \"Times New Roman\"; font: 12px var(--f); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::String("Times New Roman".into()))
        );
        // Unquoted multi-word family name: the ident run joins with
        // spaces instead of splitting into separate families (idents are
        // lowercased by the value parser, matching the eager parser).
        let s = style(
            &doc,
            div,
            "div { --f: Times New Roman; font: 12px var(--f); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::String("Times New Roman".into()))
        );
        // Multi-word name as one comma group among several.
        let s = style(
            &doc,
            div,
            "div { --f: Times New Roman, serif; font: 12px var(--f); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Times New Roman".into()),
                CssValue::String("serif".into()),
            ]))
        );
    }

    #[test]
    fn deferred_border_radius_slash_var_spliced_in_list() {
        let (doc, div, _) = sample();
        // `border-radius: var(--r) 5px` with `--r: 1px/2px` is
        // `border-radius: 1px/2px 5px`: horizontal `[1px]`, vertical
        // `[2px, 5px]`.
        let s = style(
            &doc,
            div,
            "div { --r: 1px/2px; border-radius: var(--r) 5px; }",
            &vp(),
        );
        let pair = |h: f32, v: f32| {
            CssValue::List(vec![
                CssValue::Length(h, LengthUnit::Px),
                CssValue::Length(v, LengthUnit::Px),
            ])
        };
        assert_eq!(s.get("border-top-left-radius"), Some(&pair(1.0, 2.0)));
        assert_eq!(s.get("border-top-right-radius"), Some(&pair(1.0, 5.0)));
        assert_eq!(s.get("border-bottom-right-radius"), Some(&pair(1.0, 2.0)));
        assert_eq!(s.get("border-bottom-left-radius"), Some(&pair(1.0, 5.0)));
        // Slash variable after a literal: `5px 1px/2px`.
        let s = style(
            &doc,
            div,
            "div { --r: 1px/2px; border-radius: 5px var(--r); }",
            &vp(),
        );
        assert_eq!(s.get("border-top-left-radius"), Some(&pair(5.0, 2.0)));
        assert_eq!(s.get("border-top-right-radius"), Some(&pair(1.0, 2.0)));
    }

    #[test]
    fn deferred_shorthand_var_delimiter_markers_preserved() {
        let (doc, div, _) = sample();
        // `font: var(--x) serif` with `--x: 12px/1.5`: the `/` inside the
        // custom property must still separate font-size from line-height
        // after substitution — the declaration must not be dropped.
        let s = style(
            &doc,
            div,
            "div { --x: 12px/1.5; font: var(--x) serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-family"), Some("serif"));

        // `border-radius: var(--r)` with `--r: 4px/8px` keeps the
        // horizontal/vertical split.
        let s = style(
            &doc,
            div,
            "div { --r: 4px/8px; border-radius: var(--r); }",
            &vp(),
        );
        let corner = CssValue::List(vec![
            CssValue::Length(4.0, LengthUnit::Px),
            CssValue::Length(8.0, LengthUnit::Px),
        ]);
        for c in [
            "border-top-left-radius",
            "border-top-right-radius",
            "border-bottom-right-radius",
            "border-bottom-left-radius",
        ] {
            assert_eq!(s.get(c), Some(&corner), "{c}");
        }

        // Style/weight prefix + slash + comma-separated families, all
        // inside one variable: full expansion preserved including the
        // comma.
        let s = style(
            &doc,
            div,
            "div { --f: italic bold 12px/30px Georgia, serif; font: var(--f); }",
            &vp(),
        );
        assert_eq!(s.get_str("font-style"), Some("italic"));
        assert_eq!(s.get_str("font-weight"), Some("bold"));
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("line-height"),
            Some(&CssValue::Length(30.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Georgia".into()),
                CssValue::String("serif".into()),
            ]))
        );

        // A quoted multi-word family through `var()` stays one family
        // name (not spliced into separate tokens).
        let s = style(
            &doc,
            div,
            "div { --f: \"Times New Roman\"; font: italic 12px var(--f); }",
            &vp(),
        );
        assert_eq!(s.get_str("font-style"), Some("italic"));
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::String("Times New Roman".into()))
        );
    }

    #[test]
    fn deferred_font_shorthand_comma_in_declaration_tokens() {
        let (doc, div, _) = sample();
        // A top-level comma in the declaration's own tokens must keep
        // its family boundary through substitution, exactly as if the
        // `var()` had been substituted before parsing.
        let two = || {
            CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("serif".into()),
            ])
        };
        // `font: var(--f), serif` with `--f: 12px Arial`.
        let s = style(
            &doc,
            div,
            "div { --f: 12px Arial; font: var(--f), serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("font-family"), Some(&two()));
        // `font: 12px var(--f), serif` with `--f: Arial`.
        let s = style(
            &doc,
            div,
            "div { --f: Arial; font: 12px var(--f), serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("font-family"), Some(&two()));
        // `font: 12px Arial, var(--f)` with `--f: serif`.
        let s = style(
            &doc,
            div,
            "div { --f: serif; font: 12px Arial, var(--f); }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("font-family"), Some(&two()));
        // Families containing commas inside the variable splice into the
        // enclosing comma structure: `font: 12px Arial, sans-serif,
        // serif`.
        let s = style(
            &doc,
            div,
            "div { --f: Arial, sans-serif; font: 12px var(--f), serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("sans-serif".into()),
                CssValue::String("serif".into()),
            ]))
        );
        // Three comma groups in the declaration itself.
        let s = style(
            &doc,
            div,
            "div { --a: 12px Arial; --b: sans-serif; font: var(--a), var(--b), serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("font-family"),
            Some(&CssValue::List(vec![
                CssValue::String("Arial".into()),
                CssValue::String("sans-serif".into()),
                CssValue::String("serif".into()),
            ]))
        );
    }

    #[test]
    fn deferred_slash_var_memoized_across_declarations() {
        let (doc, div, _) = sample();
        // `--x` is used standalone (`line-height`, resolved first and
        // memoized) and inside a list (`font`): the memoized slash
        // marker must survive for the in-list use.
        let s = style(
            &doc,
            div,
            "div { --x: 12px/1.5; line-height: var(--x); font: var(--x) serif; }",
            &vp(),
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(12.0, LengthUnit::Px))
        );
        assert_eq!(s.get("line-height"), Some(&CssValue::Number(1.5)));
        assert_eq!(s.get_str("font-family"), Some("serif"));
    }

    #[test]
    fn custom_property_grid_line_slash_via_var() {
        let (doc, div, _) = sample();
        // `--gc: 1/3` substituted into `grid-column` keeps the `/` as
        // the start/end marker the grid grammar understands.
        let s = style(
            &doc,
            div,
            "div { --gc: 1/3; grid-column: var(--gc); }",
            &vp(),
        );
        assert_eq!(
            s.get("grid-column"),
            Some(&CssValue::List(vec![
                CssValue::List(vec![CssValue::Number(1.0)]),
                CssValue::List(vec![CssValue::Number(3.0)]),
            ]))
        );
    }

    #[test]
    fn custom_property_two_slashes_invalid_at_computed_value_time() {
        let (doc, div, _) = sample();
        // `--x: 1/2/3` is a valid custom-property declaration (kept, not
        // dropped at parse), but the stream cannot parse as a width, so
        // the `var()` fallback applies ...
        let s = style(
            &doc,
            div,
            "div { --x: 1/2/3; width: var(--x, 10px); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
        // ... and without a fallback the declaration is invalid at
        // computed-value time (width falls back to its initial value).
        let s = style(&doc, div, "div { --x: 1/2/3; width: var(--x); }", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
    }

    #[test]
    fn gap_fr_invalid_at_computed_value_time() {
        let (doc, div, _) = sample();
        // `fr` is not a valid gap: the declaration is dropped (row-gap
        // falls back to its initial value), never silently zeroed.
        let s = style(&doc, div, "div { gap: 1fr; }", &vp());
        assert_eq!(
            s.get("row-gap"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("column-gap"),
            // `column-gap`'s initial is `normal` per spec (1em in
            // multicol, 0 in grid/flex); only `row-gap` keeps the
            // zero-length initial in this engine.
            Some(&kw("normal"))
        );
        let s = style(&doc, div, "div { row-gap: 2fr; }", &vp());
        assert_eq!(
            s.get("row-gap"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // ... including through `var()` substitution.
        let s = style(&doc, div, "div { --g: 1fr; row-gap: var(--g); }", &vp());
        assert_eq!(
            s.get("row-gap"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // Sanity: real gaps still compute.
        let s = style(&doc, div, "div { gap: 8px 4px; }", &vp());
        assert_eq!(
            s.get("row-gap"),
            Some(&CssValue::Length(8.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("column-gap"),
            Some(&CssValue::Length(4.0, LengthUnit::Px))
        );
        let s = style(&doc, div, "div { row-gap: normal; }", &vp());
        assert_eq!(s.get("row-gap"), Some(&kw("normal")));
    }

    #[test]
    fn gap_shorthand_var_substitution_expands_to_longhands() {
        let (doc, div, _) = sample();
        // `gap: var(--g)` is deferred with the var unresolved; after
        // substitution it must expand into row-gap/column-gap, not land
        // in the unused `gap` slot.
        let s = style(&doc, div, "div { --g: 8px; gap: var(--g); }", &vp());
        assert_eq!(
            s.get("row-gap"),
            Some(&CssValue::Length(8.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("column-gap"),
            Some(&CssValue::Length(8.0, LengthUnit::Px))
        );
        // Two values through a single custom property.
        let s = style(&doc, div, "div { --g: 8px 4px; gap: var(--g); }", &vp());
        assert_eq!(
            s.get("row-gap"),
            Some(&CssValue::Length(8.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("column-gap"),
            Some(&CssValue::Length(4.0, LengthUnit::Px))
        );
        // `grid-gap` is the same shorthand for the deferred path.
        let s = style(
            &doc,
            div,
            "div { --g: 8px 4px; grid-gap: var(--g); }",
            &vp(),
        );
        assert_eq!(
            s.get("row-gap"),
            Some(&CssValue::Length(8.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("column-gap"),
            Some(&CssValue::Length(4.0, LengthUnit::Px))
        );
        // Fallback substitution works through the shorthand too.
        let s = style(&doc, div, "div { gap: var(--missing, 8px); }", &vp());
        assert_eq!(
            s.get("row-gap"),
            Some(&CssValue::Length(8.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("column-gap"),
            Some(&CssValue::Length(8.0, LengthUnit::Px))
        );
        // Unresolvable var with no fallback: the declaration is invalid
        // at computed-value time, so both longhands keep their initials.
        let s = style(&doc, div, "div { gap: var(--missing); }", &vp());
        assert_eq!(
            s.get("row-gap"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        assert_eq!(s.get("column-gap"), Some(&kw("normal")));
        // Three components are invalid for `gap` even after substitution.
        let s = style(&doc, div, "div { --g: 1px 2px 3px; gap: var(--g); }", &vp());
        assert_eq!(
            s.get("row-gap"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        assert_eq!(s.get("column-gap"), Some(&kw("normal")));
    }

    #[test]
    fn overflow_shorthand_expands_to_longhands() {
        let (doc, div, _) = sample();
        // A lone shorthand sets both axes.
        let s = style(&doc, div, "div { overflow: hidden; }", &vp());
        assert_eq!(s.get("overflow-x"), Some(&kw("hidden")));
        assert_eq!(s.get("overflow-y"), Some(&kw("hidden")));
        // Two values: first is `overflow-x`, second `overflow-y` (the
        // `visible` axis computes to `auto` per CSS Overflow 3 §3:
        // see `overflow_cross_axis_computed_adjustment`).
        let s = style(&doc, div, "div { overflow: hidden visible; }", &vp());
        assert_eq!(s.get("overflow-x"), Some(&kw("hidden")));
        assert_eq!(s.get("overflow-y"), Some(&kw("auto")));
        // A later longhand overrides the shorthand, and vice versa:
        // the expansion records declaration order in the cascade.
        let s = style(
            &doc,
            div,
            "div { overflow: hidden; overflow-x: visible; }",
            &vp(),
        );
        assert_eq!(s.get("overflow-x"), Some(&kw("auto")));
        assert_eq!(s.get("overflow-y"), Some(&kw("hidden")));
        let s = style(
            &doc,
            div,
            "div { overflow-x: visible; overflow: hidden; }",
            &vp(),
        );
        assert_eq!(s.get("overflow-x"), Some(&kw("hidden")));
        assert_eq!(s.get("overflow-y"), Some(&kw("hidden")));
        // Three values are invalid: the declaration is dropped and both
        // longhands keep their initial values.
        let s = style(&doc, div, "div { overflow: hidden visible auto; }", &vp());
        assert_eq!(s.get("overflow-x"), Some(&kw("visible")));
        assert_eq!(s.get("overflow-y"), Some(&kw("visible")));
        // A bad component invalidates the whole shorthand: the declaration
        // drops and both longhands keep their initial values (no partial
        // application).
        let s = style(&doc, div, "div { overflow: hidden bogus; }", &vp());
        assert_eq!(s.get("overflow-x"), Some(&kw("visible")));
        assert_eq!(s.get("overflow-y"), Some(&kw("visible")));
    }

    #[test]
    fn overflow_clip_keyword_round_trip() {
        let (doc, div, _) = sample();
        // CSS Overflow 3: `clip` is a valid value for the shorthand and
        // both longhands. It must survive the cascade — previously it
        // was dropped and the box did not clip at all.
        let s = style(&doc, div, "div { overflow: clip; }", &vp());
        assert_eq!(s.get("overflow-x"), Some(&kw("clip")));
        assert_eq!(s.get("overflow-y"), Some(&kw("clip")));
        // Two-value shorthand: first is `overflow-x`, second `overflow-y`.
        let s = style(&doc, div, "div { overflow: clip hidden; }", &vp());
        assert_eq!(s.get("overflow-x"), Some(&kw("clip")));
        assert_eq!(s.get("overflow-y"), Some(&kw("hidden")));
        // Longhands accept `clip` on their own.
        let s = style(&doc, div, "div { overflow-x: clip; }", &vp());
        assert_eq!(s.get("overflow-x"), Some(&kw("clip")));
        assert_eq!(s.get("overflow-y"), Some(&kw("visible")));
        let s = style(&doc, div, "div { overflow-y: clip; }", &vp());
        assert_eq!(s.get("overflow-x"), Some(&kw("visible")));
        assert_eq!(s.get("overflow-y"), Some(&kw("clip")));
    }

    #[test]
    fn shorthand_with_invalid_component_drops_whole_declaration() {
        let (doc, div, p) = sample();
        // `margin: 10px bogus` is invalid as a whole: every longhand
        // keeps its initial value, not just the bad one.
        let s = style(&doc, div, "div { margin: 10px bogus; }", &vp());
        for side in ["top", "right", "bottom", "left"] {
            assert_eq!(
                s.get(&format!("margin-{side}")),
                Some(&CssValue::Length(0.0, LengthUnit::Px))
            );
        }
        // Same for the other box shorthands and border shorthands.
        let s = style(&doc, div, "div { padding: 5px bogus; }", &vp());
        assert_eq!(
            s.get("padding-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        let s = style(&doc, div, "div { border-style: solid bogus; }", &vp());
        assert_eq!(s.get("border-top-style"), Some(&kw("none")));
        let s = style(&doc, div, "div { border-width: 2px bogus; }", &vp());
        assert_eq!(s.get("border-top-width"), Some(&kw("medium")));
        // A lone CSS-wide keyword still expands: the cascade intercepts
        // it per longhand.
        let s = style(
            &doc,
            p,
            "div { margin: 10px; } p { margin: inherit; }",
            &vp(),
        );
        assert_eq!(
            s.get("margin-top"),
            Some(&CssValue::Length(10.0, LengthUnit::Px))
        );
        let s = style(
            &doc,
            p,
            "div { overflow: hidden; } p { overflow: inherit; }",
            &vp(),
        );
        assert_eq!(s.get("overflow-x"), Some(&kw("hidden")));
        assert_eq!(s.get("overflow-y"), Some(&kw("hidden")));
        // The deferred `var()` path validates after substitution too.
        let s = style(
            &doc,
            div,
            "div { --o: contain bogus; overscroll-behavior: var(--o); }",
            &vp(),
        );
        assert_eq!(s.get("overscroll-behavior-x"), Some(&kw("auto")));
        assert_eq!(s.get("overscroll-behavior-y"), Some(&kw("auto")));
        let s = style(
            &doc,
            div,
            "div { --m: 10px bogus; margin: var(--m); }",
            &vp(),
        );
        assert_eq!(
            s.get("margin-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
    }

    #[test]
    fn overflow_cross_axis_computed_adjustment() {
        // CSS Overflow 3 §3: impossible `visible`/`clip` cross-axis
        // combinations adjust at computed-value time (WPT
        // `css-overflow/parsing/overflow-computed.html`). `hidden` and
        // the axis pairs with each other are untouched.
        let (doc, div, p) = sample();
        // WPT matrix, as (declared x, declared y) -> (computed x, computed y).
        let cases = [
            ("visible", "visible", "visible", "visible"),
            ("hidden", "hidden", "hidden", "hidden"),
            ("scroll", "scroll", "scroll", "scroll"),
            ("auto", "auto", "auto", "auto"),
            ("clip", "clip", "clip", "clip"),
            ("visible", "hidden", "auto", "hidden"),
            ("hidden", "visible", "hidden", "auto"),
            ("visible", "scroll", "auto", "scroll"),
            ("scroll", "visible", "scroll", "auto"),
            ("visible", "auto", "auto", "auto"),
            ("auto", "visible", "auto", "auto"),
            ("clip", "scroll", "clip", "scroll"),
            ("scroll", "clip", "scroll", "clip"),
            ("clip", "auto", "clip", "auto"),
            ("auto", "clip", "auto", "clip"),
            ("visible", "clip", "visible", "clip"),
            ("clip", "visible", "clip", "visible"),
            ("hidden", "clip", "hidden", "clip"),
            ("clip", "hidden", "clip", "hidden"),
        ];
        for (x, y, ex, ey) in cases {
            let css = format!("div {{ overflow: {x} {y}; }}");
            let s = style(&doc, div, &css, &vp());
            assert_eq!(s.get("overflow-x"), Some(&kw(ex)), "{css}");
            assert_eq!(s.get("overflow-y"), Some(&kw(ey)), "{css}");
        }
        // A lone longhand still adjusts against the other axis's value.
        let s = style(&doc, div, "div { overflow-x: scroll; }", &vp());
        assert_eq!(s.get("overflow-x"), Some(&kw("scroll")));
        assert_eq!(s.get("overflow-y"), Some(&kw("auto")));
        // Explicit `inherit` reads the parent's *adjusted* value.
        let s = style(
            &doc,
            p,
            "div { overflow-x: scroll; } p { overflow-y: inherit; }",
            &vp(),
        );
        assert_eq!(s.get("overflow-y"), Some(&kw("auto")));
    }

    #[test]
    fn overflow_cold_single_axis_resolution_is_adjusted() {
        // B10: resolving one axis cold (no full `compute` first) must
        // memoize and return the fixup-adjusted pair, in either query
        // order. Before the fix, `finish_prop` memoized the raw
        // `visible` and later queries read that stale value back.
        let (doc, div, p) = sample();
        let arc_kw = |s: &str| Some(Arc::new(kw(s)));
        // Query x first, then y.
        let author = parse_stylesheet("div { overflow-x: visible; overflow-y: hidden; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        assert_eq!(
            styler.resolved_prop(div, props::OVERFLOW_X, "overflow-x"),
            arc_kw("auto")
        );
        assert_eq!(
            styler.resolved_prop(div, props::OVERFLOW_Y, "overflow-y"),
            arc_kw("hidden")
        );
        assert_eq!(
            styler.memo_get(div, props::OVERFLOW_X),
            Some(arc_kw("auto"))
        );
        assert_eq!(
            styler.memo_get(div, props::OVERFLOW_Y),
            Some(arc_kw("hidden"))
        );
        // Query y first, then x, on a fresh styler: the unspecified
        // axis (`visible` by initial) adjusts against `scroll`.
        let author = parse_stylesheet("div { overflow-x: scroll; }");
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        assert_eq!(
            styler.resolved_prop(div, props::OVERFLOW_Y, "overflow-y"),
            arc_kw("auto")
        );
        assert_eq!(
            styler.resolved_prop(div, props::OVERFLOW_X, "overflow-x"),
            arc_kw("scroll")
        );
        // `visible` against `clip` is not adjusted (the exact rule
        // `fixup_overflow_axes` implements), even cold.
        let author = parse_stylesheet("div { overflow-x: visible; overflow-y: clip; }");
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        assert_eq!(
            styler.resolved_prop(div, props::OVERFLOW_X, "overflow-x"),
            arc_kw("visible")
        );
        assert_eq!(
            styler.resolved_prop(div, props::OVERFLOW_Y, "overflow-y"),
            arc_kw("clip")
        );
        // A cold full style of the child (parent never styled first)
        // inherits the parent's adjusted axis, not its raw `visible`.
        let author = parse_stylesheet("div { overflow-x: scroll; } p { overflow-y: inherit; }");
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let s = styler.style_for(p);
        assert_eq!(s.get("overflow-y"), Some(&kw("auto")));
    }

    #[test]
    fn overflow_logical_longhands_fold_to_physical_axes() {
        // CSS Overflow 3: `overflow-block`/`overflow-inline` are
        // flow-relative. They keep their own computed values and fold
        // onto `overflow-x`/`overflow-y` under the element's writing
        // mode, with the later-in-cascade declaration winning.
        let (doc, div, p) = sample();
        let s = style(&doc, div, "div { overflow-block: hidden; }", &vp());
        assert_eq!(s.get("overflow-block"), Some(&kw("hidden")));
        assert_eq!(s.get("overflow-y"), Some(&kw("hidden")));
        // `overflow-x: visible` computes to `auto` against the hidden
        // block axis (CSS Overflow 3 §3 cross-axis adjustment).
        assert_eq!(s.get("overflow-x"), Some(&kw("auto")));
        let s = style(&doc, div, "div { overflow-inline: scroll; }", &vp());
        assert_eq!(s.get("overflow-inline"), Some(&kw("scroll")));
        assert_eq!(s.get("overflow-x"), Some(&kw("scroll")));
        // In vertical writing modes block is the horizontal axis.
        let s = style(
            &doc,
            div,
            "div { writing-mode: vertical-rl; overflow-block: hidden; overflow-inline: clip; }",
            &vp(),
        );
        assert_eq!(s.get("overflow-x"), Some(&kw("hidden")));
        assert_eq!(s.get("overflow-y"), Some(&kw("clip")));
        // Later declaration wins across the logical/physical boundary.
        let s = style(
            &doc,
            div,
            "div { overflow-y: auto; overflow-block: hidden; }",
            &vp(),
        );
        assert_eq!(s.get("overflow-y"), Some(&kw("hidden")));
        let s = style(
            &doc,
            div,
            "div { overflow-block: hidden; overflow-y: auto; }",
            &vp(),
        );
        assert_eq!(s.get("overflow-y"), Some(&kw("auto")));
        // The fold feeds the cross-axis adjustment: `overflow-inline:
        // visible` with `overflow-y: scroll` adjusts the physical x.
        let s = style(
            &doc,
            div,
            "div { overflow-inline: visible; overflow-y: scroll; }",
            &vp(),
        );
        assert_eq!(s.get("overflow-inline"), Some(&kw("visible")));
        assert_eq!(s.get("overflow-x"), Some(&kw("auto")));
        assert_eq!(s.get("overflow-y"), Some(&kw("scroll")));
        // Non-inherited, initial `visible`, invalid values rejected.
        let s = style(&doc, p, "div { overflow-block: hidden; }", &vp());
        assert_eq!(s.get("overflow-block"), Some(&kw("visible")));
        assert_eq!(s.get("overflow-x"), Some(&kw("visible")));
        let s = style(&doc, div, "div { overflow-inline: bogus; }", &vp());
        assert_eq!(s.get("overflow-inline"), Some(&kw("visible")));
        assert_eq!(s.get("overflow-x"), Some(&kw("visible")));
    }

    #[test]
    fn scroll_margin_keeps_negative_math_results() {
        // Negative lengths are valid `scroll-margin` (CSS Scroll Snap
        // 1): literals and `calc()` results alike stay negative in the
        // computed value (WPT `scroll-margin-computed.html` expects
        // `calc(10px - 0.5em)` to compute to `-10px`).
        let (doc, div, _) = sample();
        let neg = CssValue::Length(-10.0, LengthUnit::Px);
        let s = style(&doc, div, "div { scroll-margin-top: -10px; }", &vp());
        assert_eq!(s.get("scroll-margin-top"), Some(&neg));
        let s = style(&doc, div, "div { scroll-margin-top: calc(-10px); }", &vp());
        assert_eq!(s.get("scroll-margin-top"), Some(&neg));
        // Via the shorthand too.
        let s = style(&doc, div, "div { scroll-margin: -10px; }", &vp());
        for side in ["top", "right", "bottom", "left"] {
            assert_eq!(s.get(&format!("scroll-margin-{side}")), Some(&neg));
        }
    }

    #[test]
    fn scroll_padding_negative_math_clamps_to_zero() {
        // A *literal* negative is invalid at computed-value time
        // (falls back to the `auto` initial), but a math expression that
        // evaluates negative clamps to zero (CSS Values 4 range
        // checking; WPT `scroll-padding-computed.html`).
        let (doc, div, _) = sample();
        let zero = CssValue::Length(0.0, LengthUnit::Px);
        let s = style(&doc, div, "div { scroll-padding-top: -10px; }", &vp());
        assert_eq!(s.get("scroll-padding-top"), Some(&kw("auto")));
        let s = style(
            &doc,
            div,
            "div { scroll-padding-top: calc(5px - 20px); }",
            &vp(),
        );
        assert_eq!(s.get("scroll-padding-top"), Some(&zero));
        // Through the shorthand and `var()` substitution too.
        let s = style(&doc, div, "div { scroll-padding: calc(-1px); }", &vp());
        assert_eq!(s.get("scroll-padding-top"), Some(&zero));
        let s = style(
            &doc,
            div,
            "div { --p: calc(5px - 20px); scroll-padding-top: var(--p); }",
            &vp(),
        );
        assert_eq!(s.get("scroll-padding-top"), Some(&zero));
        // A percentage-bearing result keeps an unknown sign until
        // layout and stays deferred.
        let s = style(
            &doc,
            div,
            "div { scroll-padding-top: calc(50% + 60px); }",
            &vp(),
        );
        assert_eq!(
            s.get("scroll-padding-top"),
            Some(&CssValue::MathDeferred {
                pct: 50.0,
                px: 60.0,
                had_pct: true,
                affine: true
            })
        );
    }

    #[test]
    fn scroll_snap_type_canonical_computed_form() {
        // `proximity` is the default strictness: a pair naming it folds
        // to the bare axis keyword in the computed value, the canonical
        // form CSSOM serializes (WPT
        // `scroll-snap-type-computed.html`). `mandatory` pairs keep
        // their pair shape.
        let (doc, div, _) = sample();
        for (axis, strict, expected) in [
            ("x", "mandatory", "x mandatory"),
            ("y", "mandatory", "y mandatory"),
            ("x", "proximity", "x"),
            ("inline", "proximity", "inline"),
            ("both", "mandatory", "both mandatory"),
        ] {
            let css = format!("div {{ scroll-snap-type: {axis} {strict}; }}");
            let s = style(&doc, div, &css, &vp());
            let got = match s.get("scroll-snap-type") {
                Some(CssValue::List(items)) => items
                    .iter()
                    .map(|v| match v {
                        CssValue::Keyword(k) => k.as_str().to_owned(),
                        other => format!("{other:?}"),
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
                Some(CssValue::Keyword(k)) => k.as_str().to_owned(),
                other => format!("{other:?}"),
            };
            assert_eq!(got, expected, "{css}");
        }
        // Via `var()` substitution as well.
        let s = style(
            &doc,
            div,
            "div { --t: both proximity; scroll-snap-type: var(--t); }",
            &vp(),
        );
        assert_eq!(s.get("scroll-snap-type"), Some(&kw("both")));
        let s = style(
            &doc,
            div,
            "div { --t: x mandatory; scroll-snap-type: var(--t); }",
            &vp(),
        );
        assert_eq!(
            s.get("scroll-snap-type"),
            Some(&CssValue::List(vec![kw("x"), kw("mandatory")]))
        );
        // The `SnapType` helper still reads the canonical form.
        let s = style(&doc, div, "div { scroll-snap-type: y mandatory; }", &vp());
        assert_eq!(s.scroll_snap_type(), SnapType::MandatoryY);
    }

    #[test]
    fn overflow_shorthand_expands_through_var() {
        let (doc, div, _) = sample();
        // The deferred `var()` shorthand substitutes then expands to
        // the longhands.
        let s = style(&doc, div, "div { --o: hidden; overflow: var(--o); }", &vp());
        assert_eq!(s.get("overflow-x"), Some(&kw("hidden")));
        assert_eq!(s.get("overflow-y"), Some(&kw("hidden")));
        let s = style(
            &doc,
            div,
            "div { --o: scroll auto; overflow: var(--o); }",
            &vp(),
        );
        assert_eq!(s.get("overflow-x"), Some(&kw("scroll")));
        assert_eq!(s.get("overflow-y"), Some(&kw("auto")));
        // Unresolvable `var()` drops the declaration: initial values.
        let s = style(&doc, div, "div { overflow: var(--missing); }", &vp());
        assert_eq!(s.get("overflow-x"), Some(&kw("visible")));
        assert_eq!(s.get("overflow-y"), Some(&kw("visible")));
    }

    #[test]
    fn scroll_margin_padding_shorthand_expands_to_longhands() {
        let (doc, div, _) = sample();
        // One value sets all four sides.
        let s = style(&doc, div, "div { scroll-margin: 8px; }", &vp());
        for side in ["top", "right", "bottom", "left"] {
            assert_eq!(
                s.get(&format!("scroll-margin-{side}")),
                Some(&CssValue::Length(8.0, LengthUnit::Px))
            );
        }
        // Two values: vertical, horizontal.
        let s = style(&doc, div, "div { scroll-padding: 8px 4px; }", &vp());
        assert_eq!(
            s.get("scroll-padding-top"),
            Some(&CssValue::Length(8.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-padding-right"),
            Some(&CssValue::Length(4.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-padding-bottom"),
            Some(&CssValue::Length(8.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-padding-left"),
            Some(&CssValue::Length(4.0, LengthUnit::Px))
        );
        // Initial values with nothing specified: `0px` for scroll-margin,
        // `auto` for scroll-padding.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(
            s.get("scroll-margin-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        assert_eq!(s.get("scroll-padding-top"), Some(&kw("auto")));
        // `auto` is valid on scroll-padding only; it is invalid on
        // scroll-margin (CSS Scroll Snap 1: `scroll-margin` takes
        // `<length>`, no `auto`), so it falls back to `0px`.
        let s = style(
            &doc,
            div,
            "div { scroll-margin: auto; scroll-padding: auto; }",
            &vp(),
        );
        assert_eq!(
            s.get("scroll-margin-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        assert_eq!(s.get("scroll-padding-top"), Some(&kw("auto")));
        // Percentages are invalid scroll margins (fall back to `0px`)
        // but valid scroll padding.
        let s = style(
            &doc,
            div,
            "div { scroll-margin: 10%; scroll-padding: 10%; }",
            &vp(),
        );
        assert_eq!(
            s.get("scroll-margin-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-padding-top"),
            Some(&CssValue::Percentage(10.0))
        );
        // Negative scroll-margin is valid (CSS Scroll Snap 1 places no
        // negativity restriction on scroll-margin); negative
        // scroll-padding is invalid at computed-value time and falls
        // back to `auto`.
        let s = style(
            &doc,
            div,
            "div { scroll-margin: -5px; scroll-padding: -5px; }",
            &vp(),
        );
        assert_eq!(
            s.get("scroll-margin-top"),
            Some(&CssValue::Length(-5.0, LengthUnit::Px))
        );
        assert_eq!(s.get("scroll-padding-top"), Some(&kw("auto")));
        // A later longhand overrides the shorthand.
        let s = style(
            &doc,
            div,
            "div { scroll-margin: 8px; scroll-margin-top: 2px; }",
            &vp(),
        );
        assert_eq!(
            s.get("scroll-margin-top"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-margin-bottom"),
            Some(&CssValue::Length(8.0, LengthUnit::Px))
        );
    }

    #[test]
    fn scroll_margin_padding_expands_through_var() {
        let (doc, div, _) = sample();
        // The deferred `var()` shorthand substitutes then expands to
        // the per-side longhands.
        let s = style(
            &doc,
            div,
            "div { --m: 8px 4px; scroll-margin: var(--m); }",
            &vp(),
        );
        assert_eq!(
            s.get("scroll-margin-top"),
            Some(&CssValue::Length(8.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-margin-right"),
            Some(&CssValue::Length(4.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-margin-bottom"),
            Some(&CssValue::Length(8.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-margin-left"),
            Some(&CssValue::Length(4.0, LengthUnit::Px))
        );
        let s = style(
            &doc,
            div,
            "div { --p: 10%; scroll-padding: var(--p); }",
            &vp(),
        );
        assert_eq!(
            s.get("scroll-padding-top"),
            Some(&CssValue::Percentage(10.0))
        );
        // Unresolvable `var()` drops the declaration: initial values.
        let s = style(&doc, div, "div { scroll-margin: var(--missing); }", &vp());
        assert_eq!(
            s.get("scroll-margin-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
    }

    #[test]
    fn scroll_snap_longhands_parse_validate_and_default() {
        let (doc, div, _) = sample();
        // Axis + strictness pairs land in the longhand slot.
        let s = style(&doc, div, "div { scroll-snap-type: x mandatory; }", &vp());
        assert_eq!(
            s.get("scroll-snap-type"),
            Some(&CssValue::List(vec![kw("x"), kw("mandatory")]))
        );
        assert_eq!(s.scroll_snap_type(), SnapType::MandatoryX);
        // A lone axis means proximity.
        let s = style(&doc, div, "div { scroll-snap-type: y; }", &vp());
        assert_eq!(s.scroll_snap_type(), SnapType::ProximityY);
        let s = style(
            &doc,
            div,
            "div { scroll-snap-type: both proximity; }",
            &vp(),
        );
        assert_eq!(s.scroll_snap_type(), SnapType::ProximityBoth);
        // Logical axes map to the physical axis the snap engine works in.
        let s = style(
            &doc,
            div,
            "div { scroll-snap-type: block mandatory; }",
            &vp(),
        );
        assert_eq!(s.scroll_snap_type(), SnapType::MandatoryY);
        let s = style(&doc, div, "div { scroll-snap-type: inline; }", &vp());
        assert_eq!(s.scroll_snap_type(), SnapType::ProximityX);
        // Defaults with nothing specified: `none` / `none` / `normal`.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(s.get("scroll-snap-type"), Some(&kw("none")));
        assert_eq!(s.get("scroll-snap-align"), Some(&kw("none")));
        assert_eq!(s.get("scroll-snap-stop"), Some(&kw("normal")));
        assert_eq!(s.scroll_snap_type(), SnapType::None);
        assert_eq!(s.scroll_snap_align(), (SnapAlign::None, SnapAlign::None));
        assert_eq!(s.scroll_snap_stop(), SnapStop::Normal);
        // Align takes one or two keywords; a single value repeats.
        let s = style(&doc, div, "div { scroll-snap-align: start end; }", &vp());
        assert_eq!(s.scroll_snap_align(), (SnapAlign::Start, SnapAlign::End));
        let s = style(&doc, div, "div { scroll-snap-align: center; }", &vp());
        assert_eq!(
            s.scroll_snap_align(),
            (SnapAlign::Center, SnapAlign::Center)
        );
        // Stop is a single keyword.
        let s = style(&doc, div, "div { scroll-snap-stop: always; }", &vp());
        assert_eq!(s.scroll_snap_stop(), SnapStop::Always);
        // Invalid values drop the declaration: the initials win.
        let s = style(&doc, div, "div { scroll-snap-type: diagonal; }", &vp());
        assert_eq!(s.scroll_snap_type(), SnapType::None);
        let s = style(&doc, div, "div { scroll-snap-type: mandatory; }", &vp());
        assert_eq!(s.scroll_snap_type(), SnapType::None);
        let s = style(&doc, div, "div { scroll-snap-type: x maybe; }", &vp());
        assert_eq!(s.scroll_snap_type(), SnapType::None);
        let s = style(&doc, div, "div { scroll-snap-align: start bogus; }", &vp());
        assert_eq!(s.scroll_snap_align(), (SnapAlign::None, SnapAlign::None));
        let s = style(&doc, div, "div { scroll-snap-stop: sometimes; }", &vp());
        assert_eq!(s.scroll_snap_stop(), SnapStop::Normal);
        // Unresolvable `var()` drops the declaration: initial values.
        let s = style(
            &doc,
            div,
            "div { scroll-snap-type: var(--missing); scroll-snap-stop: var(--missing); }",
            &vp(),
        );
        assert_eq!(s.scroll_snap_type(), SnapType::None);
        assert_eq!(s.scroll_snap_stop(), SnapStop::Normal);
        // Not inherited: the child's values are the initials, not the
        // parent's, while the element that specified them keeps them.
        let (doc, div, p) = sample();
        let s = style(&doc, p, "div { scroll-snap-type: x mandatory; }", &vp());
        assert_eq!(s.scroll_snap_type(), SnapType::None);
        let s = style(&doc, div, "div { scroll-snap-type: x mandatory; }", &vp());
        assert_eq!(s.scroll_snap_type(), SnapType::MandatoryX);
    }

    #[test]
    fn scroll_snap_align_none_mixed_pairs() {
        let (doc, div, _) = sample();
        // Grammar is `[ none | start | end | center ]{1,2}`: a `none`
        // in either position is valid and disables snapping on that
        // axis only — the other axis keeps its alignment (the pair is
        // `(block, inline)`).
        let s = style(&doc, div, "div { scroll-snap-align: none start; }", &vp());
        assert_eq!(
            s.get("scroll-snap-align"),
            Some(&CssValue::List(vec![kw("none"), kw("start")]))
        );
        assert_eq!(s.scroll_snap_align(), (SnapAlign::None, SnapAlign::Start));
        let s = style(&doc, div, "div { scroll-snap-align: start none; }", &vp());
        assert_eq!(s.scroll_snap_align(), (SnapAlign::Start, SnapAlign::None));
        let s = style(&doc, div, "div { scroll-snap-align: none end; }", &vp());
        assert_eq!(s.scroll_snap_align(), (SnapAlign::None, SnapAlign::End));
        let s = style(&doc, div, "div { scroll-snap-align: center none; }", &vp());
        assert_eq!(s.scroll_snap_align(), (SnapAlign::Center, SnapAlign::None));
        // `none none` is grammatical too (both axes off, as `none`).
        let s = style(&doc, div, "div { scroll-snap-align: none none; }", &vp());
        assert_eq!(s.scroll_snap_align(), (SnapAlign::None, SnapAlign::None));
        // Three values still exceed the `{1,2}` grammar and drop.
        let s = style(
            &doc,
            div,
            "div { scroll-snap-align: none start end; }",
            &vp(),
        );
        assert_eq!(s.get("scroll-snap-align"), Some(&kw("none")));
        assert_eq!(s.scroll_snap_align(), (SnapAlign::None, SnapAlign::None));
    }

    #[test]
    fn scroll_snap_type_logical_axes_follow_writing_mode() {
        let (doc, div, _) = sample();
        // vertical-lr: the inline axis runs vertically (physical y) and
        // the block axis horizontally (physical x) — the logical
        // keywords swap relative to horizontal-tb.
        let s = style(
            &doc,
            div,
            "div { writing-mode: vertical-lr; scroll-snap-type: inline mandatory; }",
            &vp(),
        );
        assert_eq!(s.scroll_snap_type(), SnapType::MandatoryY);
        let s = style(
            &doc,
            div,
            "div { writing-mode: vertical-lr; scroll-snap-type: block mandatory; }",
            &vp(),
        );
        assert_eq!(s.scroll_snap_type(), SnapType::MandatoryX);
        // vertical-rl behaves the same for axis identity; a lone axis
        // keyword still means proximity.
        let s = style(
            &doc,
            div,
            "div { writing-mode: vertical-rl; scroll-snap-type: inline; }",
            &vp(),
        );
        assert_eq!(s.scroll_snap_type(), SnapType::ProximityY);
        let s = style(
            &doc,
            div,
            "div { writing-mode: vertical-rl; scroll-snap-type: block; }",
            &vp(),
        );
        assert_eq!(s.scroll_snap_type(), SnapType::ProximityX);
        // Physical axis keywords never swap, whatever the writing mode.
        let s = style(
            &doc,
            div,
            "div { writing-mode: vertical-lr; scroll-snap-type: x mandatory; }",
            &vp(),
        );
        assert_eq!(s.scroll_snap_type(), SnapType::MandatoryX);
        let s = style(
            &doc,
            div,
            "div { writing-mode: vertical-lr; scroll-snap-type: y mandatory; }",
            &vp(),
        );
        assert_eq!(s.scroll_snap_type(), SnapType::MandatoryY);
        let s = style(
            &doc,
            div,
            "div { writing-mode: vertical-lr; scroll-snap-type: both mandatory; }",
            &vp(),
        );
        assert_eq!(s.scroll_snap_type(), SnapType::MandatoryBoth);
        // The computed value itself stays in the specified (logical)
        // form — only the physical mapping in `scroll_snap_type` swaps.
        let s = style(
            &doc,
            div,
            "div { writing-mode: vertical-lr; scroll-snap-type: inline mandatory; }",
            &vp(),
        );
        assert_eq!(
            s.get("scroll-snap-type"),
            Some(&CssValue::List(vec![kw("inline"), kw("mandatory")]))
        );
        // horizontal-tb keeps the original mapping.
        let s = style(
            &doc,
            div,
            "div { writing-mode: horizontal-tb; scroll-snap-type: inline mandatory; }",
            &vp(),
        );
        assert_eq!(s.scroll_snap_type(), SnapType::MandatoryX);
    }

    fn px(v: f32) -> CssValue {
        CssValue::Length(v, LengthUnit::Px)
    }

    #[test]
    fn logical_margin_resolves_against_direction() {
        let (doc, div, _) = sample();
        // LTR: inline-start -> left, block-end -> bottom.
        let s = style(
            &doc,
            div,
            "div { margin-inline-start: 10px; margin-block-end: 5px; }",
            &vp(),
        );
        assert_eq!(s.get("margin-left"), Some(&px(10.0)));
        assert_eq!(s.get("margin-right"), Some(&px(0.0)));
        assert_eq!(s.get("margin-bottom"), Some(&px(5.0)));
        assert_eq!(s.get("margin-top"), Some(&px(0.0)));
        // The logical slot keeps its own flow-relative computed value.
        assert_eq!(s.get("margin-inline-start"), Some(&px(10.0)));
        // RTL: inline-start -> right, inline-end -> left.
        let s = style(
            &doc,
            div,
            "div { direction: rtl; margin-inline-start: 10px; margin-inline-end: 20px; }",
            &vp(),
        );
        assert_eq!(s.get("margin-right"), Some(&px(10.0)));
        assert_eq!(s.get("margin-left"), Some(&px(20.0)));
    }

    #[test]
    fn logical_padding_and_inset_resolve() {
        let (doc, div, _) = sample();
        // RTL: inset-inline-start -> right, padding-inline-end -> left;
        // block-start -> top regardless of direction.
        let s = style(
            &doc,
            div,
            "div { direction: rtl; padding-inline-end: 8px; inset-inline-start: 3px; inset-block-start: 4px; }",
            &vp(),
        );
        assert_eq!(s.get("padding-left"), Some(&px(8.0)));
        assert_eq!(s.get("padding-right"), Some(&px(0.0)));
        assert_eq!(s.get("right"), Some(&px(3.0)));
        assert_eq!(s.get("left"), Some(&kw("auto")));
        assert_eq!(s.get("top"), Some(&px(4.0)));
        // Negative logical padding is invalid at computed-value time.
        let s = style(&doc, div, "div { padding-inline-start: -5px; }", &vp());
        assert_eq!(s.get("padding-left"), Some(&px(0.0)));
        assert_eq!(s.get("padding-inline-start"), Some(&px(0.0)));
    }

    #[test]
    fn logical_size_maps_to_physical_axis() {
        let (doc, div, _) = sample();
        // horizontal-tb: inline-size -> width, block-size -> height.
        let s = style(
            &doc,
            div,
            "div { inline-size: 100px; block-size: 50px; }",
            &vp(),
        );
        assert_eq!(s.get("width"), Some(&px(100.0)));
        assert_eq!(s.get("height"), Some(&px(50.0)));
        // vertical-rl: inline-size -> height, block-size -> width.
        let s = style(
            &doc,
            div,
            "div { writing-mode: vertical-rl; inline-size: 100px; block-size: 50px; }",
            &vp(),
        );
        assert_eq!(s.get("height"), Some(&px(100.0)));
        assert_eq!(s.get("width"), Some(&px(50.0)));
        // `auto` is valid on both axes.
        let s = style(&doc, div, "div { inline-size: auto; }", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
    }

    #[test]
    fn logical_sides_in_vertical_writing_modes() {
        let (doc, div, _) = sample();
        // vertical-rl: inline-start -> top, block-start -> right,
        // block-end -> left.
        let s = style(
            &doc,
            div,
            "div { writing-mode: vertical-rl; margin-inline-start: 10px; margin-block-start: 20px; margin-block-end: 30px; }",
            &vp(),
        );
        assert_eq!(s.get("margin-top"), Some(&px(10.0)));
        assert_eq!(s.get("margin-right"), Some(&px(20.0)));
        assert_eq!(s.get("margin-left"), Some(&px(30.0)));
        // vertical-lr: block-start -> left.
        let s = style(
            &doc,
            div,
            "div { writing-mode: vertical-lr; margin-block-start: 20px; }",
            &vp(),
        );
        assert_eq!(s.get("margin-left"), Some(&px(20.0)));
        // `direction: rtl` flips the inline axis bottom-to-top in
        // vertical writing modes (CSS Writing Modes 3 §6.4).
        let s = style(
            &doc,
            div,
            "div { writing-mode: vertical-rl; direction: rtl; margin-inline-start: 10px; }",
            &vp(),
        );
        assert_eq!(s.get("margin-bottom"), Some(&px(10.0)));
        assert_eq!(s.get("margin-top"), Some(&px(0.0)));
    }

    #[test]
    fn logical_vs_physical_cascade_order() {
        let (doc, div, _) = sample();
        // Later in the cascade wins between a logical longhand and its
        // physical counterpart, like any two competing declarations.
        let s = style(
            &doc,
            div,
            "div { margin-inline-start: 20px; margin-left: 10px; }",
            &vp(),
        );
        assert_eq!(s.get("margin-left"), Some(&px(10.0)));
        let s = style(
            &doc,
            div,
            "div { margin-left: 10px; margin-inline-start: 20px; }",
            &vp(),
        );
        assert_eq!(s.get("margin-left"), Some(&px(20.0)));
        // `!important` outranks source order across the pair.
        let s = style(
            &doc,
            div,
            "div { margin-left: 10px !important; margin-inline-start: 20px; }",
            &vp(),
        );
        assert_eq!(s.get("margin-left"), Some(&px(10.0)));
        let s = style(
            &doc,
            div,
            "div { margin-left: 10px; margin-inline-start: 20px !important; }",
            &vp(),
        );
        assert_eq!(s.get("margin-left"), Some(&px(20.0)));
        // Specificity outranks source order across the pair too.
        let s = style(
            &doc,
            div,
            "#d { margin-left: 10px; } .a { margin-inline-start: 20px; }",
            &vp(),
        );
        assert_eq!(s.get("margin-left"), Some(&px(10.0)));
        // The other physical sides are unaffected by the fold.
        let s = style(
            &doc,
            div,
            "div { margin-inline-start: 20px; margin-right: 7px; }",
            &vp(),
        );
        assert_eq!(s.get("margin-left"), Some(&px(20.0)));
        assert_eq!(s.get("margin-right"), Some(&px(7.0)));
    }

    #[test]
    fn logical_shorthands_expand() {
        let (doc, div, _) = sample();
        // `margin-inline: auto` — the centering idiom — expands to the
        // flow-relative longhands and folds to the physical sides.
        let s = style(&doc, div, "div { margin-inline: auto; }", &vp());
        assert_eq!(s.get("margin-left"), Some(&kw("auto")));
        assert_eq!(s.get("margin-right"), Some(&kw("auto")));
        // Two values: start then end.
        let s = style(&doc, div, "div { margin-inline: 1px 2px; }", &vp());
        assert_eq!(s.get("margin-left"), Some(&px(1.0)));
        assert_eq!(s.get("margin-right"), Some(&px(2.0)));
        // Block axis.
        let s = style(&doc, div, "div { margin-block: 3px 4px; }", &vp());
        assert_eq!(s.get("margin-top"), Some(&px(3.0)));
        assert_eq!(s.get("margin-bottom"), Some(&px(4.0)));
        // Padding and inset shorthands.
        let s = style(&doc, div, "div { padding-inline: 5px; }", &vp());
        assert_eq!(s.get("padding-left"), Some(&px(5.0)));
        assert_eq!(s.get("padding-right"), Some(&px(5.0)));
        let s = style(&doc, div, "div { inset-inline: 6px 7px; }", &vp());
        assert_eq!(s.get("left"), Some(&px(6.0)));
        assert_eq!(s.get("right"), Some(&px(7.0)));
        let s = style(&doc, div, "div { inset-block: 8px 9px; }", &vp());
        assert_eq!(s.get("top"), Some(&px(8.0)));
        assert_eq!(s.get("bottom"), Some(&px(9.0)));
        // Border logical shorthands.
        let s = style(&doc, div, "div { border-inline-width: 2px; }", &vp());
        assert_eq!(s.get("border-left-width"), Some(&px(2.0)));
        assert_eq!(s.get("border-right-width"), Some(&px(2.0)));
        // Three values are invalid: the declaration drops, initials apply.
        let s = style(&doc, div, "div { margin-inline: 1px 2px 3px; }", &vp());
        assert_eq!(s.get("margin-left"), Some(&px(0.0)));
        assert_eq!(s.get("margin-right"), Some(&px(0.0)));
        // `var()`-deferred shorthands expand after substitution.
        let s = style(
            &doc,
            div,
            "div { --m: 8px; margin-inline: var(--m) 9px; }",
            &vp(),
        );
        assert_eq!(s.get("margin-left"), Some(&px(8.0)));
        assert_eq!(s.get("margin-right"), Some(&px(9.0)));
    }

    #[test]
    fn unspecified_logical_leaves_physical_alone() {
        let (doc, div, _) = sample();
        // No logical declarations: physical values (and initials) are
        // untouched by the fold.
        let s = style(&doc, div, "div { margin-left: 5px; }", &vp());
        assert_eq!(s.get("margin-left"), Some(&px(5.0)));
        assert_eq!(s.get("margin-right"), Some(&px(0.0)));
        assert_eq!(s.get("margin-inline-start"), Some(&px(0.0)));
        assert_eq!(s.get("width"), Some(&kw("auto")));
    }

    #[test]
    fn writing_mode_initial_inherits_and_validates() {
        let (doc, div, p) = sample();
        // Initial value.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(s.get("writing-mode"), Some(&kw("horizontal-tb")));
        // Inherited.
        let s = style(&doc, p, "#d { writing-mode: vertical-rl; }", &vp());
        assert_eq!(s.get("writing-mode"), Some(&kw("vertical-rl")));
        // `sideways-*` needs glyph-orientation support the text engine
        // lacks: rejected, falls back to the initial value.
        let s = style(&doc, div, "div { writing-mode: sideways-rl; }", &vp());
        assert_eq!(s.get("writing-mode"), Some(&kw("horizontal-tb")));
    }

    #[test]
    fn unicode_bidi_validates_computes_and_is_not_inherited() {
        let (doc, div, p) = sample();
        // Every keyword validates and computes verbatim; an unknown
        // keyword is rejected.
        for word in [
            "normal",
            "embed",
            "isolate",
            "bidi-override",
            "isolate-override",
            "plaintext",
        ] {
            assert!(
                is_valid_for_property("unicode-bidi", &kw(word)),
                "{word} must validate"
            );
            let s = style(
                &doc,
                div,
                &format!("div {{ unicode-bidi: {word}; }}"),
                &vp(),
            );
            assert_eq!(s.get("unicode-bidi"), Some(&kw(word)), "{word}");
            assert_eq!(s.get_str("unicode-bidi"), Some(word), "{word}");
        }
        assert!(!is_valid_for_property("unicode-bidi", &kw("bogus")));
        // Initial value is `normal`.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(s.get("unicode-bidi"), Some(&kw("normal")));
        // Invalid keyword rejected at computed-value time: falls back
        // to the initial `normal`.
        let s = style(&doc, div, "div { unicode-bidi: bogus; }", &vp());
        assert_eq!(s.get("unicode-bidi"), Some(&kw("normal")));
        // Computed round-trip for `plaintext` through the name
        // lookup, the const-index slot and the string view.
        let s = style(&doc, div, "div { unicode-bidi: plaintext; }", &vp());
        assert_eq!(s.get("unicode-bidi"), Some(&kw("plaintext")));
        assert_eq!(s.get_by_index(props::UNICODE_BIDI), Some(&kw("plaintext")));
        assert_eq!(s.get("unicode-bidi"), s.get_by_index(props::UNICODE_BIDI));
        assert_eq!(s.get_str("unicode-bidi"), Some("plaintext"));
        assert_eq!(s.get_str_by_index(props::UNICODE_BIDI), Some("plaintext"));
        // Not inherited (CSS Writing Modes 3): a child of a
        // `plaintext` parent computes `normal` unless it sets the
        // property itself.
        let s = style(&doc, p, "div { unicode-bidi: plaintext; }", &vp());
        assert_eq!(s.get("unicode-bidi"), Some(&kw("normal")));
        let s = style(
            &doc,
            p,
            "div { unicode-bidi: plaintext; } p { unicode-bidi: isolate; }",
            &vp(),
        );
        assert_eq!(s.get("unicode-bidi"), Some(&kw("isolate")));
    }

    #[test]
    fn logical_border_resolves() {
        let (doc, div, _) = sample();
        let s = style(
            &doc,
            div,
            "div { border-inline-start-width: 3px; border-inline-start-style: solid; border-block-end-color: red; }",
            &vp(),
        );
        assert_eq!(s.get("border-left-width"), Some(&px(3.0)));
        assert_eq!(s.get("border-left-style"), Some(&kw("solid")));
        assert_eq!(
            s.get("border-bottom-color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // RTL flips the inline side; the untouched sides keep initials.
        let s = style(
            &doc,
            div,
            "div { direction: rtl; border-inline-start-width: 3px; }",
            &vp(),
        );
        assert_eq!(s.get("border-right-width"), Some(&px(3.0)));
        assert_eq!(s.get("border-left-width"), Some(&kw("medium")));
    }

    #[test]
    fn logical_values_substitute_var_and_math() {
        let (doc, div, _) = sample();
        // `var()` substitution and `calc()` evaluation happen before
        // the logical fold, so the physical slot gets the final value.
        let s = style(
            &doc,
            div,
            "div { direction: rtl; --m: 7px; margin-inline-start: var(--m); padding-block-start: calc(2px + 3px); }",
            &vp(),
        );
        assert_eq!(s.get("margin-right"), Some(&px(7.0)));
        assert_eq!(s.get("padding-top"), Some(&px(5.0)));
        // An invalid-at-computed-value-time logical declaration drops to
        // the initial value and does not clobber a specified physical.
        let s = style(
            &doc,
            div,
            "div { margin-inline-start: bogus; margin-left: 9px; }",
            &vp(),
        );
        assert_eq!(s.get("margin-inline-start"), Some(&px(0.0)));
        assert_eq!(s.get("margin-left"), Some(&px(9.0)));
    }

    #[test]
    fn overscroll_behavior_shorthand_expands_to_longhands() {
        let (doc, div, _) = sample();
        // One value sets both axes.
        let s = style(&doc, div, "div { overscroll-behavior: contain; }", &vp());
        assert_eq!(s.get("overscroll-behavior-x"), Some(&kw("contain")));
        assert_eq!(s.get("overscroll-behavior-y"), Some(&kw("contain")));
        // Two values: x then y.
        let s = style(&doc, div, "div { overscroll-behavior: auto none; }", &vp());
        assert_eq!(s.get("overscroll-behavior-x"), Some(&kw("auto")));
        assert_eq!(s.get("overscroll-behavior-y"), Some(&kw("none")));
        // Initial value is `auto` on both axes.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(s.get("overscroll-behavior-x"), Some(&kw("auto")));
        assert_eq!(s.get("overscroll-behavior-y"), Some(&kw("auto")));
        // A component outside the grammar invalidates the whole shorthand:
        // the declaration drops and both axes keep their initial `auto`
        // (no partial application).
        let s = style(
            &doc,
            div,
            "div { overscroll-behavior: contain bogus; }",
            &vp(),
        );
        assert_eq!(s.get("overscroll-behavior-x"), Some(&kw("auto")));
        assert_eq!(s.get("overscroll-behavior-y"), Some(&kw("auto")));
        // A later longhand overrides the shorthand.
        let s = style(
            &doc,
            div,
            "div { overscroll-behavior: contain; overscroll-behavior-y: none; }",
            &vp(),
        );
        assert_eq!(s.get("overscroll-behavior-x"), Some(&kw("contain")));
        assert_eq!(s.get("overscroll-behavior-y"), Some(&kw("none")));
        // The deferred `var()` shorthand substitutes then expands.
        let s = style(
            &doc,
            div,
            "div { --o: contain none; overscroll-behavior: var(--o); }",
            &vp(),
        );
        assert_eq!(s.get("overscroll-behavior-x"), Some(&kw("contain")));
        assert_eq!(s.get("overscroll-behavior-y"), Some(&kw("none")));
        // Unresolvable `var()` drops the declaration: initial values.
        let s = style(
            &doc,
            div,
            "div { overscroll-behavior: var(--missing); }",
            &vp(),
        );
        assert_eq!(s.get("overscroll-behavior-x"), Some(&kw("auto")));
        assert_eq!(s.get("overscroll-behavior-y"), Some(&kw("auto")));
    }

    #[test]
    fn place_self_shorthand_expands_to_longhands() {
        let (doc, div, _) = sample();
        // One value sets both longhands.
        let s = style(&doc, div, "div { place-self: center; }", &vp());
        assert_eq!(s.get("align-self"), Some(&kw("center")));
        assert_eq!(s.get("justify-self"), Some(&kw("center")));
        // Two values: align-self then justify-self.
        let s = style(&doc, div, "div { place-self: start end; }", &vp());
        assert_eq!(s.get("align-self"), Some(&kw("start")));
        assert_eq!(s.get("justify-self"), Some(&kw("end")));
        // Unspecified: the initial `auto` on both.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(s.get("align-self"), Some(&kw("auto")));
        assert_eq!(s.get("justify-self"), Some(&kw("auto")));
        // A component outside the grammar invalidates the whole
        // shorthand: the declaration drops and both longhands keep
        // their initial `auto` (no partial application).
        let s = style(&doc, div, "div { place-self: center 10px; }", &vp());
        assert_eq!(s.get("align-self"), Some(&kw("auto")));
        assert_eq!(s.get("justify-self"), Some(&kw("auto")));
        // A later longhand overrides the shorthand.
        let s = style(
            &doc,
            div,
            "div { place-self: center; justify-self: start; }",
            &vp(),
        );
        assert_eq!(s.get("align-self"), Some(&kw("center")));
        assert_eq!(s.get("justify-self"), Some(&kw("start")));
        // The deferred `var()` shorthand substitutes then expands.
        let s = style(
            &doc,
            div,
            "div { --p: start end; place-self: var(--p); }",
            &vp(),
        );
        assert_eq!(s.get("align-self"), Some(&kw("start")));
        assert_eq!(s.get("justify-self"), Some(&kw("end")));
        // A substituted value outside the grammar drops the
        // declaration: initial values.
        let s = style(
            &doc,
            div,
            "div { --p: center bogus; place-self: var(--p); }",
            &vp(),
        );
        assert_eq!(s.get("align-self"), Some(&kw("auto")));
        assert_eq!(s.get("justify-self"), Some(&kw("auto")));
        // Unresolvable `var()` drops the declaration: initial values.
        let s = style(&doc, div, "div { place-self: var(--missing); }", &vp());
        assert_eq!(s.get("align-self"), Some(&kw("auto")));
        assert_eq!(s.get("justify-self"), Some(&kw("auto")));
    }

    #[test]
    fn scroll_behavior_parses_smooth_and_defaults_to_auto() {
        let (doc, div, _) = sample();
        let s = style(&doc, div, "div { scroll-behavior: smooth; }", &vp());
        assert_eq!(s.get("scroll-behavior"), Some(&kw("smooth")));
        // An invalid keyword drops the declaration: the initial `auto`
        // wins.
        let s = style(&doc, div, "div { scroll-behavior: fast; }", &vp());
        assert_eq!(s.get("scroll-behavior"), Some(&kw("auto")));
        // Unspecified: the initial value, `auto`.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(s.get("scroll-behavior"), Some(&kw("auto")));
    }

    #[test]
    fn scrollbar_gutter_rejects_duplicate_stable() {
        let (doc, div, _) = sample();
        // `stable stable` is not `stable && both-edges?`: the
        // declaration drops and the initial `auto` wins.
        let s = style(&doc, div, "div { scrollbar-gutter: stable stable; }", &vp());
        assert_eq!(s.get("scrollbar-gutter"), Some(&kw("auto")));
        // Exactly one `stable` plus one `both-edges`, either order.
        let s = style(
            &doc,
            div,
            "div { scrollbar-gutter: stable both-edges; }",
            &vp(),
        );
        assert_eq!(
            s.get("scrollbar-gutter"),
            Some(&CssValue::List(vec![kw("stable"), kw("both-edges")]))
        );
        let s = style(
            &doc,
            div,
            "div { scrollbar-gutter: both-edges stable; }",
            &vp(),
        );
        assert_eq!(
            s.get("scrollbar-gutter"),
            Some(&CssValue::List(vec![kw("both-edges"), kw("stable")]))
        );
        // A lone `both-edges` or a doubled `both-edges` has no `stable`:
        // invalid.
        let s = style(&doc, div, "div { scrollbar-gutter: both-edges; }", &vp());
        assert_eq!(s.get("scrollbar-gutter"), Some(&kw("auto")));
        let s = style(
            &doc,
            div,
            "div { scrollbar-gutter: both-edges both-edges; }",
            &vp(),
        );
        assert_eq!(s.get("scrollbar-gutter"), Some(&kw("auto")));
        // Single keywords still work.
        let s = style(&doc, div, "div { scrollbar-gutter: stable; }", &vp());
        assert_eq!(s.get("scrollbar-gutter"), Some(&kw("stable")));
    }

    #[test]
    fn touch_action_registers_with_auto_initial_and_grammar() {
        let (doc, div, _) = sample();
        // Unspecified: the initial value, `auto`.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(s.get("touch-action"), Some(&kw("auto")));
        // Single keywords pass through.
        for single in [
            "none",
            "pan-x",
            "pan-y",
            "pan-left",
            "pan-right",
            "pan-up",
            "pan-down",
            "manipulation",
        ] {
            let s = style(
                &doc,
                div,
                &format!("div {{ touch-action: {single}; }}"),
                &vp(),
            );
            assert_eq!(s.get("touch-action"), Some(&kw(single)), "{single}");
        }
        // Two-keyword values pick at most one per axis group, either order.
        for pair in [
            "pan-x pan-y",
            "pan-y pan-x",
            "pan-left pan-up",
            "pan-up pan-left",
            "pan-x pan-down",
            "pan-right pan-y",
        ] {
            let s = style(
                &doc,
                div,
                &format!("div {{ touch-action: {pair}; }}"),
                &vp(),
            );
            let CssValue::List(items) = s.get("touch-action").expect("list value") else {
                panic!("{pair} should compute to a keyword list");
            };
            assert_eq!(items.len(), 2, "{pair}");
        }
        // `auto`/`none`/`manipulation` never combine; same-axis pairs are
        // rejected; unknown keywords are rejected. The declaration drops
        // and the initial `auto` wins.
        for bad in [
            "auto pan-x",
            "none pan-y",
            "pan-x manipulation",
            "pan-x pan-left",
            "pan-y pan-up",
            "pan-x pan-y pan-up",
            "pan-x bogus",
            "5px",
        ] {
            let s = style(&doc, div, &format!("div {{ touch-action: {bad}; }}"), &vp());
            assert_eq!(s.get("touch-action"), Some(&kw("auto")), "{bad}");
        }
        // Keywords are ASCII case-insensitive at parse time.
        let s = style(&doc, div, "div { touch-action: PAN-X; }", &vp());
        assert_eq!(s.get("touch-action"), Some(&kw("pan-x")));
    }

    #[test]
    fn scroll_logical_longhands_compute_and_fold() {
        let (doc, div, _) = sample();
        // Logical longhands keep their own computed values and fold onto
        // the physical slots (horizontal-tb, ltr: inline-start -> left).
        let s = style(&doc, div, "div { scroll-margin-inline: 5px; }", &vp());
        assert_eq!(
            s.get("scroll-margin-inline-start"),
            Some(&CssValue::Length(5.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-margin-inline-end"),
            Some(&CssValue::Length(5.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-margin-left"),
            Some(&CssValue::Length(5.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-margin-right"),
            Some(&CssValue::Length(5.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-margin-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // Two-value block form: start/top, end/bottom.
        let s = style(&doc, div, "div { scroll-padding-block: 1px 2px; }", &vp());
        assert_eq!(
            s.get("scroll-padding-block-start"),
            Some(&CssValue::Length(1.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-padding-block-end"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-padding-top"),
            Some(&CssValue::Length(1.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-padding-bottom"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        // Direct longhand declarations are not dropped: they compute and
        // fold too.
        let s = style(&doc, div, "div { scroll-margin-block-start: 9px; }", &vp());
        assert_eq!(
            s.get("scroll-margin-block-start"),
            Some(&CssValue::Length(9.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-margin-top"),
            Some(&CssValue::Length(9.0, LengthUnit::Px))
        );
        // Unspecified logical longhands take the physical initials.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(
            s.get("scroll-margin-inline-start"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        assert_eq!(s.get("scroll-padding-inline-start"), Some(&kw("auto")));
    }

    #[test]
    fn scroll_logical_shorthands_expand_through_var() {
        let (doc, div, _) = sample();
        // A `var()`-deferred logical scroll shorthand substitutes then
        // expands to the logical longhands, which fold physically.
        let s = style(
            &doc,
            div,
            "div { --m: 5px 6px; scroll-margin-inline: var(--m); }",
            &vp(),
        );
        assert_eq!(
            s.get("scroll-margin-inline-start"),
            Some(&CssValue::Length(5.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-margin-inline-end"),
            Some(&CssValue::Length(6.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-margin-left"),
            Some(&CssValue::Length(5.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-margin-right"),
            Some(&CssValue::Length(6.0, LengthUnit::Px))
        );
        let s = style(
            &doc,
            div,
            "div { --p: 10%; scroll-padding-block: var(--p); }",
            &vp(),
        );
        assert_eq!(
            s.get("scroll-padding-block-start"),
            Some(&CssValue::Percentage(10.0))
        );
        assert_eq!(
            s.get("scroll-padding-top"),
            Some(&CssValue::Percentage(10.0))
        );
        assert_eq!(
            s.get("scroll-padding-bottom"),
            Some(&CssValue::Percentage(10.0))
        );
        // An invalid component fails the deferred expansion probe: the
        // whole declaration drops, so the initials win.
        let s = style(
            &doc,
            div,
            "div { --bad: auto; scroll-margin-inline: var(--bad); }",
            &vp(),
        );
        assert_eq!(
            s.get("scroll-margin-inline-start"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("scroll-margin-left"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // Unresolvable `var()` drops the declaration: initial values.
        let s = style(
            &doc,
            div,
            "div { scroll-padding-inline: var(--missing); }",
            &vp(),
        );
        assert_eq!(s.get("scroll-padding-inline-start"), Some(&kw("auto")));
        assert_eq!(s.get("scroll-padding-left"), Some(&kw("auto")));
    }

    #[test]
    fn scroll_initial_target_computes() {
        let (doc, div, _) = sample();
        // Initial value per CSS Scroll Snap 2 §3.1.2: `none`.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(s.get("scroll-initial-target"), Some(&kw("none")));
        let s = style(&doc, div, "div { scroll-initial-target: nearest; }", &vp());
        assert_eq!(s.get("scroll-initial-target"), Some(&kw("nearest")));
        // Outside `none | nearest`: the declaration drops.
        let s = style(&doc, div, "div { scroll-initial-target: center; }", &vp());
        assert_eq!(s.get("scroll-initial-target"), Some(&kw("none")));
    }

    #[test]
    fn overflow_clip_margin_computes_and_rejects_negatives() {
        let (doc, div, _) = sample();
        // Initial value per CSS Overflow 3 §3.2: `0px`.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(
            s.get("overflow-clip-margin"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        let s = style(&doc, div, "div { overflow-clip-margin: 4px; }", &vp());
        assert_eq!(
            s.get("overflow-clip-margin"),
            Some(&CssValue::Length(4.0, LengthUnit::Px))
        );
        // `<visual-box> || <length>`, either order.
        let s = style(
            &doc,
            div,
            "div { overflow-clip-margin: content-box 4px; }",
            &vp(),
        );
        assert_eq!(
            s.get("overflow-clip-margin"),
            Some(&CssValue::List(vec![
                kw("content-box"),
                CssValue::Length(4.0, LengthUnit::Px)
            ]))
        );
        // Negative lengths are outside `<length [0,∞]>`: dropped.
        let s = style(&doc, div, "div { overflow-clip-margin: -5px; }", &vp());
        assert_eq!(
            s.get("overflow-clip-margin"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        let s = style(
            &doc,
            div,
            "div { overflow-clip-margin: border-box -5px; }",
            &vp(),
        );
        assert_eq!(
            s.get("overflow-clip-margin"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // Unitless zero is a valid `<length>`.
        let s = style(&doc, div, "div { overflow-clip-margin: 0; }", &vp());
        assert_eq!(s.get("overflow-clip-margin"), Some(&CssValue::Number(0.0)));
    }

    #[test]
    fn overflow_clip_margin_rejects_negative_math() {
        // `<length [0,∞]>` (CSS Overflow 3 §3.2, CSS Values 4 range
        // checking): a percentage-bearing `calc()` that can resolve
        // negative is invalid — the sign is unknowable until layout, so
        // anything not provably non-negative is rejected. WPT
        // `css/css-overflow/parsing/overflow-clip-margin.html` pins all
        // three as invalid values (19/29; the post-verify w50
        // serialization fix e4810a9 exposed the missing parse-time
        // check).
        let (doc, div, _) = sample();
        let initial = CssValue::Length(0.0, LengthUnit::Px);
        for (css, decl) in [
            (
                "div { overflow-clip-margin: calc(100% - 0.5em); }",
                "overflow-clip-margin: calc(100% - 0.5em)",
            ),
            (
                "div { overflow-clip-margin: calc(100% - 50px); }",
                "overflow-clip-margin: calc(100% - 50px)",
            ),
            (
                "div { overflow-clip-margin: border-box calc(0.5em - 100%); }",
                "overflow-clip-margin: border-box calc(0.5em - 100%)",
            ),
        ] {
            // Parse time (the CSSOM `style` setter path): the declaration
            // is dropped, so `getPropertyValue` reads back empty.
            let decls = parse_declarations(decl);
            assert_eq!(decls.len(), 1, "{css}");
            assert!(
                !is_valid_for_property("overflow-clip-margin", &decls[0].value),
                "{css} must be invalid at parse time"
            );
            // Computed-value time: invalid at computed-value time, so the
            // property falls back to its initial value.
            let s = style(&doc, div, css, &vp());
            assert_eq!(s.get("overflow-clip-margin"), Some(&initial), "{css}");
        }
    }

    #[test]
    fn overflow_clip_margin_accepts_non_negative_math() {
        // A percentage-bearing `calc()` that is provably non-negative for
        // every non-negative percentage base stays valid: it evaluates to
        // the deferred percentage form resolved later by layout.
        let (doc, div, _) = sample();
        for decl in [
            "overflow-clip-margin: border-box calc(10px + 5%)",
            "overflow-clip-margin: calc(0.5em + 100%)",
        ] {
            let decls = parse_declarations(decl);
            assert_eq!(decls.len(), 1, "{decl}");
            assert!(
                is_valid_for_property("overflow-clip-margin", &decls[0].value),
                "{decl} must be valid at parse time"
            );
        }
        // `border-box calc(10px + 5%)`: pair form, deferred math kept.
        let s = style(
            &doc,
            div,
            "div { overflow-clip-margin: border-box calc(10px + 5%); }",
            &vp(),
        );
        match s.get("overflow-clip-margin") {
            Some(CssValue::List(items)) => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], kw("border-box"));
                assert!(
                    matches!(items[1], CssValue::MathDeferred { px, pct, .. } if px >= 0.0 && pct >= 0.0),
                    "expected non-negative deferred math, got {:?}",
                    items[1]
                );
            }
            other => panic!("expected box+math pair, got {other:?}"),
        }
        // `calc(0.5em + 100%)` at the 16px default font: `8px + 100%`.
        let s = style(
            &doc,
            div,
            "div { overflow-clip-margin: calc(0.5em + 100%); }",
            &vp(),
        );
        match s.get("overflow-clip-margin") {
            Some(CssValue::MathDeferred { px, pct, .. }) => {
                assert!((px - 8.0).abs() < 1e-3, "px = {px}");
                assert!((pct - 100.0).abs() < 1e-3, "pct = {pct}");
            }
            other => panic!("expected deferred math, got {other:?}"),
        }
        // Multi-value form, reversed order: `<length> <visual-box>`.
        let s = style(
            &doc,
            div,
            "div { overflow-clip-margin: calc(10px + 5%) content-box; }",
            &vp(),
        );
        match s.get("overflow-clip-margin") {
            Some(CssValue::List(items)) => {
                assert_eq!(items.len(), 2);
                assert!(
                    matches!(items[0], CssValue::MathDeferred { px, pct, .. } if px >= 0.0 && pct >= 0.0),
                    "expected non-negative deferred math, got {:?}",
                    items[0]
                );
                assert_eq!(items[1], kw("content-box"));
            }
            other => panic!("expected math+box pair, got {other:?}"),
        }
    }

    #[test]
    fn justify_items_is_not_inherited() {
        let (doc, div, p) = sample();
        // `justify-items` is not an inherited property per CSS Box
        // Alignment: the child's computed value is the initial value,
        // not the parent's.
        let s = style(&doc, p, "div { justify-items: center; }", &vp());
        assert_eq!(s.get("justify-items"), Some(&kw("normal")));
        // ... while the element that specified it keeps its value.
        let s = style(&doc, div, "div { justify-items: center; }", &vp());
        assert_eq!(s.get("justify-items"), Some(&kw("center")));
    }

    // ---- Dynamic pseudo-classes through Styler ----

    #[test]
    fn element_state_default_is_static_matching() {
        // Without an explicit state, dynamic pseudo-classes never match:
        // identical to the old `matches`-based cascade.
        let (doc, div, _) = sample();
        let author = parse_stylesheet("div { color: red; } div:hover { color: blue; }");
        let ua_ss = ua();
        let sheets = [&author];
        let styler = Styler::new(&doc, &sheets, &ua_ss, vp());
        let s = styler.style_for(div);
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
    }

    #[test]
    fn element_state_hover_applies_hover_rules() {
        let (doc, div, _) = sample();
        let author = parse_stylesheet("div { color: red; } div:hover { color: blue; }");
        let ua_ss = ua();
        let sheets = [&author];
        let styler = Styler::new(&doc, &sheets, &ua_ss, vp());

        styler.set_element_state(
            div,
            ElementState {
                hover: true,
                ..Default::default()
            },
        );
        let s = styler.style_for(div);
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );

        // Clearing the state drops the :hover rule (memoized cascade
        // results are invalidated on state change).
        styler.set_element_state(div, ElementState::default());
        let s = styler.style_for(div);
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
    }

    #[test]
    fn style_memo_survives_repeat_queries_until_state_changes() {
        // Repeat `style_for` calls in one epoch return equal styles
        // (the finished style is memoized per node); a state change
        // retires the memo and the next query recomputes.
        let (doc, div, _) = sample();
        let author = parse_stylesheet("div { color: red; } div:hover { color: blue; }");
        let ua_ss = ua();
        let sheets = [&author];
        let styler = Styler::new(&doc, &sheets, &ua_ss, vp());
        let red = CssValue::Color(Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        });
        let blue = CssValue::Color(Color {
            r: 0,
            g: 0,
            b: 255,
            a: 255,
        });
        // A no-op state change (same state) does not invalidate: the
        // memoized style is returned unchanged.
        styler.set_element_state(div, ElementState::default());
        assert_eq!(styler.style_for(div).get("color"), Some(&red));
        assert_eq!(styler.style_for(div).get("color"), Some(&red));
        // Hovering retires the memo: the recomputed style reflects the
        // new state, and repeat queries in the new epoch agree.
        styler.set_element_state(
            div,
            ElementState {
                hover: true,
                ..Default::default()
            },
        );
        assert_eq!(styler.style_for(div).get("color"), Some(&blue));
        assert_eq!(styler.style_for(div).get("color"), Some(&blue));
        // Clearing the state recomputes again.
        styler.set_element_state(div, ElementState::default());
        assert_eq!(styler.style_for(div).get("color"), Some(&red));
    }

    #[test]
    fn element_state_focus_and_active() {
        let (doc, div, _) = sample();
        let author = parse_stylesheet("div:active { color: green; } div:focus { color: purple; }");
        let ua_ss = ua();
        let sheets = [&author];
        let styler = Styler::new(&doc, &sheets, &ua_ss, vp());

        styler.set_element_state(
            div,
            ElementState {
                active: true,
                ..Default::default()
            },
        );
        assert_eq!(
            styler.style_for(div).get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 128,
                b: 0,
                a: 255
            }))
        );
        styler.set_element_state(
            div,
            ElementState {
                focus: true,
                ..Default::default()
            },
        );
        assert_eq!(
            styler.style_for(div).get("color"),
            Some(&CssValue::Color(Color {
                r: 128,
                g: 0,
                b: 128,
                a: 255
            }))
        );
    }

    #[test]
    fn element_state_is_per_node() {
        // State on one node must not leak into another node's matching.
        // `background-color` is not inherited, so the child's value is
        // unaffected by the parent's state.
        let (doc, div, p) = sample();
        let author = parse_stylesheet(
            "div:hover { background-color: blue; } p:hover { background-color: green; }",
        );
        let ua_ss = ua();
        let sheets = [&author];
        let styler = Styler::new(&doc, &sheets, &ua_ss, vp());

        styler.set_element_state(
            div,
            ElementState {
                hover: true,
                ..Default::default()
            },
        );
        assert_eq!(
            styler.style_for(div).get("background-color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );
        // `p` has no state: its :hover rule does not apply, and the div's
        // state does not leak to it (unset computes to the initial
        // transparent).
        assert_eq!(
            styler.style_for(p).get("background-color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 0
            }))
        );
    }

    #[test]
    fn rule_index_large_stylesheet_still_correct() {
        // Many rules across every bucket: the index must select the same
        // winners the brute-force loop did (specificity, source order,
        // !important, and inline styles all interact).
        let mut css = String::new();
        for i in 0..200 {
            css.push_str(&format!(".noise-{i} {{ color: rgb(1, 1, 1); }}\n"));
            css.push_str(&format!("#noise-{i} {{ color: rgb(2, 2, 2); }}\n"));
            css.push_str(&format!("span.noise-{i} {{ color: rgb(3, 3, 3); }}\n"));
        }
        css.push_str(
            "div { color: red; } .a { color: green; } #d { color: blue; } \
             div.a { color: yellow; } div#a.a { color: purple; } \
             [data-z] { color: orange; } div:first-child { color: teal; } \
             div:not(.zzz) { color: pink; } #d { color: black !important; }",
        );
        let (doc, div, _) = sample();
        let s = style(&doc, div, &css, &vp());
        // `!important` #d wins over everything else.
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            }))
        );
    }

    // ---- CRITICAL: rem/calc() font-size on the root must not recurse ----

    #[test]
    fn rem_on_root_resolves_against_initial() {
        // Per CSS Values 4, `rem` on the root element resolves against
        // the initial 16px, never recursively (this used to overflow the
        // stack on remote-stylesheet input). Computed `font-size` keeps
        // the relative unit; termination is the regression under test.
        let (doc, _) = Document::parse_html(b"<html><body><div></div></body></html>");
        let html = doc.document_element().unwrap();
        let s = style(&doc, html, "html { font-size: 1rem; }", &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(1.0, LengthUnit::Rem))
        );
        // And a descendant's `rem` math resolves against the root's
        // `rem`-derived size (2 * 16px), not 16px and not recursively.
        let div = doc.get_elements_by_tag_name("div").first().unwrap();
        let s = style(
            &doc,
            div,
            "html { font-size: 2rem; } div { width: calc(1rem + 1px); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(33.0, LengthUnit::Px))
        );
    }

    #[test]
    fn calc_font_size_on_root_does_not_recurse() {
        // `html{font-size:calc(1rem + 1px)}`: `math_context` asks for the
        // root font size while it is being computed. Must terminate (fail
        // closed), not overflow the stack.
        let (doc, _) = Document::parse_html(b"<html><body><div></div></body></html>");
        let html = doc.document_element().unwrap();
        let s = style(&doc, html, "html { font-size: calc(1rem + 1px); }", &vp());
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(17.0, LengthUnit::Px))
        );
    }

    #[test]
    fn rem_on_descendant_uses_root_size() {
        let (doc, _) = Document::parse_html(b"<html><body><div></div></body></html>");
        let div = doc.get_elements_by_tag_name("div").first().unwrap();
        let s = style(
            &doc,
            div,
            "html { font-size: 20px; } div { width: calc(1rem + 1px); }",
            &vp(),
        );
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(21.0, LengthUnit::Px))
        );
    }

    #[test]
    fn custom_property_css_wide_keywords() {
        let (doc, div, p) = sample();
        // `--x: initial` is guaranteed-invalid: never exposed literally.
        let s = style(&doc, p, "p { --x: initial; }", &vp());
        assert_eq!(s.get("--x"), None);
        // `var(--x)` for a guaranteed-invalid property is invalid at
        // computed-value time: the declaration drops, so `color`
        // computes to its initial black.
        let s = style(&doc, p, "p { --x: initial; color: var(--x); }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // ...but a fallback still applies.
        let s = style(
            &doc,
            p,
            "p { --x: initial; color: var(--x, green); }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 128,
                b: 0,
                a: 255
            }))
        );
        // Guaranteed-invalid does not inherit the grandparent's value:
        // the parent's `initial` shadows it for the whole subtree.
        let s = style(
            &doc,
            p,
            "div { --x: red; } #d { --x: initial; } p { color: var(--x, green); }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 128,
                b: 0,
                a: 255
            }))
        );
        // `--x: inherit` takes the parent's value.
        let s = style(
            &doc,
            p,
            "div { --x: red; } p { --x: inherit; color: var(--x); }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // `--x: inherit` with no ancestor value is guaranteed-invalid.
        let s = style(
            &doc,
            p,
            "p { --x: inherit; color: var(--x, green); }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 128,
                b: 0,
                a: 255
            }))
        );
        // `unset` behaves like `inherit` for custom properties.
        let s = style(
            &doc,
            p,
            "div { --x: red; } p { --x: unset; color: var(--x); }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // Sanity: the parent's own declaration still resolves normally.
        // (`red` is a named color, not a keyword.)
        let s = style(&doc, div, "div { --x: red; }", &vp());
        assert_eq!(
            s.get("--x"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // Guaranteed-invalid inside math: the fallback applies, and
        // without one the declaration is invalid at computed-value time.
        let s = style(
            &doc,
            p,
            "p { --x: initial; width: calc(var(--x, 5px) + 1px); }",
            &vp(),
        );
        assert_eq!(s.get("width"), Some(&CssValue::Length(6.0, LengthUnit::Px)));
        let s = style(
            &doc,
            p,
            "p { --x: initial; width: calc(var(--x) + 1px); }",
            &vp(),
        );
        assert_eq!(s.get("width"), Some(&kw("auto")));
    }

    #[test]
    fn negative_padding_and_border_width_rejected() {
        let (doc, _, p) = sample();
        // Literal negatives are invalid at computed-value time: the
        // declarations drop, so the properties compute to their
        // initials (0 for padding, `medium` for border-width) — never
        // the negative length.
        let s = style(&doc, p, "p { padding-top: -5px; }", &vp());
        assert_eq!(
            s.get("padding-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        let s = style(&doc, p, "p { border-top-width: -2px; }", &vp());
        assert_eq!(s.get("border-top-width"), Some(&kw("medium")));
        // Negative math results are invalid too (post-evaluation check).
        let s = style(&doc, p, "p { padding-top: calc(1px - 10px); }", &vp());
        assert_eq!(
            s.get("padding-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        let s = style(&doc, p, "p { border-top-width: calc(1px - 10px); }", &vp());
        assert_eq!(s.get("border-top-width"), Some(&kw("medium")));
        // Margins may be negative; text-indent may be negative.
        let s = style(&doc, p, "p { margin-top: -5px; text-indent: -1em; }", &vp());
        assert_eq!(
            s.get("margin-top"),
            Some(&CssValue::Length(-5.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("text-indent"),
            Some(&CssValue::Length(-1.0, LengthUnit::Em))
        );
        // Non-negative values still pass.
        let s = style(
            &doc,
            p,
            "p { padding-top: 5px; border-top-width: thin; }",
            &vp(),
        );
        assert_eq!(
            s.get("padding-top"),
            Some(&CssValue::Length(5.0, LengthUnit::Px))
        );
        assert_eq!(s.get("border-top-width"), Some(&kw("thin")));
    }

    #[test]
    fn negative_deferred_math_padding_rejected() {
        let (doc, _, p) = sample();
        // `min(-5px, 20%)` is non-affine, so it freezes as
        // `MathDeferred { pct: 0.0, px: -5.0, had_pct: true }` (base-0
        // approximation). The sign is known at computed-value time, so
        // the negative-padding gate rejects it exactly like a literal
        // `-5px` (CSS 2.1 §8.4) instead of leaking -5px into layout.
        let s = style(&doc, p, "p { padding-top: min(-5px, 20%); }", &vp());
        assert_eq!(
            s.get("padding-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // The same gate closes the pre-existing affine hole:
        // `calc(-5px + 0%)` folds to `pct: 0.0, px: -5.0` and is
        // unconditionally negative, so it is invalid too.
        let s = style(&doc, p, "p { padding-top: calc(-5px + 0%); }", &vp());
        assert_eq!(
            s.get("padding-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // A genuinely percentage-dependent negative still passes
        // through for layout to judge at used-value time: `min(-5px,
        // -20%)` may resolve non-negative once the base is known.
        let s = style(&doc, p, "p { padding-top: calc(-5px + 20%); }", &vp());
        assert_eq!(
            s.get("padding-top"),
            Some(&CssValue::MathDeferred {
                pct: 20.0,
                px: -5.0,
                had_pct: true,
                affine: true
            })
        );
        // Positive deferred math still passes through untouched.
        let s = style(&doc, p, "p { padding-top: max(10px, 0%); }", &vp());
        assert_eq!(
            s.get("padding-top"),
            Some(&CssValue::MathDeferred {
                pct: 0.0,
                px: 10.0,
                had_pct: true,
                affine: false
            })
        );
    }

    #[test]
    fn revert_rolls_back_to_lower_origin() {
        let (doc, div, _) = sample();
        // UA sets `div { display: block }`; author `revert` rolls back
        // to it.
        let s = style(&doc, div, "div { display: revert; }", &vp());
        assert_eq!(s.get("display"), Some(&kw("block")));
        // `revert-layer` without a layer model is `revert`.
        let s = style(&doc, div, "div { display: revert-layer; }", &vp());
        assert_eq!(s.get("display"), Some(&kw("block")));
        // Inline `revert` is author origin: still rolls back to UA.
        let mut doc2 = doc;
        doc2.set_attribute(div, "style", "display: revert");
        let author = parse_stylesheet("div { display: inline; }");
        let s = compute_style(&doc2, div, &[&author], &ua(), &vp());
        assert_eq!(s.get("display"), Some(&kw("block")));
        // `!important` revert still leaves the author origin entirely.
        let s = style(&doc2, div, "div { display: revert !important; }", &vp());
        assert_eq!(s.get("display"), Some(&kw("block")));
        // No lower origin declares `color`: `revert` behaves as `unset`.
        // `color` is inherited, so an explicit `unset` with nothing to
        // inherit falls back to the initial value (black).
        let s = style(
            &doc2,
            div,
            "div { color: red; } div { color: revert; }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // A lower `revert` has nothing to roll back to: `unset`.
        let s = style(&doc2, div, "div { display: revert; }", &vp());
        assert_eq!(s.get("display"), Some(&kw("block")));
    }

    #[test]
    fn revert_via_var_does_not_loop() {
        let (doc, div, _) = sample();
        // The lower-origin winner resolves to `revert` through `var()`:
        // this must behave as `unset` (not loop forever). `display` is
        // not inherited, so `unset` gives the initial `inline`.
        let ua_sheet = parse_stylesheet("div { --d: revert; display: var(--d); }");
        let s = compute_style(&doc, div, &[], &ua_sheet, &vp());
        // Sanity: without the author `revert`, the UA `var(--d)` itself
        // is `revert` in the lowest origin -> `unset` -> `inline`.
        assert_eq!(s.get("display"), Some(&kw("inline")));
        // Now the author `revert` rolls back to that UA declaration,
        // which resolves to `revert` -> `unset` -> `inline` (no hang).
        let author = parse_stylesheet("div { display: revert; }");
        let s = compute_style(&doc, div, &[&author], &ua_sheet, &vp());
        assert_eq!(s.get("display"), Some(&kw("inline")));
    }

    #[test]
    fn revert_custom_property() {
        let (doc, div, p) = sample();
        // No lower origin: `revert` behaves as `unset`, and custom
        // properties inherit, so the parent's value wins.
        let s = style(
            &doc,
            p,
            "div { --x: red; } p { --x: revert; color: var(--x); }",
            &vp(),
        );
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // ...or guaranteed-invalid when no ancestor defines it.
        let s = style(&doc, p, "p { --x: revert; color: var(--x, green); }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 128,
                b: 0,
                a: 255
            }))
        );
        // Lower-origin winner selection: a UA `--x` survives an author
        // `revert`. (`blue` is a named color, not a keyword.)
        let ua = parse_stylesheet("div { --x: blue; }");
        let author = parse_stylesheet("div { --x: revert; }");
        let s = compute_style(&doc, div, &[&author], &ua, &vp());
        assert_eq!(
            s.get("--x"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );
    }

    #[test]
    fn index_backed_style_contract() {
        // Locks in the array-backed ComputedStyle: every supported
        // property resolves (initial fallback), unknown names miss,
        // inheritance flows through the index-addressed memo, the
        // font-size reentry during math evaluation agrees, and
        // restyling the same node is stable.
        let (doc, div, p) = sample();
        let author = parse_stylesheet(
            "div { color: red; font-size: 20px; } p { font-size: 2em; width: calc(1em + 10px); }",
        );
        let ua = ua();
        let vp = vp();
        let sheets = [&author];
        let styler = Styler::new(&doc, &sheets, &ua, vp);
        let s = styler.style_for(p);
        for prop in COMPUTED_PROPS {
            assert!(s.get(prop).is_some(), "no computed value for {prop}");
        }
        assert_eq!(s.get("not-a-property"), None);
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        // 2em of the parent's 20px, kept relative at computed-value time.
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(2.0, LengthUnit::Em))
        );
        // The em inside calc() resolved against the memoized font size.
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(50.0, LengthUnit::Px))
        );
        let s2 = styler.style_for(p);
        for prop in COMPUTED_PROPS {
            assert_eq!(s.get(prop), s2.get(prop), "restyle drift for {prop}");
        }
        // Unrelated nodes do not share memo state.
        let sd = styler.style_for(div);
        assert_eq!(
            sd.get("font-size"),
            Some(&CssValue::Length(20.0, LengthUnit::Px))
        );
    }

    #[test]
    fn grid_line_validation_rejects_hostile_lines() {
        let (doc, div, _) = sample();
        // Huge line numbers would force a huge placement grid in layout:
        // invalid at computed-value time, so the declaration is dropped
        // and the property falls back to its initial value `auto`.
        let s = style(&doc, div, ".a { grid-row: 1/1000000; }", &vp());
        assert_eq!(s.get("grid-row"), Some(&kw("auto")));
        let s = style(&doc, div, ".a { grid-column: 3000000000; }", &vp());
        assert_eq!(s.get("grid-column"), Some(&kw("auto")));
        // CSS requires <integer>; 0 is never a valid line.
        let s = style(&doc, div, ".a { grid-row: 0; }", &vp());
        assert_eq!(s.get("grid-row"), Some(&kw("auto")));
        let s = style(&doc, div, ".a { grid-row: 1.5; }", &vp());
        assert_eq!(s.get("grid-row"), Some(&kw("auto")));
        let s = style(&doc, div, ".a { grid-row: span 2.5; }", &vp());
        assert_eq!(s.get("grid-row"), Some(&kw("auto")));
        let s = style(&doc, div, ".a { grid-row: span 0; }", &vp());
        assert_eq!(s.get("grid-row"), Some(&kw("auto")));
    }

    #[test]
    fn grid_line_validation_keeps_valid_placements() {
        let (doc, div, _) = sample();
        let s = style(&doc, div, ".a { grid-row: auto; }", &vp());
        assert_eq!(s.get("grid-row"), Some(&kw("auto")));
        let s = style(&doc, div, ".a { grid-row: 2; }", &vp());
        assert_eq!(s.get("grid-row"), Some(&CssValue::Number(2.0)));
        let s = style(&doc, div, ".a { grid-row: -1; }", &vp());
        assert_eq!(s.get("grid-row"), Some(&CssValue::Number(-1.0)));
        let s = style(&doc, div, ".a { grid-row: span 3; }", &vp());
        assert!(s.get("grid-row").is_some());
        // `/`-separated start/end survives validation.
        let s = style(&doc, div, ".a { grid-row: 1 / 3; }", &vp());
        assert!(s.get("grid-row").is_some());
        let s = style(&doc, div, ".a { grid-column: 2 / span 2; }", &vp());
        assert!(s.get("grid-column").is_some());
    }

    #[test]
    fn fragmentation_props_cascade() {
        let (doc, div, _) = sample();
        // Declared values surface through `get_str`/`get`, which is what
        // bowser_layout's fragmentation resolver reads.
        let s = style(
            &doc,
            div,
            ".a { break-before: page; break-after: avoid; break-inside: avoid; }",
            &vp(),
        );
        assert_eq!(s.get_str("break-before"), Some("page"));
        assert_eq!(s.get_str("break-after"), Some("avoid"));
        assert_eq!(s.get_str("break-inside"), Some("avoid"));
        let s = style(&doc, div, ".a { widows: 3; orphans: 4; }", &vp());
        assert_eq!(s.get("widows"), Some(&CssValue::Number(3.0)));
        assert_eq!(s.get("orphans"), Some(&CssValue::Number(4.0)));
    }

    #[test]
    fn fragmentation_props_default_to_initial() {
        let (doc, div, _) = sample();
        // No declarations: initial values fill in (`auto` / `2`).
        let s = style(&doc, div, ".a { color: red; }", &vp());
        assert_eq!(s.get_str("break-before"), Some("auto"));
        assert_eq!(s.get_str("break-after"), Some("auto"));
        assert_eq!(s.get_str("break-inside"), Some("auto"));
        assert_eq!(s.get("widows"), Some(&CssValue::Number(2.0)));
        assert_eq!(s.get("orphans"), Some(&CssValue::Number(2.0)));
    }

    #[test]
    fn break_before_after_accept_full_keyword_set() {
        let (doc, div, _) = sample();
        // CSS Fragmentation 3 break-before/break-after keyword set.
        for word in [
            "auto",
            "avoid",
            "page",
            "left",
            "right",
            "recto",
            "verso",
            "avoid-page",
            "column",
            "avoid-column",
        ] {
            let css = format!(".a {{ break-before: {word}; break-after: {word}; }}");
            let s = style(&doc, div, &css, &vp());
            assert_eq!(s.get_str("break-before"), Some(word), "word={word}");
            assert_eq!(s.get_str("break-after"), Some(word), "word={word}");
        }
        // CSS-wide keywords still work on these properties.
        let s = style(&doc, div, ".a { break-before: initial; }", &vp());
        assert_eq!(s.get_str("break-before"), Some("auto"));
    }

    #[test]
    fn fragmentation_props_reject_invalid() {
        let (doc, div, _) = sample();
        // Fail closed: invalid values are dropped, the initial value
        // fills in.
        let s = style(&doc, div, ".a { break-before: banana; }", &vp());
        assert_eq!(s.get_str("break-before"), Some("auto"));
        let s = style(&doc, div, ".a { break-after: 5; }", &vp());
        assert_eq!(s.get_str("break-after"), Some("auto"));
        // `page` is not in break-inside's avoid-only keyword set.
        let s = style(&doc, div, ".a { break-inside: page; }", &vp());
        assert_eq!(s.get_str("break-inside"), Some("auto"));
        let s = style(&doc, div, ".a { break-inside: column; }", &vp());
        assert_eq!(s.get_str("break-inside"), Some("auto"));
        // `widows`/`orphans` require an integer >= 1.
        let s = style(&doc, div, ".a { widows: 0; }", &vp());
        assert_eq!(s.get("widows"), Some(&CssValue::Number(2.0)));
        let s = style(&doc, div, ".a { orphans: -1; }", &vp());
        assert_eq!(s.get("orphans"), Some(&CssValue::Number(2.0)));
        let s = style(&doc, div, ".a { widows: 2.5; }", &vp());
        assert_eq!(s.get("widows"), Some(&CssValue::Number(2.0)));
        let s = style(&doc, div, ".a { widows: red; }", &vp());
        assert_eq!(s.get("widows"), Some(&CssValue::Number(2.0)));
        // But 1 is valid.
        let s = style(&doc, div, ".a { widows: 1; orphans: 1; }", &vp());
        assert_eq!(s.get("widows"), Some(&CssValue::Number(1.0)));
        assert_eq!(s.get("orphans"), Some(&CssValue::Number(1.0)));
    }

    #[test]
    fn column_props_cascade() {
        let (doc, div, _) = sample();
        // Initials: count/width `auto`, gap `normal` (1em in multicol).
        let s = style(&doc, div, ".a { color: red; }", &vp());
        assert_eq!(s.get("column-count"), Some(&kw("auto")));
        assert_eq!(s.get("column-width"), Some(&kw("auto")));
        assert_eq!(s.get("column-gap"), Some(&kw("normal")));
        // Longhands validate: count needs an integer >= 1, width a
        // non-negative length.
        let s = style(
            &doc,
            div,
            ".a { column-count: 3; column-width: 12em; }",
            &vp(),
        );
        assert_eq!(s.get("column-count"), Some(&CssValue::Number(3.0)));
        assert_eq!(
            s.get("column-width"),
            Some(&CssValue::Length(12.0, LengthUnit::Em))
        );
        let s = style(&doc, div, ".a { column-count: 0; }", &vp());
        assert_eq!(s.get("column-count"), Some(&kw("auto")));
        let s = style(&doc, div, ".a { column-count: 1.5; }", &vp());
        assert_eq!(s.get("column-count"), Some(&kw("auto")));
        let s = style(&doc, div, ".a { column-width: -5px; }", &vp());
        assert_eq!(s.get("column-width"), Some(&kw("auto")));
        // `columns` shorthand: either order, omitted side resets to auto.
        let s = style(&doc, div, ".a { columns: 12em 3; }", &vp());
        assert_eq!(s.get("column-count"), Some(&CssValue::Number(3.0)));
        assert_eq!(
            s.get("column-width"),
            Some(&CssValue::Length(12.0, LengthUnit::Em))
        );
        let s = style(&doc, div, ".a { columns: 2 20px; }", &vp());
        assert_eq!(s.get("column-count"), Some(&CssValue::Number(2.0)));
        assert_eq!(
            s.get("column-width"),
            Some(&CssValue::Length(20.0, LengthUnit::Px))
        );
        let s = style(&doc, div, ".a { columns: 2; }", &vp());
        assert_eq!(s.get("column-count"), Some(&CssValue::Number(2.0)));
        assert_eq!(s.get("column-width"), Some(&kw("auto")));
        let s = style(&doc, div, ".a { columns: 10em; }", &vp());
        assert_eq!(
            s.get("column-width"),
            Some(&CssValue::Length(10.0, LengthUnit::Em))
        );
        assert_eq!(s.get("column-count"), Some(&kw("auto")));
        // Two widths (or anything else) is invalid: shorthand dropped,
        // both sides keep their initial.
        let s = style(&doc, div, ".a { columns: 1px 2px; }", &vp());
        assert_eq!(s.get("column-count"), Some(&kw("auto")));
        assert_eq!(s.get("column-width"), Some(&kw("auto")));
        // Deferred `var()` shorthand expands in order-independent form.
        let s = style(
            &doc,
            div,
            ".a { --c: 3; --w: 12em; columns: var(--c) var(--w); }",
            &vp(),
        );
        assert_eq!(s.get("column-count"), Some(&CssValue::Number(3.0)));
        assert_eq!(
            s.get("column-width"),
            Some(&CssValue::Length(12.0, LengthUnit::Em))
        );
    }

    #[test]
    fn column_span_and_fill_cascade() {
        let (doc, div, _) = sample();
        // Initials: `column-span: none`, `column-fill: balance`.
        let s = style(&doc, div, ".a { color: red; }", &vp());
        assert_eq!(s.get_str("column-span"), Some("none"));
        assert_eq!(s.get_str("column-fill"), Some("balance"));
        // Parse -> cascade -> get_str round trip for the keywords the
        // multicol layout worker reads.
        let s = style(&doc, div, ".a { column-span: all; }", &vp());
        assert_eq!(s.get_str("column-span"), Some("all"));
        let s = style(&doc, div, ".a { column-fill: auto; }", &vp());
        assert_eq!(s.get_str("column-fill"), Some("auto"));
        // Single-keyword grammars: anything else is invalid at
        // computed-value time and keeps the initial.
        let s = style(&doc, div, ".a { column-span: foo; }", &vp());
        assert_eq!(s.get_str("column-span"), Some("none"));
        let s = style(&doc, div, ".a { column-fill: fill; }", &vp());
        assert_eq!(s.get_str("column-fill"), Some("balance"));
        let s = style(&doc, div, ".a { column-span: none; }", &vp());
        assert_eq!(s.get_str("column-span"), Some("none"));
    }

    #[test]
    fn column_rule_shorthand_cascade() {
        let (doc, div, _) = sample();
        // Initials mirror border-*: width `medium`, style `none`, color
        // `currentcolor`.
        let s = style(&doc, div, ".a { color: red; }", &vp());
        assert_eq!(s.get("column-rule-width"), Some(&kw("medium")));
        assert_eq!(s.get("column-rule-style"), Some(&kw("none")));
        assert_eq!(s.get("column-rule-color"), Some(&kw("currentcolor")));
        // Shorthand expands eagerly: any component order, omitted parts
        // reset to initial.
        let s = style(&doc, div, ".a { column-rule: 2px solid red; }", &vp());
        assert_eq!(
            s.get("column-rule-width"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        assert_eq!(s.get("column-rule-style"), Some(&kw("solid")));
        assert_eq!(
            s.get("column-rule-color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
        let s = style(&doc, div, ".a { column-rule: solid 2px red; }", &vp());
        assert_eq!(
            s.get("column-rule-width"),
            Some(&CssValue::Length(2.0, LengthUnit::Px))
        );
        assert_eq!(s.get("column-rule-style"), Some(&kw("solid")));
        let s = style(&doc, div, ".a { column-rule: dashed; }", &vp());
        assert_eq!(s.get("column-rule-width"), Some(&kw("medium")));
        assert_eq!(s.get("column-rule-style"), Some(&kw("dashed")));
        assert_eq!(s.get("column-rule-color"), Some(&kw("currentcolor")));
        let s = style(&doc, div, ".a { column-rule: thick; }", &vp());
        assert_eq!(s.get("column-rule-width"), Some(&kw("thick")));
        assert_eq!(s.get("column-rule-style"), Some(&kw("none")));
        // Unclassifiable or duplicated components drop the shorthand.
        let s = style(&doc, div, ".a { column-rule: 1px 2px; }", &vp());
        assert_eq!(s.get("column-rule-width"), Some(&kw("medium")));
        assert_eq!(s.get("column-rule-style"), Some(&kw("none")));
        let s = style(&doc, div, ".a { column-rule: solid dotted; }", &vp());
        assert_eq!(s.get("column-rule-style"), Some(&kw("none")));
        // Deferred `var()` shorthand expands after substitution.
        let s = style(
            &doc,
            div,
            ".a { --w: 3px; column-rule: var(--w) solid; }",
            &vp(),
        );
        assert_eq!(
            s.get("column-rule-width"),
            Some(&CssValue::Length(3.0, LengthUnit::Px))
        );
        assert_eq!(s.get("column-rule-style"), Some(&kw("solid")));
        assert_eq!(s.get("column-rule-color"), Some(&kw("currentcolor")));
        // Longhand validation mirrors border-*: width is a non-negative
        // length or thin/medium/thick, style the border-style keyword
        // set, color `<color> | currentcolor`.
        let s = style(&doc, div, ".a { column-rule-width: thin; }", &vp());
        assert_eq!(s.get_str("column-rule-width"), Some("thin"));
        let s = style(&doc, div, ".a { column-rule-width: -2px; }", &vp());
        assert_eq!(s.get_str("column-rule-width"), Some("medium"));
        let s = style(&doc, div, ".a { column-rule-style: groovy; }", &vp());
        assert_eq!(s.get_str("column-rule-style"), Some("none"));
        let s = style(&doc, div, ".a { column-rule-color: blue; }", &vp());
        assert_eq!(
            s.get("column-rule-color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            }))
        );
    }

    #[test]
    fn percentage_calc_rejected_where_percentages_invalid() {
        let (doc, div, _) = sample();
        // Percentages are not valid `scroll-margin` (CSS Scroll Snap 1),
        // so a percentage `calc()` must fail the post-evaluation re-check
        // (it evaluates to `MathDeferred`) and fall back to the initial
        // `0px` instead of surviving validation.
        let s = style(&doc, div, ".a { scroll-margin-top: calc(10%); }", &vp());
        assert_eq!(
            s.get("scroll-margin-top"),
            Some(&CssValue::Length(0.0, LengthUnit::Px))
        );
        // Percentage-free math still evaluates normally.
        let s = style(
            &doc,
            div,
            ".a { scroll-margin-top: calc(10px + 5px); }",
            &vp(),
        );
        assert_eq!(
            s.get("scroll-margin-top"),
            Some(&CssValue::Length(15.0, LengthUnit::Px))
        );
        // Percentages are not valid `<line-width>` either: percentage
        // `calc()` falls back to `medium` for both `column-rule-width`
        // and `border-*-width`.
        let s = style(&doc, div, ".a { column-rule-width: calc(10%); }", &vp());
        assert_eq!(s.get("column-rule-width"), Some(&kw("medium")));
        let s = style(&doc, div, ".a { border-top-width: calc(10%); }", &vp());
        assert_eq!(s.get("border-top-width"), Some(&kw("medium")));
        // Percentage-free math still evaluates for border widths.
        let s = style(
            &doc,
            div,
            ".a { border-top-width: calc(1px + 2px); }",
            &vp(),
        );
        assert_eq!(
            s.get("border-top-width"),
            Some(&CssValue::Length(3.0, LengthUnit::Px))
        );
    }

    #[test]
    fn widows_orphans_inherit() {
        let (doc, _, p) = sample();
        // `widows`/`orphans` are inherited (CSS Fragmentation 3).
        let s = style(&doc, p, "div { widows: 3; orphans: 4; }", &vp());
        assert_eq!(s.get("widows"), Some(&CssValue::Number(3.0)));
        assert_eq!(s.get("orphans"), Some(&CssValue::Number(4.0)));
        // `break-*` are not inherited: the child gets the initial.
        let s = style(&doc, p, "div { break-before: page; }", &vp());
        assert_eq!(s.get_str("break-before"), Some("auto"));
        // Break hints also work from inline styles.
        let mut doc2 = Document::new();
        let root = doc2.root();
        let d2 = doc2.create_element("div");
        doc2.append_child(root, d2);
        doc2.set_attribute(d2, "style", "break-after: page; widows: 5;");
        let s = style(&doc2, d2, "", &vp());
        assert_eq!(s.get_str("break-after"), Some("page"));
        assert_eq!(s.get("widows"), Some(&CssValue::Number(5.0)));
    }

    #[test]
    fn mix_blend_mode_cascades() {
        let (doc, div, p) = sample();
        // Initial value is `normal`.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(s.get_str("mix-blend-mode"), Some("normal"));
        // The six supported keywords validate.
        for mode in [
            "normal", "multiply", "screen", "overlay", "darken", "lighten",
        ] {
            let s = style(
                &doc,
                div,
                &format!("div {{ mix-blend-mode: {mode}; }}"),
                &vp(),
            );
            assert_eq!(s.get_str("mix-blend-mode"), Some(mode), "{mode}");
        }
        // Unsupported keywords are rejected: the initial value applies.
        let s = style(&doc, div, "div { mix-blend-mode: difference; }", &vp());
        assert_eq!(s.get_str("mix-blend-mode"), Some("normal"));
        // Not inherited: the child keeps the initial value.
        let s = style(&doc, p, "div { mix-blend-mode: multiply; }", &vp());
        assert_eq!(s.get_str("mix-blend-mode"), Some("normal"));
    }

    #[test]
    fn transform_cascades() {
        use crate::value::TransformFunction;
        let (doc, div, p) = sample();
        // Initial value is `none`.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(s.get_str("transform"), Some("none"));
        // `none` computes to the keyword.
        let s = style(&doc, div, "div { transform: none; }", &vp());
        assert_eq!(s.get_str("transform"), Some("none"));
        // Function lists compute to `Transform` values.
        let s = style(&doc, div, "div { transform: translate(10px, 20%); }", &vp());
        assert_eq!(
            s.get("transform"),
            Some(&CssValue::Transform(vec![TransformFunction::Translate {
                x: CssValue::Length(10.0, LengthUnit::Px),
                y: CssValue::Percentage(20.0),
            }]))
        );
        let s = style(
            &doc,
            div,
            "div { transform: scale(2) rotate(45deg); }",
            &vp(),
        );
        // A function list computes to a single `Transform` with one
        // entry per function.
        match s.get("transform") {
            Some(CssValue::Transform(fns)) => {
                assert_eq!(fns.len(), 2);
                assert!(matches!(fns[0], TransformFunction::Scale { .. }));
                assert!(matches!(fns[1], TransformFunction::Rotate { .. }));
            }
            other => panic!("expected Transform, got {other:?}"),
        }
        // Valid-but-unrenderable functions are retained opaque instead
        // of collapsing to `none`: the box still becomes a containing
        // block for fixed descendants.
        for good in [
            "div { transform: skew(10deg); }",
            "div { transform: matrix(1, 0, 0, 1, 0, 0); }",
            "div { transform: perspective(500px); }",
            "div { transform: translate3d(1px, 2px, 3px); }",
            "div { transform: translate(calc(10px + 5px)); }",
        ] {
            let s = style(&doc, div, good, &vp());
            assert!(
                matches!(s.get("transform"), Some(CssValue::Transform(_))),
                "{good}"
            );
        }
        // Invalid values are rejected: the initial value applies.
        for bad in [
            "div { transform: translate(); }",
            "div { transform: translate(1 2 3 4 5); }",
            "div { transform: scale(2px); }",
            "div { transform: rotate(10px); }",
            "div { transform: matrix(); }",
            "div { transform: foo(); }",
            "div { transform: none translate(10px); }",
        ] {
            let s = style(&doc, div, bad, &vp());
            assert_eq!(s.get_str("transform"), Some("none"), "{bad}");
        }
        // Not inherited: the child keeps the initial value.
        let s = style(&doc, p, "div { transform: translate(10px); }", &vp());
        assert_eq!(s.get_str("transform"), Some("none"));
    }

    #[test]
    fn fixed_cb_properties_cascade() {
        let (doc, div, p) = sample();
        // Initial values.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(s.get_str("will-change"), Some("auto"));
        assert_eq!(s.get_str("perspective"), Some("none"));
        assert_eq!(s.get_str("contain"), Some("none"));
        assert!(matches!(
            s.get("filter"),
            Some(CssValue::Filter(Filter::None))
        ));

        // will-change: a single hint and a hint list.
        let s = style(&doc, div, "div { will-change: transform; }", &vp());
        assert_eq!(s.get_str("will-change"), Some("transform"));
        let s = style(&doc, div, "div { will-change: transform, opacity; }", &vp());
        assert_eq!(
            s.get("will-change"),
            Some(&CssValue::List(vec![kw("transform"), kw("opacity")]))
        );
        // Hint keywords are case-insensitive.
        let s = style(&doc, div, "div { will-change: Transform, OPACITY; }", &vp());
        assert_eq!(
            s.get("will-change"),
            Some(&CssValue::List(vec![kw("transform"), kw("opacity")]))
        );
        // Invalid: `auto` must stand alone, `none` is not a value, and
        // non-ident values are rejected.
        for bad in [
            "div { will-change: auto, transform; }",
            "div { will-change: transform, auto; }",
            "div { will-change: none; }",
            "div { will-change: 5px; }",
            "div { will-change: rotate(45deg); }",
        ] {
            let s = style(&doc, div, bad, &vp());
            assert_eq!(s.get_str("will-change"), Some("auto"), "{bad}");
        }

        // filter: function lists compute to `Filter` values.
        let s = style(
            &doc,
            div,
            "div { filter: blur(5px) grayscale(50%); }",
            &vp(),
        );
        match s.get("filter") {
            Some(CssValue::Filter(Filter::List(fns))) => {
                assert_eq!(fns.len(), 2);
                assert_eq!(fns[0].name, "blur");
                assert_eq!(fns[0].raw_args, "5px");
                assert_eq!(fns[1].name, "grayscale");
                assert_eq!(fns[1].raw_args, "50%");
            }
            other => panic!("expected Filter::List, got {other:?}"),
        }
        // `none` computes to the initial value.
        let s = style(&doc, div, "div { filter: none; }", &vp());
        assert!(matches!(
            s.get("filter"),
            Some(CssValue::Filter(Filter::None))
        ));
        // Unknown functions, empty bodies, and comma-separated lists
        // are invalid: the initial value applies.
        for bad in [
            "div { filter: frobnicate(1); }",
            "div { filter: blur(); }",
            "div { filter: blur(5px), grayscale(50%); }",
        ] {
            let s = style(&doc, div, bad, &vp());
            assert!(
                matches!(s.get("filter"), Some(CssValue::Filter(Filter::None))),
                "{bad}"
            );
        }

        // perspective: a length or `none`.
        let s = style(&doc, div, "div { perspective: 500px; }", &vp());
        assert_eq!(
            s.get("perspective"),
            Some(&CssValue::Length(500.0, LengthUnit::Px))
        );
        let s = style(&doc, div, "div { perspective: none; }", &vp());
        assert_eq!(s.get_str("perspective"), Some("none"));
        // Negative lengths, percentages, and unitless nonzero numbers
        // are invalid.
        for bad in [
            "div { perspective: -5px; }",
            "div { perspective: 50%; }",
            "div { perspective: 5; }",
        ] {
            let s = style(&doc, div, bad, &vp());
            assert_eq!(s.get_str("perspective"), Some("none"), "{bad}");
        }

        // contain: single keywords.
        for (css, expect) in [
            ("div { contain: none; }", "none"),
            ("div { contain: strict; }", "strict"),
            ("div { contain: content; }", "content"),
            ("div { contain: paint; }", "paint"),
            ("div { contain: size; }", "size"),
        ] {
            let s = style(&doc, div, css, &vp());
            assert_eq!(s.get_str("contain"), Some(expect), "{css}");
        }
        // Subsets are order-independent.
        let s = style(&doc, div, "div { contain: size layout; }", &vp());
        assert_eq!(
            s.get("contain"),
            Some(&CssValue::List(vec![kw("size"), kw("layout")]))
        );
        let s = style(&doc, div, "div { contain: paint size; }", &vp());
        assert_eq!(
            s.get("contain"),
            Some(&CssValue::List(vec![kw("paint"), kw("size")]))
        );
        // Duplicates, unknown keywords, and mixing `none`/`strict`/
        // `content` with parts are invalid.
        for bad in [
            "div { contain: paint paint; }",
            "div { contain: none paint; }",
            "div { contain: strict paint; }",
            "div { contain: banana; }",
            "div { contain: size size layout style paint paint; }",
        ] {
            let s = style(&doc, div, bad, &vp());
            assert_eq!(s.get_str("contain"), Some("none"), "{bad}");
        }

        // None of the four inherit: the child keeps the initial values.
        let s = style(
            &doc,
            p,
            "div { will-change: transform; filter: blur(5px); perspective: 500px; contain: paint; }",
            &vp(),
        );
        assert_eq!(s.get_str("will-change"), Some("auto"));
        assert!(matches!(
            s.get("filter"),
            Some(CssValue::Filter(Filter::None))
        ));
        assert_eq!(s.get_str("perspective"), Some("none"));
        assert_eq!(s.get_str("contain"), Some("none"));

        // CSS-wide keywords still work on the new properties.
        let s = style(
            &doc,
            p,
            "div { filter: blur(5px); } p { filter: inherit; }",
            &vp(),
        );
        assert!(matches!(
            s.get("filter"),
            Some(CssValue::Filter(Filter::List(_)))
        ));
    }

    #[test]
    fn transform_origin_cascades() {
        use crate::value::TransformOrigin;
        let (doc, div, p) = sample();
        let initial = CssValue::List(vec![CssValue::Percentage(50.0), CssValue::Percentage(50.0)]);
        // Initial value is `50% 50%`.
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(s.get("transform-origin"), Some(&initial));
        // Keywords, lengths, and percentages validate and compute as-is.
        let s = style(&doc, div, "div { transform-origin: left top; }", &vp());
        assert_eq!(
            s.get("transform-origin"),
            Some(&CssValue::List(vec![
                CssValue::Keyword("left".into()),
                CssValue::Keyword("top".into()),
            ]))
        );
        let s = style(&doc, div, "div { transform-origin: 10px 25%; }", &vp());
        assert_eq!(
            s.get("transform-origin"),
            Some(&CssValue::List(vec![
                CssValue::Length(10.0, LengthUnit::Px),
                CssValue::Percentage(25.0),
            ]))
        );
        // The computed value parses into a pivot via `TransformOrigin`.
        let origin = TransformOrigin::parse(s.get("transform-origin").unwrap()).unwrap();
        assert_eq!(
            origin,
            TransformOrigin {
                x: CssValue::Length(10.0, LengthUnit::Px),
                y: CssValue::Percentage(25.0),
            }
        );
        // Invalid values are rejected: the initial value applies.
        for bad in [
            "div { transform-origin: left right; }",
            "div { transform-origin: 1px 2px 3px; }",
            "div { transform-origin: red; }",
        ] {
            let s = style(&doc, div, bad, &vp());
            assert_eq!(s.get("transform-origin"), Some(&initial), "{bad}");
        }
        // Not inherited: the child keeps the initial value.
        let s = style(&doc, p, "div { transform-origin: left top; }", &vp());
        assert_eq!(s.get("transform-origin"), Some(&initial));
    }

    #[test]
    fn aspect_ratio_parses_and_computes() {
        let (doc, div, _) = sample();
        let ratio = |r: f32, auto_preferred: bool| CssValue::AspectRatio {
            ratio: r,
            auto_preferred,
        };
        // Initial value is `auto` (no preferred ratio).
        let s = style(&doc, div, "div { color: red; }", &vp());
        assert_eq!(s.get("aspect-ratio"), Some(&kw("auto")));
        // `w / h` with and without spaces; a bare number means `n / 1`.
        let s = style(&doc, div, "div { aspect-ratio: 16/9; }", &vp());
        assert_eq!(s.get("aspect-ratio"), Some(&ratio(16.0 / 9.0, false)));
        let s = style(&doc, div, "div { aspect-ratio: 4 / 3; }", &vp());
        assert_eq!(s.get("aspect-ratio"), Some(&ratio(4.0 / 3.0, false)));
        let s = style(&doc, div, "div { aspect-ratio: 2; }", &vp());
        assert_eq!(s.get("aspect-ratio"), Some(&ratio(2.0, false)));
        let s = style(&doc, div, "div { aspect-ratio: auto; }", &vp());
        assert_eq!(s.get("aspect-ratio"), Some(&kw("auto")));
        // `auto <ratio>` (either order): ratio with the auto-preferred flag.
        let s = style(&doc, div, "div { aspect-ratio: auto 16/9; }", &vp());
        assert_eq!(s.get("aspect-ratio"), Some(&ratio(16.0 / 9.0, true)));
        let s = style(&doc, div, "div { aspect-ratio: 16/9 auto; }", &vp());
        assert_eq!(s.get("aspect-ratio"), Some(&ratio(16.0 / 9.0, true)));
    }

    #[test]
    fn aspect_ratio_rejects_invalid() {
        let (doc, div, _) = sample();
        // Zero/negative denominators, negative numerators, doubled
        // slashes, and non-numbers are invalid: the declaration drops
        // and the initial `auto` applies.
        for css in [
            "div { aspect-ratio: 16/0; }",
            "div { aspect-ratio: 0/0; }",
            "div { aspect-ratio: -16/9; }",
            "div { aspect-ratio: 16/-9; }",
            "div { aspect-ratio: 1/2/3; }",
            "div { aspect-ratio: /9; }",
            "div { aspect-ratio: 16/; }",
            "div { aspect-ratio: auto auto; }",
            // `auto` may appear at most once (`auto || <ratio>`).
            "div { aspect-ratio: auto 16/9 auto; }",
            "div { aspect-ratio: banana; }",
            "div { aspect-ratio: 16px/9; }",
        ] {
            let s = style(&doc, div, css, &vp());
            assert_eq!(s.get("aspect-ratio"), Some(&kw("auto")), "{css}");
        }
    }

    #[test]
    fn aspect_ratio_var_substitution() {
        let (doc, div, _) = sample();
        let ratio = |r: f32| CssValue::AspectRatio {
            ratio: r,
            auto_preferred: false,
        };
        // A `var()`-substituted ratio folds after substitution.
        let s = style(
            &doc,
            div,
            "div { --r: 16/9; aspect-ratio: var(--r); }",
            &vp(),
        );
        assert_eq!(s.get("aspect-ratio"), Some(&ratio(16.0 / 9.0)));
        // ...including through a fallback.
        let s = style(
            &doc,
            div,
            "div { aspect-ratio: var(--missing, 4/3); }",
            &vp(),
        );
        assert_eq!(s.get("aspect-ratio"), Some(&ratio(4.0 / 3.0)));
        // An invalid substituted ratio is invalid at computed-value time.
        let s = style(
            &doc,
            div,
            "div { --r: 1/0; aspect-ratio: var(--r); }",
            &vp(),
        );
        assert_eq!(s.get("aspect-ratio"), Some(&kw("auto")));
    }

    #[test]
    fn aspect_ratio_not_inherited() {
        let (doc, _, p) = sample();
        let s = style(&doc, p, "div { aspect-ratio: 16/9; }", &vp());
        assert_eq!(s.get("aspect-ratio"), Some(&kw("auto")));
    }

    #[test]
    fn aspect_ratio_css_wide_keywords() {
        let (doc, _, p) = sample();
        let ratio = CssValue::AspectRatio {
            ratio: 16.0 / 9.0,
            auto_preferred: false,
        };
        // CSS-wide keywords survive the eager aspect-ratio parse and
        // apply in the cascade. `aspect-ratio` is not inherited, so a
        // dropped `inherit` would wrongly compute `auto` instead of the
        // parent's ratio.
        let s = style(
            &doc,
            p,
            "div { aspect-ratio: 16/9; } p { aspect-ratio: inherit; }",
            &vp(),
        );
        assert_eq!(s.get("aspect-ratio"), Some(&ratio));
        for (css, expected) in [
            ("p { aspect-ratio: initial; }", kw("auto")),
            ("p { aspect-ratio: unset; }", kw("auto")),
            // `revert` with no lower origin behaves as `unset`.
            ("p { aspect-ratio: revert; }", kw("auto")),
            ("p { aspect-ratio: revert-layer; }", kw("auto")),
        ] {
            let s = style(&doc, p, css, &vp());
            assert_eq!(s.get("aspect-ratio"), Some(&expected), "{css}");
        }
    }

    #[test]
    fn aspect_ratio_calc() {
        let (doc, div, _) = sample();
        let ratio = |r: f32| CssValue::AspectRatio {
            ratio: r,
            auto_preferred: false,
        };
        // `calc()` is a valid `<number>` (css-sizing-4 §3.1): the eager
        // parse keeps the `Math` value and the cascade evaluates it
        // before folding.
        let s = style(&doc, div, "div { aspect-ratio: calc(2); }", &vp());
        assert_eq!(s.get("aspect-ratio"), Some(&ratio(2.0)));
        let s = style(&doc, div, "div { aspect-ratio: calc(1 + 1); }", &vp());
        assert_eq!(s.get("aspect-ratio"), Some(&ratio(2.0)));
        // ...including through `var()` substitution.
        let s = style(
            &doc,
            div,
            "div { --r: calc(3); aspect-ratio: var(--r); }",
            &vp(),
        );
        assert_eq!(s.get("aspect-ratio"), Some(&ratio(3.0)));
        // Invalid math is invalid at computed-value time: initial `auto`.
        let s = style(&doc, div, "div { aspect-ratio: calc(1/0); }", &vp());
        assert_eq!(s.get("aspect-ratio"), Some(&kw("auto")));
    }

    #[test]
    fn invalid_logical_winner_does_not_corrupt_fold() {
        let (doc, div, _) = sample();
        // A declaration that is invalid at computed-value time behaves
        // as unset: it must not beat a valid counterpart declaration in
        // the logical-to-physical fold, whichever comes later in the
        // cascade.
        let s = style(
            &doc,
            div,
            "div { margin-left: 9px; margin-inline-start: bogus; }",
            &vp(),
        );
        assert_eq!(s.get("margin-left"), Some(&px(9.0)));
        let s = style(
            &doc,
            div,
            "div { margin-inline-start: 20px; margin-left: bogus; }",
            &vp(),
        );
        assert_eq!(s.get("margin-left"), Some(&px(20.0)));
        // `revert` with no lower origin is VALID (`unset` semantics) and
        // keeps its cascade position: the logical winner still wins and
        // the physical slot takes the initial value.
        let s = style(
            &doc,
            div,
            "div { margin-left: 9px; margin-inline-start: revert; }",
            &vp(),
        );
        assert_eq!(s.get("margin-left"), Some(&px(0.0)));
    }

    // ---- style-sharing cache (wave 36 css-perf) ----

    fn red() -> CssValue {
        CssValue::Color(Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        })
    }

    fn blue() -> CssValue {
        CssValue::Color(Color {
            r: 0,
            g: 0,
            b: 255,
            a: 255,
        })
    }

    /// Three identical `<p class="x">` siblings under a div.
    fn sharing_sample() -> (Document, NodeId, Vec<NodeId>) {
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let mut ps = Vec::new();
        for _ in 0..3 {
            let p = doc.create_element("p");
            doc.set_attribute(p, "class", "x");
            doc.append_child(div, p);
            ps.push(p);
        }
        (doc, div, ps)
    }

    #[test]
    fn sharing_identical_siblings_share() {
        let (doc, div, ps) = sharing_sample();
        let author = parse_stylesheet(".x { color: red; } div p { margin-top: 2px; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let _ = styler.style_for(div);
        let styles: Vec<ComputedStyle> = ps.iter().map(|&p| styler.style_for(p)).collect();
        for s in &styles {
            assert_eq!(s.get("color"), Some(&red()));
            assert_eq!(s.get("margin-top"), Some(&px(2.0)));
        }
        // div + first p compute; the other two adopt the cached style.
        assert_eq!(styler.share_hits.get(), 2);
        assert_eq!(styler.share_misses.get(), 2);
    }

    #[test]
    fn sharing_does_not_share_across_differences() {
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let p1 = doc.create_element("p");
        doc.set_attribute(p1, "class", "x");
        doc.append_child(div, p1);
        let p2 = doc.create_element("p");
        doc.set_attribute(p2, "class", "y");
        doc.append_child(div, p2);
        let author = parse_stylesheet(".x { color: red; } .y { color: blue; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let _ = styler.style_for(div);
        let s1 = styler.style_for(p1);
        let s2 = styler.style_for(p2);
        assert_eq!(s1.get("color"), Some(&red()));
        assert_eq!(s2.get("color"), Some(&blue()));
        assert_eq!(styler.share_hits.get(), 0);
    }

    #[test]
    fn sharing_nth_child_uses_position() {
        let mut doc = Document::new();
        let root = doc.root();
        let ul = doc.create_element("ul");
        doc.append_child(root, ul);
        let mut lis = Vec::new();
        for _ in 0..4 {
            let li = doc.create_element("li");
            doc.append_child(ul, li);
            lis.push(li);
        }
        let author = parse_stylesheet("li:nth-child(2n) { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let _ = styler.style_for(ul);
        let styles: Vec<ComputedStyle> = lis.iter().map(|&li| styler.style_for(li)).collect();
        // Positions differ, so nothing shares: every li computes.
        assert_eq!(styler.share_hits.get(), 0);
        assert_eq!(styles[0].get("color"), styles[2].get("color"));
        assert_eq!(styles[1].get("color"), styles[3].get("color"));
        assert_eq!(styles[1].get("color"), Some(&red()));
        assert_ne!(styles[0].get("color"), styles[1].get("color"));
    }

    #[test]
    fn sharing_sibling_combinator_uses_preceding_context() {
        let (doc, div, ps) = sharing_sample();
        let author = parse_stylesheet("p + p { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let _ = styler.style_for(div);
        let styles: Vec<ComputedStyle> = ps.iter().map(|&p| styler.style_for(p)).collect();
        // Preceding-sibling context differs per element: no sharing.
        assert_eq!(styler.share_hits.get(), 0);
        assert_ne!(styles[0].get("color"), Some(&red()));
        assert_eq!(styles[1].get("color"), Some(&red()));
        assert_eq!(styles[2].get("color"), Some(&red()));
    }

    #[test]
    fn sharing_empty_pseudo_uses_emptiness() {
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let p1 = doc.create_element("p");
        doc.append_child(div, p1);
        let p2 = doc.create_element("p");
        doc.append_child(div, p2);
        let p3 = doc.create_element("p");
        let text = doc.create_text_node("x");
        doc.append_child(p3, text);
        doc.append_child(div, p3);
        let author = parse_stylesheet("p:empty { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let _ = styler.style_for(div);
        let s1 = styler.style_for(p1);
        let s2 = styler.style_for(p2);
        let s3 = styler.style_for(p3);
        assert_eq!(s1.get("color"), Some(&red()));
        assert_eq!(s2.get("color"), Some(&red()));
        assert_ne!(s3.get("color"), Some(&red()));
        // The two empty paragraphs share; the non-empty one computes.
        assert_eq!(styler.share_hits.get(), 1);
    }

    #[test]
    fn sharing_disabled_with_has() {
        let mut doc = Document::new();
        let root = doc.root();
        let body = doc.create_element("body");
        doc.append_child(root, body);
        let mut divs = Vec::new();
        for _ in 0..2 {
            let div = doc.create_element("div");
            let p = doc.create_element("p");
            doc.append_child(div, p);
            doc.append_child(body, div);
            divs.push(div);
        }
        let author = parse_stylesheet("div:has(p) { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let styles: Vec<ComputedStyle> = divs.iter().map(|&d| styler.style_for(d)).collect();
        // Correct results, but `:has()` disables the sharing cache.
        for s in &styles {
            assert_eq!(s.get("color"), Some(&red()));
        }
        assert_eq!(styler.share_hits.get(), 0);
    }

    #[test]
    fn sharing_disabled_pseudo_not_poisoned_by_legend_position() {
        // `:disabled`/`:enabled` observe the first-`legend` exception
        // inside a disabled `fieldset`: two identical `<input>`s, one in
        // the first legend (enabled) and one in the second legend
        // (disabled) of disabled fieldsets, must not share a style.
        let mut doc = Document::new();
        let root = doc.root();
        let body = doc.create_element("body");
        doc.append_child(root, body);
        let mut inputs = Vec::new();
        for first in [true, false] {
            let fs = doc.create_element("fieldset");
            doc.set_attribute(fs, "disabled", "");
            let legend = doc.create_element("legend");
            let input = doc.create_element("input");
            doc.append_child(legend, input);
            doc.append_child(fs, legend);
            if !first {
                // A second legend: its input is disabled, unlike the
                // first legend's.
                let legend2 = doc.create_element("legend");
                let input2 = doc.create_element("input");
                doc.append_child(legend2, input2);
                doc.append_child(fs, legend2);
                inputs.push(input2);
            } else {
                inputs.push(input);
            }
            doc.append_child(body, fs);
        }
        let author =
            parse_stylesheet("input:disabled { color: red; } input:enabled { color: blue; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        // Style the enabled (first-legend) input first so a poisoned
        // cache would hand its style to the disabled one.
        let s0 = styler.style_for(inputs[0]);
        let s1 = styler.style_for(inputs[1]);
        assert_eq!(s0.get("color"), Some(&blue()));
        assert_eq!(s1.get("color"), Some(&red()));
    }

    #[test]
    fn sharing_skipped_for_sibling_and_structural_selectors() {
        // Sibling combinators and structural pseudo-classes make every
        // key unique per document (0% hit rate) while still paying the
        // context builds: sharing is skipped outright, like `:has()`.
        let (doc, div, ps) = sharing_sample();
        for css in ["p + p { color: red; }", "p:nth-child(2) { color: red; }"] {
            let author = parse_stylesheet(css);
            let ua_sheet = ua();
            let author_refs = [&author];
            let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
            let _ = styler.style_for(div);
            for &p in &ps {
                let _ = styler.style_for(p);
            }
            assert_eq!(styler.share_hits.get(), 0, "{css}");
            assert!(
                styler.share_cache.borrow().is_empty(),
                "{css}: nothing published"
            );
        }
    }

    #[test]
    fn epoch_bump_drops_share_cache() {
        // The share cache is hash-keyed, so stale generations are never
        // overwritten: every epoch bump must drop it, or each
        // hover/focus change leaks a full generation of entries.
        let (doc, div, ps) = sharing_sample();
        let author = parse_stylesheet(".x { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let _ = styler.style_for(div);
        let _ = styler.style_for(ps[0]);
        let _ = styler.style_for(ps[1]);
        assert!(
            !styler.share_cache.borrow().is_empty(),
            "a generation of entries must exist before the bump"
        );
        styler.set_element_state(
            ps[0],
            ElementState {
                hover: true,
                ..ElementState::default()
            },
        );
        assert!(
            styler.share_cache.borrow().is_empty(),
            "epoch bump must drop the share cache"
        );
    }

    #[test]
    fn sharing_state_change_invalidates() {
        let (doc, div, ps) = sharing_sample();
        let author = parse_stylesheet("p:hover { color: red; } p { color: blue; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let _ = styler.style_for(div);
        let b1 = styler.style_for(ps[0]);
        let b2 = styler.style_for(ps[1]);
        assert_eq!(b1.get("color"), Some(&blue()));
        assert_eq!(b2.get("color"), Some(&blue()));
        assert_eq!(styler.share_hits.get(), 1);
        // Hovering the first paragraph bumps the epoch: its key changes
        // and the old cache entries retire.
        styler.set_element_state(
            ps[0],
            ElementState {
                hover: true,
                ..ElementState::default()
            },
        );
        let h1 = styler.style_for(ps[0]);
        let h2 = styler.style_for(ps[1]);
        assert_eq!(h1.get("color"), Some(&red()));
        assert_eq!(h2.get("color"), Some(&blue()));
        // No cross-contamination: the hovered element never adopts the
        // plain entry and vice versa.
        assert_eq!(styler.share_hits.get(), 1);
    }

    #[test]
    fn sharing_nested_identical_subtrees_share() {
        let mut doc = Document::new();
        let root = doc.root();
        // One shared parent: a document node takes a single document
        // element, so the two subtrees hang off a common wrapper.
        let body = doc.create_element("body");
        doc.append_child(root, body);
        let mut pairs = Vec::new();
        for _ in 0..2 {
            let div = doc.create_element("div");
            doc.set_attribute(div, "class", "w");
            let span = doc.create_element("span");
            doc.set_attribute(span, "class", "x");
            doc.append_child(div, span);
            doc.append_child(body, div);
            pairs.push((div, span));
        }
        let author = parse_stylesheet(".w .x { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        // Root-first: div, span, div, span. The second div shares the
        // first's style, and the second span shares the first's — the
        // parent keys line up through the shared ancestors.
        for (div, span) in &pairs {
            let _ = styler.style_for(*div);
            let s = styler.style_for(*span);
            assert_eq!(s.get("color"), Some(&red()));
        }
        assert_eq!(styler.share_hits.get(), 2);
    }

    #[test]
    fn sharing_inheritance_through_adopted_parent() {
        let mut doc = Document::new();
        let root = doc.root();
        let body = doc.create_element("body");
        doc.append_child(root, body);
        let mut ps = Vec::new();
        for _ in 0..2 {
            let div = doc.create_element("div");
            let p = doc.create_element("p");
            doc.append_child(div, p);
            doc.append_child(body, div);
            ps.push((div, p));
        }
        let author = parse_stylesheet("div { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        // The second paragraph adopts the first's finished style; its
        // inherited `color` must equal what it would inherit from its
        // own (distinct) parent.
        for (div, p) in &ps {
            let _ = styler.style_for(*div);
            let s = styler.style_for(*p);
            assert_eq!(s.get("color"), Some(&red()));
        }
        assert_eq!(styler.share_hits.get(), 2);
    }

    #[test]
    fn sharing_custom_properties_through_adopted_env() {
        let mut doc = Document::new();
        let root = doc.root();
        let body = doc.create_element("body");
        doc.append_child(root, body);
        let mut ps = Vec::new();
        for _ in 0..2 {
            let div = doc.create_element("div");
            let p = doc.create_element("p");
            doc.append_child(div, p);
            doc.append_child(body, div);
            ps.push((div, p));
        }
        let author = parse_stylesheet("div { --x: red; } p { color: var(--x); }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        for (div, p) in &ps {
            let _ = styler.style_for(*div);
            let s = styler.style_for(*p);
            assert_eq!(s.get("color"), Some(&red()));
        }
        assert_eq!(styler.share_hits.get(), 2);
    }

    #[test]
    fn inline_style_parse_cached() {
        let mut doc = Document::new();
        let root = doc.root();
        let body = doc.create_element("body");
        doc.append_child(root, body);
        let d1 = doc.create_element("div");
        doc.set_attribute(d1, "style", "color: red; margin: 1px;");
        doc.append_child(body, d1);
        let d2 = doc.create_element("div");
        doc.set_attribute(d2, "style", "color: red; margin: 1px;");
        doc.append_child(body, d2);
        let d3 = doc.create_element("div");
        doc.set_attribute(d3, "style", "color: blue;");
        doc.append_child(body, d3);
        let author = parse_stylesheet("");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let s1 = styler.style_for(d1);
        let s2 = styler.style_for(d2);
        let s3 = styler.style_for(d3);
        assert_eq!(s1.get("color"), Some(&red()));
        assert_eq!(s2.get("color"), Some(&red()));
        assert_eq!(s3.get("color"), Some(&blue()));
        assert_eq!(s1.get("margin-top"), Some(&px(1.0)));
        // Two distinct `style` strings parsed, not three.
        assert_eq!(styler.inline_parses.get(), 2);
    }

    #[test]
    fn sharing_inheritance_distinguishes_parents() {
        // No combinators anywhere: the parent key must still separate
        // elements whose inherited values differ. (`#a { color: red }`
        // inherits into p1 but not p2.)
        let mut doc = Document::new();
        let root = doc.root();
        let body = doc.create_element("body");
        doc.append_child(root, body);
        let d1 = doc.create_element("div");
        doc.set_attribute(d1, "id", "a");
        doc.append_child(body, d1);
        let d2 = doc.create_element("div");
        doc.set_attribute(d2, "id", "b");
        doc.append_child(body, d2);
        let p1 = doc.create_element("p");
        doc.append_child(d1, p1);
        let p2 = doc.create_element("p");
        doc.append_child(d2, p2);
        let author = parse_stylesheet("#a { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        let _ = styler.style_for(d1);
        let s1 = styler.style_for(p1);
        let _ = styler.style_for(d2);
        let s2 = styler.style_for(p2);
        assert_eq!(s1.get("color"), Some(&red()));
        assert_ne!(s2.get("color"), Some(&red()));
        assert_eq!(styler.share_hits.get(), 0);
    }

    #[test]
    fn sharing_deep_chain_no_overflow() {
        let mut doc = Document::new();
        let mut parent = doc.root();
        for _ in 0..3000 {
            let div = doc.create_element("div");
            doc.append_child(parent, div);
            parent = div;
        }
        let author = parse_stylesheet("div { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        // Cold: only the deepest node is styled. The sharing-key walk
        // and the inheritance climb must both be iterative.
        let s = styler.style_for(parent);
        assert_eq!(s.get("color"), Some(&red()));
    }

    fn shadow_layers<'a>(style: &'a ComputedStyle, prop: &str) -> &'a Vec<Shadow> {
        match style.get(prop) {
            Some(CssValue::Shadows(shadows)) => shadows,
            other => panic!("expected Shadows for {prop}, got {other:?}"),
        }
    }

    fn px_len(v: &CssValue) -> f32 {
        match v {
            CssValue::Length(n, LengthUnit::Px) => *n,
            CssValue::Number(n) => *n,
            _ => panic!("expected px length, got {v:?}"),
        }
    }

    #[test]
    fn box_shadow_computes_to_layers() {
        let (doc, div, _) = sample();
        // `none` computes to the empty layer list.
        let s = style(&doc, div, ".a { box-shadow: none; }", &vp());
        assert_eq!(s.get("box-shadow"), Some(&CssValue::Shadows(Vec::new())));
        // Full layer: offsets, blur, spread, color, inset.
        let s = style(
            &doc,
            div,
            ".a { box-shadow: inset 2px 3px 4px 5px red; }",
            &vp(),
        );
        let layers = shadow_layers(&s, "box-shadow");
        assert_eq!(layers.len(), 1);
        let sh = &layers[0];
        assert!(sh.inset);
        assert_eq!(px_len(&sh.offset_x), 2.0);
        assert_eq!(px_len(&sh.offset_y), 3.0);
        assert_eq!(px_len(&sh.blur), 4.0);
        assert_eq!(px_len(&sh.spread), 5.0);
        assert!(matches!(sh.color, CssValue::Color(_)));
        // Omitted color defaults to `currentcolor`; omitted blur/spread
        // default to zero.
        let s = style(&doc, div, ".a { box-shadow: 1px 2px; }", &vp());
        let sh = &shadow_layers(&s, "box-shadow")[0];
        assert_eq!(sh.color, CssValue::Keyword("currentcolor".into()));
        assert_eq!(px_len(&sh.blur), 0.0);
        assert_eq!(px_len(&sh.spread), 0.0);
        assert!(!sh.inset);
        // Multiple layers stay in source order.
        let s = style(
            &doc,
            div,
            ".a { box-shadow: 1px 1px red, inset 2px 2px 2px blue; }",
            &vp(),
        );
        let layers = shadow_layers(&s, "box-shadow");
        assert_eq!(layers.len(), 2);
        assert!(!layers[0].inset);
        assert!(layers[1].inset);
    }

    #[test]
    fn box_shadow_invalid_falls_back_to_initial() {
        let (doc, div, _) = sample();
        // Negative blur, spread on `text-shadow`, `inset` on
        // `text-shadow`, and bad component counts are all invalid at
        // computed-value time: the property takes its initial value.
        for bad in [
            ".a { box-shadow: 2px 3px -1px red; }",
            ".a { box-shadow: 2px; }",
            ".a { box-shadow: 2px 3px 4px 5px 6px red; }",
            ".a { text-shadow: inset 1px 2px; }",
            ".a { text-shadow: 1px 2px 3px 4px red; }",
        ] {
            let prop = bad.split_whitespace().nth(2).unwrap().trim_end_matches(':');
            let s = style(&doc, div, bad, &vp());
            assert_eq!(
                s.get(prop),
                Some(&CssValue::Shadows(Vec::new())),
                "{bad} should compute to the initial value"
            );
        }
        // `text-shadow` without `inset`/spread computes normally.
        let s = style(&doc, div, ".a { text-shadow: 1px 2px 3px black; }", &vp());
        let sh = &shadow_layers(&s, "text-shadow")[0];
        assert_eq!(px_len(&sh.blur), 3.0);
        assert!(!sh.inset);
    }

    #[test]
    fn shadow_inheritance_rules() {
        let (doc, _, p) = sample();
        // `text-shadow` is inherited; `box-shadow` is not (CSS Text
        // Decoration 3 §4.3 / CSS Backgrounds 3 §7).
        let s = style(
            &doc,
            p,
            "div { text-shadow: 1px 2px black; box-shadow: 3px 4px red; }",
            &vp(),
        );
        let sh = &shadow_layers(&s, "text-shadow")[0];
        assert_eq!(px_len(&sh.offset_x), 1.0);
        assert_eq!(s.get("box-shadow"), Some(&CssValue::Shadows(Vec::new())));
    }

    #[test]
    fn shadow_var_substitution() {
        let (doc, div, _) = sample();
        // A `var()` substituting to one shadow.
        let s = style(
            &doc,
            div,
            ".a { --s: 2px 3px red; box-shadow: var(--s); }",
            &vp(),
        );
        let sh = &shadow_layers(&s, "box-shadow")[0];
        assert_eq!(px_len(&sh.offset_x), 2.0);
        assert!(matches!(sh.color, CssValue::Color(_)));
        // A `var()` substituting to a comma-separated list keeps its
        // groups.
        let s = style(
            &doc,
            div,
            ".a { --s: 1px 1px red, inset 2px 2px blue; box-shadow: var(--s); }",
            &vp(),
        );
        let layers = shadow_layers(&s, "box-shadow");
        assert_eq!(layers.len(), 2);
        assert!(!layers[0].inset);
        assert!(layers[1].inset);
        // A `var()` substituted to `none` computes to no shadow.
        let s = style(&doc, div, ".a { --s: none; box-shadow: var(--s); }", &vp());
        assert_eq!(s.get("box-shadow"), Some(&CssValue::Shadows(Vec::new())));
        // A `var()` for a single component resolves inside the layer.
        let s = style(
            &doc,
            div,
            ".a { --c: red; box-shadow: 2px 3px var(--c); }",
            &vp(),
        );
        let sh = &shadow_layers(&s, "box-shadow")[0];
        assert!(matches!(sh.color, CssValue::Color(_)));
        // An invalid substitution drops the declaration.
        let s = style(&doc, div, ".a { --s: 2px; box-shadow: var(--s); }", &vp());
        assert_eq!(s.get("box-shadow"), Some(&CssValue::Shadows(Vec::new())));
    }

    #[test]
    fn shadow_math_evaluates() {
        let (doc, div, _) = sample();
        // `calc()` in a shadow evaluates against font/viewport metrics
        // at computed-value time.
        let s = style(
            &doc,
            div,
            ".a { box-shadow: calc(1px + 2px) 0 red; }",
            &vp(),
        );
        let sh = &shadow_layers(&s, "box-shadow")[0];
        assert_eq!(sh.offset_x, CssValue::Length(3.0, LengthUnit::Px));
        // Relative units keep their authored form at computed-value time
        // (like `width: 1em`); the consumer resolves them with
        // `length_to_px` against font/viewport metrics.
        let s = style(
            &doc,
            div,
            ".a { font-size: 20px; box-shadow: 1em 0 red; }",
            &vp(),
        );
        let sh = &shadow_layers(&s, "box-shadow")[0];
        assert_eq!(sh.offset_x, CssValue::Length(1.0, LengthUnit::Em));
        // Percentage math has no base in a shadow: invalid at
        // computed-value time.
        let s = style(
            &doc,
            div,
            ".a { box-shadow: calc(10% + 1px) 0 red; }",
            &vp(),
        );
        assert_eq!(s.get("box-shadow"), Some(&CssValue::Shadows(Vec::new())));
        // A negative `calc()` blur is invalid after evaluation.
        let s = style(&doc, div, ".a { box-shadow: 0 0 calc(-1px) red; }", &vp());
        assert_eq!(s.get("box-shadow"), Some(&CssValue::Shadows(Vec::new())));
    }

    #[test]
    fn sizing_accepts_intrinsic_keywords() {
        // CSS Sizing 3 §2: `width`/`min-width`/`height`/`min-height`
        // accept the intrinsic keywords.
        for prop in ["width", "min-width", "height", "min-height"] {
            assert!(is_valid_for_property(prop, &kw("fit-content")), "{prop}");
            assert!(is_valid_for_property(prop, &kw("min-content")), "{prop}");
            assert!(is_valid_for_property(prop, &kw("max-content")), "{prop}");
            assert!(is_valid_for_property(prop, &kw("stretch")), "{prop}");
            assert!(is_valid_for_property(prop, &kw("auto")), "{prop}");
            assert!(!is_valid_for_property(prop, &kw("none")), "{prop}");
        }
        // Logical axes mirror the physical ones.
        for kwd in ["fit-content", "min-content", "max-content"] {
            assert!(is_valid_for_property("inline-size", &kw(kwd)), "{kwd}");
            assert!(is_valid_for_property("block-size", &kw(kwd)), "{kwd}");
        }
        // `max-width`/`max-height` take `none` plus the intrinsic
        // keywords, plus `stretch` (CSS Sizing 3 §5).
        for prop in ["max-width", "max-height"] {
            assert!(is_valid_for_property(prop, &kw("none")), "{prop}");
            for kwd in ["fit-content", "min-content", "max-content", "stretch"] {
                assert!(is_valid_for_property(prop, &kw(kwd)), "{prop} {kwd}");
            }
            assert!(!is_valid_for_property(prop, &kw("auto")), "{prop}");
        }
        // CSS Flexbox 1 §9.2: `flex-basis: content | <'width'>` — the
        // `content` keyword plus the full `width` grammar (including
        // `stretch`, CSS Sizing 3 §5).
        assert!(is_valid_for_property("flex-basis", &kw("content")));
        for kwd in [
            "auto",
            "fit-content",
            "min-content",
            "max-content",
            "stretch",
        ] {
            assert!(is_valid_for_property("flex-basis", &kw(kwd)), "{kwd}");
        }
    }

    #[test]
    fn fill_available_aliases_scoped_to_sizing_grammars() {
        // css-sizing-4 legacy aliases normalize to `stretch` in sizing
        // grammars only (Chrome rejects them for alignment): the
        // mapping lives at the sizing validation arms, never in the
        // generic value parser (table-review finding 3).
        for prop in [
            "width",
            "min-width",
            "height",
            "min-height",
            "max-width",
            "max-height",
            "flex-basis",
        ] {
            for alias in ["-webkit-fill-available", "-moz-available"] {
                assert!(is_valid_for_property(prop, &kw(alias)), "{prop} {alias}");
            }
        }
        // Alignment properties (and `display`'s whitelist) reject the
        // aliases — pre-0a8ec4e behavior restored.
        for prop in [
            "align-self",
            "align-content",
            "justify-self",
            "align-items",
            "justify-content",
            "place-self",
            "display",
        ] {
            for alias in ["-webkit-fill-available", "-moz-available"] {
                assert!(!is_valid_for_property(prop, &kw(alias)), "{prop} {alias}");
            }
        }
    }

    #[test]
    fn fill_available_alias_computes_to_stretch_on_sizing() {
        let (doc, div, _) = sample();
        // The alias computes to the canonical `stretch` keyword so
        // layout's `spec_len_of` fills the containing block (the TWR
        // table-6 fix).
        let s = style(&doc, div, "#d { width: -webkit-fill-available; }", &vp());
        assert_eq!(s.get("width"), Some(&kw("stretch")));
        let s = style(&doc, div, "#d { max-height: -moz-available; }", &vp());
        assert_eq!(s.get("max-height"), Some(&kw("stretch")));
        // On alignment properties the alias is dropped as invalid.
        let s = style(
            &doc,
            div,
            "#d { align-self: -webkit-fill-available; }",
            &vp(),
        );
        assert_eq!(s.get("align-self"), Some(&kw("auto")));
    }

    #[test]
    fn sizing_intrinsic_keywords_reach_computed_style() {
        let (doc, div, _) = sample();
        for kwd in ["fit-content", "min-content", "max-content"] {
            let s = style(&doc, div, &format!("#d {{ width: {kwd}; }}"), &vp());
            assert_eq!(s.get("width"), Some(&kw(kwd)), "{kwd}");
        }
        // `inline-size` folds into `width` after the cascade.
        let s = style(&doc, div, "#d { inline-size: fit-content; }", &vp());
        assert_eq!(s.get("width"), Some(&kw("fit-content")));
        // `height`/`min-height` keep the intrinsic keywords too (CSS
        // Sizing 3 §2); layout resolves them against content
        // (auto-like on the block axis until a measure pass lands).
        for kwd in ["fit-content", "min-content", "max-content"] {
            let s = style(&doc, div, &format!("#d {{ height: {kwd}; }}"), &vp());
            assert_eq!(s.get("height"), Some(&kw(kwd)), "{kwd}");
            let s = style(&doc, div, &format!("#d {{ min-height: {kwd}; }}"), &vp());
            assert_eq!(s.get("min-height"), Some(&kw(kwd)), "{kwd}");
        }
        // `block-size` folds into `height` after the cascade.
        let s = style(&doc, div, "#d { block-size: min-content; }", &vp());
        assert_eq!(s.get("height"), Some(&kw("min-content")));
        // `max-height` keeps `none` and the intrinsic keywords.
        let s = style(&doc, div, "#d { max-height: max-content; }", &vp());
        assert_eq!(s.get("max-height"), Some(&kw("max-content")));
        // `stretch` survives the cascade on the properties CSS Sizing 3
        // §5 allows it on (no longer dropped to the initial value).
        let s = style(&doc, div, "#d { max-width: stretch; }", &vp());
        assert_eq!(s.get("max-width"), Some(&kw("stretch")));
        let s = style(&doc, div, "#d { flex-basis: stretch; }", &vp());
        assert_eq!(s.get("flex-basis"), Some(&kw("stretch")));
    }

    #[test]
    fn justify_content_accepts_left_right() {
        // CSS Box Alignment 3 §4.2: `left`/`right` are valid
        // <content-position> keywords for `justify-content`.
        for kwd in ["left", "right"] {
            assert!(is_valid_for_property("justify-content", &kw(kwd)), "{kwd}");
        }
        let (doc, div, _) = sample();
        let s = style(&doc, div, "div { justify-content: left; }", &vp());
        assert_eq!(s.get("justify-content"), Some(&kw("left")));
        let s = style(&doc, div, "div { justify-content: right; }", &vp());
        assert_eq!(s.get("justify-content"), Some(&kw("right")));
    }

    /// `style_for_shared` returns the memoized `Rc` itself: a repeat
    /// query for the same node is the same allocation (`Rc::ptr_eq`),
    /// and its value equals what `style_for` deep-clones.
    #[test]
    fn style_for_shared_returns_memoized_rc() {
        let (doc, div, _) = sample();
        let author = parse_stylesheet("#d { color: red; }");
        let author_refs = [&author];
        let ua_sheet = ua();
        let viewport = vp();
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, viewport);
        let a = styler.style_for_shared(div);
        let b = styler.style_for_shared(div);
        assert!(
            Rc::ptr_eq(&a, &b),
            "memoized style is shared, not recomputed"
        );
        let owned = styler.style_for(div);
        assert_eq!(
            format!("{a:?}"),
            format!("{owned:?}"),
            "shared and cloned forms agree"
        );
    }

    /// `valign` presentational hints still map after the allocation-free
    /// restructure: mixed-case tags and values match, invalid values and
    /// non-table elements get no hint. An invalid value on a `td` maps
    /// to no hint, so the cell inherits `middle` through the WHATWG
    /// HTML rendering §15.3.8 UA chain (`tbody { vertical-align:
    /// middle; }`, `tr`/`td` inherit it — added w76fix, replacing the
    /// w72 direct `td, th { vertical-align: middle; }` rule, which
    /// gave the same answer here but blocked inheritance from a `tr`
    /// author rule or row `valign` hint) over the `baseline` initial
    /// value — `middle` is the spec-correct result here, not
    /// `baseline` (the `div` case below still shows `baseline`,
    /// since no UA rule applies there).
    #[test]
    fn valign_hint_maps_case_insensitively() {
        let (mut doc, _table, td) = table_doc(&[]);
        doc.set_attribute(td, "valign", "TOP");
        let s = style(&doc, td, "", &vp());
        assert_eq!(s.get_str("vertical-align"), Some("top"));

        let (mut doc, _table, td) = table_doc(&[]);
        doc.set_attribute(td, "valign", "bogus");
        let s = style(&doc, td, "", &vp());
        // No hint for an invalid value: the cell inherits `middle`
        // through the §15.3.8 UA chain (see the doc comment above).
        assert_eq!(s.get_str("vertical-align"), Some("middle"));

        // `valign` on a non-row/section/cell element is not a hint.
        let (mut doc, div, _) = sample();
        doc.set_attribute(div, "valign", "top");
        let s = style(&doc, div, "", &vp());
        assert_eq!(s.get_str("vertical-align"), Some("baseline"));
    }

    #[test]
    fn tr_vertical_align_propagates_to_cells_through_inherit_chain() {
        // WHATWG HTML rendering §15.3.8 UA chain: row groups and
        // direct table rows get `vertical-align: middle`, rows and
        // cells inherit it. A `tr` author rule — or a row/section
        // `valign` hint, which beats the UA rule on the row — must
        // therefore reach a bogus-`valign` cell, where the old direct
        // `td, th { vertical-align: middle; }` UA rule used to win
        // over inheritance and yield `middle`.
        let build = |with_tbody: bool| {
            let mut doc = Document::new();
            let root = doc.root();
            let table = doc.create_element("table");
            doc.append_child(root, table);
            let parent = if with_tbody {
                let tbody = doc.create_element("tbody");
                doc.append_child(table, tbody);
                tbody
            } else {
                table
            };
            let tr = doc.create_element("tr");
            doc.append_child(parent, tr);
            let td = doc.create_element("td");
            doc.set_attribute(td, "valign", "bogus");
            doc.append_child(tr, td);
            (doc, parent, tr, td)
        };
        // Plain table: the chain yields `middle`, via `tbody` with a
        // section parent and via `table > tr` without one.
        for with_tbody in [true, false] {
            let (doc, _parent, _tr, td) = build(with_tbody);
            let s = style(&doc, td, "", &vp());
            assert_eq!(s.get_str("vertical-align"), Some("middle"));
        }
        // Author `tr` rule propagates to the cell.
        let (doc, _parent, _tr, td) = build(true);
        let s = style(&doc, td, "tr { vertical-align: top; }", &vp());
        assert_eq!(s.get_str("vertical-align"), Some("top"));
        // Row `valign` hint propagates to the cell.
        let (mut doc, _parent, tr, td) = build(true);
        doc.set_attribute(tr, "valign", "top");
        let s = style(&doc, td, "", &vp());
        assert_eq!(s.get_str("vertical-align"), Some("top"));
        // Section `valign` hint propagates through the row to the cell.
        let (mut doc, parent, _tr, td) = build(true);
        doc.set_attribute(parent, "valign", "bottom");
        let s = style(&doc, td, "", &vp());
        assert_eq!(s.get_str("vertical-align"), Some("bottom"));
        // A cell's own valid `valign` hint still beats the chain.
        let (doc, _parent, _tr, td) = build(true);
        let mut doc = doc;
        doc.set_attribute(td, "valign", "bottom");
        let s = style(&doc, td, "tr { vertical-align: top; }", &vp());
        assert_eq!(s.get_str("vertical-align"), Some("bottom"));
    }

    #[test]
    fn orphan_cell_dimension_attributes_map_without_ancestor_table() {
        // WHATWG HTML rendering §15.3.8's `td`/`th` dimension
        // mapping is unconditional on the element: a JS-created cell
        // appended outside a table (or styled `display: table-cell`
        // outside one) gets its `width`/`height` hints just like a
        // cell inside a table, while only the table-derived
        // border/padding hints need an ancestor table.
        let mut doc = Document::new();
        let root = doc.root();
        let td = doc.create_element("td");
        doc.set_attribute(td, "width", "120");
        doc.set_attribute(td, "height", "40");
        doc.append_child(root, td);
        let s = style(&doc, td, "", &vp());
        assert_eq!(
            s.get("width"),
            Some(&CssValue::Length(120.0, LengthUnit::Px))
        );
        assert_eq!(
            s.get("height"),
            Some(&CssValue::Length(40.0, LengthUnit::Px))
        );
        // `th` maps identically, including under an author
        // `display: table-cell` rule outside a table.
        let mut doc = Document::new();
        let root = doc.root();
        let th = doc.create_element("th");
        doc.set_attribute(th, "width", "25%");
        doc.append_child(root, th);
        let s = style(&doc, th, "th { display: table-cell; }", &vp());
        assert_eq!(s.get("width"), Some(&CssValue::Percentage(25.0)));
        // Zero is still ignored outside a table.
        let mut doc = Document::new();
        let root = doc.root();
        let td = doc.create_element("td");
        doc.set_attribute(td, "width", "0");
        doc.append_child(root, td);
        let s = style(&doc, td, "", &vp());
        assert_eq!(s.get("width"), Some(&kw("auto")));
    }

    /// The `dir` attribute maps to the CSS `direction` property as a
    /// presentational hint: `ltr`/`rtl` match case-insensitively and
    /// beat UA declarations but lose to author CSS. `auto` (which
    /// needs first-strong bidi resolution the text engine lacks) and
    /// bogus values map nothing.
    #[test]
    fn dir_hint_maps_to_direction() {
        let (mut doc, div, _) = sample();
        doc.set_attribute(div, "dir", "RTL");
        let s = style(&doc, div, "", &vp());
        assert_eq!(s.get_str("direction"), Some("rtl"));

        let (mut doc, div, _) = sample();
        doc.set_attribute(div, "dir", "ltr");
        let s = style(&doc, div, "", &vp());
        assert_eq!(s.get_str("direction"), Some("ltr"));

        // An author declaration beats the hint.
        let (mut doc, div, _) = sample();
        doc.set_attribute(div, "dir", "rtl");
        let s = style(&doc, div, "div { direction: ltr; }", &vp());
        assert_eq!(s.get_str("direction"), Some("ltr"));

        // `auto` maps nothing yet (no bidi resolution): the initial
        // `ltr` wins.
        let (mut doc, div, _) = sample();
        doc.set_attribute(div, "dir", "auto");
        let s = style(&doc, div, "", &vp());
        assert_eq!(s.get_str("direction"), Some("ltr"));

        // A bogus value maps nothing.
        let (mut doc, div, _) = sample();
        doc.set_attribute(div, "dir", "sideways");
        let s = style(&doc, div, "", &vp());
        assert_eq!(s.get_str("direction"), Some("ltr"));
    }

    #[test]
    fn shadow_child_inherits_from_host() {
        // Per spec, shadow content inherits from the host element, not
        // the unlinked shadow-root fragment: a shadow child with no
        // explicit styles picks up the host's inherited properties.
        let mut doc = Document::new();
        let root = doc.root();
        let host = doc.create_element("div");
        doc.set_attribute(host, "id", "host");
        doc.append_child(root, host);
        let frag = doc
            .attach_shadow_root(host, bowser_dom::ShadowRootMode::Open)
            .expect("attach shadow root");
        let kid = doc.create_element("span");
        doc.set_attribute(kid, "id", "kid");
        doc.append_child(frag, kid);
        let css = "#host { color: red; font-size: 20px; }";
        let s = style(&doc, kid, css, &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })),
            "shadow child inherits color from the host"
        );
        assert_eq!(
            s.get("font-size"),
            Some(&CssValue::Length(20.0, LengthUnit::Px)),
            "shadow child inherits font-size from the host"
        );
    }

    /// Helper: a `<div id="host">` with an open shadow root holding a
    /// `<span id="kid">`. Returns (doc, host, kid).
    fn shadow_doc() -> (Document, NodeId, NodeId) {
        let mut doc = Document::new();
        let root = doc.root();
        let host = doc.create_element("div");
        doc.set_attribute(host, "id", "host");
        doc.append_child(root, host);
        let frag = doc
            .attach_shadow_root(host, bowser_dom::ShadowRootMode::Open)
            .expect("attach shadow root");
        let kid = doc.create_element("span");
        doc.set_attribute(kid, "id", "kid");
        doc.append_child(frag, kid);
        (doc, host, kid)
    }

    #[test]
    fn shadow_child_var_reads_host_custom_property() {
        // `var(--brand)` on a shadow child resolves against the host's
        // custom properties: the custom-property environment crosses the
        // shadow boundary via `style_parent`.
        let (doc, host, kid) = shadow_doc();
        let css = "#host { --brand: red; } span { color: var(--brand); }";
        let kid_style = style(&doc, kid, css, &vp());
        assert_eq!(
            kid_style.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })),
            "var(--brand) on a shadow child resolves from the host"
        );
        let host_style = style(&doc, host, css, &vp());
        assert_eq!(
            kid_style.get("--brand"),
            host_style.get("--brand"),
            "shadow child sees the host's --brand value"
        );
    }

    #[test]
    fn shadow_child_own_custom_property_wins() {
        // A shadow child's own declaration of a custom property beats
        // the host's: the child's winners merge over the host env.
        let (doc, _host, kid) = shadow_doc();
        let css = "#host { --brand: red; } #kid { --brand: blue; } span { color: var(--brand); }";
        let s = style(&doc, kid, css, &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            })),
            "the shadow child's own --brand overrides the host's"
        );
    }

    #[test]
    fn shadow_grandchild_inherits_through_shadow_parent() {
        // The environment merges down the shadow tree: a grandchild with
        // no declaration of its own still sees the host's property.
        let (mut doc, _host, kid) = shadow_doc();
        let leaf = doc.create_element("span");
        doc.append_child(kid, leaf);
        let css = "#host { --brand: red; } span { color: var(--brand); }";
        let s = style(&doc, leaf, css, &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })),
            "shadow grandchild inherits --brand through the shadow parent"
        );
    }

    #[test]
    fn shadow_child_custom_property_inherit_reads_host() {
        // `--x: inherit` on a shadow child reads the host's value.
        // Before the `style_parent` routing in `custom_env` this
        // degraded to guaranteed-invalid: the walk ended at the unlinked
        // fragment, so the merged ancestors map held no `--x`.
        let (doc, host, kid) = shadow_doc();
        let css = "#host { --x: purple; } #kid { --x: inherit; }";
        let kid_style = style(&doc, kid, css, &vp());
        let host_style = style(&doc, host, css, &vp());
        let v = kid_style.get("--x").expect("--x is set on the kid");
        assert!(
            !is_guaranteed_invalid(v),
            "--x: inherit must not be guaranteed-invalid on a shadow child"
        );
        assert_eq!(
            v,
            host_style.get("--x").expect("--x is set on the host"),
            "--x: inherit on a shadow child reads the host's value"
        );
    }

    #[test]
    fn shadow_style_sharing_not_poisoned_across_hosts() {
        // Two hosts with different `--x` values and structurally
        // identical shadow subtrees: the sharing key folds the host's
        // full context via `style_parent`, so the second shadow child
        // must not adopt the first child's cached env.
        let mut doc = Document::new();
        let root = doc.root();
        let mut kids = Vec::new();
        for id in ["h1", "h2"] {
            let host = doc.create_element("div");
            doc.set_attribute(host, "id", id);
            doc.append_child(root, host);
            let frag = doc
                .attach_shadow_root(host, bowser_dom::ShadowRootMode::Open)
                .expect("attach shadow root");
            let kid = doc.create_element("span");
            doc.set_attribute(kid, "id", "kid");
            doc.append_child(frag, kid);
            kids.push(kid);
        }
        let css = "#h1 { --x: red; } #h2 { --x: blue; } span { color: var(--x); }";
        let author = parse_stylesheet(css);
        let author_refs = [&author];
        let ua_sheet = ua();
        let styler = Styler::new(&doc, &author_refs, &ua_sheet, vp());
        assert_eq!(
            styler.style_for(kids[0]).get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })),
        );
        assert_eq!(
            styler.style_for(kids[1]).get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 255
            })),
            "second shadow child resolves --x from its own host, not the first child's donor env"
        );
    }

    #[test]
    fn nested_shadow_roots_custom_property_composes() {
        // Host-in-shadow-tree: the leaf's environment merges the inner
        // host chain and the outer host, composing across nesting.
        let mut doc = Document::new();
        let root = doc.root();
        let outer = doc.create_element("div");
        doc.set_attribute(outer, "id", "outer");
        doc.append_child(root, outer);
        let frag1 = doc
            .attach_shadow_root(outer, bowser_dom::ShadowRootMode::Open)
            .expect("attach outer shadow root");
        let inner = doc.create_element("div");
        doc.set_attribute(inner, "id", "inner");
        doc.append_child(frag1, inner);
        let frag2 = doc
            .attach_shadow_root(inner, bowser_dom::ShadowRootMode::Open)
            .expect("attach inner shadow root");
        let leaf = doc.create_element("span");
        doc.append_child(frag2, leaf);
        let css = "#outer { --brand: red; } span { color: var(--brand); }";
        let s = style(&doc, leaf, css, &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })),
            "nested shadow leaf resolves --brand from the outer host"
        );
    }

    #[test]
    fn shadow_var_fallback_still_works() {
        // Unresolvable `var()` with a fallback stays valid inside
        // shadow trees.
        let (doc, _host, kid) = shadow_doc();
        let s = style(&doc, kid, "span { color: var(--missing, green); }", &vp());
        assert_eq!(
            s.get("color"),
            Some(&CssValue::Color(Color {
                r: 0,
                g: 128,
                b: 0,
                a: 255
            })),
            "var() fallback resolves inside shadow trees"
        );
    }
}

/// Wave 65 perf-builder tests: cross-node candidate-vec sharing by
/// style-sharing key, and `Rc`-shared `var()` memo payloads.
#[cfg(test)]
mod w65_p_cascade_tests {
    use super::*;
    use crate::parse::parse_stylesheet;

    fn ua() -> Stylesheet {
        default_ua_stylesheet()
    }

    fn vp() -> Viewport {
        Viewport { w: 800.0, h: 600.0 }
    }

    fn two_paragraphs() -> (Document, NodeId, NodeId) {
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        let p1 = doc.create_element("p");
        doc.set_attribute(p1, "class", "x");
        doc.append_child(div, p1);
        let p2 = doc.create_element("p");
        doc.set_attribute(p2, "class", "x");
        doc.append_child(div, p2);
        (doc, p1, p2)
    }

    fn styler_for<'x>(
        doc: &'x Document,
        author_refs: &'x [&'x Stylesheet],
        ua_sheet: &'x Stylesheet,
    ) -> Styler<'x> {
        Styler::new(doc, author_refs, ua_sheet, vp())
    }

    #[test]
    fn candidate_vec_shared_by_sharing_key() {
        let (doc, p1, p2) = two_paragraphs();
        let author = parse_stylesheet("p.x { color: red; } div p { margin-top: 2px; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = styler_for(&doc, &author_refs, &ua_sheet);
        let c1 = styler.candidates(p1);
        assert_eq!(styler.cand_key_hits.get(), 0, "first build is a miss");
        assert!(!c1.is_empty(), "fixture must match some rules");
        let c2 = styler.candidates(p2);
        assert_eq!(styler.cand_key_hits.get(), 1, "second build shares by key");
        assert!(
            Rc::ptr_eq(&c1, &c2),
            "one Rc<Vec<Candidate>> for both nodes"
        );
        // Memo semantics unchanged: the per-node slab still records both.
        let epoch = styler.epoch.get();
        assert!(styler.cands.borrow().probe(p1, epoch).is_some());
        assert!(styler.cands.borrow().probe(p2, epoch).is_some());
        // And the normal style path still agrees on both nodes.
        let s1 = styler.style_for(p1);
        let s2 = styler.style_for(p2);
        assert_eq!(s1.get("color"), s2.get("color"));
        assert_eq!(s1.get("margin-top"), s2.get("margin-top"));
    }

    #[test]
    fn candidate_sharing_disabled_with_has() {
        // `:has()` disables the sharing key: both nodes build their own
        // vec, results stay correct, and the key cache stays out of it.
        let (doc, p1, p2) = two_paragraphs();
        let author = parse_stylesheet("div:has(p) { color: red; } p.x { margin-top: 2px; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = styler_for(&doc, &author_refs, &ua_sheet);
        let c1 = styler.candidates(p1);
        let c2 = styler.candidates(p2);
        assert_eq!(styler.cand_key_hits.get(), 0, "no key, no sharing");
        assert!(!Rc::ptr_eq(&c1, &c2));
        assert_eq!(c1.len(), c2.len());
        let s1 = styler.style_for(p1);
        let s2 = styler.style_for(p2);
        assert_eq!(s1.get("margin-top"), s2.get("margin-top"));
    }

    #[test]
    fn candidate_key_cache_cleared_on_epoch_bump() {
        let (doc, p1, p2) = two_paragraphs();
        let author = parse_stylesheet("p.x { color: red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = styler_for(&doc, &author_refs, &ua_sheet);
        let _ = styler.candidates(p1);
        let _ = styler.candidates(p2);
        assert_eq!(styler.cand_key_hits.get(), 1);
        // A state change bumps the epoch and drops the keyed cache: the
        // next build is a miss again, then sharing resumes.
        styler.set_element_state(
            p1,
            ElementState {
                hover: true,
                ..ElementState::default()
            },
        );
        let c3 = styler.candidates(p1);
        assert_eq!(styler.cand_key_hits.get(), 1, "cache was cleared");
        let c4 = styler.candidates(p2);
        assert_eq!(styler.cand_key_hits.get(), 2, "sharing resumes");
        assert!(Rc::ptr_eq(&c3, &c4));
        assert!(!c3.is_empty());
    }

    fn memo_id(styler: &Styler<'_>, name: &str) -> u32 {
        styler.var_names.borrow_mut().intern(name)
    }

    #[test]
    fn var_memo_slash_payload_is_rc_shared() {
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        doc.set_attribute(
            div,
            "style",
            "--x: 12px/1.5; margin-top: var(--x); margin-bottom: var(--x);",
        );
        let author = parse_stylesheet("");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = styler_for(&doc, &author_refs, &ua_sheet);
        let _ = styler.style_for(div);
        // The second `var(--x)` resolved through the memo; both probes
        // must hand out the same allocation — no per-reference clone of
        // the group `Vec`s.
        let id = memo_id(&styler, "--x");
        let m1 = styler.var_memo_probe(div, id, false).expect("--x memoized");
        let m2 = styler.var_memo_probe(div, id, false).expect("--x memoized");
        // Both probes hand out the same slash payload allocation: the
        // memo clone is a pointer bump, not a per-reference deep clone
        // of the group `Vec`s.
        match (&m1, &m2) {
            (MemoVar::Slash(s1), MemoVar::Slash(s2)) => {
                assert!(Rc::ptr_eq(s1, s2), "slash payload shared, not cloned")
            }
            _ => panic!("expected slash-marked memo for --x"),
        }
    }

    #[test]
    fn var_memo_comma_payload_is_rc_shared() {
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        doc.set_attribute(
            div,
            "style",
            "--y: 1px, 2px; margin-top: var(--y); margin-bottom: var(--y);",
        );
        let author = parse_stylesheet("");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = styler_for(&doc, &author_refs, &ua_sheet);
        let _ = styler.style_for(div);
        let id = memo_id(&styler, "--y");
        let m1 = styler.var_memo_probe(div, id, false).expect("--y memoized");
        let m2 = styler.var_memo_probe(div, id, false).expect("--y memoized");
        // Both probes hand out the same comma payload allocation: the
        // memo clone is a pointer bump, not a per-reference deep clone
        // of the group `Vec`s.
        match (&m1, &m2) {
            (MemoVar::Comma(g1), MemoVar::Comma(g2)) => {
                assert!(Rc::ptr_eq(g1, g2), "comma payload shared, not cloned")
            }
            _ => panic!("expected comma-marked memo for --y"),
        }
    }

    #[test]
    fn var_resolution_results_unchanged() {
        // Behavior pin: slash/comma `var()` substitutions resolve exactly
        // as before the `Rc` sharing.
        let mut doc = Document::new();
        let root = doc.root();
        let div = doc.create_element("div");
        doc.append_child(root, div);
        doc.set_attribute(div, "style", "--w: 2px;");
        let author = parse_stylesheet("div { border: var(--w) solid red; }");
        let ua_sheet = ua();
        let author_refs = [&author];
        let styler = styler_for(&doc, &author_refs, &ua_sheet);
        let s = styler.style_for(div);
        assert_eq!(
            s.get("border-top-width"),
            Some(&CssValue::Length(2.0, crate::value::LengthUnit::Px))
        );
        assert_eq!(
            s.get("border-top-style"),
            Some(&CssValue::Keyword("solid".into()))
        );
    }
}
