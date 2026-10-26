//! Style value resolution: from `bowser_css::CssValue` to used numbers.
//!
//! Assumptions about the CSS implementation (documented for review):
//! - Shorthands (`margin`, `padding`, `border-width`, `border`) are expanded
//!   to longhands by the CSS crate; only longhands are read here.
//! - `ComputedStyle::get` returns the cascaded value for the element; no
//!   inheritance is expected from it (font-size inheritance is handled by
//!   the caller passing the parent font size down).

use crate::FontMetrics;
use bowser_css::{CalcSizeBasis, ComputedStyle, CssValue, LengthUnit, Styler, length_to_px, props};
use bowser_dom::NodeId;
use std::rc::Rc;

/// Fallback root font size when the root element specifies nothing usable.
pub(crate) const DEFAULT_ROOT_FONT_SIZE: f32 = 16.0;

/// Largest finite used length, in px. Resolved lengths clamp to
/// ±this so hostile inputs (`1e30px`, `calc()` overflow, `Infinity` from
/// the parser) stay finite and their sums cannot overflow `f32` to
/// infinity. 1e15 px is ~1e13 m at 96 CSS px/in: far beyond any physical
/// display, far below `f32::MAX` (~3.4e38).
pub(crate) const MAX_LAYOUT_PX: f32 = 1e15;

/// Map any non-finite or out-of-range length to a finite in-range one:
/// NaN -> 0.0, ±infinity -> ±[`MAX_LAYOUT_PX`], |v| > MAX -> ±MAX.
//
// Public so paint's display-list builder resolves through the same
// finite clamp layout uses (a hostile `border-width: 1e30px` must not
// paint at 1e30px/inf when layout clamped it to 1e15px).
// `clippy::manual_clamp` is allowed: `f32::clamp` deliberately does not
// fix NaN, so the explicit branch is the whole point.
#[allow(clippy::manual_clamp)]
pub fn sane_len(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else if v > MAX_LAYOUT_PX {
        MAX_LAYOUT_PX
    } else if v < -MAX_LAYOUT_PX {
        -MAX_LAYOUT_PX
    } else {
        v
    }
}

/// Style-resolution swap point (wave B): the single place layout asks CSS
/// for an element's style.
///
/// Takes the document-wide [`Styler`]: cascade and inheritance memoization
/// persist across elements, so styling the whole document is linear in its
/// size instead of quadratic in its depth. The `Styler` is built once per
/// [`crate::layout`] call (in `boxes::build_box_tree`) from the real
/// cascade's `viewport`, so `@media` queries keep working; the UA
/// the styler — it is never re-parsed per element or per layout call.
///
/// Returns the memoized style shared by reference count: one refcount
/// bump per element per style pass instead of a deep clone of the style's
/// delta entries. The `Rc` is threaded through the box arena into the
/// finished [`crate::LayoutBox`] tree.
pub(crate) fn compute_element_style(styler: &Styler, node: NodeId) -> Rc<ComputedStyle> {
    styler.style_for_shared(node)
}

/// Resolved `display` relevant to layout. `table` and the table-internal
/// values get dedicated table layout in `layout::layout_table`; `flex`
/// gets flexbox; `grid` gets grid layout in `layout::layout_grid`; the
/// inline-level outer variants (`inline-flex`, `inline-grid`,
/// `inline-table`) keep their inner layout and become atomic inline-level
/// boxes; everything else (`list-item`, …) is bucketed into block or
/// inline-block; see [`display_of`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Display {
    None,
    Block,
    Inline,
    InlineBlock,
    /// `display: inline-flex`: inline-level outer box (an atomic
    /// [`BoxType::InlineBlock`](crate::BoxType::InlineBlock) in the box
    /// tree) with the flexbox inner layout. The raw keyword is still
    /// consulted by `layout::flex_kind` for the block/inline-level
    /// distinction.
    InlineFlex,
    /// `display: inline-grid`: inline-level outer box (atomic
    /// `InlineBlock`) with the grid inner layout (see
    /// `layout::layout_grid`).
    InlineGrid,
    /// `display: table`: block-level table wrapper (see
    /// `layout::layout_table`).
    Table,
    /// `display: inline-table`: inline-level outer box (atomic
    /// `InlineBlock`) with the table inner layout (see
    /// `layout::layout_table`).
    InlineTable,
    /// `display: grid`: block-level grid container (see
    /// `layout::layout_grid`).
    Grid,
    /// `table-row-group`.
    TableRowGroup,
    /// `table-header-group`: behaves like a `<thead>` element (CSS 2.1
    /// §17.2.1). The computed display wins over the element's tag when
    /// stamping [`SectionKind`]: row-group boxes with this display
    /// stamp [`SectionKind::Header`](crate::boxes::SectionKind::Header),
    /// even for a `<tfoot>` restyled to it.
    TableHeaderGroup,
    /// `table-footer-group`: behaves like a `<tfoot>` element (CSS 2.1
    /// §17.2.1). The computed display wins over the element's tag when
    /// stamping [`SectionKind`]: row-group boxes with this display
    /// stamp [`SectionKind::Footer`](crate::boxes::SectionKind::Footer),
    /// even for a `<thead>` restyled to it.
    TableFooterGroup,
    /// `table-row`.
    TableRow,
    /// `table-cell`.
    TableCell,
    /// `table-column` / `table-column-group`: generate column boxes
    /// (see `layout::layout_table`); their specified widths feed the
    /// table layout algorithm and their borders participate in
    /// collapsed-border conflict resolution.
    TableColumn,
    TableColumnGroup,
    /// `display: contents`: generates no box; children hoist to the parent.
    Contents,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WhiteSpace {
    Normal,
    Pre,
    /// `pre-wrap`: spaces and newlines are preserved like `pre`, but
    /// lines also wrap at break opportunities (spaces), with a
    /// break-all fallback for words longer than the line (see
    /// `boxes::wrap_pre_text`).
    PreWrap,
    /// `pre-line`: sequences of spaces/tabs collapse to a single space
    /// (see `boxes::collapse_ws`), newlines are preserved as forced
    /// breaks, and lines soft-wrap at spaces (see
    /// `boxes::wrap_pre_line_text`). Unlike `pre-wrap`, spaces at line
    /// boundaries are trimmed.
    PreLine,
    /// `break-spaces`: like `pre-wrap` (CSS text-3 §4.1.2) — spaces and
    /// newlines are preserved and lines wrap at spaces — except every
    /// preserved space takes up space (including trailing spaces at
    /// end-of-line, which are never trimmed) and each space is its own
    /// wrap opportunity. Shares the `pre-wrap` machinery (see
    /// `boxes::wrap_pre_text`); the remaining spec delta (intrinsic-size
    /// accounting of trailing spaces) is approximated by measuring pieces
    /// verbatim.
    BreakSpaces,
    NoWrap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TextAlign {
    Left,
    Center,
    Right,
    Justify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VAlign {
    Baseline,
    Top,
    Bottom,
    Middle,
    Sub,
    Super,
}

/// Resolved `text-transform`. It is an inherited property in CSS, so the
/// value a text node transforms by is its containing element's value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TextTransform {
    None,
    Uppercase,
    Lowercase,
    Capitalize,
}

pub(crate) fn text_transform_of(style: &ComputedStyle) -> TextTransform {
    match style.get_str_by_index(props::TEXT_TRANSFORM) {
        Some(v) if v.eq_ignore_ascii_case("uppercase") => TextTransform::Uppercase,
        Some(v) if v.eq_ignore_ascii_case("lowercase") => TextTransform::Lowercase,
        Some(v) if v.eq_ignore_ascii_case("capitalize") => TextTransform::Capitalize,
        _ => TextTransform::None,
    }
}

/// A specified length: either `auto` or a value needing a base to resolve.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SpecLen {
    Auto,
    /// css-sizing-3 §5: `min(max-content, max(min-content, available))`.
    FitContent,
    /// css-sizing-3 §5: the min-content size.
    MinContent,
    /// css-sizing-3 §5: the max-content size.
    MaxContent,
    /// css-sizing-3 §5: `stretch` — fill the containing block. Not
    /// statically resolvable (the containing-block size is only known
    /// at used-value time), so [`SpecLen::resolve`] returns `None`
    /// and the used-value sites handle it (`specified_height` in
    /// layout.rs fills the CB height; block-table width sites now fill
    /// the CB after the fill-available pass, remaining width sites
    /// treat it as `auto`).
    Stretch,
    Px(f32),
    Em(f32),
    Rem(f32),
    Vw(f32),
    Vh(f32),
    Vmin(f32),
    Vmax(f32),
    Pct(f32),
    /// Deferred `calc()` with percentages, `value = pct / 100 * pct_base + px`.
    /// Two forms (see [`CssValue::MathDeferred`]): the affine form carries
    /// true coefficients — the cascade verified affinity and folded every
    /// font/viewport-relative leaf into the constant — so layout only
    /// plugs in the real containing-block base here, at used-value time
    /// (A2 split: the cascade cannot know the base; see
    /// `bowser_css::Styler::eval_math_for_prop`). The non-affine form
    /// (`min()`/`max()`/`clamp()` with `%`) freezes as `pct: 0.0` with
    /// the base-0 approximation in `px`; resolving the coefficients
    /// against any base reproduces that historical value. The `affine`
    /// bit tells the two forms apart: `false` means the coefficients
    /// are a frozen base-0 approximation, not a true coefficient pair,
    /// so consumers that must not treat it as a percentage participant
    /// (table column sizing) gate on it. Stays `Copy`: two `f32`s and
    /// two `bool`s.
    Math {
        pct: f32,
        px: f32,
        /// Whether the original `calc()` tree syntactically contained a `%`
        /// leaf. Consumers deciding cyclic-ness (CSS Values 4 §8.1.4) must
        /// check this bit, not `pct != 0.0`: `calc(10px + 0%)` folds to
        /// `pct: 0.0, px: 10.0` but behaves as `auto` against an indefinite
        /// base. Copied through from `CssValue::MathDeferred::had_pct`.
        had_pct: bool,
        /// Whether the `pct`/`px` pair is a genuine affine coefficient
        /// pair (the cascade verified affinity) as opposed to the frozen
        /// base-0 approximation of non-affine percentage math. Copied
        /// through from `CssValue::MathDeferred::affine`.
        affine: bool,
    },
}

impl SpecLen {
    /// Resolve against the given bases. `pct_base` is the containing-block
    /// width for most properties (CSS resolves even vertical
    /// margin/padding percentages against the containing-block *width*).
    /// `Math` evaluates the cascade-deferred coefficients against that
    /// same base.
    pub(crate) fn resolve(
        self,
        font_size: f32,
        root_font_size: f32,
        vp_w: f32,
        vp_h: f32,
        pct_base: f32,
    ) -> Option<f32> {
        let raw = match self {
            SpecLen::Auto => return None,
            // Intrinsic keywords are not statically resolvable: they
            // measure content at used-value time, so they behave like
            // `auto` here (unresolvable against a base). `stretch`
            // fills the containing block, whose size the used-value
            // site supplies — also unresolvable here.
            SpecLen::FitContent | SpecLen::MinContent | SpecLen::MaxContent | SpecLen::Stretch => {
                return None;
            }
            SpecLen::Pct(v) => v * pct_base / 100.0,
            SpecLen::Math { pct, px, .. } => pct * pct_base / 100.0 + px,
            SpecLen::Px(v) => {
                length_to_px(v, &LengthUnit::Px, font_size, root_font_size, vp_w, vp_h)
            }
            SpecLen::Em(v) => {
                length_to_px(v, &LengthUnit::Em, font_size, root_font_size, vp_w, vp_h)
            }
            SpecLen::Rem(v) => {
                length_to_px(v, &LengthUnit::Rem, font_size, root_font_size, vp_w, vp_h)
            }
            SpecLen::Vw(v) => {
                length_to_px(v, &LengthUnit::Vw, font_size, root_font_size, vp_w, vp_h)
            }
            SpecLen::Vh(v) => {
                length_to_px(v, &LengthUnit::Vh, font_size, root_font_size, vp_w, vp_h)
            }
            SpecLen::Vmin(v) => {
                length_to_px(v, &LengthUnit::Vmin, font_size, root_font_size, vp_w, vp_h)
            }
            SpecLen::Vmax(v) => {
                length_to_px(v, &LengthUnit::Vmax, font_size, root_font_size, vp_w, vp_h)
            }
        };
        // Hostile values (1e30px, calc() overflow, parser infinities)
        // clamp to a finite bound instead of poisoning layout with
        // infinities. Sign is preserved: negative margins are legal.
        Some(sane_len(raw))
    }
}

/// A specified margin: `auto` is meaningful (centering), unlike padding.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SpecMargin {
    Auto,
    Len(SpecLen),
}

/// Read a length-ish property: missing/`auto`/unrecognized -> `Auto`.
pub(crate) fn spec_len_of(value: Option<&CssValue>) -> SpecLen {
    match value {
        None => SpecLen::Auto,
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("auto") => SpecLen::Auto,
        // css-sizing-3 intrinsic keywords (validated by the cascade for
        // `width`/`min-width`/`height`/`min-height`/`inline-size`/
        // `block-size`; unreachable elsewhere).
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("fit-content") => SpecLen::FitContent,
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("min-content") => SpecLen::MinContent,
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("max-content") => SpecLen::MaxContent,
        // css-sizing-3 §5 `stretch` (validated by the cascade for the
        // same sizing properties).
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("stretch") => SpecLen::Stretch,
        Some(CssValue::Length(v, unit)) => match unit {
            LengthUnit::Px => SpecLen::Px(*v),
            // Absolute units: fold to px here (no font/viewport context
            // needed); CSS Values 4 §5.2 anchors `96px = 1in`.
            LengthUnit::Cm => SpecLen::Px(*v * 96.0 / 2.54),
            LengthUnit::Mm => SpecLen::Px(*v * 96.0 / 25.4),
            LengthUnit::In => SpecLen::Px(*v * 96.0),
            LengthUnit::Pt => SpecLen::Px(*v * 96.0 / 72.0),
            LengthUnit::Pc => SpecLen::Px(*v * 16.0),
            LengthUnit::Q => SpecLen::Px(*v * 96.0 / 101.6),
            LengthUnit::Em => SpecLen::Em(*v),
            LengthUnit::Rem => SpecLen::Rem(*v),
            // `ex` is the x-height; without font metrics, 0.5em is the
            // CSS-specified fallback.
            LengthUnit::Ex => SpecLen::Em(*v * 0.5),
            LengthUnit::Vw => SpecLen::Vw(*v),
            LengthUnit::Vh => SpecLen::Vh(*v),
            LengthUnit::Vmin => SpecLen::Vmin(*v),
            LengthUnit::Vmax => SpecLen::Vmax(*v),
        },
        Some(CssValue::Percentage(v)) => SpecLen::Pct(*v),
        // Layout-deferred percentage math (A2): the cascade left either
        // affine coefficients or the frozen base-0 form (see
        // `SpecLen::Math`); both resolve against the real
        // containing-block base in `SpecLen::resolve`.
        Some(CssValue::MathDeferred {
            pct,
            px,
            had_pct,
            affine,
        }) => SpecLen::Math {
            pct: *pct,
            px: *px,
            had_pct: *had_pct,
            affine: *affine,
        },
        // Unitless numbers are invalid in CSS except for zero; be lenient
        // and treat them as px.
        Some(CssValue::Number(v)) => SpecLen::Px(*v),
        // `calc-size()` with a keyword basis (CSS Values 4 §7; H1): the
        // cascade evaluates definite bases itself and leaves keyword
        // bases (`auto`, `max-content`, `min-content`, `fit-content`,
        // `stretch`) for layout, which reads them through
        // `CssValue::calc_size_basis`. Map each to the intrinsic
        // `SpecLen` the existing used-width arms already measure, so the
        // basis is content-measured instead of silently degrading to
        // `auto` (observably wrong for floats, abspos, and flex items).
        // The calc-sum's `size` interpolation is applied at used-value
        // time once the basis is measured (H1 wiring, still open: the
        // used-width sites in `layout.rs` must re-read the specified
        // value, measure the basis with the existing intrinsic arms,
        // then `substitute_size` + `MathExpr::evaluate`); until that
        // wiring lands the basis alone governs.
        //
        // The keyword mapping keeps the fixed-mode table trigger
        // (`table_is_fixed`) correct without touching it: `auto` and
        // `max-content` bases stay auto-ish (never fixed),
        // `min-content`/`fit-content` keep fixed mode, exactly like the
        // bare keywords. `stretch` has no `SpecLen` form; `auto` fill is
        // its block-level equivalent. A definite basis that survived the
        // cascade (unevaluable math) keeps the historical defensive
        // `Auto`.
        Some(v) if v.calc_size_basis().is_some() => {
            match v.calc_size_basis().and_then(CalcSizeBasis::keyword) {
                Some("auto") | Some("stretch") => SpecLen::Auto,
                Some("max-content") => SpecLen::MaxContent,
                Some("min-content") => SpecLen::MinContent,
                Some("fit-content") => SpecLen::FitContent,
                // Definite bases (`<length>`, `<percentage>`, math) and
                // unsubstituted `var()` report no keyword.
                _ => SpecLen::Auto,
            }
        }
        // A raw `Math` never survives the cascade (`compute_prop`
        // evaluates or defers every one); `Auto` is defensive.
        _ => SpecLen::Auto,
    }
}

/// Read a margin property: missing -> 0 (the CSS initial value); only an
/// explicit `auto` keyword stays `Auto`.
pub(crate) fn spec_margin_of(value: Option<&CssValue>) -> SpecMargin {
    match value {
        None => SpecMargin::Len(SpecLen::Px(0.0)),
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("auto") => SpecMargin::Auto,
        _ => SpecMargin::Len(spec_len_of(value)),
    }
}

/// Resolve `font-size`. Percentages and `em` refer to the parent font size;
/// `rem` to the root font size; `vw`/`vh` to the viewport.
///
/// Shared by paint's display-list builder (which re-reads raw `font-size`
/// values against the same bases layout used), so the mapping lives here
/// exactly once; lengths resolve through
/// [`bowser_css::length_to_px`].
pub fn resolve_font_size(
    value: Option<&CssValue>,
    parent_fs: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
) -> f32 {
    let fs = match value {
        None => return parent_fs,
        Some(CssValue::Length(v, unit)) => length_to_px(*v, unit, parent_fs, root_fs, vp_w, vp_h),
        Some(CssValue::Percentage(p)) => parent_fs * p / 100.0,
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
    };
    // Font sizes are lengths: finite and nonnegative. `f32::max` already
    // maps NaN to 0.0; `sane_len` additionally bounds infinities from
    // hostile values like `font-size: 1e30px`.
    sane_len(fs).max(0.0)
}

/// First family name of the computed `font-family` value. Mirrors
/// [`crate::layout::css_font_family_name`], duplicated here because
/// `style` is the layer `layout` builds on (importing `layout` from
/// `style` would invert that dependency).
pub(crate) fn font_family_name(style: &ComputedStyle) -> &str {
    match style.get_by_index(props::FONT_FAMILY) {
        Some(CssValue::List(items)) => items
            .iter()
            .find_map(|v| match v {
                CssValue::String(s) => Some(s.as_str()),
                CssValue::Keyword(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("serif"),
        _ => style
            .get_str_by_index(props::FONT_FAMILY)
            .unwrap_or("serif"),
    }
}

/// Resolve `line-height` to an absolute pixel value. `normal` (and any
/// missing / unresolvable value, which degrades to `normal`) resolves
/// through the face's real typographic line height
/// ([`FontMetrics::line_height`]) instead of the historical hardcoded
/// 1.2em: with [`crate::FixedFontMetrics`] that call returns exactly
/// 1.2em, so placeholder-metric tests keep their exact values, while
/// a real backend (paint's `FontCache`) returns its typo metrics.
/// This is the single production resolution site: every other
/// `line-height` consumer reads the `StyleData` value built here
/// (or inherits it), never re-resolves the computed value.
pub(crate) fn resolve_line_height(
    value: Option<&CssValue>,
    font_size: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    fonts: &dyn FontMetrics,
    family_name: &str,
) -> f32 {
    let normal = fonts.line_height(font_size, fonts.intern_family(family_name));
    let raw = match value {
        None => normal,
        Some(CssValue::Keyword(_)) => normal,
        Some(CssValue::Number(n)) => (n * font_size).max(0.0),
        Some(CssValue::Percentage(p)) => (p / 100.0 * font_size).max(0.0),
        Some(CssValue::Length(..)) => spec_len_of(value)
            .resolve(font_size, root_fs, vp_w, vp_h, font_size)
            .unwrap_or(normal)
            .max(0.0),
        _ => normal,
    };
    // Like font sizes: finite and nonnegative even for hostile inputs.
    sane_len(raw).max(0.0)
}

/// Map the computed `display` to a layout display. `table` and the
/// table-internal values get dedicated table layout (`layout::layout_table`);
/// `flex` gets flexbox; `grid` gets grid layout (`layout::layout_grid`).
/// The inline-level outer variants (`inline-flex`, `inline-grid`,
/// `inline-table`) keep their inner layout algorithm and become atomic
/// inline-level boxes (see [`Display::InlineFlex`] and friends).
///
/// There is deliberately no tag-name fallback table here: the cascade —
/// `bowser_css::default_ua_stylesheet()` plus author rules, resolved by
/// the [`Styler`] before this runs — is the single source of truth for
/// HTML `display` defaults. A local copy would drift from the UA
/// stylesheet (contract duplication), so unknown values fall back to the
/// CSS initial value (`inline`) and block-level values without dedicated
/// layout collapse to block.
pub(crate) fn display_of(style: &ComputedStyle) -> Display {
    let from_keyword = |d: &str| {
        // Zero-alloc ASCII case-insensitive matching: CSS keywords are
        // ASCII case-insensitive, so `eq_ignore_ascii_case` has the same
        // semantics as matching on `d.to_ascii_lowercase()` with no
        // per-box allocation.
        let any_of = |ks: &[&str]| ks.iter().any(|k| d.eq_ignore_ascii_case(k));
        if d.eq_ignore_ascii_case("none") {
            Display::None
        } else if any_of(&["block", "list-item", "flex"]) {
            Display::Block
        } else if d.eq_ignore_ascii_case("grid") {
            Display::Grid
        } else if d.eq_ignore_ascii_case("table") {
            Display::Table
        } else if d.eq_ignore_ascii_case("inline") {
            Display::Inline
        } else if d.eq_ignore_ascii_case("contents") {
            Display::Contents
        // The three inline-level outer displays keep their inner layout
        // algorithm: `layout_block` dispatches `InlineFlex` to flexbox
        // (via `layout::flex_kind`), `InlineGrid` to `layout_grid`, and
        // `InlineTable` to `layout_table`. Their outer box is the atomic
        // inline-level `InlineBlock`, so line layout, shrink-to-fit, and
        // `vertical-align` all work unchanged.
        } else if d.eq_ignore_ascii_case("inline-block") {
            Display::InlineBlock
        } else if d.eq_ignore_ascii_case("inline-flex") {
            Display::InlineFlex
        } else if d.eq_ignore_ascii_case("inline-grid") {
            Display::InlineGrid
        } else if d.eq_ignore_ascii_case("inline-table") {
            Display::InlineTable
        } else if d.eq_ignore_ascii_case("table-row-group") {
            Display::TableRowGroup
        } else if d.eq_ignore_ascii_case("table-header-group") {
            Display::TableHeaderGroup
        } else if d.eq_ignore_ascii_case("table-footer-group") {
            Display::TableFooterGroup
        } else if d.eq_ignore_ascii_case("table-row") {
            Display::TableRow
        } else if d.eq_ignore_ascii_case("table-cell") {
            Display::TableCell
        } else if d.eq_ignore_ascii_case("table-column") {
            Display::TableColumn
        } else if d.eq_ignore_ascii_case("table-column-group") {
            Display::TableColumnGroup
        // `flow-root` is a block container; `table-caption` lays out as a
        // plain block child of the table (caption-side top/bottom honored
        // by table layout).
        } else if any_of(&["flow-root", "table-caption"]) {
            Display::Block
        // Anything else is treated as inline: the CSS initial value of
        // `display`.
        } else {
            Display::Inline
        }
    };
    let d = match style.get_str_by_index(props::DISPLAY) {
        Some(d) => from_keyword(d),
        // The `Styler` always resolves `display` (initial value
        // `inline`), so this only fires for hand-built styles outside
        // the normal pipeline.
        None => Display::Inline,
    };
    // CSS 2.1 §9.7: a floated box is blockified — `inline`,
    // `inline-block`, and the inline-level outer displays
    // (`inline-flex`, `inline-grid`, `inline-table`) compute to `block`
    // when `float` is not `none` (`display: none` still generates no
    // box).
    if d != Display::None
        && float_of(style) != Float::None
        && matches!(
            d,
            Display::Inline
                | Display::InlineBlock
                | Display::InlineFlex
                | Display::InlineGrid
                | Display::InlineTable
        )
    {
        Display::Block
    } else {
        d
    }
}

/// Resolved `position`. `fixed` and `sticky` are parsed; how layout
/// treats them is documented in `super::layout::layout_position` (`fixed`
/// lays out like `absolute` against the viewport, `sticky` like
/// `relative`).
///
/// Public (not just layout-internal) because `bowser_app`'s scroll code
/// needs the same out-of-flow classification when baking paint trees;
/// one shared definition, no mirrored enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Position {
    Static,
    Relative,
    Absolute,
    Fixed,
    Sticky,
}

/// See [`Position`]: shared with `bowser_app` so the paint-shift
/// classification cannot drift from layout's.
pub fn position_of(style: &ComputedStyle) -> Position {
    match style.get_str_by_index(props::POSITION) {
        Some(v) if v.eq_ignore_ascii_case("relative") => Position::Relative,
        Some(v) if v.eq_ignore_ascii_case("absolute") => Position::Absolute,
        Some(v) if v.eq_ignore_ascii_case("fixed") => Position::Fixed,
        Some(v) if v.eq_ignore_ascii_case("sticky") => Position::Sticky,
        // Missing, `static`, or unrecognized: the CSS initial value.
        _ => Position::Static,
    }
}

/// Resolved `z-index`: `auto` (and missing/unrecognized) is `None`.
/// Layout records it on [`StyleData`]; paint-side stacking order is a
/// follow-up.
pub(crate) fn z_index_of(style: &ComputedStyle) -> Option<i32> {
    match style.get_by_index(props::Z_INDEX) {
        Some(CssValue::Number(n)) => Some(*n as i32),
        _ => None,
    }
}

/// Resolved `float`. `position: absolute | fixed` wins over `float`
/// (CSS 2.1 §9.7); layout checks out-of-flow first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Float {
    None,
    Left,
    Right,
}

pub(crate) fn float_of(style: &ComputedStyle) -> Float {
    match style.get_str_by_index(props::FLOAT) {
        Some(v) if v.eq_ignore_ascii_case("left") => Float::Left,
        Some(v) if v.eq_ignore_ascii_case("right") => Float::Right,
        // Missing, `none`, or unrecognized: the CSS initial value.
        _ => Float::None,
    }
}

/// Resolved `clear`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Clear {
    None,
    Left,
    Right,
    Both,
}

pub(crate) fn clear_of(style: &ComputedStyle) -> Clear {
    match style.get_str_by_index(props::CLEAR) {
        Some(v) if v.eq_ignore_ascii_case("left") => Clear::Left,
        Some(v) if v.eq_ignore_ascii_case("right") => Clear::Right,
        Some(v) if v.eq_ignore_ascii_case("both") => Clear::Both,
        // Missing, `none`, or unrecognized: the CSS initial value.
        _ => Clear::None,
    }
}

pub(crate) fn white_space_of(style: &ComputedStyle) -> WhiteSpace {
    match style.get_str_by_index(props::WHITE_SPACE) {
        Some("pre") => WhiteSpace::Pre,
        Some("pre-line") => WhiteSpace::PreLine,
        Some("pre-wrap") => WhiteSpace::PreWrap,
        Some("break-spaces") => WhiteSpace::BreakSpaces,
        Some("nowrap") => WhiteSpace::NoWrap,
        _ => WhiteSpace::Normal,
    }
}

pub(crate) fn text_align_of(style: &ComputedStyle) -> TextAlign {
    match style.get_str_by_index(props::TEXT_ALIGN) {
        Some("center") => TextAlign::Center,
        Some("right") => TextAlign::Right,
        Some("justify") => TextAlign::Justify,
        _ => TextAlign::Left,
    }
}

pub(crate) fn valign_of(style: &ComputedStyle) -> VAlign {
    match style.get_str_by_index(props::VERTICAL_ALIGN) {
        Some("top") => VAlign::Top,
        Some("bottom") => VAlign::Bottom,
        Some("middle") => VAlign::Middle,
        Some("sub") => VAlign::Sub,
        Some("super") => VAlign::Super,
        _ => VAlign::Baseline,
    }
}

/// Which box `width`/`height` (and the min/max sizes) refer to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BoxSizing {
    ContentBox,
    BorderBox,
}

pub(crate) fn box_sizing_of(style: &ComputedStyle) -> BoxSizing {
    match style.get_str_by_index(props::BOX_SIZING) {
        Some("border-box") => BoxSizing::BorderBox,
        _ => BoxSizing::ContentBox,
    }
}

/// Resolved `writing-mode` (CSS Writing Modes 3): which physical axes
/// the inline and block directions run along. The cascade folds every
/// logical property (`margin-inline-start`, `inline-size`, …) into its
/// physical counterpart against this mode (see `bowser_css`), so by
/// box-build time the physical slots already carry flow-relative
/// values; the mode itself is kept on the box for axis-sensitive
/// layout decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WritingMode {
    HorizontalTb,
    VerticalRl,
    VerticalLr,
}

pub(crate) fn writing_mode_of(style: &ComputedStyle) -> WritingMode {
    match style.get_str_by_index(props::WRITING_MODE) {
        Some(v) if v.eq_ignore_ascii_case("vertical-rl") => WritingMode::VerticalRl,
        Some(v) if v.eq_ignore_ascii_case("vertical-lr") => WritingMode::VerticalLr,
        // Missing, `horizontal-tb`, or unrecognized: the CSS initial value.
        _ => WritingMode::HorizontalTb,
    }
}

/// Resolved `aspect-ratio` (css-sizing-4 §3.1): `None` for `auto` (no
/// preferred ratio). `ratio` is `width / height`, always positive and
/// finite; `auto_preferred` is set for `auto <ratio>` (prefer the box's
/// intrinsic ratio, fall back to `ratio` — only observable on replaced
/// elements, which carry an intrinsic size).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AspectRatio {
    pub ratio: f32,
    pub auto_preferred: bool,
}

pub(crate) fn aspect_ratio_of(style: &ComputedStyle) -> Option<AspectRatio> {
    match style.get_by_index(props::ASPECT_RATIO) {
        Some(CssValue::AspectRatio {
            ratio,
            auto_preferred,
        }) if *ratio > 0.0 && ratio.is_finite() => Some(AspectRatio {
            ratio: *ratio,
            auto_preferred: *auto_preferred,
        }),
        _ => None,
    }
}

/// Resolved `border-*-style` as far as layout cares: `none` and `hidden`
/// force the used border width to zero (CSS 2.1 §8.5.3); every other
/// style keeps the specified `border-width`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BorderStyle {
    None,
    Hidden,
    Other,
}

impl BorderStyle {
    /// Whether this style zeroes the used border width.
    pub(crate) fn zeroes_width(self) -> bool {
        !matches!(self, BorderStyle::Other)
    }
}

pub(crate) fn border_style_of(style: &ComputedStyle, prop: u8) -> BorderStyle {
    match style.get_str_by_index(prop) {
        Some(s) if s.eq_ignore_ascii_case("hidden") => BorderStyle::Hidden,
        // The initial value is `none`, so a missing property resolves
        // the same way.
        Some(s) if s.eq_ignore_ascii_case("none") => BorderStyle::None,
        None => BorderStyle::None,
        _ => BorderStyle::Other,
    }
}

/// The resolved style data every box carries. Specified lengths stay
/// specified until layout, when the containing-block width is known.
#[derive(Clone, Debug)]
pub(crate) struct StyleData {
    pub display: Display,
    pub font_size: f32,
    pub line_height: f32,
    pub white_space: WhiteSpace,
    pub text_align: TextAlign,
    pub valign: VAlign,
    pub text_transform: TextTransform,
    /// Resolved `direction`: true when the nearest explicitly specified
    /// value on the element or its ancestors is `rtl`. Inherited, so it
    /// is threaded down during box construction (like font size) rather
    /// than walked up per query.
    pub rtl: bool,
    /// `unicode-bidi: plaintext` for the paragraphs this box lays
    /// out. The property is NOT inherited (CSS Writing Modes 3 §2.7):
    /// `plaintext` sets the base direction of the bidi paragraphs for
    /// which the element's box is the *containing block*, and no
    /// others. Semantics by box kind (threaded during box
    /// construction, see `boxes::bidi_plaintext`):
    ///
    /// - block container (block / inline-block / flex / grid / table /
    ///   table-internal): true iff this element's OWN computed
    ///   `unicode-bidi` is `plaintext`. An ancestor's `plaintext` never
    ///   crosses a block-container boundary into this box.
    /// - inline, text, and anonymous boxes: these share a containing
    ///   block, so the flag is that containing block's resolved value,
    ///   propagated across them. Anonymous blocks in particular
    ///   inherit it from the block container whose inline content they
    ///   wrap (see `StyleData::inherit`), so a block's own direct
    ///   inline content keeps its plaintext behavior when wrapped.
    ///
    /// Layout consults the flag on the block container performing
    /// inline layout (anonymous blocks included).
    pub bidi_plaintext: bool,
    /// Resolved `writing-mode`: the axes the inline/block directions
    /// run along. Inherited in the cascade, so the computed value read
    /// here is already correct; threaded down like `rtl` for boxes
    /// without their own style.
    pub writing_mode: WritingMode,
    /// Resolved `width`/`height`. The logical `inline-size`/`block-size`
    /// are folded into these slots by the cascade against the
    /// element's `writing-mode`.
    pub width: SpecLen,
    pub height: SpecLen,
    pub min_width: SpecLen,
    pub max_width: SpecLen,
    pub min_height: SpecLen,
    pub max_height: SpecLen,
    pub box_sizing: BoxSizing,
    /// Resolved `aspect-ratio` (`None` for `auto`). Not inherited.
    pub aspect_ratio: Option<AspectRatio>,
    /// top, right, bottom, left. The cascade resolves the logical
    /// longhands (`margin-inline-start`, …) into these physical slots
    /// against the element's `direction`/`writing-mode`, so these are
    /// already flow-relative by box-build time.
    pub margin: [SpecMargin; 4],
    pub padding: [SpecLen; 4],
    pub border: [SpecLen; 4],
    /// Resolved `border-*-style`, top, right, bottom, left. `none` and
    /// `hidden` force the used border width to zero in `resolve_metrics`.
    pub border_style: [BorderStyle; 4],
    /// Resolved `position`.
    pub position: Position,
    /// `top`, `right`, `bottom`, `left` offsets (`auto` when unspecified).
    /// The logical insets (`inset-inline-start`, …) are folded into
    /// these slots by the cascade.
    pub offset: [SpecLen; 4],
    /// Resolved `z-index` (`None` for `auto`).
    pub z_index: Option<i32>,
    /// Resolved `float`.
    pub float: Float,
    /// Resolved `clear`.
    pub clear: Clear,
}

impl StyleData {
    /// Inheritable subset for anonymous and text boxes (no element style).
    /// `vertical-align` is deliberately reset: it is not an inherited
    /// property (its initial value is `baseline`).
    pub(crate) fn inherit(parent: &StyleData) -> StyleData {
        StyleData {
            display: Display::Block,
            font_size: parent.font_size,
            line_height: parent.line_height,
            white_space: parent.white_space,
            text_align: parent.text_align,
            valign: VAlign::Baseline,
            text_transform: parent.text_transform,
            rtl: parent.rtl,
            bidi_plaintext: parent.bidi_plaintext,
            writing_mode: parent.writing_mode,
            width: SpecLen::Auto,
            height: SpecLen::Auto,
            min_width: SpecLen::Auto,
            max_width: SpecLen::Auto,
            min_height: SpecLen::Auto,
            max_height: SpecLen::Auto,
            box_sizing: BoxSizing::ContentBox,
            aspect_ratio: None,
            margin: [SpecMargin::Len(SpecLen::Px(0.0)); 4],
            padding: [SpecLen::Px(0.0); 4],
            border: [SpecLen::Px(0.0); 4],
            border_style: [BorderStyle::None; 4],
            position: Position::Static,
            offset: [SpecLen::Auto; 4],
            z_index: None,
            // `float` and `clear` are not inherited (CSS initial values).
            float: Float::None,
            clear: Clear::None,
        }
    }

    pub(crate) fn from_style(
        style: &ComputedStyle,
        display: Display,
        parent_fs: f32,
        root_fs: f32,
        vp_w: f32,
        vp_h: f32,
        fonts: &dyn FontMetrics,
    ) -> StyleData {
        let font_size = resolve_font_size(
            style.get_by_index(props::FONT_SIZE),
            parent_fs,
            root_fs,
            vp_w,
            vp_h,
        );
        StyleData {
            display,
            font_size,
            line_height: resolve_line_height(
                style.get_by_index(props::LINE_HEIGHT),
                font_size,
                root_fs,
                vp_w,
                vp_h,
                fonts,
                font_family_name(style),
            ),
            white_space: white_space_of(style),
            text_align: text_align_of(style),
            valign: valign_of(style),
            text_transform: text_transform_of(style),
            // Filled in by the caller with the inherited value
            // (`direction` inherits; `from_style` only sees the element's
            // own style).
            rtl: false,
            // Likewise filled in by the caller: `unicode-bidi` does not
            // inherit in the cascade, so the plaintext scoping thread
            // lives in box construction, not here.
            bidi_plaintext: false,
            // `writing-mode` inherits in the cascade, so the computed
            // value read here already reflects ancestors.
            writing_mode: writing_mode_of(style),
            width: spec_len_of(style.get_by_index(props::WIDTH)),
            height: spec_len_of(style.get_by_index(props::HEIGHT)),
            min_width: spec_len_of(style.get_by_index(props::MIN_WIDTH)),
            max_width: spec_len_of(style.get_by_index(props::MAX_WIDTH)),
            min_height: spec_len_of(style.get_by_index(props::MIN_HEIGHT)),
            max_height: spec_len_of(style.get_by_index(props::MAX_HEIGHT)),
            box_sizing: box_sizing_of(style),
            aspect_ratio: aspect_ratio_of(style),
            margin: [
                spec_margin_of(style.get_by_index(props::MARGIN_TOP)),
                spec_margin_of(style.get_by_index(props::MARGIN_RIGHT)),
                spec_margin_of(style.get_by_index(props::MARGIN_BOTTOM)),
                spec_margin_of(style.get_by_index(props::MARGIN_LEFT)),
            ],
            padding: [
                spec_len_of(style.get_by_index(props::PADDING_TOP)),
                spec_len_of(style.get_by_index(props::PADDING_RIGHT)),
                spec_len_of(style.get_by_index(props::PADDING_BOTTOM)),
                spec_len_of(style.get_by_index(props::PADDING_LEFT)),
            ],
            border: [
                spec_len_of(style.get_by_index(props::BORDER_TOP_WIDTH)),
                spec_len_of(style.get_by_index(props::BORDER_RIGHT_WIDTH)),
                spec_len_of(style.get_by_index(props::BORDER_BOTTOM_WIDTH)),
                spec_len_of(style.get_by_index(props::BORDER_LEFT_WIDTH)),
            ],
            border_style: [
                border_style_of(style, props::BORDER_TOP_STYLE),
                border_style_of(style, props::BORDER_RIGHT_STYLE),
                border_style_of(style, props::BORDER_BOTTOM_STYLE),
                border_style_of(style, props::BORDER_LEFT_STYLE),
            ],
            position: position_of(style),
            offset: [
                spec_len_of(style.get_by_index(props::TOP)),
                spec_len_of(style.get_by_index(props::RIGHT)),
                spec_len_of(style.get_by_index(props::BOTTOM)),
                spec_len_of(style.get_by_index(props::LEFT)),
            ],
            z_index: z_index_of(style),
            float: float_of(style),
            clear: clear_of(style),
        }
    }
}

/// Collapse two adjoining vertical margins per CSS 2.1 §8.3.1: the maximum
/// of non-negative margins, the minimum of non-positive ones, else the sum.
pub(crate) fn collapse(a: f32, b: f32) -> f32 {
    if a >= 0.0 && b >= 0.0 {
        a.max(b)
    } else if a <= 0.0 && b <= 0.0 {
        a.min(b)
    } else {
        a + b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedFontMetrics, Size, layout};

    // ---- A2: deferred percentage math resolves at used-value time ----

    #[test]
    fn deferred_math_resolves_against_containing_block() {
        // `width: calc(50% - 20px)` arrives from the cascade as
        // coefficients; layout plugs in the real containing-block base.
        let spec = spec_len_of(Some(&CssValue::MathDeferred {
            pct: 50.0,
            px: -20.0,
            had_pct: true,
            affine: true,
        }));
        assert!(matches!(spec, SpecLen::Math { .. }));
        let w = spec.resolve(16.0, 16.0, 1000.0, 800.0, 1000.0);
        assert_eq!(w, Some(480.0));
    }

    #[test]
    fn deferred_math_carries_had_pct_bit() {
        // `calc(10px + 0%)` folds to a zero percentage coefficient, but
        // the cascade marks `had_pct` and the CssValue→SpecLen conversion
        // preserves it, so intrinsic sizing can treat it as `auto`
        // (CSS Values 4 §8.1.4) instead of a definite 10px.
        let spec = spec_len_of(Some(&CssValue::MathDeferred {
            pct: 0.0,
            px: 10.0,
            had_pct: true,
            affine: true,
        }));
        assert!(matches!(
            spec,
            SpecLen::Math {
                pct,
                px,
                had_pct,
                affine
            } if pct == 0.0 && px == 10.0 && had_pct && affine
        ));
        // The bit is metadata only: resolution is unchanged.
        assert_eq!(spec.resolve(16.0, 16.0, 1000.0, 800.0, 1000.0), Some(10.0));
    }

    #[test]
    fn spec_len_math_stays_copy() {
        // A `bool` keeps `SpecLen::Math` (and hence `SpecLen`) `Copy`,
        // which the flag-threading design relies on.
        fn assert_copy<T: Copy>() {}
        assert_copy::<SpecLen>();
    }

    #[test]
    fn calc_percentage_width_end_to_end() {
        // Regression (A2): `width: calc(50% - 20px)` inside a 1000px
        // containing block lays out at 480px, not the old base-0
        // approximation (-20px, clamped).
        let (doc, _) =
            bowser_dom::Document::parse_html(br#"<div class="a"><div class="b"></div></div>"#);
        let ss = bowser_css::parse_stylesheet(
            ".a { width: 1000px; } .b { width: calc(50% - 20px); height: 5px; }",
        );
        let l = layout(
            &doc,
            &[&ss],
            Size {
                w: 1000.0,
                h: 800.0,
            },
            &FixedFontMetrics,
        );
        let b = &l.children[0].children[0].children[0];
        assert!(
            (b.rect.w - 480.0).abs() < 0.01,
            "calc(50% - 20px) of 1000px must be 480px, got {}",
            b.rect.w
        );
    }

    // ---- H1: keyword-basis calc-size() keeps its basis ----

    /// Parse `width: <…>` and return the specified value, so the tests
    /// exercise the real parser output rather than a hand-built AST.
    fn parsed_width_value(decl: &str) -> CssValue {
        let mut decls = bowser_css::parse_declarations(decl);
        assert_eq!(decls.len(), 1, "expected one declaration from {decl:?}");
        decls.pop().unwrap().value
    }

    #[test]
    fn calc_size_keyword_basis_maps_to_intrinsic_spec() {
        // The keyword basis must survive `spec_len_of` as the intrinsic
        // `SpecLen` the used-width arms measure — not the old silent
        // `Auto` fallthrough (H1).
        let v = parsed_width_value("width: calc-size(max-content, size + 10px)");
        assert!(matches!(spec_len_of(Some(&v)), SpecLen::MaxContent));
        let v = parsed_width_value("width: calc-size(min-content, size + 10px)");
        assert!(matches!(spec_len_of(Some(&v)), SpecLen::MinContent));
        let v = parsed_width_value("width: calc-size(fit-content, size + 10px)");
        assert!(matches!(spec_len_of(Some(&v)), SpecLen::FitContent));
        // `stretch` has no `SpecLen` form; `auto` fill is its block-level
        // equivalent, and both stay auto-ish for the fixed-mode table
        // trigger.
        let v = parsed_width_value("width: calc-size(stretch, size + 10px)");
        assert!(matches!(spec_len_of(Some(&v)), SpecLen::Auto));
    }

    #[test]
    fn calc_size_auto_basis_stays_auto_ish() {
        // `calc-size(auto, size)` still behaves auto-ish: `Auto` feeds
        // the normal auto paths (and keeps the fixed-mode table trigger
        // disabled, like bare `auto`).
        let v = parsed_width_value("width: calc-size(auto, size + 0px)");
        assert!(matches!(spec_len_of(Some(&v)), SpecLen::Auto));
    }

    #[test]
    fn calc_size_definite_basis_survivor_stays_auto() {
        // The cascade evaluates definite bases (`calc-size(100px, …)`)
        // itself; if one ever survived (unevaluable math), it keeps the
        // historical defensive `Auto` — no regression.
        let v = parsed_width_value("width: calc-size(100px, size + 10px)");
        assert!(matches!(spec_len_of(Some(&v)), SpecLen::Auto));
        let v = parsed_width_value("width: calc-size(50%, size + 10px)");
        assert!(matches!(spec_len_of(Some(&v)), SpecLen::Auto));
    }

    fn layout_block_width(css: &str) -> f32 {
        let (doc, _) = bowser_dom::Document::parse_html(br#"<div class="a">hello</div>"#);
        let ss = bowser_css::parse_stylesheet(css);
        let l = layout(
            &doc,
            &[&ss],
            Size {
                w: 1000.0,
                h: 800.0,
            },
            &FixedFontMetrics,
        );
        l.children[0].children[0].rect.w
    }

    #[test]
    fn calc_size_max_content_basis_end_to_end() {
        // H1 basis half: `width: calc-size(max-content, size + 10px)` on a
        // block must measure the max-content basis instead of degrading
        // to `auto` fill. Compared against the bare `width: max-content`
        // keyword: identical used widths. (Applying the `+ 10px`
        // calc-sum needs the used-value wiring in `layout.rs`; this pins
        // the basis behavior.)
        let w_calc = layout_block_width(".a { width: calc-size(max-content, size + 10px); }");
        let w_kw = layout_block_width(".a { width: max-content; }");
        assert!(
            (w_calc - w_kw).abs() < 0.01,
            "calc-size(max-content, …) basis must match bare max-content: got {w_calc} vs {w_kw}"
        );
        assert!(
            (w_calc - 1000.0).abs() > 1.0,
            "calc-size(max-content, …) must not degrade to auto fill, got {w_calc}"
        );
    }

    #[test]
    fn calc_size_auto_basis_end_to_end() {
        // `width: calc-size(auto, size)` behaves auto-ish end to end: a
        // block fills its containing block, exactly like `width: auto`.
        let w_calc = layout_block_width(".a { width: calc-size(auto, size); }");
        let w_auto = layout_block_width(".a { width: auto; }");
        assert!(
            (w_calc - w_auto).abs() < 0.01,
            "calc-size(auto, …) must behave like auto: got {w_calc} vs {w_auto}"
        );
        assert!(
            (w_calc - 1000.0).abs() < 0.01,
            "calc-size(auto, …) block must fill the 1000px containing block, got {w_calc}"
        );
    }
}
