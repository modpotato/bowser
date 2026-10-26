//! CSS values: [`CssValue`], colors, lengths, and the value parser.
//!
//! [`parse_value`] turns a declaration's token slice into a [`CssValue`].
//! Whitespace-separated runs become [`CssValue::List`]; top-level commas
//! also flatten into [`CssValue::List`] (one entry per comma group), so
//! callers that care about comma structure (e.g. `font-family`) work from
//! the list directly.
//!
//! Math functions (`calc()`, `min()`, `max()`, `clamp()`) parse into a
//! [`CssValue::Math`] expression tree. Relative units inside them need
//! layout context (font size, viewport) to resolve, so evaluation is a
//! separate step: [`CssValue::evaluate_math`] with a [`MathContext`].
//!
//! Color functions `rgb()`, `hsl()`, `hwb()`, `lab()`, and `lch()` all
//! convert to the sRGB [`Color`] at parse time.
//!
//! `var()` references are preserved as [`CssValue::Var`] holding the custom
//! property name and the fallback (if any) — parsed, except when the
//! fallback holds a top-level `/` or `,`, which is kept as verbatim
//! [`CssValue::Tokens`] — so the cascade step can substitute them later.
//! `currentcolor` has no representable color and is kept as
//! `Keyword("currentcolor")`.

use crate::cbprops::Filter;
use crate::kwatom::Keyword;
use crate::token::Token;

/// A parsed CSS property value.
#[derive(Clone, Debug, PartialEq)]
pub enum CssValue {
    Keyword(Keyword),
    Length(f32, LengthUnit),
    Percentage(f32),
    Color(Color),
    String(String),
    Url(String),
    Number(f32),
    List(Vec<CssValue>),
    /// `var(--name)` or `var(--name, fallback)`.
    ///
    /// Holds the custom property name verbatim (case preserved, since custom
    /// property names are case-sensitive) and the parsed fallback value. A
    /// fallback holding a top-level `/` or `,` is kept as verbatim
    /// [`CssValue::Tokens`] instead — the generic value parser would
    /// reject the `/` and flatten the comma groups — and is parsed at
    /// substitution time exactly like a custom property's token stream
    /// (see [`crate::Styler::resolve`]).
    Var(String, Option<Box<CssValue>>),
    /// A `calc()`, `min()`, `max()`, or `clamp()` expression, parsed but not
    /// yet evaluated. Resolve with [`CssValue::evaluate_math`].
    Math(MathExpr),
    /// A `calc()`-family computed value deferred to layout: the cascade
    /// substituted `var()`s and extracted `value = pct / 100 * base + px`.
    /// Affine-in-percentage trees defer as true coefficients (the cascade
    /// verified affinity and folded every font/viewport-relative leaf
    /// into the constant); non-affine percentage math (`min()`/`max()`/
    /// `clamp()` with `%`) cannot fold to coefficients, so it defers as
    /// `pct: 0.0, px: <base-0 approximation>` — no coefficient pair can
    /// capture the tree, and resolving against any base yields exactly
    /// the historical base-0 value. The two forms are otherwise
    /// value-identical (e.g. `min(50px, 20%)` and `calc(0% + 0px)` both
    /// fold to `pct: 0.0, px: 0.0`), so the `affine` discriminator keeps
    /// them apart: consumers that must treat a frozen approximation
    /// differently from a true coefficient pair (table column sizing)
    /// check this bit, not the coefficient values. Both forms carry
    /// `had_pct`, so the syntactic percentage record survives
    /// evaluation either way.
    /// Layout plugs in the real containing-block base at used-value time
    /// (see `bowser_layout`'s `SpecLen::resolve`). Only produced for
    /// properties whose percentage base is a containing-block size;
    /// `font-size`/`line-height` percentages resolve against font metrics
    /// in the cascade instead.
    MathDeferred {
        pct: f32,
        px: f32,
        /// Whether the original `calc()` tree syntactically contained a
        /// `%` leaf, captured via [`MathExpr::contains_percentage`] at
        /// fold time. Consumers deciding cyclic-ness (CSS Values 4
        /// §8.1.4: a percentage against an indefinite base makes the
        /// whole value behave as `auto`) must check this bit, not
        /// `pct != 0.0`: `calc(10px + 0%)` folds to `pct: 0.0, px: 10.0`
        /// yet still carries a syntactic percentage.
        had_pct: bool,
        /// Discriminator between the two deferred forms. `true` means
        /// the cascade verified affinity and the `pct`/`px` pair is a
        /// genuine coefficient pair (`value = pct / 100 * base + px`
        /// for the real base). `false` means the value is the frozen
        /// base-0 approximation of non-affine percentage math
        /// (`min()`/`max()`/`clamp()` with `%`), valid only as a
        /// resolved length, never as a percentage participant.
        affine: bool,
    },
    /// A `<number>fr` flexible track fraction (the grid `fr` unit). Only
    /// meaningful inside grid track listings (`grid-template-columns` /
    /// `grid-template-rows`); every other property's grammar validator
    /// rejects it, so it never leaks into e.g. `width`.
    Fr(f32),
    /// `minmax(<min>, <max>)` from grid track listings: the track's base
    /// size is at least `<min>` and its growth is limited by `<max>`.
    /// Each side is a track breadth — a length/percentage, an `fr`
    /// fraction (invalid as the minimum), `min-content`, `max-content`,
    /// `auto`, or math (`calc()` etc., evaluated by the cascade) — with
    /// `minmax()` itself unable to nest. Only meaningful inside grid
    /// track listings; layout resolves it in track sizing.
    MinMax(Box<CssValue>, Box<CssValue>),
    /// `calc-size(<basis>, <calc-sum>)` (CSS Values 4 §7): the sizing
    /// basis is resolved first, then the `<calc-sum>` applies to it; the
    /// sum's `size` interpolation keyword is [`MathExpr::Size`]. The
    /// cascade evaluates this to a plain length (or defers it as
    /// [`CssValue::MathDeferred`] when percentages are involved) for
    /// definite bases (see [`CalcSizeBasis::is_definite`]). Keyword
    /// bases (`auto`/`max-content`/`min-content`/`fit-content`/`stretch`)
    /// need content measurement the CSS crate cannot do, so they
    /// survive the cascade untouched for layout, which reads the basis
    /// through [`CssValue::calc_size_basis`]. `var()` in the basis
    /// position is rejected at parse time (its fallback would need a
    /// basis grammar, not a calc grammar); `var()` inside the calc-sum
    /// substitutes normally. Both fields are boxed so the variant does
    /// not grow [`CssValue`] (see [`CssValue::MinMax`).
    CalcSize {
        basis: Box<CalcSizeBasis>,
        calc: Box<MathExpr>,
    },
    /// `repeat(auto-fill|auto-fit, <track-list>)` from grid track
    /// listings, passed through *unexpanded*: the repetition count
    /// depends on the axis size, which is only known at layout time
    /// (layout expands it where the axis size is known). A dedicated
    /// variant — not a `List([Keyword(mode), List(tracks)])` — so the
    /// cascade's comma-group normalization can never flatten a
    /// single-track descriptor into the byte-identical shape of the
    /// invalid bare `auto-fill <track>` listing. Integer `repeat()`
    /// still expands eagerly into a [`CssValue::List`] at parse time.
    Repeat {
        mode: RepeatMode,
        tracks: Vec<CssValue>,
    },
    /// The raw token stream of a custom property (`--*`) value, stored
    /// verbatim at parse time. Custom properties are token-preserving
    /// per CSS: the stream is parsed only when a `var()` reference
    /// substitutes it (see [`crate::Styler::resolve`]), so values like
    /// `--x: 1/2/3` survive instead of being dropped, and a `/` in the
    /// stream is still visible to the substitution site. Never appears
    /// in a computed style: resolution always parses it first.
    Tokens(Vec<Token>),
    /// A preferred `aspect-ratio` (`width / height`): the `w / h` of an
    /// `aspect-ratio: w / h` declaration, stored as a single quotient
    /// plus the `auto`-preferred flag from `auto <ratio>`
    /// (css-sizing-4 §3.1). `ratio` is always positive and finite; a
    /// bare `auto` computes to `Keyword("auto")` (no preferred ratio),
    /// exactly like `width: auto`.
    AspectRatio {
        /// `width / height` (positive, finite).
        ratio: f32,
        /// True for `auto <ratio>`: prefer the box's intrinsic ratio
        /// and fall back to `ratio` (only observable on replaced
        /// elements, which carry an intrinsic size).
        auto_preferred: bool,
    },
    /// A `transform` function list (CSS Transforms 1 §8): `none` parses
    /// to `Keyword("none")`; anything else parses here, one entry per
    /// function, in a single `Vec` no matter how the list was written
    /// (a space-separated list accumulates into this one vec at the
    /// value level rather than arriving as `List` of `Transform`s).
    /// The containing-block-relevant subset (`translate()` / `scale()`
    /// / `rotate()` and their single-axis forms) parses structurally;
    /// every other transform function (`matrix()` / `matrix3d()`,
    /// `skew()` / `skewx()` / `skewy()`, `perspective()`, the `*3d()`
    /// and `*z()` forms, and strict-subset functions with nested
    /// arguments such as `translate(calc(10px + 5px))`) is retained
    /// opaque as [`TransformFunction::Other`]. A valid-but-unrenderable
    /// function still counts as non-`none` (it makes the box the
    /// containing block for `position: fixed` descendants); only truly
    /// invalid syntax (unknown functions, wrong arity, empty bodies)
    /// fails the declaration. The functions are stored so future work
    /// can render them; layout only needs `none` vs non-`none` today.
    Transform(Vec<TransformFunction>),
    /// A `filter` function list (Filter Effects 1 §8): `none` parses to
    /// [`Filter::None`]; anything else parses to [`Filter::List`] with
    /// one entry per function (see [`FilterFunction`], which is retained
    /// opaque like [`TransformFunction::Other`]). Only the
    /// containing-block effect is implemented (a non-`none` filter makes
    /// the box the containing block for fixed/absolute descendants);
    /// rendering the filter itself is future work.
    Filter(Filter),
    /// A `clip-path` value (CSS Shapes 1 / css-clip-path-1 §4): `none`,
    /// one of the `<basic-shape>` functions, or a `url()` reference.
    /// Length leaves are [`CssValue`]s (`Length`, `Percentage`,
    /// unitless `Number`, `Math`, or post-cascade `MathDeferred`),
    /// resolved against the reference box (the border box) at paint
    /// time by `bowser_paint`.
    ClipPath(ClipPath),
    /// A `box-shadow` / `text-shadow` layer list (CSS Backgrounds 3 §7 /
    /// CSS Text Decoration 3 §4.3): `none` computes to the empty list;
    /// otherwise one [`Shadow`] per comma-separated `<shadow>`, front to
    /// back (the first listed paints on top). Produced by the
    /// token-level parser (`parse_shadow_value` in parse.rs) and by the
    /// cascade's `fold_shadow_value`, which assembles `var()`-substituted
    /// component lists into this same shape before grammar validation,
    /// so consumers only ever match this one variant.
    Shadows(Vec<Shadow>),
    /// A CSS gradient image function (`linear-gradient()` /
    /// `radial-gradient()` / `conic-gradient()` and the `repeating-`
    /// variants; CSS Images 3/4), retained verbatim: `name` is the
    /// lowercased function name and `source` the full function text
    /// re-serialized from tokens (whitespace-collapsed), ready to hand
    /// to `bowser_paint::gradient::parse_gradient` at
    /// style-application time. `bowser_css` cannot depend on
    /// `bowser_paint` (the dependency runs the other way), so the
    /// gradient is *not* structurally validated here — only an empty
    /// body drops the declaration. Full spec validation lives in
    /// paint's parser; a gradient that fails there paints nothing,
    /// which matches the rendering of a dropped declaration.
    Gradient {
        /// Lowercased function name, e.g. `linear-gradient`.
        name: String,
        /// Full function text, e.g. `linear-gradient(red, blue)`.
        source: String,
    },
    /// An `attr()` reference in `content` (CSS Generated Content 3
    /// §2.2): `attr(<name> [string] [, <fallback>]?)`. Only the string
    /// form is parsed (other `<type-or-unit>`s are out of scope and
    /// drop the declaration). `bowser_layout` resolves this against the
    /// originating element's attributes at box-synthesis time (see
    /// `pseudo::resolve_content_text`): a missing attribute with no
    /// fallback contributes the empty string. Boxed so the variant
    /// does not grow [`CssValue`] (see [`CssValue::MinMax`]).
    Attr(Box<AttrData>),
    /// A `counter()` / `counters()` reference in `content` (CSS
    /// Generated Content 3 §2.2, CSS Lists 3 §7):
    /// `counter(<name> [, <style>])` /
    /// `counters(<name>, <separator> [, <style>])`. `bowser_layout`
    /// resolves these against the counter scopes rooted at the
    /// originating element (built from `counter-reset` /
    /// `counter-increment`) at box-synthesis time (see
    /// `pseudo::resolve_content_text`). Boxed so the variant does not
    /// grow [`CssValue`] (see [`CssValue::MinMax`]).
    Counter(Box<CounterData>),
}

/// Payload of [`CssValue::Attr`]: an `attr()` reference in `content`.
#[derive(Clone, Debug, PartialEq)]
pub struct AttrData {
    /// Lowercased attribute name.
    pub name: String,
    /// Parsed fallback (`attr(name, fallback)`), if any.
    pub fallback: Option<Box<CssValue>>,
}

/// Payload of [`CssValue::Counter`]: a `counter()` / `counters()`
/// reference in `content`.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterData {
    /// Lowercased counter name.
    pub name: String,
    /// Lowercased counter-style name, if given (`None` = `decimal`;
    /// unknown styles degrade to `decimal` at render).
    pub style: Option<String>,
    /// The `counters()` separator string, if given (`None` for
    /// `counter()`).
    pub separator: Option<String>,
}

/// One `<shadow>` layer of `box-shadow` / `text-shadow` (CSS
/// Backgrounds 3 §7.2 / CSS Text Decoration 3 §4.3).
///
/// Length fields are [`CssValue`]s so relative units and `calc()` keep
/// their authored form until computed-value time: each is one of
/// `Length(..)`, unitless-zero `Number(0.0)`, unevaluated `Math`, or
/// `MathDeferred` (percentages never survive: shadows have no percentage
/// base, so percentage math is rejected). `color` is `Color(..)` or
/// `Keyword("currentcolor")` (the default when the color is omitted).
/// `inset` and a non-zero `spread` only occur for `box-shadow`
/// (`text-shadow` layers always have `inset == false` and a zero
/// spread).
#[derive(Clone, Debug, PartialEq)]
pub struct Shadow {
    /// Horizontal shadow offset (any sign).
    pub offset_x: CssValue,
    /// Vertical shadow offset (any sign).
    pub offset_y: CssValue,
    /// Blur radius: non-negative (a negative radius is invalid).
    pub blur: CssValue,
    /// Spread distance (any sign; `box-shadow` only).
    pub spread: CssValue,
    /// Shadow color: `Color(..)` or `Keyword("currentcolor")`.
    pub color: CssValue,
    /// The `inset` keyword was present (`box-shadow` only).
    pub inset: bool,
}

impl Shadow {
    /// Whether `v` is a valid shadow length component: a `<length>` in
    /// any unit, unitless zero (CSS-wide), or math standing in for a
    /// length. Percentages are not `<length>`s; `Math` holding a
    /// percentage is rejected because shadows have no percentage base
    /// to resolve it against. The deferred gate checks the `had_pct`
    /// bit, not the folded coefficient: `calc(0% + 10px)` folds to
    /// `pct: 0.0` but still contains a percentage (CSS Values 4 §8.1.4)
    /// and is invalid where percentages are not allowed.
    fn is_shadow_length(v: &CssValue) -> bool {
        match v {
            CssValue::Length(_, _) => true,
            CssValue::Number(n) => *n == 0.0,
            CssValue::Math(e) => !e.contains_percentage(),
            CssValue::MathDeferred { had_pct, .. } => !had_pct,
            _ => false,
        }
    }

    /// Whether `v` is a negative computed length (a literal negative
    /// length, or evaluated math that came out negative).
    fn is_negative_computed(v: &CssValue) -> bool {
        match v {
            CssValue::Length(n, _) => *n < 0.0,
            CssValue::MathDeferred { pct, px, .. } => *pct == 0.0 && *px < 0.0,
            _ => false,
        }
    }

    /// Assemble one `<shadow>` from a comma group's parsed components:
    /// `inset? && <length>{2,4} && <color>?` for `box-shadow`
    /// (`allow_inset`), `<length>{2,3} && <color>?` for `text-shadow`.
    /// The `&&` lets `inset` and the color appear in any order relative
    /// to the lengths; the lengths themselves fill offset-x, offset-y,
    /// blur, spread positionally. At most one color (a `Color` or
    /// `currentcolor`); `inset` at most once and only when
    /// `allow_inset`. A negative blur radius is invalid (CSS
    /// Backgrounds 3 §7.2: "Negative values are not allowed"); anything
    /// that is not a length, a color, or `inset` is invalid.
    pub(crate) fn from_components(items: &[CssValue], allow_inset: bool) -> Option<Shadow> {
        // `text-shadow` has no spread slot and no `inset`.
        let max_lengths = if allow_inset { 4 } else { 3 };
        let mut lengths: [Option<&CssValue>; 4] = [None, None, None, None];
        let mut n_lengths = 0usize;
        let mut color: Option<&CssValue> = None;
        let mut inset = false;
        for item in items {
            if matches!(item, CssValue::Keyword(k) if k == "inset") {
                if !allow_inset || inset {
                    return None;
                }
                inset = true;
            } else if matches!(item, CssValue::Color(_))
                || matches!(item, CssValue::Keyword(k) if k == "currentcolor")
            {
                if color.is_some() {
                    return None;
                }
                color = Some(item);
            } else if Self::is_shadow_length(item) {
                if n_lengths >= max_lengths {
                    return None;
                }
                lengths[n_lengths] = Some(item);
                n_lengths += 1;
            } else {
                return None;
            }
        }
        if n_lengths < 2 {
            return None;
        }
        // The third length is the blur radius: a literal negative blur
        // is invalid here; a negative `calc()` result is caught by
        // `components_valid` after math evaluation.
        if let Some(CssValue::Length(blur, _)) = lengths[2]
            && *blur < 0.0
        {
            return None;
        }
        let shadow = Shadow {
            offset_x: lengths[0].expect("at least two lengths").clone(),
            offset_y: lengths[1].expect("at least two lengths").clone(),
            blur: lengths[2].cloned().unwrap_or(CssValue::Number(0.0)),
            spread: lengths[3].cloned().unwrap_or(CssValue::Number(0.0)),
            color: color
                .cloned()
                .unwrap_or(CssValue::Keyword("currentcolor".into())),
            inset,
        };
        Some(shadow)
    }

    /// Whether the shadow still satisfies the grammar after math
    /// evaluation (checked on both grammar-validation passes, so it
    /// must also hold for the pre-evaluation shapes `from_components`
    /// produces): every field is a valid component, the blur radius is
    /// non-negative, and no percentage math survived. The parser and
    /// `from_components` establish this for `Math` leaves; evaluation
    /// can only narrow the shapes (a unitless `calc()` result becomes a
    /// bare `Number`, which only zero may be).
    pub(crate) fn components_valid(&self) -> bool {
        Self::is_shadow_length(&self.offset_x)
            && Self::is_shadow_length(&self.offset_y)
            && Self::is_shadow_length(&self.blur)
            && !Self::is_negative_computed(&self.blur)
            && Self::is_shadow_length(&self.spread)
            && (matches!(&self.color, CssValue::Color(_))
                || matches!(&self.color, CssValue::Keyword(k) if k == "currentcolor"))
    }
}

/// A `clip-path` value (CSS Shapes 1 / css-clip-path-1 §4): `none`, one
/// of the `<basic-shape>` functions, or a `url()` reference.
///
/// Length leaves are [`CssValue`]s — `Length`, `Percentage`, unitless
/// `Number`, unevaluated `Math`, or post-cascade `MathDeferred` —
/// resolved against the reference box (the border box) at paint time.
/// `Var` leaves can survive when a `var()` sits inside a shape function
/// (the cascade only substitutes whole-value `var()`s for this
/// property); paint treats them as unresolvable and skips the clip.
#[derive(Clone, Debug, PartialEq)]
pub enum ClipPath {
    /// `none`: the initial value; no clipping.
    None,
    /// `inset()` / `circle()` / `ellipse()` / `polygon()`. Boxed: the
    /// shape holds [`CssValue`] leaves inline, so without indirection
    /// `CssValue` would be infinitely sized.
    Shape(Box<BasicShape>),
    /// `url(...)`: an SVG `<clipPath>` reference. Parsed and retained
    /// for `getComputedStyle` round-tripping but not rendered (paints
    /// as `none`).
    Url(String),
}

/// One `<basic-shape>` (CSS Shapes 1 §5–§8) as used by `clip-path`.
#[derive(Clone, Debug, PartialEq)]
pub enum BasicShape {
    /// `inset(<top> <right> <bottom> <left> [round <radii>]?)`: one to
    /// four offsets expanded to four sides (CSS 1–4-value rule);
    /// `radii` is `round <radius>{1,4}` expanded the same way, `None`
    /// when absent.
    Inset {
        offsets: [CssValue; 4],
        radii: Option<[CssValue; 4]>,
    },
    /// `circle([<radius>]? [at <position>]?)`.
    Circle {
        radius: ShapeRadius,
        /// `at <position>`; defaults to `50% 50%` (the center).
        cx: CssValue,
        cy: CssValue,
    },
    /// `ellipse([<rx> <ry>]? [at <position>]?)`; each radius is a
    /// [`ShapeRadius`], resolved per axis (CSS Shapes 1 §6.2): a
    /// `closest-side` `rx` is the distance from the center to the
    /// nearer of the left/right sides, and likewise `ry` for the
    /// top/bottom sides. Omitted radii default to `closest-side`.
    Ellipse {
        rx: ShapeRadius,
        ry: ShapeRadius,
        cx: CssValue,
        cy: CssValue,
    },
    /// `polygon([<fill-rule>,]? [<x> <y>]#)`. The fill rule is accepted
    /// and ignored: paint always uses the even-odd rule.
    Polygon { points: Vec<(CssValue, CssValue)> },
}

/// A `<shape-radius>` (CSS Shapes 1 §6): `circle()`'s radius and each
/// of `ellipse()`'s radii. Resolution is center-aware (see
/// `resolve_clip_path` in `bowser_paint`).
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeRadius {
    /// `<length-percentage>`.
    Length(CssValue),
    /// `closest-side`: for `circle()`, the distance from the center to
    /// the closest side of the reference box (also the default when
    /// `circle()`'s radius is omitted); for `ellipse()`, per axis —
    /// `rx` uses the nearer of the left/right sides, `ry` the nearer
    /// of the top/bottom sides (also the default when `ellipse()`'s
    /// radii are omitted).
    ClosestSide,
    /// `farthest-side`: center to the farthest side of the reference
    /// box (`ellipse()` resolves this per axis, like `closest-side`).
    FarthestSide,
}

impl ClipPath {
    /// Length leaves in canonical order, for math evaluation: inset
    /// offsets (4) then radii (4, when present); circle radius (when a
    /// length), `cx`, `cy`; ellipse `rx`, `ry`, `cx`, `cy`; polygon
    /// points flattened (`x0, y0, x1, y1, …`).
    pub(crate) fn leaves(&self) -> Vec<CssValue> {
        match self {
            ClipPath::None | ClipPath::Url(_) => Vec::new(),
            ClipPath::Shape(shape) => shape.leaves(),
        }
    }

    /// Rebuild from evaluated leaves in [`leaves`](Self::leaves) order;
    /// `None` when the leaf count does not match the shape (an
    /// evaluation failed and the whole value is invalid).
    pub(crate) fn with_leaves(&self, leaves: &[CssValue]) -> Option<ClipPath> {
        match self {
            ClipPath::None => Some(ClipPath::None),
            ClipPath::Url(u) => Some(ClipPath::Url(u.clone())),
            ClipPath::Shape(shape) => shape.with_leaves(leaves).map(Box::new).map(ClipPath::Shape),
        }
    }
}

impl BasicShape {
    /// Length leaves in canonical order (see [`ClipPath::leaves`]).
    pub(crate) fn leaves(&self) -> Vec<CssValue> {
        match self {
            BasicShape::Inset { offsets, radii } => {
                let mut out = Vec::with_capacity(8);
                out.extend(offsets.iter().cloned());
                if let Some(radii) = radii {
                    out.extend(radii.iter().cloned());
                }
                out
            }
            BasicShape::Circle { radius, cx, cy } => {
                let mut out = Vec::with_capacity(3);
                if let ShapeRadius::Length(v) = radius {
                    out.push(v.clone());
                }
                out.push(cx.clone());
                out.push(cy.clone());
                out
            }
            BasicShape::Ellipse { rx, ry, cx, cy } => {
                let mut out = Vec::with_capacity(4);
                if let ShapeRadius::Length(v) = rx {
                    out.push(v.clone());
                }
                if let ShapeRadius::Length(v) = ry {
                    out.push(v.clone());
                }
                out.push(cx.clone());
                out.push(cy.clone());
                out
            }
            BasicShape::Polygon { points } => {
                let mut out = Vec::with_capacity(points.len() * 2);
                for (x, y) in points {
                    out.push(x.clone());
                    out.push(y.clone());
                }
                out
            }
        }
    }

    /// Rebuild from evaluated leaves in [`leaves`](Self::leaves) order.
    pub(crate) fn with_leaves(&self, leaves: &[CssValue]) -> Option<BasicShape> {
        let mut it = leaves.iter();
        let mut next = || it.next().cloned();
        match self {
            BasicShape::Inset { radii, .. } => {
                let offsets = [next()?, next()?, next()?, next()?];
                let radii = match radii {
                    Some(_) => Some([next()?, next()?, next()?, next()?]),
                    None => None,
                };
                if it.next().is_some() {
                    return None;
                }
                Some(BasicShape::Inset { offsets, radii })
            }
            BasicShape::Circle { radius, .. } => {
                let radius = match radius {
                    ShapeRadius::Length(_) => ShapeRadius::Length(next()?),
                    ShapeRadius::ClosestSide => ShapeRadius::ClosestSide,
                    ShapeRadius::FarthestSide => ShapeRadius::FarthestSide,
                };
                let (cx, cy) = (next()?, next()?);
                if it.next().is_some() {
                    return None;
                }
                Some(BasicShape::Circle { radius, cx, cy })
            }
            BasicShape::Ellipse { rx, ry, .. } => {
                // Like `circle()`'s radius: only `<length-percentage>`
                // radii are math-evaluated leaves; `closest-side` /
                // `farthest-side` resolve against the reference box.
                let mut shape_radius = |r: &ShapeRadius| -> Option<ShapeRadius> {
                    Some(match r {
                        ShapeRadius::Length(_) => ShapeRadius::Length(next()?),
                        ShapeRadius::ClosestSide => ShapeRadius::ClosestSide,
                        ShapeRadius::FarthestSide => ShapeRadius::FarthestSide,
                    })
                };
                let (rx, ry) = (shape_radius(rx)?, shape_radius(ry)?);
                let (cx, cy) = (next()?, next()?);
                if it.next().is_some() {
                    return None;
                }
                Some(BasicShape::Ellipse { rx, ry, cx, cy })
            }
            BasicShape::Polygon { points } => {
                let mut rebuilt = Vec::with_capacity(points.len());
                for _ in points {
                    rebuilt.push((next()?, next()?));
                }
                if it.next().is_some() {
                    return None;
                }
                Some(BasicShape::Polygon { points: rebuilt })
            }
        }
    }
}

/// Split tokens into top-level whitespace-separated groups
/// (whitespace inside functions/parens never separates). Used by the
/// `<basic-shape>` parsers below.
fn shape_ws_groups(body: &[Token]) -> Vec<Vec<Token>> {
    let mut groups: Vec<Vec<Token>> = Vec::new();
    let mut current: Vec<Token> = Vec::new();
    let mut depth: u32 = 0;
    let flush = |current: &mut Vec<Token>, groups: &mut Vec<Vec<Token>>| {
        if !current.is_empty() {
            groups.push(std::mem::take(current));
        }
    };
    for t in body {
        match t {
            Token::LParen => depth += 1,
            Token::RParen => depth = depth.saturating_sub(1),
            Token::Function(_) => depth += 1,
            Token::Whitespace if depth == 0 => {
                flush(&mut current, &mut groups);
                continue;
            }
            _ => {}
        }
        current.push(t.clone());
    }
    flush(&mut current, &mut groups);
    groups
}

/// Whether `v` is a valid `<basic-shape>` length component: a `<length>`
/// in any unit, a `<percentage>`, a unitless number (treated as px, the
/// same leniency layout applies), math standing in for either, or a
/// `var()` leaf the cascade has not substituted (paint rejects those).
fn is_shape_length(v: &CssValue) -> bool {
    matches!(
        v,
        CssValue::Length(_, _)
            | CssValue::Percentage(_)
            | CssValue::Number(_)
            | CssValue::Math(_)
            | CssValue::MathDeferred { .. }
            | CssValue::Var(_, _)
    )
}

/// Parse one `<basic-shape>` length component from its token slice via
/// the generic value parser.
fn parse_shape_length(toks: &[Token]) -> Result<CssValue, ValueError> {
    let v = parse_value(toks)?;
    if is_shape_length(&v) {
        Ok(v)
    } else {
        Err(ValueError)
    }
}

/// Expand a 1–4-value list to four sides (CSS 1–4-value rule).
fn expand_four(mut vals: Vec<CssValue>) -> Result<[CssValue; 4], ValueError> {
    if vals.is_empty() || vals.len() > 4 {
        return Err(ValueError);
    }
    while vals.len() < 4 {
        // 1→[a,a,a,a], 2→[a,b,a,b], 3→[a,b,c,b]: missing entries copy
        // the opposite side's value; a single value fills all sides
        // (indexing `len() - 2` would underflow on a 1-element vec).
        let fill = if vals.len() == 1 {
            vals[0].clone()
        } else {
            vals[vals.len() - 2].clone()
        };
        vals.push(fill);
    }
    Ok([
        vals.remove(0),
        vals.remove(0),
        vals.remove(0),
        vals.remove(0),
    ])
}

/// Parse `at <position>` (one or two components) into `(cx, cy)`
/// (css-values-4 §10.2): a single value sets the horizontal axis only
/// and the vertical defaults to `center`; two values are horizontal
/// then vertical. `left`/`right` bind to the horizontal axis,
/// `top`/`bottom` to the vertical, while `<length-percentage>` and
/// `center` fit either axis. The swapped keyword order (`top left`)
/// is accepted as a leniency.
fn parse_shape_position(toks: &[Token]) -> Result<(CssValue, CssValue), ValueError> {
    let groups = shape_ws_groups(toks);
    if groups.is_empty() || groups.len() > 2 {
        return Err(ValueError);
    }
    // One component's axis affinity: `left`/`right` are horizontal
    // only, `top`/`bottom` vertical only, and `<length-percentage>` /
    // `center` fit either axis (they take the axis their slot implies).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum PosAxis {
        Horizontal,
        Vertical,
        Ambiguous,
    }
    // Classify one component into (axis affinity, value).
    fn classify(toks: &[Token]) -> Result<(PosAxis, CssValue), ValueError> {
        if toks.len() != 1 {
            // Only single-token components are positions here; anything
            // longer (e.g. a `calc()` token stream) parses as a length,
            // which is axis-ambiguous.
            let v = parse_shape_length(toks)?;
            return Ok((PosAxis::Ambiguous, v));
        }
        match &toks[0] {
            Token::Ident(kw) => {
                let pct = |p: f32| CssValue::Percentage(p);
                match kw.to_ascii_lowercase().as_str() {
                    "left" => Ok((PosAxis::Horizontal, pct(0.0))),
                    "right" => Ok((PosAxis::Horizontal, pct(100.0))),
                    "top" => Ok((PosAxis::Vertical, pct(0.0))),
                    "bottom" => Ok((PosAxis::Vertical, pct(100.0))),
                    "center" => Ok((PosAxis::Ambiguous, pct(50.0))),
                    _ => Err(ValueError),
                }
            }
            _ => {
                let v = parse_shape_length(toks)?;
                Ok((PosAxis::Ambiguous, v))
            }
        }
    }
    let center = || CssValue::Percentage(50.0);
    match groups.as_slice() {
        [one] => {
            let (axis, v) = classify(one)?;
            // A single value sets the horizontal axis; the vertical
            // defaults to `center` — unless the value is an explicit
            // vertical keyword (`at top`), which centers the horizontal.
            match axis {
                PosAxis::Vertical => Ok((center(), v)),
                _ => Ok((v, center())),
            }
        }
        [first, second] => {
            let (a1, v1) = classify(first)?;
            let (a2, v2) = classify(second)?;
            // Reject `left right` (two horizontals) and `top bottom`
            // (two verticals).
            let two_h = a1 == PosAxis::Horizontal && a2 == PosAxis::Horizontal;
            let two_v = a1 == PosAxis::Vertical && a2 == PosAxis::Vertical;
            if two_h || two_v {
                return Err(ValueError);
            }
            // Each component takes its explicit axis; an ambiguous
            // component takes the axis its position implies (first is
            // horizontal, second vertical), so the swapped order
            // (`top left`, `top 20%`) uncrosses to `(left, top)`.
            let (cx, cy) = match (a1, a2) {
                (PosAxis::Vertical, _) | (PosAxis::Ambiguous, PosAxis::Horizontal) => (v2, v1),
                _ => (v1, v2),
            };
            Ok((cx, cy))
        }
        _ => Err(ValueError),
    }
}

/// Parse one `<shape-radius>` (CSS Shapes 1 §6): `<length-percentage>`
/// or the `closest-side` / `farthest-side` keywords.
fn parse_shape_radius(toks: &[Token]) -> Result<ShapeRadius, ValueError> {
    if let [Token::Ident(kw)] = toks {
        if kw.eq_ignore_ascii_case("closest-side") {
            return Ok(ShapeRadius::ClosestSide);
        }
        if kw.eq_ignore_ascii_case("farthest-side") {
            return Ok(ShapeRadius::FarthestSide);
        }
    }
    parse_shape_length(toks).map(ShapeRadius::Length)
}

/// Find a top-level `at` keyword and split the body around it.
fn split_at_keyword(body: &[Token]) -> Option<(Vec<Token>, Vec<Token>)> {
    let mut depth: u32 = 0;
    for (i, t) in body.iter().enumerate() {
        match t {
            Token::LParen => depth += 1,
            Token::RParen => depth = depth.saturating_sub(1),
            Token::Function(_) => depth += 1,
            Token::Ident(kw) if depth == 0 && kw.eq_ignore_ascii_case("at") => {
                return Some((body[..i].to_vec(), body[i + 1..].to_vec()));
            }
            _ => {}
        }
    }
    None
}

/// Parse a `<basic-shape>` function body (CSS Shapes 1 §5–§8): `name`
/// is the lowercased function name, `body` the tokens between the
/// parens. Used by the generic value parser (so `var()` substitution
/// re-parses through it) and by `clip-path`'s dedicated entry point.
pub(crate) fn parse_basic_shape(name: &str, body: &[Token]) -> Result<ClipPath, ValueError> {
    let body = trim_tokens(body);
    match name {
        "inset" => {
            // `inset(<length-percentage>{1,4} [round <radius>{1,4}]?)`.
            let groups = shape_ws_groups(body);
            let round_at = groups.iter().position(
                |g| matches!(g.as_slice(), [Token::Ident(kw)] if kw.eq_ignore_ascii_case("round")),
            );
            let (off_groups, rad_groups) = match round_at {
                Some(i) => (&groups[..i], Some(&groups[i + 1..])),
                None => (&groups[..], None),
            };
            let mut offsets = Vec::with_capacity(off_groups.len());
            for g in off_groups {
                offsets.push(parse_shape_length(g)?);
            }
            let offsets = expand_four(offsets)?;
            let radii = match rad_groups {
                Some(gs) => {
                    let mut rs = Vec::with_capacity(gs.len());
                    for g in gs {
                        rs.push(parse_shape_length(g)?);
                    }
                    Some(expand_four(rs)?)
                }
                None => None,
            };
            Ok(ClipPath::Shape(Box::new(BasicShape::Inset {
                offsets,
                radii,
            })))
        }
        "circle" => {
            // `circle([<radius>]? [at <position>]?)`.
            let (rad_toks, pos_toks) = match split_at_keyword(body) {
                Some((r, p)) => (r, Some(p)),
                None => (body.to_vec(), None),
            };
            let rad_groups = shape_ws_groups(&rad_toks);
            let radius = match rad_groups.as_slice() {
                // CSS Shapes 1 §6.1: an omitted `circle()` radius
                // defaults to `closest-side` (the inscribed circle).
                [] => ShapeRadius::ClosestSide,
                [g] => parse_shape_radius(g)?,
                _ => return Err(ValueError),
            };
            let (cx, cy) = match pos_toks {
                Some(p) => parse_shape_position(&p)?,
                None => (CssValue::Percentage(50.0), CssValue::Percentage(50.0)),
            };
            Ok(ClipPath::Shape(Box::new(BasicShape::Circle {
                radius,
                cx,
                cy,
            })))
        }
        "ellipse" => {
            // `ellipse([<rx> <ry>]? [at <position>]?)`.
            let (rad_toks, pos_toks) = match split_at_keyword(body) {
                Some((r, p)) => (r, Some(p)),
                None => (body.to_vec(), None),
            };
            let rad_groups = shape_ws_groups(&rad_toks);
            let (rx, ry) = match rad_groups.as_slice() {
                // CSS Shapes 1 §6.2: omitted `ellipse()` radii default
                // to `closest-side` per axis (resolved center-aware in
                // `resolve_clip_path`).
                [] => (ShapeRadius::ClosestSide, ShapeRadius::ClosestSide),
                [a, b] => (parse_shape_radius(a)?, parse_shape_radius(b)?),
                _ => return Err(ValueError),
            };
            let (cx, cy) = match pos_toks {
                Some(p) => parse_shape_position(&p)?,
                None => (CssValue::Percentage(50.0), CssValue::Percentage(50.0)),
            };
            Ok(ClipPath::Shape(Box::new(BasicShape::Ellipse {
                rx,
                ry,
                cx,
                cy,
            })))
        }
        "polygon" => {
            // `polygon([<fill-rule>,]? [<length-percentage>
            // <length-percentage>]#)`.
            let point_groups = split_comma_tokens(body);
            let mut points = Vec::with_capacity(point_groups.len());
            for (i, g) in point_groups.iter().enumerate() {
                // An optional leading fill rule (`nonzero | evenodd`).
                if i == 0
                    && matches!(g.as_slice(), [Token::Ident(kw)]
                        if kw.eq_ignore_ascii_case("nonzero")
                            || kw.eq_ignore_ascii_case("evenodd"))
                {
                    continue;
                }
                let pair = shape_ws_groups(g);
                match pair.as_slice() {
                    [x, y] => points.push((parse_shape_length(x)?, parse_shape_length(y)?)),
                    _ => return Err(ValueError),
                }
            }
            if points.is_empty() {
                return Err(ValueError);
            }
            Ok(ClipPath::Shape(Box::new(BasicShape::Polygon { points })))
        }
        _ => Err(ValueError),
    }
}

/// Split tokens into top-level comma-separated groups (commas inside
/// functions/parens never separate).
fn split_comma_tokens(body: &[Token]) -> Vec<Vec<Token>> {
    let mut groups: Vec<Vec<Token>> = Vec::new();
    let mut current: Vec<Token> = Vec::new();
    let mut depth: u32 = 0;
    for t in body {
        match t {
            Token::LParen => depth += 1,
            Token::RParen => depth = depth.saturating_sub(1),
            Token::Function(_) => depth += 1,
            Token::Comma if depth == 0 => {
                groups.push(std::mem::take(&mut current));
                continue;
            }
            _ => {}
        }
        current.push(t.clone());
    }
    groups.push(current);
    groups
}

/// One function in a [`CssValue::Transform`] list (CSS Transforms 1
/// §8). Arguments are stored as parsed values: lengths/percentages for
/// `translate()` (percentages resolve against the reference box at
/// used-value time), plain numbers for `scale()`, and degrees for
/// `rotate()` (normalized from `deg`/`grad`/`rad`/`turn` at parse
/// time).
#[derive(Clone, Debug, PartialEq)]
pub enum TransformFunction {
    /// `translate(<length-percentage>, <length-percentage>?)` (also
    /// `translatex()` / `translatey()`, which set the other axis to
    /// zero). A single argument leaves `y` as `Number(0.0)`.
    Translate { x: CssValue, y: CssValue },
    /// `scale(<number>, <number>?)` (also `scalex()` / `scaley()`). A
    /// single argument duplicates into `y`.
    Scale { x: CssValue, y: CssValue },
    /// `rotate(<angle>)`, normalized to degrees.
    Rotate { angle_deg: f32 },
    /// Any other syntactically valid transform function, retained
    /// opaque: `matrix()` / `matrix3d()`, `skew()` / `skewx()` /
    /// `skewy()`, `perspective()`, `translate3d()` / `translatez()`,
    /// `scale3d()` / `scalez()`, `rotate3d()` / `rotatex()` /
    /// `rotatey()` / `rotatez()`, and strict-subset functions whose
    /// arguments contain nested functions (e.g.
    /// `translate(calc(10px + 5px))`), which cannot be evaluated until
    /// math evaluation lands. `name` is the lowercased function name;
    /// `raw_args` is the argument body re-serialized from tokens
    /// (whitespace-trimmed) so `getComputedStyle` round-trips it.
    /// Present so a valid-but-unrenderable function counts as
    /// non-`none` for containing-block purposes instead of collapsing
    /// the declaration to `none`.
    Other { name: String, raw_args: String },
}

// ---------------------------------------------------------------------------
// Transform matrix evaluation (CSS Transforms 1 §6-§8, §14-§15).
// ---------------------------------------------------------------------------

/// A 2D affine transformation matrix in CSS `matrix(a, b, c, d, e, f)`
/// form:
///
/// ```text
/// | a c e |
/// | b d f |
/// | 0 0 1 |
/// ```
///
/// A point `(x, y)` maps to `(a*x + c*y + e, b*x + d*y + f)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransformMatrix {
    /// `matrix()`'s `a`: the x-scale / cosine component.
    pub a: f32,
    /// `matrix()`'s `b`: the y-skew / sine component.
    pub b: f32,
    /// `matrix()`'s `c`: the x-skew / negative-sine component.
    pub c: f32,
    /// `matrix()`'s `d`: the y-scale / cosine component.
    pub d: f32,
    /// `matrix()`'s `e`: the x translation, in px.
    pub e: f32,
    /// `matrix()`'s `f`: the y translation, in px.
    pub f: f32,
}

impl TransformMatrix {
    /// The six components of CSS `matrix(a, b, c, d, e, f)`.
    pub fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> Self {
        TransformMatrix { a, b, c, d, e, f }
    }

    /// The identity matrix.
    pub fn identity() -> Self {
        TransformMatrix::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    }

    /// A pure translation by `(tx, ty)` px.
    pub fn translate(tx: f32, ty: f32) -> Self {
        TransformMatrix::new(1.0, 0.0, 0.0, 1.0, tx, ty)
    }

    /// Whether this is exactly the identity matrix.
    pub fn is_identity(&self) -> bool {
        *self == TransformMatrix::identity()
    }

    /// The matrix product `self * rhs`: `rhs` applies to the point first,
    /// then `self`. Composing a `transform` list left-to-right
    /// (`acc = acc.mul(&next)`) therefore applies the rightmost function
    /// first, per CSS Transforms 1 §8.
    pub fn mul(&self, rhs: &TransformMatrix) -> TransformMatrix {
        TransformMatrix::new(
            self.a * rhs.a + self.c * rhs.b,
            self.b * rhs.a + self.d * rhs.b,
            self.a * rhs.c + self.c * rhs.d,
            self.b * rhs.c + self.d * rhs.d,
            self.a * rhs.e + self.c * rhs.f + self.e,
            self.b * rhs.e + self.d * rhs.f + self.f,
        )
    }

    /// Transform a point.
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        )
    }
}

/// Evaluation context for [`TransformFunction::to_matrix`]: the reference
/// box that percentages resolve against, plus the length resolver the
/// caller (layout) supplies.
///
/// Percentages in `translate()` resolve against the reference box at
/// used-value time (CSS Transforms 1 §8: x against the width, y against
/// the height), so the reference size travels with the call instead of
/// living in the parsed value.
pub struct TransformContext<'a> {
    /// Width of the reference box in px: the percentage base for x-axis
    /// lengths (`translate()` x, `transform-origin` x).
    pub reference_width: f32,
    /// Height of the reference box in px: the percentage base for y-axis
    /// lengths.
    pub reference_height: f32,
    /// Resolve a length/percentage [`CssValue`] to px. `pct_base` is the
    /// px value that `100%` maps to (one of the reference-box sides
    /// above). Return `None` when the value cannot be resolved with the
    /// information available (e.g. `em` without font metrics, or an
    /// unevaluated `calc()`); the whole evaluation then fails closed
    /// (returns `None`).
    pub resolve_length: &'a dyn Fn(&CssValue, f32) -> Option<f32>,
}

/// A [`TransformContext::resolve_length`] for callers without font or
/// viewport context: absolute lengths (via [`length_to_px`]),
/// percentages, layout-deferred percentage math
/// ([`CssValue::MathDeferred`], which carries its own `pct`/`px`
/// coefficients), and unitless zero. Relative units (`em`/`rem`/`vw`/…),
/// unevaluated `calc()`, and anything else fail here; pass a richer
/// callback when those must resolve.
pub fn resolve_absolute_length(value: &CssValue, pct_base: f32) -> Option<f32> {
    match value {
        CssValue::Number(n) if *n == 0.0 => Some(0.0),
        CssValue::Percentage(p) => Some(p / 100.0 * pct_base),
        CssValue::MathDeferred { pct, px, .. } => Some(pct / 100.0 * pct_base + px),
        CssValue::Length(n, unit) => match unit {
            LengthUnit::Px
            | LengthUnit::Cm
            | LengthUnit::Mm
            | LengthUnit::In
            | LengthUnit::Pt
            | LengthUnit::Pc
            | LengthUnit::Q => Some(length_to_px(*n, unit, 0.0, 0.0, 0.0, 0.0)),
            // Relative units need font/viewport context the caller owns.
            _ => None,
        },
        _ => None,
    }
}

impl TransformFunction {
    /// Evaluate this function to a 2D [`TransformMatrix`].
    ///
    /// The structural functions (`translate()`/`scale()`/`rotate()` and
    /// the single-axis forms) always evaluate. The opaque functions
    /// (`matrix()`, `skew()`/`skewx()`/`skewy()`, `matrix3d()`, the
    /// `*3d()`/`*z()` forms, `rotatex()`/`rotatey()`/`rotatez()`,
    /// `perspective()`) are evaluated from their retained argument text:
    ///
    /// * `matrix(a, b, c, d, e, f)` evaluates directly (`e`/`f` are px).
    /// * `skew()`/`skewx()`/`skewy()` evaluate to their spec matrices
    ///   (CSS Transforms 1 §15.1).
    /// * 3D functions are flattened by orthographic projection onto the
    ///   z = 0 plane: the 2D matrix is the top-left 2×2 of the 3×3 linear
    ///   part plus the x/y translation. That is exact for 3D rotations
    ///   (including `rotatex()`/`rotatey()`/`rotate3d()`),
    ///   `translate3d()`/`translatez()` (a z-only translation is
    ///   invisible orthographically), `scale3d()`/`scalez()`, and
    ///   `matrix3d()` whose perspective row is `(0, 0, 0, 1)` with no
    ///   z-coupling. Dropping perspective division (a `matrix3d()` with a
    ///   non-trivial perspective row) is the one documented
    ///   approximation; a lone `perspective()` is the identity (it only
    ///   affects child 3D transforms, which are flattened anyway), and a
    ///   zero `rotate3d()` axis is the identity per spec.
    ///
    /// Returns `None` when an argument cannot be parsed or a length
    /// cannot be resolved with `ctx` — fail closed, so the caller treats
    /// the whole `transform` as unrenderable.
    pub fn to_matrix(&self, ctx: &TransformContext) -> Option<TransformMatrix> {
        match self {
            TransformFunction::Translate { x, y } => {
                let tx = (ctx.resolve_length)(x, ctx.reference_width)?;
                let ty = (ctx.resolve_length)(y, ctx.reference_height)?;
                Some(TransformMatrix::translate(tx, ty))
            }
            TransformFunction::Scale { x, y } => {
                let sx = finite_number(x)?;
                let sy = finite_number(y)?;
                Some(TransformMatrix::new(sx, 0.0, 0.0, sy, 0.0, 0.0))
            }
            TransformFunction::Rotate { angle_deg } => {
                let (s, c) = angle_deg.to_radians().sin_cos();
                Some(TransformMatrix::new(c, s, -s, c, 0.0, 0.0))
            }
            TransformFunction::Other { name, raw_args } => {
                eval_opaque_transform(name, raw_args, ctx)
            }
        }
    }
}

/// A finite [`CssValue::Number`]: the only shape `scale()` arguments take
/// (the parser rejects anything else, but fail closed anyway).
fn finite_number(v: &CssValue) -> Option<f32> {
    match v {
        CssValue::Number(n) if n.is_finite() => Some(*n),
        _ => None,
    }
}

/// Compose a `transform` function list into a single matrix, multiplying
/// left-to-right per CSS Transforms 1 §8 — so the rightmost function
/// applies to the point first (see [`TransformMatrix::mul`]). Returns
/// `None` if any function fails to evaluate.
pub fn compose_transforms(
    functions: &[TransformFunction],
    ctx: &TransformContext,
) -> Option<TransformMatrix> {
    let mut acc = TransformMatrix::identity();
    for f in functions {
        acc = acc.mul(&f.to_matrix(ctx)?);
    }
    Some(acc)
}

/// A parsed `transform-origin` (CSS Transforms 1 §14): the 2D point the
/// `transform` matrix pivots around. Keywords are normalized to
/// percentages at parse time (`left` → `0%`, `center` → `50%`, …);
/// lengths, percentages, and math are kept as-is and resolve against the
/// reference box at use time through [`TransformContext`].
#[derive(Clone, Debug, PartialEq)]
pub struct TransformOrigin {
    /// X offset: a length, a percentage of the reference-box width, or
    /// math; keywords normalized to [`CssValue::Percentage`].
    pub x: CssValue,
    /// Y offset: a length, a percentage of the reference-box height, or
    /// math; keywords normalized to [`CssValue::Percentage`].
    pub y: CssValue,
}

impl TransformOrigin {
    /// The initial value, `50% 50%`.
    pub fn initial() -> Self {
        TransformOrigin {
            x: CssValue::Percentage(50.0),
            y: CssValue::Percentage(50.0),
        }
    }

    /// Parse a specified or computed `transform-origin` value: one or two
    /// of `left | center | right | top | bottom`, `<length-percentage>`,
    /// or math (kept opaque for the caller's resolver). One component
    /// sets that axis and centers the other (`left` → `0% 50%`, `top` →
    /// `50% 0%`); with two components the first is x and the second y,
    /// except a leading y-only keyword (`top left`) is swapped to match
    /// the spec's `&&` grammar. Returns `None` for anything outside the
    /// grammar.
    pub fn parse(value: &CssValue) -> Option<Self> {
        fn is_len(v: &CssValue) -> bool {
            matches!(
                v,
                CssValue::Length(..)
                    | CssValue::Percentage(_)
                    | CssValue::Math(_)
                    | CssValue::MathDeferred { .. }
            ) || matches!(v, CssValue::Number(n) if *n == 0.0)
        }
        /// `(x, y)` percentages for a keyword; `center` serves both axes.
        fn keyword_pct(kw: &str) -> (Option<f32>, Option<f32>) {
            match kw.to_ascii_lowercase().as_str() {
                "left" => (Some(0.0), None),
                "right" => (Some(100.0), None),
                "top" => (None, Some(0.0)),
                "bottom" => (None, Some(100.0)),
                "center" => (Some(50.0), Some(50.0)),
                _ => (None, None),
            }
        }
        fn as_x(v: &CssValue) -> Option<CssValue> {
            match v {
                CssValue::Keyword(k) => keyword_pct(k).0.map(CssValue::Percentage),
                v if is_len(v) => Some(v.clone()),
                _ => None,
            }
        }
        fn as_y(v: &CssValue) -> Option<CssValue> {
            match v {
                CssValue::Keyword(k) => keyword_pct(k).1.map(CssValue::Percentage),
                v if is_len(v) => Some(v.clone()),
                _ => None,
            }
        }
        /// A leading y-only keyword (`top`, `bottom`): the spec's `&&`
        /// grammar accepts it before an x component (`top left`).
        fn is_y_only_keyword(v: &CssValue) -> bool {
            matches!(v, CssValue::Keyword(k) if {
                let k = k.to_ascii_lowercase();
                k == "top" || k == "bottom"
            })
        }
        let items: &[CssValue] = match value {
            CssValue::List(items) => items,
            single => std::slice::from_ref(single),
        };
        match items {
            [a] => {
                if let Some(x) = as_x(a) {
                    Some(TransformOrigin {
                        x,
                        y: CssValue::Percentage(50.0),
                    })
                } else {
                    as_y(a).map(|y| TransformOrigin {
                        x: CssValue::Percentage(50.0),
                        y,
                    })
                }
            }
            [a, b] => match (as_x(a), as_y(b)) {
                (Some(x), Some(y)) => Some(TransformOrigin { x, y }),
                // Lenient `top left` / `bottom 10px`: swap a leading
                // y-only keyword with the x component behind it.
                _ if is_y_only_keyword(a) => match (as_x(b), as_y(a)) {
                    (Some(x), Some(y)) => Some(TransformOrigin { x, y }),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }

    /// Pivot `matrix` around this origin: `T(origin) · matrix · T(−origin)`
    /// (CSS Transforms 1 §14). The origin resolves against `ctx`'s
    /// reference box (x against the width, y against the height).
    /// Returns `None` when the origin cannot be resolved.
    pub fn apply(
        &self,
        matrix: &TransformMatrix,
        ctx: &TransformContext,
    ) -> Option<TransformMatrix> {
        let ox = (ctx.resolve_length)(&self.x, ctx.reference_width)?;
        let oy = (ctx.resolve_length)(&self.y, ctx.reference_height)?;
        Some(
            TransformMatrix::translate(ox, oy)
                .mul(matrix)
                .mul(&TransformMatrix::translate(-ox, -oy)),
        )
    }
}

/// Evaluate a computed `transform` property value to a single 2D matrix,
/// pivoted around `origin` (see [`TransformOrigin::apply`]).
///
/// * `none` (keyword) → the identity matrix.
/// * [`CssValue::Transform`] → the composed function list, pivoted.
/// * a [`CssValue::List`] of `Transform`s (e.g. after `var()`
///   substitution merged several) → flattened, composed, pivoted.
/// * anything else → `None` (fail closed).
pub fn evaluate_transform(
    transform: &CssValue,
    origin: &TransformOrigin,
    ctx: &TransformContext,
) -> Option<TransformMatrix> {
    match transform {
        CssValue::Keyword(k) if k.eq_ignore_ascii_case("none") => Some(TransformMatrix::identity()),
        CssValue::Transform(fns) => origin.apply(&compose_transforms(fns, ctx)?, ctx),
        CssValue::List(items) if items.iter().all(|i| matches!(i, CssValue::Transform(_))) => {
            let mut acc = TransformMatrix::identity();
            for fns in items.iter().filter_map(|i| match i {
                CssValue::Transform(fns) => Some(fns),
                _ => None,
            }) {
                acc = acc.mul(&compose_transforms(fns, ctx)?);
            }
            origin.apply(&acc, ctx)
        }
        _ => None,
    }
}

/// Evaluate one opaque [`TransformFunction::Other`] from its retained
/// argument text. See [`TransformFunction::to_matrix`] for the
/// flattening model.
fn eval_opaque_transform(
    name: &str,
    raw_args: &str,
    ctx: &TransformContext,
) -> Option<TransformMatrix> {
    let args = split_transform_args(raw_args);
    match name {
        "matrix" => {
            let [a, b, c, d, e, f] = args.as_slice() else {
                return None;
            };
            Some(TransformMatrix::new(
                parse_css_number(a)?,
                parse_css_number(b)?,
                parse_css_number(c)?,
                parse_css_number(d)?,
                parse_css_number(e)?,
                parse_css_number(f)?,
            ))
        }
        "matrix3d" => {
            // Column-major `matrix3d(a1, …, a16)`; keep the top-left 2×2
            // of the linear part and the x/y translation.
            if args.len() != 16 {
                return None;
            }
            let n: Option<Vec<f32>> = args.iter().map(|s| parse_css_number(s)).collect();
            let n = n?;
            Some(TransformMatrix::new(n[0], n[1], n[4], n[5], n[12], n[13]))
        }
        "skew" => {
            let (ax_deg, ay_deg) = match args.as_slice() {
                [ax] => (parse_css_angle_deg(ax)?, 0.0),
                [ax, ay] => (parse_css_angle_deg(ax)?, parse_css_angle_deg(ay)?),
                _ => return None,
            };
            // CSS Transforms 1 §15.1: `skew(ax, ay)` is
            // `matrix(1, tan(ay), tan(ax), 1, 0, 0)`.
            let (tx, ty) = (ax_deg.to_radians().tan(), ay_deg.to_radians().tan());
            if !tx.is_finite() || !ty.is_finite() {
                return None;
            }
            Some(TransformMatrix::new(1.0, ty, tx, 1.0, 0.0, 0.0))
        }
        "skewx" => {
            let [a] = args.as_slice() else { return None };
            let t = parse_css_angle_deg(a)?.to_radians().tan();
            if !t.is_finite() {
                return None;
            }
            Some(TransformMatrix::new(1.0, 0.0, t, 1.0, 0.0, 0.0))
        }
        "skewy" => {
            let [a] = args.as_slice() else { return None };
            let t = parse_css_angle_deg(a)?.to_radians().tan();
            if !t.is_finite() {
                return None;
            }
            Some(TransformMatrix::new(1.0, t, 0.0, 1.0, 0.0, 0.0))
        }
        "translate3d" => {
            let [tx, ty, tz] = args.as_slice() else {
                return None;
            };
            let tx = resolve_length_arg(tx, ctx.reference_width, ctx)?;
            let ty = resolve_length_arg(ty, ctx.reference_height, ctx)?;
            // `tz` must be a length, but the z offset is dropped under
            // orthographic projection.
            parse_length_value(tz)?;
            Some(TransformMatrix::translate(tx, ty))
        }
        "translatez" => {
            // A pure z translation is invisible under orthographic
            // projection.
            let [tz] = args.as_slice() else { return None };
            parse_length_value(tz)?;
            Some(TransformMatrix::identity())
        }
        "scale3d" => {
            let [sx, sy, sz] = args.as_slice() else {
                return None;
            };
            let (sx, sy) = (parse_css_number(sx)?, parse_css_number(sy)?);
            parse_css_number(sz)?;
            Some(TransformMatrix::new(sx, 0.0, 0.0, sy, 0.0, 0.0))
        }
        "scalez" => {
            let [s] = args.as_slice() else { return None };
            parse_css_number(s)?;
            Some(TransformMatrix::identity())
        }
        "rotate3d" => {
            let [x, y, z, a] = args.as_slice() else {
                return None;
            };
            let (x, y, z) = (
                parse_css_number(x)?,
                parse_css_number(y)?,
                parse_css_number(z)?,
            );
            let theta = parse_css_angle_deg(a)?.to_radians();
            Some(rotate_3d_matrix(x, y, z, theta))
        }
        "rotatex" => {
            let [a] = args.as_slice() else { return None };
            Some(rotate_3d_matrix(
                1.0,
                0.0,
                0.0,
                parse_css_angle_deg(a)?.to_radians(),
            ))
        }
        "rotatey" => {
            let [a] = args.as_slice() else { return None };
            Some(rotate_3d_matrix(
                0.0,
                1.0,
                0.0,
                parse_css_angle_deg(a)?.to_radians(),
            ))
        }
        "rotatez" => {
            let [a] = args.as_slice() else { return None };
            let (s, c) = parse_css_angle_deg(a)?.to_radians().sin_cos();
            Some(TransformMatrix::new(c, s, -s, c, 0.0, 0.0))
        }
        "perspective" => {
            // A lone `perspective()` only affects child 3D transforms
            // (which are flattened anyway): the identity.
            let [d] = args.as_slice() else { return None };
            parse_length_value(d)?;
            Some(TransformMatrix::identity())
        }
        _ => None,
    }
}

/// Flatten a 3D axis-angle rotation to 2D by orthographic projection: the
/// top-left 2×2 of the Rodrigues rotation matrix (the z row is
/// discarded). A zero (or non-finite) axis cannot be normalized: per spec
/// the rotation has no effect, so this is the identity.
fn rotate_3d_matrix(x: f32, y: f32, z: f32, theta_rad: f32) -> TransformMatrix {
    let len = (x * x + y * y + z * z).sqrt();
    if len <= 0.0 || !len.is_finite() {
        return TransformMatrix::identity();
    }
    let (ux, uy, uz) = (x / len, y / len, z / len);
    let (s, c) = theta_rad.sin_cos();
    let t = 1.0 - c;
    TransformMatrix::new(
        c + t * ux * ux,
        t * ux * uy + s * uz,
        t * ux * uy - s * uz,
        c + t * uy * uy,
        0.0,
        0.0,
    )
}

/// Split a retained transform argument body on top-level commas; the body
/// may hold nested functions such as `calc()`.
fn split_transform_args(raw: &str) -> Vec<&str> {
    let mut args = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, ch) in raw.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                args.push(raw[start..i].trim());
                start = i + ch.len_utf8();
            }
            _ => {}
        }
    }
    args.push(raw[start..].trim());
    args
}

/// A bare CSS number (no unit) from retained argument text: finite, else
/// `None`.
fn parse_css_number(s: &str) -> Option<f32> {
    let (n, unit) = split_number_unit(s)?;
    if unit.is_empty() { Some(n) } else { None }
}

/// A CSS angle in degrees from retained argument text: `deg`/`grad`/`rad`/
/// `turn` (case-insensitive), or a unitless zero (the unitless-zero rule,
/// CSS Values 4 §6.2).
pub fn parse_css_angle_deg(s: &str) -> Option<f32> {
    let (n, unit) = split_number_unit(s)?;
    if unit.is_empty() {
        return if n == 0.0 { Some(0.0) } else { None };
    }
    // CSS Values 4 §6.2: 360deg = 400grad = 2πrad = 1turn.
    let deg = match unit.to_ascii_lowercase().as_str() {
        "deg" => n,
        "grad" => n * 360.0 / 400.0,
        "rad" => n * 180.0 / std::f32::consts::PI,
        "turn" => n * 360.0,
        _ => return None,
    };
    Some(deg)
}

/// A CSS `<length-percentage>` from retained argument text, as a
/// [`CssValue`] for the caller's [`TransformContext::resolve_length`].
/// Unitless zero is a valid length; any other unitless number is not.
fn parse_length_value(s: &str) -> Option<CssValue> {
    let (n, unit) = split_number_unit(s)?;
    if unit.is_empty() {
        return if n == 0.0 {
            Some(CssValue::Number(0.0))
        } else {
            None
        };
    }
    if unit == "%" {
        return Some(CssValue::Percentage(n));
    }
    let unit = match unit.to_ascii_lowercase().as_str() {
        "px" => LengthUnit::Px,
        "cm" => LengthUnit::Cm,
        "mm" => LengthUnit::Mm,
        "in" => LengthUnit::In,
        "pt" => LengthUnit::Pt,
        "pc" => LengthUnit::Pc,
        "q" => LengthUnit::Q,
        "em" => LengthUnit::Em,
        "rem" => LengthUnit::Rem,
        "ex" => LengthUnit::Ex,
        "vw" => LengthUnit::Vw,
        "vh" => LengthUnit::Vh,
        "vmin" => LengthUnit::Vmin,
        "vmax" => LengthUnit::Vmax,
        _ => return None,
    };
    Some(CssValue::Length(n, unit))
}

/// Resolve one retained length/percentage argument to px.
fn resolve_length_arg(s: &str, pct_base: f32, ctx: &TransformContext) -> Option<f32> {
    let v = parse_length_value(s)?;
    (ctx.resolve_length)(&v, pct_base)
}

/// Split `s` into a leading number and a trailing unit suffix (letters or
/// `%`): `"10px"` → `(10.0, "px")`, `"50%"` → `(50.0, "%")`, `"2"` →
/// `(2.0, "")`. Returns `None` when the number part is missing or not
/// finite. The split point is the last non-letter, non-`%` character, so
/// scientific notation (`1e3`) keeps its exponent.
fn split_number_unit(s: &str) -> Option<(f32, &str)> {
    let s = s.trim();
    let mut unit_start = 0;
    for (i, c) in s.char_indices() {
        if !(c.is_ascii_alphabetic() || c == '%') {
            unit_start = i + c.len_utf8();
        }
    }
    let (num_str, unit) = s.split_at(unit_start);
    if num_str.is_empty() {
        return None;
    }
    let n: f32 = num_str.parse().ok()?;
    if !n.is_finite() {
        return None;
    }
    Some((n, unit))
}

/// The auto-repetition mode of a [`CssValue::Repeat`]: `repeat()`'s
/// first argument when it is `auto-fill` or `auto-fit` (case-insensitive
/// per CSS, folded the same way [`CssValue::Keyword`]s are).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepeatMode {
    /// `repeat(auto-fill, …)`: repeat as many tracks as fit.
    Fill,
    /// `repeat(auto-fit, …)`: like `auto-fill`, but empty repeated tracks
    /// collapse (they take no space and draw no gap).
    Fit,
}

/// Supported absolute/relative length units.
#[derive(Clone, Debug, PartialEq)]
pub enum LengthUnit {
    Px,
    /// Absolute units, in px per CSS Values 4 §5.2 (`96px = 1in`).
    Cm,
    Mm,
    In,
    Pt,
    Pc,
    Q,
    Em,
    Rem,
    /// The x-height of the font (CSS Values 4 §5.1.2). Evaluated as
    /// 0.5em when font metrics are unavailable.
    Ex,
    Vw,
    Vh,
    Vmin,
    Vmax,
}

/// Supported angle units (CSS Values 4 §6.1: `360deg = 400grad = 2πrad
/// = 1turn`).
///
/// Angles appear in math expressions (`sin(90deg)`) and are normalized
/// to radians at parse time into [`MathExpr::Angle`] leaves, so the
/// `<angle>`/`<number>` type distinction survives evaluation (see
/// [`MathExpr`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AngleUnit {
    Deg,
    Grad,
    Rad,
    Turn,
}

impl AngleUnit {
    /// Convert `value` in this unit to radians.
    pub fn to_radians(self, value: f32) -> f32 {
        match self {
            AngleUnit::Deg => value.to_radians(),
            AngleUnit::Grad => value * std::f32::consts::PI / 200.0,
            AngleUnit::Rad => value,
            AngleUnit::Turn => value * 2.0 * std::f32::consts::PI,
        }
    }
}

/// Convert a `(value, unit)` length to px against the given bases:
/// `em_basis` is the font size `em` resolves against (the element's own
/// computed font size for lengths, the parent's for `font-size` itself),
/// then the root font size (`rem`) and the viewport dimensions
/// (`vw`/`vh`).
///
/// Canonical home of the `LengthUnit -> px` factor table. The CSS math
/// evaluator, layout's style resolution, and paint's display-list builder
/// all resolve through this one function so the three cannot drift.
pub fn length_to_px(
    value: f32,
    unit: &LengthUnit,
    em_basis: f32,
    root_font_size: f32,
    viewport_width: f32,
    viewport_height: f32,
) -> f32 {
    match unit {
        LengthUnit::Px => value,
        // CSS Values 4 §5.2: absolute units are anchored to `96px = 1in`.
        LengthUnit::Cm => value * 96.0 / 2.54,
        LengthUnit::Mm => value * 96.0 / 25.4,
        LengthUnit::In => value * 96.0,
        LengthUnit::Pt => value * 96.0 / 72.0,
        LengthUnit::Pc => value * 16.0,
        LengthUnit::Q => value * 96.0 / 101.6,
        LengthUnit::Em => value * em_basis,
        LengthUnit::Rem => value * root_font_size,
        // `ex` is the font's x-height; without metrics, 0.5em is the
        // CSS-specified fallback (matches `FontMetrics::x_height` default).
        LengthUnit::Ex => value * em_basis * 0.5,
        LengthUnit::Vw => value * viewport_width / 100.0,
        LengthUnit::Vh => value * viewport_height / 100.0,
        LengthUnit::Vmin => value * viewport_width.min(viewport_height) / 100.0,
        LengthUnit::Vmax => value * viewport_width.max(viewport_height) / 100.0,
    }
}

/// An RGBA color. `a` is 0 (transparent) .. 255 (opaque).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

/// A parsed math-function expression tree: `calc()`/`min()`/`max()`/
/// `clamp()` plus the CSS Values 4 math functions (`abs()`, `sign()`,
/// `round()`, `mod()`, `rem()`, `hypot()`, `sqrt()`, `pow()`, `exp()`,
/// `log()`, the trig functions, `if()` and `toggle()`).
///
/// Leaf types follow the CSS calc typing rules loosely: lengths carry their
/// unit, percentages resolve against [`MathContext::percentage_base`],
/// angle dimensions become [`MathExpr::Angle`] leaves holding radians
/// (see [`AngleUnit`]), and bare numbers stay unitless. Mixed-type
/// arithmetic (`1px + 2`, `90deg + 1`) fails at evaluation time.
///
/// Per-function typing follows CSS Values 4 §10 (see each variant's docs
/// for the local deviations: `asin()`/`acos()`/`atan()` return radians
/// as `<number>` rather than the spec's `<angle>`).
#[derive(Clone, Debug, PartialEq)]
pub enum MathExpr {
    Number(f32),
    Percentage(f32),
    Length(f32, LengthUnit),
    /// An angle in radians, from a `deg`/`grad`/`rad`/`turn` dimension
    /// (CSS Values 4 §6.1: `360deg = 400grad = 2πrad = 1turn`). Kept as
    /// a distinct leaf — never folded to [`MathExpr::Number`] — so the
    /// `<angle>`/`<number>` type distinction survives evaluation:
    /// `Angle ± Angle → Angle`, `Angle ×/÷ Number → Angle`, and mixing
    /// an angle with a number or length (`90deg + 1`, `90deg * 1px`) is
    /// invalid math. Evaluates to [`MathValue::Angle`].
    Angle(f32),
    /// The `size` interpolation keyword of `calc-size()`: the resolved
    /// value of the calc-size basis. Only meaningful inside a
    /// `calc-size()` calc-sum (the parser only produces it there);
    /// standalone evaluation fails closed (`None`), exactly like an
    /// unsubstituted `var()` leaf. [`MathExpr::substitute_size`]
    /// replaces it with the evaluated basis before evaluation.
    Size,
    /// `var(--name)` or `var(--name, fallback)` inside a math function.
    /// Substituted by the cascade before evaluation; [`MathExpr::evaluate`]
    /// fails closed (`None`) on a leaf that was never substituted.
    Var(String, Option<Box<MathExpr>>),
    Neg(Box<MathExpr>),
    Add(Box<MathExpr>, Box<MathExpr>),
    Sub(Box<MathExpr>, Box<MathExpr>),
    Mul(Box<MathExpr>, Box<MathExpr>),
    Div(Box<MathExpr>, Box<MathExpr>),
    Min(Vec<MathExpr>),
    Max(Vec<MathExpr>),
    Clamp(Box<MathExpr>, Box<MathExpr>, Box<MathExpr>),
    /// `abs(A)`: absolute value, same type as `A` (CSS Values 4 §10.6).
    Abs(Box<MathExpr>),
    /// `sign(A)`: `-1`/`1`/`0⁺`/`0⁻` as a `<number>`, whatever `A`'s type
    /// (CSS Values 4 §10.6).
    Sign(Box<MathExpr>),
    /// `round(<strategy>?, A, B?)`: `A` rounded to a multiple of `B`
    /// (CSS Values 4 §10.3). `B` defaults to `1` and may only be omitted
    /// for `<number>` `A`; the result keeps `A`'s type.
    Round {
        strategy: RoundStrategy,
        a: Box<MathExpr>,
        b: Option<Box<MathExpr>>,
    },
    /// `mod(A, B)`: remainder with the sign of `B` (CSS Values 4 §10.3).
    /// `A` and `B` must share a type; the result keeps it.
    Mod(Box<MathExpr>, Box<MathExpr>),
    /// `rem(A, B)`: remainder with the sign of `A` (CSS Values 4 §10.3).
    /// Same typing as [`MathExpr::Mod`].
    Rem(Box<MathExpr>, Box<MathExpr>),
    /// `hypot(A, ...)`: `sqrt(sum(A²))` over one or more calculations,
    /// all of a consistent type; the result keeps that type
    /// (CSS Values 4 §10.5).
    Hypot(Vec<MathExpr>),
    /// `sqrt(A)`: square root of a `<number>` (CSS Values 4 §10.5).
    Sqrt(Box<MathExpr>),
    /// `pow(A, B)`: `A` raised to `B`; both must be `<number>`s
    /// (CSS Values 4 §10.5).
    Pow(Box<MathExpr>, Box<MathExpr>),
    /// `exp(A)`: `e` raised to `A`; `A` must be a `<number>`
    /// (CSS Values 4 §10.5).
    Exp(Box<MathExpr>),
    /// `log(A, B?)`: logarithm of `A` base `B` (default `e`); both must
    /// be `<number>`s (CSS Values 4 §10.5).
    Log {
        a: Box<MathExpr>,
        b: Option<Box<MathExpr>>,
    },
    /// `sin(A)`/`cos(A)`/`tan(A)`: `A` must be a `<number>` or an `<angle>`
    /// (CSS Values 4 §10.4). Angle dimensions (`deg`/`grad`/`rad`/`turn`)
    /// parse to [`MathExpr::Angle`] leaves (radians, see [`AngleUnit`]),
    /// and a bare `<number>` is interpreted as radians; the result is a
    /// `<number>`.
    Sin(Box<MathExpr>),
    Cos(Box<MathExpr>),
    Tan(Box<MathExpr>),
    /// `asin(A)`/`acos(A)`/`atan(A)`: `A` must be a `<number>`. Per spec
    /// these return an `<angle>`; the result here is the radian value as
    /// a `<number>` (documented deviation — unlike [`MathExpr::Atan2`],
    /// they do not keep the `<angle>` type).
    Asin(Box<MathExpr>),
    Acos(Box<MathExpr>),
    Atan(Box<MathExpr>),
    /// `atan2(A, B)`: angle of the point `(B, A)`; `A` and `B` must have
    /// a consistent type (both `<number>`, both `<length>`, or both
    /// `<angle>`). Per spec the result is an `<angle>`: radians come
    /// back as [`MathValue::Angle`]. Because angles keep their type
    /// (see [`MathExpr::Angle`]), mixing is invalid: `atan2(90deg, 1)`
    /// is a type error, not a silent radian fold.
    Atan2(Box<MathExpr>, Box<MathExpr>),
    /// `if( <if-condition> : <calc-sum>; ... )` (CSS Values 5 §8.3):
    /// the first branch whose condition is true wins. Only `else` is
    /// evaluable here (see [`IfCondition`]); no true branch means
    /// invalid math (`None`).
    If(Vec<IfBranch>),
    /// `toggle(A, ...)`: picks a value from the list. Per spec the
    /// inherited value selects the *next* entry, but [`MathContext`]
    /// carries no inherited value, so this always resolves to the first
    /// argument (the spec's no-match case). Documented deviation.
    Toggle(Vec<MathExpr>),
}

/// Rounding strategy for [`MathExpr::Round`] (CSS Values 4 §10.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundStrategy {
    /// Ties (exactly between two multiples) choose the multiple toward
    /// `+∞`. The default when the strategy is omitted.
    Nearest,
    /// Choose the multiple toward `+∞`.
    Up,
    /// Choose the multiple toward `-∞`.
    Down,
    /// Choose the multiple closest to zero.
    ToZero,
    /// Snap as a line width (toward whole device pixels, never rounding
    /// a non-zero value to zero). Parsed but not evaluated: device-pixel
    /// snapping needs information [`MathContext`] does not carry, so
    /// evaluation fails closed (`None`).
    LineWidth,
}

/// One `condition : value` branch of [`MathExpr::If`].
#[derive(Clone, Debug, PartialEq)]
pub struct IfBranch {
    pub condition: IfCondition,
    pub value: MathExpr,
}

/// A condition in an [`MathExpr::If`] branch (CSS Values 5 §8.3).
#[derive(Clone, Debug, PartialEq)]
pub enum IfCondition {
    /// The `else` keyword: always true.
    Else,
    /// Any other condition: `supports()`/`media()`/`style()` tests and
    /// their boolean combinations. None are evaluable in a math context
    /// — there are no media features, no computed style to query, and no
    /// feature-support table here — so per CSS Values 5 §3.2 a top-level
    /// unknown condition resolves to false. The raw condition tokens are
    /// kept verbatim so serialization round-trips.
    Unevaluable(Vec<Token>),
}

impl IfCondition {
    /// Evaluate the condition: `else` is true; anything unevaluable is
    /// unknown, and a top-level unknown resolves to false (CSS Values 5
    /// §3.2, Kleene logic).
    fn eval(&self) -> bool {
        matches!(self, IfCondition::Else)
    }
}

/// Maximum nesting depth for math expressions: `calc()`/paren/`var()`
/// nesting past this fails parsing (mirroring `MAX_FUNC_NESTING` in
/// `selector.rs`). Bounds parser and evaluator recursion against
/// adversarial stylesheets such as `calc(` followed by 100k parens.
/// Note: `substitute_math_vars` spends this budget only on actual
/// `var()` expansions (the `Var` arm); pure tree descent (`Add`/`Neg`/
/// `Min`/…) is guarded separately by `MAX_MATH_SUBST_STRUCT_DEPTH`
/// (1024). A calc-wrapped `var()` chain like
/// `--ci: calc(var(--c{i-1}) + 0px)` therefore costs one unit per link,
/// and the practical acyclic-chain limit is the full 64 links (pinned
/// by `calc_var_chain_depth_boundary_pinned` in `cascade.rs`).
pub(crate) const MAX_MATH_DEPTH: u32 = 64;

/// A fully evaluated math expression: a unitless number, a length in px,
/// or an angle in radians.
///
/// The `<angle>` type exists so parse-time angle dimensions stay
/// distinguishable from plain `<number>`s through evaluation (CSS Values
/// 4 §10): `calc(90deg)` evaluates to `Angle(π/2)`, not `Number(π/2)`,
/// so `opacity: calc(90deg)` fails type checking instead of computing an
/// opacity of π/2. No property in this crate takes `<angle>` through
/// `calc()`, so an `Angle` result is invalid at computed-value time (see
/// [`Styler::eval_math_for_prop`](crate::cascade::Styler::eval_math_for_prop)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MathValue {
    Number(f32),
    Length(f32),
    /// An angle in radians.
    Angle(f32),
}

/// Font/viewport metrics that relative units in a [`MathExpr`] resolve
/// against.
#[derive(Clone, Copy, Debug, Default)]
pub struct MathContext {
    /// Element font size in px, for `em`.
    pub font_size: f32,
    /// Root element font size in px, for `rem`.
    pub root_font_size: f32,
    /// Viewport width in px, for `vw`.
    pub viewport_width: f32,
    /// Viewport height in px, for `vh`.
    pub viewport_height: f32,
    /// Base that `%` resolves against (e.g. the containing block extent).
    pub percentage_base: f32,
}

impl MathContext {
    pub fn new(
        font_size: f32,
        root_font_size: f32,
        viewport_width: f32,
        viewport_height: f32,
    ) -> Self {
        Self {
            font_size,
            root_font_size,
            viewport_width,
            viewport_height,
            percentage_base: 0.0,
        }
    }

    pub fn with_percentage_base(mut self, base: f32) -> Self {
        self.percentage_base = base;
        self
    }
}

impl MathExpr {
    /// Evaluate the expression against `ctx`. Lengths come back as px;
    /// percentages resolve against the context's percentage base.
    /// Returns `None` for invalid math: type mismatches, division by zero,
    /// an empty `min()`/`max()`, an unsubstituted `var()` leaf, or
    /// expression nesting past [`MAX_MATH_DEPTH`].
    pub fn evaluate(&self, ctx: &MathContext) -> Option<MathValue> {
        self.eval(ctx, 0)
    }

    /// Whether this expression can evaluate to an `<angle>`-typed value.
    /// Type-based, not syntactic: math functions that consume an angle
    /// and return a `<number>` (`sin()`/`cos()`/`tan()`/`sign()`) never
    /// make this true, because a consumed angle can never leak back out
    /// as angle-typed. The inverse trig functions (`asin()`/`acos()`/
    /// `atan()`/`atan2()`) always return `<angle>`, so they are always
    /// true. `Neg`/`Abs`/`Sqrt`/`Exp` recurse (type-preserving, the last
    /// two conservatively); everything else keeps its recursion.
    /// `var()` never triggers rejection at set time: it is parse-valid
    /// in any property and type-checks at computed-value time (CSS
    /// Custom Properties L1); an unresolvable `var()` with an angle
    /// fallback correctly becomes IACVT there.
    ///
    /// Angles never belong in `<length-percentage>` contexts, so a math
    /// value for which this is true is invalid for e.g. inset properties
    /// at parse time (CSSOM `test_invalid_value` expects the set to be
    /// a no-op).
    pub fn contains_angle(&self) -> bool {
        match self {
            MathExpr::Angle(_) => true,
            MathExpr::Number(_)
            | MathExpr::Percentage(_)
            | MathExpr::Length(_, _)
            | MathExpr::Size => false,
            // var() type-checks at computed-value time, so it is never
            // rejected at set time, even with an angle-typed fallback.
            MathExpr::Var(..) => false,
            // sin()/cos()/tan()/sign() accept <angle> but always return
            // <number>: a consumed angle can never leak out as
            // angle-typed, so these never make the expression
            // angle-typed (CSS Values 4 §10.4).
            MathExpr::Sin(_) | MathExpr::Cos(_) | MathExpr::Tan(_) | MathExpr::Sign(_) => false,
            // asin()/acos()/atan()/atan2() always return <angle>, whatever
            // their operands (CSS Values 4 §10.4).
            MathExpr::Asin(_) | MathExpr::Acos(_) | MathExpr::Atan(_) | MathExpr::Atan2(_, _) => {
                true
            }
            MathExpr::Neg(e) | MathExpr::Abs(e) | MathExpr::Sqrt(e) | MathExpr::Exp(e) => {
                e.contains_angle()
            }
            MathExpr::Add(a, b)
            | MathExpr::Sub(a, b)
            | MathExpr::Mul(a, b)
            | MathExpr::Div(a, b)
            | MathExpr::Mod(a, b)
            | MathExpr::Rem(a, b)
            | MathExpr::Pow(a, b) => a.contains_angle() || b.contains_angle(),
            MathExpr::Min(v) | MathExpr::Max(v) | MathExpr::Hypot(v) | MathExpr::Toggle(v) => {
                v.iter().any(|e| e.contains_angle())
            }
            MathExpr::Clamp(a, b, c) => {
                a.contains_angle() || b.contains_angle() || c.contains_angle()
            }
            MathExpr::Round { a, b, .. } => {
                a.contains_angle() || b.as_ref().is_some_and(|x| x.contains_angle())
            }
            MathExpr::Log { a, b } => {
                a.contains_angle() || b.as_ref().is_some_and(|x| x.contains_angle())
            }
            MathExpr::If(branches) => branches.iter().any(|br| br.value.contains_angle()),
        }
    }

    /// Whether the tree still holds a `var()` leaf needing cascade
    /// substitution. Iterative over the (parser depth-capped) tree.
    /// Depth-capped evaluation: `depth` counts expression nesting so a
    /// hand-built (non-parser) tree cannot overflow the stack either.
    fn eval(&self, ctx: &MathContext, depth: u32) -> Option<MathValue> {
        if depth > MAX_MATH_DEPTH {
            return None;
        }
        let d = depth + 1;
        match self {
            MathExpr::Number(n) => Some(MathValue::Number(*n)),
            MathExpr::Percentage(p) => Some(MathValue::Length(p / 100.0 * ctx.percentage_base)),
            MathExpr::Angle(r) => Some(MathValue::Angle(*r)),
            MathExpr::Length(n, unit) => Some(MathValue::Length(length_to_px(
                *n,
                unit,
                ctx.font_size,
                ctx.root_font_size,
                ctx.viewport_width,
                ctx.viewport_height,
            ))),
            // Unsubstituted `var()`: only the cascade can resolve these;
            // fail closed here. `Size` (the `calc-size()` interpolation
            // keyword) likewise fails closed: only substitution of the
            // resolved basis (see `MathExpr::substitute_size`) makes it
            // evaluable.
            MathExpr::Var(..) | MathExpr::Size => None,
            MathExpr::Neg(e) => Some(e.eval(ctx, d)?.neg()),
            MathExpr::Add(a, b) => MathValue::add(a.eval(ctx, d)?, b.eval(ctx, d)?),
            MathExpr::Sub(a, b) => MathValue::add(a.eval(ctx, d)?, b.eval(ctx, d)?.neg()),
            MathExpr::Mul(a, b) => MathValue::mul(a.eval(ctx, d)?, b.eval(ctx, d)?),
            MathExpr::Div(a, b) => MathValue::div(a.eval(ctx, d)?, b.eval(ctx, d)?),
            MathExpr::Min(items) => MathValue::min_max(items, ctx, true, d),
            MathExpr::Max(items) => MathValue::min_max(items, ctx, false, d),
            MathExpr::Clamp(lo, v, hi) => {
                let (lo, v, hi) = (lo.eval(ctx, d)?, v.eval(ctx, d)?, hi.eval(ctx, d)?);
                // Resolved as max(MIN, min(VAL, MAX)) per CSS Values 4.
                match (lo, v, hi) {
                    (MathValue::Number(lo), MathValue::Number(v), MathValue::Number(hi)) => {
                        Some(MathValue::Number(lo.max(v.min(hi))))
                    }
                    (MathValue::Length(lo), MathValue::Length(v), MathValue::Length(hi)) => {
                        Some(MathValue::Length(lo.max(v.min(hi))))
                    }
                    (MathValue::Angle(lo), MathValue::Angle(v), MathValue::Angle(hi)) => {
                        Some(MathValue::Angle(lo.max(v.min(hi))))
                    }
                    _ => None,
                }
            }
            MathExpr::Abs(e) => {
                let v = e.eval(ctx, d)?;
                // `abs()` keeps its input's type (CSS Values 4 §10.6).
                Some(match v {
                    MathValue::Number(n) => MathValue::Number(n.abs()),
                    MathValue::Length(l) => MathValue::Length(l.abs()),
                    MathValue::Angle(r) => MathValue::Angle(r.abs()),
                })
            }
            MathExpr::Sign(e) => {
                let n = match e.eval(ctx, d)? {
                    MathValue::Number(n) => n,
                    MathValue::Length(l) => l,
                    MathValue::Angle(r) => r,
                };
                // `sign()` always returns a <number> (CSS Values 4 §10.6).
                // ±0 is neither positive nor negative, so it survives as
                // 0⁺/0⁻ here.
                Some(MathValue::Number(if n > 0.0 {
                    1.0
                } else if n < 0.0 {
                    -1.0
                } else {
                    n
                }))
            }
            MathExpr::Round { strategy, a, b } => {
                let (ax, kind) = math_scalar_and_kind(a.eval(ctx, d)?);
                let bx = match b {
                    Some(b) => {
                        let (bx, b_kind) = math_scalar_and_kind(b.eval(ctx, d)?);
                        // A and B must have a consistent type (spec §10.3).
                        if b_kind != kind {
                            return None;
                        }
                        bx
                    }
                    // B may be omitted only for <number> A, defaulting to
                    // 1 (spec §10.3); otherwise the function is invalid.
                    None => {
                        if kind != MathKind::Number {
                            return None;
                        }
                        1.0
                    }
                };
                // A zero B makes the result NaN (spec §10.3.1): invalid.
                if bx == 0.0 {
                    return None;
                }
                let q = ax / bx;
                let r = match strategy {
                    RoundStrategy::Nearest => {
                        // Ties choose the multiple toward +∞ (spec §10.3).
                        // `f32::round` rounds ties away from zero, which
                        // disagrees for negative quotients, so detect the
                        // exact halfway case explicitly.
                        let lo = q.floor();
                        if q - lo == 0.5 { lo + 1.0 } else { q.round() }
                    }
                    RoundStrategy::Up => q.ceil(),
                    RoundStrategy::Down => q.floor(),
                    RoundStrategy::ToZero => q.trunc(),
                    // Line-width snapping needs a device pixel, which
                    // `MathContext` does not carry: fail closed.
                    RoundStrategy::LineWidth => return None,
                };
                let v = r * bx;
                if !v.is_finite() {
                    return None;
                }
                Some(math_value_of_kind(v, kind))
            }
            MathExpr::Mod(a, b) => math_stepped(a.eval(ctx, d)?, b.eval(ctx, d)?, true),
            MathExpr::Rem(a, b) => math_stepped(a.eval(ctx, d)?, b.eval(ctx, d)?, false),
            MathExpr::Hypot(items) => {
                // All arguments must share one type; the result keeps it
                // (spec §10.5). An empty list cannot parse, but fail closed
                // anyway for hand-built trees.
                let mut sum = 0.0f32;
                let mut kind = MathKind::Number;
                let mut first = true;
                for item in items {
                    let (x, k) = math_scalar_and_kind(item.eval(ctx, d)?);
                    if first {
                        kind = k;
                        first = false;
                    } else if k != kind {
                        return None;
                    }
                    sum += x * x;
                }
                if first {
                    return None;
                }
                let v = sum.sqrt();
                if !v.is_finite() {
                    return None;
                }
                Some(math_value_of_kind(v, kind))
            }
            MathExpr::Sqrt(e) => {
                // `sqrt()` takes a <number> only (spec §10.5); a negative
                // argument is NaN: invalid.
                let MathValue::Number(n) = e.eval(ctx, d)? else {
                    return None;
                };
                let v = n.sqrt();
                if v.is_finite() {
                    Some(MathValue::Number(v))
                } else {
                    None
                }
            }
            MathExpr::Pow(a, b) => {
                // Both arguments must be <number>s (spec §10.5).
                let (MathValue::Number(x), MathValue::Number(y)) =
                    (a.eval(ctx, d)?, b.eval(ctx, d)?)
                else {
                    return None;
                };
                let v = x.powf(y);
                if v.is_finite() {
                    Some(MathValue::Number(v))
                } else {
                    None
                }
            }
            MathExpr::Exp(e) => {
                let MathValue::Number(n) = e.eval(ctx, d)? else {
                    return None;
                };
                let v = n.exp();
                if v.is_finite() {
                    Some(MathValue::Number(v))
                } else {
                    None
                }
            }
            MathExpr::Log { a, b } => {
                // Both arguments must be <number>s; the base defaults to
                // e (spec §10.5). A base of 1 or ≤ 0, a negative
                // argument, or a zero argument (→ -∞) are all NaN per
                // §10.5.1: invalid here.
                let MathValue::Number(x) = a.eval(ctx, d)? else {
                    return None;
                };
                if x <= 0.0 {
                    return None;
                }
                let v = match b {
                    Some(b) => {
                        let MathValue::Number(bb) = b.eval(ctx, d)? else {
                            return None;
                        };
                        if bb.is_nan() || bb <= 0.0 || bb == 1.0 {
                            return None;
                        }
                        x.ln() / bb.ln()
                    }
                    // log(A) is the natural logarithm, computed directly
                    // rather than as ln(A)/ln(e) to avoid rounding the
                    // base.
                    None => x.ln(),
                };
                if v.is_finite() {
                    Some(MathValue::Number(v))
                } else {
                    None
                }
            }
            MathExpr::Sin(e) => math_trig(e.eval(ctx, d)?, f32::sin),
            MathExpr::Cos(e) => math_trig(e.eval(ctx, d)?, f32::cos),
            MathExpr::Tan(e) => math_trig(e.eval(ctx, d)?, f32::tan),
            MathExpr::Asin(e) => math_inverse_trig(e.eval(ctx, d)?, f32::asin),
            MathExpr::Acos(e) => math_inverse_trig(e.eval(ctx, d)?, f32::acos),
            MathExpr::Atan(e) => math_inverse_trig(e.eval(ctx, d)?, f32::atan),
            MathExpr::Atan2(y, x) => {
                // A and B must have a consistent type (spec §10.4): both
                // `<number>`, both `<length>`, or both `<angle>`. Per
                // spec the result is an `<angle>`; radians come back as
                // `MathValue::Angle`, so mixing (`atan2(90deg, 1)`) is a
                // type error rather than a silent radian fold.
                let (yn, y_kind) = math_scalar_and_kind(y.eval(ctx, d)?);
                let (xn, x_kind) = math_scalar_and_kind(x.eval(ctx, d)?);
                if y_kind != x_kind || !yn.is_finite() || !xn.is_finite() {
                    return None;
                }
                let v = yn.atan2(xn);
                if v.is_finite() {
                    Some(MathValue::Angle(v))
                } else {
                    None
                }
            }
            MathExpr::If(branches) => {
                // First true branch wins (CSS Values 5 §8.3); no true
                // branch is the guaranteed-invalid value: invalid math.
                for branch in branches {
                    if branch.condition.eval() {
                        return branch.value.eval(ctx, d);
                    }
                }
                None
            }
            MathExpr::Toggle(items) => {
                // Without an inherited value (unavailable in
                // `MathContext`) the spec's no-match case applies:
                // resolve to the first argument. An empty list cannot
                // parse; fail closed for hand-built trees.
                items.first()?.eval(ctx, d)
            }
        }
    }

    /// Whether the tree contains a [`MathExpr::Percentage`] leaf anywhere.
    ///
    /// This is the parse-time *cyclic-percentage detection* per CSS Values 4
    /// §10.9 ("A value *contains a percentage* if its type is
    /// «["percent" → 1]», or its type's percent hint is non-null") and
    /// css-sizing-3 §5.2.1 ("an expression containing a percentage (such as
    /// 10% or `calc(10px + 0%)`)"): percentage-containment is *syntactic*;
    /// cyclic-ness is containment plus an indefinite base, decided by
    /// layout's cyclic-ness gates. Any percentage leaf makes the whole
    /// expression cyclic-capable — even one that folds to zero
    /// (`calc(10px + 0%)`), because the base it resolves against may itself
    /// depend on the value being computed.
    ///
    /// The cascade uses this to split math evaluation: percentage-free
    /// math evaluates in the cascade, but math containing `%` (outside
    /// `font-size`/`line-height`, whose base is font metrics) defers to
    /// layout as [`CssValue::MathDeferred`] coefficients, carrying the
    /// syntactic percentage record in its `had_pct` bit so layout's
    /// cyclic-ness gates (which treat cyclic percentages as `auto`/the
    /// initial value) see it. Parse-time grammar validators (e.g.
    /// [`Shadow::is_shadow_length`]) use it to reject percentage-bearing
    /// math where percentages have no base to resolve against.
    ///
    /// Note: detection is syntactic. A [`MathExpr::Var`] leaf may
    /// substitute to a percentage at computed-value time, so the cascade
    /// re-checks after substitution (see `Styler::eval_math_for_prop`).
    /// Iterative, so hand-built deep trees cannot overflow the stack.
    pub(crate) fn contains_percentage(&self) -> bool {
        let mut stack = vec![self];
        while let Some(e) = stack.pop() {
            match e {
                MathExpr::Percentage(_) => return true,
                MathExpr::Number(_)
                | MathExpr::Angle(_)
                | MathExpr::Length(..)
                | MathExpr::Var(..)
                | MathExpr::Size => {}
                MathExpr::Neg(x) => stack.push(x),
                MathExpr::Abs(x)
                | MathExpr::Sign(x)
                | MathExpr::Sqrt(x)
                | MathExpr::Exp(x)
                | MathExpr::Sin(x)
                | MathExpr::Cos(x)
                | MathExpr::Tan(x)
                | MathExpr::Asin(x)
                | MathExpr::Acos(x)
                | MathExpr::Atan(x) => stack.push(x),
                MathExpr::Add(a, b)
                | MathExpr::Sub(a, b)
                | MathExpr::Mul(a, b)
                | MathExpr::Div(a, b)
                | MathExpr::Mod(a, b)
                | MathExpr::Rem(a, b)
                | MathExpr::Pow(a, b)
                | MathExpr::Atan2(a, b) => {
                    stack.push(a);
                    stack.push(b);
                }
                MathExpr::Round { a, b, .. } | MathExpr::Log { a, b } => {
                    stack.push(a);
                    if let Some(b) = b {
                        stack.push(b);
                    }
                }
                MathExpr::Min(items) | MathExpr::Max(items) | MathExpr::Hypot(items) => {
                    stack.extend(items.iter())
                }
                MathExpr::Toggle(items) => stack.extend(items.iter()),
                MathExpr::If(branches) => stack.extend(branches.iter().map(|b| &b.value)),
                MathExpr::Clamp(a, b, c) => {
                    stack.push(a);
                    stack.push(b);
                    stack.push(c);
                }
            }
        }
        false
    }

    /// Whether the tree contains a [`MathExpr::Var`] leaf anywhere.
    /// The cascade's deferred-shorthand gate (`value_contains_var`) uses
    /// this: a `var()` nested inside `calc()`/`min()`/`max()`/`clamp()`
    /// still forces the declaration down the substitution path, so e.g.
    /// `font: 12px/calc(var(--lh)) serif` expands after substitution
    /// instead of being dropped as an unexpanded shorthand. Iterative,
    /// so hand-built deep trees cannot overflow the stack.
    pub(crate) fn contains_var(&self) -> bool {
        let mut stack = vec![self];
        while let Some(e) = stack.pop() {
            match e {
                MathExpr::Var(..) => return true,
                MathExpr::Number(_)
                | MathExpr::Angle(_)
                | MathExpr::Percentage(_)
                | MathExpr::Length(..)
                | MathExpr::Size => {}
                MathExpr::Neg(x) => stack.push(x),
                MathExpr::Abs(x)
                | MathExpr::Sign(x)
                | MathExpr::Sqrt(x)
                | MathExpr::Exp(x)
                | MathExpr::Sin(x)
                | MathExpr::Cos(x)
                | MathExpr::Tan(x)
                | MathExpr::Asin(x)
                | MathExpr::Acos(x)
                | MathExpr::Atan(x) => stack.push(x),
                MathExpr::Add(a, b)
                | MathExpr::Sub(a, b)
                | MathExpr::Mul(a, b)
                | MathExpr::Div(a, b)
                | MathExpr::Mod(a, b)
                | MathExpr::Rem(a, b)
                | MathExpr::Pow(a, b)
                | MathExpr::Atan2(a, b) => {
                    stack.push(a);
                    stack.push(b);
                }
                MathExpr::Round { a, b, .. } | MathExpr::Log { a, b } => {
                    stack.push(a);
                    if let Some(b) = b {
                        stack.push(b);
                    }
                }
                MathExpr::Min(items) | MathExpr::Max(items) | MathExpr::Hypot(items) => {
                    stack.extend(items.iter())
                }
                MathExpr::Toggle(items) => stack.extend(items.iter()),
                MathExpr::If(branches) => stack.extend(branches.iter().map(|b| &b.value)),
                MathExpr::Clamp(a, b, c) => {
                    stack.push(a);
                    stack.push(b);
                    stack.push(c);
                }
            }
        }
        false
    }

    /// Whether the tree is affine in its percentage leaves: `calc()`-style
    /// `Neg`/`Add`/`Sub`, and `Mul`/`Div` where at most one side carries
    /// percentages (the other is percentage-free, e.g. `2 * 50%`).
    /// `min()`/`max()`/`clamp()` are never affine, and neither is a `%`
    /// times/divided-by a `%` (invalid CSS anyway).
    ///
    /// The cascade uses this for the evaluation split: an affine tree with
    /// percentages defers to layout as two coefficients
    /// (`value = pct / 100 * base + px`, recovered by evaluating at base 0
    /// and base 100), because every font/viewport-relative leaf
    /// contributes equally to both evaluations and cancels out of the
    /// percentage coefficient. Call only after `var()` substitution;
    /// `Var` leaves report non-affine.
    pub(crate) fn is_affine_in_percentage(&self) -> bool {
        match self {
            MathExpr::Number(_)
            | MathExpr::Angle(_)
            | MathExpr::Percentage(_)
            | MathExpr::Length(..) => true,
            // `Var` leaves report non-affine (call only after `var()`
            // substitution); `Size` likewise stands for an unknown basis
            // that may itself be percentage-bearing, so it cannot claim
            // affinity either. After `substitute_size` no `Size` remains.
            MathExpr::Var(..) | MathExpr::Size => false,
            MathExpr::Neg(e) => e.is_affine_in_percentage(),
            MathExpr::Add(a, b) | MathExpr::Sub(a, b) => {
                a.is_affine_in_percentage() && b.is_affine_in_percentage()
            }
            MathExpr::Mul(a, b) => {
                (a.is_affine_in_percentage() && !b.contains_percentage())
                    || (b.is_affine_in_percentage() && !a.contains_percentage())
            }
            // Division is affine only when the denominator is
            // percentage-free: `x / 50%` is `x / (0.5 * base)`, not
            // linear in the base.
            MathExpr::Div(a, b) => a.is_affine_in_percentage() && !b.contains_percentage(),
            // None of the Values 4 math functions are affine in `%`:
            // `abs()`, `sign()`, `round()`, `mod()`/`rem()`, `hypot()`,
            // `sqrt()`, `pow()`, `exp()`, `log()`, and the trig functions
            // are all non-linear (or step functions) in their arguments.
            MathExpr::Abs(_)
            | MathExpr::Sign(_)
            | MathExpr::Round { .. }
            | MathExpr::Mod(..)
            | MathExpr::Rem(..)
            | MathExpr::Hypot(_)
            | MathExpr::Sqrt(_)
            | MathExpr::Pow(..)
            | MathExpr::Exp(_)
            | MathExpr::Log { .. }
            | MathExpr::Sin(_)
            | MathExpr::Cos(_)
            | MathExpr::Tan(_)
            | MathExpr::Asin(_)
            | MathExpr::Acos(_)
            | MathExpr::Atan(_)
            | MathExpr::Atan2(..) => false,
            // `if()` picks a branch on a condition that never depends on
            // the percentage base, so it is affine exactly when every
            // branch value is. `toggle()` degrades to its first argument
            // here, but cycling is a non-math selection: stay
            // conservative and report non-affine.
            MathExpr::If(branches) => branches.iter().all(|b| b.value.is_affine_in_percentage()),
            MathExpr::Toggle(_) => false,
            MathExpr::Min(_) | MathExpr::Max(_) | MathExpr::Clamp(..) => false,
        }
    }

    /// Extract `(pct, px)` with `value = pct / 100 * percentage_base + px`
    /// for a percentage-affine tree, via evaluations at percentage bases 0
    /// and 100 (font/viewport leaves cancel out of the difference).
    /// Returns `None` when the tree is not affine or evaluation fails
    /// (e.g. division by zero). Call only after `var()` substitution.
    pub(crate) fn affine_coefficients(&self, ctx: &MathContext) -> Option<(f32, f32)> {
        if !self.is_affine_in_percentage() {
            return None;
        }
        let base0 = self.evaluate(&ctx.with_percentage_base(0.0));
        let base100 = self.evaluate(&ctx.with_percentage_base(100.0));
        match (base0, base100) {
            // `%` always evaluates to `Length` here; any other shape is a
            // defensive invalid.
            (Some(MathValue::Length(a)), Some(MathValue::Length(b))) => Some((b - a, a)),
            _ => None,
        }
    }

    /// Replace every [`MathExpr::Size`] leaf (the `calc-size()`
    /// interpolation keyword) with a clone of `basis`. Returns `None`
    /// when the substitution would exceed [`MAX_MATH_DEPTH`] nesting.
    /// Recursive, depth-capped like [`MathExpr::evaluate`]: the parser
    /// already caps tree depth at [`MAX_MATH_DEPTH`], and the counter
    /// here guards hand-built trees too.
    pub fn substitute_size(&self, basis: &MathExpr) -> Option<MathExpr> {
        self.substitute_size_depth(basis, 0)
    }

    fn substitute_size_depth(&self, basis: &MathExpr, depth: u32) -> Option<MathExpr> {
        if depth > MAX_MATH_DEPTH {
            return None;
        }
        let d = depth + 1;
        match self {
            MathExpr::Size => Some(basis.clone()),
            MathExpr::Number(_)
            | MathExpr::Angle(_)
            | MathExpr::Percentage(_)
            | MathExpr::Length(..)
            | MathExpr::Var(..) => Some(self.clone()),
            MathExpr::Neg(e) => e
                .substitute_size_depth(basis, d)
                .map(|e| MathExpr::Neg(Box::new(e))),
            MathExpr::Add(a, b) => Some(MathExpr::Add(
                Box::new(a.substitute_size_depth(basis, d)?),
                Box::new(b.substitute_size_depth(basis, d)?),
            )),
            MathExpr::Sub(a, b) => Some(MathExpr::Sub(
                Box::new(a.substitute_size_depth(basis, d)?),
                Box::new(b.substitute_size_depth(basis, d)?),
            )),
            MathExpr::Mul(a, b) => Some(MathExpr::Mul(
                Box::new(a.substitute_size_depth(basis, d)?),
                Box::new(b.substitute_size_depth(basis, d)?),
            )),
            MathExpr::Div(a, b) => Some(MathExpr::Div(
                Box::new(a.substitute_size_depth(basis, d)?),
                Box::new(b.substitute_size_depth(basis, d)?),
            )),
            MathExpr::Min(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(item.substitute_size_depth(basis, d)?);
                }
                Some(MathExpr::Min(out))
            }
            MathExpr::Max(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(item.substitute_size_depth(basis, d)?);
                }
                Some(MathExpr::Max(out))
            }
            MathExpr::Clamp(a, b, c) => Some(MathExpr::Clamp(
                Box::new(a.substitute_size_depth(basis, d)?),
                Box::new(b.substitute_size_depth(basis, d)?),
                Box::new(c.substitute_size_depth(basis, d)?),
            )),
            MathExpr::Abs(e) => e
                .substitute_size_depth(basis, d)
                .map(|e| MathExpr::Abs(Box::new(e))),
            MathExpr::Sign(e) => e
                .substitute_size_depth(basis, d)
                .map(|e| MathExpr::Sign(Box::new(e))),
            MathExpr::Sqrt(e) => e
                .substitute_size_depth(basis, d)
                .map(|e| MathExpr::Sqrt(Box::new(e))),
            MathExpr::Exp(e) => e
                .substitute_size_depth(basis, d)
                .map(|e| MathExpr::Exp(Box::new(e))),
            MathExpr::Sin(e) => e
                .substitute_size_depth(basis, d)
                .map(|e| MathExpr::Sin(Box::new(e))),
            MathExpr::Cos(e) => e
                .substitute_size_depth(basis, d)
                .map(|e| MathExpr::Cos(Box::new(e))),
            MathExpr::Tan(e) => e
                .substitute_size_depth(basis, d)
                .map(|e| MathExpr::Tan(Box::new(e))),
            MathExpr::Asin(e) => e
                .substitute_size_depth(basis, d)
                .map(|e| MathExpr::Asin(Box::new(e))),
            MathExpr::Acos(e) => e
                .substitute_size_depth(basis, d)
                .map(|e| MathExpr::Acos(Box::new(e))),
            MathExpr::Atan(e) => e
                .substitute_size_depth(basis, d)
                .map(|e| MathExpr::Atan(Box::new(e))),
            MathExpr::Mod(a, b) => Some(MathExpr::Mod(
                Box::new(a.substitute_size_depth(basis, d)?),
                Box::new(b.substitute_size_depth(basis, d)?),
            )),
            MathExpr::Rem(a, b) => Some(MathExpr::Rem(
                Box::new(a.substitute_size_depth(basis, d)?),
                Box::new(b.substitute_size_depth(basis, d)?),
            )),
            MathExpr::Pow(a, b) => Some(MathExpr::Pow(
                Box::new(a.substitute_size_depth(basis, d)?),
                Box::new(b.substitute_size_depth(basis, d)?),
            )),
            MathExpr::Atan2(a, b) => Some(MathExpr::Atan2(
                Box::new(a.substitute_size_depth(basis, d)?),
                Box::new(b.substitute_size_depth(basis, d)?),
            )),
            MathExpr::Round { strategy, a, b } => Some(MathExpr::Round {
                strategy: *strategy,
                a: Box::new(a.substitute_size_depth(basis, d)?),
                b: match b {
                    Some(b) => Some(Box::new(b.substitute_size_depth(basis, d)?)),
                    None => None,
                },
            }),
            MathExpr::Log { a, b } => Some(MathExpr::Log {
                a: Box::new(a.substitute_size_depth(basis, d)?),
                b: match b {
                    Some(b) => Some(Box::new(b.substitute_size_depth(basis, d)?)),
                    None => None,
                },
            }),
            MathExpr::Hypot(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(item.substitute_size_depth(basis, d)?);
                }
                Some(MathExpr::Hypot(out))
            }
            MathExpr::Toggle(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(item.substitute_size_depth(basis, d)?);
                }
                Some(MathExpr::Toggle(out))
            }
            MathExpr::If(branches) => {
                let mut out = Vec::with_capacity(branches.len());
                for branch in branches {
                    out.push(IfBranch {
                        condition: branch.condition.clone(),
                        value: branch.value.substitute_size_depth(basis, d)?,
                    });
                }
                Some(MathExpr::If(out))
            }
        }
    }
}

/// The `<calc-size-basis>` of a `calc-size()` (CSS Values 4 §7): either a
/// sizing keyword, which resolves against the box's content and needs
/// layout to measure, or a definite length/percentage/calc, which the
/// CSS crate evaluates itself.
#[derive(Clone, Debug, PartialEq)]
pub enum CalcSizeBasis {
    /// `auto`: the automatic size.
    Auto,
    /// `max-content`: the max-content intrinsic size.
    MaxContent,
    /// `min-content`: the min-content intrinsic size.
    MinContent,
    /// `fit-content`: the fit-content size.
    FitContent,
    /// `stretch` (and its `available` alias): fill the available space.
    Stretch,
    /// A definite `<length>` basis.
    Length(f32, LengthUnit),
    /// A definite `<percentage>` basis (resolves against the property's
    /// percentage base, like any other percentage).
    Percentage(f32),
    /// A `calc()`/`min()`/`max()`/`clamp()` basis (evaluated by the
    /// cascade like any other math value).
    Math(MathExpr),
    /// A `var()` reference in basis position. Per CSS Values 4,
    /// substitution precedes grammar checking, so the reference is
    /// preserved verbatim (name plus raw fallback tokens) and the
    /// cascade substitutes it before basis validation (see
    /// `Styler::eval_calc_size_for_prop` in cascade.rs). The fallback
    /// cannot be parsed eagerly: it may be a sizing keyword
    /// (`var(--x, max-content)`), which is not a math leaf.
    Var {
        /// Custom property name, `--`-prefixed.
        name: String,
        /// Raw fallback token stream, if `var()` had a non-empty one;
        /// checked against the basis grammar only when the property is
        /// undefined (or not a valid basis) at computed-value time.
        fallback: Option<Vec<Token>>,
    },
}

impl CalcSizeBasis {
    /// Whether the basis resolves without content measurement: lengths,
    /// percentages, and math. Keyword bases (`auto`, `max-content`,
    /// `min-content`, `fit-content`, `stretch`) need layout. A `var()`
    /// basis is not definite until the cascade substitutes it.
    pub fn is_definite(&self) -> bool {
        matches!(
            self,
            CalcSizeBasis::Length(..) | CalcSizeBasis::Percentage(_) | CalcSizeBasis::Math(_)
        )
    }

    /// The sizing keyword for keyword bases (`"auto"`, `"max-content"`,
    /// `"min-content"`, `"fit-content"`, `"stretch"`); `None` for
    /// definite bases and for `var()` bases, which only the cascade can
    /// resolve. Lets layout ask "is this `auto`/`max-content`
    /// based" without matching the enum.
    pub fn keyword(&self) -> Option<&'static str> {
        match self {
            CalcSizeBasis::Auto => Some("auto"),
            CalcSizeBasis::MaxContent => Some("max-content"),
            CalcSizeBasis::MinContent => Some("min-content"),
            CalcSizeBasis::FitContent => Some("fit-content"),
            CalcSizeBasis::Stretch => Some("stretch"),
            CalcSizeBasis::Length(..)
            | CalcSizeBasis::Percentage(_)
            | CalcSizeBasis::Math(_)
            | CalcSizeBasis::Var { .. } => None,
        }
    }

    /// The basis as a math expression, for substituting into the
    /// calc-sum's `size` leaves via [`MathExpr::substitute_size`];
    /// `None` for keyword bases, which only layout can resolve, and for
    /// `var()` bases, which the cascade substitutes first.
    pub fn as_math_expr(&self) -> Option<MathExpr> {
        match self {
            CalcSizeBasis::Length(n, unit) => Some(MathExpr::Length(*n, unit.clone())),
            CalcSizeBasis::Percentage(p) => Some(MathExpr::Percentage(*p)),
            CalcSizeBasis::Math(e) => Some(e.clone()),
            CalcSizeBasis::Auto
            | CalcSizeBasis::MaxContent
            | CalcSizeBasis::MinContent
            | CalcSizeBasis::FitContent
            | CalcSizeBasis::Stretch
            | CalcSizeBasis::Var { .. } => None,
        }
    }

    /// Serialize the basis back to CSS source text: sizing keywords
    /// verbatim, lengths/percentages with units, math wrapped in
    /// `calc()`, and `var()` references (with their raw fallback)
    /// verbatim, so the output re-parses to an equal basis.
    fn to_css(&self) -> String {
        match self {
            CalcSizeBasis::Auto => "auto".to_owned(),
            CalcSizeBasis::MaxContent => "max-content".to_owned(),
            CalcSizeBasis::MinContent => "min-content".to_owned(),
            CalcSizeBasis::FitContent => "fit-content".to_owned(),
            CalcSizeBasis::Stretch => "stretch".to_owned(),
            CalcSizeBasis::Length(n, unit) => format!("{n}{}", length_unit_suffix(unit)),
            CalcSizeBasis::Percentage(p) => format!("{p}%"),
            CalcSizeBasis::Math(e) => format!("calc({})", e.to_css()),
            CalcSizeBasis::Var { name, fallback } => match fallback {
                Some(fb) => format!("var({name}, {})", tokens_to_css(fb)),
                None => format!("var({name})"),
            },
        }
    }
}

/// The CSS source suffix for a [`LengthUnit`] (`Px` -> `"px"`, ...).
fn length_unit_suffix(unit: &LengthUnit) -> &'static str {
    match unit {
        LengthUnit::Px => "px",
        LengthUnit::Cm => "cm",
        LengthUnit::Mm => "mm",
        LengthUnit::In => "in",
        LengthUnit::Pt => "pt",
        LengthUnit::Pc => "pc",
        LengthUnit::Q => "q",
        LengthUnit::Em => "em",
        LengthUnit::Rem => "rem",
        LengthUnit::Ex => "ex",
        LengthUnit::Vw => "vw",
        LengthUnit::Vh => "vh",
        LengthUnit::Vmin => "vmin",
        LengthUnit::Vmax => "vmax",
    }
}

impl MathExpr {
    /// Serialize the expression tree back to CSS source text, as the
    /// body of a `calc()`: `size` leaves (only valid inside a
    /// `calc-size()` calc-sum) and `var()` leaves round-trip verbatim,
    /// operators keep CSS spacing, and nested sums/products
    /// parenthesize only where precedence demands it, so the output
    /// re-parses to an equal tree.
    fn to_css(&self) -> String {
        self.to_css_prec(0)
    }

    /// `parent` is the precedence of the enclosing operator: a child
    /// with strictly lower precedence parenthesizes. Sums and
    /// differences are precedence 1, products and quotients 2,
    /// everything else 3.
    fn to_css_prec(&self, parent: u8) -> String {
        let paren = |s: String, prec: u8| {
            if prec < parent { format!("({s})") } else { s }
        };
        match self {
            MathExpr::Number(n) => format!("{n}"),
            MathExpr::Percentage(p) => format!("{p}%"),
            // Angles round-trip as radians: the leaf already holds the
            // radian measure, so `90deg` serializes as its `rad`
            // equivalent and re-parses to an equal tree.
            MathExpr::Angle(r) => format!("{r}rad"),
            MathExpr::Length(n, unit) => format!("{n}{}", length_unit_suffix(unit)),
            MathExpr::Size => "size".to_owned(),
            MathExpr::Var(name, None) => format!("var({name})"),
            MathExpr::Var(name, Some(fallback)) => {
                format!("var({name}, {})", fallback.to_css())
            }
            // A negated negation needs explicit parens: `--x` would
            // tokenize as an identifier, not two unary minuses.
            MathExpr::Neg(e) => match e.as_ref() {
                MathExpr::Neg(_) => format!("-({})", e.to_css()),
                _ => format!("-{}", e.to_css_prec(4)),
            },
            MathExpr::Add(a, b) => paren(format!("{} + {}", a.to_css_prec(1), b.to_css_prec(1)), 1),
            // `a - (b + c)` must keep its parens: without them the sum
            // would re-parse as `(a - b) + c`.
            MathExpr::Sub(a, b) => paren(format!("{} - {}", a.to_css_prec(1), b.to_css_prec(2)), 1),
            MathExpr::Mul(a, b) => paren(format!("{} * {}", a.to_css_prec(2), b.to_css_prec(2)), 2),
            // `a / (b / c)` must keep its parens for the same reason.
            MathExpr::Div(a, b) => paren(format!("{} / {}", a.to_css_prec(2), b.to_css_prec(3)), 2),
            MathExpr::Min(items) => format!(
                "min({})",
                items
                    .iter()
                    .map(|i| i.to_css())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            MathExpr::Max(items) => format!(
                "max({})",
                items
                    .iter()
                    .map(|i| i.to_css())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            MathExpr::Clamp(a, b, c) => {
                format!("clamp({}, {}, {})", a.to_css(), b.to_css(), c.to_css())
            }
            MathExpr::Abs(e) => format!("abs({})", e.to_css()),
            MathExpr::Sign(e) => format!("sign({})", e.to_css()),
            MathExpr::Sqrt(e) => format!("sqrt({})", e.to_css()),
            MathExpr::Exp(e) => format!("exp({})", e.to_css()),
            MathExpr::Sin(e) => format!("sin({})", e.to_css()),
            MathExpr::Cos(e) => format!("cos({})", e.to_css()),
            MathExpr::Tan(e) => format!("tan({})", e.to_css()),
            MathExpr::Asin(e) => format!("asin({})", e.to_css()),
            MathExpr::Acos(e) => format!("acos({})", e.to_css()),
            MathExpr::Atan(e) => format!("atan({})", e.to_css()),
            MathExpr::Mod(a, b) => format!("mod({}, {})", a.to_css(), b.to_css()),
            MathExpr::Rem(a, b) => format!("rem({}, {})", a.to_css(), b.to_css()),
            MathExpr::Pow(a, b) => format!("pow({}, {})", a.to_css(), b.to_css()),
            MathExpr::Atan2(a, b) => format!("atan2({}, {})", a.to_css(), b.to_css()),
            MathExpr::Round { strategy, a, b } => match b {
                Some(b) => format!(
                    "round({}, {}, {})",
                    strategy.as_str(),
                    a.to_css(),
                    b.to_css()
                ),
                None => format!("round({}, {})", strategy.as_str(), a.to_css()),
            },
            MathExpr::Log { a, b } => match b {
                Some(b) => format!("log({}, {})", a.to_css(), b.to_css()),
                None => format!("log({})", a.to_css()),
            },
            MathExpr::Hypot(items) => format!(
                "hypot({})",
                items
                    .iter()
                    .map(|i| i.to_css())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            MathExpr::Toggle(items) => format!(
                "toggle({})",
                items
                    .iter()
                    .map(|i| i.to_css())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            MathExpr::If(branches) => format!(
                "if({})",
                branches
                    .iter()
                    .map(|b| format!("{}: {}", b.condition.to_css(), b.value.to_css()))
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        }
    }
}

impl RoundStrategy {
    /// The CSS `<rounding-strategy>` keyword.
    fn as_str(self) -> &'static str {
        match self {
            RoundStrategy::Nearest => "nearest",
            RoundStrategy::Up => "up",
            RoundStrategy::Down => "down",
            RoundStrategy::ToZero => "to-zero",
            RoundStrategy::LineWidth => "line-width",
        }
    }
}

impl IfCondition {
    /// Serialize the condition back to source text: `else`, or the
    /// verbatim condition tokens for unevaluable conditions.
    fn to_css(&self) -> String {
        match self {
            IfCondition::Else => "else".to_owned(),
            IfCondition::Unevaluable(toks) => crate::token::serialize_tokens(toks),
        }
    }
}

impl MathValue {
    fn neg(self) -> MathValue {
        match self {
            MathValue::Number(n) => MathValue::Number(-n),
            MathValue::Length(l) => MathValue::Length(-l),
            MathValue::Angle(r) => MathValue::Angle(-r),
        }
    }

    /// CSS Values 4 §10.2 typing: like types add (`Angle ± Angle →
    /// Angle`); anything mixed (`Angle + Number`, `Angle + Length`) is
    /// invalid.
    fn add(a: MathValue, b: MathValue) -> Option<MathValue> {
        match (a, b) {
            (MathValue::Number(x), MathValue::Number(y)) => Some(MathValue::Number(x + y)),
            (MathValue::Length(x), MathValue::Length(y)) => Some(MathValue::Length(x + y)),
            (MathValue::Angle(x), MathValue::Angle(y)) => Some(MathValue::Angle(x + y)),
            _ => None,
        }
    }

    /// `Angle ×/÷ Number → Angle`; `Angle × Angle`, `Angle × Length`
    /// (and `Length × Length`) are type errors (CSS Values 4 §10.2).
    fn mul(a: MathValue, b: MathValue) -> Option<MathValue> {
        match (a, b) {
            (MathValue::Number(x), MathValue::Number(y)) => Some(MathValue::Number(x * y)),
            (MathValue::Number(x), MathValue::Length(y))
            | (MathValue::Length(y), MathValue::Number(x)) => Some(MathValue::Length(x * y)),
            (MathValue::Number(x), MathValue::Angle(y))
            | (MathValue::Angle(y), MathValue::Number(x)) => Some(MathValue::Angle(x * y)),
            _ => None,
        }
    }

    fn div(a: MathValue, b: MathValue) -> Option<MathValue> {
        let MathValue::Number(d) = b else {
            return None;
        };
        if d == 0.0 {
            return None;
        }
        match a {
            MathValue::Number(x) => Some(MathValue::Number(x / d)),
            MathValue::Length(x) => Some(MathValue::Length(x / d)),
            MathValue::Angle(x) => Some(MathValue::Angle(x / d)),
        }
    }

    /// Compare on the fly: no `Vec` allocation per evaluation. Mixed
    /// numbers, lengths and angles (or an empty argument list) are
    /// invalid.
    fn min_max(
        items: &[MathExpr],
        ctx: &MathContext,
        is_min: bool,
        depth: u32,
    ) -> Option<MathValue> {
        let pick = |a: f32, b: f32| if is_min { a.min(b) } else { a.max(b) };
        let mut best: Option<MathValue> = None;
        for item in items {
            let v = item.eval(ctx, depth)?;
            best = Some(match (best, v) {
                (None, v) => v,
                (Some(MathValue::Number(a)), MathValue::Number(b)) => MathValue::Number(pick(a, b)),
                (Some(MathValue::Length(a)), MathValue::Length(b)) => MathValue::Length(pick(a, b)),
                (Some(MathValue::Angle(a)), MathValue::Angle(b)) => MathValue::Angle(pick(a, b)),
                // Mixed types are invalid math.
                _ => return None,
            });
        }
        best
    }
}

/// The three math-value kinds the evaluator distinguishes: unitless
/// numbers, lengths (px), and angles (radians). Used by the math-function
/// evaluator arms, which mostly need "same type or invalid" checks rather
/// than the full [`MathValue`] algebra.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MathKind {
    Number,
    Length,
    Angle,
}

/// Split an evaluated [`MathValue`] into its scalar and its kind.
fn math_scalar_and_kind(v: MathValue) -> (f32, MathKind) {
    match v {
        MathValue::Number(n) => (n, MathKind::Number),
        MathValue::Length(l) => (l, MathKind::Length),
        MathValue::Angle(r) => (r, MathKind::Angle),
    }
}

/// Rebuild a [`MathValue`] of `kind` from a scalar: the shared tail of
/// the kind-preserving math functions (`mod()`/`rem()`/`round()`/
/// `hypot()`).
fn math_value_of_kind(v: f32, kind: MathKind) -> MathValue {
    match kind {
        MathKind::Number => MathValue::Number(v),
        MathKind::Length => MathValue::Length(v),
        MathKind::Angle => MathValue::Angle(v),
    }
}

/// `mod(A, B)` (`is_mod`) or `rem(A, B)`: the remainder of `A` divided
/// by `B`, with the sign of `B` (modulus) or of `A` (remainder), per
/// CSS Values 4 §10.3. Both must share a type; a zero `B` is NaN:
/// invalid.
fn math_stepped(a: MathValue, b: MathValue, is_mod: bool) -> Option<MathValue> {
    let (x, x_kind) = math_scalar_and_kind(a);
    let (y, y_kind) = math_scalar_and_kind(b);
    if x_kind != y_kind || y == 0.0 {
        return None;
    }
    // mod() floors the quotient (sign of B); rem() truncates it (sign
    // of A) — spec §10.3.
    let q = x / y;
    let k = if is_mod { q.floor() } else { q.trunc() };
    let v = x - k * y;
    if !v.is_finite() {
        return None;
    }
    Some(math_value_of_kind(v, x_kind))
}

/// `sin()`/`cos()`/`tan()`: the argument must be a `<number>` or an
/// `<angle>` (CSS Values 4 §10.4); an angle argument is already in
/// radians (see [`MathExpr::Angle`]). The result is a `<number>`. An
/// infinite argument is NaN per §10.4.1: invalid.
fn math_trig(v: MathValue, f: fn(f32) -> f32) -> Option<MathValue> {
    let n = match v {
        MathValue::Number(n) | MathValue::Angle(n) => n,
        MathValue::Length(_) => return None,
    };
    if !n.is_finite() {
        return None;
    }
    let r = f(n);
    if r.is_finite() {
        Some(MathValue::Number(r))
    } else {
        None
    }
}

/// `asin()`/`acos()`/`atan()`: the argument must be a `<number>`; the
/// result is radians as a `<number>` (this crate has no `<angle>` type).
/// `|A| > 1` is NaN for `asin()`/`acos()` per §10.4.1: invalid.
fn math_inverse_trig(v: MathValue, f: fn(f32) -> f32) -> Option<MathValue> {
    let MathValue::Number(n) = v else {
        return None;
    };
    if !n.is_finite() {
        return None;
    }
    let r = f(n);
    // `f32::asin`/`acos` already return NaN outside [-1, 1]; the
    // finiteness check folds that to invalid uniformly.
    if r.is_finite() {
        Some(MathValue::Number(r))
    } else {
        None
    }
}

impl CssValue {
    /// Evaluate a [`CssValue::Math`] expression against `ctx`, returning the
    /// concrete value: [`CssValue::Number`] for unitless results,
    /// [`CssValue::Length`] in px for lengths. Any other value is returned
    /// unchanged. `None` means the expression is invalid math.
    pub fn evaluate_math(&self, ctx: &MathContext) -> Option<CssValue> {
        match self {
            CssValue::Math(e) => match e.evaluate(ctx)? {
                MathValue::Number(n) => Some(CssValue::Number(n)),
                MathValue::Length(px) => Some(CssValue::Length(px, LengthUnit::Px)),
                // `CssValue` has no angle form: an angle result flattens
                // to its radian measure. The cascade never routes an
                // angle here (angles are invalid at computed-value time,
                // see `Styler::eval_math_for_prop`); this arm exists so
                // angle-typed math (e.g. `atan2()`) stays observable in
                // unit tests.
                MathValue::Angle(r) => Some(CssValue::Number(r)),
            },
            _ => Some(self.clone()),
        }
    }

    /// Serialize a [`CssValue::Math`] expression tree back to its CSS
    /// `calc()` source form (operators keep CSS spacing, parens only
    /// where precedence demands). Returns `None` for non-`Math` values.
    /// Lets CSSOM serialization round-trip math without leaking the
    /// expression tree's `Debug` form.
    pub fn serialize_math(&self) -> Option<String> {
        match self {
            CssValue::Math(expr) => Some(format!("calc({})", expr.to_css())),
            _ => None,
        }
    }

    /// Whether this value is math that can evaluate to an
    /// `<angle>`-typed value. Angles never belong in
    /// `<length-percentage>` contexts; CSSOM validation uses this to
    /// reject `calc(20deg)` for inset properties at parse time.
    pub fn contains_angle(&self) -> bool {
        matches!(self, CssValue::Math(e) if e.contains_angle())
    }

    /// If this value is `calc-size()`, the parsed basis (see
    /// [`CalcSizeBasis`]); `None` otherwise. Lets layout ask whether a
    /// `calc-size()` is `auto`/`max-content`-based without matching the    /// [`CssValue`] enum.
    pub fn calc_size_basis(&self) -> Option<&CalcSizeBasis> {
        match self {
            CssValue::CalcSize { basis, .. } => Some(basis),
            _ => None,
        }
    }

    /// Evaluate a [`CssValue::CalcSize`] against `ctx`: resolve a definite
    /// basis to a math expression, substitute it for the calc-sum's
    /// `size` leaves, and evaluate. Returns `None` for keyword bases
    /// (only layout can resolve those — the value survives the cascade
    /// untouched; see [`CssValue::calc_size_basis`]) and for invalid
    /// math. Any non-`calc-size()` value is returned unchanged.
    pub fn evaluate_calc_size(&self, ctx: &MathContext) -> Option<CssValue> {
        match self {
            CssValue::CalcSize { basis, calc } => {
                let base = basis.as_math_expr()?;
                let tree = calc.substitute_size(&base)?;
                match tree.evaluate(ctx)? {
                    MathValue::Number(n) => Some(CssValue::Number(n)),
                    MathValue::Length(px) => Some(CssValue::Length(px, LengthUnit::Px)),
                    // `calc-size()` computes to a length: an angle result
                    // is invalid math here.
                    MathValue::Angle(_) => None,
                }
            }
            _ => Some(self.clone()),
        }
    }

    /// Serialize a [`CssValue::CalcSize`] back to valid CSS source text
    /// (`calc-size(<basis>, <calc-sum>)`), so e.g. `el.style.width`
    /// readback re-parses. The basis serializes as its keyword,
    /// length/percentage, `calc(...)`-wrapped math, or verbatim
    /// `var()`; the calc-sum serializes as a calc body with the `size`
    /// leaf intact. Returns `None` for any other value.
    ///
    /// This is the real serializer behind the `calc-size()` arm of CSS
    /// value serialization (compare the `{:?}` Debug fallback the
    /// `Math` arm still uses in `bowser_js`'s `css_value_to_string`):
    /// the output is valid CSS and re-parses to an equal value.
    pub fn serialize_calc_size(&self) -> Option<String> {
        match self {
            CssValue::CalcSize { basis, calc } => {
                Some(format!("calc-size({}, {})", basis.to_css(), calc.to_css()))
            }
            _ => None,
        }
    }

    /// Whether the value syntactically contains a percentage leaf anywhere:
    /// a bare [`CssValue::Percentage`], or one nested inside a
    /// `calc()`/`min()`/`max()`/`clamp()` tree (via
    /// [`MathExpr::contains_percentage`]), a `minmax()` pair, an
    /// unexpanded `repeat()` track list, a shadow layer's length fields,
    /// or a list.
    ///
    /// Value-level wrapper around the tree walk, mirroring the
    /// `var()`-shaped `value_contains_var` gate in `parse.rs`. Detection
    /// is syntactic: a `var()` reference is opaque (it may substitute to a
    /// percentage), so [`CssValue::Var`] reports `false` and the cascade
    /// re-checks after substitution; [`CssValue::Tokens`] (unparsed
    /// custom-property streams) are opaque for the same reason. No
    /// production callers yet — where the value shape is already known,
    /// parse-time validators reach into [`MathExpr`] directly (e.g.
    /// [`Shadow::is_shadow_length`] matches `CssValue::Math(e)` and calls
    /// `e.contains_percentage()`).
    // No production callers yet; kept for future parse-time consumers.
    #[allow(dead_code)]
    pub(crate) fn contains_percentage(&self) -> bool {
        match self {
            CssValue::Percentage(_) => true,
            CssValue::Math(e) => e.contains_percentage(),
            CssValue::CalcSize { basis, calc } => {
                let basis_has_pct = match basis.as_ref() {
                    CalcSizeBasis::Percentage(_) => true,
                    CalcSizeBasis::Math(e) => e.contains_percentage(),
                    _ => false,
                };
                basis_has_pct || calc.contains_percentage()
            }
            CssValue::List(items) => items.iter().any(|i| i.contains_percentage()),
            CssValue::MinMax(min, max) => min.contains_percentage() || max.contains_percentage(),
            CssValue::Repeat { tracks, .. } => tracks.iter().any(|t| t.contains_percentage()),
            CssValue::Shadows(shadows) => shadows.iter().any(|s| {
                [&s.offset_x, &s.offset_y, &s.blur, &s.spread]
                    .into_iter()
                    .any(|v| v.contains_percentage())
            }),
            _ => false,
        }
    }
}

/// Value parse failure. Declarations whose value fails to parse are dropped.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ValueError;

/// Parse a declaration value from its token slice.
pub(crate) fn parse_value(tokens: &[Token]) -> Result<CssValue, ValueError> {
    parse_value_depth(tokens, 0, false)
}

/// Parse a declaration value from its token slice, preserving the case
/// of identifier keywords (`Arial` stays `Arial` instead of becoming
/// `arial`) and skipping color canonicalization (a color name like `Red`
/// stays an identifier instead of becoming a `Color`). Used only for
/// custom-property substitution feeding the deferred `font` shorthand
/// expansion, so the deferred path agrees with the eager token parser
/// (`parse_font_shorthand`), which reads family names straight from the
/// tokens. Everywhere else CSS keywords are matched case-insensitively
/// against the lowercased form, so the canonical parse stays lowercasing.
pub(crate) fn parse_value_preserving_case(tokens: &[Token]) -> Result<CssValue, ValueError> {
    parse_value_depth(tokens, 0, true)
}

/// Depth-capped value parsing: `var()` fallbacks recurse into
/// [`parse_value`], so the depth budget bounds adversarial
/// `var(--a, var(--b, ...))` chains (see [`MAX_MATH_DEPTH`]).
fn parse_value_depth(
    tokens: &[Token],
    depth: u32,
    preserve_case: bool,
) -> Result<CssValue, ValueError> {
    if depth > MAX_MATH_DEPTH {
        return Err(ValueError);
    }
    // Split top-level commas into groups.
    let mut groups: Vec<&[Token]> = Vec::new();
    let mut start = 0;
    let mut comma_depth: u32 = 0;
    // Track depth-relevant function nesting (`var`/`calc-size` and every
    // math function recurse with `depth + 1`). If the nesting in the tokens
    // already exceeds the remaining budget, fail fast: without this,
    // 100k nested `var()` fallbacks do 64 O(n) scans (one per depth
    // level) before hitting the limit (B9). A stack tracks which opens
    // were counted, so unrelated functions (e.g. `rgb()`) don't skew it.
    let mut counted_stack: Vec<bool> = Vec::new();
    let mut fn_depth: u32 = 0;
    for (i, t) in tokens.iter().enumerate() {
        match t {
            Token::LParen | Token::LBracket => {
                comma_depth += 1;
                counted_stack.push(false);
            }
            Token::Function(name)
                if {
                    let lower = name.to_ascii_lowercase();
                    // `var()`/`calc-size()` recurse with `depth + 1`, as do
                    // all math functions (see `parse_math_fn_body`).
                    lower == "var" || lower == "calc-size" || is_math_fn_name(&lower)
                } =>
            {
                comma_depth += 1;
                counted_stack.push(true);
                fn_depth += 1;
                if depth + fn_depth > MAX_MATH_DEPTH {
                    return Err(ValueError);
                }
            }
            Token::Function(_) => {
                comma_depth += 1;
                counted_stack.push(false);
            }
            Token::RParen | Token::RBracket => {
                comma_depth = comma_depth.saturating_sub(1);
                if counted_stack.pop().unwrap_or(false) {
                    fn_depth = fn_depth.saturating_sub(1);
                }
            }
            Token::Comma if comma_depth == 0 => {
                groups.push(&tokens[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    groups.push(&tokens[start..]);

    let mut parsed = Vec::with_capacity(groups.len());
    for group in groups {
        parsed.push(parse_value_group(group, depth, preserve_case)?);
    }
    if parsed.len() == 1 {
        Ok(parsed.pop().expect("one group"))
    } else {
        Ok(CssValue::List(parsed))
    }
}

/// Parse one comma group: whitespace-separated items, or a single item.
fn parse_value_group(
    tokens: &[Token],
    depth: u32,
    preserve_case: bool,
) -> Result<CssValue, ValueError> {
    let mut items = Vec::new();
    let mut start = 0;
    let mut i = 0;
    let mut paren_depth: u32 = 0;
    while i <= tokens.len() {
        // Whitespace inside functions/parens never separates items.
        if i < tokens.len() {
            match tokens[i] {
                Token::LParen | Token::Function(_) | Token::LBracket => paren_depth += 1,
                Token::RParen | Token::RBracket => paren_depth = paren_depth.saturating_sub(1),
                _ => {}
            }
        }
        let is_sep = i == tokens.len()
            || (paren_depth == 0
                && matches!(tokens[i], Token::Whitespace | Token::Eof | Token::Semicolon));
        if is_sep {
            if start < i {
                items.push(parse_item(&tokens[start..i], depth, preserve_case)?);
            }
            start = i + 1;
        }
        i += 1;
    }
    if items.is_empty() {
        return Err(ValueError);
    }
    if items.len() == 1 {
        Ok(items.pop().expect("one item"))
    } else if items.iter().all(|i| matches!(i, CssValue::Transform(_))) {
        // A transform function list is one `Transform` with one entry
        // per function (see `CssValue::Transform`): accumulate the whole
        // whitespace-separated list here so consumers never see the
        // dual `List`-of-single-`Transform` shape.
        let mut fns = Vec::new();
        for item in items {
            if let CssValue::Transform(inner) = item {
                fns.extend(inner);
            }
        }
        Ok(CssValue::Transform(fns))
    } else {
        Ok(CssValue::List(items))
    }
}

fn parse_item(tokens: &[Token], depth: u32, preserve_case: bool) -> Result<CssValue, ValueError> {
    match tokens {
        [Token::Ident(name)] => {
            // Case-preserving mode (deferred `font` substitution only):
            // the ident keeps its authored case and is never
            // color-canonicalized, so the font expansion sees the same
            // token stream the eager parser sees (`font: 12px Red` reads
            // `Red` straight from the tokens as a family name; the
            // deferred path must hand it over as an identifier too).
            // Keyword *matching* on this path is the font expansion's,
            // which lowercases before comparing (see `parse_font_prefix`
            // in cascade.rs); `expand_font_value` and `expand_font_slash`
            // lowercase where the eager parser does.
            if preserve_case {
                return Ok(CssValue::Keyword(name.clone().into()));
            }
            let lower = name.to_ascii_lowercase();
            // NOTE: the legacy `-webkit-fill-available`/`-moz-available`
            // aliases are *sizing-grammar* aliases only (css-sizing-4;
            // Chrome rejects them for alignment properties), so they
            // stay plain keywords here. The cascade's sizing validation
            // normalizes them to `stretch` (see `is_valid_for_property`
            // and the fill-available fold in `compute_prop`).
            if lower == "currentcolor" {
                return Ok(CssValue::Keyword(lower.into()));
            }
            // Named colors and `transparent` fold to RGBA here on
            // purpose: the cascade and computed style consume plain
            // `Color` (see `is_valid_for_property`), so the keyword
            // is not carried in the value. Specified-style
            // serialization recovers it from the raw declaration
            // text instead (see `specified_color_keyword` in
            // `bowser_js`, used by `decl_value_preserving_math`):
            // that keeps `el.style` keyword-faithful while computed
            // style keeps the rgb()/rgba() form.
            if lower == "transparent" {
                return Ok(CssValue::Color(Color {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 0,
                }));
            }
            if let Some((r, g, b)) = named_color(&lower) {
                return Ok(CssValue::Color(Color { r, g, b, a: 255 }));
            }
            Ok(CssValue::Keyword(lower.into()))
        }
        [Token::Number { value: n, .. }] => Ok(CssValue::Number(*n)),
        [Token::Dimension { value: n, unit, .. }] => match unit.to_ascii_lowercase().as_str() {
            "px" => Ok(CssValue::Length(*n, LengthUnit::Px)),
            "cm" => Ok(CssValue::Length(*n, LengthUnit::Cm)),
            "mm" => Ok(CssValue::Length(*n, LengthUnit::Mm)),
            "in" => Ok(CssValue::Length(*n, LengthUnit::In)),
            "pt" => Ok(CssValue::Length(*n, LengthUnit::Pt)),
            "pc" => Ok(CssValue::Length(*n, LengthUnit::Pc)),
            "q" => Ok(CssValue::Length(*n, LengthUnit::Q)),
            "em" => Ok(CssValue::Length(*n, LengthUnit::Em)),
            "rem" => Ok(CssValue::Length(*n, LengthUnit::Rem)),
            "ex" => Ok(CssValue::Length(*n, LengthUnit::Ex)),
            "vw" => Ok(CssValue::Length(*n, LengthUnit::Vw)),
            "vh" => Ok(CssValue::Length(*n, LengthUnit::Vh)),
            "vmin" => Ok(CssValue::Length(*n, LengthUnit::Vmin)),
            "vmax" => Ok(CssValue::Length(*n, LengthUnit::Vmax)),
            // The grid `fr` unit: a negative or non-finite fraction is
            // invalid and drops the whole declaration (CSS grid §7.2.3).
            // A non-finite `fr` (e.g. `1e999fr`) would poison track sizing
            // with NaN (`free * inf / inf`).
            "fr" if *n >= 0.0 && n.is_finite() => Ok(CssValue::Fr(*n)),
            _ => Err(ValueError),
        },
        [Token::Percentage(p)] => Ok(CssValue::Percentage(*p)),
        [Token::Hash(hex, _)] => parse_hex(hex).map(CssValue::Color).ok_or(ValueError),
        [Token::String(s)] => Ok(CssValue::String(s.clone())),
        [Token::Url(u)] => Ok(CssValue::Url(u.clone())),
        [Token::Function(name), rest @ ..] => {
            // `rest` is the function body plus its closing RParen.
            let body = rest.strip_suffix(&[Token::RParen]).ok_or(ValueError)?;
            parse_function(&name.to_ascii_lowercase(), body, depth, preserve_case)
        }
        _ => Err(ValueError),
    }
}

/// Parse `attr(<name> [string] [, <fallback>]?)` (see
/// [`parse_function`]). Only the string form is supported — any other
/// `<type-or-unit>` drops the declaration — and the name is an
/// `<attr-name>` (ident or string). The fallback, when present, parses
/// as a generic value (depth-budgeted like `var()` fallbacks).
fn parse_attr(body: &[Token], depth: u32, preserve_case: bool) -> Result<CssValue, ValueError> {
    if depth > MAX_MATH_DEPTH {
        return Err(ValueError);
    }
    // Split the name from the fallback at the first top-level comma
    // (mirroring `parse_var`): bracket nesting joins function nesting
    // in the depth count, so a comma inside `[]` or a nested function
    // does not start the fallback.
    let mut comma = None;
    let mut paren_depth: u32 = 0;
    for (i, t) in body.iter().enumerate() {
        match t {
            Token::Function(_) | Token::LParen | Token::LBracket => paren_depth += 1,
            Token::RParen | Token::RBracket => paren_depth = paren_depth.saturating_sub(1),
            Token::Comma if paren_depth == 0 => {
                comma = Some(i);
                break;
            }
            _ => {}
        }
    }
    let (name_toks, fallback_toks) = match comma {
        Some(i) => (&body[..i], Some(&body[i + 1..])),
        None => (body, None),
    };
    let name_toks: Vec<&Token> = name_toks
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .collect();
    let name = match name_toks.as_slice() {
        // `<attr-name>`: an ident or string, lowercased (attribute
        // lookup is ASCII case-insensitive for HTML elements).
        [Token::Ident(n)] => n.to_ascii_lowercase(),
        [Token::String(n)] => n.to_ascii_lowercase(),
        // The optional `<type-or-unit>`: only `string` is supported.
        [Token::Ident(n), Token::Ident(ty)] if ty.eq_ignore_ascii_case("string") => {
            n.to_ascii_lowercase()
        }
        [Token::String(n), Token::Ident(ty)] if ty.eq_ignore_ascii_case("string") => {
            n.to_ascii_lowercase()
        }
        _ => return Err(ValueError),
    };
    if name.is_empty() {
        return Err(ValueError);
    }
    let fallback = match fallback_toks {
        Some(ft) if ft.iter().all(|t| matches!(t, Token::Whitespace)) => {
            return Err(ValueError); // empty fallback
        }
        Some(ft) => Some(Box::new(parse_value_depth(ft, depth + 1, preserve_case)?)),
        None => None,
    };
    Ok(CssValue::Attr(Box::new(AttrData { name, fallback })))
}

/// Parse `counter(<name> [, <style>])` / `counters(<name>, <separator>
/// [, <style>])` (see [`parse_function`]). Arguments split on
/// top-level commas; the name and style are idents (lowercased), the
/// `counters()` separator a string. Unknown counter styles are kept
/// verbatim and degrade to `decimal` at render.
fn parse_counter_fn(plural: bool, body: &[Token], depth: u32) -> Result<CssValue, ValueError> {
    if depth > MAX_MATH_DEPTH {
        return Err(ValueError);
    }
    // Split the body on top-level commas into one token slice per
    // argument (mirroring `parse_transform_function`).
    let mut args: Vec<&[Token]> = Vec::new();
    let mut start = 0;
    let mut paren_depth: u32 = 0;
    for (i, t) in body.iter().enumerate() {
        match t {
            Token::Function(_) | Token::LParen | Token::LBracket => paren_depth += 1,
            Token::RParen | Token::RBracket => paren_depth = paren_depth.saturating_sub(1),
            Token::Comma if paren_depth == 0 => {
                args.push(&body[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    args.push(&body[start..]);
    let arg = |i: usize| -> Result<Vec<&Token>, ValueError> {
        let toks: Vec<&Token> = args
            .get(i)
            .ok_or(ValueError)?
            .iter()
            .filter(|t| !matches!(t, Token::Whitespace))
            .collect();
        if toks.is_empty() {
            return Err(ValueError);
        }
        Ok(toks)
    };
    // Arity: `counter()` takes 1-2 args, `counters()` 2-3.
    if plural && args.len() != 2 && args.len() != 3 {
        return Err(ValueError);
    }
    if !plural && args.len() != 1 && args.len() != 2 {
        return Err(ValueError);
    }
    let name = match arg(0)?.as_slice() {
        [Token::Ident(n)] => n.to_ascii_lowercase(),
        _ => return Err(ValueError),
    };
    if name.is_empty() {
        return Err(ValueError);
    }
    let (separator, style_arg) = if plural {
        let sep = match arg(1)?.as_slice() {
            [Token::String(s)] => s.clone(),
            _ => return Err(ValueError),
        };
        let style = if args.len() == 3 {
            Some(ident_arg(&arg(2)?)?)
        } else {
            None
        };
        (Some(sep), style)
    } else {
        let style = if args.len() == 2 {
            Some(ident_arg(&arg(1)?)?)
        } else {
            None
        };
        (None, style)
    };
    Ok(CssValue::Counter(Box::new(CounterData {
        name,
        style: style_arg,
        separator,
    })))
}

/// One lowercased ident argument (the `counter()` / `counters()`
/// `<counter-style>`).
fn ident_arg(toks: &[&Token]) -> Result<String, ValueError> {
    match toks {
        [Token::Ident(n)] => Ok(n.to_ascii_lowercase()),
        _ => Err(ValueError),
    }
}

fn parse_function(
    name: &str,
    body: &[Token],
    depth: u32,
    preserve_case: bool,
) -> Result<CssValue, ValueError> {
    match name {
        "rgb" | "rgba" => parse_rgb(body),
        "hsl" | "hsla" => parse_hsl(body),
        "hwb" => parse_hwb(body),
        "lab" => parse_lab(body),
        "lch" => parse_lch(body),
        name if is_math_fn_name(name) => parse_math_function(name, body, depth),
        "calc-size" => parse_calc_size(body, depth),
        "repeat" => parse_repeat(body, depth, preserve_case),
        "minmax" => parse_minmax(body, depth, preserve_case),
        "url" => {
            let b: Vec<&Token> = body
                .iter()
                .filter(|t| !matches!(t, Token::Whitespace))
                .collect();
            match b.as_slice() {
                [Token::String(s)] => Ok(CssValue::Url(s.clone())),
                // Lenient: unquoted url() that tokenized as a function.
                [Token::Ident(s)] => Ok(CssValue::Url(s.clone())),
                _ => Err(ValueError),
            }
        }
        "var" => parse_var(body, depth, preserve_case),
        // CSS Generated Content 3 §2.2 `content` functions: `attr()`
        // (string form + fallback) and `counter()` / `counters()`
        // resolve element-scoped at box-synthesis time (see
        // `pseudo::resolve_content_text`), so they parse here into
        // dedicated `CssValue` shapes instead of hitting the catch-all
        // `Err` below (which drops the whole declaration).
        "attr" => parse_attr(body, depth, preserve_case),
        "counter" => parse_counter_fn(false, body, depth),
        "counters" => parse_counter_fn(true, body, depth),
        // CSS Images 3/4 gradient functions: retained verbatim as
        // `CssValue::Gradient` (see the variant docs). Only an empty
        // body drops the declaration; structural validation is
        // `bowser_paint::gradient::parse_gradient`'s job.
        //
        // A `var()` anywhere in the body cannot resolve until
        // computed-value time (custom properties are element-scoped),
        // so the whole function is kept as a verbatim token stream
        // instead: the cascade substitutes the references and re-parses
        // into a real `Gradient` (see `Styler::resolve`'s `Tokens` arm).
        // Freezing the `var()` into `source` instead would hand paint's
        // gradient parser a stream it chokes on, painting nothing.
        "linear-gradient"
        | "repeating-linear-gradient"
        | "radial-gradient"
        | "repeating-radial-gradient"
        | "conic-gradient"
        | "repeating-conic-gradient" => {
            let body = trim_tokens(body);
            if body.is_empty() {
                return Err(ValueError);
            }
            if body
                .iter()
                .any(|t| matches!(t, Token::Function(n) if n.eq_ignore_ascii_case("var")))
            {
                let mut toks = Vec::with_capacity(body.len() + 2);
                toks.push(Token::Function(name.to_string()));
                toks.extend(body.iter().cloned());
                toks.push(Token::RParen);
                return Ok(CssValue::Tokens(toks));
            }
            Ok(CssValue::Gradient {
                name: name.to_string(),
                source: format!("{name}({})", tokens_to_css(body)),
            })
        }
        // CSS Transforms 1: `translate()` / `scale()` / `rotate()` and
        // the single-axis forms parse structurally; every other
        // transform function (`matrix()` / `matrix3d()`, `skew()` /
        // `skewx()` / `skewy()`, `perspective()`, the `*3d()` and `*z()`
        // forms) is retained opaque (see `parse_transform_function`) so
        // a valid-but-unrenderable function still counts as non-`none`
        // for containing-block purposes. Unknown functions stay invalid.
        "translate" | "translatex" | "translatey" | "scale" | "scalex" | "scaley" | "rotate"
        | "matrix" | "matrix3d" | "skew" | "skewx" | "skewy" | "perspective" | "translate3d"
        | "translatez" | "scale3d" | "scalez" | "rotate3d" | "rotatex" | "rotatey" | "rotatez" => {
            parse_transform_function(name, body)
        }
        // CSS Shapes 1 `<basic-shape>` functions (used by `clip-path`;
        // see `parse_basic_shape`). Parsed structurally here so the
        // cascade's `var()`-substitution re-parse lands on the same
        // `CssValue::ClipPath` shape the eager parser produces.
        "inset" | "circle" | "ellipse" | "polygon" => {
            parse_basic_shape(name, body).map(CssValue::ClipPath)
        }
        _ => Err(ValueError),
    }
}

/// Parse a `transform` function from its raw body tokens (see
/// [`parse_function`]). Arguments split on top-level commas.
/// `translate()` / `scale()` / `rotate()` (and the single-axis forms)
/// with plain arguments parse structurally, and every number must be
/// finite. Every other transform function — and strict-subset functions
/// whose arguments contain nested functions such as `calc()`, which
/// cannot be evaluated until math evaluation lands — is retained opaque
/// as [`TransformFunction::Other`] so the declaration still counts as
/// non-`none` for containing-block purposes. Only truly invalid syntax
/// is rejected: unknown functions never reach here, empty bodies like
/// `matrix()` fail, and the structural subset keeps its arity and
/// argument-type checks.
/// One transform-function argument: exactly one non-whitespace token.
fn one_token(arg: &[Token]) -> Result<&Token, ValueError> {
    let mut toks = arg.iter().filter(|t| !matches!(t, Token::Whitespace));
    match (toks.next(), toks.next()) {
        (Some(t), None) => Ok(t),
        _ => Err(ValueError),
    }
}

/// Trim leading/trailing whitespace tokens from a slice.
fn trim_tokens(tokens: &[Token]) -> &[Token] {
    let mut t = tokens;
    while matches!(t.first(), Some(Token::Whitespace)) {
        t = &t[1..];
    }
    while matches!(t.last(), Some(Token::Whitespace)) {
        t = &t[..t.len() - 1];
    }
    t
}

/// Re-serialize tokens to CSS source text, collapsing whitespace runs
/// to a single space. Used to retain opaque transform functions
/// verbatim enough for `getComputedStyle` round-tripping.
pub(crate) fn tokens_to_css(tokens: &[Token]) -> String {
    fn num(n: f32, out: &mut String) {
        // `{}` on `f32` prints the shortest round-trip form (`2`, not
        // `2.0`).
        out.push_str(&format!("{n}"));
    }
    let mut out = String::new();
    // Start "already spaced" so leading whitespace emits nothing.
    let mut spaced = true;
    for t in tokens {
        match t {
            Token::Whitespace => {
                if !spaced {
                    out.push(' ');
                    spaced = true;
                }
            }
            Token::Ident(s) => {
                out.push_str(s);
                spaced = false;
            }
            Token::Function(s) => {
                out.push_str(s);
                out.push('(');
                spaced = false;
            }
            Token::AtKeyword(s) => {
                out.push('@');
                out.push_str(s);
                spaced = false;
            }
            Token::Hash(s, _) => {
                out.push('#');
                out.push_str(s);
                spaced = false;
            }
            Token::String(s) => {
                out.push('"');
                for c in s.chars() {
                    if c == '"' || c == '\\' {
                        out.push('\\');
                    }
                    out.push(c);
                }
                out.push('"');
                spaced = false;
            }
            Token::Url(s) => {
                out.push_str("url(\"");
                for c in s.chars() {
                    if c == '"' || c == '\\' {
                        out.push('\\');
                    }
                    out.push(c);
                }
                out.push_str("\")");
                spaced = false;
            }
            Token::BadString | Token::BadUrl | Token::Eof => {}
            Token::Number { value, .. } => {
                num(*value, &mut out);
                spaced = false;
            }
            Token::Dimension { value, unit, .. } => {
                num(*value, &mut out);
                out.push_str(unit);
                spaced = false;
            }
            Token::Percentage(p) => {
                num(*p, &mut out);
                out.push('%');
                spaced = false;
            }
            Token::Delim(c) => {
                out.push(*c);
                spaced = false;
            }
            Token::Colon => {
                out.push(':');
                spaced = false;
            }
            Token::Semicolon => {
                out.push(';');
                spaced = false;
            }
            Token::Comma => {
                out.push(',');
                spaced = false;
            }
            Token::LBracket => {
                out.push('[');
                spaced = false;
            }
            Token::RBracket => {
                out.push(']');
                spaced = false;
            }
            Token::LParen => {
                out.push('(');
                spaced = false;
            }
            Token::RParen => {
                out.push(')');
                spaced = false;
            }
            Token::LBrace => {
                out.push('{');
                spaced = false;
            }
            Token::RBrace => {
                out.push('}');
                spaced = false;
            }
            Token::Cdo => {
                out.push_str("<!--");
                spaced = false;
            }
            Token::Cdc => {
                out.push_str("-->");
                spaced = false;
            }
        }
    }
    out
}

/// Whether serialized CSS `source` (e.g. an opaque [`CssValue::Gradient`]
/// source retained verbatim) names a `var()` function reference. ASCII
/// case-insensitive, allocation-free. Used by the `value_contains_var`
/// gates so a verbatim source that still mentions `var(` is deferred for
/// substitution instead of freezing the reference.
pub(crate) fn css_source_contains_var(source: &str) -> bool {
    source.as_bytes().windows(4).any(|w| {
        w[3] == b'('
            && (w[0] == b'v' || w[0] == b'V')
            && (w[1] == b'a' || w[1] == b'A')
            && (w[2] == b'r' || w[2] == b'R')
    })
}

fn parse_transform_function(name: &str, body: &[Token]) -> Result<CssValue, ValueError> {
    // Split the body on top-level commas into one token slice per
    // argument.
    let mut args: Vec<&[Token]> = Vec::new();
    let mut start = 0;
    let mut depth: u32 = 0;
    for (i, t) in body.iter().enumerate() {
        match t {
            Token::Function(_) | Token::LParen | Token::LBracket => depth += 1,
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            Token::Comma if depth == 0 => {
                args.push(&body[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    args.push(&body[start..]);
    // A nested function or parenthesized group in the body (e.g.
    // `calc()`) cannot be evaluated until math evaluation lands: the
    // whole function is retained opaque below instead of dropping the
    // declaration to `none`.
    let nested = body
        .iter()
        .any(|t| matches!(t, Token::Function(_) | Token::LParen));
    let finite = |n: f32| n.is_finite().then_some(n).ok_or(ValueError);
    // `<length-percentage>` for `translate()`: a length, a percentage,
    // or a unitless zero (the CSS-wide rule). Anything else (numbers,
    // `fr`, angles, keywords) is invalid here.
    let length_or_pct = |arg: &[Token]| -> Result<CssValue, ValueError> {
        let t = one_token(arg)?;
        match t {
            Token::Percentage(p) => Ok(CssValue::Percentage(finite(*p)?)),
            Token::Number { value: n, .. } if *n == 0.0 => Ok(CssValue::Number(0.0)),
            Token::Dimension { value, unit, .. } => {
                let n = finite(*value)?;
                match unit.to_ascii_lowercase().as_str() {
                    "px" => Ok(CssValue::Length(n, LengthUnit::Px)),
                    "cm" => Ok(CssValue::Length(n, LengthUnit::Cm)),
                    "mm" => Ok(CssValue::Length(n, LengthUnit::Mm)),
                    "in" => Ok(CssValue::Length(n, LengthUnit::In)),
                    "pt" => Ok(CssValue::Length(n, LengthUnit::Pt)),
                    "pc" => Ok(CssValue::Length(n, LengthUnit::Pc)),
                    "q" => Ok(CssValue::Length(n, LengthUnit::Q)),
                    "em" => Ok(CssValue::Length(n, LengthUnit::Em)),
                    "rem" => Ok(CssValue::Length(n, LengthUnit::Rem)),
                    "ex" => Ok(CssValue::Length(n, LengthUnit::Ex)),
                    "vw" => Ok(CssValue::Length(n, LengthUnit::Vw)),
                    "vh" => Ok(CssValue::Length(n, LengthUnit::Vh)),
                    "vmin" => Ok(CssValue::Length(n, LengthUnit::Vmin)),
                    "vmax" => Ok(CssValue::Length(n, LengthUnit::Vmax)),
                    _ => Err(ValueError),
                }
            }
            _ => Err(ValueError),
        }
    };
    // `<number>` for `scale()`: a bare number only.
    let number = |arg: &[Token]| -> Result<CssValue, ValueError> {
        match one_token(arg)? {
            Token::Number { value: n, .. } => Ok(CssValue::Number(finite(*n)?)),
            _ => Err(ValueError),
        }
    };
    // `<angle>` for `rotate()`, normalized to degrees (CSS Values 4
    // §6.2: 360deg = 400grad = 2πrad = 1turn). A unitless zero is
    // accepted, matching the unitless-zero rule for lengths.
    let angle_deg = |arg: &[Token]| -> Result<f32, ValueError> {
        match one_token(arg)? {
            Token::Number { value: n, .. } if *n == 0.0 => Ok(0.0),
            Token::Dimension { value, unit, .. } => {
                let n = finite(*value)?;
                match unit.to_ascii_lowercase().as_str() {
                    "deg" => Ok(n),
                    "grad" => Ok(n * 360.0 / 400.0),
                    "rad" => Ok(n * 180.0 / std::f32::consts::PI),
                    "turn" => Ok(n * 360.0),
                    _ => Err(ValueError),
                }
            }
            _ => Err(ValueError),
        }
    };
    let zero = || CssValue::Number(0.0);
    // The structural subset only applies to plain arguments; any other
    // transform function name — or a strict-subset function with nested
    // arguments — is retained opaque so it still counts as non-`none`.
    let structural = !nested
        && matches!(
            name,
            "translate" | "translatex" | "translatey" | "scale" | "scalex" | "scaley" | "rotate"
        );
    if !structural {
        let body = trim_tokens(body);
        if body.is_empty() {
            // `matrix()` and friends with an empty body: truly invalid
            // syntax, not an opaque function.
            return Err(ValueError);
        }
        return Ok(CssValue::Transform(vec![TransformFunction::Other {
            name: name.to_string(),
            raw_args: tokens_to_css(body),
        }]));
    }
    let f = match name {
        "translate" => match args.as_slice() {
            [x] => TransformFunction::Translate {
                x: length_or_pct(x)?,
                y: zero(),
            },
            [x, y] => TransformFunction::Translate {
                x: length_or_pct(x)?,
                y: length_or_pct(y)?,
            },
            _ => return Err(ValueError),
        },
        "translatex" => match args.as_slice() {
            [x] => TransformFunction::Translate {
                x: length_or_pct(x)?,
                y: zero(),
            },
            _ => return Err(ValueError),
        },
        "translatey" => match args.as_slice() {
            [y] => TransformFunction::Translate {
                x: zero(),
                y: length_or_pct(y)?,
            },
            _ => return Err(ValueError),
        },
        "scale" => match args.as_slice() {
            [x] => {
                let x = number(x)?;
                TransformFunction::Scale { x: x.clone(), y: x }
            }
            [x, y] => TransformFunction::Scale {
                x: number(x)?,
                y: number(y)?,
            },
            _ => return Err(ValueError),
        },
        "scalex" => match args.as_slice() {
            [x] => {
                let x = number(x)?;
                TransformFunction::Scale {
                    x,
                    y: CssValue::Number(1.0),
                }
            }
            _ => return Err(ValueError),
        },
        "scaley" => match args.as_slice() {
            [y] => {
                let y = number(y)?;
                TransformFunction::Scale {
                    x: CssValue::Number(1.0),
                    y,
                }
            }
            _ => return Err(ValueError),
        },
        "rotate" => match args.as_slice() {
            [a] => TransformFunction::Rotate {
                angle_deg: angle_deg(a)?,
            },
            _ => return Err(ValueError),
        },
        _ => return Err(ValueError),
    };
    Ok(CssValue::Transform(vec![f]))
}

/// Non-whitespace tokens of a function body.
fn fn_args(body: &[Token]) -> Vec<&Token> {
    body.iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .collect()
}

/// `repeat(<count>, <track-list>)` from grid track listings. An integer
/// count expands eagerly into a [`CssValue::List`] with the track list
/// repeated `count` times — layout then sizes each track independently,
/// which is equivalent for the supported track types (fixed lengths,
/// `fr`, `auto`). `repeat(auto-fill|auto-fit, …)` passes through
/// *unexpanded* as [`CssValue::Repeat`] (the parser contract layout's
/// auto-fill/fit expansion relies on): the repetition count depends on
/// the axis size, which is only known at layout time.
/// Auto-repetition may not nest inside another `repeat()` (grid §7.2.3.2);
/// a nested descriptor fails the value.
/// The *expanded* track count is capped at [`MAX_REPEAT_TRACKS`]: the cap
/// applies to the product, not per nesting level, so a hostile nested
/// `repeat(1000, repeat(1000, repeat(1000, 1px)))` fails the value instead
/// of allocating ~1e9 tracks.
const MAX_REPEAT_TRACKS: usize = 1000;

/// The `repeat()` auto-repetition mode, if `name` is `auto-fill` or
/// `auto-fit` (case-insensitive, per CSS).
fn auto_repeat_mode(name: &str) -> Option<RepeatMode> {
    if name.eq_ignore_ascii_case("auto-fill") {
        Some(RepeatMode::Fill)
    } else if name.eq_ignore_ascii_case("auto-fit") {
        Some(RepeatMode::Fit)
    } else {
        None
    }
}

/// Whether `v` is a `repeat(auto-fill|auto-fit, …)` pass-through
/// descriptor: the [`CssValue::Repeat`] variant layout's auto-fill/fit
/// expansion matches on.
fn is_auto_repeat_descriptor(v: &CssValue) -> bool {
    matches!(v, CssValue::Repeat { .. })
}

/// Whether `v` is, or directly contains, a `repeat(auto-fill|auto-fit, …)`
/// pass-through descriptor. Auto-repetition may not nest inside another
/// `repeat()` (grid §7.2.3.2), so either shape fails the value.
fn contains_auto_repeat(v: &CssValue) -> bool {
    is_auto_repeat_descriptor(v)
        || matches!(v, CssValue::List(items) if items.iter().any(is_auto_repeat_descriptor))
}

fn parse_repeat(body: &[Token], depth: u32, preserve_case: bool) -> Result<CssValue, ValueError> {
    // Split at the first top-level comma: the repetition count, then the
    // track list.
    let mut paren_depth: u32 = 0;
    let mut split = None;
    for (i, t) in body.iter().enumerate() {
        match t {
            Token::LParen | Token::Function(_) | Token::LBracket => paren_depth += 1,
            Token::RParen | Token::RBracket => paren_depth = paren_depth.saturating_sub(1),
            Token::Comma if paren_depth == 0 => {
                split = Some(i);
                break;
            }
            _ => {}
        }
    }
    let split = split.ok_or(ValueError)?;
    let count_toks: Vec<&Token> = body[..split]
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .collect();
    // `auto-fill` / `auto-fit` pass through unexpanded as a dedicated
    // `Repeat` descriptor: layout expands them where the axis size is
    // known (see `CssValue::Repeat`).
    if let [Token::Ident(mode)] = count_toks.as_slice()
        && let Some(auto) = auto_repeat_mode(mode)
    {
        let parsed = parse_value_depth(&body[split + 1..], depth + 1, preserve_case)?;
        if contains_auto_repeat(&parsed) {
            return Err(ValueError);
        }
        let tracks = match parsed {
            CssValue::List(items) => items,
            single => vec![single],
        };
        return Ok(CssValue::Repeat { mode: auto, tracks });
    }
    let count = match count_toks.as_slice() {
        [Token::Number { value: n, .. }] if n.fract() == 0.0 && *n >= 1.0 && *n <= 1000.0 => {
            *n as usize
        }
        _ => return Err(ValueError),
    };
    let parsed = parse_value_depth(&body[split + 1..], depth + 1, preserve_case)?;
    if contains_auto_repeat(&parsed) {
        return Err(ValueError);
    }
    let tracks = match parsed {
        CssValue::List(items) => items,
        single => vec![single],
    };
    // Cap the expanded product, not just the per-level count: nested
    // repeats multiply, so check with overflow-safe arithmetic.
    let total = tracks.len().checked_mul(count).ok_or(ValueError)?;
    if total > MAX_REPEAT_TRACKS {
        return Err(ValueError);
    }
    let mut out = Vec::with_capacity(total);
    for _ in 0..count {
        out.extend(tracks.iter().cloned());
    }
    Ok(CssValue::List(out))
}

/// Whether a parsed value is a valid `minmax()` side (a track breadth):
/// a length/percentage, an `fr` fraction, `min-content`, `max-content`,
/// `auto`, or math (evaluated later by the cascade). `minmax()` itself
/// cannot nest, and anything else (colors, keywords like `red`) fails
/// the whole value.
fn is_track_breadth(v: &CssValue) -> bool {
    match v {
        CssValue::Length(..) | CssValue::Percentage(_) | CssValue::Fr(_) | CssValue::Math(_) => {
            true
        }
        CssValue::Keyword(k) => matches!(
            k.to_ascii_lowercase().as_str(),
            "auto" | "min-content" | "max-content"
        ),
        _ => false,
    }
}

/// `minmax(<min>, <max>)` from grid track listings. Both sides parse as
/// track breadths (see [`is_track_breadth`]); per grid §7.2.4 the
/// minimum may not be an `fr` fraction and `minmax()` may not nest —
/// either fails the whole value (the declaration is dropped), matching
/// how `repeat()` handles its invalid forms. A minimum larger than the
/// maximum is *not* rejected here: layout clamps the track to the
/// minimum (the maximum is ignored).
fn parse_minmax(body: &[Token], depth: u32, preserve_case: bool) -> Result<CssValue, ValueError> {
    // Split at the first top-level comma: the minimum, then the maximum.
    let mut paren_depth: u32 = 0;
    let mut split = None;
    for (i, t) in body.iter().enumerate() {
        match t {
            Token::LParen | Token::Function(_) | Token::LBracket => paren_depth += 1,
            Token::RParen | Token::RBracket => paren_depth = paren_depth.saturating_sub(1),
            Token::Comma if paren_depth == 0 => {
                split = Some(i);
                break;
            }
            _ => {}
        }
    }
    let split = split.ok_or(ValueError)?;
    let min = parse_value_depth(&body[..split], depth + 1, preserve_case)?;
    let max = parse_value_depth(&body[split + 1..], depth + 1, preserve_case)?;
    if !is_track_breadth(&min) || !is_track_breadth(&max) {
        return Err(ValueError);
    }
    // An `fr` minimum is invalid at computed-value time (grid §7.2.4).
    if matches!(min, CssValue::Fr(_)) {
        return Err(ValueError);
    }
    Ok(CssValue::MinMax(Box::new(min), Box::new(max)))
}

fn parse_rgb(body: &[Token]) -> Result<CssValue, ValueError> {
    let args = fn_args(body);
    let (parts, alpha) = split_color_args(&args)?;
    if parts.len() != 3 {
        return Err(ValueError);
    }
    let mut rgb = [0u8; 3];
    for (i, part) in parts.iter().enumerate() {
        rgb[i] = match part {
            [Token::Number { value: n, .. }] => clamp_u8(n.round()),
            [Token::Percentage(p)] => clamp_u8((p / 100.0 * 255.0).round()),
            _ => return Err(ValueError),
        };
    }
    let a = match alpha {
        None => 255,
        Some(toks) => parse_alpha(toks)?,
    };
    Ok(CssValue::Color(Color {
        r: rgb[0],
        g: rgb[1],
        b: rgb[2],
        a,
    }))
}

/// Component groups plus optional alpha group from [`split_color_args`].
type ColorArgs<'a> = (Vec<&'a [&'a Token]>, Option<&'a [&'a Token]>);

/// Split a color function's arguments into component groups plus an optional
/// alpha group, enforcing the grammar:
///
/// * comma syntax (`rgb(1, 2, 3)`, `rgba(1, 2, 3, 0.5)`) and space syntax
///   (`rgb(1 2 3 / 0.5)`) never mix: a `/` in comma syntax is invalid;
/// * in space syntax the alpha must be `/`-separated: a bare 4th argument
///   (`rgb(1 2 3 0.5)`) is invalid;
/// * empty components (`hsl(0,,100%,50%)`) are invalid.
fn split_color_args<'a>(args: &'a [&'a Token]) -> Result<ColorArgs<'a>, ValueError> {
    let slash = args.iter().position(|t| matches!(t, Token::Delim('/')));
    let (end, slash_alpha) = match slash {
        Some(i) => (i, Some(&args[i + 1..])),
        None => (args.len(), None),
    };
    let comps: &'a [&'a Token] = &args[..end];
    if comps.iter().any(|t| matches!(t, Token::Comma)) {
        if slash.is_some() {
            return Err(ValueError);
        }
        let mut parts: Vec<&'a [&'a Token]> = Vec::new();
        let mut start = 0;
        for (i, t) in comps.iter().enumerate() {
            if matches!(t, Token::Comma) {
                parts.push(&comps[start..i]);
                start = i + 1;
            }
        }
        parts.push(&comps[start..]);
        if parts.iter().any(|p| p.is_empty()) {
            return Err(ValueError);
        }
        // Comma syntax may carry alpha as a 4th argument.
        let alpha = if parts.len() == 4 {
            Some(parts.pop().expect("fourth component group"))
        } else {
            None
        };
        return Ok((parts, alpha));
    }
    // Space syntax: exactly three components; alpha only via `/`.
    if comps.len() != 3 {
        return Err(ValueError);
    }
    let parts: Vec<&'a [&'a Token]> = (0..comps.len()).map(|i| &args[i..i + 1]).collect();
    Ok((parts, slash_alpha))
}

fn parse_alpha(toks: &[&Token]) -> Result<u8, ValueError> {
    match toks {
        [Token::Number { value: n, .. }] => Ok(clamp_u8((n.clamp(0.0, 1.0) * 255.0).round())),
        [Token::Percentage(p)] => Ok(clamp_u8((p.clamp(0.0, 100.0) / 100.0 * 255.0).round())),
        _ => Err(ValueError),
    }
}

fn parse_hsl(body: &[Token]) -> Result<CssValue, ValueError> {
    let args = fn_args(body);
    let (parts, alpha) = split_color_args(&args)?;
    if parts.len() != 3 {
        return Err(ValueError);
    }
    let h = parse_hue_angle(parts[0])?;
    let s = match parts[1] {
        [Token::Percentage(p)] => p.clamp(0.0, 100.0) / 100.0,
        _ => return Err(ValueError),
    };
    let l = match parts[2] {
        [Token::Percentage(p)] => p.clamp(0.0, 100.0) / 100.0,
        _ => return Err(ValueError),
    };
    let (r, g, b) = hsl_to_rgb(h, s, l);
    let a = match alpha {
        None => 255,
        Some(toks) => parse_alpha(toks)?,
    };
    Ok(CssValue::Color(Color { r, g, b, a }))
}

/// Hue as a fraction of a full turn in [0, 1). A bare number is degrees;
/// `deg`, `rad`, `grad`, and `turn` units are converted to degrees first.
fn parse_hue_angle(part: &[&Token]) -> Result<f32, ValueError> {
    let deg = match part {
        [Token::Number { value: n, .. }] => *n,
        [Token::Dimension { value: n, unit, .. }] => match unit.to_ascii_lowercase().as_str() {
            "deg" => *n,
            "rad" => *n * 180.0 / std::f32::consts::PI,
            "grad" => *n * 360.0 / 400.0,
            "turn" => *n * 360.0,
            _ => return Err(ValueError),
        },
        _ => return Err(ValueError),
    };
    Ok(deg.rem_euclid(360.0) / 360.0)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let (r, g, b) = hsl_to_rgb_f(h, s, l);
    (quantize_u8(r), quantize_u8(g), quantize_u8(b))
}

/// HSL to sRGB as 0..1 floats.
fn hsl_to_rgb_f(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - ((h6 % 2.0) - 1.0).abs());
    let (r1, g1, b1) = match h6 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    (r1 + m, g1 + m, b1 + m)
}

fn quantize_u8(v: f32) -> u8 {
    clamp_u8((v * 255.0).round())
}

fn parse_hwb(body: &[Token]) -> Result<CssValue, ValueError> {
    let args = fn_args(body);
    let (parts, alpha) = split_color_args(&args)?;
    if parts.len() != 3 {
        return Err(ValueError);
    }
    let h = parse_hue_angle(parts[0])?;
    let w = match parts[1] {
        [Token::Percentage(p)] => p.clamp(0.0, 100.0) / 100.0,
        _ => return Err(ValueError),
    };
    let b = match parts[2] {
        [Token::Percentage(p)] => p.clamp(0.0, 100.0) / 100.0,
        _ => return Err(ValueError),
    };
    let (r, g, b) = hwb_to_rgb(h, w, b);
    let a = match alpha {
        None => 255,
        Some(toks) => parse_alpha(toks)?,
    };
    Ok(CssValue::Color(Color { r, g, b, a }))
}

/// HWB to sRGB. `h` is a fraction of a turn in [0, 1); `w`/`b` are
/// whiteness/blackness in [0, 1].
fn hwb_to_rgb(h: f32, w: f32, b: f32) -> (u8, u8, u8) {
    // Whiteness + blackness over 1 renormalize proportionally (CSS Color 4).
    let (w, b) = if w + b > 1.0 {
        let sum = w + b;
        (w / sum, b / sum)
    } else {
        (w, b)
    };
    let (r, g, bl) = hsl_to_rgb_f(h, 1.0, 0.5);
    let mix = |c: f32| quantize_u8(c * (1.0 - w - b) + w);
    (mix(r), mix(g), mix(bl))
}

fn parse_lab(body: &[Token]) -> Result<CssValue, ValueError> {
    let args = fn_args(body);
    let (parts, alpha) = split_color_args(&args)?;
    if parts.len() != 3 {
        return Err(ValueError);
    }
    let l = parse_lab_lightness(parts[0])?;
    let a = parse_lab_ab(parts[1])?;
    let b = parse_lab_ab(parts[2])?;
    let (r, g, b) = lab_to_rgb(l, a, b);
    let a = match alpha {
        None => 255,
        Some(toks) => parse_alpha(toks)?,
    };
    Ok(CssValue::Color(Color { r, g, b, a }))
}

fn parse_lch(body: &[Token]) -> Result<CssValue, ValueError> {
    let args = fn_args(body);
    let (parts, alpha) = split_color_args(&args)?;
    if parts.len() != 3 {
        return Err(ValueError);
    }
    let l = parse_lab_lightness(parts[0])?;
    let c = match parts[1] {
        [Token::Number { value: n, .. }] => n.max(0.0),
        [Token::Percentage(p)] => (p * 1.5).max(0.0),
        _ => return Err(ValueError),
    };
    let h_rad = parse_hue_angle(parts[2])? * 2.0 * std::f32::consts::PI;
    let (r, g, b) = lab_to_rgb(l, c * h_rad.cos(), c * h_rad.sin());
    let a = match alpha {
        None => 255,
        Some(toks) => parse_alpha(toks)?,
    };
    Ok(CssValue::Color(Color { r, g, b, a }))
}

/// Lab/LCH lightness: a number or percentage in 0..100.
fn parse_lab_lightness(part: &[&Token]) -> Result<f32, ValueError> {
    match part {
        [Token::Number { value: n, .. }] => Ok(n.clamp(0.0, 100.0)),
        [Token::Percentage(p)] => Ok(p.clamp(0.0, 100.0)),
        _ => Err(ValueError),
    }
}

/// Lab a/b axes: bare numbers, or percentages of +/-125 (CSS Color 4).
fn parse_lab_ab(part: &[&Token]) -> Result<f32, ValueError> {
    match part {
        [Token::Number { value: n, .. }] => Ok(*n),
        [Token::Percentage(p)] => Ok(p * 1.25),
        _ => Err(ValueError),
    }
}

/// CIELAB (D65) to sRGB: Lab -> XYZ -> linear sRGB -> gamma-encoded sRGB.
fn lab_to_rgb(l: f32, a: f32, b: f32) -> (u8, u8, u8) {
    let fy = (l + 16.0) / 116.0;
    // D65 reference white.
    let x = 0.95047 * lab_f_inv(fy + a / 500.0);
    let y = lab_f_inv(fy);
    let z = 1.08883 * lab_f_inv(fy - b / 200.0);
    // XYZ (D65) -> linear sRGB.
    let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.969266 * x + 1.8760108 * y + 0.0415560 * z;
    let bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    (srgb_encode(r), srgb_encode(g), srgb_encode(bl))
}

/// Inverse of the Lab `f` nonlinearity.
fn lab_f_inv(t: f32) -> f32 {
    const EPSILON: f32 = 216.0 / 24389.0;
    const KAPPA: f32 = 24389.0 / 27.0;
    let t3 = t * t * t;
    if t3 > EPSILON {
        t3
    } else {
        (116.0 * t - 16.0) / KAPPA
    }
}

/// Linear-light to sRGB transfer function, quantized to u8.
fn srgb_encode(c: f32) -> u8 {
    let c = if c > 0.0031308 {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * c
    };
    quantize_u8(c)
}

fn clamp_u8(v: f32) -> u8 {
    v.clamp(0.0, 255.0) as u8
}

/// Parse a `calc()`, `min()`, `max()`, or `clamp()` function body into a
/// [`MathExpr`]. `name` is already lowercased.
fn parse_math_function(name: &str, body: &[Token], depth: u32) -> Result<CssValue, ValueError> {
    let toks: Vec<&Token> = body
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .collect();
    parse_math_fn_body(name, &toks, depth, false).map(CssValue::Math)
}

/// Parse a `calc()`/`min()`/`max()`/`clamp()` from an already-collected
/// function body (the `Function` token and closing paren excluded), for
/// token-level shorthand parsers such as the `font` shorthand that cannot
/// go through [`parse_value`].
pub(crate) fn parse_math_value(
    name: &str,
    body: &[&Token],
    depth: u32,
) -> Result<CssValue, ValueError> {
    let toks: Vec<&Token> = body
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .copied()
        .collect();
    parse_math_fn_body(&name.to_ascii_lowercase(), &toks, depth, false).map(CssValue::Math)
}

/// Parse a `calc-size(<basis>, <calc-sum>)` function body into a
/// [`CssValue::CalcSize`]. The body holds exactly two top-level
/// comma-separated arguments: the basis (a sizing keyword, a
/// length/percentage, or a `calc()`/`min()`/`max()`/`clamp()`) and the
/// calc-sum, in which the `size` interpolation keyword is a valid
/// operand (parsed with the `size_leaf` flag).
fn parse_calc_size(body: &[Token], depth: u32) -> Result<CssValue, ValueError> {
    if depth > MAX_MATH_DEPTH {
        return Err(ValueError);
    }
    let toks: Vec<&Token> = body
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .collect();
    let args = split_math_args(&toks)?;
    if args.len() != 2 {
        return Err(ValueError);
    }
    let basis = parse_calc_size_basis(args[0], depth + 1)?;
    let calc = parse_calc_expr(args[1], depth + 1, true)?;
    Ok(CssValue::CalcSize {
        basis: Box::new(basis),
        calc: Box::new(calc),
    })
}

/// Parse the `<calc-size-basis>` of a `calc-size()`: a sizing keyword
/// (`auto`, `max-content`, `min-content`, `fit-content`,
/// `stretch`/`available`), a length, a percentage, a math function
/// (`calc()`/`min()`/`max()`/`clamp()` and the other math functions), or
/// a `var()` reference. The
/// `var()` is preserved verbatim (name plus raw fallback tokens):
/// per CSS Values 4 substitution precedes grammar checking, so the
/// cascade substitutes it before basis validation (see
/// `Styler::eval_calc_size_for_prop` in cascade.rs).
pub(crate) fn parse_calc_size_basis(
    toks: &[&Token],
    depth: u32,
) -> Result<CalcSizeBasis, ValueError> {
    if depth > MAX_MATH_DEPTH {
        return Err(ValueError);
    }
    match toks {
        [Token::Ident(kw)] => match kw.to_ascii_lowercase().as_str() {
            "auto" => Ok(CalcSizeBasis::Auto),
            "max-content" => Ok(CalcSizeBasis::MaxContent),
            "min-content" => Ok(CalcSizeBasis::MinContent),
            "fit-content" => Ok(CalcSizeBasis::FitContent),
            "stretch" | "available" => Ok(CalcSizeBasis::Stretch),
            _ => Err(ValueError),
        },
        [Token::Dimension { value: n, unit, .. }] => match math_unit(unit) {
            Some(u) => Ok(CalcSizeBasis::Length(*n, u)),
            None => Err(ValueError),
        },
        [Token::Percentage(p)] => Ok(CalcSizeBasis::Percentage(*p)),
        // CSS-wide rule: unitless zero is a valid `<length>`.
        [Token::Number { value: n, .. }] if *n == 0.0 => {
            Ok(CalcSizeBasis::Length(0.0, LengthUnit::Px))
        }
        [Token::Function(name), .., Token::RParen] => {
            let lower = name.to_ascii_lowercase();
            // A `var()` basis cannot be validated yet: substitution
            // precedes grammar checking, so it is preserved verbatim
            // for the cascade.
            if lower == "var" {
                return parse_calc_size_basis_var(&toks[1..toks.len() - 1], depth);
            }
            if !is_math_fn_name(&lower) {
                return Err(ValueError);
            }
            let inner = &toks[1..toks.len() - 1];
            parse_math_fn_body(&lower, inner, depth, false).map(CalcSizeBasis::Math)
        }
        _ => Err(ValueError),
    }
}

/// Parse `var(--name)` / `var(--name, fallback)` in `calc-size()`
/// basis position. The name must be a `--`-prefixed ident. The fallback
/// is kept as raw tokens: it may be a sizing keyword (e.g.
/// `var(--x, max-content)`), which the math grammar cannot represent,
/// so it is checked against the basis grammar only if the property is
/// undefined at computed-value time. An empty fallback behaves like no
/// fallback (substituting nothing is invalid for a basis).
fn parse_calc_size_basis_var(body: &[&Token], depth: u32) -> Result<CalcSizeBasis, ValueError> {
    if depth > MAX_MATH_DEPTH {
        return Err(ValueError);
    }
    // Split the name from the fallback at the top-level comma.
    let mut comma = None;
    let mut paren_depth: u32 = 0;
    for (i, t) in body.iter().enumerate() {
        match t {
            Token::Function(_) | Token::LParen | Token::LBracket => paren_depth += 1,
            Token::RParen | Token::RBracket => paren_depth = paren_depth.saturating_sub(1),
            Token::Comma if paren_depth == 0 => {
                comma = Some(i);
                break;
            }
            _ => {}
        }
    }
    let (name_toks, fallback_toks) = match comma {
        Some(i) => (&body[..i], Some(&body[i + 1..])),
        None => (body, None),
    };
    let name_toks: Vec<&Token> = name_toks
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .copied()
        .collect();
    let name = match name_toks.as_slice() {
        [Token::Ident(n)] if n.starts_with("--") => n.clone(),
        _ => return Err(ValueError),
    };
    let fallback = match fallback_toks {
        Some(ft) => {
            // Kept verbatim (whitespace included) so serialization
            // round-trips; the cascade filters whitespace before
            // checking the fallback against the basis grammar.
            let ft: Vec<Token> = ft.iter().map(|t| (*t).clone()).collect();
            let non_ws = ft.iter().any(|t| !matches!(t, Token::Whitespace));
            if non_ws { Some(ft) } else { None }
        }
        None => None,
    };
    Ok(CalcSizeBasis::Var { name, fallback })
}

/// Split a `min()`/`max()`/`clamp()` argument list on top-level commas.
/// Empty arguments are invalid.
fn split_math_args<'a>(toks: &'a [&'a Token]) -> Result<Vec<&'a [&'a Token]>, ValueError> {
    let mut args: Vec<&[&Token]> = Vec::new();
    let mut start = 0;
    let mut depth: u32 = 0;
    for (i, t) in toks.iter().enumerate() {
        match t {
            Token::Function(_) | Token::LParen | Token::LBracket => depth += 1,
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            Token::Comma if depth == 0 => {
                args.push(&toks[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    args.push(&toks[start..]);
    if args.iter().any(|a| a.is_empty()) {
        return Err(ValueError);
    }
    Ok(args)
}

/// Parse a full calc expression: `<calc-sum>` consuming all of `toks`.
/// Depth-capped: every paren group, nested math function, and `var()`
/// fallback consumes one level of the [`MAX_MATH_DEPTH`] budget.
/// `size_leaf` allows the `calc-size()` `size` interpolation keyword as
/// an operand (see [`CalcParser::size_leaf`]).
fn parse_calc_expr(toks: &[&Token], depth: u32, size_leaf: bool) -> Result<MathExpr, ValueError> {
    if depth > MAX_MATH_DEPTH {
        return Err(ValueError);
    }
    let mut p = CalcParser {
        toks,
        pos: 0,
        depth,
        unary_depth: 0,
        size_leaf,
    };
    let e = p.parse_sum()?;
    if p.pos != toks.len() {
        return Err(ValueError);
    }
    Ok(e)
}

/// Recursive-descent parser for the calc grammar:
///
/// ```text
/// <calc-sum>     = <calc-product> (('+' | '-') <calc-product>)*
/// <calc-product> = <calc-value> (('*' | '/') <calc-value>)*
/// <calc-value>   = number | dimension | percentage | '(' <calc-sum> ')'
///                | <math-function> | ('+' | '-') <calc-value>
/// ```
struct CalcParser<'a> {
    toks: &'a [&'a Token],
    pos: usize,
    /// Nesting depth of paren groups and math functions so far; every
    /// descent past [`MAX_MATH_DEPTH`] fails the parse.
    depth: u32,
    /// Leading `+`/`-` signs consumed for the current operand. Each sign
    /// nests one `parse_unary` frame, so the run is capped at
    /// [`MAX_MATH_DEPTH`]: `calc(` followed by 100k `-` must fail the
    /// parse, not overflow the stack. Reset by [`CalcParser::parse_primary`]
    /// once the operand's primary is reached.
    unary_depth: u32,
    /// Whether the `calc-size()` `size` interpolation keyword parses as
    /// an operand ([`MathExpr::Size`]). Set only for the calc-sum of a
    /// `calc-size()` (and math functions nested inside it); a bare
    /// `calc()` never accepts `size`.
    size_leaf: bool,
}

impl<'a> CalcParser<'a> {
    fn peek(&self) -> Option<&'a Token> {
        self.toks.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<&'a Token> {
        let t = self.toks.get(self.pos).copied();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn parse_sum(&mut self) -> Result<MathExpr, ValueError> {
        let mut left = self.parse_product()?;
        loop {
            match self.peek() {
                Some(Token::Delim('+')) => {
                    self.bump();
                    left = MathExpr::Add(Box::new(left), Box::new(self.parse_product()?));
                }
                Some(Token::Delim('-')) => {
                    self.bump();
                    left = MathExpr::Sub(Box::new(left), Box::new(self.parse_product()?));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_product(&mut self) -> Result<MathExpr, ValueError> {
        let mut left = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(Token::Delim('*')) => {
                    self.bump();
                    left = MathExpr::Mul(Box::new(left), Box::new(self.parse_unary()?));
                }
                Some(Token::Delim('/')) => {
                    self.bump();
                    left = MathExpr::Div(Box::new(left), Box::new(self.parse_unary()?));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// Account for one leading `+`/`-` sign: each sign nests one
    /// `parse_unary` frame, so a run past [`MAX_MATH_DEPTH`] fails the
    /// parse instead of overflowing the stack.
    fn bump_unary_depth(&mut self) -> Result<(), ValueError> {
        self.unary_depth += 1;
        if self.unary_depth > MAX_MATH_DEPTH {
            return Err(ValueError);
        }
        Ok(())
    }

    fn parse_unary(&mut self) -> Result<MathExpr, ValueError> {
        match self.peek() {
            Some(Token::Delim('-')) => {
                self.bump();
                self.bump_unary_depth()?;
                Ok(MathExpr::Neg(Box::new(self.parse_unary()?)))
            }
            Some(Token::Delim('+')) => {
                self.bump();
                self.bump_unary_depth()?;
                self.parse_unary()
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<MathExpr, ValueError> {
        // The operand's primary ends the leading-sign run.
        self.unary_depth = 0;
        match self.bump() {
            Some(Token::Number { value: n, .. }) => Ok(MathExpr::Number(*n)),
            Some(Token::Percentage(p)) => Ok(MathExpr::Percentage(*p)),
            Some(Token::Dimension { value: n, unit, .. }) => match math_unit(unit) {
                Some(u) => Ok(MathExpr::Length(*n, u)),
                // Angles keep their type: `deg`/`grad`/`rad`/`turn`
                // become `MathExpr::Angle` leaves holding radians (see
                // [`AngleUnit`]), so the `<angle>`/`<number>` distinction
                // survives evaluation (CSS Values 4 §10.4).
                None => match angle_unit(unit) {
                    Some(a) => Ok(MathExpr::Angle(a.to_radians(*n))),
                    None => Err(ValueError),
                },
            },
            // The `calc-size()` `size` interpolation keyword: only valid
            // when parsing a `calc-size()` calc-sum (`size_leaf`).
            Some(Token::Ident(name)) if self.size_leaf && name.eq_ignore_ascii_case("size") => {
                Ok(MathExpr::Size)
            }
            Some(Token::LParen) => {
                let depth = self.depth + 1;
                if depth > MAX_MATH_DEPTH {
                    return Err(ValueError);
                }
                // A fresh parser for the group keeps the depth budget
                // explicit; positions are synced back afterwards.
                let mut sub = CalcParser {
                    toks: self.toks,
                    pos: self.pos,
                    depth,
                    unary_depth: 0,
                    size_leaf: self.size_leaf,
                };
                let e = sub.parse_sum()?;
                self.pos = sub.pos;
                match self.bump() {
                    Some(Token::RParen) => Ok(e),
                    _ => Err(ValueError),
                }
            }
            Some(Token::Function(name)) => {
                if name.eq_ignore_ascii_case("var") {
                    self.parse_math_var()
                } else {
                    self.parse_nested_math_fn(name)
                }
            }
            _ => Err(ValueError),
        }
    }

    /// Collect the raw body of a function whose `Function` token was
    /// already consumed: tokens up to the matching `)` (whitespace kept).
    fn collect_fn_body(&mut self) -> Result<Vec<&'a Token>, ValueError> {
        let mut body: Vec<&Token> = Vec::new();
        let mut depth: u32 = 0;
        loop {
            match self.bump() {
                None => return Err(ValueError),
                Some(Token::RParen) if depth == 0 => break,
                Some(t @ (Token::Function(_) | Token::LParen | Token::LBracket)) => {
                    depth += 1;
                    body.push(t);
                }
                Some(t @ (Token::RParen | Token::RBracket)) => {
                    depth = depth.saturating_sub(1);
                    body.push(t);
                }
                Some(t) => body.push(t),
            }
        }
        Ok(body)
    }

    /// Parse `var(--name)` / `var(--name, fallback)` inside a math
    /// expression. The fallback must itself parse as a calc operand; the
    /// whole `var()` becomes a [`MathExpr::Var`] leaf that the cascade
    /// substitutes before evaluation.
    fn parse_math_var(&mut self) -> Result<MathExpr, ValueError> {
        let body = self.collect_fn_body()?;
        // Split the name from the fallback at the top-level comma.
        let mut comma = None;
        let mut depth: u32 = 0;
        for (i, t) in body.iter().enumerate() {
            match t {
                Token::Function(_) | Token::LParen | Token::LBracket => depth += 1,
                Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
                Token::Comma if depth == 0 => {
                    comma = Some(i);
                    break;
                }
                _ => {}
            }
        }
        let (name_toks, fallback_toks) = match comma {
            Some(i) => (&body[..i], Some(&body[i + 1..])),
            None => (&body[..], None),
        };
        let name_toks: Vec<&Token> = name_toks
            .iter()
            .filter(|t| !matches!(t, Token::Whitespace))
            .copied()
            .collect();
        let name = match name_toks.as_slice() {
            [Token::Ident(n)] if n.starts_with("--") => n.clone(),
            _ => return Err(ValueError),
        };
        let fallback = match fallback_toks {
            Some(ft) => {
                let ft: Vec<&Token> = ft
                    .iter()
                    .filter(|t| !matches!(t, Token::Whitespace))
                    .copied()
                    .collect();
                if ft.is_empty() {
                    return Err(ValueError);
                }
                Some(Box::new(parse_calc_expr(
                    &ft,
                    self.depth + 1,
                    self.size_leaf,
                )?))
            }
            None => None,
        };
        Ok(MathExpr::Var(name, fallback))
    }

    /// Parse a math function nested inside a calc expression, e.g. the
    /// `min(..)` in `calc(min(1px, 2px) * 2)`. Anything that is not a math
    /// function (`var()`, `rgb()`, ...) is rejected. The `size_leaf`
    /// flag rides along so `size` stays valid inside math functions
    /// nested in a `calc-size()` calc-sum (e.g.
    /// `calc-size(100px, min(size, 200px))`).
    fn parse_nested_math_fn(&mut self, name: &str) -> Result<MathExpr, ValueError> {
        let body = self.collect_fn_body()?;
        let body: Vec<&Token> = body
            .into_iter()
            .filter(|t| !matches!(t, Token::Whitespace))
            .collect();
        parse_math_fn_body(
            &name.to_ascii_lowercase(),
            &body,
            self.depth + 1,
            self.size_leaf,
        )
    }
}

/// Whether `name` (already lowercased) is a math function that parses
/// into a [`MathExpr`] tree: `calc()`/`min()`/`max()`/`clamp()` plus the
/// CSS Values 4 math functions (`abs()`, `sign()`, `round()`, `mod()`,
/// `rem()`, `hypot()`, `sqrt()`, `pow()`, `exp()`, `log()`, the trig
/// functions) and `if()`/`toggle()`.
pub(crate) fn is_math_fn_name(name: &str) -> bool {
    matches!(
        name,
        "calc"
            | "min"
            | "max"
            | "clamp"
            | "abs"
            | "sign"
            | "round"
            | "mod"
            | "rem"
            | "hypot"
            | "sqrt"
            | "pow"
            | "exp"
            | "log"
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "atan2"
            | "if"
            | "toggle"
    )
}

/// Body-level dispatch shared by top-level and nested math functions.
/// `depth` is the nesting level of this function (capped at
/// [`MAX_MATH_DEPTH`]). `size_leaf` allows the `calc-size()` `size`
/// interpolation keyword as an operand; it is only ever true when
/// parsing (transitively) inside a `calc-size()` calc-sum.
fn parse_math_fn_body(
    name: &str,
    body: &[&Token],
    depth: u32,
    size_leaf: bool,
) -> Result<MathExpr, ValueError> {
    if depth > MAX_MATH_DEPTH {
        return Err(ValueError);
    }
    match name {
        "calc" => parse_calc_expr(body, depth, size_leaf),
        "min" | "max" => {
            let mut items = Vec::new();
            for arg in split_math_args(body)? {
                items.push(parse_calc_expr(arg, depth, size_leaf)?);
            }
            if items.is_empty() {
                return Err(ValueError);
            }
            Ok(if name == "min" {
                MathExpr::Min(items)
            } else {
                MathExpr::Max(items)
            })
        }
        "clamp" => {
            let args = split_math_args(body)?;
            if args.len() != 3 {
                return Err(ValueError);
            }
            Ok(MathExpr::Clamp(
                Box::new(parse_calc_expr(args[0], depth, size_leaf)?),
                Box::new(parse_calc_expr(args[1], depth, size_leaf)?),
                Box::new(parse_calc_expr(args[2], depth, size_leaf)?),
            ))
        }
        // Single-`<calc-sum>` functions (CSS Values 4 §10.4–10.6). A
        // top-level comma fails inside `parse_calc_expr`, so
        // `abs(1px, 2px)` is rejected without a separate arity check.
        "abs" => Ok(MathExpr::Abs(Box::new(parse_calc_expr(
            body, depth, size_leaf,
        )?))),
        "sign" => Ok(MathExpr::Sign(Box::new(parse_calc_expr(
            body, depth, size_leaf,
        )?))),
        "sqrt" => Ok(MathExpr::Sqrt(Box::new(parse_calc_expr(
            body, depth, size_leaf,
        )?))),
        "exp" => Ok(MathExpr::Exp(Box::new(parse_calc_expr(
            body, depth, size_leaf,
        )?))),
        "sin" => Ok(MathExpr::Sin(Box::new(parse_calc_expr(
            body, depth, size_leaf,
        )?))),
        "cos" => Ok(MathExpr::Cos(Box::new(parse_calc_expr(
            body, depth, size_leaf,
        )?))),
        "tan" => Ok(MathExpr::Tan(Box::new(parse_calc_expr(
            body, depth, size_leaf,
        )?))),
        "asin" => Ok(MathExpr::Asin(Box::new(parse_calc_expr(
            body, depth, size_leaf,
        )?))),
        "acos" => Ok(MathExpr::Acos(Box::new(parse_calc_expr(
            body, depth, size_leaf,
        )?))),
        "atan" => Ok(MathExpr::Atan(Box::new(parse_calc_expr(
            body, depth, size_leaf,
        )?))),
        // Two-`<calc-sum>` functions.
        "mod" => {
            let (a, b) = parse_two_calc_sums(body, depth, size_leaf)?;
            Ok(MathExpr::Mod(Box::new(a), Box::new(b)))
        }
        "rem" => {
            let (a, b) = parse_two_calc_sums(body, depth, size_leaf)?;
            Ok(MathExpr::Rem(Box::new(a), Box::new(b)))
        }
        "pow" => {
            let (a, b) = parse_two_calc_sums(body, depth, size_leaf)?;
            Ok(MathExpr::Pow(Box::new(a), Box::new(b)))
        }
        "atan2" => {
            let (a, b) = parse_two_calc_sums(body, depth, size_leaf)?;
            Ok(MathExpr::Atan2(Box::new(a), Box::new(b)))
        }
        // `round(<rounding-strategy>?, A, B?)` (CSS Values 4 §10.3): an
        // optional leading strategy keyword, then one or two calculations.
        "round" => {
            let args = split_math_args(body)?;
            let (strategy, rest) = match args.first().and_then(|a| round_strategy_of(a)) {
                Some(s) => (s, &args[1..]),
                None => (RoundStrategy::Nearest, args.as_slice()),
            };
            match rest {
                [a] => Ok(MathExpr::Round {
                    strategy,
                    a: Box::new(parse_calc_expr(a, depth, size_leaf)?),
                    b: None,
                }),
                [a, b] => Ok(MathExpr::Round {
                    strategy,
                    a: Box::new(parse_calc_expr(a, depth, size_leaf)?),
                    b: Some(Box::new(parse_calc_expr(b, depth, size_leaf)?)),
                }),
                _ => Err(ValueError),
            }
        }
        // `log(A, B?)` (CSS Values 4 §10.5): the base defaults to e.
        "log" => {
            let args = split_math_args(body)?;
            match args.as_slice() {
                [a] => Ok(MathExpr::Log {
                    a: Box::new(parse_calc_expr(a, depth, size_leaf)?),
                    b: None,
                }),
                [a, b] => Ok(MathExpr::Log {
                    a: Box::new(parse_calc_expr(a, depth, size_leaf)?),
                    b: Some(Box::new(parse_calc_expr(b, depth, size_leaf)?)),
                }),
                _ => Err(ValueError),
            }
        }
        // Comma-separated one-or-more `<calc-sum>`s.
        "hypot" => {
            let mut items = Vec::new();
            for arg in split_math_args(body)? {
                items.push(parse_calc_expr(arg, depth, size_leaf)?);
            }
            if items.is_empty() {
                return Err(ValueError);
            }
            Ok(MathExpr::Hypot(items))
        }
        "toggle" => {
            let mut items = Vec::new();
            for arg in split_math_args(body)? {
                items.push(parse_calc_expr(arg, depth, size_leaf)?);
            }
            if items.is_empty() {
                return Err(ValueError);
            }
            Ok(MathExpr::Toggle(items))
        }
        // `if( <if-condition> : <calc-sum>; ... )` (CSS Values 5 §8.3).
        "if" => parse_if_fn(body, depth, size_leaf),
        _ => Err(ValueError),
    }
}

/// Parse exactly two comma-separated `<calc-sum>`s for the two-argument
/// math functions (`mod()`, `rem()`, `pow()`, `atan2()`).
fn parse_two_calc_sums(
    body: &[&Token],
    depth: u32,
    size_leaf: bool,
) -> Result<(MathExpr, MathExpr), ValueError> {
    let args = split_math_args(body)?;
    match args.as_slice() {
        [a, b] => Ok((
            parse_calc_expr(a, depth, size_leaf)?,
            parse_calc_expr(b, depth, size_leaf)?,
        )),
        _ => Err(ValueError),
    }
}

/// If a math-function argument is a lone identifier naming a
/// `<rounding-strategy>` (`nearest | up | down | to-zero | line-width`,
/// ASCII case-insensitive), return the strategy.
fn round_strategy_of(arg: &[&Token]) -> Option<RoundStrategy> {
    match arg {
        [Token::Ident(kw)] => match kw.to_ascii_lowercase().as_str() {
            "nearest" => Some(RoundStrategy::Nearest),
            "up" => Some(RoundStrategy::Up),
            "down" => Some(RoundStrategy::Down),
            "to-zero" => Some(RoundStrategy::ToZero),
            "line-width" => Some(RoundStrategy::LineWidth),
            _ => None,
        },
        _ => None,
    }
}

/// Parse `if( [ <if-branch> ; ]* <if-branch> ;? )` (CSS Values 5 §8.3):
/// semicolon-separated `<if-condition> : <calc-sum>` branches. Branch
/// values are restricted to `<calc-sum>` — this is the math-function
/// parser, so a non-math branch value (e.g. a color keyword) fails the
/// parse and drops the declaration. An empty branch value is likewise
/// rejected: it is not a `<calc-sum>`.
fn parse_if_fn(body: &[&Token], depth: u32, size_leaf: bool) -> Result<MathExpr, ValueError> {
    let mut branches = Vec::new();
    for branch in split_if_branches(body)? {
        // Split the branch at the first top-level colon: the condition
        // is a `<declaration-value>` that excludes top-level colons, so
        // the first one always separates condition from value.
        let mut cond_end = None;
        let mut nesting: u32 = 0;
        for (i, t) in branch.iter().enumerate() {
            match t {
                Token::Function(_) | Token::LParen | Token::LBracket => nesting += 1,
                Token::RParen | Token::RBracket => nesting = nesting.saturating_sub(1),
                Token::Colon if nesting == 0 => {
                    cond_end = Some(i);
                    break;
                }
                _ => {}
            }
        }
        let c = cond_end.ok_or(ValueError)?;
        let condition = parse_if_condition(&branch[..c])?;
        let value_toks = &branch[c + 1..];
        if value_toks.is_empty() {
            return Err(ValueError);
        }
        branches.push(IfBranch {
            condition,
            value: parse_calc_expr(value_toks, depth, size_leaf)?,
        });
    }
    if branches.is_empty() {
        return Err(ValueError);
    }
    Ok(MathExpr::If(branches))
}

/// Split an `if()` body on top-level semicolons. A single trailing `;`
/// is allowed by the grammar (`;?`); empty branches are invalid.
fn split_if_branches<'a>(toks: &'a [&'a Token]) -> Result<Vec<&'a [&'a Token]>, ValueError> {
    let mut branches: Vec<&[&Token]> = Vec::new();
    let mut start = 0;
    let mut nesting: u32 = 0;
    for (i, t) in toks.iter().enumerate() {
        match t {
            Token::Function(_) | Token::LParen | Token::LBracket => nesting += 1,
            Token::RParen | Token::RBracket => nesting = nesting.saturating_sub(1),
            Token::Semicolon if nesting == 0 => {
                branches.push(&toks[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    branches.push(&toks[start..]);
    // Drop one trailing empty branch for the grammar's optional `;?`.
    if branches.last().is_some_and(|b| b.is_empty()) {
        branches.pop();
    }
    if branches.is_empty() || branches.iter().any(|b| b.is_empty()) {
        return Err(ValueError);
    }
    Ok(branches)
}

/// Parse an `if()` branch condition: the `else` keyword, or anything
/// else. Only `else` is evaluable in a math context (no media features,
/// no computed style, no feature-support table); every other condition
/// is kept verbatim as unevaluable (unknown, i.e. false at the top
/// level per CSS Values 5 §3.2). An empty condition is invalid.
fn parse_if_condition(toks: &[&Token]) -> Result<IfCondition, ValueError> {
    let non_ws: Vec<&Token> = toks
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .copied()
        .collect();
    match non_ws.as_slice() {
        [] => Err(ValueError),
        [Token::Ident(kw)] if kw.eq_ignore_ascii_case("else") => Ok(IfCondition::Else),
        _ => Ok(IfCondition::Unevaluable(
            non_ws.into_iter().cloned().collect(),
        )),
    }
}

/// Length units allowed inside math expressions.
fn math_unit(unit: &str) -> Option<LengthUnit> {
    Some(match unit.to_ascii_lowercase().as_str() {
        "px" => LengthUnit::Px,
        "cm" => LengthUnit::Cm,
        "mm" => LengthUnit::Mm,
        "in" => LengthUnit::In,
        "pt" => LengthUnit::Pt,
        "pc" => LengthUnit::Pc,
        "q" => LengthUnit::Q,
        "em" => LengthUnit::Em,
        "rem" => LengthUnit::Rem,
        "ex" => LengthUnit::Ex,
        "vw" => LengthUnit::Vw,
        "vh" => LengthUnit::Vh,
        "vmin" => LengthUnit::Vmin,
        "vmax" => LengthUnit::Vmax,
        _ => return None,
    })
}

/// Angle units allowed inside math expressions (CSS Values 4 §6.1).
/// `pub(crate)` for the cascade's `var()`-substitution angle pre-pass,
/// which folds custom-property angle dimensions to [`MathExpr::Angle`].
pub(crate) fn angle_unit(unit: &str) -> Option<AngleUnit> {
    Some(match unit.to_ascii_lowercase().as_str() {
        "deg" => AngleUnit::Deg,
        "grad" => AngleUnit::Grad,
        "rad" => AngleUnit::Rad,
        "turn" => AngleUnit::Turn,
        _ => return None,
    })
}

fn parse_var(body: &[Token], depth: u32, preserve_case: bool) -> Result<CssValue, ValueError> {
    if depth > MAX_MATH_DEPTH {
        return Err(ValueError);
    }
    // Split the name from the fallback at the FIRST top-level comma and
    // stop scanning there: the fallback slice is parsed as-is (commas
    // inside it stay significant, e.g. `var(--x, rgb(1, 2, 3))`), so each
    // nesting level costs O(1) instead of re-scanning and re-cloning the
    // whole remaining body. A full scan per level made 100k nested
    // `var()` fallbacks quadratic (~22s); this keeps it linear.
    //
    // Bracket nesting joins paren/function nesting in the depth count
    // (matching `parse_math_var` and the `split_top_*` helpers): a comma
    // inside `[]` does not start the fallback.
    let mut comma = None;
    let mut paren_depth: u32 = 0;
    for (i, t) in body.iter().enumerate() {
        match t {
            Token::LParen | Token::Function(_) | Token::LBracket => paren_depth += 1,
            Token::RParen | Token::RBracket => paren_depth = paren_depth.saturating_sub(1),
            Token::Comma if paren_depth == 0 => {
                comma = Some(i);
                break;
            }
            _ => {}
        }
    }
    let (name_toks, fallback_toks) = match comma {
        Some(i) => (&body[..i], Some(&body[i + 1..])),
        None => (body, None),
    };
    let name_toks: Vec<&Token> = name_toks
        .iter()
        .filter(|t| !matches!(t, Token::Whitespace))
        .collect();
    let name = match name_toks.as_slice() {
        [Token::Ident(n)] if n.starts_with("--") => n.clone(),
        _ => return Err(ValueError),
    };
    let fallback = match fallback_toks {
        Some(ft) if ft.iter().all(|t| matches!(t, Token::Whitespace)) => {
            return Err(ValueError); // empty fallback
        }
        // A top-level `/` or `,` in the fallback cannot survive the
        // generic value parser (the `/` is a parse error, the `,` would
        // flatten comma groups): keep the verbatim token stream so
        // substitution sees the same slash/comma structure a custom
        // property's tokens carry (see `Styler::resolve`'s `Tokens` arm:
        // the `/` still splits an enclosing list, commas still group).
        // Only the fallback slice is scanned, so the per-level cost
        // stays proportional to the fallback (bounded by
        // `MAX_MATH_DEPTH` levels overall).
        Some(ft) if fallback_has_top_level_delim(ft) => {
            Some(Box::new(CssValue::Tokens(ft.to_vec())))
        }
        Some(ft) => Some(Box::new(parse_value_depth(ft, depth + 1, preserve_case)?)),
        None => None,
    };
    Ok(CssValue::Var(name, fallback))
}

/// Whether `toks` hold a top-level `/` or `,` — i.e. one outside any
/// `()`/`[]`/function nesting. Used by [`parse_var`]: such a fallback is
/// kept as verbatim [`CssValue::Tokens`] instead of going through the
/// generic value parser, which would reject the `/` and flatten the
/// comma groups.
fn fallback_has_top_level_delim(toks: &[Token]) -> bool {
    let mut depth: u32 = 0;
    for t in toks {
        match t {
            Token::LParen | Token::Function(_) | Token::LBracket => depth += 1,
            Token::RParen | Token::RBracket => depth = depth.saturating_sub(1),
            Token::Delim('/') | Token::Comma if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

fn parse_hex(hex: &str) -> Option<Color> {
    fn digit(c: u8) -> Option<u8> {
        (c as char).to_digit(16).map(|d| d as u8)
    }
    let bytes = hex.as_bytes();
    let (r, g, b, a) = match bytes.len() {
        3 | 4 => {
            let mut v = [0u8; 4];
            for (i, &c) in bytes.iter().enumerate() {
                v[i] = digit(c)? * 17;
            }
            if bytes.len() == 3 {
                (v[0], v[1], v[2], 255)
            } else {
                (v[0], v[1], v[2], v[3])
            }
        }
        6 | 8 => {
            let mut v = [0u8; 4];
            for (i, pair) in bytes.chunks(2).enumerate() {
                v[i] = digit(pair[0])? * 16 + digit(pair[1])?;
            }
            if bytes.len() == 6 {
                (v[0], v[1], v[2], 255)
            } else {
                (v[0], v[1], v[2], v[3])
            }
        }
        _ => return None,
    };
    Some(Color { r, g, b, a })
}

/// Parse a simple CSS color from source text: one of the 148 named
/// colors (case-insensitive), `#rgb` / `#rgba` / `#rrggbb` / `#rrggbbaa`
/// hex, or `transparent`. Functional notations (`rgb()`, `hsl()`, …)
/// are NOT handled — use the full value parser for those.
///
/// Exported for `bowser_paint`'s `filter`/`drop-shadow()` argument
/// parsing, which works from retained function-argument text the way
/// the gradient parser works from retained source text.
pub fn parse_css_color(text: &str) -> Option<Color> {
    let s = text.trim();
    if s.eq_ignore_ascii_case("transparent") {
        return Some(Color {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        });
    }
    if let Some(hex) = s.strip_prefix('#') {
        return parse_hex(hex);
    }
    let (r, g, b) = named_color(&s.to_ascii_lowercase())?;
    Some(Color { r, g, b, a: 255 })
}

/// The 148 CSS named colors (lowercase name -> rgb).
fn named_color(name: &str) -> Option<(u8, u8, u8)> {
    Some(match name {
        "aliceblue" => (240, 248, 255),
        "antiquewhite" => (250, 235, 215),
        "aqua" => (0, 255, 255),
        "aquamarine" => (127, 255, 212),
        "azure" => (240, 255, 255),
        "beige" => (245, 245, 220),
        "bisque" => (255, 228, 196),
        "black" => (0, 0, 0),
        "blanchedalmond" => (255, 235, 205),
        "blue" => (0, 0, 255),
        "blueviolet" => (138, 43, 226),
        "brown" => (165, 42, 42),
        "burlywood" => (222, 184, 135),
        "cadetblue" => (95, 158, 160),
        "chartreuse" => (127, 255, 0),
        "chocolate" => (210, 105, 30),
        "coral" => (255, 127, 80),
        "cornflowerblue" => (100, 149, 237),
        "cornsilk" => (255, 248, 220),
        "crimson" => (220, 20, 60),
        "cyan" => (0, 255, 255),
        "darkblue" => (0, 0, 139),
        "darkcyan" => (0, 139, 139),
        "darkgoldenrod" => (184, 134, 11),
        "darkgray" => (169, 169, 169),
        "darkgreen" => (0, 100, 0),
        "darkgrey" => (169, 169, 169),
        "darkkhaki" => (189, 183, 107),
        "darkmagenta" => (139, 0, 139),
        "darkolivegreen" => (85, 107, 47),
        "darkorange" => (255, 140, 0),
        "darkorchid" => (153, 50, 204),
        "darkred" => (139, 0, 0),
        "darksalmon" => (233, 150, 122),
        "darkseagreen" => (143, 188, 143),
        "darkslateblue" => (72, 61, 139),
        "darkslategray" => (47, 79, 79),
        "darkslategrey" => (47, 79, 79),
        "darkturquoise" => (0, 206, 209),
        "darkviolet" => (148, 0, 211),
        "deeppink" => (255, 20, 147),
        "deepskyblue" => (0, 191, 255),
        "dimgray" => (105, 105, 105),
        "dimgrey" => (105, 105, 105),
        "dodgerblue" => (30, 144, 255),
        "firebrick" => (178, 34, 34),
        "floralwhite" => (255, 250, 240),
        "forestgreen" => (34, 139, 34),
        "fuchsia" => (255, 0, 255),
        "gainsboro" => (220, 220, 220),
        "ghostwhite" => (248, 248, 255),
        "gold" => (255, 215, 0),
        "goldenrod" => (218, 165, 32),
        "gray" => (128, 128, 128),
        "green" => (0, 128, 0),
        "greenyellow" => (173, 255, 47),
        "grey" => (128, 128, 128),
        "honeydew" => (240, 255, 240),
        "hotpink" => (255, 105, 180),
        "indianred" => (205, 92, 92),
        "indigo" => (75, 0, 130),
        "ivory" => (255, 255, 240),
        "khaki" => (240, 230, 140),
        "lavender" => (230, 230, 250),
        "lavenderblush" => (255, 240, 245),
        "lawngreen" => (124, 252, 0),
        "lemonchiffon" => (255, 250, 205),
        "lightblue" => (173, 216, 230),
        "lightcoral" => (240, 128, 128),
        "lightcyan" => (224, 255, 255),
        "lightgoldenrodyellow" => (250, 250, 210),
        "lightgray" => (211, 211, 211),
        "lightgreen" => (144, 238, 144),
        "lightgrey" => (211, 211, 211),
        "lightpink" => (255, 182, 193),
        "lightsalmon" => (255, 160, 122),
        "lightseagreen" => (32, 178, 170),
        "lightskyblue" => (135, 206, 250),
        "lightslategray" => (119, 136, 153),
        "lightslategrey" => (119, 136, 153),
        "lightsteelblue" => (176, 196, 222),
        "lightyellow" => (255, 255, 224),
        "lime" => (0, 255, 0),
        "limegreen" => (50, 205, 50),
        "linen" => (250, 240, 230),
        "magenta" => (255, 0, 255),
        "maroon" => (128, 0, 0),
        "mediumaquamarine" => (102, 205, 170),
        "mediumblue" => (0, 0, 205),
        "mediumorchid" => (186, 85, 211),
        "mediumpurple" => (147, 112, 219),
        "mediumseagreen" => (60, 179, 113),
        "mediumslateblue" => (123, 104, 238),
        "mediumspringgreen" => (0, 250, 154),
        "mediumturquoise" => (72, 209, 204),
        "mediumvioletred" => (199, 21, 133),
        "midnightblue" => (25, 25, 112),
        "mintcream" => (245, 255, 250),
        "mistyrose" => (255, 228, 225),
        "moccasin" => (255, 228, 181),
        "navajowhite" => (255, 222, 173),
        "navy" => (0, 0, 128),
        "oldlace" => (253, 245, 230),
        "olive" => (128, 128, 0),
        "olivedrab" => (107, 142, 35),
        "orange" => (255, 165, 0),
        "orangered" => (255, 69, 0),
        "orchid" => (218, 112, 214),
        "palegoldenrod" => (238, 232, 170),
        "palegreen" => (152, 251, 152),
        "paleturquoise" => (175, 238, 238),
        "palevioletred" => (219, 112, 147),
        "papayawhip" => (255, 239, 213),
        "peachpuff" => (255, 218, 185),
        "peru" => (205, 133, 63),
        "pink" => (255, 192, 203),
        "plum" => (221, 160, 221),
        "powderblue" => (176, 224, 230),
        "purple" => (128, 0, 128),
        "rebeccapurple" => (102, 51, 153),
        "red" => (255, 0, 0),
        "rosybrown" => (188, 143, 143),
        "royalblue" => (65, 105, 225),
        "saddlebrown" => (139, 69, 19),
        "salmon" => (250, 128, 114),
        "sandybrown" => (244, 164, 96),
        "seagreen" => (46, 139, 87),
        "seashell" => (255, 245, 238),
        "sienna" => (160, 82, 45),
        "silver" => (192, 192, 192),
        "skyblue" => (135, 206, 235),
        "slateblue" => (106, 90, 205),
        "slategray" => (112, 128, 144),
        "slategrey" => (112, 128, 144),
        "snow" => (255, 250, 250),
        "springgreen" => (0, 255, 127),
        "steelblue" => (70, 130, 180),
        "tan" => (210, 180, 140),
        "teal" => (0, 128, 128),
        "thistle" => (216, 191, 216),
        "tomato" => (255, 99, 71),
        "turquoise" => (64, 224, 208),
        "violet" => (238, 130, 238),
        "wheat" => (245, 222, 179),
        "white" => (255, 255, 255),
        "whitesmoke" => (245, 245, 245),
        "yellow" => (255, 255, 0),
        "yellowgreen" => (154, 205, 50),
        _ => return None,
    })
}

/// CSS system font keywords for the `font` shorthand: `caption`, `icon`,
/// `menu`, `message-box`, `small-caption`, `status-bar` (CSS Fonts 3 §3.2).
/// A system keyword must be the *sole* value of the shorthand; it
/// computes to the UA's font for that purpose, expanding to all six
/// `font` longhands. The caller decides sole-value-ness; this function
/// only maps the keyword to its six longhand values in
/// style/variant/weight/size/line-height/family order (`None` when the
/// lowercased keyword is not a system font).
///
/// UA metrics source: Bowser has no system-font query API yet (no
/// fontconfig / NSFont / `SystemParametersInfo` binding anywhere in the
/// tree — font resolution lives in `bowser_paint::fonts::FontCache` and
/// only handles generic families). These are therefore fixed placeholder
/// metrics: 13px (12px for `small-caption`) sans-serif at normal weight,
/// with `normal` line-height, which is what the UA font's own metrics
/// would supply. `sans-serif` is the generic family paint's
/// `FontCache::resolve` degrades everything to, so the placeholder always
/// paints with a real face. When a real system-font source lands,
/// replace this table (and the doc) with queried values; both `font`
/// expansion paths read it, so the swap is this one function.
pub(crate) fn system_font_values(keyword: &str) -> Option<[CssValue; 6]> {
    let size_px = match keyword {
        "caption" | "icon" | "menu" | "message-box" | "status-bar" => 13.0,
        "small-caption" => 12.0,
        _ => return None,
    };
    let normal = || CssValue::Keyword("normal".into());
    Some([
        normal(),                                  // font-style
        normal(),                                  // font-variant
        normal(),                                  // font-weight
        CssValue::Length(size_px, LengthUnit::Px), // font-size
        normal(),                                  // line-height
        CssValue::String("sans-serif".into()),     // font-family
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kwatom::KeywordAtom;
    use crate::token::tokenize;

    fn v(input: &str) -> Result<CssValue, ValueError> {
        parse_value(&tokenize(input))
    }

    fn ok(input: &str) -> CssValue {
        v(input).expect("value should parse")
    }

    fn ctx() -> MathContext {
        MathContext::new(16.0, 16.0, 800.0, 600.0)
    }

    fn math(input: &str, ctx: &MathContext) -> Option<CssValue> {
        ok(input).evaluate_math(ctx)
    }

    #[test]
    fn serialize_math_round_trips_calc_source_form() {
        // `serialize_math` renders real CSS, not the expression tree's
        // `Debug` form, so CSSOM style serialization round-trips math.
        let v = ok("calc(0px)");
        assert_eq!(v.serialize_math().as_deref(), Some("calc(0px)"));
        let v = ok("calc(10px + 0.5em)");
        assert_eq!(v.serialize_math().as_deref(), Some("calc(10px + 0.5em)"));
        // Non-math values have no math serialization.
        assert_eq!(ok("10px").serialize_math(), None);
        assert_eq!(ok("auto").serialize_math(), None);
    }

    #[test]
    fn lengths() {
        assert_eq!(ok("10px"), CssValue::Length(10.0, LengthUnit::Px));
        assert_eq!(ok("1.5em"), CssValue::Length(1.5, LengthUnit::Em));
        assert_eq!(ok("2rem"), CssValue::Length(2.0, LengthUnit::Rem));
        assert_eq!(ok("50vw"), CssValue::Length(50.0, LengthUnit::Vw));
        assert_eq!(ok("100vh"), CssValue::Length(100.0, LengthUnit::Vh));
        assert_eq!(ok("-3px"), CssValue::Length(-3.0, LengthUnit::Px));
        assert_eq!(ok("0"), CssValue::Number(0.0));
    }

    #[test]
    fn absolute_length_units() {
        assert_eq!(ok("2.54cm"), CssValue::Length(2.54, LengthUnit::Cm));
        assert_eq!(ok("25.4mm"), CssValue::Length(25.4, LengthUnit::Mm));
        assert_eq!(ok("1in"), CssValue::Length(1.0, LengthUnit::In));
        assert_eq!(ok("72pt"), CssValue::Length(72.0, LengthUnit::Pt));
        assert_eq!(ok("6pc"), CssValue::Length(6.0, LengthUnit::Pc));
        assert_eq!(ok("101.6q"), CssValue::Length(101.6, LengthUnit::Q));
        assert_eq!(ok("10vmin"), CssValue::Length(10.0, LengthUnit::Vmin));
        assert_eq!(ok("10vmax"), CssValue::Length(10.0, LengthUnit::Vmax));
        // Case-insensitive like the other units.
        assert_eq!(ok("1CM"), CssValue::Length(1.0, LengthUnit::Cm));
        assert_eq!(ok("1VMIN"), CssValue::Length(1.0, LengthUnit::Vmin));
    }

    #[test]
    fn absolute_length_units_convert_to_px() {
        // CSS Values 4 §5.2: `96px = 1in`.
        let px = |v: f32, u: &LengthUnit| length_to_px(v, u, 16.0, 16.0, 800.0, 600.0);
        assert!((px(1.0, &LengthUnit::In) - 96.0).abs() < 1e-4);
        assert!((px(2.54, &LengthUnit::Cm) - 96.0).abs() < 1e-3);
        assert!((px(25.4, &LengthUnit::Mm) - 96.0).abs() < 1e-3);
        assert!((px(72.0, &LengthUnit::Pt) - 96.0).abs() < 1e-4);
        assert!((px(6.0, &LengthUnit::Pc) - 96.0).abs() < 1e-4);
        assert!((px(101.6, &LengthUnit::Q) - 96.0).abs() < 1e-3);
        // `vmin`/`vmax` resolve against the smaller/larger viewport side.
        assert!((px(10.0, &LengthUnit::Vmin) - 60.0).abs() < 1e-4);
        assert!((px(10.0, &LengthUnit::Vmax) - 80.0).abs() < 1e-4);
    }

    #[test]
    fn unknown_unit_fails() {
        assert!(v("5xyz").is_err());
        assert!(v("10deg").is_err());
    }

    #[test]
    fn percentages_and_numbers() {
        assert_eq!(ok("50%"), CssValue::Percentage(50.0));
        assert_eq!(ok("1.5"), CssValue::Number(1.5));
        assert_eq!(ok("-2"), CssValue::Number(-2.0));
    }

    #[test]
    fn keywords_lowercased() {
        assert_eq!(ok("Auto"), CssValue::Keyword("auto".into()));
        assert_eq!(ok("NONE"), CssValue::Keyword("none".into()));
    }

    #[test]
    fn fill_available_aliases_stay_keywords_at_parse() {
        // The legacy fill-available aliases are sizing-grammar aliases
        // only (css-sizing-4; Chrome rejects them for alignment
        // properties), so the generic value parser keeps them as plain
        // lowercased keywords instead of normalizing to `stretch`. The
        // cascade's sizing validation does the `stretch` normalization
        // (see `is_valid_for_property`'s sizing arms).
        let webkit = CssValue::Keyword("-webkit-fill-available".into());
        let moz = CssValue::Keyword("-moz-available".into());
        assert_eq!(ok("-webkit-fill-available"), webkit);
        assert_eq!(ok("-moz-available"), moz);
        assert_eq!(ok("-WEBKIT-FILL-AVAILABLE"), webkit);
        assert_eq!(ok("stretch"), CssValue::Keyword("stretch".into()));
    }

    #[test]
    fn keyword_equality_and_identity_across_parses() {
        // Two parses of the same keyword (even with different authored
        // case) produce equal values sharing one interned atom: no heap
        // string per occurrence.
        let atom_of = |v: &CssValue| match v {
            CssValue::Keyword(k) => k.atom(),
            other => panic!("expected keyword, got {other:?}"),
        };
        let a = ok("auto");
        let b = ok("auto");
        let c = ok("AUTO");
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(atom_of(&a), atom_of(&b));
        assert_eq!(atom_of(&a), atom_of(&c));
        assert_eq!(atom_of(&a), KeywordAtom::lookup("auto").unwrap());

        // Unknown keywords still compare by string value across parses;
        // they ride the heap sidecar instead of the atom table.
        let u1 = ok("notacolor");
        let u2 = ok("notacolor");
        assert_eq!(u1, u2);
        assert_eq!(atom_of(&u1), KeywordAtom::UNKNOWN);
        assert_ne!(u1, a);
    }

    #[test]
    fn hex_colors() {
        assert_eq!(
            ok("#fff"),
            CssValue::Color(Color {
                r: 255,
                g: 255,
                b: 255,
                a: 255
            })
        );
        assert_eq!(
            ok("#0000"),
            CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 0
            })
        );
        assert_eq!(
            ok("#ff0000"),
            CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
        assert_eq!(
            ok("#ff000080"),
            CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 128
            })
        );
        assert_eq!(
            ok("#ABC"),
            CssValue::Color(Color {
                r: 170,
                g: 187,
                b: 204,
                a: 255
            })
        );
        assert!(v("#ff").is_err());
        assert!(v("#fffff").is_err());
        assert!(v("#ggg").is_err());
    }

    #[test]
    fn named_colors() {
        assert_eq!(
            ok("red"),
            CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
        assert_eq!(
            ok("RebeccaPurple"),
            CssValue::Color(Color {
                r: 102,
                g: 51,
                b: 153,
                a: 255
            })
        );
        assert_eq!(
            ok("transparent"),
            CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 0
            })
        );
        assert_eq!(ok("currentcolor"), CssValue::Keyword("currentcolor".into()));
        assert_eq!(ok("notacolor"), CssValue::Keyword("notacolor".into()));
    }

    #[test]
    fn rgb_comma_syntax() {
        assert_eq!(
            ok("rgb(255, 0, 0)"),
            CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
        assert_eq!(
            ok("rgba(255, 0, 0, 0.5)"),
            CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 128
            })
        );
        assert_eq!(
            ok("rgba(100%, 0%, 0%, 50%)"),
            CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 128
            })
        );
        // Out-of-range values clamp.
        assert_eq!(
            ok("rgb(300, -5, 0)"),
            CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
    }

    #[test]
    fn rgb_space_syntax() {
        assert_eq!(
            ok("rgb(255 0 0)"),
            CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
        assert_eq!(
            ok("rgb(255 0 0 / 0.5)"),
            CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 128
            })
        );
        assert_eq!(
            ok("rgba(0 0 255 / 50%)"),
            CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 128
            })
        );
        assert!(v("rgb(255, 0)").is_err());
        assert!(v("rgb(1,2,3,4,5)").is_err());
    }

    #[test]
    fn hsl_syntax() {
        // hsl(0, 100%, 50%) == red
        assert_eq!(
            ok("hsl(0, 100%, 50%)"),
            CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            })
        );
        // hsl(120, 100%, 50%) == lime
        assert_eq!(
            ok("hsl(120, 100%, 50%)"),
            CssValue::Color(Color {
                r: 0,
                g: 255,
                b: 0,
                a: 255
            })
        );
        // Space syntax with slash alpha.
        assert_eq!(
            ok("hsl(240 100% 50% / 0.5)"),
            CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 255,
                a: 128
            })
        );
        assert_eq!(
            ok("hsla(0, 0%, 0%, 1)"),
            CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            })
        );
        assert!(v("hsl(0, 50%, 50%, 1, 2)").is_err());
    }

    #[test]
    fn rgb_hsl_reject_invalid_combos() {
        // Space syntax: a bare 4th argument without `/` is invalid.
        assert!(v("rgb(1 2 3 0.5)").is_err());
        assert!(v("hsl(0 0% 0% 1)").is_err());
        // Comma syntax must not mix with `/` alpha.
        assert!(v("rgb(1,2,3 / 0.5)").is_err());
        assert!(v("hsl(0, 0%, 0% / 1)").is_err());
        assert!(v("rgba(1,2,3,4 / 0.5)").is_err());
        // Empty components are invalid.
        assert!(v("hsl(0,,100%,50%)").is_err());
        assert!(v("rgb(1,,2,3)").is_err());
        assert!(v("rgb(,1,2,3)").is_err());
        assert!(v("rgb(1,2,3,)").is_err());
        // The valid forms still parse.
        assert_eq!(
            ok("rgb(1 2 3 / 0.5)"),
            CssValue::Color(Color {
                r: 1,
                g: 2,
                b: 3,
                a: 128
            })
        );
        assert_eq!(
            ok("rgba(1,2,3,0.5)"),
            CssValue::Color(Color {
                r: 1,
                g: 2,
                b: 3,
                a: 128
            })
        );
        assert_eq!(
            ok("hsl(0 0% 0% / 1)"),
            CssValue::Color(Color {
                r: 0,
                g: 0,
                b: 0,
                a: 255
            })
        );
    }

    #[test]
    fn hsl_hue_angle_units() {
        let red = CssValue::Color(Color {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        });
        let cyan = CssValue::Color(Color {
            r: 0,
            g: 255,
            b: 255,
            a: 255,
        });
        // deg behaves like a bare number.
        assert_eq!(ok("hsl(120deg, 100%, 50%)"), ok("hsl(120, 100%, 50%)"));
        assert_eq!(ok("hsl(0deg, 100%, 50%)"), red);
        // turn / grad / rad convert to degrees.
        assert_eq!(ok("hsl(0.5turn, 100%, 50%)"), cyan);
        assert_eq!(ok("hsl(200grad, 100%, 50%)"), cyan);
        assert_eq!(
            ok("hsl(3.14159265rad, 100%, 50%)"),
            ok("hsl(180, 100%, 50%)")
        );
        // Hue wraps past 360.
        assert_eq!(ok("hsl(480deg, 100%, 50%)"), ok("hsl(120, 100%, 50%)"));
        assert_eq!(ok("hsl(1turn, 100%, 50%)"), red);
        // Unknown angle units are rejected.
        assert!(v("hsl(90foo, 100%, 50%)").is_err());
        assert!(v("hsl(90px, 100%, 50%)").is_err());
    }

    #[test]
    fn strings_and_urls() {
        assert_eq!(ok("\"hi\""), CssValue::String("hi".into()));
        assert_eq!(ok("url(a.png)"), CssValue::Url("a.png".into()));
        assert_eq!(ok("url(\"a.png\")"), CssValue::Url("a.png".into()));
    }

    #[test]
    fn gradient_functions_retained_verbatim() {
        let g = |name: &str, source: &str| CssValue::Gradient {
            name: name.into(),
            source: source.into(),
        };
        assert_eq!(
            ok("linear-gradient(red, blue)"),
            g("linear-gradient", "linear-gradient(red, blue)")
        );
        // Function names are case-insensitive; the stored name is
        // lowercased while arguments keep their authored case.
        assert_eq!(
            ok("LINEAR-GRADIENT(RED, BLUE)"),
            g("linear-gradient", "linear-gradient(RED, BLUE)")
        );
        // All six function forms parse.
        for name in [
            "linear-gradient",
            "repeating-linear-gradient",
            "radial-gradient",
            "repeating-radial-gradient",
            "conic-gradient",
            "repeating-conic-gradient",
        ] {
            let src = format!("{name}(red)");
            assert_eq!(ok(&src), g(name, &src), "{name} should parse");
        }
        // Nested functions and stop positions survive verbatim
        // (whitespace runs collapse to a single space).
        assert_eq!(
            ok("radial-gradient(circle at 50% 50%, rgba(255, 0, 0, 0.5) 10px, blue 90%)"),
            g(
                "radial-gradient",
                "radial-gradient(circle at 50% 50%, rgba(255, 0, 0, 0.5) 10px, blue 90%)"
            )
        );
        // Empty bodies are invalid and drop the declaration.
        assert!(v("linear-gradient()").is_err());
        assert!(v("conic-gradient(   )").is_err());
        // Unknown gradient-like functions stay invalid.
        assert!(v("fancy-gradient(red)").is_err());
    }

    #[test]
    fn gradient_with_var_defers_as_tokens() {
        // A `var()` in the gradient body cannot resolve until
        // computed-value time (custom properties are element-scoped):
        // the parser keeps the whole function as a verbatim token
        // stream so the cascade can substitute and re-parse it into a
        // real `Gradient`.
        let CssValue::Tokens(t) = ok("linear-gradient(var(--a), blue)") else {
            panic!("expected Tokens");
        };
        // The stream is the full function: `linear-gradient(` … `)`.
        assert!(matches!(t.first(), Some(Token::Function(n)) if n == "linear-gradient"));
        assert_eq!(t.last(), Some(&Token::RParen));
        assert!(
            t.iter()
                .any(|t| matches!(t, Token::Function(n) if n.eq_ignore_ascii_case("var")))
        );
        // Nested `var()` (e.g. in a stop position) defers too, and the
        // function-name check is case-insensitive.
        assert!(matches!(
            ok("RADIAL-GRADIENT(circle, VAR(--a) 10%, blue)"),
            CssValue::Tokens(_)
        ));
        // No `var()`: still a verbatim `Gradient`.
        assert!(matches!(
            ok("linear-gradient(red, blue)"),
            CssValue::Gradient { .. }
        ));
    }

    #[test]
    fn attr_function_parses() {
        // String attr: bare name, `string` type-or-unit, and fallback.
        assert_eq!(
            ok("attr(data-label)"),
            CssValue::Attr(Box::new(AttrData {
                name: "data-label".into(),
                fallback: None,
            }))
        );
        assert_eq!(
            ok("attr(DATA-LABEL)"),
            CssValue::Attr(Box::new(AttrData {
                name: "data-label".into(),
                fallback: None,
            }))
        );
        assert_eq!(
            ok("attr(data-label string)"),
            CssValue::Attr(Box::new(AttrData {
                name: "data-label".into(),
                fallback: None,
            }))
        );
        assert_eq!(
            ok("attr(data-label, \"fallback\")"),
            CssValue::Attr(Box::new(AttrData {
                name: "data-label".into(),
                fallback: Some(Box::new(CssValue::String("fallback".into()))),
            }))
        );
        assert_eq!(
            ok("attr(\"data-label\" string, \"fb\")"),
            CssValue::Attr(Box::new(AttrData {
                name: "data-label".into(),
                fallback: Some(Box::new(CssValue::String("fb".into()))),
            }))
        );
        // Non-string type-or-unit is out of scope: drops the
        // declaration.
        assert!(v("attr(data-label url)").is_err());
        assert!(v("attr(data-label integer)").is_err());
        // Empty name, empty body, empty fallback: invalid.
        assert!(v("attr()").is_err());
        assert!(v("attr( )").is_err());
        assert!(v("attr(data-label,)").is_err());
        assert!(v("attr(, \"fb\")").is_err());
    }

    #[test]
    fn counter_functions_parse() {
        assert_eq!(
            ok("counter(chapter)"),
            CssValue::Counter(Box::new(CounterData {
                name: "chapter".into(),
                style: None,
                separator: None,
            }))
        );
        assert_eq!(
            ok("counter(chapter, upper-roman)"),
            CssValue::Counter(Box::new(CounterData {
                name: "chapter".into(),
                style: Some("upper-roman".into()),
                separator: None,
            }))
        );
        assert_eq!(
            ok("counters(chapter, \".\")"),
            CssValue::Counter(Box::new(CounterData {
                name: "chapter".into(),
                style: None,
                separator: Some(".".into()),
            }))
        );
        assert_eq!(
            ok("counters(chapter, \".\", lower-alpha)"),
            CssValue::Counter(Box::new(CounterData {
                name: "chapter".into(),
                style: Some("lower-alpha".into()),
                separator: Some(".".into()),
            }))
        );
        // Arity: `counter()` takes 1-2 args, `counters()` 2-3.
        assert!(v("counter()").is_err());
        assert!(v("counter(a, b, c)").is_err());
        assert!(v("counters(chapter)").is_err());
        assert!(v("counters(chapter, \".\", decimal, x)").is_err());
        // The `counters()` separator must be a string; names and
        // styles are idents.
        assert!(v("counters(chapter, .)").is_err());
        assert!(v("counter(\"chapter\")").is_err());
        assert!(v("counter(chapter, \"decimal\")").is_err());
    }

    #[test]
    fn var_references() {
        assert_eq!(
            ok("var(--main-color)"),
            CssValue::Var("--main-color".into(), None)
        );
        assert_eq!(
            ok("var(--x, red)"),
            CssValue::Var(
                "--x".into(),
                Some(Box::new(CssValue::Color(Color {
                    r: 255,
                    g: 0,
                    b: 0,
                    a: 255
                })))
            )
        );
        assert_eq!(
            ok("var(--x, 1px solid)"),
            CssValue::Var(
                "--x".into(),
                Some(Box::new(CssValue::List(vec![
                    CssValue::Length(1.0, LengthUnit::Px),
                    CssValue::Keyword("solid".into()),
                ])))
            )
        );
        // Case of the custom property name is preserved.
        assert_eq!(ok("var(--MyVar)"), CssValue::Var("--MyVar".into(), None));
        assert!(v("var(x)").is_err());
        assert!(v("var()").is_err());
    }

    #[test]
    fn var_fallback_delimiter_structure() {
        // Commas inside nested functions/brackets do not start the
        // fallback: the boundary is the first top-level comma.
        assert_eq!(
            ok("var(--x, rgb(1, 2, 3))"),
            CssValue::Var(
                "--x".into(),
                Some(Box::new(CssValue::Color(Color {
                    r: 1,
                    g: 2,
                    b: 3,
                    a: 255
                })))
            )
        );
        // A top-level `/` or `,` in the fallback keeps the verbatim token
        // stream: the generic parser would reject the `/` and flatten the
        // comma groups, but substitution must still see the structure.
        assert!(matches!(
            ok("var(--x, 12px/1.5)"),
            CssValue::Var(name, Some(fb))
                if name == "--x" && matches!(*fb, CssValue::Tokens(_))
        ));
        assert!(matches!(
            ok("var(--x, Arial, sans-serif)"),
            CssValue::Var(name, Some(fb))
                if name == "--x" && matches!(*fb, CssValue::Tokens(_))
        ));
        // A `/` nested inside a function is not top-level: the fallback
        // is parsed as before (a `Math` value here, not verbatim
        // tokens).
        assert!(matches!(
            ok("var(--x, calc(1px / 2))"),
            CssValue::Var(_, Some(fb)) if matches!(*fb, CssValue::Math(_))
        ));
        // No top-level delimiter: parsed as before.
        assert!(matches!(
            ok("var(--x, 1px solid)"),
            CssValue::Var(_, Some(fb)) if matches!(*fb, CssValue::List(_))
        ));
    }

    #[test]
    fn lists() {
        assert_eq!(
            ok("1px solid"),
            CssValue::List(vec![
                CssValue::Length(1.0, LengthUnit::Px),
                CssValue::Keyword("solid".into()),
            ])
        );
        // Commas flatten into the list too.
        assert_eq!(
            ok("red, blue"),
            CssValue::List(vec![
                CssValue::Color(Color {
                    r: 255,
                    g: 0,
                    b: 0,
                    a: 255
                }),
                CssValue::Color(Color {
                    r: 0,
                    g: 0,
                    b: 255,
                    a: 255
                }),
            ])
        );
    }

    #[test]
    fn invalid_values_fail() {
        assert!(v("").is_err());
        assert!(v("   ").is_err());
        // Gradient functions are valid values now (see
        // `gradient_functions_retained_verbatim`).
        assert!(v("#fff #fff #").is_err());
    }

    #[test]
    fn calc_basic_arithmetic() {
        let ctx = ctx();
        let px = |n: f32| CssValue::Length(n, LengthUnit::Px);
        assert_eq!(math("calc(1px + 2px)", &ctx), Some(px(3.0)));
        assert_eq!(math("calc(10px - 4px)", &ctx), Some(px(6.0)));
        // Absolute and viewport-relative units mix with px in calc().
        assert_eq!(math("calc(1in + 1px)", &ctx), Some(px(97.0)));
        assert_eq!(math("calc(1cm + 1cm)", &ctx), Some(px(2.0 * 96.0 / 2.54)));
        assert_eq!(math("calc(10vmin + 10vmax)", &ctx), Some(px(140.0)));
        // `*` and `/` bind tighter than `+`.
        assert_eq!(math("calc(1px + 2px * 3)", &ctx), Some(px(7.0)));
        assert_eq!(math("calc((1px + 2px) * 3)", &ctx), Some(px(9.0)));
        // Nested calc().
        assert_eq!(math("calc(calc(1px + 2px) * 3)", &ctx), Some(px(9.0)));
        assert_eq!(
            math("calc(calc(10px / 2) - calc(1px + 1px))", &ctx),
            Some(px(3.0))
        );
        assert_eq!(math("calc(10px / 4)", &ctx), Some(px(2.5)));
        // Unitless numbers stay numbers.
        assert_eq!(math("calc(2 * 3)", &ctx), Some(CssValue::Number(6.0)));
        assert_eq!(math("calc(7 / 2)", &ctx), Some(CssValue::Number(3.5)));
        // Unary minus.
        assert_eq!(math("calc(-(1px + 2px))", &ctx), Some(px(-3.0)));
        assert_eq!(math("calc(-5px + 2px)", &ctx), Some(px(-3.0)));
        // Non-math values pass through unchanged.
        assert_eq!(ok("10px").evaluate_math(&ctx), Some(px(10.0)));
        assert_eq!(
            ok("red").evaluate_math(&ctx),
            Some(CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }))
        );
    }

    #[test]
    fn calc_mixed_units() {
        let ctx = MathContext::new(16.0, 20.0, 800.0, 600.0).with_percentage_base(200.0);
        let px = |n: f32| CssValue::Length(n, LengthUnit::Px);
        assert_eq!(math("calc(1em + 4px)", &ctx), Some(px(20.0)));
        assert_eq!(math("calc(2rem * 2)", &ctx), Some(px(80.0)));
        assert_eq!(math("calc(50vw + 10px)", &ctx), Some(px(410.0)));
        assert_eq!(math("calc(10vh - 5px)", &ctx), Some(px(55.0)));
        assert_eq!(math("calc(50% + 10px)", &ctx), Some(px(110.0)));
        assert_eq!(math("calc(100% / 4)", &ctx), Some(px(50.0)));
        // Percentages of an unset base resolve against 0.
        let plain = MathContext::new(16.0, 16.0, 800.0, 600.0);
        assert_eq!(math("calc(50%)", &plain), Some(px(0.0)));
    }

    #[test]
    fn calc_min_max() {
        let ctx = ctx();
        let px = |n: f32| CssValue::Length(n, LengthUnit::Px);
        assert_eq!(math("min(3px, 1px, 2px)", &ctx), Some(px(1.0)));
        assert_eq!(math("max(3px, 1px)", &ctx), Some(px(3.0)));
        // Mixed units inside min/max resolve against the context.
        assert_eq!(math("min(10px, 2em)", &ctx), Some(px(10.0)));
        assert_eq!(math("max(10px, 2em)", &ctx), Some(px(32.0)));
        // Nested math functions.
        assert_eq!(
            math("calc(min(10px, 2em) + max(1px, 5px))", &ctx),
            Some(px(15.0))
        );
        assert_eq!(math("min(1px, max(2px, 3px), 4px)", &ctx), Some(px(1.0)));
        // Unitless min/max stay numbers.
        assert_eq!(math("max(1, 5, 3)", &ctx), Some(CssValue::Number(5.0)));
    }

    #[test]
    fn calc_clamp_boundaries() {
        let ctx = ctx();
        let px = |n: f32| CssValue::Length(n, LengthUnit::Px);
        // In range: the middle value wins.
        assert_eq!(math("clamp(1px, 5px, 10px)", &ctx), Some(px(5.0)));
        // Below the minimum pins to the minimum.
        assert_eq!(math("clamp(1px, -5px, 10px)", &ctx), Some(px(1.0)));
        // Above the maximum pins to the maximum.
        assert_eq!(math("clamp(1px, 100px, 10px)", &ctx), Some(px(10.0)));
        // Unitless clamp.
        assert_eq!(math("clamp(0, 0.7, 1)", &ctx), Some(CssValue::Number(0.7)));
        // Nested inside calc, mixed units.
        assert_eq!(
            math("calc(clamp(1px, 5em, 10px) * 2)", &ctx),
            Some(px(20.0))
        );
    }

    #[test]
    fn calc_invalid() {
        let ctx = ctx();
        // Parse failures.
        assert!(v("calc()").is_err());
        assert!(v("calc(1px +)").is_err());
        assert!(v("calc(1px 2px)").is_err());
        assert!(v("calc((1px + 2px)").is_err());
        assert!(v("min()").is_err());
        assert!(v("min(1px 2px)").is_err());
        assert!(v("clamp(1px, 2px)").is_err());
        assert!(v("clamp(1px, 2px, 3px, 4px)").is_err());
        assert!(v("calc(10xyz + 2px)").is_err());
        assert!(v("calc(rgb(1, 2, 3))").is_err());
        // Evaluation failures: type mismatches and division by zero.
        assert_eq!(math("calc(1px + 2)", &ctx), None);
        assert_eq!(math("calc(1px - 2)", &ctx), None);
        assert_eq!(math("calc(1px * 2px)", &ctx), None);
        assert_eq!(math("calc(2px / 1px)", &ctx), None);
        assert_eq!(math("calc(1px / 0)", &ctx), None);
        assert_eq!(math("calc(2 / 0)", &ctx), None);
        assert_eq!(math("min(1px, 2)", &ctx), None);
        assert_eq!(math("clamp(1px, 2px, 3)", &ctx), None);
        // An unsubstituted `var()` leaf fails closed at evaluation time
        // (the cascade substitutes these before evaluating).
        assert_eq!(math("calc(var(--x) + 1px)", &ctx), None);
        assert_eq!(math("calc(var(--x, 10px) + var(--y))", &ctx), None);
    }

    #[test]
    fn var_inside_math_parses() {
        // `var()` is a valid math leaf now; the cascade substitutes it.
        assert!(v("calc(var(--x) + 1px)").is_ok());
        assert!(v("calc(var(--x, 10px) + 5px)").is_ok());
        assert!(v("min(var(--a), var(--b, 2px))").is_ok());
        assert!(v("calc(var(--x) * var(--y, 2))").is_ok());
        // Malformed `var()` is still rejected.
        assert!(v("calc(var(x) + 1px)").is_err());
        assert!(v("calc(var(--x, ) + 1px)").is_err());
        assert!(v("calc(var(--x, red) + 1px)").is_err());
        // Fallbacks can nest `var()`.
        assert!(v("calc(var(--x, var(--y, 3px)) + 1px)").is_ok());
    }

    #[test]
    fn math_nesting_depth_capped() {
        // Balanced but absurdly deep nesting fails instead of overflowing
        // the stack (the `calc(` + 100k parens attack).
        let deep_parens = format!("calc({}1px{})", "(".repeat(10_000), ")".repeat(10_000));
        assert!(v(&deep_parens).is_err());
        let deep_calc = format!("{}1px{}", "calc(".repeat(10_000), ")".repeat(10_000));
        assert!(v(&deep_calc).is_err());
        // Nested `var()` fallbacks recurse through the value parser too.
        let mut deep_var = "1px".to_string();
        for i in 0..10_000 {
            deep_var = format!("var(--a{i}, {deep_var})");
        }
        assert!(v(&deep_var).is_err());
        // Shallow nesting still parses and evaluates.
        let shallow = format!("{}1px{}", "calc(".repeat(10), ")".repeat(10));
        let ctx = ctx();
        assert_eq!(
            v(&shallow).unwrap().evaluate_math(&ctx),
            Some(CssValue::Length(1.0, LengthUnit::Px))
        );
    }

    #[test]
    fn math_abs_sign() {
        let ctx = ctx();
        let px = |n: f32| CssValue::Length(n, LengthUnit::Px);
        // abs() keeps its input's type (CSS Values 4 §10.6).
        assert_eq!(math("abs(-4)", &ctx), Some(CssValue::Number(4.0)));
        assert_eq!(math("abs(4)", &ctx), Some(CssValue::Number(4.0)));
        assert_eq!(math("abs(-3px)", &ctx), Some(px(3.0)));
        assert_eq!(math("abs(3px)", &ctx), Some(px(3.0)));
        assert_eq!(math("abs(3px - 8px)", &ctx), Some(px(5.0)));
        assert_eq!(math("abs(-0.0)", &ctx), Some(CssValue::Number(0.0)));
        // sign() always returns a <number> (CSS Values 4 §10.6).
        assert_eq!(math("sign(-4)", &ctx), Some(CssValue::Number(-1.0)));
        assert_eq!(math("sign(4)", &ctx), Some(CssValue::Number(1.0)));
        assert_eq!(math("sign(0)", &ctx), Some(CssValue::Number(0.0)));
        assert_eq!(math("sign(-3px)", &ctx), Some(CssValue::Number(-1.0)));
        assert_eq!(math("sign(2.5)", &ctx), Some(CssValue::Number(1.0)));
        // Nested inside calc().
        assert_eq!(math("calc(abs(-2px) * 3)", &ctx), Some(px(6.0)));
        // Arity errors.
        assert!(v("abs()").is_err());
        assert!(v("abs(1px, 2px)").is_err());
        assert!(v("sign()").is_err());
        assert!(v("sign(1, 2)").is_err());
    }

    #[test]
    fn math_round() {
        let ctx = ctx();
        let px = |n: f32| CssValue::Length(n, LengthUnit::Px);
        // Default strategy is nearest; B defaults to 1 for <number> A.
        assert_eq!(math("round(5.6, 1)", &ctx), Some(CssValue::Number(6.0)));
        assert_eq!(math("round(5.4, 1)", &ctx), Some(CssValue::Number(5.0)));
        assert_eq!(math("round(5.6)", &ctx), Some(CssValue::Number(6.0)));
        assert_eq!(math("round(5)", &ctx), Some(CssValue::Number(5.0)));
        // nearest: exact ties choose the multiple toward +∞ (spec §10.3).
        assert_eq!(
            math("round(nearest, 5, 2)", &ctx),
            Some(CssValue::Number(6.0))
        );
        assert_eq!(
            math("round(nearest, 5.5, 2)", &ctx),
            Some(CssValue::Number(6.0))
        );
        assert_eq!(
            math("round(nearest, -5, 2)", &ctx),
            Some(CssValue::Number(-4.0))
        );
        // The other strategies.
        assert_eq!(math("round(up, 5.1, 2)", &ctx), Some(CssValue::Number(6.0)));
        assert_eq!(
            math("round(up, -5.1, 2)", &ctx),
            Some(CssValue::Number(-4.0))
        );
        assert_eq!(
            math("round(down, 5.9, 2)", &ctx),
            Some(CssValue::Number(4.0))
        );
        assert_eq!(
            math("round(down, -5.1, 2)", &ctx),
            Some(CssValue::Number(-6.0))
        );
        assert_eq!(
            math("round(to-zero, 5.9, 2)", &ctx),
            Some(CssValue::Number(4.0))
        );
        assert_eq!(
            math("round(to-zero, -5.9, 2)", &ctx),
            Some(CssValue::Number(-4.0))
        );
        // Lengths keep their type; strategy keywords are case-insensitive.
        assert_eq!(math("round(7px, 4px)", &ctx), Some(px(8.0)));
        assert_eq!(math("round(down, 7px, 4px)", &ctx), Some(px(4.0)));
        assert_eq!(
            math("round(NEAREST, 5.5, 2)", &ctx),
            Some(CssValue::Number(6.0))
        );
        // line-width parses (valid grammar) but cannot evaluate without
        // a device pixel: fails closed.
        assert!(v("round(line-width, 10px)").is_ok());
        assert_eq!(math("round(line-width, 10px)", &ctx), None);
        // Evaluation failures: B omitted for a length, zero B (NaN per
        // spec §10.3.1), mixed types.
        assert_eq!(math("round(5px)", &ctx), None);
        assert_eq!(math("round(5, 0)", &ctx), None);
        assert_eq!(math("round(5px, 2)", &ctx), None);
        // Parse failures: empty, too many args, unknown strategy-like
        // first arg is not a valid <calc-sum>.
        assert!(v("round()").is_err());
        assert!(v("round(1, 2, 3, 4)").is_err());
        assert!(v("round(5px, 2px, 3px)").is_err());
        assert!(v("round(bogus, 5, 2)").is_err());
    }

    #[test]
    fn math_mod_rem() {
        let ctx = ctx();
        let px = |n: f32| CssValue::Length(n, LengthUnit::Px);
        // mod() takes the sign of B; rem() the sign of A (spec §10.3).
        assert_eq!(math("mod(18, 5)", &ctx), Some(CssValue::Number(3.0)));
        assert_eq!(math("mod(-18, 5)", &ctx), Some(CssValue::Number(2.0)));
        assert_eq!(math("rem(-18, 5)", &ctx), Some(CssValue::Number(-3.0)));
        assert_eq!(math("rem(18, 5)", &ctx), Some(CssValue::Number(3.0)));
        assert_eq!(math("mod(140, -90)", &ctx), Some(CssValue::Number(-40.0)));
        assert_eq!(math("rem(140, -90)", &ctx), Some(CssValue::Number(50.0)));
        // Lengths keep their type.
        assert_eq!(math("mod(18px, 5px)", &ctx), Some(px(3.0)));
        assert_eq!(math("rem(-18px, 5px)", &ctx), Some(px(-3.0)));
        // Zero B is NaN (spec §10.3.1); mixed types are invalid.
        assert_eq!(math("mod(5, 0)", &ctx), None);
        assert_eq!(math("rem(5, 0)", &ctx), None);
        assert_eq!(math("mod(5px, 2)", &ctx), None);
        assert_eq!(math("rem(5, 2px)", &ctx), None);
        // Arity errors.
        assert!(v("mod(1)").is_err());
        assert!(v("mod(1, 2, 3)").is_err());
        assert!(v("rem()").is_err());
    }

    #[test]
    fn math_hypot() {
        let ctx = ctx();
        let px = |n: f32| CssValue::Length(n, LengthUnit::Px);
        assert_eq!(math("hypot(3, 4)", &ctx), Some(CssValue::Number(5.0)));
        assert_eq!(math("hypot(3px, 4px)", &ctx), Some(px(5.0)));
        assert_eq!(math("hypot(2)", &ctx), Some(CssValue::Number(2.0)));
        assert_eq!(math("hypot(-3px, 4px)", &ctx), Some(px(5.0)));
        assert_eq!(math("hypot(1, 2, 2)", &ctx), Some(CssValue::Number(3.0)));
        // Mixed types are invalid; empty is a parse error.
        assert_eq!(math("hypot(1px, 2)", &ctx), None);
        assert!(v("hypot()").is_err());
    }

    #[test]
    fn math_sqrt_pow_exp_log() {
        let ctx = ctx();
        // sqrt() takes a <number> only (spec §10.5).
        assert_eq!(math("sqrt(16)", &ctx), Some(CssValue::Number(4.0)));
        assert_eq!(math("sqrt(2)", &ctx), Some(CssValue::Number(2f32.sqrt())));
        assert_eq!(math("sqrt(-1)", &ctx), None);
        assert_eq!(math("sqrt(4px)", &ctx), None);
        // pow() needs two <number>s (spec §10.5).
        assert_eq!(math("pow(2, 3)", &ctx), Some(CssValue::Number(8.0)));
        assert_eq!(math("pow(9, 0.5)", &ctx), Some(CssValue::Number(3.0)));
        assert_eq!(math("pow(-8, 2)", &ctx), Some(CssValue::Number(64.0)));
        assert_eq!(math("pow(-8, 0.5)", &ctx), None);
        assert_eq!(math("pow(2px, 3)", &ctx), None);
        assert_eq!(math("pow(2, 3px)", &ctx), None);
        // exp() takes a <number>.
        assert_eq!(math("exp(0)", &ctx), Some(CssValue::Number(1.0)));
        assert_eq!(
            math("exp(1)", &ctx),
            Some(CssValue::Number(std::f32::consts::E))
        );
        assert_eq!(math("exp(1px)", &ctx), None);
        // log(A, B?): base defaults to e; base 1 or ≤ 0, negative or
        // zero A are NaN per spec §10.5.1.
        assert_eq!(math("log(8, 2)", &ctx), Some(CssValue::Number(3.0)));
        assert_eq!(math("log(100)", &ctx), Some(CssValue::Number(100f32.ln())));
        assert_eq!(math("log(10, 10)", &ctx), Some(CssValue::Number(1.0)));
        assert_eq!(math("log(8, 1)", &ctx), None);
        assert_eq!(math("log(8, -2)", &ctx), None);
        assert_eq!(math("log(8, 0)", &ctx), None);
        assert_eq!(math("log(-1)", &ctx), None);
        assert_eq!(math("log(0)", &ctx), None);
        assert_eq!(math("log(1px)", &ctx), None);
        // Arity errors.
        assert!(v("sqrt()").is_err());
        assert!(v("sqrt(1, 2)").is_err());
        assert!(v("pow(2)").is_err());
        assert!(v("pow(1, 2, 3)").is_err());
        assert!(v("exp()").is_err());
        assert!(v("log()").is_err());
        assert!(v("log(1, 2, 3)").is_err());
    }

    #[test]
    fn math_trig() {
        let ctx = ctx();
        let approx = |input: &str| match math(input, &ctx) {
            Some(CssValue::Number(n)) => n,
            other => panic!("expected Number for {input:?}, got {other:?}"),
        };
        let close = |input: &str, expected: f32| {
            assert!(
                (approx(input) - expected).abs() < 1e-5,
                "{input} should be ~{expected}"
            );
        };
        close("sin(0)", 0.0);
        close("cos(0)", 1.0);
        close("tan(0)", 0.0);
        close("sin(1.5707963)", 1.0);
        close("cos(3.1415927)", -1.0);
        close("asin(1)", std::f32::consts::FRAC_PI_2);
        close("acos(0)", std::f32::consts::FRAC_PI_2);
        close("atan(1)", std::f32::consts::FRAC_PI_4);
        close("atan2(1, 1)", std::f32::consts::FRAC_PI_4);
        close("atan2(0, -1)", std::f32::consts::PI);
        // atan2() allows consistent dimensions (spec §10.4); the result
        // is radians as a <number> (no <angle> type here).
        close("atan2(1px, 1px)", std::f32::consts::FRAC_PI_4);
        // Nested inside calc().
        close("calc(sin(0) + cos(0))", 1.0);
        // Type errors: a <length> argument parses but fails at
        // evaluation. |A| > 1 is NaN for asin()/acos(); mixed atan2
        // types are invalid. Angle arguments are covered by
        // math_trig_angle_units below.
        assert_eq!(math("cos(1px)", &ctx), None);
        assert_eq!(math("asin(2)", &ctx), None);
        assert_eq!(math("acos(-2)", &ctx), None);
        assert_eq!(math("atan2(1px, 2)", &ctx), None);
        // Arity errors.
        assert!(v("sin()").is_err());
        assert!(v("sin(1, 2)").is_err());
        assert!(v("atan2(1)").is_err());
        assert!(v("atan2(1, 2, 3)").is_err());
    }

    #[test]
    fn math_angle_keeps_type_through_evaluation() {
        let ctx = ctx();
        // Evaluate the `calc()` tree directly: `evaluate_math` flattens
        // angles to radian numbers (CssValue has no angle form), which
        // would hide the very distinction under test.
        let ev = |input: &str| match ok(input) {
            CssValue::Math(e) => e.evaluate(&ctx),
            other => panic!("expected Math for {input:?}, got {other:?}"),
        };
        let angle = |input: &str| match ev(input) {
            Some(MathValue::Angle(r)) => r,
            other => panic!("expected Angle for {input:?}, got {other:?}"),
        };
        let close = |input: &str, expected: f32| {
            let r = angle(input);
            assert!(
                (r - expected).abs() < 1e-5,
                "{input} should be ~{expected}, got {r}"
            );
        };
        // An angle leaf evaluates to an Angle, never a Number: the
        // <angle>/<number> distinction survives evaluation.
        assert!(matches!(ev("calc(90deg)"), Some(MathValue::Angle(_))));
        assert!(matches!(ev("calc(1)"), Some(MathValue::Number(_))));
        // Angle arithmetic typing (CSS Values 4 §10.2): Angle±Angle→
        // Angle, Angle×/÷Number→Angle.
        close("calc(90deg + 90deg)", std::f32::consts::PI);
        close("calc(90deg - 30deg)", std::f32::consts::FRAC_PI_3);
        close("calc(90deg * 2)", std::f32::consts::PI);
        close("calc(2 * 90deg)", std::f32::consts::PI);
        close("calc(90deg / 2)", std::f32::consts::FRAC_PI_4);
        close("calc(-90deg)", -std::f32::consts::FRAC_PI_2);
        close("calc(abs(-90deg))", std::f32::consts::FRAC_PI_2);
        close(
            "calc(clamp(0deg, 90deg, 180deg))",
            std::f32::consts::FRAC_PI_2,
        );
        close("calc(min(90deg, 180deg))", std::f32::consts::FRAC_PI_2);
        close("calc(max(90deg, 180deg))", std::f32::consts::PI);
        // sign() of an angle is still a <number> (spec §10.6).
        assert_eq!(ev("calc(sign(90deg))"), Some(MathValue::Number(1.0)));
        // Mixed-type arithmetic is invalid math: Angle+Number,
        // Angle×Length, Length÷Angle, Angle×Angle, Angle÷Angle.
        assert_eq!(ev("calc(90deg + 1)"), None);
        assert_eq!(ev("calc(1 + 90deg)"), None);
        assert_eq!(ev("calc(90deg * 1px)"), None);
        assert_eq!(ev("calc(1px / 90deg)"), None);
        assert_eq!(ev("calc(90deg * 90deg)"), None);
        assert_eq!(ev("calc(90deg / 90deg)"), None);
        assert_eq!(ev("calc(min(90deg, 1))"), None);
        assert_eq!(ev("calc(max(90deg, 10px))"), None);
        // atan2(): arguments must share a type; the result is an <angle>.
        close("atan2(90deg, 90deg)", std::f32::consts::FRAC_PI_4);
        close("atan2(1px, 1px)", std::f32::consts::FRAC_PI_4);
        close("atan2(1, 2)", 1f32.atan2(2.0));
        assert_eq!(ev("atan2(90deg, 1)"), None);
        assert_eq!(ev("atan2(1, 90deg)"), None);
        assert_eq!(ev("atan2(90deg, 1px)"), None);
        assert_eq!(ev("atan2(1px, 90deg)"), None);
        // An angle result composes with angle arithmetic.
        close("calc(atan2(1, 1) + 45deg)", std::f32::consts::FRAC_PI_2);
        // ... but not with numbers.
        assert_eq!(ev("calc(atan2(1, 1) + 1)"), None);
    }

    #[test]
    fn math_trig_angle_units() {
        let ctx = ctx();
        let approx = |input: &str| match math(input, &ctx) {
            Some(CssValue::Number(n)) => n,
            other => panic!("expected Number for {input:?}, got {other:?}"),
        };
        let close = |input: &str, expected: f32| {
            assert!(
                (approx(input) - expected).abs() < 1e-5,
                "{input} should be ~{expected}"
            );
        };
        // Every angle unit on sin()/cos()/tan() (CSS Values 4 §10.4):
        // 90deg = 100grad = π/2rad = 0.25turn.
        close("sin(90deg)", 1.0);
        close("sin(100grad)", 1.0);
        close("sin(1.5707963rad)", 1.0);
        close("sin(0.25turn)", 1.0);
        close("sin(30deg)", 0.5);
        close("sin(180deg)", 0.0);
        close("sin(0.5turn)", 0.0);
        close("sin(200grad)", 0.0);
        close("sin(-90deg)", -1.0);
        close("cos(0deg)", 1.0);
        close("cos(180deg)", -1.0);
        close("cos(0.5turn)", -1.0);
        close("cos(100grad)", 0.0);
        close("cos(360deg)", 1.0);
        close("tan(45deg)", 1.0);
        close("tan(50grad)", 1.0);
        close("tan(0.125turn)", 1.0);
        close("tan(0.7853982rad)", 1.0);
        // Units are case-insensitive, like every other CSS unit.
        close("sin(90DEG)", 1.0);
        close("cos(0.5TURN)", -1.0);
        // Angles compose with the rest of the calc grammar: the fold to
        // radians happens at parse time.
        close("sin(45deg + 45deg)", 1.0);
        close("cos(2 * 90deg)", -1.0);
        close("sin(calc(90deg))", 1.0);
        close("sin(min(90deg, 180deg))", 1.0);
        // atan2() with angle arguments: both angles fold to radian
        // <number>s, so the consistent-type rule still holds; a length
        // mixed with an angle is invalid.
        close("atan2(90deg, 90deg)", std::f32::consts::FRAC_PI_4);
        close("atan2(180deg, 360deg)", 0.5f32.atan2(1.0));
        close("atan2(1px, 1px)", std::f32::consts::FRAC_PI_4);
        assert_eq!(math("atan2(90deg, 1px)", &ctx), None);
        assert_eq!(math("atan2(1px, 90deg)", &ctx), None);
        // A <length> is still not an angle: it parses but fails at
        // evaluation.
        assert!(v("sin(10px)").is_ok());
        assert_eq!(math("sin(10px)", &ctx), None);
        assert_eq!(math("tan(1em)", &ctx), None);
        // Unknown units still fail parsing.
        assert!(v("sin(90dge)").is_err());
        assert!(v("cos(1foo)").is_err());
    }

    #[test]
    fn contains_angle_is_type_based_not_syntactic() {
        // The CSSOM setter path for inset properties (e.g.
        // `el.style.top = "calc(20deg)"`) rejects via
        // `CssValue::contains_angle()`. The check is type-based — "can
        // this evaluate to an <angle>" — not "does an angle leaf appear
        // anywhere".
        let angle_typed = |input: &str| {
            v(input)
                .unwrap_or_else(|_| panic!("{input} should parse"))
                .contains_angle()
        };
        // sin()/cos()/tan()/sign() consume an angle and always return a
        // <number>: a consumed angle can never leak out as angle-typed,
        // so these are accepted for inset properties (the stylesheet
        // path evaluates `calc(sin(90deg) * 10px)` to 10px).
        assert!(!angle_typed("calc(sin(90deg) * 10px)"));
        assert!(!angle_typed("calc(10px * sin(90deg))"));
        assert!(!angle_typed("calc(cos(90deg) * 10px)"));
        assert!(!angle_typed("calc(tan(45deg) * 10px)"));
        assert!(!angle_typed("calc(sign(90deg) * 10px)"));
        assert!(!angle_typed("calc(sign(10px) * 10px)"));
        // var() is never rejected at set time: it is parse-valid in any
        // property and type-checks at computed-value time (CSS Custom
        // Properties L1). The angle fallback is reachable inside calc().
        assert!(!angle_typed("calc(var(--u, 20deg))"));
        assert!(!angle_typed("calc(var(--u, 20deg) + 10px)"));
        assert!(!angle_typed("calc(var(--u) + 10px)"));
        // Angle-typed expressions are still rejected.
        assert!(angle_typed("calc(20deg)"));
        assert!(angle_typed("calc(20deg + 10deg)"));
        assert!(angle_typed("calc(20deg * 2)"));
        assert!(angle_typed("calc(-20deg)"));
        assert!(angle_typed("calc(abs(20deg))"));
        assert!(angle_typed("calc(min(20deg, 10deg))"));
        assert!(angle_typed("calc(clamp(10deg, 20deg, 30deg))"));
        // asin()/acos()/atan()/atan2() always return <angle>.
        assert!(angle_typed("calc(asin(0.5))"));
        assert!(angle_typed("calc(acos(0.5))"));
        assert!(angle_typed("calc(atan(0.5))"));
        assert!(angle_typed("calc(atan2(1, 2))"));
        // sqrt()/exp() recurse conservatively; neg()/abs() are
        // type-preserving.
        assert!(angle_typed("calc(sqrt(20deg))"));
        assert!(angle_typed("calc(exp(20deg))"));
        assert!(!angle_typed("calc(sqrt(4) * 10px)"));
        assert!(!angle_typed("calc(-10px)"));
        assert!(!angle_typed("calc(abs(-10px))"));
        // Plain number/length math was never rejected and still isn't.
        assert!(!angle_typed("calc(10px + 5%)"));
        assert!(!angle_typed("calc(20 * 2)"));
        assert!(!angle_typed("10px"));
    }

    /// The CSSOM setter path (`el.style.top = ...`) parses the
    /// declaration and drops it when the grammar validator or the
    /// angle check rejects it (mirrors `parse_valid_style_decls` in
    /// bowser_js' bindings.rs).
    fn cssom_setter_accepts(prop: &str, value_src: &str) -> bool {
        crate::parse_declarations(&format!("{prop}: {value_src}"))
            .iter()
            .any(|d| {
                crate::is_valid_for_property(d.property.as_str(), &d.value)
                    && !d.value.contains_angle()
            })
    }

    #[test]
    fn cssom_setter_path_angle_math_acceptance() {
        // Angle-consuming math is accepted: sin() returns <number>, so
        // the stylesheet path evaluates this to 10px.
        assert!(cssom_setter_accepts("top", "calc(sin(90deg) * 10px)"));
        // var() with an angle fallback is not rejected at set time.
        assert!(cssom_setter_accepts("top", "calc(var(--u, 20deg))"));
        assert!(cssom_setter_accepts("top", "calc(var(--u, 20deg) + 10px)"));
        // Angle-typed math is still rejected for inset properties.
        assert!(!cssom_setter_accepts("top", "calc(20deg)"));
        assert!(!cssom_setter_accepts("top", "calc(asin(0.5))"));
        assert!(!cssom_setter_accepts("top", "calc(atan2(1, 2))"));
        // Sanity: plain lengths were and are accepted.
        assert!(cssom_setter_accepts("top", "10px"));
        assert!(cssom_setter_accepts("top", "calc(10px + 5%)"));
    }

    #[test]
    fn math_if() {
        let ctx = ctx();
        let px = |n: f32| CssValue::Length(n, LengthUnit::Px);
        // `else` always wins.
        assert_eq!(math("if(else: 10px)", &ctx), Some(px(10.0)));
        assert_eq!(math("if(ELSE: 10px)", &ctx), Some(px(10.0)));
        // supports()/media()/style() are unevaluable in a math context:
        // unknown → false, so the else branch wins (Values 5 §3.2).
        assert_eq!(
            math("if(media(width > 100px): 1px; else: 2px)", &ctx),
            Some(px(2.0))
        );
        assert_eq!(
            math("if(style(--x: 1): 1px; else: 2px)", &ctx),
            Some(px(2.0))
        );
        assert_eq!(
            math("if(supports(display: grid): 1px; else: 2px)", &ctx),
            Some(px(2.0))
        );
        // First true branch wins; later branches never run.
        assert_eq!(math("if(else: 1px; else: 2px)", &ctx), Some(px(1.0)));
        // Boolean combinations of unknown tests stay unknown → false.
        assert_eq!(
            math("if(not media(width > 100px): 1px; else: 2px)", &ctx),
            Some(px(2.0))
        );
        // Branch values are full <calc-sum>s; nesting inside calc() works.
        assert_eq!(math("if(else: 1px + 2px)", &ctx), Some(px(3.0)));
        assert_eq!(math("calc(if(else: 2px) * 3)", &ctx), Some(px(6.0)));
        // A trailing `;` is allowed by the grammar.
        assert_eq!(math("if(else: 1px;)", &ctx), Some(px(1.0)));
        // No true branch: the guaranteed-invalid value → invalid math.
        assert_eq!(math("if(media(width > 100px): 1px)", &ctx), None);
        // Parse failures: empty body, missing colon, empty condition,
        // empty value, non-math branch value, empty branch.
        assert!(v("if()").is_err());
        assert!(v("if(else)").is_err());
        assert!(v("if(else 1px)").is_err());
        assert!(v("if(: 1px)").is_err());
        assert!(v("if(else:)").is_err());
        assert!(v("if(else: red)").is_err());
        assert!(v("if(else: 1px;; else: 2px)").is_err());
    }

    #[test]
    fn math_toggle() {
        let ctx = ctx();
        let px = |n: f32| CssValue::Length(n, LengthUnit::Px);
        // No inherited value is available in `MathContext`, so the
        // spec's no-match case applies: the first argument wins.
        assert_eq!(math("toggle(1px, 2px, 3px)", &ctx), Some(px(1.0)));
        assert_eq!(math("toggle(5)", &ctx), Some(CssValue::Number(5.0)));
        assert_eq!(math("calc(toggle(2px, 4px) * 2)", &ctx), Some(px(4.0)));
        assert!(v("toggle()").is_err());
    }

    #[test]
    fn math_new_fns_nesting_depth_capped() {
        // Deeply nested new math functions fail instead of overflowing
        // the stack, honoring the same MAX_MATH_DEPTH budget.
        let deep_abs = format!("{}-3px{}", "abs(".repeat(10_000), ")".repeat(10_000));
        assert!(v(&deep_abs).is_err());
        let deep_round = format!("{}5{}", "round(".repeat(10_000), ", 1)".repeat(10_000));
        assert!(v(&deep_round).is_err());
        let deep_if = format!("{}else: 1px{}", "if(".repeat(10_000), ")".repeat(10_000));
        assert!(v(&deep_if).is_err());
        // Nested inside calc() counts too.
        let deep_nested = format!("calc({}-3px{})", "abs(".repeat(10_000), ")".repeat(10_000));
        assert!(v(&deep_nested).is_err());
        // Shallow nesting still parses and evaluates.
        let ctx = ctx();
        assert_eq!(
            math("abs(abs(abs(-3px)))", &ctx),
            Some(CssValue::Length(3.0, LengthUnit::Px))
        );
        assert_eq!(
            math("round(round(5.6, 1), 1)", &ctx),
            Some(CssValue::Number(6.0))
        );
    }

    #[test]
    fn math_new_fns_serialize_round_trip() {
        // Serialized trees re-parse to equal trees.
        // `to_css()` serializes the tree as a `calc()` body, so re-wrap
        // it for the round trip.
        let roundtrip = |input: &str| {
            let e = math_expr(input);
            let back = format!("calc({})", e.to_css());
            assert_eq!(math_expr(&back), e, "round-trip of {input}");
        };
        roundtrip("abs(-3px)");
        roundtrip("sign(2px - 5px)");
        roundtrip("round(5.5, 2)");
        roundtrip("round(up, 5.5, 2)");
        roundtrip("round(5.5)");
        roundtrip("mod(18px, 5px)");
        roundtrip("rem(-18, 5)");
        roundtrip("hypot(3px, 4px)");
        roundtrip("sqrt(16)");
        roundtrip("pow(2, 3)");
        roundtrip("exp(1)");
        roundtrip("log(8, 2)");
        roundtrip("log(8)");
        roundtrip("sin(1)");
        roundtrip("atan2(1, 2)");
        roundtrip("toggle(1px, 2px)");
        roundtrip("if(else: 1px)");
        roundtrip("if(media(width > 100px): 1px; else: 2px)");
        // Multi-token conditions: `not` + function must not merge into
        // one ident on re-parse (`notmedia(...)`); the serializer
        // inserts the required separator (CSS Syntax 3 §5.4.1).
        roundtrip("if(not media(width > 100px): 1px; else: 2px)");
        roundtrip("if(not supports(display: grid): 1px; else: 2px)");
        roundtrip("calc(abs(-2px) + round(3.2, 1) * 1px)");
    }

    #[test]
    fn math_new_fns_var_and_percentage() {
        let ctx = MathContext::new(16.0, 16.0, 800.0, 600.0).with_percentage_base(200.0);
        let px = |n: f32| CssValue::Length(n, LengthUnit::Px);
        // Percentages resolve against the context's base inside the new
        // functions too.
        assert_eq!(math("abs(-50%)", &ctx), Some(px(100.0)));
        assert_eq!(math("round(55%, 20%)", &ctx), Some(px(120.0)));
        assert_eq!(math("hypot(30%, 40%)", &ctx), Some(px(100.0)));
        // Percentage containment is syntactic: the new variants recurse.
        assert!(math_expr("abs(50%)").contains_percentage());
        assert!(math_expr("round(1%, 2%)").contains_percentage());
        assert!(math_expr("if(else: 50%)").contains_percentage());
        assert!(!math_expr("abs(50px)").contains_percentage());
        // var() leaves parse inside the new functions and are found by
        // the cascade's substitution gate; unsubstituted they fail
        // closed at evaluation time.
        assert!(v("abs(var(--x, -3px))").is_ok());
        assert!(v("round(var(--a), var(--b, 2))").is_ok());
        assert!(v("if(else: var(--x, 1px))").is_ok());
        assert!(v("toggle(var(--x, 1px), 2px)").is_ok());
        assert!(math_expr("abs(var(--x, -3px))").contains_var());
        assert!(math_expr("hypot(var(--x), 4px)").contains_var());
        assert!(!math_expr("abs(-3px)").contains_var());
        assert_eq!(math("abs(var(--x, -3px))", &ctx), None);
        // Non-affine percentage math still evaluates (base-0 path in the
        // cascade relies on `evaluate`, which the new arms extend).
        assert_eq!(math("sign(50% - 150%)", &ctx), Some(CssValue::Number(-1.0)));
    }

    #[test]
    fn calc_size_parses_keyword_bases() {
        let calc_size = |input: &str| match ok(input) {
            CssValue::CalcSize { basis, calc } => (*basis, *calc),
            other => panic!("expected CalcSize, got {other:?}"),
        };
        // The WPT fixed-layout-2 shapes.
        let (basis, calc) = calc_size("calc-size(auto, size)");
        assert_eq!(basis, CalcSizeBasis::Auto);
        assert_eq!(calc, MathExpr::Size);
        let (basis, _) = calc_size("calc-size(max-content, size)");
        assert_eq!(basis, CalcSizeBasis::MaxContent);
        let (basis, _) = calc_size("calc-size(min-content, size)");
        assert_eq!(basis, CalcSizeBasis::MinContent);
        let (basis, _) = calc_size("calc-size(fit-content, size)");
        assert_eq!(basis, CalcSizeBasis::FitContent);
        let (basis, _) = calc_size("calc-size(stretch, size)");
        assert_eq!(basis, CalcSizeBasis::Stretch);
        // `available` is the legacy alias of `stretch`; matching is
        // case-insensitive like other CSS keywords.
        let (basis, _) = calc_size("calc-size(available, size)");
        assert_eq!(basis, CalcSizeBasis::Stretch);
        let (basis, calc) = calc_size("CALC-SIZE(AUTO, SIZE)");
        assert_eq!(basis, CalcSizeBasis::Auto);
        assert_eq!(calc, MathExpr::Size);
    }

    #[test]
    fn calc_size_parses_definite_bases() {
        let basis_of = |input: &str| match ok(input) {
            CssValue::CalcSize { basis, .. } => *basis,
            other => panic!("expected CalcSize, got {other:?}"),
        };
        assert_eq!(
            basis_of("calc-size(100px, size * 2)"),
            CalcSizeBasis::Length(100.0, LengthUnit::Px)
        );
        assert_eq!(
            basis_of("calc-size(2em, size)"),
            CalcSizeBasis::Length(2.0, LengthUnit::Em)
        );
        assert_eq!(
            basis_of("calc-size(50%, size)"),
            CalcSizeBasis::Percentage(50.0)
        );
        // Unitless zero is a valid `<length>` basis.
        assert_eq!(
            basis_of("calc-size(0, size)"),
            CalcSizeBasis::Length(0.0, LengthUnit::Px)
        );
        // A calc() basis parses to math.
        assert!(matches!(
            basis_of("calc-size(calc(10px + 5%), size)"),
            CalcSizeBasis::Math(_)
        ));
        assert!(matches!(
            basis_of("calc-size(min(10px, 20%), size)"),
            CalcSizeBasis::Math(_)
        ));
    }

    #[test]
    fn calc_size_parses_calc_sum() {
        let calc_of = |input: &str| match ok(input) {
            CssValue::CalcSize { calc, .. } => *calc,
            other => panic!("expected CalcSize, got {other:?}"),
        };
        // `size` is a first-class operand with the usual precedence.
        assert_eq!(
            calc_of("calc-size(max-content, 10px + 5%)"),
            MathExpr::Add(
                Box::new(MathExpr::Length(10.0, LengthUnit::Px)),
                Box::new(MathExpr::Percentage(5.0)),
            )
        );
        assert_eq!(
            calc_of("calc-size(100px, size * 2)"),
            MathExpr::Mul(Box::new(MathExpr::Size), Box::new(MathExpr::Number(2.0)))
        );
        // `size` stays valid inside nested math functions and parens.
        assert_eq!(
            calc_of("calc-size(100px, min(size, 200px))"),
            MathExpr::Min(vec![
                MathExpr::Size,
                MathExpr::Length(200.0, LengthUnit::Px),
            ])
        );
        assert_eq!(
            calc_of("calc-size(100px, (size + 10px) * 2)"),
            MathExpr::Mul(
                Box::new(MathExpr::Add(
                    Box::new(MathExpr::Size),
                    Box::new(MathExpr::Length(10.0, LengthUnit::Px)),
                )),
                Box::new(MathExpr::Number(2.0)),
            )
        );
        // `size` is rejected in a bare `calc()`: it only exists inside
        // `calc-size()`.
        assert!(v("calc(size + 1px)").is_err());
        assert!(v("calc(1px + size)").is_err());
        // ...and in other math functions.
        assert!(v("min(size, 1px)").is_err());
    }

    #[test]
    fn calc_size_rejects_invalid() {
        // Wrong arity.
        assert!(v("calc-size(auto)").is_err());
        assert!(v("calc-size()").is_err());
        assert!(v("calc-size(auto, size, 10px)").is_err());
        assert!(v("calc-size(, size)").is_err());
        assert!(v("calc-size(auto,)").is_err());
        assert!(v("calc-size(auto, )").is_err());
        // Bad basis.
        assert!(v("calc-size(bogus, size)").is_err());
        assert!(v("calc-size(10, size)").is_err());
        assert!(v("calc-size(10deg, size)").is_err());
        assert!(v("calc-size(10px 20px, size)").is_err());
        // `var()` in basis position is preserved for the cascade:
        // substitution precedes grammar checking (CSS Values 4), so it
        // parses here and the cascade substitutes it before basis
        // validation. A non-`--` name or a non-function is still
        // invalid.
        assert!(v("calc-size(var(x), size)").is_err());
        assert!(v("calc-size(var(), size)").is_err());
        assert!(v("calc-size(var(--x, bogus bogus), size)").is_ok());
        // Nested `calc-size()` is rejected in both positions.
        assert!(v("calc-size(calc-size(10px, size), size)").is_err());
        assert!(v("calc-size(10px, calc-size(size, size))").is_err());
        // Keywords are not calc operands.
        assert!(v("calc-size(auto, auto)").is_err());
        assert!(v("calc-size(10px, size + auto)").is_err());
    }

    #[test]
    fn calc_size_parses_var_basis() {
        let basis_of = |input: &str| match ok(input) {
            CssValue::CalcSize { basis, .. } => *basis,
            other => panic!("expected CalcSize, got {other:?}"),
        };
        // `var()` in basis position is preserved verbatim for the
        // cascade (substitution precedes grammar checking).
        assert_eq!(
            basis_of("calc-size(var(--x), size)"),
            CalcSizeBasis::Var {
                name: "--x".to_owned(),
                fallback: None,
            }
        );
        // The fallback is raw tokens: it may be a sizing keyword,
        // which the math grammar cannot represent.
        match basis_of("calc-size(var(--x, max-content), size)") {
            CalcSizeBasis::Var { name, fallback } => {
                assert_eq!(name, "--x");
                let fb = fallback.expect("fallback should be kept");
                assert_eq!(tokens_to_css(&fb), "max-content");
            }
            other => panic!("expected Var basis, got {other:?}"),
        }
        match basis_of("calc-size(var(--x, 10px + 5px), size)") {
            CalcSizeBasis::Var { name, fallback } => {
                assert_eq!(name, "--x");
                let fb = fallback.expect("fallback should be kept");
                // Whitespace was already filtered upstream in
                // `parse_calc_size`; the stream still re-tokenizes
                // identically.
                assert_eq!(tokens_to_css(&fb), "10px+5px");
            }
            other => panic!("expected Var basis, got {other:?}"),
        }
        // An empty fallback behaves like no fallback.
        assert_eq!(
            basis_of("calc-size(var(--x,), size)"),
            CalcSizeBasis::Var {
                name: "--x".to_owned(),
                fallback: None,
            }
        );
        // Matching is case-insensitive like other CSS functions.
        assert!(matches!(
            basis_of("calc-size(VAR(--x), size)"),
            CalcSizeBasis::Var { .. }
        ));
        // A `var()` basis is not definite and has no keyword or math
        // form until the cascade substitutes it.
        let var_basis = basis_of("calc-size(var(--x), size)");
        assert!(!var_basis.is_definite());
        assert_eq!(var_basis.keyword(), None);
        assert_eq!(var_basis.as_math_expr(), None);
    }

    #[test]
    fn calc_size_serializes_round_trip() {
        // Parse -> serialize -> re-parse -> equal, for every basis
        // shape and calc-sum shape the serializer supports.
        let round_trip = |input: &str| {
            let first = ok(input);
            let css = first.serialize_calc_size().expect("should serialize");
            let second = ok(&css);
            assert_eq!(first, second, "round-trip failed for {input} -> {css}");
            css
        };
        assert_eq!(round_trip("calc-size(auto, size)"), "calc-size(auto, size)");
        assert_eq!(
            round_trip("calc-size(max-content, 10px + size)"),
            "calc-size(max-content, 10px + size)"
        );
        assert_eq!(
            round_trip("calc-size(min-content, size * 2)"),
            "calc-size(min-content, size * 2)"
        );
        assert_eq!(
            round_trip("calc-size(fit-content, size - 5px)"),
            "calc-size(fit-content, size - 5px)"
        );
        assert_eq!(
            round_trip("calc-size(stretch, size / 4)"),
            "calc-size(stretch, size / 4)"
        );
        assert_eq!(
            round_trip("calc-size(100px, size + 10px)"),
            "calc-size(100px, size + 10px)"
        );
        assert_eq!(round_trip("calc-size(2em, size)"), "calc-size(2em, size)");
        assert_eq!(
            round_trip("calc-size(50%, size + 10px)"),
            "calc-size(50%, size + 10px)"
        );
        // A calc() basis keeps its `calc()` wrapper.
        assert_eq!(
            round_trip("calc-size(calc(10px + 5%), size)"),
            "calc-size(calc(10px + 5%), size)"
        );
        // Nested math functions keep their form; precedence parens
        // survive where re-parsing needs them.
        assert_eq!(
            round_trip("calc-size(100px, min(size, 200px))"),
            "calc-size(100px, min(size, 200px))"
        );
        assert_eq!(
            round_trip("calc-size(100px, clamp(50px, size, 80px))"),
            "calc-size(100px, clamp(50px, size, 80px))"
        );
        assert_eq!(
            round_trip("calc-size(100px, size - (10px + 5px))"),
            "calc-size(100px, size - (10px + 5px))"
        );
        assert_eq!(
            round_trip("calc-size(100px, (size + 10px) * 2)"),
            "calc-size(100px, (size + 10px) * 2)"
        );
        // `var()` bases and `var()` leaves round-trip verbatim.
        assert_eq!(
            round_trip("calc-size(var(--x), size)"),
            "calc-size(var(--x), size)"
        );
        assert_eq!(
            round_trip("calc-size(var(--x, max-content), size + 1px)"),
            "calc-size(var(--x, max-content), size + 1px)"
        );
        assert_eq!(
            round_trip("calc-size(100px, size + var(--pad))"),
            "calc-size(100px, size + var(--pad))"
        );
        // Non-`calc-size()` values do not serialize.
        assert_eq!(ok("10px").serialize_calc_size(), None);
        assert_eq!(ok("calc(1px + 1px)").serialize_calc_size(), None);
        assert_eq!(ok("auto").serialize_calc_size(), None);
    }

    #[test]
    fn calc_size_evaluates_definite_bases() {
        let ctx = MathContext::new(16.0, 16.0, 800.0, 600.0).with_percentage_base(200.0);
        let px = |n: f32| CssValue::Length(n, LengthUnit::Px);
        let eval = |input: &str| ok(input).evaluate_calc_size(&ctx);
        assert_eq!(eval("calc-size(100px, size * 2)"), Some(px(200.0)));
        assert_eq!(eval("calc-size(100px, size / 4)"), Some(px(25.0)));
        assert_eq!(eval("calc-size(10px, size + 5px)"), Some(px(15.0)));
        assert_eq!(eval("calc-size(100px, size)"), Some(px(100.0)));
        assert_eq!(eval("calc-size(100px, SIZE)"), Some(px(100.0)));
        // The basis can itself be a calc.
        assert_eq!(
            eval("calc-size(calc(10px + 20px), size * 2)"),
            Some(px(60.0))
        );
        // Percentages resolve against the context's percentage base.
        assert_eq!(eval("calc-size(50%, size)"), Some(px(100.0)));
        assert_eq!(eval("calc-size(100px, size + 10%)"), Some(px(120.0)));
        // Relative units in the basis resolve against font metrics.
        assert_eq!(eval("calc-size(2em, size * 2)"), Some(px(64.0)));
        // `size` inside nested math functions substitutes too.
        assert_eq!(eval("calc-size(100px, min(size, 150px))"), Some(px(100.0)));
        assert_eq!(eval("calc-size(100px, max(size, 150px))"), Some(px(150.0)));
        assert_eq!(
            eval("calc-size(100px, clamp(50px, size, 80px))"),
            Some(px(80.0))
        );
        // Keyword bases cannot evaluate without layout: fail closed.
        let kw_none = |input: &str| assert_eq!(ok(input).evaluate_calc_size(&ctx), None);
        kw_none("calc-size(auto, size)");
        kw_none("calc-size(max-content, size * 2)");
        kw_none("calc-size(min-content, size)");
        kw_none("calc-size(fit-content, size)");
        kw_none("calc-size(stretch, size)");
        // A lone `size` leaf fails closed like an unsubstituted `var()`.
        assert_eq!(MathExpr::Size.evaluate(&ctx), None);
        // Invalid math still fails: type mismatch, division by zero.
        assert_eq!(eval("calc-size(100px, size + 2)"), None);
        assert_eq!(eval("calc-size(100px, size / 0)"), None);
    }

    #[test]
    fn calc_size_basis_accessors() {
        // `calc_size_basis()` exposes the parsed basis; other values
        // report `None`.
        assert_eq!(
            ok("calc-size(auto, size)").calc_size_basis(),
            Some(&CalcSizeBasis::Auto)
        );
        assert_eq!(
            ok("calc-size(max-content, size)").calc_size_basis(),
            Some(&CalcSizeBasis::MaxContent)
        );
        assert_eq!(
            ok("calc-size(10px, size)").calc_size_basis(),
            Some(&CalcSizeBasis::Length(10.0, LengthUnit::Px))
        );
        assert_eq!(ok("10px").calc_size_basis(), None);
        assert_eq!(ok("auto").calc_size_basis(), None);
        assert_eq!(ok("calc(1px + 1px)").calc_size_basis(), None);
        // Keyword/definiteness queries.
        assert!(!CalcSizeBasis::Auto.is_definite());
        assert!(!CalcSizeBasis::MaxContent.is_definite());
        assert!(!CalcSizeBasis::MinContent.is_definite());
        assert!(!CalcSizeBasis::FitContent.is_definite());
        assert!(!CalcSizeBasis::Stretch.is_definite());
        assert!(CalcSizeBasis::Length(1.0, LengthUnit::Px).is_definite());
        assert!(CalcSizeBasis::Percentage(50.0).is_definite());
        assert!(CalcSizeBasis::Math(MathExpr::Number(1.0)).is_definite());
        assert_eq!(CalcSizeBasis::Auto.keyword(), Some("auto"));
        assert_eq!(CalcSizeBasis::MaxContent.keyword(), Some("max-content"));
        assert_eq!(CalcSizeBasis::MinContent.keyword(), Some("min-content"));
        assert_eq!(CalcSizeBasis::FitContent.keyword(), Some("fit-content"));
        assert_eq!(CalcSizeBasis::Stretch.keyword(), Some("stretch"));
        assert_eq!(CalcSizeBasis::Length(1.0, LengthUnit::Px).keyword(), None);
        // `substitute_size` replaces every `size` leaf with the basis.
        let calc = MathExpr::Mul(
            Box::new(MathExpr::Add(
                Box::new(MathExpr::Size),
                Box::new(MathExpr::Size),
            )),
            Box::new(MathExpr::Number(2.0)),
        );
        let base = MathExpr::Length(10.0, LengthUnit::Px);
        let len = || MathExpr::Length(10.0, LengthUnit::Px);
        assert_eq!(
            calc.substitute_size(&base),
            Some(MathExpr::Mul(
                Box::new(MathExpr::Add(Box::new(len()), Box::new(len()))),
                Box::new(MathExpr::Number(2.0)),
            ))
        );
        // The basis expression form exists only for definite bases.
        assert_eq!(
            CalcSizeBasis::Length(5.0, LengthUnit::Px).as_math_expr(),
            Some(MathExpr::Length(5.0, LengthUnit::Px))
        );
        assert_eq!(
            CalcSizeBasis::Percentage(25.0).as_math_expr(),
            Some(MathExpr::Percentage(25.0))
        );
        assert_eq!(CalcSizeBasis::Auto.as_math_expr(), None);
        assert_eq!(CalcSizeBasis::MaxContent.as_math_expr(), None);
    }

    #[test]
    fn hsl_known_vectors() {
        let c = |r, g, b| CssValue::Color(Color { r, g, b, a: 255 });
        assert_eq!(ok("hsl(60, 100%, 50%)"), c(255, 255, 0)); // yellow
        assert_eq!(ok("hsl(180, 100%, 50%)"), c(0, 255, 255)); // cyan
        assert_eq!(ok("hsl(300, 100%, 50%)"), c(255, 0, 255)); // magenta
        assert_eq!(ok("hsl(0, 0%, 50%)"), c(128, 128, 128)); // gray
        assert_eq!(ok("hsl(0, 0%, 100%)"), c(255, 255, 255)); // white
        assert_eq!(ok("hsl(39, 100%, 50%)"), c(255, 166, 0)); // orange
    }

    #[test]
    fn hwb_vectors() {
        let c = |r, g, b| CssValue::Color(Color { r, g, b, a: 255 });
        // Pure hues match the primaries.
        assert_eq!(ok("hwb(0 0% 0%)"), c(255, 0, 0));
        assert_eq!(ok("hwb(120 0% 0%)"), c(0, 255, 0));
        assert_eq!(ok("hwb(240 0% 0%)"), c(0, 0, 255));
        // Full whiteness -> white, full blackness -> black.
        assert_eq!(ok("hwb(0 100% 0%)"), c(255, 255, 255));
        assert_eq!(ok("hwb(0 0% 100%)"), c(0, 0, 0));
        // Half white mixed into red; half black mixed into red.
        assert_eq!(ok("hwb(0 50% 0%)"), c(255, 128, 128));
        assert_eq!(ok("hwb(0 0% 50%)"), c(128, 0, 0));
        // w + b > 1 renormalizes: 60/60 -> 50/50 over red = gray.
        assert_eq!(ok("hwb(0 60% 60%)"), c(128, 128, 128));
        // Alpha and comma syntax.
        assert_eq!(
            ok("hwb(0 0% 0% / 50%)"),
            CssValue::Color(Color {
                r: 255,
                g: 0,
                b: 0,
                a: 128
            })
        );
        assert_eq!(ok("hwb(120, 0%, 0%)"), c(0, 255, 0));
        assert!(v("hwb(0 0%)").is_err());
        assert!(v("hwb(0, 0%, 0%, 1, 2)").is_err());
    }

    #[test]
    fn lab_vectors() {
        let c = |r, g, b| CssValue::Color(Color { r, g, b, a: 255 });
        assert_eq!(ok("lab(0 0 0)"), c(0, 0, 0));
        assert_eq!(ok("lab(100 0 0)"), c(255, 255, 255));
        assert_eq!(ok("lab(100% 0 0)"), c(255, 255, 255));
        // Achromatic input is achromatic output.
        let gray = match ok("lab(50 0 0)") {
            CssValue::Color(gray) => gray,
            other => panic!("expected color, got {other:?}"),
        };
        assert_eq!(gray.r, gray.g);
        assert_eq!(gray.g, gray.b);
        // D65 Lab of the sRGB primaries (standard reference values).
        assert_eq!(ok("lab(53.24 80.09 67.2)"), c(255, 0, 0));
        assert_eq!(ok("lab(32.3 79.19 -107.87)"), c(0, 0, 255));
        // Alpha and invalid arg counts.
        assert_eq!(
            ok("lab(100 0 0 / 50%)"),
            CssValue::Color(Color {
                r: 255,
                g: 255,
                b: 255,
                a: 128
            })
        );
        assert!(v("lab(50 0)").is_err());
        assert!(v("lab(50 0 0 0)").is_err());
    }

    #[test]
    fn lch_vectors() {
        let c = |r, g, b| CssValue::Color(Color { r, g, b, a: 255 });
        // Zero chroma matches achromatic lab.
        assert_eq!(ok("lch(50 0 0)"), ok("lab(50 0 0)"));
        assert_eq!(ok("lch(0 0 0)"), c(0, 0, 0));
        assert_eq!(ok("lch(100% 0 0)"), c(255, 255, 255));
        assert!(v("lch(50 10)").is_err());
    }

    // ---- B1: leading unary signs are depth-capped ----

    #[test]
    fn hundred_k_unary_minuses_rejected() {
        // 100k leading `-` signs: the parser must reject without
        // recursing (B1). MAX_MATH_DEPTH caps the unary chain.
        let input = format!("calc({}1px)", "-".repeat(100_000));
        let start = std::time::Instant::now();
        let result = v(&input);
        let elapsed = start.elapsed();
        assert!(result.is_err(), "100k unary minuses must be rejected");
        assert!(
            elapsed.as_secs() < 5,
            "rejection must be fast, took {elapsed:?}"
        );
    }

    #[test]
    fn reasonable_unary_chain_parses() {
        // A handful of unary signs still works.
        let ctx = ctx();
        assert_eq!(
            math("calc(- -5px)", &ctx),
            Some(CssValue::Length(5.0, LengthUnit::Px))
        );
        assert_eq!(
            math("calc(+ - +5px)", &ctx),
            Some(CssValue::Length(-5.0, LengthUnit::Px))
        );
    }

    // ---- B9: nested var() fallbacks are linear ----

    #[test]
    fn hundred_k_nested_var_fallback_rejected_fast() {
        // 100k nested `var(--a, ...)` fallbacks: the parser fails fast
        // on exceeding MAX_MATH_DEPTH (B9) instead of doing 64 O(n)
        // scans. Tokenization is timed separately: it is O(n) and
        // unavoidably slow in debug builds (~1.4s for 500k tokens).
        let mut input = String::from("var(--a");
        for _ in 0..100_000 {
            input.push_str(", var(--a");
        }
        input.push_str(", 1px");
        for _ in 0..100_001 {
            input.push(')');
        }
        let toks = tokenize(&input);
        let start = std::time::Instant::now();
        let result = parse_value(&toks);
        let elapsed = start.elapsed();
        // The var chain exceeds the depth limit; what matters is the
        // parser rejects it quickly without hanging.
        assert!(result.is_err());
        assert!(
            elapsed.as_secs() < 1,
            "100k nested fallbacks must parse in <1s, took {elapsed:?}"
        );
    }

    // ---- A2: affine percentage analysis ----

    #[test]
    fn affine_percentage_detection() {
        let affine = |input: &str| {
            let expr = match ok(input) {
                CssValue::Math(e) => e,
                other => panic!("expected Math, got {other:?}"),
            };
            expr.is_affine_in_percentage()
        };
        // Affine: linear in `%`.
        assert!(affine("calc(50% - 20px)"));
        assert!(affine("calc(50% + 20px)"));
        assert!(affine("calc(2 * 10% + 5px)"));
        assert!(affine("calc(10% / 2)"));
        // Not affine: `%` on both sides of `*` or in a divisor.
        assert!(!affine("calc(50% * 2%)"));
        assert!(!affine("calc(100px / 50%)"));
        // `50% * 10px` is structurally affine, but a CSS type error
        // (length times length): coefficient extraction fails and the
        // declaration is dropped.
        assert!(affine("calc(50% * 10px)"));
        let expr = match ok("calc(50% * 10px)") {
            CssValue::Math(e) => e,
            other => panic!("expected Math, got {other:?}"),
        };
        assert!(expr.affine_coefficients(&ctx()).is_none());
        // min/max/clamp are not affine (branches on the percentage).
        assert!(!affine("min(50%, 20px)"));
        assert!(!affine("max(50%, 20px)"));
        assert!(!affine("clamp(10px, 50%, 100px)"));
    }

    #[test]
    fn affine_coefficient_extraction() {
        // `calc(50% - 20px)` extracts to pct=50, px=-20: with a 1000px
        // base the value is 480px (A2).
        let expr = match ok("calc(50% - 20px)") {
            CssValue::Math(e) => e,
            other => panic!("expected Math, got {other:?}"),
        };
        let (pct, px) = expr.affine_coefficients(&ctx()).expect("must be affine");
        assert!((pct - 50.0).abs() < 1e-4, "pct={pct}");
        assert!((px + 20.0).abs() < 1e-4, "px={px}");
        assert!((pct * 1000.0 / 100.0 + px - 480.0).abs() < 1e-3);
    }

    // ---- Cyclic-percentage detection (CSS Values 4 §10.9) ----

    /// Extract the [`MathExpr`] from a parsed `calc()`/`min()`/`max()`/
    /// `clamp()` value for the cyclic-percentage tests below.
    fn math_expr(input: &str) -> MathExpr {
        match ok(input) {
            CssValue::Math(e) => e,
            other => panic!("expected Math for {input:?}, got {other:?}"),
        }
    }

    #[test]
    fn cyclic_percentage_detection_basic() {
        // A percentage leaf anywhere in the tree marks the expression as
        // cyclic-capable, including inside nested math functions.
        for input in [
            "calc(50%)",
            "calc(10px + 50%)",
            "calc(50% - 20px)",
            "calc(2 * 10%)",
            "calc(10% / 2)",
            "calc(-50%)",
            "calc(-(50% + 10px))",
            "calc((50%))",
            "min(50%, 20px)",
            "max(10px, 50%)",
            "clamp(10px, 50%, 100px)",
            "calc(min(10px, 20%) + 5px)",
            "calc(max(1px, 2%) * 3)",
        ] {
            assert!(
                math_expr(input).contains_percentage(),
                "{input} should detect a percentage"
            );
        }
        // No percentage leaf: not cyclic-capable.
        for input in [
            "calc(10px + 5px)",
            "calc(2 * 3)",
            "calc(10em + 5vw)",
            "min(10px, 20px)",
            "max(1, 5)",
            "clamp(1px, 5px, 10px)",
        ] {
            assert!(
                !math_expr(input).contains_percentage(),
                "{input} should not detect a percentage"
            );
        }
    }

    #[test]
    fn cyclic_percentage_zero_still_counts() {
        // CSS Values 4 §10.9 / css-sizing-3 §5.2.1: cyclic-ness is
        // syntactic. A percentage that folds to zero still marks the
        // expression — `calc(10px + 0%)` is cyclic, not a definite 10px —
        // so the parser must preserve the leaf instead of folding it away
        // at parse time.
        let expr = math_expr("calc(10px + 0%)");
        assert!(expr.contains_percentage());
        assert!(
            matches!(
                expr,
                MathExpr::Add(_, ref b) if matches!(**b, MathExpr::Percentage(p) if p == 0.0)
            ),
            "the zero percentage leaf must survive parsing, got {expr:?}"
        );
        // Same for min()/max()/clamp() with an all-zero percentage.
        assert!(math_expr("min(10px, 0%)").contains_percentage());
        assert!(math_expr("max(0%, 20px)").contains_percentage());
        assert!(math_expr("clamp(0%, 5px, 10px)").contains_percentage());
    }

    #[test]
    fn cyclic_percentage_var_fallback() {
        // A `var()` leaf is opaque at parse time: no syntactic percentage,
        // even when the fallback holds one. The cascade re-checks after
        // substitution (a substituted percentage still defers correctly),
        // so parse-time detection must not peek into the fallback —
        // otherwise `box-shadow: calc(var(--x, 50%))` would be wrongly
        // rejected at parse time when `--x` is a valid length.
        assert!(!math_expr("calc(var(--x, 50%) + 10px)").contains_percentage());
        assert!(!math_expr("calc(var(--x, 50%))").contains_percentage());
        assert!(!math_expr("calc(var(--x) + 10px)").contains_percentage());
        assert!(!math_expr("min(var(--x, 50%), 20px)").contains_percentage());
        // A percentage OUTSIDE the `var()` still counts.
        assert!(math_expr("calc(var(--x) + 50%)").contains_percentage());
    }

    #[test]
    fn cyclic_percentage_value_level() {
        // The `CssValue`-level helper mirrors the tree walk for parse-time
        // consumers.
        assert!(ok("calc(50% + 10px)").contains_percentage());
        assert!(ok("min(50%, 20px)").contains_percentage());
        assert!(ok("50%").contains_percentage());
        assert!(!ok("calc(10px + 5px)").contains_percentage());
        assert!(!ok("10px").contains_percentage());
        assert!(!ok("auto").contains_percentage());
        // `var()` is opaque at the value level too.
        assert!(!v("var(--x)").unwrap().contains_percentage());
        assert!(!v("var(--x, 50%)").unwrap().contains_percentage());
    }

    #[test]
    fn cyclic_percentage_value_level_minmax_repeat() {
        // `minmax()` pairs and unexpanded `repeat()` track listings
        // recurse into their tracks: a percentage anywhere in them counts.
        assert!(ok("minmax(50%, 1fr)").contains_percentage());
        assert!(ok("minmax(10px, calc(50% + 1px))").contains_percentage());
        assert!(!ok("minmax(10px, 1fr)").contains_percentage());
        assert!(ok("repeat(auto-fill, 50%)").contains_percentage());
        assert!(ok("repeat(auto-fit, minmax(50px, 50%))").contains_percentage());
        assert!(!ok("repeat(auto-fill, 100px)").contains_percentage());
    }

    #[test]
    fn cyclic_percentage_shadow_gate() {
        // Shadows have no percentage base: a percentage-bearing `calc()`
        // is not a valid shadow length (CSS Values 4 §10.9 — the
        // percentage can never resolve). The parse-time gate keys off the
        // same syntactic detection.
        assert!(!Shadow::is_shadow_length(&ok("calc(10% + 1px)")));
        assert!(!Shadow::is_shadow_length(&ok("calc(10px + 0%)")));
        assert!(!Shadow::is_shadow_length(&ok("min(1px, 50%)")));
        // Percentage-free math stays valid.
        assert!(Shadow::is_shadow_length(&ok("calc(1px + 2px)")));
        assert!(Shadow::is_shadow_length(&ok("calc(var(--x) + 1px)")));
    }

    #[test]
    fn repeat_expansion_is_capped() {
        // Boundary: 1000 total tracks is fine.
        match ok("repeat(1000, 1px)") {
            CssValue::List(items) => assert_eq!(items.len(), 1000),
            other => panic!("expected List, got {other:?}"),
        }
        // Nested repeats that stay within the cap expand.
        match ok("repeat(10, repeat(10, 1px))") {
            CssValue::List(items) => assert_eq!(items.len(), 100),
            other => panic!("expected List, got {other:?}"),
        }
        // A per-level count above the cap fails.
        assert!(v("repeat(1001, 1px)").is_err());
    }

    #[test]
    fn nested_repeat_product_is_capped() {
        // The cap applies to the expanded product, not per level:
        // without it this would `with_capacity(1e9)` (~32 GiB) and abort.
        assert!(v("repeat(1000, repeat(1000, 1px))").is_err());
        assert!(v("repeat(1000, repeat(1000, repeat(1000, 1px)))").is_err());
        assert!(v("repeat(500, repeat(3, 1px))").is_err());
        // Just under the cap still works: 333 * 3 = 999.
        match ok("repeat(333, repeat(3, 1px))") {
            CssValue::List(items) => assert_eq!(items.len(), 999),
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn repeat_auto_fill_fit_pass_through() {
        // The parser contract layout's auto-fill/fit expansion relies on:
        // a dedicated `Repeat` descriptor, unexpanded.
        match ok("repeat(auto-fill, 100px)") {
            CssValue::Repeat { mode, tracks } => {
                assert_eq!(mode, RepeatMode::Fill);
                assert_eq!(tracks, vec![CssValue::Length(100.0, LengthUnit::Px)]);
            }
            other => panic!("expected Repeat, got {other:?}"),
        }
        // Multi-track listing, minmax inside, and the mode keyword is
        // case-insensitive (folded like every keyword).
        match ok("repeat(AUTO-FIT, minmax(50px, 1fr) 10px)") {
            CssValue::Repeat { mode, tracks } => {
                assert_eq!(mode, RepeatMode::Fit);
                assert_eq!(tracks.len(), 2);
            }
            other => panic!("expected Repeat, got {other:?}"),
        }
        // Auto-repetition may not nest inside another repeat().
        assert!(v("repeat(auto-fill, repeat(auto-fit, 100px))").is_err());
        assert!(v("repeat(2, repeat(auto-fill, 100px))").is_err());
        // A non-keyword, non-integer first argument still fails.
        assert!(v("repeat(foo, 100px)").is_err());
        assert!(v("repeat(1.5, 100px)").is_err());
    }

    #[test]
    fn non_finite_fr_is_rejected() {
        // `1e999fr` parses to infinity; accepted it would poison grid
        // track sizing with NaN (`free * inf / inf`).
        assert!(v("1e999fr").is_err());
        assert_eq!(ok("2fr"), CssValue::Fr(2.0));
        assert_eq!(ok("0fr"), CssValue::Fr(0.0));
        // Negative fr stays invalid.
        assert!(v("-1fr").is_err());
    }

    #[test]
    fn minmax_parses_track_breadths() {
        assert_eq!(
            ok("minmax(100px, 1fr)"),
            CssValue::MinMax(
                Box::new(CssValue::Length(100.0, LengthUnit::Px)),
                Box::new(CssValue::Fr(1.0)),
            )
        );
        assert_eq!(
            ok("minmax(min-content, max-content)"),
            CssValue::MinMax(
                Box::new(CssValue::Keyword("min-content".into())),
                Box::new(CssValue::Keyword("max-content".into())),
            )
        );
        assert_eq!(
            ok("minmax(10%, auto)"),
            CssValue::MinMax(
                Box::new(CssValue::Percentage(10.0)),
                Box::new(CssValue::Keyword("auto".into())),
            )
        );
        // Math sides parse (the cascade evaluates them later).
        assert!(matches!(
            ok("minmax(10px, calc(50%))"),
            CssValue::MinMax(_, _)
        ));
        // A minimum larger than the maximum parses; layout clamps the
        // track to the minimum.
        assert!(matches!(ok("minmax(200px, 100px)"), CssValue::MinMax(_, _)));
    }

    #[test]
    fn minmax_rejects_invalid_forms() {
        // An `fr` minimum is invalid (grid §7.2.4).
        assert!(v("minmax(1fr, 2fr)").is_err());
        assert!(v("minmax(1fr, 100px)").is_err());
        // `minmax()` cannot nest.
        assert!(v("minmax(minmax(1px, 2px), 3px)").is_err());
        assert!(v("minmax(1px, minmax(2px, 3px))").is_err());
        // Exactly two track breadths: missing/extra sides and non-breadth
        // keywords fail the whole value.
        assert!(v("minmax(100px)").is_err());
        assert!(v("minmax(1px, 2px, 3px)").is_err());
        assert!(v("minmax(, 1fr)").is_err());
        assert!(v("minmax(red, 1px)").is_err());
        assert!(v("minmax(1px, red)").is_err());
        assert!(v("minmax(auto-fit, 1px)").is_err());
    }

    #[test]
    fn minmax_nests_inside_repeat() {
        // `repeat()` expansion keeps each `minmax()` as one track.
        match ok("repeat(2, minmax(50px, 1fr))") {
            CssValue::List(items) => {
                assert_eq!(items.len(), 2);
                for item in &items {
                    match item {
                        CssValue::MinMax(min, max) => {
                            assert_eq!(**min, CssValue::Length(50.0, LengthUnit::Px));
                            assert_eq!(**max, CssValue::Fr(1.0));
                        }
                        other => panic!("expected minmax(50px, 1fr), got {other:?}"),
                    }
                }
            }
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn transform_functions_parse() {
        use TransformFunction::*;
        // `none` stays a keyword (it never becomes a `Transform`).
        assert_eq!(ok("none"), CssValue::Keyword("none".into()));
        match ok("translate(10px, 20%)") {
            CssValue::Transform(fns) => assert_eq!(
                fns,
                vec![Translate {
                    x: CssValue::Length(10.0, LengthUnit::Px),
                    y: CssValue::Percentage(20.0),
                }]
            ),
            other => panic!("expected Transform, got {other:?}"),
        }
        // One-arg translate() zeroes the other axis.
        match ok("translate(5px)") {
            CssValue::Transform(fns) => assert_eq!(
                fns,
                vec![Translate {
                    x: CssValue::Length(5.0, LengthUnit::Px),
                    y: CssValue::Number(0.0),
                }]
            ),
            other => panic!("expected Transform, got {other:?}"),
        }
        match ok("translatex(1em)") {
            CssValue::Transform(fns) => assert_eq!(
                fns,
                vec![Translate {
                    x: CssValue::Length(1.0, LengthUnit::Em),
                    y: CssValue::Number(0.0),
                }]
            ),
            other => panic!("expected Transform, got {other:?}"),
        }
        match ok("translatey(50%)") {
            CssValue::Transform(fns) => assert_eq!(
                fns,
                vec![Translate {
                    x: CssValue::Number(0.0),
                    y: CssValue::Percentage(50.0),
                }]
            ),
            other => panic!("expected Transform, got {other:?}"),
        }
        // scale() takes numbers; one arg duplicates.
        match ok("scale(2)") {
            CssValue::Transform(fns) => assert_eq!(
                fns,
                vec![Scale {
                    x: CssValue::Number(2.0),
                    y: CssValue::Number(2.0),
                }]
            ),
            other => panic!("expected Transform, got {other:?}"),
        }
        match ok("scale(2, 0.5)") {
            CssValue::Transform(fns) => assert_eq!(
                fns,
                vec![Scale {
                    x: CssValue::Number(2.0),
                    y: CssValue::Number(0.5),
                }]
            ),
            other => panic!("expected Transform, got {other:?}"),
        }
        // rotate() normalizes angles to degrees.
        match ok("rotate(45deg)") {
            CssValue::Transform(fns) => assert_eq!(fns, vec![Rotate { angle_deg: 45.0 }]),
            other => panic!("expected Transform, got {other:?}"),
        }
        match ok("rotate(0.5turn)") {
            CssValue::Transform(fns) => assert_eq!(fns, vec![Rotate { angle_deg: 180.0 }]),
            other => panic!("expected Transform, got {other:?}"),
        }
        match ok("rotate(100grad)") {
            CssValue::Transform(fns) => assert_eq!(fns, vec![Rotate { angle_deg: 90.0 }]),
            other => panic!("expected Transform, got {other:?}"),
        }
        // A space-separated function list accumulates into a single
        // `Transform` with one entry per function.
        match ok("translate(10px) scale(2) rotate(45deg)") {
            CssValue::Transform(fns) => {
                assert_eq!(fns.len(), 3);
                assert!(matches!(fns[0], Translate { .. }));
                assert!(matches!(fns[1], Scale { .. }));
                assert!(matches!(fns[2], Rotate { .. }));
            }
            other => panic!("expected Transform, got {other:?}"),
        }
        // Two functions: one `Transform`, two entries.
        match ok("translate(10px) scale(2)") {
            CssValue::Transform(fns) => assert_eq!(fns.len(), 2),
            other => panic!("expected Transform, got {other:?}"),
        }
    }

    #[test]
    fn transform_opaque_functions_parse() {
        use TransformFunction::*;
        // Valid-but-unrenderable functions are retained opaque as
        // `Other` instead of collapsing the declaration to `none`.
        for (input, name, raw) in [
            ("matrix(1, 0, 0, 1, 0, 0)", "matrix", "1, 0, 0, 1, 0, 0"),
            (
                "matrix3d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)",
                "matrix3d",
                "1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1",
            ),
            ("skew(10deg)", "skew", "10deg"),
            ("perspective(500px)", "perspective", "500px"),
            ("translate3d(1px, 2px, 3px)", "translate3d", "1px, 2px, 3px"),
            ("translatez(5px)", "translatez", "5px"),
            ("scale3d(1, 2, 3)", "scale3d", "1, 2, 3"),
            ("scalez(2)", "scalez", "2"),
            ("rotate3d(1, 0, 0, 45deg)", "rotate3d", "1, 0, 0, 45deg"),
            ("rotatex(45deg)", "rotatex", "45deg"),
            ("rotatey(45deg)", "rotatey", "45deg"),
            ("rotatez(45deg)", "rotatez", "45deg"),
        ] {
            match ok(input) {
                CssValue::Transform(fns) => assert_eq!(
                    fns,
                    vec![Other {
                        name: name.to_string(),
                        raw_args: raw.to_string(),
                    }],
                    "{input}"
                ),
                other => panic!("expected Transform, got {other:?}"),
            }
        }
        // `skewx(10deg) skewy(5deg)` is one `Transform` with two opaque
        // entries.
        match ok("skewx(10deg) skewy(5deg)") {
            CssValue::Transform(fns) => assert_eq!(fns.len(), 2),
            other => panic!("expected Transform, got {other:?}"),
        }
        // Nested functions (e.g. `calc()`) in strict-subset functions
        // are retained opaque until math evaluation lands.
        match ok("translate(calc(10px + 5px))") {
            CssValue::Transform(fns) => assert_eq!(
                fns,
                vec![Other {
                    name: "translate".to_string(),
                    raw_args: "calc(10px + 5px)".to_string(),
                }]
            ),
            other => panic!("expected Transform, got {other:?}"),
        }
        match ok("rotate(calc(45deg))") {
            CssValue::Transform(fns) => {
                assert!(matches!(fns.as_slice(), [Other { .. }]));
            }
            other => panic!("expected Transform, got {other:?}"),
        }
        // Mixed structural and opaque functions in one list.
        match ok("translate(10px) skew(10deg)") {
            CssValue::Transform(fns) => {
                assert_eq!(fns.len(), 2);
                assert!(matches!(fns[0], Translate { .. }));
                assert!(matches!(fns[1], Other { .. }));
            }
            other => panic!("expected Transform, got {other:?}"),
        }
    }

    #[test]
    fn transform_functions_reject_garbage() {
        // Unknown functions are invalid (they never parsed before, and
        // still don't).
        assert!(v("foo()").is_err());
        assert!(v("translate3()").is_err());
        // Empty bodies are truly invalid syntax, even for otherwise
        // opaque functions.
        assert!(v("matrix()").is_err());
        assert!(v("skew()").is_err());
        assert!(v("perspective( )").is_err());
        // Wrong arity or bad argument types in the structural subset.
        assert!(v("translate()").is_err());
        assert!(v("translate(1px, 2px, 3px)").is_err());
        assert!(v("translate(1 2 3 4 5)").is_err());
        assert!(v("translate(10deg)").is_err());
        assert!(v("translate(red)").is_err());
        assert!(v("scale()").is_err());
        assert!(v("scale(2px)").is_err());
        assert!(v("scale(50%)").is_err());
        assert!(v("rotate()").is_err());
        assert!(v("rotate(10px)").is_err());
        assert!(v("rotate(45)").is_err());
        // Non-finite numbers are rejected.
        assert!(v("scale(1e999)").is_err());
        assert!(v("translate(1e999px)").is_err());
    }

    // --- Transform matrix evaluation ---

    fn tctx(w: f32, h: f32) -> TransformContext<'static> {
        TransformContext {
            reference_width: w,
            reference_height: h,
            resolve_length: &resolve_absolute_length,
        }
    }

    /// Evaluate the single function in `input` against a `w`×`h`
    /// reference box.
    fn eval_fn(input: &str, w: f32, h: f32) -> Option<TransformMatrix> {
        let fns = match ok(input) {
            CssValue::Transform(fns) => fns,
            other => panic!("expected Transform, got {other:?}"),
        };
        assert_eq!(fns.len(), 1, "expected one function in {input}");
        let ctx = tctx(w, h);
        fns[0].to_matrix(&ctx)
    }

    /// Evaluate the whole function list in `input`.
    fn eval_list(input: &str, w: f32, h: f32) -> Option<TransformMatrix> {
        let fns = match ok(input) {
            CssValue::Transform(fns) => fns,
            other => panic!("expected Transform, got {other:?}"),
        };
        let ctx = tctx(w, h);
        compose_transforms(&fns, &ctx)
    }

    fn assert_matrix_exact(m: &TransformMatrix, expected: [f32; 6]) {
        assert_eq!([m.a, m.b, m.c, m.d, m.e, m.f], expected, "{m:?}");
    }

    fn assert_matrix_approx(m: &TransformMatrix, expected: [f32; 6]) {
        let got = [m.a, m.b, m.c, m.d, m.e, m.f];
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-5,
                "component {i}: got {g}, expected {e} (full: {m:?})"
            );
        }
    }

    #[test]
    fn transform_matrix_mul_and_apply() {
        let id = TransformMatrix::identity();
        assert!(id.is_identity());
        let t = TransformMatrix::translate(10.0, 20.0);
        assert!(!t.is_identity());
        // Identity is the unit of multiplication.
        assert_eq!(id.mul(&t), t);
        assert_eq!(t.mul(&id), t);
        // mul applies rhs first: translate-then-scale vs scale-then-translate.
        let s = TransformMatrix::new(2.0, 0.0, 0.0, 3.0, 0.0, 0.0);
        assert_eq!(t.mul(&s).apply(1.0, 1.0), (12.0, 23.0));
        assert_eq!(s.mul(&t).apply(1.0, 1.0), (22.0, 63.0));
    }

    #[test]
    fn transform_to_matrix_translate() {
        assert_matrix_exact(
            &eval_fn("translate(10px, 20px)", 200.0, 100.0).unwrap(),
            [1.0, 0.0, 0.0, 1.0, 10.0, 20.0],
        );
        // Percentages resolve against the reference box (x/width, y/height).
        assert_matrix_exact(
            &eval_fn("translate(50%, 25%)", 200.0, 100.0).unwrap(),
            [1.0, 0.0, 0.0, 1.0, 100.0, 25.0],
        );
        assert_matrix_exact(
            &eval_fn("translatex(5px)", 200.0, 100.0).unwrap(),
            [1.0, 0.0, 0.0, 1.0, 5.0, 0.0],
        );
        assert_matrix_exact(
            &eval_fn("translatey(25%)", 200.0, 100.0).unwrap(),
            [1.0, 0.0, 0.0, 1.0, 0.0, 25.0],
        );
        // One-arg translate() zeroes the other axis.
        assert_matrix_exact(
            &eval_fn("translate(7px)", 200.0, 100.0).unwrap(),
            [1.0, 0.0, 0.0, 1.0, 7.0, 0.0],
        );
        // Absolute units convert; relative units fail with the default resolver.
        assert_matrix_exact(
            &eval_fn("translate(1in, 0px)", 200.0, 100.0).unwrap(),
            [1.0, 0.0, 0.0, 1.0, 96.0, 0.0],
        );
        assert!(eval_fn("translate(1em, 0px)", 200.0, 100.0).is_none());
    }

    #[test]
    fn transform_to_matrix_scale() {
        assert_matrix_exact(
            &eval_fn("scale(2)", 0.0, 0.0).unwrap(),
            [2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        );
        assert_matrix_exact(
            &eval_fn("scale(2, 0.5)", 0.0, 0.0).unwrap(),
            [2.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        );
        assert_matrix_exact(
            &eval_fn("scalex(3)", 0.0, 0.0).unwrap(),
            [3.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );
        assert_matrix_exact(
            &eval_fn("scaley(0.5)", 0.0, 0.0).unwrap(),
            [1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        );
        // Degenerate: scale(0) collapses every point to the origin.
        let m = eval_fn("scale(0)", 0.0, 0.0).unwrap();
        assert_matrix_exact(&m, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(m.apply(42.0, -7.0), (0.0, 0.0));
        // Negative scales mirror.
        assert_matrix_exact(
            &eval_fn("scale(-1, 1)", 0.0, 0.0).unwrap(),
            [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );
    }

    #[test]
    fn transform_to_matrix_rotate() {
        // rotate(90deg) = matrix(0, 1, -1, 0, 0, 0); trig needs epsilon.
        assert_matrix_approx(
            &eval_fn("rotate(90deg)", 0.0, 0.0).unwrap(),
            [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
        );
        assert_matrix_approx(
            &eval_fn("rotate(180deg)", 0.0, 0.0).unwrap(),
            [-1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        );
        assert_matrix_approx(
            &eval_fn("rotate(-90deg)", 0.0, 0.0).unwrap(),
            [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        );
        // Angle units normalize: 0.25turn == 100grad == 90deg.
        let deg = eval_fn("rotate(90deg)", 0.0, 0.0).unwrap();
        for other in ["rotate(0.25turn)", "rotate(100grad)"] {
            let m = eval_fn(other, 0.0, 0.0).unwrap();
            assert_matrix_approx(&m, [deg.a, deg.b, deg.c, deg.d, deg.e, deg.f]);
        }
        // rotate(0deg) is exactly the identity.
        assert!(eval_fn("rotate(0deg)", 0.0, 0.0).unwrap().is_identity());
    }

    #[test]
    fn transform_to_matrix_matrix_fn() {
        assert_matrix_exact(
            &eval_fn("matrix(1, 2, 3, 4, 5, 6)", 0.0, 0.0).unwrap(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        // e/f are px offsets.
        let m = eval_fn("matrix(1, 0, 0, 1, 10, 20)", 0.0, 0.0).unwrap();
        assert_eq!(m.apply(0.0, 0.0), (10.0, 20.0));
        // Wrong arity or non-numbers fail closed.
        assert!(eval_fn("matrix(1, 2, 3)", 0.0, 0.0).is_none());
        assert!(eval_fn("matrix(1, 2, 3, 4, 5, 6, 7)", 0.0, 0.0).is_none());
        assert!(eval_fn("matrix(1, 0, 0, 1, 10px, 0)", 0.0, 0.0).is_none());
        assert!(eval_fn("matrix(a, 0, 0, 1, 0, 0)", 0.0, 0.0).is_none());
    }

    #[test]
    fn transform_to_matrix_skew() {
        // CSS Transforms 1 §15.1: skew(ax, ay) = matrix(1, tan(ay), tan(ax), 1, 0, 0).
        assert_matrix_approx(
            &eval_fn("skew(45deg)", 0.0, 0.0).unwrap(),
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        );
        assert_matrix_approx(
            &eval_fn("skew(45deg, 45deg)", 0.0, 0.0).unwrap(),
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        );
        assert_matrix_approx(
            &eval_fn("skewx(45deg)", 0.0, 0.0).unwrap(),
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        );
        assert_matrix_approx(
            &eval_fn("skewy(45deg)", 0.0, 0.0).unwrap(),
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        );
        // Zero skew is exactly the identity.
        assert!(eval_fn("skew(0deg)", 0.0, 0.0).unwrap().is_identity());
        // A unitless non-zero angle is not valid; three args is wrong arity.
        assert!(eval_fn("skew(45)", 0.0, 0.0).is_none());
        assert!(eval_fn("skew(10deg, 20deg, 30deg)", 0.0, 0.0).is_none());
        // Nested functions fail closed.
        assert!(eval_fn("skew(calc(10deg))", 0.0, 0.0).is_none());
    }

    #[test]
    fn transform_composition_applies_right_to_left() {
        // `translate(10px, 0) scale(2)`: the point is scaled first, then
        // translated: (1,0) -> (2,0) -> (12,0).
        let m = eval_list("translate(10px, 0) scale(2)", 0.0, 0.0).unwrap();
        assert_matrix_exact(&m, [2.0, 0.0, 0.0, 2.0, 10.0, 0.0]);
        assert_eq!(m.apply(1.0, 0.0), (12.0, 0.0));
        // Reversed order translates first: (1,0) -> (11,0) -> (22,0).
        let m = eval_list("scale(2) translate(10px, 0)", 0.0, 0.0).unwrap();
        assert_matrix_exact(&m, [2.0, 0.0, 0.0, 2.0, 20.0, 0.0]);
        assert_eq!(m.apply(1.0, 0.0), (22.0, 0.0));
        // A rotation rotates the translation that follows it in the list:
        // `rotate(90deg) translate(100px, 0)` moves the origin to (0, 100).
        let m = eval_list("rotate(90deg) translate(100px, 0)", 0.0, 0.0).unwrap();
        let (x, y) = m.apply(0.0, 0.0);
        assert!(
            (x - 0.0).abs() < 1e-5 && (y - 100.0).abs() < 1e-5,
            "got ({x}, {y})"
        );
        // An empty list composes to the identity.
        let ctx = tctx(10.0, 10.0);
        assert!(compose_transforms(&[], &ctx).unwrap().is_identity());
        // One failing function fails the whole composition.
        let bad = vec![
            TransformFunction::Translate {
                x: CssValue::Length(1.0, LengthUnit::Px),
                y: CssValue::Number(0.0),
            },
            TransformFunction::Other {
                name: "matrix".to_string(),
                raw_args: "1, 2".to_string(),
            },
        ];
        assert!(compose_transforms(&bad, &ctx).is_none());
    }

    #[test]
    fn transform_3d_functions_flatten() {
        // matrix3d: the 2D-relevant components of a column-major 4x4.
        assert_matrix_exact(
            &eval_fn(
                "matrix3d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 0, 1)",
                0.0,
                0.0,
            )
            .unwrap(),
            [1.0, 0.0, 0.0, 1.0, 10.0, 20.0],
        );
        assert!(
            eval_fn(
                "matrix3d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 0)",
                0.0,
                0.0
            )
            .is_none()
        );
        // translate3d drops z; x/y percentages resolve per-axis.
        assert_matrix_exact(
            &eval_fn("translate3d(10px, 20px, 30px)", 0.0, 0.0).unwrap(),
            [1.0, 0.0, 0.0, 1.0, 10.0, 20.0],
        );
        assert_matrix_exact(
            &eval_fn("translate3d(50%, 50%, 0px)", 200.0, 100.0).unwrap(),
            [1.0, 0.0, 0.0, 1.0, 100.0, 50.0],
        );
        // Pure z translation / scaling is invisible in 2D.
        assert!(eval_fn("translatez(30px)", 0.0, 0.0).unwrap().is_identity());
        assert!(eval_fn("scalez(4)", 0.0, 0.0).unwrap().is_identity());
        assert_matrix_exact(
            &eval_fn("scale3d(2, 3, 4)", 0.0, 0.0).unwrap(),
            [2.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        );
        // 3D rotations project orthographically: rotatex(90°) foreshortens
        // y to zero, rotatey(90°) foreshortens x.
        assert_matrix_approx(
            &eval_fn("rotatex(90deg)", 0.0, 0.0).unwrap(),
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_matrix_approx(
            &eval_fn("rotatey(90deg)", 0.0, 0.0).unwrap(),
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );
        // rotatez and a z-axis rotate3d match the 2D rotate().
        let r2 = eval_fn("rotate(90deg)", 0.0, 0.0).unwrap();
        for other in ["rotatez(90deg)", "rotate3d(0, 0, 1, 90deg)"] {
            let m = eval_fn(other, 0.0, 0.0).unwrap();
            assert_matrix_approx(&m, [r2.a, r2.b, r2.c, r2.d, r2.e, r2.f]);
        }
        // rotate3d about x matches rotatex.
        let rx = eval_fn("rotatex(45deg)", 0.0, 0.0).unwrap();
        let r3x = eval_fn("rotate3d(1, 0, 0, 45deg)", 0.0, 0.0).unwrap();
        assert_matrix_approx(&r3x, [rx.a, rx.b, rx.c, rx.d, rx.e, rx.f]);
        // A zero rotate3d axis is the identity per spec.
        assert!(
            eval_fn("rotate3d(0, 0, 0, 45deg)", 0.0, 0.0)
                .unwrap()
                .is_identity()
        );
        // A lone perspective() only affects child 3D transforms: identity.
        assert!(
            eval_fn("perspective(500px)", 0.0, 0.0)
                .unwrap()
                .is_identity()
        );
        // Unknown opaque functions fail closed.
        let unknown = TransformFunction::Other {
            name: "bogus".to_string(),
            raw_args: "1".to_string(),
        };
        let ctx = tctx(10.0, 10.0);
        assert!(unknown.to_matrix(&ctx).is_none());
    }

    #[test]
    fn transform_origin_parse() {
        let pct = CssValue::Percentage;
        let px = |n: f32| CssValue::Length(n, LengthUnit::Px);
        let kw = |s: &str| CssValue::Keyword(s.into());
        let origin = |x: CssValue, y: CssValue| TransformOrigin { x, y };
        // Keywords normalize to percentages; one component centers the other.
        assert_eq!(
            TransformOrigin::parse(&kw("left")),
            Some(origin(pct(0.0), pct(50.0)))
        );
        assert_eq!(
            TransformOrigin::parse(&kw("right")),
            Some(origin(pct(100.0), pct(50.0)))
        );
        assert_eq!(
            TransformOrigin::parse(&kw("center")),
            Some(origin(pct(50.0), pct(50.0)))
        );
        assert_eq!(
            TransformOrigin::parse(&kw("top")),
            Some(origin(pct(50.0), pct(0.0)))
        );
        assert_eq!(
            TransformOrigin::parse(&kw("bottom")),
            Some(origin(pct(50.0), pct(100.0)))
        );
        // Pairs: first is x, second is y.
        assert_eq!(
            TransformOrigin::parse(&ok("left top")),
            Some(origin(pct(0.0), pct(0.0)))
        );
        assert_eq!(
            TransformOrigin::parse(&ok("right bottom")),
            Some(origin(pct(100.0), pct(100.0)))
        );
        // A leading y-only keyword swaps (`top left`).
        assert_eq!(
            TransformOrigin::parse(&ok("top left")),
            Some(origin(pct(0.0), pct(0.0)))
        );
        assert_eq!(
            TransformOrigin::parse(&ok("bottom 10px")),
            Some(origin(px(10.0), pct(100.0)))
        );
        // Lengths and percentages.
        assert_eq!(
            TransformOrigin::parse(&ok("10px 20px")),
            Some(origin(px(10.0), px(20.0)))
        );
        assert_eq!(
            TransformOrigin::parse(&ok("25%")),
            Some(origin(pct(25.0), pct(50.0)))
        );
        assert_eq!(
            TransformOrigin::parse(&ok("10px")),
            Some(origin(px(10.0), pct(50.0)))
        );
        assert_eq!(
            TransformOrigin::parse(&ok("center 10px")),
            Some(origin(pct(50.0), px(10.0)))
        );
        // Unitless zero is a valid length.
        assert_eq!(
            TransformOrigin::parse(&ok("0 0")),
            Some(origin(CssValue::Number(0.0), CssValue::Number(0.0)))
        );
        // The initial value parses to 50% 50%.
        assert_eq!(TransformOrigin::initial(), origin(pct(50.0), pct(50.0)));
        // Outside the grammar: three components, two x keywords, colors,
        // bare non-zero numbers, and a cross-axis keyword second.
        assert_eq!(TransformOrigin::parse(&ok("1px 2px 3px")), None);
        assert_eq!(TransformOrigin::parse(&ok("left right")), None);
        assert_eq!(TransformOrigin::parse(&ok("red")), None);
        assert_eq!(TransformOrigin::parse(&ok("10")), None);
        assert_eq!(TransformOrigin::parse(&ok("10px left")), None);
    }

    #[test]
    fn transform_origin_apply_shifts_pivot() {
        // rotate(90°) about the center of a 100×100 box: the right-center
        // point (100, 50) swings clockwise to the bottom-center (50, 100).
        let ctx = tctx(100.0, 100.0);
        let m = eval_list("rotate(90deg)", 100.0, 100.0).unwrap();
        let origin = TransformOrigin::parse(&ok("center")).unwrap();
        let pivoted = origin.apply(&m, &ctx).unwrap();
        let (x, y) = pivoted.apply(100.0, 50.0);
        assert!(
            (x - 50.0).abs() < 1e-4 && (y - 100.0).abs() < 1e-4,
            "got ({x}, {y})"
        );
        // Without the origin (pivot at 0,0) the same point lands at (-50, 100).
        let (x, y) = m.apply(100.0, 50.0);
        assert!(
            (x + 50.0).abs() < 1e-4 && (y - 100.0).abs() < 1e-4,
            "got ({x}, {y})"
        );
        // Pivoting the identity is still the identity.
        let id = TransformMatrix::identity();
        assert!(origin.apply(&id, &ctx).unwrap().is_identity());
        // An unresolvable origin fails closed.
        let bad_origin = TransformOrigin {
            x: CssValue::Length(1.0, LengthUnit::Em),
            y: CssValue::Percentage(50.0),
        };
        assert!(bad_origin.apply(&m, &ctx).is_none());
    }

    #[test]
    fn evaluate_transform_end_to_end() {
        let ctx = tctx(100.0, 100.0);
        let center = TransformOrigin::initial();
        // `none` is the identity regardless of origin.
        let none = evaluate_transform(&ok("none"), &center, &ctx).unwrap();
        assert!(none.is_identity());
        // Pure translations are pivot-independent.
        let m = evaluate_transform(&ok("translate(10px, 0)"), &center, &ctx).unwrap();
        assert_matrix_exact(&m, [1.0, 0.0, 0.0, 1.0, 10.0, 0.0]);
        // rotate(90deg) about the center: the top-left corner swings
        // clockwise to the top-right.
        let m = evaluate_transform(&ok("rotate(90deg)"), &center, &ctx).unwrap();
        let (x, y) = m.apply(0.0, 0.0);
        assert!(
            (x - 100.0).abs() < 1e-4 && (y - 0.0).abs() < 1e-4,
            "got ({x}, {y})"
        );
        // A List of Transforms flattens (the post-var()-substitution shape).
        let list = CssValue::List(vec![ok("translate(10px, 0)"), ok("scale(2)")]);
        let zero = TransformOrigin {
            x: CssValue::Percentage(0.0),
            y: CssValue::Percentage(0.0),
        };
        let m = evaluate_transform(&list, &zero, &ctx).unwrap();
        assert_matrix_exact(&m, [2.0, 0.0, 0.0, 2.0, 10.0, 0.0]);
        // Anything else fails closed.
        assert!(evaluate_transform(&ok("auto"), &center, &ctx).is_none());
    }

    #[test]
    fn resolve_absolute_length_handles_units() {
        let r = resolve_absolute_length;
        assert_eq!(r(&CssValue::Length(2.0, LengthUnit::Px), 100.0), Some(2.0));
        assert_eq!(r(&CssValue::Length(1.0, LengthUnit::In), 100.0), Some(96.0));
        assert_eq!(r(&CssValue::Percentage(50.0), 200.0), Some(100.0));
        assert_eq!(r(&CssValue::Number(0.0), 200.0), Some(0.0));
        assert_eq!(
            r(
                &CssValue::MathDeferred {
                    pct: 50.0,
                    px: 10.0,
                    had_pct: true,
                    affine: true,
                },
                200.0
            ),
            Some(110.0)
        );
        // Relative units and unevaluated math need layout context.
        assert_eq!(r(&CssValue::Length(1.0, LengthUnit::Em), 100.0), None);
        assert_eq!(r(&CssValue::Number(5.0), 100.0), None);
    }

    #[test]
    fn shadow_length_rejects_percentage_holding_deferred_math() {
        // `calc(0% + 10px)` folds to `pct: 0.0, px: 10.0` but
        // syntactically contains a percentage (`had_pct`); shadows have
        // no percentage base, so per CSS Values 4 §8.1.4 it is not a
        // valid shadow length even though the folded coefficient is zero.
        let pct_math = CssValue::MathDeferred {
            pct: 0.0,
            px: 10.0,
            had_pct: true,
            affine: true,
        };
        assert!(!Shadow::is_shadow_length(&pct_math));
        // A pure-length calc (no percentage leaf) stays valid.
        let len_math = CssValue::MathDeferred {
            pct: 0.0,
            px: 10.0,
            had_pct: false,
            affine: true,
        };
        assert!(Shadow::is_shadow_length(&len_math));
        assert!(Shadow::is_shadow_length(&CssValue::Length(
            10.0,
            LengthUnit::Px
        )));
        assert!(!Shadow::is_shadow_length(&CssValue::Percentage(10.0)));
    }

    // --- clip-path `<basic-shape>` regression tests (wave-47 F1/F3/F4/F5) ---

    fn clip_path_of(css: &str) -> ClipPath {
        // Parses `clip-path: …` through the real declaration pipeline
        // and asserts it yields exactly one declaration (a dropped or
        // duplicated declaration fails the count, not just the value).
        let decls = crate::parse::parse_declarations(css);
        assert_eq!(
            decls.len(),
            1,
            "expected exactly one declaration from {css:?}"
        );
        assert_eq!(decls[0].property, "clip-path");
        match &decls[0].value {
            CssValue::ClipPath(cp) => cp.clone(),
            other => panic!("expected ClipPath, got {other:?} from {css:?}"),
        }
    }

    fn cpx(v: f32) -> CssValue {
        CssValue::Length(v, LengthUnit::Px)
    }

    fn cpp(v: f32) -> CssValue {
        CssValue::Percentage(v)
    }

    #[test]
    fn clip_path_inset_single_value_expands_to_four() {
        // F1: `inset(10px)` underflowed `expand_four` (`1 - 2` on a
        // 1-element vec) and panicked in debug and release. It must
        // parse to one declaration with the value on all four sides.
        assert_eq!(
            clip_path_of("clip-path: inset(10px)"),
            ClipPath::Shape(Box::new(BasicShape::Inset {
                offsets: [cpx(10.0), cpx(10.0), cpx(10.0), cpx(10.0)],
                radii: None,
            }))
        );
        // The 2- and 3-value expansions are unchanged.
        assert_eq!(
            clip_path_of("clip-path: inset(10px 20px)"),
            ClipPath::Shape(Box::new(BasicShape::Inset {
                offsets: [cpx(10.0), cpx(20.0), cpx(10.0), cpx(20.0)],
                radii: None,
            }))
        );
        assert_eq!(
            clip_path_of("clip-path: inset(10px 20px 30px)"),
            ClipPath::Shape(Box::new(BasicShape::Inset {
                offsets: [cpx(10.0), cpx(20.0), cpx(30.0), cpx(20.0)],
                radii: None,
            }))
        );
        // A single `round` radius expands the same way.
        assert_eq!(
            clip_path_of("clip-path: inset(10px round 5px)"),
            ClipPath::Shape(Box::new(BasicShape::Inset {
                offsets: [cpx(10.0), cpx(10.0), cpx(10.0), cpx(10.0)],
                radii: Some([cpx(5.0), cpx(5.0), cpx(5.0), cpx(5.0)]),
            }))
        );
    }

    #[test]
    fn clip_path_circle_omitted_radius_defaults_closest_side() {
        // F3: CSS Shapes 1 §6.1 — `circle()` with no radius is
        // `closest-side` (inscribed), not `farthest-side`.
        assert_eq!(
            clip_path_of("clip-path: circle()"),
            ClipPath::Shape(Box::new(BasicShape::Circle {
                radius: ShapeRadius::ClosestSide,
                cx: cpp(50.0),
                cy: cpp(50.0),
            }))
        );
        // Explicit keywords still parse.
        assert_eq!(
            clip_path_of("clip-path: circle(farthest-side)"),
            ClipPath::Shape(Box::new(BasicShape::Circle {
                radius: ShapeRadius::FarthestSide,
                cx: cpp(50.0),
                cy: cpp(50.0),
            }))
        );
    }

    #[test]
    fn clip_path_ellipse_omitted_radii_default_closest_side() {
        // F4: CSS Shapes 1 §6.2 — omitted `ellipse()` radii are
        // `closest-side` per axis, which diverges from `50% 50%` for
        // off-center shapes like `ellipse(at 20% 20%)`.
        assert_eq!(
            clip_path_of("clip-path: ellipse(at 20% 20%)"),
            ClipPath::Shape(Box::new(BasicShape::Ellipse {
                rx: ShapeRadius::ClosestSide,
                ry: ShapeRadius::ClosestSide,
                cx: cpp(20.0),
                cy: cpp(20.0),
            }))
        );
        assert_eq!(
            clip_path_of("clip-path: ellipse()"),
            ClipPath::Shape(Box::new(BasicShape::Ellipse {
                rx: ShapeRadius::ClosestSide,
                ry: ShapeRadius::ClosestSide,
                cx: cpp(50.0),
                cy: cpp(50.0),
            }))
        );
        // Explicit length radii still parse.
        assert_eq!(
            clip_path_of("clip-path: ellipse(40% 30% at 50% 50%)"),
            ClipPath::Shape(Box::new(BasicShape::Ellipse {
                rx: ShapeRadius::Length(cpp(40.0)),
                ry: ShapeRadius::Length(cpp(30.0)),
                cx: cpp(50.0),
                cy: cpp(50.0),
            }))
        );
        // Explicit `closest-side` / `farthest-side` keywords parse.
        assert_eq!(
            clip_path_of("clip-path: ellipse(closest-side farthest-side)"),
            ClipPath::Shape(Box::new(BasicShape::Ellipse {
                rx: ShapeRadius::ClosestSide,
                ry: ShapeRadius::FarthestSide,
                cx: cpp(50.0),
                cy: cpp(50.0),
            }))
        );
    }

    #[test]
    fn clip_path_single_position_value_sets_horizontal_only() {
        // F5: css-values-4 — one `<position>` value sets the horizontal
        // axis; the vertical defaults to `center` (50%), not the value.
        assert_eq!(
            clip_path_of("clip-path: circle(50% at 20%)"),
            ClipPath::Shape(Box::new(BasicShape::Circle {
                radius: ShapeRadius::Length(cpp(50.0)),
                cx: cpp(20.0),
                cy: cpp(50.0),
            }))
        );
        // An explicit vertical keyword centers the horizontal axis.
        assert_eq!(
            clip_path_of("clip-path: circle(50% at top)"),
            ClipPath::Shape(Box::new(BasicShape::Circle {
                radius: ShapeRadius::Length(cpp(50.0)),
                cx: cpp(50.0),
                cy: cpp(0.0),
            }))
        );
        assert_eq!(
            clip_path_of("clip-path: circle(50% at left)"),
            ClipPath::Shape(Box::new(BasicShape::Circle {
                radius: ShapeRadius::Length(cpp(50.0)),
                cx: cpp(0.0),
                cy: cpp(50.0),
            }))
        );
    }

    #[test]
    fn clip_path_two_position_values_still_work() {
        // Two values are (horizontal, vertical), unchanged by F5 …
        assert_eq!(
            clip_path_of("clip-path: circle(50% at 20% 30%)"),
            ClipPath::Shape(Box::new(BasicShape::Circle {
                radius: ShapeRadius::Length(cpp(50.0)),
                cx: cpp(20.0),
                cy: cpp(30.0),
            }))
        );
        // … `left right` (two horizontals) and `top bottom` (two
        // verticals) stay invalid: the declaration is dropped.
        assert!(crate::parse::parse_declarations("clip-path: circle(at left right)").is_empty());
        assert!(crate::parse::parse_declarations("clip-path: circle(at top bottom)").is_empty());
    }
}
