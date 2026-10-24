//! CSS `filter`, `backdrop-filter`, and `clip-path`: parsing, resolution,
//! and rasterization shared by the CPU and WGPU backends.
//!
//! Pipeline (mirrors `gradient.rs`):
//!
//! 1. `bowser_css` parses `filter`/`backdrop-filter` into
//!    [`CssValue::Filter`](bowser_css::CssValue::Filter) (one
//!    [`FilterFunction`](bowser_css::FilterFunction) per function: name
//!    plus retained argument text) and `clip-path` into
//!    [`CssValue::ClipPath`](bowser_css::CssValue::ClipPath) (structured
//!    shapes with [`CssValue`](bowser_css::CssValue) length leaves).
//! 2. [`resolve_filter_chain`] turns the retained filter functions into
//!    backend-neutral [`FilterOp`]s (lengths resolved to px against the
//!    element's font metrics and viewport). [`plan_filter_passes`]
//!    folds the chain into [`FilterPass`]es: adjacent color ops merge
//!    into one [`ColorMatrix`], adjacent blurs merge by σ.
//! 3. [`resolve_clip_path`] binds a `clip-path` value to the element's
//!    border box, producing a [`ResolvedClip`]: `inset()` becomes a rect
//!    (or a rounded rect with `round`), `circle()`/`ellipse()` are
//!    tessellated to polygons, `polygon()` is used as authored.
//! 4. Backends rasterize: [`apply_color_matrix_rgba8`] and
//!    [`apply_drop_shadow`] are the CPU pixel kernels (the WGPU backend
//!    mirrors them in its `SHADER_FILTER`); clip polygons evaluate
//!    through the shared clip-stack coverage in `clip.rs`.
//!
//! Deliberate limitations (documented, not silent):
//!
//! - `url()` filters reference SVG filter primitives and are never
//!   rendered: the function is dropped from the chain.
//! - Filter-function arguments parse from their retained text; anything
//!   the argument parser does not understand (`calc()`, `var()`,
//!   functional color notations in `drop-shadow()`) drops that function,
//!   not the whole chain.
//! - `clip-path: url(...)` is parsed and retained but not rendered
//!   (paints as `none`).
//! - A filter group's blur spill is not cut by `clip-path`/`overflow`
//!   clips: clips apply to the group's content before filtering (both
//!   backends do the same), where Blink cuts the spill after. Group
//!   bounds still inflate by [`filter_spill`] so blurred pixels paint.
//! - A `var()` nested inside a `clip-path` shape is not substituted (the
//!   cascade only folds whole-value `var()`s for this property); an
//!   unresolvable leaf makes the clip a no-op.

use std::f32::consts::PI;

use bowser_css::{
    BasicShape, ClipPath, CssValue, Filter, LengthUnit, ShapeRadius, length_to_px,
    parse_css_angle_deg, parse_css_color,
};
use bowser_layout::{Rect, sane_len};

use crate::borders::clamp_radii;
use crate::cpu::box_blur_sigma_rgba;
use crate::display_list::Color;

// ---------------------------------------------------------------------------
// Shared length context
// ---------------------------------------------------------------------------

/// Font/viewport bases for resolving `filter`/`clip-path` lengths to px.
#[derive(Clone, Copy, Debug)]
pub struct LengthCtx {
    /// The element's own computed font size in px (`em` basis).
    pub em: f32,
    /// Root font size in px (`rem` basis).
    pub root: f32,
    /// Viewport size in px (`vw`/`vh`/`vmin`/`vmax` bases).
    pub viewport: (f32, f32),
}

// ---------------------------------------------------------------------------
// Filter ops
// ---------------------------------------------------------------------------

/// One resolved CSS filter function (Filter Effects 1 §8): lengths in
/// px, angles in degrees, factors unitless (`50%` → `0.5`).
#[derive(Clone, Debug, PartialEq)]
pub enum FilterOp {
    /// `blur(<length>)`: gaussian radius in px (σ = radius / 2, matching
    /// the `box-shadow` blur convention).
    Blur(f32),
    /// `brightness(<number> | <percentage>)`.
    Brightness(f32),
    /// `contrast(<number> | <percentage>)`.
    Contrast(f32),
    /// `grayscale(<number> | <percentage>)`.
    Grayscale(f32),
    /// `hue-rotate(<angle>)`, in degrees.
    HueRotate(f32),
    /// `invert(<number> | <percentage>)`.
    Invert(f32),
    /// `opacity(<number> | <percentage>)`.
    Opacity(f32),
    /// `saturate(<number> | <percentage>)`.
    Saturate(f32),
    /// `sepia(<number> | <percentage>)`.
    Sepia(f32),
    /// `drop-shadow(<length>{2,3} <color>?)`.
    DropShadow {
        offset: (f32, f32),
        blur: f32,
        color: Color,
    },
}

/// Parse a `<number> | <percentage>` argument to a plain factor:
/// `50%` → `0.5`, `2` → `2.0`. Unitless zero included; non-finite
/// rejected.
fn parse_factor(s: &str) -> Option<f32> {
    let s = s.trim();
    let v = if let Some(num) = s.strip_suffix('%') {
        num.trim().parse::<f32>().ok()? / 100.0
    } else {
        s.parse::<f32>().ok()?
    };
    if v.is_finite() { Some(v) } else { None }
}

/// Split `s` into its leading numeric part and the unit suffix:
/// `"5px"` → `(5.0, "px")`, `"50%"` → `(50.0, "%")`, `"0"` → `(0.0, "")`.
/// An `e`/`E` only joins the number as a float exponent when followed
/// by `[+-]?<digit>` — otherwise the unit starts there, so `"1em"`
/// splits as `(1.0, "em")` instead of the unparseable `("1e", "m")`.
fn split_number_unit(s: &str) -> Option<(f32, &str)> {
    let s = s.trim();
    let bytes = s.as_bytes();
    let mut end = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_ascii_digit() || matches!(c, '.' | '+' | '-') {
            end = i + 1;
            i += 1;
        } else if c == 'e' || c == 'E' {
            let mut j = i + 1;
            if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
                j += 1;
            }
            if j < bytes.len() && (bytes[j] as char).is_ascii_digit() {
                while j < bytes.len() && (bytes[j] as char).is_ascii_digit() {
                    j += 1;
                }
                end = j;
                i = j;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    let num = s[..end].parse::<f32>().ok()?;
    if !num.is_finite() {
        return None;
    }
    Some((num, s[end..].trim()))
}

/// Map a CSS length unit name to [`LengthUnit`].
fn length_unit(name: &str) -> Option<LengthUnit> {
    Some(match name.to_ascii_lowercase().as_str() {
        "px" => LengthUnit::Px,
        "cm" => LengthUnit::Cm,
        "mm" => LengthUnit::Mm,
        "in" => LengthUnit::In,
        "pt" => LengthUnit::Pt,
        "pc" => LengthUnit::Pc,
        "q" => LengthUnit::Q,
        "em" => LengthUnit::Em,
        "rem" => LengthUnit::Rem,
        "vw" => LengthUnit::Vw,
        "vh" => LengthUnit::Vh,
        "vmin" => LengthUnit::Vmin,
        "vmax" => LengthUnit::Vmax,
        _ => return None,
    })
}

/// Resolve one filter-function `<length>` argument to px. Unitless zero
/// is valid (CSS-wide); unitless nonzero is rejected, like browsers.
fn parse_filter_length(s: &str, ctx: &LengthCtx) -> Option<f32> {
    let (v, unit) = split_number_unit(s)?;
    if unit.is_empty() {
        return if v == 0.0 { Some(0.0) } else { None };
    }
    let lu = length_unit(unit)?;
    Some(sane_len(length_to_px(
        v,
        &lu,
        ctx.em,
        ctx.root,
        ctx.viewport.0,
        ctx.viewport.1,
    )))
}

/// Parse one `drop-shadow()` from its retained argument text:
/// `<length>{2,3} && <color>?` (the color defaults to `currentcolor`).
fn parse_drop_shadow(args: &str, ctx: &LengthCtx, current: Color) -> Option<FilterOp> {
    let parts: Vec<&str> = args.split_whitespace().collect();
    if parts.len() < 2 || parts.len() > 4 {
        return None;
    }
    let mut lengths: Vec<f32> = Vec::with_capacity(3);
    let mut color: Option<Color> = None;
    for p in parts {
        if let Some(l) = parse_filter_length(p, ctx) {
            if lengths.len() == 3 {
                return None;
            }
            lengths.push(l);
            continue;
        }
        let c = parse_css_color(p)?;
        if color.is_some() {
            return None;
        }
        color = Some(Color::rgba8(c.r, c.g, c.b, c.a));
    }
    if lengths.len() < 2 {
        return None;
    }
    let blur = if lengths.len() == 3 { lengths[2] } else { 0.0 };
    // A negative blur radius is invalid (Filter Effects 1 §8).
    if blur < 0.0 {
        return None;
    }
    Some(FilterOp::DropShadow {
        offset: (lengths[0], lengths[1]),
        blur,
        color: color.unwrap_or(current),
    })
}

/// Resolve one retained filter function to a [`FilterOp`]; `None` when
/// the function is unrenderable (`url()`) or its arguments do not
/// parse — the caller drops just that function.
fn resolve_filter_fn(name: &str, args: &str, ctx: &LengthCtx, current: Color) -> Option<FilterOp> {
    let one_factor = |op: fn(f32) -> FilterOp| {
        let mut parts = args.split_whitespace();
        let (a, extra) = (parts.next()?, parts.next());
        if extra.is_some() {
            return None;
        }
        Some(op(parse_factor(a)?))
    };
    match name {
        "blur" => {
            let mut parts = args.split_whitespace();
            let (a, extra) = (parts.next()?, parts.next());
            if extra.is_some() {
                return None;
            }
            let r = parse_filter_length(a, ctx)?;
            // A negative radius is invalid (Filter Effects 1 §8).
            if r < 0.0 {
                return None;
            }
            Some(FilterOp::Blur(r))
        }
        "brightness" => one_factor(FilterOp::Brightness),
        "contrast" => one_factor(FilterOp::Contrast),
        "grayscale" => one_factor(FilterOp::Grayscale),
        "hue-rotate" => {
            let mut parts = args.split_whitespace();
            let (a, extra) = (parts.next()?, parts.next());
            if extra.is_some() {
                return None;
            }
            Some(FilterOp::HueRotate(parse_css_angle_deg(a)?))
        }
        "invert" => one_factor(FilterOp::Invert),
        "opacity" => one_factor(FilterOp::Opacity),
        "saturate" => one_factor(FilterOp::Saturate),
        "sepia" => one_factor(FilterOp::Sepia),
        "drop-shadow" => parse_drop_shadow(args, ctx, current),
        // `url()` references SVG filter primitives: never rendered.
        _ => None,
    }
}

/// Resolve a parsed `filter`/`backdrop-filter` value to paint-time ops,
/// in source order. `Filter::None` resolves to empty (the caller emits
/// no group). Functions that fail argument parsing are dropped
/// individually; the rest of the chain still applies.
pub fn resolve_filter_chain(filter: &Filter, ctx: &LengthCtx, current: Color) -> Vec<FilterOp> {
    match filter {
        Filter::None => Vec::new(),
        Filter::List(fns) => fns
            .iter()
            .filter_map(|f| resolve_filter_fn(&f.name, &f.raw_args, ctx, current))
            .collect(),
    }
}

/// Canonical memo key for a `filter`/`backdrop-filter` value: the
/// length context and `currentcolor` color, then the function names and
/// retained argument texts joined. The resolved chain depends on the
/// [`LengthCtx`] (per-box font size, root font size, viewport) and on
/// `current` (per-box `color`, the default drop-shadow color), so both
/// join the key: two boxes sharing `blur(1em)` with different font
/// sizes — or sharing `drop-shadow(2px 2px)` with different `color`s —
/// must not share a cache entry. Floats encode as their bit patterns,
/// so the key is canonical for every finite value. `None` for
/// `Filter::None` (the caller skips memoization entirely — no group).
pub fn filter_memo_key(filter: &Filter, ctx: &LengthCtx, current: Color) -> Option<String> {
    match filter {
        Filter::None => None,
        Filter::List(fns) => {
            if fns.is_empty() {
                return None;
            }
            let mut key = format!(
                "ctx:{:08x}:{:08x}:{:08x}:{:08x}:cur:{:02x}{:02x}{:02x}{:02x}:",
                ctx.em.to_bits(),
                ctx.root.to_bits(),
                ctx.viewport.0.to_bits(),
                ctx.viewport.1.to_bits(),
                current.r8(),
                current.g8(),
                current.b8(),
                current.a8(),
            );
            for f in fns {
                key.push_str(&f.name);
                key.push('(');
                key.push_str(&f.raw_args);
                key.push(')');
                key.push(' ');
            }
            Some(key)
        }
    }
}

// ---------------------------------------------------------------------------
// Color matrices
// ---------------------------------------------------------------------------

/// A 4×5 color matrix over unpremultiplied `(r, g, b, a)`: row `i` is
/// `(m[i][0]·r + m[i][1]·g + m[i][2]·b + m[i][3]·a + m[i][4])`.
/// Every non-blur, non-drop-shadow filter function is one of these
/// (Filter Effects 1 §8.1), so adjacent color ops fold into a single
/// pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorMatrix {
    /// Row-major: `m[row][col]`, col 4 is the constant offset.
    pub m: [[f32; 5]; 4],
}

impl ColorMatrix {
    /// Identity.
    pub fn identity() -> Self {
        Self {
            m: [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ],
        }
    }

    /// Compose: `self.after(other)` applies `other` first, then `self`
    /// — the CSS left-to-right filter order when folding a run.
    pub fn after(self, other: Self) -> Self {
        let mut m = [[0.0f32; 5]; 4];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate().take(4) {
                *v = self.m[i][0] * other.m[0][j]
                    + self.m[i][1] * other.m[1][j]
                    + self.m[i][2] * other.m[2][j]
                    + self.m[i][3] * other.m[3][j];
            }
            row[4] = self.m[i][0] * other.m[0][4]
                + self.m[i][1] * other.m[1][4]
                + self.m[i][2] * other.m[2][4]
                + self.m[i][3] * other.m[3][4]
                + self.m[i][4];
        }
        Self { m }
    }

    /// Apply to one unpremultiplied pixel.
    pub fn apply(&self, px: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.m[i][0] * px[0]
                + self.m[i][1] * px[1]
                + self.m[i][2] * px[2]
                + self.m[i][3] * px[3]
                + self.m[i][4];
        }
        out
    }

    /// The matrix for one color [`FilterOp`] (`Blur`/`DropShadow` are not
    /// color ops; mapping them is a logic error).
    pub fn for_op(op: &FilterOp) -> Self {
        debug_assert!(
            !matches!(op, FilterOp::Blur(_) | FilterOp::DropShadow { .. }),
            "blur/drop-shadow are not color-matrix ops"
        );
        let mut m = Self::identity().m;
        match *op {
            FilterOp::Brightness(a) => {
                m[0][0] = a;
                m[1][1] = a;
                m[2][2] = a;
            }
            FilterOp::Contrast(a) => {
                let t = (1.0 - a) * 0.5;
                m[0][0] = a;
                m[0][4] = t;
                m[1][1] = a;
                m[1][4] = t;
                m[2][2] = a;
                m[2][4] = t;
            }
            FilterOp::Opacity(a) => {
                m[3][3] = a;
            }
            FilterOp::Invert(a) => {
                for (i, row) in m.iter_mut().enumerate().take(3) {
                    row[i] = 1.0 - 2.0 * a;
                    row[4] = a;
                }
            }
            FilterOp::Grayscale(a) => {
                // lerp(identity, luminance, a).
                let (r, g, b) = (0.2126, 0.7152, 0.0722);
                let rows = [[r, g, b], [r, g, b], [r, g, b]];
                for i in 0..3 {
                    for j in 0..3 {
                        m[i][j] = (1.0 - a) * (if i == j { 1.0 } else { 0.0 }) + a * rows[i][j];
                    }
                }
            }
            FilterOp::Sepia(a) => {
                // lerp(identity, sepia, a).
                let rows = [
                    [0.393, 0.769, 0.189],
                    [0.349, 0.686, 0.168],
                    [0.272, 0.534, 0.131],
                ];
                for i in 0..3 {
                    for j in 0..3 {
                        m[i][j] = (1.0 - a) * (if i == j { 1.0 } else { 0.0 }) + a * rows[i][j];
                    }
                }
            }
            FilterOp::Saturate(a) => {
                let rows = [
                    [0.213 + 0.787 * a, 0.715 - 0.715 * a, 0.072 - 0.072 * a],
                    [0.213 - 0.213 * a, 0.715 + 0.285 * a, 0.072 - 0.072 * a],
                    [0.213 - 0.213 * a, 0.715 - 0.715 * a, 0.072 + 0.928 * a],
                ];
                for i in 0..3 {
                    m[i][0] = rows[i][0];
                    m[i][1] = rows[i][1];
                    m[i][2] = rows[i][2];
                }
            }
            FilterOp::HueRotate(deg) => {
                let (s, c) = deg.to_radians().sin_cos();
                m[0][0] = 0.213 + c * 0.787 - s * 0.213;
                m[0][1] = 0.715 - c * 0.715 - s * 0.715;
                m[0][2] = 0.072 - c * 0.072 + s * 0.928;
                m[1][0] = 0.213 - c * 0.213 + s * 0.143;
                m[1][1] = 0.715 + c * 0.285 + s * 0.140;
                m[1][2] = 0.072 - c * 0.072 - s * 0.283;
                m[2][0] = 0.213 - c * 0.213 - s * 0.787;
                m[2][1] = 0.715 - c * 0.715 + s * 0.715;
                m[2][2] = 0.072 + c * 0.928 + s * 0.072;
            }
            FilterOp::Blur(_) | FilterOp::DropShadow { .. } => unreachable!(),
        }
        Self { m }
    }
}

// ---------------------------------------------------------------------------
// Filter passes
// ---------------------------------------------------------------------------

/// Integer box-blur half-width whose 3 (H, V) pass pairs approximate
/// gaussian `sigma` (in px): three box passes of width `2r+1` have
/// total variance `3·r·(r+1)`, so `r² + r − σ² = 0`. Shared by both
/// backends (and the drop-shadow passes) so the CPU and WGPU blur the
/// same radius.
pub fn blur_box_radius(sigma: f32) -> u32 {
    if sigma <= 0.0 {
        return 0;
    }
    ((-1.0 + (1.0 + 4.0 * sigma * sigma).sqrt()) / 2.0)
        .round()
        .max(1.0) as u32
}

/// One backend-executable filter pass: the unit both backends implement.
#[derive(Clone, Debug)]
pub enum FilterPass {
    /// Apply a folded [`ColorMatrix`] to every pixel (unpremultiplied
    /// f32, clamped to [0, 1] afterwards).
    Color(ColorMatrix),
    /// Gaussian blur with the CSS px radius (σ = radius / 2), applied in
    /// premultiplied space like `box-shadow` blurs.
    Blur(f32),
    /// `drop-shadow()`: the tinted, blurred alpha drawn offset *under*
    /// the source.
    DropShadow {
        offset: (f32, f32),
        blur: f32,
        color: Color,
    },
}

/// Plan a filter chain as passes: adjacent color ops fold into one
/// matrix (CSS applies them left to right), adjacent blurs merge
/// (`σ²` adds for gaussians). `DropShadow` always stands alone — it is
/// not linear in the source. Pass order is the source order; backends
/// clamp each pass's output to [0, 1].
pub fn plan_filter_passes(ops: &[FilterOp]) -> Vec<FilterPass> {
    let mut passes: Vec<FilterPass> = Vec::new();
    for op in ops {
        match op {
            FilterOp::Blur(r) => {
                if *r <= 0.0 || !r.is_finite() {
                    continue;
                }
                match passes.last_mut() {
                    Some(FilterPass::Blur(acc)) => *acc = (*acc * *acc + r * r).sqrt(),
                    _ => passes.push(FilterPass::Blur(*r)),
                }
            }
            FilterOp::DropShadow {
                offset,
                blur,
                color,
            } => {
                passes.push(FilterPass::DropShadow {
                    offset: *offset,
                    blur: *blur,
                    color: *color,
                });
            }
            color_op => {
                let m = ColorMatrix::for_op(color_op);
                match passes.last_mut() {
                    Some(FilterPass::Color(acc)) => *acc = m.after(*acc),
                    _ => passes.push(FilterPass::Color(m)),
                }
            }
        }
    }
    passes
}

/// Conservative page-px spill of a filter chain beyond the group's paint
/// bounds. A gaussian blur of CSS radius `r` (σ = `r` / 2) is rendered
/// as three box passes of half-width [`blur_box_radius`] per side, so
/// the true support is `3 · blur_box_radius(σ)` per side, plus 1px for
/// the ±0.5 texel rounding; each drop-shadow spills its offset plus the
/// same blur support of its own blur. Backends inflate isolated group
/// bounds by this so filtered pixels are not culled.
pub fn filter_spill(ops: &[FilterOp]) -> f32 {
    fn blur_support(r: f32) -> f32 {
        3.0 * (blur_box_radius(r / 2.0) as f32 + 1.0)
    }
    let mut spill = 0.0f32;
    for op in ops {
        match op {
            FilterOp::Blur(r) => spill += blur_support(*r),
            FilterOp::DropShadow { offset, blur, .. } => {
                spill += offset.0.abs().max(offset.1.abs()) + blur_support(*blur);
            }
            _ => {}
        }
    }
    spill.max(0.0)
}

// ---------------------------------------------------------------------------
// CPU pixel kernels
// ---------------------------------------------------------------------------

/// Apply one [`FilterPass::Color`] matrix to premultiplied RGBA8 bytes
/// in place. Each pixel unpremultiplies to f32, takes the matrix,
/// clamps to [0, 1], and premultiplies back — the same pipeline the
/// WGPU `fs_filter_color` entry point runs, so the backends agree
/// within rounding. Fully transparent pixels stay untouched.
pub fn apply_color_matrix_rgba8(buf: &mut [u8], m: &ColorMatrix) {
    debug_assert_eq!(buf.len() % 4, 0);
    let (chunks, _) = buf.as_chunks_mut::<4>();
    for px in chunks {
        let a = f32::from(px[3]) / 255.0;
        if a <= 0.0 {
            continue;
        }
        let r = f32::from(px[0]) / 255.0 / a;
        let g = f32::from(px[1]) / 255.0 / a;
        let b = f32::from(px[2]) / 255.0 / a;
        let [r, g, b, a2] = m.apply([r, g, b, a]);
        let (r, g, b, a2) = (
            r.clamp(0.0, 1.0),
            g.clamp(0.0, 1.0),
            b.clamp(0.0, 1.0),
            a2.clamp(0.0, 1.0),
        );
        px[0] = (r * a2 * 255.0).round() as u8;
        px[1] = (g * a2 * 255.0).round() as u8;
        px[2] = (b * a2 * 255.0).round() as u8;
        px[3] = (a2 * 255.0).round() as u8;
    }
}

/// Source-over composite of premultiplied `src` over premultiplied
/// `dst`, in place on `dst` (f32 0..1 components).
fn over_premultiplied(dst: &mut [f32; 4], src: [f32; 4]) {
    let inv = 1.0 - src[3];
    dst[0] = src[0] + dst[0] * inv;
    dst[1] = src[1] + dst[1] * inv;
    dst[2] = src[2] + dst[2] * inv;
    dst[3] = src[3] + dst[3] * inv;
}

/// Paint a [`FilterPass::DropShadow`] over premultiplied RGBA8 `buf`
/// (`w` × `h`) in place: the source's alpha, tinted with `color` and
/// blurred with σ = `blur` / 2, is drawn offset by `offset` *under*
/// the source. `scratch` and `tmp` are caller-provided same-size
/// scratch buffers (the painter reuses one allocation across passes);
/// the offset snaps to whole texels, exactly like the WGPU
/// drop-shadow pass.
pub fn apply_drop_shadow(
    buf: &mut [u8],
    size: (usize, usize),
    scratch: &mut [u8],
    tmp: &mut [u8],
    offset: (f32, f32),
    blur: f32,
    color: Color,
) {
    let (w, h) = size;
    assert_eq!(buf.len(), w * h * 4);
    assert_eq!(scratch.len(), buf.len());
    assert_eq!(tmp.len(), buf.len());
    if w == 0 || h == 0 {
        return;
    }
    let (cr, cg, cb) = (
        f32::from(color.r8()) / 255.0,
        f32::from(color.g8()) / 255.0,
        f32::from(color.b8()) / 255.0,
    );
    let ca = f32::from(color.a8()) / 255.0;
    // Tint the source alpha with the shadow color (premultiplied).
    let (dst_chunks, _) = scratch.as_chunks_mut::<4>();
    let (src_chunks, _) = buf.as_chunks::<4>();
    for (dst, src) in dst_chunks.iter_mut().zip(src_chunks.iter()) {
        let a = f32::from(src[3]) / 255.0 * ca;
        dst[0] = (cr * a * 255.0).round() as u8;
        dst[1] = (cg * a * 255.0).round() as u8;
        dst[2] = (cb * a * 255.0).round() as u8;
        dst[3] = (a * 255.0).round() as u8;
    }
    // Blur the shadow (same box-blur-as-gaussian the `Blur` pass uses).
    box_blur_sigma_rgba(scratch, tmp, w, h, blur / 2.0);
    // Composite: source OVER the offset shadow. The offset snaps to
    // whole texels, exactly like the WGPU drop-shadow pass.
    let (ox, oy) = (offset.0.round() as isize, offset.1.round() as isize);
    let (w, h) = (w as isize, h as isize);
    for y in 0..h {
        for x in 0..w {
            let di = (y * w + x) as usize * 4;
            let (sx, sy) = (x - ox, y - oy);
            let mut accum = [0.0f32; 4];
            if sx >= 0 && sx < w && sy >= 0 && sy < h {
                let si = (sy * w + sx) as usize * 4;
                accum = [
                    f32::from(scratch[si]) / 255.0,
                    f32::from(scratch[si + 1]) / 255.0,
                    f32::from(scratch[si + 2]) / 255.0,
                    f32::from(scratch[si + 3]) / 255.0,
                ];
            }
            let src = [
                f32::from(buf[di]) / 255.0,
                f32::from(buf[di + 1]) / 255.0,
                f32::from(buf[di + 2]) / 255.0,
                f32::from(buf[di + 3]) / 255.0,
            ];
            over_premultiplied(&mut accum, src);
            buf[di] = (accum[0].clamp(0.0, 1.0) * 255.0).round() as u8;
            buf[di + 1] = (accum[1].clamp(0.0, 1.0) * 255.0).round() as u8;
            buf[di + 2] = (accum[2].clamp(0.0, 1.0) * 255.0).round() as u8;
            buf[di + 3] = (accum[3].clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// clip-path resolution
// ---------------------------------------------------------------------------

/// A `clip-path` value bound to a concrete reference box: what the
/// backends intersect with the clip stack. All coordinates are absolute
/// page px.
#[derive(Clone, Debug)]
pub enum ResolvedClip {
    /// `inset()` without `round`: a plain rect.
    Rect(Rect),
    /// `inset()` with `round <radii>`: radii CSS-clamped to the rect.
    RoundedRect { rect: Rect, radii: [f32; 4] },
    /// `circle()`/`ellipse()` tessellated, or `polygon()` as authored:
    /// vertices in page px, wound either way (even-odd rule).
    Polygon { points: Vec<[f32; 2]> },
}

/// Tessellation segments for `circle()`/`ellipse()`: also the polygon
/// vertex cap the WGPU clip uniform carries per entry
/// (`MAX_POLY_VERTS` in `clip.rs`).
pub const CLIP_CIRCLE_SEGMENTS: usize = 48;

/// Cap on authored `polygon()` vertices (hostile-input bound).
const MAX_POLYGON_POINTS: usize = 4096;

/// Resolve one `clip-path` length leaf to px: percentages against
/// `pct_basis`, `MathDeferred` coefficients against the same basis
/// (the frozen base-0 form contributes its constant), lengths against
/// the font/viewport context, unitless numbers as px (layout's
/// leniency). `None` for unresolvable leaves (`var()` the cascade did
/// not substitute, unevaluated `Math`, …).
fn clip_len(v: &CssValue, ctx: &LengthCtx, pct_basis: f32) -> Option<f32> {
    match v {
        CssValue::Percentage(p) => Some(sane_len(p / 100.0 * pct_basis)),
        CssValue::MathDeferred { pct, px, .. } => Some(sane_len(pct / 100.0 * pct_basis + px)),
        CssValue::Length(n, unit) => Some(sane_len(length_to_px(
            *n,
            unit,
            ctx.em,
            ctx.root,
            ctx.viewport.0,
            ctx.viewport.1,
        ))),
        CssValue::Number(n) => Some(sane_len(*n)),
        _ => None,
    }
}

/// Tessellate an ellipse centered at `(cx, cy)` (page px) into
/// [`CLIP_CIRCLE_SEGMENTS`] vertices. Computed once per display-list
/// build and shared by both backends, so they agree bit-for-bit.
fn tessellate_ellipse(cx: f32, cy: f32, rx: f32, ry: f32) -> Vec<[f32; 2]> {
    (0..CLIP_CIRCLE_SEGMENTS)
        .map(|i| {
            let a = i as f32 / CLIP_CIRCLE_SEGMENTS as f32 * 2.0 * PI;
            [cx + rx * a.cos(), cy + ry * a.sin()]
        })
        .collect()
}

/// Resolve a `clip-path` value against the reference box (the border
/// box) to a [`ResolvedClip`]. Returns `None` for `none`, `url()`,
/// degenerate shapes (empty polygon, non-positive radius, non-finite
/// geometry), and unresolvable leaves — the caller then emits no clip
/// item, which is the zero-cost fast path.
pub fn resolve_clip_path(cp: &ClipPath, ref_box: &Rect, ctx: &LengthCtx) -> Option<ResolvedClip> {
    let shape = match cp {
        ClipPath::None | ClipPath::Url(_) => return None,
        ClipPath::Shape(shape) => shape.as_ref(),
    };
    if !ref_box.w.is_finite() || !ref_box.h.is_finite() {
        return None;
    }
    match shape {
        BasicShape::Inset { offsets, radii } => {
            let [t, r, b, l] = offsets;
            let (top, right, bottom, left) = (
                clip_len(t, ctx, ref_box.h)?,
                clip_len(r, ctx, ref_box.w)?,
                clip_len(b, ctx, ref_box.h)?,
                clip_len(l, ctx, ref_box.w)?,
            );
            let rect = Rect {
                x: ref_box.x + left,
                y: ref_box.y + top,
                w: ref_box.w - left - right,
                h: ref_box.h - top - bottom,
            };
            match radii {
                None => Some(ResolvedClip::Rect(rect)),
                Some(rs) => {
                    let [rtl, rtr, rbr, rbl] = rs;
                    // Circular radii resolve against the width, like
                    // `border-radius`'s horizontal radius.
                    let radii = [
                        clip_len(rtl, ctx, ref_box.w)?,
                        clip_len(rtr, ctx, ref_box.w)?,
                        clip_len(rbr, ctx, ref_box.w)?,
                        clip_len(rbl, ctx, ref_box.w)?,
                    ];
                    Some(ResolvedClip::RoundedRect {
                        rect,
                        radii: clamp_radii(rect, radii),
                    })
                }
            }
        }
        BasicShape::Circle { radius, cx, cy } => {
            let (cx, cy) = (
                ref_box.x + clip_len(cx, ctx, ref_box.w)?,
                ref_box.y + clip_len(cy, ctx, ref_box.h)?,
            );
            // Percentage radii resolve against
            // `sqrt(w² + h²) / √2` (CSS Shapes 1 §6).
            let diag = (ref_box.w * ref_box.w + ref_box.h * ref_box.h).sqrt() / 2.0f32.sqrt();
            let r = match radius {
                ShapeRadius::Length(v) => clip_len(v, ctx, diag)?,
                ShapeRadius::ClosestSide => (cx - ref_box.x)
                    .min((ref_box.x + ref_box.w) - cx)
                    .min(cy - ref_box.y)
                    .min((ref_box.y + ref_box.h) - cy),
                ShapeRadius::FarthestSide => (cx - ref_box.x)
                    .max((ref_box.x + ref_box.w) - cx)
                    .max(cy - ref_box.y)
                    .max((ref_box.y + ref_box.h) - cy),
            };
            if !r.is_finite() || r <= 0.0 {
                return None;
            }
            Some(ResolvedClip::Polygon {
                points: tessellate_ellipse(cx, cy, r, r),
            })
        }
        BasicShape::Ellipse { rx, ry, cx, cy } => {
            let (cx, cy) = (
                ref_box.x + clip_len(cx, ctx, ref_box.w)?,
                ref_box.y + clip_len(cy, ctx, ref_box.h)?,
            );
            // Each radius is a `<shape-radius>`, resolved per axis (CSS
            // Shapes 1 §6.2): `closest-side`/`farthest-side` measure
            // from the center to the nearer/farther side on that axis.
            let axis_radius = |r: &ShapeRadius, basis: f32, c: f32, lo: f32| -> Option<f32> {
                match r {
                    ShapeRadius::Length(v) => clip_len(v, ctx, basis),
                    ShapeRadius::ClosestSide => Some((c - lo).min(lo + basis - c)),
                    ShapeRadius::FarthestSide => Some((c - lo).max(lo + basis - c)),
                }
            };
            let (rx, ry) = (
                axis_radius(rx, ref_box.w, cx, ref_box.x)?,
                axis_radius(ry, ref_box.h, cy, ref_box.y)?,
            );
            if !rx.is_finite() || !ry.is_finite() || rx <= 0.0 || ry <= 0.0 {
                return None;
            }
            Some(ResolvedClip::Polygon {
                points: tessellate_ellipse(cx, cy, rx, ry),
            })
        }
        BasicShape::Polygon { points } => {
            if points.len() < 3 || points.len() > MAX_POLYGON_POINTS {
                return None;
            }
            // Past the backend vertex cap, uniformly decimate to
            // [`CLIP_CIRCLE_SEGMENTS`] vertices (`i·n/48` is strictly
            // increasing for `n > 48`, so no vertex repeats) instead of
            // truncating: truncation would chord-close a partial arc —
            // a 100-gon would render as a ~172° wedge — while decimation
            // keeps the full outline.
            let n = points.len();
            let mut out = Vec::with_capacity(n.min(CLIP_CIRCLE_SEGMENTS));
            for i in 0..n.min(CLIP_CIRCLE_SEGMENTS) {
                let (x, y) = if n <= CLIP_CIRCLE_SEGMENTS {
                    &points[i]
                } else {
                    &points[i * n / CLIP_CIRCLE_SEGMENTS]
                };
                let (x, y) = (
                    ref_box.x + clip_len(x, ctx, ref_box.w)?,
                    ref_box.y + clip_len(y, ctx, ref_box.h)?,
                );
                if !x.is_finite() || !y.is_finite() {
                    return None;
                }
                out.push([x, y]);
            }
            if out.len() < 3 {
                return None;
            }
            Some(ResolvedClip::Polygon { points: out })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> LengthCtx {
        LengthCtx {
            em: 16.0,
            root: 16.0,
            viewport: (800.0, 600.0),
        }
    }

    fn factor_fn(name: &str, args: &str) -> Filter {
        Filter::List(vec![bowser_css::FilterFunction {
            name: name.to_string(),
            raw_args: args.to_string(),
        }])
    }

    #[test]
    fn factor_and_length_args_parse() {
        let c = ctx();
        let black = Color::rgba8(0, 0, 0, 255);
        let ops = resolve_filter_chain(&factor_fn("brightness", "50%"), &c, black);
        assert_eq!(ops, vec![FilterOp::Brightness(0.5)]);
        let ops = resolve_filter_chain(&factor_fn("contrast", "2"), &c, black);
        assert_eq!(ops, vec![FilterOp::Contrast(2.0)]);
        let ops = resolve_filter_chain(&factor_fn("blur", "5px"), &c, black);
        assert_eq!(ops, vec![FilterOp::Blur(5.0)]);
        // Unknown args drop the function, not the chain.
        let f = Filter::List(vec![
            bowser_css::FilterFunction {
                name: "blur".to_string(),
                raw_args: "calc(1px + 1px)".to_string(),
            },
            bowser_css::FilterFunction {
                name: "grayscale".to_string(),
                raw_args: "1".to_string(),
            },
        ]);
        let ops = resolve_filter_chain(&f, &c, black);
        assert_eq!(ops, vec![FilterOp::Grayscale(1.0)]);
        // Negative blur is invalid.
        let ops = resolve_filter_chain(&factor_fn("blur", "-5px"), &c, black);
        assert!(ops.is_empty());
        // url() is never rendered.
        let ops = resolve_filter_chain(&factor_fn("url", "#f"), &c, black);
        assert!(ops.is_empty());
    }

    #[test]
    fn hue_rotate_and_drop_shadow_parse() {
        let c = ctx();
        let black = Color::rgba8(0, 0, 0, 255);
        let ops = resolve_filter_chain(&factor_fn("hue-rotate", "180deg"), &c, black);
        assert_eq!(ops, vec![FilterOp::HueRotate(180.0)]);
        let ops = resolve_filter_chain(&factor_fn("hue-rotate", "0.5turn"), &c, black);
        assert_eq!(ops, vec![FilterOp::HueRotate(180.0)]);
        let ops = resolve_filter_chain(&factor_fn("drop-shadow", "2px 3px 4px red"), &c, black);
        assert_eq!(
            ops,
            vec![FilterOp::DropShadow {
                offset: (2.0, 3.0),
                blur: 4.0,
                color: Color::rgba8(255, 0, 0, 255),
            }]
        );
        // Color defaults to currentcolor.
        let ops = resolve_filter_chain(&factor_fn("drop-shadow", "2px 3px"), &c, black);
        assert_eq!(
            ops,
            vec![FilterOp::DropShadow {
                offset: (2.0, 3.0),
                blur: 0.0,
                color: black,
            }]
        );
        // Hex colors work; a negative blur drops the function.
        let ops =
            resolve_filter_chain(&factor_fn("drop-shadow", "1px 1px -2px #00ff00"), &c, black);
        assert!(ops.is_empty());
    }

    #[test]
    fn color_matrix_folding() {
        // brightness(0.5) then contrast(2): folded left-to-right.
        let ops = vec![FilterOp::Brightness(0.5), FilterOp::Contrast(2.0)];
        let passes = plan_filter_passes(&ops);
        assert_eq!(passes.len(), 1);
        let m = match &passes[0] {
            FilterPass::Color(m) => m,
            _ => panic!("expected a color pass"),
        };
        // brightness halves, then contrast doubles around the 0.5 pivot:
        // r -> 0.5r -> 2*(0.5r) - 0.5 = r - 0.5.
        let [r, _, _, _] = m.apply([0.8, 0.2, 0.4, 1.0]);
        assert!((r - 0.3).abs() < 1e-5, "r={r}");
    }

    #[test]
    fn blur_passes_merge() {
        let ops = vec![FilterOp::Blur(3.0), FilterOp::Blur(4.0)];
        let passes = plan_filter_passes(&ops);
        assert_eq!(passes.len(), 1);
        match &passes[0] {
            FilterPass::Blur(r) => assert!((r - 5.0).abs() < 1e-5, "r={r}"),
            _ => panic!("expected a blur pass"),
        }
        // A drop-shadow breaks the run.
        let ops = vec![
            FilterOp::Blur(3.0),
            FilterOp::DropShadow {
                offset: (0.0, 0.0),
                blur: 0.0,
                color: Color::rgba8(0, 0, 0, 255),
            },
            FilterOp::Blur(4.0),
        ];
        assert_eq!(plan_filter_passes(&ops).len(), 3);
    }

    #[test]
    fn grayscale_matrix_values() {
        // grayscale(1) of pure red = 0.2126 (the spec matrix).
        let m = ColorMatrix::for_op(&FilterOp::Grayscale(1.0));
        let [r, g, b, a] = m.apply([1.0, 0.0, 0.0, 1.0]);
        assert!((r - 0.2126).abs() < 1e-4, "r={r}");
        assert!((g - 0.2126).abs() < 1e-4, "g={g}");
        assert!((b - 0.2126).abs() < 1e-4, "b={b}");
        assert_eq!(a, 1.0);
        // grayscale(0) is the identity.
        let m = ColorMatrix::for_op(&FilterOp::Grayscale(0.0));
        assert_eq!(m.apply([0.3, 0.5, 0.7, 0.9]), [0.3, 0.5, 0.7, 0.9]);
    }

    #[test]
    fn invert_matrix_values() {
        // invert(1): r -> 1 - r; invert(0.5): r -> 0.5. Approximate:
        // the f32 multiply-add is not bit-exact (0.6 -> 0.39999998).
        let m = ColorMatrix::for_op(&FilterOp::Invert(1.0));
        let [r, g, b, a] = m.apply([0.2, 0.4, 0.6, 1.0]);
        for (v, e) in [(r, 0.8), (g, 0.6), (b, 0.4), (a, 1.0)] {
            assert!((v - e).abs() < 1e-6, "v={v} e={e}");
        }
        let m = ColorMatrix::for_op(&FilterOp::Invert(0.5));
        let [r, _, _, _] = m.apply([0.2, 0.4, 0.6, 1.0]);
        assert!((r - 0.5).abs() < 1e-6, "r={r}");
    }

    #[test]
    fn cpu_color_matrix_round_trip() {
        // brightness(0.5) on opaque red: (255,0,0,255) -> (128,0,0,255).
        let mut buf = vec![255u8, 0, 0, 255];
        let m = ColorMatrix::for_op(&FilterOp::Brightness(0.5));
        apply_color_matrix_rgba8(&mut buf, &m);
        assert_eq!(buf, vec![128, 0, 0, 255]);
        // Transparent pixels are untouched.
        let mut buf = vec![255u8, 0, 0, 0];
        apply_color_matrix_rgba8(&mut buf, &m);
        assert_eq!(buf, vec![255, 0, 0, 0]);
        // opacity(0.5): alpha halves, rgb stays premultiplied-consistent.
        let mut buf = vec![200u8, 100, 50, 200];
        let m = ColorMatrix::for_op(&FilterOp::Opacity(0.5));
        apply_color_matrix_rgba8(&mut buf, &m);
        assert_eq!(buf[3], 100);
        assert!((buf[0] as f32 - 100.0).abs() <= 1.0, "buf={buf:?}");
    }

    #[test]
    fn cpu_drop_shadow_basic() {
        // 3x3 white pixel center, shadow offset (1, 0), no blur, black.
        let mut buf = vec![0u8; 3 * 3 * 4];
        let c = 4 * 4;
        buf[c] = 255;
        buf[c + 1] = 255;
        buf[c + 2] = 255;
        buf[c + 3] = 255;
        let mut scratch = vec![0u8; buf.len()];
        let mut tmp = vec![0u8; buf.len()];
        apply_drop_shadow(
            &mut buf,
            (3, 3),
            &mut scratch,
            &mut tmp,
            (1.0, 0.0),
            0.0,
            Color::rgba8(0, 0, 0, 255),
        );
        // Source pixel still white.
        assert_eq!(&buf[c..c + 4], &[255, 255, 255, 255]);
        // Shadow pixel one to the right is black.
        let s = 4 * 5;
        assert_eq!(&buf[s..s + 4], &[0, 0, 0, 255]);
        // Everywhere else transparent.
        assert_eq!(&buf[0..4], &[0, 0, 0, 0]);
    }

    #[test]
    fn clip_path_inset_resolves() {
        let c = ctx();
        let r = Rect {
            x: 10.0,
            y: 20.0,
            w: 100.0,
            h: 80.0,
        };
        let cp = ClipPath::Shape(Box::new(BasicShape::Inset {
            offsets: [
                CssValue::Length(5.0, LengthUnit::Px),
                CssValue::Percentage(10.0),
                CssValue::Number(0.0),
                CssValue::Length(5.0, LengthUnit::Px),
            ],
            radii: None,
        }));
        match resolve_clip_path(&cp, &r, &c) {
            Some(ResolvedClip::Rect(rr)) => {
                assert_eq!(rr.x, 15.0);
                assert_eq!(rr.y, 25.0);
                assert_eq!(rr.w, 85.0); // 100 - 5 - 10% of 100
                assert_eq!(rr.h, 75.0);
            }
            other => panic!("expected Rect, got {other:?}"),
        }
    }

    #[test]
    fn clip_path_circle_tessellates() {
        let c = ctx();
        let r = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 100.0,
        };
        let cp = ClipPath::Shape(Box::new(BasicShape::Circle {
            radius: ShapeRadius::FarthestSide,
            cx: CssValue::Percentage(50.0),
            cy: CssValue::Percentage(50.0),
        }));
        match resolve_clip_path(&cp, &r, &c) {
            Some(ResolvedClip::Polygon { points }) => {
                assert_eq!(points.len(), CLIP_CIRCLE_SEGMENTS);
                // Farthest side from the center of a 100x100 box: 50.
                for [x, y] in &points {
                    let d = ((x - 50.0).powi(2) + (y - 50.0).powi(2)).sqrt();
                    assert!((d - 50.0).abs() < 1e-3, "d={d}");
                }
            }
            other => panic!("expected Polygon, got {other:?}"),
        }
        // A non-positive radius is degenerate: no clip.
        let cp = ClipPath::Shape(Box::new(BasicShape::Circle {
            radius: ShapeRadius::Length(CssValue::Length(0.0, LengthUnit::Px)),
            cx: CssValue::Percentage(50.0),
            cy: CssValue::Percentage(50.0),
        }));
        assert!(resolve_clip_path(&cp, &r, &c).is_none());
    }

    #[test]
    fn clip_path_polygon_passthrough() {
        let c = ctx();
        let r = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 100.0,
        };
        let cp = ClipPath::Shape(Box::new(BasicShape::Polygon {
            points: vec![
                (CssValue::Percentage(0.0), CssValue::Percentage(0.0)),
                (CssValue::Percentage(100.0), CssValue::Percentage(0.0)),
                (CssValue::Percentage(50.0), CssValue::Percentage(100.0)),
            ],
        }));
        match resolve_clip_path(&cp, &r, &c) {
            Some(ResolvedClip::Polygon { points }) => {
                assert_eq!(points, vec![[0.0, 0.0], [200.0, 0.0], [100.0, 100.0]]);
            }
            other => panic!("expected Polygon, got {other:?}"),
        }
        // Fewer than 3 points: no clip.
        let cp = ClipPath::Shape(Box::new(BasicShape::Polygon {
            points: vec![(CssValue::Percentage(0.0), CssValue::Percentage(0.0))],
        }));
        assert!(resolve_clip_path(&cp, &r, &c).is_none());
    }

    #[test]
    fn filter_spill_bounds_blur() {
        // blur(10px): σ = 5, box half-width 5, support 3·(5+1) = 18.
        let ops = vec![FilterOp::Blur(10.0)];
        assert!((filter_spill(&ops) - 18.0).abs() < 1e-5);
        let ops = vec![FilterOp::Brightness(2.0)];
        assert_eq!(filter_spill(&ops), 0.0);
        // drop-shadow: max(|3|, |-4|) = 4 offset + 3·(3+1) = 12 blur
        // support (σ = 3, box half-width 3).
        let ops = vec![FilterOp::DropShadow {
            offset: (3.0, -4.0),
            blur: 6.0,
            color: Color::rgba8(0, 0, 0, 255),
        }];
        assert!((filter_spill(&ops) - 16.0).abs() < 1e-5);
        // No-op blur still reserves the ±0.5 rounding slop, never
        // negative.
        let ops = vec![FilterOp::Blur(0.0)];
        assert!((filter_spill(&ops) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn filter_memo_key_covers_length_ctx_and_currentcolor() {
        let black = Color::rgba8(0, 0, 0, 255);
        let red = Color::rgba8(255, 0, 0, 255);
        let f = factor_fn("blur", "1em");
        let c16 = ctx();
        let mut c32 = ctx();
        c32.em = 32.0;
        // Same text, different font size: different keys (F7).
        assert_ne!(
            filter_memo_key(&f, &c16, black),
            filter_memo_key(&f, &c32, black)
        );
        // Same text, different root/viewport: different keys.
        let mut c_root = ctx();
        c_root.root = 20.0;
        assert_ne!(
            filter_memo_key(&f, &c16, black),
            filter_memo_key(&f, &c_root, black)
        );
        let mut c_vp = ctx();
        c_vp.viewport = (1024.0, 768.0);
        assert_ne!(
            filter_memo_key(&f, &c16, black),
            filter_memo_key(&f, &c_vp, black)
        );
        // Same text, different currentcolor: different keys (F7).
        let ds = factor_fn("drop-shadow", "2px 2px");
        assert_ne!(
            filter_memo_key(&ds, &c16, black),
            filter_memo_key(&ds, &c16, red)
        );
        // Identical inputs: identical keys (the memo still hits).
        assert_eq!(
            filter_memo_key(&f, &c16, black),
            filter_memo_key(&f, &c16, black)
        );
        // `none` never memoizes.
        assert_eq!(filter_memo_key(&Filter::None, &c16, black), None);
    }

    #[test]
    fn clip_path_oversized_polygon_decimates() {
        let c = ctx();
        let r = Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 100.0,
        };
        // Regular 100-gon, px leaves, centered (100, 50) radius 40.
        let authored: Vec<(f32, f32)> = (0..100)
            .map(|i| {
                let a = i as f32 / 100.0 * 2.0 * PI;
                (100.0 + 40.0 * a.cos(), 50.0 + 40.0 * a.sin())
            })
            .collect();
        let cp = ClipPath::Shape(Box::new(BasicShape::Polygon {
            points: authored
                .iter()
                .map(|(x, y)| (CssValue::Number(*x), CssValue::Number(*y)))
                .collect(),
        }));
        match resolve_clip_path(&cp, &r, &c) {
            Some(ResolvedClip::Polygon { points }) => {
                // Decimated to the backend cap, not truncated to a
                // wedge: vertex 0 kept, vertex 50 (the far side) still
                // present at index 24, last vertex near the end.
                assert_eq!(points.len(), CLIP_CIRCLE_SEGMENTS);
                let close = |p: [f32; 2], q: (f32, f32)| {
                    (p[0] - q.0).abs() < 1e-3 && (p[1] - q.1).abs() < 1e-3
                };
                assert!(close(points[0], authored[0]), "first vertex kept");
                assert!(
                    close(points[24], authored[50]),
                    "far side survives decimation, got {:?}",
                    points[24]
                );
                assert!(
                    close(points[47], authored[97]),
                    "tail reaches the end, got {:?}",
                    points[47]
                );
                // No vertex repeats (strictly increasing source
                // indices for n > 48).
                for w in points.windows(2) {
                    assert_ne!(w[0], w[1]);
                }
            }
            other => panic!("expected Polygon, got {other:?}"),
        }
        // At exactly the cap, the polygon still passes through
        // untouched.
        let cp = ClipPath::Shape(Box::new(BasicShape::Polygon {
            points: (0..CLIP_CIRCLE_SEGMENTS)
                .map(|i| (CssValue::Number(i as f32), CssValue::Number(0.0)))
                .collect(),
        }));
        match resolve_clip_path(&cp, &r, &c) {
            Some(ResolvedClip::Polygon { points }) => {
                assert_eq!(points.len(), CLIP_CIRCLE_SEGMENTS);
                assert_eq!(points[47], [47.0, 0.0]);
            }
            other => panic!("expected Polygon, got {other:?}"),
        }
    }

    #[test]
    fn split_number_unit_handles_exponent_like_units() {
        // `1em` is `(1.0, "em")`, not the unparseable `("1e", "m")`.
        assert_eq!(split_number_unit("1em"), Some((1.0, "em")));
        assert_eq!(split_number_unit("2.5rem"), Some((2.5, "rem")));
        // Real exponents still parse.
        assert_eq!(split_number_unit("1e3px"), Some((1000.0, "px")));
        assert_eq!(split_number_unit("1E-2px"), Some((0.01, "px")));
        assert_eq!(split_number_unit("2e+2"), Some((200.0, "")));
        // `e` with no exponent digits starts the unit.
        assert_eq!(split_number_unit("1e"), Some((1.0, "e")));
        assert_eq!(split_number_unit("5px"), Some((5.0, "px")));
        assert_eq!(split_number_unit("0"), Some((0.0, "")));
        assert_eq!(split_number_unit(""), None);
        // Relative units now resolve: `blur(1em)` at 16px → 16px.
        let c = ctx();
        let black = Color::rgba8(0, 0, 0, 255);
        let ops = resolve_filter_chain(&factor_fn("blur", "1em"), &c, black);
        assert_eq!(ops, vec![FilterOp::Blur(16.0)]);
    }
}
