//! Border, text-decoration, and list-marker tessellation.
//!
//! Converts CSS borders, text decorations, and list markers into the
//! axis-aligned colored [`Quad`]s consumed by the paint layer. All math is
//! `f32` in device-independent pixels; the only allocation is the returned
//! [`Vec`]. The border tessellators also come in `_into` variants that
//! clear and fill a caller-supplied buffer, so hot paint loops can pool
//! one buffer instead of allocating per bordered box.
//!
//! Deliberate approximations, each documented where it is made:
//! * `dotted` borders are square dots, not circles.
//! * `double` borders are two solid bands with a transparent gap.
//! * `groove`/`ridge`/`inset`/`outset` borders are a two-tone
//!   lightened/darkened split, not true 3D bevels.
//! * Rounded corners are stepped quarter-annulus strips (`CORNER_STEPS`
//!   per corner); each strip conservatively covers the band, so corners
//!   may overpaint the true arc by at most one strip height.
//! * `dashed`/`dotted` corners are not tessellated along the curve: with
//!   nonzero radii the straight bars are shortened and the corner zones
//!   are left empty for those styles.
//! * Corners use the adjacent horizontal side's style/color and its width
//!   as the band width; when the vertical side's width differs, the join
//!   may notch or overlap slightly.
//! * Border radii are clamped per the CSS "reduce all radii by one factor"
//!   rule, approximated per side-pair instead of with a single global
//!   factor.
//! * `Disc` markers are filled squares, not circles.
//! * `Decimal`/`DecimalLeadingZero` markers are text, not geometry; see
//!   [`marker_text`].
//! * Wavy text decorations are tessellated as sine waves; see
//!   [`decoration_geometry`].

use bowser_layout::Rect;

use crate::display_list::{BorderStyle, Color, DecoKind, MarkerKind, Quad};

/// Side indices into the `[top, right, bottom, left]` parameter arrays.
const TOP: usize = 0;
const RIGHT: usize = 1;
const BOTTOM: usize = 2;
const LEFT: usize = 3;

/// Corner indices into the `[top-left, top-right, bottom-right,
/// bottom-left]` radii array.
const CTL: usize = 0;
const CTR: usize = 1;
const CBR: usize = 2;
const CBL: usize = 3;

/// Horizontal strips used to approximate one rounded corner band.
const CORNER_STEPS: u32 = 8;

/// Upper bound on quads emitted for one patterned border element: the
/// wavy-decoration sample count and the dash/dot tiles along one side.
/// Side length and border width are both attacker-influenced (huge
/// length, tiny positive width), so without a cap the tessellation
/// loops and allocation grow without bound. Past the cap the pattern
/// simply stops mid-side — at 4096 quads the side is already covered
/// far beyond any visible range.
const MAX_BORDER_PATTERN_QUADS: usize = 4096;

/// Tessellate a CSS border into quads.
///
/// `rect` is the border box. `widths`, `styles`, and `colors` are ordered
/// `[top, right, bottom, left]`.
///
/// Layout convention: the top and bottom bars span the full width of `rect`;
/// the left and right bars sit between them. Sides with zero width or with
/// `None`/`Hidden` style produce no quads. Corner overlaps are avoided by
/// construction: adjacent bars never share pixels.
///
/// * `Solid` emits one quad per side.
/// * `Dashed` tiles butt-capped dashes along each side with a 3:2
///   dash:gap ratio scaled by the side width (dash length `3*w`,
///   gap `2*w`); a trailing partial dash is clipped to the side length.
/// * `Dotted` tiles `w`x`w` squares (circle approximation) on a `2*w`
///   period, centered on the side's midline.
/// * `Double` splits each side into outer and inner thirds in the side
///   color with the middle third left transparent (approximation).
/// * `Groove`/`Inset` split each side into outer/inner halves with the
///   outer half darkened and the inner half lightened; `Ridge`/`Outset`
///   do the reverse (two-tone approximation of the 3D bevel).
///
/// For rounded corners see [`tessellate_border_rounded`]; this is the
/// sharp-corner specialization (`radii = [0; 4]`).
///
/// For the pooling variant see [`tessellate_border_into`].
pub fn tessellate_border(
    rect: Rect,
    widths: [f32; 4],
    styles: [BorderStyle; 4],
    colors: [Color; 4],
) -> Vec<Quad> {
    let mut quads = Vec::new();
    tessellate_border_into(&mut quads, rect, widths, styles, colors);
    quads
}

/// Tessellate a CSS border into `out` instead of allocating.
///
/// `out` is cleared first, so the caller may reuse one buffer across
/// calls (the CPU display-item walk pools one per backend). The quads
/// written are bit-identical to [`tessellate_border`]'s return value;
/// see that function for the per-style semantics.
pub fn tessellate_border_into(
    out: &mut Vec<Quad>,
    rect: Rect,
    widths: [f32; 4],
    styles: [BorderStyle; 4],
    colors: [Color; 4],
) {
    tessellate_border_rounded_into(out, rect, widths, styles, colors, [0.0; 4]);
}

/// Tessellate a CSS border with rounded corners into quads.
///
/// `radii` is the outer border radius per corner, ordered
/// `[top-left, top-right, bottom-right, bottom-left]`. Radii are clamped
/// per the CSS rule (adjacent radii sharing a side are scaled down
/// together if their sum exceeds the side length; negative radii become
/// zero).
///
/// Each side's straight bar is shortened so it stops at the corner zones;
/// each corner zone is tessellated as a stepped quarter-annulus band
/// (see [`corner_annulus`]) in the adjacent *horizontal* side's
/// style/color, with that side's width as the band width. `Dashed` and
/// `Dotted` corners are skipped (the straight portions are still dashed
/// or dotted).
///
/// For the pooling variant see [`tessellate_border_rounded_into`].
pub fn tessellate_border_rounded(
    rect: Rect,
    widths: [f32; 4],
    styles: [BorderStyle; 4],
    colors: [Color; 4],
    radii: [f32; 4],
) -> Vec<Quad> {
    let mut quads = Vec::new();
    tessellate_border_rounded_into(&mut quads, rect, widths, styles, colors, radii);
    quads
}

/// Tessellate a CSS border with rounded corners into `out` instead of
/// allocating.
///
/// `out` is cleared first, so the caller may reuse one buffer across
/// calls (the CPU display-item walk pools one per backend). The quads
/// written are bit-identical to [`tessellate_border_rounded`]'s return
/// value; see that function for the radii and per-style semantics.
pub fn tessellate_border_rounded_into(
    out: &mut Vec<Quad>,
    rect: Rect,
    widths: [f32; 4],
    styles: [BorderStyle; 4],
    colors: [Color; 4],
    radii: [f32; 4],
) {
    out.clear();
    let radii = clamp_radii(rect, radii);
    for side in 0..4 {
        let w = widths[side];
        // Finite and positive: NaN or infinite widths would tessellate
        // NaN/infinite geometry below.
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let color = colors[side];
        let bar = bar_rect(rect, side, widths, radii);
        if !bar.w.is_finite() || bar.w <= 0.0 || !bar.h.is_finite() || bar.h <= 0.0 {
            continue;
        }
        match styles[side] {
            BorderStyle::None | BorderStyle::Hidden => {}
            BorderStyle::Solid => {
                out.push(Quad { rect: bar, color });
            }
            BorderStyle::Dashed => tessellate_dashes(out, bar, side, w, color),
            BorderStyle::Dotted => tessellate_dots(out, bar, side, w, color),
            BorderStyle::Double => {
                // Two bands, outer and inner thirds, middle third transparent.
                out.push(Quad {
                    rect: band_of(bar, side, 0.0, 1.0 / 3.0),
                    color,
                });
                out.push(Quad {
                    rect: band_of(bar, side, 2.0 / 3.0, 1.0),
                    color,
                });
            }
            BorderStyle::Groove | BorderStyle::Inset => {
                out.push(Quad {
                    rect: band_of(bar, side, 0.0, 0.5),
                    color: darken(color),
                });
                out.push(Quad {
                    rect: band_of(bar, side, 0.5, 1.0),
                    color: lighten(color),
                });
            }
            BorderStyle::Ridge | BorderStyle::Outset => {
                out.push(Quad {
                    rect: band_of(bar, side, 0.0, 0.5),
                    color: lighten(color),
                });
                out.push(Quad {
                    rect: band_of(bar, side, 0.5, 1.0),
                    color: darken(color),
                });
            }
        }
    }
    tessellate_corners(out, rect, widths, styles, colors, radii);
}

/// Tessellate a list marker into quads.
///
/// `(x, y)` is the marker center and `size` its nominal diameter.
///
/// * `Disc` is a filled `size`x`size` square (circle approximation).
/// * `Circle` is a hollow square outline with line width `size/8`.
/// * `Square` is a filled `size`x`size` square.
/// * `Decimal(_)`/`DecimalLeadingZero(_)` return no quads: numerals are
///   rendered as text by the display-list builder (see [`marker_text`]),
///   so the marker tessellator must not emit them.
pub fn marker_geometry(kind: MarkerKind, x: f32, y: f32, size: f32, color: Color) -> Vec<Quad> {
    if size <= 0.0 {
        return Vec::new();
    }
    let half = size / 2.0;
    let outer = Rect {
        x: x - half,
        y: y - half,
        w: size,
        h: size,
    };
    match kind {
        MarkerKind::Disc | MarkerKind::Square => vec![Quad { rect: outer, color }],
        MarkerKind::Circle => {
            let line = size / 8.0;
            let inner = size - 2.0 * line;
            vec![
                // Top bar.
                Quad {
                    rect: Rect {
                        x: outer.x,
                        y: outer.y,
                        w: size,
                        h: line,
                    },
                    color,
                },
                // Bottom bar.
                Quad {
                    rect: Rect {
                        x: outer.x,
                        y: outer.y + size - line,
                        w: size,
                        h: line,
                    },
                    color,
                },
                // Left bar (between top and bottom).
                Quad {
                    rect: Rect {
                        x: outer.x,
                        y: outer.y + line,
                        w: line,
                        h: inner,
                    },
                    color,
                },
                // Right bar.
                Quad {
                    rect: Rect {
                        x: outer.x + size - line,
                        y: outer.y + line,
                        w: line,
                        h: inner,
                    },
                    color,
                },
            ]
        }
        // Contract: numerals are text, not marker geometry.
        MarkerKind::Decimal(_) | MarkerKind::DecimalLeadingZero(_) => Vec::new(),
    }
}

/// Numeral text for a decimal list marker, if the kind renders as text.
///
/// `Decimal(n)` formats as `n` (`"3"`); `DecimalLeadingZero(n)` pads to at
/// least two digits (`"03"`; `"12"` stays `"12"`). Returns `None` for
/// glyph markers (`Disc`/`Circle`/`Square`), which are geometry, not text.
pub fn marker_text(kind: MarkerKind) -> Option<String> {
    match kind {
        MarkerKind::Decimal(n) => Some(n.to_string()),
        MarkerKind::DecimalLeadingZero(n) => Some(format!("{n:02}")),
        MarkerKind::Disc | MarkerKind::Circle | MarkerKind::Square => None,
    }
}

/// Tessellate a text decoration into quads.
///
/// `Underline`, `Overline`, and `LineThrough` all emit a single rect from
/// `x0` to `x1` at `y` with the given `thickness` in the given `color`;
/// the caller positions `y` for the specific decoration kind. The wavy
/// variants (`WavyUnderline`, `WavyOverline`, `WavyLineThrough`) emit a
/// sine wave centered on the straight line's middle (`y + thickness /
/// 2`), tessellated into axis-aligned quad slabs; see
/// [`wavy_decoration_geometry`].
/// Degenerate inputs (`x1 <= x0`, non-positive thickness) and
/// non-finite inputs return no quads.
pub fn decoration_geometry(
    kind: DecoKind,
    x0: f32,
    x1: f32,
    y: f32,
    thickness: f32,
    color: Color,
) -> Vec<Quad> {
    if !x0.is_finite() || !x1.is_finite() || x1 <= x0 || !thickness.is_finite() || thickness <= 0.0
    {
        return Vec::new();
    }
    match kind {
        DecoKind::Underline | DecoKind::Overline | DecoKind::LineThrough => vec![Quad {
            rect: Rect {
                x: x0,
                y,
                w: x1 - x0,
                h: thickness,
            },
            color,
        }],
        DecoKind::WavyUnderline | DecoKind::WavyOverline | DecoKind::WavyLineThrough => {
            wavy_decoration_geometry(x0, x1, y, thickness, color)
        }
    }
}

/// Tessellate a wavy text decoration (sine wave) into quads.
///
/// The wave is centered on the middle of the straight line's rect
/// (`y + thickness / 2`) with amplitude `thickness` and period
/// `4 * thickness` (our chosen metrics; CSS leaves wavy metrics
/// unspecified), sampled at eight slabs per period starting at phase
/// zero at `x0`. Each sample interval becomes one axis-aligned quad
/// spanning the interval's vertical range plus the stroke thickness, so
/// adjacent slabs share edges and the wave renders without gaps.
///
/// The sample count is capped at [`MAX_BORDER_PATTERN_QUADS`]:
/// `width / thickness` is attacker-influenced (huge width, tiny
/// positive thickness) and previously drove an unbounded
/// allocation/loop. Non-finite inputs yield no quads.
fn wavy_decoration_geometry(x0: f32, x1: f32, y: f32, thickness: f32, color: Color) -> Vec<Quad> {
    use std::f32::consts::TAU;
    if !x0.is_finite() || !x1.is_finite() || !y.is_finite() || !thickness.is_finite() {
        return Vec::new();
    }
    let width = x1 - x0;
    let period = 4.0 * thickness;
    let steps = ((width / (period / 8.0)).ceil() as usize).clamp(2, MAX_BORDER_PATTERN_QUADS);
    let center = y + thickness / 2.0;
    let amp = thickness;
    let dx = width / steps as f32;
    let wave_y = |x: f32| center + amp * ((x - x0) * TAU / period).sin();
    let mut quads = Vec::with_capacity(steps);
    let (mut prev_x, mut prev_y) = (x0, wave_y(x0));
    for i in 1..=steps {
        let x = x0 + i as f32 * dx;
        let wy = wave_y(x);
        let top = prev_y.min(wy) - thickness / 2.0;
        let bottom = prev_y.max(wy) + thickness / 2.0;
        quads.push(Quad {
            rect: Rect {
                x: prev_x,
                y: top,
                w: x - prev_x,
                h: bottom - top,
            },
            color,
        });
        prev_x = x;
        prev_y = wy;
    }
    quads
}

/// Clamp corner radii: negatives become zero, and each pair of corners
/// sharing a side is scaled down together when their sum exceeds the
/// side length (per-side-pair approximation of the CSS single-factor
/// reduction rule).
/// CSS corner-radius clamping: radii are non-negative, and when two
/// adjacent radii sum past the side between them they scale down
/// proportionally. Shared by the border tessellator and the rounded
/// clip (both clip to the border box's outer curve, so they must
/// agree).
pub(crate) fn clamp_radii(rect: Rect, radii: [f32; 4]) -> [f32; 4] {
    let mut r = radii;
    for v in &mut r {
        *v = v.max(0.0);
    }
    for (a, b, len) in [
        (CTL, CTR, rect.w),
        (CBR, CBL, rect.w),
        (CTR, CBR, rect.h),
        (CBL, CTL, rect.h),
    ] {
        let sum = r[a] + r[b];
        if sum > len && sum > 0.0 {
            let f = len / sum;
            r[a] *= f;
            r[b] *= f;
        }
    }
    r
}

/// The straight bar occupied by one side of the border, shortened so it
/// stops at the corner zones.
///
/// With zero radii this is the classic layout: top and bottom bars span
/// the full width; left and right bars sit between them so corners are
/// not double-covered. With nonzero radii the left/right bars start and
/// end at the corner zones instead; when a radius is smaller than the
/// adjacent bar height the bars may overlap slightly in the corner
/// region (documented approximation; the later-emitted side wins).
fn bar_rect(rect: Rect, side: usize, widths: [f32; 4], radii: [f32; 4]) -> Rect {
    let (wt, wr, wb, wl) = (widths[TOP], widths[RIGHT], widths[BOTTOM], widths[LEFT]);
    // Vertical insets for the left/right bars: clear the corner zones when
    // rounded, otherwise sit between the top/bottom bars.
    let it_l = if radii[CTL] > 0.0 { radii[CTL] } else { wt };
    let ib_l = if radii[CBL] > 0.0 { radii[CBL] } else { wb };
    let it_r = if radii[CTR] > 0.0 { radii[CTR] } else { wt };
    let ib_r = if radii[CBR] > 0.0 { radii[CBR] } else { wb };
    match side {
        TOP => Rect {
            x: rect.x + radii[CTL],
            y: rect.y,
            w: (rect.w - radii[CTL] - radii[CTR]).max(0.0),
            h: wt,
        },
        BOTTOM => Rect {
            x: rect.x + radii[CBL],
            y: rect.y + rect.h - wb,
            w: (rect.w - radii[CBL] - radii[CBR]).max(0.0),
            h: wb,
        },
        LEFT => Rect {
            x: rect.x,
            y: rect.y + it_l,
            w: wl,
            h: (rect.h - it_l - ib_l).max(0.0),
        },
        _ => Rect {
            x: rect.x + rect.w - wr,
            y: rect.y + it_r,
            w: wr,
            h: (rect.h - it_r - ib_r).max(0.0),
        },
    }
}

/// A sub-band of one side's bar covering the thickness fraction
/// `[outer_frac, inner_frac)`, where `0.0` is the outermost edge.
///
/// For top/left bars the outer edge is the low coordinate; for
/// bottom/right bars it is the high coordinate.
fn band_of(bar: Rect, side: usize, outer_frac: f32, inner_frac: f32) -> Rect {
    let span = inner_frac - outer_frac;
    match side {
        TOP | LEFT => {
            if side == TOP {
                Rect {
                    x: bar.x,
                    y: bar.y + bar.h * outer_frac,
                    w: bar.w,
                    h: bar.h * span,
                }
            } else {
                Rect {
                    x: bar.x + bar.w * outer_frac,
                    y: bar.y,
                    w: bar.w * span,
                    h: bar.h,
                }
            }
        }
        _ => {
            if side == BOTTOM {
                Rect {
                    x: bar.x,
                    y: bar.y + bar.h * (1.0 - inner_frac),
                    w: bar.w,
                    h: bar.h * span,
                }
            } else {
                Rect {
                    x: bar.x + bar.w * (1.0 - inner_frac),
                    y: bar.y,
                    w: bar.w * span,
                    h: bar.h,
                }
            }
        }
    }
}

/// Tessellate the four rounded corner zones.
///
/// Each corner is rendered in the adjacent horizontal side's style and
/// color with that side's width as the band width. `Dashed`/`Dotted`
/// corners are skipped (documented approximation).
fn tessellate_corners(
    quads: &mut Vec<Quad>,
    rect: Rect,
    widths: [f32; 4],
    styles: [BorderStyle; 4],
    colors: [Color; 4],
    radii: [f32; 4],
) {
    // (corner, horizontal side, x sign, y sign). The corner occupies
    // x = cx - sx*u, y = cy - sy*v for u, v >= 0.
    const CORNERS: [(usize, usize, f32, f32); 4] = [
        (CTL, TOP, 1.0, 1.0),
        (CTR, TOP, -1.0, 1.0),
        (CBR, BOTTOM, -1.0, -1.0),
        (CBL, BOTTOM, 1.0, -1.0),
    ];
    for (corner, hside, sx, sy) in CORNERS {
        let r = radii[corner];
        if r <= 0.0 {
            continue;
        }
        let w = widths[hside];
        if w <= 0.0 {
            continue;
        }
        let color = colors[hside];
        // The corner zone is `r`x`r` at the rect corner, so the arc center
        // sits `r` in from both edges.
        let (cx, cy) = match corner {
            CTL => (rect.x + r, rect.y + r),
            CTR => (rect.x + rect.w - r, rect.y + r),
            CBR => (rect.x + rect.w - r, rect.y + rect.h - r),
            _ => (rect.x + r, rect.y + rect.h - r),
        };
        // Band fractions (outer_frac, inner_frac) of the side width,
        // mirroring the straight-side band splits.
        let bands: &[(f32, f32, Color)] = match styles[hside] {
            BorderStyle::None | BorderStyle::Hidden => continue,
            // Dashed/dotted corners are not tessellated along the curve.
            BorderStyle::Dashed | BorderStyle::Dotted => continue,
            BorderStyle::Solid => &[(0.0, 1.0, color)],
            BorderStyle::Double => &[(0.0, 1.0 / 3.0, color), (2.0 / 3.0, 1.0, color)],
            BorderStyle::Groove | BorderStyle::Inset => {
                &[(0.0, 0.5, darken(color)), (0.5, 1.0, lighten(color))]
            }
            BorderStyle::Ridge | BorderStyle::Outset => {
                &[(0.0, 0.5, lighten(color)), (0.5, 1.0, darken(color))]
            }
        };
        for &(outer_frac, inner_frac, band_color) in bands {
            corner_annulus(
                quads,
                CornerArc {
                    cx,
                    cy,
                    sx,
                    sy,
                    r_outer: r - w * outer_frac,
                    r_inner: r - w * inner_frac,
                    color: band_color,
                },
            );
        }
    }
}

/// A rounded corner's band: the quarter-annulus between radii `r_outer`
/// and `r_inner` around `(cx, cy)` in one quadrant.
///
/// `(sx, sy)` orient the quadrant: the corner occupies
/// `x = cx - sx*u`, `y = cy - sy*v` for `u, v >= 0`, so the band is the
/// set of points with `r_inner <= sqrt(u^2 + v^2) <= r_outer`.
struct CornerArc {
    cx: f32,
    cy: f32,
    sx: f32,
    sy: f32,
    r_outer: f32,
    r_inner: f32,
    color: Color,
}

/// Emit stepped horizontal strips covering the corner arc's band.
///
/// `CORNER_STEPS` strips; each strip's rect conservatively covers the
/// band over its `v` range (a superset of the true arc region), so
/// overpaint is bounded by one strip height. Strips with an empty
/// x-range (near the arc's tangent point) are skipped.
fn corner_annulus(quads: &mut Vec<Quad>, arc: CornerArc) {
    let CornerArc {
        cx,
        cy,
        sx,
        sy,
        r_outer,
        r_inner,
        color,
    } = arc;
    if r_outer <= 0.0 || r_inner >= r_outer {
        return;
    }
    let r_inner = r_inner.max(0.0);
    let h = r_outer / CORNER_STEPS as f32;
    let mut v0 = 0.0f32;
    for _ in 0..CORNER_STEPS {
        let v1 = (v0 + h).min(r_outer);
        // u_outer falls as v grows: the strip's widest reach is at v0.
        let u_outer = (r_outer * r_outer - v0 * v0).max(0.0).sqrt();
        // u_inner falls as v grows: the strip's innermost reach is at v1.
        let u_inner = (r_inner * r_inner - v1 * v1).max(0.0).sqrt();
        if u_inner < u_outer {
            let (x0, x1) = if sx > 0.0 {
                (cx - u_outer, cx - u_inner)
            } else {
                (cx + u_inner, cx + u_outer)
            };
            let (y0, y1) = if sy > 0.0 {
                (cy - v1, cy - v0)
            } else {
                (cy + v0, cy + v1)
            };
            quads.push(Quad {
                rect: Rect {
                    x: x0,
                    y: y0,
                    w: x1 - x0,
                    h: y1 - y0,
                },
                color,
            });
        }
        v0 = v1;
    }
}

/// Tile butt-capped dashes along one side's bar.
///
/// Dash length is `3*w`, gap is `2*w` (3:2 ratio scaled by the side width);
/// a trailing partial dash is clipped to the bar. The dash count is
/// capped at [`MAX_BORDER_PATTERN_QUADS`]: iteration count grows as
/// side length / width, both attacker-influenced. Non-finite inputs
/// emit nothing.
fn tessellate_dashes(quads: &mut Vec<Quad>, bar: Rect, side: usize, w: f32, color: Color) {
    if !w.is_finite() || w <= 0.0 || !bar.w.is_finite() || !bar.h.is_finite() {
        return;
    }
    let dash = 3.0 * w;
    let period = 5.0 * w;
    let horizontal = side == TOP || side == BOTTOM;
    let (start, len) = if horizontal {
        (bar.x, bar.w)
    } else {
        (bar.y, bar.h)
    };
    let mut pos = start;
    let mut emitted = 0usize;
    // The cap also bounds float-stall: `pos += period` can stop
    // advancing when `period` is tiny next to `pos`, which would
    // otherwise spin forever.
    while pos < start + len && emitted < MAX_BORDER_PATTERN_QUADS {
        let end = (pos + dash).min(start + len);
        let dash_rect = if horizontal {
            Rect {
                x: pos,
                y: bar.y,
                w: end - pos,
                h: bar.h,
            }
        } else {
            Rect {
                x: bar.x,
                y: pos,
                w: bar.w,
                h: end - pos,
            }
        };
        quads.push(Quad {
            rect: dash_rect,
            color,
        });
        emitted += 1;
        pos += period;
    }
}

/// Tile square dots along one side's bar.
///
/// Each dot is a `w`x`w` square (circle approximation) on a `2*w` period,
/// centered on the bar's midline; a trailing partial dot is clipped.
/// The dot count is capped at [`MAX_BORDER_PATTERN_QUADS`] (see
/// [`tessellate_dashes`]); non-finite inputs emit nothing.
fn tessellate_dots(quads: &mut Vec<Quad>, bar: Rect, side: usize, w: f32, color: Color) {
    if !w.is_finite() || w <= 0.0 || !bar.w.is_finite() || !bar.h.is_finite() {
        return;
    }
    let period = 2.0 * w;
    let horizontal = side == TOP || side == BOTTOM;
    let (start, len, cross_start, cross_len) = if horizontal {
        (bar.x, bar.w, bar.y, bar.h)
    } else {
        (bar.y, bar.h, bar.x, bar.w)
    };
    let cross_off = cross_start + (cross_len - w).max(0.0) / 2.0;
    let mut pos = start;
    let mut emitted = 0usize;
    while pos < start + len && emitted < MAX_BORDER_PATTERN_QUADS {
        let end = (pos + w).min(start + len);
        let dot_rect = if horizontal {
            Rect {
                x: pos,
                y: cross_off,
                w: end - pos,
                h: w.min(cross_len),
            }
        } else {
            Rect {
                x: cross_off,
                y: pos,
                w: w.min(cross_len),
                h: end - pos,
            }
        };
        quads.push(Quad {
            rect: dot_rect,
            color,
        });
        emitted += 1;
        pos += period;
    }
}

/// Move a color halfway toward white, preserving alpha.
fn lighten(c: Color) -> Color {
    fn up(v: f32) -> f32 {
        (v + (1.0 - v) * 0.5).clamp(0.0, 1.0)
    }
    Color::from_f32(up(c.r()), up(c.g()), up(c.b()), c.a())
}

/// Scale a color's intensity down, preserving alpha.
fn darken(c: Color) -> Color {
    fn down(v: f32) -> f32 {
        (v * 0.6).clamp(0.0, 1.0)
    }
    Color::from_f32(down(c.r()), down(c.g()), down(c.b()), c.a())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn red() -> Color {
        Color::rgba8(255, 0, 0, 255)
    }

    fn solid_styles() -> [BorderStyle; 4] {
        [BorderStyle::Solid; 4]
    }

    /// One border-tessellation input for the `_into` equivalence
    /// tests: rect, per-side widths, per-side styles, per-corner radii.
    type BorderCase = (Rect, [f32; 4], [BorderStyle; 4], [f32; 4]);

    #[test]
    fn solid_border_quad_count_and_coverage() {
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 50.0,
        };
        let quads = tessellate_border(rect, [2.0; 4], solid_styles(), [red(); 4]);
        assert_eq!(quads.len(), 4);
        // Top bar spans the full width.
        assert!(quads.iter().any(|q| q.rect
            == Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 2.0
            }));
        // Bottom bar spans the full width.
        assert!(quads.iter().any(|q| q.rect
            == Rect {
                x: 0.0,
                y: 48.0,
                w: 100.0,
                h: 2.0
            }));
        // Left and right bars sit between top and bottom (no corner overlap).
        assert!(quads.iter().any(|q| q.rect
            == Rect {
                x: 0.0,
                y: 2.0,
                w: 2.0,
                h: 46.0
            }));
        assert!(quads.iter().any(|q| q.rect
            == Rect {
                x: 98.0,
                y: 2.0,
                w: 2.0,
                h: 46.0
            }));
    }

    #[test]
    fn solid_border_unequal_widths() {
        let rect = Rect {
            x: 10.0,
            y: 10.0,
            w: 100.0,
            h: 50.0,
        };
        let quads = tessellate_border(rect, [4.0, 1.0, 8.0, 3.0], solid_styles(), [red(); 4]);
        assert_eq!(quads.len(), 4);
        assert!(quads.iter().any(|q| q.rect
            == Rect {
                x: 10.0,
                y: 10.0,
                w: 100.0,
                h: 4.0
            }));
        assert!(quads.iter().any(|q| q.rect
            == Rect {
                x: 10.0,
                y: 52.0,
                w: 100.0,
                h: 8.0
            }));
        assert!(quads.iter().any(|q| q.rect
            == Rect {
                x: 10.0,
                y: 14.0,
                w: 3.0,
                h: 38.0
            }));
        assert!(quads.iter().any(|q| q.rect
            == Rect {
                x: 109.0,
                y: 14.0,
                w: 1.0,
                h: 38.0
            }));
    }

    #[test]
    fn dashed_pattern_periodicity() {
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 10.0,
        };
        let styles = [
            BorderStyle::Dashed,
            BorderStyle::None,
            BorderStyle::None,
            BorderStyle::None,
        ];
        let quads = tessellate_border(rect, [2.0, 0.0, 0.0, 0.0], styles, [red(); 4]);
        // Width 2 -> dash 6, gap 4, period 10 over length 100: 10 dashes,
        // the last clipped to 96..100.
        assert_eq!(quads.len(), 10);
        for (i, q) in quads.iter().enumerate() {
            let x0 = i as f32 * 10.0;
            assert_eq!(q.rect.x, x0);
            assert_eq!(q.rect.w, 6.0f32.min(100.0 - x0));
            assert_eq!(q.rect.y, 0.0);
            assert_eq!(q.rect.h, 2.0);
        }
    }

    #[test]
    fn dashed_vertical_side() {
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 55.0,
        };
        let styles = [
            BorderStyle::None,
            BorderStyle::Dashed,
            BorderStyle::None,
            BorderStyle::None,
        ];
        // Right side, width 5: bar is x=5..10, y=0..55; dash 15, gap 10.
        let quads = tessellate_border(rect, [0.0, 5.0, 0.0, 0.0], styles, [red(); 4]);
        // Dashes at 0..15, 25..40, 50..55 (clipped): 3 dashes.
        assert_eq!(quads.len(), 3);
        assert_eq!(quads[0].rect.y, 0.0);
        assert_eq!(quads[0].rect.h, 15.0);
        assert_eq!(quads[2].rect.y, 50.0);
        assert_eq!(quads[2].rect.h, 5.0);
        assert!(quads.iter().all(|q| q.rect.x == 5.0 && q.rect.w == 5.0));
    }

    #[test]
    fn dotted_pattern() {
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 10.0,
        };
        let styles = [
            BorderStyle::Dotted,
            BorderStyle::None,
            BorderStyle::None,
            BorderStyle::None,
        ];
        let quads = tessellate_border(rect, [2.0, 0.0, 0.0, 0.0], styles, [red(); 4]);
        // Width 2 -> 2x2 dots on a 4 period over 100: 25 dots.
        assert_eq!(quads.len(), 25);
        for (i, q) in quads.iter().enumerate() {
            assert_eq!(q.rect.x, i as f32 * 4.0);
            assert_eq!(q.rect.w, 2.0);
            assert_eq!(q.rect.h, 2.0);
        }
    }

    #[test]
    fn zero_width_and_none_styles_produce_no_quads() {
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 50.0,
        };
        assert!(tessellate_border(rect, [0.0; 4], solid_styles(), [red(); 4]).is_empty());
        assert!(tessellate_border(rect, [2.0; 4], [BorderStyle::None; 4], [red(); 4]).is_empty());
        assert!(tessellate_border(rect, [2.0; 4], [BorderStyle::Hidden; 4], [red(); 4]).is_empty());
    }

    #[test]
    fn double_border_two_bands_per_side() {
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 90.0,
            h: 60.0,
        };
        let quads = tessellate_border(rect, [6.0; 4], [BorderStyle::Double; 4], [red(); 4]);
        assert_eq!(quads.len(), 8);
        // Top side: outer third y=0..2, inner third y=4..6 (approx: thirds
        // are not exactly representable in f32).
        for e in [
            Rect {
                x: 0.0,
                y: 0.0,
                w: 90.0,
                h: 2.0,
            },
            Rect {
                x: 0.0,
                y: 4.0,
                w: 90.0,
                h: 2.0,
            },
        ] {
            assert!(
                quads.iter().any(|q| {
                    (q.rect.x - e.x).abs() < 1e-4
                        && (q.rect.y - e.y).abs() < 1e-4
                        && (q.rect.w - e.w).abs() < 1e-4
                        && (q.rect.h - e.h).abs() < 1e-4
                }),
                "missing band near {e:?}"
            );
        }
    }

    #[test]
    fn groove_and_ridge_two_tone_split() {
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 50.0,
        };
        let groove = tessellate_border(rect, [4.0; 4], [BorderStyle::Groove; 4], [red(); 4]);
        let ridge = tessellate_border(rect, [4.0; 4], [BorderStyle::Ridge; 4], [red(); 4]);
        assert_eq!(groove.len(), 8);
        assert_eq!(ridge.len(), 8);
        // Groove: outer half darkened, inner half lightened; ridge: reverse.
        let top_groove: Vec<&Quad> = groove.iter().filter(|q| q.rect.y < 2.0).collect();
        assert_eq!(top_groove.len(), 1);
        assert!(top_groove[0].color.r() < 1.0);
        let top_ridge: Vec<&Quad> = ridge.iter().filter(|q| q.rect.y < 2.0).collect();
        assert_eq!(top_ridge.len(), 1);
        assert_eq!(top_ridge[0].color.r(), 1.0);
        // Bands tile the side without gaps or overlap: outer 0..2, inner 2..4.
        assert!(groove.iter().any(|q| q.rect
            == Rect {
                x: 0.0,
                y: 0.0,
                w: 100.0,
                h: 2.0
            }));
        assert!(groove.iter().any(|q| q.rect
            == Rect {
                x: 0.0,
                y: 2.0,
                w: 100.0,
                h: 2.0
            }));
    }

    #[test]
    fn rounded_zero_radii_matches_plain() {
        let rect = Rect {
            x: 5.0,
            y: 7.0,
            w: 120.0,
            h: 80.0,
        };
        let styles = [
            BorderStyle::Double,
            BorderStyle::Dashed,
            BorderStyle::Dotted,
            BorderStyle::Groove,
        ];
        let a = tessellate_border(rect, [6.0, 3.0, 4.0, 5.0], styles, [red(); 4]);
        let b = tessellate_border_rounded(rect, [6.0, 3.0, 4.0, 5.0], styles, [red(); 4], [0.0; 4]);
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn rounded_rect_sane_geometry() {
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 60.0,
        };
        let r = 10.0;
        let quads = tessellate_border_rounded(rect, [4.0; 4], solid_styles(), [red(); 4], [r; 4]);
        assert!(!quads.is_empty());
        // Every quad stays inside the border box.
        for q in &quads {
            assert!(q.rect.x >= rect.x - 1e-4, "quad left of box: {q:?}");
            assert!(q.rect.y >= rect.y - 1e-4, "quad above box: {q:?}");
            assert!(
                q.rect.x + q.rect.w <= rect.x + rect.w + 1e-4,
                "quad right of box: {q:?}"
            );
            assert!(
                q.rect.y + q.rect.h <= rect.y + rect.h + 1e-4,
                "quad below box: {q:?}"
            );
            assert!(q.rect.w > 0.0 && q.rect.h > 0.0, "degenerate quad: {q:?}");
        }
        // Straight bars are shortened to the corner zones.
        assert!(quads.iter().any(|q| q.rect
            == Rect {
                x: 10.0,
                y: 0.0,
                w: 80.0,
                h: 4.0
            }));
        assert!(quads.iter().any(|q| q.rect
            == Rect {
                x: 0.0,
                y: 10.0,
                w: 4.0,
                h: 40.0
            }));
        // Corner quads exist in each corner zone...
        let in_tl = quads
            .iter()
            .filter(|q| q.rect.x < 10.0 && q.rect.y < 10.0)
            .count();
        let in_tr = quads
            .iter()
            .filter(|q| q.rect.x + q.rect.w > 90.0 && q.rect.y < 10.0)
            .count();
        let in_bl = quads
            .iter()
            .filter(|q| q.rect.x < 10.0 && q.rect.y + q.rect.h > 50.0)
            .count();
        let in_br = quads
            .iter()
            .filter(|q| q.rect.x + q.rect.w > 90.0 && q.rect.y + q.rect.h > 50.0)
            .count();
        assert!(in_tl > 0 && in_tr > 0 && in_bl > 0 && in_br > 0);
        assert_eq!(in_tl, in_tr);
        assert_eq!(in_tl, in_bl);
        assert_eq!(in_tl, in_br);
        // ...and stay clear of the clipped corner tip: no quad may cover
        // the rect corner point itself (0,0), which the radius cuts off.
        for q in &quads {
            let covers_tip = q.rect.x <= 0.0 && q.rect.y <= 0.0 && q.rect.x + q.rect.w > 0.0;
            assert!(!covers_tip, "quad covers clipped corner tip: {q:?}");
        }
    }

    #[test]
    fn rounded_corners_follow_arc() {
        // Corner strips hug the arc: at the top edge the band meets the
        // top bar near the arc center (x=20); descending, its outer edge
        // sweeps left to the rect's left edge where it meets the left bar.
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 100.0,
        };
        let quads = tessellate_border_rounded(
            rect,
            [4.0, 0.0, 0.0, 0.0],
            solid_styles(),
            [red(); 4],
            [20.0, 0.0, 0.0, 0.0],
        );
        let mut strips: Vec<&Quad> = quads
            .iter()
            .filter(|q| q.rect.y < 20.0 && q.rect.x < 20.0)
            .collect();
        strips.sort_by(|a, b| a.rect.y.partial_cmp(&b.rect.y).unwrap());
        assert!(strips.len() >= 4, "expected corner strips, got {strips:?}");
        // Left edge moves left (away from the center x=20) going down.
        for w in strips.windows(2) {
            assert!(
                w[1].rect.x <= w[0].rect.x + 1e-4,
                "corner not monotonic: {w:?}"
            );
        }
        // The strips never cross the arc center line x=20; the topmost
        // strip starts near the center and the bottommost at the left edge.
        assert!(strips.iter().all(|q| q.rect.x + q.rect.w <= 20.0 + 1e-4));
        assert!(strips[0].rect.x >= 20.0 - 10.0);
        assert!(strips.last().unwrap().rect.x <= 0.0 + 1e-4);
    }

    #[test]
    fn rounded_radii_clamped_to_sides() {
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 40.0,
        };
        // Adjacent radii summing past the side length are scaled down.
        let a = tessellate_border_rounded(
            rect,
            [4.0; 4],
            solid_styles(),
            [red(); 4],
            [60.0, 60.0, 0.0, 0.0],
        );
        let b = tessellate_border_rounded(
            rect,
            [4.0; 4],
            solid_styles(),
            [red(); 4],
            [80.0, 80.0, 0.0, 0.0],
        );
        assert_eq!(a, b);
        // Negative radii behave as zero.
        let c = tessellate_border_rounded(rect, [4.0; 4], solid_styles(), [red(); 4], [-5.0; 4]);
        assert_eq!(
            c,
            tessellate_border(rect, [4.0; 4], solid_styles(), [red(); 4])
        );
    }

    #[test]
    fn rounded_double_and_groove_corners() {
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 100.0,
        };
        // Double: 2 bands per corner; the corner annuli leave the middle
        // third empty, so the TL zone holds strictly more, smaller quads
        // than the solid case.
        let double = tessellate_border_rounded(
            rect,
            [9.0; 4],
            [BorderStyle::Double; 4],
            [red(); 4],
            [18.0; 4],
        );
        let solid =
            tessellate_border_rounded(rect, [9.0; 4], solid_styles(), [red(); 4], [18.0; 4]);
        let tl_double = double
            .iter()
            .filter(|q| q.rect.x + q.rect.w <= 18.0 + 1e-4 && q.rect.y + q.rect.h <= 18.0 + 1e-4)
            .count();
        let tl_solid = solid
            .iter()
            .filter(|q| q.rect.x + q.rect.w <= 18.0 + 1e-4 && q.rect.y + q.rect.h <= 18.0 + 1e-4)
            .count();
        assert!(tl_double > 0 && tl_solid > 0);
        assert!(tl_double > tl_solid);
        // Groove corners are two-tone: both a darkened and a lightened
        // quad appear in the TL zone.
        let groove = tessellate_border_rounded(
            rect,
            [8.0; 4],
            [BorderStyle::Groove; 4],
            [red(); 4],
            [16.0; 4],
        );
        let tl_groove: Vec<&Quad> = groove
            .iter()
            .filter(|q| q.rect.x + q.rect.w <= 16.0 + 1e-4 && q.rect.y + q.rect.h <= 16.0 + 1e-4)
            .collect();
        assert!(tl_groove.iter().any(|q| q.color.r() < 1.0));
        assert!(tl_groove.iter().any(|q| (q.color.r() - 1.0).abs() < 1e-6));
    }

    #[test]
    fn rounded_dashed_corners_skipped() {
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 60.0,
        };
        let quads = tessellate_border_rounded(
            rect,
            [4.0; 4],
            [BorderStyle::Dashed; 4],
            [red(); 4],
            [10.0; 4],
        );
        // Straight bars are shortened but still dashed...
        assert!(quads.iter().any(|q| q.rect
            == Rect {
                x: 10.0,
                y: 0.0,
                w: 12.0,
                h: 4.0
            }));
        // ...while the corner zones stay empty (documented approximation).
        assert!(quads.iter().all(|q| !(q.rect.x < 10.0 && q.rect.y < 10.0)));
    }

    #[test]
    fn marker_kinds() {
        // Disc: one filled square centered on (x, y).
        let disc = marker_geometry(MarkerKind::Disc, 10.0, 10.0, 8.0, red());
        assert_eq!(disc.len(), 1);
        assert_eq!(
            disc[0].rect,
            Rect {
                x: 6.0,
                y: 6.0,
                w: 8.0,
                h: 8.0
            }
        );
        // Square: same as disc.
        assert_eq!(
            marker_geometry(MarkerKind::Square, 10.0, 10.0, 8.0, red()).len(),
            1
        );
        // Circle: hollow square outline, four bars.
        let circle = marker_geometry(MarkerKind::Circle, 10.0, 10.0, 8.0, red());
        assert_eq!(circle.len(), 4);
        assert!(circle.iter().any(|q| q.rect
            == Rect {
                x: 6.0,
                y: 6.0,
                w: 8.0,
                h: 1.0
            }));
        // Decimal: numerals are text; no marker quads.
        assert!(marker_geometry(MarkerKind::Decimal(3), 10.0, 10.0, 8.0, red()).is_empty());
        // Decimal-leading-zero: also text; no marker quads.
        assert!(
            marker_geometry(MarkerKind::DecimalLeadingZero(3), 10.0, 10.0, 8.0, red()).is_empty()
        );
        // Degenerate size: no quads.
        assert!(marker_geometry(MarkerKind::Disc, 10.0, 10.0, 0.0, red()).is_empty());
    }

    #[test]
    fn marker_text_formats_numerals() {
        assert_eq!(marker_text(MarkerKind::Decimal(3)), Some("3".to_string()));
        assert_eq!(marker_text(MarkerKind::Decimal(42)), Some("42".to_string()));
        assert_eq!(
            marker_text(MarkerKind::DecimalLeadingZero(3)),
            Some("03".to_string())
        );
        assert_eq!(
            marker_text(MarkerKind::DecimalLeadingZero(12)),
            Some("12".to_string())
        );
        assert_eq!(marker_text(MarkerKind::Disc), None);
        assert_eq!(marker_text(MarkerKind::Circle), None);
        assert_eq!(marker_text(MarkerKind::Square), None);
    }

    #[test]
    fn decoration_straight_kinds() {
        let underline = decoration_geometry(DecoKind::Underline, 5.0, 25.0, 40.0, 2.0, red());
        assert_eq!(underline.len(), 1);
        assert_eq!(
            underline[0].rect,
            Rect {
                x: 5.0,
                y: 40.0,
                w: 20.0,
                h: 2.0
            }
        );
        assert_eq!(
            decoration_geometry(DecoKind::Overline, 5.0, 25.0, 40.0, 2.0, red()).len(),
            1
        );
        assert_eq!(
            decoration_geometry(DecoKind::LineThrough, 5.0, 25.0, 40.0, 2.0, red()).len(),
            1
        );
        // The quad carries the requested color.
        assert_eq!(underline[0].color, red());
        // Degenerate inputs.
        assert!(decoration_geometry(DecoKind::Underline, 25.0, 5.0, 40.0, 2.0, red()).is_empty());
        assert!(decoration_geometry(DecoKind::Underline, 5.0, 25.0, 40.0, 0.0, red()).is_empty());
    }

    #[test]
    fn decoration_wavy_kinds_emit_wave_quads() {
        // thickness 2: period 8, 8 slabs per period.
        for kind in [
            DecoKind::WavyUnderline,
            DecoKind::WavyOverline,
            DecoKind::WavyLineThrough,
        ] {
            let quads = decoration_geometry(kind, 0.0, 32.0, 40.0, 2.0, red());
            assert_eq!(quads.len(), 32, "{kind:?}: 4 periods x 8 slabs");
            // Slabs tile the full span without gaps or overlap.
            assert!((quads[0].rect.x - 0.0).abs() < 1e-6);
            let mut prev_end = 0.0f32;
            for q in &quads {
                assert!((q.rect.x - prev_end).abs() < 1e-5, "{kind:?}: gap/overlap");
                assert_eq!(q.color, red());
                prev_end = q.rect.x + q.rect.w;
            }
            assert!((prev_end - 32.0).abs() < 1e-5);
            // The wave oscillates around the straight line's middle
            // (y + thickness / 2 = 41) with amplitude = thickness.
            let center = 41.0f32;
            let amp = 2.0f32;
            let t = 2.0f32;
            let min_top = quads.iter().map(|q| q.rect.y).fold(f32::INFINITY, f32::min);
            let max_bottom = quads
                .iter()
                .map(|q| q.rect.y + q.rect.h)
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                (min_top - (center - amp - t / 2.0)).abs() < 0.6,
                "{kind:?}: wave top {min_top}"
            );
            assert!(
                (max_bottom - (center + amp + t / 2.0)).abs() < 0.6,
                "{kind:?}: wave bottom {max_bottom}"
            );
            // The first slab starts at phase zero, i.e. on the center line.
            let first = &quads[0].rect;
            assert!(first.y <= center && center <= first.y + first.h);
            // Narrow spans still get at least two slabs.
            let narrow = decoration_geometry(kind, 0.0, 1.0, 40.0, 2.0, red());
            assert!(narrow.len() >= 2, "{kind:?}: narrow span");
            // Degenerate inputs stay empty.
            assert!(decoration_geometry(kind, 5.0, 5.0, 40.0, 2.0, red()).is_empty());
            assert!(decoration_geometry(kind, 0.0, 32.0, 40.0, 0.0, red()).is_empty());
        }
    }

    /// The wavy-decoration sample count is capped: a huge width with a
    /// tiny positive thickness would otherwise allocate and loop
    /// without bound (width / (thickness / 2) samples).
    #[test]
    fn wavy_decoration_sample_count_is_capped() {
        // Uncapped this would request ~2e12 samples.
        let quads = decoration_geometry(DecoKind::WavyUnderline, 0.0, 1e9, 40.0, 0.001, red());
        assert!(!quads.is_empty(), "still tessellates");
        assert!(
            quads.len() <= MAX_BORDER_PATTERN_QUADS,
            "sample count capped, got {}",
            quads.len()
        );
        // Non-finite inputs emit nothing instead of NaN geometry.
        assert!(
            decoration_geometry(DecoKind::WavyUnderline, 0.0, 32.0, 40.0, f32::NAN, red())
                .is_empty()
        );
        assert!(
            decoration_geometry(
                DecoKind::WavyUnderline,
                0.0,
                f32::INFINITY,
                40.0,
                2.0,
                red()
            )
            .is_empty()
        );
        assert!(
            decoration_geometry(DecoKind::WavyUnderline, f32::NAN, 32.0, 40.0, 2.0, red())
                .is_empty()
        );
    }

    /// Dashed/dotted tessellation along one side is capped: iteration
    /// count grows as side length / width, both attacker-influenced
    /// (huge side, tiny positive width).
    #[test]
    fn dashed_dotted_side_quad_count_is_capped() {
        let huge = Rect {
            x: 0.0,
            y: 0.0,
            w: 1e7,
            h: 100.0,
        };
        for style in [BorderStyle::Dashed, BorderStyle::Dotted] {
            // Uncapped the top side alone would emit ~2e8 dashes.
            let quads = tessellate_border(huge, [0.01; 4], [style; 4], [red(); 4]);
            assert!(!quads.is_empty(), "{style:?}: still tessellates");
            assert!(
                quads.len() <= 4 * MAX_BORDER_PATTERN_QUADS,
                "{style:?}: per-side count capped, got {}",
                quads.len()
            );
        }
        // Non-finite widths emit nothing instead of NaN geometry.
        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 50.0,
        };
        assert!(tessellate_border(rect, [f32::NAN; 4], solid_styles(), [red(); 4]).is_empty());
    }

    /// The pooling `_into` tessellators must produce bit-identical
    /// output to the allocating wrappers: the CPU display-item walk
    /// paints from a pooled scratch buffer (`CpuBackend::border_scratch`),
    /// and any divergence there would change rendering.
    #[test]
    fn border_into_matches_fresh_allocation() {
        let cases: Vec<BorderCase> = vec![
            // Sharp corners, mixed styles.
            (
                Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 100.0,
                    h: 50.0,
                },
                [2.0, 3.0, 4.0, 5.0],
                [
                    BorderStyle::Solid,
                    BorderStyle::Dashed,
                    BorderStyle::Dotted,
                    BorderStyle::Double,
                ],
                [0.0; 4],
            ),
            // Rounded: bevel styles plus corner annuli.
            (
                Rect {
                    x: 1.5,
                    y: -2.5,
                    w: 200.0,
                    h: 120.0,
                },
                [6.0; 4],
                [
                    BorderStyle::Groove,
                    BorderStyle::Ridge,
                    BorderStyle::Inset,
                    BorderStyle::Outset,
                ],
                [18.0, 12.0, 24.0, 6.0],
            ),
            // Degenerate: zero widths, None/Hidden, negative radii.
            (
                Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 40.0,
                    h: 40.0,
                },
                [0.0, 2.0, 0.0, 2.0],
                [
                    BorderStyle::None,
                    BorderStyle::Hidden,
                    BorderStyle::Solid,
                    BorderStyle::Dashed,
                ],
                [-5.0, 200.0, 0.0, 3.0],
            ),
            // Non-finite widths: no quads, no NaN geometry.
            (
                Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 100.0,
                    h: 50.0,
                },
                [f32::NAN, f32::INFINITY, 2.0, 2.0],
                solid_styles(),
                [0.0; 4],
            ),
        ];
        let mut pooled: Vec<Quad> = Vec::new();
        for (rect, widths, styles, radii) in &cases {
            let colors = [red(); 4];
            let fresh = tessellate_border(*rect, *widths, *styles, colors);
            // Junk pre-fill: proves `_into` clears, not just extends.
            pooled.push(Quad {
                rect: Rect {
                    x: -1.0,
                    y: -1.0,
                    w: -1.0,
                    h: -1.0,
                },
                color: red(),
            });
            tessellate_border_into(&mut pooled, *rect, *widths, *styles, colors);
            assert_eq!(pooled, fresh, "sharp-corner case {rect:?}");

            let fresh_rounded = tessellate_border_rounded(*rect, *widths, *styles, colors, *radii);
            pooled.clear();
            pooled.push(Quad {
                rect: Rect {
                    x: -1.0,
                    y: -1.0,
                    w: -1.0,
                    h: -1.0,
                },
                color: red(),
            });
            tessellate_border_rounded_into(&mut pooled, *rect, *widths, *styles, colors, *radii);
            assert_eq!(pooled, fresh_rounded, "rounded case {rect:?} {radii:?}");
        }
    }

    /// The pooled `_into` tessellators must not allocate once the
    /// buffer is warm, while the allocating wrappers allocate per
    /// call. Measured with the crate's shared per-thread counting
    /// allocator (see `images::test_alloc`).
    #[test]
    fn border_into_pooling_eliminates_per_call_allocations() {
        use crate::images::test_alloc::ALLOCATED_BYTES;

        let rect = Rect {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 50.0,
        };
        let widths = [2.0, 3.0, 4.0, 5.0];
        let styles = [
            BorderStyle::Solid,
            BorderStyle::Dashed,
            BorderStyle::Dotted,
            BorderStyle::Double,
        ];
        let colors = [red(); 4];

        // Fresh allocation: the returned Vec allocates per call.
        let before = ALLOCATED_BYTES.get();
        let fresh = tessellate_border(rect, widths, styles, colors);
        let after = ALLOCATED_BYTES.get();
        assert!(
            after > before,
            "fresh tessellate_border must allocate ({} bytes)",
            after.saturating_sub(before)
        );
        assert!(!fresh.is_empty());

        // Pooled: warm the buffer once (this call may allocate), then
        // the steady-state call must touch the heap not at all.
        let mut pooled: Vec<Quad> = Vec::new();
        tessellate_border_into(&mut pooled, rect, widths, styles, colors);
        assert_eq!(pooled, fresh);
        let before = ALLOCATED_BYTES.get();
        tessellate_border_into(&mut pooled, rect, widths, styles, colors);
        let after = ALLOCATED_BYTES.get();
        assert_eq!(
            after,
            before,
            "warm pooled tessellation must not allocate ({} bytes)",
            after.saturating_sub(before)
        );
    }
}
