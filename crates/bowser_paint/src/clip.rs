//! Rounded-rect clip geometry shared by the CPU and GPU backends.
//!
//! A clip-stack entry is a bounding box plus four circular corner radii
//! (`[top-left, top-right, bottom-right, bottom-left]`, the same order
//! the [`DisplayItem::Border`](crate::display_list::DisplayItem::Border)
//! item uses). All-zero radii mean a plain rect clip.
//!
//! # Coverage contract
//!
//! [`clip_stack_coverage_byte`] is the single definition of fractional
//! clip semantics, extending the rect-only formula the crate already
//! shares ([`clip_coverage_byte`](crate::cpu::clip_coverage_byte) on
//! the CPU, `clip_coverage` in the WGSL `SHADER_COMMON` prelude):
//!
//! * all-zero radii: the exact rect formula, bit-for-bit
//!   (`(r.min(px+1) - l.max(px)).clamp(0,1)` per axis, product,
//!   round-half-away-from-zero via `floor(v + 0.5)`);
//! * otherwise: the exact rect formula for pixels that do not touch a
//!   rounded corner's square region, and 8x8 subpixel sampling of
//!   rounded-rect membership for pixels that do.
//!
//! The sampling is an approximation, not the exact fractional area
//! under the arc: each of the 64 sub-cells votes by its center, so a
//! pixel's byte can be off by at most (#cells an arc passes through)/64
//! (a dense sweep over corner positions and radii from 0.05px to 30px
//! finds a quarter-circle arc crosses at most 17 of the 64 cells —
//! large radii asymptote to the straight-line maximum of 15 — i.e.
//! worst-case error < 17/64 ≈ 0.27 coverage, typically far less; with
//! several rounded entries stacked the bound applies per entry).
//! This is the documented tightly-bounded approximation; an analytic circle-segment area would
//! need `acos`/`sqrt`, whose precision is not bit-identical between
//! Rust `f32` and WGSL, while the sampling uses only `+ - * /` (by a
//! power of two) and comparisons in the same order on both sides, so
//! the CPU and GPU agree bit-for-bit.
//!
//! Nested clips compose by intersecting the entries: the coverage of a
//! pixel is the coverage of the *geometric intersection* of the stack's
//! shapes. No entry-combination rule is needed (and none is sound in
//! general: the intersection of two rounded rects is not a rounded
//! rect). [`clip_stack_coverage_byte`] implements this exactly, up to
//! the 8x8 sampling grid, for any mix of rect and rounded entries:
//! rects are intersected geometrically first (exact), and corner pixels
//! are subsampled with per-sample membership in *every* entry.

use bowser_layout::Rect;

use crate::cpu::ClipEntry;

/// Corner order for every radii array in this module: top-left,
/// top-right, bottom-right, bottom-left.
pub(crate) const CORNERS: usize = 4;

/// How many clip-stack entries the GPU clip uniform carries. The CPU
/// clip mask evaluates the whole stack (no cap); the GPU collapses
/// deeper stacks' oldest entries, soundly, in
/// [`snapshot_clips`].
///
/// Mirrored as the `entries` array size in `SHADER_COMMON`
/// (`wgpu_be.rs`): keep the two in sync.
pub(crate) const MAX_CLIP_ENTRIES: usize = 8;

/// Polygon vertices per GPU clip entry: `circle()`/`ellipse()` are
/// tessellated to exactly this many vertices by
/// [`crate::filter::resolve_clip_path`], and authored `polygon()`s are
/// capped here. Mirrored as the `poly` array size in `SHADER_COMMON`
/// (`wgpu_be.rs`): keep the two in sync.
pub(crate) const MAX_POLY_VERTS: usize = 48;

/// One clip-stack entry in page px for the GPU lowering: bounding rect
/// plus circular corner radii `[TL, TR, BR, BL]` in page px, plus an
/// optional `clip-path` polygon (`poly[..poly_len]`, even-odd rule;
/// `poly_len == 0` means "not a polygon").
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PageClip {
    pub rect: Rect,
    pub radii: [f32; CORNERS],
    pub poly: [[f32; 2]; MAX_POLY_VERTS],
    pub poly_len: u8,
}

impl Default for PageClip {
    fn default() -> Self {
        Self::rect(Rect {
            x: 0.0,
            y: 0.0,
            w: 0.0,
            h: 0.0,
        })
    }
}

impl PageClip {
    /// Plain rect entry.
    pub(crate) fn rect(rect: Rect) -> Self {
        Self {
            rect,
            radii: [0.0; CORNERS],
            poly: [[0.0; 2]; MAX_POLY_VERTS],
            poly_len: 0,
        }
    }

    /// `clip-path` polygon entry: `rect` is the polygon's bounding box
    /// (for bound-culling), `points` the page-px vertices (capped at
    /// [`MAX_POLY_VERTS`]).
    pub(crate) fn polygon(rect: Rect, points: &[[f32; 2]]) -> Self {
        let mut poly = [[0.0; 2]; MAX_POLY_VERTS];
        let n = points.len().min(MAX_POLY_VERTS);
        poly[..n].copy_from_slice(&points[..n]);
        Self {
            rect,
            radii: [0.0; CORNERS],
            poly,
            poly_len: n as u8,
        }
    }

    /// Whether this is a polygon entry.
    pub(crate) fn is_polygon(&self) -> bool {
        self.poly_len > 0
    }
}

/// The clip state one [`Op::Scissor`](crate::wgpu_be::Op::Scissor) op
/// carries: up to [`MAX_CLIP_ENTRIES`] entries, oldest first. The
/// fragment shader evaluates the entries' geometric intersection
/// per pixel ([`clip_stack_coverage_byte`]); the executor derives the
/// coarse integer scissor from the entries' intersected bounding rect.
///
/// `len == 0` disables clipping. A one-entry snapshot whose rect is
/// empty is the empty clip (draws skipped); it is distinct from
/// disabled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ClipSnapshot {
    pub len: u8,
    pub entries: [PageClip; MAX_CLIP_ENTRIES],
}

impl ClipSnapshot {
    /// Empty snapshot: clipping disabled.
    pub(crate) fn none() -> Self {
        Self {
            len: 0,
            entries: [PageClip::default(); MAX_CLIP_ENTRIES],
        }
    }

    /// Entries as a slice.
    pub(crate) fn as_slice(&self) -> &[PageClip] {
        &self.entries[..self.len as usize]
    }
}

/// Collapse a page-px clip stack to a [`ClipSnapshot`].
///
/// All-rect stacks collapse to their exact intersected bound (one
/// entry; rect∩rect is a rect, so this is exact, not conservative) —
/// which also preserves the lowering's "redundant rect push emits no
/// op" dedup, since the snapshot no longer sees the stack depth. An
/// empty stack snapshots to `len == 0` (clipping disabled); an
/// all-rect stack with an empty intersection snapshots to one empty
/// entry (the empty clip, distinct from disabled).
///
/// Stacks deeper than [`MAX_CLIP_ENTRIES`] that contain a rounded
/// entry collapse their oldest entries into one: the intersected
/// bounding rect, inset on every side by the largest corner radius
/// among the collapsed entries, with radii zeroed. Sound (the
/// collapsed entry is a subset of every collapsed input): a point at
/// least `rmax` inside every edge of the intersection sits at least
/// `rmax >= r` from both edges meeting any collapsed corner of radius
/// `r`, hence outside that corner's cut square and inside the entry's
/// straight region.
pub(crate) fn snapshot_clips(stack: &[PageClip]) -> ClipSnapshot {
    let mut snap = ClipSnapshot::none();
    if stack.is_empty() {
        return snap;
    }
    // Polygon path: a polygon never collapses into the all-rect fast
    // path below (its shape is not its bbox). Plain rects still
    // collapse exactly (rect∩rect is a rect); rounded and polygon
    // entries are kept individually, newest filling the remaining
    // slots. Past the cap the snapshot degrades to the empty clip —
    // sound (paints nothing) where dropping a constraint would paint
    // outside the true clip. Pathological input only: real stacks are
    // far shallower.
    if stack.iter().any(|e| e.is_polygon()) {
        let mut x0 = f32::NEG_INFINITY;
        let mut y0 = f32::NEG_INFINITY;
        let mut x1 = f32::INFINITY;
        let mut y1 = f32::INFINITY;
        let mut have_rect = false;
        let mut others: Vec<&PageClip> = Vec::new();
        for e in stack {
            if e.is_polygon() || e.radii != [0.0; CORNERS] {
                others.push(e);
            } else {
                have_rect = true;
                let r = e.rect;
                x0 = x0.max(r.x);
                y0 = y0.max(r.y);
                x1 = x1.min(r.x + r.w);
                y1 = y1.min(r.y + r.h);
            }
        }
        let mut n = 0usize;
        if have_rect {
            snap.entries[0] = PageClip::rect(Rect {
                x: x0,
                y: y0,
                w: (x1 - x0).max(0.0),
                h: (y1 - y0).max(0.0),
            });
            n = 1;
        }
        if others.len() > MAX_CLIP_ENTRIES - n {
            #[cfg(debug_assertions)]
            eprintln!("[bowser_paint] clip stack exceeds shaped-entry cap; using empty clip");
            snap.entries[0] = PageClip::rect(Rect {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
            });
            snap.len = 1;
            return snap;
        }
        for e in others {
            snap.entries[n] = *e;
            n += 1;
        }
        snap.len = n as u8;
        return snap;
    }
    if stack.iter().all(|e| e.radii == [0.0; CORNERS]) {
        let mut x0 = f32::NEG_INFINITY;
        let mut y0 = f32::NEG_INFINITY;
        let mut x1 = f32::INFINITY;
        let mut y1 = f32::INFINITY;
        for e in stack {
            let r = e.rect;
            x0 = x0.max(r.x);
            y0 = y0.max(r.y);
            x1 = x1.min(r.x + r.w);
            y1 = y1.min(r.y + r.h);
        }
        snap.entries[0] = PageClip::rect(Rect {
            x: x0,
            y: y0,
            w: (x1 - x0).max(0.0),
            h: (y1 - y0).max(0.0),
        });
        snap.len = 1;
        return snap;
    }
    if stack.len() <= MAX_CLIP_ENTRIES {
        for (i, e) in stack.iter().enumerate() {
            snap.entries[i] = *e;
        }
        snap.len = stack.len() as u8;
        return snap;
    }
    // Collapse the oldest `stack.len() - (MAX - 1)` entries.
    let keep = MAX_CLIP_ENTRIES - 1;
    let (old, new) = stack.split_at(stack.len() - keep);
    let mut x0 = f32::NEG_INFINITY;
    let mut y0 = f32::NEG_INFINITY;
    let mut x1 = f32::INFINITY;
    let mut y1 = f32::INFINITY;
    let mut rmax = 0.0f32;
    for e in old {
        let r = e.rect;
        x0 = x0.max(r.x);
        y0 = y0.max(r.y);
        x1 = x1.min(r.x + r.w);
        y1 = y1.min(r.y + r.h);
        for c in e.radii {
            rmax = rmax.max(c);
        }
    }
    rmax = rmax.max(0.0);
    // Inset past every collapsed corner cut; may go empty, which the
    // executor treats as an empty clip (draws skipped).
    let collapsed = PageClip::rect(Rect {
        x: x0 + rmax,
        y: y0 + rmax,
        w: (x1 - rmax - (x0 + rmax)).max(0.0),
        h: (y1 - rmax - (y0 + rmax)).max(0.0),
    });
    snap.entries[0] = collapsed;
    for (i, e) in new.iter().enumerate() {
        snap.entries[i + 1] = *e;
    }
    snap.len = MAX_CLIP_ENTRIES as u8;
    snap
}

/// Point-in-rounded-rect test for one subpixel sample, using only
/// exactly-rounded f32 ops (`+ - *`, comparisons) so the CPU and the
/// WGSL mirror agree bit-for-bit. `(x0, y0, x1, y1)` is the bounding
/// box; `radii` the circular corner radii `[TL, TR, BR, BL]`.
fn point_in_rounded_rect(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    radii: &[f32; CORNERS],
    sx: f32,
    sy: f32,
) -> bool {
    if sx < x0 || sx >= x1 || sy < y0 || sy >= y1 {
        return false;
    }
    let [rtl, rtr, rbr, rbl] = *radii;
    // Each corner box is tested only when its radius is positive; a
    // zero/negative radius is a square corner (the sample is inside by
    // the rect test above). Circle tests compare squared distances —
    // no sqrt, no division.
    if rtl > 0.0 && sx < x0 + rtl && sy < y0 + rtl {
        let dx = sx - (x0 + rtl);
        let dy = sy - (y0 + rtl);
        return dx * dx + dy * dy <= rtl * rtl;
    }
    if rtr > 0.0 && sx > x1 - rtr && sy < y0 + rtr {
        let dx = sx - (x1 - rtr);
        let dy = sy - (y0 + rtr);
        return dx * dx + dy * dy <= rtr * rtr;
    }
    if rbr > 0.0 && sx > x1 - rbr && sy > y1 - rbr {
        let dx = sx - (x1 - rbr);
        let dy = sy - (y1 - rbr);
        return dx * dx + dy * dy <= rbr * rbr;
    }
    if rbl > 0.0 && sx < x0 + rbl && sy > y1 - rbl {
        let dx = sx - (x0 + rbl);
        let dy = sy - (y1 - rbl);
        return dx * dx + dy * dy <= rbl * rbl;
    }
    true
}

/// Fractional coverage of the device pixel `(x, y)` — the unit square
/// `[x, x+1) x [y, y+1)` — by the clip stack's geometric intersection,
/// quantized to a mask byte.
///
/// Algorithm (mirrored op-for-op by `clip_coverage` in the WGSL
/// `SHADER_COMMON` prelude; keep them in sync):
/// 1. Intersect every entry's rect. An empty intersection, or any
///    degenerate entry, covers nothing: byte 0.
/// 2. If the pixel touches no rounded corner square, every rounded
///    entry is 0/1 over the whole pixel, so the exact rect formula on
///    the intersection is exact.
/// 3. Otherwise 8x8-subsample: a sub-cell counts iff its center is
///    inside *every* entry ([`point_in_rounded_rect`], which is the
///    rect test for all-zero radii). This is the true intersection
///    coverage up to the sampling grid (see the module docs for the
///    error bound).
///
/// An all-zero-radii stack reduces to
/// [`clip_coverage_byte`](crate::cpu::clip_coverage_byte) on the
/// intersection — the same bytes the old rect-only code computed — so
/// plain [`DisplayItem::PushClip`](crate::display_list::DisplayItem::PushClip)
/// stacks are bit-identical before and after.
/// Point-in-polygon (even-odd rule) for one subpixel sample, using
/// only exactly-rounded f32 ops (`+ - * /`, comparisons) so the CPU
/// and the WGSL mirror agree bit-for-bit. `pts` needs at least 3
/// vertices; fewer always reports outside.
fn point_in_polygon(sx: f32, sy: f32, pts: &[[f32; 2]]) -> bool {
    let n = pts.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    for i in 0..n {
        let (x1, y1) = (pts[i][0], pts[i][1]);
        let (x2, y2) = (pts[(i + 1) % n][0], pts[(i + 1) % n][1]);
        if (y1 > sy) != (y2 > sy) {
            let xinters = (x2 - x1) * (sy - y1) / (y2 - y1) + x1;
            if sx < xinters {
                inside = !inside;
            }
        }
    }
    inside
}

/// Whether any edge's bounding box overlaps the pixel's unit square
/// `[px, px+1)` × `[py, py+1)`: a conservative (superset) trigger for
/// "a polygon edge crosses this pixel". Edge bboxes use only
/// comparisons, and an edge crossing the square always has an
/// overlapping bbox, so sampling is never wrongly skipped; it may be
/// triggered unnecessarily, which only costs time.
fn polygon_touches_pixel(pts: &[[f32; 2]], px: f32, py: f32) -> bool {
    let n = pts.len();
    if n < 3 {
        return false;
    }
    let (qx, qy) = (px + 1.0, py + 1.0);
    for i in 0..n {
        let (x1, y1) = (pts[i][0], pts[i][1]);
        let (x2, y2) = (pts[(i + 1) % n][0], pts[(i + 1) % n][1]);
        if x1.min(x2) <= qx && x1.max(x2) >= px && y1.min(y2) <= qy && y1.max(y2) >= py {
            return true;
        }
    }
    false
}

/// Whether `pts` is a polygon entry (at least 3 vertices).
pub(crate) fn clip_stack_coverage_byte(stack: &[ClipEntry], x: usize, y: usize) -> u8 {
    let px = x as f32;
    let py = y as f32;
    // 1. Intersect the entry rects.
    let mut l = f32::NEG_INFINITY;
    let mut t = f32::NEG_INFINITY;
    let mut r = f32::INFINITY;
    let mut b = f32::INFINITY;
    for e in stack {
        // A degenerate entry empties the intersection: coverage 0.
        let Some(er) = e.rect else {
            return 0;
        };
        l = l.max(er.x());
        t = t.max(er.y());
        r = r.min(er.right());
        b = b.min(er.bottom());
    }
    if l >= r || t >= b {
        return 0;
    }
    // Exact rect coverage of the intersection — identical to
    // `clip_coverage_byte`.
    let cx = (r.min(px + 1.0) - l.max(px)).clamp(0.0, 1.0);
    let cy = (b.min(py + 1.0) - t.max(py)).clamp(0.0, 1.0);
    let rect_byte = (cx * cy * 255.0 + 0.5).floor() as u8;
    // 2. Does the pixel touch any positive-radius corner square?
    // TL: [x0, x0+rtl) x [y0, y0+rtl); TR: (x1-rtr, x1] x [y0, y0+rtr);
    // BR: (x1-rbr, x1] x (y1-rbr, y1]; BL: [x0, x0+rbl) x (y1-rbl, y1].
    let mut in_corner = false;
    for e in stack {
        // `Some`: returned above on `None`.
        let er = e.rect.unwrap();
        let x0 = er.x();
        let y0 = er.y();
        let x1 = er.right();
        let y1 = er.bottom();
        let rtl = e.radii[0].max(0.0);
        let rtr = e.radii[1].max(0.0);
        let rbr = e.radii[2].max(0.0);
        let rbl = e.radii[3].max(0.0);
        in_corner = in_corner
            || (rtl > 0.0 && px < x0 + rtl && px + 1.0 > x0 && py < y0 + rtl && py + 1.0 > y0)
            || (rtr > 0.0 && px + 1.0 > x1 - rtr && px < x1 && py < y0 + rtr && py + 1.0 > y0)
            || (rbr > 0.0 && px + 1.0 > x1 - rbr && px < x1 && py + 1.0 > y1 - rbr && py < y1)
            || (rbl > 0.0 && px < x0 + rbl && px + 1.0 > x0 && py + 1.0 > y1 - rbl && py < y1);
        if in_corner {
            break;
        }
    }
    if !in_corner {
        // Polygon entries: the pixel's unit square either misses the
        // entry's bounding box (coverage 0 — the polygon lies inside
        // its bbox), is crossed by a polygon edge (needs the sampling
        // path below), or lies strictly inside/outside the polygon (the
        // center sample decides; no edge crosses the square, so the
        // square is uniformly in or out). Entries fully covering the
        // pixel contribute 1 and `rect_byte` stands.
        for e in stack {
            let Some(pts) = e.poly_points() else {
                continue;
            };
            if pts.len() < 3 {
                // Degenerate: paints nothing (defensive; the backends
                // never build such entries with a non-empty rect).
                return 0;
            }
            // `Some`: returned above on `None`.
            let er = e.rect.unwrap();
            let (x0, y0, x1, y1) = (er.x(), er.y(), er.right(), er.bottom());
            if px + 1.0 <= x0 || px >= x1 || py + 1.0 <= y0 || py >= y1 {
                return 0;
            }
            if polygon_touches_pixel(pts, px, py) {
                in_corner = true;
                break;
            }
            if !point_in_polygon(px + 0.5, py + 0.5, pts) {
                return 0;
            }
        }
    }
    if !in_corner {
        return rect_byte;
    }
    // 3. 8x8 subpixel sampling with per-sample all-entry membership.
    // Sample offsets are (i+0.5)/8 — exact in binary fp — and the
    // count/64 scale is exact too, so only the correctly-rounded `*`/`+`
    // can round, the same on both backends.
    let mut inside = 0u32;
    for j in 0..8u32 {
        let sy = py + (j as f32 + 0.5) / 8.0;
        for i in 0..8u32 {
            let sx = px + (i as f32 + 0.5) / 8.0;
            let mut ok = true;
            for e in stack {
                // `Some`: returned above on `None`.
                let er = e.rect.unwrap();
                let entry_ok = match e.poly_points() {
                    Some(pts) => point_in_polygon(sx, sy, pts),
                    None => point_in_rounded_rect(
                        er.x(),
                        er.y(),
                        er.right(),
                        er.bottom(),
                        &e.radii,
                        sx,
                        sy,
                    ),
                };
                if !entry_ok {
                    ok = false;
                    break;
                }
            }
            if ok {
                inside += 1;
            }
        }
    }
    ((inside as f32) * (255.0 / 64.0) + 0.5).floor() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::clip_coverage_byte;
    use tiny_skia::Rect as TsRect;

    /// Build a device-px clip entry for tests.
    fn entry(x: f32, y: f32, w: f32, h: f32, radii: [f32; 4]) -> ClipEntry {
        ClipEntry {
            rect: TsRect::from_xywh(x, y, w, h),
            radii,
            poly: None,
        }
    }

    fn stack1(x: f32, y: f32, w: f32, h: f32, radii: [f32; 4]) -> [ClipEntry; 1] {
        [entry(x, y, w, h, radii)]
    }

    /// All-zero radii must reproduce `clip_coverage_byte` bit-for-bit
    /// over a grid of fractional rects and pixels — including nested
    /// stacks, which must use the entries' geometric intersection.
    #[test]
    fn zero_radii_match_rect_formula_exactly() {
        let rects = [
            (10.5, 0.0, 110.75, 100.0),
            (0.0, 20.1, 100.0, 50.6),
            (10.9, 20.1, 61.4, 50.6),
            (0.25, 0.25, 199.75, 99.75),
            (-5.5, -3.25, 7.75, 9.125),
        ];
        for &(l, t, r, b) in &rects {
            let s = stack1(l, t, r - l, b - t, [0.0; 4]);
            for y in 0..120usize {
                for x in 0..220usize {
                    assert_eq!(
                        clip_stack_coverage_byte(&s, x, y),
                        clip_coverage_byte(l, t, r, b, x, y),
                        "divergence at rect ({l},{t},{r},{b}) pixel ({x},{y})"
                    );
                }
            }
        }
        // Nested fractional rects: the per-pixel minimum would give
        // 128 here, the geometric intersection gives 64.
        let nested = [
            entry(0.5, 0.0, 99.5, 100.0, [0.0; 4]),
            entry(0.0, 0.5, 100.0, 99.5, [0.0; 4]),
        ];
        assert_eq!(clip_stack_coverage_byte(&nested, 0, 0), 64);
        assert_eq!(clip_coverage_byte(0.5, 0.5, 100.0, 100.0, 0, 0), 64);
    }

    /// A fully-rounded (circular) clip: center pixels fully covered,
    /// corner pixels empty, arc pixels partial.
    #[test]
    fn circular_clip_covers_center_not_corners() {
        // 100x100 box, radius 50 = circle centered (50,50).
        let s = stack1(0.0, 0.0, 100.0, 100.0, [50.0; 4]);
        let cov = |x, y| clip_stack_coverage_byte(&s, x, y);
        assert_eq!(cov(50, 50), 255, "center fully covered");
        assert_eq!(cov(0, 0), 0, "corner pixel outside the disc");
        assert_eq!(cov(99, 0), 0);
        assert_eq!(cov(0, 99), 0);
        assert_eq!(cov(99, 99), 0);
        assert_eq!(cov(50, 0), 255, "top edge midpoint fully covered");
        assert_eq!(cov(0, 50), 255, "left edge midpoint fully covered");
        // Pixel (14,14): the arc cuts through it — some sub-samples land
        // inside the disc, some outside.
        let arc = cov(14, 14);
        assert!(arc > 0 && arc < 255, "arc pixel partial, got {arc}");
        // Pixel (15,15): farthest corner (15,15) is 49.5 < 50 from the
        // center, so the whole pixel is inside the disc.
        assert_eq!(cov(15, 15), 255);
    }

    /// Corner symmetry: a single TL radius clips only the TL corner.
    #[test]
    fn single_corner_radius_clips_one_corner() {
        let s = stack1(0.0, 0.0, 100.0, 40.0, [20.0, 0.0, 0.0, 0.0]);
        let cov = |x, y| clip_stack_coverage_byte(&s, x, y);
        assert_eq!(cov(0, 0), 0, "rounded TL corner cut");
        assert_eq!(cov(99, 0), 255, "square TR corner kept");
        assert_eq!(cov(0, 39), 255, "square BL corner kept");
        assert_eq!(cov(99, 39), 255, "square BR corner kept");
        assert_eq!(cov(50, 20), 255, "middle fully covered");
    }

    /// Straight-edge pixels of a rounded clip keep the exact rect
    /// formula (no sampling error away from corners).
    #[test]
    fn straight_edges_stay_exact() {
        let s = stack1(10.5, 0.0, 100.25, 100.0, [20.0, 0.0, 0.0, 20.0]);
        for x in [10usize, 11, 60, 109, 110] {
            // y=50: far from both corner rows.
            assert_eq!(
                clip_stack_coverage_byte(&s, x, 50),
                clip_coverage_byte(10.5, 0.0, 110.75, 100.0, x, 50),
                "straight-edge pixel ({x},50) must match the rect formula"
            );
        }
    }

    /// Nested rounded + rect clips evaluate the geometric intersection:
    /// every 8x8 sample must agree with an f64 analytic reference, so
    /// the bytes agree exactly (samples within 1e-6 of a boundary are
    /// skipped — f32/f64 may disagree exactly on the edge). The test
    /// region is the outer clip's TL corner square, so every pixel
    /// takes the sampling path; the inner rect's fractional edges
    /// (x=10.5, y=5.25) cross the arc region, which is exactly where
    /// a per-entry minimum would go wrong.
    #[test]
    fn nested_intersection_matches_f64_reference() {
        let stack = [
            entry(0.0, 0.0, 100.0, 100.0, [20.0, 0.0, 0.0, 0.0]),
            entry(10.5, 5.25, 80.0, 90.0, [0.0; 4]),
        ];
        fn inside(sx: f64, sy: f64) -> bool {
            let in_outer = (0.0..100.0).contains(&sx) && (0.0..100.0).contains(&sy) && {
                if sx < 20.0 && sy < 20.0 {
                    let dx = sx - 20.0;
                    let dy = sy - 20.0;
                    dx * dx + dy * dy <= 400.0
                } else {
                    true
                }
            };
            let in_inner = (10.5..90.5).contains(&sx) && (5.25..95.25).contains(&sy);
            in_outer && in_inner
        }
        fn near_boundary(sx: f64, sy: f64) -> bool {
            if (sx - 10.5).abs() < 1e-6
                || (sx - 90.5).abs() < 1e-6
                || (sy - 5.25).abs() < 1e-6
                || (sy - 95.25).abs() < 1e-6
            {
                return true;
            }
            if sx < 20.0 && sy < 20.0 {
                let dx = sx - 20.0;
                let dy = sy - 20.0;
                if (dx * dx + dy * dy - 400.0).abs() < 1e-4 {
                    return true;
                }
            }
            false
        }
        let mut checked = 0u32;
        for y in 0..20usize {
            for x in 0..20usize {
                let mut count = 0u32;
                let mut skip = false;
                for j in 0..8u32 {
                    for i in 0..8u32 {
                        let sx = x as f64 + (i as f64 + 0.5) / 8.0;
                        let sy = y as f64 + (j as f64 + 0.5) / 8.0;
                        if near_boundary(sx, sy) {
                            skip = true;
                            break;
                        }
                        if inside(sx, sy) {
                            count += 1;
                        }
                    }
                    if skip {
                        break;
                    }
                }
                if skip {
                    continue;
                }
                // count*255/64 is exact in binary fp (count*255 <
                // 2^24), so the byte follows from the count exactly.
                let expected = (count as f64 * 255.0 / 64.0 + 0.5).floor() as u8;
                assert_eq!(
                    clip_stack_coverage_byte(&stack, x, y),
                    expected,
                    "pixel ({x},{y})"
                );
                checked += 1;
            }
        }
        assert!(checked > 300, "must exercise arc pixels, checked={checked}");
    }

    fn page_clip(x: f32, y: f32, w: f32, h: f32, radii: [f32; 4]) -> PageClip {
        PageClip {
            rect: Rect { x, y, w, h },
            radii,
            poly: [[0.0; 2]; MAX_POLY_VERTS],
            poly_len: 0,
        }
    }

    /// Empty stack snapshots to disabled (`len == 0`).
    #[test]
    fn snapshot_empty_stack_is_disabled() {
        let snap = snapshot_clips(&[]);
        assert_eq!(snap.len, 0);
        assert!(snap.as_slice().is_empty());
    }

    /// All-rect stacks collapse to their exact intersected bound as one
    /// entry — so redundant rect pushes snapshot identically (the
    /// lowering's no-op dedup keeps working).
    #[test]
    fn snapshot_all_rect_collapses_to_bound() {
        let a = page_clip(0.0, 0.0, 100.0, 100.0, [0.0; 4]);
        let b = page_clip(10.0, 20.0, 100.0, 100.0, [0.0; 4]);
        let one = snapshot_clips(&[a]);
        let two = snapshot_clips(&[a, b]);
        let two_dup = snapshot_clips(&[a, b, b]);
        assert_eq!(one.len, 1);
        assert_eq!(one.as_slice()[0].rect.x, 0.0);
        assert_eq!(two.len, 1);
        let bound = two.as_slice()[0].rect;
        assert_eq!(
            (bound.x, bound.y, bound.w, bound.h),
            (10.0, 20.0, 90.0, 80.0)
        );
        assert_eq!(two, two_dup, "redundant push must snapshot identically");
    }

    /// All-rect with an empty intersection: one empty entry (the empty
    /// clip), not disabled.
    #[test]
    fn snapshot_all_rect_empty_intersection_is_empty_clip() {
        let a = page_clip(0.0, 0.0, 10.0, 10.0, [0.0; 4]);
        let b = page_clip(20.0, 20.0, 10.0, 10.0, [0.0; 4]);
        let snap = snapshot_clips(&[a, b]);
        assert_eq!(snap.len, 1);
        let r = snap.as_slice()[0].rect;
        assert_eq!((r.w, r.h), (0.0, 0.0));
    }

    /// Rounded entries survive the snapshot in order; short stacks are
    /// kept whole.
    #[test]
    fn snapshot_keeps_rounded_entries() {
        let a = page_clip(0.0, 0.0, 100.0, 100.0, [10.0, 0.0, 0.0, 0.0]);
        let b = page_clip(0.0, 0.0, 100.0, 100.0, [0.0; 4]);
        let snap = snapshot_clips(&[a, b]);
        assert_eq!(snap.len, 2);
        assert_eq!(snap.as_slice()[0].radii, [10.0, 0.0, 0.0, 0.0]);
        assert_eq!(snap.as_slice()[1].radii, [0.0; 4]);
    }

    /// Deep stacks with a rounded entry collapse the oldest entries
    /// into a sound (subset) bound entry.
    #[test]
    fn snapshot_deep_stack_collapses_oldest_soundly() {
        let mut stack = Vec::new();
        // 9 nested rects shrinking by 5px per side each...
        for i in 0..9u32 {
            let m = i as f32 * 5.0;
            stack.push(page_clip(m, m, 200.0 - 2.0 * m, 200.0 - 2.0 * m, [0.0; 4]));
        }
        // ...plus one rounded entry on top.
        stack.push(page_clip(0.0, 0.0, 200.0, 200.0, [30.0, 0.0, 0.0, 0.0]));
        let snap = snapshot_clips(&stack);
        assert_eq!(snap.len, MAX_CLIP_ENTRIES as u8);
        // Collapsed entry: intersection of the oldest three rects
        // ([0,0,200,200], [5,5,190,190], [10,10,180,180] ->
        // [10,10,180,180]), radii zeroed.
        let collapsed = snap.as_slice()[0];
        assert_eq!(collapsed.radii, [0.0; 4]);
        assert_eq!(
            (
                collapsed.rect.x,
                collapsed.rect.y,
                collapsed.rect.w,
                collapsed.rect.h
            ),
            (10.0, 10.0, 180.0, 180.0)
        );
        // The collapsed bound must sit inside every collapsed input: it
        // is a subset, so coverage can only shrink (sound).
        for e in &stack[..3] {
            assert!(collapsed.rect.x >= e.rect.x);
            assert!(collapsed.rect.y >= e.rect.y);
            assert!(collapsed.rect.x + collapsed.rect.w <= e.rect.x + e.rect.w);
            assert!(collapsed.rect.y + collapsed.rect.h <= e.rect.y + e.rect.h);
        }
        // The rounded top entry survives untouched at the end.
        assert_eq!(
            snap.as_slice()[MAX_CLIP_ENTRIES - 1].radii,
            [30.0, 0.0, 0.0, 0.0]
        );
    }
}
