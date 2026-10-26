//! Relative and absolute positioning (child of the `layout` module).
//!
//! The flow passes skip out-of-flow boxes (`position: absolute |
//! fixed`): those take no space in normal flow and are recorded in
//! [`Forest::abs_items`] via [`record_abs_item`] instead. After normal
//! flow, [`layout_abs_items`] lays each recorded subtree out against its
//! containing block, and [`apply_relative_offsets`] shifts
//! `position: relative` boxes without changing their flow footprint.
//! Everything stays in the provisional frame until the single
//! `apply_pending_shifts` pass at the end of `layout_forest`, so static
//! and out-of-flow subtrees alike move exactly once.
//!
//! `position: sticky` keeps its in-flow footprint like `relative`, but its
//! final position additionally depends on the scroll offset: after
//! `layout_forest` (rects final, all deferred shifts flushed),
//! [`apply_sticky_constraints`] — driven by `crate::layout_with_scroll` —
//! constrains each sticky box per CSS Position 3 §3.1.
//!
//! Implemented (CSS 2.1 §10.1, §10.3.7, §10.6.4; CSS Position 3 §3.1;
//! CSS Transforms 1 §3):
//! - containing block = nearest positioned ancestor's padding box, or the
//!   initial containing block (viewport) when there is none; `fixed` is
//!   against the viewport unless an ancestor establishes a fixed
//!   containing block (a non-`none` `transform`, a `will-change` naming a
//!   stacking-context trigger, a non-`none` `filter`, a non-`none`
//!   `perspective`, or `contain: paint | layout | strict | content`), in
//!   which case the nearest such ancestor's padding box is the
//!   containing block; `absolute` diverts to the nearest positioned or
//!   fixed-CB-establishing ancestor.
//! - `top`/`right`/`bottom`/`left` offsets; over-constrained values drop
//!   the end side in LTR (`right`/`bottom`), the start side in RTL.
//! - `auto` offsets fall back to the static position recorded during flow:
//!   for out-of-flow children of flex containers it is derived as if the
//!   child were the sole flex item (flexbox §4.2: main-start edge per
//!   `flex-direction`, `align-self` on the cross axis; `justify-content`
//!   does not apply), and for grid children as the sole grid item
//!   (css-grid-1 §9.2, honoring `justify-self`/`align-self`).
//! - `auto` width on out-of-flow boxes uses shrink-to-fit, unless both
//!   horizontal insets are specified — then it solves from the insets
//!   (containing block minus insets minus margins, CSS 2.1 §10.3.7); the
//!   same holds vertically (§10.6.4) only when the containing-block
//!   height is definite — against an indefinite height the stretch rule
//!   does not apply and `height: auto` keeps its content height.
//!   `auto` margins compute to zero.
//! - an out-of-flow percentage (or percentage-bearing `calc()`) `height`
//!   resolves against the containing block's *used* height even when that
//!   height is indefinite (browser consensus; WPT
//!   css-position/position-relative-016 t8/t9) — but only the height:
//!   percentage `top`/`bottom` on the out-of-flow box itself still
//!   compute to `auto` (CSS 2.1 §10.6.4).
//! - relative offsets shift the border box only; siblings and the box's
//!   own flow footprint are untouched.
//! - sticky: the box keeps its laid-out position; the insets only
//!   constrain it (CSS Position 3 §3.4 — they are *not* an up-front
//!   relative shift). Per axis, the nearest scroll container with a
//!   matching scrollable axis (the viewport's when it scrolls with the
//!   viewport; the two axes may use different scrollports) supplies the
//!   sticky view rectangle (its scrollport inset by the sticky offsets);
//!   the box's border box is clamped into it (it sticks when scrolling
//!   would move a border edge outside, and unsticks when scrolled back),
//!   then its "position box" (CSS Position 3 §3.4: the margin box, except
//!   a side whose margin edge lies within its own margin of the
//!   containing-block edge uses that distance instead) is hard-bounded by
//!   its containing block — the nearest block-container ancestor (CSS
//!   2.1 §10.1 with the 2012-08-01 erratum; anonymous block boxes are
//!   skipped per §9.2.1.1). When the containing block is itself a scroll
//!   container the bound uses its scrollable overflow size (Firefox/Chrome
//!   use the scrollbox there, not the client box); when the position box
//!   exceeds the bound the proviso is vacuous and skipped. An axis whose
//!   insets are both `auto` is unconstrained. Over-constrained scrollport
//!   ranges resolve to the start side (top always; left in LTR, right in
//!   RTL — the containing block's writing mode).
//!
//! Documented simplifications:
//! - the static position is the hypothetical in-flow position
//!   (CSS 2.1 §10.3.7/§10.6.4), margins included: block-level recording
//!   sites pass the border-box origin `resolve_metrics` computed for the
//!   box in flow (its margins — including `auto`-margin centering on a
//!   definite width — are already folded in), while inline-level boxes
//!   record the margin-box anchor (just after the nearest preceding
//!   placed fragment — the new line's start fragment after a forced
//!   break — else the content-box origin, vertically the line's
//!   baseline) and `abs_position` adds the used margins when the
//!   offsets are `auto`.
//! - percentage `top`/`bottom` on relatively positioned boxes compute to
//!   `auto` (no offset) when the containing block's height is indefinite
//!   (browser-consensus behavior per WebKit bug 26396); lengths always
//!   resolve. Pinned by
//!   `relative_pct_top_computes_to_auto_against_indefinite_cb_height`.
//! - out-of-flow grid children derive their static position as the sole
//!   grid item (css-grid-1 §9.2, honoring `justify-self`/`align-self`).
//! - `z-index` is parsed onto `StyleData` and paint emits CSS 2.1
//!   Appendix E order with stacking contexts: positioned boxes with
//!   non-`auto` `z-index` create stacking contexts, as do boxes with
//!   `opacity < 1` or non-`normal` `mix-blend-mode` (the only supported
//!   non-positioned stacking-context creators), and static flex/grid
//!   items with non-`auto` `z-index` join the z-index steps. See
//!   `bowser_paint::display_list::creates_stacking_context` /
//!   `joins_z_steps` for the exact set.
//! - sticky: each sticky box constrains per axis against the scrollport
//!   (padding box) of the nearest scroll container with a matching
//!   scrollable axis on its anchoring chain, or the viewport's when it
//!   scrolls with the viewport. The walk follows the box's *anchoring chain* rather than raw DOM ancestry: crossing
//!   an `absolute` ancestor re-anchors it at that box's containing
//!   block (the `cb_inside`/`fixed_cb_inside` tracking
//!   `crate::scrollable_extent` uses), so a sticky box inside an
//!   abspos subtree constrains against the scroller that actually
//!   moves it — and still falls back to the viewport when the
//!   containing block sits outside every scroller. A nested scroller
//!   constrains at the live offset the caller passes in
//!   `nested_offsets` (keyed by the scroller's DOM node, since arena
//!   indices are not stable across layouts); a scroller with no entry
//!   constrains at offset zero — exact, not an approximation: the
//!   caller guarantees every nested scroll container with a nonzero
//!   offset has an entry (see `ScrollTree::nested_offsets`, which
//!   omits only zero-offset containers), and layout owns no
//!   per-scroller scroll state of its own, so zero is the only sound
//!   fallback.
//!   The scrollport constraint operates on the border box; the
//!   containing-block bound (see below) on the "position box" (CSS
//!   Position 3 §3.4: the margin box, except a side whose margin edge
//!   lies within its own margin of the containing-block edge uses that
//!   distance instead). The containing block is the nearest
//!   block-container ancestor's content box (CSS 2.1 §10.1 with the
//!   2012-08-01 erratum; anonymous block boxes are skipped per
//!   §9.2.1.1), the viewport for the root — but when that block is
//!   itself a scroll container the bound uses its scrollable overflow
//!   size, since Firefox/Chrome constrain against the scrollbox there.
//!   A sticky box inside a `fixed` subtree resolves against its own
//!   containing context statically: the walk is bounded by the fixed
//!   ancestor — scroll containers inside the subtree still constrain
//!   at their live offsets, while with none the box constrains against
//!   the viewport frame at a zero offset (document scrolling cannot
//!   move a viewport-locked subtree; only the containing-block bound
//!   applies). With a zero scroll offset an unstuck sticky box stays
//!   exactly where it was laid out.

use super::layout_grid::{self, ItemAlign};
use super::{
    BlockInput, BoxType, FlexAlign, FlexDir, FontMetrics, Forest, LayoutScratch, Position,
    PrefWidthCache, SpecLen, SpecMargin, Task, flex_align_of, flex_dir_of, flex_kind, fresh_floats,
    run_tasks,
};
use crate::boxes::AbsItem;
use bowser_css::{CssValue, props};
use std::collections::HashMap;

/// Out-of-flow positions: laid out by the post-pass, taking no space in
/// normal flow. Delegates to the `boxes` predicate (`boxes` owns the
/// forest; `layout` depends on it, not the other way around).
pub(crate) fn is_out_of_flow(forest: &Forest, idx: usize) -> bool {
    crate::boxes::is_out_of_flow(forest, idx)
}

/// Any non-`static` position: establishes a containing block for
/// out-of-flow descendants (CSS 2.1 §10.1).
fn is_positioned(forest: &Forest, idx: usize) -> bool {
    !matches!(forest.nodes[idx].data.position, Position::Static)
}

/// Whether the box establishes a containing block for `position: fixed`
/// and absolutely positioned descendants: any fixed-CB trigger — a
/// non-`none` `transform` (CSS Transforms 1 §3), a `will-change` naming
/// `transform`/`filter`/`perspective` (CSS Position 3 §2.1; other
/// stacking-context hints like `opacity` do not count), a non-`none`
/// `filter` (Filter Effects 1), a non-`none` `perspective` (CSS
/// Transforms 2), or `contain: paint | layout | strict | content` (CSS
/// Contain 1/2).
fn has_fixed_cb_trigger(forest: &Forest, idx: usize) -> bool {
    forest.nodes[idx]
        .style
        .as_deref()
        .is_some_and(crate::establishes_fixed_cb)
}

/// Resolve the containing block of the out-of-flow box `node`: the
/// nearest ancestor establishing a containing block, or `None` for the
/// initial containing block. For `position: fixed` the walk stops at the
/// nearest ancestor with a fixed-CB trigger (CSS Transforms 1 §3 et
/// al.): that ancestor's padding box becomes the containing block
/// instead of the viewport. With no such ancestor, `fixed` is against
/// the viewport. For `position: absolute` the walk stops at the nearest
/// ancestor that is positioned *or* establishes a fixed containing
/// block: a non-positioned `transform`/`filter`/`will-change`/
/// `perspective`/`contain` ancestor still diverts the box (CSS
/// Transforms 1 §3), even when a positioned ancestor sits above it.
fn resolve_containing_block(forest: &Forest, node: usize) -> Option<usize> {
    let fixed = forest.nodes[node].data.position == Position::Fixed;
    let absolute = forest.nodes[node].data.position == Position::Absolute;
    debug_assert!(fixed || absolute, "out-of-flow only");
    let mut cur = forest.nodes[node].parent;
    // Hang guard: a parent cycle is unreachable from well-formed box
    // construction (parents are created before children, so every index
    // strictly decreases along the chain), but an internal bug in
    // anonymous wrapping or table fix-ups could stitch one in. An
    // acyclic walk can visit at most `nodes.len()` distinct nodes before
    // reaching the root, so capping the steps at the node count
    // terminates any cycle without costing the common case anything;
    // the guard fails closed (initial containing block) instead of
    // hanging the layout thread.
    let mut steps = forest.nodes.len();
    while let Some(p) = cur {
        if steps == 0 {
            return None;
        }
        steps -= 1;
        if fixed {
            if has_fixed_cb_trigger(forest, p) {
                return Some(p);
            }
        } else if is_positioned(forest, p) || has_fixed_cb_trigger(forest, p) {
            return Some(p);
        }
        cur = forest.nodes[p].parent;
    }
    None
}

/// Record an out-of-flow box met during normal flow, with the border-box
/// origin flow would have given it (`static_x`/`static_y`, provisional
/// frame). Idempotent: the inline-flex estimate pass and flex stretch
/// re-layouts can meet the same box twice.
pub(crate) fn record_abs_item(forest: &mut Forest, node: usize, static_x: f32, static_y: f32) {
    record_abs(forest, node, static_x, static_y, false);
}

/// Record an out-of-flow box met during inline flow, with the margin-box
/// anchor flow would have given it: the point where the box's margin box
/// would begin (just after the nearest preceding placed fragment, else
/// the containing block's content-box origin). The used margins are not
/// folded in; [`abs_position`] adds them when the offsets on an axis are
/// `auto` (CSS 2.1 §10.3.7/§10.6.4: the static position is the
/// hypothetical in-flow position, margins included).
pub(crate) fn record_abs_item_margin_edge(
    forest: &mut Forest,
    node: usize,
    static_x: f32,
    static_y: f32,
) {
    record_abs(forest, node, static_x, static_y, true);
}

fn record_abs(
    forest: &mut Forest,
    node: usize,
    static_x: f32,
    static_y: f32,
    margins_pending: bool,
) {
    // O(1) per-node dedup via the flag on `Node`: the old linear scan of
    // `abs_items` was O(items) per record, quadratic overall.
    if forest.nodes[node].abspos_recorded {
        return;
    }
    forest.nodes[node].abspos_recorded = true;
    let containing_block = resolve_containing_block(forest, node);
    forest.abs_items.push(AbsItem {
        node,
        containing_block,
        static_x,
        static_y,
        margins_pending,
    });
}

/// Whether a `SpecLen` carries a percentage component: `Pct`, or a
/// `Math` calc whose source syntactically contained a `%` leaf
/// (`had_pct`). Per CSS 2.1 §10.6.4 / CSS Values 4 §8.1.4 such a length
/// computes to `auto` when resolved against an indefinite height — even
/// when the calc folds to a zero percentage coefficient (e.g.
/// `calc(10px + 0%)` folds to `pct: 0.0, px: 10.0` but still behaves as
/// `auto`). Shared with `layout`'s abspos sizing so both agree on which
/// insets go indefinite.
pub(crate) fn has_pct(s: SpecLen) -> bool {
    matches!(s, SpecLen::Pct(_)) || matches!(s, SpecLen::Math { had_pct, .. } if had_pct)
}

/// Containing-block rect (padding box) in the provisional frame, as
/// `(x, y, w, h)`, plus whether its height is definite. `None` is the
/// initial containing block (viewport): always definite.
fn cb_rect(forest: &Forest, cb: Option<usize>) -> (f32, f32, f32, f32, bool) {
    match cb {
        None => (0.0, 0.0, forest.vp_w, forest.vp_h, true),
        Some(c) => {
            let n = &forest.nodes[c];
            let m = &n.metrics;
            (
                n.rect.x + m.bl,
                n.rect.y + m.bt,
                (n.rect.w - m.bl - m.br).max(0.0),
                (n.rect.h - m.bt - m.bb).max(0.0),
                n.height_definite,
            )
        }
    }
}

/// Used self-alignment keyword for one axis of an absolutely-positioned
/// box (css-position-3 §4.3): how the margin box sits inside the
/// inset-modified containing block when both insets are specified and
/// neither margin is `auto`.
#[derive(Clone, Copy, PartialEq, Debug)]
enum AbsSelfAlign {
    Start,
    End,
    Center,
}

/// Used self-alignment for one abspos axis: the keyword plus whether
/// `unsafe` was explicitly specified (an explicit `<overflow-position>`
/// prefix; the cascade does not produce one yet, so the second form is
/// forward-compatibility). `horizontal` selects `justify-self` (where
/// `left`/`right` are physical) vs `align-self`; `rtl` is the box's own
/// direction, which it inherits through the containing-block chain.
///
/// `normal`/`auto` behave as `start`; `stretch` and `baseline` fall back
/// to `start` (the abspos size is fixed by abspos sizing, so there is
/// nothing to stretch, and there is no baseline-sharing group).
/// `auto` resolving against the container's `justify-items`/
/// `align-items` is not threaded through: no WPT covers an abspos box
/// with `*-self: auto` under a non-`normal` `*-items` container, and the
/// containing-block node is not at hand here.
fn abs_self_align(
    style: Option<&bowser_css::ComputedStyle>,
    prop: u8,
    horizontal: bool,
    rtl: bool,
) -> (AbsSelfAlign, bool) {
    fn keyword(v: &CssValue) -> Option<&str> {
        match v {
            CssValue::Keyword(k) => Some(k.as_str()),
            _ => None,
        }
    }
    // An optional `<overflow-position>` (`safe`/`unsafe`) prefix plus
    // the `<self-position>` keyword.
    let (kw, explicit_unsafe) = match style.and_then(|s| s.get_by_index(prop)) {
        Some(CssValue::List(items)) if items.len() == 2 => {
            let overflow = match keyword(&items[0]) {
                Some(k) if k.eq_ignore_ascii_case("unsafe") => Some(true),
                Some(k) if k.eq_ignore_ascii_case("safe") => Some(false),
                _ => None,
            };
            match (overflow, keyword(&items[1])) {
                (Some(u), Some(k)) => (k, u),
                _ => return (AbsSelfAlign::Start, false),
            }
        }
        Some(v) => match keyword(v) {
            Some(k) => (k, false),
            None => return (AbsSelfAlign::Start, false),
        },
        None => return (AbsSelfAlign::Start, false),
    };
    let end = |k: &str| {
        k.eq_ignore_ascii_case("end")
            || k.eq_ignore_ascii_case("self-end")
            || k.eq_ignore_ascii_case("flex-end")
            || k.eq_ignore_ascii_case("flow-end")
    };
    // `self-start`/`self-end` resolve against the box's own direction;
    // plain `start`/`end` against the containing block's — the two agree
    // through inheritance in practice, and this function only sees the
    // box's own `rtl`. On the vertical axis there is no direction flip:
    // `start` is the top. Physical `left`/`right` never flip.
    let align = if kw.eq_ignore_ascii_case("center") {
        AbsSelfAlign::Center
    } else if horizontal && kw.eq_ignore_ascii_case("left") {
        AbsSelfAlign::Start
    } else if horizontal && kw.eq_ignore_ascii_case("right") {
        AbsSelfAlign::End
    } else if end(kw) {
        if rtl && horizontal {
            AbsSelfAlign::Start
        } else {
            AbsSelfAlign::End
        }
    } else {
        // `start`, `self-start`, `auto`, `normal`, `stretch`,
        // `baseline`, and anything unrecognized behave as `start` here.
        if rtl && horizontal {
            AbsSelfAlign::End
        } else {
            AbsSelfAlign::Start
        }
    };
    (align, explicit_unsafe)
}

/// Border-box origin along one abspos axis under css-position-3 §4.3
/// self-alignment: align the margin box (`m0 + size + m1`) within the
/// inset-modified containing block (`imcb_start`, `imcb_size`) per
/// `align`, then back off to the border-box origin. Without an explicit
/// `unsafe`, an overflowing margin box is adjusted to minimize overflow
/// (start-aligned); `stretch` never reaches here (it falls back to
/// `start` in [`abs_self_align`]).
fn align_abspos_axis(
    imcb_start: f32,
    imcb_size: f32,
    m0: f32,
    size: f32,
    m1: f32,
    align: AbsSelfAlign,
    explicit_unsafe: bool,
) -> f32 {
    let mb = m0 + size + m1;
    let align = if mb > imcb_size && !explicit_unsafe {
        AbsSelfAlign::Start
    } else {
        align
    };
    let margin_start = match align {
        AbsSelfAlign::Start => imcb_start,
        AbsSelfAlign::End => imcb_start + imcb_size - mb,
        AbsSelfAlign::Center => imcb_start + (imcb_size - mb) / 2.0,
    };
    margin_start + m0
}

/// Used border-box origin of an out-of-flow box in the provisional frame
/// (CSS 2.1 §10.3.7, §10.6.4). The subtree is already laid out, so the
/// used width/height come from its provisional rect; `auto` margins were
/// forced to zero for sizing by the `abspos` `Task::Block`, but when both
/// insets on an axis resolve against a *definite* containing block, the
/// used margins solve from the constraint equation instead (both `auto`
/// centers, one `auto` takes the remainder); vertically an indefinite
/// containing-block height skips the margin solve, keeping `auto`
/// margins at zero (the stretch rule does not apply there, so there is
/// no meaningful remainder — see `abspos_solved_sizes`).
/// Percentage `top` / `bottom` compute to `auto` (falling back to the static
/// position) when the containing-block height is indefinite
/// (`cb_h_definite == false`, CSS 2.1 §10.6.4); lengths always resolve.
#[allow(clippy::too_many_arguments)]
fn abs_position(
    forest: &Forest,
    node: usize,
    cb_x: f32,
    cb_y: f32,
    cb_w: f32,
    cb_h: f32,
    cb_h_definite: bool,
    static_x: f32,
    static_y: f32,
    margins_pending: bool,
) -> (f32, f32) {
    let (fs, rtl, margin, off, bw, bh, ml, mr, mt, mb) = {
        let n = &forest.nodes[node];
        let d = &n.data;
        (
            d.font_size,
            d.rtl,
            d.margin,
            d.offset,
            n.rect.w,
            n.rect.h,
            n.metrics.ml,
            n.metrics.mr,
            n.metrics.mt,
            n.metrics.mb,
        )
    };
    let (root_fs, vp_w, vp_h) = (forest.root_font_size, forest.vp_w, forest.vp_h);
    // `auto` margins compute to zero for out-of-flow boxes; percentage
    // margins resolve against the containing-block width.
    let mu = |side: SpecMargin, used: f32| match side {
        SpecMargin::Auto => 0.0,
        SpecMargin::Len(s) => s.resolve(fs, root_fs, vp_w, vp_h, cb_w).unwrap_or(used),
    };
    let (mut ml_u, mut mr_u, mut mt_u, mut mb_u) = (
        mu(margin[3], ml),
        mu(margin[1], mr),
        mu(margin[0], mt),
        mu(margin[2], mb),
    );
    let o = |s: SpecLen, base: f32| s.resolve(fs, root_fs, vp_w, vp_h, base);
    // Percentage `top`/`bottom` compute to `auto` when the
    // containing-block height is indefinite (CSS 2.1 §10.6.4); lengths
    // always resolve. Horizontal percentages always resolve against the
    // (always definite) containing-block width.
    let ov = |s: SpecLen| {
        if !cb_h_definite && has_pct(s) {
            None
        } else {
            o(s, cb_h)
        }
    };
    let (top, right, bottom, left) = (ov(off[0]), o(off[1], cb_w), ov(off[2]), o(off[3], cb_w));
    // css-position-3 §4.2 (abspos insets): when both insets on an axis
    // are non-`auto` and their sum exceeds the containing-block size,
    // the inset-modified containing block would be negative-sized; the
    // weaker (end) inset is then adjusted so the IMCB is zero-sized
    // instead. Horizontally the end side is `right` in LTR, `left` in
    // RTL; vertically it is always `bottom`. The vertical clamp needs a
    // definite containing-block height to mean anything.
    let (left, right) = match (left, right) {
        (Some(l), Some(r)) if l + r > cb_w => {
            if rtl {
                (Some(cb_w - r), Some(r))
            } else {
                (Some(l), Some(cb_w - l))
            }
        }
        x => x,
    };
    let (top, bottom) = match (top, bottom) {
        (Some(t), Some(b)) if cb_h_definite && t + b > cb_h => (Some(t), Some(cb_h - t)),
        x => x,
    };
    // CSS 2.1 §10.3.7 / §10.6.4: when both insets on an axis resolve,
    // `auto` margins solve from the constraint equation instead of
    // computing to zero — both `auto` splits the remainder equally
    // (centering), one `auto` takes the whole remainder. Horizontally
    // (§10.3.7) a negative remainder instead pins the start margin to
    // zero (`margin-left` in LTR, `margin-right` in RTL) and puts the
    // whole deficit on the end margin. Vertically (§10.6.4) there is no
    // such clause: the remainder always splits evenly. With
    // `width: auto` (or `height: auto`) the used size already consumed
    // the remainder in `abspos_solved_sizes`, so the split still yields
    // zero and this is a no-op there.
    let solve = |a: SpecMargin,
                 b: SpecMargin,
                 au: f32,
                 bu: f32,
                 box_len: f32,
                 cb_len: f32,
                 i0: f32,
                 i1: f32,
                 horizontal: bool| {
        let rem = |other: f32| cb_len - i0 - i1 - box_len - other;
        match (a, b) {
            (SpecMargin::Auto, SpecMargin::Auto) => {
                let rem = cb_len - i0 - i1 - box_len;
                if horizontal && rem < 0.0 {
                    if rtl { (rem, 0.0) } else { (0.0, rem) }
                } else {
                    (rem / 2.0, rem / 2.0)
                }
            }
            (SpecMargin::Auto, _) => (rem(bu), bu),
            (_, SpecMargin::Auto) => (au, rem(au)),
            _ => (au, bu),
        }
    };
    if let (Some(l), Some(r)) = (left, right) {
        (ml_u, mr_u) = solve(margin[3], margin[1], ml_u, mr_u, bw, cb_w, l, r, true);
    }
    // Vertical auto-margin solving likewise needs a definite
    // containing-block height: against an indefinite height the
    // §10.6.4 stretch rule does not apply (see `abspos_solved_sizes`),
    // so the equation's remainder is meaningless and `auto` margins
    // stay zero — they are only solved when the box's own height also
    // solved against that height.
    if cb_h_definite && let (Some(t), Some(b)) = (top, bottom) {
        (mt_u, mb_u) = solve(margin[0], margin[2], mt_u, mb_u, bh, cb_h, t, b, false);
    }
    // Over-constrained values ignore the end side in LTR (`right`), the
    // start side in RTL (`left`); `bottom` is always the dropped side
    // vertically. All-`auto` falls back to the static position.
    //
    // css-grid-1 §9.2: the static position of an abspos child of a grid
    // container is determined as if it were the sole grid item in the
    // grid's content box, honoring `justify-self`/`align-self`. The
    // recorded static position is the content-box origin; the margin box
    // is aligned within the content size here, where the used box size
    // is known. `stretch` behaves as `start`: the abspos size is fixed
    // by abspos sizing, so there is nothing to stretch.
    let (static_x, static_y) = match forest.nodes[node].parent {
        Some(p) if forest.nodes[p].box_type == BoxType::Grid => {
            let pn = &forest.nodes[p];
            let pm = &pn.metrics;
            let cw = (pn.rect.w - pm.bl - pm.br - pm.pl - pm.pr).max(0.0);
            let ch = (pn.rect.h - pm.bt - pm.bb - pm.pt - pm.pb).max(0.0);
            let ps = pn.style.as_deref();
            let cs = forest.nodes[node].style.as_deref();
            let ja = layout_grid::self_align_of(
                cs,
                props::JUSTIFY_SELF,
                layout_grid::item_align_of(ps, props::JUSTIFY_ITEMS),
            );
            let aa = layout_grid::self_align_of(
                cs,
                props::ALIGN_SELF,
                layout_grid::item_align_of(ps, props::ALIGN_ITEMS),
            );
            let sx = match ja {
                ItemAlign::End => static_x + cw - mr_u - bw,
                ItemAlign::Center => static_x + (cw - (ml_u + bw + mr_u)) / 2.0 + ml_u,
                _ => static_x + ml_u,
            };
            let sy = match aa {
                ItemAlign::End => static_y + ch - mb_u - bh,
                ItemAlign::Center => static_y + (ch - (mt_u + bh + mb_u)) / 2.0 + mt_u,
                _ => static_y + mt_u,
            };
            (sx, sy)
        }
        // flexbox §4.2: the static position of an abspos child of a flex
        // container is derived as if it were the sole flex item. The main
        // axis starts at the container's main-start edge per
        // `flex-direction` (`column-reverse` flips to the bottom edge; on
        // the row axis `row-reverse` XORs with the container's
        // `direction`, mirroring item layout in `layout_flex`); the cross
        // axis honors `align-self`, with `auto` falling back to the
        // container's `align-items`. `justify-content` does not apply to
        // abspos children. `stretch` behaves as `start` (the abspos size
        // is fixed by abspos sizing, so there is nothing to stretch), and
        // a sole item's `baseline` aligns to its own baseline, i.e.
        // `start` as well.
        Some(p) if flex_kind(forest, p).is_some() => {
            let pn = &forest.nodes[p];
            let pm = &pn.metrics;
            let cw = (pn.rect.w - pm.bl - pm.br - pm.pl - pm.pr).max(0.0);
            let ch = (pn.rect.h - pm.bt - pm.bb - pm.pt - pm.pb).max(0.0);
            let ps = pn.style.as_deref();
            let cs = forest.nodes[node].style.as_deref();
            let dir = flex_dir_of(ps);
            // The container's own `direction` orients the main axis (it
            // inherits to the child, but the child may override it).
            let rtl = pn.data.rtl;
            let horizontal = matches!(dir, FlexDir::Row | FlexDir::RowRev);
            // CSS Box Alignment 3 §4.1: the static position treats the
            // abspos child as the sole flex item, so `self-start` /
            // `self-end` resolve against the *child's* direction. The
            // `Start`/`End` mapping below is already physical (no
            // container-direction flip), so the child's own rtl-ness is
            // the whole story on a horizontal cross axis.
            let self_rtl = !horizontal && forest.nodes[node].data.rtl;
            let items_align = flex_align_of(
                ps.and_then(|s| s.get_str_by_index(props::ALIGN_ITEMS)),
                self_rtl,
            )
            .unwrap_or(FlexAlign::Stretch);
            let align = flex_align_of(
                cs.and_then(|s| s.get_str_by_index(props::ALIGN_SELF)),
                self_rtl,
            )
            .unwrap_or(items_align);
            let main_at_start = match dir {
                FlexDir::Col => true,
                FlexDir::ColRev => false,
                FlexDir::Row => !rtl,
                FlexDir::RowRev => rtl,
            };
            if horizontal {
                let sx = if main_at_start {
                    static_x + ml_u
                } else {
                    static_x + cw - mr_u - bw
                };
                let sy = match align {
                    FlexAlign::End => static_y + ch - mb_u - bh,
                    FlexAlign::Center => static_y + (ch - (mt_u + bh + mb_u)) / 2.0 + mt_u,
                    _ => static_y + mt_u,
                };
                (sx, sy)
            } else {
                let sy = if main_at_start {
                    static_y + mt_u
                } else {
                    static_y + ch - mb_u - bh
                };
                let sx = match align {
                    FlexAlign::End => static_x + cw - mr_u - bw,
                    FlexAlign::Center => static_x + (cw - (ml_u + bw + mr_u)) / 2.0 + ml_u,
                    _ => static_x + ml_u,
                };
                (sx, sy)
            }
        }
        // Inline-flow recording sites pass the margin-box anchor (the
        // point where the margin box would begin in flow), so the used
        // margins — `auto` already zeroed above, percentages resolved
        // against the containing-block width — shift the border-box
        // origin to the hypothetical in-flow position (CSS 2.1
        // §10.3.7/§10.6.4). Block-level sites recorded the border-box
        // origin directly, with their margins already folded in by
        // `resolve_metrics`, so they pass through untouched.
        _ if margins_pending => (static_x + ml_u, static_y + mt_u),
        _ => (static_x, static_y),
    };
    // css-position-3 §4.3: an `auto` inset aligns the margin box to the
    // stronger inset's edge; with both insets specified, an `auto`
    // margin resolves per §4.2 (already solved above) and the box sits
    // at the start inset — otherwise the margin box self-aligns within
    // the inset-modified containing block per `justify-self` /
    // `align-self`. All-`auto` falls back to the static position.
    // Over-constrained insets already dropped the end side in LTR
    // (`right`), the start side in RTL (`left`); `bottom` is always the
    // dropped side vertically.
    let cs = forest.nodes[node].style.as_deref();
    let fx = match (left, right) {
        (Some(l), None) => cb_x + l + ml_u,
        (None, Some(r)) => cb_x + cb_w - r - mr_u - bw,
        (None, None) => static_x,
        (Some(l), Some(r))
            if matches!(margin[3], SpecMargin::Auto) || matches!(margin[1], SpecMargin::Auto) =>
        {
            if rtl {
                cb_x + cb_w - r - mr_u - bw
            } else {
                cb_x + l + ml_u
            }
        }
        (Some(l), Some(r)) => {
            let (align, explicit_unsafe) = abs_self_align(cs, props::JUSTIFY_SELF, true, rtl);
            align_abspos_axis(
                cb_x + l,
                (cb_w - l - r).max(0.0),
                ml_u,
                bw,
                mr_u,
                align,
                explicit_unsafe,
            )
        }
    };
    let fy = match (top, bottom) {
        (Some(t), None) => cb_y + t + mt_u,
        (None, Some(b)) => cb_y + cb_h - b - mb_u - bh,
        (None, None) => static_y,
        (Some(t), Some(_))
            if matches!(margin[0], SpecMargin::Auto) || matches!(margin[2], SpecMargin::Auto) =>
        {
            cb_y + t + mt_u
        }
        (Some(t), Some(b)) => {
            let (align, explicit_unsafe) = abs_self_align(cs, props::ALIGN_SELF, false, false);
            align_abspos_axis(
                cb_y + t,
                (cb_h - t - b).max(0.0),
                mt_u,
                bh,
                mb_u,
                align,
                explicit_unsafe,
            )
        }
    };
    (fx, fy)
}

/// Post-pass: lay out every recorded out-of-flow subtree against its
/// containing block.
///
/// Each item runs as a `Task::Block` with `abspos: true` at a provisional
/// origin; the used position is then computed from the offsets (or the
/// static position) and recorded as a deferred shift on the subtree root,
/// so the single `apply_pending_shifts` pass at the end of
/// `layout_forest` moves it exactly once along with everything else.
/// Items recorded while laying out a subtree (nested out-of-flow boxes)
/// are picked up by the index loop in document order, so a containing
/// block that is itself out-of-flow is always placed first.
pub(crate) fn layout_abs_items(
    forest: &mut Forest,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
) {
    let mut tasks: Vec<Task> = Vec::new();
    let mut scratch = LayoutScratch::new();
    let mut i = 0;
    while i < forest.abs_items.len() {
        let (node, cb, static_x, static_y, margins_pending) = {
            let a = &forest.abs_items[i];
            (
                a.node,
                a.containing_block,
                a.static_x,
                a.static_y,
                a.margins_pending,
            )
        };
        let (cb_x, cb_y, cb_w, cb_h, cb_h_definite) = cb_rect(forest, cb);
        // `abspos: true`: auto margins compute to zero and an `auto`
        // width uses shrink-to-fit (see `layout_block`) — unless both
        // insets on an axis are specified, in which case the used size
        // solves from the insets (CSS 2.1 §10.3.7/§10.6.4). Laid out at a
        // provisional origin; the deferred shift below moves it.
        // Percentage insets resolve against an indefinite CB height as
        // `auto` (CSS 2.1 §10.6.4), so `cb_h` stays `None` here and
        // `abs_position` keeps `cb_h_definite: false`: `abspos_solved_sizes`
        // and `abs_position` agree on the `ov` treatment only when both
        // see the indefinite height. A percentage (or percentage-bearing
        // `calc()`) *height* is the exception: browsers resolve it against
        // the CB's *used* height (WPT css-position/position-relative-016
        // t8/t9), which is final by now — flow is done and items lay out
        // outer-first, so an abspos CB's rect already has its final size.
        // Gated strictly to non-replaced boxes with a percentage height:
        // `top: 0`-only, `min-height`-only and all-`auto` boxes stay
        // indefinite (t13/t14/t15), and replaced-height definiteness is
        // untouched.
        let abs_pct_height_base = if !cb_h_definite
            && forest.nodes[node].intrinsic.is_none()
            && has_pct(forest.nodes[node].data.height)
        {
            Some(cb_h)
        } else {
            None
        };
        tasks.push(Task::Block(BlockInput {
            idx: node,
            cb_x: 0.0,
            cb_w,
            cb_h: if cb_h_definite { Some(cb_h) } else { None },
            abs_pct_height_base,
            shrink: false,
            width_override: None,
            width_override_final: false,
            height_override: None,
            height_is_min: false,
            collapsed_border_wins: None,
            abspos: true,
            floats: fresh_floats(),
            is_float: false,
        }));
        run_tasks(forest, &mut tasks, fonts, pw_cache, &mut scratch);
        let (fx, fy) = abs_position(
            forest,
            node,
            cb_x,
            cb_y,
            cb_w,
            cb_h,
            cb_h_definite,
            static_x,
            static_y,
            margins_pending,
        );
        let (px, py) = {
            let n = &forest.nodes[node];
            (n.rect.x, n.rect.y)
        };
        // Deferred like every other shift: the final `apply_pending_shifts`
        // moves the subtree once. `+=` keeps any shift already on the root.
        let n = &mut forest.nodes[node];
        n.pend_x += fx - px;
        n.pend_y += fy - py;
        i += 1;
    }
}

/// Resolve `top`/`right`/`bottom`/`left` against the containing-block
/// dimensions for relative/sticky offsetting. Percentages resolve against
/// the containing block: the parent's content box for in-flow boxes, the
/// viewport for the root. Percentage `top`/`bottom` compute to `auto`
/// (`None`) when the containing-block height is indefinite (CSS 2.1
/// §10.6.4); lengths always resolve.
#[allow(clippy::too_many_arguments)]
fn resolve_insets(
    off: [SpecLen; 4],
    fs: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    cb_w: f32,
    cb_h: f32,
    cb_h_definite: bool,
) -> (Option<f32>, Option<f32>, Option<f32>, Option<f32>) {
    let r = |s: SpecLen, base: f32| s.resolve(fs, root_fs, vp_w, vp_h, base);
    let rv = |s: SpecLen| {
        // A percentage (or percentage-bearing calc) of an indefinite
        // height computes to `auto`.
        if !cb_h_definite && has_pct(s) {
            None
        } else {
            r(s, cb_h)
        }
    };
    (rv(off[0]), r(off[1], cb_w), rv(off[2]), r(off[3], cb_w))
}

/// The `(dx, dy)` shift both `relative` and `sticky` boxes take from
/// their offsets: sticky is "positioned similarly to a relatively
/// positioned box" (CSS Position 3 §3.1). CSS 2.1 §10.3.4/§10.6.4: `left`
/// wins over `right` in LTR (`right` wins in RTL); `top` wins over
/// `bottom`. `rtl` is the *containing block's* direction, not the
/// element's own.
fn relative_shift(
    top: Option<f32>,
    right: Option<f32>,
    bottom: Option<f32>,
    left: Option<f32>,
    rtl: bool,
) -> (f32, f32) {
    let dx = if rtl {
        right.map(|v| -v).or(left).unwrap_or(0.0)
    } else {
        left.or(right.map(|v| -v)).unwrap_or(0.0)
    };
    let dy = top.or(bottom.map(|v| -v)).unwrap_or(0.0);
    (dx, dy)
}

/// Fold the final ancestor shifts into the static position of
/// `position: fixed` boxes on each `auto`-offset axis.
///
/// A fixed box is laid out while its out-of-flow ancestor subtree sits at
/// a provisional origin, so the static position recorded for it (used
/// when every offset on an axis is `auto`) lives in that subtree's
/// provisional frame. `apply_pending_shifts` restarts shift accumulation
/// at each fixed box — a viewport-anchored fixed box must not inherit
/// in-flow ancestor shifts — but the static position still has to move
/// with the subtree frame it was recorded in. On each `auto` axis, add
/// the ancestors' total deferred shift to the box's own pending shift.
///
/// The walk mirrors `resolve_containing_block` for fixed boxes: the
/// `apply_pending_shifts` walk already supplies the carried shift of the
/// nearest fixed-CB-establishing ancestor at the fixed box, so ancestors
/// at or above it must not be added again (that would double-count the
/// containing block's shift). The walk stops *before* the nearest such
/// ancestor (exclusive) and at the nearest fixed ancestor (inclusive: a
/// fixed ancestor's own shift moves its subtree frame). Together the
/// carried shift plus this fixup equal exactly the applied shift of the
/// fixed box's parent frame, in every ancestor configuration. Boxes with
/// an explicit offset on an axis are untouched: their position resolves
/// against the containing block and never reads the static position.
fn fix_fixed_static_frames(forest: &mut Forest) {
    for i in 0..forest.abs_items.len() {
        let node = forest.abs_items[i].node;
        let n = &forest.nodes[node];
        if !matches!(n.data.position, Position::Fixed) {
            continue;
        }
        let off = n.data.offset;
        let auto_x = matches!(off[3], SpecLen::Auto) && matches!(off[1], SpecLen::Auto);
        let auto_y = matches!(off[0], SpecLen::Auto) && matches!(off[2], SpecLen::Auto);
        if !auto_x && !auto_y {
            continue;
        }
        let (mut dx, mut dy) = (0.0, 0.0);
        let mut cur = n.parent;
        while let Some(p) = cur {
            let anc = &forest.nodes[p];
            // The containing block's shift arrives via the carried shift
            // in `apply_pending_shifts`; adding its pend (or anything
            // above it) here would count it twice.
            if has_fixed_cb_trigger(forest, p) {
                break;
            }
            dx += anc.pend_x;
            dy += anc.pend_y;
            let is_fixed = matches!(anc.data.position, Position::Fixed);
            cur = anc.parent;
            if is_fixed {
                break;
            }
        }
        let n = &mut forest.nodes[node];
        if auto_x {
            n.pend_x += dx;
        }
        if auto_y {
            n.pend_y += dy;
        }
    }
}

/// `position: relative` shifts the box by its offsets without changing
/// its flow footprint: the shift accumulates into the deferred-shift
/// fields the final `apply_pending_shifts` pass flushes. Runs after the
/// out-of-flow pass so percentage offsets resolve against containing
/// blocks that are themselves out of flow. (`sticky` takes the same
/// relative shift, but inside [`apply_sticky_constraints`], which runs
/// after `layout_forest` with final rects and a scroll offset.)
pub(crate) fn apply_relative_offsets(forest: &mut Forest) {
    let (root_fs, vp_w, vp_h) = (forest.root_font_size, forest.vp_w, forest.vp_h);
    for i in 0..forest.nodes.len() {
        if !matches!(forest.nodes[i].data.position, Position::Relative) {
            continue;
        }
        // Percentages resolve against the containing block: the parent's
        // content box for in-flow boxes, the viewport for the root.
        // Percentage `top`/`bottom` compute to `auto` (no offset) when the
        // containing-block height is indefinite (CSS 2.1 §10.6.4); lengths
        // always resolve.
        let (cb_w, cb_h, cb_h_definite, cb_rtl, fs, off) = {
            let n = &forest.nodes[i];
            let parent = n.parent;
            let (cb_w, cb_h, cb_h_definite) = match parent {
                Some(p) => {
                    let pn = &forest.nodes[p];
                    (pn.metrics.content_w, pn.content_h, pn.height_definite)
                }
                None => (vp_w, vp_h, true),
            };
            // CSS Position 3 §3.3: over-constrained `left`/`right` on a
            // relatively positioned box drops the end side per the
            // *containing block's* direction, not the element's own
            // (mirrors the `cb_rtl` fix in the block-width path); the
            // root falls back to its own direction. (Not CSS 2.1
            // §10.3.4 — that section governs absolutely positioned
            // boxes.)
            let cb_rtl = parent
                .map(|p| forest.nodes[p].data.rtl)
                .unwrap_or(n.data.rtl);
            (
                cb_w,
                cb_h,
                cb_h_definite,
                cb_rtl,
                n.data.font_size,
                n.data.offset,
            )
        };
        let (top, right, bottom, left) =
            resolve_insets(off, fs, root_fs, vp_w, vp_h, cb_w, cb_h, cb_h_definite);
        let (dx, dy) = relative_shift(top, right, bottom, left, cb_rtl);
        let n = &mut forest.nodes[i];
        n.pend_x += dx;
        n.pend_y += dy;
    }
    // A fixed box's recorded static position lives in its ancestor
    // subtree's provisional frame (see [`fix_fixed_static_frames`]); fold
    // the now-final ancestor shifts in so `auto`-offset fixed boxes land
    // at their true flow position in final coordinates.
    fix_fixed_static_frames(forest);
}

/// Whether the box can serve as the containing block for an in-flow
/// sticky box: a block container, or a box establishing a formatting
/// context (CSS 2.1 §10.1 rule 2 as corrected by the 2012-08-01 erratum:
/// "the nearest ancestor box that is a block container or which
/// establishes a formatting context"). `Block` covers `display: flex`
/// (the flex inner layout dispatches off `StyleData::display`, not the
/// box type); the table wrapper box establishes a block formatting
/// context per the same day's §17.4 erratum. Anonymous block boxes are
/// excluded: CSS 2.1 §9.2.1.1 ignores them when resolving percentages,
/// using the closest non-anonymous ancestor instead, so the constraint
/// and the inset resolution agree on one box.
fn is_sticky_cb_candidate(forest: &Forest, idx: usize) -> bool {
    let bt = forest.nodes[idx].box_type;
    if bt == BoxType::AnonymousBlock {
        return false;
    }
    matches!(
        bt,
        BoxType::Block | BoxType::InlineBlock | BoxType::TableCell | BoxType::Grid | BoxType::Table
    )
}

/// Containing-block content box `(x, y, w, h)` plus the CB node and whether
/// its height is definite, for the in-flow sticky constraint: the nearest
/// block-container ancestor's content box (see
/// [`is_sticky_cb_candidate`]), or the viewport (`None`) for the root.
/// Rects must be final (post `apply_pending_shifts`); see
/// [`apply_sticky_constraints`].
fn sticky_cb(forest: &Forest, i: usize) -> (Option<usize>, f32, f32, f32, f32, bool) {
    let mut cur = forest.nodes[i].parent;
    while let Some(p) = cur {
        if is_sticky_cb_candidate(forest, p) {
            let pn = &forest.nodes[p];
            let m = &pn.metrics;
            // The containing block is the content box: the origin is
            // the border-box origin plus border *and* padding, matching
            // the content size (`content_w`/`content_h`) returned
            // alongside it. Using only the border (the padding-box
            // origin) shifts the whole bound interval by the CB's
            // padding, so a sticky box pinned to a padded CB's end
            // (or start) lands a padding-width off, and the position
            // box's margin-edge distances are measured against the
            // wrong edge.
            // w78fix L2: the width/height pair below mixes sources
            // deliberately, and the mix is safe — it is structural,
            // not two different measurements. `BoxMetrics` carries
            // only a `content_w` (the width is resolved with the
            // containing-block width in `resolve_metrics`, once, up
            // front); there is no `content_h` field on it, because
            // the used content height resolves only after content
            // layout, in `finish_content_height`, which stamps it on
            // the node itself. `apply_sticky_constraints` runs after
            // `layout_forest`, when rects are final, so both fields
            // are the node's final used content-box size in their
            // axis. No behavior change is intended here.
            return (
                Some(p),
                pn.rect.x + m.bl + m.pl,
                pn.rect.y + m.bt + m.pt,
                m.content_w,
                pn.content_h,
                pn.height_definite,
            );
        }
        cur = forest.nodes[p].parent;
    }
    (None, 0.0, 0.0, forest.vp_w, forest.vp_h, true)
}

/// Scrollable overflow size `(w, h)` of the containing block `cb`,
/// measured from its content-box origin `(cb_x, cb_y)`: the union of the
/// content box and the border boxes of real scrollable-overflow
/// contributors, lower-bounded by the content size. Used for the sticky
/// containing-block bound when the CB is itself a scroll container:
/// Firefox/Chrome constrain the sticky box against the scrollbox
/// (`scrollWidth`/`scrollHeight`) there, not the client box, so a sticky
/// box can stick through the CB's scrolled content instead of being
/// pinned to the visible 100px (see the WPT
/// `position-sticky-*-overconstrained` suite).
///
/// Only contributors that scroll with `cb` count: in-flow boxes and
/// floats, plus absolutely positioned boxes whose containing block is
/// `cb` or below it. `position: fixed` boxes are viewport-anchored (CSS
/// Overflow 3) and are skipped unconditionally, with their subtrees; an
/// abspos box whose containing block lies outside the scroller scrolls
/// with that block instead, so its subtree is skipped as well (mirrors
/// the scrollable-overflow walk in `super::layout`).
fn cb_scrollable_size(
    forest: &Forest,
    cb: usize,
    cb_x: f32,
    cb_y: f32,
    cb_w: f32,
    cb_h: f32,
) -> (f32, f32) {
    let (mut w, mut h) = (cb_w, cb_h);
    let mut stack = vec![cb];
    while let Some(n) = stack.pop() {
        let node = &forest.nodes[n];
        if n != cb {
            if node.data.position == Position::Fixed {
                // Viewport-anchored: contributes to no ancestor's
                // scrollable overflow; the subtree is locked with it.
                continue;
            }
            if node.data.position == Position::Absolute
                && !matches!(
                    resolve_containing_block(forest, n),
                    Some(c) if at_or_below(forest, c, Some(cb))
                )
            {
                // Containing block outside this scroller: scrolls with
                // that block, not with `cb`; the subtree goes with it.
                continue;
            }
            w = w.max(node.rect.x + node.rect.w - cb_x);
            h = h.max(node.rect.y + node.rect.h - cb_y);
        }
        stack.extend(node.children.iter().copied());
    }
    (w.max(0.0), h.max(0.0))
}

/// Scrollport `(origin, size, live scroll offset)` for one axis of the
/// sticky box `i`: the content box of the nearest scroll container with
/// a matching scrollable axis (see [`sticky_scroller`]), or the
/// viewport's when the box scrolls with the viewport on that axis. The
/// scroll offset is measured from the content origin, so the view
/// rectangle's origin in document coordinates is the content-box
/// origin, not the padding-box origin: a scroller's padding offsets
/// the stuck position by that padding (WPT
/// `position-sticky-overflow-padding`: with 20px top padding a
/// `top: 0` box sticks at scroll + 20, and at scroll 315 the
/// containing-block bound, not the scrollport, pins it at 320). The
/// nested scroller's offset comes from `nested_offsets` (see
/// [`nested_scroll_offset`]); inside a `fixed` subtree with no scroll
/// container of its own the viewport frame constrains at a zero offset
/// (see [`apply_sticky_constraints`]).
#[allow(clippy::too_many_arguments)]
fn sticky_scrollport_axis(
    forest: &Forest,
    i: usize,
    fixed_limit: Option<usize>,
    nested_offsets: &HashMap<bowser_dom::NodeId, (f32, f32)>,
    horizontal: bool,
    doc_scroll: f32,
    vp_size: f32,
) -> (f32, f32, f32) {
    match sticky_scroller(forest, i, fixed_limit, horizontal) {
        Some(p) => {
            let n = &forest.nodes[p];
            let m = &n.metrics;
            let (sx, sy) = nested_scroll_offset(forest, p, nested_offsets);
            let s = if horizontal { sx } else { sy };
            if horizontal {
                (
                    n.rect.x + m.bl + m.pl,
                    (n.rect.w - m.bl - m.br - m.pl - m.pr).max(0.0),
                    s,
                )
            } else {
                (
                    n.rect.y + m.bt + m.pt,
                    (n.rect.h - m.bt - m.bb - m.pt - m.pb).max(0.0),
                    s,
                )
            }
        }
        None if fixed_limit.is_some() => (0.0, vp_size, 0.0),
        None => (0.0, vp_size, doc_scroll),
    }
}

/// Clamp `v` into `[lo, hi]`; when the range is over-constrained
/// (`lo > hi`) the start side wins (`lo`, or `hi` when the start side is
/// the trailing edge, e.g. horizontal in RTL).
fn clamp_start_wins(v: f32, lo: f32, hi: f32, start_is_lo: bool) -> f32 {
    if lo > hi {
        if start_is_lo { lo } else { hi }
    } else {
        v.clamp(lo, hi)
    }
}

/// The nearest `position: fixed` ancestor of `idx`, if any: its subtree
/// is viewport-locked, so the sticky constraint's scroll-container walk
/// is bounded by it (see [`apply_sticky_constraints`]).
fn nearest_fixed_ancestor(forest: &Forest, idx: usize) -> Option<usize> {
    let mut cur = forest.nodes[idx].parent;
    while let Some(p) = cur {
        if matches!(forest.nodes[p].data.position, Position::Fixed) {
            return Some(p);
        }
        cur = forest.nodes[p].parent;
    }
    None
}

/// Whether `c` is `limit` itself or a descendant of it (`true` when
/// there is no limit).
fn at_or_below(forest: &Forest, mut c: usize, limit: Option<usize>) -> bool {
    let Some(limit) = limit else {
        return true;
    };
    loop {
        if c == limit {
            return true;
        }
        match forest.nodes[c].parent {
            Some(p) => c = p,
            None => return false,
        }
    }
}

/// Shift a subtree's final rects by `(dx, dy)`, text fragments included.
/// Deeper `fixed` subtrees are viewport-locked and do not inherit the
/// shift, mirroring `apply_pending_shifts` in `super::layout`.
///
/// A fixed box under a fixed-CB trigger is the exception: its
/// containing block is that ancestor (CSS Transforms 1 §3 et al.), so it
/// must ride that ancestor's applied shift. The walk carries the applied
/// shift of the nearest such ancestor down the same way
/// `apply_pending_shifts` does (see its carried-frame docs); a deeper
/// fixed subtree restarts from the carried shift instead of being
/// skipped, and its descendants move rigidly with it. With no trigger
/// ancestor on the walk the carry is `(0, 0)`, which is exactly the old
/// skip behavior. The carry is initialized from the walk root only:
/// trigger ancestors above the walk root did not move and must not feed
/// it.
fn shift_subtree(forest: &mut Forest, idx: usize, dx: f32, dy: f32) {
    // Stack entries: (node, shift to apply, carried containing-block shift).
    let mut stack = vec![(idx, dx, dy, 0.0f32, 0.0f32)];
    while let Some((i, ox, oy, cx, cy)) = stack.pop() {
        let (ax, ay, ncx, ncy, children) = {
            let n = &mut forest.nodes[i];
            // The walk root always shifts; deeper `fixed` subtrees are
            // viewport-locked unless they ride a fixed-CB containing
            // block (see doc comment).
            let (ax, ay) = if i != idx && matches!(n.data.position, Position::Fixed) {
                (cx, cy)
            } else {
                (ox, oy)
            };
            n.rect.x += ax;
            n.rect.y += ay;
            // Paint reads fragments, so they move with their boxes.
            for f in n.fragments.iter_mut() {
                f.rect.x += ax;
                f.rect.y += ay;
                f.baseline += ay;
            }
            // The containing-block frame the children see: this box's own
            // applied shift when it establishes a fixed containing
            // block, else the one carried down.
            let (ncx, ncy) = if n.style.as_deref().is_some_and(crate::establishes_fixed_cb) {
                (ax, ay)
            } else {
                (cx, cy)
            };
            (ax, ay, ncx, ncy, n.children.clone())
        };
        // Children move rigidly with their parent: the shift applied to
        // this node is what they inherit.
        for c in children {
            stack.push((c, ax, ay, ncx, ncy));
        }
    }
}

/// Whether the box is a scroll container on the given axis, using the same
/// canonical `overflow` classification as [`crate::scroll_containers`] so
/// the two cannot diverge. In particular `clip` is not a scroll container
/// on that axis (CSS Overflow 3 §2).
fn is_scroll_container_axis(forest: &Forest, idx: usize, horizontal: bool) -> bool {
    forest.nodes[idx].style.as_ref().is_some_and(|s| {
        let ((cx, cy), _) = crate::overflow_axes(s);
        if horizontal { cx } else { cy }
    })
}

/// Nearest ancestor scroll container of the sticky box `i` with a matching
/// scrollable axis, if any: the first such scroll container on the box's
/// *anchoring chain* — its DOM ancestors, except that crossing an
/// `absolute` ancestor re-anchors the walk at that box's containing block
/// ([`resolve_containing_block`]): an out-of-flow subtree is positioned
/// against its containing block, so it scrolls with the scroll
/// containers of the block's ancestor chain, not with containers that
/// sit between the subtree and its block in DOM order (mirrors the
/// `cb_inside`/`fixed_cb_inside` tracking in
/// `crate::scrollable_extent`). CSS Position 3 §3.4 constrains each axis
/// against the nearest scroll container *with a matching scrollable
/// axis*, so the two axes may resolve to different scrollports (a
/// single-axis scroll container only matches its scrollable axis).
/// `None` when the box scrolls with the viewport on that axis (or, under
/// a `fixed` limit, with the viewport frame: see below).
///
/// `fixed_limit` is the nearest `fixed` ancestor of `i`, if any: the
/// walk never looks above it (a fixed subtree is viewport-locked and
/// does not scroll with anything above it), though the fixed box itself
/// may be the scroll container. An `absolute` ancestor's containing
/// block always sits at or below the fixed limit — the fixed ancestor
/// is itself positioned, so it bounds the nearest-positioned-ancestor
/// search — but the walk checks defensively and falls back to `None`
/// rather than escaping the limit.
fn sticky_scroller(
    forest: &Forest,
    i: usize,
    fixed_limit: Option<usize>,
    horizontal: bool,
) -> Option<usize> {
    let mut cur = forest.nodes[i].parent;
    while let Some(p) = cur {
        if is_scroll_container_axis(forest, p, horizontal) {
            return Some(p);
        }
        if forest.nodes[p].data.position == Position::Absolute {
            // Re-anchor at the containing block: scroll containers
            // between the subtree and its block do not move it.
            match resolve_containing_block(forest, p) {
                Some(c) if at_or_below(forest, c, fixed_limit) => {
                    cur = Some(c);
                    continue;
                }
                _ => return None,
            }
        }
        if Some(p) == fixed_limit {
            return None;
        }
        cur = forest.nodes[p].parent;
    }
    None
}

/// One side's effective margin for the sticky "position box" (CSS
/// Position 3 §3.4): `margin`, unless the margin edge lies within
/// `margin` of the corresponding containing-block edge, in which case
/// that distance is used instead. This collapses remainder-absorbing
/// used margins — e.g. the CSS 2.1 §10.3.3 end margin that fills the
/// containing block for a fixed-width block-level box — so they cannot
/// pin the box, while real specified margins still constrain.
fn sticky_eff_margin(margin_edge: f32, cb_edge: f32, margin: f32) -> f32 {
    let d = (margin_edge - cb_edge).abs();
    if d < margin { d } else { margin }
}

/// Live scroll offset of the nested scroller `p` for the sticky
/// constraint: the caller's `nested_offsets` entry when present, else
/// zero.
///
/// Layout owns no per-scroller scroll state (it never scrolls nested
/// containers itself), so the map is the only source of live offsets.
/// The fallback is exact, not an approximation: the caller guarantees
/// every nested scroll container with a nonzero offset has an entry
/// (see `ScrollTree::nested_offsets`, which omits only zero-offset
/// containers), so a missing entry always means offset zero.
fn nested_scroll_offset(
    forest: &Forest,
    p: usize,
    nested_offsets: &HashMap<bowser_dom::NodeId, (f32, f32)>,
) -> (f32, f32) {
    forest.nodes[p]
        .dom
        .and_then(|d| nested_offsets.get(&d).copied())
        .unwrap_or((0.0, 0.0))
}

/// Sticky positioning (CSS Position 3 §3.4).
///
/// Must run after `layout_forest` — rects are final document coordinates
/// and every deferred shift is flushed — so the constraint is computed
/// directly in document space and applied by shifting the subtree (see
/// [`shift_subtree`]). Called from `crate::layout_with_scroll` (and its
/// `_and_offsets` variants); the plain `crate::layout` entry keeps a
/// zero scroll offset, under which an unstuck sticky box stays exactly
/// where it was laid out.
///
/// Each sticky box keeps its in-flow footprint and its laid-out position;
/// the insets only constrain it — they are *not* an up-front relative
/// shift (CSS Position 3 §3.4: "the element stays where it was laid
/// out"; cf. csswg/csswg-drafts#11298). Per axis the constraint is:
/// 1. the sticky view rectangle: the scrollport of the nearest scroll
///    container with a matching scrollable axis on the box's anchoring
///    chain (see [`sticky_scroller`]: the walk re-anchors at out-of-flow
///    ancestors' containing blocks, so a sticky box inside an `absolute`
///    subtree still constrains against the scroller that actually moves
///    it; the two axes may resolve to different scrollports), or the
///    viewport's when the box scrolls with the viewport — inset by the
///    sticky offsets (an `auto` inset is a zero inset, unless both in an
///    axis are `auto`, in which case the axis is unconstrained). The
///    box's *border* box is clamped into it: it sticks when scrolling
///    would move a border edge outside the rectangle, and unsticks when
///    scrolled back. Over-constrained ranges resolve to the start side —
///    the `top` edge always, the `left` edge in LTR and the `right` edge
///    in RTL (the containing block's writing mode). The nested scroller's
///    offset comes from `nested_offsets`, keyed by the scroller's DOM
///    node (arena indices are not stable across layouts); a scroller
///    with no entry constrains at offset zero — exact, not an
///    approximation (see [`nested_scroll_offset`]). Inside a `fixed`
///    subtree the walk is bounded by the fixed ancestor: scroll
///    containers within the subtree (including the fixed box itself)
///    still constrain at their live offsets, but with none the box
///    resolves statically against the viewport frame at a zero offset —
///    the subtree is viewport-locked, and in this layout's
///    document-coordinate model its frame already sits at the unscrolled
///    position, so scrolling the document cannot move it and only the
///    containing-block bound applies;
/// 2. hard-bounded by its containing block (see [`sticky_cb`]), applied
///    to the "position box" (the margin box, except a side whose margin
///    edge lies within its own margin of the containing-block edge uses
///    that distance instead): when the containing block is itself a
///    scroll container the bound uses its scrollable overflow size (see
///    [`cb_scrollable_size`]), and when the position box exceeds the
///    bound the proviso is vacuous and skipped — so a sticky box never
///    leaves its containing block, but an over-constrained one still
///    sticks through its scrolled content.
pub(crate) fn apply_sticky_constraints(
    forest: &mut Forest,
    scroll_x: f32,
    scroll_y: f32,
    nested_offsets: &HashMap<bowser_dom::NodeId, (f32, f32)>,
) {
    let (root_fs, vp_w, vp_h) = (forest.root_font_size, forest.vp_w, forest.vp_h);
    // Scrollable-overflow sizes of scroll-container containing blocks,
    // memoized per CB node (see `cb_scrollable_size`).
    let mut scrollable_cache: HashMap<usize, (f32, f32)> = HashMap::new();
    for i in 0..forest.nodes.len() {
        if !matches!(forest.nodes[i].data.position, Position::Sticky) {
            continue;
        }
        // A `fixed` ancestor bounds the scroll-container walk (see the
        // fn docs, step 1): the subtree is viewport-locked, so nothing
        // above the fixed box can scroll it.
        let fixed_limit = nearest_fixed_ancestor(forest, i);
        let (
            cb_node,
            cb_x,
            cb_y,
            cb_w,
            cb_h,
            cb_h_definite,
            fs,
            rtl,
            off,
            bw,
            bh,
            ml_u,
            mr_u,
            mt_u,
            mb_u,
            rx,
            ry,
        ) = {
            let n = &forest.nodes[i];
            let (cb_node, cb_x, cb_y, cb_w, cb_h, cb_h_definite) = sticky_cb(forest, i);
            let m = &n.metrics;
            (
                cb_node,
                cb_x,
                cb_y,
                cb_w,
                cb_h,
                cb_h_definite,
                n.data.font_size,
                n.data.rtl,
                n.data.offset,
                n.rect.w,
                n.rect.h,
                m.ml,
                m.mr,
                m.mt,
                m.mb,
                n.rect.x,
                n.rect.y,
            )
        };
        // The over-constrained "start wins" rule is interpreted in the
        // containing block's writing mode (CSS Position 3 §3.4); with
        // the viewport as CB fall back to the box's own direction, which
        // it inherits from the root.
        let cb_rtl = cb_node.map(|c| forest.nodes[c].data.rtl).unwrap_or(rtl);
        let (top, right, bottom, left) =
            resolve_insets(off, fs, root_fs, vp_w, vp_h, cb_w, cb_h, cb_h_definite);
        // The box starts at its laid-out position: the insets constrain
        // it, they do not shift it.
        let (px, py) = (rx, ry);
        // The "position box" (CSS Position 3 §3.4): the margin box,
        // except a side whose margin edge lies within its own margin of
        // the corresponding containing-block edge uses that distance
        // instead of the margin. This collapses remainder-absorbing used
        // margins (e.g. the §10.3.3 end margin filling the containing
        // block) so they cannot pin the box. The effective margins are
        // evaluated at the laid-out position; the bottom edge uses
        // +infinity when the block height is indefinite, keeping the
        // full bottom margin.
        let cb_bottom = if cb_h_definite {
            cb_y + cb_h
        } else {
            f32::INFINITY
        };
        let e_ml = sticky_eff_margin(px - ml_u, cb_x, ml_u);
        let e_mt = sticky_eff_margin(py - mt_u, cb_y, mt_u);
        let e_mr = sticky_eff_margin(px + bw + mr_u, cb_x + cb_w, mr_u);
        let e_mb = sticky_eff_margin(py + bh + mb_u, cb_bottom, mb_u);
        let (mw, mh) = (bw + e_ml + e_mr, bh + e_mt + e_mb);
        // 1. scrollport (sticky view rectangle) constraint on the border
        //    box, per axis against the nearest scroll container with a
        //    matching scrollable axis (see [`sticky_scroller`]). An axis
        //    whose insets are both `auto` is unconstrained: the box stays
        //    where it was laid out there.
        let (sp_x, sp_w, sp_sx) =
            sticky_scrollport_axis(forest, i, fixed_limit, nested_offsets, true, scroll_x, vp_w);
        let (sp_y, sp_h, sp_sy) = sticky_scrollport_axis(
            forest,
            i,
            fixed_limit,
            nested_offsets,
            false,
            scroll_y,
            vp_h,
        );
        let qx = if left.is_none() && right.is_none() {
            px
        } else {
            let s_lo_x = left.map(|l| sp_x + sp_sx + l).unwrap_or(f32::NEG_INFINITY);
            let s_hi_x = right
                .map(|r| sp_x + sp_sx + sp_w - r - bw)
                .unwrap_or(f32::INFINITY);
            clamp_start_wins(px, s_lo_x, s_hi_x, !cb_rtl)
        };
        let qy = if top.is_none() && bottom.is_none() {
            py
        } else {
            let s_lo_y = top.map(|t| sp_y + sp_sy + t).unwrap_or(f32::NEG_INFINITY);
            let s_hi_y = bottom
                .map(|b| sp_y + sp_sy + sp_h - b - bh)
                .unwrap_or(f32::INFINITY);
            clamp_start_wins(py, s_lo_y, s_hi_y, true)
        };
        // 2. containing-block bound on the position box (see [`sticky_cb`]).
        //    When the CB is itself a scroll container the bound uses its
        //    scrollable overflow size (see [`cb_scrollable_size`]); when
        //    the position box exceeds the bound the proviso is vacuous
        //    and the axis is skipped.
        let (bound_w, bound_h) = match cb_node {
            Some(c) => {
                let sx = is_scroll_container_axis(forest, c, true);
                let sy = is_scroll_container_axis(forest, c, false);
                if sx || sy {
                    let (sw, sh) = *scrollable_cache
                        .entry(c)
                        .or_insert_with(|| cb_scrollable_size(forest, c, cb_x, cb_y, cb_w, cb_h));
                    (if sx { sw } else { cb_w }, if sy { sh } else { cb_h })
                } else {
                    (cb_w, cb_h)
                }
            }
            None => (vp_w, vp_h),
        };
        let fx = if left.is_none() && right.is_none() || mw > bound_w {
            qx
        } else {
            clamp_start_wins(qx - e_ml, cb_x, cb_x + bound_w - mw, !cb_rtl) + e_ml
        };
        let fy = if top.is_none() && bottom.is_none() || cb_h_definite && mh > bound_h {
            qy
        } else {
            let c_hi_y = if cb_h_definite {
                cb_y + bound_h - mh
            } else {
                f32::INFINITY
            };
            clamp_start_wins(qy - e_mt, cb_y, c_hi_y, true) + e_mt
        };
        // Back to the border-box origin for the subtree shift.
        let (ddx, ddy) = (fx - rx, fy - ry);
        if ddx != 0.0 || ddy != 0.0 {
            shift_subtree(forest, i, ddx, ddy);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        BoxType, FixedFontMetrics, IntrinsicDims, LayoutBox, Position, Size, layout,
        layout_with_intrinsics, layout_with_scroll, layout_with_scroll_and_offsets,
    };
    use std::collections::HashMap;

    const VP: Size = Size { w: 800.0, h: 600.0 };

    /// A raw box-tree forest for tests that need to corrupt the tree
    /// itself (e.g. stitching in a parent cycle no constructor would
    /// produce). Layout pipeline state, not a laid-out document.
    fn forest_of(html: &str, css: &str) -> crate::boxes::Forest {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let ua = crate::ua_stylesheet();
        crate::boxes::build_box_tree(&doc, &[&ss], ua, VP.w, VP.h, &HashMap::new())
            .expect("box tree builds")
            .0
    }

    #[test]
    fn resolve_containing_block_parent_cycle_terminates() {
        // A parent cycle is unreachable from well-formed box
        // construction (only via an internal anonymous-wrapping/table
        // fix-up bug); without the hang guard the walk would spin
        // forever. The guard must fail closed: here no node on the
        // cyclic chain establishes a containing block, so the initial
        // containing block wins.
        let mut forest = forest_of(
            "<html><body><div class=\"wrap\"><div class=\"abs\"></div></div></body></html>",
            ".abs { position: absolute; }",
        );
        let abs = forest
            .nodes
            .iter()
            .position(|n| n.data.position == Position::Absolute)
            .expect("abspos box");
        let wrap = forest.nodes[abs].parent.expect("abspos has a parent");
        let body = forest.nodes[wrap].parent.expect("wrap has a parent");
        // Stitch a 2-cycle between the wrap and the body; the walk
        // bounces between them and must still terminate.
        forest.nodes[body].parent = Some(wrap);
        assert_eq!(super::resolve_containing_block(&forest, abs), None);
        // Sanity: a positioned ancestor on the acyclic prefix still
        // resolves normally — the guard only stops the unbounded case.
        let forest2 = forest_of(
            "<html><body><div class=\"rel\"><div class=\"abs\"></div></div></body></html>",
            ".rel { position: relative; } .abs { position: absolute; }",
        );
        let abs2 = forest2
            .nodes
            .iter()
            .position(|n| n.data.position == Position::Absolute)
            .expect("abspos box");
        let rel = forest2
            .nodes
            .iter()
            .position(|n| n.data.position == Position::Relative)
            .expect("relative box");
        assert_eq!(super::resolve_containing_block(&forest2, abs2), Some(rel));
    }

    fn layout_scrolled(html: &str, css: &str, scroll: (f32, f32)) -> LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        layout_with_scroll(&doc, &[&ss], VP, &FixedFontMetrics, scroll)
    }

    /// The `<body>` box (`<head>` is `display: none`, so it is the root's
    /// first child).
    fn body(l: &LayoutBox) -> &LayoutBox {
        &l.children[0]
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    const TOP_CSS: &str = ".spacer { height: 100px; } \
         .sticky { position: sticky; top: 10px; height: 50px; } \
         .filler { height: 2000px; }";
    const TOP_HTML: &str =
        r#"<div class="spacer"></div><div class="sticky"></div><div class="filler"></div>"#;

    #[test]
    fn zero_space_between_floats_bfc_full_layout() {
        // WPT css/CSS2/floats/zero-space-between-floats-00{1,2,3}.html:
        // the zero-width BFC fits in the zero-width gap between the
        // floats. The WPT checkLayout() reads mid-parse (parse-shim
        // artifact, cataloged in CSS2.ini); the full post-parse layout
        // here is correct.
        for (n, html, expect) in [
            (
                "001",
                "<div id=\"container\" style=\"position:relative; width:200px;\"><div style=\"float:left; width:100px; height:200px;\"></div><div style=\"float:right; width:100px; height:200px;\"></div><div style=\"overflow:hidden; width:0; height:200px;\"></div></div>",
                (100.0, 0.0),
            ),
            (
                "002",
                "<div id=\"container\" style=\"position:relative; width:200px;\"><div style=\"float:left; width:100px; height:200px;\"></div><div style=\"float:right; width:100px; height:200px;\"></div><div style=\"float:right; width:100%; height:10px;\"></div><div style=\"overflow:hidden; width:0; height:200px;\"></div></div>",
                (100.0, 0.0),
            ),
            (
                "003",
                "<div id=\"container\" style=\"position:relative; width:200px;\"><div style=\"float:left; width:100px; height:100px;\"></div><div style=\"float:right; width:100px; height:100px;\"></div><div style=\"float:right; width:100%; height:100px;\"></div><div style=\"overflow:hidden; clear:left; width:0;\"></div></div>",
                (0.0, 100.0),
            ),
        ] {
            let l = layout_scrolled(html, "", (0.0, 0.0));
            fn find_bfc(b: &LayoutBox, depth: usize) -> Option<&LayoutBox> {
                if depth >= 3 && b.box_type == BoxType::Block && b.rect.w == 0.0 {
                    return Some(b);
                }
                b.children.iter().find_map(|c| find_bfc(c, depth + 1))
            }
            let bfc = find_bfc(&l, 0).expect("bfc div");
            assert!(
                approx(bfc.rect.x, expect.0) && approx(bfc.rect.y, expect.1),
                "{n}: bfc at ({}, {}), expected ({}, {})",
                bfc.rect.x,
                bfc.rect.y,
                expect.0,
                expect.1
            );
        }
    }

    #[test]
    fn sticky_top_sticks_when_scrolled_past() {
        let l = layout_scrolled(TOP_HTML, TOP_CSS, (0.0, 200.0));
        let sticky = &body(&l).children[1];
        // Natural y=100; scroll 200 pushes the inset top edge to 210, so
        // the box sticks 10px below the viewport top.
        assert!(approx(sticky.rect.y, 210.0), "y={}", sticky.rect.y);
        assert!(approx(sticky.rect.x, 0.0), "x={}", sticky.rect.x);
    }

    #[test]
    fn sticky_top_unsticks_when_scrolled_back() {
        let l = layout_scrolled(TOP_HTML, TOP_CSS, (0.0, 0.0));
        let sticky = &body(&l).children[1];
        // Scroll back: the inset top edge (10) is above the box, so it
        // rests exactly where it was laid out (100): the inset only
        // constrains, it never shifts the box.
        assert!(approx(sticky.rect.y, 100.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_top_does_not_stick_before_threshold() {
        let l = layout_scrolled(TOP_HTML, TOP_CSS, (0.0, 50.0));
        let sticky = &body(&l).children[1];
        // Inset top edge at 60 < 100: not yet stuck.
        assert!(approx(sticky.rect.y, 100.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_zero_scroll_matches_plain_layout_entry() {
        // `layout` (no scroll offset): an unstuck sticky box stays
        // exactly where it was laid out.
        let (doc, _) = bowser_dom::Document::parse_html(TOP_HTML.as_bytes());
        let ss = bowser_css::parse_stylesheet(TOP_CSS);
        let l = layout(&doc, &[&ss], VP, &FixedFontMetrics);
        let sticky = &body(&l).children[1];
        assert!(approx(sticky.rect.y, 100.0), "y={}", sticky.rect.y);
    }

    const BOTTOM_CSS: &str = ".spacer { height: 700px; } \
         .sticky { position: sticky; bottom: 20px; height: 50px; } \
         .filler { height: 2000px; }";

    #[test]
    fn sticky_bottom_sticks_to_viewport_bottom() {
        let l = layout_scrolled(TOP_HTML, BOTTOM_CSS, (0.0, 0.0));
        let sticky = &body(&l).children[1];
        // Natural y=700; the inset bottom edge is at 600-20-50=530, so
        // the box sticks 20px above the viewport bottom.
        assert!(approx(sticky.rect.y, 530.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_bottom_unsticks_when_scrolled_down() {
        let l = layout_scrolled(TOP_HTML, BOTTOM_CSS, (0.0, 170.0));
        let sticky = &body(&l).children[1];
        // Inset bottom edge at 170+600-20-50=700: the box is back at its
        // laid-out position (700).
        assert!(approx(sticky.rect.y, 700.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_never_leaves_its_containing_block() {
        let html =
            r#"<div class="wrap"><div class="sticky"></div></div><div class="filler"></div>"#;
        let css = ".wrap { height: 200px; } \
             .sticky { position: sticky; top: 10px; height: 50px; } \
             .filler { height: 2000px; }";
        let l = layout_scrolled(html, css, (0.0, 500.0));
        let sticky = &body(&l).children[0].children[0];
        // Scroll alone would stick the box at 510, but the 200px parent
        // ends at 200: the box sticks to the parent's bottom edge
        // (200-50=150) and scrolls away with it.
        assert!(approx(sticky.rect.y, 150.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_left_sticks_when_scrolled_horizontally() {
        // Also pins the CSS Position 3 §3.4 "position box" refinement:
        // the fixed-width block's §10.3.3 remainder-absorbing used
        // right margin (1950px) collapses to the 10px distance from its
        // margin edge to the containing block's edge, so the box sticks
        // instead of being pinned at x=0 by the full used margin box.
        let html = r#"<div class="wide"><div class="sticky"></div></div>"#;
        let css = ".wide { width: 2000px; height: 50px; } \
             .sticky { position: sticky; left: 10px; width: 50px; height: 50px; }";
        let l = layout_scrolled(html, css, (500.0, 0.0));
        let sticky = &body(&l).children[0].children[0];
        // Natural x=0; scroll 500 pushes the inset left edge to 510.
        assert!(approx(sticky.rect.x, 510.0), "x={}", sticky.rect.x);
        assert!(approx(sticky.rect.y, 0.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_position_box_keeps_specified_margins() {
        // The §3.4 refinement only collapses a side whose margin edge is
        // within its own margin of the CB edge: the specified
        // `margin-top: 20px` (far from the edge) is kept for the
        // containing-block bound, while the remainder-absorbing used
        // right margin still collapses. The scrollport constraint,
        // however, applies to the border box: with `top: 10px` the box
        // sticks with its border edge 10px below the viewport top
        // (200 + 10 = 210) — the margin is irrelevant whilst stuck
        // (WPT css/css-position/position-sticky-margins.html).
        let html = r#"<div class="wide"><div class="spacer"></div><div class="sticky"></div><div class="filler"></div></div>"#;
        let css = ".wide { width: 2000px; } .spacer { height: 100px; } \
             .sticky { position: sticky; left: 10px; top: 10px; width: 50px; height: 50px; margin-top: 20px; } \
             .filler { height: 2000px; }";
        let l = layout_scrolled(html, css, (500.0, 200.0));
        let sticky = &body(&l).children[0].children[1];
        assert!(approx(sticky.rect.x, 510.0), "x={}", sticky.rect.x);
        assert!(approx(sticky.rect.y, 210.0), "y={}", sticky.rect.y);
    }

    fn layout_plain(html: &str, css: &str) -> LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        layout(&doc, &[&ss], VP, &FixedFontMetrics)
    }

    #[test]
    fn abspos_negative_imcb_clamps_weaker_inset() {
        // css-position-3 §4.2: `inset: 18px` in a 20px container makes
        // the inset-modified containing block negative-sized, so the
        // weaker (end) inset is adjusted to make it zero-sized. With
        // `margin: auto` and an 18px box: horizontal remainder
        // 20-18-2-18 = -18 pins the start margin to 0 (x = 18);
        // vertical remainder -18 splits evenly (y = 18-9 = 9).
        // (WPT css/css-position/position-absolute-with-negative-sized-imcb.html,
        // case 1.)
        let html = r#"<div class="c"><div class="a"></div></div>"#;
        let css = "body { margin: 0; } \
             .c { position: relative; width: 20px; height: 20px; } \
             .a { position: absolute; margin: auto; width: 18px; height: 18px; inset: 18px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.x, 18.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 9.0), "y={}", a.rect.y);
        // RTL: the weaker inset is `left`; the box's right edge pins to
        // the adjusted inset (x = 20-18-18 = -16).
        let css_rtl = "body { margin: 0; } \
             .c { position: relative; width: 20px; height: 20px; direction: rtl; } \
             .a { position: absolute; margin: auto; width: 18px; height: 18px; inset: 18px; }";
        let l = layout_plain(html, css_rtl);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.x, -16.0), "rtl x={}", a.rect.x);
        assert!(approx(a.rect.y, 9.0), "rtl y={}", a.rect.y);
    }

    // --- abspos self-alignment within the inset-modified containing
    //     block (css-position-3 §4.3) ---

    #[test]
    fn abspos_self_align_center_within_imcb() {
        // Both insets specified and no `auto` margins: the margin box
        // aligns within the inset-modified containing block per
        // `justify-self`/`align-self`. 20px CB, `left: 2px; right: 2px`
        // (IMCB 2..18), 10px box, `place-self: center` → centered at 5.
        let html = r#"<div class="c"><div class="a"></div></div>"#;
        let css = "body { margin: 0; } \
             .c { position: relative; width: 20px; height: 20px; } \
             .a { position: absolute; left: 2px; right: 2px; top: 2px; bottom: 2px; \
                  width: 10px; height: 10px; place-self: center; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.x, 5.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 5.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_self_align_end_within_imcb() {
        // `justify-self: end` / `align-self: end` pin the margin box to
        // the IMCB end: 2 + 16 - 10 = 8.
        let html = r#"<div class="c"><div class="a"></div></div>"#;
        let css = "body { margin: 0; } \
             .c { position: relative; width: 20px; height: 20px; } \
             .a { position: absolute; left: 2px; right: 2px; top: 2px; bottom: 2px; \
                  width: 10px; height: 10px; justify-self: end; align-self: end; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.x, 8.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 8.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_self_align_start_rtl_uses_end_edge() {
        // In RTL `justify-self: start` is the right edge: 2 + 16 - 10 = 8.
        let html = r#"<div class="c"><div class="a"></div></div>"#;
        let css = "body { margin: 0; } \
             .c { position: relative; width: 20px; height: 20px; direction: rtl; } \
             .a { position: absolute; left: 2px; right: 2px; top: 2px; bottom: 2px; \
                  width: 10px; height: 10px; justify-self: start; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.x, 8.0), "rtl x={}", a.rect.x);
    }

    #[test]
    fn abspos_self_align_normal_keeps_legacy_position() {
        // No self-alignment: the weaker inset resolves (§3.5.2) exactly
        // as before — LTR pins to the start inset, RTL to the end inset.
        let html = r#"<div class="c"><div class="a"></div></div>"#;
        let css = "body { margin: 0; } \
             .c { position: relative; width: 20px; height: 20px; } \
             .a { position: absolute; left: 2px; right: 2px; top: 2px; bottom: 2px; \
                  width: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.x, 2.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 2.0), "y={}", a.rect.y);
        let css_rtl = "body { margin: 0; } \
             .c { position: relative; width: 20px; height: 20px; direction: rtl; } \
             .a { position: absolute; left: 2px; right: 2px; top: 2px; bottom: 2px; \
                  width: 10px; height: 10px; }";
        let l = layout_plain(html, css_rtl);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.x, 8.0), "rtl x={}", a.rect.x);
    }

    #[test]
    fn abspos_align_axis_overflow_modes() {
        // Zero-size IMCB at 10, 10px margin box: `unsafe center` honors
        // the centering (5); without an explicit `unsafe` the overflowing
        // box is adjusted to minimize overflow (start, 10); `unsafe end`
        // pins the margin-box end to the IMCB end (0).
        use super::{AbsSelfAlign, align_abspos_axis};
        assert!(approx(
            align_abspos_axis(10.0, 0.0, 0.0, 10.0, 0.0, AbsSelfAlign::Center, true),
            5.0
        ));
        assert!(approx(
            align_abspos_axis(10.0, 0.0, 0.0, 10.0, 0.0, AbsSelfAlign::Center, false),
            10.0
        ));
        assert!(approx(
            align_abspos_axis(10.0, 0.0, 0.0, 10.0, 0.0, AbsSelfAlign::End, true),
            0.0
        ));
        // Specified margins ride along: the margin box (1 + 10 + 3)
        // centers in a 20px IMCB at 3, so the border box sits at 4.
        assert!(approx(
            align_abspos_axis(0.0, 20.0, 1.0, 10.0, 3.0, AbsSelfAlign::Center, false),
            4.0
        ));
    }

    // --- abspos rects are final in the published tree (scrollable
    //     overflow consumers read them downstream) ---

    #[test]
    fn abspos_final_rect_past_cb_edge() {
        // WPT css-position/position-absolute-scrollable-overflow-001.html
        // case 1: the abspos box overflows its containing block; the
        // published rect must carry the final (post-`abs_position`)
        // geometry, not the provisional origin.
        let html = r#"<div class="cb"><div class="a"></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { position: relative; width: 100px; height: 100px; } \
             .a { position: absolute; left: -50px; top: 0; width: 10px; height: 500px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.x, -50.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 0.0), "y={}", a.rect.y);
        assert!(approx(a.rect.w, 10.0), "w={}", a.rect.w);
        assert!(approx(a.rect.h, 500.0), "h={}", a.rect.h);
    }

    // --- abspos replaced min/max constraint probe (intrinsics present) ---

    #[test]
    fn abspos_replaced_minmax_probe() {
        // WPT css-position/position-absolute-replaced-minmax.html spec 1
        // (img 200x150, max-width: 180px): width clamps to 180, height
        // transfers through the ratio to 135; the inset equation then
        // re-solves against the clamped size (left auto -> solve).
        let html = r#"<div class="c"><img id="t"></div>"#;
        let css = "body { margin: 0; } \
             .c { position: relative; width: 400px; height: 200px; } \
             #t { position: absolute; padding: 10px; bottom: 0; right: 0; \
                  border-width: 3px 7px 9px 11px; border-style: solid; \
                  margin: 1px 3px 7px 9px; max-width: 180px; }";
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let t = doc.get_element_by_id("t").expect("img node");
        let ss = bowser_css::parse_stylesheet(css);
        let intrinsics = HashMap::from([(t, IntrinsicDims::from_size(200.0, 150.0))]);
        let l = layout_with_intrinsics(&doc, &[&ss], VP, &FixedFontMetrics, &intrinsics);
        let img = &body(&l).children[0].children[0];
        // border box: 180 + 20 + 18 = 218 wide, 135 + 20 + 12 = 167 tall.
        assert!(approx(img.rect.w, 218.0), "w={}", img.rect.w);
        assert!(approx(img.rect.h, 167.0), "h={}", img.rect.h);
        // left = 400 - 9 - 218 - 3 = 170 -> border x 179;
        // top = 200 - 1 - 167 - 7 = 25 -> border y 26.
        assert!(approx(img.rect.x, 179.0), "x={}", img.rect.x);
        assert!(approx(img.rect.y, 26.0), "y={}", img.rect.y);
    }

    // --- abspos scrollable-overflow discovery probe ---

    #[test]
    fn abspos_overflow_discovered_by_scroll_containers() {
        // layout-side half of WPT
        // css-position/position-absolute-scrollable-overflow-001.html:
        // the abspos child's final rect must feed the scroll container's
        // scrollable overflow region.
        let html = r#"<div class="cb"><div class="a"></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { position: relative; width: 100px; height: 100px; overflow: auto; } \
             .a { position: absolute; left: -50px; top: 0; width: 10px; height: 500px; }";
        let l = layout_plain(html, css);
        let scrollers = crate::scroll_containers(&l);
        assert_eq!(scrollers.len(), 1, "one scroll container");
        let info = &scrollers[0];
        // max = overflow extent minus scrollport, floored at zero:
        // vertical overflow reaches 500 in a 100px scrollport.
        assert!(approx(info.max.1, 400.0), "max-y={:?}", info.max);
        assert!(approx(info.max.0, 0.0), "max-x={:?}", info.max);
    }

    #[test]
    fn relative_pct_top_computes_to_auto_against_indefinite_cb_height() {
        // Pinned conformance: percentage `top`/`bottom` on a relatively
        // positioned box compute to `auto` (no offset) when the
        // containing block's height is indefinite. The §10.6.4 errata add
        // no correction here — this is the intended spec outcome (cf.
        // WebKit bug 26396, fixed to compute to `auto`) — so the box
        // stays at its static position below the 100px spacer instead of
        // shifting by a percentage of the used height.
        let html = r#"<div id="cb"><div class="sp"></div><div id="r"></div></div>"#;
        let css = "body { margin: 0; } \
             .sp { height: 100px; } \
             #r { position: relative; top: 50%; bottom: 25%; height: 10px; }";
        let l = layout_plain(html, css);
        let r = &body(&l).children[0].children[1];
        assert!(approx(r.rect.y, 100.0), "y={}", r.rect.y);
    }

    #[test]
    fn relative_length_top_still_resolves_against_indefinite_cb_height() {
        // Control: lengths always resolve, even when the CB height is
        // indefinite.
        let html = r#"<div id="cb"><div class="sp"></div><div id="r"></div></div>"#;
        let css = "body { margin: 0; } \
             .sp { height: 100px; } \
             #r { position: relative; top: 30px; height: 10px; }";
        let l = layout_plain(html, css);
        let r = &body(&l).children[0].children[1];
        assert!(approx(r.rect.y, 130.0), "y={}", r.rect.y);
    }

    #[test]
    fn relative_pct_top_resolves_against_definite_cb_height() {
        // Control: with a definite CB height the percentage resolves
        // normally (50% of 400px).
        let html = r#"<div id="cb"><div class="sp"></div><div id="r"></div></div>"#;
        let css = "body { margin: 0; } #cb { height: 400px; } \
             .sp { height: 100px; } \
             #r { position: relative; top: 50%; height: 10px; }";
        let l = layout_plain(html, css);
        let r = &body(&l).children[0].children[1];
        assert!(approx(r.rect.y, 300.0), "y={}", r.rect.y);
    }

    // --- abspos percentage heights against an indefinite CB height
    //     (WPT css-position/position-relative-016) ---

    #[test]
    fn abspos_pct_height_definite_cb_control() {
        // t1 control: definite CB height — unchanged behavior. The
        // abspos `height: 100%` is 200px, so the relative child's
        // `top: 50%` shifts it by 100.
        let html = r#"<div class="cb"><div class="a"><div class="r"></div></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { position: relative; height: 200px; } \
             .a { position: absolute; height: 100%; } \
             .r { position: relative; top: 50%; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        let r = &a.children[0];
        assert!(approx(a.rect.h, 200.0), "a.h={}", a.rect.h);
        assert!(
            approx(r.rect.y - a.rect.y, 100.0),
            "dy={}",
            r.rect.y - a.rect.y
        );
    }

    #[test]
    fn abspos_pct_height_resolves_against_indefinite_cb_used_height() {
        // t8: an abspos `height: 100%` resolves against the containing
        // block's *used* height (20px) even though the CB height is
        // indefinite, making it definite — so the relative child's
        // `top: 50%` resolves to 10 instead of computing to `auto`.
        let html = r#"<div class="cb"><div class="h"></div><div class="a"><div class="r"></div></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { position: relative; } .h { height: 20px; } \
             .a { position: absolute; height: 100%; } \
             .r { position: relative; top: 50%; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[1];
        let r = &a.children[0];
        assert!(approx(a.rect.h, 20.0), "a.h={}", a.rect.h);
        assert!(
            approx(r.rect.y - a.rect.y, 10.0),
            "dy={}",
            r.rect.y - a.rect.y
        );
    }

    #[test]
    fn abspos_pct_height_inline_display_against_indefinite_cb() {
        // t9: `display: inline` on the abspos box still lays out as a
        // block-level out-of-flow box; `top: 100%` of the 20px used
        // height is 20.
        let html = r#"<div class="cb"><div class="h"></div><div class="a"><div class="r"></div></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { position: relative; } .h { height: 20px; } \
             .a { position: absolute; height: 100%; display: inline; } \
             .r { position: relative; top: 100%; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[1];
        let r = &a.children[0];
        assert!(approx(a.rect.h, 20.0), "a.h={}", a.rect.h);
        assert!(
            approx(r.rect.y - a.rect.y, 20.0),
            "dy={}",
            r.rect.y - a.rect.y
        );
    }

    #[test]
    fn abspos_calc_pct_height_resolves_against_indefinite_cb() {
        // t4 shape against an indefinite CB: a percentage-bearing
        // `calc()` height also resolves against the used height.
        let html = r#"<div class="cb"><div class="h"></div><div class="a"><div class="r"></div></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { position: relative; } .h { height: 20px; } \
             .a { position: absolute; height: calc(100% - 0px); } \
             .r { position: relative; top: 50%; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[1];
        let r = &a.children[0];
        assert!(approx(a.rect.h, 20.0), "a.h={}", a.rect.h);
        assert!(
            approx(r.rect.y - a.rect.y, 10.0),
            "dy={}",
            r.rect.y - a.rect.y
        );
    }

    #[test]
    fn abspos_pct_inset_still_auto_against_indefinite_cb_height() {
        // The used-height resolution is strictly for the *height*: a
        // percentage `top` on the abspos box itself still computes to
        // `auto` (CSS 2.1 §10.6.4) — `abspos_solved_sizes` and
        // `abs_position` must agree — so the box keeps its static
        // position below the 20px spacer instead of shifting by 10.
        let html = r#"<div class="cb"><div class="h"></div><div class="a"><div class="r"></div></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { position: relative; } .h { height: 20px; } \
             .a { position: absolute; height: 100%; top: 50%; } \
             .r { position: relative; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[1];
        assert!(approx(a.rect.h, 20.0), "a.h={}", a.rect.h);
        assert!(approx(a.rect.y, 20.0), "a.y={}", a.rect.y);
    }

    #[test]
    fn abspos_top_only_keeps_indefinite_height() {
        // t13 (negative): `top: 0` alone does not make the height
        // definite — the child's `top: 50%` still computes to `auto`.
        let html = r#"<div class="cb"><div class="a"><div class="r"></div></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { position: relative; height: 200px; } \
             .a { position: absolute; top: 0; } \
             .r { position: relative; top: 50%; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        let r = &a.children[0];
        assert!(
            approx(r.rect.y - a.rect.y, 0.0),
            "dy={}",
            r.rect.y - a.rect.y
        );
    }

    #[test]
    fn abspos_min_height_only_keeps_indefinite_height() {
        // t14 (negative): `min-height: 100%` must NOT confer
        // definiteness — the child's `top: 50%` still computes to `auto`.
        let html = r#"<div class="cb"><div class="a"><div class="r"></div></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { position: relative; height: 200px; } \
             .a { position: absolute; min-height: 100%; } \
             .r { position: relative; top: 50%; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        let r = &a.children[0];
        assert!(
            approx(r.rect.y - a.rect.y, 0.0),
            "dy={}",
            r.rect.y - a.rect.y
        );
    }

    #[test]
    fn abspos_all_auto_keeps_indefinite_height() {
        // t15 (negative): all-`auto` sizing stays indefinite.
        let html = r#"<div class="cb"><div class="a"><div class="r"></div></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { position: relative; height: 200px; } \
             .a { position: absolute; } \
             .r { position: relative; top: 50%; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        let r = &a.children[0];
        assert!(
            approx(r.rect.y - a.rect.y, 0.0),
            "dy={}",
            r.rect.y - a.rect.y
        );
    }

    #[test]
    fn abspos_pct_height_child_bottom_control() {
        // t16 control: `bottom: 50%` shifts the child up by half the
        // definite CB height.
        let html = r#"<div class="cb"><div class="a"><div class="r"></div></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { position: relative; height: 200px; } \
             .a { position: absolute; height: 100%; } \
             .r { position: relative; bottom: 50%; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        let r = &a.children[0];
        assert!(
            approx(r.rect.y - a.rect.y, -100.0),
            "dy={}",
            r.rect.y - a.rect.y
        );
    }

    #[test]
    fn relative_overconstrained_insets_use_cb_direction() {
        // CSS 2.1 §10.3.4: over-constrained `left`/`right` drops the end
        // side per the *containing block's* direction, not the element's
        // own. RTL CB with an LTR child: `right` wins, shifting left.
        let html = r#"<div class="cb"><div class="r"></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { direction: rtl; } \
             .r { position: relative; left: 30px; right: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let r = &body(&l).children[0].children[0];
        assert!(approx(r.rect.x, -10.0), "x={}", r.rect.x);
        // Mirror: LTR CB with an RTL child — `left` wins.
        let css2 = "body { margin: 0; } \
             .cb { direction: ltr; } \
             .r { position: relative; direction: rtl; left: 30px; right: 10px; height: 10px; }";
        let l2 = layout_plain(html, css2);
        let r2 = &body(&l2).children[0].children[0];
        assert!(approx(r2.rect.x, 30.0), "x={}", r2.rect.x);
    }

    // `height: stretch` support for abspos boxes is landing separately;
    // these WPT css-position/position-relative-016 shapes (t3/t18) are
    // ignored until the tree builds green with it. They pin the expected
    // values for that work.
    #[test]
    #[ignore]
    fn abspos_stretch_height_fills_definite_cb() {
        let html = r#"<div class="cb"><div class="a"><div class="r"></div></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { position: relative; height: 200px; } \
             .a { position: absolute; height: stretch; } \
             .r { position: relative; top: 50%; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        let r = &a.children[0];
        assert!(approx(a.rect.h, 200.0), "a.h={}", a.rect.h);
        assert!(
            approx(r.rect.y - a.rect.y, 100.0),
            "dy={}",
            r.rect.y - a.rect.y
        );
    }

    #[test]
    #[ignore]
    fn abspos_stretch_height_child_bottom() {
        let html = r#"<div class="cb"><div class="a"><div class="r"></div></div></div>"#;
        let css = "body { margin: 0; } \
             .cb { position: relative; height: 200px; } \
             .a { position: absolute; height: stretch; } \
             .r { position: relative; bottom: 50%; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        let r = &a.children[0];
        assert!(
            approx(r.rect.y - a.rect.y, -100.0),
            "dy={}",
            r.rect.y - a.rect.y
        );
    }

    // --- `position: fixed` static-position frame (fix 1) ---

    #[test]
    fn fixed_auto_offsets_follow_relative_shifted_parent() {
        // A fixed box with all-`auto` offsets takes its static position in
        // final document coordinates: the relative shift of its parent
        // moves the static position even though shift accumulation
        // restarts at the fixed box itself.
        let html = r#"<div id="p"><div id="s"></div><div id="f"></div></div>"#;
        let css = "body { margin: 0; } #p { position: relative; left: 100px; top: 50px; } #s { height: 30px; } #f { position: fixed; width: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let f = &body(&l).children[0].children[1];
        assert!(approx(f.rect.x, 100.0), "x={}", f.rect.x);
        assert!(approx(f.rect.y, 80.0), "y={}", f.rect.y);
    }

    #[test]
    fn fixed_auto_offsets_nested_in_abspos_parent() {
        // The fixed box sits inside an out-of-flow ancestor subtree laid
        // out at a provisional origin: the ancestor's placement shift
        // still applies to the recorded static position.
        let html = r#"<div id="o"><div id="p"><div id="f"></div></div></div>"#;
        let css = "body { margin: 0; } #o { position: relative; } #p { position: absolute; left: 100px; top: 50px; width: 200px; height: 200px; } #f { position: fixed; width: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let f = &body(&l).children[0].children[0].children[0];
        assert!(approx(f.rect.x, 100.0), "x={}", f.rect.x);
        assert!(approx(f.rect.y, 50.0), "y={}", f.rect.y);
    }

    #[test]
    fn fixed_auto_offsets_nested_in_fixed_parent() {
        // A fixed ancestor's own shift moves its subtree frame (so the
        // inner fixed box follows it), but in-flow shifts above the
        // ancestor never leak through.
        let html = r#"<div id="o"><div id="p"><div id="f"></div></div></div>"#;
        let css = "body { margin: 0; } #o { position: relative; left: 100px; top: 50px; } #p { position: fixed; width: 200px; height: 200px; } #f { position: fixed; width: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let p = &body(&l).children[0].children[0];
        let f = &p.children[0];
        assert!(approx(p.rect.x, 100.0), "px={}", p.rect.x);
        assert!(approx(p.rect.y, 50.0), "py={}", p.rect.y);
        assert!(approx(f.rect.x, 100.0), "fx={}", f.rect.x);
        assert!(approx(f.rect.y, 50.0), "fy={}", f.rect.y);
    }

    #[test]
    fn fixed_explicit_offsets_ignore_ancestor_shifts() {
        // Explicit viewport offsets never read the static position, so
        // ancestor shifts must not move the box.
        let html = r#"<div id="p"><div id="f"></div></div>"#;
        let css = "body { margin: 0; } #p { position: relative; left: 100px; top: 50px; } #f { position: fixed; left: 10px; top: 20px; width: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let f = &body(&l).children[0].children[0];
        assert!(approx(f.rect.x, 10.0), "x={}", f.rect.x);
        assert!(approx(f.rect.y, 20.0), "y={}", f.rect.y);
    }

    #[test]
    fn fixed_explicit_offsets_follow_shifted_transformed_cb() {
        // A fixed box with explicit insets is positioned against its
        // transformed containing block's *provisional* rect, so the
        // ancestor's deferred shift must ride along: the shift restart
        // at the fixed box restarts from the carried transformed-ancestor
        // shift, not from zero.
        let html = r#"<div id="o"><div id="t"><div id="f"></div></div></div>"#;
        let css = "body { margin: 0; } \
             #o { position: relative; top: 100px; } \
             #t { transform: translateX(0); } \
             #f { position: fixed; left: 5px; top: 7px; width: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let t = &body(&l).children[0].children[0];
        let f = &t.children[0];
        assert!(approx(t.rect.y, 100.0), "ty={}", t.rect.y);
        assert!(approx(f.rect.x, 5.0), "fx={}", f.rect.x);
        assert!(approx(f.rect.y, 107.0), "fy={}", f.rect.y);
    }

    #[test]
    fn fixed_auto_offsets_with_fixed_ancestor_before_transformed_cb() {
        // A fixed ancestor sits between the fixed descendant and its
        // transformed containing block: `resolve_containing_block` walks
        // past the fixed ancestor to the transformed one, and the
        // auto-offset fixup must do the same — adding the fixed
        // ancestor's pend while the walk supplies the transformed
        // ancestor's carried shift, each exactly once.
        let html = r#"<div id="t"><div id="fo"><div id="f"></div></div></div>"#;
        let css = "body { margin: 0; } \
             #t { transform: translate(0px, 0px); position: relative; top: 60px; } \
             #fo { position: fixed; left: 10px; top: 20px; width: 200px; height: 100px; } \
             #f { position: fixed; width: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let t = &body(&l).children[0];
        let fo = &t.children[0];
        let f = &fo.children[0];
        // The fixed ancestor rides its transformed CB's shift.
        assert!(approx(fo.rect.x, 10.0), "fox={}", fo.rect.x);
        assert!(approx(fo.rect.y, 80.0), "foy={}", fo.rect.y);
        // The auto-offset descendant lands on the ancestor's final
        // origin: ancestor pend via the fixup, transformed-CB shift via
        // the carried shift — no double count.
        assert!(approx(f.rect.x, 10.0), "fx={}", f.rect.x);
        assert!(approx(f.rect.y, 80.0), "fy={}", f.rect.y);
    }

    // --- `position: fixed` vs transformed ancestors (CSS Transforms 1 §3) ---

    #[test]
    fn fixed_insets_resolve_against_transformed_ancestor() {
        // A non-`none` transform makes the ancestor the containing block
        // for fixed descendants: insets resolve against its padding box,
        // not the viewport. (The transform itself is not rendered — only
        // containing-block resolution is implemented — so the ancestor
        // stays where flow put it.)
        let html = r#"<div id="t"><div id="f"></div></div>"#;
        let css = "body { margin: 0; } \
             #t { transform: translate(10px, 20px); margin: 30px; width: 200px; height: 100px; } \
             #f { position: fixed; left: 7px; top: 5px; width: 50px; height: 40px; }";
        let l = layout_plain(html, css);
        let t = &body(&l).children[0];
        let f = &t.children[0];
        assert!(approx(t.rect.x, 30.0), "tx={}", t.rect.x);
        assert!(approx(t.rect.y, 30.0), "ty={}", t.rect.y);
        assert!(approx(f.rect.x, 37.0), "fx={}", f.rect.x);
        assert!(approx(f.rect.y, 35.0), "fy={}", f.rect.y);
    }

    #[test]
    fn fixed_without_transformed_ancestor_uses_viewport() {
        // Control: with no transformed ancestor the fixed box still
        // resolves against the viewport, even nested inside a plain
        // in-flow parent.
        let html = r#"<div id="t"><div id="f"></div></div>"#;
        let css = "body { margin: 0; } \
             #t { margin: 30px; width: 200px; height: 100px; } \
             #f { position: fixed; left: 7px; top: 5px; width: 50px; height: 40px; }";
        let l = layout_plain(html, css);
        let f = &body(&l).children[0].children[0];
        assert!(approx(f.rect.x, 7.0), "x={}", f.rect.x);
        assert!(approx(f.rect.y, 5.0), "y={}", f.rect.y);
    }

    #[test]
    fn transform_none_does_not_create_containing_block() {
        // `transform: none` is not a containing block for fixed
        // descendants: the box stays viewport-anchored.
        let html = r#"<div id="t"><div id="f"></div></div>"#;
        let css = "body { margin: 0; } \
             #t { transform: none; margin: 30px; width: 200px; height: 100px; } \
             #f { position: fixed; left: 7px; top: 5px; width: 50px; height: 40px; }";
        let l = layout_plain(html, css);
        let f = &body(&l).children[0].children[0];
        assert!(approx(f.rect.x, 7.0), "x={}", f.rect.x);
        assert!(approx(f.rect.y, 5.0), "y={}", f.rect.y);
    }

    #[test]
    fn fixed_uses_nearest_transformed_ancestor() {
        // The walk stops at the *nearest* transformed ancestor: the
        // outer transformed (and positioned) grandparent is skipped.
        // (`padding` on the outer box blocks margin collapsing so the
        // two padding boxes sit at distinct origins.)
        let html = r#"<div id="o"><div id="i"><div id="f"></div></div></div>"#;
        let css = "body { margin: 0; } \
             #o { position: relative; transform: scale(2); margin: 10px; padding: 7px; width: 400px; height: 300px; } \
             #i { transform: rotate(45deg); margin: 20px; width: 200px; height: 100px; } \
             #f { position: fixed; left: 3px; top: 4px; width: 50px; height: 40px; }";
        let l = layout_plain(html, css);
        // Inner padding-box origin: (10 + 7 + 20, 10 + 7 + 20).
        let f = &body(&l).children[0].children[0].children[0];
        assert!(approx(f.rect.x, 40.0), "x={}", f.rect.x);
        assert!(approx(f.rect.y, 41.0), "y={}", f.rect.y);
    }

    #[test]
    fn fixed_prefers_transformed_ancestor_over_positioned() {
        // For `fixed`, a positioned but untransformed ancestor is *not*
        // the containing block: the transformed (static) ancestor wins
        // over the positioned grandparent.
        let html = r#"<div id="o"><div id="t"><div id="f"></div></div></div>"#;
        let css = "body { margin: 0; } \
             #o { position: relative; margin: 50px; padding: 5px; width: 400px; height: 300px; } \
             #t { transform: scale(2); margin: 10px; width: 200px; height: 100px; } \
             #f { position: fixed; left: 0; top: 0; width: 50px; height: 40px; }";
        let l = layout_plain(html, css);
        // Transformed ancestor's padding-box origin: (50 + 5 + 10, ...).
        // (Against the positioned grandparent it would be (55, 55).)
        let f = &body(&l).children[0].children[0].children[0];
        assert!(approx(f.rect.x, 65.0), "x={}", f.rect.x);
        assert!(approx(f.rect.y, 65.0), "y={}", f.rect.y);
    }

    #[test]
    fn fixed_insets_resolve_against_filtered_ancestor() {
        // A non-`none` `filter` also establishes a fixed containing block
        // (Filter Effects 1): insets resolve against the filtered
        // ancestor's padding box, not the viewport.
        let html = r#"<div id="t"><div id="f"></div></div>"#;
        let css = "body { margin: 0; } \
             #t { filter: blur(5px); margin: 30px; width: 200px; height: 100px; } \
             #f { position: fixed; left: 7px; top: 5px; width: 50px; height: 40px; }";
        let l = layout_plain(html, css);
        let t = &body(&l).children[0];
        let f = &t.children[0];
        assert!(approx(t.rect.x, 30.0), "tx={}", t.rect.x);
        assert!(approx(t.rect.y, 30.0), "ty={}", t.rect.y);
        assert!(approx(f.rect.x, 37.0), "fx={}", f.rect.x);
        assert!(approx(f.rect.y, 35.0), "fy={}", f.rect.y);
    }

    #[test]
    fn absolute_diverts_to_transformed_ancestor_over_positioned() {
        // For `absolute`, a non-positioned fixed-CB trigger between the
        // box and the nearest positioned ancestor wins the containing
        // block (CSS Transforms 1 §3): insets resolve against the
        // transformed ancestor's padding box, not the positioned
        // grandparent's.
        let html = r#"<div id="o"><div id="t"><div id="a"></div></div></div>"#;
        let css = "body { margin: 0; } \
             #o { position: relative; margin: 50px; padding: 5px; width: 400px; height: 300px; } \
             #t { transform: scale(2); margin: 10px; width: 200px; height: 100px; } \
             #a { position: absolute; left: 3px; top: 4px; width: 50px; height: 40px; }";
        let l = layout_plain(html, css);
        // Transformed ancestor's padding-box origin: (50 + 5 + 10, ...).
        // (Against the positioned grandparent it would be (58, 59).)
        let a = &body(&l).children[0].children[0].children[0];
        assert!(approx(a.rect.x, 68.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 69.0), "y={}", a.rect.y);
    }

    // --- sticky shift: fixed descendants of transformed ancestors ---

    #[test]
    fn sticky_shift_carries_fixed_under_transformed_ancestor() {
        // Repro: `position: sticky` outer > `transform` middle >
        // `position: fixed` inner. When the sticky constraint shifts the
        // subtree, the transformed middle moves; the fixed box's
        // containing block is that middle (CSS Transforms 1 §3), so it
        // must ride the middle's applied shift instead of detaching.
        let html = r#"<div id="sp"></div><div id="st"><div id="t"><div id="f"></div></div></div><div id="fill"></div>"#;
        let css = "body { margin: 0; } \
             #sp { height: 100px; } \
             #st { position: sticky; top: 10px; height: 50px; } \
             #t { transform: translateX(0px); height: 40px; } \
             #f { position: fixed; left: 7px; top: 5px; width: 50px; height: 40px; } \
             #fill { height: 2000px; }";
        let un = layout_scrolled(html, css, (0.0, 0.0));
        let scrolled = layout_scrolled(html, css, (0.0, 200.0));
        let st_u = &body(&un).children[1];
        let st_s = &body(&scrolled).children[1];
        let (t_u, f_u) = (&st_u.children[0], &st_u.children[0].children[0]);
        let (t_s, f_s) = (&st_s.children[0], &st_s.children[0].children[0]);
        // The sticky constraint fired: the subtree moved down.
        let ddy = st_s.rect.y - st_u.rect.y;
        assert!(ddy > 0.0, "sticky did not shift: ddy={ddy}");
        // The transformed middle moved with the subtree...
        assert!(
            approx(t_s.rect.y - t_u.rect.y, ddy),
            "middle did not ride the sticky shift: {}",
            t_s.rect.y - t_u.rect.y
        );
        // ...and the fixed box kept its offset against its containing
        // block, i.e. it rode the same shift.
        assert!(
            approx(f_s.rect.y - t_s.rect.y, f_u.rect.y - t_u.rect.y),
            "fixed detached from transformed CB: {} vs {}",
            f_s.rect.y - t_s.rect.y,
            f_u.rect.y - t_u.rect.y
        );
        assert!(
            approx(f_s.rect.x - t_s.rect.x, f_u.rect.x - t_u.rect.x),
            "fixed moved horizontally: {} vs {}",
            f_s.rect.x - t_s.rect.x,
            f_u.rect.x - t_u.rect.x
        );
    }

    #[test]
    fn sticky_shift_skips_fixed_without_transformed_ancestor() {
        // Control: with no transformed ancestor the fixed box stays
        // viewport-locked when the sticky constraint shifts the subtree
        // (the old skip behavior is unchanged).
        let html = r#"<div id="sp"></div><div id="st"><div id="m"><div id="f"></div></div></div><div id="fill"></div>"#;
        let css = "body { margin: 0; } \
             #sp { height: 100px; } \
             #st { position: sticky; top: 10px; height: 50px; } \
             #m { height: 40px; } \
             #f { position: fixed; left: 7px; top: 5px; width: 50px; height: 40px; } \
             #fill { height: 2000px; }";
        let un = layout_scrolled(html, css, (0.0, 0.0));
        let scrolled = layout_scrolled(html, css, (0.0, 200.0));
        let st_u = &body(&un).children[1];
        let st_s = &body(&scrolled).children[1];
        // The sticky constraint fired.
        assert!(st_s.rect.y - st_u.rect.y > 0.0, "sticky did not shift");
        // The fixed box did not move with it: viewport-locked.
        let f_u = &st_u.children[0].children[0];
        let f_s = &st_s.children[0].children[0];
        assert!(
            approx(f_s.rect.y, f_u.rect.y),
            "fixed moved vertically: {} vs {}",
            f_s.rect.y,
            f_u.rect.y
        );
        assert!(
            approx(f_s.rect.x, f_u.rect.x),
            "fixed moved horizontally: {} vs {}",
            f_s.rect.x,
            f_u.rect.x
        );
    }

    // --- containing-block padding box (fix 2) ---

    const PAD_CB_CSS: &str = "body { margin: 0; } #cb { position: relative; \
         border-left: 4px solid black; border-right: 12px solid black; \
         border-top: 8px solid black; border-bottom: 16px solid black; \
         padding: 20px; width: 200px; height: 100px; }";

    #[test]
    fn abspos_offsets_resolve_against_padding_box() {
        // Offsets resolve against the containing block's padding box
        // (border-box origin plus borders), not the content edge.
        let html = r#"<div id="cb"><div id="a"></div></div>"#;
        let css = format!(
            "{PAD_CB_CSS} #a {{ position: absolute; left: 0; top: 0; width: 10px; height: 10px; }}"
        );
        let l = layout_plain(html, &css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.x, 4.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 8.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_right_bottom_offsets_resolve_against_padding_box() {
        let html = r#"<div id="cb"><div id="a"></div></div>"#;
        let css = format!(
            "{PAD_CB_CSS} #a {{ position: absolute; right: 0; bottom: 0; width: 10px; height: 10px; }}"
        );
        let l = layout_plain(html, &css);
        let a = &body(&l).children[0].children[0];
        // Border box w = 200 + 4 + 12 + 40 = 256, padding-box w = 240:
        // right:0 => x = 4 + 240 - 10.
        assert!(approx(a.rect.x, 234.0), "x={}", a.rect.x);
        // Border box h = 100 + 8 + 16 + 40 = 164, padding-box h = 140:
        // bottom:0 => y = 8 + 140 - 10.
        assert!(approx(a.rect.y, 138.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_percentage_offsets_resolve_against_padding_box() {
        let html = r#"<div id="cb"><div id="a"></div></div>"#;
        let css = format!(
            "{PAD_CB_CSS} #a {{ position: absolute; left: 50%; top: 50%; width: 10px; height: 10px; }}"
        );
        let l = layout_plain(html, &css);
        let a = &body(&l).children[0].children[0];
        // x = 4 + 240 * 0.5, y = 8 + 140 * 0.5.
        assert!(approx(a.rect.x, 124.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 78.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_offsets_use_nearest_positioned_ancestor_padding_box() {
        // The containing block is the nearest positioned ancestor: its
        // padding box wins even when an outer positioned ancestor has a
        // different padding box.
        let html = r#"<div id="outer"><div id="inner"><div id="a"></div></div></div>"#;
        let css = "body { margin: 0; } #outer { position: relative; border: 10px solid black; padding: 20px; } \
             #inner { position: relative; border-left: 4px solid black; border-top: 8px solid black; padding: 5px; width: 100px; height: 50px; } \
             #a { position: absolute; left: 0; top: 0; width: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0].children[0];
        // #inner's border box starts at #outer's content-box origin
        // (10 + 20, 10 + 20); the padding-box origin adds #inner's own
        // borders (4, 8).
        assert!(approx(a.rect.x, 34.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 38.0), "y={}", a.rect.y);
    }

    // --- abspos stretch from opposing insets (CSS 2.1 §10.3.7/§10.6.4) ---

    #[test]
    fn abspos_opposing_insets_stretch_width() {
        // `left: 0; right: 0; width: auto` stretches to the containing
        // block instead of shrink-to-fit: 800px wide despite narrow text.
        let html = r#"<div id="a">short text</div>"#;
        let css = "body { margin: 0; } #a { position: absolute; left: 0; right: 0; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0];
        assert!(approx(a.rect.w, 800.0), "w={}", a.rect.w);
        assert!(approx(a.rect.x, 0.0), "x={}", a.rect.x);
    }

    #[test]
    fn abspos_opposing_insets_stretch_with_margins() {
        // Used width = CB − insets − margins; the box anchors at the
        // start inset plus its own margin.
        let html = r#"<div id="a">short text</div>"#;
        let css = "body { margin: 0; } #a { position: absolute; left: 10px; right: 10px; \
             margin-left: 20px; margin-right: 30px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0];
        assert!(approx(a.rect.w, 730.0), "w={}", a.rect.w);
        assert!(approx(a.rect.x, 30.0), "x={}", a.rect.x);
    }

    #[test]
    fn abspos_opposing_insets_stretch_with_border_padding() {
        // Padding/border subtract from the solved content width; the
        // border box spans inset to inset.
        let html = r#"<div id="a">short text</div>"#;
        let css = "body { margin: 0; } #a { position: absolute; left: 10px; right: 10px; \
             padding: 5px; border: 2px solid black; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0];
        assert!(approx(a.rect.w, 780.0), "w={}", a.rect.w);
        assert!(approx(a.rect.x, 10.0), "x={}", a.rect.x);
    }

    #[test]
    fn abspos_opposing_insets_min_width_clamps() {
        // min-width clamps the solved width (min wins); the over-constrained
        // end inset is dropped, so the box stays anchored at `left`.
        let html = r#"<div id="a">short text</div>"#;
        let css =
            "body { margin: 0; } #a { position: absolute; left: 0; right: 0; min-width: 900px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0];
        assert!(approx(a.rect.w, 900.0), "w={}", a.rect.w);
        assert!(approx(a.rect.x, 0.0), "x={}", a.rect.x);
    }

    #[test]
    fn abspos_opposing_insets_max_width_clamps() {
        let html = r#"<div id="a">short text</div>"#;
        let css =
            "body { margin: 0; } #a { position: absolute; left: 0; right: 0; max-width: 500px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0];
        assert!(approx(a.rect.w, 500.0), "w={}", a.rect.w);
        assert!(approx(a.rect.x, 0.0), "x={}", a.rect.x);
    }

    #[test]
    fn abspos_single_inset_still_shrink_to_fit() {
        // Only one inset specified: `width: auto` stays shrink-to-fit.
        let html = r#"<div id="a">short text</div>"#;
        let css = "body { margin: 0; } #a { position: absolute; left: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0];
        assert!(a.rect.w < 800.0, "w={}", a.rect.w);
        assert!(approx(a.rect.x, 10.0), "x={}", a.rect.x);
    }

    #[test]
    fn abspos_opposing_insets_stretch_height() {
        // `top: 10px; bottom: 20px; height: auto` stretches against a
        // definite containing-block height.
        let html = r#"<div id="cb"><div id="a">x</div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; height: 400px; } \
             #a { position: absolute; top: 10px; bottom: 20px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.h, 370.0), "h={}", a.rect.h);
        assert!(approx(a.rect.y, 10.0), "y={}", a.rect.y);
    }

    // --- abspos opposing LENGTH insets vs an indefinite CB height
    //     (CSS 2.1 §10.6.4 rule 5) ---

    // The content height of a one-line "x" with the test font metrics:
    // line height = 1.2 * default 16px font size.
    const ONE_LINE_H: f32 = 19.2;

    #[test]
    fn abspos_length_insets_indefinite_cb_height_keeps_content_height() {
        // `top: 10px; bottom: 10px; height: auto` against an indefinite
        // containing-block height: the §10.6.4 rule-5 stretch needs a
        // definite height, so browsers fall back to `height: auto` —
        // the box keeps its content height (it must NOT solve against
        // zero and collapse). Over-constrained, `top` wins.
        let html = r#"<div id="cb"><div id="a">x</div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; } \
             #a { position: absolute; top: 10px; bottom: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.h, ONE_LINE_H), "h={}", a.rect.h);
        assert!(approx(a.rect.y, 10.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_length_insets_indefinite_cb_height_auto_margins_zero() {
        // Same setup with `margin: auto 0`: with no meaningful
        // constraint-equation remainder, `auto` margins compute to zero
        // instead of solving against the containing block's raw used
        // height — the box stays anchored at `top` with content height.
        let html = r#"<div id="cb"><div id="a">x</div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; } \
             #a { position: absolute; top: 10px; bottom: 10px; margin: auto 0; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.h, ONE_LINE_H), "h={}", a.rect.h);
        assert!(approx(a.rect.y, 10.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_pct_insets_indefinite_cb_height_stays_auto_sizing() {
        // Percentage opposing insets compute to `auto` against an
        // indefinite CB height (CSS 2.1 §10.6.4): `height: auto` never
        // enters the stretch equation and the box keeps content height.
        // (Sizing side of the existing
        // `abspos_pct_opposing_insets_stay_auto_against_indefinite_cb_height`
        // positioning test, which this one also re-verifies.)
        let html = r#"<div id="cb"><div class="sp"></div><div id="a">x</div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; } \
             .sp { height: 100px; } \
             #a { position: absolute; top: 10%; bottom: 20%; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[1];
        assert!(approx(a.rect.h, ONE_LINE_H), "h={}", a.rect.h);
        assert!(approx(a.rect.y, 100.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_opposing_insets_max_height_clamps() {
        let html = r#"<div id="cb"><div id="a">x</div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; height: 400px; } \
             #a { position: absolute; top: 10px; bottom: 20px; max-height: 100px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.h, 100.0), "h={}", a.rect.h);
        assert!(approx(a.rect.y, 10.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_auto_height_single_inset_uses_content_height() {
        // Only one vertical inset: `height: auto` falls back to content
        // height.
        let html = r#"<div id="cb"><div id="a">x</div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; height: 400px; } \
             #a { position: absolute; top: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        assert!(a.rect.h < 390.0, "h={}", a.rect.h);
        assert!(approx(a.rect.y, 10.0), "y={}", a.rect.y);
    }

    // --- percentage vertical insets vs an indefinite CB height
    //     (CSS 2.1 §10.6.4) ---

    #[test]
    fn abspos_pct_top_computes_to_auto_against_indefinite_cb_height() {
        // The containing block has no specified height (indefinite), so
        // `top: 50%` computes to `auto` and the box falls back to its
        // static position (100, after the spacer) instead of resolving
        // 50% of the used height (110 * 0.5 = 55).
        let html = r#"<div id="cb"><div class="sp"></div><div id="a"></div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; } \
             .sp { height: 100px; } \
             #a { position: absolute; top: 50%; width: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[1];
        assert!(approx(a.rect.y, 100.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_zero_pct_calc_top_computes_to_auto_against_indefinite_cb_height() {
        // `top: calc(10px + 0%)` folds to `Math { pct: 0.0, px: 10.0 }`
        // but syntactically contains a percentage (`had_pct`), so per
        // CSS Values 4 §8.1.4 it is cyclic against the indefinite
        // containing-block height and computes to `auto` — the static
        // position (100) wins, not the folded 10px.
        let html = r#"<div id="cb"><div class="sp"></div><div id="a"></div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; } \
             .sp { height: 100px; } \
             #a { position: absolute; top: calc(10px + 0%); width: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[1];
        assert!(approx(a.rect.y, 100.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_pct_bottom_computes_to_auto_against_indefinite_cb_height() {
        // Same for `bottom`: indefinite CB height means `auto`, so the
        // static position wins.
        let html = r#"<div id="cb"><div class="sp"></div><div id="a"></div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; } \
             .sp { height: 100px; } \
             #a { position: absolute; bottom: 50%; width: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[1];
        assert!(approx(a.rect.y, 100.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_length_top_still_resolves_against_indefinite_cb_height() {
        // Lengths always resolve, even when the CB height is indefinite.
        let html = r#"<div id="cb"><div class="sp"></div><div id="a"></div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; } \
             .sp { height: 100px; } \
             #a { position: absolute; top: 30px; width: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[1];
        assert!(approx(a.rect.y, 30.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_pct_top_resolves_against_definite_cb_height() {
        // Control: with a definite CB height the percentage resolves
        // normally (50% of the 140px padding-box height).
        let html = r#"<div id="cb"><div id="a"></div></div>"#;
        let css = format!(
            "{PAD_CB_CSS} #a {{ position: absolute; top: 50%; width: 10px; height: 10px; }}"
        );
        let l = layout_plain(html, &css);
        let a = &body(&l).children[0].children[0];
        // Padding-box origin y = 8, height 140: y = 8 + 70.
        assert!(approx(a.rect.y, 78.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_pct_opposing_insets_stay_auto_against_indefinite_cb_height() {
        // `top: 10%; bottom: 20%; height: auto` against an indefinite CB
        // height: both insets compute to `auto` (CSS 2.1 §10.6.4), so the
        // height must NOT stretch against the used height — sizing and
        // positioning must agree. The box keeps its content height at
        // the static position (below the 100px spacer).
        let html = r#"<div id="cb"><div class="sp"></div><div id="a">x</div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; } \
             .sp { height: 100px; } \
             #a { position: absolute; top: 10%; bottom: 20%; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[1];
        assert!(approx(a.rect.y, 100.0), "y={}", a.rect.y);
        // Stretched against the 100px used height it would be 70px; the
        // content height is far smaller.
        assert!(a.rect.h < 69.0, "h={}", a.rect.h);
    }

    // --- abspos auto margins vs opposing insets (CSS 2.1 §10.3.7/§10.6.4) ---

    #[test]
    fn abspos_auto_margins_center_with_opposing_insets() {
        // `left`, `width` and `right` all specified with
        // `margin-left/right: auto`: the auto margins split the
        // remainder equally (centering) instead of computing to zero.
        let html = r#"<div id="cb"><div id="a"></div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; width: 400px; height: 200px; } \
             #a { position: absolute; left: 10px; right: 10px; width: 100px; height: 50px; margin: 0 auto; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        // Remainder 400 - 10 - 10 - 100 = 280, split equally: x = 10 + 140.
        assert!(approx(a.rect.x, 150.0), "x={}", a.rect.x);
    }

    #[test]
    fn abspos_single_auto_margin_takes_remainder() {
        // Only `margin-left` is `auto`: it takes the whole remainder.
        let html = r#"<div id="cb"><div id="a"></div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; width: 400px; height: 200px; } \
             #a { position: absolute; left: 10px; right: 10px; width: 100px; height: 50px; margin-left: auto; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        // x = 10 + 280.
        assert!(approx(a.rect.x, 290.0), "x={}", a.rect.x);
    }

    #[test]
    fn abspos_auto_margins_center_vertically() {
        // Same on the vertical axis (CSS 2.1 §10.6.4).
        let html = r#"<div id="cb"><div id="a"></div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; width: 400px; height: 400px; } \
             #a { position: absolute; top: 10px; bottom: 10px; width: 50px; height: 100px; margin: auto 0; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        // Remainder 400 - 10 - 10 - 100 = 280, split equally: y = 10 + 140.
        assert!(approx(a.rect.y, 150.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_width_auto_opposing_insets_keeps_zero_margins() {
        // `width: auto` with both insets specified: the width solves
        // from the insets and auto margins stay zero (untouched by the
        // §10.3.7 margin solving).
        let html = r#"<div id="cb"><div id="a"></div></div>"#;
        let css = "body { margin: 0; } #cb { position: relative; width: 400px; height: 200px; } \
             #a { position: absolute; left: 10px; right: 20px; height: 50px; margin: 0 auto; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.x, 10.0), "x={}", a.rect.x);
        assert!(approx(a.rect.w, 370.0), "w={}", a.rect.w);
    }

    // --- sticky inside a nested scroll container ---

    const NESTED_HTML: &str = r#"<div id="sc"><div class="spacer"></div><div class="sticky"></div><div class="filler"></div></div>"#;
    const NESTED_TOP_CSS: &str = "body { margin: 0; } #sc { overflow: auto; height: 200px; } \
         .spacer { height: 50px; } .sticky { position: sticky; top: 10px; height: 50px; } \
         .filler { height: 2000px; }";

    #[test]
    fn sticky_in_nested_scroller_ignores_viewport_scroll() {
        // The sticky box scrolls with the nested container, not the
        // viewport: a large viewport scroll offset must not move it. It
        // constrains against the container's scrollport (at offset zero,
        // the only nested offset layout sees), so it rests exactly where
        // it was laid out: 50.
        let l = layout_scrolled(NESTED_HTML, NESTED_TOP_CSS, (0.0, 500.0));
        let sticky = &body(&l).children[0].children[1];
        assert!(approx(sticky.rect.y, 50.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_in_nested_scroller_tracks_container_scrollport() {
        // The container sits 300px down the page; its scrollport — not
        // the viewport's — anchors the stick range. The box is laid out
        // at 300, but the container's inset top edge is at 300 + 10 =
        // 310, so it sticks there even with the viewport scrolled to
        // 400.
        let html = r#"<div class="pre"></div><div id="sc"><div class="sticky"></div><div class="filler"></div></div>"#;
        let css = "body { margin: 0; } .pre { height: 300px; } \
             #sc { overflow: auto; height: 200px; } \
             .sticky { position: sticky; top: 10px; height: 50px; } \
             .filler { height: 2000px; }";
        let l = layout_scrolled(html, css, (0.0, 400.0));
        let sticky = &body(&l).children[1].children[0];
        assert!(approx(sticky.rect.y, 310.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_in_nested_scroller_keeps_containing_block_bound() {
        // The containing-block hard bound still applies under the nested
        // scroller path. The sticky box's containing block is `.cb`, a
        // plain 200px block *inside* the scroller (not the scroller
        // itself, whose bound would use its scrollable overflow): the
        // box cannot be pushed past `.cb`'s end edge (200 - 50 = 150),
        // even though it is laid out at 160.
        let html = r#"<div id="sc"><div class="cb"><div class="spacer"></div><div class="sticky"></div></div><div class="filler"></div></div>"#;
        let css = "body { margin: 0; } #sc { overflow: auto; height: 200px; } \
             .cb { height: 200px; } \
             .spacer { height: 160px; } .sticky { position: sticky; top: 10px; height: 50px; } \
             .filler { height: 2000px; }";
        let l = layout_scrolled(html, css, (0.0, 500.0));
        let sticky = &body(&l).children[0].children[0].children[1];
        assert!(approx(sticky.rect.y, 150.0), "y={}", sticky.rect.y);
    }

    // --- sticky inside an abspos subtree (out-of-flow boundary) ---

    #[test]
    fn sticky_in_abspos_subtree_constrains_against_viewport() {
        // The abspos ancestor's containing block (the initial containing
        // block here: no positioned ancestor between it and the root)
        // sits outside the DOM-ancestor scroll container, so the sticky
        // box does not scroll with that container. The ancestor walk must
        // stop at the abspos boundary and fall back to the viewport:
        // scrolled to 500, the box sticks 10px below the viewport top
        // (510). Against the container's scrollport (offset zero) it
        // would wrongly rest at its laid-out position (100).
        let html = r#"<div id="sc"><div id="a"><div class="spacer"></div><div class="sticky"></div><div class="filler"></div></div></div>"#;
        let css = "body { margin: 0; } #sc { overflow: auto; height: 200px; } \
             #a { position: absolute; } \
             .spacer { height: 100px; } .sticky { position: sticky; top: 10px; height: 50px; } \
             .filler { height: 2000px; }";
        let l = layout_scrolled(html, css, (0.0, 500.0));
        let sticky = &body(&l).children[0].children[0].children[1];
        assert!(approx(sticky.rect.y, 510.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_in_abspos_subtree_rests_at_zero_scroll() {
        // Same DOM, no scroll: the box rests exactly where it was laid
        // out (100), constrained against the viewport either way.
        let html = r#"<div id="sc"><div id="a"><div class="spacer"></div><div class="sticky"></div><div class="filler"></div></div></div>"#;
        let css = "body { margin: 0; } #sc { overflow: auto; height: 200px; } \
             #a { position: absolute; } \
             .spacer { height: 100px; } .sticky { position: sticky; top: 10px; height: 50px; } \
             .filler { height: 2000px; }";
        let l = layout_scrolled(html, css, (0.0, 0.0));
        let sticky = &body(&l).children[0].children[0].children[1];
        assert!(approx(sticky.rect.y, 100.0), "y={}", sticky.rect.y);
    }

    // --- sticky inside an abspos subtree whose CB is inside the
    //     scroller (the walk re-anchors at the containing block) ---

    #[test]
    fn sticky_in_abspos_subtree_constrains_against_inner_scroller() {
        // The abspos ancestor's containing block is the scroller itself
        // (`position: relative`), so the sticky box scrolls with it:
        // the walk re-anchors at the containing block instead of
        // stopping at the abspos boundary. #sc is scrolled 200, so the
        // inset top edge of its scrollport is at 0 + 200 + 10 = 210 and
        // the box sticks there. Against the viewport at the passed-in
        // (zero) document scroll it would rest at its laid-out position
        // (100).
        let html = r#"<div id="sc"><div id="a"><div class="spacer"></div><div class="sticky"></div><div class="filler"></div></div></div>"#;
        let css = "body { margin: 0; } \
             #sc { position: relative; overflow: auto; height: 200px; } \
             #a { position: absolute; } \
             .spacer { height: 100px; } .sticky { position: sticky; top: 10px; height: 50px; } \
             .filler { height: 2000px; }";
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let mut offsets = HashMap::new();
        offsets.insert(doc.get_element_by_id("sc").expect("sc"), (0.0, 200.0));
        let l = layout_with_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            (0.0, 0.0),
            &offsets,
        );
        let sticky = &body(&l).children[0].children[0].children[1];
        assert!(approx(sticky.rect.y, 210.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_in_abspos_subtree_cb_above_scroller_uses_viewport() {
        // The abspos ancestor's containing block (#rel) sits *above*
        // the scroll container in DOM order: the re-anchored walk jumps
        // from #a to #rel and never meets #sc, so the box constrains
        // against the viewport. Scrolled to 500 it sticks at 510;
        // against #sc's scrollport (offset zero) it would wrongly rest
        // at its relative position.
        let html = r#"<div id="rel"><div id="sc"><div id="a"><div class="spacer"></div><div class="sticky"></div><div class="filler"></div></div></div></div>"#;
        let css = "body { margin: 0; } #rel { position: relative; } \
             #sc { overflow: auto; height: 200px; } \
             #a { position: absolute; } \
             .spacer { height: 100px; } .sticky { position: sticky; top: 10px; height: 50px; } \
             .filler { height: 2000px; }";
        let l = layout_scrolled(html, css, (0.0, 500.0));
        let sticky = &body(&l).children[0].children[0].children[0].children[1];
        assert!(approx(sticky.rect.y, 510.0), "y={}", sticky.rect.y);
    }

    // --- nested sticky (WPT css-position/sticky/position-sticky-nested-*.html) ---

    #[test]
    fn nested_sticky_programmatic_dom_matches_parse() {
        // Build the DOM the way script does (`createElement` +
        // `setAttribute` + `appendChild`) instead of parsing: the box
        // tree and sticky constraints must agree with the parsed
        // version.
        let mut doc = bowser_dom::Document::new();
        let html_el = doc.create_element("html");
        let body_el = doc.create_element("body");
        doc.append_child(doc.root(), html_el);
        doc.append_child(html_el, body_el);
        let sc = doc.create_element("div");
        doc.set_attribute(
            sc,
            "style",
            "position: relative; width: 100px; height: 200px; overflow: hidden;",
        );
        let contents = doc.create_element("div");
        doc.set_attribute(contents, "style", "height: 500px;");
        let pre = doc.create_element("div");
        doc.set_attribute(pre, "style", "height: 100px;");
        let container = doc.create_element("div");
        doc.set_attribute(container, "style", "height: 300px;");
        let filler = doc.create_element("div");
        doc.set_attribute(filler, "style", "height: 100px;");
        let outer = doc.create_element("div");
        doc.set_attribute(
            outer,
            "style",
            "position: sticky; top: 50px; height: 100px;",
        );
        let inner = doc.create_element("div");
        doc.set_attribute(inner, "style", "position: sticky; top: 60px; height: 50px;");
        doc.append_child(outer, inner);
        doc.append_child(container, filler);
        doc.append_child(container, outer);
        doc.append_child(contents, pre);
        doc.append_child(contents, container);
        doc.append_child(sc, contents);
        doc.append_child(body_el, sc);
        let ss = bowser_css::parse_stylesheet("body { margin: 0; }");
        let l = layout(&doc, &[&ss], VP, &FixedFontMetrics);
        let outer_b = &body(&l).children[0].children[0].children[1].children[1];
        let inner_b = &outer_b.children[0];
        assert!(approx(outer_b.rect.y, 200.0), "outer y={}", outer_b.rect.y);
        assert!(approx(inner_b.rect.y, 200.0), "inner y={}", inner_b.rect.y);
    }

    #[test]
    fn nested_sticky_inline_styles_match_stylesheet() {
        // Same tree but with inline `style` attributes (the dynamic
        // DOM path): must agree with the stylesheet version.
        let html = r#"<div id="sc" style="position: relative; width: 100px; height: 200px; overflow: hidden;"><div style="height: 500px;"><div style="height: 100px;"></div><div style="height: 300px;"><div style="height: 100px;"></div><div style="position: sticky; top: 50px; height: 100px;"><div style="position: sticky; top: 60px; height: 50px;"></div></div></div></div></div>"#;
        let css = "body { margin: 0; }";
        let l = layout_plain(html, css);
        let outer = &body(&l).children[0].children[0].children[1].children[1];
        let inner = &outer.children[0];
        assert!(approx(outer.rect.y, 200.0), "outer y={}", outer.rect.y);
        assert!(approx(inner.rect.y, 200.0), "inner y={}", inner.rect.y);
    }

    #[test]
    fn nested_sticky_zero_scroll_via_scroll_entry() {
        // Same as above but through `layout_with_scroll_and_offsets`
        // (the driver's entry): document scroll (0,0), no nested
        // offsets. Must agree with the plain entry.
        let html = r#"<div id="sc"><div id="contents"><div id="pre"></div><div id="container"><div id="filler"></div><div id="outer"><div id="inner"></div></div></div></div></div>"#;
        let css = "body { margin: 0; } \
             #sc { position: relative; width: 100px; height: 200px; overflow: hidden; } \
             #contents { height: 500px; } \
             #pre { height: 100px; } \
             #container { height: 300px; } \
             #filler { height: 100px; } \
             #outer { position: sticky; top: 50px; height: 100px; } \
             #inner { position: sticky; top: 60px; height: 50px; }";
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let l = layout_with_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            (0.0, 0.0),
            &HashMap::new(),
        );
        let outer = &body(&l).children[0].children[0].children[1].children[1];
        let inner = &outer.children[0];
        assert!(approx(outer.rect.y, 200.0), "outer y={}", outer.rect.y);
        assert!(approx(inner.rect.y, 200.0), "inner y={}", inner.rect.y);
    }

    #[test]
    fn nested_sticky_zero_scroll_inner_unshifted() {
        // No scroll: neither sticky box moves; the inner keeps its
        // in-flow position (offsetTop 0 relative to the outer).
        let html = r#"<div id="sc"><div id="contents"><div id="pre"></div><div id="container"><div id="filler"></div><div id="outer"><div id="inner"></div></div></div></div></div>"#;
        let css = "body { margin: 0; } \
             #sc { position: relative; width: 100px; height: 200px; overflow: hidden; } \
             #contents { height: 500px; } \
             #pre { height: 100px; } \
             #container { height: 300px; } \
             #filler { height: 100px; } \
             #outer { position: sticky; top: 50px; height: 100px; } \
             #inner { position: sticky; top: 60px; height: 50px; }";
        let l = layout_plain(html, css);
        let outer = &body(&l).children[0].children[0].children[1].children[1];
        let inner = &outer.children[0];
        assert!(approx(outer.rect.y, 200.0), "outer y={}", outer.rect.y);
        assert!(approx(inner.rect.y, 200.0), "inner y={}", inner.rect.y);
    }

    #[test]
    fn nested_sticky_inner_sticks_before_outer() {
        // WPT position-sticky-nested-top.html test 2: #sc scrolled 145.
        // The outer sticky (top:50) has not reached its sticking point
        // (its top sits at 200-145=55 >= 50), but the inner sticky
        // (top:60, inside the outer) sticks: 145+60=205.
        let html = r#"<div id="sc"><div id="contents"><div id="pre"></div><div id="container"><div id="filler"></div><div id="outer"><div id="inner"></div></div></div></div></div>"#;
        let css = "body { margin: 0; } \
             #sc { position: relative; width: 100px; height: 200px; overflow: hidden; } \
             #contents { height: 500px; } \
             #pre { height: 100px; } \
             #container { height: 300px; } \
             #filler { height: 100px; } \
             #outer { position: sticky; top: 50px; height: 100px; } \
             #inner { position: sticky; top: 60px; height: 50px; }";
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let mut offsets = HashMap::new();
        offsets.insert(doc.get_element_by_id("sc").expect("sc"), (0.0, 145.0));
        let l = layout_with_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            (0.0, 0.0),
            &offsets,
        );
        let outer = &body(&l).children[0].children[0].children[1].children[1];
        let inner = &outer.children[0];
        assert!(approx(outer.rect.y, 200.0), "outer y={}", outer.rect.y);
        assert!(approx(inner.rect.y, 205.0), "inner y={}", inner.rect.y);
    }

    #[test]
    fn nested_sticky_scroll_100_probe() {
        // WPT position-sticky-nested-top.html test 1: #sc scrolled 100.
        // Neither sticky box sticks: outer stays at 200, inner at 200
        // (offsetTop 0 relative to the outer).
        let html = r#"<div id="sc"><div id="contents"><div id="pre"></div><div id="container"><div id="filler"></div><div id="outer"><div id="inner"></div></div></div></div></div>"#;
        let css = "body { margin: 0; } \
             #sc { position: relative; width: 100px; height: 200px; overflow: hidden; } \
             #contents { height: 500px; } \
             #pre { height: 100px; } \
             #container { height: 300px; } \
             #filler { height: 100px; } \
             #outer { position: sticky; top: 50px; height: 100px; } \
             #inner { position: sticky; top: 60px; height: 50px; }";
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let mut offsets = HashMap::new();
        offsets.insert(doc.get_element_by_id("sc").expect("sc"), (0.0, 100.0));
        let l = layout_with_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            (0.0, 0.0),
            &offsets,
        );
        let outer = &body(&l).children[0].children[0].children[1].children[1];
        let inner = &outer.children[0];
        assert!(approx(outer.rect.y, 200.0), "outer y={}", outer.rect.y);
        assert!(approx(inner.rect.y, 200.0), "inner y={}", inner.rect.y);
    }

    #[test]
    fn nested_sticky_scroll_200_300_probe() {
        // WPT position-sticky-nested-top.html tests 3-4 at the layout
        // level (offsets delivered directly): scroll 200 -> both stuck
        // (outer at 250, inner at 260, i.e. offsetTop 10 relative to the
        // outer); scroll 300 -> outer pinned by its CB (300), inner
        // pinned by the outer (350, offsetTop 50).
        let html = r#"<div id="sc"><div id="contents"><div id="pre"></div><div id="container"><div id="filler"></div><div id="outer"><div id="inner"></div></div></div></div></div>"#;
        let css = "body { margin: 0; } \
             #sc { position: relative; width: 100px; height: 200px; overflow: hidden; } \
             #contents { height: 500px; } \
             #pre { height: 100px; } \
             #container { height: 300px; } \
             #filler { height: 100px; } \
             #outer { position: sticky; top: 50px; height: 100px; } \
             #inner { position: sticky; top: 60px; height: 50px; }";
        for (scroll, outer_y, inner_y) in [(200.0, 250.0, 260.0), (300.0, 300.0, 350.0)] {
            let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
            let ss = bowser_css::parse_stylesheet(css);
            let mut offsets = HashMap::new();
            offsets.insert(doc.get_element_by_id("sc").expect("sc"), (0.0, scroll));
            let l = layout_with_scroll_and_offsets(
                &doc,
                &[&ss],
                VP,
                &FixedFontMetrics,
                (0.0, 0.0),
                &offsets,
            );
            let outer = &body(&l).children[0].children[0].children[1].children[1];
            let inner = &outer.children[0];
            assert!(
                approx(outer.rect.y, outer_y),
                "scroll {scroll}: outer y={}",
                outer.rect.y
            );
            assert!(
                approx(inner.rect.y, inner_y),
                "scroll {scroll}: inner y={}",
                inner.rect.y
            );
        }
    }

    // --- negative-IMCB unsafe-center alignment (layout half of case 4) ---

    #[test]
    fn abspos_unsafe_center_zero_imcb() {
        // WPT css-position/position-absolute-with-negative-sized-imcb.html
        // case 4 (`place-self: unsafe center`): the cascade drops the
        // `<overflow-position>` prefix (CSS-side gap), but once it is
        // honored the layout half centers the margin box on the
        // zero-sized IMCB instead of start-aligning it. Direct unit test
        // of `align_abspos_axis`: LTR inset 15 in a 20px container puts
        // the zero IMCB at 15, so a 10px box centers at 10; safe (the
        // default the cascade currently yields) start-aligns at 15.
        let centered =
            super::align_abspos_axis(15.0, 0.0, 0.0, 10.0, 0.0, super::AbsSelfAlign::Center, true);
        assert!(approx(centered, 10.0), "unsafe centered={}", centered);
        let safe = super::align_abspos_axis(
            15.0,
            0.0,
            0.0,
            10.0,
            0.0,
            super::AbsSelfAlign::Center,
            false,
        );
        assert!(approx(safe, 15.0), "safe start-aligned={}", safe);
        // RTL inset 15: the weaker (start) inset adjusts, the zero IMCB
        // sits at 5, unsafe center puts the box at 0.
        let rtl =
            super::align_abspos_axis(5.0, 0.0, 0.0, 10.0, 0.0, super::AbsSelfAlign::Center, true);
        assert!(approx(rtl, 0.0), "rtl unsafe centered={}", rtl);
    }

    // --- dynamic containing-block re-resolution ---

    #[test]
    fn abspos_dynamic_cb_reresolves_across_layouts() {
        // WPT css-position/position-absolute-dynamic-containing-block.html
        // (absolute half): moving `position: relative` from #outer to
        // #intermediate re-resolves the abspos CB on the next layout;
        // `resolve_containing_block` must not cache the old answer.
        let html =
            r#"<div id="outer"><div><div id="inter"><div id="target">x</div></div></div></div>"#;
        let base = "body { margin: 0; } div { padding: 5px; } \
             #outer { width: 400px; height: 300px; } \
             #inter { width: 300px; height: 200px; } \
             #target { position: absolute; top: 0; left: 0; bottom: 0; right: 0; }";
        let l = layout_plain(html, &format!("{base} #outer {{ position: relative; }}"));
        let t = &body(&l).children[0].children[0].children[0].children[0];
        assert!(approx(t.rect.h, 310.0), "outer CB h={}", t.rect.h);
        let l = layout_plain(html, &format!("{base} #inter {{ position: relative; }}"));
        let t = &body(&l).children[0].children[0].children[0].children[0];
        assert!(approx(t.rect.h, 210.0), "intermediate CB h={}", t.rect.h);
    }

    // --- fixed containing block via will-change (static probe) ---

    #[test]
    fn fixed_cb_will_change_transform_static() {
        // WPT css-position/position-absolute-dynamic-containing-block.html
        // (fixed half): a `will-change: transform` ancestor establishes
        // the containing block for a fixed descendant; with all insets 0
        // the box stretches to the ancestor's padding box.
        let html = r#"<div id="outer"><div><div id="target">TTT</div></div></div>"#;
        let css = "body { margin: 0; } \
             div { padding: 5px; } \
             #outer { width: 400px; height: 300px; will-change: transform; } \
             #target.fixed { position: fixed; top: 0; left: 0; bottom: 0; right: 0; }";
        // Without .fixed: in-flow.
        let l = layout_plain(html, css);
        let t = &body(&l).children[0].children[0].children[0];
        eprintln!("in-flow h={}", t.rect.h);
        // With .fixed: stretched to #outer's padding box (400+10 x 300+10).
        let css2 = "body { margin: 0; } \
             div { padding: 5px; } \
             #outer { width: 400px; height: 300px; will-change: transform; } \
             #target { position: fixed; top: 0; left: 0; bottom: 0; right: 0; }";
        let l = layout_plain(html, css2);
        let t = &body(&l).children[0].children[0].children[0];
        assert!(approx(t.rect.w, 410.0), "w={}", t.rect.w);
        assert!(approx(t.rect.h, 310.0), "h={}", t.rect.h);
    }

    // --- sticky inside a `fixed` subtree ---

    #[test]
    fn sticky_in_fixed_subtree_constrains_against_inner_scroller() {
        // The fixed panel is viewport-locked, but #sc inside it scrolls
        // its own content: the sticky box constrains against #sc's
        // scrollport at its live offset (150), so the inset top edge
        // puts it at 0 + 150 + 10 = 160. The containing block is #sc
        // itself — a scroll container — so the hard bound uses its
        // scrollable overflow (2150px), not its 200px content box, and
        // does not clamp the box. The document scroll (400) must not
        // leak in: the walk is bounded by the fixed ancestor.
        let html = r#"<div id="f"><div id="sc"><div class="spacer"></div><div class="sticky"></div><div class="filler"></div></div></div>"#;
        let css = "body { margin: 0; } \
             #f { position: fixed; top: 0; left: 0; width: 300px; height: 300px; } \
             #sc { overflow: auto; height: 200px; } \
             .spacer { height: 100px; } .sticky { position: sticky; top: 10px; height: 50px; } \
             .filler { height: 2000px; }";
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let mut offsets = HashMap::new();
        offsets.insert(doc.get_element_by_id("sc").expect("sc"), (0.0, 150.0));
        let l = layout_with_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            (0.0, 400.0),
            &offsets,
        );
        let sticky = &body(&l).children[0].children[0].children[1];
        assert!(approx(sticky.rect.y, 160.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_in_fixed_subtree_resolves_statically() {
        // No scroll container inside the fixed subtree: the box
        // resolves statically against its containing context — it stays
        // where it was laid out (50 + 100 = 150) and the document
        // scroll (500) cannot move it (the viewport frame constrains at
        // a zero offset). Constraining against the live document scroll
        // would have clamped it to the fixed panel's end edge (400).
        let html = r#"<div id="f"><div class="spacer"></div><div class="sticky"></div></div>"#;
        let css = "body { margin: 0; } \
             #f { position: fixed; top: 50px; left: 0; width: 300px; height: 400px; } \
             .spacer { height: 100px; } .sticky { position: sticky; top: 10px; height: 50px; }";
        let l = layout_scrolled(html, css, (0.0, 500.0));
        let sticky = &body(&l).children[0].children[1];
        assert!(approx(sticky.rect.y, 150.0), "y={}", sticky.rect.y);
        // And at zero document scroll it rests at the same position:
        // static either way.
        let l = layout_scrolled(html, css, (0.0, 0.0));
        let sticky = &body(&l).children[0].children[1];
        assert!(approx(sticky.rect.y, 150.0), "y={}", sticky.rect.y);
    }

    // --- sticky margin box and containing block (fix 2) ---

    #[test]
    fn sticky_constrains_border_box_against_scrollport() {
        // CSS Position 3 §3.4: the scrollport (sticky view rectangle)
        // constraint applies to the border box, not the margin box. With
        // `margin: 20px 0 30px` and `top: 10px`, the box sticks with its
        // *border* edge 10px below the viewport top (200 + 10 = 210) —
        // the margin is irrelevant whilst stuck (WPT
        // css/css-position/position-sticky-margins.html, "Whilst stuck").
        // The §3.4 "position box" refinement still governs the
        // containing-block bound (see
        // `sticky_margin_box_never_leaves_containing_block`).
        let html = TOP_HTML;
        let css = "body { margin: 0; } .spacer { height: 100px; } \
             .sticky { position: sticky; top: 10px; height: 50px; margin: 20px 0 30px; } \
             .filler { height: 2000px; }";
        let l = layout_scrolled(html, css, (0.0, 200.0));
        let sticky = &body(&l).children[1];
        assert!(approx(sticky.rect.y, 210.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_margin_box_never_leaves_containing_block() {
        // The containing-block hard bound applies to the margin box too:
        // with `margin-bottom: 30px` the box sticks to the parent's end
        // edge with its margin edge against it (border box at 200 − 80 =
        // 120), not at the old border-box bound of 150.
        let html =
            r#"<div class="wrap"><div class="sticky"></div></div><div class="filler"></div>"#;
        let css = "body { margin: 0; } .wrap { height: 200px; } \
             .sticky { position: sticky; top: 10px; height: 50px; margin-bottom: 30px; } \
             .filler { height: 2000px; }";
        let l = layout_scrolled(html, css, (0.0, 500.0));
        let sticky = &body(&l).children[0].children[0];
        assert!(approx(sticky.rect.y, 120.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_uses_nearest_block_container_as_containing_block() {
        // The containing block is the nearest block-container ancestor
        // (CSS 2.1 §10.1 with the 2012-08-01 erratum), not the parent's
        // content box: the sticky inline's parent is an inline box, so the
        // 100px `#wrap` bounds it. Scrolled far, it sticks to `#wrap`'s
        // end edge (100 − 19.2 = 80.8) and scrolls away with it, instead
        // of floating at the scrollport inset (510) past its block.
        let html = r#"<div id="wrap"><span id="outer"><span id="st">x</span></span></div><div class="filler"></div>"#;
        let css = "body { margin: 0; } #wrap { height: 100px; } \
             #st { position: sticky; top: 10px; } \
             .filler { height: 2000px; }";
        let l = layout_scrolled(html, css, (0.0, 500.0));
        let st = &body(&l).children[0].children[0].children[0];
        assert!(approx(st.rect.y, 80.8), "y={}", st.rect.y);
    }

    // --- sticky inside a nested scroller with live scroll offsets ---

    const LIVE_OFFSET_CSS: &str = "body { margin: 0; } \
         #sc { overflow-y: scroll; height: 120px; } \
         .spacer { height: 100px; } \
         .sticky { position: sticky; top: 10px; height: 50px; } \
         .filler { height: 2000px; }";
    const LIVE_OFFSET_HTML: &str = r#"<div id="sc"><div class="inner"><div class="spacer"></div><div class="sticky"></div><div class="filler"></div></div></div>"#;

    /// Layout `NESTED_HTML` for a viewport at rest, with the nested
    /// scroller's offset map built from `nested` when `present`, and
    /// return the sticky box's final y. Box path: body → #sc → .inner →
    /// .sticky (children[1]).
    fn nested_sticky_y(nested: (f32, f32), present: bool) -> f32 {
        let (doc, _) = bowser_dom::Document::parse_html(LIVE_OFFSET_HTML.as_bytes());
        let ss = bowser_css::parse_stylesheet(LIVE_OFFSET_CSS);
        let mut offsets = HashMap::new();
        if present {
            let sc = doc.get_element_by_id("sc").expect("sc");
            offsets.insert(sc, nested);
        }
        let l = layout_with_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            (0.0, 0.0),
            &offsets,
        );
        body(&l).children[0].children[0].children[1].rect.y
    }

    #[test]
    fn sticky_in_nested_scroller_constrains_at_live_offset() {
        // The scroller is scrolled 200px: the inset top edge of its
        // 120px scrollport is at 0 + 200 + 10 = 210, so the box (laid out
        // at 100) sticks there. The containing block (.inner, 2150px
        // tall) does not clamp it.
        let y = nested_sticky_y((0.0, 200.0), true);
        assert!(approx(y, 210.0), "y={y}");
    }

    #[test]
    fn sticky_in_nested_scroller_constrains_at_zero_without_entry() {
        // No map entry: the scroller constrains at offset zero — exact,
        // not an approximation (see `nested_scroll_offset`): the caller
        // guarantees an entry for every scroller with a nonzero offset,
        // so the inset range [10, ∞) leaves the box resting exactly
        // where it was laid out (100) instead of sticking.
        let y = nested_sticky_y((0.0, 0.0), false);
        assert!(approx(y, 100.0), "y={y}");
    }

    #[test]
    fn sticky_in_nested_scroller_zero_entry_matches_absent() {
        // An explicit zero entry is identical to no entry: offset zero.
        let y = nested_sticky_y((0.0, 0.0), true);
        assert!(approx(y, 100.0), "y={y}");
    }

    #[test]
    fn sticky_nested_scroller_offsets_apply_per_scroller() {
        // Two nested scrollers; only #a has a live-offset entry. The
        // entry-less scroller (#b) constrains at offset zero — the
        // fallback is per scroller, so a missing entry never disturbs a
        // present one (and vice versa). This pins the `nested_offsets`
        // contract: every scroller with a nonzero offset has an entry.
        let html = r#"<div id="a"><div class="sp"></div><div class="s"></div><div class="fill"></div></div><div id="b"><div class="sp"></div><div class="s"></div><div class="fill"></div></div>"#;
        let css = "body { margin: 0; } \
             #a { overflow: auto; height: 400px; } \
             #b { overflow: auto; height: 400px; } \
             .sp { height: 100px; } \
             .s { position: sticky; top: 10px; height: 50px; } \
             .fill { height: 2000px; }";
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let mut offsets = HashMap::new();
        offsets.insert(doc.get_element_by_id("a").expect("a"), (0.0, 150.0));
        let l = layout_with_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            (0.0, 0.0),
            &offsets,
        );
        let a_sticky = &body(&l).children[0].children[1];
        let b_sticky = &body(&l).children[1].children[1];
        // #a is scrolled 150px: the inset top edge of its scrollport is at
        // 0 + 150 + 10 = 160, above the box's laid-out position (100),
        // so it sticks at 160.
        assert!(approx(a_sticky.rect.y, 160.0), "a y={}", a_sticky.rect.y);
        // #b has no entry, so it constrains at offset zero: the inset top
        // edge of its scrollport (at y=400) is at 400 + 10 = 410, below
        // the box's laid-out position (400 + 100 = 500), so it rests at
        // 500.
        assert!(approx(b_sticky.rect.y, 500.0), "b y={}", b_sticky.rect.y);
    }

    // --- WPT position-sticky overconstrained replications ---
    //
    // css/css-position/sticky/position-sticky-{top-and-bottom,
    // left-and-right,horizontal-rtl}-overconstrained.html pin down the
    // over-constrained rule: when the sticky view rectangle is smaller
    // than the box, the *start* edge wins — the top edge always, the
    // left edge in LTR and the right edge in RTL (the containing
    // block's writing mode). The WPT drives these with real nested
    // scrolling (`scroller.scrollTop = …`); here the scroller's offset
    // is passed as a live nested offset. `offsetTop`/`offsetLeft` in
    // the WPT are relative to the positioned scroller/container, which
    // sit at the document origin here (`body { margin: 0 }`), so they
    // compare directly against `rect`.

    /// Layout the top/bottom overconstrained WPT with the scroller at a
    /// live vertical offset; return the sticky box's final y.
    fn overconstrained_top_bottom_y(scroll: f32) -> f32 {
        let html = r#"<div id="sc"><div class="padding"></div><div class="sticky"></div><div class="padding"></div></div>"#;
        let css = "body { margin: 0; } \
             #sc { height: 100px; overflow: auto; position: relative; } \
             .padding { height: 200px; } \
             .sticky { position: sticky; background: green; top: 20px; bottom: 0; height: 200px; }";
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let mut offsets = HashMap::new();
        offsets.insert(doc.get_element_by_id("sc").expect("sc"), (0.0, scroll));
        let l = layout_with_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            (0.0, 0.0),
            &offsets,
        );
        body(&l).children[0].children[1].rect.y
    }

    #[test]
    fn sticky_cb_scrollable_size_ignores_fixed_descendant() {
        // A `position: fixed` descendant is viewport-anchored: per CSS
        // Overflow 3 it contributes to no ancestor's scrollable
        // overflow. #fx at top:5000px must not inflate the scroller's
        // scrollable size, or the sticky containing-block bound would
        // never engage and the box would ride past its CB (y=500) instead
        // of stopping at its content end (600 − 200 = 400).
        let html = r#"<div id="sc"><div class="sticky"></div><div class="filler"></div><div id="fx"></div></div>"#;
        let css = "body { margin: 0; } \
             #sc { height: 100px; overflow: auto; position: relative; } \
             .sticky { position: sticky; top: 0; height: 200px; } \
             .filler { height: 400px; } \
             #fx { position: fixed; top: 5000px; }";
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let mut offsets = HashMap::new();
        offsets.insert(doc.get_element_by_id("sc").expect("sc"), (0.0, 500.0));
        let l = layout_with_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            (0.0, 0.0),
            &offsets,
        );
        let sticky = &body(&l).children[0].children[0];
        assert!(approx(sticky.rect.y, 400.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_overconstrained_top_wins_at_scroll_start() {
        // Scrollport [0,100] inset by top:20/bottom:0 against the 200px
        // box: the range [20,-100] is over-constrained, so the start
        // (top) edge wins and the box sits at 20.
        assert!(approx(overconstrained_top_bottom_y(0.0), 20.0));
    }

    #[test]
    fn sticky_overconstrained_top_wins_while_scrolling() {
        // scroll 160: range [180,60] → 180; scroll 220: [240,120] → 240.
        assert!(approx(overconstrained_top_bottom_y(160.0), 180.0));
        assert!(approx(overconstrained_top_bottom_y(220.0), 240.0));
    }

    #[test]
    fn sticky_overconstrained_uses_scrollable_cb_bound() {
        // scroll 500: the scrollport range [520,400] resolves to 520,
        // but the containing block (#sc) is itself a scroll container,
        // so the hard bound uses its scrollable overflow (600px), not
        // its 100px content box: the box stops at 600 − 200 = 400.
        assert!(approx(overconstrained_top_bottom_y(500.0), 400.0));
    }

    /// Layout the left/right overconstrained WPT with the scroller at a
    /// live horizontal offset; return the sticky box's final x.
    fn overconstrained_left_right_x(scroll: f32) -> f32 {
        let html = r#"<div id="sc"><div class="padding"></div><div class="sticky"></div><div class="padding"></div></div>"#;
        let css = "body { margin: 0; } \
             #sc { width: 100px; height: 100px; overflow: auto; position: relative; display: flex; } \
             .padding { width: 200px; height: 100px; flex: none; } \
             .sticky { position: sticky; background: green; left: 20px; right: 0; width: 200px; height: 100px; flex: none; }";
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let mut offsets = HashMap::new();
        offsets.insert(doc.get_element_by_id("sc").expect("sc"), (scroll, 0.0));
        let l = layout_with_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            (0.0, 0.0),
            &offsets,
        );
        body(&l).children[0].children[1].rect.x
    }

    #[test]
    fn sticky_overconstrained_left_wins_ltr() {
        // LTR: the over-constrained range resolves to the left (start)
        // edge — 20, 180, 240 — and at scroll 500 the scrollable-overflow
        // containing-block bound stops the box at 600 − 200 = 400.
        assert!(approx(overconstrained_left_right_x(0.0), 20.0));
        assert!(approx(overconstrained_left_right_x(160.0), 180.0));
        assert!(approx(overconstrained_left_right_x(220.0), 240.0));
        assert!(approx(overconstrained_left_right_x(500.0), 400.0));
    }

    /// Layout the RTL overconstrained WPT with the scroller at a live
    /// horizontal offset; return the sticky box's final x. The sticky
    /// box itself is `direction: ltr`, but its containing block
    /// (`.container`) is `direction: rtl`: the over-constrained range
    /// resolves to the *right* edge (start in the CB's writing mode).
    /// `.container` left-aligns (its own CB, `#sc`, is LTR), so `rect.x`
    /// compares directly against the WPT's `offsetLeft`.
    fn overconstrained_rtl_x(scroll: f32) -> f32 {
        let html = r#"<div id="sc"><div class="container"><div class="padding"></div><div class="sticky"></div><div class="padding"></div></div></div>"#;
        let css = "body { margin: 0; } \
             #sc { width: 100px; overflow: auto; position: relative; } \
             .container { display: flex; position: relative; width: 600px; direction: rtl; } \
             .padding { width: 200px; height: 100px; flex: none; } \
             .sticky { position: sticky; background: green; left: 0; right: 0; width: 200px; height: 100px; flex: none; direction: ltr; }";
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let mut offsets = HashMap::new();
        offsets.insert(doc.get_element_by_id("sc").expect("sc"), (scroll, 0.0));
        let l = layout_with_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            (0.0, 0.0),
            &offsets,
        );
        body(&l).children[0].children[0].children[1].rect.x
    }

    #[test]
    fn sticky_overconstrained_right_wins_in_rtl_cb() {
        // Over-constrained [scroll, scroll−100] resolves to the right
        // (start) edge: scroll 0 → the scrollport clamp gives −100, but
        // the containing-block bound pulls it back to 0; then 80, 120,
        // and 400 at scroll 500 (bound: 600 − 200).
        assert!(approx(overconstrained_rtl_x(0.0), 0.0));
        assert!(approx(overconstrained_rtl_x(180.0), 80.0));
        assert!(approx(overconstrained_rtl_x(220.0), 120.0));
        assert!(approx(overconstrained_rtl_x(500.0), 400.0));
    }

    #[test]
    fn sticky_uses_per_axis_scroll_container() {
        // CSS Position 3 §3.4: each axis constrains against the nearest
        // scroll container *with a matching scrollable axis*. `#sc`
        // scrolls horizontally only (`overflow-x: scroll`,
        // `overflow-y: clip` — `clip` is not a scroll container per CSS
        // Overflow 3 §2): the x axis constrains against `#sc`'s
        // scrollport at its live offset, the y axis against the
        // viewport's.
        let html = r#"<div id="sc"><div class="spacer"></div><div class="sticky"></div><div class="filler"></div></div>"#;
        let css = "body { margin: 0; } \
             #sc { width: 200px; height: 2000px; overflow-x: scroll; overflow-y: clip; } \
             .spacer { width: 500px; height: 100px; } \
             .sticky { position: sticky; left: 10px; top: 10px; width: 50px; height: 50px; } \
             .filler { width: 2000px; height: 2000px; }";
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let mut offsets = HashMap::new();
        offsets.insert(doc.get_element_by_id("sc").expect("sc"), (150.0, 0.0));
        let l = layout_with_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            (0.0, 300.0),
            &offsets,
        );
        let sticky = &body(&l).children[0].children[1];
        // x: `#sc` scrolled 150 → the inset left edge is at
        // 0 + 150 + 10 = 160: sticks.
        assert!(approx(sticky.rect.x, 160.0), "x={}", sticky.rect.x);
        // y: `#sc` is not a y scroll container, so the box constrains
        // against the viewport: scrolled 300 → the inset top edge is at
        // 0 + 300 + 10 = 310: sticks.
        assert!(approx(sticky.rect.y, 310.0), "y={}", sticky.rect.y);
    }

    #[test]
    fn sticky_with_auto_insets_stays_put() {
        // CSS Position 3 §3.4: "If both inset properties in a given axis
        // are auto, no offsets are added in that axis." A sticky box
        // with all insets auto is unconstrained in both axes: it stays
        // exactly where it was laid out, even when scrolled.
        let html =
            r#"<div class="spacer"></div><div class="sticky"></div><div class="filler"></div>"#;
        let css = ".spacer { height: 100px; } \
             .sticky { position: sticky; height: 50px; } \
             .filler { height: 2000px; }";
        let l = layout_scrolled(html, css, (0.0, 500.0));
        let sticky = &body(&l).children[1];
        assert!(approx(sticky.rect.y, 100.0), "y={}", sticky.rect.y);
    }

    // --- abspos triage: `will-change: transform` establishes a
    //     containing block for fixed descendants (static case) ---

    #[test]
    fn fixed_cb_established_by_will_change_transform() {
        // CSS Transforms 1: `will-change: transform` makes the element a
        // containing block for fixed-position descendants. Static case
        // (no dynamic invalidation involved): the fixed child with
        // `inset: 0` resolves against the 310px-tall transformed
        // ancestor, so its height is 310.
        let html = r#"<div id="outer"><div id="f"></div></div>"#;
        let css = "body { margin: 0; } \
             #outer { will-change: transform; height: 310px; } \
             #f { position: fixed; inset: 0; }";
        let l = layout_plain(html, css);
        let f = &body(&l).children[0].children[0];
        assert!(approx(f.rect.h, 310.0), "h={}", f.rect.h);
    }

    // --- negative remainder with both `auto` margins (CSS 2.1 §10.3.7) ---

    const NEG_REM_HTML: &str = r#"<div id="cb"><div id="a"></div></div>"#;

    #[test]
    fn abspos_both_auto_margins_negative_remainder_ltr() {
        // CB = 400px; left:10, right:10, width:500 → remainder −120.
        // In LTR `margin-left` is pinned to 0 and the whole deficit goes
        // on `margin-right`, so the border box starts at the left inset
        // (x = 10), not centered at −50.
        let css = "body { margin: 0; } #cb { position: relative; width: 400px; } \
             #a { position: absolute; left: 10px; right: 10px; width: 500px; margin: 0 auto; }";
        let l = layout_plain(NEG_REM_HTML, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.w, 500.0), "w={}", a.rect.w);
        assert!(approx(a.rect.x, 10.0), "x={}", a.rect.x);
    }

    #[test]
    fn abspos_both_auto_margins_negative_remainder_rtl() {
        // Mirrored: in RTL `margin-right` is pinned to 0 and the deficit
        // goes on `margin-left`. The 400px CB itself sits at x=400 inside
        // the 800px RTL body, so the border box's right edge lands at the
        // right inset: x = 400 + 400 − 10 − 500 = 290.
        // (The body is RTL so the over-constrained #cb resolves per
        // CSS 2.1 §10.3.3 against its containing block's direction.)
        let css = "body { margin: 0; direction: rtl; } #cb { position: relative; width: 400px; direction: rtl; } \
             #a { position: absolute; left: 10px; right: 10px; width: 500px; margin: 0 auto; }";
        let l = layout_plain(NEG_REM_HTML, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.w, 500.0), "w={}", a.rect.w);
        assert!(approx(a.rect.x, 290.0), "x={}", a.rect.x);
    }

    #[test]
    fn abspos_both_auto_margins_positive_remainder_still_centers() {
        // Non-negative remainder keeps the old centering behavior:
        // CB 400, insets 10/10, width 100 → remainder 280, box at x = 150.
        let css = "body { margin: 0; } #cb { position: relative; width: 400px; } \
             #a { position: absolute; left: 10px; right: 10px; width: 100px; margin: 0 auto; }";
        let l = layout_plain(NEG_REM_HTML, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.w, 100.0), "w={}", a.rect.w);
        assert!(approx(a.rect.x, 150.0), "x={}", a.rect.x);
    }

    // --- abspos static position in flex containers (flexbox §4.2) ---

    /// The static position of an abspos flex child is derived as if it
    /// were the sole flex item: the main-start edge per `flex-direction`.
    /// 600x400 container, 100x50 child, all offsets `auto`, body margin
    /// zeroed so the container sits at the origin.
    #[test]
    fn abspos_flex_static_position_row_directions() {
        for (dir, ex, ey) in [
            ("row", 0.0, 0.0),
            ("row-reverse", 500.0, 0.0),
            ("column", 0.0, 0.0),
            ("column-reverse", 0.0, 350.0),
        ] {
            let css = format!(
                "body {{ margin: 0; }} \
                 .c {{ display: flex; position: relative; width: 600px; height: 400px; \
                      flex-direction: {dir}; }} \
                 .a {{ position: absolute; width: 100px; height: 50px; }}"
            );
            let l = layout_plain(r#"<div class="c"><div class="a"></div></div>"#, &css);
            let a = &body(&l).children[0].children[0];
            assert!(approx(a.rect.x, ex), "{dir}: x = {}", a.rect.x);
            assert!(approx(a.rect.y, ey), "{dir}: y = {}", a.rect.y);
        }
    }

    /// The cross axis honors `align-self` (`auto` falls back to the
    /// container's `align-items`): start/stretch at the cross-start edge,
    /// end at the cross-end edge, center in the middle.
    #[test]
    fn abspos_flex_static_position_align_self_cross_axis() {
        for (dir, self_align, ex, ey) in [
            ("row", "stretch", 0.0, 0.0),
            ("row", "flex-start", 0.0, 0.0),
            ("row", "flex-end", 0.0, 350.0),
            ("row", "center", 0.0, 175.0),
            ("row", "auto", 0.0, 350.0), // align-items: flex-end fallback
            ("column", "flex-start", 0.0, 0.0),
            ("column", "flex-end", 500.0, 0.0),
            ("column", "center", 250.0, 0.0),
        ] {
            let css = format!(
                "body {{ margin: 0; }} \
                 .c {{ display: flex; position: relative; width: 600px; height: 400px; \
                      flex-direction: {dir}; align-items: flex-end; }} \
                 .a {{ position: absolute; width: 100px; height: 50px; \
                      align-self: {self_align}; }}"
            );
            let l = layout_plain(r#"<div class="c"><div class="a"></div></div>"#, &css);
            let a = &body(&l).children[0].children[0];
            assert!(approx(a.rect.x, ex), "{dir} {self_align}: x = {}", a.rect.x);
            assert!(approx(a.rect.y, ey), "{dir} {self_align}: y = {}", a.rect.y);
        }
    }

    /// Margins shift the border-box origin off the content-box edges: the
    /// main-start edge gets the start margin, and cross alignment works
    /// on the margin box.
    #[test]
    fn abspos_flex_static_position_margins() {
        // `row`: main-start is the left edge, cross-start the top edge.
        let css = "body { margin: 0; } \
             .c { display: flex; position: relative; width: 600px; height: 400px; } \
             .a { position: absolute; width: 100px; height: 50px; \
                  margin: 10px 20px 30px 40px; }";
        let l = layout_plain(r#"<div class="c"><div class="a"></div></div>"#, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.x, 40.0), "x = {}", a.rect.x);
        assert!(approx(a.rect.y, 10.0), "y = {}", a.rect.y);

        // `row-reverse`: main-start is the right edge, so the end margin
        // (20px) applies on the main axis: x = 600 - 20 - 100 = 480.
        let css = "body { margin: 0; } \
             .c { display: flex; position: relative; width: 600px; height: 400px; \
                  flex-direction: row-reverse; } \
             .a { position: absolute; width: 100px; height: 50px; \
                  margin: 10px 20px 30px 40px; }";
        let l = layout_plain(r#"<div class="c"><div class="a"></div></div>"#, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.x, 480.0), "x = {}", a.rect.x);
        assert!(approx(a.rect.y, 10.0), "y = {}", a.rect.y);

        // Centered cross axis works on the margin box:
        // y = (400 - (10 + 50 + 30)) / 2 + 10 = 165.
        let css = "body { margin: 0; } \
             .c { display: flex; position: relative; width: 600px; height: 400px; } \
             .a { position: absolute; width: 100px; height: 50px; \
                  margin: 10px 20px 30px 40px; align-self: center; }";
        let l = layout_plain(r#"<div class="c"><div class="a"></div></div>"#, css);
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.x, 40.0), "x = {}", a.rect.x);
        assert!(approx(a.rect.y, 165.0), "y = {}", a.rect.y);
    }

    /// `justify-content` does not apply to abspos children, and in-flow
    /// siblings do not move the sole-item derivation: the child stays at
    /// the main-start edge even with `justify-content: flex-end` pushing
    /// the in-flow item to the far end.
    #[test]
    fn abspos_flex_static_position_ignores_justify_content_and_siblings() {
        let css = "body { margin: 0; } \
             .c { display: flex; position: relative; width: 600px; height: 400px; \
                  justify-content: flex-end; } \
             .i { width: 200px; height: 50px; } \
             .a { position: absolute; width: 100px; height: 50px; }";
        let l = layout_plain(
            r#"<div class="c"><div class="i"></div><div class="a"></div></div>"#,
            css,
        );
        let c = &body(&l).children[0];
        let (i, a) = (&c.children[0], &c.children[1]);
        assert!(approx(i.rect.x, 400.0), "i.x = {}", i.rect.x);
        assert!(approx(a.rect.x, 0.0), "a.x = {}", a.rect.x);
        assert!(approx(a.rect.y, 0.0), "a.y = {}", a.rect.y);
    }

    /// `direction: rtl` flips the row main axis (main-start becomes the
    /// right edge); `row-reverse` XORs with it back to the left edge.
    /// (In RTL the block-level container itself right-aligns in the
    /// viewport, so positions are asserted relative to the container.)
    #[test]
    fn abspos_flex_static_position_row_rtl() {
        for (dir, ex) in [("row", 500.0), ("row-reverse", 0.0)] {
            let css = format!(
                "body {{ margin: 0; }} \
                 .c {{ display: flex; position: relative; width: 600px; height: 400px; \
                      direction: rtl; flex-direction: {dir}; }} \
                 .a {{ position: absolute; width: 100px; height: 50px; }}"
            );
            let l = layout_plain(r#"<div class="c"><div class="a"></div></div>"#, &css);
            let c = &body(&l).children[0];
            let a = &c.children[0];
            assert!(
                approx(a.rect.x, c.rect.x + ex),
                "{dir}: x = {}",
                a.rect.x - c.rect.x
            );
            assert!(
                approx(a.rect.y, c.rect.y),
                "{dir}: y = {}",
                a.rect.y - c.rect.y
            );
        }
    }

    // --- abspos containing block of a multi-line positioned inline (CSS 2.1 §10.1) ---

    /// CSS 2.1 §10.1: when the containing-block-establishing ancestor is
    /// inline-level, the containing block is the bounding box around the
    /// ancestor's padding boxes — the union of its line-box fragments,
    /// not any single fragment. A positioned inline wrapping two lines
    /// of unequal length must resolve `right`/`bottom` against the
    /// union's bottom-right corner.
    #[test]
    fn abspos_cb_is_union_of_positioned_inline_fragments() {
        // 16px font at 9.6px/char: five "aaaaa" words in a 200px box put
        // three words (163.2px) on line 1 and two (105.6px) on line 2;
        // line height 19.2px. The inline's padding-box union is
        // (0,0)-(163.2,38.4), so `right: 0; bottom: 0` on the 10x10
        // abspos child lands at (153.2, 28.4). Against the first
        // fragment alone the y would be 9.2; against the last alone the
        // x would be 95.6.
        let html = r#"<div class="w"><span class="pos">aaaaa aaaaa aaaaa aaaaa aaaaa<span class="abs"></span></span></div>"#;
        let css = "body { margin: 0; } \
             .w { width: 200px; } \
             .pos { position: relative; } \
             .abs { position: absolute; right: 0; bottom: 0; width: 10px; height: 10px; }";
        let l = layout_plain(html, css);
        let pos = &body(&l).children[0].children[0];
        // Sanity: the inline really did split across two lines of
        // unequal length.
        assert!(approx(pos.rect.w, 163.2), "pos.w={}", pos.rect.w);
        assert!(approx(pos.rect.h, 38.4), "pos.h={}", pos.rect.h);
        let abs_ = pos
            .children
            .iter()
            .find(|c| c.box_type != BoxType::Text)
            .expect("abspos child");
        assert!(approx(abs_.rect.x, 153.2), "x={}", abs_.rect.x);
        assert!(approx(abs_.rect.y, 28.4), "y={}", abs_.rect.y);
    }

    // --- static position of inline-level abspos boxes (CSS 2.1
    //     §10.3.7/§10.6.4: the hypothetical in-flow position) ---

    /// The non-text child of `.w` (the abspos span) in the mid-content
    /// tests below.
    fn inline_abs_child(l: &LayoutBox, w: usize) -> &LayoutBox {
        body(l).children[w]
            .children
            .iter()
            .find(|c| c.box_type != BoxType::Text)
            .expect("abspos child")
    }

    #[test]
    fn abspos_inline_mid_content_static_position_adds_margins() {
        // The span sits between "aaa" and "bbb" in inline flow. With
        // all insets `auto` its static position is the hypothetical
        // in-flow position: the point just after "aaa" (18px at 10px
        // font, 0.6em/char) is the margin-box anchor, vertically the
        // line's baseline (9.6px at 10px font), and the used margins
        // shift the border box to (18 + 15, 9.6 + 5) = (33, 14.6).
        let html = r#"<div class="w">aaa<span id="a"></span>bbb</div>"#;
        let css = "body { margin: 0; } .w { font-size: 10px; } \
             #a { position: absolute; width: 20px; height: 20px; \
                  margin-left: 15px; margin-top: 5px; }";
        let l = layout_plain(html, css);
        let a = inline_abs_child(&l, 0);
        assert!(approx(a.rect.x, 33.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 14.6), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_inline_mid_content_static_position_without_margins() {
        // Control: no margins — the box anchors exactly at the end of
        // the preceding fragment horizontally (18) and at the line's
        // baseline vertically (9.6 at 10px font), mid-content rather
        // than at the content-box origin.
        let html = r#"<div class="w">aaa<span id="a"></span>bbb</div>"#;
        let css = "body { margin: 0; } .w { font-size: 10px; } \
             #a { position: absolute; width: 20px; height: 20px; }";
        let l = layout_plain(html, css);
        let a = inline_abs_child(&l, 0);
        assert!(approx(a.rect.x, 18.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 9.6), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_inline_static_position_uses_line_baseline() {
        // C2: the static position of an inline-level abspos box is the
        // hypothetical in-flow position (CSS 2.1 §10.3.7/§10.6.4). The
        // vertical anchor is the line box's baseline — where the inline
        // box would sit — not the preceding fragment's top. At 16px the
        // fragment top is 0 while the baseline sits one ascent lower
        // (0.8 * 1.2 * 16 = 15.36).
        let html = r#"<div class="w">ab<span id="a">x</span>cd</div>"#;
        let css = "body { margin: 0; } .w { font-size: 16px; } \
             #a { position: absolute; }";
        let l = layout_plain(html, css);
        let a = inline_abs_child(&l, 0);
        // Sanity: the horizontal anchor is still the end of "ab"
        // (2 chars * 0.6em * 16px).
        assert!(approx(a.rect.x, 19.2), "x={}", a.rect.x);
        let tb = body(&l).children[0]
            .children
            .iter()
            .find(|c| c.box_type == BoxType::Text)
            .expect("text box");
        let baseline = tb.fragments[0].baseline;
        assert!(
            approx(a.rect.y, baseline),
            "y={} baseline={}",
            a.rect.y,
            baseline
        );
    }

    #[test]
    fn abspos_inline_static_position_after_forced_break() {
        // C2: after a forced break the box anchors to the new line's
        // start, not to the previous line's fragment. (`<br>` as an
        // element carries no break in the inline walk, so the forced
        // break here is a `\n` in `pre-wrap` text — a hard-line start
        // per `Frag::forced_break_before`.)
        let html = "<div class=\"w\">ab\n<span id=\"a\">x</span>cd</div>";
        let css = "body { margin: 0; } .w { font-size: 16px; white-space: pre-wrap; } \
             #a { position: absolute; }";
        let l = layout_plain(html, css);
        let a = inline_abs_child(&l, 0);
        let w = &body(&l).children[0];
        // New line's start: the content-box origin x (left-aligned, no
        // indent, no floats) ...
        assert!(
            approx(a.rect.x, w.rect.x),
            "x={} content_x={}",
            a.rect.x,
            w.rect.x
        );
        // ... and the new line's baseline, read off the "cd" fragment
        // on the second line.
        let last_text = w
            .children
            .iter()
            .rfind(|c| c.box_type == BoxType::Text)
            .expect("text boxes");
        let baseline2 = last_text.fragments[0].baseline;
        assert!(
            approx(a.rect.y, baseline2),
            "y={} baseline2={}",
            a.rect.y,
            baseline2
        );
    }

    #[test]
    fn abspos_inline_at_content_start_adds_margins() {
        // No preceding fragment: the anchor is the content-box origin,
        // and the margins still shift the border box from it.
        let html = r#"<div class="w"><span id="a"></span>bbb</div>"#;
        let css = "body { margin: 0; } .w { font-size: 10px; } \
             #a { position: absolute; width: 20px; height: 20px; \
                  margin-left: 12px; margin-top: 3px; }";
        let l = layout_plain(html, css);
        let a = inline_abs_child(&l, 0);
        assert!(approx(a.rect.x, 12.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 3.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_inline_margins_apply_only_to_static_axis() {
        // `top` is specified, so the vertical axis resolves against the
        // containing block (the viewport) and never reads the static
        // position; the horizontal axis is all-`auto` and keeps the
        // static position plus `margin-left`.
        let html = r#"<div class="w">aaa<span id="a"></span>bbb</div>"#;
        let css = "body { margin: 0; } .w { font-size: 10px; } \
             #a { position: absolute; top: 40px; width: 20px; height: 20px; \
                  margin-left: 15px; }";
        let l = layout_plain(html, css);
        let a = inline_abs_child(&l, 0);
        assert!(approx(a.rect.x, 33.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 40.0), "y={}", a.rect.y);
    }

    #[test]
    fn abspos_block_level_margins_shift_static_position() {
        // Control for the block-level recording path: its static
        // position is the border-box origin flow would have given the
        // box, margins included — below the 100px spacer by the
        // collapsed 7px top margin, and 30px in from the content edge.
        let html = r#"<div><div class="sp"></div><div id="a"></div></div>"#;
        let css = "body { margin: 0; } .sp { height: 100px; } \
             #a { position: absolute; width: 50px; height: 20px; \
                  margin-left: 30px; margin-top: 7px; }";
        let l = layout_plain(html, css);
        let a = &body(&l).children[0].children[1];
        assert!(approx(a.rect.x, 30.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 107.0), "y={}", a.rect.y);
    }
}
