//! v1 multi-column layout (CSS Multicol 1, subset).
//!
//! A block container with `column-count` / `column-width` resolving to 2+
//! columns becomes a *multicol container*: its children lay out against one
//! column's width (narrowed in `layout_block`'s prologue), then the
//! completion paths redistribute them across the columns —
//! [`balance_block_columns`] for block children (called from
//! `resume_block`), [`distribute_inline_lines`] for line boxes (called from
//! `do_inline_layout`). Redistribution only records deferred
//! (`pend_x`/`pend_y`) shifts, so the single `apply_pending_shifts` pass at
//! the end of `layout_forest` moves whole subtrees; parents position
//! following siblings off the container's corrected height, so no
//! sibling/ancestor fix-up pass is needed.
//!
//! Column count resolution (CSS Multicol 1 §3.4): `column-count: N` wins
//! when `column-width` is `auto`; a lone `column-width: W` derives
//! `N = max(1, floor((avail + gap) / (W + gap)))`; when both are set,
//! `N = min(count, derived)`. `column-count: 1` (or a single derived
//! column) is plain block layout. `column-gap: normal` computes to 1em;
//! the grid/flex `column-gap` shares the property name and keeps treating
//! `normal` as 0 (see `layout_grid::parse_gaps`).
//!
//! # Documented v1 simplifications
//!
//! - Fill: with an auto height, columns are balanced — the column target
//!   is `max(total_height / N, tallest_unsplittable_child)` and children
//!   fill column by column; lumpy content can leave the last column taller
//!   than the target (the container grows to fit). Splittable children
//!   fragment instead of forcing the target up. Under a definite height,
//!   `column-fill: balance` (the initial value) caps the target at the
//!   height, while `column-fill: auto` fills columns sequentially to the
//!   brim (registered in `bowser_css` as `props::COLUMN_FILL`).
//! - Block-level children fragment across column boundaries at line-box
//!   boundaries (band slicing mirroring the page splitter); children with
//!   `break-inside: avoid`, nested multicol containers, flex containers,
//!   and children without line boxes still move whole (overflowing the last
//!   column when nothing fits).
//! - Line boxes are the atomic unit for inline content; an inline box
//!   (e.g. a `<span>` background) straddling a column break gets the union
//!   rect of its fragments rather than per-column fragments.
//! - Floats are assigned to the column their in-flow position falls
//!   into (banded by the balance target, per spanner segment), keeping
//!   the within-column placement the flow pass computed against one
//!   column width; a float that does not fit the column's remaining
//!   height whole moves to the next column's top. Per-column float lists
//!   are real: column 0 keeps the flow pass's exact intrusion, later
//!   columns recompute it against the floats assigned there, so moved
//!   block children and lines wrap around (or, when narrower than the
//!   broken width, step below) column-2+ floats. `clear` is re-evaluated
//!   per column (push-down only; stale flow-pass clearance is never
//!   undone). Lines moved to later columns are re-broken against that
//!   column's float geometry: over-wide lines wrap beside the floats and
//!   under-full lines pull following words up (pre-formatted runs keep
//!   their flow-pass shape). A float pushed whole out of column 1 no
//!   longer leaves a phantom indent on column-1 content: the column-0
//!   shift is recomputed against the column's placed floats, shedding
//!   intrusion from departed floats (never growing past the flow pass's
//!   shift). Known approximations: narrowed BFC widths are kept
//!   (re-expanding them needs a child re-layout the balance pass cannot
//!   do); ancestor floats intruding into columns past the first are shed
//!   as before (only the container's own floats are assigned); the
//!   column target `col_h` is computed from flow-pass line totals; when
//!   mid-distribution re-breaking (wrap/pull-up) changes the line count,
//!   a second balance pass re-derives the target from the actual
//!   post-rebreak line heights and re-distributes, restoring the balance
//!   (minimize the maximum column height). When re-breaking changes
//!   nothing the fast path stays single-pass.
//! - `column-span: all` places the child in a full-width spanner row; the
//!   segments before and after balance independently. The flow pass lays
//!   the spanner's inner content out against the full content width up
//!   front (see the `layout_block` prologue in `layout.rs`); the balance
//!   pass only widens its border box to the same width.
//! - `column-rule` is resolved by the distribution passes (see
//!   [`take_column_rules`]) and painted by the paint worker
//!   (`paint_column_rules` in `bowser_paint`'s display list builder).
//! - Forced `break-before` / `break-after: column` (plus the page-break
//!   aliases, which fragment as column breaks inside multicol, and
//!   `always`) are honored for block children; a forced break into a fresh
//!   column is a no-op (no empty column is created). Inline-level forced
//!   breaks are honored too: `do_inline_layout` splits line boxes at the
//!   break and reports the following line's index in
//!   `InlineLayoutOutput::forced_break_lines`, and
//!   [`distribute_inline_lines`] starts a fresh column at each reported
//!   index (a no-op when the line already begins a fresh column; past the
//!   last column the line continues below the column block).
//! - Absolutely positioned descendants: block-level ones are recorded
//!   during the single-column flow pass, then [`balance_block_columns`]
//!   re-bases each direct abspos child's static position onto the column
//!   its in-flow placeholder lands in, so `auto` offsets resolve in final
//!   column geometry. Abspos boxes nested inside a moved in-flow child
//!   need no adjustment — they ride that child's deferred shift — and
//!   inline-level abspos descendants already resolve after line
//!   distribution (see `do_inline_layout`).
//! - A definite container height is the per-column height; content past
//!   `N` columns overflows the last column (and the container, per
//!   `overflow`).
//! - Nested multicol containers distribute inner-first (the child's
//!   `resume_block` completes before the parent's), so the outer pass
//!   treats an inner container as one atomic child.

use super::{FloatCtx, FloatEntry, Frag, Line};
use crate::BoxType;
use crate::boxes::{Forest, is_out_of_flow};
use crate::hash::{FxBuildHasher, FxHashMap};
use crate::style::{Clear, Float, WhiteSpace, sane_len, spec_len_of};
use bowser_css::{ComputedStyle, CssValue, props};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Tolerance for float dust when testing a child against the column
/// height (px). Layout sums lengths in f32, so a child meant to sit
/// exactly on the boundary can land a fraction of a sub-pixel off.
const EPS: f32 = 0.001;

/// Hard cap on the column count: a hostile `column-count: 1e30` must
/// size vectors sanely, not OOM the tab.
const MAX_COLUMNS: usize = 1024;

// One whole-subtree move performed by [`split_off_below`]'s `Keep`
// arm: `child` was reparented out of `parent`'s children (at `index`,
// its position in the pre-split child list) into the continuation
// subtree. `rel_y` is the child's pre-split border-box top relative to
// the parent's (`child.rect.y - parent.rect.y`), captured before
// `intersect_band` mutates the moved rect. A re-layout over the same
// forest (flex-column measure→real, flex stretch re-layout, grid second
// pass) re-runs the balance with the old continuations still in
// `children`; the stale strip re-inserts each moved child at its
// recorded index and shifts its subtree back by exactly `rel_y`, so the
// second balance pass re-splits the original geometry instead of a
// below-band-shaped hole.
struct SplitMove {
    parent: usize,
    child: usize,
    index: usize,
    rel_y: f32,
}

/// The destructive footprint of one [`split_off_below`] call, recorded
/// so the stale strip can exactly undo it before re-partitioning:
///
/// - `continuation` / `original`: the split's below-band continuation
///   root and the original child it was split from (the old pair).
/// - `moves`: every whole subtree the `Keep` arm reparented out of the
///   original (in kid order per parent, so ascending inserts restore the
///   exact child order). Processed in reverse split order: a child moved
///   twice (a re-split continuation's move) lands back at its earliest
///   origin.
/// - `shells`: `(original shell, pre-split rect.h)` for every
///   straddling `(original, continuation-shell)` pair. The second pass's
///   flow re-computes these heights from the damaged (child-missing)
///   subtrees; restoring them keeps the re-split's band classification
///   exact.
struct SplitRecord {
    continuation: usize,
    original: usize,
    moves: Vec<SplitMove>,
    shells: Vec<(usize, f32)>,
}

// Split-continuation records left by the last `balance_block_columns`
// pass, keyed by the container's arena index. A re-layout over the same
// forest (flex-column measure→real, flex stretch re-layout, grid second
// pass) re-runs the balance with the old continuations still in
// `children`; the balance entry strips them before partitioning so a
// stale clone is never re-split, and re-inserts the recorded whole
// moved subtrees into the original (exact order restoration, pre-split
// geometry) so the second pass operates on the full content, not the
// above-band remainder. A record from a dropped forest cannot
// false-positive: stripping additionally requires the pair to sit
// adjacent in this container's `children` with the same DOM node — an
// exact-clone signature a fresh layout cannot produce by accident.
thread_local! {
    static MULTICOL_CONTINUATIONS: RefCell<HashMap<usize, Vec<SplitRecord>>> =
        RefCell::new(HashMap::new());
}

// Previously applied static-position deltas for
// `adjust_abspos_static_positions`, keyed by abspos node:
//
// `(prev_x, prev_y, dx, dy)` — the recorded static position before this
// container's adjustment and the delta that was applied. Balance can run
// twice over one forest (flex measure→real, grid second pass) before the
// abspos post-pass consumes the items, so the earlier delta is reverted
// before the fresh one is applied; the stored pre-adjustment position
// authenticates the record, so a fresh forest reusing arena indices is
// never mis-reverted.
thread_local! {
    static ABS_STATIC_DELTAS: RefCell<HashMap<usize, (f32, f32, f32, f32)>> =
        RefCell::new(HashMap::new());
}

/// `column-rule-style` line styles that draw a rule. `none` and `hidden`
/// never appear here: [`resolve_column_rule`] maps them to "no rule".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnRuleStyle {
    Solid,
    Dashed,
    Dotted,
    Double,
    Groove,
    Ridge,
    Inset,
    Outset,
}

/// Resolved `column-rule` geometry for one multicol container, recorded
/// during column distribution for a later paint worker to turn into
/// display-list items.
///
/// Matching: the paint worker drains the table once per layout pass with
/// [`take_column_rules`], then matches each spec to its container — by
/// `dom_node` against `LayoutBox::node` when `Some`, else by geometry:
/// recompute the expected rule offsets/height for each rule-bearing
/// multicol container and take the spec with equal `x_offsets`/`height`
/// (specs with identical geometry are interchangeable, so an approximate
/// match draws the right pixels). Specs that match no final container
/// (stale re-layout records) draw nothing and are ignored.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnRuleSpec {
    /// Arena index of the multicol container. `Some` on both distribution
    /// paths ([`distribute_inline_lines_for_container`] receives it from
    /// `do_inline_layout`); `None` only for specs recorded through the
    /// legacy [`distribute_inline_lines`] wrapper, which has no identity
    /// to pass. Re-layout dedupe prefers [`ColumnRuleSpec::dom_node`]
    /// (stable across passes); the index only identifies a container
    /// within one pass's forest.
    pub container: Option<usize>,
    /// DOM node of the container when known (`Some` for element boxes on
    /// both distribution paths; the inline path receives it via
    /// [`distribute_inline_lines_for_container`]). `None` for anonymous
    /// containers and for specs recorded through the legacy
    /// [`distribute_inline_lines`] wrapper.
    pub dom_node: Option<bowser_dom::NodeId>,
    /// Center x of each rule (one per column gap, `columns - 1` entries),
    /// in px relative to the container's content-box left edge. Relative
    /// — not absolute — so an ancestor column distribution that shifts
    /// the container after it recorded cannot stale the geometry: the
    /// paint worker adds the container's final content-box x.
    pub x_offsets: Vec<f32>,
    /// Rule height in px: the container's final content height (the value
    /// the distributing pass returned).
    pub height: f32,
    /// Resolved rule width in px (`thin`/`medium`/`thick` → 1/3/5).
    pub width: f32,
    /// Rule line style.
    pub style: ColumnRuleStyle,
    /// Resolved rule color (`currentcolor` → the container's used `color`).
    pub color: bowser_css::Color,
}

// Recorded `ColumnRuleSpec`s, in record order. Drained per layout pass
// by `take_column_rules`, and cleared at the start of every pass by
// `begin_multicol_pass` so a re-layout never accumulates stale specs
// when the drain is skipped.
thread_local! {
    static COLUMN_RULES: RefCell<Vec<ColumnRuleSpec>> = const { RefCell::new(Vec::new()) };
}

/// Drain and return every recorded [`ColumnRuleSpec`], clearing the
/// table. The paint worker calls this once per layout pass and emits one
/// display-list item per rule in each spec.
///
/// Re-export note: `layout_multicol` is `pub(crate)` under the private
/// `layout` module; expose this to `bowser_paint` with
/// `pub use layout::layout_multicol::{ColumnRuleSpec, ColumnRuleStyle, take_column_rules};`
/// in `lib.rs`.
#[allow(dead_code)] // Consumed by bowser_paint once the lib.rs re-export lands.
pub fn take_column_rules() -> Vec<ColumnRuleSpec> {
    COLUMN_RULES.with(|t| t.borrow_mut().drain(..).collect())
}

/// Record one container's [`ColumnRuleSpec`]. A re-layout re-records the
/// same container, so a stale spec with the same identity is dropped
/// first (see [`same_rule_container`]): without this, a re-layout with
/// changed geometry leaves the old spec alongside the new one — table
/// growth across passes, and a stale spec claimed by an earlier-visited
/// container can leave the rightful container rule-less. Exact
/// duplicates (unchanged re-layouts) are not recorded twice.
fn record_column_rule(spec: ColumnRuleSpec) {
    COLUMN_RULES.with(|t| {
        let mut t = t.borrow_mut();
        t.retain(|s| !same_rule_container(s, &spec));
        if !t.contains(&spec) {
            t.push(spec);
        }
    });
}

/// Whether two recorded specs belong to the same multicol container.
/// Prefers the DOM node: arena indices are rebuilt on every layout pass,
/// so an index only identifies a container within one pass's forest,
/// while the DOM node is stable across re-layouts of the same document.
/// Specs with no identity on either side (the legacy inline path records
/// `container: None` / `dom_node: None`) never match: they keep the old
/// exact-duplicate suppression in [`record_column_rule`].
fn same_rule_container(a: &ColumnRuleSpec, b: &ColumnRuleSpec) -> bool {
    match (a.dom_node, b.dom_node) {
        (Some(x), Some(y)) => x == y,
        _ => matches!((a.container, b.container), (Some(x), Some(y)) if x == y),
    }
}

/// Center x offsets (relative to the content-box left edge) of the
/// `count - 1` column rules: each rule is centered in its column gap.
/// `count >= 2` always (see [`multicol_params`]).
fn rule_x_offsets(mc: &MulticolParams) -> Vec<f32> {
    (0..mc.count - 1)
        .map(|k| mc.col_w + mc.gap / 2.0 + k as f32 * (mc.col_w + mc.gap))
        .collect()
}

/// Resolved `column-rule` line appearance (width/style/color), or `None`
/// when no rule is drawn. Resolved once in [`multicol_params`] and handed
/// to the distribution passes via [`PENDING_RULES`] (it cannot ride on
/// [`MulticolParams`]: that struct is also built literally by tests in
/// `layout.rs`, so it cannot gain a field).
#[derive(Clone, Copy, Debug)]
struct ResolvedColumnRule {
    width: f32,
    style: ColumnRuleStyle,
    color: bowser_css::Color,
}

// Resolved `column-rule`s awaiting their container's distribution pass,
// in prologue order: `multicol_params` pushes one entry per multicol
// container; `balance_block_columns` / `distribute_inline_lines` pop
// exactly one via `pop_pending_rule`. Nesting is LIFO — an inner
// container's whole layout nests inside the outer's — so the matching
// entry is found by scanning from the top for the container's params.
// Entries for empty containers (whose `layout_block` returns before
// distributing) are dead; the scan drops them opportunistically, and
// `begin_multicol_pass` clears the whole table at the start of every
// layout pass so dead entries can never leak across passes. Direct
// `do_inline_layout` test calls bypass the prologue, so an empty scan
// simply yields no rule.
thread_local! {
    static PENDING_RULES: RefCell<Vec<(MulticolParams, Option<ResolvedColumnRule>)>> =
        const { RefCell::new(Vec::new()) };
}

/// Drop all per-pass multicol state at the start of a layout pass: dead
/// [`PENDING_RULES`] entries (one leaks per empty multicol container per
/// pass otherwise, and a dead entry lingering into the next pass can
/// hijack a later container's rule when the params match) and recorded
/// [`ColumnRuleSpec`]s (stale across re-layouts otherwise). The
/// coordinator calls this once per pass at the top of `layout_forest`
/// (`crates/bowser_layout/src/layout.rs`).
pub(crate) fn begin_multicol_pass() {
    PENDING_RULES.with(|t| t.borrow_mut().clear());
    COLUMN_RULES.with(|t| t.borrow_mut().clear());
}

/// Pop this container's pending rule (pushed by [`multicol_params`]).
/// Scans from the top for `mc`'s params, dropping dead entries from empty
/// containers. A live entry can never sit above the popping container's
/// own entry (LIFO nesting), so dropping on mismatch is safe. Residual
/// risk: two containers with byte-identical multicol params where the
/// inner one is empty — the outer then inherits the inner's rule.
/// (Requires identical count/gap/width/fill *and* an empty inner multicol
/// container *and* differing rules.)
fn pop_pending_rule(mc: &MulticolParams) -> Option<ResolvedColumnRule> {
    PENDING_RULES.with(|t| {
        let mut t = t.borrow_mut();
        while let Some((params, rule)) = t.pop() {
            if params == *mc {
                return rule;
            }
            // Dead entry (empty container that never distributed): dropped.
        }
        None
    })
}

/// Resolve the `column-rule` longhands to a drawable rule, or `None`
/// when nothing draws: `column-rule-style` is `none` (the initial),
/// `hidden`, or unrecognized, or the resolved width is not positive.
/// Width keywords follow the CSS 2.1 §8.5.3 UA convention
/// (`thin`/`medium`/`thick` → 1/3/5px); `currentcolor` resolves to the
/// container's used `color` (the cascade never leaves `color` itself as
/// `currentcolor`), mirroring the opaque-black fallback `claim_color`
/// (layout_table.rs) uses for border colors.
fn resolve_column_rule(
    style: Option<&ComputedStyle>,
    font_size: f32,
    root_font_size: f32,
    vp_w: f32,
    vp_h: f32,
    content_w: f32,
) -> Option<ResolvedColumnRule> {
    let style = style?;
    // Initial `none` (missing) yields no rule. CSS keywords are ASCII
    // case-insensitive, so `eq_ignore_ascii_case` avoids the per-box
    // `to_ascii_lowercase` allocation.
    let rule_style = match style.get_str_by_index(props::COLUMN_RULE_STYLE)? {
        s if s.eq_ignore_ascii_case("solid") => ColumnRuleStyle::Solid,
        s if s.eq_ignore_ascii_case("dashed") => ColumnRuleStyle::Dashed,
        s if s.eq_ignore_ascii_case("dotted") => ColumnRuleStyle::Dotted,
        s if s.eq_ignore_ascii_case("double") => ColumnRuleStyle::Double,
        s if s.eq_ignore_ascii_case("groove") => ColumnRuleStyle::Groove,
        s if s.eq_ignore_ascii_case("ridge") => ColumnRuleStyle::Ridge,
        s if s.eq_ignore_ascii_case("inset") => ColumnRuleStyle::Inset,
        s if s.eq_ignore_ascii_case("outset") => ColumnRuleStyle::Outset,
        // "none", "hidden", and unrecognized keywords: no rule.
        _ => return None,
    };
    let width = match style.get_by_index(props::COLUMN_RULE_WIDTH) {
        Some(CssValue::Keyword(k)) => {
            if k.eq_ignore_ascii_case("thin") {
                1.0
            } else if k.eq_ignore_ascii_case("thick") {
                5.0
            } else {
                // "medium" (the initial) and unrecognized keywords.
                3.0
            }
        }
        v => spec_len_of(v)
            .resolve(font_size, root_font_size, vp_w, vp_h, content_w)
            .map(sane_len)
            .unwrap_or(3.0)
            .max(0.0),
    };
    if width <= 0.0 {
        return None;
    }
    Some(ResolvedColumnRule {
        width,
        style: rule_style,
        color: resolve_rule_color(style),
    })
}

/// Resolve `column-rule-color`: a concrete color is used directly;
/// `currentcolor` (the initial) resolves to the container's used `color`;
/// anything else falls back to opaque black.
fn resolve_rule_color(style: &ComputedStyle) -> bowser_css::Color {
    const BLACK: bowser_css::Color = bowser_css::Color {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    let text_color = match style.get_by_index(props::COLOR) {
        Some(CssValue::Color(c)) => *c,
        _ => BLACK,
    };
    match style.get_by_index(props::COLUMN_RULE_COLOR) {
        Some(CssValue::Color(c)) => *c,
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("currentcolor") => text_color,
        // Missing → initial `currentcolor`.
        None => text_color,
        _ => BLACK,
    }
}

/// Resolved multi-column parameters for one container.
///
/// `PartialEq` supports the pending column-rule handoff (see
/// `PENDING_RULES`): the entry is matched to the distributing container
/// by its params.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MulticolParams {
    /// Column count (>= 2; anything less is plain block layout).
    pub count: usize,
    /// Used gap between adjacent columns, in px.
    pub gap: f32,
    /// Used column width, in px.
    pub col_w: f32,
    /// `column-fill: auto`: fill columns sequentially under a definite
    /// height instead of balancing. With an auto height `auto` behaves as
    /// `balance` (CSS Multicol 1). Read from the container's style here
    /// because the distribution passes cannot see it.
    pub fill_auto: bool,
}

/// Resolve `column-count` / `column-width` / `column-gap` for a block
/// container with content width `content_w`. Returns `None` when the
/// container is not a multicol container (both longhands `auto`, or the
/// resolved count is < 2).
pub(super) fn multicol_params(
    style: Option<&ComputedStyle>,
    font_size: f32,
    root_font_size: f32,
    vp_w: f32,
    vp_h: f32,
    content_w: f32,
) -> Option<MulticolParams> {
    // `column-count: <integer> | auto`.
    let count = match style.and_then(|s| s.get_by_index(props::COLUMN_COUNT)) {
        Some(CssValue::Number(n)) if n.is_finite() && *n >= 1.0 => {
            Some((*n as usize).clamp(1, MAX_COLUMNS))
        }
        _ => None,
    };
    // `column-width: <length> | auto`; a non-positive used width is
    // degenerate (infinite columns), so it counts as `auto`.
    let width = match style.and_then(|s| s.get_by_index(props::COLUMN_WIDTH)) {
        Some(v) => {
            let w = spec_len_of(Some(v))
                .resolve(font_size, root_font_size, vp_w, vp_h, content_w)
                .map(sane_len)
                .unwrap_or(0.0);
            if w > 0.0 { Some(w) } else { None }
        }
        None => None,
    };
    // `column-gap: <length> | normal`; `normal` is 1em in multicol.
    let gap = match style.and_then(|s| s.get_by_index(props::COLUMN_GAP)) {
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("normal") => font_size,
        v => spec_len_of(v)
            .resolve(font_size, root_font_size, vp_w, vp_h, content_w)
            .map(sane_len)
            .unwrap_or(font_size)
            .max(0.0),
    };
    // CSS Multicol 1 §3.4.
    let derived = |w: f32| ((content_w + gap) / (w + gap)).floor().max(1.0) as usize;
    let mut n = match (count, width) {
        (Some(c), None) => c,
        (None, Some(w)) => derived(w),
        (Some(c), Some(w)) => c.min(derived(w)),
        (None, None) => return None,
    };
    n = n.clamp(1, MAX_COLUMNS);
    if n < 2 {
        return None;
    }
    let col_w = ((content_w - (n as f32 - 1.0) * gap) / n as f32).max(0.0);
    // `column-fill: auto | balance` (initial `balance`).
    let fill_auto = matches!(
        style.and_then(|s| s.get_str_by_index(props::COLUMN_FILL)),
        Some(v) if v.eq_ignore_ascii_case("auto")
    );
    let mc = MulticolParams {
        count: n,
        gap,
        col_w,
        fill_auto,
    };
    // Hand the resolved `column-rule` to the distribution pass (see
    // `PENDING_RULES`): the passes receive `mc` but not the style.
    let rule = resolve_column_rule(style, font_size, root_font_size, vp_w, vp_h, content_w);
    PENDING_RULES.with(|t| t.borrow_mut().push((mc, rule)));
    Some(mc)
}

/// Forced break for `break-before` / `break-after` in a column context:
/// `column` plus the page-break aliases (which fragment as column breaks
/// inside a multicol container, CSS Fragmentation 3 §3.1) and `always`.
fn forced_col_break(style: Option<&ComputedStyle>, prop: u8) -> bool {
    super::forced_break_of(style, prop)
}

/// `break-inside: avoid` (or its column/page/region aliases): the box
/// never fragments across columns.
fn break_inside_avoid(style: Option<&ComputedStyle>) -> bool {
    matches!(
        style.and_then(|s| s.get_str_by_index(props::BREAK_INSIDE)),
        Some(v) if v.eq_ignore_ascii_case("avoid")
            || v.eq_ignore_ascii_case("avoid-column")
            || v.eq_ignore_ascii_case("avoid-page")
            || v.eq_ignore_ascii_case("avoid-region")
    )
}

/// `(orphans, widows)` for a block child (CSS Fragmentation 3 §5;
/// initial 2, clamped to a sane bound).
fn orphans_widows_of(style: Option<&ComputedStyle>) -> (u32, u32) {
    fn prop_u32(style: Option<&ComputedStyle>, prop: u8) -> u32 {
        style
            .and_then(|s| s.get_by_index(prop))
            .and_then(|v| match v {
                CssValue::Number(n) if *n >= 1.0 => Some((*n as u32).min(10_000)),
                _ => None,
            })
            .unwrap_or(2)
    }
    (
        prop_u32(style, props::ORPHANS),
        prop_u32(style, props::WIDOWS),
    )
}

/// Whether a child can ever split at a line boundary (cheap,
/// style-level): a plain block box without `break-inside: avoid` and
/// without nested multicol / flex content. The line-count
/// (orphans/widows) gate is checked per split attempt in
/// [`try_split_child`].
fn maybe_splittable(forest: &Forest, c: usize) -> bool {
    let nd = &forest.nodes[c];
    matches!(nd.box_type, BoxType::Block | BoxType::AnonymousBlock)
        && nd.multicol.is_none()
        && super::flex_kind(forest, c).is_none()
        && !break_inside_avoid(nd.style.as_deref())
}

/// Whether `idx` is a `column-span: all` spanner: a block-level, in-flow,
/// non-floated box opting into a full-width spanner row. (Only direct
/// children of the multicol container are considered, and the caller
/// already filters to those.)
pub(super) fn is_spanner(forest: &Forest, idx: usize) -> bool {
    matches!(
        forest.nodes[idx].box_type,
        BoxType::Block | BoxType::AnonymousBlock
    ) && !super::is_float(forest, idx)
        && !is_out_of_flow(forest, idx)
        && matches!(
            forest.nodes[idx]
                .style
                .as_ref()
                .and_then(|s| s.get_str_by_index(props::COLUMN_SPAN)),
            Some(v) if v.eq_ignore_ascii_case("all")
        )
}

/// Width of a full-width spanner row: the whole multicol content width.
/// The `layout_block` prologue lays spanner children out against this up
/// front; [`balance_block_columns`] widens the spanner's border box to
/// the same width after the segments balance.
pub(super) fn spanner_row_width(mc: &MulticolParams) -> f32 {
    mc.count as f32 * mc.col_w + (mc.count as f32 - 1.0) * mc.gap
}

/// Y placement of a unit in `col` of the current column row: a fresh
/// column starts at the row top — except column 0 of the first row keeps
/// its original y so the first child's position (and any collapsed top
/// margin) is unchanged — while rows after a spanner preserve the flow
/// gap below the spanner. Otherwise the original gap to the previous
/// occupant of the column is preserved.
#[allow(clippy::too_many_arguments)]
fn place_y(
    col: usize,
    fresh: bool,
    orig_y: f32,
    new_bottom: &[f32],
    orig_bottom: &[f32],
    row_top_placed: f32,
    row_orig_top: f32,
    first_row: bool,
) -> f32 {
    if fresh {
        if first_row && col > 0 {
            // The first row starts at the container's content top, which
            // is `row_top_placed` by construction.
            row_top_placed
        } else {
            row_top_placed + (orig_y - row_orig_top)
        }
    } else {
        new_bottom[col] + (orig_y - orig_bottom[col])
    }
}

/// A float assigned to a column: the placed [`FloatEntry`] plus the
/// float's document order (its position in the container's child list;
/// continuations inherit their original's). The (d) `clear` fix-up only
/// clears *preceding* floats (CSS 2.1 §9.5.2), so the order travels with
/// the entry. Entries built from context floats rather than direct
/// children (the inline distributor) carry `usize::MAX`: no order info,
/// and no fix-up consumes it there.
type ColFloat = (FloatEntry, usize);

/// Horizontal space taken by one column's placed floats across the
/// vertical band `[y0, y1)`: the x shift a box in this column and band
/// needs to sit past the column's left floats, and the remaining width.
/// Mirrors `line_float_avail` (layout.rs), but the per-column lists are
/// pre-filtered to the floats horizontally overlapping the column, so no
/// content-x recentering is needed: entries are absolute, in the placed
/// frame.
fn column_float_insets(
    col_floats: &[ColFloat],
    col_x: f32,
    col_w: f32,
    y0: f32,
    y1: f32,
) -> (f32, f32) {
    let mut left: f32 = 0.0;
    let mut right: f32 = 0.0;
    for (f, _) in col_floats {
        if f.y < y1 && f.y + f.h > y0 {
            if f.left {
                left = left.max(f.x + f.w - col_x);
            } else {
                right = right.max(col_x + col_w - f.x);
            }
        }
    }
    (left.max(0.0), (col_w - left - right).max(0.0))
}

/// Record the deferred shifts moving child `c` into column `col` at
/// vertical position `ny`. Per-column float lists: column 0 keeps the
/// flow pass's float intrusion (`dx_old`, recovered from the border-box
/// x minus the used left margin), minus the phantom kind — a float
/// pushed whole out of column 1 by [`assign_column_floats`] leaves
/// column-1 content the flow pass intruded beside it with a stale
/// indent, so the intrusion is recomputed against the floats still in
/// the column (direct floats in `col_floats` at placed geometry, other
/// context floats in `ctx_floats`) and whatever is gone is shed — never
/// grown past the flow pass's shift, and never shedding intrusion from
/// ancestor floats (they never move). Later columns recompute the shift
/// against the floats assigned to that column (`col_floats`, in the
/// placed frame), so content wrapping around a column-2+ float shifts
/// past it instead of shedding the intrusion or overlapping the float.
/// Moved children keep their left margin plus the new intrusion.
/// Whether `node` is `ancestor` itself or nested under it (parent-chain
/// walk).
fn under_or_self(forest: &Forest, mut node: usize, ancestor: usize) -> bool {
    loop {
        if node == ancestor {
            return true;
        }
        match forest.nodes.get(node).and_then(|n| n.parent) {
            Some(p) => node = p,
            None => return false,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn place_child(
    forest: &mut Forest,
    c: usize,
    col: usize,
    mc: &MulticolParams,
    content_x: f32,
    ny: f32,
    orig_x: f32,
    orig_y: f32,
    ml: f32,
    h: f32,
    col_floats: &[ColFloat],
    ctx_floats: &[FloatEntry],
) {
    let col_x = col as f32 * (mc.col_w + mc.gap);
    let dx_old = orig_x - content_x - ml;
    // Column 0 keeps the flow pass's intrusion, except the phantom kind:
    // a float pushed whole out of column 1 by `assign_column_floats`
    // (b) is gone from this column's placed floats, so the indent the
    // flow pass put beside it is shed. Intrusion from floats still here
    // is kept: direct floats travel in `col_floats` at placed geometry,
    // other context floats (ancestors, which never move, and nested
    // floats at flow geometry) in `ctx_floats`. A float nested inside
    // this child sits inside its border box and would inflate the inset
    // past the flow pass's shift, so it is skipped. The shift never grows
    // past the flow pass's (the flow pass already cleared and fit
    // against these floats). Later columns recompute the intrusion
    // against the floats assigned there — but only for BFC children: the
    // flow pass shifts BFC boxes past floats (narrowing them), while
    // non-BFC boxes keep full width and let floats intrude into their
    // content instead, so shifting them would double-count the intrusion
    // their lines already carry.
    let dx_new = if col == 0 {
        let mut placed = column_float_insets(col_floats, content_x, mc.col_w, ny, ny + h).0;
        for e in ctx_floats {
            if !e.left || under_or_self(forest, e.node, c) {
                continue;
            }
            if e.y < ny + h && e.y + e.h > ny {
                placed = placed.max(e.x + e.w - content_x);
            }
        }
        dx_old.min(placed.max(0.0))
    } else if super::establishes_bfc(forest, c) {
        column_float_insets(col_floats, content_x + col_x, mc.col_w, ny, ny + h).0
    } else {
        0.0
    };
    // Keep the intrusion strictly below one column width, with a full
    // EPS of margin past the recovery formula's decision boundary: the
    // abspos-anchor recovery inverts `content_x + k * stride + ml +
    // dx_new` for `k` by backing one column width off and rounding up,
    // which mis-recovers `k + 1` when `dx_new` lands within EPS of
    // `col_w` — and f32 dust at the exact `col_w - EPS` boundary could
    // still tip the `ceil` either way, so the clamp stops one EPS short
    // of it. The clamp moves the placed x by less than two thousandths
    // of a pixel.
    let dx_new = dx_new.min(mc.col_w - 2.0 * EPS).max(0.0);
    // Assigned, not accumulated: the flow pass leaves a direct multicol
    // child's own pending shift at 0, and nothing between passes adds to
    // it (table placement targets descendants), so the computed total is
    // the whole shift. Accumulating here double-applied the offset when a
    // measure pass and the real pass both ran the balance with no clear
    // in between. This matches `record_shift`'s set-not-accumulate
    // convention: rects stay provisional until `apply_pending_shifts`.
    let nd = &mut forest.nodes[c];
    nd.pend_x = content_x + col_x + ml + dx_new - orig_x;
    nd.pend_y = ny - orig_y;
}

/// Per-column float fix-ups for a block child placed at `ny` in column
/// `col`, applied before [`place_child`] records the shift:
///
/// (d) `clear` is re-evaluated against the column's placed floats: the
/// flow pass cleared against column-1 floats, so a child moved to a
/// column whose floats sit lower is pushed below them. One-sided: stale
/// flow-pass clearance is never undone (the hypothetical position is not
/// recoverable post-hoc). Per CSS 2.1 §9.5.2 only *preceding* floats
/// clear: floats at or after the child's document order (`child_order`)
/// are skipped, so a `clear` never pushes the child below a float that
/// follows it in DOM order.
///
/// (e) A BFC child's used width was narrowed beside column-1 floats; when
/// it no longer fits beside this column's floats it steps below the
/// intruding floats, mirroring the flow pass's no-room policy. (True
/// re-breaking of its lines is future work; non-BFC children keep the
/// flow behavior of letting floats intrude into their content.)
///
/// Column 0 keeps the flow pass's geometry exactly: the flow pass
/// already cleared and fit against these same floats.
///
/// Returns the adjusted `ny`. Bounded: each (e) step strictly lowers the
/// band past an intruding float's bottom, so each step removes at least
/// one float from the set extending below `ny` — at most
/// `floats.len()` steps.
#[allow(clippy::too_many_arguments)]
fn column_ny_fixups(
    forest: &Forest,
    c: usize,
    child_order: usize,
    col: usize,
    col_floats: &[Vec<ColFloat>],
    mc: &MulticolParams,
    content_x: f32,
    mb_w: f32,
    mut ny: f32,
    h: f32,
) -> f32 {
    if col == 0 {
        return ny;
    }
    let col_x = content_x + col as f32 * (mc.col_w + mc.gap);
    let floats = &col_floats[col];
    // (d) Re-evaluate `clear` against this column's floats.
    let clear = forest.nodes[c].data.clear;
    if clear != Clear::None {
        let mut bottom = ny;
        for (f, order) in floats.iter() {
            // CSS 2.1 §9.5.2: clearance only accounts for preceding
            // floats. A float later in document order than the child
            // never pushes it down (the flow pass never saw it either).
            if *order >= child_order {
                continue;
            }
            let relevant = match clear {
                Clear::Left => f.left,
                Clear::Right => !f.left,
                Clear::Both => true,
                Clear::None => false,
            };
            if relevant {
                bottom = bottom.max(f.y + f.h);
            }
        }
        ny = ny.max(bottom);
    }
    // (e) Step below the column's floats while the used width does not
    // fit beside them. Non-BFC children keep full width with intruding
    // floats (their lines wrap); only BFC children narrow/clear.
    if super::establishes_bfc(forest, c) {
        for _ in 0..floats.len() {
            let (_, avail) = column_float_insets(floats, col_x, mc.col_w, ny, ny + h);
            if avail > 0.0 && mb_w <= avail + EPS {
                break;
            }
            let mut bottom = ny;
            for (f, _) in floats.iter() {
                if f.y < ny + h && f.y + f.h > ny {
                    bottom = bottom.max(f.y + f.h);
                }
            }
            if bottom <= ny + EPS {
                break;
            }
            ny = bottom;
        }
    }
    ny
}

/// Place a block child (or split fragment) into column `col`: run the
/// (d)/(e) float fix-ups on `ny`, then record the deferred shift with
/// the column's float intrusion. Returns the final `ny` for the
/// caller's `new_bottom`/`max_bottom` accounting.
#[allow(clippy::too_many_arguments)]
fn place_child_col(
    forest: &mut Forest,
    c: usize,
    child_order: usize,
    col: usize,
    mc: &MulticolParams,
    content_x: f32,
    ny: f32,
    orig_x: f32,
    orig_y: f32,
    ml: f32,
    mb_w: f32,
    h: f32,
    col_floats: &[Vec<ColFloat>],
    ctx_floats: &[FloatEntry],
) -> f32 {
    let ny = column_ny_fixups(
        forest,
        c,
        child_order,
        col,
        col_floats,
        mc,
        content_x,
        mb_w,
        ny,
        h,
    );
    place_child(
        forest,
        c,
        col,
        mc,
        content_x,
        ny,
        orig_x,
        orig_y,
        ml,
        h,
        &col_floats[col],
        ctx_floats,
    );
    ny
}

/// Assign the container's direct float children to columns by their
/// in-flow (pre-balance) positions, and record the deferred shifts
/// moving them there.
///
/// Each float keeps the within-column placement the flow pass computed
/// against one column width (`place_float`): its x becomes the column
/// offset plus that within-column offset — i.e. it floats against its
/// column's width. Its y keeps the within-band offset, so it stays next
/// to the content it was laid out beside. A float that does not fit in
/// the column's remaining height whole moves to the next column's top
/// (stacked below floats already assigned there); past the last column
/// it overflows, mirroring content past `N` columns.
///
/// Returns the per-column float lists in the placed frame (for the
/// (c)/(d)/(e) fix-ups) and, when `float_ctx` is `Some`, updates the
/// float context entries to the placed geometry so BFC containment and
/// later `place_float` band queries see final positions.
///
/// `seg_orig` is the segment's original-frame `(top, bottom)`; floats
/// whose in-flow margin-box top falls outside it belong to another
/// segment. `row_top_placed` is the row's placed top. Only direct float
/// children are assigned here: nested floats ride their ancestors'
/// subtree shifts, and ancestor floats are not this container's to move.
/// `doc_order` maps each child to its document order (continuations
/// inherit their original's); each assigned float records its order so
/// the (d) `clear` fix-up only clears preceding floats (CSS 2.1 §9.5.2).
#[allow(clippy::too_many_arguments)]
fn assign_column_floats(
    forest: &mut Forest,
    idx: usize,
    mc: &MulticolParams,
    content_x: f32,
    seg_orig: (f32, f32),
    row_top_placed: f32,
    col_h: f32,
    float_ctx: &mut FloatCtx,
    doc_order: &HashMap<usize, usize>,
) -> Vec<Vec<ColFloat>> {
    let n = mc.count;
    let mut col_floats: Vec<Vec<ColFloat>> = vec![Vec::new(); n];
    // Max placed margin-box bottom per column: pushed floats stack below
    // floats already assigned to the next column.
    let mut cursor = vec![row_top_placed; n];
    // Direct float children in document order whose in-flow position
    // falls in this segment. Out-of-flow children are excluded even
    // though `is_float` is true for `position: absolute` + `float` (the
    // engine keeps `data.float` set; the convention is that layout
    // checks out-of-flow first, as the `kids` filter below does): an
    // abspos float is placed by the abspos post-pass, and giving it
    // column placement here would both plant a phantom `ColFloat` and
    // stack the column shift under the abspos `pend_x += fx - px`
    // accumulation.
    let seg_floats: Vec<usize> = forest.nodes[idx]
        .children
        .iter()
        .copied()
        .filter(|&c| super::is_float(forest, c) && !is_out_of_flow(forest, c))
        .filter(|&c| {
            let t = forest.nodes[c].rect.y - forest.nodes[c].metrics.mt;
            t >= seg_orig.0 - EPS && t < seg_orig.1 + EPS
        })
        .collect();
    // Whether any float context entry was moved (drives the reindex).
    let mut ctx_dirty = false;
    // Entry ids already matched this run (duplicate geometries match in
    // order).
    let mut matched: Vec<bool> = Vec::new();
    for &f in &seg_floats {
        let (rect_x, rect_y, ml, mt, mr, mb, w, h, pend_x, pend_y, is_left) = {
            let nd = &forest.nodes[f];
            let m = &nd.metrics;
            (
                nd.rect.x,
                nd.rect.y,
                m.ml,
                m.mt,
                m.mr,
                m.mb,
                nd.rect.w,
                nd.rect.h,
                nd.pend_x,
                nd.pend_y,
                matches!(nd.data.float, Float::Left),
            )
        };
        // In-flow margin-box top, relative to the segment's original top.
        let t_rel = (rect_y - mt - seg_orig.0).max(0.0);
        // Band (column) by pre-balance position; a zero target puts
        // everything in column 1.
        let orig_k = if col_h > EPS {
            ((t_rel / col_h).floor() as usize).min(n - 1)
        } else {
            0
        };
        let mh = mt + h + mb;
        let mw = ml + w + mr;
        // (b) Whole-float push: when the float does not fit in the
        // column's remaining height, it moves whole to the next column
        // (past the last column it overflows, like other content).
        let in_band = t_rel - orig_k as f32 * col_h;
        let mut k = orig_k;
        if col_h > EPS && in_band + mh > col_h + EPS && k + 1 < n {
            k += 1;
        }
        let pushed = k != orig_k;
        let margin_top = if pushed {
            let my = cursor[k].max(row_top_placed);
            cursor[k] = my + mh;
            my
        } else {
            let my = row_top_placed + in_band;
            cursor[k] = cursor[k].max(my + mh);
            my
        };
        let margin_x = content_x + k as f32 * (mc.col_w + mc.gap) + (rect_x - content_x);
        let border_y = margin_top + mt;
        // Deferred shifts, set-not-accumulate like `place_child`.
        forest.nodes[f].pend_x = margin_x + ml - rect_x;
        forest.nodes[f].pend_y = border_y - rect_y;
        // The float's document order travels with the entry for the (d)
        // `clear` fix-up (CSS 2.1 §9.5.2: only preceding floats clear).
        // `seg_floats` already iterates in document order, but the order
        // is read from the map so continuations inserted by earlier
        // segments' splits cannot shift it.
        let order = doc_order.get(&f).copied().unwrap_or(usize::MAX);
        col_floats[k].push((
            FloatEntry {
                x: margin_x,
                y: margin_top,
                w: mw,
                h: mh,
                left: is_left,
                node: f,
            },
            order,
        ));
        // Move the float context entry to the placed geometry. The entry
        // is matched by node identity first: each float owns exactly one
        // context entry, so identity is unambiguous. Geometry-only
        // matching can misattribute when a nested float's entry shares
        // this float's margin-box geometry and side and sits earlier in
        // the vec: the scan would claim (and corrupt) the nested entry
        // while this float's own entry stays stale at in-flow geometry —
        // one corrupted entry plus one phantom float. The geometry arms
        // below are the fallback: on the first run the entry sits at the
        // in-flow box; on a re-layout re-run the entry already carries
        // the previous run's placed box, which is the in-flow box plus
        // the still-pending shift recorded then.
        if matched.len() != float_ctx.entries.len() {
            matched = vec![false; float_ctx.entries.len()];
        }
        let (mx, my) = (rect_x - ml, rect_y - mt);
        let mut id = None;
        for (i, e) in float_ctx.entries.iter().enumerate() {
            if matched[i] || e.node != f {
                continue;
            }
            id = Some(i);
            break;
        }
        if id.is_none() {
            for (i, e) in float_ctx.entries.iter().enumerate() {
                if matched[i] || e.left != is_left {
                    continue;
                }
                let inflow = (e.x - mx).abs() < EPS
                    && (e.y - my).abs() < EPS
                    && (e.w - mw).abs() < EPS
                    && (e.h - mh).abs() < EPS;
                // A re-layout re-run matches the previous run's placed box
                // (in-flow box plus the still-pending shift); like the inflow
                // arm it checks the full margin-box geometry, so a resized
                // float cannot match a stale entry.
                let rerun = (e.x - (mx + pend_x)).abs() < EPS
                    && (e.y - (my + pend_y)).abs() < EPS
                    && (e.w - mw).abs() < EPS
                    && (e.h - mh).abs() < EPS;
                if inflow || rerun {
                    id = Some(i);
                    break;
                }
            }
        }
        if let Some(i) = id {
            matched[i] = true;
            let e = &mut float_ctx.entries[i];
            e.x = margin_x;
            e.y = margin_top;
            ctx_dirty = true;
        }
    }
    if ctx_dirty {
        // Later `place_float` band queries must see the placed geometry:
        // rebuild the interval index over the moved entries.
        float_ctx.reindex();
    }
    col_floats
}

/// Move the float context entries of floats nested inside moved block
/// children to their placed geometry.
///
/// `assign_column_floats` only updates direct float children (matched by
/// margin-box geometry); a float nested inside a split or whole-moved
/// block child rides its ancestors' deferred (`pend_x`/`pend_y`) subtree
/// shifts, so its entry keeps the stale in-flow geometry. The stale entry
/// then poisons BFC containment (container measures too short) and lets
/// phantom floats narrow lines in the wrong column (non-BFC case).
///
/// The walk accumulates ancestor pend deltas down the container's
/// subtree and, for each float node with a non-zero effective shift,
/// rewrites its entry (matched by node id) from the node's provisional
/// margin box plus the shift, then reindexes once. Rewriting from the
/// node geometry (rather than adding the delta onto the entry) is
/// idempotent across balance re-runs over the same forest.
///
/// Direct float children of the container are skipped: their entries
/// were already moved to the placed geometry by `assign_column_floats`,
/// and touching them again would double-shift them (their nested floats
/// live in the float's own fresh context anyway). Out-of-flow subtrees
/// are skipped: they never split, so their floats carry no column shift.
fn sync_nested_float_entries(forest: &Forest, idx: usize, float_ctx: &mut FloatCtx) {
    if float_ctx.entries.is_empty() {
        return;
    }
    // Node id -> entry index, so the subtree walk matches floats in O(1).
    let mut id_of: FxHashMap<usize, usize> =
        FxHashMap::with_capacity_and_hasher(float_ctx.entries.len(), FxBuildHasher);
    for (i, e) in float_ctx.entries.iter().enumerate() {
        id_of.insert(e.node, i);
    }
    let mut dirty = false;
    // (node, accumulated ancestor pend dx, dy).
    let mut stack: Vec<(usize, f32, f32)> = Vec::new();
    for &c in &forest.nodes[idx].children {
        stack.push((c, 0.0, 0.0));
    }
    while let Some((i, ax, ay)) = stack.pop() {
        if is_out_of_flow(forest, i) {
            continue;
        }
        let nd = &forest.nodes[i];
        let (px, py, ml, mt, parent, rect_x, rect_y) = (
            nd.pend_x,
            nd.pend_y,
            nd.metrics.ml,
            nd.metrics.mt,
            nd.parent,
            nd.rect.x,
            nd.rect.y,
        );
        if super::is_float(forest, i) {
            // Direct float children were already placed by
            // `assign_column_floats`; only nested floats need the
            // ancestor-shift fix-up.
            if parent != Some(idx)
                && let Some(&ei) = id_of.get(&i)
            {
                let (sx, sy) = (ax + px, ay + py);
                // The zero-shift gate stays deliberately. A zero effective
                // shift means no ancestor of this float moved in this pass,
                // so its entry — written at float-placement time from the
                // same provisional frame the node geometry is read from —
                // is already correct and there is nothing to heal.
                // Relaxing the gate to rewrite on any geometry disagreement
                // would corrupt a real case: floats belonging to a *nested*
                // multicol that shares this context carry entries at the
                // inner container's column geometry, which legitimately
                // disagrees with their provisional node boxes. With zero
                // outer shift those entries must be left alone; the inner
                // container's own assign/sync owns them. (With nonzero
                // shift the rewrite is still correct: the accumulated
                // shift includes the float's own inner column shift, so
                // node box + total shift lands on the true placed
                // geometry.)
                if sx != 0.0 || sy != 0.0 {
                    let e = &mut float_ctx.entries[ei];
                    // Provisional margin box plus the effective shift.
                    e.x = rect_x - ml + sx;
                    e.y = rect_y - mt + sy;
                    dirty = true;
                }
            }
            // A float's subtree floats live in the float's own fresh
            // context, never this one: no need to descend.
            continue;
        }
        let (nx, ny) = (ax + px, ay + py);
        // No `children.clone()`: the borrow of `nd` ends before this
        // re-borrow, so pushing child ids allocates nothing per node.
        for &c in &forest.nodes[i].children {
            stack.push((c, nx, ny));
        }
    }
    if dirty {
        float_ctx.reindex();
    }
}

// Line boxes (`(top, bottom)`, in the child's original frame) of a block
// child's flow content, rebuilt from its text fragments — one fragment
// per laid-out line per text box, like the page splitter's
// `collect_line_boxes`: fragments whose y-extents overlap belong to the
// same line. Out-of-flow and float subtrees are skipped (they never
// split). The comparison happens in one frame: rects are
// absolute-provisional (pre-`apply_pending_shifts`), exactly like the
// split geometry (`split_y`, `orig_y`), and the balance's rigid shifts
// preserve relative positions — so deferred shifts are deliberately
// *not* added here. Mixing them in compared final-frame fragment
// positions against provisional-frame break positions.
#[cfg(test)]
thread_local! {
    /// Counts `collect_lines` invocations: the perf regression test
    /// asserts one line-map per splittable child, not one per split.
    static COLLECT_LINES_CALLS: RefCell<usize> = const { RefCell::new(0) };
    /// Counts `clone_node` invocations: the path-copy perf test asserts
    /// per-split clone work is O(depth), not O(subtree).
    static CLONE_NODE_CALLS: RefCell<usize> = const { RefCell::new(0) };
}

fn collect_lines(forest: &Forest, root: usize) -> Vec<(f32, f32)> {
    #[cfg(test)]
    COLLECT_LINES_CALLS.with(|c| *c.borrow_mut() += 1);
    let mut frags: Vec<(f32, f32)> = Vec::new();
    let mut stack = vec![root];
    while let Some(o) = stack.pop() {
        let nd = &forest.nodes[o];
        if o != root {
            if is_out_of_flow(forest, o) || super::is_float(forest, o) {
                continue;
            }
            // Only descend into boxes the band filter would recurse into:
            // atomic boxes (inline-blocks, tables, `break-inside: avoid`,
            // nested multicol) ride whole and contribute no splittable
            // lines, so their lines must not vote on the split point.
            let splittable = matches!(
                nd.box_type,
                BoxType::Block | BoxType::AnonymousBlock | BoxType::Inline
            ) && nd.multicol.is_none()
                && super::flex_kind(forest, o).is_none()
                && !break_inside_avoid(nd.style.as_deref());
            if !splittable && nd.box_type != BoxType::Text {
                continue;
            }
        }
        if nd.box_type == BoxType::Text {
            for f in &nd.fragments {
                let t = f.rect.y;
                let b = t + f.rect.h;
                if b > t {
                    frags.push((t, b));
                }
            }
            continue;
        }
        for &k in nd.children.iter() {
            stack.push(k);
        }
    }
    frags.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    merge_lines(&frags)
}

/// Merge top-sorted text fragments into line boxes: fragments whose
/// y-extents overlap (`t < last.1 - EPS`) extend the current line, others
/// start a new one. Line bottoms come out exactly non-decreasing — a new
/// line's bottom is clamped up to the previous line's bottom when a
/// sub-EPS-tall fragment would otherwise end below it (`b < last.1`).
/// That clamp is what makes the `partition_point` in `try_split_child`
/// valid: its predicate `l.1 <= split_y + EPS` is only a true partition
/// (all-true then all-false) over a non-decreasing bottoms sequence, and
/// an adjacent [false, true] inversion lets the binary search return an
/// unspecified index — a different k, a different orphans/widows
/// verdict, and a wrongly snapped break.
///
/// The merge/push decisions are unchanged by the clamp, so the line
/// tops and the line count match the unnormalized merge exactly; each
/// bottom moves up by less than EPS (at a push, `b > t >= last.1 - EPS`
/// for the *previous* bottom, and the merge arm only ever raises via
/// `max`), i.e. inside the band filter's own tolerance. The clamp also
/// preserves the old linear scan's k: with raw bottoms `b_i` and
/// normalized `b'_i = max(b_i, b'_{i-1})`, the first index with
/// `b_i > split_y + EPS` equals the first with `b'_i > split_y + EPS`
/// (at the normalized first-false index `j`, all earlier normalized
/// bottoms are `<= split_y + EPS`, hence so are the earlier raw
/// bottoms, and `b'_j > split_y + EPS` forces `b_j > split_y + EPS`
/// since `b'_j` is a max over `b_0..=b_j`).
fn merge_lines(sorted_frags: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut lines: Vec<(f32, f32)> = Vec::with_capacity(sorted_frags.len());
    for &(t, b) in sorted_frags {
        // Fragments arrive sorted by top; `t < last.1 - EPS` means the
        // fragment overlaps the previous line vertically and extends it.
        let merge = lines.last().is_some_and(|last| t < last.1 - EPS);
        if merge {
            let last = lines.last_mut().expect("merge implies a last line");
            last.1 = last.1.max(b);
            continue;
        }
        // Otherwise this fragment starts a new line — but a fragment
        // shorter than EPS can end *below* the previous line's bottom,
        // which would make the bottoms sequence non-monotonic at this
        // seam. Clamp up (see the doc comment above for why).
        let b = lines.last().map_or(b, |last| b.max(last.1));
        lines.push((t, b));
    }
    // The `partition_point` in `try_split_child` is only valid on a true
    // partition: this pins the non-decreasing-bottoms invariant the
    // binary search relies on.
    debug_assert!(
        lines.windows(2).all(|w| w[0].1 <= w[1].1),
        "merge_lines: line bottoms must be exactly non-decreasing"
    );
    lines
}

/// Clone one arena node field-for-field. The arena's [`Node`] has no
/// `Clone` impl by design (boxes are built once), so the split
/// continuation builds its copy explicitly. Child/parent links are left
/// for the caller to remap; outside list-item markers stay with the
/// original (first) fragment.
fn clone_node(forest: &Forest, o: usize) -> crate::boxes::Node {
    #[cfg(test)]
    CLONE_NODE_CALLS.with(|c| *c.borrow_mut() += 1);
    let n = &forest.nodes[o];
    crate::boxes::Node {
        box_type: n.box_type,
        style: n.style.clone(),
        data: n.data.clone(),
        text: n.text.clone(),
        children: Vec::new(),
        parent: None,
        metrics: n.metrics.clone(),
        rect: n.rect,
        pend_x: n.pend_x,
        pend_y: n.pend_y,
        content_h: n.content_h,
        height_definite: n.height_definite,
        mt_place: n.mt_place,
        mb_place: n.mb_place,
        mt_place_done: n.mt_place_done,
        top_collapsed: n.top_collapsed,
        baseline: n.baseline,
        pref_w: n.pref_w,
        dom: n.dom,
        intrinsic: n.intrinsic,
        fragments: n.fragments.clone(),
        colspan: n.colspan,
        rowspan: n.rowspan,
        abspos_recorded: n.abspos_recorded,
        collapsed_borders: n.collapsed_borders.clone(),
        marker: None,
        multicol: n.multicol.clone(),
        section_kind: n.section_kind,
        phantom_w: n.phantom_w,
        phantom_h: n.phantom_h,
        from_table_element: n.from_table_element,
        is_br: n.is_br,
        natural_content_h: n.natural_content_h,
    }
}

/// Build the below-band continuation of a split child by *path copying*:
/// only the straddling ancestor chain is cloned, so one split costs
/// O(depth), not O(subtree).
///
/// `c` straddles `split_y` (in `c`'s original frame); returns the
/// continuation root, parented to `container` but not yet linked into
/// its `children` (the caller inserts it via [`insert_child_after_at`]).
/// The continuation root is a shallow shell clone of `c` ([`clone_node`]
/// copies deferred shifts, fragments, and `abspos_recorded` exactly
/// like the old deep clone, so OOF drop rules and the deferred-shift
/// convention carry over), intersected with the below band, then
/// populated top-down: each child is classified with [`band_decision`]
/// for the below band — the same predicate [`band_filter`] uses, so the
/// kept node sets are identical to the old clone-then-filter:
///
/// - `Drop`: stays in `c`'s tree; the caller's above-band `band_filter`
///   keeps or drops it exactly as before (out-of-flow boxes land here,
///   so they stay in the first fragment only).
/// - `Keep`: the whole subtree is *moved* (reparented) under the
///   continuation — no clone. `intersect_band` on the moved root is a
///   proven no-op for strictly-below boxes and reproduces the old
///   clone's sub-EPS top snap otherwise. The move is recorded in the
///   returned [`SplitRecord`] so the stale strip can re-insert the
///   subtree into the original at its exact pre-split position before a
///   re-layout re-partitions (see [`MULTICOL_CONTINUATIONS`]).
/// - `Recurse`: the straddling shell is cloned and intersected with the
///   band while the original stays for the above band; the pair is
///   revisited for the shell's children — this is the copied path.
/// - `FilterText`: straddling text is cloned keeping only its below-band
///   line fragments, mirroring `band_filter`'s text arm (floats ride
///   whole via `Keep`, never split).
///
/// The caller then runs `band_filter(forest, c, split_y, true)` on the
/// remainder. Band decisions depend only on geometry, not on tree
/// shape, so running it after the detach visits the same nodes with the
/// same verdicts — the first fragment ends up exactly as before, minus
/// the children that moved below.
///
/// Returns the continuation root plus the split's [`SplitRecord`]
/// (destructive footprint for the stale strip).
fn split_off_below(
    forest: &mut Forest,
    c: usize,
    container: usize,
    split_y: f32,
) -> (usize, SplitRecord) {
    let mut record = SplitRecord {
        continuation: 0,
        original: c,
        moves: Vec::new(),
        shells: Vec::new(),
    };
    let cont = {
        let b = clone_node(forest, c);
        let ni = forest.nodes.len();
        forest.nodes.push(b);
        ni
    };
    record.continuation = cont;
    forest.nodes[cont].parent = Some(container);
    intersect_band(forest, cont, split_y, false);
    // (original, continuation-shell) pairs for straddling boxes; the
    // original stays in `c`'s tree for the above-band filter.
    let mut stack = vec![(c, cont)];
    while let Some((o, co)) = stack.pop() {
        // Pre-split height: the re-layout flow recomputes this from the
        // damaged subtree, so the stale strip restores it for an exact
        // re-split classification.
        record.shells.push((o, forest.nodes[o].rect.h));
        let kids = std::mem::take(&mut forest.nodes[o].children);
        let mut kept = Vec::with_capacity(kids.len());
        let mut ckids = Vec::new();
        for (j, k) in kids.iter().enumerate() {
            let k = *k;
            let below = {
                let nd = &forest.nodes[k];
                let fy0 = nd.rect.y;
                band_decision(
                    nd.box_type,
                    is_out_of_flow(forest, k),
                    super::is_float(forest, k),
                    nd.multicol.is_some(),
                    super::flex_kind(forest, k).is_some(),
                    nd.style.as_deref(),
                    fy0,
                    fy0 + nd.rect.h,
                    split_y,
                    false,
                )
            };
            match below {
                BandDec::Drop => kept.push(k),
                BandDec::FilterText => {
                    let mut b = clone_node(forest, k);
                    let mut kept_any = false;
                    b.fragments.retain(|f| {
                        let keep = f.rect.y >= split_y - EPS;
                        kept_any |= keep;
                        keep
                    });
                    if kept_any {
                        let ck = forest.nodes.len();
                        b.parent = Some(co);
                        forest.nodes.push(b);
                        intersect_band(forest, ck, split_y, false);
                        ckids.push(ck);
                    }
                    kept.push(k);
                }
                BandDec::Recurse => {
                    let mut b = clone_node(forest, k);
                    b.parent = Some(co);
                    let ck = forest.nodes.len();
                    forest.nodes.push(b);
                    intersect_band(forest, ck, split_y, false);
                    ckids.push(ck);
                    kept.push(k);
                    stack.push((k, ck));
                }
                BandDec::Keep => {
                    // Whole subtree rides below: reparent, no clone.
                    // `band_decision`'s branch order makes `Keep` for
                    // both `keep_above` values unreachable (a
                    // within-EPS degenerate box hits `fy1 <= split_y +
                    // EPS` → below-band `Drop`), so there is no
                    // both-fragments case to clone for.
                    let rel_y = forest.nodes[k].rect.y - forest.nodes[o].rect.y;
                    forest.nodes[k].parent = Some(co);
                    // Floats ride whole: a straddling float moved below
                    // keeps its full height (page-splitter parity), never
                    // sliced at the band edge.
                    if !super::is_float(forest, k) {
                        intersect_band(forest, k, split_y, false);
                    }
                    ckids.push(k);
                    // Record the move for the stale strip: a re-layout
                    // re-inserts the subtree at this index and shifts it
                    // back by `rel_y`, restoring the pre-split geometry
                    // (see [`MULTICOL_CONTINUATIONS`]).
                    record.moves.push(SplitMove {
                        parent: o,
                        child: k,
                        index: j,
                        rel_y,
                    });
                }
            }
        }
        forest.nodes[o].children = kept;
        forest.nodes[co].children = ckids;
    }
    (cont, record)
}

/// Shift a subtree's absolute y positions by `dy`: border boxes and
/// text-fragment rects/baselines. Deferred shifts (`pend_x`/`pend_y`)
/// are deltas, not positions, and are left alone.
fn shift_subtree_y(forest: &mut Forest, root: usize, dy: f32) {
    if dy == 0.0 {
        return;
    }
    let mut stack = vec![root];
    while let Some(o) = stack.pop() {
        let nd = &mut forest.nodes[o];
        nd.rect.y += dy;
        for f in &mut nd.fragments {
            f.rect.y += dy;
            f.baseline += dy;
        }
        stack.extend(nd.children.iter().copied());
    }
}

/// Undo one balance pass's destructive splits (see [`SplitRecord`]):
/// re-insert every moved whole subtree into its original parent at its
/// recorded index and shift it back to its pre-split relative position,
/// and restore the straddling shells' pre-split heights. Records are
/// processed in reverse split order so a child moved twice (a re-split
/// continuation's move) lands back at its earliest origin. Moves within
/// one record are in kid order per parent, so ascending inserts
/// reconstruct the exact child order.
///
/// After this, the original's child list and geometry match the
/// pre-split flow closely enough for the second balance pass to
/// re-derive the same splits: the flow pass already rebuilt text
/// fragments and recomputed kept rects, and the recorded `rel_y` puts
/// moved subtrees at their exact pre-split offsets.
fn restore_split_records(forest: &mut Forest, records: &[SplitRecord]) {
    for r in records.iter().rev() {
        for m in &r.moves {
            // Detach from wherever it sits now (the stale continuation's
            // subtree); the strip already unlinked the continuation
            // itself from the container.
            if let Some(p) = forest.nodes[m.child].parent {
                forest.nodes[p].children.retain(|&x| x != m.child);
            }
            {
                let children = &mut forest.nodes[m.parent].children;
                let at = m.index.min(children.len());
                children.insert(at, m.child);
            }
            forest.nodes[m.child].parent = Some(m.parent);
            let dy = forest.nodes[m.parent].rect.y + m.rel_y - forest.nodes[m.child].rect.y;
            shift_subtree_y(forest, m.child, dy);
        }
        for &(o, h) in &r.shells {
            forest.nodes[o].rect.h = h;
        }
    }
}

/// Which side of a split band a box belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BandDec {
    Keep,
    Drop,
    Recurse,
    FilterText,
}

/// One node's band verdict for [`band_filter`].
#[allow(clippy::too_many_arguments)]
fn band_decision(
    box_type: BoxType,
    is_oof: bool,
    is_float: bool,
    has_multicol: bool,
    is_flex: bool,
    style: Option<&ComputedStyle>,
    fy0: f32,
    fy1: f32,
    split_y: f32,
    keep_above: bool,
) -> BandDec {
    if box_type == BoxType::Text {
        return BandDec::FilterText;
    }
    if is_oof {
        // Out-of-flow boxes never split; the first fragment keeps them
        // (the abs-pos post-pass lays them out from its own records).
        return if keep_above {
            BandDec::Keep
        } else {
            BandDec::Drop
        };
    }
    if is_float {
        // Floats never split: a straddling float rides whole to the next
        // column instead of being sliced mid-box (page-splitter parity:
        // `layout_fragmentation` moves a float straddling a page boundary
        // whole to the top of the next page). A non-straddling float
        // rides with the fragment holding its top edge.
        if fy0 < split_y - EPS && fy1 > split_y + EPS {
            return if keep_above {
                BandDec::Drop
            } else {
                BandDec::Keep
            };
        }
        let starts_above = fy0 < split_y - EPS;
        return if keep_above == starts_above {
            BandDec::Keep
        } else {
            BandDec::Drop
        };
    }
    if fy1 <= split_y + EPS {
        return if keep_above {
            BandDec::Keep
        } else {
            BandDec::Drop
        };
    }
    if fy0 >= split_y - EPS {
        return if keep_above {
            BandDec::Drop
        } else {
            BandDec::Keep
        };
    }
    // Straddling: recurse into splittable boxes, otherwise ride whole
    // with the fragment holding the top.
    let splittable = matches!(
        box_type,
        BoxType::Block | BoxType::AnonymousBlock | BoxType::Inline
    ) && !has_multicol
        && !is_flex
        && !break_inside_avoid(style);
    if splittable {
        BandDec::Recurse
    } else if keep_above {
        BandDec::Keep
    } else {
        BandDec::Drop
    }
}

/// Intersect a kept box's border box with the fragment band. Both the box
/// and the band are absolute-provisional (see [`collect_lines`]), so no
/// ancestor-shift mapping is needed.
fn intersect_band(forest: &mut Forest, o: usize, split_y: f32, keep_above: bool) {
    let nd = &mut forest.nodes[o];
    if keep_above {
        nd.rect.h = (nd.rect.h.min(split_y - nd.rect.y)).max(0.0);
    } else {
        let new_y = split_y.max(nd.rect.y);
        nd.rect.h = (nd.rect.y + nd.rect.h - new_y).max(0.0);
        nd.rect.y = new_y;
    }
}

/// Restrict the subtree at `root` to one side of `split_y` (in the
/// child's provisional frame: every rect compared is absolute and
/// pre-shift, matching the split geometry — see [`collect_lines`]):
/// `keep_above` keeps `(-∞, split_y]`, otherwise `[split_y, +∞)`. Band
/// slicing mirrors the page splitter (`split_subtree`): text nodes keep
/// only their in-band line fragments; straddling plain block,
/// anonymous-block, and inline boxes recurse; everything unbreakable
/// rides whole with the fragment holding its top. Kept boxes are
/// intersected with the band.
fn band_filter(forest: &mut Forest, root: usize, split_y: f32, keep_above: bool) {
    let mut dec: FxHashMap<usize, BandDec> = FxHashMap::default();
    let mut order: Vec<usize> = Vec::new();
    let mut stack = vec![root];
    // Pass 1 (pre-order): decide per node.
    while let Some(o) = stack.pop() {
        order.push(o);
        let nd = &forest.nodes[o];
        let fy0 = nd.rect.y;
        let d = band_decision(
            nd.box_type,
            is_out_of_flow(forest, o),
            super::is_float(forest, o),
            nd.multicol.is_some(),
            super::flex_kind(forest, o).is_some(),
            nd.style.as_deref(),
            fy0,
            fy0 + nd.rect.h,
            split_y,
            keep_above,
        );
        dec.insert(o, d);
        if d == BandDec::Recurse {
            for &k in nd.children.iter() {
                stack.push(k);
            }
        }
    }
    // Pass 2 (bottom-up: reversed pre-order visits children before
    // parents): prune children, filter text fragments, intersect kept
    // boxes with the band.
    for &o in order.iter().rev() {
        match dec[&o] {
            BandDec::Drop => {}
            BandDec::Keep => {
                // Floats ride whole: never slice them at the band edge.
                // A kept float is fully inside its band already (a
                // straddling float moves whole to the next fragment), so
                // skipping is a no-op except at EPS-edge degenerates.
                if !super::is_float(forest, o) {
                    intersect_band(forest, o, split_y, keep_above);
                }
            }
            BandDec::Recurse => {
                let kept: Vec<usize> = forest.nodes[o]
                    .children
                    .iter()
                    .copied()
                    .filter(|k| matches!(dec.get(k), Some(d) if *d != BandDec::Drop))
                    .collect();
                forest.nodes[o].children = kept;
                intersect_band(forest, o, split_y, keep_above);
            }
            BandDec::FilterText => {
                let mut kept_any = false;
                {
                    let nd = &mut forest.nodes[o];
                    nd.fragments.retain(|f| {
                        let t = f.rect.y;
                        let keep = if keep_above {
                            t + f.rect.h <= split_y + EPS
                        } else {
                            t >= split_y - EPS
                        };
                        kept_any |= keep;
                        keep
                    });
                }
                if kept_any {
                    intersect_band(forest, o, split_y, keep_above);
                } else {
                    dec.insert(o, BandDec::Drop);
                }
            }
        }
    }
}

/// Insert `new` into the container's children at index `pos + 1`,
/// preserving paint/document order for a split continuation. The caller
/// tracks the live index (see `balance_block_columns`), so there is no
/// O(n) `position` scan; the `Vec::insert` memmove of the tail remains —
/// O(n) per split, unavoidable with a `Vec` child list (a gap buffer or
/// linked order would trade this for allocation or pointer chasing the
/// hot layout paths don't want).
fn insert_child_after_at(forest: &mut Forest, container: usize, pos: usize, new: usize) {
    let children = &mut forest.nodes[container].children;
    // `pos` is always a live child index — tracked by the caller as
    // `child_pos[c] + continuations.len()` for original children or
    // `last_insert_pos` for re-split continuations — so `pos + 1` is a
    // valid insert position with no clamp: `pos < children.len()`
    // implies `pos + 1 <= children.len()`.
    debug_assert!(
        pos < children.len(),
        "insert position must be a live child index, got {pos} with {} children",
        children.len()
    );
    children.insert(pos + 1, new);
}

/// One line-map per split candidate: node -> (shared ordered line list,
/// offset of the first line still present in that fragment). The list is
/// shared via `Rc` across a child's continuation chain (see
/// `try_split_child`).
type LineMapCache = FxHashMap<usize, (Rc<[(f32, f32)]>, usize)>;

/// Try to split block child `c` of multicol container `container` at
/// `split_y` (in `c`'s original frame): lines above the break stay in `c`,
/// the rest move to a continuation node inserted right after `c`.
/// Returns the continuation's index and the *snapped* break position, or
/// `None` when `c` cannot split at a line boundary (no lines,
/// `break-inside: avoid`, or the orphans/widows constraints cannot be met).
///
/// `line_cache` maps a node to its line map: the full ordered line list
/// (shared via `Rc` across the node's continuation chain) plus the offset
/// of the first line still present in that fragment. The map is computed
/// once per child and inherited as a suffix by each continuation — band
/// filtering keeps every line below the snapped break intact at its
/// original y, so the continuation's lines are exactly `lines[k..]`.
/// This replaces a per-split `collect_lines` (full subtree walk + sort).
fn try_split_child(
    forest: &mut Forest,
    container: usize,
    c: usize,
    split_y: f32,
    line_cache: &mut LineMapCache,
) -> Option<(usize, f32, SplitRecord)> {
    // Only plain block boxes split. Nested multicol containers are atomic
    // (they distribute inner-first), flex containers keep their items
    // whole, and `break-inside: avoid` opts out explicitly.
    {
        let nd = &forest.nodes[c];
        if !matches!(nd.box_type, BoxType::Block | BoxType::AnonymousBlock) || nd.multicol.is_some()
        {
            return None;
        }
    }
    if super::flex_kind(forest, c).is_some() {
        return None;
    }
    let style = forest.nodes[c].style.clone();
    if break_inside_avoid(style.as_deref()) {
        return None;
    }
    // One line-map per child (see above); a failed split leaves the
    // subtree untouched, so a later retry with a new `split_y` reuses it.
    let (shared, start) = line_cache
        .entry(c)
        .or_insert_with(|| {
            let lines = collect_lines(forest, c);
            (Rc::from(lines.into_boxed_slice()), 0)
        })
        .clone();
    let lines = &shared[start..];
    // The break keeps at least `orphans` lines above and `widows` below
    // it (CSS Fragmentation 3 §5).
    let (orphans, widows) = orphans_widows_of(style.as_deref());
    let n_lines = lines.len();
    // `lines` is sorted by top and merged so bottoms are exactly
    // non-decreasing (`merge_lines` clamps each new line's bottom up to
    // the previous line's, so a sub-EPS fragment cannot invert the
    // sequence). The predicate `l.1 <= split_y + EPS` is therefore a true
    // partition (all-true then all-false), and `partition_point` finds
    // exactly the first-false index a linear scan would — the binary
    // search is only valid because the normalization proof in
    // `merge_lines` holds.
    let k = lines.partition_point(|l| l.1 <= split_y + EPS);
    if k < orphans as usize || n_lines - k < widows as usize {
        return None;
    }
    // Snap the break to the bottom of the last fully-above line: a raw
    // `split_y` can straddle a line, and a straddling line satisfies
    // neither band predicate, so it would be dropped from *both*
    // fragments. `k >= 1` here (orphans >= 1), and merged lines never
    // overlap, so no line straddles the snapped position. The snapped
    // position is returned so the caller places fragments off the same
    // geometry the band filter used.
    let split_y = lines[k - 1].1;
    // Path-copy the below-band continuation, then band-filter the
    // original to the above band: the original keeps content above
    // `split_y`, the continuation keeps content at or below it.
    let (cont, record) = split_off_below(forest, c, container, split_y);
    band_filter(forest, c, split_y, true);
    // Fragments of one box exchange no margins.
    forest.nodes[c].mb_place = 0.0;
    forest.nodes[cont].mt_place = 0.0;
    // The continuation inherits the line map as the suffix below the
    // break (see the doc comment above).
    line_cache.insert(cont, (shared, start + k));
    Some((cont, split_y, record))
}

/// Balance target for one segment: the definite height when filling
/// sequentially, else a balanced share of the segment's stacked height —
/// never less than the tallest *unsplittable* child (an unbreakable child
/// always fits a fresh column) and, under a definite height, never more
/// than the height itself. Splittable children fragment instead of
/// forcing the target up.
fn segment_col_h(
    forest: &Forest,
    seg: &[usize],
    row_orig_top: f32,
    n: usize,
    hspec: Option<f32>,
    fill_auto: bool,
) -> f32 {
    let mut seg_total = 0.0f32;
    let mut seg_tallest_rigid = 0.0f32;
    for &c in seg {
        let nd = &forest.nodes[c];
        seg_total = seg_total.max(nd.rect.y + nd.rect.h - row_orig_top);
        if !maybe_splittable(forest, c) {
            seg_tallest_rigid = seg_tallest_rigid.max(nd.rect.h);
        }
    }
    match (hspec, fill_auto) {
        (Some(h), true) => h.max(0.0),
        (Some(h), false) => (seg_total / n as f32)
            .max(seg_tallest_rigid)
            .min(h)
            .max(0.0),
        (None, _) => (seg_total / n as f32).max(seg_tallest_rigid).max(0.0),
    }
}

/// Distribute a multicol container's in-flow block children across its
/// columns. Children were laid out stacked in one column of width
/// `col_w`; each keeps its width and moves to `(column_x, column_y)` via
/// deferred shifts. Forced `break-before`/`break-after: column` start a
/// fresh column; children that overflow a column split at a line boundary
/// (with a continuation in the next column); `column-span: all` children
/// take full-width spanner rows between independently balanced segments.
/// Returns the new natural content height (tallest column bottom minus
/// `content_top`). Out-of-flow children are left where the flow pass put
/// them; direct float children are assigned to columns by their in-flow
/// positions (see [`assign_column_floats`]), and `float_ctx`'s entries
/// are updated to the placed geometry.
pub(super) fn balance_block_columns(
    forest: &mut Forest,
    idx: usize,
    mc: &MulticolParams,
    content_x: f32,
    content_top: f32,
    hspec: Option<f32>,
    float_ctx: &mut FloatCtx,
) -> f32 {
    // Pop this container's pending `column-rule` (pushed by
    // `multicol_params`). This happens on every path — including the
    // empty-children early return below — so the push/pop pairing stays
    // exact (see `PENDING_RULES`).
    let rule = pop_pending_rule(mc);
    // Live positions of the container's children, captured once. The
    // stale strip below reuses the map for O(1) adjacency checks (no
    // `children.iter().position(...)` per recorded continuation), and the
    // split loop reuses it to insert each continuation at a tracked index
    // instead of scanning for it (see `insert_child_after_at`). Every
    // insert lands right after the child being processed, i.e. before
    // every not-yet-processed child, so an original child's live index is
    // its captured index plus the number of inserts so far.
    let mut child_pos: HashMap<usize, usize> = forest.nodes[idx]
        .children
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();
    // Strip split continuations left by a previous balance pass over
    // this container (see `MULTICOL_CONTINUATIONS`): a re-layout over the
    // same forest re-runs the balance with the stale clones still in
    // `children`, and re-splitting them corrupts the fragments. `stale`
    // is a set (the `cont_roots` trick) so the strip below is O(1) per
    // child instead of O(children × splits). Accepted records also
    // restore the splits' destructive footprint — moved whole subtrees
    // go back into the original at their exact pre-split positions —
    // before partitioning, so the second pass re-splits the full
    // content, not the above-band remainder the flow pass left behind.
    let accepted: Vec<SplitRecord> = MULTICOL_CONTINUATIONS
        .with(|m| m.borrow_mut().remove(&idx).unwrap_or_default())
        .into_iter()
        .filter(|r| {
            let (ci, oi) = (r.continuation, r.original);
            let adjacent = child_pos
                .get(&ci)
                .is_some_and(|&pos| pos > 0 && forest.nodes[idx].children[pos - 1] == oi);
            adjacent
                && forest.nodes.get(ci).is_some_and(|n| n.parent == Some(idx))
                && forest.nodes.get(oi).is_some_and(|n| n.parent == Some(idx))
                && forest.nodes[ci].dom == forest.nodes[oi].dom
        })
        .collect();
    let stale: HashSet<usize> = accepted.iter().map(|r| r.continuation).collect();
    if !stale.is_empty() {
        forest.nodes[idx].children.retain(|c| !stale.contains(c));
        // The strip shifted every index after a removed child: re-capture
        // the positions so the split loop's tracked inserts stay exact.
        child_pos = forest.nodes[idx]
            .children
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        restore_split_records(forest, &accepted);
    }
    // Document order of each direct child, for the (d) `clear` fix-up
    // (CSS 2.1 §9.5.2: only preceding floats clear). Captured after the
    // strip, so it holds exactly the original children in document
    // order; split continuations created below inherit their original's
    // order (see the split site). `child_pos` keeps tracking live
    // indices separately — the two maps serve different fix-ups.
    let mut doc_order: HashMap<usize, usize> = child_pos.clone();
    let kids: Vec<usize> = forest.nodes[idx]
        .children
        .iter()
        .copied()
        .filter(|&c| !is_out_of_flow(forest, c) && !super::is_float(forest, c))
        .collect();
    if kids.is_empty() {
        return 0.0;
    }
    let n = mc.count;
    // Partition the children into column-row segments separated by
    // `column-span: all` spanners (task 4): each segment balances
    // independently, each spanner takes a full-width row between them.
    // Without spanners this is one segment and the loop below behaves
    // like the old whole-child distributor.
    let mut segments: Vec<(Vec<usize>, Option<usize>)> = vec![(Vec::new(), None)];
    for &c in &kids {
        if is_spanner(forest, c) {
            let last = segments.last_mut().expect("segments never empty");
            last.1 = Some(c);
            segments.push((Vec::new(), None));
        } else {
            segments.last_mut().expect("segments never empty").0.push(c);
        }
    }
    // `column-fill: auto` fills sequentially, but only under a definite
    // height; with an auto height it behaves as `balance` (CSS Multicol 1).
    let fill_auto = mc.fill_auto && hspec.is_some();

    // Per-segment original-frame `(top, bottom)`: the first segment starts
    // at the content top; each later segment starts below the previous
    // spanner and ends at the next spanner's top (the last runs to
    // infinity). Float assignment needs these up front to find each
    // float's segment from its in-flow position.
    let mut seg_orig: Vec<(f32, f32)> = Vec::with_capacity(segments.len());
    {
        let mut top = content_top;
        for (_, spanner) in &segments {
            let bottom = spanner.map_or(f32::INFINITY, |sp| forest.nodes[sp].rect.y);
            seg_orig.push((top, bottom));
            if let Some(sp) = spanner {
                let nd = &forest.nodes[*sp];
                top = nd.rect.y + nd.rect.h;
            }
        }
    }
    // Per-segment balance targets, precomputed: float assignment (which
    // runs before each segment's child loop so the per-column float
    // lists are ready for the (c)/(d)/(e) fix-ups) needs `col_h` up
    // front.
    let seg_col_h: Vec<f32> = segments
        .iter()
        .enumerate()
        .map(|(i, (seg, _))| segment_col_h(forest, seg, seg_orig[i].0, n, hspec, fill_auto))
        .collect();

    let mut max_bottom = content_top;
    // Current column-row geometry: placed and original tops. The first
    // row starts at the container's content top in both frames.
    let mut row_top_placed = content_top;
    let mut row_orig_top = content_top;
    let mut first_row = true;
    // `break-after: column` state (task 1): a plain pending flag for
    // unsplit children, plus a deferred flag for split boxes whose
    // break-after must fire after their last continuation.
    let mut force_col = false;
    let mut deferred_after = false;
    // Continuation records from splits: [`SplitRecord`]s carrying the
    // continuation root, the original, and the destructive footprint.
    // Continuations never re-take the box's break-before (already
    // honored by the first fragment); the records are also stored in
    // `MULTICOL_CONTINUATIONS` so a re-layout strips the stale clones
    // and restores the moved subtrees before re-partitioning.
    // `cont_roots` mirrors the continuation roots as a set for O(1)
    // membership tests in the child loop (a linear scan per child was
    // O(children × splits)).
    let mut continuations: Vec<SplitRecord> = Vec::new();
    let mut cont_roots: HashSet<usize> = HashSet::new();
    // Forest position of the most recently inserted continuation (see
    // the split site below).
    let mut last_insert_pos = 0usize;
    // Per-child line maps for split attempts (see `try_split_child`):
    // computed once per child, inherited as suffixes by continuations.
    let mut line_cache: LineMapCache = FxHashMap::default();

    for (seg_i, (seg, spanner)) in segments.iter().enumerate() {
        let col_h = seg_col_h[seg_i];
        // Multicol float model: assign this segment's direct floats to
        // columns by their in-flow positions before placing children, so
        // the per-column float lists are ready for the (c)/(d)/(e)
        // fix-ups in the child loop below.
        let col_floats = assign_column_floats(
            forest,
            idx,
            mc,
            content_x,
            seg_orig[seg_i],
            row_top_placed,
            col_h,
            &mut *float_ctx,
            &doc_order,
        );
        // Float context entries that are not this container's direct
        // float children (those travel in `col_floats` at placed
        // geometry): ancestor floats, which never move, plus nested
        // floats at flow geometry. `place_child` uses them for the
        // column-0 phantom-indent shed.
        let direct_floats: HashSet<usize> =
            col_floats.iter().flatten().map(|(e, _)| e.node).collect();
        let ctx_floats: Vec<FloatEntry> = float_ctx
            .entries
            .iter()
            .filter(|e| !direct_floats.contains(&e.node))
            .copied()
            .collect();

        let mut col = 0usize;
        let mut new_bottom = vec![row_top_placed; n];
        let mut orig_bottom = vec![row_orig_top; n];
        // The segment's children, grown as splits append continuations.
        let children: Vec<usize> = seg.clone();
        let mut i = 0usize;
        // Split continuations are processed immediately next, before the
        // following sibling. Instead of `children.insert(i + 1, cont)`
        // (an O(n) memmove per split), the continuation waits in this
        // slot and the loop head takes it first: identical visit order
        // at O(1).
        let mut pending: Option<usize> = None;
        while i < children.len() || pending.is_some() {
            let c = match pending.take() {
                Some(p) => p,
                None => {
                    let c = children[i];
                    i += 1;
                    c
                }
            };
            let child_style = forest.nodes[c].style.clone();
            let (orig_x, orig_y, w, h, ml, mr) = {
                let nd = &forest.nodes[c];
                (
                    nd.rect.x,
                    nd.rect.y,
                    nd.rect.w,
                    nd.rect.h,
                    nd.metrics.ml,
                    nd.metrics.mr,
                )
            };
            // Used margin-box width: the (e) fit fix-up tests it against
            // the column's available width beside floats.
            let mb_w = ml + w + mr;
            let is_cont = cont_roots.contains(&c);
            // Document order of the child under placement, for the (d)
            // `clear` fix-up (CSS 2.1 §9.5.2: only preceding floats
            // clear). Continuations inherit their original's order
            // (recorded at the split site below); an unknown child falls
            // back to `usize::MAX`, i.e. the old consider-all-floats
            // behavior.
            let child_order = doc_order.get(&c).copied().unwrap_or(usize::MAX);
            // Forced column breaks (task 1).
            let mut forced = force_col;
            force_col = false;
            if !is_cont {
                if deferred_after {
                    forced = true;
                    deferred_after = false;
                }
                forced |= forced_col_break(child_style.as_deref(), props::BREAK_BEFORE);
            }
            let fresh = new_bottom[col] <= row_top_placed + EPS;
            let mut ny;
            if forced && !fresh && col + 1 < n {
                col += 1;
                ny = place_y(
                    col,
                    true,
                    orig_y,
                    &new_bottom,
                    &orig_bottom,
                    row_top_placed,
                    row_orig_top,
                    first_row,
                );
            } else if forced && !fresh {
                // Past the last column: start a fresh overflow row below
                // the column block.
                ny = new_bottom[col].max(row_top_placed + col_h);
                ny = place_child_col(
                    forest,
                    c,
                    child_order,
                    col,
                    mc,
                    content_x,
                    ny,
                    orig_x,
                    orig_y,
                    ml,
                    mb_w,
                    h,
                    &col_floats,
                    &ctx_floats,
                );
                new_bottom[col] = ny + h;
                orig_bottom[col] = orig_y + h;
                max_bottom = max_bottom.max(ny + h);
                if !is_cont {
                    force_col = forced_col_break(child_style.as_deref(), props::BREAK_AFTER);
                }
                continue;
            } else {
                ny = place_y(
                    col,
                    fresh,
                    orig_y,
                    &new_bottom,
                    &orig_bottom,
                    row_top_placed,
                    row_orig_top,
                    first_row,
                );
            }
            // (d)/(e) float fix-ups before the fit/split test, so a child
            // pushed below its column's floats re-tests fit against the
            // column target.
            ny = column_ny_fixups(
                forest,
                c,
                child_order,
                col,
                &col_floats,
                mc,
                content_x,
                mb_w,
                ny,
                h,
            );
            // Fit, split, or move whole: re-test after advancing into a
            // fresh column so a tall child can still split to fill it.
            loop {
                if ny + h - row_top_placed > col_h + EPS {
                    let split_y = orig_y + (row_top_placed + col_h - ny);
                    if let Some((cont, split_y, record)) =
                        try_split_child(forest, idx, c, split_y, &mut line_cache)
                    {
                        // Fragment 1 keeps this column; the continuation
                        // is processed next, in the following column — or
                        // stacked below when already in the last one.
                        // `split_y` is the snapped break position (see
                        // `try_split_child`).
                        pending = Some(cont);
                        // Live index of `c` in the container's children:
                        // a continuation was inserted by the immediately
                        // preceding split and is processed next via the
                        // `pending` slot, so no insert has happened since
                        // and it still sits at `last_insert_pos`; an
                        // original child's captured index has shifted by
                        // one per insert so far (`continuations.len()`).
                        let pos = if is_cont {
                            last_insert_pos
                        } else {
                            child_pos
                                .get(&c)
                                .copied()
                                .expect("split child is a container child")
                                + continuations.len()
                        };
                        insert_child_after_at(forest, idx, pos, cont);
                        last_insert_pos = pos + 1;
                        // `record` already carries this split's
                        // `(continuation, original)` identity.
                        continuations.push(record);
                        cont_roots.insert(cont);
                        // The continuation inherits its original's
                        // document order (transitively: splitting a
                        // continuation keeps the same order), so the (d)
                        // `clear` fix-up treats it like the original.
                        if let Some(&o) = doc_order.get(&c) {
                            doc_order.insert(cont, o);
                        }
                        let frag_h = (split_y - orig_y).max(0.0);
                        let ny = place_child_col(
                            forest,
                            c,
                            child_order,
                            col,
                            mc,
                            content_x,
                            ny,
                            orig_x,
                            orig_y,
                            ml,
                            mb_w,
                            frag_h,
                            &col_floats,
                            &ctx_floats,
                        );
                        new_bottom[col] = ny + frag_h;
                        orig_bottom[col] = split_y;
                        max_bottom = max_bottom.max(ny + frag_h);
                        if col + 1 < n {
                            col += 1;
                        }
                        if !is_cont {
                            deferred_after |=
                                forced_col_break(child_style.as_deref(), props::BREAK_AFTER);
                        }
                        break;
                    }
                    if new_bottom[col] > row_top_placed + EPS && col + 1 < n {
                        col += 1;
                        ny = place_y(
                            col,
                            true,
                            orig_y,
                            &new_bottom,
                            &orig_bottom,
                            row_top_placed,
                            row_orig_top,
                            first_row,
                        );
                        // A fresh column has a fresh float list: re-run
                        // the (d)/(e) fix-ups for the new column.
                        ny = column_ny_fixups(
                            forest,
                            c,
                            child_order,
                            col,
                            &col_floats,
                            mc,
                            content_x,
                            mb_w,
                            ny,
                            h,
                        );
                        continue;
                    }
                }
                // Fits, or overflows the last column whole (as before).
                let ny = place_child_col(
                    forest,
                    c,
                    child_order,
                    col,
                    mc,
                    content_x,
                    ny,
                    orig_x,
                    orig_y,
                    ml,
                    mb_w,
                    h,
                    &col_floats,
                    &ctx_floats,
                );
                new_bottom[col] = ny + h;
                orig_bottom[col] = orig_y + h;
                max_bottom = max_bottom.max(ny + h);
                if !is_cont {
                    force_col = forced_col_break(child_style.as_deref(), props::BREAK_AFTER);
                }
                break;
            }
        }

        // A `column-span: all` child takes a full-width row below the
        // tallest column, preserving its flow gap to the segment above;
        // the next segment balances in a fresh row below the spanner.
        if let Some(sp) = spanner {
            let (orig_x, orig_y, h, ml, mr) = {
                let nd = &forest.nodes[*sp];
                (
                    nd.rect.x,
                    nd.rect.y,
                    nd.rect.h,
                    nd.metrics.ml,
                    nd.metrics.mr,
                )
            };
            let seg_placed_end = new_bottom.iter().fold(row_top_placed, |a, &b| a.max(b));
            let seg_orig_end = orig_bottom.iter().fold(row_orig_top, |a, &b| a.max(b));
            let ny = seg_placed_end + (orig_y - seg_orig_end);
            // Full content width. The flow pass laid the spanner out
            // against the full content width already (see the
            // `layout_block` prologue), so this only widens the border
            // box to match, shedding any flow-pass float intrusion like
            // any non-column-0 placement.
            let full_w = spanner_row_width(mc);
            {
                let nd = &mut forest.nodes[*sp];
                // Assigned, not accumulated: see `place_child`.
                nd.pend_x = content_x + ml - orig_x;
                nd.pend_y = ny - orig_y;
                nd.rect.w = (full_w - ml - mr).max(0.0);
            }
            max_bottom = max_bottom.max(ny + h);
            row_top_placed = ny + h;
            row_orig_top = orig_y + h;
            first_row = false;
            // A pending `break-after: column` survives the spanner row:
            // it fires on the first child of the next segment. (Resetting
            // it here silently dropped the forced break.)
        }
    }
    // Record this pass's continuations so a re-layout over the same
    // forest strips the stale clones and restores the moved subtrees
    // before partitioning (see `MULTICOL_CONTINUATIONS`).
    if !continuations.is_empty() {
        MULTICOL_CONTINUATIONS.with(|m| {
            m.borrow_mut().insert(idx, continuations);
        });
    }
    // A float nested inside a split or whole-moved block child rides its
    // ancestors' deferred shifts; `assign_column_floats` only moved
    // direct floats, so sync the nested entries to placed geometry before
    // the height below (and later band queries) read them.
    sync_nested_float_entries(forest, idx, float_ctx);
    // Re-base direct abspos children's static positions onto the column
    // their in-flow placeholder lands in (see
    // `adjust_abspos_static_positions`).
    adjust_abspos_static_positions(forest, idx, mc, content_x);
    let height = (max_bottom - content_top).max(0.0);
    // Record the `column-rule` geometry for the paint worker (see
    // `ColumnRuleSpec`).
    if let Some(rule) = rule {
        record_column_rule(ColumnRuleSpec {
            container: Some(idx),
            dom_node: forest.nodes[idx].dom,
            x_offsets: rule_x_offsets(mc),
            height,
            width: rule.width,
            style: rule.style,
            color: rule.color,
        });
    }
    height
}

/// Re-base the recorded static positions of the container's direct
/// absolutely-positioned children onto the column their in-flow
/// placeholder lands in after balancing.
///
/// Each out-of-flow child is recorded in [`Forest::abs_items`] during the
/// single-column flow pass with the static position flow would have given
/// it — i.e. in column-1 geometry. The abspos post-pass
/// (`layout_position::layout_abs_items`) resolves `auto` offsets against
/// that static position, so without this fix an abspos box whose
/// placeholder balanced into column N would be placed from its column-1
/// static position.
///
/// For each direct out-of-flow child, the anchor is the nearest preceding
/// sibling that advances the block cursor (in-flow and non-float: floats
/// and out-of-flow boxes take no space in flow, so they never determine
/// the placeholder). The scan runs over the post-split child order, so a
/// split continuation anchors what follows it. The placeholder's
/// fragmented position is the anchor's placed position plus the flow-pass
/// offset between the static point and the anchor's origin — equivalently
/// the static point shifted by `(k * (col_w + gap), anchor.pend_y)`,
/// where `k` is the anchor's column. The x shift is pure column geometry
/// (the anchor's own float-intrusion delta must not leak into the abspos
/// x); the y shift is the anchor's full deferred shift, which is exact
/// because deferred shifts are rigid subtree moves.
///
/// Only direct children are adjusted. An abspos box nested inside a moved
/// in-flow child rides that child's deferred shift via
/// `apply_pending_shifts`, which already preserves its within-frame
/// offset; adjusting it again would double-count the column shift.
///
/// Idempotency: balance can run twice over one forest (flex
/// measure→real, grid second pass) before the abspos post-pass consumes
/// the items, so a previously applied delta is reverted before the fresh
/// one is applied. The stored pre-adjustment position authenticates the
/// [`ABS_STATIC_DELTAS`] record: a fresh forest reusing arena indices
/// fails the check and is treated as unadjusted.
fn adjust_abspos_static_positions(
    forest: &mut Forest,
    idx: usize,
    mc: &MulticolParams,
    content_x: f32,
) {
    // node -> `abs_items` index, for direct out-of-flow children of `idx`.
    let item_of: HashMap<usize, usize> = forest
        .abs_items
        .iter()
        .enumerate()
        .filter(|(_, a)| forest.nodes[a.node].parent == Some(idx))
        .map(|(i, a)| (a.node, i))
        .collect();
    if item_of.is_empty() {
        return;
    }
    let stride = mc.col_w + mc.gap;
    let max_col = mc.count.saturating_sub(1) as i32;
    // Nearest preceding sibling that advances the block cursor, as the
    // `(dx, dy)` shift the placeholder takes into the anchor's column.
    let mut anchor: Option<(f32, f32)> = None;
    let children = forest.nodes[idx].children.clone();
    for &c in &children {
        if is_out_of_flow(forest, c) {
            let Some(&item_i) = item_of.get(&c) else {
                continue;
            };
            let Some((dx, dy)) = anchor else {
                // No in-flow content precedes: the placeholder sits at
                // the container's content top in column 1, where the
                // recorded static position already is.
                continue;
            };
            // Revert a delta applied by an earlier balance run over this
            // forest (see `ABS_STATIC_DELTAS`). The stored pre-adjustment
            // position authenticates the record: a fresh forest reusing
            // arena indices fails the check and is treated as unadjusted.
            let (base, reverted) = ABS_STATIC_DELTAS.with(|m| {
                let mut m = m.borrow_mut();
                match m.remove(&c) {
                    Some((px, py, odx, ody))
                        if forest.abs_items[item_i].static_x == px + odx
                            && forest.abs_items[item_i].static_y == py + ody =>
                    {
                        ((px, py), true)
                    }
                    _ => {
                        let a = &forest.abs_items[item_i];
                        ((a.static_x, a.static_y), false)
                    }
                }
            });
            if !reverted && dx == 0.0 && dy == 0.0 {
                continue;
            }
            let item = &mut forest.abs_items[item_i];
            item.static_x = base.0 + dx;
            item.static_y = base.1 + dy;
            if dx == 0.0 && dy == 0.0 {
                // Reverted to the recorded position; nothing to re-record.
                continue;
            }
            ABS_STATIC_DELTAS.with(|m| {
                m.borrow_mut().insert(c, (base.0, base.1, dx, dy));
            });
        } else if !super::is_float(forest, c) {
            // In-flow block-level sibling (spanners included): its
            // deferred shift maps the flow-pass frame onto its column.
            let n = &forest.nodes[c];
            let placed_x = n.rect.x + n.pend_x;
            // Column index from the placed x: the anchor's border box
            // sits at `content_x + k * stride + ml + dx_new` with the
            // float-intrusion `dx_new` strictly below one column width
            // (a BFC child that does not fit beside the column's floats
            // steps below them instead; the shift site additionally
            // clamps `dx_new` to at most `col_w - 2*EPS`, keeping f32
            // dust clear of the rounding boundary), so backing one
            // column width off and rounding up recovers `k` exactly.
            let rel = placed_x - n.metrics.ml - content_x;
            let k = ((rel - mc.col_w + EPS) / stride).ceil() as i32;
            let k = k.clamp(0, max_col) as f32;
            anchor = Some((k * stride, n.pend_y));
        }
    }
}

/// Effective advance of one placed fragment for line re-breaking: the
/// distance to the next fragment's origin on its line, or its own width
/// for the last fragment. Inline-box spacing (`frag_lead`/`frag_trail`)
/// is baked into the placed x positions by `do_inline_layout`, so
/// deriving the advance from them preserves it through a re-break
/// without access to the spacing tables. Justified word gaps are baked
/// in the same way (stale justification is kept, not recomputed).
fn frag_pitch(frags: &[Frag], j: usize, end: usize) -> f32 {
    if j + 1 < end {
        (frags[j + 1].x - frags[j].x).max(0.0)
    } else {
        frags[j].w.max(0.0)
    }
}

/// Laid width of the placed fragments in `[start, end)`.
fn range_width(frags: &[Frag], start: usize, end: usize) -> f32 {
    let mut w = 0.0;
    for j in start..end {
        w += frag_pitch(frags, j, end);
    }
    w
}

/// Vertical band of the placed fragments in `[start, end)`; `None` when
/// no fragment is placed.
fn frag_band(frags: &[Frag], start: usize, end: usize) -> Option<(f32, f32)> {
    let mut top = f32::INFINITY;
    let mut bottom = f32::NEG_INFINITY;
    for f in &frags[start..end] {
        if !f.placed {
            continue;
        }
        top = top.min(f.abs_top);
        bottom = bottom.max(f.abs_top + (f.rb - f.rt).max(0.0));
    }
    if top.is_finite() {
        Some((top, (bottom - top).max(0.0)))
    } else {
        None
    }
}

/// Whether a line's fragments can be re-broken: only the wrappable
/// white-spaces are understood. Pre-formatted runs (`pre`,
/// `pre-wrap`/`pre-line`/`break-spaces` pieces arrive pre-fit) keep
/// their flow-pass shape and are only shifted.
fn rebreakable_line(frags: &[Frag], start: usize, end: usize) -> bool {
    frags[start..end]
        .iter()
        .all(|f| matches!(f.ws, WhiteSpace::Normal | WhiteSpace::NoWrap))
}

/// Re-break the fragments in `[start, end)` against `avail`, mirroring
/// `do_inline_layout`'s wrap rules at fragment granularity: leading
/// trimmable spaces are skipped (unplaced) like the flow pass, a
/// fragment with `break_before` starts a new line, and a wrappable
/// (`WhiteSpace::Normal`) fragment that would overflow starts a new
/// line; the first fragment always stays, overflowing like the flow
/// pass. Returns the output lines as `(placed_start, placed_end,
/// next_start)`: trailing trimmable spaces are excluded from
/// `[placed_start, placed_end)` and the next line begins at `next_start`,
/// skipping them exactly like the flow pass. The caller unplaces
/// `[placed_end, next_start)` and repositions fragments.
fn wrap_frags(
    frags: &mut [Frag],
    start: usize,
    end: usize,
    avail: f32,
) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let mut s = start;
    while s < end {
        // Leading trimmable spaces are skipped at each line start, like
        // the flow pass (they are unplaced here; a later pull-up can
        // still find them as the separator via their width).
        while s < end && frags[s].is_space && !frags[s].break_before && super::ws_trims(frags[s].ws)
        {
            frags[s].placed = false;
            s += 1;
        }
        if s >= end {
            break;
        }
        let mut w = 0.0;
        let mut e_trimmed = s;
        let mut j = s;
        while j < end {
            let fw = frag_pitch(frags, j, end);
            if frags[j].break_before && j > s {
                break;
            }
            if super::ws_wraps(frags[j].ws) && j > s && w + fw > avail + EPS {
                break;
            }
            w += fw;
            j += 1;
            if !super::ws_trims(frags[j - 1].ws) || !frags[j - 1].is_space {
                e_trimmed = j;
            }
        }
        // The `j > s` guards consume at least one fragment per pass, so
        // this terminates. An all-space run emits an empty placed range;
        // the caller unplaces it and the line is skipped downstream.
        out.push((s, e_trimmed, j));
        s = j;
    }
    out
}

/// Working line table entry for line re-breaking in
/// [`distribute_inline_lines_for_container`]: `lines` is immutable, and
/// the passes after distribution only read fragments, so re-breaking
/// rewrites fragment ranges here instead of in `lines`. `start` advances
/// as following lines pull head frags up; `orig` is the index into
/// `lines` (`None` for lines created by re-breaking); `stack` places the
/// line directly below the previous one instead of via `place_y`.
/// `avail` is the flow pass's available width for the line: pull-up
/// only re-expands a line when the new column offers more than the flow
/// pass broke against.
struct WorkLine {
    start: usize,
    end: usize,
    dx: f32,
    avail: f32,
    orig: Option<usize>,
    stack: bool,
    /// Placement shift applied to this line's fragments by the
    /// distribution pass. Recorded so the 2-pass re-balance (see
    /// [`distribute_inline_lines_for_container`]) can undo the first
    /// pass's shifts and re-distribute from the re-broken flow frame.
    shift_x: f32,
    shift_y: f32,
    /// Flow-frame x of the line's first fragment: the line's legitimate
    /// content start, including any `text-indent` or alignment offset
    /// (but excluding the column placement shift). The stolen-head
    /// re-anchor in [`pull_up_frags`] aligns the remaining fragments to
    /// this recorded anchor instead of inferring the line start as
    /// `content_x + dx`, which would silently destroy indent/alignment
    /// offsets.
    anchor_x: f32,
}

/// Install a [`wrap_frags`] result into the working line table: the first
/// output line keeps `work[pos]`'s slot, the rest are inserted right
/// after it as stacked lines (`stack: true`), each reset to the
/// flow-frame line origin while preserving intra-line relative
/// positions. Trimmed trailing spaces (`[placed_end, next_start)`) are
/// unplaced. Returns whether the wrap produced more than one line.
fn install_wrap(
    frags: &mut [Frag],
    work: &mut Vec<WorkLine>,
    pos: usize,
    content_x: f32,
    outs: &[(usize, usize, usize)],
) -> bool {
    if outs.len() <= 1 {
        return false;
    }
    let pdx = work[pos].dx;
    let x0 = content_x + pdx;
    let (fs, fe, fnext) = outs[0];
    for f in frags.iter_mut().take(fnext).skip(fe) {
        f.placed = false;
    }
    work[pos].start = fs;
    work[pos].end = fe;
    // One splice (not one insert per line): the memmove happens once.
    let conts: Vec<WorkLine> = outs[1..]
        .iter()
        .map(|&(os, oe, onext)| {
            for f in frags.iter_mut().take(onext).skip(oe) {
                f.placed = false;
            }
            let fx = frags[os].x;
            for f in &mut frags[os..oe] {
                f.x = x0 + (f.x - fx);
            }
            // A wrapped continuation counts as broken at its laid width,
            // so a later band offering more room can pull words up onto
            // it.
            let ia = range_width(frags, os, oe);
            WorkLine {
                start: os,
                end: oe,
                dx: pdx,
                avail: ia,
                orig: None,
                stack: true,
                shift_x: 0.0,
                shift_y: 0.0,
                // Continuations were just reset to the flow-frame line
                // origin `x0` (preserving intra-line relative
                // positions), so `x0` is their legitimate content
                // start for the stolen-head re-anchor.
                anchor_x: x0,
            }
        })
        .collect();
    work.splice(pos + 1..pos + 1, conts);
    true
}

/// Pull the head fragments of `work[pos + 1]` up onto the line at
/// `work[pos]` while they fit in `avail`, in the flow frame (the caller
/// shifts afterwards). The trimmed space in the gap between the lines
/// is restored as the separator; a pull never crosses a hard break
/// (`break_before`). Trailing trimmable spaces of the pulled run are
/// unplaced and skipped, like the flow pass. When the next line is
/// exhausted it is removed from the table; the freed slot height is
/// returned (O(1)) so the caller can close the hole lazily via its
/// running shift instead of scanning every later line's fragments here
/// (O(F) per absorption). Returns `(removed, slot_h)`: whether the next
/// line was removed (the line count changed — the caller re-balances
/// then) and the freed slot height (0 when nothing was absorbed).
fn pull_up_frags(
    frags: &mut [Frag],
    work: &mut Vec<WorkLine>,
    pos: usize,
    avail: f32,
) -> (bool, f32) {
    let next = pos + 1;
    if next >= work.len() {
        return (false, 0.0);
    }
    let (ns, ne) = (work[next].start, work[next].end);
    if ns >= ne || frags[ns].break_before {
        return (false, 0.0);
    }
    // Slot height the next line occupies in the flow frame (compact: no
    // holes — every absorbed line closes its own below). Needed to close
    // the hole if the line is absorbed.
    let next_h = frag_band(frags, ns, ne).map(|(_, h)| h).unwrap_or(0.0);
    let pe = work[pos].end;
    // The separator is the trimmed space in the gap between the lines
    // (the flow pass trimmed it at the break); without one the words
    // abut (zero-width break opportunity).
    let mut sep: Option<(usize, f32)> = None;
    for (k, f) in frags.iter().enumerate().take(ns).skip(pe) {
        if f.is_space && f.w > 0.0 {
            sep = Some((k, f.w));
            break;
        }
    }
    let (top_next, _) = frag_band(frags, ns, ne).expect("pull-up source is non-empty");
    let (top_cur, _) = frag_band(frags, work[pos].start, pe).expect("pull-up target is non-empty");
    let mut w = range_width(frags, work[pos].start, pe);
    let last = pe - 1;
    let mut cursor = frags[last].x + frag_pitch(frags, last, pe);
    let mut take_end = pe;
    let mut trim_end = pe;
    let mut budget_ok = true;
    if let Some((si, sw)) = sep {
        if w + sw <= avail + EPS {
            // Restore the separator at the target line's top (like the
            // pulled words below); its `rt` differs from the words', so
            // it is not derived from the first fragment's baseline.
            frags[si].x = cursor;
            frags[si].abs_top = top_cur;
            frags[si].placed = true;
            w += sw;
            cursor += sw;
            take_end = si + 1;
        } else {
            budget_ok = false;
        }
    }
    if budget_ok {
        let mut j = ns;
        while j < ne {
            if frags[j].break_before {
                break;
            }
            let fw = frag_pitch(frags, j, ne);
            if super::ws_wraps(frags[j].ws) && w + fw > avail + EPS {
                break;
            }
            let rel = frags[j].abs_top - top_next;
            frags[j].x = cursor;
            frags[j].abs_top = top_cur + rel;
            w += fw;
            cursor += fw;
            take_end = j + 1;
            if !super::ws_trims(frags[j].ws) || !frags[j].is_space {
                trim_end = j + 1;
            }
            j += 1;
        }
    }
    if take_end > pe {
        for f in frags.iter_mut().take(take_end).skip(trim_end) {
            f.placed = false;
        }
        work[pos].end = trim_end;
        work[next].start = take_end;
        // A partial pull steals the next line's head: its remaining
        // frags start past the line's legitimate content start (they
        // were positioned by an earlier pull-up), but the line is now
        // independent and must start at its recorded anchor — otherwise
        // a later pull-up budgets from the wrong x and appends words
        // past the column, and the line itself is placed indented. The
        // anchor is the flow-frame x of the line's first fragment
        // (recorded on the `WorkLine`), not `content_x + dx`: the
        // latter assumes every line starts at the flow origin, which
        // is false for `text-indent` first lines or aligned text, and
        // would silently destroy the indent/alignment. The shift is
        // x-only, so bands, widths, and placement shifts are
        // unaffected.
        if work[next].start < work[next].end {
            let origin = work[next].anchor_x;
            let first_x = frags[work[next].start..work[next].end]
                .iter()
                .filter(|f| f.placed)
                .map(|f| f.x)
                .fold(f32::INFINITY, f32::min);
            if first_x.is_finite() && (first_x - origin).abs() > EPS {
                let shift = first_x - origin;
                for f in &mut frags[work[next].start..work[next].end] {
                    f.x -= shift;
                }
            }
        }
        if work[next].start >= work[next].end {
            // The absorbed line's flow slot is now a hole: return its
            // height for the caller to close lazily (a stacked
            // continuation shares its parent's slot, so it leaves no
            // hole). The old code shifted every later line's fragments
            // up here, O(F) per absorption; the scalar keeps it O(1).
            // Without the closure, `place_y` preserves the hole as a
            // visible gap and the column grows by one line per absorbed
            // line.
            let slot_h = if work[next].stack { 0.0 } else { next_h };
            work.remove(next);
            return (true, slot_h);
        }
    }
    (false, 0.0)
}

/// Legacy entry point without a container identity: the recorded spec
/// carries `container: None` / `dom_node: None` and relies on
/// exact-duplicate suppression. Prefer
/// [`distribute_inline_lines_for_container`], which threads the
/// container's identity through for re-layout dedupe.
#[allow(dead_code)] // Legacy entry point; all in-tree callers use the `_for_container` variant.
#[allow(clippy::too_many_arguments)]
pub(super) fn distribute_inline_lines(
    lines: &[Line],
    frags: &mut [Frag],
    mc: &MulticolParams,
    content_x: f32,
    content_top: f32,
    hspec: Option<f32>,
    forced_breaks: &[usize],
    floats: &[FloatEntry],
) -> f32 {
    distribute_inline_lines_for_container(
        None,
        None,
        lines,
        frags,
        mc,
        content_x,
        content_top,
        hspec,
        forced_breaks,
        floats,
    )
}

/// One balance pass over the working line table.
///
/// `work` is in the flow frame on entry (fragments at flow positions,
/// re-broken line ranges from any earlier pass kept); the pass shifts
/// each placed line's fragments into its column and records the shift
/// on the [`WorkLine`] so the caller can undo it. Mid-pass re-breaking
/// (wrap/pull-up against a column's float geometry) can change the
/// line count; the returned flag tells the caller, which then
/// re-derives the balance target from the actual post-rebreak line
/// heights and re-runs the pass (2-pass re-balance).
///
/// Pull-up holes are closed lazily: each absorption records its freed
/// slot height in the returned per-line table (`absorbed_h[i]` is the
/// slot freed by `work[i]`; 0 when it absorbed nothing — at most one
/// absorption per line per pass) instead of shifting later lines, and
/// a running shift keeps `place_y` on the compacted frame. The caller
/// folds the table into the 2-pass undo sweep so the re-run starts
/// from a fully compacted frame.
///
/// Returns `(max_bottom, rebreak_changed, absorbed_h)`.
#[allow(clippy::too_many_arguments)]
fn distribute_lines_pass(
    work: &mut Vec<WorkLine>,
    frags: &mut [Frag],
    mc: &MulticolParams,
    content_x: f32,
    content_top: f32,
    col_h: f32,
    forced_breaks: &[usize],
    col_floats: &[Vec<ColFloat>],
) -> (f32, bool, Vec<f32>) {
    let n = mc.count;
    let mut col = 0usize;
    let mut new_bottom = vec![content_top; n];
    let mut orig_bottom = vec![content_top; n];
    let mut max_bottom = content_top;
    // Set when mid-pass re-breaking (wrap/pull-up against a column's
    // float geometry) changes the line count: the balance target was
    // derived from flow-pass totals, so the caller re-derives it and
    // re-runs the pass (2-pass re-balance).
    let mut rebreak_changed = false;
    // `forced_breaks` is strictly ascending and `work` visits original
    // lines in ascending index order (re-broken lines carry `orig: None`
    // and never match), so a single pointer finds each break in O(1)
    // amortized time.
    let mut fb_i = 0usize;
    let mut pos = 0usize;
    // Lazily closed pull-up holes: absorbing `work[pos + 1]` frees its
    // flow slot, and the old code closed the hole by shifting every
    // later line's fragments up (O(F) per absorption, O(L*F) per pass).
    // `pull_up_frags` now just returns the freed height (O(1)); it
    // accumulates into `pending_hole`, which folds into `cum_hole` at
    // the top of each iteration — the total hole closed above
    // `work[pos]`. The fold point is exact: absorptions only ever
    // remove the line right after the current one, so a hole closed
    // during iteration `pos` sits entirely below `work[pos]` and must
    // not shift it. Band tops read from `frags` are therefore in the
    // raw (stale) flow frame; the compacted frame the eager scan used
    // to maintain physically is `top_raw - cum_hole`.
    let mut cum_hole = 0.0f32;
    let mut pending_hole = 0.0f32;
    // Freed slot heights per work-line index, for the 2-pass undo's
    // hole-closing sweep (see above). An absorber keeps its index: later
    // structural edits only ever touch indices past the current `pos`.
    let mut absorbed_h: Vec<f32> = Vec::new();
    while pos < work.len() {
        cum_hole += pending_hole;
        pending_hole = 0.0;
        let orig = work[pos].orig;
        if let Some(idx) = orig {
            while fb_i < forced_breaks.len() && forced_breaks[fb_i] < idx {
                fb_i += 1;
            }
        }
        let forced =
            orig.is_some_and(|idx| fb_i < forced_breaks.len() && forced_breaks[fb_i] == idx);
        // A line emptied by pull-up contributes nothing; emptied lines
        // are removed eagerly, so this is only a backstop.
        let (top_raw, mut h) = match frag_band(frags, work[pos].start, work[pos].end) {
            Some(b) => b,
            None => {
                pos += 1;
                continue;
            }
        };
        // Compacted flow-frame top: `place_y`'s output `ny` is absolute
        // placed-frame and feeds placed-frame float band tests, so it
        // must be computed from the compacted top, while `pull_up_frags`
        // below keeps working in the raw frame (its math is
        // shift-invariant).
        let mut top = top_raw - cum_hole;
        let stack = work[pos].stack;
        let fresh = new_bottom[col] <= content_top + EPS;
        let mut ny = if stack {
            // A re-broken continuation stacks below its parent line.
            new_bottom[col]
        } else {
            place_y(
                col,
                fresh,
                top,
                &new_bottom,
                &orig_bottom,
                content_top,
                content_top,
                true,
            )
        };
        if forced && !fresh {
            // An inline-level forced break starts a fresh column — or,
            // past the last column, continues below the column block
            // (mirroring the block-children overflow row). Into a fresh
            // column it is a no-op (no empty column is created).
            if col + 1 < n {
                col += 1;
                ny = place_y(
                    col,
                    true,
                    top,
                    &new_bottom,
                    &orig_bottom,
                    content_top,
                    content_top,
                    true,
                );
            } else {
                ny = new_bottom[col].max(content_top + col_h);
            }
        } else if !fresh && ny + h - content_top > col_h + EPS && col + 1 < n {
            col += 1;
            ny = place_y(
                col,
                true,
                top,
                &new_bottom,
                &orig_bottom,
                content_top,
                content_top,
                true,
            );
        }
        // Per-column float lists: recompute the line's float intrusion
        // against the floats overlapping this column. Column 0 keeps the
        // flow pass's exact `dx`; later columns shed intrusion from
        // floats that do not reach them, and re-break around floats that
        // do (wrapping or pulling words; stepping below only with no
        // room).
        let mut col_x = content_x + col as f32 * (mc.col_w + mc.gap);
        let (mut new_dx, mut new_avail) = if col == 0 {
            (work[pos].dx, f32::INFINITY)
        } else {
            column_float_insets(&col_floats[col], col_x, mc.col_w, ny, ny + h)
        };
        let mut width = range_width(frags, work[pos].start, work[pos].end);
        if col > 0 {
            let s = work[pos].start;
            let e = work[pos].end;
            let rebreakable = rebreakable_line(frags, s, e);
            let first_fw = frag_pitch(frags, s, e);
            let first_unbreakable = frags[s].ib.is_some() || !super::ws_wraps(frags[s].ws);
            // The flow pass's no-room policy: no usable space beside the
            // floats at all, or an unbreakable first fragment wider than
            // the narrowed space — the whole line steps below the floats
            // instead of overlapping them.
            let no_room = new_avail <= 0.0
                || (new_avail < mc.col_w - EPS && first_unbreakable && first_fw > new_avail + EPS);
            if width > new_avail + EPS && (!rebreakable || no_room) {
                // (e): the line was broken against a wider width than
                // this column offers beside its floats; step it below
                // the intruding floats (mirroring the flow pass's
                // no-room policy) instead of overlapping them. Bounded by
                // the float count: each step moves `ny` to an intruding
                // float's bottom, strictly shrinking the set of floats
                // extending below `ny`. A stepped line can overflow the
                // column bottom, so column fit is re-checked and the line
                // moves whole to the next column (or the overflow row
                // past the last), mirroring the block path; the fresh
                // column re-runs the step against its own floats.
                loop {
                    for _ in 0..col_floats[col].len() {
                        let mut bottom = ny;
                        for (f, _) in col_floats[col].iter() {
                            if f.y < ny + h && f.y + f.h > ny {
                                bottom = bottom.max(f.y + f.h);
                            }
                        }
                        if bottom <= ny + EPS {
                            break;
                        }
                        ny = bottom;
                        (new_dx, new_avail) =
                            column_float_insets(&col_floats[col], col_x, mc.col_w, ny, ny + h);
                        if width <= new_avail + EPS {
                            break;
                        }
                    }
                    if ny + h - content_top <= col_h + EPS || col + 1 >= n {
                        break;
                    }
                    col += 1;
                    ny = place_y(
                        col,
                        true,
                        top,
                        &new_bottom,
                        &orig_bottom,
                        content_top,
                        content_top,
                        true,
                    );
                    col_x = content_x + col as f32 * (mc.col_w + mc.gap);
                    (new_dx, new_avail) =
                        column_float_insets(&col_floats[col], col_x, mc.col_w, ny, ny + h);
                    if width <= new_avail + EPS {
                        break;
                    }
                }
                if ny + h - content_top > col_h + EPS && col + 1 >= n {
                    // Past the last column: continue below the column
                    // block (mirroring the block-children overflow row).
                    ny = new_bottom[col].max(content_top + col_h);
                    new_dx = column_float_insets(&col_floats[col], col_x, mc.col_w, ny, ny + h).0;
                }
            } else if rebreakable && !no_room {
                // Re-break the line against this column's float geometry
                // (CSS 2.1 §9.5): an over-wide line wraps into several
                // lines fitting the narrowed space, and an under-full
                // line pulls following words up — instead of keeping the
                // flow-pass break shape that another column's floats
                // dictated. Fragments are relocated in the flow frame;
                // the shift below maps them to the placed frame.
                let outs = wrap_frags(frags, s, e, new_avail);
                if install_wrap(frags, work, pos, content_x, &outs) {
                    // Wrapped: continuations were inserted right after
                    // this line; later iterations place them. The line
                    // count changed: the caller re-balances.
                    rebreak_changed = true;
                } else {
                    // The line fits: pull the next line's head fragments
                    // up while they fit, so a line narrowed by another
                    // column's floats re-expands — but only when this
                    // column offers more than the line was broken
                    // against (a flow-pass line with spare room inside
                    // its break width ended there deliberately). Never
                    // across a forced column break (the next line must
                    // start its fresh column).
                    let next = pos + 1;
                    let next_forced = work.get(next).is_some_and(|nw| {
                        nw.orig.is_some_and(|idx| {
                            fb_i < forced_breaks.len() && forced_breaks[fb_i] == idx
                        })
                    });
                    if !next_forced && new_avail > work[pos].avail + EPS {
                        // A removed next line changes the line count:
                        // the caller re-balances. The freed slot feeds
                        // the lazy hole shift (closed above the *next*
                        // line, never this one) and is recorded for the
                        // 2-pass undo's hole-closing sweep.
                        let (removed, slot_h) = pull_up_frags(frags, work, pos, new_avail);
                        pending_hole += slot_h;
                        if removed {
                            if absorbed_h.len() <= pos {
                                absorbed_h.resize(pos + 1, 0.0);
                            }
                            absorbed_h[pos] = slot_h;
                            rebreak_changed = true;
                        }
                    }
                }
                // Refresh the band and width after re-breaking, then
                // re-test the insets: a taller line can newly overlap a
                // float. One bounded re-wrap (no pull-up) covers it.
                // The refreshed band is in the raw flow frame; recover
                // the compacted top (a hole closed by this iteration's
                // pull-up sits below this line, so only the folded
                // `cum_hole` applies).
                (top, h) = frag_band(frags, work[pos].start, work[pos].end)
                    .expect("re-broken line is non-empty");
                top -= cum_hole;
                width = range_width(frags, work[pos].start, work[pos].end);
                (new_dx, new_avail) =
                    column_float_insets(&col_floats[col], col_x, mc.col_w, ny, ny + h);
                if width > new_avail + EPS {
                    let s = work[pos].start;
                    let e = work[pos].end;
                    let outs = wrap_frags(frags, s, e, new_avail);
                    rebreak_changed |= install_wrap(frags, work, pos, content_x, &outs);
                    (top, h) = frag_band(frags, work[pos].start, work[pos].end)
                        .expect("re-broken line is non-empty");
                    // Raw flow frame again: recover the compacted top.
                    top -= cum_hole;
                    (new_dx, _) =
                        column_float_insets(&col_floats[col], col_x, mc.col_w, ny, ny + h);
                }
                // A re-broken line can grow taller (pulled-up fragments):
                // re-check the column fit and move whole to the next
                // column, mirroring the block path.
                if !fresh && ny + h - content_top > col_h + EPS && col + 1 < n {
                    col += 1;
                    ny = place_y(
                        col,
                        true,
                        top,
                        &new_bottom,
                        &orig_bottom,
                        content_top,
                        content_top,
                        true,
                    );
                    col_x = content_x + col as f32 * (mc.col_w + mc.gap);
                    (new_dx, _) =
                        column_float_insets(&col_floats[col], col_x, mc.col_w, ny, ny + h);
                }
            }
        }
        let dx = (col_x - content_x) + (new_dx - work[pos].dx);
        let dy = ny - top;
        // Recorded so the 2-pass re-balance can undo this pass's shifts
        // (see `distribute_inline_lines_for_container`). The fragments
        // still sit at raw flow-frame positions, `cum_hole` above the
        // compacted frame `dy` was measured from, so the shift actually
        // applied (and recorded, so the undo restores the raw frame
        // exactly) is `dy - cum_hole`.
        let shift_y = dy - cum_hole;
        let (ws, we) = (work[pos].start, work[pos].end);
        work[pos].shift_x = dx;
        work[pos].shift_y = shift_y;
        if dx != 0.0 || shift_y != 0.0 {
            for f in &mut frags[ws..we] {
                f.x += dx;
                f.abs_top += shift_y;
            }
        }
        new_bottom[col] = ny + h;
        orig_bottom[col] = top + h;
        max_bottom = max_bottom.max(ny + h);
        pos += 1;
    }
    (max_bottom, rebreak_changed, absorbed_h)
}

/// Distribute a multicol container's line boxes across its columns.
/// Runs inside `do_inline_layout` right after fragment positioning, so
/// inline-block re-anchoring, rect unions, and fragment emission all see
/// the shifted positions.
///
/// Per-column float lists: column 0 keeps the flow pass's float
/// intrusion (`line.dx`) exactly; later columns recompute the intrusion
/// against the floats overlapping that column (from `floats`, the
/// container's float context entries — ancestor floats, since a floated
/// box blockifies and can never be a direct child of inline content).
/// A line moved to a later column is re-broken against that column's
/// float geometry: an over-wide line wraps into several lines fitting
/// the narrowed space (instead of stepping the whole line below the
/// floats), and an under-full line pulls following words up (instead of
/// keeping the narrowed flow-pass break shape). Only the flow pass's
/// no-room case still steps a line below the floats: no usable space at
/// all, or an unbreakable first fragment wider than the narrowed space.
/// Pre-formatted runs keep their flow-pass shape (pure shift).
///
/// `container` / `dom_node` identify the multicol container for
/// [`ColumnRuleSpec`] dedupe (`do_inline_layout` passes `Some(idx)` and
/// the container's DOM node); a re-layout then replaces the container's
/// stale spec instead of accumulating alongside it.
///
/// `forced_breaks` are the strictly ascending line indices (into `lines`,
/// see `InlineLayoutOutput::forced_break_lines`) that must start a fresh
/// column: inline-level forced `break-before` / `break-after: column`.
/// A forced break into a fresh column is a no-op (no empty column is
/// created); past the last column the line continues below the column
/// block, mirroring the block-children overflow row.
///
/// Returns the new natural content height
/// (tallest column bottom minus `content_top`).
#[allow(clippy::too_many_arguments)]
pub(super) fn distribute_inline_lines_for_container(
    container: Option<usize>,
    dom_node: Option<bowser_dom::NodeId>,
    lines: &[Line],
    frags: &mut [Frag],
    mc: &MulticolParams,
    content_x: f32,
    content_top: f32,
    hspec: Option<f32>,
    forced_breaks: &[usize],
    floats: &[FloatEntry],
) -> f32 {
    // Pop this container's pending `column-rule` (pushed by
    // `multicol_params`; see `balance_block_columns` for the pairing).
    let rule = pop_pending_rule(mc);
    // Line extents from placed fragments (unplaced ones were trimmed).
    struct Extent {
        idx: usize,
        top: f32,
        h: f32,
    }
    let mut extents: Vec<Extent> = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        let mut top = f32::INFINITY;
        let mut bottom = f32::NEG_INFINITY;
        for f in &frags[line.start..line.end] {
            if !f.placed {
                continue;
            }
            top = top.min(f.abs_top);
            bottom = bottom.max(f.abs_top + (f.rb - f.rt).max(0.0));
        }
        if top.is_finite() {
            extents.push(Extent {
                idx: i,
                top,
                h: (bottom - top).max(0.0),
            });
        }
    }
    if extents.is_empty() {
        return 0.0;
    }
    let n = mc.count;
    // Per-column float lists: the context floats whose margin boxes
    // horizontally overlap each column (absolute frame, matching the
    // entries). Built once; each line band-tests its column's list.
    // Ancestor floats moved by an ancestor multicol balance are seen
    // pre-shift (their entries are provisional until
    // `apply_pending_shifts`); nested multicol is inner-first, so the
    // relative error only affects floats escaping their own container.
    // These entries are not direct children, so they carry the
    // no-order `usize::MAX` (the inline path never runs the (d) clear
    // fix-up).
    let mut col_floats: Vec<Vec<ColFloat>> = vec![Vec::new(); n];
    for fl in floats {
        for (k, list) in col_floats.iter_mut().enumerate() {
            let cx0 = content_x + k as f32 * (mc.col_w + mc.gap);
            if fl.x < cx0 + mc.col_w + EPS && fl.x + fl.w > cx0 + EPS {
                list.push((*fl, usize::MAX));
            }
        }
    }
    let total = extents
        .iter()
        .map(|e| e.top + e.h - content_top)
        .fold(0.0f32, f32::max);
    let tallest = extents.iter().map(|e| e.h).fold(0.0f32, f32::max);
    // `column-fill` (task 4): `auto` fills sequentially under a definite
    // height; `balance` (the initial, and the used value for auto heights)
    // spreads lines evenly, capped at a definite height.
    let col_h = match (hspec, mc.fill_auto) {
        (Some(h), true) => h.max(0.0),
        (Some(h), false) => (total / n as f32).max(tallest).min(h).max(0.0),
        (None, _) => (total / n as f32).max(tallest).max(0.0),
    };
    let mut work: Vec<WorkLine> = extents
        .iter()
        .map(|e| {
            let line = &lines[e.idx];
            WorkLine {
                start: line.start,
                end: line.end,
                dx: line.dx,
                avail: line.avail,
                orig: Some(e.idx),
                stack: false,
                shift_x: 0.0,
                shift_y: 0.0,
                // The line's legitimate content start in the flow
                // frame: the first fragment's x, which carries any
                // `text-indent` or alignment offset the flow pass
                // applied. (The extent guarantees at least one placed
                // fragment, so `line.start` is in bounds.)
                anchor_x: frags[line.start].x,
            }
        })
        .collect();
    let (max_bottom, rebreak_changed, absorbed_h) = distribute_lines_pass(
        &mut work,
        frags,
        mc,
        content_x,
        content_top,
        col_h,
        forced_breaks,
        &col_floats,
    );
    let mut max_bottom = max_bottom;
    if rebreak_changed {
        // 2-pass re-balance: mid-distribution re-breaking changed
        // the line count, so the target derived from flow-pass totals
        // is stale. Re-derive the target from the actual post-rebreak
        // line heights first: `frag_band` heights are
        // translation-invariant (the pass shifts each line's
        // fragments uniformly), so the re-derivation needs no undo.
        // Only when the target actually changed are the first pass's
        // placement shifts undone (the re-broken line ranges stay) and
        // the pass re-run; an unchanged target would reproduce the
        // first pass exactly, so the first pass's placement stands.
        // (Undoing the shifts without re-running the pass would strand
        // fragments at flow-frame positions while `max_bottom` still
        // reflects the placed layout.)
        let mut total2 = 0.0f32;
        let mut tallest2 = 0.0f32;
        for w in work.iter() {
            if let Some((_, h)) = frag_band(frags, w.start, w.end) {
                total2 += h;
                tallest2 = tallest2.max(h);
            }
        }
        let col_h2 = match (hspec, mc.fill_auto) {
            (Some(h), true) => h.max(0.0),
            (Some(h), false) => (total2 / n as f32).max(tallest2).min(h).max(0.0),
            (None, _) => (total2 / n as f32).max(tallest2).max(0.0),
        };
        if (col_h2 - col_h).abs() > EPS {
            // Undo the first pass's placement shifts and close its
            // pull-up holes in a single O(F) sweep: the lazy scalar
            // left the flow frame stale by the absorbed slots, and the
            // re-run's own shift starts from zero — without the
            // closure it would under-compact every line below a
            // first-pass absorption. `absorbed_h[i]` is the slot freed
            // by `work[i]` (0 when it absorbed nothing); the hole sits
            // below `work[i]`, hence shift-then-accumulate: each line
            // moves up by exactly the holes above it.
            let mut debt = 0.0f32;
            for (i, w) in work.iter().enumerate() {
                let (sx, sy) = (w.shift_x, w.shift_y);
                let hole = if i < absorbed_h.len() {
                    absorbed_h[i]
                } else {
                    0.0
                };
                if sx != 0.0 || sy != 0.0 || debt != 0.0 {
                    for f in frags[w.start..w.end].iter_mut() {
                        f.x -= sx;
                        f.abs_top -= sy + debt;
                    }
                }
                debt += hole;
            }
            for w in work.iter_mut() {
                w.shift_x = 0.0;
                w.shift_y = 0.0;
            }
            let (max_bottom2, _, _) = distribute_lines_pass(
                &mut work,
                frags,
                mc,
                content_x,
                content_top,
                col_h2,
                forced_breaks,
                &col_floats,
            );
            max_bottom = max_bottom2;
        }
    }
    let height = (max_bottom - content_top).max(0.0);
    // Record the `column-rule` geometry for the paint worker (see
    // `ColumnRuleSpec`). The container identity lets a re-layout replace
    // this container's stale spec instead of accumulating alongside it
    // (see `record_column_rule`).
    if let Some(rule) = rule {
        record_column_rule(ColumnRuleSpec {
            container,
            dom_node,
            x_offsets: rule_x_offsets(mc),
            height,
            width: rule.width,
            style: rule.style,
            color: rule.color,
        });
    }
    height
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::{
        BorderStyle, BoxSizing, Display, Position, SpecLen, SpecMargin, StyleData, TextAlign,
        TextTransform, VAlign, WhiteSpace, WritingMode,
    };
    use crate::{FixedFontMetrics, LayoutBox, Rect, Size, layout};

    const VP: Size = Size { w: 800.0, h: 600.0 };

    fn layout_html(html: &str, css: &str) -> LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        layout(&doc, &[&ss], VP, &FixedFontMetrics)
    }

    /// The `<body>` box (`<head>` is `display: none`).
    fn body(l: &LayoutBox) -> &LayoutBox {
        &l.children[0]
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn block_children_balance_across_columns() {
        let l = layout_html(
            r#"<div class="mc"><div class="a"></div><div class="b"></div><div class="c"></div><div class="d"></div></div>"#,
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .a, .b, .c, .d { height: 30px; }",
        );
        let mc = &body(&l).children[0];
        // total 120, col_h 60: a, b in column 1; c, d move whole to column 2.
        let (a, b, c, d) = (
            &mc.children[0],
            &mc.children[1],
            &mc.children[2],
            &mc.children[3],
        );
        assert!(approx(a.rect.x, mc.rect.x));
        assert!(approx(b.rect.x, mc.rect.x));
        assert!(approx(b.rect.y, mc.rect.y + 30.0));
        assert!(approx(c.rect.x, mc.rect.x + 200.0));
        assert!(approx(c.rect.y, mc.rect.y));
        assert!(approx(d.rect.x, mc.rect.x + 200.0));
        assert!(approx(d.rect.y, mc.rect.y + 30.0));
        assert!(approx(mc.rect.h, 60.0));
    }

    #[test]
    fn column_gap_calc_percentage_resolves_against_content_width() {
        // `column-gap` accepts percentages (CSS Multicol 1 §4.1): on a
        // 400px container `calc(10px + 10%)` is a 50px gap, so the
        // second column starts at 175 + 50 = 225. No `had_pct` gate is
        // needed: the percentage base (the container's content width)
        // is definite and independent of the gap, so there is no
        // resolution cycle.
        let l = layout_html(
            r#"<div class="mc"><div class="a"></div><div class="b"></div><div class="c"></div><div class="d"></div></div>"#,
            ".mc { width: 400px; column-count: 2; column-gap: calc(10px + 10%); } \
             .a, .b, .c, .d { height: 30px; }",
        );
        let mc = &body(&l).children[0];
        let c = &mc.children[2];
        assert!(
            approx(c.rect.x, mc.rect.x + 225.0),
            "second column x={}, want 225",
            c.rect.x - mc.rect.x
        );
    }

    #[test]
    fn column_width_calc_percentage_is_invalid() {
        // `column-width` takes no percentages (CSS Multicol 1 §3): a
        // `calc()` with percentage leaves is invalid at computed-value
        // time and computes to `auto` (see the cascade validation), so
        // `column-count: 3` alone governs — three 400/3 ≈ 133.33px
        // columns, and the third child starts the second column there.
        // (Without the validation fix the percentage resolved and the
        // width derived 2 columns.) Layout needs no `had_pct` gate:
        // only lengths reach it.
        let l = layout_html(
            r#"<div class="mc"><div class="a"></div><div class="b"></div><div class="c"></div><div class="d"></div><div class="e"></div><div class="f"></div></div>"#,
            ".mc { width: 400px; column-count: 3; column-width: calc(100px + 10%); column-gap: 0; } \
             .a, .b, .c, .d, .e, .f { height: 30px; }",
        );
        let mc = &body(&l).children[0];
        let third = &mc.children[2];
        assert!(
            approx(third.rect.x, mc.rect.x + 400.0 / 3.0),
            "third child x={}, want 133.33",
            third.rect.x - mc.rect.x
        );
    }

    #[test]
    fn forced_break_before_column_moves_child() {
        let l = layout_html(
            r#"<div class="mc"><div class="a"></div><div class="b"></div><div class="c"></div></div>"#,
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .a { height: 10px; } .b { height: 10px; break-before: column; } .c { height: 50px; }",
        );
        let mc = &body(&l).children[0];
        // total 70, col_h = max(35, 50) = 50: without the forced break, b
        // would stay in column 1 (10 + 10 <= 50); the break pushes it to
        // the top of column 2. c overflows the last column (no lines to
        // split).
        let (a, b, c) = (&mc.children[0], &mc.children[1], &mc.children[2]);
        assert!(approx(a.rect.x, mc.rect.x));
        assert!(approx(b.rect.x, mc.rect.x + 200.0));
        assert!(approx(b.rect.y, a.rect.y));
        assert!(approx(c.rect.x, mc.rect.x + 200.0));
        assert!(approx(mc.rect.h, 60.0));
    }

    #[test]
    fn forced_break_after_column_pushes_next_child() {
        let l = layout_html(
            r#"<div class="mc"><div class="a"></div><div class="b"></div><div class="c"></div></div>"#,
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .a { height: 10px; break-after: column; } .b { height: 10px; } .c { height: 50px; }",
        );
        let mc = &body(&l).children[0];
        let (a, b) = (&mc.children[0], &mc.children[1]);
        assert!(approx(a.rect.x, mc.rect.x));
        assert!(approx(b.rect.x, mc.rect.x + 200.0));
        assert!(approx(b.rect.y, a.rect.y));
    }

    #[test]
    fn forced_break_before_first_child_is_noop() {
        let l = layout_html(
            r#"<div class="mc"><div class="a"></div><div class="b"></div></div>"#,
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .a { height: 10px; break-before: column; } .b { height: 10px; }",
        );
        let mc = &body(&l).children[0];
        // A forced break into a fresh column creates no empty column: a
        // stays in column 1, b moves whole to column 2 by fit.
        let (a, b) = (&mc.children[0], &mc.children[1]);
        assert_eq!(mc.children.len(), 2);
        assert!(approx(a.rect.x, mc.rect.x));
        assert!(approx(b.rect.x, mc.rect.x + 200.0));
    }

    /// Ten explicit lines via `white-space: pre` (deterministic line
    /// boxes, no wrapping involved).
    fn ten_lines_html() -> String {
        let text = (1..=10)
            .map(|k| format!("l{k}"))
            .collect::<Vec<_>>()
            .join("\n");
        format!(r#"<div class="mc"><div class="a"></div><div class="t">{text}</div></div>"#)
    }

    #[test]
    fn tall_block_child_splits_across_columns() {
        let l = layout_html(
            &ten_lines_html(),
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .a { height: 80px; } .t { white-space: pre; line-height: 20px; }",
        );
        let mc = &body(&l).children[0];
        // total = 80 + 200 = 280, col_h = 140. .a fills column 1; .t does
        // not fit the remaining 60px, so it splits after line 3 (60px),
        // the 7 remaining lines continuing at the top of column 2.
        assert_eq!(mc.children.len(), 3);
        let (a, t1, t2) = (&mc.children[0], &mc.children[1], &mc.children[2]);
        assert!(approx(a.rect.h, 80.0));
        assert!(approx(t1.rect.x, mc.rect.x));
        assert!(approx(t1.rect.y, mc.rect.y + 80.0));
        assert!(approx(t1.rect.h, 60.0));
        assert!(approx(t2.rect.x, mc.rect.x + 200.0));
        assert!(approx(t2.rect.y, mc.rect.y));
        assert!(approx(t2.rect.h, 140.0));
        // Text fragments partitioned by line: 3 above the break, 7 below.
        assert_eq!(t1.children[0].fragments.len(), 3);
        assert_eq!(t2.children[0].fragments.len(), 7);
        assert!(approx(mc.rect.h, 140.0));
    }

    #[test]
    fn split_snaps_to_line_boundary_no_line_lost() {
        // Non-aligned column height: total = 90 + 200 = 290, col_h = 145,
        // so the raw break (55px into the text) straddles line 3. The
        // split must snap to the bottom of line 2: the fragments hold 2
        // and 8 lines, summing to the original 10 — a straddling line
        // satisfies neither band predicate and would otherwise be dropped
        // from both fragments.
        let l = layout_html(
            &ten_lines_html(),
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .a { height: 90px; } .t { white-space: pre; line-height: 20px; }",
        );
        let mc = &body(&l).children[0];
        assert_eq!(mc.children.len(), 3);
        let (a, t1, t2) = (&mc.children[0], &mc.children[1], &mc.children[2]);
        assert!(approx(a.rect.h, 90.0));
        assert!(approx(t1.rect.h, 40.0));
        assert_eq!(t1.children[0].fragments.len(), 2);
        assert!(approx(t2.rect.x, mc.rect.x + 200.0));
        assert!(approx(t2.rect.y, mc.rect.y));
        assert!(approx(t2.rect.h, 160.0));
        assert_eq!(t2.children[0].fragments.len(), 8);
        assert!(approx(mc.rect.h, 160.0));
    }

    #[test]
    fn break_inside_avoid_keeps_child_whole() {
        let l = layout_html(
            &ten_lines_html(),
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .a { height: 80px; } .t { white-space: pre; line-height: 20px; break-inside: avoid; }",
        );
        let mc = &body(&l).children[0];
        // No split: .t moves whole to column 2, overflowing it (200 > 140).
        assert_eq!(mc.children.len(), 2);
        let t = &mc.children[1];
        assert!(approx(t.rect.x, mc.rect.x + 200.0));
        assert!(approx(t.rect.h, 200.0));
        assert_eq!(t.children[0].fragments.len(), 10);
    }

    #[test]
    fn first_column_keeps_float_intrusion() {
        let l = layout_html(
            r#"<div class="f"></div><div class="mc"><div class="b"></div></div>"#,
            ".f { float: left; width: 100px; height: 20px; } \
             .mc { width: 400px; column-count: 2; column-gap: 0; } \
             .b { height: 30px; overflow: hidden; }",
        );
        let mc = &body(&l).children[1];
        // .b establishes a BFC: the flow pass shifts it right past the
        // float (x = 100). The float fits in column 1, so it stays, and
        // .b keeps its intrusion (see
        // `phantom_indent_shed_when_float_pushed_out_of_column_one` for
        // the pushed-float case).
        let b = &mc.children[0];
        assert!(
            approx(b.rect.x, mc.rect.x + 100.0),
            "b kept its intrusion {b:?}"
        );
    }

    #[test]
    fn later_column_sheds_float_shift_but_keeps_margin() {
        let l = layout_html(
            r#"<div class="f"></div><div class="mc"><div class="a"></div><div class="b"></div></div>"#,
            ".f { float: left; width: 100px; height: 150px; } \
             .mc { width: 400px; column-count: 2; column-gap: 0; } \
             .a { height: 100px; } .b { height: 30px; overflow: hidden; margin-left: 20px; }",
        );
        let mc = &body(&l).children[1];
        // total = 130, col_h = max(65, 100) = 100: .a fills column 1, .b
        // moves to column 2. In flow .b sat at x = 100 (float) + 20
        // (margin); in column 2 no float intrudes, so x = 200 + 20.
        let (a, b) = (&mc.children[0], &mc.children[1]);
        assert!(approx(a.rect.x, mc.rect.x));
        assert!(
            approx(b.rect.x, mc.rect.x + 220.0),
            "b shed the float shift {b:?}"
        );
    }

    #[test]
    fn later_column_lines_shed_float_intrusion() {
        let text = (1..=10)
            .map(|k| format!("l{k}"))
            .collect::<Vec<_>>()
            .join("\n");
        let l = layout_html(
            &format!(r#"<div class="f"></div><div class="mc">{text}</div>"#),
            ".f { float: left; width: 100px; height: 300px; } \
             .mc { width: 400px; column-count: 2; column-gap: 0; white-space: pre; line-height: 20px; }",
        );
        let mc = &body(&l).children[1];
        // col_w = 400; the float intrudes into every line (dx = 100).
        // total = 200, col_h = 100: lines 1-5 stay in column 1 keeping the
        // intrusion, lines 6-10 move to column 2 with the stale shift shed.
        let text_box = &mc.children[0];
        assert_eq!(text_box.box_type, BoxType::Text);
        let frags = &text_box.fragments;
        assert_eq!(frags.len(), 10);
        assert!(approx(frags[0].rect.x, mc.rect.x + 100.0));
        assert!(approx(frags[5].rect.x, mc.rect.x + 200.0));
        assert!(approx(frags[5].rect.y, mc.rect.y));
    }

    #[test]
    fn float_moves_to_column_of_inflow_position() {
        let l = layout_html(
            r#"<div class="mc"><div class="t"></div><div class="f"></div><div class="u"></div></div>"#,
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .t { height: 80px; } .f { float: left; width: 100px; height: 60px; } \
             .u { height: 80px; break-inside: avoid; }",
        );
        let mc = &body(&l).children[0];
        // col_w = 200. Flow: .t [0,80]; .f floats at [0,100]x[80,140];
        // .u at y=80. seg_total = 160, col_h = 80: .t fills column 1,
        // .f's in-flow top (80) falls in column 2's band, .u moves whole
        // to column 2 (unsplittable).
        assert_eq!(mc.children.len(), 3);
        let (t, f, u) = (&mc.children[0], &mc.children[1], &mc.children[2]);
        assert!(approx(t.rect.x, mc.rect.x));
        assert!(approx(t.rect.y, mc.rect.y));
        // The float floats against column 2's width: x = column offset,
        // y keeps the within-band offset.
        assert!(approx(f.rect.x, mc.rect.x + 200.0), "float x {f:?}");
        assert!(approx(f.rect.y, mc.rect.y), "float y {f:?}");
        // .u is not a BFC: it keeps full width at the column offset (the
        // float intrudes into its content, not its box).
        assert!(approx(u.rect.x, mc.rect.x + 200.0), "u x {u:?}");
        assert!(approx(u.rect.y, mc.rect.y), "u y {u:?}");
    }

    #[test]
    fn float_pushed_whole_to_next_column_when_it_does_not_fit() {
        let l = layout_html(
            r#"<div class="mc"><div class="t"></div><div class="f"></div><div class="u"></div></div>"#,
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .t { height: 50px; } .f { float: left; width: 100px; height: 80px; } \
             .u { height: 110px; overflow: hidden; }",
        );
        let mc = &body(&l).children[0];
        // col_w = 200. Flow: .t [0,50]; .f at [0,100]x[50,130]; .u is a
        // BFC narrowed beside the float ([100,200]x[50,160]). seg_total =
        // 160, col_h = 80: .t fills column 1; .f's in-flow top (50) is in
        // column 1's band but 50 + 80 > 80, so it moves whole to column
        // 2's top instead of fragmenting; .u moves whole to column 2 and
        // wraps beside the float.
        assert_eq!(mc.children.len(), 3);
        let (f, u) = (&mc.children[1], &mc.children[2]);
        assert!(approx(f.rect.x, mc.rect.x + 200.0), "float x {f:?}");
        assert!(approx(f.rect.y, mc.rect.y), "float y {f:?}");
        assert!(approx(f.rect.h, 80.0), "float keeps full height {f:?}");
        // .u was narrowed to 100px beside the flow-pass float; in column
        // 2 it shifts past the moved float instead of overlapping it.
        assert!(approx(u.rect.x, mc.rect.x + 300.0), "u x {u:?}");
        assert!(approx(u.rect.y, mc.rect.y), "u y {u:?}");
    }

    #[test]
    fn clear_reevaluated_against_column_floats() {
        let l = layout_html(
            r#"<div class="mc"><div class="t"></div><div class="f"></div><div class="c"></div></div>"#,
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .t { height: 50px; } .f { float: left; width: 100px; height: 80px; } \
             .c { height: 30px; clear: left; break-inside: avoid; }",
        );
        let mc = &body(&l).children[0];
        // Flow: .f at [0,100]x[50,130]; .c clears it, landing at y=130.
        // col_h = 80: .f is pushed whole to column 2's top ([200,300]x
        // [0,80]); .c's in-flow top (130) falls in column 2's band at
        // offset 50, but `clear: left` re-evaluates against column 2's
        // floats and pushes it just below the float.
        assert_eq!(mc.children.len(), 3);
        let (f, c) = (&mc.children[1], &mc.children[2]);
        assert!(approx(f.rect.x, mc.rect.x + 200.0), "float x {f:?}");
        assert!(approx(f.rect.y, mc.rect.y), "float y {f:?}");
        assert!(approx(c.rect.x, mc.rect.x + 200.0), "c x {c:?}");
        assert!(approx(c.rect.y, mc.rect.y + 80.0), "c y {c:?}");
    }

    #[test]
    fn clear_ignores_float_following_in_document_order() {
        // CSS 2.1 §9.5.2: `clear` only clears *preceding* floats. `.b`
        // (clear: left) moves whole to column 2, where the later-in-DOM
        // float `.f` was assigned; the (d) fix-up must not push `.b`
        // below it.
        let l = layout_html(
            r#"<div class="mc"><div class="t"></div><div class="b"></div><div class="f"></div></div>"#,
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .t { height: 80px; } .b { height: 30px; clear: left; } \
             .f { float: left; width: 50px; height: 60px; }",
        );
        let mc = &body(&l).children[0];
        // Flow: .t fills 0..80, .b at 80..110, .f floats at 110..170.
        // col_h = 110/2 = 55 (empty blocks are splittable, so nothing is
        // rigid): .t overflows column 1 whole; .b moves whole to column
        // 2's top. .f's in-flow top (110) falls in column 2's band at
        // offset 55, so it lands at y=55..115 there — but `.b` stays at
        // the column top: the following float does not clear it.
        assert_eq!(mc.children.len(), 3);
        let (t, b, f) = (&mc.children[0], &mc.children[1], &mc.children[2]);
        assert!(approx(t.rect.x, mc.rect.x), "t x {t:?}");
        assert!(approx(t.rect.y, mc.rect.y), "t y {t:?}");
        assert!(approx(b.rect.x, mc.rect.x + 200.0), "b x {b:?}");
        assert!(approx(b.rect.y, mc.rect.y), "b y {b:?}");
        assert!(approx(f.rect.x, mc.rect.x + 200.0), "float x {f:?}");
        assert!(approx(f.rect.y, mc.rect.y + 55.0), "float y {f:?}");
    }

    #[test]
    fn nested_float_entry_synced_to_placed_geometry() {
        // A float nested inside a block child that column balancing moves
        // whole: only direct floats get their context entries updated by
        // `assign_column_floats`, so without the nested-float sync the
        // entry keeps its stale in-flow geometry and the BFC containment
        // measures the container too short.
        let l = layout_html(
            r#"<div class="mc"><div class="t"></div><div class="f1"></div><div class="f2d"></div><div class="b"><div class="f3"></div></div></div>"#,
            ".mc { width: 400px; column-count: 2; column-gap: 0; overflow: hidden; } \
             .t { height: 20px; } .f1 { float: left; width: 100px; height: 60px; } \
             .f2d { float: left; width: 50px; height: 60px; } \
             .b { height: 30px; clear: left; } .f3 { float: left; width: 50px; height: 90px; }",
        );
        let mc = &body(&l).children[0];
        // col_w = 200, col_h = 25. Both direct floats are pushed whole to
        // column 2 ([200,300]x[0,60] and [300,350]x[60,120]). .b's in-flow
        // top (60) falls in column 2's band; the (d) `clear` fix-up pushes
        // it below the column's floats to y=120, and the nested float .f3
        // rides along: its real bottom is 120+90=210 while the stale
        // in-flow entry bottom is 150.
        let b = &mc.children[3];
        let f3 = &b.children[0];
        assert!(approx(b.rect.x, mc.rect.x + 200.0), "b x {b:?}");
        assert!(approx(b.rect.y, mc.rect.y + 120.0), "b y {b:?}");
        assert!(approx(f3.rect.x, mc.rect.x + 200.0), "f3 x {f3:?}");
        assert!(approx(f3.rect.y, mc.rect.y + 120.0), "f3 y {f3:?}");
        assert!(
            approx(mc.rect.h, 210.0),
            "container covers the moved nested float {mc:?}"
        );
    }

    #[test]
    fn clear_sibling_after_container_resolves_against_moved_nested_float() {
        // Non-BFC container: its floats live in the parent's shared
        // context. A `clear` sibling laid out after the container must
        // clear the nested float's real column-2 position, not the stale
        // in-flow entry.
        let l = layout_html(
            r#"<div class="mc"><div class="t"></div><div class="f1"></div><div class="f2d"></div><div class="b"><div class="f3"></div></div></div><div class="clr"></div>"#,
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .t { height: 20px; } .f1 { float: left; width: 100px; height: 60px; } \
             .f2d { float: left; width: 50px; height: 60px; } \
             .b { height: 30px; clear: left; } .f3 { float: left; width: 50px; height: 90px; } \
             .clr { height: 20px; clear: left; }",
        );
        let mc = &body(&l).children[0];
        let clr = &body(&l).children[1];
        // Same placement as above: the nested float really sits at
        // [200,250]x[120,210]. The stale entry bottom (150) equals the
        // container height, so only the synced entry pushes the clear
        // sibling below the float.
        assert!(approx(mc.rect.h, 150.0), "container height {mc:?}");
        assert!(
            approx(clr.rect.y, mc.rect.y + 210.0),
            "clear sibling clears the moved nested float {clr:?}"
        );
    }

    #[test]
    fn bfc_height_covers_moved_floats() {
        let l = layout_html(
            r#"<div class="mc"><div class="t"></div><div class="f"></div></div>"#,
            ".mc { width: 400px; column-count: 2; column-gap: 0; overflow: hidden; } \
             .t { height: 50px; } .f { float: left; width: 100px; height: 80px; }",
        );
        let mc = &body(&l).children[0];
        // The float's in-flow top (50) lands past column 1 (col_h = 25),
        // so it moves to column 2 at y=25, bottom 105. The BFC
        // containment height must use the placed float geometry, not the
        // stale flow-pass entry.
        let f = &mc.children[1];
        assert!(approx(f.rect.x, mc.rect.x + 200.0), "float x {f:?}");
        assert!(approx(f.rect.y, mc.rect.y + 25.0), "float y {f:?}");
        assert!(
            approx(mc.rect.h, 105.0),
            "container covers the moved float {mc:?}"
        );
    }

    #[test]
    fn column_two_lines_step_below_column_two_float() {
        // An ancestor right float overlapping column 2's x-range: lines
        // moved there were broken wider than the space beside the float,
        // so they step below it instead of overlapping it.
        let text = (1..=10)
            .map(|k| format!("l{k:032}"))
            .collect::<Vec<_>>()
            .join("\n");
        let l = layout_html(
            &format!(r#"<div class="f"></div><div class="mc">{text}</div>"#),
            ".f { float: right; width: 300px; height: 60px; } \
             .mc { width: 700px; column-count: 2; column-gap: 0; white-space: pre; line-height: 20px; }",
        );
        let mc = &body(&l).children[1];
        // col_w = 350. The float sits at [500,800]x[0,60]: no intrusion
        // in column 1, so lines break full width. total = 200, col_h =
        // 100: lines 1-5 stay in column 1; lines 6-10 move to column 2
        // ([350,700]), where the float leaves 150px — narrower than the
        // broken lines, so the first moved line steps below the float.
        let text_box = &mc.children[0];
        assert_eq!(text_box.box_type, BoxType::Text);
        let frags = &text_box.fragments;
        assert_eq!(frags.len(), 10);
        assert!(approx(frags[0].rect.x, mc.rect.x));
        assert!(approx(frags[5].rect.x, mc.rect.x + 350.0));
        assert!(
            approx(frags[5].rect.y, mc.rect.y + 60.0),
            "line 6 steps below the float {f:?}",
            f = frags[5].rect
        );
    }

    /// Group placed text fragments into visual lines by y, returning
    /// `(y, min_x, width)` per line in y order. Used by the re-break
    /// tests to observe the post-distribution line shape. Callers
    /// pre-filter by column: lines in different columns share y values.
    fn frag_line_shapes<'a>(
        frags: impl Iterator<Item = &'a crate::TextFragment>,
    ) -> Vec<(f32, f32, f32)> {
        let mut lines: Vec<(f32, f32, f32)> = Vec::new(); // (y, min_x, max_right)
        for f in frags {
            let y = (f.rect.y * 10.0).round() / 10.0;
            match lines.iter_mut().find(|l| (l.0 - y).abs() < 0.2) {
                Some(l) => {
                    l.1 = l.1.min(f.rect.x);
                    l.2 = l.2.max(f.rect.x + f.rect.w);
                }
                None => lines.push((y, f.rect.x, f.rect.x + f.rect.w)),
            }
        }
        lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        lines
            .into_iter()
            .map(|(y, x0, x1)| (y, x0, (x1 - x0).max(0.0)))
            .collect()
    }

    #[test]
    fn column_two_lines_reexpand_when_float_does_not_reach() {
        // An ancestor left float narrows the flow-pass break to 100px;
        // it does not reach column 2, so lines moved there re-break to
        // the full 200px instead of keeping the narrowed break shape.
        // The pull-ups change the line count, so the 2-pass re-balance
        // re-derives the target from the post-rebreak count (300/2 =
        // 150) and re-distributes: 7 narrowed lines in column 1 and 7
        // re-expanded lines in column 2 (140px each), instead of the
        // stale flow-pass split (10 narrowed vs 5 merged).
        let text = vec!["word"; 40].join(" ");
        let l = layout_html(
            &format!(r#"<div class="f"></div><div class="mc">{text}</div>"#),
            ".f { float: left; width: 100px; height: 500px; } \
             .mc { width: 400px; column-count: 2; column-gap: 0; line-height: 20px; }",
        );
        let mc = &body(&l).children[1];
        // col_w = 200. Flow: lines break 2 words per line beside the
        // float (avail 100). total = 20 lines * 20px = 400, col_h = 200
        // for the first pass; the re-break pull-ups merge pairs, and the
        // second pass rebalances the 15 survivors at col_h = 150.
        let text_box = &mc.children[0];
        assert_eq!(text_box.box_type, BoxType::Text);
        let col1: Vec<_> = frag_line_shapes(
            text_box
                .fragments
                .iter()
                .filter(|f| f.rect.x < mc.rect.x + 200.0),
        );
        let col2: Vec<_> = frag_line_shapes(
            text_box
                .fragments
                .iter()
                .filter(|f| f.rect.x >= mc.rect.x + 200.0),
        );
        assert_eq!(col1.len(), 7, "7 narrowed column-1 lines");
        assert_eq!(col2.len(), 7, "7 re-expanded column-2 lines");
        for (y, x0, w) in &col1 {
            assert!(
                approx(*x0, mc.rect.x + 100.0),
                "column-1 line keeps its intrusion (y={y}, x0={x0})"
            );
            assert!(
                *w < 120.0,
                "column-1 line keeps its narrowed break shape (y={y}, w={w})"
            );
        }
        for (y, x0, w) in &col2 {
            assert!(
                approx(*x0, mc.rect.x + 200.0),
                "re-broken line sits in column 2 (y={y}, x0={x0})"
            );
            assert!(
                *x0 + *w <= mc.rect.x + 400.0 + 0.01,
                "column-2 line stays inside the column (y={y}, x0={x0}, w={w})"
            );
        }
        // Six of the seven column-2 lines re-expanded past the 100px
        // flow break (the last holds the leftover 2 words).
        let wide = col2.iter().filter(|(_, _, w)| *w > 150.0).count();
        assert_eq!(wide, 6, "6 re-expanded column-2 lines, {col2:?}");
        // Balanced: both columns are 140px (was 200px vs 100px).
        let h1 = col1.iter().map(|(y, _, _)| y + 20.0).fold(0.0f32, f32::max) - mc.rect.y;
        let h2 = col2.iter().map(|(y, _, _)| y + 20.0).fold(0.0f32, f32::max) - mc.rect.y;
        assert!(approx(h1, 140.0), "column-1 height {h1}");
        assert!(approx(h2, 140.0), "column-2 height {h2}");
        assert!(approx(mc.rect.h, 140.0), "container height {}", mc.rect.h);
    }

    #[test]
    fn column_two_lines_wrap_beside_column_two_float() {
        // An ancestor right float overlaps only column 2: lines moved
        // there were broken full-width (350px) and re-break to wrap
        // beside the float (300px) instead of stepping below it.
        let text = vec!["word"; 70].join(" ");
        let l = layout_html(
            &format!(r#"<div class="f"></div><div class="mc">{text}</div>"#),
            ".f { float: right; width: 150px; height: 200px; } \
             .mc { width: 700px; column-count: 2; column-gap: 0; line-height: 20px; }",
        );
        let mc = &body(&l).children[1];
        // col_w = 350. Flow: the float ([650,800]) does not reach
        // column 1's band, so lines break 7 words per line (326.4px).
        // total = 10 lines * 20px = 200, col_h = 100: lines 1-5 stay in
        // column 1; lines 6-10 move to column 2, where the float leaves
        // 300px — the moved lines wrap beside it.
        let text_box = &mc.children[0];
        assert_eq!(text_box.box_type, BoxType::Text);
        let frags = &text_box.fragments;
        let col2: Vec<_> = frags
            .iter()
            .filter(|f| f.rect.x >= mc.rect.x + 350.0 - 0.5)
            .collect();
        assert!(!col2.is_empty(), "some lines moved to column 2");
        let min_y = col2.iter().map(|f| f.rect.y).fold(f32::INFINITY, f32::min);
        assert!(
            min_y < mc.rect.y + 150.0,
            "moved lines wrap beside the float instead of stepping below it (min_y={min_y})"
        );
        // No moved line inside the float's vertical band is wider than
        // the 300px beside it.
        for (y, _, w) in frag_line_shapes(
            text_box
                .fragments
                .iter()
                .filter(|f| f.rect.x >= mc.rect.x + 350.0 - 0.5),
        ) {
            if y < mc.rect.y + 190.0 {
                assert!(
                    w <= 305.0,
                    "column-2 line wraps beside the float (y={y}, w={w})"
                );
            }
        }
    }

    /// Six lines of three 30px fragments feeding
    /// `distribute_inline_lines_for_container` directly, for the
    /// re-balance test below.
    fn rebalance_test_input() -> (Vec<Line>, Vec<Frag>) {
        let mut lines = Vec::new();
        let mut frags = Vec::new();
        for k in 0..6 {
            lines.push(Line {
                start: k * 3,
                end: k * 3 + 3,
                width: 90.0,
                top: 0.0,
                dx: 0.0,
                avail: 100.0,
                y: k as f32 * 10.0,
            });
            for j in 0..3 {
                frags.push(Frag {
                    w: 30.0,
                    ascent: 8.0,
                    descent: 2.0,
                    lh: 10.0,
                    valign: VAlign::Baseline,
                    vshift: 0.0,
                    ib: None,
                    is_space: false,
                    break_before: false,
                    forced_break_before: false,
                    ws: WhiteSpace::Normal,
                    text_range: Some(0..1),
                    placed: true,
                    aligned: true,
                    x: j as f32 * 30.0,
                    rt: 0.0,
                    rb: 10.0,
                    abs_top: k as f32 * 10.0,
                });
            }
        }
        (lines, frags)
    }

    #[test]
    fn rebalance_after_wrap_restores_column_balance() {
        // Six 90px lines in two 100px columns; a right float narrows
        // column 2 to 50px, so lines moved there wrap 1 -> 3 lines
        // mid-distribution. The flow-pass target (60/2 = 30) is stale
        // after that: a single pass strands lines 3-5 in column 2
        // (30px vs 90px). The 2-pass re-balance re-derives the target
        // from the post-wrap line count (120/2 = 60) and re-distributes:
        // column 1 takes lines 0-3 plus line 3's continuations (60px)
        // and lines 4-5 re-wrap into a balanced column 2 (60px).
        let mc = MulticolParams {
            count: 2,
            gap: 0.0,
            col_w: 100.0,
            fill_auto: false,
        };
        let float = FloatEntry {
            x: 150.0,
            y: 0.0,
            w: 50.0,
            h: 1000.0,
            left: false,
            node: 0,
        };
        let (lines, mut frags) = rebalance_test_input();
        let h = distribute_inline_lines_for_container(
            Some(1),
            None,
            &lines,
            &mut frags,
            &mc,
            0.0,
            0.0,
            None,
            &[],
            &[float],
        );
        assert!(approx(h, 60.0), "balanced height, got {h}");
        // Lines 0-3 start in column 1 (x = 0)...
        for fi in [0, 3, 6, 9] {
            assert!(
                approx(frags[fi].x, 0.0),
                "line-start frag {fi} in column 1, x={}",
                frags[fi].x
            );
        }
        // ...including line 3's wrap continuations (frags 10, 11)...
        for fi in [10, 11] {
            assert!(
                approx(frags[fi].x, 0.0),
                "continuation frag {fi} in column 1, x={}",
                frags[fi].x
            );
        }
        // ...while lines 4-5 and their continuations balance column 2
        // (x = 100).
        for fi in [12, 13, 14, 15, 16, 17] {
            assert!(
                approx(frags[fi].x, 100.0),
                "frag {fi} in column 2, x={}",
                frags[fi].x
            );
        }
    }

    #[test]
    fn phantom_indent_shed_when_float_pushed_out_of_column_one() {
        // The float does not fit column 1's remaining height whole, so
        // it moves whole to column 2's top; the BFC block child staying
        // in column 1 sheds the indent the flow pass put beside the
        // float (keeping its narrowed width: re-expanding it needs a
        // child re-layout the balance pass cannot do).
        let l = layout_html(
            r#"<div class="mc"><div class="t"></div><div class="f"></div><div class="u"></div><div class="w"></div></div>"#,
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .t { height: 20px; } .f { float: left; width: 100px; height: 70px; } \
             .u { height: 30px; overflow: hidden; } .w { height: 60px; break-inside: avoid; }",
        );
        let mc = &body(&l).children[0];
        // col_w = 200. Flow: .f at [0,100]x[20,90]; .u is narrowed
        // beside it (x = 100, w = 100). total = 110, col_h = max(55, 60)
        // = 60: the float (20 + 70 > 60) moves whole to column 2's top;
        // .u stays in column 1 and sheds the phantom indent.
        assert_eq!(mc.children.len(), 4);
        let (f, u, w) = (&mc.children[1], &mc.children[2], &mc.children[3]);
        assert!(approx(f.rect.x, mc.rect.x + 200.0), "float x {f:?}");
        assert!(approx(f.rect.y, mc.rect.y), "float y {f:?}");
        assert!(approx(u.rect.x, mc.rect.x), "phantom indent shed {u:?}");
        assert!(approx(u.rect.w, 100.0), "narrowed BFC width kept {u:?}");
        assert!(approx(w.rect.x, mc.rect.x + 200.0), "w x {w:?}");
    }

    /// Hand-built lines/fragments driving `distribute_inline_lines`
    /// directly (bypassing the stylesheet; `column-fill` is a registered
    /// `bowser_css` property).
    fn fill_test_input() -> (Vec<Line>, Vec<Frag>) {
        let mut lines = Vec::new();
        let mut frags = Vec::new();
        for k in 0..4 {
            lines.push(Line {
                start: k,
                end: k + 1,
                width: 10.0,
                top: 0.0,
                dx: 0.0,
                avail: 100.0,
                y: k as f32 * 10.0,
            });
            frags.push(Frag {
                w: 10.0,
                ascent: 8.0,
                descent: 2.0,
                lh: 10.0,
                valign: VAlign::Baseline,
                vshift: 0.0,
                ib: None,
                is_space: false,
                break_before: false,
                forced_break_before: false,
                ws: WhiteSpace::Normal,
                text_range: Some(0..1),
                placed: true,
                aligned: true,
                x: 0.0,
                rt: 0.0,
                rb: 10.0,
                abs_top: k as f32 * 10.0,
            });
        }
        (lines, frags)
    }

    #[test]
    fn column_fill_auto_fills_sequentially() {
        // Four 10px lines under a definite 40px height.
        let mc = MulticolParams {
            count: 2,
            gap: 0.0,
            col_w: 100.0,
            fill_auto: true,
        };
        let (lines, mut frags) = fill_test_input();
        let h = distribute_inline_lines_for_container(
            Some(1),
            None,
            &lines,
            &mut frags,
            &mc,
            0.0,
            0.0,
            Some(40.0),
            &[],
            &[],
        );
        // Sequential: every line stays in column 1 (40 <= 40).
        assert!(approx(h, 40.0));
        for f in &frags {
            assert!(approx(f.x, 0.0));
        }
    }

    #[test]
    fn column_fill_balance_spreads_evenly() {
        let mc = MulticolParams {
            count: 2,
            gap: 0.0,
            col_w: 100.0,
            fill_auto: false,
        };
        let (lines, mut frags) = fill_test_input();
        let h = distribute_inline_lines_for_container(
            Some(1),
            None,
            &lines,
            &mut frags,
            &mc,
            0.0,
            0.0,
            Some(40.0),
            &[],
            &[],
        );
        // Balance under a definite height: 2 lines per column (target
        // min(40 / 2, 40) = 20); the columns are short, the container
        // keeps its definite height via the caller's clamping.
        assert!(approx(h, 20.0));
        assert!(approx(frags[0].x, 0.0));
        assert!(approx(frags[1].x, 0.0));
        assert!(approx(frags[2].x, 100.0));
        assert!(approx(frags[3].x, 100.0));
    }

    /// Eight 90px lines under `column-fill: auto` with a definite 60px
    /// height; a right float narrows column 2 to 50px, so lines moved
    /// there wrap 1 -> 3 mid-distribution (`rebreak_changed`). With
    /// `fill: auto` the re-derived target equals the definite height by
    /// construction, so no second pass runs — but the first pass's
    /// placement must stand: undoing its shifts without re-running the
    /// pass strands every fragment at flow-frame x (piled in column 1)
    /// while the height still reports the placed layout.
    #[test]
    fn fill_auto_rebreak_keeps_pass1_placement() {
        fn wide_lines(n: usize) -> (Vec<Line>, Vec<Frag>) {
            let mut lines = Vec::new();
            let mut frags = Vec::new();
            for k in 0..n {
                lines.push(Line {
                    start: k * 3,
                    end: k * 3 + 3,
                    width: 90.0,
                    top: 0.0,
                    dx: 0.0,
                    avail: 100.0,
                    y: k as f32 * 10.0,
                });
                for j in 0..3 {
                    frags.push(Frag {
                        w: 30.0,
                        ascent: 8.0,
                        descent: 2.0,
                        lh: 10.0,
                        valign: VAlign::Baseline,
                        vshift: 0.0,
                        ib: None,
                        is_space: false,
                        break_before: false,
                        forced_break_before: false,
                        ws: WhiteSpace::Normal,
                        text_range: Some(0..1),
                        placed: true,
                        aligned: true,
                        x: j as f32 * 30.0,
                        rt: 0.0,
                        rb: 10.0,
                        abs_top: k as f32 * 10.0,
                    });
                }
            }
            (lines, frags)
        }
        let mc = MulticolParams {
            count: 2,
            gap: 0.0,
            col_w: 100.0,
            fill_auto: true,
        };
        let float = FloatEntry {
            x: 150.0,
            y: 0.0,
            w: 50.0,
            h: 1000.0,
            left: false,
            node: 0,
        };
        let (lines, mut frags) = wide_lines(8);
        let h = distribute_inline_lines_for_container(
            Some(1),
            None,
            &lines,
            &mut frags,
            &mc,
            0.0,
            0.0,
            Some(60.0),
            &[],
            &[float],
        );
        // Sequential fill: lines 0-5 fill column 1 (60px); lines 6-7
        // wrap into a 60px column 2. Both columns are 60px: balanced.
        assert!(approx(h, 60.0), "filled height, got {h}");
        for fi in [0, 3, 6, 9, 12, 15] {
            assert!(
                approx(frags[fi].x, 0.0),
                "line-start frag {fi} in column 1, x={}",
                frags[fi].x
            );
        }
        // Lines 6-7 (and their wrap continuations) are distributed to
        // column 2, not stranded at the flow-frame x by an undone pass.
        for fi in [18, 19, 20, 21, 22, 23] {
            assert!(
                approx(frags[fi].x, 100.0),
                "frag {fi} in column 2, x={}",
                frags[fi].x
            );
        }
    }

    /// A partial pull-up steals an indented line's head: the re-anchor
    /// must close the stolen-head gap against the line's recorded
    /// content start (flow origin + indent), not against the bare flow
    /// origin — otherwise the `text-indent` is silently destroyed.
    #[test]
    fn pull_up_reanchor_preserves_text_indent() {
        // Four 40px lines; every line carries a 50px float intrusion,
        // and line 3 is text-indented by 20px (its frags start at 70).
        let mut lines = Vec::new();
        let mut frags = Vec::new();
        for k in 0..4 {
            lines.push(Line {
                start: k * 4,
                end: k * 4 + 4,
                width: 40.0,
                top: 0.0,
                dx: 50.0,
                avail: 50.0,
                y: k as f32 * 10.0,
            });
            let x0 = if k == 3 { 70.0 } else { 50.0 };
            for j in 0..4 {
                frags.push(Frag {
                    w: 10.0,
                    ascent: 8.0,
                    descent: 2.0,
                    lh: 10.0,
                    valign: VAlign::Baseline,
                    vshift: 0.0,
                    ib: None,
                    is_space: false,
                    break_before: false,
                    forced_break_before: false,
                    ws: WhiteSpace::Normal,
                    text_range: Some(0..1),
                    placed: true,
                    aligned: true,
                    x: x0 + j as f32 * 10.0,
                    rt: 0.0,
                    rb: 10.0,
                    abs_top: k as f32 * 10.0,
                });
            }
        }
        let mc = MulticolParams {
            count: 2,
            gap: 0.0,
            col_w: 100.0,
            fill_auto: false,
        };
        // A right float narrows column 2 to 60px: wider than the
        // flow-pass 50px (so line 2 pulls line 3's head up) but too
        // narrow for the whole 40px line (so the pull is partial).
        let float = FloatEntry {
            x: 160.0,
            y: 0.0,
            w: 40.0,
            h: 1000.0,
            left: false,
            node: 0,
        };
        let h = distribute_inline_lines_for_container(
            Some(1),
            None,
            &lines,
            &mut frags,
            &mc,
            0.0,
            0.0,
            None,
            &[],
            &[float],
        );
        assert!(approx(h, 20.0), "balanced height, got {h}");
        // Line 3's remaining frags start at the column-2 line origin
        // (100) plus the 20px indent — the indent survives the
        // stolen-head re-anchor.
        assert!(
            approx(frags[14].x, 120.0),
            "indent survives pull-up re-anchor, x={}",
            frags[14].x
        );
        assert!(
            approx(frags[15].x, 130.0),
            "indent survives pull-up re-anchor, x={}",
            frags[15].x
        );
    }

    /// Lazy hole closure matches the eager implementation: twelve
    /// 2-fragment lines where column 2 (narrowed to 60px by a right
    /// float, still wider than the flow pass's 20px) pulls whole
    /// following lines up and absorbs them — several absorptions per
    /// balance pass, plus a partial pull, plus the 2-pass re-balance
    /// with its undo path. The expected `abs_top`s were recorded from
    /// the pre-refactor eager hole-closure implementation; any missed
    /// `cum_hole` adjustment (or double-counted hole across the
    /// re-balance) shows up here as a gap or an overlap.
    #[test]
    fn lazy_hole_closure_matches_eager_placement() {
        let n_lines = 12usize;
        let mut lines = Vec::new();
        let mut frags = Vec::new();
        for k in 0..n_lines {
            lines.push(Line {
                start: k * 2,
                end: k * 2 + 2,
                width: 20.0,
                top: 0.0,
                dx: 0.0,
                avail: 20.0,
                y: k as f32 * 10.0,
            });
            for j in 0..2 {
                frags.push(Frag {
                    w: 10.0,
                    ascent: 8.0,
                    descent: 2.0,
                    lh: 10.0,
                    valign: VAlign::Baseline,
                    vshift: 0.0,
                    ib: None,
                    is_space: false,
                    break_before: false,
                    forced_break_before: false,
                    ws: WhiteSpace::Normal,
                    text_range: Some(0..1),
                    placed: true,
                    aligned: true,
                    x: j as f32 * 10.0,
                    rt: 0.0,
                    rb: 10.0,
                    abs_top: k as f32 * 10.0,
                });
            }
        }
        let mc = MulticolParams {
            count: 2,
            gap: 0.0,
            col_w: 100.0,
            fill_auto: false,
        };
        let float = FloatEntry {
            x: 160.0,
            y: 0.0,
            w: 40.0,
            h: 1000.0,
            left: false,
            node: 0,
        };
        let h = distribute_inline_lines_for_container(
            Some(1),
            None,
            &lines,
            &mut frags,
            &mc,
            0.0,
            0.0,
            None,
            &[],
            &[float],
        );
        assert!(approx(h, 40.0), "balanced height, got {h}");
        // Column 1 keeps lines 0-3; column 2 packs (4,5) at 0,
        // (6,7,8) at 10 and (9,10,11) at 20 — contiguous, no holes.
        let tops = [
            0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0, //
            0.0, 0.0, 0.0, 0.0, //
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, //
            20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
        ];
        for (i, (f, want)) in frags.iter().zip(tops.iter()).enumerate() {
            assert!(
                approx(f.abs_top, *want),
                "frag {i} top={}, want {want}",
                f.abs_top,
            );
        }
    }

    #[test]
    fn place_y_row_math() {
        // First row: fresh non-zero columns snap to the row top; column 0
        // keeps its original y (collapsed top margin).
        let nb = vec![0.0f32, 0.0];
        let ob = vec![0.0f32, 0.0];
        assert!(approx(
            place_y(0, true, 50.0, &nb, &ob, 0.0, 0.0, true),
            50.0
        ));
        assert!(approx(
            place_y(1, true, 50.0, &nb, &ob, 0.0, 0.0, true),
            0.0
        ));
        // Later rows (after a spanner): fresh columns preserve the flow
        // gap below the spanner — row placed top 200, original top 170,
        // child original y 190 -> 220 in every column.
        assert!(approx(
            place_y(0, true, 190.0, &nb, &ob, 200.0, 170.0, false),
            220.0
        ));
        assert!(approx(
            place_y(1, true, 190.0, &nb, &ob, 200.0, 170.0, false),
            220.0
        ));
        // Non-fresh columns preserve the gap to the previous occupant.
        let nb = vec![70.0f32, 0.0];
        let ob = vec![60.0f32, 0.0];
        assert!(approx(
            place_y(0, false, 90.0, &nb, &ob, 0.0, 0.0, true),
            100.0
        ));
    }

    /// A tall child fragments into many continuations across many
    /// columns; the line map must be collected once, not once per split
    /// (operation-counted linearity check for the O(n^2) fixes: the
    /// continuation-root scan, the child-list insert, and the line
    /// collection are all O(1) amortized per split now).
    #[test]
    fn many_splits_collect_lines_once() {
        // 300 lines (6000px) across 50 columns: balanced target 120px
        // (6 lines), so the child splits 49 times into 50 fragments.
        let text = (1..=300)
            .map(|k| format!("l{k}"))
            .collect::<Vec<_>>()
            .join("\n");
        COLLECT_LINES_CALLS.with(|c| *c.borrow_mut() = 0);
        let l = layout_html(
            &format!(r#"<div class="mc"><div class="t">{text}</div></div>"#),
            ".mc { width: 1000px; column-count: 50; column-gap: 0; } \
             .t { white-space: pre; line-height: 20px; }",
        );
        let mc = &body(&l).children[0];
        assert_eq!(
            mc.children.len(),
            50,
            "the child fragments into 50 (49 splits)"
        );
        let calls = COLLECT_LINES_CALLS.with(|c| *c.borrow());
        assert_eq!(
            calls, 1,
            "collect_lines must run once per splittable child, not once per split"
        );
        // All lines survive across the fragments: 300 text fragments total.
        let total: usize = mc
            .children
            .iter()
            .map(|t| t.children[0].fragments.len())
            .sum();
        assert_eq!(total, 300);
    }

    /// `column-rule` geometry is recorded for the paint worker: one spec
    /// per container, rule centered in the gap, resolved width/style/color.
    #[test]
    fn column_rule_geometry_recorded() {
        let _ = take_column_rules(); // drain stragglers from other tests
        let l = layout_html(
            r#"<div class="mc"><div class="a"></div><div class="b"></div></div>"#,
            ".mc { width: 420px; column-count: 2; column-gap: 20px; \
             column-rule: 2px dashed red; } \
             .a, .b { height: 30px; }",
        );
        let mc = &body(&l).children[0];
        let specs = take_column_rules();
        assert_eq!(specs.len(), 1, "one spec per multicol container");
        let spec = &specs[0];
        // col_w = (420 - 20) / 2 = 200; the single rule sits at the gap
        // center, relative to the content-box left edge.
        assert_eq!(spec.x_offsets.len(), 1);
        assert!(approx(spec.x_offsets[0], 210.0));
        // total = 60, balanced col_h = 30.
        assert!(approx(spec.height, 30.0));
        assert!(approx(spec.width, 2.0));
        assert_eq!(spec.style, ColumnRuleStyle::Dashed);
        assert_eq!(
            spec.color,
            bowser_css::Color {
                r: 255,
                g: 0,
                b: 0,
                a: 255
            }
        );
        assert!(spec.container.is_some());
        // The spec's container is the multicol box itself: its LayoutBox
        // carries the same DOM node.
        assert_eq!(spec.dom_node, mc.node);
    }

    /// `column-rule: medium solid` resolves the width to 3px; the default
    /// (`column-rule-style: none`) records nothing.
    #[test]
    fn column_rule_medium_width_and_none_default() {
        let _ = take_column_rules();
        let _ = layout_html(
            r#"<div class="mc"><div class="a"></div></div>"#,
            ".mc { width: 400px; column-count: 2; column-rule: medium solid blue; } \
             .a { height: 30px; }",
        );
        let specs = take_column_rules();
        assert_eq!(specs.len(), 1);
        assert!(approx(specs[0].width, 3.0));
        assert_eq!(specs[0].style, ColumnRuleStyle::Solid);

        // No `column-rule` at all: nothing recorded.
        let _ = layout_html(
            r#"<div class="mc"><div class="a"></div></div>"#,
            ".mc { width: 400px; column-count: 2; } .a { height: 30px; }",
        );
        assert!(take_column_rules().is_empty());
    }

    /// `currentcolor` resolves to the container's used `color`.
    #[test]
    fn column_rule_currentcolor_uses_text_color() {
        let _ = take_column_rules();
        let _ = layout_html(
            r#"<div class="mc"><div class="a"></div></div>"#,
            ".mc { width: 400px; column-count: 2; color: rgb(10, 20, 30); \
             column-rule: 1px solid currentcolor; } \
             .a { height: 30px; }",
        );
        let specs = take_column_rules();
        assert_eq!(specs.len(), 1);
        assert_eq!(
            specs[0].color,
            bowser_css::Color {
                r: 10,
                g: 20,
                b: 30,
                a: 255
            }
        );
    }

    /// Inline content with a rule records a spec too (container unknown
    /// on that path — see `ColumnRuleSpec`).
    #[test]
    fn column_rule_recorded_for_inline_content() {
        let _ = take_column_rules();
        let text = (1..=4)
            .map(|k| format!("l{k}"))
            .collect::<Vec<_>>()
            .join("\n");
        let _ = layout_html(
            &format!(r#"<div class="mc">{text}</div>"#),
            ".mc { width: 400px; column-count: 2; column-gap: 0; \
             white-space: pre; line-height: 20px; \
             column-rule: 4px double green; }",
        );
        let specs = take_column_rules();
        assert_eq!(specs.len(), 1);
        let spec = &specs[0];
        assert!(spec.container.is_some());
        assert!(spec.dom_node.is_some());
        assert_eq!(spec.x_offsets.len(), 1);
        assert!(approx(spec.x_offsets[0], 200.0));
        assert!(approx(spec.width, 4.0));
        assert_eq!(spec.style, ColumnRuleStyle::Double);
    }

    /// An inline-level forced break starts a fresh column at the reported
    /// line index.
    #[test]
    fn inline_forced_break_starts_fresh_column() {
        let mc = MulticolParams {
            count: 2,
            gap: 0.0,
            col_w: 100.0,
            fill_auto: false,
        };
        let (lines, mut frags) = fill_test_input();
        // Four 10px lines; without the break, balance puts lines 0-1 in
        // column 1 and 2-3 in column 2 (col_h = 20). A forced break at
        // line 1 pushes lines 1-3 to column 2 instead.
        let h = distribute_inline_lines_for_container(
            Some(1),
            None,
            &lines,
            &mut frags,
            &mc,
            0.0,
            0.0,
            None,
            &[1],
            &[],
        );
        // Lines 1-3 are forced into column 2 (30px, overflowing its
        // 20px balanced target).
        assert!(approx(h, 30.0));
        assert!(approx(frags[0].x, 0.0));
        assert!(approx(frags[1].x, 100.0));
        assert!(approx(frags[2].x, 100.0));
        assert!(approx(frags[3].x, 100.0));
        // The forced line starts at the column top.
        assert!(approx(frags[1].abs_top, 0.0));
    }

    /// A forced break at a line that already begins a fresh column is a
    /// no-op (no empty column is created).
    #[test]
    fn inline_forced_break_into_fresh_column_is_noop() {
        let mc = MulticolParams {
            count: 2,
            gap: 0.0,
            col_w: 100.0,
            fill_auto: false,
        };
        let (lines, mut frags) = fill_test_input();
        // Line 2 begins column 2 by balance already; forcing it changes
        // nothing.
        let h = distribute_inline_lines_for_container(
            Some(1),
            None,
            &lines,
            &mut frags,
            &mc,
            0.0,
            0.0,
            None,
            &[2],
            &[],
        );
        assert!(approx(h, 20.0));
        assert!(approx(frags[0].x, 0.0));
        assert!(approx(frags[1].x, 0.0));
        assert!(approx(frags[2].x, 100.0));
        assert!(approx(frags[3].x, 100.0));
    }

    /// Real DOM node ids for the spec-identity tests (`NodeId`'s fields
    /// are private, so harvest them from a scratch layout). The scratch
    /// stylesheet sets no `column-rule`, so the layout records nothing.
    fn scratch_multicol_nodes() -> Vec<bowser_dom::NodeId> {
        let _ = take_column_rules();
        let l = layout_html(
            r#"<div class="mc1">a</div><div class="mc2">b</div>"#,
            ".mc1, .mc2 { width: 400px; column-count: 2; }",
        );
        let nodes: Vec<bowser_dom::NodeId> =
            body(&l).children.iter().filter_map(|b| b.node).collect();
        assert_eq!(nodes.len(), 2);
        nodes
    }

    const SPEC_BLACK: bowser_css::Color = bowser_css::Color {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };

    /// Re-layout with changed geometry drops the stale orphan spec: the
    /// same inline-path container re-recorded at a new rule offset must
    /// replace its old spec, not accumulate alongside it (table growth
    /// across passes). The old dedupe only fired for `container: Some`
    /// and relied on exact-duplicate suppression otherwise, so the
    /// changed-geometry spec survived next to the stale one.
    #[test]
    fn relayout_changed_geometry_replaces_stale_orphan_spec() {
        let nodes = scratch_multicol_nodes();
        let node = nodes[0];
        // The inline path records `container: None`; identity comes from
        // the DOM node (stable across passes, unlike the arena index).
        let spec = |x: f32| ColumnRuleSpec {
            container: None,
            dom_node: Some(node),
            x_offsets: vec![x],
            height: 20.0,
            width: 2.0,
            style: ColumnRuleStyle::Solid,
            color: SPEC_BLACK,
        };
        // Pass 1 (e.g. before a viewport resize) records the rule at
        // x = 200; pass 2 re-records the same container at x = 300.
        record_column_rule(spec(200.0));
        record_column_rule(spec(300.0));
        let specs = take_column_rules();
        assert_eq!(
            specs.len(),
            1,
            "stale orphan spec must be replaced, not accumulated: {specs:?}"
        );
        assert!(approx(specs[0].x_offsets[0], 300.0));
    }

    /// Two different inline-path containers with identical geometry keep
    /// separate specs. The old exact-duplicate suppression collapsed them
    /// to one, so a paint worker matching by geometry could hand the
    /// single spec to the wrong container and leave the rightful one
    /// rule-less (claim-ordering theft).
    #[test]
    fn distinct_containers_do_not_collapse_identical_geometry() {
        let nodes = scratch_multicol_nodes();
        for node in nodes.iter().take(2) {
            record_column_rule(ColumnRuleSpec {
                container: None,
                dom_node: Some(*node),
                x_offsets: vec![200.0],
                height: 20.0,
                width: 2.0,
                style: ColumnRuleStyle::Solid,
                color: SPEC_BLACK,
            });
        }
        let specs = take_column_rules();
        assert_eq!(
            specs.len(),
            2,
            "identical geometry must not collapse across containers: {specs:?}"
        );
    }

    /// `begin_multicol_pass` clears both per-pass tables: a dead pending
    /// entry (pushed for an empty container, never popped) must not leak
    /// into the next pass, and recorded specs must not go stale.
    #[test]
    fn begin_multicol_pass_clears_pending_and_recorded() {
        let _ = take_column_rules();
        let mc = MulticolParams {
            count: 2,
            gap: 0.0,
            col_w: 100.0,
            fill_auto: false,
        };
        PENDING_RULES.with(|t| t.borrow_mut().push((mc, None)));
        record_column_rule(ColumnRuleSpec {
            container: None,
            dom_node: None,
            x_offsets: vec![1.0],
            height: 1.0,
            width: 1.0,
            style: ColumnRuleStyle::Solid,
            color: SPEC_BLACK,
        });
        begin_multicol_pass();
        assert!(
            pop_pending_rule(&mc).is_none(),
            "dead pending entry must not survive a pass boundary"
        );
        assert!(
            take_column_rules().is_empty(),
            "recorded specs must not survive a pass boundary"
        );
    }

    /// `merge_lines` without the bottom clamp: the pre-normalization
    /// merge, kept here as the spec for the old behavior (same
    /// merge/push decisions, no clamping).
    fn old_merge_lines(sorted_frags: &[(f32, f32)]) -> Vec<(f32, f32)> {
        let mut lines: Vec<(f32, f32)> = Vec::with_capacity(sorted_frags.len());
        for &(t, b) in sorted_frags {
            match lines.last_mut() {
                Some(last) if t < last.1 - EPS => {
                    last.1 = last.1.max(b);
                }
                _ => lines.push((t, b)),
            }
        }
        lines
    }

    /// Split-position equivalence (P1): the binary search over the
    /// normalized bottoms must pick the same k as the old linear scan
    /// over the unnormalized merge, on adversarial sub-EPS fragment
    /// heights. A sub-EPS fragment starting a new line can end below the
    /// previous line's bottom, producing an adjacent [false, true]
    /// inversion of the `l.1 <= split_y + EPS` predicate — exactly the
    /// case where `partition_point` on the raw sequence is unspecified.
    #[test]
    fn split_position_matches_old_linear_scan() {
        // Hand-built inversions: a tall line, then a sub-EPS fragment
        // starting within EPS of its bottom but ending below it.
        let mut frags: Vec<(f32, f32)> = vec![
            (0.0, 20.0),
            (20.0 - EPS / 2.0, 20.0 - EPS / 4.0),
            (20.0, 40.0),
            (40.0 - EPS * 0.75, 40.0 - EPS * 0.5),
        ];
        // Fuzz: mostly normal lines with occasional sub-EPS slivers
        // starting within EPS of the previous bottom. Deterministic LCG
        // so the case stays fixed.
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f32) / ((1u64 << 31) as f32)
        };
        let mut y = 40.0;
        for _ in 0..200 {
            let h = if next() < 0.15 {
                next() * EPS * 0.9 + 0.00001 // 0 < h < EPS sliver
            } else {
                8.0 + next() * 24.0
            };
            let t = y - next() * EPS * 0.6;
            frags.push((t, t + h));
            y += h + next() * 2.0;
        }
        frags.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let new_lines = merge_lines(&frags);
        let old_lines = old_merge_lines(&frags);
        // Same merge structure: identical line count and tops; bottoms
        // only ever move up, by less than EPS.
        assert_eq!(
            new_lines.len(),
            old_lines.len(),
            "normalization must not change merge decisions"
        );
        for (i, (n, o)) in new_lines.iter().zip(old_lines.iter()).enumerate() {
            assert_eq!(n.0, o.0, "line {i} top changed");
            assert!(n.1 >= o.1, "line {i} bottom moved down: {} -> {}", o.1, n.1);
            assert!(
                n.1 - o.1 <= EPS,
                "line {i} bottom moved more than EPS: {} -> {}",
                o.1,
                n.1
            );
        }
        // The normalized output is a true partition input.
        assert!(
            new_lines.windows(2).all(|w| w[0].1 <= w[1].1),
            "normalized bottoms must be exactly non-decreasing"
        );
        // Sweep split positions across the whole range, dense around the
        // inversion seams: binary search on normalized bottoms must agree
        // with the old linear scan on raw bottoms everywhere.
        let max_b = new_lines.last().map(|l| l.1).unwrap_or(0.0);
        let mut thresholds: Vec<f32> = Vec::new();
        let mut t = -1.0;
        while t < max_b + 1.0 {
            thresholds.push(t);
            t += 0.37;
        }
        for l in old_lines.iter().chain(new_lines.iter()) {
            for d in [-EPS, -EPS / 2.0, 0.0, EPS / 2.0, EPS] {
                thresholds.push(l.1 + d);
            }
        }
        for &split_y in &thresholds {
            let k_new = new_lines.partition_point(|l| l.1 <= split_y + EPS);
            let k_old = old_lines
                .iter()
                .position(|l| l.1 > split_y + EPS)
                .unwrap_or(old_lines.len());
            assert_eq!(
                k_new, k_old,
                "split_y={split_y}: binary search k={k_new} != old linear scan k={k_old}"
            );
        }
    }

    fn numbered_lines(n: u32) -> String {
        (1..=n)
            .map(|k| format!("l{k}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The split loop's index tracking: an original child's continuation
    /// is inserted at `child_pos[c] + continuations.len()`, a re-split
    /// continuation at `last_insert_pos`. Two tall text children across
    /// three balanced columns exercise both branches — T1 splits twice
    /// (its first continuation re-splits) and T2 splits after two
    /// continuations already exist. Every continuation must land
    /// immediately after the fragment it continues: grouped by DOM node
    /// (continuations share their original's node id), the children must
    /// form one contiguous run per original, in original document order,
    /// with every line conserved exactly once per run. Any off-by-one in
    /// the tracked index would interleave or reorder the runs.
    #[test]
    fn continuation_insert_positions_track_live_index() {
        let html = format!(
            r#"<div class="mc"><div class="t1">{}</div><div class="t2">{}</div></div>"#,
            numbered_lines(100),
            numbered_lines(45)
        );
        let l = layout_html(
            &html,
            ".mc { width: 600px; column-count: 3; column-gap: 0; } \
             .t1, .t2 { white-space: pre; line-height: 20px; orphans: 1; widows: 1; }",
        );
        let mc = &body(&l).children[0];
        // total = 2900px, col_h = 966.67: T1 (100 lines) splits after
        // line 48 and its continuation after 48 more; T2 (45 lines)
        // splits after line 44 in the third column.
        assert_eq!(mc.children.len(), 5, "two originals + three continuations");
        let counts: Vec<usize> = mc
            .children
            .iter()
            .map(|c| c.children[0].fragments.len())
            .collect();
        assert_eq!(counts, [48, 48, 4, 44, 1], "fragment line counts per child");
        // Column placement: T1 in column 1, its first continuation in
        // column 2, everything else in column 3.
        for (c, x) in mc.children.iter().zip([0.0, 200.0, 400.0, 400.0, 400.0]) {
            assert!(
                approx(c.rect.x - mc.rect.x, x),
                "child at wrong column: {:?}",
                c.rect
            );
        }
        // Runs by DOM node: one contiguous run per original, in document
        // order, each conserving its lines exactly once.
        let nodes: Vec<_> = mc.children.iter().map(|c| c.node).collect();
        let mut runs: Vec<Option<bowser_dom::NodeId>> = Vec::new();
        for n in &nodes {
            if runs.last() != Some(n) {
                assert!(
                    !runs.contains(n),
                    "node {n:?} appears in two separate runs: {nodes:?}"
                );
                runs.push(*n);
            }
        }
        assert_eq!(runs.len(), 2, "one run per original: {nodes:?}");
        for (run, expected) in runs.iter().zip([100usize, 45usize]) {
            let total: usize = nodes
                .iter()
                .zip(counts.iter())
                .filter(|(n, _)| *n == run)
                .map(|(_, k)| *k)
                .sum();
            assert_eq!(total, expected, "run {run:?} must conserve its lines");
        }
    }

    /// Total text fragments in a fragment's whole subtree (only text
    /// boxes carry fragments, so this counts lines across nesting).
    fn subtree_frags(b: &LayoutBox) -> usize {
        b.fragments.len() + b.children.iter().map(subtree_frags).sum::<usize>()
    }

    /// Count boxes of exactly `w` x `h` in a subtree.
    fn count_sized(b: &LayoutBox, w: f32, h: f32) -> usize {
        usize::from(approx(b.rect.w, w) && approx(b.rect.h, h))
            + b.children
                .iter()
                .map(|c| count_sized(c, w, h))
                .sum::<usize>()
    }

    /// Path-copy golden: a nested splittable child (`.t > .w + .x`)
    /// fragments across three columns. Each split clones only the
    /// straddling chain (`.t` shell, `.w`/`.x` shell, text node) and
    /// moves whole subtrees below the break — the materialized
    /// fragments must match the old deep-clone geometry exactly: three
    /// fragments of 20 lines at the three column x-offsets, every line
    /// conserved once.
    #[test]
    fn nested_split_across_three_columns_path_copy() {
        let html = format!(
            r#"<div class="mc"><div class="t"><div class="w">{}</div><div class="x">{}</div></div></div>"#,
            numbered_lines(30),
            numbered_lines(30)
        );
        let l = layout_html(
            &html,
            ".mc { width: 600px; column-count: 3; column-gap: 0; } \
             .w, .x { white-space: pre; line-height: 20px; orphans: 1; widows: 1; }",
        );
        let mc = &body(&l).children[0];
        // total = 1200px, col_h = 400: 20 lines per column.
        assert_eq!(mc.children.len(), 3, "two splits into three fragments");
        for (c, x) in mc.children.iter().zip([0.0, 200.0, 400.0]) {
            assert!(
                approx(c.rect.x - mc.rect.x, x),
                "fragment at wrong column: {:?}",
                c.rect
            );
            assert!(approx(c.rect.h, 400.0), "fragment height: {:?}", c.rect);
        }
        let counts: Vec<usize> = mc.children.iter().map(subtree_frags).collect();
        assert_eq!(counts, [20, 20, 20], "lines per fragment");
        // Fragment 1 keeps only `.w` (`.x` moved whole below); the first
        // continuation holds the `.w` shell plus the moved `.x`.
        assert_eq!(mc.children[0].children.len(), 1);
        assert_eq!(mc.children[1].children.len(), 2);
        assert_eq!(mc.children[2].children.len(), 1);
        assert!(approx(mc.rect.h, 400.0));
    }

    /// A float below the break rides whole with the continuation and an
    /// out-of-flow child stays in the first fragment only — the
    /// path-copy move must preserve both rules exactly.
    #[test]
    fn split_float_rides_whole_oof_stays_in_first_fragment() {
        let html = format!(
            r#"<div class="mc"><div class="t"><div class="pre">{}</div><div class="fl"></div><div class="post">{}</div><div class="ab"></div></div></div>"#,
            numbered_lines(12),
            numbered_lines(8)
        );
        let l = layout_html(
            &html,
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .pre, .post { white-space: pre; line-height: 20px; orphans: 1; widows: 1; } \
             .fl { float: left; width: 50px; height: 40px; } \
             .ab { position: absolute; width: 30px; height: 30px; }",
        );
        let mc = &body(&l).children[0];
        // total in-flow = 400px, col_h = 200: 10 lines per column. The
        // float's top (240px) is below the break, so it moves whole with
        // the continuation; the abspos child stays in fragment 1.
        assert_eq!(mc.children.len(), 2);
        let (t1, t2) = (&mc.children[0], &mc.children[1]);
        assert!(approx(t1.rect.h, 200.0));
        assert!(approx(t2.rect.x - mc.rect.x, 200.0));
        assert!(approx(t2.rect.h, 200.0));
        assert_eq!(subtree_frags(t1) + subtree_frags(t2), 20);
        assert_eq!(
            count_sized(t1, 50.0, 40.0) + count_sized(t2, 50.0, 40.0),
            1,
            "float must appear exactly once, whole"
        );
        assert_eq!(
            count_sized(t2, 50.0, 40.0),
            1,
            "float rides the continuation"
        );
        assert!(
            approx(
                t2.children
                    .iter()
                    .find(|b| approx(b.rect.w, 50.0) && approx(b.rect.h, 40.0))
                    .map(|b| b.rect.h)
                    .unwrap_or(0.0),
                40.0
            ),
            "float keeps its full height (never split)"
        );
        assert_eq!(count_sized(t1, 30.0, 30.0), 1, "oof stays in fragment 1");
        assert_eq!(
            count_sized(t2, 30.0, 30.0),
            0,
            "oof must not leak into the continuation"
        );
    }

    /// A float straddling the column break rides whole to the next column
    /// instead of being sliced at `split_y` — page-splitter parity
    /// (`layout_fragmentation` moves a straddling float whole to the top
    /// of the next page instead of slicing it mid-box). Regression: the
    /// old `band_filter` pass 2 ran `intersect_band` on the `Keep`
    /// verdict, clipping the float at `split_y` in the first fragment
    /// while the continuation dropped it, so the bottom half vanished.
    #[test]
    fn straddling_float_rides_whole_to_next_column() {
        let html = format!(
            r#"<div class="mc"><div class="t"><div class="fl"></div><div class="pre">{}</div></div></div>"#,
            numbered_lines(20)
        );
        let l = layout_html(
            &html,
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .pre { white-space: pre; line-height: 20px; orphans: 1; widows: 1; } \
             .fl { float: left; width: 50px; height: 300px; }",
        );
        let mc = &body(&l).children[0];
        // total = 400px (20 lines wrapping beside the float), col_h =
        // 200: the break snaps to line 10's bottom (200px), straddling
        // the 300px float [0, 300]. The float must ride whole with the
        // continuation, not be sliced at the break.
        assert_eq!(mc.children.len(), 2);
        let (t1, t2) = (&mc.children[0], &mc.children[1]);
        assert!(approx(t1.rect.h, 200.0));
        assert!(approx(t2.rect.x - mc.rect.x, 200.0));
        assert_eq!(subtree_frags(t1) + subtree_frags(t2), 20);
        assert_eq!(
            count_sized(t1, 50.0, 300.0) + count_sized(t2, 50.0, 300.0),
            1,
            "float must appear exactly once, whole"
        );
        assert_eq!(
            count_sized(t2, 50.0, 300.0),
            1,
            "straddling float rides the continuation"
        );
        assert_eq!(
            count_sized(t1, 50.0, 200.0),
            0,
            "no sliced remnant left in fragment 1"
        );
        assert!(
            approx(
                t2.children
                    .iter()
                    .find(|b| approx(b.rect.w, 50.0) && approx(b.rect.h, 300.0))
                    .map(|b| b.rect.h)
                    .unwrap_or(0.0),
                300.0
            ),
            "float keeps its full height (never split)"
        );
    }

    /// Per-split clone cost is O(depth), not O(subtree): ten 30-line
    /// wrappers (21 subtree nodes) split 49 times across 50 columns.
    /// Path-copy clones ~3 nodes per split (continuation shell, one
    /// straddling wrapper shell, one text node); the old deep clone
    /// copied all 21 subtree nodes per split.
    #[test]
    fn many_splits_clone_only_path_not_subtree() {
        let wrappers = (0..10)
            .map(|_| format!(r#"<div class="w">{}</div>"#, numbered_lines(30)))
            .collect::<Vec<_>>()
            .join("");
        let html = format!(r#"<div class="mc"><div class="t">{wrappers}</div></div>"#);
        CLONE_NODE_CALLS.with(|c| *c.borrow_mut() = 0);
        let l = layout_html(
            &html,
            ".mc { width: 1000px; column-count: 50; column-gap: 0; } \
             .w { white-space: pre; line-height: 20px; orphans: 1; widows: 1; }",
        );
        let mc = &body(&l).children[0];
        let splits = mc.children.len() - 1;
        assert_eq!(mc.children.len(), 50, "49 splits into 50 fragments");
        let total: usize = mc.children.iter().map(subtree_frags).sum();
        assert_eq!(total, 300, "all lines survive the path-copy splits");
        let calls = CLONE_NODE_CALLS.with(|c| *c.borrow());
        // Measured: 129 calls for 49 splits (~2.6 per split: the
        // continuation shell, one straddling wrapper shell, and one text
        // node — a split at a wrapper boundary clones only the shell).
        // The old deep clone copied all 21 subtree nodes per split
        // (1029 calls), so this bound fails loudly on any regression.
        assert!(
            calls <= 8 * splits,
            "clone_node calls {calls} exceed 8 per split ({splits} splits): not path-copy"
        );
    }

    /// Re-layout over a split multicol container: as a column-flex item
    /// with auto height, the container is laid out twice over the same
    /// forest (measure→real), so `balance_block_columns` runs twice.
    /// The second pass's stale strip must restore the whole subtrees
    /// the first pass's splits moved below the band (`.x` here) into
    /// the original at their exact pre-split positions — otherwise the
    /// strip drops the continuation with the moved content inside and
    /// the second pass re-splits an above-band-only remainder. Before
    /// the fix, 30 of the 60 lines vanished and the surviving 30
    /// misbalanced into six 10-line fragments; after the fix the
    /// re-layout balances identically to the single-pass baseline.
    #[test]
    fn relayout_over_split_multicol_restores_moved_content() {
        let flex_html = format!(
            r#"<div class="fx"><div class="mc"><div class="t"><div class="w">{}</div><div class="x">{}</div></div></div></div>"#,
            numbered_lines(30),
            numbered_lines(30)
        );
        let l = layout_html(
            &flex_html,
            ".fx { display: flex; flex-direction: column; width: 600px; } \
             .mc { column-count: 3; column-gap: 0; } \
             .w, .x { white-space: pre; line-height: 20px; orphans: 1; widows: 1; }",
        );
        let mc = &body(&l).children[0].children[0];
        // Baseline: the same multicol balanced once, outside flex.
        let plain_html = format!(
            r#"<div class="mc"><div class="t"><div class="w">{}</div><div class="x">{}</div></div></div>"#,
            numbered_lines(30),
            numbered_lines(30)
        );
        let l2 = layout_html(
            &plain_html,
            ".mc { width: 600px; column-count: 3; column-gap: 0; } \
             .w, .x { white-space: pre; line-height: 20px; orphans: 1; widows: 1; }",
        );
        let base = &body(&l2).children[0];
        // Below-band content intact: every line survives the re-layout.
        let total: usize = mc.children.iter().map(subtree_frags).sum();
        assert_eq!(total, 60, "re-layout must conserve every line");
        // Columns balance identically to the single-pass baseline:
        // three 20-line fragments at the three column x-offsets.
        assert_eq!(mc.children.len(), 3, "two splits into three fragments");
        assert_eq!(mc.children.len(), base.children.len());
        for (c, b) in mc.children.iter().zip(base.children.iter()) {
            assert!(approx(c.rect.h, 400.0), "fragment height: {:?}", c.rect);
            assert!(approx(c.rect.h, b.rect.h), "fragment height: {:?}", c.rect);
            assert_eq!(subtree_frags(c), 20, "lines per fragment");
            assert_eq!(subtree_frags(c), subtree_frags(b));
            assert!(
                approx(c.rect.x - mc.rect.x, b.rect.x - base.rect.x),
                "fragment column: {:?}",
                c.rect
            );
        }
        let xs: Vec<f32> = mc.children.iter().map(|c| c.rect.x - mc.rect.x).collect();
        assert!(
            xs.iter()
                .zip([0.0, 200.0, 400.0])
                .all(|(x, e)| approx(*x, e)),
            "column x-offsets: {xs:?}"
        );
        assert!(approx(mc.rect.h, 400.0), "container height: {:?}", mc.rect);
        assert!(approx(mc.rect.h, base.rect.h));
    }

    /// Minimal `StyleData` for the white-box `assign_column_floats` test
    /// below (H1): only `float`/`position` vary.
    fn test_style_data(float: Float, position: Position) -> StyleData {
        StyleData {
            display: Display::Block,
            font_size: 16.0,
            line_height: 20.0,
            white_space: WhiteSpace::Normal,
            text_align: TextAlign::Left,
            valign: VAlign::Baseline,
            text_transform: TextTransform::None,
            rtl: false,
            bidi_plaintext: false,
            writing_mode: WritingMode::HorizontalTb,
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
            position,
            offset: [SpecLen::Auto; 4],
            z_index: None,
            float,
            clear: Clear::None,
        }
    }

    #[test]
    fn assign_column_floats_matches_entry_by_node_identity() {
        // H1: a nested float's entry shares the direct float's
        // margin-box geometry and side and sits earlier in the context
        // vec. Geometry-only matching would claim the nested entry for
        // the direct float (corrupting it) and leave the direct float's
        // own entry stale at in-flow geometry — one corrupted entry plus
        // one phantom float. Node-identity matching must move exactly
        // the direct float's own entry.
        let mut forest = Forest {
            nodes: Vec::new(),
            root_font_size: 16.0,
            vp_w: 800.0,
            vp_h: 600.0,
            abs_items: Vec::new(),
            gutter_propagated_root: None,
            root_is_html: false,
            union_frag_visits: 0,
        };
        let mc = forest.add_node(
            BoxType::Block,
            None,
            test_style_data(Float::None, Position::Static),
            None,
            None,
        );
        let b = forest.add_node(
            BoxType::Block,
            None,
            test_style_data(Float::None, Position::Static),
            None,
            Some(mc),
        );
        let f3 = forest.add_node(
            BoxType::Block,
            None,
            test_style_data(Float::Left, Position::Static),
            None,
            Some(b),
        );
        let f = forest.add_node(
            BoxType::Block,
            None,
            test_style_data(Float::Left, Position::Static),
            None,
            Some(mc),
        );
        for n in [f3, f] {
            forest.nodes[n].rect = Rect {
                x: 10.0,
                y: 100.0,
                w: 50.0,
                h: 60.0,
            };
        }
        let mut float_ctx = FloatCtx::new();
        // Nested entry first, with the same geometry and side as the
        // direct float's entry.
        for &node in &[f3, f] {
            float_ctx.entries.push(FloatEntry {
                x: 10.0,
                y: 100.0,
                w: 50.0,
                h: 60.0,
                left: true,
                node,
            });
        }
        let params = MulticolParams {
            count: 2,
            gap: 0.0,
            col_w: 200.0,
            fill_auto: false,
        };
        let doc_order: HashMap<usize, usize> = [(f, 0)].into_iter().collect();
        let col_floats = assign_column_floats(
            &mut forest,
            mc,
            &params,
            10.0,         // content_x
            (0.0, 200.0), // seg_orig
            0.0,          // row_top_placed
            100.0,        // col_h
            &mut float_ctx,
            &doc_order,
        );
        // In-flow top 100 with col_h 100 lands in column 2 at (210, 0).
        assert_eq!(col_floats[0].len(), 0);
        assert_eq!(col_floats[1].len(), 1);
        let (placed, order) = &col_floats[1][0];
        assert_eq!(placed.node, f);
        assert!(approx(placed.x, 210.0), "placed x {placed:?}");
        assert!(approx(placed.y, 0.0), "placed y {placed:?}");
        assert_eq!(*order, 0);
        // The nested entry is untouched; the direct float's own entry
        // moved to the placed geometry (no phantom at in-flow).
        assert_eq!(float_ctx.entries[0].node, f3);
        assert!(approx(float_ctx.entries[0].x, 10.0));
        assert!(approx(float_ctx.entries[0].y, 100.0));
        assert_eq!(float_ctx.entries[1].node, f);
        assert!(approx(float_ctx.entries[1].x, 210.0));
        assert!(approx(float_ctx.entries[1].y, 0.0));
        // The deferred shift points at the placed box.
        assert!(approx(forest.nodes[f].pend_x, 200.0));
        assert!(approx(forest.nodes[f].pend_y, -100.0));
    }

    #[test]
    fn abspos_float_child_gets_no_column_shift() {
        // M1: `position: absolute` + `float` keeps `data.float` set, but
        // the abspos post-pass owns the box. Column assignment must skip
        // it (like the `kids` filter does): otherwise it would plant a
        // phantom `ColFloat` and stack the column shift under the abspos
        // `pend_x += fx - px` accumulation, displacing the box.
        let html = r#"<div class="mc"><div class="t"></div><div class="a"></div></div>"#;
        let css_base = ".mc { position: relative; width: 400px; column-count: 2; column-gap: 0; } \
             .t { height: 100px; } \
             .a { position: absolute; left: 30px; top: 40px; margin-left: 25px; width: 50px; height: 60px; }";
        let css_float = ".mc { position: relative; width: 400px; column-count: 2; column-gap: 0; } \
             .t { height: 100px; } \
             .a { position: absolute; float: left; left: 30px; top: 40px; margin-left: 25px; \
                  width: 50px; height: 60px; }";
        let l_base = layout_html(html, css_base);
        let l_float = layout_html(html, css_float);
        let a_base = &body(&l_base).children[0].children[1];
        let a_float = &body(&l_float).children[0].children[1];
        // Adding `float` to an abspos box must not move it.
        assert!(
            approx(a_float.rect.x, a_base.rect.x),
            "abspos+float x must equal plain abspos x: {a_float:?} vs {a_base:?}"
        );
        assert!(
            approx(a_float.rect.y, a_base.rect.y),
            "abspos+float y must equal plain abspos y: {a_float:?} vs {a_base:?}"
        );
    }

    #[test]
    fn spanner_inner_content_lays_out_at_full_width() {
        // 400px container, 2 x 200px columns, no gap. The spanner's text
        // must wrap against the full 400px row, not one 200px column:
        // with FixedFontMetrics (9.6px/char at 16px) a full-width line
        // holds ~41 chars while a column-width line holds ~20.
        let text = "word ".repeat(80);
        let html = format!(
            r#"<div class="mc"><div class="a"></div><div class="sp">{text}</div><div class="b"></div></div>"#
        );
        let l = layout_html(
            &html,
            ".mc { width: 400px; column-count: 2; column-gap: 0; } \
             .a, .b { height: 10px; } .sp { column-span: all; }",
        );
        let mc = &body(&l).children[0];
        assert_eq!(mc.children.len(), 3);
        let sp = &mc.children[1];
        // Full-width row: the border box is widened to the whole content
        // width, starting at the container's content x.
        assert!(approx(sp.rect.x, mc.rect.x));
        assert!(approx(sp.rect.w, 400.0));
        // The first text line uses the full width: group the text box's
        // per-fragment rects by baseline and measure the first line's
        // extent (a column-width line caps at ~192px here; a full-width
        // line spans ~384px).
        let frags = &sp.children[0].fragments;
        assert!(!frags.is_empty(), "spanner text must produce fragments");
        let base0 = frags[0].baseline;
        let (mut x0, mut x1) = (f32::INFINITY, f32::NEG_INFINITY);
        for f in frags.iter().filter(|f| (f.baseline - base0).abs() < 0.01) {
            x0 = x0.min(f.rect.x);
            x1 = x1.max(f.rect.x + f.rect.w);
        }
        let line_w = x1 - x0;
        assert!(
            line_w > 200.0,
            "spanner line must exceed one column width, got {line_w}"
        );
    }

    #[test]
    fn spanner_row_skips_bfc_float_narrowing() {
        // A tall ancestor float overlaps the spanner's row. The spanner
        // is a BFC root (`overflow: hidden`), but per the spec judgment
        // in `layout_block` it keeps the full row width instead of
        // narrowing beside the float the way an ordinary BFC child
        // would (CSS 2.1 §9.5 would shift it past the float and shrink
        // it to the remaining 300px).
        let text = "word ".repeat(80);
        let l = layout_html(
            &format!(
                r#"<div class="f"></div><div class="mc"><div class="a"></div><div class="sp">{text}</div><div class="b"></div></div>"#
            ),
            "body { margin: 0; } .f { float: left; width: 100px; height: 500px; } \
             .mc { width: 400px; column-count: 2; column-gap: 0; } \
             .a { height: 100px; } .sp { column-span: all; overflow: hidden; } .b { height: 100px; }",
        );
        let mc = &body(&l).children[1];
        // Segment 1 (.a, 100px) balances to col_h = 100, so the spanner
        // row starts at y = 100 — inside the float's 0..500 band.
        let sp = &mc.children[1];
        assert!(approx(sp.rect.x, mc.rect.x), "spanner x {sp:?}");
        assert!(
            approx(sp.rect.w, 400.0),
            "spanner stays full-width, w={}",
            sp.rect.w
        );
        // The text uses the full row: with FixedFontMetrics a 400px line
        // holds 8 words (~384px); narrowed to 300px beside the float it
        // would wrap at 6 words (~288px).
        let frags = &sp.children[0].fragments;
        assert!(!frags.is_empty(), "spanner text must produce fragments");
        let base0 = frags[0].baseline;
        let (mut x0, mut x1) = (f32::INFINITY, f32::NEG_INFINITY);
        for f in frags.iter().filter(|f| (f.baseline - base0).abs() < 0.01) {
            x0 = x0.min(f.rect.x);
            x1 = x1.max(f.rect.x + f.rect.w);
        }
        let line_w = x1 - x0;
        assert!(
            line_w > 320.0,
            "spanner line must use the full row, got {line_w}"
        );
    }

    #[test]
    fn abspos_static_position_follows_column_two() {
        // The abspos placeholder sits after .t2, which balances into
        // column 2: the static position must be re-based onto column 2
        // instead of staying at the column-1 flow-pass position.
        let l = layout_html(
            r#"<div class="mc"><div class="t1"></div><div class="t2"></div><div class="a"></div></div>"#,
            ".mc { position: relative; width: 400px; column-count: 2; column-gap: 0; } \
             .t1, .t2 { height: 100px; } \
             .a { position: absolute; width: 50px; height: 60px; }",
        );
        let mc = &body(&l).children[0];
        // total = 200, col_h = 100: .t2 moves whole to column 2.
        let (t2, a) = (&mc.children[1], &mc.children[2]);
        assert!(approx(t2.rect.x, mc.rect.x + 200.0));
        assert!(approx(t2.rect.y, mc.rect.y));
        // Old (column-1) behavior would put .a at (mc.x, mc.y + 200).
        assert!(approx(a.rect.x, mc.rect.x + 200.0), "abspos x {a:?}");
        assert!(approx(a.rect.y, mc.rect.y + 100.0), "abspos y {a:?}");
    }

    #[test]
    fn abspos_static_position_stays_in_column_one() {
        // The placeholder sits between .t1 and .t2 with .t1 in column 1:
        // no re-basing must happen.
        let l = layout_html(
            r#"<div class="mc"><div class="t1"></div><div class="a"></div><div class="t2"></div></div>"#,
            ".mc { position: relative; width: 400px; column-count: 2; column-gap: 0; } \
             .t1, .t2 { height: 100px; } \
             .a { position: absolute; width: 50px; height: 60px; }",
        );
        let mc = &body(&l).children[0];
        // total = 200, col_h = 100: .t1 stays in column 1, .t2 moves
        // whole to column 2; .a's placeholder is after .t1 (column 1).
        let a = &mc.children[1];
        assert!(approx(a.rect.x, mc.rect.x), "abspos x {a:?}");
        assert!(approx(a.rect.y, mc.rect.y + 100.0), "abspos y {a:?}");
    }

    #[test]
    fn abspos_anchor_recovery_exact_when_intrusion_near_column_width() {
        // A left float intruding to within EPS of the column width: the
        // anchor's placed x is `k * stride + dx` with `dx` inside
        // `(col_w - EPS, col_w)`. Recovering `k` by backing one column
        // width off and rounding up must yield the anchor's true column
        // (column 1, `k = 0`), not `k + 1` — the shift site clamps `dx`
        // below `col_w - 2*EPS` to keep the `ceil` exact.
        let l = layout_html(
            r#"<div class="mc"><div class="f"></div><div class="anchor"></div><div class="a"></div><div class="rest"></div></div>"#,
            ".mc { position: relative; width: 600px; column-count: 3; column-gap: 10px; } \
             .f { float: left; width: 193.3328px; height: 60px; } \
             .anchor { overflow: hidden; height: 40px; } \
             .a { position: absolute; width: 10px; height: 10px; } \
             .rest { height: 500px; }",
        );
        let mc = &body(&l).children[0];
        // col_w = (600 - 20) / 3 = 193.333, stride = 203.333. Total =
        // 600, col_h = max(200, 500) = 500: the float and the anchor
        // stay in column 1 (the anchor squeezes beside the float with
        // dx = 193.3328, within EPS of col_w), `.rest` fills column 2.
        // The abspos static x must stay in column 1, not jump one
        // stride into column 2.
        let a = &mc.children[2];
        assert!(approx(a.rect.x, mc.rect.x), "abspos x {a:?}");
    }

    #[test]
    fn abspos_nested_in_moved_block_keeps_within_frame_offset() {
        // .a is nested inside .t2, which balances into column 2. It rides
        // .t2's deferred shift, preserving its within-frame offset; the
        // re-basing must not touch it (that would double-count the column
        // shift).
        let l = layout_html(
            r#"<div class="mc"><div class="t1"></div><div class="t2"><div class="a"></div></div></div>"#,
            ".mc { position: relative; width: 400px; column-count: 2; column-gap: 0; } \
             .t1, .t2 { height: 100px; } \
             .a { position: absolute; width: 50px; height: 60px; }",
        );
        let mc = &body(&l).children[0];
        let t2 = &mc.children[1];
        let a = &t2.children[0];
        assert!(approx(t2.rect.x, mc.rect.x + 200.0));
        assert!(approx(t2.rect.y, mc.rect.y));
        // In flow .a sat at .t2's top-left; it must keep that offset in
        // column 2, not gain a second column shift.
        assert!(approx(a.rect.x, mc.rect.x + 200.0), "abspos x {a:?}");
        assert!(approx(a.rect.y, mc.rect.y), "abspos y {a:?}");
    }

    #[test]
    fn abspos_explicit_offsets_ignore_column_rebasing() {
        // With `left`/`top` set, the static position is never read; the
        // re-basing must not move the box.
        let l = layout_html(
            r#"<div class="mc"><div class="t1"></div><div class="t2"></div><div class="a"></div></div>"#,
            ".mc { position: relative; width: 400px; column-count: 2; column-gap: 0; } \
             .t1, .t2 { height: 100px; } \
             .a { position: absolute; left: 30px; top: 40px; width: 50px; height: 60px; }",
        );
        let mc = &body(&l).children[0];
        let a = &mc.children[2];
        assert!(approx(a.rect.x, mc.rect.x + 30.0), "abspos x {a:?}");
        assert!(approx(a.rect.y, mc.rect.y + 40.0), "abspos y {a:?}");
    }

    #[test]
    fn abspos_after_split_child_anchors_to_continuation_column() {
        // .t splits across the columns; the abspos placeholder follows
        // all of .t's content, so it anchors to the continuation in
        // column 2.
        let text = (1..=10)
            .map(|k| format!("l{k}"))
            .collect::<Vec<_>>()
            .join("\n");
        let l = layout_html(
            &format!(r#"<div class="mc"><div class="t">{text}</div><div class="a"></div></div>"#),
            ".mc { position: relative; width: 400px; column-count: 2; column-gap: 0; } \
             .t { white-space: pre; line-height: 20px; } \
             .a { position: absolute; width: 50px; height: 60px; }",
        );
        let mc = &body(&l).children[0];
        // total = 200, col_h = 100: .t splits 5/5; the continuation sits
        // at the top of column 2 and .a's placeholder follows it.
        assert_eq!(mc.children.len(), 3);
        let (t2, a) = (&mc.children[1], &mc.children[2]);
        assert!(approx(t2.rect.x, mc.rect.x + 200.0));
        assert!(approx(t2.rect.y, mc.rect.y));
        assert!(approx(a.rect.x, mc.rect.x + 200.0), "abspos x {a:?}");
        assert!(approx(a.rect.y, mc.rect.y + 100.0), "abspos y {a:?}");
    }

    #[test]
    fn abspos_rebasing_idempotent_across_relayout() {
        // A multicol container inside a flex-column item is laid out twice
        // over one forest (measure, then real); the static-position delta
        // must not apply twice.
        let l = layout_html(
            r#"<div class="wrap"><div class="mc"><div class="t1"></div><div class="t2"></div><div class="a"></div></div></div>"#,
            ".wrap { display: flex; flex-direction: column; width: 400px; } \
             .mc { position: relative; column-count: 2; column-gap: 0; } \
             .t1, .t2 { height: 100px; } \
             .a { position: absolute; width: 50px; height: 60px; }",
        );
        let mc = &body(&l).children[0].children[0];
        let a = &mc.children[2];
        // Same as the single-pass case: column 2, not double-shifted.
        assert!(approx(a.rect.x, mc.rect.x + 200.0), "abspos x {a:?}");
        assert!(approx(a.rect.y, mc.rect.y + 100.0), "abspos y {a:?}");
    }
}
