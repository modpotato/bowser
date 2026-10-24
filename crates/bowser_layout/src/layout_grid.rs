//! Grid layout (`display: grid`).
//!
//! Two phases on the shared task stack, mirroring the table layout:
//! [`layout_grid`] parses the track listings, sizes the column tracks,
//! auto-places items in row order, and pushes one [`Task::Block`] per
//! item with the item's grid-area content width as a width override
//! (or a shrink-to-fit task when `justify-items` is not `stretch`).
//! [`resume_grid`] runs after every item completes: it sizes the row
//! tracks (fixed lengths resolve directly, `fr` shares definite
//! container height, `auto`/implicit tracks size to content), aligns
//! items inside their areas, and finishes the container height.
//!
//! Documented simplifications for this MVP pass:
//! - Track listings support fixed lengths, percentages, `fr`, `auto`,
//!   `minmax()`, `repeat(<integer>, …)` (expanded eagerly by the CSS
//!   parser), and `repeat(auto-fill|auto-fit, …)` (expanded here, where
//!   the axis size is known — see below). A `minmax()` track
//!   resolves its minimum and maximum independently and clamps the base
//!   size into the range (a minimum larger than the maximum wins);
//!   `min-content` / `max-content` bounds measure content like `auto`.
//! - `repeat(auto-fill, …)` / `repeat(auto-fit, …)`: the repetition count
//!   is the largest N such that N repetitions of the *whole* track list
//!   fit in the container (grid §7.2.3.2): with k definite tracks
//!   summing to S px, `floor((container + gap) / (S + k * gap))`, at
//!   least one. Each track counts with its definite maximum
//!   (`minmax()`), else its definite minimum. An indefinite track
//!   size — or an indefinite container size — falls back to a single
//!   repetition. `auto-fit` additionally collapses repeated
//!   tracks that end up empty (no item placed into or spanning across
//!   them): they take no space and draw no gap. Parser contract: the CSS
//!   parser passes these through as the dedicated `CssValue::Repeat`
//!   descriptor instead of dropping the declaration, so
//!   stylesheet-driven auto-fill/fit works.
//! - `grid-auto-flow: dense` backfills holes: each auto-positioned item
//!   scans from the grid start, filling gaps left by
//!   definite-positioned or spanned items. Sparse flow (the default)
//!   never backtracks, so auto-placed items stay in order.
//!   `grid-auto-flow: column` places column-major (by transposing the
//!   placement problem); `column dense` backfills in column-major order.
//!   `grid-auto-flow` is carried by the cascade (`COMPUTED_PROPS`,
//!   initial `row`), so a stylesheet declaration reaches layout and
//!   dense/column packing activates.
//! - Subgrid: when a grid item has `grid-template-columns: subgrid` /
//!   `grid-template-rows: subgrid` on an axis, the track list is
//!   inherited from the parent grid (css-grid-1 §2.3) — nested subgrids
//!   defer upward, but a non-grid parent (or a grid with no real
//!   listing) makes the axis behave as `none`. On the column axis the
//!   subgrid resolves against the parent grid's *used* track
//!   sizes/positions for the spanned subset (read off the parent's
//!   pending phase-one frame), so its tracks align with the parent's.
//!   On the row axis the parent's used geometry does not exist yet —
//!   rows size in phase two, after the subgrid's own items complete —
//!   so the subgrid instead takes the spanned subset of the parent's
//!   *listing* from the pending phase-one frame: tracks whose used size
//!   is fully determined by the listing — fixed lengths (percentages
//!   against the parent's definite content height) and `minmax()` with
//!   equal fixed bounds — freeze at their exact used sizes (resolved
//!   with the parent's font metrics and height, the same bases the
//!   parent's own resume uses), so definite rows align across nested
//!   subgrids. Every other track (`auto`, `fr`, content-bounded
//!   `minmax()`) keeps its parsed breadth and sizes from the subgrid's
//!   own content — the old approximation — as does inter-track space
//!   distributed by the parent's `align-content` (`space-between` et
//!   al. against a definite parent height). On either inherited axis
//!   the subgrid's own gaps and content alignment do not apply: the
//!   parent's gap spaces the inherited tracks, which pack from the
//!   start edge. With no grid ancestor carrying a real listing the axis
//!   degrades to a single `auto` track.
//! - `order` reorders auto-placement (stable: equal orders keep DOM
//!   order; painting order is not re-sorted). `justify-content` /
//!   `align-content` distribute leftover space after track sizing, like
//!   flex's `distribute_lines`, except that `normal` (the initial value)
//!   behaves as `stretch` (css-grid-1 §10.4/§11.2): content-sized tracks
//!   that took no `fr` share grow equally to absorb the free space,
//!   still respecting `minmax()` maxima.
//! - `grid-auto-columns` / `grid-auto-rows` size implicit tracks (the
//!   listing cycles past the explicit tracks; an empty listing means
//!   content-sized `auto`, as before).
//! - Per-item `justify-self` / `align-self` (`start` | `end` | `center`
//!   | `stretch`, plus the `flex-start`/`flex-end`/`self-start`/`self-end`
//!   aliases) override the container's `justify-items` / `align-items`;
//!   `auto` (the initial value) defers to the container and `normal`
//!   behaves as `stretch`.
//! - Item margins resolve and collapse normally inside the item, but
//!   alignment positions the border box: a nonzero inline margin may
//!   push the box past the area edge, and stretched items with auto
//!   height are re-laid out at their stretched size (percentage heights
//!   and nested stretching inside them resolve against the final size).
//! - Definite `grid-row` / `grid-column` positions that collide place
//!   anyway (items may overlap); fully definite items never move the
//!   auto-placement cursor (css-grid-1 §8.5 step 1), the sparse cursor
//!   never backtracks into earlier rows (dense flow resets it to the
//!   grid start after each item, backfilling holes), and an item whose
//!   definite rows exceed the explicit grid grows the implicit grid
//!   downward. A lone `span <n>` on either axis (`grid-row: span 2`)
//!   auto-places with that span (§8.3); the implicit column count is
//!   sized up front for the largest auto column span (§8.5 step 3), so
//!   a spanning item wraps to a fresh row instead of growing implicit
//!   columns mid-row.
//! - Out-of-flow children take no grid space; their static position is
//!   the grid content-box origin, aligned per `justify-self` /
//!   `align-self` as if the child were the sole grid item in the
//!   content box (css-grid-1 §9.2). `float` / `clear` have no effect on
//!   a grid item (css-grid-1 §5.1): a floated child stays a grid item
//!   and is placed and sized like any other item.
//! - Percentage row tracks resolve against the container's definite
//!   content height and behave as `auto` otherwise. Item percentage
//!   heights in the measure pass resolve against the container's
//!   specified height (`auto` when the container is auto-height).
//! - An empty explicit listing (`grid-template-columns: none`) means
//!   every auto-placed item wraps to a fresh implicit row, so items
//!   stack vertically (one implicit row per item).
//! - Min-content sizing ([`grid_min_width`], the `min_content_width`
//!   counterpart of [`grid_pref_width`]) sizes each column track from
//!   its items' min-content contributions: single-span items first,
//!   then a multi-span item's deficit is spread equally over the
//!   content-sized tracks it spans. The equal spread is a
//!   simplification — css-grid-1 §12.5 distributes a spanning item's
//!   extra space with regard to the spanned tracks' own sizing
//!   functions — and mirrors the max-content pass's spanning treatment,
//!   so the two intrinsic passes cannot diverge on span handling.

use std::collections::HashSet;

use bowser_css::{ComputedStyle, CssValue, MAX_GRID_LINE, RepeatMode, props};

use super::BoxType;
use super::layout_position;
use super::{
    AutoRedriveCheck, BlockInput, FontMetrics, PrefWidthCache, SpecLen, SpecMargin, Task, WalkBufs,
    auto_scrollbar_redrive, compute_mt_place, definite_specified_content_width,
    finish_content_height, fresh_floats, intrinsic_content_width, is_block_level,
    min_content_width_pooled, preferred_width_mcw, record_shift, resolve_metrics,
    saturate_used_size, spec_len_of, specified_height,
};
use crate::boxes::Forest;

/// How a grid item is aligned along one axis inside its grid area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ItemAlign {
    Start,
    End,
    Center,
    Stretch,
}

/// A grid item's placement after auto-placement.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlacedItem {
    pub(crate) node: usize,
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) rowspan: usize,
    pub(crate) colspan: usize,
    /// Effective inline-axis alignment: the item's `justify-self`, or
    /// the container's `justify-items` when it is `auto`.
    pub(crate) justify: ItemAlign,
    /// Effective block-axis alignment: the item's `align-self`, or the
    /// container's `align-items` when it is `auto`.
    pub(crate) align: ItemAlign,
}

/// Resume frame for [`Task::Grid`]: everything phase one computed that
/// phase two needs after the item block tasks complete.
#[derive(Debug, Clone)]
pub(crate) struct GridResume {
    pub(crate) idx: usize,
    pub(crate) content_x: f32,
    pub(crate) content_top: f32,
    pub(crate) content_w: f32,
    pub(crate) cb_h: Option<f32>,
    pub(crate) hspec: Option<f32>,
    pub(crate) font_size: f32,
    pub(crate) items: Vec<PlacedItem>,
    pub(crate) col_x: Vec<f32>,
    pub(crate) col_w: Vec<f32>,
    pub(crate) col_gap: f32,
    pub(crate) row_gap: f32,
    /// Row track listing, explicit tracks followed by implicit tracks
    /// filled from `grid-auto-rows` (parallel to `row_collapse`, which
    /// is `false` for every implicit track).
    pub(crate) row_template: Vec<TrackSize>,
    /// Per row track (parallel to `row_template`): collapse when empty
    /// (`repeat(auto-fit, …)` tracks only).
    pub(crate) row_collapse: Vec<bool>,
    pub(crate) nrows: usize,
    /// `align-content`: leftover block-axis space distributes after row
    /// sizing (columns distribute in phase one, directly into `col_x`).
    pub(crate) align_content: ContentAlign,
    /// Second-pass resume: stretched items were re-laid out at their
    /// stretched heights, so the stored row geometry is reused instead
    /// of re-sized (re-sizing from the stretched heights would be
    /// circular).
    pub(crate) second_pass: bool,
    pub(crate) row_h: Vec<f32>,
    pub(crate) row_y: Vec<f32>,
    pub(crate) natural_h: f32,
    /// The original block input this grid layout was driven from: the
    /// `overflow-y: auto` scrollbar re-drive re-runs `layout_grid` with
    /// the identical input against a narrowed width.
    pub(crate) input: BlockInput,
    /// Whether this resume already is the `overflow-y: auto`
    /// scrollbar-width re-drive: the re-drive runs at most once per
    /// frame, so narrowing can never oscillate or loop to a fixpoint.
    pub(crate) auto_redriven: bool,
}

/// One side of a `minmax()` track: a fixed length/percentage, an `fr`
/// factor, or a content-based size. `auto`, `min-content`, and
/// `max-content` all measure content in this simplified pass.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MinMaxBound {
    Fixed(SpecLen),
    Fr(f32),
    Content,
}

/// One track in a template listing.
#[derive(Debug, Clone, Copy)]
pub(crate) enum TrackSize {
    /// A fixed length or percentage.
    Fixed(SpecLen),
    /// A flexible `fr` factor (minimum is its `auto` content size).
    Fr(f32),
    /// Content-sized.
    Auto,
    /// `minmax(min, max)`: the track's base size is at least `min` and
    /// its growth is limited by `max` (each side resolved independently,
    /// then clamped; a minimum larger than the maximum wins).
    MinMax { min: MinMaxBound, max: MinMaxBound },
}

/// One side of a `grid-row` / `grid-column` placement.
#[derive(Debug, Clone, Copy, PartialEq)]
enum GridLine {
    Auto,
    /// One-based explicit line (negative counts from the explicit end).
    Line(i32),
    /// `span <n>`: cover n tracks.
    Span(u32),
}

/// Parsed `grid-row` / `grid-column` value.
#[derive(Debug, Clone, Copy)]
struct GridPlacement {
    start: GridLine,
    end: GridLine,
}

/// An in-flow grid item before placement.
#[derive(Debug, Clone, Copy)]
struct ItemSpec {
    node: usize,
    row: GridPlacement,
    col: GridPlacement,
}

/// An item after auto-placement.
#[derive(Debug, Clone, Copy)]
struct Placed {
    node: usize,
    row: usize,
    col: usize,
    rowspan: usize,
    colspan: usize,
}

/// Map one `minmax()` side to a [`MinMaxBound`]. The CSS parser already
/// rejected invalid breadths (an `fr` minimum, nested `minmax()`), so
/// anything unrecognized here degrades to [`MinMaxBound::Content`] as
/// defense in depth for hand-built [`ComputedStyle`]s.
fn minmax_bound(v: &CssValue) -> MinMaxBound {
    match v {
        CssValue::Fr(f) if f.is_finite() => MinMaxBound::Fr(*f),
        CssValue::Fr(_) => MinMaxBound::Content,
        CssValue::Keyword(k)
            if k.eq_ignore_ascii_case("auto")
                || k.eq_ignore_ascii_case("min-content")
                || k.eq_ignore_ascii_case("max-content") =>
        {
            MinMaxBound::Content
        }
        v => match spec_len_of(Some(v)) {
            SpecLen::Auto => MinMaxBound::Content,
            s => MinMaxBound::Fixed(s),
        },
    }
}

/// Clamp a resolved track size into its `minmax()` bounds. When the
/// minimum exceeds the maximum the maximum is ignored and the minimum
/// wins (grid §7.2.4); `f32::clamp` would panic on an inverted range.
fn clamp_track(size: f32, min: f32, max: f32) -> f32 {
    size.max(min).min(max.max(min))
}

/// Resolve a `minmax()` minimum to px against `base`. Fixed
/// lengths/percentages resolve; `fr` (rejected by the parser; defensive)
/// and content-based minima floor at zero — the measured content need
/// still applies through the content-sizing pass.
fn minmax_floor(
    min: MinMaxBound,
    font_size: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    base: f32,
) -> f32 {
    match min {
        MinMaxBound::Fixed(s) => s
            .resolve(font_size, root_fs, vp_w, vp_h, base)
            .unwrap_or(0.0)
            .max(0.0),
        _ => 0.0,
    }
}

/// Resolve a `minmax()` maximum to px against `base`. Fixed
/// lengths/percentages resolve; `fr` and content-based maxima impose no
/// cap here (`fr` maxima share free space, content maxima size to
/// content).
fn minmax_ceil(
    max: MinMaxBound,
    font_size: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    base: f32,
) -> f32 {
    match max {
        MinMaxBound::Fixed(s) => s
            .resolve(font_size, root_fs, vp_w, vp_h, base)
            .unwrap_or(0.0)
            .max(0.0),
        _ => f32::INFINITY,
    }
}

/// Normalize one `minmax()` bound for a row track: a percentage bound
/// against an indefinite container height cannot resolve and degrades to
/// content-based, exactly like a percentage `Fixed` track degrading to
/// `auto` in [`resume_grid`]. A `calc()` only counts as percentage-like
/// when it syntactically contained a `%` leaf (`had_pct`, CSS Values 4
/// §8.1.4): a pure-length `calc(10px)` is a definite length and resolves.
fn normalize_row_bound(bound: MinMaxBound, hspec: Option<f32>) -> MinMaxBound {
    match bound {
        MinMaxBound::Fixed(s)
            if hspec.is_none()
                && matches!(s, SpecLen::Pct(_) | SpecLen::Math { had_pct: true, .. }) =>
        {
            MinMaxBound::Content
        }
        b => b,
    }
}

/// Normalize one `minmax()` bound for a column track: the column
/// counterpart of [`normalize_row_bound`]. The intrinsic passes have no
/// definite container width (`wspec` is always `None` there), so a
/// percentage bound against the indefinite axis cannot resolve and
/// degrades to content-based, exactly like a percentage `Fixed` track
/// degrading to `auto`. A `calc()` only counts as percentage-like when
/// it syntactically contained a `%` leaf (`had_pct`, CSS Values 4
/// §8.1.4): a pure-length `calc(10px + 5px)` is a definite length and
/// still resolves against the 0.0 base.
fn normalize_col_bound(bound: MinMaxBound, wspec: Option<f32>) -> MinMaxBound {
    match bound {
        MinMaxBound::Fixed(s)
            if wspec.is_none()
                && matches!(s, SpecLen::Pct(_) | SpecLen::Math { had_pct: true, .. }) =>
        {
            MinMaxBound::Content
        }
        b => b,
    }
}
/// Cap on `repeat(auto-fill|auto-fit, …)` expansion: mirrors the CSS
/// parser's cap on eager integer `repeat()` expansion (a hostile
/// hand-built style could otherwise derive a huge count from a tiny
/// track size and blow up the track vecs).
const MAX_AUTO_REPEAT: usize = 1000;

/// Definite size of one repeated track for auto-fill/fit counting, in
/// px: fixed lengths/percentages resolve against `base`; `minmax()` uses
/// its definite maximum, else its definite minimum (grid §7.2.3.2); `fr`,
/// `auto`, content-based sizes, and nested lists (flattened by the
/// caller) are indefinite (`None`).
fn repeat_stride_px(
    v: &CssValue,
    font_size: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    base: f32,
) -> Option<f32> {
    let resolve = |s: SpecLen| {
        s.resolve(font_size, root_fs, vp_w, vp_h, base)
            .filter(|px| px.is_finite() && *px >= 0.0)
    };
    let bound_px = |b: &CssValue| match minmax_bound(b) {
        MinMaxBound::Fixed(s) => resolve(s),
        _ => None,
    };
    match v {
        CssValue::MinMax(min, max) => bound_px(max).or_else(|| bound_px(min)),
        // `fr`, content keywords, and nested lists carry no definite size.
        CssValue::Fr(_) | CssValue::List(_) => None,
        CssValue::Keyword(k)
            if k.eq_ignore_ascii_case("auto")
                || k.eq_ignore_ascii_case("min-content")
                || k.eq_ignore_ascii_case("max-content") =>
        {
            None
        }
        v => match spec_len_of(Some(v)) {
            SpecLen::Auto => None,
            s => resolve(s),
        },
    }
}

/// Pure stride arithmetic for [`auto_repeat_count_for`]: how many
/// `track_px + gap` strides fit in `container + gap` (each extra track
/// adds one track and one gap), at least one. An indefinite container
/// (`None`) or a non-positive stride falls back to a single repetition.
fn auto_repeat_count(container: Option<f32>, track_px: f32, gap: f32) -> usize {
    let container = match container {
        Some(c) if c.is_finite() && c >= 0.0 => c,
        _ => return 1,
    };
    let stride = track_px + gap;
    if !stride.is_finite() || stride <= 0.0 {
        return 1;
    }
    (((container + gap) / stride).floor() as usize).max(1)
}

/// How many times `repeat(auto-fill|auto-fit, tracks)` repeats: the
/// largest N such that N repetitions of the *whole* track list fit in
/// the container (grid §7.2.3.2), at least one. With k definite tracks
/// summing to `sum_px` and gap `gap`, N repetitions occupy
/// `N * sum_px + (N * k - 1) * gap` px, so
/// `N = floor((base + gap) / (sum_px + k * gap))`. An indefinite track
/// size — or an indefinite container (`base` is `None`) — falls back to
/// a single repetition (documented simplification).
fn auto_repeat_count_for(
    tracks: &[CssValue],
    font_size: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    base: Option<f32>,
    gap: f32,
) -> usize {
    let base_px = match base {
        Some(b) if b.is_finite() && b >= 0.0 => b,
        _ => return 1,
    };
    // Iterative scan: the repeated list may nest one level (the parser
    // expands an inner integer `repeat()` to a nested list), and the
    // never-recurse-into-style-values invariant applies here too. Every
    // track must resolve to a definite size; the first indefinite one
    // bails to a single repetition.
    let mut sum_px = 0.0f32;
    let mut track_count = 0usize;
    let mut stack: Vec<&CssValue> = tracks.iter().rev().collect();
    while let Some(t) = stack.pop() {
        match t {
            CssValue::List(inner) => stack.extend(inner.iter().rev()),
            t => {
                track_count += 1;
                match repeat_stride_px(t, font_size, root_fs, vp_w, vp_h, base_px) {
                    Some(px) => sum_px += px,
                    None => return 1,
                }
            }
        }
    }
    if track_count == 0 {
        return 1;
    }
    // Per-repetition stride `sum_px + k * gap` against the container
    // extended by the one trailing gap that never materializes — the
    // same `(container + gap) / stride` shape as `auto_repeat_count`,
    // with the extra `k - 1` gaps folded into its `track_px` argument.
    let track_px = sum_px + (track_count as f32 - 1.0) * gap;
    auto_repeat_count(Some(base_px), track_px, gap)
}

/// Push one non-list track value. (The [`CssValue::List`] arms of
/// [`push_template_item`] flatten or expand before reaching here.)
fn push_track_value(
    out: &mut Vec<TrackSize>,
    collapse: &mut Vec<bool>,
    v: &CssValue,
    collapsible: bool,
) {
    let track = match v {
        CssValue::Fr(f) if f.is_finite() => TrackSize::Fr(*f),
        // A non-finite `fr` factor cannot come from the CSS parser
        // (it is rejected there); a hand-built `ComputedStyle`
        // carrying one would poison track sizing with NaN
        // (`free * inf / inf`), so it degrades to `auto` like any
        // other unrecognized track size.
        CssValue::Fr(_) => TrackSize::Auto,
        CssValue::Keyword(k) if k.eq_ignore_ascii_case("auto") => TrackSize::Auto,
        // `minmax()`: each side resolves independently; the track
        // sizing pass clamps the base size into the range. Nested
        // `minmax()` inside `repeat()` arrives here the same way
        // (repeat expansion is just a nested list).
        CssValue::MinMax(min, max) => TrackSize::MinMax {
            min: minmax_bound(min),
            max: minmax_bound(max),
        },
        // Anything else that resolves to a length/percentage is a
        // fixed track; anything unrecognized (e.g. `max-content`)
        // degrades to `auto`.
        v => match spec_len_of(Some(v)) {
            SpecLen::Auto => TrackSize::Auto,
            s => TrackSize::Fixed(s),
        },
    };
    out.push(track);
    collapse.push(collapsible);
}

/// Push one track-listing item: `repeat(auto-fill|auto-fit, …)`
/// descriptors expand to the computed repetition count (each expanded
/// track is marked collapsible for `auto-fit`); other nested lists
/// flatten; single values push one track. Iterative, per the crate's
/// never-recurse-into-style-values invariant (deeply nested lists
/// from hostile stylesheets overflow the stack). The stack carries each
/// value's collapsibility alongside it so the two can never desync,
/// however hostile the nesting.
#[allow(clippy::too_many_arguments)]
fn push_template_item(
    out: &mut Vec<TrackSize>,
    collapse: &mut Vec<bool>,
    v: &CssValue,
    font_size: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    base: Option<f32>,
    gap: f32,
) {
    let mut stack: Vec<(&CssValue, bool)> = vec![(v, false)];
    while let Some((v, collapsible)) = stack.pop() {
        // `repeat(auto-fill|auto-fit, …)` descriptor (parser contract,
        // see the module docs): expand here, where the axis size is
        // known. The count is capped like the parser's eager integer
        // repeats so a hostile hand-built style cannot force a huge
        // track vec. The descriptor is a dedicated `CssValue::Repeat`
        // variant — never confused with a bare `auto-fill` keyword in
        // the listing, which the CSS grammar rejects (and which degrades
        // to `auto` below like any unknown keyword).
        if let CssValue::Repeat { mode, tracks } = v {
            let n = auto_repeat_count_for(tracks, font_size, root_fs, vp_w, vp_h, base, gap)
                .min(MAX_AUTO_REPEAT);
            let fit = *mode == RepeatMode::Fit;
            for _ in 0..n {
                // Reversed: the stack is LIFO, so items are visited in
                // source order exactly like the recursive version.
                stack.extend(tracks.iter().rev().map(|t| (t, fit)));
            }
            continue;
        }
        match v {
            CssValue::List(items) => {
                stack.extend(items.iter().rev().map(|t| (t, collapsible)));
            }
            v => push_track_value(out, collapse, v, collapsible),
        }
    }
}

/// Parse a track listing (`grid-template-columns` / `grid-template-rows`).
/// `none` (the initial value) and anything unparsable yield no tracks;
/// nested lists from `repeat()` expansion are flattened iteratively.
///
/// Returns the tracks plus, per track, whether it collapses when empty:
/// only tracks expanded from `repeat(auto-fit, …)` are collapsible.
/// `base` is the definite axis size used to count `auto-fill`/`auto-fit`
/// repetitions (`None` = indefinite: a single repetition); `gap` is the
/// relevant gap, counted between repetitions.
#[allow(clippy::too_many_arguments)]
fn parse_tracks(
    style: Option<&ComputedStyle>,
    prop: u8,
    font_size: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    base: Option<f32>,
    gap: f32,
) -> (Vec<TrackSize>, Vec<bool>) {
    let value = match style.and_then(|s| s.get_by_index(prop)) {
        None => return (Vec::new(), Vec::new()),
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("none") => {
            return (Vec::new(), Vec::new());
        }
        Some(v) => v,
    };
    // The whole value goes through `push_template_item` in one call: its
    // stack flattens plain lists element by element, and it matches the
    // `repeat(auto-fill|auto-fit, …)` descriptor (`CssValue::Repeat`)
    // before flattening. Pushing the list's elements one by one instead
    // would split a bare `repeat(auto-fill, …)` declaration into a bogus
    // `auto` track plus the unexpanded track list, since the descriptor
    // *is* the whole value in that case.
    let mut out = Vec::new();
    let mut collapse = Vec::new();
    push_template_item(
        &mut out,
        &mut collapse,
        value,
        font_size,
        root_fs,
        vp_w,
        vp_h,
        base,
        gap,
    );
    (out, collapse)
}

/// Whether the node's own `prop` listing is the `subgrid` keyword: the
/// item wants its parent's track list for that axis.
fn is_subgrid(style: Option<&ComputedStyle>, prop: u8) -> bool {
    matches!(style.and_then(|s| s.get_by_index(prop)), Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("subgrid"))
}

/// Whether the node is a grid container (`display: grid` or the
/// inline-level `display: inline-grid`).
fn is_grid_container(forest: &Forest, idx: usize) -> bool {
    matches!(
        forest.nodes[idx].data.display,
        crate::style::Display::Grid | crate::style::Display::InlineGrid
    )
}

/// The grid ancestor a subgrid inherits its track list for `prop` from
/// (css-grid-1 §2.3). Nested subgrids defer to their own parent, so the
/// walk skips ancestors whose listing is itself the `subgrid` keyword;
/// the first other ancestor decides: a grid container with a real
/// listing is the source, anything else (a non-grid box, or a grid with
/// no listing, i.e. `none`) means the axis behaves as `none`.
fn subgrid_source(forest: &Forest, idx: usize, prop: u8) -> Option<usize> {
    let mut cur = forest.nodes[idx].parent;
    while let Some(p) = cur {
        let style = forest.nodes[p].style.as_deref();
        if is_subgrid(style, prop) {
            cur = forest.nodes[p].parent;
            continue;
        }
        // A real listing is any computed value other than the initial
        // `none`: a keyword like `subgrid` is handled above, a length
        // list (e.g. `100px 100px`) is a listing.
        let has_listing = match style.and_then(|s| s.get_by_index(prop)) {
            None => false,
            Some(CssValue::Keyword(k)) => !k.eq_ignore_ascii_case("none"),
            Some(_) => true,
        };
        if is_grid_container(forest, p) && has_listing {
            return Some(p);
        }
        return None;
    }
    None
}

/// The pending phase-one [`Task::Grid`] frame for grid container
/// `parent`, if it is still on the task stack. Grid items (including
/// nested grids) complete before their container's resume pops, so a
/// subgrid laying out can read its parent's phase-one data — used
/// column geometry, placement, and track listings — without new node
/// fields. Returns `None` during intrinsic sizing, which never runs the
/// track passes.
fn parent_grid_frame(tasks: &[Task], parent: usize) -> Option<&GridResume> {
    tasks.iter().rev().find_map(|t| match t {
        Task::Grid(fr) if fr.idx == parent => Some(&**fr),
        _ => None,
    })
}

/// The parent grid's used column geometry (positions, sizes) for a
/// subgrid item. The parent's phase-one resume frame is still pending
/// on the task stack while its items lay out — grid items (including
/// nested grids) complete before it pops — so the used geometry is
/// available here without new node fields. Returns `(col_x, col_w)`
/// (positions, then sizes), or `None` when no pending frame for
/// `parent` is on the stack (e.g. intrinsic sizing, which never runs
/// the track passes).
fn parent_used_columns(tasks: &[Task], parent: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    parent_grid_frame(tasks, parent).map(|fr| (fr.col_x.clone(), fr.col_w.clone()))
}

/// The index range `[s, e)` of parent tracks contained in the interval
/// `[x0, x1]`: the subset of the parent's tracks the subgrid spans. The
/// parent sizes the subgrid item's containing block to exactly the
/// spanned tracks (positions included, so distributed content-alignment
/// spacing counts), making the containment exact up to float rounding.
fn subgrid_track_subset(col_x: &[f32], col_w: &[f32], x0: f32, x1: f32) -> Option<(usize, usize)> {
    let eps = 0.001;
    let mut start = None;
    let mut end = 0;
    for (i, (&px, &pw)) in col_x.iter().zip(col_w.iter()).enumerate() {
        if px >= x0 - eps && px + pw <= x1 + eps {
            if start.is_none() {
                start = Some(i);
            }
            end = i + 1;
        } else if start.is_some() {
            break;
        }
    }
    start.map(|s| (s, end))
}

/// The exact used size of a parent row track whose size is fully
/// determined by the track listing — no content measurement involved —
/// or `None` when the track sizes from content or shares free space.
/// Fixed lengths resolve against the parent's font metrics and definite
/// content height (a percentage track against an indefinite height
/// degrades to content-sized, like in [`resume_grid`]); a `minmax()`
/// whose fixed bounds resolve to the same size clamps to that size
/// whatever the content measures. `fr`, `auto`, and content-bounded
/// tracks are never definite: their used sizes need the parent's items,
/// which have not all laid out when a subgrid resolves its rows.
fn definite_row_track_size(
    track: &TrackSize,
    font_size: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    hspec: Option<f32>,
) -> Option<f32> {
    // Resolve one fixed bound; `None` when it needs an indefinite
    // percentage base or is not a fixed length at all.
    fn bound(
        b: MinMaxBound,
        font_size: f32,
        root_fs: f32,
        vp_w: f32,
        vp_h: f32,
        hspec: Option<f32>,
    ) -> Option<f32> {
        match b {
            // A `calc()` bound only needs an indefinite percentage base
            // when it syntactically contained a `%` leaf (`had_pct`,
            // CSS Values 4 §8.1.4); a pure-length `calc()` resolves like
            // any other fixed length.
            MinMaxBound::Fixed(s)
                if hspec.is_some()
                    || !matches!(s, SpecLen::Pct(_) | SpecLen::Math { had_pct: true, .. }) =>
            {
                s.resolve(font_size, root_fs, vp_w, vp_h, hspec.unwrap_or(0.0))
                    .map(|v| v.max(0.0))
            }
            _ => None,
        }
    }
    match *track {
        TrackSize::Fixed(s) => bound(MinMaxBound::Fixed(s), font_size, root_fs, vp_w, vp_h, hspec),
        TrackSize::MinMax { min, max } => {
            let lo = bound(min, font_size, root_fs, vp_w, vp_h, hspec)?;
            let hi = bound(max, font_size, root_fs, vp_w, vp_h, hspec)?;
            (lo == hi).then_some(lo)
        }
        TrackSize::Fr(_) | TrackSize::Auto => None,
    }
}

/// The spanned subset of the parent grid's row listing for a
/// `grid-template-rows: subgrid` container, with definite tracks frozen
/// at their exact used sizes (see [`definite_row_track_size`]), plus
/// the parent's row gap — the subgrid's own row gap does not apply to
/// the inherited axis. The subgrid's placement in the parent comes from
/// the parent's pending phase-one frame, so nested subgrids resolve
/// against their direct parent's already-subset listing and definite
/// sizes chain upward exactly. Returns `None` when the parent grid has
/// no pending phase-one frame (intrinsic sizing) or this subgrid's
/// placement in it cannot be found; the caller then keeps the
/// parsed-listing fallback.
fn subgrid_row_subset(
    forest: &Forest,
    tasks: &[Task],
    idx: usize,
) -> Option<(Vec<TrackSize>, f32)> {
    let parent = forest.nodes[idx]
        .parent
        .filter(|&p| is_grid_container(forest, p))?;
    let fr = parent_grid_frame(tasks, parent)?;
    let me = fr.items.iter().find(|it| it.node == idx)?;
    let end = (me.row + me.rowspan).min(fr.row_template.len());
    let start = me.row.min(end);
    if start == end {
        return None;
    }
    let (root_fs, vp_w, vp_h) = (forest.root_font_size, forest.vp_w, forest.vp_h);
    let mut out = Vec::with_capacity(end - start);
    for t in &fr.row_template[start..end] {
        match definite_row_track_size(t, fr.font_size, root_fs, vp_w, vp_h, fr.hspec) {
            Some(px) => out.push(TrackSize::Fixed(SpecLen::Px(px))),
            None => out.push(*t),
        }
    }
    Some((out, fr.row_gap))
}

/// Parse one axis' track listing for grid container `idx`, honoring
/// `subgrid`: when the container's own listing is the `subgrid`
/// keyword, the parent grid's listing for that axis is parsed and
/// reused instead (grid §2.3). The ancestor's font size, used content
/// size, and gap feed the parse so `repeat(auto-fill, …)` counts like
/// the ancestor's own listing; the ancestor's specified height feeds
/// the row axis (percentages still degrade to content-sized without a
/// definite height, like the container's own). With no grid ancestor
/// carrying a real listing the axis degrades to the old single-`auto`
/// behavior. This is the track *listing* only: on the column axis
/// [`layout_grid`] additionally resolves the subgrid against the parent
/// grid's used track sizes/positions for the spanned subset (see
/// [`parent_used_columns`]); on the row axis [`layout_grid`] freezes
/// the definite tracks of the spanned subset at their used sizes (see
/// [`subgrid_row_subset`]) — the parent's used row geometry still does
/// not exist when the subgrid lays out, so indefinite row tracks keep
/// this parsed listing and size from the subgrid's own content. The
/// subgrid keeps its own auto-flow; its own gaps and content alignment
/// do not apply to an inherited axis.
///
/// `intrinsic` marks parses serving intrinsic sizing (`grid_pref_width`
/// / `grid_min_width`): the parent axis is indefinite there — the
/// parent's used content width does not exist yet (its metrics are
/// stale or zero) — so the inherited column listing resolves against
/// no base: percentage track breadths degrade to content-sized,
/// `repeat(auto-fill|auto-fit)` collapses to a single repetition, and
/// percentage gaps resolve against zero, exactly like the container's
/// own listing under a `None` base. Layout passes `false` and keeps
/// the parent's real content width.
#[allow(clippy::too_many_arguments)]
fn parse_axis_tracks(
    forest: &Forest,
    idx: usize,
    prop: u8,
    font_size: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    base: Option<f32>,
    gap: f32,
    intrinsic: bool,
) -> (Vec<TrackSize>, Vec<bool>) {
    let style = forest.nodes[idx].style.as_deref();
    if !is_subgrid(style, prop) {
        return parse_tracks(style, prop, font_size, root_fs, vp_w, vp_h, base, gap);
    }
    let Some(p) = subgrid_source(forest, idx, prop) else {
        return (Vec::new(), Vec::new());
    };
    let pstyle = forest.nodes[p].style.as_deref();
    let pfs = forest.nodes[p].data.font_size;
    // Intrinsic sizing measures against an indefinite parent axis (see
    // above); layout resolves against the parent's used content width.
    let pw = if intrinsic {
        0.0
    } else {
        forest.nodes[p].metrics.content_w
    };
    if prop == props::GRID_TEMPLATE_COLUMNS {
        let (_, pgap) = parse_gaps(pstyle, pfs, root_fs, vp_w, vp_h, pw);
        let pbase = if intrinsic { None } else { Some(pw) };
        parse_tracks(pstyle, prop, pfs, root_fs, vp_w, vp_h, pbase, pgap)
    } else {
        // The ancestor's containing-block height is not threaded here,
        // so percentage row tracks resolve against an indefinite height
        // (degrading to content-sized); every other breadth reuses the
        // ancestor's listing exactly.
        let phspec = specified_height(forest, p, None);
        let (pgap, _) = parse_gaps(pstyle, pfs, root_fs, vp_w, vp_h, pw);
        parse_tracks(pstyle, prop, pfs, root_fs, vp_w, vp_h, phspec, pgap)
    }
}

/// The `order` used for grid auto-placement: the computed `order`
/// value, saturating into `i32` range (`auto`/unparseable values sort as
/// 0, the initial value). Sorting is stable, so equal orders keep DOM
/// order. Painting order is not re-sorted (out of scope for this pass).
fn order_of(style: Option<&ComputedStyle>) -> i32 {
    match style.and_then(|s| s.get_by_index(props::ORDER)) {
        // `order` is an <integer> with no spec bound (css-grid-1 §4.2):
        // saturate through `i64` instead of clamping to an arbitrary
        // range, so `order: 2000000` still sorts before `order: 3000000`.
        // Float-to-int `as` casts saturate on overflow; NaN becomes 0.
        Some(CssValue::Number(n)) => {
            let v = *n as i64;
            i32::try_from(v).unwrap_or(if v < 0 { i32::MIN } else { i32::MAX })
        }
        _ => 0,
    }
}

/// Extend an explicit track listing with implicit tracks sized from the
/// `grid-auto-columns` / `grid-auto-rows` listing: implicit track `i`
/// (zero-based past the explicit tracks) takes listing entry
/// `i % len`. An empty auto listing means `auto` (the old behavior).
fn implicit_tracks(template: &[TrackSize], auto: &[TrackSize], n: usize) -> Vec<TrackSize> {
    let mut out = Vec::with_capacity(n.max(template.len()));
    out.extend_from_slice(template);
    if auto.is_empty() {
        out.resize(n, TrackSize::Auto);
    } else {
        let mut i = 0;
        while out.len() < n {
            out.push(auto[i % auto.len()]);
            i += 1;
        }
    }
    out
}

/// Resolve the row/column gap pair from the `row-gap` / `column-gap`
/// longhands (`gap` / `grid-gap` expand into them at parse time).
/// Percentage gaps resolve against the content width and clamp to zero.
fn parse_gaps(
    style: Option<&ComputedStyle>,
    font_size: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    content_w: f32,
) -> (f32, f32) {
    let len = |v: Option<&CssValue>| {
        spec_len_of(v)
            .resolve(font_size, root_fs, vp_w, vp_h, content_w)
            .unwrap_or(0.0)
            .max(0.0)
    };
    let row_gap = len(style.and_then(|s| s.get_by_index(props::ROW_GAP)));
    let col_gap = len(style.and_then(|s| s.get_by_index(props::COLUMN_GAP)));
    (row_gap, col_gap)
}

/// Parse one side of a `grid-row` / `grid-column` value. The cascade
/// already rejects non-integer, zero, and out-of-range line numbers; the
/// checks here are defense in depth so a hand-built [`ComputedStyle`]
/// (or any future caller) cannot force a huge placement grid: invalid
/// lines degrade to `auto` and valid ones clamp to
/// [`MAX_GRID_LINE`].
fn parse_line_side(v: &CssValue) -> GridLine {
    match v {
        CssValue::Keyword(k) if k.eq_ignore_ascii_case("auto") => GridLine::Auto,
        CssValue::Number(n) => {
            if n.fract() != 0.0 || *n == 0.0 {
                GridLine::Auto
            } else {
                GridLine::Line(n.clamp(-(MAX_GRID_LINE as f32), MAX_GRID_LINE as f32) as i32)
            }
        }
        CssValue::List(items) => match items.as_slice() {
            [CssValue::Keyword(k), CssValue::Number(n)] if k.eq_ignore_ascii_case("span") => {
                if n.fract() != 0.0 || *n < 1.0 {
                    GridLine::Auto
                } else {
                    GridLine::Span((*n as u32).clamp(1, 1000))
                }
            }
            [single] => parse_line_side(single),
            _ => GridLine::Auto,
        },
        _ => GridLine::Auto,
    }
}

/// Parse a `grid-row` / `grid-column` value into its start/end sides.
/// The CSS parser stores the `/`-separated form as
/// `List([List(before), List(after)])`; a bare value only sets the
/// start side.
fn parse_placement(style: Option<&ComputedStyle>, prop: u8) -> GridPlacement {
    let (start, end) = match style.and_then(|s| s.get_by_index(prop)) {
        Some(CssValue::List(items)) if items.iter().all(|v| matches!(v, CssValue::List(_))) => {
            match items.as_slice() {
                [a, b] => (parse_line_side(a), parse_line_side(b)),
                _ => (GridLine::Auto, GridLine::Auto),
            }
        }
        Some(v) => (parse_line_side(v), GridLine::Auto),
        None => (GridLine::Auto, GridLine::Auto),
    };
    GridPlacement { start, end }
}

/// `justify-items` / `align-items`: `normal` (the initial value)
/// behaves as `stretch`; unrecognized values fall back to `stretch`.
/// `baseline` has no baseline-alignment engine behind it yet, so it
/// stretches as an approximation — not a spec fallback (css-align-3
/// falls back to `start` when baseline alignment is not supported),
/// just the least surprising placeholder until a baseline engine
/// lands.
pub(crate) fn item_align_of(style: Option<&ComputedStyle>, prop: u8) -> ItemAlign {
    match style.and_then(|s| s.get_str_by_index(prop)) {
        Some(v)
            if v.eq_ignore_ascii_case("start")
                || v.eq_ignore_ascii_case("flex-start")
                || v.eq_ignore_ascii_case("self-start") =>
        {
            ItemAlign::Start
        }
        Some(v)
            if v.eq_ignore_ascii_case("end")
                || v.eq_ignore_ascii_case("flex-end")
                || v.eq_ignore_ascii_case("self-end") =>
        {
            ItemAlign::End
        }
        Some(v) if v.eq_ignore_ascii_case("center") => ItemAlign::Center,
        _ => ItemAlign::Stretch,
    }
}

/// Per-item `justify-self` / `align-self`: `auto` (the initial value)
/// defers to the container's `justify-items` / `align-items`
/// (`fallback`); `normal` behaves as `stretch`; anything unrecognized
/// falls back to the container value. `baseline` stretches (same
/// missing-baseline-engine approximation as `item_align_of`): it must
/// not inherit the container's alignment, which would silently change
/// the item's axis position.
pub(crate) fn self_align_of(
    style: Option<&ComputedStyle>,
    prop: u8,
    fallback: ItemAlign,
) -> ItemAlign {
    // A `safe`/`unsafe`-prefixed value keeps the `<self-position>` keyword
    // as its second list item (see `split_place_self` in bowser_css); the
    // safe/unsafe overflow refinement is not applied here — the positional
    // keyword is used as-is (documented approximation; `unsafe` is exact).
    let kw: Option<&str> = match style.and_then(|s| s.get_by_index(prop)) {
        Some(CssValue::List(items)) if items.len() == 2 => match &items[1] {
            CssValue::Keyword(k) => Some(k.as_str()),
            _ => None,
        },
        Some(CssValue::Keyword(k)) => Some(k.as_str()),
        _ => None,
    };
    match kw {
        None => fallback,
        Some(v) if v.eq_ignore_ascii_case("auto") => fallback,
        Some(v)
            if v.eq_ignore_ascii_case("start")
                || v.eq_ignore_ascii_case("flex-start")
                || v.eq_ignore_ascii_case("self-start") =>
        {
            ItemAlign::Start
        }
        Some(v)
            if v.eq_ignore_ascii_case("end")
                || v.eq_ignore_ascii_case("flex-end")
                || v.eq_ignore_ascii_case("self-end") =>
        {
            ItemAlign::End
        }
        Some(v) if v.eq_ignore_ascii_case("center") => ItemAlign::Center,
        Some(v)
            if v.eq_ignore_ascii_case("stretch")
                || v.eq_ignore_ascii_case("normal")
                || v.eq_ignore_ascii_case("baseline") =>
        {
            ItemAlign::Stretch
        }
        _ => fallback,
    }
}

/// Whether `grid-auto-flow` requests dense packing: the value contains
/// the `dense` keyword (`dense`, `row dense`, `dense column`, …),
/// case-insensitive. Anything else — `row`, `column`, `sparse`, or an
/// absent property — means sparse flow, the default.
///
/// `grid-auto-flow` is in the cascade's `COMPUTED_PROPS` (initial
/// `row`), so a stylesheet declaration reaches layout; the caller reads
/// the computed value off the grid container's style.
fn dense_flow(value: Option<&str>) -> bool {
    value.is_some_and(|v| {
        v.split_whitespace()
            .any(|w| w.eq_ignore_ascii_case("dense"))
    })
}

/// Whether `grid-auto-flow` requests column-major placement: the value
/// contains the `column` keyword (`column`, `column dense`, …),
/// case-insensitive. Anything else — `row`, `dense`, `sparse`, or an
/// absent property — means row-major placement, the default.
fn column_flow(value: Option<&str>) -> bool {
    value.is_some_and(|v| {
        v.split_whitespace()
            .any(|w| w.eq_ignore_ascii_case("column"))
    })
}

/// Content distribution along one axis (`justify-content` /
/// `align-content`): how leftover space (the container's content size
/// minus tracks and gutters) is distributed after track sizing, the
/// grid analogue of flex's `distribute_lines`. `normal` (the initial
/// value) behaves as `stretch`: content-sized tracks that took no `fr`
/// share grow equally to absorb the space, still respecting `minmax()`
/// maxima (css-grid-1 §10.4/§11.2, css-align-3 §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContentAlign {
    Start,
    End,
    Center,
    Between,
    Around,
    Evenly,
    Stretch,
}

/// Parse `justify-content` / `align-content` into a [`ContentAlign`].
/// The initial `normal` (explicit or absent) behaves as `stretch`;
/// anything unrecognized falls back to it too (the cascade drops
/// invalid values, so an unrecognized computed value is just the
/// initial value by another name).
fn content_align_of(style: Option<&ComputedStyle>, prop: u8) -> ContentAlign {
    match style.and_then(|s| s.get_str_by_index(prop)) {
        Some(v)
            if v.eq_ignore_ascii_case("end")
                || v.eq_ignore_ascii_case("flex-end")
                || v.eq_ignore_ascii_case("right") =>
        {
            ContentAlign::End
        }
        Some(v)
            if v.eq_ignore_ascii_case("start")
                || v.eq_ignore_ascii_case("flex-start")
                || v.eq_ignore_ascii_case("left") =>
        {
            ContentAlign::Start
        }
        Some(v) if v.eq_ignore_ascii_case("center") => ContentAlign::Center,
        Some(v) if v.eq_ignore_ascii_case("space-between") => ContentAlign::Between,
        Some(v) if v.eq_ignore_ascii_case("space-around") => ContentAlign::Around,
        Some(v) if v.eq_ignore_ascii_case("space-evenly") => ContentAlign::Evenly,
        _ => ContentAlign::Stretch,
    }
}

/// Track positions along one axis starting at `origin`, distributing
/// leftover space (`total` minus tracks and gutters) per `align` after
/// sizing — the grid analogue of flex's `distribute_lines`. `total` is
/// the definite content size (`None` = indefinite: pack from the start
/// edge, so `align-content` is a no-op on auto-height containers).
/// Collapsed `auto-fit` tracks take no space and draw no gap. With
/// [`ContentAlign::Start`] (or no free space) this reproduces the old
/// pack-from-the-start positions exactly.
///
/// `stretch` carries the per-track data used by [`ContentAlign::Stretch`]:
/// `(mask, max)`. Tracks with `mask` set (content-sized tracks that took
/// no `fr` share) grow equally to absorb the free space, each capped at
/// its `max` (`minmax()` maxima, `f32::INFINITY` when unconstrained);
/// leftover from capped tracks redistributes over the rest. Growth is
/// applied in place to `sizes`, so callers that derive areas or sums from
/// the sizes afterwards see the stretched values.
fn layout_track_positions(
    sizes: &mut [f32],
    collapsed: &[bool],
    origin: f32,
    total: Option<f32>,
    gap: f32,
    align: ContentAlign,
    stretch: Option<(&[bool], &[f32])>,
) -> Vec<f32> {
    let live = sizes
        .iter()
        .enumerate()
        .filter(|(i, _)| !collapsed.get(*i).copied().unwrap_or(false))
        .count();
    // Gutters only separate tracks that take space, exactly like the
    // free-space computation in the track sizing passes.
    let gaps = gap * (live as f32 - 1.0).max(0.0);
    let used: f32 = sizes.iter().sum::<f32>() + gaps;
    let free = total.map(|t| (t - used).max(0.0)).unwrap_or(0.0);
    // `normal`/`stretch`: grow the stretchable tracks equally to absorb
    // the free space, respecting each track's maximum; a track that hits
    // its cap drops out and the remainder redistributes.
    if align == ContentAlign::Stretch
        && let Some((mask, max)) = stretch
    {
        let stretchable = |i: usize, s: f32| {
            mask.get(i).copied().unwrap_or(false)
                && !collapsed.get(i).copied().unwrap_or(false)
                && s < max.get(i).copied().unwrap_or(f32::INFINITY)
        };
        let mut remaining = free;
        while remaining > 0.0 {
            let count = sizes
                .iter()
                .enumerate()
                .filter(|(i, s)| stretchable(*i, **s))
                .count();
            if count == 0 {
                break;
            }
            let share = remaining / count as f32;
            let mut capped = false;
            let mut consumed = 0.0f32;
            for (i, s) in sizes.iter_mut().enumerate() {
                if !stretchable(i, *s) {
                    continue;
                }
                let room = (max.get(i).copied().unwrap_or(f32::INFINITY) - *s).max(0.0);
                let add = share.min(room);
                if add < share {
                    capped = true;
                }
                *s += add;
                consumed += add;
            }
            remaining -= consumed;
            if !capped {
                break;
            }
        }
    }
    // (start cursor, extra gap between live tracks), mirroring
    // `distribute_lines` in the flex module.
    let (mut cursor, mut extra) = (0.0f32, 0.0f32);
    if live > 0 && free > 0.0 {
        match align {
            ContentAlign::End => cursor = free,
            ContentAlign::Center => cursor = free / 2.0,
            ContentAlign::Between if live > 1 => extra = free / (live - 1) as f32,
            ContentAlign::Around => {
                let g = free / live as f32;
                cursor = g / 2.0;
                extra = g;
            }
            ContentAlign::Evenly => {
                let g = free / (live + 1) as f32;
                cursor = g;
                extra = g;
            }
            // `Start`, `Stretch` (growth already applied above), and
            // `Between` with a single live track: pack from the start
            // edge (or no free space at all).
            _ => {}
        }
    }
    let mut pos = Vec::with_capacity(sizes.len());
    let mut x = origin + cursor;
    for (i, &w) in sizes.iter().enumerate() {
        pos.push(x);
        x += w;
        // A collapsed `auto-fit` track draws no gap: gutters only
        // separate tracks that take space.
        if !collapsed.get(i).copied().unwrap_or(false) {
            x += gap + extra;
        }
    }
    pos
}

/// Wrap each maximal run of non-block-level children of the grid
/// container `idx` in an anonymous block box (grid §5.1: each in-flow
/// child becomes a grid item, and a run of text/inline children forms
/// one anonymous item). Idempotent: wrappers are block-level, so a
/// second call finds no raw runs.
fn wrap_grid_text_runs(forest: &mut Forest, idx: usize) {
    // Scan before allocating: when every child is block-level the
    // wrapping is a no-op, and that is the common case (idempotence
    // means the second call on the same container always lands here).
    // Index iteration instead of `children.clone()`: nothing in the scan
    // or the wrap below mutates `nodes[idx].children` itself (anonymous
    // wrappers are appended as fresh nodes and wired in by the final
    // assignment), so reading each child by index is the same snapshot
    // without the O(kids) allocation.
    let nkids = forest.nodes[idx].children.len();
    if (0..nkids).all(|i| {
        let c = forest.nodes[idx].children[i];
        is_block_level(forest.nodes[c].box_type)
    }) {
        return;
    }
    let parent_data = forest.nodes[idx].data.clone();
    let mut roots: Vec<usize> = Vec::with_capacity(nkids);
    let mut run: Vec<usize> = Vec::new();
    for i in 0..nkids {
        let c = forest.nodes[idx].children[i];
        if is_block_level(forest.nodes[c].box_type) || layout_position::is_out_of_flow(forest, c) {
            if !run.is_empty() {
                roots.push(forest.make_anonymous(
                    &parent_data,
                    std::mem::take(&mut run),
                    idx,
                    true,
                ));
            }
            roots.push(c);
        } else {
            run.push(c);
        }
    }
    if !run.is_empty() {
        roots.push(forest.make_anonymous(&parent_data, run, idx, true));
    }
    forest.nodes[idx].children = roots;
}

/// Whether the node is whitespace-only text (directly or through
/// anonymous wrappers); such runs are not grid items.
///
/// Iterative: the recursive version violated the crate's
/// never-recurse-into-the-box-tree invariant (deep anonymous wrapper
/// chains from hostile documents overflow the stack).
fn is_ws_only(forest: &Forest, idx: usize) -> bool {
    let mut stack = vec![idx];
    while let Some(i) = stack.pop() {
        match forest.nodes[i].box_type {
            BoxType::Text => {
                if !forest.nodes[i]
                    .text
                    .as_deref()
                    .is_some_and(|t| t.chars().all(char::is_whitespace))
                {
                    return false;
                }
            }
            BoxType::AnonymousBlock => {
                stack.extend(forest.nodes[i].children.iter().copied());
            }
            _ => return false,
        }
    }
    true
}

/// The span a placement contributes when it is not fully definite
/// (css-grid-1 §8.3): a lone `span <n>` on either side auto-places with
/// that span (`grid-row: span 2`, `grid-row: auto / span 2`); two spans
/// keep the start's, the end's is ignored.
fn span_of(place: GridPlacement) -> usize {
    match (place.start, place.end) {
        (GridLine::Span(s), _) | (_, GridLine::Span(s)) => (s as usize).clamp(1, MAX_DEFINITE_SPAN),
        _ => 1,
    }
}

/// Resolve a definite placement into a zero-based start index and span
/// along one axis with `explicit` tracks. Returns `None` when the
/// placement is not fully definite (auto-placement must fill it in;
/// see [`span_of`] for the span such a placement still contributes).
/// Spans are capped at [`MAX_DEFINITE_SPAN`]: [`place_items`] marks every
/// covered cell in its sparse occupancy set, so an uncapped span is a
/// direct memory blowup (`grid-row: 1/1000000` would insert ~1e12 keys).
/// Far start lines are harmless on their own: only covered cells cost
/// memory, never the empty space before them.
const MAX_DEFINITE_SPAN: usize = 1000;

fn definite_range(place: GridPlacement, explicit: usize) -> Option<(usize, usize)> {
    // Negative lines count back from the explicit end: line `-n` is the
    // line after the `(explicit + 1 - n)`-th track, so `-1` is the line
    // after the last explicit track (zero-based index `explicit`).
    // `unsigned_abs` is used instead of negation so `i32::MIN` cannot
    // overflow; saturating arithmetic keeps huge magnitudes in bounds
    // (line zero, rejected by the parser, keeps the old clamp).
    let from_end = |l: i32| {
        explicit
            .saturating_add(1)
            .saturating_sub((l.unsigned_abs() as usize).max(1))
    };
    match (place.start, place.end) {
        (GridLine::Line(a), GridLine::Line(b)) => {
            // Each side resolves independently (negative lines count
            // back from the explicit end); the range is then ordered so
            // a reversed placement still covers its tracks
            // (`grid-column: 1 / -1` spans the whole explicit grid).
            // Lines before the grid clamp to zero (implicit tracks
            // before the explicit grid are not supported).
            let ra = if a >= 1 {
                (a - 1) as usize
            } else {
                from_end(a)
            };
            let rb = if b >= 1 {
                (b - 1) as usize
            } else {
                from_end(b)
            };
            let (start, end) = if ra <= rb { (ra, rb) } else { (rb, ra) };
            let span = end.saturating_sub(start).clamp(1, MAX_DEFINITE_SPAN);
            Some((start, span))
        }
        (GridLine::Line(a), GridLine::Span(s)) => {
            let start = if a >= 1 {
                (a - 1) as usize
            } else {
                from_end(a)
            };
            Some((start, (s as usize).clamp(1, MAX_DEFINITE_SPAN)))
        }
        (GridLine::Span(s), GridLine::Line(b)) => {
            let end = if b >= 1 {
                (b - 1) as usize
            } else {
                from_end(b)
            };
            let s = (s as usize).clamp(1, MAX_DEFINITE_SPAN);
            Some((end.saturating_sub(s), s))
        }
        // A single line (`grid-column: 3`, i.e. `3 / auto`) pins the
        // start track with span 1.
        (GridLine::Line(a), GridLine::Auto) => {
            let start = if a >= 1 {
                (a - 1) as usize
            } else {
                from_end(a)
            };
            Some((start, 1))
        }
        _ => None,
    }
}

/// Row-major auto-placement (css-grid-1 §8.5). Items with a definite
/// column range keep their columns; items with a definite row range
/// keep their rows; everything else fills the first free rectangle at
/// or after the cursor. Fully definite items never move the
/// auto-placement cursor (spec step 1 positions them before the cursor
/// exists); definite-row items scan their row from the cursor column
/// when the cursor is on that row. The grid grows implicit
/// rows/columns as needed. Definite placements that collide are kept
/// anyway (items may overlap).
///
/// `dense` selects `grid-auto-flow: dense` packing (grid §8.5): the
/// cursor resets to the grid start after each item, so auto-positioned
/// items backfill holes left by definite-positioned or spanned items
/// instead of only ever moving forward. Sparse flow (the default) never
/// backtracks, keeping auto-placed items in order.
///
/// Occupancy is a sparse set of packed `(row, col)` keys rather than a
/// dense grid: definite start lines reach ±10000 while spans cap at
/// 1000, so a dense `Vec<Vec<bool>>` allocated ~121MB for a single
/// `grid-row: 10000/10001; grid-column: 10000/10001` item. The set holds
/// at most the cells the placed items actually cover, so hostile start
/// lines cannot amplify a ~40-byte declaration into a huge allocation.
fn place_items_row_major(
    specs: &[ItemSpec],
    explicit_cols: usize,
    explicit_rows: usize,
    dense: bool,
) -> (Vec<Placed>, usize, usize) {
    fn occ_key(row: usize, col: usize) -> u64 {
        ((row as u64) << 32) | col as u64
    }
    let mut occupied: HashSet<u64> = HashSet::new();
    let mut placed = Vec::with_capacity(specs.len());
    // css-grid-1 §8.5 step 3: size the implicit grid's columns before
    // auto-placement. Definite column positions extend it, and the
    // largest column span among items without a definite column
    // position widens it, so a spanning auto item first tries to sit
    // beside earlier items instead of wrapping to a fresh row. The
    // no-explicit-columns case keeps its wrap-every-item behavior (one
    // implicit row per auto item), so the pre-growth only applies with
    // explicit columns.
    let mut flow_cols = explicit_cols.max(1);
    if explicit_cols > 0 {
        for spec in specs {
            match definite_range(spec.col, explicit_cols) {
                Some((c, s)) => flow_cols = flow_cols.max(c + s),
                None => flow_cols = flow_cols.max(span_of(spec.col)),
            }
        }
    }
    let mut ncols = flow_cols;
    let mut nrows = explicit_rows.max(1);
    // Auto-placement cursor (sparse: never moves backwards).
    let (mut cur_row, mut cur_col) = (0usize, 0usize);

    for spec in specs {
        let col_def = definite_range(spec.col, explicit_cols);
        let row_def = definite_range(spec.row, explicit_rows);
        // A span-only placement (`grid-row: span 2`) is not definite,
        // but still contributes its span to auto-placement (css-grid-1
        // §8.3, §8.5); the occupancy marks below then advance the
        // cursor past the spanned rows/columns.
        let colspan = col_def.map(|(_, s)| s).unwrap_or_else(|| span_of(spec.col));
        let rowspan = row_def.map(|(_, s)| s).unwrap_or_else(|| span_of(spec.row));
        // Fully definite items are positioned before the auto-placement
        // cursor exists (spec step 1): they never move it.
        let definite_both = col_def.is_some() && row_def.is_some();

        let (row, col) = match (row_def, col_def) {
            (Some((r, _)), Some((c, _))) => {
                // Fully definite: place as specified, even on collision.
                (r, c)
            }
            (Some((r, _)), None) => {
                // Definite row, auto column: scan the row from the
                // cursor column when the cursor is on this row,
                // otherwise from column zero.
                let start_c = if r == cur_row { cur_col } else { 0 };
                let mut c = start_c;
                loop {
                    if (c..c + colspan).all(|cc| !occupied.contains(&occ_key(r, cc))) {
                        break;
                    }
                    c += 1;
                }
                (r, c)
            }
            (None, Some((c, _))) => {
                // Definite column, auto row: scan rows from the cursor.
                let mut r = cur_row;
                loop {
                    if (0..rowspan).all(|dr| {
                        (c..c + colspan).all(|cc| !occupied.contains(&occ_key(r + dr, cc)))
                    }) {
                        break;
                    }
                    r += 1;
                }
                (r, c)
            }
            (None, None) => {
                // Fully automatic: first free rectangle at/after the
                // cursor (css-grid-1 §8.5 step 4). The overflow check
                // runs before the overlap check at each candidate: a
                // spanning item that would overflow the implicit grid
                // wraps to a fresh row instead of growing implicit
                // columns mid-row.
                let (mut r, mut c) = (cur_row, cur_col);
                loop {
                    if explicit_cols == 0 {
                        // No explicit columns: each auto item takes a
                        // fresh implicit row (sparse) or backfills the
                        // first free row (dense).
                        if (0..rowspan).all(|dr| {
                            (0..colspan).all(|cc| !occupied.contains(&occ_key(r + dr, cc)))
                        }) {
                            c = 0;
                            break;
                        }
                        r += 1;
                        continue;
                    }
                    if c + colspan > flow_cols {
                        c = 0;
                        r += 1;
                        continue;
                    }
                    if (0..rowspan).all(|dr| {
                        (c..c + colspan).all(|cc| !occupied.contains(&occ_key(r + dr, cc)))
                    }) {
                        break;
                    }
                    c += 1;
                }
                (r, c)
            }
        };

        for dr in 0..rowspan {
            for dc in 0..colspan {
                occupied.insert(occ_key(row + dr, col + dc));
            }
        }
        ncols = ncols.max(col + colspan);
        nrows = nrows.max(row + rowspan);
        if dense {
            // Dense packing: reset the cursor to the grid start so the
            // next auto-positioned item scans from the beginning and
            // backfills holes left by definite-positioned or spanned
            // items, instead of advancing past them.
            cur_row = 0;
            cur_col = 0;
        } else if !definite_both && (row > cur_row || (row == cur_row && col + colspan > cur_col)) {
            // Advance the sparse cursor past the placed item. Fully
            // definite items are exempt (spec step 1 never moves the
            // cursor). With no explicit columns every item wraps to a
            // fresh implicit row, so auto-placed items stack vertically
            // instead of piling into row 0 with unbounded columns.
            cur_row = row;
            cur_col = col + colspan;
            if explicit_cols == 0 || cur_col >= flow_cols {
                cur_col = 0;
                cur_row += 1;
            }
        }
        placed.push(Placed {
            node: spec.node,
            row,
            col,
            rowspan,
            colspan,
        });
    }
    (placed, ncols, nrows)
}

/// Auto-placement honoring `grid-auto-flow`. Row-major flow (the
/// default) places row by row; column-major flow (`column`, `column
/// dense`) places column by column. Column-major placement transposes
/// the problem: each spec's row/column placements and the explicit
/// track counts swap, the transposed specs place row-major, and the
/// results swap back. The transpose is exact — definite placements
/// resolve per axis, the cursor advances down-then-across, and the
/// no-explicit-tracks wrap becomes one implicit column per item — so
/// dense/sparse backfill behaves the same in both orientations.
fn place_items(
    specs: &[ItemSpec],
    explicit_cols: usize,
    explicit_rows: usize,
    dense: bool,
    column: bool,
) -> (Vec<Placed>, usize, usize) {
    if !column {
        return place_items_row_major(specs, explicit_cols, explicit_rows, dense);
    }
    let swapped: Vec<ItemSpec> = specs
        .iter()
        .map(|s| ItemSpec {
            node: s.node,
            row: s.col,
            col: s.row,
        })
        .collect();
    let (mut placed, ncols_t, nrows_t) =
        place_items_row_major(&swapped, explicit_rows, explicit_cols, dense);
    for p in &mut placed {
        std::mem::swap(&mut p.row, &mut p.col);
        std::mem::swap(&mut p.rowspan, &mut p.colspan);
    }
    // Transposed counts swap back: transposed columns are real rows.
    (placed, nrows_t, ncols_t)
}

/// Definite specified content width of a grid item, for track sizing.
///
/// Shared floor for the intrinsic ([`grid_pref_width`]) and used
/// ([`layout_grid`]) track-sizing passes; mirrors
/// `definite_specified_content_width` (layout.rs): a `calc()` that
/// syntactically contained a percentage leaf is cyclic and behaves as
/// `auto` (CSS Values 4 §8.1.4 — the `had_pct` gate, so even
/// `calc(10px + 0%)` counts), the width resolves against `pct_base`
/// (`None` in the intrinsic pass, where the base is indefinite and a
/// bare percentage is likewise cyclic per CSS 2.1 §10.2;
/// `Some(content_w)` in the used pass, where percentages must
/// resolve), min/max-width clamp the value in the box-sizing's own
/// terms (min wins when over-constrained), and the box-sizing is
/// stripped via the caller's border+padding extras `own_bp`.
/// Returns `None` for `auto` or cyclic widths.
fn grid_item_specified_content_width(
    forest: &Forest,
    i: usize,
    pct_base: Option<f32>,
    own_bp: f32,
) -> Option<f32> {
    let n = &forest.nodes[i];
    let d = &n.data;
    // Whether percentage lengths are cyclic here: only against an
    // indefinite base (the intrinsic pass). The used pass threads a
    // definite base, where percentages must resolve.
    let pct_cyclic = pct_base.is_none();
    // CSS Values 4 §8.1.4 syntactic cyclic-ness (mirrors
    // `definite_specified_content_width`): the gate checks the
    // `had_pct` bit, not the folded coefficient.
    let specified = match d.width {
        SpecLen::Math { had_pct, .. } if had_pct && n.intrinsic.is_none() => return None,
        SpecLen::Pct(_) if pct_cyclic && n.intrinsic.is_none() => return None,
        w => w.resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            pct_base.unwrap_or(0.0),
        ),
    }?
    .max(0.0);
    let resolve = |s: SpecLen| {
        s.resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            pct_base.unwrap_or(0.0),
        )
    };
    // min/max-width clamp in the box-sizing's own terms; percentage
    // expressions are cyclic here too and behave as `none`/`auto`.
    let constraint = |s: SpecLen, none: f32| match s {
        SpecLen::Math { had_pct, .. } if had_pct && n.intrinsic.is_none() => none,
        SpecLen::Pct(_) if pct_cyclic && n.intrinsic.is_none() => none,
        w => resolve(w).unwrap_or(none),
    };
    let min_w = constraint(d.min_width, 0.0).max(0.0);
    let max_w = constraint(d.max_width, f32::INFINITY).max(0.0);
    let clamped = specified.min(max_w).max(min_w);
    // The caller works in content-box terms (the item's own extras ride
    // along separately), so strip the border-box width under
    // `border-box` — otherwise the extras get counted twice.
    Some(match d.box_sizing {
        crate::style::BoxSizing::BorderBox => (clamped - own_bp).max(0.0),
        crate::style::BoxSizing::ContentBox => clamped.max(0.0),
    })
}

/// Phase one of grid layout: parse the grid, place items, size the
/// column tracks, and push one [`Task::Block`] per item. The matching
/// [`Task::Grid`] resume is pushed first so it runs after the items.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layout_grid(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    idx: usize,
    input: &BlockInput,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    kids_x: f32,
    kids_w: f32,
    // Whether this run already is the `overflow-y: auto` scrollbar-width
    // re-drive: carried into the [`GridResume`] so the completion does
    // not re-drive a second time.
    redriven: bool,
) {
    // The `layout_block` prologue already resolved the container's own
    // metrics and collapsed its top margin through the first in-flow
    // child; grid items never collapse with their container, so undo
    // that here (same treatment as the flex container path).
    {
        let n = &mut forest.nodes[idx];
        n.mt_place = n.metrics.mt;
        n.top_collapsed = false;
    }
    {
        let mut cur = forest.nodes[idx]
            .children
            .first()
            .copied()
            .filter(|&c| is_block_level(forest.nodes[c].box_type));
        while let Some(c) = cur {
            forest.nodes[c].mt_place_done = false;
            cur = forest.nodes[c]
                .children
                .first()
                .copied()
                .filter(|&k| is_block_level(forest.nodes[k].box_type));
        }
    }
    if forest.nodes[idx].parent.is_none() {
        let mt = forest.nodes[idx].mt_place;
        forest.nodes[idx].rect.y = mt;
    }

    // `kids_x` / `kids_w` are the `layout_block` prologue's
    // scrollbar-gutter-reserved content origin / width for children:
    // tracks size and items place against those, not the node's
    // unreserved metrics (identical to them when no reservation
    // applies).
    let (content_x, content_top, content_w, font_size) = {
        let n = &forest.nodes[idx];
        (
            kids_x,
            n.rect.y + n.metrics.bt + n.metrics.pt,
            kids_w,
            n.data.font_size,
        )
    };
    let hspec = input
        .height_override
        .or_else(|| specified_height(forest, idx, input.cb_h));

    // A contiguous run of in-flow text and inline-level children forms
    // a single anonymous block grid item (grid §5.1).
    wrap_grid_text_runs(forest, idx);

    let (
        col_template,
        col_collapse,
        row_template,
        row_collapse,
        auto_cols,
        auto_rows,
        row_gap,
        col_gap,
        dense,
        column,
        justify,
        align,
        justify_content,
        align_content,
    ) = {
        let style = forest.nodes[idx].style.as_deref();
        let (root_fs, vp_w, vp_h) = (forest.root_font_size, forest.vp_w, forest.vp_h);
        let (row_gap, col_gap) = parse_gaps(style, font_size, root_fs, vp_w, vp_h, content_w);
        // `repeat(auto-fill|auto-fit, …)` counts against the definite
        // axis size: the content width for columns, the specified
        // height (if any) for rows. `subgrid` on either axis reuses the
        // nearest ancestor's listing for that axis.
        let (col_template, col_collapse) = parse_axis_tracks(
            forest,
            idx,
            props::GRID_TEMPLATE_COLUMNS,
            font_size,
            root_fs,
            vp_w,
            vp_h,
            Some(content_w),
            col_gap,
            false,
        );
        let (row_template, row_collapse) = parse_axis_tracks(
            forest,
            idx,
            props::GRID_TEMPLATE_ROWS,
            font_size,
            root_fs,
            vp_w,
            vp_h,
            hspec,
            row_gap,
            false,
        );
        // Implicit track sizes (`grid-auto-columns` / `grid-auto-rows`,
        // initial `auto`): a single `auto` track when unset.
        let (auto_cols, _) = parse_tracks(
            style,
            props::GRID_AUTO_COLUMNS,
            font_size,
            root_fs,
            vp_w,
            vp_h,
            Some(content_w),
            col_gap,
        );
        let (auto_rows, _) = parse_tracks(
            style,
            props::GRID_AUTO_ROWS,
            font_size,
            root_fs,
            vp_w,
            vp_h,
            hspec,
            row_gap,
        );
        let auto_flow = style.and_then(|s| s.get_str_by_index(props::GRID_AUTO_FLOW));
        (
            col_template,
            col_collapse,
            row_template,
            row_collapse,
            auto_cols,
            auto_rows,
            row_gap,
            col_gap,
            dense_flow(auto_flow),
            column_flow(auto_flow),
            item_align_of(style, props::JUSTIFY_ITEMS),
            item_align_of(style, props::ALIGN_ITEMS),
            content_align_of(style, props::JUSTIFY_CONTENT),
            content_align_of(style, props::ALIGN_CONTENT),
        )
    };

    // Subgrid column axis: resolve against the parent grid's used track
    // sizes/positions for the spanned subset (css-grid-1 §2.3). The
    // subgrid item's containing block is exactly the spanned tracks, so
    // the subset is found by geometric containment; the tracks become
    // fixed at their used sizes and the positions below are overridden
    // with the parent's, so the subgrid's own gap and content alignment
    // do not apply to the inherited axis. Without a usable parent frame
    // (intrinsic sizing, non-grid parent) the parsed listing stands.
    let subgrid_cols = is_subgrid(
        forest.nodes[idx].style.as_deref(),
        props::GRID_TEMPLATE_COLUMNS,
    );
    let subgrid_geom: Option<(Vec<f32>, Vec<f32>)> = if subgrid_cols {
        forest.nodes[idx]
            .parent
            .filter(|&p| is_grid_container(forest, p))
            .and_then(|p| parent_used_columns(tasks, p))
            .and_then(|(px, pw)| {
                subgrid_track_subset(&px, &pw, input.cb_x, input.cb_x + input.cb_w)
                    .map(|(s, e)| (pw[s..e].to_vec(), px[s..e].to_vec()))
            })
    } else {
        None
    };
    let (col_template, col_collapse): (Vec<TrackSize>, Vec<bool>) = match &subgrid_geom {
        Some((widths, _)) => (
            widths
                .iter()
                .map(|&w| TrackSize::Fixed(SpecLen::Px(w)))
                .collect(),
            vec![false; widths.len()],
        ),
        None => (col_template, col_collapse),
    };

    // Subgrid row axis: resolve against the parent grid's track listing
    // for the spanned subset (css-grid-1 §2.3), read off the parent's
    // pending phase-one frame. The parent's *used* row geometry does
    // not exist yet — rows size in phase two, after every item
    // (including this subgrid) completes — so only tracks whose used
    // size is fully determined by the listing freeze at their exact
    // used sizes (see `subgrid_row_subset`); every other track keeps its
    // parsed breadth and sizes from the subgrid's own content, the old
    // approximation. Like the column axis, the subgrid's own row gap
    // and content alignment do not apply to the inherited axis: the
    // parent's row gap spaces the inherited tracks, which pack from the
    // start edge. Without a usable parent frame (intrinsic sizing,
    // non-grid parent) the parsed listing stands.
    let (row_template, row_collapse, row_gap, align_content): (
        Vec<TrackSize>,
        Vec<bool>,
        f32,
        ContentAlign,
    ) = if is_subgrid(
        forest.nodes[idx].style.as_deref(),
        props::GRID_TEMPLATE_ROWS,
    ) {
        match subgrid_row_subset(forest, tasks, idx) {
            Some((tracks, gap)) => {
                let n = tracks.len();
                (tracks, vec![false; n], gap, ContentAlign::Start)
            }
            None => (row_template, row_collapse, row_gap, align_content),
        }
    } else {
        (row_template, row_collapse, row_gap, align_content)
    };

    // Collect in-flow items. Out-of-flow children take no grid space;
    // their static position is the content-box origin (aligned per
    // §9.2 by the abspos post-pass, where the used size is known).
    // `float` and
    // `clear` have no effect on a grid item (css-grid-1 §5.1): a floated
    // child is still a grid item. Whitespace-only runs are not items.
    // `order` reorders placement (stable: equal orders keep DOM order).
    let mut specs = Vec::new();
    // Index iteration instead of `children.clone()`: the loop never
    // mutates `nodes[idx].children` (`record_abs_item` only flags the
    // child and pushes an `abs_items` record), so this reads the same
    // snapshot without the O(kids) allocation.
    let nkids = forest.nodes[idx].children.len();
    for k in 0..nkids {
        let c = forest.nodes[idx].children[k];
        if layout_position::is_out_of_flow(forest, c) {
            layout_position::record_abs_item(forest, c, content_x, content_top);
            continue;
        }
        if is_ws_only(forest, c) {
            continue;
        }
        let style = forest.nodes[c].style.as_deref();
        specs.push(ItemSpec {
            node: c,
            row: parse_placement(style, props::GRID_ROW),
            col: parse_placement(style, props::GRID_COLUMN),
        });
    }
    specs.sort_by_key(|s| order_of(forest.nodes[s.node].style.as_deref()));

    let (placed, ncols, nrows) = place_items(
        &specs,
        col_template.len(),
        row_template.len(),
        dense,
        column,
    );
    // Implicit tracks size from `grid-auto-columns` (`auto` when unset).
    let col_tracks = implicit_tracks(&col_template, &auto_cols, ncols);

    // Resolve each item's metrics early: the border/padding extras feed
    // the stretch width override below, and `compute_mt_place` gives
    // the provisional block-start position inside the item task.
    for p in &placed {
        resolve_metrics(forest, p.node, 0.0, content_w, false, None, false);
        compute_mt_place(forest, p.node, hspec);
    }

    // Size the column tracks. Fixed lengths resolve against the
    // content width; `auto`, `fr`, and implicit tracks measure their
    // items' preferred widths (plus border/padding) first. `fr` tracks
    // keep their content minimum and share leftover space.
    let (root_fs, vp_w, vp_h) = (forest.root_font_size, forest.vp_w, forest.vp_h);
    let mut col_w = vec![0.0f32; ncols];
    let mut col_fr = vec![0.0f32; ncols];
    let mut col_content = vec![false; ncols];
    // `minmax()` bounds per track (`[0, ∞)` for every other track, so
    // the clamp below is the identity for them).
    let mut col_min = vec![0.0f32; ncols];
    let mut col_max = vec![f32::INFINITY; ncols];
    // `repeat(auto-fit, …)`: a repeated track with no item placed into
    // or spanning across it collapses — it keeps its zero init (no size,
    // no `fr` share, no content measurement) and draws no gap.
    let mut col_empty = vec![true; ncols];
    for p in &placed {
        let end = (p.col + p.colspan).min(ncols);
        col_empty[p.col..end].fill(false);
    }
    let col_collapsed: Vec<bool> = (0..ncols)
        .map(|c| col_collapse.get(c).copied().unwrap_or(false) && col_empty[c])
        .collect();
    // Explicit tracks followed by implicit tracks sized from
    // `grid-auto-columns` (`auto` when unset).
    for (c, t) in col_tracks.iter().enumerate() {
        if col_collapsed[c] {
            continue;
        }
        match t {
            TrackSize::Fixed(s) => {
                col_w[c] = s
                    .resolve(font_size, root_fs, vp_w, vp_h, content_w)
                    .unwrap_or(0.0)
                    .max(0.0)
            }
            TrackSize::Fr(f) => {
                col_fr[c] = *f;
                col_content[c] = true;
            }
            TrackSize::Auto => col_content[c] = true,
            TrackSize::MinMax { min, max } => {
                col_min[c] = minmax_floor(*min, font_size, root_fs, vp_w, vp_h, content_w);
                col_max[c] = minmax_ceil(*max, font_size, root_fs, vp_w, vp_h, content_w);
                // Every `minmax()` track measures content: the track
                // grows from its minimum toward its maximum to fit
                // content (grid §12.4), and the clamp into `[min, max]`
                // happens after measurement. A `fr` maximum additionally
                // shares free space after the base is set.
                col_content[c] = true;
                if let MinMaxBound::Fr(f) = max {
                    col_fr[c] = *f;
                }
            }
        }
    }
    // Implicit tracks size from `grid-auto-columns` via `col_tracks`
    // above; nothing left to default here.
    //
    // One scratch triple serves every item's intrinsic measurement
    // below instead of allocating a node-sized `mcw` plus fresh walk
    // buffers per item.
    let mut int_mcw: Vec<f32> = Vec::new();
    let mut int_bufs = WalkBufs::new();
    let mut int_inner = WalkBufs::new();
    let mut need = vec![0.0f32; ncols];
    if col_content.iter().any(|&f| f) {
        // Non-spanning items contribute their full preferred width to
        // their track. An item with a definite specified width
        // contributes that width (not just its intrinsic content).
        for p in placed
            .iter()
            .filter(|p| p.colspan == 1 && col_content[p.col])
        {
            let w = preferred_width_mcw(
                forest,
                p.node,
                fonts,
                pw_cache,
                &mut int_mcw,
                &mut int_bufs,
                &mut int_inner,
            );
            let m = &forest.nodes[p.node].metrics;
            // A definite specified width sizes the track, not just the
            // intrinsic content width (which is zero for empty items).
            // `em`/`ex` resolve against the *item's* font size, not the
            // container's. A percentage-bearing `calc()` is cyclic here
            // (CSS Values 4 §8.1.4) and behaves as `auto` — the
            // `had_pct` gate inside the helper.
            let specified = grid_item_specified_content_width(
                forest,
                p.node,
                Some(content_w),
                m.bl + m.pl + m.pr + m.br,
            );
            let w = specified.map(|s| s.max(w)).unwrap_or(w);
            let n = w + m.bl + m.pl + m.pr + m.br;
            if n > need[p.col] {
                need[p.col] = n;
            }
        }
        // Spanning items distribute any deficit evenly over the
        // content-sized tracks they span.
        for p in placed.iter().filter(|p| p.colspan > 1) {
            let sized: Vec<usize> = (p.col..p.col + p.colspan)
                .filter(|&c| col_content[c])
                .collect();
            if sized.is_empty() {
                continue;
            }
            let w = preferred_width_mcw(
                forest,
                p.node,
                fonts,
                pw_cache,
                &mut int_mcw,
                &mut int_bufs,
                &mut int_inner,
            );
            let m = &forest.nodes[p.node].metrics;
            // A definite specified width on a spanning item raises the
            // span total (mirrors the intrinsic pass): without the floor
            // `preferred_width` of a block-level item misses its own
            // definite width, diverging from the intrinsic measurement.
            // The percentage base is the grid's content width — the same
            // base the used width resolution uses — and a percentage-bearing
            // `calc()` is cyclic here, behaving as `auto` (the `had_pct`
            // gate inside the helper).
            let specified = grid_item_specified_content_width(
                forest,
                p.node,
                Some(content_w),
                m.bl + m.pl + m.pr + m.br,
            );
            let w = specified.map(|s| s.max(w)).unwrap_or(w);
            let n = w + m.bl + m.pl + m.pr + m.br;
            let cur: f32 = (p.col..p.col + p.colspan)
                .map(|c| if col_content[c] { need[c] } else { col_w[c] })
                .sum();
            if n > cur {
                let add = (n - cur) / sized.len() as f32;
                for &c in &sized {
                    need[c] += add;
                }
            }
        }
    }
    {
        // Every track's base size is its fixed width or its measured
        // need, clamped into its `minmax()` bounds; `fr` tracks then
        // share whatever space is left, clamped again after the share.
        // (Non-`minmax()` tracks clamp against `[0, ∞)`, i.e. the clamp
        // is the identity and this matches the old code exactly.) This
        // runs unconditionally: a `minmax()` track with fixed bounds is
        // not content-sized, but still needs its minimum applied.
        let mut base = vec![0.0f32; ncols];
        for c in 0..ncols {
            let b = if col_content[c] { need[c] } else { col_w[c] };
            // An infinite content base (a fixed-layout table with a
            // percentage width: infinite max-content intrinsic size per
            // Blink) cannot outgrow a definite container: cap an `fr`
            // track's infinite base at the container width so the share
            // below resolves against finite free space. (A definite
            // `1fr` grid holds its width; only an indefinite,
            // saturated-huge container lets the track stay huge — the
            // WPT `table-fixed-minmax.html` grid subtests pin both
            // behaviors.) Non-`fr` content-sized tracks keep the
            // infinite base and saturate below.
            let b = if col_fr[c] > 0.0 && b.is_infinite() {
                b.min(content_w)
            } else {
                b
            };
            base[c] = clamp_track(b, col_min[c], col_max[c]);
        }
        let base_sum: f32 = base.iter().sum();
        // Gutters only separate tracks that take space: collapsed
        // `auto-fit` tracks draw no gap.
        let live_cols = ncols - col_collapsed.iter().filter(|&&b| b).count();
        let gaps = col_gap * (live_cols as f32 - 1.0).max(0.0);
        let fr_total: f32 = col_fr.iter().sum();
        let free = (content_w - gaps - base_sum).max(0.0);
        for c in 0..ncols {
            if col_fr[c] > 0.0 && fr_total > 0.0 {
                col_w[c] = clamp_track(
                    base[c] + free * col_fr[c] / fr_total,
                    col_min[c],
                    col_max[c],
                );
            } else {
                col_w[c] = base[c];
            }
            // Saturate an infinite track (see the `fr` cap above) so raw
            // infinities never reach item used rects (see
            // `MAX_USED_SIZE`).
            col_w[c] = saturate_used_size(col_w[c]);
        }
    }

    // `justify-content` distributes leftover inline-axis space after
    // track sizing; `normal` behaves as `stretch`, growing content-sized
    // tracks that took no `fr` share (capped at their `minmax()` maxima).
    let stretchable: Vec<bool> = (0..ncols)
        .map(|c| col_content[c] && col_fr[c] == 0.0)
        .collect();
    let col_stretch = (justify_content == ContentAlign::Stretch)
        .then_some((stretchable.as_slice(), col_max.as_slice()));
    let col_x = layout_track_positions(
        &mut col_w,
        &col_collapsed,
        content_x,
        Some(content_w),
        col_gap,
        justify_content,
        col_stretch,
    );
    // Subgrid: take the parent grid's track positions for the spanned
    // subset, so the subgrid's tracks align with the parent's.
    let col_x = match &subgrid_geom {
        Some((_, xs)) => xs.clone(),
        None => col_x,
    };

    // Per-item alignment: each item's `justify-self` / `align-self`
    // overrides the container's `justify-items` / `align-items` (`auto`
    // defers to the container). Phase two aligns every item from these.
    let item_aligns: Vec<(ItemAlign, ItemAlign)> = placed
        .iter()
        .map(|p| {
            let style = forest.nodes[p.node].style.as_deref();
            (
                self_align_of(style, props::JUSTIFY_SELF, justify),
                self_align_of(style, props::ALIGN_SELF, align),
            )
        })
        .collect();

    // Implicit row tracks size from `grid-auto-rows` (`auto` when
    // unset); they never collapse.
    let row_template = implicit_tracks(&row_template, &auto_rows, nrows);
    let mut row_collapse_full = row_collapse.clone();
    row_collapse_full.resize(nrows, false);

    tasks.push(Task::Grid(Box::new(GridResume {
        idx,
        content_x,
        content_top,
        content_w,
        cb_h: input.cb_h,
        hspec,
        font_size,
        items: placed
            .iter()
            .zip(item_aligns.iter())
            .map(|(p, &(j, a))| PlacedItem {
                node: p.node,
                row: p.row,
                col: p.col,
                rowspan: p.rowspan,
                colspan: p.colspan,
                justify: j,
                align: a,
            })
            .collect(),
        col_x: col_x.clone(),
        col_w: col_w.clone(),
        col_gap,
        row_gap,
        row_template,
        row_collapse: row_collapse_full,
        nrows,
        align_content,
        second_pass: false,
        row_h: Vec::new(),
        row_y: Vec::new(),
        natural_h: 0.0,
        input: input.clone(),
        auto_redriven: redriven,
    })));

    // Push the item tasks in reverse so they pop in placement order.
    // Each item gets a fresh float list: a grid item contains its own
    // floats, like a table cell. An item whose effective inline-axis
    // alignment is not `stretch` shrink-to-fits instead of filling its
    // area (the resume pass shifts it into its aligned position).
    for (p, &(item_justify, _)) in placed.iter().zip(item_aligns.iter()).rev() {
        let end_c = (p.col + p.colspan).min(ncols);
        let area_x = col_x[p.col];
        // Derive the area from the distributed track positions: summing
        // raw track sizes plus raw gaps would drop the extra
        // `justify-content` spacing (`space-between` et al.) between the
        // spanned tracks, sizing a spanning stretched item too narrow.
        let area_w: f32 = if end_c > p.col {
            col_x[end_c - 1] + col_w[end_c - 1] - col_x[p.col]
        } else {
            0.0
        };
        let mt_place = forest.nodes[p.node].mt_place;
        // `stretch` only fills the area when the item's inline size is
        // `auto`: a definite `width` (e.g. `10em`) keeps its specified
        // size (css-align §6.4). Auto inline margins suppress the
        // stretch sizing (css-grid-1 §10.2: auto margins disable the
        // self-alignment effects in that axis): the item shrink-to-fits
        // and the margins absorb the free space at placement instead.
        let width_is_auto = matches!(forest.nodes[p.node].data.width, SpecLen::Auto);
        let auto_inline_margin = {
            let mg = &forest.nodes[p.node].data.margin;
            matches!(mg[3], SpecMargin::Auto) || matches!(mg[1], SpecMargin::Auto)
        };
        let (width_override, shrink) =
            if item_justify == ItemAlign::Stretch && width_is_auto && !auto_inline_margin {
                let m = &forest.nodes[p.node].metrics;
                let extras = m.bl + m.pl + m.pr + m.br;
                (Some((area_w - extras).max(0.0)), false)
            } else {
                (None, true)
            };
        // Provisional position; the block task re-resolves metrics from
        // `cb_x`, and the resume pass shifts the subtree into its final
        // aligned position.
        forest.nodes[p.node].rect.x = area_x;
        forest.nodes[p.node].rect.y = content_top + mt_place;
        tasks.push(Task::Block(BlockInput {
            abs_pct_height_base: None,
            idx: p.node,
            cb_x: area_x,
            cb_w: area_w,
            cb_h: hspec,
            shrink,
            width_override,
            width_override_final: false,
            height_override: None,
            height_is_min: false,
            collapsed_border_wins: None,
            abspos: false,
            floats: fresh_floats(),
            is_float: false,
        }));
    }
}

/// Phase two of grid layout: every item has completed its block
/// layout, so size the row tracks, align items in their areas, shift
/// subtrees into place, and finish the container height.
pub(crate) fn resume_grid(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    fr: GridResume,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    scratch: &mut super::LayoutScratch,
) {
    let idx = fr.idx;
    let nrows = fr.nrows;
    let (fs, root_fs, vp_w, vp_h) = (
        fr.font_size,
        forest.root_font_size,
        forest.vp_w,
        forest.vp_h,
    );

    // Row geometry. On the second pass (stretched items were re-laid
    // out at their stretched heights) the stored geometry is reused:
    // re-sizing from the stretched heights would be circular.
    let (row_h, row_y, natural_h): (Vec<f32>, Vec<f32>, f32) = if fr.second_pass {
        (fr.row_h.clone(), fr.row_y.clone(), fr.natural_h)
    } else {
        // Size the row tracks. Fixed lengths resolve against the
        // container's definite content height; `fr` and `auto` tracks size
        // to their items' border-box heights first (`fr` keeps that as its
        // minimum).
        let mut row_h = vec![0.0f32; nrows];
        let mut row_fr = vec![0.0f32; nrows];
        let mut row_content = vec![false; nrows];
        // `minmax()` bounds per track (`[0, ∞)` for every other track).
        let mut row_min = vec![0.0f32; nrows];
        let mut row_max = vec![f32::INFINITY; nrows];
        // `repeat(auto-fit, …)`: a repeated track with no item placed into
        // or spanning across it collapses — it keeps its zero init (no size,
        // no `fr` share, no content measurement) and draws no gap.
        let mut row_empty = vec![true; nrows];
        for p in &fr.items {
            let end = (p.row + p.rowspan).min(nrows);
            row_empty[p.row..end].fill(false);
        }
        let row_collapsed: Vec<bool> = (0..nrows)
            .map(|r| fr.row_collapse.get(r).copied().unwrap_or(false) && row_empty[r])
            .collect();
        // Gutters only separate tracks that take space: collapsed `auto-fit`
        // tracks draw no gap.
        let live_rows = nrows - row_collapsed.iter().filter(|&&b| b).count();
        let row_gaps = fr.row_gap * (live_rows as f32 - 1.0).max(0.0);
        for (r, t) in fr.row_template.iter().enumerate() {
            if row_collapsed[r] {
                continue;
            }
            match t {
                TrackSize::Fixed(s) => {
                    if matches!(s, SpecLen::Pct(_) | SpecLen::Math { had_pct: true, .. })
                        && fr.hspec.is_none()
                    {
                        // A percentage track against an indefinite container
                        // height cannot resolve and behaves as `auto`: it
                        // sizes to content.
                        row_content[r] = true;
                    } else {
                        // Percentage tracks with an indefinite height were
                        // degraded to content-sized above; every other fixed
                        // length resolves without needing the height (absolute
                        // units ignore the base).
                        row_h[r] = s
                            .resolve(fs, root_fs, vp_w, vp_h, fr.hspec.unwrap_or(0.0))
                            .unwrap_or(0.0)
                            .max(0.0);
                    }
                }
                TrackSize::Fr(f) => {
                    row_fr[r] = *f;
                    row_content[r] = true;
                }
                TrackSize::Auto => row_content[r] = true,
                TrackSize::MinMax { min, max } => {
                    // Percentage bounds against an indefinite container
                    // height degrade to content-based, like a percentage
                    // `Fixed` track degrading to `auto` above.
                    let min = normalize_row_bound(*min, fr.hspec);
                    let max = normalize_row_bound(*max, fr.hspec);
                    let base = fr.hspec.unwrap_or(0.0);
                    row_min[r] = minmax_floor(min, fs, root_fs, vp_w, vp_h, base);
                    row_max[r] = minmax_ceil(max, fs, root_fs, vp_w, vp_h, base);
                    // Every `minmax()` track measures content so it can grow
                    // from its minimum toward its maximum (grid §12.4); the
                    // clamp into `[min, max]` happens after measurement.
                    row_content[r] = true;
                    if let MinMaxBound::Fr(f) = max {
                        row_fr[r] = f;
                    }
                }
            }
        }
        for p in fr
            .items
            .iter()
            .filter(|p| p.rowspan == 1 && row_content[p.row])
        {
            let h = forest.nodes[p.node].rect.h;
            if h > row_h[p.row] {
                row_h[p.row] = h;
            }
        }
        // Clamp every track's base size into its `minmax()` bounds (the
        // identity for every other track, whose bounds are `[0, ∞)`).
        for r in 0..nrows {
            row_h[r] = clamp_track(row_h[r], row_min[r], row_max[r]);
        }
        // Definite container height: `fr` rows keep their content minimum
        // and share the remaining space.
        let fr_total: f32 = row_fr.iter().sum();
        if fr_total > 0.0
            && let Some(h) = fr.hspec
        {
            let used: f32 = row_h.iter().sum();
            let free = (h - row_gaps - used).max(0.0);
            for r in 0..nrows {
                if row_fr[r] > 0.0 {
                    row_h[r] = clamp_track(
                        row_h[r] + free * row_fr[r] / fr_total,
                        row_min[r],
                        row_max[r],
                    );
                }
            }
        }
        // Spanning items distribute any height deficit evenly over the
        // content-sized rows they span.
        for p in fr.items.iter().filter(|p| p.rowspan > 1) {
            let end = (p.row + p.rowspan).min(nrows);
            let sized: Vec<usize> = (p.row..end).filter(|&r| row_content[r]).collect();
            if sized.is_empty() {
                continue;
            }
            let cur: f32 = (p.row..end).map(|r| row_h[r]).sum::<f32>()
                + fr.row_gap * end.saturating_sub(p.row + 1) as f32;
            let h = forest.nodes[p.node].rect.h;
            if h > cur {
                let add = (h - cur) / sized.len() as f32;
                for &r in &sized {
                    row_h[r] += add;
                }
            }
        }
        // Spanning items may have grown tracks past their `minmax()` maxima;
        // clamp once more (the identity for non-`minmax()` tracks).
        for r in 0..nrows {
            row_h[r] = clamp_track(row_h[r], row_min[r], row_max[r]);
        }

        // `align-content` distributes leftover block-axis space after row
        // sizing; `normal` behaves as `stretch`, growing content-sized
        // rows that took no `fr` share (it is a no-op when the container
        // height is indefinite).
        let row_stretchable: Vec<bool> = (0..nrows)
            .map(|r| row_content[r] && row_fr[r] == 0.0)
            .collect();
        let row_stretch = (fr.align_content == ContentAlign::Stretch)
            .then_some((row_stretchable.as_slice(), row_max.as_slice()));
        let row_y = layout_track_positions(
            &mut row_h,
            &row_collapsed,
            fr.content_top,
            fr.hspec,
            fr.row_gap,
            fr.align_content,
            row_stretch,
        );
        let natural_h = row_h.iter().sum::<f32>() + row_gaps;
        (row_h, row_y, natural_h)
    };

    // Align every item in its grid area and shift its subtree into
    // place. Placement is computed for every item before anything
    // mutates: on the first pass, stretched items with auto height are
    // re-laid out at their stretched size (below) instead of merely
    // growing their border box with top-aligned content.
    struct ItemPlace {
        dx: f32,
        dy: f32,
        grow: Option<f32>,
        area_x: f32,
        area_w: f32,
    }
    let mut places: Vec<ItemPlace> = Vec::with_capacity(fr.items.len());
    for p in &fr.items {
        let end_c = (p.col + p.colspan).min(fr.col_w.len());
        let end_r = (p.row + p.rowspan).min(nrows);
        let area_x = fr.col_x[p.col];
        // Derive the area from the distributed track positions, like
        // phase one: raw track sizes plus raw gaps would drop the extra
        // content-alignment spacing between the spanned tracks.
        let area_w: f32 = if end_c > p.col {
            fr.col_x[end_c - 1] + fr.col_w[end_c - 1] - fr.col_x[p.col]
        } else {
            0.0
        };
        let area_y = row_y[p.row];
        let area_h: f32 = if end_r > p.row {
            row_y[end_r - 1] + row_h[end_r - 1] - row_y[p.row]
        } else {
            0.0
        };
        let (px, py, mt, mb, h, w, height_auto, margin, ml_u, mr_u) = {
            let n = &forest.nodes[p.node];
            // Alignment positions the border box from the collapsed
            // placement margins (`mt_place`, collapsed through the
            // first-child chain like flex's cross-start margin), matching
            // the provisional position phase one used.
            (
                n.rect.x,
                n.rect.y,
                n.mt_place,
                n.metrics.mb,
                n.rect.h,
                n.rect.w,
                matches!(n.data.height, SpecLen::Auto),
                n.data.margin,
                n.metrics.ml,
                n.metrics.mr,
            )
        };
        // css-grid-1 §10.2: auto margins absorb positive free space
        // *before* alignment runs, so with any auto margin on an axis the
        // alignment properties (`justify-self` / `align-self`) are a
        // no-op there: both auto margins split the free space evenly
        // (centering), a single auto margin takes it all, and
        // over-constrained (non-positive free space) auto margins
        // behave as zero (then the alignment properties apply).
        // Free space is measured treating auto margins as zero: the
        // block path already resolved inline auto margins for
        // definite-width items (CSS 2.1 §10.3.3), so measuring against
        // the used margins would see free=0 and wrongly apply the
        // alignment shift on top of the resolved margins. Auto block
        // margins resolve to zero in the block path, so the full
        // margin-box free space is available on that axis. Mirrors the
        // auto-margin math of `flex_justify` / the flex cross-axis pass.
        let auto_ml = matches!(margin[3], SpecMargin::Auto);
        let auto_mr = matches!(margin[1], SpecMargin::Auto);
        let auto_mt = matches!(margin[0], SpecMargin::Auto);
        let auto_mb = matches!(margin[2], SpecMargin::Auto);
        // Each item aligns per its own `justify-self` / `align-self`
        // (falling back to the container's `justify-items` /
        // `align-items`), not the container-wide value.
        let dx = {
            // §10.3.3 distribution against zeroed auto margins (see the
            // comment above): alignment is a no-op while an auto margin
            // absorbs free space on this axis.
            let fixed_ml = if auto_ml { 0.0 } else { ml_u };
            let fixed_mr = if auto_mr { 0.0 } else { mr_u };
            let free = area_w - fixed_ml - w - fixed_mr;
            let n_auto = usize::from(auto_ml) + usize::from(auto_mr);
            if free > 0.0 && n_auto > 0 {
                let each = free / n_auto as f32;
                area_x + fixed_ml + if auto_ml { each } else { 0.0 } - px
            } else {
                match p.justify {
                    ItemAlign::Stretch | ItemAlign::Start => 0.0,
                    ItemAlign::Center => area_x + (area_w - w) / 2.0 - px,
                    ItemAlign::End => area_x + area_w - w - px,
                }
            }
        };
        let (dy, grow) = {
            let free = area_h - mt - h - mb;
            let n_auto = usize::from(auto_mt) + usize::from(auto_mb);
            if free > 0.0 && n_auto > 0 {
                // Auto margins also beat `stretch` growth on this axis
                // (the item keeps its laid-out height; the margins take
                // the space), so no second-pass re-layout is requested.
                let each = free / n_auto as f32;
                (area_y + mt + if auto_mt { each } else { 0.0 } - py, None)
            } else {
                match p.align {
                    // Aspect-ratio arm takes precedence over stretch: a
                    // replaced item with an intrinsic ratio keeps the
                    // ratio-transferred height from phase one (see
                    // `replaced_minmax_transfer` in the block path)
                    // instead of growing into the area.
                    ItemAlign::Stretch
                        if height_auto && super::intrinsic_ratio(forest, p.node).is_none() =>
                    {
                        let target = (area_h - mt - mb).max(0.0);
                        (
                            area_y + mt - py,
                            if target > h { Some(target) } else { None },
                        )
                    }
                    _ => {
                        let top = match p.align {
                            ItemAlign::End => area_y + area_h - h,
                            ItemAlign::Center => area_y + (area_h - h) / 2.0,
                            _ => area_y + mt,
                        };
                        (top - py, None)
                    }
                }
            }
        };
        places.push(ItemPlace {
            dx,
            dy,
            grow,
            area_x,
            area_w,
        });
    }

    // Stretched items with auto height grow their border box to the
    // area: re-lay them out at the stretched size (first pass only) so
    // their content — percentage heights, nested stretching — uses the
    // final size instead of staying top-aligned at the pre-stretch
    // height. Deferred shifts from the first layout would apply twice,
    // so they are cleared before the re-layout (like flex's
    // `align-self: stretch` re-run). The second pass then aligns
    // everything without further growth.
    if !fr.second_pass && places.iter().any(|pl| pl.grow.is_some()) {
        tasks.push(Task::Grid(Box::new(GridResume {
            idx: fr.idx,
            content_x: fr.content_x,
            content_top: fr.content_top,
            content_w: fr.content_w,
            cb_h: fr.cb_h,
            hspec: fr.hspec,
            font_size: fr.font_size,
            items: fr.items.clone(),
            col_x: fr.col_x.clone(),
            col_w: fr.col_w.clone(),
            col_gap: fr.col_gap,
            row_gap: fr.row_gap,
            row_template: fr.row_template.clone(),
            row_collapse: fr.row_collapse.clone(),
            nrows: fr.nrows,
            align_content: fr.align_content,
            second_pass: true,
            row_h: row_h.clone(),
            row_y: row_y.clone(),
            natural_h,
            input: fr.input.clone(),
            auto_redriven: fr.auto_redriven,
        })));
        // Push in reverse so they pop in placement order.
        for (p, pl) in fr.items.iter().zip(places.iter()).rev() {
            let Some(target) = pl.grow else { continue };
            let node = p.node;
            super::clear_pending_shifts(forest, node);
            let (mt_place, vextras, width_override, shrink) = {
                let n = &forest.nodes[node];
                let m = &n.metrics;
                let extras = m.bl + m.pl + m.pr + m.br;
                // Like phase one: auto inline margins suppress the
                // stretch sizing (css-grid-1 §10.2), or the re-layout
                // would undo the shrink-to-fit from the first pass.
                let auto_inline_margin = matches!(n.data.margin[3], SpecMargin::Auto)
                    || matches!(n.data.margin[1], SpecMargin::Auto);
                let (width_override, shrink) = if p.justify == ItemAlign::Stretch
                    && matches!(n.data.width, SpecLen::Auto)
                    && !auto_inline_margin
                {
                    (Some((pl.area_w - extras).max(0.0)), false)
                } else {
                    (None, true)
                };
                (
                    n.mt_place,
                    m.bt + m.pt + m.pb + m.bb,
                    width_override,
                    shrink,
                )
            };
            // Provisional position, like phase one; the second resume
            // shifts the subtree into its final aligned position.
            forest.nodes[node].rect.x = pl.area_x;
            forest.nodes[node].rect.y = fr.content_top + mt_place;
            tasks.push(Task::Block(BlockInput {
                abs_pct_height_base: None,
                idx: node,
                cb_x: pl.area_x,
                cb_w: pl.area_w,
                cb_h: fr.hspec,
                shrink,
                width_override,
                width_override_final: false,
                // A content height, like a specified height: the border
                // box lands on the stretch target.
                height_override: Some((target - vextras).max(0.0)),
                height_is_min: false,
                collapsed_border_wins: None,
                abspos: false,
                floats: fresh_floats(),
                is_float: false,
            }));
        }
        return;
    }

    for (p, pl) in fr.items.iter().zip(places.iter()) {
        if pl.dx != 0.0 || pl.dy != 0.0 {
            record_shift(forest, p.node, pl.dx, pl.dy);
        }
        if let Some(t) = pl.grow {
            // Second pass only (the first pass re-lays out instead):
            // rounding residue after the re-layout.
            forest.nodes[p.node].rect.h = t;
        }
    }

    finish_content_height(forest, idx, fr.cb_h, fr.hspec, natural_h);
    // Inline-level grids align by the first baseline of their inner
    // content (css-align-3; v1 approximation in
    // `super::first_inner_baseline`); block-level grids keep the
    // border-box bottom.
    forest.nodes[idx].baseline =
        if forest.nodes[idx].data.display == crate::style::Display::InlineGrid {
            super::first_inner_baseline(forest, idx)
        } else {
            forest.nodes[idx].rect.h
        };
    // Shared `overflow-y: auto` scrollbar re-drive: when the shown
    // scrollbar went unreserved, re-run the whole grid once against
    // the narrowed width. The original input is replayed unchanged —
    // only the children's available width shrinks — and the re-driven
    // resume carries `auto_redriven`, so this cannot loop. Stale
    // out-of-flow static positions were already cleared by the helper,
    // so the second pass re-records them fresh.
    let input = fr.input.clone();
    let check = AutoRedriveCheck {
        idx,
        natural_h,
        content_h: forest.nodes[idx].content_h,
        content_w: fr.content_w,
        content_x: fr.content_x,
        auto_redriven: fr.auto_redriven,
    };
    if let Some((kids_x2, kids_w2)) = auto_scrollbar_redrive(forest, scratch, &check) {
        layout_grid(
            forest, tasks, idx, &input, fonts, pw_cache, kids_x2, kids_w2, true,
        );
    }
}

/// Consult the grid container's own specified `width` for its intrinsic
/// contribution (css-sizing-3 §5.2), mirroring the flex/multicol arms of
/// `layout::preferred_width_mcw`:
/// - `width: min-content` contributes the min-content size, measured by
///   the min-content walk over the caller's pooled buffers;
/// - a definite specified width (or a replaced box) contributes exactly
///   its content-box width (a definite `width` via
///   [`definite_specified_content_width`], else the replaced box's
///   [`intrinsic_content_width`]);
/// - otherwise (`auto`, `max-content`, `fit-content`, cyclic
///   percentages) the track-based `intrinsic` stands: the fit-content
///   clamp against the available width happens at used-value time,
///   where the base is definite (the float/abspos shrink-to-fit
///   formula). That track sum is still clamped by the grid's own
///   `min-width`/`max-width` ([`super::clamp_content_width_by_minmax`],
///   mirroring the block/flex content-based fallbacks): the definite
///   arms above clamp inside their own resolution, but the raw
///   intrinsic sum does not.
fn grid_pref_adjusted(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    intrinsic: f32,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
) -> f32 {
    if matches!(forest.nodes[idx].data.width, SpecLen::MinContent) {
        return min_content_width_pooled(forest, idx, fonts, mcw, bufs);
    }
    definite_specified_content_width(forest, idx)
        .or_else(|| intrinsic_content_width(forest, idx))
        .unwrap_or_else(|| super::clamp_content_width_by_minmax(forest, idx, intrinsic))
}

/// Preferred width of the grid container: wrap the text runs, then sum
/// the preferred widths of the column tracks plus gaps. Fixed tracks
/// resolve against a zero base; `auto`/`fr`/implicit tracks take the
/// widest preferred item width they hold (plus border/padding).
pub(crate) fn grid_pref_width(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
) -> f32 {
    wrap_grid_text_runs(forest, idx);

    // One scratch triple serves every item's intrinsic measurement
    // below instead of allocating a node-sized `mcw` plus fresh walk
    // buffers per item.
    let mut int_mcw: Vec<f32> = Vec::new();
    let mut int_bufs = WalkBufs::new();
    let mut int_inner = WalkBufs::new();

    // `float` has no effect on grid items (css-grid-1 §5.1); `order`
    // reorders placement (stable).
    let mut specs = Vec::new();
    // Index iteration instead of `children.clone()`: the loop never
    // mutates `nodes[idx].children`, so this reads the same snapshot
    // without the O(kids) allocation.
    let nkids = forest.nodes[idx].children.len();
    for k in 0..nkids {
        let c = forest.nodes[idx].children[k];
        if layout_position::is_out_of_flow(forest, c) || is_ws_only(forest, c) {
            continue;
        }
        let style = forest.nodes[c].style.as_deref();
        specs.push(ItemSpec {
            node: c,
            row: parse_placement(style, props::GRID_ROW),
            col: parse_placement(style, props::GRID_COLUMN),
        });
    }
    specs.sort_by_key(|s| order_of(forest.nodes[s.node].style.as_deref()));
    let (col_template, row_template, col_gap, dense, column, auto_cols) = {
        let style = forest.nodes[idx].style.as_deref();
        let (root_fs, vp_w, vp_h, font_size) = (
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            forest.nodes[idx].data.font_size,
        );
        let (_row_gap, col_gap) = parse_gaps(style, font_size, root_fs, vp_w, vp_h, 0.0);
        // Intrinsic sizing has no definite axis size: `auto-fill`/`auto-fit`
        // fall back to a single uncollapsed repetition (documented), and a
        // `subgrid` reuses the parent grid's listing resolved against an
        // indefinite parent axis — percentage track breadths degrade to
        // content-sized instead of resolving against the parent's
        // not-yet-laid-out content width (the used-geometry resolution
        // only runs during layout, not here).
        let (col_template, _) = parse_axis_tracks(
            forest,
            idx,
            props::GRID_TEMPLATE_COLUMNS,
            font_size,
            root_fs,
            vp_w,
            vp_h,
            None,
            col_gap,
            true,
        );
        let (row_template, _) = parse_axis_tracks(
            forest,
            idx,
            props::GRID_TEMPLATE_ROWS,
            font_size,
            root_fs,
            vp_w,
            vp_h,
            None,
            0.0,
            true,
        );
        let (auto_cols, _) = parse_tracks(
            style,
            props::GRID_AUTO_COLUMNS,
            font_size,
            root_fs,
            vp_w,
            vp_h,
            None,
            col_gap,
        );
        let auto_flow = style.and_then(|s| s.get_str_by_index(props::GRID_AUTO_FLOW));
        (
            col_template,
            row_template,
            col_gap,
            dense_flow(auto_flow),
            column_flow(auto_flow),
            auto_cols,
        )
    };
    let (placed, ncols, _nrows) = place_items(
        &specs,
        col_template.len(),
        row_template.len(),
        dense,
        column,
    );
    // Implicit tracks size from `grid-auto-columns` (`auto` when unset).
    let col_tracks = implicit_tracks(&col_template, &auto_cols, ncols);
    // Intrinsic fast path (see `grid_pref_adjusted`): a definite specified
    // width (or a replaced box) contributes exactly its content-box width,
    // and `width: min-content` contributes the min-content walk — neither
    // consults the track-based intrinsic size computed below, so the
    // per-item `preferred_width_mcw` / `min_content_width` walks are
    // skipped. The arms observe the same forest state as the late path:
    // `wrap_grid_text_runs` has run, and the track machinery's mutations
    // are either re-done inside the min-content walk (`resolve_metrics`
    // in `grid_min_width`) or invisible to it (pref-width cache fills).
    if matches!(forest.nodes[idx].data.width, SpecLen::MinContent) {
        let mut early_mcw = Vec::new();
        let mut early_bufs = WalkBufs::new();
        return min_content_width_pooled(forest, idx, fonts, &mut early_mcw, &mut early_bufs);
    }
    if let Some(definite) = definite_specified_content_width(forest, idx)
        .or_else(|| intrinsic_content_width(forest, idx))
    {
        return definite;
    }

    let (root_fs, vp_w, vp_h, font_size) = (
        forest.root_font_size,
        forest.vp_w,
        forest.vp_h,
        forest.nodes[idx].data.font_size,
    );
    let mut col_w = vec![0.0f32; ncols];
    let mut col_content = vec![false; ncols];
    let mut col_min = vec![0.0f32; ncols];
    let mut col_max = vec![f32::INFINITY; ncols];
    // Explicit tracks followed by implicit tracks sized from
    // `grid-auto-columns` (`auto` when unset).
    for (c, t) in col_tracks.iter().enumerate() {
        match t {
            TrackSize::Fixed(s) => {
                // A percentage or percentage-bearing `calc()` track
                // against an indefinite container width cannot resolve
                // and behaves as `auto`: it sizes to content (the column
                // counterpart of the row-axis gate in `resume_grid`).
                if matches!(
                    normalize_col_bound(MinMaxBound::Fixed(*s), None),
                    MinMaxBound::Content
                ) {
                    col_content[c] = true;
                } else {
                    col_w[c] = s
                        .resolve(font_size, root_fs, vp_w, vp_h, 0.0)
                        .unwrap_or(0.0)
                        .max(0.0)
                }
            }
            TrackSize::Fr(_) | TrackSize::Auto => col_content[c] = true,
            TrackSize::MinMax { min, max } => {
                // Percentage bounds against an indefinite container
                // width degrade to content-based, like a percentage
                // `Fixed` track degrading to `auto` above.
                let min = normalize_col_bound(*min, None);
                let max = normalize_col_bound(*max, None);
                col_min[c] = minmax_floor(min, font_size, root_fs, vp_w, vp_h, 0.0);
                col_max[c] = minmax_ceil(max, font_size, root_fs, vp_w, vp_h, 0.0);
                // Every `minmax()` track measures content so it can grow
                // from its minimum toward its maximum (grid §12.4); the
                // clamp into `[min, max]` happens after measurement.
                col_content[c] = true;
            }
        }
    }
    let mut need = vec![0.0f32; ncols];
    if col_content.iter().any(|&f| f) {
        for p in placed
            .iter()
            .filter(|p| p.colspan == 1 && col_content[p.col])
        {
            resolve_metrics(forest, p.node, 0.0, 0.0, false, None, false);
            let w = preferred_width_mcw(
                forest,
                p.node,
                fonts,
                pw_cache,
                &mut int_mcw,
                &mut int_bufs,
                &mut int_inner,
            );
            let m = &forest.nodes[p.node].metrics;
            // A definite specified width floors the track (mirrors the
            // used pass): the intrinsic base is indefinite (`None`), so
            // a percentage-bearing `calc()` is cyclic → `auto` (the
            // `had_pct` gate) and a bare percentage max-width behaves
            // as `none` instead of resolving against a zero base.
            let specified =
                grid_item_specified_content_width(forest, p.node, None, m.bl + m.pl + m.pr + m.br);
            let w = specified.map(|s| s.max(w)).unwrap_or(w);
            let n = w + m.bl + m.pl + m.pr + m.br;
            if n > need[p.col] {
                need[p.col] = n;
            }
        }
        for p in placed.iter().filter(|p| p.colspan > 1) {
            let sized: Vec<usize> = (p.col..p.col + p.colspan)
                .filter(|&c| col_content[c])
                .collect();
            if sized.is_empty() {
                continue;
            }
            resolve_metrics(forest, p.node, 0.0, 0.0, false, None, false);
            let w = preferred_width_mcw(
                forest,
                p.node,
                fonts,
                pw_cache,
                &mut int_mcw,
                &mut int_bufs,
                &mut int_inner,
            );
            let m = &forest.nodes[p.node].metrics;
            // A definite specified width on a spanning item raises the
            // span total; the deficit logic below distributes it over
            // the content-sized tracks it spans.
            let specified =
                grid_item_specified_content_width(forest, p.node, None, m.bl + m.pl + m.pr + m.br);
            let w = specified.map(|s| s.max(w)).unwrap_or(w);
            let n = w + m.bl + m.pl + m.pr + m.br;
            let cur: f32 = (p.col..p.col + p.colspan)
                .map(|c| if col_content[c] { need[c] } else { col_w[c] })
                .sum();
            if n > cur {
                let add = (n - cur) / sized.len() as f32;
                for &c in &sized {
                    need[c] += add;
                }
            }
        }
    }
    // Apply the final bounds unconditionally: a `minmax()` track with
    // fixed bounds is not content-sized, but still needs its minimum.
    // (For every other track this is the identity.)
    for c in 0..ncols {
        let b = if col_content[c] { need[c] } else { col_w[c] };
        col_w[c] = clamp_track(b, col_min[c], col_max[c]);
    }

    let gaps = col_gap * (ncols as f32 - 1.0).max(0.0);
    let intrinsic = col_w.iter().sum::<f32>() + gaps;
    grid_pref_adjusted(forest, idx, fonts, intrinsic, &mut int_mcw, &mut int_bufs)
}

/// Min-content width of the grid container (css-grid-1 §12.5): the
/// counterpart of [`grid_pref_width`] for the min-content contribution.
/// Same track walk — wrap the text runs, place the items, sum the
/// column tracks plus gaps — except content-sized tracks measure their
/// items' *min-content* contributions instead of their preferred
/// widths. Fixed tracks resolve against a zero base (no definite
/// container width exists this early); a definite specified item width
/// floors its track exactly as in the max-content pass.
pub(crate) fn grid_min_width(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    mcw: &mut Vec<f32>,
) -> f32 {
    wrap_grid_text_runs(forest, idx);

    // One pooled walk pair serves every item's `min_content_width`
    // below instead of allocating fresh walk buffers per item (the
    // node-indexed `mcw` already comes from the caller).
    let mut int_bufs = WalkBufs::new();

    // `float` has no effect on grid items (css-grid-1 §5.1); `order`
    // reorders placement (stable).
    let mut specs = Vec::new();
    // Index iteration instead of `children.clone()`: the loop never
    // mutates `nodes[idx].children`, so this reads the same snapshot
    // without the O(kids) allocation.
    let nkids = forest.nodes[idx].children.len();
    for k in 0..nkids {
        let c = forest.nodes[idx].children[k];
        if layout_position::is_out_of_flow(forest, c) || is_ws_only(forest, c) {
            continue;
        }
        let style = forest.nodes[c].style.as_deref();
        specs.push(ItemSpec {
            node: c,
            row: parse_placement(style, props::GRID_ROW),
            col: parse_placement(style, props::GRID_COLUMN),
        });
    }
    specs.sort_by_key(|s| order_of(forest.nodes[s.node].style.as_deref()));
    let (col_template, row_template, col_gap, dense, column, auto_cols) = {
        let style = forest.nodes[idx].style.as_deref();
        let (root_fs, vp_w, vp_h, font_size) = (
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            forest.nodes[idx].data.font_size,
        );
        let (_row_gap, col_gap) = parse_gaps(style, font_size, root_fs, vp_w, vp_h, 0.0);
        // Intrinsic sizing has no definite axis size: `auto-fill`/`auto-fit`
        // fall back to a single uncollapsed repetition, and `subgrid`
        // reuses the parent grid's listing resolved against an indefinite
        // parent axis (as in `grid_pref_width`).
        let (col_template, _) = parse_axis_tracks(
            forest,
            idx,
            props::GRID_TEMPLATE_COLUMNS,
            font_size,
            root_fs,
            vp_w,
            vp_h,
            None,
            col_gap,
            true,
        );
        let (row_template, _) = parse_axis_tracks(
            forest,
            idx,
            props::GRID_TEMPLATE_ROWS,
            font_size,
            root_fs,
            vp_w,
            vp_h,
            None,
            0.0,
            true,
        );
        let (auto_cols, _) = parse_tracks(
            style,
            props::GRID_AUTO_COLUMNS,
            font_size,
            root_fs,
            vp_w,
            vp_h,
            None,
            col_gap,
        );
        let auto_flow = style.and_then(|s| s.get_str_by_index(props::GRID_AUTO_FLOW));
        (
            col_template,
            row_template,
            col_gap,
            dense_flow(auto_flow),
            column_flow(auto_flow),
            auto_cols,
        )
    };
    let (placed, ncols, _nrows) = place_items(
        &specs,
        col_template.len(),
        row_template.len(),
        dense,
        column,
    );
    // Implicit tracks size from `grid-auto-columns` (`auto` when unset).
    let col_tracks = implicit_tracks(&col_template, &auto_cols, ncols);

    let (root_fs, vp_w, vp_h, font_size) = (
        forest.root_font_size,
        forest.vp_w,
        forest.vp_h,
        forest.nodes[idx].data.font_size,
    );
    let mut col_w = vec![0.0f32; ncols];
    let mut col_content = vec![false; ncols];
    let mut col_min = vec![0.0f32; ncols];
    let mut col_max = vec![f32::INFINITY; ncols];
    for (c, t) in col_tracks.iter().enumerate() {
        match t {
            TrackSize::Fixed(s) => {
                // A percentage or percentage-bearing `calc()` track
                // against an indefinite container width cannot resolve
                // and behaves as `auto`: it sizes to content (the column
                // counterpart of the row-axis gate in `resume_grid`).
                if matches!(
                    normalize_col_bound(MinMaxBound::Fixed(*s), None),
                    MinMaxBound::Content
                ) {
                    col_content[c] = true;
                } else {
                    col_w[c] = s
                        .resolve(font_size, root_fs, vp_w, vp_h, 0.0)
                        .unwrap_or(0.0)
                        .max(0.0)
                }
            }
            TrackSize::Fr(_) | TrackSize::Auto => col_content[c] = true,
            TrackSize::MinMax { min, max } => {
                // Percentage bounds against an indefinite container
                // width degrade to content-based, like a percentage
                // `Fixed` track degrading to `auto` above.
                let min = normalize_col_bound(*min, None);
                let max = normalize_col_bound(*max, None);
                col_min[c] = minmax_floor(min, font_size, root_fs, vp_w, vp_h, 0.0);
                col_max[c] = minmax_ceil(max, font_size, root_fs, vp_w, vp_h, 0.0);
                // A `minmax()` track's minimum is the largest of its
                // fixed floor and its items' min-content contributions,
                // so it measures content like the max-content pass; the
                // clamp into `[min, max]` happens after measurement.
                col_content[c] = true;
            }
        }
    }
    let mut need = vec![0.0f32; ncols];
    if col_content.iter().any(|&f| f) {
        // Single-span items first: each content-sized track takes the
        // largest min-content contribution (plus border/padding) of the
        // items it holds.
        for p in placed
            .iter()
            .filter(|p| p.colspan == 1 && col_content[p.col])
        {
            resolve_metrics(forest, p.node, 0.0, 0.0, false, None, false);
            let w = min_content_width_pooled(forest, p.node, fonts, mcw, &mut int_bufs);
            let m = &forest.nodes[p.node].metrics;
            // A definite specified width floors the contribution
            // (mirrors the max-content pass): the intrinsic base is
            // indefinite (`None`), so a percentage-bearing `calc()` is
            // cyclic → `auto` and a bare percentage max-width behaves
            // as `none` instead of resolving against a zero base.
            let specified =
                grid_item_specified_content_width(forest, p.node, None, m.bl + m.pl + m.pr + m.br);
            let w = specified.map(|s| s.max(w)).unwrap_or(w);
            let n = w + m.bl + m.pl + m.pr + m.br;
            if n > need[p.col] {
                need[p.col] = n;
            }
        }
        // Multi-span items: spread the deficit (the item's min-content
        // contribution over the span's current total) equally over the
        // content-sized tracks it spans — the documented simplification
        // in the module header.
        for p in placed.iter().filter(|p| p.colspan > 1) {
            let sized: Vec<usize> = (p.col..p.col + p.colspan)
                .filter(|&c| col_content[c])
                .collect();
            if sized.is_empty() {
                continue;
            }
            resolve_metrics(forest, p.node, 0.0, 0.0, false, None, false);
            let w = min_content_width_pooled(forest, p.node, fonts, mcw, &mut int_bufs);
            let m = &forest.nodes[p.node].metrics;
            let specified =
                grid_item_specified_content_width(forest, p.node, None, m.bl + m.pl + m.pr + m.br);
            let w = specified.map(|s| s.max(w)).unwrap_or(w);
            let n = w + m.bl + m.pl + m.pr + m.br;
            let cur: f32 = (p.col..p.col + p.colspan)
                .map(|c| if col_content[c] { need[c] } else { col_w[c] })
                .sum();
            if n > cur {
                let add = (n - cur) / sized.len() as f32;
                for &c in &sized {
                    need[c] += add;
                }
            }
        }
    }
    // Apply the final bounds unconditionally: a `minmax()` track with
    // fixed bounds is not content-sized, but still needs its minimum.
    for c in 0..ncols {
        let b = if col_content[c] { need[c] } else { col_w[c] };
        col_w[c] = clamp_track(b, col_min[c], col_max[c]);
    }

    let gaps = col_gap * (ncols as f32 - 1.0).max(0.0);
    col_w.iter().sum::<f32>() + gaps
}

#[cfg(test)]
mod grid_tests {
    use super::*;

    fn auto_spec(node: usize) -> ItemSpec {
        let auto = GridPlacement {
            start: GridLine::Auto,
            end: GridLine::Auto,
        };
        ItemSpec {
            node,
            row: auto,
            col: auto,
        }
    }

    #[test]
    fn definite_line_ranges_resolve() {
        // `grid-column: 1 / 3` covers columns 0..2.
        let p = GridPlacement {
            start: GridLine::Line(1),
            end: GridLine::Line(3),
        };
        assert_eq!(definite_range(p, 4), Some((0, 2)));
        // `span 2` from line 2.
        let p = GridPlacement {
            start: GridLine::Line(2),
            end: GridLine::Span(2),
        };
        assert_eq!(definite_range(p, 4), Some((1, 2)));
        // Negative lines count back from the explicit end: with 4
        // explicit tracks line `-1` is the line after the last track.
        let p = GridPlacement {
            start: GridLine::Line(-1),
            end: GridLine::Line(-1),
        };
        assert_eq!(definite_range(p, 4), Some((4, 1)));
        // `grid-column: 1 / -1` covers all 4 tracks.
        let p = GridPlacement {
            start: GridLine::Line(1),
            end: GridLine::Line(-1),
        };
        assert_eq!(definite_range(p, 4), Some((0, 4)));
        // `-2 / -1` is the last track; lines past the start clamp to 0.
        let p = GridPlacement {
            start: GridLine::Line(-2),
            end: GridLine::Line(-1),
        };
        assert_eq!(definite_range(p, 4), Some((3, 1)));
        let p = GridPlacement {
            start: GridLine::Line(-9),
            end: GridLine::Line(-1),
        };
        assert_eq!(definite_range(p, 4), Some((0, 4)));
        // Auto placement is not definite.
        assert_eq!(definite_range(auto_spec(0).col, 4), None);
    }

    #[test]
    fn auto_placement_fills_rows_in_order() {
        let specs: Vec<ItemSpec> = (0..5).map(auto_spec).collect();
        let (placed, ncols, nrows) = place_items(&specs, 2, 0, false, false);
        assert_eq!(ncols, 2);
        assert_eq!(nrows, 3);
        assert_eq!((placed[0].row, placed[0].col), (0, 0));
        assert_eq!((placed[3].row, placed[3].col), (1, 1));
        assert_eq!((placed[4].row, placed[4].col), (2, 0));
    }

    #[test]
    fn column_flow_fills_columns_top_to_bottom() {
        // `grid-auto-flow: column` with two explicit rows: auto items
        // fill column 0 top-to-bottom, then move to column 1.
        let specs: Vec<ItemSpec> = (0..5).map(auto_spec).collect();
        let (placed, ncols, nrows) = place_items(&specs, 0, 2, false, true);
        assert_eq!((ncols, nrows), (3, 2));
        assert_eq!((placed[0].row, placed[0].col), (0, 0));
        assert_eq!((placed[1].row, placed[1].col), (1, 0));
        assert_eq!((placed[2].row, placed[2].col), (0, 1));
        assert_eq!((placed[3].row, placed[3].col), (1, 1));
        assert_eq!((placed[4].row, placed[4].col), (0, 2));
    }

    #[test]
    fn column_dense_backfills_column_holes() {
        // Column flow with a pinned item: sparse leaves the hole above
        // it; dense backfills it in column-major order.
        let pinned = |node: usize| ItemSpec {
            node,
            row: GridPlacement {
                start: GridLine::Line(2),
                end: GridLine::Line(3),
            },
            col: GridPlacement {
                start: GridLine::Auto,
                end: GridLine::Auto,
            },
        };
        let specs = vec![pinned(0), auto_spec(1), auto_spec(2)];
        // Sparse column flow: item 1 goes to (0, 1), item 2 to (1, 1).
        let (placed, _, _) = place_items(&specs, 0, 2, false, true);
        assert_eq!((placed[0].row, placed[0].col), (1, 0));
        assert_eq!((placed[1].row, placed[1].col), (0, 1));
        assert_eq!((placed[2].row, placed[2].col), (1, 1));
        // Dense column flow: item 1 backfills (0, 0).
        let (placed, _, _) = place_items(&specs, 0, 2, true, true);
        assert_eq!((placed[1].row, placed[1].col), (0, 0));
        assert_eq!((placed[2].row, placed[2].col), (0, 1));
    }

    #[test]
    fn auto_placement_wraps_rows_with_no_explicit_columns() {
        // `grid-template-columns: none` (the default): every auto item
        // wraps to a fresh implicit row instead of piling into row 0.
        let specs: Vec<ItemSpec> = (0..3).map(auto_spec).collect();
        let (placed, ncols, nrows) = place_items(&specs, 0, 0, false, false);
        assert_eq!((ncols, nrows), (1, 3));
        assert_eq!((placed[0].row, placed[0].col), (0, 0));
        assert_eq!((placed[1].row, placed[1].col), (1, 0));
        assert_eq!((placed[2].row, placed[2].col), (2, 0));
    }

    #[test]
    fn definite_columns_pin_and_grow_implicit_rows() {
        // Two items pinned to column 1 (zero-based 0); the second must
        // drop to the next row.
        let pin_col = |node: usize| ItemSpec {
            node,
            row: GridPlacement {
                start: GridLine::Auto,
                end: GridLine::Auto,
            },
            col: GridPlacement {
                start: GridLine::Line(1),
                end: GridLine::Line(2),
            },
        };
        let specs = vec![pin_col(0), pin_col(1)];
        let (placed, ncols, nrows) = place_items(&specs, 2, 0, false, false);
        assert_eq!((placed[0].row, placed[0].col), (0, 0));
        assert_eq!((placed[1].row, placed[1].col), (1, 0));
        assert_eq!((ncols, nrows), (2, 2));
    }

    #[test]
    fn dense_with_no_explicit_columns_backfills_rows() {
        // No explicit columns: auto items stack vertically, one per
        // implicit row. Sparse leaves rows 0..1 empty after the pinned
        // item; dense backfills them instead of piling into row 0.
        let pinned = |node: usize| ItemSpec {
            node,
            row: GridPlacement {
                start: GridLine::Line(2),
                end: GridLine::Line(3),
            },
            col: GridPlacement {
                start: GridLine::Auto,
                end: GridLine::Auto,
            },
        };
        let specs = vec![pinned(0), auto_spec(1), auto_spec(2)];
        let (placed, ncols, nrows) = place_items(&specs, 0, 0, false, false);
        assert_eq!((placed[0].row, placed[0].col), (1, 0));
        assert_eq!((placed[1].row, placed[1].col), (2, 0));
        assert_eq!((placed[2].row, placed[2].col), (3, 0));
        assert_eq!((ncols, nrows), (1, 4));
        let (placed, ncols, nrows) = place_items(&specs, 0, 0, true, false);
        assert_eq!((placed[0].row, placed[0].col), (1, 0));
        assert_eq!((placed[1].row, placed[1].col), (0, 0));
        assert_eq!((placed[2].row, placed[2].col), (2, 0));
        assert_eq!((ncols, nrows), (1, 3));
    }

    #[test]
    fn dense_flow_parses_keywords() {
        assert!(dense_flow(Some("dense")));
        assert!(dense_flow(Some("row dense")));
        assert!(dense_flow(Some("dense column")));
        assert!(dense_flow(Some("DENSE")));
        assert!(dense_flow(Some("  dense  ")));
        assert!(!dense_flow(Some("row")));
        assert!(!dense_flow(Some("column")));
        assert!(!dense_flow(Some("sparse")));
        assert!(!dense_flow(None));
        assert!(!dense_flow(Some("")));
    }

    #[test]
    fn dense_backfills_holes_left_by_definite_items() {
        // Two explicit columns. Item 0 is pinned to row 2 (`2 / 3`,
        // zero-based row 1); items 1..3 are fully automatic.
        let pinned = |node: usize| ItemSpec {
            node,
            row: GridPlacement {
                start: GridLine::Line(2),
                end: GridLine::Line(3),
            },
            col: GridPlacement {
                start: GridLine::Auto,
                end: GridLine::Auto,
            },
        };
        let specs = vec![pinned(0), auto_spec(1), auto_spec(2), auto_spec(3)];
        // Sparse (the default): the cursor never backtracks, so row 0
        // stays empty and everything piles up after the pinned item.
        let (placed, ncols, nrows) = place_items(&specs, 2, 0, false, false);
        assert_eq!((placed[0].row, placed[0].col), (1, 0));
        assert_eq!((placed[1].row, placed[1].col), (1, 1));
        assert_eq!((placed[2].row, placed[2].col), (2, 0));
        assert_eq!((placed[3].row, placed[3].col), (2, 1));
        assert_eq!((ncols, nrows), (2, 3));
        // Dense: items 1 and 2 backfill row 0; item 3 takes the free
        // slot after the pinned item.
        let (placed, ncols, nrows) = place_items(&specs, 2, 0, true, false);
        assert_eq!((placed[0].row, placed[0].col), (1, 0));
        assert_eq!((placed[1].row, placed[1].col), (0, 0));
        assert_eq!((placed[2].row, placed[2].col), (0, 1));
        assert_eq!((placed[3].row, placed[3].col), (1, 1));
        assert_eq!((ncols, nrows), (2, 2));
    }

    #[test]
    fn dense_backfills_hole_beside_spanning_item() {
        // Three explicit columns. Item 0 spans columns 2-3 of row 0
        // (`grid-column: 2 / 4`), leaving (0, 0) empty; items 1..3 are
        // fully automatic.
        let span_cols = |node: usize| ItemSpec {
            node,
            row: GridPlacement {
                start: GridLine::Auto,
                end: GridLine::Auto,
            },
            col: GridPlacement {
                start: GridLine::Line(2),
                end: GridLine::Line(4),
            },
        };
        let specs = vec![span_cols(0), auto_spec(1), auto_spec(2), auto_spec(3)];
        // Sparse: the cursor advances past the span, so (0, 0) is never
        // filled.
        let (placed, _, _) = place_items(&specs, 3, 0, false, false);
        assert_eq!((placed[0].row, placed[0].col), (0, 1));
        assert_eq!(placed[0].colspan, 2);
        assert_eq!((placed[1].row, placed[1].col), (1, 0));
        assert_eq!((placed[2].row, placed[2].col), (1, 1));
        assert_eq!((placed[3].row, placed[3].col), (1, 2));
        // Dense: item 1 backfills (0, 0).
        let (placed, _, _) = place_items(&specs, 3, 0, true, false);
        assert_eq!((placed[0].row, placed[0].col), (0, 1));
        assert_eq!((placed[1].row, placed[1].col), (0, 0));
        assert_eq!((placed[2].row, placed[2].col), (1, 0));
        assert_eq!((placed[3].row, placed[3].col), (1, 1));
    }

    #[test]
    fn auto_repeat_count_fits_strides() {
        // (500 + 10) / (100 + 10) = 4.54… → 4 tracks.
        assert_eq!(auto_repeat_count(Some(500.0), 100.0, 10.0), 4);
        // Exact fit, no gap: 3 tracks.
        assert_eq!(auto_repeat_count(Some(300.0), 100.0, 0.0), 3);
        // A container smaller than one track still yields one track.
        assert_eq!(auto_repeat_count(Some(50.0), 100.0, 10.0), 1);
        // Indefinite container or degenerate stride: one repetition.
        assert_eq!(auto_repeat_count(None, 100.0, 10.0), 1);
        assert_eq!(auto_repeat_count(Some(500.0), 0.0, 0.0), 1);
        assert_eq!(auto_repeat_count(Some(-10.0), 100.0, 10.0), 1);
    }

    #[test]
    fn auto_repeat_count_uses_definite_track_size() {
        use bowser_css::LengthUnit;
        let len = |px: f32| CssValue::Length(px, LengthUnit::Px);
        let fr1 = CssValue::Fr(1.0);
        // Fixed track: (500 + 10) / (100 + 10) → 4.
        assert_eq!(
            auto_repeat_count_for(&[len(100.0)], 16.0, 16.0, 800.0, 600.0, Some(500.0), 10.0),
            4
        );
        // `minmax()`: an indefinite maximum falls back to the definite
        // minimum (100px → 4 tracks, like above).
        let mm = CssValue::MinMax(Box::new(len(100.0)), Box::new(fr1.clone()));
        assert_eq!(
            auto_repeat_count_for(&[mm], 16.0, 16.0, 800.0, 600.0, Some(500.0), 10.0),
            4
        );
        // `minmax()`: a definite maximum counts (200px → 2 tracks).
        let mm = CssValue::MinMax(Box::new(len(50.0)), Box::new(len(200.0)));
        assert_eq!(
            auto_repeat_count_for(&[mm], 16.0, 16.0, 800.0, 600.0, Some(500.0), 10.0),
            2
        );
        // Indefinite track (`1fr`) or indefinite container: one repetition.
        assert_eq!(
            auto_repeat_count_for(&[fr1], 16.0, 16.0, 800.0, 600.0, Some(500.0), 10.0),
            1
        );
        assert_eq!(
            auto_repeat_count_for(&[len(100.0)], 16.0, 16.0, 800.0, 600.0, None, 10.0),
            1
        );
        // Nested lists (the parser expands an inner integer `repeat()` to
        // a nested list) count as one multi-track repetition: 2 tracks of
        // 100px each → floor((500 + 10) / (200 + 2 * 10)) = 2
        // repetitions, 4 tracks total.
        let nested = CssValue::List(vec![len(100.0), len(100.0)]);
        assert_eq!(
            auto_repeat_count_for(&[nested], 16.0, 16.0, 800.0, 600.0, Some(500.0), 10.0),
            2
        );
    }

    #[test]
    fn auto_repeat_count_sums_whole_track_list() {
        use bowser_css::LengthUnit;
        let len = |px: f32| CssValue::Length(px, LengthUnit::Px);
        let count = |tracks: &[CssValue]| {
            auto_repeat_count_for(tracks, 16.0, 16.0, 800.0, 600.0, Some(500.0), 10.0)
        };
        // Regression (review): `repeat(auto-fill, 100px 50px)` in a
        // 500px container with 10px gap is 3 repetitions (6 tracks):
        // floor((500 + 10) / (150 + 2 * 10)) = 3. The old first-track
        // scan computed floor(510 / 110) = 4 (8 tracks = 870px),
        // overflowing the container.
        assert_eq!(count(&[len(100.0), len(50.0)]), 3);
        // Three-track mixed listing: floor((500 + 10) / (175 + 3 * 10))
        // = floor(510 / 205) = 2 repetitions (6 tracks).
        assert_eq!(count(&[len(100.0), len(50.0), len(25.0)]), 2);
        // One repetition exactly fills: (500 + 10) / (500 + 10) = 1.
        assert_eq!(count(&[len(400.0), len(100.0)]), 1);
        // Zero gap: floor(500 / 150) = 3.
        assert_eq!(
            auto_repeat_count_for(
                &[len(100.0), len(50.0)],
                16.0,
                16.0,
                800.0,
                600.0,
                Some(500.0),
                0.0
            ),
            3
        );
        // An indefinite track anywhere in the list falls back to a
        // single repetition (documented simplification).
        assert_eq!(count(&[len(100.0), CssValue::Fr(1.0)]), 1);
        // Empty track list: a single repetition.
        assert_eq!(count(&[]), 1);
    }

    #[test]
    fn push_template_item_expands_auto_repeat_descriptors() {
        use bowser_css::{LengthUnit, RepeatMode};
        // Parser contract shape: the dedicated `Repeat` descriptor.
        let descriptor = |mode: RepeatMode| CssValue::Repeat {
            mode,
            tracks: vec![CssValue::Length(100.0, LengthUnit::Px)],
        };
        let expand = |v: &CssValue| {
            let mut out = Vec::new();
            let mut collapse = Vec::new();
            push_template_item(
                &mut out,
                &mut collapse,
                v,
                16.0,
                16.0,
                800.0,
                600.0,
                Some(500.0),
                10.0,
            );
            (out, collapse)
        };
        // `auto-fill`: 4 tracks, none collapsible.
        let (tracks, collapse) = expand(&descriptor(RepeatMode::Fill));
        assert_eq!(tracks.len(), 4);
        assert_eq!(collapse.len(), 4);
        assert!(collapse.iter().all(|&c| !c));
        assert!(tracks.iter().all(|t| matches!(t, TrackSize::Fixed(_))));
        // `auto-fit`: 4 tracks, all collapsible when empty.
        let (tracks, collapse) = expand(&descriptor(RepeatMode::Fit));
        assert_eq!(tracks.len(), 4);
        assert!(collapse.iter().all(|&c| c));
        assert!(tracks.iter().all(|t| matches!(t, TrackSize::Fixed(_))));
        // Indefinite container: a single repetition.
        let mut out = Vec::new();
        let mut collapse = Vec::new();
        push_template_item(
            &mut out,
            &mut collapse,
            &descriptor(RepeatMode::Fill),
            16.0,
            16.0,
            800.0,
            600.0,
            None,
            10.0,
        );
        assert_eq!(out.len(), 1);
        // A bare `auto-fill` keyword that is NOT the descriptor shape
        // keeps the old behavior (degrades to `auto`, like any unknown
        // keyword).
        let mut out = Vec::new();
        let mut collapse = Vec::new();
        push_template_item(
            &mut out,
            &mut collapse,
            &CssValue::Keyword("auto-fill".into()),
            16.0,
            16.0,
            800.0,
            600.0,
            Some(500.0),
            10.0,
        );
        assert!(matches!(out.as_slice(), [TrackSize::Auto]));
        assert_eq!(collapse, vec![false]);
    }

    #[test]
    fn push_template_item_expands_multi_track_auto_repeat() {
        use bowser_css::{LengthUnit, RepeatMode};
        // Regression (review): `repeat(auto-fill, 100px 50px)` in a
        // 500px container with 10px gap must yield 3 repetitions × 2
        // tracks = 6 tracks (510px of stride budget / 170px per
        // repetition), not 4 × 2 = 8 tracks from the old first-track
        // scan. Parser contract shape: the dedicated `Repeat`
        // descriptor.
        let descriptor = CssValue::Repeat {
            mode: RepeatMode::Fill,
            tracks: vec![
                CssValue::Length(100.0, LengthUnit::Px),
                CssValue::Length(50.0, LengthUnit::Px),
            ],
        };
        let mut out = Vec::new();
        let mut collapse = Vec::new();
        push_template_item(
            &mut out,
            &mut collapse,
            &descriptor,
            16.0,
            16.0,
            800.0,
            600.0,
            Some(500.0),
            10.0,
        );
        assert_eq!(out.len(), 6);
        assert_eq!(collapse, vec![false; 6]);
        let widths: Vec<f32> = out
            .iter()
            .map(|t| match t {
                TrackSize::Fixed(s) => s.resolve(16.0, 16.0, 800.0, 600.0, 500.0).unwrap_or(-1.0),
                _ => -1.0,
            })
            .collect();
        assert_eq!(widths, vec![100.0, 50.0, 100.0, 50.0, 100.0, 50.0]);
        // `auto-fit` marks every expanded track collapsible.
        let fit_descriptor = CssValue::Repeat {
            mode: RepeatMode::Fit,
            tracks: vec![
                CssValue::Length(100.0, LengthUnit::Px),
                CssValue::Length(50.0, LengthUnit::Px),
            ],
        };
        let mut out = Vec::new();
        let mut collapse = Vec::new();
        push_template_item(
            &mut out,
            &mut collapse,
            &fit_descriptor,
            16.0,
            16.0,
            800.0,
            600.0,
            Some(500.0),
            10.0,
        );
        assert_eq!(out.len(), 6);
        assert!(collapse.iter().all(|&c| c));
    }

    #[test]
    fn parse_line_side_rejects_hostile_numbers() {
        let num = |n: f32| CssValue::Number(n);
        // CSS requires <integer>; 0 is never a valid line.
        assert_eq!(parse_line_side(&num(0.0)), GridLine::Auto);
        assert_eq!(parse_line_side(&num(1.5)), GridLine::Auto);
        assert_eq!(parse_line_side(&num(-2.5)), GridLine::Auto);
        // Out-of-range magnitudes clamp instead of saturating `as i32`
        // to `i32::MAX` (the cascade rejects them outright; this is the
        // layout-side backstop).
        assert_eq!(parse_line_side(&num(3e9)), GridLine::Line(MAX_GRID_LINE));
        assert_eq!(parse_line_side(&num(-3e9)), GridLine::Line(-MAX_GRID_LINE));
        assert_eq!(parse_line_side(&num(7.0)), GridLine::Line(7));
        // Invalid spans degrade to auto; huge spans clamp.
        let span = |n: f32| {
            parse_line_side(&CssValue::List(vec![
                CssValue::Keyword("span".into()),
                CssValue::Number(n),
            ]))
        };
        assert_eq!(span(0.0), GridLine::Auto);
        assert_eq!(span(2.5), GridLine::Auto);
        assert_eq!(span(1e9), GridLine::Span(1000));
        assert_eq!(span(3.0), GridLine::Span(3));
    }

    #[test]
    fn definite_range_clamps_spans_and_negation() {
        // `grid-row: 1/100000` would allocate ~1e5 rows without the cap.
        let p = GridPlacement {
            start: GridLine::Line(1),
            end: GridLine::Line(100_000),
        };
        assert_eq!(definite_range(p, 4), Some((0, 1000)));
        let p = GridPlacement {
            start: GridLine::Line(1),
            end: GridLine::Span(5000),
        };
        assert_eq!(definite_range(p, 4), Some((0, 1000)));
        // `i32::MIN` must not overflow the negation in the
        // negative-line branch.
        let p = GridPlacement {
            start: GridLine::Line(i32::MIN),
            end: GridLine::Line(2),
        };
        assert_eq!(definite_range(p, 4), Some((0, 1)));
    }

    #[test]
    fn place_items_stays_bounded_on_hostile_lines() {
        // Even if hostile lines bypassed the cascade, the placement grid
        // stays bounded: lines clamp to ±10000, spans to 1000.
        let spec = |node: usize| ItemSpec {
            node,
            row: GridPlacement {
                start: GridLine::Line(1),
                end: GridLine::Line(MAX_GRID_LINE),
            },
            col: GridPlacement {
                start: GridLine::Line(1),
                end: GridLine::Line(MAX_GRID_LINE),
            },
        };
        let (placed, ncols, nrows) = place_items(&[spec(0)], 0, 0, false, false);
        assert_eq!((placed[0].row, placed[0].col), (0, 0));
        assert_eq!((placed[0].rowspan, placed[0].colspan), (1000, 1000));
        assert!(nrows <= 1001 && ncols <= 1001, "{nrows}x{ncols}");
    }

    #[test]
    fn place_items_far_start_lines_stay_sparse() {
        // `grid-row: 10000/10001; grid-column: 10000/10001`: start lines
        // are only bounded at ±10000 while spans cap at 1000, so the old
        // dense `Vec<Vec<bool>>` occupancy grid zero-filled ~121MB
        // (11000×11000 bools) for this single item. The sparse set holds
        // only the covered cell. The lines bypass the cascade here (it
        // would drop `10001` as out of range), exactly like the hostile
        // test above: this is the defense-in-depth path for a hand-built
        // `ComputedStyle` or any future caller.
        let spec = |node: usize| ItemSpec {
            node,
            row: GridPlacement {
                start: GridLine::Line(10000),
                end: GridLine::Line(10001),
            },
            col: GridPlacement {
                start: GridLine::Line(10000),
                end: GridLine::Line(10001),
            },
        };
        let (placed, ncols, nrows) = place_items(&[spec(0)], 0, 0, false, false);
        assert_eq!((placed[0].row, placed[0].col), (9999, 9999));
        assert_eq!((placed[0].rowspan, placed[0].colspan), (1, 1));
        // The grid extents still report the far placement (implicit
        // tracks are cheap 1-D vectors downstream); only the occupancy
        // had to stay bounded, which the sparse set guarantees by
        // construction.
        assert_eq!((ncols, nrows), (10000, 10000));
    }

    #[test]
    fn hostile_grid_lines_layout_completes() {
        use crate::{FixedFontMetrics, Size, layout};
        // Both declarations are invalid at computed-value time and
        // dropped; the item auto-places and layout must finish with
        // finite geometry instead of allocating a ~1e12-cell grid.
        let (doc, _) = bowser_dom::Document::parse_html(
            "<div class=\"g\"><div class=\"a\">x</div></div>".as_bytes(),
        );
        let ss = bowser_css::parse_stylesheet(
            ".g { display: grid; grid-template-columns: 100px; } \
             .a { grid-row: 1/1000000; grid-column: 3000000000; }",
        );
        let l = layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics);
        let g = &l.children[0].children[0];
        let a = &g.children[0];
        for r in [&g.rect, &a.rect] {
            assert!(r.w.is_finite() && r.h.is_finite() && r.x.is_finite() && r.y.is_finite());
        }
    }

    #[test]
    fn stylesheet_auto_fill_expands_single_track_descriptor() {
        use crate::{FixedFontMetrics, Size, layout};
        // Regression: a single-track `repeat(auto-fill, …)` descriptor
        // used to be flattened by the cascade's comma-group
        // normalization into the byte-identical shape of the invalid
        // bare `auto-fill <track>` listing. The descriptor is now a
        // dedicated `CssValue::Repeat` variant, so the track parser
        // expands it unambiguously: floor((350 + 10) / (100 + 10)) = 3
        // repetitions (the second item sits at x=120, not x=10).
        let (doc, _) = bowser_dom::Document::parse_html(
            "<div class=\"g\"><div class=\"a\"></div><div class=\"b\"></div><div class=\"c\"></div></div>"
                .as_bytes(),
        );
        let ss = bowser_css::parse_stylesheet(
            ".g { display: grid; width: 350px; column-gap: 10px; \
             grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); } \
             .a, .b, .c { height: 10px; }",
        );
        let l = layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics);
        let g = &l.children[0].children[0];
        let (a, b, c) = (&g.children[0], &g.children[1], &g.children[2]);
        // floor((350 + 10) / (100 + 10)) = 3 repetitions; each
        // minmax(100px, 1fr) takes base 100 then shares the 30px of free
        // space via 1fr, ending at 110px.
        let close = |a: f32, b: f32| (a - b).abs() < 0.01;
        assert!(close(a.rect.x, 0.0), "a.x={}", a.rect.x);
        assert!(close(a.rect.w, 110.0), "a.w={}", a.rect.w);
        assert!(close(b.rect.x, 120.0), "b.x={}", b.rect.x);
        assert!(close(c.rect.x, 240.0), "c.x={}", c.rect.x);
    }

    #[test]
    fn stylesheet_auto_fill_multi_track_counts_whole_list() {
        use crate::{FixedFontMetrics, Size, layout};
        // Regression (review): `repeat(auto-fill, 100px 50px)` in a
        // 500px container with 10px gap repeats 3 times (6 tracks:
        // floor((500 + 10) / (150 + 2 * 10))), exactly filling the
        // container (3 * 150 + 5 * 10 = 500). The old first-track scan
        // produced 4 repetitions (8 tracks, 870px), overflowing it.
        let (doc, _) = bowser_dom::Document::parse_html(
            "<div class=\"g\"><div class=\"a\"></div><div class=\"b\"></div><div class=\"c\"></div>\
             <div class=\"d\"></div><div class=\"e\"></div><div class=\"f\"></div></div>"
                .as_bytes(),
        );
        let ss = bowser_css::parse_stylesheet(
            ".g { display: grid; width: 500px; column-gap: 10px; \
             grid-template-columns: repeat(auto-fill, 100px 50px); } \
             .a, .b, .c, .d, .e, .f { height: 10px; }",
        );
        let l = layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics);
        let g = &l.children[0].children[0];
        assert_eq!(g.children.len(), 6);
        let close = |a: f32, b: f32| (a - b).abs() < 0.01;
        let expected = [
            (0.0, 100.0),
            (110.0, 50.0),
            (170.0, 100.0),
            (280.0, 50.0),
            (340.0, 100.0),
            (450.0, 50.0),
        ];
        for (child, &(x, w)) in g.children.iter().zip(expected.iter()) {
            assert!(close(child.rect.x, x), "x={} want {x}", child.rect.x);
            assert!(close(child.rect.w, w), "w={} want {w}", child.rect.w);
        }
    }

    #[test]
    fn stylesheet_plain_listing_flattens_whole_value() {
        use crate::{FixedFontMetrics, Size, layout};
        // `parse_tracks` passes the whole listing value through
        // `push_template_item` in one call (its stack flattens plain
        // lists); a plain multi-track listing must lay out exactly as
        // before that refactor.
        let (doc, _) = bowser_dom::Document::parse_html(
            "<div class=\"g\"><div class=\"a\"></div><div class=\"b\"></div><div class=\"c\"></div></div>"
                .as_bytes(),
        );
        let ss = bowser_css::parse_stylesheet(
            ".g { display: grid; width: 400px; grid-template-columns: 100px 1fr 2fr; } \
             .a, .b, .c { height: 10px; }",
        );
        let l = layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics);
        let g = &l.children[0].children[0];
        let (a, b, c) = (&g.children[0], &g.children[1], &g.children[2]);
        let close = |a: f32, b: f32| (a - b).abs() < 0.01;
        assert!(close(a.rect.w, 100.0), "a.w={}", a.rect.w);
        assert!(close(b.rect.x, 100.0), "b.x={}", b.rect.x);
        assert!(close(b.rect.w, 100.0), "b.w={}", b.rect.w);
        assert!(close(c.rect.x, 200.0), "c.x={}", c.rect.x);
        assert!(close(c.rect.w, 200.0), "c.w={}", c.rect.w);
    }

    #[test]
    fn minmax_bound_maps_breadths() {
        use bowser_css::LengthUnit;
        assert!(matches!(
            minmax_bound(&CssValue::Length(10.0, LengthUnit::Px)),
            MinMaxBound::Fixed(_)
        ));
        assert!(matches!(
            minmax_bound(&CssValue::Percentage(50.0)),
            MinMaxBound::Fixed(_)
        ));
        assert!(matches!(minmax_bound(&CssValue::Fr(2.0)), MinMaxBound::Fr(f) if f == 2.0));
        for kw in ["auto", "min-content", "max-content"] {
            assert!(
                matches!(
                    minmax_bound(&CssValue::Keyword(kw.into())),
                    MinMaxBound::Content
                ),
                "{kw}"
            );
        }
        // Unrecognized values degrade to content-based (defense in depth;
        // the parser already rejected them).
        assert!(matches!(
            minmax_bound(&CssValue::Keyword("red".into())),
            MinMaxBound::Content
        ));
    }

    #[test]
    fn clamp_track_min_wins_over_max() {
        assert_eq!(clamp_track(50.0, 100.0, 200.0), 100.0);
        assert_eq!(clamp_track(250.0, 100.0, 200.0), 200.0);
        assert_eq!(clamp_track(150.0, 100.0, 200.0), 150.0);
        // A minimum larger than the maximum: the maximum is ignored and
        // the track keeps its minimum (grid §7.2.4).
        assert_eq!(clamp_track(50.0, 200.0, 100.0), 200.0);
        assert_eq!(clamp_track(500.0, 200.0, 100.0), 200.0);
    }

    #[test]
    fn self_align_of_maps_keywords() {
        // `None` (no style) always defers to the container value; the
        // keyword mapping itself is covered end-to-end by the layout
        // tests in `tests.rs` (start/end/center/stretch/auto/normal).
        assert_eq!(
            self_align_of(None, props::JUSTIFY_SELF, ItemAlign::Center),
            ItemAlign::Center
        );
        assert_eq!(
            self_align_of(None, props::ALIGN_SELF, ItemAlign::End),
            ItemAlign::End
        );
        assert_eq!(
            self_align_of(None, props::JUSTIFY_SELF, ItemAlign::Stretch),
            ItemAlign::Stretch
        );
    }

    #[test]
    fn hostile_far_start_lines_layout_completes() {
        use crate::{FixedFontMetrics, Size, layout};
        // Largest line numbers the cascade accepts: the item lands at
        // (9998, 9998) and layout must finish with finite geometry
        // instead of allocating a ~100MB dense occupancy grid.
        let (doc, _) = bowser_dom::Document::parse_html(
            "<div class=\"g\"><div class=\"a\">x</div></div>".as_bytes(),
        );
        let ss = bowser_css::parse_stylesheet(
            ".g { display: grid; grid-template-columns: 100px; } \
             .a { grid-row: 9999/10000; grid-column: 9999/10000; }",
        );
        let l = layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics);
        let g = &l.children[0].children[0];
        let a = &g.children[0];
        for r in [&g.rect, &a.rect] {
            assert!(r.w.is_finite() && r.h.is_finite() && r.x.is_finite() && r.y.is_finite());
        }
    }

    fn grid_e2e(html: &str, css: &str) -> crate::LayoutBox {
        use crate::{FixedFontMetrics, Size, layout};
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics)
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// Box tree plus the grid node, for tests that measure a grid's
    /// intrinsic width directly without running layout.
    fn forest_grid(html: &str, css: &str) -> (crate::boxes::Forest, usize) {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let ua = bowser_css::default_ua_stylesheet();
        let Some((forest, root)) = crate::boxes::build_box_tree(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
        ) else {
            panic!("no root box");
        };
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::Grid {
                return (forest, i);
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        panic!("no grid box");
    }

    #[test]
    fn spanning_item_area_includes_distributed_spacing() {
        // `justify-content: space-between` puts 200px between the two
        // 100px tracks in a 400px container; the item spanning both must
        // fill the 400px area, not just the 200px of raw track sizes.
        let l = grid_e2e(
            r#"<div class="g"><div class="a">x</div></div>"#,
            ".g { display: grid; grid-template-columns: 100px 100px; \
             justify-content: space-between; width: 400px; } \
             .a { grid-column: 1 / 3; height: 20px; }",
        );
        let g = &l.children[0].children[0];
        let a = &g.children[0];
        assert!(approx(a.rect.x, g.rect.x), "x={}", a.rect.x);
        assert!(approx(a.rect.w, 400.0), "w={}", a.rect.w);
    }

    #[test]
    fn floated_child_is_still_a_grid_item() {
        // `float` has no effect on a grid item (css-grid-1 §5.1): the
        // floated child takes the first cell instead of floating at the
        // content-box origin outside grid placement.
        let l = grid_e2e(
            r#"<div class="g"><div class="f"></div><div class="a"></div></div>"#,
            ".g { display: grid; grid-template-columns: 100px 100px; } \
             .f { float: left; height: 10px; } \
             .a { height: 20px; }",
        );
        let g = &l.children[0].children[0];
        let (f, a) = (&g.children[0], &g.children[1]);
        assert!(approx(f.rect.x, g.rect.x), "fx={}", f.rect.x);
        assert!(approx(f.rect.w, 100.0), "fw={}", f.rect.w);
        assert!(approx(a.rect.x, g.rect.x + 100.0), "ax={}", a.rect.x);
    }

    #[test]
    fn lower_float_narrows_grid_item_area() {
        // As with flex: a preceding float whose margin box starts below
        // the grid container's top still overlaps the container's body,
        // so the track area narrows past it (CSS 2.1 §9.5). Grid items
        // themselves still ignore `float`/`clear`.
        let l = grid_e2e(
            r#"<div class="f1"></div><div class="f2"></div><div class="g"><div class="a"></div></div>"#,
            ".f1 { float: left; width: 400px; height: 60px; } \
             .f2 { float: left; width: 700px; height: 60px; clear: left; } \
             .g { display: grid; grid-template-columns: 100px; height: 200px; } \
             .a { height: 50px; }",
        );
        let kids = &l.children[0].children;
        let (f2, g) = (&kids[1], &kids[2]);
        let a = &g.children[0];
        let overlaps = a.rect.x < f2.rect.x + f2.rect.w
            && f2.rect.x < a.rect.x + a.rect.w
            && a.rect.y < f2.rect.y + f2.rect.h
            && f2.rect.y < a.rect.y + a.rect.h;
        assert!(
            !overlaps,
            "grid item {:?} overlaps lower float {:?}",
            a.rect, f2.rect
        );
        assert!(approx(a.rect.x, 700.0), "a.x = {}", a.rect.x);
    }

    #[test]
    fn subgrid_spans_subset_of_parent_tracks() {
        // Parent: three 100px tracks in a 400px container with
        // `space-between` → tracks at +0, +150, +300. The subgrid spans
        // the first two, so its items must sit at +0 and +150 on the
        // parent's used tracks — not at +0/+100 from its own origin.
        let l = grid_e2e(
            r#"<div class="g"><div class="sg"><div class="i"></div><div class="i"></div></div></div>"#,
            ".g { display: grid; grid-template-columns: 100px 100px 100px; \
             justify-content: space-between; width: 400px; } \
             .sg { display: grid; grid-template-columns: subgrid; grid-column: 1 / 3; } \
             .i { height: 20px; }",
        );
        let g = &l.children[0].children[0];
        let sg = &g.children[0];
        let (i0, i1) = (&sg.children[0], &sg.children[1]);
        assert!(approx(i0.rect.x, g.rect.x), "x0={}", i0.rect.x);
        assert!(approx(i0.rect.w, 100.0), "w0={}", i0.rect.w);
        assert!(approx(i1.rect.x, g.rect.x + 150.0), "x1={}", i1.rect.x);
        assert!(approx(i1.rect.w, 100.0), "w1={}", i1.rect.w);
    }

    #[test]
    fn subgrid_with_non_grid_parent_degrades_to_none() {
        // `grid-template-columns: subgrid` on a box whose parent is not
        // a grid container behaves as `none`: the tracks come from the
        // implicit grid, not from some further ancestor's listing — so
        // with no explicit columns the two items stack vertically instead
        // of sitting side by side in the ancestor's 100px tracks.
        let l = grid_e2e(
            r#"<div class="g"><div class="wrap"><div class="sg"><div class="i"></div><div class="i"></div></div></div></div>"#,
            ".g { display: grid; grid-template-columns: 100px 100px; } \
             .sg { display: grid; grid-template-columns: subgrid; } \
             .i { height: 20px; }",
        );
        let g = &l.children[0].children[0];
        let sg = &g.children[0].children[0];
        let (i0, i1) = (&sg.children[0], &sg.children[1]);
        assert!(
            approx(i1.rect.x, i0.rect.x),
            "x1={} x0={}",
            i1.rect.x,
            i0.rect.x
        );
        assert!(
            approx(i1.rect.y, i0.rect.y + 20.0),
            "y1={} y0={}",
            i1.rect.y,
            i0.rect.y
        );
    }

    #[test]
    fn subgrid_rows_freeze_parent_definite_tracks() {
        // Parent rows are `50%` of a definite 400px height with a 10px
        // row gap. The subgrid spans both rows: its tracks must freeze
        // at the parent's used 200px (not degrade to content-sized:
        // the subgrid itself is auto-height, so the old parsed-listing
        // fallback sized them from the subgrid's own content).
        let l = grid_e2e(
            r#"<div class="g"><div class="sg"><div class="a"></div><div class="b"></div></div></div>"#,
            ".g { display: grid; grid-template-columns: 300px; \
             grid-template-rows: 50% 50%; height: 400px; row-gap: 10px; } \
             .sg { display: grid; grid-template-rows: subgrid; grid-row: 1 / 3; }",
        );
        let g = &l.children[0].children[0];
        let sg = &g.children[0];
        let (a, b) = (&sg.children[0], &sg.children[1]);
        assert!(approx(a.rect.y, g.rect.y), "ay={}", a.rect.y);
        assert!(approx(a.rect.h, 200.0), "ah={}", a.rect.h);
        assert!(approx(b.rect.y, g.rect.y + 210.0), "by={}", b.rect.y);
        assert!(approx(b.rect.h, 200.0), "bh={}", b.rect.h);
    }

    #[test]
    fn nested_subgrid_rows_align_with_parent_tracks() {
        // Two levels of row subgrids over `100px 60px` parent rows with
        // a 10px gap. The inner subgrid inherits the parent's row gap
        // (its own is zero), so its second item sits at +110, and the
        // definite 100px/60px sizes chain through the middle subgrid
        // exactly.
        let l = grid_e2e(
            r#"<div class="g"><div class="s1"><div class="s2"><div class="x"></div><div class="y"></div></div></div></div>"#,
            ".g { display: grid; grid-template-columns: 200px; \
             grid-template-rows: 100px 60px; row-gap: 10px; } \
             .s1 { display: grid; grid-template-rows: subgrid; grid-row: 1 / 3; } \
             .s2 { display: grid; grid-template-rows: subgrid; grid-row: 1 / 3; }",
        );
        let g = &l.children[0].children[0];
        let s2 = &g.children[0].children[0];
        let (x, y) = (&s2.children[0], &s2.children[1]);
        assert!(approx(x.rect.y, g.rect.y), "xy={}", x.rect.y);
        assert!(approx(x.rect.h, 100.0), "xh={}", x.rect.h);
        assert!(approx(y.rect.y, g.rect.y + 110.0), "yy={}", y.rect.y);
        assert!(approx(y.rect.h, 60.0), "yh={}", y.rect.h);
    }

    #[test]
    fn subgrid_both_axes_align_with_parent() {
        // Both axes subgridded: columns keep the used-geometry behavior
        // (`space-between` puts the parent tracks at +0/+150/+300) while
        // rows now resolve against the parent's definite 50px/70px
        // tracks with the parent's 8px row gap.
        let l = grid_e2e(
            r#"<div class="g"><div class="sg"><div class="i"></div><div class="i"></div><div class="i"></div><div class="i"></div></div></div>"#,
            ".g { display: grid; grid-template-columns: 100px 100px 100px; \
             grid-template-rows: 50px 70px; justify-content: space-between; \
             width: 400px; row-gap: 8px; } \
             .sg { display: grid; grid-template-columns: subgrid; \
             grid-template-rows: subgrid; grid-column: 1 / 3; grid-row: 1 / 3; }",
        );
        let g = &l.children[0].children[0];
        let sg = &g.children[0];
        let (i0, i1, i2, i3) = (
            &sg.children[0],
            &sg.children[1],
            &sg.children[2],
            &sg.children[3],
        );
        assert!(approx(i0.rect.x, g.rect.x), "x0={}", i0.rect.x);
        assert!(approx(i0.rect.w, 100.0), "w0={}", i0.rect.w);
        assert!(approx(i1.rect.x, g.rect.x + 150.0), "x1={}", i1.rect.x);
        assert!(approx(i1.rect.w, 100.0), "w1={}", i1.rect.w);
        assert!(approx(i0.rect.y, g.rect.y), "y0={}", i0.rect.y);
        assert!(approx(i0.rect.h, 50.0), "h0={}", i0.rect.h);
        assert!(approx(i2.rect.x, g.rect.x), "x2={}", i2.rect.x);
        assert!(approx(i2.rect.y, g.rect.y + 58.0), "y2={}", i2.rect.y);
        assert!(approx(i2.rect.h, 70.0), "h2={}", i2.rect.h);
        assert!(approx(i3.rect.x, g.rect.x + 150.0), "x3={}", i3.rect.x);
        assert!(approx(i3.rect.y, g.rect.y + 58.0), "y3={}", i3.rect.y);
    }

    #[test]
    fn subgrid_rows_indefinite_fallback_stays_sane() {
        // Content-sized (`auto`) parent rows cannot resolve before the
        // parent's own phase two, so the subgrid keeps sizing them from
        // its own content — but it still inherits the parent's row gap
        // on the subgridded axis.
        let l = grid_e2e(
            r#"<div class="g"><div class="sg"><div class="a"></div><div class="b"></div></div></div>"#,
            ".g { display: grid; grid-template-columns: 200px; \
             grid-template-rows: auto auto; row-gap: 10px; } \
             .sg { display: grid; grid-template-rows: subgrid; grid-row: 1 / 3; } \
             .a { height: 20px; } .b { height: 30px; }",
        );
        let g = &l.children[0].children[0];
        let sg = &g.children[0];
        let (a, b) = (&sg.children[0], &sg.children[1]);
        for r in [&a.rect, &b.rect, &sg.rect, &g.rect] {
            assert!(r.x.is_finite() && r.y.is_finite() && r.w.is_finite() && r.h.is_finite());
        }
        assert!(approx(a.rect.y, sg.rect.y), "ay={}", a.rect.y);
        assert!(
            approx(b.rect.y, a.rect.y + 30.0),
            "by={} ay={}",
            b.rect.y,
            a.rect.y
        );
    }

    #[test]
    fn subgrid_rows_with_non_grid_parent_degrade_to_none() {
        // `grid-template-rows: subgrid` on a box whose parent is not a
        // grid container behaves as `none`: with no explicit tracks the
        // items stack vertically instead of aligning to the
        // grandparent's 100px rows.
        let l = grid_e2e(
            r#"<div class="g"><div class="wrap"><div class="sg"><div class="i"></div><div class="i"></div></div></div></div>"#,
            ".g { display: grid; grid-template-rows: 100px 100px; \
             grid-template-columns: 200px; } \
             .sg { display: grid; grid-template-rows: subgrid; } \
             .i { height: 20px; }",
        );
        let g = &l.children[0].children[0];
        let sg = &g.children[0].children[0];
        let (i0, i1) = (&sg.children[0], &sg.children[1]);
        assert!(
            approx(i1.rect.x, i0.rect.x),
            "x1={} x0={}",
            i1.rect.x,
            i0.rect.x
        );
        assert!(
            approx(i1.rect.y, i0.rect.y + 20.0),
            "y1={} y0={}",
            i1.rect.y,
            i0.rect.y
        );
    }

    #[test]
    fn normal_content_alignment_stretches_auto_tracks() {
        // `justify-content: normal` (the initial value) behaves as
        // `stretch` (css-grid-1 §10.4): the `auto` track absorbs the
        // leftover 300px instead of leaving it at the end.
        let l = grid_e2e(
            r#"<div class="g"><div class="a"></div><div class="b"></div></div>"#,
            ".g { display: grid; grid-template-columns: 100px auto; width: 400px; } \
             .a, .b { height: 20px; }",
        );
        let g = &l.children[0].children[0];
        let (a, b) = (&g.children[0], &g.children[1]);
        assert!(approx(a.rect.w, 100.0), "aw={}", a.rect.w);
        assert!(approx(b.rect.x, g.rect.x + 100.0), "bx={}", b.rect.x);
        assert!(approx(b.rect.w, 300.0), "bw={}", b.rect.w);
    }

    #[test]
    fn stretch_content_alignment_respects_minmax_maximum() {
        // `stretch` grows content-sized tracks but never past their
        // `minmax()` maximum: the 120px-capped track keeps 120px and the
        // leftover space stays undistributed.
        let l = grid_e2e(
            r#"<div class="g"><div class="a">x</div></div>"#,
            ".g { display: grid; grid-template-columns: minmax(50px, 120px); \
             justify-content: stretch; width: 400px; } \
             .a { height: 20px; }",
        );
        let g = &l.children[0].children[0];
        let a = &g.children[0];
        assert!(approx(a.rect.w, 120.0), "w={}", a.rect.w);
    }

    #[test]
    fn em_item_width_resolves_against_item_font_size() {
        // The `auto` track sizes to the item's definite `10em` width:
        // with the item at 20px and the container at 10px the track is
        // 200px — not 100px from the container's font size.
        let l = grid_e2e(
            r#"<div class="g"><div class="a"></div></div>"#,
            ".g { display: grid; grid-template-columns: auto; font-size: 10px; } \
             .a { width: 10em; font-size: 20px; height: 20px; }",
        );
        let g = &l.children[0].children[0];
        let a = &g.children[0];
        assert!(approx(a.rect.w, 200.0), "w={}", a.rect.w);
    }

    #[test]
    fn order_above_one_million_sorts_by_true_value() {
        // `order` is an unbounded <integer> (css-grid-1 §4.2): values
        // above the old ±1e6 clamp must keep their relative order instead
        // of collapsing onto the clamp and falling back to DOM order.
        let l = grid_e2e(
            r#"<div class="g"><div class="a">a</div><div class="b">b</div></div>"#,
            ".g { display: grid; grid-template-columns: 100px 100px; } \
             .a { order: 3000000; width: 30px; height: 20px; } \
             .b { order: 2000000; width: 60px; height: 20px; }",
        );
        let g = &l.children[0].children[0];
        assert_eq!(g.children.len(), 2);
        // Identify the items by width (order-independent); the lower
        // order (`.b`, 2000000) must take the first column.
        for c in &g.children {
            if approx(c.rect.w, 60.0) {
                assert!(
                    approx(c.rect.x, g.rect.x),
                    "order:2000000 item not in first column: x={}",
                    c.rect.x
                );
            } else if approx(c.rect.w, 30.0) {
                assert!(
                    approx(c.rect.x, g.rect.x + 100.0),
                    "order:3000000 item not in second column: x={}",
                    c.rect.x
                );
            } else {
                panic!("unexpected item width {}", c.rect.w);
            }
        }
    }

    #[test]
    fn definite_border_box_width_sizes_track_in_content_terms() {
        // `box-sizing: border-box; width: 200px; padding: 0 10px`: the
        // definite 200px is a border-box width. Track sizing must strip
        // it to content-box terms (180) before comparing against the
        // content-box preferred width — otherwise the track grows to
        // 200 + 20 = 220px by counting the padding twice.
        let l = grid_e2e(
            r#"<div class="g"><div class="i"></div><div class="j"></div></div>"#,
            ".g { display: grid; grid-template-columns: auto auto; justify-content: start; } \
             .i { box-sizing: border-box; width: 200px; padding: 0 10px; height: 20px; } \
             .j { height: 20px; }",
        );
        let g = &l.children[0].children[0];
        let (i, j) = (&g.children[0], &g.children[1]);
        // The item keeps its definite border-box width ...
        assert!(approx(i.rect.w, 200.0), "item w={}", i.rect.w);
        // ... and the track is exactly that wide (not 220).
        assert!(
            approx(j.rect.x, g.rect.x + 200.0),
            "second track x={} (track 1 is 220 wide)",
            j.rect.x
        );
    }

    #[test]
    fn grid_pref_width_floors_on_definite_item_width() {
        // The intrinsic pass must floor on a definite item width like
        // the used pass does: the empty 200px item contributes 200px to
        // its `auto` track, so the shrink-to-fit float around the grid
        // measures 200px, not 0.
        let l = grid_e2e(
            r#"<div class="f"><div class="g"><div class="i"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .g { display: grid; grid-template-columns: auto; } \
             .i { width: 200px; height: 20px; }",
        );
        let f = &l.children[0].children[0];
        assert!(approx(f.rect.w, 200.0), "float w={}", f.rect.w);
    }

    #[test]
    fn inline_grid_pref_width_floors_on_definite_item_width() {
        // Same floor through the `InlineGrid` arm of `preferred_width`:
        // a shrink-to-fit inline-grid holding one empty 200px item is
        // 200px wide.
        let l = grid_e2e(
            r#"<div class="g"><div class="i"></div></div>"#,
            ".g { display: inline-grid; grid-template-columns: auto; } \
             .i { width: 200px; height: 20px; }",
        );
        let g = &l.children[0].children[0];
        assert!(approx(g.rect.w, 200.0), "inline-grid w={}", g.rect.w);
    }

    #[test]
    fn grid_pref_width_min_content_width_uses_min_content_walk() {
        // Item 1: a `display: grid` box with `width: min-content` inside
        // a shrink-to-fit float contributes its min-content size, not the
        // full max-content track sum (css-sizing-3 §5.2). At 10px type one
        // char measures 6px: "aaa bbb" is 42px max-content but 18px
        // min-content ("aaa"), so the two `auto` tracks are 18/18 = 36px,
        // not 42/18 = 60px.
        let l = grid_e2e(
            r#"<div class="f"><div class="g"><div class="i">aaa bbb</div><div class="j">ccc</div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .g { display: grid; grid-template-columns: auto auto; width: min-content; font-size: 10px; }",
        );
        // body > div.f > div.g.
        let f = &l.children[0].children[0];
        let g = &f.children[0];
        assert!(
            approx(f.rect.w, 36.0),
            "float should shrink to the min-content 36px, not max-content 60px: {}",
            f.rect.w
        );
        assert!(
            approx(g.rect.w, 36.0),
            "grid used width should be the min-content size: {}",
            g.rect.w
        );
    }

    #[test]
    fn grid_pref_width_definite_width_contributes_exactly() {
        // Item 1: `width: 200px` on a grid inside a shrink-to-fit float
        // contributes exactly 200px, not the 90px max-content track sum
        // (css-sizing-3 §5.2).
        let l = grid_e2e(
            r#"<div class="f"><div class="g"><div class="i">aaa bbb ccc ddd</div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .g { display: grid; grid-template-columns: auto; width: 200px; font-size: 10px; }",
        );
        // body > div.f > div.g.
        let f = &l.children[0].children[0];
        let g = &f.children[0];
        assert!(
            approx(f.rect.w, 200.0),
            "float should shrink to the definite 200px: {}",
            f.rect.w
        );
        assert!(
            approx(g.rect.w, 200.0),
            "grid used width should be the definite 200px: {}",
            g.rect.w
        );
    }

    #[test]
    fn grid_pref_width_definite_width_skips_item_intrinsic_walks() {
        // Finding 1 witness: a definite-width grid's intrinsic contribution
        // is the definite width itself (`grid_pref_adjusted`'s definite arm
        // discards the track-based intrinsic), so the per-item
        // `preferred_width_mcw` walks must never run. A panicking font spy
        // proves the early return is taken: any item intrinsic measurement
        // would advance text and panic.
        struct PanicFonts;
        impl crate::FontMetrics for PanicFonts {
            fn advance(&self, _text: &str, _font_size: f32, _family: crate::FontFamilyId) -> f32 {
                panic!("item intrinsic walk ran for a definite-width grid")
            }
            fn line_height(&self, _font_size: f32, _family: crate::FontFamilyId) -> f32 {
                panic!("item intrinsic walk ran for a definite-width grid")
            }
        }
        let (mut forest, grid) = forest_grid(
            r#"<div class="g"><div class="i">aaa bbb ccc ddd</div></div>"#,
            ".g { display: grid; grid-template-columns: auto; width: 200px; font-size: 10px; }",
        );
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let pref = grid_pref_width(&mut forest, grid, &PanicFonts, &mut cache);
        assert!(
            approx(pref, 200.0),
            "definite width should contribute exactly without measuring items: {pref}"
        );
    }

    #[test]
    fn inline_grid_pref_width_definite_width_contributes_exactly() {
        // Item 1 through an inline-grid box: a definite width contributes
        // exactly, through the `InlineGrid` arm of `preferred_width`.
        let l = grid_e2e(
            r#"<div class="g"><div class="i">aaa bbb ccc ddd</div></div>"#,
            ".g { display: inline-grid; grid-template-columns: auto; width: 200px; font-size: 10px; }",
        );
        let g = &l.children[0].children[0];
        assert!(approx(g.rect.w, 200.0), "inline-grid w={}", g.rect.w);
    }

    #[test]
    fn grid_pref_width_fit_content_clamps_at_used_time() {
        // Item 1: `width: fit-content` keeps the max-content track sum as
        // its intrinsic contribution; the fit-content clamp
        // (min(max-content, max(min-content, available))) applies at
        // used-value time, where the base is definite. The float's
        // `max-width` narrows the available width below max-content
        // (138px), so the grid sizes to the fit-content size (100px),
        // not full max-content.
        let l = grid_e2e(
            r#"<div class="f"><div class="g"><div class="i">aaa bbb ccc ddd eee fff</div></div></div>"#,
            "body { margin: 0; } .f { float: left; max-width: 100px; } \
             .g { display: grid; grid-template-columns: auto; width: fit-content; font-size: 10px; }",
        );
        // body > div.f > div.g.
        let f = &l.children[0].children[0];
        let g = &f.children[0];
        assert!(
            approx(f.rect.w, 100.0),
            "float should clamp to max-width 100px: {}",
            f.rect.w
        );
        assert!(
            approx(g.rect.w, 100.0),
            "grid used width should be the fit-content size 100px: {}",
            g.rect.w
        );
    }

    #[test]
    fn grid_pref_width_pct_max_width_is_cyclic_not_zero() {
        // `width: 200px; max-width: 50%` on a grid item in intrinsic
        // track sizing: the percentage max-width is cyclic against the
        // indefinite intrinsic base and behaves as `none` — it must not
        // resolve against the zero base (which would clamp the track's
        // contribution to ~0 and under-size the track). The item is
        // `display: table` so its preferred width does not already
        // floor at the definite width and mask the helper.
        let l = grid_e2e(
            r#"<div class="g"><div class="i">x</div></div>"#,
            "body { margin: 0; } \
             .g { display: inline-grid; grid-template-columns: auto; } \
             .i { display: table; width: 200px; max-width: 50%; }",
        );
        let g = &l.children[0].children[0];
        assert!(approx(g.rect.w, 200.0), "inline-grid w={}", g.rect.w);
    }

    #[test]
    fn grid_pref_width_spanning_item_definite_width_distributes() {
        // A definite width on a spanning item raises the span total,
        // distributed over the content-sized tracks by the deficit
        // logic: two `auto` tracks share the 200px.
        let l = grid_e2e(
            r#"<div class="f"><div class="g"><div class="i"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .g { display: grid; grid-template-columns: auto auto; } \
             .i { grid-column: 1 / 3; width: 200px; height: 20px; }",
        );
        let f = &l.children[0].children[0];
        assert!(approx(f.rect.w, 200.0), "float w={}", f.rect.w);
    }

    #[test]
    fn grid_used_pass_spanning_item_definite_width_distributes() {
        // The used pass must floor on a spanning item's definite width
        // exactly like the intrinsic pass does: without the floor the
        // empty 200px item contributes 0 in the used pass while the
        // intrinsic pass measured 200 — an intrinsic/used divergence on
        // the block-level grid the wave fixed. The second track starts
        // 100px in, so the tracks sum to 200px and the item never
        // overflows them.
        let l = grid_e2e(
            r#"<div class="g"><div class="i"></div><div class="j"></div></div>"#,
            "body { margin: 0; } \
             .g { display: grid; grid-template-columns: auto auto; justify-content: start; } \
             .i { grid-column: 1 / 3; width: 200px; height: 20px; } \
             .j { grid-column: 2; grid-row: 2; height: 20px; }",
        );
        let g = &l.children[0].children[0];
        let (i, j) = (&g.children[0], &g.children[1]);
        assert!(
            approx(j.rect.x, g.rect.x + 100.0),
            "second track x={} (span total not distributed over tracks)",
            j.rect.x
        );
        assert!(
            approx(i.rect.w, 200.0),
            "item w={} (tracks too narrow; item overflows the span)",
            i.rect.w
        );
        assert!(
            approx(i.rect.x + i.rect.w, g.rect.x + 200.0),
            "item right edge {} overflows 200px tracks",
            i.rect.x + i.rect.w
        );
    }

    #[test]
    fn grid_used_pass_percent_calc_treated_as_auto() {
        // CSS Values 4 §8.1.4: a percentage-bearing `calc()` is cyclic
        // during track sizing and behaves as `auto` — the used pass must
        // not resolve it against the container width. The empty item's
        // `width: calc(50px + 50%)` contributes nothing to its `auto`
        // track (the viewport is 800px wide; the old code measured
        // 0.5 * 800 + 50 = 450px).
        let l = grid_e2e(
            r#"<div class="g"><div class="i"></div><div class="j"></div></div>"#,
            "body { margin: 0; } \
             .g { display: grid; grid-template-columns: auto auto; justify-content: start; } \
             .i { width: calc(50px + 50%); height: 20px; } \
             .j { height: 20px; }",
        );
        let g = &l.children[0].children[0];
        let (i, j) = (&g.children[0], &g.children[1]);
        // The first track measures 0: the cyclic width is `auto` for
        // track sizing, and the empty item has no content.
        assert!(
            approx(j.rect.x, g.rect.x),
            "second track should start at the grid edge, x={}",
            j.rect.x
        );
        // The item itself keeps its definite width, resolved against its
        // (zero-width) grid area: 0.5 * 0 + 50.
        assert!(approx(i.rect.w, 50.0), "item w={}", i.rect.w);
    }

    // A `calc()` bound only counts as percentage-like when it
    // syntactically contained a `%` leaf (`had_pct`, CSS Values 4
    // §8.1.4): a pure-length `calc(10px)` (`had_pct: false`) is a
    // definite length and must not degrade to content-based against an
    // indefinite container height.
    #[test]
    fn row_minmax_bound_gates_on_had_pct_not_bare_math() {
        let pure = || {
            MinMaxBound::Fixed(SpecLen::Math {
                pct: 0.0,
                px: 10.0,
                had_pct: false,
                affine: true,
            })
        };
        let pct_math = || {
            MinMaxBound::Fixed(SpecLen::Math {
                pct: 0.0,
                px: 10.0,
                had_pct: true,
                affine: true,
            })
        };
        // `normalize_row_bound`: a pure-length calc survives as a fixed
        // bound; a percentage-holding calc degrades to content.
        assert!(matches!(
            normalize_row_bound(pure(), None),
            MinMaxBound::Fixed(_)
        ));
        assert!(matches!(
            normalize_row_bound(pct_math(), None),
            MinMaxBound::Content
        ));
        // `definite_row_track_size`: `minmax(calc(10px), calc(10px))`
        // against an indefinite height is a definite 10px track.
        let track = TrackSize::MinMax {
            min: pure(),
            max: pure(),
        };
        assert_eq!(
            definite_row_track_size(&track, 16.0, 16.0, 800.0, 600.0, None),
            Some(10.0)
        );
    }

    // The column counterpart of
    // `row_minmax_bound_gates_on_had_pct_not_bare_math`: a `calc()`
    // bound only counts as percentage-like when it syntactically
    // contained a `%` leaf (`had_pct`, CSS Values 4 §8.1.4). A
    // pure-length `calc(10px)` (`had_pct: false`) is a definite length
    // and must not degrade against an indefinite container width; a
    // definite width (`Some`) never degrades either.
    #[test]
    fn col_minmax_bound_gates_on_had_pct_not_bare_math() {
        let pure = || {
            MinMaxBound::Fixed(SpecLen::Math {
                pct: 0.0,
                px: 10.0,
                had_pct: false,
                affine: true,
            })
        };
        let pct_math = || {
            MinMaxBound::Fixed(SpecLen::Math {
                pct: 0.0,
                px: 10.0,
                had_pct: true,
                affine: true,
            })
        };
        let pct = || MinMaxBound::Fixed(SpecLen::Pct(0.5));
        // `normalize_col_bound`: a pure-length calc survives as a fixed
        // bound; a percentage or percentage-holding calc degrades to
        // content against an indefinite width.
        assert!(matches!(
            normalize_col_bound(pure(), None),
            MinMaxBound::Fixed(_)
        ));
        assert!(matches!(
            normalize_col_bound(pct_math(), None),
            MinMaxBound::Content
        ));
        assert!(matches!(
            normalize_col_bound(pct(), None),
            MinMaxBound::Content
        ));
        // A definite container width resolves percentages: no gate.
        assert!(matches!(
            normalize_col_bound(pct(), Some(400.0)),
            MinMaxBound::Fixed(_)
        ));
        assert!(matches!(
            normalize_col_bound(pct_math(), Some(400.0)),
            MinMaxBound::Fixed(_)
        ));
    }

    #[test]
    fn grid_pref_width_percent_column_track_sizes_to_content() {
        // A `50%` column track in the intrinsic pass has no definite
        // width to resolve against and behaves as `auto`: the
        // shrink-to-fit float measures the item's 200px, not a 0px
        // track (the used pass stays as it was — it resolves against
        // the real content width).
        let l = grid_e2e(
            r#"<div class="f"><div class="g"><div class="i"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .g { display: grid; grid-template-columns: 50%; } \
             .i { width: 200px; height: 20px; }",
        );
        let f = &l.children[0].children[0];
        assert!(approx(f.rect.w, 200.0), "float w={}", f.rect.w);
    }

    #[test]
    fn grid_pref_width_percent_calc_column_degrades_to_content() {
        // CSS Values 4 §8.1.4: a percentage-bearing `calc()` is cyclic
        // in track sizing even when the percentage folds to zero
        // (`calc(10px + 0%)` has `had_pct: true`): the track degrades
        // to content instead of resolving to 10px against the 0.0 base.
        let l = grid_e2e(
            r#"<div class="f"><div class="g"><div class="i"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .g { display: grid; grid-template-columns: calc(10px + 0%); } \
             .i { width: 200px; height: 20px; }",
        );
        let f = &l.children[0].children[0];
        assert!(approx(f.rect.w, 200.0), "float w={}", f.rect.w);
    }

    #[test]
    fn grid_pref_width_pure_calc_column_resolves() {
        // A pure-length `calc(10px + 5px)` (`had_pct: false`) is a
        // definite length: it still resolves to 15px against the 0.0
        // base and does not degrade to content.
        let l = grid_e2e(
            r#"<div class="f"><div class="g"><div class="i"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .g { display: grid; grid-template-columns: calc(10px + 5px); } \
             .i { height: 20px; }",
        );
        let f = &l.children[0].children[0];
        assert!(approx(f.rect.w, 15.0), "float w={}", f.rect.w);
    }

    #[test]
    fn grid_pref_width_percent_minmax_max_degrades_to_content() {
        // `minmax(10px, 50%)` in the intrinsic pass: the percentage
        // maximum degrades to content-based instead of clamping the
        // track to 50%-of-0 = 0, so the 200px item measures 200px, not
        // 10px.
        let l = grid_e2e(
            r#"<div class="f"><div class="g"><div class="i"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .g { display: grid; grid-template-columns: minmax(10px, 50%); } \
             .i { width: 200px; height: 20px; }",
        );
        let f = &l.children[0].children[0];
        assert!(approx(f.rect.w, 200.0), "float w={}", f.rect.w);
    }

    #[test]
    fn grid_min_width_percent_column_track_sizes_to_content() {
        // Same gate in the min-content pass: `width: min-content`
        // routes through `grid_min_width`, and the `50%` track must
        // size to the 200px item rather than resolving to 0.
        let l = grid_e2e(
            r#"<div class="g"><div class="i"></div></div>"#,
            "body { margin: 0; } \
             .g { display: grid; width: min-content; grid-template-columns: 50%; } \
             .i { width: 200px; height: 20px; }",
        );
        let g = &l.children[0].children[0];
        assert!(approx(g.rect.w, 200.0), "grid w={}", g.rect.w);
    }

    #[test]
    fn grid_min_width_percent_minmax_max_degrades_to_content() {
        // `minmax(10px, 50%)` in the min-content pass: the percentage
        // maximum degrades to content-based, so the track measures the
        // 200px item instead of clamping to 10px.
        let l = grid_e2e(
            r#"<div class="g"><div class="i"></div></div>"#,
            "body { margin: 0; } \
             .g { display: grid; width: min-content; grid-template-columns: minmax(10px, 50%); } \
             .i { width: 200px; height: 20px; }",
        );
        let g = &l.children[0].children[0];
        assert!(approx(g.rect.w, 200.0), "grid w={}", g.rect.w);
    }

    #[test]
    fn fixed_pct_table_in_max_content_grid_is_saturated_infinite() {
        // WPT `css/css-tables/tentative/table-fixed-minmax.html`: a
        // `width: max-content` grid propagates the fixed-percentage
        // table's infinite max-content, and the table lays out at the
        // saturated size — finite, wider than the viewport.
        let l = grid_e2e(
            r#"<div class="g"><table class="t"><tr><td></td></tr></table></div>"#,
            "body { margin: 0; } \
             .g { display: grid; grid-template-columns: 1fr; width: max-content; } \
             .t { table-layout: fixed; width: 100%; border-spacing: 0; } \
             td { width: 100px; padding: 0; }",
        );
        let g = &l.children[0].children[0];
        let t = &g.children[0];
        assert!(g.rect.w.is_finite(), "grid w={}", g.rect.w);
        assert!(g.rect.w > 800.0, "grid w={}", g.rect.w);
        assert!(t.rect.w.is_finite(), "table w={}", t.rect.w);
        assert!(t.rect.w > 800.0, "table w={}", t.rect.w);
    }

    #[test]
    fn fixed_pct_table_in_auto_grid_is_not_infinite() {
        // WPT `css/css-tables/tentative/table-fixed-minmax.html`: a
        // definite-width (`width: auto`) grid does not propagate the
        // infinite intrinsic size — the `1fr` track fills the container
        // instead of taking the infinite content base.
        let l = grid_e2e(
            r#"<div class="g"><table class="t"><tr><td></td></tr></table></div>"#,
            "body { margin: 0; } \
             .g { display: grid; grid-template-columns: 1fr; } \
             .t { table-layout: fixed; width: 100%; border-spacing: 0; } \
             td { width: 100px; padding: 0; }",
        );
        let g = &l.children[0].children[0];
        let t = &g.children[0];
        assert!(approx(g.rect.w, 800.0), "grid w={}", g.rect.w);
        assert!(approx(t.rect.w, 800.0), "table w={}", t.rect.w);
    }

    #[test]
    fn grid_pref_width_clamped_by_own_max_width() {
        // `grid_pref_adjusted`: the track-based intrinsic sum honors the
        // grid's own `max-width`, mirroring the block/flex content-based
        // fallbacks (the definite arms clamp inside their own
        // resolution, but the raw track sum did not).
        let (mut forest, grid) = forest_grid(
            r#"<div class="g"><div class="i"></div></div>"#,
            "body { margin: 0; } \
             .g { display: grid; grid-template-columns: 100px 100px; max-width: 150px; } \
             .i { height: 20px; }",
        );
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let pref = grid_pref_width(&mut forest, grid, &crate::FixedFontMetrics, &mut cache);
        assert!(
            approx(pref, 150.0),
            "max-width clamps the track sum: {pref}"
        );
    }

    #[test]
    fn grid_pref_width_floored_by_own_min_width() {
        // Symmetric floor: the track-based intrinsic sum honors the
        // grid's own `min-width`.
        let (mut forest, grid) = forest_grid(
            r#"<div class="g"><div class="i"></div></div>"#,
            "body { margin: 0; } \
             .g { display: grid; grid-template-columns: 100px; min-width: 150px; } \
             .i { height: 20px; }",
        );
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let pref = grid_pref_width(&mut forest, grid, &crate::FixedFontMetrics, &mut cache);
        assert!(
            approx(pref, 150.0),
            "min-width floors the track sum: {pref}"
        );
    }

    #[test]
    fn subgrid_pref_width_resolves_parent_listing_against_indefinite_axis() {
        // `parse_axis_tracks` in intrinsic mode: the subgrid reuses the
        // parent grid's *listing*, but the parent axis is indefinite —
        // the parent's used content width does not exist during
        // intrinsic sizing. `repeat(auto-fill, 100px)` must expand to a
        // single repetition (like the container's own listing under a
        // `None` base), not count against the parent's stale
        // `metrics.content_w` (poisoned to 800px here to simulate a
        // previous layout pass: the old code measured 800px).
        let (mut forest, parent) = forest_grid(
            r#"<div class="p"><div class="s"><div class="i"></div></div></div>"#,
            "body { margin: 0; } \
             .p { display: grid; grid-template-columns: repeat(auto-fill, 100px); } \
             .s { display: grid; grid-template-columns: subgrid; } \
             .i { height: 20px; }",
        );
        forest.nodes[parent].metrics.content_w = 800.0;
        let sub = forest.nodes[parent]
            .children
            .iter()
            .copied()
            .find(|&c| forest.nodes[c].box_type == BoxType::Grid)
            .expect("subgrid box");
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let pref = grid_pref_width(&mut forest, sub, &crate::FixedFontMetrics, &mut cache);
        assert!(approx(pref, 100.0), "single auto-fill repetition: {pref}");
    }
}
