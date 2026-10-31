//! The layout engine.
//!
//! Everything runs on an explicit task stack over the box arena, so layout
//! is iterative and safe on adversarially deep DOMs. Box construction,
//! margin-collapse walks, inline-block pre-layout, fragment building,
//! preferred-width measurement, subtree shifts, and arena-to-tree
//! conversion are all loops. The one exception is the vertical intrinsic
//! measurement family (`vertical_inline_min`, `vertical_inline_max`,
//! `vertical_atomic_inline`, `vertical_block_child`,
//! `vertical_flow_content`), which recurses over nested inline/block
//! structure; it threads an explicit depth counter capped at
//! `VERTICAL_INTRINSIC_DEPTH_CAP` (256), past which the measurement
//! declines honestly and the caller falls back to the pre-vertical-intrinsic
//! horizontal measure (the documented LIMITATION fallback) instead of
//! overflowing the stack on adversarially deep DOMs.
//!
//! Coordinate system: every `rect` is a border box in absolute page
//! coordinates, origin at the top-left of the initial containing block
//! (the viewport origin).
//!
//! Documented simplifications (see also `lib.rs`):
//! - Out-of-flow boxes (`position: absolute | fixed`) are recorded during
//!   flow and laid out by the `layout_position` post-pass; floats are
//!   placed per CSS 2.1 §9.5.1 (per-side stacking, clearance, BFC
//!   containment, line-box shortening) with the simplifications noted on
//!   the float helpers below.
//! - `vertical-align: baseline` splits the line height across the baseline
//!   by the used face's ascent/descent ratio from `FontMetrics`
//!   (`split_line_height`; the 0.8/0.2 split survives only as the trait
//!   default). Length values shift the box (positive raises it).
//! - Line boxes include the block's line-height as a strut (an empty
//!   inline box), flooring the line-box height.
//! - `text-align: justify` spreads each line's leftover space across its
//!   word gaps (the block's last line is exempt, aligning per
//!   `text-align-last`); `start`/`end` map to `left`/`right` per the
//!   element's `direction`.
//! - `text-indent` offsets the first line box.
//! - Inline-block shrink-to-fit is `min(preferred, available)`; the
//!   preferred width includes each child's used horizontal
//!   padding/border/margins (percentages resolve against a zero base,
//!   `auto` margins count as zero).

use super::boxes::{Forest, is_block_level, wrap_pre_line_text, wrap_pre_text};
use super::style::{
    AspectRatio, BorderStyle, BoxSizing, Clear, Display, Float, Position, SpecLen, SpecMargin,
    TextAlign, VAlign, WhiteSpace, WritingMode, collapse, sane_len, spec_len_of,
};
use super::{BoxType, FontFamilyId, FontMetrics, Rect};
use bowser_css::{ComputedStyle, CssValue, props};
use bowser_dom::{NodeId, QuirksMode};
use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;

/// Multi-line flexbox (`flex-wrap: wrap | wrap-reverse`): line collection,
/// per-line resolution, and `align-content` stacking. Chosen by
/// `layout_flex` when `flex-wrap` is not `nowrap`. The file sits next to
/// `layout.rs` (rather than in a `layout/` subdirectory) so the module
/// tree stays flat; `#[path]` keeps it a child of the `layout` module.
/// (A `layout.rs` file plus a `layout/` directory would otherwise read as
/// two owners of the same module.)
#[path = "layout_flex_wrap.rs"]
mod layout_flex_wrap;

/// Relative and absolute positioning: containing-block resolution,
/// `top`/`right`/`bottom`/`left` offsets for relative (shift without
/// changing the flow footprint) and absolute/fixed (out-of-flow,
/// positioned per containing block) boxes, and the post-pass that lays
/// out out-of-flow subtrees after normal flow. Kept next to `layout.rs`
/// (rather than in a `layout/` subdirectory) so the module tree stays
/// flat; `#[path]` keeps it a child of the `layout` module.
#[path = "layout_position.rs"]
mod layout_position;

// Re-exported for `crate::layout_with_scroll` (lib.rs): the sticky
// constraint pass runs after `layout_forest`, against final rects.
pub(crate) use layout_position::apply_sticky_constraints;

/// Table layout (`display: table`): table → row-group → row → cell grid
/// collection, simplified automatic column-width resolution, cell
/// placement, and row-height resolution. Kept next to `layout.rs` (rather
/// than in a `layout/` subdirectory) so the module tree stays flat;
/// `#[path]` keeps it a child of the `layout` module.
#[path = "layout_table.rs"]
pub(crate) mod layout_table;

/// Grid layout (`display: grid`): explicit tracks from
/// `grid-template-columns` / `grid-template-rows` (fixed lengths, `fr`,
/// `repeat()`), sparse auto-placement in row order, `grid-column` /
/// `grid-row` line/span placement, `gap`, content-sized implicit tracks,
/// and `justify-items` / `align-items` (stretch default). Kept next to
/// `layout.rs` (rather than in a `layout/` subdirectory) so the module
/// tree stays flat; `#[path]` keeps it a child of the `layout` module.
#[path = "layout_grid.rs"]
mod layout_grid;

/// v1 multi-column layout (CSS Multicol 1, subset): column-count/width/gap
/// resolution plus the column-distribution passes for block children and
/// line boxes. Kept next to `layout.rs` (rather than in a `layout/`
/// subdirectory) so the module tree stays flat; `#[path]` keeps it a
/// child of the `layout` module. `pub(crate)` so `boxes` can name the
/// `MulticolParams` type for the arena node's sidecar field.
#[path = "layout_multicol.rs"]
pub(crate) mod layout_multicol;

// ---------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------

/// Input for laying out one block-level box.
#[derive(Clone)]
pub(crate) struct BlockInput {
    idx: usize,
    cb_x: f32,
    cb_w: f32,
    cb_h: Option<f32>,
    /// Out-of-flow percentage-height base: when the containing-block
    /// height is indefinite (`cb_h: None`) but this absolutely positioned
    /// box has a percentage (or percentage-bearing `calc()`) `height`,
    /// browsers resolve it against the containing block's *used* height
    /// instead of computing to `auto` (WPT
    /// css-position/position-relative-016 t8/t9). `layout_abs_items`
    /// records the used height — final by then, since flow is done and
    /// items lay out outer-first — and `layout_block` resolves the
    /// percentage against it via [`specified_height`]. Deliberately
    /// separate from `cb_h`: percentage *insets* still compute to `auto`
    /// against the indefinite height (`abspos_solved_sizes` and
    /// `abs_position` keep seeing `cb_h: None` / `cb_h_definite: false`),
    /// and only non-replaced boxes with a percentage height ever set
    /// this, so `top: 0`-only / `min-height`-only / all-`auto` boxes stay
    /// indefinite (t13/t14/t15) and replaced-height definiteness is
    /// untouched.
    abs_pct_height_base: Option<f32>,
    shrink: bool,
    /// Caller-determined content width (flex items): skips shrink-to-fit
    /// and forces auto margins to zero.
    width_override: Option<f32>,
    /// When true, `width_override` is a final flex §9.7 main size: it
    /// already carries min/max clamping (minimum wins on conflict) and
    /// must not be re-clamped here.
    width_override_final: bool,
    /// Caller-determined content height (stretched flex items): used in
    /// place of the specified height.
    height_override: Option<f32>,
    /// The specified (or overridden) height is a *minimum*, not a fixed
    /// height: the used content height is `max(natural, height)` (CSS 2.1
    /// §17.5.3 — table cells and rows treat `height` as a minimum, so
    /// content taller than the specified height grows the cell instead
    /// of overflowing it). Only the table layout sets this, for cells.
    height_is_min: bool,
    /// Winning collapsed-border widths `[top, right, bottom, left]` for a
    /// table cell under `border-collapse: collapse`. The winners are
    /// centered on the shared grid lines (CSS 2.1 §17.6.2.2), so the
    /// cell's used border widths become half the winning widths — not
    /// the cell's own (possibly losing) specified widths. Only the table
    /// layout sets this, for cells.
    collapsed_border_wins: Option<[f32; 4]>,
    /// Out-of-flow root (`position: absolute | fixed`), laid out by the
    /// `layout_position` post-pass: `auto` margins compute to zero, then
    /// the constraint equation solves them when both offsets on the axis
    /// are specified (centering when both are `auto`); an `auto` width
    /// uses shrink-to-fit, and vertical margins never collapse.
    abspos: bool,
    /// Floats placed so far in this block formatting context, shared by
    /// reference: non-BFC descendants read and append to the same list,
    /// so a float placed by an earlier sibling intrudes into later
    /// siblings' content. Each BFC root starts a fresh list (floats
    /// never cross a BFC boundary).
    floats: FloatList,
    /// Floated box (`float: left | right`, in-flow): auto margins
    /// compute to zero, an `auto` width uses shrink-to-fit (CSS 2.1
    /// §10.3.5), vertical margins never collapse, and the box is
    /// positioned by the float algorithm when its `Task::Block`
    /// completes instead of advancing the flow cursor.
    is_float: bool,
}

// Manual `Debug`: the shared float list (`FloatCtx`) has no `Debug`
// impl, and deriving one would cascade through the float index — the
// entry count is all a debug print needs.
impl std::fmt::Debug for BlockInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockInput")
            .field("idx", &self.idx)
            .field("cb_x", &self.cb_x)
            .field("cb_w", &self.cb_w)
            .field("cb_h", &self.cb_h)
            .field("abs_pct_height_base", &self.abs_pct_height_base)
            .field("shrink", &self.shrink)
            .field("width_override", &self.width_override)
            .field("width_override_final", &self.width_override_final)
            .field("height_override", &self.height_override)
            .field("height_is_min", &self.height_is_min)
            .field("collapsed_border_wins", &self.collapsed_border_wins)
            .field("abspos", &self.abspos)
            .field("floats_len", &self.floats.borrow().entries.len())
            .field("is_float", &self.is_float)
            .finish()
    }
}

/// One placed float: its margin box in the provisional frame, plus its
/// side. Floats stack per side (CSS 2.1 §9.5.1); margins never collapse
/// but they do take up space, so entries cover the margin box.
///
/// `node` is the float's forest node id, so passes that move float
/// subtrees whole (multicol column assignment) can match an entry to
/// its box after the geometry changed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FloatEntry {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub left: bool,
    pub node: usize,
}

/// The floats affecting one block formatting context, shared by
/// reference (see [`BlockInput::floats`]).
///
/// `entries` is the source of truth in placement order (other code
/// slices it, e.g. BFC containment); `index` is an interval index over
/// the same entries so [`place_float`]'s band queries run in
/// O(log F + k) instead of scanning all F floats.
pub(crate) struct FloatCtx {
    pub entries: Vec<FloatEntry>,
    index: FloatIndex,
    /// Reusable explicit stack for [`FloatIndex`] band queries: no
    /// allocation per query.
    query_stack: Vec<u32>,
    /// Maximum outer (margin-box) top over all placed floats, cached
    /// on push/reindex so [`place_float`] can apply CSS 2.1 §9.5.1
    /// rule 5's floated-box clause in O(1) instead of scanning every
    /// earlier float per placement.
    max_top: f32,
}

impl FloatCtx {
    pub fn new() -> Self {
        FloatCtx {
            entries: Vec::new(),
            index: FloatIndex::new(),
            query_stack: Vec::new(),
            max_top: f32::NEG_INFINITY,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Append a placed float, indexing it for band queries.
    pub fn push(&mut self, e: FloatEntry) {
        let id = self.entries.len() as u32;
        // Same `y + h` the band predicate computes; cached so the
        // index never recomputes it.
        let bottom = e.y + e.h;
        self.max_top = self.max_top.max(e.y);
        self.entries.push(e);
        self.index.insert(id, e.y, bottom);
    }

    /// Rebuild the interval index from the current entries. Used after
    /// an out-of-band entry mutation (multicol column assignment moves
    /// already-placed floats) so later band queries see the final
    /// geometry.
    pub fn reindex(&mut self) {
        self.index = FloatIndex::new();
        self.max_top = f32::NEG_INFINITY;
        for (i, e) in self.entries.iter().enumerate() {
            self.max_top = self.max_top.max(e.y);
            self.index.insert(i as u32, e.y, e.y + e.h);
        }
    }

    /// Drop every entry from `len` onward and rebuild the interval
    /// index plus `max_top` from what remains. Directly truncating
    /// `entries` would leave stale index nodes whose ids alias (or
    /// run past) the re-pushed floats, and a stale `max_top` — so all
    /// truncation goes through here.
    pub fn truncate(&mut self, len: usize) {
        self.entries.truncate(len);
        self.reindex();
    }

    /// Indexed [`line_float_avail`]: the same `(left intrusion,
    /// available width)` pair as the free function, but the interval
    /// index visits only the floats overlapping `[y0, y1)` instead of
    /// scanning all of `entries` — O(log F + k) per band rather than
    /// O(F).
    ///
    /// No y-ordering of `entries` is assumed (multicol column
    /// assignment moves entries out of insertion order; see
    /// [`FloatCtx::reindex`]): the index answers the band predicate
    /// directly. The accumulation is order-independent maxima, so the
    /// result is bit-identical to the naive scan (differentially
    /// tested in `float_band_query_tests`).
    ///
    /// Not yet called by layout (call-site migration is separate
    /// work); exercised by the differential tests below.
    #[allow(dead_code)]
    pub fn line_float_avail(
        &mut self,
        content_x: f32,
        content_w: f32,
        y0: f32,
        y1: f32,
    ) -> (f32, f32) {
        let FloatCtx {
            entries,
            index,
            query_stack,
            ..
        } = self;
        let mut left = 0.0f32;
        let mut right = 0.0f32;
        index.for_each_in_band(y0, y1, query_stack, |id| {
            let f = &entries[id as usize];
            if f.left {
                left = left.max(f.x + f.w - content_x);
            } else {
                right = right.max(content_x + content_w - f.x);
            }
        });
        (left, (content_w - left - right).max(0.0))
    }

    /// Indexed [`intruding_float_bottom`]: the lowest bottom margin
    /// edge of the floats overlapping `[y0, y1)` (`y0` when none do),
    /// visiting only band-overlapping floats via the interval index.
    /// Bit-identical to the naive scan (differentially tested).
    ///
    /// Not yet called by layout (call-site migration is separate
    /// work); exercised by the differential tests below.
    #[allow(dead_code)]
    pub fn intruding_float_bottom(&mut self, y0: f32, y1: f32) -> f32 {
        let FloatCtx {
            entries,
            index,
            query_stack,
            ..
        } = self;
        let mut bottom = y0;
        index.for_each_in_band(y0, y1, query_stack, |id| {
            let f = &entries[id as usize];
            bottom = bottom.max(f.y + f.h);
        });
        bottom
    }

    /// Indexed [`min_intruding_float_bottom`]: the shallowest bottom
    /// margin edge of the floats overlapping `[y0, y1)` (`y0` when none
    /// do), visiting only band-overlapping floats via the interval
    /// index. Bit-identical to the naive scan (differentially tested).
    ///
    /// Not yet called by layout (call-site migration is separate
    /// work); exercised by the differential tests below.
    #[allow(dead_code)]
    pub fn min_intruding_float_bottom(&mut self, y0: f32, y1: f32) -> f32 {
        let FloatCtx {
            entries,
            index,
            query_stack,
            ..
        } = self;
        let mut bottom = f32::INFINITY;
        index.for_each_in_band(y0, y1, query_stack, |id| {
            let f = &entries[id as usize];
            bottom = bottom.min(f.y + f.h);
        });
        if bottom.is_infinite() { y0 } else { bottom }
    }

    /// Indexed [`clearance_y`]: the lowest bottom margin edge of the
    /// floats on the relevant side(s), or `None` when no such float
    /// exists. Subtrees whose augmented maximum bottom edge cannot
    /// beat the running best are pruned, so the common case touches
    /// far fewer than F floats; worst case is still O(F). Bit-identical
    /// to the naive scan (differentially tested).
    ///
    /// Not yet called by layout (call-site migration is separate
    /// work); exercised by the differential tests below.
    #[allow(dead_code)]
    pub fn clearance_y(&mut self, clear: Clear) -> Option<f32> {
        let FloatCtx {
            entries,
            index,
            query_stack,
            ..
        } = self;
        match clear {
            Clear::None => None,
            Clear::Left => index.max_bottom_where(query_stack, |id| {
                let f = &entries[id as usize];
                f.left.then_some(f.y + f.h)
            }),
            Clear::Right => index.max_bottom_where(query_stack, |id| {
                let f = &entries[id as usize];
                (!f.left).then_some(f.y + f.h)
            }),
            Clear::Both => index.max_bottom_where(query_stack, |id| {
                let f = &entries[id as usize];
                Some(f.y + f.h)
            }),
        }
    }
}

pub(crate) type FloatList = Rc<RefCell<FloatCtx>>;

pub(crate) fn fresh_floats() -> FloatList {
    Rc::new(RefCell::new(FloatCtx::new()))
}

/// Interval index over placed floats: a treap keyed by float top edge
/// (`y`, ties broken by insertion order), each node augmented with the
/// maximum bottom edge in its subtree.
///
/// A vertical-band query `[lo, hi)` visits exactly the floats with
/// `f.y < hi && f.y + f.h > lo` — the same set the naive O(F) scan
/// tests — in expected O(log F + k) time: a subtree whose max bottom
/// is at or below `lo` cannot overlap the band, and a right subtree is
/// entered only when the node's top is below `hi` (right-subtree tops
/// are all `>=` the node's top). Insert is expected O(log F) with
/// deterministic splitmix64 priorities (no RNG state).
struct FloatIndex {
    /// Arena; index 0 is the nil sentinel (`max_bottom` =
    /// `-INFINITY` so pruning works uniformly).
    nodes: Vec<FloatNode>,
    root: u32,
}

struct FloatNode {
    /// Index into [`FloatCtx::entries`].
    entry: u32,
    top: f32,
    bottom: f32,
    max_bottom: f32,
    priority: u64,
    left: u32,
    right: u32,
}

/// Deterministic 64-bit mixer for treap priorities.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl FloatIndex {
    fn new() -> Self {
        FloatIndex {
            nodes: vec![FloatNode {
                entry: 0,
                top: 0.0,
                bottom: 0.0,
                max_bottom: f32::NEG_INFINITY,
                priority: 0,
                left: 0,
                right: 0,
            }],
            root: 0,
        }
    }

    fn insert(&mut self, entry: u32, top: f32, bottom: f32) {
        let priority = splitmix64(u64::from(entry).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let node = self.nodes.len() as u32;
        self.nodes.push(FloatNode {
            entry,
            top,
            bottom,
            max_bottom: bottom,
            priority,
            left: 0,
            right: 0,
        });
        self.root = self.insert_at(self.root, node);
    }

    /// BST insert on `(top, entry)`, then heap-restore on `priority`.
    fn insert_at(&mut self, cur: u32, node: u32) -> u32 {
        if cur == 0 {
            return node;
        }
        let (node_top, node_entry) = {
            let n = &self.nodes[node as usize];
            (n.top, n.entry)
        };
        let (cur_top, cur_entry) = {
            let c = &self.nodes[cur as usize];
            (c.top, c.entry)
        };
        // `(top, entry)` ordering with total f32 comparison; NaN tops
        // sort to the right deterministically (comparisons are false).
        let go_left = node_top < cur_top || (node_top == cur_top && node_entry < cur_entry);
        if go_left {
            let left = self.insert_at(self.nodes[cur as usize].left, node);
            self.nodes[cur as usize].left = left;
            if self.nodes[left as usize].priority < self.nodes[cur as usize].priority {
                return self.rotate_right(cur);
            }
        } else {
            let right = self.insert_at(self.nodes[cur as usize].right, node);
            self.nodes[cur as usize].right = right;
            if self.nodes[right as usize].priority < self.nodes[cur as usize].priority {
                return self.rotate_left(cur);
            }
        }
        self.update(cur);
        cur
    }

    fn rotate_right(&mut self, y: u32) -> u32 {
        let x = self.nodes[y as usize].left;
        self.nodes[y as usize].left = self.nodes[x as usize].right;
        self.nodes[x as usize].right = y;
        self.update(y);
        self.update(x);
        x
    }

    fn rotate_left(&mut self, x: u32) -> u32 {
        let y = self.nodes[x as usize].right;
        self.nodes[x as usize].right = self.nodes[y as usize].left;
        self.nodes[y as usize].left = x;
        self.update(x);
        self.update(y);
        y
    }

    fn update(&mut self, i: u32) {
        let n = &self.nodes[i as usize];
        let (bottom, left, right) = (n.bottom, n.left, n.right);
        let max_bottom = bottom
            .max(self.nodes[left as usize].max_bottom)
            .max(self.nodes[right as usize].max_bottom);
        self.nodes[i as usize].max_bottom = max_bottom;
    }

    /// Call `visit(entry)` for every indexed float overlapping the band
    /// `[lo, hi)`, i.e. `top < hi && bottom > lo`. `stack` is a reusable
    /// explicit stack (no allocation, no recursion depth risk).
    fn for_each_in_band(&self, lo: f32, hi: f32, stack: &mut Vec<u32>, mut visit: impl FnMut(u32)) {
        stack.clear();
        if self.root != 0 {
            stack.push(self.root);
        }
        while let Some(i) = stack.pop() {
            let n = &self.nodes[i as usize];
            if n.top < hi && n.bottom > lo {
                visit(n.entry);
            }
            // Left subtree: tops are `<=` this node's top, so any of
            // them may sit below `hi`; prune on max bottom.
            let l = n.left;
            if l != 0 && self.nodes[l as usize].max_bottom > lo {
                stack.push(l);
            }
            // Right subtree: tops are `>=` this node's top; only
            // enter when some of them can sit below `hi`.
            if n.top < hi {
                let r = n.right;
                if r != 0 && self.nodes[r as usize].max_bottom > lo {
                    stack.push(r);
                }
            }
        }
    }

    /// Maximum bottom edge over the entries accepted by `accept`
    /// (which maps an entry id to its bottom edge, or `None` to
    /// reject it), or `None` when no entry is accepted. Subtrees whose
    /// augmented `max_bottom` cannot beat the running best are pruned;
    /// `stack` is the reusable explicit stack (no allocation, no
    /// recursion depth risk).
    ///
    /// Soundness of the prune: every bottom edge in a subtree is at
    /// most that subtree root's `max_bottom`, so when
    /// `max_bottom <= best` no entry in the subtree — accepted or
    /// not — can beat `best`. The result equals the naive
    /// max-over-accepted scan regardless of visit order.
    ///
    /// Only used by [`FloatCtx::clearance_y`] (see its note).
    #[allow(dead_code)]
    fn max_bottom_where(
        &self,
        stack: &mut Vec<u32>,
        mut accept: impl FnMut(u32) -> Option<f32>,
    ) -> Option<f32> {
        let mut best: Option<f32> = None;
        stack.clear();
        if self.root != 0 {
            stack.push(self.root);
        }
        while let Some(i) = stack.pop() {
            let n = &self.nodes[i as usize];
            if best.is_some_and(|b| n.max_bottom <= b) {
                continue;
            }
            if let Some(bottom) = accept(n.entry) {
                best = Some(best.map_or(bottom, |b: f32| b.max(bottom)));
            }
            // `accept` may have raised `best`; re-check per child.
            let l = n.left;
            if l != 0 && best.is_none_or(|b| self.nodes[l as usize].max_bottom > b) {
                stack.push(l);
            }
            let r = n.right;
            if r != 0 && best.is_none_or(|b| self.nodes[r as usize].max_bottom > b) {
                stack.push(r);
            }
        }
        best
    }
}

/// Whether the box is a floated in-flow box. Out-of-flow boxes
/// (`position: absolute | fixed`) are checked first by callers:
/// absolute positioning wins over `float` (CSS 2.1 §9.7).
fn is_float(forest: &Forest, idx: usize) -> bool {
    !matches!(forest.nodes[idx].data.float, Float::None)
}

/// Place a float's margin box: the highest position at or below
/// `y_start` where it fits between the floats already placed in this
/// context (CSS 2.1 §9.5.1). Returns the margin-box origin.
/// A float that fits nowhere still places at `y_start`, overflowing to
/// the inline end like an over-wide line box.
///
/// Implemented rules: 1 (outer edge inside the block edge), 2
/// (same-side floats beside or above), 3 (the outer edge stops at the
/// nearest overlapping opposite-side float's outer edge), 5's
/// floated-box clause (the outer top may not be higher than the outer
/// top of any earlier floated box in this context: the candidate `y`
/// is floored below at the maximum outer top of the floats already
/// placed here, before the band walk starts), 7 (a float
/// that fits beside an earlier same-side float stays inside the block
/// edge; a float wider than the containing block has nowhere legal to
/// go and overflows instead), and 8/9 (highest possible position,
/// then furthest toward the float side: on a miss the band walk
/// retries at the next float bottom in ascending order, so a short
/// float cannot strand a later float below a tall neighbor).
/// Rule 4 (not above the containing block's top) is enforced by the
/// caller's clamp; rules 5/6's block/line-box clause (not above the
/// top of any earlier in-flow block box, or line box containing
/// earlier content) is enforced by
/// the caller, which floors `y_start` at the highest such top (the
/// flow cursor alone is not enough: a later sibling's negative margin
/// can drag it above an earlier sibling's top).
///
/// Each candidate band `[y, y + mh)` queries [`FloatCtx`]'s interval
/// index instead of scanning every placed float, so each band query
/// costs O(log F + k) (k = genuinely overlapping floats) rather than
/// O(F). The rule 5 floated-box floor above is O(1) (cached
/// `max_top`). The honest bound for a whole placement is
/// O(steps x (log F + k)): a placement retries at most one candidate
/// per distinct float bottom it passes — each retry moves strictly
/// past at least one bottom — so steps <= F, but in the adversarial
/// case (many floats, each stepping past all previous bottoms, with
/// k ~ F overlapping per band) that is O(F^2) worst case per
/// placement, not O(F log F). The visited set
/// is exactly the naive scan's set and min/max-accumulation is
/// order-independent, so results are bit-identical to the naive scan
/// (differentially tested).
fn place_float(
    ctx: &mut FloatCtx,
    content_x: f32,
    content_w: f32,
    left: bool,
    mw: f32,
    mh: f32,
    y_start: f32,
) -> (f32, f32) {
    // Split borrows: the visitor reads `entries` while the query reuses
    // `query_stack`.
    let FloatCtx {
        entries,
        index,
        query_stack,
        max_top,
    } = ctx;
    // CSS 2.1 §9.5.1 rule 5, floated-box clause: this float's outer
    // (margin-box) top may not be higher than the outer top of any
    // earlier float in this context, even when a gap beside a tall
    // early float would otherwise fit it above a later float's top.
    let mut y = y_start.max(*max_top);
    loop {
        let hi = y + mh;
        let mut used_l: f32 = 0.0;
        let mut used_r: f32 = 0.0;
        // Bottoms of the floats overlapping this band. `next` is the
        // lowest such bottom — the next candidate y when the float
        // does not fit beside them (rules 8/9: as high as possible);
        // `lowest` is the highest, used only to detect the empty band.
        let mut next = f32::INFINITY;
        let mut lowest = y;
        index.for_each_in_band(y, hi, query_stack, |e| {
            let f = &entries[e as usize];
            if f.left {
                used_l = used_l.max(f.x + f.w - content_x);
            } else {
                used_r = used_r.max(content_x + content_w - f.x);
            }
            let bottom = f.y + f.h;
            next = next.min(bottom);
            lowest = lowest.max(bottom);
        });
        // Either it fits beside the overlapping floats (each earlier
        // same-side float is then beside or above it, rules 1-2, and
        // its outer edge stops at the nearest overlapping
        // opposite-side float's outer edge, rule 3), or nothing
        // overlaps and it places anyway.
        if mw <= content_w - used_l - used_r || lowest <= y {
            let x = if left {
                content_x + used_l
            } else {
                content_x + content_w - used_r - mw
            };
            return (x, y);
        }
        // Rules 8/9: retry at the next float bottom below this band —
        // the highest position that can free space. Checking every
        // candidate bottom in ascending order is complete: any legal
        // placement at y* implies a fit at the largest candidate
        // bottom at or below y* (a float that shrinks `used_*` between
        // the two has its own bottom in between, contradicting the
        // choice of candidate).
        y = next;
    }
}

#[cfg(test)]
mod float_index_tests {
    use super::{FloatCtx, FloatEntry, place_float, place_float_reference};

    /// Deterministic xorshift64: reproducible pseudo-random inputs with
    /// no external dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Place one float through both implementations and require
    /// bit-identical results, then record it in both states.
    #[allow(clippy::too_many_arguments)]
    fn place_both(
        ctx: &mut FloatCtx,
        reference: &mut Vec<FloatEntry>,
        content_w: f32,
        left: bool,
        mw: f32,
        mh: f32,
        y_start: f32,
        i: usize,
    ) {
        let (x, y) = place_float(ctx, 0.0, content_w, left, mw, mh, y_start);
        let (rx, ry) = place_float_reference(reference, 0.0, content_w, left, mw, mh, y_start);
        assert_eq!(x.to_bits(), rx.to_bits(), "x mismatch at float {i}");
        assert_eq!(y.to_bits(), ry.to_bits(), "y mismatch at float {i}");
        let e = FloatEntry {
            x,
            y,
            w: mw,
            h: mh,
            left,
            node: i,
        };
        ctx.push(e);
        reference.push(e);
    }

    #[test]
    fn indexed_placement_matches_reference_random() {
        // Mixed left/right floats, varied dimensions and start heights,
        // with starts biased onto a few repeated values so bands share
        // top/bottom edges (the interval index's boundary conditions).
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut ctx = FloatCtx::new();
        let mut reference: Vec<FloatEntry> = Vec::new();
        for i in 0..2000 {
            let left = rng.below(2) == 0;
            let mw = 10.0 + rng.below(191) as f32;
            let mh = 5.0 + rng.below(116) as f32;
            let y_start = [0.0, 40.0, 80.0, rng.below(400) as f32][rng.below(4) as usize];
            place_both(&mut ctx, &mut reference, 400.0, left, mw, mh, y_start, i);
        }
        assert_eq!(ctx.len(), reference.len());
    }

    #[test]
    fn five_hundred_floats_match_reference() {
        // Regression: a dense float soup (alternating sides, stacked
        // starts) must place every float, matching the reference scan
        // exactly. The index keeps each placement sublinear; the
        // generous time bound below only guards against an accidental
        // return to quadratic behavior, not normal jitter.
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        let mut ctx = FloatCtx::new();
        let mut reference: Vec<FloatEntry> = Vec::new();
        let t0 = std::time::Instant::now();
        for i in 0..500 {
            let left = i % 2 == 0;
            let mw = 20.0 + rng.below(141) as f32;
            let mh = 10.0 + rng.below(91) as f32;
            // Starts pile onto the same rows, forcing deep band scans
            // in the reference implementation.
            let y_start = (rng.below(6) * 30) as f32;
            place_both(&mut ctx, &mut reference, 320.0, left, mw, mh, y_start, i);
        }
        let dt = t0.elapsed();
        assert_eq!(ctx.len(), 500);
        assert!(
            dt.as_secs() < 10,
            "500 float placements took {dt:?}: possible quadratic regression"
        );
    }

    #[test]
    fn touching_edges_do_not_overlap() {
        // A float whose band exactly touches another's bottom edge
        // shares no band with it: the second float starts a new row at
        // the left edge (y=50) instead of being pushed below a
        // phantom overlap. Both implementations must agree.
        let mut ctx = FloatCtx::new();
        let mut reference: Vec<FloatEntry> = Vec::new();
        place_both(&mut ctx, &mut reference, 400.0, true, 100.0, 50.0, 0.0, 0);
        place_both(&mut ctx, &mut reference, 400.0, true, 100.0, 50.0, 50.0, 1);
        let (x, y) = (ctx.entries[1].x, ctx.entries[1].y);
        assert_eq!(
            (x.to_bits(), y.to_bits()),
            (0.0f32.to_bits(), 50.0f32.to_bits())
        );
    }

    #[test]
    fn tuck_under_short_float_is_highest_possible() {
        // Rules 8/9: after a tall left float A (200x100) and a short
        // left float B (200x10) beside it, a third left float C
        // (150x10) tucks into the notch below B, beside A — at y=10,
        // not stranded below A's bottom at y=100.
        let mut ctx = FloatCtx::new();
        let mut reference: Vec<FloatEntry> = Vec::new();
        place_both(&mut ctx, &mut reference, 400.0, true, 200.0, 100.0, 0.0, 0);
        place_both(&mut ctx, &mut reference, 400.0, true, 200.0, 10.0, 0.0, 1);
        place_both(&mut ctx, &mut reference, 400.0, true, 150.0, 10.0, 0.0, 2);
        let c = ctx.entries[2];
        assert_eq!(
            (c.x.to_bits(), c.y.to_bits()),
            (200.0f32.to_bits(), 10.0f32.to_bits()),
            "tucked float at {c:?}"
        );
    }

    #[test]
    fn later_float_top_not_higher_than_earlier_float_top() {
        // CSS 2.1 §9.5.1 rule 5, floated-box clause: a later float's
        // outer top may not be higher than any earlier float's outer
        // top. A (left 100x100) sits at (0, 0); B (left 350x10) cannot
        // fit beside A in a 400px container and steps to y=100; C
        // (left 50x10) would fit beside A at (100, 0), but that is
        // above B's top (100), so C must land at (350, 100), beside B.
        let mut ctx = FloatCtx::new();
        let mut reference: Vec<FloatEntry> = Vec::new();
        place_both(&mut ctx, &mut reference, 400.0, true, 100.0, 100.0, 0.0, 0);
        place_both(&mut ctx, &mut reference, 400.0, true, 350.0, 10.0, 0.0, 1);
        place_both(&mut ctx, &mut reference, 400.0, true, 50.0, 10.0, 0.0, 2);
        let (a, b, c) = (ctx.entries[0], ctx.entries[1], ctx.entries[2]);
        assert_eq!(
            (a.x.to_bits(), a.y.to_bits()),
            (0.0f32.to_bits(), 0.0f32.to_bits()),
            "A at {a:?}"
        );
        assert_eq!(
            (b.x.to_bits(), b.y.to_bits()),
            (0.0f32.to_bits(), 100.0f32.to_bits()),
            "B at {b:?}"
        );
        assert_eq!(
            (c.x.to_bits(), c.y.to_bits()),
            (350.0f32.to_bits(), 100.0f32.to_bits()),
            "C at {c:?}: rule 5 forbids (100, 0), above B's top"
        );
    }

    #[test]
    fn ascending_bottoms_step_in_order() {
        // Rules 8/9 over several candidates: A (100x30) and B (100x10)
        // stack at the left; C (350x10) must step past B's bottom (10)
        // first — where A still blocks — then past A's bottom (30).
        let mut ctx = FloatCtx::new();
        let mut reference: Vec<FloatEntry> = Vec::new();
        place_both(&mut ctx, &mut reference, 400.0, true, 100.0, 30.0, 0.0, 0);
        place_both(&mut ctx, &mut reference, 400.0, true, 100.0, 10.0, 0.0, 1);
        place_both(&mut ctx, &mut reference, 400.0, true, 350.0, 10.0, 0.0, 2);
        let c = ctx.entries[2];
        assert_eq!(
            (c.x.to_bits(), c.y.to_bits()),
            (0.0f32.to_bits(), 30.0f32.to_bits()),
            "stepped float at {c:?}"
        );
    }

    #[test]
    fn opposite_side_float_blocks_and_steps_below() {
        // Rule 3 + rules 8/9: a right float narrows the band for a
        // later left float; one too wide to fit beside it steps to
        // just below the right float's bottom, not further.
        let mut ctx = FloatCtx::new();
        let mut reference: Vec<FloatEntry> = Vec::new();
        place_both(&mut ctx, &mut reference, 400.0, false, 100.0, 40.0, 0.0, 0);
        place_both(&mut ctx, &mut reference, 400.0, true, 250.0, 10.0, 0.0, 1);
        place_both(&mut ctx, &mut reference, 400.0, true, 350.0, 10.0, 0.0, 2);
        let (l1, l2) = (ctx.entries[1], ctx.entries[2]);
        // L1 fits beside R: its right edge (250) stops before R's left
        // edge (300).
        assert_eq!(
            (l1.x.to_bits(), l1.y.to_bits()),
            (0.0f32.to_bits(), 0.0f32.to_bits()),
            "L1 at {l1:?}"
        );
        assert!(l1.x + l1.w <= 300.0, "rule 3: L1 {l1:?}");
        // L2 cannot fit beside R (350 > 300), so it tucks directly
        // under R's bottom at y=40.
        assert_eq!(
            (l2.x.to_bits(), l2.y.to_bits()),
            (0.0f32.to_bits(), 40.0f32.to_bits()),
            "L2 at {l2:?}"
        );
    }

    #[test]
    fn overwide_float_overflows_inline_end() {
        // Rule 7's conflict case: a float wider than the containing
        // block cannot fit beside or below anything, so it places at
        // the float side and overflows like an over-wide line box.
        let mut ctx = FloatCtx::new();
        let mut reference: Vec<FloatEntry> = Vec::new();
        place_both(&mut ctx, &mut reference, 400.0, true, 500.0, 10.0, 0.0, 0);
        let f = ctx.entries[0];
        assert_eq!(
            (f.x.to_bits(), f.y.to_bits()),
            (0.0f32.to_bits(), 0.0f32.to_bits()),
            "overwide float at {f:?}"
        );
    }
}

/// Reference O(F)-per-band float placement: the pre-index scan, kept
/// for differential testing (bit-identical results required).
#[cfg(test)]
fn place_float_reference(
    floats: &[FloatEntry],
    content_x: f32,
    content_w: f32,
    left: bool,
    mw: f32,
    mh: f32,
    y_start: f32,
) -> (f32, f32) {
    // Same rule 5 floated-box floor as `place_float`: no higher than
    // the outer top of any earlier float (naive max scan here).
    let mut y = y_start;
    for f in floats {
        y = y.max(f.y);
    }
    loop {
        let mut used_l: f32 = 0.0;
        let mut used_r: f32 = 0.0;
        // Bottoms of the floats overlapping this band: `next` is the
        // lowest (the next candidate y on a miss, rules 8/9); `lowest`
        // is the highest, used only to detect the empty band.
        let mut next = f32::INFINITY;
        let mut lowest = y;
        for f in floats {
            if f.y < y + mh && f.y + f.h > y {
                if f.left {
                    used_l = used_l.max(f.x + f.w - content_x);
                } else {
                    used_r = used_r.max(content_x + content_w - f.x);
                }
                let bottom = f.y + f.h;
                next = next.min(bottom);
                lowest = lowest.max(bottom);
            }
        }
        // Either it fits beside the overlapping floats (each earlier
        // same-side float is then beside or above it, §9.5.1 rules
        // 1-2, and its outer edge stops at the nearest overlapping
        // opposite-side float, rule 3), or nothing overlaps and it
        // places anyway.
        if mw <= content_w - used_l - used_r || lowest <= y {
            let x = if left {
                content_x + used_l
            } else {
                content_x + content_w - used_r - mw
            };
            return (x, y);
        }
        y = next;
    }
}

/// Clearance: the lowest bottom margin edge of the floats on the
/// relevant side(s), or `None` when no such float exists (CSS 2.1
/// §9.5.2). Callers compare this against the hypothetical top border
/// edge (which already folds in collapsed margins, including negative
/// ones) and take the greater; the margin box is already baked into the
/// entries.
///
/// NOTE: no y-ordering of `floats` is assumed here (and none holds in
/// general: multicol column assignment moves entries out of insertion
/// order — see [`FloatCtx::reindex`]), so this stays a full scan. The
/// indexed equivalent is [`FloatCtx::clearance_y`].
fn clearance_y(floats: &[FloatEntry], clear: Clear) -> Option<f32> {
    let mut y: Option<f32> = None;
    for f in floats {
        let relevant = match clear {
            Clear::Left => f.left,
            Clear::Right => !f.left,
            Clear::Both => true,
            Clear::None => false,
        };
        if relevant {
            y = Some(y.map_or(f.y + f.h, |b: f32| b.max(f.y + f.h)));
        }
    }
    y
}

/// Horizontal space taken by floats across the vertical band `[y0, y1)`:
/// the line-box x shift (left floats push the line right) and the
/// available width. A float intrudes when its margin box overlaps any
/// part of the band, and the worst intrusion over the band wins, so a
/// tall float starting below the band's top still narrows/shifts it
/// (CSS 2.1 §9.5). Still O(floats): `floats` is not y-ordered in
/// general (multicol column assignment moves entries out of insertion
/// order — see [`FloatCtx::reindex`]), so no early-out on
/// `f.y >= y1` is sound here. The indexed equivalent is
/// [`FloatCtx::line_float_avail`].
fn line_float_avail(
    floats: &[FloatEntry],
    content_x: f32,
    content_w: f32,
    y0: f32,
    y1: f32,
) -> (f32, f32) {
    let mut left: f32 = 0.0;
    let mut right: f32 = 0.0;
    for f in floats {
        if f.y < y1 && f.y + f.h > y0 {
            if f.left {
                left = left.max(f.x + f.w - content_x);
            } else {
                right = right.max(content_x + content_w - f.x);
            }
        }
    }
    (left, (content_w - left - right).max(0.0))
}

/// Lowest bottom margin edge of the floats overlapping the vertical
/// band `[y0, y1)` (`y0` itself when none overlap). Used to clear a box
/// or a line below floats that leave no usable space beside them.
/// Full scan — see the [`line_float_avail`] note on y-ordering. The
/// indexed equivalent is [`FloatCtx::intruding_float_bottom`].
fn intruding_float_bottom(floats: &[FloatEntry], y0: f32, y1: f32) -> f32 {
    let mut bottom = y0;
    for f in floats {
        if f.y < y1 && f.y + f.h > y0 {
            bottom = bottom.max(f.y + f.h);
        }
    }
    bottom
}

/// Shallowest bottom margin edge of the floats overlapping the vertical
/// band `[y0, y1)` (`y0` itself when none overlap). A line with no usable
/// width steps here and re-tests, settling just below the shallowest
/// intruding float instead of clearing them all at once. Full scan —
/// see the [`line_float_avail`] note on y-ordering. The indexed
/// equivalent is [`FloatCtx::min_intruding_float_bottom`].
fn min_intruding_float_bottom(floats: &[FloatEntry], y0: f32, y1: f32) -> f32 {
    let mut bottom = f32::INFINITY;
    for f in floats {
        if f.y < y1 && f.y + f.h > y0 {
            bottom = bottom.min(f.y + f.h);
        }
    }
    if bottom.is_infinite() { y0 } else { bottom }
}

#[cfg(test)]
mod float_band_query_tests {
    use super::{
        Clear, FloatCtx, FloatEntry, clearance_y, intruding_float_bottom, line_float_avail,
        min_intruding_float_bottom, place_float,
    };

    /// Deterministic xorshift64: reproducible pseudo-random inputs with
    /// no external dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
        fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (self.next() as f32 / u64::MAX as f32) * (hi - lo)
        }
    }

    fn random_entry(rng: &mut Rng, i: usize, y_lo: f32, y_hi: f32) -> FloatEntry {
        FloatEntry {
            x: rng.range_f32(0.0, 700.0),
            y: rng.range_f32(y_lo, y_hi),
            w: rng.range_f32(10.0, 120.0),
            h: rng.range_f32(0.0, 120.0),
            left: rng.below(2) == 0,
            node: i,
        }
    }

    /// Build a context two ways: `ordered` places floats through
    /// [`place_float`] like block layout (insertion order ends up
    /// y-ordered); `!ordered` pushes entries with arbitrary tops like
    /// multicol column assignment does (insertion order is NOT
    /// y-ordered). Either way the interval index mirrors `entries`.
    fn random_ctx(rng: &mut Rng, n: usize, ordered: bool) -> FloatCtx {
        let mut ctx = FloatCtx::new();
        if ordered {
            for i in 0..n {
                let e = random_entry(rng, i, 0.0, 2000.0);
                let (x, y) = place_float(&mut ctx, 0.0, 800.0, e.left, e.w, e.h, e.y);
                ctx.push(FloatEntry { x, y, ..e });
            }
        } else {
            for i in 0..n {
                ctx.push(random_entry(rng, i, -500.0, 2000.0));
            }
        }
        ctx
    }

    fn assert_same(rng: &mut Rng, ctx: &mut FloatCtx, rounds: usize, tag: &str) {
        for r in 0..rounds {
            let y0 = rng.range_f32(-600.0, 2200.0);
            // Bands from point tests to tall bands, sometimes empty,
            // sometimes covering everything.
            let y1 = y0 + [0.0001, 1.0, 16.0, 100.0, 500.0, 4000.0][r % 6];
            let content_x = rng.range_f32(0.0, 50.0);
            let content_w = 800.0 - content_x;

            let (l1, a1) = line_float_avail(&ctx.entries, content_x, content_w, y0, y1);
            let (l2, a2) = ctx.line_float_avail(content_x, content_w, y0, y1);
            assert_eq!(
                l1.to_bits(),
                l2.to_bits(),
                "{tag} line left r={r} y0={y0} y1={y1}"
            );
            assert_eq!(
                a1.to_bits(),
                a2.to_bits(),
                "{tag} line avail r={r} y0={y0} y1={y1}"
            );

            let b1 = intruding_float_bottom(&ctx.entries, y0, y1);
            let b2 = ctx.intruding_float_bottom(y0, y1);
            assert_eq!(
                b1.to_bits(),
                b2.to_bits(),
                "{tag} intruding r={r} y0={y0} y1={y1}"
            );

            let m1 = min_intruding_float_bottom(&ctx.entries, y0, y1);
            let m2 = ctx.min_intruding_float_bottom(y0, y1);
            assert_eq!(
                m1.to_bits(),
                m2.to_bits(),
                "{tag} min-intruding r={r} y0={y0} y1={y1}"
            );

            for clear in [Clear::None, Clear::Left, Clear::Right, Clear::Both] {
                let c1 = clearance_y(&ctx.entries, clear);
                let c2 = ctx.clearance_y(clear);
                assert_eq!(
                    c1.map(|x| x.to_bits()),
                    c2.map(|x| x.to_bits()),
                    "{tag} clearance {clear:?} r={r}"
                );
            }
        }
    }

    #[test]
    fn band_queries_match_naive_ordered() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        for &n in &[0usize, 1, 2, 7, 64, 300] {
            let mut ctx = random_ctx(&mut rng, n, true);
            assert_same(&mut rng, &mut ctx, 200, "ordered");
        }
    }

    #[test]
    fn band_queries_match_naive_unordered() {
        // Multicol-style: arbitrary insertion order, negative tops,
        // zero-height floats. This is the case that disproves the
        // y-ordering assumption — the indexed queries must not depend
        // on it.
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        for &n in &[0usize, 1, 2, 7, 64, 300] {
            let mut ctx = random_ctx(&mut rng, n, false);
            assert_same(&mut rng, &mut ctx, 200, "unordered");
        }
    }

    #[test]
    fn band_queries_match_after_multicol_style_reorder() {
        // Place floats in y order (block layout), then reassign their
        // tops out of order and reindex — exactly what
        // `assign_column_floats` does when it moves entries to
        // column-placed geometry (a whole-pushed float can land above
        // an earlier entry's top).
        let mut rng = Rng(0x0bad_f00d_5eed_1e55);
        let mut ctx = random_ctx(&mut rng, 128, true);
        assert!(ctx.entries.windows(2).all(|w| w[0].y <= w[1].y));
        // Reverse the tops: maximally out of order.
        let tops: Vec<f32> = ctx.entries.iter().map(|e| e.y).rev().collect();
        for (e, &y) in ctx.entries.iter_mut().zip(&tops) {
            e.y = y;
        }
        ctx.reindex();
        assert!(!ctx.entries.windows(2).all(|w| w[0].y <= w[1].y));
        assert_same(&mut rng, &mut ctx, 500, "reordered");
    }

    #[test]
    fn truncate_matches_fresh_ctx() {
        // Scrollbar re-drive regression (stale float treap after
        // truncate, pre-existing since w52): after `truncate(cut)`,
        // the context must be indistinguishable from one that only
        // ever saw the first `cut` entries — for band queries AND
        // for `place_float`'s `max_top` floor. A bare
        // `entries.truncate` leaves stale index nodes whose ids
        // alias (or run past) the re-pushed floats, and a stale
        // `max_top`; `place_float` consumes both, so the re-drive's
        // second pass would misplace floats or panic on an
        // out-of-bounds id.
        let mut rng = Rng(0x5eed_1e55_cafe_beef);
        for &n in &[1usize, 2, 7, 64, 200] {
            for &cut in &[0usize, 1, 2, n / 2, n] {
                let cut = cut.min(n);
                let full = random_ctx(&mut rng, n, true);
                let mut redriven = FloatCtx::new();
                for e in full.entries.iter() {
                    redriven.push(*e);
                }
                redriven.truncate(cut);
                assert_eq!(redriven.len(), cut);
                let mut fresh = FloatCtx::new();
                for e in full.entries.iter().take(cut) {
                    fresh.push(*e);
                }
                // Identical placement sequences on both: exercises
                // the interval index and the `max_top` floor, the
                // two consumers the re-drive's second pass relies
                // on.
                for i in 0..16 {
                    let left = rng.below(2) == 0;
                    let mw = rng.range_f32(20.0, 160.0);
                    let mh = rng.range_f32(10.0, 90.0);
                    let ys = rng.range_f32(0.0, 2000.0);
                    let p1 = place_float(&mut redriven, 0.0, 800.0, left, mw, mh, ys);
                    let p2 = place_float(&mut fresh, 0.0, 800.0, left, mw, mh, ys);
                    assert_eq!(
                        (p1.0.to_bits(), p1.1.to_bits()),
                        (p2.0.to_bits(), p2.1.to_bits()),
                        "cut={cut} n={n} i={i}"
                    );
                    redriven.push(FloatEntry {
                        x: p1.0,
                        y: p1.1,
                        w: mw,
                        h: mh,
                        left,
                        node: 1000 + i,
                    });
                    fresh.push(FloatEntry {
                        x: p2.0,
                        y: p2.1,
                        w: mw,
                        h: mh,
                        left,
                        node: 1000 + i,
                    });
                }
                // Indexed queries still match the naive scan after
                // the re-place sequence.
                assert_same(&mut rng, &mut redriven, 100, "redriven");
                assert_same(&mut rng, &mut fresh, 100, "fresh");
            }
        }
    }

    #[test]
    fn band_queries_empty_and_degenerate() {
        let mut ctx = FloatCtx::new();
        let mut rng = Rng(1);
        assert_same(&mut rng, &mut ctx, 20, "empty");
        // Zero-height and zero-width floats, negative geometry.
        ctx.push(FloatEntry {
            x: 10.0,
            y: 100.0,
            w: 50.0,
            h: 0.0,
            left: true,
            node: 0,
        });
        ctx.push(FloatEntry {
            x: 20.0,
            y: -30.0,
            w: 0.0,
            h: 40.0,
            left: false,
            node: 1,
        });
        assert_same(&mut rng, &mut ctx, 100, "degenerate");
    }

    /// Micro-benchmark: float-heavy context, many band queries —
    /// visits and wall time for the naive scans vs the indexed
    /// queries. Run with `-- --nocapture` to see the numbers.
    #[test]
    fn float_band_scan_reduction() {
        use std::time::Instant;
        let mut rng = Rng(0xc0ff_ee11_22dd_4455);
        // Unordered (adversarial for any ordering assumption), tall
        // floats so bands genuinely overlap several entries.
        let mut ctx = FloatCtx::new();
        for i in 0..3000 {
            ctx.push(FloatEntry {
                x: rng.range_f32(0.0, 700.0),
                y: rng.range_f32(0.0, 4000.0),
                w: rng.range_f32(20.0, 100.0),
                h: rng.range_f32(50.0, 400.0),
                left: rng.below(2) == 0,
                node: i,
            });
        }
        let bands: Vec<(f32, f32)> = (0..2000)
            .map(|_| {
                let y0 = rng.range_f32(0.0, 4000.0);
                (y0, y0 + 16.0)
            })
            .collect();

        // Naive: 3000 predicate evals per band by construction.
        let t0 = Instant::now();
        let mut naive_acc = 0.0f32;
        for &(y0, y1) in &bands {
            let (l, a) = line_float_avail(&ctx.entries, 0.0, 800.0, y0, y1);
            let b = intruding_float_bottom(&ctx.entries, y0, y1);
            naive_acc += l + a + b;
        }
        let naive_dt = t0.elapsed();

        // Indexed: count actual entry visits.
        let t0 = Instant::now();
        let mut indexed_acc = 0.0f32;
        let mut visits = 0u64;
        for &(y0, y1) in &bands {
            let (l, a) = ctx.line_float_avail(0.0, 800.0, y0, y1);
            let b = ctx.intruding_float_bottom(y0, y1);
            indexed_acc += l + a + b;
            let mut stack = Vec::new();
            ctx.index
                .for_each_in_band(y0, y1, &mut stack, |_| visits += 1);
        }
        let indexed_dt = t0.elapsed();
        // clearance_y over the whole set, both paths.
        let t0 = Instant::now();
        let mut c1 = 0.0f32;
        for _ in 0..2000 {
            c1 += clearance_y(&ctx.entries, Clear::Both).unwrap_or(0.0);
        }
        let clear_naive_dt = t0.elapsed();
        let t0 = Instant::now();
        let mut c2 = 0.0f32;
        for _ in 0..2000 {
            c2 += ctx.clearance_y(Clear::Both).unwrap_or(0.0);
        }
        let clear_indexed_dt = t0.elapsed();

        assert_eq!(naive_acc.to_bits(), indexed_acc.to_bits());
        assert_eq!(c1.to_bits(), c2.to_bits());
        let naive_visits = 3000u64 * bands.len() as u64 * 2; // two scans per band
        println!(
            "float_band_scan_reduction: floats=3000 bands={} \
             naive_visits={} indexed_visits={} visit_ratio={:.2}x \
             naive={:?} indexed={:?} time_ratio={:.2}x \
             clearance_naive={:?} clearance_indexed={:?}",
            bands.len(),
            naive_visits,
            visits,
            naive_visits as f64 / visits.max(1) as f64,
            naive_dt,
            indexed_dt,
            naive_dt.as_secs_f64() / indexed_dt.as_secs_f64().max(1e-9),
            clear_naive_dt,
            clear_indexed_dt,
        );
        assert!(
            visits < naive_visits,
            "index must skip floats outside the band"
        );
    }
}

/// Post-layout BFC/float band re-test (CSS 2.1 §9.5): the downward shift
/// a just-laid-out BFC child needs, or 0.0.
///
/// The pre-layout narrowing band in `resume_block` is only a point test
/// at the top edge for auto-height children (`None => cy.next_up()`),
/// so a float starting strictly below the child's top edge is invisible
/// to it even when it overlaps the child's body. Now that the true
/// border-box extent `[cy, cy + h)` is known, re-test over it: a BFC
/// root's border box must not overlap any float's margin box in the
/// same context, so on violation the child clears below the intruding
/// floats. Each pass strictly increases `cy`: the shift target is the
/// deepest bottom among floats that horizontally overlap the child's
/// border box, and the float that witnesses the violation guarantees it
/// is strictly below the current `cy`. Over a finite float set the loop
/// therefore provably terminates. Only the auto-height case is tested:
/// with a specified height the pre-layout band already covered the full
/// extent.
fn bfc_post_layout_shift(forest: &Forest, fr: &BlockFrame, c: usize) -> f32 {
    if !establishes_bfc(forest, c) || specified_height(forest, c, fr.child_cb_h).is_some() {
        return 0.0;
    }
    let (cy0, h, bx0, bx1) = {
        let n = &forest.nodes[c];
        (n.rect.y, n.rect.h, n.rect.x, n.rect.x + n.rect.w)
    };
    if h <= 0.0 {
        return 0.0;
    }
    let mut shift = 0.0;
    loop {
        let cy = cy0 + shift;
        let y1 = cy + h;
        let floats = fr.floats.borrow();
        let violated = floats
            .entries
            .iter()
            .any(|f| f.y < y1 && f.y + f.h > cy && f.x < bx1 && f.x + f.w > bx0);
        if !violated {
            break;
        }
        // The shift target uses the same horizontal-overlap predicate as
        // `violated`: a tall float that does not horizontally intrude
        // must not fling the child below the floats it actually violates.
        let bottom = floats
            .entries
            .iter()
            .filter(|f| f.y < y1 && f.y + f.h > cy && f.x < bx1 && f.x + f.w > bx0)
            .map(|f| f.y + f.h)
            .fold(cy, f32::max);
        drop(floats);
        if bottom <= cy {
            break;
        }
        shift = bottom - cy0;
    }
    shift
}

pub(crate) struct BlockFrame {
    idx: usize,
    child_i: usize,
    cursor_y: f32,
    prev_mbottom: f32,
    first: bool,
    top_collapsed: bool,
    content_x: f32,
    content_top: f32,
    content_w: f32,
    cb_h: Option<f32>,
    child_cb_h: Option<f32>,
    /// The container's own specified height as the prologue computed it.
    /// Equals `child_cb_h` except for a table cell's measurement pass
    /// (css-tables-3 §3.10.6), where the base threaded to children is
    /// indefinite but the cell's own height floor stays definite.
    own_hspec: Option<f32>,
    /// See [`BlockInput::height_is_min`]: carried into the resume pass so
    /// the final content height floors at the specified height instead of
    /// clamping to it.
    height_is_min: bool,
    /// Floats placed in this block formatting context (shared with
    /// non-BFC descendants; see [`BlockInput::floats`]).
    floats: FloatList,
    /// Length of `floats` when this frame was created: entries from
    /// here on are this subtree's own floats, used for BFC containment.
    floats_start: usize,
    /// Margin-box top stashed when a float child is encountered; the
    /// float is positioned when its `Task::Block` completes and its
    /// size is known.
    pending_float_y: Option<f32>,
    /// Maximum outer (margin-box) top (provisional frame) over the
    /// in-flow children accounted so far: each child's border top
    /// minus its top placement margin. CSS 2.1 §9.5.1 rules 5/6 forbid
    /// a float's outer top from rising above the outer top of any
    /// earlier in-flow block box or line box; the flow cursor alone
    /// does not enforce it (a later sibling's negative margin can drag
    /// the cursor above an earlier sibling's top), so the float's
    /// stashed y is floored by this.
    max_inflow_top: f32,
    /// Whether this frame already is the `overflow-y: auto`
    /// scrollbar-width re-drive (see `resume_block`'s completion
    /// branch): the re-drive runs at most once per frame, so narrowing
    /// can never oscillate or loop to a fixpoint.
    auto_redriven: bool,
}

pub(crate) struct InlineFrame {
    idx: usize,
    content_x: f32,
    content_top: f32,
    content_w: f32,
    cb_h: Option<f32>,
    /// The container's specified height as the prologue computed it —
    /// including a caller `height_override`, which re-deriving from
    /// style at resume time would drop.
    hspec: Option<f32>,
    /// See [`BlockInput::height_is_min`].
    height_is_min: bool,
    /// Floats intruding into this inline formatting context.
    floats: FloatList,
}

// The table/grid resume frames are heap-boxed: at ~568B/~304B they dwarf the
// other variants, and boxing keeps `Task` itself pointer-sized so the task
// stack's per-box push/pop traffic stays small. One alloc per table/grid
// task is the cheaper tradeoff here.
pub(crate) enum Task {
    Block(BlockInput),
    ResumeBlock(BlockFrame),
    ResumeInline(InlineFrame),
    Flex(FlexCtx),
    Table(Box<layout_table::TableResume>),
    Grid(Box<layout_grid::GridResume>),
}

/// Entry point: lay out the whole tree under `root`.
pub(crate) fn layout_forest(forest: &mut Forest, root: usize, fonts: &dyn FontMetrics) {
    layout_forest_with_root_cb(forest, root, fonts, None);
}

/// Full-document entry point: like [`layout_forest`], but the root is
/// laid out against the already-placed containing-block rect
/// `root_cb_rect` (`(x, w)`, `None` for the natural initial containing
/// block), and `viewport_scrollers` carries the `(document_element,
/// body)` DOM ids whose boxes skip the `overflow-y: auto`
/// scrollbar-width re-drive in `resume_block` — the app never paints a
/// nested scrollbar for them, so their vertical scrollbar geometry
/// belongs to the viewport pass.
pub(crate) fn layout_forest_document(
    forest: &mut Forest,
    root: usize,
    fonts: &dyn FontMetrics,
    root_cb_rect: Option<(f32, f32)>,
    viewport_scrollers: (Option<NodeId>, Option<NodeId>),
) {
    if root_cb_rect.is_none() && viewport_scrollers == (None, None) {
        layout_forest(forest, root, fonts);
    } else {
        layout_forest_inner(forest, root, fonts, root_cb_rect, viewport_scrollers);
    }
}

/// Like [`layout_forest`], but the root is laid out against a
/// containing-block width of `root_cb_w` instead of the viewport
/// width (wave 51 subtree layout): `forest.vp_w`/`vp_h` keep holding
/// the *true* viewport for every viewport-unit resolution during the
/// pass, and only the root task's containing-block width is split off
/// onto this parameter. The two quantities were previously conflated
/// by overwriting `forest.vp_w` with the containing width, which made
/// `vw`/`vmin`/`vmax` lengths inside a subtree pass resolve against
/// the containing width instead of the viewport.
pub(crate) fn layout_forest_with_root_cb(
    forest: &mut Forest,
    root: usize,
    fonts: &dyn FontMetrics,
    root_cb_w: Option<f32>,
) {
    layout_forest_inner(
        forest,
        root,
        fonts,
        root_cb_w.map(|w| (0.0, w)),
        (None, None),
    );
}

fn layout_forest_inner(
    forest: &mut Forest,
    root: usize,
    fonts: &dyn FontMetrics,
    root_cb_rect: Option<(f32, f32)>,
    viewport_scrollers: (Option<NodeId>, Option<NodeId>),
) {
    // Drop per-pass multicol state (pending column-rules, recorded
    // column-rule specs) before task creation so no stale entry leaks
    // into this pass.
    layout_multicol::begin_multicol_pass();
    // CSS 2.1 §17.2.1: generate the missing ancestor table boxes for
    // orphan table-internal boxes (e.g. a `<tbody>` moved out from under
    // its `<table>` by script) before task creation, so table layout
    // always sees a well-parented grid (WPT `html-to-css-mapping-1`).
    layout_table::wrap_orphan_table_boxes(forest);
    // CSS Overflow 4 §2.2: the root element's `scrollbar-gutter`
    // propagates to the viewport, so a `stable` gutter reserves its
    // strip against the *viewport*: the initial containing block the
    // root is laid out against shrinks by the reservation (and shifts
    // by the inline-start strip under `both-edges`), leaving the
    // root's border box — and everything sized from it — narrower by
    // the gutter. `layout_block` must not reserve the gutter again for
    // the root's children (see `Forest::gutter_propagated_root`).
    let (root_cb_x, root_cb_w) = match (root_cb_rect, forest.nodes[root].style.as_ref()) {
        // Viewport `overflow-y: auto` re-run: the caller already
        // narrowed (and shifted past an inline-start strip) the
        // initial containing block, per-side-maxed with the
        // stable-gutter reservation — so the root counts as
        // gutter-propagated exactly like the reservation arm below,
        // and neither reservation applies again.
        (Some((x, w)), _) => {
            forest.gutter_propagated_root = Some(root);
            (x, w)
        }
        (None, Some(style)) if forest.root_is_html => {
            let (gl, gr) = crate::viewport_gutter_reservation(style);
            if gl > 0.0 || gr > 0.0 {
                forest.gutter_propagated_root = Some(root);
                (gl, (forest.vp_w - gl - gr).max(0.0))
            } else {
                (0.0, forest.vp_w)
            }
        }
        (None, _) => (0.0, forest.vp_w),
    };
    let mut tasks = vec![Task::Block(BlockInput {
        abs_pct_height_base: None,
        idx: root,
        cb_x: root_cb_x,
        cb_w: root_cb_w,
        cb_h: Some(forest.vp_h),
        shrink: false,
        width_override: None,
        width_override_final: false,
        height_override: None,
        height_is_min: false,
        collapsed_border_wins: None,
        abspos: false,
        floats: fresh_floats(),
        is_float: false,
    })];
    // Preferred widths are memoized per layout pass (see below); the
    // cache is created fresh here so entries never survive past one
    // `layout()` call.
    let mut pw_cache = PrefWidthCache::new(forest.nodes.len());
    // Scratch buffers reused across every container in this pass (see
    // `LayoutScratch`).
    let mut scratch = LayoutScratch::new();
    scratch.viewport_scroller_doms = viewport_scrollers;
    run_tasks(forest, &mut tasks, fonts, &mut pw_cache, &mut scratch);
    // Out-of-flow (`position: absolute | fixed`) subtrees were recorded,
    // not laid out, by the flow passes above; lay each out against its
    // containing block now, while every rect is still in the provisional
    // frame (the single `apply_pending_shifts` pass below moves static
    // and out-of-flow subtrees alike, so frames stay consistent).
    layout_position::layout_abs_items(forest, fonts, &mut pw_cache);
    // `position: relative` (and `sticky`, treated as relative) shifts the
    // box without changing its flow footprint: accumulate the offsets
    // into the deferred-shift fields the final pass flushes.
    layout_position::apply_relative_offsets(forest);
    // Deferred inline-block shifts (recorded O(1) per fragment during
    // inline layout) apply here in a single pass, so deeply nested
    // inline-blocks cost O(nodes) total instead of O(depth * nodes).
    // Rects were provisional until now; everything after this (paint,
    // `convert_forest`) sees final positions.
    apply_pending_shifts(forest, root);
    // Outside list-item markers hang in the margin area off final
    // rects: place them now that every border box is final.
    super::marker::position_markers(forest, fonts);
}

fn run_tasks(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    scratch: &mut LayoutScratch,
) {
    while let Some(task) = tasks.pop() {
        match task {
            Task::Block(input) => layout_block(forest, tasks, &input, fonts, pw_cache, scratch),
            Task::ResumeBlock(frame) => {
                resume_block(forest, tasks, frame, fonts, pw_cache, scratch)
            }
            Task::Table(frame) => layout_table::resume_table(forest, tasks, *frame),
            Task::Grid(frame) => {
                layout_grid::resume_grid(forest, tasks, *frame, fonts, pw_cache, scratch)
            }
            Task::Flex(ctx) => flex_resume(forest, tasks, ctx, fonts, pw_cache, scratch),
            Task::ResumeInline(frame) => {
                // Borrow, don't clone: `do_inline_layout` only reads the
                // list, and no task can `borrow_mut` it while this runs
                // (the sole push site is `resume_block`'s float
                // placement, which never interleaves with inline
                // layout).
                let floats = frame.floats.borrow();
                // Multicol params were stashed on the node in
                // `layout_block`'s prologue; copy them out before the
                // `&mut forest` call below.
                let mc = forest.nodes[frame.idx].multicol.as_deref().copied();
                let hspec = frame.hspec;
                let out = do_inline_layout(
                    forest,
                    frame.idx,
                    frame.content_x,
                    frame.content_top,
                    frame.content_w,
                    &floats.entries,
                    fonts,
                    scratch,
                    mc.as_ref(),
                    hspec,
                );
                finish_content_height(
                    forest,
                    frame.idx,
                    frame.cb_h,
                    min_floor_hspec(frame.height_is_min, hspec, out.height),
                    out.height,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Metric resolution
// ---------------------------------------------------------------------------

/// Resolve one specified margin against `cb_w`; `None` means `auto`.
fn margin_opt(
    spec: SpecMargin,
    cb_w: f32,
    fs: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
) -> Option<f32> {
    match spec {
        SpecMargin::Auto => None,
        SpecMargin::Len(s) => s.resolve(fs, root_fs, vp_w, vp_h, cb_w),
    }
}

/// A vertical-table ancestor's inline-axis (physical height) specified
/// size for the [`vwm_table_cb_h`] fold: the mirror of
/// [`specified_height`] along the inline axis. A vertical table's block
/// axis is horizontal, so [`specified_height`] (via
/// [`table_block_spec`]) returns its physical *width* — but the fold
/// accumulates physical *heights*, so the ancestor must contribute its
/// inline size instead: the physical `height` spec (the inline spec
/// under the [`resolve_metrics`] transpose, see [`table_inline_spec`])
/// resolved against the containing-block height `cb_h` (CSS 2.1 §10.5),
/// with the vertical border/padding as the inline-axis box-sizing
/// extras. Like [`specified_height`] this replays specified sizes only —
/// no min/max clamp, no overrides.
fn vwm_table_inline_specified(forest: &Forest, idx: usize, cb_h: Option<f32>) -> Option<f32> {
    debug_assert!(is_vertical_table(forest, idx));
    let d = &forest.nodes[idx].data;
    match table_inline_spec(forest, idx) {
        // Tables are never replaced, so `auto` (like a cyclic spec)
        // contributes no definite inline size: the fold goes indefinite,
        // the same way `specified_height`'s `Auto` arm yields `None` for
        // a non-replaced box.
        SpecLen::Auto => None,
        // Percentage heights need a definite containing-block height.
        SpecLen::Pct(_) if cb_h.is_none() => None,
        // CSS Values 4 §8.1.4 + CSS 2.1 §10.5: a `calc()` that
        // syntactically contained a `%` leaf is cyclic against an
        // indefinite containing-block height and behaves as `auto`,
        // exactly like a bare percentage — never resolving against the
        // zero base.
        SpecLen::Math { had_pct, .. } if had_pct && cb_h.is_none() => None,
        // CSS Sizing 3 §5: `stretch` fills the containing-block height.
        // Against an indefinite CB height there is nothing to fill, so
        // it behaves as `auto`.
        SpecLen::Stretch => {
            let base = cb_h?;
            let v = base.max(0.0);
            Some(match d.box_sizing {
                BoxSizing::BorderBox => {
                    let m = &forest.nodes[idx].metrics;
                    (v - (m.bt + m.pt + m.pb + m.bb)).max(0.0)
                }
                BoxSizing::ContentBox => v,
            })
        }
        spec => {
            let base = cb_h.unwrap_or(0.0);
            spec.resolve(
                d.font_size,
                forest.root_font_size,
                forest.vp_w,
                forest.vp_h,
                base,
            )
            .map(|v| {
                let v = v.max(0.0);
                match d.box_sizing {
                    // The inline axis is vertical: strip the vertical
                    // border/padding.
                    BoxSizing::BorderBox => {
                        let m = &forest.nodes[idx].metrics;
                        (v - (m.bt + m.pt + m.pb + m.bb)).max(0.0)
                    }
                    BoxSizing::ContentBox => v,
                }
            })
        }
    }
}

/// The definite containing-block *height* for `idx`, or `None` when the
/// CB height is indefinite. A vertical table's inline-axis specs are the
/// physical `height`/`min-height`/`max-height`, and per CSS 2.1 §10.5
/// their percentages resolve against the CB *height* — not `cb_w` — so
/// `resolve_metrics` uses this as the percentage base for those specs
/// when it is definite (margin/padding percentages still resolve
/// against the CB width, per spec).
///
/// `resolve_metrics` takes no `cb_h` parameter, so this reconstructs the
/// `cb_h` that `layout_block` threaded into the box. Out-of-flow boxes
/// read their recorded containing block's padding-box height, definite
/// exactly when `layout_block` saw it definite (the `height_definite`
/// stamp is final by the abspos post-pass). In-flow boxes fold the
/// ancestor chain of specified content heights (each parent's `hspec`,
/// i.e. `specified_height` against its own CB height), bottoming out at
/// the definite initial containing block. A vertical-table ancestor's
/// block axis is horizontal, so it contributes its inline-axis
/// (physical height) specified size instead of its physical width (see
/// [`vwm_table_inline_specified`]). Ancestor `height_override`s
/// (flex/grid stretch), abspos solved sizes, and aspect-ratio-
/// transferred heights are not replayed: in those corners the base reads
/// indefinite and the historical `cb_w` base applies, as before.
fn vwm_table_cb_h(forest: &Forest, idx: usize) -> Option<f32> {
    /// An out-of-flow box's CB height: the recorded containing block's
    /// padding-box height (mirrors `layout_position::cb_rect`), definite
    /// exactly when `layout_block` saw it definite. `None` when `idx`
    /// is not out-of-flow.
    fn abs_cb_h(forest: &Forest, idx: usize) -> Option<Option<f32>> {
        forest
            .abs_items
            .iter()
            .find(|a| a.node == idx)
            .map(|a| match a.containing_block {
                None => Some(forest.vp_h),
                Some(c) => {
                    let n = &forest.nodes[c];
                    n.height_definite
                        .then(|| (n.rect.h - n.metrics.bt - n.metrics.bb).max(0.0))
                }
            })
    }
    if let Some(h) = abs_cb_h(forest, idx) {
        return h;
    }
    // In-flow: gather ancestors up to the first out-of-flow box or the
    // root, then fold specified heights from the outside in (each
    // parent's `hspec` — this box's CB height — resolves against its own
    // CB height). Iterative per the crate's never-recurse invariant.
    let mut ancestors: Vec<usize> = Vec::new();
    let boundary: Option<f32>;
    let mut cur = idx;
    // Hang guard (mirrors `resolve_containing_block` in
    // `layout_position`): a parent cycle is unreachable from well-formed
    // box construction, but an internal bug in anonymous wrapping or
    // table fix-ups could stitch one in. Capping the steps at the node
    // count terminates any cycle; the guard fails closed at the initial
    // containing block instead of hanging the layout thread.
    let mut steps = forest.nodes.len();
    loop {
        if steps == 0 {
            boundary = Some(forest.vp_h);
            break;
        }
        steps -= 1;
        let p = match forest.nodes[cur].parent {
            Some(p) => p,
            // The layout root's CB is the initial containing block:
            // definite, viewport-sized.
            None => {
                boundary = Some(forest.vp_h);
                break;
            }
        };
        ancestors.push(p);
        if let Some(h) = abs_cb_h(forest, p) {
            boundary = h;
            break;
        }
        cur = p;
    }
    let mut cb_h = boundary;
    for &a in ancestors.iter().rev() {
        cb_h = if is_vertical_table(forest, a) {
            // A vertical table's block axis is horizontal, so
            // `specified_height` would contribute its physical *width*
            // here; the fold accumulates physical *heights*, so the
            // ancestor contributes its inline-axis (physical height)
            // size instead.
            vwm_table_inline_specified(forest, a, cb_h)
        } else {
            specified_height(forest, a, cb_h)
        };
    }
    cb_h
}

/// Resolve border/padding/margins/width into `metrics`, and set the
/// border-box x/width. `block_level` is false for inline-blocks (their
/// `auto` margins compute to zero instead of centering). `width_override`
/// forces the content width (shrink-to-fit). When `override_final` is
/// true, the override is a final flex §9.7 main size and skips the
/// min/max clamp, which would otherwise undo its min-wins outcome.
/// Idempotent.
fn resolve_metrics(
    forest: &mut Forest,
    idx: usize,
    cb_x: f32,
    cb_w: f32,
    block_level: bool,
    width_override: Option<f32>,
    override_final: bool,
) {
    // Tables in vertical writing modes consume width/height logically:
    // the width axis below is the inline axis, so it resolves from the
    // physical `height`/`min-height`/`max-height` properties (and the
    // border-box extras assembled into `rect.w` are the vertical ones).
    let vwm_table = is_vertical_table(forest, idx);
    // A vertical table's inline-axis specs (physical `height` /
    // `min-height` / `max-height`, destructured below) resolve their
    // percentages against the containing-block *height* when definite
    // (CSS 2.1 §10.5), not `cb_w`. Margins/padding/borders below keep
    // `cb_w`: their percentages resolve against the CB width per spec.
    let inline_pct_base = if vwm_table {
        vwm_table_cb_h(forest, idx).unwrap_or(cb_w)
    } else {
        cb_w
    };
    // Orthogonal-caption split base (w75f3, item 1): a horizontal
    // caption in a vertical table lays out against the table's block
    // size (`cb_w`, 300) for inline sizing, but css-writing-modes-3
    // resolves its percentage margins/padding against the containing
    // block's inline size — the table's inline content size (200),
    // which the parent table's already-resolved metrics carry as
    // `content_w`. The nonzero gate keeps intrinsic passes (where the
    // table has not been resolved yet and `content_w` is still 0) on
    // the old `cb_w` base. Only real captions qualify: a horizontal
    // float directly under a vertical table keeps `cb_w`, since its
    // containing-block inline size in its own writing mode is the
    // block size. Width/min/max percentages below stay on `cb_w`
    // (the inline-sizing base); only margins/padding use this base.
    let pct_base = {
        let is_orth_caption = matches!(
            forest.nodes[idx].data.writing_mode,
            WritingMode::HorizontalTb
        ) && forest.nodes[idx].parent.is_some_and(|p| {
            is_vertical_table(forest, p)
                && forest.nodes[idx]
                    .style
                    .as_ref()
                    .and_then(|s| s.get_str_by_index(props::DISPLAY))
                    .is_some_and(|v| v.eq_ignore_ascii_case("table-caption"))
        });
        if is_orth_caption {
            forest.nodes[idx]
                .parent
                .map(|p| forest.nodes[p].metrics.content_w)
                .filter(|&w| w > 0.0)
                .unwrap_or(cb_w)
        } else {
            cb_w
        }
    };
    let (
        fs,
        root_fs,
        vp_w,
        vp_h,
        width_spec,
        min_width,
        max_width,
        box_sizing,
        margin,
        padding,
        border,
        border_style,
        offset,
    ) = {
        let d = &forest.nodes[idx].data;
        (
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            if vwm_table { d.height } else { d.width },
            if vwm_table { d.min_height } else { d.min_width },
            if vwm_table { d.max_height } else { d.max_width },
            d.box_sizing,
            d.margin,
            d.padding,
            d.border,
            d.border_style,
            d.offset,
        )
    };
    let len0 = |s: SpecLen, base: f32| s.resolve(fs, root_fs, vp_w, vp_h, base).unwrap_or(0.0);
    // `border-style: none | hidden` forces the used border width to zero
    // (CSS 2.1 §8.5.3), whatever `border-width` says.
    let bw = |w: SpecLen, st: BorderStyle| {
        if st.zeroes_width() {
            0.0
        } else {
            len0(w, cb_w)
        }
    };
    let bl = bw(border[3], border_style[3]);
    let br = bw(border[1], border_style[1]);
    let bt = bw(border[0], border_style[0]);
    let bb = bw(border[2], border_style[2]);
    let pl = len0(padding[3], pct_base);
    let pr = len0(padding[1], pct_base);
    let pt = len0(padding[0], pct_base);
    let pb = len0(padding[2], pct_base);
    let ml = margin_opt(margin[3], pct_base, fs, root_fs, vp_w, vp_h);
    let mr = margin_opt(margin[1], pct_base, fs, root_fs, vp_w, vp_h);
    let mt = margin_opt(margin[0], pct_base, fs, root_fs, vp_w, vp_h).unwrap_or(0.0);
    let mb = margin_opt(margin[2], pct_base, fs, root_fs, vp_w, vp_h).unwrap_or(0.0);

    // The width axis is the inline axis: for a vertical table the
    // border-box extras on this axis are the vertical border/padding.
    let axis_extras = if vwm_table {
        bt + pt + pb + bb
    } else {
        bl + pl + pr + br
    };
    // min/max-width clamp the used width in the box-sizing's own terms
    // (content width for `content-box`, border-box width for
    // `border-box`); when both constrain, min-width wins per spec.
    // For a vertical table these are the transposed `min_height` /
    // `max_height` (the inline axis), destructured above: their
    // percentages resolve against the CB height when definite.
    let min_w = min_width
        .resolve(fs, root_fs, vp_w, vp_h, inline_pct_base)
        .unwrap_or(0.0)
        .max(0.0);
    let max_w = max_width
        .resolve(fs, root_fs, vp_w, vp_h, inline_pct_base)
        .unwrap_or(f32::INFINITY)
        .max(0.0);
    let clamp_w = |w: f32| w.min(max_w).max(min_w);
    let content_w = match width_override {
        // Shrink-to-fit hands us a content width already.
        // A final flex §9.7 main size already went through min/max
        // clamping with min-wins; re-clamping would undo that outcome.
        Some(w) if override_final => w.max(0.0),
        // The override is a content width, but `clamp_w` applies min/max
        // in the box-sizing's own terms (border-box width under
        // `box-sizing: border-box`). Clamp in content terms instead:
        // convert to border-box, clamp with min-width winning when
        // over-constrained (as in `clamp_w`), then strip the box's own
        // extras back off. Otherwise e.g. `min-width: 300px` on a
        // border-box box with 10px borders floors the content at 300
        // (border box 320) instead of 280 (border box 300).
        Some(w) => match box_sizing {
            BoxSizing::BorderBox => {
                ((w.max(0.0) + axis_extras).min(max_w).max(min_w) - axis_extras).max(0.0)
            }
            BoxSizing::ContentBox => clamp_w(w.max(0.0)),
        },
        None => match width_spec {
            // Replaced elements: the intrinsic width, still clamped by
            // min/max-width. The intrinsic width is a *content* width;
            // clamp it in the box-sizing's own terms like the `Some(w)`
            // arm below (under `border-box` min/max-width bound the
            // border-box width), so e.g. `min-width: 200px` on a
            // border-box replaced box floors the border box at 200.
            //
            // Two refinements on the raw intrinsic width:
            // - unknown intrinsic width (no `w`) defaults to 300px
            //   (CSS 2.1 §10.3.2);
            // - unknown intrinsic width *with* an intrinsic ratio (SVG
            //   with a `viewBox` but no `width`/`height`) fill-resolves
            //   against the available width instead — the intrinsic
            //   width *is* the container width there (WPT
            //   `position-absolute-replaced-minmax` viewBox-only case).
            SpecLen::Auto => match forest.nodes[idx].intrinsic {
                Some(d) => match (d.w, d.h, d.ratio_or_derived()) {
                    // ViewBox-only SVG (no width, no height): the intrinsic
                    // width *is* the available width (fill), like a
                    // non-replaced block. For abspos with both left/right
                    // specified, the width solves from the insets (CSS 2.1
                    // §10.3.7); otherwise it fills the containing block.
                    // Strip the extras for the content box, mirroring the
                    // `None` arm below.
                    (None, None, Some(_)) => {
                        let fill = match (
                            offset[3].resolve(fs, root_fs, vp_w, vp_h, cb_w),
                            offset[1].resolve(fs, root_fs, vp_w, vp_h, cb_w),
                        ) {
                            (Some(l), Some(r)) => {
                                (cb_w - l - r - ml.unwrap_or(0.0) - mr.unwrap_or(0.0)).max(0.0)
                            }
                            _ => (cb_w - ml.unwrap_or(0.0) - mr.unwrap_or(0.0)).max(0.0),
                        };
                        match box_sizing {
                            BoxSizing::BorderBox => (clamp_w(fill) - axis_extras).max(0.0),
                            BoxSizing::ContentBox => clamp_w(fill - axis_extras).max(0.0),
                        }
                    }
                    // Known height + ratio, auto width: transfer through
                    // the ratio (CSS 2.1 §10.3.2).
                    (None, Some(h), Some(ratio)) => {
                        let iw = (h * ratio).max(0.0);
                        match box_sizing {
                            BoxSizing::BorderBox => {
                                ((iw + axis_extras).min(max_w).max(min_w) - axis_extras).max(0.0)
                            }
                            BoxSizing::ContentBox => clamp_w(iw),
                        }
                    }
                    _ => {
                        let iw = d.width_or_default().max(0.0);
                        match box_sizing {
                            BoxSizing::BorderBox => {
                                ((iw + axis_extras).min(max_w).max(min_w) - axis_extras).max(0.0)
                            }
                            BoxSizing::ContentBox => clamp_w(iw),
                        }
                    }
                },
                None => {
                    let fill = (cb_w - ml.unwrap_or(0.0) - mr.unwrap_or(0.0)).max(0.0);
                    match box_sizing {
                        BoxSizing::BorderBox => (clamp_w(fill) - axis_extras).max(0.0),
                        BoxSizing::ContentBox => clamp_w(fill - axis_extras).max(0.0),
                    }
                }
            },
            // Content-based widths with no override: `resolve_metrics`
            // cannot measure content (no font context), so they resolve
            // like `auto` (fill) here. Every real path overwrites this
            // with a measured width via an override (`layout_block`'s
            // shrink/table/content-based arms, flex/grid item setup);
            // the fill is only ever provisional.
            SpecLen::FitContent | SpecLen::MinContent | SpecLen::MaxContent => {
                let fill = (cb_w - ml.unwrap_or(0.0) - mr.unwrap_or(0.0)).max(0.0);
                match box_sizing {
                    BoxSizing::BorderBox => (clamp_w(fill) - axis_extras).max(0.0),
                    BoxSizing::ContentBox => clamp_w(fill - axis_extras).max(0.0),
                }
            }
            spec => {
                // The transposed inline spec of a vertical table (the
                // physical `height`): percentages resolve against the CB
                // height when definite (`inline_pct_base`).
                let specified = len0(spec, inline_pct_base).max(0.0);
                match box_sizing {
                    // The specified width already includes padding/border.
                    BoxSizing::BorderBox => (clamp_w(specified) - axis_extras).max(0.0),
                    BoxSizing::ContentBox => clamp_w(specified),
                }
            }
        },
    };
    let width_is_auto = width_override.is_none() && matches!(width_spec, SpecLen::Auto);
    let fixed = axis_extras + content_w;
    // CSS 2.1 §10.3.3: over-constrained values ignore the end margin in
    // the inline direction — `margin-right` under LTR, `margin-left`
    // under `direction: rtl` — where `direction` is the *containing
    // block's*, not the element's own. For block-level boxes the CB is
    // the parent (abspos/fixed resolve their own margins elsewhere).
    let cb_rtl = forest.nodes[idx]
        .parent
        .map(|p| forest.nodes[p].data.rtl)
        .unwrap_or(forest.nodes[idx].data.rtl);
    let (ml_u, mr_u) = if width_is_auto {
        (ml.unwrap_or(0.0), mr.unwrap_or(0.0))
    } else if block_level {
        match (ml, mr) {
            (None, None) => {
                let rem = cb_w - fixed;
                (rem / 2.0, rem / 2.0)
            }
            (None, Some(mr)) => (cb_w - fixed - mr, mr),
            (Some(ml), None) => (ml, cb_w - fixed - ml),
            (Some(_), Some(mr)) if cb_rtl => (cb_w - fixed - mr, mr),
            (Some(ml), Some(_)) => (ml, cb_w - fixed - ml),
        }
    } else {
        (ml.unwrap_or(0.0), mr.unwrap_or(0.0))
    };

    let n = &mut forest.nodes[idx];
    n.metrics.ml = ml_u;
    n.metrics.mr = mr_u;
    n.metrics.mt = mt;
    n.metrics.mb = mb;
    n.metrics.bl = bl;
    n.metrics.br = br;
    n.metrics.bt = bt;
    n.metrics.bb = bb;
    n.metrics.pl = pl;
    n.metrics.pr = pr;
    n.metrics.pt = pt;
    n.metrics.pb = pb;
    n.metrics.content_w = content_w;
    n.rect.x = cb_x + ml_u;
    // Axis-appropriate border-box assembly (see `axis_extras` above).
    n.rect.w = if vwm_table {
        n.metrics.bt + n.metrics.pt + content_w + n.metrics.pb + n.metrics.bb
    } else {
        bl + pl + content_w + pr + br
    };
}

/// Box-sizing used for `height` / `min-height` / `max-height`
/// resolution on a node: WHATWG Quirks §3.13 (the table cell height
/// box sizing quirk) — in full quirks mode only, elements with a
/// computed `display` of `table-cell` act as `border-box` for the
/// height properties. Width resolution keeps the specified
/// box-sizing, and limited-quirks / no-quirks documents use the
/// specified value for height too.
fn height_box_sizing(forest: &Forest, idx: usize) -> BoxSizing {
    if forest.quirks_mode == QuirksMode::Quirks
        && forest.nodes[idx].data.display == Display::TableCell
    {
        BoxSizing::BorderBox
    } else {
        forest.nodes[idx].data.box_sizing
    }
}

/// The specified content height, if any. Percentage heights resolve against
/// the containing-block height; `auto` (or a percentage — or a `calc()`
/// containing a percentage — of an `auto` height) is `None`. Under
/// `box-sizing: border-box` the specified height includes padding/border,
/// which are stripped for the content box.
fn specified_height(forest: &Forest, idx: usize, cb_h: Option<f32>) -> Option<f32> {
    let d = &forest.nodes[idx].data;
    // A vertical table's block (algorithm-height) size resolves from the
    // physical `width` property.
    let height_spec = table_block_spec(forest, idx);
    match height_spec {
        // Replaced elements: the intrinsic height is the used *content*
        // height when CSS height is auto. `box-sizing` only reinterprets a
        // specified height; with `auto` there is none, so border/padding
        // are never stripped from the intrinsic size.
        SpecLen::Auto => forest.nodes[idx]
            .intrinsic
            .map(|d| d.height_or_default().max(0.0)),
        // Percentage heights need a definite containing-block height.
        SpecLen::Pct(_) if cb_h.is_none() => None,
        // CSS Values 4 §8.1.4 + CSS 2.1 §10.5: a `calc()` that
        // syntactically contained a `%` leaf is cyclic against an
        // indefinite containing-block height and behaves as `auto`,
        // exactly like a bare percentage — never resolving against the
        // zero base. The gate checks `had_pct`, not the folded
        // coefficient, so `calc(10px + 0%)` is cyclic too.
        SpecLen::Math { had_pct, .. } if had_pct && cb_h.is_none() => None,
        // CSS Sizing 3 §5: `stretch` fills the containing-block
        // height. Against an indefinite CB height there is nothing to
        // fill, so it behaves as `auto`. Under `box-sizing:
        // border-box` the stretched border-box height is the CB
        // height, so padding/border are stripped like any specified
        // height.
        SpecLen::Stretch => {
            let base = cb_h?;
            let v = base.max(0.0);
            Some(match height_box_sizing(forest, idx) {
                BoxSizing::BorderBox => {
                    let m = &forest.nodes[idx].metrics;
                    let bp = if is_vertical_table(forest, idx) {
                        m.bl + m.pl + m.pr + m.br
                    } else {
                        m.bt + m.pt + m.pb + m.bb
                    };
                    (v - bp).max(0.0)
                }
                BoxSizing::ContentBox => v,
            })
        }
        spec => {
            let base = cb_h.unwrap_or(0.0);
            spec.resolve(
                d.font_size,
                forest.root_font_size,
                forest.vp_w,
                forest.vp_h,
                base,
            )
            .map(|v| {
                let v = v.max(0.0);
                match height_box_sizing(forest, idx) {
                    BoxSizing::BorderBox => {
                        let m = &forest.nodes[idx].metrics;
                        // A vertical table's block axis is horizontal:
                        // strip the horizontal border/padding.
                        let bp = if is_vertical_table(forest, idx) {
                            m.bl + m.pl + m.pr + m.br
                        } else {
                            m.bt + m.pt + m.pb + m.bb
                        };
                        (v - bp).max(0.0)
                    }
                    BoxSizing::ContentBox => v,
                }
            })
        }
    }
}

/// Clamp a content height by min-height/max-height. Percentages resolve
/// against the containing-block height and are ignored when it is
/// indefinite. min-height wins over max-height when both constrain. Under
/// `box-sizing: border-box` the min/max refer to the border-box height.
fn clamp_height(forest: &Forest, idx: usize, cb_h: Option<f32>, content_h: f32) -> f32 {
    let d = &forest.nodes[idx].data;
    let base = cb_h.unwrap_or(0.0);
    let resolve = |s: SpecLen| match s {
        SpecLen::Auto => None,
        SpecLen::Pct(_) if cb_h.is_none() => None,
        // CSS Values 4 §8.1.4: a `calc()` that syntactically contained
        // a `%` leaf is cyclic against an indefinite containing-block
        // height and behaves as the property's initial value — the
        // `min-height` arm falls back to 0, the `max-height` arm to
        // `none` (CSS 2.1 §10.7, §10.4) — never resolving against the
        // zero base. The gate checks `had_pct`, not the folded
        // coefficient.
        SpecLen::Math { had_pct, .. } if had_pct && cb_h.is_none() => None,
        spec => spec.resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            base,
        ),
    };
    // A vertical table's block (algorithm-height) size clamps against the
    // physical `min-width`/`max-width`, and the border-box extras on the
    // block axis are the horizontal border/padding.
    let vwm_table = is_vertical_table(forest, idx);
    let min_h = resolve(if vwm_table { d.min_width } else { d.min_height })
        .unwrap_or(0.0)
        .max(0.0);
    // Table cells ignore `max-height` (css-tables-3 height distribution
    // only defines `height` as a minimum; WPT
    // table-cell-overflow-explicit-height-002 requires a 20px max-height
    // to not clip a 300px cell).
    let max_h = if forest.nodes[idx].box_type == BoxType::TableCell {
        f32::INFINITY
    } else {
        resolve(if vwm_table { d.max_width } else { d.max_height })
            .unwrap_or(f32::INFINITY)
            .max(0.0)
    };
    let extras = if vwm_table {
        let m = &forest.nodes[idx].metrics;
        m.bl + m.pl + m.pr + m.br
    } else {
        let m = &forest.nodes[idx].metrics;
        m.bt + m.pt + m.pb + m.bb
    };
    match height_box_sizing(forest, idx) {
        BoxSizing::BorderBox => {
            let border_h = (content_h + extras).min(max_h).max(min_h);
            (border_h - extras).max(0.0)
        }
        BoxSizing::ContentBox => content_h.min(max_h).max(min_h),
    }
}

/// Transfer a definite content width through the preferred
/// `aspect-ratio` to a content height (css-sizing-4 §3.1): `height =
/// width / ratio`, in the `box-sizing`'s own terms — the ratio maps
/// content-box to content-box under `content-box`, and border-box to
/// border-box under `border-box` (padding/border re-added before the
/// transfer, stripped after).
fn aspect_ratio_content_height(
    ratio: f32,
    content_w: f32,
    box_sizing: BoxSizing,
    extras_h: f32,
    extras_v: f32,
) -> f32 {
    match box_sizing {
        BoxSizing::ContentBox => content_w / ratio,
        BoxSizing::BorderBox => ((content_w + extras_h) / ratio - extras_v).max(0.0),
    }
    .max(0.0)
}

/// Reverse transfer: a definite content height through the preferred
/// `aspect-ratio` to a content width (`width = height * ratio`), in the
/// `box-sizing`'s own terms.
fn aspect_ratio_content_width(
    ratio: f32,
    content_h: f32,
    box_sizing: BoxSizing,
    extras_h: f32,
    extras_v: f32,
) -> f32 {
    match box_sizing {
        BoxSizing::ContentBox => content_h * ratio,
        BoxSizing::BorderBox => ((content_h + extras_v) * ratio - extras_h).max(0.0),
    }
    .max(0.0)
}

/// Intrinsic width/height ratio of a replaced element (`None` for
/// non-replaced boxes or a degenerate/empty intrinsic size). Size
/// calculations involving a replaced element's natural aspect ratio work
/// with the content-box dimensions always (css-sizing-4 §4.1) — unlike
/// the `aspect-ratio` *property* transfers above, which map the
/// box-sizing's own box for a specified `<ratio>`.
pub(crate) fn intrinsic_ratio(forest: &Forest, idx: usize) -> Option<f32> {
    forest.nodes[idx].intrinsic?.ratio_or_derived()
}

/// Resolve `min-width`/`max-width` against the containing-block width in
/// the box-sizing's own terms (content width for `content-box`,
/// border-box width for `border-box`); min wins on conflict (CSS 2.1
/// §10.4). Shared by `resolve_metrics` and [`replaced_minmax_transfer`].
fn minmax_width_terms(forest: &Forest, idx: usize, cb_w: f32) -> (f32, f32) {
    let d = &forest.nodes[idx].data;
    let min_w = d
        .min_width
        .resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            cb_w,
        )
        .unwrap_or(0.0)
        .max(0.0);
    let max_w = d
        .max_width
        .resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            cb_w,
        )
        .unwrap_or(f32::INFINITY)
        .max(0.0);
    (min_w, max_w)
}

/// Resolve `min-height`/`max-height` to (min, max) in the box-sizing's own
/// terms (like [`minmax_width_terms`]): percentages resolve against the
/// containing-block height when definite, else compute to `auto`.
fn minmax_height_terms(forest: &Forest, idx: usize, cb_h: Option<f32>) -> (f32, f32) {
    let d = &forest.nodes[idx].data;
    let base = cb_h.unwrap_or(0.0);
    let resolve = |s: SpecLen| match s {
        SpecLen::Auto => None,
        SpecLen::Pct(_) if cb_h.is_none() => None,
        SpecLen::Math { had_pct, .. } if had_pct && cb_h.is_none() => None,
        spec => spec.resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            base,
        ),
    };
    let vwm_table = is_vertical_table(forest, idx);
    let min_h = resolve(if vwm_table { d.min_width } else { d.min_height })
        .unwrap_or(0.0)
        .max(0.0);
    let max_h = if forest.nodes[idx].box_type == BoxType::TableCell {
        f32::INFINITY
    } else {
        resolve(if vwm_table { d.max_width } else { d.max_height })
            .unwrap_or(f32::INFINITY)
            .max(0.0)
    };
    (min_h, max_h)
}

/// Clamp a content width by `min-width`/`max-width` in the box-sizing's
/// own terms (CSS 2.1 §10.4; min wins on conflict). `extras_h` is the
/// box's own border+padding width.
fn clamp_content_width_own_terms(
    min_w: f32,
    max_w: f32,
    box_sizing: BoxSizing,
    extras_h: f32,
    content_w: f32,
) -> f32 {
    match box_sizing {
        BoxSizing::BorderBox => {
            // Convert the border-box min/max bounds into content-box
            // terms once instead of round-tripping `content_w` through
            // border-box and back: when nothing is violated the width
            // passes through bit-identical, so `replaced_minmax_transfer`'s
            // `intrinsic_w != Some(w)` early-out still recognizes the
            // intrinsic width instead of tripping on `(w + extras) -
            // extras` float dust and recomputing the height through the
            // ratio.
            let min_c = (min_w - extras_h).max(0.0);
            let max_c = (max_w - extras_h).max(0.0);
            content_w.max(0.0).min(max_c).max(min_c)
        }
        BoxSizing::ContentBox => content_w.max(0.0).min(max_w).max(min_w),
    }
}

/// Sequential min/max violation algorithm for a replaced element with
/// an intrinsic ratio (CSS 2.1 §10.4 / §10.7):
///
///   1. clamp the tentative width by `min-width`/`max-width` (min wins on
///      conflict),
///   2. if the height is `auto`, recompute it from the used width through
///      the ratio.
///
/// Replaced-element min/max constraint table (CSS 2.1 §10.4). The
/// tentative sizes are clamped by min/max, with the intrinsic ratio
/// preserved according to which constraints win:
/// - no violation: tentative sizes stand;
/// - single-axis violation: clamp that axis, transfer through the ratio
///   to the other axis if it is auto;
/// - both axes over (max/max) or both under (min/min): the tighter
///   constraint wins (smaller quotient for max, larger for min), the
///   other axis transfers through the ratio;
/// - mixed min/max violations: both constraints win independently, no
///   ratio transfer.
///
/// The clamps apply in the `box-sizing`'s own terms; the ratio always
/// maps content-box dimensions ([`intrinsic_ratio`]). `tentative_w` /
/// `tentative_h` are content sizes: the intrinsic size for an `auto`
/// axis (after the §10.3.2 reverse transfer the caller folds in), the
/// used size otherwise. `width_final` pins the width (a flex §9.7 main
/// size — flexbox §9.4 step 5 never transfers back into it, and its
/// min/max outcome, including the automatic minimum, must not be
/// re-clamped); the height still derives from it. Returns the used
/// content width and height. When the width is the intrinsic width the
/// height transfer is the identity, so the common no-violation case
/// passes through untouched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn replaced_minmax_transfer(
    forest: &Forest,
    idx: usize,
    cb_w: f32,
    cb_h: Option<f32>,
    tentative_w: f32,
    tentative_h: f32,
    ratio: f32,
    width_auto: bool,
    height_auto: bool,
    width_final: bool,
) -> (f32, f32) {
    // CSS 2.1 §10.4: sequential min/max with ratio transfers.
    // 1. Clamp width by min/max-width.
    // 2. If height is auto, transfer width through ratio to height.
    // 3. Clamp height by min/max-height.
    // 4. If height was clamped and width is auto (not final, not fixed by
    //    a winning min/max-width), transfer back through ratio to width.
    // For both-axes violations (max/max or min/min), the tighter
    // constraint wins per the §10.4 table: compare the constraint ratios
    // and let the winner determine both axes via the ratio.
    let mut w = tentative_w.max(0.0);
    let mut h = tentative_h.max(0.0);

    let d = &forest.nodes[idx].data;
    let m = &forest.nodes[idx].metrics;
    let (min_w, max_w) = minmax_width_terms(forest, idx, cb_w);
    let (min_h, max_h) = minmax_height_terms(forest, idx, cb_h);

    // Check for both-axes violations to apply the §10.4 table. Use the
    // intrinsic sizes for the violation check (the tentative may already
    // be clamped upstream in `resolve_metrics`).
    let intrinsic = forest.nodes[idx].intrinsic;
    let iw = intrinsic.and_then(|d| d.w).unwrap_or(w);
    let ih = intrinsic.and_then(|d| d.h).unwrap_or(h);
    // For viewBox-only, the intrinsic width is the fill; use tentative.
    let iw = if intrinsic.is_some_and(|d| d.w.is_none()) {
        w
    } else {
        iw.max(0.0)
    };
    let ih = ih.max(0.0);

    let extras_w = m.bl + m.pl + m.pr + m.br;
    let extras_h = m.bt + m.pt + m.pb + m.bb;
    let to_cw = |v: f32| match d.box_sizing {
        BoxSizing::BorderBox => (v - extras_w).max(0.0),
        BoxSizing::ContentBox => v,
    };
    let to_ch = |v: f32| match d.box_sizing {
        BoxSizing::BorderBox => (v - extras_h).max(0.0),
        BoxSizing::ContentBox => v,
    };
    let min_wc = to_cw(min_w);
    let max_wc = to_cw(max_w);
    let min_hc = to_ch(min_h);
    let max_hc = to_ch(max_h);

    let w_over = iw > max_wc + 0.01;
    let w_under = iw < min_wc - 0.01;
    let h_over = ih > max_hc + 0.01;
    let h_under = ih < min_hc - 0.01;

    // Both over: tighter max wins.
    if w_over && h_over {
        if max_wc / iw <= max_hc / ih {
            w = max_wc;
            h = (w / ratio).max(0.0);
        } else {
            h = max_hc;
            w = (h * ratio).max(0.0);
        }
        return (w, h);
    }
    // Both under: tighter min wins.
    if w_under && h_under {
        if min_wc / iw >= min_hc / ih {
            w = min_wc;
            h = (w / ratio).max(0.0);
        } else {
            h = min_hc;
            w = (h * ratio).max(0.0);
        }
        return (w, h);
    }
    // Mixed: both win independently.
    if (w_over && h_under) || (w_under && h_over) {
        if w_over {
            w = max_wc;
        } else {
            w = min_wc;
        }
        if h_over {
            h = max_hc;
        } else {
            h = min_hc;
        }
        return (w, h);
    }

    // Single-axis or no violation: sequential clamp with transfers.
    let mut width_at_bound = false;
    if !width_final {
        let clamped = clamp_content_width_own_terms(min_w, max_w, d.box_sizing, extras_w, w);
        width_at_bound = (min_w > 0.0 && (w - min_wc).abs() < 0.01)
            || (max_w < f32::INFINITY && (w - max_wc).abs() < 0.01);
        w = clamped;
    }
    if height_auto {
        let intrinsic = forest.nodes[idx].intrinsic;
        let intrinsic_w = intrinsic.and_then(|d| d.w).map(|w| w.max(0.0));
        let intrinsic_h_known = intrinsic.is_some_and(|d| d.h.is_some());
        if Some(w) != intrinsic_w || !intrinsic_h_known {
            h = (w / ratio).max(0.0);
        }
    }
    let clamped_h = clamp_height(forest, idx, cb_h, h);
    if clamped_h != h {
        h = clamped_h;
        if width_auto && !width_final && !width_at_bound {
            w = (h * ratio).max(0.0);
        }
    }
    (w, h)
}

/// Reverse `aspect-ratio` transfer for intrinsic sizing (css-sizing-4
/// §3.1): a non-replaced box with a preferred `aspect-ratio`, an `auto`
/// (or cyclic) width, and a definite height contributes the
/// ratio-derived content width (`width = height * ratio`) to its
/// intrinsic sizes — mirroring the shrink-to-fit reverse transfer in
/// the block width path. Returns `None` for replaced boxes (their
/// intrinsic size rules instead), when the height is indefinite
/// (percentages and percentage-bearing `calc()` are cyclic against
/// the indefinite intrinsic base and behave as `auto`, exactly like
/// [`specified_height`] with no `cb_h`), and when `height` itself is
/// `auto`. The definite height is clamped by `min-height` /
/// `max-height` in the box-sizing's own terms before the transfer,
/// and the resulting width by `min-width` / `max-width`, so the
/// contribution matches the used size the engine would derive.
///
/// Callers chain this after [`definite_specified_content_width`] and
/// [`intrinsic_content_width`] (both `None` here: the width is `auto`
/// and the box is non-replaced), before the content-based fallback.
fn aspect_ratio_intrinsic_width(forest: &Forest, i: usize) -> Option<f32> {
    let ar = forest.nodes[i].data.aspect_ratio?;
    if forest.nodes[i].intrinsic.is_some() {
        return None;
    }
    let d = &forest.nodes[i].data;
    let h = match d.height {
        SpecLen::Auto | SpecLen::Pct(_) => return None,
        SpecLen::Math { had_pct, .. } if had_pct => return None,
        spec => spec
            .resolve(
                d.font_size,
                forest.root_font_size,
                forest.vp_w,
                forest.vp_h,
                0.0,
            )?
            .max(0.0),
    };
    // Clamp the border-box height by min/max-height (percentage
    // clamps are cyclic here and drop out), then transfer through
    // the ratio in box-sizing-aware terms.
    let min_h = minmax_constraint(forest, i, d.min_height, 0.0).max(0.0);
    let max_h = minmax_constraint(forest, i, d.max_height, f32::INFINITY).max(0.0);
    let content_h = match d.box_sizing {
        BoxSizing::BorderBox => (h.min(max_h).max(min_h) - intrinsic_v_bp(forest, i)).max(0.0),
        BoxSizing::ContentBox => h.min(max_h).max(min_h).max(0.0),
    };
    let w = aspect_ratio_content_width(
        ar.ratio,
        content_h,
        d.box_sizing,
        own_horizontal_bp(forest, i),
        intrinsic_v_bp(forest, i),
    );
    Some(clamp_content_width_by_minmax(forest, i, w))
}

/// Apply `aspect-ratio` forward transfer (css-sizing-4 §3.1) to a box's
/// automatic height: when the used height is otherwise auto, derive the
/// content height from the used content width through the preferred
/// ratio. Returns the new `hspec`.
///
/// A specified `<ratio>` wins over the box's intrinsic ratio: the
/// intrinsic-height fallback `specified_height` gives replaced elements
/// for `height: auto` still counts as auto, so only a genuinely
/// specified height — or a caller height override — blocks the transfer.
/// `auto <ratio>` instead prefers the intrinsic ratio, keeping intrinsic
/// sizing. Flex/table/grid containers size themselves via their own
/// algorithms (no transfer here — items laid out through this path still
/// transfer, since their width is caller-determined).
fn aspect_ratio_hspec(
    forest: &Forest,
    idx: usize,
    hspec: Option<f32>,
    content_w: f32,
    height_override: bool,
) -> Option<f32> {
    let ar = match forest.nodes[idx].data.aspect_ratio {
        Some(ar) => ar,
        None => return hspec,
    };
    let n = &forest.nodes[idx];
    // `auto <ratio>` prefers the box's intrinsic ratio (css-sizing-4
    // §3.1): replaced elements keep their intrinsic sizing, which
    // `hspec` already carries.
    if ar.auto_preferred && n.intrinsic.is_some() {
        return hspec;
    }
    if hspec.is_some() {
        // The intrinsic-height fallback still counts as auto: a
        // specified `<ratio>` overrides it. Any other definite height —
        // specified, abspos-solved, or a caller override — wins.
        let intrinsic_fallback =
            !height_override && n.intrinsic.is_some() && matches!(n.data.height, SpecLen::Auto);
        if !intrinsic_fallback {
            return hspec;
        }
    }
    if flex_kind(forest, idx).is_some()
        || matches!(
            forest.nodes[idx].data.display,
            Display::Table | Display::InlineTable | Display::Grid | Display::InlineGrid
        )
    {
        return hspec;
    }
    let d = &forest.nodes[idx].data;
    let m = &forest.nodes[idx].metrics;
    Some(aspect_ratio_content_height(
        ar.ratio,
        content_w,
        d.box_sizing,
        m.bl + m.pl + m.pr + m.br,
        m.bt + m.pt + m.pb + m.bb,
    ))
}

// ---------------------------------------------------------------------------
// Margin collapsing
// ---------------------------------------------------------------------------

/// Whether the block can collapse its own top/bottom margins through
/// itself: no children, no vertical border/padding, and auto height. A
/// percentage height against an indefinite containing-block height counts
/// as auto, matching `specified_height` (and the empty-block branch in
/// `layout_block`, which keys off it).
fn is_empty_collapsible(forest: &Forest, idx: usize, cb_h: Option<f32>) -> bool {
    let n = &forest.nodes[idx];
    // "Empty" for margin collapsing means no in-flow children: out-of-flow
    // children take no space and their margins never collapse (CSS 2.1
    // §8.3.1), so they do not make the box non-empty.
    let no_inflow_children = !n
        .children
        .iter()
        .any(|&c| !layout_position::is_out_of_flow(forest, c) && !is_float(forest, c));
    no_inflow_children
        && n.metrics.bt + n.metrics.pt + n.metrics.pb + n.metrics.bb == 0.0
        && specified_height(forest, idx, cb_h).is_none()
        && min_height_is_zero(forest, idx, cb_h)
}

/// Whether `idx`'s specified `min-height` resolves to zero (the initial
/// value). An empty box with a positive `min-height` is not
/// collapse-transparent: its own margins cannot collapse through it
/// (CSS 2.1 §8.3.1). Percentages resolve against a zero base when the
/// containing-block height is indefinite, which §10.7 also treats as
/// zero; `auto` resolves to `None` and counts as zero too.
fn min_height_is_zero(forest: &Forest, idx: usize, cb_h: Option<f32>) -> bool {
    let d = &forest.nodes[idx].data;
    d.min_height
        .resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            cb_h.unwrap_or(0.0),
        )
        .is_none_or(|v| v <= 0.0)
}

/// Whether `idx` is a block-level box with no content whose own margins
/// collapse through it. Such a box is transparent to margin collapsing:
/// the margins adjoining it above and below form a single set that
/// collapses once, instead of being placed both above and below it.
fn is_collapse_transparent(forest: &Forest, idx: usize, cb_h: Option<f32>) -> bool {
    is_block_level(forest.nodes[idx].box_type) && is_empty_collapsible(forest, idx, cb_h)
}

fn no_top_bp(forest: &Forest, idx: usize) -> bool {
    let m = &forest.nodes[idx].metrics;
    m.bt + m.pt == 0.0
}

/// Whether `idx` establishes a block formatting context for its contents:
/// `display: flow-root`, `overflow`/`overflow-x`/`overflow-y` other than
/// `visible`, a float, or a flex/grid container. Margins never collapse
/// through a BFC boundary (CSS 2.1 §8.3.1): a BFC root's top/bottom
/// margins do not collapse with its in-flow children's, and flex/grid
/// item margins never collapse with the container's.
fn establishes_bfc(forest: &Forest, idx: usize) -> bool {
    // Floats establish a new block formatting context (CSS 2.1 §9.4.1):
    // their floats are contained and their margins never collapse with
    // the parent's.
    if is_float(forest, idx) {
        return true;
    }
    // Flex containers: children are flex items, blockified, whose margins
    // never collapse with the container's.
    if flex_kind(forest, idx).is_some() {
        return true;
    }
    let style = forest.nodes[idx].style.as_ref();
    if matches!(
        style.and_then(|s| s.get_str_by_index(props::DISPLAY)),
        Some(d) if d.eq_ignore_ascii_case("flow-root")
            || d.eq_ignore_ascii_case("grid")
            || d.eq_ignore_ascii_case("inline-grid")
    ) {
        return true;
    }
    // Table wrappers and table cells establish a BFC (CSS 2.1 §9.4.1):
    // cell content margins never collapse through the cell, and table
    // margins never collapse with row/cell margins. Rows and row groups
    // do not establish one.
    if matches!(
        forest.nodes[idx].box_type,
        BoxType::Table | BoxType::TableCell
    ) {
        return true;
    }
    // Any non-visible overflow axis establishes a BFC (the cascade keeps
    // `overflow-x`/`overflow-y` as independent properties).
    [props::OVERFLOW, props::OVERFLOW_X, props::OVERFLOW_Y]
        .into_iter()
        .any(|prop| {
            matches!(
                style.and_then(|s| s.get_str_by_index(prop)),
                Some(v) if !v.eq_ignore_ascii_case("visible")
            )
        })
}

/// Compute the margin to place above `idx`'s border box: the node's own top
/// margin collapsed through any chain of first in-flow block children
/// (which requires no top border/padding at each level). `cb_h` is the
/// containing-block height `idx` lays out against; it is threaded down the
/// chain so percentage heights resolve the same way they will at layout
/// time. Memoized per node so deep chains stay O(n). Assumes `idx`'s own
/// metrics are resolved; the walk resolves descendants with a dummy `cb_x`
/// (only margins matter, and the real resolution overwrites `rect.x` before
/// it is ever read).
/// The first in-flow block-level child: the margin-collapse chain link.
/// Out-of-flow children are skipped (their margins never collapse,
/// CSS 2.1 §8.3.1); floats are out of flow too (§9.4).
fn first_inflow_block_child(forest: &Forest, idx: usize) -> Option<usize> {
    forest.nodes[idx].children.iter().copied().find(|&c| {
        is_block_level(forest.nodes[c].box_type)
            && !layout_position::is_out_of_flow(forest, c)
            && !is_float(forest, c)
    })
}

fn compute_mt_place(forest: &mut Forest, idx: usize, cb_h: Option<f32>) {
    if forest.nodes[idx].mt_place_done {
        return;
    }
    // Phase 1: walk down the first-child chain, collecting it, until a node
    // whose placement margin is known or the chain cannot continue. Each
    // entry carries its own containing-block height.
    let mut chain: Vec<(usize, Option<f32>)> = vec![(idx, cb_h)];
    loop {
        let (cur, cur_cb_h) = *chain.last().expect("chain is never empty");
        if forest.nodes[cur].mt_place_done {
            break;
        }
        if is_empty_collapsible(forest, cur, cur_cb_h) {
            let (mt, mb) = {
                let m = &forest.nodes[cur].metrics;
                (m.mt, m.mb)
            };
            let n = &mut forest.nodes[cur];
            n.mt_place = collapse(mt, mb);
            n.top_collapsed = false;
            n.mt_place_done = true;
            break;
        }
        // The top margin collapses with the first in-flow block-level
        // child's (see `first_inflow_block_child`).
        let first = first_inflow_block_child(forest, cur);
        match first {
            // A BFC root's top margin never collapses with its children's
            // (CSS 2.1 §8.3.1); the walk stops at the boundary.
            Some(fc) if no_top_bp(forest, cur) && !establishes_bfc(forest, cur) => {
                let cw = forest.nodes[cur].metrics.content_w;
                resolve_metrics(forest, fc, 0.0, cw, true, None, false);
                // The first child's percentage heights resolve against
                // cur's content-box height, which is definite only when
                // specified.
                let fc_cb_h = specified_height(forest, cur, cur_cb_h);
                chain.push((fc, fc_cb_h));
            }
            _ => {
                let mt = forest.nodes[cur].metrics.mt;
                let n = &mut forest.nodes[cur];
                n.mt_place = mt;
                n.top_collapsed = false;
                n.mt_place_done = true;
                break;
            }
        }
    }
    // Phase 2: unwind, collapsing each level's own margin with the next.
    for w in chain.windows(2).rev() {
        let (a, _) = w[0];
        let (b, _) = w[1];
        if forest.nodes[a].mt_place_done {
            continue;
        }
        let mt = forest.nodes[a].metrics.mt;
        let b_place = forest.nodes[b].mt_place;
        let n = &mut forest.nodes[a];
        n.mt_place = collapse(mt, b_place);
        n.top_collapsed = true;
        n.mt_place_done = true;
    }
}

// ---------------------------------------------------------------------------
// Block layout
// ---------------------------------------------------------------------------

/// When the specified height is a minimum (see
/// [`BlockInput::height_is_min`]), the used height floors at it: the
/// effective specified height is `max(specified, natural)`, so content
/// taller than the specified height grows the box instead of the box
/// clamping to the specified height. `None` (auto height) is unaffected.
fn min_floor_hspec(height_is_min: bool, hspec: Option<f32>, natural_h: f32) -> Option<f32> {
    match (height_is_min, hspec) {
        (true, Some(h)) => Some(h.max(natural_h)),
        _ => hspec,
    }
}

/// Set content height / border-box height / effective bottom margin for a
/// block whose content height is `natural_h` (lines, or empty). Inline
/// content never collapses the bottom margin through. The used height is
/// clamped by min-height/max-height.
fn finish_content_height(
    forest: &mut Forest,
    idx: usize,
    cb_h: Option<f32>,
    hspec: Option<f32>,
    natural_h: f32,
) {
    let content_h = clamp_height(forest, idx, cb_h, hspec.unwrap_or(natural_h));
    // The height axis is the block axis: for a vertical table the
    // border-box extras on this axis are the horizontal border/padding.
    let vwm_table = is_vertical_table(forest, idx);
    let (e1, e2, e3, e4, mb) = {
        let m = &forest.nodes[idx].metrics;
        if vwm_table {
            // Block-end margin: physical right for vertical-lr
            // (block left-to-right), physical left for vertical-rl
            // (block right-to-left).
            let be = if matches!(forest.nodes[idx].data.writing_mode, WritingMode::VerticalRl) {
                m.ml
            } else {
                m.mr
            };
            (m.bl, m.pl, m.pr, m.br, be)
        } else {
            (m.bt, m.pt, m.pb, m.bb, m.mb)
        }
    };
    let n = &mut forest.nodes[idx];
    n.content_h = content_h;
    n.natural_content_h = natural_h;
    n.rect.h = e1 + e2 + content_h + e3 + e4;
    n.mb_place = mb;
}

/// Correct a collapsing-border `content-box` table's border-box width:
/// the table has no padding and its border box extends only half the
/// specified border beyond the content on each side (CSS 2.1 §17.6.2),
/// but `resolve_metrics` added the full border and padding. Mirrors the
/// height-axis correction in `layout_table::place_table_rows` (WPT
/// `td-box-sizing-002`: content-box + collapse with a 100px specified
/// width, 10px padding, and 20/40px borders must give 130, not 180).
fn collapse_content_box_table_width(forest: &mut Forest, idx: usize) {
    let n = &forest.nodes[idx];
    if !matches!(n.data.display, Display::Table | Display::InlineTable)
        || n.data.box_sizing != BoxSizing::ContentBox
        || !layout_table::table_border_model(forest, idx).0
    {
        return;
    }
    let vertical = is_vertical_table(forest, idx);
    let n = &mut forest.nodes[idx];
    let content_w = n.metrics.content_w;
    let m = &n.metrics;
    // Inline axis: physical-horizontal for horizontal tables,
    // physical-vertical for vertical tables.
    let (b1, b2) = if vertical { (m.bt, m.bb) } else { (m.bl, m.br) };
    n.rect.w = (b1 / 2.0 + content_w + b2 / 2.0).max(0.0);
}

/// Whether `idx` is a table in a vertical writing mode. Tables consume
/// `width`/`height` logically (css-tables-3 §3, css-writing-modes-3 §7):
/// the table's inline size resolves from the `width` property in
/// horizontal writing modes but from the `height` property in vertical
/// ones, and vice versa for the block size. Only the table's own writing
/// mode matters; the containing block's is irrelevant. The layout
/// algorithm itself runs unchanged in (inline, block) space; the width
/// axis below is the inline axis and the height axis the block axis, and
/// [`transpose_vwm_table`] maps the placed rects back to physical space.
pub(crate) fn is_vertical_table(forest: &Forest, idx: usize) -> bool {
    let d = &forest.nodes[idx].data;
    matches!(d.display, Display::Table | Display::InlineTable)
        && matches!(
            d.writing_mode,
            WritingMode::VerticalLr | WritingMode::VerticalRl
        )
}

/// The spec feeding a table's inline (algorithm-width) axis: the physical
/// `height` property in vertical writing modes, `width` otherwise.
pub(crate) fn table_inline_spec(forest: &Forest, idx: usize) -> SpecLen {
    let d = &forest.nodes[idx].data;
    if is_vertical_table(forest, idx) {
        d.height
    } else {
        d.width
    }
}

/// The spec feeding a table's block (algorithm-height) axis: the physical
/// `width` property in vertical writing modes, `height` otherwise.
fn table_block_spec(forest: &Forest, idx: usize) -> SpecLen {
    let d = &forest.nodes[idx].data;
    if is_vertical_table(forest, idx) {
        d.width
    } else {
        d.height
    }
}

/// Transpose one node's rect and deferred shift between physical and
/// (inline, block) space around the table's border-box origin `(tx, ty)`
/// (physical size `tw` for the vertical-rl mirror); optionally recurse
/// into children. Iterative per the crate's never-recurse invariant.
///
/// `sizes_only` is for cell-child subtrees in the forward direction:
/// the block engine laid them out against the cell's algorithm-space
/// rect, so their positions are already (inline, block) — only the
/// sizes need swapping (the engine's physical w/h become the algorithm's
/// inline/block extents), and the mirror never applies. Captions and
/// floats (`sizes_only = false`) are in true physical space, so their
/// positions transpose as well; the backward direction
/// ([`transpose_vwm_table`]) additionally mirrors for vertical-rl, which
/// is what makes the round trip exact.
#[allow(clippy::too_many_arguments)]
fn transpose_vwm_subtree(
    forest: &mut Forest,
    root: usize,
    tx: f32,
    ty: f32,
    tw: f32,
    mirror: bool,
    include_root: bool,
    sizes_only: bool,
) {
    let mut stack = vec![(root, include_root)];
    while let Some((i, self_)) = stack.pop() {
        // Index iteration instead of `children.clone()`: the body only
        // mutates rect/shift fields, never `nodes[i].children`, so reading
        // each child by index is the same snapshot without the O(kids)
        // allocation.
        let nkids = forest.nodes[i].children.len();
        if self_ {
            let n = &mut forest.nodes[i];
            if sizes_only {
                // Positions already algorithm; swap the physical
                // (w, h) into (inline, block) extents. Deferred shifts
                // are likewise already in (inline, block) terms.
                // The mirror never applies here (see above).
                std::mem::swap(&mut n.rect.w, &mut n.rect.h);
            } else {
                let (dx, dy) = (n.rect.x - tx, n.rect.y - ty);
                let (w, h) = (n.rect.w, n.rect.h);
                let (px, py) = (n.pend_x, n.pend_y);
                if mirror {
                    n.rect.x = tx + (tw - (dy + h));
                    n.rect.y = ty + dx;
                    n.pend_x = -py;
                    n.pend_y = px;
                } else {
                    n.rect.x = tx + dy;
                    n.rect.y = ty + dx;
                    n.pend_x = py;
                    n.pend_y = px;
                }
                n.rect.w = h;
                n.rect.h = w;
            }
        }
        for k in 0..nkids {
            stack.push((forest.nodes[i].children[k], true));
        }
    }
}

/// Move a vertical table's laid-out cell/caption/float subtrees into
/// (inline, block) space at the start of `resume_table`, before the
/// row-height machinery measures them.
///
/// Cell tasks lay out through the horizontal-only block engine with the
/// column (inline) width as the width override, so each cell's own rect
/// is mixed-space: `rect.x/y` and `rect.w` are already the algorithm's
/// (inline) values, but `rect.h` is vertical content, not the block
/// size. The children's positions are likewise already algorithm-space
/// (the engine positioned them against the cell's algorithm rect); only
/// their sizes are physical and need swapping. This swaps each child's
/// subtree sizes (not the cell's own rect) and stamps the precomputed
/// block size (`TableResume::cell_block`) into the cell's `rect.h`;
/// captions and floats transpose whole (their physical rects are
/// already correct). `place_table_rows`' end transpose then inverts all
/// of it exactly; the vertical-rl mirror lives only in that backward
/// direction.
pub(crate) fn vwm_cells_to_logical(forest: &mut Forest, fr: &layout_table::TableResume) {
    let idx = fr.idx;
    let (tx, ty) = {
        let n = &forest.nodes[idx];
        (n.rect.x, n.rect.y)
    };
    // §3.10.6 second-pass resume: only the re-laid (pass2) cells are back
    // in physical space; captions, floats, and non-pass2 cells are
    // already in (inline, block) space from the measurement resume and
    // must not be transposed twice.
    if fr.phase2 {
        for &ci in &fr.phase2_cells {
            if let (Some(cell), Some(&bh)) = (fr.cells.get(ci), fr.cell_block.get(ci)) {
                transpose_vwm_subtree(forest, cell.node, tx, ty, 0.0, false, false, true);
                forest.nodes[cell.node].rect.h = bh;
            }
        }
        return;
    }
    for (cell, &bh) in fr.cells.iter().zip(fr.cell_block.iter()) {
        // Cells that never laid out (collapsed rows/columns) have no
        // children and a zero rect: the size swap is a no-op and their
        // `cell_block` is 0.0, so the stamp changes nothing.
        // sizes_only: child positions are already algorithm-space; the
        // vertical-rl mirror lives only in the backward direction.
        transpose_vwm_subtree(forest, cell.node, tx, ty, 0.0, false, false, true);
        // The cell's own rect: x/y/w are already algorithm-space (the
        // width override put the column width in rect.w); stamp the
        // block size into rect.h.
        forest.nodes[cell.node].rect.h = bh;
    }
    for &c in fr.captions_top.iter().chain(fr.captions_bottom.iter()) {
        transpose_vwm_subtree(forest, c, tx, ty, 0.0, false, true, false);
    }
    for &f in fr.floats.iter() {
        transpose_vwm_subtree(forest, f, tx, ty, 0.0, false, true, false);
    }
}

/// Transpose a vertical table's placed rects from (inline, block) space
/// back to physical space. Runs once, at the end of
/// `layout_table::place_table_rows`, when every table-internal rect and
/// deferred shift is final in algorithm space (cell/caption/float
/// subtrees were moved there by [`vwm_cells_to_logical`]).
///
/// Mapping: physical = (block, inline) around the table's border-box
/// origin. In `vertical-lr` that is a pure axis swap (inline-start is the
/// top, block-start the left, so both spaces share the top-left corner);
/// in `vertical-rl` the block axis additionally mirrors (block-start is
/// the right edge).
///
/// Only table-internal boxes transpose (the table's own size, captions,
/// row groups, rows, cells, column boxes, floats): their geometry came
/// out of the table algorithm, after [`vwm_cells_to_logical`] moved the
/// cell/caption/float subtrees into (inline, block) space up front. Cell
/// *contents* were laid out by the horizontal-only block engine, so
/// their intra-cell arrangement stays horizontal — real vertical cell
/// content is follow-up work. Vertical intrinsic measurement covers
/// block-level children with definite heights (see
/// [`vertical_cell_min_content`] / [`vertical_cell_max_content`], used by
/// the column machinery when `TableGrid::vertical` is set); vertical text
/// line stacking, inline children, and auto-height content still measure
/// horizontally — follow-up. Per-cell padding/border in column widths is
/// transposed via `TableGrid::vertical` — `cell_h_pad`.
/// Deferred shifts transpose with their nodes so the final
/// `apply_pending_shifts` pass stays consistent. The table's own position is parent-space
/// physical and untouched, as is its own pending shift (recorded later,
/// in physical space, by the abspos / relative passes).
///
/// Invertibility limits: asymmetric table border/padding (metrics stay
/// physical while the border box assembles axis-appropriately),
/// abspos/fixed descendants of the table (their static positions were
/// recorded in algorithm space), and the inline-axis available space
/// (still the physical containing-block width, not its height).
pub(crate) fn transpose_vwm_table(forest: &mut Forest, fr: &layout_table::TableResume) {
    let idx = fr.idx;
    let mirror = matches!(forest.nodes[idx].data.writing_mode, WritingMode::VerticalRl);
    // The table's own size transposes; its position is parent-space
    // physical and stays. The content extents swap with the size so both
    // read physical below.
    let (tx, ty, tw) = {
        let n = &mut forest.nodes[idx];
        std::mem::swap(&mut n.rect.w, &mut n.rect.h);
        std::mem::swap(&mut n.content_h, &mut n.metrics.content_w);
        (n.rect.x, n.rect.y, n.rect.w)
    };
    // Cell/caption/float subtrees were moved to (inline, block) space by
    // `vwm_cells_to_logical`; transpose them back whole. Rows, groups,
    // and column boxes only ever lived in algorithm space: transpose
    // their own rects only (their children are the cells, transposed
    // separately above — no recursion here).
    fn transpose_root_only(forest: &mut Forest, i: usize, tx: f32, ty: f32, tw: f32, mirror: bool) {
        let n = &mut forest.nodes[i];
        let (dx, dy) = (n.rect.x - tx, n.rect.y - ty);
        let (w, h) = (n.rect.w, n.rect.h);
        let (px, py) = (n.pend_x, n.pend_y);
        if mirror {
            n.rect.x = tx + (tw - (dy + h));
            n.rect.y = ty + dx;
            n.pend_x = -py;
            n.pend_y = px;
        } else {
            n.rect.x = tx + dy;
            n.rect.y = ty + dx;
            n.pend_x = py;
            n.pend_y = px;
        }
        n.rect.w = h;
        n.rect.h = w;
    }
    for &c in fr.captions_top.iter().chain(fr.captions_bottom.iter()) {
        transpose_vwm_subtree(forest, c, tx, ty, tw, mirror, true, false);
    }
    for cell in fr.cells.iter() {
        transpose_vwm_subtree(forest, cell.node, tx, ty, tw, mirror, true, false);
    }
    for rn in fr.row_nodes.iter().flatten() {
        transpose_root_only(forest, *rn, tx, ty, tw, mirror);
    }
    for &g in fr.group_nodes.iter() {
        transpose_root_only(forest, g, tx, ty, tw, mirror);
    }
    for spec in fr.columns.iter() {
        for node in spec.col_node.into_iter().chain(spec.group_node) {
            transpose_root_only(forest, node, tx, ty, tw, mirror);
        }
    }
    for &f in fr.floats.iter() {
        transpose_vwm_subtree(forest, f, tx, ty, tw, mirror, true, false);
    }
}

/// Solve the CSS 2.1 §10.3.7 / §10.6.4 constraint equations for an
/// absolutely positioned box with `auto` width/height: when both offsets
/// on an axis are specified (non-`auto`), the `auto` size is the
/// remaining space (margins, borders, padding subtracted). Returns
/// `(width, height)`, with `None` per axis when the equation does not
/// apply: an offset is `auto` (shrink-to-fit / static position then
/// apply), the size is not `auto`, or the box is replaced.
///
/// The vertical equation additionally requires a *definite*
/// containing-block height: CSS 2.1 §10.6.4 solves the constraint only
/// "if the containing block's height does not depend on this element's
/// height". With an indefinite height and `height: auto`, browsers do not
/// stretch against the used height; the box falls back to its content
/// height (a percentage inset already computed to `auto` above, so that
/// case is `None` already — the length-inset case is what `None` `cb_h`
/// must also reject).
///
/// Call only after `resolve_metrics` populated `metrics` (the equation
/// needs the used border/padding). `auto` margins compute to zero on
/// out-of-flow boxes; percentage margins resolve against the
/// containing-block width.
fn abspos_solved_sizes(
    forest: &Forest,
    idx: usize,
    cb_w: f32,
    cb_h: Option<f32>,
) -> (Option<f32>, Option<f32>) {
    let n = &forest.nodes[idx];
    if n.intrinsic.is_some() {
        return (None, None);
    }
    let d = &n.data;
    let (fs, root_fs, vp_w, vp_h) = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
    let mu = |side: SpecMargin| match side {
        SpecMargin::Auto => 0.0,
        SpecMargin::Len(s) => s.resolve(fs, root_fs, vp_w, vp_h, cb_w).unwrap_or(0.0),
    };
    let o = |s: SpecLen, base: f32| s.resolve(fs, root_fs, vp_w, vp_h, base);
    // Mirrors `layout_position`'s `ov`: a percentage inset against an
    // indefinite containing-block height computes to `auto` (CSS 2.1
    // §10.6.4) instead of resolving against the used height, so sizing
    // and positioning agree. Lengths always resolve.
    let ov = |s: SpecLen| {
        if cb_h.is_none() && layout_position::has_pct(s) {
            None
        } else {
            o(s, cb_h.unwrap_or(0.0))
        }
    };
    let m = &n.metrics;
    // css-position-3 §4.2: when both insets on an axis are non-`auto`
    // and their sum exceeds the containing-block size, the weaker
    // (end) inset is adjusted so the inset-modified containing block
    // is zero-sized instead of negative — mirroring `abs_position`,
    // so sizing and positioning solve against the same insets.
    // Horizontally the end side is `right` in LTR, `left` in RTL;
    // vertically it is always `bottom`, and the vertical clamp needs
    // a definite containing-block height to mean anything.
    let (left, right) = match (o(d.offset[3], cb_w), o(d.offset[1], cb_w)) {
        (Some(l), Some(r)) if l + r > cb_w => {
            if d.rtl {
                (Some(cb_w - r), Some(r))
            } else {
                (Some(l), Some(cb_w - l))
            }
        }
        x => x,
    };
    let (top, bottom) = match (ov(d.offset[0]), ov(d.offset[2]), cb_h) {
        (Some(t), Some(b), Some(h)) if t + b > h => (Some(t), Some(h - t)),
        (t, b, _) => (t, b),
    };
    // Tables in vertical writing modes consume width/height logically, but
    // the constraint equations below stay physical: the width equation
    // (left/right insets) solves the physical width — the block size for
    // a vertical table — and the height equation (top/bottom insets) the
    // physical height (the inline size). The caller consumes the results
    // as (inline, block), so swap the tuple for vertical tables.
    let width = match (d.width, left, right) {
        (SpecLen::Auto, Some(l), Some(r)) => Some(
            (cb_w - l - r - mu(d.margin[3]) - mu(d.margin[1]) - (m.bl + m.pl + m.pr + m.br))
                .max(0.0),
        ),
        _ => None,
    };
    let height = match (d.height, top, bottom) {
        // The §10.6.4 rule-5 stretch needs a definite containing-block
        // height. With an indefinite height (`cb_h: None`) the equation
        // would solve against 0 and zero the box; browsers instead treat
        // it as `height: auto` (content height below).
        (SpecLen::Auto, Some(t), Some(b)) => cb_h.map(|h| {
            (h - t - b - mu(d.margin[0]) - mu(d.margin[2]) - (m.bt + m.pt + m.pb + m.bb)).max(0.0)
        }),
        _ => None,
    };
    if is_vertical_table(forest, idx) {
        (height, width)
    } else {
        (width, height)
    }
}

fn layout_block(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    input: &BlockInput,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    scratch: &mut LayoutScratch,
) {
    let idx = input.idx;
    let block_level = is_block_level(forest.nodes[idx].box_type);
    // Tables consume width/height logically: in a vertical writing mode
    // the width axis below is the inline axis, resolving from the
    // physical `height` property (and vice versa for heights).
    let width_spec = table_inline_spec(forest, idx);
    // Out-of-flow roots (`position: absolute | fixed`) take no part in
    // margin centering: auto margins compute to zero (CSS 2.1 §10.3.7).
    // Floats likewise: auto margins compute to zero (CSS 2.1 §10.3.5).
    let flow_margins = block_level && !input.abspos && !input.is_float;

    // 1. Horizontal metrics.
    // Replaced elements are atomic with an intrinsic size: no
    // shrink-to-fit; the intrinsic fallbacks in `resolve_metrics` /
    // `specified_height` apply when CSS width/height are auto.
    if let Some(w) = input.width_override {
        // Caller-determined content width (flex items): auto margins
        // compute to zero instead of centering.
        resolve_metrics(
            forest,
            idx,
            input.cb_x,
            input.cb_w,
            false,
            Some(w),
            input.width_override_final,
        );
        // Collapsed table borders: the winning widths (centered on the
        // shared grid lines, CSS 2.1 §17.6.2.2) replace the cell's own
        // specified border widths, so the content box, border box, and
        // row-height accounting all use the collapsed geometry instead
        // of the cell's (possibly losing) specified widths. Re-stamp
        // `rect.w`, which `resolve_metrics` computed with the specified
        // widths. Only table cells set this (see `layout_table`).
        if let Some(wins) = input.collapsed_border_wins {
            let n = &mut forest.nodes[idx];
            n.metrics.bl = wins[3] / 2.0;
            n.metrics.br = wins[1] / 2.0;
            n.metrics.bt = wins[0] / 2.0;
            n.metrics.bb = wins[2] / 2.0;
            let (bl, pl, cw, pr, br) = {
                let m = &n.metrics;
                (m.bl, m.pl, m.content_w, m.pr, m.br)
            };
            n.rect.w = bl + pl + cw + pr + br;
        }
    } else if (input.shrink || input.abspos || input.is_float)
        && forest.nodes[idx].intrinsic.is_none()
        && matches!(
            width_spec,
            SpecLen::Auto | SpecLen::FitContent | SpecLen::MinContent | SpecLen::MaxContent
        )
    {
        // Shrink-to-fit (lite): min(preferred, available), then strip the
        // box's own padding/border to get the content width. Floats use
        // shrink-to-fit per CSS 2.1 §10.3.5; absolutely positioned boxes
        // per §10.3.7 — but only when an offset is `auto`: with `left`
        // and `right` both specified, `width: auto` solves to the
        // remaining space instead.
        let pref = preferred_width(forest, idx, fonts, pw_cache);
        // CSS 2.1 §17.5.2.2 CAPMIN for shrink-to-fit tables (floated,
        // out-of-flow, `inline-table`) already lives inside `pref`:
        // `table_pref_width`'s inner gate floors the preferred width by
        // the caption's *max-content* outer width, and max-content is
        // always at least the caption's min-content (CAPMIN). A further
        // `.max(table_caption_min_width(...))` here would be dead — and
        // it mixes the content-box `pref` with the outer CAPMIN.
        resolve_metrics(forest, idx, 0.0, input.cb_w, false, None, false);
        let extras = {
            let m = &forest.nodes[idx].metrics;
            m.bl + m.pl + m.pr + m.br
        };
        let (solved_w, solved_h) = if input.abspos {
            abspos_solved_sizes(forest, idx, input.cb_w, input.cb_h)
        } else {
            (None, None)
        };
        // Preferred aspect ratio, reverse transfer (css-sizing-4 §3.1):
        // with a definite height and an auto width, the content width
        // derives from the height through the ratio instead of
        // shrink-to-fit. Replaced elements keep intrinsic sizing (this
        // branch already excludes them).
        let content_w = solved_w.unwrap_or_else(|| {
            let d = &forest.nodes[idx].data;
            let h_def = input
                .height_override
                .or(solved_h)
                .or_else(|| specified_height(forest, idx, input.cb_h))
                // Mirror the used-height resolution in the `hspec`
                // computation below: an abspos percentage height against
                // an indefinite CB height is definite for sizing on both
                // axes.
                .or_else(|| {
                    input
                        .abs_pct_height_base
                        .and_then(|base| specified_height(forest, idx, Some(base)))
                });
            match (d.aspect_ratio, h_def) {
                (Some(ar), Some(h)) => {
                    let m = &forest.nodes[idx].metrics;
                    aspect_ratio_content_width(
                        ar.ratio,
                        h,
                        d.box_sizing,
                        m.bl + m.pl + m.pr + m.br,
                        m.bt + m.pt + m.pb + m.bb,
                    )
                }
                _ => {
                    // `width: fit-content | min-content | max-content` on
                    // a shrink-to-fit box (float, abspos, inline-block,
                    // grid/flex item): the css-sizing-3 content-based
                    // formula instead of the plain shrink-to-fit.
                    // (`width_spec` is the inline-axis spec: the physical
                    // `height` for a vertical table.)
                    let spec = width_spec;
                    match spec {
                        SpecLen::MinContent => (min_content_width_pooled(
                            forest,
                            idx,
                            fonts,
                            &mut scratch.mcw,
                            &mut scratch.int_bufs,
                        ) - extras)
                            .max(0.0),
                        SpecLen::MaxContent => {
                            // An infinite max-content (a fixed-layout table
                            // with a percentage width: infinite intrinsic
                            // size per Blink) saturates to a huge finite
                            // used width (see `MAX_USED_SIZE`).
                            saturate_used_size((pref - extras).max(0.0))
                        }
                        SpecLen::FitContent => {
                            let mc = min_content_width_pooled(
                                forest,
                                idx,
                                fonts,
                                &mut scratch.mcw,
                                &mut scratch.int_bufs,
                            );
                            let used_bb = pref.min(mc.max(input.cb_w));
                            (used_bb - extras).max(0.0)
                        }
                        _ => (pref.min(input.cb_w) - extras).max(0.0),
                    }
                }
            }
        });
        resolve_metrics(
            forest,
            idx,
            input.cb_x,
            input.cb_w,
            false,
            Some(content_w),
            false,
        );
    } else if !input.shrink
        && !input.abspos
        && !input.is_float
        && block_level
        && forest.nodes[idx].intrinsic.is_none()
        && matches!(forest.nodes[idx].data.display, Display::Table)
        && matches!(
            width_spec,
            SpecLen::Auto
                | SpecLen::FitContent
                | SpecLen::MinContent
                | SpecLen::MaxContent
                | SpecLen::Stretch
        )
    {
        // CSS 2.1 §17.5.2.2: a block-level table with `width: auto`
        // shrink-to-fits instead of filling its containing block. The
        // used width is max(MAX, CAPMIN) when that is narrower than the
        // containing block, floored by max(MIN, CAPMIN); the column
        // distribution in `layout_table` grows the table past this floor
        // to the true column minima when over-constrained. Unlike the
        // shrink branch above, auto side margins still center the table
        // (CSS 2.1 §10.3.3), so the width resolves as an override with
        // block-level margins.
        let capmin = layout_table::table_caption_min_width(
            forest,
            idx,
            fonts,
            &mut scratch.mcw,
            &mut scratch.table_grid,
        );
        let min_w =
            min_content_width_pooled(forest, idx, fonts, &mut scratch.mcw, &mut scratch.int_bufs);
        // CSS 2.1 §17.5.2.2: the table MIN includes the inter-column
        // spacing (`min_content_width` excludes it by design; zero under
        // `border-collapse: collapse`).
        let (_, spacing_h, _) = layout_table::table_border_model(forest, idx);
        let mut grid = scratch.table_grid.take_grid();
        layout_table::collect_table_grid_into(forest, idx, &mut scratch.table_grid, &mut grid);
        let ncols = grid.collapsed_cols.iter().filter(|c| !**c).count();
        let spacing_total = if ncols == 0 {
            0.0
        } else {
            spacing_h * (ncols as f32 + 1.0)
        };
        scratch.table_grid.put_grid(grid);
        // Resolve padding/border/margins once to learn the specified
        // side margins (`width: auto` zeroes auto margins here).
        resolve_metrics(forest, idx, 0.0, input.cb_w, false, None, false);
        let (ml, mr, extras) = {
            let m = &forest.nodes[idx].metrics;
            // `capmin`/`min_w`/`pref` are all inline-axis measures, so
            // the extras stripped for `capmin_c` must be the inline-axis
            // border+padding: physical top/bottom in a vertical writing
            // mode, physical left/right otherwise. (The surrounding
            // `ml`/`mr`/`avail` physicality predates wave 65 — left as
            // is.)
            let inline_bp = if is_vertical_table(forest, idx) {
                m.bt + m.pt + m.pb + m.bb
            } else {
                m.bl + m.pl + m.pr + m.br
            };
            (m.ml, m.mr, inline_bp)
        };
        let box_sizing = forest.nodes[idx].data.box_sizing;
        // The block-level table's preferred width comes from its columns
        // only: the caption's preferred (max-content) width never floors
        // it (that floor is for shrink-to-fit tables). The caption
        // contributes its *min-content* (CAPMIN) at used-width time,
        // and CAPMIN is an outer (border-box plus margins) width, so it
        // floors the table's border-box width, not its content box:
        // comparing against the content box overgrows a bordered table
        // (WPT `caption.html` table 6: 50px borders + a 50px-min caption
        // must stay 100px wide, not 150px).
        let pref = layout_table::table_pref_width(
            forest,
            idx,
            fonts,
            pw_cache,
            &mut scratch.table_grid,
            false,
        );
        let capmin_c = (capmin - extras).max(0.0);
        let floor_c = (min_w + spacing_total).max(capmin_c);
        let pref_c = pref.max(capmin_c);
        let avail = (input.cb_w - ml - mr).max(0.0);
        // Shrink-to-fit resolves in the box-sizing's own terms: for
        // `border-box` the used border-box width is
        // max(MIN_c + E, min(MAX_c + E, avail)); strip the table's own
        // padding/border (like the shrink branch above) to hand
        // `resolve_metrics` the content width its override expects.
        // `fit-content` shrink-fits a table exactly like `auto`
        // (the formula below is already `min(pref, max(min, avail))`
        // plus the CAPMIN floor); `min-content`/`max-content` pin the
        // table to its floor/preferred width. (`width_spec` is the
        // inline-axis spec from the top of `layout_block`.)
        let content_w = match box_sizing {
            BoxSizing::BorderBox => {
                let bb = match width_spec {
                    SpecLen::MinContent => floor_c + extras,
                    SpecLen::MaxContent => pref_c + extras,
                    // Fill semantics: the used border-box width is the
                    // full available width (`stretch`/`fill-available`
                    // ignore intrinsic sizes).
                    SpecLen::Stretch => avail,
                    _ => (floor_c + extras).max((pref_c + extras).min(avail)),
                };
                (bb - extras).max(0.0)
            }
            BoxSizing::ContentBox => match width_spec {
                SpecLen::MinContent => floor_c,
                SpecLen::MaxContent => pref_c,
                SpecLen::Stretch => (avail - extras).max(0.0),
                _ => floor_c.max(pref_c.min(avail)),
            },
        };
        resolve_metrics(
            forest,
            idx,
            input.cb_x,
            input.cb_w,
            flow_margins,
            Some(content_w),
            false,
        );
        // Collapsing-border content-box tables: the border box extends
        // only half the specified border beyond the content (no padding).
        collapse_content_box_table_width(forest, idx);
    } else if !input.shrink
        && !input.abspos
        && !input.is_float
        && block_level
        && forest.nodes[idx].intrinsic.is_none()
        && !matches!(forest.nodes[idx].data.display, Display::Table)
        && matches!(
            forest.nodes[idx].data.width,
            SpecLen::FitContent | SpecLen::MinContent | SpecLen::MaxContent
        )
    {
        // css-sizing-3 §5: `width: fit-content | min-content |
        // max-content` on an in-flow block-level box. Unlike `auto` the
        // used width is definite, so `auto` side margins center the box
        // (CSS 2.1 §10.3.3) — hence `flow_margins`, and the override
        // makes `resolve_metrics` treat the width as non-auto for the
        // margin equation while still applying the min/max clamps.
        resolve_metrics(forest, idx, 0.0, input.cb_w, false, None, false);
        let (ml, mr, extras) = {
            let m = &forest.nodes[idx].metrics;
            (m.ml, m.mr, m.bl + m.pl + m.pr + m.br)
        };
        let spec = forest.nodes[idx].data.width;
        let avail = (input.cb_w - ml - mr - extras).max(0.0);
        let content_w = content_based_width(
            forest,
            idx,
            spec,
            fonts,
            pw_cache,
            &mut scratch.mcw,
            &mut scratch.int_bufs,
            &mut scratch.int_inner,
            avail,
        );
        // css-sizing-3 §4 (F4): `min-width`/`max-width` accept the
        // content-based keywords too, but `SpecLen::resolve` returns
        // `None` for them, which `resolve_metrics`' clamp below silently
        // defaults to 0/∞. Measure them here — where the font context
        // exists — and fold them into the override; the clamp below is
        // then a no-op for these specs (its broken 0/∞ defaults can only
        // widen, never narrow, the pre-clamped width).
        let (min_spec, max_spec) = {
            let d = &forest.nodes[idx].data;
            (d.min_width, d.max_width)
        };
        let lo = measured_minmax_width(
            forest,
            idx,
            min_spec,
            fonts,
            pw_cache,
            &mut scratch.mcw,
            &mut scratch.int_bufs,
            &mut scratch.int_inner,
            avail,
        );
        let hi = measured_minmax_width(
            forest,
            idx,
            max_spec,
            fonts,
            pw_cache,
            &mut scratch.mcw,
            &mut scratch.int_bufs,
            &mut scratch.int_inner,
            avail,
        );
        let content_w = clamp_measured(content_w, lo, hi);
        // An infinite max-content (a fixed-layout table with a
        // percentage width: infinite intrinsic size per Blink) saturates
        // to a huge finite used width (see `MAX_USED_SIZE`); raw
        // infinities must never reach used rects.
        let content_w = saturate_used_size(content_w);
        resolve_metrics(
            forest,
            idx,
            input.cb_x,
            input.cb_w,
            flow_margins,
            Some(content_w),
            false,
        );
    } else {
        resolve_metrics(
            forest,
            idx,
            input.cb_x,
            input.cb_w,
            flow_margins,
            None,
            false,
        );
        // Collapsing-border content-box tables: the border box extends
        // only half the specified border beyond the content (no padding).
        collapse_content_box_table_width(forest, idx);
    }

    // 2. Collapsed top margin. The root has no parent to position it, so it
    //    places itself at the viewport origin plus its margin.
    //    Out-of-flow roots never collapse margins (CSS 2.1 §8.3.1): the
    //    placement margin is just the box's own top margin. Float margins
    //    never collapse either (CSS 2.1 §9.5).
    if input.abspos || input.is_float {
        let mt = forest.nodes[idx].metrics.mt;
        let n = &mut forest.nodes[idx];
        n.mt_place = mt;
        n.mt_place_done = true;
        n.top_collapsed = false;
    } else {
        compute_mt_place(forest, idx, input.cb_h);
    }
    if forest.nodes[idx].parent.is_none() {
        let mt = forest.nodes[idx].mt_place;
        forest.nodes[idx].rect.y = mt;
    }

    // 3. Content-box origin.
    let (content_x, content_top, content_w) = {
        let n = &forest.nodes[idx];
        (
            n.rect.x + n.metrics.bl + n.metrics.pl,
            n.rect.y + n.metrics.bt + n.metrics.pt,
            n.metrics.content_w,
        )
    };
    // v1 multi-column layout: when `column-count` / `column-width`
    // resolve to 2+ columns on a block container, children lay out
    // against one column's width here; the completion paths
    // (`resume_block` for block children, `do_inline_layout` for line
    // boxes) then distribute them across the columns. Stashed on the
    // node so the completion paths can read it back.
    // w79: this gate moves together with `multicol_intrinsic_params`
    // (never one alone): an `InlineBlock` box carrying
    // `display: inline-block` qualifies alongside `Block` /
    // `AnonymousBlock`, so an unpromoted multicol inline-block lays
    // out as multicol exactly as its intrinsic sizes measure it.
    // The existing exclusions are kept verbatim: `flex_kind` still
    // claims flex/inline-flex, and the table/grid displays still
    // claim their own engines below (they share the `InlineBlock`
    // box type, so the display key — not the box type — is what
    // keeps them out).
    let multicol = if (matches!(
        forest.nodes[idx].box_type,
        BoxType::Block | BoxType::AnonymousBlock
    ) || (forest.nodes[idx].box_type == BoxType::InlineBlock
        && forest.nodes[idx].data.display == Display::InlineBlock))
        && flex_kind(forest, idx).is_none()
        && !matches!(
            forest.nodes[idx].data.display,
            Display::Table | Display::Grid | Display::InlineTable | Display::InlineGrid
        ) {
        let n = &forest.nodes[idx];
        layout_multicol::multicol_params(
            n.style.as_deref(),
            n.data.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            content_w,
        )
    } else {
        None
    };
    if let Some(mc) = &multicol {
        forest.nodes[idx].multicol = Some(Box::new(*mc));
    }
    // Width children lay out against: one column for multicol containers,
    // the full content width otherwise.
    let kids_w = multicol.as_ref().map_or(content_w, |m| m.col_w);
    // CSS Overflow 4 `scrollbar-gutter: stable`: reserve space for the
    // scrollbar even when none is shown, so content doesn't shift when
    // a scrollbar appears or disappears. The reservation narrows the
    // width children lay out against (not the box's own width) and
    // shifts the children's content origin past a left (inline-start
    // under `direction: rtl`, or the mirrored `both-edges` strip)
    // reservation.
    //
    // CSS Overflow 3 §2.2: `overflow-y: scroll` additionally forces the
    // vertical classic scrollbar to display whether or not content
    // overflows, and a displayed classic scrollbar consumes its
    // thickness from the content box. That forced strip sits at the
    // inline-end edge — the same edge a shown scrollbar paints — so it
    // never stacks with the gutter reservation above: the per-side max
    // wins, since a shown scrollbar paints exactly the reserved strip.
    // (The propagated-root arm skips both: the viewport owns the
    // root's scrollbar geometry.)
    let (gl, gr) = if forest.gutter_propagated_root == Some(idx) {
        // The root's gutter was already reserved against the viewport
        // (the containing block `layout_forest` sized this box from),
        // so reserving it again here would subtract it twice.
        (0.0, 0.0)
    } else if let Some(style) = forest.nodes[idx].style.as_ref() {
        let (gl, gr) = crate::scrollbar_gutter_reservation(style);
        let (fl, fr) = crate::forced_scrollbar_reservation(style);
        (gl.max(fl), gr.max(fr))
    } else {
        (0.0, 0.0)
    };
    let kids_w = (kids_w - gl - gr).max(0.0);
    // In the propagated-root arm `gl` is 0 (the viewport reservation
    // already shifted this box's own origin), so this never shifts
    // twice.
    let kids_x = content_x + gl;
    let hspec = input.height_override.or_else(|| {
        // CSS 2.1 §10.6.4: an abspos `height: auto` with `top` and
        // `bottom` both specified solves to the remaining space
        // (margins, borders, padding subtracted) — but only against a
        // definite containing-block height; `abspos_solved_sizes` returns
        // `None` for the indefinite case, and an `auto` offset falls back
        // to the content height below.
        if input.abspos {
            let (_, solved_h) = abspos_solved_sizes(forest, idx, input.cb_w, input.cb_h);
            if let Some(h) = solved_h {
                return Some(h);
            }
            // An abspos percentage (or percentage-bearing `calc()`)
            // height resolves against the containing block's *used*
            // height even when that height is indefinite (browser
            // consensus; WPT css-position/position-relative-016 t8/t9).
            // `abs_pct_height_base` is only set for that gated case, so
            // `specified_height` reuses its normal percentage / calc /
            // box-sizing handling here. Percentage *insets* still compute
            // to `auto`: `cb_h` stays `None` for `abspos_solved_sizes`
            // and `abs_position` keeps `cb_h_definite: false`, so the two
            // agree exactly as before.
            if let Some(base) = input.abs_pct_height_base {
                return specified_height(forest, idx, Some(base));
            }
        }
        specified_height(forest, idx, input.cb_h)
    });
    // Preferred aspect ratio, forward transfer (css-sizing-4 §3.1): with
    // a definite used width and an auto height, the content height
    // derives from the used content width through the ratio, in the
    // box-sizing's own terms (see `aspect_ratio_hspec`). The transferred
    // height flows through `clamp_height` like a specified height, so
    // min/max still win per spec.
    let hspec = aspect_ratio_hspec(
        forest,
        idx,
        hspec,
        content_w,
        input.height_override.is_some(),
    );
    // css-tables-3 §3.10.6: percentage heights in a table cell's content
    // are indefinite during the cell's measurement pass — they resolve
    // against the cell's *used* height only in the final pass, when
    // `resume_table` re-lays the cell with a definite `height_override`.
    // The cell's own height keeps the real `hspec` above (a specified
    // cell height is a minimum for its row); only the percentage base
    // threaded to its children becomes indefinite here.
    let cell_measure_pass =
        forest.nodes[idx].box_type == BoxType::TableCell && input.height_override.is_none();
    // w76fix2 L4: a column keyword-measurement task carries a zero
    // minimum-height override (`height_override: Some(0)`,
    // `height_is_min: true`) so the basis keyword, not the specified
    // height, sets the measured height. That zero is a floor, not a
    // definite height: `%`-height descendants must resolve against
    // an indefinite base (behaving as `auto`) during measurement,
    // not against 0. Gated on the exact override pair, so table cells
    // (override `None`) and real definite heights are untouched.
    let measure_zero_min = input.height_is_min && input.height_override == Some(0.0);
    let child_hspec = if cell_measure_pass || measure_zero_min {
        None
    } else {
        hspec
    };
    // Whether the used height is definite (explicitly specified, an
    // override, solved above, or transferred from the aspect ratio):
    // the `position: relative` post-pass needs it, because percentage
    // `top`/`bottom` compute to `auto` (no offset) when the
    // containing-block height is indefinite (browser-consensus behavior
    // per WebKit bug 26396).
    forest.nodes[idx].height_definite = hspec.is_some();

    // Replaced-element sequential min/max transfer (CSS 2.1 §10.4 /
    // §10.7): with the used width (`resolve_metrics`, above) and `hspec`
    // known, clamp the width by min/max-width, transfer through the
    // intrinsic ratio into an auto height, clamp the height by
    // min/max-height, and transfer back into an auto width — before
    // child layout (replaced boxes are childless; the empty-block arm
    // below consumes the transferred height). Runs only when the
    // intrinsic ratio governs: a specified `aspect-ratio` `<ratio>`
    // keeps its existing `aspect_ratio_hspec` path, and flex/table/grid
    // containers keep their own sizing. `height_definite` above is
    // untouched (a transferred auto height is still definite, exactly
    // as before).
    let hspec = {
        let ratio = intrinsic_ratio(forest, idx);
        let ratio_governs = matches!(
            forest.nodes[idx].data.aspect_ratio,
            None | Some(AspectRatio {
                auto_preferred: true,
                ..
            })
        );
        let is_container = flex_kind(forest, idx).is_some()
            || matches!(
                forest.nodes[idx].data.display,
                Display::Table | Display::InlineTable | Display::Grid | Display::InlineGrid
            );
        if let (Some(ratio), true, false) = (ratio, ratio_governs, is_container) {
            let d = &forest.nodes[idx].data;
            let width_auto = input.width_override.is_none() && matches!(d.width, SpecLen::Auto);
            // The used height is ratio-determined when no caller
            // override fixed it and the specified height is `auto` —
            // or behaves as `auto` (a percentage / percentage-bearing
            // `calc()` against an indefinite containing-block height,
            // matching `specified_height`; `abspos_solved_sizes`
            // already bailed for replaced boxes, so no solved height
            // can make it definite here).
            let height_auto = input.height_override.is_none()
                && match d.height {
                    SpecLen::Auto => true,
                    SpecLen::Pct(_) | SpecLen::Math { had_pct: true, .. } => input.cb_h.is_none(),
                    _ => false,
                };
            let intrinsic_h = forest.nodes[idx]
                .intrinsic
                .map(|d| d.height_or_default().max(0.0))
                .unwrap_or(0.0);
            let tentative_h = hspec.unwrap_or(intrinsic_h);
            // CSS 2.1 §10.3.2: an auto width with a definite height
            // transfers through the intrinsic ratio (content-box terms,
            // css-sizing-4 §4.1) before min/max apply.
            let tentative_w = if width_auto && !height_auto {
                (tentative_h * ratio).max(0.0)
            } else {
                forest.nodes[idx].metrics.content_w
            };
            let (w, h) = replaced_minmax_transfer(
                forest,
                idx,
                input.cb_w,
                input.cb_h,
                tentative_w,
                tentative_h,
                ratio,
                width_auto,
                height_auto,
                input.width_override_final,
            );
            // A transfer back into the width (min/max-height winning)
            // re-stamps the used width; auto margins were already
            // zeroed for auto widths in `resolve_metrics`, so the
            // margin equation needs no re-solve.
            let n = &mut forest.nodes[idx];
            n.metrics.content_w = w;
            n.rect.w = n.metrics.bl + n.metrics.pl + w + n.metrics.pr + n.metrics.br;
            Some(h)
        } else {
            hspec
        }
    };

    if forest.nodes[idx].children.is_empty() {
        // Empty block: when collapsible, its own margins already collapsed
        // into mt_place (which doubles as the bottom placement, so the
        // adjoining set collapses once); otherwise the bottom margin stands
        // alone and the block is opaque to collapsing.
        let mb_place = if is_empty_collapsible(forest, idx, input.cb_h) {
            forest.nodes[idx].mt_place
        } else {
            forest.nodes[idx].metrics.mb
        };
        let content_h = clamp_height(forest, idx, input.cb_h, hspec.unwrap_or(0.0));
        let n = &mut forest.nodes[idx];
        n.content_h = content_h;
        // Recompute bt/pt/pb/bb without holding the borrow across the
        // assignment.
        let (bt, pt, pb, bb) = (n.metrics.bt, n.metrics.pt, n.metrics.pb, n.metrics.bb);
        n.rect.h = bt + pt + content_h + pb + bb;
        n.mb_place = mb_place;
        n.baseline = n.rect.h;
        return;
    }

    // Flex containers get their own layout (the block/inline paths below
    // do not apply). The scrollbar-gutter reservation computed above
    // (`kids_x` / `kids_w`) goes with the dispatch: the flex/grid/table
    // engines derive their content origin/width from the node's own
    // metrics and would otherwise discard it.
    if flex_kind(forest, idx).is_some() {
        layout_flex(
            forest, tasks, idx, input, fonts, pw_cache, scratch, kids_x, kids_w, false,
        );
        return;
    }

    // Tables get their own layout (the block/inline paths below do not
    // apply): the table → row-group → row → cell grid, column widths, and
    // row heights resolve in `layout_table`. `inline-table` is the same
    // algorithm with an atomic inline-level outer box.
    if matches!(
        forest.nodes[idx].data.display,
        Display::Table | Display::InlineTable
    ) {
        layout_table::layout_table(
            forest,
            tasks,
            idx,
            input,
            fonts,
            pw_cache,
            kids_x,
            kids_w,
            &mut scratch.table_grid,
        );
        return;
    }

    // Grid containers get their own layout (the block/inline paths below
    // do not apply): track sizing, auto-placement, and item alignment
    // resolve in `layout_grid`. `inline-grid` is the same algorithm with
    // an atomic inline-level outer box.
    if matches!(
        forest.nodes[idx].data.display,
        Display::Grid | Display::InlineGrid
    ) {
        layout_grid::layout_grid(
            forest, tasks, idx, input, fonts, pw_cache, kids_x, kids_w, false,
        );
        return;
    }

    let all_inline = forest.nodes[idx].children.iter().all(|&c| {
        // Out-of-flow children do not affect the formatting-context
        // choice: only in-flow content counts.
        layout_position::is_out_of_flow(forest, c) || !is_block_level(forest.nodes[c].box_type)
    });
    if all_inline {
        // Inline formatting context. Inline-blocks need their sizes before
        // line breaking, so pre-lay them out (provisional origin) on the
        // same task stack, then resume.
        let ibs_empty = {
            collect_inline_blocks(forest, idx, scratch);
            scratch.ib_out.is_empty()
        };
        if ibs_empty {
            // Borrow, don't clone: `do_inline_layout` only reads the
            // list, and the only `borrow_mut` push site
            // (`resume_block`'s float placement) cannot run while this
            // synchronous call holds the borrow.
            let floats = input.floats.borrow();
            let out = do_inline_layout(
                forest,
                idx,
                kids_x,
                content_top,
                kids_w,
                &floats.entries,
                fonts,
                scratch,
                multicol.as_ref(),
                hspec,
            );
            finish_content_height(
                forest,
                idx,
                input.cb_h,
                min_floor_hspec(input.height_is_min, hspec, out.height),
                out.height,
            );
        } else {
            tasks.push(Task::ResumeInline(InlineFrame {
                idx,
                content_x: kids_x,
                content_top,
                content_w: kids_w,
                cb_h: input.cb_h,
                hspec,
                height_is_min: input.height_is_min,
                floats: input.floats.clone(),
            }));
            // `ib_out` is borrowed here; `tasks` never touches scratch,
            // so the borrow is sound. Cleared after use so the buffer
            // is reused by the next container.
            for &ib in scratch.ib_out.iter().rev() {
                // Percentage heights inside inline-blocks resolve against
                // the container's definite specified content height
                // (`hspec`); `None` when it is indefinite, matching block
                // children.
                tasks.push(Task::Block(BlockInput {
                    abs_pct_height_base: None,
                    idx: ib,
                    cb_x: 0.0,
                    cb_w: kids_w,
                    cb_h: child_hspec,
                    shrink: true,
                    width_override: None,
                    width_override_final: false,
                    height_override: None,
                    height_is_min: false,
                    collapsed_border_wins: None,
                    abspos: false,
                    // Inline-blocks establish a BFC: their floats are
                    // contained and never leak into this context.
                    floats: fresh_floats(),
                    is_float: false,
                }));
            }
            scratch.ib_out.clear();
        }
    } else {
        // Block formatting context.
        let top_collapsed = forest.nodes[idx].top_collapsed;
        tasks.push(Task::ResumeBlock(BlockFrame {
            idx,
            child_i: 0,
            cursor_y: content_top,
            prev_mbottom: 0.0,
            first: true,
            top_collapsed,
            content_x: kids_x,
            content_top,
            content_w: kids_w,
            cb_h: input.cb_h,
            child_cb_h: child_hspec,
            own_hspec: hspec,
            height_is_min: input.height_is_min,
            floats: input.floats.clone(),
            floats_start: input.floats.borrow().len(),
            pending_float_y: None,
            max_inflow_top: f32::NEG_INFINITY,
            auto_redriven: false,
        }));
    }
}

fn resume_block(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    mut fr: BlockFrame,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    scratch: &mut LayoutScratch,
) {
    let idx = fr.idx;
    // Account for the child that just finished (if any). Out-of-flow
    // children (`position: absolute | fixed`) take no space: when the
    // just-finished child is one of those, flow state is untouched.
    // Floats likewise take no space in normal flow, but they are
    // positioned now that their size is known (CSS 2.1 §9.5).
    if fr.child_i > 0 {
        let just = forest.nodes[idx].children[fr.child_i - 1];
        if layout_position::is_out_of_flow(forest, just) {
            // No-op: flow state untouched.
        } else if is_float(forest, just) {
            let fy = fr
                .pending_float_y
                .take()
                .expect("float child always stashes its y at encounter");
            let left = matches!(forest.nodes[just].data.float, Float::Left);
            let (ml, mt, mw, mh) = {
                let n = &forest.nodes[just];
                let m = &n.metrics;
                (m.ml, m.mt, m.ml + n.rect.w + m.mr, m.mt + n.rect.h + m.mb)
            };
            // One borrow for placement and the push: `place_float` needs
            // `&mut` for its reusable query stack.
            let (fx, fy) = {
                let mut floats = fr.floats.borrow_mut();
                place_float(&mut floats, fr.content_x, fr.content_w, left, mw, mh, fy)
            };
            // CSS 2.1 §9.5.1: a float's top may not be higher than its
            // containing block's top. `place_float` only ever moves down
            // from its `y_start`, so the margin-box top can land above
            // the content-box top only when `y_start` already sits above
            // it — possible when an earlier sibling's negative margins
            // pulled the flow cursor above `content_top` (e.g. a first
            // child with a negative top margin inside a bordered parent).
            // Clamp defensively; the normal case never engages it.
            let fy = fy.max(fr.content_top);
            {
                let n = &mut forest.nodes[just];
                // Margin-box origin to border-box origin.
                n.rect.x = fx + ml;
                n.rect.y = fy + mt;
            }
            fr.floats.borrow_mut().push(FloatEntry {
                x: fx,
                y: fy,
                w: mw,
                h: mh,
                left,
                node: just,
            });
            // The cursor does not advance and `fr.first` stays set:
            // floats are out of flow for margin collapsing (CSS 2.1
            // §9.4), exactly like out-of-flow skips above.
        } else {
            let prev = just;
            // A BFC root's border box must not overlap the margin box of
            // any float in the same context (CSS 2.1 §9.5). The
            // pre-layout narrowing band is only a point test at the top
            // edge for auto-height children, so re-test over the true
            // extent now that the height is known; on violation the child
            // clears below the intruding floats. The shift is deferred
            // (`pend_y`), exactly like out-of-flow placement, so nested
            // abspos static positions stay in a consistent provisional
            // frame until `apply_pending_shifts`.
            let bfc_shift = bfc_post_layout_shift(forest, &fr, prev);
            if bfc_shift > 0.0 {
                forest.nodes[prev].pend_y += bfc_shift;
            }
            let (prev_top, prev_bottom, prev_mb_place, prev_mt_place) = {
                let p = &forest.nodes[prev];
                // `pend_y` carries the shift above: rects stay
                // provisional until the final pass.
                let top = p.rect.y + p.pend_y;
                (top, top + p.rect.h, p.mb_place, p.mt_place)
            };
            fr.cursor_y = prev_bottom;
            // CSS 2.1 §9.5.1 rules 5/6: a later float may not rise above
            // this in-flow box's outer (margin) top — its border top
            // minus its top placement margin — not merely its border
            // top (its line boxes sit at or below it).
            if !prev_top.is_nan() {
                fr.max_inflow_top = fr.max_inflow_top.max(prev_top - prev_mt_place);
            }
            // A margin-transparent child does not break the adjoining-margin
            // set: instead of emitting its bottom margin a second time, fold
            // its through-collapsed margin into the pending pool, so the whole
            // set collapses exactly once at the next opaque child (or the
            // parent's bottom margin). A first child already absorbed into the
            // parent's top margin contributes nothing further.
            if is_collapse_transparent(forest, prev, fr.child_cb_h) {
                // `fr.first` is still set exactly when this is the first
                // in-flow child (out-of-flow skips never clear it).
                if fr.first && fr.top_collapsed {
                    fr.prev_mbottom = 0.0;
                } else {
                    fr.prev_mbottom = collapse(fr.prev_mbottom, prev_mt_place);
                }
            } else {
                fr.prev_mbottom = prev_mb_place;
            }
            fr.first = false;
        }
    }
    let kids_len = forest.nodes[idx].children.len();
    if fr.child_i < kids_len {
        let c = forest.nodes[idx].children[fr.child_i];
        if layout_position::is_out_of_flow(forest, c) {
            // Out of flow: record for the `layout_position` post-pass and
            // take no space. `compute_mt_place` is deliberately not run:
            // out-of-flow boxes never collapse margins (CSS 2.1 §8.3.1),
            // and it would leave stale collapse state on the subtree the
            // post-pass lays out. The static position is the border-box
            // origin flow would have given the box (own margins only).
            // No `Task::Block` is pushed: the post-pass lays the subtree
            // out against its containing block.
            resolve_metrics(forest, c, fr.content_x, fr.content_w, true, None, false);
            // F1: content-based widths resolve as fill (provisional) in
            // `resolve_metrics`; measure before recording the static
            // position so auto margins center on the measured width
            // (`layout_position` consumes `static_x` when left/right are
            // auto). css-sizing-3 §4 min/max keywords are measured here
            // too, like the block arm.
            if matches!(
                forest.nodes[c].data.width,
                SpecLen::FitContent | SpecLen::MinContent | SpecLen::MaxContent
            ) {
                let (ml, mr, extras) = {
                    let m = &forest.nodes[c].metrics;
                    (m.ml, m.mr, m.bl + m.pl + m.pr + m.br)
                };
                let spec = forest.nodes[c].data.width;
                let avail = (fr.content_w - ml - mr - extras).max(0.0);
                let (min_spec, max_spec) = {
                    let d = &forest.nodes[c].data;
                    (d.min_width, d.max_width)
                };
                let w = content_based_width(
                    forest,
                    c,
                    spec,
                    fonts,
                    pw_cache,
                    &mut scratch.mcw,
                    &mut scratch.int_bufs,
                    &mut scratch.int_inner,
                    avail,
                );
                let lo = measured_minmax_width(
                    forest,
                    c,
                    min_spec,
                    fonts,
                    pw_cache,
                    &mut scratch.mcw,
                    &mut scratch.int_bufs,
                    &mut scratch.int_inner,
                    avail,
                );
                let hi = measured_minmax_width(
                    forest,
                    c,
                    max_spec,
                    fonts,
                    pw_cache,
                    &mut scratch.mcw,
                    &mut scratch.int_bufs,
                    &mut scratch.int_inner,
                    avail,
                );
                resolve_metrics(
                    forest,
                    c,
                    fr.content_x,
                    fr.content_w,
                    true,
                    // Saturate an infinite max-content to a huge finite
                    // used width (see `MAX_USED_SIZE`).
                    Some(saturate_used_size(clamp_measured(w, lo, hi))),
                    false,
                );
            }
            let mt = forest.nodes[c].metrics.mt;
            let gap = if fr.first && fr.top_collapsed {
                0.0
            } else {
                collapse(if fr.first { 0.0 } else { fr.prev_mbottom }, mt)
            };
            let (sx, sy) = (forest.nodes[c].rect.x, fr.cursor_y + gap);
            layout_position::record_abs_item(forest, c, sx, sy);
            fr.child_i += 1;
            tasks.push(Task::ResumeBlock(fr));
            return;
        }
        if is_float(forest, c) {
            // Floats take no space in normal flow and never collapse
            // margins (CSS 2.1 §8.3.1, §9.5): resolve metrics without
            // centering (auto margins compute to zero, §10.3.5), apply
            // the float's own `clear`, and lay the subtree out now (an
            // `auto` width shrink-to-fits, §10.3.5). Positioning waits
            // for the subtree's size, so the margin-box top is stashed
            // on the frame; the float is placed when its `Task::Block`
            // completes. A float establishes a BFC, so its contents get
            // a fresh float list.
            resolve_metrics(forest, c, fr.content_x, fr.content_w, false, None, false);
            // Margin-box top: margins never collapse on floats, so the
            // margin sits whole between the cursor and the border box.
            // `clear` clears the border edge (CSS 2.1 §9.5.2): the margin
            // may overlap the float, so the float bottom is offset by the
            // top margin to keep `fy` a margin-box top.
            let mut fy = fr.cursor_y;
            // CSS 2.1 §9.5.1 rules 5/6: the float's outer top may not
            // be higher than the outer top of any earlier in-flow block
            // box (or line box containing earlier content). The cursor
            // alone does not enforce it — a later sibling's negative
            // margin can drag the cursor above an earlier sibling's
            // outer top — so floor by the highest in-flow outer top
            // seen so far.
            fy = fy.max(fr.max_inflow_top);
            let clear = forest.nodes[c].data.clear;
            if clear != Clear::None
                && let Some(cy) = clearance_y(&fr.floats.borrow().entries, clear)
            {
                fy = fy.max(cy - forest.nodes[c].metrics.mt);
            }
            fr.pending_float_y = Some(fy);
            fr.child_i += 1;
            let (child_cb_h, cb_x, cb_w) = (fr.child_cb_h, fr.content_x, fr.content_w);
            tasks.push(Task::ResumeBlock(fr));
            tasks.push(Task::Block(BlockInput {
                abs_pct_height_base: None,
                idx: c,
                cb_x,
                cb_w,
                cb_h: child_cb_h,
                shrink: false,
                width_override: None,
                width_override_final: false,
                height_override: None,
                height_is_min: false,
                collapsed_border_wins: None,
                abspos: false,
                floats: fresh_floats(),
                is_float: true,
            }));
            return;
        }
        // `clear` introduces clearance per CSS 2.1 §9.5.2, computed from
        // the hypothetical top border edge (as if `clear` were `none`);
        // the used position is the greater of that edge and the lowest
        // relevant float bottom. Applied after the collapsed-margin gap
        // so margins — including negative ones — fold into the
        // hypothetical position instead of stacking on top of the
        // clearance. The float entries' margin boxes already include
        // their margins.
        // `column-span: all` spanners lay out against the full multicol
        // content width up front — not one column width — so their inner
        // content (headings, wrapping text) uses the whole spanner row.
        // `balance_block_columns` widens the border box to this same
        // width and sheds the flow-pass float intrusion, so the spanner
        // also skips the BFC float-narrowing below: it takes the full
        // row unconditionally.
        let span_w = if layout_multicol::is_spanner(forest, c) {
            forest.nodes[idx]
                .multicol
                .as_deref()
                .map(layout_multicol::spanner_row_width)
        } else {
            None
        };
        let (flow_x, flow_w) = span_w.map_or((fr.content_x, fr.content_w), |w| (fr.content_x, w));
        let clear = forest.nodes[c].data.clear;
        resolve_metrics(forest, c, flow_x, flow_w, true, None, false);
        compute_mt_place(forest, c, fr.child_cb_h);
        let mt_place = forest.nodes[c].mt_place;
        // A margin-transparent empty block joins the adjoining set rather
        // than breaking it: no gap is emitted for it here (its
        // through-collapsed margin folds into the pending pool when it
        // finishes, above). Adjoining vertical margins collapse; when the
        // parent's top margin already absorbed this child's, there is no
        // gap either.
        let transparent_fold =
            is_collapse_transparent(forest, c, fr.child_cb_h) && !(fr.first && fr.top_collapsed);
        let gap = if transparent_fold || (fr.first && fr.top_collapsed) {
            0.0
        } else {
            collapse(if fr.first { 0.0 } else { fr.prev_mbottom }, mt_place)
        };
        let hypo = fr.cursor_y + gap;
        forest.nodes[c].rect.y = if clear != Clear::None
            && let Some(cy) = clearance_y(&fr.floats.borrow().entries, clear)
        {
            hypo.max(cy)
        } else {
            hypo
        };
        fr.child_i += 1;
        // A BFC root's border box must not overlap the margin box of any
        // float in the same BFC (CSS 2.1 §9.5): intruding floats narrow
        // the child to the space beside them (x shifted past left floats,
        // width shrunk). When no space remains at this height, the child
        // clears below the intruding floats. (Non-BFC children keep the
        // full width: floats may intrude into their content, narrowing
        // only their line boxes.)
        let (child_cb_h, cb_x, cb_w) = if span_w.is_some() {
            // Spanner: full row, no float narrowing (see above).
            (fr.child_cb_h, flow_x, flow_w)
        } else if establishes_bfc(forest, c) {
            let floats = fr.floats.borrow();
            // A BFC root's used margin-box width: with `width: auto` it
            // fills whatever space remains beside the floats, but with a
            // specified width it keeps that width and must clear below
            // any floats it cannot fit beside (CSS 2.1 §9.5). An auto
            // width still cannot shrink below `min-width`: when the
            // min-width (in margin-box terms) exceeds the space beside
            // the floats, the box clears below them instead of
            // overlapping their margin boxes.
            let (specified_w, border_w, min_mb_w) = {
                let n = &forest.nodes[c];
                let m = &n.metrics;
                let d = &n.data;
                let min_w = d
                    .min_width
                    .resolve(
                        d.font_size,
                        forest.root_font_size,
                        forest.vp_w,
                        forest.vp_h,
                        fr.content_w,
                    )
                    .unwrap_or(0.0)
                    .max(0.0);
                // min-width constrains the content box under
                // `content-box` sizing, the border box under
                // `border-box`; the fit test below needs margin-box
                // terms.
                let min_mb_w = match d.box_sizing {
                    BoxSizing::BorderBox => min_w,
                    BoxSizing::ContentBox => min_w + m.bl + m.pl + m.pr + m.br,
                } + m.ml
                    + m.mr;
                (
                    !matches!(d.width, SpecLen::Auto),
                    m.bl + m.pl + m.content_w + m.pr + m.br,
                    min_mb_w,
                )
            };
            // F3: for content-based widths the fill-provisional `used_w`
            // above is wrong (it fills the whole content box); the fit
            // test measures the content-based width against the narrowed
            // available width instead.
            let content_spec = match forest.nodes[c].data.width {
                s @ (SpecLen::FitContent | SpecLen::MinContent | SpecLen::MaxContent) => Some(s),
                _ => None,
            };
            // With auto side margins and a definite (content-based)
            // width, CSS 2.1 §10.3.3 absorbs the remainder into the
            // margins: the margin box exactly fills the available width,
            // so the box always fits beside the floats.
            let auto_margins = content_spec.is_some() && {
                let d = &forest.nodes[c].data;
                matches!(d.margin[3], SpecMargin::Auto) || matches!(d.margin[1], SpecMargin::Auto)
            };
            let (fit_ml, fit_mr, fit_extras, fit_min_spec, fit_max_spec) = {
                let n = &forest.nodes[c];
                let m = &n.metrics;
                let d = &n.data;
                (
                    m.ml,
                    m.mr,
                    m.bl + m.pl + m.pr + m.br,
                    d.min_width,
                    d.max_width,
                )
            };
            let mut cy = forest.nodes[c].rect.y;
            // Floats overlapping the child's vertical extent narrow it,
            // not just floats at its top edge: a float whose margin box
            // starts below the top (top margin, clearance, float
            // stacking) still overlaps the child's body, and a BFC
            // root's border box must not overlap any float in the same
            // context (CSS 2.1 §9.5). The band is the specified border-
            // box height when known; otherwise the top edge is
            // point-tested as before (`next_up` keeps the old `f.y <= cy`
            // boundary exactly under the band predicate).
            let border_pad = {
                let m = &forest.nodes[c].metrics;
                m.bt + m.pt + m.pb + m.bb
            };
            // Precompute the band height so the clearance loop below
            // needs no `&forest` borrow: measuring content-based widths
            // (F3) takes `&mut forest` per candidate width.
            let band_h: Option<f32> = match specified_height(forest, c, fr.child_cb_h) {
                // Clamp by min-height/max-height, mirroring the used
                // height: a min-height taller than the specified height
                // extends the band floats must not intrude into.
                Some(h) => Some(border_pad + clamp_height(forest, c, fr.child_cb_h, h)),
                None => None,
            };
            let y1 = |cy: f32| match band_h {
                Some(h) => cy + h,
                None => cy.next_up(),
            };
            let (mut dx, mut avail) =
                line_float_avail(&floats.entries, fr.content_x, fr.content_w, cy, y1(cy));
            loop {
                // The width the box would actually occupy at this height,
                // in border-box terms: CSS 2.1 §9.5 constrains the BFC
                // root's *border box* against the floats' margin boxes,
                // while margins re-resolve against the narrowed width in
                // the Block pass below — so the fit test strips margins.
                // For a specified width that is the used border-box
                // width; for `width: auto` the available space floored
                // by min-width; for content-based widths the measured
                // width against the narrowed available width (F3).
                let fit = if auto_margins {
                    avail
                } else if let Some(spec) = content_spec {
                    bfc_content_fit_width(
                        forest,
                        c,
                        spec,
                        fit_ml,
                        fit_mr,
                        fit_extras,
                        fit_min_spec,
                        fit_max_spec,
                        fonts,
                        pw_cache,
                        &mut scratch.mcw,
                        &mut scratch.int_bufs,
                        &mut scratch.int_inner,
                        avail,
                    ) - fit_ml
                        - fit_mr
                } else if specified_w {
                    border_w
                } else {
                    avail.max(min_mb_w)
                };
                // A box with a definite used width fits even a zero-width
                // gap between floats (WPT zero-space-between-floats: its
                // border box does not overlap any float's margin box). A
                // `width: auto` box instead needs positive space beside
                // the floats; with none it clears below them.
                let needs_positive_space = !specified_w && content_spec.is_none();
                if fit <= avail && (!needs_positive_space || avail > 0.0) {
                    break;
                }
                let bottom = intruding_float_bottom(&floats.entries, cy, y1(cy));
                if bottom <= cy {
                    break;
                }
                cy = bottom;
                (dx, avail) =
                    line_float_avail(&floats.entries, fr.content_x, fr.content_w, cy, y1(cy));
            }
            drop(floats);
            forest.nodes[c].rect.y = cy;
            (fr.child_cb_h, fr.content_x + dx, avail.max(0.0))
        } else {
            (fr.child_cb_h, fr.content_x, fr.content_w)
        };
        // Non-BFC children share this context's float list (a float
        // placed by an earlier sibling intrudes into their content);
        // BFC children get a fresh list (floats never cross a BFC
        // boundary).
        let child_floats = if establishes_bfc(forest, c) {
            fresh_floats()
        } else {
            fr.floats.clone()
        };
        let child_is_float = is_float(forest, c);
        tasks.push(Task::ResumeBlock(fr));
        tasks.push(Task::Block(BlockInput {
            abs_pct_height_base: None,
            idx: c,
            cb_x,
            cb_w,
            cb_h: child_cb_h,
            shrink: false,
            width_override: None,
            width_override_final: false,
            height_override: None,
            height_is_min: false,
            collapsed_border_wins: None,
            abspos: false,
            floats: child_floats,
            is_float: child_is_float,
        }));
    } else {
        // All children placed: height and bottom-margin collapse. The
        // height spec is the one the prologue computed (`own_hspec`):
        // it carries caller overrides (flex main sizes, solved abspos
        // heights) that re-deriving from style would drop. This is the
        // container's own spec — the base threaded to children
        // (`child_cb_h`) is indefinite for a table cell's measurement
        // pass (css-tables-3 §3.10.6).
        let hspec = fr.own_hspec;
        let mut natural_h = (fr.cursor_y - fr.content_top).max(0.0);
        // v1 multi-column layout: redistribute the single-column flow
        // into N columns; the natural height becomes the tallest column.
        // Floats (left in the first column by the flow pass) still extend
        // the height below.
        // `MulticolParams` is `Copy`: lift it out before the `&mut forest`
        // call below so the immutable borrow does not span it.
        let mc = forest.nodes[idx].multicol.as_deref().copied();
        if let Some(mc) = mc {
            natural_h = layout_multicol::balance_block_columns(
                forest,
                idx,
                &mc,
                fr.content_x,
                fr.content_top,
                hspec,
                // The balance re-places direct float children into
                // columns; their context entries are updated to the
                // placed geometry in the same pass.
                &mut fr.floats.borrow_mut(),
            );
        }
        // A block formatting context contains its floats: extend the
        // height to cover the floats this subtree placed (CSS 2.1
        // §10.6.7). Entries before `floats_start` belong to ancestors.
        if establishes_bfc(forest, idx) {
            let floats = fr.floats.borrow();
            for f in &floats.entries[fr.floats_start..] {
                natural_h = natural_h.max(f.y + f.h - fr.content_top);
            }
        }
        let (bt, pt, pb, bb, mb) = {
            let m = &forest.nodes[idx].metrics;
            (m.bt, m.pt, m.pb, m.bb, m.mb)
        };
        let last = forest.nodes[idx]
            .children
            .iter()
            .rev()
            .copied()
            .find(|&l| !layout_position::is_out_of_flow(forest, l) && !is_float(forest, l));
        // The bottom margin collapses with the last child's when there is
        // no bottom border/padding and the height is auto. Only true
        // block-level children participate (inline-blocks are atomic), and
        // never through a BFC boundary (CSS 2.1 §8.3.1). Out-of-flow
        // children are skipped above: they take no space and their margins
        // never collapse with the parent's; floats likewise (CSS 2.1
        // §9.5).
        let mb_place = match (hspec.is_none() && bb + pb == 0.0, last) {
            (true, Some(l))
                if matches!(
                    forest.nodes[l].box_type,
                    BoxType::Block | BoxType::AnonymousBlock
                ) && !establishes_bfc(forest, idx) =>
            {
                collapse(mb, forest.nodes[l].mb_place)
            }
            _ => {
                if hspec.is_none() {
                    // The last child's bottom margin does not collapse with
                    // the parent's, so an auto-height parent still contains
                    // the child's margin box (CSS 2.1 §10.6.3): extend the
                    // natural height to its margin-box bottom.
                    if let Some(l) = last {
                        let n = &forest.nodes[l];
                        // `pend_y` carries a post-layout BFC/float shift
                        // (see `bfc_post_layout_shift`): rects stay
                        // provisional until the final pass.
                        natural_h = natural_h
                            .max(n.rect.y + n.pend_y + n.rect.h + n.mb_place - fr.content_top);
                    }
                }
                mb
            }
        };
        let content_h = clamp_height(
            forest,
            idx,
            fr.cb_h,
            min_floor_hspec(fr.height_is_min, hspec, natural_h).unwrap_or(natural_h),
        );
        let n = &mut forest.nodes[idx];
        n.content_h = content_h;
        n.natural_content_h = natural_h;
        n.rect.h = bt + pt + content_h + pb + bb;
        n.mb_place = mb_place;
        n.baseline = n.rect.h;
        // Shared `overflow-y: auto` scrollbar re-drive (see
        // [`auto_scrollbar_redrive`]): when the shown scrollbar went
        // unreserved, re-drive this box's children once against the
        // narrowed width. Stale out-of-flow static positions were
        // already cleared by the helper, so the second pass re-records
        // them fresh.
        if let Some((kids_x2, kids_w2)) = auto_scrollbar_redrive(
            forest,
            scratch,
            &AutoRedriveCheck {
                idx,
                natural_h,
                content_h,
                content_w: fr.content_w,
                content_x: fr.content_x,
                auto_redriven: fr.auto_redriven,
            },
        ) {
            // Drop this pass's float placements from the shared
            // context: entries before `floats_start` belong to
            // ancestors and stay; the re-drive re-places this
            // subtree's own floats. Goes through `FloatCtx::truncate`
            // (not `entries.truncate`) so the interval index and
            // `max_top` are rebuilt from the surviving entries — a
            // bare truncate would leave stale index nodes whose ids
            // alias the re-pushed floats.
            fr.floats.borrow_mut().truncate(fr.floats_start);
            tasks.push(Task::ResumeBlock(BlockFrame {
                idx,
                child_i: 0,
                cursor_y: fr.content_top,
                prev_mbottom: 0.0,
                first: true,
                top_collapsed: fr.top_collapsed,
                content_x: kids_x2,
                content_top: fr.content_top,
                content_w: kids_w2,
                cb_h: fr.cb_h,
                child_cb_h: fr.child_cb_h,
                own_hspec: fr.own_hspec,
                height_is_min: fr.height_is_min,
                floats: fr.floats,
                floats_start: fr.floats_start,
                pending_float_y: None,
                max_inflow_top: f32::NEG_INFINITY,
                auto_redriven: true,
            }));
        }
    }
}

/// Decision inputs for the shared `overflow-y: auto` scrollbar re-drive
/// ([`auto_scrollbar_redrive`]): the box, its natural vs. used content
/// heights, the content-box geometry the completed pass laid out
/// against, and whether this frame already is a re-drive.
pub(crate) struct AutoRedriveCheck {
    pub(crate) idx: usize,
    pub(crate) natural_h: f32,
    pub(crate) content_h: f32,
    pub(crate) content_w: f32,
    pub(crate) content_x: f32,
    pub(crate) auto_redriven: bool,
}

/// Drop this layout pass's out-of-flow static-position records for a
/// subtree about to be re-driven: the re-drive re-lays out the whole
/// subtree, so every out-of-flow descendant is re-encountered and
/// re-recorded — but `record_abs` early-returns on `abspos_recorded`
/// (see `layout_position`), which would keep the first pass's stale
/// `static_x`/`static_y` (the `static_y` predates the rewrap growth of
/// all preceding content, so it can be arbitrarily stale). Clearing the
/// flags and filtering `abs_items` lets the second pass record fresh
/// positions.
///
/// The subtree root itself is skipped: its own record belongs to the
/// ancestor pass that placed it, which the re-drive does not repeat —
/// dropping it would lose the box's positioning entirely.
fn clear_redrive_abspos(forest: &mut Forest, idx: usize) {
    // No `children.clone()`: the inner walk already extends from
    // `iter().copied()`; seed the same way.
    let mut stack: Vec<usize> = Vec::new();
    stack.extend(forest.nodes[idx].children.iter().copied());
    while let Some(n) = stack.pop() {
        if layout_position::is_out_of_flow(forest, n) {
            forest.nodes[n].abspos_recorded = false;
        }
        stack.extend(forest.nodes[n].children.iter().copied());
    }
    let nodes = &forest.nodes;
    forest.abs_items.retain(|a| nodes[a.node].abspos_recorded);
}

/// Shared `overflow-y: auto` scrollbar re-drive for the block, flex,
/// and grid completions.
///
/// CSS Overflow 3 §2.2: `overflow-y: auto` shows the classic vertical
/// scrollbar exactly when content overflows the padding box, and the
/// shown scrollbar consumes its thickness from the content box. The
/// `layout_block` prologue narrowed the children's width from style
/// facts alone — before children were laid out — so this real
/// (unforced, unreserved) overflow went unaccounted: the painted
/// scrollbar overlays the last SCROLLBAR_THICKNESS px of content. Now
/// that the used content height is known, detect the definite-height
/// overflow (`natural_h > content_h`; an auto height grows to fit
/// instead — floats are already folded into `natural_h` by the caller,
/// and `height_is_min` floors rather than clamps) and report the
/// narrowed `(kids_x, kids_w)` for one re-drive of this box's children.
///
/// The strip takes the per-side max with the prologue's reservations
/// (a shown scrollbar paints exactly the reserved strip, never
/// stacking) and shifts the origin past an inline-start (rtl) strip.
/// The document element / body never re-drive here: the app never
/// paints a nested scrollbar for them, so their scrollbar geometry
/// belongs to the viewport pass. Bounded to one extra pass per frame:
/// the single `auto_redriven` bound prevents loops, so no fixpoint is
/// needed or allowed — note the bound is load-bearing, not the
/// overflow direction: replaced / aspect-ratio children can make
/// `natural_h` *shrink* after narrowing, leaving a bounded cosmetic
/// residual instead of looping.
///
/// On `Some`, the subtree's stale out-of-flow static positions are
/// already cleared (see [`clear_redrive_abspos`]); the caller pushes
/// its own engine-specific re-drive task.
pub(crate) fn auto_scrollbar_redrive(
    forest: &mut Forest,
    scratch: &LayoutScratch,
    c: &AutoRedriveCheck,
) -> Option<(f32, f32)> {
    let idx = c.idx;
    let (al, ar) = forest.nodes[idx]
        .style
        .as_deref()
        .map(crate::auto_scrollbar_reservation)
        .unwrap_or((0.0, 0.0));
    let is_viewport_scroller = forest.nodes[idx].dom.is_some_and(|d| {
        scratch.viewport_scroller_doms.0 == Some(d) || scratch.viewport_scroller_doms.1 == Some(d)
    });
    if (al > 0.0 || ar > 0.0)
        && c.natural_h > c.content_h + 0.001
        && !c.auto_redriven
        && !is_viewport_scroller
    {
        // The reservations the prologue already applied, exactly as
        // it computed them.
        let (gl, gr) = if forest.gutter_propagated_root == Some(idx) {
            (0.0, 0.0)
        } else if let Some(style) = forest.nodes[idx].style.as_ref() {
            let (gl, gr) = crate::scrollbar_gutter_reservation(style);
            let (fl, fr_) = crate::forced_scrollbar_reservation(style);
            (gl.max(fl), gr.max(fr_))
        } else {
            (0.0, 0.0)
        };
        let (nl, nr) = (gl.max(al), gr.max(ar));
        let kids_w2 = (c.content_w - (nl - gl) - (nr - gr)).max(0.0);
        if kids_w2 < c.content_w - 0.001 {
            let kids_x2 = c.content_x - gl + nl;
            clear_redrive_abspos(forest, idx);
            return Some((kids_x2, kids_w2));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Inline layout
// ---------------------------------------------------------------------------

/// One fragment of inline content on a line.
struct Frag {
    w: f32,
    ascent: f32,
    descent: f32,
    lh: f32,
    valign: VAlign,
    /// Extra vertical shift from `vertical-align: <length>` in px
    /// (positive raises the box); 0.0 for keyword values.
    vshift: f32,
    /// Arena index when this fragment is an inline-block.
    ib: Option<usize>,
    is_space: bool,
    break_before: bool,
    /// Distinct from `break_before`: true only when this fragment's
    /// break is a `\n` paragraph separator (UAX #9 P1) — a hard-line
    /// start in `pre` / `pre-wrap` / `pre-line` / `break-spaces` text
    /// (threaded from `PreWrapPiece::forced_break`). Soft wraps set
    /// `break_before` (via pre-fit pieces) but never this flag. Bidi
    /// paragraph segmentation for `unicode-bidi: plaintext` splits on
    /// this flag only; line breaking keeps using `break_before`, so
    /// non-bidi paths are unaffected.
    forced_break_before: bool,
    /// The text run's own computed `white-space` (for inline-blocks, the
    /// block's own). Line breaking derives its wrap/pre/trim mode from
    /// this per fragment, never from the containing block or flex item:
    /// a `normal` span inside a `pre` block still wraps, and a
    /// `pre-wrap` span inside a `normal` block keeps its spaces.
    ws: WhiteSpace,
    /// Byte range into the text node's (transformed) text that this
    /// fragment renders; `None` for inline-block fragments. Feeds
    /// `TextFragment::text_range` in the paint contract.
    text_range: Option<Range<usize>>,
    /// Positioned on a line (false for trimmed-away spaces).
    placed: bool,
    /// Aligned vertically (pass 1 vs pass 2).
    aligned: bool,
    /// Absolute x of the fragment's left edge.
    x: f32,
    /// Top/bottom relative to the line's baseline.
    rt: f32,
    rb: f32,
    abs_top: f32,
}

struct Line {
    start: usize,
    end: usize,
    width: f32,
    /// Top line-box extent relative to the baseline (from [`align_line`]).
    top: f32,
    /// Left-float intrusion at the line's top: x shift of the line box.
    dx: f32,
    /// Available width at the line's top after float intrusions.
    avail: f32,
    /// Absolute y of the line's top. Float clearance can shift a line
    /// below `content_top +` the stacked heights, so this is recorded at
    /// break time rather than re-accumulated.
    y: f32,
}

/// Fragment range of one inline/text box: its slice of the fragment
/// list plus the indices of directly nested ranges. (Module level so
/// [`LayoutScratch`] can own the buffer.)
///
/// Child-range indices, allocation-free for the common case: an inline
/// box almost always nests only a few child ranges (a `<b>` around one
/// text run has exactly one), so the first four are stored inline and
/// only deeper nesting spills to a heap `Vec`.
struct RangeChildren {
    inline: [usize; 4],
    /// Valid entries in `inline` (0..=4); `spill` is only used once this
    /// reaches 4.
    len: u8,
    spill: Vec<usize>,
}

impl RangeChildren {
    fn new() -> Self {
        RangeChildren {
            inline: [0; 4],
            len: 0,
            spill: Vec::new(),
        }
    }

    fn push(&mut self, v: usize) {
        if self.len < 4 {
            self.inline[self.len as usize] = v;
            self.len += 1;
        } else {
            self.spill.push(v);
        }
    }

    fn iter(&self) -> impl Iterator<Item = &usize> {
        self.inline[..self.len as usize]
            .iter()
            .chain(self.spill.iter())
    }
}

struct FragRange {
    box_idx: usize,
    start: usize,
    end: usize,
    children: RangeChildren,
}

/// Inline-tree walk step for fragment collection. (Module level so
/// [`LayoutScratch`] can own the buffer.)
enum InlineWalkItem {
    Enter(usize),
    /// (box, range index); the range was pushed at `Enter` time and
    /// its `end` is filled in here.
    Exit(usize, usize),
}

/// Reusable per-layout-pass scratch buffers.
///
/// Several hot per-container `Vec`s — inline fragment collection and
/// line breaking, `collect_inline_blocks`, flex item staging, and
/// intrinsic-size measurement — were allocated fresh for every
/// container on every layout pass. They are hoisted here instead: one
/// struct is created per top-level [`layout_forest`] /
/// [`layout_position::layout_abs_items`] call and threaded through the
/// task loop, and each user clears (never drops) its buffers, so
/// capacity survives across containers. All uses are strictly
/// sequential — no function holds a scratch borrow across a nested
/// layout call — so sharing one struct across the whole pass is sound.
pub(crate) struct LayoutScratch {
    /// `do_inline_layout`: collected fragments.
    frags: Vec<Frag>,
    /// `do_inline_layout`: fragment ranges per inline box.
    frag_ranges: Vec<FragRange>,
    /// `do_inline_layout`: inline-tree walk stack.
    walk: Vec<InlineWalkItem>,
    /// `do_inline_layout`: open range indices.
    open_ranges: Vec<usize>,
    /// `do_inline_layout`: out-of-flow boxes met during the walk.
    inline_abs: Vec<(usize, Option<usize>)>,
    /// `do_inline_layout`: broken lines.
    lines: Vec<Line>,
    /// `do_inline_layout`: per-fragment inline-level forced-break marks,
    /// parallel to `frags` (kept out of [`Frag`] so the struct literal
    /// stays constructible elsewhere in the crate). `frag_force_before[i]`
    /// means an inline-level `break-before` forced break (see
    /// [`forced_break_of`]) applies at the break opportunity before
    /// fragment `i` — the first fragment of an inline box or the single
    /// fragment of an inline-block — and `frag_force_after[i]` the same
    /// for `break-after` after the box's last fragment. Only populated
    /// for multicol containers (inline forced breaks are honored only
    /// where a column can start); all false otherwise.
    frag_force_before: Vec<bool>,
    frag_force_after: Vec<bool>,
    /// `do_inline_layout`: pooled paragraph base directions for the
    /// `unicode-bidi: plaintext` pass (one per paragraph).
    para_dirs: Vec<bool>,
    /// `do_inline_layout`: pooled per-fragment paragraph index,
    /// parallel to `frags` (see the `para_dirs` paragraph pass).
    frag_para: Vec<usize>,
    /// `collect_inline_blocks`: collected inline-block indices.
    ib_out: Vec<usize>,
    /// `collect_inline_blocks`: walk stack.
    ib_stack: Vec<usize>,
    /// Reusable buffer for `min_content_width_pooled`'s node-indexed widths.
    mcw: Vec<f32>,
    /// Pooled preorder-walk buffers for intrinsic measurement
    /// ([`preferred_width_mcw`] / [`min_content_width_pooled`]), reused
    /// instead of allocating per measurement. Shared by the top-level
    /// intrinsic call sites and the flex arms (`flex_build_items`,
    /// `layout_flex_wrap::wrap_place_column`) — nested measurements
    /// (the table/grid arms) hoist their own pair per container.
    /// Every use clears the buffers on entry and runs to completion
    /// before the next begins (sequential, never interleaved or
    /// re-entrant), so one pair per layout pass is enough.
    int_bufs: WalkBufs,
    /// Second pooled pair for the nested measurements inside
    /// [`preferred_width_mcw`] (its F7 `width: min-content` arms) and
    /// the `content_based_width` / `measured_minmax_width` paths: the
    /// outer pair stays borrowed by the in-progress walk, so nested
    /// walks take their own. Same sequential, never-interleaved
    /// discipline as `int_bufs`.
    int_inner: WalkBufs,
    /// Staged `Task::Block`s for column flex-item measurement.
    measure_tasks: Vec<Task>,
    /// Staged `Task::Block`s for flex-item layout.
    child_tasks: Vec<Task>,
    /// Staged `Task::Block`s for stretched flex-item relayout.
    relayout: Vec<Task>,
    /// `flex_lengths`: per-item frozen flags.
    frozen: Vec<bool>,
    /// `flex_lengths`: unfrozen item indices, ascending.
    active: Vec<usize>,
    /// `flex_lengths`: frozen item indices.
    frozen_idx: Vec<usize>,
    /// `flex_lengths`: per-item violation magnitudes.
    violations: Vec<f32>,
    /// `flex_build_items`: staged flex items. Taken per container and
    /// restored when the container's [`FlexCtx`] completes in
    /// `flex_resume` (capacity survives across containers). The
    /// estimate pass for `inline-flex` restores it before the real
    /// pass re-takes it; a re-driven container's nested pass takes a
    /// different (or fresh) buffer while the outer context still owns
    /// its own — the buffer is never live in two places at once.
    flex_items: Vec<FlexItem>,
    /// `(document_element, body)` DOM ids for a full-document pass
    /// (`(None, None)` otherwise): boxes built from these nodes never
    /// take the `overflow-y: auto` scrollbar-width re-drive in
    /// `resume_block` — the app never paints a nested scrollbar for
    /// them (they are the viewport scroller), so their vertical
    /// scrollbar geometry belongs to the viewport pass. Set by
    /// `layout_forest_inner` right after construction.
    pub viewport_scroller_doms: (Option<NodeId>, Option<NodeId>),
    /// Pooled table-grid buffers for [`layout_table::collect_table_grid_into`]:
    /// one collection per table per pass after warmup allocates nothing.
    /// Checked out per table and returned when done; a nested table
    /// finds the pool empty and allocates a fresh grid instead.
    pub(crate) table_grid: layout_table::TableGridPool,
}

impl LayoutScratch {
    pub fn new() -> Self {
        LayoutScratch {
            frags: Vec::new(),
            frag_ranges: Vec::new(),
            walk: Vec::new(),
            open_ranges: Vec::new(),
            inline_abs: Vec::new(),
            lines: Vec::new(),
            frag_force_before: Vec::new(),
            frag_force_after: Vec::new(),
            para_dirs: Vec::new(),
            frag_para: Vec::new(),
            ib_out: Vec::new(),
            ib_stack: Vec::new(),
            mcw: Vec::new(),
            int_bufs: WalkBufs::new(),
            int_inner: WalkBufs::new(),
            measure_tasks: Vec::new(),
            child_tasks: Vec::new(),
            relayout: Vec::new(),
            frozen: Vec::new(),
            active: Vec::new(),
            frozen_idx: Vec::new(),
            violations: Vec::new(),
            flex_items: Vec::new(),
            viewport_scroller_doms: (None, None),
            // `collect_table_grid_into`: one pooled grid serves every
            // collection in the pass (intrinsic pre-passes, caption
            // widths, the block width branch, `layout_table`). Checked
            // out per table and returned when done; a nested table
            // finds it empty and allocates its own.
            table_grid: layout_table::TableGridPool::new(),
        }
    }
}

/// Split a used line height across the baseline by the face's real
/// ascent/descent ratio ([`FontMetrics::ascent_descent`]), so the baseline
/// sits at the typographic position instead of the historical fixed
/// First family name of the computed `font-family` value. Mirrors
/// `bowser_paint::render::css_font_family`, duplicated here because
/// layout must not depend on paint (crate order is css -> layout ->
/// paint).
pub(crate) fn css_font_family_name(style: &ComputedStyle) -> &str {
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

/// Used font family for the box at `idx`, interned through `fonts`.
/// CSS `font-family` is inherited: the winner is the nearest
/// ancestor-or-self with a computed style (text and anonymous boxes
/// carry none of their own). Resolved once per text run — the
/// per-word/fragment [`FontMetrics`] calls below then pay only the
/// id-keyed lookup, never a string hash per call.
pub(crate) fn family_id_for(forest: &Forest, idx: usize, fonts: &dyn FontMetrics) -> FontFamilyId {
    let mut cur = Some(idx);
    while let Some(i) = cur {
        let n = &forest.nodes[i];
        if let Some(style) = n.style.as_deref() {
            return fonts.intern_family(css_font_family_name(style));
        }
        cur = n.parent;
    }
    FontFamilyId::DEFAULT
}

/// Split `line_height` across the baseline by the `family` face's
/// ascent/descent ratio (see [`FontMetrics::ascent_descent`]).
/// Returns `(above, below)`; the total stays `line_height`, so the
/// strut and single-font lines keep their exact height and only the
/// baseline's position inside the line box moves. With the default
/// [`FontMetrics::ascent_descent`] this reproduces the old 0.8/0.2
/// split exactly.
fn split_line_height(
    fonts: &dyn FontMetrics,
    font_size: f32,
    line_height: f32,
    family: FontFamilyId,
) -> (f32, f32) {
    let (a, d) = fonts.ascent_descent(font_size, family);
    // CSS 2.1 §10.8.1: the font's em box (a+d) is centered in the used
    // line height; the half-leading goes above the ascent and below the
    // descent. The baseline sits at half-leading + ascent from the line
    // top — NOT at line_height * a/(a+d) (which misplaces the baseline
    // whenever line-height != font-size, e.g. 25px/2 Ahem puts it at 40
    // instead of 32.5, breaking baseline-td alignment).
    let half_leading = (line_height - (a + d)) / 2.0;
    (half_leading + a, half_leading + d)
}

/// Vertically align one line's fragments and return the line-box
/// extents relative to the baseline. Pass 1 anchors baseline fragments
/// relative to baseline 0 (`vertical-align: <length>` shifts the
/// fragment: positive lengths raise it, CSS 2.1 §10.8.1); pass 2 aligns
/// top/bottom/middle against the pass-1 extents. The strut (the block's
/// own font as an empty inline box, split by its real ascent/descent —
/// see [`split_line_height`]) participates in every line box, flooring
/// the line-box height and anchoring the baseline that table
/// cell-baseline alignment compares across cells. Idempotent: re-running
/// recomputes the same values.
fn align_line(frags: &mut [Frag], strut_top: f32, strut_bottom: f32, x_height: f32) -> (f32, f32) {
    let mut top = strut_top;
    let mut bottom = strut_bottom;
    for f in frags.iter_mut() {
        let shift = match f.valign {
            VAlign::Baseline => -f.vshift,
            VAlign::Sub => 0.15 * f.lh,
            VAlign::Super => -0.35 * f.lh,
            VAlign::Top | VAlign::Bottom | VAlign::Middle => continue,
        };
        f.rt = -f.ascent + shift;
        f.rb = f.descent + shift;
        f.aligned = true;
        top = top.min(f.rt);
        bottom = bottom.max(f.rb);
    }
    for f in frags.iter_mut() {
        if f.aligned {
            continue;
        }
        let h = f.ascent + f.descent;
        let (t, b) = match f.valign {
            VAlign::Top => (top, top + h),
            VAlign::Bottom => (bottom - h, bottom),
            // CSS 2.1 §10.8.1: the vertical midpoint of the box aligns
            // with the parent's baseline plus half the parent's
            // x-height — i.e. half the x-height *above* the baseline
            // (baseline-relative coordinates grow downward).
            VAlign::Middle => {
                let m = -x_height / 2.0;
                (m - h / 2.0, m + h / 2.0)
            }
            _ => unreachable!("pass 1 handled the rest"),
        };
        f.rt = t;
        f.rb = b;
        f.aligned = true;
    }
    // Recompute the extents over the strut plus all fragments.
    top = strut_top;
    bottom = strut_bottom;
    for f in frags.iter() {
        top = top.min(f.rt);
        bottom = bottom.max(f.rb);
    }
    (top, bottom)
}

/// True when a run with this computed `white-space` collapses and trims
/// spaces at line boundaries (only `normal` and `nowrap`; `pre`,
/// `pre-wrap`, `pre-line`, and `break-spaces` preserve them).
fn ws_trims(ws: WhiteSpace) -> bool {
    matches!(ws, WhiteSpace::Normal | WhiteSpace::NoWrap)
}

/// True when a run with this computed `white-space` may soft-wrap.
/// `pre`, `pre-wrap`/`pre-line`/`break-spaces` (pre-fit pieces) and
/// `nowrap` runs never break mid-run; only `normal` runs wrap at spaces.
fn ws_wraps(ws: WhiteSpace) -> bool {
    ws == WhiteSpace::Normal
}

/// Parameters for emitting one text node's fragments.
struct TextFragParams<'a> {
    text: &'a str,
    ws: WhiteSpace,
    fs: f32,
    lh: f32,
    /// The text run's used font family, interned once per run (see
    /// [`family_id_for`]).
    family: FontFamilyId,
    valign: VAlign,
    vshift: f32,
    /// Pre-fit width for `pre-wrap`/`pre-line`/`break-spaces` (the content
    /// width the pieces are wrapped to); must be `Some` when `ws` is
    /// `PreWrap`, `PreLine`, or `BreakSpaces`, ignored otherwise.
    max_width: Option<f32>,
}

/// Split a text node's content into fragments. `normal`/`nowrap` split on
/// (collapsed) spaces and on U+200B (zero-width space, a wrapping-point
/// marker per CSS text-3 §5 that contributes no advance); `pre` keeps runs and splits on newlines (each a
/// forced break); `pre-wrap` goes through [`wrap_pre_text`], so spaces
/// are preserved verbatim, `\n` forces breaks, and soft wraps land on
/// spaces (one fragment per piece, pre-fit to `max_width`); `pre-line`
/// goes through [`wrap_pre_line_text`], which is the same pre-fit
/// machinery but with spaces/tabs collapsed and line-boundary spaces
/// trimmed (one fragment per piece, pre-fit to `max_width`);
/// `break-spaces` goes through [`wrap_pre_text`] exactly like `pre-wrap`
/// (every space preserved and measured, each its own wrap opportunity).
fn emit_text_frags(frags: &mut Vec<Frag>, p: TextFragParams<'_>, fonts: &dyn FontMetrics) {
    // Baseline extents come from the used face's real ascent/descent,
    // ratio-split across the used line height (see `split_line_height`):
    // the line box keeps the full line height, but the baseline sits at
    // the face's typographic position. The fixed 0.8/0.2 probe split
    // survives only as the `FontMetrics::ascent_descent` default, so
    // existing implementors see no change.
    let (ascent, descent) = split_line_height(fonts, p.fs, p.lh, p.family);
    let mk = |w: f32,
              is_space: bool,
              break_before: bool,
              forced_break_before: bool,
              text_range: Option<Range<usize>>| Frag {
        w,
        ascent,
        descent,
        lh: p.lh,
        valign: p.valign,
        vshift: p.vshift,
        ib: None,
        is_space,
        break_before,
        forced_break_before,
        ws: p.ws,
        text_range,
        placed: false,
        aligned: false,
        x: 0.0,
        rt: 0.0,
        rb: 0.0,
        abs_top: 0.0,
    };
    if matches!(
        p.ws,
        WhiteSpace::PreWrap | WhiteSpace::PreLine | WhiteSpace::BreakSpaces
    ) {
        let max_width = p.max_width.unwrap_or(f32::INFINITY);
        let base = p.text.as_ptr() as usize;
        let pieces = if p.ws == WhiteSpace::PreLine {
            wrap_pre_line_text(p.text, max_width, &|s| fonts.advance(s, p.fs, p.family))
        } else {
            wrap_pre_text(p.text, max_width, &|s| fonts.advance(s, p.fs, p.family))
        };
        for piece in pieces {
            // Every piece borrows from `p.text` (see `wrap_pre_text`), so
            // the pointer difference is the piece's byte offset into it.
            let start = piece.text.as_ptr() as usize - base;
            debug_assert!(start + piece.text.len() <= p.text.len());
            frags.push(mk(
                piece.width,
                false,
                piece.break_before,
                piece.forced_break,
                Some(start..start + piece.text.len()),
            ));
        }
        return;
    }
    if p.ws == WhiteSpace::Pre {
        let mut off = 0;
        for (k, piece) in p.text.split('\n').enumerate() {
            let end = off + piece.len();
            // In `pre` every break is a `\n` break: a line break AND
            // a paragraph separator.
            frags.push(mk(
                fonts.advance(piece, p.fs, p.family),
                false,
                k > 0,
                k > 0,
                Some(off..end),
            ));
            off = end + 1; // past the '\n'
        }
        return;
    }
    let space_w = fonts.advance(" ", p.fs, p.family);
    let mut off = 0;
    for (k, part) in p.text.split(' ').enumerate() {
        if k > 0 {
            frags.push(mk(space_w, true, false, false, Some(off..off + 1)));
            off += 1;
        }
        // U+200B is a break opportunity with zero advance (CSS text-3 §5):
        // split the word around it so the line breaker can wrap there,
        // emitting the marker itself as a zero-width, invisible fragment.
        let mut woff = off;
        for (zk, word) in part.split('\u{200b}').enumerate() {
            if zk > 0 {
                frags.push(mk(
                    0.0,
                    false,
                    false,
                    false,
                    Some(woff..woff + '\u{200b}'.len_utf8()),
                ));
                woff += '\u{200b}'.len_utf8();
            }
            if !word.is_empty() {
                let end = woff + word.len();
                frags.push(mk(
                    fonts.advance(word, p.fs, p.family),
                    false,
                    false,
                    false,
                    Some(woff..end),
                ));
                woff = end;
            }
        }
        off = woff;
    }
}

/// Collapse adjacent collapsible spaces across text-node/inline
/// boundaries (CSS text-3 §4.1.2). Each text node's text is already
/// collapsed internally (`boxes::collapse_ws`); only the seam between two
/// nodes can hold two collapsible spaces. The duplicate is zeroed rather
/// than removed so fragment ranges (which index by position) stay valid;
/// a zero-width non-space fragment is layout- and paint-inert.
fn collapse_adjacent_spaces(frags: &mut [Frag]) {
    let collapsible = |f: &Frag| {
        f.is_space
            && !f.break_before
            && matches!(
                f.ws,
                WhiteSpace::Normal | WhiteSpace::NoWrap | WhiteSpace::PreLine
            )
    };
    let mut prev_space = false;
    for f in frags.iter_mut() {
        let cur = collapsible(f);
        if cur && prev_space {
            f.w = 0.0;
            f.is_space = false;
            prev_space = false;
        } else {
            prev_space = cur;
        }
    }
}

/// Collect inline-block descendants (not descending into nested
/// inline-blocks; those are handled by their own layout). Out-of-flow
/// subtrees (`position: absolute | fixed`) are skipped wholesale: they
/// take no space in the line and are laid out by the `layout_position`
/// post-pass.
fn collect_inline_blocks(forest: &Forest, idx: usize, scratch: &mut LayoutScratch) {
    // Buffers are cleared, never dropped: capacity survives across
    // containers.
    let out = &mut scratch.ib_out;
    out.clear();
    let stack = &mut scratch.ib_stack;
    stack.clear();
    stack.extend(forest.nodes[idx].children.iter().copied());
    while let Some(i) = stack.pop() {
        if layout_position::is_out_of_flow(forest, i) {
            continue;
        }
        match forest.nodes[i].box_type {
            BoxType::InlineBlock => out.push(i),
            BoxType::Inline => stack.extend(forest.nodes[i].children.iter().copied()),
            _ => {}
        }
    }
}

/// Record a deferred shift for a subtree root: O(1), no tree walk.
///
/// Inline layout's inline-block placement (step 4 of `do_inline_layout`)
/// used to walk and rewrite the whole subtree per placed fragment;
/// deeply nested inline-blocks re-shifted shared subtrees once per
/// nesting level (quadratic). Now the delta is stored on
/// the subtree root and applied once, bottom-up, by `apply_pending_shifts`
/// at the end of `layout_forest`, so every node moves exactly once.
///
/// The delta is always measured against the *provisional* rect (rects are
/// never mutated before the final pass), so recording *sets* the pending
/// offset rather than accumulating it. Rects stay provisional in between,
/// which is exactly what the placement code after it expects.
fn record_shift(forest: &mut Forest, idx: usize, dx: f32, dy: f32) {
    let n = &mut forest.nodes[idx];
    n.pend_x = dx;
    n.pend_y = dy;
}

/// Drop deferred shifts for a subtree about to be re-laid out (flex
/// `align-self: stretch` re-runs `Task::Block` on the item).
///
/// Re-layout recomputes provisional rects and re-records fresh shifts;
/// stale entries would apply twice, so they are zeroed up front. One
/// walk per re-laid-out item; items are disjoint, so O(nodes) total.
fn clear_pending_shifts(forest: &mut Forest, idx: usize) {
    let mut stack = vec![idx];
    while let Some(i) = stack.pop() {
        let n = &mut forest.nodes[i];
        n.pend_x = 0.0;
        n.pend_y = 0.0;
        stack.extend(n.children.iter().copied());
    }
}

/// Apply every deferred shift in a single pass, pushing accumulated
/// offsets down the tree.
///
/// Each node's final position is its provisional rect plus the sum of the
/// pending offsets on the path from the layout root — exactly what the old
/// per-fragment eager subtree walks produced, but every node is
/// visited once, so nested inline-blocks cost O(nodes) total instead of
/// O(depth * nodes).
///
/// Fixed-position subtrees are the exception: they are laid out against
/// the viewport, whose origin never moves, so in-flow ancestor shifts
/// (relative offsets, inline-block re-anchoring) must not accumulate
/// into them. Accumulation restarts at each fixed box (its own pending
/// shift is still applied); descendants of the fixed box accumulate from
/// there as usual.
///
/// A fixed box under a fixed-CB trigger is the exception to the
/// exception: its containing block is that ancestor (CSS Transforms 1
/// §3 et al.), and it was positioned against that ancestor's
/// *provisional* rect, so it must ride the ancestor's final shift. The
/// walk carries the applied shift of the nearest such ancestor down
/// (the same carried-frame pattern `scroll_containers_with_offset`
/// uses); a fixed box restarts accumulation from the carried shift
/// instead of zero, which is exactly the no-op `(0, 0)` carry for
/// viewport-anchored fixed boxes.
fn apply_pending_shifts(forest: &mut Forest, root: usize) {
    let mut stack = vec![(root, 0.0f32, 0.0f32, 0.0f32, 0.0f32)];
    while let Some((i, ax, ay, cx, cy)) = stack.pop() {
        // Index iteration instead of `children.clone()`: the body only
        // mutates rect/shift fields (and fragments), never
        // `nodes[i].children`, so reading each child by index is the same
        // snapshot without the O(kids) allocation.
        let nkids = forest.nodes[i].children.len();
        let (nx, ny, ncx, ncy) = {
            let n = &mut forest.nodes[i];
            let (ax, ay) = if matches!(n.data.position, Position::Fixed) {
                (cx + n.pend_x, cy + n.pend_y)
            } else {
                (ax + n.pend_x, ay + n.pend_y)
            };
            n.pend_x = 0.0;
            n.pend_y = 0.0;
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
            (ax, ay, ncx, ncy)
        };
        for k in 0..nkids {
            let c = forest.nodes[i].children[k];
            stack.push((c, nx, ny, ncx, ncy));
        }
    }
}

/// `vertical-align: <length>` as a pixel shift. Percentages resolve against
/// the line height; keyword values (and a missing property) give 0.0.
/// Positive lengths raise the box (CSS 2.1 §10.8.1).
fn valign_len_shift(
    style: Option<&ComputedStyle>,
    font_size: f32,
    line_height: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
) -> f32 {
    spec_len_of(style.and_then(|s| s.get_by_index(props::VERTICAL_ALIGN)))
        .resolve(font_size, root_fs, vp_w, vp_h, line_height)
        .unwrap_or(0.0)
}

/// The `vertical-align: <length>` shift for a text fragment, read from the
/// immediate inline parent's specified style. (`vertical-align` is not
/// inherited; `boxes` copies only the immediate parent's keyword into the
/// text box's `StyleData`.)
fn text_frag_vshift(forest: &Forest, text_idx: usize) -> f32 {
    match forest.nodes[text_idx].parent {
        Some(p) if forest.nodes[p].box_type == BoxType::Inline => {
            let n = &forest.nodes[p];
            valign_len_shift(
                n.style.as_deref(),
                n.data.font_size,
                n.data.line_height,
                forest.root_font_size,
                forest.vp_w,
                forest.vp_h,
            )
        }
        _ => 0.0,
    }
}

/// Whether the block's inline content is right-to-left: the resolved
/// `direction`, inherited during box construction (see `StyleData::rtl`),
/// so this is O(1) instead of an ancestor walk per inline container.
fn is_rtl(forest: &Forest, idx: usize) -> bool {
    forest.nodes[idx].data.rtl
}

/// Whether `c` has a strong right-to-left bidi class (UAX #9 types
/// R / AL). This is an explicit bidi-class range table — every range
/// below is a maximal run of assigned codepoints whose Unicode
/// `Bidi_Class` is R or AL (Unicode 15.0 `DerivedBidiClass`) — NOT a
/// set of script-block approximations: the weak/neutral slots inside
/// the RTL script blocks (EN/AN digits, ES/CS/ET punctuation, NSM
/// marks, BN controls) are deliberately absent, so e.g. the Arabic
/// block contributes 061B..=064A and 0671..=06D5 but not the
/// Arabic-Indic digits (0660..=0669, AN) between them. Callers gate
/// on `char::is_alphabetic`, so the few non-letter strong slots that
/// remain (e.g. Hebrew punctuation U+05BE, class R) never classify
/// either.
///
/// Coverage, by script/family: Hebrew (incl. presentation forms),
/// Arabic (incl. Supplements A/B, Extended-A/B, presentation forms-A/B
/// and the mathematical alphabetic symbols), Syriac (incl.
/// Supplement), Thaana, Nko, Samaritan, Mandaic, and the historic
/// RTL scripts — Imperial Aramaic, Palmyrene, Nabataean, Hatran,
/// Phoenician, Lydian, Meroitic Hieroglyphs/Cursive, Kharoshthi,
/// Old Persian, Old Turkic, Old Hungarian, Old Arabic, South Arabian,
/// Manichaean, Avestan, Pahlavi (Inscriptional/Parthian/Psalter),
/// Mahajani, Palmyrene-era and later additions (Elymaic, Chorasmian,
/// Yezidi, Old Sogdian, Old Uyghur, Hanifi Rohingya), Mende Kikakui
/// and Adlam. Mongolian is intentionally absent: its bidi class is
/// L, not R/AL.
fn is_strong_rtl_char(c: char) -> bool {
    matches!(c as u32,
        0x05BE | 0x05C0 | 0x05C3 | 0x05C6
        | 0x05D0..=0x05EA | 0x05EF..=0x05F4 | 0x0608 | 0x060B
        | 0x060D | 0x061B..=0x064A | 0x066D..=0x066F | 0x0671..=0x06D5
        | 0x06E5..=0x06E6 | 0x06EE..=0x06EF | 0x06FA..=0x070D | 0x070F..=0x0710
        | 0x0712..=0x072F | 0x074D..=0x07A5 | 0x07B1 | 0x07C0..=0x07EA
        | 0x07F4..=0x07F5 | 0x07FA | 0x07FE..=0x0815 | 0x081A
        | 0x0824 | 0x0828 | 0x0830..=0x083E | 0x0840..=0x0858
        | 0x085E | 0x0860..=0x086A | 0x0870..=0x088E | 0x08A0..=0x08C9
        | 0x200F | 0xFB1D | 0xFB1F..=0xFB28 | 0xFB2A..=0xFB36
        | 0xFB38..=0xFB3C | 0xFB3E | 0xFB40..=0xFB41 | 0xFB43..=0xFB44
        | 0xFB46..=0xFB4F | 0xFB50..=0xFBC2 | 0xFBD3..=0xFD3D | 0xFD50..=0xFD8F
        | 0xFD92..=0xFDC7 | 0xFDF0..=0xFDFC | 0xFE70..=0xFE74 | 0xFE76..=0xFEFC
        | 0x10800..=0x10805 | 0x10808 | 0x1080A..=0x10835 | 0x10837..=0x10838
        | 0x1083C | 0x1083F..=0x10855 | 0x10857..=0x1089E | 0x108A7..=0x108AF
        | 0x108E0..=0x108F2 | 0x108F4..=0x108F5 | 0x108FB..=0x1091B | 0x10920..=0x10939
        | 0x1093F | 0x10980..=0x109B7 | 0x109BC..=0x109CF | 0x109D2..=0x10A00
        | 0x10A10..=0x10A13 | 0x10A15..=0x10A17 | 0x10A19..=0x10A35 | 0x10A40..=0x10A48
        | 0x10A50..=0x10A58 | 0x10A60..=0x10A9F | 0x10AC0..=0x10AE4 | 0x10AEB..=0x10AF6
        | 0x10B00..=0x10B35 | 0x10B40..=0x10B55 | 0x10B58..=0x10B72 | 0x10B78..=0x10B91
        | 0x10B99..=0x10B9C | 0x10BA9..=0x10BAF | 0x10C00..=0x10C48 | 0x10C80..=0x10CB2
        | 0x10CC0..=0x10CF2 | 0x10CFA..=0x10CFF | 0x10D00..=0x10D23 | 0x10E80..=0x10EA9
        | 0x10EAD | 0x10EB0..=0x10EB1 | 0x10F00..=0x10F27 | 0x10F30..=0x10F45
        | 0x10F51..=0x10F59 | 0x10F70..=0x10F81 | 0x10F86..=0x10F89 | 0x10FB0..=0x10FCB
        | 0x10FE0..=0x10FF6 | 0x1E800..=0x1E8C4 | 0x1E8C7..=0x1E8CF | 0x1E900..=0x1E943
        | 0x1E94B | 0x1E950..=0x1E959 | 0x1E95E..=0x1E95F | 0x1EC71..=0x1ECB4
        | 0x1ED01..=0x1ED3D | 0x1EE00..=0x1EE03 | 0x1EE05..=0x1EE1F | 0x1EE21..=0x1EE22
        | 0x1EE24 | 0x1EE27 | 0x1EE29..=0x1EE32 | 0x1EE34..=0x1EE37
        | 0x1EE39 | 0x1EE3B | 0x1EE42 | 0x1EE47
        | 0x1EE49 | 0x1EE4B | 0x1EE4D..=0x1EE4F | 0x1EE51..=0x1EE52
        | 0x1EE54 | 0x1EE57 | 0x1EE59 | 0x1EE5B
        | 0x1EE5D | 0x1EE5F | 0x1EE61..=0x1EE62 | 0x1EE64
        | 0x1EE67..=0x1EE6A | 0x1EE6C..=0x1EE72 | 0x1EE74..=0x1EE77 | 0x1EE79..=0x1EE7C
        | 0x1EE7E | 0x1EE80..=0x1EE89 | 0x1EE8B..=0x1EE9B | 0x1EEA1..=0x1EEA3
        | 0x1EEA5..=0x1EEA9 | 0x1EEAB..=0x1EEBB)
}

/// The base direction of a `unicode-bidi: plaintext` paragraph from
/// its first strong character (UAX #9 P2/P3): `Some(true)` when the
/// first strong character is R/AL, `Some(false)` when it is strong
/// L, and `None` when the text has no strong character at all
/// (leading digits, punctuation and spaces are weak/neutral and are
/// skipped — the first strong character after them decides).
///
/// The strong format characters count even though they are not
/// alphabetic (UAX #9 P2 scans by bidi type, not by letter-ness):
/// U+200E LEFT-TO-RIGHT MARK is strong L and terminates the scan as
/// LTR, U+200F RIGHT-TO-LEFT MARK is strong R, and U+061C ARABIC
/// LETTER MARK is strong AL. They are matched explicitly BEFORE the
/// alphabetic check, which would otherwise skip them (they are
/// `Cf`, not letters) and let a later opposite-direction letter
/// decide instead. Digits (EN/AN) and NSM marks are not alphabetic
/// and carry no strong class here, so they stay skipped.
fn first_strong_rtl(text: &str) -> Option<bool> {
    for c in text.chars() {
        match c {
            // LRM: strong L.
            '\u{200E}' => return Some(false),
            // RLM: strong R. ALM: strong AL.
            '\u{200F}' | '\u{061C}' => return Some(true),
            _ => {}
        }
        if c.is_alphabetic() {
            return Some(is_strong_rtl_char(c));
        }
    }
    None
}

/// `text-align: start | end` resolve against `direction`; every other value
/// uses the parsed `StyleData` value.
fn effective_text_align(forest: &Forest, idx: usize) -> TextAlign {
    effective_text_align_with_rtl(forest, idx, is_rtl(forest, idx))
}

/// [`effective_text_align`] against an explicit base direction instead
/// of the box's resolved `direction`: the seam `unicode-bidi:
/// plaintext` feeds each paragraph's first-strong direction through.
fn effective_text_align_with_rtl(forest: &Forest, idx: usize, rtl: bool) -> TextAlign {
    // Anonymous wrappers carry no style, so the raw `start` / `end`
    // keyword is read from the nearest styled ancestor — the block
    // container whose inline content the wrapper holds (the same
    // climb `text_indent_px` makes). `StyleData::text_align` cannot
    // substitute: it folds `start` to a physical side at parse time.
    // Without the climb, an anonymous block under
    // `text-align: start` + RTL (by `direction` or by a plaintext
    // paragraph direction) would silently align left.
    let raw = {
        let mut cur = idx;
        loop {
            let n = &forest.nodes[cur];
            if let Some(s) = &n.style {
                break s.get_str_by_index(props::TEXT_ALIGN);
            }
            match n.parent {
                Some(p) if n.box_type == BoxType::AnonymousBlock => cur = p,
                _ => break None,
            }
        }
    };
    match raw {
        Some(v) if v.eq_ignore_ascii_case("start") => {
            if rtl {
                TextAlign::Right
            } else {
                TextAlign::Left
            }
        }
        Some(v) if v.eq_ignore_ascii_case("end") => {
            if rtl {
                TextAlign::Left
            } else {
                TextAlign::Right
            }
        }
        _ => forest.nodes[idx].data.text_align,
    }
}

/// `text-indent` in px, applied to the first line box. Read from the
/// block's own specified style; anonymous wrappers carry no style, so one
/// holding the block's first line inherits it from the parent.
/// Percentages resolve against the content width.
fn text_indent_px(forest: &Forest, idx: usize, content_w: f32) -> f32 {
    let mut cur = idx;
    loop {
        let n = &forest.nodes[cur];
        let indent = n
            .style
            .as_ref()
            .and_then(|s| s.get_by_index(props::TEXT_INDENT));
        if indent.is_some() {
            return spec_len_of(indent)
                .resolve(
                    n.data.font_size,
                    forest.root_font_size,
                    forest.vp_w,
                    forest.vp_h,
                    content_w,
                )
                .unwrap_or(0.0);
        }
        let first_line_anon = n.box_type == BoxType::AnonymousBlock
            && n.parent.is_some_and(|p| {
                // The anonymous wrapper holds the block's first line only
                // when no earlier sibling generates line boxes of its own:
                // every earlier sibling must be out of flow (floated or
                // absolutely/fixed positioned). A leading float must not
                // block the inheritance (crbug 1014247 scenario: the float
                // precedes the anonymous wrapper holding the first line).
                let children = &forest.nodes[p].children;
                let pos = children
                    .iter()
                    .position(|c| *c == cur)
                    .unwrap_or(children.len());
                children[..pos].iter().all(|s| {
                    let d = &forest.nodes[*s].data;
                    !matches!(d.float, Float::None)
                        || matches!(d.position, Position::Absolute | Position::Fixed)
                })
            });
        match n.parent {
            Some(p) if first_line_anon => cur = p,
            _ => return 0.0,
        }
    }
}

/// Bound on the per-line break/re-test iterations in [`do_inline_layout`].
/// Each pass either shifts the line below at least one float (finitely many
/// floats) or re-breaks against a strictly narrower band (finitely many
/// fragments can move to the next line), so the loop always settles; the
/// cap is a pure backstop.
const MAX_LINE_RETEST: u32 = 32;

/// Output of [`do_inline_layout`]: the inline content's natural height
/// plus the inline-level forced-break line indices for the multicol
/// distribution pass.
pub(crate) struct InlineLayoutOutput {
    /// Natural content height: post-distribution (tallest column bottom
    /// minus `content_top`) for multicol containers, else the stacked
    /// line height.
    pub height: f32,
    /// Line indices (into the step-2 `Line` list, strictly ascending)
    /// that must start a fresh column: inline-level forced
    /// `break-before` / `break-after` breaks. Per CSS Fragmentation
    /// §5.2.1 a forced break on an inline-level box breaks the line box
    /// at that point; the line box after the break starts a new column.
    /// Recorded only for multicol containers (inline-level forced breaks
    /// are honored only where a column can start); empty otherwise. A
    /// forced break at the very start of the content is a no-op and is
    /// not recorded (line 0 already begins a fresh column — no empty
    /// column is created).
    ///
    /// Handoff contract for the multicol pass:
    /// `layout_multicol::distribute_inline_lines` consumes these indices
    /// and starts a fresh column at each of them.
    // (dead_code: the field is written by the inline layout pass and read
    // by `distribute_inline_lines`; unit tests also read it directly.)
    #[allow(dead_code)]
    pub forced_break_lines: Vec<usize>,
}

/// Push the fragment for an HTML `<br>` element: a forced line break
/// (HTML rendering, "the `br` element"). The fragment is zero-width and
/// carries the `<br>`'s own font metrics, so a `<br>` in a larger font
/// stretches its line the way a text fragment would; it renders no text
/// (`text_range` is `None`) and takes no space (`is_space` is false, so
/// it is never trimmed or collapsed).
///
/// The break lands *after* the `<br>`: `break_before` consumes a pending
/// break armed by an earlier `<br>` (so consecutive `<br>`s stack empty
/// lines), and the caller arms the pending break for the next fragment
/// after pushing. A pending break still armed when the walk ends (a
/// trailing `<br>`) gets a final empty fragment from
/// [`push_trailing_br_frag`], so the break produces its empty line; the
/// empty line's height comes from the block strut, as usual.
fn push_br_frag(
    forest: &Forest,
    frags: &mut Vec<Frag>,
    box_idx: usize,
    break_before: bool,
    fonts: &dyn FontMetrics,
) {
    let n = &forest.nodes[box_idx];
    let (ascent, descent) = split_line_height(
        fonts,
        n.data.font_size,
        n.data.line_height,
        family_id_for(forest, box_idx, fonts),
    );
    frags.push(Frag {
        w: 0.0,
        ascent,
        descent,
        lh: n.data.line_height,
        valign: n.data.valign,
        vshift: 0.0,
        ib: None,
        is_space: false,
        break_before,
        forced_break_before: false,
        ws: n.data.white_space,
        text_range: None,
        placed: false,
        aligned: false,
        x: 0.0,
        rt: 0.0,
        rb: 0.0,
        abs_top: 0.0,
    });
}

/// The empty fragment that realizes a trailing `<br>`'s break: with no
/// content after the `<br>` the line breaker would otherwise end the
/// block at the `<br>`'s line. Zero metrics — the empty line is sized by
/// the block strut alone.
fn push_trailing_br_frag(frags: &mut Vec<Frag>) {
    frags.push(Frag {
        w: 0.0,
        ascent: 0.0,
        descent: 0.0,
        lh: 0.0,
        valign: VAlign::Baseline,
        vshift: 0.0,
        ib: None,
        is_space: false,
        break_before: true,
        forced_break_before: false,
        ws: WhiteSpace::Normal,
        text_range: None,
        placed: false,
        aligned: false,
        x: 0.0,
        rt: 0.0,
        rb: 0.0,
        abs_top: 0.0,
    });
}

#[allow(clippy::too_many_arguments)]
/// Lay out the inline content of `idx`.
fn do_inline_layout(
    forest: &mut Forest,
    idx: usize,
    content_x: f32,
    content_top: f32,
    content_w: f32,
    floats: &[FloatEntry],
    fonts: &dyn FontMetrics,
    scratch: &mut LayoutScratch,
    multicol: Option<&layout_multicol::MulticolParams>,
    multicol_hspec: Option<f32>,
) -> InlineLayoutOutput {
    let block_lh = {
        let d = &forest.nodes[idx].data;
        d.line_height
    };
    let text_align = effective_text_align(forest, idx);
    let indent = text_indent_px(forest, idx, content_w);
    // Exact non-zero check for the first-line indent, used by the
    // float fit sites below: `!= 0.0` is NaN-unsafe (`NaN != 0.0` is
    // true), so spell it out. (clippy::double_comparisons would fold
    // this back into `!= 0.0`; the explicit form is intentional.)
    #[allow(clippy::double_comparisons)]
    let indent_nonzero = indent > 0.0 || indent < 0.0;

    // 1. Build fragments, recording each inline/text box's fragment range.
    //    Ranges nest: a box's fragments are exactly its subtree's, so each
    //    range links to its directly nested child ranges. Step 5 then
    //    unions them bottom-up in a single pass instead of re-scanning the
    //    fragment list once per box (which was O(fragments x depth)).
    // Scratch buffers are cleared, never dropped: capacity survives
    // across containers. (`FragRange`/`InlineWalkItem` live at module
    // level so the struct can own these buffers.)
    let frags = &mut scratch.frags;
    frags.clear();
    let ranges = &mut scratch.frag_ranges;
    ranges.clear();
    let stack = &mut scratch.walk;
    stack.clear();
    stack.extend(
        forest.nodes[idx]
            .children
            .iter()
            .rev()
            .map(|&c| InlineWalkItem::Enter(c)),
    );
    // Range indices of inline boxes whose `Enter` was seen but whose `Exit`
    // has not: the innermost entry is the enclosing box of any range
    // pushed now. A range is pushed at `Enter`, so a child's index is
    // always larger than its parent's.
    let open_ranges = &mut scratch.open_ranges;
    open_ranges.clear();
    // Out-of-flow boxes (`position: absolute | fixed`) met during the
    // walk, with the index of the fragment emitted just before each (for
    // static-position resolution after line breaking). They emit no
    // fragments and take no space in the line.
    let inline_abs = &mut scratch.inline_abs;
    inline_abs.clear();
    // A `<br>` arms the forced break for the *next* fragment (the break
    // lands after the `<br>`); the next fragment pushed consumes it via
    // `break_before`. Still armed at the end of the walk, a trailing
    // `<br>` gets its empty line from `push_trailing_br_frag`.
    let mut pending_br_break = false;
    while let Some(item) = stack.pop() {
        match item {
            InlineWalkItem::Enter(c) => {
                if layout_position::is_out_of_flow(forest, c) {
                    // The whole subtree is out of flow: skip it (the
                    // `layout_position` post-pass lays it out against its
                    // containing block).
                    inline_abs.push((c, frags.len().checked_sub(1)));
                    continue;
                }
                match forest.nodes[c].box_type {
                    BoxType::Text => {
                        let start = frags.len();
                        // Borrow the text in place: the clone is unnecessary
                        // (only `frags` is mutated below, not `forest`).
                        let vshift = text_frag_vshift(forest, c);
                        let n = &forest.nodes[c];
                        emit_text_frags(
                            frags,
                            TextFragParams {
                                text: n.text.as_deref().unwrap_or(""),
                                ws: n.data.white_space,
                                fs: n.data.font_size,
                                lh: n.data.line_height,
                                family: family_id_for(forest, c, fonts),
                                valign: n.data.valign,
                                vshift,
                                max_width: if matches!(
                                    n.data.white_space,
                                    WhiteSpace::PreWrap
                                        | WhiteSpace::PreLine
                                        | WhiteSpace::BreakSpaces
                                ) {
                                    Some(content_w)
                                } else {
                                    None
                                },
                            },
                            fonts,
                        );
                        // A `<br>` is a forced line break: collapsible
                        // spaces at the start of the new line are removed
                        // (CSS text-3 §4.1.2 — zeroed, not removed, so
                        // fragment ranges stay valid, per the
                        // `collapse_adjacent_spaces` convention), and the
                        // break lands before the first surviving fragment.
                        // When the whole text run is spaces the break
                        // stays armed for whatever follows.
                        if pending_br_break {
                            let mut s = start;
                            while s < frags.len() && frags[s].is_space && ws_trims(frags[s].ws) {
                                frags[s].w = 0.0;
                                frags[s].is_space = false;
                                s += 1;
                            }
                            if s < frags.len() {
                                frags[s].break_before = true;
                                pending_br_break = false;
                            }
                        }
                        let ri = ranges.len();
                        if let Some(&p) = open_ranges.last() {
                            ranges[p].children.push(ri);
                        }
                        ranges.push(FragRange {
                            box_idx: c,
                            start,
                            end: frags.len(),
                            children: RangeChildren::new(),
                        });
                    }
                    BoxType::Inline => {
                        // `<br>`: a forced line break. It emits a single
                        // zero-width fragment (see `push_br_frag`) and no
                        // range children — it has no rendered descendants.
                        // The break lands after the `<br>`: this fragment
                        // consumes a pending break from an earlier `<br>`,
                        // and arms the break for the next fragment.
                        if forest.nodes[c].is_br {
                            let start = frags.len();
                            let bb = std::mem::replace(&mut pending_br_break, false);
                            push_br_frag(forest, frags, c, bb, fonts);
                            pending_br_break = true;
                            let ri = ranges.len();
                            if let Some(&p) = open_ranges.last() {
                                ranges[p].children.push(ri);
                            }
                            ranges.push(FragRange {
                                box_idx: c,
                                start,
                                end: frags.len(),
                                children: RangeChildren::new(),
                            });
                            continue;
                        }
                        // Inline boxes never had their metrics resolved, so
                        // their horizontal margin/border/padding were silently
                        // dropped. Resolve them here (width is meaningless for
                        // a non-replaced inline box; hand a zero content width
                        // so only the spacing resolves) and lay the spacing out
                        // around the box's first/last fragments below.
                        resolve_metrics(forest, c, 0.0, content_w, false, Some(0.0), false);
                        let start = frags.len();
                        let ri = ranges.len();
                        if let Some(&p) = open_ranges.last() {
                            ranges[p].children.push(ri);
                        }
                        ranges.push(FragRange {
                            box_idx: c,
                            start,
                            end: usize::MAX,
                            children: RangeChildren::new(),
                        });
                        open_ranges.push(ri);
                        stack.push(InlineWalkItem::Exit(c, ri));
                        stack.extend(
                            forest.nodes[c]
                                .children
                                .iter()
                                .rev()
                                .map(|&k| InlineWalkItem::Enter(k)),
                        );
                    }
                    BoxType::InlineBlock => {
                        // Already laid out at a provisional origin; atomic.
                        // Aligned by the baseline of its last line box.
                        let (w, ascent, descent, lh, va, vshift, ib_ws) = {
                            let n = &forest.nodes[c];
                            let m = &n.metrics;
                            let ascent = (n.baseline + m.mt).max(0.0);
                            let descent = (n.rect.h + m.mb - n.baseline).max(0.0);
                            let vshift = valign_len_shift(
                                n.style.as_deref(),
                                n.data.font_size,
                                n.data.line_height,
                                forest.root_font_size,
                                forest.vp_w,
                                forest.vp_h,
                            );
                            (
                                m.ml + m.bl + m.pl + m.content_w + m.pr + m.br + m.mr,
                                ascent,
                                descent,
                                n.data.line_height,
                                n.data.valign,
                                vshift,
                                n.data.white_space,
                            )
                        };
                        frags.push(Frag {
                            w,
                            ascent,
                            descent,
                            lh,
                            valign: va,
                            vshift,
                            ib: Some(c),
                            is_space: false,
                            break_before: false,
                            forced_break_before: false,
                            ws: ib_ws,
                            text_range: None,
                            placed: false,
                            aligned: false,
                            x: 0.0,
                            rt: 0.0,
                            rb: 0.0,
                            abs_top: 0.0,
                        });
                        // An inline-block after a `<br>` starts the new
                        // line (atomic: never a trimmable space).
                        if pending_br_break {
                            if let Some(f) = frags.last_mut() {
                                f.break_before = true;
                            }
                            pending_br_break = false;
                        }
                    }
                    _ => {}
                }
            }
            InlineWalkItem::Exit(c, ri) => {
                debug_assert_eq!(ranges[ri].box_idx, c);
                debug_assert_eq!(open_ranges.last(), Some(&ri));
                open_ranges.pop();
                ranges[ri].end = frags.len();
            }
        }
    }

    // A trailing `<br>` armed a break no fragment consumed: push the
    // empty fragment that realizes it, so the break produces its empty
    // (strut-height) line instead of being lost.
    if pending_br_break {
        push_trailing_br_frag(frags);
    }

    // Inline-box spacing: per CSS 2.1 §8.6 an inline box's left
    // margin/border/padding applies before its first fragment and its
    // right spacing after its last fragment (vertical spacing has no
    // layout effect). Nested boxes accumulate at shared boundaries.
    // When a boundary fragment is a trimmable space dropped during line
    // breaking, its share transfers to a surviving fragment (CSS 2.1
    // §8.6 / §16.6.1) instead of vanishing with the space: leading
    // spaces skipped at a line start hand their shares to the line's
    // first fragment, trailing spaces trimmed at a line end hand theirs
    // to the last surviving fragment (joining the line width so
    // alignment accounts for it), and the shares of a dropped all-space
    // line carry to the next line's first fragment.
    //
    // Whitespace collapsing crosses text-node/inline boundaries here,
    // after all fragments are emitted and before spacing/line breaking.
    collapse_adjacent_spaces(frags);

    // 1a. `unicode-bidi: plaintext` paragraph directions (CSS Writing
    //     Modes 3 §2.7 / UAX #9 P1-P3): when the block is in a
    //     plaintext scope (see `StyleData::bidi_plaintext`), each bidi
    //     paragraph's base direction comes from its first strong
    //     character instead of the block's `direction`. A paragraph is
    //     the inline content between paragraph separators (UAX #9 P1):
    //     segmentation splits ONLY at `Frag::forced_break_before` —
    //     the distinct flag threaded from `PreWrapPiece::forced_break`
    //     that marks `\n`-forced breaks in `pre`-family text. It must
    //     NOT split at `Frag::break_before`: pre-fit pieces set that
    //     for soft wraps too, and a soft wrap is not a paragraph
    //     boundary — a wrapped line keeps its paragraph's direction
    //     even when the line itself starts with a word of the other
    //     direction. Block boundaries are separate `do_inline_layout`
    //     calls, so each block container's paragraphs are exactly its
    //     own.
    //     Paragraphs with no strong character fall back to the block's
    //     resolved `direction`. The result is one direction per
    //     paragraph plus, per fragment, its paragraph index, so step 3
    //     can resolve `text-align: start | end` (and the line placement
    //     offset that follows from it) per paragraph — a wrapped line
    //     keeps its paragraph's direction even when its own first
    //     character is neutral or of the other direction. Empty unless
    //     plaintext applies, in which case the block direction rules
    //     every line exactly as before.
    // Pooled across `do_inline_layout` calls (capacity survives):
    // the paragraph-direction list and the per-fragment paragraph
    // indices reuse `scratch.para_dirs` / `scratch.frag_para`
    // instead of allocating per block container (`frags` itself is
    // already pooled). The reuse is safe because of the
    // task-ordering invariant, not just borrowck: both buffers are
    // cleared on entry, populated only in this block, and consumed
    // only later in this same `do_inline_layout` call (the per-line
    // direction lookup below) — no reader ever spans a call
    // boundary, so stale pooled contents are unreachable. Disjoint
    // scratch fields, so these borrows coexist with the live `frags`
    // borrow above; never re-entered (inline-block contents lay out
    // through the block path, not a nested `do_inline_layout`).
    let para_dirs: &mut Vec<bool> = &mut scratch.para_dirs;
    para_dirs.clear();
    let frag_para: &mut Vec<usize> = &mut scratch.frag_para;
    frag_para.clear();
    if forest.nodes[idx].data.bidi_plaintext {
        let block_rtl = is_rtl(forest, idx);
        // The text each fragment renders, via its text box's
        // fragment range (inline-block fragments render no text
        // here — their own block container derives its own
        // paragraphs when it is laid out).
        let mut frag_texts: Vec<Option<&str>> = vec![None; frags.len()];
        for r in ranges.iter() {
            if forest.nodes[r.box_idx].box_type != BoxType::Text {
                continue;
            }
            let text = forest.nodes[r.box_idx].text.as_deref().unwrap_or("");
            for fi in r.start..r.end.min(frags.len()) {
                if let Some(range) = frags[fi].text_range.clone() {
                    frag_texts[fi] = Some(&text[range]);
                }
            }
        }
        let mut cur: Option<bool> = None;
        let mut cur_para = 0usize;
        frag_para.resize(frags.len(), 0);
        for fi in 0..frags.len() {
            if fi > 0 && frags[fi].forced_break_before {
                para_dirs.push(cur.unwrap_or(block_rtl));
                cur = None;
                cur_para += 1;
            }
            frag_para[fi] = cur_para;
            if cur.is_none()
                && let Some(t) = frag_texts[fi]
            {
                cur = first_strong_rtl(t);
            }
        }
        para_dirs.push(cur.unwrap_or(block_rtl));
        // Not a length check: `frag_para.resize(frags.len(), 0)` above
        // fixed the length, so asserting it would be vacuous. The real
        // fill invariant is that every fragment's paragraph index
        // names a paragraph that was actually pushed.
        debug_assert!(
            frag_para.iter().all(|&p| p < para_dirs.len()),
            "every fragment must point at a pushed paragraph"
        );
        debug_assert_eq!(
            para_dirs.len(),
            cur_para + 1,
            "one base direction per paragraph"
        );
    }

    // 1b. Inline-level forced breaks (multicol only): `break-before` /
    //     `break-after` forced values (see `forced_break_of`) on
    //     inline-level boxes (inline boxes and inline-blocks) break the
    //     line box at the adjacent break opportunity — before the box's
    //     first fragment / after its last (CSS Fragmentation §5.2.1).
    //     The marks live in the scratch sidecars parallel to `frags`
    //     (see `LayoutScratch::frag_force_before`); the line breaker
    //     splits at them below and step 2b records the resulting line
    //     indices for the multicol distribution pass. Outside multicol
    //     there is no column to break into, so the marks stay clear.
    {
        let force_before = &mut scratch.frag_force_before;
        force_before.clear();
        force_before.resize(frags.len(), false);
        let force_after = &mut scratch.frag_force_after;
        force_after.clear();
        force_after.resize(frags.len(), false);
        if multicol.is_some() {
            for (k, f) in frags.iter().enumerate() {
                // Atomic inline-level boxes carry their arena index.
                if let Some(ib) = f.ib {
                    let st = forest.nodes[ib].style.as_deref();
                    force_before[k] = forced_break_of(st, props::BREAK_BEFORE);
                    force_after[k] = forced_break_of(st, props::BREAK_AFTER);
                }
            }
            for r in ranges.iter() {
                if forest.nodes[r.box_idx].box_type != BoxType::Inline || r.start >= r.end {
                    continue;
                }
                let st = forest.nodes[r.box_idx].style.as_deref();
                if forced_break_of(st, props::BREAK_BEFORE) {
                    force_before[r.start] = true;
                }
                if forced_break_of(st, props::BREAK_AFTER) {
                    force_after[r.end - 1] = true;
                }
            }
        }
    }
    let force_before = &scratch.frag_force_before[..];
    let force_after = &scratch.frag_force_after[..];

    let mut frag_lead = vec![0.0f32; frags.len()];
    let mut frag_trail = vec![0.0f32; frags.len()];
    for r in ranges.iter() {
        if forest.nodes[r.box_idx].box_type != BoxType::Inline || r.start >= r.end {
            continue;
        }
        let m = &forest.nodes[r.box_idx].metrics;
        frag_lead[r.start] += m.ml + m.bl + m.pl;
        frag_trail[r.end - 1] += m.mr + m.br + m.pr;
    }

    // 2. Break into lines. The wrap/pre/trim mode comes from each text
    //    run's own computed `white-space` (`Frag::ws`), never the block's:
    //    a `normal` span inside a `pre` block still wraps at spaces, and a
    //    `pre-wrap` span inside a `normal` block keeps its spaces and
    //    breaks only at its own forced breaks. `pre-wrap`/`pre-line`/
    //    `break-spaces` pieces arrive pre-fit from `wrap_pre_text` /
    //    `wrap_pre_line_text` (see `emit_text_frags`): spaces are never
    //    trimmed and lines break only at forced breaks. (`pre-line` pieces
    //    are already line-boundary-trimmed by `wrap_pre_line_text`.)
    //
    //    Floats intruding anywhere in a line's vertical band narrow that
    //    line and shift its origin (CSS 2.1 §9.5): each line is broken
    //    against its own available width, starting from the band the line
    //    is estimated to occupy (the block's line-height, exact for
    //    ordinary text lines). Once the line's true height resolves, the
    //    band test re-runs; when the true band narrows the line, its
    //    fragments are re-broken against the true width (bounded
    //    re-iteration until the break matches the band), and when the
    //    true band leaves no room the line shifts below the floats just
    //    like the estimated band. The next line's float test uses the
    //    true top.
    let lines = &mut scratch.lines;
    lines.clear();
    let mut i = 0;
    let nf = frags.len();
    let mut y_est = content_top;
    // Inline-box spacing shares shed by a dropped all-space line (see the
    // `all_space` branch below); carried to the next line's first
    // surviving fragment so they are not lost with the dropped line.
    let mut pending_lead = 0.0f32;
    let mut pending_trail = 0.0f32;
    'line: while i < nf {
        let skip_from = i;
        while i < nf && frags[i].is_space && !frags[i].break_before && ws_trims(frags[i].ws) {
            i += 1;
        }
        if i >= nf {
            break;
        }
        let start = i;
        // Inline-box boundary spacing with trimmable space (CSS 2.1 §8.6 /
        // §16.6.1): the leading spaces skipped above must not take their
        // inline-box spacing shares with them. Transfer the shares (plus
        // any pending from a dropped all-space line) to the line's first
        // surviving fragment, before breaking, so the spacing still
        // counts toward the line width and can force a wrap.
        if start > skip_from || pending_lead != 0.0 || pending_trail != 0.0 {
            for k in skip_from..start {
                pending_lead += frag_lead[k];
                pending_trail += frag_trail[k];
                frag_lead[k] = 0.0;
                frag_trail[k] = 0.0;
            }
            // Both shares sit adjacent to the surviving fragment on its
            // leading side: `pending_trail` is the right spacing of a box
            // *ending* at the skipped space. They render before the
            // survivor, never after it.
            frag_lead[start] += pending_lead + pending_trail;
            pending_lead = 0.0;
            pending_trail = 0.0;
        }
        let first_fw = frags[start].w + frag_lead[start] + frag_trail[start];
        let first_unbreakable = frags[start].ib.is_some() || !ws_wraps(frags[start].ws);
        let is_first_line = lines.is_empty();
        // The band the line is broken against starts as the estimate
        // `[y, y+block_lh)` and switches to the true `[y, y+lh)` once
        // `align_line` resolves the line's height below.
        let mut y = y_est;
        let mut band_h = block_lh;
        let mut j;
        let mut w_trimmed;
        let mut end_trimmed;
        let mut lh;
        let mut ltop;
        let mut dx;
        let mut avail_f;
        let mut iter = 0u32;
        loop {
            (dx, avail_f) = line_float_avail(floats, content_x, content_w, y, y + band_h);
            // A float covering the full width across the line's band leaves no
            // room: an unbreakable first fragment would overlap the float
            // (CSS 2.1 §9.5). More generally, when floats narrow the line
            // and the first fragment cannot wrap (`white-space: nowrap`
            // and friends, or an atomic inline-block) and is wider than the
            // narrowed space, the line shifts below the shallowest intruding
            // float(s) and re-tests, until it fits or no floats intrude.
            // With no float intrusion an over-wide fragment overflows
            // normally.
            // (The line's height resolves below, so the next line's float
            // test uses the true top.)
            loop {
                // crbug 1014247: with a non-zero first-line `text-indent`,
                // the line starts at the indent position (the placement
                // below sets `dx = 0` and adds the indent), so the
                // shift-below-float fit check measures from the indent
                // (`avail_f + dx - indent`), not from the float-narrowed
                // band edge: re-adding `dx` keeps the left-float exception
                // (a leading float must not force an atomic inline below
                // it when the atomic fits from the indent position) while
                // `avail_f` still subtracts right-float intrusion
                // (CSS 2.1 §9.4.2/§16.1). A float spanning the full band
                // still displaces the line (CSS 2.1 §9.5), via
                // `avail_f <= 0.0`.
                let fit_w = if is_first_line && indent_nonzero {
                    (avail_f + dx - indent).max(0.0)
                } else {
                    avail_f
                };
                let blocked = avail_f <= 0.0
                    || fit_w <= 0.0
                    || (fit_w < content_w && first_unbreakable && first_fw > fit_w);
                if !blocked {
                    break;
                }
                let bottom = min_intruding_float_bottom(floats, y, y + band_h);
                if bottom <= y {
                    break;
                }
                y = bottom;
                (dx, avail_f) = line_float_avail(floats, content_x, content_w, y, y + band_h);
            }
            // The first line's available width is reduced by `text-indent`
            // (which may be negative, widening it). When a non-zero indent
            // is present, the line starts at the indent position: the
            // left-float intrusion (`dx`) does not narrow the fit check
            // further (crbug 1014247 — a leading float must not force an
            // atomic inline to wrap when it fits from the indent
            // position), but right-float intrusion still does
            // (CSS 2.1 §9.4.2/§16.1): measure from the indent position to
            // the float-shortened right edge.
            let avail = if is_first_line {
                if indent_nonzero {
                    (avail_f + dx - indent).max(0.0)
                } else {
                    avail_f
                }
            } else {
                avail_f
            };
            let mut w = 0.0;
            w_trimmed = 0.0;
            end_trimmed = start;
            j = start;
            // A line made only of trimmable spaces is dropped (below).
            let mut all_space = true;
            while j < nf {
                // Inline-box spacing counts toward the line width so padding
                // and borders can force a wrap, like any other fragment width.
                let fw = frags[j].w + frag_lead[j] + frag_trail[j];
                let fbb = frags[j].break_before;
                if fbb && j > start {
                    break;
                }
                // Inline-level forced `break-before` (multicol): split the
                // line box here (CSS Fragmentation §5.2.1). At `j == start`
                // the line already begins at the break, so only the
                // recording in step 2b applies.
                if force_before[j] && j > start {
                    break;
                }
                let fws = frags[j].ws;
                if ws_wraps(fws) && j > start && w + fw > avail {
                    break;
                }
                w += fw;
                let fba = force_after[j];
                j += 1;
                if !ws_trims(fws) || !frags[j - 1].is_space {
                    w_trimmed = w;
                    end_trimmed = j;
                    all_space = false;
                }
                // Inline-level forced `break-after` (multicol): the line
                // box ends after this fragment; the next line starts a
                // fresh column (recorded in step 2b).
                if fba {
                    break;
                }
            }
            if all_space {
                // A line of only (trimmable) spaces: dropped. A shift above
                // still consumed the vertical space, so `y_est` advances.
                // The spaces' inline-box spacing shares carry to the next
                // line's first surviving fragment (see the leading
                // transfer above) rather than vanishing with the line.
                for k in start..j {
                    pending_lead += frag_lead[k];
                    pending_trail += frag_trail[k];
                    frag_lead[k] = 0.0;
                    frag_trail[k] = 0.0;
                }
                y_est = y;
                i = j;
                continue 'line;
            }
            // The strut is the block's own font as an empty inline box: its
            // share of the block line height comes from the face's real
            // ascent/descent, which floors the line-box height and anchors
            // the baseline (the quantity table cell-baseline alignment
            // compares across cells).
            let block_fs = forest.nodes[idx].data.font_size;
            let (strut_ascent, strut_descent) =
                split_line_height(fonts, block_fs, block_lh, family_id_for(forest, idx, fonts));
            // CSS 2.1 §10.8.1 `vertical-align: middle` centers on the
            // block's baseline minus half its x-height.
            let x_height = fonts.x_height(block_fs, family_id_for(forest, idx, fonts));
            let (top, bottom) = align_line(
                &mut frags[start..end_trimmed],
                -strut_ascent,
                strut_descent,
                x_height,
            );
            ltop = top;
            lh = (bottom - top).max(0.0);
            // The estimated band can differ from the true one when a tall
            // inline-level box stretches the line: re-test float intrusion
            // over the line's actual `[y, y+height)` band (still O(floats)).
            // The true band always covers the estimated one (`lh` includes
            // the strut), so the re-test can only narrow the line, never
            // widen it.
            let (fdx, favail) = line_float_avail(floats, content_x, content_w, y, y + lh);
            iter += 1;
            let narrowed = favail < avail_f || fdx != dx;
            // crbug 1014247: like the pre-break shift above, the true-band
            // re-test measures the atomic-inline fit from the first-line
            // indent position, not from the float-narrowed band edge:
            // re-adding `fdx` keeps the left-float exception while
            // `favail` still subtracts right-float intrusion.
            let retest_fit_w = if is_first_line && indent_nonzero {
                (favail + fdx - indent).max(0.0)
            } else {
                favail
            };
            let blocked = favail <= 0.0
                || retest_fit_w <= 0.0
                || (retest_fit_w < content_w && first_unbreakable && first_fw > retest_fit_w);
            if iter >= MAX_LINE_RETEST || (!narrowed && !blocked) {
                dx = fdx;
                avail_f = favail;
                break;
            }
            if blocked {
                // The true band leaves no usable room: shift the line below
                // the shallowest intruding float and restart from the
                // estimated band, mirroring the pre-break shift above.
                let bottom = min_intruding_float_bottom(floats, y, y + lh);
                if bottom <= y {
                    dx = fdx;
                    avail_f = favail;
                    break;
                }
                y = bottom;
                band_h = block_lh;
            } else {
                // The true band narrows the line: re-break its fragments
                // against the true width. Each pass drops fragments to the
                // next line (or settles), so this terminates; the cap is a
                // backstop.
                band_h = lh;
            }
        }
        // Trailing trimmable spaces (`end_trimmed..j`) are dropped from
        // the line; transfer their inline-box spacing shares to the last
        // surviving fragment so an inline box ending in a trimmed space
        // keeps its end spacing (CSS 2.1 §8.6 / §16.6.1). This runs once,
        // after the float re-break loop settles, so a share is never
        // transferred twice. The transferred width joins the line width
        // so `text-align` accounts for the spacing.
        if end_trimmed > start {
            let mut tl = 0.0f32;
            let mut tt = 0.0f32;
            for k in end_trimmed..j {
                tl += frag_lead[k];
                tt += frag_trail[k];
                frag_lead[k] = 0.0;
                frag_trail[k] = 0.0;
            }
            let surv = end_trimmed - 1;
            // Both shares sit adjacent to the surviving fragment on its
            // trailing side: `tl` is the left spacing of a box *starting*
            // at the trimmed space, `tt` the right spacing of a box
            // *ending* there. They render after the survivor, never
            // before it.
            frag_trail[surv] += tl + tt;
            w_trimmed += tl + tt;
        }
        lines.push(Line {
            start,
            end: end_trimmed,
            width: w_trimmed,
            top: ltop,
            dx,
            avail: avail_f,
            y,
        });
        y_est = y + lh;
        i = j;
    }

    // 2b. Inline-level forced-break line indices for the multicol
    //     distribution pass (see `InlineLayoutOutput::forced_break_lines`).
    //     A line starts a fresh column when a `break-after` mark fired on
    //     the previous line (the in-loop split ends the line right after
    //     the marked fragment, so the mark sits in the previous line's
    //     fragment span through its trailing trimmed spaces) or a
    //     `break-before` mark sits on this line's first fragment or on a
    //     trimmed fragment skipped between the lines. Line 0 is never
    //     recorded: a forced break at the very start of the content is a
    //     no-op (no empty column is created). Only computed for multicol
    //     containers; the marks are all clear otherwise.
    let forced_break_lines: Vec<usize> = if multicol.is_some() {
        let mut out = Vec::new();
        for (li, line) in lines.iter().enumerate() {
            if li == 0 {
                continue;
            }
            let prev = &lines[li - 1];
            let mut fresh = force_after[prev.start..line.start].iter().any(|&m| m);
            let scan_end = line.start.min(nf.saturating_sub(1));
            fresh |= (prev.end..=scan_end).any(|k| force_before[k]);
            if fresh {
                out.push(li);
            }
        }
        out
    } else {
        Vec::new()
    };

    // 3. X positions per line (vertical alignment already resolved in
    //    the breaking loop above). Each line carries its absolute y from
    //    break time, which already accounts for float clearance shifts.
    let mut last_baseline = content_top;
    let n_lines = lines.len();
    for (li, line) in lines.iter().enumerate() {
        // Under `unicode-bidi: plaintext` the line's paragraph supplies
        // the base direction (step 1a); every other block uses the one
        // block-level alignment computed at the top.
        let line_align = if para_dirs.is_empty() || line.start >= frag_para.len() {
            text_align
        } else {
            let para = frag_para[line.start];
            debug_assert!(
                para < para_dirs.len(),
                "paragraph index {para} out of range for {} dirs (line.start={})",
                para_dirs.len(),
                line.start
            );
            effective_text_align_with_rtl(forest, idx, para_dirs[para])
        };
        // `text-align: justify` (CSS 2.1 §16.2): the line's leftover
        // space is spread across the word gaps — one extra advance per
        // surviving space fragment. The last line of the block is exempt
        // (it aligns per `text-align-last`, whose initial `auto` behaves
        // as `start`), as is a line with no gaps or no leftover space.
        // The leftover is measured against the float-shortened available
        // width, so a line beside a float justifies within its narrowed
        // band. (Simplification: a non-final line ending at a forced
        // break is still justified; CSS text-3 §7 would exempt it too via
        // `text-align-last`.)
        let mut extra_per_gap = 0.0f32;
        if line_align == TextAlign::Justify && li + 1 < n_lines {
            // The first line's available width was reduced by
            // `text-indent` at break time; mirror that here, including
            // the left-float re-add (crbug 1014247): with a non-zero
            // indent the line starts at the indent position, so the
            // leftover is measured against `avail_f + dx - indent`
            // (`line.dx` holds the raw left-float intrusion).
            let avail = if li == 0 {
                let dx = if indent_nonzero { line.dx } else { 0.0 };
                (line.avail + dx - indent).max(0.0)
            } else {
                line.avail
            };
            let leftover = avail - line.width;
            let gaps = frags[line.start..line.end]
                .iter()
                .filter(|f| f.is_space)
                .count();
            if leftover > 0.0 && gaps > 0 {
                extra_per_gap = leftover / gaps as f32;
            }
        }
        let off = match line_align {
            TextAlign::Left | TextAlign::Justify => 0.0,
            TextAlign::Center => (line.avail - line.width) / 2.0,
            TextAlign::Right => line.avail - line.width,
        };
        // `text-indent` offsets the first line box; `dx` shifts the line
        // right past left floats. When the first line has a non-zero
        // indent, the indent determines the start position and the float
        // intrusion is not added (see the fit-check above, crbug
        // 1014247). Inline-box spacing is laid out around each box's
        // first/last fragments.
        let y = line.y;
        let dx = if li == 0 && indent != 0.0 {
            0.0
        } else {
            line.dx
        };
        let mut x = content_x + dx + off + if li == 0 { indent } else { 0.0 };
        for (k, f) in frags[line.start..line.end].iter_mut().enumerate() {
            let j = line.start + k;
            x += frag_lead[j];
            f.x = x;
            f.abs_top = y + (f.rt - line.top);
            f.placed = true;
            x += f.w + frag_trail[j];
            // Justified word gap: the space keeps its natural advance;
            // the cursor takes the extra share, widening the gap to the
            // next word.
            if extra_per_gap > 0.0 && f.is_space {
                x += extra_per_gap;
            }
        }
        last_baseline = y - line.top;
    }

    // v1 multi-column layout (inline content): distribute the line boxes
    // across the columns right after positioning, so the out-of-flow
    // static positions below, inline-block re-anchoring, rect unions, and
    // fragment emission all see shifted positions. Returns the new
    // natural content height (tallest column bottom minus content_top).
    // `forced_break_lines` (step 2b) is threaded into
    // `distribute_inline_lines_for_container` so the pass starts a fresh
    // column at each of those line indices; the container identity lets a
    // re-layout replace this container's recorded column-rule spec.
    let inline_h = match multicol {
        Some(mc) => layout_multicol::distribute_inline_lines_for_container(
            Some(idx),
            forest.nodes[idx].dom,
            &lines[..],
            &mut *frags,
            mc,
            content_x,
            content_top,
            multicol_hspec,
            &forced_break_lines,
            floats,
        ),
        None => y_est - content_top,
    };

    // Out-of-flow boxes skipped from the fragment walk above: resolve
    // their static positions now that fragments carry final x/y. The
    // anchor is the margin-box edge just after the nearest preceding
    // placed fragment on its line; with no preceding placed fragment,
    // the content-box origin. The box's used margins are not folded in
    // here: `record_abs_item_margin_edge` marks them pending and
    // `layout_position::abs_position` adds them when the offsets are
    // `auto`, giving the hypothetical in-flow border-box position
    // (CSS 2.1 §10.3.7/§10.6.4): horizontally just after the nearest
    // preceding placed fragment — the new line's start fragment after a
    // forced break — and vertically at that fragment's line baseline.
    for (node, prev_frag) in inline_abs.drain(..) {
        let mut fi = prev_frag;
        let mut pos = None;
        while let Some(i) = fi {
            debug_assert!(
                i < frags.len(),
                "abspos anchor fragment {i} out of range for {} frags",
                frags.len()
            );
            let f = &frags[i];
            if f.placed {
                // Baseline-align the hypothetical box in its line box
                // (CSS 2.1 §10.3.7/§10.6.4): the vertical anchor is the
                // line's baseline (`abs_top - rt`; `rt` is the fragment's
                // top relative to the baseline), not the preceding
                // fragment's top. After a forced break the walk lands on
                // the hard-line-start fragment, so the box anchors to the
                // new line's start x and baseline.
                pos = Some((f.x + f.w, f.abs_top - f.rt));
                break;
            }
            fi = i.checked_sub(1);
        }
        let (sx, sy) = pos.unwrap_or((content_x, content_top));
        layout_position::record_abs_item_margin_edge(forest, node, sx, sy);
    }

    // Record the last line's baseline relative to the border-box top, for
    // inline-block baseline alignment by ancestors.
    {
        let (bt, pt) = {
            let m = &forest.nodes[idx].metrics;
            (m.bt, m.pt)
        };
        let n = &mut forest.nodes[idx];
        n.baseline = if lines.is_empty() {
            bt + pt
        } else {
            (last_baseline - content_top) + bt + pt
        };
    }

    // 4. Move inline-block subtrees from their provisional origin into place.
    // The deltas are recorded on the subtree roots (O(1) each) and applied
    // in the single `apply_pending_shifts` pass at the end of
    // `layout_forest`: nested inline-blocks no longer re-shift shared
    // subtrees once per nesting level.
    for f in frags.iter() {
        let ib = f.ib;
        if !f.placed || ib.is_none() {
            continue;
        }
        let ib = ib.expect("checked");
        let (bx, by, ml, m_top) = {
            let n = &forest.nodes[ib];
            (n.rect.x, n.rect.y, n.metrics.ml, n.metrics.mt)
        };
        let (fx, ftop) = (f.x, f.abs_top);
        record_shift(forest, ib, (fx + ml) - bx, (ftop + m_top) - by);
    }

    // 5. Union rects for inline and text boxes, bottom-up in a single
    //    pass. A range's rect is the union of its directly-owned fragments
    //    (those not covered by a nested child range, e.g. an inline-block
    //    child's fragment) and its child ranges' rects. Children always
    //    have larger indices than their parent, so iterating in reverse
    //    visits children before parents; every fragment is scanned exactly
    //    once, by its deepest containing range.
    #[derive(Clone, Copy)]
    struct Union {
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
    }
    impl Union {
        fn empty() -> Union {
            Union {
                x0: f32::INFINITY,
                y0: f32::INFINITY,
                x1: f32::NEG_INFINITY,
                y1: f32::NEG_INFINITY,
            }
        }
        fn add(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
            self.x0 = self.x0.min(x0);
            self.y0 = self.y0.min(y0);
            self.x1 = self.x1.max(x1);
            self.y1 = self.y1.max(y1);
        }
        fn add_frag(&mut self, f: &Frag) {
            self.add(f.x, f.abs_top, f.x + f.w, f.abs_top + (f.rb - f.rt));
        }
        fn is_empty(self) -> bool {
            !self.x0.is_finite()
        }
    }
    let mut unions: Vec<Union> = vec![Union::empty(); ranges.len()];
    for (ri, r) in ranges.iter().enumerate().rev() {
        let mut u = Union::empty();
        // Child ranges are disjoint and ordered by fragment index; the
        // gaps between them hold this range's directly-owned fragments.
        let mut cur = r.start;
        for &ch in r.children.iter() {
            let c = &ranges[ch];
            debug_assert!(c.start >= cur, "child ranges are ordered and disjoint");
            for f in frags[cur..c.start].iter() {
                #[cfg(test)]
                {
                    forest.union_frag_visits += 1;
                }
                if f.placed {
                    u.add_frag(f);
                }
            }
            let cu = unions[ch];
            if !cu.is_empty() {
                u.add(cu.x0, cu.y0, cu.x1, cu.y1);
            }
            cur = c.end;
        }
        for f in frags[cur..r.end].iter() {
            #[cfg(test)]
            {
                forest.union_frag_visits += 1;
            }
            if f.placed {
                u.add_frag(f);
            }
        }
        unions[ri] = u;
        // An inline box's border box extends past its fragments by its
        // border/padding (fragment x positions already sit inside the
        // lead/trail spacing, so the union is widened back out by it).
        // Margins are excluded: they offset placement but sit outside the
        // border box (CSS 2.1 §8.6). The widened union is stored so nested
        // inline boxes propagate their spacing to ancestors.
        if !u.is_empty() && forest.nodes[r.box_idx].box_type == BoxType::Inline {
            let m = &forest.nodes[r.box_idx].metrics;
            u.x0 -= m.bl + m.pl;
            u.x1 += m.br + m.pr;
            unions[ri] = u;
        }
        let n = &mut forest.nodes[r.box_idx];
        n.rect = if u.is_empty() {
            Rect {
                x: content_x,
                y: content_top,
                w: 0.0,
                h: 0.0,
            }
        } else {
            Rect {
                x: u.x0,
                y: u.y0,
                w: (u.x1 - u.x0).max(0.0),
                h: (u.y1 - u.y0).max(0.0),
            }
        };
    }

    // 6. Per-line fragments for text boxes (paint/JS contract). One
    //    `TextFragment` per placed line fragment: `rect` is the fragment's
    //    placed rect in absolute page coordinates, `baseline` the line's
    //    baseline (`abs_top - rt`: `rt` is relative to the baseline), and
    //    `text_range` the byte range into the box's (transformed) text
    //    that the fragment renders. Non-empty only for `BoxType::Text`.
    //    The vec is cleared first: a subtree re-laid out (flex
    //    `align-self: stretch` runs a second `Task::Block`) regenerates
    //    its fragments, and appending again would double-draw text at
    //    stale positions.
    for r in ranges.iter() {
        if forest.nodes[r.box_idx].box_type != BoxType::Text {
            continue;
        }
        let out = &mut forest.nodes[r.box_idx].fragments;
        out.clear();
        for f in frags[r.start..r.end].iter() {
            if !f.placed {
                continue;
            }
            out.push(super::TextFragment {
                rect: Rect {
                    x: f.x,
                    y: f.abs_top,
                    w: f.w,
                    h: (f.rb - f.rt).max(0.0),
                },
                baseline: f.abs_top - f.rt,
                text_range: f.text_range.clone().unwrap_or(0..0),
            });
        }
    }

    // `inline_h` is the post-distribution height when this is a multicol
    // container, else the plain stacked height.
    InlineLayoutOutput {
        height: inline_h,
        forced_break_lines,
    }
}

// ---------------------------------------------------------------------------
// Preferred widths (inline-block shrink-to-fit)
// ---------------------------------------------------------------------------

/// Memoized preferred widths for one layout pass, keyed by arena index.
/// Inline-block shrink-to-fit asks for the preferred width of every nested
/// inline-block; without caching each call re-walks the whole subtree,
/// which is O(subtree^2) for nested inline-blocks. A cached node's entire
/// subtree was walked to produce it (entries are written bottom-up over
/// complete walks, and the tree does not change mid-pass), so a hit prunes
/// the whole subtree. The cache is created fresh in `layout_forest`, which
/// invalidates it per `layout()` call. Cached values are content-box
/// preferred widths that already include nested child padding/border/
/// margins (see [`intrinsic_h_extras`]); the contribution definition is
/// uniform across the pass, so no per-entry invalidation is needed beyond
/// the fresh-per-`layout()` cache.
pub(crate) struct PrefWidthCache {
    /// Per-node generation stamp; the entry is valid iff it equals `cur`.
    stamps: Vec<u64>,
    cur: u64,
    /// Memoized automatic-table grid-maximum widths (see
    /// [`crate::layout::layout_table::table_auto_grid_max_width`]),
    /// per node, stamped with the generation that computed them: a hit
    /// requires the stored generation to equal `cur`, so
    /// [`PrefWidthCache::next_gen`] invalidates these exactly like the
    /// boolean stamps above. Only `Some` grid-max results are stored
    /// (fixed and vertical tables return `None` cheaply already).
    grid_max: Vec<Option<(u64, f32)>>,
}

impl PrefWidthCache {
    // pub(crate) so `tests.rs` can measure a single node's preferred
    // width directly (w78fix H1 pref-level pin); production callers
    // already construct it this way.
    pub(crate) fn new(node_count: usize) -> PrefWidthCache {
        PrefWidthCache {
            stamps: vec![0; node_count],
            cur: 1,
            grid_max: vec![None; node_count],
        }
    }

    /// The cached grid-max for `idx` in the current generation, if any.
    pub(crate) fn grid_max_get(&self, idx: usize) -> Option<f32> {
        self.grid_max
            .get(idx)
            .copied()
            .flatten()
            .filter(|&(g, _)| g == self.cur)
            .map(|(_, v)| v)
    }

    /// Store the grid-max for `idx` in the current generation.
    pub(crate) fn grid_max_set(&mut self, idx: usize, value: f32) {
        if idx >= self.grid_max.len() {
            self.grid_max.resize(idx + 1, None);
        }
        self.grid_max[idx] = Some((self.cur, value));
    }

    fn get(&self, idx: usize) -> bool {
        self.stamps.get(idx).is_some_and(|&s| s == self.cur)
    }

    fn set(&mut self, idx: usize) {
        if idx >= self.stamps.len() {
            self.stamps.resize(idx + 1, 0);
        }
        self.stamps[idx] = self.cur;
    }

    /// Invalidate every entry without reallocating: after the bump no
    /// previously stamped node reads as cached, exactly like a fresh
    /// cache. Used where a cache is hoisted across repeated
    /// measurements ([`min_content_width_pooled`]'s flex arm reuses one
    /// cache across the flex nodes of a walk, bumping per node).
    fn next_gen(&mut self) {
        self.cur = self.cur.wrapping_add(1);
        // Stamp 0 means "never stamped" (`new` starts `cur` at 1 and
        // the stamp vec is zero-initialized), so it must never become
        // the current generation, even across a `u64` wrap.
        if self.cur == 0 {
            self.cur = 1;
        }
    }
}

/// Pooled preorder-walk buffers for the intrinsic-width measurements
/// ([`preferred_width_mcw`], [`min_content_width_pooled`]): one walk
/// serves many measurements instead of allocating per call. Bundled
/// so the multi-buffer signatures below stay readable. Every use
/// clears the walks on entry and runs to completion before the next
/// begins (sequential, never interleaved or re-entrant); a nested
/// measurement takes a *separate* `WalkBufs` — never an in-progress
/// outer walk's — which the borrow checker enforces where the buffers
/// are threaded as parameters.
///
/// `flex` is the lazily built scratch for
/// [`min_content_width_pooled`]'s flex arm (see [`FlexScratch`]):
/// hoisted here so it lives as long as the walk buffers themselves —
/// one reusable aux buffer, one generation-stamped cache and one
/// nested walk pair serve every flex node of every walk that reuses
/// this `WalkBufs`, instead of allocating per flex node (or per
/// walk). `clear` deliberately leaves it alone.
pub(crate) struct WalkBufs {
    /// Preorder collection order.
    pub(crate) order: Vec<usize>,
    /// Preorder walk stack.
    pub(crate) stack: Vec<usize>,
    /// Flex-arm scratch, built on first use.
    pub(crate) flex: Option<Box<FlexScratch>>,
}

impl WalkBufs {
    pub(crate) fn new() -> WalkBufs {
        WalkBufs {
            order: Vec::new(),
            stack: Vec::new(),
            flex: None,
        }
    }

    /// Clear both walks for reuse. The flex scratch is long-lived and
    /// is intentionally kept: it stays sound across clears because
    /// every flex-arm use bumps the nested cache's generation via
    /// [`FlexScratch::begin_node`] before reading any stamp (and both
    /// dispatches clear the nested walk pair on entry), so no stale
    /// entry can leak from a previous walk into the next.
    pub(crate) fn clear(&mut self) {
        self.order.clear();
        self.stack.clear();
    }
}

/// Scratch for [`min_content_width_pooled`]'s flex arm, hoisted into
/// [`WalkBufs`] so it is reused across walks. The arm's clamp
/// adjustment measures content-based `min-width`/`max-width` keywords
/// on the flex items while the outer walk's buffers are borrowed by
/// the per-node loop, so it needs scratch of its own: `aux` is the
/// node-indexed width buffer for those nested measurements, `pw` the
/// pref-width cache they memoize through (invalidated per flex node
/// via [`PrefWidthCache::next_gen`] — semantically a fresh cache each
/// time, without the O(node-count) alloc+zero), and `bufs`/`inner`
/// the nested walk pair.
pub(crate) struct FlexScratch {
    /// Reusable width buffer for the nested clamp measurements.
    pub(crate) aux: Vec<f32>,
    /// Generation-stamped cache for the nested clamp measurements.
    pub(crate) pw: PrefWidthCache,
    /// Nested walk pair for the clamp's content-keyword measurements.
    pub(crate) bufs: WalkBufs,
    /// Second nested walk pair (see [`WalkBufs`]).
    pub(crate) inner: WalkBufs,
}

impl FlexScratch {
    fn new(node_count: usize) -> FlexScratch {
        FlexScratch {
            aux: Vec::new(),
            pw: PrefWidthCache::new(node_count),
            bufs: WalkBufs::new(),
            inner: WalkBufs::new(),
        }
    }

    /// Begin one flex node's clamp measurements: bump the nested
    /// cache's generation so it reads as fresh. Every flex-arm use
    /// must call this before touching `pw` — the bump is what keeps
    /// the hoisted scratch sound across flex nodes and across walks,
    /// so funnel all uses through this accessor rather than calling
    /// [`PrefWidthCache::next_gen`] directly.
    fn begin_node(&mut self) {
        self.pw.next_gen();
    }
}

/// Used horizontal padding/border/margins of one box for the intrinsic
/// passes below, resolved against a zero base: percentage padding and
/// margins have no definite containing block this early (the same base
/// `table_pref_width` uses for cells), and `auto` margins contribute
/// zero to intrinsic sizes. Margins are excluded for internal table
/// boxes (CSS 2.1 §8.3): box construction already zeroes them, but the
/// intrinsic passes must uphold the rule themselves, since
/// `table_pref_width` builds column widths from border/padding only —
/// counting margins here would let min-content exceed preferred for a
/// margined cell. Mirrors the horizontal half of [`resolve_metrics`]
/// but is read-only, so the intrinsic passes stay side-effect free (no
/// `metrics`/`rect` writes to leak into later passes).
pub(crate) fn intrinsic_h_extras(forest: &Forest, idx: usize) -> f32 {
    let n = &forest.nodes[idx];
    let d = &n.data;
    let (fs, rfs, vpw, vph) = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
    // CSS 2.1 §8.3: margins do not apply to internal table boxes
    // (cells, rows, row groups, columns); the table wrapper itself keeps
    // them.
    let (ml, mr) = if matches!(
        n.box_type,
        BoxType::TableRowGroup
            | BoxType::TableRow
            | BoxType::TableCell
            | BoxType::TableColumn
            | BoxType::TableColumnGroup
    ) {
        (0.0, 0.0)
    } else {
        (
            margin_opt(d.margin[3], 0.0, fs, rfs, vpw, vph).unwrap_or(0.0),
            margin_opt(d.margin[1], 0.0, fs, rfs, vpw, vph).unwrap_or(0.0),
        )
    };
    // The border/padding half (with its `border-style: none | hidden`
    // zeroing) lives in `intrinsic_h_bp`.
    ml + intrinsic_h_bp(forest, idx) + mr
}

#[cfg(test)]
mod intrinsic_extras_tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// Box tree for `<table><tr><td>hello</td></tr></table>` with a
    /// margined cell; returns the forest plus the table and cell node
    /// ids. The cell's `data.margin` is set explicitly: box construction
    /// zeroes internal-table-box margins (CSS 2.1 §8.3), but the
    /// intrinsic passes must exclude margins for those boxes regardless
    /// of what `data.margin` holds.
    fn margined_cell_forest() -> (crate::boxes::Forest, usize, usize) {
        let (doc, _) = bowser_dom::Document::parse_html(b"<table><tr><td>hello</td></tr></table>");
        let ss = bowser_css::parse_stylesheet(
            "table { margin: 15px; padding: 0; border: none; border-spacing: 0; } \
             td { font-size: 10px; margin: 20px; padding: 4px; border: none; }",
        );
        let ua = bowser_css::default_ua_stylesheet();
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut table = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::Table {
                table = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let table = table.expect("table box");
        let grid = layout_table::collect_table_grid(&forest, table);
        assert_eq!(grid.cells.len(), 1);
        let cell = grid.cells[0].node;
        forest.nodes[cell].data.margin = [SpecMargin::Len(SpecLen::Px(20.0)); 4];
        (forest, table, cell)
    }

    #[test]
    fn internal_table_boxes_exclude_margins_from_extras() {
        let (forest, _, cell) = margined_cell_forest();
        // Padding 4px each side, no border; the 20px margins are
        // excluded per CSS 2.1 §8.3.
        assert!(approx(intrinsic_h_extras(&forest, cell), 8.0));
    }

    #[test]
    fn table_wrapper_keeps_its_margins() {
        // Control: the exclusion is scoped to *internal* table boxes;
        // the table wrapper itself is a normal margin participant.
        let (forest, table, _) = margined_cell_forest();
        assert!(approx(intrinsic_h_extras(&forest, table), 30.0));
    }

    #[test]
    fn margined_cell_min_content_matches_preferred() {
        // Regression: min-content and preferred must agree for a
        // margined cell. Before the §8.3 exclusion, min-content added
        // the cell margins (via `intrinsic_h_extras`) while
        // `table_pref_width` built the column from border/padding only,
        // so min-content exceeded preferred by the margins.
        let (mut forest, table, _) = margined_cell_forest();
        let fonts = crate::FixedFontMetrics;
        // Single word: min-content and preferred agree on the content
        // width; both exclude the 20px cell margins (padding only).
        let content = fonts.advance("hello", 10.0, FontFamilyId::DEFAULT);
        let mut scratch = Vec::new();
        let mc = min_content_width(&mut forest, table, &fonts, &mut scratch);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let pref = preferred_width(&mut forest, table, &fonts, &mut cache);
        assert!(approx(mc, content + 8.0), "min-content {mc}");
        assert!(approx(pref, content + 8.0), "preferred {pref}");
    }

    /// Box tree for a `writing-mode: vertical-lr` table with a single
    /// cell whose content is `inner_html`; returns the forest plus the
    /// cell node id (the first `TableCell` in DFS order, i.e. the outer
    /// cell even when the content nests another table).
    fn vwm_cell_forest(inner_html: &str) -> (crate::boxes::Forest, usize) {
        let html = format!("<table><tr><td>{inner_html}</td></tr></table>");
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(
            "table { writing-mode: vertical-lr; border-spacing: 0; } \
             td { font-size: 10px; padding: 0; }",
        );
        let ua = bowser_css::default_ua_stylesheet();
        let Some((forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut cell = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::TableCell {
                cell = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        (forest, cell.expect("cell box"))
    }

    #[test]
    fn vwm_definite_content_counts_child_padding() {
        // The child's inline contribution is its border box: a 100px
        // child with 10px vertical padding contributes 120, not 100.
        let (forest, cell) = vwm_cell_forest("<div style=\"height:100px;padding:10px 0\"></div>");
        assert!(approx(
            vertical_definite_content(&forest, cell).unwrap(),
            120.0
        ));
    }

    #[test]
    fn vwm_definite_content_counts_child_margins() {
        // Inline-axis margins never collapse: a 100px child with 20px
        // vertical margins contributes 140, not 100.
        let (forest, cell) = vwm_cell_forest("<div style=\"height:100px;margin:20px 0\"></div>");
        assert!(approx(
            vertical_definite_content(&forest, cell).unwrap(),
            140.0
        ));
    }

    #[test]
    fn vwm_definite_content_border_box_child() {
        // Under `border-box` the 100px height already includes the
        // padding: the contribution is the 100px border box, not the
        // 60px content box.
        let (forest, cell) = vwm_cell_forest(
            "<div style=\"height:100px;padding:20px 0;box-sizing:border-box\"></div>",
        );
        assert!(approx(
            vertical_definite_content(&forest, cell).unwrap(),
            100.0
        ));
    }

    #[test]
    fn vwm_definite_content_nested_table_clamps_physically() {
        // A nested vertical table child clamps its physical height
        // against its physical `min-height`/`max-height`: the
        // `min-width:500px` (a block-axis clamp) must not leak into the
        // inline contribution, and `min-height:200px` must apply.
        let (forest, cell) = vwm_cell_forest(
            "<table style=\"height:100px;min-width:500px\"><tr><td>x</td></tr></table>",
        );
        assert!(approx(
            vertical_definite_content(&forest, cell).unwrap(),
            100.0
        ));
        let (forest, cell) = vwm_cell_forest(
            "<table style=\"height:100px;min-height:200px\"><tr><td>x</td></tr></table>",
        );
        assert!(approx(
            vertical_definite_content(&forest, cell).unwrap(),
            200.0
        ));
    }

    #[test]
    fn table_min_content_counts_spanning_cells_and_specified_widths() {
        // Regression: the table arm of `min_content_width` used to
        // consider only colspan == 1 cells, ignoring spanning-cell
        // minima (step 3), specified `<col>` widths (step 2), and
        // `min-width` floors. It now shares `auto_column_minima` with
        // `layout_table`.
        let (doc, _) = bowser_dom::Document::parse_html(
            b"<table><colgroup><col class=\"w\"><col></colgroup>\
              <tr><td colspan=\"2\">abcdefghijklmnopqrstuvwxyzabcd</td></tr>\
              <tr><td>x</td><td class=\"mw\">y</td></tr></table>",
        );
        let ss = bowser_css::parse_stylesheet(
            "table { border-collapse: collapse; border-spacing: 0; } \
             td { font-size: 10px; padding: 0; border: none; } \
             col.w { width: 80px; } td.mw { min-width: 70px; }",
        );
        let ua = bowser_css::default_ua_stylesheet();
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut table = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::Table {
                table = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let table = table.expect("table box");
        let fonts = crate::FixedFontMetrics;
        let grid = layout_table::collect_table_grid(&forest, table);
        assert_eq!(grid.ncols, 2);
        // Per-column minima through the shared helper: col 0 is floored
        // by the 80px `col` width, col 1 by the 70px `min-width`, and the
        // 30-char spanning cell's excess goes entirely to col 1. Per CSS
        // Tables 3 §3.9.3.2 rule 1, a spanning cell's excess is
        // distributed only over the spanned columns that are not
        // constrained: col 0 is constrained by its specified `col`
        // width, so col 1 (floored only by `min-width`) takes the whole
        // excess. WPT colspan-redistribution table 12 confirms this
        // shape (excess to the single non-constrained spanned column,
        // not an even spread over every spanned column).
        let (_, spacing_h, _) = layout_table::table_border_model(&forest, table);
        let col_spec = layout_table::column_specified_widths(&mut forest, &grid, 0.0, 0.0);
        let mut pw_cache = PrefWidthCache::new(forest.nodes.len());
        let mut mcw_scratch = Vec::new();
        let mut int_bufs = WalkBufs::new();
        let mut int_inner = WalkBufs::new();
        let (col_max, col_max_nonspanning) = layout_table::auto_column_maxima(
            &mut forest,
            &grid,
            &col_spec,
            spacing_h,
            &fonts,
            &mut pw_cache,
            &mut mcw_scratch,
            &mut int_bufs,
            &mut int_inner,
        );
        let mins = layout_table::auto_column_minima(
            &mut forest,
            &grid,
            &col_spec,
            spacing_h,
            0.0,
            &fonts,
            &col_max,
            &col_max_nonspanning,
        );
        let span_need = fonts.advance(
            "abcdefghijklmnopqrstuvwxyzabcd",
            10.0,
            FontFamilyId::DEFAULT,
        );
        let (e0, e1) = if span_need > 150.0 {
            (80.0, 70.0 + (span_need - 150.0))
        } else {
            (80.0, 70.0)
        };
        assert!(approx(mins[0], e0), "col0 min {} != {e0}", mins[0]);
        assert!(approx(mins[1], e1), "col1 min {} != {e1}", mins[1]);
        // The `min_content_width` table arm sums the same minima (the
        // old arm returned just the two 1-char cells' widths here).
        let mut scratch = Vec::new();
        let mc = min_content_width(&mut forest, table, &fonts, &mut scratch);
        assert!(approx(mc, e0 + e1), "min-content {mc} != {}", e0 + e1);
    }

    #[test]
    fn clamp_content_width_own_terms_border_box_passthrough_is_exact() {
        // Layout-L4: the border-box arm converts the min/max bounds
        // into content-box terms instead of round-tripping the width
        // through `(w + extras) - extras`. With these values the old
        // round-trip lost a ulp (51.86627197… instead of
        // 51.86627578…), which defeated `replaced_minmax_transfer`'s
        // `intrinsic_w != Some(w)` early-out and recomputed the height
        // through the ratio with dust.
        let w = 51.866276f32;
        let extras = 21.661085f32;
        assert_eq!(
            clamp_content_width_own_terms(0.0, f32::INFINITY, BoxSizing::BorderBox, extras, w),
            w
        );
        // Clamps still apply in the box-sizing's own terms, min wins on
        // conflict — identical outcomes to the old formulation.
        assert_eq!(
            clamp_content_width_own_terms(0.0, 105.0, BoxSizing::BorderBox, 10.0, 100.0),
            95.0
        );
        assert_eq!(
            clamp_content_width_own_terms(120.0, f32::INFINITY, BoxSizing::BorderBox, 10.0, 100.0),
            110.0
        );
        assert_eq!(
            clamp_content_width_own_terms(0.0, 50.0, BoxSizing::BorderBox, 10.0, 100.0),
            40.0
        );
        assert_eq!(
            clamp_content_width_own_terms(200.0, 150.0, BoxSizing::BorderBox, 10.0, 100.0),
            190.0
        );
    }
}

/// Definite specified content width of an atomic inline-level box
/// (`BoxType::InlineBlock`: inline-block, inline-table, inline-flex,
/// inline-grid, and replaced inline elements all share the box type),
/// for intrinsic sizing (css-sizing-3 §5.2).
///
/// Content-based used widths (css-sizing-3 §5): `min-content` is the
/// min-content size, `max-content` the preferred size, and
/// `fit-content` is `min(max-content, max(min-content, available))`.
/// Returns the *content-box* width; callers apply `min-width` /
/// `max-width` through their usual clamp paths. `avail` is the
/// available content-box width (containing-block width minus used side
/// margins and the box's own border/padding). `bufs`/`inner` are pooled
/// preorder-walk scratch (see [`WalkBufs`]): `bufs` serves the outer
/// measurement, `inner` the nested walks inside
/// [`preferred_width_mcw`]'s F7 `width: min-content` arms.
#[allow(clippy::too_many_arguments)]
pub(crate) fn content_based_width(
    forest: &mut Forest,
    idx: usize,
    spec: SpecLen,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
    avail: f32,
) -> f32 {
    match spec {
        SpecLen::MinContent => min_content_width_pooled(forest, idx, fonts, mcw, bufs),
        SpecLen::MaxContent => preferred_width_mcw(forest, idx, fonts, pw_cache, mcw, bufs, inner),
        SpecLen::FitContent => {
            let mc = min_content_width_pooled(forest, idx, fonts, mcw, bufs);
            let pc = preferred_width_mcw(forest, idx, fonts, pw_cache, mcw, bufs, inner);
            pc.min(mc.max(avail))
        }
        _ => {
            // `pub(crate)` with all current callers guarding on the
            // content-based keywords: a debug assertion instead of a
            // hard panic, so a future unguarded caller degrades to a
            // zero width in release rather than crashing (F6).
            debug_assert!(false, "content_based_width called with {spec:?}");
            0.0
        }
    }
}

/// Content-box width for a `flex-basis` content keyword (w76).
/// Like [`content_based_width`], except the basis overrides the
/// item's main-size property, so a definite `width` must not floor
/// the measurement: `width:100px` + `flex-basis:max-content` over
/// 300px of content is 300, not 100.
/// How each shape measures (w76fix finding 1):
/// - Generic (block / inline-block) boxes with a definite width and
///   no replaced intrinsic size read their raw content back from the
///   children's already measured sizes instead — the generic
///   run/max for max-content, the children's min-content maxima for
///   min-content — because the intrinsic dispatches floor such a box
///   at its definite width.
/// - Flex containers measure with the flex intrinsic algorithm
///   (item sum for a row, widest item for a column) over the
///   children's walked sizes, again without the definite floor the
///   dispatches apply (block precedence there would report the width
///   itself, and the generic read-back would report the widest
///   child instead of the item sum).
/// - Multicol blocks measure with the column transform
///   ([`multicol_min_content_width`] /
///   [`multicol_max_content_width`]) over the single-column content
///   measure, without the definite floor.
/// - Tables with a definite width measure from their columns
///   directly (maxima plus spacing for max-content, minima plus
///   spacing for min-content): the dispatched value floors such a
///   table at its width (`table_pref_width`'s fast path), so keeping
///   it would defeat the basis. Fixed-layout and vertical tables
///   keep the dispatched value (no column-based size exists for
///   them here). Auto-width tables never reach the column arm —
///   their dispatched value is already column-based.
/// - Grids and inline-grids with a definite width measure their
///   tracks directly (w77, via
///   [`layout_grid::grid_pref_width_ignoring_definite`]): the
///   dispatched value floors such a grid at its width
///   (`grid_pref_width`'s fast path), so keeping it would defeat
///   the basis, exactly as for tables above. Ordinary layout keeps
///   the fast path — only this basis arm uses the
///   definite-ignoring variant. A grid's min-content basis needs
///   no arm: `grid_min_width` has no definite fast path, so the
///   dispatched min-content value is already track-based.
///   Inline-* boxes as flex items are blockified (w77g for tables,
///   w78 for block/flex/grid, css-flexbox-1 §4): an in-flow inline-*
///   child of a flex container is promoted to a `Block` / `Grid` /
///   `Table` box carrying its own style, so it IS the item, its own
///   `flex-basis` applies, and it reaches the table column arm above
///   (inline-table) or the grid track arm above (inline-grid, via
///   its unchanged `InlineGrid` display) like a block-level box of
///   that shape. A promoted inline-flex reaches the flex arm via
///   `flex_kind`, which still reads its raw display.
///
/// Known limitation (w76fix L7), fixed in w77: a replaced box (an
/// intrinsic size is present) used to keep the dispatched
/// definite-width floor — an `<img>` with `width:100px` +
/// `flex-basis:max-content` over a 300px natural size measured 100.
/// [`flex_basis_keyword_size`] now has a replaced arm returning the
/// intrinsic natural size (box-sizing-adjusted) instead.
/// The returned value is content-box, like [`content_based_width`];
/// it is never border-box-stripped (the keyword measures content,
/// unlike a definite length basis, which resolves in box-sizing
/// terms).
#[allow(clippy::too_many_arguments)]
fn flex_basis_content_width(
    forest: &mut Forest,
    idx: usize,
    spec: SpecLen,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
    avail: f32,
) -> f32 {
    match spec {
        SpecLen::MinContent | SpecLen::MaxContent => {
            flex_basis_keyword_size(forest, idx, spec, fonts, pw_cache, mcw, bufs, inner, avail)
        }
        SpecLen::FitContent => {
            // Min first: the preferred walk below reuses `mcw` for
            // nested measurements, so the min read must land before it.
            let mn = flex_basis_keyword_size(
                forest,
                idx,
                SpecLen::MinContent,
                fonts,
                pw_cache,
                mcw,
                bufs,
                inner,
                avail,
            );
            let mx = flex_basis_keyword_size(
                forest,
                idx,
                SpecLen::MaxContent,
                fonts,
                pw_cache,
                mcw,
                bufs,
                inner,
                avail,
            );
            mx.min(mn.max(avail))
        }
        _ => {
            debug_assert!(false, "flex_basis_content_width called with {spec:?}");
            0.0
        }
    }
}

/// Whether the generic read-back bypass applies to `idx`: a generic
/// (block / inline-block) box with a definite width and no replaced
/// intrinsic size. Flex containers, multicol blocks, inline-tables
/// and inline-grids are not generic here even when their box type
/// matches: their own intrinsic algorithms define the measurement
/// (see [`flex_basis_content_width`]). Tables and grids never match
/// the box-type arm in the first place.
fn flex_basis_bypass_applies(forest: &Forest, idx: usize) -> bool {
    matches!(
        forest.nodes[idx].box_type,
        BoxType::Block | BoxType::AnonymousBlock | BoxType::InlineBlock
    ) && definite_specified_content_width(forest, idx).is_some()
        && forest.nodes[idx].intrinsic.is_none()
        && flex_kind(forest, idx).is_none()
        && multicol_intrinsic_params(forest, idx).is_none()
        && !matches!(
            forest.nodes[idx].data.display,
            Display::InlineTable | Display::InlineGrid
        )
}

/// One keyword (`MinContent` or `MaxContent`) of
/// [`flex_basis_content_width`], dispatched by shape: the generic
/// read-back, the flex intrinsic algorithm, the multicol transform,
/// or the plain dispatched value for everything else.
#[allow(clippy::too_many_arguments)]
fn flex_basis_keyword_size(
    forest: &mut Forest,
    idx: usize,
    spec: SpecLen,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
    avail: f32,
) -> f32 {
    if flex_basis_bypass_applies(forest, idx) {
        return match spec {
            SpecLen::MinContent => flex_basis_raw_min(forest, idx, fonts, mcw, bufs),
            SpecLen::MaxContent => {
                flex_basis_raw_max(forest, idx, fonts, pw_cache, mcw, bufs, inner)
            }
            _ => {
                debug_assert!(false, "flex_basis_keyword_size called with {spec:?}");
                0.0
            }
        };
    }
    // w77 (replaced): a replaced box with a definite width and an
    // intrinsic size measures its intrinsic natural size for a
    // content-keyword basis, not the dispatched definite-width
    // floor — `<img width:100px>` + `flex-basis:max-content` over
    // a 300px natural size bases at 300. `intrinsic_content_width`
    // is already content-box (border-box strips border/padding)
    // and min/max-clamped, matching the other arms' return terms.
    // Gated on a definite width so the auto-width path — whose
    // dispatched value is already intrinsic — is untouched.
    if definite_specified_content_width(forest, idx).is_some()
        && forest.nodes[idx].intrinsic.is_some()
        && let Some(intrinsic) = intrinsic_content_width(forest, idx)
    {
        return intrinsic;
    }
    if flex_kind(forest, idx).is_some() {
        return flex_basis_flex_size(forest, idx, spec, fonts, pw_cache, mcw, bufs, inner);
    }
    if let Some(cols) = multicol_intrinsic_params(forest, idx) {
        return flex_basis_multicol_size(forest, idx, &cols, spec, fonts, mcw, bufs);
    }
    // w76fix finding 1 (tables): the dispatched value floors a
    // definite-width table at its width (`table_pref_width`'s fast
    // path, pinned by `table_pref_width_definite_width_contributes_exactly`),
    // so a keyword basis over a definite width must measure the
    // columns directly instead: maxima plus spacing for max-content
    // (the same sum the infinite-base substitution uses), minima
    // plus spacing for min-content. Fixed-layout and vertical
    // tables have no column-based size here (`None` from the grid
    // helper / the fixed-vertical gate below) and fall through to
    // the dispatched value, as before. Auto-width tables never reach
    // this arm — their dispatched value is already column-based —
    // so only the definite-width hole changes behavior. (An
    // in-flow `inline-table` flex item DOES arrive here since w77g
    // blockification: it is promoted to a `Table` box carrying its
    // own style, so it is the item (not a style-less wrapper) and
    // its own basis reaches this column arm; the `display` arm
    // additionally covers nodes measured before promotion.)
    if definite_specified_content_width(forest, idx).is_some()
        && (forest.nodes[idx].box_type == BoxType::Table
            || forest.nodes[idx].data.display == Display::InlineTable)
    {
        match spec {
            SpecLen::MaxContent => {
                if let Some(g) = layout_table::table_auto_grid_max_width(
                    forest, idx, fonts, pw_cache, mcw, bufs, inner,
                ) {
                    return g;
                }
            }
            SpecLen::MinContent => {
                if let Some(m) = flex_basis_table_min(forest, idx, fonts) {
                    return m;
                }
            }
            _ => {}
        }
    }
    // w77 (grids, mirroring the table arm above): the dispatched
    // value floors a definite-width grid at its width
    // (`grid_pref_width`'s fast path), so a max-content basis over
    // a definite width must measure the tracks directly instead,
    // via the definite-ignoring variant in `layout_grid.rs` (same
    // track computation, fast path skipped, min/max-width clamp
    // kept). Only max-content needs the arm: the min-content
    // dispatch (`grid_min_width`) has no definite fast path, so a
    // min-content basis is already track-based, and fit-content
    // reaches this arm through its max-content half. Replaced
    // grids never reach here (the replaced arm above returns
    // first), and auto-width grids never need it (their dispatched
    // value is already track-based) — only the definite-width
    // hole changes behavior. Ordinary layout is untouched:
    // `grid_pref_width` keeps its fast path.
    // w77fix: a `width: min-content` grid needs the arm too.
    // `definite_specified_content_width` returns `None` for
    // `MinContent` (`SpecLen::resolve` never resolves it), so the
    // gate above never fired for this shape — and the dispatched
    // `grid_pref_width` MinContent fast path returns the min walk
    // (120), not the track max (252) the max-content basis needs.
    // The ignoring variant skips that fast path as well, so the
    // gate is extended to the min-content-width case for
    // max-content bases only.
    if (definite_specified_content_width(forest, idx).is_some()
        || matches!(forest.nodes[idx].data.width, SpecLen::MinContent))
        && matches!(spec, SpecLen::MaxContent)
        && (forest.nodes[idx].box_type == BoxType::Grid
            || forest.nodes[idx].data.display == Display::InlineGrid)
    {
        return layout_grid::grid_pref_width_ignoring_definite(forest, idx, fonts, pw_cache);
    }
    content_based_width(forest, idx, spec, fonts, pw_cache, mcw, bufs, inner, avail)
}

/// Column-based min-content of a definite-width table for a
/// content-keyword `flex-basis` (w76fix finding 1): the min walk
/// sums column minima only, so the separate-borders spacing total
/// is added, mirroring `table_pref_width`'s min-content fast path.
/// Returns `None` for fixed-layout and vertical tables, which keep
/// the dispatched value.
/// Spacing belongs to the column side only (w76fix2 M2): the
/// block-level table floor is `(min_w + spacing_total).max(capmin_c)`,
/// so a caption-dominated table must floor at CAPMIN alone — adding
/// spacing after the max overstated it by one spacing total.
fn flex_basis_table_min(forest: &mut Forest, idx: usize, fonts: &dyn FontMetrics) -> Option<f32> {
    let mut pool = layout_table::TableGridPool::new();
    let mut grid = pool.take_grid();
    layout_table::collect_table_grid_into(forest, idx, &mut pool, &mut grid);
    if grid.fixed || grid.vertical {
        pool.put_grid(grid);
        return None;
    }
    let (_, spacing_h, _) = layout_table::table_border_model(forest, idx);
    let nvis = grid.collapsed_cols.iter().filter(|&&c| !c).count();
    let spacing_total = if nvis == 0 {
        0.0
    } else {
        spacing_h * (nvis as f32 + 1.0)
    };
    pool.put_grid(grid);
    let (cols, capmin_c) = table_min_content_split(forest, idx, fonts, &mut pool);
    Some((cols + spacing_total).max(capmin_c))
}

/// Intrinsic size of a flex container for one `flex-basis` content
/// keyword, with the definite-width floor dropped (see
/// [`flex_basis_content_width`]): the flex algorithm's own sum/max
/// over the children's walked sizes. The populating walk runs first
/// (min-content into `mcw`, preferred into `pref_w`); the intrinsic
/// computation itself runs on the `WalkBufs`-resident [`FlexScratch`],
/// exactly like the dispatch walks' flex arms, so no fresh
/// allocation lands on this path. `spec` is `MinContent` or
/// `MaxContent` only.
#[allow(clippy::too_many_arguments)]
fn flex_basis_flex_size(
    forest: &mut Forest,
    idx: usize,
    spec: SpecLen,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
) -> f32 {
    let mode = match spec {
        SpecLen::MinContent => IntrinsicMode::Min,
        SpecLen::MaxContent => IntrinsicMode::Max,
        _ => {
            debug_assert!(false, "flex_basis_flex_size called with {spec:?}");
            return 0.0;
        }
    };
    match mode {
        IntrinsicMode::Min => {
            let _ = min_content_width_pooled(forest, idx, fonts, mcw, bufs);
        }
        IntrinsicMode::Max => {
            let _ = preferred_width_mcw(forest, idx, fonts, pw_cache, mcw, bufs, inner);
        }
    }
    let flex = bufs
        .flex
        .get_or_insert_with(|| Box::new(FlexScratch::new(forest.nodes.len())));
    flex.begin_node();
    // The closure reads the outer walk's slots (the populating walk
    // above), while the intrinsic computation's own buffers live in
    // the scratch — disjoint, so no borrow conflict.
    match mode {
        IntrinsicMode::Min => flex_intrinsic_width(
            forest,
            idx,
            IntrinsicMode::Min,
            fonts,
            &mut flex.pw,
            &mut flex.aux,
            &mut flex.bufs,
            &mut flex.inner,
            |f, c| mcw.get(c).copied().unwrap_or(0.0) + intrinsic_h_extras(f, c),
        ),
        IntrinsicMode::Max => flex_intrinsic_width(
            forest,
            idx,
            IntrinsicMode::Max,
            fonts,
            &mut flex.pw,
            &mut flex.aux,
            &mut flex.bufs,
            &mut flex.inner,
            |f, c| f.nodes[c].pref_w + intrinsic_h_extras(f, c),
        ),
    }
}

/// Intrinsic size of a multicol container for one `flex-basis`
/// content keyword, with the definite-width floor dropped (see
/// [`flex_basis_content_width`]): the column transform over the
/// single-column content measure. `spec` is `MinContent` or
/// `MaxContent` only.
#[allow(clippy::too_many_arguments)]
fn flex_basis_multicol_size(
    forest: &mut Forest,
    idx: usize,
    cols: &IntrinsicCols,
    spec: SpecLen,
    fonts: &dyn FontMetrics,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
) -> f32 {
    match spec {
        SpecLen::MinContent => {
            let _ = min_content_width_pooled(forest, idx, fonts, mcw, bufs);
            let content = forest.nodes[idx]
                .children
                .iter()
                .filter(|&&c| !layout_position::is_out_of_flow(forest, c))
                .map(|&c| mcw.get(c).copied().unwrap_or(0.0) + intrinsic_h_extras(forest, c))
                .fold(0.0f32, f32::max);
            multicol_min_content_width(cols, content)
        }
        SpecLen::MaxContent => {
            let content = multicol_content_min_width(forest, idx, fonts, mcw, bufs);
            let in_flow = forest.nodes[idx]
                .children
                .iter()
                .filter(|&&c| !layout_position::is_out_of_flow(forest, c))
                .count();
            multicol_max_content_width(cols, content, in_flow)
        }
        _ => {
            debug_assert!(false, "flex_basis_multicol_size called with {spec:?}");
            0.0
        }
    }
}

/// Unclamped min-content of an item for a content-keyword
/// `flex-basis`, ignoring the definite-width floor (w76fix finding
/// 2): the generic read-back when the bypass applies, else the raw
/// (unclamped) pooled min-content. The automatic minimum (flexbox
/// §4.5) is the min-content *size*, and a `max-width` does not bound
/// it — so this never routes through the clamped pooled dispatch.
fn keyword_min_content_unclamped(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
) -> f32 {
    if flex_basis_bypass_applies(forest, idx) {
        flex_basis_raw_min(forest, idx, fonts, mcw, bufs)
    } else if definite_specified_content_width(forest, idx).is_some()
        && (forest.nodes[idx].box_type == BoxType::Table
            || forest.nodes[idx].data.display == Display::InlineTable)
        && let Some(m) = flex_basis_table_min(forest, idx, fonts)
    {
        // Same column-based minimum as the keyword base (w76fix
        // finding 1): without it the automatic minimum would sit
        // one spacing total below the base for definite-width
        // tables. Gated on definite width like the base arm, so
        // auto-width tables keep the pre-existing spacing-less
        // pooled value on both sides.
        m
    } else {
        min_content_width_pooled_raw(forest, idx, fonts, mcw, bufs)
    }
}

/// Raw min-content of a generic box: the max over its in-flow
/// children of their min-content plus extras, mirroring the generic
/// arm of `min_content_width_pooled_impl` (an inline child's own
/// slot already sums its run). Read immediately after the walk,
/// before any preferred-width walk can reuse the same scratch.
fn flex_basis_raw_min(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
) -> f32 {
    let _ = min_content_width_pooled(forest, idx, fonts, mcw, bufs);
    forest.nodes[idx]
        .children
        .iter()
        .filter(|&&c| !layout_position::is_out_of_flow(forest, c))
        .map(|&c| mcw.get(c).copied().unwrap_or(0.0) + intrinsic_h_extras(forest, c))
        .fold(0.0f32, f32::max)
}

/// Raw max-content of a generic box: the children's preferred widths
/// via [`generic_preferred_width`], after the preferred walk has
/// populated them (the walk itself floors the root at its definite
/// width, which is exactly what this read-back discards).
#[allow(clippy::too_many_arguments)]
fn flex_basis_raw_max(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
) -> f32 {
    let _ = preferred_width_mcw(forest, idx, fonts, pw_cache, mcw, bufs, inner);
    generic_preferred_width(forest, idx)
}

/// Finite flex base for a flex item that is a chain of single
/// block-level wrappers around one automatic-layout table (w76).
/// A wrapper's generic preferred width propagates the table's
/// infinite preferred width unchanged, so the item's measured base
/// is infinite and would saturate to `MAX_USED_SIZE` in
/// `flex_lengths`. Walking the single-child chain to the table and
/// substituting [`layout_table::table_auto_grid_max_width`] plus
/// the accumulated extras (each wrapper's, then the table's own
/// border/padding/margins) gives the item's content-box base: the
/// table's grid sum plus wrapper extras. The grid-max call itself
/// keeps the fixed/vertical exclusions and reuses the
/// `PrefWidthCache` grid-max memo (generation-stamped, so no stale
/// cross-table value can leak in); this walk never bypasses it.
/// Deliberately narrow: only single in-flow-child block wrappers
/// (no flex/grid/multicol intermediate, whose sizing is not the
/// generic run/max), and it never touches
/// `preferred_width_mcw`'s block arm — block/float/grid sizing
/// outside flex items is out of scope. Returns `None` for any other
/// shape, leaving the base on the saturated path.
/// Narrower still (w76fix): the item itself must be generic (a
/// flex-container or multicol item owns its base through its own
/// intrinsic algorithm); the scan skips whitespace-only children
/// (pretty-printed wrappers leave empty/whitespace anonymous blocks
/// around the table); and any floated box, or any intermediate with
/// a definite/percentage width or a min/max that would cap its fill,
/// bails out — `None` (saturate) is safer than a finite base the
/// wrapper would not actually fill.
#[allow(clippy::too_many_arguments)]
fn flex_wrapper_table_grid_max(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
) -> Option<f32> {
    // The item itself must be a generic box: a flex-container,
    // multicol, grid or table item gets its base from its own
    // intrinsic algorithm, never from this substitution.
    if flex_kind(forest, idx).is_some()
        || multicol_intrinsic_params(forest, idx).is_some()
        || matches!(forest.nodes[idx].box_type, BoxType::Grid | BoxType::Table)
    {
        return None;
    }
    let mut acc = 0.0f32;
    let mut cur = idx;
    loop {
        let mut sole: Option<usize> = None;
        for &child in &forest.nodes[cur].children {
            if layout_position::is_out_of_flow(forest, child) {
                continue;
            }
            // Pretty-printed wrappers carry whitespace-only children
            // (empty anonymous blocks, or anonymous blocks wrapping a
            // newline run) around the table; they are not real
            // siblings and must not break the sole-child chain.
            if layout_table::is_ws_only(forest, child) {
                continue;
            }
            if sole.is_some() {
                return None;
            }
            sole = Some(child);
        }
        let child = sole?;
        if matches!(
            forest.nodes[child].data.display,
            Display::Table | Display::InlineTable
        ) {
            // A floated terminal table sizes shrink-to-fit, not to
            // the wrapper fill the substitution assumes.
            if forest.nodes[child].data.float != Float::None {
                return None;
            }
            let g = layout_table::table_auto_grid_max_width(
                forest, child, fonts, pw_cache, mcw, bufs, inner,
            )?;
            return Some(g + acc + intrinsic_h_extras(forest, child));
        }
        if !matches!(
            forest.nodes[child].box_type,
            BoxType::Block | BoxType::AnonymousBlock
        ) || flex_kind(forest, child).is_some()
            || multicol_intrinsic_params(forest, child).is_some()
        {
            return None;
        }
        // Intermediates must actually fill the wrapper: a float, a
        // definite/percentage width, or a min/max that would cap the
        // fill all break the "wrapper = grid + extras" identity.
        let d = &forest.nodes[child].data;
        if d.float != Float::None
            || spec_is_definite_or_percentage(d.width)
            || spec_caps_min_fill(d.min_width)
            || !matches!(d.max_width, SpecLen::Auto)
        {
            return None;
        }
        acc += intrinsic_h_extras(forest, child);
        cur = child;
    }
}

/// A width spec that pins an intermediate wrapper instead of letting
/// it fill: any statically resolvable length or percentage. `auto`,
/// the content keywords, and `stretch` (which is itself a fill) do
/// not pin.
fn spec_is_definite_or_percentage(spec: SpecLen) -> bool {
    !matches!(
        spec,
        SpecLen::Auto
            | SpecLen::FitContent
            | SpecLen::MinContent
            | SpecLen::MaxContent
            | SpecLen::Stretch
    )
}

/// A `min-width` spec that would cap an intermediate wrapper's fill.
/// `auto` never caps, and an explicit zero never binds; anything
/// else (lengths, percentages, content keywords) can hold the
/// wrapper above the substituted size.
fn spec_caps_min_fill(spec: SpecLen) -> bool {
    !matches!(spec, SpecLen::Auto) && !matches!(spec, SpecLen::Px(0.0))
}

/// Resolve a `min-width`/`max-width` spec that may itself be a
/// content-based keyword (css-sizing-3 §4): `SpecLen::resolve` returns
/// `None` for those, which clamps silently default to 0/∞ (F4), so
/// measure here where the font context exists. Returns `None` for
/// `auto` and statically unresolvable specs so the caller keeps its own
/// default. The returned value is a *content-box* width, matching
/// [`content_based_width`]; `avail` is the available content-box width
/// (see its docs). `bufs`/`inner` are the pooled walk buffers (see
/// [`WalkBufs`]), threaded through to the nested measurements.
/// Keyword measurements route through [`flex_basis_content_width`]
/// (w76fix finding 8), not the raw dispatches: a definite width must
/// not floor a keyword min/max either — `width:100px` +
/// `min-width:min-content` over a 60px longest word is 60, not 100.
#[allow(clippy::too_many_arguments)]
fn measured_minmax_width(
    forest: &mut Forest,
    idx: usize,
    spec: SpecLen,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
    avail: f32,
) -> Option<f32> {
    match spec {
        SpecLen::MinContent | SpecLen::MaxContent | SpecLen::FitContent => Some(
            flex_basis_content_width(forest, idx, spec, fonts, pw_cache, mcw, bufs, inner, avail)
                .max(0.0),
        ),
        _ => None,
    }
}

/// Clamp a measured content width by measured `min-width`/`max-width`
/// values (`None` = not a content-based keyword, skip). Mirrors
/// `resolve_metrics`' min-wins `clamp_w`: when both constrain,
/// min-width wins.
fn clamp_measured(w: f32, lo: Option<f32>, hi: Option<f32>) -> f32 {
    match (lo, hi) {
        (Some(l), Some(h)) => w.min(h).max(l),
        (Some(l), None) => w.max(l),
        (None, Some(h)) => w.min(h),
        (None, None) => w,
    }
}

/// Margin-box width a content-based-width BFC root would occupy beside
/// floats given the narrowed available width `avail` (F3): the
/// fill-provisional `used_w` from `resolve_metrics` is wrong for these
/// specs, so measure against the narrowed width instead. `ml`/`mr` are
/// the box's used side margins and `extras` its horizontal
/// border+padding (from the provisional metrics); css-sizing-3 §4
/// min/max keywords clamp the measured width, min winning.
#[allow(clippy::too_many_arguments)]
fn bfc_content_fit_width(
    forest: &mut Forest,
    idx: usize,
    spec: SpecLen,
    ml: f32,
    mr: f32,
    extras: f32,
    min_spec: SpecLen,
    max_spec: SpecLen,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
    avail: f32,
) -> f32 {
    let content_avail = (avail - ml - mr - extras).max(0.0);
    let w = content_based_width(
        forest,
        idx,
        spec,
        fonts,
        pw_cache,
        mcw,
        bufs,
        inner,
        content_avail,
    );
    let lo = measured_minmax_width(
        forest,
        idx,
        min_spec,
        fonts,
        pw_cache,
        mcw,
        bufs,
        inner,
        content_avail,
    );
    let hi = measured_minmax_width(
        forest,
        idx,
        max_spec,
        fonts,
        pw_cache,
        mcw,
        bufs,
        inner,
        content_avail,
    );
    ml + mr + extras + clamp_measured(w, lo, hi)
}

/// The box's own horizontal border+padding total (used widths), for the
/// box-sizing strip in [`definite_specified_content_width`],
/// [`intrinsic_content_width`], and [`flex_item_intrinsic_contribution`]
/// (`flex-basis` never covers the item's margins, so the intrinsic
/// strip must exclude them too). `border-style: none | hidden` forces the
/// used border width to zero (CSS 2.1 §8.5.3), whatever `border-width`
/// says.
pub(crate) fn own_horizontal_bp(forest: &Forest, i: usize) -> f32 {
    let n = &forest.nodes[i];
    let d = &n.data;
    let len0 = |s: SpecLen| {
        s.resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            0.0,
        )
        .unwrap_or(0.0)
    };
    let bw = |w: SpecLen, st: BorderStyle| {
        if st.zeroes_width() { 0.0 } else { len0(w) }
    };
    bw(d.border[3], d.border_style[3])
        + len0(d.padding[3])
        + len0(d.padding[1])
        + bw(d.border[1], d.border_style[1])
}

/// The box's own vertical border+padding total (used widths): the
/// inline-axis counterpart of [`own_horizontal_bp`] for vertical tables,
/// whose inline (algorithm-width) axis is vertical.
pub(crate) fn own_vertical_bp(forest: &Forest, i: usize) -> f32 {
    let n = &forest.nodes[i];
    let d = &n.data;
    let len0 = |s: SpecLen| {
        s.resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            0.0,
        )
        .unwrap_or(0.0)
    };
    let bw = |w: SpecLen, st: BorderStyle| {
        if st.zeroes_width() { 0.0 } else { len0(w) }
    };
    bw(d.border[0], d.border_style[0])
        + len0(d.padding[0])
        + len0(d.padding[2])
        + bw(d.border[2], d.border_style[2])
}

/// Resolve a `min-width`/`max-width` spec in the box-sizing's own terms
/// (border-box width under `border-box`, content width under
/// `content-box`); `none` is the initial-value fallback (`0.0` for min,
/// `INFINITY` for max). Percentage expressions are cyclic here and
/// behave as `none`/`auto`; a bare percentage is cyclic for the same
/// reason (CSS 2.1 §10.2) and never resolves against the zero base. The
/// syntactic-cyclic gate (`had_pct`) is skipped for replaced elements,
/// whose intrinsic size anchors percentage resolution instead (see
/// [`definite_specified_content_width`]).
pub(crate) fn minmax_constraint(forest: &Forest, i: usize, s: SpecLen, none: f32) -> f32 {
    let n = &forest.nodes[i];
    let d = &n.data;
    match s {
        SpecLen::Math { had_pct, .. } if had_pct && n.intrinsic.is_none() => none,
        SpecLen::Pct(_) if n.intrinsic.is_none() => none,
        w => w
            .resolve(
                d.font_size,
                forest.root_font_size,
                forest.vp_w,
                forest.vp_h,
                0.0,
            )
            .unwrap_or(none),
    }
}

/// Intrinsic contribution of a replaced box with no definite specified
/// width (css-sizing-3 §5.2): its intrinsic width, clamped by
/// `min-width`/`max-width` exactly like a specified width (§4: min wins
/// when over-constrained), with the same box-sizing strip. Returns
/// `None` for non-replaced boxes. Callers consult this when
/// [`definite_specified_content_width`] returns `None`, so a childless
/// replaced box (whose `intrinsic` is set at box build) still
/// contributes its intrinsic size to ancestor measurements: grid
/// content-sized tracks, flex item intrinsic sums, table column minima,
/// and nested shrink-to-fit.
pub(crate) fn intrinsic_content_width(forest: &Forest, i: usize) -> Option<f32> {
    let intrinsic_w = forest.nodes[i].intrinsic?.width_or_default().max(0.0);
    let d = &forest.nodes[i].data;
    let min_w = minmax_constraint(forest, i, d.min_width, 0.0).max(0.0);
    let max_w = minmax_constraint(forest, i, d.max_width, f32::INFINITY).max(0.0);
    let clamped = intrinsic_w.min(max_w).max(min_w);
    Some(match d.box_sizing {
        BoxSizing::BorderBox => (clamped - own_horizontal_bp(forest, i)).max(0.0),
        BoxSizing::ContentBox => clamped.max(0.0),
    })
}

/// Returns `Some(content_w)` when `width` is definite, `None` when it
/// is `auto` or cyclic. A preferred size property specified as an
/// expression containing a percentage is cyclic for non-replaced boxes,
/// so the entire value behaves as the initial value (`auto`) per
/// §5.2.1 rule 1; pure percentages resolve against a zero base, i.e.
/// they behave as `auto` (§5.2.1). Cyclic-ness is syntactic (CSS Values
/// 4 §8.1.4): the gate checks the `had_pct` bit, not the folded
/// coefficient, so even `calc(10px + 0%)` — which folds to
/// `Math { pct: 0.0, px: 10.0 }` — behaves as `auto`. The returned
/// value is a content-box width (the box's own border/padding are
/// stripped under `border-box`), clamped by `min-width`/`max-width` in
/// the box-sizing's own terms, with `min-width` winning when
/// over-constrained.
pub(crate) fn definite_specified_content_width(forest: &Forest, i: usize) -> Option<f32> {
    let n = &forest.nodes[i];
    let d = &n.data;
    // A vertical table's inline (algorithm-width) size resolves from the
    // physical `height`/`min-height`/`max-height`, with the vertical
    // border/padding as the inline box-sizing extras.
    let vwm_table = is_vertical_table(forest, i);
    let width_spec = if vwm_table { d.height } else { d.width };
    // CSS Values 4 §8.1.4 syntactic cyclic-ness: a `calc()` that
    // contained a percentage leaf is cyclic against an indefinite base
    // even when it folds to `pct: 0.0` (e.g. `calc(10px + 0%)`), so the
    // gate checks the `had_pct` bit, not the folded coefficient. A bare
    // percentage is likewise cyclic against an indefinite base and
    // behaves as `auto` (CSS 2.1 §10.2), never resolving against the
    // zero base here.
    let specified = match width_spec {
        SpecLen::Math { had_pct, .. } if had_pct && n.intrinsic.is_none() => return None,
        SpecLen::Pct(_) if n.intrinsic.is_none() => return None,
        w => w.resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            0.0,
        ),
    }?;
    let own_bp = if vwm_table {
        own_vertical_bp(forest, i)
    } else {
        own_horizontal_bp(forest, i)
    };
    // min/max-width clamp the width in the box-sizing's own terms
    // (border-box width under `border-box`, content width under
    // `content-box`). Percentage expressions are cyclic here and behave
    // as `none`/`auto`; a bare percentage is cyclic for the same reason
    // (CSS 2.1 §10.2) and never resolves against the zero base.
    let min_w = minmax_constraint(
        forest,
        i,
        if vwm_table { d.min_height } else { d.min_width },
        0.0,
    )
    .max(0.0);
    let max_w = minmax_constraint(
        forest,
        i,
        if vwm_table { d.max_height } else { d.max_width },
        f32::INFINITY,
    )
    .max(0.0);
    let clamped = specified.min(max_w).max(min_w);
    // The caller works in content-box terms (the box's own extras ride
    // along via the caller), so strip the box's own border/padding
    // under `border-box`: the outer contribution is then exactly
    // specified_border_box.
    Some(match d.box_sizing {
        BoxSizing::BorderBox => (clamped - own_bp).max(0.0),
        BoxSizing::ContentBox => clamped.max(0.0),
    })
}

/// Returns `Some(content_w)` when the physical `width` is definite.
/// Unlike [`definite_specified_content_width`], this never transposes
/// for vertical tables: it resolves the physical `width`/`min-width`/
/// `max-width` with the horizontal border/padding. The table preferred-
/// width consumers ([`table_pref_adjusted`], [`table_pref_width`]) need
/// the physical width, not the inline (transposed) size.
pub(crate) fn definite_specified_physical_width(forest: &Forest, i: usize) -> Option<f32> {
    let n = &forest.nodes[i];
    let d = &n.data;
    let specified = match d.width {
        SpecLen::Math { had_pct, .. } if had_pct && n.intrinsic.is_none() => return None,
        SpecLen::Pct(_) if n.intrinsic.is_none() => return None,
        w => w.resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            0.0,
        ),
    }?;
    let own_bp = own_horizontal_bp(forest, i);
    let min_w = minmax_constraint(forest, i, d.min_width, 0.0).max(0.0);
    let max_w = minmax_constraint(forest, i, d.max_width, f32::INFINITY).max(0.0);
    let clamped = specified.min(max_w).max(min_w);
    Some(match d.box_sizing {
        BoxSizing::BorderBox => (clamped - own_bp).max(0.0),
        BoxSizing::ContentBox => clamped.max(0.0),
    })
}

/// Clamp a content-based intrinsic width by the box's `min-width` /
/// `max-width`, mirroring [`definite_specified_content_width`]'s
/// box-sizing-aware clamp: the constraints apply in the box-sizing's
/// own terms (border-box width under `border-box`, content width under
/// `content-box`), and `min-width` wins when over-constrained.
/// Percentage constraints are cyclic here and behave as `none` /
/// `auto` (see [`minmax_constraint`]).
///
/// The content-based `unwrap_or` fallbacks in [`definite_width_floor`]
/// and [`min_content_width_pooled`] report raw content widths; without
/// this clamp a `max-width` on an `auto`-width block would not bound
/// its intrinsic contribution the way it bounds flex items (see
/// [`flex_item_intrinsic_contribution`] via [`clamp_flex_main`]) and
/// the way [`definite_specified_content_width`] /
/// [`intrinsic_content_width`] clamp their own results.
pub(crate) fn clamp_content_width_by_minmax(forest: &Forest, i: usize, content_w: f32) -> f32 {
    let d = &forest.nodes[i].data;
    let min_w = minmax_constraint(forest, i, d.min_width, 0.0).max(0.0);
    let max_w = minmax_constraint(forest, i, d.max_width, f32::INFINITY).max(0.0);
    match d.box_sizing {
        BoxSizing::BorderBox => {
            let bp = own_horizontal_bp(forest, i);
            ((content_w + bp).min(max_w).max(min_w) - bp).max(0.0)
        }
        BoxSizing::ContentBox => content_w.min(max_w).max(min_w).max(0.0),
    }
}

/// A definite specified width floors the intrinsic contribution of the
/// box types the `width` property sizes (css-sizing-3 §5.2): block boxes
/// contribute their definite width outright (content overflowing it does
/// not grow the contribution — symmetric with the `InlineBlock` arms),
/// while a table cell's specified width floors per CSS 2.1 §17.5.2.2
/// (`max(MCW, W)`). Internal table boxes other than cells
/// (`table-row`, `table-row-group`, `table-column`,
/// `table-column-group`) ignore `width` (CSS 2.1 §10.2), so they keep
/// the generic content-based sizing; tables and grids have their own
/// sizing arms and never reach this helper. A replaced box with `width:
/// auto` contributes its intrinsic width (css-sizing-3 §5.2) via
/// [`intrinsic_content_width`] instead of the (empty) content-based
/// size.
///
/// F7: `width: min-content` sizes the box to its min-content size, so
/// that — not the generic max-content — is the box's intrinsic
/// contribution to its parent.
fn definite_width_floor(
    forest: &mut Forest,
    i: usize,
    content_based: f32,
    fonts: &dyn FontMetrics,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
) -> f32 {
    if matches!(forest.nodes[i].data.width, SpecLen::MinContent) {
        return min_content_width_pooled(forest, i, fonts, mcw, bufs);
    }
    let definite = match forest.nodes[i].box_type {
        BoxType::Block | BoxType::AnonymousBlock => {
            return definite_specified_content_width(forest, i)
                .or_else(|| intrinsic_content_width(forest, i))
                .or_else(|| aspect_ratio_intrinsic_width(forest, i))
                .unwrap_or_else(|| clamp_content_width_by_minmax(forest, i, content_based));
        }
        BoxType::TableCell => definite_specified_content_width(forest, i),
        _ => {
            return intrinsic_content_width(forest, i)
                .or_else(|| aspect_ratio_intrinsic_width(forest, i))
                .unwrap_or_else(|| clamp_content_width_by_minmax(forest, i, content_based));
        }
    };
    match definite {
        Some(d) => d.max(content_based),
        None => intrinsic_content_width(forest, i)
            .or_else(|| aspect_ratio_intrinsic_width(forest, i))
            .unwrap_or_else(|| clamp_content_width_by_minmax(forest, i, content_based)),
    }
}

/// Which intrinsic size an intrinsic dispatch is measuring.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum IntrinsicMode {
    Min,
    Max,
}

/// Intrinsic width of a flex container from its items' intrinsic
/// contributions (css-flexbox-1 §9.9 / css-sizing-3): for a row
/// container, the sum of the items' contributions plus the column
/// gaps between them; for a column container, the widest item's
/// contribution (the main axis is vertical, so gaps contribute
/// nothing to the width) — except a column-wrap container with a
/// definite height, whose items break into columns and whose width
/// is the columns' sum plus cross gaps (see
/// [`flex_column_wrap_width`]). Items are the grouped flex items,
/// not the raw children: consecutive inline-level children coalesce
/// into one anonymous item, as in [`wrap_flex_text_runs`].
/// `child_w` resolves one child's contribution
/// in the caller's terms — max-content (`pref_w`) for
/// [`preferred_width_mcw`], min-content for [`min_content_width_pooled`]
/// — reusing the same per-item intrinsic sizing both dispatches already
/// computed for the children, mirroring the item-sum the direct
/// `inline-flex` shrink-to-fit pass in [`layout_flex`] performs over
/// laid-out items. Out-of-flow children are not flex items and are
/// skipped, as in `flex_build_items`. Percentage gaps resolve to zero
/// here: their base is the container's own indefinite intrinsic size
/// (css-sizing-3), so `flex_gaps` is called with no percentage bases.
///
/// `mode` selects which intrinsic size is being measured. The two
/// differ for a wrapping row container: its max-content is still the
/// single-line sum (against an indefinite base no line ever breaks),
/// but its min-content is the widest single item's min-content —
/// lines may break between every pair of items, so neither the other
/// items nor the inter-item gaps contribute to the minimum width.
///
/// Each item's raw `child_w` contribution is adjusted by
/// [`flex_item_intrinsic_contribution`] (flex-basis and min/max
/// clamping) before it is summed/maxed here. The raw widths are
/// collected first, under a shared borrow, because the adjustment
/// needs `&mut Forest` (it reuses [`clamp_flex_main`]).
#[allow(clippy::too_many_arguments)]
fn flex_intrinsic_width(
    forest: &mut Forest,
    i: usize,
    mode: IntrinsicMode,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
    child_w: impl Fn(&Forest, usize) -> f32,
) -> f32 {
    // Container facts are copied out up front: the per-item adjustment
    // below needs `&mut Forest`, so no borrow of the container's style
    // may live across it.
    let (horizontal, wrap, main_gap, col_gaps, container_h) = {
        let style = forest.nodes[i].style.as_deref();
        let d = &forest.nodes[i].data;
        let bases = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
        let horizontal = matches!(flex_dir_of(style), FlexDir::Row | FlexDir::RowRev);
        let (main_gap, _) = flex_gaps(style, true, bases, None, None);
        // Column direction: `(row-gap, column-gap)` in main/cross
        // terms, for the column-wrap simulation below. Its percentage
        // main gap resolves against the definite container height
        // when there is one; the cross gap's base (the width) is the
        // indefinite size being measured, so it stays `None`.
        let container_h = flex_container_definite_content_height(forest, i);
        let col_gaps = flex_gaps(style, false, bases, container_h, None);
        (
            horizontal,
            flex_wrap_of(style),
            main_gap,
            col_gaps,
            container_h,
        )
    };
    // Group the raw children into flex items the way
    // `wrap_flex_text_runs` will when layout reaches this container:
    // each maximal run of consecutive inline-level children is ONE
    // anonymous item, not one item per fragment. Measuring the raw
    // children instead counted a phantom column gap per seam in
    // max-content and summed the fragments' longest words in
    // min-content. Block-level children are single items;
    // out-of-flow children are not items and break a run, as in
    // `wrap_flex_text_runs`.
    struct RawItem {
        /// First (or only) child of the item: the representative for
        /// `order` and for the column-wrap main-size estimate.
        rep: usize,
        /// Raw outer contributions of the item's parts, in order.
        parts: Vec<f32>,
        /// Whether this item is a coalesced inline run.
        run: bool,
    }
    let mut items: Vec<RawItem> = Vec::new();
    // Whether the previous raw child continues a run: block-level and
    // out-of-flow children both break one, as in `wrap_flex_text_runs`.
    let mut run_open = false;
    for &c in &forest.nodes[i].children {
        if layout_position::is_out_of_flow(forest, c) {
            run_open = false;
            continue;
        }
        let w = child_w(forest, c);
        // w78: mirror `wrap_flex_text_runs`' blockification — an
        // in-flow inline-* box is a single styled item, not part of
        // a coalesced run, so its contribution goes through
        // `flex_item_intrinsic_contribution` (basis/margins) instead
        // of the run's unstyled sum/max. (This runs before layout
        // promotes the node, so the `display` check, not the
        // post-promotion `box_type`, is what identifies it here;
        // nodes boxes pass 3 already promoted are block-level and
        // take the same arm via the first disjunct.)
        let is_blockified_item = forest.nodes[c].box_type == BoxType::InlineBlock
            && matches!(
                forest.nodes[c].data.display,
                Display::InlineBlock
                    | Display::InlineFlex
                    | Display::InlineGrid
                    | Display::InlineTable
            );
        if is_block_level(forest.nodes[c].box_type) || is_blockified_item {
            run_open = false;
            items.push(RawItem {
                rep: c,
                parts: vec![w],
                run: false,
            });
        } else if run_open {
            items
                .last_mut()
                .expect("open run has an item")
                .parts
                .push(w);
        } else {
            run_open = true;
            items.push(RawItem {
                rep: c,
                parts: vec![w],
                run: true,
            });
        }
    }
    let mut contribs: Vec<(usize, f32)> = Vec::with_capacity(items.len());
    for it in items {
        let w = if it.run {
            // One anonymous item: max-content is the sum of its
            // parts (they share one line), min-content the widest
            // part — exactly what the anonymous block the run is
            // wrapped into measures as (its min-content dispatch
            // takes the max over its direct children).
            match mode {
                IntrinsicMode::Max => it.parts.iter().sum(),
                IntrinsicMode::Min => it.parts.iter().copied().fold(0.0, f32::max),
            }
        } else {
            flex_item_intrinsic_contribution(
                forest,
                it.rep,
                mode,
                horizontal,
                it.parts[0],
                fonts,
                pw_cache,
                mcw,
                bufs,
                inner,
            )
        };
        contribs.push((it.rep, w));
    }
    let n = contribs.len();
    let mut sum = 0.0f32;
    let mut maxw = 0.0f32;
    for &(_, w) in &contribs {
        sum += w;
        maxw = maxw.max(w);
    }
    if !horizontal {
        // Column wrap with a definite container height: items break
        // into columns against that height, and the width is the sum
        // of the columns' widths plus the cross (column) gaps —
        // not the single widest item. Items whose main size needs a
        // measuring layout use the single-line estimate inside
        // `flex_column_wrap_width` so the simulation still runs.
        if wrap != FlexWrap::NoWrap
            && let Some(ch) = container_h
        {
            return flex_column_wrap_width(forest, &contribs, ch, col_gaps);
        }
        return maxw;
    }
    if n == 0 {
        return 0.0;
    }
    if mode == IntrinsicMode::Min && wrap != FlexWrap::NoWrap {
        return maxw;
    }
    sum + main_gap * (n - 1) as f32
}

/// Width of a wrapping column flex container with definite content
/// height `container_h`: pack the items (in `order`, ties in document
/// order) into columns against that height — an item starts a new
/// column when it no longer fits the current one, per flexbox §9.3
/// line collection over hypothetical main sizes — then sum the
/// columns' widest-item widths plus the cross gaps between columns.
/// `contribs` is `(representative node, width contribution)` per
/// item; `gaps` is the column direction's `(main_gap, cross_gap)`.
/// Items whose main size needs a measuring layout use the
/// single-line estimate from [`flex_column_item_main_outer`] instead
/// of aborting the simulation.
fn flex_column_wrap_width(
    forest: &Forest,
    contribs: &[(usize, f32)],
    container_h: f32,
    gaps: (f32, f32),
) -> f32 {
    let (main_gap, cross_gap) = gaps;
    let mut sized: Vec<(i32, usize, f32, f32)> = Vec::with_capacity(contribs.len());
    for (k, &(rep, w)) in contribs.iter().enumerate() {
        let main = flex_column_item_main_outer(forest, rep, container_h);
        sized.push((flex_order_of(forest, rep), k, w, main));
    }
    sized.sort_by_key(|&(order, k, _, _)| (order, k));
    let mut total = 0.0f32;
    let mut cols = 0usize;
    let mut col_w = 0.0f32;
    let mut col_h = 0.0f32;
    let mut col_items = 0usize;
    for &(_, _, w, main) in &sized {
        // One main gap stride between adjacent items in a column.
        let gap = if col_items > 0 { main_gap } else { 0.0 };
        if col_items > 0 && col_h + gap + main > container_h + 1e-6 {
            total += col_w;
            cols += 1;
            col_w = w;
            col_h = main;
            col_items = 1;
        } else {
            col_w = col_w.max(w);
            col_h += gap + main;
            col_items += 1;
        }
    }
    if col_items == 0 {
        return 0.0;
    }
    total += col_w;
    cols += 1;
    total + cross_gap * (cols - 1) as f32
}

/// The `order` of a flex item (flexbox §5.4), mirroring
/// `flex_build_items`.
fn flex_order_of(forest: &Forest, c: usize) -> i32 {
    match forest.nodes[c]
        .style
        .as_ref()
        .and_then(|s| s.get_by_index(props::ORDER))
    {
        Some(CssValue::Number(n)) => *n as i32,
        _ => 0,
    }
}

/// Vertical border+padding of one box, resolved against a zero base
/// (the vertical counterpart of the border/padding half of
/// [`intrinsic_h_extras`]).
fn intrinsic_v_bp(forest: &Forest, idx: usize) -> f32 {
    let n = &forest.nodes[idx];
    let d = &n.data;
    let (fs, rfs, vpw, vph) = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
    let len0 = |s: SpecLen| s.resolve(fs, rfs, vpw, vph, 0.0).unwrap_or(0.0);
    let bw = |w: SpecLen, st: BorderStyle| {
        if st.zeroes_width() { 0.0 } else { len0(w) }
    };
    bw(d.border[0], d.border_style[0])
        + len0(d.padding[0])
        + len0(d.padding[2])
        + bw(d.border[2], d.border_style[2])
}

/// Horizontal border+padding of one box, resolved against a zero base
/// (the border/padding half of [`intrinsic_h_extras`], factored out for
/// the table min-content CAPMIN unit conversion).
fn intrinsic_h_bp(forest: &Forest, idx: usize) -> f32 {
    let d = &forest.nodes[idx].data;
    let (fs, rfs, vpw, vph) = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
    let len0 = |s: SpecLen| s.resolve(fs, rfs, vpw, vph, 0.0).unwrap_or(0.0);
    let bw = |w: SpecLen, st: BorderStyle| {
        if st.zeroes_width() { 0.0 } else { len0(w) }
    };
    bw(d.border[3], d.border_style[3])
        + len0(d.padding[3])
        + len0(d.padding[1])
        + bw(d.border[1], d.border_style[1])
}

/// A flex container's definite content-box height, when its `height`
/// is a definite length: percentages and percentage-bearing `calc()`
/// are cyclic against the (unknown) containing-block height here and
/// make the height indefinite, as in [`specified_height`] with no
/// `cb_h`. `border-box` heights strip the container's own vertical
/// border/padding.
fn flex_container_definite_content_height(forest: &Forest, i: usize) -> Option<f32> {
    let d = &forest.nodes[i].data;
    let resolved = match d.height {
        SpecLen::Pct(_) => return None,
        SpecLen::Math { had_pct, .. } if had_pct => return None,
        spec => spec.resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            0.0,
        )?,
    };
    Some(match d.box_sizing {
        BoxSizing::BorderBox => (resolved - intrinsic_v_bp(forest, i)).max(0.0),
        BoxSizing::ContentBox => resolved.max(0.0),
    })
}

/// One column flex item's outer main (vertical) size for the
/// column-wrap simulation: the hypothetical main size
/// `flex_build_items` would compute (flex-basis over the specified
/// height over the intrinsic height, clamped by min/max-height),
/// plus the item's vertical border/padding and margins. When the
/// size needs a measuring layout (auto basis, auto height, no
/// intrinsic size) it is estimated as a single line — the item's
/// `line-height`. That is an approximation, not an exact measure:
/// it holds only for a single unwrapped text run in the max-content
/// pass; in the min-content pass multi-line text measures taller
/// than one line, and an item with several block children (or
/// forced breaks) is taller in both passes. In those cases the
/// estimate is a lower bound, which still lets the simulation run
/// instead of collapsing to the widest-item width. Percentages
/// resolve against the container's definite content height
/// `container_h`, as in the engine.
fn flex_column_item_main_outer(forest: &Forest, c: usize, container_h: f32) -> f32 {
    let d = &forest.nodes[c].data;
    let (fs, rfs, vpw, vph) = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
    let basis = spec_len_of(
        forest.nodes[c]
            .style
            .as_ref()
            .and_then(|s| s.get_by_index(props::FLEX_BASIS)),
    );
    // Single-line content estimate for the unmeasurable cases below.
    let one_line = d.line_height.max(0.0);
    let content = match basis {
        SpecLen::Auto => match d.height {
            SpecLen::Auto => match forest.nodes[c].intrinsic {
                Some(d) => d.height_or_default().max(0.0),
                None => one_line,
            },
            spec => match spec.resolve(fs, rfs, vpw, vph, container_h) {
                Some(v) => {
                    let v = v.max(0.0);
                    match d.box_sizing {
                        BoxSizing::BorderBox => (v - intrinsic_v_bp(forest, c)).max(0.0),
                        BoxSizing::ContentBox => v,
                    }
                }
                // Content-based keywords (`min-content` etc.): the
                // vertical intrinsic size is unmodeled; estimate.
                None => one_line,
            },
        },
        // `flex-basis` is in the item's box-sizing terms
        // (css-flexbox-1 §9.2): this is the *inner* (content-box)
        // main size — the vertical extras are added back below —
        // so strip them under `border-box`, else the padding/border
        // count twice (mirrors `flex_item_intrinsic_contribution`).
        spec => match spec.resolve(fs, rfs, vpw, vph, container_h) {
            Some(v) => {
                let resolved = v.max(0.0);
                match d.box_sizing {
                    BoxSizing::BorderBox => resolved - intrinsic_v_bp(forest, c),
                    BoxSizing::ContentBox => resolved,
                }
            }
            None => one_line,
        },
    };
    let resolve = |s: SpecLen| s.resolve(fs, rfs, vpw, vph, container_h);
    let min_v = resolve(d.min_height).unwrap_or(0.0).max(0.0);
    let max_v = resolve(d.max_height).unwrap_or(f32::INFINITY).max(0.0);
    let clamped = content.min(max_v).max(min_v).max(0.0);
    let mt = margin_opt(d.margin[0], 0.0, fs, rfs, vpw, vph).unwrap_or(0.0);
    let mb = margin_opt(d.margin[2], 0.0, fs, rfs, vpw, vph).unwrap_or(0.0);
    clamped + intrinsic_v_bp(forest, c) + mt + mb
}

/// Adjust one flex item's raw intrinsic contribution (`raw_outer`:
/// the content width `child_w` reported, plus the item's horizontal
/// extras) the way the real engine sizes the item's hypothetical main
/// size in [`flex_build_items`] (css-flexbox-1 §9.2/§9.7): for a row
/// container, resolve `flex-basis` over the raw content (a definite
/// length only — a percentage or percentage-bearing `calc()` is
/// cyclic against the container's indefinite intrinsic size and
/// behaves as `content`, exactly as `basis_or` does there), then
/// clamp through [`clamp_flex_main`] by the item's `min-width` /
/// `max-width` (percentage clamps are likewise cyclic here and drop
/// out inside that helper). A definite `flex-basis` resolves in the
/// item's box-sizing terms (css-flexbox-1 §9.2): the base here is the
/// *inner* (content-box) size, so only the item's own border+padding
/// are stripped under `border-box` — `flex-basis` never covers the
/// item's margins — before the `+ extras` below adds the full extras
/// back; otherwise the padding/border count twice. In the min-content pass
/// the automatic minimum size applies as in the engine: when
/// `min-width` is `auto` the result floors at the item's min-content size — which is what
/// the raw contribution already is in that pass. A definite `width`
/// on the item needs no special casing: it already flowed through
/// the child's own intrinsic dispatch (including its clamp) into the
/// raw contribution, exactly as `flex_build_items` reads it from the
/// item's resolved metrics.
///
/// For a column container the width is the *cross* size: `flex-basis`
/// (a main-axis, i.e. vertical, size) does not apply, only the
/// `min-width`/`max-width` clamp of an auto or content-based width —
/// a definite width already carries its clamp from the child's own
/// dispatch, so it is left alone (mirroring the column cross path in
/// `flex_build_items`, which clamps only content-based widths).
#[allow(clippy::too_many_arguments)]
fn flex_item_intrinsic_contribution(
    forest: &mut Forest,
    c: usize,
    mode: IntrinsicMode,
    horizontal: bool,
    raw_outer: f32,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
) -> f32 {
    let extras = intrinsic_h_extras(forest, c);
    let content = (raw_outer - extras).max(0.0);
    let (min_spec, max_spec, fs) = {
        let d = &forest.nodes[c].data;
        (d.min_width, d.max_width, d.font_size)
    };
    let (rfs, vpw, vph) = (forest.root_font_size, forest.vp_w, forest.vp_h);
    if !horizontal {
        if definite_specified_content_width(forest, c).is_some() {
            return raw_outer;
        }
        let (clamped, _, _) = clamp_flex_main(
            forest,
            c,
            content,
            min_spec,
            max_spec,
            fs,
            rfs,
            vpw,
            vph,
            None,
            true,
            fonts,
            pw_cache,
            mcw,
            bufs,
            inner,
            f32::INFINITY,
        );
        return clamped + extras;
    }
    let basis = spec_len_of(
        forest.nodes[c]
            .style
            .as_ref()
            .and_then(|s| s.get_by_index(props::FLEX_BASIS)),
    );
    let base = match basis {
        SpecLen::Auto => content,
        SpecLen::Pct(_) => content,
        SpecLen::Math { had_pct, .. } if had_pct => content,
        // w76: mirror the row arm's keyword arms. The keyword, not
        // the width-derived `content`, drives the measurement:
        // MinContent measures min-content, MaxContent max-content.
        // FitContent is mode-threaded (w76fix finding 10): in the
        // min-content pass it measures min-content, because a
        // min-content constraint leaves no available space for the
        // fit clamp to grow into; the max pass keeps the indefinite
        // `avail = INFINITY` clamp, which collapses to max-content.
        // MaxContent in the min pass keeps the max-content base:
        // flexbox §9.9.3 makes the min-content contribution the
        // larger of the outer min-content and outer preferred
        // (max-content) sizes, capped/floored by the flex base, so
        // the base value dominates there (pinned by test (g)).
        // Like the row arm, the measured size is content-box and is
        // not border-box-stripped; the definite arm below keeps its
        // strip.
        SpecLen::MinContent | SpecLen::MaxContent => flex_basis_content_width(
            forest,
            c,
            basis,
            fonts,
            pw_cache,
            mcw,
            bufs,
            inner,
            f32::INFINITY,
        ),
        SpecLen::FitContent => {
            let eff = match mode {
                IntrinsicMode::Min => SpecLen::MinContent,
                IntrinsicMode::Max => SpecLen::FitContent,
            };
            flex_basis_content_width(
                forest,
                c,
                eff,
                fonts,
                pw_cache,
                mcw,
                bufs,
                inner,
                f32::INFINITY,
            )
        }
        // w76fix finding 4: an intrinsic pass has no definite
        // container to fill, so stretch falls back to max-content.
        SpecLen::Stretch => flex_basis_content_width(
            forest,
            c,
            SpecLen::MaxContent,
            fonts,
            pw_cache,
            mcw,
            bufs,
            inner,
            f32::INFINITY,
        ),
        spec => {
            let resolved = spec
                .resolve(fs, rfs, vpw, vph, 0.0)
                .unwrap_or(content)
                .max(0.0);
            // `flex-basis` resolves in the item's box-sizing terms
            // (css-flexbox-1 §9.2): this base is the *inner*
            // (content-box) size — the item's extras are added back
            // below — so strip only the border+padding under
            // `border-box`, else the padding/border count twice. The
            // margins are not part of `flex-basis` at all: stripping
            // them here would shrink the contribution (a 200px basis
            // with 30px side margins would contribute 200 instead of
            // the correct 260), so they stay out of the strip and
            // come back with `extras` below.
            match forest.nodes[c].data.box_sizing {
                BoxSizing::BorderBox => resolved - own_horizontal_bp(forest, c),
                BoxSizing::ContentBox => resolved,
            }
        }
    };
    let (clamped, _, _) = clamp_flex_main(
        forest,
        c,
        base,
        min_spec,
        max_spec,
        fs,
        rfs,
        vpw,
        vph,
        None,
        true,
        fonts,
        pw_cache,
        mcw,
        bufs,
        inner,
        f32::INFINITY,
    );
    let mut v = clamped;
    if mode == IntrinsicMode::Min && matches!(min_spec, SpecLen::Auto) {
        // Automatic minimum (flexbox §4.5): the item's min-content
        // *size*. `content` above went through the item's min/max
        // clamp, which the automatic minimum ignores (min wins when
        // over-constrained, css-sizing-3 §4) — so re-measure
        // unclamped when a `max-width` could have bound it. With
        // `max-width: none` (and `min-width: auto`) the clamped value
        // already is the size. For a content-keyword (or stretch)
        // basis the dispatched min-content is additionally floored
        // at a definite width (w76fix finding 2), so the minimum
        // comes from the keyword measurement instead. §4.5's
        // specified-size-suggestion cap is inert for keyword bases:
        // the flex base is at least the min-content size (a
        // max/fit clamp sits above it; a grid substitution sums
        // maxima above minima), so the cap never binds below it.
        let auto_min = match basis {
            SpecLen::MinContent | SpecLen::MaxContent | SpecLen::FitContent | SpecLen::Stretch => {
                keyword_min_content_unclamped(forest, c, fonts, mcw, bufs).max(0.0)
            }
            _ => {
                if matches!(max_spec, SpecLen::Auto) {
                    content
                } else {
                    (min_content_width_pooled_raw(forest, c, fonts, mcw, bufs)).max(0.0)
                }
            }
        };
        v = v.max(auto_min);
    }
    v + extras
}

// ---------------------------------------------------------------------------
// Multicol intrinsic widths
// ---------------------------------------------------------------------------

/// Resolved multicol parameters for intrinsic sizing: the specified
/// column count (`None` = `auto`), the specified column width in px
/// (`None` = `auto`), and the used column gap in px.
struct IntrinsicCols {
    count: Option<usize>,
    width: Option<f32>,
    gap: f32,
}

/// Whether `i` is a multicol container for intrinsic sizing, and its
/// resolved `(count, column-width, gap)`. Only block containers reach
/// here (flex, table and grid boxes are claimed by their own intrinsic
/// arms first), mirroring the block-container gate at `layout_block`'s
/// `multicol_params` call site. Resolution is against an indefinite
/// base — percentages behave as `auto` (css-sizing-3 §5.2.1).
///
/// The longhands are resolved here directly, never through the shared
/// [`layout_multicol::multicol_params`]: that function unconditionally
/// pushes the resolved `column-rule` into the `PENDING_RULES`
/// thread-local (the column-rule handoff for real layout), so measuring
/// through it would plant a dead rule entry per multicol container per
/// measurement — with `col_w = 0.0`, which a later `pop_pending_rule`
/// could match against the wrong container and silently drop its
/// percentage `column-rule-width`. A `column-count`-specified container
/// is therefore detected straight from the `column-count` longhand
/// (the shared resolver's derived-column-count path needs a definite
/// width and cannot fire against the indefinite base anyway), and
/// column-width-involved containers resolve their longhands here with
/// the same per-longhand semantics.
fn multicol_intrinsic_params(forest: &Forest, i: usize) -> Option<IntrinsicCols> {
    // w79: both gates move together (see `layout_block`'s multicol
    // gate). A `Block`/`AnonymousBlock` box qualifies as before, and
    // so does an `InlineBlock` box carrying `display: inline-block`
    // — the unpromoted shape of a multicol inline-block, which real
    // layout blockifies (via `wrap_flex_text_runs` / the flex/grid
    // promotion triggers) and lays out as multicol. The display
    // key is what keeps inline-flex/grid/table out: they share the
    // `InlineBlock` box type but carry their own display, and their
    // own earlier arms in both intrinsic dispatches (flex arm,
    // grid arm, table arm) claim them before the multicol arm is
    // reached; a replaced inline box (`display: inline`) is not a
    // block container at all and is excluded the same way.
    if !(matches!(
        forest.nodes[i].box_type,
        BoxType::Block | BoxType::AnonymousBlock
    ) || (forest.nodes[i].box_type == BoxType::InlineBlock
        && forest.nodes[i].data.display == Display::InlineBlock))
    {
        return None;
    }
    let n = &forest.nodes[i];
    let style = n.style.as_deref()?;
    let (fs, rfs, vw, vh) = (
        n.data.font_size,
        forest.root_font_size,
        forest.vp_w,
        forest.vp_h,
    );
    // Resolve the longhands directly against the same indefinite
    // base (percentages read as `auto`). Never call the shared
    // `multicol_params` here: it pushes into `PENDING_RULES`, a
    // handoff that belongs to real layout, not measurement (see the
    // doc comment above).
    let width = match style.get_by_index(props::COLUMN_WIDTH) {
        Some(v) => {
            let w = spec_len_of(Some(v))
                .resolve(fs, rfs, vw, vh, 0.0)
                .map(sane_len)
                .unwrap_or(0.0);
            if w > 0.0 { Some(w) } else { None }
        }
        None => None,
    };
    let count = match style.get_by_index(props::COLUMN_COUNT) {
        Some(CssValue::Number(n)) if n.is_finite() && *n >= 1.0 => {
            // Mirrors `layout_multicol::MAX_COLUMNS` (private there).
            Some((*n as usize).clamp(1, 1024))
        }
        _ => None,
    };
    let gap = match style.get_by_index(props::COLUMN_GAP) {
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("normal") => fs,
        v => spec_len_of(v)
            .resolve(fs, rfs, vw, vh, 0.0)
            .map(sane_len)
            .unwrap_or(fs)
            .max(0.0),
    };
    match (count, width) {
        // `column-count` specified with `column-width: auto` (or
        // unresolvable against the indefinite base): the container has
        // exactly that many columns. (Absolute lengths resolve
        // positively against the zero base, so `column-width` alone —
        // or together with `column-count` — always lands in the width
        // arm; the shared resolver's `None` for those came from
        // `derived(w) = 1 → n < 2`, not from the width failing to
        // resolve.) A lone `column-count: 1` is a single column —
        // plain block sizing.
        (Some(c), None) if c >= 2 => Some(IntrinsicCols {
            count: Some(c),
            width: None,
            gap,
        }),
        // `column-width` involved (`column-width` alone, or together
        // with `column-count`).
        (_, Some(w)) => Some(IntrinsicCols {
            count,
            width: Some(w),
            gap,
        }),
        // No effective longhand: not a multicol container.
        _ => None,
    }
}

/// Largest min-content inline-size contribution of `i`'s in-flow
/// children (content-box): the single-column content measure both
/// multicol intrinsic arms build on. Out-of-flow children are skipped
/// (css-sizing-3: intrinsic sizes are in-flow-only). Sibling subtrees are
/// disjoint, so measuring each child's subtree stays linear in the
/// multicol subtree.
fn multicol_content_min_width(
    forest: &mut Forest,
    i: usize,
    fonts: &dyn FontMetrics,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
) -> f32 {
    let kids: Vec<usize> = forest.nodes[i]
        .children
        .iter()
        .copied()
        .filter(|&c| !layout_position::is_out_of_flow(forest, c))
        .collect();
    let mut l = 0.0f32;
    for c in kids {
        l = l.max(
            min_content_width_pooled(forest, c, fonts, mcw, bufs) + intrinsic_h_extras(forest, c),
        );
    }
    l
}

/// Min-content size of a multicol container (css-multicol intrinsic
/// sizes, csswg css-multicol#4952): with a non-`auto` column width, the
/// smaller of the column width and the single-column content measure;
/// otherwise the content measure times the column count (`auto` → 1)
/// plus the column gaps.
fn multicol_min_content_width(cols: &IntrinsicCols, content: f32) -> f32 {
    match cols.width {
        Some(w) => w.min(content),
        None => {
            let n = cols.count.unwrap_or(1) as f32;
            content * n + cols.gap * (n - 1.0)
        }
    }
}

/// Max-content size of a multicol container with unrestrained column
/// heights (csswg css-multicol#4952): with a non-`auto` column count,
/// the count times the larger of the column width (`auto` → zero) and
/// the single-column content measure, plus the column gaps. With
/// `column-count: auto` the spec multiplies the column width by the
/// number of columns from taking all allowed breaks — approximated as
/// one column per in-flow child (each child boundary is a break
/// opportunity), minimum one.
fn multicol_max_content_width(cols: &IntrinsicCols, content: f32, in_flow_kids: usize) -> f32 {
    match cols.count {
        Some(n) => {
            let n = n as f32;
            n * cols.width.unwrap_or(0.0).max(content) + cols.gap * (n - 1.0)
        }
        None => {
            let c = in_flow_kids.max(1) as f32;
            cols.width.unwrap_or(0.0) * c + cols.gap * (c - 1.0)
        }
    }
}

/// Generic preferred-width sizing for [`preferred_width`]: block-level
/// children contribute their max, inline-level children sum into runs.
/// Each child's used horizontal padding/border/margins ride along with
/// its contribution (via [`intrinsic_h_extras`]).
pub(crate) fn generic_preferred_width(forest: &Forest, i: usize) -> f32 {
    let mut maxw = 0.0f32;
    let mut run = 0.0f32;
    for &c in &forest.nodes[i].children {
        // Out-of-flow children do not contribute to intrinsic sizes
        // (css-sizing-3 §5.2): skipped without breaking the inline run —
        // flow lays the surrounding text out as one line box.
        if layout_position::is_out_of_flow(forest, c) {
            continue;
        }
        // `<br>` is a forced line break even at max-content (soft wraps
        // never trigger there): the run ends here.
        if forest.nodes[c].is_br {
            maxw = maxw.max(run);
            run = 0.0;
            continue;
        }
        // The child's margin box contributes: its preferred content
        // width plus its used horizontal padding/border/margins.
        let cw = forest.nodes[c].pref_w + intrinsic_h_extras(forest, c);
        if is_block_level(forest.nodes[c].box_type) {
            maxw = maxw.max(cw);
            run = 0.0;
        } else {
            run += cw;
            maxw = maxw.max(run);
        }
    }
    maxw
}

/// A block-level box's definite physical `height`, or `None` when the
/// height is `auto`, cyclic, or otherwise unresolvable this early. In a
/// vertical writing mode the physical height is the box's inline-axis
/// extent, so a definite height is an exact vertical intrinsic
/// contribution. The cyclic gates mirror [`clamp_height`]: percentages
/// (and percentage-bearing `calc()`) against the indefinite intrinsic
/// base behave as `auto` (CSS Values 4 §8.1.4).
fn definite_vertical_height(forest: &Forest, idx: usize) -> Option<f32> {
    let n = &forest.nodes[idx];
    let d = &n.data;
    match d.height {
        SpecLen::Auto
        | SpecLen::FitContent
        | SpecLen::MinContent
        | SpecLen::MaxContent
        | SpecLen::Stretch => None,
        SpecLen::Pct(_) => None,
        SpecLen::Math { had_pct: true, .. } => None,
        spec => spec.resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            0.0,
        ),
    }
}

/// Whether a box is ignorable whitespace for intrinsic sizing: a
/// whitespace-only text node, or an anonymous block wrapping only such
/// text (the box builder wraps inter-block whitespace runs in anonymous
/// blocks). Collapsed whitespace contributes nothing to intrinsic sizes.
fn is_ignorable_whitespace(forest: &Forest, idx: usize) -> bool {
    let n = &forest.nodes[idx];
    match n.box_type {
        BoxType::Text => n.text.as_deref().unwrap_or("").trim().is_empty(),
        BoxType::AnonymousBlock => n.children.iter().all(|&c| {
            let cn = &forest.nodes[c];
            cn.box_type == BoxType::Text && cn.text.as_deref().unwrap_or("").trim().is_empty()
        }),
        _ => false,
    }
}

/// The vertical (inline-axis) content contribution of a table cell in a
/// vertical writing mode, or `None` when the cell's content is not
/// vertically measurable yet. Only block-level children with a definite
/// physical `height` are modeled: block children stack along the block
/// axis, so the inline contribution is the largest clamped height.
/// Definite heights make min-content and max-content equal, so one value
/// serves both. Out-of-flow children never contribute (css-sizing-3
/// §5.2), and collapsible whitespace between blocks contributes
/// nothing.
fn vertical_definite_content(forest: &Forest, idx: usize) -> Option<f32> {
    let mut acc = 0.0f32;
    for &c in &forest.nodes[idx].children {
        if layout_position::is_out_of_flow(forest, c) {
            continue;
        }
        if is_ignorable_whitespace(forest, c) {
            continue;
        }
        let cn = &forest.nodes[c];
        if !is_block_level(cn.box_type) {
            return None;
        }
        let h = definite_vertical_height(forest, c)?;
        acc = acc.max(vertical_child_margin_box(forest, c, h).max(0.0));
    }
    Some(acc)
}

/// A block child's inline-axis (physical vertical) margin-box
/// contribution for [`vertical_definite_content`]: its definite physical
/// `height` `h` (in the box-sizing's own terms), clamped by the physical
/// `min-height`/`max-height`, converted to border-box, plus its vertical
/// border/padding and margins. Inline-axis margins never collapse, so
/// they count in full.
///
/// The clamp is physical: unlike [`clamp_height`] it never applies the
/// vertical-table transpose, so a nested vertical table child clamps
/// against its own physical `min-height`/`max-height` rather than its
/// `min-width`/`max-width` (the transpose reads the block-axis clamps,
/// which is correct for the table's own block size but wrong for a
/// physical-height contribution). Extras resolve from style against a
/// zero base — the intrinsic passes run before `resolve_metrics` — with
/// the same `border-style: none | hidden` and CSS 2.1 §8.3 internal-table
/// margin rules as [`intrinsic_h_extras`].
fn vertical_child_margin_box(forest: &Forest, idx: usize, h: f32) -> f32 {
    let n = &forest.nodes[idx];
    let d = &n.data;
    let (fs, rfs, vpw, vph) = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
    let len0 = |s: SpecLen| s.resolve(fs, rfs, vpw, vph, 0.0).unwrap_or(0.0).max(0.0);
    let bw = |w: SpecLen, st: BorderStyle| {
        if st.zeroes_width() { 0.0 } else { len0(w) }
    };
    let bp = bw(d.border[0], d.border_style[0])
        + len0(d.padding[0])
        + len0(d.padding[2])
        + bw(d.border[2], d.border_style[2]);
    // CSS 2.1 §8.3: margins do not apply to internal table boxes.
    let margins = if matches!(
        n.box_type,
        BoxType::TableRowGroup
            | BoxType::TableRow
            | BoxType::TableCell
            | BoxType::TableColumn
            | BoxType::TableColumnGroup
    ) {
        0.0
    } else {
        margin_opt(d.margin[0], 0.0, fs, rfs, vpw, vph).unwrap_or(0.0)
            + margin_opt(d.margin[2], 0.0, fs, rfs, vpw, vph).unwrap_or(0.0)
    };
    // Cyclic percentages behave as `auto` (CSS Values 4 §8.1.4); table
    // cells ignore `max-height` (css-tables-3 height distribution only
    // defines `height` as a minimum) — both mirroring `clamp_height`.
    let resolve = |s: SpecLen| match s {
        SpecLen::Auto => None,
        SpecLen::Pct(_) => None,
        SpecLen::Math { had_pct: true, .. } => None,
        spec => spec.resolve(fs, rfs, vpw, vph, 0.0),
    };
    let min_h = resolve(d.min_height).unwrap_or(0.0).max(0.0);
    let max_h = if n.box_type == BoxType::TableCell {
        f32::INFINITY
    } else {
        resolve(d.max_height).unwrap_or(f32::INFINITY).max(0.0)
    };
    // Clamp in the box-sizing's own terms: `min-height`/`max-height`
    // refer to the border box under `border-box`, the content box
    // otherwise (mirroring `clamp_height`).
    let border_box = match d.box_sizing {
        BoxSizing::BorderBox => h.min(max_h).max(min_h),
        BoxSizing::ContentBox => (h + bp).min(max_h + bp).max(min_h + bp),
    };
    border_box.max(0.0) + margins
}

/// Which intrinsic pass a vertical measurement serves. Block children
/// stack along the block axis in both passes, but inline children
/// combine differently: min-content keeps the largest unbreakable run,
/// max-content the whole unwrapped line.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VerticalIntrinsicMode {
    Min,
    Max,
}

/// Maximum recursion depth of the vertical intrinsic measurement family
/// ([`vertical_inline_min`], [`vertical_inline_max`],
/// [`vertical_atomic_inline`], [`vertical_block_child`],
/// [`vertical_flow_content`]) — the only recursive functions in the
/// crate. Past this depth the measurement declines (`None`) and the
/// caller falls back to the honest horizontal measure (the `None` path
/// in [`vertical_cell_min_content`] / [`vertical_cell_max_content`]),
/// so an adversarially deep DOM inside a vertical-rl table cell cannot
/// overflow the stack.
const VERTICAL_INTRINSIC_DEPTH_CAP: u32 = 256;

/// The vertical (physical top + bottom) border and padding of a box, in
/// the intrinsic pass's zero-base terms: the border/padding half of
/// [`vertical_child_margin_box`], factored out for the inline-level and
/// percentage-resolution paths. That function's body is the
/// definite-height fast path and stays exactly as it was.
fn vertical_bp(forest: &Forest, idx: usize) -> f32 {
    let d = &forest.nodes[idx].data;
    let (fs, rfs, vpw, vph) = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
    let len0 = |s: SpecLen| s.resolve(fs, rfs, vpw, vph, 0.0).unwrap_or(0.0).max(0.0);
    let bw = |w: SpecLen, st: BorderStyle| {
        if st.zeroes_width() { 0.0 } else { len0(w) }
    };
    bw(d.border[0], d.border_style[0])
        + len0(d.padding[0])
        + len0(d.padding[2])
        + bw(d.border[2], d.border_style[2])
}

/// The vertical (physical top + bottom) border, padding and margins of
/// a box: the extras half of [`vertical_child_margin_box`], for
/// inline-level contributions, which ride along with the content
/// measure exactly as [`intrinsic_h_extras`] does horizontally. Same
/// `border-style: none | hidden` zeroing and CSS 2.1 §8.3 internal-table
/// margin rules.
fn vertical_v_extras(forest: &Forest, idx: usize) -> f32 {
    let n = &forest.nodes[idx];
    let d = &n.data;
    let (fs, rfs, vpw, vph) = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
    // CSS 2.1 §8.3: margins do not apply to internal table boxes.
    let margins = if matches!(
        n.box_type,
        BoxType::TableRowGroup
            | BoxType::TableRow
            | BoxType::TableCell
            | BoxType::TableColumn
            | BoxType::TableColumnGroup
    ) {
        0.0
    } else {
        margin_opt(d.margin[0], 0.0, fs, rfs, vpw, vph).unwrap_or(0.0)
            + margin_opt(d.margin[2], 0.0, fs, rfs, vpw, vph).unwrap_or(0.0)
    };
    vertical_bp(forest, idx) + margins
}

/// A box's definite physical content-box height, or `None` when the
/// height is indefinite. Seeds percentage-height resolution for
/// descendants during vertical intrinsic measurement (CSS Values 4
/// §8.1.4: percentages against a definite containing-block height are
/// definite; against an indefinite one they behave as `auto`).
fn definite_content_box_height(forest: &Forest, idx: usize) -> Option<f32> {
    let h = definite_vertical_height(forest, idx)?;
    Some(match forest.nodes[idx].data.box_sizing {
        BoxSizing::BorderBox => (h - vertical_bp(forest, idx)).max(0.0),
        _ => h.max(0.0),
    })
}

/// Vertical min-content contribution of an inline-level box: the largest
/// unbreakable run's extent along the vertical inline axis. Mirrors the
/// horizontal min-content word split: an `Inline` box is one unbreakable
/// run (its children's contributions sum); atomic boxes go through
/// [`vertical_atomic_inline`]. `None` when the box has no vertical
/// model — the caller falls back honestly.
///
/// `depth` is the [`VERTICAL_INTRINSIC_DEPTH_CAP`] counter: past the cap
/// the measurement declines (`None`) instead of recursing deeper.
fn vertical_inline_min(
    forest: &Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    depth: u32,
) -> Option<f32> {
    if depth > VERTICAL_INTRINSIC_DEPTH_CAP {
        return None;
    }
    let n = &forest.nodes[idx];
    match n.box_type {
        BoxType::Text => {
            let text = n.text.as_deref().unwrap_or("");
            let fs = n.data.font_size;
            let family = family_id_for(forest, idx, fonts);
            let run = |s: &str| fonts.advance(s, fs, family);
            // The run's vertical extent is its advance: a
            // horizontal-metrics proxy, exact for stacked (e.g. CJK)
            // glyphs and an overestimate for rotated Latin. True
            // vertical glyph advances do not exist in paint/text yet —
            // a wave-scale project on its own (see the fallback
            // LIMITATION in `vertical_cell_min_content`).
            Some(match n.data.white_space {
                WhiteSpace::NoWrap => run(text),
                WhiteSpace::Pre => text.split('\n').map(run).fold(0.0, f32::max),
                _ => text.split([' ', '\n', '\t']).map(run).fold(0.0, f32::max),
            })
        }
        BoxType::Inline => {
            let mut sum = 0.0;
            for &c in &n.children {
                if layout_position::is_out_of_flow(forest, c) {
                    continue;
                }
                if is_ignorable_whitespace(forest, c) {
                    continue;
                }
                sum += vertical_inline_min(forest, c, fonts, depth + 1)?
                    + vertical_v_extras(forest, c);
            }
            Some(sum)
        }
        _ => vertical_atomic_inline(forest, idx, fonts, VerticalIntrinsicMode::Min, depth),
    }
}

/// Vertical max-content contribution of an inline-level box: the whole
/// unwrapped line's extent along the vertical inline axis. At
/// max-content soft breaks never trigger, so only forced breaks (`pre`
/// / `pre-wrap` / `pre-line` newlines) split the line; each line's
/// vertical extent is at least the container's line-height strut and at
/// least its content advance (same advance proxy as
/// [`vertical_inline_min`]).
///
/// `depth` is the [`VERTICAL_INTRINSIC_DEPTH_CAP`] counter (see
/// [`vertical_inline_min`]).
fn vertical_inline_max(
    forest: &Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    strut: f32,
    depth: u32,
) -> Option<f32> {
    if depth > VERTICAL_INTRINSIC_DEPTH_CAP {
        return None;
    }
    let n = &forest.nodes[idx];
    match n.box_type {
        BoxType::Text => {
            let text = n.text.as_deref().unwrap_or("");
            let fs = n.data.font_size;
            let family = family_id_for(forest, idx, fonts);
            let line = |s: &str| strut.max(fonts.advance(s, fs, family));
            Some(match n.data.white_space {
                WhiteSpace::Pre | WhiteSpace::PreWrap | WhiteSpace::PreLine => {
                    text.split('\n').map(line).sum()
                }
                _ => line(text),
            })
        }
        BoxType::Inline => {
            let mut sum = 0.0;
            for &c in &n.children {
                if layout_position::is_out_of_flow(forest, c) {
                    continue;
                }
                if is_ignorable_whitespace(forest, c) {
                    continue;
                }
                sum += vertical_inline_max(forest, c, fonts, strut, depth + 1)?
                    + vertical_v_extras(forest, c);
            }
            Some(sum)
        }
        _ => vertical_atomic_inline(forest, idx, fonts, VerticalIntrinsicMode::Max, depth),
    }
}

/// Vertical extent of an atomic inline-level box (inline-block,
/// replaced elements): the intrinsic height for replaced boxes, the
/// exact height for definite specified heights (both clamped by
/// `min-height` / `max-height` via [`vertical_child_margin_box`]), and
/// the recursively measured content for auto-height inline-blocks
/// (their shrink-to-fit contribution in this mode). `None` for anything
/// else — e.g. an auto-height inline-table/inline-grid, whose
/// column/track sizing has no transposed model yet.
///
/// `depth` is the [`VERTICAL_INTRINSIC_DEPTH_CAP`] counter (see
/// [`vertical_inline_min`]).
fn vertical_atomic_inline(
    forest: &Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    mode: VerticalIntrinsicMode,
    depth: u32,
) -> Option<f32> {
    if depth > VERTICAL_INTRINSIC_DEPTH_CAP {
        return None;
    }
    let n = &forest.nodes[idx];
    // Replaced elements (`<img>` etc.): the intrinsic height is the
    // physical vertical extent.
    if let Some(d) = n.intrinsic {
        return Some(vertical_child_margin_box(
            forest,
            idx,
            d.height_or_default().max(0.0),
        ));
    }
    if let Some(h) = definite_vertical_height(forest, idx) {
        return Some(vertical_child_margin_box(forest, idx, h));
    }
    if n.box_type == BoxType::InlineBlock {
        let content = vertical_flow_content(forest, idx, fonts, mode, None, depth + 1, false)?;
        return Some(vertical_child_margin_box(forest, idx, content));
    }
    None
}

/// One block-level child's vertical (inline-axis) margin-box
/// contribution: the definite height when specified (the same value
/// [`vertical_definite_content`] contributes for it), a percentage
/// height resolved against the nearest definite ancestor content
/// height, or — for `auto` (and cyclic-percentage / intrinsic-keyword)
/// heights — the content measured by [`vertical_flow_content`].
/// `ancestor_h` is that nearest definite ancestor content-box height
/// (`None` inside auto-height subtrees, where percentages stay cyclic).
/// `None` when the child has no vertical model (indefinite-size nested
/// tables, grids, flex containers and friends — their column/track/flex
/// sizing has no transposed model yet).
///
/// `depth` is the [`VERTICAL_INTRINSIC_DEPTH_CAP`] counter (see
/// [`vertical_inline_min`]).
fn vertical_block_child(
    forest: &Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    mode: VerticalIntrinsicMode,
    ancestor_h: Option<f32>,
    depth: u32,
    quirks: bool,
) -> Option<f32> {
    if depth > VERTICAL_INTRINSIC_DEPTH_CAP {
        return None;
    }
    if let Some(h) = definite_vertical_height(forest, idx) {
        return Some(vertical_child_margin_box(forest, idx, h));
    }
    // Only plain block containers recurse: nested tables, grids, flex
    // containers and multicol need their own transposed sizing
    // algorithms (out of scope) — they decline honestly.
    if !matches!(
        forest.nodes[idx].box_type,
        BoxType::Block | BoxType::AnonymousBlock | BoxType::TableCell
    ) {
        return None;
    }
    // Percentages against a definite ancestor are definite (CSS Values
    // 4 §8.1.4); without one they are cyclic and behave as `auto`.
    if let SpecLen::Pct(p) = forest.nodes[idx].data.height
        && let Some(base) = ancestor_h
    {
        return Some(vertical_child_margin_box(
            forest,
            idx,
            (p * base / 100.0).max(0.0),
        ));
    }
    let content = vertical_flow_content(forest, idx, fonts, mode, None, depth + 1, quirks)?;
    Some(vertical_child_margin_box(forest, idx, content))
}

/// The vertical (inline-axis) content contribution of a block container
/// in a vertical writing mode, or `None` when any content has no
/// vertical model yet. Block-level children stack along the block axis,
/// so their contribution is the largest child's; inline-level children
/// share the container's line boxes — the largest unbreakable run for
/// [`VerticalIntrinsicMode::Min`], the largest line's extent (floored by
/// the line-height strut) for [`VerticalIntrinsicMode::Max`]. A block
/// child starts a new line group, so in Max mode each inline run closes
/// at the next block child (mirroring [`generic_preferred_width`'s]
/// run/maxw split); in Min mode the max runs over all inline children,
/// since the largest unbreakable run is line-group independent.
/// Out-of-flow children never contribute (css-sizing-3 §5.2) and
/// collapsible whitespace contributes nothing, as in
/// [`vertical_definite_content`].
///
/// `depth` is the [`VERTICAL_INTRINSIC_DEPTH_CAP`] counter (see
/// [`vertical_inline_min`]).
///
/// `quirks` is the vertical twin of the horizontal §3.8 gate (see
/// [`quirks_cell_inline_min_content`]): in full quirks mode, when
/// the container being measured is a table cell (or the anonymous
/// block wrapping its inline run), consecutive `img` children in
/// Min mode have no soft wrap opportunity between them along the
/// vertical inline axis, so they sum instead of maxing. It
/// propagates only into anonymous-block children (the wrapper is
/// part of the cell's own formatting context); real block children
/// and nested inline-blocks are separate formatting contexts and
/// always measure with `false`. Max mode is unaffected.
fn vertical_flow_content(
    forest: &Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    mode: VerticalIntrinsicMode,
    ancestor_h: Option<f32>,
    depth: u32,
    quirks: bool,
) -> Option<f32> {
    if depth > VERTICAL_INTRINSIC_DEPTH_CAP {
        return None;
    }
    let strut = forest.nodes[idx].data.line_height.max(0.0);
    let mut block_max = 0.0f32;
    // Min mode: largest unbreakable run over all inline children.
    let mut inline_min_acc = 0.0f32;
    // Min mode with the quirks arm: the open run of consecutive
    // `img` extents, closed (into `inline_min_acc`) by any breaker.
    let mut inline_min_run = 0.0f32;
    // Max mode: largest inline run; the current run closes at each
    // block child.
    let mut inline_max = 0.0f32;
    let mut run = 0.0f32;
    let mut has_inline = false;
    let mut run_has_inline = false;
    for &c in &forest.nodes[idx].children {
        if layout_position::is_out_of_flow(forest, c) {
            continue;
        }
        // w78fix2: the vertical twin of the horizontal §3.8
        // white-space fix — in quirks Min mode a preserved
        // whitespace child must reach `vertical_quirks_run_step`
        // (which joins it to the run) instead of being skipped.
        // Every other case (non-quirks, Max mode, collapsible
        // whitespace) keeps the old skip.
        if is_ignorable_whitespace(forest, c)
            && !(quirks
                && mode == VerticalIntrinsicMode::Min
                && !quirks_ws_is_collapsible(forest, c))
        {
            continue;
        }
        if is_block_level(forest.nodes[c].box_type) {
            if mode == VerticalIntrinsicMode::Max && run_has_inline {
                inline_max = inline_max.max(run.max(strut));
            }
            run = 0.0;
            run_has_inline = false;
            // A block child ends any open quirks run: it is not
            // part of the cell's inline run.
            inline_min_acc = inline_min_acc.max(inline_min_run);
            inline_min_run = 0.0;
            block_max = block_max.max(vertical_block_child(
                forest,
                c,
                fonts,
                mode,
                ancestor_h,
                depth + 1,
                quirks && forest.nodes[c].box_type == BoxType::AnonymousBlock,
            )?);
        } else {
            has_inline = true;
            match mode {
                VerticalIntrinsicMode::Min => {
                    if quirks {
                        vertical_quirks_run_step(
                            forest,
                            c,
                            fonts,
                            depth + 1,
                            &mut inline_min_acc,
                            &mut inline_min_run,
                        )?;
                    } else {
                        inline_min_acc =
                            inline_min_acc.max(vertical_inline_min(forest, c, fonts, depth + 1)?);
                    }
                }
                VerticalIntrinsicMode::Max => {
                    run_has_inline = true;
                    run += vertical_inline_max(forest, c, fonts, strut, depth + 1)?;
                }
            }
        }
    }
    // Inline content occupies at least one strut: even a lone short run
    // forms a line box of the container's line-height.
    let inline_total = match mode {
        VerticalIntrinsicMode::Min => {
            if has_inline {
                inline_min_acc.max(inline_min_run).max(strut)
            } else {
                0.0
            }
        }
        VerticalIntrinsicMode::Max => {
            if run_has_inline {
                inline_max = inline_max.max(run.max(strut));
            }
            inline_max
        }
    };
    Some(block_max.max(inline_total))
}

/// One inline child's step in the vertical quirks run walk (see
/// [`vertical_flow_content`]): a floated child or any non-`img`
/// child closes the open run and contributes its own vertical
/// extent; an `img` extends the run; an `Inline` container (other
/// than `<br>`) recurses transparently, sharing both accumulators —
/// the horizontal [`quirks_run_walk`] with vertical measures.
/// Returns `None` when a child has no vertical model, declining
/// the whole cell exactly as the non-quirks path does.
fn vertical_quirks_run_step(
    forest: &Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    depth: u32,
    acc: &mut f32,
    run: &mut f32,
) -> Option<()> {
    if depth > VERTICAL_INTRINSIC_DEPTH_CAP {
        return None;
    }
    let n = &forest.nodes[idx];
    if is_ignorable_whitespace(forest, idx) {
        // Vertical twin of the horizontal w78fix2 preserved-space
        // handling: collapsible whitespace is skipped (no effect
        // on the run), preserved whitespace joins the run at its
        // measured advance (newlines force-break between
        // segments), and an anonymous block recurses transparently.
        // Measured directly with `fonts`, not via
        // `vertical_inline_min` (which splits `pre-wrap` /
        // `break-spaces` text on spaces and gives 0 for a lone
        // space — right for a breakable run, wrong here).
        if quirks_ws_is_collapsible(forest, idx) {
            return Some(());
        }
        if n.box_type == BoxType::AnonymousBlock {
            for &c in &n.children {
                if layout_position::is_out_of_flow(forest, c) {
                    continue;
                }
                vertical_quirks_run_step(forest, c, fonts, depth + 1, acc, run)?;
            }
        } else {
            let text = n.text.as_deref().unwrap_or("");
            let fs = n.data.font_size;
            let family = family_id_for(forest, idx, fonts);
            for (seg_i, seg) in text.split('\n').enumerate() {
                if seg_i > 0 {
                    *acc = acc.max(*run);
                    *run = 0.0;
                }
                *run += fonts.advance(seg, fs, family);
            }
        }
        return Some(());
    }
    if is_float(forest, idx) {
        *acc = acc.max(*run);
        *run = 0.0;
        *acc = acc.max(vertical_inline_min(forest, idx, fonts, depth)?);
    } else if n.box_type == BoxType::InlineBlock && n.is_img && n.intrinsic.is_some() {
        *run += vertical_inline_min(forest, idx, fonts, depth)?;
    } else if n.box_type == BoxType::Inline && !n.is_br {
        for &c in &n.children {
            if layout_position::is_out_of_flow(forest, c) {
                continue;
            }
            if is_ignorable_whitespace(forest, c) && quirks_ws_is_collapsible(forest, c) {
                continue;
            }
            vertical_quirks_run_step(forest, c, fonts, depth + 1, acc, run)?;
        }
    } else {
        *acc = acc.max(*run);
        *run = 0.0;
        *acc = acc.max(vertical_inline_min(forest, idx, fonts, depth)?);
    }
    Some(())
}

/// Vertical min-content size of a table cell: the min-content measure
/// along a vertical table's inline axis (the physical vertical axis).
/// Used by the column machinery ([`layout_table::auto_column_minima`])
/// when `TableGrid::vertical` is set.
///
/// In full quirks mode the §3.8 run-summing arm applies here too
/// (see [`vertical_flow_content`]'s `quirks` parameter): consecutive
/// `img` children of the cell sum along the vertical inline axis
/// instead of maxing. The definite-height fast path above runs
/// first, so a definite inline size still short-circuits before
/// any run summing.
///
/// The definite-height fast path ([`vertical_definite_content`]) runs
/// first; otherwise [`vertical_flow_content`] measures the cell
/// vertically: inline children (text runs, inline boxes, atomic
/// inline-level boxes) by their vertical extent, and auto-height block
/// containers by recursing into their content, with percentage heights
/// resolved against the nearest definite ancestor. Content with no
/// vertical model yet (indefinite-size nested tables/grids/flex
/// containers, replaced boxes with no intrinsic size or definite
/// height) declines the whole cell to the horizontal min-content
/// measure — the pre-vertical-intrinsic behavior, so no
/// previously-measuring content can regress (see the LIMITATION note at
/// the fallback).
pub(crate) fn vertical_cell_min_content(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    mcw_scratch: &mut Vec<f32>,
    int_bufs: &mut WalkBufs,
) -> f32 {
    // §3.8 arm (see `vertical_flow_content`): the cell itself, or
    // the anonymous block wrapping its inline run, in a full-quirks
    // document. A definite inline size short-circuits above, so no
    // separate spec check is needed here.
    let quirks = forest.quirks_mode == QuirksMode::Quirks
        && (forest.nodes[idx].box_type == BoxType::TableCell
            || forest.nodes[idx].box_type == BoxType::AnonymousBlock
                && forest.nodes[idx]
                    .parent
                    .is_some_and(|p| forest.nodes[p].box_type == BoxType::TableCell));
    match vertical_definite_content(forest, idx).or_else(|| {
        vertical_flow_content(
            forest,
            idx,
            fonts,
            VerticalIntrinsicMode::Min,
            definite_content_box_height(forest, idx),
            0,
            quirks,
        )
    }) {
        Some(v) => v,
        // LIMITATION: some content in this cell has no vertical model
        // yet — indefinite-size nested tables/grids/flex containers,
        // replaced boxes with no intrinsic size or definite height (see
        // `vertical_flow_content`). The cell then measures its
        // *horizontal* min-content width and reports it on the vertical
        // axis: the wrong axis, kept deliberately. It is exactly the
        // pre-vertical-intrinsic behavior, so content that measured
        // before keeps measuring, and tests pin this approximation
        // instead of the code silently claiming a true vertical
        // measure. A real fix needs vertical glyph-run metrics in
        // paint/text plus transposed table/track sizing — a wave-scale
        // project on its own.
        None => min_content_width_pooled(forest, idx, fonts, mcw_scratch, int_bufs),
    }
}

/// Vertical max-content size of a table cell: the max-content measure
/// along a vertical table's inline axis. Used by the column machinery
/// ([`layout_table::auto_column_maxima`],
/// [`layout_table::table_pref_width`]) when `TableGrid::vertical` is
/// set. Same extended-measure-then-honest-fallback contract as
/// [`vertical_cell_min_content`].
pub(crate) fn vertical_cell_max_content(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw_scratch: &mut Vec<f32>,
    int_bufs: &mut WalkBufs,
    int_inner: &mut WalkBufs,
) -> f32 {
    match vertical_definite_content(forest, idx).or_else(|| {
        vertical_flow_content(
            forest,
            idx,
            fonts,
            VerticalIntrinsicMode::Max,
            definite_content_box_height(forest, idx),
            0,
            false,
        )
    }) {
        Some(v) => v,
        // LIMITATION: same as `vertical_cell_min_content`'s fallback —
        // content with no vertical model yet measures its *horizontal*
        // max-content width and reports it on the vertical axis. Wrong
        // axis, kept deliberately as the pre-vertical-intrinsic
        // behavior; see that function for the full note.
        None => preferred_width_mcw(
            forest,
            idx,
            fonts,
            pw_cache,
            mcw_scratch,
            int_bufs,
            int_inner,
        ),
    }
}

/// Vertical min-content extent of a table *caption* subtree when the
/// table is in a vertical writing mode. The stacking math of
/// [`vertical_definite_content`] is generic to any block container: a
/// caption's inline axis is the physical vertical axis, its block axis
/// the horizontal one, so in-flow block children stack along the block
/// axis and the vertical extent is the largest clamped definite
/// child height. Reuses [`vertical_cell_min_content`] wholesale —
/// including its conservative fallback: any unmodeled content
/// (auto-height blocks, inline-level children, text, replaced boxes,
/// vertical text line stacking) falls back to the horizontal
/// min-content measure, so previously-measuring caption content cannot
/// regress. Used by [`layout_table::table_caption_min_width`].
pub(crate) fn vertical_caption_min_content(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    mcw_scratch: &mut Vec<f32>,
    int_bufs: &mut WalkBufs,
) -> f32 {
    vertical_cell_min_content(forest, idx, fonts, mcw_scratch, int_bufs)
}

/// Vertical max-content extent of a table *caption* subtree when the
/// table is in a vertical writing mode. Same contract as
/// [`vertical_caption_min_content`] with max-content semantics:
/// definite heights make min-content and max-content equal, so both
/// read the same value off [`vertical_cell_max_content`], and
/// unmodeled content falls back to the horizontal max-content measure.
/// Used by [`layout_table::table_caption_pref_width`].
pub(crate) fn vertical_caption_max_content(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw_scratch: &mut Vec<f32>,
    int_bufs: &mut WalkBufs,
    int_inner: &mut WalkBufs,
) -> f32 {
    vertical_cell_max_content(
        forest,
        idx,
        fonts,
        pw_cache,
        mcw_scratch,
        int_bufs,
        int_inner,
    )
}

/// Intrinsic preferred content width, iterative post-order. Block-level
/// children contribute their max; inline-level children sum into runs.
/// Each child's used horizontal padding/border/margins ride along with
/// its contribution (via [`intrinsic_h_extras`]); the box's own are left
/// for the caller, which keeps `PrefWidthCache` entries (content-box
/// widths) consistent.
///
/// The `mcw` buffer is [`min_content_width_pooled`]'s reusable
/// scratch, needed for the F7 `width: min-content` contribution arm.
/// `bufs` is caller-provided preorder-walk scratch (see [`WalkBufs`]):
/// pooled in [`LayoutScratch`] (`int_bufs`/`int_inner`) at top-level
/// call sites and hoisted per container in the table/grid intrinsic
/// loops, so one zeroed walk serves many measurements instead of
/// allocating per call. `inner` is the second pooled pair for the
/// nested F7 walks: they run while this walk's `bufs.order` is
/// borrowed by the per-node loop below, so they need their own.
/// Both are cleared on entry and must not alias an in-progress outer
/// intrinsic walk's buffers — nested measurements (the table, grid
/// and multicol arms below) use their own — which the borrow
/// checker enforces where the buffers are threaded as parameters.
/// The flex arm is the exception: its clamp adjustment runs nested
/// content-keyword measurements while this walk's `bufs.order` is
/// borrowed by the per-node loop, so it cannot reuse `bufs` itself —
/// but it *can* borrow the disjoint `bufs.flex` field, the
/// [`WalkBufs`]-resident [`FlexScratch`] shared with
/// [`min_content_width_pooled`]'s flex arm (see its docs). One
/// generation-stamped cache and one nested walk pair per
/// pooled-buffer lifetime serve every flex node of every walk, and a
/// nested flex container inside the clamp reuses the same scratch
/// instead of lazily building a second one per call.
/// Callers without any reusable buffer use [`preferred_width`], which
/// checks out the thread-local pooled triple instead of allocating a
/// fresh one per call (only a re-entrant caller gets a fresh triple).
/// Scrollbar-gutter reservation folded into a scroll container's
/// intrinsic content width (css-sizing-3 §5: the intrinsic size is the
/// content-box size the box needs). At layout time
/// [`scrollbar_gutter_reservation`](crate::scrollbar_gutter_reservation)
/// / [`forced_scrollbar_reservation`](crate::forced_scrollbar_reservation)
/// come out of the scroll container's content box ([`layout_block`]'s
/// prologue, which every container's outer box passes through — the
/// flex/grid/table engines derive their content origin/width from its
/// reserved `kids_x` / `kids_w`), so a shrink-to-fit scroller whose
/// intrinsic size measured content only would lose the gutter from its
/// children's width. Chrome includes the gutter in the scroll
/// container's intrinsic size.
///
/// Returns `0.0` unless the box is a scroll container with a non-zero
/// reservation — `overflow-y: scroll`/`auto`/`hidden` with
/// `scrollbar-gutter: stable` (or `stable both-edges`, the symmetric
/// case), or `overflow-y: scroll` forcing the classic scrollbar — and
/// the width is content-based: a definite specified width already
/// contains the gutter (it is reserved from that exact width), so it
/// contributes nothing extra. The propagated root is skipped like the
/// layout-time subtraction (the viewport reserved its gutter, not the
/// root's own box). Internal table boxes other than a standalone cell
/// never see the prologue subtraction (the table engine lays them out
/// from the table's reserved width), and non-atomic inline boxes never
/// go through [`layout_block`], so neither folds the gutter in —
/// mirroring exactly the boxes the layout-time subtraction applies to.
/// `Text` nodes have no style and contribute nothing.
///
/// Both intrinsic walks apply this at their per-node finalization:
/// [`preferred_width_mcw`] after the arm dispatch (its `width:
/// min-content` arms delegate to [`min_content_width_pooled`], which
/// folds the reservation in itself, so they are skipped there), and
/// [`min_content_width_pooled`] before the `min-width` / `max-width`
/// clamp, so the constraints bound the whole intrinsic size.
fn intrinsic_scrollbar_reservation(forest: &Forest, i: usize) -> f32 {
    // Only boxes whose own content box passes through `layout_block`'s
    // prologue see the layout-time subtraction; a cell inside a real
    // table is laid out by the table engine from the table's reserved
    // width instead.
    let box_type = forest.nodes[i].box_type;
    if !matches!(
        box_type,
        BoxType::Block
            | BoxType::AnonymousBlock
            | BoxType::InlineBlock
            | BoxType::Table
            | BoxType::Grid
            | BoxType::TableCell
    ) || (box_type == BoxType::TableCell && cell_in_table_context(forest, i))
    {
        return 0.0;
    }
    if forest.gutter_propagated_root == Some(i) {
        return 0.0;
    }
    let Some(style) = forest.nodes[i].style.as_ref() else {
        return 0.0;
    };
    let (gl, gr) = crate::scrollbar_gutter_reservation(style);
    let (fl, fr) = crate::forced_scrollbar_reservation(style);
    let reservation = gl.max(fl) + gr.max(fr);
    if reservation == 0.0 {
        return 0.0;
    }
    // A definite specified width already contains the gutter: it is
    // reserved from that exact content width at layout time.
    if definite_specified_content_width(forest, i).is_some() {
        return 0.0;
    }
    reservation
}

/// Whether a text node's `white-space` lets its edge spaces collapse
/// against a neighbour's (the same set
/// [`collapse_adjacent_spaces`] treats as collapsible): `pre`-family
/// modes preserve spaces, so their seams never collapse.
fn seam_collapsible_ws(ws: WhiteSpace) -> bool {
    matches!(
        ws,
        WhiteSpace::Normal | WhiteSpace::NoWrap | WhiteSpace::PreLine
    )
}

/// Last text descendant of an inline box in document order, skipping
/// out-of-flow and floated children (they produce no line fragments,
/// so they never break a seam). Returns `(Some(t), false)` for a
/// text node, `(None, false)` for an inline with no in-flow
/// children at all (transparent to the seam), and `(None, true)`
/// when the inline ends in a breaking box (a block-level child, a
/// replaced box, a `<br>`).
fn last_text_descendant(forest: &Forest, mut idx: usize) -> (Option<usize>, bool) {
    loop {
        let kids = &forest.nodes[idx].children;
        let mut found = None;
        for &c in kids.iter().rev() {
            if layout_position::is_out_of_flow(forest, c) || is_float(forest, c) {
                continue;
            }
            found = Some(c);
            break;
        }
        let Some(c) = found else {
            return (None, false);
        };
        if forest.nodes[c].box_type == BoxType::Text {
            return (Some(c), false);
        }
        if forest.nodes[c].box_type == BoxType::Inline {
            idx = c;
            continue;
        }
        return (None, true);
    }
}

/// Previous text node in the same inline run as `idx` (w77): walk
/// left through siblings, descending into trailing inlines, and up
/// through enclosing inlines; a block-level, replaced, or `<br>`
/// sibling, a non-inline parent, or the walk root itself ends the
/// run. Mirrors the fragment adjacency
/// [`collapse_adjacent_spaces`] sees in line layout, where inline
/// wrappers contribute no fragments of their own.
fn prev_text_in_run(forest: &Forest, idx: usize, root: usize) -> Option<usize> {
    let mut cur = idx;
    loop {
        if cur == root {
            return None;
        }
        let parent = forest.nodes[cur].parent?;
        let kids = &forest.nodes[parent].children;
        let pos = kids.iter().position(|&c| c == cur)?;
        for &c in kids[..pos].iter().rev() {
            if layout_position::is_out_of_flow(forest, c) || is_float(forest, c) {
                continue;
            }
            if forest.nodes[c].box_type == BoxType::Text {
                return Some(c);
            }
            if forest.nodes[c].box_type == BoxType::Inline {
                match last_text_descendant(forest, c) {
                    (Some(t), _) => return Some(t),
                    (None, true) => return None,
                    (None, false) => continue,
                }
            }
            return None;
        }
        if forest.nodes[parent].box_type == BoxType::Inline {
            cur = parent;
            continue;
        }
        return None;
    }
}

/// Whether the max-content walk must drop one space advance from
/// text node `idx`: its own text starts with a collapsible space
/// and the previous text node in its inline run ends with one.
/// This is the intrinsic-width twin of
/// [`collapse_adjacent_spaces`]; line layout itself is untouched,
/// so there is no double correction (fragments are computed from
/// the text, never from `pref_w`). The search stops at the walk
/// root, so a subtree measurement never inherits a seam from
/// outside the measured subtree.
fn seam_space_drop(forest: &Forest, idx: usize, root: usize) -> bool {
    let n = &forest.nodes[idx];
    if !seam_collapsible_ws(n.data.white_space) {
        return false;
    }
    let text = n.text.as_deref().unwrap_or("");
    if !text.starts_with(' ') {
        return false;
    }
    let Some(prev) = prev_text_in_run(forest, idx, root) else {
        return false;
    };
    let p = &forest.nodes[prev];
    seam_collapsible_ws(p.data.white_space) && p.text.as_deref().unwrap_or("").ends_with(' ')
}

fn preferred_width_mcw(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
) -> f32 {
    if cache.get(idx) {
        return forest.nodes[idx].pref_w;
    }
    bufs.clear();
    bufs.stack.push(idx);
    while let Some(i) = bufs.stack.pop() {
        // A cached subtree is skipped wholesale (see `PrefWidthCache`).
        if cache.get(i) {
            continue;
        }
        bufs.order.push(i);
        bufs.stack
            .extend(forest.nodes[i].children.iter().rev().copied());
    }
    // One pooled table grid serves every table met in this walk; each
    // table checks it out and returns it, so nested tables never alias.
    let mut table_grid_pool = layout_table::TableGridPool::new();
    for i in bufs.order.drain(..).rev() {
        let w = match forest.nodes[i].box_type {
            BoxType::Text => {
                let n = &forest.nodes[i];
                let text = n.text.as_deref().unwrap_or("");
                let fs = n.data.font_size;
                let family = family_id_for(forest, i, fonts);
                // Max-content must cover the width line layout
                // itself sums for the same run: `emit_text_frags`
                // measures each word (and each space) separately and
                // the breaker adds those fragment widths in order, so
                // a shaped whole-run advance can sit a few float ulps
                // *below* the fragment sum (real case, Noto Serif at
                // 16px: "2 above me, 2 below me?" shapes to 189.888
                // whole-run but 189.888031 as fragments). A
                // shrink-to-fit box or auto table column sized from
                // the whole-run value then wraps its own max-content
                // (caption table 8: td 60px tall, under1 at 120).
                // For `normal`/`nowrap` — the modes `emit_text_frags`
                // splits on spaces (and U+200B) — take the max of the
                // fragment sum in breaker order and the whole-run
                // advance: the fragment sum guarantees the run fits
                // its own max-content *per text node* (see below), and
                // the whole-run side keeps the historical value
                // wherever it is the larger (the reverse ulp
                // direction, e.g. Fixed metrics on a long caption
                // line, where the fragment sum lands an ulp below the
                // whole-run product). `pre`-family modes keep the
                // whole-run advance alone: their fragments are
                // pre-fit pieces, not this split.
                // Cross-node seam (w77): `collapse_adjacent_spaces`
                // zeroes one space at a text-node seam in line layout;
                // this walk used to count both seam spaces, overcounting
                // max-content by one space advance whenever text is
                // split across nodes. Drop the leading space here when
                // the previous text node in the run ends with a
                // collapsible space (see [`seam_space_drop`]); the
                // per-node guarantee above then holds across nodes too.
                let base = if matches!(n.data.white_space, WhiteSpace::Normal | WhiteSpace::NoWrap)
                {
                    let mut w = 0.0f32;
                    for (k, part) in text.split(' ').enumerate() {
                        if k > 0 {
                            w += fonts.advance(" ", fs, family);
                        }
                        for sub in part.split('\u{200b}') {
                            if !sub.is_empty() {
                                w += fonts.advance(sub, fs, family);
                            }
                        }
                    }
                    w.max(fonts.advance(text, fs, family))
                } else {
                    fonts.advance(text, fs, family)
                };
                if seam_space_drop(forest, i, idx) {
                    (base - fonts.advance(" ", fs, family)).max(0.0)
                } else {
                    base
                }
            }
            // `<br>` forces a line break even at max-content (no soft wraps
            // there), so the max-content width is the widest
            // break-delimited segment (flat case; nested `<br>` sums at
            // the outer level — same known limitation as min-content).
            BoxType::Inline => {
                let children = &forest.nodes[i].children;
                let mut best = 0.0f32;
                let mut seg = 0.0f32;
                for &c in children.iter() {
                    if layout_position::is_out_of_flow(forest, c) {
                        continue;
                    }
                    if forest.nodes[c].is_br {
                        best = best.max(seg);
                        seg = 0.0;
                    } else {
                        seg += forest.nodes[c].pref_w + intrinsic_h_extras(forest, c);
                    }
                }
                best.max(seg)
            }
            // Tables size by column, not by summing every cell: the
            // column structure comes from the same grid collection the
            // table layout itself uses. `inline-table` is an `InlineBlock`
            // box whose inner layout is the table algorithm, so it sizes
            // the same way.
            _ if forest.nodes[i].data.display == Display::InlineTable => {
                // An `inline-table` is shrink-to-fit (css-sizing-3
                // §5.2), so the caption's max-content floor always
                // applies — the same value the inner gate computes.
                layout_table::table_pref_width(forest, i, fonts, cache, &mut table_grid_pool, true)
            }
            BoxType::Table => {
                // The caption's max-content floor only feeds
                // shrink-to-fit sizing (floated, out-of-flow,
                // `inline-table`): compute the flag from the node's own
                // state — exactly `table_pref_width`'s inner gate
                // predicate — instead of paying a full caption walk for
                // a block-level table whose result the gate discards.
                let caption_floor = is_float(forest, i)
                    || layout_position::is_out_of_flow(forest, i)
                    || forest.nodes[i].data.display == Display::InlineTable;
                layout_table::table_pref_width(
                    forest,
                    i,
                    fonts,
                    cache,
                    &mut table_grid_pool,
                    caption_floor,
                )
            }
            // Grids size by track, not by summing every item: column
            // widths come from the same track sizing the grid layout
            // itself uses (fixed tracks resolve against a zero base).
            // `inline-grid` is an `InlineBlock` box with the grid inner
            // layout, so it sizes the same way.
            _ if forest.nodes[i].data.display == Display::InlineGrid => {
                layout_grid::grid_pref_width(forest, i, fonts, cache)
            }
            BoxType::Grid => layout_grid::grid_pref_width(forest, i, fonts, cache),
            // Flex containers size by item, not by the widest child:
            // a row container's max-content is the items' sum plus
            // gaps, a column container's the widest item (see
            // [`flex_intrinsic_width`]). Without this arm a flex
            // container measured by a shrink-to-fit ancestor fell to
            // the generic widest-item sizing. Definite widths keep
            // the same precedence as the arms below: `width:
            // min-content` contributes the min-content size, and a
            // definite width contributes exactly (content overflowing
            // it does not grow the contribution). The inline-flex
            // arm is keyed off `display`, not `box_type` (w78fix H1):
            // a promoted inline-flex item is a `Block` box, an
            // unpromoted one an `InlineBlock` box, and keying off the
            // box type gave the two different fallbacks (no
            // aspect-ratio fallback and no min/max clamp on the
            // intrinsic side for the unpromoted shape, the
            // [`definite_width_floor`] formula for the promoted one).
            // The display-keyed arm below uses the
            // [`definite_width_floor`] formula for both, so promotion
            // cannot change the preferred width; a plain `flex`
            // container reaches the same formula through
            // [`definite_width_floor`] itself.
            _ if flex_kind(forest, i).is_some() => {
                // The clamp adjustment's nested content-keyword
                // measurements run while this walk's `bufs.order` is
                // borrowed by the per-node loop, so they cannot reuse
                // `bufs` itself; they use the `WalkBufs`-resident
                // [`FlexScratch`] instead (`bufs.flex` is a disjoint
                // field borrow from `bufs.order`'s drain). The scratch
                // is shared with [`min_content_width_pooled`]'s flex
                // arm — one generation-stamped cache and one nested
                // walk pair per pooled-buffer lifetime — so a nested
                // flex container inside the clamp reuses it instead of
                // building a second scratch per call.
                let flex = bufs
                    .flex
                    .get_or_insert_with(|| Box::new(FlexScratch::new(forest.nodes.len())));
                flex.begin_node();
                let intrinsic = flex_intrinsic_width(
                    forest,
                    i,
                    IntrinsicMode::Max,
                    fonts,
                    &mut flex.pw,
                    &mut flex.aux,
                    &mut flex.bufs,
                    &mut flex.inner,
                    |f, c| f.nodes[c].pref_w + intrinsic_h_extras(f, c),
                );
                if matches!(forest.nodes[i].data.width, SpecLen::MinContent) {
                    min_content_width_pooled(forest, i, fonts, mcw, inner)
                } else if forest.nodes[i].data.display == Display::InlineFlex {
                    // w78fix H1 (see the arm comment above): the
                    // promoted and unpromoted shapes share this
                    // formula — definite, then replaced intrinsic,
                    // then aspect-ratio transfer, then the
                    // min/max-clamped flex intrinsic.
                    definite_specified_content_width(forest, i)
                        .or_else(|| intrinsic_content_width(forest, i))
                        .or_else(|| aspect_ratio_intrinsic_width(forest, i))
                        .unwrap_or_else(|| clamp_content_width_by_minmax(forest, i, intrinsic))
                } else {
                    definite_width_floor(forest, i, intrinsic, fonts, mcw, inner)
                }
            }
            // Multicol containers size by column, not by the generic
            // max-of-runs: per the css-multicol intrinsic sizing rules
            // (csswg css-multicol#4952) the max-content size is built
            // from the single-column content measure (the largest
            // min-content contribution of the in-flow contents — not
            // the summed inline runs), the resolved column count and
            // the column gap. Definite widths keep the block
            // precedence of the fallthrough arm: `width: min-content`
            // contributes the min-content size, a definite specified
            // width contributes exactly (content overflowing it does
            // not grow the contribution — css-sizing-3 §5.2).
            // NOTE (w79): `multicol_intrinsic_params` and
            // `layout_block`'s multicol gate moved together to also
            // accept an `InlineBlock` box carrying
            // `display: inline-block`, so a multicol inline-block
            // measures and lays out as multicol regardless of
            // promotion state. The pre-w79 divergence noted here
            // (widening only the intrinsic gate would size a box
            // as multicol that real layout laid out as a plain
            // block) is closed by that joint move.
            _ if let Some(cols) = multicol_intrinsic_params(forest, i) => {
                if matches!(forest.nodes[i].data.width, SpecLen::MinContent) {
                    min_content_width_pooled(forest, i, fonts, mcw, inner)
                } else {
                    let in_flow = forest.nodes[i]
                        .children
                        .iter()
                        .filter(|&&c| !layout_position::is_out_of_flow(forest, c))
                        .count();
                    let content = multicol_content_min_width(forest, i, fonts, mcw, inner);
                    let intrinsic = multicol_max_content_width(&cols, content, in_flow);
                    definite_specified_content_width(forest, i)
                        .or_else(|| intrinsic_content_width(forest, i))
                        .unwrap_or(intrinsic)
                }
            }
            // Atomic inline-level boxes (inline-block, and replaced
            // inline elements, which share the `InlineBlock` box
            // type): a definite specified width contributes exactly
            // its content width to the max-content size (css-sizing-3
            // §5.2), symmetric with the min-content arm.
            // `inline-table` and `inline-grid` are caught by the
            // guards above and keep their column/track sizing, and
            // `inline-flex` by the flex arm above. When the width is
            // `auto` (or cyclic), the contribution is the generic
            // max-of-runs sizing below.
            //
            // F7: `width: min-content` sizes the box to its min-content
            // size, so that — not the generic max-content — is its
            // intrinsic contribution to its parent.
            //
            // w78fix H1: these arms are keyed off `display` as well
            // as `box_type`, so a promoted inline-block (a `Block`
            // box carrying `display: inline-block`, from boxes pass 3
            // or `wrap_flex_text_runs` / `wrap_grid_text_runs`) takes
            // the same arm as an unpromoted one instead of falling
            // through to [`definite_width_floor`]. The formula is
            // [`definite_width_floor`]'s `Block` formula — definite,
            // then replaced intrinsic, then aspect-ratio transfer,
            // then the min/max-clamped generic size — so the two
            // promotion triggers agree on every fallback, not just on
            // the definite width (the old `InlineBlock` arm had no
            // aspect-ratio fallback and no min/max clamp on the
            // generic size, so a promoted aspect-ratio box measured
            // 100 where the unpromoted shape measured 0, and a
            // max-width-bounded auto box measured clamped where the
            // unpromoted shape measured raw).
            _ if (forest.nodes[i].box_type == BoxType::InlineBlock
                || forest.nodes[i].data.display == Display::InlineBlock)
                && matches!(forest.nodes[i].data.width, SpecLen::MinContent) =>
            {
                min_content_width_pooled(forest, i, fonts, mcw, inner)
            }
            _ if forest.nodes[i].box_type == BoxType::InlineBlock
                || forest.nodes[i].data.display == Display::InlineBlock =>
            {
                match definite_specified_content_width(forest, i) {
                    Some(definite) => definite,
                    // A replaced box with `width: auto` (or cyclic)
                    // contributes its intrinsic width (css-sizing-3 §5.2),
                    // not its (empty) generic content size.
                    None => intrinsic_content_width(forest, i)
                        .or_else(|| aspect_ratio_intrinsic_width(forest, i))
                        .unwrap_or_else(|| {
                            clamp_content_width_by_minmax(
                                forest,
                                i,
                                generic_preferred_width(forest, i),
                            )
                        }),
                }
            }
            _ => definite_width_floor(
                forest,
                i,
                generic_preferred_width(forest, i),
                fonts,
                mcw,
                inner,
            ),
        };
        // A scroll container's gutter reservation (stable
        // `scrollbar-gutter`, or the forced `overflow-y: scroll`
        // scrollbar) comes out of the content box at layout time, so the
        // intrinsic max-content size includes it — see
        // [`intrinsic_scrollbar_reservation`]. `width: min-content` arms
        // delegate to [`min_content_width_pooled`], which folds the
        // reservation in itself, so they add nothing here.
        let inline_spec = if forest.nodes[i].box_type == BoxType::Table
            || forest.nodes[i].data.display == Display::InlineTable
        {
            table_inline_spec(forest, i)
        } else {
            forest.nodes[i].data.width
        };
        let gutter = if matches!(inline_spec, SpecLen::MinContent) {
            0.0
        } else {
            intrinsic_scrollbar_reservation(forest, i)
        };
        // Fold the reservation inside the `min-width` / `max-width`
        // clamp, like [`min_content_width_pooled`] does: `w` is already
        // clamped by its arm and the clamp is idempotent, so re-clamping
        // only pulls the added gutter under a dominating `max-width`
        // instead of over-reporting by the reservation (up to 2 *
        // SCROLLBAR_THICKNESS under `stable both-edges`).
        forest.nodes[i].pref_w = if gutter > 0.0 {
            clamp_content_width_by_minmax(forest, i, w + gutter)
        } else {
            w
        };
        cache.set(i);
    }
    forest.nodes[idx].pref_w
}

/// Pooled scratch for [`preferred_width`]: the node-indexed min-content
/// width buffer plus the two [`WalkBufs`] walk pairs
/// [`preferred_width_mcw`] needs — `bufs` for the outer preorder walk,
/// `inner` for the nested per-node min-content measurements.
///
/// This is the thread-local twin of the explicit pooling the sibling
/// [`min_content_width_pooled`] gets at its call sites: `preferred_width`
/// is called once per intrinsic-width query with no caller-owned buffers
/// in scope, so one set per thread serves every call instead of
/// allocating a fresh triple per query. Reuse is sound for the same
/// reason fresh buffers were: [`min_content_width_pooled`] writes every
/// slot bottom-up before any read and both walks are cleared on entry,
/// so a stale entry is never observed.
///
/// The checkout is never re-entrant: no production path inside
/// [`preferred_width_mcw`] calls back into [`preferred_width`] — the
/// nested measurements all take their buffers as explicit parameters —
/// so `try_borrow_mut` only ever fails for a hypothetical future
/// re-entrant caller, which falls back to a fresh triple (exactly the
/// pre-pooling behavior) instead of panicking.
struct PrefWidthScratch {
    /// Node-indexed width buffer for the nested min-content measurements.
    mcw: Vec<f32>,
    /// Preorder walk buffers for the outer max-content walk.
    bufs: WalkBufs,
    /// Walk buffers for the nested per-node min-content measurements.
    inner: WalkBufs,
}

impl PrefWidthScratch {
    fn new() -> PrefWidthScratch {
        PrefWidthScratch {
            mcw: Vec::new(),
            bufs: WalkBufs::new(),
            inner: WalkBufs::new(),
        }
    }
}

thread_local! {
    static PREF_WIDTH_SCRATCH: RefCell<PrefWidthScratch> = RefCell::new(PrefWidthScratch::new());
}

/// [`preferred_width_mcw`] for callers without a reusable
/// [`min_content_width_pooled`] scratch buffer or pooled walk buffers
/// (other modules): checks out the thread-local [`PrefWidthScratch`]
/// instead of allocating a fresh buffer triple per call. Pooled reuse
/// is sound — see [`PrefWidthScratch`]; a re-entrant checkout falls
/// back to a fresh triple.
pub(crate) fn preferred_width(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    cache: &mut PrefWidthCache,
) -> f32 {
    // Re-entrancy probe: `try_borrow_mut` fails only if the scratch is
    // already checked out on this thread, and `try_with` fails only
    // while the thread-local is being destroyed. No production path
    // inside `preferred_width_mcw` calls back into this wrapper (its
    // nested measurements take explicit buffers), so the probe only
    // ever fires for a hypothetical future re-entrant caller — which
    // gets a fresh triple, exactly the pre-pooling behavior, instead of
    // a `RefCell` panic.
    let busy = PREF_WIDTH_SCRATCH
        .try_with(|cell| cell.try_borrow_mut().is_err())
        .unwrap_or(true);
    if busy {
        let mut mcw = Vec::new();
        let mut bufs = WalkBufs::new();
        let mut inner = WalkBufs::new();
        return preferred_width_mcw(forest, idx, fonts, cache, &mut mcw, &mut bufs, &mut inner);
    }
    // The probe's borrow is released before this runs, and no code on
    // this thread runs between the two straight-line steps, so the
    // `RefCell` state cannot change in between and this checkout cannot
    // fail.
    PREF_WIDTH_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let s = &mut *scratch;
        preferred_width_mcw(
            forest,
            idx,
            fonts,
            cache,
            &mut s.mcw,
            &mut s.bufs,
            &mut s.inner,
        )
    })
}

// ---------------------------------------------------------------------------
// Arena -> public tree
// ---------------------------------------------------------------------------

/// Whether `idx` is a flex/grid item: an in-flow child of a flex or grid
/// container. Per css-flexbox-1 and css-grid-1 §4.2, `z-index` values
/// other than `auto` create a stacking context on flex/grid items even
/// when `position: static`, so [`convert_forest`] must not drop the
/// resolved value for them the way it does for ordinary static boxes.
/// Mirrors paint's `is_flex_or_grid_container` on the *parent* (raw
/// `display` keyword, since [`Display`] buckets `flex`/`inline-flex`/
/// `inline-grid` away).
fn is_flex_or_grid_item(forest: &Forest, idx: usize) -> bool {
    let Some(p) = forest.nodes[idx].parent else {
        return false;
    };
    let pn = &forest.nodes[p];
    if pn.data.display == Display::Grid {
        return true;
    }
    matches!(
        pn.style.as_ref().and_then(|s| s.get_str_by_index(props::DISPLAY)),
        Some(d)
            if d.eq_ignore_ascii_case("flex")
                || d.eq_ignore_ascii_case("inline-flex")
                || d.eq_ignore_ascii_case("grid")
                || d.eq_ignore_ascii_case("inline-grid")
    )
}

/// Convert the arena into the public `LayoutBox` tree, iterative
/// post-order (children before parents).
pub(crate) fn convert_forest(
    forest: &mut Forest,
    root: usize,
) -> (super::LayoutBox, Vec<super::MarkerBox>) {
    let mut order: Vec<usize> = Vec::new();
    let mut stack = vec![root];
    while let Some(i) = stack.pop() {
        order.push(i);
        stack.extend(forest.nodes[i].children.iter().copied());
    }
    let mut out: Vec<Option<super::LayoutBox>> = (0..forest.nodes.len()).map(|_| None).collect();
    let mut markers: Vec<super::MarkerBox> = Vec::new();
    for i in order.into_iter().rev() {
        // Read before `n` borrows the node mutably below.
        let flex_grid_item = is_flex_or_grid_item(forest, i);
        let n = &mut forest.nodes[i];
        let children: Vec<super::LayoutBox> = n
            .children
            .drain(..)
            .map(|c| out[c].take().expect("children convert before parents"))
            .collect();
        // The shared arena `Arc<str>` moves straight through to the
        // public tree: no `String` materialization, no memcpy. Every
        // layout pass before this point consumed it by reference /
        // `Arc` clone.
        let text = n.text.take();
        // The memoized `Rc<ComputedStyle>` moves straight through: one
        // refcount bump per element box per pass, no deep clone of the
        // style's delta entries.
        let style = n.style.take();
        let node = n.dom;
        // The boxed fragment list moves out as a plain `Vec` for the
        // public tree (no reallocation: the `Vec` header moves out of
        // its box).
        let fragments = std::mem::take(&mut n.fragments).into_vec();
        // `z-index` applies to positioned boxes only (CSS 2.1 §9.9.1); on
        // static boxes it computes to `auto` — except on flex/grid items,
        // where css-flexbox-1 / css-grid-1 §4.2 give non-`auto` `z-index`
        // stacking-context semantics even when `position: static`.
        let z_index = match n.data.position {
            Position::Static if !flex_grid_item => None,
            _ => n.data.z_index,
        };
        let collapsed_borders = n.collapsed_borders.take();
        if let Some(m) = n.marker.take() {
            markers.push(super::MarkerBox {
                text: m.text,
                rect: m.rect.unwrap_or(Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 0.0,
                    h: 0.0,
                }),
                style: m.style,
                font_size: m.font_size,
                node,
            });
        }
        out[i] = Some(super::LayoutBox {
            rect: n.rect,
            children,
            box_type: n.box_type,
            style,
            text,
            node,
            fragments,
            z_index,
            collapsed_borders,
            phantom_w: n.phantom_w,
            phantom_h: n.phantom_h,
        });
    }
    (out[root].take().expect("root converts"), markers)
}

// ---------------------------------------------------------------------------
// Flexbox layout
// ---------------------------------------------------------------------------
//
// Flexbox: `display: flex` / `display: inline-flex` containers lay out
// their children as flex items. Supported: `flex-direction`
// (row/row-reverse/column/column-reverse), `flex-wrap`
// (nowrap/wrap/wrap-reverse), `justify-content`
// (flex-start/center/flex-end/space-between/space-around/space-evenly),
// `align-items`/`align-self`
// (stretch/center/flex-start/flex-end/baseline), `align-content`
// (stretch/center/flex-start/flex-end/space-between/space-around/
// space-evenly), and `flex-grow`/`flex-shrink`/`flex-basis` resolution.
// With `flex-wrap: wrap | wrap-reverse` items break into flex lines
// (flexbox §9.3); each line resolves exactly like a single-line container
// and the lines stack per `align-content` (flexbox §9.5). The multi-line
// path lives in `layout_flex_wrap`; `nowrap` keeps the single-line driver
// below.
//
// The driver is the same task stack as block layout (`Task::Flex`
// resumes): row containers place children at provisional cross positions
// and finish the cross axis on resume; column containers with auto
// flex-basis/auto height items measure those first, then place.
//
// Documented simplifications for this pass:
// - `flex-wrap: wrap | wrap-reverse` break items into lines per flexbox
//   §9.3 (see `layout_flex_wrap`); a forced `break-before`/`break-after`
//   value starts/ends a flex line. Column wrap needs a definite
//   container height to break; with an auto height everything stays on
//   one line.
// - Text and inline-level children become atomic flex items handled by a
//   small greedy line breaker here rather than the full inline engine:
//   nested inline boxes contribute their text runs, but their own
//   border/padding and non-baseline `vertical-align` are ignored.
// - Whitespace-only text items collapse to zero size.
// - On the column axis an auto cross size fills the container width
//   (equivalent to `stretch`); only specified widths take part in
//   cross-axis alignment.
// - Flex item margins never collapse with each other or with the
//   container. Inside an item, normal block-container collapsing still
//   applies (the item's top margin can absorb its first child's).
// - Baseline alignment ignores item margins.
// - A stretched text item keeps its natural height and is positioned at
//   the cross start.

/// Raw `display` flex kind. `style::display_of` buckets `flex` into
/// block/inline-block, so the raw value is consulted here: `Some(true)`
/// is `display: flex` (block-level), `Some(false)` is `inline-flex`.
pub(crate) fn flex_kind(forest: &Forest, idx: usize) -> Option<bool> {
    let d = forest.nodes[idx]
        .style
        .as_ref()
        .and_then(|s| s.get_str_by_index(props::DISPLAY))?;
    if d.eq_ignore_ascii_case("flex") {
        Some(true)
    } else if d.eq_ignore_ascii_case("inline-flex") {
        Some(false)
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FlexDir {
    Row,
    RowRev,
    Col,
    ColRev,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FlexJustify {
    Start,
    End,
    Center,
    Between,
    Around,
    Evenly,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FlexAlign {
    Stretch,
    Start,
    End,
    Center,
    Baseline,
}

pub(crate) fn flex_dir_of(style: Option<&ComputedStyle>) -> FlexDir {
    match style.and_then(|s| s.get_str_by_index(props::FLEX_DIRECTION)) {
        Some(d) if d.eq_ignore_ascii_case("row-reverse") => FlexDir::RowRev,
        Some(d) if d.eq_ignore_ascii_case("column") => FlexDir::Col,
        Some(d) if d.eq_ignore_ascii_case("column-reverse") => FlexDir::ColRev,
        _ => FlexDir::Row,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FlexWrap {
    NoWrap,
    Wrap,
    WrapReverse,
}

fn flex_wrap_of(style: Option<&ComputedStyle>) -> FlexWrap {
    match style.and_then(|s| s.get_str_by_index(props::FLEX_WRAP)) {
        Some(w) if w.eq_ignore_ascii_case("wrap") => FlexWrap::Wrap,
        Some(w) if w.eq_ignore_ascii_case("wrap-reverse") => FlexWrap::WrapReverse,
        // `nowrap` | missing
        _ => FlexWrap::NoWrap,
    }
}

/// Whether a `break-before`/`break-after` computed value forces a break
/// (CSS Fragmentation §3 forced-break family): `column` plus the
/// page-break aliases (which fragment as column breaks inside multicol)
/// and `always`, all matched case-insensitively. In a flex container the
/// forced break becomes a flex line break (flexbox §9.3).
fn forced_break_of(style: Option<&ComputedStyle>, prop: u8) -> bool {
    matches!(
        style.and_then(|s| s.get_str_by_index(prop)),
        Some(v) if v.eq_ignore_ascii_case("page")
            || v.eq_ignore_ascii_case("left")
            || v.eq_ignore_ascii_case("right")
            || v.eq_ignore_ascii_case("recto")
            || v.eq_ignore_ascii_case("verso")
            || v.eq_ignore_ascii_case("column")
            || v.eq_ignore_ascii_case("always")
    )
}

fn flex_justify_of(style: Option<&ComputedStyle>) -> FlexJustify {
    match style.and_then(|s| s.get_str_by_index(props::JUSTIFY_CONTENT)) {
        Some(v) if v.eq_ignore_ascii_case("center") => FlexJustify::Center,
        Some(v) if v.eq_ignore_ascii_case("flex-end") || v.eq_ignore_ascii_case("end") => {
            FlexJustify::End
        }
        Some(v) if v.eq_ignore_ascii_case("space-between") => FlexJustify::Between,
        Some(v) if v.eq_ignore_ascii_case("space-around") => FlexJustify::Around,
        Some(v) if v.eq_ignore_ascii_case("space-evenly") => FlexJustify::Evenly,
        // CSS Box Alignment 3 §4.2: `left`/`right` are *physical*
        // content-position keywords — they always pack against the
        // physical left/right edge and never flip with the container's
        // `direction`. The mapping onto `Start`/`End` compensates for
        // the direction-relative main axis so the result stays
        // physical: in an rtl row container main-start is the right
        // edge, so `left` must map to `End` there.
        Some(v) if v.eq_ignore_ascii_case("left") || v.eq_ignore_ascii_case("right") => {
            let rtl = style
                .and_then(|s| s.get_str_by_index(props::DIRECTION))
                .is_some_and(|d| d.eq_ignore_ascii_case("rtl"));
            match (v.eq_ignore_ascii_case("left"), rtl) {
                (true, false) | (false, true) => FlexJustify::Start,
                _ => FlexJustify::End,
            }
        }
        // `flex-start` | `start` | `normal` | missing
        _ => FlexJustify::Start,
    }
}

/// `None` for `auto`/`normal`/missing (the caller substitutes the default).
/// CSS Box Alignment 3 §4.1: `self-start` / `self-end` resolve against
/// the *subject's* (the flex item's) writing mode and direction, not the
/// container's. `self_rtl` is true when the subject's inline-start edge
/// is the axis's *end* edge — i.e. the item's inline direction runs
/// opposite to the axis's start direction — so `self-start` maps to
/// `End` and `self-end` to `Start`.
pub(crate) fn flex_align_of(value: Option<&str>, self_rtl: bool) -> Option<FlexAlign> {
    match value {
        Some(v) if v.eq_ignore_ascii_case("stretch") => Some(FlexAlign::Stretch),
        Some(v) if v.eq_ignore_ascii_case("flex-start") || v.eq_ignore_ascii_case("start") => {
            Some(FlexAlign::Start)
        }
        Some(v) if v.eq_ignore_ascii_case("self-start") => Some(if self_rtl {
            FlexAlign::End
        } else {
            FlexAlign::Start
        }),
        Some(v) if v.eq_ignore_ascii_case("flex-end") || v.eq_ignore_ascii_case("end") => {
            Some(FlexAlign::End)
        }
        Some(v) if v.eq_ignore_ascii_case("self-end") => Some(if self_rtl {
            FlexAlign::Start
        } else {
            FlexAlign::End
        }),
        Some(v) if v.eq_ignore_ascii_case("center") => Some(FlexAlign::Center),
        Some(v) if v.eq_ignore_ascii_case("baseline") => Some(FlexAlign::Baseline),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AlignContent {
    Stretch,
    Start,
    End,
    Center,
    Between,
    Around,
    Evenly,
}

/// `align-content` (flexbox §9.5); default `stretch`.
fn align_content_of(style: Option<&ComputedStyle>) -> AlignContent {
    match style.and_then(|s| s.get_str_by_index(props::ALIGN_CONTENT)) {
        Some(v) if v.eq_ignore_ascii_case("flex-start") || v.eq_ignore_ascii_case("start") => {
            AlignContent::Start
        }
        Some(v) if v.eq_ignore_ascii_case("flex-end") || v.eq_ignore_ascii_case("end") => {
            AlignContent::End
        }
        Some(v) if v.eq_ignore_ascii_case("center") => AlignContent::Center,
        Some(v) if v.eq_ignore_ascii_case("space-between") => AlignContent::Between,
        Some(v) if v.eq_ignore_ascii_case("space-around") => AlignContent::Around,
        Some(v) if v.eq_ignore_ascii_case("space-evenly") => AlignContent::Evenly,
        // `stretch` | `normal` | missing
        _ => AlignContent::Stretch,
    }
}

/// Resolve one flex gap longhand (`row-gap` / `column-gap`; the `gap`
/// / `grid-gap` shorthands expand into them at cascade time). Percentage
/// gaps resolve against the content-box size in the gap's own dimension;
/// against an indefinite size they are zero (css-align-3 §2.2), and the
/// used value is never negative.
fn flex_gap_len(
    style: Option<&ComputedStyle>,
    prop: u8,
    font_size: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    pct_base: Option<f32>,
) -> f32 {
    let spec = spec_len_of(style.and_then(|s| s.get_by_index(prop)));
    if matches!(spec, SpecLen::Pct(_)) && pct_base.is_none() {
        return 0.0;
    }
    spec.resolve(font_size, root_fs, vp_w, vp_h, pct_base.unwrap_or(0.0))
        .unwrap_or(0.0)
        .max(0.0)
}

/// Resolve both flex gap longhands with axis mapping: the main-axis gap
/// is `column-gap` for row containers and `row-gap` for column
/// containers; the cross-axis gap is the other longhand. `bases` is
/// `(font_size, root_font_size, vp_w, vp_h)`.
fn flex_gaps(
    style: Option<&ComputedStyle>,
    horizontal: bool,
    bases: (f32, f32, f32, f32),
    main_pct: Option<f32>,
    cross_pct: Option<f32>,
) -> (f32, f32) {
    let (font_size, root_fs, vp_w, vp_h) = bases;
    if horizontal {
        (
            flex_gap_len(
                style,
                props::COLUMN_GAP,
                font_size,
                root_fs,
                vp_w,
                vp_h,
                main_pct,
            ),
            flex_gap_len(
                style,
                props::ROW_GAP,
                font_size,
                root_fs,
                vp_w,
                vp_h,
                cross_pct,
            ),
        )
    } else {
        (
            flex_gap_len(
                style,
                props::ROW_GAP,
                font_size,
                root_fs,
                vp_w,
                vp_h,
                main_pct,
            ),
            flex_gap_len(
                style,
                props::COLUMN_GAP,
                font_size,
                root_fs,
                vp_w,
                vp_h,
                cross_pct,
            ),
        )
    }
}

/// A numeric flex factor (`flex-grow`/`flex-shrink`), clamped to be
/// non-negative per spec and to a finite bound: `flex-grow: 1e999`
/// parses to infinity, and an infinite factor makes [`flex_lengths`]
/// compute `remaining / inf * inf` = NaN, whose total violation never
/// compares `== 0.0`, hanging the distribution loop forever.
// `clippy::manual_clamp` is allowed: `f32::clamp` deliberately does not
// fix NaN, so the explicit min/max (which maps NaN to 0.0) is the point.
const MAX_FLEX_FACTOR: f32 = 1e6;

/// Saturating bound for used sizes derived from infinite intrinsic
/// sizes. A fixed-layout table with a percentage width has an infinite
/// max-content intrinsic inline size per Blink (WPT
/// `css/css-tables/tentative/table-fixed-minmax.html`), and that
/// infinity propagates through flex/grid max-content into used sizes.
/// Blink saturates `LayoutUnit` the same way: raw infinities must never
/// reach used rects — downstream paint/overflow/transform math goes NaN
/// on them — while the size stays observably "infinite" (the WPT asserts
/// `> 1268px`).
pub(crate) const MAX_USED_SIZE: f32 = 1e7;

/// Saturate an infinite used size to [`MAX_USED_SIZE`]; finite values
/// pass through unchanged. NaN is not sanitized here — callers must
/// never produce it (see the non-finite guard in [`flex_lengths`]).
pub(crate) fn saturate_used_size(v: f32) -> f32 {
    if v.is_infinite() {
        v.signum() * MAX_USED_SIZE
    } else {
        v
    }
}

// `clippy::manual_clamp` is allowed: `f32::clamp` deliberately does not
// fix NaN, so the explicit min/max (which maps NaN to 0.0) is the point.
#[allow(clippy::manual_clamp)]
fn flex_factor(style: Option<&ComputedStyle>, prop: u8, default: f32) -> f32 {
    match style.and_then(|s| s.get_by_index(prop)) {
        Some(CssValue::Number(n)) => n.max(0.0).min(MAX_FLEX_FACTOR),
        _ => default,
    }
}

/// One flex item: a direct child of the flex container.
#[derive(Clone, Debug)]
struct FlexItem {
    idx: usize,
    /// The `order` property (initial 0): items lay out in ascending
    /// order, ties in document order (flexbox §5.4).
    order: i32,
    grow: f32,
    shrink: f32,
    /// Hypothetical main size (content box), clamped by min/max.
    base: f32,
    /// Post-flexing main size (content box).
    target: f32,
    min_main: f32,
    max_main: f32,
    /// Main-start/main-end and cross-start/cross-end margins.
    m_ms: f32,
    m_me: f32,
    m_cs: f32,
    m_ce: f32,
    /// Whether the main-start/main-end margin was `auto`: auto margins
    /// absorb positive free space before `justify-content` runs.
    auto_ms: bool,
    auto_me: bool,
    /// Whether the cross-start/cross-end margin was `auto`: auto margins
    /// absorb positive free space before `align-self` runs, and any auto
    /// cross margin suppresses `stretch` (flexbox §8.3, §9.8).
    auto_cs: bool,
    auto_ce: bool,
    /// Automatic minimum main size (min-content) when `min-width` /
    /// `min-height` is `auto`; 0.0 when a min is explicitly set or the
    /// axis is unmodeled (column main sizes). Floors shrink targets so
    /// items cannot collapse below their content.
    auto_min: f32,
    /// Border+padding totals on each axis.
    main_bp: f32,
    cross_bp: f32,
    align: FlexAlign,
    /// Whether the cross size (width on the column axis) is auto.
    cross_auto: bool,
    /// Final cross-axis content size.
    cross: f32,
    /// Natural cross-axis content size (before stretch).
    natural_cross: f32,
    /// Baseline offset from the border-box cross-start edge.
    baseline: f32,
    /// Border-box main-start offset from the container content-box
    /// main-start edge (physical, after reverse mapping).
    main_pos: f32,
    /// Border-box cross-start offset from the container content-box
    /// cross-start edge.
    cross_pos: f32,
    /// Column only: the natural main size needs a measuring layout.
    needs_measure: bool,
    /// Column only: when `needs_measure` is set for a content-keyword
    /// basis, the width the measuring layout uses instead of the
    /// final cross size — the keyword width (min/max-content, or the
    /// fit-content clamp by the container width). `None` measures at
    /// the cross size, as before. Both the single-line column path
    /// and the wrap-column path (`layout_flex_wrap`) read this field
    /// for their measuring tasks.
    measure_cross: Option<f32>,
    /// Column only: when `needs_measure` is set for a content-keyword
    /// basis, the measuring layout must ignore the item's specified
    /// height (a `flex-basis` keyword overrides the main-size
    /// property, css-flexbox-1 §9.2). The task carries a zero
    /// minimum-height override instead, so the measured height is
    /// the natural height at the keyword width. Only generic boxes
    /// get this: flex/grid/table/multicol items keep the old
    /// definite-height-wins measurement (their own layout owns the
    /// height override semantics).
    measure_height_min: bool,
    /// Row only: needs a second layout pass with the stretched height.
    stretched: bool,
    /// Forced flex line break before this item (`break-before` with a
    /// forced-break value, flexbox §9.3): it starts a new flex line.
    force_break_before: bool,
    /// Forced flex line break after this item (`break-after` with a
    /// forced-break value, flexbox §9.3): its flex line ends after it.
    force_break_after: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FlexStage {
    /// Row: children laid out at provisional cross positions; now finish
    /// the cross axis.
    RowFinish,
    /// Row: stretched items re-laid out; now finalize the container.
    RowDone,
    /// Column: natural main sizes measured; now place for real.
    ColumnMeasured,
    /// Column: children laid out; now finalize the container.
    ColumnDone,
    /// Row wrap (`flex-wrap: wrap | wrap-reverse`): children laid out at
    /// provisional cross positions; now stack the flex lines and finish
    /// the cross axis.
    WrapRowFinish,
    /// Row wrap: stretched items re-laid out; now finalize the container.
    WrapRowDone,
    /// Column wrap: natural main sizes measured; now collect flex lines
    /// and place for real.
    WrapColumnMeasured,
    /// Column wrap: children laid out; now finalize the container.
    WrapColumnDone,
}

#[derive(Clone, Debug)]
pub(crate) struct FlexCtx {
    idx: usize,
    content_x: f32,
    content_top: f32,
    content_w: f32,
    cb_h: Option<f32>,
    hspec: Option<f32>,
    reverse: bool,
    /// Column only: `direction: rtl` flips the cross axis (flexbox
    /// §5.1: cross-start is the right edge). Item cross margins are
    /// stored cross-start-first at build; placement mirrors the
    /// cross-start-relative offset to physical x.
    cross_rtl: bool,
    justify: FlexJustify,
    items: Vec<FlexItem>,
    stage: FlexStage,
    /// Container content-box cross size, fixed in `RowFinish`.
    container_cross: f32,
    /// Wrap path (`flex-wrap: wrap | wrap-reverse`): flex lines as
    /// `(start, end)` ranges into `items` (flexbox §9.3).
    lines: Vec<(usize, usize)>,
    /// Wrap path: `align-content` (flexbox §9.5).
    align_content: AlignContent,
    /// Wrap path: `wrap-reverse` stacks lines in the reverse cross
    /// direction.
    wrap_reverse: bool,
    /// Resolved main-axis gap (flexbox §8.2): `column-gap` for row
    /// containers, `row-gap` for column containers. Gaps are constant
    /// strides between adjacent items on a line; they count toward line
    /// breaking (§9.3) and are subtracted before flexible lengths (§9.7)
    /// and justification (§9.6) run.
    main_gap: f32,
    /// Resolved cross-axis gap: `row-gap` for row containers,
    /// `column-gap` for column containers. Fixed stride between adjacent
    /// flex lines (flexbox §9.5).
    cross_gap: f32,
    /// The original block input this flex layout was driven from: the
    /// `overflow-y: auto` scrollbar re-drive re-runs `layout_flex` with
    /// the identical input against a narrowed width.
    input: BlockInput,
    /// Whether this context already is the `overflow-y: auto`
    /// scrollbar-width re-drive: the re-drive runs at most once per
    /// frame, so narrowing can never oscillate or loop to a fixpoint.
    auto_redriven: bool,
}

/// Clamp a flex base size by the main-axis min/max, with min-width
/// winning when over-constrained (mirrors `resolve_metrics`).
///
/// Percentage clamps resolve against the container's content-box main
/// size; against an indefinite size they behave as no clamp.
///
/// css-sizing-3 §4 (F4): `min-width`/`max-width` accept the
/// content-based keywords; `SpecLen::resolve` returns `None` for them,
/// so measure here where the font context exists. Only the horizontal
/// axis is measurable (min-content height is unmodeled); the vertical
/// axis keeps the static resolution. `avail` is the item's available
/// content-box width (see [`content_based_width`]); it is unused when
/// `horizontal` is false.
///
/// Returns `(clamped, min, max)`.
#[allow(clippy::too_many_arguments)]
fn clamp_flex_main(
    forest: &mut Forest,
    idx: usize,
    base: f32,
    min_s: SpecLen,
    max_s: SpecLen,
    fs: f32,
    root_fs: f32,
    vp_w: f32,
    vp_h: f32,
    main_pct_base: Option<f32>,
    horizontal: bool,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
    avail: f32,
) -> (f32, f32, f32) {
    let resolve = |s: SpecLen| match (s, main_pct_base) {
        (SpecLen::Pct(_), None) => None,
        // CSS Values 4 §8.1.4: a `calc()` that syntactically contained
        // a `%` leaf is cyclic against an indefinite main-size base and
        // behaves as `auto`/`none`, like a bare percentage (mirrors the
        // gate in `basis_or` and the cross-size clamp). `None` maps to
        // the caller's default: 0.0 for min, INFINITY for max.
        (SpecLen::Math { had_pct, .. }, None) if had_pct => None,
        _ => s.resolve(fs, root_fs, vp_w, vp_h, main_pct_base.unwrap_or(0.0)),
    };
    // Content-based keywords measure against the item's available
    // width; every other spec resolves statically exactly as before.
    let min_v = if horizontal {
        measured_minmax_width(
            &mut *forest,
            idx,
            min_s,
            fonts,
            pw_cache,
            mcw,
            bufs,
            inner,
            avail,
        )
        .unwrap_or_else(|| resolve(min_s).unwrap_or(0.0))
    } else {
        resolve(min_s).unwrap_or(0.0)
    }
    .max(0.0);
    let max_v = if horizontal {
        measured_minmax_width(
            &mut *forest,
            idx,
            max_s,
            fonts,
            pw_cache,
            mcw,
            bufs,
            inner,
            avail,
        )
        .unwrap_or_else(|| resolve(max_s).unwrap_or(f32::INFINITY))
    } else {
        resolve(max_s).unwrap_or(f32::INFINITY)
    }
    .max(0.0);
    (base.min(max_v).max(min_v).max(0.0), min_v, max_v)
}

/// Min-content width of a flex item (CSS flexbox §4.5: the automatic
/// minimum size is the min-content size, not max-content). A text run
/// contributes its longest unbreakable run under its own `white-space`
/// (`normal`: longest word; `pre`: longest forced-broken segment;
/// `nowrap`: the whole run; `pre-wrap`/`pre-line`/`break-spaces`: longest
/// word — `break-spaces` only adds a break opportunity *after* each
/// preserved space, so a spaceless word is still unbreakable). Inline
/// boxes sum their children; other boxes take the max over children.
/// Returns content width only; border/padding ride along via the item's
/// `main_bp`.
///
/// Iterative post-order over an explicit stack: the recursive version
/// overflowed the Rust stack on deep subtrees inside flex items, and
/// every sibling helper is iterative for the same reason.
///
/// `scratch` is a caller-owned reusable buffer indexed by node id; it is
/// only grown (never reallocated per call), so repeated calls over many
/// items share one allocation. Slots are written bottom-up before any
/// read: every node read here is a descendant of `idx`, hence present in
/// `order` and evaluated earlier in the reversed pass, so stale slots
/// from a previous call are never observed.
///
/// Each child's used horizontal padding/border/margins ride along with
/// its contribution (via [`intrinsic_h_extras`]); the box's own are left
/// for the caller.
///
/// Atomic inline-level boxes (`BoxType::InlineBlock`: inline-block,
/// inline-table, inline-flex, inline-grid, and replaced inline elements)
/// with a definite specified width contribute exactly that width to the
/// min-content size (css-sizing-3 §5.2): the contribution is the
/// content-box of a hypothetical auto-sized float containing only that
/// box with a zero-sized containing block, and a definite-width box lays
/// out at its definite width there, so content overflowing it does not
/// grow the contribution. The returned content width is the specified
/// content width (the specified width minus the box's own
/// border/padding under `border-box`); when the width is `auto` (or
/// resolves against no definite base), the contribution is the max of
/// the children's intrinsic contributions. Percentages resolve against
/// a zero base here, i.e. they behave as `auto` (css-sizing-3 §5.2.1).
/// Min-content width of a table box (CSS 2.1 §17.5.2.2): the
/// automatic-layout column minima via the shared
/// [`layout_table::auto_column_minima`] helper — spanning cells
/// (step 3), specified `<col>`/`<colgroup>` widths (steps 2/4),
/// specified cell widths (step 1: W > MCW), and `min-width` floors all
/// contribute. Percentages resolve against a zero base (no definite
/// table width exists this early); inter-column spacing stays excluded
/// by design (the caller adds it). CSS 2.1 §17.5.2.2 CAPMIN — the
/// widest caption's outer minimum — also floors the total: the walk's
/// convention is content-box, so CAPMIN is stripped of the table's
/// inline-axis border+padding first (physical top/bottom in a vertical
/// writing mode), mirroring the block-level table's used-width
/// `capmin_c` discipline. Shared by the `Table` arm of
/// [`min_content_width`] and its `inline-table` guard (an `inline-table`
/// box is an `InlineBlock` whose inner layout is the table algorithm).
/// Split form of [`table_min_content_width`]: the column-minima sum
/// (spacing excluded, by the walk's convention) and the stripped
/// CAPMIN floor, separately, so callers that add spacing to the
/// column side only (see [`flex_basis_table_min`]) need not undo the
/// max afterwards.
fn table_min_content_split(
    forest: &mut Forest,
    i: usize,
    fonts: &dyn FontMetrics,
    pool: &mut layout_table::TableGridPool,
) -> (f32, f32) {
    let mut grid = pool.take_grid();
    layout_table::collect_table_grid_into(forest, i, pool, &mut grid);
    let (_, spacing_h, _) = layout_table::table_border_model(forest, i);
    let col_spec = layout_table::column_specified_widths(forest, &grid, 0.0, 0.0);
    let mut pw_cache = PrefWidthCache::new(forest.nodes.len());
    let mut mcw_scratch = Vec::new();
    let mut int_bufs = WalkBufs::new();
    let mut int_inner = WalkBufs::new();
    let (col_max, col_max_nonspanning) = layout_table::auto_column_maxima(
        forest,
        &grid,
        &col_spec,
        spacing_h,
        fonts,
        &mut pw_cache,
        &mut mcw_scratch,
        &mut int_bufs,
        &mut int_inner,
    );
    let cols: f32 = layout_table::auto_column_minima(
        forest,
        &grid,
        &col_spec,
        spacing_h,
        0.0,
        fonts,
        &col_max,
        &col_max_nonspanning,
    )
    .iter()
    .sum();
    pool.put_grid(grid);
    // CSS 2.1 §17.5.2.2: the used width is at least max(MIN, CAPMIN).
    // CAPMIN (`table_caption_min_width`) is an *outer* width; this walk
    // returns content-box, so strip the table's inline-axis
    // border+padding first (physical top/bottom in a vertical writing
    // mode), mirroring the block-level table's used-width `capmin_c`
    // discipline. Folding the unconverted outer CAPMIN in here mixes
    // units and overgrows bordered tables (WPT `caption.html` table 6:
    // 50px borders + a 50px-min caption must stay 100px, not 150px).
    let capmin = layout_table::table_caption_min_width(forest, i, fonts, &mut mcw_scratch, pool);
    let inline_bp = if is_vertical_table(forest, i) {
        intrinsic_v_bp(forest, i)
    } else {
        intrinsic_h_bp(forest, i)
    };
    (cols, (capmin - inline_bp).max(0.0))
}

fn table_min_content_width(
    forest: &mut Forest,
    i: usize,
    fonts: &dyn FontMetrics,
    pool: &mut layout_table::TableGridPool,
) -> f32 {
    let (cols, capmin_c) = table_min_content_split(forest, i, fonts, pool);
    cols.max(capmin_c)
}

/// Whether box `i` is a cell of an actual table grid: such a cell is
/// sized by the table algorithm, whose intrinsic rules (CSS Tables
/// 3 §3.8.3) differ from the standalone definite-floor rule in
/// [`min_content_width`]. The walk goes up only through the row
/// structure a cell can belong to — `table-row` / `table-row-group`
/// boxes (including the anonymous ones the engine inserts for row
/// structure; a bare cell directly under the table also counts, as
/// the grid collection groups those into rows itself) — and succeeds
/// on reaching the table box. Any other intervening box (a block, an
/// inline, and in particular another table cell: a `display:
/// table-cell` box nested inside a `td` through other boxes belongs
/// to no grid and would get its own anonymous table per CSS 2.1
/// §17.2.1) ends the walk as *not* in a table context, even if a
/// table exists further up. Iterative, per the crate's
/// never-recurse invariant.
fn cell_in_table_context(forest: &Forest, i: usize) -> bool {
    let mut cur = forest.nodes[i].parent;
    // Hang guard (mirrors `resolve_containing_block` in
    // `layout_position`): a parent cycle is unreachable from well-formed
    // box construction, but an internal bug could stitch one in. Capping
    // the steps at the node count terminates any cycle; the guard fails
    // closed (standalone cell) instead of hanging the layout thread.
    let mut steps = forest.nodes.len();
    while let Some(p) = cur {
        if steps == 0 {
            return false;
        }
        steps -= 1;
        let n = &forest.nodes[p];
        match n.box_type {
            BoxType::TableRow | BoxType::TableRowGroup => cur = n.parent,
            BoxType::Table => return true,
            _ if n.data.display == Display::InlineTable => return true,
            _ => return false,
        }
    }
    false
}

/// [`min_content_width_pooled`] for callers without a reusable walk
/// buffer (tests): a fresh [`WalkBufs`] per call. `pub(crate)` for
/// the w78fix2 forest-level min-content pin in `tests.rs` (like
/// `PrefWidthCache::new` for the w78fix H1 pref-level test).
#[cfg(test)]
pub(crate) fn min_content_width(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    scratch: &mut Vec<f32>,
) -> f32 {
    let mut bufs = WalkBufs::new();
    min_content_width_pooled(forest, idx, fonts, scratch, &mut bufs)
}

/// Min-content width, iterative post-order over [`WalkBufs`].
///
/// `scratch` is the node-indexed width buffer, resized to the forest on
/// first use and written bottom-up before any read, so sharing one
/// buffer across sequential calls is sound. `bufs` is caller-provided
/// preorder-walk scratch: pooled in [`LayoutScratch`]
/// (`int_bufs`/`int_inner`) at top-level call sites and hoisted per
/// container in the table/grid intrinsic loops, so one walk serves many
/// measurements instead of allocating per call. It is cleared on entry
/// and must not alias an in-progress outer intrinsic walk's buffers —
/// nested measurements (the table/grid arms below) use their own —
/// which the borrow checker enforces where the buffers are threaded as
/// parameters.
///
/// The flex arm's clamp adjustment needs scratch of its own (`out` is
/// the live min-content buffer the child closure reads, and this
/// dispatch carries no pref-width cache): one reusable aux width
/// buffer, one generation-stamped [`PrefWidthCache`] (invalidated per
/// flex node via [`PrefWidthCache::next_gen`] instead of reallocated),
/// and one nested walk pair for the content-keyword measurements are
/// hoisted into the [`WalkBufs`] itself (see [`FlexScratch`]) — not per
/// flex node and not per walk — so a flex-heavy subtree costs buffer
/// growth once per pooled-buffer lifetime.
///
/// Every node's slot is clamped by its own `min-width` / `max-width`
/// ([`clamp_content_width_by_minmax`]): a content-based contribution
/// that ignores them would over-report where the box's used size is
/// bounded, the way [`definite_specified_content_width`] and
/// [`intrinsic_content_width`] already clamp. See
/// [`min_content_width_pooled_raw`] for the unclamped twin, used where
/// the intrinsic *size* (not the used width) is needed — the flex
/// automatic minimum (flexbox §4.5), which `max-width` does not bound
/// (min wins when over-constrained, css-sizing-3 §4).
fn min_content_width_pooled(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    scratch: &mut Vec<f32>,
    bufs: &mut WalkBufs,
) -> f32 {
    min_content_width_pooled_impl(forest, idx, fonts, scratch, bufs, true)
}

/// Unclamped twin of [`min_content_width_pooled`]: the min-content
/// *size* with no `min-width` / `max-width` clamp applied. The flex
/// automatic minimum (flexbox §4.5) is the min-content size, and a
/// `max-width` does not bound it — when `min-width: auto` the used
/// minimum is the unclamped size (min wins over max, css-sizing-3 §4).
fn min_content_width_pooled_raw(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    scratch: &mut Vec<f32>,
    bufs: &mut WalkBufs,
) -> f32 {
    min_content_width_pooled_impl(forest, idx, fonts, scratch, bufs, false)
}

/// Min-content width of a table cell's inline formatting context in
/// full quirks mode (WHATWG Quirks §3.8, the table cell width
/// calculation quirk): `img` elements that are inline-level replaced
/// elements in that context have no soft wrap opportunity before or
/// after them, so a run of them contributes its summed width
/// instead of the widest single element. Applies to the cell node
/// itself and to an anonymous block wrapping its inline run alike —
/// both iterate the same direct children.
///
/// Three w77fix corrections to the original form:
/// - A float interrupts inline flow, so a floated replaced element
///   breaks the run (and contributes its own width, like any other
///   breaker) instead of joining it — floated non-replaced children
///   already broke runs, and the two cases must agree.
/// - An `Inline` container is transparent to the run: its children
///   are walked in place, sharing the run state, so a run continues
///   across a span boundary (`<img><span><img></span><img>` sums
///   all three — they are consecutive `img`s in one formatting
///   context) while text inside the span still breaks it
///   (`<span><img>hello<img></span>` contributes the largest
///   single run inside the span, not the span's summed width).
/// - Only `img` elements join a run (§3.8 names `img`, not
///   replaced elements generally): any other replaced element
///   (`object`, `video`, `input`, …) breaks the run and contributes
///   its own width. `is_img` is stamped at box construction (see
///   [`Node::is_img`]); the forest carries no tag names.
fn quirks_cell_inline_min_content(
    forest: &Forest,
    idx: usize,
    out: &[f32],
    fonts: &dyn FontMetrics,
) -> f32 {
    let mut best = 0.0f32;
    let mut run = 0.0f32;
    quirks_run_walk(forest, idx, out, &mut best, &mut run, fonts);
    best.max(run)
}

/// Whether an ignorable-whitespace node (see
/// [`is_ignorable_whitespace`]) collapses away, so the §3.8 walk
/// may skip it without changing the run: `normal`, `nowrap` and
/// `pre-line` spaces collapse (the same set
/// [`seam_collapsible_ws`] treats as collapsible), while `pre`,
/// `pre-wrap` and `break-spaces` spaces are preserved and keep
/// their width. An anonymous block wrapping only whitespace text
/// collapses exactly when every such text child collapses (its
/// definition admits no other child kinds).
fn quirks_ws_is_collapsible(forest: &Forest, idx: usize) -> bool {
    let n = &forest.nodes[idx];
    match n.box_type {
        BoxType::Text => seam_collapsible_ws(n.data.white_space),
        BoxType::AnonymousBlock => n.children.iter().all(|&c| {
            let cn = &forest.nodes[c];
            cn.box_type == BoxType::Text && seam_collapsible_ws(cn.data.white_space)
        }),
        _ => false,
    }
}

/// One level of the §3.8 walk (see
/// [`quirks_cell_inline_min_content`]): `best` is the widest
/// closed run or breaker so far, `run` the open run. `Inline`
/// containers (other than `<br>`, which breaks the run) recurse
/// transparently, sharing both accumulators. Collapsible
/// whitespace nodes are skipped at the loop head (w78fix M3):
/// §3.8 removes the soft wrap opportunity adjacent to `img`s, so
/// an intervening space or newline must not break the run in
/// either axis — the vertical twin
/// [`vertical_quirks_run_step`] plus `vertical_flow_content`'s
/// loop-head skip already summed across it, while this walk used
/// to treat the whitespace text node as a breaker and measure
/// `<img> <img>` as max(50, 50) instead of 100. The loop-head skip
/// also covers `Inline` recursion: a nested container's whitespace
/// children are skipped by the recursive call's own loop head.
///
/// w78fix2: the skip is white-space-aware (see
/// [`quirks_ws_is_collapsible`]). Skipping unconditionally was
/// white-space-blind: a preserved space in a `pre` cell was
/// dropped, so `<img> <img>` measured 100 instead of
/// 100 + space. A preserved whitespace text node instead joins
/// the open run at its measured advance — it cannot be skipped
/// (its width is real) and it cannot break the run (that would
/// give max(50, 50) again). A preserved newline is a forced
/// break, so a newline-containing node splits at each `\n`:
/// each segment's advance joins the run on its own line, and
/// the run closes between segments. The advance is measured
/// directly with `fonts` rather than read from `out`: the
/// generic min-content text arm splits `pre-wrap` /
/// `break-spaces` text on spaces (giving 0 for a lone space),
/// which is right for a breakable run but wrong here, where
/// §3.8 has removed the break opportunity. An anonymous block
/// wrapping preserved whitespace recurses transparently, like
/// an `Inline` container, so its text children take the same
/// path through the recursive call's loop head.
fn quirks_run_walk(
    forest: &Forest,
    idx: usize,
    out: &[f32],
    best: &mut f32,
    run: &mut f32,
    fonts: &dyn FontMetrics,
) {
    for &c in forest.nodes[idx].children.iter() {
        if layout_position::is_out_of_flow(forest, c) {
            continue;
        }
        if is_ignorable_whitespace(forest, c) {
            if quirks_ws_is_collapsible(forest, c) {
                continue;
            }
            if forest.nodes[c].box_type == BoxType::AnonymousBlock {
                quirks_run_walk(forest, c, out, best, run, fonts);
            } else {
                let n = &forest.nodes[c];
                let text = n.text.as_deref().unwrap_or("");
                let fs = n.data.font_size;
                let family = family_id_for(forest, c, fonts);
                for (seg_i, seg) in text.split('\n').enumerate() {
                    if seg_i > 0 {
                        *best = best.max(*run);
                        *run = 0.0;
                    }
                    *run += fonts.advance(seg, fs, family);
                }
                *run += intrinsic_h_extras(forest, c);
            }
            continue;
        }
        let n = &forest.nodes[c];
        if is_float(forest, c) {
            *best = best.max(*run);
            *run = 0.0;
            *best = best.max(out[c] + intrinsic_h_extras(forest, c));
        } else if n.box_type == BoxType::InlineBlock && n.is_img && n.intrinsic.is_some() {
            *run += out[c] + intrinsic_h_extras(forest, c);
        } else if n.box_type == BoxType::Inline && !n.is_br {
            quirks_run_walk(forest, c, out, best, run, fonts);
        } else {
            *best = best.max(*run);
            *run = 0.0;
            *best = best.max(out[c] + intrinsic_h_extras(forest, c));
        }
    }
}

fn min_content_width_pooled_impl(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    scratch: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    clamp: bool,
) -> f32 {
    // Preorder collection, then reverse: children always precede their
    // parent, so one reversed pass evaluates bottom-up.
    bufs.clear();
    bufs.stack.push(idx);
    while let Some(i) = bufs.stack.pop() {
        bufs.order.push(i);
        bufs.stack.extend(forest.nodes[i].children.iter().copied());
    }
    if scratch.len() < forest.nodes.len() {
        scratch.resize(forest.nodes.len(), 0.0);
    }
    let out = &mut *scratch;
    // One pooled table grid serves every table met in this walk; each
    // table checks it out and returns it, so nested tables never alias.
    let mut table_grid_pool = layout_table::TableGridPool::new();
    // Visit by index rather than via `iter()`: the flex arm below
    // takes `&mut bufs.flex`, which an iterator borrowing `bufs.order`
    // would conflict with.
    let mut k = bufs.order.len();
    while k > 0 {
        k -= 1;
        let i = bufs.order[k];
        // The table arm needs `&mut` (column widths resolve metrics),
        // so borrow per arm instead of holding one shared `&` across
        // the match.
        let v = match forest.nodes[i].box_type {
            BoxType::Text => {
                let n = &forest.nodes[i];
                let text = n.text.as_deref().unwrap_or("");
                let fs = n.data.font_size;
                let family = family_id_for(forest, i, fonts);
                let run = |s: &str| fonts.advance(s, fs, family);
                match n.data.white_space {
                    WhiteSpace::NoWrap => run(text),
                    WhiteSpace::Pre => text.split('\n').map(run).fold(0.0, f32::max),
                    _ => text.split([' ', '\n', '\t']).map(run).fold(0.0, f32::max),
                }
            }
            BoxType::Inline => {
                let n = &forest.nodes[i];
                // `<br>` forces a line break, so the min-content width is
                // the widest break-delimited segment, not the sum (flat
                // case; a `<br>` nested deeper than one inline level still
                // sums at the outer level — a known limitation).
                let mut best = 0.0f32;
                let mut seg = 0.0f32;
                for &c in n.children.iter() {
                    if layout_position::is_out_of_flow(forest, c) {
                        continue;
                    }
                    if forest.nodes[c].is_br {
                        best = best.max(seg);
                        seg = 0.0;
                    } else {
                        seg += out[c] + intrinsic_h_extras(forest, c);
                    }
                }
                best.max(seg)
            }
            // Tables size by column, not by summing every cell (see
            // [`table_min_content_width`]). `inline-table` is an
            // `InlineBlock` box whose inner layout is the table
            // algorithm, so it sizes the same way — the guard mirrors
            // `preferred_width_mcw`.
            _ if forest.nodes[i].data.display == Display::InlineTable => {
                table_min_content_width(forest, i, fonts, &mut table_grid_pool)
            }
            BoxType::Table => table_min_content_width(forest, i, fonts, &mut table_grid_pool),
            // Grids size by track, not by the widest child: the
            // min-content contribution is the sum of the column tracks'
            // minima plus gaps (css-grid-1 §12.5, via
            // [`layout_grid::grid_min_width`]). `inline-grid` is an
            // `InlineBlock` box with the grid inner layout, so it sizes
            // the same way — the guard mirrors `preferred_width_mcw`.
            _ if forest.nodes[i].data.display == Display::InlineGrid => {
                layout_grid::grid_min_width(forest, i, fonts, &mut *out)
            }
            BoxType::Grid => layout_grid::grid_min_width(forest, i, fonts, &mut *out),
            // Flex containers size by item, in min-content terms (see
            // [`flex_intrinsic_width`]): a row container's min-content
            // is the items' min-content sum plus gaps, a column
            // container's the widest item minimum. Definite widths
            // keep the precedence of the arms below (atomic boxes and
            // blocks contribute exactly, cells floor).
            _ if flex_kind(forest, i).is_some() => {
                // The clamp adjustment needs its own scratch: `out`
                // is the live min-content buffer (the closure reads
                // it), and this dispatch has no pref-width cache, so
                // the `WalkBufs`-resident [`FlexScratch`] serves the
                // clamp's content-keyword min/max measurements. The
                // cache is invalidated per flex node by generation
                // bump — semantically a fresh cache each time, without
                // the O(node-count) alloc+zero.
                let flex = bufs
                    .flex
                    .get_or_insert_with(|| Box::new(FlexScratch::new(forest.nodes.len())));
                flex.begin_node();
                let intrinsic = flex_intrinsic_width(
                    forest,
                    i,
                    IntrinsicMode::Min,
                    fonts,
                    &mut flex.pw,
                    &mut flex.aux,
                    &mut flex.bufs,
                    &mut flex.inner,
                    |f, c| out[c] + intrinsic_h_extras(f, c),
                );
                // A flex container is never a table cell, so there is no
                // cell guard here (unlike the generic arm below):
                // `flex_kind` requires `display: flex` /
                // `inline-flex`, which `display_of` maps to the
                // `Block` / `InlineBlock` box types — a `TableCell`
                // box always carries `display: table-cell`, and
                // anonymous cells have no style at all, so
                // `flex_kind` is `None` for both. Pinned by
                // `flex_display_never_produces_table_cell`.
                debug_assert_ne!(forest.nodes[i].box_type, BoxType::TableCell);
                definite_specified_content_width(forest, i)
                    .or_else(|| intrinsic_content_width(forest, i))
                    .unwrap_or(intrinsic)
            }
            // Multicol containers size by column (see the
            // `preferred_width_mcw` arm): the min-content size is the
            // single-column content measure — the largest min-content
            // contribution of the in-flow children — transformed by the
            // column-width / column-count / column-gap rules (csswg
            // css-multicol#4952). Out-of-flow children are skipped.
            // w78fix2: this arm sits ABOVE the display-keyed
            // inline-block arm below, mirroring
            // `preferred_width_mcw` (multicol, then inline-block).
            // Before 1214788 the min-content inline-block arm was
            // box-type-only, so a promoted multicol inline-block (a
            // `Block` box carrying `display: inline-block`) fell
            // through to this arm; the display-keyed arm captured
            // it instead and measured kids-max instead of the
            // column-transformed size, while max-content still
            // measured the column transform. Since w79 the
            // unpromoted shape takes this arm too:
            // `multicol_intrinsic_params` accepts an `InlineBlock`
            // box carrying `display: inline-block`, in lockstep
            // with `layout_block`'s multicol gate.
            _ if let Some(cols) = multicol_intrinsic_params(forest, i) => {
                let content = forest.nodes[i]
                    .children
                    .iter()
                    .filter(|&&c| !layout_position::is_out_of_flow(forest, c))
                    .map(|&c| out[c] + intrinsic_h_extras(forest, c))
                    .fold(0.0f32, f32::max);
                let content_based = multicol_min_content_width(&cols, content);
                // Definite widths keep the block precedence of the
                // fallthrough arm (css-sizing-3 §5.2).
                definite_specified_content_width(forest, i)
                    .or_else(|| intrinsic_content_width(forest, i))
                    .or_else(|| aspect_ratio_intrinsic_width(forest, i))
                    .unwrap_or(content_based)
            }
            // Atomic inline-level boxes (inline-block, inline-flex, and
            // replaced inline elements; `inline-table` and `inline-grid`
            // are caught by the guards above and keep their column/track
            // sizing): a definite specified width contributes exactly its
            // content width to the min-content size (css-sizing-3 §5.2) —
            // content overflowing that width does not grow the
            // contribution. When the width is `auto` (or cyclic), the
            // contribution is the max of the children's intrinsic
            // contributions — except for replaced boxes, which
            // contribute their intrinsic width (css-sizing-3 §5.2).
            // w78fix H1/L1: keyed off `display` as well as `box_type`,
            // like the max-content arm, so a promoted inline-block
            // takes this arm instead of the generic `Block` arm below.
            // The two formulas already agreed (definite, intrinsic,
            // aspect, kids), so this changes no value — it only keeps
            // the dispatch from drifting apart again. The multicol
            // arm above takes precedence for promoted multicol
            // boxes (see its comment); this arm covers the rest.
            _ if forest.nodes[i].box_type == BoxType::InlineBlock
                || forest.nodes[i].data.display == Display::InlineBlock =>
            {
                let kids = forest.nodes[i]
                    .children
                    .iter()
                    .filter(|&&c| !layout_position::is_out_of_flow(forest, c))
                    .map(|&c| out[c] + intrinsic_h_extras(forest, c))
                    .fold(0.0, f32::max);
                definite_specified_content_width(forest, i)
                    .or_else(|| intrinsic_content_width(forest, i))
                    .or_else(|| aspect_ratio_intrinsic_width(forest, i))
                    .unwrap_or(kids)
            }
            // Block-level boxes with a definite specified width contribute
            // exactly that width to the min-content size (css-sizing-3
            // §5.2), symmetric with the max-content path's
            // `definite_width_floor` — content overflowing it does not
            // grow the contribution. When the width is `auto` (or
            // cyclic), the contribution is the max of the children's
            // intrinsic contributions, as before — except for replaced
            // boxes, which contribute their intrinsic width
            // (css-sizing-3 §5.2). `width: min-content`
            // never resolves, so F7 (the box sizes to its min-content)
            // is preserved by the kids fallback.
            _ => {
                // WHATWG Quirks §3.8: the min-content of a table
                // cell's inline formatting context, in a full-quirks
                // document whose cell has an `auto` width, measures
                // with no soft wrap opportunities between consecutive
                // inline replaced elements (see
                // `quirks_cell_inline_min_content`). The context is
                // either the cell itself (its inline children stay
                // direct children when no block-level sibling forces
                // an anonymous wrapper) or the anonymous block
                // wrapping the cell's inline run. Limited-quirks and
                // no-quirks documents and cells with a definite width
                // keep the normal max fold. Vertical-writing-mode
                // cells measure through the vertical passes instead
                // ([`vertical_cell_min_content`], which carries its
                // own §3.8 run-summing arm) — the gate here is
                // horizontal-only because the vertical axis is
                // measured there, not because the quirk is absent.
                let quirks_cell_run = forest.quirks_mode == QuirksMode::Quirks
                    && forest.nodes[i].data.writing_mode == WritingMode::HorizontalTb
                    && (forest.nodes[i].box_type == BoxType::TableCell
                        && matches!(forest.nodes[i].data.width, SpecLen::Auto)
                        || forest.nodes[i].box_type == BoxType::AnonymousBlock
                            && forest.nodes[i].parent.is_some_and(|p| {
                                forest.nodes[p].box_type == BoxType::TableCell
                                    && matches!(forest.nodes[p].data.width, SpecLen::Auto)
                            }));
                let kids = if quirks_cell_run {
                    quirks_cell_inline_min_content(forest, i, out, fonts)
                } else {
                    forest.nodes[i]
                        .children
                        .iter()
                        .filter(|&&c| !layout_position::is_out_of_flow(forest, c))
                        .map(|&c| out[c] + intrinsic_h_extras(forest, c))
                        .fold(0.0, f32::max)
                };
                match forest.nodes[i].box_type {
                    BoxType::Block | BoxType::AnonymousBlock => {
                        definite_specified_content_width(forest, i)
                            .or_else(|| intrinsic_content_width(forest, i))
                            .or_else(|| aspect_ratio_intrinsic_width(forest, i))
                            .unwrap_or(kids)
                    }
                    // A table cell's definite width floors its
                    // min-content contribution, symmetric with the
                    // max-content side's `definite_width_floor`
                    // (`max(MCW, W)`): content wider than the definite
                    // width still wins, and a definite width wider
                    // than the content raises the minimum — but only
                    // for a cell measured outside a table context.
                    // Inside a real table, CSS Tables 3 §3.8.3 rules
                    // instead: a specified `width` floors the column
                    // *maximum*, never the minimum, and the table
                    // passes (`auto_column_minima` and friends, which
                    // call this function on cells directly) apply
                    // that rule themselves on the unfloored value —
                    // flooring here would break them (see
                    // `specified_cell_width_does_not_floor_column_minimum`).
                    // A standalone `display: table-cell` box (no table
                    // ancestor) has no table algorithm to defer to, so
                    // direct cell-subtree measurement gets the floor.
                    BoxType::TableCell if !cell_in_table_context(forest, i) => {
                        match definite_specified_content_width(forest, i) {
                            Some(d) => d.max(kids),
                            None => intrinsic_content_width(forest, i)
                                .or_else(|| aspect_ratio_intrinsic_width(forest, i))
                                .unwrap_or(kids),
                        }
                    }
                    // A cell inside a real table reports its unfloored
                    // min-content to the table algorithm; a replaced
                    // cell's is its intrinsic width (css-sizing-3 §5.2).
                    _ => intrinsic_content_width(forest, i).unwrap_or(kids),
                }
            }
        };
        // As in [`preferred_width_mcw`]: a scroll container's intrinsic
        // min-content size includes the gutter reservation (see
        // [`intrinsic_scrollbar_reservation`]), folded in before the
        // `min-width` / `max-width` clamp so the constraints bound the
        // whole intrinsic size.
        let v = v + intrinsic_scrollbar_reservation(forest, i);
        out[i] = if clamp {
            clamp_content_width_by_minmax(forest, i, v)
        } else {
            v
        };
    }
    out[idx]
}

/// Wrap each maximal run of non-block-level children of the flex
/// container `idx` in an anonymous block box (flexbox §4), rewriting the
/// container's child list. Each contiguous run of text and inline boxes
/// becomes one anonymous flex item instead of N separate items.
///
/// Exception (w77g, extended w78, css-flexbox-1 §4 blockification):
/// an in-flow `display: inline-block | inline-flex | inline-grid |
/// inline-table` child is NOT wrapped — it blockifies to a real
/// `Block` / `Block` / `Grid` / `Table` box carrying its own style,
/// so its `flex-basis`, `flex-grow`/`flex-shrink`, and margins reach
/// the item instead of being lost on a style-less wrapper (the
/// wrapper's basis is `auto`, its grow 0). Only the box type is
/// mutated: the display-keyed inner dispatch still sees the original
/// display, so a promoted inline-flex (`flex_kind() == Some(false)`)
/// still lays out as flex and a promoted inline-grid still as grid.
/// The promotion is idempotent: a promoted node is block-level, so a
/// second pass (e.g. the inline-flex shrink-to-fit estimate running
/// before the real pass) takes the early return or the block-level
/// arm and leaves it alone. Other inline-level children (text runs,
/// plain inline boxes) keep the wrapper path, which they rely on:
/// text has no style of its own to carry, and the wrapper is the item
/// those shapes lay out through. Out-of-flow inline-* boxes are not
/// flex items at all, so they are left as `InlineBlock` boxes for the
/// absolute-positioning path.
fn wrap_flex_text_runs(forest: &mut Forest, idx: usize) {
    // Blockification, joint note (w77fix finding 12, extended w78):
    // this is the SECOND of two promotion triggers for inline-*
    // flex items — the first is the `needs_wrap` branch in
    // `boxes.rs` pass 3, which fires when the flex container has a
    // block-level child. This one fires when it does not (a text run
    // or a lone inline-* item means not all children are
    // block-level). Both promote the item to a direct box carrying
    // its own style, so `flex-basis`/margins survive; the w77fix
    // joint test and the w78 joint tests pin both paths to identical
    // item geometry per value.
    //
    // All four inline outer displays promote (css-flexbox-1 §4):
    // inline-block and inline-flex to `Block`, inline-grid to
    // `Grid`, inline-table to `Table`. The w77g/w77fix comments
    // that only inline-table promoted are superseded: promoting the
    // others was deferred then because the `BoxType` changes' contact
    // with display-keyed inner layout in `layout_block` was
    // unverified, and the inline-block wrapper shape was pinned by
    // `w77g_flex_inline_block_item_still_wrapped` (inverted in w78).
    // Verified now: `layout_block` dispatches flex/grid/table off
    // `display` (and `flex_kind` off the raw keyword), not off the
    // box type, so mutating only the box type keeps inner layout
    // identical while the item itself carries its flex properties.
    // Index iteration instead of `children.clone()`: nothing in the loop
    // below mutates `nodes[idx].children` itself (the anonymous wrappers
    // are appended as fresh nodes and wired in by the final assignment),
    // so reading each child by index is the same snapshot without the
    // O(kids) allocation.
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
        // Out-of-flow children are not flex items: keep them out of the
        // anonymous runs (they become their own roots, skipped later).
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
        } else if forest.nodes[c].box_type == BoxType::InlineBlock
            && matches!(
                forest.nodes[c].data.display,
                Display::InlineBlock
                    | Display::InlineFlex
                    | Display::InlineGrid
                    | Display::InlineTable
            )
        {
            // w78 blockification (see the fn docs): flush any pending
            // run, then promote this node to an item of its own.
            // Its style is untouched, so `flex-basis` etc. survive;
            // the inner-layout dispatch arms key off `display` as
            // well as `box_type`, so the promoted node still sizes
            // and lays out as its original inner shape (flex, grid,
            // or table). Inline-flex has no `Flex` box type, so it
            // blockifies to `Block` like inline-block.
            if !run.is_empty() {
                roots.push(forest.make_anonymous(
                    &parent_data,
                    std::mem::take(&mut run),
                    idx,
                    true,
                ));
            }
            forest.nodes[c].box_type = match forest.nodes[c].data.display {
                Display::InlineGrid => BoxType::Grid,
                Display::InlineTable => BoxType::Table,
                _ => BoxType::Block,
            };
            roots.push(c);
        } else {
            run.push(c);
        }
    }
    if !run.is_empty() {
        roots.push(forest.make_anonymous(&parent_data, run, idx, true));
    }
    // Route the tree through the wrappers: paint and later passes must
    // see the same structure the item list was built from.
    forest.nodes[idx].children = roots;
}

/// Build the item list for a flex container. `content_w` is the
/// container's content-box width (the main size for rows, the cross size
/// for columns); `hspec` its definite content height, if any.
#[allow(clippy::too_many_arguments)]
fn flex_build_items(
    forest: &mut Forest,
    idx: usize,
    content_w: f32,
    hspec: Option<f32>,
    horizontal: bool,
    align_items: FlexAlign,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    // Reusable scratch (see `LayoutScratch`): `mcw` serves every
    // item's `min_content_width` instead of allocating per item.
    scratch: &mut LayoutScratch,
    // The pooled item buffer (`LayoutScratch::flex_items`, taken by the
    // caller): cleared and refilled here instead of allocating a fresh
    // `Vec<FlexItem>` per container per pass.
    items: &mut Vec<FlexItem>,
) {
    // Flexbox §4: a contiguous run of in-flow text and inline-level
    // children forms a single anonymous block flex item. Wrap each such
    // run before building items so the run lays out as one item instead
    // of N. Wrappers are block-level, so a second call (the inline-flex
    // shrink-to-fit estimate runs before the real pass) finds no raw
    // text runs and re-wraps nothing: the transform is idempotent.
    wrap_flex_text_runs(forest, idx);
    // Pooled buffer: clear instead of allocating; capacity survives
    // across containers.
    items.clear();
    // Index iteration instead of `children.clone()`: the loop never
    // mutates `nodes[idx].children` (out-of-flow children are only
    // recorded; wrappers were already built above), so this reads the
    // same snapshot without the O(kids) allocation.
    let nkids = forest.nodes[idx].children.len();
    for i in 0..nkids {
        let c = forest.nodes[idx].children[i];
        if layout_position::is_out_of_flow(forest, c) {
            // Out-of-flow children are not flex items (flexbox §4). Record
            // the container's content-box origin; `abs_position` derives the
            // static position from it as if the child were the sole flex
            // item (flexbox §4.2): the main-start edge per
            // `flex-direction` (with `row` honoring the container's
            // `direction`), `align-self` (`auto` → `align-items`) on the
            // cross axis, and the child's margins. `justify-content` does
            // not apply to abspos children. `record_abs_item` is
            // idempotent: the inline-flex estimate pass calls this builder
            // twice.
            let (sx, sy) = {
                let n = &forest.nodes[idx];
                (
                    n.rect.x + n.metrics.bl + n.metrics.pl,
                    n.rect.y + n.metrics.bt + n.metrics.pt,
                )
            };
            layout_position::record_abs_item(forest, c, sx, sy);
            continue;
        }
        let st = forest.nodes[c].style.as_deref();
        let grow = flex_factor(st, props::FLEX_GROW, 0.0);
        let shrink = flex_factor(st, props::FLEX_SHRINK, 1.0);
        // CSS Box Alignment 3 §4.1: `self-start`/`self-end` on
        // `align-self` resolve against the *item's* direction, not the
        // container's. On a horizontal cross axis (column flex) an item
        // whose `direction` differs from the container's flips the
        // mapping; on a vertical cross axis (row flex) `direction` is
        // irrelevant and the keywords behave as `start`/`end`. The
        // container-relative `Start`/`End` below is then physicalized
        // against the container's cross direction as before.
        let self_rtl = !horizontal && (forest.nodes[c].data.rtl != is_rtl(forest, idx));
        let align = flex_align_of(
            st.and_then(|s| s.get_str_by_index(props::ALIGN_SELF)),
            self_rtl,
        )
        .unwrap_or(align_items);
        // `order`: integer, initial 0 (flexbox §5.4).
        let order = match st.and_then(|s| s.get_by_index(props::ORDER)) {
            Some(CssValue::Number(n)) => *n as i32,
            _ => 0,
        };
        // Forced flex line breaks (flexbox §9.3), honored when lines are
        // collected in `layout_flex_wrap::collect_lines`.
        let force_break_before = forced_break_of(st, props::BREAK_BEFORE);
        let force_break_after = forced_break_of(st, props::BREAK_AFTER);
        let (fs, rfs, vpw, vph) = {
            let d = &forest.nodes[c].data;
            (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h)
        };
        // Margin percentages always resolve against the containing-block
        // width.
        let mo = |s: SpecMargin| margin_opt(s, content_w, fs, rfs, vpw, vph).unwrap_or(0.0);
        let (marg_ms, marg_me, marg_cs, marg_ce) = {
            let d = &forest.nodes[c].data;
            if horizontal {
                (
                    mo(d.margin[3]),
                    mo(d.margin[1]),
                    mo(d.margin[0]),
                    mo(d.margin[2]),
                )
            } else {
                (
                    mo(d.margin[0]),
                    mo(d.margin[2]),
                    mo(d.margin[3]),
                    mo(d.margin[1]),
                )
            }
        };
        // Column + `direction: rtl`: the cross axis is horizontal and
        // cross-start follows inline direction (flexbox §5.1), so store
        // the cross margins in cross-start/cross-end order.
        let cross_rtl = !horizontal && is_rtl(forest, idx);
        let (marg_cs, marg_ce) = if cross_rtl {
            (marg_ce, marg_cs)
        } else {
            (marg_cs, marg_ce)
        };
        // Whether the main-axis margins were `auto`: they absorb positive
        // free space in `flex_justify`.
        let (auto_ms, auto_me) = {
            let d = &forest.nodes[c].data;
            if horizontal {
                (
                    matches!(d.margin[3], SpecMargin::Auto),
                    matches!(d.margin[1], SpecMargin::Auto),
                )
            } else {
                (
                    matches!(d.margin[0], SpecMargin::Auto),
                    matches!(d.margin[2], SpecMargin::Auto),
                )
            }
        };
        // Whether the cross-start/cross-end margins were `auto`: they
        // absorb positive free space in `flex_row_finish`.
        let (auto_cs, auto_ce) = {
            let d = &forest.nodes[c].data;
            if horizontal {
                (
                    matches!(d.margin[0], SpecMargin::Auto),
                    matches!(d.margin[2], SpecMargin::Auto),
                )
            } else {
                (
                    matches!(d.margin[3], SpecMargin::Auto),
                    matches!(d.margin[1], SpecMargin::Auto),
                )
            }
        };
        let (auto_cs, auto_ce) = if cross_rtl {
            (auto_ce, auto_cs)
        } else {
            (auto_cs, auto_ce)
        };
        let basis = spec_len_of(st.and_then(|s| s.get_by_index(props::FLEX_BASIS)));
        // `flex-basis: content` (flexbox §9.2): the flex base size is the
        // item's max-content size, not the main-size property.
        // `spec_len_of` has no `SpecLen` form for the keyword (it falls
        // through to `Auto`), so the specified value is checked directly.
        // Read here, before the `&mut Forest` uses below.
        let basis_is_content = matches!(
            st.and_then(|s| s.get_by_index(props::FLEX_BASIS)),
            Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("content")
        );
        let main_pct_base = if horizontal { Some(content_w) } else { hspec };
        // Hypothetical main size from `flex-basis`, falling back to the
        // main-size property and then to content. (Defined below, after
        // `main_bp` is known, so a definite basis can be converted out
        // of the item's box-sizing terms.)

        if horizontal {
            // Provisional metrics for border/padding/margins. A specified
            // width resolves now; an auto or content-based width gets a
            // zero placeholder — the real base comes from `flex-basis`
            // or the measured content width.
            let wo = match forest.nodes[c].data.width {
                SpecLen::Auto | SpecLen::FitContent | SpecLen::MinContent | SpecLen::MaxContent => {
                    Some(0.0)
                }
                _ => None,
            };
            resolve_metrics(forest, c, 0.0, content_w, false, wo, false);
            let (main_bp, cross_bp) = {
                let m = &forest.nodes[c].metrics;
                (m.bl + m.pl + m.pr + m.br, m.bt + m.pt + m.pb + m.bb)
            };
            // css-flexbox-1 §9.2: a definite `flex-basis` resolves in the
            // item's box-sizing terms, so under `border-box` it covers the
            // item's own border+padding. Strip them to get the engine's
            // content-box base — otherwise the extras count twice, once in
            // the basis and once in `main_bp`'s outer size. Read before the
            // `&mut Forest` uses below. The `content` fallbacks in
            // `basis_or` are already content-box sizes and are not stripped.
            let basis_strip = match forest.nodes[c].data.box_sizing {
                BoxSizing::BorderBox => main_bp,
                BoxSizing::ContentBox => 0.0,
            };
            // `keyword` is the `flex-basis` content-keyword size
            // measured at the `b` site (it needs `&mut Forest` and the
            // pooled scratch, so it cannot be computed inside this
            // closure). The keyword arms return it unstripped: like
            // the `Auto`/`content` fallbacks it is already a
            // content-box size, whereas a definite length basis
            // resolves in box-sizing terms and is stripped. The
            // `%`/`calc()` cyclic gates below are unchanged.
            let basis_or = |content: f32, keyword: Option<f32>| match basis {
                SpecLen::Auto => content,
                SpecLen::MinContent | SpecLen::MaxContent | SpecLen::FitContent => {
                    keyword.unwrap_or(content)
                }
                SpecLen::Pct(_) if main_pct_base.is_none() => content,
                // CSS Values 4 §8.1.4: a `calc()` that syntactically
                // contained a `%` leaf is cyclic against an indefinite
                // main-size base and behaves as `content`, like a bare
                // percentage. (Unreachable for row items — `main_pct_base`
                // is always `Some` here — but mirrors the `Pct` arm so the
                // gate cannot be lost if that changes.)
                SpecLen::Math { had_pct, .. } if had_pct && main_pct_base.is_none() => content,
                spec => (spec
                    .resolve(fs, rfs, vpw, vph, main_pct_base.unwrap_or(0.0))
                    .map(|v| v - basis_strip)
                    .unwrap_or(content))
                .max(0.0),
            };
            // F5: the available width for content-based measurement
            // excludes the item's own used side margins and
            // border/padding, exactly like the block arm — not the raw
            // container content width.
            let item_avail = {
                let m = &forest.nodes[c].metrics;
                (content_w - m.ml - m.mr - (m.bl + m.pl + m.pr + m.br)).max(0.0)
            };
            let content = match forest.nodes[c].data.width {
                SpecLen::Auto => {
                    if let Some(d) = forest.nodes[c].intrinsic {
                        d.width_or_default().max(0.0)
                    } else {
                        // One shared `mcw` and one pooled walk serve
                        // every item (see `LayoutScratch`).
                        preferred_width_mcw(
                            forest,
                            c,
                            fonts,
                            pw_cache,
                            &mut scratch.mcw,
                            &mut scratch.int_bufs,
                            &mut scratch.int_inner,
                        )
                    }
                }
                // css-sizing-3: content-based widths measure content for
                // the hypothetical main size.
                spec @ (SpecLen::FitContent | SpecLen::MinContent | SpecLen::MaxContent) => {
                    content_based_width(
                        forest,
                        c,
                        spec,
                        fonts,
                        pw_cache,
                        &mut scratch.mcw,
                        &mut scratch.int_bufs,
                        &mut scratch.int_inner,
                        item_avail,
                    )
                }
                _ => forest.nodes[c].metrics.content_w,
            };
            // Content-keyword bases measure the keyword, not the
            // main-size property: `SpecLen::resolve` returns `None`
            // for them, so the old `basis_or` fallback silently used
            // `content` (derived from `width`) — min-content and
            // fit-content wrongly yielded max-content, and
            // `width:100px` + `flex-basis:max-content` yielded 100.
            // `flex_basis_content_width` overrides a definite width
            // (see its docs); `item_avail` clamps fit-content, exactly
            // like the width-keyword arm above.
            let keyword_base = match basis {
                SpecLen::MinContent | SpecLen::MaxContent | SpecLen::FitContent => {
                    Some(flex_basis_content_width(
                        forest,
                        c,
                        basis,
                        fonts,
                        pw_cache,
                        &mut scratch.mcw,
                        &mut scratch.int_bufs,
                        &mut scratch.int_inner,
                        item_avail,
                    ))
                }
                _ => None,
            };
            let b = if basis_is_content {
                // `flex-basis: content`: measure the max-content size
                // directly (see above); `basis_or` would fall back to
                // the main-size property instead.
                preferred_width_mcw(
                    forest,
                    c,
                    fonts,
                    pw_cache,
                    &mut scratch.mcw,
                    &mut scratch.int_bufs,
                    &mut scratch.int_inner,
                )
            } else if matches!(basis, SpecLen::Stretch) {
                // w76fix finding 4: `flex-basis: stretch` fills the
                // available space (css-sizing-3 §5: `stretch` fills
                // the containing block). `SpecLen::resolve` returns
                // `None` for it, which `basis_or` would silently
                // degrade to the content size; the content-box
                // available width is the fill size instead (margins
                // and border/padding already excluded from
                // `item_avail`, like the block arm's fill).
                item_avail
            } else {
                basis_or(content, keyword_base)
            };
            // An infinite flex base from an automatic-layout table
            // (its `width:100%` column drives the preferred width to
            // infinity) must not reach `flex_lengths`, which can only
            // saturate it to `MAX_USED_SIZE`: the table's base is its
            // finite column-maximum sum instead (WPT table-minmax:
            // flex `1 1 auto` grows 160 to 355, `0 0 auto` stays 160).
            // The substitution fires whenever the measured base is
            // infinite, regardless of keyword (Auto/content/
            // MaxContent/FitContent): a MinContent base is a regular
            // min-content size, so the guard is naturally inert for
            // it. Fixed-layout tables keep infinity (WPT
            // table-fixed-minmax saturation) — `table_auto_grid_max_width`
            // returns `None` for them, as for vertical tables — as do
            // definite bases, which are never infinite.
            let b = if b.is_infinite()
                && matches!(
                    forest.nodes[c].data.display,
                    Display::Table | Display::InlineTable
                )
                && let Some(g) = layout_table::table_auto_grid_max_width(
                    forest,
                    c,
                    fonts,
                    pw_cache,
                    &mut scratch.mcw,
                    &mut scratch.int_bufs,
                    &mut scratch.int_inner,
                ) {
                g
            } else {
                b
            };
            // Same substitution through an interposed block wrapper:
            // a wrapper's generic preferred width propagates the
            // table's infinity unchanged, so without this the item
            // saturates to `MAX_USED_SIZE` instead of the grid sum
            // plus the wrappers' extras. Runs only while the base is
            // still infinite, so it cannot rewrite a finite base.
            let b = if b.is_infinite()
                && let Some(g) = flex_wrapper_table_grid_max(
                    forest,
                    c,
                    fonts,
                    pw_cache,
                    &mut scratch.mcw,
                    &mut scratch.int_bufs,
                    &mut scratch.int_inner,
                ) {
                g
            } else {
                b
            };
            // Automatic minimum: the item's min-content *size*
            // (flexbox §4.5), unless a min-width is explicitly set. The
            // size is unclamped: `max-width` does not bound the
            // automatic minimum (min wins when over-constrained,
            // css-sizing-3 §4). For a content-keyword (or stretch)
            // basis the dispatched min-content is additionally floored
            // at a definite width (w76fix finding 2), so the minimum
            // comes from the keyword measurement instead — a
            // `width:100px` + `flex-basis:min-content` item over a
            // 60px longest word floors at 60, not 100.
            let auto_min = if matches!(forest.nodes[c].data.min_width, SpecLen::Auto) {
                match basis {
                    SpecLen::MinContent
                    | SpecLen::MaxContent
                    | SpecLen::FitContent
                    | SpecLen::Stretch => keyword_min_content_unclamped(
                        forest,
                        c,
                        fonts,
                        &mut scratch.mcw,
                        &mut scratch.int_bufs,
                    ),
                    _ => min_content_width_pooled_raw(
                        forest,
                        c,
                        fonts,
                        &mut scratch.mcw,
                        &mut scratch.int_bufs,
                    ),
                }
            } else {
                0.0
            };
            let (cross_auto, min_spec, max_spec) = {
                let d = &forest.nodes[c].data;
                (matches!(d.width, SpecLen::Auto), d.min_width, d.max_width)
            };
            let (base, min_main, max_main) = clamp_flex_main(
                forest,
                c,
                b,
                min_spec,
                max_spec,
                fs,
                rfs,
                vpw,
                vph,
                Some(content_w),
                true,
                fonts,
                pw_cache,
                &mut scratch.mcw,
                &mut scratch.int_bufs,
                &mut scratch.int_inner,
                item_avail,
            );
            items.push(FlexItem {
                idx: c,
                order,
                grow,
                shrink,
                base,
                target: base,
                min_main,
                max_main,
                m_ms: marg_ms,
                m_me: marg_me,
                m_cs: marg_cs,
                m_ce: marg_ce,
                auto_ms,
                auto_me,
                auto_cs,
                auto_ce,
                auto_min,
                main_bp,
                cross_bp,
                align,
                cross_auto,
                cross: 0.0,
                natural_cross: 0.0,
                baseline: 0.0,
                main_pos: 0.0,
                cross_pos: 0.0,
                needs_measure: false,
                measure_cross: None,
                measure_height_min: false,
                stretched: false,
                force_break_before,
                force_break_after,
            });
        } else {
            // Column: the cross size (width) is final now.
            // Content-based widths measure content instead of filling.
            // The provisional pass resolves the item's own
            // margins/border/padding first, so the measurement's
            // available width excludes them — exactly like the block
            // arm (F5), not the raw container content width. css-sizing-3
            // §4 min/max keywords are measured here too (F4).
            let content_spec = matches!(
                forest.nodes[c].data.width,
                SpecLen::FitContent | SpecLen::MinContent | SpecLen::MaxContent
            );
            if content_spec {
                resolve_metrics(forest, c, 0.0, content_w, false, Some(0.0), false);
            }
            let cross_override = match forest.nodes[c].data.width {
                spec @ (SpecLen::FitContent | SpecLen::MinContent | SpecLen::MaxContent) => {
                    let (ml, mr, extras) = {
                        let m = &forest.nodes[c].metrics;
                        (m.ml, m.mr, m.bl + m.pl + m.pr + m.br)
                    };
                    let avail = (content_w - ml - mr - extras).max(0.0);
                    let (min_spec, max_spec) = {
                        let d = &forest.nodes[c].data;
                        (d.min_width, d.max_width)
                    };
                    let w = content_based_width(
                        forest,
                        c,
                        spec,
                        fonts,
                        pw_cache,
                        &mut scratch.mcw,
                        &mut scratch.int_bufs,
                        &mut scratch.int_inner,
                        avail,
                    );
                    let lo = measured_minmax_width(
                        forest,
                        c,
                        min_spec,
                        fonts,
                        pw_cache,
                        &mut scratch.mcw,
                        &mut scratch.int_bufs,
                        &mut scratch.int_inner,
                        avail,
                    );
                    let hi = measured_minmax_width(
                        forest,
                        c,
                        max_spec,
                        fonts,
                        pw_cache,
                        &mut scratch.mcw,
                        &mut scratch.int_bufs,
                        &mut scratch.int_inner,
                        avail,
                    );
                    Some(clamp_measured(w, lo, hi))
                }
                _ => None,
            };
            resolve_metrics(forest, c, 0.0, content_w, false, cross_override, false);
            let (main_bp, cross_bp, cross) = {
                let m = &forest.nodes[c].metrics;
                (
                    m.bt + m.pt + m.pb + m.bb,
                    m.bl + m.pl + m.pr + m.br,
                    m.content_w,
                )
            };
            let (b, needs_measure, measure_cross, measure_height_min) = match basis {
                SpecLen::Auto => match specified_height(forest, c, hspec) {
                    Some(h) => (h, false, None, false),
                    None => match forest.nodes[c].intrinsic {
                        Some(d) => (d.height_or_default().max(0.0), false, None, false),
                        None => (0.0, true, None, false),
                    },
                },
                SpecLen::Pct(_) if hspec.is_none() => (0.0, true, None, false),
                // CSS Values 4 §8.1.4: a `calc()` that syntactically
                // contained a `%` leaf is cyclic against an indefinite
                // main-size base and behaves as `content` (measured),
                // like a bare percentage.
                SpecLen::Math { had_pct, .. } if had_pct && hspec.is_none() => {
                    (0.0, true, None, false)
                }
                // w76fix finding 7: content-keyword bases measure the
                // item at the *keyword width* (not the cross size):
                // min-content width gives the wrapped (multi-line)
                // height, max-content width the unwrapped height,
                // fit-content the container-clamped width. The old
                // path measured all three at the cross size, so text
                // items gave the same height for every keyword.
                // w76fix finding 4: `Stretch` with a definite
                // container main size fills it (border-box strip, as
                // for definite bases); with an indefinite container
                // it measures content at the cross size.
                SpecLen::MinContent | SpecLen::MaxContent | SpecLen::FitContent => {
                    let (ml, mr) = {
                        let m = &forest.nodes[c].metrics;
                        (m.ml, m.mr)
                    };
                    let avail = (content_w - ml - mr - cross_bp).max(0.0);
                    let mc_w = min_content_width_pooled(
                        forest,
                        c,
                        fonts,
                        &mut scratch.mcw,
                        &mut scratch.int_bufs,
                    );
                    let pc_w = preferred_width_mcw(
                        forest,
                        c,
                        fonts,
                        pw_cache,
                        &mut scratch.mcw,
                        &mut scratch.int_bufs,
                        &mut scratch.int_inner,
                    );
                    let kw_w = match basis {
                        SpecLen::MinContent => mc_w,
                        SpecLen::MaxContent => pc_w,
                        _ => pc_w.min(mc_w.max(avail)),
                    };
                    // A `flex-basis` keyword overrides the main-size
                    // property (flexbox §9.2), so the measuring task
                    // must not clamp the natural height to the
                    // specified height. Only generic boxes get the
                    // zero minimum-height override; flex/grid/table/
                    // multicol items keep the old definite-height-wins
                    // measurement, and their own layout owns the
                    // override semantics. (w76fix2 L4: `%`-height
                    // descendants of such items resolve against an
                    // indefinite base — i.e. as `auto` — during
                    // measurement, since the zero override is a floor
                    // rather than a definite height.)
                    let height_min = specified_height(forest, c, hspec).is_some()
                        && flex_kind(forest, c).is_none()
                        && multicol_intrinsic_params(forest, c).is_none()
                        && !matches!(
                            forest.nodes[c].data.display,
                            Display::Table
                                | Display::InlineTable
                                | Display::Grid
                                | Display::InlineGrid
                        );
                    (0.0, true, Some(kw_w.max(0.0)), height_min)
                }
                SpecLen::Stretch => match hspec {
                    Some(h) => {
                        let strip = match forest.nodes[c].data.box_sizing {
                            BoxSizing::BorderBox => main_bp,
                            BoxSizing::ContentBox => 0.0,
                        };
                        ((h - strip).max(0.0), false, None, false)
                    }
                    None => (0.0, true, None, false),
                },
                // css-flexbox-1 §9.2: a definite `flex-basis` resolves in
                // the item's box-sizing terms, so under `border-box` it
                // covers the item's own border+padding — strip them to
                // get the engine's content-box base (mirrors the row
                // arm's `basis_strip`; the `Auto`/`content` fallbacks are
                // already content-box sizes).
                spec => {
                    let strip = match forest.nodes[c].data.box_sizing {
                        BoxSizing::BorderBox => main_bp,
                        BoxSizing::ContentBox => 0.0,
                    };
                    (
                        (spec
                            .resolve(fs, rfs, vpw, vph, hspec.unwrap_or(0.0))
                            .map(|v| v - strip)
                            .unwrap_or(0.0))
                        .max(0.0),
                        false,
                        None,
                        false,
                    )
                }
            };
            let (cross_auto, min_spec, max_spec) = {
                let d = &forest.nodes[c].data;
                (matches!(d.width, SpecLen::Auto), d.min_height, d.max_height)
            };
            let (base, min_main, max_main) = clamp_flex_main(
                forest,
                c,
                b,
                min_spec,
                max_spec,
                fs,
                rfs,
                vpw,
                vph,
                hspec,
                false,
                fonts,
                pw_cache,
                &mut scratch.mcw,
                &mut scratch.int_bufs,
                &mut scratch.int_inner,
                0.0,
            );
            items.push(FlexItem {
                idx: c,
                order,
                grow,
                shrink,
                base,
                target: base,
                min_main,
                max_main,
                m_ms: marg_ms,
                m_me: marg_me,
                m_cs: marg_cs,
                m_ce: marg_ce,
                auto_ms,
                auto_me,
                auto_cs,
                auto_ce,
                // Min-content height on the column axis is unmodeled.
                auto_min: 0.0,
                main_bp,
                cross_bp,
                align,
                cross_auto,
                cross,
                natural_cross: 0.0,
                baseline: 0.0,
                main_pos: 0.0,
                cross_pos: 0.0,
                needs_measure,
                measure_cross,
                measure_height_min,
                stretched: false,
                force_break_before,
                force_break_after,
            });
        }
    }
    // `order` (flexbox §5.4): ascending, stable for ties (document order).
    items.sort_by_key(|it| it.order);
}

/// Resolve flex-grow/flex-shrink for one line (CSS flexbox §9.7):
/// distribute free space over the unfrozen items, clamp each target to its
/// min/max, freeze violators by the sign of the total violation, and repeat
/// until everything is frozen. An indefinite main size means no free space,
/// so nothing flexes. Targets floor at the item's automatic minimum size
/// (`auto_min`: min-content when `min-width`/`min-height` is `auto`), and
/// the minimum wins over the maximum on conflict (CSS 2.1 §10.4).
/// Resolve flexible lengths per flexbox §9.7. The main-axis `gap` is a
/// constant stride between adjacent items: it is subtracted from the
/// free space up front and never flexed.
fn flex_lengths(
    items: &mut [FlexItem],
    main_size: Option<f32>,
    main_gap: f32,
    scratch: &mut LayoutScratch,
) {
    if items.is_empty() {
        return;
    }
    // Total inter-item gap on this line: `n - 1` strides.
    let gaps = main_gap * items.len().saturating_sub(1) as f32;
    for it in items.iter_mut() {
        it.target = it.base;
    }
    // Clamp a raw target to min/max. The floor includes the automatic
    // minimum size; when it exceeds the max, the max is ignored and the
    // floor wins (CSS 2.1 §10.4).
    let clamp_target = |it: &FlexItem, raw: f32| {
        let floor = it.min_main.max(it.auto_min).max(0.0);
        if floor > it.max_main {
            raw.max(floor)
        } else {
            raw.min(it.max_main).max(floor)
        }
    };
    // §9.7 steps 2-3: outer sizes use the flex base for unfrozen items
    // and the target main size for frozen ones.
    let outer_base = |it: &FlexItem| it.base + it.main_bp + it.m_ms + it.m_me;
    let outer_target = |it: &FlexItem| it.target + it.main_bp + it.m_ms + it.m_me;
    let total_base: f32 = items.iter().map(outer_base).sum();
    let initial_free = main_size.map(|m| m - total_base - gaps).unwrap_or(0.0);
    // Epsilon, not exact zero: `main_size - total_base` accumulates float
    // error, and a dust-sized remainder must not trigger distribution.
    if initial_free.abs() < 1e-6 {
        // No free space, but the automatic minimum still applies. The
        // saturation keeps an infinite flex base (a fixed-layout table
        // with a percentage width: infinite max-content intrinsic size)
        // out of the used target.
        for it in items.iter_mut() {
            it.target = saturate_used_size(clamp_target(it, it.base));
        }
        return;
    }
    let growing = initial_free > 0.0;
    // Raw flex factor, used for the §9.7 step 3 partial-fill rule.
    let raw_factor = |it: &FlexItem| if growing { it.grow } else { it.shrink };
    // Distribution factor: grow uses the raw factor, shrink scales it by
    // the flex base size (§9.7 step 3).
    let scaled_factor = |it: &FlexItem| {
        if growing {
            it.grow
        } else {
            it.shrink * it.base
        }
    };
    // Active-set bookkeeping: `active` holds the unfrozen item indices in
    // ascending order (only these are visited each pass, so frozen items
    // cost nothing after they freeze), `frozen_idx` the frozen ones.
    // `violations` is a scratch buffer reused across passes instead of
    // reallocated per pass. Each pass freezes at least one item (or ends
    // the loop), so this terminates after at most `items.len()` passes.
    // All four buffers live in `LayoutScratch` (cleared, never dropped)
    // so capacity survives across containers.
    let frozen = &mut scratch.frozen;
    frozen.clear();
    frozen.resize(items.len(), false);
    let active = &mut scratch.active;
    active.clear();
    active.extend(0..items.len());
    let frozen_idx = &mut scratch.frozen_idx;
    frozen_idx.clear();
    let violations = &mut scratch.violations;
    violations.clear();
    violations.resize(items.len(), 0.0);
    loop {
        // §9.7 step 3: frozen items contribute their outer target size,
        // unfrozen items their outer flex base size.
        let mut used = 0.0f32;
        let mut unfrozen_factor_sum = 0.0f32;
        let mut total_scaled = 0.0f32;
        for &i in frozen_idx.iter() {
            used += outer_target(&items[i]);
        }
        for &i in active.iter() {
            let it = &items[i];
            used += outer_base(it);
            // §9.7 step 3: when the sum of the *unfrozen* flex factors is
            // below 1, fill only `initial_free_space * factor_sum`,
            // leaving the rest unused. The sum is recomputed each pass
            // as items freeze.
            unfrozen_factor_sum += raw_factor(it);
            total_scaled += scaled_factor(it);
        }
        let mut remaining = main_size.map(|m| m - used - gaps).unwrap_or(0.0);
        if unfrozen_factor_sum < 1.0 {
            let fill = initial_free * unfrozen_factor_sum;
            if remaining.abs() > fill.abs() {
                remaining = fill;
            }
        }
        if total_scaled <= 0.0 {
            break;
        }
        // Non-finite free space, distribution factors, or distribution
        // quotient (e.g. an infinite flex base from a fixed-layout table
        // with a percentage width, whose max-content intrinsic size is
        // infinite per Blink): the distribute-and-freeze fixpoint cannot
        // converge — `remaining / total_scaled` is NaN, `clamp_target`
        // passes NaN through (`f32::min` ignores NaN), every violation
        // is NaN, and no item ever freezes. The quotient can also be
        // non-finite while both operands are finite: a denormal
        // `total_scaled` (e.g. `flex-basis: 1e-40px` makes the scaled
        // shrink-factor sum a denormal) makes `-50 / 1e-40` overflow to
        // -inf, and the zero-scaled item then computes
        // `base + -inf * 0.0 = NaN` with the same never-freeze hang.
        // Saturate every active item at its clamped base and stop: the
        // item keeps its intrinsic size, capped to a huge finite
        // constant so raw infinities never reach used rects. This
        // preserves the loop invariant — each pass freezes at least one
        // item or breaks.
        // Divide once, before multiplying, so huge factors stay finite.
        let unit = remaining / total_scaled;
        if !remaining.is_finite() || !total_scaled.is_finite() || !unit.is_finite() {
            for &i in active.iter() {
                let it = &mut items[i];
                it.target = saturate_used_size(clamp_target(it, it.base));
            }
            break;
        }
        // §9.7 step 4: distribute the space, clamping every unfrozen
        // target and recording each min/max violation.
        let mut total_violation = 0.0f32;
        for &i in active.iter() {
            let it = &mut items[i];
            let raw = it.base + unit * scaled_factor(it);
            let clamped = clamp_target(it, raw);
            let violation = clamped - raw;
            violations[i] = violation;
            total_violation += violation;
            it.target = clamped;
        }
        // §9.7 step 5: freeze by the sign of the total violation.
        if total_violation == 0.0 {
            break;
        }
        let freeze_min = total_violation > 0.0;
        for &i in active.iter() {
            let violation = violations[i];
            let is_violator = if freeze_min {
                violation > 0.0
            } else {
                violation < 0.0
            };
            if is_violator {
                frozen[i] = true;
                frozen_idx.push(i);
            }
        }
        // `retain` preserves `active`'s ascending order; the frozen flags
        // make the membership test O(1) per item.
        active.retain(|&i| !frozen[i]);
    }
    // Final safety clamp (idempotent for frozen targets), saturated so
    // an infinite flex base never leaks a raw infinity into a used
    // rect (see `MAX_USED_SIZE`).
    for it in items.iter_mut() {
        it.target = saturate_used_size(clamp_target(it, it.target));
    }
}

/// Main-axis positions from `justify-content`. `main_pos` is the
/// border-box offset from the container content-box main-start edge;
/// reversed directions mirror around the main-end edge.
///
/// The main-axis `gap` is a fixed stride between adjacent items: free
/// space is computed after subtracting it (§9.6), and the cursor
/// advances by one gap between items.
///
/// Auto margins on the main axis absorb positive free space before
/// `justify-content` runs (CSS flexbox §8.1), so with auto margins
/// present, justification behaves as `flex-start`. Cross-axis auto
/// margins likewise absorb positive free space before `align-self`
/// (flexbox §8.3); see `flex_row_finish`.
fn flex_justify(
    items: &mut [FlexItem],
    main_size: Option<f32>,
    main_gap: f32,
    justify: FlexJustify,
    reverse: bool,
) {
    if items.is_empty() {
        return;
    }
    let n = items.len();
    // Total inter-item gap on this line: `n - 1` strides.
    let gaps = main_gap * n.saturating_sub(1) as f32;
    let total: f32 = items
        .iter()
        .map(|it| it.target + it.main_bp + it.m_ms + it.m_me)
        .sum();
    let free = main_size.map(|m| m - total - gaps).unwrap_or(0.0);
    // Auto margins absorb positive free space before `justify-content`
    // runs: with any auto margin present, justification behaves as
    // flex-start. The absorbed space is kept out of `m_ms`/`m_me` (which
    // stay the specified margins: callers recover the border-box x via
    // `bx - m_ms`).
    let auto_n: usize = items
        .iter()
        .map(|it| usize::from(it.auto_ms) + usize::from(it.auto_me))
        .sum();
    let each = if free > 0.0 && auto_n > 0 {
        free / auto_n as f32
    } else {
        0.0
    };
    let (cursor_start, gap) = if each > 0.0 {
        (0.0, 0.0)
    } else {
        match justify {
            FlexJustify::End => (free, 0.0),
            FlexJustify::Center => (free / 2.0, 0.0),
            FlexJustify::Between if n > 1 && free > 0.0 => (0.0, free / (n - 1) as f32),
            FlexJustify::Around if free > 0.0 => {
                let g = free / n as f32;
                (g / 2.0, g)
            }
            FlexJustify::Evenly if free > 0.0 => {
                let g = free / (n + 1) as f32;
                (g, g)
            }
            // Start, single-item Between, or negative free space: flex-start.
            _ => (0.0, 0.0),
        }
    };
    let size = main_size.unwrap_or(total + gaps);
    let mut cursor = cursor_start;
    for (k, it) in items.iter_mut().enumerate() {
        let extra_ms = if it.auto_ms { each } else { 0.0 };
        let extra_me = if it.auto_me { each } else { 0.0 };
        let outer = it.m_ms + it.target + it.main_bp + it.m_me;
        it.main_pos = if reverse {
            size - cursor - it.target - it.main_bp - it.m_me - extra_me
        } else {
            cursor + it.m_ms + extra_ms
        };
        cursor += extra_ms + outer + extra_me + gap;
        // One fixed gap stride between adjacent items (never after the
        // last: `space-between` et al. already account for line edges).
        if k + 1 < n {
            cursor += main_gap;
        }
    }
}

/// Entry point for flex containers, called from `layout_block` after the
/// prologue resolved the container's horizontal metrics. `kids_x` /
/// `kids_w` are the prologue's scrollbar-gutter-reserved content origin
/// / width for children (see `layout_block`): items lay out against
/// those, not the node's unreserved metrics.
#[allow(clippy::too_many_arguments)]
fn layout_flex(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    idx: usize,
    input: &BlockInput,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    scratch: &mut LayoutScratch,
    kids_x: f32,
    kids_w: f32,
    // Whether this run already is the `overflow-y: auto` scrollbar-width
    // re-drive: carried into the [`FlexCtx`] so the completion does not
    // re-drive a second time.
    redriven: bool,
) {
    let block_level = flex_kind(forest, idx) == Some(true);
    // The gutter reservation in prologue terms: the shift past a left
    // (inline-end under RTL, or `both-edges`) strip and the total
    // reserved width, relative to this node's own unreserved content
    // geometry. Both are zero when no reservation applies — including
    // the propagated-root case, which the prologue already zeroes.
    let (gutter_shift, gutter_total) = {
        let n = &forest.nodes[idx];
        let own_x = n.rect.x + n.metrics.bl + n.metrics.pl;
        (kids_x - own_x, (n.metrics.content_w - kids_w).max(0.0))
    };
    // The `layout_block` prologue already ran `compute_mt_place`, which
    // collapses through the first flex item. Flex items never collapse
    // margins with their container, so undo that: the container keeps its
    // own top margin, and the touched first-child chain is reset for
    // recomputation with final item metrics. The walk uses the same
    // first-in-flow-block-child predicate as `compute_mt_place`, so the
    // reset covers exactly the chain the collapse pass may have walked.
    {
        let n = &mut forest.nodes[idx];
        n.mt_place = n.metrics.mt;
        n.top_collapsed = false;
    }
    {
        let mut cur = first_inflow_block_child(forest, idx);
        while let Some(c) = cur {
            forest.nodes[c].mt_place_done = false;
            cur = first_inflow_block_child(forest, c);
        }
    }
    if forest.nodes[idx].parent.is_none() {
        let mt = forest.nodes[idx].mt_place;
        forest.nodes[idx].rect.y = mt;
    }

    let style = forest.nodes[idx].style.as_deref();
    let dir = flex_dir_of(style);
    let horizontal = matches!(dir, FlexDir::Row | FlexDir::RowRev);
    // `direction: rtl` flips the row main axis (main-start becomes the
    // right edge). On the row axis the flip combines with `row-reverse`
    // by XOR: rtl+row and ltr+row-reverse both reverse, but rtl+
    // row-reverse cancels back to normal order. The column main axis
    // is unaffected by `direction`, but the column *cross* axis is
    // horizontal, so `direction: rtl` flips it: cross-start becomes the
    // right edge (flexbox §5.1).
    let row_rev = matches!(dir, FlexDir::RowRev);
    let row_axis = matches!(dir, FlexDir::Row | FlexDir::RowRev);
    let reverse = matches!(dir, FlexDir::ColRev) || (row_rev ^ (is_rtl(forest, idx) && row_axis));
    let justify = flex_justify_of(style);
    // `align-items` resolves once per container (a pre-existing
    // simplification: strictly, css-align-3 §4.1 resolves `self-start` /
    // `self-end` against each item's own direction, as the `align-self`
    // site above does). `false` keeps the historical mapping.
    let align_items = flex_align_of(
        style.and_then(|s| s.get_str_by_index(props::ALIGN_ITEMS)),
        false,
    )
    .unwrap_or(FlexAlign::Stretch);
    let wrap = flex_wrap_of(style);
    let align_content = align_content_of(style);

    let hspec = specified_height(forest, idx, input.cb_h);
    // `inline-flex` shrink-to-fit: size the container to the items'
    // max-content main sizes instead of the prologue's inline-block
    // estimate. Two passes (flexbox §9.3, CSS 2.1 §10.3.7): when the
    // container wraps, collect flex lines at the definite available
    // width first and fit to the longest line; a single-line container
    // still fits the unwrapped total.
    let mut content_w = if block_level {
        kids_w
    } else {
        forest.nodes[idx].metrics.content_w
    };
    // Flex gaps (flexbox §8.2). Percentage gaps resolve against the
    // content-box size in their own dimension (css-align-3 §2.2); the
    // cross size may be indefinite (`hspec` is `None`), in which case
    // the gap is zero. Resolved before the shrink-to-fit pass below so
    // line collection there can count gaps toward the break decision,
    // but an `inline-flex` container's final content-box width is only
    // known after that pass: its width-dependent percentage bases are
    // `None` here (percentage gaps resolve to 0 during intrinsic
    // sizing, css-sizing-3) and both gaps are re-resolved against the
    // final `content_w` after `resolve_metrics` below.
    let (mut main_gap, mut cross_gap) = {
        let d = &forest.nodes[idx].data;
        let (fs, rfs, vpw, vph) = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
        // Width-dependent percentage bases are unknown for
        // `inline-flex` until the shrink-to-fit pass; height bases
        // (`hspec`) are already final.
        let main_pct: Option<f32> = if horizontal {
            block_level.then_some(content_w)
        } else {
            hspec
        };
        let cross_pct: Option<f32> = if horizontal {
            hspec
        } else {
            block_level.then_some(content_w)
        };
        flex_gaps(style, horizontal, (fs, rfs, vpw, vph), main_pct, cross_pct)
    };
    if !block_level {
        // w78: a caller-determined content width wins over
        // shrink-to-fit. This is the blockified-inline-flex case:
        // a promoted inline-flex flex item arrives here with a
        // final flex §9.7 main size (`width_override_final`), and a
        // promoted inline-flex grid item with a stretch area (or a
        // column-flex measure width) arrives with a plain override.
        // The prologue already resolved `metrics.content_w` from
        // that override; re-running the estimate below would discard
        // it and size the container from its inner items instead
        // (a `flex: 0 0 200px` inline-flex item laid out at its 80px
        // content width, basis and grow lost). Unpromoted
        // inline-flex containers (line layout, floats, abspos) carry
        // no override and still shrink-to-fit below, as before.
        // The gap re-resolution and gutter narrowing after this
        // block still run: they only need the final `content_w`.
        if input.width_override.is_none() {
            // Pooled across containers (see `LayoutScratch::flex_items`):
            // the estimate pass restores the buffer right after the
            // longest-line computation, before the real pass below
            // re-takes it.
            let mut items = std::mem::take(&mut scratch.flex_items);
            flex_build_items(
                forest,
                idx,
                input.cb_w,
                hspec,
                horizontal,
                align_items,
                fonts,
                pw_cache,
                scratch,
                &mut items,
            );
            let wrap_main = if matches!(wrap, FlexWrap::NoWrap) {
                None
            } else {
                Some(input.cb_w)
            };
            let longest_line: f32 = layout_flex_wrap::collect_lines(&items, wrap_main, main_gap)
                .iter()
                .map(|&(s, e)| {
                    let items_sum: f32 = items[s..e]
                        .iter()
                        .map(|it| it.base + it.main_bp + it.m_ms + it.m_me)
                        .sum();
                    // Inter-item gaps are part of the line's extent (§9.3).
                    items_sum + main_gap * (e.saturating_sub(s).saturating_sub(1)) as f32
                })
                .fold(0.0f32, f32::max);
            // The estimate is done with the items: hand the buffer back to
            // the scratch so the real pass below (and later containers)
            // reuse its capacity.
            scratch.flex_items = items;
            let shrink_w = longest_line.min(input.cb_w).max(0.0);
            resolve_metrics(
                forest,
                idx,
                input.cb_x,
                input.cb_w,
                false,
                Some(shrink_w),
                false,
            );
            content_w = forest.nodes[idx].metrics.content_w;
        }
        // Re-resolve the gaps against the final content-box width: the
        // estimate pass above zeroed width-dependent percentage gaps
        // (css-sizing-3), and the stale prologue width must not leak
        // into the used values (css-align-3 §2.2).
        {
            let style = forest.nodes[idx].style.as_deref();
            let d = &forest.nodes[idx].data;
            let (fs, rfs, vpw, vph) =
                (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
            let main_pct: Option<f32> = if horizontal { Some(content_w) } else { hspec };
            let cross_pct: Option<f32> = if horizontal { hspec } else { Some(content_w) };
            (main_gap, cross_gap) =
                flex_gaps(style, horizontal, (fs, rfs, vpw, vph), main_pct, cross_pct);
        }
        // The shrink-to-fit pass sized the box from the items; the
        // gutter reservation narrows the area the items lay out
        // against inside that box (the box's own width is unchanged),
        // mirroring the block-level arm's `kids_w`.
        content_w = (content_w - gutter_total).max(0.0);
    }
    let (content_x, content_top) = {
        let n = &forest.nodes[idx];
        (
            n.rect.x + n.metrics.bl + n.metrics.pl + gutter_shift,
            n.rect.y + n.metrics.bt + n.metrics.pt,
        )
    };

    let mut ctx = FlexCtx {
        idx,
        content_x,
        content_top,
        content_w,
        cb_h: input.cb_h,
        hspec,
        reverse,
        cross_rtl: !horizontal && is_rtl(forest, idx),
        justify,
        items: Vec::new(),
        stage: FlexStage::RowFinish,
        container_cross: 0.0,
        lines: Vec::new(),
        align_content,
        wrap_reverse: wrap == FlexWrap::WrapReverse,
        main_gap,
        cross_gap,
        input: input.clone(),
        auto_redriven: redriven,
    };
    // The pooled buffer moves into the context; `flex_resume` hands it
    // back to the scratch when the container completes, so later
    // (including nested) containers reuse its capacity.
    let mut items = std::mem::take(&mut scratch.flex_items);
    flex_build_items(
        forest,
        idx,
        content_w,
        hspec,
        horizontal,
        align_items,
        fonts,
        pw_cache,
        scratch,
        &mut items,
    );
    ctx.items = items;

    if wrap != FlexWrap::NoWrap {
        layout_flex_wrap::place(
            forest, tasks, &mut ctx, fonts, pw_cache, horizontal, scratch,
        );
    } else if horizontal {
        flex_place_row(forest, tasks, ctx, scratch);
    } else if ctx.items.iter().any(|it| it.needs_measure) {
        // Column items with auto flex-basis and auto height need their
        // natural main size measured before the line can be resolved.
        ctx.stage = FlexStage::ColumnMeasured;
        // Staged in scratch and restored after: capacity survives
        // across containers.
        let mut measure_tasks = std::mem::take(&mut scratch.measure_tasks);
        for it in ctx.items.iter().rev() {
            if !it.needs_measure {
                continue;
            }
            measure_tasks.push(Task::Block(BlockInput {
                abs_pct_height_base: None,
                idx: it.idx,
                cb_x: 0.0,
                cb_w: content_w,
                cb_h: hspec,
                shrink: false,
                // w76fix finding 7: keyword-basis items measure at
                // the keyword width, not the final cross size; the
                // final layout still uses `it.cross` (see
                // `flex_place_column`), so only the measured height
                // (the base) changes. A zero minimum-height override
                // replaces a specified height for generic keyword
                // items, so the basis keyword — not the height
                // property — determines the measured height.
                width_override: Some(it.measure_cross.unwrap_or(it.cross)),
                width_override_final: false,
                height_override: if it.measure_height_min {
                    Some(0.0)
                } else {
                    None
                },
                height_is_min: it.measure_height_min,
                collapsed_border_wins: None,
                abspos: false,
                floats: fresh_floats(),
                is_float: false,
            }));
        }
        tasks.push(Task::Flex(ctx));
        tasks.append(&mut measure_tasks);
        scratch.measure_tasks = measure_tasks;
    } else {
        flex_place_column(forest, tasks, ctx, scratch);
    }
}

/// Place a row container's items: resolve flexible lengths, lay out
/// children at provisional (flex-start) cross positions, and resume to
/// finish the cross axis.
///
/// `ctx` is taken by value and moved into the resume task: the old
/// `ctx.clone()` here cost one `items` allocation plus a ~112B/item
/// memcpy per row container per layout for no reason — the caller's copy
/// is never used again.
fn flex_place_row(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    mut ctx: FlexCtx,
    scratch: &mut LayoutScratch,
) {
    flex_lengths(&mut ctx.items, Some(ctx.content_w), ctx.main_gap, scratch);
    // Final horizontal metrics and collapsed top margins before
    // positioning.
    for it in ctx.items.iter_mut() {
        resolve_metrics(
            forest,
            it.idx,
            0.0,
            ctx.content_w,
            false,
            Some(it.target),
            true,
        );
        compute_mt_place(forest, it.idx, ctx.hspec);
        it.m_cs = forest.nodes[it.idx].mt_place;
    }
    flex_justify(
        &mut ctx.items,
        Some(ctx.content_w),
        ctx.main_gap,
        ctx.justify,
        ctx.reverse,
    );
    ctx.stage = FlexStage::RowFinish;
    // Staged in scratch and restored after: capacity survives across
    // containers.
    let mut child_tasks = std::mem::take(&mut scratch.child_tasks);
    for it in ctx.items.iter_mut().rev() {
        let bx = ctx.content_x + it.main_pos;
        // Provisional cross position; `RowFinish` shifts to the final one.
        let by = ctx.content_top + it.m_cs;
        it.cross_pos = it.m_cs;
        let n = &mut forest.nodes[it.idx];
        n.rect.x = bx;
        n.rect.y = by;
        child_tasks.push(Task::Block(BlockInput {
            abs_pct_height_base: None,
            idx: it.idx,
            cb_x: bx - it.m_ms,
            cb_w: ctx.content_w,
            cb_h: ctx.hspec,
            shrink: false,
            width_override: Some(it.target),
            width_override_final: true,
            height_override: None,
            height_is_min: false,
            collapsed_border_wins: None,
            abspos: false,
            floats: fresh_floats(),
            is_float: false,
        }));
    }
    tasks.push(Task::Flex(ctx));
    tasks.append(&mut child_tasks);
    scratch.child_tasks = child_tasks;
}

/// Place a column container's items: the cross size (width) is definite,
/// so positions are final and children lay out exactly once.
fn flex_place_column(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    mut ctx: FlexCtx,
    scratch: &mut LayoutScratch,
) {
    flex_lengths(&mut ctx.items, ctx.hspec, ctx.main_gap, scratch);
    // Collapse each item's top margin with its first child before
    // distributing main-axis positions.
    for it in ctx.items.iter_mut() {
        compute_mt_place(forest, it.idx, Some(it.target));
        it.m_ms = forest.nodes[it.idx].mt_place;
    }
    flex_justify(
        &mut ctx.items,
        ctx.hspec,
        ctx.main_gap,
        ctx.justify,
        ctx.reverse,
    );
    ctx.stage = FlexStage::ColumnDone;
    // Staged in scratch and restored after: capacity survives across
    // containers.
    let mut child_tasks = std::mem::take(&mut scratch.child_tasks);
    for it in ctx.items.iter_mut().rev() {
        // Cross-axis auto margins absorb positive free space before
        // `align-self` (flexbox §8.3) and suppress the auto-width fill
        // (flexbox §9.8: stretch needs both cross margins non-auto).
        let auto_n = usize::from(it.auto_cs) + usize::from(it.auto_ce);
        // Cross size: auto widths fill (like stretch); specified widths
        // keep their size and are aligned within the column.
        if it.cross_auto && auto_n == 0 {
            // Aspect-ratio arm (flexbox §9.4 step 5): a replaced item
            // with an intrinsic ratio sizes its cross axis through the
            // ratio from the §9.7 main size instead of filling the
            // column — the fill must not distort it. The main size is
            // final, so the transfer never grows it back.
            it.cross = match intrinsic_ratio(forest, it.idx) {
                Some(ratio) => {
                    let (w, _) = replaced_minmax_transfer(
                        forest,
                        it.idx,
                        ctx.content_w,
                        Some(it.target),
                        (it.target * ratio).max(0.0),
                        it.target,
                        ratio,
                        true,
                        false,
                        false,
                    );
                    w
                }
                None => (ctx.content_w - it.m_cs - it.m_ce - it.cross_bp).max(0.0),
            };
        }
        let outer = it.cross + it.cross_bp;
        let free = ctx.content_w - it.m_cs - outer - it.m_ce;
        it.cross_pos = if free > 0.0 && auto_n > 0 {
            let each = free / auto_n as f32;
            it.m_cs + if it.auto_cs { each } else { 0.0 }
        } else {
            match it.align {
                FlexAlign::Start | FlexAlign::Stretch | FlexAlign::Baseline => it.m_cs,
                FlexAlign::End => ctx.content_w - it.m_ce - outer,
                FlexAlign::Center => it.m_cs + (ctx.content_w - it.m_cs - it.m_ce - outer) / 2.0,
            }
        };
        let bx = ctx.content_x
            + if ctx.cross_rtl {
                // Column cross axis follows inline direction (flexbox
                // §5.1): with `direction: rtl` cross-start is the right
                // edge, so mirror the cross-start-relative offset to
                // physical coordinates.
                ctx.content_w - it.cross_pos - outer
            } else {
                it.cross_pos
            };
        let by = ctx.content_top + it.main_pos;
        resolve_metrics(
            forest,
            it.idx,
            0.0,
            ctx.content_w,
            false,
            Some(it.cross),
            false,
        );
        let n = &mut forest.nodes[it.idx];
        n.rect.x = bx;
        n.rect.y = by;
        child_tasks.push(Task::Block(BlockInput {
            abs_pct_height_base: None,
            idx: it.idx,
            // `cb_x` is the physical margin-box left edge: with
            // `direction: rtl` the physical left margin is the
            // cross-end one (margins were swapped at item build).
            cb_x: bx - if ctx.cross_rtl { it.m_ce } else { it.m_cs },
            cb_w: ctx.content_w,
            cb_h: Some(it.target),
            shrink: false,
            width_override: Some(it.cross),
            width_override_final: false,
            height_override: Some(it.target),
            height_is_min: false,
            collapsed_border_wins: None,
            abspos: false,
            floats: fresh_floats(),
            is_float: false,
        }));
    }
    tasks.push(Task::Flex(ctx));
    tasks.append(&mut child_tasks);
    scratch.child_tasks = child_tasks;
}

/// Row cross axis: with natural item heights known, size the container,
/// align items, stretch where applicable (re-laying those out), and shift
/// the rest into place.
///
/// Returns `Some(ctx)` when the container completed directly (no stretch
/// relayout follows) and `None` when the stretched-item relayout was
/// staged: in the latter case `ctx` was moved into the resume task, not
/// cloned — the old `ctx.clone()` here cost one `items` allocation plus
/// a ~112B/item memcpy per stretched row container per layout.
fn flex_row_finish(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    mut ctx: FlexCtx,
    scratch: &mut LayoutScratch,
) -> Option<FlexCtx> {
    for it in ctx.items.iter_mut() {
        it.natural_cross = forest.nodes[it.idx].content_h;
        it.baseline = forest.nodes[it.idx].baseline;
    }
    let container_cross = match ctx.hspec {
        Some(h) => h,
        None => ctx
            .items
            .iter()
            .map(|it| it.m_cs + it.natural_cross + it.cross_bp + it.m_ce)
            .fold(0.0, f32::max),
    };
    ctx.container_cross = container_cross;
    let max_bl = ctx
        .items
        .iter()
        .filter(|it| it.align == FlexAlign::Baseline)
        .map(|it| it.m_cs + it.baseline)
        .fold(0.0, f32::max);
    for it in ctx.items.iter_mut() {
        let outer = it.natural_cross + it.cross_bp;
        // Cross-axis auto margins absorb positive free space before
        // `align-self` (flexbox §8.3); with any auto cross margin present
        // alignment is skipped, and `stretch` does not apply either
        // (flexbox §9.8: stretch needs both cross margins non-auto).
        let free = container_cross - it.m_cs - outer - it.m_ce;
        let auto_n = usize::from(it.auto_cs) + usize::from(it.auto_ce);
        let (cross, pos) = if free > 0.0 && auto_n > 0 {
            let each = free / auto_n as f32;
            (
                it.natural_cross,
                it.m_cs + if it.auto_cs { each } else { 0.0 },
            )
        } else {
            match it.align {
                FlexAlign::Stretch => {
                    // Stretch only applies to auto cross sizes.
                    let auto_cross = !it.auto_cs
                        && !it.auto_ce
                        && matches!(forest.nodes[it.idx].data.height, SpecLen::Auto);
                    if auto_cross {
                        // Aspect-ratio arm (flexbox §9.4 step 5): a
                        // replaced item with an intrinsic ratio keeps its
                        // hypothetical cross size — transferred from the
                        // §9.7 main size by `layout_block`'s sequential
                        // min/max pass, already in `natural_cross` —
                        // instead of stretching (stretch must not distort
                        // it). No re-layout is needed.
                        if intrinsic_ratio(forest, it.idx).is_some() {
                            (it.natural_cross, it.m_cs)
                        } else {
                            let c = (container_cross - it.m_cs - it.m_ce - it.cross_bp).max(0.0);
                            it.stretched = true;
                            (c, it.m_cs)
                        }
                    } else {
                        (it.natural_cross, it.m_cs)
                    }
                }
                FlexAlign::Start => (it.natural_cross, it.m_cs),
                FlexAlign::End => (it.natural_cross, container_cross - it.m_ce - outer),
                FlexAlign::Center => (
                    it.natural_cross,
                    it.m_cs + (container_cross - it.m_cs - it.m_ce - outer) / 2.0,
                ),
                FlexAlign::Baseline => (it.natural_cross, max_bl - it.baseline),
            }
        };
        it.cross = cross;
        if it.stretched {
            // Final position is known; the re-layout below fills the
            // stretched height (children need the rect first).
            let n = &mut forest.nodes[it.idx];
            n.rect.y = ctx.content_top + pos;
            it.cross_pos = pos;
        } else {
            let dy = pos - it.cross_pos;
            if dy != 0.0 {
                // Deferred cross-axis shift: O(1) per item, applied once
                // by `apply_pending_shifts` after `run_tasks`. `RowFinish`
                // runs exactly once per item and `dy` is measured against
                // the provisional rect, matching `record_shift`'s
                // set-semantics contract.
                record_shift(forest, it.idx, 0.0, dy);
            }
            it.cross_pos = pos;
        }
    }
    if ctx.items.iter().any(|it| it.stretched) {
        ctx.stage = FlexStage::RowDone;
        // Staged in scratch and restored after: capacity survives
        // across containers.
        let mut relayout = std::mem::take(&mut scratch.relayout);
        for it in ctx.items.iter().rev() {
            if !it.stretched {
                continue;
            }
            // The subtree is re-laid out from provisional rects below;
            // drop its deferred inline-block shifts first so the fresh
            // pass re-records them instead of applying stale ones twice.
            clear_pending_shifts(forest, it.idx);
            relayout.push(Task::Block(BlockInput {
                abs_pct_height_base: None,
                idx: it.idx,
                cb_x: ctx.content_x + it.main_pos - it.m_ms,
                cb_w: ctx.content_w,
                cb_h: ctx.hspec,
                shrink: false,
                width_override: Some(it.target),
                width_override_final: true,
                height_override: Some(it.cross),
                height_is_min: false,
                collapsed_border_wins: None,
                abspos: false,
                floats: fresh_floats(),
                is_float: false,
            }));
        }
        tasks.push(Task::Flex(ctx));
        tasks.append(&mut relayout);
        scratch.relayout = relayout;
        None
    } else {
        flex_finish_container(forest, &ctx, container_cross);
        Some(ctx)
    }
}

/// First baseline of the inner content of an inline-level flex, grid, or
/// table container, measured from the container's border-box top (v1).
///
/// CSS 2.1 §10.8.1 gives `inline-table` the baseline of its first row;
/// css-flexbox-1 and css-align-3 give `inline-flex` / `inline-grid` the
/// first baseline set of their inner content. v1 approximates all three
/// with the first content-derived baseline found in a pre-order walk of
/// the in-flow children: structural boxes that carry no baseline of
/// their own (table row groups, rows, columns — their `baseline` field
/// is the 0.0 default) are descended through, out-of-flow children are
/// skipped, and the first other child wins. Its baseline (relative to
/// its own border-box top) is shifted by its offset from the
/// container's border-box top, including deferred shifts recorded on
/// the path (row/cell placement records shifts instead of mutating
/// rects; see `record_shift`). With no inner content the border-box
/// bottom is returned, preserving the historical behavior.
fn first_inner_baseline(forest: &Forest, idx: usize) -> f32 {
    let top = forest.nodes[idx].rect.y;
    // (node, accumulated deferred y-shift on the path from the
    // container's children down to and including the node).
    let mut stack: Vec<(usize, f32)> = forest.nodes[idx]
        .children
        .iter()
        .rev()
        .map(|&c| (c, forest.nodes[c].pend_y))
        .collect();
    while let Some((i, acc)) = stack.pop() {
        if layout_position::is_out_of_flow(forest, i) {
            continue;
        }
        let n = &forest.nodes[i];
        if matches!(
            n.box_type,
            BoxType::TableRowGroup
                | BoxType::TableRow
                | BoxType::TableColumn
                | BoxType::TableColumnGroup
        ) {
            stack.extend(
                n.children
                    .iter()
                    .rev()
                    .map(|&c| (c, acc + forest.nodes[c].pend_y)),
            );
            continue;
        }
        return (n.rect.y + acc) - top + n.baseline;
    }
    forest.nodes[idx].rect.h
}

/// Set the container's height, bottom margin, and baseline once all items
/// are placed. `natural` is the content-box cross/main extent the items
/// need; a specified container size wins.
fn flex_finish_container(forest: &mut Forest, ctx: &FlexCtx, natural: f32) {
    let idx = ctx.idx;
    let content_h = clamp_height(forest, idx, ctx.cb_h, ctx.hspec.unwrap_or(natural));
    let (bt, pt, pb, bb, mb) = {
        let m = &forest.nodes[idx].metrics;
        (m.bt, m.pt, m.pb, m.bb, m.mb)
    };
    // Inline-level flex containers align by the first baseline of their
    // inner content (see `first_inner_baseline`); block-level containers
    // keep the border-box bottom.
    let baseline = if forest.nodes[idx].data.display == Display::InlineFlex {
        first_inner_baseline(forest, idx)
    } else {
        bt + pt + content_h + pb + bb
    };
    let n = &mut forest.nodes[idx];
    n.content_h = content_h;
    n.rect.h = bt + pt + content_h + pb + bb;
    // Flex containers never collapse margins with their items.
    n.mb_place = mb;
    n.baseline = baseline;
}

fn flex_resume(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    mut ctx: FlexCtx,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    scratch: &mut LayoutScratch,
) {
    match ctx.stage {
        FlexStage::RowFinish => {
            if let Some(mut ctx) = flex_row_finish(forest, tasks, ctx, scratch) {
                // `flex_row_finish` completed the container directly
                // (no `align-self: stretch` relayout, so no `RowDone`
                // follows): run the re-drive check here. The stretch
                // path checks in `RowDone` instead.
                let natural = flex_row_natural(&ctx);
                flex_auto_redrive(forest, tasks, &ctx, natural, fonts, pw_cache, scratch);
                // The context is done: hand the pooled item buffer back
                // to the scratch (see `LayoutScratch::flex_items`).
                scratch.flex_items = std::mem::take(&mut ctx.items);
            }
        }
        FlexStage::RowDone => {
            flex_finish_container(forest, &ctx, ctx.container_cross);
            let natural = flex_row_natural(&ctx);
            flex_auto_redrive(forest, tasks, &ctx, natural, fonts, pw_cache, scratch);
            scratch.flex_items = std::mem::take(&mut ctx.items);
        }
        FlexStage::ColumnMeasured => {
            for it in ctx.items.iter_mut() {
                if it.needs_measure {
                    let measured = forest.nodes[it.idx].content_h.max(0.0);
                    it.base = measured.min(it.max_main).max(it.min_main).max(0.0);
                    it.needs_measure = false;
                }
            }
            flex_place_column(forest, tasks, ctx, scratch);
        }
        FlexStage::ColumnDone => {
            let items_sum: f32 = ctx
                .items
                .iter()
                .map(|it| it.m_ms + it.target + it.main_bp + it.m_me)
                .sum();
            // One main-axis gap stride between adjacent items.
            let natural = items_sum + ctx.main_gap * ctx.items.len().saturating_sub(1) as f32;
            flex_finish_container(forest, &ctx, natural);
            flex_auto_redrive(forest, tasks, &ctx, natural, fonts, pw_cache, scratch);
        }
        FlexStage::WrapRowFinish => {
            layout_flex_wrap::wrap_row_finish(forest, tasks, &mut ctx, scratch);
            if ctx.stage == FlexStage::WrapRowFinish {
                // `wrap_row_finish` completed the container directly
                // (no stretch relayout, so no `WrapRowDone` follows):
                // run the re-drive check here.
                let natural = flex_wrap_row_natural(&ctx);
                flex_auto_redrive(forest, tasks, &ctx, natural, fonts, pw_cache, scratch);
                scratch.flex_items = std::mem::take(&mut ctx.items);
            } else {
                // Stretched: the context was cloned into the resume
                // task; this original is dropped here, so its pooled
                // item buffer goes back to the scratch (the clone
                // carries its own deep copy).
                scratch.flex_items = std::mem::take(&mut ctx.items);
            }
        }
        FlexStage::WrapRowDone => {
            layout_flex_wrap::wrap_row_done(forest, &ctx);
            let natural = flex_wrap_row_natural(&ctx);
            flex_auto_redrive(forest, tasks, &ctx, natural, fonts, pw_cache, scratch);
            scratch.flex_items = std::mem::take(&mut ctx.items);
        }
        FlexStage::WrapColumnMeasured => {
            for it in ctx.items.iter_mut() {
                if it.needs_measure {
                    let measured = forest.nodes[it.idx].content_h.max(0.0);
                    it.base = measured.min(it.max_main).max(it.min_main).max(0.0);
                    it.needs_measure = false;
                }
            }
            layout_flex_wrap::wrap_place_column(forest, tasks, &mut ctx, fonts, pw_cache, scratch);
        }
        FlexStage::WrapColumnDone => {
            layout_flex_wrap::wrap_column_done(forest, &ctx);
            // `natural` mirrors `wrap_column_done`'s line-extent maximum
            // (see `layout_flex_wrap.rs`): the largest main-axis line
            // extent. Kept in sync by hand — the wrap module is owned
            // elsewhere, so the loop is duplicated rather than the
            // signature changed.
            let mut natural = 0.0f32;
            for &(s, e) in &ctx.lines {
                let items_sum: f32 = ctx.items[s..e]
                    .iter()
                    .map(|it| it.m_ms + it.target + it.main_bp + it.m_me)
                    .sum();
                let line_main =
                    items_sum + ctx.main_gap * (e.saturating_sub(s).saturating_sub(1)) as f32;
                natural = natural.max(line_main);
            }
            flex_auto_redrive(forest, tasks, &ctx, natural, fonts, pw_cache, scratch);
            scratch.flex_items = std::mem::take(&mut ctx.items);
        }
    }
}

/// True natural cross size of a row flex container: with a definite
/// `hspec`, `container_cross` is already the specified height, which
/// would mask overflow in the re-drive check — measure the items
/// instead (mirrors `flex_row_finish`'s auto-height branch).
fn flex_row_natural(ctx: &FlexCtx) -> f32 {
    ctx.items
        .iter()
        .map(|it| it.m_cs + it.natural_cross + it.cross_bp + it.m_ce)
        .fold(0.0f32, f32::max)
}

/// True natural cross size of a wrapping row flex container (mirrors
/// `wrap_row_finish`'s auto-height branch): the stacked line extents
/// plus inter-line gaps.
fn flex_wrap_row_natural(ctx: &FlexCtx) -> f32 {
    let mut natural = 0.0f32;
    for &(s, e) in &ctx.lines {
        natural += ctx.items[s..e]
            .iter()
            .map(|it| it.m_cs + it.natural_cross + it.cross_bp + it.m_ce)
            .fold(0.0f32, f32::max);
    }
    natural + ctx.cross_gap * ctx.lines.len().saturating_sub(1) as f32
}

/// Run the shared `overflow-y: auto` scrollbar re-drive check after a
/// flex container finished: when a shown scrollbar went unreserved,
/// re-run the whole flex layout once against the narrowed width. The
/// original [`BlockInput`] is replayed unchanged — only the children's
/// available width shrinks — and the re-driven context carries
/// `auto_redriven`, so the second completion cannot re-drive again.
fn flex_auto_redrive(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    ctx: &FlexCtx,
    natural: f32,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    scratch: &mut LayoutScratch,
) {
    let idx = ctx.idx;
    let check = AutoRedriveCheck {
        idx,
        natural_h: natural,
        content_h: forest.nodes[idx].content_h,
        content_w: ctx.content_w,
        content_x: ctx.content_x,
        auto_redriven: ctx.auto_redriven,
    };
    if let Some((kids_x2, kids_w2)) = auto_scrollbar_redrive(forest, scratch, &check) {
        let input = ctx.input.clone();
        layout_flex(
            forest, tasks, idx, &input, fonts, pw_cache, scratch, kids_x2, kids_w2, true,
        );
    }
}

#[cfg(test)]
mod flex_tests {
    use crate::{BoxType, FixedFontMetrics, LayoutBox, Size, layout};

    const VP: Size = Size { w: 800.0, h: 600.0 };

    fn layout_html(html: &str, css: &str) -> LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        layout(&doc, &[&ss], VP, &FixedFontMetrics)
    }

    fn body(l: &LayoutBox) -> &LayoutBox {
        &l.children[0]
    }

    /// The first child of `<body>`: the flex container under test.
    fn first(l: &LayoutBox) -> &LayoutBox {
        &body(l).children[0]
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// Direct `(min-content, max-content)` measurement of the first
    /// flex container in the document, through both intrinsic
    /// dispatches (the `layout_html` tests only observe the used size
    /// a shrink-to-fit ancestor derives from them).
    fn measure_flex_intrinsic(html: &str, css: &str) -> (f32, f32) {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let ua = bowser_css::default_ua_stylesheet();
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut fc = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if super::flex_kind(&forest, i).is_some() {
                fc = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let fc = fc.expect("flex container");
        let fonts = crate::FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, fc, &fonts, &mut scratch);
        let mut cache = super::PrefWidthCache::new(forest.nodes.len());
        let pc = super::preferred_width(&mut forest, fc, &fonts, &mut cache);
        (mc, pc)
    }

    #[test]
    fn row_grow_distributes_free_space() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; } \
             .a { flex-basis: 100px; flex-grow: 1; } \
             .b { flex-basis: 100px; flex-grow: 2; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        let b = &c.children[1];
        // Free space 400px split 1:2.
        assert!(approx(a.rect.w, 233.333));
        assert!(approx(b.rect.w, 366.667));
        assert!(approx(a.rect.x, c.rect.x));
        assert!(approx(b.rect.x, c.rect.x + 233.333));
    }

    #[test]
    fn infinite_flex_factor_does_not_hang() {
        // `flex-grow: 1e999` / `flex-shrink: 1e999` parse to infinity;
        // the factor clamp keeps `flex_lengths` finite so the
        // distribution loop terminates (`remaining / inf * inf` = NaN
        // would never converge).
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; } \
             .a { flex-basis: 100px; flex-grow: 1e999; } \
             .b { flex-basis: 100px; flex-shrink: 1e999; }",
        );
        let c = first(&l);
        let (a, b) = (&c.children[0], &c.children[1]);
        assert!(a.rect.w.is_finite() && b.rect.w.is_finite());
        // Clamped grow takes all free space: 500 + 100 = 600.
        assert!(approx(a.rect.w, 500.0));
        assert!(approx(b.rect.w, 100.0));
    }

    #[test]
    fn denormal_flex_basis_shrink_does_not_hang() {
        // A denormal flex base (1e-40px) makes the total scaled shrink
        // factor a denormal, so `remaining / total_scaled` overflows to
        // -inf while both operands are finite — the old non-finite
        // guard missed it. The zero-scaled item then computes
        // `raw = base + -inf * 0.0 = NaN`, every violation is NaN, no
        // item ever freezes, and the distribute-and-freeze loop hangs.
        // The quotient must be covered by the escape: active items
        // saturate at their clamped base.
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 50px; } \
             .a { flex-basis: 1e-40px; flex-shrink: 1; } \
             .b { flex-basis: 100px; flex-shrink: 0; }",
        );
        let c = first(&l);
        let (a, b) = (&c.children[0], &c.children[1]);
        assert!(a.rect.w.is_finite() && b.rect.w.is_finite());
        assert!(approx(a.rect.w, 0.0));
        assert!(approx(b.rect.w, 100.0));
    }

    #[test]
    fn fixed_pct_table_in_min_content_flex_stays_finite_container() {
        // WPT `css/css-tables/tentative/table-fixed-minmax.html`
        // ("scenario 1"): a fixed-layout table with a percentage width
        // has an infinite max-content intrinsic size, but a
        // `width: min-content` flex container stays finite — the table's
        // min-content is regular — while the table item keeps its
        // infinite size, saturated to a huge finite constant. The
        // distribute-and-freeze loop in `flex_lengths` cannot converge
        // on the NaN the infinite base produces (`remaining / NaN`),
        // so it must break instead of hanging.
        let l = layout_html(
            r#"<div class="c"><table class="t"><tr><td></td></tr></table></div>"#,
            ".c { display: flex; width: min-content; } \
             .t { table-layout: fixed; width: 100%; flex: 0 0 content; min-width: 0px; \
                  border-spacing: 0; } \
             td { width: 100px; padding: 0; }",
        );
        let c = first(&l);
        let t = &c.children[0];
        // The container is finite: the table's regular min-content.
        assert!(c.rect.w.is_finite());
        assert!(approx(c.rect.w, 100.0));
        // The table itself is "infinite": saturated, finite, wider than
        // the viewport.
        assert!(t.rect.w.is_finite());
        assert!(t.rect.w > 800.0);
    }

    #[test]
    fn fixed_pct_table_in_min_content_flex_wrapper_stays_finite() {
        // WPT `css/css-tables/tentative/table-fixed-minmax.html`
        // ("scenario 2"): the fixed-percentage table nested in a block
        // wrapper inside a `width: min-content` flex container. The
        // container still measures finite (the wrapper's min-content is
        // regular) while the table lays out saturated-infinite.
        let l = layout_html(
            r#"<div class="c"><div class="w"><table class="t"><tr><td></td></tr></table></div></div>"#,
            ".c { display: flex; width: min-content; } \
             .w { flex: 0 0 auto; min-width: 0px; } \
             .t { table-layout: fixed; width: 100%; border-spacing: 0; } \
             td { width: 100px; padding: 0; }",
        );
        let c = first(&l);
        let w = &c.children[0];
        let t = &w.children[0];
        assert!(c.rect.w.is_finite());
        assert!(approx(c.rect.w, 100.0));
        assert!(t.rect.w.is_finite());
        assert!(t.rect.w > 800.0);
    }

    #[test]
    fn fixed_pct_table_in_max_content_flex_is_saturated_infinite() {
        // WPT `css/css-tables/tentative/table-fixed-minmax.html`: a
        // `width: max-content` flex container propagates the table's
        // infinite max-content, and the table lays out at the saturated
        // size — finite, wider than the viewport.
        let l = layout_html(
            r#"<div class="c"><table class="t"><tr><td></td></tr></table><div class="d"></div></div>"#,
            ".c { display: flex; width: max-content; } \
             .t { table-layout: fixed; width: 100%; border-spacing: 0; } \
             td { width: 100px; padding: 0; } \
             .d { width: 100px; flex-shrink: 0; }",
        );
        let c = first(&l);
        let t = &c.children[0];
        assert!(c.rect.w.is_finite());
        assert!(c.rect.w > 800.0);
        assert!(t.rect.w.is_finite());
        assert!(t.rect.w > 800.0);
    }

    #[test]
    fn row_shrink_takes_space_proportionally() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 300px; } \
             .a { flex-basis: 200px; flex-shrink: 1; } \
             .b { flex-basis: 200px; flex-shrink: 2; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        let b = &c.children[1];
        // Overflow 100px taken proportional to shrink * base (200 vs 400).
        assert!(approx(a.rect.w, 166.667));
        assert!(approx(b.rect.w, 133.333));
        assert!(approx(b.rect.x, c.rect.x + 166.667));
    }

    #[test]
    fn row_without_flex_factors_keeps_basis() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; } \
             .a { flex-basis: 100px; } .b { flex-basis: 150px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        let b = &c.children[1];
        assert!(approx(a.rect.w, 100.0));
        assert!(approx(b.rect.w, 150.0));
        assert!(approx(b.rect.x, c.rect.x + 100.0));
    }

    #[test]
    fn row_basis_percentage_resolves_against_container() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; width: 600px; } .a { flex-basis: 50%; }",
        );
        let c = first(&l);
        assert!(approx(c.children[0].rect.w, 300.0));
    }

    #[test]
    fn row_basis_clamped_by_max_width() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; width: 600px; } \
             .a { flex-basis: 500px; max-width: 200px; }",
        );
        let c = first(&l);
        assert!(approx(c.children[0].rect.w, 200.0));
    }

    #[test]
    fn column_stacks_items_vertically() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; flex-direction: column; } \
             .a { height: 50px; } .b { height: 70px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        let b = &c.children[1];
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(b.rect.y, c.rect.y + 50.0));
        assert!(approx(c.rect.h, 120.0));
        // Cross axis fills the container width.
        assert!(approx(a.rect.w, 800.0));
    }

    #[test]
    fn column_grow_distributes_free_space() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; flex-direction: column; height: 600px; } \
             .a { flex-basis: 100px; flex-grow: 1; } \
             .b { flex-basis: 100px; flex-grow: 1; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        let b = &c.children[1];
        assert!(approx(a.rect.h, 300.0));
        assert!(approx(b.rect.h, 300.0));
        assert!(approx(b.rect.y, c.rect.y + 300.0));
    }

    #[test]
    fn column_auto_basis_measures_content_height() {
        let l = layout_html(
            r#"<div class="c"><div class="a">hi</div></div>"#,
            ".c { display: flex; flex-direction: column; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // "hi" at 16px: one line, 1.2 * 16 = 19.2px tall.
        assert!(approx(a.rect.h, 19.2));
        assert!(approx(c.rect.h, 19.2));
    }

    #[test]
    fn row_reverse_mirrors_positions() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; flex-direction: row-reverse; width: 600px; } \
             .a { flex-basis: 100px; } .b { flex-basis: 100px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        let b = &c.children[1];
        assert!(approx(a.rect.x, c.rect.x + 500.0));
        assert!(approx(b.rect.x, c.rect.x + 400.0));
    }

    #[test]
    fn column_reverse_mirrors_positions() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; flex-direction: column-reverse; } \
             .a { height: 50px; } .b { height: 70px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        let b = &c.children[1];
        assert!(approx(a.rect.y, c.rect.y + 70.0));
        assert!(approx(b.rect.y, c.rect.y));
    }

    #[test]
    fn justify_center() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; justify-content: center; } \
             .a { flex-basis: 100px; } .b { flex-basis: 100px; }",
        );
        let c = first(&l);
        assert!(approx(c.children[0].rect.x, c.rect.x + 200.0));
        assert!(approx(c.children[1].rect.x, c.rect.x + 300.0));
    }

    #[test]
    fn justify_flex_end() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; justify-content: flex-end; } \
             .a { flex-basis: 100px; } .b { flex-basis: 100px; }",
        );
        let c = first(&l);
        assert!(approx(c.children[0].rect.x, c.rect.x + 400.0));
        assert!(approx(c.children[1].rect.x, c.rect.x + 500.0));
    }

    #[test]
    fn justify_space_between() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; justify-content: space-between; } \
             .a { flex-basis: 100px; } .b { flex-basis: 100px; }",
        );
        let c = first(&l);
        assert!(approx(c.children[0].rect.x, c.rect.x));
        assert!(approx(c.children[1].rect.x, c.rect.x + 500.0));
    }

    #[test]
    fn justify_space_around() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; justify-content: space-around; } \
             .a { flex-basis: 100px; } .b { flex-basis: 100px; }",
        );
        let c = first(&l);
        assert!(approx(c.children[0].rect.x, c.rect.x + 100.0));
        assert!(approx(c.children[1].rect.x, c.rect.x + 400.0));
    }

    #[test]
    fn justify_space_evenly() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; justify-content: space-evenly; } \
             .a { flex-basis: 100px; } .b { flex-basis: 100px; }",
        );
        let c = first(&l);
        assert!(approx(c.children[0].rect.x, c.rect.x + 133.333));
        assert!(approx(c.children[1].rect.x, c.rect.x + 366.667));
    }

    #[test]
    fn align_items_center() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; height: 200px; align-items: center; } \
             .a { width: 100px; height: 50px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        assert!(approx(a.rect.y, c.rect.y + 75.0));
        assert!(approx(a.rect.h, 50.0));
    }

    #[test]
    fn align_items_flex_end() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; height: 200px; align-items: flex-end; } \
             .a { width: 100px; height: 50px; }",
        );
        let c = first(&l);
        assert!(approx(c.children[0].rect.y, c.rect.y + 150.0));
    }

    #[test]
    fn align_items_stretch_fills_container_height() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; height: 200px; } .a { width: 100px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        assert!(approx(a.rect.h, 200.0));
        assert!(approx(a.rect.y, c.rect.y));
    }

    #[test]
    fn align_self_overrides_align_items() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; height: 200px; align-items: flex-start; } \
             .a { width: 100px; height: 50px; } \
             .b { width: 100px; height: 50px; align-self: center; }",
        );
        let c = first(&l);
        assert!(approx(c.children[0].rect.y, c.rect.y));
        assert!(approx(c.children[1].rect.y, c.rect.y + 75.0));
    }

    #[test]
    fn align_self_self_start_uses_item_direction() {
        // CSS Box Alignment 3 §4.1: `self-start` resolves against the
        // *item's* direction, not the container's. Column flex with
        // `direction: rtl` puts cross-start at the physical right edge,
        // but the ltr item's `self-start` is the physical left edge.
        let l = layout_html(
            r#"<div class="c"><div class="a" dir="ltr"></div></div>"#,
            ".c { display: flex; flex-direction: column; direction: rtl; width: 200px; height: 200px; } \
             .a { width: 50px; height: 50px; align-self: self-start; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        assert!(approx(a.rect.x, c.rect.x));
    }

    #[test]
    fn align_self_self_end_uses_item_direction() {
        // The rtl item's `self-end` is its inline-end: the physical left
        // edge, even though the ltr container's cross-end is the
        // physical right edge.
        let l = layout_html(
            r#"<div class="c"><div class="a" dir="rtl"></div></div>"#,
            ".c { display: flex; flex-direction: column; width: 200px; height: 200px; } \
             .a { width: 50px; height: 50px; align-self: self-end; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        assert!(approx(a.rect.x, c.rect.x));
    }

    #[test]
    fn align_self_self_start_matching_direction() {
        // The item inherits the container's rtl: `self-start` is the
        // physical right edge, same as the container-relative start.
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; flex-direction: column; direction: rtl; width: 200px; height: 200px; } \
             .a { width: 50px; height: 50px; align-self: self-start; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        assert!(approx(a.rect.x, c.rect.x + 150.0));
    }

    #[test]
    fn align_items_baseline_aligns_baselines() {
        let l = layout_html(
            r#"<div class="c"><div class="a">x</div><div class="b">x</div></div>"#,
            ".c { display: flex; align-items: baseline; } \
             .a { font-size: 16px; } .b { font-size: 32px; }",
        );
        let c = first(&l);
        // Block-level children are separate flex items (inline children
        // would merge into one anonymous item per flexbox §4).
        let a = &c.children[0];
        let b = &c.children[1];
        // TextFragment.baseline is absolute; the two lines' baselines
        // must coincide.
        let ba = a.children[0].fragments[0].baseline;
        let bb = b.children[0].fragments[0].baseline;
        assert!(approx(ba, bb));
        // The smaller text sits lower so the baselines meet.
        assert!(a.rect.y > b.rect.y);
    }

    #[test]
    fn text_child_becomes_sized_item() {
        let l = layout_html(r#"<div class="c">hi</div>"#, ".c { display: flex; }");
        let c = first(&l);
        // The text run is wrapped in an anonymous block flex item
        // (flexbox §4); the item sizes to the run.
        let w = &c.children[0];
        assert_eq!(w.box_type, BoxType::AnonymousBlock);
        let t = &w.children[0];
        assert_eq!(t.box_type, BoxType::Text);
        // "hi" at 16px with 0.6 advance per char.
        assert!(approx(w.rect.w, 19.2));
        assert!(approx(w.rect.h, 19.2));
        assert_eq!(t.fragments.len(), 1);
    }

    #[test]
    fn item_with_inline_content_sizes_to_content() {
        let l = layout_html(
            r#"<div class="c"><div class="a">hi</div></div>"#,
            ".c { display: flex; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        assert!(approx(a.rect.h, 19.2));
        assert!(approx(c.rect.h, 19.2));
    }

    #[test]
    fn inline_flex_shrink_to_fit() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: inline-flex; } \
             .a { width: 100px; height: 10px; } .b { width: 100px; height: 10px; }",
        );
        let c = &body(&l).children[0];
        assert!(approx(c.rect.w, 200.0));
        assert!(approx(c.children[1].rect.x, c.rect.x + 100.0));
    }

    #[test]
    fn flex_row_intrinsic_width_in_shrink_to_fit_parent() {
        // A flex container measured by a shrink-to-fit ancestor (here
        // a float) must contribute its row sum plus gaps, not its
        // widest item: before the flex intrinsic arms, the generic
        // sizing reported 100 here and the float wrapped at 100px.
        let l = layout_html(
            r#"<div class="f"><div class="c"><div class="a"></div><div class="b"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .c { display: flex; column-gap: 20px; } \
             .a { width: 100px; height: 10px; } .b { width: 50px; height: 10px; }",
        );
        let f = first(&l);
        assert!(
            approx(f.rect.w, 170.0),
            "float w {} != items 100 + 50 + gap 20",
            f.rect.w
        );
        let c = &f.children[0];
        assert!(approx(c.rect.w, 170.0), "flex w {}", c.rect.w);
    }

    #[test]
    fn flex_column_intrinsic_width_in_shrink_to_fit_parent() {
        // Column direction: the intrinsic width is the widest item —
        // the items stack vertically, so neither their sum nor the
        // (cross-axis) column gap contributes.
        let l = layout_html(
            r#"<div class="f"><div class="c"><div class="a"></div><div class="b"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .c { display: flex; flex-direction: column; column-gap: 20px; } \
             .a { width: 100px; height: 10px; } .b { width: 50px; height: 10px; }",
        );
        let f = first(&l);
        assert!(
            approx(f.rect.w, 100.0),
            "float w {} != widest item 100",
            f.rect.w
        );
    }

    #[test]
    fn inline_flex_intrinsic_width_in_shrink_to_fit_parent() {
        // The atomic `inline-flex` form takes the same flex arm (its
        // box type is `InlineBlock`): a float around it sizes to the
        // row sum plus gaps.
        let l = layout_html(
            r#"<div class="f"><div class="c"><div class="a"></div><div class="b"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .c { display: inline-flex; column-gap: 20px; } \
             .a { width: 100px; height: 10px; } .b { width: 50px; height: 10px; }",
        );
        let f = first(&l);
        assert!(
            approx(f.rect.w, 170.0),
            "float w {} != items 100 + 50 + gap 20",
            f.rect.w
        );
    }

    #[test]
    fn flex_intrinsic_widths_measured_directly() {
        // Direct measurement of both dispatches (the float tests above
        // only observe the max-content end): row = sum + gaps in both
        // min- and max-content terms for definite-width items; column
        // = widest item in both.
        fn intrinsic(html: &str, css: &str) -> (f32, f32) {
            let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
            let ss = bowser_css::parse_stylesheet(css);
            let ua = bowser_css::default_ua_stylesheet();
            let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
                &doc,
                &[&ss],
                &ua,
                800.0,
                600.0,
                &std::collections::HashMap::new(),
                &crate::FixedFontMetrics,
            ) else {
                panic!("no root box");
            };
            let mut fc = None;
            let mut stack = vec![root];
            while let Some(i) = stack.pop() {
                if super::flex_kind(&forest, i).is_some() {
                    fc = Some(i);
                    break;
                }
                stack.extend(forest.nodes[i].children.iter().copied());
            }
            let fc = fc.expect("flex container");
            let fonts = crate::FixedFontMetrics;
            let mut scratch = Vec::new();
            let mc = super::min_content_width(&mut forest, fc, &fonts, &mut scratch);
            let mut cache = super::PrefWidthCache::new(forest.nodes.len());
            let pc = super::preferred_width(&mut forest, fc, &fonts, &mut cache);
            (mc, pc)
        }
        let html = r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#;
        let (mc, pc) = intrinsic(
            html,
            ".c { display: flex; column-gap: 20px; } \
             .a { width: 100px; height: 10px; } .b { width: 50px; height: 10px; }",
        );
        assert!(approx(mc, 170.0), "row flex min-content {mc}");
        assert!(approx(pc, 170.0), "row flex max-content {pc}");
        let (mc, pc) = intrinsic(
            html,
            ".c { display: flex; flex-direction: column; } \
             .a { width: 100px; height: 10px; } .b { width: 50px; height: 10px; }",
        );
        assert!(approx(mc, 100.0), "column flex min-content {mc}");
        assert!(approx(pc, 100.0), "column flex max-content {pc}");
    }

    #[test]
    fn pooled_intrinsic_widths_match_fresh_on_flex_tree() {
        // P1 follow-up: the intrinsic-width walk buffers are pooled
        // (`WalkBufs`, plus the `WalkBufs`-resident flex-arm scratch
        // with its generation-stamped cache). Pooling must not change
        // any measured width, and the steady-state walk must not grow
        // the buffers.
        //
        // Two trees, each exercising every pooled path: nested flex
        // containers (the per-flex-node cache/aux hoist), content-based
        // `min-width`/`max-width` on items (the clamp's nested
        // measurements), `width: min-content` (the F7 arms), and
        // `fit-content` (min+max in one call). The second tree puts a
        // table (and a grid) inside the flex containers, so
        // buffer-state leakage between `preferred_width_mcw`'s F7 arms
        // (using `inner`) and subsequent
        // `min_content_width_pooled` flex-arm calls (resident
        // `bufs.flex`, whose `flex.bufs`/`flex.inner` are different
        // objects from `inner`) is exercised with interleaved
        // table/grid walks.
        //
        // The pooled buffers are shared across both trees: the scratch
        // built while measuring the first tree is reused for the
        // second, so leakage shows up as a width mismatch instead of
        // hiding behind fresh buffers.
        use crate::style::SpecLen;

        let mut mcw = Vec::new();
        let mut bufs = super::WalkBufs::new();
        let mut inner = super::WalkBufs::new();

        let cases = [
            (
                "flex",
                r#"<div class="outer"><div class="row"><div class="a">alpha beta gamma</div><div class="b">delta</div><div class="nested"><div class="c">epsilon zeta</div><div class="d">eta theta iota kappa</div></div></div><div class="e">lambda mu</div></div>"#,
                "body { margin: 0; } div { font-size: 10px; } \
                 .outer { display: flex; flex-direction: column; } \
                 .row { display: flex; } \
                 .a { min-width: min-content; } \
                 .b { max-width: max-content; } \
                 .nested { display: flex; width: min-content; } \
                 .c { min-width: fit-content; } \
                 .d { max-width: fit-content; } \
                 .e { width: fit-content; }",
            ),
            (
                "table-in-flex",
                r#"<div class="outer"><div class="row"><table class="t"><tr><td class="c1">alpha beta gamma</td><td class="c2">delta epsilon</td></tr></table><div class="nested"><div class="x">zeta eta</div><div class="y">theta iota kappa</div></div></div><div class="gr"><div class="g1">lambda mu</div><div class="g2">nu</div></div></div>"#,
                "body { margin: 0; } div { font-size: 10px; } td { font-size: 10px; } \
                 .outer { display: flex; flex-direction: column; } \
                 .row { display: flex; } \
                 .c1 { min-width: min-content; } \
                 .c2 { max-width: max-content; } \
                 .nested { display: flex; width: min-content; } \
                 .x { min-width: fit-content; } \
                 .y { max-width: fit-content; } \
                 .gr { display: grid; grid-template-columns: auto auto; } \
                 .g1 { min-width: min-content; }",
            ),
        ];
        for (name, html, css) in cases {
            with_forest(html, css, |forest, root| {
                let fonts = crate::FixedFontMetrics;
                // Every flex container plus the root: the pooled walks
                // visit all of them (the table/grid arms fire on the
                // descendants of these targets).
                let mut targets = vec![root];
                let mut stack = vec![root];
                while let Some(i) = stack.pop() {
                    if super::flex_kind(forest, i).is_some() {
                        targets.push(i);
                    }
                    stack.extend(forest.nodes[i].children.iter().copied());
                }
                assert!(
                    targets.len() >= 4,
                    "{name}: expected nested flex containers, found {}",
                    targets.len()
                );

                // 1. Bit-identical: pooled vs the fresh-buffer wrappers.
                for &t in &targets {
                    let pooled_min =
                        super::min_content_width_pooled(forest, t, &fonts, &mut mcw, &mut bufs);
                    let fresh_min = super::min_content_width(forest, t, &fonts, &mut mcw);
                    assert_eq!(pooled_min, fresh_min, "min-content differs at node {t}");
                    let mut cache = super::PrefWidthCache::new(forest.nodes.len());
                    let pooled_max = super::preferred_width_mcw(
                        forest, t, &fonts, &mut cache, &mut mcw, &mut bufs, &mut inner,
                    );
                    let mut cache = super::PrefWidthCache::new(forest.nodes.len());
                    let fresh_max = super::preferred_width(forest, t, &fonts, &mut cache);
                    assert_eq!(pooled_max, fresh_max, "max-content differs at node {t}");
                    // `content_based_width`: reused buffers agree with
                    // fresh buffers (buffer-reuse integrity).
                    for spec in [
                        SpecLen::MinContent,
                        SpecLen::MaxContent,
                        SpecLen::FitContent,
                    ] {
                        let mut cache = super::PrefWidthCache::new(forest.nodes.len());
                        let reused = super::content_based_width(
                            forest, t, spec, &fonts, &mut cache, &mut mcw, &mut bufs, &mut inner,
                            500.0,
                        );
                        let mut cache = super::PrefWidthCache::new(forest.nodes.len());
                        let mut fresh_bufs = super::WalkBufs::new();
                        let mut fresh_inner = super::WalkBufs::new();
                        let fresh = super::content_based_width(
                            forest,
                            t,
                            spec,
                            &fonts,
                            &mut cache,
                            &mut mcw,
                            &mut fresh_bufs,
                            &mut fresh_inner,
                            500.0,
                        );
                        assert_eq!(reused, fresh, "{spec:?} differs at node {t}");
                    }
                }

                // 2. Steady-state allocation: after warming the pooled
                // buffers, a full re-measure grows nothing.
                for &t in &targets {
                    let mut cache = super::PrefWidthCache::new(forest.nodes.len());
                    super::min_content_width_pooled(forest, t, &fonts, &mut mcw, &mut bufs);
                    super::preferred_width_mcw(
                        forest, t, &fonts, &mut cache, &mut mcw, &mut bufs, &mut inner,
                    );
                }
                let caps = (
                    bufs.order.capacity(),
                    bufs.stack.capacity(),
                    inner.order.capacity(),
                    inner.stack.capacity(),
                );
                for &t in &targets {
                    let mut cache = super::PrefWidthCache::new(forest.nodes.len());
                    super::min_content_width_pooled(forest, t, &fonts, &mut mcw, &mut bufs);
                    super::preferred_width_mcw(
                        forest, t, &fonts, &mut cache, &mut mcw, &mut bufs, &mut inner,
                    );
                }
                assert_eq!(
                    (
                        bufs.order.capacity(),
                        bufs.stack.capacity(),
                        inner.order.capacity(),
                        inner.stack.capacity()
                    ),
                    caps,
                    "pooled walk buffers reallocated on the steady-state path"
                );
            });
        }
    }

    /// Build the box tree for `html`/`css` and hand the forest plus root
    /// to `f`.
    fn with_forest<R>(
        html: &str,
        css: &str,
        f: impl FnOnce(&mut crate::boxes::Forest, usize) -> R,
    ) -> R {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let ua = bowser_css::default_ua_stylesheet();
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        f(&mut forest, root)
    }

    /// First box in preorder matching `pred`.
    fn find_box(
        forest: &crate::boxes::Forest,
        root: usize,
        mut pred: impl FnMut(usize, &crate::boxes::Forest) -> bool,
    ) -> usize {
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if pred(i, forest) {
                return i;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        panic!("no matching box");
    }

    /// Direct `(min-content, max-content)` of the box matching `pred`,
    /// through both intrinsic dispatches.
    fn measure_intrinsic(
        html: &str,
        css: &str,
        pred: impl FnMut(usize, &crate::boxes::Forest) -> bool,
    ) -> (f32, f32) {
        with_forest(html, css, |forest, root| {
            let t = find_box(forest, root, pred);
            let fonts = crate::FixedFontMetrics;
            let mut scratch = Vec::new();
            let mc = super::min_content_width(forest, t, &fonts, &mut scratch);
            let mut cache = super::PrefWidthCache::new(forest.nodes.len());
            let pc = super::preferred_width(forest, t, &fonts, &mut cache);
            (mc, pc)
        })
    }

    /// The multicol container under test: the block box with
    /// `column-count` or `column-width` specified.
    fn is_multicol_box(i: usize, forest: &crate::boxes::Forest) -> bool {
        forest.nodes[i].box_type == BoxType::Block
            && forest.nodes[i].style.as_ref().is_some_and(|s| {
                s.get_by_index(bowser_css::props::COLUMN_COUNT).is_some()
                    || s.get_by_index(bowser_css::props::COLUMN_WIDTH).is_some()
            })
    }

    /// A block box with an out-of-flow child.
    fn is_abspos_parent(i: usize, forest: &crate::boxes::Forest) -> bool {
        forest.nodes[i].box_type == BoxType::Block
            && forest.nodes[i]
                .children
                .iter()
                .any(|&c| crate::boxes::is_out_of_flow(forest, c))
    }

    #[test]
    fn abspos_child_does_not_inflate_block_intrinsic_widths() {
        // css-sizing-3 §5.2: intrinsic sizes are in-flow-only.
        // `<div style="width:fit-content"><div
        // style="position:absolute;width:500px">` collapses instead of
        // shrink-fitting to ~500px.
        let (mc, pc) = measure_intrinsic(
            r#"<div class="p"><div class="b"></div></div>"#,
            ".b { position: absolute; width: 500px; height: 10px; }",
            is_abspos_parent,
        );
        assert!(approx(mc, 0.0), "min-content {mc} != 0");
        assert!(approx(pc, 0.0), "max-content {pc} != 0");
    }

    #[test]
    fn abspos_child_does_not_inflate_shrink_to_fit() {
        // The shrink-to-fit consumer reads `preferred_width` directly,
        // so the fix flows through: the float sizes to its in-flow
        // content, not the 500px abspos child.
        let l = layout_html(
            r#"<div class="f"><div class="a"></div><div class="b"></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .a { width: 100px; height: 10px; } \
             .b { position: absolute; width: 500px; height: 10px; }",
        );
        let f = first(&l);
        assert!(
            approx(f.rect.w, 100.0),
            "float w {} != in-flow 100",
            f.rect.w
        );
    }

    #[test]
    fn abspos_inline_child_ignored_in_intrinsic_sums() {
        // The `Inline` arms of both dispatches skip out-of-flow
        // children: the 400px abspos span must not join the inline run.
        // FixedFontMetrics advances 0.6em per char at font-size 10px.
        let (mc, pc) = measure_intrinsic(
            r#"<div class="f"><span class="s">ab<span class="o"></span>cd</span></div>"#,
            ".f { float: left; font-size: 10px; } .o { position: absolute; width: 400px; }",
            |i, f| super::is_float(f, i),
        );
        assert!(approx(mc, 24.0), "min-content {mc} != 24");
        assert!(approx(pc, 24.0), "max-content {pc} != 24");
    }

    #[test]
    fn multicol_max_content_counts_columns_not_children() {
        // csswg css-multicol#4952: with a non-auto `column-count` the
        // max-content size is count × max(column-width → 0, largest
        // min-content contribution) + gaps — not the generic
        // max-of-children. L = 100 (widest fixed-width child).
        let (mc, pc) = measure_intrinsic(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { column-count: 2; column-gap: 20px; } \
             .a { width: 100px; height: 10px; } .b { width: 50px; height: 10px; }",
            is_multicol_box,
        );
        assert!(approx(mc, 220.0), "multicol min-content {mc}");
        assert!(approx(pc, 220.0), "multicol max-content {pc}");
    }

    #[test]
    fn multicol_min_content_takes_min_of_column_width() {
        // csswg css-multicol#4952: with a non-auto `column-width` the
        // min-content size is min(column-width, largest min-content
        // contribution).
        let (mc, pc) = measure_intrinsic(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { column-width: 80px; } .a { width: 100px; height: 10px; }",
            is_multicol_box,
        );
        assert!(approx(mc, 80.0), "multicol min-content {mc}");
        assert!(approx(pc, 80.0), "multicol max-content {pc}");
    }

    #[test]
    fn multicol_max_content_combines_count_and_width() {
        // `column-count: 2; column-width: 80px`: max-content is
        // 2 × max(80, L = 100) + 10; min-content is min(80, 100).
        let (mc, pc) = measure_intrinsic(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { column-count: 2; column-width: 80px; column-gap: 10px; } \
             .a { width: 100px; height: 10px; }",
            is_multicol_box,
        );
        assert!(approx(mc, 80.0), "multicol min-content {mc}");
        assert!(approx(pc, 210.0), "multicol max-content {pc}");
    }

    #[test]
    fn multicol_intrinsic_skips_abspos_children() {
        // Out-of-flow children contribute to neither multicol
        // intrinsic size (css-sizing-3 §5.2).
        let (mc, pc) = measure_intrinsic(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { column-count: 2; column-gap: 20px; } \
             .a { width: 100px; height: 10px; } \
             .b { position: absolute; width: 500px; height: 10px; }",
            is_multicol_box,
        );
        assert!(approx(mc, 220.0), "multicol min-content {mc}");
        assert!(approx(pc, 220.0), "multicol max-content {pc}");
    }

    #[test]
    fn multicol_max_content_column_width_only_multi_child() {
        // csswg css-multicol#4952: with `column-count: auto` the
        // max-content size is approximated as one column per in-flow
        // child (each child boundary is a break opportunity):
        // 5 × 200 + 4 × 10. Min-content stays min(column-width, L).
        let (mc, pc) = measure_intrinsic(
            r#"<div class="c"><div class="a"></div><div class="a"></div><div class="a"></div><div class="a"></div><div class="a"></div></div>"#,
            ".c { column-width: 200px; column-gap: 10px; } \
             .a { width: 100px; height: 10px; }",
            is_multicol_box,
        );
        assert!(approx(mc, 100.0), "multicol min-content {mc}");
        assert!(approx(pc, 1040.0), "multicol max-content {pc}");
    }

    #[test]
    fn multicol_definite_width_contributes_exactly() {
        // A definite `width` on the multicol container contributes
        // exactly, like any block box (css-sizing-3 §5.2) — content
        // narrower than it does not shrink the contribution.
        let (mc, pc) = measure_intrinsic(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { column-count: 2; column-gap: 20px; width: 300px; } \
             .a { width: 100px; height: 10px; }",
            is_multicol_box,
        );
        assert!(approx(mc, 300.0), "multicol min-content {mc}");
        assert!(approx(pc, 300.0), "multicol max-content {pc}");
    }

    #[test]
    fn multicol_min_content_aspect_ratio_transfers() {
        // R3-4: a multicol container with `aspect-ratio` and a
        // definite height transfers the height through the ratio in
        // the min-content pass too. Before the fix the min arm fell
        // through to the column-based content measure (0 here for the
        // empty child) while the container's 2/1 ratio over 50px of
        // height calls for 100.
        let (mc, _) = measure_intrinsic(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { column-count: 2; column-gap: 20px; aspect-ratio: 2/1; \
                  height: 50px; } \
             .a { height: 10px; }",
            is_multicol_box,
        );
        assert!(approx(mc, 100.0), "multicol min-content {mc}");
    }

    #[test]
    fn single_column_count_is_not_multicol() {
        // `column-count: 1` is a single column: plain block sizing, not
        // the multicol arm.
        let (mc, pc) = measure_intrinsic(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { column-count: 1; } \
             .a { width: 100px; height: 10px; } .b { width: 50px; height: 10px; }",
            |i, f| {
                f.nodes[i].box_type == BoxType::Block
                    && f.nodes[i]
                        .style
                        .as_ref()
                        .is_some_and(|s| s.get_by_index(bowser_css::props::COLUMN_COUNT).is_some())
            },
        );
        assert!(approx(mc, 100.0), "min-content {mc}");
        assert!(approx(pc, 100.0), "max-content {pc}");
    }

    #[test]
    fn float_shrink_to_fit_uses_multicol_max_content() {
        // End to end: a float around a 2-column container shrink-fits
        // to the multicol max-content (2 × 100 + 20), not the generic
        // widest-child width.
        let l = layout_html(
            r#"<div class="f"><div class="c"><div class="a"></div><div class="b"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .c { column-count: 2; column-gap: 20px; } \
             .a { width: 100px; height: 10px; } .b { width: 50px; height: 10px; }",
        );
        let f = first(&l);
        assert!(approx(f.rect.w, 220.0), "float w {} != 220", f.rect.w);
    }

    #[test]
    fn flex_wrap_row_min_content_is_widest_item() {
        // C3: a wrapping row container's min-content is the widest
        // single item's min-content — lines may break between every
        // pair, so the other items and the column gaps contribute
        // nothing. Before the fix the min-content arm fed the items
        // into the single-row sum (~300 + gaps here). Max-content is
        // unchanged: against an indefinite base no line breaks, so it
        // stays the single-row sum plus gaps.
        let html = r#"<div class="c"><div class="a"></div><div class="a"></div><div class="a"></div></div>"#;
        let items = ".a { width: 100px; height: 10px; }";
        let (mc, pc) = measure_flex_intrinsic(
            html,
            &format!(".c {{ display: flex; flex-wrap: wrap; column-gap: 20px; }} {items}"),
        );
        assert!(approx(mc, 100.0), "wrap row min-content {mc} != 100");
        assert!(approx(pc, 340.0), "wrap row max-content {pc} != 340");
        // `wrap-reverse` breaks lines the same way for sizing.
        let (mc, _) = measure_flex_intrinsic(
            html,
            &format!(".c {{ display: flex; flex-wrap: wrap-reverse; column-gap: 20px; }} {items}"),
        );
        assert!(approx(mc, 100.0), "wrap-reverse row min-content {mc}");
        // Nowrap control: min-content stays the single-row sum.
        let (mc, pc) = measure_flex_intrinsic(
            html,
            &format!(".c {{ display: flex; column-gap: 20px; }} {items}"),
        );
        assert!(approx(mc, 340.0), "nowrap row min-content {mc} != 340");
        assert!(approx(pc, 340.0), "nowrap row max-content {pc} != 340");
    }

    #[test]
    fn flex_intrinsic_basis_and_minmax_clamp_items() {
        // C4: the intrinsic arms used to take each item's raw
        // max/min-content straight as its contribution, ignoring
        // `flex-basis` and the min/max clamp the real engine applies
        // in `flex_build_items` + `clamp_flex_main`. Observed through
        // a shrink-to-fit float parent (max-content) and by direct
        // measurement (both arms).
        let html = r#"<div class="c"><div class="a"></div></div>"#;
        // Undercount: basis 200px over 50px content contributes 200.
        let (mc, pc) = measure_flex_intrinsic(
            html,
            ".c { display: flex; } .a { width: 50px; height: 10px; flex-basis: 200px; }",
        );
        assert!(approx(pc, 200.0), "basis-200 max-content {pc}");
        assert!(approx(mc, 200.0), "basis-200 min-content {mc}");
        let l = layout_html(
            r#"<div class="f"><div class="c"><div class="a"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .c { display: flex; } .a { width: 50px; height: 10px; flex-basis: 200px; }",
        );
        assert!(
            approx(first(&l).rect.w, 200.0),
            "float w {} != basis 200",
            first(&l).rect.w
        );
        // Overcount: basis 30px under 100px content contributes the
        // clamped hypothetical 30 to max-content.
        let (_, pc) = measure_flex_intrinsic(
            html,
            ".c { display: flex; } .a { width: 100px; height: 10px; flex-basis: 30px; }",
        );
        assert!(approx(pc, 30.0), "basis-30 max-content {pc}");
        let l = layout_html(
            r#"<div class="f"><div class="c"><div class="a"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .c { display: flex; } .a { width: 100px; height: 10px; flex-basis: 30px; }",
        );
        assert!(
            approx(first(&l).rect.w, 30.0),
            "float w {} != basis 30",
            first(&l).rect.w
        );
        // Percentage basis is cyclic against the indefinite intrinsic
        // base and behaves as content (100px), as in `basis_or`.
        let (_, pc) = measure_flex_intrinsic(
            html,
            ".c { display: flex; } .a { width: 100px; height: 10px; flex-basis: 50%; }",
        );
        assert!(approx(pc, 100.0), "pct-basis max-content {pc}");
    }

    #[test]
    fn flex_intrinsic_auto_width_item_minmax_clamp() {
        // C4 (clamp arm): an auto-width item's contribution is
        // clamped by its own min/max width, which neither intrinsic
        // arm applied before. The item's content is a 50px (resp.
        // 100px) inner box.
        let html = r#"<div class="c"><div class="a"><div class="i"></div></div></div>"#;
        let (mc, pc) = measure_flex_intrinsic(
            html,
            ".c { display: flex; } .a { min-width: 200px; } .i { width: 50px; height: 10px; }",
        );
        assert!(approx(pc, 200.0), "min-width max-content {pc}");
        assert!(approx(mc, 200.0), "min-width min-content {mc}");
        let l = layout_html(
            r#"<div class="f"><div class="c"><div class="a"><div class="i"></div></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .c { display: flex; } .a { min-width: 200px; } .i { width: 50px; height: 10px; }",
        );
        assert!(
            approx(first(&l).rect.w, 200.0),
            "float w {} != min-width 200",
            first(&l).rect.w
        );
        // Max-width caps the max-content contribution at 30; the
        // min-content pass keeps the automatic minimum (the item's
        // 100px min-content, `min-width` being auto), as the engine's
        // floor in `flex_lengths` does.
        let (mc, pc) = measure_flex_intrinsic(
            html,
            ".c { display: flex; } .a { max-width: 30px; } .i { width: 100px; height: 10px; }",
        );
        assert!(approx(pc, 30.0), "max-width max-content {pc}");
        assert!(approx(mc, 100.0), "max-width min-content {mc}");
    }

    #[test]
    fn flex_intrinsic_text_run_is_one_item() {
        // C9: `flex_intrinsic_width` iterated the raw forest
        // children because it cannot call `wrap_flex_text_runs`
        // (which needs `&mut`), so a run like `aa <b>bb</b>` measured
        // as two items when a parent measured the container before
        // `layout_flex` ever ran: a phantom column gap at the seam in
        // max-content, and the sum of the fragments' longest words in
        // min-content. The run is one anonymous item: max-content is
        // the sum of its parts with no seam gap, min-content the
        // widest word across the run. At 10px the fixed metrics give
        // 6px per char: parts are 18 ("aa ") and 12 ("bb").
        let (mc, pc) = measure_flex_intrinsic(
            r#"<div class="c">aa <b>bb</b></div>"#,
            ".c { display: flex; column-gap: 20px; font-size: 10px; }",
        );
        assert!(approx(pc, 30.0), "run max-content {pc} != 30");
        assert!(approx(mc, 12.0), "run min-content {mc} != 12");
        // A block item after the run takes exactly one gap, not one
        // per fragment seam: 30 + 50 + 20.
        let (mc, pc) = measure_flex_intrinsic(
            r#"<div class="c">aa <b>bb</b><div class="x"></div></div>"#,
            ".c { display: flex; column-gap: 20px; font-size: 10px; } \
             .x { width: 50px; height: 10px; }",
        );
        assert!(approx(pc, 100.0), "run+block max-content {pc} != 100");
        assert!(approx(mc, 82.0), "run+block min-content {mc} != 82");
    }

    #[test]
    fn flex_intrinsic_column_wrap_with_definite_height() {
        // C9 (column arm): a column-wrap container with a definite
        // height breaks items into columns, so its intrinsic width
        // is the columns' widths plus cross gaps — not the widest
        // item the old early return reported. Three 30px-tall items
        // in a 60px-tall container pack two per column: two columns
        // of 100px plus one 10px column gap.
        let html = r#"<div class="c"><div class="a"></div><div class="a"></div><div class="a"></div></div>"#;
        let items = ".a { width: 100px; height: 30px; }";
        let (mc, pc) = measure_flex_intrinsic(
            html,
            &format!(
                ".c {{ display: flex; flex-direction: column; flex-wrap: wrap; \
                 height: 60px; column-gap: 10px; }} {items}"
            ),
        );
        assert!(approx(pc, 210.0), "column-wrap max-content {pc} != 210");
        assert!(approx(mc, 210.0), "column-wrap min-content {mc} != 210");
        // Controls: nowrap stays one column; wrap without a definite
        // height cannot break, so it also stays one column.
        let (mc, pc) = measure_flex_intrinsic(
            html,
            &format!(
                ".c {{ display: flex; flex-direction: column; height: 60px; \
                 column-gap: 10px; }} {items}"
            ),
        );
        assert!(approx(pc, 100.0), "column nowrap max-content {pc}");
        assert!(approx(mc, 100.0), "column nowrap min-content {mc}");
        let (_, pc) = measure_flex_intrinsic(
            html,
            &format!(
                ".c {{ display: flex; flex-direction: column; flex-wrap: wrap; \
                 column-gap: 10px; }} {items}"
            ),
        );
        assert!(approx(pc, 100.0), "column wrap indefinite max {pc}");
    }

    #[test]
    fn flex_container_positions_in_block_flow() {
        let l = layout_html(
            r#"<div class="top"></div><div class="c"><div class="a"></div></div>"#,
            ".top { height: 30px; } \
             .c { display: flex; width: 400px; } .a { flex-basis: 100px; }",
        );
        let c = &body(&l).children[1];
        assert!(approx(c.rect.y, 30.0));
        assert!(approx(c.rect.w, 400.0));
        assert!(approx(c.children[0].rect.w, 100.0));
    }

    #[test]
    fn flex_item_margins_do_not_collapse() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; flex-direction: column; } \
             .a { height: 50px; margin-bottom: 10px; } \
             .b { height: 70px; margin-top: 20px; }",
        );
        let c = first(&l);
        let b = &c.children[1];
        // Adjacent flex item margins add (10 + 20), they never collapse.
        assert!(approx(b.rect.y, c.rect.y + 50.0 + 10.0 + 20.0));
        assert!(approx(c.rect.h, 50.0 + 10.0 + 20.0 + 70.0));
    }

    #[test]
    fn row_grow_redistributes_around_max_width() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; } \
             .a { flex-basis: 100px; flex-grow: 1; max-width: 100px; } \
             .b { flex-basis: 100px; flex-grow: 1; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        let b = &c.children[1];
        // 400px free: both items want +200, but .a clamps at max-width and
        // freezes; the leftover is redistributed to the unfrozen item,
        // whose share is recomputed from its flex base (§9.7 step 3).
        assert!(approx(a.rect.w, 100.0));
        assert!(approx(b.rect.w, 500.0));
        assert!(approx(b.rect.x, c.rect.x + 100.0));
    }

    #[test]
    fn row_shrink_redistributes_around_min_width() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div><div class="d"></div></div>"#,
            ".c { display: flex; width: 100px; } \
             .a { flex-basis: 100px; flex-shrink: 1; min-width: 60px; } \
             .b { flex-basis: 100px; flex-shrink: 1; } \
             .d { flex-basis: 100px; flex-shrink: 3; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        let b = &c.children[1];
        let d = &c.children[2];
        // -200px to remove, scaled factors 100/100/300: .d would shrink
        // to -20px, floors at 0 (min violation) and freezes; -100px
        // remains for .a/.b, and .a would shrink to 50px but floors at
        // min-width 60 and freezes; .b takes the last -60px.
        assert!(approx(a.rect.w, 60.0));
        assert!(approx(b.rect.w, 40.0));
        assert!(approx(d.rect.w, 0.0));
        assert!(approx(a.rect.w + b.rect.w + d.rect.w, 100.0));
    }

    #[test]
    fn row_grow_partial_fill_below_unit_factor_sum() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 300px; } \
             .a { flex-basis: 100px; flex-grow: 0.2; } \
             .b { flex-basis: 100px; flex-grow: 0.2; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        let b = &c.children[1];
        // Factor sum 0.4 < 1: only 100px * 0.4 = 40px of the free space is
        // filled, split evenly; the remaining 60px stays unused (§9.7).
        assert!(approx(a.rect.w, 120.0));
        assert!(approx(b.rect.w, 120.0));
    }

    #[test]
    fn row_grow_freeze_uses_total_violation_sign() {
        let l = layout_html(
            r#"<div class="c"><div class="x"></div><div class="y">aaaaaaaaaa</div></div>"#,
            ".c { display: flex; width: 300px; } \
             .x { flex-basis: 100px; flex-grow: 1; max-width: 110px; } \
             .y { flex-grow: 0.01; max-width: 50px; font-size: 10px; }",
        );
        let c = first(&l);
        let x = &c.children[0];
        let y = &c.children[1];
        // .x clamps at max-width 110 (max violation) while .y's raw target
        // falls below its automatic minimum (60px min-content wins the
        // min/max conflict). The total violation is negative, so only the
        // max violator freezes. On the next pass only .y is unfrozen, so
        // the §9.7 partial-fill rule caps its share at 1% of the free
        // space; the 60px automatic minimum floors the result.
        assert!(approx(x.rect.w, 110.0));
        assert!(approx(y.rect.w, 60.0));
    }

    #[test]
    fn row_grow_huge_factor_stays_finite() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; } \
             .a { flex-basis: 100px; flex-grow: 1e38; } \
             .b { flex-basis: 100px; flex-grow: 1; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        let b = &c.children[1];
        // The huge factor must not overflow to infinity: dividing the
        // free space by the factor total before multiplying keeps the
        // shares finite (500/100, summing to the container width).
        assert!(a.rect.w.is_finite());
        assert!(b.rect.w.is_finite());
        assert!(approx(a.rect.w, 500.0));
        assert!(approx(b.rect.w, 100.0));
    }

    #[test]
    fn row_shrink_auto_min_survives_post_flex_max_width() {
        let l = layout_html(
            r#"<div class="c"><div class="a">aaaaaaaaaa</div></div>"#,
            ".c { display: flex; width: 40px; } \
             .a { flex-basis: 100px; max-width: 50px; font-size: 10px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // The flex target floors at the 60px automatic minimum (min wins
        // over max-width), and block layout must not re-clamp it back to
        // the 50px max-width.
        assert!(approx(a.rect.w, 60.0));
    }

    #[test]
    fn row_zero_free_space_applies_auto_min() {
        let l = layout_html(
            r#"<div class="c"><div class="a">aaaaaaaaaa</div></div>"#,
            ".c { display: flex; width: 40px; } \
             .a { flex-basis: 40px; font-size: 10px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // No free space, so nothing flexes, but the 60px automatic
        // minimum size still applies to the base.
        assert!(approx(a.rect.w, 60.0));
    }

    #[test]
    fn row_shrink_floors_at_min_content() {
        let l = layout_html(
            r#"<div class="c"><div class="a">aaaaaaaaaa</div></div>"#,
            ".c { display: flex; width: 50px; } .a { font-size: 10px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // Min-content is 60px ("aaaaaaaaaa" at 6px/char); the 50px
        // container cannot shrink the item below it (automatic minimum
        // size), so the item overflows instead of collapsing.
        assert!(approx(a.rect.w, 60.0));
    }

    #[test]
    fn row_shrink_pct_width_item_floors_at_content_min_content() {
        // A percentage `width` is cyclic against the indefinite
        // intrinsic base (CSS 2.1 §10.2) and behaves as `auto` in the
        // intrinsic passes: the min-content contribution is the word
        // width (34 chars at 9.6px/char), not 50% resolved against a
        // zero base (which would contribute 50px and let the automatic
        // minimum collapse below the content).
        let l = layout_html(
            r#"<div class="c"><div class="a">wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww</div><div class="b"></div></div>"#,
            ".c { display: flex; width: 200px; } \
             .a { width: 50%; } .b { width: 150px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        assert!(approx(a.rect.w, 326.4));
    }

    #[test]
    fn row_shrink_respects_explicit_min_width() {
        let l = layout_html(
            r#"<div class="c"><div class="a">aaaaaaaaaa</div></div>"#,
            ".c { display: flex; width: 50px; } \
             .a { font-size: 10px; min-width: 0px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // An explicit `min-width: 0` opts out of the automatic minimum:
        // the item shrinks to the 50px container.
        assert!(approx(a.rect.w, 50.0));
    }

    #[test]
    fn row_rtl_flips_main_axis() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; direction: rtl; width: 600px; } \
             .a { flex-basis: 100px; } .b { flex-basis: 100px; }",
        );
        let c = first(&l);
        // Like row-reverse: main-start is the right edge.
        assert!(approx(c.children[0].rect.x, c.rect.x + 500.0));
        assert!(approx(c.children[1].rect.x, c.rect.x + 400.0));
    }

    #[test]
    fn column_rtl_flex_start_packs_right() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; flex-direction: column; direction: rtl; \
             width: 600px; align-items: flex-start; } \
             .a { width: 100px; height: 40px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // flexbox §5.1: the column cross axis is horizontal, so
        // `direction: rtl` moves cross-start to the right edge and
        // `flex-start` packs there.
        assert!(approx(a.rect.x, c.rect.x + 500.0));
    }

    #[test]
    fn column_rtl_flex_end_packs_left() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; flex-direction: column; direction: rtl; \
             width: 600px; align-items: flex-end; } \
             .a { width: 100px; height: 40px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // Cross-end is the left edge under `direction: rtl`.
        assert!(approx(a.rect.x, c.rect.x));
    }

    #[test]
    fn column_rtl_cross_auto_margin_absorbs_at_cross_start() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; flex-direction: column; direction: rtl; \
             width: 600px; } \
             .a { width: 100px; height: 40px; margin-right: auto; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // `margin-right: auto` is the cross-start margin under
        // `direction: rtl`: it absorbs all 500px of free space and
        // pushes the item to the left (cross-end) edge.
        assert!(approx(a.rect.x, c.rect.x));
    }

    #[test]
    fn column_ltr_flex_start_still_packs_left() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; flex-direction: column; \
             width: 600px; align-items: flex-start; } \
             .a { width: 100px; height: 40px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // No `direction: rtl`: cross-start stays the left edge.
        assert!(approx(a.rect.x, c.rect.x));
    }

    #[test]
    fn row_auto_margins_absorb_free_space() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; width: 600px; } \
             .a { flex-basis: 100px; margin-left: auto; }",
        );
        let c = first(&l);
        // The 500px of free space goes to the auto margin, not to
        // justify-content (which would left-align the item).
        assert!(approx(c.children[0].rect.x, c.rect.x + 500.0));
    }

    #[test]
    fn row_auto_margins_beat_justify_content() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; width: 600px; justify-content: flex-end; } \
             .a { flex-basis: 100px; margin-left: auto; margin-right: auto; }",
        );
        let c = first(&l);
        // Auto margins absorb the free space first, so justify-content:
        // flex-end has nothing left to distribute: the item centers.
        assert!(approx(c.children[0].rect.x, c.rect.x + 250.0));
    }

    #[test]
    fn row_cross_auto_margins_center_item() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; width: 600px; height: 200px; } \
             .a { flex-basis: 100px; height: 40px; margin: auto; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // `margin: auto` absorbs free space on both axes (flexbox §8.3):
        // the item centers horizontally (main axis) and vertically
        // (cross axis).
        assert!(approx(a.rect.x, c.rect.x + 250.0));
        assert!(approx(a.rect.y, c.rect.y + 80.0));
    }

    #[test]
    fn row_cross_auto_margin_top_pushes_to_cross_end() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; width: 600px; height: 200px; } \
             .a { flex-basis: 100px; height: 40px; margin-top: auto; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // `margin-top: auto` absorbs all 160px of cross-axis free space,
        // pushing the item to the cross end.
        assert!(approx(a.rect.y, c.rect.y + 160.0));
    }

    #[test]
    fn column_cross_auto_margins_center_item() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; flex-direction: column; width: 600px; height: 200px; } \
             .a { height: 40px; width: 100px; margin: auto; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // `margin: auto` centers on the main (vertical) and cross
        // (horizontal) axes.
        assert!(approx(a.rect.y, c.rect.y + 80.0));
        assert!(approx(a.rect.x, c.rect.x + 250.0));
    }

    #[test]
    fn column_cross_auto_margin_left_pushes_to_cross_end() {
        // Column container: `margin-left: auto` is the cross-start auto
        // margin; it absorbs the free space and pushes the item right.
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            ".c { display: flex; flex-direction: column; width: 600px; height: 200px; } \
             .a { height: 40px; width: 100px; margin-left: auto; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        assert!(approx(a.rect.x, c.rect.x + 500.0));
    }

    #[test]
    fn column_block_item_uses_flexed_main_size() {
        let l = layout_html(
            r#"<div class="c"><div class="t">hi</div></div>"#,
            ".c { display: flex; flex-direction: column; height: 200px; } \
             .t { flex-grow: 1; }",
        );
        let c = first(&l);
        let t = &c.children[0];
        // Natural height is 19.2px ("hi" at 16px); flex-grow: 1 stretches
        // the item's used main size to the full 200px.
        assert!(approx(t.rect.h, 200.0));
        assert!(approx(c.rect.h, 200.0));
    }

    #[test]
    fn anonymous_flex_item_ignores_inner_flex_props() {
        // `flex-grow` on an inline child has no effect: the child is not
        // a flex item, the anonymous wrapper is (flexbox §4), and it
        // carries the initial `flex-grow: 0`.
        let l = layout_html(
            r#"<div class="c"><span class="t">hi</span></div>"#,
            ".c { display: flex; flex-direction: column; height: 200px; } \
             .t { flex-grow: 1; }",
        );
        let c = first(&l);
        let w = &c.children[0];
        assert_eq!(w.box_type, BoxType::AnonymousBlock);
        // Natural height only; the inner span's flex-grow does not apply.
        assert!(approx(w.rect.h, 19.2));
    }

    #[test]
    fn order_reorders_flex_items() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div><div class="d"></div></div>"#,
            ".c { display: flex; width: 300px; } \
             .a { width: 100px; order: 2; } .b { width: 100px; order: 1; } \
             .d { width: 100px; }",
        );
        let c = first(&l);
        // Tree order stays document order (.a, .b, .d); visual order
        // follows `order`: .d (0), .b (1), .a (2).
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        assert!(approx(d.rect.x, c.rect.x));
        assert!(approx(b.rect.x, c.rect.x + 100.0));
        assert!(approx(a.rect.x, c.rect.x + 200.0));
    }

    #[test]
    fn row_shrink_floors_at_min_content_two_words() {
        let l = layout_html(
            r#"<div class="c"><div class="a">aaaaaaaaaa bbbbbbbbbb</div></div>"#,
            ".c { display: flex; width: 80px; } .a { font-size: 10px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // The automatic minimum is the min-content size (longest word,
        // 60px), not the max-content advance (126px): the item shrinks to
        // the 80px container instead of overflowing it.
        assert!(approx(a.rect.w, 80.0));
    }

    #[test]
    fn lower_float_narrows_flex_item_area() {
        // A preceding float whose margin box starts below the flex
        // container's top (here pushed down by `clear`) still overlaps
        // the container's body: the item area narrows past it instead of
        // letting items overlap the float (CSS 2.1 §9.5).
        let l = layout_html(
            r#"<div class="f1"></div><div class="f2"></div><div class="c"><div class="a"></div></div>"#,
            ".f1 { float: left; width: 400px; height: 60px; } \
             .f2 { float: left; width: 700px; height: 60px; clear: left; } \
             .c { display: flex; height: 200px; } \
             .a { flex-basis: 100px; height: 50px; }",
        );
        let f2 = &body(&l).children[1];
        let a = &body(&l).children[2].children[0];
        let overlaps = a.rect.x < f2.rect.x + f2.rect.w
            && f2.rect.x < a.rect.x + a.rect.w
            && a.rect.y < f2.rect.y + f2.rect.h
            && f2.rect.y < a.rect.y + a.rect.h;
        assert!(
            !overlaps,
            "flex item {:?} overlaps lower float {:?}",
            a.rect, f2.rect
        );
        // The item sits past the lower float's right edge.
        assert!(approx(a.rect.x, 700.0), "a.x = {}", a.rect.x);
    }

    #[test]
    fn row_shrink_text_item_floors_at_min_content_two_words() {
        let l = layout_html(
            r#"<div class="c">aaaaaaaaaa bbbbbbbbbb</div>"#,
            ".c { display: flex; width: 80px; font-size: 10px; }",
        );
        let c = first(&l);
        let a = &c.children[0];
        // Same floor for a bare-text flex item: longest word 60px, so the
        // item fits the 80px container (and wraps to two lines).
        assert!(approx(a.rect.w, 80.0));
        assert!(approx(a.rect.h, 24.0));
    }

    #[test]
    fn row_rtl_row_reverse_cancels() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; direction: rtl; flex-direction: row-reverse; width: 600px; } \
             .a { flex-basis: 100px; } .b { flex-basis: 100px; }",
        );
        let c = first(&l);
        // rtl flips the row axis and row-reverse flips it back: items lay
        // out in normal (ltr row) order.
        assert!(approx(c.children[0].rect.x, c.rect.x));
        assert!(approx(c.children[1].rect.x, c.rect.x + 100.0));
    }

    #[test]
    fn flex_block_item_normal_span_wraps_inside_pre_wrap() {
        let l = layout_html(
            r#"<div class="c"><div class="d"><span class="s">aaa bbb ccc ddd eee fff ggg hhh</span></div></div>"#,
            ".c { display: flex; width: 100px; } \
             .d { white-space: pre-wrap; font-size: 10px; } .s { white-space: normal; }",
        );
        let c = first(&l);
        let d = &c.children[0];
        // The `normal` span's spaces are break opportunities even though
        // the item is `pre-wrap`: two 12px lines.
        assert!(approx(d.rect.h, 24.0));
    }

    #[test]
    fn flex_text_item_normal_span_wraps_inside_pre_wrap() {
        let l = layout_html(
            r#"<div class="c"><span class="t">aaa bbb <em class="e">ccc ddd eee fff ggg hhh</em></span></div>"#,
            ".c { display: flex; width: 100px; } \
             .t { white-space: pre-wrap; font-size: 10px; } .e { white-space: normal; }",
        );
        let c = first(&l);
        // The run becomes one anonymous block item; its strut inherits
        // the container's 16px font (19.2px line height), so two lines
        // are 38.4px tall even though the text is 10px.
        let t = &c.children[0];
        assert_eq!(t.box_type, BoxType::AnonymousBlock);
        // The `normal` run wraps at spaces; the `pre-wrap` run ("aaa bbb ",
        // 48px) fills the first line alone: two lines.
        assert!(approx(t.rect.h, 38.4));
    }

    #[test]
    fn flex_text_item_pre_wrap_span_keeps_spaces() {
        let l = layout_html(
            "<div class=\"c\"><span class=\"t\"><span class=\"s\">aaa bbb\nccc</span></span></div>",
            ".c { display: flex; width: 100px; } \
             .t { font-size: 10px; } .s { white-space: pre-wrap; }",
        );
        let c = first(&l);
        // One anonymous block item (strut: inherited 16px/19.2px).
        let t = &c.children[0];
        assert_eq!(t.box_type, BoxType::AnonymousBlock);
        // Reverse direction: the `pre-wrap` span keeps its spaces and
        // breaks only at its forced break: two lines.
        assert!(approx(t.rect.h, 38.4));
    }

    #[test]
    fn flex_shorthand_two_children_split_row() {
        // Regression: `flex: 1` expands to `1 1 0%`, so two children of a
        // 600px row split the free space 300/300 (unexpanded, the
        // declaration was dropped and both items sized 0).
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; } \
             .a { flex: 1; } .b { flex: 1; }",
        );
        let c = first(&l);
        assert!(approx(c.children[0].rect.w, 300.0));
        assert!(approx(c.children[1].rect.w, 300.0));
        assert!(approx(c.children[1].rect.x, c.rect.x + 300.0));
    }

    #[test]
    fn flex_shorthand_none_and_basis_forms() {
        // `flex: none` -> `0 0 auto`: the item keeps its content width
        // (empty -> 0) while the `flex: 1` sibling takes the rest.
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; } \
             .a { flex: none; } .b { flex: 1; }",
        );
        let c = first(&l);
        assert!(approx(c.children[0].rect.w, 0.0));
        assert!(approx(c.children[1].rect.w, 600.0));
        // Two-value `flex: 2 1 0%` vs `flex: 1` splits 400/200.
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; } \
             .a { flex: 2 1; } .b { flex: 1; }",
        );
        let c = first(&l);
        assert!(approx(c.children[0].rect.w, 400.0));
        assert!(approx(c.children[1].rect.w, 200.0));
    }

    #[test]
    fn row_gap_nowrap_separates_items() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div><div class="d"></div></div>"#,
            ".c { display: flex; width: 600px; column-gap: 20px; } \
             .a { flex-basis: 100px; height: 10px; } \
             .b { flex-basis: 100px; height: 10px; } \
             .d { flex-basis: 100px; height: 10px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // One 20px stride between each adjacent pair.
        assert!(approx(a.rect.x, c.rect.x));
        assert!(approx(b.rect.x, c.rect.x + 120.0));
        assert!(approx(d.rect.x, c.rect.x + 240.0));
    }

    #[test]
    fn row_gap_shorthand_sets_both_axes() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; gap: 20px; } \
             .a { flex-basis: 100px; height: 10px; } \
             .b { flex-basis: 100px; height: 10px; }",
        );
        let c = first(&l);
        // `gap` expands to `row-gap` + `column-gap`; the main-axis stride
        // is the 20px `column-gap`.
        assert!(approx(c.children[1].rect.x, c.rect.x + 120.0));
    }

    #[test]
    fn row_gap_two_value_shorthand_splits_axes() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; gap: 5px 20px; } \
             .a { flex-basis: 100px; height: 10px; } \
             .b { flex-basis: 100px; height: 10px; }",
        );
        let c = first(&l);
        // `gap: <row> <column>`: the row-axis stride is the 20px column
        // gap, not the 5px row gap.
        assert!(approx(c.children[1].rect.x, c.rect.x + 120.0));
    }

    #[test]
    fn row_gap_subtracted_before_flex_grow() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 600px; column-gap: 20px; } \
             .a { flex-basis: 100px; flex-grow: 1; } \
             .b { flex-basis: 100px; flex-grow: 1; }",
        );
        let c = first(&l);
        let (a, b) = (&c.children[0], &c.children[1]);
        // Free space is 600 - 200 - 20 = 380, split evenly: each item is
        // 290px wide and the 20px gap sits between them.
        assert!(approx(a.rect.w, 290.0));
        assert!(approx(b.rect.w, 290.0));
        assert!(approx(b.rect.x, c.rect.x + 310.0));
    }

    #[test]
    fn row_gap_nowrap_shrink_counts_gap() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div><div class="d"></div></div>"#,
            ".c { display: flex; width: 300px; column-gap: 20px; } \
             .a { flex-basis: 120px; height: 10px; } \
             .b { flex-basis: 120px; height: 10px; } \
             .d { flex-basis: 120px; height: 10px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // 360px of items plus two 20px gaps overflow by 100px, shrunk
        // proportionally: each item is 86.67px.
        assert!(approx(a.rect.w, 260.0 / 3.0));
        assert!(approx(b.rect.x, c.rect.x + 260.0 / 3.0 + 20.0));
        assert!(approx(d.rect.x, c.rect.x + 2.0 * (260.0 / 3.0 + 20.0)));
    }

    #[test]
    fn row_gap_space_between_adds_to_gap() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 300px; column-gap: 20px; justify-content: space-between; } \
             .a { flex-basis: 100px; height: 10px; } \
             .b { flex-basis: 100px; height: 10px; }",
        );
        let c = first(&l);
        // Free space (300 - 200 - 20 = 80) goes between the items on top
        // of the fixed 20px gap.
        assert!(approx(c.children[0].rect.x, c.rect.x));
        assert!(approx(c.children[1].rect.x, c.rect.x + 200.0));
    }

    #[test]
    fn row_gap_nowrap_reverse_stays_symmetric() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; flex-direction: row-reverse; width: 300px; column-gap: 20px; } \
             .a { flex-basis: 100px; height: 10px; } \
             .b { flex-basis: 100px; height: 10px; }",
        );
        let c = first(&l);
        // Reversed: the first item sits at the right edge, the 20px gap
        // stride mirrors with the positions.
        assert!(approx(c.children[0].rect.x, c.rect.x + 200.0));
        assert!(approx(c.children[1].rect.x, c.rect.x + 80.0));
    }

    #[test]
    fn column_gap_nowrap_stacks_with_strides() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div><div class="d"></div></div>"#,
            ".c { display: flex; flex-direction: column; width: 200px; row-gap: 15px; } \
             .a { height: 40px; } .b { height: 40px; } .d { height: 40px; }",
        );
        let c = first(&l);
        let (a, b, d) = (&c.children[0], &c.children[1], &c.children[2]);
        // The column main gap is `row-gap`: 15px between items, and the
        // auto container height includes both strides.
        assert!(approx(a.rect.y, c.rect.y));
        assert!(approx(b.rect.y, c.rect.y + 55.0));
        assert!(approx(d.rect.y, c.rect.y + 110.0));
        assert!(approx(c.rect.h, 150.0));
    }

    #[test]
    fn column_gap_nowrap_justify_center() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; flex-direction: column; width: 200px; height: 200px; \
               row-gap: 20px; justify-content: center; } \
             .a { height: 40px; } .b { height: 40px; }",
        );
        let c = first(&l);
        // 100px of content (40 + 20 + 40) centers in 200px: 50px above.
        assert!(approx(c.children[0].rect.y, c.rect.y + 50.0));
        assert!(approx(c.children[1].rect.y, c.rect.y + 110.0));
    }

    #[test]
    fn row_percentage_column_gap_resolves_against_width() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; width: 300px; column-gap: 10%; } \
             .a { flex-basis: 100px; height: 10px; } \
             .b { flex-basis: 100px; height: 10px; }",
        );
        let c = first(&l);
        // 10% of the 300px content width: a 30px stride.
        assert!(approx(c.children[1].rect.x, c.rect.x + 130.0));
    }

    #[test]
    fn column_percentage_row_gap_resolves_against_definite_height() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; flex-direction: column; width: 200px; height: 200px; \
               row-gap: 10%; } \
             .a { height: 40px; } .b { height: 40px; }",
        );
        let c = first(&l);
        // 10% of the definite 200px height: a 20px stride.
        assert!(approx(c.children[1].rect.y, c.rect.y + 60.0));
    }

    #[test]
    fn column_percentage_row_gap_zero_with_auto_height() {
        let l = layout_html(
            r#"<div class="c"><div class="a"></div><div class="b"></div></div>"#,
            ".c { display: flex; flex-direction: column; width: 200px; row-gap: 10%; } \
             .a { height: 40px; } .b { height: 40px; }",
        );
        let c = first(&l);
        // The height is indefinite, so the percentage gap is zero.
        assert!(approx(c.children[1].rect.y, c.rect.y + 40.0));
        assert!(approx(c.rect.h, 80.0));
    }

    #[test]
    fn break_spaces_does_not_collapse_end_to_end() {
        // Regression: `white-space: break-spaces` must survive the cascade
        // and keep every space. Shrink-to-fit measures the uncollapsed
        // run: "a  b" is 4 chars x 6px = 24px at 10px font; collapsed it
        // would be 18px.
        let l = layout_html(
            r#"<div class="c"><span class="t">a  b</span></div>"#,
            ".t { display: inline-block; white-space: break-spaces; font-size: 10px; }",
        );
        let t = &body(&l).children[0].children[0];
        assert!(approx(t.rect.w, 24.0), "w = {}", t.rect.w);
        // And the cascade must not fall back to `normal` for the keyword.
        let l = layout_html(
            r#"<div class="c"><span class="t">a  b</span></div>"#,
            ".t { display: inline-block; white-space: normal; font-size: 10px; }",
        );
        let t = &body(&l).children[0].children[0];
        assert!(approx(t.rect.w, 18.0), "w = {}", t.rect.w);
    }

    #[test]
    fn tall_line_rebreaks_against_true_float_band() {
        // A tall inline-block stretches the line's true band beyond the
        // estimated `[y, y+line-height)`. A float beginning in the true
        // band's lower portion (here pushed below an earlier float by
        // `clear`) must narrow the line's *fragments*, not just its
        // recorded shift/width: breaking only against the estimated width
        // leaves text overlapping the float.
        let l = layout_html(
            r#"<div class="c"><div class="g"></div><div class="f"></div><div class="t">aaa aaa aaa aaa aaa aaa <span class="ib"></span> bbb bbb bbb bbb bbb bbb bbb bbb</div></div>"#,
            ".c { width: 400px; } \
             .g { float: left; width: 100px; height: 50px; } \
             .f { float: right; width: 100px; height: 100px; clear: left; } \
             .ib { display: inline-block; width: 50px; height: 100px; }",
        );
        let c = &body(&l).children[0];
        let f = &c.children[1];
        // The right float clears the 50px left float: x in [300,400),
        // y in [50,150).
        assert!(approx(f.rect.x, 300.0), "f.x = {}", f.rect.x);
        assert!(approx(f.rect.y, 50.0), "f.y = {}", f.rect.y);
        let fr = &f.rect;
        let overlaps = |r: &crate::Rect| {
            r.x < fr.x + fr.w && fr.x < r.x + r.w && r.y < fr.y + fr.h && fr.y < r.y + r.h
        };
        // No laid-out text fragment and no inline-level box may intersect
        // the float.
        fn check(b: &LayoutBox, fr: &crate::Rect, overlaps: &dyn Fn(&crate::Rect) -> bool) {
            for tf in &b.fragments {
                assert!(
                    !overlaps(&tf.rect),
                    "text fragment {:?} overlaps float {:?}",
                    tf.rect,
                    fr
                );
            }
            if b.box_type == BoxType::InlineBlock {
                assert!(
                    !overlaps(&b.rect),
                    "inline-block {:?} overlaps float {:?}",
                    b.rect,
                    fr
                );
            }
            for ch in &b.children {
                check(ch, fr, overlaps);
            }
        }
        check(&c.children[2], fr, &overlaps);
    }

    #[test]
    fn intrinsic_content_fallback_clamped_by_minmax() {
        // Task 9: content-based intrinsic fallbacks are clamped by the
        // box's own min-width/max-width, like flex items (and like
        // `definite_specified_content_width` /
        // `intrinsic_content_width` already clamp).
        // Max-width caps a shrink-to-fit float at 30, not the 100px
        // content width.
        let l = layout_html(
            r#"<div class="f"><div class="a"><div class="i"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .a { max-width: 30px; } .i { width: 100px; height: 10px; }",
        );
        assert!(
            approx(first(&l).rect.w, 30.0),
            "float w {} != max-width 30",
            first(&l).rect.w
        );
        // Min-width floors it at 200 despite the 50px content.
        let l = layout_html(
            r#"<div class="f"><div class="a"><div class="i"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .a { min-width: 200px; } .i { width: 50px; height: 10px; }",
        );
        assert!(
            approx(first(&l).rect.w, 200.0),
            "float w {} != min-width 200",
            first(&l).rect.w
        );
    }

    #[test]
    fn aspect_ratio_reverse_transfer_intrinsic_width() {
        // Task 10: a non-replaced box with `aspect-ratio` and a
        // definite height contributes the ratio-derived width
        // (`height * ratio`) to its intrinsic sizes, mirroring the
        // engine's shrink-to-fit reverse transfer.
        let leaf_block = |i: usize, f: &crate::boxes::Forest| {
            f.nodes[i].box_type == BoxType::Block
                && !f.nodes[i]
                    .children
                    .iter()
                    .any(|&c| f.nodes[c].box_type == BoxType::Block)
        };
        let (mc, pc) = measure_intrinsic(
            r#"<div class="a"></div>"#,
            ".a { aspect-ratio: 2 / 1; height: 50px; }",
            leaf_block,
        );
        assert!(approx(mc, 100.0), "min-content {mc} != 100");
        assert!(approx(pc, 100.0), "max-content {pc} != 100");
        // The derived width is still bounded by min/max-width.
        let (_, pc) = measure_intrinsic(
            r#"<div class="a"></div>"#,
            ".a { aspect-ratio: 2 / 1; height: 50px; max-width: 60px; }",
            leaf_block,
        );
        assert!(approx(pc, 60.0), "max-width max-content {pc} != 60");
        // And the shrink-to-fit consumer agrees.
        let l = layout_html(
            r#"<div class="f"><div class="a"></div></div>"#,
            "body { margin: 0; } .f { float: left; } \
             .a { aspect-ratio: 2 / 1; height: 50px; }",
        );
        assert!(
            approx(first(&l).rect.w, 100.0),
            "float w {} != ratio 100",
            first(&l).rect.w
        );
    }

    #[test]
    fn flex_intrinsic_basis_border_box_strips_extras() {
        // Task 11: a definite `flex-basis` resolves in the item's
        // box-sizing terms (css-flexbox-1 §9.2) — under `border-box`
        // the 200px basis is the outer size, so the intrinsic
        // contribution is 200, not 200 + 20 padding.
        let html = r#"<div class="c"><div class="a"></div></div>"#;
        let (mc, pc) = measure_flex_intrinsic(
            html,
            ".c { display: flex; } \
             .a { flex-basis: 200px; box-sizing: border-box; \
                  padding-left: 10px; padding-right: 10px; height: 10px; }",
        );
        assert!(approx(pc, 200.0), "border-box basis max-content {pc}");
        assert!(approx(mc, 200.0), "border-box basis min-content {mc}");
        let l = layout_html(
            r#"<div class="f"><div class="c"><div class="a"></div></div></div>"#,
            "body { margin: 0; } .f { float: left; } .c { display: flex; } \
             .a { flex-basis: 200px; box-sizing: border-box; \
                  padding-left: 10px; padding-right: 10px; height: 10px; }",
        );
        assert!(
            approx(first(&l).rect.w, 200.0),
            "float w {} != basis 200",
            first(&l).rect.w
        );
    }

    #[test]
    fn flex_column_wrap_basis_border_box_strips_extras() {
        // Task 11 (column): the column-wrap simulation's main size is
        // the *inner* size — a 40px border-box basis with 20px of
        // vertical padding packs as 40px outer (20 inner + 20 pad),
        // so two items share a 100px column; without the strip each
        // packs as 60px and every item takes its own column.
        let html = r#"<div class="c"><div class="a"></div><div class="a"></div><div class="a"></div><div class="a"></div></div>"#;
        let (_, pc) = measure_flex_intrinsic(
            html,
            ".c { display: flex; flex-direction: column; flex-wrap: wrap; height: 100px; } \
             .a { width: 100px; flex-basis: 40px; box-sizing: border-box; \
                  padding-top: 10px; padding-bottom: 10px; }",
        );
        assert!(approx(pc, 200.0), "column-wrap max-content {pc} != 200");
    }

    #[test]
    fn flex_column_wrap_unmeasurable_mains_simulate() {
        // Task 12: items whose main size needs a measuring layout
        // (auto basis, auto height, no intrinsic size — e.g. text)
        // no longer abort the column-wrap simulation to the
        // widest-item width. In the max-content pass each text item
        // is one line (line-height 20px), so five 18px-wide items in
        // a 50px-tall container pack 2/2/1 into three columns.
        // FixedFontMetrics advances 0.6em per char at font-size 10px.
        let html = r#"<div class="c"><div class="a">aaa</div><div class="a">aaa</div><div class="a">aaa</div><div class="a">aaa</div><div class="a">aaa</div></div>"#;
        let (mc, pc) = measure_flex_intrinsic(
            html,
            ".c { display: flex; flex-direction: column; flex-wrap: wrap; height: 50px; } \
             .a { font-size: 10px; line-height: 20px; }",
        );
        assert!(approx(pc, 54.0), "column-wrap max-content {pc} != 54");
        assert!(approx(mc, 54.0), "column-wrap min-content {mc} != 54");
    }

    #[test]
    fn abspos_solved_sizes_adjust_negative_imcb() {
        // css-position-3 §4.2 (WPT
        // position-absolute-with-negative-sized-imcb.html, case 3):
        // `inset: 18px` in a 20px container makes the inset-modified
        // containing block negative-sized, so the weaker (end) inset
        // is adjusted to make it zero-sized — and the auto width
        // solves against the ADJUSTED insets:
        // 20 - 18 - 2 - (-10) - (-10) = 20, not 20 - 18 - 18 + 20 = 4.
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            "body { margin: 0; } \
             .c { position: relative; width: 20px; height: 20px; } \
             .a { position: absolute; inset: 18px; margin: -10px; }",
        );
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.w, 20.0), "w={}", a.rect.w);
        assert!(approx(a.rect.h, 20.0), "h={}", a.rect.h);
        assert!(approx(a.rect.x, 8.0), "x={}", a.rect.x);
        assert!(approx(a.rect.y, 8.0), "y={}", a.rect.y);
    }

    #[test]
    fn flex_intrinsic_border_box_basis_strips_only_border_padding() {
        // R3-2: `flex-basis` resolves in the item's box-sizing terms
        // (css-flexbox-1 §9.2) but never covers the item's margins, so
        // the `border-box` strip in `flex_item_intrinsic_contribution`
        // must remove only border+padding. With `flex-basis: 200px`,
        // 10px side padding and 30px side margins the item contributes
        // 200 + 60 = 260. The old code stripped the margins too and
        // reported 200, so a shrink-to-fit container under-measured
        // and the item overflowed it.
        let html = r#"<div class="c"><div class="a"></div></div>"#;
        let css = ".c { display: flex; } \
             .a { flex-basis: 200px; box-sizing: border-box; \
                  margin: 0 30px; padding: 0 10px; }";
        let (mc, pc) = measure_flex_intrinsic(html, css);
        assert!(approx(pc, 260.0), "border-box basis max-content {pc}");
        assert!(approx(mc, 260.0), "border-box basis min-content {mc}");
        // Intrinsic/used consistency: a `max-content` container's used
        // width is exactly the intrinsic contribution above.
        // (The item's own used width is a separate known engine-side
        // `basis_or` double-count, untouched by this fix.)
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            "body { margin: 0; } .c { display: flex; width: max-content; } \
             .a { flex-basis: 200px; box-sizing: border-box; \
                  margin: 0 30px; padding: 0 10px; flex-shrink: 0; }",
        );
        assert!(approx(first(&l).rect.w, 260.0), "w={}", first(&l).rect.w);
        // PROBE: the item's own used width. A definite `flex-basis`
        // resolves in the item's box-sizing terms (css-flexbox-1
        // §9.2), so the 200px border-box basis covers the 20px of
        // horizontal padding: the used border box is exactly 200, not
        // 200 + 20.
        let a2 = &body(&l).children[0].children[0];
        assert!(approx(a2.rect.w, 200.0), "item w={}", a2.rect.w);
    }

    #[test]
    fn flex_column_basis_border_box_strips_main_bp() {
        // Column mirror of the row probe above: a definite vertical
        // `flex-basis` resolves in the item's box-sizing terms
        // (css-flexbox-1 §9.2), so the 200px border-box basis covers
        // the 20px of vertical padding — the used border-box height is
        // exactly 200, not 200 + 20.
        let l = layout_html(
            r#"<div class="c"><div class="a"></div></div>"#,
            "body { margin: 0; } .c { display: flex; flex-direction: column; } \
             .a { flex-basis: 200px; box-sizing: border-box; \
                  padding: 10px 0; flex-shrink: 0; }",
        );
        let a = &body(&l).children[0].children[0];
        assert!(approx(a.rect.h, 200.0), "item h={}", a.rect.h);
    }

    #[test]
    fn flex_pct_table_base_is_column_max_sum() {
        // WPT `css/css-tables/tentative/table-minmax.html` (flex
        // cases): a table with a `width:100%` column beside a 100px-max
        // column has a 160px flex base (30 + 100 + 30 spacing), not
        // its infinite percentage-floor preferred width saturated to
        // `MAX_USED_SIZE`. WPT expects the table at 355 there, not
        // 360: its container is 600px *border-box* with 5px borders
        // (590px content, 390px free, +195 each on equal grow). This
        // test's container is 600px of content (400px free, +200
        // each), so the table lands at 360 and the sibling at 240 —
        // the same engine result under the test's own geometry, not
        // a 5px discrepancy in the base.
        let l = layout_html(
            r#"<div class="c"><table class="t"><tr>
             <td style="width:100%"><div style="width:30px"></div></td>
             <td><div style="width:50px"></div><div style="width:50px"></div></td>
             </tr></table><div class="s">40px</div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; } \
             .t { flex: 1 1 auto; } .s { flex: 1 1 40px; }",
        );
        let c = first(&l);
        assert!(
            approx(c.children[0].rect.w, 360.0),
            "table w={}",
            c.children[0].rect.w
        );
        assert!(
            approx(c.children[1].rect.w, 240.0),
            "sibling w={}",
            c.children[1].rect.w
        );
    }

    #[test]
    fn flex_pct_table_no_grow_stays_at_column_max_sum() {
        // Same table with `flex: 0 0 auto` stays at its 160px base
        // while the `flex: 1 1 auto` sibling absorbs the rest.
        let l = layout_html(
            r#"<div class="c"><table class="t"><tr>
             <td style="width:100%"><div style="width:30px"></div></td>
             <td><div style="width:50px"></div><div style="width:50px"></div></td>
             </tr></table><div class="s">auto</div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; } \
             .t { flex: 0 0 auto; } .s { flex: 1 1 auto; }",
        );
        let c = first(&l);
        assert!(
            approx(c.children[0].rect.w, 160.0),
            "table w={}",
            c.children[0].rect.w
        );
    }

    // w76 flex-basis saturation generality: content-keyword bases
    // must measure the keyword, not fall back to the main-size
    // property / width-derived content. At 10px type one char is 6px.
    #[test]
    fn w76_row_basis_min_content_uses_min_not_max() {
        // (a) width:auto, wide content: min-content is the longest
        // word (60), not the unwrapped line (192).
        let l = layout_html(
            r#"<div class="c"><div class="a">aaaaaaaaaa bbbbbbbbbb cccccccccc</div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             .a { font-size: 10px; flex-basis: min-content; flex-grow: 0; flex-shrink: 0; }",
        );
        let a = &first(&l).children[0];
        assert!(a.rect.w.is_finite());
        assert!(approx(a.rect.w, 60.0), "min-content item w={}", a.rect.w);
    }

    #[test]
    fn w76_row_basis_fit_content_clamps_to_avail() {
        // (b) container narrower than max-content: fit-content is
        // min(max, max(min, avail)) = min(192, max(60, 100)) = 100.
        let l = layout_html(
            r#"<div class="c"><div class="a">aaaaaaaaaa bbbbbbbbbb cccccccccc</div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 100px; } \
             .a { font-size: 10px; flex-basis: fit-content; flex-grow: 0; flex-shrink: 0; }",
        );
        let a = &first(&l).children[0];
        assert!(a.rect.w.is_finite());
        assert!(approx(a.rect.w, 100.0), "fit-content item w={}", a.rect.w);
    }

    #[test]
    fn w77fix_min_content_width_grid_basis_max_content_uses_track_max() {
        // w77fix finding 6 (fail-before probe): a grid whose own
        // `width` is `min-content`, used as a flex item with
        // `flex-basis: max-content`, must base at its track
        // max-content (two `auto` tracks over 126px lines = 252),
        // not at its min-content (two 60px longest words = 120).
        // `SpecLen::resolve` returns `None` for `MinContent`, so
        // the definite-width gate in `flex_basis_keyword_size`
        // never fired for this shape and the dispatched
        // `grid_pref_width` MinContent fast path returned the min
        // walk instead of the track max.
        let html = r#"<div class="c"><div class="g"><div class="i">aaaaaaaaaa bbbbbbbbbb</div><div class="i">aaaaaaaaaa bbbbbbbbbb</div></div></div>"#;
        let css = "body { margin: 0; } .c { display: flex; width: 600px; } \
             .g { display: grid; grid-template-columns: auto auto; width: min-content; \
                  font-size: 10px; flex-basis: max-content; flex-grow: 0; flex-shrink: 0; }";
        let l = layout_html(html, css);
        let g = &first(&l).children[0];
        assert!(
            approx(g.rect.w, 252.0),
            "min-content-width grid basis-max item w {}, want 252 (track max, not min 120)",
            g.rect.w
        );
        // The intrinsic arm agrees: with an auto-width flex
        // container, its max-content is the item's max-content
        // basis (a definite container width would floor the
        // measurement at 600 instead).
        let (_, pc) = measure_flex_intrinsic(
            html,
            "body { margin: 0; } .c { display: flex; } \
             .g { display: grid; grid-template-columns: auto auto; width: min-content; \
                  font-size: 10px; flex-basis: max-content; flex-grow: 0; flex-shrink: 0; }",
        );
        assert!(
            approx(pc, 252.0),
            "flex max-content {pc}, want 252 (not the 120 min-content floor)"
        );
    }

    #[test]
    fn w76_row_basis_max_content_ignores_width() {
        // (c) width:100px must not win over flex-basis:max-content:
        // the single 50-char word measures 300.
        let word = "a".repeat(50);
        let html = format!(r#"<div class="c"><div class="a">{word}</div></div>"#);
        let l = layout_html(
            &html,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             .a { font-size: 10px; width: 100px; flex-basis: max-content; \
                  flex-grow: 0; flex-shrink: 0; }",
        );
        let a = &first(&l).children[0];
        assert!(a.rect.w.is_finite());
        assert!(approx(a.rect.w, 300.0), "max-content item w={}", a.rect.w);
    }

    #[test]
    fn w76_row_basis_max_content_direct_pct_table_is_grid_max() {
        // (d) direct auto table, 100% column: base is the column-max
        // sum (160), not the infinite preferred width saturated to 1e7.
        let l = layout_html(
            r#"<div class="c"><table class="t"><tr>
             <td style="width:100%"><div style="width:30px"></div></td>
             <td><div style="width:50px"></div><div style="width:50px"></div></td>
             </tr></table></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; } \
             .t { flex-grow: 0; flex-shrink: 0; flex-basis: max-content; }",
        );
        let t = &first(&l).children[0];
        assert!(t.rect.w.is_finite());
        assert!(approx(t.rect.w, 160.0), "table w={}", t.rect.w);
    }

    #[test]
    fn w76_row_basis_max_content_direct_pct_table_grows() {
        // (d, grow) WPT table-minmax geometry: 600px border-box
        // container (590 content), bases 160 + 40, free 390 split
        // equally, so the table grows 160 -> 355, not to 1e7.
        // Longhands, not `flex: 1 1 max-content`: the css crate's
        // shorthand expansion only accepts auto/content bases, so
        // that shorthand is invalid here (out of this file's scope).
        let l = layout_html(
            r#"<div class="c"><table class="t"><tr>
             <td style="width:100%"><div style="width:30px"></div></td>
             <td><div style="width:50px"></div><div style="width:50px"></div></td>
             </tr></table><div class="s">40px</div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; \
             box-sizing: border-box; border: 5px solid black; } \
             table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; } \
             .t { flex-grow: 1; flex-shrink: 1; flex-basis: max-content; } .s { flex: 1 1 40px; }",
        );
        let c = first(&l);
        assert!(c.children[0].rect.w.is_finite());
        assert!(
            approx(c.children[0].rect.w, 355.0),
            "table w={}",
            c.children[0].rect.w
        );
        assert!(
            approx(c.children[1].rect.w, 235.0),
            "sibling w={}",
            c.children[1].rect.w
        );
    }

    #[test]
    fn w76_row_basis_fit_content_direct_pct_table_clamps_to_avail() {
        // (d, fit-content) the keyword clamps, it does not grid-substitute:
        // min(max, max(min, avail)) = min(inf, max(110, 600)) = 600.
        // The guard stays for the infinite-avail case; here the base is
        // finite by the clamp itself, so it must not be rewritten to 160.
        let l = layout_html(
            r#"<div class="c"><table class="t"><tr>
             <td style="width:100%"><div style="width:30px"></div></td>
             <td><div style="width:50px"></div><div style="width:50px"></div></td>
             </tr></table></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; } \
             .t { flex-grow: 0; flex-shrink: 0; flex-basis: fit-content; }",
        );
        let t = &first(&l).children[0];
        assert!(t.rect.w.is_finite());
        assert!(approx(t.rect.w, 600.0), "table w={}", t.rect.w);
    }

    #[test]
    fn w76_row_basis_max_content_fixed_pct_table_still_saturates() {
        // Fixed-table control for (d): a fixed-layout 100% table keeps
        // infinity, so its base saturates (finite, far past the
        // container) instead of being rewritten to the grid sum. The
        // container is indefinite (width:min-content, as in the WPT
        // scenario): in a definite 600px container the table's own
        // width:100% caps its used width at 600, which would mask the
        // saturation this control is meant to pin.
        let l = layout_html(
            r#"<div class="c"><table class="t"><tr><td></td></tr></table></div>"#,
            "body { margin: 0; } .c { display: flex; width: min-content; } \
             .t { table-layout: fixed; width: 100%; flex-grow: 0; flex-shrink: 0; \
                  flex-basis: max-content; border-spacing: 0; } td { width: 100px; padding: 0; }",
        );
        let t = &first(&l).children[0];
        assert!(t.rect.w.is_finite());
        assert!(t.rect.w > 800.0, "fixed table w={}", t.rect.w);
    }

    #[test]
    fn w76_row_basis_auto_wrapper_around_pct_table_is_grid_plus_extras() {
        // (e) interposed block wrapper: table grid sum 160 plus the
        // wrapper's 10px + 10px padding, so the wrapper item is 180
        // and the table itself is 160, not 1e7.
        let l = layout_html(
            r#"<div class="c"><div class="w"><table class="t"><tr>
             <td style="width:100%"><div style="width:30px"></div></td>
             <td><div style="width:50px"></div><div style="width:50px"></div></td>
             </tr></table></div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; } \
             .w { flex: 0 0 auto; padding: 0 10px; }",
        );
        let c = first(&l);
        let w = &c.children[0];
        let t = &w.children[0];
        assert!(w.rect.w.is_finite() && t.rect.w.is_finite());
        assert!(approx(w.rect.w, 180.0), "wrapper w={}", w.rect.w);
        assert!(approx(t.rect.w, 160.0), "table w={}", t.rect.w);
    }

    #[test]
    fn w76_row_basis_auto_double_wrapper_around_pct_table() {
        // (e, recursion) two single block wrappers: 160 + inner 5+5
        // padding + outer 10+10 padding = 190 for the outer item.
        let l = layout_html(
            r#"<div class="c"><div class="w1"><div class="w2"><table class="t"><tr>
             <td style="width:100%"><div style="width:30px"></div></td>
             <td><div style="width:50px"></div><div style="width:50px"></div></td>
             </tr></table></div></div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; } \
             .w1 { flex: 0 0 auto; padding: 0 10px; } \
             .w2 { padding: 0 5px; }",
        );
        let c = first(&l);
        let w1 = &c.children[0];
        let w2 = &w1.children[0];
        let t = &w2.children[0];
        assert!(w1.rect.w.is_finite() && t.rect.w.is_finite());
        assert!(approx(w1.rect.w, 190.0), "outer wrapper w={}", w1.rect.w);
        assert!(approx(t.rect.w, 160.0), "table w={}", t.rect.w);
    }

    #[test]
    fn w76_row_basis_auto_wrapper_around_fixed_pct_table_still_saturates() {
        // Fixed-table control for (e): the wrapper walk must not
        // substitute a fixed-layout table, so the table saturates.
        let l = layout_html(
            r#"<div class="c"><div class="w"><table class="t"><tr><td></td></tr></table></div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             .w { flex: 0 0 auto; min-width: 0px; } \
             .t { table-layout: fixed; width: 100%; border-spacing: 0; } \
             td { width: 100px; padding: 0; }",
        );
        let w = &first(&l).children[0];
        let t = &w.children[0];
        assert!(t.rect.w.is_finite());
        assert!(t.rect.w > 800.0, "fixed table w={}", t.rect.w);
    }

    #[test]
    fn w76_column_basis_content_keywords_measure_height() {
        // (f) column: a content-keyword basis must go through the
        // measuring path, not collapse to 0. Each item's content is
        // a 90px-tall block, so every keyword yields 90.
        // No whitespace between items: inter-item text runs would
        // become anonymous flex items of their own.
        let l = layout_html(
            r#"<div class="c"><div class="a"><div style="height: 90px"></div></div><div class="b"><div style="height: 90px"></div></div><div class="d"><div style="height: 90px"></div></div></div>"#,
            "body { margin: 0; } .c { display: flex; flex-direction: column; width: 200px; } \
             .a { flex-basis: max-content; flex-shrink: 0; } \
             .b { flex-basis: min-content; flex-shrink: 0; } \
             .d { flex-basis: fit-content; flex-shrink: 0; }",
        );
        let c = first(&l);
        let hs: Vec<f32> = c.children.iter().map(|it| it.rect.h).collect();
        for (i, it) in c.children.iter().enumerate() {
            assert!(it.rect.h.is_finite());
            assert!(
                approx(it.rect.h, 90.0),
                "column item {i} h={} all={hs:?}",
                it.rect.h
            );
        }
    }

    #[test]
    fn w76_container_intrinsic_basis_max_content_probe() {
        // (g) probe, measured before asserting: the flex container's
        // intrinsic sizes with an item whose width:100px is overridden
        // by flex-basis:max-content over 192px of text. The basis, not
        // the width or the intrinsic mode, drives the contribution, so
        // both passes report the max-content size (192). Spec ground
        // (css-flexbox-1 §9.9.3, w76fix finding 14): the main-size
        // min-content contribution is the larger of the item's outer
        // min-content size and its outer preferred (max-content) size,
        // capped by the flex base size when the item is not growable
        // (here grow defaults to 0 and the base is 192) — so the min
        // pass keeps 192 rather than the shrinkable 60.
        let (mc, pc) = measure_flex_intrinsic(
            r#"<div class="c"><div class="a">aaaaaaaaaa bbbbbbbbbb cccccccccc</div></div>"#,
            "body { margin: 0; } .c { display: flex; } \
             .a { font-size: 10px; width: 100px; flex-basis: max-content; }",
        );
        assert!(mc.is_finite() && pc.is_finite());
        assert!(approx(pc, 192.0), "max-content {pc} != 192");
        assert!(approx(mc, 192.0), "min-content {mc} != 192");
    }

    // w76fix finding 1: a nested flex container item must measure its
    // own intrinsic algorithm (item sum), not the generic max over its
    // children, when a definite width would otherwise floor it.
    #[test]
    fn w76fix_nested_flex_basis_max_content_uses_item_sum() {
        let l = layout_html(
            r#"<div class="c"><div class="a"><div class="b"></div><div class="b"></div></div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             .a { display: flex; width: 100px; flex-basis: max-content; \
                  flex-grow: 0; flex-shrink: 0; } \
             .b { width: 150px; flex-shrink: 0; }",
        );
        let a = &first(&l).children[0];
        assert!(a.rect.w.is_finite());
        assert!(approx(a.rect.w, 300.0), "nested flex item w={}", a.rect.w);
    }

    #[test]
    fn w76fix_nested_flex_basis_min_content_uses_item_sum() {
        let l = layout_html(
            r#"<div class="c"><div class="a"><div class="b"></div><div class="b"></div></div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             .a { display: flex; width: 100px; flex-basis: min-content; \
                  flex-grow: 0; flex-shrink: 0; } \
             .b { width: 150px; flex-shrink: 0; }",
        );
        let a = &first(&l).children[0];
        assert!(a.rect.w.is_finite());
        assert!(approx(a.rect.w, 300.0), "nested flex item w={}", a.rect.w);
    }

    // w76fix finding 1 (tables): a definite-width table item with a
    // content-keyword basis measures its columns, not its width.
    // Column maxima here are 50 (fixed div) + 324 (five 60px words
    // plus four 6px spaces, unwrapped) + 30 spacing = 404; column
    // minima are 50 + 60 (longest word) + 30 = 140. With the old
    // dispatched definite floor the max base was 100 and table
    // layout grew the item only to the min-plus-spacing 140, so
    // 404 discriminates the column arm from the floor.
    // NOTE on the task's inline-table shape: since w77g
    // blockification, an inline-table flex item is promoted to a
    // `Table` box carrying its own style, so `flex-basis` on the
    // table DOES reach the item — the shape is now testable (see
    // the inline-table blockification tests in `tests.rs`). The
    // block-level table below is kept as the form of the same
    // column-sum claim that was testable before that promotion
    // landed, when the item really was a style-less wrapper.
    #[test]
    fn w76fix_table_basis_max_content_uses_column_sum() {
        let l = layout_html(
            r#"<div class="c"><table class="t"><tr><td><div style="width:50px"></div></td><td>aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa</td></tr></table></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { font-size: 10px; padding: 0; } \
             .t { width: 100px; flex-basis: max-content; \
                  flex-grow: 0; flex-shrink: 0; }",
        );
        let t = &first(&l).children[0];
        assert!(t.rect.w.is_finite());
        assert!(approx(t.rect.w, 404.0), "table w={}", t.rect.w);
    }

    // Same table, min-content basis: the final width (140) is
    // pinned, but it does not discriminate the base — table layout
    // grows a 110 dispatched minimum to min-plus-spacing 140 either
    // way. The base itself is pinned by
    // `w76fix_table_basis_direct_min_max` below.
    #[test]
    fn w76fix_table_basis_min_content_final_width() {
        let l = layout_html(
            r#"<div class="c"><table class="t"><tr><td><div style="width:50px"></div></td><td>aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa</td></tr></table></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { font-size: 10px; padding: 0; } \
             .t { width: 100px; flex-basis: min-content; \
                  flex-grow: 0; flex-shrink: 0; }",
        );
        let t = &first(&l).children[0];
        assert!(t.rect.w.is_finite());
        assert!(approx(t.rect.w, 140.0), "table w={}", t.rect.w);
    }

    /// Direct `flex-basis` keyword size of the first table in the
    /// document, bypassing layout: pins the base itself, which
    /// table layout growth would otherwise mask in a final width.
    fn table_basis(html: &str, css: &str, spec: crate::style::SpecLen) -> f32 {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let ua = bowser_css::default_ua_stylesheet();
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut found = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::Table {
                found = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().rev().copied());
        }
        let t = found.expect("table box");
        let fonts = FixedFontMetrics;
        let mut pw_cache = super::PrefWidthCache::new(forest.nodes.len());
        let mut mcw = Vec::new();
        let mut bufs = super::WalkBufs::new();
        let mut inner = super::WalkBufs::new();
        super::flex_basis_content_width(
            &mut forest,
            t,
            spec,
            &fonts,
            &mut pw_cache,
            &mut mcw,
            &mut bufs,
            &mut inner,
            f32::INFINITY,
        )
    }

    // Direct base pins for the table above: max-content 404
    // (maxima + spacing), min-content 140 (minima + spacing), both
    // ignoring the definite 100px width.
    #[test]
    fn w76fix_table_basis_direct_min_max() {
        let html = r#"<table class="t"><tr><td><div style="width:50px"></div></td><td>aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa aaaaaaaaaa</td></tr></table>"#;
        let css = "table { border-spacing: 10px 10px; } td { font-size: 10px; padding: 0; } \
             .t { width: 100px; }";
        assert!(
            approx(
                table_basis(html, css, crate::style::SpecLen::MaxContent),
                404.0
            ),
            "max base"
        );
        assert!(
            approx(
                table_basis(html, css, crate::style::SpecLen::MinContent),
                140.0
            ),
            "min base"
        );
    }

    // w76fix2 M2: a caption-dominated definite-width table floors
    // its min-content base at CAPMIN alone (200), not at
    // max(columns, CAPMIN) + spacing (230). Columns minima are
    // 30 + 50 = 80, spacing total 30 (10px * (2 + 1)); the old
    // formula returned 230.
    #[test]
    fn w76fix2_table_basis_min_content_caption_dominated() {
        let html = r#"<table class="t"><caption><div style="width:200px"></div></caption><tr><td><div style="width:30px"></div></td><td><div style="width:50px"></div></td></tr></table>"#;
        let css = "table { border-spacing: 10px 10px; } td { padding: 0; } \
             .t { width: 100px; }";
        let m = table_basis(html, css, crate::style::SpecLen::MinContent);
        assert!(approx(m, 200.0), "caption-dominated min base {m} != 200");
    }

    // w76fix2 M2 control: same shape with a small caption floors at
    // columns + spacing (30 + 50 + 30 = 110); spacing still belongs
    // to the column side when the columns dominate.
    #[test]
    fn w76fix2_table_basis_min_content_column_dominated_control() {
        let html = r#"<table class="t"><caption><div style="width:50px"></div></caption><tr><td><div style="width:30px"></div></td><td><div style="width:50px"></div></td></tr></table>"#;
        let css = "table { border-spacing: 10px 10px; } td { padding: 0; } \
             .t { width: 100px; }";
        let m = table_basis(html, css, crate::style::SpecLen::MinContent);
        assert!(approx(m, 110.0), "column-dominated min base {m} != 110");
    }

    // w76fix finding 4 (shorthand end to end): `flex: 0 0 stretch`
    // parses through the css shorthand sets and fills the definite
    // column container — the pre-w76 invalid-shorthand fallback
    // (`0 1 auto`, content measure 90) must not reappear.
    #[test]
    fn w76fix_column_basis_stretch_shorthand_fills_definite_container() {
        let l = layout_html(
            r#"<div class="c"><div class="a">aaaaaaaaaa bbbbbbbbbb cccccccccc</div></div>"#,
            "body { margin: 0; } .c { display: flex; flex-direction: column; width: 200px; height: 300px; } \
             .a { font-size: 10px; flex: 0 0 stretch; }",
        );
        let a = &first(&l).children[0];
        assert!(a.rect.h.is_finite());
        assert!(approx(a.rect.h, 300.0), "stretch shorthand h={}", a.rect.h);
    }

    // w76fix finding 2: width:100px + flex-basis:min-content +
    // min-width:auto over text with a 60px longest word must stay 60:
    // the automatic minimum is the keyword-measured min-content, not
    // the definite-width-floored min-content (100).
    #[test]
    fn w76fix_row_basis_min_content_not_refloored_by_auto_min() {
        let l = layout_html(
            r#"<div class="c"><div class="a">aaaaaaaaaa bbbbbbbbbb cccccccccc</div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             .a { font-size: 10px; width: 100px; flex-basis: min-content; \
                  flex-grow: 0; flex-shrink: 0; min-width: auto; }",
        );
        let a = &first(&l).children[0];
        assert!(a.rect.w.is_finite());
        assert!(approx(a.rect.w, 60.0), "min-content item w={}", a.rect.w);
    }

    // w76fix finding 3: (e, pretty-printed) newlines inside the wrapper
    // wrap into anonymous blocks around the table; the sole-child scan
    // must skip them and still substitute the grid sum + extras.
    #[test]
    fn w76fix_row_basis_auto_pretty_wrapper_around_pct_table() {
        let l = layout_html(
            "<div class=\"c\"><div class=\"w\">\n<table class=\"t\"><tr>\n<td style=\"width:100%\"><div style=\"width:30px\"></div></td>\n<td><div style=\"width:50px\"></div><div style=\"width:50px\"></div></td>\n</tr></table>\n</div></div>",
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; } \
             .w { flex: 0 0 auto; padding: 0 10px; }",
        );
        let c = first(&l);
        let w = &c.children[0];
        let t = w
            .children
            .iter()
            .find(|ch| ch.box_type == BoxType::Table)
            .expect("table child");
        assert!(w.rect.w.is_finite() && t.rect.w.is_finite());
        assert!(approx(w.rect.w, 180.0), "wrapper w={}", w.rect.w);
        assert!(approx(t.rect.w, 160.0), "table w={}", t.rect.w);
    }

    // w76fix finding 4 (row): flex-basis:stretch fills the available
    // space (600), it does not degrade to the content size.
    #[test]
    fn w76fix_row_basis_stretch_fills_available() {
        let l = layout_html(
            r#"<div class="c"><div class="a"><div style="height: 90px"></div></div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             .a { flex-basis: stretch; flex-grow: 0; flex-shrink: 0; }",
        );
        let a = &first(&l).children[0];
        assert!(a.rect.w.is_finite());
        assert!(approx(a.rect.w, 600.0), "stretch row item w={}", a.rect.w);
    }

    // w76fix finding 4 (column, definite): stretch fills the
    // container's definite main size (300), not the 90px content.
    #[test]
    fn w76fix_column_basis_stretch_fills_definite_container() {
        let l = layout_html(
            r#"<div class="c"><div class="a"><div style="height: 90px"></div></div></div>"#,
            "body { margin: 0; } .c { display: flex; flex-direction: column; \
                  width: 200px; height: 300px; } \
             .a { flex-basis: stretch; flex-shrink: 0; }",
        );
        let a = &first(&l).children[0];
        assert!(a.rect.h.is_finite());
        assert!(
            approx(a.rect.h, 300.0),
            "stretch column item h={}",
            a.rect.h
        );
    }

    // w76fix finding 4 (column, indefinite): with no definite container
    // main size, stretch falls back to measuring the content (90).
    #[test]
    fn w76fix_column_basis_stretch_measures_content_when_indefinite() {
        let l = layout_html(
            r#"<div class="c"><div class="a"><div style="height: 90px"></div></div></div>"#,
            "body { margin: 0; } .c { display: flex; flex-direction: column; width: 200px; } \
             .a { flex-basis: stretch; flex-shrink: 0; }",
        );
        let a = &first(&l).children[0];
        assert!(a.rect.h.is_finite());
        assert!(approx(a.rect.h, 90.0), "stretch column item h={}", a.rect.h);
    }

    // w76fix finding 4 (intrinsic): an intrinsic pass has no definite
    // container to fill, so stretch falls back to max-content (192).
    #[test]
    fn w76fix_container_intrinsic_basis_stretch_falls_back_to_max_content() {
        let (mc, pc) = measure_flex_intrinsic(
            r#"<div class="c"><div class="a">aaaaaaaaaa bbbbbbbbbb cccccccccc</div></div>"#,
            "body { margin: 0; } .c { display: flex; } \
             .a { font-size: 10px; flex-basis: stretch; }",
        );
        assert!(mc.is_finite() && pc.is_finite());
        assert!(approx(pc, 192.0), "max-content {pc} != 192");
        assert!(approx(mc, 192.0), "min-content {mc} != 192");
    }

    // w76fix finding 7: column keyword bases measure at the keyword
    // width. Container 130px: min-content width 60 wraps the text to 3
    // lines, fit-content width 130 to 2 lines, max-content width 192
    // to 1 line, so the heights stand in a 3:2:1 ratio.
    #[test]
    fn w76fix_column_basis_content_keywords_measure_at_keyword_width() {
        let l = layout_html(
            r#"<div class="c"><div class="a">aaaaaaaaaa bbbbbbbbbb cccccccccc</div><div class="b">aaaaaaaaaa bbbbbbbbbb cccccccccc</div><div class="d">aaaaaaaaaa bbbbbbbbbb cccccccccc</div></div>"#,
            "body { margin: 0; } .c { display: flex; flex-direction: column; width: 130px; } \
             .a { font-size: 10px; flex-basis: min-content; flex-shrink: 0; } \
             .b { font-size: 10px; flex-basis: fit-content; flex-shrink: 0; } \
             .d { font-size: 10px; flex-basis: max-content; flex-shrink: 0; }",
        );
        let c = first(&l);
        let (ha, hb, hd) = (
            c.children[0].rect.h,
            c.children[1].rect.h,
            c.children[2].rect.h,
        );
        assert!(ha.is_finite() && hb.is_finite() && hd.is_finite());
        assert!(
            ha > hb && hb > hd,
            "min {ha} / fit {hb} / max {hd} not 3:2:1 lines"
        );
        assert!(
            approx(ha, hb + hd),
            "min {ha} != fit {hb} + max {hd} (3 lines != 2 + 1)"
        );
    }

    // w76fix finding 7 (definite-height): a column keyword basis
    // overrides the main-size property, so a height:90px item with a
    // min-content basis measures its 3-line keyword height, exactly
    // like the same item without a specified height.
    #[test]
    fn w76fix_column_basis_min_content_ignores_definite_height() {
        let l = layout_html(
            r#"<div class="c"><div class="a">aaaaaaaaaa bbbbbbbbbb cccccccccc</div><div class="b">aaaaaaaaaa bbbbbbbbbb cccccccccc</div></div>"#,
            "body { margin: 0; } .c { display: flex; flex-direction: column; width: 130px; } \
             .a { font-size: 10px; flex-basis: min-content; flex-shrink: 0; } \
             .b { font-size: 10px; flex-basis: min-content; flex-shrink: 0; height: 90px; }",
        );
        let c = first(&l);
        assert!(
            approx(c.children[0].rect.h, c.children[1].rect.h),
            "plain h={} vs definite-height h={}",
            c.children[0].rect.h,
            c.children[1].rect.h
        );
    }

    // w76fix2 L4: a generic column item with height:300px and a
    // max-content basis measures at its content, and its sole
    // %-height child resolves against an indefinite base (as `auto`)
    // during that measurement — not against the zero minimum-height
    // override. The child's natural height is 90 (its inner block),
    // so the measured base — and the final item height — is 90, not
    // ~0. Before the fix the child resolved 100% against 0 and the
    // base collapsed.
    #[test]
    fn w76fix2_column_basis_pct_height_child_measures_natural() {
        let l = layout_html(
            r#"<div class="c"><div class="a"><div class="k"><div style="height: 90px"></div></div></div></div>"#,
            "body { margin: 0; } .c { display: flex; flex-direction: column; width: 200px; } \
             .a { flex-basis: max-content; flex-shrink: 0; height: 300px; } \
             .k { height: 100%; }",
        );
        let a = &first(&l).children[0];
        let k = &a.children[0];
        assert!(a.rect.h.is_finite() && k.rect.h.is_finite());
        assert!(approx(a.rect.h, 90.0), "column item h={}", a.rect.h);
        assert!(approx(k.rect.h, 90.0), "pct-height child h={}", k.rect.h);
    }

    // w76fix finding 8: width:100px + flex-basis:min-content (60) +
    // min-width:min-content must stay 60: the min clamp measures the
    // keyword, not the definite width (100).
    #[test]
    fn w76fix_row_basis_min_content_min_width_keyword_not_raised_to_width() {
        let l = layout_html(
            r#"<div class="c"><div class="a">aaaaaaaaaa bbbbbbbbbb cccccccccc</div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             .a { font-size: 10px; width: 100px; flex-basis: min-content; \
                  flex-grow: 0; flex-shrink: 0; min-width: min-content; }",
        );
        let a = &first(&l).children[0];
        assert!(a.rect.w.is_finite());
        assert!(approx(a.rect.w, 60.0), "min-content item w={}", a.rect.w);
    }

    // w76fix finding 8 (intrinsic mirror): width:100px +
    // flex-basis:max-content (192) + max-width:fit-content must keep
    // 192 in both passes: the fit-content clamp must not resolve
    // against the width-derived 100.
    #[test]
    fn w76fix_container_intrinsic_max_width_fit_content_not_capped_by_width() {
        let (mc, pc) = measure_flex_intrinsic(
            r#"<div class="c"><div class="a">aaaaaaaaaa bbbbbbbbbb cccccccccc</div></div>"#,
            "body { margin: 0; } .c { display: flex; } \
             .a { font-size: 10px; width: 100px; flex-basis: max-content; max-width: fit-content; }",
        );
        assert!(mc.is_finite() && pc.is_finite());
        assert!(approx(pc, 192.0), "max-content {pc} != 192");
        assert!(approx(mc, 192.0), "min-content {mc} != 192");
    }

    // w76fix finding 10: in the min-content pass a fit-content basis
    // measures min-content (60), not max-content (192); the max pass
    // still reports 192.
    #[test]
    fn w76fix_container_intrinsic_basis_fit_content_min_pass_uses_min_content() {
        let (mc, pc) = measure_flex_intrinsic(
            r#"<div class="c"><div class="a">aaaaaaaaaa bbbbbbbbbb cccccccccc</div></div>"#,
            "body { margin: 0; } .c { display: flex; } \
             .a { font-size: 10px; flex-basis: fit-content; }",
        );
        assert!(mc.is_finite() && pc.is_finite());
        assert!(approx(pc, 192.0), "max-content {pc} != 192");
        assert!(approx(mc, 60.0), "min-content {mc} != 60");
    }

    // w76fix finding 11: a flex-container item whose sole-child chain
    // leads to a table must not get the grid substitution: its own
    // flex intrinsic algorithm owns the base, so it saturates.
    #[test]
    fn w76fix_wrapper_flex_item_itself_not_substituted() {
        let l = layout_html(
            r#"<div class="c"><div class="a"><table class="t"><tr>
             <td style="width:100%"><div style="width:30px"></div></td>
             <td><div style="width:50px"></div><div style="width:50px"></div></td>
             </tr></table></div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; } \
             .a { display: flex; flex: 0 0 auto; }",
        );
        let a = &first(&l).children[0];
        assert!(a.rect.w.is_finite());
        assert!(
            a.rect.w > 800.0,
            "flex item w={} (want saturation, not 160)",
            a.rect.w
        );
    }

    // w76fix finding 11 (multicol): a multicol item whose sole-child
    // chain leads to a table must not get the grid substitution:
    // pre-fix the bypass read the generic max over children
    // (infinite through the table) and the wrapper walk substituted
    // the grid sum (160). Post-fix the gate excludes multicol, so
    // the base is the multicol arm itself: 2 columns over the
    // single-column content measure (the table's min-content 80 =
    // column minima 30 + 50, spacing excluded from the min walk)
    // plus one 16px column gap (normal = 1em at the default 16px
    // font) = 176. NOT the dispatched definite 100: finding 1 drops
    // the definite floor for multicol, so the basis overrides the
    // width here (an earlier draft of this test expected 100; that
    // confused the dispatched floor with the multicol arm value).
    #[test]
    fn w76fix_wrapper_multicol_item_itself_not_substituted() {
        let l = layout_html(
            r#"<div class="c"><div class="a"><table class="t"><tr>
             <td style="width:100%"><div style="width:30px"></div></td>
             <td><div style="width:50px"></div><div style="width:50px"></div></td>
             </tr></table></div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; } \
             .a { column-count: 2; width: 100px; flex-basis: max-content; \
                  flex-grow: 0; flex-shrink: 0; }",
        );
        let a = &first(&l).children[0];
        assert!(a.rect.w.is_finite());
        assert!(approx(a.rect.w, 176.0), "multicol item w={}", a.rect.w);
    }

    // w76fix finding 9: a percentage-width intermediate does not fill
    // the wrapper, so the grid substitution must not fire (saturate).
    #[test]
    fn w76fix_wrapper_pct_intermediate_not_substituted() {
        let l = layout_html(
            r#"<div class="c"><div class="w"><div class="w2"><table class="t"><tr>
             <td style="width:100%"><div style="width:30px"></div></td>
             <td><div style="width:50px"></div><div style="width:50px"></div></td>
             </tr></table></div></div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; } \
             .w { flex: 0 0 auto; padding: 0 10px; } \
             .w2 { width: 50%; }",
        );
        let w = &first(&l).children[0];
        assert!(w.rect.w.is_finite());
        assert!(
            w.rect.w > 800.0,
            "wrapper w={} (want saturation, not 180)",
            w.rect.w
        );
    }

    // w76fix finding 9: same for a floated terminal table (its used
    // width is shrink-to-fit, not the wrapper fill).
    #[test]
    fn w76fix_wrapper_floated_terminal_table_not_substituted() {
        let l = layout_html(
            r#"<div class="c"><div class="w"><table class="t"><tr>
             <td style="width:100%"><div style="width:30px"></div></td>
             <td><div style="width:50px"></div><div style="width:50px"></div></td>
             </tr></table></div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; } \
             .w { flex: 0 0 auto; padding: 0 10px; } \
             .t { float: left; }",
        );
        let w = &first(&l).children[0];
        assert!(w.rect.w.is_finite());
        assert!(
            w.rect.w > 800.0,
            "wrapper w={} (want saturation, not 180)",
            w.rect.w
        );
    }

    // w76fix finding 9: same for an intermediate with a min-width that
    // would cap its fill.
    #[test]
    fn w76fix_wrapper_minmax_intermediate_not_substituted() {
        let l = layout_html(
            r#"<div class="c"><div class="w"><div class="w2"><table class="t"><tr>
             <td style="width:100%"><div style="width:30px"></div></td>
             <td><div style="width:50px"></div><div style="width:50px"></div></td>
             </tr></table></div></div></div>"#,
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; } \
             .w { flex: 0 0 auto; padding: 0 10px; } \
             .w2 { min-width: 500px; }",
        );
        let w = &first(&l).children[0];
        assert!(w.rect.w.is_finite());
        assert!(
            w.rect.w > 800.0,
            "wrapper w={} (want saturation, not 180)",
            w.rect.w
        );
    }
}

#[cfg(test)]
mod inline_forced_break_tests {
    //! Inline-level forced column breaks: `do_inline_layout` splits the
    //! line box at the break and records the fresh-column line indices in
    //! [`InlineLayoutOutput::forced_break_lines`].
    use super::layout_multicol::MulticolParams;
    use super::*;
    use crate::{FixedFontMetrics, Size};
    use std::collections::HashMap;

    const VP: Size = Size { w: 800.0, h: 600.0 };

    fn forest_of(html: &str, css: &str) -> Forest {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let ua = crate::ua_stylesheet();
        crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            ua,
            VP.w,
            VP.h,
            &HashMap::new(),
            &crate::FixedFontMetrics,
        )
        .expect("box tree builds")
        .0
    }

    /// The `.mc` block container's arena index.
    fn find_mc(forest: &Forest) -> usize {
        forest
            .nodes
            .iter()
            .position(|n| {
                n.box_type == BoxType::Block
                    && n.style
                        .as_ref()
                        .and_then(|s| s.get_by_index(props::COLUMN_COUNT))
                        .is_some_and(|v| matches!(v, CssValue::Number(_)))
            })
            .expect("multicol container node")
    }

    fn layout_inline(forest: &mut Forest, multicol: Option<&MulticolParams>) -> InlineLayoutOutput {
        let idx = find_mc(forest);
        let mut scratch = LayoutScratch::new();
        let floats: &[FloatEntry] = &[];
        do_inline_layout(
            forest,
            idx,
            0.0,
            0.0,
            200.0,
            floats,
            &FixedFontMetrics,
            &mut scratch,
            multicol,
            None,
        )
    }

    fn mc_params() -> MulticolParams {
        MulticolParams {
            count: 2,
            gap: 0.0,
            col_w: 200.0,
            fill_auto: false,
        }
    }

    const HTML: &str = r#"<div class="mc">aaa <span class="br">bbb</span> ccc</div>"#;
    const CSS_MC: &str = ".mc { width: 400px; column-count: 2; column-gap: 0; } ";

    #[test]
    fn inline_break_before_column_records_line() {
        let mut forest = forest_of(HTML, &format!("{CSS_MC}.br {{ break-before: column; }}"));
        let mc = mc_params();
        let out = layout_inline(&mut forest, Some(&mc));
        // "aaa" stays on line 0; the span's forced break splits the line
        // box, so line 1 (starting with "bbb") must begin a fresh column.
        assert_eq!(out.forced_break_lines, vec![1]);
    }

    #[test]
    fn inline_break_after_column_records_line() {
        let mut forest = forest_of(HTML, &format!("{CSS_MC}.br {{ break-after: column; }}"));
        let mc = mc_params();
        let out = layout_inline(&mut forest, Some(&mc));
        // The line box ends after the span ("aaa bbb"); the following
        // line ("ccc") must begin a fresh column.
        assert_eq!(out.forced_break_lines, vec![1]);
    }

    #[test]
    fn inline_break_before_page_alias_records_line() {
        // Page-break aliases fragment as column breaks inside multicol.
        let mut forest = forest_of(HTML, &format!("{CSS_MC}.br {{ break-before: page; }}"));
        let mc = mc_params();
        let out = layout_inline(&mut forest, Some(&mc));
        assert_eq!(out.forced_break_lines, vec![1]);
    }

    #[test]
    fn inline_break_before_first_line_is_noop() {
        // A forced break at the very start of the content creates no
        // empty column: line 0 already begins a fresh column.
        let mut forest = forest_of(
            r#"<div class="mc"><span class="br">aaa</span> bbb</div>"#,
            &format!("{CSS_MC}.br {{ break-before: column; }}"),
        );
        let mc = mc_params();
        let out = layout_inline(&mut forest, Some(&mc));
        assert!(out.forced_break_lines.is_empty());
    }

    #[test]
    fn inline_forced_breaks_ignored_outside_multicol() {
        // Outside a multicol container there is no column to break into:
        // `break-before: column` behaves as `auto` — no line split, and
        // nothing is recorded.
        let css_break = format!("{CSS_MC}.br {{ break-before: column; }}");
        let css_plain = format!("{CSS_MC}.br {{ }}");
        let mut f_break = forest_of(HTML, &css_break);
        let mut f_plain = forest_of(HTML, &css_plain);
        let out_break = layout_inline(&mut f_break, None);
        let out_plain = layout_inline(&mut f_plain, None);
        assert!(out_break.forced_break_lines.is_empty());
        assert!(out_plain.forced_break_lines.is_empty());
        // Identical heights: the forced break did not split the line.
        assert!((out_break.height - out_plain.height).abs() < 0.001);
    }
}

#[cfg(test)]
mod br_tests {
    //! `<br>` forced line breaks: the element emits a zero-width fragment
    //! that breaks the line *after* it (HTML rendering, "the `br`
    //! element"). Covers mid-content, leading, trailing, and consecutive
    //! `<br>`s, plus the break-aware intrinsic widths.
    use super::*;
    use crate::{FixedFontMetrics, Size};
    use std::collections::HashMap;

    const VP: Size = Size { w: 800.0, h: 600.0 };

    fn forest_of(html: &str, css: &str) -> Forest {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let ua = crate::ua_stylesheet();
        crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            ua,
            VP.w,
            VP.h,
            &HashMap::new(),
            &crate::FixedFontMetrics,
        )
        .expect("box tree builds")
        .0
    }

    fn find_div(forest: &Forest) -> usize {
        forest
            .nodes
            .iter()
            .position(|n| n.box_type == BoxType::Block && (n.data.font_size - 10.0).abs() < 0.01)
            .expect("test div node")
    }

    /// Broken-line snapshot from a direct [`do_inline_layout`] run.
    struct Snap {
        nfrags: usize,
        line_starts: Vec<usize>,
        line_ends: Vec<usize>,
        height: f32,
    }

    fn layout_lines(html: &str, content_w: f32) -> Snap {
        let css = ".t { font-size: 10px; line-height: 10px; }";
        let mut forest = forest_of(html, css);
        let idx = find_div(&forest);
        let mut scratch = LayoutScratch::new();
        let out = do_inline_layout(
            &mut forest,
            idx,
            0.0,
            0.0,
            content_w,
            &[],
            &FixedFontMetrics,
            &mut scratch,
            None,
            None,
        );
        Snap {
            nfrags: scratch.frags.len(),
            line_starts: scratch.lines.iter().map(|l| l.start).collect(),
            line_ends: scratch.lines.iter().map(|l| l.end).collect(),
            height: out.height,
        }
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn br_breaks_line_after_it() {
        // "aa<br>bb": the break lands after the `<br>`; the `<br>`
        // fragment itself sits at the end of the first line.
        let s = layout_lines(r#"<div class="t">aa<br>bb</div>"#, 100.0);
        assert_eq!(s.nfrags, 3);
        assert_eq!(s.line_starts, vec![0, 2]);
        assert_eq!(s.line_ends, vec![2, 3]);
        assert!(approx(s.height, 20.0), "height={}", s.height);
    }

    #[test]
    fn leading_br_makes_empty_first_line() {
        // "<br>aa": the empty line before the break is real.
        let s = layout_lines(r#"<div class="t"><br>aa</div>"#, 100.0);
        assert_eq!(s.nfrags, 2);
        assert_eq!(s.line_starts, vec![0, 1]);
        assert_eq!(s.line_ends, vec![1, 2]);
        assert!(approx(s.height, 20.0), "height={}", s.height);
    }

    #[test]
    fn trailing_br_makes_empty_last_line() {
        // "aa<br>": the break after the `<br>` still produces its empty
        // line (strut-height).
        let s = layout_lines(r#"<div class="t">aa<br></div>"#, 100.0);
        assert_eq!(s.nfrags, 3);
        assert_eq!(s.line_starts, vec![0, 2]);
        assert_eq!(s.line_ends, vec![2, 3]);
        assert!(approx(s.height, 20.0), "height={}", s.height);
    }

    #[test]
    fn consecutive_brs_stack_empty_lines() {
        // "<br><br>": three line boxes (before, between, after).
        let s = layout_lines(r#"<div class="t"><br><br></div>"#, 100.0);
        assert_eq!(s.nfrags, 3);
        assert_eq!(s.line_starts, vec![0, 1, 2]);
        assert_eq!(s.line_ends, vec![1, 2, 3]);
        assert!(approx(s.height, 30.0), "height={}", s.height);
    }

    #[test]
    fn br_only_block_is_one_strut_line() {
        // A lone `<br>` is one empty line, not zero lines.
        let s = layout_lines(r#"<div class="t"><br></div>"#, 100.0);
        assert_eq!(s.nfrags, 2);
        assert_eq!(s.line_starts, vec![0, 1]);
        assert!(approx(s.height, 20.0), "height={}", s.height);
    }

    #[test]
    fn br_min_content_is_widest_segment() {
        // min-content of "aa<br>bbbbbb" is the widest segment, not the sum.
        let css = ".t { font-size: 10px; line-height: 10px; }";
        let mut forest = forest_of(r#"<div class="t">aa<br>bbbbbb</div>"#, css);
        let idx = find_div(&forest);
        let mut scratch = vec![0.0f32];
        let w = min_content_width(&mut forest, idx, &FixedFontMetrics, &mut scratch);
        // FixedFontMetrics advances 0.6em: "bbbbbb" is 36px.
        assert!(approx(w, 36.0), "w={w}");
    }

    #[test]
    fn br_max_content_is_widest_segment() {
        // max-content likewise takes the widest break-delimited segment.
        let css = ".t { font-size: 10px; line-height: 10px; }";
        let mut forest = forest_of(r#"<div class="t">aa<br>bbbbbb</div>"#, css);
        let idx = find_div(&forest);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let w = preferred_width(&mut forest, idx, &FixedFontMetrics, &mut cache);
        assert!(approx(w, 36.0), "w={w}");
    }

    #[test]
    fn br_in_block_layout_grows_height() {
        // End-to-end through `layout()`: two lines, 20px.
        let (doc, _) = bowser_dom::Document::parse_html(b"<div class=\"t\">aa<br>bb</div>");
        let ss = bowser_css::parse_stylesheet(
            "body { margin: 0; } .t { font-size: 10px; line-height: 10px; }",
        );
        let l = crate::layout(&doc, &[&ss], VP, &FixedFontMetrics);
        let div = &l.children[0].children[0];
        assert!(approx(div.rect.h, 20.0), "h={}", div.rect.h);
    }
}

#[cfg(test)]
mod line_layout_tests {
    //! Regression tests for `text-align: justify` inter-word spreading
    //! and for the inline-box boundary spacing transfer when a trimmable
    //! space is dropped during line breaking (CSS 2.1 §8.6 / §16.6.1).
    use super::*;
    use crate::{FixedFontMetrics, Size};
    use std::collections::HashMap;

    const VP: Size = Size { w: 800.0, h: 600.0 };

    fn forest_of(html: &str, css: &str) -> Forest {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let ua = crate::ua_stylesheet();
        crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            ua,
            VP.w,
            VP.h,
            &HashMap::new(),
            &crate::FixedFontMetrics,
        )
        .expect("box tree builds")
        .0
    }

    /// Arena index of the test `<div>`: the block carrying the given
    /// explicit `text-align` value. (The UA sheet sets `text-align:
    /// start` on every block, so matching "any `text-align`" would hit
    /// the root.)
    fn find_block(forest: &Forest, align: &str) -> usize {
        forest
            .nodes
            .iter()
            .position(|n| {
                n.box_type == BoxType::Block
                    && n.style
                        .as_ref()
                        .and_then(|s| s.get_str_by_index(props::TEXT_ALIGN))
                        .is_some_and(|v| v == align)
            })
            .expect("test block node")
    }

    /// Fragment x/width/placement snapshot plus the broken lines, from a
    /// direct [`do_inline_layout`] run.
    struct Snap {
        xs: Vec<f32>,
        ws: Vec<f32>,
        placed: Vec<bool>,
        line_widths: Vec<f32>,
        line_starts: Vec<usize>,
        line_ends: Vec<usize>,
    }

    fn layout_lines(
        html: &str,
        css: &str,
        align: &str,
        content_w: f32,
        floats: &[FloatEntry],
    ) -> Snap {
        let mut forest = forest_of(html, css);
        let idx = find_block(&forest, align);
        let mut scratch = LayoutScratch::new();
        do_inline_layout(
            &mut forest,
            idx,
            0.0,
            0.0,
            content_w,
            floats,
            &FixedFontMetrics,
            &mut scratch,
            None,
            None,
        );
        Snap {
            xs: scratch.frags.iter().map(|f| f.x).collect(),
            ws: scratch.frags.iter().map(|f| f.w).collect(),
            placed: scratch.frags.iter().map(|f| f.placed).collect(),
            line_widths: scratch.lines.iter().map(|l| l.width).collect(),
            line_starts: scratch.lines.iter().map(|l| l.start).collect(),
            line_ends: scratch.lines.iter().map(|l| l.end).collect(),
        }
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    // With `FixedFontMetrics` at the 16px default, a word like "aa" is
    // 19.2px and a space 9.6px.

    #[test]
    fn justify_two_word_line_spreads_leftover_across_gap() {
        // "aa bb" is 19.2 + 9.6 + 19.2 = 48px in a 60px line: the 12px
        // leftover spreads across the single word gap, so "bb" ends
        // flush with the line's right edge.
        let s = layout_lines(
            r#"<div class="j">aa bb cc</div>"#,
            ".j { width: 60px; text-align: justify; }",
            "justify",
            60.0,
            &[],
        );
        assert_eq!(s.line_starts, vec![0, 4]);
        assert_eq!(s.line_ends, vec![3, 5]);
        assert!(approx(s.xs[0], 0.0), "aa.x={}", s.xs[0]);
        // The space keeps its natural advance; the gap growth is
        // cursor-only.
        assert!(approx(s.xs[1], 19.2), "sp.x={}", s.xs[1]);
        assert!(approx(s.ws[1], 9.6));
        // "bb" shifted right by the 12px justification share.
        assert!(approx(s.xs[2], 40.8), "bb.x={}", s.xs[2]);
        assert!(approx(s.xs[2] + s.ws[2], 60.0));
        // The trimmed trailing space is not placed.
        assert!(!s.placed[3]);
    }

    #[test]
    fn justify_last_line_is_exempt() {
        // Three lines: the first two justify, the last aligns start.
        let s = layout_lines(
            r#"<div class="j">aa bb cc dd ee</div>"#,
            ".j { width: 60px; text-align: justify; }",
            "justify",
            60.0,
            &[],
        );
        assert_eq!(s.line_starts, vec![0, 4, 8]);
        assert_eq!(s.line_ends, vec![3, 7, 9]);
        // First line justified...
        assert!(approx(s.xs[2], 40.8), "bb.x={}", s.xs[2]);
        // ...middle line justified like the first...
        assert!(approx(s.xs[4], 0.0), "cc.x={}", s.xs[4]);
        assert!(approx(s.xs[6], 40.8), "dd.x={}", s.xs[6]);
        // ...but the last line is exempt: "ee" sits at the start.
        assert!(approx(s.xs[8], 0.0), "ee.x={}", s.xs[8]);
    }

    #[test]
    fn justify_float_adjacent_line_uses_narrowed_width() {
        // A 30px left float narrows the 200px line to 170px: the leftover
        // is measured against the narrowed band, so the last word still
        // ends flush at x=200.
        let floats = [FloatEntry {
            x: 0.0,
            y: 0.0,
            w: 30.0,
            h: 100.0,
            left: true,
            node: usize::MAX,
        }];
        let s = layout_lines(
            r#"<div class="j">aa bb cc dd ee ff gg</div>"#,
            ".j { width: 200px; text-align: justify; }",
            "justify",
            200.0,
            &floats,
        );
        // 6 words fit in 170px: 6*19.2 + 5*9.6 = 163.2, leaving 6.8px
        // over 5 gaps.
        assert_eq!(s.line_starts[0], 0);
        assert_eq!(s.line_ends[0], 11);
        let extra = 6.8 / 5.0;
        assert!(approx(s.xs[0], 30.0), "aa.x={}", s.xs[0]);
        assert!(approx(s.xs[2], 30.0 + 28.8 + extra), "bb.x={}", s.xs[2]);
        assert!(
            approx(s.xs[10], 30.0 + 5.0 * 28.8 + 5.0 * extra),
            "ff.x={}",
            s.xs[10]
        );
        assert!(approx(s.xs[10] + s.ws[10], 200.0));
        // The last line ("gg") is exempt and starts past the float.
        assert!(approx(s.xs[12], 30.0), "gg.x={}", s.xs[12]);
    }

    #[test]
    fn trailing_trimmed_space_transfers_inline_box_spacing() {
        // The span's 2px right padding sits on its trailing space; the
        // space is trimmed at the line break, so the padding transfers
        // to "bb" and joins the line width instead of vanishing with the
        // space.
        let s = layout_lines(
            r#"<div class="t">aa <span class="s">bb </span>cc</div>"#,
            ".t { width: 60px; text-align: left; } .s { padding-right: 2px; }",
            "left",
            60.0,
            &[],
        );
        // Line 1 is [aa, sp, bb]; the trailing space (frag 3) is dropped.
        assert_eq!(s.line_starts, vec![0, 4]);
        assert_eq!(s.line_ends, vec![3, 5]);
        assert!(!s.placed[3]);
        // 19.2 + 9.6 + 19.2 = 48, plus the transferred 2px of padding
        // (48 without the transfer).
        assert!(
            approx(s.line_widths[0], 50.0),
            "line width={}",
            s.line_widths[0]
        );
        // "bb" itself does not move; the spacing trails it.
        assert!(approx(s.xs[2], 28.8), "bb.x={}", s.xs[2]);
    }

    #[test]
    fn leading_trimmed_space_transfers_inline_box_spacing() {
        // The span's 10px left padding + 2px left border sit on its
        // leading space; the space is skipped at the line start, so the
        // 12px transfers to "bb" instead of vanishing.
        let s = layout_lines(
            r#"<div class="t"><span class="s"> bb</span> cc</div>"#,
            ".t { width: 50px; text-align: left; } .s { padding-left: 10px; border-left: 2px solid black; }",
            "left",
            50.0,
            &[],
        );
        assert_eq!(s.line_starts, vec![1, 3]);
        assert_eq!(s.line_ends, vec![2, 4]);
        assert!(!s.placed[0]);
        // "bb" starts after the transferred 12px of box spacing (0
        // without the transfer).
        assert!(approx(s.xs[1], 12.0), "bb.x={}", s.xs[1]);
        // The line width counts the spacing, so it can force the wrap.
        assert!(
            approx(s.line_widths[0], 31.2),
            "line width={}",
            s.line_widths[0]
        );
    }

    #[test]
    fn trailing_trimmed_space_lead_share_renders_after_survivor() {
        // The span starts at the trimmed trailing space, so its 3px of
        // padding-left is the *lead* share of that space. The trimmed
        // space sits adjacent to "bb" on its trailing side, so the 3px
        // must render *after* "bb", not before it.
        let s = layout_lines(
            r#"<div class="t">aa bb<span class="s"> </span>cc</div>"#,
            ".t { width: 62px; text-align: left; } .s { padding-left: 3px; }",
            "left",
            62.0,
            &[],
        );
        // Line 1 is [aa, sp, bb]; the span's space (frag 3) is dropped.
        assert_eq!(s.line_starts, vec![0, 4]);
        assert_eq!(s.line_ends, vec![3, 5]);
        assert!(!s.placed[3]);
        // "bb" itself does not move: the 3px trails it (31.8 when the
        // share is wrongly added to bb's lead).
        assert!(approx(s.xs[2], 28.8), "bb.x={}", s.xs[2]);
        // The line width still counts the spacing: 48 + 3.
        assert!(
            approx(s.line_widths[0], 51.0),
            "line width={}",
            s.line_widths[0]
        );
    }

    #[test]
    fn leading_skipped_space_trail_share_renders_before_survivor() {
        // The span ends at the skipped leading space, so its 2px of
        // padding-right is the *trail* share of that space. The skipped
        // space sits adjacent to "bb" on its leading side, so the 2px
        // must render *before* "bb", not after it.
        let s = layout_lines(
            r#"<div class="t"><span class="s"> </span>bb cc</div>"#,
            ".t { width: 45px; text-align: left; } .s { padding-right: 2px; }",
            "left",
            45.0,
            &[],
        );
        assert_eq!(s.line_starts, vec![1, 3]);
        assert_eq!(s.line_ends, vec![2, 4]);
        assert!(!s.placed[0]);
        // "bb" starts after the transferred 2px of box spacing (0 when
        // the share is wrongly added to bb's trail).
        assert!(approx(s.xs[1], 2.0), "bb.x={}", s.xs[1]);
        // The line width counts the spacing, so it can force the wrap:
        // 19.2 + 2.
        assert!(
            approx(s.line_widths[0], 21.2),
            "line width={}",
            s.line_widths[0]
        );
    }
}

#[cfg(test)]
mod bfc_float_band_tests {
    //! Regression tests for the post-layout BFC/float band re-test
    //! ([`super::bfc_post_layout_shift`], CSS 2.1 §9.5).
    use crate::{FixedFontMetrics, LayoutBox, Size, layout};

    const VP: Size = Size { w: 800.0, h: 600.0 };

    fn layout_html(html: &str, css: &str) -> LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        layout(&doc, &[&ss], VP, &FixedFontMetrics)
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn bfc_clears_float_starting_below_its_top_edge() {
        // f0 (left, 100x200) sits at the top; the spacer advances the
        // cursor to 60. f1 (left, 350x100) cannot fit beside f0, so it
        // steps below it to y=200 — strictly below the BFC's top edge
        // (cy=60). The pre-layout band is only a point test for the
        // auto-height BFC, so it narrows the BFC beside f0 (x=100,
        // w=300) without ever seeing f1; the 300px-tall body would then
        // overlap f1's margin box ([0,350) x [200,300)). Post-layout the
        // BFC must clear below f1 instead (CSS 2.1 §9.5).
        let l = layout_html(
            r#"<div class="w"><div class="f0"></div><div class="a"></div><div class="f1"></div><div class="bfc"><div class="in"></div></div></div>"#,
            "body { margin: 0; } .w { width: 400px; } \
             .f0 { float: left; width: 100px; height: 200px; } \
             .a { height: 60px; } \
             .f1 { float: left; width: 350px; height: 100px; } \
             .bfc { overflow: hidden; } .in { height: 300px; }",
        );
        let w = &l.children[0].children[0];
        let (f0, f1, bfc) = (&w.children[0], &w.children[2], &w.children[3]);
        assert!(approx(f1.rect.y, 200.0), "f1.y={}", f1.rect.y);
        // Cleared below f1: no overlap between the BFC's border box and
        // any float's margin box.
        assert!(approx(bfc.rect.y, 300.0), "bfc.y={}", bfc.rect.y);
        for f in [f0, f1] {
            let x_overlap = f.rect.x < bfc.rect.x + bfc.rect.w && f.rect.x + f.rect.w > bfc.rect.x;
            let y_overlap = f.rect.y < bfc.rect.y + bfc.rect.h && f.rect.y + f.rect.h > bfc.rect.y;
            assert!(
                !(x_overlap && y_overlap),
                "BFC {:?} overlaps float {:?}",
                bfc.rect,
                f.rect
            );
        }
        // The parent still contains the shifted child: its height covers
        // the BFC's shifted bottom.
        assert!(
            w.rect.h >= bfc.rect.y + bfc.rect.h - w.rect.y - 0.01,
            "w.h={} bfc bottom={}",
            w.rect.h,
            bfc.rect.y + bfc.rect.h
        );
    }

    #[test]
    fn zero_width_bfc_fits_zero_width_gap_between_floats() {
        // WPT css/CSS2/floats/zero-space-between-floats-001: a zero-width
        // BFC root fits in the zero-width gap between a left and a right
        // float instead of clearing below them (CSS 2.1 §9.5: its border
        // box does not overlap any float's margin box).
        let l = layout_html(
            r#"<div class="w"><div class="fl"></div><div class="fr"></div><div class="bfc"></div></div>"#,
            "body { margin: 0; } .w { width: 200px; position: relative; } \
             .fl { float: left; width: 100px; height: 200px; } \
             .fr { float: right; width: 100px; height: 200px; } \
             .bfc { overflow: hidden; width: 0; height: 200px; }",
        );
        let w = &l.children[0].children[0];
        let bfc = &w.children[2];
        assert!(approx(bfc.rect.x, 100.0), "bfc.x={}", bfc.rect.x);
        assert!(approx(bfc.rect.y, 0.0), "bfc.y={}", bfc.rect.y);
        assert!(approx(bfc.rect.w, 0.0), "bfc.w={}", bfc.rect.w);
    }

    #[test]
    fn bfc_beside_visible_floats_is_not_shifted() {
        // Control: when every overlapping float is visible to the
        // pre-layout point band, the post-layout re-test is a no-op.
        let l = layout_html(
            r#"<div class="w"><div class="f0"></div><div class="f1"></div><div class="bfc"><div class="in"></div></div></div>"#,
            "body { margin: 0; } .w { width: 400px; } \
             .f0 { float: left; width: 400px; height: 150px; } \
             .f1 { float: left; width: 100px; height: 100px; } \
             .bfc { overflow: hidden; } .in { height: 30px; }",
        );
        let w = &l.children[0].children[0];
        let bfc = &w.children[2];
        // Clears below f0 (cy=150) and narrows beside f1, as before.
        assert!(approx(bfc.rect.y, 150.0), "bfc.y={}", bfc.rect.y);
        assert!(approx(bfc.rect.x, 100.0), "bfc.x={}", bfc.rect.x);
        assert!(approx(bfc.rect.w, 300.0), "bfc.w={}", bfc.rect.w);
    }
}

#[cfg(test)]
mod box_sizing_override_tests {
    //! `width_override` carries a content-box width, but `resolve_metrics`
    //! clamped it with `clamp_w`, which applies min/max in the box-sizing's
    //! own terms (border-box terms under `box-sizing: border-box`).
    use crate::{
        BoxType, FixedFontMetrics, FontFamilyId, FontMetrics, IntrinsicDims, LayoutBox, Size,
        layout,
    };

    const VP: Size = Size { w: 800.0, h: 600.0 };

    fn layout_html(html: &str, css: &str) -> LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        layout(&doc, &[&ss], VP, &FixedFontMetrics)
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    fn find_box(l: &LayoutBox, bt: BoxType) -> &LayoutBox {
        let mut stack = vec![l];
        while let Some(b) = stack.pop() {
            if b.box_type == bt {
                return b;
            }
            stack.extend(b.children.iter());
        }
        panic!("no {bt:?} box")
    }

    #[test]
    fn border_box_table_min_width_clamps_override_in_content_terms() {
        // Repro: `width: auto` tables shrink-to-fit through the override
        // arm. `min-width: 300px` under `border-box` with 10px borders
        // must floor the *content* at 280 (border box 300), not clamp
        // the content-box override in border-box terms (content 300,
        // border box 320).
        let l = layout_html(
            "<table><tr><td>x</td></tr></table>",
            "body { margin: 0; } \
             table { box-sizing: border-box; width: auto; min-width: 300px; \
                     border: 10px solid black; border-spacing: 0; } \
             td { font-size: 10px; padding: 0; border: none; }",
        );
        let table = find_box(&l, BoxType::Table);
        assert!(
            approx(table.rect.w, 300.0),
            "table border-box w={}",
            table.rect.w
        );
    }

    #[test]
    fn specified_width_border_box_unaffected() {
        // Control: the plain specified-width arm already clamps in
        // border-box terms and then strips extras, so it must not change.
        let l = layout_html(
            "<table><tr><td>x</td></tr></table>",
            "body { margin: 0; } \
             table { box-sizing: border-box; width: 200px; \
                     border: 10px solid black; border-spacing: 0; } \
             td { font-size: 10px; padding: 0; border: none; }",
        );
        let table = find_box(&l, BoxType::Table);
        assert!(
            approx(table.rect.w, 200.0),
            "table border-box w={}",
            table.rect.w
        );
    }

    #[test]
    fn inline_block_specified_width_floors_min_content() {
        // css-sizing-3 §5.2: an atomic inline-level box with a definite
        // specified width contributes exactly that width to the
        // min-content size (WPT
        // css/css-tables/width-distribution/computing-table-width-0.html:
        // a caption's `<div style="display:inline-block; width:100px">`
        // contributes 100px to CAPMIN). Previously the catch-all arm
        // dropped the specified width entirely.
        let (doc, _) = bowser_dom::Document::parse_html(b"<div class=\"ib\">a b</div>");
        let ss = bowser_css::parse_stylesheet(
            ".ib { display: inline-block; width: 100px; font-size: 10px; \
              padding: 0; border: none; }",
        );
        let ua = bowser_css::default_ua_stylesheet();
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut ib = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::InlineBlock {
                ib = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let ib = ib.expect("inline-block box");
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, ib, &fonts, &mut scratch);
        // Specified 100px floors the children's intrinsic contribution
        // (longest word is a single 10px glyph, far narrower).
        assert!(approx(mc, 100.0), "min-content {mc}");
    }

    #[test]
    fn inline_block_border_box_width_strips_own_padding_border() {
        // `min_content_width` returns content-box width (the caller adds
        // the box's own extras), so under `border-box` the 100px
        // specified width minus 10px padding per side floors at 80.
        let (doc, _) = bowser_dom::Document::parse_html(b"<div class=\"ib\">a b</div>");
        let ss = bowser_css::parse_stylesheet(
            ".ib { display: inline-block; width: 100px; box-sizing: border-box; \
              font-size: 10px; padding: 10px; border: none; }",
        );
        let ua = bowser_css::default_ua_stylesheet();
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut ib = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::InlineBlock {
                ib = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let ib = ib.expect("inline-block box");
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, ib, &fonts, &mut scratch);
        assert!(approx(mc, 80.0), "min-content {mc}");
    }

    #[test]
    fn inline_block_auto_width_still_uses_children() {
        // Control: with `width: auto` the specified-width floor does not
        // apply; the contribution is the max of the children's intrinsic
        // widths, as before.
        let (doc, _) = bowser_dom::Document::parse_html(b"<div class=\"ib\">a b</div>");
        let ss = bowser_css::parse_stylesheet(
            ".ib { display: inline-block; font-size: 10px; padding: 0; border: none; }",
        );
        let ua = bowser_css::default_ua_stylesheet();
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut ib = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::InlineBlock {
                ib = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let ib = ib.expect("inline-block box");
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, ib, &fonts, &mut scratch);
        let word = fonts.advance("a", 10.0, FontFamilyId::DEFAULT);
        assert!(approx(mc, word), "min-content {mc}, word {word}");
    }

    #[test]
    fn inline_block_calc_with_percentage_treated_as_auto_for_min_content() {
        // css-sizing-3 §5.2.1 rule 1: for a non-replaced box, a width
        // specified as an expression containing a percentage is cyclic,
        // so the entire value behaves as the initial value (`auto`) —
        // `calc(20px + 50%)` floors nothing, not even the 20px length
        // part. Previously the length part floored the min-content at
        // 20px.
        let (doc, _) = bowser_dom::Document::parse_html(b"<div class=\"ib\">a b</div>");
        let ss = bowser_css::parse_stylesheet(
            ".ib { display: inline-block; width: calc(20px + 50%); font-size: 10px; \
              padding: 0; border: none; }",
        );
        let ua = bowser_css::default_ua_stylesheet();
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut ib = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::InlineBlock {
                ib = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let ib = ib.expect("inline-block box");
        assert!(forest.nodes[ib].intrinsic.is_none());
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, ib, &fonts, &mut scratch);
        let word = fonts.advance("a", 10.0, FontFamilyId::DEFAULT);
        assert!(approx(mc, word), "min-content {mc}, word {word}");
    }

    #[test]
    fn inline_block_calc_with_percentage_replaced_resolves_against_zero() {
        // Control for the rule-1 test: resolve-against-zero is correct
        // for replaced boxes (css-sizing-3 §5.2.1 rule 3), so an `<img>`
        // with `width: calc(20px + 50%)` still floors at the 20px length
        // part.
        let (doc, _) = bowser_dom::Document::parse_html(b"<img class=\"i\">");
        let root = doc.document_element().expect("document element");
        let body = doc
            .children(root)
            .into_iter()
            .find(|&c| {
                matches!(&doc.kind(c), bowser_dom::NodeKind::Element(d) if d.tag_name == "body")
            })
            .expect("body");
        let img = doc
            .children(body)
            .into_iter()
            .find(|&c| {
                matches!(&doc.kind(c), bowser_dom::NodeKind::Element(d) if d.tag_name == "img")
            })
            .expect("img");
        let ss = bowser_css::parse_stylesheet(".i { width: calc(20px + 50%); }");
        let ua = bowser_css::default_ua_stylesheet();
        let mut intrinsics = std::collections::HashMap::new();
        intrinsics.insert(img, IntrinsicDims::from_size(100.0, 50.0));
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &intrinsics,
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut ib = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::InlineBlock {
                ib = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let ib = ib.expect("inline-block box");
        assert!(forest.nodes[ib].intrinsic.is_some());
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, ib, &fonts, &mut scratch);
        assert!(approx(mc, 20.0), "min-content {mc}");
    }

    #[test]
    fn inline_block_definite_width_not_grown_by_overflowing_content() {
        // css-sizing-3 §5.2: the min-content contribution is exactly the
        // definite specified width, even when the content is wider — the
        // hypothetical float lays the box out at its definite width, and
        // the overflow does not grow it. The longest word here is 6px
        // (`FixedFontMetrics` advances 0.6em), wider than the 5px width.
        let (doc, _) = bowser_dom::Document::parse_html(b"<div class=\"ib\">a b</div>");
        let ss = bowser_css::parse_stylesheet(
            ".ib { display: inline-block; width: 5px; font-size: 10px; \
              padding: 0; border: none; }",
        );
        let ua = bowser_css::default_ua_stylesheet();
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut ib = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::InlineBlock {
                ib = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let ib = ib.expect("inline-block box");
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, ib, &fonts, &mut scratch);
        assert!(approx(mc, 5.0), "min-content {mc}");
    }

    /// Build a forest for a single `.ib` inline-block div and return the
    /// forest plus the inline-block's arena index.
    fn ib_forest(text: &str, css: &str) -> (crate::boxes::Forest, usize) {
        let html = format!("<div class=\"ib\">{text}</div>");
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let ua = bowser_css::default_ua_stylesheet();
        let Some((forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut ib = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::InlineBlock {
                ib = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        (forest, ib.expect("inline-block box"))
    }

    /// `(min-content, max-content)` content widths of the `.ib` box.
    fn ib_intrinsic_widths(forest: &mut crate::boxes::Forest, ib: usize) -> (f32, f32) {
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(forest, ib, &fonts, &mut scratch);
        let mut cache = super::PrefWidthCache::new(forest.nodes.len());
        let pc = super::preferred_width(forest, ib, &fonts, &mut cache);
        (mc, pc)
    }

    #[test]
    fn inline_block_definite_width_symmetric_min_max_content() {
        // The wave-38 specified-width arm landed on `min_content_width`
        // only: a 200px inline-block with 300px kids contributed 200 to
        // min-content but 300 to max-content, and the bad max-content
        // value fed shrink-to-fit, block-table auto width, and flex/grid
        // base sizes. css-sizing-3 §5.2 applies to both intrinsic sizes,
        // so both must be 200.
        let word = "a".repeat(50); // 50 * 6px = 300px at 10px font-size
        let (mut forest, ib) = ib_forest(
            &word,
            ".ib { display: inline-block; width: 200px; font-size: 10px; \
              padding: 0; border: none; }",
        );
        let (mc, pc) = ib_intrinsic_widths(&mut forest, ib);
        assert!(approx(mc, 200.0), "min-content {mc}");
        assert!(approx(pc, 200.0), "max-content {pc}");
    }

    /// Build a forest for a standalone `display: table-cell` div (no
    /// table ancestor — the direct cell-subtree measurement case) and
    /// return it plus the cell's arena index.
    fn cell_forest(text: &str, css: &str) -> (crate::boxes::Forest, usize) {
        let html = format!("<div class=\"cell\">{text}</div>");
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let ua = bowser_css::default_ua_stylesheet();
        let Some((forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut cell = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::TableCell {
                cell = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        (forest, cell.expect("table-cell box"))
    }

    #[test]
    fn table_cell_min_content_definite_width_floors() {
        // Symmetric with the max-content side's `definite_width_floor`
        // (`max(MCW, W)`): a standalone cell with a 200px definite
        // width and 12px of content contributes 200 to min-content
        // too. Before the fix the min-content dispatch's catch-all
        // skipped the floor for cells and reported the 12px content
        // width.
        let (mut forest, cell) = cell_forest(
            "hi",
            ".cell { display: table-cell; width: 200px; font-size: 10px; padding: 0; border: none; }",
        );
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, cell, &fonts, &mut scratch);
        assert!(approx(mc, 200.0), "cell min-content {mc}");
        let mut cache = super::PrefWidthCache::new(forest.nodes.len());
        let pc = super::preferred_width(&mut forest, cell, &fonts, &mut cache);
        assert!(approx(pc, 200.0), "cell max-content {pc}");
    }

    #[test]
    fn table_cell_min_content_floor_is_a_floor_not_a_cap() {
        // The floor is `max(definite, content)`: content wider than
        // the definite width still wins (unlike a block's definite
        // width, which contributes exactly).
        let word = "a".repeat(50); // 300px at 10px font-size
        let (mut forest, cell) = cell_forest(
            &word,
            ".cell { display: table-cell; width: 200px; font-size: 10px; padding: 0; border: none; }",
        );
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, cell, &fonts, &mut scratch);
        assert!(approx(mc, 300.0), "cell min-content {mc}");
    }

    #[test]
    fn table_cell_in_table_min_content_not_floored() {
        // Context pin for the guard above: a cell inside a real table
        // keeps the CSS Tables 3 §3.8.3 rule even under direct
        // measurement — the specified width floors the column maximum,
        // not the minimum — because the table passes call
        // `min_content_width` on cells and apply §3.8.3 themselves
        // (cf. `specified_cell_width_does_not_floor_column_minimum`).
        let (doc, _) =
            bowser_dom::Document::parse_html(b"<table><tr><td class=\"cell\">hi</td></tr></table>");
        let ss = bowser_css::parse_stylesheet(
            ".cell { width: 200px; font-size: 10px; padding: 0; border: none; }",
        );
        let ua = bowser_css::default_ua_stylesheet();
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut cell = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::TableCell {
                cell = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let cell = cell.expect("table-cell box");
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, cell, &fonts, &mut scratch);
        assert!(approx(mc, 12.0), "in-table cell min-content {mc}");
    }

    #[test]
    fn nested_table_cell_inside_td_gets_definite_floor() {
        // C6: the guard used to walk the ENTIRE parent chain, so a
        // `display: table-cell` box nested inside a real `td` through
        // ordinary divs reported "in table context" from the far-up
        // table — though it belongs to no grid (it would get its own
        // anonymous table) — and direct measurement lost its definite
        // floor (200px width measured as the 12px content). The walk
        // now goes up only through row/row-group structure and fails
        // on the intervening block (and on the outer cell itself).
        let (doc, _) = bowser_dom::Document::parse_html(
            b"<table><tr><td><div><div class=\"inner\">hi</div></div></td></tr></table>",
        );
        let ss = bowser_css::parse_stylesheet(
            ".inner { display: table-cell; width: 200px; font-size: 10px; \
             padding: 0; border: none; } \
             td { font-size: 10px; padding: 0; border: none; }",
        );
        let ua = bowser_css::default_ua_stylesheet();
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        // The inner cell is the TableCell whose specified width is
        // the 200px one (the outer `td` has no specified width).
        let mut inner = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::TableCell
                && super::definite_specified_content_width(&forest, i).is_some()
            {
                inner = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let inner = inner.expect("inner table-cell box");
        assert!(
            !super::cell_in_table_context(&forest, inner),
            "nested cell must not report table context"
        );
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, inner, &fonts, &mut scratch);
        assert!(approx(mc, 200.0), "nested cell min-content {mc} != 200");
    }

    #[test]
    fn bare_cell_under_row_group_reports_table_context() {
        // Guard shape pin: a bare cell directly under a row group
        // (no intervening row — the grid collection groups those
        // itself) still counts as in-table, as does a cell under an
        // anonymous row structure.
        let (doc, _) = bowser_dom::Document::parse_html(
            b"<table><tbody><td class=\"cell\">hi</td></tbody></table>",
        );
        let ss = bowser_css::parse_stylesheet(
            ".cell { width: 200px; font-size: 10px; padding: 0; border: none; }",
        );
        let ua = bowser_css::default_ua_stylesheet();
        let Some((mut forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut cell = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == BoxType::TableCell {
                cell = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let cell = cell.expect("table-cell box");
        assert!(super::cell_in_table_context(&forest, cell));
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, cell, &fonts, &mut scratch);
        assert!(approx(mc, 12.0), "bare in-table cell min-content {mc}");
    }

    /// Build a forest for `html`/`css` and return it with its root box.
    fn table_forest(html: &[u8], css: &str) -> (crate::boxes::Forest, usize) {
        let (doc, _) = bowser_dom::Document::parse_html(html);
        let ss = bowser_css::parse_stylesheet(css);
        let ua = bowser_css::default_ua_stylesheet();
        let Some((forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        (forest, root)
    }

    /// First node under `root` satisfying `pred` (preorder).
    fn find_node(
        forest: &crate::boxes::Forest,
        root: usize,
        mut pred: impl FnMut(&crate::boxes::Forest, usize) -> bool,
    ) -> Option<usize> {
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if pred(forest, i) {
                return Some(i);
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        None
    }

    #[test]
    fn flex_display_never_produces_table_cell() {
        // Reachability pin for the min-content flex arm's removed cell
        // guard: a `td` with `display: flex` becomes a `Block` box
        // (never `TableCell`), while a plain `td` stays `TableCell`
        // but has no flex display — so `box_type == TableCell &&
        // flex_kind().is_some()` is unrepresentable.
        let (forest, root) = table_forest(
            b"<table><tr><td class=\"a\">hi</td><td class=\"b\">hi</td></tr></table>",
            ".a { display: flex; width: 200px; font-size: 10px; padding: 0; border: none; } \
             .b { font-size: 10px; padding: 0; border: none; } \
             td { font-size: 10px; padding: 0; border: none; }",
        );
        let flex_td = find_node(&forest, root, |f, i| {
            f.nodes[i].box_type == BoxType::Block
                && super::definite_specified_content_width(f, i) == Some(200.0)
        })
        .expect("display:flex td becomes a Block box");
        assert_eq!(super::flex_kind(&forest, flex_td), Some(true));
        let plain_td = find_node(&forest, root, |f, i| {
            f.nodes[i].box_type == BoxType::TableCell
        })
        .expect("plain td stays a TableCell");
        assert_eq!(super::flex_kind(&forest, plain_td), None);
    }

    #[test]
    fn display_flex_td_preferred_widths() {
        // `display: flex` on the `td` itself: it is a block-level flex
        // container, not a cell, so the definite width contributes
        // exactly to both min- and max-content — the (removed)
        // min-content cell guard and the max-content
        // `definite_width_floor` cell arm can never disagree, because
        // neither is reachable for a flex container.
        let (mut forest, root) = table_forest(
            b"<table><tr><td class=\"cell\">hello world</td></tr></table>",
            ".cell { display: flex; width: 200px; font-size: 10px; padding: 0; border: none; }",
        );
        let flex_td = find_node(&forest, root, |f, i| {
            f.nodes[i].box_type == BoxType::Block && super::flex_kind(f, i).is_some()
        })
        .expect("display:flex td");
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, flex_td, &fonts, &mut scratch);
        assert!(approx(mc, 200.0), "display:flex td min-content {mc}");
        let mut cache = super::PrefWidthCache::new(forest.nodes.len());
        let pc = super::preferred_width(&mut forest, flex_td, &fonts, &mut cache);
        assert!(approx(pc, 200.0), "display:flex td max-content {pc}");
    }

    #[test]
    fn flex_in_table_cell_preferred_widths() {
        // A flex container nested inside a real table cell with a
        // definite width contributes exactly that width to both
        // min- and max-content (block precedence, like any
        // block-level box) — symmetric across the two flex arms.
        let (mut forest, root) = table_forest(
            b"<table><tr><td><div class=\"f\">hello world</div></td></tr></table>",
            ".f { display: flex; width: 200px; font-size: 10px; padding: 0; border: none; } \
             td { font-size: 10px; padding: 0; border: none; }",
        );
        let flex_div = find_node(&forest, root, |f, i| {
            f.nodes[i].box_type == BoxType::Block && super::flex_kind(f, i).is_some()
        })
        .expect("flex div in cell");
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, flex_div, &fonts, &mut scratch);
        assert!(approx(mc, 200.0), "flex-in-cell min-content {mc}");
        let mut cache = super::PrefWidthCache::new(forest.nodes.len());
        let pc = super::preferred_width(&mut forest, flex_div, &fonts, &mut cache);
        assert!(approx(pc, 200.0), "flex-in-cell max-content {pc}");
    }

    #[test]
    fn inline_block_definite_width_shrink_to_fit() {
        // End-to-end: a shrink-to-fitting float wraps an inline-block
        // with `width: 200px` and 300px kids. The float's preferred width
        // must come from the definite 200px, not the 300px kids.
        let word = "a".repeat(50);
        let html = format!("<div class=\"f\"><div class=\"ib\">{word}</div></div>");
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let f_id = doc
            .get_elements_by_class_name("f")
            .into_iter()
            .next()
            .expect("div.f");
        let ss = bowser_css::parse_stylesheet(
            "body { margin: 0; } .f { float: left; } \
             .ib { display: inline-block; width: 200px; font-size: 10px; \
                   padding: 0; border: none; }",
        );
        let l = layout(&doc, &[&ss], VP, &FixedFontMetrics);
        let mut found = None;
        let mut stack = vec![&l];
        while let Some(b) = stack.pop() {
            if b.node == Some(f_id) {
                found = Some(b);
                break;
            }
            stack.extend(b.children.iter());
        }
        let f = found.expect("float box");
        assert!(approx(f.rect.w, 200.0), "float w={}", f.rect.w);
    }

    #[test]
    fn inline_block_auto_width_preferred_still_uses_children() {
        // Control: with `width: auto` the new `InlineBlock` arm in
        // `preferred_width` must fall through to the generic
        // max-of-runs sizing, exactly as the old catch-all arm did.
        let word = "a".repeat(50);
        let (mut forest, ib) = ib_forest(
            &word,
            ".ib { display: inline-block; font-size: 10px; \
              padding: 0; border: none; }",
        );
        let (mc, pc) = ib_intrinsic_widths(&mut forest, ib);
        assert!(approx(mc, 300.0), "min-content {mc}");
        assert!(approx(pc, 300.0), "max-content {pc}");
    }

    #[test]
    fn inline_block_calc_with_percentage_preferred_treated_as_auto() {
        // Preferred-width counterpart of
        // `inline_block_calc_with_percentage_treated_as_auto_for_min_content`:
        // a percentage-containing `calc()` is cyclic (§5.2.1 rule 1) and
        // behaves as `auto` in both intrinsic sizes.
        let (mut forest, ib) = ib_forest(
            "a b",
            ".ib { display: inline-block; width: calc(20px + 50%); font-size: 10px; \
              padding: 0; border: none; }",
        );
        let (mc, pc) = ib_intrinsic_widths(&mut forest, ib);
        let word = FixedFontMetrics.advance("a", 10.0, FontFamilyId::DEFAULT);
        assert!(approx(mc, word), "min-content {mc}, word {word}");
        // Preferred width advances the whole text run, unsplit.
        let full = FixedFontMetrics.advance("a b", 10.0, FontFamilyId::DEFAULT);
        assert!(approx(pc, full), "max-content {pc}, full {full}");
    }

    #[test]
    fn inline_block_calc_zero_percent_treated_as_auto() {
        // CSS Values 4 §8.1.4: cyclic-ness is syntactic. `calc(10px +
        // 0%)` folds to `Math { pct: 0.0, px: 10.0 }`, but it contained
        // a percentage leaf (`had_pct`), so it behaves as `auto` in
        // intrinsic sizing — not as a definite 10px. Under the old
        // value-based `pct != 0.0` gate this wrongly contributed 10px.
        let word = "a".repeat(50); // 300px at 10px font-size
        let (mut forest, ib) = ib_forest(
            &word,
            ".ib { display: inline-block; width: calc(10px + 0%); font-size: 10px; \
              padding: 0; border: none; }",
        );
        let (mc, pc) = ib_intrinsic_widths(&mut forest, ib);
        assert!(approx(mc, 300.0), "min-content {mc}");
        assert!(approx(pc, 300.0), "max-content {pc}");
    }

    #[test]
    fn inline_block_min_max_width_clamp_intrinsic_widths() {
        // `min-width`/`max-width` clamp the definite specified width in
        // the box-sizing's own terms; min-width wins when both constrain.
        let word = "a".repeat(50);
        let base = ".ib { display: inline-block; width: 200px; font-size: 10px; \
                    padding: 0; border: none;";
        let cases = [
            ("min-width floors", "min-width: 250px; }", 250.0),
            ("max-width caps", "max-width: 150px; }", 150.0),
            (
                "min-width wins when over-constrained",
                "min-width: 250px; max-width: 150px; }",
                250.0,
            ),
        ];
        for (name, extra, expected) in cases {
            let (mut forest, ib) = ib_forest(&word, &format!("{base} {extra}"));
            let (mc, pc) = ib_intrinsic_widths(&mut forest, ib);
            assert!(approx(mc, expected), "{name}: min-content {mc}");
            assert!(approx(pc, expected), "{name}: max-content {pc}");
        }
        // Border-box: the clamp applies to the border-box width, then the
        // box's own padding is stripped, so 200px - 2*10px = 180px.
        let (mut forest, ib) = ib_forest(
            &word,
            ".ib { display: inline-block; width: 200px; box-sizing: border-box; \
              font-size: 10px; padding: 0 10px; border: none; }",
        );
        let (mc, pc) = ib_intrinsic_widths(&mut forest, ib);
        assert!(approx(mc, 180.0), "border-box min-content {mc}");
        assert!(approx(pc, 180.0), "border-box max-content {pc}");
    }

    #[test]
    fn table_cell_min_content_aspect_ratio_transfers() {
        // R3-4: a standalone `display: table-cell` with
        // `aspect-ratio` and a definite height transfers the height
        // through the ratio in both intrinsic passes. Before the fix
        // the min-content arm's standalone-cell branch fell through to
        // the (empty) kids and reported 0 while max-content reported
        // 100.
        let (mut forest, cell) = cell_forest(
            "",
            ".cell { display: table-cell; aspect-ratio: 2/1; height: 50px; \
              font-size: 10px; padding: 0; border: none; }",
        );
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mc = super::min_content_width(&mut forest, cell, &fonts, &mut scratch);
        assert!(approx(mc, 100.0), "cell min-content {mc}");
        let mut cache = super::PrefWidthCache::new(forest.nodes.len());
        let pc = super::preferred_width(&mut forest, cell, &fonts, &mut cache);
        assert!(approx(pc, 100.0), "cell max-content {pc}");
    }
}

#[cfg(test)]
mod float_cb_top_clamp_tests {
    //! CSS 2.1 §9.5.1: a float's top may not be higher than its
    //! containing block's top.
    use crate::{FixedFontMetrics, LayoutBox, Size, layout};

    const VP: Size = Size { w: 800.0, h: 600.0 };

    fn layout_html(html: &str, css: &str) -> LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        layout(&doc, &[&ss], VP, &FixedFontMetrics)
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn float_top_clamped_to_containing_block_top() {
        // A first child with a negative top margin pulls its border box
        // above the parent's content top (the border blocks margin
        // collapsing, so the negative margin applies in full); the flow
        // cursor the float is encountered at then sits above the
        // content-box top. §9.5.1 still forbids the float's margin-box
        // top from rising above the containing block's top: it clamps to
        // the content-box top instead of following the cursor down.
        // Without the clamp the float's margin-box top would land at
        // 5 - 50 + 10 = -35.
        let l = layout_html(
            r#"<div class="w"><div class="a"></div><div class="f"></div></div>"#,
            "body { margin: 0; } \
             .w { width: 400px; border: 5px solid black; } \
             .a { margin-top: -50px; height: 10px; } \
             .f { float: left; width: 100px; height: 50px; }",
        );
        let w = &l.children[0].children[0];
        // `.w`'s border box starts at y=0; the content-box top is below
        // the 5px top border.
        let content_top = w.rect.y + 5.0;
        let f = &w.children[1];
        assert!(
            approx(f.rect.y, content_top),
            "f.y={} content_top={}",
            f.rect.y,
            content_top
        );
    }

    #[test]
    fn float_not_higher_than_earlier_inflow_block_top() {
        // CSS 2.1 §9.5.1 rule 5: the float's outer top may not be
        // higher than the outer top of any earlier in-flow block box.
        // `.x` sits low (border top=105, no margin, so outer top=105);
        // `.m` is then pulled far above it by a negative top margin,
        // dragging the flow cursor to -385. Without the in-flow-top
        // floor the float would follow the cursor (and the
        // containing-block clamp) up to y=5. The binding floor is
        // `.m`'s own outer top: its border top (-395) minus its top
        // placement margin (-1000) is 605, `.x`'s bottom edge. (A
        // border-top floor stopped at `.x`'s 105 instead; rule 5
        // speaks of outer tops, so the margined `.m` binds.)
        let l = layout_html(
            r#"<div class="w"><div class="p"></div><div class="x"></div><div class="m"></div><div class="f"></div></div>"#,
            "body { margin: 0; } \
             .w { width: 400px; border: 5px solid black; } \
             .p { height: 100px; } \
             .x { height: 500px; } \
             .m { margin-top: -1000px; height: 10px; } \
             .f { float: left; width: 100px; height: 50px; }",
        );
        let w = &l.children[0].children[0];
        let (x, m, f) = (&w.children[1], &w.children[2], &w.children[3]);
        assert!(approx(x.rect.y, 105.0), "x.y={}", x.rect.y);
        assert!(approx(m.rect.y, -395.0), "m.y={}", m.rect.y);
        assert!(
            approx(f.rect.y, 605.0),
            "f.y={} (expected .m outer top 605)",
            f.rect.y
        );
    }

    #[test]
    fn float_not_higher_than_margined_inflow_block_outer_top() {
        // CSS 2.1 §9.5.1 rules 5/6, outer-top form: `.m`'s negative
        // top margin pulls its border box up to y=55, above the point
        // the float is encountered (cursor at `.m`'s bottom, 65), but
        // `.m`'s outer (margin) top stays at 105 — the edge its
        // -50px margin reaches up to, level with `.p`'s bottom. A
        // border-top floor would let the float sit at the cursor
        // (y=65); the outer-top floor holds it at y=105.
        let l = layout_html(
            r#"<div class="w"><div class="p"></div><div class="m"></div><div class="f"></div></div>"#,
            "body { margin: 0; } \
             .w { width: 400px; border: 5px solid black; } \
             .p { height: 100px; } \
             .m { margin-top: -50px; height: 10px; } \
             .f { float: left; width: 100px; height: 50px; }",
        );
        let w = &l.children[0].children[0];
        let (m, f) = (&w.children[1], &w.children[2]);
        assert!(approx(m.rect.y, 55.0), "m.y={}", m.rect.y);
        assert!(
            approx(f.rect.y, 105.0),
            "f.y={} (expected .m outer top 105, not border top 55 / cursor 65)",
            f.rect.y
        );
    }

    #[test]
    fn float_top_clamp_is_noop_in_normal_flow() {
        // Control: without the negative-margin cursor excursion the
        // clamp never engages and normal float placement is untouched.
        let l = layout_html(
            r#"<div class="w"><div class="a"></div><div class="f"></div></div>"#,
            "body { margin: 0; } \
             .w { width: 400px; border: 5px solid black; } \
             .a { height: 60px; } \
             .f { float: left; width: 100px; height: 50px; }",
        );
        let w = &l.children[0].children[0];
        let f = &w.children[1];
        // `.a` occupies [5, 65); the float sits right below it.
        assert!(approx(f.rect.y, 65.0), "f.y={}", f.rect.y);
    }
}

#[cfg(test)]
mod height_calc_cyclic_tests {
    //! CSS 2.1 §10.5-10.7 + CSS Values 4 §8.1.4: `height`, `min-height`
    //! and `max-height` given as a `calc()` containing a percentage
    //! leaf are cyclic against an indefinite containing-block height
    //! and behave as `auto` / `0` / `none` respectively — never
    //! resolving against a zero base. Mirrors the wave-48 width-axis
    //! gates (`definite_specified_content_width`, flex-basis).
    use crate::{FixedFontMetrics, LayoutBox, Size, layout};

    const VP: Size = Size { w: 800.0, h: 600.0 };

    fn layout_html(html: &str, css: &str) -> LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        layout(&doc, &[&ss], VP, &FixedFontMetrics)
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// Border-box height of `.a`, the only child of `.w`.
    fn a_h(css: &str) -> f32 {
        let l = layout_html(r#"<div class="w"><div class="a"></div></div>"#, css);
        l.children[0].children[0].children[0].rect.h
    }

    #[test]
    fn calc_pct_height_with_indefinite_cb_behaves_as_auto() {
        // `height: calc(10px + 50%)` against an auto-height parent is
        // cyclic: the empty box sizes as `auto` (0), not as the 10px a
        // zero-base resolution would give.
        let h = a_h("body { margin: 0; } .w { width: 400px; } .a { height: calc(10px + 50%); }");
        assert!(approx(h, 0.0), "h={h}");
    }

    #[test]
    fn calc_pct_height_with_definite_cb_resolves() {
        // Control: with a definite 200px containing-block height the
        // same calc resolves normally (10 + 50% of 200 = 110).
        let h = a_h("body { margin: 0; } .w { width: 400px; height: 200px; } \
             .a { height: calc(10px + 50%); }");
        assert!(approx(h, 110.0), "h={h}");
    }

    #[test]
    fn calc_zero_pct_height_is_cyclic() {
        // Syntactic cyclic-ness: `calc(10px + 0%)` folds to
        // `Math { pct: 0.0, px: 10.0 }` but contained a `%` leaf, so
        // it behaves as `auto` against an indefinite base — not 10px.
        let h = a_h("body { margin: 0; } .w { width: 400px; } .a { height: calc(10px + 0%); }");
        assert!(approx(h, 0.0), "h={h}");
    }

    #[test]
    fn pure_px_calc_height_still_resolves() {
        // Control: a `calc()` with no percentage leaf is not cyclic
        // and resolves against any base.
        let h = a_h("body { margin: 0; } .w { width: 400px; } .a { height: calc(10px + 5px); }");
        assert!(approx(h, 15.0), "h={h}");
    }

    #[test]
    fn calc_pct_min_height_with_indefinite_cb_behaves_as_zero() {
        // `min-height: calc(50% + 10px)` against an indefinite base is
        // cyclic and behaves as 0: the empty box stays 0 tall instead
        // of flooring at the zero-base 10px.
        let h =
            a_h("body { margin: 0; } .w { width: 400px; } .a { min-height: calc(50% + 10px); }");
        assert!(approx(h, 0.0), "h={h}");
    }

    #[test]
    fn calc_pct_min_height_with_definite_cb_floors() {
        // Control: with a definite 200px parent the min-height
        // resolves to 110px and floors the empty box.
        let h = a_h("body { margin: 0; } .w { width: 400px; height: 200px; } \
             .a { min-height: calc(50% + 10px); }");
        assert!(approx(h, 110.0), "h={h}");
    }

    #[test]
    fn calc_pct_max_height_with_indefinite_cb_behaves_as_none() {
        // `max-height: calc(50% + 10px)` against an indefinite base is
        // cyclic and behaves as `none`: the 150px height is not
        // clamped to the 10px a zero-base resolution would give.
        let h = a_h("body { margin: 0; } .w { width: 400px; } \
             .a { height: 150px; max-height: calc(50% + 10px); }");
        assert!(approx(h, 150.0), "h={h}");
    }

    #[test]
    fn calc_pct_max_height_with_definite_cb_clamps() {
        // Control: with a definite 200px parent the max-height
        // resolves to 110px and clamps the 150px height.
        let h = a_h("body { margin: 0; } .w { width: 400px; height: 200px; } \
             .a { height: 150px; max-height: calc(50% + 10px); }");
        assert!(approx(h, 110.0), "h={h}");
    }
}

#[cfg(test)]
mod grid_min_content_tests {
    //! css-grid-1 §12.5 / css-sizing-3 §5.2: a grid container's
    //! min-content contribution is the sum of its column tracks'
    //! minima plus gaps (via `layout_grid::grid_min_width`), and
    //! `inline-grid` / `inline-table` route through the same
    //! track/table paths as their block-level counterparts instead of
    //! the generic max-of-children `InlineBlock` arm.
    use super::min_content_width;
    use crate::BoxType;
    use crate::FixedFontMetrics;
    use crate::style::Display;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// Build a forest for `html`/`css` and return it with the arena
    /// index of the first box (in child order) matching `pred`.
    fn forest_find(
        html: &str,
        css: &str,
        pred: impl Fn(&crate::boxes::Node) -> bool,
    ) -> (crate::boxes::Forest, usize) {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let ua = bowser_css::default_ua_stylesheet();
        let Some((forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut found = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if pred(&forest.nodes[i]) {
                found = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().rev().copied());
        }
        (forest, found.expect("matching box"))
    }

    /// Min-content content width of `idx` (a fresh scratch buffer).
    fn forest_min_w(forest: &mut crate::boxes::Forest, idx: usize) -> f32 {
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        min_content_width(forest, idx, &fonts, &mut scratch)
    }

    #[test]
    fn grid_min_content_fixed_tracks_and_gap() {
        // Fixed tracks resolve against a zero base and the gaps add on:
        // the min-content contribution is the track sum (160), not the
        // max of the children's contributions (24) the old generic arm
        // produced.
        let html = r#"<div class="g"><div class="i">aaaa</div><div class="i">bb</div></div>"#;
        let css = ".g { display: grid; grid-template-columns: 100px 50px; column-gap: 10px; } \
             .i { font-size: 10px; }";
        let (mut forest, g) = forest_find(html, css, |n| n.box_type == BoxType::Grid);
        assert!(approx(forest_min_w(&mut forest, g), 160.0));
    }

    #[test]
    fn grid_min_content_fr_and_auto_tracks_sum_minima() {
        // Content-sized tracks (`auto`, `fr` whose minimum is the
        // content size) measure their items' min-content contributions:
        // 1fr takes the 24px word, auto the 12px word — 36 in total, not
        // the 24px the max-of-children arm gave.
        let html = r#"<div class="g"><div class="i">aaaa bbbb</div><div class="i">cc</div></div>"#;
        let css = ".g { display: grid; grid-template-columns: 1fr auto; } \
             .i { font-size: 10px; }";
        let (mut forest, g) = forest_find(html, css, |n| n.box_type == BoxType::Grid);
        assert!(approx(forest_min_w(&mut forest, g), 36.0));
    }

    #[test]
    fn grid_min_content_spanning_item_spreads_deficit() {
        // Single-span items size first (12px in column 1); the spanning
        // item's 60px min-content contribution minus the span's current
        // 12px spreads equally over the two content-sized tracks, so the
        // total is 60 rather than 12.
        let html = r#"<div class="g"><div class="s">aaaaaaaaaa</div><div class="i">bb</div></div>"#;
        let css = ".g { display: grid; grid-template-columns: auto auto; } \
             .s { grid-column: 1 / 3; font-size: 10px; } \
             .i { font-size: 10px; }";
        let (mut forest, g) = forest_find(html, css, |n| n.box_type == BoxType::Grid);
        assert!(approx(forest_min_w(&mut forest, g), 60.0));
    }

    #[test]
    fn grid_min_content_definite_item_width_floors_track() {
        // A definite specified item width floors its track's minimum,
        // exactly as in the max-content pass: the empty 200px item
        // contributes 200px to its `auto` track.
        let html = r#"<div class="g"><div class="i"></div></div>"#;
        let css = ".g { display: grid; grid-template-columns: auto; } \
             .i { width: 200px; }";
        let (mut forest, g) = forest_find(html, css, |n| n.box_type == BoxType::Grid);
        assert!(approx(forest_min_w(&mut forest, g), 200.0));
    }

    #[test]
    fn inline_grid_min_content_uses_tracks() {
        // `inline-grid` is an `InlineBlock` with the grid inner layout:
        // it sizes by track like `grid` (36), not by the old
        // max-of-children `InlineBlock` arm (24).
        let html = r#"<div class="g"><div class="i">aaaa bbbb</div><div class="i">cc</div></div>"#;
        let css = ".g { display: inline-grid; grid-template-columns: auto auto; } \
             .i { font-size: 10px; }";
        let (mut forest, g) = forest_find(html, css, |n| n.data.display == Display::InlineGrid);
        assert!(approx(forest_min_w(&mut forest, g), 36.0));
    }

    #[test]
    fn inline_table_min_content_matches_table() {
        // `inline-table` takes the table min-content path: column minima
        // 24 + 12, identical to the block-level table (the old
        // `InlineBlock` arm gave the max-of-children 24).
        let html = r#"<table class="t"><tr><td>aaaa</td><td>bb</td></tr></table>"#;
        let cell_css = "td { font-size: 10px; padding: 0; border: none; }";
        let css = format!(".t {{ display: inline-table; border-spacing: 0; }} {cell_css}");
        let (mut forest, it) = forest_find(html, &css, |n| n.data.display == Display::InlineTable);
        let css = format!(".t {{ display: table; border-spacing: 0; }} {cell_css}");
        let (mut forest2, t) = forest_find(html, &css, |n| n.box_type == BoxType::Table);
        let mc_it = forest_min_w(&mut forest, it);
        let mc_t = forest_min_w(&mut forest2, t);
        assert!(approx(mc_it, 36.0), "inline-table {mc_it}");
        assert!(approx(mc_t, 36.0), "table {mc_t}");
        assert!(approx(mc_it, mc_t), "inline-table {mc_it} != table {mc_t}");
    }
}

/// w65: vertical intrinsic measurement beyond definite heights —
/// inline children (text runs, inline boxes) and auto-height block
/// content in vertical tables.
///
/// `FixedFontMetrics` advances 0.6px per char per px of font size and
/// reports line-height 1.2x, so at `font-size: 10px` one char is 6px
/// wide and the line-height strut is 12px. The tests assert the
/// *vertical* extents the new measure produces — several pinned values
/// differ from what the old horizontal fallback reported on the wrong
/// axis (noted per test).
#[cfg(test)]
mod w65_f_vintr_tests {
    use super::{
        PrefWidthCache, WalkBufs, min_content_width, vertical_cell_max_content,
        vertical_cell_min_content,
    };
    use crate::boxes::Forest;
    use crate::{BoxType, FixedFontMetrics};

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// Build a forest for `html`/`css` and return it with the arena
    /// index of the first box (in document order) matching `pred`.
    fn forest_find(
        html: &str,
        css: &str,
        pred: impl Fn(&crate::boxes::Node) -> bool,
    ) -> (Forest, usize) {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let ua = bowser_css::default_ua_stylesheet();
        let Some((forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &std::collections::HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut found = None;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if pred(&forest.nodes[i]) {
                found = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().rev().copied());
        }
        (forest, found.expect("matching box"))
    }

    /// First table cell in document order — the cell under test. The
    /// table carries `writing-mode: vertical-rl`, so the cell's inline
    /// axis is the physical vertical axis.
    fn forest_cell(html: &str, css: &str) -> (Forest, usize) {
        forest_find(html, css, |n| n.box_type == BoxType::TableCell)
    }

    /// Shared base CSS: vertical table, zeroed cell padding/border so
    /// measures are pure content extents.
    const VCELL: &str = ".t { writing-mode: vertical-rl; border-spacing: 0; } \
         td { font-size: 10px; padding: 0; border: none; }";

    fn cell_min(forest: &mut Forest, cell: usize) -> f32 {
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mut bufs = WalkBufs::new();
        vertical_cell_min_content(forest, cell, &fonts, &mut scratch, &mut bufs)
    }

    fn cell_max(forest: &mut Forest, cell: usize) -> f32 {
        let fonts = FixedFontMetrics;
        let mut scratch = Vec::new();
        let mut bufs = WalkBufs::new();
        let mut inner = WalkBufs::new();
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        vertical_cell_max_content(
            forest,
            cell,
            &fonts,
            &mut cache,
            &mut scratch,
            &mut bufs,
            &mut inner,
        )
    }

    #[test]
    fn inline_text_measures_vertical_extent() {
        // Inline text in a vertical cell: min-content is the longest
        // word's vertical run (4 chars * 6px = 24), max-content the
        // whole unwrapped line (9 chars * 6px = 54).
        let html = r#"<table class="t"><tr><td>aaaa bbbb</td></tr></table>"#;
        let (mut forest, cell) = forest_cell(html, VCELL);
        assert!(approx(cell_min(&mut forest, cell), 24.0));
        assert!(approx(cell_max(&mut forest, cell), 54.0));
    }

    #[test]
    fn inline_min_content_floored_by_line_height_strut() {
        // One short word: the longest run is 6px, but the line box is
        // the 12px strut (line-height 1.2 * 10px), so the vertical
        // min-content is 12. The old horizontal fallback reported 6.
        let html = r#"<table class="t"><tr><td>a</td></tr></table>"#;
        let (mut forest, cell) = forest_cell(html, VCELL);
        assert!(approx(cell_min(&mut forest, cell), 12.0));
        assert!(approx(cell_max(&mut forest, cell), 12.0));
    }

    #[test]
    fn pre_text_max_content_sums_forced_lines() {
        // `white-space: pre`: each forced break starts a new vertical
        // line box — max(12, 6) + max(12, 12) + max(12, 18) = 42. The
        // old horizontal max-content fallback reported
        // advance("a\nbb\nccc") = 48 on the wrong axis.
        let html = "<table class=\"t\"><tr><td class=\"p\">a\nbb\nccc</td></tr></table>";
        let css = [VCELL, " .p { white-space: pre; }"].concat();
        let (mut forest, cell) = forest_cell(html, &css);
        assert!(approx(cell_max(&mut forest, cell), 42.0));
        // Min-content: longest run 18px, above the strut.
        assert!(approx(cell_min(&mut forest, cell), 18.0));
    }

    #[test]
    fn inline_box_children_combine_as_one_run() {
        // `<b>` is an unbreakable run for min-content (children sum),
        // and joins the line for max-content: min = max(12, 12) = 12
        // floored by the strut, max = 12 + 12 + 12 = 36.
        let html = r#"<table class="t"><tr><td>a <b>bb</b> c</td></tr></table>"#;
        let (mut forest, cell) = forest_cell(html, VCELL);
        assert!(approx(cell_min(&mut forest, cell), 12.0));
        assert!(approx(cell_max(&mut forest, cell), 36.0));
    }

    #[test]
    fn auto_height_blocks_measure_content_recursively() {
        // Two nested auto-height divs (previously: the whole cell fell
        // back to the horizontal measure). The text measures 12px
        // vertically in both passes via the strut floor; each div
        // contributes exactly its content.
        let html = r#"<table class="t"><tr><td><div class="o"><div class="i">a</div></div></td></tr></table>"#;
        let css = [VCELL, " .o { font-size: 10px; } .i { font-size: 10px; }"].concat();
        let (mut forest, cell) = forest_cell(html, &css);
        assert!(approx(cell_min(&mut forest, cell), 12.0));
        assert!(approx(cell_max(&mut forest, cell), 12.0));
    }

    #[test]
    fn auto_height_block_with_margins_and_padding() {
        // Vertical border/padding/margins ride along with the measured
        // content: 10px top + 10px bottom padding around the 12px text
        // line = 32.
        let html = r#"<table class="t"><tr><td><div class="d">a</div></td></tr></table>"#;
        let css = [VCELL, " .d { font-size: 10px; padding: 10px 0; }"].concat();
        let (mut forest, cell) = forest_cell(html, &css);
        assert!(approx(cell_min(&mut forest, cell), 32.0));
        assert!(approx(cell_max(&mut forest, cell), 32.0));
    }

    #[test]
    fn pct_height_resolves_against_definite_cell_height() {
        // The cell's height is definite (100px), so the 50% div is a
        // definite 50px vertical contribution — previously the
        // percentage declined the whole cell to the fallback.
        let html = r#"<table class="t"><tr><td class="c"><div class="d"></div></td></tr></table>"#;
        let css = [VCELL, " .c { height: 100px; } .d { height: 50%; }"].concat();
        let (mut forest, cell) = forest_cell(html, &css);
        assert!(approx(cell_min(&mut forest, cell), 50.0));
        assert!(approx(cell_max(&mut forest, cell), 50.0));
    }

    #[test]
    fn definite_height_fast_path_unchanged() {
        // Regression pin: definite-height block children take the exact
        // fast path, untouched by the extension.
        let html = r#"<table class="t"><tr><td><div class="d"></div></td></tr></table>"#;
        let css = [VCELL, " .d { height: 30px; }"].concat();
        let (mut forest, cell) = forest_cell(html, &css);
        assert!(approx(cell_min(&mut forest, cell), 30.0));
        assert!(approx(cell_max(&mut forest, cell), 30.0));
    }

    #[test]
    fn nested_auto_table_falls_back_honestly() {
        // A nested table with indefinite size has no vertical model:
        // the cell takes the documented horizontal fallback — equal to
        // the horizontal min-content width — rather than a fabricated
        // vertical number.
        let html = r#"<table class="t"><tr><td><table class="in"><tr><td>aaaa</td></tr></table></td></tr></table>"#;
        let (mut forest, cell) = forest_cell(html, VCELL);
        let v = cell_min(&mut forest, cell);
        let h = {
            let fonts = FixedFontMetrics;
            let mut scratch = Vec::new();
            min_content_width(&mut forest, cell, &fonts, &mut scratch)
        };
        assert!(v > 0.0);
        assert!(approx(v, h), "fallback {v} != horizontal min-content {h}");
    }
}

/// Wave 65 fix-3a end-to-end test: the vertical-aware CAPMIN floor on
/// the *used* inline size of a block-level vertical table (via
/// [`crate::layout`], not the intrinsic helpers).
#[cfg(test)]
mod w65_fix3a_tests {
    use crate::{BoxType, FixedFontMetrics, LayoutBox, Size, layout};

    const VP: Size = Size { w: 800.0, h: 600.0 };

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    fn layout_html(html: &str, css: &str) -> LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        layout(&doc, &[&ss], VP, &FixedFontMetrics)
    }

    fn find_table(l: &LayoutBox) -> &LayoutBox {
        let mut stack = vec![l];
        while let Some(b) = stack.pop() {
            if b.box_type == BoxType::Table {
                return b;
            }
            stack.extend(b.children.iter());
        }
        panic!("table box not found");
    }

    #[test]
    fn vertical_capmin_floors_used_inline_size() {
        // A block-level vertical-rl table whose caption has a 60px
        // vertical extent while its columns only measure 10px on the
        // vertical axis. CSS 2.1 §17.5.2.2 CAPMIN floors the *used*
        // inline size to the caption's outer minimum (60). The table
        // carries 5px physical top/bottom borders — the inline-axis
        // border+padding in a vertical writing mode — so the content
        // inline size is 50 and the border-box inline size (the physical
        // height after the vertical transpose) is exactly 60. With
        // physical-horizontal extras the table would use 60px of content
        // and report 70px instead.
        let l = layout_html(
            r#"<table class="t"><caption><div class="cap">x</div></caption><tr><td><div class="c">x</div></td></tr></table>"#,
            "body { margin: 0; } \
             .t { writing-mode: vertical-rl; border-spacing: 0; \
                  border-top: 5px solid black; border-bottom: 5px solid black; } \
             td, caption { padding: 0; font-size: 10px; } \
             .cap { height: 60px; } .c { height: 10px; }",
        );
        let t = find_table(&l);
        assert!(
            approx(t.rect.h, 60.0),
            "used inline size (physical height) = {}",
            t.rect.h
        );
    }
}

/// Wave 76 caption table 8 root cause: a shaped whole-run advance can
/// sit below the sum of the line breaker's per-fragment advances (f32
/// double-rounding: Noto Serif at 16px shapes "2 above me, 2 below
/// me?" to 189.888 whole-run but 189.888031 as word+space fragments),
/// so a shrink-to-fit box sized from the whole-run value wrapped its
/// own max-content. The max-content measurement must therefore cover
/// the breaker's fragment sum (taking the max of the fragment sum and
/// the whole-run advance).
#[cfg(test)]
mod w76_measure_tests {
    use crate::{BoxType, FontFamilyId, FontMetrics, LayoutBox, Size, layout};

    const VP: Size = Size { w: 800.0, h: 600.0 };

    /// Models a shaped backend whose whole-run advance falls just below
    /// its fragment sum: a spaceless run measures `0.6 * font_size` per
    /// character, but a run containing a space measures the fragment
    /// sum minus a small shortfall (the shaped-vs-fragment gap).
    struct WholeRunShortfallMetrics;

    impl FontMetrics for WholeRunShortfallMetrics {
        fn advance(&self, text: &str, font_size: f32, _family: FontFamilyId) -> f32 {
            let per_char = 0.6 * font_size;
            // `len() > 1` keeps the single-space fragment on the
            // spaceless path: `" ".contains(' ')` is true, but a lone
            // space is one fragment, not a whole run.
            if text.len() > 1 && text.contains(' ') {
                let mut sum = 0.0f32;
                for (k, part) in text.split(' ').enumerate() {
                    if k > 0 {
                        sum += per_char;
                    }
                    sum += part.chars().count() as f32 * per_char;
                }
                sum - 0.03125
            } else {
                text.chars().count() as f32 * per_char
            }
        }

        fn line_height(&self, font_size: f32, _family: FontFamilyId) -> f32 {
            1.2 * font_size
        }
    }

    fn find_inline_block(l: &LayoutBox) -> &LayoutBox {
        let mut stack = vec![l];
        while let Some(b) = stack.pop() {
            if b.box_type == BoxType::InlineBlock {
                return b;
            }
            stack.extend(b.children.iter());
        }
        panic!("inline-block box not found");
    }

    #[test]
    fn shrink_to_fit_box_does_not_wrap_its_own_max_content() {
        // Fragment sum at 10px: words 12+12+12, spaces 6+6 = 48; the
        // whole-run advance is 47.96875, so sizing from the whole run
        // would wrap the last word onto a second 10px line.
        let (doc, _) = bowser_dom::Document::parse_html(
            b"<div style=\"display:inline-block;line-height:10px;font-size:10px\">aa bb cc</div>",
        );
        let l = layout(&doc, &[], VP, &WholeRunShortfallMetrics);
        let ib = find_inline_block(&l);
        assert!(
            (ib.rect.w - 48.0).abs() < 0.01,
            "inline-block width {}, want 48 (fragment sum)",
            ib.rect.w
        );
        assert!(
            (ib.rect.h - 10.0).abs() < 0.01,
            "inline-block height {}, want 10 (one line, not wrapped)",
            ib.rect.h
        );
    }
}
