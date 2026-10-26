//! Box tree construction: DOM -> arena of layout boxes.
//!
//! Passes, all iterative (safe on adversarially deep DOMs):
//! 1. Raw tree: one box per rendered element/text node. `display: none`
//!    subtrees are skipped wholesale.
//! 2. An `inline` box that ended up containing block-level children becomes
//!    a block box (documented simplification; CSS would split the inline
//!    box around the blocks).
//! 3. Consecutive inline-level children of a block or inline-block are
//!    wrapped in anonymous block boxes, but only when mixed with
//!    block-level children (a block with purely inline content needs no
//!    anonymous boxes). For an inline-block this gives the inline runs a
//!    normal inline formatting context inside the block-level wrapper.
//! 4. Whitespace-only text nodes adjacent to block-level boxes (or at the
//!    edges of a block) are dropped; survivors keep a single space
//!    (`pre`/`pre-wrap` text is exempt: every space is significant).
//!
//! `display: contents` generates no box at all: the element's children
//! hoist to the parent box (handled in pass 1). Inheritance still passes
//! through the element — its font size and `white-space` travel with the
//! hoisted children.

use super::style::WhiteSpace;
use super::style::{
    Clear, DEFAULT_ROOT_FONT_SIZE, Display, Float, Position, SpecLen, SpecMargin, StyleData,
    TextTransform, compute_element_style, display_of, resolve_font_size, sane_len,
};
use super::{BoxType, FontMetrics, IntrinsicDims, Rect};
use bowser_css::{ComputedStyle, Styler, Stylesheet, Viewport, props};
use bowser_dom::{Document, NodeId, NodeKind, QuirksMode};
use std::borrow::Cow;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

/// Horizontal/vertical box metrics, resolved once the containing-block
/// width is known. Vertical margins are *un*collapsed here; collapsing
/// happens during block layout.
#[derive(Clone, Debug)]
pub(crate) struct BoxMetrics {
    pub ml: f32,
    pub mr: f32,
    pub mt: f32,
    pub mb: f32,
    pub bl: f32,
    pub br: f32,
    pub bt: f32,
    pub bb: f32,
    pub pl: f32,
    pub pr: f32,
    pub pt: f32,
    pub pb: f32,
    pub content_w: f32,
}

impl BoxMetrics {
    fn zero() -> BoxMetrics {
        BoxMetrics {
            ml: 0.0,
            mr: 0.0,
            mt: 0.0,
            mb: 0.0,
            bl: 0.0,
            br: 0.0,
            bt: 0.0,
            bb: 0.0,
            pl: 0.0,
            pr: 0.0,
            pt: 0.0,
            pb: 0.0,
            content_w: 0.0,
        }
    }
}

/// Which table section a [`BoxType::TableRowGroup`] box represents,
/// stamped from the element's computed `display` (which wins) with the
/// tag name as the fallback, at box construction (CSS Tables 3 §3
/// table structure). Row groups generated from non-section elements
/// (e.g. a `<div>` with `display: table-row-group`) are
/// [`SectionKind::Body`]: they behave as body groups for
/// section-specific semantics. Stamping here (rather than re-deriving
/// the tag during table layout, where no `Document` is in scope) is
/// what lets table layout distinguish header/footer groups — e.g. for
/// `visibility: collapse` reporting per CSS Tables 3 §5.4 and
/// repeated header/footer groups under fragmentation (CSS Tables 3
/// §6) — without a DOM lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SectionKind {
    /// `display: table-header-group` (or a `<thead>` with plain
    /// `display: table-row-group`).
    Header,
    /// `<tbody>`, and row groups from non-section elements with plain
    /// `display: table-row-group` (see [`section_kind_of_tag_and_display`]).
    Body,
    /// `display: table-footer-group` (or a `<tfoot>` with plain
    /// `display: table-row-group`).
    Footer,
}

/// Section kind for a row-group box from the generating element's
/// computed `display` and tag name: the *display wins*.
/// `Display::TableHeaderGroup` → [`SectionKind::Header`] and
/// `Display::TableFooterGroup` → [`SectionKind::Footer`], regardless of
/// the element's tag — a `<thead>` restyled to `table-footer-group`
/// stamps Footer, and a `<tfoot>` restyled to `table-header-group`
/// stamps Header (CSS 2.1 §17.2.1: those displays "behave like a THEAD
/// element"). Only with plain `display: table-row-group` does the tag
/// disambiguate: `thead` → Header, `tfoot` → Footer, anything else
/// (`tbody`, or a `<div>` restyled to plain `table-row-group`) →
/// [`SectionKind::Body`]. CSS Tables 3 §6.1 does not formally define
/// which boxes count as headers for repetition, so this display-based
/// disambiguation is the defensible choice.
fn section_kind_of_tag_and_display(tag: &str, display: Display) -> SectionKind {
    match display {
        Display::TableHeaderGroup => SectionKind::Header,
        Display::TableFooterGroup => SectionKind::Footer,
        _ => {
            if tag.eq_ignore_ascii_case("thead") {
                SectionKind::Header
            } else if tag.eq_ignore_ascii_case("tfoot") {
                SectionKind::Footer
            } else {
                SectionKind::Body
            }
        }
    }
}

/// Whether `tag` is an HTML table element (`table`, `caption`, `col`,
/// `colgroup`, `thead`, `tbody`, `tfoot`, `tr`, `td`, `th`), ASCII
/// case-insensitive like [`section_kind_of_tag_and_display`]. Used to
/// stamp [`Node::from_table_element`] at box construction.
fn is_table_element_tag(tag: &str) -> bool {
    tag.eq_ignore_ascii_case("table")
        || tag.eq_ignore_ascii_case("caption")
        || tag.eq_ignore_ascii_case("col")
        || tag.eq_ignore_ascii_case("colgroup")
        || tag.eq_ignore_ascii_case("thead")
        || tag.eq_ignore_ascii_case("tbody")
        || tag.eq_ignore_ascii_case("tfoot")
        || tag.eq_ignore_ascii_case("tr")
        || tag.eq_ignore_ascii_case("td")
        || tag.eq_ignore_ascii_case("th")
}

/// Inline-first per-line text fragments for an arena [`Node`].
///
/// The old `Option<Box<Vec<_>>>` paid two allocations per text box that
/// carried fragments (the box plus the `Vec` buffer, grown to 4 elements
/// / 160 B on first push) — and every text box in the tree carries at
/// least one fragment once inline layout runs. The common case is
/// exactly one fragment (a text run that fits on a single line), so the
/// first fragment now lives inline: zero allocation. Only the second
/// push spills to a heap `Vec`, allocated for exactly 2 and grown from
/// there.
///
/// [`into_vec`][Self::into_vec] moves a spilled vec out with no
/// reallocation, and packs the inline case into a single-element `Vec`
/// (40 B, not the old 160 B buffer).
///
/// Cost: 48 bytes on the arena node instead of 8 (see the
/// [`node_size_tests`] budget). Only [`BoxType::Text`] boxes ever carry
/// fragments, and the per-text-box allocation saving dominates on
/// text-heavy pages.
///
/// The slice-like API (`Deref<Target = [TextFragment]>`, [`retain`][Self::retain],
/// [`push`][Self::push], [`clear`][Self::clear], `IntoIterator` for `&`
/// and `&mut`) is unchanged, so every existing `node.fragments` use
/// site keeps compiling; only `convert_forest`'s `mem::take` into the
/// public tree's plain `Vec` converts explicitly, via
/// [`into_vec`][Self::into_vec].
#[derive(Clone, Debug, Default)]
pub(crate) enum FragmentList {
    /// No fragments. No allocation.
    #[default]
    Empty,
    /// Exactly one fragment, stored inline. No allocation.
    One(super::TextFragment),
    /// Two or more fragments, on the heap.
    Many(Vec<super::TextFragment>),
}

impl FragmentList {
    /// Empty list: no allocation.
    pub(crate) fn new() -> Self {
        Self::Empty
    }

    /// Move the fragments out as a plain `Vec`. A spilled vec moves out
    /// with no reallocation; the inline case packs into a
    /// single-element vec.
    pub(crate) fn into_vec(self) -> Vec<super::TextFragment> {
        match self {
            Self::Empty => Vec::new(),
            Self::One(f) => vec![f],
            Self::Many(v) => v,
        }
    }

    /// Append one fragment. The first push stores inline (no
    /// allocation); the second spills to a heap vec (one allocation).
    pub(crate) fn push(&mut self, f: super::TextFragment) {
        match self {
            Self::Empty => *self = Self::One(f),
            Self::Many(v) => v.push(f),
            Self::One(_) => {
                let Self::One(g) = std::mem::replace(self, Self::Empty) else {
                    unreachable!("FragmentList::push matched `One`");
                };
                *self = Self::Many(vec![g, f]);
            }
        }
    }

    /// Drop all fragments. A spilled vec keeps its allocation (mirrors
    /// `Vec::clear`); the inline case just empties.
    pub(crate) fn clear(&mut self) {
        match self {
            Self::Many(v) => v.clear(),
            _ => *self = Self::Empty,
        }
    }

    /// Keep only the fragments matching `pred` (mirrors `Vec::retain`).
    pub(crate) fn retain(&mut self, mut pred: impl FnMut(&super::TextFragment) -> bool) {
        match self {
            Self::Empty => {}
            Self::One(f) => {
                if !pred(f) {
                    *self = Self::Empty;
                }
            }
            Self::Many(v) => v.retain(pred),
        }
    }
}

impl std::ops::Deref for FragmentList {
    type Target = [super::TextFragment];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Empty => &[],
            Self::One(f) => std::slice::from_ref(f),
            Self::Many(v) => v,
        }
    }
}

impl std::ops::DerefMut for FragmentList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Empty => &mut [],
            Self::One(f) => std::slice::from_mut(f),
            Self::Many(v) => v,
        }
    }
}

impl<'a> IntoIterator for &'a FragmentList {
    type Item = &'a super::TextFragment;
    type IntoIter = std::slice::Iter<'a, super::TextFragment>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut FragmentList {
    type Item = &'a mut super::TextFragment;
    type IntoIter = std::slice::IterMut<'a, super::TextFragment>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// One box in the arena.
pub(crate) struct Node {
    pub box_type: BoxType,
    /// The memoized computed style, shared with the styler's memo by
    /// reference count (one bump per element per style pass, no deep
    /// clone). Read-only during layout (`as_ref` / `take` at the
    /// arena-to-tree boundary); never mutated in place.
    pub style: Option<Rc<ComputedStyle>>,
    /// Inline, not boxed: `StyleData` is 304B and every arena node
    /// carries one, so boxing it was one heap allocation per box per
    /// layout pass. The nodes already live in a `Vec` (one allocation
    /// total, amortized), so inlining removes the per-box alloc with no
    /// extra bookkeeping.
    pub data: StyleData,
    /// Text payload for [`BoxType::Text`] boxes, shared with the DOM's
    /// interned `Arc<str>` whenever whitespace collapsing and
    /// `text-transform` are no-ops (see [`shared_text`]): storing the
    /// `Arc` is a refcount bump, not a copy, and every re-layout /
    /// arena clone (e.g. multicol splits) shares the same allocation.
    /// Only text whose bytes actually changed gets a fresh `Arc`.
    pub text: Option<Arc<str>>,
    pub children: Vec<usize>,
    pub parent: Option<usize>,
    pub metrics: BoxMetrics,
    /// Border box in absolute page coordinates.
    pub rect: Rect,
    /// Deferred position adjustment for this subtree's root, recorded by
    /// inline layout's inline-block placement (step 4 of
    /// `do_inline_layout`) instead of eagerly rewriting the subtree.
    /// Rects stay provisional until a single bottom-up pass at the end of
    /// `layout_forest` pushes the accumulated offsets down; each nesting
    /// level then costs O(1) rather than O(subtree).
    pub pend_x: f32,
    pub pend_y: f32,
    pub content_h: f32,
    /// Natural content height before `height_is_min` minimums and
    /// min-height/max-height clamping. The table baseline-shift code
    /// uses it (falling back to `content_h` when NaN) so a specified
    /// `height` absorbs the baseline shift instead of growing the row.
    /// Recorded by the block layout paths; NaN elsewhere.
    pub natural_content_h: f32,
    /// Whether the used content height came from a definite source (an
    /// explicitly specified height, a caller override, or a solved
    /// abspos height) rather than content. Set in the `layout_block`
    /// prologue; read by the `position: relative` post-pass, where
    /// percentage `top`/`bottom` compute to `auto` against an indefinite
    /// containing-block height (CSS 2.1 §10.6.4).
    pub height_definite: bool,
    /// Collapsed margin used above/below the border box.
    pub mt_place: f32,
    pub mb_place: f32,
    pub mt_place_done: bool,
    /// Whether `mt_place` absorbed the first child's top margin.
    pub top_collapsed: bool,
    /// Distance from the border-box top to the baseline of the last line
    /// box (for inline-block baseline alignment); border-box height when
    /// there are no line boxes.
    pub baseline: f32,
    /// Intrinsic preferred content width (for inline-block shrink-to-fit).
    pub pref_w: f32,
    /// The DOM node this box was built from. `None` for anonymous boxes
    /// (paint/JS contract: `LayoutBox::node`).
    pub dom: Option<NodeId>,
    /// Intrinsic dimensions for replaced elements (`<img>` etc.), from
    /// the intrinsics map; `None` for non-replaced elements.
    pub intrinsic: Option<IntrinsicDims>,
    /// Per-line fragments for text boxes, recorded during inline layout
    /// (paint/JS contract: `LayoutBox::fragments`). Stored inline-first
    /// behind [`FragmentList`]: the common zero/one-fragment case costs
    /// no allocation, and only two-or-more fragments spill to a heap
    /// `Vec` (see [`FragmentList`]).
    pub fragments: FragmentList,
    /// `colspan` / `rowspan` for [`BoxType::TableCell`] boxes, parsed from
    /// the element's attributes (1 when absent or invalid, clamped to
    /// 1000 per the HTML spec's "rules for parsing non-negative
    /// integers"); 0 means "span to the end of the table / section"
    /// (`colspan="0"` / `rowspan="0"`), resolved at grid placement.
    /// Doubles as the `span` attribute for [`BoxType::TableColumn`] /
    /// [`BoxType::TableColumnGroup`] boxes (1 when absent or invalid,
    /// clamped to [1, 1000]); 1 for every other box type.
    pub colspan: u32,
    pub rowspan: u32,
    /// Section kind for [`BoxType::TableRowGroup`] boxes (see
    /// [`SectionKind`]); `None` for every other box type.
    pub section_kind: Option<SectionKind>,
    /// Whether this node was already recorded in [`Forest::abs_items`].
    /// Makes `layout_position::record_abs_item` idempotent in O(1): the
    /// inline-flex estimate pass and flex stretch re-layouts can meet the
    /// same out-of-flow box twice, and the old linear scan of `abs_items`
    /// made that quadratic.
    pub abspos_recorded: bool,
    /// Collapsed-border winners this cell paints ([top, right, bottom,
    /// left]); `Some` only for [`BoxType::TableCell`] nodes in a
    /// `border-collapse: collapse` table, set by
    /// `layout_table::resume_table`. Moved onto
    /// [`super::LayoutBox::collapsed_borders`] by `convert_forest`.
    pub collapsed_borders: Option<Box<[crate::CollapsedBorder; 4]>>,
    /// `list-style-position: outside` marker for a `display: list-item`
    /// box (see [`super::marker`]): never a flow child, positioned by
    /// the post-pass `position_markers`. `None` for every other box
    /// (and for `inside` markers, which are inline boxes in the flow).
    /// Boxed: only list items carry one.
    pub marker: Option<Box<MarkerLayout>>,
    /// v1 multi-column parameters when this block container establishes a
    /// multicol formatting context (`column-count` / `column-width`
    /// resolving to 2+ columns). Set in `layout_block`'s prologue so
    /// children lay out against one column width; consumed by the
    /// completion paths (`resume_block`, `do_inline_layout`), which
    /// redistribute children across the columns. `None` = plain block
    /// layout. Internal only: the public [`super::LayoutBox`] shape is
    /// unchanged. Boxed: only multicol containers carry one.
    pub multicol: Option<Box<crate::layout::layout_multicol::MulticolParams>>,
    /// Whether this table's width comes purely from border-spacing
    /// phantom gaps (no cells to measure). Stamped by
    /// [`crate::layout::layout_table::table_sized_by_phantom`] at the end
    /// of table layout; moved onto [`super::LayoutBox::phantom_w`]
    /// by `convert_forest`. Only meaningful on [`BoxType::Table`] nodes.
    /// Paired with [`Self::phantom_h`]: the two dimensions stamp
    /// independently (a caption exempts only the width; a specified
    /// height keeps its number while the width stays phantom), so the
    /// single-bool OR lost which dimension was phantom.
    pub phantom_w: bool,
    /// Whether this table's height comes purely from border-spacing
    /// phantom gaps (no cells to measure). See [`Self::phantom_w`].
    pub phantom_h: bool,
    /// Whether this box was built from an HTML table element (`table`,
    /// `caption`, `col`, `colgroup`, `thead`, `tbody`, `tfoot`, `tr`,
    /// `td`, `th`) rather than from `display: table-*` on another
    /// element or from an anonymous box. Stamped at box construction
    /// while the tag name is in scope; read by the orphan-table fixup
    /// ([`crate::layout::layout_table::wrap_orphan_table_boxes`]), which
    /// only wraps element-originated orphans (plus the anonymous
    /// wrappers it creates itself, stamped at creation).
    pub from_table_element: bool,
    /// Whether this box came from an HTML `<br>` element. Stamped at
    /// box construction while the tag name is in scope (same reason
    /// as [`Node::from_table_element`]); read by inline layout, which
    /// turns it into a forced line break. Only honored for
    /// [`BoxType::Inline`] boxes — a `<br>` restyled to `display: block`
    /// lays out as an ordinary empty block.
    pub is_br: bool,
    /// Whether this box came from an HTML `<img>` element. Stamped
    /// at box construction while the tag name is in scope (same
    /// reason as [`Node::is_br`]); read by the quirks §3.8
    /// run-summing pass, which names `img` elements specifically —
    /// other replaced elements must not join a run.
    pub is_img: bool,
    /// Whether this box came from a `td`/`th` element carrying a
    /// `nowrap` attribute. Stamped at box construction while the
    /// document is in scope (same reason as [`Node::is_br`]); read
    /// by the quirks §3.9 minimum-width pass in table layout.
    pub nowrap_attr: bool,
}

/// An `list-style-position: outside` marker attached to a list-item
/// box: laid out by the post-pass `position_markers`, never a flow
/// child, moved onto [`super::MarkerBox`] by `convert_forest`.
#[derive(Clone, Debug)]
pub(crate) struct MarkerLayout {
    /// Marker text ("•", "1.", …).
    pub text: String,
    /// Resolved marker font size in px.
    pub font_size: f32,
    /// `::marker` computed style when a `::marker` rule matched the
    /// list item (`None` = the marker inherits the item's style).
    pub style: Option<ComputedStyle>,
    /// Whether the item's direction is rtl (the marker then goes on
    /// the inline-end side).
    pub rtl: bool,
    /// Final border box in absolute page coordinates, filled by
    /// `position_markers`.
    pub rect: Option<Rect>,
}

impl Node {
    fn new(
        box_type: BoxType,
        style: Option<Rc<ComputedStyle>>,
        data: StyleData,
        text: Option<Arc<str>>,
        parent: Option<usize>,
    ) -> Node {
        Node {
            box_type,
            style,
            data,
            text,
            children: Vec::new(),
            parent,
            metrics: BoxMetrics::zero(),
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
            },
            pend_x: 0.0,
            pend_y: 0.0,
            content_h: 0.0,
            natural_content_h: f32::NAN,
            height_definite: false,
            mt_place: 0.0,
            mb_place: 0.0,
            mt_place_done: false,
            top_collapsed: false,
            baseline: 0.0,
            pref_w: 0.0,
            dom: None,
            intrinsic: None,
            fragments: FragmentList::new(),
            colspan: 1,
            rowspan: 1,
            section_kind: None,
            abspos_recorded: false,
            collapsed_borders: None,
            marker: None,
            multicol: None,
            phantom_w: false,
            phantom_h: false,
            from_table_element: false,
            is_br: false,
            is_img: false,
            nowrap_attr: false,
        }
    }
}

/// One out-of-flow box (`position: absolute | fixed`) recorded during
/// normal flow, with its resolved containing block (`None` for the
/// initial containing block / viewport).
///
/// The box takes no space in flow: the flow passes skip it and record it
/// here, and the `layout_position` post-pass lays out each item's subtree
/// against its containing block after normal flow finishes. `static_x` /
/// `static_y` are the position the box would have had in flow
/// (provisional frame, i.e. before `apply_pending_shifts`), used when the
/// corresponding offsets are `auto` (CSS 2.1 §10.3.7/§10.6.4): the
/// border-box origin, except for boxes recorded from inline flow, where
/// they are the margin-box anchor (see `margins_pending`).
pub(crate) struct AbsItem {
    pub node: usize,
    pub containing_block: Option<usize>,
    pub static_x: f32,
    pub static_y: f32,
    /// Whether the recorded static position is the margin-box anchor
    /// (the point where the box's margin box would begin in flow)
    /// rather than the border-box origin: `layout_position` adds the
    /// used margins when the offsets on an axis are `auto`. Set for
    /// boxes recorded from inline flow, whose anchor is the point just
    /// after the preceding fragment with no margins folded in yet;
    /// block-level recording sites pass the border-box origin directly
    /// (their margins are already folded in by `resolve_metrics`).
    pub margins_pending: bool,
}

/// The box arena plus layout-wide bases.
pub(crate) struct Forest {
    pub nodes: Vec<Node>,
    pub root_font_size: f32,
    /// The *true* viewport size: the base every viewport-unit
    /// (`vw`/`vh`/`vmin`/`vmax`) resolution during layout reads. A
    /// subtree pass must never overwrite these with the subtree's
    /// containing-block size — that width travels separately into
    /// `layout_forest` (see `layout::layout_forest_with_root_cb`).
    pub vp_w: f32,
    pub vp_h: f32,
    /// Out-of-flow boxes recorded during normal flow (they take no space
    /// there), each with its resolved containing block (`None` for the
    /// initial containing block). Laid out by the post-pass in
    /// `super::layout::layout_position` after normal flow, before the
    /// single `apply_pending_shifts` pass.
    pub abs_items: Vec<AbsItem>,
    /// The layout root's node index when its `scrollbar-gutter` was
    /// propagated to the viewport (CSS Overflow 4 §2.2) and the initial
    /// containing block was shrunk by the reservation in
    /// `layout_forest`: the root's own children must not reserve the
    /// gutter a second time in `layout_block`. `None` on every other
    /// pass.
    pub gutter_propagated_root: Option<usize>,
    /// Whether the forest root is a document's `html` element, as
    /// opposed to the root of a partial (subtree) layout built by the
    /// script host for a mid-parse measurement, whose root is whatever
    /// element the measurement targeted. Viewport-level rules (the
    /// root `scrollbar-gutter` propagation in `layout_forest`) apply
    /// only to a true `html` root: a subtree root that happens to carry
    /// `scrollbar-gutter: stable` (a scroll container, a `body`) must
    /// not shrink the partial pass's containing block.
    pub root_is_html: bool,
    /// The document's quirks mode, copied from
    /// [`Document::quirks_mode`] at box-tree construction (w77 quirks
    /// plumbing): layout and intrinsic sizing branch on this, never on
    /// a re-parse of the doctype. The styler reads the same mode from
    /// the document itself, so style and layout cannot disagree.
    pub quirks_mode: QuirksMode,
    /// Test-only: counts fragment visits in the inline rect-union pass
    /// (layout.rs step 5 of `do_inline_layout`). The union visits each
    /// fragment exactly once, so tests can assert the single-pass bound
    /// without wall-clock timing.
    #[cfg(test)]
    pub union_frag_visits: usize,
}

impl Forest {
    pub(crate) fn add_node(
        &mut self,
        box_type: BoxType,
        style: Option<Rc<ComputedStyle>>,
        data: StyleData,
        text: Option<Arc<str>>,
        parent: Option<usize>,
    ) -> usize {
        let idx = self.nodes.len();
        if let Some(p) = parent {
            self.nodes[p].children.push(idx);
        }
        self.nodes
            .push(Node::new(box_type, style, data, text, parent));
        idx
    }

    pub(crate) fn make_anonymous(
        &mut self,
        parent_data: &StyleData,
        children: Vec<usize>,
        parent: usize,
        // Whether the wrapper starts a fresh bidi scope instead of
        // sharing the parent's: anonymous flex/grid items are
        // independent layout units whose computed `unicode-bidi` is
        // the initial `normal`, so the parent's plaintext scope must
        // not cross into them (see `bidi_plaintext`'s block-boundary
        // rule). Pass-3 inline-run wrappers are transparent
        // fragmentations of the parent's own content flow and keep
        // sharing its scope (pinned by
        // `plaintext_propagates_across_inline_and_anonymous_boxes`).
        reset_bidi_scope: bool,
    ) -> usize {
        let idx = self.make_anonymous_box(BoxType::AnonymousBlock, parent_data, children, parent);
        if reset_bidi_scope {
            self.nodes[idx].data.bidi_plaintext = false;
        }
        idx
    }

    /// Anonymous box of an explicit type (for table fix-up: anonymous
    /// rows and cells per CSS 2.1 §17.2.1). Table-internal boxes take no
    /// margins, floats, or clearance.
    pub(crate) fn make_anonymous_box(
        &mut self,
        box_type: BoxType,
        parent_data: &StyleData,
        children: Vec<usize>,
        parent: usize,
    ) -> usize {
        let idx = self.nodes.len();
        for &c in &children {
            self.nodes[c].parent = Some(idx);
        }
        let mut data = StyleData::inherit(parent_data);
        if matches!(
            box_type,
            BoxType::TableRowGroup | BoxType::TableRow | BoxType::TableCell
        ) {
            // Anonymous table wrappers are independent layout units
            // (and table-internal boxes establish a bidi scope — see
            // `establishes_bidi_scope`) with no element style, so
            // their computed `unicode-bidi` is the initial `normal`:
            // per the block-boundary rule (`bidi_plaintext`) the
            // parent's plaintext scope stops here instead of leaking
            // into the wrapper's paragraphs. (Styled block
            // descendants derive their own flag from their own
            // `unicode-bidi` regardless.)
            data.bidi_plaintext = false;
        }
        if matches!(
            box_type,
            BoxType::TableRowGroup | BoxType::TableRow | BoxType::TableCell
        ) {
            data.margin = [SpecMargin::Len(SpecLen::Px(0.0)); 4];
            data.float = Float::None;
            data.clear = Clear::None;
        }
        self.nodes
            .push(Node::new(box_type, None, data, None, Some(parent)));
        self.nodes[idx].children = children;
        idx
    }
}

/// Block-level box types participate in block formatting contexts and
/// margin collapsing. `Table` is block-level in flow (it stacks and
/// collapses margins like a block); `TableCell` is a block container for
/// its contents; `Grid` is a block-level grid container (its items never
/// collapse margins with it). `TableRowGroup` and `TableRow` are internal
/// to table layout and never participate in block flow.
pub(crate) fn is_block_level(box_type: BoxType) -> bool {
    matches!(
        box_type,
        BoxType::Block
            | BoxType::AnonymousBlock
            | BoxType::Table
            | BoxType::TableCell
            | BoxType::Grid
    )
}

/// Whether a forest node is out of flow (`position: absolute | fixed`).
/// Local predicate (rather than the `layout` module's) because `layout`
/// depends on `boxes`, not the other way around.
pub(crate) fn is_out_of_flow(forest: &Forest, idx: usize) -> bool {
    matches!(
        forest.nodes[idx].data.position,
        Position::Absolute | Position::Fixed
    )
}

fn tag_name(doc: &Document, node: NodeId) -> &str {
    match doc.kind(node) {
        NodeKind::Element(data) => data.tag_name.as_str(),
        _ => "",
    }
}

/// Collapse whitespace per `white-space`: `normal`/`nowrap` collapse runs
/// to a single space; `pre`/`pre-wrap`/`break-spaces` keep everything;
/// collapses runs of spaces/tabs (and other non-newline whitespace) to a
/// single space but preserves each newline as a forced break: segment
/// breaks are not collapsible for `pre-line`, so each one is transformed
/// into a preserved line feed (CSS text-3 §4.1.3 — the §4.1.2 rule that a
/// sequence of segment breaks converts to a single forced line break
/// applies only to collapsible segment breaks). `\r\n` and lone `\r`
/// normalize to `\n`.
/// Fast pre-scan for [`collapse_ws`] in the `normal`/`nowrap` modes:
/// true when any character would change the output — a non-space
/// collapsible whitespace char (it maps to `' '`), or a run of 2+
/// collapsible whitespace chars (it collapses). Single ASCII spaces pass
/// through untouched, so the common text node returns borrowed.
fn needs_collapse(text: &str) -> bool {
    let mut prev_ws = false;
    for ch in text.chars() {
        if matches!(ch, ' ' | '\t' | '\n' | '\r' | '\u{000C}') {
            if ch != ' ' || prev_ws {
                return true;
            }
            prev_ws = true;
        } else {
            prev_ws = false;
        }
    }
    false
}

/// Fast pre-scan for [`collapse_ws`] in `pre-line` mode: true when any
/// character would change the output — a `\r` (it normalizes to `\n`),
/// any `\t` or form feed (they map to `' '`), or a run of 2+ spaces (it
/// collapses). Single ASCII spaces and newlines pass through untouched,
/// so the common text node returns borrowed.
fn needs_preline_collapse(text: &str) -> bool {
    let mut prev_sp = false;
    for ch in text.chars() {
        if ch == '\r' || ch == '\t' || ch == '\u{000C}' {
            return true;
        }
        if ch == ' ' {
            if prev_sp {
                return true;
            }
            prev_sp = true;
        } else {
            prev_sp = false;
        }
    }
    false
}

fn collapse_ws(text: &str, ws: WhiteSpace) -> Cow<'_, str> {
    match ws {
        WhiteSpace::Pre | WhiteSpace::PreWrap | WhiteSpace::BreakSpaces => Cow::Borrowed(text),
        WhiteSpace::PreLine => {
            if !needs_preline_collapse(text) {
                return Cow::Borrowed(text);
            }
            let mut out = String::with_capacity(text.len());
            let mut in_space = false;
            let mut chars = text.chars().peekable();
            while let Some(ch) = chars.next() {
                match ch {
                    // Segment break: normalize CRLF/CR to LF. Each break
                    // is preserved (not merged): for `pre-line` segment
                    // breaks are not collapsible (CSS text-3 §4.1.3).
                    '\n' => {
                        out.push('\n');
                        in_space = false;
                    }
                    '\r' => {
                        if chars.peek() == Some(&'\n') {
                            chars.next();
                        }
                        out.push('\n');
                        in_space = false;
                    }
                    ' ' | '\t' | '\u{000C}' => {
                        if !in_space {
                            out.push(' ');
                        }
                        in_space = true;
                    }
                    _ => {
                        out.push(ch);
                        in_space = false;
                    }
                }
            }
            Cow::Owned(out)
        }
        _ => {
            if !needs_collapse(text) {
                return Cow::Borrowed(text);
            }
            let mut out = String::with_capacity(text.len());
            let mut in_ws = false;
            for ch in text.chars() {
                if ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\u{000C}' {
                    if !in_ws {
                        out.push(' ');
                    }
                    in_ws = true;
                } else {
                    out.push(ch);
                    in_ws = false;
                }
            }
            Cow::Owned(out)
        }
    }
}

struct BuildItem {
    dom: NodeId,
    parent: usize,
    inherit_fs: f32,
    ws: WhiteSpace,
    /// `text-transform` of the containing element. Like `ws`, it travels
    /// on the build item so `display: contents` elements still transform
    /// their hoisted text.
    tt: TextTransform,
    /// Resolved `direction`: the element's own specified value wins,
    /// otherwise the inherited one. Travels on the build item so
    /// `display: contents` elements still pass it to hoisted children.
    rtl: bool,
    /// `unicode-bidi: plaintext` scope of the *containing block* this
    /// node participates in (see [`bidi_plaintext`]): the resolved
    /// plaintext flag of the nearest ancestor block container, carried
    /// through any inline / `display: contents` boxes in between.
    /// Travels like `rtl` so `display: contents` elements still pass it
    /// to hoisted children; a child that is itself a block container
    /// ignores it and derives its flag from its own computed value.
    bidi: bool,
}

/// Resolve `direction` for a box: the element's own specified value, or
/// the inherited `parent_rtl` when unspecified.
pub(crate) fn direction_rtl(style: &ComputedStyle, parent_rtl: bool) -> bool {
    match style.get_str_by_index(props::DIRECTION) {
        Some(d) => d.eq_ignore_ascii_case("rtl"),
        None => parent_rtl,
    }
}

/// The element's own computed `unicode-bidi` is `plaintext`.
fn own_bidi_plaintext(style: &ComputedStyle) -> bool {
    style
        .get_str_by_index(props::UNICODE_BIDI)
        .is_some_and(|v| v.eq_ignore_ascii_case("plaintext"))
}

/// Whether a box with this `display` is a block container that
/// establishes its own inline formatting context for its children
/// (block, inline-block, the inner flex/grid/table containers, and
/// the table-internal boxes). Such a box is the containing block of
/// the bidi paragraphs formed by its inline content, so
/// `unicode-bidi: plaintext` on it — and only on it — sets those
/// paragraphs' base direction (CSS Writing Modes 3 §2.7).
fn establishes_bidi_scope(display: Display) -> bool {
    matches!(
        display,
        Display::Block
            | Display::InlineBlock
            | Display::InlineFlex
            | Display::InlineGrid
            | Display::InlineTable
            | Display::Table
            | Display::Grid
            | Display::TableRowGroup
            | Display::TableHeaderGroup
            | Display::TableFooterGroup
            | Display::TableRow
            | Display::TableCell
            | Display::TableColumn
            | Display::TableColumnGroup
    )
}

/// Resolve `unicode-bidi: plaintext` for a box. The property is NOT
/// inherited (CSS Writing Modes 3 §2.7): `plaintext` sets the base
/// direction of the bidi paragraphs for which *that box* is the
/// containing block, and no others. So:
///
/// - a block container (see [`establishes_bidi_scope`]) derives the
///   flag from its OWN computed `unicode-bidi` alone — an ancestor's
///   `plaintext` (including an explicit scope it sits inside) does not
///   cross the block boundary, and an explicit `unicode-bidi: normal`
///   here is indistinguishable from the initial value, both meaning
///   "no plaintext paragraphs in this container";
/// - an inline or `display: contents` box generates no containing
///   block of its own: it shares its ancestor's containing block, so
///   the ancestor's scope propagates across it (`parent_bidi`), joined
///   with its own `plaintext` for the content it contributes.
///
/// Other `unicode-bidi` values have no layout behavior this wave.
fn bidi_plaintext(style: &ComputedStyle, parent_bidi: bool, display: Display) -> bool {
    if establishes_bidi_scope(display) {
        own_bidi_plaintext(style)
    } else {
        parent_bidi || own_bidi_plaintext(style)
    }
}

/// Tags treated as replaced elements in v1: when their CSS width/height
/// are `auto`, the box takes its size from the intrinsics map (missing
/// entry => 0x0). `textarea` and `select` are included so their internal
/// rendering (and generated content: no `::before`/`::after` inside
/// them) matches their form-control box model.
/// Replaced elements whose intrinsic dimensions come from the CSS spec
/// default rather than a decoded resource: `<iframe>`/`<embed>`/
/// `<object>`/`<video>` with no map entry size to the CSS 2.1 §10.3.2
/// 300x150 default (unknown dimensions, no ratio). `<img>` keeps the v1
/// 0x0 (a missing/broken image lays out empty; only decoded resources
/// provide its size).
fn default_intrinsic_for_tag(tag: &str) -> Option<IntrinsicDims> {
    match tag {
        "iframe" | "embed" | "object" | "video" => Some(IntrinsicDims::unknown()),
        _ => None,
    }
}

/// Sanitize a caller-supplied intrinsics entry: NaN/negative/infinite
/// dimensions degrade to finite nonnegative instead of poisoning
/// layout. A missing entry falls back to the tag default (300x150
/// unknown for `<iframe>` etc., 0x0 for `<img>`).
fn sanitize_intrinsic(entry: Option<IntrinsicDims>, tag: &str) -> IntrinsicDims {
    match entry {
        Some(d) => IntrinsicDims {
            w: d.w.map(|w| sane_len(w).max(0.0)),
            h: d.h.map(|h| sane_len(h).max(0.0)),
            ratio: d.ratio.filter(|r| *r > 0.0 && r.is_finite()),
        },
        None => default_intrinsic_for_tag(tag).unwrap_or(IntrinsicDims::from_size(0.0, 0.0)),
    }
}

fn is_replaced_tag(tag: &str) -> bool {
    matches!(
        tag,
        "img"
            | "video"
            | "audio"
            | "canvas"
            | "iframe"
            | "embed"
            | "object"
            | "input"
            | "textarea"
            | "select"
    )
}

/// Apply `text-transform` per CSS: uppercase/lowercase map the whole run;
/// capitalize uppercases the first cased character of each
/// whitespace-delimited word and leaves the rest untouched. Runs during
/// whitespace collapsing, before line breaking, so breaking measures
/// transformed advances.
/// True when `char::to_uppercase` (or `to_lowercase`) maps `c` to
/// something other than exactly `c` itself — i.e. the case mapping would
/// change the text. Conservative: any multi-char or differing mapping
/// counts as a change.
fn case_maps_differently(c: char, upper: bool) -> bool {
    if upper {
        !c.to_uppercase().eq(core::iter::once(c))
    } else {
        !c.to_lowercase().eq(core::iter::once(c))
    }
}

pub(crate) fn apply_text_transform(text: &str, tt: TextTransform) -> Cow<'_, str> {
    match tt {
        TextTransform::None => Cow::Borrowed(text),
        TextTransform::Uppercase => {
            if !text.chars().any(|c| case_maps_differently(c, true)) {
                return Cow::Borrowed(text);
            }
            Cow::Owned(text.to_uppercase())
        }
        TextTransform::Lowercase => {
            if !text.chars().any(|c| case_maps_differently(c, false)) {
                return Cow::Borrowed(text);
            }
            Cow::Owned(text.to_lowercase())
        }
        TextTransform::Capitalize => {
            // Pre-scan: only the first character of each
            // whitespace-delimited word can change, so most text (already
            // capitalized or all-lowercase single words) returns borrowed.
            let mut word_start = true;
            let needs = text.chars().any(|ch| {
                if ch.is_whitespace() {
                    word_start = true;
                    false
                } else {
                    let first = word_start;
                    word_start = false;
                    first && case_maps_differently(ch, true)
                }
            });
            if !needs {
                return Cow::Borrowed(text);
            }
            let mut out = String::with_capacity(text.len());
            let mut word_start = true;
            for ch in text.chars() {
                if ch.is_whitespace() {
                    word_start = true;
                    out.push(ch);
                } else {
                    if word_start {
                        out.extend(ch.to_uppercase());
                    } else {
                        out.push(ch);
                    }
                    word_start = false;
                }
            }
            Cow::Owned(out)
        }
    }
}

/// Collapse whitespace and apply `text-transform` to a DOM text
/// payload, returning the result as a shared `Arc<str>`.
///
/// Zero-copy fast path: both [`collapse_ws`] and
/// [`apply_text_transform`] return `Cow::Borrowed` when they change
/// nothing, and a borrowed result that is pointer-identical to
/// `source` means the whole chain was a no-op — so the DOM's `Arc` is
/// cloned (a refcount bump, no allocation or byte copy). Only when
/// the chain actually changed bytes is a fresh `Arc<str>` allocated
/// from the owned result. `None` for empty output (the caller drops
/// the text box, exactly as the old `into_owned` + `is_empty` check
/// did).
pub(crate) fn shared_text(
    source: &Arc<str>,
    ws: WhiteSpace,
    tt: TextTransform,
) -> Option<Arc<str>> {
    let collapsed = collapse_ws(source, ws);
    let transformed = apply_text_transform(&collapsed, tt);
    if transformed.is_empty() {
        return None;
    }
    match transformed {
        Cow::Borrowed(s) if std::ptr::eq(s, &**source) => Some(Arc::clone(source)),
        // Borrowed but not the source slice: neither helper slices,
        // so this is unreachable in practice; copy defensively rather
        // than share the wrong allocation.
        Cow::Borrowed(s) => Some(Arc::from(s)),
        Cow::Owned(s) => Some(Arc::from(s)),
    }
}

/// One visual line produced by [`wrap_pre_text`], consumed by
/// `emit_text_frags` in `layout.rs` (one fragment per piece).
///
/// `Copy`: every field is `Copy`, which lets [`PrePieces`] move pieces
/// out of its inline buffer without `unsafe`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PreWrapPiece<'a> {
    /// The line's text, borrowed from the input. Whitespace is preserved
    /// verbatim — including a space at the wrap point, which stays at the
    /// end of the broken line — except a `\r` immediately before a `\n`,
    /// which is normalized away (CSS newline handling).
    pub text: &'a str,
    /// Measured width of [`PreWrapPiece::text`] via the caller's measure
    /// function, so the emitter does not measure twice.
    pub width: f32,
    /// True for every piece except the first: the piece starts a new
    /// line, either after a forced `\n` break or after a soft wrap.
    pub break_before: bool,
    /// True only for the first piece of a hard line other than the
    /// very first: this piece's `break_before` is a `\n` paragraph
    /// separator (UAX #9 P1), not a soft wrap. Soft-wrapped pieces
    /// have `break_before` set but this flag clear — the distinction
    /// bidi paragraph segmentation (`unicode-bidi: plaintext`) needs,
    /// since paragraphs split only at paragraph separators, never at
    /// soft wraps. Set on exactly one piece per `\n`.
    pub forced_break: bool,
}

/// Inline capacity of [`PrePieces`]: a pre-formatted block that fits in
/// this many visual lines never touches the heap.
const PRE_PIECES_INLINE: usize = 8;

/// Small-vector of [`PreWrapPiece`]s produced by [`wrap_pre_text`] /
/// [`wrap_pre_line_text`].
///
/// The old code returned a `Vec`, i.e. one heap allocation per
/// pre-formatted block even for a single line. The common case is a
/// handful of visual lines, so the first [`PRE_PIECES_INLINE`] pieces
/// live inline; only longer blocks spill to a heap `Vec` (one
/// allocation, sized exactly).
///
/// The slice-like API (`Deref`, `Index`, [`retain`][Self::retain]) plus
/// by-value [`IntoIterator`] keep every use site working: the
/// `emit_text_frags` consumer in `layout.rs` iterates by value, and the
/// unit tests index and slice as before.
#[derive(Clone, Debug)]
pub(crate) struct PrePieces<'a> {
    inline: [PreWrapPiece<'a>; PRE_PIECES_INLINE],
    /// Live inline count when `spill` is `None`; unused otherwise.
    len: usize,
    spill: Option<Vec<PreWrapPiece<'a>>>,
}

impl<'a> PrePieces<'a> {
    fn new() -> Self {
        Self {
            inline: [PreWrapPiece {
                text: "",
                width: 0.0,
                break_before: false,
                forced_break: false,
            }; PRE_PIECES_INLINE],
            len: 0,
            spill: None,
        }
    }

    /// Append one piece. Stays inline (no allocation) until the inline
    /// capacity is exhausted, then spills to a heap vec (one
    /// allocation).
    fn push(&mut self, piece: PreWrapPiece<'a>) {
        if let Some(v) = self.spill.as_mut() {
            v.push(piece);
            return;
        }
        if self.len < PRE_PIECES_INLINE {
            self.inline[self.len] = piece;
            self.len += 1;
            return;
        }
        let mut v = Vec::with_capacity(PRE_PIECES_INLINE + 1);
        v.extend_from_slice(&self.inline);
        v.push(piece);
        self.spill = Some(v);
    }

    fn as_slice(&self) -> &[PreWrapPiece<'a>] {
        self.spill.as_deref().unwrap_or(&self.inline[..self.len])
    }

    fn as_mut_slice(&mut self) -> &mut [PreWrapPiece<'a>] {
        if let Some(v) = self.spill.as_mut() {
            v.as_mut_slice()
        } else {
            &mut self.inline[..self.len]
        }
    }

    /// Keep only the pieces matching `pred` (mirrors `Vec::retain`).
    pub(crate) fn retain(&mut self, mut pred: impl FnMut(&PreWrapPiece<'a>) -> bool) {
        if let Some(v) = self.spill.as_mut() {
            v.retain(pred);
            return;
        }
        // `PreWrapPiece: Copy`, so this is a plain memmove loop.
        let mut kept = 0usize;
        for i in 0..self.len {
            if pred(&self.inline[i]) {
                self.inline[kept] = self.inline[i];
                kept += 1;
            }
        }
        self.len = kept;
    }
}

impl<'a> std::ops::Deref for PrePieces<'a> {
    type Target = [PreWrapPiece<'a>];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a> std::ops::DerefMut for PrePieces<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'a> std::ops::Index<usize> for PrePieces<'a> {
    type Output = PreWrapPiece<'a>;

    fn index(&self, i: usize) -> &Self::Output {
        &self.as_slice()[i]
    }
}

impl<'a> std::ops::Index<std::ops::RangeFrom<usize>> for PrePieces<'a> {
    type Output = [PreWrapPiece<'a>];

    fn index(&self, r: std::ops::RangeFrom<usize>) -> &Self::Output {
        &self.as_slice()[r]
    }
}

impl<'a> std::ops::IndexMut<std::ops::RangeFrom<usize>> for PrePieces<'a> {
    fn index_mut(&mut self, r: std::ops::RangeFrom<usize>) -> &mut Self::Output {
        &mut self.as_mut_slice()[r]
    }
}

impl<'a> IntoIterator for PrePieces<'a> {
    type Item = PreWrapPiece<'a>;
    type IntoIter = PrePiecesIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        PrePiecesIter {
            pieces: self,
            idx: 0,
        }
    }
}

/// By-value iterator over [`PrePieces`] (see [`IntoIterator`]).
pub(crate) struct PrePiecesIter<'a> {
    pieces: PrePieces<'a>,
    idx: usize,
}

impl<'a> Iterator for PrePiecesIter<'a> {
    type Item = PreWrapPiece<'a>;

    fn next(&mut self) -> Option<PreWrapPiece<'a>> {
        // `PreWrapPiece: Copy`: no partial-move bookkeeping needed.
        let piece = *self.pieces.as_slice().get(self.idx)?;
        self.idx += 1;
        Some(piece)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.pieces.as_slice().len() - self.idx;
        (n, Some(n))
    }
}

impl<'a> ExactSizeIterator for PrePiecesIter<'a> {}

/// Segment `white-space: pre-wrap` text into visual lines. `break-spaces`
/// shares this path (CSS text-3 §4.1.2): every preserved space takes up
/// space — including trailing spaces at end-of-line, which are kept with
/// their measured width — and each space is its own wrap opportunity, so
/// runs of spaces may split across lines.
///
/// `text` must already be whitespace-preserved (see [`collapse_ws`]) but
/// may contain `\\n`. Each `\\n` forces a break; within a hard line, lines
/// wrap greedily at spaces (break opportunities), and a word longer than
/// `max_width` falls back to breaking between characters (break-all).
/// Every space survives: a run of spaces is preserved as real width,
/// never collapsed or trimmed.
///
/// `measure` maps a string to its advance width; the line breaker passes
/// `|s| fonts.advance(s, font_size)` with `max_width` set to the
/// containing block's content width.
///
/// Wiring: `emit_text_frags` in `layout.rs` emits one fragment per piece
/// — `break_before` becomes the fragment's line break, `forced_break`
/// its distinct paragraph-separator flag, and `width` its advance —
/// and the line breaker treats the block as wrapping. Pieces
/// are pre-fit to `max_width`, so the breaker only ever breaks at
/// `break_before`.
///
pub(crate) fn wrap_pre_text<'a>(
    text: &'a str,
    max_width: f32,
    measure: &dyn Fn(&str) -> f32,
) -> PrePieces<'a> {
    let mut pieces = PrePieces::new();
    for (hi, hard) in text.split('\n').enumerate() {
        // CRLF counts as one forced break.
        let hard = hard.strip_suffix('\r').unwrap_or(hard);
        // Every hard line after the first starts at a `\n` paragraph
        // separator: its first piece carries `forced_break`.
        wrap_hard_line(hard, max_width, measure, &mut pieces, hi > 0);
    }
    pieces
}

/// Segment `white-space: pre-line` text into visual lines.
///
/// `text` must already be collapsed for pre-line (see [`collapse_ws`]):
/// runs of spaces/tabs are single spaces and `\n` marks forced breaks.
/// Each hard line wraps exactly like [`wrap_pre_text`] (reusing
/// [`wrap_hard_line`], including its break-all fallback), but unlike
/// `pre-wrap` spaces at line boundaries do not survive: a leading space
/// on the first piece of each hard line and trailing spaces on every
/// piece are stripped (CSS 2.1 §16.6.1), and pieces that strip down to
/// nothing are dropped unless they carry a forced break (an empty line
/// still breaks).
///
/// Wiring: same as [`wrap_pre_text`] — one fragment per piece, pieces
/// pre-fit to `max_width`, and the line breaker treats the block as
/// pre-like (breaks only at `break_before`).
pub(crate) fn wrap_pre_line_text<'a>(
    text: &'a str,
    max_width: f32,
    measure: &dyn Fn(&str) -> f32,
) -> PrePieces<'a> {
    let mut pieces = PrePieces::new();
    for (hi, hard) in text.split('\n').enumerate() {
        let base = pieces.len();
        wrap_hard_line(hard, max_width, measure, &mut pieces, hi > 0);
        for (j, piece) in pieces[base..].iter_mut().enumerate() {
            let mut t = piece.text;
            if j == 0 {
                t = t.trim_start_matches(' ');
            }
            t = t.trim_end_matches(' ');
            piece.text = t;
            piece.width = measure(t);
        }
    }
    // A `forced_break` piece is never dropped here: it is the first
    // piece of a hard line after the first, so its `break_before` is
    // set, and the retain keeps `break_before` pieces even when they
    // strip to empty — the flag (and the empty line) reaches layout.
    pieces.retain(|p| !p.text.is_empty() || p.break_before);
    pieces
}

/// A word (maximal non-space run) or a single space, with byte offsets
/// into the hard line. Each space is its own token so runs of spaces are
/// preserved one by one; only the ASCII space is a break opportunity
/// (v1 simplification).
struct PreTok {
    start: usize,
    end: usize,
    is_space: bool,
    width: f32,
}

/// Lazy word/space tokenizer for one `pre-wrap` hard line (see
/// [`wrap_hard_line`]): yields each maximal non-space run and each
/// single space with its pre-measured width, so the greedy line filler
/// never materializes the old per-line `Vec<PreTok>`. Only the ASCII
/// space is a break opportunity (v1 simplification, as before).
struct PreTokens<'a, 'm> {
    hard: &'a str,
    chars: std::str::CharIndices<'a>,
    /// Byte offset where the in-progress word started, if any.
    word_start: Option<usize>,
    /// Byte offset of a space seen while a word was pending: the word
    /// is yielded first, this space next.
    pending_space: Option<usize>,
    space_w: f32,
    measure: &'m dyn Fn(&str) -> f32,
}

impl<'a, 'm> Iterator for PreTokens<'a, 'm> {
    type Item = PreTok;

    fn next(&mut self) -> Option<PreTok> {
        if let Some(ws) = self.word_start.take() {
            // Finish the pending word: it ends at the next space or at
            // end of input. Byte offsets stay on char boundaries: both
            // come from `char_indices`.
            for (i, ch) in self.chars.by_ref() {
                if ch == ' ' {
                    self.pending_space = Some(i);
                    let width = (self.measure)(&self.hard[ws..i]);
                    return Some(PreTok {
                        start: ws,
                        end: i,
                        is_space: false,
                        width,
                    });
                }
            }
            let width = (self.measure)(&self.hard[ws..]);
            return Some(PreTok {
                start: ws,
                end: self.hard.len(),
                is_space: false,
                width,
            });
        }
        if let Some(i) = self.pending_space.take() {
            return Some(PreTok {
                start: i,
                end: i + 1,
                is_space: true,
                width: self.space_w,
            });
        }
        if let Some((i, ch)) = self.chars.by_ref().next() {
            if ch == ' ' {
                return Some(PreTok {
                    start: i,
                    end: i + 1,
                    is_space: true,
                    width: self.space_w,
                });
            }
            // Start a word; the word arm above finishes it.
            self.word_start = Some(i);
            return self.next();
        }
        None
    }
}

fn wrap_hard_line<'a>(
    hard: &'a str,
    max_width: f32,
    measure: &dyn Fn(&str) -> f32,
    pieces: &mut PrePieces<'a>,
    // True when this hard line follows a `\n`: the FIRST piece this
    // call pushes gets `forced_break` (a paragraph separator); pieces
    // pushed for soft wraps within the line never do.
    first_forced: bool,
) {
    let call_start = pieces.len();
    // Space width is measured once per hard line, unconditionally, as
    // before (even blank lines paid this).
    let space_w = measure(" ");
    // Tokens stream lazily instead of materializing the old per-line
    // `Vec<PreTok>`: each word/space is measured once, on yield, while
    // the greedy filler consumes every token exactly once, so `measure`
    // is still called exactly once per word and once per space per hard
    // line. The call ORDER differs slightly from the old code — words
    // are measured on yield, interleaved with the piece measures below,
    // instead of all tokens measured up front — but the measure function
    // is pure, so the results are observationally identical.
    let mut toks = PreTokens {
        hard,
        chars: hard.char_indices(),
        word_start: None,
        pending_space: None,
        space_w,
        measure,
    }
    .peekable();

    if toks.peek().is_none() {
        // Blank hard line (from `a\n\nb` or a trailing `\n`): the empty
        // piece still forces the line break.
        pieces.push(PreWrapPiece {
            text: &hard[..0],
            width: 0.0,
            break_before: !pieces.is_empty(),
            forced_break: first_forced && pieces.len() == call_start,
        });
        return;
    }

    // Greedy fill: take tokens while they fit. The first token is always
    // taken, so even a zero/negative `max_width` makes progress.
    while let Some(t) = toks.next() {
        let mut w = t.width;
        let mut end = t.end;
        // `single` replaces the old `end == start + 1` check: exactly
        // one token taken so far.
        let mut single = true;
        while let Some(n) = toks.peek() {
            if w + n.width > max_width {
                break;
            }
            w += n.width;
            end = n.end;
            single = false;
            toks.next();
        }
        if single && !t.is_space && t.width > max_width {
            // Overlong word: break-all fallback between characters.
            // Only the first chunk of the hard line can carry the
            // `\n` forced break; later chunks are soft wraps.
            let chunk_forced = first_forced && pieces.len() == call_start;
            push_word_chunks(
                hard,
                t.start,
                t.end,
                max_width,
                measure,
                pieces,
                chunk_forced,
            );
        } else {
            let text = &hard[t.start..end];
            pieces.push(PreWrapPiece {
                text,
                width: measure(text),
                break_before: !pieces.is_empty(),
                forced_break: first_forced && pieces.len() == call_start,
            });
        }
    }
}

/// Break-all fallback: split `[word_start, word_end)` between characters
/// into chunks that fit `max_width`, pushing one piece per chunk.
fn push_word_chunks<'a>(
    hard: &'a str,
    word_start: usize,
    word_end: usize,
    max_width: f32,
    measure: &dyn Fn(&str) -> f32,
    pieces: &mut PrePieces<'a>,
    // True when the first chunk pushed here is also the first piece
    // of a hard line that follows a `\n` (see `wrap_hard_line`).
    first_forced: bool,
) {
    let call_start = pieces.len();
    let mut cs = word_start;
    while cs < word_end {
        let mut ce = cs;
        let mut w = 0.0;
        while ce < word_end {
            let cl = hard[ce..].chars().next().map_or(0, |c| c.len_utf8());
            if cl == 0 {
                break;
            }
            let cw = measure(&hard[ce..ce + cl]);
            if ce > cs && w + cw > max_width {
                break;
            }
            w += cw;
            ce += cl;
        }
        // The first char is always taken: progress even when a single
        // char is wider than `max_width`.
        let text = &hard[cs..ce];
        pieces.push(PreWrapPiece {
            text,
            width: measure(text),
            break_before: !pieces.is_empty(),
            forced_break: first_forced && pieces.len() == call_start,
        });
        cs = ce;
    }
}

/// Collect a node's children in document order into `buf` via the
/// `first_child`/`next_sibling` cursors — O(1) per hop, no per-call
/// allocation, unlike [`Document::children`]. `buf` is cleared first; it
/// is meant to be reused across pushes (see `build_box_tree`).
fn collect_children(doc: &Document, node: NodeId, buf: &mut Vec<NodeId>) {
    buf.clear();
    // A shadow host renders its shadow tree instead of its light-DOM
    // children: with no `<slot>` support there is nothing to project
    // light children into, so the shadow children simply become the
    // host box's children (the flat-tree slotting step of the full
    // spec is a documented follow-up). The fragment itself never
    // becomes a box — only its children do.
    let node = match doc.shadow_root(node) {
        Some(shadow) => shadow.root,
        None => node,
    };
    let mut cursor = doc.first_child(node);
    while let Some(n) = cursor {
        buf.push(n);
        cursor = doc.next_sibling(n);
    }
}

/// Build the box tree. Returns `None` when there is no rendered root
/// (no document element, or it is `display: none`).
///
/// `intrinsics` maps DOM nodes of replaced elements (`<img>` etc.) to
/// their intrinsic sizes; see [`is_replaced_tag`].
///
/// Test-only entry: the placeholder metrics return exactly 1.2em for
/// `line-height: normal`, so box-tree tests keep their exact
/// historical values. Production entries pass the real backend
/// through `build_box_tree_with_fonts` instead.
#[cfg(test)]
pub(crate) fn build_box_tree(
    doc: &Document,
    author: &[&Stylesheet],
    ua: &Stylesheet,
    vp_w: f32,
    vp_h: f32,
    intrinsics: &HashMap<NodeId, IntrinsicDims>,
) -> Option<(Forest, usize)> {
    build_box_tree_rooted(
        doc,
        author,
        ua,
        vp_w,
        vp_h,
        intrinsics,
        None,
        &crate::FixedFontMetrics,
    )
}

/// Like [`build_box_tree`], but `line-height: normal` resolves
/// through `fonts` (see [`super::style::resolve_line_height`]).
pub(crate) fn build_box_tree_with_fonts(
    doc: &Document,
    author: &[&Stylesheet],
    ua: &Stylesheet,
    vp_w: f32,
    vp_h: f32,
    intrinsics: &HashMap<NodeId, IntrinsicDims>,
    fonts: &dyn FontMetrics,
) -> Option<(Forest, usize)> {
    build_box_tree_rooted(doc, author, ua, vp_w, vp_h, intrinsics, None, fonts)
}

/// Build the box tree for the subtree rooted at `subtree_root` only
/// (wave 51 incremental layout): the same passes as
/// [`build_box_tree_with_fonts`], but the forest root is
/// `subtree_root`'s box and only its descendants are built. Returns
/// `None` when the subtree has no rendered root (not an element, a
/// `<template>`, detached, `display: none` on the root or on any
/// ancestor — an ancestor's `display: none` suppresses the whole
/// subtree in a whole-document pass too).
///
/// Inherited style context is *not* approximated: the ancestor chain
/// from the document element down to `subtree_root`'s parent is styled
/// root-first through the same [`Styler`], threading the inherited
/// font size and `direction` exactly as the whole-document pass
/// threads them through [`BuildItem`]s, and `root_font_size` still
/// comes from the document element (so `rem` inside the subtree means
/// what it means in the full tree). Everything below the root is
/// byte-for-byte the whole-document construction.
///
/// `line-height: normal` resolves through `fonts` (see
/// [`super::style::resolve_line_height`]).
// Mirrors the whole-document entry's argument set plus the subtree
// root and the font backend: the established shape for this crate's
// box-tree entries, hence the argument count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_box_tree_subtree_with_fonts(
    doc: &Document,
    author: &[&Stylesheet],
    ua: &Stylesheet,
    vp_w: f32,
    vp_h: f32,
    intrinsics: &HashMap<NodeId, IntrinsicDims>,
    subtree_root: NodeId,
    fonts: &dyn FontMetrics,
) -> Option<(Forest, usize)> {
    build_box_tree_rooted(
        doc,
        author,
        ua,
        vp_w,
        vp_h,
        intrinsics,
        Some(subtree_root),
        fonts,
    )
}

/// Count the DOM nodes under `root` (inclusive) for arena pre-sizing.
///
/// `Forest::nodes` holds 576 B nodes since fragments went inline-first;
/// growing it geometrically from `Vec::new()` copies every node
/// ~log2(N) times on large pages (each doubling step is a fresh
/// allocation + memcpy + free, and `layout()` builds the forest twice
/// when the scrollbar re-run triggers). One cheap O(nodes) walk up
/// front replaces that with (usually) a single allocation. The count is
/// a floor — anonymous boxes add more — so some growth can remain on
/// box-heavy pages; still strictly better than starting from zero.
///
/// Iterative with an explicit stack: recursion would risk stack overflow
/// on hostile DOM depth. The stack itself is one small, short-lived
/// allocation.
fn count_dom_nodes(doc: &Document, root: NodeId) -> usize {
    let mut count = 0usize;
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        count += 1;
        let mut kid = doc.first_child(n);
        while let Some(k) = kid {
            stack.push(k);
            kid = doc.next_sibling(k);
        }
    }
    count
}

// Mirrors the whole-document entry's argument set plus the subtree
// root and the font backend: the established shape for this crate's
// box-tree entries, hence the argument count.
#[allow(clippy::too_many_arguments)]
fn build_box_tree_rooted(
    doc: &Document,
    author: &[&Stylesheet],
    ua: &Stylesheet,
    vp_w: f32,
    vp_h: f32,
    intrinsics: &HashMap<NodeId, IntrinsicDims>,
    subtree_root: Option<NodeId>,
    fonts: &dyn FontMetrics,
) -> Option<(Forest, usize)> {
    let doc_root_el = doc.document_element()?;
    let root_el = match subtree_root {
        Some(node) => {
            // Only attached elements root a subtree layout; text and
            // other node kinds have no element box of their own (their
            // boxes are generated in their parent's formatting
            // context), and template contents never render.
            if !matches!(doc.kind(node), NodeKind::Element(_)) || tag_name(doc, node) == "template"
            {
                return None;
            }
            node
        }
        None => doc_root_el,
    };

    // One styler for the whole document: cascade/inheritance memoization
    // persists across elements (linear styling; see `Styler` docs).
    let styler = Styler::new(doc, author, ua, Viewport { w: vp_w, h: vp_h });

    // Root font size: the root element's computed font-size against the
    // 16px initial value (`rem` on the root correctly means 16px).
    let doc_root_style = compute_element_style(&styler, doc_root_el);
    let root_fs = resolve_font_size(
        doc_root_style.get_by_index(props::FONT_SIZE),
        DEFAULT_ROOT_FONT_SIZE,
        DEFAULT_ROOT_FONT_SIZE,
        vp_w,
        vp_h,
    );

    // Inherited context for a subtree root: thread the font size and
    // `direction` down the ancestor chain (document element first, so
    // the styler's memoization is reused root-first). For the
    // whole-document root both stay at their initial values.
    let mut parent_fs = DEFAULT_ROOT_FONT_SIZE;
    let mut parent_rtl = false;
    let mut parent_bidi = false;
    if subtree_root.is_some() && root_el != doc_root_el {
        let mut chain: Vec<NodeId> = Vec::new();
        let mut cursor = doc.parent(root_el);
        while let Some(ancestor) = cursor {
            chain.push(ancestor);
            if ancestor == doc_root_el {
                break;
            }
            cursor = doc.parent(ancestor);
        }
        // The chain must reach the document element; a root in a
        // detached or shadow tree that never reaches it is not part of
        // the rendered document.
        if chain.last() != Some(&doc_root_el) {
            return None;
        }
        chain.reverse();
        for ancestor in chain {
            let style = if ancestor == doc_root_el {
                doc_root_style.clone()
            } else {
                compute_element_style(&styler, ancestor)
            };
            let display = display_of(&style);
            if display == Display::None {
                return None;
            }
            if ancestor == doc_root_el {
                parent_fs = root_fs;
            } else {
                parent_fs =
                    StyleData::from_style(&style, display, parent_fs, root_fs, vp_w, vp_h, fonts)
                        .font_size;
            }
            parent_rtl = direction_rtl(&style, parent_rtl);
            parent_bidi = bidi_plaintext(&style, parent_bidi, display);
        }
    }

    let root_style = if root_el == doc_root_el {
        doc_root_style
    } else {
        compute_element_style(&styler, root_el)
    };

    let root_display = display_of(&root_style);
    if root_display == Display::None {
        return None;
    }

    let mut forest = Forest {
        // Pre-size the arena (see `count_dom_nodes`): geometric growth
        // from empty would otherwise dominate transient allocation on
        // large pages. Computed after the `display: none` early return
        // below, so a non-rendered root pays nothing.
        nodes: Vec::with_capacity(count_dom_nodes(doc, root_el)),
        root_font_size: root_fs,
        vp_w,
        vp_h,
        abs_items: Vec::new(),
        gutter_propagated_root: None,
        // A subtree root is never the viewport-propagated `html` root
        // (see `Forest::root_is_html`): viewport-level rules must not
        // shrink a partial pass's containing block.
        root_is_html: subtree_root.is_none()
            && matches!(&doc.kind(root_el), NodeKind::Element(e) if e.tag_name == "html"),
        quirks_mode: doc.quirks_mode(),
        #[cfg(test)]
        union_frag_visits: 0,
    };

    let mut root_data = StyleData::from_style(
        &root_style,
        root_display,
        parent_fs,
        root_fs,
        vp_w,
        vp_h,
        fonts,
    );
    root_data.rtl = direction_rtl(&root_style, parent_rtl);
    root_data.bidi_plaintext = bidi_plaintext(&root_style, parent_bidi, root_display);
    let root_box_type = match root_data.display {
        Display::Block => BoxType::Block,
        Display::Inline => BoxType::Inline,
        // The inline-level outer displays are atomic inline-level boxes:
        // their inner layout (flex/grid/table) is dispatched off
        // `StyleData::display`, not the box type.
        Display::InlineBlock | Display::InlineFlex | Display::InlineGrid | Display::InlineTable => {
            BoxType::InlineBlock
        }
        Display::Table => BoxType::Table,
        Display::Grid => BoxType::Grid,
        // Table-internal displays on the root degrade to block (CSS would
        // synthesize the missing table wrappers; that fix-up is table
        // layout's job for real tables, not the root).
        Display::TableRowGroup
        | Display::TableHeaderGroup
        | Display::TableFooterGroup
        | Display::TableRow
        | Display::TableCell => BoxType::Block,
        Display::TableColumn | Display::TableColumnGroup => BoxType::Block,
        // `display: contents` on the root computes to `block` (CSS display-3).
        Display::Contents => BoxType::Block,
        Display::None => return None,
    };
    let root_idx = forest.add_node(root_box_type, Some(root_style), root_data, None, None);
    forest.nodes[root_idx].dom = Some(root_el);
    if subtree_root.is_some() && is_replaced_tag(tag_name(doc, root_el)) {
        // The whole-document path sets intrinsics in pass 1 for every
        // element *below* the root; a replaced subtree root needs the
        // same treatment (sanitized like pass 1) or it would size 0x0.
        let tag = tag_name(doc, root_el);
        forest.nodes[root_idx].intrinsic =
            Some(sanitize_intrinsic(intrinsics.get(&root_el).copied(), tag));
    }

    // Pass 1: raw tree, iterative.
    let mut stack: Vec<BuildItem> = Vec::new();
    // Scratch buffer reused by every child push below: children are
    // collected via the `first_child`/`next_sibling` cursors (no
    // per-node allocation, unlike `Document::children`) and then pushed
    // onto the stack in reverse so the stack pops in document order.
    let mut child_buf: Vec<NodeId> = Vec::new();
    {
        let r = &forest.nodes[root_idx];
        let (fs, ws, tt, rtl, bidi) = (
            r.data.font_size,
            r.data.white_space,
            r.data.text_transform,
            r.data.rtl,
            r.data.bidi_plaintext,
        );
        collect_children(doc, root_el, &mut child_buf);
        for &c in child_buf.iter().rev() {
            stack.push(BuildItem {
                dom: c,
                parent: root_idx,
                inherit_fs: fs,
                ws,
                tt,
                rtl,
                bidi,
            });
        }
    }
    // Stylesheet-level gate for the per-element `::before` / `::after`
    // query below: one scan per document, so documents without
    // generated-content rules pay ~zero in the hot box-construction
    // loop (see `pseudo`).
    let has_generated_content = styler.has_generated_content_rules();
    // Deferred `::before` / `::after` boxes: the single-pass stack loop
    // builds an element's children *after* its box, so the pseudo
    // attach (prepend for `::before`, append for `::after`) runs once
    // the loop ends and every child list is complete. Only grows when
    // a pseudo actually renders.
    let mut pending_pseudos: Vec<(
        usize,
        super::pseudo::PseudoKind,
        super::pseudo::PseudoContent,
    )> = Vec::new();
    // Deferred `::after` *queries*: `::after` is the last child, so its
    // `content` text must resolve against the post-children counter
    // snapshot, which only exists once the subtree is processed. Maps
    // the originating DOM node to its box index and its (already
    // resolved, counter-independent) pseudo style; the element's
    // marker `BuildItem` below pops after all children and resolves
    // the text then. Only grows while subtrees are open.
    let mut pending_after: HashMap<NodeId, (usize, Rc<ComputedStyle>)> = HashMap::new();
    // Counter scopes for `counter()` / `counters()` in `content` (see
    // `pseudo::CounterScopes`): threaded only when some stylesheet
    // adjusts counters (see `Styler::has_counter_adjustments`).
    // Otherwise every counter reads as 0 and the stack stays empty.
    let has_counter_adjustments = styler.has_counter_adjustments();
    let mut counter_scopes = super::pseudo::CounterScopes::new();
    while let Some(item) = stack.pop() {
        // Deferred-`::after` marker: this element's subtree (every
        // child) is fully processed, so resolve its `::after` text
        // against the post-children snapshot and queue the box with
        // the other pseudos. The marker is just the element's own
        // `BuildItem` re-pushed below; it is distinguished here by its
        // presence in `pending_after` (element items are processed
        // before their entry is inserted).
        if has_counter_adjustments && let Some((box_idx, style)) = pending_after.remove(&item.dom) {
            let scopes = counter_scopes.snapshot_after(item.dom);
            if let Some(text) =
                super::pseudo::resolve_content_text(&styler, doc, item.dom, &style, &scopes)
            {
                pending_pseudos.push((
                    box_idx,
                    super::pseudo::PseudoKind::After,
                    super::pseudo::PseudoContent { style, text },
                ));
            }
            continue;
        }
        match doc.kind(item.dom) {
            NodeKind::Element(_) => {
                let tag = tag_name(doc, item.dom);
                // Template contents never render.
                let is_template = tag == "template";
                let style = compute_element_style(&styler, item.dom);
                let display = display_of(&style);
                // Enter this element's counter scope before the
                // `display: none` early-out: counters are an
                // element-tree concept (CSS Lists 3 §4), so a
                // non-rendered element's own `counter-reset` /
                // `counter-increment` still applies to the scopes its
                // siblings observe. (Its descendants are never visited,
                // so their adjustments are not applied.)
                // `display: contents` elements scope counters too
                // (their adjustments apply to the hoisted children).
                // The snapshot feeds the `::before` query below;
                // `::after` resolves against the post-children
                // snapshot via its deferred marker.
                let counter_snapshot: Vec<super::pseudo::CounterFrame> = if has_counter_adjustments
                {
                    counter_scopes.enter(doc, item.dom, &style)
                } else {
                    Vec::new()
                };
                if display == Display::None {
                    continue;
                }
                if display == Display::Contents {
                    // No box: children hoist to the parent box. The
                    // element still participates in inheritance, so its
                    // font size, `white-space`, `text-transform`, and
                    // `direction` travel with them.
                    let data = StyleData::from_style(
                        &style,
                        display,
                        item.inherit_fs,
                        root_fs,
                        vp_w,
                        vp_h,
                        fonts,
                    );
                    let (fs, ws, tt) = (data.font_size, data.white_space, data.text_transform);
                    let rtl = direction_rtl(&style, item.rtl);
                    // No box, no containing block: the ancestor scope
                    // propagates across `display: contents` unchanged
                    // (joined with this element's own `plaintext`).
                    let bidi = bidi_plaintext(&style, item.bidi, display);
                    collect_children(doc, item.dom, &mut child_buf);
                    for &c in child_buf.iter().rev() {
                        stack.push(BuildItem {
                            dom: c,
                            parent: item.parent,
                            inherit_fs: fs,
                            ws,
                            tt,
                            rtl,
                            bidi,
                        });
                    }
                    continue;
                }
                let mut data = StyleData::from_style(
                    &style,
                    display,
                    item.inherit_fs,
                    root_fs,
                    vp_w,
                    vp_h,
                    fonts,
                );
                data.rtl = direction_rtl(&style, item.rtl);
                // A block container derives this from its own computed
                // `unicode-bidi` alone; an inline box shares its
                // containing block's scope (see `bidi_plaintext`).
                data.bidi_plaintext = bidi_plaintext(&style, item.bidi, display);
                // Margins, floats, and clearance do not apply to internal
                // table boxes (CSS 2.1 §8.3.1/§9.5/§9.7); the table wrapper
                // itself keeps them.
                if matches!(
                    display,
                    Display::TableRowGroup
                        | Display::TableHeaderGroup
                        | Display::TableFooterGroup
                        | Display::TableRow
                        | Display::TableCell
                        | Display::TableColumn
                        | Display::TableColumnGroup
                ) {
                    data.margin = [SpecMargin::Len(SpecLen::Px(0.0)); 4];
                    data.float = Float::None;
                    data.clear = Clear::None;
                }
                let is_replaced = is_replaced_tag(tag);
                let box_type = match display {
                    Display::Block => BoxType::Block,
                    Display::Table => BoxType::Table,
                    Display::Grid => BoxType::Grid,
                    Display::TableRowGroup
                    | Display::TableHeaderGroup
                    | Display::TableFooterGroup => BoxType::TableRowGroup,
                    Display::TableRow => BoxType::TableRow,
                    Display::TableCell => BoxType::TableCell,
                    Display::TableColumn => BoxType::TableColumn,
                    Display::TableColumnGroup => BoxType::TableColumnGroup,
                    // Replaced elements are atomic inline-level boxes (CSS
                    // 2.1 §9.2.2.1); with no dedicated replaced layout,
                    // `InlineBlock` is the closest atomic model.
                    Display::Inline if is_replaced => BoxType::InlineBlock,
                    Display::Inline => BoxType::Inline,
                    // Atomic inline-level outer box; the inner layout
                    // (flex/grid/table) dispatches off
                    // `StyleData::display` in `layout_block`.
                    Display::InlineBlock
                    | Display::InlineFlex
                    | Display::InlineGrid
                    | Display::InlineTable => BoxType::InlineBlock,
                    // Handled above; kept for exhaustiveness.
                    Display::None | Display::Contents => continue,
                };
                let idx = forest.add_node(box_type, Some(style), data, None, Some(item.parent));
                forest.nodes[idx].dom = Some(item.dom);
                if box_type == BoxType::TableRowGroup {
                    // Stamp the section kind from the tag name while the
                    // `Document` is in scope (table layout has only the
                    // `Forest`); see [`SectionKind`].
                    forest.nodes[idx].section_kind =
                        Some(section_kind_of_tag_and_display(tag, display));
                }
                // Whether this box came from an HTML table element (see
                // [`Node::from_table_element`]): the tag name is only in
                // scope here; table layout sees just the `Forest`.
                forest.nodes[idx].from_table_element = is_table_element_tag(tag);
                // `<br>` forces a line break in inline layout (see
                // [`Node::is_br`]); stamped here for the same reason.
                forest.nodes[idx].is_br = tag == "br";
                // `<img>` identity for the quirks §3.8 run pass (see
                // [`Node::is_img`]): only `img` elements join a run.
                forest.nodes[idx].is_img = tag.eq_ignore_ascii_case("img");
                // `nowrap` on `td`/`th` for the quirks §3.9 pass
                // (see [`Node::nowrap_attr`]): presence of the
                // attribute, whatever its value.
                if box_type == BoxType::TableCell
                    && (tag.eq_ignore_ascii_case("td") || tag.eq_ignore_ascii_case("th"))
                {
                    forest.nodes[idx].nowrap_attr = doc.get_attribute(item.dom, "nowrap").is_some();
                }
                // `display: list-item` generates a marker box alongside
                // the principal (block) box (see `marker`).
                {
                    let is_list_item = forest.nodes[idx]
                        .style
                        .as_ref()
                        .and_then(|s| s.get_str_by_index(props::DISPLAY))
                        .is_some_and(|d| d.eq_ignore_ascii_case("list-item"));
                    if is_list_item {
                        let spec = {
                            let s = forest.nodes[idx].style.as_ref().expect("style just set");
                            super::marker::spec_for(doc, &styler, item.dom, s)
                        };
                        if let Some(spec) = spec {
                            super::marker::attach(
                                &mut forest,
                                idx,
                                spec,
                                root_fs,
                                vp_w,
                                vp_h,
                                fonts,
                            );
                        }
                    }
                }
                // `::before` / `::after` generated content (see
                // `pseudo`): `::before` is the first child, so its
                // query runs now against the entry snapshot and its box
                // attaches after the loop. `::after` is the last child:
                // only its (counter-independent, memoized) style
                // resolves now; its `content` text resolves against the
                // post-children snapshot when the deferred marker below
                // pops, then attaches after the loop like `::before`.
                // Without counter adjustments the entry snapshot is
                // already final, so `::after` queries inline as before.
                // Replaced elements never get pseudos; `display: none`
                // / `contents` already `continue`d above.
                if has_generated_content && !is_replaced {
                    if let Some(content) = super::pseudo::query_pseudo_content(
                        &styler,
                        doc,
                        item.dom,
                        super::pseudo::PseudoKind::Before,
                        &counter_snapshot,
                    ) {
                        pending_pseudos.push((idx, super::pseudo::PseudoKind::Before, content));
                    }
                    if has_counter_adjustments {
                        if let Some(style) = super::pseudo::query_after_style(&styler, item.dom) {
                            pending_after.insert(item.dom, (idx, style));
                            // Deferred-`::after` marker: pushed before
                            // the children below, so it pops after every
                            // child — the `::after` text then resolves
                            // against the post-children snapshot (see the
                            // top of the loop). Only `dom` matters to the
                            // marker path; the rest rides along ignored.
                            stack.push(BuildItem {
                                dom: item.dom,
                                parent: idx,
                                inherit_fs: item.inherit_fs,
                                ws: item.ws,
                                tt: item.tt,
                                rtl: item.rtl,
                                bidi: item.bidi,
                            });
                        }
                    } else if let Some(content) = super::pseudo::query_pseudo_content(
                        &styler,
                        doc,
                        item.dom,
                        super::pseudo::PseudoKind::After,
                        &counter_snapshot,
                    ) {
                        pending_pseudos.push((idx, super::pseudo::PseudoKind::After, content));
                    }
                }
                if box_type == BoxType::TableCell {
                    // `colspan` / `rowspan`: invalid or missing means 1;
                    // the HTML spec clamps to [0, 1000], where 0 means
                    // "span to the end of the table / section" (resolved
                    // at grid placement in `layout_table`).
                    let span = |name: &str| {
                        doc.get_attribute(item.dom, name)
                            .and_then(|v| v.trim().parse::<u32>().ok())
                            .unwrap_or(1)
                            .min(1000)
                    };
                    forest.nodes[idx].colspan = span("colspan");
                    forest.nodes[idx].rowspan = span("rowspan");
                }
                if matches!(box_type, BoxType::TableColumn | BoxType::TableColumnGroup) {
                    // `span` on `col` / `colgroup`: invalid or missing
                    // means 1, clamped to [1, 1000] (a zero span is
                    // invalid per HTML and degrades to 1). Stored in the
                    // `colspan` field, which doubles as the column span.
                    let span = doc
                        .get_attribute(item.dom, "span")
                        .and_then(|v| v.trim().parse::<u32>().ok())
                        .filter(|&v| v >= 1)
                        .unwrap_or(1)
                        .min(1000);
                    forest.nodes[idx].colspan = span;
                }
                if is_replaced {
                    // Intrinsic dimensions from the map; a missing entry
                    // is 0x0 for `<img>` (documented v1 behavior) and the
                    // CSS 300x150 default for `<iframe>`/`<embed>`/
                    // `<object>`/`<video>` (see
                    // [`default_intrinsic_for_tag`]). CSS width/height
                    // still override at layout time. Map entries are
                    // sanitized: NaN/negative/infinite sizes from the
                    // caller degrade to finite nonnegative instead of
                    // poisoning layout.
                    forest.nodes[idx].intrinsic =
                        Some(sanitize_intrinsic(intrinsics.get(&item.dom).copied(), tag));
                }
                if !is_template {
                    let (fs, ws, tt, rtl, bidi) = {
                        let n = &forest.nodes[idx];
                        (
                            n.data.font_size,
                            n.data.white_space,
                            n.data.text_transform,
                            n.data.rtl,
                            n.data.bidi_plaintext,
                        )
                    };
                    collect_children(doc, item.dom, &mut child_buf);
                    for &c in child_buf.iter().rev() {
                        stack.push(BuildItem {
                            dom: c,
                            parent: idx,
                            inherit_fs: fs,
                            ws,
                            tt,
                            rtl,
                            bidi,
                        });
                    }
                }
            }
            NodeKind::Text(t) => {
                // `text-transform` applies during whitespace collapsing,
                // before line breaking, so line breaking measures
                // transformed advances. Both steps borrow when they are
                // no-ops, and `shared_text` then stores the DOM's own
                // `Arc<str>` (zero copy); a fresh `Arc` is allocated
                // only when the bytes actually changed.
                let Some(transformed) = shared_text(t, item.ws, item.tt) else {
                    continue;
                };
                let parent = &forest.nodes[item.parent];
                let mut data = StyleData::inherit(&parent.data);
                // `direction` inherits through `display: contents` like
                // font size does: `item.rtl` carries the nearest specified
                // value among DOM ancestors (the box parent's misses
                // `display: contents` elements, which generate no box).
                data.rtl = item.rtl;
                // Text is not a containing block: it takes the
                // plaintext scope of the containing block it renders
                // in, which `item.bidi` carries through inline and
                // `display: contents` ancestors.
                data.bidi_plaintext = item.bidi;
                // Through `display: contents` the text hoists to the
                // parent box, but inheritance still passes through the
                // contents element: `inherit_fs` carries that element's
                // font size (without `display: contents` it always equals
                // the box parent's). Scale the inherited line height with
                // it; exact for `normal`/multiplier line heights.
                if (item.inherit_fs - parent.data.font_size).abs() > 0.01 {
                    data.line_height *= item.inherit_fs / parent.data.font_size;
                    data.font_size = item.inherit_fs;
                }
                // The text fragment stands in for its inline box during
                // line layout, so it aligns with the inline element's own
                // `vertical-align`. (`vertical-align` is not inherited, so
                // `inherit` resets it to baseline for block-level parents.)
                if parent.box_type == BoxType::Inline {
                    data.valign = parent.data.valign;
                }
                forest.add_node(
                    BoxType::Text,
                    None,
                    data,
                    Some(transformed),
                    Some(item.parent),
                );
                let text_idx = forest.nodes.len() - 1;
                forest.nodes[text_idx].dom = Some(item.dom);
            }
            _ => {
                // Comments, PIs, doctypes: no box, but descend just in
                // case (e.g. document fragments).
                collect_children(doc, item.dom, &mut child_buf);
                for &c in child_buf.iter().rev() {
                    stack.push(BuildItem {
                        dom: c,
                        parent: item.parent,
                        inherit_fs: item.inherit_fs,
                        ws: item.ws,
                        tt: item.tt,
                        rtl: item.rtl,
                        bidi: item.bidi,
                    });
                }
            }
        }
    }

    // Attach the deferred `::before` / `::after` boxes: every child
    // list is complete now, so `::before` prepends and `::after`
    // appends (see `pseudo::attach`). Runs before passes 2-4 so the
    // pseudo boxes flow through anonymous wrapping and whitespace
    // handling like any other inline-level box.
    for (idx, kind, content) in pending_pseudos {
        let origin_style = forest.nodes[idx]
            .style
            .as_ref()
            .expect("style just set")
            .clone();
        super::pseudo::attach(&mut forest, idx, kind, content, &origin_style, fonts);
    }

    // Pass 2: an inline box containing block-level children becomes a block
    // box (documented simplification).
    for i in 0..forest.nodes.len() {
        if forest.nodes[i].box_type == BoxType::Inline
            && forest.nodes[i]
                .children
                .iter()
                .any(|&c| is_block_level(forest.nodes[c].box_type))
        {
            forest.nodes[i].box_type = BoxType::Block;
            // As a block container it is now the containing block of
            // its inline content, so its plaintext flag comes from its
            // own computed `unicode-bidi` alone — the ancestor scope it
            // carried as an inline box no longer applies. (Its block
            // children already derived their own flags; its inline
            // runs are re-wrapped by pass 3 from this updated data.)
            let own = forest.nodes[i].style.as_deref().map(own_bidi_plaintext);
            if let Some(own) = own {
                forest.nodes[i].data.bidi_plaintext = own;
            }
        }
    }

    // Pass 3: wrap consecutive inline-level runs in anonymous blocks when
    // mixed with block-level children. This also covers inline-blocks and
    // table cells: their inline runs get a normal inline formatting
    // context inside the anonymous wrapper instead of zero-height rects
    // with no line layout. Tables get the same treatment so stray text
    // directly under a table becomes an anonymous block (table layout
    // skips whitespace-only ones and lays out the rest as captions).
    // Out-of-flow children (`position: absolute | fixed`) are hoisted past
    // the split: they neither join an inline run nor break one, so the
    // in-flow content around them still forms a single run (and a single
    // line).
    let mut i = 0;
    while i < forest.nodes.len() {
        let needs_wrap = matches!(
            forest.nodes[i].box_type,
            BoxType::Block
                | BoxType::InlineBlock
                | BoxType::TableCell
                | BoxType::Table
                | BoxType::Grid
        ) && forest.nodes[i]
            .children
            .iter()
            .any(|&c| is_block_level(forest.nodes[c].box_type));
        if needs_wrap {
            let parent_data = forest.nodes[i].data.clone();
            // w78: a flex or grid container's inline-level children
            // blockify (css-flexbox-1 §4, css-grid-1 §5.1) instead
            // of being swept into the anonymous wrapper below: the
            // wrapper is style-less, so the item's `flex-basis` /
            // placement / margins would never reach the item. This
            // is trigger one of two (see the joint notes on
            // `wrap_flex_text_runs` in `layout.rs` and
            // `wrap_grid_text_runs` in `layout_grid.rs` for trigger
            // two). The flex gate is the parent's raw `display`
            // (the bucketed `Display` collapses `flex` to `Block`,
            // so the raw keyword is what identifies a flex container
            // here, as in `layout::flex_kind`); the grid gate is the
            // raw keyword or the bucketed `Display`, which keeps
            // `Grid` / `InlineGrid` distinct. Non-flex/grid parents
            // keep the old wrapping: an inline-* box in block flow
            // stays an inline-level `InlineBlock` box. Only the box
            // type is mutated, never the style or `display`: the
            // display-keyed inner dispatch in `layout_block` (flex
            // via `flex_kind`, grid, table) still sees the original
            // display, so a promoted inline-flex still lays out as
            // flex and a promoted inline-grid still as grid.
            let parent_is_flex = forest.nodes[i]
                .style
                .as_ref()
                .and_then(|s| s.get_str_by_index(props::DISPLAY))
                .is_some_and(|d| {
                    d.eq_ignore_ascii_case("flex") || d.eq_ignore_ascii_case("inline-flex")
                });
            let parent_is_grid = forest.nodes[i]
                .style
                .as_ref()
                .and_then(|s| s.get_str_by_index(props::DISPLAY))
                .is_some_and(|d| {
                    d.eq_ignore_ascii_case("grid") || d.eq_ignore_ascii_case("inline-grid")
                })
                || matches!(
                    forest.nodes[i].data.display,
                    Display::Grid | Display::InlineGrid
                );
            let kids = std::mem::take(&mut forest.nodes[i].children);
            let mut new_children: Vec<usize> = Vec::with_capacity(kids.len());
            let mut run: Vec<usize> = Vec::new();
            // Table-internal boxes are never inline runs: like
            // block-level boxes they break runs and are kept as-is, so a
            // row group next to a caption is not swept into an anonymous
            // wrapper (table layout only looks for direct children).
            // Column boxes break runs too: they are structural, and
            // their widths resolve positionally.
            let breaks_run = |bt: BoxType| {
                is_block_level(bt)
                    || matches!(
                        bt,
                        BoxType::TableRowGroup
                            | BoxType::TableRow
                            | BoxType::TableColumn
                            | BoxType::TableColumnGroup
                    )
            };
            for c in kids {
                if is_out_of_flow(&forest, c) {
                    new_children.push(c);
                } else if (parent_is_flex || parent_is_grid)
                    && forest.nodes[c].box_type == BoxType::InlineBlock
                    && matches!(
                        forest.nodes[c].data.display,
                        Display::InlineBlock
                            | Display::InlineFlex
                            | Display::InlineGrid
                            | Display::InlineTable
                    )
                {
                    // w78 blockification (see above): flush any
                    // pending run, then promote the item to a direct
                    // child carrying its own style. The mapping is
                    // the spec's blockification table, with one
                    // Bowser-shaped entry: there is no `Flex` box
                    // type, so inline-flex blockifies to `Block`
                    // (its raw display still yields
                    // `flex_kind() == Some(false)`, so `layout_block`
                    // still dispatches it to `layout_flex`).
                    if !run.is_empty() {
                        let a =
                            forest.make_anonymous(&parent_data, std::mem::take(&mut run), i, false);
                        new_children.push(a);
                    }
                    forest.nodes[c].box_type = match forest.nodes[c].data.display {
                        Display::InlineGrid => BoxType::Grid,
                        Display::InlineTable => BoxType::Table,
                        _ => BoxType::Block,
                    };
                    new_children.push(c);
                } else if breaks_run(forest.nodes[c].box_type) {
                    if !run.is_empty() {
                        let a =
                            forest.make_anonymous(&parent_data, std::mem::take(&mut run), i, false);
                        new_children.push(a);
                    }
                    new_children.push(c);
                } else {
                    run.push(c);
                }
            }
            if !run.is_empty() {
                let a = forest.make_anonymous(&parent_data, run, i, false);
                new_children.push(a);
            }
            forest.nodes[i].children = new_children;
        }
        i += 1;
    }

    // Pass 4: drop whitespace-only text nodes adjacent to block-level boxes
    // or at block edges (pre-formatted text is exempt). In-place
    // compaction: the drop predicate reads the original neighbor list, and
    // compaction only ever writes to indices below the read cursor, so a
    // neighbor is never clobbered before it is read.
    for i in 0..forest.nodes.len() {
        let box_type = forest.nodes[i].box_type;
        if !matches!(
            box_type,
            BoxType::Block | BoxType::AnonymousBlock | BoxType::TableCell | BoxType::Grid
        ) {
            continue;
        }
        let mut kids = std::mem::take(&mut forest.nodes[i].children);
        let n = kids.len();
        let mut w = 0;
        for k in 0..n {
            let c = kids[k];
            let cn = &forest.nodes[c];
            let drop_it = cn.box_type == BoxType::Text
                && cn
                    .text
                    .as_deref()
                    .is_some_and(|t| t.chars().all(|ch| ch == ' '))
                && !matches!(
                    cn.data.white_space,
                    WhiteSpace::Pre | WhiteSpace::PreWrap | WhiteSpace::BreakSpaces
                )
                && {
                    let prev_block = k == 0 || is_block_level(forest.nodes[kids[k - 1]].box_type);
                    let next_block =
                        k + 1 == n || is_block_level(forest.nodes[kids[k + 1]].box_type);
                    prev_block || next_block
                };
            if !drop_it {
                kids[w] = c;
                w += 1;
            }
        }
        kids.truncate(w);
        forest.nodes[i].children = kids;
    }

    Some((forest, root_idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrinsic_defaults_by_tag() {
        // `<iframe>`/`<embed>`/`<object>`/`<video>` with no loaded
        // content: unknown dimensions (300x150 default, no ratio) per
        // CSS 2.1 §10.3.2; `<img>` with no entry stays 0x0.
        let iframe = sanitize_intrinsic(None, "iframe");
        assert_eq!(iframe.w, None);
        assert_eq!(iframe.h, None);
        assert_eq!(iframe.ratio, None);
        assert_eq!(iframe.width_or_default(), 300.0);
        assert_eq!(iframe.height_or_default(), 150.0);
        let img = sanitize_intrinsic(None, "img");
        assert_eq!(img, IntrinsicDims::from_size(0.0, 0.0));
        // Caller-supplied entries survive sanitization.
        let svg = sanitize_intrinsic(
            Some(IntrinsicDims {
                w: None,
                h: None,
                ratio: Some(10.0),
            }),
            "img",
        );
        assert_eq!(svg.ratio, Some(10.0));
        assert_eq!(svg.width_or_default(), 300.0);
    }

    #[test]
    fn section_kind_of_tag_and_display_maps_sections() {
        // The display wins: `table-header-group` / `table-footer-group`
        // stamp Header/Footer regardless of the element's tag, and only
        // plain `table-row-group` falls back to the tag (CSS 2.1
        // §17.2.1: the section displays "behave like a THEAD element").
        // Case-insensitive: the HTML parser lowercases, but XML trees
        // may not.
        assert_eq!(
            section_kind_of_tag_and_display("thead", Display::TableRowGroup),
            SectionKind::Header
        );
        assert_eq!(
            section_kind_of_tag_and_display("THEAD", Display::TableRowGroup),
            SectionKind::Header
        );
        assert_eq!(
            section_kind_of_tag_and_display("tfoot", Display::TableRowGroup),
            SectionKind::Footer
        );
        assert_eq!(
            section_kind_of_tag_and_display("TFOOT", Display::TableRowGroup),
            SectionKind::Footer
        );
        // Display wins over the tag: a thead restyled as a footer
        // group stamps Footer, and vice versa.
        assert_eq!(
            section_kind_of_tag_and_display("thead", Display::TableFooterGroup),
            SectionKind::Footer
        );
        assert_eq!(
            section_kind_of_tag_and_display("THEAD", Display::TableFooterGroup),
            SectionKind::Footer
        );
        assert_eq!(
            section_kind_of_tag_and_display("tfoot", Display::TableHeaderGroup),
            SectionKind::Header
        );
        assert_eq!(
            section_kind_of_tag_and_display("TFOOT", Display::TableHeaderGroup),
            SectionKind::Header
        );
        assert_eq!(
            section_kind_of_tag_and_display("tbody", Display::TableRowGroup),
            SectionKind::Body
        );
        assert_eq!(
            section_kind_of_tag_and_display("div", Display::TableRowGroup),
            SectionKind::Body
        );
        assert_eq!(
            section_kind_of_tag_and_display("div", Display::TableHeaderGroup),
            SectionKind::Header
        );
        assert_eq!(
            section_kind_of_tag_and_display("div", Display::TableFooterGroup),
            SectionKind::Footer
        );
        assert_eq!(
            section_kind_of_tag_and_display("tr", Display::TableRowGroup),
            SectionKind::Body
        );
    }

    #[test]
    fn forest_carries_document_quirks_mode() {
        // w77 quirks plumbing: parse -> document -> box tree, for all
        // three modes, so layout branches on the parsed mode.
        for (html, expected) in [
            (
                b"<!DOCTYPE html><html><body><p>x</p></body></html>".as_slice(),
                QuirksMode::NoQuirks,
            ),
            (
                b"<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\"><html><body><p>x</p></body></html>".as_slice(),
                QuirksMode::LimitedQuirks,
            ),
            (
                b"<html><body><p>x</p></body></html>".as_slice(),
                QuirksMode::Quirks,
            ),
        ] {
            let (doc, _) = Document::parse_html(html);
            assert_eq!(doc.quirks_mode(), expected);
            let ua = crate::ua_stylesheet();
            let (forest, _) = build_box_tree_with_fonts(&doc, &[], ua, 800.0, 600.0, &HashMap::new(), &crate::FixedFontMetrics)
                .expect("box tree builds");
            assert_eq!(forest.quirks_mode, expected);
        }
    }

    #[test]
    fn div_display_table_header_group_stamps_header() {
        // A `<div style="display: table-header-group">` stamps Header,
        // not Body (CSS 2.1 §17.2.1).
        let (doc, _) = Document::parse_html(
            b"<html><body><div style=\"display: table-header-group\"></div></body></html>",
        );
        let ua = crate::ua_stylesheet();
        let (forest, _) = build_box_tree_with_fonts(
            &doc,
            &[],
            ua,
            800.0,
            600.0,
            &HashMap::new(),
            &crate::FixedFontMetrics,
        )
        .expect("box tree builds");
        let group = forest
            .nodes
            .iter()
            .find(|n| n.box_type == BoxType::TableRowGroup)
            .expect("row-group box");
        assert_eq!(group.section_kind, Some(SectionKind::Header));
    }

    #[test]
    fn thead_display_table_footer_group_stamps_footer() {
        // Display wins over the tag: a `<thead>` restyled to
        // `table-footer-group` stamps Footer, not Header (CSS 2.1
        // §17.2.1).
        let (doc, _) = Document::parse_html(
            b"<html><body><table><thead style=\"display: table-footer-group\"><tr><td>x</td></tr></thead></table></body></html>",
        );
        let ua = crate::ua_stylesheet();
        let (forest, _) = build_box_tree_with_fonts(
            &doc,
            &[],
            ua,
            800.0,
            600.0,
            &HashMap::new(),
            &crate::FixedFontMetrics,
        )
        .expect("box tree builds");
        let group = forest
            .nodes
            .iter()
            .find(|n| n.box_type == BoxType::TableRowGroup)
            .expect("row-group box");
        assert_eq!(group.section_kind, Some(SectionKind::Footer));
    }

    #[test]
    fn tfoot_display_table_header_group_stamps_header() {
        // Display wins over the tag: a `<tfoot>` restyled to
        // `table-header-group` stamps Header, not Footer (CSS 2.1
        // §17.2.1).
        let (doc, _) = Document::parse_html(
            b"<html><body><table><tfoot style=\"display: table-header-group\"><tr><td>x</td></tr></tfoot></table></body></html>",
        );
        let ua = crate::ua_stylesheet();
        let (forest, _) = build_box_tree_with_fonts(
            &doc,
            &[],
            ua,
            800.0,
            600.0,
            &HashMap::new(),
            &crate::FixedFontMetrics,
        )
        .expect("box tree builds");
        let group = forest
            .nodes
            .iter()
            .find(|n| n.box_type == BoxType::TableRowGroup)
            .expect("row-group box");
        assert_eq!(group.section_kind, Some(SectionKind::Header));
    }

    /// 10px per char, like `FixedFontMetrics` at ~16.7px font size.
    fn measure(s: &str) -> f32 {
        s.chars().count() as f32 * 10.0
    }

    fn texts<'a>(pieces: &[PreWrapPiece<'a>]) -> Vec<(&'a str, bool)> {
        pieces.iter().map(|p| (p.text, p.break_before)).collect()
    }

    fn forced(pieces: &[PreWrapPiece<'_>]) -> Vec<bool> {
        pieces.iter().map(|p| p.forced_break).collect()
    }

    #[test]
    fn prewrap_preserves_spaces_without_wrapping() {
        let p = wrap_pre_text("a  b", 1000.0, &measure);
        assert_eq!(texts(&p), [("a  b", false)]);
        assert!((p[0].width - 40.0).abs() < 0.001);
    }

    #[test]
    fn prewrap_newlines_force_breaks() {
        let p = wrap_pre_text("ab\ncd", 1000.0, &measure);
        assert_eq!(texts(&p), [("ab", false), ("cd", true)]);
    }

    #[test]
    fn prewrap_forced_break_only_at_newlines_not_soft_wraps() {
        // Soft wraps set `break_before` but never `forced_break`
        // (UAX #9 P1: soft wraps are not paragraph separators).
        let p = wrap_pre_text("aaa bbb ccc", 45.0, &measure);
        assert_eq!(forced(&p), [false, false, false]);
        // A `\n` marks exactly its own first piece — even when the
        // hard lines themselves soft-wrap on both sides of it.
        let p = wrap_pre_text("aaa bbb\nccc ddd", 45.0, &measure);
        assert_eq!(
            texts(&p),
            [
                ("aaa ", false),
                ("bbb", true),
                ("ccc ", true),
                ("ddd", true)
            ]
        );
        assert_eq!(forced(&p), [false, false, true, false]);
        // Overlong-word break-all chunks after a `\n`: only the first
        // chunk is the paragraph separator.
        let p = wrap_pre_text("ab\ncdefgh", 25.0, &measure);
        assert_eq!(forced(&p), [false, true, false, false]);
        // A blank hard line's empty piece carries the separator.
        let p = wrap_pre_text("a\n\nb", 1000.0, &measure);
        assert_eq!(forced(&p), [false, true, true]);
        // Same distinction through the pre-line segmenter.
        let p = wrap_pre_line_text("aaa bbb\nccc", 45.0, &measure);
        assert_eq!(forced(&p), [false, false, true]);
    }

    #[test]
    fn prewrap_wraps_at_spaces() {
        // "aaa " is 40px; "aaa bbb" (70px) does not fit in 45px.
        let p = wrap_pre_text("aaa bbb ccc", 45.0, &measure);
        assert_eq!(texts(&p), [("aaa ", false), ("bbb ", true), ("ccc", true)]);
        assert!((p[0].width - 40.0).abs() < 0.001);
        assert!((p[2].width - 30.0).abs() < 0.001);
    }

    #[test]
    fn prewrap_overlong_word_breaks_between_chars() {
        let p = wrap_pre_text("abcdefgh", 25.0, &measure);
        assert_eq!(
            texts(&p),
            [("ab", false), ("cd", true), ("ef", true), ("gh", true)]
        );
    }

    #[test]
    fn prewrap_overlong_word_after_fitting_text() {
        // "aaa " fits in 45px; "bbbbbbbb" (80px) falls back to char breaks.
        let p = wrap_pre_text("aaa bbbbbbbb", 45.0, &measure);
        assert_eq!(texts(&p), [("aaa ", false), ("bbbb", true), ("bbbb", true)]);
    }

    #[test]
    fn prewrap_word_exactly_max_width_is_not_split() {
        let p = wrap_pre_text("abcd", 40.0, &measure);
        assert_eq!(texts(&p), [("abcd", false)]);
    }

    #[test]
    fn prewrap_blank_lines_are_kept() {
        let p = wrap_pre_text("a\n\nb", 1000.0, &measure);
        assert_eq!(texts(&p), [("a", false), ("", true), ("b", true)]);
    }

    #[test]
    fn prewrap_trailing_newline_keeps_empty_line() {
        let p = wrap_pre_text("ab\n", 1000.0, &measure);
        assert_eq!(texts(&p), [("ab", false), ("", true)]);
    }

    #[test]
    fn prewrap_crlf_is_one_break() {
        let p = wrap_pre_text("a\r\nb", 1000.0, &measure);
        assert_eq!(texts(&p), [("a", false), ("b", true)]);
    }

    #[test]
    fn prewrap_leading_spaces_are_preserved() {
        let p = wrap_pre_text("  ab", 1000.0, &measure);
        assert_eq!(texts(&p), [("  ab", false)]);
    }

    #[test]
    fn prewrap_zero_max_width_still_makes_progress() {
        let p = wrap_pre_text("ab", 0.0, &measure);
        assert_eq!(texts(&p), [("a", false), ("b", true)]);
    }

    #[test]
    fn prewrap_empty_input() {
        let p = wrap_pre_text("", 100.0, &measure);
        assert_eq!(texts(&p), [("", false)]);
    }

    #[test]
    fn prewrap_multibyte_chars_stay_on_char_boundaries() {
        // "héllo " is 60px; "héllo wörld" does not fit.
        let p = wrap_pre_text("héllo wörld", 60.0, &measure);
        assert_eq!(texts(&p), [("héllo ", false), ("wörld", true)]);
    }

    #[test]
    fn preline_collapse_ws_collapses_spaces_and_keeps_newlines() {
        assert_eq!(collapse_ws("a  b\t\tc", WhiteSpace::PreLine), "a b c");
        assert_eq!(collapse_ws("a  b\nc", WhiteSpace::PreLine), "a b\nc");
        assert_eq!(collapse_ws("a \u{000C} b", WhiteSpace::PreLine), "a b");
    }

    #[test]
    fn preline_collapse_ws_maps_lone_tab_and_formfeed_to_space() {
        // Regression: a lone tab or form feed (no 2-run) must still fire
        // the pre-scan — the allocating arm maps each to a single space.
        assert_eq!(collapse_ws("a\tb", WhiteSpace::PreLine), "a b");
        assert_eq!(collapse_ws("a\u{000C}b", WhiteSpace::PreLine), "a b");
    }

    #[test]
    fn preline_collapse_ws_normalizes_crlf_and_preserves_newlines() {
        assert_eq!(collapse_ws("a\r\nb", WhiteSpace::PreLine), "a\nb");
        assert_eq!(collapse_ws("a\rb", WhiteSpace::PreLine), "a\nb");
        // Segment breaks are not collapsible for pre-line: each one is a
        // preserved line feed (CSS text-3 §4.1.3).
        assert_eq!(collapse_ws("a\n\n\nb", WhiteSpace::PreLine), "a\n\n\nb");
        assert_eq!(collapse_ws("a\r\n\r\nb", WhiteSpace::PreLine), "a\n\nb");
    }

    #[test]
    fn preline_collapse_ws_leaves_other_modes_alone() {
        assert_eq!(collapse_ws("a  b\nc", WhiteSpace::Pre), "a  b\nc");
        assert_eq!(collapse_ws("a  b\nc", WhiteSpace::PreWrap), "a  b\nc");
        assert_eq!(collapse_ws("a  b\nc", WhiteSpace::BreakSpaces), "a  b\nc");
        assert_eq!(collapse_ws("a  b\nc", WhiteSpace::Normal), "a b c");
    }

    #[test]
    fn break_spaces_preserves_spaces_verbatim() {
        // Regression: `white-space: break-spaces` must not collapse runs.
        assert_eq!(collapse_ws("a  b", WhiteSpace::BreakSpaces), "a  b");
        assert_eq!(collapse_ws("a \t b", WhiteSpace::BreakSpaces), "a \t b");
        assert_eq!(collapse_ws("a  \n b", WhiteSpace::BreakSpaces), "a  \n b");
    }

    #[test]
    fn break_spaces_wrap_keeps_trailing_space_width() {
        // `wrap_pre_text` keeps the space at the wrap point with real
        // width and offers a break opportunity between consecutive
        // spaces: "a   b" at 30px (10px/char) fits "a  " (30px) then
        // wraps, leaving " b" for the next line.
        let p = wrap_pre_text("a   b", 30.0, &measure);
        assert_eq!(texts(&p), [("a  ", false), (" b", true)]);
        // Trailing spaces at end-of-line take up space: the single piece
        // measures the full run.
        let p = wrap_pre_text("a  ", 1000.0, &measure);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].text, "a  ");
        assert!((p[0].width - 30.0).abs() < 0.01);
    }

    #[test]
    fn preline_wrap_newlines_force_breaks() {
        let p = wrap_pre_line_text("ab\ncd", 1000.0, &measure);
        assert_eq!(texts(&p), [("ab", false), ("cd", true)]);
    }

    #[test]
    fn preline_wrap_wraps_at_spaces_and_trims_boundaries() {
        // "aaa bbb ccc" at 45px: unlike pre-wrap, the space at the wrap
        // point is trimmed from the end of the broken line.
        let p = wrap_pre_line_text("aaa bbb ccc", 45.0, &measure);
        assert_eq!(texts(&p), [("aaa", false), ("bbb", true), ("ccc", true)]);
        assert!((p[0].width - 30.0).abs() < 0.001);
    }

    #[test]
    fn preline_wrap_trims_leading_space_after_forced_break() {
        let p = wrap_pre_line_text("a\n b", 1000.0, &measure);
        assert_eq!(texts(&p), [("a", false), ("b", true)]);
    }

    #[test]
    fn preline_wrap_trims_surrounding_spaces() {
        let p = wrap_pre_line_text("  ab  ", 1000.0, &measure);
        assert_eq!(texts(&p), [("ab", false)]);
    }

    #[test]
    fn preline_wrap_trailing_newline_keeps_empty_line() {
        let p = wrap_pre_line_text("ab\n", 1000.0, &measure);
        assert_eq!(texts(&p), [("ab", false), ("", true)]);
    }

    #[test]
    fn preline_wrap_overlong_word_breaks_between_chars() {
        let p = wrap_pre_line_text("abcdefgh", 25.0, &measure);
        assert_eq!(
            texts(&p),
            [("ab", false), ("cd", true), ("ef", true), ("gh", true)]
        );
    }

    #[test]
    fn preline_wrap_all_space_input_yields_no_pieces() {
        let p = wrap_pre_line_text("   ", 1000.0, &measure);
        assert!(p.is_empty());
    }

    #[test]
    fn shared_text_noop_shares_source_arc() {
        // `normal` with nothing to collapse + no transform: borrowed
        // all the way through, so the stored Arc is the source Arc.
        let source: Arc<str> = Arc::from("hello world");
        let out =
            shared_text(&source, WhiteSpace::Normal, TextTransform::None).expect("non-empty text");
        assert!(Arc::ptr_eq(&out, &source));
        assert_eq!(&*out, "hello world");
        // `pre*` preserves verbatim: any bytes at all borrow.
        let source: Arc<str> = Arc::from("a  b\tc\nd");
        let out =
            shared_text(&source, WhiteSpace::Pre, TextTransform::None).expect("non-empty text");
        assert!(Arc::ptr_eq(&out, &source));
        // A transform that changes no character also borrows
        // (already-uppercase text under `uppercase`).
        let source: Arc<str> = Arc::from("HELLO WORLD");
        let out = shared_text(&source, WhiteSpace::Normal, TextTransform::Uppercase)
            .expect("non-empty text");
        assert!(Arc::ptr_eq(&out, &source));
    }

    #[test]
    fn shared_text_changed_bytes_allocate_fresh_arc() {
        // Collapse changes the bytes: fresh storage, collapsed content.
        let source: Arc<str> = Arc::from("a  b");
        let out =
            shared_text(&source, WhiteSpace::Normal, TextTransform::None).expect("non-empty text");
        assert!(!Arc::ptr_eq(&out, &source));
        assert_eq!(&*out, "a b");
        // Transform changes the bytes (collapse alone was a no-op).
        let source: Arc<str> = Arc::from("hello");
        let out = shared_text(&source, WhiteSpace::Normal, TextTransform::Uppercase)
            .expect("non-empty text");
        assert!(!Arc::ptr_eq(&out, &source));
        assert_eq!(&*out, "HELLO");
        // Pre-line CRLF normalization changes the bytes too.
        let source: Arc<str> = Arc::from("a\r\nb");
        let out =
            shared_text(&source, WhiteSpace::PreLine, TextTransform::None).expect("non-empty text");
        assert!(!Arc::ptr_eq(&out, &source));
        assert_eq!(&*out, "a\nb");
        // Empty output is dropped (no box), as with the old code.
        let source: Arc<str> = Arc::from("");
        assert!(shared_text(&source, WhiteSpace::Normal, TextTransform::None).is_none());
    }

    #[test]
    fn build_box_tree_text_node_shares_dom_arc() {
        let (doc, _) =
            bowser_dom::Document::parse_html(b"<html><body><p>hello world</p></body></html>");
        let dom_text: Arc<str> = doc
            .descendants(doc.document_element().expect("document element"))
            .into_iter()
            .find_map(|n| match doc.kind(n) {
                NodeKind::Text(t) => Some(Arc::clone(t)),
                _ => None,
            })
            .expect("DOM text node");
        let ua = crate::ua_stylesheet();
        let (forest, _) = build_box_tree_with_fonts(
            &doc,
            &[],
            ua,
            800.0,
            600.0,
            &HashMap::new(),
            &crate::FixedFontMetrics,
        )
        .expect("box tree builds");
        let layout_text = forest
            .nodes
            .iter()
            .find(|n| n.box_type == BoxType::Text)
            .and_then(|n| n.text.as_ref())
            .expect("layout text node");
        assert!(Arc::ptr_eq(layout_text, &dom_text));
        assert_eq!(&**layout_text, "hello world");
    }

    #[test]
    fn anonymous_table_wrappers_reset_bidi_plaintext_scope() {
        // N3: the table fix-up wraps stray content in anonymous
        // rows/cells whose computed `unicode-bidi` is the initial
        // `normal` — the table's plaintext scope must stop at the
        // block boundary instead of leaking into the wrappers'
        // paragraphs (where it would flip each paragraph's base
        // direction to first-strong).
        let (doc, _) = Document::parse_html(b"<div class=\"t\">x</div>");
        let ss = bowser_css::parse_stylesheet(".t { display: table; unicode-bidi: plaintext; }");
        let ua = crate::ua_stylesheet();
        let (mut forest, _) = build_box_tree_with_fonts(
            &doc,
            &[&ss],
            ua,
            800.0,
            600.0,
            &HashMap::new(),
            &crate::FixedFontMetrics,
        )
        .expect("box tree builds");
        let table = forest
            .nodes
            .iter()
            .position(|n| n.box_type == BoxType::Table)
            .expect("table box");
        assert!(
            forest.nodes[table].data.bidi_plaintext,
            "the table takes its own plaintext scope"
        );
        crate::layout::layout_table::wrap_table_stray_content(&mut forest, table);
        let anon = |bt: BoxType| {
            forest
                .nodes
                .iter()
                .find(|n| n.box_type == bt && n.style.is_none())
                .expect("anonymous wrapper")
        };
        assert!(
            !anon(BoxType::TableRow).data.bidi_plaintext,
            "anonymous row resets the plaintext scope"
        );
        assert!(
            !anon(BoxType::TableCell).data.bidi_plaintext,
            "anonymous cell resets the plaintext scope"
        );
    }

    /// Find the forest box generated for the first element with `tag`.
    fn box_for_tag(forest: &Forest, doc: &bowser_dom::Document, tag: &str) -> usize {
        let dom = doc
            .get_elements_by_tag_name(tag)
            .item(0)
            .unwrap_or_else(|| panic!("expected a <{tag}> element"));
        forest
            .nodes
            .iter()
            .position(|n| n.dom == Some(dom))
            .unwrap_or_else(|| panic!("expected a box for <{tag}>"))
    }

    #[test]
    fn after_pseudo_sees_descendant_counter_increments() {
        // CSS Lists 3: `::after` is the last child, so its `counter()`
        // values observe all in-scope increments including the
        // descendants' — `ol::after` reads the total (3), not the
        // entry-time value (0). The text lands after every `li` in
        // document order (pass 3 may wrap the trailing pseudo in an
        // anonymous block, so collect text in order instead of
        // assuming a direct child).
        use bowser_dom::Document;
        let (doc, _) = Document::parse_html(b"<ol><li>a</li><li>b</li><li>c</li></ol>");
        let ua = bowser_css::default_ua_stylesheet();
        let ss = bowser_css::parse_stylesheet(
            "ol { counter-reset: n; } li { counter-increment: n; } \
             ol::after { content: \"total=\" counter(n); }",
        );
        let author = [&ss];
        let (forest, _) = build_box_tree_with_fonts(
            &doc,
            &author,
            &ua,
            800.0,
            600.0,
            &HashMap::new(),
            &crate::FixedFontMetrics,
        )
        .expect("box tree");
        let ol = box_for_tag(&forest, &doc, "ol");
        fn texts_under(forest: &Forest, idx: usize, out: &mut Vec<String>) {
            let n = &forest.nodes[idx];
            if let Some(t) = n.text.as_deref() {
                out.push(t.to_owned());
            }
            for &c in &n.children {
                texts_under(forest, c, out);
            }
        }
        let mut texts = Vec::new();
        texts_under(&forest, ol, &mut texts);
        assert_eq!(texts, vec!["a", "b", "c", "total=3"]);
    }

    #[test]
    fn display_none_element_still_applies_counter_adjustments() {
        // CSS Lists 3 §4: counters are an element-tree concept — a
        // `display: none` element's own `counter-increment` still
        // applies, so the next visible sibling observes it (2, not 1).
        use bowser_dom::Document;
        let (doc, _) =
            Document::parse_html(b"<ol><li style=\"display: none\">a</li><li>b</li></ol>");
        let ua = bowser_css::default_ua_stylesheet();
        let ss = bowser_css::parse_stylesheet(
            "ol { counter-reset: n; } li { counter-increment: n; } \
             li::before { content: counter(n) \". \"; }",
        );
        let author = [&ss];
        let (forest, _) = build_box_tree_with_fonts(
            &doc,
            &author,
            &ua,
            800.0,
            600.0,
            &HashMap::new(),
            &crate::FixedFontMetrics,
        )
        .expect("box tree");
        // Only the visible `li` generates a box …
        let visible_lis: Vec<usize> = forest
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.dom.is_some_and(|d| tag_name(&doc, d) == "li"))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(visible_lis.len(), 1);
        // … and its `::before` (first child) read the hidden sibling's
        // increment.
        let li = &forest.nodes[visible_lis[0]];
        let before_box = &forest.nodes[li.children[0]];
        assert_eq!(before_box.box_type, BoxType::Inline);
        assert_eq!(before_box.children.len(), 1);
        assert_eq!(
            forest.nodes[before_box.children[0]].text.as_deref(),
            Some("2. ")
        );
    }
}

#[cfg(test)]
mod node_size_tests {
    /// Regression budget for the arena node. `StyleData` (304B) rides
    /// inline instead of boxed: boxing it was one heap allocation per
    /// box per layout pass, and the field is hot anyway (font size,
    /// display, margins are read for nearly every node), so inlining
    /// also removes a pointer chase per access. `fragments` rides
    /// behind [`super::FragmentList`] (48B inline-first: one
    /// `TextFragment` inline, spill `Vec` for two or more): only text
    /// boxes ever carry fragments, and the common single-fragment case
    /// costs no allocation (see [`super::FragmentList`]). 576B measured
    /// 2026-10-23 (536B with the old 8B boxed fragments, 552B before
    /// the fragments box, 272B with `data` boxed); the bound leaves
    /// headroom for legitimate growth.
    #[test]
    fn node_stays_small() {
        assert!(
            std::mem::size_of::<super::Node>() <= 640,
            "Node grew past its size budget: {} bytes",
            std::mem::size_of::<super::Node>()
        );
    }
}

#[cfg(test)]
mod fragment_list_tests {
    use super::FragmentList;
    use crate::{Rect, TextFragment};

    fn frag(w: f32) -> TextFragment {
        TextFragment {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w,
                h: 10.0,
            },
            baseline: 8.0,
            text_range: 0..1,
        }
    }

    #[test]
    fn field_is_inline_first() {
        // One `TextFragment` (40B) inline + spill `Vec`; the old
        // 8-byte `Option<Box<Vec<_>>>` is gone.
        assert_eq!(std::mem::size_of::<FragmentList>(), 48);
    }

    #[test]
    fn empty_list_derefs_to_empty_slice() {
        let l = FragmentList::new();
        assert!(l.is_empty());
        assert_eq!(l.len(), 0);
        assert_eq!(l.iter().count(), 0);
        // `into_vec` on an empty list allocates nothing either.
        assert!(l.into_vec().is_empty());
    }

    #[test]
    fn push_clear_retain_round_trip() {
        let mut l = FragmentList::new();
        l.push(frag(10.0));
        l.push(frag(20.0));
        l.push(frag(30.0));
        assert_eq!(l.len(), 3);
        assert_eq!(l[1].rect.w, 20.0);
        // `retain` drops non-matching fragments in place.
        l.retain(|f| f.rect.w >= 20.0);
        assert_eq!(l.len(), 2);
        assert_eq!(l[0].rect.w, 20.0);
        // `&mut` iteration still reaches the fragments.
        for f in &mut l {
            f.rect.x += 1.0;
        }
        assert_eq!(l[0].rect.x, 1.0);
        // `clear` empties without freeing the box.
        l.clear();
        assert!(l.is_empty());
        // `into_vec` moves the buffer out without reallocating.
        l.push(frag(5.0));
        let v = l.into_vec();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].rect.w, 5.0);
    }

    #[test]
    fn clone_and_default() {
        let mut l = FragmentList::new();
        l.push(frag(7.0));
        let c = l.clone();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].rect.w, 7.0);
        let d = FragmentList::default();
        assert!(d.is_empty());
    }
}

/// Wave-65 perf tests: inline-first [`FragmentList`] (task 1) and the
/// allocation-free [`PrePieces`] pre-wrap path (task 2).
///
/// Allocation behavior is pinned structurally here (inline capacities,
/// spill sizes); absolute alloc counts are measured by the
/// `allocbench` harness in `~/workspace/scratch/w65` against
/// `bench_layout`'s inline case.
#[cfg(test)]
mod w65_p_layout_tests {
    use super::{FragmentList, PrePieces, PreWrapPiece, wrap_pre_line_text, wrap_pre_text};
    use crate::{Rect, TextFragment};

    fn frag(w: f32) -> TextFragment {
        TextFragment {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w,
                h: 10.0,
            },
            baseline: 8.0,
            text_range: 0..1,
        }
    }

    fn measure(s: &str) -> f32 {
        s.len() as f32 * 10.0
    }

    #[test]
    fn fragment_list_is_48_bytes() {
        // One inline `TextFragment` (40B) + discriminant; the spill
        // `Vec` (24B) fits under it.
        assert_eq!(std::mem::size_of::<FragmentList>(), 48);
    }

    #[test]
    fn single_fragment_into_vec_packs_one_element() {
        // The public tree still takes a `Vec`, but the common
        // single-fragment case packs into a 1-capacity (40 B) vec —
        // not the old box + 4-capacity (160 B) buffer.
        let mut l = FragmentList::new();
        l.push(frag(3.0));
        assert_eq!(l.len(), 1);
        let v = l.into_vec();
        assert_eq!(v.len(), 1);
        assert_eq!(v.capacity(), 1);
        assert_eq!(v[0].rect.w, 3.0);
    }

    #[test]
    fn second_push_spills_exact_two() {
        // The spill vec is allocated for exactly 2, not 4.
        let mut l = FragmentList::new();
        l.push(frag(1.0));
        l.push(frag(2.0));
        assert_eq!(l.len(), 2);
        assert_eq!(l[1].rect.w, 2.0);
        let v = l.into_vec();
        assert_eq!(v.capacity(), 2);
    }

    #[test]
    fn into_vec_spilled_moves_buffer() {
        let mut l = FragmentList::new();
        l.push(frag(1.0));
        l.push(frag(2.0));
        l.push(frag(3.0));
        let v = l.into_vec();
        assert_eq!(v.len(), 3);
        assert_eq!(v[2].rect.w, 3.0);
    }

    #[test]
    fn clear_and_retain_on_inline() {
        let mut l = FragmentList::new();
        l.push(frag(1.0));
        l.clear();
        assert!(l.is_empty());
        // Re-push after clear stays inline (no allocation to keep).
        l.push(frag(7.0));
        assert_eq!(l.len(), 1);
        l.retain(|f| f.rect.w > 5.0);
        assert_eq!(l.len(), 1);
        l.retain(|f| f.rect.w > 50.0);
        assert!(l.is_empty());
    }

    #[test]
    fn pre_pieces_small_block() {
        // Old code: one `Vec` per pre block even here. New: inline.
        let p = wrap_pre_text("aaa bbb", 45.0, &measure);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].text, "aaa ");
        assert_eq!(p[1].text, "bbb");
        assert!(!p[0].break_before);
        assert!(p[1].break_before);
    }

    #[test]
    fn pre_pieces_spill_yields_all_by_value() {
        // 12 hard lines -> 12 pieces > 8 inline: spill-path iteration
        // order matches the old `Vec` order.
        let text = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl";
        let p: PrePieces<'_> = wrap_pre_text(text, 1000.0, &measure);
        assert_eq!(p.len(), 12);
        let texts: Vec<&str> = p.into_iter().map(|piece| piece.text).collect();
        assert_eq!(
            texts,
            ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
        );
    }

    #[test]
    fn pre_line_trims_and_keeps_flags() {
        // Trim + retain on the inline path; break_before / forced_break
        // semantics unchanged.
        let p = wrap_pre_line_text("aaa bbb\nccc", 45.0, &measure);
        assert_eq!(p.len(), 3);
        assert_eq!(p[0].text, "aaa");
        assert_eq!(p[1].text, "bbb");
        assert_eq!(p[2].text, "ccc");
        let flags: Vec<(bool, bool)> = p
            .into_iter()
            .map(|piece| (piece.break_before, piece.forced_break))
            .collect();
        assert_eq!(flags, [(false, false), (true, false), (true, true)]);
    }

    #[test]
    fn pre_pieces_deref_slice_api() {
        // The slice-like API the old `Vec` return offered still works.
        let p = wrap_pre_text("ab\ncd", 1000.0, &measure);
        assert!(!p.is_empty());
        let first: &PreWrapPiece<'_> = &p[0];
        assert_eq!(first.text, "ab");
        assert_eq!(p.iter().count(), 2);
    }

    #[test]
    fn count_dom_nodes_counts_subtree() {
        use bowser_dom::Document;
        let (doc, _) = Document::parse_html(b"<html><body><p>a<b>b</b>c</p><p>d</p></body></html>");
        let root = doc.document_element().unwrap();
        // html, body, p, text(a), b, text(b), text(c), p, text(d) = 9.
        // (Whitespace text nodes, if any, only raise the floor — the
        // count is a pre-size hint, not exact.)
        let n = super::count_dom_nodes(&doc, root);
        assert!(n >= 9, "expected at least 9 DOM nodes, got {n}");
        // Subtree root counts just its subtree.
        let p = doc.get_elements_by_tag_name("p").item(0).unwrap();
        let m = super::count_dom_nodes(&doc, p);
        assert!(m >= 5 && m < n, "subtree count {m} out of range");
    }
}
