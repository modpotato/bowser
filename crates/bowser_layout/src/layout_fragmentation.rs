//! Print/PDF fragmentation: slicing one infinitely tall layout tree into pages.
//!
//! # Design: how fragmentation fits this engine
//!
//! Fragmentation is a pure post-pass over the single-page [`LayoutBox`]
//! tree. `layout()` still lays out one infinitely tall page, and this
//! module never re-runs layout, so the single-page path is byte-identical
//! with or without fragmentation.
//!
//! A page is a *fragmentainer*: the horizontal band `[p*H, (p+1)*H)` of the
//! original coordinate space, where `H` is the page height. Each
//! [`PageFragment`] holds a clone of the tree slice intersecting its band,
//! with rects translated so `y = 0` is the page top; paint consumes each
//! page as an ordinary `LayoutBox` tree.
//!
//! Three phases. (1) *Pagination* walks the fragmentation context's in-flow
//! block-level children in document order (normally `<body>`'s children)
//! and applies the break rules, recording one y-shift per child:
//! `break-before: page` jumps to the next page top, `break-after: page`
//! pushes following content there, `break-before/after: avoid` keeps a box
//! with its neighbor (keep-with-next via one-box lookahead), and
//! `break-inside: avoid` moves a whole box to the next page when it fits on
//! one. Boxes taller than a page always fragment: avoidance never loops and
//! never piles up unbounded shifts. (1b) *Nested-break pre-pass* walks the
//! whole tree iteratively and collects the forced-break hints pagination
//! cannot see (`break-before: page`, `break-after: page`,
//! `break-after: avoid` on boxes nested inside a flow unit) into a per-box
//! shift map. The running shift only ever moves down, mirroring phase 1's
//! cursor invariant, so nested shifts stay monotonic and page assignment
//! stays consistent. (2) *Splitting* walks the tree iteratively (no
//! recursion, so hostile DOM depth cannot overflow the stack) and clones,
//! per page, the boxes intersecting that page's band; text boxes split by
//! line fragment, so paragraphs flow across pages. A nested
//! `break-inside: avoid` box is never split: when it straddles a page
//! boundary it moves whole to the top of the next page, mirroring phase
//! 1's re-pagination of top-level avoid boxes (boxes taller than a page
//! still fragment: avoidance never loops). Table rows are likewise never
//! split mid-row when the row fits on a page, and neither are floats or
//! absolutely positioned boxes: a float or abs-pos box that straddles a
//! page boundary moves whole to the top of the next page instead of being
//! sliced mid-box. When a table's rows flow across pages, its `<thead>`
//! row groups repeat at the top of each continuation page and its
//! `<tfoot>` row groups repeat at the bottom.
//!
//! Break values resolve from the computed style: `bowser_css` registers
//! `break-before`/`break-after`/`break-inside`/`widows`/`orphans`, so
//! author stylesheets set them through the normal cascade. The explicit
//! per-node override map is only an override: entries in it win over the
//! computed style, and boxes without an entry fragment with their
//! cascade-derived hints (anonymous boxes, which have no node, always
//! use the computed style).
//!
//! `widows`/`orphans` are enforced during pagination: when a text block
//! would split across pages, the split point moves so at least `orphans`
//! lines stay at the bottom of the first fragment and at least `widows`
//! lines start the last fragment; when the first fragment cannot keep
//! `orphans` lines the whole block moves to the next page. (A block with
//! fewer lines than `orphans` + `widows` cannot satisfy both at once:
//! `orphans` wins and `widows` is best-effort.) `position: fixed` boxes
//! repeat on every page (CSS print behavior), including fixed boxes nested
//! inside flow content; absolutely positioned boxes are emitted whole on
//! the page holding their top edge (moving whole to the next page when
//! they straddle a boundary and fit on one).
//!
//! # Bounded caps (hostile inputs)
//!
//! - Page height is clamped to `[1.0, 1e15]` px (`sane_len`, then a 1px
//!   floor): zero/negative/NaN heights cannot divide by zero or spin.
//! - Page count is capped at [`MAX_PAGES`]; gigantic content is truncated,
//!   never looped over forever.
//! - Avoidance never defers a box more than one page: a taller-than-page
//!   `break-inside: avoid` box fragments instead of shifting forever.

use super::{BoxType, IntrinsicDims, LayoutBox, Rect, Size, sane_len};
use bowser_css::Stylesheet;
use bowser_css::props;
use bowser_dom::{Document, NodeId};
use std::collections::HashMap;
use std::ptr;

/// Hard cap on emitted pages: a hostile multi-kilometer-tall element must
/// truncate, not hang the printer or OOM the process. 10k pages is far
/// beyond any real document; each page only clones the boxes it shows.
pub const MAX_PAGES: usize = 10_000;

/// Tolerance for float dust when comparing against page boundaries (px).
/// Layout sums lengths in f32, so a box meant to sit exactly on a boundary
/// can land a fraction of a sub-pixel off; without this, dust would
/// trigger spurious forced breaks or missed fits.
const EPS: f32 = 0.001;

/// Fix-up rounds for widows/orphans re-pagination, per flow unit. Each
/// orphans round advances the unit a full page and each widows round either
/// settles the split or triggers an orphans round, so real documents settle
/// in 1-3 rounds; the cap only bounds hostile hint values (e.g. absurd
/// `orphans`) to a bounded extra shift.
const WIDOW_ORPHAN_PASSES: u32 = 8;

/// A break-before/break-after value (CSS Fragmentation 3 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Break {
    /// No break requirement; break wherever the paginator must.
    Auto,
    /// Avoid breaking at this point (weak hint: honored when possible).
    Avoid,
    /// Force a break: the box starts at the top of a fresh page.
    Page,
}

/// A break-inside value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BreakInside {
    Auto,
    /// Keep the box on one page when it fits; fragment it when taller
    /// than a page (never loop, never shift unboundedly).
    Avoid,
}

/// Break hints for one box.
///
/// `break-*`/`orphans`/`widows` values resolve from the computed style
/// through the normal cascade (`bowser_css` registers all five
/// properties, so author stylesheets set them directly). The per-node
/// override map passed to [`layout_fragmented_with_breaks`] wins over
/// the computed style.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreakHints {
    pub before: Break,
    pub after: Break,
    pub inside: BreakInside,
    /// Minimum line boxes kept at the bottom of the first fragment when
    /// the block splits across pages (CSS initial: 2).
    pub orphans: u32,
    /// Minimum line boxes starting the last fragment when the block splits
    /// across pages (CSS initial: 2).
    pub widows: u32,
}

impl Default for BreakHints {
    fn default() -> Self {
        BreakHints {
            before: Break::Auto,
            after: Break::Auto,
            inside: BreakInside::Auto,
            // CSS initial values.
            orphans: 2,
            widows: 2,
        }
    }
}

impl BreakHints {
    /// `page` break on both sides: start a fresh page and end on one.
    pub fn page_break() -> Self {
        BreakHints {
            before: Break::Page,
            after: Break::Auto,
            inside: BreakInside::Auto,
            orphans: 2,
            widows: 2,
        }
    }
}

/// One printed page: the slice of the layout tree for fragmentainer
/// `page_index`.
///
/// `root` mirrors the single-page tree's shape (document element box at the
/// top, fragmentation context below it) but contains only the boxes
/// intersecting this page, with every `rect` (and text fragment rect and
/// baseline) translated so `y = 0` is the page top. `size` is the page size
/// in CSS px: `(viewport.w, page_height)`.
#[derive(Clone, Debug)]
pub struct PageFragment {
    pub page_index: usize,
    pub page_count: usize,
    pub size: Size,
    pub root: LayoutBox,
}

/// Lay out `doc` and fragment it into pages of `page_height` CSS px.
///
/// Equivalent to [`layout_fragmented_with_breaks`] with no break overrides.
/// The single-page [`super::layout`] path is untouched by this: same
/// internal layout call, then a pure post-pass.
pub fn layout_fragmented(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn super::FontMetrics,
    page_height: f32,
) -> Vec<PageFragment> {
    fragment_impl(
        doc,
        author,
        viewport,
        fonts,
        page_height,
        &HashMap::new(),
        &HashMap::new(),
        (0.0, 0.0),
    )
}

/// Lay out `doc` and fragment it into pages of `page_height` CSS px,
/// sizing replaced elements from `intrinsics`.
///
/// Mirrors [`super::layout_with_intrinsics`]: replaced elements with
/// `auto` CSS width/height take their box size from `intrinsics`, keyed
/// by DOM node. Equivalent to [`layout_fragmented`] with the map
/// threaded through; an empty map behaves exactly like
/// [`layout_fragmented`].
pub fn layout_fragmented_with_intrinsics(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn super::FontMetrics,
    page_height: f32,
    intrinsics: &HashMap<NodeId, IntrinsicDims>,
) -> Vec<PageFragment> {
    fragment_impl(
        doc,
        author,
        viewport,
        fonts,
        page_height,
        &HashMap::new(),
        intrinsics,
        (0.0, 0.0),
    )
}

/// Lay out `doc` for a viewport scrolled to `scroll` (`(scroll_x,
/// scroll_y)` in CSS px) and fragment it into pages of `page_height`
/// CSS px.
///
/// Mirrors [`super::layout_with_scroll`]: this is the entry point that
/// makes `position: sticky` stick in paged output.
///
/// Sticky/scroll interaction is deliberately order-fixed: sticky
/// constraints apply against the scroll offset *before* pagination, and
/// pagination then slices the shifted tree. The pages are bands of
/// document coordinates, so a sticky box stuck near the scrollport top
/// appears on the page covering its stuck position — it may land on a
/// later page than its in-flow spot. The stuck box keeps its in-flow
/// footprint for pagination: it never drags following content down (its
/// visual shift is not a flow shift), so following boxes paginate
/// undisturbed and the stuck box may overlap them in the page slices,
/// exactly as a stuck box overlaps content in the viewport. Fixed boxes
/// still repeat on every page, and the page bands themselves do not move
/// with the scroll offset.
pub fn layout_fragmented_with_scroll(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn super::FontMetrics,
    page_height: f32,
    scroll: (f32, f32),
) -> Vec<PageFragment> {
    fragment_impl(
        doc,
        author,
        viewport,
        fonts,
        page_height,
        &HashMap::new(),
        &HashMap::new(),
        scroll,
    )
}

/// Lay out `doc` and fragment it into pages of `page_height` CSS px,
/// honoring per-node `breaks`.
///
/// `breaks` maps a DOM node to its break hints and wins over the computed
/// style (which resolves `break-*`/`widows`/`orphans` through the normal
/// cascade, since `bowser_css` registers the properties; see the module
/// docs). Boxes without an entry (and anonymous boxes, which have no
/// node) fragment with their cascade-derived hints.
pub fn layout_fragmented_with_breaks(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn super::FontMetrics,
    page_height: f32,
    breaks: &HashMap<NodeId, BreakHints>,
) -> Vec<PageFragment> {
    fragment_impl(
        doc,
        author,
        viewport,
        fonts,
        page_height,
        breaks,
        &HashMap::new(),
        (0.0, 0.0),
    )
}

/// Private plumbing behind the four public `layout_fragmented*`
/// entry points: one layout call (intrinsics + scroll threaded through),
/// then the pure pagination/splitting post-pass.
#[allow(clippy::too_many_arguments)]
fn fragment_impl(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn super::FontMetrics,
    page_height: f32,
    breaks: &HashMap<NodeId, BreakHints>,
    intrinsics: &HashMap<NodeId, IntrinsicDims>,
    scroll: (f32, f32),
) -> Vec<PageFragment> {
    let root = super::layout_impl(
        doc,
        author,
        viewport,
        fonts,
        intrinsics,
        scroll,
        &HashMap::new(),
    );
    fragment_tree(doc, &root, viewport.w, page_height, breaks)
}

/// The box whose in-flow block-level children form the fragmentation
/// context: normally `<body>`, the single block child of the document
/// element's box. Falls back to the root itself when the tree has any
/// other shape (missing body, extra root children).
fn fragmentation_context(root: &LayoutBox) -> &LayoutBox {
    if root.children.len() == 1 && is_block_level(&root.children[0]) {
        &root.children[0]
    } else {
        root
    }
}

fn is_block_level(b: &LayoutBox) -> bool {
    matches!(
        b.box_type,
        BoxType::Block | BoxType::AnonymousBlock | BoxType::Table | BoxType::Grid
    )
}

fn parse_break_value(v: &str) -> Break {
    match v {
        "avoid" | "avoid-page" | "avoid-column" | "avoid-region" => Break::Avoid,
        "page" | "left" | "right" | "recto" | "verso" => Break::Page,
        // "column" has no meaning in a page context; unknown -> Auto.
        _ => Break::Auto,
    }
}

fn parse_break_inside(v: &str) -> BreakInside {
    match v {
        "avoid" | "avoid-page" | "avoid-column" | "avoid-region" => BreakInside::Avoid,
        _ => BreakInside::Auto,
    }
}

fn css_u32(v: &bowser_css::CssValue) -> Option<u32> {
    match v {
        bowser_css::CssValue::Number(n) if *n >= 1.0 => Some(*n as u32),
        _ => None,
    }
}

/// Resolve one box's break hints: explicit override map first, then the
/// computed style, then defaults. `orphans`/`widows` are registered in
/// `bowser_css`, so author stylesheets set them through this path.
fn resolve_hints(box_: &LayoutBox, breaks: &HashMap<NodeId, BreakHints>) -> BreakHints {
    if let Some(id) = box_.node
        && let Some(h) = breaks.get(&id)
    {
        return *h;
    }
    let mut hints = BreakHints::default();
    if let Some(style) = &box_.style {
        if let Some(v) = style.get_str_by_index(props::BREAK_BEFORE) {
            hints.before = parse_break_value(v);
        }
        if let Some(v) = style.get_str_by_index(props::BREAK_AFTER) {
            hints.after = parse_break_value(v);
        }
        if let Some(v) = style.get_str_by_index(props::BREAK_INSIDE) {
            hints.inside = parse_break_inside(v);
        }
        // Registered in bowser_css: author stylesheets can set these.
        if let Some(v) = style.get_by_index(props::ORPHANS)
            && let Some(n) = css_u32(v)
        {
            hints.orphans = n;
        }
        if let Some(v) = style.get_by_index(props::WIDOWS)
            && let Some(n) = css_u32(v)
        {
            hints.widows = n;
        }
    }
    hints
}

/// Smallest page top at or after `y` (within [`EPS`]).
fn page_top_at_or_after(y: f32, page_h: f32) -> f32 {
    let k = (y / page_h).floor();
    let t = k * page_h;
    if y - t <= EPS { t } else { (k + 1.0) * page_h }
}

/// Line boxes (`(y0, y1)` extents) in `unit`'s subtree, ascending.
///
/// Rebuilt from text fragments (one fragment per laid-out line per text
/// box): fragments whose y-extents overlap belong to the same line box, so
/// e.g. `hello <b>world</b>` on one line counts once. Coordinates are the
/// unit's original (pre-pagination) ones. Iterative: hostile DOM depth
/// must not overflow the stack.
fn collect_line_boxes(
    unit: &LayoutBox,
    breaks: &HashMap<NodeId, BreakHints>,
    page_h: f32,
) -> Vec<(f32, f32)> {
    let mut frags: Vec<(f32, f32, f32, f32)> = Vec::new();
    // `oy`: accumulated float-local y origin. Float subtrees are laid out
    // with float-local y (only the float box itself carries an absolute
    // rect), so descendant fragments must be offset by each enclosing
    // float's rect.y to read in absolute coordinates.
    let mut stack = vec![(unit, true, 0.0f32, false)];
    while let Some((n, is_root, oy, item)) = stack.pop() {
        // Lines inside a box that moves whole to the next page instead
        // of splitting (`break-inside: avoid`, unbreakable table rows,
        // flex/grid items, floats, and abs-pos boxes — mirroring
        // `split_subtree`'s gate) are not part of the unit's
        // fragmentable line flow. Counting them lets a nested avoid box
        // shove the whole unit down a page through orphans, fighting the
        // move-whole behavior.
        if !is_root {
            let h = n.rect.h;
            let unbreakable = h > 0.0
                && h <= page_h
                && (n.box_type == BoxType::TableRow
                    || item
                    || resolve_hints(n, breaks).inside == BreakInside::Avoid
                    || is_float_box(n)
                    || position_of(n) == "absolute");
            if unbreakable {
                continue;
            }
        }
        if n.box_type == BoxType::Text {
            for f in &n.fragments {
                if f.rect.h > 0.0 {
                    frags.push((
                        f.rect.y + oy,
                        f.rect.y + oy + f.rect.h,
                        f.rect.x,
                        f.rect.x + f.rect.w,
                    ));
                }
            }
        }
        let coy = if is_float_box(n) { oy + n.rect.y } else { oy };
        // Children of a flex/grid container are flex/grid items: their
        // lines belong to the move-whole box, not the unit's line flow.
        let child_item = is_flex_or_grid_container(n);
        stack.extend(n.children.iter().map(|c| (c, false, coy, child_item)));
    }
    frags.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
    });
    // Merge vertically- AND horizontally-overlapping fragments into line
    // boxes. The x-overlap requirement keeps multicol columns apart: two
    // fragments at the same y in different columns are two line boxes,
    // not one. Merging them by y alone undercounts the unit's lines, so
    // orphans/widows shoves a multicol container that already satisfies
    // the requirement (e.g. one y-band = two lines, one per column,
    // counted as one < orphans: 2, pushing the whole container to the
    // next page and leaving a gap). Spans of a single line abut or
    // overlap in x, so single-column counting is unchanged. (With
    // `column-gap: 0` the columns abut and still merge: geometrically
    // indistinguishable from two spans of one line.)
    let mut merged: Vec<(f32, f32, f32, f32)> = Vec::new();
    for (y0, y1, x0, x1) in frags {
        match merged.last_mut() {
            Some(last) if y0 < last.1 - EPS && x0 < last.3 + EPS && last.2 < x1 + EPS => {
                last.0 = last.0.min(y0);
                last.1 = last.1.max(y1);
                last.2 = last.2.min(x0);
                last.3 = last.3.max(x1);
            }
            _ => merged.push((y0, y1, x0, x1)),
        }
    }
    let mut lines: Vec<(f32, f32)> = Vec::with_capacity(merged.len());
    for (y0, y1, _, _) in merged {
        lines.push((y0, y1));
    }
    lines
}

/// Mid-point of a line box: assigns the line to the page holding most of
/// it when counting lines per fragment.
fn line_mid(l: &(f32, f32)) -> f32 {
    (l.0 + l.1) * 0.5
}

/// Extra downward shift for `unit` so its page breaks respect
/// `orphans`/`widows` (CSS Fragmentation 3 §8).
///
/// `shift` is the unit's current paginated shift; the return is `>= 0`.
/// When the block would split with fewer than `orphans` lines left at the
/// bottom of the first fragment, the whole block moves to the next page
/// top. Otherwise, when the last fragment would start with fewer than
/// `widows` lines, the last break moves earlier (the block shifts down) so
/// the last `widows` lines share the last page. A block with fewer lines
/// than `orphans` + `widows` cannot satisfy both at once: `orphans` wins
/// and `widows` is best-effort. Bounded: at most
/// [`WIDOW_ORPHAN_PASSES`] rounds, each moving the unit down by about a
/// page at most.
fn widows_orphans_extra_shift(
    unit: &LayoutBox,
    shift: f32,
    page_h: f32,
    hints: &BreakHints,
    breaks: &HashMap<NodeId, BreakHints>,
) -> f32 {
    let orphans = hints.orphans.max(1) as usize;
    let mut widows = hints.widows.max(1) as usize;
    let h = unit.rect.h.max(0.0);
    if h <= 0.0 {
        return 0.0;
    }
    // Cheap straddle check before walking the subtree for lines: most
    // units never break, and this keeps the post-pass O(1) for them.
    let top0 = unit.rect.y + shift;
    if top0 + h <= (top0 / page_h).floor() * page_h + page_h + EPS {
        return 0.0;
    }
    let lines = collect_line_boxes(unit, breaks, page_h);
    let n = lines.len();
    if n == 0 {
        return 0.0;
    }
    if n < orphans + widows {
        // Jointly unsatisfiable (see docs): orphans wins, widows is
        // best-effort (a non-empty last fragment always has >= 1 line).
        widows = 1;
    }
    let bottom0 = unit.rect.y + h;
    let mut extra = 0.0f32;
    for _ in 0..WIDOW_ORPHAN_PASSES {
        let s = shift + extra;
        let top = unit.rect.y + s;
        let bottom = top + h;
        let first_page_end = (top / page_h).floor() * page_h + page_h;
        if bottom <= first_page_end + EPS {
            break; // no page break inside the unit
        }
        // Orphans: the first fragment must keep at least `orphans` lines.
        let first = lines
            .iter()
            .filter(|l| line_mid(l) + s < first_page_end - EPS)
            .count();
        if first < orphans {
            let push = first_page_end - top;
            if push <= EPS {
                break; // already at a page top: no progress possible
            }
            extra += push;
            if n <= orphans {
                // The whole block can never satisfy orphans by splitting;
                // one push keeps it as whole as possible.
                break;
            }
            continue;
        }
        // Widows: the last fragment must hold at least `widows` lines.
        let last_page_top = ((bottom - EPS) / page_h).floor() * page_h;
        let last = lines
            .iter()
            .filter(|l| line_mid(l) + s >= last_page_top - EPS)
            .count();
        if last >= widows {
            break;
        }
        // Move the last break earlier: align the widows-th line from the
        // end with the top of a page, so the last `widows` lines start the
        // last fragment cleanly (no line straddles the break).
        // Unsatisfiable when those lines span more than a page.
        let idx = n - widows;
        let line_top = lines[idx].0;
        if bottom0 - line_top >= page_h - EPS {
            break;
        }
        let page_top = ((line_top + s) / page_h).ceil() * page_h;
        let push = page_top - (line_top + s);
        if push <= EPS {
            break; // no progress possible
        }
        extra += push;
        // Loop re-checks orphans on the now-shorter first fragment.
    }
    extra
}

/// Phase 1: paginate the flow units, returning one y-shift per unit.
///
/// The walk is single-pass and O(units): every shift only moves content
/// down, avoidance moves at most one page, and nothing here iterates over
/// pixels or pages, so hostile heights cannot hang it. Widows/orphans
/// adjust each unit's shift (down only) before the break-after rules run,
/// so keep-with-next sees the final position.
fn paginate(units: &[&LayoutBox], page_h: f32, breaks: &HashMap<NodeId, BreakHints>) -> Vec<f32> {
    let mut shifts = vec![0.0f32; units.len()];
    let mut shift = 0.0f32;
    // Max bottom edge placed so far, in paginated space: keeps units
    // monotonic even if input rects overlap (floats, negative margins).
    let mut cursor = f32::NEG_INFINITY;
    for (i, u) in units.iter().enumerate() {
        let hints = resolve_hints(u, breaks);
        // A stuck sticky box's visual position is not its flow position:
        // it keeps its in-flow footprint, so it must neither snap to the
        // cursor nor push it — either would drag following content down
        // and falsify the document (and break the invariant that a huge
        // page reproduces `layout_with_scroll`). Break rules still apply
        // to the sticky box itself; its shifted rect alone decides which
        // pages it lands on, so it may overlap following content in the
        // page slices exactly as a stuck box overlaps in the viewport.
        // (At zero scroll a sticky box behaves like `relative`; skipping
        // the cursor there only drops the pre-existing habit of letting
        // relative-like shifts displace following content.)
        let sticky = u
            .style
            .as_ref()
            .and_then(|s| s.get_str_by_index(props::POSITION))
            == Some("sticky");
        let mut top = u.rect.y + shift;
        if !sticky && top < cursor {
            shift += cursor - top;
            top = cursor;
        }
        // break-before: page.
        if hints.before == Break::Page {
            let nt = page_top_at_or_after(top, page_h);
            shift += nt - top;
            top = nt;
        }
        let h = u.rect.h.max(0.0);
        let mut bottom = top + h;
        // break-inside: avoid: a box that fits on one page moves whole to
        // the next page top instead of straddling; a taller-than-page box
        // fragments (bounded: no unbounded shifting).
        let page_end = (top / page_h).floor() * page_h + page_h;
        if bottom > page_end + EPS && hints.inside == BreakInside::Avoid && h <= page_h {
            shift += page_end - top;
            top = page_end;
            bottom = top + h;
        }
        // Widows/orphans: move the split point (down only) so a text block
        // keeps enough lines on each side of the break. Runs before the
        // break-after rules so keep-with-next sees the final position.
        let extra = widows_orphans_extra_shift(u, shift, page_h, &hints, breaks);
        if extra > 0.0 {
            shift += extra;
            top += extra;
            bottom += extra;
        }
        // break-after.
        match hints.after {
            Break::Page => {
                let nt = page_top_at_or_after(bottom, page_h);
                shift += nt - bottom;
            }
            Break::Avoid => {
                // Keep-with-next, one-box lookahead: if this box sits
                // wholly on page k, the next box would start on a later
                // page, and both fit on one page together, move this box
                // down to the next page top. A forced break-before on the
                // next box wins over the avoid.
                if let Some(v) = units.get(i + 1) {
                    let vh = v.rect.h.max(0.0);
                    if resolve_hints(v, breaks).before != Break::Page {
                        let k = (top / page_h).floor();
                        let this_page_end = (k + 1.0) * page_h;
                        let v_top = v.rect.y + shift;
                        if bottom <= this_page_end + EPS
                            && v_top + EPS >= this_page_end
                            && h + vh <= page_h + EPS
                        {
                            shift += this_page_end - top;
                            top = this_page_end;
                        }
                    }
                }
            }
            Break::Auto => {}
        }
        shifts[i] = shift;
        if !sticky {
            cursor = cursor.max(top + h);
        }
    }
    shifts
}

/// One box's record from the nested-break pre-pass ([`collect_nested_breaks`]).
#[derive(Clone, Copy, Debug)]
struct NestedBreak {
    /// Extra y-shift for this box, added to its flow unit's phase-1 shift.
    /// Always 0 for split roots and for boxes inside fixed/abs-pos
    /// subtrees (out-of-flow boxes take no part in flow breaks).
    shift: f32,
    /// Minimum `(rect.y + shift)` over the box's subtree, in
    /// pre-pagination coordinates (the phase-1 base shift is added when the
    /// map is consumed, so one entry serves every split root).
    sub_min_top: f32,
    /// Maximum `(rect.y + rect.h + shift)` over the box's subtree.
    sub_max_bottom: f32,
}

fn position_of(b: &LayoutBox) -> &str {
    b.style
        .as_ref()
        .and_then(|s| s.get_str_by_index(props::POSITION))
        .unwrap_or("static")
}

fn is_out_of_flow(b: &LayoutBox) -> bool {
    matches!(position_of(b), "fixed" | "absolute")
}

/// A floated box (`float: left | right` on an in-flow box): floats never
/// split across pages, so they ride the same never-split fast-path as
/// `break-inside: avoid` in [`split_subtree`]. Out-of-flow boxes are
/// excluded: `position: absolute | fixed` wins over `float` (CSS 2.1 §9.7)
/// and those take their own paths.
fn is_float_box(b: &LayoutBox) -> bool {
    !is_out_of_flow(b)
        && matches!(
            b.style.as_ref().and_then(|s| s.get_str_by_index(props::FLOAT)),
            Some(v) if v.eq_ignore_ascii_case("left") || v.eq_ignore_ascii_case("right")
        )
}

/// A flex (`flex`/`inline-flex`) or grid container. Layout buckets `flex`
/// into [`BoxType::Block`] (and `inline-flex`/`inline-grid` into
/// `InlineBlock`), with the inner layout dispatched off the raw `display`
/// keyword — so the raw value is consulted here, mirroring paint's
/// `is_flex_or_grid_container` and layout's `is_flex_or_grid_item` parent
/// check. Marks flex/grid items (in-flow children of such a container):
/// like table rows, they never split mid-box when they fit on a page, so
/// [`split_subtree`] and [`collect_line_boxes`] set the item flag from
/// the parent when pushing children.
fn is_flex_or_grid_container(box_: &LayoutBox) -> bool {
    if box_.box_type == BoxType::Grid {
        return true;
    }
    matches!(
        box_.style.as_ref().and_then(|s| s.get_str_by_index(props::DISPLAY)),
        Some(d)
            if d.eq_ignore_ascii_case("flex")
                || d.eq_ignore_ascii_case("inline-flex")
                || d.eq_ignore_ascii_case("grid")
                || d.eq_ignore_ascii_case("inline-grid")
    )
}

/// A `<thead>`/`<tfoot>` row group: identified by DOM tag, since both lay
/// out as ordinary [`BoxType::TableRowGroup`]s. Used by [`repeat_thead`]
/// and [`repeat_tfoot`].
fn is_named_row_group(doc: &Document, b: &LayoutBox, tag: &str) -> bool {
    if b.box_type != BoxType::TableRowGroup {
        return false;
    }
    match b.node.map(|id| doc.kind(id)) {
        Some(bowser_dom::NodeKind::Element(d)) => d.tag_name == tag,
        _ => false,
    }
}

/// A `<thead>` row group. See [`is_named_row_group`].
fn is_thead_group(doc: &Document, b: &LayoutBox) -> bool {
    is_named_row_group(doc, b, "thead")
}

/// A `<tfoot>` row group. See [`is_named_row_group`].
fn is_tfoot_group(doc: &Document, b: &LayoutBox) -> bool {
    is_named_row_group(doc, b, "tfoot")
}

/// Deep shift of an already-built page-slice subtree by `(dx, dy)`:
/// rects, text fragment rects, and baselines. Iterative so hostile depth
/// cannot overflow the stack.
fn shift_tree(b: &mut LayoutBox, dx: f32, dy: f32) {
    let mut stack = vec![b];
    while let Some(n) = stack.pop() {
        // Out-of-flow subtrees (fixed / absolute) live in their own
        // coordinate frames; shifting page flow must not move them.
        if is_out_of_flow(n) {
            continue;
        }
        n.rect.x += dx;
        n.rect.y += dy;
        for f in &mut n.fragments {
            f.rect.x += dx;
            f.rect.y += dy;
            f.baseline += dy;
        }
        stack.extend(n.children.iter_mut());
    }
}

/// Absolutize every float subtree inside a whole-emitted page-slice clone.
///
/// Float layout positions only the float box itself: a float's children
/// carry absolute x but float-local y, which paint cannot use (it reads
/// `frag.rect` directly, with no parent offset to fall back on). `b` is a
/// rigid [`clone_translated`] by `dy`, so each float boundary's
/// pre-translation y origin is `rect.y - dy`; adding that — y only, never
/// x — to the boundary's subtree converts float-local y to page-relative
/// y. Nested floats re-base at their own origin, per boundary (mirroring
/// [`collect_line_boxes`]): the outer origin is applied exactly once, so
/// a nested float's descendants never get it twice.
fn absolutize_floats(b: &mut LayoutBox, dy: f32) {
    let mut stack = vec![b];
    while let Some(n) = stack.pop() {
        if is_float_box(n) {
            // The float's own box is already in final position; only its
            // subtree is float-local.
            absolutize_under(n, n.rect.y - dy, dy);
        } else {
            stack.extend(n.children.iter_mut());
        }
    }
}

/// Shift one float boundary's subtree from float-local y to page-relative
/// y. `frame` is the boundary's pre-translation y origin (the addend for
/// its local frame); `dy` is the clone's rigid translation, used to
/// recover each nested float's own local origin for re-basing. Iterative:
/// hostile DOM depth must not overflow the stack.
fn absolutize_under(n: &mut LayoutBox, frame: f32, dy: f32) {
    let mut stack: Vec<(&mut LayoutBox, f32)> = vec![(n, frame)];
    while let Some((p, fr)) = stack.pop() {
        for c in p.children.iter_mut() {
            // x is already absolute (float-local is y-only): shift y and
            // baselines into the page frame, never x.
            c.rect.y += fr;
            for f in &mut c.fragments {
                f.rect.y += fr;
                f.baseline += fr;
            }
        }
        // Re-base at nested float boundaries: a nested float's own box is
        // now page-relative, so its pre-translation local origin is
        // `rect.y - dy`, and its subtree takes that as its frame addend.
        // Non-float children stay in `p`'s frame.
        stack.extend(p.children.iter_mut().map(|c| {
            let cfr = if is_float_box(c) { c.rect.y - dy } else { fr };
            (c, cfr)
        }));
    }
}

/// Phase 1b: iterative pre-pass over the whole tree collecting nested
/// forced-break hints.
///
/// Phase 1 ([`paginate`]) only sees the fragmentation context's direct
/// in-flow children; `break-before: page`, `break-after: page`, and
/// `break-after: avoid` on deeper boxes would otherwise be silently
/// ignored. This walk visits every split subtree — each flow unit with its
/// phase-1 shift as the base, each abs-pos child and other root child with
/// base 0 — in document order and records one [`NestedBreak`] per box.
///
/// Keyed by box address, not `NodeId`: anonymous boxes have no node, yet
/// they must both ride the running shift (otherwise content after a nested
/// break could render above the break) and contribute subtree extents.
/// Split roots keep shift 0: their own hints were already handled (flow
/// units by phase 1) or are out of scope (abs-pos children).
///
/// The running shift only ever moves down, mirroring phase 1's cursor
/// invariant: content never moves up, so page assignment stays consistent.
/// `break-after` pushes apply when leaving a box, so they move following
/// siblings and never the box's own children; `break-before: page` and the
/// keep-with-next (`break-after: avoid`) lookahead apply when entering.
/// Boxes inside fixed/abs-pos subtrees keep shift 0 but still contribute
/// extents, so a nested fixed box keeps its ancestors' band tests honest
/// on every page. `break-inside` is untouched here: [`split_subtree`]'s
/// avoid fast-path already handles it (a nested avoid box that overflows
/// the page bottom is deliberate and unchanged).
fn collect_nested_breaks(
    subtrees: &[(&LayoutBox, f32)],
    page_h: f32,
    breaks: &HashMap<NodeId, BreakHints>,
) -> HashMap<*const LayoutBox, NestedBreak> {
    struct Frame<'a> {
        node: &'a LayoutBox,
        idx: usize,
        shift: f32,
        sub_min: f32,
        sub_max: f32,
        frozen: bool,
        is_root: bool,
        /// Accumulated float-origin y offset for this node's own rect:
        /// float interiors use float-local y, so a descendant's absolute
        /// y is `rect.y + fy`. Built per float boundary, mirroring
        /// [`collect_line_boxes`].
        fy: f32,
    }
    let mut map: HashMap<*const LayoutBox, NestedBreak> = HashMap::new();
    for (root, base) in subtrees {
        let base = *base;
        let mut running = 0.0f32;
        let mut stack = vec![Frame {
            node: root,
            idx: 0,
            shift: 0.0,
            sub_min: root.rect.y,
            sub_max: root.rect.y + root.rect.h,
            frozen: is_out_of_flow(root),
            is_root: true,
            fy: 0.0,
        }];
        while let Some(f) = stack.last_mut() {
            if f.idx < f.node.children.len() {
                let i = f.idx;
                f.idx += 1;
                let child: &LayoutBox = &f.node.children[i];
                let next_sib = f.node.children.get(i + 1);
                let frozen = f.frozen || is_out_of_flow(child);
                // In-flow boxes without a node (anonymous/text) carry no
                // hints but still ride the running shift, keeping shifts
                // monotonic in document order.
                let mut cshift = if frozen { 0.0 } else { running };
                if !frozen && child.node.is_some() {
                    let hints = resolve_hints(child, breaks);
                    let mut top = child.rect.y + base + running;
                    // break-before: page.
                    if hints.before == Break::Page {
                        let nt = page_top_at_or_after(top, page_h);
                        running += nt - top;
                        top = nt;
                    }
                    // break-after: avoid (keep-with-next, one-box
                    // lookahead, mirroring phase 1): if this box sits
                    // wholly on page k, the next sibling would start on a
                    // later page, and both fit on one page together, move
                    // this box down to the next page top. A forced
                    // break-before on the next sibling wins over the avoid.
                    if hints.after == Break::Avoid
                        && let Some(m) = next_sib
                        && !is_out_of_flow(m)
                        && resolve_hints(m, breaks).before != Break::Page
                    {
                        let h = child.rect.h.max(0.0);
                        let vh = m.rect.h.max(0.0);
                        let page_end = (top / page_h).floor() * page_h + page_h;
                        let m_top = m.rect.y + base + running;
                        if top + h <= page_end + EPS
                            && m_top + EPS >= page_end
                            && h + vh <= page_h + EPS
                        {
                            running += page_end - top;
                        }
                    }
                    cshift = running;
                }
                // Float interiors use float-local y: offset each child's
                // extent contribution by the enclosing float's origin
                // (accumulated per boundary), so a float at y=700 with
                // children at float-local y~10 doesn't drag its
                // ancestors' sub_min_top toward ~0 and defeat band
                // pruning.
                let cfy = f.fy
                    + if is_float_box(f.node) {
                        f.node.rect.y
                    } else {
                        0.0
                    };
                stack.push(Frame {
                    node: child,
                    idx: 0,
                    shift: cshift,
                    sub_min: child.rect.y + cfy + cshift,
                    sub_max: child.rect.y + child.rect.h + cfy + cshift,
                    frozen,
                    is_root: false,
                    fy: cfy,
                });
            } else {
                let f = stack.pop().expect("pre-pass stack never empties early");
                // break-after: page pushes following siblings, never the
                // box's own children (they were visited above, before this
                // push).
                if !f.is_root && !f.frozen && f.node.node.is_some() {
                    let hints = resolve_hints(f.node, breaks);
                    if hints.after == Break::Page {
                        let bottom = f.node.rect.y + f.node.rect.h + base + running;
                        running += page_top_at_or_after(bottom, page_h) - bottom;
                    }
                }
                map.insert(
                    f.node as *const LayoutBox,
                    NestedBreak {
                        shift: f.shift,
                        sub_min_top: f.sub_min,
                        sub_max_bottom: f.sub_max,
                    },
                );
                if let Some(p) = stack.last_mut() {
                    p.sub_min = p.sub_min.min(f.sub_min);
                    p.sub_max = p.sub_max.max(f.sub_max);
                }
            }
        }
    }
    map
}

/// Intersect `r` with the page band and translate to page coordinates.
fn page_rect(r: Rect, page_top: f32, page_h: f32) -> Rect {
    let y0 = r.y.max(page_top);
    let y1 = (r.y + r.h).min(page_top + page_h);
    Rect {
        x: r.x,
        y: y0 - page_top,
        w: r.w,
        h: (y1 - y0).max(0.0),
    }
}

/// Shallow-clone one box with its (shifted) rect intersected with the page
/// band and translated to page coordinates, and no children (the caller
/// fills them in).
fn page_box_head(node: &LayoutBox, shift: f32, page_top: f32, page_h: f32) -> LayoutBox {
    let mut b = node.clone();
    let mut r = node.rect;
    r.y += shift;
    b.rect = page_rect(r, page_top, page_h);
    b.children = Vec::new();
    b.fragments = Vec::new();
    b
}

/// Iterative deep clone with rects/fragments translated by `dy` (used for
/// `break-inside: avoid` wholes and `position: fixed` repeats).
/// Iterative: hostile DOM depth must not overflow the stack.
fn clone_translated(node: &LayoutBox, dy: f32) -> LayoutBox {
    let mut stack: Vec<(&LayoutBox, bool)> = vec![(node, false)];
    let mut built: Vec<LayoutBox> = Vec::new();
    while let Some((n, entered)) = stack.pop() {
        if !entered {
            stack.push((n, true));
            for c in n.children.iter().rev() {
                stack.push((c, false));
            }
        } else {
            let k = n.children.len();
            let start = built.len() - k;
            let children: Vec<LayoutBox> = built.drain(start..).collect();
            let mut b = n.clone();
            b.rect.y += dy;
            for f in &mut b.fragments {
                f.rect.y += dy;
                f.baseline += dy;
            }
            b.children = children;
            built.push(b);
        }
    }
    built
        .pop()
        .expect("clone_translated: node always builds one box")
}

/// Remove out-of-flow (`fixed` / `absolute`) subtrees from a cloned
/// header/footer group: they are emitted through their own paths — the
/// group's own slice whenever the group is present, which it always is on
/// any page they occupy, since their extents feed the group's band test —
/// so keeping them in the repeated group would duplicate them, and
/// translating the clone would drag fixed boxes off their viewport
/// coordinates.
fn strip_out_of_flow(b: &mut LayoutBox) {
    b.children.retain(|c| !is_out_of_flow(c));
    for c in b.children.iter_mut() {
        strip_out_of_flow(c);
    }
}

/// Repeat a multi-page table's `<thead>` row groups at the top of a
/// continuation page's table slice (print behavior).
///
/// `slice` is the page's already-assembled table fragment (children in
/// page-relative coordinates), `table` the original table box, and
/// `table_own` its shift (`w.own` in the caller). No-op unless the
/// table's span covers multiple pages and this page comes after the
/// first: the first page already shows the header naturally.
///
/// Each header group is cloned whole and translated so its top lands at
/// the page top; the slice's existing children shift down by the total
/// header height so the repeated header does not overlap the continued
/// rows. Bounded: a header group already present in the slice with
/// visible rows (a thead straddling onto this page) is not duplicated,
/// headers taller than the page are skipped, and repetition is skipped
/// for a page whose continued rows would spill past the page bottom when
/// shifted down — the table slice rect grows by exactly the header
/// height, never past the page bottom (no re-layout).
///
/// Returns the total repeated header height inserted at the top of the
/// slice (0 when nothing was repeated); [`repeat_tfoot`] uses it to keep
/// a repeated footer from overlapping the repeated header.
fn repeat_thead(
    doc: &Document,
    slice: &mut LayoutBox,
    table: &LayoutBox,
    table_own: f32,
    page_index: usize,
    page_h: f32,
) -> f32 {
    let h = table.rect.h.max(0.0);
    if h <= 0.0 {
        return 0.0;
    }
    let (p0, p1) = span_of(table.rect.y + table_own, h, page_h);
    if p0 == p1 || page_index <= p0 {
        return 0.0;
    }

    let mut header_h = 0.0f32;
    let mut heads: Vec<LayoutBox> = Vec::new();
    for g in table.children.iter().filter(|c| is_thead_group(doc, c)) {
        if g.rect.h <= 0.0 {
            continue;
        }
        // Already in this slice with visible rows (a thead straddling
        // onto this page): repeating it would duplicate rows. A group
        // that is present but whose rows all moved whole to another page
        // is present yet empty — that must not suppress repetition.
        let already_visible = g.node.is_some()
            && slice
                .children
                .iter()
                .any(|c| c.node == g.node && c.children.iter().any(|r| r.rect.h > EPS));
        if already_visible {
            continue;
        }
        // The slice's children are page-relative: translate the clone so
        // the group's top lands at the page top. Out-of-flow descendants
        // are stripped (emitted via the group's own slice); floats are
        // absolutized to the clone's frame.
        let mut head = clone_translated(g, -g.rect.y);
        strip_out_of_flow(&mut head);
        absolutize_floats(&mut head, -g.rect.y);
        heads.push(head);
        header_h += g.rect.h;
    }
    if heads.is_empty() || header_h > page_h {
        return 0.0;
    }
    // Shifting the continued rows down must not push content past the
    // page bottom: if it would, skip repetition for this page rather
    // than overflowing. Container rects are band-clipped during slicing,
    // so the true content extent is only visible at the leaves: walk the
    // in-flow descendants. Out-of-flow subtrees (fixed / absolute) live in
    // their own coordinate frames and cannot spill page flow; skip them.
    let content_bottom = {
        let mut bottom = 0.0f32;
        let mut stack: Vec<&LayoutBox> = slice.children.iter().collect();
        while let Some(n) = stack.pop() {
            if is_out_of_flow(n) {
                continue;
            }
            bottom = bottom.max(n.rect.y + n.rect.h);
            stack.extend(n.children.iter());
        }
        bottom
    };
    if content_bottom + header_h > page_h + EPS {
        return 0.0;
    }
    for c in slice.children.iter_mut() {
        shift_tree(c, 0.0, header_h);
    }
    let mut children = std::mem::take(&mut slice.children);
    let mut with_heads = Vec::with_capacity(heads.len() + children.len());
    with_heads.extend(heads);
    with_heads.append(&mut children);
    slice.children = with_heads;
    slice.rect.h += header_h;
    header_h
}

/// Repeat a multi-page table's `<tfoot>` row groups at the bottom of a
/// continuation page's table slice (print behavior: CSS 2.1 §17.5.2 —
/// the table footer repeats at the bottom of each continued page
/// fragment, mirroring the header at the top).
///
/// `slice` is the page's already-assembled table fragment (children in
/// page-relative coordinates), `table` the original table box,
/// `table_own` its shift (`w.own` in the caller), and `header_h` the
/// height [`repeat_thead`] already inserted at the top of this slice (0
/// when it repeated nothing). No-op unless the table's span covers
/// multiple pages and this page comes after the first: the first page
/// already shows the footer naturally at the table's end.
///
/// Each footer group is cloned whole and translated so the stacked
/// groups' bottom lands at the page bottom; the clones are appended after
/// the slice's existing children (document order: footer groups lay out
/// after the body rows). Unlike [`repeat_thead`], the continued rows are
/// *not* shifted: shifting them up would push the top rows off the page
/// and lose content. Instead repetition is skipped for a page whose rows
/// already reach into the footer's band.
///
/// Bounded: a footer group already present in the slice with visible rows
/// (the natural footer on the table's last page) is not duplicated;
/// footers taller than the page are skipped; a repeated header and a
/// repeated footer may not overlap each other (`header_h + foot_h` must
/// fit the page); and a footer that would overlap the continued rows is
/// skipped for that page rather than overflowing. Page count itself is
/// capped by [`MAX_PAGES`]; repetition adds no pages and iterates no
/// page loop, so a hostile tfoot can only ever suppress one clone per
/// page — never spin.
///
/// The fit check measures the continued *rows'* extent
/// ([`BoxType::TableRow`]), not the slice's container rects: row-group
/// heads are band-clipped during slicing, so on any page the table
/// continues past they span the full band and would veto repetition on
/// every true middle page. A row's rect covers its cells and their
/// contents, so the rows bound the visible row content.
fn repeat_tfoot(
    doc: &Document,
    slice: &mut LayoutBox,
    table: &LayoutBox,
    table_own: f32,
    page_index: usize,
    page_h: f32,
    header_h: f32,
) {
    let h = table.rect.h.max(0.0);
    if h <= 0.0 {
        return;
    }
    let (p0, p1) = span_of(table.rect.y + table_own, h, page_h);
    if p0 == p1 || page_index <= p0 {
        return;
    }

    let mut foot_h = 0.0f32;
    let mut feet: Vec<LayoutBox> = Vec::new();
    for g in table.children.iter().filter(|c| is_tfoot_group(doc, c)) {
        if g.rect.h <= 0.0 {
            continue;
        }
        // Already in this slice with visible rows (the natural footer on
        // the table's last page): repeating it would duplicate rows. A
        // group that is present but whose rows all moved whole to another
        // page is present yet empty — that must not suppress repetition.
        let already_visible = g.node.is_some()
            && slice
                .children
                .iter()
                .any(|c| c.node == g.node && c.children.iter().any(|r| r.rect.h > EPS));
        if already_visible {
            continue;
        }
        // The slice's children are page-relative: translate the clone so
        // the stacked groups' bottom lands at the page bottom. Out-of-flow
        // descendants are stripped (emitted via the group's own slice);
        // floats are absolutized to the clone's frame.
        let dy = page_h - foot_h - g.rect.h - g.rect.y;
        let mut foot = clone_translated(g, dy);
        strip_out_of_flow(&mut foot);
        absolutize_floats(&mut foot, dy);
        feet.push(foot);
        foot_h += g.rect.h;
    }
    if feet.is_empty() || foot_h > page_h || header_h + foot_h > page_h + EPS {
        return;
    }
    // The rows are not shifted (shifting them up would lose the top rows
    // off the page); the footer instead needs clear space at the page
    // bottom. Measure the continued rows' extent — row-group heads are
    // band-clipped and span the full band on pages the table continues
    // past, so they cannot serve as the content measure here. Out-of-flow
    // subtrees (fixed / absolute) live in their own coordinate frames and
    // cannot collide with page flow; skip them.
    let rows_bottom = {
        let mut bottom = 0.0f32;
        let mut stack: Vec<&LayoutBox> = slice.children.iter().collect();
        while let Some(n) = stack.pop() {
            if is_out_of_flow(n) {
                continue;
            }
            if n.box_type == BoxType::TableRow {
                bottom = bottom.max(n.rect.y + n.rect.h);
            }
            stack.extend(n.children.iter());
        }
        bottom
    };
    if rows_bottom + foot_h > page_h + EPS {
        return;
    }
    slice.children.extend(feet);
    // The footer sits at the page bottom; make sure the slice rect covers
    // it. Band-clipping already does on pages the table continues past —
    // this only matters if the natural footer was rowless.
    let need = page_h - slice.rect.y;
    if slice.rect.h < need {
        slice.rect.h = need;
    }
}

/// Phase 2 helper: split one subtree into the page band, honoring
/// `break-inside: avoid` (plus unbreakable table rows, flex/grid items,
/// floats, and abs-pos boxes) for nested boxes, repeating nested `position: fixed`
/// boxes whole on every page, and repeating `<thead>` row groups at the
/// top and `<tfoot>` row groups at the bottom of continuation pages for
/// multi-page tables. `nested`
/// is the phase-1b pre-pass map: each box's own shift plus its subtree's
/// shifted extents. `item` is whether the split root is itself a
/// flex/grid item (an in-flow child of a flex/grid container), set by the
/// caller from the parent — item-hood is a property of the parent, so the
/// root cannot say for itself. Returns `None` when nothing intersects the
/// band. Iterative post-order over the tree (stack-safe).
#[allow(clippy::too_many_arguments)]
fn split_subtree(
    doc: &Document,
    node: &LayoutBox,
    band_top: f32,
    band_bottom: f32,
    page_index: usize,
    page_h: f32,
    shift: f32,
    breaks: &HashMap<NodeId, BreakHints>,
    nested: &HashMap<*const LayoutBox, NestedBreak>,
    item: bool,
) -> Option<LayoutBox> {
    struct Work<'a> {
        node: &'a LayoutBox,
        entered: bool,
        /// The box's own shift: the unit's phase-1 base plus its nested
        /// pre-pass shift.
        own: f32,
        /// Whether this node is a flex/grid item: an in-flow child of a
        /// flex or grid container. Set from the parent when children are
        /// pushed (item-hood is a property of the parent — layout buckets
        /// `display: flex` into `BoxType::Block`, so the node alone cannot
        /// say). Flex/grid items ride the unbreakable fast-path, mirroring
        /// table rows.
        item: bool,
    }
    let page_top = page_index as f32 * page_h;
    let mut stack: Vec<Work> = vec![Work {
        node,
        entered: false,
        own: 0.0,
        item,
    }];
    // One entry per exited frame, in exit order; entering frames that miss
    // the band or take a fast-path push their result directly.
    let mut built: Vec<Option<LayoutBox>> = Vec::new();
    while let Some(w) = stack.pop() {
        if !w.entered {
            // `position: fixed` repeats whole on every page, mirroring the
            // context-level path: no band test, no shift, no slicing.
            if position_of(w.node) == "fixed" {
                let mut b = clone_translated(w.node, 0.0);
                // A float nested anywhere inside the fixed subtree keeps
                // float-local children; absolutize every float subtree
                // within the clone.
                absolutize_floats(&mut b, 0.0);
                built.push(Some(b));
                continue;
            }
            let entry = nested.get(&(w.node as *const LayoutBox));
            let own = shift + entry.map(|e| e.shift).unwrap_or(0.0);
            let y0 = w.node.rect.y + own;
            // Subtree-aware band test: nested forced breaks can push
            // descendants below the node's own rect, and nested fixed
            // boxes sit at viewport coordinates, so the node's own rect
            // alone cannot prune the walk.
            let sub_top = entry.map(|e| e.sub_min_top).unwrap_or(w.node.rect.y) + shift;
            let sub_bottom = entry
                .map(|e| e.sub_max_bottom)
                .unwrap_or(w.node.rect.y + w.node.rect.h)
                + shift;
            if sub_bottom <= band_top || sub_top >= band_bottom {
                built.push(None);
                continue;
            }
            // Unbreakable nested box: `break-inside: avoid`, a table row
            // (rows never split mid-row when they fit on a page), a
            // flex/grid item (flex items and grid-row members never split
            // mid-box when they fit on a page), a float (floats never
            // split across pages), or an abs-pos box (abs-pos boxes are
            // emitted whole on the page holding their top edge). Emitted
            // whole; when the box straddles a page boundary it moves to
            // the top of the next page instead of pinning to the page
            // holding its top edge and overflowing the page bottom,
            // mirroring phase 1's re-pagination of top-level avoid boxes.
            // Top-level `break-inside: avoid` flow units never reach this
            // straddling: phase 1 already re-paginated the ones that fit
            // on a page.
            let hints = resolve_hints(w.node, breaks);
            let unbreakable = (hints.inside == BreakInside::Avoid
                || w.node.box_type == BoxType::TableRow
                || w.item
                || is_float_box(w.node)
                || position_of(w.node) == "absolute")
                && w.node.rect.h > 0.0
                && w.node.rect.h <= page_h;
            if unbreakable {
                let y1 = y0 + w.node.rect.h;
                let start_page = (y0 / page_h).floor().max(0.0) as usize;
                let page_bottom = (start_page + 1) as f32 * page_h;
                // The gate above requires h <= page_h, so a straddling box
                // always fits whole on the next page: emit it there, with
                // its top snapped to the next page top.
                let (emit_page, dy) = if y1 > page_bottom + EPS {
                    (start_page + 1, page_bottom - page_top - w.node.rect.y)
                } else {
                    (start_page, own - page_top)
                };
                if emit_page == page_index {
                    let mut b = clone_translated(w.node, dy);
                    // Whole-emission: the clone is rigid, so every float
                    // subtree inside it — the unbreakable float itself, or
                    // a float nested in a moved-whole row / avoid box /
                    // abs-pos box — still carries float-local y.
                    // Absolutize them all: paint reads frag.rect
                    // directly, with no parent offset to fall back on.
                    absolutize_floats(&mut b, dy);
                    built.push(Some(b));
                } else {
                    built.push(None);
                }
                continue;
            }
            stack.push(Work {
                node: w.node,
                entered: true,
                own,
                item: w.item,
            });
            // Children of a flex/grid container are flex/grid items: flag
            // them here, where the parent is known.
            let item = is_flex_or_grid_container(w.node);
            for c in w.node.children.iter().rev() {
                stack.push(Work {
                    node: c,
                    entered: false,
                    own: 0.0,
                    item,
                });
            }
        } else {
            let n = w.node.children.len();
            let start = built.len() - n;
            let mut children = Vec::new();
            for o in built.drain(start..).flatten() {
                children.push(o);
            }
            let mut b = page_box_head(w.node, w.own, band_top, page_h);
            // Text boxes split by line fragment: keep the lines in the band.
            if w.node.box_type == BoxType::Text {
                let dy = w.own - page_top;
                let mut frags = Vec::new();
                for f in &w.node.fragments {
                    let fy0 = f.rect.y + w.own;
                    let fy1 = fy0 + f.rect.h;
                    if fy1 > band_top && fy0 < band_bottom {
                        let mut nf = f.clone();
                        nf.rect.y += dy;
                        nf.baseline += dy;
                        frags.push(nf);
                    }
                }
                if frags.is_empty() {
                    built.push(None);
                    continue;
                }
                b.fragments = frags;
            }
            b.children = children;
            if w.node.box_type == BoxType::Table {
                let header_h = repeat_thead(doc, &mut b, w.node, w.own, page_index, page_h);
                repeat_tfoot(doc, &mut b, w.node, w.own, page_index, page_h, header_h);
            }
            built.push(Some(b));
        }
    }
    built.pop().flatten()
}

/// Inclusive page span of `[y, y + h)`; `p0` clamps negative tops to 0.
fn span_of(y: f32, h: f32, page_h: f32) -> (usize, usize) {
    let y1 = y + h.max(0.0);
    let p0 = (y / page_h).floor().max(0.0) as usize;
    let p1 = if y1 <= 0.0 {
        p0
    } else {
        ((y1 / page_h).ceil() as usize).saturating_sub(1).max(p0)
    };
    (p0, p1)
}

fn fragment_tree(
    doc: &Document,
    root: &LayoutBox,
    viewport_w: f32,
    page_height: f32,
    breaks: &HashMap<NodeId, BreakHints>,
) -> Vec<PageFragment> {
    // Clamp first: NaN -> 0 -> 1px floor; negatives -> 1px floor; huge ->
    // 1e15. No division by zero, no infinite pages from a 0px page.
    let page_h = sane_len(page_height).max(1.0);
    let ctx = fragmentation_context(root);
    let ctx_is_root = ptr::eq(ctx, root);

    // Classify the context's children. Flow units are paginated; abs-pos
    // boxes keep their laid-out position and are emitted whole on the page
    // holding their top edge (never sliced); fixed boxes repeat on every
    // page.
    let mut flow_units: Vec<&LayoutBox> = Vec::new();
    let mut flow_span: Vec<(usize, usize)> = Vec::new();
    let mut abs_children: Vec<(&LayoutBox, (usize, usize))> = Vec::new();
    let mut fixed_children: Vec<&LayoutBox> = Vec::new();
    // Per ctx-children index, so page assembly preserves document order.
    enum Kind {
        Flow(usize),
        Abs(usize),
        Fixed(usize),
    }
    let mut kinds: Vec<Kind> = Vec::with_capacity(ctx.children.len());
    for child in &ctx.children {
        let pos = child
            .style
            .as_ref()
            .and_then(|s| s.get_str_by_index(props::POSITION))
            .unwrap_or("static");
        match pos {
            "absolute" => {
                let sp = span_of(child.rect.y, child.rect.h, page_h);
                kinds.push(Kind::Abs(abs_children.len()));
                abs_children.push((child, sp));
            }
            "fixed" => {
                kinds.push(Kind::Fixed(fixed_children.len()));
                fixed_children.push(child);
            }
            _ => {
                kinds.push(Kind::Flow(flow_units.len()));
                flow_units.push(child);
                flow_span.push((0, 0)); // filled after pagination
            }
        }
    }

    let shifts = paginate(&flow_units, page_h, breaks);

    // Phase 1b: collect nested forced-break shifts over the whole tree.
    // Runs after pagination so nested positions account for the phase-1
    // unit shifts. Every split root is walked: flow units with their
    // phase-1 shift as the base, abs-pos children and other root children
    // with base 0 (fixed context children are cloned whole per page and
    // never band-tested).
    let mut nested_roots: Vec<(&LayoutBox, f32)> = Vec::new();
    for (i, u) in flow_units.iter().enumerate() {
        nested_roots.push((u, shifts[i]));
    }
    for (child, _) in &abs_children {
        nested_roots.push((child, 0.0));
    }
    if !ctx_is_root {
        for child in &root.children {
            if !ptr::eq(child, ctx) {
                nested_roots.push((child, 0.0));
            }
        }
    }
    let nested = collect_nested_breaks(&nested_roots, page_h, breaks);

    let mut max_p1 = 0usize;
    for (i, u) in flow_units.iter().enumerate() {
        let sp = span_of(u.rect.y + shifts[i], u.rect.h, page_h);
        max_p1 = max_p1.max(sp.1);
        flow_span[i] = sp;
    }
    for (_, sp) in &abs_children {
        max_p1 = max_p1.max(sp.1);
    }
    // Nested shifts (and nested fixed boxes at viewport coordinates) can
    // extend a unit past its phase-1 span: widen the spans so every
    // affected page splits the unit, and grow the page count to cover the
    // shifted content.
    for (i, u) in flow_units.iter().enumerate() {
        if let Some(e) = nested.get(&(*u as *const LayoutBox)) {
            let lo = e.sub_min_top + shifts[i];
            let sp = span_of(lo, (e.sub_max_bottom + shifts[i] - lo).max(0.0), page_h);
            max_p1 = max_p1.max(sp.1);
            flow_span[i] = sp;
        }
    }
    for (child, sp) in &mut abs_children {
        if let Some(e) = nested.get(&(*child as *const LayoutBox)) {
            let newsp = span_of(
                e.sub_min_top,
                (e.sub_max_bottom - e.sub_min_top).max(0.0),
                page_h,
            );
            max_p1 = max_p1.max(newsp.1);
            *sp = newsp;
        }
    }
    // At least one page, even for empty content; capped (bounded).
    let num_pages = (max_p1 + 1).clamp(1, MAX_PAGES);

    let mut pages = Vec::with_capacity(num_pages);
    for p in 0..num_pages {
        let band_top = p as f32 * page_h;
        let band_bottom = band_top + page_h;
        // The context's page children, in document order.
        let mut ctx_kids: Vec<LayoutBox> = Vec::new();
        for kind in &kinds {
            match *kind {
                Kind::Flow(ui) => {
                    let (p0, p1) = flow_span[ui];
                    if p >= p0
                        && p <= p1.min(num_pages - 1)
                        && let Some(f) = split_subtree(
                            doc,
                            flow_units[ui],
                            band_top,
                            band_bottom,
                            p,
                            page_h,
                            shifts[ui],
                            breaks,
                            &nested,
                            // Flow units are in-flow children of the
                            // fragmentation context: a flow unit is a
                            // flex/grid item when the context is a
                            // flex/grid container (e.g. `display: flex`
                            // on `<html>`), so it rides the unbreakable
                            // fast-path like a nested item.
                            is_flex_or_grid_container(ctx),
                        )
                    {
                        ctx_kids.push(f);
                    }
                }
                Kind::Abs(ai) => {
                    let (child, (p0, p1)) = abs_children[ai];
                    if p >= p0
                        && p <= p1.min(num_pages - 1)
                        && let Some(f) = split_subtree(
                            doc,
                            child,
                            band_top,
                            band_bottom,
                            p,
                            page_h,
                            0.0,
                            breaks,
                            &nested,
                            // Abs-pos children take their own unbreakable
                            // fast-path; they are not flex/grid items.
                            false,
                        )
                    {
                        ctx_kids.push(f);
                    }
                }
                Kind::Fixed(fi) => {
                    // Repeated at the same page-relative position on every
                    // page: no band translation (dy = 0).
                    ctx_kids.push(clone_translated(fixed_children[fi], 0.0));
                }
            }
        }

        let mut proot = page_box_head(root, 0.0, band_top, page_h);
        if ctx_is_root {
            proot.children = ctx_kids;
        } else {
            let mut ctx_kids = Some(ctx_kids);
            for child in &root.children {
                if ptr::eq(child, ctx) {
                    let mut c = page_box_head(child, 0.0, band_top, page_h);
                    c.children = ctx_kids.take().expect("ctx is one of root's children");
                    proot.children.push(c);
                } else if let Some(f) = split_subtree(
                    doc,
                    child,
                    band_top,
                    band_bottom,
                    p,
                    page_h,
                    0.0,
                    breaks,
                    &nested,
                    // Non-context children of the root are items when the
                    // root itself is a flex/grid container.
                    is_flex_or_grid_container(root),
                ) {
                    proot.children.push(f);
                }
            }
        }
        pages.push(PageFragment {
            page_index: p,
            page_count: num_pages,
            size: Size {
                w: viewport_w,
                h: page_h,
            },
            root: proot,
        });
    }
    // Fill in the true page count (known only after the cap).
    let page_count = pages.len();
    for page in &mut pages {
        page.page_count = page_count;
    }
    pages
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedFontMetrics, TABLE_TEST_RESET_CSS, layout, layout_with_scroll};

    const VP: Size = Size { w: 800.0, h: 600.0 };

    fn frag(html: &str, css: &str, page_h: f32) -> Vec<PageFragment> {
        let (doc, _) = Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
        layout_fragmented_with_breaks(&doc, &[&ss], VP, &FixedFontMetrics, page_h, &HashMap::new())
    }

    fn by_id(doc: &Document, id: &str) -> NodeId {
        doc.get_element_by_id(id).expect("test element id")
    }

    fn hints(before: Break, after: Break, inside: BreakInside) -> BreakHints {
        BreakHints {
            before,
            after,
            inside,
            orphans: 2,
            widows: 2,
        }
    }

    /// Parse, then fragment with a breaks map for one element id.
    fn frag_with_break(
        html: &str,
        css: &str,
        page_h: f32,
        id: &str,
        h: BreakHints,
    ) -> (Document, Vec<PageFragment>) {
        let (doc, _) = Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
        let breaks = HashMap::from([(by_id(&doc, id), h)]);
        let pages =
            layout_fragmented_with_breaks(&doc, &[&ss], VP, &FixedFontMetrics, page_h, &breaks);
        (doc, pages)
    }

    /// The `<body>` box of a page tree.
    fn body(page: &PageFragment) -> &LayoutBox {
        &page.root.children[0]
    }

    /// The box for a DOM id inside a page tree, if present on that page.
    fn find<'a>(page: &'a PageFragment, doc: &Document, id: &str) -> Option<&'a LayoutBox> {
        let want = by_id(doc, id);
        let mut stack = vec![&page.root];
        while let Some(n) = stack.pop() {
            if n.node == Some(want) {
                return Some(n);
            }
            stack.extend(n.children.iter());
        }
        None
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn break_before_page_forces_new_page() {
        let (doc, pages) = frag_with_break(
            r#"<div id="a"></div><div id="b"></div>"#,
            "#a { height: 100px; } #b { height: 100px; }",
            400.0,
            "b",
            hints(Break::Page, Break::Auto, BreakInside::Auto),
        );
        assert_eq!(pages.len(), 2, "forced break makes two pages");
        assert_eq!(pages[0].page_index, 0);
        assert_eq!(pages[1].page_index, 1);
        assert_eq!(pages[0].page_count, 2);
        // Page 0 holds only #a; page 1 holds only #b, at the page top.
        assert!(find(&pages[0], &doc, "a").is_some());
        assert!(find(&pages[0], &doc, "b").is_none());
        assert!(find(&pages[1], &doc, "a").is_none());
        let b1 = find(&pages[1], &doc, "b").expect("b on page 2");
        assert!(approx(b1.rect.y, 0.0), "b starts at page top");
        assert!(approx(b1.rect.h, 100.0));
        assert!(approx(pages[0].size.h, 400.0));
        assert!(approx(pages[0].size.w, 800.0));
    }

    #[test]
    fn break_after_page_forces_new_page() {
        let (doc, pages) = frag_with_break(
            r#"<div id="a"></div><div id="b"></div>"#,
            "#a { height: 100px; } #b { height: 100px; }",
            400.0,
            "a",
            hints(Break::Auto, Break::Page, BreakInside::Auto),
        );
        assert_eq!(pages.len(), 2);
        let b1 = find(&pages[1], &doc, "b").expect("b on page 2");
        assert!(approx(b1.rect.y, 0.0));
    }

    #[test]
    fn break_inside_avoid_keeps_block_on_one_page() {
        // #b (300..600) straddles the 400px boundary; avoid moves it whole
        // to page 2 instead of splitting it.
        let (doc, pages) = frag_with_break(
            r#"<div id="a"></div><div id="b"></div>"#,
            "#a { height: 300px; } #b { height: 300px; }",
            400.0,
            "b",
            hints(Break::Auto, Break::Auto, BreakInside::Avoid),
        );
        assert_eq!(pages.len(), 2);
        let a0 = find(&pages[0], &doc, "a").expect("a on page 1");
        assert!(approx(a0.rect.y, 0.0));
        assert!(approx(a0.rect.h, 300.0));
        assert!(find(&pages[0], &doc, "b").is_none(), "b moved off page 1");
        let b1 = find(&pages[1], &doc, "b").expect("b on page 2");
        assert!(approx(b1.rect.y, 0.0), "b starts at page 2 top");
        assert!(approx(b1.rect.h, 300.0), "b unsplit");
    }

    #[test]
    fn content_flows_across_pages() {
        let css =
            "#a{height:300px}#b{height:300px}#c{height:300px}#d{height:300px}#e{height:300px}";
        let pages = frag(
            r#"<div id="a"></div><div id="b"></div><div id="c"></div><div id="d"></div><div id="e"></div>"#,
            css,
            400.0,
        );
        assert_eq!(pages.len(), 4, "1500px over 400px pages");
        // Page 2 (band 400..800) holds the tail of #b and the head of #c.
        assert_eq!(body(&pages[1]).children.len(), 2);
        let kids = &body(&pages[1]).children;
        assert!(approx(kids[0].rect.y, 0.0));
        assert!(approx(kids[0].rect.h, 200.0), "tail of #b");
        assert!(approx(kids[1].rect.y, 200.0));
        assert!(approx(kids[1].rect.h, 200.0), "head of #c");
        // Last page holds only the tail of #e.
        assert_eq!(body(&pages[3]).children.len(), 1);
        assert!(approx(body(&pages[3]).children[0].rect.h, 300.0));
    }

    #[test]
    fn break_after_avoid_keeps_with_next() {
        // #s fills 0..300, #a sits 300..400 (page bottom), #b follows.
        // break-after: avoid on #a pulls it onto page 2 with #b.
        let (doc, pages) = frag_with_break(
            r#"<div id="s"></div><div id="a"></div><div id="b"></div>"#,
            "#s { height: 300px; } #a { height: 100px; } #b { height: 200px; }",
            400.0,
            "a",
            hints(Break::Auto, Break::Avoid, BreakInside::Auto),
        );
        assert_eq!(pages.len(), 2);
        assert!(find(&pages[0], &doc, "a").is_none(), "a kept with b");
        let a1 = find(&pages[1], &doc, "a").expect("a on page 2");
        let b1 = find(&pages[1], &doc, "b").expect("b on page 2");
        assert!(approx(a1.rect.y, 0.0));
        assert!(approx(b1.rect.y, 100.0));
    }

    #[test]
    fn break_before_page_at_page_top_is_noop() {
        // #b already starts exactly at a page top: no blank page appears.
        let (doc, pages) = frag_with_break(
            r#"<div id="a"></div><div id="b"></div>"#,
            "#a { height: 400px; } #b { height: 100px; }",
            400.0,
            "b",
            hints(Break::Page, Break::Auto, BreakInside::Auto),
        );
        assert_eq!(pages.len(), 2, "no blank page inserted");
        let b1 = find(&pages[1], &doc, "b").expect("b on page 2");
        assert!(approx(b1.rect.y, 0.0));
    }

    #[test]
    fn fixed_box_repeats_on_every_page() {
        let pages = frag(
            r#"<div id="f"></div><div id="c"></div>"#,
            "#f { position: fixed; top: 0; height: 50px; } #c { height: 700px; }",
            400.0,
        );
        assert_eq!(pages.len(), 2);
        for (i, page) in pages.iter().enumerate() {
            assert!(
                body(page)
                    .children
                    .iter()
                    .any(|k| approx(k.rect.y, 0.0) && approx(k.rect.h, 50.0)),
                "fixed box on page {i}"
            );
        }
    }

    #[test]
    fn text_fragments_stay_inside_page_bands() {
        // One word per line (narrow block), 6 lines at 19.2px each; a
        // 38.4px page holds exactly two lines, so every emitted fragment
        // must lie inside its page band.
        let page_h = 38.4;
        let pages = frag(
            r#"<div id="t">aa aa aa aa aa aa</div>"#,
            "#t { font-size: 16px; line-height: 1.2; width: 20px; }",
            page_h,
        );
        assert_eq!(pages.len(), 3, "115.2px of lines over 38.4px pages");
        for page in &pages {
            let mut stack = vec![&page.root];
            while let Some(n) = stack.pop() {
                for f in &n.fragments {
                    assert!(
                        f.rect.y >= -0.01 && f.rect.y + f.rect.h <= page_h + 0.01,
                        "fragment inside page band"
                    );
                }
                stack.extend(n.children.iter());
            }
        }
    }

    /// Parse, then fragment with no break overrides, returning the document
    /// too (for [`find`]).
    fn frag_doc(html: &str, css: &str, page_h: f32) -> (Document, Vec<PageFragment>) {
        let (doc, _) = Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
        let pages = layout_fragmented_with_breaks(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            page_h,
            &HashMap::new(),
        );
        (doc, pages)
    }

    /// Total text fragments in the box's subtree. The test paragraphs are
    /// single text nodes laid out one word per line, so this counts lines.
    fn fragment_count(n: &LayoutBox) -> usize {
        let mut count = 0;
        let mut stack = vec![n];
        while let Some(b) = stack.pop() {
            count += b.fragments.len();
            stack.extend(b.children.iter());
        }
        count
    }

    #[test]
    fn widows_orphans_enforced_from_css() {
        // 8 lines at 19.2px = 153.6px; a 140px page would naturally split
        // 7/1, stranding one line on page 2. With orphans: 3 / widows: 3
        // the last fragment must start with at least 3 lines, so the break
        // moves earlier: line 5 aligns to the top of page 2, leaving 5
        // lines on page 1 and 3 on page 2.
        let (doc, pages) = frag_doc(
            r#"<p id="p">aa aa aa aa aa aa aa aa</p>"#,
            "body { margin: 0; } #p { margin: 0; font-size: 16px; line-height: 1.2; width: 20px; orphans: 3; widows: 3; }",
            140.0,
        );
        assert_eq!(pages.len(), 2);
        let p0 = find(&pages[0], &doc, "p").expect("p on page 1");
        let p1 = find(&pages[1], &doc, "p").expect("p on page 2");
        assert!(
            approx(p0.rect.y, 44.0),
            "split moved earlier by the widows push"
        );
        assert_eq!(fragment_count(p0), 5, "orphans: >= 3 lines stay on page 1");
        assert!(approx(p1.rect.y, 0.0), "widows lines start at page 2 top");
        // >= : a 1-ulp sliver of the boundary line may touch page 2.
        assert!(fragment_count(p1) >= 3, "widows: >= 3 lines start page 2");
    }

    #[test]
    fn orphans_pushes_whole_block_to_next_page() {
        // #s fills 0..170; the 8-line paragraph (153.6px) would naturally
        // start at 170 with only 2 lines on page 1. orphans: 3 cannot be
        // met there, so the whole paragraph moves to page 2.
        let (doc, pages) = frag_doc(
            r#"<div id="s"></div><p id="p">aa aa aa aa aa aa aa aa</p>"#,
            "body { margin: 0; } #s { height: 170px; } #p { margin: 0; font-size: 16px; line-height: 1.2; width: 20px; orphans: 3; widows: 3; }",
            200.0,
        );
        assert_eq!(pages.len(), 2);
        assert!(
            find(&pages[0], &doc, "p").is_none(),
            "no stranded lines on page 1"
        );
        let p1 = find(&pages[1], &doc, "p").expect("p on page 2");
        assert!(approx(p1.rect.y, 0.0), "p starts at page 2 top");
        assert_eq!(fragment_count(p1), 8, "p kept whole");
    }

    #[test]
    fn default_widows_orphans_are_two() {
        // 5 lines at 19.2px = 96px; an 80px page would naturally split 4/1
        // (line 4 straddles the boundary). The CSS initial values
        // (orphans: 2, widows: 2) push one line down: line 3 aligns to the
        // top of page 2, leaving 3 lines on page 1 and 2 on page 2.
        let (doc, pages) = frag_doc(
            r#"<p id="p">aa aa aa aa aa</p>"#,
            "body { margin: 0; } #p { margin: 0; font-size: 16px; line-height: 1.2; width: 20px; }",
            80.0,
        );
        assert_eq!(pages.len(), 2);
        let p0 = find(&pages[0], &doc, "p").expect("p on page 1");
        let p1 = find(&pages[1], &doc, "p").expect("p on page 2");
        assert!(
            approx(p0.rect.y, 22.4),
            "split moved earlier by the widows push"
        );
        assert_eq!(fragment_count(p0), 3);
        assert!(approx(p1.rect.y, 0.0));
        // >= : a 1-ulp sliver of the boundary line may touch page 2.
        assert!(fragment_count(p1) >= 2);
    }

    #[test]
    fn widows_orphans_honored_from_breaks_map() {
        // Hints from the per-node override map win over computed style.
        // 5 lines with orphans: 3 / widows: 3 is jointly unsatisfiable
        // (5 < 3 + 3), so orphans wins, widows is best-effort, and the
        // natural split is kept: no shift, line 4 straddles the boundary.
        let (doc, pages) = frag_with_break(
            r#"<p id="p">aa aa aa aa aa</p>"#,
            "body { margin: 0; } #p { margin: 0; font-size: 16px; line-height: 1.2; width: 20px; }",
            80.0,
            "p",
            BreakHints {
                before: Break::Auto,
                after: Break::Auto,
                inside: BreakInside::Auto,
                orphans: 3,
                widows: 3,
            },
        );
        assert_eq!(pages.len(), 2);
        let p0 = find(&pages[0], &doc, "p").expect("p on page 1");
        let p1 = find(&pages[1], &doc, "p").expect("p on page 2");
        assert!(approx(p0.rect.y, 0.0), "no pointless shift");
        assert_eq!(fragment_count(p0), 5, "natural split kept");
        assert_eq!(fragment_count(p1), 1);
    }

    #[test]
    fn widows_applies_to_last_fragment_of_long_block() {
        // 9 lines at 19.2px = 172.8px over 80px pages: naturally 4/4/1 by
        // line mid, stranding one line on page 3. Default widows: 2 moves
        // the last break earlier (line 7 aligns to the top of page 3).
        let (doc, pages) = frag_doc(
            r#"<p id="p">aa aa aa aa aa aa aa aa aa</p>"#,
            "body { margin: 0; } #p { margin: 0; font-size: 16px; line-height: 1.2; width: 20px; }",
            80.0,
        );
        assert_eq!(pages.len(), 3);
        let p0 = find(&pages[0], &doc, "p").expect("p on page 1");
        let p1 = find(&pages[1], &doc, "p").expect("p on page 2");
        let p2 = find(&pages[2], &doc, "p").expect("p on page 3");
        assert!(approx(p0.rect.y, 25.6), "last break moved earlier");
        assert_eq!(fragment_count(p0), 3);
        // Line 2 straddles the first boundary, so it is emitted on both
        // pages 1 and 2.
        assert_eq!(fragment_count(p1), 5);
        assert!(approx(p2.rect.y, 0.0), "widows lines start at page 3 top");
        // >= : a 1-ulp sliver of the boundary line may touch page 3.
        assert!(fragment_count(p2) >= 2, "widows: >= 2 lines start page 3");
    }

    #[test]
    fn single_page_matches_plain_layout() {
        // With a huge page height, fragmentation must reproduce layout().
        let html = r#"<div id="a"><div id="b"></div></div><div id="c"></div>"#;
        let css = "#a { height: 100px; } #b { height: 30px; } #c { height: 50px; }";
        let (doc, _) = Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let pages = layout_fragmented(&doc, &[&ss], VP, &FixedFontMetrics, 100000.0);
        assert_eq!(pages.len(), 1);
        let plain = layout(&doc, &[&ss], VP, &FixedFontMetrics);
        assert_trees_equal(&pages[0].root, &plain);
    }

    fn assert_trees_equal(a: &LayoutBox, b: &LayoutBox) {
        assert_eq!(a.box_type, b.box_type);
        assert!(approx(a.rect.x, b.rect.x));
        assert!(approx(a.rect.y, b.rect.y));
        assert!(approx(a.rect.w, b.rect.w));
        assert!(approx(a.rect.h, b.rect.h));
        assert_eq!(a.text, b.text);
        assert_eq!(a.children.len(), b.children.len(), "child count");
        for (x, y) in a.children.iter().zip(b.children.iter()) {
            assert_trees_equal(x, y);
        }
    }

    #[test]
    fn hostile_page_heights_cannot_hang() {
        let html = r#"<div id="a"></div>"#;
        let css = "#a { height: 600px; }";
        // Zero, negative, and NaN page heights clamp to a 1px floor.
        for h in [0.0, -50.0, f32::NAN] {
            let pages = frag(html, css, h);
            assert_eq!(pages.len(), 600, "1px pages for height {h}");
            assert!(approx(pages[0].size.h, 1.0));
        }
    }

    #[test]
    fn hostile_tall_avoid_box_fragments_bounded() {
        // break-inside: avoid on a 10000px box: taller than the page, so it
        // fragments instead of shifting forever; page count stays bounded.
        let (_doc, pages) = frag_with_break(
            r#"<div id="a"></div>"#,
            "#a { height: 10000px; }",
            400.0,
            "a",
            hints(Break::Auto, Break::Auto, BreakInside::Avoid),
        );
        assert_eq!(pages.len(), 25, "10000px / 400px pages");
    }

    #[test]
    fn hostile_gigantic_content_truncates_at_max_pages() {
        let (doc, _) = Document::parse_html(br#"<div id="a"></div>"#);
        let css = "#a { height: 5000000px; }";
        let ss = bowser_css::parse_stylesheet(css);
        let pages = layout_fragmented(&doc, &[&ss], VP, &FixedFontMetrics, 1.0);
        assert_eq!(pages.len(), MAX_PAGES, "page count capped");
        assert_eq!(pages[MAX_PAGES - 1].page_count, MAX_PAGES);
    }

    #[test]
    fn empty_document_yields_one_page() {
        let pages = frag("", "", 400.0);
        assert_eq!(pages.len(), 1);
    }

    #[test]
    fn fragmented_with_intrinsics_matches_fragmented() {
        // No replaced elements in this doc: the intrinsics map is unused,
        // so the variant must reproduce layout_fragmented exactly.
        let html = r#"<div id="a"></div><div id="b"></div>"#;
        let css = "#a { height: 300px; } #b { height: 300px; }";
        let (doc, _) = Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let base = layout_fragmented(&doc, &[&ss], VP, &FixedFontMetrics, 400.0);
        let with = layout_fragmented_with_intrinsics(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            400.0,
            &HashMap::new(),
        );
        assert_eq!(base.len(), with.len(), "same page count");
        for (a, b) in base.iter().zip(with.iter()) {
            assert_trees_equal(&a.root, &b.root);
        }
    }

    #[test]
    fn fragmented_with_intrinsics_sizes_replaced_elements() {
        // The intrinsics map actually flows through: an unsized <img>
        // takes its box size from the map.
        let (doc, _) = Document::parse_html(br#"<img id="pic">"#);
        let ss = bowser_css::parse_stylesheet("");
        let intrinsics =
            HashMap::from([(by_id(&doc, "pic"), IntrinsicDims::from_size(300.0, 150.0))]);
        let pages = layout_fragmented_with_intrinsics(
            &doc,
            &[&ss],
            VP,
            &FixedFontMetrics,
            400.0,
            &intrinsics,
        );
        assert_eq!(pages.len(), 1);
        let pic = find(&pages[0], &doc, "pic").expect("img on page 1");
        assert!(approx(pic.rect.w, 300.0), "w={}", pic.rect.w);
        assert!(approx(pic.rect.h, 150.0), "h={}", pic.rect.h);
    }

    #[test]
    fn fragmented_with_scroll_applies_sticky_before_pagination() {
        // The sticky box's in-flow spot is y = 300 (after the spacer).
        // Scrolled to 400 with top: 10 it sticks at y = 410, so
        // pagination sees the shifted tree: the box lands on page 2 at
        // page-relative y = 10 instead of page 1 at y = 300.
        let html = r#"<div class="spacer"></div><div class="sticky" id="st"></div><div class="filler" id="fl"></div>"#;
        let css = "body { margin: 0; } .spacer { height: 300px; } \
                   .sticky { position: sticky; top: 10px; height: 50px; } \
                   .filler { height: 600px; }";
        let (doc, _) = Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let scroll = (0.0, 400.0);
        let pages =
            layout_fragmented_with_scroll(&doc, &[&ss], VP, &FixedFontMetrics, 400.0, scroll);
        assert_eq!(pages.len(), 3);
        assert!(
            find(&pages[0], &doc, "st").is_none(),
            "stuck box left page 1"
        );
        let st = find(&pages[1], &doc, "st").expect("sticky on page 2");
        assert!(
            approx(st.rect.y, 10.0),
            "page-relative y, got {}",
            st.rect.y
        );
        assert!(approx(st.rect.h, 50.0));
        // The stuck box must not drag following content: the filler keeps
        // its in-flow spot (350..400 head on page 1), overlapping the
        // stuck box exactly as in the viewport.
        let fl0 = find(&pages[0], &doc, "fl").expect("filler head on page 1");
        assert!(approx(fl0.rect.y, 350.0), "y={}", fl0.rect.y);
        assert!(approx(fl0.rect.h, 50.0), "h={}", fl0.rect.h);
        // Without scroll the same box stays at its in-flow spot on page 1.
        // (An unstuck sticky box stays exactly where it was laid out:
        // the inset only constrains, it never shifts the box.)
        let base = layout_fragmented(&doc, &[&ss], VP, &FixedFontMetrics, 400.0);
        let st0 = find(&base[0], &doc, "st").expect("sticky on page 1 unscrolled");
        assert!(approx(st0.rect.y, 300.0), "y={}", st0.rect.y);
        // The single-page scroll variant reproduces layout_with_scroll
        // exactly: sticky is applied before (not during) pagination.
        let single =
            layout_fragmented_with_scroll(&doc, &[&ss], VP, &FixedFontMetrics, 100000.0, scroll);
        assert_eq!(single.len(), 1);
        let plain = layout_with_scroll(&doc, &[&ss], VP, &FixedFontMetrics, scroll);
        assert_trees_equal(&single[0].root, &plain);
    }

    #[test]
    fn nested_break_before_page_honored() {
        // #inner is nested inside #outer (the flow unit): its
        // break-before: page must still push it to the top of page 2.
        let (doc, pages) = frag_with_break(
            r#"<div id="outer"><div id="pad"></div><div id="inner"></div></div>"#,
            "body { margin: 0; } #pad { height: 300px; } #inner { height: 100px; }",
            400.0,
            "inner",
            hints(Break::Page, Break::Auto, BreakInside::Auto),
        );
        assert_eq!(pages.len(), 2);
        assert!(find(&pages[0], &doc, "inner").is_none());
        let inner = find(&pages[1], &doc, "inner").expect("inner on page 2");
        assert!(approx(inner.rect.y, 0.0), "y={}", inner.rect.y);
        assert!(approx(inner.rect.h, 100.0));
        // The preceding nested sibling is undisturbed on page 1.
        let pad = find(&pages[0], &doc, "pad").expect("pad on page 1");
        assert!(approx(pad.rect.y, 0.0));
        assert!(approx(pad.rect.h, 300.0));
    }

    #[test]
    fn nested_break_before_page_at_page_top_is_noop() {
        // #inner already starts at a page top: no blank page appears.
        let (doc, pages) = frag_with_break(
            r#"<div id="outer"><div id="inner"></div><div id="pad"></div></div>"#,
            "body { margin: 0; } #inner { height: 100px; } #pad { height: 300px; }",
            400.0,
            "inner",
            hints(Break::Page, Break::Auto, BreakInside::Auto),
        );
        assert_eq!(pages.len(), 1, "no blank page inserted");
        let inner = find(&pages[0], &doc, "inner").expect("inner on page 1");
        assert!(approx(inner.rect.y, 0.0));
    }

    #[test]
    fn nested_break_after_page_pushes_following_content() {
        // #mid's break-after: page pushes the following nested #next (not
        // #mid itself) to the top of page 2.
        let (doc, pages) = frag_with_break(
            r#"<div id="outer"><div id="pad"></div><div id="mid"></div><div id="next"></div></div>"#,
            "body { margin: 0; } #pad { height: 250px; } #mid { height: 100px; } #next { height: 100px; }",
            400.0,
            "mid",
            hints(Break::Auto, Break::Page, BreakInside::Auto),
        );
        assert_eq!(pages.len(), 2);
        let mid = find(&pages[0], &doc, "mid").expect("mid stays on page 1");
        assert!(approx(mid.rect.y, 250.0), "y={}", mid.rect.y);
        assert!(find(&pages[0], &doc, "next").is_none());
        let next = find(&pages[1], &doc, "next").expect("next on page 2");
        assert!(approx(next.rect.y, 0.0), "y={}", next.rect.y);
        assert!(approx(next.rect.h, 100.0));
    }

    #[test]
    fn nested_break_after_avoid_keeps_with_next() {
        // #a (300..400) sits wholly on page 1 while #b would start page 2;
        // break-after: avoid pulls #a down to page 2 with #b.
        let (doc, pages) = frag_with_break(
            r#"<div id="outer"><div id="s"></div><div id="a"></div><div id="b"></div></div>"#,
            "body { margin: 0; } #s { height: 300px; } #a { height: 100px; } #b { height: 200px; }",
            400.0,
            "a",
            hints(Break::Auto, Break::Avoid, BreakInside::Auto),
        );
        assert_eq!(pages.len(), 2);
        assert!(find(&pages[0], &doc, "a").is_none(), "a kept with b");
        let a1 = find(&pages[1], &doc, "a").expect("a on page 2");
        let b1 = find(&pages[1], &doc, "b").expect("b on page 2");
        assert!(approx(a1.rect.y, 0.0), "y={}", a1.rect.y);
        assert!(approx(b1.rect.y, 100.0), "y={}", b1.rect.y);
    }

    #[test]
    fn nested_fixed_repeats_on_every_page() {
        // A fixed box nested inside a flow unit repeats whole on every
        // page instead of being sliced geometrically where it falls.
        let (doc, pages) = frag_doc(
            r#"<div id="outer"><div id="f"></div><div id="c"></div></div>"#,
            "body { margin: 0; } #f { position: fixed; top: 0; height: 50px; } #c { height: 700px; }",
            400.0,
        );
        assert_eq!(pages.len(), 2);
        for (i, page) in pages.iter().enumerate() {
            let f = find(page, &doc, "f").expect("fixed box repeats");
            assert!(approx(f.rect.y, 0.0), "page {i}: y={}", f.rect.y);
            assert!(approx(f.rect.h, 50.0), "page {i}: unsliced");
        }
    }

    #[test]
    fn table_row_not_split_across_pages() {
        // Row 2 starts at y=300 and straddles the 400px boundary: it must
        // move whole to the top of page 2 instead of splitting mid-row or
        // pinning to page 1 and overflowing the page bottom.
        let html = r#"<table id="t"><tr id="r1"><td><div id="d1"></div></td></tr><tr id="r2"><td><div id="d2"></div></td></tr></table>"#;
        let css = "body { margin: 0; } #d1 { height: 300px; } #d2 { height: 150px; }";
        let (doc, pages) = frag_doc(html, css, 400.0);
        // Full row geometry from the single-page fragmentation.
        let (doc2, _) = Document::parse_html(html.as_bytes());
        // Same TABLE_TEST_RESET_CSS as `frag_doc` above, so the
        // single-page reference and the fragmented output are styled
        // identically.
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
        let single = layout_fragmented(&doc2, &[&ss], VP, &FixedFontMetrics, 100000.0);
        let full = find(&single[0], &doc2, "r2").expect("r2 in single-page");
        let (full_y, full_h) = (full.rect.y, full.rect.h);
        assert!(full_y < 400.0 && full_y + full_h > 400.0, "row straddles");
        assert_eq!(pages.len(), 2);
        assert!(
            find(&pages[0], &doc, "r2").is_none(),
            "row not pinned to page 1"
        );
        let r2 = find(&pages[1], &doc, "r2").expect("r2 whole on page 2");
        assert!(approx(r2.rect.y, 0.0), "y={}", r2.rect.y);
        assert!(approx(r2.rect.h, full_h), "h={}", r2.rect.h);
    }

    #[test]
    fn nested_break_inside_avoid_moves_to_next_page() {
        // #inner (300..450) straddles the 400px boundary with
        // break-inside: avoid: it moves whole to the top of page 2
        // instead of pinning to page 1 and overflowing the page bottom.
        let (doc, pages) = frag_with_break(
            r#"<div id="outer"><div id="s"></div><div id="inner"></div></div>"#,
            "body { margin: 0; } #s { height: 300px; } #inner { height: 150px; }",
            400.0,
            "inner",
            hints(Break::Auto, Break::Auto, BreakInside::Avoid),
        );
        assert_eq!(pages.len(), 2);
        assert!(
            find(&pages[0], &doc, "inner").is_none(),
            "inner moved off page 1"
        );
        let inner = find(&pages[1], &doc, "inner").expect("inner whole on page 2");
        assert!(approx(inner.rect.y, 0.0), "y={}", inner.rect.y);
        assert!(approx(inner.rect.h, 150.0), "h={}", inner.rect.h);
    }

    #[test]
    fn nested_avoid_taller_than_page_still_fragments() {
        // break-inside: avoid on a box taller than the page cannot move it
        // whole to the next page; it fragments geometrically like any
        // other box (avoidance never shifts unboundedly).
        let (doc, pages) = frag_with_break(
            r#"<div id="outer"><div id="s"></div><div id="tall"></div></div>"#,
            "body { margin: 0; } #s { height: 300px; } #tall { height: 500px; }",
            400.0,
            "tall",
            hints(Break::Auto, Break::Auto, BreakInside::Avoid),
        );
        assert_eq!(pages.len(), 2);
        let t0 = find(&pages[0], &doc, "tall").expect("tall fragment on page 1");
        let t1 = find(&pages[1], &doc, "tall").expect("tall fragment on page 2");
        assert!(approx(t0.rect.y, 300.0), "y={}", t0.rect.y);
        assert!(approx(t0.rect.h, 100.0), "h={}", t0.rect.h);
        assert!(approx(t1.rect.y, 0.0), "y={}", t1.rect.y);
        assert!(approx(t1.rect.h, 400.0), "h={}", t1.rect.h);
    }

    #[test]
    fn float_straddling_boundary_moves_whole_to_next_page() {
        // #fl (300..450) straddles the 400px boundary: a float never
        // splits across pages, so it moves whole to the top of page 2
        // instead of being sliced like an ordinary block. Seven lines
        // keep the default widows/orphans quiet, isolating the float
        // fast-path: without it page 1 would hold the first six lines.
        let (doc, pages) = frag_doc(
            r#"<div id="s"></div><div id="fl">aa aa aa aa aa aa aa</div>"#,
            "body { margin: 0; } #s { height: 300px; } \
             #fl { float: left; width: 20px; height: 150px; font-size: 16px; line-height: 1.2; }",
            400.0,
        );
        assert_eq!(pages.len(), 2);
        assert!(
            find(&pages[0], &doc, "fl").is_none(),
            "float moved off page 1"
        );
        let fl = find(&pages[1], &doc, "fl").expect("float whole on page 2");
        assert!(approx(fl.rect.y, 0.0), "y={}", fl.rect.y);
        assert!(approx(fl.rect.h, 150.0), "h={}", fl.rect.h);
        assert_eq!(fragment_count(fl), 7, "all lines moved whole");
    }

    #[test]
    fn float_taller_than_page_still_fragments() {
        // A float taller than the page cannot move whole; it fragments
        // geometrically (bounded: no unbounded shifting). orphans/widows
        // are pinned to 1 so the single line cannot trigger a push.
        let (doc, pages) = frag_doc(
            r#"<div id="s"></div><div id="fl">x</div>"#,
            "body { margin: 0; } #s { height: 300px; } \
             #fl { float: left; width: 100px; height: 500px; orphans: 1; widows: 1; }",
            400.0,
        );
        assert_eq!(pages.len(), 2);
        let f0 = find(&pages[0], &doc, "fl").expect("float fragment on page 1");
        let f1 = find(&pages[1], &doc, "fl").expect("float fragment on page 2");
        assert!(approx(f0.rect.y, 300.0), "y={}", f0.rect.y);
        assert!(approx(f0.rect.h, 100.0), "h={}", f0.rect.h);
        assert!(approx(f1.rect.y, 0.0), "y={}", f1.rect.y);
        assert!(approx(f1.rect.h, 400.0), "h={}", f1.rect.h);
    }

    #[test]
    fn abs_pos_box_not_sliced_across_pages() {
        // #ap (300..450) straddles the 400px boundary: absolutely
        // positioned boxes are emitted whole on the page holding their top
        // edge, moving whole to the next page when they straddle and fit
        // on one — never sliced geometrically.
        let (doc, pages) = frag_doc(
            r#"<div id="s"></div><div id="ap"></div>"#,
            "body { margin: 0; } #s { height: 300px; } #ap { position: absolute; top: 300px; left: 0; width: 100px; height: 150px; }",
            400.0,
        );
        assert_eq!(pages.len(), 2);
        assert!(
            find(&pages[0], &doc, "ap").is_none(),
            "abs box moved off page 1"
        );
        let ap = find(&pages[1], &doc, "ap").expect("abs box whole on page 2");
        assert!(approx(ap.rect.y, 0.0), "y={}", ap.rect.y);
        assert!(approx(ap.rect.h, 150.0), "h={}", ap.rect.h);
    }

    #[test]
    fn abs_pos_box_within_one_page_keeps_position() {
        // A non-straddling abs-pos box stays exactly where layout put it.
        let (doc, pages) = frag_doc(
            r#"<div id="s"></div><div id="ap"></div>"#,
            "body { margin: 0; } #s { height: 300px; } #ap { position: absolute; top: 100px; left: 0; width: 100px; height: 150px; }",
            400.0,
        );
        assert_eq!(pages.len(), 1);
        let ap = find(&pages[0], &doc, "ap").expect("abs box on page 1");
        assert!(approx(ap.rect.y, 100.0), "y={}", ap.rect.y);
        assert!(approx(ap.rect.h, 150.0), "h={}", ap.rect.h);
    }

    #[test]
    fn thead_repeats_on_continuation_pages() {
        // thead (0..40) + six 100px rows over 400px pages: rows 1-3 on
        // page 1 (row 4 moves whole to page 2, rows never split mid-row).
        // The continuation page repeats the thead whole at its top and
        // the continued rows shift down below it, unsliced.
        let html = concat!(
            r#"<table id="t"><thead id="th"><tr><td><div id="hh"></div></td></tr></thead><tbody>"#,
            r#"<tr id="r1"><td><div id="d1"></div></td></tr>"#,
            r#"<tr id="r2"><td><div id="d2"></div></td></tr>"#,
            r#"<tr id="r3"><td><div id="d3"></div></td></tr>"#,
            r#"<tr id="r4"><td><div id="d4"></div></td></tr>"#,
            r#"<tr id="r5"><td><div id="d5"></div></td></tr>"#,
            r#"<tr id="r6"><td><div id="d6"></div></td></tr>"#,
            r#"</tbody></table>"#,
        );
        let css = "body { margin: 0; } #hh { height: 40px; } \
                   #d1,#d2,#d3,#d4,#d5,#d6 { height: 100px; }";
        let (doc, pages) = frag_doc(html, css, 400.0);
        assert_eq!(pages.len(), 2);
        // Page 1 shows the header naturally, exactly once.
        let th0 = find(&pages[0], &doc, "th").expect("thead on page 1");
        assert!(approx(th0.rect.y, 0.0), "y={}", th0.rect.y);
        assert!(approx(th0.rect.h, 40.0), "h={}", th0.rect.h);
        assert!(find(&pages[0], &doc, "r4").is_none(), "r4 moved off page 1");
        // Page 2 repeats the header whole at the page top...
        let th1 = find(&pages[1], &doc, "th").expect("thead repeated on page 2");
        assert!(approx(th1.rect.y, 0.0), "y={}", th1.rect.y);
        assert!(approx(th1.rect.h, 40.0), "h={}", th1.rect.h);
        // ...and the continued rows sit below it, unsliced.
        let r4 = find(&pages[1], &doc, "r4").expect("r4 on page 2");
        assert!(approx(r4.rect.y, 40.0), "y={}", r4.rect.y);
        assert!(approx(r4.rect.h, 100.0), "h={}", r4.rect.h);
        let r6 = find(&pages[1], &doc, "r6").expect("r6 on page 2");
        assert!(approx(r6.rect.y, 180.0), "y={}", r6.rect.y);
        assert!(approx(r6.rect.h, 100.0), "h={}", r6.rect.h);
    }

    #[test]
    fn thead_repeats_when_table_starts_mid_page() {
        // Same table pushed 350px down by a spacer: the continuation pages
        // still repeat the thead at the page top (page-relative 0), not at
        // its absolute document offset.
        let html = concat!(
            r#"<div id="sp"></div><table id="t"><thead id="th"><tr><td><div id="hh"></div></td></tr></thead><tbody>"#,
            r#"<tr id="r1"><td><div id="d1"></div></td></tr>"#,
            r#"<tr id="r2"><td><div id="d2"></div></td></tr>"#,
            r#"<tr id="r3"><td><div id="d3"></div></td></tr>"#,
            r#"<tr id="r4"><td><div id="d4"></div></td></tr>"#,
            r#"<tr id="r5"><td><div id="d5"></div></td></tr>"#,
            r#"<tr id="r6"><td><div id="d6"></div></td></tr>"#,
            r#"</tbody></table>"#,
        );
        let css = "body { margin: 0; } #sp { height: 350px; } #hh { height: 40px; } \
                   #d1,#d2,#d3,#d4,#d5,#d6 { height: 100px; }";
        let (doc, pages) = frag_doc(html, css, 400.0);
        assert_eq!(pages.len(), 3);
        // Page 2: repetition is SKIPPED here — the continued rows already
        // reach y=390, so shifting them down by the 40px header would push
        // r4 past the 400px page bottom (issue 7a spill bound). r1 (moved
        // whole off the page-1 boundary) stays at the page top.
        assert!(
            find(&pages[1], &doc, "th").is_none(),
            "no thead repeated on page 2 (spill bound)"
        );
        let r1 = find(&pages[1], &doc, "r1").expect("r1 on page 2");
        assert!(approx(r1.rect.y, 0.0), "y={}", r1.rect.y);
        // Page 3: rows only reach y=190, so the header repeats at the page
        // top as before.
        let th2 = find(&pages[2], &doc, "th").expect("thead repeated on page 3");
        assert!(approx(th2.rect.y, 0.0), "y={}", th2.rect.y);
        let r5 = find(&pages[2], &doc, "r5").expect("r5 on page 3");
        assert!(approx(r5.rect.y, 40.0), "y={}", r5.rect.y);
    }

    #[test]
    fn thead_not_duplicated_on_single_page_table() {
        // A table that fits on one page shows its thead exactly once:
        // no repetition clone is emitted.
        let html = r#"<table id="t"><thead id="th"><tr><td><div id="hh"></div></td></tr></thead><tbody><tr id="r1"><td><div id="d1"></div></td></tr></tbody></table>"#;
        let css = "body { margin: 0; } #hh { height: 40px; } #d1 { height: 100px; }";
        let (doc, pages) = frag_doc(html, css, 400.0);
        assert_eq!(pages.len(), 1);
        let th = find(&pages[0], &doc, "th").expect("thead on the page");
        assert!(approx(th.rect.h, 40.0), "h={}", th.rect.h);
        // Exactly one thead box in the whole page tree.
        let mut count = 0;
        let mut stack = vec![&pages[0].root];
        let want = by_id(&doc, "th");
        while let Some(n) = stack.pop() {
            if n.node == Some(want) {
                count += 1;
            }
            stack.extend(n.children.iter());
        }
        assert_eq!(count, 1, "thead appears exactly once");
    }

    #[test]
    fn flex_body_is_fragmentation_context() {
        // `display: flex` lays out as a block-level box, so a flex <body>
        // is the fragmentation context and its children paginate as flow
        // units. Per-unit orphans then applies to #p2 alone: with only one
        // of its 8 lines fitting at the bottom of page 1 (orphans: 3), the
        // whole paragraph moves to page 2. (If the root were the context,
        // #p1's lines would count toward the orphans budget and no push
        // would fire.)
        let (doc, pages) = frag_doc(
            r#"<p id="p1">aa aa aa aa aa aa aa aa aa aa</p><p id="p2">aa aa aa aa aa aa aa aa</p>"#,
            "body { margin: 0; display: flex; flex-direction: column; } \
             p { margin: 0; font-size: 16px; line-height: 1.2; width: 20px; } \
             #p2 { orphans: 3; }",
            200.0,
        );
        assert_eq!(pages.len(), 2);
        assert!(
            find(&pages[0], &doc, "p2").is_none(),
            "orphans push moved p2 whole to page 2"
        );
        let p2 = find(&pages[1], &doc, "p2").expect("p2 on page 2");
        assert!(approx(p2.rect.y, 0.0), "y={}", p2.rect.y);
        assert_eq!(fragment_count(p2), 8, "p2 kept whole");
    }

    /// Find a box by DOM id in a plain (non-page) layout tree.
    fn find_in<'a>(root: &'a LayoutBox, doc: &Document, id: &str) -> Option<&'a LayoutBox> {
        let want = by_id(doc, id);
        let mut stack = vec![root];
        while let Some(n) = stack.pop() {
            if n.node == Some(want) {
                return Some(n);
            }
            stack.extend(n.children.iter());
        }
        None
    }

    /// All boxes for a DOM id inside a page tree.
    fn find_all<'a>(page: &'a PageFragment, doc: &Document, id: &str) -> Vec<&'a LayoutBox> {
        let want = by_id(doc, id);
        let mut out = Vec::new();
        let mut stack = vec![&page.root];
        while let Some(n) = stack.pop() {
            if n.node == Some(want) {
                out.push(n);
            }
            stack.extend(n.children.iter());
        }
        out
    }

    #[test]
    fn moved_float_keeps_child_x_absolute() {
        // Issue 1: a float's children already carry absolute x (only y is
        // float-local). Moving the float whole must shift y only — adding
        // the float's x again doubles it (probed: x=20 -> x=40).
        let (doc, pages) = frag_doc(
            r#"<div id="s"></div><div id="wrap"><div id="fl"><div id="in"></div></div></div>"#,
            "body { margin: 0; } #s { height: 300px; } #wrap { padding-left: 20px; } \
             #fl { float: left; width: 100px; height: 150px; } #in { height: 50px; }",
            400.0,
        );
        assert_eq!(pages.len(), 2);
        let fl = find(&pages[1], &doc, "fl").expect("float whole on page 2");
        assert!(approx(fl.rect.x, 20.0), "float x={}", fl.rect.x);
        let inner = find(&pages[1], &doc, "in").expect("child on page 2");
        assert!(
            approx(inner.rect.x, fl.rect.x),
            "child x={} must equal float x={} (not doubled)",
            inner.rect.x,
            fl.rect.x
        );
    }

    #[test]
    fn moved_float_child_uses_premove_origin() {
        // Issue 2: split_subtree rigid-clones the moved float by dy, so
        // absolutize must add the pre-move origin (w.node.rect.y), not the
        // post-move one — the old code was off by exactly dy (probed:
        // child at y=-350 instead of 50).
        let (doc, pages) = frag_doc(
            r#"<div id="s"></div><div id="fl"><div id="pad"></div><div id="in"></div></div>"#,
            "body { margin: 0; } #s { height: 300px; } \
             #fl { float: left; width: 100px; height: 150px; } \
             #pad { height: 50px; } #in { height: 50px; }",
            400.0,
        );
        assert_eq!(pages.len(), 2);
        let fl = find(&pages[1], &doc, "fl").expect("float whole on page 2");
        assert!(approx(fl.rect.y, 0.0), "float y={}", fl.rect.y);
        let inner = find(&pages[1], &doc, "in").expect("child on page 2");
        assert!(
            approx(inner.rect.y, 50.0),
            "child y={}: want float-local 50 at the page top",
            inner.rect.y
        );
    }

    #[test]
    fn moved_float_nested_float_rebases_once() {
        // Issue 3: a float nested in a moved-whole float must get the
        // outer origin exactly once. The old shift_tree-per-child walk
        // applied the outer origin twice to nested floats' descendants.
        let (doc, pages) = frag_doc(
            r#"<div id="s"></div><div id="fl"><div id="pad"></div><div id="g"><div id="in"></div></div></div>"#,
            "body { margin: 0; } #s { height: 300px; } \
             #fl { float: left; width: 200px; height: 300px; } \
             #pad { height: 100px; } #g { float: left; width: 100px; height: 100px; } \
             #in { height: 40px; }",
            400.0,
        );
        assert_eq!(pages.len(), 2);
        let fl = find(&pages[1], &doc, "fl").expect("outer float on page 2");
        assert!(approx(fl.rect.y, 0.0), "outer y={}", fl.rect.y);
        // #g sits at float-local y=100 (below #pad); #in is #g's first
        // child, so both land at page y=100.
        let g = find(&pages[1], &doc, "g").expect("nested float on page 2");
        assert!(approx(g.rect.y, 100.0), "nested float y={}", g.rect.y);
        let inner = find(&pages[1], &doc, "in").expect("inner child on page 2");
        assert!(
            approx(inner.rect.y, 100.0),
            "inner y={}: want 100, not double-shifted",
            inner.rect.y
        );
    }

    #[test]
    fn float_nested_in_moved_avoid_box_is_absolutized() {
        // Issue 5: absolutize must cover floats nested inside any
        // moved-whole box, not just a root float. #outer
        // (break-inside: avoid) straddles and moves whole; without the
        // fix the float inside keeps float-local children in the page
        // tree.
        let (doc, pages) = frag_with_break(
            r#"<div id="s"></div><div id="outer"><div id="fl"><div id="pad"></div><div id="in"></div></div></div>"#,
            "body { margin: 0; } #s { height: 300px; } #outer { height: 200px; } \
             #fl { float: left; width: 100px; height: 150px; } \
             #pad { height: 30px; } #in { height: 50px; }",
            400.0,
            "outer",
            hints(Break::Auto, Break::Auto, BreakInside::Avoid),
        );
        assert_eq!(pages.len(), 2);
        let outer = find(&pages[1], &doc, "outer").expect("outer on page 2");
        assert!(approx(outer.rect.y, 0.0), "outer y={}", outer.rect.y);
        let fl = find(&pages[1], &doc, "fl").expect("float on page 2");
        assert!(approx(fl.rect.y, 0.0), "float y={}", fl.rect.y);
        let inner = find(&pages[1], &doc, "in").expect("child on page 2");
        assert!(
            approx(inner.rect.y, 30.0),
            "child y={}: want float-local 30, not stale",
            inner.rect.y
        );
    }

    #[test]
    fn float_nested_in_fixed_subtree_is_absolutized() {
        // Issue 5, fixed path: a float nested inside a position: fixed
        // subtree (repeated whole on every page) must be absolutized too.
        let (doc, pages) = frag_doc(
            r#"<div id="f"><div id="fl"><div id="pad"></div><div id="in"></div></div></div><div id="c"></div>"#,
            "body { margin: 0; } #f { position: fixed; top: 10px; left: 0; width: 200px; height: 200px; } \
             #fl { float: left; width: 100px; height: 150px; } \
             #pad { height: 30px; } #in { height: 50px; } #c { height: 700px; }",
            400.0,
        );
        assert_eq!(pages.len(), 2);
        for (i, page) in pages.iter().enumerate() {
            let fl = find(page, &doc, "fl").expect("float in fixed subtree");
            let inner = find(page, &doc, "in").expect("child in fixed subtree");
            assert!(
                approx(inner.rect.y, fl.rect.y + 30.0),
                "page {i}: child y={} float y={}",
                inner.rect.y,
                fl.rect.y
            );
        }
    }

    #[test]
    fn collect_line_boxes_skips_float_and_abs_pos_lines() {
        // Issue 4: the gate must mirror split_subtree's — lines inside a
        // never-split float or abs-pos box are not part of the unit's
        // fragmentable line flow.
        let html = r#"<div id="u">aa aa <div id="fl">bb bb</div> cc cc <div id="ap">dd dd</div> ee ee</div>"#;
        let css = "body { margin: 0; } \
                   #u { position: relative; font-size: 16px; line-height: 1.2; width: 20px; } \
                   #fl { float: left; width: 20px; } \
                   #ap { position: absolute; top: 0; left: 0; width: 20px; }";
        let (doc, _) = Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
        let u = find_in(&root, &doc, "u").expect("unit box");
        // Sanity: the float and the abs-pos box really are inside #u.
        assert!(find_in(u, &doc, "fl").is_some(), "float inside unit");
        assert!(find_in(u, &doc, "ap").is_some(), "abs-pos inside unit");
        let lines = collect_line_boxes(u, &HashMap::new(), 400.0);
        // 6 fragmentable lines (aa aa / cc cc / ee ee, one word per
        // line); the float's 2 and the abs-pos box's 2 move whole and
        // must not count.
        assert_eq!(lines.len(), 6, "lines={lines:?}");
    }

    #[test]
    fn nested_break_extents_offset_float_local_y() {
        // Issue 6: float-local y must not pollute ancestors'
        // sub_min_top (it only defeated band pruning; emission was
        // already correct).
        let html = r#"<div id="s"></div><div id="u"><div id="fl"><div id="in"></div></div></div>"#;
        let css = "body { margin: 0; } #s { height: 700px; } \
                   #fl { float: left; width: 100px; height: 100px; } #in { height: 50px; }";
        let (doc, _) = Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
        let nested = collect_nested_breaks(&[(&root, 0.0)], 400.0, &HashMap::new());
        let u = find_in(&root, &doc, "u").expect("unit box");
        let e = nested.get(&(u as *const LayoutBox)).expect("unit entry");
        // Without the float-origin offset, #in's float-local y (~0..50)
        // drags the unit's sub_min_top toward ~0; with it, the unit's
        // extent starts at its own top (~700).
        assert!(
            approx(e.sub_min_top, 700.0),
            "sub_min_top={}",
            e.sub_min_top
        );
    }

    #[test]
    fn thead_repetition_skipped_when_it_would_spill() {
        // Issue 7a: shifting the continued rows down by the header height
        // must not push content past the page bottom — repetition is
        // skipped for that page instead of overflowing it.
        // thead 0..50, one 380px row straddling 400 -> whole to page 2 at
        // 0..380; 380 + 50 > 400, so no repetition on page 2.
        let html = concat!(
            r#"<table id="t"><thead id="th"><tr><td><div id="hh"></div></td></tr></thead><tbody>"#,
            r#"<tr id="r1"><td><div id="d1"></div></td></tr>"#,
            r#"</tbody></table>"#,
        );
        let css = "body { margin: 0; } #hh { height: 50px; } #d1 { height: 380px; }";
        let (doc, pages) = frag_doc(html, css, 400.0);
        assert_eq!(pages.len(), 2);
        assert!(find(&pages[0], &doc, "th").is_some(), "header on page 1");
        let r1 = find(&pages[1], &doc, "r1").expect("r1 whole on page 2");
        assert!(approx(r1.rect.y, 0.0), "r1 y={} (unshifted)", r1.rect.y);
        // No visible thead on page 2: repetition was skipped, not
        // overflowed.
        assert!(
            find_all(&pages[1], &doc, "th")
                .iter()
                .all(|th| th.rect.h < EPS),
            "no repeated header on page 2"
        );
    }

    #[test]
    fn thead_repeats_with_multiple_tbodies() {
        // Issue 7c: thead repetition across several <tbody> sections.
        // thead 0..40 + five 100px rows = 540px over 400px pages; b2
        // (340..440) straddles and moves whole to page 2.
        let html = concat!(
            r#"<table id="t"><thead id="th"><tr><td><div id="hh"></div></td></tr></thead>"#,
            r#"<tbody><tr id="a1"><td><div id="da1"></div></td></tr>"#,
            r#"<tr id="a2"><td><div id="da2"></div></td></tr></tbody>"#,
            r#"<tbody><tr id="b1"><td><div id="db1"></div></td></tr>"#,
            r#"<tr id="b2"><td><div id="db2"></div></td></tr>"#,
            r#"<tr id="b3"><td><div id="db3"></div></td></tr></tbody></table>"#,
        );
        let css = "body { margin: 0; } #hh { height: 40px; } \
                   #da1,#da2,#db1,#db2,#db3 { height: 100px; }";
        let (doc, pages) = frag_doc(html, css, 400.0);
        assert_eq!(pages.len(), 2);
        let th1 = find(&pages[1], &doc, "th").expect("thead repeated on page 2");
        assert!(approx(th1.rect.y, 0.0), "y={}", th1.rect.y);
        assert!(approx(th1.rect.h, 40.0), "h={}", th1.rect.h);
        let b2 = find(&pages[1], &doc, "b2").expect("b2 on page 2");
        assert!(approx(b2.rect.y, 40.0), "y={}", b2.rect.y);
        let b3 = find(&pages[1], &doc, "b3").expect("b3 on page 2");
        assert!(approx(b3.rect.y, 80.0), "y={}", b3.rect.y);
    }

    #[test]
    fn thead_repeats_when_group_present_but_rowless() {
        // Issue 7b: a `position: fixed` box nested in the thead carries
        // viewport-coordinate extents, so the thead group passes its band
        // test (is "present") on pages its row never reaches. Dedup must
        // not treat a rowless group as a visible header.
        // Spacer 350 + thead 350..390 (40px) + fixed div at viewport
        // 850..950 nested in the thead cell + six 100px rows (390..990).
        // On page 3 (band 800..1200) the thead group is present via the
        // fixed box's extents but emits no row; r5/r6 continue there, so
        // the header must repeat.
        let html = concat!(
            r#"<div id="sp"></div><table id="t"><thead id="th"><tr><td>"#,
            r#"<div id="hh"></div><div id="fx"></div></td></tr></thead><tbody>"#,
            r#"<tr id="r1"><td><div id="d1"></div></td></tr>"#,
            r#"<tr id="r2"><td><div id="d2"></div></td></tr>"#,
            r#"<tr id="r3"><td><div id="d3"></div></td></tr>"#,
            r#"<tr id="r4"><td><div id="d4"></div></td></tr>"#,
            r#"<tr id="r5"><td><div id="d5"></div></td></tr>"#,
            r#"<tr id="r6"><td><div id="d6"></div></td></tr>"#,
            r#"</tbody></table>"#,
        );
        let css = "body { margin: 0; } #sp { height: 350px; } #hh { height: 40px; } \
                   #fx { position: fixed; top: 850px; height: 100px; } \
                   #d1,#d2,#d3,#d4,#d5,#d6 { height: 100px; }";
        let (doc, pages) = frag_doc(html, css, 400.0);
        assert_eq!(pages.len(), 3);
        // Page 3: the header repeats even though the thead group is
        // "present" (rowless) in the slice. Two thead groups exist: the
        // repeated head at y=0 with the visible row, and the rowless
        // carrier (shifted down with the flow) holding the fixed box.
        let heads = find_all(&pages[2], &doc, "th");
        assert_eq!(heads.len(), 2, "repeated head + rowless carrier");
        let head = heads
            .iter()
            .find(|h| approx(h.rect.y, 0.0))
            .expect("repeated head at page top");
        assert!(approx(head.rect.h, 40.0), "h={}", head.rect.h);
        assert!(
            head.children.iter().any(|r| r.rect.h > EPS),
            "repeated head carries the visible row"
        );
        let r5 = find(&pages[2], &doc, "r5").expect("r5 on page 3");
        assert!(approx(r5.rect.y, 40.0), "y={}", r5.rect.y);
        let r6 = find(&pages[2], &doc, "r6").expect("r6 on page 3");
        assert!(approx(r6.rect.y, 130.0), "y={}", r6.rect.y);
    }

    #[test]
    fn tfoot_repeats_on_continuation_pages() {
        // thead 0..40 + rows (r1..r3 100px, r4 150px, r5..r6 100px,
        // r7 70px, r8 150px, r9 100px) + tfoot 40px = 1050px over 400px
        // pages. r4 straddles onto page 2 whole, r8 onto page 3 whole.
        // Page 2 is a true middle page: the table continues past both its
        // boundaries, so the repeated footer must sit at the page bottom
        // without shifting the continued rows.
        let html = concat!(
            r#"<table id="t"><thead id="th"><tr><td><div id="hh"></div></td></tr></thead><tbody>"#,
            r#"<tr id="r1"><td><div id="d1"></div></td></tr>"#,
            r#"<tr id="r2"><td><div id="d2"></div></td></tr>"#,
            r#"<tr id="r3"><td><div id="d3"></div></td></tr>"#,
            r#"<tr id="r4"><td><div id="d4"></div></td></tr>"#,
            r#"<tr id="r5"><td><div id="d5"></div></td></tr>"#,
            r#"<tr id="r6"><td><div id="d6"></div></td></tr>"#,
            r#"<tr id="r7"><td><div id="d7"></div></td></tr>"#,
            r#"<tr id="r8"><td><div id="d8"></div></td></tr>"#,
            r#"<tr id="r9"><td><div id="d9"></div></td></tr>"#,
            r#"</tbody><tfoot id="tf"><tr><td><div id="ff"></div></td></tr></tfoot></table>"#,
        );
        let css = "body { margin: 0; } #hh { height: 40px; } \
                   #d1,#d2,#d3,#d5,#d6,#d9 { height: 100px; } \
                   #d4,#d8 { height: 150px; } #d7 { height: 70px; } \
                   #ff { height: 40px; }";
        let (doc, pages) = frag_doc(html, css, 400.0);
        assert_eq!(pages.len(), 3);
        // Page 1: header natural, no repeated footer (first page).
        let th0 = find(&pages[0], &doc, "th").expect("thead on page 1");
        assert!(approx(th0.rect.y, 0.0), "y={}", th0.rect.y);
        assert!(find(&pages[0], &doc, "tf").is_none(), "no tfoot on page 1");
        // Page 2 (middle): the footer repeats whole at the page bottom and
        // the continued rows stay exactly where the split put them.
        let tf1 = find(&pages[1], &doc, "tf").expect("tfoot repeated on page 2");
        assert!(approx(tf1.rect.y, 360.0), "y={}", tf1.rect.y);
        assert!(approx(tf1.rect.h, 40.0), "h={}", tf1.rect.h);
        assert!(
            tf1.children.iter().any(|r| r.rect.h > EPS),
            "repeated footer carries the visible row"
        );
        let r6 = find(&pages[1], &doc, "r6").expect("r6 on page 2");
        assert!(approx(r6.rect.y, 190.0), "y={} (unshifted)", r6.rect.y);
        let r7 = find(&pages[1], &doc, "r7").expect("r7 on page 2");
        assert!(approx(r7.rect.y, 290.0), "y={} (unshifted)", r7.rect.y);
        // Page 3 (last): the header repeats at the top (shifting content
        // down, thead behavior) and the footer shows once, naturally.
        let th2 = find(&pages[2], &doc, "th").expect("thead repeated on page 3");
        assert!(approx(th2.rect.y, 0.0), "y={}", th2.rect.y);
        let tf2 = find(&pages[2], &doc, "tf").expect("natural tfoot on page 3");
        assert!(approx(tf2.rect.y, 250.0), "y={}", tf2.rect.y);
        assert!(approx(tf2.rect.h, 40.0), "h={}", tf2.rect.h);
        assert_eq!(
            find_all(&pages[2], &doc, "tf").len(),
            1,
            "tfoot appears exactly once on the last page"
        );
    }

    #[test]
    fn thead_and_tfoot_combined() {
        // thead 40px + r1..r4 150px rows + tfoot 40px = 680px over 400px
        // pages: a two-page table exercising the thead+tfoot interplay.
        // r3 straddles onto page 2 whole. On page 2 the repeated thead
        // shifts the continued rows (and the natural footer) down, and the
        // natural footer must not be duplicated.
        let html = concat!(
            r#"<table id="t"><thead id="th"><tr><td><div id="hh"></div></td></tr></thead><tbody>"#,
            r#"<tr id="r1"><td><div id="d1"></div></td></tr>"#,
            r#"<tr id="r2"><td><div id="d2"></div></td></tr>"#,
            r#"<tr id="r3"><td><div id="d3"></div></td></tr>"#,
            r#"<tr id="r4"><td><div id="d4"></div></td></tr>"#,
            r#"</tbody><tfoot id="tf"><tr><td><div id="ff"></div></td></tr></tfoot></table>"#,
        );
        let css = "body { margin: 0; } #hh { height: 40px; } \
                   #d1,#d2,#d3,#d4 { height: 150px; } #ff { height: 40px; }";
        let (doc, pages) = frag_doc(html, css, 400.0);
        assert_eq!(pages.len(), 2);
        let th0 = find(&pages[0], &doc, "th").expect("thead on page 1");
        assert!(approx(th0.rect.y, 0.0), "y={}", th0.rect.y);
        assert!(find(&pages[0], &doc, "tf").is_none(), "no tfoot on page 1");
        // Page 2: repeated header at the top...
        let th1 = find(&pages[1], &doc, "th").expect("thead repeated on page 2");
        assert!(approx(th1.rect.y, 0.0), "y={}", th1.rect.y);
        assert_eq!(
            find_all(&pages[1], &doc, "th").len(),
            1,
            "thead appears exactly once"
        );
        // ...continued rows shifted below it...
        let r3 = find(&pages[1], &doc, "r3").expect("r3 on page 2");
        assert!(approx(r3.rect.y, 40.0), "y={}", r3.rect.y);
        // ...and the natural footer exactly once, below the rows.
        let tf1 = find(&pages[1], &doc, "tf").expect("natural tfoot on page 2");
        assert!(approx(tf1.rect.y, 280.0), "y={}", tf1.rect.y);
        assert!(approx(tf1.rect.h, 40.0), "h={}", tf1.rect.h);
        assert_eq!(
            find_all(&pages[1], &doc, "tf").len(),
            1,
            "tfoot appears exactly once"
        );
    }

    #[test]
    fn tfoot_repetition_skipped_when_rows_fill_page() {
        // Mirror of thead_repetition_skipped_when_it_would_spill: the
        // repeated footer must not overlap the continued rows. Ten 100px
        // rows + 40px tfoot = 1040px over 400px pages; page 2's rows fill
        // the page (0..400), so repeating the footer there would overlap
        // r8 — repetition is skipped for that page instead.
        let mut html = String::from(r#"<table id="t"><tbody>"#);
        for i in 1..=10 {
            html.push_str(&format!(
                r#"<tr id="r{i}"><td><div id="d{i}"></div></td></tr>"#
            ));
        }
        html.push_str(
            r#"</tbody><tfoot id="tf"><tr><td><div id="ff"></div></td></tr></tfoot></table>"#,
        );
        let mut css = String::from("body { margin: 0; } #ff { height: 40px; } ");
        for i in 1..=10 {
            css.push_str(&format!("#d{i} {{ height: 100px; }} "));
        }
        let (doc, pages) = frag_doc(&html, &css, 400.0);
        assert_eq!(pages.len(), 3);
        assert!(
            find(&pages[1], &doc, "tf").is_none(),
            "no repeated footer on page 2 (overlap bound)"
        );
        let r8 = find(&pages[1], &doc, "r8").expect("r8 on page 2");
        assert!(approx(r8.rect.y, 300.0), "y={} (unshifted)", r8.rect.y);
        // Page 3 shows the footer once, naturally.
        let tf2 = find(&pages[2], &doc, "tf").expect("natural tfoot on page 3");
        assert!(approx(tf2.rect.y, 200.0), "y={}", tf2.rect.y);
        assert_eq!(
            find_all(&pages[2], &doc, "tf").len(),
            1,
            "tfoot appears exactly once"
        );
    }

    #[test]
    fn tfoot_taller_than_page_is_skipped() {
        // A footer taller than the page can never fit: repetition is
        // skipped (bounded — no retry loop), and the natural footer still
        // fragments geometrically across the pages it spans. Eight 100px
        // rows + one 500px footer row = 1300px over 400px pages.
        let mut html = String::from(r#"<table id="t"><tbody>"#);
        for i in 1..=8 {
            html.push_str(&format!(
                r#"<tr id="r{i}"><td><div id="d{i}"></div></td></tr>"#
            ));
        }
        html.push_str(
            r#"</tbody><tfoot id="tf"><tr><td><div id="ff"></div></td></tr></tfoot></table>"#,
        );
        let mut css = String::from("body { margin: 0; } #ff { height: 500px; } ");
        for i in 1..=8 {
            css.push_str(&format!("#d{i} {{ height: 100px; }} "));
        }
        let (doc, pages) = frag_doc(&html, &css, 400.0);
        assert_eq!(pages.len(), 4);
        assert!(
            find(&pages[1], &doc, "tf").is_none(),
            "oversized footer not repeated on page 2"
        );
        let tf2 = find(&pages[2], &doc, "tf").expect("footer fragment on page 3");
        assert!(approx(tf2.rect.y, 0.0), "y={}", tf2.rect.y);
        assert!(approx(tf2.rect.h, 400.0), "h={}", tf2.rect.h);
        let tf3 = find(&pages[3], &doc, "tf").expect("footer fragment on page 4");
        assert!(approx(tf3.rect.y, 0.0), "y={}", tf3.rect.y);
        assert!(approx(tf3.rect.h, 100.0), "h={}", tf3.rect.h);
    }

    #[test]
    fn tfoot_empty_group_not_repeated() {
        // An empty (zero-height) tfoot has nothing to repeat: no footer
        // clone may appear on any page.
        let mut html = String::from(r#"<table id="t"><tbody>"#);
        for i in 1..=6 {
            html.push_str(&format!(
                r#"<tr id="r{i}"><td><div id="d{i}"></div></td></tr>"#
            ));
        }
        html.push_str(r#"</tbody><tfoot id="tf"></tfoot></table>"#);
        let mut css = String::from("body { margin: 0; } ");
        for i in 1..=6 {
            css.push_str(&format!("#d{i} {{ height: 100px; }} "));
        }
        let (doc, pages) = frag_doc(&html, &css, 400.0);
        assert_eq!(pages.len(), 2);
        for (i, page) in pages.iter().enumerate() {
            for tf in find_all(page, &doc, "tf") {
                assert!(
                    tf.rect.h < EPS,
                    "page {i}: empty tfoot must not repeat visibly, h={}",
                    tf.rect.h
                );
            }
        }
    }

    #[test]
    fn tfoot_with_rowspan_repeats_whole() {
        // A tfoot whose cell spans both footer rows (rowspan=2) clones
        // whole: the spanning cell fits its 80px into the 40px + 40px
        // rows without growing them (spanning cells do not shift their
        // originating row's baseline), so the group is 80px, and the
        // repeated footer on the middle page carries both rows at that
        // full height, bottom-aligned to the page.
        let html = concat!(
            r#"<table id="t"><tbody>"#,
            r#"<tr id="r1"><td><div id="d1"></div></td></tr>"#,
            r#"<tr id="r2"><td><div id="d2"></div></td></tr>"#,
            r#"<tr id="r3"><td><div id="d3"></div></td></tr>"#,
            r#"<tr id="r4"><td><div id="d4"></div></td></tr>"#,
            r#"<tr id="r5"><td><div id="d5"></div></td></tr>"#,
            r#"<tr id="r6"><td><div id="d6"></div></td></tr>"#,
            r#"<tr id="r7"><td><div id="d7"></div></td></tr>"#,
            r#"<tr id="r8"><td><div id="d8"></div></td></tr>"#,
            r#"<tr id="r9"><td><div id="d9"></div></td></tr>"#,
            r#"</tbody><tfoot id="tf">"#,
            r#"<tr id="fr1"><td rowspan="2"><div id="ff1"></div></td><td><div id="ff2"></div></td></tr>"#,
            r#"<tr id="fr2"><td><div id="ff3"></div></td></tr>"#,
            r#"</tfoot></table>"#,
        );
        let css = "body { margin: 0; } \
                   #d1,#d2,#d3,#d5,#d9 { height: 100px; } \
                   #d4,#d8 { height: 150px; } #d6 { height: 60px; } \
                   #d7 { height: 70px; } \
                   #ff1 { height: 80px; } #ff2,#ff3 { height: 40px; }";
        let (doc, pages) = frag_doc(html, css, 400.0);
        assert_eq!(pages.len(), 3);
        // Page 2 (middle): rows end at y=280, so the 120px footer fits at
        // the page bottom carrying both spanned rows.
        let tf1 = find(&pages[1], &doc, "tf").expect("tfoot repeated on page 2");
        assert!(approx(tf1.rect.y, 320.0), "y={}", tf1.rect.y);
        assert!(approx(tf1.rect.h, 80.0), "h={}", tf1.rect.h);
        let fr1 = find(&pages[1], &doc, "fr1").expect("footer row 1 repeated");
        assert!(fr1.rect.h > EPS, "footer row 1 visible");
        let fr2 = find(&pages[1], &doc, "fr2").expect("footer row 2 repeated");
        assert!(fr2.rect.h > EPS, "footer row 2 visible");
        // Page 3 (last): the footer shows once, naturally.
        let tf2 = find(&pages[2], &doc, "tf").expect("natural tfoot on page 3");
        assert!(approx(tf2.rect.y, 130.0), "y={}", tf2.rect.y);
        assert_eq!(
            find_all(&pages[2], &doc, "tf").len(),
            1,
            "tfoot appears exactly once"
        );
    }

    #[test]
    fn multicol_orphans_counts_per_column_lines() {
        // orphans/widows line counting must not merge same-y fragments
        // from different columns into one "line": the page-1 fragment
        // here holds one y-band = two lines (one per column), satisfying
        // the default orphans: 2, so the container must stay at 390, not
        // be shoved whole to the next page leaving a gap.
        let html = format!(
            "<div id=\"pad\"></div><div id=\"mc\">{}</div>",
            "word ".repeat(200)
        );
        let (doc, pages) = frag_doc(
            &html,
            "body { margin: 0; } #pad { height: 390px; } \
             #mc { column-count: 2; column-gap: 20px; width: 420px; font-size: 10px; }",
            400.0,
        );
        assert_eq!(pages.len(), 2);
        let mc0 = find(&pages[0], &doc, "mc").expect("mc starts on page 1");
        assert!(
            approx(mc0.rect.y, 390.0),
            "no spurious orphan push, y={}",
            mc0.rect.y
        );
        assert!(
            find(&pages[1], &doc, "mc").is_some(),
            "mc continues on page 2"
        );
    }

    #[test]
    fn flex_item_straddling_page_moves_whole() {
        // The concrete row-fit case: a 600px-tall flex item starting 400px
        // before the page bottom must move whole to page 2 (it fits on one
        // page), not slice 400/200 with its background/border cut across
        // pages. Flex items ride the unbreakable fast-path, mirroring
        // table rows.
        let (doc, pages) = frag_doc(
            r#"<div id="pad"></div><div id="flex"><div id="item"></div></div>"#,
            "body { margin: 0; } #pad { height: 600px; } \
             #flex { display: flex; } #item { height: 600px; width: 200px; }",
            1000.0,
        );
        assert_eq!(pages.len(), 2, "item moves whole to page 2");
        assert!(
            find(&pages[0], &doc, "item").is_none(),
            "no mid-box slice of the item on page 1"
        );
        let item = find(&pages[1], &doc, "item").expect("item whole on page 2");
        assert!(
            approx(item.rect.y, 0.0),
            "item at page 2 top, y={}",
            item.rect.y
        );
        assert!(
            approx(item.rect.h, 600.0),
            "item keeps its full height, h={}",
            item.rect.h
        );
    }

    #[test]
    fn grid_item_straddling_page_moves_whole() {
        // Grid rows are logical (no row boxes in the tree); the grid item
        // — the row member — is the atomic unit, mirroring table rows.
        let (doc, pages) = frag_doc(
            r#"<div id="pad"></div><div id="grid"><div id="gitem"></div></div>"#,
            "body { margin: 0; } #pad { height: 600px; } \
             #grid { display: grid; } #gitem { height: 600px; }",
            1000.0,
        );
        assert_eq!(pages.len(), 2, "grid item moves whole to page 2");
        assert!(
            find(&pages[0], &doc, "gitem").is_none(),
            "no mid-box slice of the grid item on page 1"
        );
        let gitem = find(&pages[1], &doc, "gitem").expect("grid item whole on page 2");
        assert!(
            approx(gitem.rect.y, 0.0),
            "grid item at page 2 top, y={}",
            gitem.rect.y
        );
        assert!(
            approx(gitem.rect.h, 600.0),
            "grid item keeps its full height, h={}",
            gitem.rect.h
        );
    }

    #[test]
    fn flex_body_flow_unit_straddling_moves_whole() {
        // The split root's `item` flag comes from the caller: a flow unit
        // that is itself a flex item (`display: flex` on <body>) rides the
        // unbreakable fast-path at the root instead of fragmenting
        // mid-box. (Nested items already got the flag from their parent;
        // the root hardcoded `false` before this fix.)
        let (doc, pages) = frag_doc(
            r#"<div id="pad"></div><div id="item"></div>"#,
            "body { margin: 0; display: flex; flex-direction: column; } \
             #pad { height: 600px; } #item { height: 600px; width: 200px; }",
            1000.0,
        );
        assert_eq!(pages.len(), 2, "item moves whole to page 2");
        assert!(
            find(&pages[0], &doc, "item").is_none(),
            "no mid-box slice of the item on page 1"
        );
        let item = find(&pages[1], &doc, "item").expect("item whole on page 2");
        assert!(
            approx(item.rect.y, 0.0),
            "item at page 2 top, y={}",
            item.rect.y
        );
        assert!(
            approx(item.rect.h, 600.0),
            "item keeps its full height, h={}",
            item.rect.h
        );
    }

    #[test]
    fn grid_row_tearing_straddling_item_moves_whole_rowmate_stays() {
        // Grid rows are logical (no row boxes in the tree): each grid item
        // is an independent unbreakable unit. A straddling item moves whole
        // to the next page while its row-mate stays behind — the visual
        // row tears. This is intended: items never split mid-box when they
        // fit on a page, mirroring table rows and flex items.
        let (doc, pages) = frag_doc(
            r#"<div id="pad"></div><div id="grid"><div id="a"></div><div id="b"></div></div>"#,
            "body { margin: 0; } #pad { height: 600px; } \
             #grid { display: grid; grid-template-columns: 200px 200px; } \
             #a { height: 600px; } #b { height: 300px; align-self: start; }",
            1000.0,
        );
        // #a spans 600..1200 in column 1, sharing row 1 with #b: it
        // straddles the page boundary, fits on a page, and moves whole to
        // page 2; #b (600..900) stays on page 1.
        assert_eq!(pages.len(), 2, "straddler moves whole to page 2");
        let b = find(&pages[0], &doc, "b").expect("row-mate stays on page 1");
        assert!(approx(b.rect.y, 600.0), "b y={}", b.rect.y);
        assert!(approx(b.rect.x, 200.0), "b in column 2, x={}", b.rect.x);
        assert!(
            find(&pages[0], &doc, "a").is_none(),
            "no mid-box slice of the straddler on page 1"
        );
        let a = find(&pages[1], &doc, "a").expect("straddler whole on page 2");
        assert!(approx(a.rect.y, 0.0), "a at page 2 top, y={}", a.rect.y);
        assert!(
            approx(a.rect.h, 600.0),
            "a keeps its full height, h={}",
            a.rect.h
        );
        assert!(approx(a.rect.x, 0.0), "a in column 1, x={}", a.rect.x);
    }

    #[test]
    fn grid_row_spanning_item_straddling_moves_whole() {
        // A row-spanning item is still one atomic box: it moves whole even
        // though it covers two rows, while the items beside it stay.
        let (doc, pages) = frag_doc(
            r#"<div id="pad"></div><div id="grid"><div id="a"></div><div id="b"></div></div>"#,
            "body { margin: 0; } #pad { height: 600px; } \
             #grid { display: grid; grid-template-columns: 200px 200px; \
                     grid-template-rows: 300px 300px; } \
             #a { grid-row: 1 / 3; } #b { height: 200px; align-self: start; }",
            1000.0,
        );
        // #a spans rows 1-2 (600..1200): it straddles, fits on a page, and
        // moves whole to page 2; #b sits in row 1 / column 2 (600..800).
        assert_eq!(pages.len(), 2, "spanner moves whole to page 2");
        let b = find(&pages[0], &doc, "b").expect("row-1 item stays on page 1");
        assert!(approx(b.rect.y, 600.0), "b y={}", b.rect.y);
        assert!(approx(b.rect.x, 200.0), "b in column 2, x={}", b.rect.x);
        assert!(
            find(&pages[0], &doc, "a").is_none(),
            "no mid-box slice of the spanner on page 1"
        );
        let a = find(&pages[1], &doc, "a").expect("spanner whole on page 2");
        assert!(approx(a.rect.y, 0.0), "a at page 2 top, y={}", a.rect.y);
        assert!(
            approx(a.rect.h, 600.0),
            "a keeps its full spanned height, h={}",
            a.rect.h
        );
    }

    #[test]
    fn inline_grid_straddling_item_moves_whole() {
        // `inline-grid` items get the same item-hood as `grid` items:
        // `is_flex_or_grid_container` consults the raw `display` keyword,
        // so the item rides the unbreakable fast-path.
        let (doc, pages) = frag_doc(
            r#"<div id="pad"></div><div id="grid"><div id="a"></div></div>"#,
            "body { margin: 0; } #pad { height: 600px; } \
             #grid { display: inline-grid; grid-template-columns: 200px; } \
             #a { height: 600px; width: 200px; }",
            1000.0,
        );
        assert_eq!(pages.len(), 2, "item moves whole to page 2");
        assert!(
            find(&pages[0], &doc, "a").is_none(),
            "no mid-box slice of the item on page 1"
        );
        let a = find(&pages[1], &doc, "a").expect("item whole on page 2");
        assert!(approx(a.rect.y, 0.0), "a at page 2 top, y={}", a.rect.y);
        assert!(
            approx(a.rect.h, 600.0),
            "a keeps its full height, h={}",
            a.rect.h
        );
    }

    #[test]
    fn flex_item_taller_than_page_still_fragments() {
        // The unbreakable gate requires h <= page_h: a 1400px flex item on
        // 1000px pages must still fragment rather than vanish or pin.
        let (doc, pages) = frag_doc(
            r#"<div id="flex"><div id="item"></div></div>"#,
            "body { margin: 0; } #flex { display: flex; } \
             #item { height: 1400px; width: 200px; }",
            1000.0,
        );
        assert_eq!(pages.len(), 2);
        assert!(
            find(&pages[0], &doc, "item").is_some(),
            "tall item fragments onto page 1"
        );
        assert!(
            find(&pages[1], &doc, "item").is_some(),
            "tall item continues on page 2"
        );
    }

    #[test]
    fn flex_item_within_page_stays_on_page() {
        // An item that fits entirely on page 1 must not be moved.
        let (doc, pages) = frag_doc(
            r#"<div id="flex"><div id="item"></div></div>"#,
            "body { margin: 0; } #flex { display: flex; } \
             #item { height: 300px; width: 200px; }",
            1000.0,
        );
        assert_eq!(pages.len(), 1);
        let item = find(&pages[0], &doc, "item").expect("item on page 1");
        assert!(approx(item.rect.y, 0.0), "y={}", item.rect.y);
        assert!(approx(item.rect.h, 300.0), "h={}", item.rect.h);
    }

    #[test]
    fn flex_item_lines_excluded_from_orphan_count() {
        // `collect_line_boxes` mirrors `split_subtree`'s unbreakable gate:
        // lines inside a flex item that moves whole to the next page must
        // not count toward the unit's orphans budget, or the moved item's
        // lines would shove the whole container down a page through
        // orphans, fighting the move-whole behavior.
        let words = "word ".repeat(15);
        let html = format!(
            r#"<div id="pad"></div><div id="flex"><div id="a"></div><div id="b">{words}</div></div>"#
        );
        let (doc, pages) = frag_doc(
            &html,
            "body { margin: 0; } #pad { height: 600px; } \
             #flex { display: flex; flex-direction: column; } \
             #a { height: 380px; width: 200px; } \
             #b { width: 40px; font-size: 16px; line-height: 20px; }",
            1000.0,
        );
        // #b lays out 15 lines x 20px = 300px tall at y 980..1280: it
        // straddles, fits on a page, and moves whole to page 2. Only one
        // of its lines would sit on page 1 (< orphans: 2), so without the
        // line-flow exclusion the container is shoved whole to page 2.
        let a = find(&pages[0], &doc, "a").expect("a on page 1");
        assert!(
            approx(a.rect.y, 600.0),
            "no spurious orphan push, y={}",
            a.rect.y
        );
        assert!(
            find(&pages[0], &doc, "b").is_none(),
            "b moved whole to page 2"
        );
        let b = find(&pages[1], &doc, "b").expect("b on page 2");
        assert!(approx(b.rect.y, 0.0), "b at page 2 top, y={}", b.rect.y);
        assert!(
            approx(b.rect.h, 300.0),
            "b keeps full height, h={}",
            b.rect.h
        );
    }
}
