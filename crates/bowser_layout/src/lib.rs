//! Block, inline, table, and flex layout for Bowser.
//!
//! `layout()` turns a `bowser_dom::Document` plus author stylesheets into a
//! tree of [`LayoutBox`]es: block layout (width resolution, margin
//! collapsing, auto heights) and inline layout (line breaking with
//! [`FontMetrics`], baseline-ish vertical alignment).
//!
//! Coordinate system: every [`LayoutBox::rect`] is a **border box** in
//! absolute page coordinates; the origin is the top-left of the initial
//! containing block (the viewport origin).
//!
//! # Assumptions about the CSS contract
//!
//! This crate codes against the `bowser_css` contract, not its source:
//! - `Styler::new(doc, author, ua, viewport)` plus `Styler::style_for(node)`
//!   returns a [`bowser_css::ComputedStyle`] with `get`/`get_str`;
//!   `default_ua_stylesheet()` supplies UA defaults (notably `display` for
//!   HTML elements, with a tag-name fallback table here in case it reports
//!   nothing).
//! - `ComputedStyle` is shared by reference count
//!   (`Option<Rc<ComputedStyle>>`), threaded from the styler's memo
//!   through the box arena; it is `None` for anonymous and text boxes,
//!   which have no element style.
//! - Shorthands (`margin`, `padding`, `border-width`, `border`) are assumed
//!   expanded to longhands by the CSS implementation; only longhands
//!   (`margin-top`, `border-top-width`, …) are read.
//! - The layout [`Size`] viewport is converted to `bowser_css::Viewport`
//!   inside `style::compute_element_style`, so `@media` queries and
//!   `vw`/`vh` units see the real viewport size.
//!
//! # Style resolution
//!
//! All style lookups go through `style::compute_element_style` (and the one
//! `ua_stylesheet()` call in [`layout`]). The UA stylesheet is parsed once
//! per process and shared; one `Styler` per `layout()` call keeps
//! cascade/inheritance memoization alive across elements, so styling is
//! linear in document size.
//!
//! # Performance
//!
//! Styling is the dominant layout cost. Observed scaling is O(elements ×
//! rules), roughly 110µs per element in release builds; keep stylesheets
//! small and selectors cheap.
//!
//! # Paint/JS contract
//!
//! [`LayoutBox`] carries two fields for paint and the JS engine:
//! - `node`: the [`bowser_dom::NodeId`] the box was built from (`None`
//!   for anonymous boxes, which have no DOM node). Lets paint map boxes
//!   back to elements (e.g. replaced elements look up their intrinsic
//!   size by node).
//! - `fragments`: for [`BoxType::Text`] boxes, one [`TextFragment`] per
//!   laid-out line fragment, in line order; empty for every other box
//!   type. Each fragment's `rect` is the fragment's placed rect in
//!   absolute page coordinates and `baseline` is the line's baseline in
//!   absolute page coordinates, so paint can place glyphs per line
//!   without re-running line breaking.
//!
//! `text-transform` is applied during whitespace collapsing, **before**
//! line breaking, so breaking measures transformed advances and
//! [`LayoutBox::text`] holds the transformed text; paint renders it
//! verbatim (no second transform pass).
//!
//! # Replaced elements and intrinsics
//!
//! [`layout_with_intrinsics`] takes a map from DOM node to intrinsic
//! [`Size`]. Replaced elements (v1: `img`, `video`, `audio`, `canvas`,
//! `iframe`, `embed`, `object`, `input`) take their box size from the map
//! when CSS `width`/`height` are `auto`; specified CSS `width`/`height`
//! (and `min`/`max` clamps) still override. A replaced element with no map
//! entry sizes to 0x0 (documented v1 behavior).
//!
//! # Documented simplifications
//!
//! - Positioning: `position: relative` offsets the box without changing
//!   its flow footprint; `position: absolute` / `fixed` boxes are laid
//!   out out-of-flow against their containing block (nearest positioned
//!   ancestor's padding box, or the viewport), honoring
//!   `top`/`right`/`bottom`/`left` with static-position fallback for
//!   `auto` offsets. `position: sticky` lays out in-flow like `relative`
//!   and is then constrained against the scroll offset passed to
//!   [`layout_with_scroll`] (the viewport's) or, for sticky boxes inside
//!   a nested scroll container, that container's offset from
//!   [`layout_with_scroll_and_offsets`] (CSS Position 3 §3.1: stuck within
//!   the scrollport inset by its offsets, hard-bounded by its containing
//!   block); with a zero scroll offset it behaves exactly like `relative`.
//!   `z-index` is resolved onto `LayoutBox::z_index` (kept for static
//!   flex/grid items, whose non-`auto` z-index forms a stacking context
//!   per css-flexbox-1 / css-grid-1); paint emits CSS 2.1 Appendix E
//!   order with stacking contexts (positioned non-`auto` z-index,
//!   `opacity` < 1, non-`normal` blend mode). `float:
//!   left | right` places floats per CSS 2.1 §9.5.1 (per-side stacking,
//!   `clear`, BFC containment, line-box shortening around floats).
//! - `display: table` (and the table-internal `table-row-group`,
//!   `table-row`, `table-cell` values) get dedicated table layout: the
//!   table → row-group → row → cell structure, simplified automatic table
//!   layout (column widths from cell content minimums, distributed to fill
//!   the table width), cell padding/borders, `border-collapse: separate` /
//!   `collapse`, row heights from cell content, and basic `colspan` /
//!   `rowspan`. `display: grid` gets dedicated grid layout (explicit
//!   tracks from `grid-template-columns` / `grid-template-rows` with
//!   fixed, `fr`, and `repeat()` sizes, auto-placement, spanning,
//!   `gap`, and `justify-items` / `align-items`; see
//!   `layout::layout_grid` for the MVP simplifications).
//!   `display: list-item` (and the remaining `inline-*` variants) are
//!   treated as `block` / `inline-block`; the principal box of a list
//!   item is a plain block box, and the item additionally gets a marker
//!   box: `list-style-position: outside` hangs it in the inline-start
//!   margin area ([`MarkerBox`]), `inside` makes it the first inline
//!   box in the flow.
//!   `flex` / `inline-flex` get dedicated flexbox layout: single-line by
//!   default, with multi-line `flex-wrap: wrap | wrap-reverse` collecting
//!   flex lines and stacking them across the container's cross size per
//!   `align-content` (see `layout_flex_wrap`).
//! - An `inline` box containing block-level children is treated as a block
//!   box (CSS would split the inline box around the blocks).
//! - Replaced elements are atomic inline-level boxes, modeled as
//!   `inline-block` (no dedicated replaced layout).
//! - An `inline-block` aligns on the baseline of its last line box;
//!   blocks without line boxes align on their bottom border edge.
//! - Margin collapsing covers adjoining siblings, parent/first-child,
//!   parent/last-child, and empty blocks, including the negative-margin
//!   rule. No clearance, no block-formatting-context isolation.
//! - Inline-block shrink-to-fit is `min(preferred, available)`; the
//!   preferred width includes nested padding/border/margins: each
//!   in-flow child contributes its preferred content width plus its
//!   used horizontal padding/border/margins (via `intrinsic_h_extras`),
//!   and the box's own used padding/border/margins ride along the
//!   same way.
//! - Line boxes include the block's `line-height` as a strut (an empty
//!   inline box with the block's font), flooring the line-box height.
//! - `vertical-align: baseline` splits the line height across the baseline
//!   by the used face's ascent/descent ratio from [`FontMetrics`]
//!   (defaulting to the historical fixed 0.8/0.2 split when the
//!   implementation provides no face metrics); length values shift the
//!   box (positive raises it).
//! - `text-align: justify` spreads leftover space across word gaps on
//!   non-final lines (last line honors `text-align-last`); `start`/`end` map to
//!   `left`/`right` per the element's `direction`.
//! - `text-indent` offsets the first line box.
//! - `white-space: pre-wrap` preserves spaces verbatim, forces breaks on
//!   newlines, and soft-wraps at spaces (with a break-all fallback for
//!   overlong words); `pre-line` collapses space/tab runs to a single
//!   space (trimming line-boundary spaces), preserves newlines as forced
//!   breaks, and soft-wraps at spaces.
//! - Flex: auto margins on the main axis absorb positive free space before
//!   `justify-content` runs; cross-axis auto margins likewise absorb
//!   positive free space (split equally among them) before `align-self`
//!   runs, which is then skipped for that item — and an auto cross margin
//!   also suppresses the `stretch` fill for the item.
//! - Flex: `align-items: baseline` on the column axis behaves as
//!   `flex-start` (known simplification).
//! - Unitless non-zero numbers are treated as `px` (CSS only allows unitless
//!   zero, but leniency here is harmless).
//! - `rem` resolves against the root element's computed font size, falling
//!   back to 16px.
//!
//! # Stack safety
//!
//! Box construction, the layout driver, margin-collapse walks, fragment
//! building, preferred-width measurement, subtree shifts, and tree
//! conversion are all iterative. One known gap: the public
//! `children: Vec<LayoutBox>` shape means dropping an extremely deep
//! `LayoutBox` tree recurses; that shape is API-mandated.

mod boxes;
mod hash;
mod layout;
mod layout_fragmentation;
mod marker;
mod pseudo;
mod style;

/// Shared with `bowser_paint`: column-rule geometry recorded during
/// multicol layout; paint drains it once per layout pass via
/// [`take_column_rules`] and emits rule display items.
pub use layout::layout_multicol::{ColumnRuleSpec, ColumnRuleStyle, take_column_rules};
/// Shared with `bowser_js`: the bindings layer measures table offset
/// geometry (`offsetWidth`, `offsetParent`, ...) through this API.
pub use layout::layout_table::{
    TableBoxGeometry, box_used_borders, table_box_geometry, table_offset_parent,
};
/// Print/PDF fragmentation: [`layout_fragmented`] slices the single-page
/// tree into per-page [`PageFragment`]s honoring `break-before` /
/// `break-after` / `break-inside` (see [`BreakHints`]). It is a pure
/// post-pass: [`layout`] itself is untouched.
pub use layout_fragmentation::{
    Break, BreakHints, BreakInside, MAX_PAGES, PageFragment, layout_fragmented,
    layout_fragmented_with_breaks, layout_fragmented_with_intrinsics,
    layout_fragmented_with_scroll,
};
/// Shared with `bowser_paint`: paint's display-list builder resolves
/// `font-size` against the same bases layout used, so it calls this
/// rather than keeping a mirrored copy.
pub use style::resolve_font_size;
/// Shared with `bowser_paint`: paint's display-list builder clamps raw
/// resolved lengths through the same finite bound layout uses, so a
/// hostile `border-width: 1e30px` paints at the same 1e15px layout used.
pub use style::sane_len;
/// Shared with `bowser_app`: the scroll backend classifies boxes as
/// out-of-flow with the same `position` resolution layout uses, so the
/// paint-shift decisions cannot drift from the layout model.
pub use style::{Position, position_of};

#[cfg(test)]
mod tests;

/// Layout unit tests were authored when the UA stylesheet carried no table
/// `border-spacing`/`padding` defaults. Test helpers prepend this reset to
/// every test stylesheet so algorithm tests keep their exact-geometry
/// assertions; a test's own later (or more specific) rules still win.
/// The UA defaults themselves are covered by WPT, not by these tests.
#[cfg(test)]
pub(crate) const TABLE_TEST_RESET_CSS: &str = "table{border-spacing:0}td,th{padding:0}";

use bowser_css::{CssValue, LengthUnit, SnapAlign, Stylesheet, props};
use bowser_dom::{Document, NodeId};
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

/// A 2D size (used for the viewport argument).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Size {
    pub w: f32,
    pub h: f32,
}

/// Intrinsic dimensions of a replaced element, as decoded from the
/// resource (or synthesized per spec defaults).
///
/// Each component is `None` when unknown: a raster image always knows
/// its pixel size (the ratio derives from `w`/`h`); an SVG may know
/// only its aspect ratio (`viewBox` with no `width`/`height`); an
/// `<iframe>` knows none and falls back to the CSS 300x150 default.
/// All-`None` with no ratio is the missing-image case, which keeps the
/// v1 0x0 sizing for `<img>` with no decodable data (only
/// `<iframe>`/`<embed>`/`<object>`/`<video>` get the 300x150 default;
/// see `boxes.rs`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntrinsicDims {
    /// Intrinsic width in CSS px, when the resource declares one.
    pub w: Option<f32>,
    /// Intrinsic height in CSS px, when the resource declares one.
    pub h: Option<f32>,
    /// Intrinsic aspect ratio (width / height), when known
    /// independently of `w`/`h` — e.g. an SVG `viewBox` with no
    /// `width`/`height`. `None` derives the ratio from `w`/`h` when
    /// both are positive and finite.
    pub ratio: Option<f32>,
}

impl IntrinsicDims {
    /// Raster-image intrinsics: both dimensions known, ratio derived.
    pub fn from_size(w: f32, h: f32) -> Self {
        IntrinsicDims {
            w: Some(w),
            h: Some(h),
            ratio: None,
        }
    }

    /// No intrinsic information at all (unknown dimensions, no ratio).
    pub fn unknown() -> Self {
        IntrinsicDims {
            w: None,
            h: None,
            ratio: None,
        }
    }

    /// Effective intrinsic width: the declared width, else the CSS 2.1
    /// §10.3.2 300px default for replaced elements with no intrinsic
    /// width.
    pub fn width_or_default(&self) -> f32 {
        self.w.unwrap_or(300.0)
    }

    /// Effective intrinsic height: the declared height, else the CSS
    /// 2.1 §10.6.2 150px default.
    pub fn height_or_default(&self) -> f32 {
        self.h.unwrap_or(150.0)
    }

    /// Effective intrinsic aspect ratio: the explicit ratio, else
    /// `w`/`h` when both are positive and finite.
    pub fn ratio_or_derived(&self) -> Option<f32> {
        if let Some(r) = self.ratio
            && r > 0.0
            && r.is_finite()
        {
            return Some(r);
        }
        match (self.w, self.h) {
            (Some(w), Some(h)) if w > 0.0 && h > 0.0 && w.is_finite() && h.is_finite() => {
                Some(w / h)
            }
            _ => None,
        }
    }
}

/// A rectangle: position plus size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// Four edge sizes: top, right, bottom, left.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EdgeSizes {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

/// What kind of box a [`LayoutBox`] is.
///
/// `Table`, `TableRowGroup`, `TableRow`, and `TableCell` are the
/// table-internal boxes produced by `display: table` and friends (see
/// `layout::layout_table`); they form the table → row-group → row → cell
/// structure, with the table's border box sized like a block-level box.
/// `TableColumn` / `TableColumnGroup` are the column boxes produced by
/// `display: table-column` / `table-column-group` (the `<col>` and
/// `<colgroup>` elements); they carry column widths and collapsed-border
/// claims but take no space in the grid.
/// `Grid` is the box produced by `display: grid` (see
/// `layout::layout_grid`): a block-level grid container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxType {
    Block,
    Inline,
    InlineBlock,
    Text,
    AnonymousBlock,
    Table,
    TableRowGroup,
    TableRow,
    TableCell,
    /// `display: table-column` / `table-column-group`: column boxes
    /// generated from `<col>` / `<colgroup>` elements. They take no space
    /// in the grid themselves; their specified widths feed the table
    /// layout algorithm and their borders participate in collapsed-border
    /// conflict resolution (see `layout::layout_table`).
    TableColumn,
    TableColumnGroup,
    Grid,
}

/// Which element a collapsed-border claim comes from: the element
/// precedence for equal-width, equal-style conflicts (CSS 2.1 §17.6.2.1)
/// is cell > row > row-group > column > column-group > table.
/// `<col>`/`<colgroup>` generate column boxes whose left/right borders
/// claim the vertical grid edges of their columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BorderClaimant {
    Cell,
    Row,
    RowGroup,
    Column,
    ColumnGroup,
    Table,
}

/// Border styles ranked for the collapsed-border conflict resolution
/// (CSS 2.1 §17.6.2.1). `Hidden` never reaches ranking: a `hidden` claim
/// suppresses every border at the shared edge. `None` ranks below every
/// visible style — a `none` claim only wins an edge nobody else claims,
/// and its used width is already zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollapsedBorderStyle {
    Hidden,
    Double,
    Solid,
    Dashed,
    Dotted,
    Ridge,
    Outset,
    Groove,
    Inset,
    None,
}

/// The winning border on one collapsed cell edge: the single border
/// paint should draw on the shared grid line instead of each adjacent
/// cell drawing its own. `Hidden` style (or zero width) means no border
/// is drawn on the edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CollapsedBorder {
    pub width: f32,
    pub style: CollapsedBorderStyle,
    pub color: bowser_css::Color,
}

impl CollapsedBorder {
    /// A no-border edge (zero width, `None` style), used for grid edges
    /// a cell does not paint; see [`LayoutBox::collapsed_borders`].
    pub const NONE: CollapsedBorder = CollapsedBorder {
        width: 0.0,
        style: CollapsedBorderStyle::None,
        color: bowser_css::Color {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        },
    };
}

/// Collapsed-border winners for every grid cell of one table, parallel
/// to the table's grid cells in DOM order; each entry is
/// `[top, right, bottom, left]`.
///
/// Paint contract: in the collapsing border model each grid line is
/// shared, so exactly one border is drawn per line — the winner stored
/// here for either adjacent cell's edge (both sides of a shared edge
/// resolve to the same winner).
#[derive(Clone, Debug, PartialEq)]
pub struct CollapsedBorders {
    pub cells: Vec<[CollapsedBorder; 4]>,
}

impl CollapsedBorders {
    /// The winning border on one edge of the `cell`-th grid cell
    /// (`side` is 0 = top, 1 = right, 2 = bottom, 3 = left).
    pub fn edge(&self, cell: usize, side: usize) -> Option<CollapsedBorder> {
        self.cells.get(cell).and_then(|e| e.get(side)).copied()
    }
}

/// One laid-out line fragment of a [`BoxType::Text`] box, for the
/// paint/JS contract.
///
/// `rect` is the fragment's placed rect in absolute page coordinates;
/// `baseline` is the line's baseline in absolute page coordinates;
/// `text_range` is the byte range into the box's (already
/// `text-transform`ed) [`LayoutBox::text`] that this fragment renders.
/// Paint renders the fragments in order without re-running line breaking.
#[derive(Clone, Debug, PartialEq)]
pub struct TextFragment {
    pub rect: Rect,
    pub baseline: f32,
    pub text_range: Range<usize>,
}

/// A laid-out `::marker` for `list-style-position: outside` (CSS Lists
/// 3): the marker of a `display: list-item` box, placed in the
/// inline-start margin area (inline-end for rtl).
///
/// `text` is the marker content ("•", "1.", …), `rect` its border box
/// in absolute page coordinates, `style` the `::marker` computed style
/// when a `::marker` rule matched the list item (`None` means the
/// marker inherits the list item's style), `font_size` the resolved
/// marker font size in px, and `node` the DOM node of the owning list
/// item (matching its principal box's [`LayoutBox::node`]).
/// `list-style-position: inside` markers are ordinary inline boxes in
/// the principal box's flow and never appear here.
/// [`LayoutBox::node`]: LayoutBox::node
#[derive(Clone, Debug)]
pub struct MarkerBox {
    pub text: String,
    pub rect: Rect,
    pub style: Option<bowser_css::ComputedStyle>,
    pub font_size: f32,
    pub node: Option<bowser_dom::NodeId>,
}

/// One laid-out box.
///
/// `rect` is the border box in absolute page coordinates. `style` is the
/// computed style for element boxes and `None` for anonymous/text boxes.
/// `text` holds the (whitespace-collapsed, `text-transform`ed) text for
/// [`BoxType::Text`] boxes. `node` is the DOM node the box was built from
/// (`None` for anonymous boxes). `fragments` holds one [`TextFragment`]
/// per laid-out line fragment for [`BoxType::Text`] boxes and is empty
/// otherwise. `z_index` is the resolved `z-index`: `None` for `auto` on
/// ordinary boxes, but preserved for static flex/grid items, whose
/// non-`auto` `z-index` creates a stacking context (css-flexbox-1 /
/// css-grid-1 §4.2); paint orders it at the Appendix E z-index steps.
/// `collapsed_borders` carries the collapsed-border winners
/// ([top, right, bottom, left], see [`CollapsedBorder`]) for a
/// [`BoxType::TableCell`] box in a `border-collapse: collapse` table:
/// the edges this cell paints, so paint draws each shared grid line's
/// single winning border instead of the cell's own CSS borders. Edges
/// the cell does not own (its right/bottom edge when another cell
/// continues the grid there) are [`CollapsedBorder::NONE`]. It is `None`
/// for every other box, and for cells in `border-collapse: separate`
/// tables. Boxed: the array is 48 bytes but `Some` only for cells in
/// collapsing tables, so the box keeps it off the hot inline footprint
/// (see the size tripwire below).
///
/// `Clone` exists for the fragmentation post-pass ([`layout_fragmented`]
/// clones per-page tree slices); the single-page layout path never clones.
#[derive(Clone, Debug)]
pub struct LayoutBox {
    pub rect: Rect,
    pub children: Vec<LayoutBox>,
    pub box_type: BoxType,
    /// The computed style for element boxes, `None` for anonymous/text
    /// boxes. Shared with the CSS styler's memo by reference count: the
    /// `Rc` is threaded from `Styler::style_for_shared` through the box
    /// arena, so converting the arena to the public tree is one refcount
    /// bump per element box, not a deep clone. Read-only for paint/JS
    /// (deref to [`bowser_css::ComputedStyle`]).
    pub style: Option<Rc<bowser_css::ComputedStyle>>,
    /// The (whitespace-collapsed, `text-transform`ed) text for
    /// [`BoxType::Text`] boxes, shared with the DOM's interned `Arc<str>`
    /// — a refcount bump at the arena-to-tree boundary, not a copy.
    /// `None` for every other box type.
    pub text: Option<Arc<str>>,
    pub node: Option<bowser_dom::NodeId>,
    pub fragments: Vec<TextFragment>,
    pub z_index: Option<i32>,
    pub collapsed_borders: Option<Box<[CollapsedBorder; 4]>>,
    /// Per-dimension phantom sizing for a cell-less table: true when the
    /// table's width (resp. height) comes purely from border-spacing
    /// phantom gaps with no cells to measure. Stamped by
    /// [`crate::layout::layout_table::table_sized_by_phantom`] at the end
    /// of table layout. The two dimensions stamp independently — the old
    /// single-bool OR lost which dimension was phantom, so the parse
    /// shim re-derived per-dimension residuals and diverged (caption
    /// bleed, min-width bleed, redistributed-height bleed, rows==0
    /// bleed); consumers read these flags directly instead.
    /// Only meaningful on [`BoxType::Table`] boxes.
    pub phantom_w: bool,
    /// See [`Self::phantom_w`]: the height-dimension phantom flag.
    pub phantom_h: bool,
}

/// Size tripwire: `LayoutBox` is stored inline in the parent's `children`
/// vec, so every byte here is memmoved on vec realloc and copied on
/// full-tree rebuilds. The cold, rarely-`Some` fields stay off the hot
/// inline footprint — `style` is a shared `Rc` (8 bytes, one refcount
/// bump per box, no deep clone) and `collapsed_borders` is boxed (the
/// array is 48 bytes but `Some` only for cells in collapsing tables).
/// `text` shares the DOM's interned `Arc<str>` (16 bytes) instead of an
/// owned `String` (24 bytes + allocation).
/// Grow this budget only deliberately, after re-measuring.
const _: () = assert!(std::mem::size_of::<LayoutBox>() <= 128);

/// Interned font-family id for [`FontMetrics`] calls.
///
/// A `u32` FNV-1a hash of the ASCII-lowercased family name (CSS family
/// names match ASCII case-insensitively). It is `Copy`, allocation-free
/// to compute, and usable as a hash-map key, so per-call family lookup
/// is a hash probe, never a string compare or allocation. The id is
/// opaque: obtain it from [`FontMetrics::intern_family`] on the same
/// metrics object the measurement calls go through (stateful backends
/// register the name behind the id there).
///
/// Trade-off: the 32-bit hash (the 64-bit FNV-1a folded by xor) can
/// collide — two distinct family names mapping to one id would measure
/// the wrong face. The risk is accepted deliberately: family-name
/// collisions need ~77k names for a 50% chance (birthday bound), while
/// real stylesheets name a handful of families, and `intern_family`
/// lets stateful backends detect a collision (two names, one id) and
/// re-key. Widening to `u64` would cost 4 bytes per id carrier for no
/// observable gain.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FontFamilyId(u32);

impl FontFamilyId {
    /// The id for `family`, computed without allocation. Backends that
    /// need the name (to resolve a face) override
    /// [`FontMetrics::intern_family`] to register it; this pure hash is
    /// what the stateless implementors ([`FixedFontMetrics`]) use.
    pub const fn of(family: &str) -> Self {
        let bytes = family.as_bytes();
        let mut hash: u64 = 0xcbf29ce484222325;
        let mut i = 0;
        while i < bytes.len() {
            let mut b = bytes[i];
            if b >= b'A' && b <= b'Z' {
                b |= 0x20;
            }
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
            i += 1;
        }
        Self((hash ^ (hash >> 32)) as u32)
    }

    /// CSS's initial `font-family: serif`.
    pub const DEFAULT: FontFamilyId = FontFamilyId::of("serif");
}

/// Font measurement, injected so layout stays font-backend agnostic.
///
/// The production implementation is `bowser_paint::fonts::FontCache`
/// (shareable via `bowser_paint::fonts::SharedFontCache`), which measures
/// with real glyph advances from the faces paint resolves; passing it
/// here is what keeps laid-out line breaks aligned with painted glyphs.
/// [`FixedFontMetrics`] is the test-only placeholder.
///
/// Every measurement takes the text run's [`FontFamilyId`] (interned
/// once per run via [`intern_family`](FontMetrics::intern_family)), so
/// runs styled with a non-default family — e.g. the Ahem test font —
/// measure with that family's face instead of the default serif face.
pub trait FontMetrics {
    /// Intern `family` (a CSS `font-family` name), returning the cheap
    /// [`FontFamilyId`] to pass to the measurement calls below.
    /// Backends that resolve faces from the name register it here; the
    /// default just hashes (see [`FontFamilyId::of`]).
    fn intern_family(&self, family: &str) -> FontFamilyId {
        FontFamilyId::of(family)
    }

    /// Horizontal advance of `text` at `font_size`, in the `family` face.
    fn advance(&self, text: &str, font_size: f32, family: FontFamilyId) -> f32;
    /// Distance between baselines of consecutive lines at `font_size`,
    /// in the `family` face.
    fn line_height(&self, font_size: f32, family: FontFamilyId) -> f32;
    /// Typographic ascent (above the baseline) and descent (below the
    /// baseline) in px at `font_size`, from the `family` face.
    ///
    /// Layout splits each line's used line height across the baseline by
    /// this ratio, so the baseline sits at the face's real typographic
    /// position. Table cell-baseline alignment compares those baselines
    /// across cells, so real metrics here are what make differently
    /// sized cells align on a common baseline.
    ///
    /// The default is the historical 0.8/0.2 split of
    /// [`FontMetrics::line_height`], so implementors without face metrics
    /// keep their exact current behavior. Override it when real metrics
    /// are available (paint's font cache can serve swash
    /// ascent/descent/leading).
    fn ascent_descent(&self, font_size: f32, family: FontFamilyId) -> (f32, f32) {
        let lh = self.line_height(font_size, family);
        (0.8 * lh, 0.2 * lh)
    }

    /// x-height (height of the `x` glyph above the baseline) in px at
    /// `font_size`, from the `family` face. Used by CSS 2.1 §10.8.1
    /// `vertical-align: middle`, which centers the box on the parent's
    /// baseline minus half the parent's x-height.
    ///
    /// The default is the css-inline-3 suggested fallback of 0.5em, so
    /// implementors without face metrics keep a sane behavior.
    /// Override it when real metrics are available.
    fn x_height(&self, font_size: f32, _family: FontFamilyId) -> f32 {
        0.5 * font_size
    }
}

/// Test-only placeholder metrics: every character advances
/// `0.6 * font_size`, and the line height is `1.2 * font_size`.
///
/// This exists so layout's unit tests can run without a font backend.
/// Production callers must NOT use it: measuring with the placeholder
/// while paint shapes text with real glyph advances misaligns laid-out
/// line breaks and positions with the painted glyphs. Pass a real
/// [`FontMetrics`] implementation instead (e.g.
/// `bowser_paint::fonts::FontCache` or `bowser_paint::fonts::SharedFontCache`,
/// which measure with real advances from the same faces paint resolves).
pub struct FixedFontMetrics;

impl FontMetrics for FixedFontMetrics {
    fn advance(&self, text: &str, font_size: f32, _family: FontFamilyId) -> f32 {
        text.chars().count() as f32 * 0.6 * font_size
    }

    fn line_height(&self, font_size: f32, _family: FontFamilyId) -> f32 {
        1.2 * font_size
    }
}

/// The UA stylesheet, parsed once per process. [`layout`] runs once per
/// document; re-parsing the UA sheet on every call was a measurable
/// per-layout cost, so it is parsed lazily on first use and shared
/// thereafter (it is immutable; the [`Styler`] only borrows it).
///
/// Public so the embedder (`bowser_app::window::try_partial_layout`)
/// can audit the exact stylesheet set the layout passes match against
/// (see `bowser_css::selectors_are_subtree_local`).
pub fn ua_stylesheet() -> &'static Stylesheet {
    static UA: OnceLock<Stylesheet> = OnceLock::new();
    UA.get_or_init(bowser_css::default_ua_stylesheet)
}

/// Lay out `doc` in a viewport of `viewport` size, returning the root box
/// (the document element's box).
///
/// `author` is the list of author stylesheets in cascade order; the UA
/// stylesheet is parsed once per process and shared.
///
/// The viewport is assumed unscrolled; for a scrolled viewport — which is
/// what makes `position: sticky` stick — use [`layout_with_scroll`].
///
/// Equivalent to [`layout_with_intrinsics`] with an empty intrinsics map
/// (every replaced element sizes to 0x0).
pub fn layout(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn FontMetrics,
) -> LayoutBox {
    layout_impl(
        doc,
        author,
        viewport,
        fonts,
        &HashMap::new(),
        (0.0, 0.0),
        &HashMap::new(),
    )
}

/// Lay out `doc` like [`layout`], but replaced elements (`<img>` etc.)
/// take their box size from `intrinsics`, keyed by DOM node.
///
/// A replaced element whose CSS `width`/`height` are `auto` uses its map
/// entry as its intrinsic size (still clamped by `min`/`max` widths and
/// heights); a specified CSS `width`/`height` overrides the map. A replaced
/// element with no map entry sizes to 0x0 (documented v1 behavior).
pub fn layout_with_intrinsics(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn FontMetrics,
    intrinsics: &HashMap<bowser_dom::NodeId, IntrinsicDims>,
) -> LayoutBox {
    layout_impl(
        doc,
        author,
        viewport,
        fonts,
        intrinsics,
        (0.0, 0.0),
        &HashMap::new(),
    )
}

/// Lay out `doc` like [`layout`], positioned for a viewport scrolled to
/// `scroll` (`(scroll_x, scroll_y)` in CSS px).
///
/// This is the entry point that makes `position: sticky` stick: each
/// sticky box is constrained per CSS Position 3 §3.1 against the
/// scrollport at this offset (see `layout::apply_sticky_constraints`).
/// With a zero offset this is identical to [`layout`]. Callers re-run
/// layout when the scroll offset changes; there is no incremental scroll
/// update.
pub fn layout_with_scroll(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn FontMetrics,
    scroll: (f32, f32),
) -> LayoutBox {
    layout_with_scroll_and_offsets(doc, author, viewport, fonts, scroll, &HashMap::new())
}

/// Lay out `doc` like [`layout_with_scroll`], but each nested scroll
/// container constrains its sticky boxes at its live scroll offset from
/// `nested_offsets`, keyed by the container's DOM node.
///
/// Arena box indices are not stable across layouts, so the map keys on
/// the scroller's DOM node — the same [`bowser_dom::NodeId`] the app's
/// scroll tree carries per container (and that [`LayoutBox::node`]
/// reports for the scroller's box). A scroller with no map entry
/// constrains at offset zero, so an empty map is exactly
/// [`layout_with_scroll`]. See the `layout_position` module docs for the
/// constraint itself.
pub fn layout_with_scroll_and_offsets(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn FontMetrics,
    scroll: (f32, f32),
    nested_offsets: &HashMap<bowser_dom::NodeId, (f32, f32)>,
) -> LayoutBox {
    layout_with_intrinsics_and_scroll_and_offsets(
        doc,
        author,
        viewport,
        fonts,
        &HashMap::new(),
        scroll,
        nested_offsets,
    )
}

/// Lay out `doc` like [`layout_with_intrinsics`], positioned for a
/// viewport scrolled to `scroll` (`(scroll_x, scroll_y)` in CSS px).
///
/// The combination the windowed renderer needs: replaced elements size
/// from `intrinsics` (so `<img>` keeps its decoded size) *and* the
/// scroll offset feeds the sticky constraints (so `position: sticky`
/// sticks — see [`layout_with_scroll`]). With a zero scroll offset this
/// is identical to [`layout_with_intrinsics`].
pub fn layout_with_intrinsics_and_scroll(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn FontMetrics,
    intrinsics: &HashMap<bowser_dom::NodeId, IntrinsicDims>,
    scroll: (f32, f32),
) -> LayoutBox {
    layout_with_intrinsics_and_scroll_and_offsets(
        doc,
        author,
        viewport,
        fonts,
        intrinsics,
        scroll,
        &HashMap::new(),
    )
}

/// Lay out `doc` like [`layout_with_intrinsics_and_scroll`], but each
/// nested scroll container constrains its sticky boxes at its live
/// scroll offset from `nested_offsets` (see
/// [`layout_with_scroll_and_offsets`): the same `(intrinsics, scroll)`
/// combination plus per-container offsets, which is what the app's
/// `refresh` needs once it tracks them.
///
/// With an empty `nested_offsets` map this is identical to
/// [`layout_with_intrinsics_and_scroll`].
pub fn layout_with_intrinsics_and_scroll_and_offsets(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn FontMetrics,
    intrinsics: &HashMap<bowser_dom::NodeId, IntrinsicDims>,
    scroll: (f32, f32),
    nested_offsets: &HashMap<bowser_dom::NodeId, (f32, f32)>,
) -> LayoutBox {
    layout_impl(
        doc,
        author,
        viewport,
        fonts,
        intrinsics,
        scroll,
        nested_offsets,
    )
}

// ---------------------------------------------------------------------------
// Scroll containers
// ---------------------------------------------------------------------------

/// Per-axis classification of a box's computed `overflow` (CSS Overflow 3
/// §2).
///
/// The canonical implementation, shared by the layout-side container
/// discovery ([`scroll_containers`]) and the app's viewport propagation
/// rule (`bowser_app::scroll`), so the two cannot diverge.
///
/// Returns `(container, user)` per axis:
///
/// * `container`: `auto`/`scroll`/`hidden` establish a scroll container
///   (css-overflow-3 §2.1: `hidden` also creates one — it clips and is
///   programmatically scrollable via `scrollTop`/`scrollTo`).
/// * `user`: only `auto`/`scroll` consume wheel/keyboard deltas. A
///   `hidden` axis never consumes a user scroll; the delta chains past
///   it to the next scroller untouched.
/// * `clip`: clips like `hidden` but is neither container nor user —
///   no ScrollTree entry, no scrollbar, and `scrollTop`/`scrollTo`
///   writes are no-ops. Clipping itself is paint-side, keyed off the
///   computed value (`bowser_paint::display_list`'s `overflow_clips`).
///
/// The cascade expands the `overflow` shorthand into `overflow-x` /
/// `overflow-y` at declaration time, so the longhands always reflect
/// author declaration order: a later longhand overrides the shorthand,
/// and a lone shorthand sets both axes. When one axis is non-`visible`
/// and the other is `visible`, the `visible` one computes to `auto`
/// (CSS Overflow 3 §2).
///
/// Classic scrollbar thickness in CSS px, used for `scrollbar-gutter:
/// stable` content-box reservation (see [`scrollbar_gutter_reservation`]).
/// Must match `bowser_app::scroll::SCROLLBAR_THICKNESS`.
pub const SCROLLBAR_THICKNESS: f32 = 12.0;

/// Content-box reservation for `scrollbar-gutter` (CSS Overflow 4 §):
/// returns the `(left, right)` inline reservation widths when the style
/// is a scroll container with `scrollbar-gutter: stable`, else `(0, 0)`.
///
/// Only the vertical axis reserves: `scrollbar-gutter` reserves space
/// for the vertical (inline-end) scrollbar. The reservation applies to
/// every scroll container — `auto`, `scroll`, *and* `hidden` (CSS
/// Overflow 3 §2: `hidden` clips but still establishes a scroll
/// container) — whether or not a scrollbar can actually appear: that is
/// the point of `stable`. `both-edges` mirrors the strip to the
/// inline-start side as well.
pub fn scrollbar_gutter_reservation(style: &bowser_css::ComputedStyle) -> (f32, f32) {
    let ((_, container_y), _) = overflow_axes(style);
    if !container_y {
        return (0.0, 0.0);
    }
    stable_gutter_amounts(style)
}

/// Content-box space consumed by a *forced* classic scrollbar (CSS
/// Overflow 3 §2.2): `overflow-y: scroll` displays the vertical
/// scrollbar whether or not any content overflows, and a displayed
/// classic scrollbar takes its thickness out of the scroll container's
/// content box — children lay out against the narrowed width. This is
/// the width WPT `scrollbar-gutter-root.html` emulates via
/// `offsetWidth` across a forced scroller, so it must equal
/// [`SCROLLBAR_THICKNESS`]: the reservation side of that reftest comes
/// from [`scrollbar_gutter_reservation`] / [`viewport_gutter_reservation`].
///
/// Returns the `(left, right)` shrink; `(0, 0)` unless the vertical
/// axis is forced (`scroll`). The strip sits at the inline-end edge —
/// the right edge under `direction: ltr`, the left under `direction:
/// rtl` — the same edge a shown scrollbar paints
/// (`bowser_app::scroll::vertical_track_rect`), so it never stacks
/// with the `scrollbar-gutter` reservation: callers take the per-side
/// max, since a shown scrollbar paints exactly the reserved strip.
/// Only the vertical axis: a forced *horizontal* scrollbar's height
/// consumption is not yet modeled (block layout resolves height
/// bottom-up, so unlike width it is not a pure layout input).
pub fn forced_scrollbar_reservation(style: &bowser_css::ComputedStyle) -> (f32, f32) {
    let oy = style
        .get_str_by_index(props::OVERFLOW_Y)
        .unwrap_or("visible");
    if !oy.eq_ignore_ascii_case("scroll") {
        return (0.0, 0.0);
    }
    // Same writing-mode bail as `stable_gutter_amounts`: under a
    // vertical writing mode this physical left/right strip cannot
    // express the inline-end edge.
    if !matches!(
        style::writing_mode_of(style),
        style::WritingMode::HorizontalTb
    ) {
        return (0.0, 0.0);
    }
    let rtl = style
        .get_str_by_index(props::DIRECTION)
        .is_some_and(|d| d.eq_ignore_ascii_case("rtl"));
    if rtl {
        (SCROLLBAR_THICKNESS, 0.0)
    } else {
        (0.0, SCROLLBAR_THICKNESS)
    }
}

/// Content-box space consumed by an `overflow-y: auto` vertical
/// scrollbar that actually appears (CSS Overflow 3 §2.2): `auto` shows
/// the classic scrollbar exactly when content overflows the padding
/// box, and the shown scrollbar takes its thickness out of the scroll
/// container's content box — children lay out against the narrowed
/// width. Unlike [`forced_scrollbar_reservation`] this is conditional:
/// the caller must have established that real vertical overflow exists
/// (layout knows that only after children are laid out, so this cannot
/// run in the `layout_block` prologue).
///
/// Returns the `(left, right)` shrink; `(0, 0)` unless the vertical
/// axis is user-scrollable-but-unforced (`auto`, including the
/// visible→auto pairing from [`overflow_axes`]: with a non-`visible`
/// `overflow-x`, a `visible` y computes to `auto` and can show the
/// scrollbar). `scroll` is excluded — its strip is forced and already
/// reserved by [`forced_scrollbar_reservation`]; `hidden`/`clip` never
/// show a scrollbar. Same geometry rules as the forced strip:
/// inline-end edge (right in ltr, left in rtl — the edge a shown
/// scrollbar paints), horizontal writing modes only, per-side max with
/// the other reservations (never additive).
pub fn auto_scrollbar_reservation(style: &bowser_css::ComputedStyle) -> (f32, f32) {
    let oy = style
        .get_str_by_index(props::OVERFLOW_Y)
        .unwrap_or("visible");
    if oy.eq_ignore_ascii_case("scroll") {
        return (0.0, 0.0);
    }
    let (_, (_, user_y)) = overflow_axes(style);
    if !user_y {
        return (0.0, 0.0);
    }
    // Same writing-mode bail as `forced_scrollbar_reservation`: under a
    // vertical writing mode this physical left/right strip cannot
    // express the inline-end edge.
    if !matches!(
        style::writing_mode_of(style),
        style::WritingMode::HorizontalTb
    ) {
        return (0.0, 0.0);
    }
    let rtl = style
        .get_str_by_index(props::DIRECTION)
        .is_some_and(|d| d.eq_ignore_ascii_case("rtl"));
    if rtl {
        (SCROLLBAR_THICKNESS, 0.0)
    } else {
        (0.0, SCROLLBAR_THICKNESS)
    }
}

/// The `(left, right)` reservation widths for a style whose
/// `scrollbar-gutter` includes `stable`, with no scroll-container
/// eligibility check: `stable both-edges` (stored as a `List` of
/// keywords) mirrors the strip to the inline-start side as well.
fn stable_gutter_amounts(style: &bowser_css::ComputedStyle) -> (f32, f32) {
    // `scrollbar-gutter` computes to `auto`, `stable`, or
    // `stable both-edges` (stored as a `List` of keywords).
    let (stable, both_edges) = match style.get_by_index(bowser_css::props::SCROLLBAR_GUTTER) {
        Some(bowser_css::CssValue::Keyword(k)) => (k.eq_ignore_ascii_case("stable"), false),
        Some(bowser_css::CssValue::List(items)) => {
            let mut stable = false;
            let mut both = false;
            for item in items {
                if let bowser_css::CssValue::Keyword(k) = item {
                    stable |= k.eq_ignore_ascii_case("stable");
                    both |= k.eq_ignore_ascii_case("both-edges");
                }
            }
            (stable, both)
        }
        _ => (false, false),
    };
    if !stable {
        return (0.0, 0.0);
    }
    // The reservation sits on the inline axis (CSS Overflow 4 §2.2:
    // `both-edges` mirrors it across the inline-start/inline-end
    // edges). Under a vertical writing mode the inline axis is
    // vertical, so there is no horizontal (left/right) strip to
    // reserve: the gutter would sit at the inline-end edge (the top
    // or bottom), which this physical left/right reservation cannot
    // express. Reserve nothing rather than shifting content along the
    // wrong axis (WPT `scrollbar-gutter-vertical-lr-002` /
    // `scrollbar-gutter-vertical-rl-002`).
    if !matches!(
        style::writing_mode_of(style),
        style::WritingMode::HorizontalTb
    ) {
        return (0.0, 0.0);
    }
    // The reservation sits at the inline-end edge (CSS Overflow 4):
    // the right edge under `direction: ltr`, the left edge under
    // `direction: rtl`. `direction` is inherited, so the computed
    // value read here is already the effective one. `both-edges`
    // mirrors the strip to the inline-start side as well, so it is
    // symmetric under either direction.
    let rtl = style
        .get_str_by_index(props::DIRECTION)
        .is_some_and(|d| d.eq_ignore_ascii_case("rtl"));
    if both_edges {
        (SCROLLBAR_THICKNESS, SCROLLBAR_THICKNESS)
    } else if rtl {
        (SCROLLBAR_THICKNESS, 0.0)
    } else {
        (0.0, SCROLLBAR_THICKNESS)
    }
}

/// The `(left, right)` gutter reservation the *viewport* takes when
/// the root element's `scrollbar-gutter` is propagated to it (CSS
/// Overflow 4 §2.2: the value set on the root element applies to the
/// viewport, and the root element's own used value is `auto`). The
/// viewport is always a scroll container for this purpose — its
/// overflow is the propagated root/body overflow, `visible` included —
/// so unlike [`scrollbar_gutter_reservation`] there is no
/// user-scrollable-axis requirement: `html { scrollbar-gutter: stable }`
/// reserves the strip even when nothing currently overflows. The
/// reservation shrinks the initial containing block the root element
/// is laid out against (see `layout_forest`), so the root's border box
/// — not just its children — excludes the gutter.
pub fn viewport_gutter_reservation(style: &bowser_css::ComputedStyle) -> (f32, f32) {
    stable_gutter_amounts(style)
}
pub fn overflow_axes(style: &bowser_css::ComputedStyle) -> ((bool, bool), (bool, bool)) {
    let visible = |v: &str| v.eq_ignore_ascii_case("visible");
    let mut ox = style
        .get_str_by_index(props::OVERFLOW_X)
        .unwrap_or("visible");
    let mut oy = style
        .get_str_by_index(props::OVERFLOW_Y)
        .unwrap_or("visible");
    if visible(ox) && !visible(oy) {
        ox = "auto";
    }
    if visible(oy) && !visible(ox) {
        oy = "auto";
    }
    let container = |v: &str| {
        v.eq_ignore_ascii_case("auto")
            || v.eq_ignore_ascii_case("scroll")
            || v.eq_ignore_ascii_case("hidden")
    };
    let user = |v: &str| v.eq_ignore_ascii_case("auto") || v.eq_ignore_ascii_case("scroll");
    // `clip` deliberately matches neither closure: it clips (paint keys
    // off the computed value) but creates no scroll container and
    // consumes no user scroll — CSS Overflow 3 §2.
    ((container(ox), container(oy)), (user(ox), user(oy)))
}

/// Whether the computed `transform` is non-`none`: such a box
/// establishes a containing block for `position: fixed` descendants
/// (CSS Transforms 1 §3). `none` computes to `Keyword("none")`; any
/// function list computes to [`CssValue::Transform`] (one entry per
/// function, or a `List` of them for a space-separated list).
pub fn has_non_none_transform(style: &bowser_css::ComputedStyle) -> bool {
    match style.get_by_index(props::TRANSFORM) {
        Some(CssValue::Transform(_)) => true,
        Some(CssValue::List(items)) => items.iter().any(|i| matches!(i, CssValue::Transform(_))),
        _ => false,
    }
}

/// Whether the computed style establishes a containing block for
/// `position: fixed` (and absolutely positioned) descendants: any of
/// the fixed-CB triggers — a non-`none` `transform` (CSS Transforms 1
/// §3, via [`has_non_none_transform`]), or one of the CSS-side
/// triggers — `will-change` naming `transform`, `filter`, or
/// `perspective` (CSS Will Change 1 §3, narrowed by CSS Position 3
/// §2.1; stacking-context-only hints like `opacity`/`mix-blend-mode`/
/// `isolation` do not count), a non-`none` `filter` (Filter Effects 1),
/// a non-`none` `perspective` (CSS Transforms 2), or `contain:
/// paint | layout | strict | content` (CSS Contain 1/2), via
/// [`bowser_css::establishes_fixed_cb`].
pub fn establishes_fixed_cb(style: &bowser_css::ComputedStyle) -> bool {
    has_non_none_transform(style) || bowser_css::establishes_fixed_cb(style)
}

/// A scroll container discovered in a layout tree: one box with a
/// scroll-container `overflow` axis (CSS Overflow 3 §2.1), with the
/// geometry the app's scroll tree needs to build nested scrollers from
/// real layout output instead of the viewport alone.
///
/// `node` is the DOM node of the scrolling box (`None` for anonymous
/// boxes; those never carry a non-`visible` `overflow`, so every entry
/// this emits is `Some` in practice). `scrollport` is the box's border
/// box in absolute page coordinates: the clip region, fixed while the
/// content scrolls under it. `max` is the per-axis maximum offset — the
/// scrollable overflow extent minus the scrollport, floored at zero.
/// `scroll_origin` is the per-axis minimum offset — content's nearest
/// edge minus the scrollport's own edge, capped at zero (negative when
/// content overflows above/left of the scrollport; `(0.0, 0.0)` when
/// nothing overflows negatively).
/// `axes` / `user_axes` are the [`overflow_axes`] classification;
/// `depth` is the box's depth in the layout tree (the root is 0), so a
/// nested container always has greater depth than its ancestors.
///
/// Viewport propagation is the caller's job: `overflow` on the document
/// element / `<body>` propagates to the viewport scroller (CSS Overflow 3
/// §2.4) instead of forming a nested container, so consumers filter the
/// root entry with their own viewport-scroller rule.
#[derive(Clone, Debug, PartialEq)]
pub struct ScrollContainerInfo {
    pub node: Option<bowser_dom::NodeId>,
    pub scrollport: Rect,
    pub max: (f32, f32),
    pub scroll_origin: (f32, f32),
    pub axes: (bool, bool),
    pub user_axes: (bool, bool),
    pub depth: usize,
}

/// Resolved `position` of a layout box, defaulting to `static` for
/// anonymous (style-less) boxes.
fn position_of_box(box_: &LayoutBox) -> style::Position {
    box_.style
        .as_ref()
        .map(|s| style::position_of(s))
        .unwrap_or(style::Position::Static)
}

/// Scrollable overflow extent of `box_`: the farthest and nearest content
/// edges per axis, seeded at the scrollport's own edges. Subtrees rooted at
/// `skip` (confirmed nested scroll containers) contribute only their own
/// box: their overflowing content is clipped by their own scrollport and
/// never extends an outer scroller's range.
///
/// Descendants contribute their border boxes *after* their `transform`s
/// (CSS Overflow 3 §3: the scrollable overflow region is the union of the
/// descendants' transformed border boxes, in the scroll container's
/// coordinate space): the walk carries each box's accumulated transform
/// (see [`child_accumulated_transform`]) and unions the bounding box of
/// the transformed corners, so e.g. a child translated past the
/// scrollport's far edge extends `max`, and one translated to negative
/// coordinates extends the negative (scroll-origin) side.
///
/// Out-of-flow boxes contribute only when they scroll with this
/// scroller: `position: fixed` is viewport-anchored and never does —
/// unless a fixed-CB trigger (a non-`none` `transform`, `will-change`
/// naming a stacking-context trigger, a non-`none` `filter`, a non-`none`
/// `perspective`, or `contain: paint | layout | strict | content`)
/// between the scroller and the box makes that ancestor its containing
/// block (CSS Transforms 1 §3; Filter Effects 1; CSS Will Change 1 §3;
/// CSS Transforms 2; CSS Contain 1/2), in which case it scrolls with the
/// scroller — and `position: absolute` only does when its containing
/// block is inside the scroller (the nearest positioned *or*
/// fixed-CB-establishing ancestor, per the same specs). The walk tracks
/// `cb_inside`: whether the nearest positioned-or-fixed-CB-establishing
/// ancestor above the box lies in the scroller's scrolling content, i.e.
/// whether an abspos box here takes its containing block from inside the
/// scroller; and `fixed_cb_inside`: whether the nearest fixed-CB
/// trigger above the box lies strictly below the scroller, i.e. whether
/// a fixed box here takes its containing block from inside the scroller.
/// (The scroller's own padding box is the scrollport frame, not scrolled
/// content, so a fixed-CB trigger on the scroller is NOT a fixed
/// containing block inside itself.)
/// The box's own `transform` evaluated to a 2D matrix in the box's
/// local frame (origin at its border-box top-left), pivoted around
/// `transform-origin` — the same evaluation paint performs for the
/// box's transform group, with the border box as the reference box
/// (`transform-box` defaults to `view-box`, the border box for HTML
/// elements). `None` when the property is absent/`none`, or when it
/// cannot be evaluated with absolute-length resolution (an `em`-based
/// `translate()`, say): paint likewise renders such a box
/// untransformed, so the extent walk treats it as untransformed too.
fn box_local_transform(b: &LayoutBox) -> Option<bowser_css::TransformMatrix> {
    let style = b.style.as_ref()?;
    if !has_non_none_transform(style) {
        return None;
    }
    let transform = style.get_by_index(props::TRANSFORM)?;
    let origin = style
        .get_by_index(props::TRANSFORM_ORIGIN)
        .and_then(bowser_css::TransformOrigin::parse)
        .unwrap_or_else(bowser_css::TransformOrigin::initial);
    let ctx = bowser_css::TransformContext {
        reference_width: b.rect.w.max(0.0),
        reference_height: b.rect.h.max(0.0),
        resolve_length: &bowser_css::resolve_absolute_length,
    };
    let m = bowser_css::evaluate_transform(transform, &origin, &ctx)?;
    [m.a, m.b, m.c, m.d, m.e, m.f]
        .iter()
        .all(|v| v.is_finite())
        .then_some(m)
}

/// The accumulated transform mapping `child`'s local coordinates (origin
/// at its border-box top-left) into the scrollable-overflow region's
/// coordinate space, given `parent_acc` (the accumulated transform of
/// `parent`, mapping parent-local coordinates into that space) and the
/// boxes' layout rects. A point of the child sits in the parent's local
/// frame at the child's layout offset within the parent, transformed by
/// the child's own `transform` first (CSS Transforms 1 §8: the
/// rightmost, innermost transform applies first). Boxes without an
/// evaluable `transform` contribute the identity local matrix.
fn child_accumulated_transform(
    parent_acc: &bowser_css::TransformMatrix,
    parent: &LayoutBox,
    child: &LayoutBox,
) -> bowser_css::TransformMatrix {
    let local = box_local_transform(child).unwrap_or_else(bowser_css::TransformMatrix::identity);
    parent_acc
        .mul(&bowser_css::TransformMatrix::translate(
            child.rect.x - parent.rect.x,
            child.rect.y - parent.rect.y,
        ))
        .mul(&local)
}

/// The `(left, top, right, bottom)` border+padding insets of a
/// split-inline wrapper's border box past its content box (the
/// child's unshifted footprint): how far the wrapper's own edges
/// spill on each side. Absolute lengths only ([`abs_len_px`]);
/// relative padding needs layout context this walk does not carry
/// and contributes zero here. A border width counts only when the
/// side's border style is set and not `none`/`hidden` (the initial
/// `none` zeroes the used width, CSS 2.1 §8.5.3).
fn split_inline_spill_insets(style: &bowser_css::ComputedStyle) -> (f32, f32, f32, f32) {
    let side = |pad: u8, bw: u8, bs: u8| {
        let padding = abs_len_px(style.get_by_index(pad));
        let border = match style.get_str_by_index(bs) {
            Some(v) if !v.eq_ignore_ascii_case("none") && !v.eq_ignore_ascii_case("hidden") => {
                abs_len_px(style.get_by_index(bw))
            }
            _ => 0.0,
        };
        padding + border
    };
    (
        side(
            props::PADDING_LEFT,
            props::BORDER_LEFT_WIDTH,
            props::BORDER_LEFT_STYLE,
        ),
        side(
            props::PADDING_TOP,
            props::BORDER_TOP_WIDTH,
            props::BORDER_TOP_STYLE,
        ),
        side(
            props::PADDING_RIGHT,
            props::BORDER_RIGHT_WIDTH,
            props::BORDER_RIGHT_STYLE,
        ),
        side(
            props::PADDING_BOTTOM,
            props::BORDER_BOTTOM_WIDTH,
            props::BORDER_BOTTOM_STYLE,
        ),
    )
}

fn scrollable_extent(box_: &LayoutBox, skip: &HashSet<*const LayoutBox>) -> (f32, f32, f32, f32) {
    let mut ox = box_.rect.x;
    let mut oy = box_.rect.y;
    let mut ex = box_.rect.x + box_.rect.w;
    let mut ey = box_.rect.y + box_.rect.h;
    let scroller_positioned = position_of_box(box_) != style::Position::Static;
    // A fixed box whose containing block is the scroller itself is
    // anchored to the scrollport frame, which does not move when the
    // scroller scrolls — its overflow can never be revealed by scrolling,
    // so it must not extend the range. Only a fixed-CB trigger strictly
    // below the scroller starts the flag as true; the walk below sets it
    // for deeper trigger boxes.
    //
    // The fourth stack element is the box's accumulated transform: the
    // matrix mapping the box's local coordinates into the region's
    // coordinate space (the scroller's frame — page coordinates with
    // the scroller's own transform, if any, factored out, since the
    // region lives in the scroll container's own coordinate system;
    // CSS Overflow 3 §3). Per that section the scrollable overflow
    // region is built from descendants' border boxes *after* applying
    // their transforms, so each box contributes the bounding box of its
    // transformed border box, and transforms compose down the tree: a
    // transformed ancestor moves its whole subtree's contributions.
    // The scroller's own accumulated transform is the translation to
    // its border-box origin, so a direct child's accumulated transform
    // is `translate(child origin) · child-local-transform`.
    let scroller_acc = bowser_css::TransformMatrix::translate(box_.rect.x, box_.rect.y);
    let mut stack: Vec<(&LayoutBox, bool, bool, bowser_css::TransformMatrix)> = box_
        .children
        .iter()
        .map(|c| {
            (
                c,
                scroller_positioned,
                false,
                child_accumulated_transform(&scroller_acc, box_, c),
            )
        })
        .collect();
    while let Some((b, cb_inside, fixed_cb_inside, acc)) = stack.pop() {
        let pos = position_of_box(b);
        // Viewport-anchored fixed boxes never scroll with any scroller,
        // so neither the box nor its subtree extends the range. A fixed
        // box whose containing block is the nearest fixed-CB trigger
        // inside this scroller scrolls with it and contributes.
        if pos == style::Position::Fixed && !fixed_cb_inside {
            continue;
        }
        // Absolutely positioned against a containing block outside the
        // scroller: does not scroll with it either.
        if pos == style::Position::Absolute && !cb_inside {
            continue;
        }
        // Skip the *self* contribution of a split-inline wrapper (a
        // `BoxType::Block` box whose element's computed `display` is
        // `inline`): layout generates that box as the flow container
        // for a block-level child of an inline element, and its rect
        // is the child's *unshifted* flow footprint — the element's
        // real (fragmented) boxes carry no such border box, so folding
        // it in would report the unshifted extent of a relatively
        // positioned child instead of its shifted content (WPT
        // `scrollable-overflow-relative-positioned-block-in-inline`:
        // 704 instead of 404). Its children still walk and contribute.
        //
        // The wrapper's own border/padding spill is real, though: the
        // wrapper lays out with its full style, so its border box
        // extends past the child's unshifted footprint (its content
        // box) by the border+padding on each side, and that ring
        // paints and must be reachable by scrolling. Contribute only
        // the wrapper rect edges on sides that actually have such an
        // inset — with no border/padding on a side, the wrapper's
        // edge on that side *is* the unshifted footprint edge and
        // stays skipped, preserving the 704-vs-404 behavior above.
        let split_inline_wrapper = b.box_type == BoxType::Block
            && b.style.as_ref().is_some_and(|s| {
                matches!(s.get("display"), Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("inline"))
            });
        if split_inline_wrapper {
            let style = b.style.as_deref().expect("wrapper has a style");
            let (il, it, ir, ib) = split_inline_spill_insets(style);
            if acc.is_identity() {
                if il > 0.0 {
                    ox = ox.min(b.rect.x);
                }
                if ir > 0.0 {
                    ex = ex.max(b.rect.x + b.rect.w);
                }
                if it > 0.0 {
                    oy = oy.min(b.rect.y);
                }
                if ib > 0.0 {
                    ey = ey.max(b.rect.y + b.rect.h);
                }
            } else {
                // Each spilling edge contributes its transformed
                // endpoints' extent on the edge's own axis.
                if il > 0.0 {
                    for (px, py) in [(0.0, 0.0), (0.0, b.rect.h)] {
                        let (tx, _) = acc.apply(px, py);
                        ox = ox.min(tx);
                    }
                }
                if ir > 0.0 {
                    for (px, py) in [(b.rect.w, 0.0), (b.rect.w, b.rect.h)] {
                        let (tx, _) = acc.apply(px, py);
                        ex = ex.max(tx);
                    }
                }
                if it > 0.0 {
                    for (px, py) in [(0.0, 0.0), (b.rect.w, 0.0)] {
                        let (_, ty) = acc.apply(px, py);
                        oy = oy.min(ty);
                    }
                }
                if ib > 0.0 {
                    for (px, py) in [(0.0, b.rect.h), (b.rect.w, b.rect.h)] {
                        let (_, ty) = acc.apply(px, py);
                        ey = ey.max(ty);
                    }
                }
            }
        } else {
            if acc.is_identity() {
                ex = ex.max(b.rect.x + b.rect.w);
                ey = ey.max(b.rect.y + b.rect.h);
                ox = ox.min(b.rect.x);
                oy = oy.min(b.rect.y);
            } else {
                // The transformed border box is a parallelogram; its
                // bounding box is the bounding box of the four transformed
                // corners. This is where transformed content reaches the
                // region: a `translate()` shifts it (possibly to negative
                // coordinates — the "unreachable" negative region), a
                // `scale()`/`rotate()` grows or reorients it.
                for (px, py) in [
                    (0.0, 0.0),
                    (b.rect.w, 0.0),
                    (0.0, b.rect.h),
                    (b.rect.w, b.rect.h),
                ] {
                    let (tx, ty) = acc.apply(px, py);
                    ox = ox.min(tx);
                    oy = oy.min(ty);
                    ex = ex.max(tx);
                    ey = ey.max(ty);
                }
            }
        }
        if !skip.contains(&(b as *const LayoutBox)) {
            // A positioned box that scrolls with the scroller is itself
            // a containing block inside it; so is a box establishing a
            // fixed containing block (transform/filter/will-change/
            // perspective/contain), since absolute descendants take their
            // containing block from it; otherwise the nearest such
            // ancestor above is unchanged.
            let child_cb_inside = pos != style::Position::Static
                || cb_inside
                || b.style.as_ref().is_some_and(|s| establishes_fixed_cb(s));
            // A box establishing a fixed containing block that scrolls
            // with the scroller becomes the fixed-position containing
            // block for the boxes below it; otherwise the nearest such
            // ancestor above is unchanged.
            let child_fixed_cb_inside =
                fixed_cb_inside || b.style.as_ref().is_some_and(|s| establishes_fixed_cb(s));
            stack.extend(b.children.iter().map(|c| {
                (
                    c,
                    child_cb_inside,
                    child_fixed_cb_inside,
                    child_accumulated_transform(&acc, b, c),
                )
            }));
        }
    }
    (ox, oy, ex, ey)
}

/// Find the scroll containers of `root` in preorder (parents before
/// children, siblings in layout order), with the geometry a scroll tree
/// needs per container: `scrollport` (border box), `max` (scrollable
/// overflow extent minus scrollport, floored at zero), `scroll_origin`
/// (minimum content edge minus scrollport, capped at zero), and the
/// [`overflow_axes`] classification.
///
/// Three passes, all iterative (the crate's never-recurse invariant):
/// collect candidates in preorder, confirm deepest-first so a
/// container's extent skips already-confirmed (deeper) containers'
/// subtrees, then restore preorder. Every box with a container
/// `overflow` axis is reported even when nothing currently overflows
/// (its maximum is then zero): this matches browsers, where
/// `overflow: auto` (and `overflow: hidden`) always creates a scroll
/// container.
///
/// This is [`scroll_containers_with_offset`] with a zero viewport
/// scroll offset: prefer that variant when the layout was produced for
/// a scrolled viewport (see [`layout_with_scroll`]), so `position:
/// fixed` containers — laid out in viewport coordinates — are reported
/// in absolute page coordinates like everything else.
pub fn scroll_containers(root: &LayoutBox) -> Vec<ScrollContainerInfo> {
    scroll_containers_with_offset(root, (0.0, 0.0))
}

/// Like [`scroll_containers`], for a layout produced with the viewport
/// scrolled to `scroll` (`(scroll_x, scroll_y)` in CSS px — the same
/// offset passed to [`layout_with_scroll`]).
///
/// In-flow geometry is already in absolute page coordinates, but
/// `position: fixed` boxes (and boxes inside `fixed` subtrees) are
/// viewport-anchored: their reported `scrollport`s are translated by
/// `scroll` into page coordinates. Exception: a fixed box under a
/// fixed-CB trigger (a non-`none` `transform`, a `will-change` naming a
/// stacking-context trigger, a non-`none` `filter`, a non-`none`
/// `perspective`, or `contain: paint | layout | strict | content`) is
/// positioned against that ancestor (CSS Transforms 1 §3; Filter
/// Effects 1; CSS Will Change 1 §3; CSS Transforms 2; CSS Contain 1/2)
/// and scrolls with it, so its rect is already in page coordinates and
/// is *not* translated. `max` is translation-invariant and needs no
/// adjustment.
pub fn scroll_containers_with_offset(
    root: &LayoutBox,
    scroll: (f32, f32),
) -> Vec<ScrollContainerInfo> {
    struct Candidate<'a> {
        box_: &'a LayoutBox,
        axes: (bool, bool),
        user_axes: (bool, bool),
        depth: usize,
        /// The box's rect is viewport-anchored rather than page
        /// coordinates: `position: fixed` against the viewport (or inside
        /// a viewport-anchored fixed subtree). A fixed box under a
        /// fixed-CB trigger is positioned against that ancestor instead
        /// (CSS Transforms 1 §3 et al.) and scrolls with it, so it is
        /// *not* viewport-anchored when that ancestor's own frame is in
        /// page coordinates.
        viewport_anchored: bool,
    }

    // Pass 1: collect candidates in preorder (parents before children,
    // siblings in layout order). Iterative, per the crate's
    // never-recurse invariant.
    let mut candidates: Vec<Candidate> = Vec::new();
    // Per box: whether its own rect is viewport-anchored, and — for
    // resolving a fixed box's frame — the viewport-anchoredness of its
    // nearest strict ancestor establishing a fixed containing block
    // (`None` when there is none).
    let mut stack: Vec<(&LayoutBox, usize, bool, Option<bool>)> = vec![(root, 0, false, None)];
    while let Some((box_, depth, viewport_anchored, cb_frame)) = stack.pop() {
        // Anonymous boxes have no style and can never scroll.
        let fixed = box_
            .style
            .as_ref()
            .is_some_and(|s| style::position_of(s) == style::Position::Fixed);
        let cb_trigger = box_.style.as_ref().is_some_and(|s| establishes_fixed_cb(s));
        // A fixed box lives in its containing block's frame: the
        // viewport's when no fixed-CB trigger intervenes, else the
        // nearest such ancestor's frame (whose own viewport-anchoredness
        // was carried down). Anything else lives in its DOM parent's
        // frame.
        let viewport_anchored = if fixed {
            cb_frame.unwrap_or(true)
        } else {
            viewport_anchored
        };
        // The containing-block frame the children see: this box's own
        // frame when it establishes one, else the one carried down.
        let child_frame = if cb_trigger {
            Some(viewport_anchored)
        } else {
            cb_frame
        };
        if box_.node.is_some()
            && let Some(style) = box_.style.as_ref()
        {
            let (axes, user_axes) = overflow_axes(style);
            if axes.0 || axes.1 {
                candidates.push(Candidate {
                    box_,
                    axes,
                    user_axes,
                    depth,
                    viewport_anchored,
                });
            }
        }
        for child in box_.children.iter().rev() {
            stack.push((child, depth + 1, viewport_anchored, child_frame));
        }
    }
    // Confirm deepest-first: a container's extent skips the subtrees of
    // already-confirmed (deeper) containers.
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(candidates[i].depth));
    let mut confirmed: HashSet<*const LayoutBox> = HashSet::new();
    let mut out: Vec<(usize, ScrollContainerInfo)> = Vec::new();
    for i in order {
        let c = &candidates[i];
        let (ox, oy, ex, ey) = scrollable_extent(c.box_, &confirmed);
        let mx = if c.axes.0 {
            (ex - (c.box_.rect.x + c.box_.rect.w)).max(0.0)
        } else {
            0.0
        };
        let my = if c.axes.1 {
            (ey - (c.box_.rect.y + c.box_.rect.h)).max(0.0)
        } else {
            0.0
        };
        // Deliberately not `direction`-dependent: the minimum x offset is
        // always anchored at the minimum content edge. Offsets shift
        // content by `-offset` uniformly, so negative offsets reveal
        // leftward overflow under both LTR and `direction: rtl` (whose
        // content lays out right-aligned, making leftward overflow the
        // common RTL case). Anchoring at `-max` instead would strand
        // leftward overflow whenever the right edge overflows by less —
        // `max` only measures the right edge.
        let sx = if c.axes.0 {
            (ox - c.box_.rect.x).min(0.0)
        } else {
            0.0
        };
        let sy = if c.axes.1 {
            (oy - c.box_.rect.y).min(0.0)
        } else {
            0.0
        };
        confirmed.insert(c.box_ as *const LayoutBox);
        let mut scrollport = c.box_.rect;
        if c.viewport_anchored {
            scrollport.x += scroll.0;
            scrollport.y += scroll.1;
        }
        out.push((
            i,
            ScrollContainerInfo {
                node: c.box_.node,
                scrollport,
                max: (mx, my),
                scroll_origin: (sx, sy),
                axes: c.axes,
                user_axes: c.user_axes,
                depth: c.depth,
            },
        ));
    }
    // Back to preorder (parents before children, siblings in layout
    // order): the confirmation pass above ran deepest-first.
    out.sort_by_key(|(preorder, _)| *preorder);
    out.into_iter().map(|(_, info)| info).collect()
}

/// The layout tree: the root [`LayoutBox`] produced by [`layout`].
/// (A plain alias — the tree *is* the box hierarchy.)
pub type LayoutTree = LayoutBox;

/// A scroll snap area: one in-flow snap target's snap rectangle (its
/// border box expanded by its `scroll-margin`), in its scroll
/// container's content coordinates, with the per-axis
/// `scroll-snap-align` values.
#[derive(Clone, Copy, Debug)]
pub struct SnapArea {
    pub rect: Rect,
    pub align_x: SnapAlign,
    pub align_y: SnapAlign,
    /// The snap target to report for this area: the area's own box node,
    /// retargeted to the shadow host when the box sits inside a shadow
    /// root (per the shadow-DOM event retargeting rule, the page must
    /// see the host, never a node inside a closed shadow tree). `None`
    /// only when the box carries no DOM node at all.
    pub target: Option<NodeId>,
}

/// Resolve an absolute length to px. Relative units (`em`, `%`,
/// viewport units, …) need font/viewport context this query does not
/// carry, so they resolve to zero; `auto` and anything else does too.
fn abs_len_px(value: Option<&CssValue>) -> f32 {
    match value {
        Some(CssValue::Length(v, unit)) => match unit {
            LengthUnit::Px => *v,
            // Absolute units fold to px here (no font/viewport context
            // needed); CSS Values 4 §5.2 anchors `96px = 1in`.
            LengthUnit::Cm => *v * 96.0 / 2.54,
            LengthUnit::Mm => *v * 96.0 / 25.4,
            LengthUnit::In => *v * 96.0,
            LengthUnit::Pt => *v * 96.0 / 72.0,
            LengthUnit::Pc => *v * 16.0,
            LengthUnit::Q => *v * 96.0 / 101.6,
            _ => 0.0,
        },
        _ => 0.0,
    }
}

/// Collect the scroll snap areas of one scroll container: the in-flow
/// descendants (in preorder) whose `scroll-snap-align` is not `none` on
/// at least one axis. Each area's rectangle is the box's border box
/// expanded by its `scroll-margin`, in the container's content
/// coordinates (container border box minus its border and padding).
///
/// Each area also carries its snap `target`: the area's box node,
/// retargeted to the shadow host when the box sits inside a shadow
/// root (the page must never observe a node inside a shadow tree as a
/// snap target). `doc` supplies the shadow-root mapping; it is also
/// the document the layout tree was built from.
///
/// Mirrors [`scroll_containers`]: a viewport-anchored `position: fixed`
/// subtree never participates, but a fixed box under a fixed-CB trigger
/// between it and the container takes its containing block from that
/// ancestor (CSS Transforms 1 §3 et al.), scrolls with the container's
/// content, and is treated like a normal in-flow box (mirrors the
/// `fixed_cb_inside` tracking in [`scrollable_extent`]). An absolutely
/// positioned box only participates when its containing block — the
/// nearest positioned or fixed-CB-establishing ancestor — is inside the
/// container, except for the viewport scroller (the tree root box),
/// where the ICB counts as inside: an abspos box with no positioned
/// ancestor scrolls with the document and is a legitimate snap area.
/// Iterative, per the crate's never-recurse invariant. An
/// unknown `container_node` yields no areas.
pub fn snap_areas_for_container(
    tree: &LayoutTree,
    container_node: NodeId,
    doc: &Document,
) -> Vec<SnapArea> {
    // Locate the container box by node (iterative, preorder).
    let mut stack: Vec<&LayoutBox> = vec![tree];
    let container = loop {
        match stack.pop() {
            None => return Vec::new(),
            Some(b) if b.node == Some(container_node) => break b,
            Some(b) => stack.extend(b.children.iter()),
        }
    };
    // Container content-box origin, in page coordinates.
    let cstyle = container.style.as_deref();
    let get = |prop: u8| cstyle.and_then(|s| s.get_by_index(prop));
    let ox = container.rect.x
        + abs_len_px(get(props::BORDER_LEFT_WIDTH))
        + abs_len_px(get(props::PADDING_LEFT));
    let oy = container.rect.y
        + abs_len_px(get(props::BORDER_TOP_WIDTH))
        + abs_len_px(get(props::PADDING_TOP));
    // The container itself is a containing block for abspos descendants
    // when it is positioned. The viewport scroller — the tree root box —
    // is the exception: an abspos box with no positioned ancestor takes
    // its containing block from the ICB, which scrolls with the document,
    // so it is a legitimate snap area. (For a nested scroller the ICB
    // lies outside, so the seed stays false there.)
    let is_viewport = std::ptr::eq(container, tree);
    let container_cb = is_viewport || position_of_box(container) != style::Position::Static;
    // Preorder walk: children pushed reversed so `pop` yields them in
    // layout order. Tracks whether the nearest containing block above
    // is inside the container (mirrors `scrollable_extent`).
    let mut out = Vec::new();
    // A fixed-CB trigger on the container is NOT a fixed containing
    // block inside itself: a fixed child is laid out against the
    // container's padding box, i.e. the scrollport frame, which does not
    // move when the container scrolls — so it is still
    // viewport-anchored for snap purposes. Only a trigger box strictly
    // below the container starts the flag as true (mirrors
    // `scrollable_extent`).
    let mut walk: Vec<(&LayoutBox, bool, bool)> = container
        .children
        .iter()
        .rev()
        .map(|c| (c, container_cb, false))
        .collect();
    while let Some((b, cb_inside, fixed_cb_inside)) = walk.pop() {
        let pos = position_of_box(b);
        // Viewport-anchored: never a snap target of this container. A
        // fixed box under a fixed-CB trigger between it and the
        // container takes its containing block from that ancestor
        // (CSS Transforms 1 §3 et al.), scrolls with the container's
        // content, and is a legitimate snap target.
        if pos == style::Position::Fixed && !fixed_cb_inside {
            continue;
        }
        // Absolutely positioned against a containing block outside the
        // container: does not scroll with it.
        if pos == style::Position::Absolute && !cb_inside {
            continue;
        }
        if let Some(style) = b.style.as_deref() {
            // `scroll_snap_align()` returns (block, inline) in the target's
            // own writing mode; the engine consumes physical x/y, so map
            // the logical axes onto physical ones. In horizontal writing
            // modes the inline axis is horizontal and block is vertical;
            // in vertical writing modes they swap (mirrors how the cascade
            // folds `inline-size`/`block-size` onto `width`/`height`).
            let (block, inline) = style.scroll_snap_align();
            let (align_x, align_y) = match style::writing_mode_of(style) {
                style::WritingMode::HorizontalTb => (inline, block),
                _ => (block, inline),
            };
            if align_x != SnapAlign::None || align_y != SnapAlign::None {
                let mt = abs_len_px(style.get_by_index(props::SCROLL_MARGIN_TOP));
                let mr = abs_len_px(style.get_by_index(props::SCROLL_MARGIN_RIGHT));
                let mb = abs_len_px(style.get_by_index(props::SCROLL_MARGIN_BOTTOM));
                let ml = abs_len_px(style.get_by_index(props::SCROLL_MARGIN_LEFT));
                let r = b.rect;
                // Shadow-descendant snap areas retarget to the host:
                // the reported target is never a node inside a shadow
                // tree.
                let target = b.node.map(|n| doc.shadow_host_for(n).unwrap_or(n));
                out.push(SnapArea {
                    rect: Rect {
                        x: sane_len(r.x - ml - ox),
                        y: sane_len(r.y - mt - oy),
                        w: sane_len(r.w + ml + mr),
                        h: sane_len(r.h + mt + mb),
                    },
                    align_x,
                    align_y,
                    target,
                });
            }
        }
        // A positioned box that scrolls with the container is itself a
        // containing block inside it; so is a box establishing a fixed
        // containing block, since absolute descendants take their
        // containing block from it; otherwise the nearest such ancestor
        // above is unchanged.
        let child_cb_inside = pos != style::Position::Static
            || cb_inside
            || b.style.as_ref().is_some_and(|s| establishes_fixed_cb(s));
        // A box establishing a fixed containing block that scrolls with
        // the container becomes the fixed-position containing block for
        // the boxes below it; otherwise the nearest such ancestor above
        // is unchanged.
        let child_fixed_cb_inside =
            fixed_cb_inside || b.style.as_ref().is_some_and(|s| establishes_fixed_cb(s));
        walk.extend(
            b.children
                .iter()
                .rev()
                .map(|c| (c, child_cb_inside, child_fixed_cb_inside)),
        );
    }
    out
}

/// Lay out only the subtree rooted at `subtree_root`, given the
/// `containing_width` its parent formatting context offered it in the
/// previous whole-document layout (wave 51 incremental layout).
///
/// Mirrors [`layout_with_intrinsics_and_scroll_and_offsets`]' argument
/// set for the subtree root: same author sheets, viewport (for the
/// cascade, media queries, and `vh`), intrinsics, and scroll offsets
/// (sticky constraints inside the subtree bake at the same offsets a
/// full pass would use).
///
/// Returns `None` when the subtree has no rendered root (see
/// `boxes::build_box_tree_subtree_with_fonts`).
///
/// Coordinate frame: the returned tree is laid out as if the subtree
/// root's containing block origin were `(0, 0)` — its `rect`s are
/// relative to that origin, *not* absolute page coordinates. The
/// caller (the app's splice path) compares the root's size and its
/// position relative to the containing block against the previous
/// tree, and translates the whole returned tree by the previous box's
/// containing-block origin before splicing it in.
///
/// Viewport units resolve against the *real* viewport in this pass,
/// exactly as in a whole-document pass: the forest keeps the true
/// `viewport` in its viewport fields (the base every `vw`/`vh`/
/// `vmin`/`vmax` resolution in the layout passes reads), and
/// `containing_width` travels separately as the root's
/// containing-block width (see
/// `layout::layout_forest_with_root_cb`). The two quantities were
/// previously conflated by overwriting the forest's viewport width
/// with `containing_width`, which mis-measured viewport units inside
/// the subtree (a grid `25vw` track in a 400px containing block in an
/// 800px viewport sized against 400px); the app-side caller still
/// scans subtree styles for viewport units as defense in depth, but
/// correctness no longer depends on that scan. One residual base
/// difference remains, by design of the entry's argument set: the
/// root's containing-block *height* is the viewport height (the
/// parent's height is not passed in), so a subtree root with a
/// percentage height can measure differently here — the app-side
/// caller gates percentage heights on the candidate root back to the
/// full path. Custom properties need no special handling here: the
/// cascade substitutes `var()` at computed-value time against the
/// real viewport, so a `vw` hidden in a custom property reaches layout
/// already substituted into the consuming property, in both passes
/// identically.
// Mirrors the whole-document entry's argument set plus the subtree
// root and its containing width: the established shape for this
// crate's layout entries, hence the argument count.
#[allow(clippy::too_many_arguments)]
pub fn layout_subtree_with_intrinsics_and_scroll_and_offsets(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn FontMetrics,
    intrinsics: &HashMap<bowser_dom::NodeId, IntrinsicDims>,
    scroll: (f32, f32),
    nested_offsets: &HashMap<bowser_dom::NodeId, (f32, f32)>,
    subtree_root: bowser_dom::NodeId,
    containing_width: f32,
) -> Option<LayoutBox> {
    let ua: &Stylesheet = ua_stylesheet();
    let viewport = Size {
        w: style::sane_len(viewport.w).max(0.0),
        h: style::sane_len(viewport.h).max(0.0),
    };
    let (mut forest, root) = boxes::build_box_tree_subtree_with_fonts(
        doc,
        author,
        ua,
        viewport.w,
        viewport.h,
        intrinsics,
        subtree_root,
        fonts,
    )?;
    // The forest's viewport fields already hold the true viewport
    // (from `build_box_tree_subtree_with_fonts` above): do NOT overwrite them
    // with the containing width. The root's containing-block width
    // goes to the layout entry as its own quantity instead.
    let containing_width = style::sane_len(containing_width).max(0.0);
    layout::layout_forest_with_root_cb(&mut forest, root, fonts, Some(containing_width));
    let (scroll_x, scroll_y) = (
        style::sane_len(scroll.0).max(0.0),
        style::sane_len(scroll.1).max(0.0),
    );
    layout::apply_sticky_constraints(&mut forest, scroll_x, scroll_y, nested_offsets);
    Some(layout::convert_forest(&mut forest, root).0)
}

pub(crate) fn layout_impl(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn FontMetrics,
    intrinsics: &HashMap<bowser_dom::NodeId, IntrinsicDims>,
    scroll: (f32, f32),
    nested_offsets: &HashMap<bowser_dom::NodeId, (f32, f32)>,
) -> LayoutBox {
    layout_impl_with_markers(
        doc,
        author,
        viewport,
        fonts,
        intrinsics,
        scroll,
        nested_offsets,
    )
    .0
}

/// Like [`layout_impl`], but additionally returns the
/// `list-style-position: outside` markers (see [`MarkerBox`]), each
/// tagged with its owning list item's DOM node. The public entry
/// points drop the markers; paint joins them back to boxes via
/// [`LayoutBox::node`] when it needs them.
pub(crate) fn layout_impl_with_markers(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn FontMetrics,
    intrinsics: &HashMap<bowser_dom::NodeId, IntrinsicDims>,
    scroll: (f32, f32),
    nested_offsets: &HashMap<bowser_dom::NodeId, (f32, f32)>,
) -> (LayoutBox, Vec<MarkerBox>) {
    layout_impl_with_markers_inner(
        doc,
        author,
        viewport,
        fonts,
        intrinsics,
        scroll,
        nested_offsets,
        None,
    )
}

/// `layout_impl_with_markers` with an explicit root containing-block
/// rect: `Some((x, w))` re-runs the pass against an already-narrowed
/// initial containing block (the viewport `overflow-y: auto`
/// scrollbar re-run — see [`viewport_auto_scrollbar_rect`]).
// Mirrors the whole-document entry's argument set plus the re-run
// rect: the established shape for this crate's layout entries, hence
// the argument count.
#[allow(clippy::too_many_arguments)]
fn layout_impl_with_markers_inner(
    doc: &Document,
    author: &[&Stylesheet],
    viewport: Size,
    fonts: &dyn FontMetrics,
    intrinsics: &HashMap<bowser_dom::NodeId, IntrinsicDims>,
    scroll: (f32, f32),
    nested_offsets: &HashMap<bowser_dom::NodeId, (f32, f32)>,
    root_cb_rect: Option<(f32, f32)>,
) -> (LayoutBox, Vec<MarkerBox>) {
    let ua: &Stylesheet = ua_stylesheet();
    // Sanitize the viewport once: NaN/negative/infinite dimensions from
    // the caller degrade to finite nonnegative instead of poisoning
    // every length resolved against them.
    let viewport = Size {
        w: style::sane_len(viewport.w).max(0.0),
        h: style::sane_len(viewport.h).max(0.0),
    };
    let Some((mut forest, root)) = boxes::build_box_tree_with_fonts(
        doc, author, ua, viewport.w, viewport.h, intrinsics, fonts,
    ) else {
        return (
            LayoutBox {
                rect: Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 0.0,
                    h: 0.0,
                },
                children: Vec::new(),
                box_type: BoxType::Block,
                style: None,
                text: None,
                node: None,
                fragments: Vec::new(),
                z_index: None,
                collapsed_borders: None,
                phantom_w: false,
                phantom_h: false,
            },
            Vec::new(),
        );
    };
    // Boxes built from the document element / body never take the
    // `overflow-y: auto` scrollbar-width re-drive in `resume_block`:
    // the app never paints a nested scrollbar for them (they are the
    // viewport scroller), so their vertical scrollbar geometry belongs
    // to the viewport pass below.
    let viewport_scrollers = (
        doc.document_element(),
        doc.get_elements_by_tag_name("body").item(0),
    );
    layout::layout_forest_document(&mut forest, root, fonts, root_cb_rect, viewport_scrollers);
    // Sticky constraints need final rects: `layout_forest` flushed every
    // deferred shift into `rect`, so the pass below works in document
    // coordinates. Layout never scrolls nested containers itself — the
    // `scroll` argument drives only the viewport scroller; nested
    // containers' live offsets arrive via `nested_offsets` (empty for
    // every entry point but `layout_with_scroll_and_offsets` and
    // `layout_with_intrinsics_and_scroll_and_offsets`). Layout reports
    // every nested scroll container's geometry via [`scroll_containers`],
    // so the app builds its scroll tree from real layout output instead
    // of the viewport alone.
    let (scroll_x, scroll_y) = (
        style::sane_len(scroll.0).max(0.0),
        style::sane_len(scroll.1).max(0.0),
    );
    layout::apply_sticky_constraints(&mut forest, scroll_x, scroll_y, nested_offsets);
    let (root_box, markers) = layout::convert_forest(&mut forest, root);
    // CSS Overflow 3 §2.2/§2.4: an `overflow-y: auto`/`scroll` viewport
    // shows the classic vertical scrollbar exactly when the document
    // overflows it (`scroll` forces it; the re-run only fires on real
    // overflow, when the app paints), and the shown scrollbar consumes
    // its thickness from the initial containing block — but the first
    // pass sized the root against the full viewport, so the painted
    // scrollbar overlays the last SCROLLBAR_THICKNESS px of root
    // content. Re-run once against the narrowed ICB (bounded: the
    // re-run never re-runs).
    if root_cb_rect.is_none()
        && let Some(rect) =
            viewport_auto_scrollbar_rect(doc, forest.root_is_html, &root_box, viewport)
    {
        return layout_impl_with_markers_inner(
            doc,
            author,
            viewport,
            fonts,
            intrinsics,
            scroll,
            nested_offsets,
            Some(rect),
        );
    }
    (root_box, markers)
}

/// The body's computed style, for viewport `overflow` propagation —
/// mirrors `bowser_app::scroll::body_style`: the style of the layout
/// box built for the `<body>` element, if the document has one and it
/// produced a box.
///
/// Iterative: the tree can be arbitrarily deep.
fn body_style<'a>(doc: &Document, root: &'a LayoutBox) -> Option<&'a bowser_css::ComputedStyle> {
    let body = doc.get_elements_by_tag_name("body").item(0)?;
    let mut stack = vec![root];
    while let Some(box_) = stack.pop() {
        if box_.node == Some(body) {
            return box_.style.as_deref();
        }
        stack.extend(box_.children.iter());
    }
    None
}

/// The viewport scroller's used `overflow-y` (CSS Overflow 3 §2.4),
/// mirroring `bowser_app::scroll::viewport_propagated_axes`: the root
/// element's used value wins unless both axes are `visible` — then the
/// body's used value propagates instead. A both-`visible` source
/// behaves as `auto` on the viewport itself (it scrolls on overflow).
fn viewport_propagated_overflow_y<'a>(doc: &'a Document, root: &'a LayoutBox) -> &'a str {
    fn both_visible(style: &bowser_css::ComputedStyle) -> bool {
        let ((cx, cy), _) = overflow_axes(style);
        !cx && !cy
    }
    /// Used `overflow-y` with the CSS Overflow 3 §2 visible→auto
    /// pairing applied (`visible` computes to `auto` when the other
    /// axis is non-`visible`).
    fn used_y(style: &bowser_css::ComputedStyle) -> &str {
        let oy = style
            .get_str_by_index(props::OVERFLOW_Y)
            .unwrap_or("visible");
        if oy.eq_ignore_ascii_case("visible") {
            let ox = style
                .get_str_by_index(props::OVERFLOW_X)
                .unwrap_or("visible");
            if !ox.eq_ignore_ascii_case("visible") {
                return "auto";
            }
        }
        oy
    }
    let propagated = match root.style.as_deref() {
        // The root element's values win unless both are `visible`.
        Some(s) if !both_visible(s) => Some(s),
        // Root is `visible`/`visible` (or styless): the body's values
        // propagate; without a usable body everything stays scrollable.
        _ => body_style(doc, root).filter(|s| !both_visible(s)),
    };
    // Both `visible`: behaves as `auto` on the viewport.
    propagated.map_or("auto", used_y)
}

/// Document scrollable-overflow range for the viewport scroller —
/// `((min_x, min_y), (max_x, max_y))`: the scroll origin (floored at
/// zero) and each maximum as the overflow extent minus the viewport
/// size (floored at zero). Mirrors
/// `bowser_app::scroll::ScrollTree::document_max_scroll`: the union of
/// in-flow border boxes, skipping `position: fixed` subtrees
/// (viewport-frame, never document overflow) and nested scroll
/// containers (their overflowing content is clipped by their own
/// scrollport and never extends the viewport's range). The document
/// element and `<body>` are the viewport scroller itself, never nested
/// containers.
fn viewport_max_scroll(
    doc: &Document,
    root: &LayoutBox,
    viewport: Size,
) -> ((f32, f32), (f32, f32)) {
    let document_element = doc.document_element();
    let body = doc.get_elements_by_tag_name("body").item(0);
    let nested: HashSet<NodeId> = scroll_containers(root)
        .into_iter()
        .filter_map(|c| c.node)
        .filter(|n| Some(*n) != document_element && Some(*n) != body)
        .collect();
    let (mut nx, mut ny) = (0.0f32, 0.0f32);
    let (mut ex, mut ey) = (0.0f32, 0.0f32);
    let mut stack = vec![(root, false)];
    while let Some((box_, fixed_above)) = stack.pop() {
        let fixed = fixed_above
            || box_
                .style
                .as_ref()
                .is_some_and(|s| style::position_of(s) == style::Position::Fixed);
        if !fixed {
            nx = nx.min(box_.rect.x);
            ny = ny.min(box_.rect.y);
            ex = ex.max(box_.rect.x + box_.rect.w);
            ey = ey.max(box_.rect.y + box_.rect.h);
        }
        let is_nested = box_.node.is_some_and(|n| nested.contains(&n));
        if !is_nested {
            stack.extend(box_.children.iter().map(|c| (c, fixed)));
        }
    }
    (
        (nx.min(0.0), ny.min(0.0)),
        ((ex - viewport.w).max(0.0), (ey - viewport.h).max(0.0)),
    )
}

/// The narrowed root containing-block `(x, w)` for the viewport
/// `overflow-y` scrollbar re-run, or `None` when no re-run is needed:
/// the propagated used `overflow-y` is `auto` or `scroll`, the writing
/// mode is horizontal, and the document really overflows the viewport
/// vertically — the same condition the app keys the painted scrollbar
/// off (`user && max > min`: `max_y > 0`, or a negative scroll origin
/// `min_y < 0` with `max_y == 0`).
///
/// The strip sits at the inline-end edge (the left edge under root
/// `direction: rtl`, mirroring the app's viewport scrollbar track)
/// and takes the per-side max with the propagated `scrollbar-gutter`
/// reservation — a shown scrollbar paints exactly the reserved strip,
/// so the two never stack. `None` when the gutter already reserves
/// the strip.
fn viewport_auto_scrollbar_rect(
    doc: &Document,
    root_is_html: bool,
    root: &LayoutBox,
    viewport: Size,
) -> Option<(f32, f32)> {
    let oy = viewport_propagated_overflow_y(doc, root);
    // `scroll` forces the classic scrollbar whether or not content
    // overflows; the first pass never reserved its strip against the
    // viewport (only the stable gutter is propagated there), so a
    // scrolled document needs the same narrowing as `auto`.
    if !(oy.eq_ignore_ascii_case("auto") || oy.eq_ignore_ascii_case("scroll")) {
        return None;
    }
    // Same writing-mode bail as the box-level reservations: under a
    // vertical writing mode this physical left/right strip cannot
    // express the inline-end edge.
    let horizontal = root
        .style
        .as_ref()
        .is_none_or(|s| matches!(style::writing_mode_of(s), style::WritingMode::HorizontalTb));
    if !horizontal {
        return None;
    }
    let ((_, min_y), (_, max_y)) = viewport_max_scroll(doc, root, viewport);
    if max_y <= 0.0 && min_y >= 0.0 {
        return None;
    }
    // The reservation the first pass already applied against the
    // viewport, mirroring `layout_forest`'s prologue exactly.
    let (gl, gr) = match root.style.as_ref() {
        Some(s) if root_is_html => viewport_gutter_reservation(s),
        _ => (0.0, 0.0),
    };
    let rtl = root
        .style
        .as_ref()
        .and_then(|s| s.get_str_by_index(props::DIRECTION))
        .is_some_and(|d| d.eq_ignore_ascii_case("rtl"));
    let (sl, sr) = if rtl {
        (SCROLLBAR_THICKNESS, 0.0)
    } else {
        (0.0, SCROLLBAR_THICKNESS)
    };
    let (nl, nr) = (gl.max(sl), gr.max(sr));
    if nl <= gl && nr <= gr {
        // The stable gutter already reserves the strip the shown
        // scrollbar paints: nothing more to consume.
        return None;
    }
    Some((nl, (viewport.w - nl - nr).max(0.0)))
}

#[cfg(test)]
mod transform_extent_tests {
    //! Pinning tests for transform-aware [`scrollable_extent`]: the
    //! scrollable overflow region unions descendants' border boxes
    //! *after* their `transform`s (CSS Overflow 3 §3).

    use super::{FixedFontMetrics, Size, layout, scroll_containers};

    const VP: Size = Size { w: 800.0, h: 600.0 };

    /// The `ScrollContainerInfo` of `#s` for a tiny document.
    fn container(css: &str, html: &str) -> super::ScrollContainerInfo {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let root = layout(&doc, &[&ss], VP, &FixedFontMetrics);
        let s = doc.get_element_by_id("s").expect("s element");
        scroll_containers(&root)
            .into_iter()
            .find(|c| c.node == Some(s))
            .expect("s container")
    }

    const DOC: &str = "<html><body><div id=\"s\"><div id=\"c\">x</div></div></body></html>";
    const BASE: &str = "body { margin: 0; } \
         #s { overflow: auto; width: 100px; height: 100px; } ";

    #[test]
    fn extent_translate_extends_max() {
        // A 100px child exactly filling the 100px scrollport,
        // translated 50px right: the region grows 50px on the right.
        let c = container(
            &format!("{BASE} #c {{ width: 100px; height: 100px; transform: translate(50px, 0); }}"),
            DOC,
        );
        assert_eq!(c.max, (50.0, 0.0));
        assert_eq!(c.scroll_origin, (0.0, 0.0));
    }

    #[test]
    fn extent_translate_negative_extends_origin() {
        // The "unreachable region" shape: a child translated to
        // negative coordinates. Its bounds must extend the region's
        // minimum edges (the scroll origin), not be dropped.
        let c = container(
            &format!(
                "{BASE} #c {{ width: 20px; height: 20px; transform: translate(-30px, -40px); }}"
            ),
            DOC,
        );
        assert_eq!(c.max, (0.0, 0.0));
        assert_eq!(c.scroll_origin, (-30.0, -40.0));
    }

    #[test]
    fn extent_scale_about_center_origin() {
        // scale(2) about the default 50% 50% origin: a 100x100 child
        // grows to (-50,-50)..(150,150) — 50px of overflow on every
        // side.
        let c = container(
            &format!("{BASE} #c {{ width: 100px; height: 100px; transform: scale(2); }}"),
            DOC,
        );
        assert_eq!(c.max, (50.0, 50.0));
        assert_eq!(c.scroll_origin, (-50.0, -50.0));
    }

    #[test]
    fn extent_rotate_90_swaps_axes() {
        // A 100x50 child rotated 90deg about its center (50,25): the
        // transformed bounds are x 25..75, y -25..75 — the tall axis
        // now overflows upward by 25px.
        let c = container(
            &format!("{BASE} #c {{ width: 100px; height: 50px; transform: rotate(90deg); }}"),
            DOC,
        );
        assert_eq!(c.max, (0.0, 0.0));
        // sin/cos of 90deg in f32 leaves ~1e-6 of dust on the edges.
        assert!(
            (c.scroll_origin.0 - 0.0).abs() < 0.001 && (c.scroll_origin.1 + 25.0).abs() < 0.001,
            "scroll_origin {:?}",
            c.scroll_origin
        );
    }

    #[test]
    fn extent_nested_transforms_compose() {
        // Outer translate(60px, 0) wrapping an inner child laid out at
        // (50,50), 60x60, with its own translate(0, 30px): the inner
        // box lands at (110,80)..(170,140), so the region's far edges
        // come from the composed transform (max x 70, max y 40),
        // beyond the outer box's own contribution (max x 60).
        let c = container(
            &format!(
                "{BASE} \
                 #a {{ width: 100px; height: 100px; transform: translate(60px, 0); }} \
                 #b {{ position: absolute; left: 50px; top: 50px; \
                       width: 60px; height: 60px; transform: translate(0, 30px); }}"
            ),
            "<html><body><div id=\"s\"><div id=\"a\"><div id=\"b\">x</div></div></div></body></html>",
        );
        assert_eq!(c.max, (70.0, 40.0));
    }

    #[test]
    fn extent_transformed_nested_container_contributes_own_box_only() {
        // A nested scroll container contributes only its own
        // (transformed) box to the outer region; its overflowing
        // content stays clipped by its own scrollport. The nested
        // 50x50 box translated 120px right reaches x=170 (max 70),
        // while its 500px child must not leak into the outer region.
        let c = container(
            &format!(
                "{BASE} \
                 #n {{ overflow: auto; width: 50px; height: 50px; \
                       transform: translate(120px, 0); }} \
                 #deep {{ width: 500px; height: 500px; }}"
            ),
            "<html><body><div id=\"s\"><div id=\"n\"><div id=\"deep\">x</div></div></div></body></html>",
        );
        assert_eq!(c.max, (70.0, 0.0));
    }

    #[test]
    fn extent_untransformed_child_unchanged() {
        // Control: without transforms the region is the plain union of
        // border boxes (existing behavior).
        let c = container(
            &format!("{BASE} #c {{ width: 150px; height: 130px; }}"),
            DOC,
        );
        assert_eq!(c.max, (50.0, 30.0));
        assert_eq!(c.scroll_origin, (0.0, 0.0));
    }
}

#[cfg(test)]
mod subtree_layout_tests {
    //! Differential tests for the subtree layout entry (wave 51): a
    //! standalone subtree pass must reproduce the corresponding box of
    //! the whole-document pass (size, and position relative to the
    //! containing block), including inherited font sizes from
    //! ancestors outside the subtree.

    use super::{
        FixedFontMetrics, LayoutBox, Size, layout,
        layout_subtree_with_intrinsics_and_scroll_and_offsets,
    };
    use std::collections::HashMap;

    const VP: Size = Size { w: 800.0, h: 600.0 };

    fn find(box_: &LayoutBox, node: bowser_dom::NodeId) -> Option<&LayoutBox> {
        if box_.node == Some(node) {
            return Some(box_);
        }
        box_.children.iter().find_map(|c| find(c, node))
    }

    fn find_parent(box_: &LayoutBox, node: bowser_dom::NodeId) -> Option<&LayoutBox> {
        if box_.children.iter().any(|c| c.node == Some(node)) {
            return Some(box_);
        }
        box_.children.iter().find_map(|c| find_parent(c, node))
    }

    #[test]
    fn subtree_layout_matches_full_layout_geometry() {
        // The wrapper sets a non-default font size: the subtree root
        // inherits it, so a pass that ignored ancestor context would
        // measure the text at 16px instead of 32px and fail the height
        // comparison.
        let (doc, _) = bowser_dom::Document::parse_html(
            b"<html><body><div id=\"wrap\"><p id=\"leaf\">hello world</p>\
              <p id=\"sib\">other</p></div></body></html>",
        );
        let ss = bowser_css::parse_stylesheet(
            "body { margin: 0; } #wrap { font-size: 32px; padding: 10px; } \
             p { margin: 0 0 6px 0; }",
        );
        let fonts = FixedFontMetrics;
        let full = layout(&doc, &[&ss], VP, &fonts);
        let wrap = doc.get_element_by_id("wrap").expect("wrap");
        let leaf = doc.get_element_by_id("leaf").expect("leaf");
        let full_leaf = find(&full, leaf).expect("leaf box in full layout");
        let full_parent = find_parent(&full, leaf).expect("leaf parent in full layout");
        assert_eq!(full_parent.node, Some(wrap));
        // The containing width the full pass offered the leaf: the
        // wrapper's content width (padding 10px per side, no borders).
        let containing_width = full_parent.rect.w - 20.0;

        let sub = layout_subtree_with_intrinsics_and_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &fonts,
            &HashMap::new(),
            (0.0, 0.0),
            &HashMap::new(),
            leaf,
            containing_width,
        )
        .expect("subtree layout");
        assert_eq!(sub.node, Some(leaf));
        assert_eq!(sub.box_type, full_leaf.box_type);
        // Same border-box size as the full pass.
        assert_eq!(sub.rect.w, full_leaf.rect.w);
        assert_eq!(sub.rect.h, full_leaf.rect.h);
        // Same position relative to the containing block (the wrapper's
        // content origin): the subtree frame puts that origin at 0,0,
        // and the wrapper's content origin in the full tree is its
        // border box + 10px padding (body margin is 0).
        assert_eq!(sub.rect.x, full_leaf.rect.x - (full_parent.rect.x + 10.0));
        assert_eq!(sub.rect.y, full_leaf.rect.y - (full_parent.rect.y + 10.0));
        // The subtree's descendants came along.
        assert_eq!(sub.text, full_leaf.text);
        assert_eq!(sub.children.len(), full_leaf.children.len());
    }

    /// Compare a subtree pass against the full pass for `id` in a
    /// document whose wrapper is 400px wide in the 800px viewport:
    /// returns `(full_box, sub_box)` border-box sizes.
    fn subtree_vs_full(html: &str, css: &str, id: &str) -> ((f32, f32), (f32, f32)) {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let fonts = FixedFontMetrics;
        let full = layout(&doc, &[&ss], VP, &fonts);
        let target = doc.get_element_by_id(id).expect("target");
        let full_box = find(&full, target).expect("target box in full layout");
        let parent = find_parent(&full, target).expect("target parent in full layout");
        // The wrapper carries no padding/border in these fixtures, so
        // its border-box width is the content width it offered.
        let containing_width = parent.rect.w;
        let sub = layout_subtree_with_intrinsics_and_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &fonts,
            &HashMap::new(),
            (0.0, 0.0),
            &HashMap::new(),
            target,
            containing_width,
        )
        .expect("subtree layout");
        ((full_box.rect.w, full_box.rect.h), (sub.rect.w, sub.rect.h))
    }

    #[test]
    fn subtree_layout_resolves_vw_against_real_viewport() {
        // Regression (wave 51 rework F3): the subtree pass overwrote
        // the forest viewport width with the 400px containing width,
        // so `width: 25vw` measured 100px instead of 200px (25% of
        // the real 800px viewport).
        let (full, sub) = subtree_vs_full(
            "<html><body><div id=\"wrap\"><p id=\"leaf\">hi</p></div></body></html>",
            "body { margin: 0; } #wrap { width: 400px; } \
             #leaf { width: 25vw; height: 10px; margin: 0; }",
            "leaf",
        );
        assert_eq!(full, (200.0, 10.0));
        assert_eq!(sub, full, "subtree vw must use the real viewport");
    }

    #[test]
    fn subtree_layout_resolves_calc_vw_and_grid_vw_track() {
        // The reviewer's probe shape: a grid subtree in a 400px
        // wrapper in an 800px viewport, `grid-template-columns:
        // 1fr 25vw 100px`. Pre-fix the subtree pass resolved 25vw
        // against 400px (100px) and the 1fr track absorbed the error
        // (200px instead of 100px).
        let (doc, _) = bowser_dom::Document::parse_html(
            b"<html><body><div id=\"wrap\"><div id=\"grid\">\
              <div class=\"c\">a</div><div class=\"c\">b</div><div class=\"c\">c</div>\
              </div></div></body></html>",
        );
        let ss = bowser_css::parse_stylesheet(
            "body { margin: 0; } #wrap { width: 400px; } \
             #grid { display: grid; grid-template-columns: 1fr 25vw 100px; } \
             .c { height: 10px; } \
             #grid .c:first-child { width: calc(10px + 5vw); }",
        );
        let fonts = FixedFontMetrics;
        let full = layout(&doc, &[&ss], VP, &fonts);
        let grid = doc.get_element_by_id("grid").expect("grid");
        let full_grid = find(&full, grid).expect("grid box in full layout");
        let full_widths: Vec<f32> = full_grid.children.iter().map(|c| c.rect.w).collect();
        assert_eq!(
            full_widths,
            vec![50.0, 200.0, 100.0],
            "full pass: calc(10px + 5vw)=50px item, 25vw=200px track, 100px track"
        );
        let wrap = doc.get_element_by_id("wrap").expect("wrap");
        let containing_width = find(&full, wrap).expect("wrap box").rect.w;
        let sub = layout_subtree_with_intrinsics_and_scroll_and_offsets(
            &doc,
            &[&ss],
            VP,
            &fonts,
            &HashMap::new(),
            (0.0, 0.0),
            &HashMap::new(),
            grid,
            containing_width,
        )
        .expect("subtree layout");
        let sub_widths: Vec<f32> = sub.children.iter().map(|c| c.rect.w).collect();
        assert_eq!(sub_widths, full_widths, "subtree grid tracks match full");
        assert_eq!(
            (sub.rect.w, sub.rect.h),
            (full_grid.rect.w, full_grid.rect.h)
        );
    }

    #[test]
    fn subtree_layout_rejects_non_element_and_detached_roots() {
        let (mut doc, _) =
            bowser_dom::Document::parse_html(b"<html><body><p id=\"x\">hi</p></body></html>");
        let ss = bowser_css::parse_stylesheet("");
        let fonts = FixedFontMetrics;
        let p = doc.get_element_by_id("x").expect("x");
        let text = doc.first_child(p).expect("text");
        // A text node has no element box of its own.
        assert!(
            layout_subtree_with_intrinsics_and_scroll_and_offsets(
                &doc,
                &[&ss],
                VP,
                &fonts,
                &HashMap::new(),
                (0.0, 0.0),
                &HashMap::new(),
                text,
                800.0,
            )
            .is_none()
        );
        // A detached element is not in the rendered document.
        let detached = doc.create_element("div");
        assert!(
            layout_subtree_with_intrinsics_and_scroll_and_offsets(
                &doc,
                &[&ss],
                VP,
                &fonts,
                &HashMap::new(),
                (0.0, 0.0),
                &HashMap::new(),
                detached,
                800.0,
            )
            .is_none()
        );
    }
}

#[cfg(test)]
mod w65_p_layout_tests {
    use super::*;

    /// Wave-65 budget tripwire (test form of the `const` assert above):
    /// `LayoutBox` is stored inline in the parent's `children` vec, so
    /// the fragment smallvec work must not grow it past 128 bytes.
    #[test]
    fn layout_box_stays_within_128b() {
        assert!(
            std::mem::size_of::<LayoutBox>() <= 128,
            "LayoutBox grew past its size budget: {} bytes",
            std::mem::size_of::<LayoutBox>()
        );
    }

    /// `TextFragment`'s size is load-bearing for `FragmentList`'s
    /// inline slot (boxes.rs): pin it so growth is deliberate.
    #[test]
    fn text_fragment_size_pinned() {
        assert_eq!(std::mem::size_of::<TextFragment>(), 40);
    }
}
