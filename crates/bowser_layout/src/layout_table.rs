//! Table layout (`display: table`).
//!
//! The table → row-group → row → cell structure resolves in two phases on
//! the shared task stack. [`layout_table`] collects the grid, resolves the
//! table's own block-like box, computes automatic column widths, places
//! every cell at a provisional position, and pushes one [`Task::Block`]
//! per cell (plus caption blocks) with the cell's content width as a
//! width override — so cell *contents* lay out through the normal block /
//! inline machinery. [`resume_table`] runs after all cells complete: row
//! heights come from the tallest cell in each row (`rowspan` cells grow
//! the rows they span), short cells stretch to their row height, and rows
//! shift down to their final positions through the deferred-shift
//! mechanism (`record_shift` + the single `apply_pending_shifts` pass).
//!
//! Column widths (automatic table layout, CSS Tables 3): each column's
//! minimum and maximum come from its cells' outer measures (§3.8.3) —
//! a specified `width` floors the *maximum* of a constrained column
//! (`max(min-content, width)`), never the minimum, and a `min-width`
//! floors both. Spanning cells distribute their intrinsic excess over
//! the spanned columns in increasing span order, per column keeping
//! the largest contribution, following the §3.9.3.2 excess rules
//! (non-constrained originating columns first, proportionally to
//! max-content, then constrained, percentage, and equal fallbacks).
//! Used widths come from the §3.9.3 sizing guesses — min-content,
//! min-content-percentage, min-content-specified, max-content —
//! linearly interpolated between the two guesses bracketing the
//! assignable width, with any excess past the max-content guess
//! distributed per §3.9.3.2. Percentage widths are intrinsic
//! percentages (§3.8.3): they floor the table's intrinsic width
//! (each percentage column's measure over its percentage, and the
//! non-percentage measures over the non-percentage share) and steer
//! the guesses, rather than resolving as px minima. An
//! over-constrained total grows the table instead of squeezing
//! columns below min-content — including the degenerate
//! spacing-starved case where `border-spacing` alone meets or
//! exceeds the content width, where the table grows to fit spacing
//! plus columns. The column widths always tile the table's content
//! width exactly (up to the table's own `max-width`, past which the
//! columns overflow instead of growing the table further).
//!
//! Documented simplifications for this pass:
//! - `table-layout: fixed` resolves column widths from the table's width,
//!   the `<col>` / `<colgroup>` widths, and the first row's specified
//!   cell widths (CSS 2.1 §17.5.2.1), without measuring content: a
//!   `col` with a non-`auto` width pins each column in the element's
//!   span, a `colgroup` width applies to each covered column without
//!   its own `col` width (at full value), and first-row cells only
//!   fill columns without one. Columns without a specified width share the remaining table
//!   width equally. An over-constrained set of specified widths grows
//!   the table (the used width is the greater of the specified width
//!   and the columns' sum) — except that percentage pins resolve
//!   against the assignable width (content minus border-spacing) and,
//!   when they mix with fixed-length pins, absorb
//!   (assignable − fixed) proportionally instead of growing the table
//!   (CSS Tables 3 §3.9.3.1).
//! - `border-collapse: collapse` removes the inter-cell spacing and abuts
//!   cell border boxes. The conflicting-border resolution (CSS 2.1
//!   §17.6.2.1) is computed per cell edge by [`resolve_collapsed_borders`]
//!   and exposed on [`TableResume`] as [`CollapsedBorders`], then handed
//!   to each cell's [`crate::LayoutBox::collapsed_borders`], so paint
//!   draws the single winning border on each shared grid line instead of
//!   each cell drawing its own. `separate` (the default) inserts
//!   `border-spacing` around and between cells. Column and column-group
//!   borders claim the vertical grid edges of their columns (element
//!   precedence cell > row > row-group > column > column-group > table);
//!   their claims are folded into the per-cell winners, so paint must
//!   not draw column borders of its own in the collapsing model.
//! - Both `border-spacing` values are honored (horizontal then vertical,
//!   CSS 2.1 §17.5.2); a single value applies to both axes.
//! - `colspan` / `rowspan` place into the grid with occupancy tracking;
//!   `colspan="0"` spans to the end of the table's columns and
//!   `rowspan="0"` spans to the end of the row-group section (the HTML
//!   "span to the end" rule; both degrade to 1 when there is nothing
//!   left to span).
//! - Cells stretch vertically to their row height; `vertical-align:
//!   top | middle | bottom | baseline` on the cell aligns its content
//!   within the row height (`sub`/`super` behave as `baseline`).
//!   Baseline alignment uses each cell's `baseline` field (the last line
//!   box's baseline, or the border-box height when the cell has no line
//!   boxes).
//! - A specified `height` on the table is a minimum for the total
//!   (CSS 2.1 §17.5.3): content taller than it grows the table. Any
//!   excess beyond captions, spacing, and the natural row heights is
//!   distributed per unit (row group, or the bare rows): units without
//!   a specified height and with at least one unsized row absorb it,
//!   sharing proportionally to their heights (evenly when all are
//!   zero-height), and within a unit the share goes to its unsized rows
//!   the same way; when no unit can absorb, the last rowed unit takes
//!   it all, growing even sized rows proportionally. A rowless group
//!   absorbs into its own rect height.
//! - A specified `height` on a row is a minimum for that row's height;
//!   on a row group it is a minimum for the sum of its rows' heights,
//!   distributed over the group's unsized rows proportionally (evenly
//!   when all zero-height), or over all its rows proportionally when
//!   every row is sized (CSS 2.1 §17.5.3). Percentage row heights
//!   resolve against the row group's specified height first, then the
//!   table's; a percentage row/row-group height that would exceed the
//!   table's specified height is capped there. A row counts as *sized*
//!   when it or a cell originating in it has a definite `height`.
//!   Row-group boxes carry their section kind (`thead` / `tbody` /
//!   `tfoot` — see [`SectionKind`], plumbed through
//!   [`TableGrid::group_kinds`]); the percentage-height rules above
//!   still treat all groups alike.
//! - `rowspan` cells spread their height deficit over the spanned rows
//!   by priority (see [`distribute_rowspan_excess`]): percentage rows
//!   grow toward their percentage size, then rows originating other
//!   spanning cells share evenly, then unconstrained non-empty rows
//!   grow proportionally, then (all spanned rows empty) the last row
//!   takes it all, else constrained rows grow proportionally. Spanning
//!   cells are processed innermost-first, taller first among equal
//!   spans. Cells spanning multiple rows do not contribute their
//!   baseline to their originating row's baseline.
//! - An over-constrained column set grows the table past its specified
//!   width (never past its `max-width`; the columns overflow instead),
//!   and a table centered with `auto` side margins grows symmetrically
//!   so it stays centered.
//! - `width` on a cell feeds its column's minimum, and a `min-width` on a
//!   cell floors its column's minimum (both in border-box terms: `min-width`
//!   constrains the content box under `content-box` sizing); a specified width on
//!   a `col` floors its column's *maximum* in the automatic layout (never the
//!   minimum: column.min = td.min per WPT `column-widths`) and
//!   pins the column in `table-layout: fixed`, while a `colgroup` width
//!   applies to each covered column without its own `col` width (at
//!   full value: a minimum in the automatic layout, a pin in
//!   `table-layout: fixed`).
//! - Stray content directly under the table (or row groups / rows) that is
//!   not a proper table child gets anonymous wrappers in place (CSS 2.1
//!   §17.2.1): anonymous table rows directly under the table (or row
//!   group), anonymous table cells directly under a row. Floated children of the table are laid out
//!   through the normal float machinery inside the table's block
//!   formatting context; out-of-flow children are recorded for the
//!   `layout_position` post-pass with a static position at the table's
//!   content origin.
//! - `caption-side: bottom` on the caption element is honored; anything
//!   else captions as top.
//! - Degenerate separate-borders input: when `border-spacing` alone meets
//!   or exceeds the table's content width, the table grows to fit the
//!   spacing plus the columns (never past its own `max-width`, in which
//!   case the columns overflow), so the exact-tiling invariant holds
//!   there too.

use super::BoxType;
use super::layout_position;
use super::{
    BlockInput, FloatCtx, FloatEntry, FontMetrics, PrefWidthCache, SpecLen, Task, WalkBufs,
    clamp_content_width_by_minmax, clearance_y, definite_specified_physical_width, establishes_bfc,
    finish_content_height, fresh_floats, generic_preferred_width, intrinsic_content_width,
    intrinsic_h_extras, is_float, min_content_width_pooled, place_float, preferred_width_mcw,
    record_shift, resolve_metrics, specified_height, table_inline_spec,
    vertical_caption_max_content, vertical_caption_min_content, vertical_cell_max_content,
    vertical_cell_min_content,
};
use crate::boxes::Forest;
use crate::boxes::SectionKind;
use crate::style::{
    BoxSizing, Clear, Display, Float, SpecMargin, VAlign, WhiteSpace, WritingMode, spec_len_of,
};
use crate::{
    BorderClaimant, CollapsedBorder, CollapsedBorderStyle, CollapsedBorders, LayoutBox, Position,
    position_of, resolve_font_size,
};
use bowser_css::{Color, CssValue, props};
use bowser_dom::QuirksMode;

/// One cell placed in the grid: its arena node plus grid coordinates.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GridCell {
    pub node: usize,
    pub row: usize,
    pub col: usize,
    pub colspan: usize,
    pub rowspan: usize,
    /// The cell's inline-axis padding/border for intrinsic column
    /// sizing, resolved once per layout in `collect_table_grid_into`
    /// (see `cell_intrinsic_pad`): the intrinsic passes used to
    /// recompute it per cell with identical arguments (4× total across
    /// `auto_column_maxima` / `auto_column_minima`).
    pub intrinsic_pad: f32,
}

impl GridCell {
    /// Whether the cell was authored as a single-column cell: the DOM
    /// `colspan` before track merging shrank it. A cell that spanned
    /// several tracks had its specified width applying to the span, so
    /// it must not constrain the single merged track it now occupies
    /// (WPT `column-track-merging`: the `colspan=10` cell's 50px does
    /// not pin its merged column, which absorbs the table's excess
    /// width instead). `colspan="0"` (span to end) never counts as
    /// single-column.
    fn was_single_col(&self, forest: &Forest) -> bool {
        forest.nodes[self.node].colspan == 1
    }
}

/// The table structure collected from the box tree: grid cells in DOM
/// order, caption blocks, and out-of-flow children.
///
/// `Default` is the empty grid: [`TableGridPool::take_grid`] hands a
/// checked-out buffer back as a fresh default when the pool is empty
/// (nested table while the outer table holds its grid).
#[derive(Default)]
pub(crate) struct TableGrid {
    pub cells: Vec<GridCell>,
    pub nrows: usize,
    pub ncols: usize,
    /// Whether the table uses `table-layout: fixed` (CSS 2.1 §17.5.2.1).
    /// Intrinsic passes consult this instead of re-reading the style:
    /// under fixed layout only first-row specified cell widths affect
    /// column widths, in both the used-width and the intrinsic paths.
    pub fixed: bool,
    /// `fixed` with a non-definite specified table width (`auto`,
    /// `min-content`, `max-content`, `fit-content` — including a
    /// `calc-size()` with one of those keyword bases, which
    /// `spec_len_of` already maps to the matching `SpecLen`). The
    /// table's used width is then the fixed-layout width — the pinned
    /// first-row column widths (CSS 2.1 §17.5.2.1) — so the intrinsic
    /// passes must not measure content (WPT
    /// `css/css-tables/fixed-layout-2.html`).
    pub fixed_indefinite: bool,
    /// Whether the table uses `border-collapse: collapse`. Under the
    /// collapsing border model the cell's own borders are drawn centered
    /// on the shared grid lines (see `resolve_collapsed_borders`), so
    /// they must not inflate the column's intrinsic widths the way
    /// separated-model borders do.
    pub collapse: bool,
    /// Whether the table is in a vertical writing mode. Column (inline)
    /// widths then consume the *vertical* cell padding/border, not the
    /// horizontal (see `cell_h_pad`).
    pub vertical: bool,
    /// CSS 2.1 §17.5 `visibility: collapse` per grid row, parallel to
    /// the grid rows (length `nrows`). A collapsed row takes no height
    /// and draws no spacing, unlike `visibility: hidden` which keeps its
    /// space. Set from the row box's computed style (inherited, so a row
    /// in a collapsed row group already carries the keyword); anonymous
    /// rows carry no style and read their row group instead.
    pub collapsed_rows: Vec<bool>,
    /// CSS 2.1 §17.5 `visibility: collapse` per logical column, parallel
    /// to the columns. A collapsed column takes no width and draws no
    /// spacing. Set from the covering `col` box's computed style, or the
    /// `colgroup`'s when the column has no `col` box.
    pub collapsed_cols: Vec<bool>,
    /// Row box per grid row, parallel to the collected rows (`None` for
    /// rows synthesized from bare cells, whose cell rect doubles as the
    /// row rect).
    pub row_nodes: Vec<Option<usize>>,
    /// Row-group box per grid row (`None` for bare rows/cells directly
    /// under the table), parallel to the collected rows.
    pub row_group: Vec<Option<usize>>,
    /// Row-group boxes in DOM order (for union rects).
    pub group_nodes: Vec<usize>,
    /// Section kind per row-group box, parallel to
    /// [`group_nodes`](Self::group_nodes): which table section
    /// (`thead` / `tbody` / `tfoot`) each group came from (see
    /// [`SectionKind`]). Consumers needing section-specific semantics —
    /// `visibility: collapse` reporting per CSS Tables 3 §5.4, repeated
    /// header/footer groups under fragmentation (CSS Tables 3 §6) —
    /// read the kind here instead of re-deriving the tag from the DOM.
    /// Staged plumbing: no in-tree consumer yet — the intended readers
    /// are `visibility: collapse` rendering per CSS Tables 3 §5.4 and
    /// repeated header/footer groups under fragmentation (CSS Tables 3
    /// §6). The parallel invariant is enforced by construction: every
    /// push goes through [`push_row_group`]; the `debug_assert_eq!` in
    /// [`collect_table_grid`] remains as a belt-and-suspenders check.
    // Staged plumbing: no non-test consumer reads this yet (the
    // intended readers are `visibility: collapse` rendering and
    // fragmented header/footer repetition); the allow goes away with
    // the first real reader.
    #[allow(dead_code)]
    group_kinds: Vec<SectionKind>,
    /// Caption boxes (`display: table-caption`), in DOM order; split
    /// top/bottom by `caption-side` when tasks are pushed.
    /// Stray inline runs are wrapped in anonymous blocks first, so they
    /// land here too instead of being dropped.
    pub captions: Vec<usize>,
    /// One entry per logical column, in column order: which `<col>` /
    /// `<colgroup>` boxes cover it. Always has `ncols` entries (one per
    /// logical column); `col_node` / `group_node` are `None` where no
    /// covering box declares that column.
    pub columns: Vec<ColumnSpec>,
    /// The `col` boxes in DOM order (deduplicated).
    pub col_nodes: Vec<usize>,
    /// The `colgroup` boxes in DOM order (deduplicated).
    pub col_group_nodes: Vec<usize>,
    /// Floated children of the table / row groups / rows (laid out
    /// through the float machinery in the table's formatting context).
    pub floats: Vec<usize>,
    pub abs_children: Vec<usize>,
}

/// Which column boxes cover one logical table column.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ColumnSpec {
    /// The `col` box covering this column, if any (a `col` width wins
    /// over its group's for the columns in its `span`).
    pub col_node: Option<usize>,
    /// The `colgroup` box covering this column, if any.
    pub group_node: Option<usize>,
}

/// Whitespace-only content: text nodes (or anonymous blocks wrapping them)
/// left between rows by pretty-printed HTML. Skipped by grid collection,
/// by the cell-baseline first-child skip, and by `layout`'s wrapper
/// scan (w76fix finding 3), which needs the same predicate for
/// whitespace-only children around a table.
/// Recursive through block/inline wrappers whose entire in-flow
/// subtree is whitespace (w76fix finding 6): a `<span> </span>` or
/// `<div> </div>` around nothing but whitespace must not count as a
/// real first child. Out-of-flow descendants never count. An empty
/// named wrapper is NOT whitespace-only (it is a real empty box);
/// only anonymous blocks keep the vacuous rule, since pass 4 leaves
/// empty ones behind exactly where whitespace was dropped.
/// Collapsed-whitespace gate (js-review M6b): only whitespace that
/// collapses counts. Under `pre` / `pre-wrap` / `break-spaces` the
/// whitespace is preserved and creates line boxes, so it is not
/// skippable; under `pre-line` spaces collapse but newlines are
/// preserved breaks, so a newline-bearing run is not skippable
/// either. The text node's own `white_space` (inherited at box
/// construction) is the authority — a wrapper's property reaches the
/// text through inheritance.
///
/// Iterative: the recursive version violated the crate's
/// never-recurse-into-the-box-tree invariant (deep anonymous wrapper
/// chains from hostile documents overflow the stack).
pub(crate) fn is_ws_only(forest: &Forest, idx: usize) -> bool {
    let mut stack = vec![idx];
    while let Some(i) = stack.pop() {
        let n = &forest.nodes[i];
        match n.box_type {
            BoxType::Text => {
                if n.is_br {
                    return false;
                }
                let t = n.text.as_deref().unwrap_or("");
                if !t.chars().all(char::is_whitespace) {
                    return false;
                }
                if !t.is_empty() {
                    match n.data.white_space {
                        WhiteSpace::Pre | WhiteSpace::PreWrap | WhiteSpace::BreakSpaces => {
                            return false;
                        }
                        WhiteSpace::PreLine if t.contains('\n') => return false,
                        _ => {}
                    }
                }
            }
            BoxType::AnonymousBlock => {
                for &c in &n.children {
                    if !layout_position::is_out_of_flow(forest, c) {
                        stack.push(c);
                    }
                }
            }
            // Named wrappers recurse only when they actually wrap
            // something in-flow: an *empty* block/inline box is a real
            // box, not whitespace (an empty `%` div is a cell's first
            // block child for baselines; a foster-parented-empty
            // `tbody` restyled to block is a stray run, not a dropped
            // one). Anonymous blocks keep the vacuous-true rule: pass
            // 4 leaves genuinely empty ones behind exactly where
            // whitespace was dropped, so they *are* the whitespace
            // residue for grid collection and the wrapper scan.
            BoxType::Block | BoxType::Inline => {
                let mut any = false;
                for &c in &n.children {
                    if !layout_position::is_out_of_flow(forest, c) {
                        stack.push(c);
                        any = true;
                    }
                }
                if !any {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

/// Whether the table uses `table-layout: fixed` (CSS 2.1 §17.5.2.1),
/// gated by the CSS Tables 3 fixed-mode trigger: fixed layout applies
/// unless the table's *specified* inline size is the bare `auto` or
/// `max-content` keyword (csswg #10937; Gecko/Blink agree). The rule is
/// syntactic: a `calc-size()` function is a specified function value, not
/// a keyword, so it triggers fixed mode with *any* basis, including
/// `auto` and `max-content` (fixed-layout-2.html asserts the trigger
/// fires "except when inline-size is auto or max-content", and hardcodes
/// fixed mode on for every `calc-size()` variant). Only the bare
/// `auto`/`max-content` keywords suppress fixed mode. Genuinely
/// unparseable width functions are dropped by the cascade, leaving the
/// width at `auto`, which correctly disables fixed mode. Shared by the
/// used-width path and the intrinsic passes so both agree on which rows'
/// specified cell widths count.
fn table_is_fixed(forest: &Forest, idx: usize) -> bool {
    let Some(style) = forest.nodes[idx].style.as_ref() else {
        return false;
    };
    if !style
        .get_str_by_index(props::TABLE_LAYOUT)
        .is_some_and(|v| v.eq_ignore_ascii_case("fixed"))
    {
        return false;
    }
    let width = style.get_by_index(if super::is_vertical_table(forest, idx) {
        // The trigger reads the *specified* inline size: the physical
        // `height` for vertical tables.
        props::HEIGHT
    } else {
        props::WIDTH
    });
    // A `calc-size()` of any basis is a specified function value, not the
    // bare keyword, so it keeps fixed mode (WPT fixed-layout-2.html:
    // `calc-size(auto, size)` and `calc-size(max-content, size)` trigger
    // fixed mode even though their bases, as bare keywords, would
    // suppress it). Only the bare keywords below suppress.
    if width.and_then(CssValue::calc_size_basis).is_some() {
        return true;
    }
    // The trigger reads the *specified* width, before any intrinsic
    // resolution: only the bare `auto` and `max-content` keywords (and
    // anything the cascade could not parse) suppress fixed mode;
    // `<length>`, `<percentage>`, `min-content` and `fit-content` keep it.
    !matches!(spec_len_of(width), SpecLen::Auto | SpecLen::MaxContent)
}

/// Whether the table's specified width is non-definite: `auto` or an
/// intrinsic keyword (`min-content`, `max-content`, `fit-content`). A
/// `calc-size()` with a keyword basis maps to the matching `SpecLen`
/// in [`spec_len_of`], so those are covered too. Under
/// `table-layout: fixed` such a table's used width is the fixed-layout
/// width — the pinned first-row column widths (CSS 2.1 §17.5.2.1) —
/// never content measurement (WPT
/// `css/css-tables/fixed-layout-2.html`).
fn table_width_indefinite(forest: &Forest, idx: usize) -> bool {
    // The inline-axis spec: the physical `height` for a vertical table.
    matches!(
        table_inline_spec(forest, idx),
        SpecLen::Auto | SpecLen::MaxContent | SpecLen::MinContent | SpecLen::FitContent
    )
}

/// Collect the table structure: rows from row groups (or bare rows, or
/// consecutive bare cells sharing one anonymous row — CSS would
/// synthesize the missing wrappers), captions, and out-of-flow children.
/// Iterative; never recurses into cell contents.
/// Hard cap on the table grid's total cells (rows × columns). Each of
/// `colspan` / `rowspan` is clamped to 1000 per the HTML parsing rules,
/// but that bounds neither axis' product: 1000 rows of
/// `<td rowspan="1000" colspan="1000">` would otherwise force a ~1e9-cell
/// occupancy grid (~1GB) from a ~55KB document. Spans are shrunk to fit
/// the cap; the occupancy grid then holds at most this many bools.
const MAX_TABLE_GRID_CELLS: usize = 1_000_000;

/// Flush accumulated consecutive bare cells as a single anonymous row
/// (CSS 2.1 §17.2.1). `group` is the row group the cells came from, if
/// any.
fn flush_bare_cells(
    rows: &mut Vec<Vec<usize>>,
    nrow: &mut usize,
    row_nodes: &mut Vec<Option<usize>>,
    row_group: &mut Vec<Option<usize>>,
    pending: &mut Vec<usize>,
    group: Option<usize>,
) {
    if pending.is_empty() {
        return;
    }
    if *nrow == rows.len() {
        rows.push(Vec::new());
    }
    // Move the run into its row slot, keeping the slot's (empty) buffer
    // in `pending` for the next run: slots at `*nrow..` are always empty
    // (cleared on pool checkout), so no cell is ever lost or duplicated.
    std::mem::swap(&mut rows[*nrow], pending);
    row_nodes.push(None);
    row_group.push(group);
    *nrow += 1;
}

/// Whether every column a cell spans is collapsed (CSS 2.1 §17.5): the
/// cell takes no space and never paints, so [`layout_table`] skips its
/// layout task outright — its rect stays zero and its text never
/// shapes, so paint emits nothing for it.
fn cell_cols_collapsed(collapsed_cols: &[bool], cell: &GridCell) -> bool {
    let end = (cell.col + cell.colspan).min(collapsed_cols.len());
    (cell.col..end).all(|cc| collapsed_cols[cc])
}

/// Whether a cell intersects any collapsed column (CSS 2.1 §17.5).
fn cell_spans_collapsed_col(collapsed_cols: &[bool], cell: &GridCell) -> bool {
    let end = (cell.col + cell.colspan).min(collapsed_cols.len());
    (cell.col..end).any(|cc| collapsed_cols[cc])
}

/// Whether a cell intersects any collapsed row (CSS 2.1 §17.5).
fn cell_spans_collapsed_row(collapsed_rows: &[bool], cell: &GridCell) -> bool {
    let end = (cell.row + cell.rowspan).min(collapsed_rows.len());
    (cell.row..end).any(|r| collapsed_rows[r])
}

/// Whether a cell was laid out at all: cells in a collapsed row, or
/// spanning only collapsed columns, never get a layout task (see
/// [`cell_cols_collapsed`]).
fn cell_laid_out(collapsed_rows: &[bool], collapsed_cols: &[bool], cell: &GridCell) -> bool {
    !collapsed_rows.get(cell.row).copied().unwrap_or(false)
        && !cell_cols_collapsed(collapsed_cols, cell)
}

/// Whether a cell is laid out purely for height measurement although it
/// never paints or places: a non-spanning cell in a collapsed row (its
/// content height feeds the specified-table-height distribution —
/// css-tables-3 §3.10.5 counts collapsed rows as auto-height rows; WPT
/// `visibility-collapse-row-004`), or a non-spanning cell in a visible
/// row whose columns are all collapsed (its height still sizes the row;
/// WPT `visibility-collapse-col-001`). After measurement the cell is
/// detached (zero rect, children dropped) so paint never sees it.
fn cell_measured_for_height(
    collapsed_rows: &[bool],
    collapsed_cols: &[bool],
    cell: &GridCell,
) -> bool {
    cell.rowspan == 1 && !cell_laid_out(collapsed_rows, collapsed_cols, cell)
}

/// Whether a cell feeds column-width computation (CSS 2.1 §17.5.5): the
/// suppression of a collapsed row "does not otherwise affect the layout
/// of the table", so a collapsed row's cells still size their columns
/// (WPT `visibility-collapse-row-001`); but a cell intersecting a
/// collapsed column contributes nothing — not even to its visible
/// spanned columns (WPT `visibility-collapse-colspan-001`: the 30px
/// `colspan=3` cell over a collapsed middle column does not grow the
/// outer columns).
fn cell_width_contributes(collapsed_cols: &[bool], cell: &GridCell) -> bool {
    !cell_spans_collapsed_col(collapsed_cols, cell)
}

/// Whether a cell's height sizes its rows (CSS 2.1 §17.5): a cell
/// intersecting a collapsed row *and* a collapsed column never grows
/// the visible rows it spans (WPT `visibility-collapse-rowcol-002`:
/// the 25px `rowspan=2` cell over a collapsed row *and* column leaves
/// the visible row at the 10px cell's height). A cell intersecting only
/// a collapsed row is still absorbed by its visible rows (unit test
/// `visibility_collapse_rowspan_cell_grows_visible_rows_only`), and
/// cells in collapsed columns still size their (visible) rows (WPT
/// `visibility-collapse-col-001`).
fn cell_height_contributes(
    collapsed_rows: &[bool],
    collapsed_cols: &[bool],
    cell: &GridCell,
) -> bool {
    !(cell_spans_collapsed_row(collapsed_rows, cell)
        && cell_spans_collapsed_col(collapsed_cols, cell))
}

/// Whether the box's computed `visibility` is `collapse` (CSS 2.1 §17.5).
/// `visibility` is inherited in the cascade, so a row inside a collapsed
/// row group (or a `col` inside a collapsed `colgroup`) usually already
/// carries the keyword on its own computed style — unless it explicitly
/// overrides back to `visible`, in which case the caller still collapses
/// it via the row group's flag (a collapsed row group removes the whole
/// group as a unit); anonymous rows/cells carry no style and are checked
/// against their row group by the caller.
fn visibility_collapsed(forest: &Forest, idx: usize) -> bool {
    forest.nodes[idx]
        .style
        .as_ref()
        .and_then(|s| s.get_str_by_index(props::VISIBILITY))
        .is_some_and(|v| v.eq_ignore_ascii_case("collapse"))
}

/// Section kind of a row-group box node: `Some(SectionKind::Header)`
/// for `<thead>`, `Some(SectionKind::Footer)` for `<tfoot>`,
/// `Some(SectionKind::Body)` for `<tbody>` and for row groups
/// generated from non-section elements. Returns `None` for
/// non-row-group nodes: only row-group boxes are stamped at
/// construction (see [`SectionKind`]).
pub(crate) fn row_group_section_kind(forest: &Forest, group: usize) -> Option<SectionKind> {
    if forest.nodes[group].box_type == BoxType::TableRowGroup {
        forest.nodes[group].section_kind
    } else {
        None
    }
}

/// Push a row-group box and its section kind onto the parallel
/// `group_nodes` / `group_kinds` vectors, keeping the parallel
/// invariant by construction (the `debug_assert_eq!` in
/// [`collect_table_grid`] double-checks it).
fn push_row_group(
    group_nodes: &mut Vec<usize>,
    group_kinds: &mut Vec<SectionKind>,
    node: usize,
    kind: SectionKind,
) {
    group_nodes.push(node);
    group_kinds.push(kind);
}

/// Whether the box is a real table caption: `display: table-caption` in
/// the cascade (the box itself is a plain block). Distinguishes actual
/// `<caption>` elements from stray block-level content directly under
/// the table, which per CSS 2.1 §17.2.1 gets anonymous table wrappers
/// instead of caption treatment.
fn is_table_caption(forest: &Forest, c: usize) -> bool {
    matches!(forest.nodes[c].box_type, BoxType::Block | BoxType::Grid)
        && forest.nodes[c]
            .style
            .as_ref()
            .and_then(|s| s.get_str_by_index(props::DISPLAY))
            .is_some_and(|v| v.eq_ignore_ascii_case("table-caption"))
}

/// Wrap runs of stray content directly under the table in anonymous
/// Wrap a stray-content run in an anonymous box, unless the run is
/// whitespace-only (CSS Tables 3 §2.2.1 stage 1 rules 3-4: whitespace
/// between table-internal boxes is dropped, not wrapped). Returns the
/// anonymous box, or `None` for a dropped run.
fn flush_stray_run(
    forest: &mut Forest,
    parent: usize,
    box_type: BoxType,
    run: &mut Vec<usize>,
) -> Option<usize> {
    if run.is_empty() {
        return None;
    }
    if run.iter().all(|&c| is_ws_only(forest, c)) {
        run.clear();
        return None;
    }
    let data = forest.nodes[parent].data.clone();
    let anon = forest.make_anonymous_box(box_type, &data, std::mem::take(run), parent);
    Some(anon)
}

/// table rows (CSS 2.1 §17.2.1), so grid collection lays them out in
/// place instead of dropping them. Anything that is not a proper table
/// child (row group, row, column group, column, caption), out-of-flow,
/// or floated is stray — including bare cells (CSS Tables 3 §2.2.1
/// rule 2.1 wraps every sequence of non-proper children, cells
/// included) and whitespace (which never breaks a run: "consecutive"
/// skips whitespace-only anonymous inlines, §2.1.1); each run becomes
/// one anonymous row, cell-wrapped by [`wrap_row_strays`], while
/// whitespace-only runs are dropped (§2.2.1 stage 1 rules 3-4).
/// Pass 3 of box construction only wraps when the table already has a
/// block-level child, so a table with nothing but stray inline content
/// still carries raw inline children here. Row groups and rows get the
/// same fix-up recursively (anonymous rows for stray group content,
/// anonymous cells for stray row content). Called by [`layout_table`]
/// and [`table_pref_width`] before [`collect_table_grid`]; the
/// min-content width path does not wrap (stray content is not part of
/// that lower bound).
pub(crate) fn wrap_table_stray_content(forest: &mut Forest, idx: usize) {
    let kids = std::mem::take(&mut forest.nodes[idx].children);
    let mut wrapped: Vec<usize> = Vec::with_capacity(kids.len());
    let mut run: Vec<usize> = Vec::new();
    for c in kids {
        let bt = forest.nodes[c].box_type;
        let oof = layout_position::is_out_of_flow(forest, c);
        // Proper table children (CSS Tables 3 §2.1.1: track groups,
        // tracks, captions) break runs and are kept as-is. Bare cells
        // are *not* proper children: rule 2.1 wraps them into the
        // anonymous row with the surrounding strays
        // (`table-model-fixup` 2.3).
        let proper = !oof
            && (matches!(
                bt,
                BoxType::TableRowGroup
                    | BoxType::TableRow
                    | BoxType::TableColumn
                    | BoxType::TableColumnGroup
            ) || is_table_caption(forest, c));
        let stray = !oof && !is_float(forest, c) && !proper;
        if stray {
            run.push(c);
        } else {
            if let Some(anon) = flush_stray_run(forest, idx, BoxType::TableRow, &mut run) {
                wrap_row_strays(forest, anon);
                wrapped.push(anon);
            }
            wrapped.push(c);
        }
    }
    if let Some(anon) = flush_stray_run(forest, idx, BoxType::TableRow, &mut run) {
        wrap_row_strays(forest, anon);
        wrapped.push(anon);
    }
    forest.nodes[idx].children = wrapped;
    // Recurse into row groups and rows for their own stray content (this
    // also reaches the anonymous rows created above: re-wrapping one is
    // a no-op since its children are already cells).
    let kids = forest.nodes[idx].children.clone();
    for c in kids {
        match forest.nodes[c].box_type {
            BoxType::TableRowGroup => wrap_row_group_strays(forest, c),
            BoxType::TableRow => wrap_row_strays(forest, c),
            _ => {}
        }
    }
}

/// Wrap stray content directly under a row group in anonymous rows
/// (CSS Tables 3 §2.2.1 rule 2.2: every sequence of children which are
/// not table-row boxes — cells included — is wrapped). Whitespace never
/// breaks a run (§2.1.1 "consecutive"); whitespace-only runs are
/// dropped (§2.2.1 stage 1 rules 3-4). Each row's children are then
/// cell-wrapped by [`wrap_row_strays`].
fn wrap_row_group_strays(forest: &mut Forest, group: usize) {
    let kids = std::mem::take(&mut forest.nodes[group].children);
    let mut wrapped: Vec<usize> = Vec::with_capacity(kids.len());
    let mut run: Vec<usize> = Vec::new();
    for c in kids {
        let bt = forest.nodes[c].box_type;
        let is_row = bt == BoxType::TableRow && !layout_position::is_out_of_flow(forest, c);
        // CSS Tables 3 §2.2.1 rule 2.2 wraps every sequence of non-row
        // children — including table-column boxes (they are not rows).
        // Misparented columns must not break the run
        // (`table-model-fixup` 2.2); they are dropped at row level by
        // `wrap_row_strays` since they cannot define outer columns.
        let stray = !layout_position::is_out_of_flow(forest, c) && !is_float(forest, c) && !is_row;
        if stray {
            run.push(c);
        } else {
            if let Some(anon) = flush_stray_run(forest, group, BoxType::TableRow, &mut run) {
                wrap_row_strays(forest, anon);
                wrapped.push(anon);
            }
            wrapped.push(c);
            if is_row {
                wrap_row_strays(forest, c);
            }
        }
    }
    if let Some(anon) = flush_stray_run(forest, group, BoxType::TableRow, &mut run) {
        wrap_row_strays(forest, anon);
        wrapped.push(anon);
    }
    forest.nodes[group].children = wrapped;
}

/// Wrap orphan table-internal boxes in their missing ancestor table
/// boxes (CSS 2.1 §17.2.1): a `<tbody>`, `<tr>`, `<td>`, … with no table
/// ancestor — e.g. after script moves it out from under its `<table>` —
/// gets anonymous wrappers generated around runs of consecutive
/// siblings, innermost level first (cell → row → row-group → table;
/// column → column-group → table; row-group / column-group / caption →
/// table). Only boxes built from actual table elements are candidates
/// (plus the anonymous wrappers created here, which join the chain via
/// `Node::from_table_element`); `display: table-*` on other elements
/// keeps its existing behavior.
///
/// Runs at the top of every `layout_forest_inner` pass, before task
/// creation, so table layout always sees a well-parented grid (WPT
/// `html-to-css-mapping-1`: the orphan `tbody` measures 10px wide
/// inside its anonymous table). The pass is idempotent — wrapped
/// orphans end up under proper table parents — and terminates: each
/// pass wraps every orphan one level up a bounded chain (`Table` is
/// never an orphan).
pub(crate) fn wrap_orphan_table_boxes(forest: &mut Forest) {
    // Safety cap: the longest chain is cell → row → row-group → table
    // (three wrapping passes plus a final clean pass).
    for _ in 0..8 {
        let mut wrapped_any = false;
        // Snapshot the node count: wrappers are appended past it and
        // join the fixup on the next pass.
        let count = forest.nodes.len();
        for i in 0..count {
            wrapped_any |= wrap_orphan_children(forest, i);
        }
        if !wrapped_any {
            break;
        }
    }
}

/// The anonymous wrapper an orphan table box needs (CSS 2.1 §17.2.1),
/// or `None` when the box is not an orphan table box: not built from a
/// table element, not table-internal, or already under a proper table
/// container (a `Table` box, or an `InlineBlock` with `display:
/// inline-table`).
fn is_orphan_table_box(forest: &Forest, idx: usize) -> Option<BoxType> {
    let n = &forest.nodes[idx];
    if !n.from_table_element {
        return None;
    }
    let wrapper = match n.box_type {
        BoxType::TableCell => BoxType::TableRow,
        BoxType::TableRow => BoxType::TableRowGroup,
        BoxType::TableColumn => BoxType::TableColumnGroup,
        BoxType::TableRowGroup | BoxType::TableColumnGroup => BoxType::Table,
        _ if is_table_caption(forest, idx) => BoxType::Table,
        _ => return None,
    };
    let parent = n.parent?;
    let p = &forest.nodes[parent];
    let p_is_table = p.box_type == BoxType::Table
        || (p.box_type == BoxType::InlineBlock && p.data.display == Display::InlineTable);
    let proper = match n.box_type {
        BoxType::TableCell => p.box_type == BoxType::TableRow,
        BoxType::TableRow => p.box_type == BoxType::TableRowGroup || p_is_table,
        BoxType::TableColumn => p.box_type == BoxType::TableColumnGroup || p_is_table,
        _ => p_is_table,
    };
    if proper { None } else { Some(wrapper) }
}

/// Wrap one run of consecutive orphan siblings in an anonymous
/// `wrapper` box (a child of `parent`). Returns the wrapper, or `None`
/// for an empty run. The wrapper is stamped `from_table_element` so the
/// fixup chain keeps going: it is itself an orphan until wrapped in
/// turn (e.g. the anonymous row-group around an orphan row still needs
/// its anonymous table).
fn wrap_orphan_run(
    forest: &mut Forest,
    parent: usize,
    wrapper: BoxType,
    run: &mut Vec<usize>,
) -> Option<usize> {
    if run.is_empty() {
        return None;
    }
    let data = forest.nodes[parent].data.clone();
    let anon = forest.make_anonymous_box(wrapper, &data, std::mem::take(run), parent);
    forest.nodes[anon].from_table_element = true;
    // `make_anonymous_box` leaves `display` at the inherited block
    // value, but layout dispatches to the table engine off
    // `data.display`, so stamp the wrapper's own display.
    forest.nodes[anon].data.display = match wrapper {
        BoxType::Table => Display::Table,
        BoxType::TableRow => Display::TableRow,
        BoxType::TableRowGroup => Display::TableRowGroup,
        BoxType::TableColumn => Display::TableColumn,
        BoxType::TableColumnGroup => Display::TableColumnGroup,
        _ => Display::Block,
    };
    Some(anon)
}

/// Flush the open orphan run (if any) into an anonymous wrapper child
/// of `parent`. Returns whether a wrapper was created.
fn flush_orphan_run(
    forest: &mut Forest,
    parent: usize,
    new_children: &mut Vec<usize>,
    run: &mut Vec<usize>,
    run_wrapper: &mut Option<BoxType>,
) -> bool {
    let Some(wrapper) = run_wrapper.take() else {
        return false;
    };
    if let Some(anon) = wrap_orphan_run(forest, parent, wrapper, run) {
        new_children.push(anon);
        true
    } else {
        false
    }
}

/// Wrap each maximal run of consecutive orphan table-box children of
/// `parent` in the anonymous wrapper its level needs (see
/// [`wrap_orphan_table_boxes`]). Out-of-flow and floated children break
/// runs and are kept as-is; whitespace adjacent to a run is dropped
/// (CSS Tables 3 §2.2.1 stage 1 rules 3-4). Returns whether anything was
/// wrapped.
fn wrap_orphan_children(forest: &mut Forest, parent: usize) -> bool {
    // Cheap scan first: rebuilding every child list on every pass would
    // churn the allocator for the common no-orphan case.
    let mut has_orphan = false;
    for &c in &forest.nodes[parent].children {
        if !layout_position::is_out_of_flow(forest, c)
            && !is_float(forest, c)
            && is_orphan_table_box(forest, c).is_some()
        {
            has_orphan = true;
            break;
        }
    }
    if !has_orphan {
        return false;
    }
    let kids = std::mem::take(&mut forest.nodes[parent].children);
    let mut changed = false;
    let mut new_children: Vec<usize> = Vec::with_capacity(kids.len());
    let mut run: Vec<usize> = Vec::new();
    let mut run_wrapper: Option<BoxType> = None;
    // Whitespace seen since the last non-whitespace child: dropped when
    // it leads or trails an orphan run (adjacent to table-internal
    // boxes), kept as normal content otherwise.
    let mut pending_ws: Vec<usize> = Vec::new();
    for c in kids {
        if is_ws_only(forest, c) {
            pending_ws.push(c);
            continue;
        }
        let wrapper = if layout_position::is_out_of_flow(forest, c) || is_float(forest, c) {
            None
        } else {
            is_orphan_table_box(forest, c)
        };
        match wrapper {
            Some(w) => {
                // Leading whitespace is dropped with the run.
                pending_ws.clear();
                if run_wrapper != Some(w) {
                    changed |= flush_orphan_run(
                        forest,
                        parent,
                        &mut new_children,
                        &mut run,
                        &mut run_wrapper,
                    );
                    run_wrapper = Some(w);
                }
                run.push(c);
            }
            None => {
                let flushed = flush_orphan_run(
                    forest,
                    parent,
                    &mut new_children,
                    &mut run,
                    &mut run_wrapper,
                );
                changed |= flushed;
                if flushed {
                    // Trailing whitespace adjacent to the run is dropped.
                    pending_ws.clear();
                } else {
                    new_children.append(&mut pending_ws);
                }
                new_children.push(c);
            }
        }
    }
    if flush_orphan_run(
        forest,
        parent,
        &mut new_children,
        &mut run,
        &mut run_wrapper,
    ) {
        changed = true;
        pending_ws.clear();
    }
    new_children.extend(pending_ws);
    forest.nodes[parent].children = new_children;
    changed
}

/// Wrap stray content directly under a row in anonymous cells
/// (CSS Tables 3 §2.2.1 rule 2.3). Each *sequence* of consecutive
/// non-cell, non-whitespace children shares one anonymous cell:
/// a stray joins the open cell, which flushes only at a real cell,
/// an out-of-flow box, a float, or a misparented column (dropped).
/// Whitespace-only children join the adjacent cell — leading
/// whitespace joins the following cell, trailing whitespace the
/// preceding one — so `" <span>x</span> "` still collapses into a
/// single anonymous cell (WPT `table-model-fixup` 1.4). Whitespace
/// with no adjacent non-whitespace content is dropped (§2.2.1
/// stage 1 rules 3-4).
fn wrap_row_strays(forest: &mut Forest, row: usize) {
    let kids = std::mem::take(&mut forest.nodes[row].children);
    let mut wrapped: Vec<usize> = Vec::with_capacity(kids.len());
    // Whitespace seen since the last non-whitespace child (or the
    // start): joins the next anonymous cell, or is dropped when no
    // non-whitespace child follows.
    let mut pending_ws: Vec<usize> = Vec::new();
    // The anonymous cell under construction; always holds at least one
    // non-whitespace child once non-empty.
    let mut open: Vec<usize> = Vec::new();
    for c in kids {
        let bt = forest.nodes[c].box_type;
        let is_cell = bt == BoxType::TableCell && !layout_position::is_out_of_flow(forest, c);
        // Misparented table-column boxes (e.g. a `col` inside a row
        // group) cannot define outer columns; drop them instead of
        // wrapping them in anonymous cells (`table-model-fixup` 2.2).
        let is_misparented_col = matches!(bt, BoxType::TableColumn | BoxType::TableColumnGroup);
        if is_misparented_col {
            if let Some(anon) = flush_stray_run(forest, row, BoxType::TableCell, &mut open) {
                wrapped.push(anon);
            }
            pending_ws.clear();
            continue;
        }
        if is_cell || layout_position::is_out_of_flow(forest, c) || is_float(forest, c) {
            // Real cells, out-of-flow boxes and floats are kept as-is:
            // close the open cell, and drop whitespace before a real
            // cell (stage 1: whitespace adjacent to table-internal
            // boxes is dropped).
            if let Some(anon) = flush_stray_run(forest, row, BoxType::TableCell, &mut open) {
                wrapped.push(anon);
            }
            pending_ws.clear();
            wrapped.push(c);
            continue;
        }
        if is_ws_only(forest, c) {
            // Whitespace joins the open cell (trailing) or waits for
            // the next one (leading).
            if open.is_empty() {
                pending_ws.push(c);
            } else {
                open.push(c);
            }
            continue;
        }
        // A non-whitespace stray child joins the open anonymous
        // cell (CSS Tables 3 §2.2.1 rule 2.3: one cell per sequence
        // of consecutive non-cell children), carrying any leading
        // whitespace with it when it opens the cell.
        if open.is_empty() {
            open = std::mem::take(&mut pending_ws);
        }
        open.push(c);
    }
    if let Some(anon) = flush_stray_run(forest, row, BoxType::TableCell, &mut open) {
        wrapped.push(anon);
    }
    // Leftover leading whitespace with no following non-whitespace
    // child is dropped (stage 1 rules 3-4).
    forest.nodes[row].children = wrapped;
}

/// CSS Tables 3 §3.3.2 track merging for fixed-mode tables (only rows
/// are eligible; every column is preserved). Iteratively merges a row
/// track that has no table-row box defining it explicitly and is
/// spanned by exactly the same set of cells as the previous track: the
/// spanning cells' `rowspan` shrinks by one to compensate and cells
/// originating past the deleted track shift up. Returns the new row
/// count.
///
/// `placed` holds `(node, row, col, colspan, rowspan, _)` tuples in row
/// order; `row_nodes` is parallel to the collected rows (`None` for
/// anonymous rows and for rowspan-extension tracks past its length).
/// Merges can only fire at tracks with no originating cells — an
/// anonymous collected row always has at least one originating cell,
/// which can never be in the previous track's spanning set — so the
/// scan starts past the collected rows.
///
/// The mergeable tracks are found with a single incremental sweep
/// instead of re-scanning all placed cells per track: a track `t` has
/// the same spanning set as `t - 1` exactly when no cell starts or ends
/// at `t` (every placed cell spans at least one track and each cell
/// node is placed once, so the symmetric difference of consecutive
/// spanning sets is precisely the cells starting or ending at `t`).
/// Starts are walked in row order straight from `placed`; ends are
/// pre-sorted once. The sweep is O(P log P + R) with a constant number
/// of allocations, versus O(R * P log P) with two allocations per
/// track for the naive rescan.
fn merge_fixed_redundant_rows(
    placed: &mut [(usize, usize, usize, usize, usize, usize)],
    row_nodes: &[Option<usize>],
    nrows: usize,
    ends: &mut Vec<usize>,
    deleted: &mut Vec<usize>,
) -> usize {
    let start = row_nodes.len().max(1);
    // Sorted end tracks (`row + rowspan`) of all cells.
    ends.clear();
    ends.extend(
        placed
            .iter()
            .map(|&(_, row, _, _, rowspan, _)| row + rowspan),
    );
    ends.sort_unstable();
    let mut si = placed.partition_point(|&(_, row, _, _, _, _)| row < start);
    let mut ei = ends.partition_point(|&end| end < start);
    // Mergeable tracks in original track coordinates, ascending.
    deleted.clear();
    for t in start..nrows {
        let mut changed = false;
        while si < placed.len() && placed[si].1 == t {
            changed = true;
            si += 1;
        }
        while ei < ends.len() && ends[ei] == t {
            changed = true;
            ei += 1;
        }
        if !changed {
            deleted.push(t);
        }
    }
    // Apply the deletions in ascending original-track order. A cell
    // loses one rowspan per deleted track it spans and shifts up by one
    // per deleted track below its origin — the same per-cell update the
    // old iterative loop applied at each merge, batched into one pass.
    // (A deleted track never holds an originating cell: one would sit in
    // its spanning set but not the previous track's, contradicting the
    // equality that made the track mergeable.)
    for cell in placed.iter_mut() {
        let (_, row, _, _, rowspan, _) = cell;
        let below = deleted.partition_point(|&d| d < *row);
        let spanned = deleted.partition_point(|&d| d < *row + *rowspan) - below;
        *row -= below;
        *rowspan -= spanned;
    }
    nrows - deleted.len()
}

/// Core column-track merge sweep over a caller-supplied `defined`
/// bitset: a track `t >= 1` merges into `t - 1` when `defined[t]` is
/// false and no cell originates at `t`. When `check_ends` is set, a
/// non-empty track additionally requires the exact same spanning set
/// as the previous track (CSS Tables 3 §3.3.2) — i.e. no cell ends at
/// `t` either; an *empty* track (spanned by no cell at all) still
/// merges when a cell ends at it, because there is nothing in the
/// track to preserve (WPT `column-track-merging` col_auto: the empty
/// track 1 merges despite the first cell ending at it, while the
/// colspan=2/colspan=3 regression keeps its spanned track 2). When
/// clear, only originating cells block the merge. Deleted tracks
/// shrink the spanning cells' `colspan` by one per track and shift
/// cells originating past them left; the declared `col`/`colgroup`
/// ranges are remapped into the surviving track coordinates. Returns
/// the new column count.
///
/// (`placed` is in row order, so the start track list is pre-sorted
/// here; the sweep stays O(P log P + C) with a constant number of
/// allocations.)
fn merge_redundant_columns_inner(
    placed: &mut [(usize, usize, usize, usize, usize, usize)],
    col_ranges: &mut [(usize, usize, usize)],
    group_ranges: &mut [(usize, usize, usize)],
    defined: &[bool],
    ncols: usize,
    check_ends: bool,
) -> usize {
    // Sorted start (`col`) tracks of all cells.
    let mut starts: Vec<usize> = placed.iter().map(|&(_, _, col, _, _, _)| col).collect();
    starts.sort_unstable();
    let ends: Vec<usize> = if check_ends {
        let mut e: Vec<usize> = placed
            .iter()
            .map(|&(_, _, col, colspan, _, _)| col + colspan)
            .collect();
        e.sort_unstable();
        e
    } else {
        Vec::new()
    };
    // Track 0 has no previous track, so the sweep starts at 1; skip any
    // cell starts at track 0 up front to keep the `starts[si] >= t`
    // invariant (end tracks are always >= 1 since every span is >= 1).
    // `si`/`ei` then count starts/ends at or below the current track,
    // so `si - ei` is the number of cells spanning track `t` (each cell
    // contributes exactly one start and one end).
    let mut si = starts.partition_point(|&s| s < 1);
    let mut ei = 0;
    // Mergeable tracks in original track coordinates, ascending.
    let mut deleted: Vec<usize> = Vec::new();
    for (t, &is_defined) in defined.iter().enumerate().skip(1) {
        let mut start_at_t = false;
        while si < starts.len() && starts[si] == t {
            start_at_t = true;
            si += 1;
        }
        let mut end_at_t = false;
        if check_ends {
            while ei < ends.len() && ends[ei] <= t {
                if ends[ei] == t {
                    end_at_t = true;
                }
                ei += 1;
            }
        }
        // An originating cell always blocks the merge. A cell ending at
        // `t` blocks it only when the track is non-empty: an empty
        // track has no spanning set to preserve, so it merges away
        // (its spacing goes with it) rather than surviving as a
        // zero-width track that still draws border-spacing.
        let empty = check_ends && si == ei;
        let blocked = start_at_t || (end_at_t && !empty);
        if !blocked && !is_defined {
            deleted.push(t);
        }
    }
    // Apply the deletions in ascending original-track order: a cell
    // loses one colspan per deleted track it spans and shifts left by
    // one per deleted track below its origin. (A deleted track never
    // holds an originating cell: one would sit in its spanning set but
    // not the previous track's, contradicting the equality that made
    // the track mergeable.)
    for cell in placed.iter_mut() {
        let (_, _, col, colspan, _, _) = cell;
        let left = deleted.partition_point(|&d| d < *col);
        let spanned = deleted.partition_point(|&d| d < *col + *colspan) - left;
        *col -= left;
        *colspan -= spanned;
    }
    // Remap the declared ranges into the surviving coordinates: ranges
    // shift left over deleted tracks, and `colgroup` ranges (but never
    // `col` ranges, whose tracks are never deleted) can shrink.
    let below = |t: usize| deleted.partition_point(|&d| d < t);
    for r in col_ranges.iter_mut() {
        let new_start = r.1 - below(r.1);
        r.2 = r.1 + r.2 - below(r.1 + r.2) - new_start;
        r.1 = new_start;
    }
    for r in group_ranges.iter_mut() {
        // (node, start, end).
        let new_start = r.1 - below(r.1);
        r.2 -= below(r.2);
        r.1 = new_start;
    }
    ncols - deleted.len()
}

#[allow(dead_code)] // Unwired pending the auto-mode track-merge work; covered by unit tests.
/// Column-track analog of [`merge_fixed_redundant_rows`]: the same
/// incremental sweep over `col`/`colspan` start/end tracks, with every
/// `col`-covered track defined explicitly (fixed-mode rule: no column
/// merging). See [`merge_redundant_columns_inner`] for the sweep.
fn merge_fixed_redundant_columns(
    placed: &mut [(usize, usize, usize, usize, usize, usize)],
    col_ranges: &mut [(usize, usize, usize)],
    group_ranges: &mut [(usize, usize, usize)],
    ncols: usize,
) -> usize {
    // Tracks covered by a `col` box are defined explicitly and can never
    // merge. A `colgroup` box alone does not define its tracks (it is a
    // table-column-group box, not a table-column box), so its tracks
    // stay eligible.
    let mut defined = vec![false; ncols];
    for &(_, start, span) in col_ranges.iter() {
        for d in defined
            .iter_mut()
            .take((start + span).min(ncols))
            .skip(start)
        {
            *d = true;
        }
    }
    merge_redundant_columns_inner(placed, col_ranges, group_ranges, &defined, ncols, true)
}

/// Whether a `col` box's specified width defines its tracks against
/// auto-mode track merging (CSS Tables 3 §3.3.2, WPT
/// `column-track-merging`): a `col` with `width: auto` (or a zero
/// width) leaves its tracks mergeable; any other specified width pins
/// them. For vertical tables the read is the transposed inline spec
/// (`height`).
fn col_width_defines_tracks(forest: &Forest, col_node: usize, vertical: bool) -> bool {
    match inline_axis_spec(&forest.nodes[col_node].data, vertical) {
        SpecLen::Auto => false,
        SpecLen::Px(v)
        | SpecLen::Em(v)
        | SpecLen::Rem(v)
        | SpecLen::Vw(v)
        | SpecLen::Vh(v)
        | SpecLen::Vmin(v)
        | SpecLen::Vmax(v)
        | SpecLen::Pct(v) => v != 0.0,
        // `fit-content`/`min-content`/`max-content`, `calc()`, and
        // anything else specified: conservative, defines the tracks.
        _ => true,
    }
}

/// Pooled buffers for table-grid collection.
///
/// A table's grid is collected 5-8 times per layout pass (the intrinsic
/// pre-passes, caption widths, the block width branch, `layout_table`
/// itself), and every collection allocated ~20 `Vec`s plus one per-row
/// `Vec`. The pool carries capacity — never data — across collections,
/// so repeat collections after warmup allocate nothing. Checking a grid
/// out ([`take_grid`]), collecting into it ([`collect_table_grid_into`]),
/// and returning it ([`put_grid`]) keeps the buffers alive across the pass.
///
/// Reentrancy: a table holds its checked-out grid for its whole layout
/// (cell contents lay out while it is checked out), so a nested table's
/// collection finds the pool empty and falls back to a fresh grid,
/// returning it when done. The outer table's grid is unaffected — it
/// owns its buffer for the duration.
#[derive(Default)]
pub(crate) struct TableGridPool {
    /// The reusable output grid.
    grid: TableGrid,
    /// Collected rows in visual order; inner vecs are retained across
    /// collections and only `..nrows` entries are live per collection.
    rows: Vec<Vec<usize>>,
    section_last: Vec<(Option<usize>, usize)>,
    /// Occupancy grid; inner vecs retained, only `..occ_len` live.
    occupied: Vec<Vec<bool>>,
    placed: Vec<(usize, usize, usize, usize, usize, bool, usize)>,
    /// Per-row placement order, reused across rows.
    order: Vec<usize>,
    fixed: Vec<(usize, usize, usize, usize, usize, usize)>,
    /// Zero-colspan rebuild occupancy (only used on that path).
    occ: Vec<Vec<bool>>,
    col_ranges: Vec<(usize, usize, usize)>,
    group_ranges: Vec<(usize, usize, usize)>,
    /// Bare-cell run accumulator, shared by the table-level and
    /// group-level runs (never live simultaneously).
    bare: Vec<usize>,
    /// [`merge_fixed_redundant_rows`] scratch.
    merge_ends: Vec<usize>,
    merge_deleted: Vec<usize>,
}

// Cells of one row: out-of-flow children are hoisted out (they take
// no space in the grid); floated children lay out through the float
// machinery; everything else that is not a cell is skipped.
fn row_cells_into(
    forest: &Forest,
    row: usize,
    abs_children: &mut Vec<usize>,
    floats: &mut Vec<usize>,
    cells: &mut Vec<usize>,
) {
    cells.clear();
    for &c in forest.nodes[row].children.iter() {
        if layout_position::is_out_of_flow(forest, c) {
            abs_children.push(c);
        } else if is_float(forest, c) {
            floats.push(c);
        } else if forest.nodes[c].box_type == BoxType::TableCell {
            cells.push(c);
        }
    }
}

impl TableGridPool {
    /// Check out the pooled grid buffer. Pair with [`put_grid`]: the
    /// pool is empty (and a fresh grid is handed out) while a grid is
    /// checked out, so nested tables never alias the outer table's
    /// buffer.
    pub(crate) fn take_grid(&mut self) -> TableGrid {
        std::mem::take(&mut self.grid)
    }

    /// Return a checked-out grid to the pool; its buffers serve the
    /// next collection.
    pub(crate) fn put_grid(&mut self, grid: TableGrid) {
        self.grid = grid;
    }

    /// An empty pool: every buffer allocates on first use.
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Collect the table grid into `grid`, reusing `pool`'s buffers.
/// Every buffer is cleared first, so the result is identical to a
/// fresh [`collect_table_grid`] call. A free function (rather than a
/// [`TableGridPool`] method) so the long body below keeps its shape.
pub(crate) fn collect_table_grid_into(
    forest: &Forest,
    idx: usize,
    pool: &mut TableGridPool,
    grid: &mut TableGrid,
) {
    let TableGridPool {
        grid: _,
        rows,
        section_last,
        occupied,
        placed,
        order,
        fixed,
        occ,
        col_ranges,
        group_ranges,
        bare,
        merge_ends,
        merge_deleted,
    } = pool;
    // The pool carries capacity, never data: clear every buffer so
    // the collection below reads exactly what it writes. Row and
    // occupancy inners are cleared but retained; only `..nrow` /
    // `..occ_len` entries are live per collection.
    for r in rows.iter_mut() {
        r.clear();
    }
    for o in occupied.iter_mut() {
        o.clear();
    }
    section_last.clear();
    placed.clear();
    fixed.clear();
    col_ranges.clear();
    group_ranges.clear();
    bare.clear();
    grid.cells.clear();
    grid.collapsed_rows.clear();
    grid.collapsed_cols.clear();
    grid.row_nodes.clear();
    grid.row_group.clear();
    grid.group_nodes.clear();
    grid.group_kinds.clear();
    grid.captions.clear();
    grid.columns.clear();
    grid.col_nodes.clear();
    grid.col_group_nodes.clear();
    grid.floats.clear();
    grid.abs_children.clear();
    // Live row count in `rows` (`rows[..nrow]`); live occupancy
    // rows in `occupied` (`occupied[..occ_len]`).
    let mut nrow = 0usize;
    let mut occ_len = 0usize;

    let nchildren = forest.nodes[idx].children.len();
    // Consecutive bare cells directly under the table share one
    // anonymous row (CSS 2.1 §17.2.1); anything but inter-cell
    // whitespace flushes the run (via `bare`). Declared column
    // cursor: the next logical column index covered by `col` /
    // `colgroup` structure.
    let mut decl_col: usize = 0;
    for ci in 0..nchildren {
        let c = forest.nodes[idx].children[ci];
        let is_bare = forest.nodes[c].box_type == BoxType::TableCell
            && !layout_position::is_out_of_flow(forest, c);
        if !is_bare && !is_ws_only(forest, c) {
            flush_bare_cells(
                rows,
                &mut nrow,
                &mut grid.row_nodes,
                &mut grid.row_group,
                bare,
                None,
            );
        }
        if layout_position::is_out_of_flow(forest, c) {
            grid.abs_children.push(c);
            continue;
        }
        if is_float(forest, c) {
            grid.floats.push(c);
            continue;
        }
        match forest.nodes[c].box_type {
            BoxType::TableRowGroup => {
                let kind = row_group_section_kind(forest, c)
                    .expect("row-group boxes are stamped with a section kind at construction");
                push_row_group(&mut grid.group_nodes, &mut grid.group_kinds, c, kind);
                let nrows = forest.nodes[c].children.len();
                // Consecutive bare cells in the group share one
                // anonymous row (CSS 2.1 §17.2.1); anything but
                // inter-cell whitespace flushes the run.
                for ri in 0..nrows {
                    let r = forest.nodes[c].children[ri];
                    let is_gbare = forest.nodes[r].box_type == BoxType::TableCell
                        && !layout_position::is_out_of_flow(forest, r);
                    if !is_gbare && !is_ws_only(forest, r) {
                        flush_bare_cells(
                            rows,
                            &mut nrow,
                            &mut grid.row_nodes,
                            &mut grid.row_group,
                            bare,
                            Some(c),
                        );
                    }
                    if layout_position::is_out_of_flow(forest, r) {
                        grid.abs_children.push(r);
                    } else if is_float(forest, r) {
                        grid.floats.push(r);
                    } else if forest.nodes[r].box_type == BoxType::TableRow {
                        if rows.len() == nrow {
                            rows.push(Vec::new());
                        }
                        row_cells_into(
                            forest,
                            r,
                            &mut grid.abs_children,
                            &mut grid.floats,
                            &mut rows[nrow],
                        );
                        grid.row_nodes.push(Some(r));
                        grid.row_group.push(Some(c));
                        nrow += 1;
                    } else if is_gbare {
                        bare.push(r);
                    }
                    // Anything else under a row group (column boxes,
                    // stray content) is skipped: it is not grid content.
                }
                flush_bare_cells(
                    rows,
                    &mut nrow,
                    &mut grid.row_nodes,
                    &mut grid.row_group,
                    bare,
                    Some(c),
                );
            }
            BoxType::TableRow => {
                if rows.len() == nrow {
                    rows.push(Vec::new());
                }
                row_cells_into(
                    forest,
                    c,
                    &mut grid.abs_children,
                    &mut grid.floats,
                    &mut rows[nrow],
                );
                grid.row_nodes.push(Some(c));
                grid.row_group.push(None);
                nrow += 1;
            }
            BoxType::TableCell => {
                if is_bare {
                    bare.push(c);
                }
            }
            BoxType::TableColumn => {
                // A bare `col` directly under the table (outside any
                // `colgroup`) still declares columns.
                if !grid.col_nodes.contains(&c) {
                    grid.col_nodes.push(c);
                }
                let span = forest.nodes[c].colspan.max(1) as usize;
                col_ranges.push((c, decl_col, span));
                decl_col += span;
            }
            BoxType::TableColumnGroup => {
                if !grid.col_group_nodes.contains(&c) {
                    grid.col_group_nodes.push(c);
                }
                let group_start = decl_col;
                let mut covered = 0usize;
                for &k in forest.nodes[c].children.iter() {
                    if layout_position::is_out_of_flow(forest, k) {
                        grid.abs_children.push(k);
                    } else if is_float(forest, k) {
                        grid.floats.push(k);
                    } else if forest.nodes[k].box_type == BoxType::TableColumn {
                        if !grid.col_nodes.contains(&k) {
                            grid.col_nodes.push(k);
                        }
                        let span = forest.nodes[k].colspan.max(1) as usize;
                        col_ranges.push((k, group_start + covered, span));
                        covered += span;
                    }
                    // Anything else inside a `colgroup` (text, stray
                    // boxes) is not column structure and is skipped.
                }
                // A `colgroup` with no `col` children spans its own
                // `span` attribute's columns itself.
                if covered == 0 {
                    covered = forest.nodes[c].colspan.max(1) as usize;
                }
                group_ranges.push((c, group_start, group_start + covered));
                decl_col = group_start + covered;
            }
            // Real captions (`display: table-caption`; the box itself is a
            // plain block) lay out full-width above/below the grid. Other
            // block-level children are stray content, which
            // `wrap_table_stray_content` wraps in anonymous rows/cells
            // before collection; anything still bare here (the min-content
            // width path skips the wrap, and stray content is not part of
            // that lower bound) is dropped.
            BoxType::Block | BoxType::Grid
                if !is_ws_only(forest, c) && is_table_caption(forest, c) =>
            {
                grid.captions.push(c);
            }
            _ => {}
        }
    }
    flush_bare_cells(
        rows,
        &mut nrow,
        &mut grid.row_nodes,
        &mut grid.row_group,
        bare,
        None,
    );

    // Place cells into the grid, tracking rowspan occupancy. `rows` only
    // holds rows that contributed cells; rowspan can extend the grid past
    // the last such row (those rows stay empty and zero-height).
    //
    // `rowspan="0"` ("span to the end of the row-group section") is
    // resolved during placement: the section's rows are all collected
    // already, so the occupancy marks the full span up front and cells in
    // later rows are placed around it instead of overlapping it once the
    // span resolves. `colspan="0"` ("span to the end of the column group,
    // or the table") still resolves in the second pass against the final
    // column count; within each row those cells are placed last (in DOM
    // order), so no cell placed after them in the row lands inside the
    // columns the span will eventually cover.
    //
    // Bound the total grid area: shrink the spans so the filled
    // rectangle `(r..r+rs) \u00d7 (c..c+cs)` keeps the whole occupancy
    // grid within `MAX_TABLE_GRID_CELLS`. Rows/columns only ever
    // grow to `r + rs` / `c + cs`, and `r + rs` / `c + cs` are
    // each at most `MAX_TABLE_GRID_CELLS` by induction, so the
    // divisions below cannot divide by zero. When the grid
    // already reached the cap on an axis (`room_cols == 0` or
    // `room_rows == 0`), the cell is dropped outright: forcing a
    // 1-wide column/row here would push the occupancy grid past
    // the cap, so the soft clamp's "keeps the whole occupancy
    // grid within" claim would be false at the boundary.
    //
    // Last collected row of each row-group section, for `rowspan="0"`
    // resolution during placement (rows outside any group form one
    // section keyed by `None`).
    for (rr, g) in grid.row_group.iter().enumerate() {
        match section_last.iter_mut().find(|(gg, _)| *gg == *g) {
            Some(entry) => entry.1 = rr,
            None => section_last.push((*g, rr)),
        }
    }

    // (cell node, row, col, effective colspan, effective rowspan, colspan
    // was zero, index within `row`), in placement order. The effective
    // spans are the occupancy spans after the grid-cap shrink, so a
    // `GridCell` built from them never claims cells outside the marked
    // grid; the second pass only grows zero colspans to the end (with a
    // displacing fixup, below).
    for (r, row) in rows[..nrow].iter().enumerate() {
        let gi = grid.row_group[r];
        if occupied.len() <= r {
            occupied.push(Vec::new());
        }
        occ_len = occ_len.max(r + 1);
        // Placement order within the row: cells with a nonzero colspan
        // first (DOM order), then `colspan="0"` cells (DOM order). The
        // sort is stable, so DOM order is kept within each group.
        order.clear();
        order.extend(0..row.len());
        order.sort_by_key(|&i| forest.nodes[row[i]].colspan == 0);
        let mut c = 0;
        for &pi in order.iter() {
            let cell = row[pi];
            while occupied[r].get(c).copied().unwrap_or(false) {
                c += 1;
            }
            let cs_raw = forest.nodes[cell].colspan.min(1000);
            let rs_raw = forest.nodes[cell].rowspan.min(1000);
            let cs_zero = cs_raw == 0;
            // Provisional spans for occupancy: a zero colspan covers one
            // column for now (the second pass grows it to the end); a
            // zero rowspan resolves immediately against the section's
            // final row count.
            let cs = cs_raw.max(1) as usize;
            let rs = if rs_raw == 0 {
                section_last
                    .iter()
                    .find(|(g, _)| *g == gi)
                    .map(|&(_, last)| last)
                    .unwrap_or(r)
                    .saturating_sub(r)
                    + 1
            } else {
                rs_raw as usize
            };
            let max_cols = MAX_TABLE_GRID_CELLS / (r + rs).max(1);
            let room_cols = max_cols.saturating_sub(c);
            if room_cols == 0 {
                continue;
            }
            let cs = cs.min(room_cols);
            let max_rows = MAX_TABLE_GRID_CELLS / (c + cs).max(1);
            let room_rows = max_rows.saturating_sub(r);
            if room_rows == 0 {
                continue;
            }
            let rs = rs.min(room_rows);
            for rr in r..r + rs {
                if occupied.len() <= rr {
                    occupied.push(Vec::new());
                }
                occ_len = occ_len.max(rr + 1);
                if occupied[rr].len() < c + cs {
                    occupied[rr].resize(c + cs, false);
                }
                occupied[rr][c..c + cs].fill(true);
            }
            placed.push((cell, r, c, cs, rs, cs_zero, pi));
            c += cs;
        }
    }
    let nrows = occ_len;
    let mut ncols = occupied[..occ_len]
        .iter()
        .map(|r| r.len())
        .max()
        .unwrap_or(0);
    // Declared column structure can widen the grid: the logical column
    // count is the greater of the cell grid and the declared columns.
    ncols = ncols.max(decl_col);

    // Second pass: resolve `colspan="0"` ("span to the end") now that the
    // logical column count is known.
    let col_end = |c: usize| {
        let mut end = ncols;
        for &(_, gstart, gend) in group_ranges.iter() {
            if gstart <= c && c < gend.min(ncols) {
                end = gend.min(ncols);
                break;
            }
        }
        end
    };
    // (cell node, row, col, colspan, rowspan, index within `row`).
    fixed.clear();
    fixed.reserve(placed.len());
    if placed
        .iter()
        .any(|&(_, _, _, _, rs, cs_zero, _)| cs_zero && rs != 1)
    {
        // A multi-row zero-colspan cell's resolved span can cover cells
        // placed later (its provisional 1-column mark let later rows
        // place into columns the final span covers). Rebuild occupancy
        // from the final spans in placement order, displacing conflicting
        // cells rightward instead of overlapping them.
        for o in occ.iter_mut() {
            o.clear();
        }
        let mut grown = ncols;
        for &(cell, r, c0, cs_eff, rs_eff, cs_zero, pi) in placed.iter() {
            let end = if cs_zero { col_end(c0) } else { ncols };
            let (mut c, mut cs, mut rs) = (
                c0,
                if cs_zero {
                    end.saturating_sub(c0).max(1)
                } else {
                    cs_eff
                },
                rs_eff,
            );
            let mut drop = false;
            loop {
                let max_cols = MAX_TABLE_GRID_CELLS / (r + rs).max(1);
                let room_cols = max_cols.saturating_sub(c);
                if room_cols == 0 {
                    drop = true;
                    break;
                }
                cs = cs.min(room_cols);
                let max_rows = MAX_TABLE_GRID_CELLS / (c + cs).max(1);
                let room_rows = max_rows.saturating_sub(r);
                if room_rows == 0 {
                    drop = true;
                    break;
                }
                rs = rs.min(room_rows);
                while occ.len() < r + rs {
                    occ.push(Vec::new());
                }
                let mut conflict = false;
                for row in occ.iter_mut().skip(r).take(rs) {
                    if row.len() < c + cs {
                        row.resize(c + cs, false);
                    }
                    if row[c..c + cs].contains(&true) {
                        conflict = true;
                        break;
                    }
                }
                if !conflict {
                    break;
                }
                c += 1;
                if cs_zero {
                    cs = end.saturating_sub(c).max(1);
                }
            }
            if drop {
                continue;
            }
            for row in occ.iter_mut().skip(r).take(rs) {
                row[c..c + cs].fill(true);
            }
            grown = grown.max(c + cs);
            fixed.push((cell, r, c, cs, rs, pi));
        }
        ncols = grown;
    } else {
        for &(cell, r, c, cs_eff, rs_eff, cs_zero, pi) in placed.iter() {
            // `colspan="0"`: span from the start column to the end of the
            // containing column group, or to the end of the table when the
            // cell sits outside any declared group (at least 1 column).
            let cs = if cs_zero {
                col_end(c).saturating_sub(c).max(1)
            } else {
                cs_eff
            };
            fixed.push((cell, r, c, cs, rs_eff, pi));
        }
    }
    // Back to DOM order: `fixed` is in placement order (zero-colspan
    // cells deferred within their row).
    fixed.sort_by_key(|&(_, r, _, _, _, pi)| (r, pi));
    // CSS Tables 3 §3.3.2 track merging: in fixed mode only rows are
    // eligible, so every column is preserved. A row track with no
    // table-row box defining it explicitly, spanned by exactly the same
    // set of cells as the previous track, merges into that track: the
    // spanning cells' `rowspan` shrinks by one and cells originating
    // past the deleted track shift up. (WPT
    // `html5-table-formatting-fixed-layout-1.html`: one `rowspan=3`
    // cell over three implicit row tracks lays out as a single row.)
    // In auto mode, column tracks merge instead (every track eligible;
    // a `col` with a specified non-zero width defines its tracks, a
    // `colgroup` alone does not) — see the `else` branch below and
    // [`merge_redundant_columns_inner`]. The fixed-mode column-track
    // half of the sweep, [`merge_fixed_redundant_columns`], stays
    // unwired: the spec limits fixed-mode merging to rows.
    let mut nrows = nrows;
    let fixed_flag = table_is_fixed(forest, idx);
    if fixed_flag {
        nrows =
            merge_fixed_redundant_rows(fixed, &grid.row_nodes, nrows, merge_ends, merge_deleted);
    } else {
        // CSS Tables 3 §3.3.2 auto-mode column track merging: a column
        // track with no originating cells merges into the previous
        // track, unless a `col` box with a specified non-zero width
        // defines it (WPT `column-track-merging`). A `colgroup` alone
        // never defines tracks. A cell ending at the track blocks the
        // merge only when the track is non-empty — an empty track has
        // no spanning set to preserve, so it merges away (with its
        // spacing) instead of surviving as a zero-width track that
        // still draws border-spacing (see
        // [`merge_redundant_columns_inner`]). The merge shrinks
        // spanning cells' `colspan` and remaps the declared ranges, so
        // `ncols` and the per-column descriptors below see the merged
        // grid.
        let mut defined = vec![false; ncols];
        for &(node, start, span) in col_ranges.iter() {
            if !col_width_defines_tracks(forest, node, super::is_vertical_table(forest, idx)) {
                continue;
            }
            for d in defined
                .iter_mut()
                .take((start + span).min(ncols))
                .skip(start)
            {
                *d = true;
            }
        }
        // Tracks spanned by a `colspan="0"` cell ("span to the end") are
        // defined: the cell explicitly claims them, so they never merge.
        for (cell, _, col, colspan, _, _) in fixed.iter() {
            if forest.nodes[*cell].colspan == 0 {
                for d in defined
                    .iter_mut()
                    .take((*col + *colspan).min(ncols))
                    .skip(*col)
                {
                    *d = true;
                }
            }
        }
        ncols =
            merge_redundant_columns_inner(fixed, col_ranges, group_ranges, &defined, ncols, true);
    }
    grid.cells.extend(
        fixed
            .drain(..)
            .map(|(node, row, col, colspan, rowspan, _)| GridCell {
                node,
                row,
                col,
                colspan,
                rowspan,
                // Resolved below, once `collapse`/`vertical` are set.
                intrinsic_pad: 0.0,
            }),
    );

    // Fill the per-column descriptors from the declared ranges.
    let columns = &mut grid.columns;
    columns.clear();
    columns.resize(
        ncols,
        ColumnSpec {
            col_node: None,
            group_node: None,
        },
    );
    for &(node, start, span) in col_ranges.iter() {
        for col in columns
            .iter_mut()
            .take((start + span).min(ncols))
            .skip(start)
        {
            col.col_node = Some(node);
        }
    }
    for &(node, start, end) in group_ranges.iter() {
        for col in columns.iter_mut().take(end.min(ncols)).skip(start) {
            col.group_node = Some(node);
        }
    }

    // CSS 2.1 §17.5 `visibility: collapse`: collapsed rows/columns take
    // no space (unlike `hidden`). The flags below zero the collapsed
    // lines during width/height resolution; paint skips their subtrees
    // outright. A row group collapses every row it contains (anonymous
    // rows included) as a unit: a row that explicitly overrides back to
    // `visible` still takes no space, since its row group is removed
    // from the display and the row has no boxes to show in (WPT
    // `visibility-collapse-row-group-001`: "Collapsing a row group but
    // making its rows visible should still result in a collapsed row
    // group"). `rowspan` can extend the grid past the last collected
    // row; those extra rows have no boxes and never collapse.
    let collapsed_rows = &mut grid.collapsed_rows;
    collapsed_rows.clear();
    collapsed_rows.resize(nrows, false);
    for (r, flag) in collapsed_rows.iter_mut().enumerate() {
        let row_collapsed = grid
            .row_nodes
            .get(r)
            .copied()
            .flatten()
            .is_some_and(|n| visibility_collapsed(forest, n));
        let group_collapsed = grid
            .row_group
            .get(r)
            .copied()
            .flatten()
            .is_some_and(|g| visibility_collapsed(forest, g));
        *flag = row_collapsed || group_collapsed;
    }
    grid.collapsed_cols.clear();
    grid.collapsed_cols.extend(grid.columns.iter().map(|spec| {
        spec.col_node
            .is_some_and(|n| visibility_collapsed(forest, n))
            || spec
                .group_node
                .is_some_and(|n| visibility_collapsed(forest, n))
    }));

    debug_assert_eq!(
        grid.group_nodes.len(),
        grid.group_kinds.len(),
        "group_kinds stays parallel to group_nodes"
    );
    grid.nrows = nrows;
    grid.ncols = ncols;
    grid.fixed = fixed_flag;
    // Fixed mode with a non-definite specified width sizes the table
    // from the pinned first-row column widths alone (CSS 2.1
    // §17.5.2.1); the intrinsic passes must not measure content
    // (WPT `css/css-tables/fixed-layout-2.html`).
    grid.fixed_indefinite = fixed_flag && table_width_indefinite(forest, idx);
    grid.collapse = table_border_model(forest, idx).0;
    grid.vertical = super::is_vertical_table(forest, idx);
    // Each cell's intrinsic padding, resolved once per layout: the
    // intrinsic passes (`auto_column_maxima` / `auto_column_minima`)
    // read `cell.intrinsic_pad` instead of recomputing
    // `cell_intrinsic_pad` per cell with identical arguments.
    let (collapse, vertical) = (grid.collapse, grid.vertical);
    for cell in grid.cells.iter_mut() {
        cell.intrinsic_pad = cell_intrinsic_pad(forest, cell.node, collapse, vertical);
    }
}

/// Fresh-collection entry point over [`collect_table_grid_into`], for
/// tests with no pool at hand.
#[cfg(test)]
pub(crate) fn collect_table_grid(forest: &Forest, idx: usize) -> TableGrid {
    let mut pool = TableGridPool::new();
    let mut grid = pool.take_grid();
    collect_table_grid_into(forest, idx, &mut pool, &mut grid);
    grid
}

/// The table's border model: whether `border-collapse` is `collapse`, and
/// the horizontal/vertical `border-spacing` (CSS 2.1 §17.5.2: one value
/// sets both axes; 0 when unspecified or when collapsing).
pub(crate) fn table_border_model(forest: &Forest, idx: usize) -> (bool, f32, f32) {
    let n = &forest.nodes[idx];
    let style = n.style.as_ref();
    let collapse = style
        .and_then(|s| s.get_str_by_index(props::BORDER_COLLAPSE))
        .is_some_and(|v| v.eq_ignore_ascii_case("collapse"));
    let spacing = if collapse {
        (0.0, 0.0)
    } else {
        let (fs, root_fs, vp_w, vp_h, tw) = (
            n.data.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            n.metrics.content_w,
        );
        let len = |v: Option<&CssValue>| {
            v.and_then(|v| spec_len_of(Some(v)).resolve(fs, root_fs, vp_w, vp_h, tw))
                .unwrap_or(0.0)
                .max(0.0)
        };
        match style.and_then(|s| s.get_by_index(props::BORDER_SPACING)) {
            // Two values: horizontal then vertical.
            Some(CssValue::List(items)) => {
                let h = len(items.first());
                let v = items.get(1).map_or(h, |v| len(Some(v)));
                (h, v)
            }
            // One value: both axes.
            v => {
                let s = len(v);
                (s, s)
            }
        }
    };
    // For vertical tables, columns run vertically (inline axis): the
    // physical horizontal spacing applies between rows (block axis),
    // and the physical vertical spacing applies between columns.
    // Transpose so the algorithm-space uses (which assume spacing_h is
    // inter-column) stay correct.
    let (sh, sv) = spacing;
    let (sh, sv) = if super::is_vertical_table(forest, idx) {
        (sv, sh)
    } else {
        (sh, sv)
    };
    (collapse, sh, sv)
}

/// Resolve the specified widths of the table's column boxes: one entry
/// per logical column, the effective specified width in px, border-box
/// (`None` when neither its `col` nor its `colgroup` specifies one). A
/// `col` width applies to each column in the element's own span and wins
/// over its group's. A `colgroup` width applies to each covered column
/// without its own `col` width, at full value (in the automatic layout
/// the result is a minimum per CSS 2.1 §17.5.2.2; under
/// `table-layout: fixed` it pins the column per §17.5.2.1). Column-box
/// metrics are resolved first against the table's content width (their
/// containing-block width) so the result accounts for box-sizing and
/// percentages.
///
/// The second vector flags, per column, whether the width came only from
/// a `colgroup` (no `col` width of its own). A `col` width floors the
/// column's *maximum* (never the minimum — see `auto_column_minima`),
/// while a `colgroup` spread floors the minimum only; automatic layout
/// consults the flag to apply the maximum raise to `col` widths alone.
/// For vertical tables the spec read is the transposed inline spec
/// (`height`); the border-box size is likewise the vertical one.
pub(crate) fn column_specified_widths(
    forest: &mut Forest,
    grid: &TableGrid,
    cb_x: f32,
    cb_w: f32,
) -> (Vec<Option<f32>>, Vec<bool>) {
    for &node in grid.col_nodes.iter().chain(grid.col_group_nodes.iter()) {
        resolve_metrics(forest, node, cb_x, cb_w, false, None, false);
    }
    fn box_w(forest: &Forest, grid: &TableGrid, n: usize, cb_w: f32) -> Option<f32> {
        let node = &forest.nodes[n];
        // csswg#94 (see `table_col_spec`): a percentage-bearing `calc()`
        // with a nonzero length part counts as `auto` for column sizing.
        // A vertical table's inline axis is physical vertical: the
        // column's inline spec is its `height`, so a `width` on the
        // column (the block axis) must not floor the column's inline
        // size, and a `height:N%` resolves to the intrinsic percentage.
        let spec = table_col_spec(inline_axis_spec(&node.data, grid.vertical));
        if matches!(spec, SpecLen::Auto) {
            None
        } else if grid.vertical {
            // The inline border-box size: resolve the `height` spec
            // against the inline base (`cb_w` is the algorithm-space
            // table width) and apply box-sizing with the resolved
            // vertical border/padding (`resolve_metrics` above filled
            // the physical width into the metrics, which is the block
            // axis here).
            let d = &node.data;
            let m = &node.metrics;
            let v = spec
                .resolve(
                    d.font_size,
                    forest.root_font_size,
                    forest.vp_w,
                    forest.vp_h,
                    cb_w,
                )
                .unwrap_or(0.0)
                .max(0.0);
            let bp = m.bt + m.pt + m.pb + m.bb;
            Some(
                match d.box_sizing {
                    BoxSizing::BorderBox => v,
                    BoxSizing::ContentBox => v + bp,
                }
                .max(0.0),
            )
        } else {
            let m = &node.metrics;
            Some((m.content_w + m.bl + m.pl + m.pr + m.br).max(0.0))
        }
    }
    // A `col` width applies to each column in its own span.
    let mut out: Vec<Option<f32>> = grid
        .columns
        .iter()
        .map(|spec| spec.col_node.and_then(|n| box_w(forest, grid, n, cb_w)))
        .collect();
    // A `colgroup` width applies to each covered column without its own
    // `col` width, at full value: the group's width is the width of each
    // column it generates (WPT `colgroup-col`: a 100px `span=3` group
    // yields 100px columns, not 33px; a 100px group with 50px/150px
    // cols gives its auto col the full 100px). A `col` width overrides
    // the group's for its column. Group-sourced columns are flagged:
    // they raise the column minimum only.
    let mut group_only = vec![false; grid.columns.len()];
    for &g in grid.col_group_nodes.iter() {
        let Some(w) = box_w(forest, grid, g, cb_w) else {
            continue;
        };
        for (cc, spec) in grid.columns.iter().enumerate() {
            if spec.group_node == Some(g) && out[cc].is_none() {
                out[cc] = Some(w);
                group_only[cc] = true;
            }
        }
    }
    (out, group_only)
}

/// Add `extra` to `col_w` for the candidate columns: each candidate takes
/// `extra * w / total` of its weight (with no weights, or a zero total,
/// the candidates share equally). Shared by [`distribute_excess_width`]
/// (automatic layout) and the fixed-layout excess-distribution branch —
/// the two previously carried line-identical local copies.
fn share_excess_width(col_w: &mut [f32], cands: &[usize], weights: &[f32], extra: f32) {
    let total: f32 = weights.iter().sum();
    if total > 0.0 {
        for (&cc, &w) in cands.iter().zip(weights.iter()) {
            col_w[cc] += extra * w / total;
        }
    } else if !cands.is_empty() {
        let each = extra / cands.len() as f32;
        for &cc in cands {
            col_w[cc] += each;
        }
    }
}

/// Intrinsic percentage fraction of each column's effective
/// `<col>`/`<colgroup>` width (the `<col>` wins over its group), or
/// 0.0 when the effective width is not a pure percentage. Used to
/// rebase percentage-derived specified widths from the
/// spacing-inclusive content width onto the assignable width (see
/// `layout_table`): the fraction mirrors the spec
/// `column_specified_widths` resolved, so the rebase subtracts
/// exactly the percentage's share of the inter-column spacing.
fn col_spec_pct_fractions(forest: &Forest, grid: &TableGrid) -> Vec<f32> {
    let vertical = grid.vertical;
    grid.columns
        .iter()
        .map(|spec| {
            spec.col_node
                // A `col` whose own normalized spec is `Auto`
                // (`width: auto`, or a mixed `calc(% + px)` the csswg#94
                // `table_col_spec` normalization collapses to `Auto`)
                // defers to its group, exactly as
                // `column_specified_widths` does. Otherwise a
                // `<col style="width:auto">` inside a percentage
                // `<colgroup>` would resolve the group's percentage
                // against the spacing-inclusive content width and skip
                // the rebase below.
                .filter(|&n| {
                    !matches!(
                        table_col_spec(inline_axis_spec(&forest.nodes[n].data, vertical)),
                        SpecLen::Auto
                    )
                })
                .or(spec.group_node)
                .and_then(|n| {
                    pct_fraction(table_col_spec(inline_axis_spec(
                        &forest.nodes[n].data,
                        vertical,
                    )))
                })
                .unwrap_or(0.0)
        })
        .collect()
}

/// Per-column state for the §3.9.3.2 excess-distribution rules: see
/// [`column_excess_infos`].
struct ColExcessInfo {
    gone: bool,
    constrained: bool,
    pct: f32,
    max: f32,
    originating: bool,
    /// Largest definite specified width (px) from the column/group
    /// boxes or span-1 cells; 0.0 if none. Used to tell a truly
    /// pinned column (specified >= max) from one whose max is
    /// content-driven (max > specified).
    specified: f32,
}

/// Reusable scratch buffers for [`distribute_spanning_max`] and
/// [`distribute_spanning_min`]: the per-column excess-distribution
/// state, the span mask, the distribution list, and the fill
/// step's candidate columns plus their headroom weights. The
/// `infos`/`mask` reuse predates wave 61; that wave moved the
/// per-cell distribution list (`vis`) and the fill step's candidate
/// and headroom-weight buffers (`cands`/`weights`) into the scratch,
/// so reusing the buffers removes 3 allocations per spanning cell.
/// Call sites declare one and pass `&mut`.
#[derive(Default)]
struct SpanningCandidateScratch {
    infos: Vec<ColExcessInfo>,
    mask: Vec<bool>,
    /// Distribution list of the cell currently being distributed,
    /// rebuilt from the span range by [`distribute_spanning_min`]
    /// and [`distribute_spanning_max`] so the caller needs no
    /// per-cell `Vec`: the non-collapsed spanned columns, or the
    /// non-pinned ones when the max side skips fixed-layout pins.
    vis: Vec<usize>,
    /// Fill-step candidate columns (headroom above the minimum),
    /// cleared and refilled per [`distribute_spanning_min`] call.
    cands: Vec<usize>,
    /// Headroom weights parallel to [`Self::cands`].
    weights: Vec<f32>,
}

/// Build the per-column state the §3.9.3.2 rules consult.
///
/// A column is *constrained* when its column box, its column-group box,
/// or any cell spanning only it has a non-`auto`, non-percentage
/// `width` (with the csswg#94 `calc()` normalization from
/// [`table_col_spec`]); its *intrinsic percentage* is the largest
/// percentage width among those. `max` is the column's max-content
/// width from `col_max` (the measures the caller is distributing
/// against), and `originating` records whether any laid-out cell
/// originates in the column. When `pcts` is given, the intrinsic
/// percentages come from it instead of the span-1 measures — the
/// used-width excess distribution consults the *final* intrinsic
/// percentages (including spanning-cell redistribution, see
/// [`intrinsic_column_pcts`]), while each intrinsic spanning
/// distribution consults the running pre-update state (see
/// [`PctRunningState`]): the percentages as of before that cell's own
/// redistribution, in (span, DOM) order.
fn column_excess_infos(
    forest: &Forest,
    grid: &TableGrid,
    col_max: &[f32],
    pcts: Option<&[f32]>,
) -> Vec<ColExcessInfo> {
    let mut infos = Vec::new();
    column_excess_infos_into(forest, grid, col_max, pcts, &mut infos);
    infos
}

/// [`column_excess_infos`] refilling a reused buffer: `clear` +
/// `extend` keeps the capacity across [`distribute_spanning_max`] /
/// [`distribute_spanning_min`] calls instead of allocating a fresh
/// `Vec` per spanning cell. The refilled contents are identical to
/// the allocating version.
fn column_excess_infos_into(
    forest: &Forest,
    grid: &TableGrid,
    col_max: &[f32],
    pcts: Option<&[f32]>,
    infos: &mut Vec<ColExcessInfo>,
) {
    infos.clear();
    infos.extend(
        grid.collapsed_cols
            .iter()
            .zip(col_max.iter())
            .map(|(&gone, &max)| ColExcessInfo {
                gone,
                constrained: false,
                pct: 0.0,
                max,
                originating: false,
                specified: 0.0,
            }),
    );
    // Column and column-group boxes constrain, or contribute an
    // intrinsic percentage to, each column they cover. The spec reads
    // are inline-axis (transposed for vertical tables): a `height:N%`
    // becomes the intrinsic percentage there, while a `width:N%` (the
    // block axis) must not.
    for (cc, spec) in grid.columns.iter().enumerate() {
        if infos[cc].gone {
            continue;
        }
        for node in [spec.col_node, spec.group_node].into_iter().flatten() {
            match table_col_spec(inline_axis_spec(&forest.nodes[node].data, grid.vertical)) {
                SpecLen::Auto => {}
                SpecLen::Pct(p) => {
                    if pcts.is_none() {
                        infos[cc].pct = infos[cc].pct.max(p / 100.0);
                    }
                }
                SpecLen::Px(px) => {
                    infos[cc].constrained = true;
                    infos[cc].specified = infos[cc].specified.max(px);
                }
                _ => {
                    infos[cc].constrained = true;
                }
            }
        }
    }
    // Cells: every width-contributing cell originates in its first
    // column — collapsed-row cells never lay out, but they still size
    // columns (CSS 2.1 §17.5.5), so they contribute here too; only
    // cells spanning exactly that column feed its constrainedness and
    // intrinsic percentage (css-tables-3 §3.8.3 "constrainedness").
    for cell in &grid.cells {
        if !cell_width_contributes(&grid.collapsed_cols, cell) {
            continue;
        }
        if cell.col < infos.len() && !infos[cell.col].gone {
            infos[cell.col].originating = true;
            // Constrainedness reads the pre-merge span: a merged
            // spanning cell's specified width was for the span, not
            // the single track (see `GridCell::was_single_col`).
            if cell.was_single_col(forest) {
                match table_col_spec(inline_axis_spec(
                    &forest.nodes[cell.node].data,
                    grid.vertical,
                )) {
                    SpecLen::Auto => {}
                    SpecLen::Pct(p) => {
                        if pcts.is_none() {
                            infos[cell.col].pct = infos[cell.col].pct.max(p / 100.0);
                        }
                    }
                    SpecLen::Px(px) => {
                        infos[cell.col].constrained = true;
                        infos[cell.col].specified = infos[cell.col].specified.max(px);
                    }
                    _ => {
                        infos[cell.col].constrained = true;
                    }
                }
            }
        }
    }

    if let Some(pcts) = pcts {
        for (info, &p) in infos.iter_mut().zip(pcts.iter()) {
            if !info.gone {
                info.pct = p;
            }
        }
    }
}

/// CSS Tables 3 §3.9.3.2 "distributing excess width to columns": add
/// `extra` to `col_w` following the rules in order — non-constrained
/// columns first (proportionally to max-content, else equally), then
/// constrained columns (proportionally to max-content), then
/// percentage columns (proportionally to the percentage), then
/// equally — where the first rule with any candidate columns takes
/// the whole excess. When `only` is given, candidates are restricted
/// to the flagged columns (a spanning cell's excess distributes only
/// over the columns it spans, for intrinsic width calculation).
/// Collapsed columns are never candidates and keep their zero width.
///
/// `cands`/`weights` are caller-reused buffers (the [`SpanningCandidateScratch`]
/// fields at the spanning-distribution call sites, one-shot locals at
/// [`distribute_excess_width`]): cleared on entry, so no allocation
/// happens per call after the first.
fn distribute_excess_among(
    col_w: &mut [f32],
    infos: &[ColExcessInfo],
    extra: f32,
    only: Option<&[bool]>,
    cands: &mut Vec<usize>,
    weights: &mut Vec<f32>,
) {
    let allowed = |cc: usize| only.is_none_or(|m| m[cc]);
    // Each rule below is tried in order; the first with candidates takes
    // the whole excess. A non-empty `weights` distributes proportionally
    // to it, an empty one equally.
    cands.clear();
    weights.clear();
    // Rule 1: non-constrained columns with no intrinsic percentage and
    // nonzero max-content, proportionally to max-content.
    for (cc, info) in infos.iter().enumerate() {
        if allowed(cc)
            && !info.gone
            && !info.constrained
            && info.pct == 0.0
            && info.max > 0.0
            && info.originating
        {
            cands.push(cc);
            weights.push(info.max);
        }
    }
    if !cands.is_empty() {
        share_excess_width(col_w, &cands[..], &weights[..], extra);
        return;
    }
    // Rule 2: remaining non-constrained columns with no intrinsic
    // percentage (zero max-content here, by rule 1), equally.
    cands.clear();
    weights.clear();
    for (cc, info) in infos.iter().enumerate() {
        if allowed(cc) && !info.gone && !info.constrained && info.pct == 0.0 && info.originating {
            cands.push(cc);
        }
    }
    if !cands.is_empty() {
        share_excess_width(col_w, &cands[..], &weights[..], extra);
        return;
    }
    // Rule 3: constrained columns with no intrinsic percentage and
    // nonzero max-content, proportionally to max-content.
    cands.clear();
    for (cc, info) in infos.iter().enumerate() {
        if allowed(cc) && !info.gone && info.constrained && info.pct == 0.0 && info.max > 0.0 {
            cands.push(cc);
            weights.push(info.max);
        }
    }
    if !cands.is_empty() {
        share_excess_width(col_w, &cands[..], &weights[..], extra);
        return;
    }
    // Rule 4: percentage columns, proportionally to the percentage.
    cands.clear();
    weights.clear();
    for (cc, info) in infos.iter().enumerate() {
        if allowed(cc) && !info.gone && info.pct > 0.0 {
            cands.push(cc);
            weights.push(info.pct);
        }
    }
    if !cands.is_empty() {
        share_excess_width(col_w, &cands[..], &weights[..], extra);
        return;
    }
    // Rule 5: columns with originating cells, equally.
    cands.clear();
    for (cc, info) in infos.iter().enumerate() {
        if allowed(cc) && !info.gone && info.originating {
            cands.push(cc);
        }
    }
    if !cands.is_empty() {
        share_excess_width(col_w, &cands[..], &weights[..], extra);
        return;
    }
    // Rule 6: all visible columns, equally.
    cands.clear();
    for (cc, info) in infos.iter().enumerate() {
        if allowed(cc) && !info.gone {
            cands.push(cc);
        }
    }
    share_excess_width(col_w, &cands[..], &weights[..], extra);
}

/// One spanning cell's max-content excess distributed over the spanned
/// columns' maxima, per §3.9.3.2 invoked for intrinsic width
/// calculation (the §3.8.3 editorial note points the spanning
/// distribution at these rules). Unlike the old per-level snapshot
/// merge, this updates `col_max` in place: spanning cells are
/// processed cumulatively in (span, DOM) order, each distributing
/// over the *current* measures — a later cell sees the earlier
/// cells' contributions (WPT `colspan-redistribution` tables 29-31:
/// the 400px cell's excess weights by the maxima the 200px cell
/// already raised).
///
/// `unconstrained_cell` marks a cell with an `auto` (non-percentage,
/// non-length) width: such a cell's max-content excess never lands on
/// a truly pinned spanned column (specified width >= max-content) —
/// only constrained spanning cells distribute max-width over pinned
/// columns (WPT `colspan-redistribution` table 25 vs 24). A column
/// whose max-content exceeds its specified width is content-driven,
/// not pinned, and still takes excess (table 13). Pinned columns are
/// therefore ineligible recipients here; when every spanned column
/// is pinned the excess is dropped, not forced elsewhere.
///
/// The distribution list is rebuilt from the span range in
/// `scratch.vis` (no per-cell `Vec` at the call sites): with
/// `skip_pinned` it excludes pinned columns (the preferred-width
/// path's fixed-layout pins, which that call site used to exclude
/// via its own `open` list); without it the list excludes collapsed
/// columns, exactly like the old `vis` argument.
#[allow(clippy::too_many_arguments)]
fn distribute_spanning_max(
    forest: &Forest,
    grid: &TableGrid,
    col_max: &mut [f32],
    col: usize,
    end: usize,
    need_max: f32,
    gap: f32,
    unconstrained_cell: bool,
    cell_pct: f32,
    skip_pinned: Option<&[bool]>,
    pcts: Option<&[f32]>,
    scratch: &mut SpanningCandidateScratch,
) {
    // Destructure so the overflow step can borrow `mask`/`infos`
    // while the distribution list is read, and the final call can
    // borrow `cands`/`weights` mutably alongside: the fields are
    // disjoint, so no borrow ever overlaps a live one.
    let SpanningCandidateScratch {
        infos,
        mask,
        vis,
        cands,
        weights,
    } = scratch;
    // Rebuild the distribution list from the span range in the
    // reused buffer: no per-cell allocation after the first.
    vis.clear();
    match skip_pinned {
        Some(pinned) => vis.extend((col..end).filter(|&cc| !pinned[cc])),
        None => vis.extend((col..end).filter(|&cc| !grid.collapsed_cols[cc])),
    }
    if vis.is_empty() {
        return;
    }
    let span_sum: f32 = vis.iter().map(|&cc| col_max[cc]).sum::<f32>() + gap;
    let excess = need_max - span_sum;
    if excess <= 0.0 {
        return;
    }
    column_excess_infos_into(forest, grid, col_max, pcts, &mut *infos);
    // A spanning cell whose percentage width failed to redistribute
    // sizes nothing (WPT `colspan-redistribution` table 4: the 19%
    // cell over the 20% column distributes neither min nor max).
    if cell_pct > 0.0 {
        let spanned_pct: f32 = vis.iter().map(|&cc| infos[cc].pct).sum();
        if cell_pct <= spanned_pct {
            return;
        }
    }
    // If every spanned column has an intrinsic percentage, the
    // spanning cell's content does not size the columns (see
    // `distribute_spanning_min`).
    let all_pct = vis.iter().all(|&cc| infos[cc].pct > 0.0 && !infos[cc].gone);
    if all_pct {
        return;
    }
    if unconstrained_cell {
        // An unconstrained spanning cell does not grow a truly
        // pinned column's maximum: if the column's max-content width
        // is at or below its definite specified width, the specified
        // width pins it and the excess must go elsewhere (WPT
        // `colspan-redistribution` table 25: the 200px cell leaves
        // the 50px `<col>` columns alone). If content exceeds the
        // specified width the max is content-driven and the column
        // still takes excess (table 13: the 300px cell raises the
        // 20px-specified/100px-content column to 146px).
        for &cc in vis.iter() {
            let info = &mut infos[cc];
            // The fudge is float-noise scale (1e-3): content exceeding
            // the specified width by any real amount makes the column
            // content-driven, not pinned. A larger epsilon (the old
            // 0.5) misclassifies a column like 50px-specified/50.4px-
            // content as pinned and drops the spanning excess.
            if info.constrained && info.specified >= col_max[cc] - 1e-3 {
                info.gone = true;
            }
        }
    }
    mask.clear();
    mask.resize(col_max.len(), false);
    for &cc in vis.iter() {
        mask[cc] = true;
    }
    distribute_excess_among(col_max, infos, excess, Some(&mask[..]), cands, weights);
}

/// One spanning cell's min-content excess distributed over the spanned
/// columns' minima, per §3.9.3.2 invoked for intrinsic width
/// calculation. Cumulative like [`distribute_spanning_max`]: `col_min`
/// is updated in place and each cell sees the earlier cells'
/// contributions.
///
/// The distribution runs in three steps, all measured from Chromium
/// (TablesNG):
///
/// 1. Percentage columns resolve first: each takes `pct * (need - gap)`
///    floored at its current minimum (WPT `colspan-redistribution`
///    table 23 for the mixed span where the 40% column takes 80px and
///    the fixed column the rest). This holds whether or not the
///    distributing cell itself has a percentage width.
/// 2. Fill: remaining excess brings columns from their minima toward
///    their maxima, proportionally to the headroom (`max - min`)
///    (WPT table 19: the 150px cell over 70/70/50px maxima fills the
///    two fixed columns to 42/42 and leaves the auto column at 50).
/// 3. Overflow: any excess left after the fill follows the §3.9.3.2
///    rules uncapped — non-constrained columns first (WPT table 18:
///    the 500px cell's overflow lands on the auto column), then
///    constrained columns proportionally to max-content (WPT tables
///    11/12: the 300px cell's overflow yields 146/146 minima), then
///    percentage columns proportionally to percentage.
///
/// `cell_pct` is the cell's specified percentage width, if any: when
/// it does not exceed the spanned columns' intrinsic percentages the
/// cell contributes no min/max-content at all — its percentage failed
/// to redistribute, so the cell sizes nothing (WPT table 4: the 19%
/// cell over a 20% column leaves the columns at 20/80).
#[allow(clippy::too_many_arguments)]
fn distribute_spanning_min(
    forest: &Forest,
    grid: &TableGrid,
    col_min: &mut [f32],
    col_max: &[f32],
    col: usize,
    end: usize,
    need_min: f32,
    spacing_h: f32,
    cell_pct: f32,
    pcts: Option<&[f32]>,
    scratch: &mut SpanningCandidateScratch,
) {
    // Destructure so the fill step can borrow `cands`/`weights`
    // mutably while reading `vis`/`infos`, and the overflow step can
    // borrow `mask`/`infos` after those borrows end: the fields are
    // disjoint, so no borrow ever overlaps a live one.
    let SpanningCandidateScratch {
        infos,
        mask,
        vis,
        cands,
        weights,
    } = scratch;
    column_excess_infos_into(forest, grid, col_max, pcts, &mut *infos);
    // Rebuild the visible spanned columns from the span range in the
    // reused buffer: no per-cell allocation after the first.
    vis.clear();
    vis.extend((col..end).filter(|&cc| !grid.collapsed_cols[cc]));
    if vis.is_empty() {
        return;
    }

    if cell_pct > 0.0 {
        let spanned_pct: f32 = vis.iter().map(|&cc| infos[cc].pct).sum();
        if cell_pct <= spanned_pct {
            return;
        }
    }
    // If every spanned column has an intrinsic percentage, the
    // spanning cell's content distributes proportionally to the
    // percentages (WPT `colspan-redistribution` tables 14/16/17: the
    // 300/468/400px cells' minima land at 146/146, 280/180, 245/147).
    // When the distributing cell itself has a percentage width, the
    // cell's percentage failed to redistribute, so the cell sizes
    // nothing (WPT table 4: the 19% cell over the 20% column leaves
    // the columns alone; the 40%-over-10%/10% case likewise).
    let all_pct = vis.iter().all(|&cc| infos[cc].pct > 0.0 && !infos[cc].gone);
    // The inter-column spacing a collapsed column would have drawn is
    // excluded: the gap covers only the visible spanned columns.
    let gap = spacing_h * vis.len().saturating_sub(1) as f32;
    let distributable = need_min - gap;
    // Percentage columns resolve against the distributable first.
    // When every spanned column has an intrinsic percentage and the
    // distributing cell is auto-width, the whole min-content spreads
    // proportionally to the percentages (not to headroom): each
    // column's floor becomes `pct * distributable` raised to at least
    // its current minimum, and any remainder spreads proportionally
    // to the percentages as well (WPT `colspan-redistribution` tables
    // 14/16/17).
    let total_pct: f32 = vis.iter().map(|&cc| infos[cc].pct).sum();
    if all_pct && total_pct > 0.0 {
        if cell_pct == 0.0 {
            for &cc in vis.iter() {
                col_min[cc] = (infos[cc].pct * distributable).max(col_min[cc]);
            }
            let remainder = distributable - vis.iter().map(|&cc| col_min[cc]).sum::<f32>();
            if remainder > 0.0 {
                for &cc in vis.iter() {
                    col_min[cc] += remainder * infos[cc].pct / total_pct;
                }
            }
        }
        return;
    }
    // Step 1: percentage columns resolve against the distributable:
    // each takes `pct * distributable`, floored at its current minimum
    // (WPT table 23: the 40% column takes 80px of the 200px and the
    // fixed column the rest). This holds whether the distributing
    // cell itself has a percentage width or is auto-width — the
    // percentage resolves against the cell's size either way.
    for &cc in vis.iter() {
        let p = infos[cc].pct;
        if p > 0.0 && !infos[cc].gone {
            let share = p * distributable;
            if share > col_min[cc] {
                col_min[cc] = share;
            }
        }
    }
    let mut excess = distributable - vis.iter().map(|&cc| col_min[cc]).sum::<f32>();
    if excess <= 0.0 {
        return;
    }
    // Step 2: fill each non-percentage column from its minimum toward
    // its maximum, proportionally to the headroom. Percentage columns
    // already resolved against the distributable in step 1 and sit
    // this out (WPT `colspan-redistribution` table 23: the 40% column
    // keeps its 80px while the fixed column takes the rest). Columns
    // already at or past their maximum (including via step 1) sit out.
    // The candidate and weight buffers are cleared and refilled per
    // call, keeping their capacity across cells.
    cands.clear();
    weights.clear();
    for &cc in vis.iter() {
        let headroom = col_max[cc] - col_min[cc];
        if !infos[cc].gone && infos[cc].pct == 0.0 && headroom > 0.0 {
            cands.push(cc);
            weights.push(headroom);
        }
    }
    if !cands.is_empty() {
        let total: f32 = weights.iter().sum();
        if total > 0.0 {
            let mut filled = 0.0f32;
            for (&cc, &w) in cands.iter().zip(weights.iter()) {
                let share = (excess * w / total).min(col_max[cc] - col_min[cc]);
                col_min[cc] += share;
                filled += share;
            }
            excess -= filled;
        }
    }
    if excess <= 0.0 {
        return;
    }
    // Step 3: overflow follows the §3.9.3.2 rules, uncapped.
    mask.clear();
    mask.resize(col_min.len(), false);
    for &cc in vis.iter() {
        mask[cc] = true;
    }
    distribute_excess_among(col_min, infos, excess, Some(&mask[..]), cands, weights);
}

/// §3.9.3.2 for used width calculation: distribute `extra` (table
/// width past every column's maximum) over the visible columns.
fn distribute_excess_width(
    forest: &Forest,
    grid: &TableGrid,
    col_w: &mut [f32],
    col_max: &[f32],
    extra: f32,
    pcts: &[f32],
) {
    let infos = column_excess_infos(forest, grid, col_max, Some(pcts));
    // One-shot call (not a per-cell loop): plain local buffers, same
    // as before — nothing to reuse across calls here.
    let mut cands = Vec::new();
    let mut weights = Vec::new();
    distribute_excess_among(col_w, &infos, extra, None, &mut cands, &mut weights);
}

/// Running intrinsic-percentage state shared by the span-ordered loops
/// (CSS Tables 3 §3.8.3: measures are updated by taking into account
/// cells spanning incrementally more columns).
///
/// The state starts from the span-1 contributions — the largest
/// percentage of the column's `col` / `colgroup` boxes and its span-1
/// cells — and each spanning cell's redistribution is applied in
/// (span, DOM) order *after* that cell's own excess distribution has
/// consulted the pre-update state (a cell must not see its own
/// contribution). Values stay unclamped while a loop runs; the 100%
/// left-to-right clamp is a final step only.
///
/// [`intrinsic_column_pcts`] runs this same updater over every
/// spanning cell and then clamps, so the spanning-distribution loops
/// and the final percentage query always agree.
struct PctRunningState {
    pct: Vec<f32>,
    nonspanning_max: Vec<f32>,
}

impl PctRunningState {
    fn new(forest: &Forest, grid: &TableGrid, nonspanning_max: Vec<f32>) -> Self {
        let mut pct = vec![0.0f32; grid.ncols];
        // The intrinsic percentage is an inline-axis measure: for
        // vertical tables the spec read is the physical `height`, not
        // the physical `width` (which is the block axis there).
        let vertical = grid.vertical;
        for (cc, spec) in grid.columns.iter().enumerate() {
            if grid.collapsed_cols[cc] {
                continue;
            }
            if let Some(n) = spec.col_node
                && let Some(f) = pct_fraction(table_col_spec(inline_axis_spec(
                    &forest.nodes[n].data,
                    vertical,
                )))
            {
                pct[cc] = pct[cc].max(f);
            }
            if let Some(g) = spec.group_node
                && let Some(f) = pct_fraction(table_col_spec(inline_axis_spec(
                    &forest.nodes[g].data,
                    vertical,
                )))
            {
                // §3.8.3 names the table-column-group's percentage
                // contribution without any division: the group's
                // percentage applies at full value to each covered
                // column, exactly like the used-width spread in
                // `column_specified_widths` (a 100px `span=3` group
                // yields 100px columns, not 33px — WPT
                // `colgroup-col`; a 50% group likewise contributes
                // 50% to each column, not 50%/3).
                pct[cc] = pct[cc].max(f);
            }
        }
        for cell in grid
            .cells
            .iter()
            .filter(|c| c.colspan == 1 && c.was_single_col(forest))
        {
            if !cell_width_contributes(&grid.collapsed_cols, cell) {
                continue;
            }
            // Intrinsic percentage reads the pre-merge span (see
            // `GridCell::was_single_col`): a merged spanning cell's
            // specified percentage was for the span, not the single
            // track (a `colspan=10` cell with `width:50%` must not
            // contribute 50% to its merged column).
            if let Some(f) = pct_fraction(table_col_spec(inline_axis_spec(
                &forest.nodes[cell.node].data,
                vertical,
            ))) {
                pct[cell.col] = pct[cell.col].max(f);
            }
        }
        Self {
            pct,
            nonspanning_max,
        }
    }

    /// One spanning cell's percentage redistribution: the cell's
    /// percentage minus the spanned columns' existing percentages is
    /// redistributed over the spanned columns that still have a 0%
    /// intrinsic percentage, proportionally to their non-spanning
    /// max-content widths (evenly when those are all zero). A
    /// `colgroup` percentage contributes its full value to each
    /// covered column (see [`Self::new`]), not a divided share.
    /// Collapsed columns contribute 0 and receive nothing. Cells
    /// without a percentage width are a no-op. The percentage read is
    /// the inline-axis spec (transposed for vertical tables), matching
    /// [`Self::new`].
    fn apply_spanning_cell(&mut self, forest: &Forest, grid: &TableGrid, cell: &GridCell) {
        let Some(f) = pct_fraction(table_col_spec(inline_axis_spec(
            &forest.nodes[cell.node].data,
            grid.vertical,
        ))) else {
            return;
        };
        let end = (cell.col + cell.colspan).min(grid.ncols);
        let existing: f32 = (cell.col..end).map(|cc| self.pct[cc]).sum();
        let remaining = f - existing;
        if remaining <= 0.0 {
            return;
        }
        // Two passes over the span instead of a `targets` Vec: the
        // first sums the weights, the second applies them.
        let mut total = 0.0f32;
        let mut ntargets = 0usize;
        for cc in cell.col..end {
            if !grid.collapsed_cols[cc] && self.pct[cc] == 0.0 {
                total += self.nonspanning_max[cc];
                ntargets += 1;
            }
        }
        if ntargets == 0 {
            return;
        }
        for cc in cell.col..end {
            if !grid.collapsed_cols[cc] && self.pct[cc] == 0.0 {
                self.pct[cc] += if total > 0.0 {
                    remaining * self.nonspanning_max[cc] / total
                } else {
                    remaining / ntargets as f32
                };
            }
        }
    }

    /// The §3.8.3 left-to-right clamp (the running total never exceeds
    /// 100%), plus the pre-clamp total: the total intrinsic percentage
    /// picks the percentage floor's rule-#2 arm (exactly 100% forces an
    /// infinite maximum table width) while an over-constrained total
    /// (past 100%) keeps the finite normalized behavior — and clamping
    /// erases the distinction, so the raw total is reported separately.
    fn into_clamped(mut self) -> (Vec<f32>, f32) {
        let raw_total: f32 = self.pct.iter().sum();
        let mut run = 0.0f32;
        for p in self.pct.iter_mut() {
            *p = (*p).min((1.0 - run).max(0.0));
            run += *p;
        }
        (self.pct, raw_total)
    }
}

/// Per-column *intrinsic percentage width* (CSS Tables 3 §3.8.3): the
/// largest percentage contribution of the column's `col` / `colgroup`
/// boxes and of its span-1 cells, then updated by spanning cells in
/// increasing span order via [`PctRunningState`], finally clamped
/// left-to-right so the running total never exceeds 100% (§3.8.3 final
/// step). Collapsed columns contribute 0 and receive nothing.
/// `nonspanning_max` is each column's max-content width based on
/// span-1 cells only. The percentage reads are inline-axis specs
/// (transposed for vertical tables), so the floor below divides
/// vertical measures by vertical percentages.
///
/// Returns the clamped per-column percentages and the pre-clamp
/// total: the percentage floor needs the raw total to tell a genuine
/// 100% (rule #2: infinite maximum table width) from an
/// over-constrained total the clamp pulled down to 100% (which keeps
/// the finite normalized behavior).
fn intrinsic_column_pcts(
    forest: &Forest,
    grid: &TableGrid,
    nonspanning_max: &[f32],
) -> (Vec<f32>, f32) {
    let mut state = PctRunningState::new(forest, grid, nonspanning_max.to_vec());
    // Spanning cells, in increasing span order (§3.8.3: measures are
    // updated by taking into account cells spanning incrementally
    // more columns) — the same (span, DOM) order the
    // spanning-distribution loops use.
    let mut spanning: Vec<&GridCell> = grid
        .cells
        .iter()
        .filter(|c| c.colspan > 1 && cell_width_contributes(&grid.collapsed_cols, c))
        .collect();
    spanning.sort_by_key(|c| c.colspan);
    for cell in spanning {
        state.apply_spanning_cell(forest, grid, cell);
    }
    state.into_clamped()
}

/// Per-column *constrainedness* (CSS Tables 3 §3.8.3): a column is
/// constrained when its `col` box, its `colgroup` box, or any cell
/// spanning only it has a non-`auto`, non-percentage `width` (with the
/// csswg#94 `calc()` normalization from [`table_col_spec`]). For vertical
/// tables the read is the transposed inline spec (`height`).
fn column_constrained(forest: &Forest, grid: &TableGrid) -> Vec<bool> {
    let mut out = vec![false; grid.ncols];
    for (cc, spec) in grid.columns.iter().enumerate() {
        for node in [spec.col_node, spec.group_node].into_iter().flatten() {
            match table_col_spec(inline_axis_spec(&forest.nodes[node].data, grid.vertical)) {
                SpecLen::Auto | SpecLen::Pct(_) => {}
                _ => out[cc] = true,
            }
        }
    }
    // Constrainedness reads the pre-merge span (see
    // `GridCell::was_single_col`): a merged spanning cell's specified
    // width was for the span, so it does not constrain its track.
    for cell in grid.cells.iter().filter(|c| c.was_single_col(forest)) {
        if !cell_width_contributes(&grid.collapsed_cols, cell) {
            continue;
        }
        match table_col_spec(inline_axis_spec(
            &forest.nodes[cell.node].data,
            grid.vertical,
        )) {
            SpecLen::Auto | SpecLen::Pct(_) => {}
            _ => out[cell.col] = true,
        }
    }
    out
}

/// Grid row index of the first row in *visual* order (CSS Tables 3 §2.1:
/// the table-header-group renders before all other rows and row groups,
/// the table-footer-group after). Fixed-mode width aggregation reads
/// "the first row" (`table-layout: fixed` considers only the cells of
/// the first row track, §3.5.1), which is the visual first row, not the
/// first DOM row (WPT `fixed-layout-1.html`: a `thead` placed after the
/// `tr` in DOM order still pins the column widths).
fn first_visual_row(forest: &Forest, grid: &TableGrid) -> usize {
    let mut best: Option<(u8, usize)> = None;
    for (r, group) in grid.row_group.iter().enumerate() {
        let rank = match group.and_then(|n| row_group_section_kind(forest, n)) {
            Some(SectionKind::Header) => 0,
            Some(SectionKind::Footer) => 2,
            _ => 1,
        };
        let key = (rank, r);
        if best.is_none_or(|b| key < b) {
            best = Some(key);
        }
    }
    best.map_or(0, |(_, r)| r)
}

/// Automatic-layout column minima per CSS Tables 3 §3.8.3, shared by
/// `layout_table` and `min_content_width`'s table arm. One entry per
/// column: the maximum over single-column cells of (min-content width +
/// padding + border, floored by the `min-width` floor via
/// [`cell_min_width_bb`]) — a cell's specified `width` never floors
/// the minimum (§3.8.3 outer min-content width; it floors the
/// *maximum* of a constrained column instead); then the specified
/// `<colgroup>` widths as further minima (a `col` width never floors
/// the minimum — WPT `column-widths` tables 18/24); finally spanning
/// cells distribute any excess over the spanned
/// minima across their columns.
///
/// `col_spec` is the output of [`column_specified_widths`]: border-box
/// widths plus the per-column group-spread-only flag. Percentages (cell
/// padding, specified widths, `min-width`) resolve against `table_w` —
/// the table's content width in `layout_table`, zero in intrinsic
/// sizing where no definite base exists yet. `spacing_h` is the
/// inter-column spacing (zero when borders collapse); a spanning cell's
/// used width covers it, so the excess spread over the columns is
/// reduced by it. Cell padding/border resolve inside via
/// `resolve_metrics`, so callers need not pre-resolve them.
/// Fixed-layout column pins (CSS 2.1 §17.5.2.1): `<col>`/`<colgroup>`
/// specified widths pin their columns first, then first-row cells with a
/// non-`auto` width pin the rest (a spanning cell's width divides over its
/// columns). Collapsed columns pin at zero width. Returns the per-column
/// border-box pin widths alongside the pin flags; unpinned columns are
/// zero. Cell metrics must be resolved before calling: the pin reads the
/// resolved border-box width, exactly as `table_pref_width`'s and
/// `layout_table`'s fixed branches do.
fn fixed_column_pins(
    forest: &Forest,
    grid: &TableGrid,
    col_spec: &(Vec<Option<f32>>, Vec<bool>),
) -> (Vec<f32>, Vec<bool>) {
    let mut pins = vec![0.0f32; grid.ncols];
    let mut pinned = vec![false; grid.ncols];
    // Collapsed columns take no width in any pass (CSS 2.1 §17.5).
    for (cc, &gone) in grid.collapsed_cols.iter().enumerate() {
        if gone {
            pinned[cc] = true;
        }
    }
    // Column boxes pin their columns first, ahead of first-row cell
    // widths. Collapsed columns stay at zero.
    for (cc, w) in col_spec.0.iter().enumerate() {
        if let Some(bb) = w
            && !grid.collapsed_cols[cc]
        {
            pins[cc] = *bb;
            pinned[cc] = true;
        }
    }
    // Fixed mode pins from the first *visual* row (see
    // `first_visual_row`), not the first DOM row.
    let first_row = first_visual_row(forest, grid);
    for cell in grid.cells.iter().filter(|c| c.row == first_row) {
        // Only cells spanning solely collapsed columns pin nothing:
        // collapsed-row cells never lay out, but they still size
        // columns (CSS 2.1 §17.5.5), so they pin here too.
        if !cell_width_contributes(&grid.collapsed_cols, cell) {
            continue;
        }
        let n = &forest.nodes[cell.node];
        if matches!(inline_axis_spec(&n.data, grid.vertical), SpecLen::Auto) {
            continue;
        }
        let m = &n.metrics;
        // The pin is the border-box inline size. For vertical tables the
        // metrics hold the physical (block-axis) width, so resolve the
        // `height` spec directly against the zero base (percentages
        // resolve to 0 here, like the horizontal path; the fixpoint
        // re-pin below resolves them against the table width).
        let bb = if grid.vertical {
            let d = &n.data;
            let v = inline_axis_spec(d, true)
                .resolve(
                    d.font_size,
                    forest.root_font_size,
                    forest.vp_w,
                    forest.vp_h,
                    0.0,
                )
                .unwrap_or(0.0)
                .max(0.0);
            let bp = m.bt + m.pt + m.pb + m.bb;
            match d.box_sizing {
                BoxSizing::BorderBox => v,
                BoxSizing::ContentBox => v + bp,
            }
            .max(0.0)
        } else {
            (m.content_w + m.bl + m.pl + m.pr + m.br).max(0.0)
        };
        let per = bb / cell.colspan.max(1) as f32;
        for cc in cell.col..(cell.col + cell.colspan).min(grid.ncols) {
            if !pinned[cc] {
                pins[cc] = per;
                pinned[cc] = true;
            }
        }
    }
    (pins, pinned)
}

/// Automatic-layout column maxima per CSS Tables 3 §3.8.3: the
/// maximum over single-column cells of the outer max-content width
/// (with the specified-width cap from [`cell_outer_max_content`]),
/// floored by specified `<col>` widths (a `<colgroup>` spread floors
/// the minimum only), then raised by spanning cells' preferred-width
/// excess distributed per §3.9.3.2.
///
/// Spanning cells distribute cumulatively in increasing span order
/// (DOM order within a level), each over the *current* maxima — see
/// [`distribute_spanning_max`]. Returns the maxima and the span-1
/// snapshot (`col_max_nonspanning`); the latter feeds
/// [`intrinsic_column_pcts`], whose redistribution is defined against
/// the measures based on cells of span up to 1.
/// Finite max-content width of an *automatic-layout* table: the sum of
/// its columns' §3.8.3 maxima ([`auto_column_maxima`]) plus inter-column
/// spacing — the table's GRIDMAX, without the percentage floor that
/// [`table_pref_width`] applies. A 100% intrinsic column percentage
/// drives that floor to infinity, which is the table's *preferred*
/// width for sizing the table itself against available space (block
/// and float paths clamp it to the available width), but not its
/// max-content contribution to an outer context: WPT
/// `css/css-tables/tentative/table-minmax.html` expects a table with a
/// `width:100%` column beside a 100px-max column to contribute
/// 30 + 100 + 30 = 160px, as a flex item's base size (flex `1 1 auto`
/// grows 160 to 355; flex `0 0 auto` stays 160) and as an outer table
/// column's maximum (the outer table then sizes to 290 and the inner
/// table to its 160 column). Returns `None` for fixed-layout tables,
/// whose percentage-width max-content stays infinite per Blink (WPT
/// `table-fixed-minmax.html`; the flex saturation test for a fixed
/// `width:100%` table relies on it). The value excludes the table's own
/// border/padding, like [`table_pref_width`]: consumers that combine it
/// with other boxes add the table's extras themselves (see
/// [`generic_preferred_width`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn table_auto_grid_max_width(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
) -> Option<f32> {
    // Vertical guard (w75f3, item 3): the column-maximum sum is an
    // INLINE-axis (physical-vertical) extent for a vertical table, so
    // it must never serve as a flex main size (physical-horizontal).
    // All four inner call sites already gate on `!grid.vertical`;
    // only the flex site in `layout.rs` called this unguarded, so a
    // vertical flex-item table with an infinite base got a wrong-axis
    // base. Returning `None` keeps the infinite base on the saturated
    // path, like a fixed-layout table.
    if let Some(g) = cache.grid_max_get(idx) {
        return Some(g);
    }
    let mut pool = TableGridPool::new();
    let mut grid = pool.take_grid();
    collect_table_grid_into(forest, idx, &mut pool, &mut grid);
    if grid.fixed || grid.vertical {
        pool.put_grid(grid);
        return None;
    }
    let (_, spacing_h, _) = table_border_model(forest, idx);
    for cell in &grid.cells {
        if cell_width_contributes(&grid.collapsed_cols, cell) {
            resolve_metrics(forest, cell.node, 0.0, 0.0, false, None, false);
        }
    }
    let col_spec = column_specified_widths(forest, &grid, 0.0, 0.0);
    let (col_max, _) = auto_column_maxima(
        forest, &grid, &col_spec, spacing_h, fonts, cache, mcw, bufs, inner,
    );
    let nvis = grid.collapsed_cols.iter().filter(|&&c| !c).count();
    let spacing_total = if nvis == 0 {
        0.0
    } else {
        spacing_h * (nvis as f32 + 1.0)
    };
    // Memoize (w75f3, item 5): without this, every infinite cell in
    // an outer pass re-collects the same inner grid, and nested %
    // tables multiply the cost per level. The value is computed from
    // zero-base column specs, so it is independent of the caller's
    // available width; the generation stamp in `PrefWidthCache`
    // invalidates it whenever the cache generation bumps (see
    // `grid_max_get` / `grid_max_set`). Bench note: the nested-%
    // regression test below exercises three levels through this path;
    // no separate bench harness exists for intrinsic measurement in
    // this crate, so the cache is pinned by correctness of the cached
    // value across repeated outer passes, not by a timing assertion.
    let result = (col_max.iter().sum::<f32>() + spacing_total).max(0.0);
    cache.grid_max_set(idx, result);
    pool.put_grid(grid);
    Some(result)
}

/// Finite max-content fallback for a cell whose measured max-content is
/// infinite: the run accumulation [`generic_preferred_width`] uses,
/// with automatic-layout table children measured at their
/// column-maximum sum ([`table_auto_grid_max_width`]) instead of their
/// (possibly infinite) preferred width, and other children at their
/// already-measured `pref_w` plus extras. Consecutive inline-level
/// children sum into one run (a text run plus an inline element plus
/// a direct auto % table contribute their sum, not their max); a
/// non-table block-level child ends the run and contributes as its own
/// maximum, and a `<br>` ends the run like a forced line break.
/// Table children join the run rather than breaking it: this is the
/// finite stand-in for the cell's max-content, where the table's grid
/// sum is the table's contribution to the line the cell's inline
/// content sits on. Returns `None` when no substitution applies (no
/// direct table child, or a fixed-layout table child keeps infinity),
/// leaving the caller on the saturated path. Only direct children
/// substitute: a table nested deeper (block > table) still propagates
/// infinity, as before.
#[allow(clippy::too_many_arguments)]
fn finite_cell_max_content(
    forest: &mut Forest,
    cell: usize,
    fonts: &dyn FontMetrics,
    cache: &mut PrefWidthCache,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
    inner: &mut WalkBufs,
) -> Option<f32> {
    let mut best: Option<f32> = None;
    let mut has_table = false;
    let mut run = 0.0f32;
    let nkids = forest.nodes[cell].children.len();
    for i in 0..nkids {
        let ch = forest.nodes[cell].children[i];
        if layout_position::is_out_of_flow(forest, ch) {
            continue;
        }
        if forest.nodes[ch].is_br {
            best = Some(best.unwrap_or(0.0).max(run));
            run = 0.0;
            continue;
        }
        let n = &forest.nodes[ch];
        if n.box_type == BoxType::Table || n.data.display == Display::InlineTable {
            has_table = true;
            let g = table_auto_grid_max_width(forest, ch, fonts, cache, mcw, bufs, inner)?;
            run += g + intrinsic_h_extras(forest, ch);
            best = Some(best.unwrap_or(0.0).max(run));
        } else if n.box_type == BoxType::AnonymousBlock {
            // The anonymous block wrapping a cell's inline run IS that
            // run (the box builder wraps text + inline elements into
            // one anonymous block when a block-level table sibling is
            // present), so it joins the run instead of breaking it —
            // otherwise a text run + inline element + direct table
            // could never sum, and this fallback would stay a max.
            run += n.pref_w + intrinsic_h_extras(forest, ch);
            best = Some(best.unwrap_or(0.0).max(run));
        } else if crate::boxes::is_block_level(n.box_type) {
            best = Some(
                best.unwrap_or(0.0)
                    .max(n.pref_w + intrinsic_h_extras(forest, ch)),
            );
            run = 0.0;
        } else {
            run += n.pref_w + intrinsic_h_extras(forest, ch);
            best = Some(best.unwrap_or(0.0).max(run));
        }
    }
    if has_table { best } else { None }
}

// The intrinsic-measure scratch travels as explicit params so
// `layout_table` reuses its own buffers instead of allocating.
#[allow(clippy::too_many_arguments)]
pub(crate) fn auto_column_maxima(
    forest: &mut Forest,
    grid: &TableGrid,
    col_spec: &(Vec<Option<f32>>, Vec<bool>),
    spacing_h: f32,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    mcw_scratch: &mut Vec<f32>,
    int_bufs: &mut WalkBufs,
    int_inner: &mut WalkBufs,
) -> (Vec<f32>, Vec<f32>) {
    let mut col_max = vec![0.0f32; grid.ncols];
    // CSS Tables 3 §3.8.3 constrainedness: an auto cell in a
    // constrained column contributes its min-content width (not its
    // max-content width) to the column maximum — see
    // `cell_outer_max_content`.
    let constrained = column_constrained(forest, grid);
    for cell in grid.cells.iter().filter(|c| c.colspan == 1) {
        // Collapsed rows never lay out and collapsed columns take no
        // width: their cells contribute nothing to the maxima either
        // (CSS 2.1 §17.5).
        if !cell_width_contributes(&grid.collapsed_cols, cell) {
            continue;
        }
        // A vertical table's column (inline) axis is the physical
        // vertical axis: measure the cell's vertical content size
        // instead of its horizontal preferred width.
        let pcw = if grid.vertical {
            vertical_cell_max_content(
                forest,
                cell.node,
                fonts,
                pw_cache,
                mcw_scratch,
                int_bufs,
                int_inner,
            )
        } else {
            preferred_width_mcw(
                forest,
                cell.node,
                fonts,
                pw_cache,
                mcw_scratch,
                int_bufs,
                int_inner,
            )
        };
        // An infinite cell max-content (a contained automatic-layout
        // table at the 100% percentage floor) must not poison this
        // column's maximum: the table contributes its finite
        // column-maximum sum instead (see `finite_cell_max_content`).
        let pcw = if !grid.vertical && pcw.is_infinite() {
            finite_cell_max_content(
                forest,
                cell.node,
                fonts,
                pw_cache,
                mcw_scratch,
                int_bufs,
                int_inner,
            )
            .unwrap_or(pcw)
        } else {
            pcw
        };
        // csswg#94: a percentage-bearing `calc()` with a nonzero
        // length part counts as `auto` here (see `table_col_spec`).
        let (spec, specified, pad) = {
            let n = &forest.nodes[cell.node];
            // A vertical table's inline axis is the physical vertical
            // axis: the cell's inline spec is its `height`, and the
            // specified content size resolves from the height
            // properties (see `cell_specified_inline_height`).
            let vertical = grid.vertical;
            let spec = table_col_spec(inline_axis_spec(&n.data, vertical));
            let specified = match spec {
                SpecLen::Auto => 0.0,
                _ if vertical => cell_specified_inline_height(forest, cell.node),
                _ => n.metrics.content_w,
            };
            (spec, specified, cell.intrinsic_pad)
        };
        // §3.8.3 outer max-content: a definite width caps the
        // contribution at `max(min-content, width)` instead of
        // flooring the max-content measure (see
        // `cell_outer_max_content`); the min-content measure is
        // only needed on that arm, and on the auto arm in a
        // constrained column, where the cell contributes its
        // min-content instead of its max-content.
        let auto_in_constrained = constrained[cell.col] && matches!(spec, SpecLen::Auto);
        let mcw = if matches!(spec, SpecLen::Px(_)) || auto_in_constrained {
            // Transposed with the preferred measure above: the
            // definite-spec arm needs the vertical min-content.
            if grid.vertical {
                vertical_cell_min_content(forest, cell.node, fonts, mcw_scratch, int_bufs)
            } else {
                min_content_width_pooled(forest, cell.node, fonts, mcw_scratch, int_bufs)
            }
        } else {
            0.0
        };
        let need_max = (cell_outer_max_content(
            spec,
            pcw.max(specified),
            mcw,
            specified,
            constrained[cell.col],
        ) + pad)
            .max(cell_min_width_bb(forest, cell.node, 0.0, grid.vertical));
        if need_max > col_max[cell.col] {
            col_max[cell.col] = need_max;
        }
    }
    // Specified `<col>` widths floor the column maxima only (never the
    // minimum — see `auto_column_minima`); a `<colgroup>` spread floors
    // the minimum only (step 4), leaving the maximum to content.
    for (cc, w) in col_spec.0.iter().enumerate() {
        if let Some(bb) = w
            && !col_spec.1[cc]
            && !grid.collapsed_cols[cc]
            && *bb > col_max[cc]
        {
            col_max[cc] = *bb;
        }
    }
    // The span-1 maxima snapshot: §3.8.3 spanning updates (and the
    // intrinsic percentage redistribution below) are defined
    // against the measures based on cells of span up to 1 before
    // any spanning cell is taken into account.
    let col_max_nonspanning = col_max.clone();
    // Multi-column cells distribute their preferred-width excess
    // over the spanned columns per §3.9.3.2 (for intrinsic width
    // calculation), in increasing span order, cumulatively — each
    // cell distributes over the current maxima (see
    // `distribute_spanning_max`). Collapsed columns take no part:
    // the excess spreads over the visible spanned columns only, and
    // the inter-column spacing a collapsed column would have drawn
    // is excluded from the gap. Under separate borders the spanning
    // cell's used width also covers the inter-column spacing between
    // the spanned columns, so the excess distributed over the columns
    // is reduced by it (`spacing_h` is 0 when borders collapse).
    let mut spanning: Vec<&GridCell> = grid
        .cells
        .iter()
        .filter(|c| c.colspan > 1 && cell_width_contributes(&grid.collapsed_cols, c))
        .collect();
    if !spanning.is_empty() {
        spanning.sort_by_key(|c| c.colspan);
        // Scratch reused by every `distribute_spanning_max` call
        // below: no per-cell allocation after the first, including
        // the distribution list (rebuilt in scratch from the span
        // range).
        let mut cand_scratch = SpanningCandidateScratch::default();
        // Running intrinsic percentages (§3.8.3): each cell's excess
        // distributes against the percentages as of before its own
        // redistribution, in (span, DOM) order — the same state
        // [`intrinsic_column_pcts`] computes.
        let mut pct_state = PctRunningState::new(forest, grid, col_max_nonspanning.clone());
        for cell in spanning {
            let end = (cell.col + cell.colspan).min(grid.ncols);
            // Allocation-free span probe: the visible-column count
            // feeds the gap, and an all-collapsed span skips the
            // content measurement below (as the old per-cell `vis`
            // did) — but the percentage redistribution still runs
            // for it, mirroring [`intrinsic_column_pcts`].
            let vis_len = (cell.col..end)
                .filter(|&cc| !grid.collapsed_cols[cc])
                .count();
            if vis_len == 0 {
                pct_state.apply_spanning_cell(forest, grid, cell);
                continue;
            }
            // Vertical table: measure the cell's vertical content size
            // (see the single-column loop above).
            let pcw = if grid.vertical {
                vertical_cell_max_content(
                    forest,
                    cell.node,
                    fonts,
                    pw_cache,
                    mcw_scratch,
                    int_bufs,
                    int_inner,
                )
            } else {
                preferred_width_mcw(
                    forest,
                    cell.node,
                    fonts,
                    pw_cache,
                    mcw_scratch,
                    int_bufs,
                    int_inner,
                )
            };
            // Same infinite-cell substitution as the span-1 loop above.
            let pcw = if !grid.vertical && pcw.is_infinite() {
                finite_cell_max_content(
                    forest,
                    cell.node,
                    fonts,
                    pw_cache,
                    mcw_scratch,
                    int_bufs,
                    int_inner,
                )
                .unwrap_or(pcw)
            } else {
                pcw
            };
            // csswg#94: a percentage-bearing `calc()` with a nonzero
            // length part counts as `auto` here (see `table_col_spec`).
            let (spec, specified, pad, cell_pct) = {
                let n = &forest.nodes[cell.node];
                // Transposed inline spec for vertical tables (see the
                // single-column loop above).
                let vertical = grid.vertical;
                let spec = table_col_spec(inline_axis_spec(&n.data, vertical));
                let specified = match spec {
                    SpecLen::Auto => 0.0,
                    _ if vertical => cell_specified_inline_height(forest, cell.node),
                    _ => n.metrics.content_w,
                };
                let cell_pct = match spec {
                    SpecLen::Pct(p) => p / 100.0,
                    _ => 0.0,
                };
                (spec, specified, cell.intrinsic_pad, cell_pct)
            };
            // §3.8.3 outer max-content, as in the single-column loop —
            // but keyed on the cell's own spec only: constrainedness is
            // per-column and a spanning cell spans several, so the
            // constrained-column min-content substitution does not apply
            // here.
            let mcw = if matches!(spec, SpecLen::Px(_)) {
                // Transposed with the preferred measure above.
                if grid.vertical {
                    vertical_cell_min_content(forest, cell.node, fonts, mcw_scratch, int_bufs)
                } else {
                    min_content_width_pooled(forest, cell.node, fonts, mcw_scratch, int_bufs)
                }
            } else {
                0.0
            };
            let gap = spacing_h * vis_len.saturating_sub(1) as f32;
            let need_max =
                cell_outer_max_content(spec, pcw.max(specified), mcw, specified, false) + pad;
            let unconstrained_cell = matches!(spec, SpecLen::Auto);
            distribute_spanning_max(
                forest,
                grid,
                &mut col_max,
                cell.col,
                end,
                need_max,
                gap,
                unconstrained_cell,
                cell_pct,
                None,
                Some(&pct_state.pct),
                &mut cand_scratch,
            );
            // The cell's own redistribution applies after its excess
            // distributed: the distribution above must not see it.
            pct_state.apply_spanning_cell(forest, grid, cell);
        }
    }
    (col_max, col_max_nonspanning)
}

#[allow(clippy::too_many_arguments)]
/// WHATWG Quirks Mode Standard §3.9 (w77fix finding 8): in a
/// full-quirks document, a `td`/`th` with a `nowrap` attribute whose
/// computed `width` is a non-zero `<length>` acts as if its outer
/// min-content width in the automatic table layout algorithm were
/// `max(computed width, outer min-content width)`. Returns `None`
/// when the quirk does not apply (not full quirks, no `nowrap`
/// attribute, no nonzero length width, fixed layout,
/// vertical table — the quirk is defined for the automatic layout
/// algorithm's horizontal inline axis).
///
/// w78fix: the gate is quirks-only again (the wave-78 admission of
/// limited-quirks is REMOVED). Limited-quirks nowrap cells no
/// longer need this arm: the cascade now implements the HTML
/// rendering presentational hint (`td[nowrap], th[nowrap]
/// { white-space: nowrap }`, see `table_presentational_hints` in
/// `bowser_css`), gated to quirks / limited-quirks documents, so a
/// limited-quirks nowrap cell measures its unwrapped line through
/// its computed `white-space` — including auto, percentage, and
/// zero widths, which this floor arm (nonzero length width only)
/// could never cover — and WPT `table-limited-quirks.html` table 3
/// passes via the hint path, not via a floor. What remains here is
/// only the genuine §3.9 floor: in full quirks, a nowrap cell
/// whose specified width exceeds its nowrap measure floors at the
/// width (the 200px short-text case, which the hint alone would
/// leave at its 64px content measure). Standards mode is
/// deliberately untouched by this floor, and the hint itself is
/// gated off there too, so the w77 standards control still wraps.
///
/// A bare floor on the §3.8.3 minimum is not enough: the WPT
/// `table-quirks` case measures the full unwrapped line (580 =
/// full line + padding, not max(50, 160)), so the cell's
/// min-content for this computation is its nowrap measure — its
/// max-content width, exactly what the presentational hint
/// (`td[nowrap], th[nowrap] { white-space: nowrap }` from the
/// HTML rendering UA stylesheet) produces. The floor
/// is on the *outer* width: a border-box length floors it
/// directly, a content-box length floors it after padding/border
/// (`pad`).
///
/// w79fix: the arm is NOT gated on the cell's resolved `white-space`.
/// Quirks §3.9 conditions only on the cell being `td`/`th`, the
/// `nowrap` attribute, and a nonzero `<length>` computed width — it
/// says nothing about `white-space`. The w78 white-space gate was
/// therefore wrong, and w79 exposed it: the w79 quirks reset correctly
/// makes a full-quirks `td`/`th[nowrap]` with a nonzero-length width
/// compute `white-space: normal` (so it wraps), which switched the
/// floor off for exactly the cells §3.9 targets — historical Gecko
/// kept both the wrap reset and the width floor.
/// The floor returns only the width's outer measure: both call sites
/// already `max` it against a `need_min` carrying the cell's actual
/// (wrapping-aware) min-content, so the result is exactly §3.9's
/// `max(computed width, outer min-content width)`. Returning the
/// old recomputed max-content measure here as well would over-floor
/// a wrapping cell to its unwrapped line (364 for the width-attr
/// fixture, where the spec/Gecko result is `max(240, 112) = 240`
/// for width 200 and 112 for width 50) — for a nowrap cell the two
/// measures coincide, so the classic fixtures are unchanged.
/// Percentages are not lengths: no quirk.
fn quirks_nowrap_min_floor(forest: &Forest, node: usize, pad: f32) -> Option<f32> {
    if forest.quirks_mode != QuirksMode::Quirks || !forest.nodes[node].nowrap_attr {
        return None;
    }
    let spec = table_col_spec(forest.nodes[node].data.width);
    let SpecLen::Px(p) = spec else {
        return None;
    };
    if p <= 0.0 {
        return None;
    }
    Some(match forest.nodes[node].data.box_sizing {
        BoxSizing::BorderBox => p,
        BoxSizing::ContentBox => p + pad,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn auto_column_minima(
    forest: &mut Forest,
    grid: &TableGrid,
    col_spec: &(Vec<Option<f32>>, Vec<bool>),
    spacing_h: f32,
    table_w: f32,
    fonts: &dyn FontMetrics,
    col_max: &[f32],
    col_max_nonspanning: &[f32],
) -> Vec<f32> {
    // Fixed mode reads "the first row" as the first *visual* row (see
    // `first_visual_row`), not the first DOM row.
    let first_row = if grid.fixed {
        first_visual_row(forest, grid)
    } else {
        0
    };
    // Resolve each cell's padding/border now; percentage padding
    // resolves against the table-width base (zero in intrinsic sizing).
    // Skipped for cells that feed no width computation (spanning solely
    // collapsed columns): `resolve_metrics` stamps `rect.w` as a side
    // effect, which a skipped cell must not keep. Collapsed-row cells
    // never lay out, but their widths still size columns (CSS 2.1
    // §17.5.5), so they are resolved here.
    for cell in &grid.cells {
        if cell_width_contributes(&grid.collapsed_cols, cell) {
            resolve_metrics(forest, cell.node, 0.0, table_w, false, None, false);
        }
    }
    // `table-layout: fixed` with a non-definite specified table width:
    // the column minima are the fixed-layout pins (CSS 2.1 §17.5.2.1),
    // never content measurement. This is the fixed-mode used-width rule
    // for `width: min-content | fit-content` and the `calc-size()`
    // keyword-basis widths (WPT `css/css-tables/fixed-layout-2.html`):
    // without it the minima measure content and the used width comes
    // out as the content width.
    if grid.fixed_indefinite {
        return fixed_column_pins(forest, grid, col_spec).0;
    }
    // One scratch triple serves every cell's intrinsic measurement
    // instead of allocating a node-sized `mcw` plus fresh walk buffers
    // per cell.
    let mut mcw_scratch = Vec::new();
    let mut int_bufs = WalkBufs::new();
    let mut col_min = vec![0.0f32; grid.ncols];
    for cell in grid.cells.iter().filter(|c| c.colspan == 1) {
        // A collapsed column takes no space and contributes nothing to
        // the minima — but a collapsed row's cells still do: they never
        // lay out, yet their widths size columns (CSS 2.1 §17.5.5).
        if !cell_width_contributes(&grid.collapsed_cols, cell) {
            continue;
        }
        // A vertical table's column (inline) axis is the physical
        // vertical axis: measure the cell's vertical min-content
        // instead of its horizontal min-content width.
        let mcw = if grid.vertical {
            vertical_cell_min_content(forest, cell.node, fonts, &mut mcw_scratch, &mut int_bufs)
        } else {
            min_content_width_pooled(forest, cell.node, fonts, &mut mcw_scratch, &mut int_bufs)
        };
        let n = &forest.nodes[cell.node];
        // CSS Tables 3 §3.8.3: the outer min-content width of a cell is
        // `max(min-width, min-content width)` — a specified `width`
        // never floors the minimum (it floors the *maximum* of a
        // constrained column instead; see the maximum passes). The one
        // exception is `table-layout: fixed`, where a first-row cell's
        // specified length is its min-content measure (§3.8.3 fixed
        // note). csswg#94: a percentage-bearing `calc()` with a nonzero
        // length part counts as `auto` here (see `table_col_spec`).
        // A vertical table's inline axis is the physical vertical axis:
        // the cell's inline spec is its `height` (see
        // `cell_specified_inline_height`).
        let vertical = grid.vertical;
        let specified = match table_col_spec(inline_axis_spec(&n.data, vertical)) {
            SpecLen::Auto => 0.0,
            // CSS Tables 3 §3.8.3: a specified width never floors the
            // minimum outside the fixed layout's first row — the
            // vertical arm only changes *which* spec resolves.
            _ if grid.fixed && cell.row == first_row => {
                if vertical {
                    cell_specified_inline_height(forest, cell.node)
                } else {
                    n.metrics.content_w
                }
            }
            _ => 0.0,
        };
        let pad = cell.intrinsic_pad;
        let min_bb = cell_min_width_bb(forest, cell.node, table_w, grid.vertical);
        let mut need_min = (mcw.max(specified) + pad).max(min_bb);
        // WHATWG Quirks §3.9 (see `quirks_nowrap_min_floor`): a
        // nowrap cell's outer min-content is floored by its
        // specified width's outer measure.
        if !grid.fixed
            && !grid.vertical
            && let Some(floor) = quirks_nowrap_min_floor(forest, cell.node, pad)
        {
            need_min = need_min.max(floor);
        }
        if need_min > col_min[cell.col] {
            col_min[cell.col] = need_min;
        }
    }
    // A `<col>` width never floors the column minimum in the automatic
    // layout: per WPT `column-widths` (column.min = td.min) a fixed
    // `<col>` floors the column *maximum* only, so a `width:1px` table
    // with a `width:100px` `<col>` and 50px of cell min-content is 66px
    // wide, not 116px (tables 18/24). Only a `<colgroup>` spread (no
    // `col` width of its own) still floors the minimum. Collapsed
    // columns keep a zero minimum regardless of specified widths.
    for (cc, w) in col_spec.0.iter().enumerate() {
        if let Some(bb) = w
            && col_spec.1[cc]
            && !grid.collapsed_cols[cc]
            && *bb > col_min[cc]
        {
            col_min[cc] = *bb;
        }
    }
    // Multi-column cells distribute their min-content excess over the
    // spanned columns per §3.9.3.2 (for intrinsic width calculation),
    // in increasing span order, cumulatively — each cell distributes
    // over the current minima (see `distribute_spanning_min`), against
    // the final maxima (`col_max`, from [`auto_column_maxima`]).
    // Collapsed columns take no part: the excess spreads over the
    // visible spanned columns only, and the inter-column spacing a
    // collapsed column would have drawn is excluded from the gap.
    // Under separate borders the spanning cell's used width also
    // covers the inter-column spacing between the spanned columns,
    // so the excess distributed over the columns is reduced by it
    // (`spacing_h` is 0 when borders collapse).
    let mut spanning: Vec<&GridCell> = grid
        .cells
        .iter()
        .filter(|c| c.colspan > 1 && cell_width_contributes(&grid.collapsed_cols, c))
        .collect();
    if !spanning.is_empty() {
        spanning.sort_by_key(|c| c.colspan);
        // Scratch reused by every `distribute_spanning_min` call
        // below: no per-cell allocation after the first, including the
        // visible-column list (rebuilt in scratch from the span range).
        let mut cand_scratch = SpanningCandidateScratch::default();
        // Running intrinsic percentages (§3.8.3), weighted by the
        // span-1 maxima like [`intrinsic_column_pcts`]: each cell's
        // excess distributes against the percentages as of before its
        // own redistribution, in (span, DOM) order.
        let mut pct_state = PctRunningState::new(forest, grid, col_max_nonspanning.to_vec());
        for cell in spanning {
            let end = (cell.col + cell.colspan).min(grid.ncols);
            // Allocation-free probe preserving the old skip of the
            // content measurement for cells spanning solely collapsed
            // columns (already excluded above by
            // `cell_width_contributes`, so this never fires in
            // practice) — but the percentage redistribution still
            // runs for it, mirroring [`intrinsic_column_pcts`].
            if (cell.col..end).all(|cc| grid.collapsed_cols[cc]) {
                pct_state.apply_spanning_cell(forest, grid, cell);
                continue;
            }
            // Vertical table: the cell's vertical min-content (see the
            // single-column loop above).
            let mcw = if grid.vertical {
                vertical_cell_min_content(forest, cell.node, fonts, &mut mcw_scratch, &mut int_bufs)
            } else {
                min_content_width_pooled(forest, cell.node, fonts, &mut mcw_scratch, &mut int_bufs)
            };
            // Same §3.8.3 outer-min-content rule as the
            // single-column loop above: a specified `width` does
            // not floor the minimum outside the fixed layout's
            // first row. csswg#94: a percentage-bearing `calc()`
            // with a nonzero length part counts as `auto` here
            // (see `table_col_spec`).
            let (specified, pad, cell_pct) = {
                let n = &forest.nodes[cell.node];
                // Transposed inline spec for vertical tables (see the
                // single-column loop above).
                let vertical = grid.vertical;
                let spec = table_col_spec(inline_axis_spec(&n.data, vertical));
                // Same §3.8.3 fixed-first-row-only rule as the
                // single-column loop above.
                let specified = match spec {
                    SpecLen::Auto => 0.0,
                    _ if grid.fixed && cell.row == first_row => {
                        if vertical {
                            cell_specified_inline_height(forest, cell.node)
                        } else {
                            n.metrics.content_w
                        }
                    }
                    _ => 0.0,
                };
                let cell_pct = match spec {
                    SpecLen::Pct(p) => p / 100.0,
                    _ => 0.0,
                };
                (specified, cell.intrinsic_pad, cell_pct)
            };
            let min_bb = cell_min_width_bb(forest, cell.node, table_w, grid.vertical);
            let mut need_min = (mcw.max(specified) + pad).max(min_bb);
            // WHATWG Quirks §3.9 (see `quirks_nowrap_min_floor`):
            // same floor as the single-column loop, distributed
            // over the spanned columns below.
            if !grid.fixed
                && !grid.vertical
                && let Some(floor) = quirks_nowrap_min_floor(forest, cell.node, pad)
            {
                need_min = need_min.max(floor);
            }
            distribute_spanning_min(
                forest,
                grid,
                &mut col_min,
                col_max,
                cell.col,
                end,
                need_min,
                spacing_h,
                cell_pct,
                Some(&pct_state.pct),
                &mut cand_scratch,
            );
            // The cell's own redistribution applies after its excess
            // distributed: the distribution above must not see it.
            pct_state.apply_spanning_cell(forest, grid, cell);
        }
    }
    col_min
}

/// Horizontal padding/border for a cell's column intrinsic widths. Under
/// `border-collapse: collapse` the cell's own borders are drawn centered
/// on the shared grid lines (see `resolve_collapsed_borders`), not inside
/// the cell's border box, so they must not inflate the column's
/// min/max-content widths — only padding counts. A losing border (e.g.
/// `border-right: 5px ridge` on a cell whose neighbor wins the shared
/// edge) would otherwise widen the column versus the reference (WPT
/// `border-conflict-resolution`).
fn cell_h_pad(m: &crate::boxes::BoxMetrics, collapse: bool, vertical: bool) -> f32 {
    if vertical {
        // Vertical table: column (inline) axis is vertical.
        m.pt + m.pb + if collapse { 0.0 } else { m.bt + m.bb }
    } else {
        m.pl + m.pr + if collapse { 0.0 } else { m.bl + m.br }
    }
}

/// A cell's inline-axis padding/border for intrinsic column sizing
/// ([`auto_column_maxima`] / [`auto_column_minima`]): `%` padding/border
/// contributes 0 (WPT `td-box-sizing-003`: "intrinsic size of %
/// padding is 0"); px values resolve base-independently. Unlike
/// [`cell_h_pad`], which reads the metrics' possibly non-zero-base `%`
/// resolve (from the final-layout cell pass or a previous layout), this
/// resolves the specified values against a zero base, so `%` widths and
/// `%` min-widths (which legitimately resolve against the table width)
/// are unaffected.
fn cell_intrinsic_pad(forest: &Forest, node: usize, collapse: bool, vertical: bool) -> f32 {
    let n = &forest.nodes[node];
    let d = &n.data;
    let (fs, root_fs, vp_w, vp_h) = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
    let len0 = |s: SpecLen| {
        s.resolve(fs, root_fs, vp_w, vp_h, 0.0)
            .unwrap_or(0.0)
            .max(0.0)
    };
    // `border-style: none | hidden` zeroes the used width (CSS 2.1
    // §8.5.3). Physical indices: top=0, right=1, bottom=2, left=3.
    let bw = |i: usize| {
        if d.border_style[i].zeroes_width() {
            0.0
        } else {
            len0(d.border[i])
        }
    };
    if vertical {
        // Vertical table: column (inline) axis is vertical.
        len0(d.padding[0]) + len0(d.padding[2]) + if collapse { 0.0 } else { bw(0) + bw(2) }
    } else {
        len0(d.padding[3]) + len0(d.padding[1]) + if collapse { 0.0 } else { bw(3) + bw(1) }
    }
}

/// A cell's `min-width` as a border-box floor for its column(s): the
/// property constrains the content box under `content-box` sizing and
/// the border box under `border-box` sizing, so the floor adds the
/// cell's padding/border only in the `content-box` case. Percentages
/// resolve against the given table width (content plus spacing in the
/// second preferred-width pass, per CSS 2.1 §17.5.2.2; a zero base in
/// the first pass, the closest definite base that early). For a vertical
/// table the inline floor is the physical `min-height`, with the
/// vertical padding/border.
fn cell_min_width_bb(forest: &Forest, node: usize, table_w: f32, vertical: bool) -> f32 {
    let n = &forest.nodes[node];
    let min_spec = inline_axis_min_spec(&n.data, vertical);
    let min_w = min_spec
        .resolve(
            n.data.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            table_w,
        )
        .unwrap_or(0.0)
        .max(0.0);
    // `%` padding/border contributes 0 to intrinsic widths (WPT
    // `td-box-sizing-003`: "intrinsic size of % padding is 0"); px
    // values resolve base-independently. Resolve the specified values
    // against a zero base rather than reading the metrics, which may
    // carry a non-zero-base `%` resolve (the `table_pref_width`
    // fixpoint re-resolves against the growing table width, and the
    // final-layout cell pass resolves against the table width).
    let d = &n.data;
    let (fs, root_fs, vp_w, vp_h) = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
    let len0 = |s: SpecLen| {
        s.resolve(fs, root_fs, vp_w, vp_h, 0.0)
            .unwrap_or(0.0)
            .max(0.0)
    };
    // `border-style: none | hidden` zeroes the used width (CSS 2.1
    // §8.5.3). Physical indices: top=0, right=1, bottom=2, left=3; a
    // vertical table's inline axis is physical vertical.
    let bw = |i: usize| {
        if d.border_style[i].zeroes_width() {
            0.0
        } else {
            len0(d.border[i])
        }
    };
    let pad = if vertical {
        bw(0) + len0(d.padding[0]) + len0(d.padding[2]) + bw(2)
    } else {
        bw(3) + len0(d.padding[3]) + len0(d.padding[1]) + bw(1)
    };
    match n.data.box_sizing {
        BoxSizing::BorderBox => min_w,
        BoxSizing::ContentBox => min_w + pad,
    }
}

/// A vertical table cell's specified inline size as a resolved
/// content-box value: the physical `height` property is the inline spec
/// in a vertical writing mode, so this mirrors the `content_w` arm of
/// `resolve_metrics` on the transposed axis (`min-height`/`max-height`
/// clamp in the box-sizing's own terms). Percentages resolve against a
/// zero base — the inline size is indefinite this early, so they compute
/// to `auto` (0.0 here); their percentage-ness is preserved in the
/// `SpecLen` the caller extracted via [`table_col_spec`].
fn cell_specified_inline_height(forest: &Forest, node: usize) -> f32 {
    let n = &forest.nodes[node];
    let d = &n.data;
    let (fs, root_fs, vp_w, vp_h) = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
    let len0 = |s: SpecLen| {
        s.resolve(fs, root_fs, vp_w, vp_h, 0.0)
            .unwrap_or(0.0)
            .max(0.0)
    };
    let min_h = len0(d.min_height);
    let max_h = d
        .max_height
        .resolve(fs, root_fs, vp_w, vp_h, 0.0)
        .unwrap_or(f32::INFINITY)
        .max(0.0);
    let specified = len0(d.height);
    let m = &n.metrics;
    let extras = m.bt + m.pt + m.pb + m.bb;
    let clamped = specified.min(max_h).max(min_h);
    match d.box_sizing {
        BoxSizing::BorderBox => (clamped - extras).max(0.0),
        BoxSizing::ContentBox => clamped,
    }
}

/// The fixed-layout rescale floor: the cell's CSS `min-width` as a
/// border-box width, or 0 when no `min-width` is specified. Unlike
/// [`cell_min_width_bb`] this does not add padding on top of a zero
/// minimum — padding floors the cell's own box during cell layout,
/// not the column, so a zero minimum must not push a rescaled column
/// back over the assignable width and grow the table spuriously. For
/// vertical tables the read is the transposed `min-height`.
fn cell_min_width_floor_bb(forest: &Forest, node: usize, table_w: f32, vertical: bool) -> f32 {
    let n = &forest.nodes[node];
    let min_w = inline_axis_min_spec(&n.data, vertical)
        .resolve(
            n.data.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            table_w,
        )
        .unwrap_or(0.0)
        .max(0.0);
    if min_w <= 0.0 {
        return 0.0;
    }
    cell_min_width_bb(forest, node, table_w, vertical)
}

/// A cell's content-box contribution to its column's maximum, per
/// CSS Tables 3 §3.8.3 (outer max-content width). In a non-constrained
/// column it is `max(min-width, width, min-content width, min(max-width,
/// max-content width))` — an auto cell contributes its max-content
/// width. In a constrained column it is `max(min-width, width,
/// min-content width, min(max-width, width))` — the specified width
/// *replaces* the max-content width, so an auto cell contributes its
/// min-content width, not its max-content width.
/// A cell with a definite specified width always sits in a constrained
/// column (it constrains the column itself), so the definite-width arm
/// is the constrained formula unconditionally: a `width: 100px` cell
/// with 300px of max-content and 30px of min-content contributes 100px,
/// not 300px. `content_max` is the cell's max-content content width,
/// `content_min` its min-content content width, `specified` its
/// resolved specified content width, and `constrained` the column's
/// §3.8.3 constrainedness (see [`column_constrained`]).
fn cell_outer_max_content(
    spec: SpecLen,
    content_max: f32,
    content_min: f32,
    specified: f32,
    constrained: bool,
) -> f32 {
    match spec {
        SpecLen::Px(_) => content_min.max(specified),
        // WPT `column-widths` tables 13/17: the auto cell in a
        // constrained column (a `width:50px` cell / `width:100px`
        // `<col>` elsewhere in the column) contributes its min-content
        // (40/110), not its max-content (80/220). `specified` is
        // always 0.0 for Auto, so no max is needed.
        SpecLen::Auto if constrained => content_min,
        _ => content_max,
    }
}

// ---------------------------------------------------------------------------
// Collapsed-border conflict resolution (CSS 2.1 §17.6.2)
//
// The result types ([`CollapsedBorderStyle`], [`CollapsedBorder`],
// [`CollapsedBorders`]) live at the crate root so `bowser_paint` can name
// them; the resolution logic itself is all here.
// ---------------------------------------------------------------------------

impl CollapsedBorderStyle {
    /// Style precedence for equal-width conflicts, CSS 2.1 §17.6.2.1:
    /// `double` > `solid` > `dashed` > `dotted` > `ridge` > `outset` >
    /// `groove` > `inset`.
    fn rank(self) -> u8 {
        match self {
            CollapsedBorderStyle::Double => 8,
            CollapsedBorderStyle::Solid => 7,
            CollapsedBorderStyle::Dashed => 6,
            CollapsedBorderStyle::Dotted => 5,
            CollapsedBorderStyle::Ridge => 4,
            CollapsedBorderStyle::Outset => 3,
            CollapsedBorderStyle::Groove => 2,
            CollapsedBorderStyle::Inset => 1,
            CollapsedBorderStyle::Hidden | CollapsedBorderStyle::None => 0,
        }
    }

    /// The ranked style for a `border-*-style` keyword; anything
    /// unrecognized (and a missing property) resolves to `None`, the
    /// initial value.
    fn of(keyword: Option<&str>) -> CollapsedBorderStyle {
        match keyword {
            Some(k) if k.eq_ignore_ascii_case("hidden") => CollapsedBorderStyle::Hidden,
            Some(k) if k.eq_ignore_ascii_case("double") => CollapsedBorderStyle::Double,
            Some(k) if k.eq_ignore_ascii_case("solid") => CollapsedBorderStyle::Solid,
            Some(k) if k.eq_ignore_ascii_case("dashed") => CollapsedBorderStyle::Dashed,
            Some(k) if k.eq_ignore_ascii_case("dotted") => CollapsedBorderStyle::Dotted,
            Some(k) if k.eq_ignore_ascii_case("ridge") => CollapsedBorderStyle::Ridge,
            Some(k) if k.eq_ignore_ascii_case("outset") => CollapsedBorderStyle::Outset,
            Some(k) if k.eq_ignore_ascii_case("groove") => CollapsedBorderStyle::Groove,
            Some(k) if k.eq_ignore_ascii_case("inset") => CollapsedBorderStyle::Inset,
            _ => CollapsedBorderStyle::None,
        }
    }
}

impl BorderClaimant {
    /// Element precedence rank for equal-width, equal-style conflicts
    /// (CSS 2.1 §17.6.2.1): cell > row > row-group > column >
    /// column-group > table. Higher wins.
    fn rank(self) -> u8 {
        match self {
            BorderClaimant::Cell => 5,
            BorderClaimant::Row => 4,
            BorderClaimant::RowGroup => 3,
            BorderClaimant::Column => 2,
            BorderClaimant::ColumnGroup => 1,
            BorderClaimant::Table => 0,
        }
    }
}

/// One element's claim on a shared edge.
#[derive(Clone, Copy)]
struct BorderClaim {
    width: f32,
    style: CollapsedBorderStyle,
    claimant: BorderClaimant,
    color: Color,
    /// Tie-break order among claims from the same element type: the
    /// claimant further to the top (horizontal edges) or left (vertical
    /// edges) wins (CSS 2.1 §17.6.2.1). Lower wins.
    order: u32,
}

/// Whether `a` beats `b` for a shared edge: wider wins, then the
/// higher-ranked style, then the higher-ranked element, then the
/// topmost/leftmost claimant.
fn claim_beats(a: &BorderClaim, b: &BorderClaim) -> bool {
    if a.width != b.width {
        return a.width > b.width;
    }
    if a.style.rank() != b.style.rank() {
        return a.style.rank() > b.style.rank();
    }
    if a.claimant.rank() != b.claimant.rank() {
        return a.claimant.rank() > b.claimant.rank();
    }
    a.order < b.order
}

/// Resolve one shared edge to its winning border (CSS 2.1 §17.6.2.1).
fn resolve_collapsed_edge(claims: &[BorderClaim]) -> CollapsedBorder {
    // `hidden` anywhere on the edge suppresses every border there.
    if claims
        .iter()
        .any(|c| c.style == CollapsedBorderStyle::Hidden)
    {
        return CollapsedBorder {
            width: 0.0,
            style: CollapsedBorderStyle::Hidden,
            color: Color {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            },
        };
    }
    let mut best: Option<&BorderClaim> = None;
    for c in claims {
        if best.is_none_or(|b| claim_beats(c, b)) {
            best = Some(c);
        }
    }
    match best {
        Some(b) => CollapsedBorder {
            width: b.width,
            style: b.style,
            color: b.color,
        },
        None => CollapsedBorder {
            width: 0.0,
            style: CollapsedBorderStyle::None,
            color: Color {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            },
        },
    }
}

/// Per-side `border-*-style` / `border-*-color` property names, indexed
/// `[top, right, bottom, left]`.
const BORDER_SIDE_STYLE: [&str; 4] = [
    "border-top-style",
    "border-right-style",
    "border-bottom-style",
    "border-left-style",
];
const BORDER_SIDE_COLOR: [&str; 4] = [
    "border-top-color",
    "border-right-color",
    "border-bottom-color",
    "border-left-color",
];

/// The used border color on one side of a node; opaque black when the
/// property is unspecified, matching the paint border path's fallback.
fn claim_color(forest: &Forest, idx: usize, side: usize) -> Color {
    match forest.nodes[idx]
        .style
        .as_ref()
        .and_then(|s| s.get(BORDER_SIDE_COLOR[side]))
    {
        Some(CssValue::Color(c)) => *c,
        _ => Color {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        },
    }
}

/// One table-internal box's claim on one of its sides: the specified
/// border width resolved to px against `pct_base` (zero for
/// `none`/`hidden`, per CSS 2.1 §8.5.3 — the same value
/// `resolve_metrics` stores as the used width), the style keyword (kept
/// so `hidden` still suppresses the edge), and the side's color.
fn claim_side(
    forest: &Forest,
    idx: usize,
    side: usize,
    claimant: BorderClaimant,
    pct_base: f32,
    order: u32,
) -> BorderClaim {
    let n = &forest.nodes[idx];
    let style = CollapsedBorderStyle::of(
        n.style
            .as_ref()
            .and_then(|s| s.get_str(BORDER_SIDE_STYLE[side])),
    );
    let width = if matches!(
        style,
        CollapsedBorderStyle::Hidden | CollapsedBorderStyle::None
    ) {
        0.0
    } else {
        n.data.border[side]
            .resolve(
                n.data.font_size,
                forest.root_font_size,
                forest.vp_w,
                forest.vp_h,
                pct_base,
            )
            .unwrap_or(0.0)
            .max(0.0)
    };
    BorderClaim {
        width,
        style,
        claimant,
        color: claim_color(forest, idx, side),
        order,
    }
}

/// Compute the collapsed-border winners for every cell edge
/// (CSS 2.1 §17.6.2). Returns `None` unless the table uses
/// `border-collapse: collapse`.
///
/// Each interior edge collects the two adjacent cells' claims (the
/// neighbor is the cell covering the grid position adjoining the
/// edge's top-left corner; spanning cells therefore resolve against a
/// single neighbor per edge). Table-boundary edges additionally collect
/// the row (horizontal edges only), row-group, and table claims, so
/// e.g. a table-level `border` participates on the outer edges and a
/// `hidden` table border suppresses them.
pub(crate) fn resolve_collapsed_borders(
    forest: &Forest,
    table: usize,
    grid: &TableGrid,
    content_w: f32,
) -> Option<CollapsedBorders> {
    let collapse = forest.nodes[table]
        .style
        .as_ref()
        .and_then(|s| s.get_str_by_index(props::BORDER_COLLAPSE))
        .is_some_and(|v| v.eq_ignore_ascii_case("collapse"));
    if !collapse || grid.nrows == 0 || grid.ncols == 0 || grid.cells.is_empty() {
        return None;
    }
    // (row, col) -> index into `grid.cells`; rows are only as long as
    // their occupied columns, so hostile sparse grids stay small.
    let mut occ: Vec<Vec<u32>> = Vec::new();
    for (i, cell) in grid.cells.iter().enumerate() {
        let r_end = (cell.row + cell.rowspan).min(grid.nrows);
        let c_end = (cell.col + cell.colspan).min(grid.ncols);
        if occ.len() < r_end {
            occ.resize(r_end, Vec::new());
        }
        for row in occ.iter_mut().take(r_end).skip(cell.row) {
            if row.len() < c_end {
                row.resize(c_end, u32::MAX);
            }
            row[cell.col..c_end].fill(i as u32);
        }
    }
    let neighbor = |r: usize, c: usize, this: usize| -> Option<usize> {
        occ.get(r)
            .and_then(|row| row.get(c))
            .copied()
            .filter(|&n| n != u32::MAX && n as usize != this)
            .map(|n| n as usize)
    };

    let mut claims = Vec::with_capacity(6);
    let mut out = Vec::with_capacity(grid.cells.len());
    for (i, cell) in grid.cells.iter().enumerate() {
        let r = cell.row;
        let c = cell.col;
        let r_end = (cell.row + cell.rowspan).min(grid.nrows);
        let c_end = (cell.col + cell.colspan).min(grid.ncols);
        let row_top = grid.row_nodes.get(r).copied().flatten();
        let row_bottom = grid
            .row_nodes
            .get(r_end.saturating_sub(1))
            .copied()
            .flatten();
        let group_top = grid.row_group.get(r).copied().flatten();
        let group_bottom = grid
            .row_group
            .get(r_end.saturating_sub(1))
            .copied()
            .flatten();

        // Top edge.
        claims.clear();
        claims.push(claim_side(
            forest,
            cell.node,
            0,
            BorderClaimant::Cell,
            content_w,
            1,
        ));
        if r > 0
            && let Some(nb) = neighbor(r - 1, c, i)
        {
            let nb_node = grid.cells[nb].node;
            claims.push(claim_side(
                forest,
                nb_node,
                2,
                BorderClaimant::Cell,
                content_w,
                0,
            ));
        }
        if let Some(rn) = row_top {
            claims.push(claim_side(forest, rn, 0, BorderClaimant::Row, content_w, 0));
        }
        if let Some(g) = group_top {
            claims.push(claim_side(
                forest,
                g,
                0,
                BorderClaimant::RowGroup,
                content_w,
                0,
            ));
        }
        if r == 0 {
            claims.push(claim_side(
                forest,
                table,
                0,
                BorderClaimant::Table,
                content_w,
                0,
            ));
            // Column and column-group top borders lie on the table's top
            // edge (columns span the full grid height).
            for cc in c..c_end {
                if let Some(spec) = grid.columns.get(cc) {
                    if let Some(col) = spec.col_node {
                        claims.push(claim_side(
                            forest,
                            col,
                            0,
                            BorderClaimant::Column,
                            content_w,
                            0,
                        ));
                    }
                    if let Some(g) = spec.group_node {
                        claims.push(claim_side(
                            forest,
                            g,
                            0,
                            BorderClaimant::ColumnGroup,
                            content_w,
                            0,
                        ));
                    }
                }
            }
        }
        let top = resolve_collapsed_edge(&claims);

        // Bottom edge.
        claims.clear();
        claims.push(claim_side(
            forest,
            cell.node,
            2,
            BorderClaimant::Cell,
            content_w,
            0,
        ));
        if r_end < grid.nrows
            && let Some(nb) = neighbor(r_end, c, i)
        {
            let nb_node = grid.cells[nb].node;
            claims.push(claim_side(
                forest,
                nb_node,
                0,
                BorderClaimant::Cell,
                content_w,
                1,
            ));
        }
        if let Some(rn) = row_bottom {
            claims.push(claim_side(forest, rn, 2, BorderClaimant::Row, content_w, 0));
        }
        if let Some(g) = group_bottom {
            claims.push(claim_side(
                forest,
                g,
                2,
                BorderClaimant::RowGroup,
                content_w,
                0,
            ));
        }
        if r_end == grid.nrows {
            claims.push(claim_side(
                forest,
                table,
                2,
                BorderClaimant::Table,
                content_w,
                0,
            ));
            // Column and column-group bottom borders lie on the table's
            // bottom edge.
            for cc in c..c_end {
                if let Some(spec) = grid.columns.get(cc) {
                    if let Some(col) = spec.col_node {
                        claims.push(claim_side(
                            forest,
                            col,
                            2,
                            BorderClaimant::Column,
                            content_w,
                            0,
                        ));
                    }
                    if let Some(g) = spec.group_node {
                        claims.push(claim_side(
                            forest,
                            g,
                            2,
                            BorderClaimant::ColumnGroup,
                            content_w,
                            0,
                        ));
                    }
                }
            }
        }
        let bottom = resolve_collapsed_edge(&claims);

        // Left edge.
        claims.clear();
        claims.push(claim_side(
            forest,
            cell.node,
            3,
            BorderClaimant::Cell,
            content_w,
            1,
        ));
        if c > 0
            && let Some(nb) = neighbor(r, c - 1, i)
        {
            let nb_node = grid.cells[nb].node;
            claims.push(claim_side(
                forest,
                nb_node,
                1,
                BorderClaimant::Cell,
                content_w,
                0,
            ));
        }
        if c == 0 {
            if let Some(g) = group_top {
                claims.push(claim_side(
                    forest,
                    g,
                    3,
                    BorderClaimant::RowGroup,
                    content_w,
                    0,
                ));
            }
            claims.push(claim_side(
                forest,
                table,
                3,
                BorderClaimant::Table,
                content_w,
                0,
            ));
        }
        // Column and column-group borders claim the vertical grid edges
        // (CSS 2.1 §17.6.2.1): the column to the left contributes its
        // right border, the column itself its left border.
        if c > 0
            && let Some(spec) = grid.columns.get(c - 1)
        {
            // A bare `colgroup` (no `col` children) still claims the
            // edge: the group claim is not gated on a column box.
            if let Some(col) = spec.col_node {
                claims.push(claim_side(
                    forest,
                    col,
                    1,
                    BorderClaimant::Column,
                    content_w,
                    0,
                ));
            }
            if let Some(g) = spec.group_node {
                claims.push(claim_side(
                    forest,
                    g,
                    1,
                    BorderClaimant::ColumnGroup,
                    content_w,
                    0,
                ));
            }
        }
        if let Some(spec) = grid.columns.get(c) {
            if let Some(col) = spec.col_node {
                claims.push(claim_side(
                    forest,
                    col,
                    3,
                    BorderClaimant::Column,
                    content_w,
                    0,
                ));
            }
            if let Some(g) = spec.group_node {
                claims.push(claim_side(
                    forest,
                    g,
                    3,
                    BorderClaimant::ColumnGroup,
                    content_w,
                    0,
                ));
            }
        }
        let left = resolve_collapsed_edge(&claims);

        // Right edge.
        claims.clear();
        claims.push(claim_side(
            forest,
            cell.node,
            1,
            BorderClaimant::Cell,
            content_w,
            0,
        ));
        if c_end < grid.ncols
            && let Some(nb) = neighbor(r, c_end, i)
        {
            let nb_node = grid.cells[nb].node;
            claims.push(claim_side(
                forest,
                nb_node,
                3,
                BorderClaimant::Cell,
                content_w,
                1,
            ));
        }
        if c_end == grid.ncols {
            if let Some(g) = group_top {
                claims.push(claim_side(
                    forest,
                    g,
                    1,
                    BorderClaimant::RowGroup,
                    content_w,
                    0,
                ));
            }
            claims.push(claim_side(
                forest,
                table,
                1,
                BorderClaimant::Table,
                content_w,
                0,
            ));
        }
        // Column and column-group borders claim the vertical grid edges:
        // the column itself contributes its right border, the column to
        // the right its left border.
        if let Some(spec) = grid.columns.get(c_end.saturating_sub(1)) {
            if let Some(col) = spec.col_node {
                claims.push(claim_side(
                    forest,
                    col,
                    1,
                    BorderClaimant::Column,
                    content_w,
                    0,
                ));
            }
            if let Some(g) = spec.group_node {
                claims.push(claim_side(
                    forest,
                    g,
                    1,
                    BorderClaimant::ColumnGroup,
                    content_w,
                    0,
                ));
            }
        }
        if c_end < grid.ncols
            && let Some(spec) = grid.columns.get(c_end)
        {
            // A bare `colgroup` (no `col` children) still claims the
            // edge: the group claim is not gated on a column box.
            if let Some(col) = spec.col_node {
                claims.push(claim_side(
                    forest,
                    col,
                    3,
                    BorderClaimant::Column,
                    content_w,
                    0,
                ));
            }
            if let Some(g) = spec.group_node {
                claims.push(claim_side(
                    forest,
                    g,
                    3,
                    BorderClaimant::ColumnGroup,
                    content_w,
                    0,
                ));
            }
        }
        let right = resolve_collapsed_edge(&claims);

        out.push([top, right, bottom, left]);
    }
    Some(CollapsedBorders { cells: out })
}

/// Resume state for a table whose cells are laid out.
pub(crate) struct TableResume {
    pub idx: usize,
    pub content_top: f32,
    /// Table content-box x (for float placement).
    pub content_x: f32,
    /// Table content-box width for float placement: the
    /// scrollbar-gutter-reserved width the columns resolved against
    /// (plus any growth the engine applied), NOT the node's unreserved
    /// metrics width — floats must not intrude into the reserved
    /// gutter strip.
    pub content_w: f32,
    pub cb_h: Option<f32>,
    /// Containing-block width, for resolving vertical-table block-axis
    /// percentages (which use the CB width, not height).
    pub cb_w: f32,
    pub hspec: Option<f32>,
    pub collapse: bool,
    pub spacing_v: f32,
    /// Horizontal border-spacing, for recomputing each cell's spanned
    /// grid width in `resume_table` (zero when borders collapse).
    pub spacing_h: f32,
    pub nrows: usize,
    pub cells: Vec<GridCell>,
    /// Per-cell block (horizontal) content size for vertical tables
    /// (parallel to `cells`): the laid-out `rect.h` carries vertical
    /// content, so `resume_table` stamps these into `rect.h` to run the
    /// row-height machinery in (inline, block) space. Empty for
    /// horizontal tables.
    pub cell_block: Vec<f32>,
    /// CSS 2.1 §17.5 per-row `visibility: collapse` flags; see
    /// [`TableGrid::collapsed_rows`]. `resume_table` zeroes these rows'
    /// heights and drops the spacing around them.
    pub collapsed_rows: Vec<bool>,
    /// CSS 2.1 §17.5 per-column `visibility: collapse` flags; see
    /// [`TableGrid::collapsed_cols`]. Column widths already resolved to
    /// zero; the flags serve the collapsed-border ownership fixup below.
    pub collapsed_cols: Vec<bool>,
    /// Collapsed-border winners per cell edge, `Some` only when the
    /// table uses `border-collapse: collapse`; see
    /// [`resolve_collapsed_borders`] for the paint contract.
    /// `resume_table` hands each cell's winners to its arena node (see
    /// [`crate::LayoutBox::collapsed_borders`]).
    pub collapsed: Option<CollapsedBorders>,
    pub captions_top: Vec<usize>,
    pub captions_bottom: Vec<usize>,
    /// Row border-box x/width (first column x to last column right edge).
    pub row_x: f32,
    pub row_w: f32,
    /// Parallel to the grid rows; see [`TableGrid`].
    pub row_nodes: Vec<Option<usize>>,
    pub row_group: Vec<Option<usize>>,
    pub group_nodes: Vec<usize>,
    /// Floated children of the table, laid out through the float
    /// machinery and positioned in the resume pass.
    pub floats: Vec<usize>,
    /// Absolute border-box x/width per column (for column-box rects).
    pub col_x: Vec<f32>,
    pub col_w: Vec<f32>,
    /// Per-column covering column boxes; see [`TableGrid`].
    pub columns: Vec<ColumnSpec>,
    /// css-tables-3 §3.10.6 second pass: when the measurement resume
    /// found cells whose content carries percentage heights, it defers
    /// placement, re-lays those cells against their used heights, and
    /// re-queues this frame with `phase2` set and the resolved heights
    /// in the `phase_*` fields; the second resume only places.
    pub phase2: bool,
    /// Resolved row heights carried into the placement resume.
    pub phase_row_h: Vec<f32>,
    /// Per-cell baseline shifts carried into the placement resume.
    pub phase_bshift: Vec<f32>,
    /// Per-group target heights carried into the placement resume.
    pub phase_group_target: Vec<f32>,
    /// Indices into `cells` of the §3.10.6 second-pass cells, carried
    /// into the placement resume. Only these cells were re-laid (back in
    /// physical space); the vertical-table forward transpose must touch
    /// only their subtrees — captions, floats, and non-pass2 cells are
    /// already in (inline, block) space from the measurement resume and
    /// must not be transposed twice.
    pub phase2_cells: Vec<usize>,
    /// Stacked block extent from `place_table_rows` (rows plus
    /// non-deferred captions, in algorithm space): the deferred
    /// caption phase's grid base. Unlike the finalized content
    /// size, it excludes the specified-size floor, which
    /// `place_deferred_captions` must not double-count (w77fix
    /// finding 7). Set by `resume_table` before
    /// [`push_deferred_caption_tasks`]; 0.0 until then.
    pub stacked_block: f32,
    /// Cursor after rows plus non-deferred captions but BEFORE
    /// the float-containment max in `place_table_rows` (w77fix
    /// finding 14): the rows' true block-end edge. The finalized
    /// content size and `stacked_block` both carry any float
    /// extension past the rows, but floats do not push normal
    /// flow — bottom deferred captions sit after the rows, not
    /// after the floats. Set alongside `stacked_block` before
    /// [`push_deferred_caption_tasks`]; 0.0 until then (and
    /// legitimately 0.0 for a rowless table whose cursor never
    /// advanced before the floats).
    pub pre_float_block: f32,
    /// Deferred orthogonal captions (w77-d task 1, extended by
    /// w77fix finding 7 to definite block sizes): horizontal
    /// captions in a vertical table, held out of the first task
    /// push and of `captions_top`/`captions_bottom` until the
    /// table's block size is finalized by placement; split by
    /// `caption-side` like those lists. [`place_deferred_captions`]
    /// positions them (and extends the table) after they re-lay.
    pub deferred_top: Vec<usize>,
    pub deferred_bottom: Vec<usize>,
    /// DOM order of every caption on each side (deferred plus
    /// non-deferred): the final placement stacks the merged lists so
    /// a deferred caption interleaves with non-deferred captions in
    /// document order instead of stacking before or after all of them.
    pub top_all: Vec<usize>,
    pub bottom_all: Vec<usize>,
    /// Final caption-placement resume (w77-d task 1): when set, the
    /// table is already placed and transposed to physical space, so
    /// this resume only runs [`place_deferred_captions`] — it must NOT
    /// run the forward transpose or the row machinery again.
    pub phase_caption: bool,
}

/// Clamp an over-constrained table growth delta so the grown table stays
/// within its own `max-width`, mirroring [`resolve_metrics`]' width
/// clamping (in the box-sizing's own terms: content width for
/// `content-box`, border-box width for `border-box`). The columns then
/// overflow the table instead of pushing it past its cap, matching
/// browsers.
fn clamp_growth_delta(forest: &Forest, idx: usize, cb_w: f32, delta: f32) -> f32 {
    let n = &forest.nodes[idx];
    let d = &n.data;
    // The growth is along the inline axis: for a vertical table the
    // inline max is the physical `max-height`.
    let max_spec = if super::is_vertical_table(forest, idx) {
        d.max_height
    } else {
        d.max_width
    };
    let max_w = max_spec
        .resolve(
            d.font_size,
            forest.root_font_size,
            forest.vp_w,
            forest.vp_h,
            cb_w,
        )
        .unwrap_or(f32::INFINITY)
        .max(0.0);
    if max_w.is_infinite() {
        return delta;
    }
    // Growth widens the content box by `delta`, and the border box by the
    // same amount; the headroom is the gap between `max-width` and the
    // already-clamped used width from `resolve_metrics`.
    let headroom = match d.box_sizing {
        BoxSizing::BorderBox => max_w - n.rect.w,
        BoxSizing::ContentBox => max_w - n.metrics.content_w,
    };
    delta.min(headroom.max(0.0))
}

/// The used inline size (physical height) of a caption in a vertical
/// writing mode whose `height` is a content-based keyword, or `None`
/// when the caption keeps the default horizontal sizing.
///
/// A vertical-writing-mode caption consumes its inline size from the
/// physical `height` property (css-writing-modes-3 §7), so
/// `height: min-content | max-content` resolves against the *vertical*
/// intrinsic size ([`vertical_caption_min_content`] /
/// [`vertical_caption_max_content`]) — not the horizontal line stack
/// the writing-mode-unaware block engine would otherwise produce from
/// `SpecLen::MinContent.resolve() == None` (WPT `caption.html`
/// table 13: at the table's 144px content width the caption text wraps
/// to 124px instead of the 100px vertical min-content).
///
/// The returned size is a content-box inline size, handed to the
/// caption's [`Task::Block`] as `height_override`, which makes the
/// block engine stamp it as the definite used height (physical height
/// IS the inline axis here). A definite `height` already resolves
/// through `specified_height`; `auto`, percentage, and `fit-content`
/// heights keep the historical horizontal behavior — the engine has no
/// vertical inline layout, so only the keywords with a well-defined
/// vertical intrinsic take this path.
///
/// Only for horizontal tables: a vertical table's captions already run
/// through the (inline, block) transpose pipeline
/// ([`vwm_cells_to_logical`]), whose behavior the w65 vertical-caption
/// tests pin — this override must not disturb it.
fn vertical_caption_inline_size(
    forest: &mut Forest,
    cap: usize,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    grid_vertical: bool,
) -> Option<f32> {
    if grid_vertical {
        return None;
    }
    let d = &forest.nodes[cap].data;
    if !matches!(
        d.writing_mode,
        WritingMode::VerticalLr | WritingMode::VerticalRl
    ) {
        return None;
    }
    if !matches!(d.height, SpecLen::MinContent | SpecLen::MaxContent) {
        return None;
    }
    // Captions are rare (usually zero or one per table); the vertical
    // intrinsic walks allocate their scratch on demand here rather than
    // threading more pooled buffers through `layout_table`.
    let mut mcw_scratch = Vec::new();
    let mut int_bufs = WalkBufs::new();
    let v = match d.height {
        SpecLen::MinContent => {
            vertical_caption_min_content(forest, cap, fonts, &mut mcw_scratch, &mut int_bufs)
        }
        _ => {
            let mut int_inner = WalkBufs::new();
            vertical_caption_max_content(
                forest,
                cap,
                fonts,
                pw_cache,
                &mut mcw_scratch,
                &mut int_bufs,
                &mut int_inner,
            )
        }
    };
    Some(v.max(0.0))
}

/// Whether a caption is held out of the first task push for the
/// deferred orthogonal-caption path (w77-d task 1, extended by
/// w77fix finding 7 to definite block sizes): a horizontal caption
/// in a vertical table. Its containing-block width is the table's
/// finalized block size, which does not exist until
/// `place_table_rows` / `finish_content_height` run, so it lays
/// out after placement (see [`push_deferred_caption_tasks`] /
/// [`place_deferred_captions`]). Definite-spec captions used to
/// keep a fast path against the specified size, but that base is
/// wrong when the rows grow past it: the caption's base is
/// max(specified, grid), and the grid is only known after
/// placement — so definite captions defer too, with their base
/// floored by the clamped specified size (see
/// [`push_deferred_caption_tasks`]). Same-writing-mode captions
/// keep the historical inline-`content_w` base and are never
/// deferred.
fn is_deferred_orthogonal_caption(forest: &Forest, cap: usize, grid_vertical: bool) -> bool {
    grid_vertical
        && matches!(
            forest.nodes[cap].data.writing_mode,
            WritingMode::HorizontalTb
        )
}

/// Content-width override for an auto-width caption in a vertical
/// table, so its border box on the table's block axis does not shrink
/// under its own physical-horizontal margins (w77-d task 2).
///
/// `place_table_rows` stacks a vertical-table caption along the block
/// axis and advances its cursor by the caption's block margins plus its
/// laid-out block extent (`mbs + h + mbe`, where `h` is the caption's
/// border box on that axis). The block engine, however, lays the caption
/// out as a horizontal block whose width axis IS the physical-horizontal
/// axis, so an auto width fills `cb_w` minus `ml`/`mr` — the same
/// margins the cursor then adds again. A caption with 40px side margins
/// in a 30px-wide inline content box collapsed from a 30px extent to 0,
/// so the table grew by only 50px (80 margins minus the 30 lost extent)
/// instead of 80 (WPT `caption-orthogonal-margin-in-table-block-axis`,
/// both subtests). This affects same-writing-mode captions (whose
/// block margins are `ml`/`mr`) and orthogonal captions (whose inline
/// margins are `ml`/`mr`) identically, because the cursor reads the
/// physical margins in both cases.
///
/// The override hands the engine the content width that yields a
/// `cb_w` border box — `cb_w` minus the caption's horizontal
/// border/padding, resolved exactly as [`super::resolve_metrics`]
/// resolves them (border widths against `cb_w`, padding against the
/// orthogonal-caption percentage base) — so the cursor's margin terms
/// are the only margin contribution. Min/max clamping in the override
/// arm then applies to the block extent itself, which is the correct
/// axis for those properties here.
///
/// Gated narrowly, and deliberately NOT applied to the definite-spec
/// fast path: when an orthogonal caption lays out against a specified
/// block size, its margin box is constrained to that size — the w75/w76
/// tests pin a 300px block base with a 40px margin giving a 260px
/// border box. The override applies only in the auto regime, where the
/// fill base is the table's inline content width. Other gates: only
/// auto `width` (a specified or content-based width never subtracts
/// margins), and only when neither horizontal margin is `auto` (with
/// an auto margin the engine's over-constrained handling would turn
/// the override into a negative used margin; auto margins keep the
/// historical fill behavior).
fn caption_block_axis_width_override(forest: &Forest, cap: usize, cb_w: f32) -> Option<f32> {
    let d = &forest.nodes[cap].data;
    if !matches!(d.width, SpecLen::Auto) {
        return None;
    }
    if matches!(d.margin[3], SpecMargin::Auto) || matches!(d.margin[1], SpecMargin::Auto) {
        return None;
    }
    // Percentage base, mirroring `resolve_metrics`' orthogonal-caption
    // split: an orthogonal caption's percentage padding resolves
    // against the parent table's inline content size when known.
    let pct_base = {
        let is_orth_caption = matches!(d.writing_mode, WritingMode::HorizontalTb)
            && forest.nodes[cap].parent.is_some_and(|p| {
                super::is_vertical_table(forest, p)
                    && forest.nodes[cap]
                        .style
                        .as_ref()
                        .and_then(|s| s.get_str_by_index(props::DISPLAY))
                        .is_some_and(|v| v.eq_ignore_ascii_case("table-caption"))
            });
        if is_orth_caption {
            forest.nodes[cap]
                .parent
                .map(|p| forest.nodes[p].metrics.content_w)
                .filter(|&w| w > 0.0)
                .unwrap_or(cb_w)
        } else {
            cb_w
        }
    };
    let (fs, root_fs, vp_w, vp_h) = (d.font_size, forest.root_font_size, forest.vp_w, forest.vp_h);
    let len = |s: SpecLen, base: f32| s.resolve(fs, root_fs, vp_w, vp_h, base).unwrap_or(0.0);
    let bl = if d.border_style[3].zeroes_width() {
        0.0
    } else {
        len(d.border[3], cb_w)
    };
    let br = if d.border_style[1].zeroes_width() {
        0.0
    } else {
        len(d.border[1], cb_w)
    };
    let pl = len(d.padding[3], pct_base);
    let pr = len(d.padding[1], pct_base);
    Some((cb_w - (bl + pl + pr + br)).max(0.0))
}

/// Lay out a `display: table` box. `layout_block` resolved the table's own
/// metrics and top margin before dispatching here; this collects the grid,
/// resolves column widths, and pushes one [`Task::Block`] per cell (plus
/// captions) with the cell's content width as a width override, followed
/// by the [`Task::Table`] resume (pushed first: the task stack is LIFO).
#[allow(clippy::too_many_arguments)]
pub(crate) fn layout_table(
    forest: &mut Forest,
    tasks: &mut Vec<Task>,
    idx: usize,
    input: &BlockInput,
    fonts: &dyn FontMetrics,
    pw_cache: &mut PrefWidthCache,
    kids_x: f32,
    kids_w: f32,
    pool: &mut TableGridPool,
) {
    let (collapse, spacing_h, spacing_v) = table_border_model(forest, idx);
    wrap_table_stray_content(forest, idx);
    // One pooled collection serves the whole table layout; the grid is
    // returned to the pool at the end (nested tables check out their
    // own while this one is held).
    let mut grid = pool.take_grid();
    collect_table_grid_into(forest, idx, pool, &mut grid);
    // `table-layout: fixed`: column widths come from the table's width
    // and the first row's specified cell widths (CSS 2.1 §17.5.2.1),
    // without measuring content.
    let fixed = grid.fixed;

    // `kids_x` / `kids_w` are the `layout_block` prologue's
    // scrollbar-gutter-reserved content origin / width for children:
    // columns resolve against those, not the node's unreserved
    // metrics (identical to them when no reservation applies).
    let (mut content_x, content_top, mut content_w) = {
        let n = &forest.nodes[idx];
        (kids_x, n.rect.y + n.metrics.bt + n.metrics.pt, kids_w)
    };
    let hspec = input.height_override.or_else(|| {
        // A vertical table's block size resolves from the physical
        // `width`; percentage widths resolve against the containing-
        // block width (always definite), not the height.
        if super::is_vertical_table(forest, idx) {
            specified_height(forest, idx, Some(input.cb_w))
        } else {
            specified_height(forest, idx, input.cb_h)
        }
    });

    // Out-of-flow children take no space; record them with a static
    // position at the table's content origin.
    for &c in &grid.abs_children {
        layout_position::record_abs_item(forest, c, content_x, content_top);
    }

    // Resolve each cell's padding/border now (margins are zeroed at box
    // construction for table-internal boxes), so column widths account
    // for them. Percentage padding resolves against the table's content
    // width, the closest definite base available this early. Cells that
    // will never lay out (collapsed row or fully-collapsed columns) are
    // skipped: their metrics feed no width computation, and
    // `resolve_metrics` stamps `rect.w` as a side effect which a skipped
    // cell must not keep.
    for cell in &grid.cells {
        if cell_width_contributes(&grid.collapsed_cols, cell) {
            resolve_metrics(forest, cell.node, 0.0, content_w, false, None, false);
        }
    }

    // Vertical writing mode: the row (block-axis) sizes need each cell's
    // block (horizontal) content size, but the cell tasks below lay out
    // with the column (inline) width as the width override, so the
    // laid-out `rect.h` (vertical content) does not carry it.
    // Measure it intrinsically here — mirroring `auto_column_maxima`'s
    // per-cell outer max-content formula — and stash it on the resume
    // frame; `resume_table` stamps it into the cells' `rect.h` so the
    // row-height machinery runs in (inline, block) space.
    let mut cell_block = vec![0.0f32; grid.cells.len()];
    if super::is_vertical_table(forest, idx) {
        let mut mcw_scratch: Vec<f32> = Vec::new();
        let mut int_bufs = WalkBufs::new();
        let mut int_inner = WalkBufs::new();
        for (i, cell) in grid.cells.iter().enumerate() {
            if !cell_width_contributes(&grid.collapsed_cols, cell) {
                continue;
            }
            let pcw = preferred_width_mcw(
                forest,
                cell.node,
                fonts,
                pw_cache,
                &mut mcw_scratch,
                &mut int_bufs,
                &mut int_inner,
            );
            let spec = table_col_spec(forest.nodes[cell.node].data.width);
            // The cell's own specified width is a floor on its block
            // size (it stays a physical property: only the table's own
            // writing mode drives the transpose).
            let specified = match spec {
                SpecLen::Auto => 0.0,
                _ => forest.nodes[cell.node].metrics.content_w,
            };
            let mcw = if matches!(spec, SpecLen::Px(_)) {
                min_content_width_pooled(forest, cell.node, fonts, &mut mcw_scratch, &mut int_bufs)
            } else {
                0.0
            };
            let pad = cell_h_pad(
                &forest.nodes[cell.node].metrics,
                collapse,
                super::is_vertical_table(forest, idx),
            );
            cell_block[i] =
                cell_outer_max_content(spec, pcw.max(specified), mcw, specified, false) + pad;
        }
    }

    // Specified `<col>` / `<colgroup>` widths: border-box widths plus
    // the per-column group-spread-only flag (see
    // `column_specified_widths`). A `colgroup` width spreads over its
    // whole span; under `table-layout: fixed` the spread pins its
    // columns, in the automatic layout it is a minimum.
    let mut col_spec = column_specified_widths(forest, &grid, content_x, content_w);

    // Column widths: fixed layout takes them from the table's width and
    // the first row's specified cell widths (no content measurement);
    // auto layout measures min-content widths and distributes the
    // leftover table width.
    let ncols = grid.ncols;
    // Collapsed columns take no width and receive none of the table's
    // distributed width (CSS 2.1 §17.5); only visible columns draw
    // inter-column spacing.
    let nvis_cols = grid.collapsed_cols.iter().filter(|&&c| !c).count();
    let mut col_w = vec![0.0f32; ncols];
    if !fixed {
        // CSS Tables 3 §3.9.2.1 rule 3 (mirroring the fixed-layout
        // rebasing below): percentage `<col>`/`<colgroup>` widths
        // resolve against the assignable width (content width minus
        // border-spacing), never against the spacing-inclusive content
        // width. `column_specified_widths` resolved them against
        // `content_w`; rebase the percentage-derived part onto the
        // assignable width before they floor the intrinsic
        // minima/maxima (WPT `colspan-redistribution` table 26: the 50%
        // `<col>` floors its column at 142px of the 284px assignable
        // width, not 154px of the 308px content width). Percentage
        // resolution is linear in the base, so subtracting
        // `pct * spacing_total` is exact for both box-sizing modes.
        // The fixed layout keeps the un-rebased pins: it rebases the
        // percentage-derived part itself further below.
        let spacing_total = if collapse || nvis_cols == 0 {
            0.0
        } else {
            (nvis_cols as f32 + 1.0) * spacing_h
        };
        let pcts = col_spec_pct_fractions(forest, &grid);
        for (cc, (w, &p)) in col_spec.0.iter_mut().zip(pcts.iter()).enumerate() {
            if p > 0.0
                && !grid.collapsed_cols[cc]
                && let Some(bb) = w
            {
                *bb = (*bb - p * spacing_total).max(0.0);
            }
        }
    }
    if fixed {
        // CSS 2.1 §17.5.2.1: column widths come from the column boxes
        // first (`<col>` / `<colgroup>` with a non-`auto` width pin each
        // column in the element's span), then from first-row cells with a
        // non-`auto` width (a spanning cell's width divides over its
        // columns). Columns without a specified width share the remaining
        // table width equally; when every column is pinned and the table
        // is wider than the pins, the excess distributes per CSS Tables 3
        // §3.9.3.2 (proportionally to length widths, then to percentage
        // widths, then equally). Cell content that does not fit overflows.
        let mut col_set = vec![false; ncols];
        // Per-column excess-distribution state for CSS Tables 3
        // §3.9.3.2 below: the pinned base-assignment width and the
        // column's intrinsic percentage fraction. The merged
        // `col_spec` pins already resolved percentages to px, so the
        // percentage is recovered from the pinning spec with the same
        // csswg#94 normalization the pin used (`table_col_spec`): a
        // `<col>` percentage wins over its group's; a group percentage
        // is the spread column's pinned px width over the table content
        // width, exactly mirroring the px spread in
        // `column_specified_widths`. Collapsed columns keep zeros and
        // stay excluded from every rule.
        let mut col_base = vec![0.0f32; ncols];
        let mut col_pct = vec![0.0f32; ncols];
        // Per-column CSS `min-width` floor (border-box) from the
        // pinning first-row cells, re-applied after the over-100%
        // rescale multiply below.
        let mut col_minw = vec![0.0f32; ncols];
        // Collapsed columns pin at zero width up front: specified
        // `<col>` widths and first-row cell widths never apply to them,
        // and the leftover-width distribution below skips set columns.
        for (cc, &gone) in grid.collapsed_cols.iter().enumerate() {
            if gone {
                col_set[cc] = true;
            }
        }
        for (cc, w) in col_spec.0.iter().enumerate() {
            if grid.collapsed_cols[cc] {
                continue;
            }
            if let Some(bb) = w {
                col_w[cc] = *bb;
                col_set[cc] = true;
                col_base[cc] = *bb;
            }
        }
        for (cc, spec) in grid.columns.iter().enumerate() {
            if grid.collapsed_cols[cc] {
                continue;
            }
            if col_spec.1[cc] {
                // Pinned by the group spread: the true percentage share
                // is the pinned px width over the table content width,
                // exactly mirroring the px spread in
                // `column_specified_widths` (`per = (w -
                // pinned_sum).max(0.0) / covered`). Dividing the group's
                // raw percentage over the covered columns double-counts
                // what the pinned columns already take: a 60% group with
                // one 20%-pinned column and one spread column records
                // 60% for the spread column when its true share is 40%.
                // The zero guard is belt-and-braces: content_w == 0
                // forces excess <= 0 below, so the share goes unused.
                if let Some(g) = spec.group_node
                    && let SpecLen::Pct(_) =
                        table_col_spec(inline_axis_spec(&forest.nodes[g].data, grid.vertical))
                {
                    col_pct[cc] = if content_w > 0.0 {
                        col_base[cc] / content_w
                    } else {
                        0.0
                    };
                }
            } else if let Some(n) = spec.col_node
                && let SpecLen::Pct(p) =
                    table_col_spec(inline_axis_spec(&forest.nodes[n].data, grid.vertical))
            {
                col_pct[cc] = p / 100.0;
            }
        }
        // Fixed mode pins from the first *visual* row (see
        // `first_visual_row`), not the first DOM row.
        let first_row = first_visual_row(forest, &grid);
        for cell in grid.cells.iter().filter(|c| c.row == first_row) {
            // Only cells spanning solely collapsed columns pin nothing:
            // collapsed-row cells never lay out, but they still size
            // columns (CSS 2.1 §17.5.5), so they pin here too.
            if !cell_width_contributes(&grid.collapsed_cols, cell) {
                continue;
            }
            let n = &forest.nodes[cell.node];
            if matches!(inline_axis_spec(&n.data, grid.vertical), SpecLen::Auto) {
                continue;
            }
            // The pin's width is the resolved border-box width (from the
            // raw spec, so a mixed `calc()` still pins as a length
            // here), but rule classification below uses the csswg#94
            // normalization: only a pure percentage (or a
            // percentage-only `calc()`) makes the column a percentage
            // participant; anything else is a length column. A spanning
            // percentage shares over its span, mirroring the px pin.
            let cell_pct = match table_col_spec(inline_axis_spec(&n.data, grid.vertical)) {
                SpecLen::Pct(p) => p / 100.0 / cell.colspan.max(1) as f32,
                _ => 0.0,
            };
            let m = &n.metrics;
            // The pin is the border-box inline size, and the spanning
            // percentage pin below uses the content-box inline size
            // instead. For vertical tables the metrics hold the physical
            // (block-axis) width, so the `height` spec resolves against
            // the table content width (the inline base) directly.
            let (bb, cw) = if grid.vertical {
                let d = &n.data;
                let v = inline_axis_spec(d, true)
                    .resolve(
                        d.font_size,
                        forest.root_font_size,
                        forest.vp_w,
                        forest.vp_h,
                        content_w,
                    )
                    .unwrap_or(0.0)
                    .max(0.0);
                let bp = m.bt + m.pt + m.pb + m.bb;
                let bb = match d.box_sizing {
                    BoxSizing::BorderBox => v,
                    BoxSizing::ContentBox => v + bp,
                }
                .max(0.0);
                let cw = match d.box_sizing {
                    BoxSizing::BorderBox => (v - bp).max(0.0),
                    BoxSizing::ContentBox => v,
                };
                (bb, cw)
            } else {
                (
                    (m.content_w + m.bl + m.pl + m.pr + m.br).max(0.0),
                    m.content_w.max(0.0),
                )
            };
            let span = cell.colspan.max(1) as f32;
            // A spanning pin covers its columns' border boxes plus the
            // internal border-spacing gaps, so a length pin subtracts
            // those gaps before dividing: a 108px cell over 2 columns
            // with 8px spacing pins 50px per column, not 54 (WPT
            // `table-width-redistribution-fixed` table 12). A spanning
            // percentage pin instead resolves per column from the
            // percentage alone — CSS Tables 3 §3.9.3.1 has percentage
            // cells ignore border and padding — so the pin is the
            // padding-stripped content-width share (rebased to the
            // assignable width below); the cell then covers its columns
            // plus the internal gaps (WPT
            // `table-width-redistribution-fixed-padding` table 8).
            // Non-spanning percentage pins keep the border-box pin: the
            // rescale absorbs padding there (WPT
            // `table-width-redistribution-fixed-padding` table 11 pins
            // 63/42/31, not 80/53/39).
            let per = if cell_pct > 0.0 && cell.colspan > 1 {
                cw / span
            } else {
                (bb - spacing_h * (span - 1.0)).max(0.0) / span
            };
            // CSS `min-width` floors the column even when the
            // over-100% rescale below would shrink the pin past it:
            // record the border-box floor per spanned column here; it
            // is re-applied after the rescale multiply (the pin itself
            // already carries it via `resolve_metrics`, so the rescale
            // denominator is unchanged).
            let floor = cell_min_width_floor_bb(forest, cell.node, content_w, grid.vertical) / span;
            for cc in cell.col..(cell.col + cell.colspan).min(ncols) {
                if !col_set[cc] {
                    col_w[cc] = per;
                    col_set[cc] = true;
                    col_base[cc] = per;
                    col_pct[cc] = cell_pct;
                }
                if floor > col_minw[cc] {
                    col_minw[cc] = floor;
                }
            }
        }
        // Horizontal spacing under separate borders (zero when borders
        // collapse); columns + spacing tile the content width exactly.
        // Collapsed columns are removed from the display, so they draw
        // no spacing either. A table with no visible columns draws no
        // spacing at all (`nvis_cols == 0`): one phantom gap otherwise
        // survives the `(nvis + 1)` formula.
        let spacing_total = if collapse || nvis_cols == 0 {
            0.0
        } else {
            (nvis_cols as f32 + 1.0) * spacing_h
        };
        let avail = (content_w - spacing_total).max(0.0);
        // CSS Tables 3 §3.9.2.1 rule 3: percentage pins resolve against
        // the *assignable* width (content width minus border-spacing),
        // never against the spacing-inclusive content width. The pins
        // were resolved against `content_w` above, so rebase their
        // percentage-derived part onto `avail`: pin = pct * avail +
        // (padding + border). Percentage resolution is linear in the
        // base, so subtracting `pct * spacing_total` is exact for both
        // box-sizing modes (a 50% cell in a 132px fixed table with 8px
        // border-spacing pins at 50px of the 100px assignable width,
        // not 66px of the 132px content width). A group spread's
        // intrinsic share mirrors its rebased px width over the
        // assignable width, exactly as it mirrored the px spread over
        // the content width before.
        for cc in 0..ncols {
            if grid.collapsed_cols[cc] || col_pct[cc] <= 0.0 {
                continue;
            }
            let cut = col_pct[cc] * spacing_total;
            col_w[cc] = (col_w[cc] - cut).max(0.0);
            col_base[cc] = (col_base[cc] - cut).max(0.0);
            if col_spec.1[cc] {
                col_pct[cc] = if avail > 0.0 {
                    col_base[cc] / avail
                } else {
                    0.0
                };
            }
        }
        let set_sum: f32 = col_w.iter().sum();
        let n_open = col_set.iter().filter(|&&s| !s).count();
        let excess = avail - set_sum;
        if excess > 0.0 {
            // CSS Tables 3 §3.9.3.2 "distributing excess width to
            // columns": excess = assignable table width minus the
            // columns' base-assignment widths. The first rule with any
            // candidate columns takes the whole excess; collapsed
            // columns are never candidates and keep their zero width.
            let mut cands: Vec<usize> = Vec::new();
            let mut weights: Vec<f32> = Vec::new();
            if n_open > 0 {
                // Rule 1: columns with no specified width share the
                // excess equally.
                for (cc, &s) in col_set.iter().enumerate() {
                    if !s {
                        cands.push(cc);
                    }
                }
                share_excess_width(&mut col_w, &cands, &weights, excess);
            } else {
                // Rule 2: every column pinned; columns with a non-zero
                // length width take the excess proportionally to those
                // widths.
                for (cc, &gone) in grid.collapsed_cols.iter().enumerate() {
                    if gone || col_pct[cc] != 0.0 || col_base[cc] <= 0.0 {
                        continue;
                    }
                    cands.push(cc);
                    weights.push(col_base[cc]);
                }
                if cands.is_empty() {
                    // Rule 3: no length widths; columns with a non-zero
                    // percentage width take the excess proportionally to
                    // their percentage-derived widths from the base
                    // assignment (the rebased pins, padding absorbed),
                    // mirroring rule 2's width-proportional split — a
                    // padded 25%/15%/10% row rebased to 46/32.4/25.6
                    // splits a 32px excess 14.15/9.97/7.88 (WPT
                    // `table-width-redistribution-fixed-padding`
                    // table 12: 60/42.5/33), not 16/9.6/6.4. Unpadded
                    // pins are proportional to their percentages anyway,
                    // so the unpadded behavior is unchanged.
                    for (cc, &gone) in grid.collapsed_cols.iter().enumerate() {
                        if gone || col_pct[cc] <= 0.0 {
                            continue;
                        }
                        cands.push(cc);
                        weights.push(col_base[cc]);
                    }
                }
                if cands.is_empty() {
                    // Rule 4: every visible column pinned at zero width;
                    // equal shares.
                    for (cc, &gone) in grid.collapsed_cols.iter().enumerate() {
                        if !gone {
                            cands.push(cc);
                        }
                    }
                }
                share_excess_width(&mut col_w, &cands, &weights, excess);
            }
        } else if set_sum + spacing_total > content_w {
            // CSS Tables 3 §3.9.3.1: over-constrained percentage pins
            // resolve against the assignable width, and fixed-length
            // pins are never rescaled. When the percentage and length
            // pins together overflow the assignable width, the length
            // pins keep their width and the percentage pins absorb the
            // remainder (`assignable − fixed_total`), rescaling
            // proportionally among themselves — with no table growth
            // (20% + 60% + 60px pins in a 132px fixed table with 8px
            // border-spacing: the 60px pin keeps its width, the
            // percentages absorb 100 − 60 = 40px in a 20:60 ratio, so
            // the columns are 10/30/60 and the table stays 132px). The
            // percentage-only case is the same rule with
            // `fixed_total == 0` (two 60% columns in a 300px fixed table
            // with 10px border-spacing pin at 135px each and the table
            // stays 300px). Collapsed columns keep their zero width.
            // Anything still over-constrained afterwards (fixed lengths
            // alone, or border spacing, overflowing) takes the growth
            // path below unchanged (CSS 2.1 §17.5.2.1: the used table
            // width is the greater of the specified width and the
            // columns' sum, spacing included).
            let fixed_total: f32 = col_pct
                .iter()
                .enumerate()
                .filter(|&(cc, &p)| p <= 0.0 && col_set[cc] && !grid.collapsed_cols[cc])
                .map(|(cc, _)| col_w[cc])
                .sum();
            // The rescale denominator is the percentage pins' exact
            // current px — the `col_w` sum over the percentage pins —
            // not a fraction times a table width: the pins are
            // border-box, so a product omits their padding/border, the
            // scale overshoots, and the growth path below fires
            // spuriously (two padded 60% cells grew a 300px table to
            // 330px). CSS Tables 3 §3.9.3.1 fixed-mode says percentage
            // cells ignore border/padding; the pin model absorbs it
            // instead.
            let pct_px: f32 = col_pct
                .iter()
                .enumerate()
                .filter(|&(cc, &p)| p > 0.0 && !grid.collapsed_cols[cc])
                .map(|(cc, _)| col_w[cc])
                .sum();
            let set_sum = if pct_px > 0.0 && fixed_total < avail {
                let scale = (avail - fixed_total) / pct_px;
                for (cc, &p) in col_pct.iter().enumerate() {
                    if p > 0.0 && !grid.collapsed_cols[cc] {
                        // The multiply must not undo the `min-width`
                        // floor the pin already carries (via
                        // `resolve_metrics`): re-apply it here, so a
                        // floored pin stays floored instead of shrinking
                        // back under its minimum.
                        col_w[cc] = (col_w[cc] * scale).max(col_minw[cc]);
                    }
                }
                col_w.iter().sum()
            } else if pct_px > 0.0 {
                // The fixed pins alone meet or exceed the assignable
                // width: there is no remainder left for the percentage
                // pins, so their percentage part resolves to 0. Rebuild
                // each percentage pin from its non-percentage extras
                // only (padding/border past the percentage-sized part;
                // zero for an unpadded pin) instead of leaving it
                // full-size and growing the table by its whole width
                // (150px + 50% in a 100px assignable width grew the
                // table to 200px instead of 150px).
                for (cc, &p) in col_pct.iter().enumerate() {
                    if p > 0.0 && !grid.collapsed_cols[cc] {
                        col_w[cc] = (col_w[cc] - p * avail).max(0.0).max(col_minw[cc]);
                    }
                }
                col_w.iter().sum()
            } else {
                set_sum
            };
            if set_sum + spacing_total > content_w {
                // Over-constrained: the table grows so columns + spacing
                // tile the grown content width exactly (CSS 2.1
                // §17.5.2.1 — the used table width is the greater of the
                // specified width and the columns' sum, spacing
                // included). This covers the fixed-length
                // over-constrained case (e.g. two 150px pins in a 200px
                // table) and the degenerate spacing-starved case where
                // `border-spacing` alone meets or exceeds the content
                // width: the table grows to fit spacing + columns
                // instead of overflowing them. The table never grows
                // past its own `max-width`: the columns then overflow
                // the table instead.
                let delta = clamp_growth_delta(
                    forest,
                    idx,
                    input.cb_w,
                    set_sum + spacing_total - content_w,
                );
                if delta > 0.0 {
                    content_w += delta;
                    let t = &mut forest.nodes[idx];
                    t.metrics.content_w = content_w;
                    t.rect.w += delta;
                    // With a specified width and `auto` side margins the table
                    // was centered on the old width; grow symmetrically so it
                    // stays centered instead of drifting toward the inline end
                    // (with `width: auto` the margins computed to zero, so the
                    // table stays put).
                    let recenter = {
                        let d = &t.data;
                        !matches!(d.width, SpecLen::Auto)
                            && matches!(d.margin[3], SpecMargin::Auto)
                            && matches!(d.margin[1], SpecMargin::Auto)
                    };
                    if recenter {
                        let shift = delta / 2.0;
                        t.rect.x -= shift;
                        t.metrics.ml -= shift;
                        t.metrics.mr -= shift;
                        content_x -= shift;
                    }
                }
            }
        }

        // CSS 2.1 §17.5.2.2 errata rule 1: the used table width is the
        // greater of the specified width, the columns' sum (grown above
        // when over-constrained), and CAPMIN. A wide caption must grow
        // the table too (a 400px minimum caption on a `table {
        // table-layout: fixed; width: 100px }` leaves the table 100px
        // wide otherwise, with the caption overflowing). This runs after
        // the over-constrained growth — which may already have grown
        // past CAPMIN, making this a no-op — and before the cell/caption
        // tasks below, since captions lay out at `content_w`. The growth
        // honors the table's own `max-width` and recenters under `auto`
        // side margins, mirroring the paths above and the auto-layout
        // branch.
        let mut capmin_scratch = Vec::new();
        let capmin = table_caption_min_width(forest, idx, fonts, &mut capmin_scratch, pool);
        // CAPMIN is an outer (border-box plus margins) width, so it
        // bounds the table's border-box width, not its content box:
        // comparing against the content box overgrows a bordered table
        // (WPT `computing-table-width-0`: 50px + 2px borders + a 100px
        // caption must stay 100px wide, not 104px).
        let border_w = forest.nodes[idx].rect.w;
        if capmin > border_w {
            let delta = clamp_growth_delta(forest, idx, input.cb_w, capmin - border_w);
            if delta > 0.0 {
                content_w += delta;
                let t = &mut forest.nodes[idx];
                t.metrics.content_w = content_w;
                t.rect.w += delta;
                // With a specified width and `auto` side margins the table
                // was centered on the old width; grow symmetrically so it
                // stays centered instead of drifting toward the inline end
                // (with `width: auto` the margins computed to zero, so the
                // table stays put).
                let recenter = {
                    let d = &t.data;
                    !matches!(d.width, SpecLen::Auto)
                        && matches!(d.margin[3], SpecMargin::Auto)
                        && matches!(d.margin[1], SpecMargin::Auto)
                };
                if recenter {
                    let shift = delta / 2.0;
                    t.rect.x -= shift;
                    t.metrics.ml -= shift;
                    t.metrics.mr -= shift;
                    content_x -= shift;
                }
            }
        }
    } else {
        // CSS 2.1 §17.5.2.2 automatic table layout.
        //
        // Column minima come from [`auto_column_minima`]: the maximum
        // over single-column cells of (min-content width + padding +
        // border, specified cell width, `min-width` floor), raised by
        // `<col>`/`<colgroup>` specified widths and by spanning cells'
        // excess (step 3). Column maxima measure the same with preferred
        // (max-content) widths and never fall below the minima. A `col`
        // width floors the maximum too (step 2); a `colgroup` spread
        // floors the minimum only (step 4), so the spread flag from
        // [`column_specified_widths`] gates the maximum raise below.
        // One scratch triple serves the min/max-content cell measures
        // below (and the caption pre-pass further below); preferred
        // widths reuse the shared [`PrefWidthCache`].
        let mut mcw_scratch: Vec<f32> = Vec::new();
        let mut int_bufs = WalkBufs::new();
        let mut int_inner = WalkBufs::new();
        let (col_max, col_max_nonspanning) = auto_column_maxima(
            forest,
            &grid,
            &col_spec,
            spacing_h,
            fonts,
            pw_cache,
            &mut mcw_scratch,
            &mut int_bufs,
            &mut int_inner,
        );
        let col_min = auto_column_minima(
            forest,
            &grid,
            &col_spec,
            spacing_h,
            content_w,
            fonts,
            &col_max,
            &col_max_nonspanning,
        );
        // A spanning cell can raise a column's minimum past its
        // maximum (the §3.9.3.2 intrinsic distribution sends the whole
        // min excess to the non-constrained columns and the whole max
        // excess likewise, at different rates): the measures are
        // deliberately *not* repaired to max >= min here. The
        // sizing-guess interpolation below treats each guess on its
        // own terms, and a column whose minimum exceeds its maximum
        // simply contributes its minimum to the early guesses and its
        // maximum to the max-content guess (WPT
        // `colspan-redistribution` table 12: min 392 / max 342 on the
        // auto column, used width 342).

        // Distribute the table's content width across columns per the
        // §3.9.3 sizing-guess interpolation below. The widths always
        // tile exactly. When the minima exceed the available width the
        // table grows to their sum instead of squeezing columns below
        // min-content (the used table width is the greater of the
        // specified width and the columns' sum).
        let spacing_total = if collapse || nvis_cols == 0 {
            0.0
        } else {
            (nvis_cols as f32 + 1.0) * spacing_h
        };
        let avail = (content_w - spacing_total).max(0.0);
        let sum_min: f32 = col_min.iter().sum();
        // Strictly greater: at exact equality the sizing-guess
        // interpolation below decides (tied guess totals land on the
        // later, larger assignment rather than the minima).
        if sum_min > avail {
            col_w.copy_from_slice(&col_min);
            if sum_min + spacing_total > content_w {
                // Over-constrained: grow the table so columns + spacing
                // tile the grown content width exactly, mirroring the
                // fixed-layout path — including the degenerate
                // spacing-starved case where `border-spacing` alone meets
                // or exceeds the content width. Never past the table's
                // own `max-width`: the columns then overflow instead.
                let delta = clamp_growth_delta(
                    forest,
                    idx,
                    input.cb_w,
                    sum_min + spacing_total - content_w,
                );
                if delta > 0.0 {
                    content_w += delta;
                    let t = &mut forest.nodes[idx];
                    t.metrics.content_w = content_w;
                    t.rect.w += delta;
                    // With a specified width and `auto` side margins the table
                    // was centered on the old width; grow symmetrically so it
                    // stays centered instead of drifting toward the inline end
                    // (with `width: auto` the margins computed to zero, so the
                    // table stays put).
                    let recenter = {
                        let d = &t.data;
                        !matches!(d.width, SpecLen::Auto)
                            && matches!(d.margin[3], SpecMargin::Auto)
                            && matches!(d.margin[1], SpecMargin::Auto)
                    };
                    if recenter {
                        let shift = delta / 2.0;
                        t.rect.x -= shift;
                        t.metrics.ml -= shift;
                        t.metrics.mr -= shift;
                        content_x -= shift;
                    }
                }
            }
        } else {
            // CSS Tables 3 §3.9.3 distribution algorithm: build the
            // four sizing-guesses and interpolate. Each guess assigns
            // every column a width by category, in nondecreasing
            // order:
            //   1. min-content: every column at its min-content width.
            //   2. min-content-percentage: percentage columns at
            //      `max(intrinsic percentage * assignable width,
            //      min-content)`, the rest at min-content.
            //   3. min-content-specified: as (2), but constrained
            //      (specified-length) columns at their max-content.
            //   4. max-content: as (2), but every other column at its
            //      max-content.
            // When the assignable width falls between two consecutive
            // guess totals, the used widths are the linear combination
            // of the two guesses that bounds it; past the max-content
            // guess, the excess distributes per §3.9.3.2 (see
            // `distribute_excess_width`). Percentage assignments are
            // evaluated against the assignable width itself, so a
            // percentage column holds the same assignment in guesses
            // 2-4 and only moves during the 1→2 interpolation.
            let (pcts, _) = intrinsic_column_pcts(forest, &grid, &col_max_nonspanning);
            let constrained = column_constrained(forest, &grid);
            let pct_assign = |cc: usize| -> f32 {
                if pcts[cc] > 0.0 {
                    (pcts[cc] * avail).max(col_min[cc])
                } else {
                    col_min[cc]
                }
            };
            let mut guesses: [Vec<f32>; 4] = [
                col_min.clone(),
                vec![0.0; ncols],
                vec![0.0; ncols],
                vec![0.0; ncols],
            ];
            for cc in 0..ncols {
                guesses[1][cc] = pct_assign(cc);
                guesses[2][cc] = if pcts[cc] > 0.0 {
                    pct_assign(cc)
                } else if constrained[cc] {
                    col_max[cc]
                } else {
                    col_min[cc]
                };
                guesses[3][cc] = if pcts[cc] > 0.0 {
                    pct_assign(cc)
                } else {
                    col_max[cc]
                };
                // A `<colgroup>`-spread floor is hard (Chrome honors
                // it): a column whose width came from a `colgroup` (not
                // a `col`) never drops below its minimum in guesses 3
                // and 4, restoring per-column monotonicity. Floored
                // totals make the branch decision below use the floor
                // too, so interpolation honors it and the post-hoc
                // `floor_delta` loop in the excess branch becomes dead
                // code (removed). The spanning-induced min>max case
                // (`colspan-redistribution` table 12) has no group
                // flag, so it still lands on its maximum.
                if col_spec.1[cc] {
                    guesses[2][cc] = guesses[2][cc].max(col_min[cc]);
                    guesses[3][cc] = guesses[3][cc].max(col_min[cc]);
                }
            }
            let sums: [f32; 4] = guesses
                .iter()
                .map(|g| g.iter().sum())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            if avail <= sums[3] {
                // Interpolate within the segment that bounds the
                // assignable width, taking the *last* guess whose
                // total does not exceed it as the segment's start:
                // when consecutive guess totals tie (e.g. GRIDMIN ==
                // GRIDMAX, where every guess sums to the same total),
                // the columns land on the later, larger assignment —
                // each column should get its preferred width when the
                // table width allows it (WPT `colspan-redistribution`
                // table 12, whose minima and maxima sum equally).
                let lo = (0..4).rev().find(|&k| sums[k] <= avail).unwrap_or(0);
                if lo == 3 {
                    col_w.copy_from_slice(&guesses[3]);
                } else {
                    let (lo_g, hi_g) = (&guesses[lo], &guesses[lo + 1]);
                    let span = sums[lo + 1] - sums[lo];
                    let t = if span > 0.0 {
                        ((avail - sums[lo]) / span).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    for cc in 0..ncols {
                        // At the segment ends the interpolation is exact:
                        // `t * (hi - lo)` with an infinite max-content
                        // guess (a fixed-layout table with a percentage
                        // width has an infinite max-content intrinsic
                        // size) computes `0.0 * INF` = NaN at `t == 0`.
                        col_w[cc] = if t == 0.0 {
                            lo_g[cc]
                        } else if t == 1.0 {
                            hi_g[cc]
                        } else {
                            lo_g[cc] + t * (hi_g[cc] - lo_g[cc])
                        };
                    }
                }
            } else {
                // Past the max-content guess: start from it and
                // distribute the excess per §3.9.3.2 — first to
                // non-constrained columns (proportionally to
                // max-content, else equally), then to constrained
                // columns (proportionally to max-content), then to
                // percentage columns (proportionally to the
                // percentage), then equally. A specified cell or
                // column width constrains its column, so the extra
                // width lands on the unconstrained columns instead of
                // leaking into the constrained one (WPT
                // `td-min-width-auto-layout.html`). Collapsed columns
                // keep their zero width throughout.
                col_w.copy_from_slice(&guesses[3]);
                // The `<colgroup>`-spread hard floor is already baked
                // into the guesses above (floored at `col_min` for
                // group-flagged columns), so the excess here is the
                // assignable width past the floored max-content total —
                // the spanning-induced min>max case
                // (`colspan-redistribution` table 12) has no group
                // floor, so it still lands on its maximum.
                distribute_excess_width(
                    forest,
                    &grid,
                    &mut col_w,
                    &col_max,
                    (avail - sums[3]).max(0.0),
                    &pcts,
                );
            }
        }

        // CSS 2.1 §17.5.2.2 errata rule 1: the used table width is the
        // greater of the specified width, MIN, and CAPMIN. The column
        // distribution above grows the table past the specified width for
        // column MIN only; a wide caption must grow it too (a 400px
        // minimum caption on a `table { width: 100px }` leaves the table
        // 100px wide otherwise, with the caption overflowing). This runs
        // after the distribution — the over-constrained path may already
        // have grown past CAPMIN, making this a no-op — and before the
        // cell/caption tasks below, since captions lay out at
        // `content_w`. The growth honors the table's own `max-width` and
        // recenters under `auto` side margins, mirroring the fixed-layout
        // over-constrained path.
        let capmin = table_caption_min_width(forest, idx, fonts, &mut mcw_scratch, pool);
        // CAPMIN is an outer (border-box plus margins) width, so it
        // bounds the table's border-box width, not its content box:
        // comparing against the content box overgrows a bordered table
        // (WPT `computing-table-width-0`: 50px + 2px borders + a 100px
        // caption must stay 100px wide, not 104px).
        let border_w = forest.nodes[idx].rect.w;
        if capmin > border_w {
            let delta = clamp_growth_delta(forest, idx, input.cb_w, capmin - border_w);
            if delta > 0.0 {
                content_w += delta;
                let t = &mut forest.nodes[idx];
                t.metrics.content_w = content_w;
                t.rect.w += delta;
                // With a specified width and `auto` side margins the table
                // was centered on the old width; grow symmetrically so it
                // stays centered instead of drifting toward the inline end
                // (with `width: auto` the margins computed to zero, so the
                // table stays put).
                let recenter = {
                    let d = &t.data;
                    !matches!(d.width, SpecLen::Auto)
                        && matches!(d.margin[3], SpecMargin::Auto)
                        && matches!(d.margin[1], SpecMargin::Auto)
                };
                if recenter {
                    let shift = delta / 2.0;
                    t.rect.x -= shift;
                    t.metrics.ml -= shift;
                    t.metrics.mr -= shift;
                    content_x -= shift;
                }
            }
        }
    }

    // Collapsed-border winners per cell edge (CSS 2.1 §17.6.2.1), for
    // paint's shared grid lines; `None` unless borders collapse.
    let collapsed = resolve_collapsed_borders(forest, idx, &grid, content_w);

    // Absolute border-box x per column. A collapsed column takes no
    // space: it neither advances the cursor nor draws spacing (CSS 2.1
    // §17.5 — the column is removed from the display, so its neighbors
    // become adjacent).
    let mut col_x = vec![0.0f32; ncols];
    let mut x = content_x + if collapse { 0.0 } else { spacing_h };
    for (i, w) in col_w.iter().enumerate() {
        col_x[i] = x;
        if !grid.collapsed_cols[i] {
            x += *w + if collapse { 0.0 } else { spacing_h };
        }
    }

    // Row boxes span the full column run (first column x to the last
    // column's right edge). With no columns the rows still span the
    // table's inner width (csswg #10132; WPT
    // `table-rows-with-zero-columns`).
    let row_x = col_x.first().copied().unwrap_or(content_x);
    let row_w = if col_x.is_empty() {
        content_w
    } else {
        col_x.last().map(|&lx| lx - row_x).unwrap_or(0.0) + col_w.last().copied().unwrap_or(0.0)
    };

    // Split captions by `caption-side` (read on each caption element).
    // Deferred orthogonal captions (w77-d task 1: horizontal caption
    // in a vertical table with an auto block size) are held out of
    // these lists — and out of the first task push below — until the
    // table's block size is finalized by placement.
    let mut captions_top = Vec::new();
    let mut captions_bottom = Vec::new();
    let mut deferred_top = Vec::new();
    let mut deferred_bottom = Vec::new();
    let mut top_all = Vec::new();
    let mut bottom_all = Vec::new();
    for &cap in &grid.captions {
        let bottom = forest.nodes[cap]
            .style
            .as_ref()
            .and_then(|s| s.get_str_by_index(props::CAPTION_SIDE))
            .is_some_and(|v| v.eq_ignore_ascii_case("bottom"));
        if bottom {
            bottom_all.push(cap);
        } else {
            top_all.push(cap);
        }
        if is_deferred_orthogonal_caption(forest, cap, grid.vertical) {
            if bottom {
                deferred_bottom.push(cap);
            } else {
                deferred_top.push(cap);
            }
        } else if bottom {
            captions_bottom.push(cap);
        } else {
            captions_top.push(cap);
        }
    }

    // Collapsed-border winners per cell, for the cell tasks below: the
    // winning widths (centered on the shared grid lines) replace the
    // cell's own specified border widths in the block layout. Computed
    // before `collapsed` moves into the resume frame.
    let mut cell_wins: Vec<Option<[f32; 4]>> = vec![None; grid.cells.len()];
    if collapse && let Some(c) = collapsed.as_ref() {
        for (ci, w) in c.cells.iter().enumerate().take(cell_wins.len()) {
            cell_wins[ci] = Some([w[0].width, w[1].width, w[2].width, w[3].width]);
        }
    }

    // The resume runs after every cell/caption task (LIFO).
    tasks.push(Task::Table(Box::new(TableResume {
        idx,
        content_top,
        content_x,
        content_w,
        cb_h: input.cb_h,
        cb_w: input.cb_w,
        hspec,
        collapse,
        spacing_v,
        spacing_h,
        nrows: grid.nrows,
        cells: grid.cells.clone(),
        cell_block,
        collapsed_rows: grid.collapsed_rows.clone(),
        collapsed_cols: grid.collapsed_cols.clone(),
        collapsed,
        captions_top,
        captions_bottom,
        deferred_top,
        deferred_bottom,
        top_all,
        bottom_all,
        phase_caption: false,
        row_x,
        row_w,
        row_nodes: grid.row_nodes.clone(),
        row_group: grid.row_group.clone(),
        group_nodes: grid.group_nodes.clone(),
        floats: grid.floats.clone(),
        col_x: col_x.clone(),
        col_w: col_w.clone(),
        columns: grid.columns.clone(),
        phase2: false,
        phase_row_h: Vec::new(),
        phase_bshift: Vec::new(),
        phase_group_target: Vec::new(),
        phase2_cells: Vec::new(),
        stacked_block: 0.0,
        pre_float_block: 0.0,
    })));

    // Re-resolve each cell's padding/border against the finalized table
    // content width: the distribution above measured with a zero base
    // (intrinsic `%` padding is 0), but final layout resolves `%`
    // padding against the table's width (WPT `td-box-sizing-003`:
    // "final size of % padding is computed against table's width").
    // `cell_pct_pass_input` and the measurement dispatch below read
    // these metrics for the width override, so this must run before
    // them. Skipped cells feed no layout (see above). `table_w` is also
    // the `%` base for the cell tasks below.
    let table_w = forest.nodes[idx].metrics.content_w;
    for cell in &grid.cells {
        if cell_width_contributes(&grid.collapsed_cols, cell) {
            resolve_metrics(forest, cell.node, 0.0, table_w, false, None, false);
        }
    }

    // Cells, in reverse DOM order. Each lays out through the normal block
    // path with its spanned column width as a width override; the resume
    // pass shifts rows down once heights are known, so every cell starts
    // provisionally at the content top. Cells in a collapsed row, or
    // spanning only collapsed columns, take no space and never paint
    // (CSS 2.1 §17.5): their task is skipped outright — the zero rect
    // contributes no row height, shapes no text, and paints nothing —
    // unless the cell is measured for height (see
    // [`cell_measured_for_height`): it lays out so its content height
    // can be read, then is detached before paint.
    // (Rowspan cells that merely *span* a collapsed row/column still lay
    // out: they paint in their visible rows/columns.)
    for (ci, cell) in grid.cells.iter().enumerate().rev() {
        if !cell_laid_out(&grid.collapsed_rows, &grid.collapsed_cols, cell)
            && !cell_measured_for_height(&grid.collapsed_rows, &grid.collapsed_cols, cell)
        {
            // Stay zero-area: earlier passes may have stamped a width
            // onto the rect, and a skipped cell takes no space.
            forest.nodes[cell.node].rect = crate::Rect {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
            };
            continue;
        }
        // Under separate borders the spanned width includes the
        // inter-column spacing between the spanned columns (`spacing_h`
        // is 0 when borders collapse). Collapsed columns take no space
        // and draw no spacing (their neighbors become adjacent), so only
        // the visible spanned columns count — mirroring the `rowspan`
        // growth pass.
        let mut vis_w = 0.0f32;
        let mut nvis = 0usize;
        let span = cell.col..cell.col + cell.colspan;
        for (w, &gone) in col_w[span.clone()].iter().zip(&grid.collapsed_cols[span]) {
            if !gone {
                vis_w += w;
                nvis += 1;
            }
        }
        let span_w: f32 = vis_w + spacing_h * nvis.saturating_sub(1) as f32;
        let m = &forest.nodes[cell.node].metrics;
        // Under `border-collapse: collapse` the winning borders are
        // centered on the shared grid lines (CSS 2.1 §17.6.2.2): the
        // cell's content is inset by half the winning width on each side
        // (plus its own padding), not by the cell's own (possibly
        // losing) specified border widths. The winners were resolved
        // above; a missing entry falls back to the cell's own borders.
        let (win_l, win_r, wins) = match cell_wins[ci] {
            Some(w) => (w[3] / 2.0, w[1] / 2.0, Some(w)),
            None => (m.bl, m.br, None),
        };
        let content_w_cell = (span_w - (win_l + m.pl + m.pr + win_r)).max(0.0);
        let cb_x = col_x[cell.col];
        // Provisional position: `resolve_metrics` inside the block task
        // sets `rect.x` from `cb_x`; `rect.y` stays provisional until the
        // resume pass shifts the row down.
        forest.nodes[cell.node].rect.x = cb_x;
        forest.nodes[cell.node].rect.y = content_top;
        tasks.push(Task::Block(BlockInput {
            abs_pct_height_base: None,
            idx: cell.node,
            cb_x,
            // `%` padding/margins on the cell resolve against the
            // table's width, not the spanned column width (WPT
            // `td-box-sizing-003`).
            cb_w: table_w,
            cb_h: input.cb_h,
            shrink: false,
            width_override: Some(content_w_cell),
            width_override_final: false,
            height_override: None,
            // CSS 2.1 §17.5.3: a specified `height` on a cell is a
            // minimum — content taller than the specified height grows
            // the cell instead of overflowing it.
            height_is_min: true,
            collapsed_border_wins: wins,
            abspos: false,
            // Cells contain their floats: each cell gets a fresh float
            // list, so floats never cross a cell boundary.
            floats: fresh_floats(),
            is_float: false,
        }));
    }

    // Captions lay out full-width like normal blocks. Deferred
    // orthogonal captions (w77-d task 1) are held out here entirely:
    // they re-lay after placement against the finalized block size.
    for &cap in grid.captions.iter().rev() {
        if is_deferred_orthogonal_caption(forest, cap, grid.vertical) {
            continue;
        }
        let child_floats = if establishes_bfc(forest, cap) {
            fresh_floats()
        } else {
            input.floats.clone()
        };
        forest.nodes[cap].rect.y = content_top;
        // A vertical-writing-mode caption's inline size is its physical
        // height: resolve `height: min-content | max-content` against
        // the vertical intrinsic size instead of the horizontal line
        // stack (WPT `caption.html` table 13).
        let vwm_inline = vertical_caption_inline_size(forest, cap, fonts, pw_cache, grid.vertical);
        // Orthogonal (horizontal) captions in a vertical table never
        // reach this loop: they are all deferred (see
        // `is_deferred_orthogonal_caption`) and lay out in
        // `push_deferred_caption_tasks` against the finalized block
        // size. Only same-mode captions take a definite base here.
        // w77fix finding 2, same-mode definite base: a vertical
        // caption in a vertical table lays out against the definite
        // block content width (clamped by the table's block-axis
        // min/max specs, exactly as the orthogonal deferred base
        // clamps), so its fill-minus-margins border box is
        // definite-block-minus-margins. Without it the caption
        // filled the table's inline `content_w` instead, and the
        // override below (an auto-regime fix) then handed it that
        // inline width outright.
        let same_mode_fast_path = if grid.vertical
            && hspec.is_some()
            && matches!(
                forest.nodes[cap].data.writing_mode,
                WritingMode::VerticalLr | WritingMode::VerticalRl
            ) {
            hspec.map(|h| super::clamp_height(forest, idx, Some(input.cb_w), h))
        } else {
            None
        };
        let caption_cb_w = same_mode_fast_path.unwrap_or(content_w);
        // w77-d task 2: in a vertical table the caption's block-axis
        // extent must not shrink under its physical-horizontal margins
        // (those margins are added again by `place_table_rows`) — but
        // only in the auto regime; a definite-spec fast-path caption
        // keeps its margin box constrained to the specified block size
        // (see `caption_block_axis_width_override`). w77fix: the
        // same gate applies to same-mode captions (orthogonal
        // captions never reach this loop — they are deferred):
        // without the `hspec.is_none()` check the override fired
        // for definite same-mode captions too, handing them the
        // table's inline content width instead of
        // definite-block-minus-margins and growing the table past
        // its definite block size.
        let caption_width_override = if grid.vertical && hspec.is_none() {
            caption_block_axis_width_override(forest, cap, caption_cb_w)
        } else {
            None
        };
        tasks.push(Task::Block(BlockInput {
            abs_pct_height_base: None,
            idx: cap,
            cb_x: content_x,
            cb_w: caption_cb_w,
            cb_h: input.cb_h,
            shrink: false,
            width_override: caption_width_override,
            width_override_final: false,
            height_override: vwm_inline,
            height_is_min: false,
            collapsed_border_wins: None,
            abspos: false,
            floats: child_floats,
            is_float: false,
        }));
    }

    // Floated children lay out through the normal float machinery inside
    // the table's block formatting context (tables establish a BFC, CSS
    // 2.1 §9.4.1); the resume pass positions them once their sizes are
    // known. Provisional position is the content top; the resume pass
    // shifts the whole subtree via a deferred shift (like captions).
    for &f in grid.floats.iter().rev() {
        forest.nodes[f].rect.y = content_top;
        tasks.push(Task::Block(BlockInput {
            abs_pct_height_base: None,
            idx: f,
            cb_x: content_x,
            cb_w: content_w,
            cb_h: input.cb_h,
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
    }
    pool.put_grid(grid);
}

/// Whether a cell's `vertical-align` resolves to baseline alignment
/// (`sub`/`super` are meaningless on table cells and behave as
/// `baseline`, per CSS 2.1 §17.5.3's value list).
fn is_baseline_valign(forest: &Forest, idx: usize) -> bool {
    matches!(
        forest.nodes[idx].data.valign,
        VAlign::Baseline | VAlign::Sub | VAlign::Super
    )
}

/// Baseline of a cell for row baseline alignment, measured from the
/// cell's border-box top like [`Forest`] nodes' `baseline` field.
///
/// The block path stamps a block container's `baseline` as its
/// border-box height, which is right for a cell whose content is a line
/// box only in the sense that the inline path overwrites it with the
/// line baseline — but for a cell whose first in-flow content is a
/// block-level box, the border-box height is the *stretched* height
/// (the row height after the §3.10.6 pass, or the content height
/// before it), not the content's baseline. Per css-align-3's first
/// baseline set, such a cell's baseline is its first block child's
/// baseline: an empty block's bottom margin edge, a text block's line
/// baseline. Using the stretched height aligns every block-only cell
/// at the row bottom, so WPT `baseline-td` "Baselines with %ge
/// children" would place both % divs at the row top (0 / 0) instead
/// of aligning their bottoms (30 / 0). Cells whose first in-flow
/// child is not block-level (text lines, inline-blocks, nested tables,
/// flex/grid) keep the node's own `baseline`, which the inline path
/// already set correctly for them.
fn cell_row_baseline(forest: &Forest, cell_node: usize) -> f32 {
    // Only a `BoxType::Block` first child triggers the walk: text runs
    // are forest children too and carry `display: Block` by default,
    // but their baseline is the first line's, while the cell's own
    // field is the last line's — re-deriving through a text child
    // collapsed multi-line text-cell shifts to the first line
    // (regressed `table_cell_vertical_align_baseline` in w75e).
    // Anonymous blocks likewise keep the node's own baseline.
    //
    // The walk descends through block-level children, accumulating
    // deferred shifts, until a node with a real first baseline. A
    // plain block with only block children has no lines, so its
    // stamped baseline is its border-box bottom (`baseline ==
    // rect.h`) — trusting that intermediate stamp for cell > div >
    // div > text reported the outer div's bottom instead of the
    // text's baseline (w75f3). Only such lineless blocks are
    // descended through; a block whose baseline sits strictly above
    // its bottom edge has lines, so its baseline is real and the
    // walk stops there (matching what
    // [`super::first_inner_baseline`] returned at depth 1). Text
    // nodes are never descended into: a text node's own `baseline`
    // field is not a line baseline. An empty block on the descent
    // keeps its own baseline (its bottom edge, per the doc comment
    // above).
    let cell_top = forest.nodes[cell_node].rect.y;
    let cell_baseline = forest.nodes[cell_node].baseline;
    let mut cur = cell_node;
    let mut acc = 0.0f32;
    loop {
        // Skip whitespace-only children at every level: pretty-printed
        // HTML leaves newline/space text (or an anonymous block
        // wrapping it) around a cell's real first block child, and
        // that whitespace must not trigger the `cur == cell_node &&
        // !is_block` early return. WPT `baseline-td` subtest 2 has
        // exactly this shape (tds with whitespace around the % divs):
        // without the skip both cells report the stretched row height
        // as their baseline, so both divs sit at the row top (0 / 0)
        // instead of aligning their bottoms (30 / 0). Skipping only
        // whitespace-only content preserves the early return for
        // genuinely non-block first content (a text line or an
        // inline-block still returns the cell's own baseline) and
        // for genuinely empty cells (no non-whitespace child at all).
        let mut first: Option<usize> = None;
        for &c in &forest.nodes[cur].children {
            if layout_position::is_out_of_flow(forest, c) {
                continue;
            }
            if is_ws_only(forest, c) {
                continue;
            }
            first = Some(c);
            break;
        }
        let Some(f) = first else {
            if cur == cell_node {
                return cell_baseline;
            }
            return (forest.nodes[cur].rect.y + acc) - cell_top + forest.nodes[cur].baseline;
        };
        let is_block = forest.nodes[f].box_type == BoxType::Block
            && forest.nodes[f].data.display == Display::Block;
        if cur == cell_node && !is_block {
            return cell_baseline;
        }
        if is_block {
            let (fb, fh) = {
                let n = &forest.nodes[f];
                (n.baseline, n.rect.h)
            };
            // A plain block with lines stamps its last-line baseline,
            // which sits strictly above its border-box bottom
            // (descent + bottom padding/border intervene); a block
            // with only block children has no lines and stamps its
            // bottom edge (`baseline == rect.h`) instead — that
            // intermediate stamp is what the walk must not trust.
            // Descend only through such lineless blocks; a lineless
            // block with no in-flow children is terminal (empty).
            // Whitespace-only children do not count: a block whose
            // only children are whitespace is empty for baseline
            // purposes, so it keeps its own bottom-edge baseline
            // rather than descending into a whitespace text node.
            let has_kids = forest.nodes[f]
                .children
                .iter()
                .any(|&c| !layout_position::is_out_of_flow(forest, c) && !is_ws_only(forest, c));
            if fb >= fh && has_kids {
                acc += forest.nodes[f].pend_y;
                cur = f;
                continue;
            }
            return (forest.nodes[f].rect.y + acc + forest.nodes[f].pend_y) - cell_top + fb;
        }
        return (forest.nodes[f].rect.y + acc + forest.nodes[f].pend_y) - cell_top
            + forest.nodes[f].baseline;
    }
}

/// Add to a node's deferred shift. Unlike [`record_shift`] (which
/// overwrites), this composes with shifts already recorded on the node:
/// cell children may carry inline-block deferred shifts from their own
/// layout, and the final `apply_pending_shifts` pass accumulates parent
/// shifts down the tree.
fn add_shift(forest: &mut Forest, idx: usize, dx: f32, dy: f32) {
    let n = &mut forest.nodes[idx];
    n.pend_x += dx;
    n.pend_y += dy;
}

/// Classification of a resolved `height` on a row or row group for the
/// vertical distribution passes (CSS 2.1 §17.5.3, css-tables-3 §3.10).
#[derive(Clone, Copy, PartialEq)]
enum HeightSpec {
    /// No definite specified height: absorbs excess (unconstrained).
    Auto,
    /// Definite absolute length (or length-ish `calc()`).
    Fixed(f32),
    /// Definite percentage, resolved against its base (see
    /// [`row_height_spec`]).
    Pct(f32),
}

impl HeightSpec {
    fn value(self) -> Option<f32> {
        match self {
            HeightSpec::Auto => None,
            HeightSpec::Fixed(v) | HeightSpec::Pct(v) => Some(v),
        }
    }

    fn is_auto(self) -> bool {
        matches!(self, HeightSpec::Auto)
    }
}

/// Resolve a row's specified `height` in content-box terms (like
/// [`specified_height`]). Percentage rows resolve against their row
/// group's own specified height first, then the table's (`hspec`); with
/// no definite base they compute to `auto` (CSS 2.1 §10.5/§17.5.3). A
/// percentage that would exceed the table's specified height is capped
/// there (WPT `table-height-redistribution` tables 13/14: a 200% row in
/// a 100px table resolves to 100px, not 200px).
fn row_height_spec(forest: &Forest, rn: usize, base: Option<f32>, cap: Option<f32>) -> HeightSpec {
    let d = &forest.nodes[rn].data;
    if matches!(d.height, SpecLen::Auto) {
        return HeightSpec::Auto;
    }
    if pct_fraction(d.height).is_some() {
        let Some(b) = base else {
            return HeightSpec::Auto;
        };
        let mut v = specified_height(forest, rn, Some(b)).unwrap_or(0.0);
        if let Some(c) = cap {
            v = v.min(c);
        }
        return HeightSpec::Pct(v);
    }
    match specified_height(forest, rn, base) {
        Some(v) => HeightSpec::Fixed(v),
        None => HeightSpec::Auto,
    }
}

/// Resolve a row group's specified `height`. Percentage groups resolve
/// against the table's specified height and are capped at it (see
/// [`row_height_spec`]).
fn group_height_spec(forest: &Forest, g: usize, hspec: Option<f32>) -> Option<f32> {
    let v = specified_height(forest, g, hspec)?;
    if pct_fraction(forest.nodes[g].data.height).is_some() {
        Some(hspec.map_or(v, |h| v.min(h)))
    } else {
        Some(v)
    }
}

/// Distribute `extra` over `cands` (row indices) proportionally to their
/// current `row_h`; when every candidate is zero-height the extra splits
/// evenly. Non-positive `extra` (or no candidates) is a no-op.
fn share_proportional(row_h: &mut [f32], cands: &[usize], extra: f32) {
    if extra <= 0.0 || cands.is_empty() {
        return;
    }
    let total: f32 = cands.iter().map(|&r| row_h[r]).sum();
    if total > 0.0 {
        for &r in cands {
            row_h[r] += extra * row_h[r] / total;
        }
    } else {
        let each = extra / cands.len() as f32;
        for &r in cands {
            row_h[r] += each;
        }
    }
}

/// Whether a row is "sized" for excess distribution: it has a definite
/// specified height itself, or a cell originating in it does (a cell's
/// specified height constrains its row, per WPT
/// `table-height-redistribution` table 31).
fn row_is_sized(forest: &Forest, fr: &TableResume, r: usize, spec: HeightSpec) -> bool {
    if !spec.is_auto() {
        return true;
    }
    fr.cells.iter().any(|c| {
        c.row == r
            && cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, c)
            && !matches!(forest.nodes[c.node].data.height, SpecLen::Auto)
    })
}

/// Spread one spanning cell's height deficit over its (visible) spanned
/// rows (css-tables-3 row layout; order and priorities follow Chrome's
/// TablesNG as pinned by WPT `rowspan-height-redistribution`):
/// 1. percentage rows grow toward their percentage size (only when the
///    table height is definite enough to resolve them);
/// 2. rows originating *other* `rowspan` cells share the rest evenly;
/// 3. unconstrained non-empty rows grow proportionally to their height;
/// 4. when every spanned row is empty, the last one takes it all;
/// 5. otherwise constrained rows grow proportionally to their height.
///
/// `need` is the cell's full required height; the deficit is measured
/// against the rows' current total, counting inter-row spacing in the
/// separated model (the cell covers it too).
fn distribute_rowspan_excess(
    fr: &TableResume,
    row_h: &mut [f32],
    row_spec: &[HeightSpec],
    span_origins: &[usize],
    cell: &GridCell,
    vis: &[usize],
    need: f32,
) {
    let mut cur: f32 = vis.iter().map(|&r| row_h[r]).sum();
    if !fr.collapse {
        cur += fr.spacing_v * vis.len().saturating_sub(1) as f32;
    }
    let mut dh = need - cur;
    if dh <= 0.0 {
        return;
    }
    // 1. Percentage rows grow toward their percentage size.
    if fr.hspec.is_some() {
        let weights: Vec<(usize, f32)> = vis
            .iter()
            .copied()
            .filter_map(|r| {
                let target = match row_spec[r] {
                    HeightSpec::Pct(v) => v,
                    _ => return None,
                };
                let w = (target - row_h[r]).max(0.0);
                (w > 0.0).then_some((r, w))
            })
            .collect();
        let wsum: f32 = weights.iter().map(|&(_, w)| w).sum();
        if wsum > 0.0 {
            let mut given = 0.0;
            for (r, w) in weights {
                let add = (dh * w / wsum).min(w);
                row_h[r] += add;
                given += add;
            }
            dh -= given;
            if dh <= 0.0 {
                return;
            }
        }
    }
    // 2. Rows (other than the cell's own) originating `rowspan` cells
    // share the rest evenly. The cell's own row never counts, even when
    // a second spanning cell originates there too (WPT tables 10/11:
    // two spanning cells side by side must not dump the taller one's
    // deficit into their shared row).
    let origins: Vec<usize> = vis
        .iter()
        .copied()
        .filter(|&r| r != cell.row && span_origins[r] > 0)
        .collect();
    if !origins.is_empty() {
        let each = dh / origins.len() as f32;
        for &r in &origins {
            row_h[r] += each;
        }
        return;
    }
    let constrained = |r: usize| !row_spec[r].is_auto();
    // 3. Unconstrained non-empty rows grow proportionally to their height.
    let free: Vec<usize> = vis
        .iter()
        .copied()
        .filter(|&r| !constrained(r) && row_h[r] > 0.0)
        .collect();
    if !free.is_empty() {
        share_proportional(row_h, &free, dh);
        return;
    }
    // 4. Every spanned row is empty: the last one takes it all.
    if vis.iter().all(|&r| row_h[r] <= 0.0) {
        if let Some(&last) = vis.last() {
            row_h[last] += dh;
        }
        return;
    }
    // 5. Constrained rows grow proportionally to their height.
    let fixed: Vec<usize> = vis.iter().copied().filter(|&r| constrained(r)).collect();
    share_proportional(row_h, &fixed, dh);
}

/// Grid index (into `fr.group_nodes`) of the row group owning `gg`
/// (`None` when the node is not a known group).
fn group_index(fr: &TableResume, gg: usize) -> Option<usize> {
    fr.group_nodes.iter().position(|&x| x == gg)
}

/// The last grid row index (exclusive) a spanning cell may cover: its
/// `rowspan` clamped by the grid size, additionally clipped at the end
/// of its row-group section. The grid occupancy may extend past the
/// last collected row (phantom rows) and into later sections, but for
/// height purposes a span never crosses into the next section (HTML
/// table model; WPT `rowspan-height-redistribution` table 15). Phantom
/// rows (past `row_group`'s length) read as ungrouped, which keeps
/// spans inside a trailing bare section behaving as before.
fn span_end(fr: &TableResume, cell: &GridCell) -> usize {
    let end = (cell.row + cell.rowspan).min(fr.nrows);
    let g = fr.row_group.get(cell.row).copied().flatten();
    let mut last = cell.row;
    for r in (cell.row + 1)..end {
        if fr.row_group.get(r).copied().flatten() != g {
            break;
        }
        last = r;
    }
    last + 1
}

/// First grid row whose group sorts after group `gi` in DOM order: the
/// insertion point for a rowless group's absorbed height. Bare
/// (ungrouped) rows sort after every group.
fn rowless_insert_at(fr: &TableResume, gi: usize) -> usize {
    // Phantom rows from a `rowspan` overflowing its row group extend
    // `nrows` past `row_group`'s length; they read as ungrouped (like
    // in `span_end`), never as a later group.
    (0..fr.nrows)
        .find(|&r| match fr.row_group.get(r).copied().flatten() {
            Some(gg) => group_index(fr, gg).is_some_and(|p| p > gi),
            None => false,
        })
        .unwrap_or(fr.nrows)
}

/// Finish a table after its cells and captions laid out: resolve row
/// heights from cell content, stretch short cells, shift rows (and
/// captions) to their final positions, and set the table's height.
/// Whether a specified length depends on a percentage base (`height:
/// 50%`, or a `calc()` that syntactically contained a `%`).
fn spec_pct_dependent(s: SpecLen) -> bool {
    matches!(s, SpecLen::Pct(_)) || matches!(s, SpecLen::Math { had_pct: true, .. })
}

/// Whether any in-flow descendant of the cell at `idx` carries a
/// percentage-dependent `height`, `min-height`, or `max-height` — the
/// trigger for the css-tables-3 §3.10.6 second pass. Out-of-flow
/// subtrees are skipped: the position post-pass resolves their
/// percentages against final rects already.
fn cell_has_pct_height_content(forest: &Forest, idx: usize) -> bool {
    let mut stack: Vec<usize> = forest.nodes[idx].children.clone();
    while let Some(c) = stack.pop() {
        if layout_position::is_out_of_flow(forest, c) {
            continue;
        }
        let n = &forest.nodes[c];
        let d = &n.data;
        if spec_pct_dependent(d.height)
            || spec_pct_dependent(d.min_height)
            || spec_pct_dependent(d.max_height)
        {
            return true;
        }
        stack.extend(n.children.iter().copied());
    }
    false
}

/// Build the §3.10.6 second-pass task for one cell: the same geometry
/// as the measurement task, but with a definite `height_override` of
/// the cell's used content height (its spanned rows' height minus its
/// own border and padding), so percentage heights in its content now
/// resolve against it. Content taller than the used height overflows
/// the cell instead of growing it, per spec.
fn cell_pct_pass_input(
    forest: &mut Forest,
    fr: &TableResume,
    ci: usize,
    row_h: &[f32],
) -> BlockInput {
    let cell = &fr.cells[ci];
    // Spanned width from the columns' absolute extents (spacing
    // included), mirroring the measurement dispatch.
    let last = cell.col + cell.colspan - 1;
    let span_w = (fr.col_x[last] + fr.col_w[last] - fr.col_x[cell.col]).max(0.0);
    let wins = fr.collapsed.as_ref().and_then(|c| {
        c.cells
            .get(ci)
            .map(|w| [w[0].width, w[1].width, w[2].width, w[3].width])
    });
    let m = &forest.nodes[cell.node].metrics;
    let (win_l, win_r) = match wins {
        Some(w) => (w[3] / 2.0, w[1] / 2.0),
        None => (m.bl, m.br),
    };
    let content_w_cell = (span_w - (win_l + m.pl + m.pr + win_r)).max(0.0);
    // Used content height: the spanned rows' height minus the cell's
    // border and padding (the same terms `finish_content_height`
    // re-adds, so the re-laid border box lands exactly on the span).
    let span_h = if cell.rowspan == 1 {
        row_h[cell.row]
    } else {
        let end = span_end(fr, cell);
        let vis: Vec<usize> = (cell.row..end)
            .filter(|&rr| !fr.collapsed_rows.get(rr).copied().unwrap_or(false))
            .collect();
        let mut s: f32 = vis.iter().map(|&rr| row_h[rr]).sum();
        if !fr.collapse {
            s += fr.spacing_v * vis.len().saturating_sub(1) as f32;
        }
        s
    };
    let used_h = (span_h - (m.bt + m.pt + m.pb + m.bb)).max(0.0);
    let cb_x = fr.col_x[cell.col];
    // Re-layout recomputes provisional rects and re-records fresh
    // shifts; stale entries would apply twice (see
    // `super::clear_pending_shifts`).
    super::clear_pending_shifts(forest, cell.node);
    forest.nodes[cell.node].rect.x = cb_x;
    forest.nodes[cell.node].rect.y = fr.content_top;
    BlockInput {
        abs_pct_height_base: None,
        idx: cell.node,
        cb_x,
        // `%` padding/margins on the cell resolve against the table's
        // width, not the spanned column width (WPT `td-box-sizing-003`).
        cb_w: fr.content_w,
        cb_h: Some(used_h),
        shrink: false,
        width_override: Some(content_w_cell),
        width_override_final: false,
        height_override: Some(used_h),
        height_is_min: false,
        collapsed_border_wins: wins,
        abspos: false,
        floats: fresh_floats(),
        is_float: false,
    }
}

/// Per-cell baseline shifts from the cells' *current* baselines: for
/// each row, the shift that drops a `vertical-align: baseline` cell's
/// content so its baseline meets the row's greatest baseline (taken
/// from the `rowspan == 1` cells originating in the row, mirroring the
/// measurement pass in [`resume_table`]). Used by the §3.10.6
/// placement resume, where the measurement-pass shifts are stale for
/// rows whose cells were re-laid in the second pass (their content —
/// and baseline — changed when its percentage heights resolved).
fn recompute_baseline_shifts(forest: &Forest, fr: &TableResume) -> Vec<f32> {
    // Vertical carve-out (w75f3, item 8): baseline alignment in the
    // logical (inline, block) space of a vertical table is unmodeled
    // — exactly like [`table_inline_baseline`]'s vertical carve-out —
    // so vertical tables keep the node's own baseline here instead of
    // the horizontal first-baseline walk.
    let vertical = super::is_vertical_table(forest, fr.idx);
    let row_bl = |node: usize| -> f32 {
        if vertical {
            forest.nodes[node].baseline
        } else {
            cell_row_baseline(forest, node)
        }
    };
    let mut bshift = vec![0.0f32; fr.cells.len()];
    for r in 0..fr.nrows {
        let mut bl_max = 0.0f32;
        for cell in fr.cells.iter().filter(|c| c.row == r && c.rowspan == 1) {
            if !cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, cell) {
                continue;
            }
            if is_baseline_valign(forest, cell.node) {
                bl_max = bl_max.max(row_bl(cell.node));
            }
        }
        for (i, cell) in fr.cells.iter().enumerate().filter(|(_, c)| c.row == r) {
            if !cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, cell) {
                continue;
            }
            if !is_baseline_valign(forest, cell.node) {
                continue;
            }
            bshift[i] = (bl_max - row_bl(cell.node)).max(0.0);
        }
    }
    bshift
}

pub(crate) fn resume_table(forest: &mut Forest, tasks: &mut Vec<Task>, fr: TableResume) {
    // Final caption-placement resume (w77-d task 1): the table is
    // already placed and transposed to physical space, so this resume
    // only positions the deferred captions (and extends the table).
    // It must run before the forward transpose below, which would
    // otherwise move already-physical subtrees a second time.
    if fr.phase_caption {
        place_deferred_captions(forest, &fr);
        return;
    }
    // Vertical writing mode: move the laid-out cell/caption/float
    // subtrees into (inline, block) space and stamp each cell's block
    // content size into its rect.h, so the row-height machinery below
    // measures the block axis. Runs on every resume: a §3.10.6 second
    // pass re-lays cells back to physical first.
    if super::is_vertical_table(forest, fr.idx) {
        super::vwm_cells_to_logical(forest, &fr);
    }
    // The resume runs in (inline, block) space: the `min-width` floor
    // reads the transposed `min-height` for vertical tables.
    let vertical = super::is_vertical_table(forest, fr.idx);
    // A cell's border box is its spanned grid width exactly. When the
    // cell's padding/border exceed that width, the width override hands
    // the block path a zero content width and it re-adds the padding,
    // inflating the border box past the grid — a 0-width fixed column's
    // padded cell measured 12px instead of 0, and an 8.6px rescaled
    // column's cell measured 12px instead of 8.6 (WPT
    // `table-width-redistribution-fixed-padding` tables 6, 9 and 15
    // expect 0, 8 and 9). Clamp the laid-out border box back to the
    // spanned width here; an explicit `min-width` still floors it.
    // Content already laid out against a zero content width, so only
    // the box — background painting and `offsetWidth` — changes.
    for cell in fr.cells.iter() {
        if !cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, cell) {
            continue;
        }
        let node = cell.node;
        let mut vis_w = 0.0f32;
        let mut nvis = 0usize;
        let span = cell.col..cell.col + cell.colspan;
        for (w, &gone) in fr.col_w[span.clone()].iter().zip(&fr.collapsed_cols[span]) {
            if !gone {
                vis_w += w;
                nvis += 1;
            }
        }
        let span_w = vis_w + fr.spacing_h * nvis.saturating_sub(1) as f32;
        if forest.nodes[node].rect.w > span_w {
            let target = span_w.max(cell_min_width_floor_bb(
                forest,
                node,
                fr.content_w,
                vertical,
            ));
            if forest.nodes[node].rect.w > target {
                forest.nodes[node].rect.w = target;
            }
        }
    }
    // css-tables-3 §3.10.6 second pass: the measurement resume already
    // resolved the row heights and re-laid every cell whose content
    // carries percentage heights against its used height; only the
    // placement pass remains.
    if fr.phase2 {
        // B7: do not reuse the measurement-pass `phase_bshift` — the
        // second pass re-laid the percentage-height cells, so their
        // baselines (and their rows' greatest baselines) may have
        // moved. Recompute the shifts from the current baselines;
        // rows without a re-laid cell recompute to the same values.
        let bshift = recompute_baseline_shifts(forest, &fr);
        let (stacked, pre_float) = place_table_rows(
            forest,
            &fr,
            &fr.phase_row_h,
            &bshift,
            &fr.phase_group_target,
        );
        let mut fr = fr;
        fr.stacked_block = stacked;
        fr.pre_float_block = pre_float;
        push_deferred_caption_tasks(forest, tasks, fr);
        return;
    }

    // Hand the collapsed-border winners to paint: each cell's arena node
    // carries the edges it must draw ([top, right, bottom, left],
    // [`crate::LayoutBox::collapsed_borders`]). A cell owns its top and
    // left edges outright; its bottom/right edges only when it ends on
    // the table's last row/column — so every shared grid line is painted
    // exactly once, by the cell below/right of it. `fr.cells` is parallel
    // to `fr.collapsed.cells` (both follow the grid order).
    if let Some(collapsed) = fr.collapsed.as_ref() {
        // A collapsed row/column is removed from the display, so the
        // last *visible* row/column owns the table's outer bottom/right
        // edges: a cell above a collapsed last row still paints its
        // bottom border, while the collapsed row's own cells never paint
        // (their subtrees are skipped). With nothing collapsed the last
        // visible row/column are the grid's last, matching the old
        // `nrows`/`ncols` comparison exactly.
        let last_row = fr.collapsed_rows.iter().rposition(|&c| !c).unwrap_or(0);
        let last_col = fr.collapsed_cols.iter().rposition(|&c| !c).unwrap_or(0);
        for (cell, edges) in fr.cells.iter().zip(collapsed.cells.iter()) {
            // Cells that never paint — measured-only for height and
            // fully skipped cells (in collapsed rows / collapsed-only
            // columns) — must not carry border winners: paint has no
            // zero-area guard and would emit them at the origin.
            if !cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, cell) {
                forest.nodes[cell.node].collapsed_borders = None;
                continue;
            }
            let mut paint = *edges;
            if cell.row + cell.rowspan <= last_row {
                paint[2] = CollapsedBorder::NONE;
            }
            if cell.col + cell.colspan <= last_col {
                paint[1] = CollapsedBorder::NONE;
            }
            forest.nodes[cell.node].collapsed_borders = Some(Box::new(paint));
        }
    }

    // Row heights from the tallest cell in each row (border-box heights,
    // now that cell contents laid out). Cells measured for height
    // ([`cell_measured_for_height`]) contribute too: a collapsed row's
    // content height feeds the specified-table-height distribution
    // below (zeroed before placement), and a cell in fully-collapsed
    // columns still sizes its visible row. A laid-out cell spanning a
    // collapsed row never grows its visible rows (see
    // [`cell_height_contributes`]).
    let mut row_h = vec![0.0f32; fr.nrows];
    for cell in fr.cells.iter().filter(|c| c.rowspan == 1) {
        let measured = cell_measured_for_height(&fr.collapsed_rows, &fr.collapsed_cols, cell);
        if !measured {
            if !cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, cell) {
                continue;
            }
            if !cell_height_contributes(&fr.collapsed_rows, &fr.collapsed_cols, cell) {
                continue;
            }
        }
        let h = forest.nodes[cell.node].rect.h;
        if h > row_h[cell.row] {
            row_h[cell.row] = h;
        }
        // Measured-only cells never paint or place: detach them now
        // that their height is captured — zero rect, drop children and
        // fragments so paint sees the same empty box as a skipped cell.
        if measured {
            let n = &mut forest.nodes[cell.node];
            n.rect = crate::Rect {
                x: 0.0,
                y: 0.0,
                w: 0.0,
                h: 0.0,
            };
            n.children.clear();
            n.fragments.clear();
        }
    }
    // Specified heights on row groups and rows, resolved once: they
    // feed the row minimums below, the `rowspan` distribution's
    // constrained / percentage classification, and the row-group and
    // table excess passes (CSS 2.1 §17.5.3). A row's percentages
    // resolve against its group's specified height first, then the
    // table's.
    let group_spec: Vec<Option<f32>> = fr
        .group_nodes
        .iter()
        .map(|&g| group_height_spec(forest, g, fr.hspec))
        .collect();
    let row_spec: Vec<HeightSpec> = (0..fr.nrows)
        .map(|r| match fr.row_nodes.get(r).copied().flatten() {
            Some(rn) => {
                let base = fr
                    .row_group
                    .get(r)
                    .copied()
                    .flatten()
                    .and_then(|gg| group_index(&fr, gg))
                    .and_then(|gi| group_spec[gi])
                    .or(fr.hspec);
                row_height_spec(forest, rn, base, fr.hspec)
            }
            None => HeightSpec::Auto,
        })
        .collect();
    // A specified `height` on a row is a minimum for that row's height
    // (CSS 2.1 §17.5.3). The minimums land before the `rowspan`
    // distribution, so spanning cells spread over the constrained
    // sizes rather than the raw content sizes.
    for (r, spec) in row_spec.iter().enumerate() {
        if let Some(v) = spec.value() {
            row_h[r] = row_h[r].max(v);
        }
    }

    // `rowspan` cells grow the rows they span (see
    // [`distribute_rowspan_excess`] for the algorithm). Cells are
    // distributed in priority order: of two cells spanning the same
    // rows the taller goes first; a span fully enclosed by another is
    // distributed before it; otherwise the higher cell goes first.
    // Collapsed rows take no part (CSS 2.1 §17.5): a cell intersecting
    // a collapsed row never grows the visible rows it spans (see
    // [`cell_height_contributes`]).
    let mut span_origins = vec![0usize; fr.nrows];
    for cell in fr.cells.iter().filter(|c| c.rowspan > 1) {
        if cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, cell)
            && cell_height_contributes(&fr.collapsed_rows, &fr.collapsed_cols, cell)
        {
            span_origins[cell.row] += 1;
        }
    }
    let mut spans: Vec<&GridCell> = fr
        .cells
        .iter()
        .filter(|c| {
            c.rowspan > 1
                && cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, c)
                && cell_height_contributes(&fr.collapsed_rows, &fr.collapsed_cols, c)
        })
        .collect();
    spans.sort_by(|a, b| {
        use std::cmp::Ordering;
        let (sa, ea) = (a.row, span_end(&fr, a));
        let (sb, eb) = (b.row, span_end(&fr, b));
        let (ha, hb) = (forest.nodes[a.node].rect.h, forest.nodes[b.node].rect.h);
        if sa == sb && ea == eb {
            hb.partial_cmp(&ha).unwrap_or(Ordering::Equal)
        } else if sa >= sb && ea <= eb {
            Ordering::Less // `a` is enclosed by `b`: the inner cell goes first.
        } else if sb >= sa && eb <= ea {
            Ordering::Greater
        } else {
            sa.cmp(&sb)
                .then_with(|| hb.partial_cmp(&ha).unwrap_or(Ordering::Equal))
        }
    });
    for cell in spans {
        let end = span_end(&fr, cell);
        let vis: Vec<usize> = (cell.row..end)
            .filter(|&r| !fr.collapsed_rows.get(r).copied().unwrap_or(false))
            .collect();
        if vis.is_empty() {
            continue;
        }
        let h = forest.nodes[cell.node].rect.h;
        distribute_rowspan_excess(&fr, &mut row_h, &row_spec, &span_origins, cell, &vis, h);
    }

    // Cell `vertical-align`: baseline alignment first. Each row's
    // baseline is the greatest baseline among the
    // `vertical-align: baseline` cells *originating in it with
    // `rowspan == 1`* (baselines are measured from the cell's
    // content-box top and every cell starts at its row's top, so they
    // are comparable). A spanning cell's baseline belongs to the rows
    // it spans as a whole: it must not shift — and grow — its first
    // row (WPT `rowspan-height-redistribution`: a tall spanning cell
    // would otherwise dump its whole height into its originating row).
    // Each baseline cell's content shifts down by the difference,
    // growing the row when the shifted cell needs more room (like the
    // `rowspan` growth above). `top`/`middle`/`bottom` need no pre-pass:
    // they only distribute the slack left after the row height is
    // known. `sub`/`super` are meaningless on table cells and align as
    // `baseline`.
    // Vertical carve-out (w75f3, item 8), mirroring
    // [`recompute_baseline_shifts`]: vertical tables keep the node's
    // own baseline instead of the horizontal first-baseline walk.
    let row_bl = |node: usize| -> f32 {
        if vertical {
            forest.nodes[node].baseline
        } else {
            cell_row_baseline(forest, node)
        }
    };
    let mut bshift = vec![0.0f32; fr.cells.len()];
    for r in 0..fr.nrows {
        let mut bl_max = 0.0f32;
        for cell in fr.cells.iter().filter(|c| c.row == r && c.rowspan == 1) {
            // Never laid out: no baseline to contribute.
            if !cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, cell) {
                continue;
            }
            if is_baseline_valign(forest, cell.node) {
                bl_max = bl_max.max(row_bl(cell.node));
            }
        }
        for (i, cell) in fr.cells.iter().enumerate().filter(|(_, c)| c.row == r) {
            if !cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, cell) {
                continue;
            }
            if !is_baseline_valign(forest, cell.node) {
                continue;
            }
            let dy = (bl_max - row_bl(cell.node)).max(0.0);
            bshift[i] = dy;
            // The shift moves the cell's content down *within* the cell:
            // the row only grows when the shifted content overflows the
            // cell's border box. A specified `height` absorbs the shift
            // (CSS 2.1 §17.5.3: it is a minimum, and the baseline shift
            // is content positioning, not extra height) — WPT
            // css-tables/tentative/baseline-td.html "Baseline and td
            // height" expects 202px, not 200px + shift. The natural
            // content height (pre-minimum) sizes the shifted content;
            // paths that did not record it fall back to `content_h`.
            let need = {
                let n = &forest.nodes[cell.node];
                let m = &n.metrics;
                let natural = if n.natural_content_h.is_nan() {
                    n.content_h
                } else {
                    n.natural_content_h
                };
                let shifted_bottom = dy + natural + m.bt + m.pt + m.pb + m.bb;
                n.rect.h.max(shifted_bottom)
            };
            if cell.rowspan == 1 {
                if need > row_h[r] {
                    row_h[r] = need;
                }
            } else {
                // A cell intersecting a collapsed row never grows rows
                // (see [`cell_height_contributes`]).
                if !cell_height_contributes(&fr.collapsed_rows, &fr.collapsed_cols, cell) {
                    continue;
                }
                // Collapsed rows take no part in the spread (see the
                // `rowspan` growth above): the excess divides over the
                // visible spanned rows only.
                let end = span_end(&fr, cell);
                let vis: Vec<usize> = (cell.row..end)
                    .filter(|&r| !fr.collapsed_rows.get(r).copied().unwrap_or(false))
                    .collect();
                if vis.is_empty() {
                    continue;
                }
                distribute_rowspan_excess(
                    &fr,
                    &mut row_h,
                    &row_spec,
                    &span_origins,
                    cell,
                    &vis,
                    need,
                );
            }
        }
    }

    // A specified `height` on a row group is a minimum for the sum of
    // its rows' heights (CSS 2.1 §17.5.3). The group's excess over its
    // rows' natural heights distributes over the group's unconstrained
    // rows proportionally to their heights (evenly when all are
    // zero-height); when every row is constrained, the constrained rows
    // grow proportionally instead. Collapsed rows take no share, and a
    // fully-collapsed group's specified height is discarded with it.
    // `group_target` is each group's final height contribution to the
    // table's natural height; rowless groups (no rows at all) keep
    // theirs as a direct rect height.
    let vis_row = |r: usize| !fr.collapsed_rows.get(r).copied().unwrap_or(false);
    // B5: `row_is_sized` scans every cell per call and the distribution
    // passes below call it O(rows x units) times (~O(U*R*C) overall).
    // It depends only on `row_spec` and the cells' specified heights,
    // neither of which changes during distribution, so resolve it once
    // per row here and thread the flags through both passes.
    let row_sized: Vec<bool> = (0..fr.nrows)
        .map(|r| row_is_sized(forest, &fr, r, row_spec[r]))
        .collect();
    let mut group_target = vec![0.0f32; fr.group_nodes.len()];
    for (gi, &g) in fr.group_nodes.iter().enumerate() {
        let rows: Vec<usize> = fr
            .row_group
            .iter()
            .enumerate()
            .filter(|(r, grp)| *r < fr.nrows && **grp == Some(g))
            .map(|(r, _)| r)
            .collect();
        let vis: Vec<usize> = rows.iter().copied().filter(|&r| vis_row(r)).collect();
        let natural: f32 = vis.iter().map(|&r| row_h[r]).sum();
        if rows.is_empty() {
            group_target[gi] = group_spec[gi].unwrap_or(0.0);
            continue;
        }
        if vis.is_empty() {
            // Fully collapsed: the group's specified height never
            // applies (see the zeroing pass below).
            group_target[gi] = 0.0;
            continue;
        }
        let target = group_spec[gi].unwrap_or(0.0).max(natural);
        group_target[gi] = target;
        let excess = target - natural;
        if excess > 0.0 {
            let free: Vec<usize> = vis.iter().copied().filter(|&r| !row_sized[r]).collect();
            if free.is_empty() {
                share_proportional(&mut row_h, &vis, excess);
            } else {
                share_proportional(&mut row_h, &free, excess);
            }
            // The rows now sum to the target.
            group_target[gi] = vis.iter().map(|&r| row_h[r]).sum::<f32>().max(target);
        }
    }

    // A specified table height distributes its excess over the natural
    // content height (CSS 2.1 §17.5.3): row groups without a specified
    // height and with at least one unsized row absorb it, sharing
    // proportionally to their heights (evenly when all are zero-height);
    // within a group the share goes to its unsized rows (same rule).
    // When no group can absorb, the last group in grid order takes it
    // all, growing its rows proportionally (even when they are sized).
    // A specified height smaller than the natural height never shrinks
    // rows — the table grows past it instead (the height is a minimum).
    // Snapshot before the table-height distribution: a row that grows
    // here (and only here) has a definite used height from the table's
    // definite height, unlike a row grown by its own / its group's
    // specified height, which stays a minimum (WPT
    // `td-box-sizing-003` table 11). The snapshot feeds the §3.10.6
    // second-pass gate below.
    let row_h_before_table_excess = row_h.clone();
    if let Some(hspec) = fr.hspec {
        let mut caps: f32 = fr
            .captions_top
            .iter()
            .chain(fr.captions_bottom.iter())
            .map(|c| forest.nodes[*c].rect.h)
            .sum();
        // w77fix finding 7: deferred captions are absent from
        // `captions_top`/`captions_bottom`, but definite ones
        // still occupy their specified base in the stacked total
        // — without this reserve the specified excess
        // distributes to the rows, and the deferred caption then
        // stacks on top of the stretched rows, double-counting
        // the specified size (300 spec + 100 rows laid out as
        // 300 rows + 300 caption = 600, not 400). The reserve is
        // the clamped specified size per deferred caption: a
        // lower bound on its eventual extent (its base is
        // max(specified, stacked)), and it always suffices to
        // zero the excess, since the reserve alone meets the
        // specified size. Auto tables never reach this block
        // (`hspec` is `None`), so auto deferred captions are
        // unaffected.
        let ndef = (fr.deferred_top.len() + fr.deferred_bottom.len()) as f32;
        if ndef > 0.0 {
            caps += ndef * super::clamp_height(forest, fr.idx, Some(fr.cb_w), hspec);
        }
        let bare: Vec<usize> = (0..fr.nrows)
            .filter(|&r| fr.row_group.get(r).copied().flatten().is_none() && vis_row(r))
            .collect();
        let bare_sum: f32 = bare.iter().map(|&r| row_h[r]).sum();
        let mut natural = caps + bare_sum;
        for t in &group_target {
            natural += t;
        }
        if !fr.collapse {
            // Collapsed rows draw no spacing (CSS 2.1 §17.5).
            let nvis = fr.collapsed_rows.iter().filter(|&&c| !c).count();
            natural += fr.spacing_v * (nvis + 1) as f32;
        }
        // Collapsed rows' measured content heights count toward the
        // natural height for the excess computation (css-tables-3
        // §3.10.5 treats collapsed rows as auto-height rows with
        // content-based reference sizes); their distribution shares
        // vanish in the zeroing pass below (WPT
        // `visibility-collapse-row-004`). `bare_sum` and `group_target`
        // above are visible-only, so there is no double count.
        natural += (0..fr.nrows)
            .filter(|&r| fr.collapsed_rows.get(r).copied().unwrap_or(false))
            .map(|r| row_h[r])
            .sum::<f32>();
        let excess = hspec - natural;
        if excess > 0.0 {
            // Row-group units in grid order; bare rows form one trailing
            // unit; rowless groups participate through their target.
            let mut targets: Vec<f32> = group_target.clone();
            targets.push(bare_sum);
            let unit_rows = |u: usize| -> Vec<usize> {
                if u < fr.group_nodes.len() {
                    let g = fr.group_nodes[u];
                    fr.row_group
                        .iter()
                        .enumerate()
                        .filter(|(r, grp)| *r < fr.nrows && **grp == Some(g) && vis_row(*r))
                        .map(|(r, _)| r)
                        .collect()
                } else {
                    bare.clone()
                }
            };
            // B5: memoize each unit's (visible) row list once — the
            // absorber scan, the fallbacks, and the share loop below
            // each re-derived it per unit before (~O(U^2 * R) row-group
            // scans on top of the per-row cell scans).
            let all_unit_rows: Vec<Vec<usize>> = (0..targets.len()).map(unit_rows).collect();
            // Wide row lists for the share loop: collapsed rows take
            // their share of the excess (it vanishes in the zeroing
            // pass), so visible rows don't absorb it (WPT
            // `visibility-collapse-row-004`). Absorption and fallback
            // decisions above stay visible-only.
            let wide_unit_rows = |u: usize| -> Vec<usize> {
                if u < fr.group_nodes.len() {
                    let g = fr.group_nodes[u];
                    fr.row_group
                        .iter()
                        .enumerate()
                        .filter(|(r, grp)| *r < fr.nrows && **grp == Some(g))
                        .map(|(r, _)| r)
                        .collect()
                } else {
                    (0..fr.nrows)
                        .filter(|&r| fr.row_group.get(r).copied().flatten().is_none())
                        .collect()
                }
            };
            let all_unit_rows_wide: Vec<Vec<usize>> =
                (0..targets.len()).map(wide_unit_rows).collect();
            let unit_sized = |u: usize| -> bool {
                if u < fr.group_nodes.len() {
                    group_spec[u].is_some()
                } else {
                    false
                }
            };
            // Whether a group unit has any rows at all, visible or
            // not. Distinct from `all_unit_rows` (visible rows only): a
            // fully-collapsed group has rows but none visible, and it
            // is *not* rowless — its rows are zeroed and never placed,
            // so height absorbed "into its own rect" would vanish.
            let unit_has_any_row: Vec<bool> = (0..targets.len())
                .map(|u| {
                    if u < fr.group_nodes.len() {
                        let g = fr.group_nodes[u];
                        fr.row_group.contains(&Some(g))
                    } else {
                        false
                    }
                })
                .collect();
            let unit_absorbs = |u: usize| -> bool {
                if unit_sized(u) {
                    return false;
                }
                if u < fr.group_nodes.len() && !unit_has_any_row[u] {
                    // A rowless group (no rows at all) with no specified
                    // height absorbs into its own rect.
                    return true;
                }
                all_unit_rows[u].iter().any(|&r| !row_sized[r])
            };
            let absorbers: Vec<usize> = (0..targets.len()).filter(|&u| unit_absorbs(u)).collect();
            let mut share = vec![0.0f32; targets.len()];
            if !absorbers.is_empty() {
                let total: f32 = absorbers.iter().map(|&u| targets[u]).sum();
                if total > 0.0 {
                    for &u in &absorbers {
                        share[u] = excess * targets[u] / total;
                    }
                } else {
                    let each = excess / absorbers.len() as f32;
                    for &u in &absorbers {
                        share[u] = each;
                    }
                }
            } else if let Some(last) = (0..targets.len()).rfind(|&u| !all_unit_rows[u].is_empty()) {
                // Nobody can absorb: the last rowed unit takes it all.
                share[last] = excess;
            } else if let Some(last) = (0..fr.group_nodes.len()).rfind(|&u| !unit_has_any_row[u]) {
                // Only rowless groups (no rows at all — a
                // fully-collapsed group does not qualify): the last
                // one takes it all.
                share[last] = excess;
            }
            for (u, s) in share.iter().enumerate() {
                if *s <= 0.0 {
                    continue;
                }
                if u < fr.group_nodes.len() {
                    let rows: &[usize] = &all_unit_rows_wide[u];
                    if rows.is_empty() {
                        group_target[u] += s;
                        continue;
                    }
                    if unit_absorbs(u) {
                        let free: Vec<usize> =
                            rows.iter().copied().filter(|&r| !row_sized[r]).collect();
                        share_proportional(&mut row_h, &free, *s);
                    } else {
                        share_proportional(&mut row_h, rows, *s);
                    }
                } else {
                    let wide_bare: Vec<usize> = (0..fr.nrows)
                        .filter(|&r| fr.row_group.get(r).copied().flatten().is_none())
                        .collect();
                    let free: Vec<usize> = wide_bare
                        .iter()
                        .copied()
                        .filter(|&r| !row_sized[r])
                        .collect();
                    if free.is_empty() {
                        share_proportional(&mut row_h, &wide_bare, *s);
                    } else {
                        share_proportional(&mut row_h, &free, *s);
                    }
                }
            }
        }
    }

    // CSS 2.1 §17.5: collapsed rows take no height. Zero them after
    // every height resolution above — a specified row/row-group height
    // never applies to a collapsed row, and excess a `rowspan` cell
    // spread onto a collapsed row is discarded with it (the cell never
    // paints) — and before rows are placed below.
    for (r, rh) in row_h.iter_mut().enumerate() {
        if fr.collapsed_rows.get(r).copied().unwrap_or(false) {
            *rh = 0.0;
        }
    }

    // css-tables-3 §3.10.6: percentage heights inside a cell were
    // indefinite during the measurement pass; now that row heights are
    // final, every cell whose content carries them is re-laid against
    // its used height (a definite `height_override`), and placement is
    // deferred to a second resume carrying the resolved heights —
    // mirroring flex's staged resume. Cells without percentage-height
    // descendants place immediately, as before.
    //
    // A row grown by the table's definite-height excess distribution
    // above also makes its (rowspan == 1) cells' used heights definite:
    // WPT `baseline-td` "Baselines with %ge children" (table
    // height:100px) expects 50% / 80% children at 50 / 80 with
    // baseline offsets 30 / 0, while a row grown only by its own or
    // its group's specified height stays indefinite (WPT
    // `td-box-sizing-003` table 11 expects the child at 0) — the
    // snapshot above is what separates the two.
    //
    // Two extensions (w75f3, item 7):
    // - Exact equality in a single-row table is definite too: with
    //   one row and a definite table height, the row's height IS the
    //   table's height even when the excess is 0 and the row did not
    //   "grow" (the old strict `>` gate missed this case).
    // - A spanning cell is definite when EVERY row it spans grew
    //   from table excess: the span's used height is then the sum of
    //   definite row heights. Growth of only some spanned rows does
    //   not make the span definite.
    //
    // Remaining incompleteness, stated plainly: a multi-row table
    // with excess 0 stays indefinite (its rows are natural, not
    // table-definite), and a span touching a collapsed row never
    // passes (a collapsed row's height is zeroed, so it cannot
    // "grow" under the strict comparison).
    let table_excess_grew: Vec<bool> = (0..fr.nrows)
        .map(|r| {
            !fr.collapsed_rows.get(r).copied().unwrap_or(false)
                && row_h[r] > row_h_before_table_excess[r]
        })
        .collect();
    let single_row_definite = fr.hspec.is_some() && fr.nrows == 1;
    let pass2: Vec<usize> = fr
        .cells
        .iter()
        .enumerate()
        .filter(|(_, cell)| {
            // Only a cell with its own specified height resolves its
            // content's percentages in the second pass, plus (w75e) a
            // rowspan == 1 cell in a row grown by definite table-height
            // excess: a height that comes from the row alone leaves
            // them indefinite (WPT `td-box-sizing-003` table 11
            // expects the child at 0).
            let span_all_grew = cell.rowspan > 1 && {
                let end = span_end(&fr, cell);
                end > cell.row
                    && (cell.row..end).all(|r| table_excess_grew.get(r).copied().unwrap_or(false))
            };
            cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, cell)
                && (specified_height(forest, cell.node, fr.cb_h).is_some()
                    || (cell.rowspan == 1
                        && (table_excess_grew.get(cell.row).copied().unwrap_or(false)
                            || (single_row_definite
                                && row_h.get(cell.row).copied().unwrap_or(0.0)
                                    >= row_h_before_table_excess
                                        .get(cell.row)
                                        .copied()
                                        .unwrap_or(0.0))))
                    || span_all_grew)
                && cell_has_pct_height_content(forest, cell.node)
        })
        .map(|(i, _)| i)
        .collect();
    if pass2.is_empty() {
        let (stacked, pre_float) = place_table_rows(forest, &fr, &row_h, &bshift, &group_target);
        let mut fr = fr;
        fr.stacked_block = stacked;
        fr.pre_float_block = pre_float;
        push_deferred_caption_tasks(forest, tasks, fr);
        return;
    }
    let inputs: Vec<BlockInput> = pass2
        .iter()
        .map(|&i| cell_pct_pass_input(forest, &fr, i, &row_h))
        .collect();
    let mut fr2 = fr;
    fr2.phase2 = true;
    fr2.phase_row_h = row_h;
    fr2.phase_bshift = bshift;
    fr2.phase_group_target = group_target;
    fr2.phase2_cells = pass2;
    // The placement resume goes on the stack first so the re-layout
    // tasks pushed after it run before it (LIFO).
    tasks.push(Task::Table(Box::new(fr2)));
    for input in inputs {
        tasks.push(Task::Block(input));
    }
}

/// Whether the table at `idx` is sized purely by border-spacing phantom
/// gaps, per dimension: `(phantom_w, phantom_h)`.
///
/// Semantics:
/// - the table is *cell-less*: no DOM-backed table-cell box anywhere in
///   its subtree. Anonymous cells from stray-content fixup have no
///   element and do not count, while nested tables' cells do count
///   (`display: none` elements generate no boxes and are not counted —
///   a `<table><tr><td style="display:none">x</td></tr></table>` is
///   cell-less and stamps, pinned by `phantom_display_none_cells_stamp`).
/// - width dimension: specified `width` is `auto`, no caption box was
///   generated, the used horizontal borders are ~zero, and the laid-out
///   physical width fits inside a single border-spacing;
/// - height dimension: specified `height` is `auto`, at least one
///   DOM-backed row exists, and the laid-out physical height fits inside
///   `v_spacing * (rows + 1)`.
///
/// The two dimensions stamp independently. The old single-bool OR
/// (`zero_w || zero_h`) lost which dimension was phantom, and the parse
/// shim's per-dimension re-derivation diverged from the engine's walks
/// (caption bleed: a caption exempts only the width; min-width bleed: a
/// min-width floor keeps the width real while the height stays phantom;
/// redistributed-height bleed: a container-stretched height stays real
/// while the width stays phantom; rows==0 bleed: no rows means no
/// height phantom). Stamping both — computed here with full context —
/// lets every consumer read the flags directly instead of re-deriving
/// them.
///
/// `spacing_h` / `spacing_v` are the [`TableResume`] algorithm-space
/// spacings (transposed for vertical tables); they are untransposed back
/// to physical here so the comparisons match the shim, which reads the
/// physical style values. Must run after `transpose_vwm_table`: the rect
/// comparisons are physical, like the final `LayoutBox` rect the shim
/// reads.
///
/// Only tables that reach the table engine are stamped: a *truly*
/// childless table takes `layout_block`'s empty-block early return and
/// never lays out here. The parse shim carries a minimal fallback for
/// those (auto width, no used borders, width inside one spacing);
/// without it their phantom width would read raw.
pub(crate) fn table_sized_by_phantom(
    forest: &Forest,
    idx: usize,
    spacing_h: f32,
    spacing_v: f32,
) -> (bool, bool) {
    // Cell-less: count DOM-backed cells/rows in the subtree (nested
    // tables included, mirroring the shim's `doc.descendants` walk).
    let mut cells = 0u32;
    let mut rows = 0u32;
    let mut stack = vec![idx];
    while let Some(n) = stack.pop() {
        for &c in forest.nodes[n].children.iter() {
            match forest.nodes[c].box_type {
                BoxType::TableCell if forest.nodes[c].dom.is_some() => cells += 1,
                BoxType::TableRow if forest.nodes[c].dom.is_some() => rows += 1,
                _ => {}
            }
            stack.push(c);
        }
    }
    if cells > 0 {
        return (false, false);
    }
    let n = &forest.nodes[idx];
    // A `<caption>` exempts the width only when it generated a box:
    // caption boxes nest under the table's own box.
    let has_caption = n.children.iter().any(|&c| is_table_caption(forest, c));
    // Physical spacings: the resume carries algorithm-space values
    // (transposed for vertical tables); untranspose to match the shim.
    // A vertical table's physical-width phantom is the vertical spacing
    // (mirroring the shim's writing-mode switch on the width check).
    let vertical = super::is_vertical_table(forest, idx);
    let h_spacing = if vertical { spacing_v } else { spacing_h };
    let v_spacing = if vertical { spacing_h } else { spacing_v };
    let w_spacing = if vertical { v_spacing } else { h_spacing };
    let width_auto = matches!(n.data.width, SpecLen::Auto);
    let height_auto = matches!(n.data.height, SpecLen::Auto);
    // Used, not specified, horizontal borders: `metrics` holds the used
    // widths here, so collapsed-table winners agree with the geometry
    // the reads use (mirroring the shim's `h_borders` argument).
    let h_borders = n.metrics.bl + n.metrics.br;
    let zero_w = width_auto && !has_caption && h_borders <= 0.001 && n.rect.w <= w_spacing + 0.001;
    let zero_h = height_auto && rows > 0 && n.rect.h <= v_spacing * (rows as f32 + 1.0) + 0.001;
    (zero_w, zero_h)
}

/// Inline-table baseline from the first row, mirroring Blink's
/// `LayoutTableSection::FirstLineBoxBaseline` and css-align-3's
/// "table rows" baseline-set rules. Runs after `transpose_vwm_table`,
/// so all rects are physical; vertical tables keep the historical
/// `first_inner_baseline` approximation (their inline axis is vertical
/// and baseline alignment there is unmodeled).
///
/// - No rows: the border-box bottom (historical behavior).
/// - First row with baseline-aligned originating cells
///   (`vertical-align: baseline | sub | super`; spanning cells count
///   for the row their span starts in): the greatest cell baseline,
///   measured from the row's top. A cell only counts when its baseline
///   clears its own border+padding, so empty cells never set the row's
///   baseline (Blink's `UpdateBaselineForCell` guard).
/// - First row with cells but none baseline-aligned: the lowest cell
///   content-bottom (Blink's fallback in `FirstLineBoxBaseline`).
/// - Empty first row: the row's block-start content edge (css-align-3
///   step 5; csswg-drafts #10137: Blink and Gecko both place it at the
///   top of the first row's content area).
fn table_inline_baseline(forest: &Forest, fr: &TableResume, idx: usize) -> f32 {
    if super::is_vertical_table(forest, idx) {
        return super::first_inner_baseline(forest, idx);
    }
    let table_top = forest.nodes[idx].rect.y;
    let border_bottom = forest.nodes[idx].rect.h;
    // First row in grid (DOM) order.
    let mut first_row: Option<usize> = None;
    for (r, rn) in fr.row_nodes.iter().enumerate() {
        if r >= fr.nrows {
            break;
        }
        if rn.is_some() {
            first_row = Some(r);
            break;
        }
    }
    let Some(r) = first_row else {
        return border_bottom;
    };
    let row_node = match fr.row_nodes[r] {
        Some(rn) => rn,
        None => return border_bottom,
    };
    // Row rects are absolute (assigned directly, never via pending
    // shifts); cells carry their placement as a pending shift.
    let row_top_rel = forest.nodes[row_node].rect.y - table_top;
    let mut row_bl: Option<f32> = None;
    let mut has_cells = false;
    let mut fallback = f32::NEG_INFINITY;
    // The fallback (lowest content-bottom) over spanning cells too, for
    // rows whose every cell spans: a rowspan cell's content-bottom spans
    // rows below, so it misrepresents the first row's baseline, but it
    // beats returning -inf.
    let mut fallback_spanning = f32::NEG_INFINITY;
    for cell in fr.cells.iter().filter(|c| c.row == r) {
        if !cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, cell) {
            continue;
        }
        has_cells = true;
        let cn = &forest.nodes[cell.node];
        let cell_top_rel = (cn.rect.y + cn.pend_y) - table_top;
        let m = &cn.metrics;
        if is_baseline_valign(forest, cell.node) && cn.baseline > m.bt + m.pt {
            // Same first-baseline rule as the row sites (w75f3): a
            // block-first cell's baseline is its first block child's
            // first baseline ([`cell_row_baseline`]), not the cell
            // node's own field (0 / last-line for block-first cells).
            // The `cn.baseline > bt + pt` empty-cell guard is
            // unchanged: it still reads the node's own baseline.
            let v = (cell_top_rel - row_top_rel) + cell_row_baseline(forest, cell.node);
            row_bl = Some(row_bl.map_or(v, |b: f32| b.max(v)));
        }
        let content_bottom = {
            // The rendered content-bottom: an empty cell stretched by its
            // row still contributes its laid-out content box (WPT
            // css-tables/tentative/baseline-table "Empty rowspanned
            // cells should not set baseline" needs the 44px-tall empty
            // td to yield 55, not its intrinsic 0).
            let m = &cn.metrics;
            let content_h = (cn.rect.h - (m.bt + m.bb + m.pt + m.pb)).max(0.0);
            cell_top_rel + m.bt + m.pt + content_h
        };
        if cell.rowspan == 1 {
            fallback = fallback.max(content_bottom);
        } else {
            // A spanning cell's content-bottom reaches into rows below;
            // it must not set the first row's baseline.
            fallback_spanning = fallback_spanning.max(content_bottom);
        }
    }
    if let Some(rb) = row_bl {
        return row_top_rel + rb;
    }
    if has_cells {
        // Prefer the non-spanning fallback; the spanning one only when
        // every first-row cell spans.
        return if fallback.is_finite() {
            fallback
        } else {
            fallback_spanning
        };
    }
    row_top_rel
}

/// Pend-x shift for a horizontal caption in a vertical table, and the
/// used values of its auto block margins (wave 76, task 2).
///
/// The caption's own block axis (physical vertical for a horizontal
/// caption) runs along the table's inline axis, which is the algorithm
/// x direction in `place_table_rows`. Its block margins (`mt`/`mb`)
/// therefore offset it along algorithm x, not along the block cursor:
/// the caption's provisional algorithm x carries no `mt` (the caption
/// task presets its physical y to the content top and `resolve_metrics`
/// maps auto block margins to 0), so the shift is exactly the used
/// block-start margin, and the used values are stamped into
/// `metrics.mt`/`mb` for the computed-style half.
///
/// Free space is the table's inline content width minus the caption's
/// extent along that axis (its algorithm `rect.w`, i.e. its physical
/// block border extent) minus the fixed margin on the non-auto side,
/// clamped at 0: both auto split it evenly, one auto takes all of it.
/// A caption with no auto block margin still shifts by its fixed `mt`,
/// which the provisional position likewise omits. Any other
/// caption/table combination (horizontal tables, same-writing-mode
/// captions, vertical captions) keeps a zero shift: their block margins
/// run along the table's block axis and are handled by the caption
/// loops' block-start margin below.
fn orthogonal_caption_inline_shift(
    forest: &mut Forest,
    cap: usize,
    table_writing_mode: WritingMode,
    content_w: f32,
) -> f32 {
    // In `place_table_rows` the caption is in (inline, block) space,
    // so its inline extent is its algorithm `rect.w`.
    let extent = forest.nodes[cap].rect.w;
    orthogonal_caption_inline_shift_for_extent(forest, cap, table_writing_mode, content_w, extent)
}

/// The inline-shift distribution with the caption's inline extent
/// supplied explicitly: `place_table_rows` passes the algorithm extent,
/// while [`place_deferred_captions`] passes the physical height (the
/// deferred caption is already in physical space there). See the doc
/// on [`orthogonal_caption_inline_shift`] for the semantics.
fn orthogonal_caption_inline_shift_for_extent(
    forest: &mut Forest,
    cap: usize,
    table_writing_mode: WritingMode,
    content_w: f32,
    extent: f32,
) -> f32 {
    if !matches!(
        table_writing_mode,
        WritingMode::VerticalLr | WritingMode::VerticalRl
    ) {
        return 0.0;
    }
    let (mt_auto, mb_auto, mt, mb) = {
        let n = &forest.nodes[cap];
        if !matches!(n.data.writing_mode, WritingMode::HorizontalTb) {
            return 0.0;
        }
        (
            matches!(n.data.margin[0], SpecMargin::Auto),
            matches!(n.data.margin[2], SpecMargin::Auto),
            n.metrics.mt,
            n.metrics.mb,
        )
    };
    if !mt_auto && !mb_auto {
        return mt;
    }
    let fixed = if mt_auto && mb_auto {
        0.0
    } else if mt_auto {
        mb
    } else {
        mt
    };
    let free = (content_w - extent - fixed).max(0.0);
    let (used_mt, used_mb) = if mt_auto && mb_auto {
        (free / 2.0, free / 2.0)
    } else if mt_auto {
        (free, mb)
    } else {
        (mt, free)
    };
    let n = &mut forest.nodes[cap];
    n.metrics.mt = used_mt;
    n.metrics.mb = used_mb;
    used_mt
}

/// A caption's block-axis margin-box extent from its physical rect and
/// metrics, for a vertical table: `(block-start margin, block-end
/// margin, border width)`. Vertical-rl's block-start is the physical
/// right margin; vertical-lr's is the left.
fn caption_block_extent(forest: &Forest, cap: usize, rl: bool) -> (f32, f32, f32) {
    let n = &forest.nodes[cap];
    let (mbs, mbe) = if rl {
        (n.metrics.mr, n.metrics.ml)
    } else {
        (n.metrics.ml, n.metrics.mr)
    };
    (mbs, mbe, n.rect.w)
}

/// After placement, re-lay the deferred orthogonal captions against
/// the table's finalized block size and schedule their placement
/// (w77-d task 1). Runs at the end of both placement paths in
/// [`resume_table`] (direct and §3.10.6 phase2); a no-op when the
/// table has no deferred captions.
///
/// State at this point: `place_table_rows` has run and
/// `transpose_vwm_table` has moved everything to physical space, so
/// the table node's `metrics.content_w` holds the finalized block
/// content size and `content_h` holds the inline content size (the
/// transpose swapped them). The caption's percentage margins/padding
/// resolve against the inline size ([`super::resolve_metrics`]'
/// orthogonal-caption base reads the parent's `metrics.content_w`),
/// so the metrics are swapped back to algorithm state for the caption
/// tasks and swapped forward again — with the extension applied — by
/// [`place_deferred_captions`].
///
/// Deferred captions lay out in a fresh float context: the first-push
/// float list is not carried on the resume frame, and captions that
/// establish a BFC already use a fresh list there.
fn push_deferred_caption_tasks(forest: &mut Forest, tasks: &mut Vec<Task>, fr: TableResume) {
    if fr.deferred_top.is_empty() && fr.deferred_bottom.is_empty() {
        return;
    }
    let idx = fr.idx;
    // Finalized block content size, read before the swap-back below.
    let block_content = forest.nodes[idx].metrics.content_w;
    {
        let n = &mut forest.nodes[idx];
        std::mem::swap(&mut n.content_h, &mut n.metrics.content_w);
    }
    let deferred: Vec<usize> = fr
        .deferred_top
        .iter()
        .chain(fr.deferred_bottom.iter())
        .copied()
        .collect();
    // w77fix finding 7: definite-spec captions defer too, so
    // their base is max(clamped specified size, stacked grid
    // extent) — the specified size when the grid is smaller
    // (the old fast path's base, exactly), the grid when it
    // grows past it. The stacked extent (`fr.stacked_block`:
    // rows plus non-deferred captions) is the right comparator,
    // not the finalized content size, which already carries the
    // specified floor and would double-count it. The auto-regime
    // width override does NOT apply to them: it would hand the
    // caption the base outright, breaking the definite
    // margin-box constraint (300 base + 40 margins = 260 border
    // box, pinned by the w75/w76 tests); without it the caption
    // fills the base minus its margins, as the fast path did.
    let definite = fr.hspec.is_some();
    let deferred_cb_w = match fr.hspec {
        Some(spec) => super::clamp_height(forest, idx, Some(fr.cb_w), spec).max(fr.stacked_block),
        None => block_content,
    };
    let (content_x, content_top, cb_h) = (fr.content_x, fr.content_top, fr.cb_h);
    let mut final_fr = fr;
    final_fr.phase_caption = true;
    // Pushed first so the caption tasks below (LIFO) run before it.
    tasks.push(Task::Table(Box::new(final_fr)));
    for &cap in deferred.iter().rev() {
        forest.nodes[cap].rect.y = content_top;
        // Auto-regime width override (w77-d task 2): the caption's
        // block-axis extent must not shrink under its horizontal
        // margins; here the fill base is the finalized block size.
        // Definite captions skip it (see above).
        let width_override = if definite {
            None
        } else {
            caption_block_axis_width_override(forest, cap, deferred_cb_w)
        };
        tasks.push(Task::Block(BlockInput {
            abs_pct_height_base: None,
            idx: cap,
            cb_x: content_x,
            cb_w: deferred_cb_w,
            cb_h,
            shrink: false,
            width_override,
            width_override_final: false,
            // Deferred captions are horizontal captions in vertical
            // tables, for which `vertical_caption_inline_size` is
            // always `None`.
            height_override: None,
            height_is_min: false,
            collapsed_border_wins: None,
            abspos: false,
            floats: fresh_floats(),
            is_float: false,
        }));
    }
}

/// Position the deferred orthogonal captions after they re-laid out
/// against the finalized block size (w77-d task 1). Runs from the
/// final caption-placement resume, when every table-internal rect is
/// already in physical space (post-transpose) — so all positioning
/// here is physical, unlike `place_table_rows`' algorithm-space
/// placement.
///
/// Block offsets run from the block-start content edge toward the
/// block-end: captions stack in merged document order per side (see
/// `top_all`/`bottom_all`), with non-deferred captions re-placed to
/// their merged positions; the already placed rows/cells/floats shift
/// toward the block-end by the top deferred captions' extent only.
/// The table's border box extends rightward by the extension (its
/// left edge is parent-anchored in both directions; for vertical-rl
/// the block-start edge is the post-extension right content edge),
/// and its block content size grows by the same amount.
///
/// Inline positions come from the caption's block margins along the
/// table's inline axis (physical vertical): the same distribution
/// `place_table_rows` applies via [`orthogonal_caption_inline_shift`],
/// but against the re-laid physical height — which is what makes an
/// auto block margin re-center when re-wrapping changed the height
/// (task 1b).
///
/// w77fix finding 14: the rows' block-end edge excludes float
/// extension. `place_table_rows` maxes its cursor with float
/// bottoms (the table's BFC contains its floats), so the
/// finalized block content size and `stacked_block` both carry
/// any float extension past the rows — but floats do not push
/// normal flow, so bottom deferred captions sit after the rows
/// (`pre_float_block`), not after the floats: the anchor
/// subtracts the float excess (`stacked_block` minus
/// `pre_float_block`), and the extension covers only the part
/// of the deferred extents the float-extended size does not
/// already contain (plus the top shift, which moves the floats
/// themselves). With no floats the excess is zero and both
/// formulas reduce to the historical ones.
fn place_deferred_captions(forest: &mut Forest, fr: &TableResume) {
    let idx = fr.idx;
    let table_wm = forest.nodes[idx].data.writing_mode;
    if !matches!(table_wm, WritingMode::VerticalLr | WritingMode::VerticalRl) {
        return;
    }
    let rl = matches!(table_wm, WritingMode::VerticalRl);
    // Block-start content edge in physical x: for vertical-lr it is
    // the content origin; for vertical-rl it is the right content
    // edge, mirroring `place_table_rows`' cursor anchor (right border
    // + padding + scrollbar-gutter reservation).
    //
    // w77fix finding 7: the table's position in its parent is
    // anchored at its left edge (the parent's writing mode
    // governs the table box's own position, not the table's
    // internal direction), so the table's `x` never moves here:
    // the border box extends rightward by the extension and the
    // block-start edge for vertical-rl is the POST-extension
    // right content edge. The old code anchored the rl edge at
    // its pre-extension value and shifted `x` left by the
    // extension, translating the whole table left by the rows'
    // extent whenever a definite caption deferred (w76 rl
    // anchor test: table at -38.4 instead of 0). The deferred
    // extents and the extension are computable up front (the
    // captions already laid out), so the edge uses them.
    let (old_x, old_w) = {
        let n = &forest.nodes[idx];
        (n.rect.x, n.rect.w)
    };
    let (br, pr) = {
        let m = &forest.nodes[idx].metrics;
        (m.br, m.pr)
    };
    let gr = if forest.gutter_propagated_root == Some(idx) {
        0.0
    } else if let Some(style) = forest.nodes[idx].style.as_ref() {
        let (_, gr) = crate::scrollbar_gutter_reservation(style);
        let (_, fr_) = crate::forced_scrollbar_reservation(style);
        gr.max(fr_)
    } else {
        0.0
    };
    // State A (swapped back by `push_deferred_caption_tasks`):
    // `content_h` is the finalized block content size.
    let block_content_first = forest.nodes[idx].content_h;
    let deferred_extent_of = |forest: &Forest, caps: &[usize]| -> f32 {
        caps.iter()
            .map(|&cap| {
                let (mbs, mbe, w) = caption_block_extent(forest, cap, rl);
                mbs + w + mbe
            })
            .sum()
    };
    let e_top_pre = deferred_extent_of(forest, &fr.deferred_top);
    let e_bottom_pre = deferred_extent_of(forest, &fr.deferred_bottom);
    let e_total_pre = e_top_pre + e_bottom_pre;
    // Float excess over the rows' true block-end edge (w77fix
    // finding 14): zero when no float extends past the rows.
    let float_excess = (fr.stacked_block - fr.pre_float_block).max(0.0);
    // Extension, as applied below (definite tables: only the part
    // the specified floor does not already cover — see below;
    // auto tables: only the part the float-extended size does
    // not already cover, but at least the top shift, which moves
    // the floats themselves toward the block-end).
    let extension_pre = if fr.hspec.is_some() {
        (fr.stacked_block + e_total_pre - block_content_first).max(0.0)
    } else {
        e_top_pre.max(e_total_pre - float_excess)
    };
    let block_start_edge = if rl {
        old_x + old_w + extension_pre - br - pr - gr
    } else {
        fr.content_x
    };
    let bottom_nondeferred: f32 = fr
        .captions_bottom
        .iter()
        .map(|&cap| {
            let (mbs, mbe, w) = caption_block_extent(forest, cap, rl);
            mbs + w + mbe
        })
        .sum();
    // w77fix finding 7: for definite tables the finalized content
    // size already carries the specified floor, which is NOT part
    // of the stacked rows — bottom captions stack after the
    // stacked extent (rows plus non-deferred captions), and the
    // table extends only by the part of the deferred extents the
    // floor does not already cover. Auto tables keep the
    // historical math (there the finalized size IS the stacked
    // extent, modulo min/max clamping).
    let rows_end_offset = if fr.hspec.is_some() {
        fr.stacked_block - bottom_nondeferred
    } else {
        // w77fix finding 14: the finalized size carries float
        // extension; subtract it so bottom captions anchor after
        // the rows, not after the floats. A min-height clamp
        // still belongs to the rows (it grows them, §17.5.3
        // excess distribution), so it stays in the anchor via
        // `block_content_first`; only the float part is removed.
        block_content_first - bottom_nondeferred - float_excess
    };
    // Stack the merged DOM-order lists: every caption on a side is
    // (re-)placed by record_shift — deferred captions for the first
    // time, non-deferred ones overwritten to their merged position
    // (identical to the first placement except for deferred captions
    // inserted before them, whose inline shift recomputes to the same
    // values since their heights and the inline content are
    // unchanged). Only deferred extents shift existing content or
    // extend the table: non-deferred extents were already counted by
    // the first placement. Returns (total extent, deferred extent).
    let place_list =
        |forest: &mut Forest, caps: &[usize], deferred: &[usize], mut cursor: f32| -> (f32, f32) {
            let start = cursor;
            let mut deferred_extent = 0.0f32;
            for &cap in caps {
                let (mbs, mbe, w) = caption_block_extent(forest, cap, rl);
                let border_offset = cursor + mbs;
                let border_x = if rl {
                    block_start_edge - border_offset - w
                } else {
                    block_start_edge + border_offset
                };
                // Same-mode captions (w77fix finding 9): their
                // inline position is fixed by first placement
                // (`orthogonal_caption_inline_shift` returns 0.0
                // for them, so their inline shift is zero and
                // their provisional position IS the placed one) —
                // re-placement here moves them only along the
                // block axis (merged order), so preserve the
                // provisional inline position (zero shift) instead
                // of forcing `content_top`, which would discard
                // any inline-start offset first placement gave
                // them.
                let (prov_x, prov_y) = {
                    let n = &forest.nodes[cap];
                    (n.rect.x, n.rect.y)
                };
                let inline = if matches!(
                    forest.nodes[cap].data.writing_mode,
                    WritingMode::HorizontalTb
                ) {
                    orthogonal_caption_inline_shift_for_extent(
                        forest,
                        cap,
                        table_wm,
                        fr.content_w,
                        forest.nodes[cap].rect.h,
                    )
                } else {
                    prov_y - fr.content_top
                };
                record_shift(
                    forest,
                    cap,
                    border_x - prov_x,
                    fr.content_top + inline - prov_y,
                );
                cursor += mbs + w + mbe;
                if deferred.contains(&cap) {
                    deferred_extent += mbs + w + mbe;
                }
            }
            (cursor - start, deferred_extent)
        };
    let (_top_total, e_top) = place_list(forest, &fr.top_all, &fr.deferred_top, 0.0);
    let (_bottom_total, e_bottom) = place_list(
        forest,
        &fr.bottom_all,
        &fr.deferred_bottom,
        rows_end_offset + e_top,
    );
    // Shift already-placed rows/cells/floats toward the block-end by
    // the top deferred captions' extent: pending-shift nodes
    // accumulate, direct rects (rows, groups, columns) move outright.
    // Captions are not shifted here — they were (re-)placed above.
    // For vertical-rl the table's left edge is parent-anchored (see
    // above), so the shift is the top deferred extent minus the
    // rightward growth: rows were placed against the pre-extension
    // right edge and must land after the top captions against the
    // post-extension one.
    if e_top != 0.0 || (rl && extension_pre != 0.0) {
        let dx = if rl { extension_pre - e_top } else { e_top };
        for cell in fr.cells.iter() {
            if cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, cell) {
                add_shift(forest, cell.node, dx, 0.0);
            }
        }
        for &f in &fr.floats {
            add_shift(forest, f, dx, 0.0);
        }
        for rn in fr.row_nodes.iter().flatten() {
            forest.nodes[*rn].rect.x += dx;
        }
        for &g in &fr.group_nodes {
            forest.nodes[g].rect.x += dx;
        }
        for spec in &fr.columns {
            for node in spec.col_node.into_iter().chain(spec.group_node) {
                forest.nodes[node].rect.x += dx;
            }
        }
    }
    // Extend the table toward the block-end and swap the metrics back
    // to physical state (block content in `metrics.content_w`). For
    // definite tables the extension is only the uncovered part (see
    // above): the floor already reserves up to the specified size.
    // The table's `x` never moves (see above): the border box grows
    // rightward in both directions.
    let e_total = e_top + e_bottom;
    let extension = extension_pre;
    {
        let n = &mut forest.nodes[idx];
        n.rect.w += extension;
        n.content_h += extension;
        // The natural total is the stacked extent plus all deferred
        // extents; for definite tables the floor-covered part must
        // not inflate it a second time (see above). For auto
        // tables the extension already excludes the
        // float-covered part (w77fix finding 14), so it applies
        // to the natural total as well.
        if fr.hspec.is_some() {
            n.natural_content_h = fr.stacked_block + e_total;
        } else {
            n.natural_content_h += extension;
        }
        std::mem::swap(&mut n.content_h, &mut n.metrics.content_w);
    }
}

/// Place rows, cells, captions, floats, and the row / row-group /
/// column rects from resolved heights, and finish the table box. Runs
/// either directly at the end of the measurement resume or, when a
/// §3.10.6 percentage second pass intervened, from the placement
/// resume with the carried heights.
fn place_table_rows(
    forest: &mut Forest,
    fr: &TableResume,
    row_h: &[f32],
    bshift: &[f32],
    group_target: &[f32],
) -> (f32, f32) {
    let idx = fr.idx;
    // Place rows top to bottom. Cells were laid out provisionally at the
    // content top; the deferred shift moves each cell subtree once (the
    // single `apply_pending_shifts` pass flushes it). Short cells
    // stretch their border box to the row height while `vertical-align`
    // positions their content within it.
    //
    // Cursor anchor (w76fix finding 5): for vertical tables the
    // algorithm cursor starts at the BLOCK-start content edge, not
    // the physical content top. The old `content_top` start was exact
    // only when the top border+padding equaled the block-start
    // border+padding (vertical-lr: `bt + pt` vs `bl + pl`, mirrored
    // for rl) — otherwise every caption, row, and float shifted by
    // the difference. The block-start edge is `content_x`'s offset
    // from the border-box left edge for lr (the prologue's
    // gutter-reserved kids origin), and `br + pr` plus the right
    // gutter reservation for rl (mirroring the block prologue's
    // propagated-root zeroing and per-side max). Horizontal tables
    // keep `content_top`, so this is a no-op there. The broader
    // asymmetric-border/padding transpose limitation documented on
    // `transpose_vwm_table` is pre-existing for the cell transpose
    // itself; this anchor removes it for the caption/row/float
    // cursor specifically.
    let cursor_start = match forest.nodes[idx].data.writing_mode {
        WritingMode::VerticalLr => {
            let n = &forest.nodes[idx];
            n.rect.y + (fr.content_x - n.rect.x)
        }
        WritingMode::VerticalRl => {
            let n = &forest.nodes[idx];
            let gr = if forest.gutter_propagated_root == Some(idx) {
                0.0
            } else if let Some(style) = n.style.as_ref() {
                let (_, gr) = crate::scrollbar_gutter_reservation(style);
                let (_, fr_) = crate::forced_scrollbar_reservation(style);
                gr.max(fr_)
            } else {
                0.0
            };
            n.rect.y + n.metrics.br + n.metrics.pr + gr
        }
        _ => fr.content_top,
    };
    let mut y = cursor_start;
    // Captions stack with their margins (CSS 2.1 §17.5.3): inter-caption
    // margins do not collapse (WPT `caption`: two 20px-margin captions
    // sit 40px apart, border-box to border-box). Negative margins are
    // honored as-is: a negative top margin pulls the caption up.
    for &cap in &fr.captions_top {
        // In a vertical writing mode the table's block axis is
        // physical-horizontal, so captions stack against the physical
        // left/right margins (metrics stay physical): block-start is
        // the right margin under `vertical-rl`, the left under
        // `vertical-lr`.
        // Follow-up (not fixed here): a caption's `width: auto`
        // resolves against the transposed content width, yielding a
        // 0-width caption in vertical writing modes — a pipeline
        // ordering problem (the caption is measured before the table's
        // inline size is known).
        //
        // Orthogonal captions do not reach this loop with a
        // wrong `cb_w`: they are deferred (w77-d task 1, extended
        // by w77fix finding 7 to definite block specs) and placed
        // by `place_deferred_captions` after they re-lay against
        // the finalized block size.
        let (mbs, mbe, h) = {
            let n = &forest.nodes[cap];
            match forest.nodes[idx].data.writing_mode {
                WritingMode::VerticalRl => (n.metrics.mr, n.metrics.ml, n.rect.h),
                WritingMode::VerticalLr => (n.metrics.ml, n.metrics.mr, n.rect.h),
                _ => (n.metrics.mt, n.metrics.mb, n.rect.h),
            }
        };
        // Delta formulation (w76 task 1): the caption's provisional
        // algorithm position already contains its inline-start margin
        // for a transposed caption (`resolve_metrics` set its physical
        // x to `cb_x + ml`, which `vwm_cells_to_logical` transposed
        // into algorithm y), so shifting by `y_border - content_top`
        // adds the block-start margin a second time (vertical-lr: the
        // same margin twice, 80 for a used 40; vertical-rl: both side
        // margins). Shift by the delta to the desired border position
        // instead. This is a no-op for horizontal tables, whose
        // provisional y IS `content_top`. Runs on both resume paths:
        // the measurement resume and the §3.10.6 placement resume both
        // call this function with the (never mutated) provisional rect.
        // w76fix finding 5: the cursor starts at the block-start
        // content edge (see above), so `y_border` is the desired
        // border position even when the top border+padding differs
        // from the block-start border+padding.
        let y_border = y + mbs;
        let prov_y = forest.nodes[cap].rect.y;
        let dx = orthogonal_caption_inline_shift(
            forest,
            cap,
            forest.nodes[idx].data.writing_mode,
            fr.content_w,
        );
        record_shift(forest, cap, dx, y_border - prov_y);
        y = y_border + h + mbe;
    }
    // Floats start below any top captions, like floats encountered in
    // flow after them (a float later in DOM order still starts here;
    // the float stack then orders them).
    let float_y_start = y;
    let mut row_tops = vec![0.0f32; fr.nrows];
    let mut any_visible_row = false;
    // Rowless row groups that absorbed height (empty `<tbody>`s with a
    // specified height, or taking table excess) occupy space at their
    // DOM-order insertion point: before the first row of a later group,
    // or after the last row.
    let mut group_anchor: Vec<Option<f32>> = vec![None; fr.group_nodes.len()];
    let mut insert: Vec<(usize, usize, f32)> = fr
        .group_nodes
        .iter()
        .enumerate()
        .filter(|(gi, _)| {
            let g = fr.group_nodes[*gi];
            let has_rows = fr
                .row_group
                .iter()
                .enumerate()
                .any(|(r, grp)| r < fr.nrows && *grp == Some(g));
            !has_rows && group_target[*gi] > 0.0
        })
        .map(|(gi, _)| (rowless_insert_at(fr, gi), gi, group_target[gi]))
        .collect();
    insert.sort_by_key(|&(at, _, _)| at);
    let mut insert = insert.into_iter().peekable();
    for r in 0..fr.nrows {
        while let Some(&(at, gi, e)) = insert.peek() {
            if at != r {
                break;
            }
            group_anchor[gi] = Some(y);
            y += e;
            insert.next();
        }
        // Collapsed rows draw no spacing around themselves either: only
        // visible rows advance past `border-spacing`.
        let gone = fr.collapsed_rows.get(r).copied().unwrap_or(false);
        if !fr.collapse && !gone {
            y += fr.spacing_v;
        }
        any_visible_row |= !gone;
        row_tops[r] = y;
        for (i, cell) in fr.cells.iter().enumerate().filter(|(_, c)| c.row == r) {
            // Never laid out (collapsed row or fully-collapsed columns):
            // the zero rect needs no placement and paints nothing.
            if !cell_laid_out(&fr.collapsed_rows, &fr.collapsed_cols, cell) {
                continue;
            }
            record_shift(forest, cell.node, 0.0, y - fr.content_top);
            let span_h = if cell.rowspan == 1 {
                row_h[r]
            } else {
                // Collapsed rows take no space and draw no spacing
                // around themselves: count the visible spanned rows
                // only, mirroring the `rowspan` growth pass above.
                let end = span_end(fr, cell);
                let vis: Vec<usize> = (cell.row..end)
                    .filter(|&rr| !fr.collapsed_rows.get(rr).copied().unwrap_or(false))
                    .collect();
                let mut s: f32 = vis.iter().map(|&rr| row_h[rr]).sum();
                if !fr.collapse {
                    s += fr.spacing_v * vis.len().saturating_sub(1) as f32;
                }
                s
            };
            // Align the cell's content within the row height: the border
            // box still stretches to the row; the children shift down by
            // the baseline adjustment plus the top/middle/bottom share of
            // the remaining slack. The shift adds to (rather than
            // overwrites) each child's pending offset, composing with
            // deferred shifts the child's own layout recorded.
            let used_h = forest.nodes[cell.node].rect.h;
            let va = forest.nodes[cell.node].data.valign;
            let kids = forest.nodes[cell.node].children.clone();
            let slack = (span_h - (used_h + bshift[i])).max(0.0);
            let dy = bshift[i]
                + match va {
                    VAlign::Top => 0.0,
                    VAlign::Middle => slack / 2.0,
                    VAlign::Bottom => slack,
                    VAlign::Baseline | VAlign::Sub | VAlign::Super => 0.0,
                };
            {
                let n = &mut forest.nodes[cell.node];
                if n.rect.h < span_h {
                    n.rect.h = span_h;
                }
            }
            if dy > 0.0 {
                for k in kids {
                    add_shift(forest, k, 0.0, dy);
                }
            }
        }
        y += row_h[r];
    }
    // Rowless groups belonging after the last row.
    for (_, gi, e) in insert {
        group_anchor[gi] = Some(y);
        y += e;
    }
    if !fr.collapse && any_visible_row {
        y += fr.spacing_v;
    }
    for &cap in &fr.captions_bottom {
        // Block-axis margins, as for top captions above.
        let (mbs, mbe, h) = {
            let n = &forest.nodes[cap];
            match forest.nodes[idx].data.writing_mode {
                WritingMode::VerticalRl => (n.metrics.mr, n.metrics.ml, n.rect.h),
                WritingMode::VerticalLr => (n.metrics.ml, n.metrics.mr, n.rect.h),
                _ => (n.metrics.mt, n.metrics.mb, n.rect.h),
            }
        };
        // Delta formulation, as for top captions above (w76 task 1);
        // the cursor start is the block-start content edge (w76fix
        // finding 5), so this stays exact with asymmetric
        // border+padding.
        let y_border = y + mbs;
        let prov_y = forest.nodes[cap].rect.y;
        let dx = orthogonal_caption_inline_shift(
            forest,
            cap,
            forest.nodes[idx].data.writing_mode,
            fr.content_w,
        );
        record_shift(forest, cap, dx, y_border - prov_y);
        y = y_border + h + mbe;
    }

    // Position floated children now that their sizes are known (CSS 2.1
    // §9.5), mirroring `resume_block`'s float placement: they stack in
    // DOM order inside the table's block formatting context without
    // advancing the flow cursor. A deferred shift moves each float
    // subtree from its provisional position at the content top.
    // w77fix finding 14: capture the cursor here, before the
    // float-containment max below — this is the rows' true
    // block-end edge (rows plus non-deferred captions), which
    // the deferred-caption phase needs unpolluted by floats.
    let pre_float_h = (y - cursor_start).max(0.0);
    let mut table_floats = FloatCtx::new();
    // The reserved content width from the resume: re-deriving from the
    // node's metrics would give the unreserved width and let floats
    // intrude up to one gutter width into the reserved strip.
    let content_w = fr.content_w;
    for &f in &fr.floats {
        let left = matches!(forest.nodes[f].data.float, Float::Left);
        let (ml, mt, mw, mh) = {
            let n = &forest.nodes[f];
            let m = &n.metrics;
            (m.ml, m.mt, m.ml + n.rect.w + m.mr, m.mt + n.rect.h + m.mb)
        };
        // CSS 2.1 §9.5.1: the float's outer (margin-box) top may not sit
        // above its containing block's content top. `float_y_start` is the
        // margin-box start — the top margin sits *below* it, exactly like
        // block flow's `pending_float_y = cursor_y` in `resume_block`;
        // it is never subtracted off the top.
        let mut fy = float_y_start;
        let clear = forest.nodes[f].data.clear;
        if clear != Clear::None
            && let Some(cy) = clearance_y(&table_floats.entries, clear)
        {
            // Margin-box top: margins never collapse on floats, so the
            // margin sits whole between the cursor and the border box.
            // `clear` clears the border edge (CSS 2.1 §9.5.2): the margin
            // may overlap the float, so the float bottom is offset by the
            // top margin to keep `fy` a margin-box top (mirrors
            // `resume_block`).
            fy = fy.max(cy - mt);
        }
        let (fx, fy) = place_float(&mut table_floats, fr.content_x, content_w, left, mw, mh, fy);
        // The table's BFC contains its floats: the content height covers
        // the lowest float margin-box bottom.
        y = y.max(fy + mh);
        table_floats.push(FloatEntry {
            x: fx,
            y: fy,
            w: mw,
            h: mh,
            left,
            node: f,
        });
        let (px, py) = {
            let n = &forest.nodes[f];
            (n.rect.x, n.rect.y)
        };
        record_shift(forest, f, fx + ml - px, fy + mt - py);
    }

    // Row and row-group rects are assigned directly as absolute rects:
    // never via pending shifts, which accumulate down the tree and would
    // move cells twice (cells already carry their own placement shift).
    for (r, row_node) in fr.row_nodes.iter().enumerate() {
        if r >= fr.nrows {
            break;
        }
        if let Some(rn) = row_node {
            let n = &mut forest.nodes[*rn];
            n.rect.x = fr.row_x;
            n.rect.y = row_tops[r];
            n.rect.w = fr.row_w;
            n.rect.h = row_h[r];
        }
    }
    // Row-group rects are the union of their rows' rects. Rowless
    // groups that absorbed height (a specified height, or a share of
    // the table excess) get their anchor rect from the placement pass;
    // other empty groups keep their zero rect.
    for (gi, &g) in fr.group_nodes.iter().enumerate() {
        let mut top = f32::INFINITY;
        let mut bottom = f32::NEG_INFINITY;
        for (r, grp) in fr.row_group.iter().enumerate() {
            if *grp == Some(g) && r < fr.nrows {
                top = top.min(row_tops[r]);
                bottom = bottom.max(row_tops[r] + row_h[r]);
            }
        }
        if top.is_finite() {
            let n = &mut forest.nodes[g];
            n.rect.x = fr.row_x;
            n.rect.y = top;
            n.rect.w = fr.row_w;
            n.rect.h = (bottom - top).max(0.0);
        } else if let Some(y0) = group_anchor[gi] {
            let n = &mut forest.nodes[g];
            n.rect.x = fr.row_x;
            n.rect.y = y0;
            n.rect.w = fr.row_w;
            n.rect.h = group_target[gi].max(0.0);
        }
    }

    // Column and column-group rects cover the grid area of the columns
    // they span. Assigned directly: column boxes take no space in the
    // grid (their children, if any, are not laid out).
    if fr.nrows > 0 {
        let grid_top = row_tops[0];
        let grid_bottom = row_tops[fr.nrows - 1] + row_h[fr.nrows - 1];
        let mut ranges: Vec<(usize, usize, usize)> = Vec::new();
        for (cc, spec) in fr.columns.iter().enumerate() {
            for node in spec.col_node.into_iter().chain(spec.group_node) {
                match ranges.iter_mut().find(|r| r.0 == node) {
                    Some(r) => {
                        r.1 = r.1.min(cc);
                        r.2 = r.2.max(cc);
                    }
                    None => ranges.push((node, cc, cc)),
                }
            }
        }
        for (node, start, end) in ranges {
            let n = &mut forest.nodes[node];
            n.rect.x = fr.col_x[start];
            n.rect.y = grid_top;
            n.rect.w = (fr.col_x[end] + fr.col_w[end] - fr.col_x[start]).max(0.0);
            n.rect.h = (grid_bottom - grid_top).max(0.0);
        }
    }

    let natural_h = (y - cursor_start).max(0.0);
    // Return the stacked extent (rows plus non-deferred captions)
    // plus the pre-float cursor (w77fix finding 14): the
    // deferred-caption phase needs the pre-float edge as the
    // rows' true block-end (floats extend the stacked extent
    // above, but must not push bottom deferred captions), distinct
    // from the specified-floored content size
    // `finish_content_height` stamps below.
    // CSS 2.1 §17.5.3: a specified table height is a *minimum* — the
    // table grows past it when its rows (or row-group minimums) need
    // more room, instead of clamping the content down to it. In the
    // collapsing border model the rows' heights already include the
    // inside halves of the collapsed borders, which are the same
    // borders the table's own border box counts in full: growing past
    // the specified height would double-count them (WPT
    // `td-box-sizing-002` tables 5/12), so the growth applies to the
    // separated border model only. When rows are collapsed, their share
    // of the specified height vanishes with them in the zeroing pass
    // above (WPT `visibility-collapse-row-004`), so the placed height
    // is final and the specified height is not re-applied as a minimum.
    let hspec_eff = if fr.collapsed_rows.iter().any(|&c| c) {
        None
    } else if fr.collapse {
        fr.hspec
    } else {
        fr.hspec.map(|h| h.max(natural_h))
    };
    // A vertical table's block-axis percentages (min-width/max-width)
    // resolve against the containing-block width, not height.
    let cb_h = if super::is_vertical_table(forest, idx) {
        Some(fr.cb_w)
    } else {
        fr.cb_h
    };
    finish_content_height(forest, idx, cb_h, hspec_eff, natural_h);
    // In the collapsing border model the table has no padding and its
    // border box extends only half the specified border beyond the
    // content on each side (CSS 2.1 §17.6.2; fantasai's
    // collapsed-outer-border: "the distance between the table's padding
    // and border edges is half the table's specified border", and "a
    // table does not have padding"). `finish_content_height` added the
    // full border and padding, so correct the border-box height here
    // (WPT `td-box-sizing-002` tables 6/13: content-box + collapse with
    // a 100px specified height, 10px padding, and 20/40px borders must
    // give 130, not 180). Content-box only, mirroring
    // `collapse_content_box_table_width`: under `border-box` the
    // specified height already counts the full border/padding, so
    // re-applying the half-border correction shrinks the table (WPT
    // `td-box-sizing-002` tables 4/5/12: border-box + collapse with
    // height:100px must stay 100, not 50).
    if fr.collapse && forest.nodes[idx].data.box_sizing == BoxSizing::ContentBox {
        let vertical = super::is_vertical_table(forest, idx);
        let n = &mut forest.nodes[idx];
        let content_h = n.content_h;
        let m = &n.metrics;
        // Block axis: physical-vertical for horizontal tables,
        // physical-horizontal for vertical tables.
        let (b1, b2) = if vertical { (m.bl, m.br) } else { (m.bt, m.bb) };
        n.rect.h = (b1 / 2.0 + content_h + b2 / 2.0).max(0.0);
    }
    // Vertical writing mode: the table consumed width/height logically,
    // so the placed (inline, block)-space rects transpose back to
    // physical here (see `super::transpose_vwm_table`).
    if super::is_vertical_table(forest, idx) {
        super::transpose_vwm_table(forest, fr);
    }
    // Inline-level tables align by the first row's baseline (CSS 2.1
    // §10.8.1; css-align-3 § "table rows" + Blink's
    // `LayoutTableSection::FirstLineBoxBaseline` in
    // `table_first_row_baseline`); block-level tables keep the
    // border-box bottom.
    forest.nodes[idx].baseline =
        if forest.nodes[idx].data.display == crate::style::Display::InlineTable {
            table_inline_baseline(forest, fr, idx)
        } else {
            forest.nodes[idx].rect.h
        };
    // Stamp the phantom sizing after the rect is final (post
    // `transpose_vwm_table`, so the predicate's physical comparisons see
    // the same geometry the shim reads from the final `LayoutBox`).
    // Runs on both resume paths: the measurement resume and the §3.10.6
    // second-pass placement resume both end here.
    let (phantom_w, phantom_h) = table_sized_by_phantom(forest, idx, fr.spacing_h, fr.spacing_v);
    forest.nodes[idx].phantom_w = phantom_w;
    forest.nodes[idx].phantom_h = phantom_h;
    (natural_h, pre_float_h)
}

/// The percentage fraction of a `SpecLen`, if it carries a percentage
/// component (`None` for `auto`, absolute lengths, and `calc()`s whose
/// percentage coefficient folded to zero — all base-independent).
fn pct_fraction(s: SpecLen) -> Option<f32> {
    // csswg#94: a `calc()` that is purely a percentage (`calc(50% +
    // 0px)`) behaves as that percentage; a `calc()` mixing a percentage
    // with a nonzero length behaves as `auto` for table sizing (it
    // cannot contribute a definite width), so it is not a percentage
    // participant either.
    match s {
        SpecLen::Pct(v) => Some(v / 100.0),
        SpecLen::Math { pct, px, .. } if px == 0.0 && pct != 0.0 => Some(pct / 100.0),
        _ => None,
    }
}

/// A table cell/column box's specified inline size: the physical
/// `height` for vertical tables (their inline axis is physical
/// vertical), otherwise the physical `width`. One idiom for the
/// transposed spec reads scattered across the column machinery.
fn inline_axis_spec(data: &crate::style::StyleData, vertical: bool) -> SpecLen {
    if vertical { data.height } else { data.width }
}

/// A table cell/column box's specified inline minimum: the physical
/// `min-height` for vertical tables, otherwise `min-width`.
fn inline_axis_min_spec(data: &crate::style::StyleData, vertical: bool) -> SpecLen {
    if vertical {
        data.min_height
    } else {
        data.min_width
    }
}

/// Mutable access to the inline-axis spec slot (`height`/`min-height`
/// for vertical tables, `width`/`min-width` otherwise): the percentage
/// normalization's save/scale/restore mutates the spec it scaled.
fn inline_axis_spec_slot(
    data: &mut crate::style::StyleData,
    vertical: bool,
    is_min_width: bool,
) -> &mut SpecLen {
    if vertical {
        if is_min_width {
            &mut data.min_height
        } else {
            &mut data.height
        }
    } else if is_min_width {
        &mut data.min_width
    } else {
        &mut data.width
    }
}

/// Normalize a specified `width` for table column sizing (csswg#94): a
/// `calc()` that is purely a percentage (`calc(50% + 0px)`, or even
/// `calc(0%)`) behaves as that percentage, while a `calc()` mixing a
/// percentage with a nonzero length (`calc(20% + 80px)`) behaves as
/// `auto` — it cannot contribute a definite width during the intrinsic
/// passes, in either layout mode (WPT `auto-layout-calc-width-001.html`,
/// `fixed-layout-calc-width-001.html`). Cyclic-ness is syntactic (CSS
/// Values 4 §8.1.4): the gate checks the `had_pct` bit, not the folded
/// coefficient, so `calc(80px + 0%)` — whose percentage folded to zero
/// — behaves as `auto`, not as a definite `80px`.
fn table_col_spec(s: SpecLen) -> SpecLen {
    match s {
        // Intrinsic sizes are not specified widths for column sizing
        // (css-sizing-3 §5): a `td` or `col` with `width: min-content`
        // must behave as `auto` here. Without this the call sites read
        // the provisional fill (`metrics.content_w`) as a definite
        // specified width, blowing the column minima up to ~the table
        // width.
        SpecLen::FitContent | SpecLen::MinContent | SpecLen::MaxContent => SpecLen::Auto,
        // No percentage leaf at all: a definite length.
        SpecLen::Math {
            had_pct: false, px, ..
        } => SpecLen::Px(px),
        // Frozen base-0 form of non-affine percentage math
        // (`min()`/`max()`/`clamp()` with `%`): the coefficients are an
        // approximation, not a true coefficient pair, so the value is
        // value-identical to `calc(0% + 0px)` yet must not participate
        // as a percentage — it behaves as `auto`.
        SpecLen::Math {
            affine: false,
            had_pct: true,
            ..
        } => SpecLen::Auto,
        // Purely a percentage (csswg#94): behaves as that percentage.
        // Only the genuine affine form reaches here; the frozen form
        // was caught by the arm above.
        SpecLen::Math { pct, px: 0.0, .. } => SpecLen::Pct(pct),
        // Anything else percentage-bearing is cyclic: `calc(20% + 80px)`
        // mixes a percentage with a length, and `calc(80px + 0%)` had a
        // percentage leaf that folded to zero — both behave as `auto`.
        SpecLen::Math { .. } => SpecLen::Auto,
        s => s,
    }
}

/// Percentage specs the second preferred-width pass re-resolves against
/// the table width: `<col>` / `<colgroup>` widths, cell widths, and cell
/// `min-width`s. Returns the deduplicated `(node, is_min_width)` list
/// plus the total percentage pressure (sum of fractions; a `<col
/// span=2>` counts once per covered column, a spanning cell or group
/// once per spec — each spec's total pressure is its percentage however
/// it spreads).
///
/// For vertical tables the reads are the transposed inline specs
/// (`height` / `min-height`): a `height:N%` participates, a `width:N%`
/// (the block axis) does not.
///
/// Cell-width participation mirrors the fixpoint body: first-row cells
/// under fixed layout (pins), all cells in the automatic layout
/// (minima). `min-width` floors apply in both.
fn pct_participants(forest: &Forest, grid: &TableGrid, fixed: bool) -> (Vec<(usize, bool)>, f32) {
    let mut parts: Vec<(usize, bool)> = Vec::new();
    let mut total = 0.0f32;
    // The pressure sums per column, but each node is scaled once: a
    // `<col span=2>` contributes twice yet is normalized once.
    let mut add = |node: usize, is_min_width: bool, frac: f32| {
        if !parts.contains(&(node, is_min_width)) {
            parts.push((node, is_min_width));
        }
        total += frac;
    };
    let vertical = grid.vertical;
    for spec in &grid.columns {
        if let Some(n) = spec.col_node
            && let Some(frac) = pct_fraction(inline_axis_spec(&forest.nodes[n].data, vertical))
        {
            add(n, false, frac);
        }
    }
    for &g in &grid.col_group_nodes {
        if let Some(frac) = pct_fraction(inline_axis_spec(&forest.nodes[g].data, vertical)) {
            add(g, false, frac);
        }
    }
    // Fixed mode reads "the first row" as the first *visual* row (see
    // `first_visual_row`), not the first DOM row.
    let first_row = if fixed {
        first_visual_row(forest, grid)
    } else {
        0
    };
    for cell in &grid.cells {
        let n = &forest.nodes[cell.node];
        // Cells that never lay out resolve no widths (see above).
        if !cell_width_contributes(&grid.collapsed_cols, cell) {
            continue;
        }
        if (!fixed || cell.row == first_row)
            && let Some(frac) = pct_fraction(inline_axis_spec(&n.data, vertical))
        {
            add(cell.node, false, frac);
        }
        if let Some(frac) = pct_fraction(inline_axis_min_spec(&n.data, vertical)) {
            add(cell.node, true, frac);
        }
    }
    (parts, total)
}

/// Whether any percentage *padding* the second pass re-resolves exists
/// (horizontal padding on cells and column boxes). For vertical tables
/// the inline axis is physical vertical, so percentage top/bottom padding
/// also participates. With no participant specs either, the fixpoint loop
/// is a no-op and skipped.
fn has_pct_padding(forest: &Forest, grid: &TableGrid) -> bool {
    let pad_pct = |node: usize| {
        let p = &forest.nodes[node].data.padding;
        // Physical indices: top=0, bottom=2, left=3, right=1.
        // A vertical table's inline axis is physical vertical, so its
        // column inline sizes consume the vertical padding instead.
        if grid.vertical {
            pct_fraction(p[0]).is_some() || pct_fraction(p[2]).is_some()
        } else {
            pct_fraction(p[1]).is_some() || pct_fraction(p[3]).is_some()
        }
    };
    grid.cells.iter().any(|c| pad_pct(c.node))
        || grid.col_nodes.iter().any(|&n| pad_pct(n))
        || grid.col_group_nodes.iter().any(|&n| pad_pct(n))
}

/// CSS 2.1 §17.5.2.2 CAPMIN: the greatest minimum caption outer width.
/// Each in-flow caption contributes its min-content width measured as a
/// hypothetical block box, plus its own padding, border, and margins.
/// Captions lay out at the table's content width, so the used table
/// width must be at least the widest caption's outer minimum. This is a
/// pre-pass: the used width resolves in `layout_block` before
/// `layout_table` runs, so caption metrics resolve against a zero base
/// here (percentages have no definite base this early).
///
/// In a vertical writing mode the floor sites compare this value against
/// the algorithm width, which is the *inline* size — physical vertical —
/// so each caption contributes its *vertical* min-content extent plus
/// its vertical extras (border/padding/margins on the top and bottom)
/// instead of the horizontal measure.
pub(crate) fn table_caption_min_width(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    scratch: &mut Vec<f32>,
    pool: &mut TableGridPool,
) -> f32 {
    wrap_table_stray_content(forest, idx);
    let mut grid = pool.take_grid();
    collect_table_grid_into(forest, idx, pool, &mut grid);
    let mut capmin = 0.0f32;
    // One pooled walk serves every caption instead of allocating per
    // caption.
    let mut int_bufs = WalkBufs::new();
    for &cap in &grid.captions {
        resolve_metrics(forest, cap, 0.0, 0.0, false, None, false);
        let outer = if grid.vertical {
            let v = vertical_caption_min_content(forest, cap, fonts, scratch, &mut int_bufs);
            let m = &forest.nodes[cap].metrics;
            v + m.bt + m.pt + m.pb + m.bb + m.mt + m.mb
        } else {
            let mcw = min_content_width_pooled(forest, cap, fonts, scratch, &mut int_bufs);
            let m = &forest.nodes[cap].metrics;
            mcw + m.bl + m.pl + m.pr + m.br + m.ml + m.mr
        };
        if outer > capmin {
            capmin = outer;
        }
    }
    pool.put_grid(grid);
    capmin
}

/// Preferred (max-content) caption width of a table: the greatest
/// preferred caption outer width. Mirrors [`table_caption_min_width`]
/// but uses [`preferred_width_mcw`] so a floated/absolutely-positioned
/// table's shrink-to-fit width accounts for a wide caption (the caption
/// lays out at the table's content width, so the used width must be at
/// least the widest caption's outer preferred width).
///
/// Like its sibling, this measures on the *vertical* axis (plus vertical
/// extras) when the table is in a vertical writing mode, since the
/// floor sites compare against the inline (physical vertical) algorithm
/// width.
pub(crate) fn table_caption_pref_width(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    cache: &mut PrefWidthCache,
    pool: &mut TableGridPool,
) -> f32 {
    wrap_table_stray_content(forest, idx);
    let mut grid = pool.take_grid();
    collect_table_grid_into(forest, idx, pool, &mut grid);
    let mut maxw = 0.0f32;
    // One scratch triple serves every caption instead of allocating a
    // node-sized `mcw` plus fresh walk buffers per caption.
    let mut mcw_scratch = Vec::new();
    let mut int_bufs = WalkBufs::new();
    let mut int_inner = WalkBufs::new();
    for &cap in &grid.captions {
        resolve_metrics(forest, cap, 0.0, 0.0, false, None, false);
        let outer = if grid.vertical {
            let w = vertical_caption_max_content(
                forest,
                cap,
                fonts,
                cache,
                &mut mcw_scratch,
                &mut int_bufs,
                &mut int_inner,
            );
            let m = &forest.nodes[cap].metrics;
            w + m.bt + m.pt + m.pb + m.bb + m.mt + m.mb
        } else {
            let w = preferred_width_mcw(
                forest,
                cap,
                fonts,
                cache,
                &mut mcw_scratch,
                &mut int_bufs,
                &mut int_inner,
            );
            let m = &forest.nodes[cap].metrics;
            w + m.bl + m.pl + m.pr + m.br + m.ml + m.mr
        };
        if outer > maxw {
            maxw = outer;
        }
    }
    pool.put_grid(grid);
    maxw
}
/// Consult the table's own specified `width` for its intrinsic
/// contribution (css-sizing-3 §5.2), mirroring the flex/multicol arms of
/// `layout::preferred_width_mcw`:
/// - `width: min-content` contributes the min-content size, measured by
///   the min-content walk over the caller's pooled buffers;
/// - a definite specified width (or a replaced box) contributes exactly
///   its content-box width (a definite `width` via
///   [`definite_specified_content_width`], else the replaced box's
///   [`intrinsic_content_width`]);
/// - otherwise (`auto`, `max-content`, `fit-content`, cyclic
///   percentages) the column-based `intrinsic` stands: the fit-content
///   clamp against the available width happens at used-value time,
///   where the base is definite (the block-table shrink path and the
///   float/abspos shrink-to-fit formula).
///
/// `spacing_total` is added to the min-content walk: the walk excludes
/// inter-column spacing by design (its callers add it), but `pref_w`
/// carries spacing on every other `table_pref_width` path, and the
/// table's used min-content width adds it too (the block-table shrink
/// path's `floor_c`).
///
/// The fallthrough intrinsic arm is clamped by the table's own
/// `min-width` / `max-width` (mirroring the block convention in
/// [`definite_width_floor`]); the definite arm and the `MinContent`
/// arm are not (the latter is already clamped inside
/// `min_content_width_pooled`).
fn table_pref_adjusted(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    intrinsic: f32,
    spacing_total: f32,
    mcw: &mut Vec<f32>,
    bufs: &mut WalkBufs,
) -> f32 {
    if matches!(table_inline_spec(forest, idx), SpecLen::MinContent) {
        return min_content_width_pooled(forest, idx, fonts, mcw, bufs) + spacing_total;
    }
    definite_specified_physical_width(forest, idx)
        .or_else(|| intrinsic_content_width(forest, idx))
        .unwrap_or_else(|| clamp_content_width_by_minmax(forest, idx, intrinsic))
}

/// Preferred (max-content) width of a table: the column structure from
/// [`collect_table_grid`], each column the widest cell preferred width
/// plus its padding/border, summed with spacing. Under
/// `table-layout: fixed`, specified column widths pin their columns
/// first, then specified first-row widths (columns without either fall
/// back to content measurement); specified cell widths from later rows
/// are ignored, symmetric with the used-width rule (CSS 2.1 §17.5.2.1),
/// so the preferred width cannot exceed the used width.
///
/// specified column widths act as minima. For a shrink-to-fit table (floated, out-of-flow, or
/// inline-table) the widest floated child's margin box floors the total:
/// floats take no space in the grid, but a float's hypothetical position
/// is inside the table's block formatting context (table wrappers
/// establish one, CSS 2.1 §9.4.1), so a narrower table would let the
/// float overflow; the floor follows from shrink-to-fit sizing (CSS 2.1
/// §10.3.5/§10.3.7) and implementation consensus. css-sizing-3 §5
/// contains no statement that floats contribute to intrinsic sizes, so
/// the behavior is marked WPT-verification-needed. A block-level
/// `width: auto` table is excluded: block layout sizes it from its
/// columns per CSS 2.1 §17.5.2.2, and floats take no part there —
/// intrinsic contributions only feed shrink-to-fit (css-sizing-3 §5.2
/// defines a float's contribution as the max-content size of its
/// hypothetical box). Used by the `preferred_width` shrink-to-fit path
/// (floated / absolutely-positioned tables).
///
/// When `caption_floor` is false, the caption's preferred (max-content)
/// width does not floor the result: a block-level table sizes from its
/// columns per CSS 2.1 §17.5.2.2, floored by the caption's
/// *min-content* (CAPMIN) at used-width time instead (WPT
/// `caption.html` table 6: an empty bordered table stays at its border
/// width rather than growing by the caption's preferred width).
pub(crate) fn table_pref_width(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    cache: &mut PrefWidthCache,
    pool: &mut TableGridPool,
    caption_floor: bool,
) -> f32 {
    wrap_table_stray_content(forest, idx);
    let mut grid = pool.take_grid();
    collect_table_grid_into(forest, idx, pool, &mut grid);
    let out = table_pref_width_collected(forest, idx, fonts, cache, pool, &grid, caption_floor);
    pool.put_grid(grid);
    out
}

/// `table_pref_width`'s column computation over an already-collected
/// grid. Split out so the pooled grid's lifetime spans the early
/// returns below (each would otherwise need its own `put_grid`).
fn table_pref_width_collected(
    forest: &mut Forest,
    idx: usize,
    fonts: &dyn FontMetrics,
    cache: &mut PrefWidthCache,
    pool: &mut TableGridPool,
    grid: &TableGrid,
    caption_floor: bool,
) -> f32 {
    // Intrinsic fast path (see `table_pref_adjusted`): a definite specified
    // width (or a replaced box) contributes exactly its content-box width,
    // and `width: min-content` contributes the min-content walk plus
    // spacing — neither consults the column-based intrinsic size computed
    // below, so the per-cell `preferred_width_mcw` / `min_content_width`
    // walks and the percentage fixpoint are skipped. The arms observe the
    // same forest state as the late path: `wrap_table_stray_content` has
    // run, and the column machinery's mutations are either re-done inside
    // the min-content walk (`resolve_metrics` in `auto_column_minima` and
    // `table_caption_min_width`) or invisible to it (pref-width cache
    // fills; the fixpoint's save/restore leaves styles untouched).
    let (_, spacing_h, _) = table_border_model(forest, idx);
    if matches!(table_inline_spec(forest, idx), SpecLen::MinContent) {
        // Guarded on the visible column count, not the column count
        // (like the fixed-indefinite arm below): an
        // all-collapsed-columns table draws no spacing. The `ncols ==
        // 0` early return further down already passes 0.0.
        let nvis = grid.collapsed_cols.iter().filter(|&&c| !c).count();
        let spacing_total = if nvis == 0 {
            0.0
        } else {
            spacing_h * (nvis as f32 + 1.0)
        };
        let mut early_mcw = Vec::new();
        let mut early_bufs = WalkBufs::new();
        return min_content_width_pooled(forest, idx, fonts, &mut early_mcw, &mut early_bufs)
            + spacing_total;
    }
    if let Some(definite) = definite_specified_physical_width(forest, idx)
        .or_else(|| intrinsic_content_width(forest, idx))
    {
        return definite;
    }
    // Blink's fixed-layout percentage-width rule (WPT
    // `css/css-tables/tentative/table-fixed-minmax.html`): a
    // fixed-layout table with a percentage width has an *infinite*
    // max-content intrinsic inline size (min-content stays regular via
    // `table_min_content_width`, which never reaches this function).
    // This short-circuits the column machinery and the percentage
    // fixpoint below. Percentage-bearing `calc()` counts: like
    // `definite_specified_content_width`, the gate is syntactic
    // (`had_pct`), matching Blink's "has a width:%" condition.
    // Containers that cannot handle an infinite child harden against
    // non-finite sizes at their own used-size boundaries: `flex_lengths`
    // saturates item targets (its distribute-and-freeze fixpoint cannot
    // converge on NaN), block/grid `width: max-content` used widths
    // saturate, `fr` grid tracks cap an infinite content base at the
    // definite container width, and the block-level auto-table shrink
    // clamp turns INF into the available width.
    let width_spec = table_inline_spec(forest, idx);
    if grid.fixed
        && (matches!(width_spec, SpecLen::Pct(_))
            || matches!(width_spec, SpecLen::Math { had_pct: true, .. }))
    {
        return f32::INFINITY;
    }
    // One scratch triple serves every float/cell measurement below
    // instead of allocating a node-sized `mcw` plus fresh walk buffers
    // per item.
    let mut mcw_scratch = Vec::new();
    let mut int_bufs = WalkBufs::new();
    let mut int_inner = WalkBufs::new();
    // The caption's preferred outer width floors the table's preferred
    // width (CSS 2.1 §17.5.2.2 analog of CAPMIN): a floated/abspos table
    // with a wide caption must size wide enough for it. Skipped for
    // block-level tables (`caption_floor == false`), which size from
    // their columns and take the caption's *min-content* as the floor
    // at used-width time instead. Further gated on shrink-to-fit below
    // (`cap_floor`): a block-level table's caption must not grow the
    // table — only its min-content floors the grid min (via
    // `table_caption_min_width`, applied as CAPMIN at the used-width
    // sites).
    let cap_pref = if caption_floor {
        table_caption_pref_width(forest, idx, fonts, cache, pool)
    } else {
        0.0
    };
    // Floated children of the table take no space in the grid, but a
    // float's hypothetical position is inside the table's block
    // formatting context (table wrappers establish one, CSS 2.1 §9.4.1),
    // so for a shrink-to-fit table (floated, out-of-flow, or
    // inline-table) the widest float's margin box floors the preferred
    // width: without this such a table sizes to its columns alone and
    // lets a wider float child overflow. Honest basis: CSS 2.1 §9.4.1
    // plus shrink-to-fit sizing per CSS 2.1 §10.3.5/§10.3.7 and
    // implementation consensus — css-sizing-3 §5 contains no statement
    // that floats contribute to intrinsic sizes, so the behavior is
    // marked WPT-verification-needed. A block-level `width: auto` table
    // is excluded: block layout sizes it from its columns per CSS 2.1
    // §17.5.2.2, and floats take no part there — intrinsic contributions
    // only feed shrink-to-fit (css-sizing-3 §5.2 defines a float's
    // contribution as the max-content size of its hypothetical box).
    // The max, not the sum, is the floor: whether floats sit side by
    // side needs a definite width to decide, and the max never
    // over-widens. A definite specified width on the float contributes
    // exactly, symmetric with the cell handling below (`width: auto`
    // resolves to zero against the zero base, leaving the content
    // preferred width). Computed before the empty-grid early return: a
    // shrink-to-fit table whose only child is a float has no columns,
    // but the float still floors the preferred width.
    let shrink_to_fit = is_float(forest, idx)
        || layout_position::is_out_of_flow(forest, idx)
        || forest.nodes[idx].data.display == crate::style::Display::InlineTable;
    // The caption's max-content floors the table's preferred width only
    // in shrink-to-fit contexts (floated, out-of-flow, inline-table):
    // for a block-level table the caption must not grow the table (WPT
    // `caption.html` table 1: a block-level auto table sizes to the
    // caption min, 50px, not its max-content, 80px) — only the caption
    // min floors the grid min (CAPMIN at the used-width sites).
    let cap_floor = if shrink_to_fit { cap_pref } else { 0.0 };
    let mut float_max = 0.0f32;
    if shrink_to_fit {
        for &f in &grid.floats {
            resolve_metrics(forest, f, 0.0, 0.0, false, None, false);
            // A vertical table's inline axis is physical vertical: the
            // float floors the inline size with its vertical max-content.
            let w = if grid.vertical {
                vertical_cell_max_content(
                    forest,
                    f,
                    fonts,
                    cache,
                    &mut mcw_scratch,
                    &mut int_bufs,
                    &mut int_inner,
                )
            } else {
                preferred_width_mcw(
                    forest,
                    f,
                    fonts,
                    cache,
                    &mut mcw_scratch,
                    &mut int_bufs,
                    &mut int_inner,
                )
            };
            let n = &forest.nodes[f];
            let m = &n.metrics;
            // The float's margin box on the inline axis: its own
            // vertical margins/border/padding around the vertical
            // max-content (the definite inline size is already folded
            // into `w` by `vertical_cell_max_content`, so no
            // `content_w` max like the horizontal arm).
            let mb = if grid.vertical {
                m.mt + w + m.bt + m.pt + m.pb + m.bb + m.mb
            } else {
                m.ml + w.max(m.content_w) + m.bl + m.pl + m.pr + m.br + m.mr
            };
            float_max = float_max.max(mb.max(0.0));
        }
    }
    if grid.ncols == 0 {
        // No columns: the caption/float floors are the whole intrinsic
        // size; the table's own specified width still governs the
        // contribution (see `table_pref_adjusted`). The caption floor
        // is the shrink-to-fit-gated `cap_floor`, not the raw caption
        // max-content.
        let intrinsic = cap_floor.max(float_max);
        return table_pref_adjusted(
            forest,
            idx,
            fonts,
            intrinsic,
            0.0,
            &mut mcw_scratch,
            &mut int_bufs,
        );
    }
    let fixed = grid.fixed;
    // Fixed mode reads "the first row" as the first *visual* row (see
    // `first_visual_row`), not the first DOM row.
    let first_row = if fixed {
        first_visual_row(forest, grid)
    } else {
        0
    };
    // Resolve cell padding/border against a zero base (percentages have no
    // definite base this early; px borders resolve exactly). Skipped for
    // cells that feed no width computation (spanning solely collapsed
    // columns): the metrics feed no width computation here. Collapsed-row
    // cells never lay out, but their widths still size columns (CSS 2.1
    // §17.5.5), so they are resolved here.
    for cell in &grid.cells {
        if cell_width_contributes(&grid.collapsed_cols, cell) {
            resolve_metrics(forest, cell.node, 0.0, 0.0, false, None, false);
        }
    }
    // Specified `<col>` / `<colgroup>` widths, border-box (percentages
    // resolve against the same zero base). A `colgroup` width spreads
    // over its span; under `table-layout: fixed` the spread pins its
    // columns, in the automatic layout it is a minimum. The preferred
    // width keeps the established behavior — both raise the column
    // floor — so the spread flag is unused here.
    let col_spec = column_specified_widths(forest, grid, 0.0, 0.0);
    let mut col_pref = vec![0.0f32; grid.ncols];
    let mut col_pinned = vec![false; grid.ncols];
    // Collapsed columns take no width in any pass below (CSS 2.1 §17.5):
    // pre-pinning keeps the fixed-layout pins and excess-spread `open`
    // filters from ever touching them; the automatic-layout loops below
    // guard on `collapsed_cols` explicitly.
    for (cc, &gone) in grid.collapsed_cols.iter().enumerate() {
        if gone {
            col_pinned[cc] = true;
        }
    }
    if fixed {
        // Column boxes pin their columns first (CSS 2.1 §17.5.2.1), ahead
        // of first-row cell widths. Collapsed columns stay at zero.
        let (pins, pinned) = fixed_column_pins(forest, grid, &col_spec);
        if grid.fixed_indefinite {
            // A non-definite specified width (`min-content`,
            // `fit-content`, `auto`, `max-content`, or a `calc-size()`
            // with a keyword basis) sizes the fixed table from the
            // pinned first-row column widths alone (CSS 2.1 §17.5.2.1):
            // content never widens it (WPT
            // `css/css-tables/fixed-layout-2.html`). Skip the
            // content-measuring passes below entirely.
            let (_, spacing_h, _) = table_border_model(forest, idx);
            let nvis = grid.collapsed_cols.iter().filter(|&&c| !c).count();
            // Guarded on visible columns, not the column count: an
            // all-collapsed-columns table draws no spacing.
            let spacing_total = if nvis == 0 {
                0.0
            } else {
                spacing_h * (nvis as f32 + 1.0)
            };
            let intrinsic = (pins.iter().sum::<f32>() + spacing_total)
                .max(cap_floor)
                .max(float_max)
                .max(0.0);
            return table_pref_adjusted(
                forest,
                idx,
                fonts,
                intrinsic,
                spacing_total,
                &mut mcw_scratch,
                &mut int_bufs,
            );
        }
        col_pref = pins;
        col_pinned = pinned;
    } else {
        // Specified column widths act as minima; content can widen the
        // column past them. Collapsed columns keep a zero floor.
        for (cc, w) in col_spec.0.iter().enumerate() {
            if let Some(bb) = w
                && !grid.collapsed_cols[cc]
                && *bb > col_pref[cc]
            {
                col_pref[cc] = *bb;
            }
        }
    }
    // CSS Tables 3 §3.8.3 constrainedness (automatic layout only):
    // an auto cell in a constrained column contributes its min-content
    // width (not its max-content width) to the column preferred width —
    // see `cell_outer_max_content`. Computed once: the loop below only
    // indexes it.
    let col_constrained = if fixed {
        vec![false; grid.ncols]
    } else {
        column_constrained(forest, grid)
    };
    for cell in grid.cells.iter().filter(|c| c.colspan == 1) {
        // Cells in a collapsed row never lay out (CSS 2.1 §17.5).
        if !cell_width_contributes(&grid.collapsed_cols, cell) {
            continue;
        }
        if col_pinned[cell.col] {
            continue;
        }
        // A vertical table's column (inline) axis is the physical
        // vertical axis: measure the cell's vertical content size
        // instead of its horizontal preferred width.
        let w = if grid.vertical {
            vertical_cell_max_content(
                forest,
                cell.node,
                fonts,
                cache,
                &mut mcw_scratch,
                &mut int_bufs,
                &mut int_inner,
            )
        } else {
            preferred_width_mcw(
                forest,
                cell.node,
                fonts,
                cache,
                &mut mcw_scratch,
                &mut int_bufs,
                &mut int_inner,
            )
        };
        // Under `table-layout: fixed` a cell outside the first row
        // contributes its content-based width, not its specified one
        // (CSS 2.1 §17.5.2.1): `preferred_width` folds a definite width
        // into the max-content size (css-sizing-3 §5.2), which would
        // reintroduce the specified-width floor through `w` below. The
        // children's `pref_w` entries are filled by the call above, so
        // the content-based measure is exact. Gated on a non-`auto`
        // specified width: `auto` cells are already content-based.
        // The gate reads the inline-axis spec (transposed for vertical
        // tables); the borrow ends before the fallback below, which
        // needs `&mut forest`.
        let non_auto_spec = {
            let n = &forest.nodes[cell.node];
            !matches!(
                table_col_spec(inline_axis_spec(&n.data, grid.vertical)),
                SpecLen::Auto
            )
        };
        let w = if fixed && cell.row != first_row && non_auto_spec {
            // Transposed with the content measure above: a vertical
            // table's content-based width is its vertical max-content.
            if grid.vertical {
                vertical_cell_max_content(
                    forest,
                    cell.node,
                    fonts,
                    cache,
                    &mut mcw_scratch,
                    &mut int_bufs,
                    &mut int_inner,
                )
            } else {
                generic_preferred_width(forest, cell.node)
            }
        } else {
            w
        };
        // An infinite cell width (a contained automatic-layout table
        // at the 100% percentage floor) must not poison this table's
        // preferred width: the table contributes its finite
        // column-maximum sum instead (see `finite_cell_max_content`).
        // The table's own preferred width still keeps the floor — only
        // its contribution as cell content is finite — so a standalone
        // block/float table still clamps to the available width.
        let w = if !grid.vertical && w.is_infinite() {
            finite_cell_max_content(
                forest,
                cell.node,
                fonts,
                cache,
                &mut mcw_scratch,
                &mut int_bufs,
                &mut int_inner,
            )
            .unwrap_or(w)
        } else {
            w
        };
        let n = &forest.nodes[cell.node];
        // Symmetric with `auto_column_minima` (and the automatic-layout
        // maxima in `layout_table`): a specified cell width floors the
        // column maximum as well as the minimum (CSS 2.1 §17.5.2.2 step
        // 2, "W: the width specified for the cell"). This floor only
        // fires for unpinned columns: under `table-layout: fixed` a
        // first-row pin can hold the maximum below the content-based
        // minimum, and the repair after the spanning loop raises it
        // back, so no universal max >= min claim is made here.
        // csswg#94: a percentage-bearing `calc()` with a nonzero length
        // part counts as `auto` here (see `table_col_spec`); pure
        // percentages resolve against the zero base to 0.0 and are
        // re-applied as minima by the second pass below. Under
        // `table-layout: fixed` only the first row's specified widths
        // count (CSS 2.1 §17.5.2.1), symmetric with the used-width
        // fixpoint — a later row's specified width must not push the
        // preferred width past the used width. First-row specified
        // cells already pinned their columns above, so this floor only
        // ever fires for unpinned columns.
        // A vertical table's inline axis is the physical vertical axis:
        // the cell's inline spec is its `height` (see
        // `cell_specified_inline_height`).
        let vertical = grid.vertical;
        let spec = table_col_spec(inline_axis_spec(&n.data, vertical));
        let specified = match spec {
            SpecLen::Auto => 0.0,
            _ if fixed && cell.row != first_row => 0.0,
            _ if vertical => cell_specified_inline_height(forest, cell.node),
            _ => n.metrics.content_w,
        };
        let m = &n.metrics;
        let pad = cell_h_pad(m, grid.collapse, grid.vertical);
        // The preferred width is also the shrink-to-fit width, so the
        // cell's `min-width` floors it: a preferred width below the floor
        // would size the table narrower than the cell allows. Percentages
        // have no definite base this early and resolve against zero, the
        // same base the column widths above use.
        // §3.8.3 outer max-content (automatic layout): a definite width
        // caps the contribution at `max(min-content, width)`; an auto
        // cell in a constrained column likewise contributes its
        // min-content instead of its max-content (see
        // `cell_outer_max_content`).
        let auto_in_constrained = col_constrained[cell.col] && matches!(spec, SpecLen::Auto);
        let outer = if !fixed && (matches!(spec, SpecLen::Px(_)) || auto_in_constrained) {
            // Transposed with the preferred measure above: the
            // definite-spec arm needs the vertical min-content.
            let mcw = if grid.vertical {
                vertical_cell_min_content(forest, cell.node, fonts, &mut mcw_scratch, &mut int_bufs)
            } else {
                min_content_width_pooled(forest, cell.node, fonts, &mut mcw_scratch, &mut int_bufs)
            };
            cell_outer_max_content(
                spec,
                w.max(specified),
                mcw,
                specified,
                col_constrained[cell.col],
            )
        } else {
            w.max(specified)
        };
        let need = (outer + pad).max(cell_min_width_bb(forest, cell.node, 0.0, grid.vertical));
        if need > col_pref[cell.col] {
            col_pref[cell.col] = need;
        }
    }
    let (_, spacing_h, _) = table_border_model(forest, idx);
    // Collapsed columns are removed from the display: they draw no
    // inter-column spacing either (CSS 2.1 §17.5). With no visible
    // columns at all there is no spacing either (the `(nvis + 1)`
    // formula would otherwise leave one phantom gap).
    let nvis_pref = grid.collapsed_cols.iter().filter(|&&c| !c).count();
    let spacing_total = if nvis_pref == 0 {
        0.0
    } else {
        spacing_h * (nvis_pref as f32 + 1.0)
    };
    // The span-1 preferred widths snapshot: the intrinsic percentage
    // floor below is defined against the non-spanning measures
    // (§3.8.3), before any spanning cell is taken into account.
    let col_pref_nonspanning = col_pref.clone();
    // The widest spanning cell also floors the table's preferred
    // width directly: a cell whose percentage failed to redistribute
    // (T4 skip) sizes no column, but the table must still fit its
    // content — the floor is the cell's border-box need minus the
    // internal spacing its span covers.
    // Spanning cells, in increasing span order with per-column
    // largest-contribution merging (§3.8.3), distributing their
    // excess per §3.9.3.2 for intrinsic width calculation — mirroring
    // `layout_table`'s maximum side. Pinned (fixed-layout specified)
    // columns keep their width; the excess spreads over the remaining
    // spanned columns only.
    let mut spanning_floor = 0.0f32;
    {
        let mut spanning: Vec<&GridCell> = grid
            .cells
            .iter()
            .filter(|c| c.colspan > 1 && cell_width_contributes(&grid.collapsed_cols, c))
            .collect();
        spanning.sort_by_key(|c| c.colspan);
        // Scratch reused by every `distribute_spanning_max` call below:
        // no per-cell allocation after the first.
        let mut cand_scratch = SpanningCandidateScratch::default();
        // Running intrinsic percentages (§3.8.3), weighted by the
        // span-1 preferred widths like the percentage floor below:
        // each cell's excess distributes against the percentages as
        // of before its own redistribution, in (span, DOM) order.
        let mut pct_state = PctRunningState::new(forest, grid, col_pref_nonspanning.clone());
        for cell in spanning {
            // Vertical table: measure the cell's vertical content size
            // (see the single-column loop above).
            let w = if grid.vertical {
                vertical_cell_max_content(
                    forest,
                    cell.node,
                    fonts,
                    cache,
                    &mut mcw_scratch,
                    &mut int_bufs,
                    &mut int_inner,
                )
            } else {
                preferred_width_mcw(
                    forest,
                    cell.node,
                    fonts,
                    cache,
                    &mut mcw_scratch,
                    &mut int_bufs,
                    &mut int_inner,
                )
            };
            // Same first-row-only rule as the single-column loop
            // above: a later row's specified width must not floor
            // the spanning maximum through `w` (css-sizing-3 §5.2
            // definite-width contribution); use the content-based
            // width instead. The gate reads the inline-axis spec
            // (transposed for vertical tables); the borrow ends before
            // the fallback below, which needs `&mut forest`.
            let non_auto_spec = {
                let n = &forest.nodes[cell.node];
                !matches!(
                    table_col_spec(inline_axis_spec(&n.data, grid.vertical)),
                    SpecLen::Auto
                )
            };
            let w = if fixed && cell.row != first_row && non_auto_spec {
                // Transposed with the content measure above: a vertical
                // table's content-based width is its vertical max-content.
                if grid.vertical {
                    vertical_cell_max_content(
                        forest,
                        cell.node,
                        fonts,
                        cache,
                        &mut mcw_scratch,
                        &mut int_bufs,
                        &mut int_inner,
                    )
                } else {
                    generic_preferred_width(forest, cell.node)
                }
            } else {
                w
            };
            // An infinite cell width (a contained automatic-layout table
            // at the 100% percentage floor) must not poison this table's
            // preferred width: the table contributes its finite
            // column-maximum sum instead (see `finite_cell_max_content`).
            // The table's own preferred width still keeps the floor — only
            // its contribution as cell content is finite — so a standalone
            // block/float table still clamps to the available width.
            let w = if !grid.vertical && w.is_infinite() {
                finite_cell_max_content(
                    forest,
                    cell.node,
                    fonts,
                    cache,
                    &mut mcw_scratch,
                    &mut int_bufs,
                    &mut int_inner,
                )
                .unwrap_or(w)
            } else {
                w
            };
            let n = &forest.nodes[cell.node];
            // Specified cell widths feed the spanning maximum per
            // the §3.8.3 outer max-content rule; see the
            // single-column loop above. Under `table-layout:
            // fixed` only the first row counts (§17.5.2.1).
            // Transposed inline spec for vertical tables (see the
            // single-column loop above).
            let vertical = grid.vertical;
            let spec = table_col_spec(inline_axis_spec(&n.data, vertical));
            let specified = match spec {
                SpecLen::Auto => 0.0,
                _ if fixed && cell.row != first_row => 0.0,
                _ if vertical => cell_specified_inline_height(forest, cell.node),
                _ => n.metrics.content_w,
            };
            let m = &n.metrics;
            let pad = cell_h_pad(m, grid.collapse, grid.vertical);
            let outer = if !fixed && matches!(spec, SpecLen::Px(_)) {
                // Transposed with the preferred measure above.
                let mcw = if grid.vertical {
                    vertical_cell_min_content(
                        forest,
                        cell.node,
                        fonts,
                        &mut mcw_scratch,
                        &mut int_bufs,
                    )
                } else {
                    min_content_width_pooled(
                        forest,
                        cell.node,
                        fonts,
                        &mut mcw_scratch,
                        &mut int_bufs,
                    )
                };
                // Spanning cell: constrainedness is per-column, so the
                // constrained-column substitution does not apply here —
                // same as `auto_column_maxima`'s spanning loop.
                cell_outer_max_content(spec, w.max(specified), mcw, specified, false)
            } else {
                w.max(specified)
            };
            let need = (outer + pad).max(cell_min_width_bb(forest, cell.node, 0.0, grid.vertical));
            let end = (cell.col + cell.colspan).min(grid.ncols);
            // Mirror `layout_table`'s maximum side (and
            // `auto_column_minima`'s minimum side): a spanning
            // cell's border box covers the inter-column spacing
            // between the spanned columns, so the span under test
            // includes it — otherwise the spread double-counts
            // the internal gaps. Collapsed columns take no part
            // in either.
            // Allocation-free span probes replacing the old per-cell
            // `vis`/`open` Vecs: the visible-column count feeds the
            // gap, and an all-collapsed or all-pinned span skips the
            // distribution below (the open-column list itself is
            // rebuilt in scratch).
            let vis_len = (cell.col..end)
                .filter(|&cc| !grid.collapsed_cols[cc])
                .count();
            if vis_len == 0 {
                pct_state.apply_spanning_cell(forest, grid, cell);
                continue;
            }
            let gap = spacing_h * vis_len.saturating_sub(1) as f32;
            if (cell.col..end).all(|cc| col_pinned[cc]) {
                // The excess distribution is skipped for all-pinned
                // spans, but the §3.8.3 percentage redistribution
                // still runs for the cell, in (span, DOM) order.
                pct_state.apply_spanning_cell(forest, grid, cell);
                continue;
            }
            // A spanning cell with a specified width floors the table's
            // preferred width even when its percentage failed to
            // redistribute (T4); an auto-width cell whose excess was
            // dropped does not grow the table (T25: the 200px cell is
            // squeezed into the 50px columns).
            if !matches!(spec, SpecLen::Auto) {
                spanning_floor = spanning_floor.max(need - gap);
            }
            // Cumulative §3.9.3.2 distribution over the current
            // preferred widths (see `distribute_spanning_max`).
            let unconstrained_cell = matches!(spec, SpecLen::Auto);
            let cell_pct = match spec {
                SpecLen::Pct(p) => p / 100.0,
                _ => 0.0,
            };
            distribute_spanning_max(
                forest,
                grid,
                &mut col_pref,
                cell.col,
                end,
                need,
                gap,
                unconstrained_cell,
                cell_pct,
                Some(&col_pinned),
                Some(&pct_state.pct),
                &mut cand_scratch,
            );
            // The cell's own redistribution applies after its excess
            // distributed: the distribution above must not see it.
            pct_state.apply_spanning_cell(forest, grid, cell);
        }
    }
    // Under `table-layout: fixed` the first-row pins above can leave a
    // column's preferred (max-content) width below its min-content
    // minimum: the pin holds the specified width while
    // `auto_column_minima` measures the (possibly wider) content. The
    // shrink-to-fit arithmetic below needs max >= min per column, so
    // raise each maximum to its minimum after the spreads above
    // (keeping the widths non-negative). In the automatic layout the
    // repair must *not* run: the §3.9.3.2 intrinsic distribution can
    // legitimately raise a column's minimum past its maximum (the
    // whole min excess and the whole max excess both land on the
    // non-constrained columns, at different rates), and inflating the
    // preferred width to the minimum would overstate GRIDMAX (WPT
    // `colspan-redistribution` table 12: min 392 / max 342, preferred
    // table width from the maxima).
    let (col_max, col_max_nonspanning) = auto_column_maxima(
        forest,
        grid,
        &col_spec,
        spacing_h,
        fonts,
        cache,
        &mut mcw_scratch,
        &mut int_bufs,
        &mut int_inner,
    );
    let col_min = auto_column_minima(
        forest,
        grid,
        &col_spec,
        spacing_h,
        0.0,
        fonts,
        &col_max,
        &col_max_nonspanning,
    );
    if fixed {
        for (p, m) in col_pref.iter_mut().zip(col_min.iter()) {
            *p = p.max(*m).max(0.0);
        }
    }
    // Percentage floor on the intrinsic table width (automatic
    // layout), the max-content analog of the percentage sizing floor:
    // a percentage column behaves, for the table's preferred width,
    // as if its width were its percentage of the total — so with P
    // the total intrinsic percentage and F the sum of the
    // non-percentage columns' preferred widths, the assignable width
    // is at least F / (1 − P) (a 50% column beside a 66px column
    // implies a 132px table), and each percentage column's own
    // preferred width divided by its percentage floors it likewise.
    // WPT rule #1 (colspan-redistribution) states each percentage
    // column sets the floor for the maximum table width from its
    // *minimum* width: column.min_width / column.percent × 100.
    // Spanning-min redistribution (`distribute_spanning_min`,
    // all-percentage branch) can raise a percentage column's minimum
    // above its max-content (e.g. colspan >1 over percentage columns),
    // while the spanning max-content distribution deliberately skips
    // all-percentage spans — so the floor must see the redistributed
    // minimum: the per-column floor uses max(col_min, col_pref) / p.
    // The max with col_pref preserves the pure preferred-width floor
    // needed elsewhere (`table-width-redistribution` table 5:
    // width:fit-content, 20% column with max-content 75 / min 50
    // expects 75/0.2 + 32, not 50/0.2 + 32).
    // The floor raises the table's intrinsic width only; the columns
    // keep their own measures and the surplus distributes at
    // used-width time (WPT `colspan-redistribution` tables 1/14/16/17,
    // `column-widths` percentage cases, `table-width-redistribution`
    // table 20).
    // `width: max-content` skips the inverse-percentage floor entirely:
    // max-content is the pure column max-content sum — "max-content
    // does not allow for assignable size to be influenced by inverse
    // percent" (WPT `table-width-redistribution` table 4 expects
    // 275 + 32 = 307, not 75/0.2 + 32 = 407). `fit-content` keeps the
    // floor (table 5 expects 407).
    let pct_floor_total = if fixed || matches!(width_spec, SpecLen::MaxContent) {
        0.0
    } else {
        let (pcts, p_raw_total) = intrinsic_column_pcts(forest, grid, &col_pref_nonspanning);
        let p_total: f32 = pcts.iter().sum();
        let mut floor_assignable = 0.0f32;
        for (cc, &p) in pcts.iter().enumerate() {
            if p > 0.0 && !grid.collapsed_cols[cc] {
                floor_assignable = floor_assignable.max(col_min[cc].max(col_pref[cc]) / p);
            }
        }
        // WPT rule #2 (`colspan-redistribution` table 22): a genuine
        // 100% total intrinsic column percentage forces the maximum
        // table grid width to infinity — the F / (1 − P) rule diverges
        // at P = 100%. The infinity is explicit: dividing here would
        // hit 1.0 − p_total == 0 (NaN when every column is percentage
        // and f_pref is 0). The block-table used-width path turns INF
        // into the available width (`pref_c.min(avail)`); shrink-to-fit
        // sites saturate the same way. The test is on the *raw*
        // (pre-clamp) total: the §3.8.3 clamp pulls an over-constrained
        // total down to 100% too, and that case keeps the finite
        // normalized behavior (per-column floor above, fr-style
        // scaling downstream), so a floated over-constrained table
        // stays content-sized instead of exploding. The EPSILON band is
        // an admitted knife-edge, not exact equality: float noise (ten
        // 10% columns sum to ~0.9999999) makes an exact `== 1.0` test
        // miss genuine 100% totals, so the band has to catch them. As
        // a consequence, a total in (1.0, 1.0 + 1.2e-7] — `50% +
        // 50.00001%` — takes the INFINITY arm while `50% + 51%`
        // keeps the finite normalized behavior. Diverging on one side
        // of 1.0 and not the other is the price of not missing the
        // noisy-100% case; exact `==` would trade the knife-edge for
        // missed real divergences.
        if (p_raw_total - 1.0).abs() <= f32::EPSILON {
            floor_assignable = f32::INFINITY;
        } else if p_total > 0.0 && p_total < 1.0 {
            let f_pref: f32 = pcts
                .iter()
                .zip(col_pref.iter())
                .filter(|(p, _)| **p == 0.0)
                .map(|(_, m)| *m)
                .sum();
            floor_assignable = floor_assignable.max(f_pref / (1.0 - p_total));
        }
        floor_assignable + spacing_total
    };
    // Percentage widths resolve against the table's width. The first
    // pass resolved them against a zero base (no definite width
    // existed yet); now the table width is known, so iterate to a
    // fixpoint (bounded: column widths only ever grow here, and the
    // loop exits as soon as the total stops moving). The fixpoint
    // re-applies percentage `min-width` floors, percentage padding,
    // fixed-layout percentage pins, and non-percentage specified
    // widths; percentage *widths* in the automatic layout are
    // intrinsic percentages instead (§3.8.3) and contribute through
    // `pct_floor_total`, not as resolved px minima.
    let mut content_w: f32 = col_pref
        .iter()
        .sum::<f32>()
        .max(float_max)
        .max(spanning_floor);
    // Percentage participants for the second pass. With no percentage
    // specs and no percentage padding anywhere, every resolution below
    // is base-independent — the first pass already applied it — so the
    // loop is skipped entirely.
    let (pct_parts, pct_total) = pct_participants(forest, grid, fixed);
    if pct_parts.is_empty() && !has_pct_padding(forest, grid) {
        let intrinsic = (content_w + spacing_total)
            .max(cap_floor)
            .max(pct_floor_total);
        return table_pref_adjusted(
            forest,
            idx,
            fonts,
            intrinsic,
            spacing_total,
            &mut mcw_scratch,
            &mut int_bufs,
        );
    }
    // Over-constrained percentages (summing past 100%) would diverge:
    // each iteration resolves them against the growing total, so two
    // 60% columns (or one 150% column) grow geometrically to
    // astronomical widths. Normalize fr-unit style — scale every
    // participant so the percentages sum to 1, which makes the fixpoint
    // a contraction — and restore the specified values afterwards.
    let scale = if pct_total > 1.0 {
        1.0 / pct_total
    } else {
        1.0
    };
    let mut saved: Vec<(usize, bool, SpecLen)> = Vec::new();
    if scale < 1.0 {
        for &(node, is_min_width) in &pct_parts {
            let slot =
                inline_axis_spec_slot(&mut forest.nodes[node].data, grid.vertical, is_min_width);
            let orig = *slot;
            let scaled = match orig {
                SpecLen::Pct(v) => SpecLen::Pct(v * scale),
                SpecLen::Math {
                    pct,
                    px,
                    had_pct,
                    affine,
                } => SpecLen::Math {
                    pct: pct * scale,
                    px,
                    had_pct,
                    affine,
                },
                _ => continue,
            };
            *slot = scaled;
            saved.push((node, is_min_width, orig));
        }
    }
    // In fixed layout the column boxes pin their columns ahead of
    // first-row cells (§17.5.2.1); the second pass must not let a
    // percentage cell width override a column-box pin.
    let col_box_pinned: Vec<bool> = col_spec.0.iter().map(|w| w.is_some()).collect();
    // Columns whose effective specified width is a *percentage* (their
    // `col` box's, or their group's spread when they have no `col`
    // width of their own): in the automatic layout a percentage width
    // is an intrinsic percentage (§3.8.3), not a px minimum — it
    // contributes through the percentage floor on the table width and
    // the used-width sizing guesses, so the fixpoint below must not
    // re-apply it as a resolved px minimum (that double-counts: the
    // resolved width grows the total it resolves against).
    let col_pct_specified: Vec<bool> = grid
        .columns
        .iter()
        .map(|spec| {
            if let Some(n) = spec.col_node {
                pct_fraction(table_col_spec(inline_axis_spec(
                    &forest.nodes[n].data,
                    grid.vertical,
                )))
                .is_some()
            } else if let Some(g) = spec.group_node {
                pct_fraction(table_col_spec(inline_axis_spec(
                    &forest.nodes[g].data,
                    grid.vertical,
                )))
                .is_some()
            } else {
                false
            }
        })
        .collect();
    // The percentage base is the table width — content plus spacing
    // (border-spacing sits inside the table box in the separated model).
    let mut prev_delta = f32::INFINITY;
    for _ in 0..32 {
        let mut grew = false;
        let table_w = content_w + spacing_total;
        // Percentage padding resolves against the same base, so
        // re-resolve cell metrics before reading border-box widths.
        // Skipped for cells that will never lay out (see above).
        for cell in &grid.cells {
            if cell_width_contributes(&grid.collapsed_cols, cell) {
                resolve_metrics(forest, cell.node, 0.0, table_w, false, None, false);
            }
        }
        // Specified `<col>` / `<colgroup>` widths re-resolved against the
        // table width: minima in the automatic layout, pins under fixed.
        // Collapsed columns keep their zero width. Percentage-sourced
        // widths are skipped in the automatic layout (see
        // `col_pct_specified` above); under fixed layout they pin.
        for (cc, w) in column_specified_widths(forest, grid, 0.0, table_w)
            .0
            .iter()
            .enumerate()
        {
            if let Some(bb) = w
                && !grid.collapsed_cols[cc]
                && (fixed || !col_pct_specified[cc])
                && *bb > col_pref[cc]
            {
                col_pref[cc] = *bb;
                grew = true;
            }
        }
        if fixed {
            // First-row percentage cell widths re-pin their columns
            // (§17.5.2.1); absolute widths pinned exactly in the first
            // pass, and column-box pins keep priority. Collapsed columns
            // keep their zero width. The first row is the first *visual*
            // row (see `first_visual_row`), not the first DOM row.
            for cell in grid.cells.iter().filter(|c| c.row == first_row) {
                // Cells that never lay out pin nothing (see above).
                if !cell_width_contributes(&grid.collapsed_cols, cell) {
                    continue;
                }
                let n = &forest.nodes[cell.node];
                if pct_fraction(inline_axis_spec(&n.data, grid.vertical)).is_none() {
                    continue;
                }
                let m = &n.metrics;
                // The pin is the border-box inline size: for vertical
                // tables the metrics hold the physical (block-axis)
                // width, so resolve the `height` percentage against the
                // inline base directly.
                let bb = if grid.vertical {
                    let d = &n.data;
                    let v = inline_axis_spec(d, true)
                        .resolve(
                            d.font_size,
                            forest.root_font_size,
                            forest.vp_w,
                            forest.vp_h,
                            table_w,
                        )
                        .unwrap_or(0.0)
                        .max(0.0);
                    let bp = m.bt + m.pt + m.pb + m.bb;
                    match d.box_sizing {
                        BoxSizing::BorderBox => v,
                        BoxSizing::ContentBox => v + bp,
                    }
                    .max(0.0)
                } else {
                    (m.content_w + m.bl + m.pl + m.pr + m.br).max(0.0)
                };
                let per = bb / cell.colspan.max(1) as f32;
                for cc in cell.col..(cell.col + cell.colspan).min(grid.ncols) {
                    if !col_box_pinned[cc] && !grid.collapsed_cols[cc] && per > col_pref[cc] {
                        col_pref[cc] = per;
                        grew = true;
                    }
                }
            }
        } else {
            // Percentage *cell* widths are not re-applied here in the
            // automatic layout: a percentage width is an intrinsic
            // percentage (§3.8.3), contributing through the percentage
            // floor on the table width (`pct_floor_total`) and the
            // used-width sizing guesses (§3.9.3), not as a px column
            // minimum resolved against a growing total — the old
            // fixpoint application double-counted, turning a 60%
            // column with a 100px minimum and a 100px fixed sibling
            // into a 310px table instead of the floored 274px (WPT
            // `colspan-redistribution` table 14). Percentage
            // `min-width` floors and percentage padding still
            // re-resolve below.
        }
        // Percentage `min-width` floors re-resolved against the table
        // width (the first pass used a zero base). Fixed-layout pins keep
        // their priority, matching the first pass. `table_w` is the
        // inline-axis size (the physical height for vertical tables),
        // the correct percentage base for the transposed `min-height`.
        for cell in &grid.cells {
            let floor = cell_min_width_bb(forest, cell.node, table_w, grid.vertical);
            if floor <= 0.0 {
                continue;
            }
            if cell.colspan == 1 {
                // Collapsed columns keep their zero width in both
                // layouts; fixed-layout pins keep their priority,
                // matching the first pass.
                if grid.collapsed_cols[cell.col] || (fixed && col_pinned[cell.col]) {
                    continue;
                }
                if floor > col_pref[cell.col] {
                    col_pref[cell.col] = floor;
                    grew = true;
                }
            } else {
                let end = (cell.col + cell.colspan).min(grid.ncols);
                // Same gap deduction as the maximum side (and
                // `auto_column_minima`): the spanning cell's border box
                // covers the internal inter-column spacing.
                let vis: Vec<usize> = (cell.col..end)
                    .filter(|&cc| !grid.collapsed_cols[cc])
                    .collect();
                if vis.is_empty() {
                    continue;
                }
                let gap = spacing_h * vis.len().saturating_sub(1) as f32;
                let span: f32 = vis.iter().map(|&cc| col_pref[cc]).sum::<f32>() + gap;
                if floor > span {
                    if fixed {
                        // Like the first pass: pinned columns keep their
                        // width; the excess spreads over the remaining
                        // open columns of the span (skipped only when
                        // none are open).
                        let open: Vec<usize> =
                            (cell.col..end).filter(|&cc| !col_pinned[cc]).collect();
                        if !open.is_empty() {
                            let add = (floor - span) / open.len() as f32;
                            for cc in open {
                                col_pref[cc] += add;
                            }
                            grew = true;
                        }
                    } else {
                        // Collapsed columns take no part in the spread.
                        let open: Vec<usize> = (cell.col..end)
                            .filter(|&cc| !grid.collapsed_cols[cc])
                            .collect();
                        if open.is_empty() {
                            continue;
                        }
                        let add = (floor - span) / open.len() as f32;
                        for cc in open {
                            col_pref[cc] += add;
                        }
                        grew = true;
                    }
                }
            }
        }
        let new_w: f32 = col_pref.iter().sum();
        let delta = new_w - content_w;
        if !grew || delta.abs() < 0.0001 {
            content_w = new_w;
            break;
        }
        // Non-shrinking growth means the percentages over-constrain the
        // table against a base that includes spacing (no finite fixpoint
        // exists): stop iterating instead of drifting.
        if delta >= prev_delta {
            content_w = new_w;
            break;
        }
        prev_delta = delta;
        content_w = new_w;
    }
    // Restore the specified percentages scaled down for normalization.
    for (node, is_min_width, orig) in saved {
        let slot = inline_axis_spec_slot(&mut forest.nodes[node].data, grid.vertical, is_min_width);
        *slot = orig;
    }
    // The spanning floor survives the fixpoint: a cell whose percentage
    // failed to redistribute sizes no column, but the table must still
    // fit its content.
    content_w = content_w.max(spanning_floor);
    let intrinsic = (content_w + spacing_total)
        .max(cap_floor)
        .max(pct_floor_total);
    // A definite-width fixed table's intrinsic contribution is its
    // definite width (CSS Tables 3 §3.9.3.1: over-constrained percentage
    // pins rescale against the assignable width instead of growing the
    // table): the fixpoint value above — which re-pins first-row
    // percentage cells against the growing table width — is discarded
    // for definite widths, so >100% pins can neither inflate nor
    // understate the contribution.
    table_pref_adjusted(
        forest,
        idx,
        fonts,
        intrinsic,
        spacing_total,
        &mut mcw_scratch,
        &mut int_bufs,
    )
}

// ---------------------------------------------------------------------------
// Public geometry API for the JS engine (`offsetWidth` and friends).
// ---------------------------------------------------------------------------
//
// The JS shim (see `bowser_js`) wires the CSSOM View geometry properties
// to these functions. They operate on the public [`LayoutBox`] tree plus
// the box's ancestor chain, so no layout internals leak across the crate
// boundary. They are re-exported from the crate root
// (`pub use layout::layout_table::{TableBoxGeometry, box_used_borders, table_box_geometry, table_offset_parent};`).
// The layout crate cannot see DOM tag names, so the `is_body` /
// `is_table_part` predicates are caller-supplied (the JS shim reads them
// off the document); they must identify the HTML `body` element and the
// HTML `td` / `th` / `table` elements respectively, not box types.

/// Measured geometry of one table-internal box (`table`,
/// `thead`/`tbody`/`tfoot`, `tr`, `td`/`th`), in the CSSOM View terms the
/// JS `offsetWidth` / `clientWidth` / `offsetLeft` family uses.
///
/// All values are px, unrounded: CSSOM View defines these properties as
/// `long`, so the JS binding rounds to the nearest integer on the way out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TableBoxGeometry {
    /// `offsetWidth`: the border-box width.
    pub offset_width: f32,
    /// `offsetHeight`: the border-box height.
    pub offset_height: f32,
    /// `clientWidth`: the padding-box width (content + padding). Bowser
    /// has no scrollbars yet, so no scrollbar gutter is subtracted.
    pub client_width: f32,
    /// `clientHeight`: the padding-box height (see [`Self::client_width`]).
    pub client_height: f32,
    /// `offsetLeft`: the border-box left edge relative to the
    /// [`table_offset_parent`]'s padding-box origin — or, when the offset
    /// parent is `None`, relative to the initial containing block origin
    /// (CSSOM View: the ICB-relative border edge when `offsetParent` is
    /// null).
    pub offset_left: f32,
    /// `offsetTop`: the border-box top edge, same reference as
    /// [`Self::offset_left`].
    pub offset_top: f32,
}

/// The CSSOM View `offsetParent` of a table-internal box.
///
/// `ancestors` is the box's ancestor chain ordered from the outermost box
/// (the initial containing block root) down to the direct parent.
/// `is_body` identifies the HTML `body` element's box; `is_table_part`
/// identifies boxes of HTML `td`, `th`, and `table` elements (the layout
/// crate cannot see tag names, so the caller supplies both).
///
/// Returns `None` when CSSOM View says `offsetParent` is null:
/// - the box is the root element (`ancestors` is empty),
/// - the box has no associated CSS layout box (`display: none`),
/// - the box is the HTML `body` element (spec step 1 — even when
///   `<html>` is positioned, `document.body.offsetParent` is null),
/// - the box's computed `position` is `fixed` and no ancestor
///   establishes a fixed-position containing block (see below).
///
/// Otherwise returns the nearest ancestor that is positioned (computed
/// `position` other than `static`), the `body` box, or — when the box
/// itself is `static` — a `td`/`th`/`table` box (CSSOM View step 2.2: the
/// retained static-element-inside-`td`/`th`/`table` rule, which the CSSOM
/// View TR kept — Chrome implements it and WPT's
/// `offsetParent_element_test.html` covers it). Only the HTML elements
/// qualify: a `div { display: table-cell }` is not a table part. Since
/// `body` is an ancestor of every other element, the search can only
/// fail for the root and `body` boxes themselves, which return `None`
/// above — so a static element with no positioned ancestor gets
/// `document.body`, never null. Boxes without a computed style
/// (anonymous boxes) are treated as `position: static` and never become
/// offset parents.
///
/// A `fixed` box is contained by the nearest ancestor that establishes
/// a fixed-position containing block — a non-`none` `transform`,
/// `filter`, `perspective`/`backdrop-filter`, a `will-change` naming one
/// of those, or `contain: paint | layout | strict | content` (CSS
/// Position 3 §2.1). Such an ancestor *is* the `offsetParent` (Chrome
/// agrees); with no such ancestor the containing block is the viewport
/// and `offsetParent` is null (CSSOM View step 3). This uses the same
/// [`crate::establishes_fixed_cb`] predicate the main abspos pass uses,
/// so the geometry API and layout agree on the fixed containing block.
pub fn table_offset_parent<'a>(
    node: &LayoutBox,
    ancestors: &[&'a LayoutBox],
    is_body: impl Fn(&LayoutBox) -> bool,
    is_table_part: impl Fn(&LayoutBox) -> bool,
) -> Option<&'a LayoutBox> {
    if ancestors.is_empty() || is_display_none(node) || is_body(node) {
        return None;
    }
    if node
        .style
        .as_deref()
        .is_some_and(|s| matches!(position_of(s), Position::Fixed))
    {
        // A fixed box is contained by the nearest ancestor that
        // establishes a fixed-position containing block — a non-`none`
        // `transform`, `filter`, `perspective`/`backdrop-filter`, a
        // `will-change` naming one of those, or `contain: paint |
        // layout | strict | content` (CSS Position 3 §2.1, CSS
        // Transforms 1 §3, Filter Effects 1 §8). That ancestor is the
        // `offsetParent`; with no such ancestor the containing block is
        // the viewport and `offsetParent` is null (CSSOM View §4.2 step
        // 3). The predicate is the same one the main abspos pass uses
        // to resolve fixed containing blocks (`resolve_containing_block`
        // in `layout_position` walks to the nearest ancestor satisfying
        // [`crate::establishes_fixed_cb`]).
        return ancestors
            .iter()
            .rev()
            .copied()
            .find(|a| a.style.as_deref().is_some_and(crate::establishes_fixed_cb));
    }
    let el_static = node
        .style
        .as_deref()
        .is_none_or(|s| matches!(position_of(s), Position::Static));
    ancestors
        .iter()
        .rev()
        .copied()
        .find(|a| is_positioned(a) || is_body(a) || (el_static && is_table_part(a)))
}

/// Measure a table-internal box for the JS `offsetWidth` / `offsetHeight` /
/// `clientWidth` / `clientHeight` / `offsetLeft` / `offsetTop` properties.
///
/// `node` is the box to measure; `ancestors` is its ancestor chain ordered
/// from the outermost box (the initial containing block root) down to the
/// direct parent — the same chain [`table_offset_parent`] takes, plus the
/// same `is_body` / `is_table_part` tag-name predicates it takes.
///
/// Box semantics, per CSSOM View:
/// - `offsetWidth`/`offsetHeight` read the border box ([`LayoutBox::rect`]
///   is always a border box in absolute page coordinates).
/// - `clientWidth`/`clientHeight` read the padding box: the border box
///   minus the used border widths. Under `border-collapse: collapse` the
///   per-cell winners on [`LayoutBox::collapsed_borders`] are the used
///   borders; otherwise the specified `border-*-width` lengths. Rows, row
///   groups, and columns carry no used borders outside the collapse
///   winners (layout never resolves borders onto their rects), so their
///   client size equals their border-box size.
/// - `offsetLeft`/`offsetTop` measure from the node's border-box edge to
///   the offset parent's padding-box origin; with no offset parent they
///   are ICB-relative. CSSOM View returns zero for the HTML `body` element
///   itself; this API cannot identify `body` from a [`LayoutBox`] alone,
///   so callers apply that rule.
///
/// Length resolution mirrors layout's own rules: px and absolute units are
/// exact, relative units resolve against the font-size chain and the
/// viewport derived from the outermost ancestor's rect, and anything
/// unresolvable (including the `thin`/`medium`/`thick` keywords, which
/// layout's `spec_len_of` likewise maps to zero) falls back to 0.
///
/// A `display: none` box (no associated CSS layout box) measures all
/// zeros.
pub fn table_box_geometry(
    node: &LayoutBox,
    ancestors: &[&LayoutBox],
    is_body: impl Fn(&LayoutBox) -> bool,
    is_table_part: impl Fn(&LayoutBox) -> bool,
) -> TableBoxGeometry {
    let zero = TableBoxGeometry {
        offset_width: 0.0,
        offset_height: 0.0,
        client_width: 0.0,
        client_height: 0.0,
        offset_left: 0.0,
        offset_top: 0.0,
    };
    if is_display_none(node) {
        return zero;
    }
    let [bt, br, bb, bl] = used_borders(node, ancestors);
    let (offset_left, offset_top) =
        match table_offset_parent(node, ancestors, is_body, is_table_part) {
            Some(op) => {
                // The offset parent's used borders resolve against *its own*
                // ancestor prefix, not the node's full chain: the node's
                // chain would walk `ancestors…node, op`, appending `op` after
                // the node, so `op`'s parent font size would be computed as
                // the node's font size (with `op` seen twice). A positioned
                // ancestor with e.g. `border-width: 0.5em` would get the
                // wrong used border.
                let op_prefix = ancestors
                    .iter()
                    .position(|a| std::ptr::eq(*a, op))
                    .map_or(ancestors, |i| &ancestors[..i]);
                let op_b = used_borders(op, op_prefix);
                (
                    node.rect.x - (op.rect.x + op_b[3]),
                    node.rect.y - (op.rect.y + op_b[0]),
                )
            }
            None => (node.rect.x, node.rect.y),
        };
    TableBoxGeometry {
        offset_width: node.rect.w,
        offset_height: node.rect.h,
        client_width: (node.rect.w - bl - br).max(0.0),
        client_height: (node.rect.h - bt - bb).max(0.0),
        offset_left,
        offset_top,
    }
}

/// Whether the box has no associated CSS layout box (`display: none`).
fn is_display_none(box_: &LayoutBox) -> bool {
    matches!(
        box_.style.as_deref().and_then(|s| s.get("display")),
        Some(CssValue::Keyword(k)) if k.eq_ignore_ascii_case("none")
    )
}

/// Whether the box can be an offset parent: its computed `position` is not
/// `static`. Boxes without a computed style (anonymous boxes) never can.
fn is_positioned(box_: &LayoutBox) -> bool {
    match box_.style.as_deref() {
        Some(style) => !matches!(position_of(style), Position::Static),
        None => false,
    }
}

/// Used border widths `[top, right, bottom, left]` in px for a laid-out box.
///
/// Under `border-collapse: collapse` the per-cell winners on
/// [`LayoutBox::collapsed_borders`] are the conflict-resolved widths, but
/// the *used* border on each edge is half the winning width: winners are
/// centered on the shared grid lines (CSS 2.1 §17.6.2.2), which is also
/// what layout stamps into the cell's used metrics. Otherwise the
/// specified `border-*-width` lengths, with `none`/`hidden` styles forcing
/// zero — mirroring `resolve_metrics`. Rows, row groups, and columns carry
/// no used borders outside the collapse winners (layout never resolves
/// borders onto their rects), so they measure zero here.
fn used_borders(box_: &LayoutBox, ancestors: &[&LayoutBox]) -> [f32; 4] {
    if let Some(winners) = box_.collapsed_borders.as_deref() {
        return [
            winners[0].width / 2.0,
            winners[1].width / 2.0,
            winners[2].width / 2.0,
            winners[3].width / 2.0,
        ];
    }
    if matches!(
        box_.box_type,
        BoxType::TableRow
            | BoxType::TableRowGroup
            | BoxType::TableColumn
            | BoxType::TableColumnGroup
    ) {
        return [0.0; 4];
    }
    let Some(style) = box_.style.as_deref() else {
        return [0.0; 4];
    };
    let (fs, root_fs, vp_w, vp_h) = font_context(box_, ancestors);
    const SIDES: [(u8, u8); 4] = [
        (props::BORDER_TOP_WIDTH, props::BORDER_TOP_STYLE),
        (props::BORDER_RIGHT_WIDTH, props::BORDER_RIGHT_STYLE),
        (props::BORDER_BOTTOM_WIDTH, props::BORDER_BOTTOM_STYLE),
        (props::BORDER_LEFT_WIDTH, props::BORDER_LEFT_STYLE),
    ];
    let mut out = [0.0f32; 4];
    for (i, (wprop, sprop)) in SIDES.iter().enumerate() {
        let hidden = matches!(
            style.get_by_index(*sprop),
            Some(CssValue::Keyword(k))
                if k.eq_ignore_ascii_case("hidden") || k.eq_ignore_ascii_case("none")
        );
        if hidden {
            continue;
        }
        // Borders never take percentages, so the base is irrelevant;
        // unresolvable values fall back to 0 exactly as `resolve_metrics`
        // does via `spec_len_of`.
        out[i] = spec_len_of(style.get_by_index(*wprop))
            .resolve(fs, root_fs, vp_w, vp_h, 0.0)
            .unwrap_or(0.0)
            .max(0.0);
    }
    out
}

/// Used border widths `[top, right, bottom, left]` in px for a laid-out
/// box, exposed for the JS engine's parse-phase geometry (`clientLeft`,
/// `clientTop`, scroll extents).
///
/// This is exactly the border set [`table_box_geometry`] subtracts when
/// computing `clientWidth`/`clientHeight`: the same collapsed-table
/// winner logic and the same ancestor-chain font-context resolution as
/// [`used_borders`], so the two geometry families can never disagree on
/// one element.
pub fn box_used_borders(node: &LayoutBox, ancestors: &[&LayoutBox]) -> [f32; 4] {
    used_borders(node, ancestors)
}

/// Font-size context for resolving relative border-width units: the node's
/// own font size, the root font size (the `rem` basis), and the viewport
/// (the `vw`/`vh` basis, read off the outermost ancestor's rect — the
/// initial containing block is the viewport).
fn font_context(node: &LayoutBox, ancestors: &[&LayoutBox]) -> (f32, f32, f32, f32) {
    let (vp_w, vp_h) = ancestors
        .first()
        .map(|r| (r.rect.w, r.rect.h))
        .unwrap_or((0.0, 0.0));
    let mut fs = 16.0;
    let mut root_fs = 16.0;
    let mut parent_fs = 16.0;
    let mut first = true;
    for b in ancestors.iter().copied().chain(std::iter::once(node)) {
        if let Some(style) = b.style.as_deref() {
            fs = resolve_font_size(style.get("font-size"), parent_fs, root_fs, vp_w, vp_h);
            if first {
                root_fs = fs;
            }
            parent_fs = fs;
        }
        first = false;
    }
    (fs, root_fs, vp_w, vp_h)
}

#[cfg(test)]
mod table_tests {
    use super::{
        MAX_TABLE_GRID_CELLS, PrefWidthCache, SectionKind, SpecLen, TableBoxGeometry, TableGrid,
        TableGridPool, WalkBufs, auto_column_maxima, auto_column_minima, collect_table_grid,
        collect_table_grid_into, column_specified_widths, first_visual_row, intrinsic_column_pcts,
        merge_fixed_redundant_columns, merge_fixed_redundant_rows, merge_redundant_columns_inner,
        resolve_collapsed_borders, row_group_section_kind, table_border_model, table_box_geometry,
        table_col_spec, table_offset_parent, table_pref_width, wrap_table_stray_content,
    };
    use crate::{CollapsedBorderStyle, FixedFontMetrics, Size, TABLE_TEST_RESET_CSS, layout};
    use bowser_css::Color;

    #[test]
    fn hostile_rowspan_colspan_stays_bounded() {
        // ~55KB of hostile spans: 1000 rows, each cell spanning
        // 1000x1000. Each axis alone clamps to 1000, but the product
        // would force a ~1e9-cell occupancy grid (~1GB); the total-area
        // cap keeps the grid within 1e6 cells.
        let mut html = String::from("<table>");
        for _ in 0..1000 {
            html.push_str("<tr><td rowspan=\"1000\" colspan=\"1000\">x</td></tr>");
        }
        html.push_str("</table>");
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet("");
        let l = layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics);
        // Completes without OOM and with finite geometry.
        let table = &l.children[0].children[0];
        let r = &table.rect;
        assert!(r.w.is_finite() && r.h.is_finite() && r.w >= 0.0 && r.h >= 0.0);
    }

    #[test]
    fn single_huge_span_cell_stays_bounded() {
        let (doc, _) = bowser_dom::Document::parse_html(
            "<table><tr><td rowspan=\"1000000\" colspan=\"1000000\">x</td></tr></table>".as_bytes(),
        );
        let ss = bowser_css::parse_stylesheet("");
        let l = layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics);
        let table = &l.children[0].children[0];
        let r = &table.rect;
        assert!(r.w.is_finite() && r.h.is_finite() && r.w >= 0.0 && r.h >= 0.0);
    }

    #[test]
    fn at_cap_boundary_extra_cell_is_dropped() {
        // B3: the table grid cap is hard at the boundary. 1001 cells
        // with `rowspan="1000"` in one row: the first 1000 each take one
        // column of a 1000x1000 grid (exactly MAX_TABLE_GRID_CELLS
        // cells); the 1001st finds no room (`room_cols == 0`) and is
        // dropped outright instead of forcing a 1001st column that would
        // push the occupancy grid past the cap.
        let mut html = String::from("<table><tr>");
        for _ in 0..1001 {
            html.push_str("<td rowspan=\"1000\">x</td>");
        }
        html.push_str("</tr></table>");
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet("");
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
        let mut stack = vec![root];
        let mut table = None;
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == crate::BoxType::Table {
                table = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let grid = collect_table_grid(&forest, table.expect("table box"));
        assert_eq!(grid.cells.len(), 1000);
        assert!(grid.nrows * grid.ncols <= MAX_TABLE_GRID_CELLS);
    }

    // ---- test helpers ----

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// Mode split (see `standards_html` in `tests.rs`): this helper
    /// does NOT inject a doctype, so doctype-less input parses as
    /// full quirks mode. Pass an explicit doctype when the fixture's
    /// mode matters.
    fn layout_table_html(html: &str, css: &str) -> crate::LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
        layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics)
    }

    /// Like [`layout_table_html`], but also returning the parsed document
    /// so geometry tests can build tag-name predicates for
    /// [`table_offset_parent`] / [`table_box_geometry`].
    ///
    /// Mode split (see `standards_html` in `tests.rs`): no doctype
    /// is injected, so doctype-less input parses as full quirks
    /// mode. Pass an explicit doctype when the fixture's mode
    /// matters.
    fn layout_table_doc(html: &str, css: &str) -> (bowser_dom::Document, crate::LayoutBox) {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
        let l = layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics);
        (doc, l)
    }

    /// Whether the box's DOM node is an element with the given tag name.
    /// Boxes with no node (anonymous boxes) never match.
    fn box_has_tag(doc: &bowser_dom::Document, b: &crate::LayoutBox, tag: &str) -> bool {
        b.node.is_some_and(
            |n| matches!(&doc.kind(n), bowser_dom::NodeKind::Element(e) if e.tag_name == tag),
        )
    }

    /// Tag-name predicates for the geometry API: the HTML `body` element,
    /// and the HTML `td` / `th` / `table` elements. Tag names, not box
    /// types — a `div { display: table-cell }` is not a table part per
    /// CSSOM View.
    fn geom_preds(
        doc: &bowser_dom::Document,
    ) -> (
        impl Fn(&crate::LayoutBox) -> bool + '_,
        impl Fn(&crate::LayoutBox) -> bool + '_,
    ) {
        (
            move |b: &crate::LayoutBox| box_has_tag(doc, b, "body"),
            move |b: &crate::LayoutBox| {
                ["td", "th", "table"].iter().any(|t| box_has_tag(doc, b, t))
            },
        )
    }

    /// Navigate a laid-out table: body > table > row-group > row > cell.
    fn cell_at(l: &crate::LayoutBox, row: usize, col: usize) -> &crate::LayoutBox {
        &row_at(l, row).children[col]
    }

    /// Find the first descendant box with border-box width `w`.
    fn find_w(l: &crate::LayoutBox, w: f32) -> &crate::LayoutBox {
        let mut stack = vec![l];
        while let Some(b) = stack.pop() {
            if approx(b.rect.w, w) {
                return b;
            }
            stack.extend(b.children.iter());
        }
        panic!("no box with width {w}");
    }

    /// Navigate to a row: body > table > row-group > row.
    fn row_at(l: &crate::LayoutBox, row: usize) -> &crate::LayoutBox {
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        let group = table
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::TableRowGroup)
            .expect("row group");
        &group.children[row]
    }

    /// Navigate to the `gi`-th row group of a laid-out table.
    fn group_at(l: &crate::LayoutBox, gi: usize) -> &crate::LayoutBox {
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        table
            .children
            .iter()
            .filter(|c| c.box_type == crate::BoxType::TableRowGroup)
            .nth(gi)
            .expect("row group")
    }

    /// Navigate to row `row` of the `gi`-th row group.
    fn grouped_row_at(l: &crate::LayoutBox, gi: usize, row: usize) -> &crate::LayoutBox {
        &group_at(l, gi).children[row]
    }

    /// Box tree plus the table node, for tests that inspect forest-level
    /// data (collapsed borders) without running layout: border claims
    /// resolve from specified values, so no layout pass is needed.
    fn forest_table(html: &str, css: &str) -> (crate::boxes::Forest, usize) {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
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
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == crate::BoxType::Table {
                return (forest, i);
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        panic!("no table box");
    }

    fn collapsed_forest(html: &str, css: &str) -> crate::CollapsedBorders {
        let (forest, table) = forest_table(html, css);
        let grid = collect_table_grid(&forest, table);
        resolve_collapsed_borders(&forest, table, &grid, 800.0).expect("collapse")
    }

    const RED: Color = Color {
        r: 255,
        g: 0,
        b: 0,
        a: 255,
    };
    const BLUE: Color = Color {
        r: 0,
        g: 0,
        b: 255,
        a: 255,
    };
    const GREEN: Color = Color {
        r: 0,
        g: 128,
        b: 0,
        a: 255,
    };

    // ---- collapsed-border conflict resolution ----

    #[test]
    fn collapsed_border_wider_width_wins() {
        // Shared edge: cell 0's right border (5px red) vs cell 1's left
        // border (3px blue). The wider border wins on both sides of the
        // shared edge.
        let collapsed = collapsed_forest(
            "<table><tr><td>x</td><td>y</td></tr></table>",
            "table { border-collapse: collapse; } \
             td { border-right: 5px solid red; border-left: 3px solid blue; }",
        );
        let c0 = &collapsed.cells[0];
        assert!(approx(c0[1].width, 5.0));
        assert_eq!(c0[1].style, CollapsedBorderStyle::Solid);
        assert_eq!(c0[1].color, RED);
        let c1 = &collapsed.cells[1];
        assert!(approx(c1[3].width, 5.0));
        assert_eq!(c1[3].style, CollapsedBorderStyle::Solid);
        assert_eq!(c1[3].color, RED);
    }

    #[test]
    fn collapsed_border_style_rank_double_beats_solid_beats_dashed() {
        // Equal widths: double > solid > dashed (CSS 2.1 §17.6.2.1).
        let collapsed = collapsed_forest(
            "<table><tr><td class=\"a\">x</td><td>y</td><td class=\"c\">z</td></tr></table>",
            "table { border-collapse: collapse; } \
             td { border: 2px solid black; } \
             .a { border-right-style: double; } \
             .c { border-left-style: dashed; }",
        );
        // Edge A|B: 2px double vs 2px solid -> double.
        let a = &collapsed.cells[0];
        assert!(approx(a[1].width, 2.0));
        assert_eq!(a[1].style, CollapsedBorderStyle::Double);
        // Edge B|C: 2px solid vs 2px dashed -> solid.
        let c = &collapsed.cells[2];
        assert!(approx(c[3].width, 2.0));
        assert_eq!(c[3].style, CollapsedBorderStyle::Solid);
    }

    #[test]
    fn collapsed_border_cell_beats_table_on_outer_edge() {
        // Same width and style on an outer edge: the cell wins over the
        // table (cell > row > row-group > column > column-group > table),
        // so the winner carries the cell's color.
        let collapsed = collapsed_forest(
            "<table><tr><td>x</td></tr></table>",
            "table { border-collapse: collapse; border: 2px solid blue; } \
             td { border: 2px solid red; }",
        );
        let c0 = &collapsed.cells[0];
        assert!(c0.iter().all(|e| {
            approx(e.width, 2.0) && e.style == CollapsedBorderStyle::Solid && e.color == RED
        }));
        // Sanity: the losing table color is nowhere on the outer edges.
        assert!(collapsed.cells[0].iter().all(|e| e.color != BLUE));
    }

    #[test]
    fn collapsed_border_row_beats_table() {
        // The row's top border wins over the table's on the top outer
        // edge (row > table); the row claims only horizontal edges, so
        // the left outer edge stays the table's.
        let collapsed = collapsed_forest(
            "<table><tr><td>x</td></tr></table>",
            "table { border-collapse: collapse; border: 3px solid blue; } \
             tr { border-top: 3px solid green; }",
        );
        let c0 = &collapsed.cells[0];
        assert!(approx(c0[0].width, 3.0));
        assert_eq!(c0[0].style, CollapsedBorderStyle::Solid);
        assert_eq!(c0[0].color, GREEN);
        assert_eq!(c0[3].color, BLUE);
    }

    #[test]
    fn collapsed_border_hidden_suppresses_edge() {
        // `hidden` on one side of a shared edge suppresses the whole
        // edge, however wide the other side's border is.
        let collapsed = collapsed_forest(
            "<table><tr><td class=\"a\">x</td><td>y</td></tr></table>",
            "table { border-collapse: collapse; } \
             td { border-left: 4px solid red; border-right: 4px solid red; } \
             .a { border-right-style: hidden; }",
        );
        let a = &collapsed.cells[0];
        assert_eq!(a[1].style, CollapsedBorderStyle::Hidden);
        assert!(approx(a[1].width, 0.0));
        // The neighbor sees the same suppressed shared edge.
        let b = &collapsed.cells[1];
        assert_eq!(b[3].style, CollapsedBorderStyle::Hidden);
        assert!(approx(b[3].width, 0.0));
        // Unaffected edges keep their borders.
        assert!(approx(b[1].width, 4.0));
        assert_eq!(b[1].style, CollapsedBorderStyle::Solid);
    }

    #[test]
    fn collapsed_borders_none_when_separate() {
        let (forest, table) = forest_table("<table><tr><td>x</td></tr></table>", "");
        let grid = collect_table_grid(&forest, table);
        assert!(resolve_collapsed_borders(&forest, table, &grid, 800.0).is_none());
    }

    // ---- table-layout: fixed ----

    #[test]
    fn fixed_layout_first_row_widths_set_columns() {
        // The first row's specified width pins its column; the other
        // column takes the remaining table width. The long content in
        // column 0 does not widen it (no content measurement).
        let l = layout_table_html(
            "<table><tr><td class=\"a\">a much longer cell content here</td><td>y</td></tr></table>",
            "table { table-layout: fixed; width: 500px; } \
             td { font-size: 10px; } .a { width: 100px; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c0.rect.w, 100.0));
        assert!(approx(c1.rect.w, 400.0));
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 500.0));
    }

    #[test]
    fn fixed_layout_unspecified_columns_share_equally_ignoring_content() {
        // No specified widths: columns split the table evenly even
        // though the first cell's content is far wider.
        let l = layout_table_html(
            "<table><tr><td>a very long piece of text that would widen an auto column</td><td>b</td></tr></table>",
            "table { table-layout: fixed; width: 600px; } td { font-size: 10px; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c0.rect.w, 300.0));
        assert!(approx(c1.rect.w, 300.0));
    }

    #[test]
    fn fixed_layout_second_row_widths_are_ignored() {
        // Only the first row's widths count: a width on a later row's
        // cell does not move the column.
        let l = layout_table_html(
            "<table><tr><td>x</td><td>y</td></tr><tr><td class=\"w\">z</td><td>q</td></tr></table>",
            "table { table-layout: fixed; width: 400px; } \
             td { font-size: 10px; } .w { width: 300px; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c0.rect.w, 200.0));
        assert!(approx(c1.rect.w, 200.0));
    }

    #[test]
    fn fixed_layout_thead_pins_columns_despite_dom_order() {
        // "The first row" is the first *visual* row (CSS Tables 3
        // §2.1/§3.5.1): a `thead` placed after the `tbody` in DOM order
        // still pins the column widths (WPT
        // `css/css-tables/fixed-layout-1.html`, "takes visual order into
        // account, not dom order").
        let l = layout_table_html(
            "<table><tbody><tr><td>x</td><td>y</td></tr></tbody>\
             <thead><tr><td class=\"w\">z</td></tr></thead></table>",
            "table { table-layout: fixed; width: 100px; border-spacing: 0; } \
             td { font-size: 10px; padding: 0; } .w { width: 100px; }",
        );
        // The tbody is the first group in DOM order; its first cell
        // sits in column 0, pinned to 100px by the thead's cell.
        let c0 = grouped_row_at(&l, 0, 0).children[0].rect.w;
        assert!(approx(c0, 100.0), "visual-first-row pin {c0} != 100");
    }

    #[test]
    fn first_visual_row_prefers_thead_over_dom_first_row() {
        // Unit coverage for `first_visual_row`: with the `thead` last
        // in DOM order, the visual first row is the thead's row, not
        // grid row 0.
        let (forest, table) = forest_table(
            "<table><tbody><tr><td>x</td></tr></tbody>\
             <thead><tr><td>y</td></tr></thead></table>",
            "",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.nrows, 2);
        assert_eq!(first_visual_row(&forest, &grid), 1);
    }

    #[test]
    fn fixed_layout_redundant_row_tracks_merge() {
        // CSS Tables 3 §3.3.2: in fixed mode only rows are eligible for
        // track merging. One `rowspan=3` cell over three implicit row
        // tracks lays out as a single row (WPT
        // `css/css-tables/html5-table-formatting-fixed-layout-1.html`):
        // 3 zero-width columns + 4x25px spacing = 100px wide, 1
        // zero-height row + 2x25px spacing = 50px tall.
        let l = layout_table_html(
            "<table><tr><td rowspan=\"3\" colspan=\"3\" class=\"z\"></td></tr></table>",
            "table { table-layout: fixed; width: 0; border-spacing: 25px; } \
             td { font-size: 10px; padding: 0; } .z { height: 0; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 100.0), "width {}", table.rect.w);
        assert!(approx(table.rect.h, 50.0), "height {}", table.rect.h);
    }

    #[test]
    fn merge_fixed_redundant_rows_shrinks_span_and_count() {
        // Unit coverage for `merge_fixed_redundant_rows`: the grid-level
        // row count drops and the spanning cell's rowspan compensates.
        let (forest, table) = forest_table(
            "<table><tr><td rowspan=\"3\">x</td></tr></table>",
            "table { table-layout: fixed; width: 100px; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.nrows, 1);
        assert_eq!(grid.cells.len(), 1);
        assert_eq!(grid.cells[0].rowspan, 1);
    }

    #[test]
    fn merge_fixed_redundant_rows_pathological_input_stays_cheap() {
        // Perf regression test for `merge_fixed_redundant_rows`: many
        // cells with huge rowspans in fixed mode. The old implementation
        // re-scanned all placed cells and sorted the spanning set twice
        // per track — O(R * P log P) with two fresh allocations per
        // track. The incremental sweep is O(P log P + R) with a constant
        // number of allocations, so this asserts an allocation bound
        // (deterministic, cannot flake) plus a generous time backstop.
        const P: usize = 1024;
        const R: usize = 1000;
        let mut placed: Vec<(usize, usize, usize, usize, usize, usize)> =
            (0..P).map(|i| (i, 0, i, 1, R, 0)).collect();
        let row_nodes: Vec<Option<usize>> = Vec::new();
        let mut ends = Vec::new();
        let mut deleted = Vec::new();
        // Warm the scratch buffers outside the counted window: the
        // pooled steady state allocates nothing.
        merge_fixed_redundant_rows(&mut placed.clone(), &row_nodes, R, &mut ends, &mut deleted);
        let start = std::time::Instant::now();
        let allocs = count_allocations(|| {
            let nrows =
                merge_fixed_redundant_rows(&mut placed, &row_nodes, R, &mut ends, &mut deleted);
            assert_eq!(nrows, 1);
        });
        let elapsed = start.elapsed();
        eprintln!("pathological fixed-rowspan merge: {allocs} allocs in {elapsed:?}");
        // Correctness: every implicit track merged away and each cell's
        // rowspan shrank to 1.
        assert!(placed.iter().all(|&(_, _, _, _, rowspan, _)| rowspan == 1));
        // The pooled sweep allocates nothing in the steady state (the
        // scratch buffers are warmed above); the old per-call vecs made
        // ~2 allocations per track (~2000 here).
        assert_eq!(allocs, 0, "allocation regression: {allocs} allocs");
        // Backstop against algorithmic blowups; debug builds are slow,
        // so this is far above the expected sub-millisecond cost.
        assert!(
            elapsed < std::time::Duration::from_secs(30),
            "time regression: {elapsed:?}"
        );
    }

    #[test]
    fn merge_fixed_redundant_columns_shrinks_span_and_count() {
        // Unit coverage for `merge_fixed_redundant_columns`: the
        // column count drops, the spanning cell's colspan compensates,
        // and cells past the deleted tracks shift left. Tuples are
        // `(node, row, col, colspan, rowspan, _)`.
        let mut placed = vec![(0, 0, 0, 3, 1, 0), (1, 0, 3, 1, 1, 0)];
        let mut col_ranges: Vec<(usize, usize, usize)> = Vec::new();
        let mut group_ranges: Vec<(usize, usize, usize)> = Vec::new();
        let ncols =
            merge_fixed_redundant_columns(&mut placed, &mut col_ranges, &mut group_ranges, 4);
        assert_eq!(ncols, 2);
        assert_eq!(placed[0], (0, 0, 0, 1, 1, 0));
        assert_eq!(placed[1], (1, 0, 1, 1, 1, 0));
    }

    #[test]
    fn merge_fixed_redundant_columns_preserves_col_defined_tracks() {
        // Tracks covered by a `col` box are defined explicitly and never
        // merge (CSS Tables 3 §3.3.2): without the guard the two empty
        // spanned tracks would collapse and `ncols` would drop to 2.
        // The `col` range itself is unaffected (its tracks are never
        // deleted, so it only ever shifts, never shrinks).
        let mut placed = vec![(0, 0, 0, 3, 1, 0), (1, 0, 3, 1, 1, 0)];
        let mut col_ranges = vec![(9, 0, 3)];
        let mut group_ranges: Vec<(usize, usize, usize)> = Vec::new();
        let ncols =
            merge_fixed_redundant_columns(&mut placed, &mut col_ranges, &mut group_ranges, 4);
        assert_eq!(ncols, 4);
        assert_eq!(placed[0].3, 3);
        assert_eq!(placed[1].2, 3);
        assert_eq!(col_ranges, vec![(9, 0, 3)]);
    }

    #[test]
    fn merge_fixed_redundant_columns_remaps_colgroup_range() {
        // A `colgroup` box alone does not define its tracks, so empty
        // spanned tracks inside its span still merge — and the group
        // range (node, start, end) shrinks over the deleted tracks.
        let mut placed = vec![(0, 0, 0, 2, 1, 0), (1, 0, 2, 2, 1, 0)];
        let mut col_ranges: Vec<(usize, usize, usize)> = Vec::new();
        let mut group_ranges = vec![(7, 0, 4)];
        let ncols =
            merge_fixed_redundant_columns(&mut placed, &mut col_ranges, &mut group_ranges, 4);
        assert_eq!(ncols, 2);
        assert_eq!(placed[0].3, 1);
        assert_eq!(placed[1].2, 1);
        assert_eq!(placed[1].3, 1);
        assert_eq!(group_ranges, vec![(7, 0, 2)]);
    }

    #[test]
    fn merge_fixed_redundant_columns_pathological_input_stays_cheap() {
        // Perf regression test for `merge_fixed_redundant_columns`: many
        // cells with huge colspans. The incremental sweep
        // is O(P log P + C) with a constant number of allocations, so
        // this asserts an allocation bound (deterministic, cannot flake)
        // plus a generous time backstop, mirroring the row-half test.
        const P: usize = 1024;
        const C: usize = 1000;
        let mut placed: Vec<(usize, usize, usize, usize, usize, usize)> =
            (0..P).map(|i| (i, i, 0, C, 1, 0)).collect();
        let mut col_ranges: Vec<(usize, usize, usize)> = Vec::new();
        let mut group_ranges: Vec<(usize, usize, usize)> = Vec::new();
        let start = std::time::Instant::now();
        let allocs = count_allocations(|| {
            let ncols =
                merge_fixed_redundant_columns(&mut placed, &mut col_ranges, &mut group_ranges, C);
            assert_eq!(ncols, 1);
        });
        let elapsed = start.elapsed();
        eprintln!("pathological fixed-colspan merge: {allocs} allocs in {elapsed:?}");
        // Correctness: every implicit track merged away and each cell's
        // colspan shrank to 1.
        assert!(placed.iter().all(|&(_, _, _, colspan, _, _)| colspan == 1));
        // The sweep allocates the defined-track flags, the two sorted
        // track lists, and the mergeable-track list: a handful of
        // allocations, independent of the track count. The bound is
        // deliberately generous.
        assert!(allocs < 64, "allocation regression: {allocs} allocs");
        // Backstop against algorithmic blowups; debug builds are slow,
        // so this is far above the expected sub-millisecond cost.
        assert!(
            elapsed < std::time::Duration::from_secs(30),
            "time regression: {elapsed:?}"
        );
    }

    #[test]
    fn merge_fixed_redundant_columns_agrees_with_naive() {
        // Differential property test: the incremental sweep agrees
        // exactly with a spec-literal iterative implementation (delete
        // the first eligible track: no originating cell, and either the
        // track is empty or its spanning set matches the previous
        // track's; repeat until fixpoint) on randomized small grids,
        // including `col`-defined tracks. Deterministic xorshift seed,
        // so no flake risk.
        type PlacedCell = (usize, usize, usize, usize, usize, usize);
        fn naive(
            placed: &[PlacedCell],
            col_ranges: &[(usize, usize, usize)],
            ncols: usize,
        ) -> (Vec<PlacedCell>, usize) {
            let mut cells = placed.to_vec();
            let mut defined = vec![false; ncols];
            for &(_, start, span) in col_ranges {
                for d in defined
                    .iter_mut()
                    .take((start + span).min(ncols))
                    .skip(start)
                {
                    *d = true;
                }
            }
            let mut nc = ncols;
            loop {
                let spans = |tt: usize| {
                    cells
                        .iter()
                        .filter(|c| c.2 <= tt && tt < c.2 + c.3)
                        .map(|c| c.0)
                        .collect::<Vec<_>>()
                };
                let mut victim = None;
                // `defined.len() == nc` at the top of each pass, so this
                // visits exactly the live tracks `1..nc`.
                for (t, &is_defined) in defined.iter().enumerate().skip(1) {
                    if is_defined {
                        continue;
                    }
                    let originates = cells.iter().any(|c| c.2 == t);
                    if originates {
                        continue;
                    }
                    let st = spans(t);
                    // Hybrid merge rule (see
                    // `merge_redundant_columns_inner`): an empty track
                    // merges even when a cell ends at it; a non-empty
                    // track needs the previous track's exact spanning
                    // set.
                    if st.is_empty() || st == spans(t - 1) {
                        victim = Some(t);
                        break;
                    }
                }
                let Some(t) = victim else { break };
                for c in cells.iter_mut() {
                    if c.2 <= t && t < c.2 + c.3 {
                        c.3 -= 1;
                    }
                    if c.2 > t {
                        c.2 -= 1;
                    }
                }
                defined.remove(t);
                nc -= 1;
            }
            (cells, nc)
        }

        let mut rng: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for case in 0..2000 {
            let ncols = 1 + (next() % 8) as usize;
            let ncells = 1 + (next() % 12) as usize;
            let mut placed = Vec::new();
            for i in 0..ncells {
                let col = (next() % ncols as u64) as usize;
                let max_span = ncols - col;
                let colspan = 1 + (next() % max_span as u64) as usize;
                let row = (next() % 4) as usize;
                placed.push((i, row, col, colspan, 1, 0));
            }
            // Sometimes a `col` box covers a random span, exercising
            // the explicit-definition guard in both implementations.
            let mut col_ranges: Vec<(usize, usize, usize)> = Vec::new();
            if ncols > 1 && next() % 3 == 0 {
                let start = (next() % (ncols as u64 - 1)) as usize;
                let span = 1 + (next() % (ncols as u64 - start as u64)) as usize;
                col_ranges.push((usize::MAX, start, span));
            }
            let mut sweep_cells = placed.clone();
            let mut sweep_ranges = col_ranges.clone();
            let mut sweep_groups: Vec<(usize, usize, usize)> = Vec::new();
            let sweep_ncols = merge_fixed_redundant_columns(
                &mut sweep_cells,
                &mut sweep_ranges,
                &mut sweep_groups,
                ncols,
            );
            let (naive_cells, naive_ncols) = naive(&placed, &col_ranges, ncols);
            assert_eq!(sweep_ncols, naive_ncols, "case {case}: ncols");
            assert_eq!(sweep_cells, naive_cells, "case {case}: cells");
        }
    }

    #[test]
    fn fixed_layout_over_constrained_grows_table() {
        // Specified columns wider than the table: the table grows to
        // their sum (CSS 2.1 §17.5.2.1).
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td></tr></table>",
            "table { table-layout: fixed; width: 200px; } \
             td { font-size: 10px; } .a { width: 150px; } .b { width: 150px; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c0.rect.w, 150.0));
        assert!(approx(c1.rect.w, 150.0));
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 300.0));
    }

    #[test]
    fn fixed_layout_pct_pins_over_100_rescale_no_growth() {
        // CSS Tables 3 §3.9.3.2: percentage pins summing past 100% of
        // the assignable width rescale proportionally to fit exactly
        // with no table growth — two 60% columns in a 300px fixed table
        // pin at 150px each and the table stays 300px.
        let l = layout_table_html(
            "<table><colgroup><col class=\"a\"><col class=\"b\"></colgroup>\
             <tr><td>x</td><td>y</td></tr></table>",
            "table { table-layout: fixed; width: 300px; } \
             td { font-size: 10px; } .a { width: 60%; } .b { width: 60%; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c0.rect.w, 150.0));
        assert!(approx(c1.rect.w, 150.0));
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 300.0));
        assert!(approx(c0.rect.w + c1.rect.w, table.rect.w));
    }

    #[test]
    fn fixed_layout_three_pct_pins_rescale_proportionally() {
        // 50/40/30 sum to 120%: rescaling by 1/1.2 gives 125/100/75px,
        // tiling the 300px table with no growth.
        let l = layout_table_html(
            "<table><colgroup><col class=\"a\"><col class=\"b\"><col class=\"c\"></colgroup>\
             <tr><td>x</td><td>y</td><td>z</td></tr></table>",
            "table { table-layout: fixed; width: 300px; } \
             td { font-size: 10px; } .a { width: 50%; } .b { width: 40%; } .c { width: 30%; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 125.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 100.0));
        assert!(approx(cell_at(&l, 0, 2).rect.w, 75.0));
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 300.0));
    }

    #[test]
    fn fixed_layout_mixed_pins_fixed_keep_pct_absorb_no_growth() {
        // CSS Tables 3 §3.9.3.1: fixed-length pins are never rescaled.
        // 60% + 60% + 100px pins in a 300px fixed table overflow the
        // assignable width, so the 100px pin keeps its width and the
        // percentage pins absorb the remaining 200px proportionally
        // (100/100) — the table does not grow.
        let l = layout_table_html(
            "<table><colgroup><col class=\"a\"><col class=\"b\"><col class=\"c\"></colgroup>\
             <tr><td>x</td><td>y</td><td>z</td></tr></table>",
            "table { table-layout: fixed; width: 300px; } \
             td { font-size: 10px; } .a { width: 60%; } .b { width: 60%; } .c { width: 100px; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 100.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 100.0));
        assert!(approx(cell_at(&l, 0, 2).rect.w, 100.0));
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 300.0),
            "table must not grow past its specified width: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_pct_rescale_targets_assignable_width() {
        // CSS Tables 3 §3.9.3.1: over-100% percentage pins rescale so
        // their sum meets the *assignable* width (content minus spacing)
        // exactly. Two 60% columns in a 300px fixed table with 10px
        // border-spacing: assignable = 270, so the columns pin at 135px
        // each and the table stays 300px. Before the fix they rescaled
        // to the content width (150px each) and the table wrongly grew
        // to 330px to fit the spacing.
        let l = layout_table_html(
            "<table><colgroup><col class=\"a\"><col class=\"b\"></colgroup>\
             <tr><td>x</td><td>y</td></tr></table>",
            "table { table-layout: fixed; width: 300px; border-spacing: 10px; } \
             td { font-size: 10px; } .a { width: 60%; } .b { width: 60%; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 135.0),
            "c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 135.0),
            "c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 300.0),
            "table should not grow past its specified width: {}",
            table.rect.w
        );
    }

    #[test]
    fn border_spacing_pct_calc_invalid() {
        // Percentages are not valid `border-spacing` (CSS 2.1 §17.5.2):
        // a `calc()` with percentage leaves is invalid at
        // computed-value time (see the cascade validation), so the
        // spacing is the initial 0 and the columns sit adjacent.
        // `table_border_model` needs no `had_pct` gate: only lengths
        // reach it, and the percentage base it would use (the table's
        // content width, read before the table's own metrics resolve)
        // cannot feed back into the spacing.
        let l = layout_table_html(
            "<table><tr><td>x</td><td>y</td></tr></table>",
            "table { width: 300px; border-spacing: calc(5px + 10%); } td { font-size: 10px; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(
            approx(c1.rect.x, c0.rect.x + c0.rect.w),
            "columns not adjacent: c0.x={} c0.w={} c1.x={}",
            c0.rect.x,
            c0.rect.w,
            c1.rect.x
        );
    }

    #[test]
    fn border_spacing_em_and_huge_values() {
        // em-based spacing resolves against the table's font size, and
        // hostile magnitudes clamp to a finite bound (`sane_len` in
        // `SpecLen::resolve`) instead of poisoning geometry: no
        // overflow check beyond that clamp is needed.
        let l = layout_table_html(
            "<table><tr><td>x</td><td>y</td></tr></table>",
            "table { width: 300px; border-spacing: 2em; font-size: 16px; } td { font-size: 10px; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(
            approx(c1.rect.x - (c0.rect.x + c0.rect.w), 32.0),
            "gap={}",
            c1.rect.x - (c0.rect.x + c0.rect.w)
        );
        // Hostile magnitude: the table grows to fit spacing + columns
        // (spacing-starved path) but every rect stays finite.
        let l = layout_table_html(
            "<table><tr><td>x</td><td>y</td></tr></table>",
            "table { border-spacing: 1000000000000000000px; } td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(
            table.rect.w.is_finite() && table.rect.w > 0.0,
            "table.w={}",
            table.rect.w
        );
        let c1 = cell_at(&l, 0, 1);
        assert!(
            c1.rect.x.is_finite() && c1.rect.y.is_finite(),
            "c1=({}, {})",
            c1.rect.x,
            c1.rect.y
        );
    }

    #[test]
    fn fixed_layout_pct_rescale_exact_with_padded_pct_cells() {
        // Percentage pins are border-box, so the rescale denominator is
        // the pins' exact px (`Σ col_w` over the percentage pins), not
        // `pct_sum * content_w`: the product omits padding/border, the
        // scale overshoots, and the growth path fires spuriously. Two
        // 60% cells with 10px padding in a 300px fixed table with 10px
        // border-spacing pin at 182px each (60% of the 270px assignable
        // width, rebased from the spacing-inclusive 300px base, plus
        // 20px padding); the assignable width rescales them to 135px
        // each and the table stays 300px. Before the fixes the product
        // (360px) overshot the scale and the table grew to 330px.
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"a\">y</td></tr></table>",
            "table { table-layout: fixed; width: 300px; border-spacing: 10px; } \
             td { font-size: 10px; padding: 10px; } td.a { width: 60%; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 135.0),
            "c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 135.0),
            "c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 300.0),
            "table should not grow past its specified width: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_mixed_pins_with_spacing_fixed_keep_pct_absorb() {
        // CSS Tables 3 §3.9.3.1 with border-spacing: 60% + 60% + 100px
        // pins in a 300px fixed table with 10px spacing (assignable
        // 260px). The 100px pin keeps its width; the percentage pins
        // absorb the remaining 160px proportionally (80/80) and the
        // table stays 300px.
        let l = layout_table_html(
            "<table><colgroup><col class=\"a\"><col class=\"b\"><col class=\"c\"></colgroup>\
             <tr><td>x</td><td>y</td><td>z</td></tr></table>",
            "table { table-layout: fixed; width: 300px; border-spacing: 10px; } \
             td { font-size: 10px; } .a { width: 60%; } .b { width: 60%; } .c { width: 100px; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 80.0),
            "c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 80.0),
            "c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 2).rect.w, 100.0),
            "c2={}",
            cell_at(&l, 0, 2).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 300.0),
            "table must not grow past its specified width: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_pct_resolves_against_assignable_width() {
        // WPT table 20 (`table-width-redistribution-fixed`): percentage
        // pins resolve against the assignable width (content minus
        // border-spacing), not the spacing-inclusive content width. A
        // 50% cell + 30px cell + auto cell in a 132px fixed table with
        // 8px border-spacing (assignable 100px) pins at 50/30; the open
        // column takes the 20px remainder, no growth.
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td><td>z</td></tr></table>",
            "table { table-layout: fixed; width: 132px; border-spacing: 8px; } \
             td { font-size: 10px; } .a { width: 50%; } .b { width: 30px; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 50.0),
            "c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 30.0),
            "c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 2).rect.w, 20.0),
            "c2={}",
            cell_at(&l, 0, 2).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 132.0),
            "table must not grow past its specified width: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_pct_plus_fixed_pins_tile_without_growth() {
        // WPT table 21 (`table-width-redistribution-fixed`): a 50% cell
        // + 50px cell in a 124px fixed table with 8px border-spacing
        // (assignable 100px) pin at 50/50 exactly; no table growth.
        // Before the fix the 50% resolved against the 124px content
        // width (62px), the pins overflowed, and the table grew to
        // 136px.
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td></tr></table>",
            "table { table-layout: fixed; width: 124px; border-spacing: 8px; } \
             td { font-size: 10px; } .a { width: 50%; } .b { width: 50px; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 50.0),
            "c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 50.0),
            "c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 124.0),
            "table must not grow past its specified width: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_overconstrained_mixed_pins_pct_absorb_proportionally() {
        // WPT table 22 (`table-width-redistribution-fixed`): 20% + 60%
        // + 60px pins in a 132px fixed table with 8px border-spacing
        // (assignable 100px) are over-constrained. The 60px pin keeps
        // its width; the percentages absorb the remaining 40px in a
        // 20:60 ratio (10/30) and the table stays 132px. Before the fix
        // the table grew to ~198px.
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td><td class=\"c\">z</td></tr></table>",
            "table { table-layout: fixed; width: 132px; border-spacing: 8px; } \
             td { font-size: 10px; } .a { width: 20%; } .b { width: 60%; } .c { width: 60px; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 10.0),
            "c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 30.0),
            "c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 2).rect.w, 60.0),
            "c2={}",
            cell_at(&l, 0, 2).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 132.0),
            "table must not grow past its specified width: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_mixed_pins_fixed_alone_over_avail_pct_resolves_zero() {
        // B3: the fixed pins alone already meet/exceed the assignable
        // width, so there is no remainder for the percentage pins to
        // absorb: the 50% pin resolves to 0 (its percentage part) and
        // the table grows only to the 150px fixed pin, not to
        // 150 + 50 = 200. Before the fix the percentage pin stayed
        // full-size in this branch and the table grew to 200px.
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td></tr></table>",
            "table { table-layout: fixed; width: 100px; } \
             td { font-size: 10px; } .a { width: 150px; } .b { width: 50%; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 150.0),
            "c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 0.0),
            "pct pin must resolve to 0: c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 150.0),
            "table grows only to the fixed pin: {}",
            table.rect.w
        );
        // Same shape with border-spacing: the table grows to
        // 150 + spacing (3 * 8px), not 150 + 50 + spacing. Assignable
        // width is 124 - 24 = 100px, so the 50% pin is 50px pre-fix.
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td></tr></table>",
            "table { table-layout: fixed; width: 124px; border-spacing: 8px; } \
             td { font-size: 10px; } .a { width: 150px; } .b { width: 50%; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 150.0));
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 0.0),
            "pct pin must resolve to 0 with spacing: c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 174.0),
            "table grows only to fixed pin + spacing: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_pct_pins_summing_100_no_spacing_double_count() {
        // WPT tables 16/18 (`table-width-redistribution-fixed`):
        // percentage-only pins summing to exactly 100% resolve against
        // the assignable width, so border-spacing is not double-counted.
        // Two 50% cells in a 124px fixed table with 8px border-spacing
        // (assignable 100px) pin at 50px each and the table stays
        // 124px; before the fix they pinned at 62px and the table grew
        // to 148px.
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"a\">y</td></tr></table>",
            "table { table-layout: fixed; width: 124px; border-spacing: 8px; } \
             td { font-size: 10px; } .a { width: 50%; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 50.0),
            "c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 50.0),
            "c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 124.0),
            "table must not grow past its specified width: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_fixed_pins_alone_overflow_still_grows() {
        // The CSS 2.1 §17.5.2.1 growth path survives the strict
        // §3.9.3.1 rule: fixed-length pins alone overflowing the table
        // still grow it (there is no percentage width to absorb the
        // over-constraint).
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td></tr></table>",
            "table { table-layout: fixed; width: 124px; border-spacing: 8px; } \
             td { font-size: 10px; } .a { width: 70px; } .b { width: 60px; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 70.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 60.0));
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 154.0),
            "fixed pins 70+60 with 24px spacing must grow the table: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_spanning_length_pin_subtracts_internal_spacing() {
        // WPT `table-width-redistribution-fixed` table 12: a spanning
        // length pin covers its columns' border boxes *plus* the
        // internal border-spacing gaps, so the per-column pin is
        // `(pin - (colspan - 1) * spacing) / colspan`, not `pin /
        // colspan`. 108px over 2 columns with 8px spacing pins 50px
        // per column (not 54); 208px pins 100px. The 300px excess then
        // distributes proportionally to the pinned lengths
        // (§3.9.3.2 rule 2): 100/100/200/200, tiling the 640px table.
        // Before the fix the pins were 54/54/104/104 and the first
        // spanning cell came out 213px instead of 208px.
        let l = layout_table_html(
            "<table><tr><td colspan=\"2\" class=\"a\">x</td><td colspan=\"2\" class=\"b\">y</td></tr>\
             <tr><td>1</td><td>2</td><td>3</td><td>4</td></tr></table>",
            "table { table-layout: fixed; width: 640px; border-spacing: 8px; } \
             td { font-size: 10px; } .a { width: 108px; } .b { width: 208px; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 208.0),
            "spanning cell c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 408.0),
            "spanning cell c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        assert!(approx(cell_at(&l, 1, 0).rect.w, 100.0));
        assert!(approx(cell_at(&l, 1, 1).rect.w, 100.0));
        assert!(approx(cell_at(&l, 1, 2).rect.w, 200.0));
        assert!(approx(cell_at(&l, 1, 3).rect.w, 200.0));
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 640.0),
            "table must not grow: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_spanning_length_pin_subtracts_internal_spacing_padded() {
        // WPT `table-width-redistribution-fixed-padding` table 7: the
        // padded variant of the above. 96px content-box over 2 columns
        // with 6px padding is a 108px border-box pin; minus the 8px
        // internal gap that is 50px per column (not 54), 196px pins
        // 100px per column. Columns tile the 300px assignable width
        // exactly, so the table stays 340px. Before the fix the pins
        // were 54/54/104/104 and the table grew to 356px.
        let l = layout_table_html(
            "<table><tr><td colspan=\"2\" class=\"a\">x</td><td colspan=\"2\" class=\"b\">y</td></tr>\
             <tr><td>1</td><td>2</td><td>3</td><td>4</td></tr></table>",
            "table { table-layout: fixed; width: 340px; border-spacing: 8px; } \
             td { font-size: 10px; padding: 6px; box-sizing: content-box; } \
             .a { width: 96px; } .b { width: 196px; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 108.0),
            "spanning cell c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 208.0),
            "spanning cell c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        assert!(
            approx(cell_at(&l, 1, 0).rect.w, 50.0),
            "c0={}",
            cell_at(&l, 1, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 1, 2).rect.w, 100.0),
            "c2={}",
            cell_at(&l, 1, 2).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 340.0),
            "table must not grow: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_spanning_pct_pin_ignores_padding() {
        // WPT `table-width-redistribution-fixed-padding` table 8: CSS
        // Tables 3 §3.9.3.1 — "cells ignore their border and padding
        // size if their width is a percentage (box-sizing is ignored)".
        // A 40%-over-2-columns pin resolves each column to 20% of the
        // 400px assignable width (80px), padding absorbed; the spanning
        // cell covers its columns plus the internal gap (168px), not
        // the percentage-resolved border box plus padding. Likewise
        // 20% pins 40px per column and the border-box 40% cell pins
        // 160px. Columns tile the assignable width exactly: no rescale,
        // no growth; the table stays 448px. Before the fix the pins
        // kept their padding and the first spanning cell came out
        // 170px instead of 168px.
        let l = layout_table_html(
            "<table><tr><td colspan=\"2\" class=\"a\">x</td><td colspan=\"2\" class=\"b\">y</td>\
             <td class=\"c\">z</td></tr>\
             <tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr></table>",
            "table { table-layout: fixed; width: 448px; border-spacing: 8px; } \
             td { font-size: 10px; padding: 6px; box-sizing: content-box; } \
             .a { width: 40%; } .b { width: 20%; } .c { width: 40%; box-sizing: border-box; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 168.0),
            "40% spanning cell c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 88.0),
            "20% spanning cell c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 2).rect.w, 160.0),
            "40% border-box cell c2={}",
            cell_at(&l, 0, 2).rect.w
        );
        for (c, w) in [80.0, 80.0, 40.0, 40.0, 160.0].iter().enumerate() {
            assert!(
                approx(cell_at(&l, 1, c).rect.w, *w),
                "row2 c{c}={}",
                cell_at(&l, 1, c).rect.w
            );
        }
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 448.0),
            "table must not grow: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_pct_rescale_reapplies_min_width_floor() {
        // A CSS `min-width` on a percentage pin is a floor on the
        // rescaled column: two 60% pins in a 300px fixed table pin at
        // 180px each (the first already floored to its 200px minimum
        // by `resolve_metrics`), rescale to 300/380 of that —
        // 157.9/142.1 — and the floor is re-applied, so the first
        // column stays 200px instead of shrinking to 157.9px. The
        // floored sum (342.1) then grows the table via the usual
        // over-constrained path. Before the fix the multiply dropped
        // the floor and the column came out 157.9px.
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td></tr></table>",
            "table { table-layout: fixed; width: 300px; } \
             td { font-size: 10px; } .a { width: 60%; min-width: 200px; } .b { width: 60%; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 200.0),
            "min-width floors the rescaled pin: c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 142.11),
            "c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 342.11),
            "floored columns grow the table: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_mixed_pct_fixed_overconstrained_padded() {
        // WPT `table-width-redistribution-fixed-padding` table 15:
        // 20% + 60% + 60px pins in a 132px fixed table with 8px
        // border-spacing and 6px cell padding (assignable 100px).
        // Fixed-length pins are never rescaled (§3.9.3.1): the 60px
        // content-box pin keeps 72px border-box, and the percentage
        // pins absorb the remaining 28px proportionally — 8.6px and
        // 19.4px — with no table growth. A cell's border box is its
        // spanned grid width exactly, so the 8.6px column's cell
        // measures 8.6px (not 12px of padding): the padding does not
        // inflate the box past the grid.
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td><td class=\"c\">z</td></tr></table>",
            "table { table-layout: fixed; width: 132px; border-spacing: 8px; } \
             td { font-size: 10px; padding: 6px; box-sizing: content-box; } \
             .a { width: 20%; } .b { width: 60%; } .c { width: 60px; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 8.62),
            "c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 19.38),
            "c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 2).rect.w, 72.0),
            "c2={}",
            cell_at(&l, 0, 2).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 132.0),
            "table must not grow: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_cell_border_box_clamped_to_grid_width() {
        // WPT `table-width-redistribution-fixed-padding` tables 6 and
        // 9: a cell's border box is its spanned grid width exactly —
        // padding never inflates it past the grid. A 0-width fixed
        // column's padded cell measures 0px (not 12px), and a
        // colspan=2 auto cell over two 0-width columns measures just
        // the 8px internal gap (not 12px).
        let l = layout_table_html(
            "<table><tr><td class=\"a\"><div>x</div></td><td class=\"b\">y</td></tr>\
             <tr><td colspan=\"2\" class=\"c\">z</td></tr></table>",
            "table { table-layout: fixed; width: 1px; border-spacing: 8px; } \
             td { font-size: 10px; padding: 6px; box-sizing: content-box; } \
             .a div { width: 50px; } .b { width: 50px; } .c { width: 200px; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 0.0),
            "0-width column cell must not inflate to padding: c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 62.0),
            "c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        assert!(
            approx(cell_at(&l, 1, 0).rect.w, 70.0),
            "colspan cell={}",
            cell_at(&l, 1, 0).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 86.0), "table={}", table.rect.w);

        // Table 9: the colspan=2 auto cell over two 0-width columns is
        // the 8px internal gap; the 100px fixed pin keeps 112px
        // border-box and the table grows to it.
        let l = layout_table_html(
            "<table><tr><td colspan=\"2\">x</td><td class=\"b\">y</td></tr>\
             <tr><td>x</td><td>y</td><td class=\"b\">z</td></tr></table>",
            "table { table-layout: fixed; width: 1px; border-spacing: 8px; } \
             td { font-size: 10px; padding: 6px; box-sizing: content-box; } .b { width: 100px; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 8.0),
            "colspan auto cell={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 112.0),
            "fixed pin cell={}",
            cell_at(&l, 0, 1).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 144.0), "table={}", table.rect.w);
    }

    #[test]
    fn fixed_layout_pct_excess_distributes_proportionally_to_base_widths() {
        // WPT `table-width-redistribution-fixed-padding` table 12:
        // 25%/15%/10% pins in a 168px fixed table with 8px
        // border-spacing and 6px cell padding (assignable 136px, pins
        // rebase to 46/32.4/25.6). The 32px excess distributes per
        // §3.9.3.2 rule 3 proportionally to the percentage-derived
        // base widths — 16/9.6/6.4 — giving 60.15/42.37/33.48,
        // matching the WPT 60/42.5/33 within its tolerance. Weighting
        // by the raw percentages instead gives 62/42/32 and misses.
        // Unpadded pins are proportional either way, so unpadded
        // behavior is unchanged.
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td><td class=\"c\">z</td></tr></table>",
            "table { table-layout: fixed; width: 168px; border-spacing: 8px; } \
             td { font-size: 10px; padding: 6px; box-sizing: content-box; } \
             .a { width: 25%; } .b { width: 15%; } .c { width: 10%; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 60.15),
            "c0={}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 42.37),
            "c1={}",
            cell_at(&l, 0, 1).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 2).rect.w, 33.48),
            "c2={}",
            cell_at(&l, 0, 2).rect.w
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 168.0),
            "table must not grow: {}",
            table.rect.w
        );
    }

    #[test]
    fn fixed_layout_colgroup_width_spreads_over_span() {
        // A `colgroup` width is one requirement for its whole span, not
        // a per-column pin: 200px over two columns pins each at 100px.
        // Both columns are length-pinned, so the 300px excess
        // distributes proportionally to the pinned widths (CSS Tables 3
        // §3.9.3.2 rule 2): 150px each, tiling the 500px table.
        let l = layout_table_html(
            "<table><colgroup class=\"g\" span=\"2\"></colgroup>\
             <tr><td>x</td><td>y</td></tr></table>",
            "table { table-layout: fixed; width: 500px; } \
             td { font-size: 10px; } .g { width: 200px; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c0.rect.w, 250.0));
        assert!(approx(c1.rect.w, 250.0));
    }

    #[test]
    fn fixed_layout_excess_distributes_proportionally_to_lengths() {
        // CSS Tables 3 §3.9.3.2 rule 2: every column pinned, so the
        // 200px excess goes to the non-zero length widths proportionally:
        // 100 + 200/3 and 200 + 400/3. The columns tile the table: the
        // row's cells sum to the full table width.
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td></tr></table>",
            "table { table-layout: fixed; width: 500px; } \
             td { font-size: 10px; } .a { width: 100px; } .b { width: 200px; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c0.rect.w, 500.0 / 3.0));
        assert!(approx(c1.rect.w, 1000.0 / 3.0));
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 500.0));
        assert!(approx(c0.rect.w + c1.rect.w, table.rect.w));
    }

    #[test]
    fn fixed_layout_open_column_takes_excess_before_pinned() {
        // CSS Tables 3 §3.9.3.2 rule 1 precedence: with one pinned and
        // one open column, the open column takes the whole excess even
        // though a proportional rule could apply to the pinned one.
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td>y</td></tr></table>",
            "table { table-layout: fixed; width: 600px; } \
             td { font-size: 10px; } .a { width: 100px; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 100.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 500.0));
    }

    #[test]
    fn fixed_layout_excess_distributes_proportionally_to_percentages() {
        // CSS Tables 3 §3.9.3.2 rule 3: percentage-pinned columns with
        // no length candidates share the excess proportionally to the
        // percentages — 20%/40% of 500px pin at 100/200, and the 200px
        // excess splits 1:2.
        let l = layout_table_html(
            "<table><colgroup><col class=\"a\"><col class=\"b\"></colgroup>\
             <tr><td>x</td><td>y</td></tr></table>",
            "table { table-layout: fixed; width: 500px; } \
             td { font-size: 10px; } .a { width: 20%; } .b { width: 40%; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 500.0 / 3.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 1000.0 / 3.0));
    }

    #[test]
    fn fixed_layout_group_pct_applies_at_full_value() {
        // WPT `colgroup-col` (by analogy for percentages): a `colgroup`
        // percentage is the percentage of each column it generates. A
        // 60% group pinning one 20% column gives the other column the
        // full 60% (600px of the 1000px table), not a 40% spread
        // remainder. The 200px excess splits 0.2:0.6 by intrinsic
        // percentage (50/150), not 0.2:0.4.
        let l = layout_table_html(
            "<table><colgroup class=\"g\"><col class=\"a\"><col></colgroup>\
             <tr><td>x</td><td>y</td></tr></table>",
            "table { table-layout: fixed; width: 1000px; border-spacing: 0; } \
             td { font-size: 10px; } .g { width: 60%; } .a { width: 20%; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 250.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 750.0));
    }

    #[test]
    fn fixed_layout_zero_width_columns_share_excess_equally() {
        // CSS Tables 3 §3.9.3.2 rule 4: every column pinned at zero
        // width — the excess splits equally.
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td></tr></table>",
            "table { table-layout: fixed; width: 400px; } \
             td { font-size: 10px; } .a { width: 0; } .b { width: 0; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 200.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 200.0));
    }

    #[test]
    fn fixed_layout_collapsed_column_takes_no_excess() {
        // `visibility: collapse` on a column: it keeps zero width and
        // takes none of the excess (CSS 2.1 §17.5); the visible
        // length-pinned column takes it all.
        let l = layout_table_html(
            "<table><colgroup><col class=\"a\"><col class=\"gone\"></colgroup>\
             <tr><td>x</td><td>y</td></tr></table>",
            "table { table-layout: fixed; width: 500px; } \
             td { font-size: 10px; } .a { width: 100px; } \
             .gone { visibility: collapse; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 500.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 0.0));
    }

    #[test]
    fn collapsed_row_cells_still_size_columns() {
        // CSS 2.1 §17.5.5: the suppression of a collapsed row "does not
        // otherwise affect the layout of the table" — the collapsed
        // row's cells still size their columns (WPT
        // `visibility-collapse-row-001`: collapsing a row must not
        // change the table's width).
        let l = layout_table_html(
            "<table><tr class=\"gone\"><td class=\"wide\">x</td></tr>\
             <tr><td>y</td></tr></table>",
            "td { font-size: 10px; padding: 0; } \
             .gone { visibility: collapse; } .wide { width: 100px; }",
        );
        // The visible row's cell stretches to the column width set by
        // the collapsed row's 100px cell.
        assert!(approx(cell_at(&l, 1, 0).rect.w, 100.0));
    }

    #[test]
    fn collapsed_row_cells_feed_min_content_width() {
        // Same rule through the intrinsic path: a collapsed row's
        // content still floors the column's min-content width.
        let l = layout_table_html(
            "<table><tr class=\"gone\"><td>widecontent</td></tr>\
             <tr><td>y</td></tr></table>",
            "td { font-size: 10px; padding: 0; } .gone { visibility: collapse; }",
        );
        let w0 = cell_at(&l, 1, 0).rect.w;
        let l2 = layout_table_html(
            "<table><tr><td>widecontent</td></tr>\
             <tr><td>y</td></tr></table>",
            "td { font-size: 10px; padding: 0; }",
        );
        // Collapsed or not, the column sizes identically.
        assert!(approx(w0, cell_at(&l2, 1, 0).rect.w));
    }

    // ---- cell vertical-align ----

    #[test]
    fn table_cell_vertical_align_middle_is_default() {
        // No `vertical-align`: the UA default for `td`/`th` is `middle`
        // (HTML rendering § "tables"), so the short cell's content is
        // centered in the 50px row: 19px above the 12px line.
        let l = layout_table_html(
            "<table><tr><td class=\"tall\">x</td><td>y</td></tr></table>",
            "td { font-size: 10px; } .tall { height: 50px; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c0.rect.h, 50.0));
        assert!(approx(c1.rect.h, 50.0));
        let inner = &c1.children[0];
        assert!(approx(inner.rect.y, c1.rect.y + 19.0));
    }

    #[test]
    fn table_cell_vertical_align_top_explicit() {
        // Explicit `vertical-align: top` keeps the short cell's content
        // at the row top.
        let l = layout_table_html(
            "<table><tr><td class=\"tall\">x</td><td class=\"t\">y</td></tr></table>",
            "td { font-size: 10px; } .tall { height: 50px; } .t { vertical-align: top; }",
        );
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c1.rect.h, 50.0));
        let inner = &c1.children[0];
        assert!(approx(inner.rect.y, c1.rect.y));
    }

    #[test]
    fn table_baseline_ignores_spanning_cell_fallback() {
        // WPT css-tables/tentative/baseline-table "Empty rowspanned
        // cells should not set baseline": the empty rowspan=2 cell's
        // 88px content-bottom must not become the inline-table's
        // baseline; the first row's non-spanning empty cell does. The
        // sibling inline-block then sits at y=15 (line baseline 55).
        let html = "<div class=\"container\">Xp<table><tr><td rowspan=\"2\" style=\"height:80px\"></td><td></td></tr><tr><td></td></tr></table><div class=\"sib\" style=\"display:inline-block\">pX</div></div>";
        let css = ".container { position: relative; height: 100px; width: 400px; font: 50px/1 Ahem; } \
                   .container table { display: inline-table; border: 15px solid green; background: gray; width: 0px; height: 0px; vertical-align: baseline; border-spacing: 0; } \
                   .container td { background: #BFB; padding: 4px; }";
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(css);
        let l = layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics);
        // body > div.container > div.sib
        let sib = &l.children[0].children[0].children[2];
        // With FixedFontMetrics the sibling sits at y=12; with Ahem (WPT)
        // it is y=15. The key assertion: the 88px rowspan cell must NOT
        // drive the baseline (which would put the sibling at y=59).
        assert!(
            sib.rect.y < 30.0,
            "sib y={} (rowspan cell must not set baseline)",
            sib.rect.y
        );
    }

    #[test]
    fn table_cell_vertical_align_middle() {
        // One 12px line in a 50px row: middle puts 19px above it.
        let l = layout_table_html(
            "<table><tr><td class=\"tall\">x</td><td class=\"mid\">y</td></tr></table>",
            "td { font-size: 10px; } .tall { height: 50px; } .mid { vertical-align: middle; }",
        );
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c1.rect.h, 50.0));
        let inner = &c1.children[0];
        assert!(approx(inner.rect.y, c1.rect.y + 19.0));
    }

    #[test]
    fn table_cell_vertical_align_bottom() {
        // One 12px line in a 50px row: bottom puts 38px above it.
        let l = layout_table_html(
            "<table><tr><td class=\"tall\">x</td><td class=\"bot\">y</td></tr></table>",
            "td { font-size: 10px; } .tall { height: 50px; } .bot { vertical-align: bottom; }",
        );
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c1.rect.h, 50.0));
        let inner = &c1.children[0];
        assert!(approx(inner.rect.y, c1.rect.y + 38.0));
    }

    #[test]
    fn table_cell_vertical_align_baseline() {
        // Two-line cell next to a one-line cell: baseline alignment
        // drops the short cell's content by exactly one line height so
        // the baselines meet. Both cells pin `vertical-align: baseline`
        // explicitly: the UA default for `td` is `middle`.
        let l = layout_table_html(
            "<table><tr><td class=\"two\">one two</td><td class=\"one\">x</td></tr></table>",
            "table { table-layout: fixed; width: 200px; } \
             td { font-size: 10px; vertical-align: baseline; } .two { width: 30px; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        // The narrow fixed column forces two 12px lines.
        assert!(approx(c0.rect.h, 24.0));
        assert!(approx(c1.rect.h, 24.0));
        let inner = &c1.children[0];
        assert!(approx(inner.rect.y, c1.rect.y + 12.0));
    }

    /// First text box in a laid-out subtree, by depth-first walk.
    fn first_text_box(b: &crate::LayoutBox) -> Option<&crate::LayoutBox> {
        let mut stack = vec![b];
        while let Some(n) = stack.pop() {
            if n.box_type == crate::BoxType::Text {
                return Some(n);
            }
            stack.extend(n.children.iter().rev());
        }
        None
    }

    #[test]
    fn w75f3_cell_row_baseline_walks_nested_blocks() {
        // Wave 75 fixer F3 (item 6): a cell whose first block child
        // nests another block before the text (cell > div > div >
        // text) must baseline-align on the inner text's baseline
        // (9.6), not on the intermediate div's stamp (its bottom
        // edge, 12, since a block with only block children has no
        // lines). Bowser's cell convention is last-line (see the
        // w75e `table_cell_vertical_align_baseline` note), so the
        // sibling's two-line div contributes its last-line baseline
        // (21.6) and the nested cell shifts by 21.6 - 9.6 = 12.
        // Fail-before: the nested cell shifts by 21.6 - 12 = 9.6.
        let l = layout_table_html(
            "<table><tr><td class=\"a\"><div><div>aaaa</div></div></td>\
             <td class=\"b\"><div class=\"narrow\">one two</div></td></tr></table>",
            "td { font-size: 10px; vertical-align: baseline; } \
             .narrow { width: 30px; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        let t0 = first_text_box(c0).expect("nested cell text");
        let t1 = first_text_box(c1).expect("sibling cell text");
        assert!(
            approx(t0.rect.y, c0.rect.y + 12.0),
            "nested text y {}, cell y {}: want +12 (21.6 last-line max minus 9.6)",
            t0.rect.y,
            c0.rect.y
        );
        assert!(
            approx(t1.rect.y, c1.rect.y),
            "sibling first-line text y {}, cell y {}: unshifted",
            t1.rect.y,
            c1.rect.y
        );
        assert!(
            approx(t1.rect.y, c1.rect.y),
            "sibling first-line text y {}, cell y {}",
            t1.rect.y,
            c1.rect.y
        );
    }

    #[test]
    fn w75f3_inline_table_baseline_uses_cell_row_baseline() {
        // Wave 75 fixer F3 (item 6, second site): an inline table's
        // first-row baseline must come from `cell_row_baseline`, not
        // the cell node's own `baseline` (0 for a cell whose first
        // child is a block: no lines, so `bt + pt`). The inline
        // table sits on a text line; with the text baseline as the
        // table baseline it needs no shift, so its top meets the
        // line's top. Fail-before: the table sits one baseline lower.
        // Standards-mode fixture (w77fix): `layout_table_html`
        // injects no doctype, so a bare fixture parses as full
        // quirks mode, where HTML rendering §15.3.8 resets
        // `font-size` at the table boundary — the 10px on `.line`
        // would NOT reach the cell, the cell's 16px line would
        // baseline at 15.36 against the outer 9.6, and the outer
        // text would legitimately shift down 5.76. That shift is
        // correct quirks behavior (bisect: 3592883, the quirks
        // font reset; not the dcdb290/81696e3 blockification,
        // which is flex-gated and passes this test), not a
        // baseline bug — so this baseline test runs in standards
        // mode, where the 10px does inherit, per the helper's own
        // "pass an explicit doctype when the mode matters" rule
        // and the 06a104c `layout_html` precedent. Assertions
        // unchanged.
        let l = layout_table_html(
            "<!DOCTYPE html><div class=\"line\">x<table class=\"it\"><tr>\
             <td><div>yy</div></td></tr></table></div>",
            ".line { font-size: 10px; } \
             table.it { display: inline-table; } \
             td { vertical-align: baseline; }",
        );
        let line = &l.children[0].children[0];
        let table = line
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::InlineBlock)
            .expect("inline table box");
        assert!(
            approx(table.rect.y, line.rect.y),
            "inline table y {}, line y {}: baseline-aligned, no shift",
            table.rect.y,
            line.rect.y
        );
        // The line's baseline is the max of its items' baselines, so
        // a wrong table baseline shifts the TEXT, not the table: with
        // the cell node's own baseline (the content-bottom fallback)
        // the text sits one baseline-gap lower.
        let text = first_text_box(line).expect("line text");
        assert!(
            approx(text.rect.y, line.rect.y),
            "line text y {}, line y {}: text must not shift for the table",
            text.rect.y,
            line.rect.y
        );
    }

    #[test]
    fn w75f3_finite_cell_max_content_sums_inline_run() {
        // Wave 75 fixer F3 (item 4): a cell whose max-content is
        // infinite (it directly contains an auto table with a 100%
        // column) must contribute the inline RUN sum to its outer
        // column, not the max over its children: text run 30 (5 chars
        // at 10px) + inline `<span>` 30 + inner table grid sum 100
        // = 160. The outer table is a `flex: 0 0 auto` item, so its
        // width is its column maximum. Fail-before: 100 (the max).
        let l = layout_table_html(
            "<div class=\"c\"><table class=\"t\"><tr><td>aaaaa<span>bbbbb</span>\
             <table class=\"inner\"><tr><td style=\"width:100%\">\
             <div style=\"width:100px\"></div></td></tr></table></td></tr></table>\
             <div class=\"s\">x</div></div>",
            "body { margin: 0; } .c { display: flex; width: 800px; } \
             table { border-spacing: 0; } td { padding: 0; font-size: 10px; } \
             td > div { display: inline-block; } \
             .t { flex: 0 0 auto; } .s { flex: 1 1 auto; }",
        );
        let outer = &l.children[0].children[0].children[0];
        assert_eq!(outer.box_type, crate::BoxType::Table);
        assert!(
            approx(outer.rect.w, 160.0),
            "outer table w {}, want 160 (30 text + 30 span + 100 inner grid)",
            outer.rect.w
        );
    }

    #[test]
    fn w75f3_vertical_flex_table_gets_no_wrong_axis_base() {
        // Wave 75 fixer F3 (item 3): a vertical flex-item table with
        // an infinite base (100% inline-axis column, transposed to
        // `height: 100%`) must NOT take its column-maximum sum
        // (30 + 100 + 30 spacing = 160, a physical-vertical extent)
        // as its horizontal flex base. With the vertical guard in
        // `table_auto_grid_max_width` the base stays infinite and
        // saturates like a fixed-layout table's, overflowing the
        // 600px container instead of landing on 160. Fail-before: 160.
        let l = layout_table_html(
            "<div class=\"c\"><table class=\"t\"><tr>\
             <td style=\"height:100%\"><div style=\"height:30px\"></div></td>\
             <td><div style=\"height:50px\"></div><div style=\"height:50px\"></div></td>\
             </tr></table><div class=\"s\">x</div></div>",
            "body { margin: 0; } .c { display: flex; width: 600px; } \
             table.t { writing-mode: vertical-lr; border-spacing: 10px 10px; } \
             td { padding: 0; } td > div { display: inline-block; } \
             .t { flex: 0 0 auto; } .s { flex: 1 1 auto; }",
        );
        let t = &l.children[0].children[0].children[0];
        let s = &l.children[0].children[0].children[1];
        assert!(
            t.rect.w > 600.0,
            "vertical flex table w {}, want saturation past 600, not the wrong-axis 160",
            t.rect.w
        );
        assert!(
            approx(s.rect.w, 9.6),
            "sibling w {}, want 9.6 (one 16px char)",
            s.rect.w
        );
    }

    #[test]
    fn w75f3_nested_pct_tables_use_cached_grid_max() {
        // Wave 75 fixer F3 (item 5): three levels of auto tables with
        // 100% columns, as a `flex: 0 0 auto` item. The innermost grid
        // sum is 100 (one 100px div); the middle table's cell is
        // infinite and falls back to that sum, and the outer table's
        // cell falls back to the middle sum — so the outer width is
        // 100. This exercises the `PrefWidthCache` grid-max memo on
        // repeated outer passes: a stale or cross-table cached value
        // would land anywhere but 100.
        let l = layout_table_html(
            "<div class=\"c\"><table class=\"t\"><tr><td>\
             <table class=\"m\"><tr><td style=\"width:100%\">\
             <table class=\"in\"><tr><td style=\"width:100%\">\
             <div style=\"width:100px\"></div></td></tr></table>\
             </td></tr></table></td></tr></table>\
             <div class=\"s\">x</div></div>",
            "body { margin: 0; } .c { display: flex; width: 800px; } \
             table { border-spacing: 0; } td { padding: 0; font-size: 10px; } \
             td > div { display: inline-block; } \
             .t { flex: 0 0 auto; } .s { flex: 1 1 auto; }",
        );
        let outer = &l.children[0].children[0].children[0];
        assert_eq!(outer.box_type, crate::BoxType::Table);
        assert!(
            approx(outer.rect.w, 100.0),
            "outer table w {}, want 100 (cached grid-max chain)",
            outer.rect.w
        );
    }

    #[test]
    fn w75f3_single_row_exact_equality_resolves_cell_pct() {
        // Wave 75 fixer F3 (item 7): a single-row table whose natural
        // height already equals its specified height (excess 0, so no
        // row "grew") still has a definite row height — the table's
        // definite height IS the row's height — so a % child of the
        // cell must resolve. Fail-before: the % div stays 0.
        let l = layout_table_html(
            "<table><tr><td><div class=\"fixed\"></div><div class=\"pct\"></div></td></tr></table>",
            "table { height: 100px; } \
             .fixed { width: 100px; height: 100px; } \
             .pct { width: 100px; height: 50%; }",
        );
        let cell = cell_at(&l, 0, 0);
        let pct = &cell.children[1];
        assert!(
            approx(pct.rect.h, 50.0),
            "pct div h {}, want 50 (50% of the definite 100px row)",
            pct.rect.h
        );
    }

    #[test]
    fn w75f3_spanning_cell_all_rows_grew_resolves_cell_pct() {
        // Wave 75 fixer F3 (item 7): a rowspan == 2 cell whose every
        // spanned row grew from table-height excess has a definite
        // used height (the grown span), so its % child must resolve
        // against the span. The span cell's own height is the sum of
        // the two grown rows, so the % child is half of it whatever
        // the excess split. Fail-before: the % div stays 0.
        let l = layout_table_html(
            "<table><tr><td rowspan=\"2\"><div class=\"pct\"></div></td>\
             <td><div class=\"r\"></div></td></tr>\
             <tr><td><div class=\"r\"></div></td></tr></table>",
            "table { height: 200px; } \
             .pct { width: 100px; height: 50%; } \
             .r { width: 100px; height: 50px; }",
        );
        let span = cell_at(&l, 0, 0);
        let pct = &span.children[0];
        assert!(
            approx(pct.rect.h, span.rect.h / 2.0),
            "pct div h {}, span h {}: want half the grown span",
            pct.rect.h,
            span.rect.h
        );
    }

    #[test]
    fn w75f3_vertical_table_baseline_cells_lay_out() {
        // Wave 75 fixer F3 (item 8): vertical-table baseline alignment
        // is unmodeled (see the `!vertical` guard at the row-baseline
        // sites, matching `table_inline_baseline`'s carve-out), so this
        // test pins only that a vertical table with baseline-aligned
        // block-first and text cells lays out to finite, sane
        // geometry through the guarded path — it does not pin a
        // baseline shift value, which the transpose makes unmodeled.
        let l = layout_table_html(
            "<table><tr><td class=\"a\"><div class=\"narrow\">one two</div></td>\
             <td class=\"b\">x</td></tr></table>",
            "table { writing-mode: vertical-lr; } \
             td { font-size: 10px; vertical-align: baseline; } \
             .narrow { width: 30px; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        for (name, b) in [("cell a", c0), ("cell b", c1)] {
            assert!(
                b.rect.w.is_finite() && b.rect.h.is_finite() && b.rect.w > 0.0 && b.rect.h > 0.0,
                "{name} rect {}/{}/{}x{} not finite positive",
                b.rect.x,
                b.rect.y,
                b.rect.w,
                b.rect.h
            );
        }
        assert!(first_text_box(c0).is_some(), "cell a text laid out");
        assert!(first_text_box(c1).is_some(), "cell b text laid out");
    }

    #[test]
    fn w75f3_inline_pct_baseline_matches_wpt_geometry() {
        // Exact WPT baseline-td %ge shape, as an inline table
        // (position:relative, 8px horizontal border-spacing, on a
        // text line): the 50% div must sit at table-top + 30 (row
        // baseline 80 minus its own 50 bottom edge), the 80% div
        // at 0. This pins the engine side of the WPT subtest; the
        // driver's offsetTop read of the same div is a separate
        // (js/shim) path, not engine layout.
        let l = layout_table_html(
            "<div>inline table<table class=\"it\"><tr>\
             <td><div class=\"d1\"></div></td>\
             <td><div class=\"d2\"></div></td></tr></table></div>",
            "table.it { display: inline-table; height: 100px; \
             position: relative; border-spacing: 8px 0; } \
             td { vertical-align: baseline; } \
             .d1 { width: 100px; height: 50%; } \
             .d2 { width: 100px; height: 80%; }",
        );
        let line = &l.children[0].children[0];
        let table = line
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::InlineBlock)
            .expect("inline table box");
        // Table children: row-group > row > cells; find divs by walk.
        let mut divs: Vec<&crate::LayoutBox> = Vec::new();
        let mut stack: Vec<&crate::LayoutBox> = vec![table];
        while let Some(n) = stack.pop() {
            if n.box_type == crate::BoxType::Block && (n.rect.w - 100.0).abs() < 0.01 {
                divs.push(n);
            }
            stack.extend(n.children.iter());
        }
        divs.sort_by(|a, b| a.rect.y.partial_cmp(&b.rect.y).unwrap());
        assert_eq!(divs.len(), 2, "two % divs");
        assert!(
            approx(divs[0].rect.y, table.rect.y),
            "80% div y {}, table y {}: want table top",
            divs[0].rect.y,
            table.rect.y
        );
        assert!(
            approx(divs[1].rect.y, table.rect.y + 30.0),
            "50% div y {}, table y {}: want table top + 30",
            divs[1].rect.y,
            table.rect.y
        );

        // Whitespace variant (w76): the WPT page's tds contain
        // newline/space text around the % divs. That whitespace must
        // not make the cell's baseline the stretched row height:
        // each cell still starts at the table top, and the 50% div
        // still sits 30 below its cell top (bottom-aligned with the
        // 80% div at 0). Before the fix both divs sat at 0.
        // Explicit tr, as in the compact case; only the td-
        // internal whitespace is new (the WPT page has bare
        // tds, covered end to end by the shim probe).
        let l = layout_table_html(
            "<div>inline table<table class=\"it\"><tr><td>\n<div class=\"d1\"></div>\n</td><td>\n<div class=\"d2\"></div>\n</td></tr></table></div>",
            "table.it { display: inline-table; height: 100px; \
                          position: relative; border-spacing: 8px 0; } \
                          td { vertical-align: baseline; } \
                          .d1 { width: 100px; height: 50%; } \
                          .d2 { width: 100px; height: 80%; }",
        );
        let line = &l.children[0].children[0];
        let table = line
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::InlineBlock)
            .expect("inline table box");
        let mut cells: Vec<&crate::LayoutBox> = Vec::new();
        let mut stack: Vec<&crate::LayoutBox> = vec![table];
        while let Some(n) = stack.pop() {
            if n.box_type == crate::BoxType::TableCell {
                cells.push(n);
            }
            stack.extend(n.children.iter());
        }
        cells.sort_by(|a, b| a.rect.x.partial_cmp(&b.rect.x).unwrap());
        assert_eq!(cells.len(), 2, "two cells");
        for (cell, want) in cells.iter().zip([30.0, 0.0]) {
            assert!(
                approx(cell.rect.y, table.rect.y),
                "cell y {}, table y {}: want table top",
                cell.rect.y,
                table.rect.y
            );
            let mut div_stack: Vec<&crate::LayoutBox> = vec![cell];
            let mut div: Option<&crate::LayoutBox> = None;
            while let Some(n) = div_stack.pop() {
                if n.box_type == crate::BoxType::Block && approx(n.rect.w, 100.0) {
                    div = Some(n);
                    break;
                }
                div_stack.extend(n.children.iter());
            }
            let div = div.expect("% div in cell");
            assert!(
                approx(div.rect.y - cell.rect.y, want),
                "div offset {}, want {want}",
                div.rect.y - cell.rect.y
            );
        }
    }

    // w76fix finding 6: the cell-baseline whitespace skip must recurse
    // through named wrappers. The span shape uses the w75f3 % div
    // geometry (row baseline 80 from the 80% div): with the skip, the
    // 50% div's baseline (50) aligns at offset 30 and the 80% div at
    // 0. Without the skip, the span shape early-returns the cell's own
    // (stretched) baseline instead.
    // The div shape pins the opposite, no-change behavior: box
    // construction drops whitespace-only text inside a block box, so
    // `<div> </div>` arrives as an *empty* div — a real first child,
    // not whitespace — and the 50% div sits at offset 80 (row
    // baseline 80 minus the empty div's baseline 0), identically
    // before and after this fix. The task's div case as literally
    // written cannot occur in this engine; only the span form (where
    // the space survives inside an inline box) exercises the skip.
    #[test]
    fn w76fix_cell_baseline_skips_named_whitespace_wrappers() {
        // Shape 1: whitespace span before the real div.
        for (label, first, wants) in [
            ("span", "<span> </span>", [30.0, 0.0]),
            ("div", "<div> </div>", [80.0, 0.0]),
        ] {
            let html = format!(
                "<div>inline table<table class=\"it\"><tr><td>{first}\
                 <div class=\"d1\"></div></td><td><div class=\"d2\"></div></td>\
                 </tr></table></div>"
            );
            let l = layout_table_html(
                &html,
                "table.it { display: inline-table; height: 100px; \
                 position: relative; border-spacing: 8px 0; } \
                 td { vertical-align: baseline; } \
                 .d1 { width: 100px; height: 50%; } \
                 .d2 { width: 100px; height: 80%; }",
            );
            let line = &l.children[0].children[0];
            let table = line
                .children
                .iter()
                .find(|c| c.box_type == crate::BoxType::InlineBlock)
                .expect("inline table box");
            let mut cells: Vec<&crate::LayoutBox> = Vec::new();
            let mut stack: Vec<&crate::LayoutBox> = vec![table];
            while let Some(n) = stack.pop() {
                if n.box_type == crate::BoxType::TableCell {
                    cells.push(n);
                }
                stack.extend(n.children.iter());
            }
            cells.sort_by(|a, b| a.rect.x.partial_cmp(&b.rect.x).unwrap());
            assert_eq!(cells.len(), 2, "two cells ({label})");
            for (cell, want) in cells.iter().zip(wants) {
                let mut div_stack: Vec<&crate::LayoutBox> = vec![cell];
                let mut div: Option<&crate::LayoutBox> = None;
                while let Some(n) = div_stack.pop() {
                    if n.box_type == crate::BoxType::Block && approx(n.rect.w, 100.0) {
                        div = Some(n);
                        break;
                    }
                    div_stack.extend(n.children.iter());
                }
                let div = div.expect("% div in cell");
                assert!(
                    approx(div.rect.y - cell.rect.y, want),
                    "{label}: div offset {}, want {want}",
                    div.rect.y - cell.rect.y
                );
            }
        }
    }

    // w76fix finding 6 (js-review M6b): the skip is gated on
    // collapsed whitespace. Under `white-space: pre` the span's space
    // is preserved, creates a line box, and must NOT be skipped: the
    // cell keeps its own (stretched, 100) baseline, so the row
    // baseline is 100, the 50% div sits one space-line (19.2 at
    // 16px) below its cell top, and the 80% div sits at 100 - 80 =
    // 20. If the gate were missing, the span would be skipped and
    // the offsets would read 30 / 0 like the collapsed case.
    #[test]
    fn w76fix_cell_baseline_pre_whitespace_not_skipped() {
        let l = layout_table_html(
            "<div>inline table<table class=\"it\"><tr><td style=\"white-space:pre\"><span> </span><div class=\"d1\"></div></td><td><div class=\"d2\"></div></td></tr></table></div>",
            "table.it { display: inline-table; height: 100px; position: relative; border-spacing: 8px 0; } td { vertical-align: baseline; } .d1 { width: 100px; height: 50%; } .d2 { width: 100px; height: 80%; }",
        );
        let line = &l.children[0].children[0];
        let table = line
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::InlineBlock)
            .expect("inline table box");
        let mut cells: Vec<&crate::LayoutBox> = Vec::new();
        let mut stack: Vec<&crate::LayoutBox> = vec![table];
        while let Some(n) = stack.pop() {
            if n.box_type == crate::BoxType::TableCell {
                cells.push(n);
            }
            stack.extend(n.children.iter());
        }
        cells.sort_by(|a, b| a.rect.x.partial_cmp(&b.rect.x).unwrap());
        assert_eq!(cells.len(), 2, "two cells");
        for (cell, want) in cells.iter().zip([19.2, 20.0]) {
            let mut div_stack: Vec<&crate::LayoutBox> = vec![cell];
            let mut div: Option<&crate::LayoutBox> = None;
            while let Some(n) = div_stack.pop() {
                if n.box_type == crate::BoxType::Block && approx(n.rect.w, 100.0) {
                    div = Some(n);
                    break;
                }
                div_stack.extend(n.children.iter());
            }
            let div = div.expect("% div in cell");
            assert!(
                approx(div.rect.y - cell.rect.y, want),
                "div offset {}, want {want}",
                div.rect.y - cell.rect.y
            );
        }
    }

    #[test]
    fn baseline_td_percent_heights_from_table_height() {
        // WPT `baseline-td` "Baselines with %ge children": table
        // height 100, baseline cells whose only content is a block
        // div at 50% / 80%. The table's definite height makes the
        // row's used height definite (unlike a row's own specified
        // height, see `row_height_alone_does_not_resolve_cell_pct_children`),
        // so the divs resolve to 50 / 80, and their bottoms — the
        // cells' first baselines — align at 80, putting the 50px div
        // at offset 30 and the 80px div at 0 from the table top.
        // Before w75e both divs resolved to 0 at offset 0.
        let l = layout_table_html(
            "<table><tr><td class=\"a\"><div class=\"d1\"></div></td>\
             <td class=\"b\"><div class=\"d2\"></div></td></tr></table>",
            "table { height: 100px; } \
             td { vertical-align: baseline; } \
             .d1 { width: 100px; height: 50%; } \
             .d2 { width: 100px; height: 80%; }",
        );
        let table = &l.children[0].children[0];
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        let d0 = &c0.children[0];
        let d1 = &c1.children[0];
        assert!(approx(table.rect.h, 100.0), "table={}", table.rect.h);
        assert!(approx(d0.rect.h, 50.0), "d0 h={}", d0.rect.h);
        assert!(approx(d1.rect.h, 80.0), "d1 h={}", d1.rect.h);
        assert!(
            approx(d0.rect.y - table.rect.y, 30.0),
            "d0 offset={}, want 30",
            d0.rect.y - table.rect.y
        );
        assert!(
            approx(d1.rect.y - table.rect.y, 0.0),
            "d1 offset={}, want 0",
            d1.rect.y - table.rect.y
        );
    }

    #[test]
    fn table_cell_vertical_align_sub_matches_baseline() {
        // CSS 2.1 §17.5.3: `sub` does not apply to table cells; the
        // cell is aligned at the baseline instead. Assert layout
        // identical to `vertical-align: baseline`.
        let sub = layout_table_html(
            "<table><tr><td>x</td><td class=\"sub\">y</td></tr></table>",
            "td { font-size: 10px; } .sub { vertical-align: sub; }",
        );
        let base = layout_table_html(
            "<table><tr><td>x</td><td class=\"base\">y</td></tr></table>",
            "td { font-size: 10px; } .base { vertical-align: baseline; }",
        );
        let c_sub = cell_at(&sub, 0, 1);
        let c_base = cell_at(&base, 0, 1);
        assert!(approx(c_sub.rect.h, c_base.rect.h));
        let inner_sub = &c_sub.children[0];
        let inner_base = &c_base.children[0];
        assert!(approx(
            inner_sub.rect.y - c_sub.rect.y,
            inner_base.rect.y - c_base.rect.y
        ));
    }

    #[test]
    fn table_cell_vertical_align_super_matches_baseline() {
        // Same as `sub`: `super` is meaningless on cells and behaves as
        // `baseline`, with no content shift.
        let sup = layout_table_html(
            "<table><tr><td>x</td><td class=\"sup\">y</td></tr></table>",
            "td { font-size: 10px; } .sup { vertical-align: super; }",
        );
        let base = layout_table_html(
            "<table><tr><td>x</td><td class=\"base\">y</td></tr></table>",
            "td { font-size: 10px; } .base { vertical-align: baseline; }",
        );
        let c_sup = cell_at(&sup, 0, 1);
        let c_base = cell_at(&base, 0, 1);
        assert!(approx(c_sup.rect.h, c_base.rect.h));
        let inner_sup = &c_sup.children[0];
        let inner_base = &c_base.children[0];
        assert!(approx(
            inner_sup.rect.y - c_sup.rect.y,
            inner_base.rect.y - c_base.rect.y
        ));
    }

    #[test]
    fn table_cell_vertical_align_sub_tall_row_matches_baseline() {
        // Next to a two-line cell, `sub` participates in the row's
        // baseline alignment exactly like `baseline`: the one-line
        // cell's content drops one line height, with no extra shift.
        // The two-line cell pins `baseline` explicitly (the UA default
        // for `td` is `middle`).
        let sub = layout_table_html(
            "<table><tr><td class=\"two\">one two</td><td class=\"sub\">x</td></tr></table>",
            "table { table-layout: fixed; width: 200px; } \
             td { font-size: 10px; } .two { width: 30px; vertical-align: baseline; } .sub { vertical-align: sub; }",
        );
        let base = layout_table_html(
            "<table><tr><td class=\"two\">one two</td><td class=\"base\">x</td></tr></table>",
            "table { table-layout: fixed; width: 200px; } \
             td { font-size: 10px; } .two { width: 30px; vertical-align: baseline; } .base { vertical-align: baseline; }",
        );
        let c_sub = cell_at(&sub, 0, 1);
        let c_base = cell_at(&base, 0, 1);
        let inner_sub = &c_sub.children[0];
        let inner_base = &c_base.children[0];
        assert!(approx(
            inner_sub.rect.y - c_sub.rect.y,
            inner_base.rect.y - c_base.rect.y
        ));
        // The one-line cell sits one 12px line below the row top.
        assert!(approx(inner_sub.rect.y - c_sub.rect.y, 12.0));
    }

    // ---- cell-baseline alignment with real font metrics ----

    /// [`FontMetrics`] with a real (non-probe) ascent/descent split: the
    /// baseline sits at 60% of the line height instead of the historical
    /// 80/20 probe split. Exercises the [`FontMetrics::ascent_descent`]
    /// wiring in table cell-baseline alignment.
    struct SixtyFortyMetrics;

    impl crate::FontMetrics for SixtyFortyMetrics {
        fn advance(&self, text: &str, font_size: f32, family: crate::FontFamilyId) -> f32 {
            FixedFontMetrics.advance(text, font_size, family)
        }
        fn line_height(&self, font_size: f32, family: crate::FontFamilyId) -> f32 {
            FixedFontMetrics.line_height(font_size, family)
        }
        fn ascent_descent(&self, font_size: f32, family: crate::FontFamilyId) -> (f32, f32) {
            let lh = self.line_height(font_size, family);
            (0.6 * lh, 0.4 * lh)
        }
    }

    /// Lay out a document with the given [`FontMetrics`] and return the
    /// public tree (800x600 viewport).
    /// Mode split (see `standards_html` in `tests.rs`): no doctype
    /// is injected, so doctype-less input parses as full quirks
    /// mode. Pass an explicit doctype when the fixture's mode
    /// matters.
    fn layout_table_html_with(
        html: &str,
        css: &str,
        fonts: &dyn crate::FontMetrics,
    ) -> crate::LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
        crate::layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, fonts)
    }

    #[test]
    fn table_cell_baseline_uses_font_metrics() {
        // Two single-line cells with different font sizes, both
        // `vertical-align: baseline`. With the 60/40 metrics the 10px
        // cell's baseline is 7.2px below its content top (12px line) and
        // the 20px cell's is 14.4px (24px line); the row baseline is the
        // max, so the small cell's content drops 7.2px. The old 80/20
        // probe split would have dropped it 9.6px instead.
        let l = layout_table_html_with(
            "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td></tr></table>",
            ".a { font-size: 10px; vertical-align: baseline; } \
             .b { font-size: 20px; vertical-align: baseline; }",
            &SixtyFortyMetrics,
        );
        let ca = cell_at(&l, 0, 0);
        let cb = cell_at(&l, 0, 1);
        let inner_a = &ca.children[0];
        let inner_b = &cb.children[0];
        // The 20px cell owns the row baseline: no shift, content at the
        // row top.
        assert!(approx(inner_b.rect.y - cb.rect.y, 0.0));
        // The 10px cell drops by the baseline difference, 14.4 - 7.2.
        assert!(approx(inner_a.rect.y - ca.rect.y, 7.2));
        // After alignment both cells' baselines coincide.
        let base_a = inner_a.rect.y + 7.2;
        let base_b = inner_b.rect.y + 14.4;
        assert!(approx(base_a, base_b));
    }

    #[test]
    fn table_cell_baseline_metrics_keep_line_height() {
        // The metric split moves the baseline inside the line box but
        // keeps the line box at the full line height: a 10px single-line
        // cell is still 12px tall under the 60/40 metrics.
        let l = layout_table_html_with(
            "<table><tr><td>x</td></tr></table>",
            "td { font-size: 10px; }",
            &SixtyFortyMetrics,
        );
        let c = cell_at(&l, 0, 0);
        assert!(approx(c.rect.h, 12.0));
    }

    #[test]
    fn collapsed_borders_exposed_on_cell_layout_boxes() {
        // The paint handoff: each cell's public LayoutBox carries the
        // edges it paints ([top, right, bottom, left]). The left cell
        // owns top/left/bottom; its right (shared) edge is NONE so the
        // grid line is painted exactly once, by the right cell's left.
        let l = layout_table_html(
            "<table><tr><td>a</td><td>b</td></tr></table>",
            "table { border-collapse: collapse; } td { border: 2px solid red; }",
        );
        let c0 = cell_at(&l, 0, 0);
        let e0 = c0
            .collapsed_borders
            .as_ref()
            .expect("left cell exposes winners");
        assert!(approx(e0[0].width, 2.0));
        assert_eq!(e0[0].style, CollapsedBorderStyle::Solid);
        assert_eq!(e0[1], crate::CollapsedBorder::NONE);
        assert!(approx(e0[2].width, 2.0));
        assert!(approx(e0[3].width, 2.0));
        let c1 = cell_at(&l, 0, 1);
        let e1 = c1
            .collapsed_borders
            .as_ref()
            .expect("right cell exposes winners");
        assert!(approx(e1[3].width, 2.0)); // shared edge, painted once here
        assert_eq!(e1[3].style, CollapsedBorderStyle::Solid);
        // Non-cell boxes carry nothing.
        let table = &l.children[0].children[0];
        assert!(table.collapsed_borders.is_none());
    }

    #[test]
    fn collapsed_borders_bottom_right_owned_by_last_row_col() {
        let l = layout_table_html(
            "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>",
            "table { border-collapse: collapse; } td { border: 2px solid red; }",
        );
        // Interior cell (0,0): only top/left owned.
        let e = cell_at(&l, 0, 0)
            .collapsed_borders
            .as_ref()
            .expect("winners");
        assert!(approx(e[0].width, 2.0));
        assert_eq!(e[1], crate::CollapsedBorder::NONE);
        assert_eq!(e[2], crate::CollapsedBorder::NONE);
        assert!(approx(e[3].width, 2.0));
        // Last-row, last-column cell (1,1): owns all four edges.
        let e = cell_at(&l, 1, 1)
            .collapsed_borders
            .as_ref()
            .expect("winners");
        assert!(e.iter().all(|b| approx(b.width, 2.0)));
    }

    #[test]
    fn separate_model_cells_expose_no_collapsed_borders() {
        let l = layout_table_html(
            "<table><tr><td>a</td></tr></table>",
            "td { border: 2px solid red; }",
        );
        let c0 = cell_at(&l, 0, 0);
        assert!(c0.collapsed_borders.is_none());
    }

    #[test]
    fn measured_cell_must_not_carry_collapsed_borders() {
        // A cell in a visible row whose columns are all collapsed never
        // paints: it is laid out only to measure its height, then
        // detached (zero rect, children dropped). The paint-handoff
        // stamp must skip it — paint reads `collapsed_borders` for
        // every TableCell with no zero-area guard, so a stamped
        // detached cell emits its border winners at the page origin.
        let l = layout_table_html(
            "<table><col><col class=\"gone\">\
             <tr><td>a</td><td>b</td></tr></table>",
            "table { border-collapse: collapse; } \
             td { padding: 0; border: 4px solid red; } \
             col.gone { visibility: collapse; }",
        );
        let c = cell_at(&l, 0, 1);
        assert!(
            approx(c.rect.w, 0.0) && approx(c.rect.h, 0.0),
            "detached rect={:?}",
            c.rect
        );
        assert!(c.children.is_empty(), "detached cell must drop children");
        assert!(
            c.collapsed_borders.is_none(),
            "border winners leaked onto a measured-only cell: {:?}",
            c.collapsed_borders
        );
    }

    #[test]
    fn skipped_cell_must_not_carry_collapsed_borders() {
        // A cell that is neither laid out nor measured — a
        // `rowspan=2` cell spanning only a collapsed column — stays
        // zero-area and must not carry border winners either. The
        // detach-only path cannot catch this variant: the cell never
        // reaches the detach block.
        let l = layout_table_html(
            "<table><col><col class=\"gone\">\
             <tr><td>a</td><td rowspan=\"2\">b</td></tr>\
             <tr><td>c</td></tr></table>",
            "table { border-collapse: collapse; } \
             td { padding: 0; border: 4px solid red; } \
             col.gone { visibility: collapse; }",
        );
        let c = cell_at(&l, 0, 1);
        assert!(
            approx(c.rect.w, 0.0) && approx(c.rect.h, 0.0),
            "skipped rect={:?}",
            c.rect
        );
        assert!(
            c.collapsed_borders.is_none(),
            "border winners leaked onto a skipped cell: {:?}",
            c.collapsed_borders
        );
    }

    #[test]
    fn visible_cell_still_carries_collapsed_borders() {
        // Control: the visible cell in the same collapsed-column table
        // keeps its border winners.
        let l = layout_table_html(
            "<table><col><col class=\"gone\">\
             <tr><td>a</td><td>b</td></tr></table>",
            "table { border-collapse: collapse; } \
             td { padding: 0; border: 4px solid red; } \
             col.gone { visibility: collapse; }",
        );
        let c = cell_at(&l, 0, 0);
        let e = c
            .collapsed_borders
            .as_ref()
            .expect("visible cell keeps winners");
        assert!(e.iter().all(|b| approx(b.width, 4.0)));
        assert!(e.iter().all(|b| b.style == CollapsedBorderStyle::Solid));
    }

    // ---- <col> / <colgroup> boxes, spans, and column structure ----

    #[test]
    fn colspan_zero_spans_to_end_of_table() {
        // `colspan="0"` spans from its start column to the end of the
        // table's columns.
        let (forest, table) = forest_table(
            "<table><tr><td>a</td><td colspan=\"0\">b</td></tr>\
             <tr><td>c</td><td>d</td><td>e</td></tr></table>",
            "",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.ncols, 3);
        // Cells are in DOM order: b is cells[1].
        assert_eq!(grid.cells[1].col, 1);
        assert_eq!(grid.cells[1].colspan, 2);
    }

    #[test]
    fn colspan_zero_stops_at_column_group_end() {
        // `colspan="0"` spans to the end of its column *group*. The
        // zero-span cell is placed after its row's other cells, so `b`
        // starts at column 2 inside the second colgroup (columns 2-3)
        // and spans exactly one column.
        let (forest, table) = forest_table(
            "<table><colgroup><col><col></colgroup><colgroup><col></colgroup>\
             <tr><td>a</td><td colspan=\"0\">b</td><td>c</td></tr></table>",
            "",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.ncols, 3);
        assert_eq!(grid.cells[1].col, 2);
        assert_eq!(grid.cells[1].colspan, 1);
        // The per-column descriptors point at the declared boxes.
        assert_eq!(grid.columns.len(), 3);
        let g0 = grid.columns[0].group_node.expect("group 0");
        assert_eq!(grid.columns[1].group_node, Some(g0));
        assert_ne!(grid.columns[2].group_node, Some(g0));
        assert!(grid.columns[0].col_node.is_some());
        assert!(grid.col_nodes.len() >= 3);
        assert_eq!(grid.col_group_nodes.len(), 2);
    }

    #[test]
    fn rowspan_zero_spans_to_section_end() {
        // `rowspan="0"` spans to the last row of its row-group section,
        // not into the next section.
        let (forest, table) = forest_table(
            "<table><tbody><tr><td rowspan=\"0\">a</td><td>b</td></tr>\
             <tr><td>c</td><td>d</td></tr></tbody>\
             <tbody><tr><td>e</td><td>f</td></tr></tbody></table>",
            "",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.nrows, 3);
        assert_eq!(grid.cells[0].rowspan, 2);
    }

    /// No two cells' final rectangles may share a grid slot.
    fn assert_no_cell_overlap(grid: &super::TableGrid) {
        for (i, a) in grid.cells.iter().enumerate() {
            for b in grid.cells.iter().skip(i + 1) {
                let overlap = a.row < b.row + b.rowspan
                    && b.row < a.row + a.rowspan
                    && a.col < b.col + b.colspan
                    && b.col < a.col + a.colspan;
                assert!(!overlap, "cells {i} ({a:?}) and ({b:?}) overlap");
            }
        }
    }

    #[test]
    fn rowspan_zero_displaces_instead_of_overlapping() {
        // `a`'s `rowspan="0"` resolves against the section end during
        // placement, so `c` is placed around it at (1,1) instead of at
        // (1,0), which the resolved span would then cover.
        let (forest, table) = forest_table(
            "<table><tr><td rowspan=\"0\">a</td><td>b</td></tr>\
             <tr><td>c</td><td>d</td></tr></table>",
            "",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.nrows, 2);
        assert_eq!(grid.ncols, 3);
        // DOM order is kept in `grid.cells`: a, b, c, d.
        assert_eq!((grid.cells[0].row, grid.cells[0].col), (0, 0));
        assert_eq!(grid.cells[0].rowspan, 2);
        assert_eq!((grid.cells[1].row, grid.cells[1].col), (0, 1));
        assert_eq!((grid.cells[2].row, grid.cells[2].col), (1, 1));
        assert_eq!((grid.cells[3].row, grid.cells[3].col), (1, 2));
        assert_no_cell_overlap(&grid);
    }

    #[test]
    fn colspan_zero_deferred_past_following_cells() {
        // A `colspan="0"` cell is placed after its row's other cells, so
        // `b` keeps column 0 and `a` spans from column 1 to the end of
        // the table instead of covering `b` once its span resolves.
        let (forest, table) = forest_table(
            "<table><tr><td colspan=\"0\">a</td><td>b</td></tr>\
             <tr><td>c</td><td>d</td><td>e</td></tr></table>",
            "",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.ncols, 3);
        // DOM order is kept in `grid.cells`: a, b, c, d, e.
        assert_eq!((grid.cells[0].row, grid.cells[0].col), (0, 1));
        assert_eq!(grid.cells[0].colspan, 2);
        assert_eq!((grid.cells[1].row, grid.cells[1].col), (0, 0));
        assert_no_cell_overlap(&grid);
    }

    #[test]
    fn floated_table_with_wide_caption_sizes_to_caption() {
        // Regression: `table_pref_width` ignored captions, so a floated
        // (shrink-to-fit) table with a caption wider than its columns
        // sized to the columns only. The caption's preferred width must
        // floor the table's preferred width.
        //
        // The caption is 40 breakable one-letter words: its min-content
        // width is a single word (9.6px), which the existing CAPMIN
        // widening in `layout_block` already covers, but its preferred
        // width is the whole unwrapped line — only `table_pref_width`
        // sees that, so this test discriminates the fix.
        let caption: String = std::iter::repeat_n("w", 40).collect::<Vec<_>>().join(" ");
        let html = format!(
            "<table style=\"float:left\"><caption>{caption}</caption>\
             <tr><td>x</td></tr></table>"
        );
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet("");
        let l = layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics);
        let table = &l.children[0].children[0];
        // Preferred width = full unwrapped line; FixedFontMetrics
        // advances every char (letters and spaces) 0.6 * 16px.
        let cap_pref = caption.chars().count() as f32 * 0.6 * 16.0;
        assert!(
            table.rect.w >= cap_pref,
            "floated table width {} < caption preferred width {cap_pref}",
            table.rect.w
        );
    }

    #[test]
    fn multirow_colspan_zero_displaces_later_rows() {
        // `colspan="0"` combined with `rowspan="2"`: the resolved span
        // covers rows 0-1, columns 1-2, so `d` is displaced to column 3
        // instead of overlapping at (1,2). The `colspan="0"` cell defines
        // its tracks against auto-mode merging.
        let (forest, table) = forest_table(
            "<table><tr><td colspan=\"0\" rowspan=\"2\">a</td><td>b</td></tr>\
             <tr><td>c</td><td>d</td></tr></table>",
            "",
        );
        let grid = collect_table_grid(&forest, table);
        // DOM order is kept in `grid.cells`: a, b, c, d.
        assert_eq!((grid.cells[0].row, grid.cells[0].col), (0, 1));
        assert_eq!(grid.cells[0].colspan, 2);
        assert_eq!(grid.cells[0].rowspan, 2);
        assert_eq!((grid.cells[3].row, grid.cells[3].col), (1, 3));
        assert_no_cell_overlap(&grid);
    }

    #[test]
    fn stray_block_gets_anonymous_wrappers_not_caption() {
        // A block-level box directly under the table that is not a proper
        // table child and not a caption (here a `tbody` restyled to
        // `display: block`; the HTML parser foster-parents raw stray
        // markup, but author styles and script-built DOM can still produce
        // such boxes) is wrapped in an anonymous table row and cell in
        // place (CSS 2.1 §17.2.1): it joins the grid instead of being
        // hoisted above the table like a caption.
        let (mut forest, table) = forest_table(
            "<table><tbody style=\"display:block\"><tr><td>x</td></tr></tbody></table>",
            "",
        );
        // Sanity: the tbody's box is a plain block, not a row group.
        assert_eq!(forest.nodes[table].children.len(), 1);
        let stray = forest.nodes[table].children[0];
        assert_eq!(forest.nodes[stray].box_type, crate::BoxType::Block);
        wrap_table_stray_content(&mut forest, table);
        let anon_row = forest.nodes[table].children[0];
        assert_eq!(forest.nodes[anon_row].box_type, crate::BoxType::TableRow);
        let anon_cell = forest.nodes[anon_row].children[0];
        assert_eq!(forest.nodes[anon_cell].box_type, crate::BoxType::TableCell);
        assert!(forest.nodes[anon_cell].children.contains(&stray));
        let grid = collect_table_grid(&forest, table);
        assert!(grid.captions.is_empty());
        assert_eq!(grid.nrows, 1);
        assert_eq!(grid.cells.len(), 1);
        assert_eq!(grid.cells[0].node, anon_cell);
    }

    #[test]
    fn stray_runs_split_around_proper_children() {
        // Consecutive stray runs each get their own anonymous row; proper
        // table children between them break the runs.
        let (mut forest, table) = forest_table(
            "<table><tbody style=\"display:block\">a</tbody>\
             <tr><td>x</td></tr>\
             <tbody style=\"display:block\">b</tbody></table>",
            "",
        );
        wrap_table_stray_content(&mut forest, table);
        let kids = &forest.nodes[table].children;
        assert_eq!(kids.len(), 3);
        for &k in &[kids[0], kids[2]] {
            assert_eq!(forest.nodes[k].box_type, crate::BoxType::TableRow);
            assert_eq!(
                forest.nodes[forest.nodes[k].children[0]].box_type,
                crate::BoxType::TableCell
            );
        }
        // The middle `<tr>` carries its parser-implied `tbody`.
        assert_eq!(
            forest.nodes[kids[1]].box_type,
            crate::BoxType::TableRowGroup
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.captions.is_empty());
        assert_eq!(grid.nrows, 3);
        assert_eq!(grid.cells.len(), 3);
    }

    #[test]
    fn real_caption_stays_a_caption() {
        // A `<caption>` element keeps caption treatment (full-width,
        // split by `caption-side`) and is not swept into an anonymous row.
        let (mut forest, table) = forest_table(
            "<table><caption>cap</caption><tr><td>x</td></tr></table>",
            "",
        );
        wrap_table_stray_content(&mut forest, table);
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.captions.len(), 1);
        assert_eq!(grid.nrows, 1);
        let cap = grid.captions[0];
        assert_eq!(forest.nodes[cap].box_type, crate::BoxType::Block);
        assert!(forest.nodes[table].children.contains(&cap));
    }

    #[test]
    fn colgroup_span_attribute_declares_columns() {
        // A `colgroup` with no `col` children spans its own `span`
        // attribute's columns. (A bare `col` under the table is wrapped
        // by the parser in an implicit colgroup.) Fixed layout preserves
        // all declared columns (no auto-mode track merging), so the
        // declaration mapping is verified here; see
        // `auto_column_track_merging` for the merging case.
        let (forest, table) = forest_table(
            "<table style=\"table-layout:fixed;width:500px\"><colgroup span=\"2\"></colgroup>\
             <col span=\"3\">\
             <tr><td>x</td></tr></table>",
            "",
        );
        let grid = collect_table_grid(&forest, table);
        // 2 (explicit group) + 3 (bare col) declared columns beat the 1
        // cell column.
        assert_eq!(grid.ncols, 5);
        assert_eq!(grid.columns.len(), 5);
        let g0 = grid.columns[0].group_node.expect("explicit group");
        assert_eq!(grid.columns[1].group_node, Some(g0));
        assert!(grid.columns[0].col_node.is_none());
        let g1 = grid.columns[2].group_node.expect("implicit group");
        assert_ne!(g0, g1);
        assert_eq!(grid.columns[3].group_node, Some(g1));
        assert_eq!(grid.columns[4].group_node, Some(g1));
        let col = grid.columns[2].col_node.expect("bare col covers 2-4");
        assert_eq!(grid.columns[3].col_node, Some(col));
        assert_eq!(grid.columns[4].col_node, Some(col));
        assert_eq!(grid.col_group_nodes.len(), 2);
    }

    #[test]
    fn collapsed_border_column_claim_beats_table() {
        // Column borders claim the vertical grid edges: the col's 4px
        // right border beats the table's 2px outer border on the shared
        // edge (element precedence column > table).
        let collapsed = collapsed_forest(
            "<table><colgroup><col></colgroup><tr><td>x</td><td>y</td></tr></table>",
            "table { border-collapse: collapse; border: 2px solid blue; } \
             col { border-right: 4px solid red; } td { font-size: 10px; }",
        );
        let right = collapsed.cells[0][1];
        assert!(approx(right.width, 4.0));
        assert_eq!(right.style, CollapsedBorderStyle::Solid);
        assert_eq!(right.color, RED);
    }

    #[test]
    fn collapsed_border_column_group_claim_participates() {
        // The column-group's left border claims the table's outer left
        // edge (columns span the full grid height).
        let collapsed = collapsed_forest(
            "<table><colgroup><col><col></colgroup><tr><td>x</td><td>y</td></tr></table>",
            "table { border-collapse: collapse; } \
             colgroup { border-left: 6px solid green; } td { font-size: 10px; }",
        );
        let left = collapsed.cells[0][3];
        assert!(approx(left.width, 6.0));
        assert_eq!(left.style, CollapsedBorderStyle::Solid);
        assert_eq!(left.color, GREEN);
    }

    #[test]
    fn collapsed_border_bare_colgroup_claims_left_interior_edge() {
        // A bare `<colgroup span="2">` (no `<col>` children) still claims
        // the interior vertical edge through its right border: the column
        // to the left of the edge has no column box to gate the claim on.
        let collapsed = collapsed_forest(
            "<table><colgroup span=\"2\"></colgroup><tr><td>x</td><td>y</td></tr></table>",
            "table { border-collapse: collapse; } \
             colgroup { border-right: 6px solid green; } td { font-size: 10px; }",
        );
        // y's left edge borders column 0, covered only by the group.
        let left = collapsed.cells[1][3];
        assert!(approx(left.width, 6.0));
        assert_eq!(left.style, CollapsedBorderStyle::Solid);
        assert_eq!(left.color, GREEN);
    }

    #[test]
    fn collapsed_border_bare_colgroup_claims_right_interior_edge() {
        // Same, for the column to the right of the edge: the bare
        // colgroup's left border claims the interior edge with no column
        // box present.
        let collapsed = collapsed_forest(
            "<table><colgroup span=\"2\"></colgroup><tr><td>x</td><td>y</td></tr></table>",
            "table { border-collapse: collapse; } \
             colgroup { border-left: 6px solid green; } td { font-size: 10px; }",
        );
        // x's right edge borders column 1, covered only by the group.
        let right = collapsed.cells[0][1];
        assert!(approx(right.width, 6.0));
        assert_eq!(right.style, CollapsedBorderStyle::Solid);
        assert_eq!(right.color, GREEN);
    }

    #[test]
    fn fixed_layout_colgroup_width_pins_each_column() {
        // WPT `colgroup-col`: the 200px group width pins each covered
        // column without its own `col` width at the full 200px (not a
        // 50px spread remainder). col0's own 150px `col` width wins for
        // its column. Both columns are length-pinned, so the 250px
        // excess distributes proportionally to the pinned widths (CSS
        // Tables 3 §3.9.3.2 rule 2): 150 + 107.14 and 200 + 142.86.
        let l = layout_table_html(
            "<table><colgroup class=\"g\"><col class=\"a\"><col></colgroup>\
             <tr><td>x</td><td>y</td></tr></table>",
            "table { table-layout: fixed; width: 600px; } \
             colgroup.g { width: 200px; } col.a { width: 150px; } td { font-size: 10px; }",
        );
        assert!(approx(
            cell_at(&l, 0, 0).rect.w,
            150.0 + 250.0 * 150.0 / 350.0
        ));
        assert!(approx(
            cell_at(&l, 0, 1).rect.w,
            200.0 + 250.0 * 200.0 / 350.0
        ));
    }

    // ---- automatic layout: max-content distribution (CSS 2.1 §17.5.2.2) ----

    #[test]
    fn auto_layout_grows_columns_toward_max_content() {
        // "bb cc dd": min 12 (longest word), max 48 (no breaks). Extra
        // table width first grows columns toward their maxima
        // proportionally to (max − min): the wrappable column takes all
        // 36 of growth, reaching [12, 48]. The 740 remainder then goes
        // to the non-constrained columns proportionally to max-content
        // (CSS Tables 3 §3.9.3.2): 12/60 and 48/60 of 740. The width is
        // specified so the distribution phases see excess width; with
        // `width: auto` the table would shrink-to-fit to 60 (see
        // `auto_width_table_shrink_to_fit`).
        let l = layout_table_html(
            "<table><tr><td>aa</td><td>bb cc dd</td></tr></table>",
            "table { width: 800px; } td { font-size: 10px; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 160.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 640.0));
    }

    #[test]
    fn auto_layout_max_excess_grows_column_toward_max() {
        // Single-column path (no colspan): column 0 has the only
        // (max − min) headroom, so the phase-1 growth lands entirely on
        // it, reaching [102, 12]; the 686 remainder goes to the
        // non-constrained columns proportionally to max-content
        // (CSS Tables 3 §3.9.3.2). Specified width, as above.
        let l = layout_table_html(
            "<table><tr><td>aa bb cc dd ee ff</td><td>gg</td></tr></table>",
            "table { width: 800px; } td { font-size: 10px; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 715.79));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 84.21));
    }

    #[test]
    fn auto_width_table_shrink_to_fit() {
        // A block-level table with `width: auto` does not fill its
        // containing block: the used width is max(MAX, CAPMIN) when that
        // is narrower than the containing block (CSS 2.1 §17.5.2.2).
        let l = layout_table_html(
            "<table><tr><td>aa</td><td>bb cc dd</td></tr></table>",
            "td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 60.0));
        assert!(approx(cell_at(&l, 0, 0).rect.w, 12.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 48.0));

        let l = layout_table_html(
            "<table><tr><td>aa bb cc dd ee ff</td><td>gg</td></tr></table>",
            "td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 114.0));
        assert!(approx(cell_at(&l, 0, 0).rect.w, 102.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 12.0));
    }

    #[test]
    fn fixed_pct_table_in_auto_table_cell_outer_fills_container() {
        // WPT `css/css-tables/tentative/table-fixed-minmax.html`: a
        // fixed-layout table with a percentage width nested in an
        // automatic-layout table cell has an infinite max-content
        // intrinsic size, so the outer `width: auto` table's
        // shrink-to-fit clamps to the available width instead of the
        // finite content size. The sizing-guess interpolation must not
        // produce NaN from `0.0 * (INF - min)`.
        let l = layout_table_html(
            "<table class=\"o\"><tr><td><table class=\"i\"><tr><td></td></tr></table></td></tr></table>",
            "body { margin: 0; } \
             table.o { table-layout: auto; width: auto; } \
             table.i { table-layout: fixed; width: 100%; border-spacing: 0; } \
             table.i td { width: 100px; padding: 0; }",
        );
        let outer = &l.children[0].children[0];
        assert!(outer.rect.w.is_finite(), "outer w={}", outer.rect.w);
        assert!(approx(outer.rect.w, 800.0), "outer w={}", outer.rect.w);
    }

    #[test]
    fn auto_width_table_min_width_floor() {
        // MIN floors the shrink-to-fit width: unbreakable content wider
        // than the containing block grows the table past it (140 a's at
        // 6px each = 840 > 800).
        let l = layout_table_html(
            &format!("<table><tr><td>{}</td></tr></table>", "a".repeat(140)),
            "td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 840.0));
        assert!(approx(cell_at(&l, 0, 0).rect.w, 840.0));
    }

    #[test]
    fn auto_width_table_caption_min_width() {
        // CAPMIN: the caption's outer min-content width (30 a's = 180)
        // beats the columns' MAX (12), so the table sizes to the caption
        // and the column absorbs the extra width.
        let l = layout_table_html(
            &format!(
                "<table><caption>{}</caption><tr><td>hi</td></tr></table>",
                "a".repeat(30)
            ),
            "td, caption { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 180.0));
        assert!(approx(cell_at(&l, 0, 0).rect.w, 180.0));
        let caption = table
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::Block)
            .expect("caption");
        assert!(approx(caption.rect.w, 180.0));
    }

    #[test]
    fn caption_negative_top_margin_pulls_caption_up() {
        // CSS 2.1 §17.5.3: caption margins apply as specified — they
        // are never clamped to zero. `margin-top: -10px` pulls the
        // caption's border box 10px above where it sits with a zero
        // top margin.
        let caption_y = |margin: &str| {
            let l = layout_table_html(
                "<table><caption>cap</caption><tr><td>x</td></tr></table>",
                &format!("td, caption {{ font-size: 10px; }} caption {{ margin-top: {margin}; }}"),
            );
            l.children[0].children[0]
                .children
                .iter()
                .find(|c| c.box_type == crate::BoxType::Block)
                .expect("caption")
                .rect
                .y
        };
        let y_zero = caption_y("0px");
        let y_neg = caption_y("-10px");
        assert!(
            approx(y_neg, y_zero - 10.0),
            "negative margin-top should pull the caption up 10px: {y_neg} vs {y_zero}"
        );
    }

    #[test]
    fn specified_width_table_caption_min_width() {
        // CSS 2.1 §17.5.2.2 errata rule 1: the used width is
        // max(W, CAPMIN, MIN), so a specified width narrower than the
        // caption's outer min-content width (30 a's = 180) grows the
        // table to the caption instead of overflowing it.
        let l = layout_table_html(
            &format!(
                "<table><caption>{}</caption><tr><td>hi</td></tr></table>",
                "a".repeat(30)
            ),
            "table { width: 100px; } td, caption { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 180.0));
        let caption = table
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::Block)
            .expect("caption");
        assert!(approx(caption.rect.w, 180.0));
    }

    #[test]
    fn specified_width_table_wider_than_caption_unchanged() {
        // Control: a specified width already past CAPMIN (200 > 180)
        // leaves the table at the specified width.
        let l = layout_table_html(
            &format!(
                "<table><caption>{}</caption><tr><td>hi</td></tr></table>",
                "a".repeat(30)
            ),
            "table { width: 200px; } td, caption { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 200.0));
    }

    #[test]
    fn fixed_layout_table_caption_min_width() {
        // CSS 2.1 §17.5.2.2 errata rule 1 under `table-layout: fixed`:
        // the used width is max(W, columns' sum, CAPMIN), so a specified
        // width narrower than the caption's outer min-content width
        // (30 a's = 180) grows the table to the caption instead of
        // overflowing it.
        let l = layout_table_html(
            &format!(
                "<table><caption>{}</caption><tr><td>hi</td></tr></table>",
                "a".repeat(30)
            ),
            "table { table-layout: fixed; width: 100px; } td, caption { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 180.0));
        let caption = table
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::Block)
            .expect("caption");
        assert!(approx(caption.rect.w, 180.0));
    }

    #[test]
    fn auto_width_table_auto_margins_center() {
        // Shrink-to-fit tables still honor `margin: auto` centering
        // (CSS 2.1 §10.3.3): the 60px table centers in the 800px viewport.
        let l = layout_table_html(
            "<table><tr><td>aa</td><td>bb cc dd</td></tr></table>",
            "table { margin: 0 auto; } td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 60.0));
        assert!(approx(table.rect.x, 370.0));
    }

    #[test]
    fn auto_layout_over_constrained_grows_table() {
        // Min-content (180) exceeds the specified width (100): the table
        // grows to the columns' sum instead of squeezing below
        // min-content (CSS 2.1 §17.5.2.2: used width is the greater of the
        // specified width and the columns' sum).
        let l = layout_table_html(
            "<table><tr><td>aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</td></tr></table>",
            "table { width: 100px; } td { font-size: 10px; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 180.0));
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 180.0));
    }

    #[test]
    fn auto_layout_spanning_cell_distributes_max_excess() {
        // A colspan=2 cell whose preferred width (192) exceeds the
        // spanned columns' maxima (72 + 24 = 96) distributes the max
        // excess per §3.9.3.2 (for intrinsic width calculation):
        // proportionally to max-content, so the maxima reach
        // [144, 48] — not the even [120, 72] spread of the old
        // simplification (WPT `colspan-redistribution` table 9 pins
        // the proportional rule). The 48 remainder past the
        // max-content guess then goes to the non-constrained columns
        // proportionally to max-content (§3.9.3.2 for used width):
        // 144/192 and 48/192 of 48, landing at 180/60.
        let l = layout_table_html(
            "<table><tr><td colspan=\"2\">bbbbbbbbbbbbbbb cccccccccccccccc</td></tr>\
             <tr><td>aaaaaaaaaaaa</td><td>aaaa</td></tr></table>",
            "table { width: 240px; } td { font-size: 10px; }",
        );
        assert!(approx(cell_at(&l, 1, 0).rect.w, 180.0));
        assert!(approx(cell_at(&l, 1, 1).rect.w, 60.0));
    }

    #[test]
    fn auto_layout_cell_min_width_prevents_collapse() {
        // `min-width` floors the column minimum: the 200px floor beats
        // both min-content (6) and the 100px table width, so the table
        // grows to the columns' sum instead of squeezing the column below
        // the floor.
        let l = layout_table_html(
            "<table><tr><td>x</td></tr></table>",
            "table { width: 100px; } td { font-size: 10px; min-width: 200px; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 200.0));
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 200.0));
    }

    #[test]
    fn auto_layout_spanning_cell_min_width_distributes() {
        // A colspan=2 cell's `min-width` spreads over the spanned columns
        // like any other minimum excess: 240px across two columns lands
        // 120 on each, and the table grows to the columns' sum.
        let l = layout_table_html(
            "<table><tr><td colspan=\"2\" class=\"w\">x</td></tr>\
             <tr><td>y</td><td>z</td></tr></table>",
            "table { width: 100px; } td { font-size: 10px; } td.w { min-width: 240px; }",
        );
        assert!(approx(cell_at(&l, 1, 0).rect.w, 120.0));
        assert!(approx(cell_at(&l, 1, 1).rect.w, 120.0));
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 240.0));
    }

    #[test]
    fn specified_cell_width_does_not_floor_column_minimum() {
        // CSS Tables 3 §3.8.3: the outer min-content width of a cell is
        // `max(min-width, min-content width)` — a specified `width`
        // floors the column *maximum*, not the minimum. A `width:
        // 100px` cell whose content needs 75px therefore lets a
        // `width: 1px` table shrink to 75px (WPT
        // `distribution-algo-min-content-specified-guess`: the box is
        // its min-content size, not the specified width).
        let l = layout_table_html(
            "<table><tr><td class=\"a\"><div class=\"d\"></div></td></tr></table>",
            "table { width: 1px; } td { font-size: 10px; padding: 0; } \
             td.a { width: 100px; } div.d { width: 75px; height: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 75.0), "table={}", table.rect.w);
        assert!(approx(cell_at(&l, 0, 0).rect.w, 75.0));
    }

    #[test]
    fn constrained_cell_width_caps_column_maximum() {
        // §3.8.3 outer max-content of a cell in a constrained column:
        // `max(min-content, width)` — the specified width replaces the
        // max-content width. A `width: 60px` cell with min-content 12
        // and max-content 102 contributes 60 to the column maximum, so
        // a shrink-to-fit table is 60 wide, not 102 (WPT
        // `column-widths`: "td.max = max(minmax.min, css.width)" for a
        // constrained TD).
        let l = layout_table_html(
            "<table><tr><td class=\"a\">aa bb cc dd ee ff gg hh</td></tr></table>",
            "td { font-size: 10px; padding: 0; } td.a { width: 60px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 60.0), "table={}", table.rect.w);
        assert!(approx(cell_at(&l, 0, 0).rect.w, 60.0));
    }

    #[test]
    fn guess_interpolation_percentage_segment() {
        // CSS Tables 3 §3.9.3: between the min-content guess (150)
        // and the min-content-percentage guess (180), only the
        // percentage column moves. Assignable 160 is a third of the
        // way, so the 50% column grows 50 -> 60 while the auto and
        // constrained columns stay at their minima (WPT
        // `table-width-redistribution` table 10).
        let l = layout_table_html(
            "<table><tr><td><div class=\"d\"></div></td>\
             <td class=\"f\"><div class=\"d\"></div></td>\
             <td class=\"p\"><div class=\"d\"></div></td></tr></table>",
            "table { width: 192px; border-spacing: 8px; } \
             td { font-size: 10px; padding: 0; } \
             td.f { width: 100px; } td.p { width: 50%; } \
             div.d { width: 50px; height: 10px; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 50.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 50.0));
        assert!(approx(cell_at(&l, 0, 2).rect.w, 60.0));
    }

    #[test]
    fn guess_interpolation_specified_segment() {
        // Between the min-content-percentage guess (84) and the
        // min-content-specified guess (172), only the constrained
        // column moves: at assignable 150 it takes 75% of its
        // min->max climb (12 -> 78), the percentage column holds its
        // percentage assignment (40% of 150 = 60), and the auto column
        // stays at its minimum (WPT `table-width-redistribution`
        // table 14).
        let l = layout_table_html(
            "<table><tr><td>aa</td><td class=\"f\">aa</td><td class=\"p\">aa</td></tr></table>",
            "table { width: 182px; border-spacing: 8px; } \
             td { font-size: 10px; padding: 0; } \
             td.f { width: 100px; } td.p { width: 40%; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 12.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 78.0));
        assert!(approx(cell_at(&l, 0, 2).rect.w, 60.0));
    }

    #[test]
    fn guess_interpolation_max_content_segment() {
        // Between the min-content-specified guess (192) and the
        // max-content guess (210), only the auto column moves: at
        // assignable 200 it climbs 4/9 of its min->max span (12 ->
        // 20), the constrained column holds its maximum, and the
        // percentage column holds 40% of 200 (WPT
        // `table-width-redistribution` table 15).
        let l = layout_table_html(
            "<table><tr><td>aa aa</td><td class=\"f\">aa</td><td class=\"p\">aa</td></tr></table>",
            "table { width: 232px; border-spacing: 8px; } \
             td { font-size: 10px; padding: 0; } \
             td.f { width: 100px; } td.p { width: 40%; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 20.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 100.0));
        assert!(approx(cell_at(&l, 0, 2).rect.w, 80.0));
    }

    #[test]
    fn spanning_excess_distributes_proportionally_to_max_content() {
        // §3.9.3.2 for intrinsic width calculation: a spanning cell's
        // excess over the spanned columns distributes proportionally
        // to their max-content widths, not equally. Columns at 75/25
        // with a 300px spanning cell (8px spacing inside the span)
        // take 144/48 of the 192 excess, landing at 219/73 (WPT
        // `colspan-redistribution` table 9).
        let l = layout_table_html(
            "<table><tr><td><div class=\"a\"></div></td><td><div class=\"b\"></div></td>\
             <td class=\"f\">x</td></tr>\
             <tr><td colspan=\"2\"><div class=\"c\"></div></td></tr></table>",
            "table { border-spacing: 8px; } td { font-size: 10px; padding: 0; } \
             td.f { width: 20px; } \
             div.a { width: 75px; height: 10px; } div.b { width: 25px; height: 10px; } \
             div.c { width: 300px; height: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 344.0), "table={}", table.rect.w);
        assert!(approx(cell_at(&l, 0, 0).rect.w, 219.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 73.0));
        assert!(approx(cell_at(&l, 1, 0).rect.w, 300.0));
    }

    #[test]
    fn spanning_excess_avoids_constrained_columns() {
        // §3.9.3.2 rule 1: the whole spanning excess (min and max)
        // lands on the non-constrained spanned columns. A 500px
        // colspan=2 cell over an auto column (40) and a constrained
        // column (width: 150px, content 100) leaves the constrained
        // column at 100/150 and grows the auto column to min 392 /
        // max 342; the used widths land on the maxima, 342/150 (WPT
        // `colspan-redistribution` table 12).
        let l = layout_table_html(
            "<table><tr><td><div class=\"a\"></div></td>\
             <td class=\"f\"><div class=\"b\"></div></td></tr>\
             <tr><td colspan=\"2\"><div class=\"c\"></div></td></tr></table>",
            "table { border-spacing: 8px; } td { font-size: 10px; padding: 0; } \
             td.f { width: 150px; } \
             div.a { width: 40px; height: 10px; } div.b { width: 100px; height: 10px; } \
             div.c { width: 500px; height: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 516.0), "table={}", table.rect.w);
        assert!(approx(cell_at(&l, 0, 0).rect.w, 342.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 150.0));
        assert!(approx(cell_at(&l, 1, 0).rect.w, 500.0));
    }

    #[test]
    fn pct_columns_floor_intrinsic_table_width() {
        // A percentage column's minimum divided by its percentage
        // floors the table's intrinsic width: two 10% columns with
        // 20px minima imply an assignable width of at least 200px,
        // so the shrink-to-fit table is 200 + 4*8 spacing = 232 wide
        // even though the columns' own measures sum to ~46 (WPT
        // `colspan-redistribution` table 1). At used time the
        // percentage columns hold 20 each and the surplus lands on
        // the auto column (§3.9.3.2 rule 1).
        let l = layout_table_html(
            "<table class=\"t\"><tr><td class=\"p\"><div class=\"d\"></div></td>\
             <td class=\"p\"><div class=\"d\"></div></td><td>x</td></tr></table>",
            "table.t { float: left; border-spacing: 8px; } \
             td { font-size: 10px; padding: 0; } td.p { width: 10%; } \
             div.d { width: 20px; height: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 232.0), "table={}", table.rect.w);
        assert!(approx(cell_at(&l, 0, 0).rect.w, 20.0));
        assert!(approx(cell_at(&l, 0, 2).rect.w, 160.0));
    }

    #[test]
    fn pct_floor_combined_non_pct_minima() {
        // The combined floor: with P = 60% in percentage columns and
        // F = 100px of non-percentage minima, the assignable width is
        // at least F / (1 − P) = 250, so the table is 250 + 3*8
        // spacing = 274 wide and the percentage column receives 60%
        // of 250 = 150 (WPT `colspan-redistribution` table 14).
        let l = layout_table_html(
            "<table class=\"t\"><tr><td class=\"p\"><div class=\"d\"></div></td>\
             <td class=\"f\"><div class=\"d\"></div></td></tr></table>",
            "table.t { float: left; border-spacing: 8px; } \
             td { font-size: 10px; padding: 0; } \
             td.p { width: 60%; } td.f { width: 100px; } \
             div.d { width: 100px; height: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 274.0), "table={}", table.rect.w);
        assert!(approx(cell_at(&l, 0, 0).rect.w, 150.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 100.0));
    }

    #[test]
    fn auto_layout_mixed_fixed_and_auto_columns() {
        // Mixed fixed/auto: the `<col>` width is the first column's
        // minimum (a specified column width participates as a minimum per
        // CSS 2.1 §17.5.2.2) while the second column sizes from content.
        // With the table width exactly at the minima's sum (212), neither
        // column collapses nor steals from the other: 200 stays 200 and
        // the content column lands on its min-content 12.
        let l = layout_table_html(
            "<table><col class=\"a\"><tr><td>aa</td><td>bb cc dd</td></tr></table>",
            "table { width: 212px; } col.a { width: 200px; } td { font-size: 10px; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 200.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 12.0));
    }

    #[test]
    fn auto_layout_floated_table_shrink_to_fit() {
        // A floated table with `width: auto` shrink-to-fits: the used
        // width is the preferred (max-content) column sum, not the full
        // containing block (CSS 2.1 §10.3.5 / §17.5.2.2).
        let l = layout_table_html(
            "<table class=\"f\"><tr><td>aa bb</td></tr></table>",
            "table.f { float: left; } td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 30.0));
        assert!(approx(cell_at(&l, 0, 0).rect.w, 30.0));
    }

    #[test]
    fn auto_layout_floated_table_min_width_floors_shrink_to_fit() {
        // The shrink-to-fit width also honors `min-width`: the preferred
        // width (30) falls below the cell's 200px floor, so the table
        // sizes to the floor.
        let l = layout_table_html(
            "<table class=\"f\"><tr><td>aa bb</td></tr></table>",
            "table.f { float: left; } td { font-size: 10px; min-width: 200px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 200.0));
        assert!(approx(cell_at(&l, 0, 0).rect.w, 200.0));
    }

    #[test]
    fn pref_width_specified_cell_width_floors_shrink_to_fit() {
        // CSS 2.1 §17.5.2.2 steps 1-2: a specified cell width floors the
        // column minimum *and* maximum, so the min-content and
        // max-content (preferred) widths treat it symmetrically. Before
        // the fix the preferred width measured content only, so an outer
        // shrink-to-fit wrapper sized to the content width while the
        // table grew past it.
        let l = layout_table_html(
            "<div class=\"w\"><table><tr><td class=\"a\">x</td></tr></table></div>",
            "div.w { float: left; } td { font-size: 10px; padding: 0; } td.a { width: 200px; }",
        );
        // body > div.w > table.
        let wrap = &l.children[0].children[0];
        let table = &wrap.children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        assert!(
            approx(wrap.rect.w, 200.0),
            "shrink-to-fit wrapper should cover the specified cell width: {}",
            wrap.rect.w
        );
        assert!(
            approx(table.rect.w, 200.0),
            "table should tile the specified width: {}",
            table.rect.w
        );
    }

    #[test]
    fn pref_width_specified_spanning_cell_width_floors_shrink_to_fit() {
        // Spanning twin: a colspan=2 cell's specified width floors the
        // spanned maximum, spreading over the span exactly like the
        // minimum side does in `auto_column_minima`.
        let l = layout_table_html(
            "<div class=\"w\"><table><tr><td class=\"a\" colspan=\"2\">x</td></tr>\
             <tr><td>y</td><td>z</td></tr></table></div>",
            "div.w { float: left; } td { font-size: 10px; padding: 0; } td.a { width: 200px; }",
        );
        // body > div.w > table.
        let wrap = &l.children[0].children[0];
        let table = &wrap.children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        assert!(
            approx(wrap.rect.w, 200.0),
            "wrapper should cover the specified spanning width: {}",
            wrap.rect.w
        );
        assert!(approx(table.rect.w, 200.0), "table={}", table.rect.w);
        // table > row-group > row 1 > cells tile the specified width.
        let group = table
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::TableRowGroup)
            .expect("row group");
        let row1 = &group.children[1];
        assert!(
            approx(row1.children[0].rect.w + row1.children[1].rect.w, 200.0),
            "span columns tile the specified width: {} + {}",
            row1.children[0].rect.w,
            row1.children[1].rect.w
        );
    }

    #[test]
    fn pref_width_spanning_spread_deducts_internal_spacing() {
        // A spanning cell's border box covers the inter-column spacing
        // between the spanned columns, so the max-side spread deducts it
        // — mirroring `layout_table`'s maxima and `auto_column_minima`.
        // A colspan=2 cell with `width: 200px` under 10px border-spacing
        // leaves 190px for the columns (95 each), not 200. Before the fix
        // the spread double-counted the 10px gap and the shrink-to-fit
        // wrapper sized to 230px instead of 220px.
        let l = layout_table_html(
            "<div class=\"w\"><table><tr><td class=\"a\" colspan=\"2\">x</td></tr></table></div>",
            "div.w { float: left; } table { border-spacing: 10px; } \
             td { font-size: 10px; padding: 0; } td.a { width: 200px; }",
        );
        // body > div.w > table.
        let wrap = &l.children[0].children[0];
        let table = &wrap.children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        assert!(
            approx(wrap.rect.w, 220.0),
            "shrink-to-fit wrapper should be 190px of columns + 30px spacing: {}",
            wrap.rect.w
        );
        assert!(approx(table.rect.w, 220.0), "table={}", table.rect.w);
    }

    #[test]
    fn fixed_layout_trigger_auto_width_uses_auto_layout() {
        // CSS Tables 3 fixed-mode trigger (csswg #10937): `table-layout:
        // fixed` with a specified `width: auto` lays out with the
        // automatic algorithm, so a specified width on a later row's
        // cell counts: the 200px floors the preferred width instead of
        // being ignored the way fixed mode would ignore it.
        let l = layout_table_html(
            "<div class=\"w\"><table><tr><td>x</td></tr><tr><td class=\"a\">y</td></tr></table></div>",
            "div.w { float: left; } table { table-layout: fixed; width: auto; } \
             td { font-size: 10px; padding: 0; } td.a { width: 200px; }",
        );
        // body > div.w > table.
        let wrap = &l.children[0].children[0];
        let table = &wrap.children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        // Auto layout: the second-row 200px specified width floors the
        // column (separate borders carry no default spacing here).
        assert!(
            approx(wrap.rect.w, 200.0),
            "preferred width should honor the second-row specified width in auto layout: {}",
            wrap.rect.w
        );
        // Shrink-to-fit sizes the table from the preferred width, so the
        // used width agrees: preferred never exceeds used.
        assert!(
            approx(table.rect.w, wrap.rect.w),
            "used width should match preferred: table={} wrap={}",
            table.rect.w,
            wrap.rect.w
        );
    }

    #[test]
    fn fixed_layout_trigger_max_content_width_disables_fixed() {
        // CSS Tables 3 fixed-mode trigger (csswg #10937): `width:
        // max-content` suppresses fixed mode just like `auto` does, so
        // the preferred width stays content-based.
        let (mut forest, table) = forest_table(
            "<table><tr><td>x</td></tr><tr><td>y</td></tr></table>",
            "table { table-layout: fixed; width: max-content; } \
             td { font-size: 10px; padding: 0; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert!(!grid.fixed, "width:max-content must not trigger fixed mode");
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut pool = TableGridPool::new();
        let pref = table_pref_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut cache,
            &mut pool,
            true,
        );
        // One-char cells measure 6px at 10px type; nothing specified to
        // widen the preferred width past the content.
        assert!(approx(pref, 6.0), "preferred {pref}");
    }

    #[test]
    fn fixed_layout_trigger_definite_width_keeps_fixed() {
        // A definite specified width keeps fixed mode: first-row
        // specified cell widths pin the columns and the used table
        // width is the specified one.
        let (forest, table) = forest_table(
            "<table><tr><td class=\"a\">x</td><td>y</td></tr></table>",
            "table { table-layout: fixed; width: 500px; } \
             td { font-size: 10px; padding: 0; } td.a { width: 200px; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.fixed, "width:500px must keep fixed mode");
        let l = layout_table_html(
            "<table><tr><td class=\"a\">x</td><td>y</td></tr></table>",
            "table { table-layout: fixed; width: 500px; } \
             td { font-size: 10px; padding: 0; } td.a { width: 200px; }",
        );
        // body > table.
        let table_box = &l.children[0].children[0];
        assert!(
            approx(table_box.rect.w, 500.0),
            "used table width should be the specified 500px: {}",
            table_box.rect.w
        );
    }

    #[test]
    fn fixed_layout_trigger_auto_table_layout_never_fixed() {
        // `table-layout: auto` never triggers fixed mode, whatever the
        // specified width is.
        for width in ["500px", "50%", "auto", "max-content", "min-content"] {
            let (forest, table) = forest_table(
                "<table><tr><td>x</td></tr></table>",
                &format!(
                    "table {{ table-layout: auto; width: {width}; }} \
                     td {{ font-size: 10px; padding: 0; }}"
                ),
            );
            let grid = collect_table_grid(&forest, table);
            assert!(
                !grid.fixed,
                "table-layout:auto with width:{width} must not be fixed"
            );
        }
    }

    #[test]
    fn fixed_layout_trigger_calc_size_keyword_bases() {
        // calc-size() parses; the fixed-mode trigger is syntactic, so
        // ALL calc-size() bases keep fixed mode on — even `auto` and
        // `max-content`, which as bare keywords would suppress it (WPT
        // fixed-layout-2.html hardcodes the trigger on for every
        // calc-size() variant).
        for width in [
            "calc-size(auto, 50px)",
            "calc-size(max-content, 50px)",
            "calc-size(min-content, 50px)",
            "calc-size(fit-content, 50px)",
            "calc-size(stretch, 50px)",
        ] {
            let (forest, table) = forest_table(
                "<table><tr><td>x</td></tr></table>",
                &format!(
                    "table {{ table-layout: fixed; width: {width}; }} \
                     td {{ font-size: 10px; padding: 0; }}"
                ),
            );
            let grid = collect_table_grid(&forest, table);
            assert!(
                grid.fixed,
                "calc-size width:{width} must keep fixed mode on"
            );
        }
    }

    #[test]
    fn fixed_layout_trigger_unparsed_width_function_disables_fixed() {
        // Robustness: a width function the cascade cannot parse at all is
        // dropped as an invalid declaration; the specified width falls
        // back to `auto`, which suppresses fixed mode.
        for width in ["foo(1px)", "bar(2%)"] {
            let (forest, table) = forest_table(
                "<table><tr><td>x</td></tr></table>",
                &format!(
                    "table {{ table-layout: fixed; width: {width}; }} \
                     td {{ font-size: 10px; padding: 0; }}"
                ),
            );
            let grid = collect_table_grid(&forest, table);
            assert!(
                !grid.fixed,
                "unparsed width:{width} must not trigger fixed mode"
            );
        }
    }

    #[test]
    fn fixed_layout_indefinite_width_sizes_from_pins() {
        // WPT `css/css-tables/fixed-layout-2.html`: in fixed mode a
        // non-definite specified table width (`min-content`,
        // `fit-content`, ...) sizes the table from the pinned first-row
        // column widths (CSS 2.1 §17.5.2.1), never from content
        // measurement. The 100px div inside the 50px-pinned first cell
        // must not widen the table past the pin.
        for width in ["min-content", "fit-content"] {
            let l = layout_table_html(
                "<div class=\"wrap\"><table><tr><td class=\"a\">\
                 <div class=\"wide\">y</div></td><td></td></tr></table></div>",
                &format!(
                    ".wrap {{ width: 0; }} \
                     table {{ table-layout: fixed; width: {width}; border-spacing: 0; }} \
                     td {{ font-size: 10px; padding: 0; }} td.a {{ width: 50px; }} .wide {{ width: 100px; }}"
                ),
            );
            let table_box = &l.children[0].children[0].children[0];
            assert_eq!(table_box.box_type, crate::BoxType::Table);
            assert!(
                approx(table_box.rect.w, 50.0),
                "width:{width}: used table width should be the 50px pin, not content: {}",
                table_box.rect.w
            );
        }
    }

    #[test]
    fn fixed_layout_auto_width_measures_content() {
        // `width: auto` suppresses fixed mode even with
        // `table-layout: fixed`: the automatic layout measures content,
        // so the 100px div in the first cell widens its column (and the
        // table) to 100px.
        let l = layout_table_html(
            "<div class=\"wrap\"><table><tr><td class=\"a\">\
             <div class=\"wide\">y</div></td><td></td></tr></table></div>",
            ".wrap { width: 0; } \
             table { table-layout: fixed; width: auto; border-spacing: 0; } \
             td { font-size: 10px; padding: 0; } td.a { width: 50px; } .wide { width: 100px; }",
        );
        let table_box = &l.children[0].children[0].children[0];
        assert_eq!(table_box.box_type, crate::BoxType::Table);
        assert!(
            approx(table_box.rect.w, 100.0),
            "width:auto must use auto layout and measure the 100px content: {}",
            table_box.rect.w
        );
    }

    #[test]
    fn pref_width_fixed_layout_max_content_covers_min_content() {
        // max-content >= min-content invariant under the fixed-layout
        // first-row-only rule: the second-row specified width must floor
        // neither side. The table's own definite width now governs its
        // max-content contribution outright (css-sizing-3 §5.2), while
        // the min-content side stays content-based and ordered below it.
        let (mut forest, table) = forest_table(
            "<table><tr><td>x</td></tr><tr><td class=\"a\">y</td></tr></table>",
            "table { table-layout: fixed; width: 500px; } \
             td { font-size: 10px; padding: 0; } td.a { width: 200px; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.fixed, "expected a fixed-layout grid");
        let col_spec = column_specified_widths(&mut forest, &grid, 0.0, 0.0);
        let (_, spacing_h, _) = table_border_model(&forest, table);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut mcw_scratch = Vec::new();
        let mut int_bufs = WalkBufs::new();
        let mut int_inner = WalkBufs::new();
        let (col_max, col_max_nonspanning) = auto_column_maxima(
            &mut forest,
            &grid,
            &col_spec,
            spacing_h,
            &FixedFontMetrics,
            &mut cache,
            &mut mcw_scratch,
            &mut int_bufs,
            &mut int_inner,
        );
        let mins = auto_column_minima(
            &mut forest,
            &grid,
            &col_spec,
            spacing_h,
            0.0,
            &FixedFontMetrics,
            &col_max,
            &col_max_nonspanning,
        );
        let mut pool = TableGridPool::new();
        let pref = table_pref_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut cache,
            &mut pool,
            true,
        );
        // The `min_content_width` table arm sums the same minima and
        // adds the inter-column spacing the same way `table_pref_width`
        // does; mirror that composition here.
        let min_total: f32 = mins.iter().sum();
        let nvis = grid.collapsed_cols.iter().filter(|&&c| !c).count();
        let spacing_total = spacing_h * (nvis as f32 + 1.0);
        assert!(
            pref + 0.01 >= min_total + spacing_total,
            "max-content ({pref}) must cover min-content ({min_total} + {spacing_total} spacing)"
        );
        // A definite table width contributes exactly its content-box
        // width (css-sizing-3 §5.2), not the content-based 6px; the
        // second-row specified width still floors neither side of the
        // min-content measure.
        assert!(approx(pref, 500.0), "preferred {pref}");
        assert!(approx(min_total, 6.0), "min-content {min_total}");
    }

    #[test]
    fn pref_width_fixed_layout_first_row_pin_below_content_repaired() {
        // Regression (a): a first-row pin holds the specified 200px
        // while the cell's 240px of unbreakable content pushes the
        // min-content size past it. The max-content contribution is the
        // table's definite 500px width (css-sizing-3 §5.2), which still
        // covers the minimum; the minimum side keeps the repair.
        let (mut forest, table) = forest_table(
            "<table><tr><td class=\"a\">xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx</td></tr></table>",
            "table { table-layout: fixed; width: 500px; } \
             td { font-size: 10px; padding: 0; } td.a { width: 200px; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.fixed, "expected a fixed-layout grid");
        let col_spec = column_specified_widths(&mut forest, &grid, 0.0, 0.0);
        let (_, spacing_h, _) = table_border_model(&forest, table);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut mcw_scratch = Vec::new();
        let mut int_bufs = WalkBufs::new();
        let mut int_inner = WalkBufs::new();
        let (col_max, col_max_nonspanning) = auto_column_maxima(
            &mut forest,
            &grid,
            &col_spec,
            spacing_h,
            &FixedFontMetrics,
            &mut cache,
            &mut mcw_scratch,
            &mut int_bufs,
            &mut int_inner,
        );
        let mins = auto_column_minima(
            &mut forest,
            &grid,
            &col_spec,
            spacing_h,
            0.0,
            &FixedFontMetrics,
            &col_max,
            &col_max_nonspanning,
        );
        let mut pool = TableGridPool::new();
        let pref = table_pref_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut cache,
            &mut pool,
            true,
        );
        let min_total: f32 = mins.iter().sum();
        let nvis = grid.collapsed_cols.iter().filter(|&&c| !c).count();
        let spacing_total = spacing_h * (nvis as f32 + 1.0);
        // 40 chars at 10px type measure 240px; the minimum sees past the
        // 200px pin. The max-content contribution is the table's definite
        // 500px width (css-sizing-3 §5.2), which still covers the minimum.
        assert!(approx(mins[0], 240.0), "min-content {mins:?}");
        assert!(
            pref + 0.01 >= min_total + spacing_total,
            "max-content ({pref}) must cover min-content ({min_total} + {spacing_total} spacing)"
        );
        assert!(approx(pref, 500.0), "preferred {pref}");
    }

    #[test]
    fn pref_width_fixed_layout_pinned_col_wide_second_row_repaired() {
        // Regression (b): row 1 pins the column at its specified 200px;
        // a second-row cell with 240px of unbreakable content raises
        // only the minimum (the preferred loop skips pinned columns).
        // The max-content contribution is the table's definite 500px
        // width (css-sizing-3 §5.2), which still covers the repaired
        // minimum; the minimum side keeps the repair.
        let (mut forest, table) = forest_table(
            "<table><tr><td class=\"a\">x</td></tr><tr><td>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx</td></tr></table>",
            "table { table-layout: fixed; width: 500px; } \
             td { font-size: 10px; padding: 0; } td.a { width: 200px; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.fixed, "expected a fixed-layout grid");
        let col_spec = column_specified_widths(&mut forest, &grid, 0.0, 0.0);
        let (_, spacing_h, _) = table_border_model(&forest, table);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut mcw_scratch = Vec::new();
        let mut int_bufs = WalkBufs::new();
        let mut int_inner = WalkBufs::new();
        let (col_max, col_max_nonspanning) = auto_column_maxima(
            &mut forest,
            &grid,
            &col_spec,
            spacing_h,
            &FixedFontMetrics,
            &mut cache,
            &mut mcw_scratch,
            &mut int_bufs,
            &mut int_inner,
        );
        let mins = auto_column_minima(
            &mut forest,
            &grid,
            &col_spec,
            spacing_h,
            0.0,
            &FixedFontMetrics,
            &col_max,
            &col_max_nonspanning,
        );
        let mut pool = TableGridPool::new();
        let pref = table_pref_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut cache,
            &mut pool,
            true,
        );
        let min_total: f32 = mins.iter().sum();
        let nvis = grid.collapsed_cols.iter().filter(|&&c| !c).count();
        let spacing_total = spacing_h * (nvis as f32 + 1.0);
        // The second row measures its 240px of content into the minimum
        // even though the pin keeps the preferred loop out of it. The
        // max-content contribution is the table's definite 500px width
        // (css-sizing-3 §5.2), which still covers the minimum.
        assert!(approx(mins[0], 240.0), "min-content {mins:?}");
        assert!(
            pref + 0.01 >= min_total + spacing_total,
            "max-content ({pref}) must cover min-content ({min_total} + {spacing_total} spacing)"
        );
        assert!(approx(pref, 500.0), "preferred {pref}");
    }

    #[test]
    fn table_pref_width_min_content_width_uses_min_content_walk() {
        // Item 1: a `display: table` box with `width: min-content` inside
        // a shrink-to-fit float contributes its min-content size, not the
        // full max-content width (css-sizing-3 §5.2). At 10px type one
        // char measures 6px: "aaa bbb ccc" is 66px max-content but 18px
        // min-content ("aaa").
        let l = layout_table_html(
            "<div class=\"f\"><table class=\"t\"><tr><td>aaa bbb ccc</td></tr></table></div>",
            "body { margin: 0; } div.f { float: left; } \
             table.t { width: min-content; border-spacing: 0; } \
             td { font-size: 10px; padding: 0; }",
        );
        // body > div.f > table.
        let f = &l.children[0].children[0];
        let table = &f.children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        assert!(
            approx(f.rect.w, 18.0),
            "float should shrink to the min-content 18px, not max-content 66px: {}",
            f.rect.w
        );
        assert!(
            approx(table.rect.w, 18.0),
            "table used width should be the min-content size: {}",
            table.rect.w
        );
    }

    #[test]
    fn table_pref_width_definite_width_contributes_exactly() {
        // Item 1: `width: 200px` on a table inside a shrink-to-fit float
        // contributes exactly 200px, not the 234px max-content width
        // (css-sizing-3 §5.2).
        let l = layout_table_html(
            "<div class=\"f\"><table class=\"t\"><tr><td>aaa bbb ccc ddd eee fff ggg hhh iii jjj</td></tr></table></div>",
            "body { margin: 0; } div.f { float: left; } \
             table.t { width: 200px; border-spacing: 0; } \
             td { font-size: 10px; padding: 0; }",
        );
        // body > div.f > table.
        let f = &l.children[0].children[0];
        let table = &f.children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        assert!(
            approx(f.rect.w, 200.0),
            "float should shrink to the definite 200px: {}",
            f.rect.w
        );
        assert!(
            approx(table.rect.w, 200.0),
            "table used width should be the definite 200px: {}",
            table.rect.w
        );
    }

    #[test]
    fn table_pref_width_definite_width_in_inline_block_parent() {
        // Item 1 through an inline-block shrink-to-fit parent instead of
        // a float: the definite width still governs the contribution.
        let l = layout_table_html(
            "<div class=\"ib\"><table class=\"t\"><tr><td>aaa bbb ccc ddd eee fff ggg hhh iii jjj</td></tr></table></div>",
            "body { margin: 0; } div.ib { display: inline-block; } \
             table.t { width: 200px; border-spacing: 0; } \
             td { font-size: 10px; padding: 0; }",
        );
        // body > div.ib > table.
        let ib = &l.children[0].children[0];
        let table = &ib.children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        assert!(
            approx(ib.rect.w, 200.0),
            "inline-block should shrink to the definite 200px: {}",
            ib.rect.w
        );
        assert!(
            approx(table.rect.w, 200.0),
            "table used width should be the definite 200px: {}",
            table.rect.w
        );
    }

    #[test]
    fn table_pref_width_definite_width_skips_cell_intrinsic_walks() {
        // Finding 1 witness: a definite-width table's intrinsic
        // contribution is the definite width itself (`table_pref_adjusted`'s
        // definite arm discards the column-based intrinsic), so the
        // per-cell `preferred_width_mcw` walks must never run. A panicking
        // font spy proves the early return is taken: any cell intrinsic
        // measurement would advance text and panic.
        struct PanicFonts;
        impl crate::FontMetrics for PanicFonts {
            fn advance(&self, _text: &str, _font_size: f32, _family: crate::FontFamilyId) -> f32 {
                panic!("cell intrinsic walk ran for a definite-width table")
            }
            fn line_height(&self, _font_size: f32, _family: crate::FontFamilyId) -> f32 {
                panic!("cell intrinsic walk ran for a definite-width table")
            }
        }
        let (mut forest, table) = forest_table(
            "<table class=\"t\"><tr><td>aaa bbb ccc ddd eee fff ggg hhh iii jjj</td></tr></table>",
            "table.t { width: 200px; border-spacing: 0; } td { font-size: 10px; padding: 0; }",
        );
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut pool = TableGridPool::new();
        let pref = table_pref_width(&mut forest, table, &PanicFonts, &mut cache, &mut pool, true);
        assert!(
            approx(pref, 200.0),
            "definite width should contribute exactly without measuring cells: {pref}"
        );
    }

    #[test]
    fn table_pref_width_fit_content_clamps_at_used_time() {
        // Item 1: `width: fit-content` keeps the max-content size as its
        // intrinsic contribution; the fit-content clamp
        // (min(max-content, max(min-content, available))) applies at
        // used-value time, where the base is definite. The float's
        // `max-width` narrows the available width below max-content
        // (234px), so the table sizes to the fit-content size (100px),
        // not full max-content.
        let html = "<div class=\"f\"><table class=\"t\"><tr><td>aaa bbb ccc ddd eee fff ggg hhh iii jjj</td></tr></table></div>";
        let css = "body { margin: 0; } div.f { float: left; max-width: 100px; } \
                   table.t { width: fit-content; border-spacing: 0; } \
                   td { font-size: 10px; padding: 0; }";
        // The intrinsic contribution stays max-content (39 chars at 6px
        // = 234px); only the used width clamps.
        let (mut forest, table) = forest_table(html, css);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut pool = TableGridPool::new();
        let pref = table_pref_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut cache,
            &mut pool,
            true,
        );
        assert!(
            approx(pref, 234.0),
            "fit-content intrinsic contribution should stay max-content 234px: {pref}"
        );
        let l = layout_table_html(html, css);
        // body > div.f > table.
        let f = &l.children[0].children[0];
        let table_box = &f.children[0];
        assert_eq!(table_box.box_type, crate::BoxType::Table);
        assert!(
            approx(f.rect.w, 100.0),
            "float should clamp to max-width 100px: {}",
            f.rect.w
        );
        assert!(
            approx(table_box.rect.w, 100.0),
            "table used width should be the fit-content size 100px: {}",
            table_box.rect.w
        );
    }

    #[test]
    fn fixed_layout_pct_pins_over_100_intrinsic_rescaled_not_inflated() {
        // Item 2: fixed table, definite width, first-row percentage pins
        // summing >100% — measured through the intrinsic path (as a
        // shrink-to-fit parent would), the contribution is the rescaled
        // definite width (CSS Tables 3 §3.9.3.1: the used pass rescales
        // over-constrained percentage pins against the assignable width
        // instead of growing the table), not the inflated second-fixpoint
        // value that re-pins first-row percentage cells against the
        // growing table width.
        let html = "<table><tr><td class=\"a\">x</td><td class=\"b\">y</td></tr></table>";
        let css = "table { table-layout: fixed; width: 500px; } \
                   td { font-size: 10px; padding: 0; } \
                   td.a { width: 60%; } td.b { width: 60%; }";
        let (mut forest, table) = forest_table(html, css);
        let grid = collect_table_grid(&forest, table);
        assert!(grid.fixed, "expected a fixed-layout grid");
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut pool = TableGridPool::new();
        let pref = table_pref_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut cache,
            &mut pool,
            true,
        );
        assert!(
            approx(pref, 500.0),
            "intrinsic contribution should be the rescaled definite 500px, not inflated: {pref}"
        );
        // End to end through a float: the table keeps its definite width
        // with the 60/60 pins rescaled to 250/250 — no growth.
        let l = layout_table_html(
            "<div class=\"f\"><table class=\"t\"><tr><td class=\"a\">x</td><td class=\"b\">y</td></tr></table></div>",
            "body { margin: 0; } div.f { float: left; } \
             table.t { table-layout: fixed; width: 500px; border-spacing: 0; } \
             td { font-size: 10px; padding: 0; } \
             td.a { width: 60%; } td.b { width: 60%; }",
        );
        // body > div.f > table > row-group > row > cells.
        let f = &l.children[0].children[0];
        let table_box = &f.children[0];
        assert_eq!(table_box.box_type, crate::BoxType::Table);
        assert!(
            approx(f.rect.w, 500.0),
            "float should shrink to the rescaled 500px: {}",
            f.rect.w
        );
        assert!(
            approx(table_box.rect.w, 500.0),
            "table should keep its definite 500px: {}",
            table_box.rect.w
        );
        let group = table_box
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::TableRowGroup)
            .expect("row group");
        let (c0, c1) = (
            &group.children[0].children[0],
            &group.children[0].children[1],
        );
        assert!(
            approx(c0.rect.w, 250.0),
            "first 60% pin should rescale to 250px: {}",
            c0.rect.w
        );
        assert!(
            approx(c1.rect.w, 250.0),
            "second 60% pin should rescale to 250px: {}",
            c1.rect.w
        );
    }

    #[test]
    fn auto_layout_over_constrained_recenter_auto_margins() {
        // Min-content (180) exceeds the specified width (100) with `auto`
        // side margins: the table grows symmetrically and stays centered
        // instead of drifting toward the inline end.
        let l = layout_table_html(
            "<table><tr><td>aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</td></tr></table>",
            "table { width: 100px; margin-left: auto; margin-right: auto; } td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 180.0));
        assert!(approx(table.rect.x, 310.0));
        assert!(approx(cell_at(&l, 0, 0).rect.x, 310.0));
    }

    #[test]
    fn fixed_layout_over_constrained_recenter_auto_margins() {
        // `table-layout: fixed` twin of the auto-layout recenter test:
        // the pinned column (180) exceeds the specified width (100) with
        // `auto` side margins, so the table grows symmetrically and stays
        // centered (800 viewport: margins 350 each, growth 80, recentered
        // x = 350 − 40 = 310).
        let l = layout_table_html(
            "<table><col class=\"a\"><tr><td>x</td></tr></table>",
            "table { width: 100px; table-layout: fixed; margin-left: auto; margin-right: auto; } \
             col.a { width: 180px; } td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 180.0));
        assert!(approx(table.rect.x, 310.0));
        assert!(approx(cell_at(&l, 0, 0).rect.x, 310.0));
    }

    #[test]
    fn auto_layout_over_constrained_growth_respects_max_width() {
        // Over-constrained growth stops at the table's own `max-width`:
        // min-content (180) wants +80 past the specified 100, but only
        // +50 fits under `max-width: 150px`; the columns overflow the
        // table instead of pushing it past its cap.
        let l = layout_table_html(
            "<table><tr><td>aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</td></tr></table>",
            "table { width: 100px; max-width: 150px; } td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 150.0));
        assert!(approx(cell_at(&l, 0, 0).rect.w, 180.0));
    }

    #[test]
    fn fixed_layout_over_constrained_growth_respects_max_width() {
        // Fixed-layout twin: the pinned column (180) wants +80 past the
        // specified 100, but `max-width: 150px` caps the growth.
        let l = layout_table_html(
            "<table><col class=\"a\"><tr><td>x</td></tr></table>",
            "table { width: 100px; max-width: 150px; table-layout: fixed; } \
             col.a { width: 180px; } td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 150.0));
        assert!(approx(cell_at(&l, 0, 0).rect.w, 180.0));
    }

    #[test]
    fn row_specified_height_is_minimum() {
        // CSS 2.1 §17.5.3: a specified `height` on a row is a minimum for
        // that row — the 100px row stays 100px even though its content is
        // one short line, while the unspecified row sizes from content.
        let l = layout_table_html(
            "<table><tr class=\"tall\"><td>x</td></tr><tr><td>y</td></tr></table>",
            "table { border-collapse: collapse; } tr.tall { height: 100px; } td { font-size: 10px; }",
        );
        let tall = row_at(&l, 0).rect.h;
        let short = row_at(&l, 1).rect.h;
        assert!(approx(tall, 100.0));
        assert!(short < 100.0);
        assert!(approx(cell_at(&l, 0, 0).rect.h, 100.0));
    }

    #[test]
    fn row_group_specified_height_spreads_over_rows() {
        // CSS 2.1 §17.5.3: a specified `height` on a row group is a
        // minimum for the sum of its rows' heights — the 150px group
        // spreads evenly over its two short rows.
        let l = layout_table_html(
            "<table><tbody class=\"tall\"><tr><td>x</td></tr><tr><td>y</td></tr></tbody></table>",
            "table { border-collapse: collapse; } tbody.tall { height: 150px; } td { font-size: 10px; }",
        );
        assert!(approx(row_at(&l, 0).rect.h, 75.0));
        assert!(approx(row_at(&l, 1).rect.h, 75.0));
    }

    // ---- w50: row-height distribution (CSS 2.1 §17.5.3 / css-tables-3) ----

    const NO_PAD: &str = "table { border-collapse: collapse; } td { padding: 0; }";

    #[test]
    fn rowspan_excess_distributes_proportionally_to_unconstrained_rows() {
        // WPT `rowspan-height-redistribution` table 5: the 100px spanning
        // cell's excess (100 - 45 - 15 = 40) splits proportionally to
        // the unconstrained rows' heights: 45+30=75, 15+10=25.
        let l = layout_table_html(
            "<table><tbody>\
             <tr><td><div style=\"height:45px\"></div></td>\
             <td rowspan=\"2\"><div style=\"height:100px\"></div></td></tr>\
             <tr><td><div style=\"height:15px\"></div></td></tr>\
             </tbody></table>",
            NO_PAD,
        );
        assert!(
            approx(row_at(&l, 0).rect.h, 75.0),
            "r0={}",
            row_at(&l, 0).rect.h
        );
        assert!(
            approx(row_at(&l, 1).rect.h, 25.0),
            "r1={}",
            row_at(&l, 1).rect.h
        );
    }

    #[test]
    fn rowspan_excess_skips_zero_height_rows() {
        // WPT table 3: the empty middle row takes none of the excess.
        let l = layout_table_html(
            "<table><tbody>\
             <tr><td><div style=\"height:20px\"></div></td>\
             <td rowspan=\"3\"><div style=\"height:100px\"></div></td></tr>\
             <tr></tr>\
             <tr><td><div style=\"height:20px\"></div></td></tr>\
             </tbody></table>",
            NO_PAD,
        );
        assert!(approx(row_at(&l, 0).rect.h, 50.0));
        assert!(approx(row_at(&l, 1).rect.h, 0.0));
        assert!(approx(row_at(&l, 2).rect.h, 50.0));
    }

    #[test]
    fn rowspan_excess_skips_constrained_row_when_unconstrained_exists() {
        // WPT table 6: the 30px fixed row keeps its height; the whole
        // excess lands on the unconstrained row.
        let l = layout_table_html(
            "<table><tbody>\
             <tr style=\"height:30px\"><td>0,0</td>\
             <td rowspan=\"2\"><div style=\"height:100px\"></div></td></tr>\
             <tr><td><div style=\"height:10px\"></div></td></tr>\
             </tbody></table>",
            NO_PAD,
        );
        assert!(
            approx(row_at(&l, 0).rect.h, 30.0),
            "r0={}",
            row_at(&l, 0).rect.h
        );
        assert!(
            approx(row_at(&l, 1).rect.h, 70.0),
            "r1={}",
            row_at(&l, 1).rect.h
        );
    }

    #[test]
    fn rowspan_excess_grows_constrained_rows_proportionally_when_no_unconstrained() {
        // WPT table 7: with every row constrained, the 20px deficit
        // (100 - 80) splits 20:20:40 -> 25/25/50.
        let l = layout_table_html(
            "<table><tbody>\
             <tr style=\"height:20px\"><td>a</td>\
             <td rowspan=\"3\"><div style=\"height:100px\"></div></td></tr>\
             <tr style=\"height:20px\"><td>b</td></tr>\
             <tr style=\"height:40px\"><td>c</td></tr>\
             </tbody></table>",
            NO_PAD,
        );
        assert!(approx(row_at(&l, 0).rect.h, 25.0));
        assert!(approx(row_at(&l, 1).rect.h, 25.0));
        assert!(approx(row_at(&l, 2).rect.h, 50.0));
    }

    #[test]
    fn rowspan_excess_over_empty_rows_goes_to_last_spanned_row() {
        // WPT table 16: spanning rows that are all empty give the whole
        // deficit to the last spanned row.
        let l = layout_table_html(
            "<table><tbody>\
             <tr><td rowspan=\"3\"><div style=\"height:100px\"></div></td></tr>\
             <tr></tr><tr></tr>\
             </tbody></table>",
            NO_PAD,
        );
        assert!(approx(row_at(&l, 0).rect.h, 0.0));
        assert!(approx(row_at(&l, 1).rect.h, 0.0));
        assert!(approx(row_at(&l, 2).rect.h, 100.0));
    }

    #[test]
    fn rowspan_enclosed_cell_distributes_first() {
        // WPT table 13: the inner 100px cell (rows 2-3) distributes
        // before the enclosing 50px cell (rows 1-4); its deficit lands
        // on the last of its spanned rows.
        let l = layout_table_html(
            "<table><tbody>\
             <tr><td rowspan=\"4\"><div style=\"height:50px\"></div></td><td></td></tr>\
             <tr><td></td><td rowspan=\"2\"><div style=\"height:100px\"></div></td></tr>\
             <tr></tr><tr></tr>\
             </tbody></table>",
            NO_PAD,
        );
        assert!(approx(row_at(&l, 0).rect.h, 0.0));
        assert!(approx(row_at(&l, 1).rect.h, 0.0));
        assert!(
            approx(row_at(&l, 2).rect.h, 100.0),
            "r2={}",
            row_at(&l, 2).rect.h
        );
        assert!(approx(row_at(&l, 3).rect.h, 0.0));
    }

    #[test]
    fn rowspan_excess_goes_to_row_originating_other_spanning_cell() {
        // WPT table 19: the 100px cell's deficit goes to the row that
        // originates the other (empty) spanning cell.
        let l = layout_table_html(
            "<table><tbody>\
             <tr><td rowspan=\"5\"><div style=\"height:100px\"></div></td><td></td></tr>\
             <tr><td rowspan=\"5\"></td><td></td></tr>\
             <tr></tr><tr></tr><tr></tr>\
             </tbody></table>",
            NO_PAD,
        );
        assert!(approx(row_at(&l, 0).rect.h, 0.0));
        assert!(
            approx(row_at(&l, 1).rect.h, 100.0),
            "r1={}",
            row_at(&l, 1).rect.h
        );
        assert!(approx(row_at(&l, 2).rect.h, 0.0));
        assert!(approx(row_at(&l, 3).rect.h, 0.0));
        assert!(approx(row_at(&l, 4).rect.h, 0.0));
    }

    #[test]
    fn rowspan_first_cell_wins_overlapping_spans() {
        // WPT table 14: the higher 50px cell distributes first (its
        // deficit lands on the last of its spanned rows), then the
        // 100px cell sees the now-nonempty row 4 and grows it.
        let l = layout_table_html(
            "<table><tbody>\
             <tr><td rowspan=\"4\"><div style=\"height:50px\"></div></td><td></td></tr>\
             <tr></tr><tr></tr>\
             <tr><td></td><td rowspan=\"3\"><div style=\"height:100px\"></div></td></tr>\
             <tr></tr><tr></tr>\
             </tbody></table>",
            NO_PAD,
        );
        for r in 0..6 {
            let h = row_at(&l, r).rect.h;
            if r == 3 {
                assert!(approx(h, 100.0), "r3={h}");
            } else {
                assert!(approx(h, 0.0), "r{r}={h}");
            }
        }
    }

    #[test]
    fn rowspan_baseline_does_not_grow_originating_row() {
        // A tall spanning cell must not dump its height into its
        // originating row through baseline alignment (WPT
        // `rowspan-height-redistribution` table 2): the rows stay 50/50.
        let l = layout_table_html(
            "<table><tbody>\
             <tr><td>0,0</td><td rowspan=\"2\"><div style=\"height:100px\"></div></td></tr>\
             <tr><td>0,1</td></tr>\
             </tbody></table>",
            NO_PAD,
        );
        assert!(
            approx(row_at(&l, 0).rect.h, 50.0),
            "r0={}",
            row_at(&l, 0).rect.h
        );
        assert!(
            approx(row_at(&l, 1).rect.h, 50.0),
            "r1={}",
            row_at(&l, 1).rect.h
        );
    }

    #[test]
    fn group_height_distributes_proportionally_to_unsized_rows() {
        // WPT `tbody-height-redistribution` table 3: the group's excess
        // (100 - 40) splits proportionally to the rows' heights.
        let l = layout_table_html(
            "<table><tbody style=\"height:100px\">\
             <tr><td><div style=\"height:10px\"></div></td></tr>\
             <tr><td><div style=\"height:30px\"></div></td></tr>\
             </tbody></table>",
            NO_PAD,
        );
        assert!(approx(row_at(&l, 0).rect.h, 25.0));
        assert!(approx(row_at(&l, 1).rect.h, 75.0));
    }

    #[test]
    fn group_height_does_not_grow_fixed_row_when_unsized_exists() {
        // WPT `tbody-height-redistribution` table 4: the fixed 10px row
        // keeps its height; the unsized row absorbs the whole excess.
        let l = layout_table_html(
            "<table><tbody style=\"height:100px\">\
             <tr style=\"height:10px\"><td></td></tr>\
             <tr><td><div style=\"height:20px\"></div></td></tr>\
             </tbody></table>",
            NO_PAD,
        );
        assert!(approx(row_at(&l, 0).rect.h, 10.0));
        assert!(approx(row_at(&l, 1).rect.h, 90.0));
    }

    #[test]
    fn group_height_pct_rows_resolve_against_group_height() {
        // WPT `tbody-height-redistribution` table 5: percentage rows
        // resolve against the row group's own height; the remaining row
        // absorbs the rest.
        let l = layout_table_html(
            "<table><tbody style=\"height:100px\">\
             <tr style=\"height:25%\"><td>x</td></tr>\
             <tr style=\"height:50%\"><td>x</td></tr>\
             <tr><td><div style=\"height:10px\"></div></td></tr>\
             </tbody></table>",
            NO_PAD,
        );
        assert!(approx(row_at(&l, 0).rect.h, 25.0));
        assert!(approx(row_at(&l, 1).rect.h, 50.0));
        assert!(approx(row_at(&l, 2).rect.h, 25.0));
    }

    #[test]
    fn group_height_empty_rows_share_evenly_sized_row_excluded() {
        // WPT `tbody-height-redistribution` table 12: the 0px row keeps
        // 0; the two unsized empty rows split the 50px group evenly.
        let l = layout_table_html(
            "<table style=\"width:50px\"><tbody style=\"height:50px\">\
             <tr style=\"height:0px\"><td></td></tr>\
             <tr><td></td></tr><tr><td></td></tr>\
             </tbody></table>",
            NO_PAD,
        );
        assert!(approx(row_at(&l, 0).rect.h, 0.0));
        assert!(approx(row_at(&l, 1).rect.h, 25.0));
        assert!(approx(row_at(&l, 2).rect.h, 25.0));
    }

    #[test]
    fn empty_group_keeps_specified_height() {
        // WPT `tbody-height-redistribution` table 10: a rowless group
        // with a specified height reports that height (and the table
        // grows to it).
        let l = layout_table_html(
            "<table><tbody style=\"height:75px\"></tbody></table>",
            NO_PAD,
        );
        assert!(
            approx(group_at(&l, 0).rect.h, 75.0),
            "tbody={}",
            group_at(&l, 0).rect.h
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.h, 75.0), "table={}", table.rect.h);
    }

    #[test]
    fn rowless_group_after_rowspan_overflow_does_not_panic() {
        // B1: a `rowspan` overflowing its row group creates phantom
        // grid rows past `row_group`'s length (`nrows` counts them,
        // `row_group` does not). A rowless group *after* that group
        // makes `rowless_insert_at` scan every grid row for a later
        // group; indexing `row_group[r]` directly panicked on the
        // first phantom row. The rowless group's 75px lands after the
        // last (phantom) row instead.
        let l = layout_table_html(
            "<table><tbody><tr><td rowspan=\"3\">x</td></tr></tbody>\
             <tbody style=\"height:75px\"></tbody></table>",
            NO_PAD,
        );
        assert!(
            approx(group_at(&l, 1).rect.h, 75.0),
            "rowless tbody={}",
            group_at(&l, 1).rect.h
        );
        let table = &l.children[0].children[0];
        assert!(table.rect.h >= 75.0, "table={}", table.rect.h);
    }

    #[test]
    fn table_height_fully_collapsed_group_does_not_absorb_excess() {
        // B4: a row group whose rows are all `visibility: collapse`
        // has rows, but no *visible* rows. It is not rowless, so it
        // must not absorb table excess into its own rect (a void: its
        // rows are zeroed and never placed) — the visible unsized
        // group takes the whole 100px excess. Before the fix the
        // rowless branch tested "no visible rows", both groups counted
        // as absorbers, and the even split left the visible group at
        // 50px with the other 50px lost.
        let l = layout_table_html(
            "<table style=\"height:100px\">\
             <tbody class=\"gone\"><tr><td></td></tr></tbody>\
             <tbody><tr><td></td></tr></tbody></table>",
            "table { border-collapse: collapse; } td { padding: 0; } \
             tbody.gone { visibility: collapse; }",
        );
        assert!(
            approx(group_at(&l, 0).rect.h, 0.0),
            "collapsed group={}",
            group_at(&l, 0).rect.h
        );
        assert!(
            approx(group_at(&l, 1).rect.h, 100.0),
            "visible group takes all excess: {}",
            group_at(&l, 1).rect.h
        );
        assert!(approx(grouped_row_at(&l, 1, 0).rect.h, 100.0));
        // A truly rowless group (no rows at all) still absorbs into
        // its own rect: split evenly with the visible group here.
        let l = layout_table_html(
            "<table style=\"height:100px\">\
             <tbody></tbody>\
             <tbody><tr><td></td></tr></tbody></table>",
            NO_PAD,
        );
        assert!(
            approx(group_at(&l, 0).rect.h, 50.0),
            "rowless group absorbs its share: {}",
            group_at(&l, 0).rect.h
        );
        assert!(approx(group_at(&l, 1).rect.h, 50.0));
    }

    #[test]
    fn table_height_sized_group_keeps_height_unsized_absorbs() {
        // WPT `table-height-redistribution` table 18: the thead's row is
        // sized (20px cell); the unsized tbody absorbs the whole table
        // excess.
        let l = layout_table_html(
            "<table style=\"height:100px;width:100px\">\
             <thead><tr><td style=\"height:20px\">thead</td></tr></thead>\
             <tbody><tr></tr></tbody></table>",
            "table { border-spacing: 0px; } td { padding: 0; }",
        );
        assert!(
            approx(group_at(&l, 0).rect.h, 20.0),
            "thead={}",
            group_at(&l, 0).rect.h
        );
        assert!(
            approx(group_at(&l, 1).rect.h, 80.0),
            "tbody={}",
            group_at(&l, 1).rect.h
        );
        assert!(approx(grouped_row_at(&l, 1, 0).rect.h, 80.0));
    }

    #[test]
    fn table_height_two_bodies_share_excess() {
        // WPT `table-height-redistribution` table 25: two unsized bodies
        // split the table excess evenly.
        let l = layout_table_html(
            "<table style=\"height:100px\"><tbody><tr></tr></tbody><tbody><tr></tr></tbody></table>",
            NO_PAD,
        );
        assert!(approx(group_at(&l, 0).rect.h, 50.0));
        assert!(approx(group_at(&l, 1).rect.h, 50.0));
    }

    #[test]
    fn table_height_specified_body_keeps_height_other_absorbs() {
        // WPT `table-height-redistribution` table 26: the 40%-specified
        // body keeps its 40px; the unsized body absorbs the remaining
        // 60px.
        let l = layout_table_html(
            "<table style=\"height:100px\"><tbody style=\"height:40%\"><tr></tr></tbody>\
             <tbody><tr></tr></tbody></table>",
            NO_PAD,
        );
        assert!(approx(group_at(&l, 0).rect.h, 40.0));
        assert!(approx(group_at(&l, 1).rect.h, 60.0));
    }

    #[test]
    fn table_height_is_minimum_grows_for_tall_row() {
        // WPT `table-height-redistribution` table 12: a specified table
        // height is a minimum — the 200px row grows the table to 200px
        // instead of the table clamping the row to 100px. (Separated
        // border model, like that test: in the collapsing model the
        // rows' border halves already count the table's own borders.)
        let l = layout_table_html(
            "<table style=\"height:100px\"><tbody><tr style=\"height:200px\"></tr></tbody></table>",
            "table { border-spacing: 0; } td { padding: 0; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.h, 200.0), "table={}", table.rect.h);
        assert!(approx(group_at(&l, 0).rect.h, 200.0));
        assert!(approx(row_at(&l, 0).rect.h, 200.0));
    }

    #[test]
    fn table_height_pct_group_capped_at_table_height() {
        // WPT `table-height-redistribution` table 13: a 200% group in a
        // 100px table resolves to 100px — percentages of the table's own
        // height cannot push the table taller than itself.
        let l = layout_table_html(
            "<table style=\"height:100px\"><tbody style=\"height:200%\"><tr></tr></tbody></table>",
            NO_PAD,
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.h, 100.0), "table={}", table.rect.h);
        assert!(approx(group_at(&l, 0).rect.h, 100.0));
    }

    #[test]
    fn table_height_pct_row_capped_at_table_height() {
        // WPT `table-height-redistribution` table 14: a 200% row in a
        // 100px table resolves to 100px.
        let l = layout_table_html(
            "<table style=\"height:100px\"><tbody><tr style=\"height:200%\"></tr></tbody></table>",
            NO_PAD,
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.h, 100.0), "table={}", table.rect.h);
        assert!(approx(row_at(&l, 0).rect.h, 100.0));
    }

    #[test]
    fn cell_pct_height_child_resolves_against_used_height() {
        // WPT `td-box-sizing-003` table 10 (css-tables-3 §3.10.6): the
        // inline-block's `height: 100%` is indefinite during the cell's
        // measurement pass — resolving it eagerly made the line box
        // 100px above the baseline plus the strut descent below it, so
        // the cell grew to 110. In the final pass it resolves against
        // the cell's used height (100) and the cell stays 100.
        let l = layout_table_html(
            "<table><tr><td style=\"height:100px;font-size:50px;line-height:1\">y<div style=\"display:inline-block;height:100%;width:50px\"></div></td></tr></table>",
            "table { border-spacing: 0; } td { padding: 0; }",
        );
        let td = cell_at(&l, 0, 0);
        assert!(approx(td.rect.h, 100.0), "td={}", td.rect.h);
        let div = find_w(&l, 50.0);
        assert!(approx(div.rect.h, 100.0), "div={}", div.rect.h);
    }

    #[test]
    fn pct_pass2_baseline_recomputed_for_placement() {
        // B7 (css-tables-3 §3.10.6): the first cell triggers the
        // percentage second pass — its inline-block is 0 tall during
        // measurement (percentage indefinite), so the cell's baseline
        // is its 50px text line's, 40px down; in pass 2 the
        // inline-block resolves to the cell's 100px used height and
        // the cell's baseline becomes 100px. The placement resume used
        // to reuse the measurement-pass `bshift`, dropping the small
        // cell's content by only 40 - 9.6 = 30.4px; with the baseline
        // recomputed from the pass-2 measurements it drops by
        // 100 - 9.6 = 90.4px and the two cells' baselines coincide.
        let l = layout_table_html(
            "<table><tr>\
             <td style=\"height:100px;font-size:50px;line-height:1\">y<div style=\"display:inline-block;height:100%;width:50px\"></div></td>\
             <td class=\"small\">x</td>\
             </tr></table>",
            "table { border-spacing: 0; } td { padding: 0; vertical-align: baseline; } \
             .small { font-size: 10px; }",
        );
        let a = cell_at(&l, 0, 0);
        let b = cell_at(&l, 0, 1);
        assert!(approx(a.rect.h, 100.0), "a={}", a.rect.h);
        assert!(approx(b.rect.h, 100.0), "b={}", b.rect.h);
        let inner_b = &b.children[0];
        assert!(
            approx(inner_b.rect.y - b.rect.y, 90.4),
            "small cell content must drop by the pass-2 baseline difference: {}",
            inner_b.rect.y - b.rect.y
        );
        // The placed baselines coincide: the inline-block's baseline
        // (its bottom, at the pass-2 height of 100px) meets the small
        // cell's text baseline (content top + its 9.6px ascent).
        let div = find_w(&l, 50.0);
        assert!(approx(div.rect.h, 100.0), "div={}", div.rect.h);
        assert!(
            approx(div.rect.y + div.rect.h, inner_b.rect.y + 9.6),
            "baselines must coincide: div bottom={} text baseline={}",
            div.rect.y + div.rect.h,
            inner_b.rect.y + 9.6
        );
    }

    #[test]
    fn row_height_alone_does_not_resolve_cell_pct_children() {
        // WPT `td-box-sizing-003` table 11: the row's 100px height does
        // not make the (auto-height) cell a percentage base — the
        // inline-block stays auto (0) while the cell takes the row
        // height. Only the cell's own specified height triggers the
        // §3.10.6 second pass.
        let l = layout_table_html(
            "<table><tr style=\"height:100px\"><td style=\"font-size:50px;line-height:1\">y<div style=\"display:inline-block;height:100%;width:50px\"></div></td></tr></table>",
            "table { border-spacing: 0; } td { padding: 0; }",
        );
        let td = cell_at(&l, 0, 0);
        assert!(approx(td.rect.h, 100.0), "td={}", td.rect.h);
        let div = find_w(&l, 50.0);
        assert!(approx(div.rect.h, 0.0), "div={}", div.rect.h);
    }

    #[test]
    fn cell_fixed_height_child_still_grows_cell() {
        // WPT `table-cell-overflow-explicit-height-001`: a 300px
        // fixed-height child grows a 20px-specified cell (a fixed
        // height is not a percentage path; the §3.10.6 passes must not
        // clamp it).
        let l = layout_table_html(
            "<table><tr><td style=\"height:20px\"><div style=\"height:300px;width:60px\"></div></td></tr></table>",
            "table { border-spacing: 0; } td { padding: 0; }",
        );
        let td = cell_at(&l, 0, 0);
        assert!(approx(td.rect.h, 300.0), "td={}", td.rect.h);
        let div = find_w(&l, 60.0);
        assert!(approx(div.rect.h, 300.0), "div={}", div.rect.h);
    }

    #[test]
    fn separate_borders_spacing_starved_grows_table() {
        // Degenerate: border-spacing (2 × 60px) alone exceeds the table's
        // 100px content width. The table grows to fit spacing + columns
        // so the exact-tiling invariant holds, instead of overflowing
        // the columns at their minima.
        let l = layout_table_html(
            "<table><tr><td>x</td></tr></table>",
            "table { width: 100px; border-spacing: 60px; } td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        let cell = cell_at(&l, 0, 0);
        assert!(table.rect.w > 100.0, "table={}", table.rect.w);
        // The grown width tiles exactly: spacing (120) + the column.
        assert!(
            approx(table.rect.w, 120.0 + cell.rect.w),
            "table={} cell={}",
            table.rect.w,
            cell.rect.w
        );
    }

    #[test]
    fn cell_specified_height_is_minimum() {
        // CSS 2.1 §17.5.3: a specified `height` on a cell is a minimum —
        // content taller than the specified height grows the cell (and
        // its row) instead of overflowing it (WPT
        // `table-cell-overflow-explicit-height-001`).
        let l = layout_table_html(
            "<table><tr><td><div class=\"tall\"></div></td></tr></table>",
            "td { height: 20px; } .tall { height: 300px; }",
        );
        let cell_h = cell_at(&l, 0, 0).rect.h;
        assert!(cell_h >= 300.0, "cell.h={cell_h}");
        assert!(approx(row_at(&l, 0).rect.h, cell_h));
    }

    #[test]
    fn cell_specified_height_minimum_holds_for_short_content() {
        // The minimum still applies: short content in a 100px cell
        // leaves the cell (and its row) 100px tall.
        let l = layout_table_html(
            "<table><tr><td>x</td></tr></table>",
            "td { height: 100px; font-size: 10px; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.h, 100.0));
        assert!(approx(row_at(&l, 0).rect.h, 100.0));
    }

    #[test]
    fn quirks_cell_height_uses_border_box_only_in_full_quirks() {
        // WHATWG Quirks §3.13: in full quirks mode a cell's `height`
        // behaves as `border-box` (content-box 100px + 40px padding
        // = 100px border box), while no-quirks and limited-quirks
        // keep content-box (140px). Mirrors WPT
        // `table-quirks.html` table 1 and `table-limited-quirks.html`
        // table 1 (there the table adds 16px of border-spacing).
        let css = "table { border-spacing: 8px; } td { height: 100px; box-sizing: content-box; padding: 20px; font-size: 10px; }";
        let quirks = layout_table_html(
            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\"><table><tr><td>x</td></tr></table>",
            css,
        );
        assert!(approx(cell_at(&quirks, 0, 0).rect.h, 100.0));
        let standards = layout_table_html("<!DOCTYPE html><table><tr><td>x</td></tr></table>", css);
        assert!(approx(cell_at(&standards, 0, 0).rect.h, 140.0));
        let limited = layout_table_html(
            "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\"><table><tr><td>x</td></tr></table>",
            css,
        );
        assert!(approx(cell_at(&limited, 0, 0).rect.h, 140.0));
    }

    #[test]
    fn quirks_cell_width_sums_consecutive_replaced_only_in_full_quirks() {
        // WHATWG Quirks §3.8: in full quirks mode, consecutive
        // inline replaced elements in a table cell have no soft wrap
        // opportunity between them, so the cell's min-content is the
        // summed 5x50px = 250px plus 40px padding = 290px, and the
        // table grows past its specified 200px width. In
        // limited-quirks mode the quirk does not apply: the cell is
        // constrained to the 200px table minus 16px spacing = 184px.
        // Mirrors WPT `table-quirks.html` table 6 and
        // `table-limited-quirks.html` table 6.
        let css = "table { width: 200px; border-spacing: 8px; } td { padding: 20px; } img { width: 50px; height: 20px; }";
        let imgs = "<img><img><img><img><img>";
        let quirks = layout_table_html(
            &format!(
                "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\"><table><tr><td>{imgs}</td></tr></table>"
            ),
            css,
        );
        assert!(approx(cell_at(&quirks, 0, 0).rect.w, 290.0));
        let limited = layout_table_html(
            &format!(
                "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\"><table><tr><td>{imgs}</td></tr></table>"
            ),
            css,
        );
        assert!(approx(cell_at(&limited, 0, 0).rect.w, 184.0));
    }

    #[test]
    fn w77fix_quirks_cell_width_run_rules_floated_nested_nonimg() {
        // WHATWG Quirks §3.8, w77fix corrections (fail-before
        // probes; all in full quirks mode, table width 50px so
        // the table grows to the column minimum — the existing
        // §3.8 test's pattern — 8px border-spacing, 20px cell
        // padding, 50px replaced widths):
        // (a) a floated `img` breaks the run instead of joining
        // it: float 50 + two in-flow 50s give min-content
        // max(50, 100) + 40 = 140, not the joined 190.
        // (b) an inline container is transparent: text inside a
        // span still breaks the run, so `<span><img>hello<img>
        // </span>` (50 + 30-word + 50 at 10px) contributes the
        // largest single run inside the span (50), not the span's
        // summed 130 — cell 90, not 170. And a run continues
        // across a span boundary: `<img><span><img></span><img>`
        // sums all three (190), since all three are consecutive
        // `img`s in one formatting context.
        // (c) only `img` joins a run: two `object`s (50 each)
        // contribute max(50, 50) + 40 = 90, not the summed 140;
        // an `img` followed by an `object` likewise gives 90.
        // Controls: three in-flow `img`s still sum to 190, and
        // two plain inline-blocks (never replaced) still give 90.
        let css = "table { width: 50px; border-spacing: 8px; } td { padding: 20px; font-size: 10px; } img, object { width: 50px; height: 20px; } .ib { display: inline-block; width: 50px; height: 20px; }";
        let quirks = |inner: &str| {
            layout_table_html(
                &format!(
                    "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\"><table><tr><td>{inner}</td></tr></table>"
                ),
                css,
            )
        };
        let cell_w = |inner: &str| cell_at(&quirks(inner), 0, 0).rect.w;
        // (a) floated img breaks the run.
        assert!(
            approx(cell_w("<img style=\"float:left\"><img><img>"), 140.0),
            "floated img cell w {}, want 140 (max(50, 100) + 40)",
            cell_w("<img style=\"float:left\"><img><img>")
        );
        // (b) nested span: largest single run, not the sum.
        assert!(
            approx(cell_w("<span><img>hello<img></span>"), 90.0),
            "nested span cell w {}, want 90 (50 + 40)",
            cell_w("<span><img>hello<img></span>")
        );
        // (b) run continues across a span boundary.
        assert!(
            approx(cell_w("<img><span><img></span><img>"), 190.0),
            "cross-span run cell w {}, want 190 (150 + 40)",
            cell_w("<img><span><img></span><img>")
        );
        // (c) non-img replaced elements break runs.
        assert!(
            approx(cell_w("<object></object><object></object>"), 90.0),
            "object run cell w {}, want 90 (50 + 40)",
            cell_w("<object></object><object></object>")
        );
        assert!(
            approx(cell_w("<img><object></object>"), 90.0),
            "img+object cell w {}, want 90 (50 + 40)",
            cell_w("<img><object></object>")
        );
        // Controls (unchanged behavior).
        assert!(
            approx(cell_w("<img><img><img>"), 190.0),
            "img run cell w {}, want 190",
            cell_w("<img><img><img>")
        );
        assert!(
            approx(
                cell_w("<div class=\"ib\"></div><div class=\"ib\"></div>"),
                90.0
            ),
            "inline-block cell w {}, want 90",
            cell_w("<div class=\"ib\"></div><div class=\"ib\"></div>")
        );
    }

    #[test]
    fn w77fix_quirks_vertical_cell_width_sums_consecutive_imgs() {
        // WHATWG Quirks §3.8, vertical twin (w77fix finding 5):
        // in a vertical-writing-mode table in full quirks mode,
        // consecutive `img`s in a cell sum along the vertical
        // inline axis instead of maxing. Two 50px-tall imgs plus
        // 20px top/bottom cell padding give a 140px inline
        // minimum; the table's 50px specified inline size (its
        // `height`) grows to it, exactly as the horizontal arm
        // grows past a narrow specified width. Without the arm
        // the cell measured max(50, 50) + 40 = 90. The
        // standards-mode twin keeps the max fold (90). The imgs
        // carry intrinsic 20x50 sizes (a missing `img` has a 0x0
        // intrinsic, which the vertical atomic arm reads before
        // any CSS height — so CSS sizing alone cannot drive
        // this measurement).
        let css = "table { writing-mode: vertical-lr; height: 50px; border-spacing: 0; } td { padding: 20px 0; } img { width: 20px; height: 50px; }";
        let inner = "<img><img>";
        let lay = |doctype: &str| {
            let html = format!("{doctype}<table><tr><td>{inner}</td></tr></table>");
            let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
            let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
            let imgs = doc.get_elements_by_tag_name("img");
            let mut map = std::collections::HashMap::new();
            for i in 0..imgs.len() {
                if let Some(n) = imgs.item(i) {
                    map.insert(n, crate::IntrinsicDims::from_size(20.0, 50.0));
                }
            }
            crate::layout_with_intrinsics(
                &doc,
                &[&ss],
                Size { w: 800.0, h: 600.0 },
                &FixedFontMetrics,
                &map,
            )
        };
        let quirks = lay("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\">");
        let cell = cell_at(&quirks, 0, 0);
        assert!(
            approx(cell.rect.h, 140.0),
            "quirks vertical cell h {}, want 140 (100 imgs + 40 padding)",
            cell.rect.h
        );
        let standards = lay("<!DOCTYPE html>");
        let cell = cell_at(&standards, 0, 0);
        assert!(
            approx(cell.rect.h, 90.0),
            "standards vertical cell h {}, want 90 (max fold, no quirk)",
            cell.rect.h
        );
    }

    #[test]
    fn w77fix_deferred_replacement_preserves_samemode_inline() {
        // w77fix finding 9: re-placement in
        // `place_deferred_captions` must preserve a same-mode
        // caption's inline position instead of forcing it to the
        // content top. Vertical-lr table with a 10px content top
        // (7px border-top + 3px padding-top): the same-mode
        // caption's inline position is y = 10, with or without a
        // deferred orthogonal caption triggering the re-placement
        // pass; its block position still follows merged order
        // (first caption, so x = 0 at the block-start edge).
        // Probed shapes (lr/rl, fixed/auto inline margins,
        // asymmetric borders) all show the same-mode inline shift
        // as zero with provisional y = content_top by
        // construction, so the preserve is a no-op there — this
        // test pins the invariant so a future inline distribution
        // for same-mode captions cannot be silently clobbered.
        let css = "table { writing-mode: vertical-lr; border-top: 7px solid black; padding-top: 3px; border-spacing: 0; } caption.v { writing-mode: vertical-lr; } caption.h { writing-mode: horizontal-tb; } td { width: 50px; height: 30px; padding: 0; }";
        let same_y = |html: &str| {
            let l = layout_table_html(html, css);
            let table = &l.children[0].children[0];
            let same = table
                .children
                .iter()
                .filter(|c| c.box_type == crate::BoxType::Block)
                .min_by(|a, b| a.rect.x.partial_cmp(&b.rect.x).unwrap())
                .expect("same-mode caption");
            (same.rect.x, same.rect.y)
        };
        let (x_with, y_with) = same_y(
            "<table><caption class=\"v\">same</caption><caption class=\"h\">orth</caption><tr><td>x</td></tr></table>",
        );
        let (_x_without, y_without) =
            same_y("<table><caption class=\"v\">same</caption><tr><td>x</td></tr></table>");
        assert!(
            approx(y_with, 10.0),
            "same-mode y with deferred trigger {y_with}, want 10 (content top)"
        );
        assert!(
            approx(y_with, y_without),
            "same-mode y with deferred trigger {y_with} != without {y_without}"
        );
        assert!(
            approx(x_with, 0.0),
            "same-mode x {x_with}, want 0 (first in merged order)"
        );
    }

    #[test]
    fn w77fix_deferred_bottom_caption_ignores_float_extension() {
        // w77fix finding 14: a float extending an auto-block
        // vertical table past its rows must not push a bottom
        // deferred caption toward the block-end. Div-table
        // fixture (real <table> children would foster-parent
        // the float out): vertical-lr table, horizontal bottom
        // caption, one 100px row, one 300px float. Fail-before:
        // the finalized block size (300, float-extended) anchored
        // the caption at x 300 with table 600. Fixed: the
        // caption anchors after the rows (x 100) and the table
        // extends only by the uncovered part (400 = max(300,
        // 100 + caption)). The caption's width stays the
        // finalized-size fill base (300) — the auto-regime
        // override hands it the base outright; only its
        // placement (and the table total) change.
        let l = layout_table_html(
            r#"<div class="t"><div class="cap">caption</div><div class="f"></div><div class="r"><div class="c">x</div></div></div>"#,
            ".t { display: table; writing-mode: vertical-lr; border-spacing: 0; } .cap { display: table-caption; caption-side: bottom; writing-mode: horizontal-tb; } .f { float: left; width: 300px; height: 20px; } .r { display: table-row; } .c { display: table-cell; width: 100px; height: 30px; padding: 0; }",
        );
        let t = &l.children[0].children[0];
        assert!(approx(t.rect.w, 400.0), "table w {} != 400", t.rect.w);
        let cap = t
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::Block && c.rect.h == 19.2)
            .expect("caption");
        assert!(approx(cap.rect.x, 100.0), "caption x {} != 100", cap.rect.x);
        assert!(approx(cap.rect.w, 300.0), "caption w {} != 300", cap.rect.w);
        let fl = t
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::Block && c.rect.h == 20.0)
            .expect("float");
        assert!(approx(fl.rect.x, 0.0), "float x {} != 0", fl.rect.x);
        assert!(approx(fl.rect.w, 300.0), "float w {} != 300", fl.rect.w);
        let row = t
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::TableRow)
            .expect("row");
        assert!(approx(row.rect.w, 100.0), "row w {} != 100", row.rect.w);
    }

    #[test]
    fn w77fix_row_stray_group_lays_out_end_to_end() {
        // w77fix finding 10: row-stray grouping (CSS Tables 3
        // §2.2.1 rule 2.3) end to end, not just grid-only. A
        // `colspan="2"` real cell, a stray `<span>`, and a real
        // cell share one row: the stray sequence forms one
        // anonymous cell (node-less) in the third column, and
        // the spanning cell's width covers its two columns.
        // FixedFontMetrics at 10px (6px/char): "real" = 24 over
        // two columns, "stray" = 30, "after" = 30; the narrow
        // 50px table grows to the column minima, 84 total.
        // (Divs with table display bypass HTML foster parenting,
        // which would move row strays out of a real `<tr>`.)
        let l = layout_table_html(
            "<div class=\"t\"><div class=\"r\"><div class=\"c\" colspan=\"2\">real</div><span>stray</span><div class=\"c\">after</div></div></div>",
            ".t { display: table; width: 50px; border-spacing: 0; } .r { display: table-row; } .c { display: table-cell; font-size: 10px; padding: 0; } span { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        assert!(
            approx(table.rect.w, 84.0),
            "table w {}, want 84 (24 + 30 + 30)",
            table.rect.w
        );
        let row = &table.children[0];
        assert_eq!(row.box_type, crate::BoxType::TableRow);
        assert_eq!(row.children.len(), 3, "three cells in the row");
        let (real, anon, after) = (&row.children[0], &row.children[1], &row.children[2]);
        assert!(real.node.is_some(), "first cell is a real cell");
        assert!(
            approx(real.rect.w, 24.0),
            "spanning cell w {}, want 24 (its two columns)",
            real.rect.w
        );
        assert!(anon.node.is_none(), "second cell is anonymous");
        assert!(
            approx(anon.rect.x, 24.0) && approx(anon.rect.w, 30.0),
            "anonymous cell at x {} w {}, want x 24 w 30 (\"stray\")",
            anon.rect.x,
            anon.rect.w
        );
        assert!(after.node.is_some(), "third cell is a real cell");
        assert!(
            approx(after.rect.x, 54.0) && approx(after.rect.w, 30.0),
            "after cell at x {} w {}, want x 54 w 30",
            after.rect.x,
            after.rect.w
        );
    }

    #[test]
    fn w77fix_definite_orthogonal_caption_rebases_on_grown_grid() {
        // w77fix finding 7: a definite-spec orthogonal caption
        // must re-base on the finalized grid block size when the
        // rows grow past the specified size. Table vertical-lr,
        // `width: 300px` (block spec), `height: 200px` (inline);
        // caption horizontal, no margins unless noted. The
        // caption's base is max(specified, grid): the specified
        // size when the grid is smaller, the grid when it grows
        // past it — the deferred path's base, which the old
        // definite fast path could not see (it laid out against
        // the specified size before the rows were measured).
        let lay = |row_w: f32, cap_css: &str| {
            let css = format!(
                "table {{ writing-mode: vertical-lr; width: 300px; height: 200px; border-spacing: 0; }} caption {{ writing-mode: horizontal-tb; {cap_css} }} td {{ width: {row_w}px; height: 30px; padding: 0; }}"
            );
            layout_table_html(
                "<table><caption>caption</caption><tr><td>x</td></tr></table>",
                &css,
            )
        };
        let geom = |l: &crate::LayoutBox| {
            let table = &l.children[0].children[0];
            let cap = table
                .children
                .iter()
                .find(|c| c.box_type == crate::BoxType::Block)
                .expect("caption");
            (table.rect.w, cap.rect.w, cap.rect.x)
        };
        // No growth: grid 100 < specified 300 — caption keeps
        // the specified base, table stacks caption + rows.
        let (tw, cw, cx) = geom(&lay(100.0, ""));
        assert!(approx(tw, 400.0), "no-growth table w {tw}, want 400");
        assert!(approx(cw, 300.0), "no-growth caption w {cw}, want 300");
        assert!(approx(cx, 0.0), "no-growth caption x {cx}, want 0");
        // Growth: grid 400 > specified 300 — caption re-bases
        // on the grid, table stacks caption + rows.
        let (tw, cw, _cx) = geom(&lay(400.0, ""));
        assert!(approx(tw, 800.0), "grown table w {tw}, want 800");
        assert!(approx(cw, 400.0), "grown caption w {cw}, want 400");
        // Margins control: no growth, 40px of caption margins —
        // the definite margin-box constraint survives deferral
        // (border 260, not the 300 the auto-regime override
        // would hand it).
        let (tw, cw, cx) = geom(&lay(100.0, "margin-left: 20px; margin-right: 20px;"));
        assert!(approx(tw, 400.0), "margins table w {tw}, want 400");
        assert!(approx(cw, 260.0), "margins caption w {cw}, want 260");
        assert!(approx(cx, 20.0), "margins caption x {cx}, want 20");
        // Bottom side: the caption stacks after the rows, not
        // after the specified floor — no-growth (rows 100) puts
        // it at x 100 with table 400; grown (rows 400) puts it
        // at x 400 with table 800.
        let lay_side = |row_w: f32| {
            let css = format!(
                "table {{ writing-mode: vertical-lr; width: 300px; height: 200px; border-spacing: 0; }} caption {{ writing-mode: horizontal-tb; caption-side: bottom; }} td {{ width: {row_w}px; height: 30px; padding: 0; }}"
            );
            layout_table_html(
                "<table><caption>caption</caption><tr><td>x</td></tr></table>",
                &css,
            )
        };
        let (tw, cw, cx) = geom(&lay_side(100.0));
        assert!(approx(tw, 400.0), "bottom no-growth table w {tw}, want 400");
        assert!(
            approx(cw, 300.0),
            "bottom no-growth caption w {cw}, want 300"
        );
        assert!(
            approx(cx, 100.0),
            "bottom no-growth caption x {cx}, want 100"
        );
        let (tw, cw, cx) = geom(&lay_side(400.0));
        assert!(approx(tw, 800.0), "bottom grown table w {tw}, want 800");
        assert!(approx(cw, 400.0), "bottom grown caption w {cw}, want 400");
        assert!(approx(cx, 400.0), "bottom grown caption x {cx}, want 400");
    }

    #[test]
    fn w77fix_quirks_nowrap_width_floors_min_content() {
        // WHATWG Quirks §3.9 (w77fix finding 8): a `td` with a
        // `nowrap` attribute and a nonzero length width measures
        // its outer min-content as the max of that width and its
        // nowrap (max-content) measure. FixedFontMetrics at
        // 20px: 0.6em advances, so "nowrap nowrap nowrap
        // nowrap" (27 chars) unwraps to 324px; with 20px cell
        // padding the nowrap outer measure is 364. The wrapping
        // min-content would be only the longest word (72) +
        // padding = 112, and a bare width floor would give
        // max(50, 112) = 112 — the WPT `table-quirks` case (580
        // under Ahem) shows the full-line measure is what counts.
        // The table's 50px specified width grows to the column
        // minimum in every case, as in the §3.8 tests.
        let css =
            "table { width: 50px; border-spacing: 0; } td { padding: 20px; font-size: 20px; }";
        let quirks_doctype = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\">";
        let lay = |doctype: &str, inner: &str| {
            layout_table_html(&format!("{doctype}<table><tr>{inner}</tr></table>"), css)
        };
        // Main case: nowrap + 50px width + multi-word text.
        let l = lay(
            quirks_doctype,
            "<td nowrap style=\"width: 50px\">nowrap nowrap nowrap nowrap</td>",
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "quirks nowrap table w {}, want 364 (full line 324 + 40 padding)",
            table.rect.w
        );
        // Width floor case: nowrap + 200px content-box width +
        // short text (nowrap measure 24 + 40 = 64) floors at the
        // width's outer measure (200 + 40 padding = 240).
        let l = lay(quirks_doctype, "<td nowrap style=\"width: 200px\">hi</td>");
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 240.0),
            "quirks nowrap floor table w {}, want 240",
            table.rect.w
        );
        // Standards control (w79: the hint is now ungated, so the
        // standards cell unwraps like the quirks/limited cases —
        // 364, not the old wrapping min-content 112; the §3.9 floor
        // itself stays full-quirks-only and plays no part here).
        let l = lay(
            "<!DOCTYPE html>",
            "<td nowrap style=\"width: 50px\">nowrap nowrap nowrap nowrap</td>",
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "standards nowrap table w {}, want 364 (hint unwraps, no floor)",
            table.rect.w
        );
        // Standards 200px/no-floor case (w79, mirror of the
        // limited-quirks 200px case): hint content measure, no
        // floor — "hi" 24 + 40 padding = 64, and the 50px table
        // grows to it.
        let l = lay(
            "<!DOCTYPE html>",
            "<td nowrap style=\"width: 200px\">hi</td>",
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 64.0),
            "standards nowrap width table w {}, want 64 (hint content measure, no floor)",
            table.rect.w
        );
        // Percentage width is not a length: no §3.9 floor — but the
        // cascade `nowrap` hint still unwraps the cell, so the
        // measure is the full line (364), not the wrapping
        // min-content (112). Same for zero width below.
        let l = lay(
            quirks_doctype,
            "<td nowrap style=\"width: 50%\">nowrap nowrap nowrap nowrap</td>",
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "quirks nowrap % width table w {}, want 364 (hint unwraps, no floor)",
            table.rect.w
        );
        // Zero width is not nonzero: no §3.9 floor, hint unwraps.
        let l = lay(
            quirks_doctype,
            "<td nowrap style=\"width: 0\">nowrap nowrap nowrap nowrap</td>",
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "quirks nowrap 0 width table w {}, want 364 (hint unwraps, no floor)",
            table.rect.w
        );
        // Spanning arm: the nowrap floor distributes over the
        // spanned columns (row 2 pins each column at 52; the
        // spanning cell's 364 floor lifts the total to 364).
        let l = layout_table_html(
            &format!(
                "{quirks_doctype}<table><tr><td colspan=\"2\" nowrap style=\"width: 50px\">nowrap nowrap nowrap nowrap</td></tr><tr><td>a</td><td>b</td></tr></table>"
            ),
            css,
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "quirks nowrap spanning table w {}, want 364",
            table.rect.w
        );
    }

    #[test]
    fn w78fix_limited_quirks_nowrap_width_floors_min_content() {
        // WPT `table-limited-quirks.html` table 3 (fail-before):
        // same fixture as `w77fix_quirks_nowrap_width_floors_min_content`
        // but under the WPT limited-quirks doctype (HTML 4.01
        // Transitional with the loose.dtd system id). Under Ahem
        // 20px the expected cell width is 580 (full line 540 + 40
        // padding); under FixedFontMetrics (0.6em advances) the
        // analogue is 364 (full line 324 + 40 padding). Pre-fix the
        // limited-quirks gate in `quirks_nowrap_min_floor` returns
        // None, so the column minimum is the wrapping min-content
        // (longest word 72 + 40 = 112) — the FixedFont analogue of
        // the WPT 160 (longest word 120 + 40 under Ahem).
        // w78fix: the limited-quirks arm is removed; the main case
        // below now passes via the cascade `nowrap` hint instead of
        // a floor, and the width / percentage / zero cases below
        // are updated to the hint-only results (no §3.9 floor in
        // limited-quirks).
        let css =
            "table { width: 50px; border-spacing: 0; } td { padding: 20px; font-size: 20px; }";
        let limited_doctype = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\">";
        let lay = |inner: &str| {
            layout_table_html(
                &format!("{limited_doctype}<table><tr>{inner}</tr></table>"),
                css,
            )
        };
        // Main case: nowrap + 50px width + multi-word text.
        let l = lay("<td nowrap style=\"width: 50px\">nowrap nowrap nowrap nowrap</td>");
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "limited-quirks nowrap table w {}, want 364 (full line 324 + 40 padding)",
            table.rect.w
        );
        // Width case (w78fix: the §3.9 floor does NOT apply in
        // limited-quirks — that was the old, removed arm — so the
        // 200px specified width does not floor the minimum; the
        // column minimum is the hint-driven content measure, 24 +
        // 40 = 64, and the 50px table grows to it).
        let l = lay("<td nowrap style=\"width: 200px\">hi</td>");
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 64.0),
            "limited-quirks nowrap width table w {}, want 64 (hint content measure, no floor)",
            table.rect.w
        );
        // Percentage width is not a length: no floor — but the hint
        // unwraps, so the measure is the full line, not 112.
        let l = lay("<td nowrap style=\"width: 50%\">nowrap nowrap nowrap nowrap</td>");
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "limited-quirks nowrap % width table w {}, want 364 (hint unwraps, no floor)",
            table.rect.w
        );
        // Zero width is not nonzero: no floor, hint unwraps.
        let l = lay("<td nowrap style=\"width: 0\">nowrap nowrap nowrap nowrap</td>");
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "limited-quirks nowrap 0 width table w {}, want 364 (hint unwraps, no floor)",
            table.rect.w
        );
        // Spanning arm: same hint-driven measure, distributed over
        // spanned columns.
        let l = layout_table_html(
            &format!(
                "{limited_doctype}<table><tr><td colspan=\"2\" nowrap style=\"width: 50px\">nowrap nowrap nowrap nowrap</td></tr><tr><td>a</td><td>b</td></tr></table>"
            ),
            css,
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "limited-quirks nowrap spanning table w {}, want 364",
            table.rect.w
        );
    }

    #[test]
    fn w78fix_limited_quirks_nowrap_author_override_and_hint() {
        // w78fix item 1 (fail-before: the author-override case
        // measures 364 and the auto-width case measures 112, both
        // wrong): with the cascade `nowrap` hint in place and the
        // limited-quirks admission removed from
        // `quirks_nowrap_min_floor`, an author `white-space: normal`
        // beats the hint and the cell wraps (112), while a nowrap
        // cell with no specified width at all unwraps via the hint
        // (364) — the reverse gap the floor arm could never close.
        let css =
            "table { width: 50px; border-spacing: 0; } td { padding: 20px; font-size: 20px; }";
        let limited_doctype = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\">";
        let lay = |inner: &str| {
            layout_table_html(
                &format!("{limited_doctype}<table><tr>{inner}</tr></table>"),
                css,
            )
        };
        let l = lay(
            "<td nowrap style=\"width: 50px; white-space: normal\">nowrap nowrap nowrap nowrap</td>",
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 112.0),
            "limited-quirks author-normal nowrap table w {}, want 112 (wrapping min-content)",
            table.rect.w
        );
        let l = lay("<td nowrap>nowrap nowrap nowrap nowrap</td>");
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "limited-quirks auto-width nowrap table w {}, want 364 (hint unwraps)",
            table.rect.w
        );
        // The hint also unwraps the percentage / zero-width cases
        // that the old floor arm left wrapping.
        let l = lay("<td nowrap style=\"width: 50%\">nowrap nowrap nowrap nowrap</td>");
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "limited-quirks %-width nowrap table w {}, want 364 (hint unwraps)",
            table.rect.w
        );
        let l = lay("<td nowrap style=\"width: 0\">nowrap nowrap nowrap nowrap</td>");
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "limited-quirks 0-width nowrap table w {}, want 364 (hint unwraps)",
            table.rect.w
        );
    }

    #[test]
    fn w78fix_full_quirks_nowrap_author_override() {
        // w78fix item 1, quirks arm (w79fix: the §3.9 floor itself is
        // no longer gated on resolved `white-space` — see
        // `quirks_nowrap_min_floor` — but this fixture is unchanged):
        // an author `white-space: normal` on a full-quirks nowrap
        // cell wraps (112) instead of being floored to the unwrapped
        // line (364, the pre-fix result), because the 50px width's
        // outer floor (90) sits below the wrapping min-content.
        let css =
            "table { width: 50px; border-spacing: 0; } td { padding: 20px; font-size: 20px; }";
        let quirks_doctype = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\">";
        let l = layout_table_html(
            &format!(
                "{quirks_doctype}<table><tr><td nowrap style=\"width: 50px; white-space: normal\">nowrap nowrap nowrap nowrap</td></tr></table>"
            ),
            css,
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 112.0),
            "full-quirks author-normal nowrap table w {}, want 112 (wrapping min-content)",
            table.rect.w
        );
    }

    #[test]
    fn w79fix_quirks_nowrap_author_override_width_floor() {
        // w79fix round 2: the discriminating author-override case
        // for the §3.9 floor. The width-50 fixture above cannot tell
        // floor-on from floor-off (outer floor 90 sits below the
        // wrapping min-content 112), and the width-200 case in
        // `w79fix_quirks_nowrap_width_attribute_wraps` is the
        // width-attribute wrap-reset case, not author override.
        // Here the width comes from author CSS (`width: 200px`)
        // with author `white-space: normal`, so the cell wraps
        // (min-content 112) and the §3.9 floor — nowrap attribute
        // plus a nonzero-length computed width, with no gating on
        // resolved `white-space` — floors its outer min-content to
        // 200 + 40 padding = 240. Floor-off would give 112: a
        // specified width alone never floors the column minimum
        // (CSS Tables 3 §3.8.3, see
        // `specified_cell_width_does_not_floor_column_minimum`),
        // and with a 50px table the assignable width is
        // max(50, minima), so the column maximum cannot pull the
        // table up to 240 either. The limited-quirks twin proves
        // the 240 is the full-quirks floor, not the CSS width
        // itself: the same cell there measures 112.
        let css =
            "table { width: 50px; border-spacing: 0; } td { padding: 20px; font-size: 20px; }";
        let quirks_doctype = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\">";
        let limited_doctype = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\">";
        let l = layout_table_html(
            &format!(
                "{quirks_doctype}<table><tr><td nowrap style=\"width: 200px; white-space: normal\">nowrap nowrap nowrap nowrap</td></tr></table>"
            ),
            css,
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 240.0),
            "full-quirks author-normal nowrap width-200 table w {}, want 240 (floor on despite author override)",
            table.rect.w
        );
        let l = layout_table_html(
            &format!(
                "{limited_doctype}<table><tr><td nowrap style=\"width: 200px; white-space: normal\">nowrap nowrap nowrap nowrap</td></tr></table>"
            ),
            css,
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 112.0),
            "limited-quirks author-normal nowrap width-200 table w {}, want 112 (no §3.9 floor outside full quirks)",
            table.rect.w
        );
    }

    #[test]
    fn w79fix_quirks_nowrap_width_attribute_wraps() {
        // HTML §15.3.8 end of Tables (w79, fail-before: the
        // quirks case measured 364, unwrapped): in full quirks
        // only, `nowrap` plus a nonzero-length `width` ATTRIBUTE
        // resets the cell to `white-space: normal`, so it wraps —
        // wrapping min-content (longest word 72 + 40 padding =
        // 112). w79fix floor interaction: the reset does NOT switch
        // off the §3.9 floor — Quirks §3.9 conditions only on the
        // nowrap attribute plus a nonzero-length computed width, and
        // historical Gecko kept both the wrap reset and the width
        // floor. The 112 below is the wrap-plus-floor result, not a
        // floor-off result: the 50px width's outer floor (50 + 40
        // padding = 90) sits below the wrapping min-content (112),
        // so floor-on and floor-off are indistinguishable here.
        // The width="200" case below discriminates: spec/Gecko floor
        // gives max(200 + 40, 112) = 240, floor-off gives 112.
        // The reset gate is full-quirks-only, so limited-quirks and
        // standards cells with the same attribute still unwrap via
        // the hint (364). CSS style width never triggers the reset.
        let css =
            "table { width: 50px; border-spacing: 0; } td { padding: 20px; font-size: 20px; }";
        let quirks_doctype = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 Transitional//EN\">";
        let limited_doctype = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\">";
        let l = layout_table_html(
            &format!(
                "{quirks_doctype}<table><tr><td nowrap width=\"50\">nowrap nowrap nowrap nowrap</td></tr></table>"
            ),
            css,
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 112.0),
            "quirks nowrap width-attr table w {}, want 112 (wraps, 90 floor below 112 min-content)",
            table.rect.w
        );
        // Discriminating floor case: width="200" floors the wrapping
        // min-content up to its outer width (200 + 40 padding = 240).
        let l = layout_table_html(
            &format!(
                "{quirks_doctype}<table><tr><td nowrap width=\"200\">nowrap nowrap nowrap nowrap</td></tr></table>"
            ),
            css,
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 240.0),
            "quirks nowrap width-attr-200 table w {}, want 240 (floor on despite wrap reset)",
            table.rect.w
        );
        // Percentage width attribute: not a length, no reset.
        let l = layout_table_html(
            &format!(
                "{quirks_doctype}<table><tr><td nowrap width=\"50%\">nowrap nowrap nowrap nowrap</td></tr></table>"
            ),
            css,
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "quirks nowrap %-width-attr table w {}, want 364 (still nowrap)",
            table.rect.w
        );
        // Gate checks: limited-quirks and standards keep the hint.
        let l = layout_table_html(
            &format!(
                "{limited_doctype}<table><tr><td nowrap width=\"50\">nowrap nowrap nowrap nowrap</td></tr></table>"
            ),
            css,
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "limited-quirks nowrap width-attr table w {}, want 364 (no reset)",
            table.rect.w
        );
        let l = layout_table_html(
            "<!DOCTYPE html><table><tr><td nowrap width=\"50\">nowrap nowrap nowrap nowrap</td></tr></table>",
            css,
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 364.0),
            "standards nowrap width-attr table w {}, want 364 (no reset)",
            table.rect.w
        );
    }

    #[test]
    fn cell_max_height_is_ignored() {
        // css-tables-3 height distribution defines only `height` as a
        // minimum; `max-height` has no effect on table cells (WPT
        // `table-cell-overflow-explicit-height-002`: a 20px max-height
        // must not clip a cell whose content is 300px tall).
        let l = layout_table_html(
            "<table><tr><td><div class=\"tall\"></div></td></tr></table>",
            "td { height: 20px; max-height: 20px; } .tall { height: 300px; }",
        );
        let cell_h = cell_at(&l, 0, 0).rect.h;
        assert!(cell_h >= 300.0, "cell.h={cell_h}");
    }

    #[test]
    fn collapsed_losing_cell_border_does_not_widen_column() {
        // WPT `border-conflict-resolution`: under `border-collapse:
        // collapse` a cell's own (losing) border is drawn centered on
        // the shared grid line, so it must not inflate the column's
        // intrinsic width — the table is exactly as wide with the
        // losing border as without it.
        let css = "table { border-collapse: collapse; border: 5px solid green; } \
                   td { font-size: 32px; }";
        let with = layout_table_html(
            "<table><tr><td style=\"border-right: 5px ridge red;\">one</td> \
             <td style=\"border-left: 5px solid blue;\">two</td></tr></table>",
            css,
        );
        let without = layout_table_html(
            "<table><tr><td>one</td><td style=\"border-left: 5px solid blue;\">two</td></tr></table>",
            css,
        );
        let w1 = with.children[0].children[0].rect.w;
        let w2 = without.children[0].children[0].rect.w;
        assert!(approx(w1, w2), "w1={w1} w2={w2}");
    }

    #[test]
    fn definite_width_block_child_sets_table_column_width() {
        // A block-level child with a definite width contributes that
        // width to the column's intrinsic widths (css-sizing-3 §5.2): the
        // auto table sizes to the 300px div, not to zero.
        let l = layout_table_html(
            "<table><tr><td><div style=\"width: 300px; height: 10px;\"></div></td></tr></table>",
            "table { border: 1px solid black; border-spacing: 0; } td { padding: 0; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 302.0), "w={}", table.rect.w);
    }

    #[test]
    fn fixed_layout_spacing_starved_grows_table() {
        // `table-layout: fixed` with no pinned columns: the open column
        // keeps its zero width, but the table still grows to fit the
        // border-spacing (2 × 60px) that alone exceeds its 100px
        // specified width.
        let l = layout_table_html(
            "<table><tr><td>x</td></tr></table>",
            "table { width: 100px; table-layout: fixed; border-spacing: 60px; } \
             td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 120.0), "table={}", table.rect.w);
    }

    #[test]
    fn spacing_starved_growth_respects_max_width() {
        // Growth never passes the table's own `max-width`: spacing (120)
        // plus the column needs more than the 110px cap allows, so the
        // table stops at the cap and the column overflows instead.
        let l = layout_table_html(
            "<table><tr><td>x</td></tr></table>",
            "table { width: 100px; max-width: 110px; border-spacing: 60px; } \
             td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        let cell = cell_at(&l, 0, 0);
        assert!(approx(table.rect.w, 110.0), "table={}", table.rect.w);
        // Capped by max-width: spacing (120) + the column no longer
        // tile the table — the column overflows instead of growing it.
        assert!(
            120.0 + cell.rect.w > table.rect.w,
            "table={} cell={}",
            table.rect.w,
            cell.rect.w
        );
    }

    #[test]
    fn auto_layout_colgroup_width_applies_at_full_value() {
        // WPT `colgroup-col`: a `colgroup` width is the width of each
        // column it generates, not a total to spread. The 200px group
        // gives col1 (no own `col` width) the full 200px. col0's own
        // 150px `col` width floors its *maximum*, not its minimum (WPT
        // `column-widths`: column.min = td.min), so with the table's
        // 100px specified width the grid minimum is just col0's content
        // (~6px) plus col1's 200px group floor: the table grows to 206,
        // col0 stays at content width, col1 at the full 200px.
        let l = layout_table_html(
            "<table><colgroup class=\"g\"><col class=\"a\"><col></colgroup>\
             <tr><td>x</td><td>y</td></tr></table>",
            "table { width: 100px; } colgroup.g { width: 200px; } col.a { width: 150px; } \
             td { font-size: 10px; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 6.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 200.0));
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 206.0));
    }

    #[test]
    fn auto_layout_colgroup_width_applies_to_each_column() {
        // WPT `colgroup-col`: a 200px `span=2` group yields 200px
        // columns, not 100px each. The table grows to 400.
        let l = layout_table_html(
            "<table><colgroup class=\"g\" span=\"2\"></colgroup>\
             <tr><td>x</td><td>y</td></tr></table>",
            "table { width: 100px; } colgroup.g { width: 200px; } \
             td { font-size: 10px; }",
        );
        assert!(approx(cell_at(&l, 0, 0).rect.w, 200.0));
        assert!(approx(cell_at(&l, 0, 1).rect.w, 200.0));
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 400.0));
    }

    #[test]
    fn column_specified_widths_flags_group_width_only() {
        // Under the WPT `column-widths` semantics, a `col` width floors the
        // column's maximum only; the column minimum comes from the cells
        // (`column.min = td.min`). A `colgroup` width floors the column
        // minimum instead. `column_specified_widths` flags the
        // group-sourced columns so automatic layout applies the maximum
        // raise to `col` widths alone (previously the group width raised
        // the maximum too).
        let (mut forest, table) = forest_table(
            "<table><colgroup><col class=\"a\"><col></colgroup>\
             <colgroup class=\"g\"><col><col></colgroup>\
             <tr><td>x</td><td>y</td><td>z</td><td>w</td></tr></table>",
            "col.a { width: 100px; } colgroup.g { width: 400px; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.ncols, 4);
        let (widths, group_only) = column_specified_widths(&mut forest, &grid, 0.0, 0.0);
        // col.a pins column 0; column 1 has no specified width; the
        // 400px group gives columns 2-3 the full 400px each, flagged
        // min-only.
        assert!(approx(widths[0].unwrap(), 100.0));
        assert!(!group_only[0]);
        assert_eq!(widths[1], None);
        assert!(!group_only[1]);
        assert!(approx(widths[2].unwrap(), 400.0));
        assert!(group_only[2]);
        assert!(approx(widths[3].unwrap(), 400.0));
        assert!(group_only[3]);
    }

    #[test]
    fn auto_layout_group_width_raises_column_minima() {
        // The 400px group width floors each covered column's minimum
        // at 400px (step 4); the maxima stay with the content. (Final
        // column widths cannot show the max side: the max>=min fixup
        // masks a group-raised maximum, so the flag above is the
        // regression guard for it.)
        let (mut forest, table) = forest_table(
            "<table><colgroup class=\"g\" span=\"2\"></colgroup>\
             <tr><td>x</td><td>y</td></tr></table>",
            "table { border-collapse: collapse; } \
             colgroup.g { width: 400px; } td { font-size: 10px; padding: 0; border: none; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.ncols, 2);
        let col_spec = column_specified_widths(&mut forest, &grid, 0.0, 0.0);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut mcw_scratch = Vec::new();
        let mut int_bufs = WalkBufs::new();
        let mut int_inner = WalkBufs::new();
        let (col_max, col_max_nonspanning) = auto_column_maxima(
            &mut forest,
            &grid,
            &col_spec,
            0.0,
            &FixedFontMetrics,
            &mut cache,
            &mut mcw_scratch,
            &mut int_bufs,
            &mut int_inner,
        );
        let mins = auto_column_minima(
            &mut forest,
            &grid,
            &col_spec,
            0.0,
            0.0,
            &FixedFontMetrics,
            &col_max,
            &col_max_nonspanning,
        );
        assert_eq!(mins.len(), 2);
        assert!(approx(mins[0], 400.0), "col0 min {}", mins[0]);
        assert!(approx(mins[1], 400.0), "col1 min {}", mins[1]);
    }

    #[test]
    fn colgroup_pct_intrinsic_applies_full_value_per_column() {
        // CSS Tables 3 §3.8.3 names the table-column-group's
        // percentage contribution with no division: a 40% `colgroup`
        // spanning two columns contributes 40% to *each* column, not
        // 20% — mirroring the used-width spread in
        // `column_specified_widths` (WPT `colgroup-col`: a 200px
        // `span=2` group yields 200px columns, not 100px).
        let (forest, table) = forest_table(
            "<table><colgroup class=\"g\" span=\"2\"></colgroup>\
             <tr><td>x</td><td>y</td></tr></table>",
            "colgroup.g { width: 40%; } td { font-size: 10px; padding: 0; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.ncols, 2);
        let (pcts, _) = intrinsic_column_pcts(&forest, &grid, &[0.0; 2]);
        assert_eq!(pcts.len(), 2);
        assert!(approx(pcts[0], 0.4), "col0 pct {}", pcts[0]);
        assert!(approx(pcts[1], 0.4), "col1 pct {}", pcts[1]);
    }

    // ---- percentage row/row-group heights (CSS 2.1 §10.5/§17.5.3) ----

    /// Navigate a laid-out table wrapped in a `<div>`:
    /// body > div > table > row-group > row.
    fn row_at_wrapped(l: &crate::LayoutBox, row: usize) -> &crate::LayoutBox {
        let div = &l.children[0].children[0];
        let table = div
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::Table)
            .expect("table");
        let group = table
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::TableRowGroup)
            .expect("row group");
        &group.children[row]
    }

    #[test]
    fn row_pct_height_in_auto_table_resolves_to_auto() {
        // The 400px belongs to the table's containing block, not the
        // table itself: with an auto-height table the 50% computes to
        // `auto`, so the row sizes from content exactly like the
        // no-height control (the old containing-block base would have
        // floored the row at 200px).
        let html = "<div><table><tr class=\"r\"><td>x</td></tr></table></div>";
        let css = "div { height: 400px; } tr.r { height: 50%; }";
        let l = layout_table_html(html, css);
        let control = layout_table_html(html, "");
        let h = row_at_wrapped(&l, 0).rect.h;
        assert!(approx(h, row_at_wrapped(&control, 0).rect.h), "h={h}");
        assert!(h < 100.0, "h={h}");
    }

    #[test]
    fn row_group_pct_height_in_auto_table_resolves_to_auto() {
        // Same for a row group: the percentage minimum must not see the
        // containing block's 400px when the table height is indefinite.
        let html = "<div><table><tbody class=\"g\"><tr><td>x</td></tr></tbody></table></div>";
        let css = "div { height: 400px; } tbody.g { height: 50%; }";
        let l = layout_table_html(html, css);
        let control = layout_table_html(html, "");
        let h = row_at_wrapped(&l, 0).rect.h;
        assert!(approx(h, row_at_wrapped(&control, 0).rect.h), "h={h}");
        assert!(h < 100.0, "h={h}");
    }

    #[test]
    fn row_pct_height_resolves_against_table_own_height() {
        // The table height is definite (200px), so 75% floors the row at
        // 150px; the table's 200px then distributes its 50px excess onto
        // the single row, ending at 200px. Against the old
        // containing-block base (the 400px div) the minimum would be
        // 300px and the row would stay at 300px.
        let html = "<div><table class=\"t\"><tr class=\"r\"><td>x</td></tr></table></div>";
        let css = "div { height: 400px; } table.t { height: 200px; } tr.r { height: 75%; }";
        let l = layout_table_html(html, css);
        let h = row_at_wrapped(&l, 0).rect.h;
        assert!(approx(h, 200.0), "h={h}");
    }

    // ---- percentage column widths in shrink-to-fit tables (CSS 2.1 §17.5.2.2) ----

    #[test]
    fn pct_col_width_in_shrink_to_fit_table() {
        // A floated table shrink-to-fits through `table_pref_width`: the
        // 50% `<col>` width resolves against the table's own width, so
        // column 0 is half the table and the table is twice the wider
        // column's content width. Before the fix the percentage resolved
        // against zero and column 0 stayed at its "x" content width.
        let html =
            "<table class=\"t\"><col class=\"c\"><tr><td>x</td><td>longer cell</td></tr></table>";
        let css_pct = "table.t { float: left; } col.c { width: 50%; } td { font-size: 10px; }";
        let css_auto = "table.t { float: left; } td { font-size: 10px; }";
        let l = layout_table_html(html, css_pct);
        let control = layout_table_html(html, css_auto);
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        let cc0 = cell_at(&control, 0, 0);
        let cc1 = cell_at(&control, 0, 1);
        // The 50% column is half the table: both columns end up at the
        // wider content width, and the table tiles them exactly.
        assert!(
            approx(c0.rect.w, c1.rect.w),
            "c0={} c1={}",
            c0.rect.w,
            c1.rect.w
        );
        assert!(
            approx(c1.rect.w, cc1.rect.w),
            "c1={} control={}",
            c1.rect.w,
            cc1.rect.w
        );
        assert!(
            approx(table.rect.w, c0.rect.w + c1.rect.w),
            "table={}",
            table.rect.w
        );
        // Without the percentage the first column stays narrow.
        assert!(cc0.rect.w < 0.5 * c0.rect.w, "control c0={}", cc0.rect.w);
    }

    #[test]
    fn pct_cell_width_in_shrink_to_fit_table() {
        // Same, but the percentage is on a first-row cell instead of a
        // `<col>`: it acts as the column minimum (§17.5.2.2).
        let html = "<table class=\"t\"><tr><td class=\"a\">x</td><td>longer cell</td></tr></table>";
        let css_pct = "table.t { float: left; } td.a { width: 50%; } td { font-size: 10px; }";
        let css_auto = "table.t { float: left; } td { font-size: 10px; }";
        let l = layout_table_html(html, css_pct);
        let control = layout_table_html(html, css_auto);
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(
            approx(c0.rect.w, c1.rect.w),
            "c0={} c1={}",
            c0.rect.w,
            c1.rect.w
        );
        assert!(
            approx(table.rect.w, c0.rect.w + c1.rect.w),
            "table={}",
            table.rect.w
        );
        assert!(
            cell_at(&control, 0, 0).rect.w < 0.5 * c0.rect.w,
            "control c0={}",
            cell_at(&control, 0, 0).rect.w
        );
    }

    #[test]
    fn pct_col_width_in_fixed_shrink_to_fit_table() {
        // `table-layout: fixed` pins the percentage `<col>` against the
        // table width (§17.5.2.1): the same 50/50 split as the automatic
        // layout.
        let html =
            "<table class=\"t\"><col class=\"c\"><tr><td>x</td><td>longer cell</td></tr></table>";
        let css = "table.t { float: left; table-layout: fixed; } \
                   col.c { width: 50%; } td { font-size: 10px; }";
        let l = layout_table_html(html, css);
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(
            approx(c0.rect.w, c1.rect.w),
            "c0={} c1={}",
            c0.rect.w,
            c1.rect.w
        );
        assert!(
            approx(table.rect.w, c0.rect.w + c1.rect.w),
            "table={}",
            table.rect.w
        );
        assert!(c0.rect.w > 1.0, "c0={}", c0.rect.w);
    }

    #[test]
    fn overconstrained_pct_columns_are_normalized_not_diverged() {
        // Two 60% columns (120% total) would diverge geometrically
        // without normalization: every iteration resolves the
        // percentages against the growing total (60%^32). Instead the
        // intrinsic percentages clamp left-to-right to 60/40 (§3.8.3),
        // so the preferred width stays content-sized: the per-column
        // floor max(col_min, col_pref)/p gives 66/0.6 = 110, and the
        // used-width §3.9.3 percentage guess assigns max(0.6*110, 66) =
        // 66 and max(0.4*110, 44) = 44. The wide wrapper keeps the
        // float's `min(preferred, available)` from masking the
        // divergence.
        let html = "<div class=\"w\"><table class=\"t\"><col class=\"a\"><col class=\"b\">\
                    <tr><td>longer cell</td><td>x</td></tr></table></div>";
        let css_pct = "div.w { width: 100000px; } table.t { float: left; } \
                       col.a { width: 60%; } col.b { width: 60%; } td { font-size: 10px; }";
        let l = layout_table_html(html, css_pct);
        // body > div.w > table.
        let table = &l.children[0].children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        let group = table
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::TableRowGroup)
            .expect("row group");
        let row = &group.children[0];
        let c0 = &row.children[0];
        let c1 = &row.children[1];
        assert!(
            table.rect.w.is_finite() && table.rect.w < 1000.0,
            "table={}",
            table.rect.w
        );
        // Clamped 60/40 shares of the 110px preferred width: both
        // columns tile the table exactly. Bounded far below the
        // ~29,000px the old fixpoint produced here.
        assert!(approx(c0.rect.w, 66.0), "c0={} c1={}", c0.rect.w, c1.rect.w);
        assert!(approx(c1.rect.w, 44.0), "c0={} c1={}", c0.rect.w, c1.rect.w);
        assert!(
            approx(table.rect.w, c0.rect.w + c1.rect.w),
            "table={}",
            table.rect.w
        );
    }

    #[test]
    fn single_overconstrained_pct_column_stays_bounded() {
        // One 150% cell width: 150%^32 without normalization.
        let html = "<table class=\"t\"><tr><td class=\"a\">x</td></tr></table>";
        let css = "table.t { float: left; } td.a { width: 150%; } td { font-size: 10px; }";
        let l = layout_table_html(html, css);
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        let w = table.rect.w;
        assert!(w.is_finite() && w < 1000.0, "table={w}");
        assert!(w > 1.0, "table={w}");
    }

    #[test]
    fn fixed_layout_spanning_min_width_spreads_over_open_columns() {
        // `table-layout: fixed` with a pinned `<col>` and a colspan=2
        // cell carrying a percentage `min-width`: the second
        // preferred-width pass spreads the min-width excess over the
        // open (unpinned) column, mirroring the first pass — it no
        // longer skips the cell outright because one spanned column is
        // pinned. (A percentage min-width resolves against zero in the
        // first pass, so only the second pass — against the table width
        // including spacing — sees it exceed the span.)
        let html = "<table class=\"t\"><col class=\"c\">\
                    <tr><td colspan=\"2\" class=\"w\">x</td></tr>\
                    <tr><td>a</td><td>b</td></tr></table>";
        let css = "table.t { float: left; table-layout: fixed; border-spacing: 10px; } \
                   col.c { width: 100px; } td.w { min-width: 100%; } td { font-size: 10px; }";
        let l = layout_table_html(html, css);
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        let c0 = cell_at(&l, 1, 0);
        let c1 = cell_at(&l, 1, 1);
        // The pinned column keeps its width; the open column absorbs the
        // min-width excess instead of staying at content width.
        assert!(approx(c0.rect.w, 100.0), "c0={}", c0.rect.w);
        assert!(c1.rect.w > 30.0, "c1={}", c1.rect.w);
        assert!(
            approx(table.rect.w, c0.rect.w + c1.rect.w + 30.0),
            "table={}",
            table.rect.w
        );
    }

    // ---- zero-column tables (WPT `table-rows-with-zero-columns`) ----

    #[test]
    fn zero_column_rows_span_inner_width_and_share_height() {
        // csswg #10132: rows of a table with no columns are as wide as
        // the table's inner width; a specified table height distributes
        // over the rows.
        let css = "table { box-sizing: border-box; width: 60px; height: 60px; }";
        let l = layout_table_html("<table><tr></tr></table>", css);
        let r = row_at(&l, 0);
        assert!(approx(r.rect.w, 60.0), "w={}", r.rect.w);
        assert!(approx(r.rect.h, 60.0), "h={}", r.rect.h);
        // Two rows split the height evenly.
        let l = layout_table_html("<table><tr></tr><tr></tr></table>", css);
        for i in 0..2 {
            let r = row_at(&l, i);
            assert!(approx(r.rect.w, 60.0), "row {i} w={}", r.rect.w);
            assert!(approx(r.rect.h, 30.0), "row {i} h={}", r.rect.h);
        }
    }

    #[test]
    fn zero_column_rows_account_for_spacing_and_borders() {
        // Border-spacing eats into the distributed height but not the
        // row width; table borders shrink the inner width the rows span.
        // NB: styled without TABLE_TEST_RESET_CSS on purpose — this test
        // exercises the `cellspacing` presentational hint, and an
        // author-level reset would clobber the hint (hints sit below
        // author rules in the cascade).
        let css = "table { box-sizing: border-box; width: 60px; height: 60px; }";
        let layout_no_reset = |html: &str| {
            let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
            let ss = bowser_css::parse_stylesheet(css);
            layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics)
        };
        let l = layout_no_reset("<table cellspacing=\"10\"><tr></tr><tr></tr></table>");
        for i in 0..2 {
            let r = row_at(&l, i);
            assert!(approx(r.rect.w, 60.0), "row {i} w={}", r.rect.w);
            assert!(approx(r.rect.h, 15.0), "row {i} h={}", r.rect.h);
        }
        let l = layout_no_reset("<table cellspacing=\"10\" border=\"5\"><tr></tr></table>");
        let r = row_at(&l, 0);
        assert!(approx(r.rect.w, 50.0), "w={}", r.rect.w);
        assert!(approx(r.rect.h, 30.0), "h={}", r.rect.h);
        let l =
            layout_no_reset("<table cellspacing=\"10\" border=\"5\"><tr></tr><tr></tr></table>");
        for i in 0..2 {
            let r = row_at(&l, i);
            assert!(approx(r.rect.w, 50.0), "row {i} w={}", r.rect.w);
            assert!(approx(r.rect.h, 10.0), "row {i} h={}", r.rect.h);
        }
    }

    // ---- `visibility: collapse` (CSS 2.1 §17.5) ----

    #[test]
    fn collect_table_grid_marks_collapsed_rows_and_columns() {
        // The grid flags drive every zeroing decision below; they read
        // the computed `visibility`, which inherits through row groups
        // and colgroups.
        let (forest, table) = forest_table(
            "<table><col class=\"gc\"><tbody class=\"gg\">\
             <tr class=\"gr\"><td>a</td></tr><tr><td>b</td></tr>\
             </tbody></table>",
            "tr.gr { visibility: collapse; } \
             tbody.gg { visibility: visible; } \
             col.gc { visibility: collapse; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.collapsed_rows, vec![true, false]);
        assert_eq!(grid.collapsed_cols, vec![true]);
    }

    #[test]
    fn collect_table_grid_row_group_collapse_covers_anonymous_rows() {
        // Anonymous rows carry no style of their own; a collapsed row
        // group still collapses them via the group box.
        let (forest, table) = forest_table(
            "<table><tbody class=\"gone\"><td>a</td><td>b</td></tbody></table>",
            "tbody.gone { visibility: collapse; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.nrows, 1);
        assert_eq!(grid.collapsed_rows, vec![true]);
    }

    #[test]
    fn visibility_collapse_row_takes_no_height() {
        // Unlike `visibility: hidden`, a collapsed row takes no space:
        // the table is exactly as tall as the single visible row, whose
        // height matches the no-collapse control.
        let html = "<table><tr><td>aaa</td></tr><tr class=\"gone\"><td>bbb</td></tr></table>";
        let css = "td { font-size: 10px; } tr.gone { visibility: collapse; }";
        let l = layout_table_html(html, css);
        let control = layout_table_html(html, "td { font-size: 10px; }");
        let single = layout_table_html(
            "<table><tr><td>aaa</td></tr></table>",
            "td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        assert!(
            approx(row_at(&l, 1).rect.h, 0.0),
            "collapsed row h={}",
            row_at(&l, 1).rect.h
        );
        assert!(
            approx(row_at(&l, 0).rect.h, row_at(&control, 0).rect.h),
            "visible row h={} control={}",
            row_at(&l, 0).rect.h,
            row_at(&control, 0).rect.h
        );
        let single_table = &single.children[0].children[0];
        assert!(
            approx(table.rect.h, single_table.rect.h),
            "table={} single={}",
            table.rect.h,
            single_table.rect.h
        );
    }

    #[test]
    fn visibility_collapse_row_group_collapses_all_rows() {
        // The keyword inherits onto the rows, so collapsing the row
        // group removes every row it contains.
        let html =
            "<table><tbody class=\"gone\"><tr><td>a</td></tr><tr><td>b</td></tr></tbody></table>";
        let css = "td { font-size: 10px; } tbody.gone { visibility: collapse; }";
        let l = layout_table_html(html, css);
        let table = &l.children[0].children[0];
        assert!(approx(row_at(&l, 0).rect.h, 0.0));
        assert!(approx(row_at(&l, 1).rect.h, 0.0));
        assert!(approx(table.rect.h, 0.0), "table={}", table.rect.h);
    }

    /// Build the box forest for `html` and collect the table grid, so
    /// tests can inspect per-row-group data (like
    /// [`TableGrid::group_kinds`]) that the laid-out [`LayoutBox`]
    /// tree does not surface.
    fn collect_grid(html: &str) -> (crate::boxes::Forest, TableGrid) {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet("");
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
        let mut stack = vec![root];
        let mut table = None;
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == crate::BoxType::Table {
                table = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let grid = collect_table_grid(&forest, table.expect("table box"));
        (forest, grid)
    }

    #[test]
    fn row_group_section_kinds_plumbed() {
        // thead/tbody/tfoot stamp Header/Body/Footer on their row-group
        // boxes, exposed in DOM order via `TableGrid::group_kinds`.
        let (_, grid) = collect_grid(
            "<table><thead><tr><td>h</td></tr></thead>\
             <tbody><tr><td>b</td></tr></tbody>\
             <tfoot><tr><td>f</td></tr></tfoot></table>",
        );
        assert_eq!(grid.group_nodes.len(), 3);
        assert_eq!(
            grid.group_kinds,
            vec![SectionKind::Header, SectionKind::Body, SectionKind::Footer]
        );
    }

    #[test]
    fn row_group_section_kind_accessor_matches_grid() {
        // `row_group_section_kind` agrees with the stamped kind, and
        // returns `None` for non-row-group nodes (only row-group boxes
        // are stamped at construction).
        let (forest, grid) = collect_grid(
            "<table><thead><tr><td>h</td></tr></thead>\
             <tfoot><tr><td>f</td></tr></tfoot></table>",
        );
        assert_eq!(
            grid.group_kinds,
            vec![SectionKind::Header, SectionKind::Footer]
        );
        for (node, kind) in grid.group_nodes.iter().zip(grid.group_kinds.iter()) {
            assert_eq!(row_group_section_kind(&forest, *node), Some(*kind));
        }
        assert_eq!(row_group_section_kind(&forest, grid.cells[0].node), None);
    }

    #[test]
    fn first_visual_row_ranks_restyled_section_display() {
        // Display wins over the tag for visual ranking (CSS Tables 3
        // §2.1: the table-header-group renders before all other rows,
        // the table-footer-group after): a `<thead>` restyled to
        // `table-footer-group` ranks as a footer group, so the body's
        // row is the visual first row.
        let (forest, grid) = collect_grid(
            "<table><thead style=\"display: table-footer-group\"><tr><td>f</td></tr></thead>\
             <tbody><tr><td>b</td></tr></tbody></table>",
        );
        assert_eq!(
            grid.group_kinds,
            vec![SectionKind::Footer, SectionKind::Body]
        );
        assert_eq!(first_visual_row(&forest, &grid), 1);
    }

    #[test]
    fn first_visual_row_tfoot_restyled_header_ranks_first() {
        // Mirror case: a `<tfoot>` restyled to `table-header-group`
        // ranks as a header group, so its row is the visual first row
        // even though it comes from a footer tag.
        let (forest, grid) = collect_grid(
            "<table><tbody><tr><td>b</td></tr></tbody>\
             <tfoot style=\"display: table-header-group\"><tr><td>h</td></tr></tfoot></table>",
        );
        assert_eq!(
            grid.group_kinds,
            vec![SectionKind::Body, SectionKind::Header]
        );
        assert_eq!(first_visual_row(&forest, &grid), 1);
    }

    #[test]
    fn collapsed_thead_keeps_header_kind() {
        // Toward `visibility: collapse` semantics for header groups
        // (CSS Tables 3 §5.4): a collapsed thead still collapses, and
        // its section kind is visible exactly where the collapse flags
        // are computed.
        let (_, grid) = collect_grid(
            "<table><thead style=\"visibility: collapse\"><tr><td>h</td></tr></thead>\
             <tbody><tr><td>b</td></tr></tbody></table>",
        );
        assert_eq!(
            grid.group_kinds,
            vec![SectionKind::Header, SectionKind::Body]
        );
        assert!(grid.collapsed_rows[0]);
        assert!(!grid.collapsed_rows[1]);
    }

    #[test]
    fn visibility_collapse_explicit_visible_row_still_collapses() {
        // A row that overrides back to `visible` inside a collapsed row
        // group still takes no space: the group is removed as a unit
        // (CSS 2.1 §17.5.5), so the row has no boxes to show in. (WPT
        // `visibility-collapse-row-group-001`.)
        let html = "<table><tbody class=\"gone\"><tr class=\"back\"><td>a</td></tr>\
                    <tr><td>b</td></tr></tbody></table>";
        let css = "td { font-size: 10px; } tbody.gone { visibility: collapse; } \
                   tr.back { visibility: visible; }";
        let l = layout_table_html(html, css);
        assert!(
            approx(row_at(&l, 0).rect.h, 0.0),
            "row0 h={}",
            row_at(&l, 0).rect.h
        );
        assert!(approx(row_at(&l, 1).rect.h, 0.0));
    }

    #[test]
    fn visibility_collapse_column_takes_no_width() {
        // A collapsed column takes no width: under fixed layout the
        // remaining column absorbs the whole table width and tiles it
        // exactly, with the collapsed column's cell at zero width.
        let html = "<table class=\"t\"><col><col class=\"gone\">\
                    <tr><td>a</td><td>b</td></tr></table>";
        let css = "table.t { table-layout: fixed; width: 300px; } \
                   td { font-size: 10px; } col.gone { visibility: collapse; }";
        let l = layout_table_html(html, css);
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c1.rect.w, 0.0), "collapsed cell w={}", c1.rect.w);
        assert!(approx(c0.rect.w, 300.0), "c0={}", c0.rect.w);
        assert!(approx(table.rect.w, 300.0), "table={}", table.rect.w);
    }

    #[test]
    fn visibility_collapse_column_auto_layout_ignores_its_content() {
        // Automatic layout: the collapsed column's (wide) content must
        // not widen the table; the table matches the single-column
        // control exactly.
        let html = "<table><col><col class=\"gone\">\
                    <tr><td>x</td><td>abcdefghijklmnopqrstuvwxyz</td></tr></table>";
        let css = "td { font-size: 10px; } col.gone { visibility: collapse; }";
        let l = layout_table_html(html, css);
        let control = layout_table_html(
            "<table><tr><td>x</td></tr></table>",
            "td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        let ctable = &control.children[0].children[0];
        assert!(
            approx(table.rect.w, ctable.rect.w),
            "table={} control={}",
            table.rect.w,
            ctable.rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 0.0),
            "collapsed cell w={}",
            cell_at(&l, 0, 1).rect.w
        );
    }

    #[test]
    fn all_collapsed_columns_draw_no_phantom_spacing() {
        // CSS 2.1 §17.5: collapsed columns are removed from the display
        // and draw no inter-column spacing. The `(nvis + 1)` spacing
        // formula is guarded on the *visible* column count (not the
        // column count): with every column collapsed the unguarded
        // formula leaves one phantom `border-spacing` gap in the
        // table's intrinsic width.
        let l = layout_table_html(
            "<div class=\"f\"><table class=\"t\">\
             <col class=\"gone\"><col class=\"gone\">\
             <tr><td>x</td><td>y</td></tr></table></div>",
            "body { margin: 0; } div.f { float: left; } \
             table.t { border-spacing: 8px; } \
             td { font-size: 10px; padding: 0; } col.gone { visibility: collapse; }",
        );
        // body > div.f > table.
        let f = &l.children[0].children[0];
        assert!(
            approx(f.rect.w, 0.0),
            "phantom spacing in all-collapsed table: {}",
            f.rect.w
        );
    }

    #[test]
    fn zero_column_table_draws_no_phantom_spacing() {
        // The degenerate case of the guard above: a table with no
        // columns at all draws no spacing either.
        let l = layout_table_html(
            "<div class=\"f\"><table class=\"t\"></table></div>",
            "body { margin: 0; } div.f { float: left; } \
             table.t { border-spacing: 8px; }",
        );
        // body > div.f > table.
        let f = &l.children[0].children[0];
        assert!(
            approx(f.rect.w, 0.0),
            "phantom spacing in zero-column table: {}",
            f.rect.w
        );
    }

    #[test]
    fn phantom_w_stamps_cell_less_table() {
        // A cell-less table whose width comes purely from the
        // border-spacing phantom stamps the width dimension: width auto,
        // no caption, no used borders, laid-out width inside a single
        // spacing. A zero-column table draws no phantom gap at all, so
        // its width is 0 — still inside the spacing, still stamped. (A
        // *truly* childless `<table></table>` takes `layout_block`'s
        // empty-block early return and never reaches the table engine;
        // the shim's fallback derivation covers it. One empty row routes
        // through the engine.)
        let l = layout_table_html(
            "<table class=\"t\"><tr></tr></table>",
            "body { margin: 0; } table.t { border-spacing: 8px; }",
        );
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        assert!(approx(table.rect.w, 0.0), "table w={}", table.rect.w);
        assert!(
            table.phantom_w,
            "cell-less phantom table width not stamped (w={})",
            table.rect.w
        );
    }

    #[test]
    fn phantom_absent_with_cells() {
        // A table with real cells is never phantom-sized, even when
        // narrow.
        let l = layout_table_html(
            "<table><tr><td>a</td></tr></table>",
            "td { font-size: 10px; padding: 0; }",
        );
        let table = &l.children[0].children[0];
        assert!(
            !table.phantom_w && !table.phantom_h,
            "table with cells stamped phantom"
        );
    }

    #[test]
    fn phantom_absent_with_specified_width() {
        // A specified width keeps its number: the shim honors it, so the
        // width stamp stays off even for a cell-less table. (Truly
        // childless, so no rows either: the height stamp stays off too.)
        let l = layout_table_html(
            "<table class=\"t\"></table>",
            "body { margin: 0; } table.t { width: 100px; border-spacing: 8px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 100.0), "table w={}", table.rect.w);
        assert!(
            !table.phantom_w && !table.phantom_h,
            "specified-width cell-less table stamped phantom"
        );
    }

    #[test]
    fn phantom_h_stamps_height_only() {
        // The dimensions stamp independently: a cell-less table with
        // rows whose height fits the vertical phantom stamps the height
        // even when the width does not qualify (specified 100px width).
        let l = layout_table_html(
            "<table class=\"t\"><tr></tr><tr></tr></table>",
            "body { margin: 0; } table.t { width: 100px; border-spacing: 8px; }",
        );
        let table = &l.children[0].children[0];
        assert!(
            !table.phantom_w,
            "specified-width table stamped width-phantom"
        );
        assert!(
            table.phantom_h,
            "height-phantom cell-less table not stamped (h={})",
            table.rect.h
        );
    }

    #[test]
    fn phantom_w_absent_with_caption() {
        // Caption bleed: a `<caption>` exempts only the width — the
        // caption box is real content, so the width is not the phantom
        // even when it fits inside one spacing. The old single-bool OR
        // plus the shim's re-derivation zeroed it anyway.
        let l = layout_table_html(
            "<table class=\"t\"><caption>c</caption><tr></tr></table>",
            "body { margin: 0; } table.t { border-spacing: 8px; } caption { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(
            !table.phantom_w,
            "caption table width wrongly phantom (w={})",
            table.rect.w
        );
    }

    #[test]
    fn phantom_w_absent_with_min_width() {
        // Min-width bleed: a `min-width` floor makes the laid-out width
        // real (it exceeds the spacing), so the width stamp stays off
        // even for a cell-less auto-width table.
        let l = layout_table_html(
            "<table class=\"t\"><tr></tr></table>",
            "body { margin: 0; } table.t { border-spacing: 8px; min-width: 50px; }",
        );
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 50.0), "table w={}", table.rect.w);
        assert!(
            !table.phantom_w,
            "min-width table width wrongly phantom (w={})",
            table.rect.w
        );
    }

    #[test]
    fn phantom_h_absent_without_rows() {
        // Rows==0 bleed: with no rows there is no vertical phantom to
        // fit inside, so the height stamp stays off even when the width
        // stamps. The old OR stamp let the shim zero the height.
        let l = layout_table_html(
            "<table class=\"t\"><tbody></tbody></table>",
            "body { margin: 0; } table.t { border-spacing: 8px; }",
        );
        let table = &l.children[0].children[0];
        assert!(
            !table.phantom_h,
            "row-less table height wrongly phantom (h={})",
            table.rect.h
        );
    }

    #[test]
    fn phantom_display_none_cells_stamp() {
        // `display: none` cells generate no boxes, so the engine counts
        // the table as cell-less and stamps the phantom — unlike the old
        // shim, which counted DOM `td`/`th`/`tr` (including
        // `display: none`) as a veto. The new behavior is the more
        // correct one: there is no cell box to measure, so the width is
        // the border-spacing phantom. Pinned explicitly.
        let l = layout_table_html(
            "<table class=\"t\"><tr><td style=\"display:none\">x</td></tr></table>",
            "body { margin: 0; } table.t { border-spacing: 8px; } td { font-size: 10px; padding: 0; }",
        );
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        assert!(
            table.phantom_w,
            "display:none-cell table width not stamped (w={})",
            table.rect.w
        );
    }

    #[test]
    fn visibility_collapse_toggle_relays_out_on_relayout() {
        // Dynamic toggle: collapsing a row and re-laying out shrinks the
        // table by exactly the row's height; the style change is picked
        // up on re-layout with no stale geometry.
        let html = "<table><tr><td>aaa</td></tr><tr class=\"t\"><td>bbb</td></tr></table>";
        let shown = layout_table_html(html, "td { font-size: 10px; }");
        let hidden = layout_table_html(
            html,
            "td { font-size: 10px; } tr.t { visibility: collapse; }",
        );
        let t_shown = &shown.children[0].children[0];
        let t_hidden = &hidden.children[0].children[0];
        let row_h = row_at(&shown, 1).rect.h;
        assert!(row_h > 0.0, "row_h={row_h}");
        assert!(
            approx(t_shown.rect.h - t_hidden.rect.h, row_h),
            "shown={} hidden={} row={}",
            t_shown.rect.h,
            t_hidden.rect.h,
            row_h
        );
        assert!(approx(row_at(&hidden, 1).rect.h, 0.0));
        // And back: un-collapsing restores the original height.
        let reshown = layout_table_html(html, "td { font-size: 10px; }");
        let t_reshown = &reshown.children[0].children[0];
        assert!(approx(t_reshown.rect.h, t_shown.rect.h));
    }

    #[test]
    fn visibility_collapse_rowspan_cell_grows_visible_rows_only() {
        // A `rowspan="2"` cell starting in a visible row and spanning a
        // collapsed row fits into the visible row alone: the collapsed
        // row stays zero and the visible row absorbs the cell's height.
        let html = "<table><tr><td rowspan=\"2\" class=\"tall\">x</td></tr>\
                    <tr class=\"gone\"><td>y</td></tr></table>";
        let css = "td { font-size: 10px; } td.tall { font-size: 40px; } \
                   tr.gone { visibility: collapse; }";
        let l = layout_table_html(html, css);
        assert!(approx(row_at(&l, 1).rect.h, 0.0));
        let tall = cell_at(&l, 0, 0);
        assert!(
            approx(row_at(&l, 0).rect.h, tall.rect.h),
            "row0={} tall={}",
            row_at(&l, 0).rect.h,
            tall.rect.h
        );
    }

    #[test]
    fn visibility_collapse_col_cells_size_rows() {
        // WPT `visibility-collapse-col-001`: collapsing a column removes
        // its width, but its cells still size their rows — the collapsed
        // table is exactly as tall as the uncollapsed control, and the
        // row takes the collapsed column's 100px cell height.
        let css = "td { padding: 0; border: 0; } col.gone { visibility: collapse; }";
        let html = "<table><col><col class=\"gone\">\
                    <tr><td><div style=\"width:100px;height:100px\"></div></td>\
                    <td><div style=\"width:50px;height:50px\"></div></td></tr></table>";
        let l = layout_table_html(html, css);
        let control = layout_table_html(html, "td { padding: 0; border: 0; }");
        let table = &l.children[0].children[0];
        let control_table = &control.children[0].children[0];
        assert!(
            approx(table.rect.h, control_table.rect.h),
            "table={} control={}",
            table.rect.h,
            control_table.rect.h
        );
        assert!(
            approx(row_at(&l, 0).rect.h, 100.0),
            "row h={}",
            row_at(&l, 0).rect.h
        );
        assert!(
            approx(table.rect.w, 100.0),
            "collapsed table w={}",
            table.rect.w
        );
    }

    #[test]
    fn visibility_collapse_colspan_cell_ignores_collapsed_col() {
        // WPT `visibility-collapse-colspan-001`: a cell spanning a
        // collapsed column contributes no width — not even to its
        // visible spanned columns. The 30px `colspan=3` cell over the
        // collapsed middle column leaves the table at the 20px of the
        // two 10px columns.
        let css = "td { padding: 0; border: 0; } col.mid { visibility: collapse; }";
        let html = "<table style=\"border-spacing: 0\"><col><col class=\"mid\"><col>\
                    <tr><td><div style=\"width:10px;height:10px\"></div></td>\
                    <td><div style=\"width:10px;height:10px\"></div></td>\
                    <td><div style=\"width:10px;height:10px\"></div></td></tr>\
                    <tr><td colspan=\"3\"><div style=\"width:30px;height:10px\"></div></td></tr>\
                    </table>";
        let l = layout_table_html(html, css);
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 20.0), "table w={}", table.rect.w);
        assert!(approx(table.rect.h, 20.0), "table h={}", table.rect.h);
    }

    #[test]
    fn visibility_collapse_row_share_of_table_height_vanishes() {
        // WPT `visibility-collapse-row-004`: a specified table height
        // distributes over all rows including collapsed ones
        // (css-tables-3 §3.10.5); the collapsed row's share vanishes
        // with it. 150px over 3 rows is 50 each; the visible two sum
        // to 100.
        let css = "td { font-size: 10px; padding: 0; border: 0; } \
                   tr.gone { visibility: collapse; }";
        let html = "<table style=\"height:150px;border-spacing: 0\">\
                    <tr><td>a</td></tr>\
                    <tr class=\"gone\"><td>hello</td></tr>\
                    <tr><td>b</td></tr></table>";
        let l = layout_table_html(html, css);
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.h, 100.0), "table h={}", table.rect.h);
        assert!(approx(row_at(&l, 1).rect.h, 0.0));
    }

    #[test]
    fn visibility_collapse_rowcol_spanning_cell_contributes_nothing() {
        // WPT `visibility-collapse-rowcol-002`: a cell spanning a
        // collapsed row *and* a collapsed column contributes neither
        // width nor height, while a cell fully in a collapsed column
        // still sizes its visible row (10px).
        let css = "td { padding: 0; border: 0; } col.gone { visibility: collapse; }";
        let html = "<table style=\"border-spacing: 0\">\
                    <col><col><col span=\"2\" class=\"gone\">\
                    <tr><td><div style=\"width:25px;height:25px\"></div></td>\
                    <td><div style=\"width:20px;height:20px\"></div></td>\
                    <td><div style=\"width:15px;height:15px\"></div></td>\
                    <td><div style=\"width:10px;height:10px\"></div></td></tr>\
                    <tr><td rowspan=\"2\" colspan=\"3\"><div style=\"width:45px;height:25px\"></div></td>\
                    <td><div style=\"width:10px;height:10px\"></div></td></tr>\
                    <tr style=\"visibility:collapse\"><td><div style=\"width:70px;height:40px\"></div></td></tr>\
                    </table>";
        let l = layout_table_html(html, css);
        let table = &l.children[0].children[0];
        assert!(approx(table.rect.w, 45.0), "table w={}", table.rect.w);
        assert!(approx(table.rect.h, 35.0), "table h={}", table.rect.h);
    }

    #[test]
    fn collapsed_row_content_sizes_columns_like_visible() {
        // A collapsed row's wide content still sizes the visible
        // columns (CSS 2.1 §17.5.5: the suppression of a collapsed row
        // "does not otherwise affect the layout of the table"; WPT
        // `visibility-collapse-row-001` requires the table width to be
        // unchanged by collapsing a row): the column widths match a
        // table laid out with the row visible.
        let css = "td { font-size: 10px; padding: 0; border: 0; } \
                   tr.gone { visibility: collapse; }";
        let with_gone = layout_table_html(
            "<table><tr><td>a</td></tr>\
             <tr class=\"gone\"><td>mmmmmmmmmmmmmmmmmmmmmmmm</td></tr></table>",
            css,
        );
        let plain = layout_table_html(
            "<table><tr><td>a</td></tr>\
             <tr><td>mmmmmmmmmmmmmmmmmmmmmmmm</td></tr></table>",
            css,
        );
        let w1 = cell_at(&with_gone, 0, 0).rect.w;
        let w2 = cell_at(&plain, 0, 0).rect.w;
        assert!(
            approx(w1, w2),
            "collapsed row did not size the column: {w1} vs {w2}"
        );
    }

    #[test]
    fn spanning_cells_exclude_collapsed_line_spacing() {
        // Separate borders: a `rowspan`/`colspan` cell spanning a
        // collapsed row/column counts inter-line spacing for the visible
        // lines only — placement draws no spacing around collapsed
        // lines (mirroring the `rowspan` growth pass).
        let css = "table { border-spacing: 10px; } \
                   td { font-size: 10px; padding: 0; border: 0; } \
                   td.tall { font-size: 40px; } \
                   tr.gone { visibility: collapse; } \
                   col.gone { visibility: collapse; }";
        let l = layout_table_html(
            "<table><col><col class=\"gone\">\
             <tr><td rowspan=\"2\" colspan=\"2\">x</td><td class=\"tall\">y</td></tr>\
             <tr class=\"gone\"><td>z</td></tr></table>",
            css,
        );
        let span = cell_at(&l, 0, 0);
        let tall = cell_at(&l, 0, 1);
        // Colspan: the collapsed column draws no spacing, so the
        // spanning cell's right edge sits exactly one border-spacing
        // left of the next visible column.
        let gap = tall.rect.x - (span.rect.x + span.rect.w);
        assert!(
            approx(gap, 10.0),
            "collapsed column spacing leaked into span width: gap={gap}"
        );
        // Rowspan: the collapsed row draws no spacing, so the stretched
        // cell exactly fills its one visible row.
        assert!(
            approx(span.rect.h, row_at(&l, 0).rect.h),
            "collapsed row spacing leaked into span height: {} vs {}",
            span.rect.h,
            row_at(&l, 0).rect.h
        );
    }

    #[test]
    fn collapse_table_collapsed_last_row_keeps_outer_border() {
        // `border-collapse: collapse`: the cell above a collapsed last
        // row owns the table's outer bottom edge (the collapsed row's
        // own cells never paint), so the shared grid line still draws.
        let l = layout_table_html(
            "<table><tr><td>a</td></tr><tr class=\"gone\"><td>b</td></tr></table>",
            "table { border-collapse: collapse; } \
             td { border: 2px solid red; font-size: 10px; } \
             tr.gone { visibility: collapse; }",
        );
        let e = cell_at(&l, 0, 0)
            .collapsed_borders
            .as_ref()
            .expect("winners");
        assert!(approx(e[2].width, 2.0), "bottom edge must paint: {e:?}");
    }

    // ---- w37 WPT triage: §17.5.2.2 remainder ----
    //
    // Regression tests driven by WPT css/css-tables testharness
    // assertions. (The full `bowser-wpt` run is blocked by pre-existing
    // `bowser_net` breakage at the base commit; these tests pin the same
    // widths through the `layout_table_html` harness.)

    /// WPT `width-distribution/td-min-width-auto-layout.html`: a
    /// `min-width` larger than the cell's `width` pins the column, and
    /// the table's extra width must not leak into the constrained
    /// column (CSS Tables 3 §3.9.3.2: excess goes to non-constrained
    /// columns first).
    #[test]
    fn wpt_td_min_width_pins_constrained_column() {
        let css = "table { width: 400px; border-collapse: collapse; } td { padding: 0; font-size: 10px; }";
        // `width:50px; min-width:150px` -> 150.
        let l = layout_table_html(
            "<table><tr><td>x</td><td style=\"width:50px; min-width:150px\">x</td></tr></table>",
            css,
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 150.0),
            "min-width:150px should override width:50px: {}",
            cell_at(&l, 0, 1).rect.w
        );
        // `width:150px; min-width:50px` -> 150.
        let l = layout_table_html(
            "<table><tr><td>x</td><td style=\"width:150px; min-width:50px\">x</td></tr></table>",
            css,
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 150.0),
            "min-width:50px should not increase width:150px: {}",
            cell_at(&l, 0, 1).rect.w
        );
        // `width:100px; min-width:100px` -> 100.
        let l = layout_table_html(
            "<table><tr><td>x</td><td style=\"width:100px; min-width:100px\">x</td></tr></table>",
            css,
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 100.0),
            "min-width:100px equal to width:100px should not change width: {}",
            cell_at(&l, 0, 1).rect.w
        );
        // `width:200px; min-width:150px; max-width:100px` -> 150:
        // min-width wins over max-width (and max-width never applies to
        // table cells per CSS 2.1 §10.4).
        let l = layout_table_html(
            "<table><tr><td>x</td><td style=\"width:200px; min-width:150px; max-width:100px\">x</td></tr></table>",
            css,
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 150.0),
            "min-width:150px should win over max-width:100px: {}",
            cell_at(&l, 0, 1).rect.w
        );
    }

    /// WPT `width-distribution/distribution-algo-min-content-percent-guess.html`:
    /// a 50% cell in a 400px table gets exactly 200px; the leftover
    /// goes to the unconstrained column.
    #[test]
    fn wpt_percent_cell_gets_exact_percentage_of_excess() {
        let l = layout_table_html(
            "<table><tr><td style=\"width:50%\">x</td><td>y</td></tr></table>",
            "table { width: 400px; border-collapse: collapse; } td { padding: 0; font-size: 10px; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 200.0),
            "50% cell should be 200px: {}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 200.0),
            "unconstrained cell should absorb the rest: {}",
            cell_at(&l, 0, 1).rect.w
        );
    }

    /// WPT `auto-layout-calc-width-001.html`: `calc()` mixing a
    /// percentage with a nonzero length on a `<col>` is treated as
    /// `auto` in auto layout (csswg#94), so the two auto columns split
    /// the 200px table evenly.
    #[test]
    fn wpt_auto_layout_calc_width_on_col_is_auto() {
        let l = layout_table_html(
            "<table><col style=\"width:calc(20% + 80px)\"><tr><td></td><td></td></tr></table>",
            "table { width: 200px; border-collapse: collapse; } td { padding: 0; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 100.0),
            "calc col should behave as auto: {}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 100.0),
            "second auto col should split evenly: {}",
            cell_at(&l, 0, 1).rect.w
        );
    }

    /// WPT `fixed-layout-calc-width-001.html`: the same csswg#94 rule in
    /// fixed layout — the `calc()` column is not pinned, so both
    /// columns share the 200px table equally.
    #[test]
    fn wpt_fixed_layout_calc_width_on_col_is_auto() {
        let l = layout_table_html(
            "<table><col style=\"width:calc(20% + 80px)\"><tr><td></td><td></td></tr></table>",
            "table { width: 200px; border-collapse: collapse; table-layout: fixed; } td { padding: 0; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 100.0),
            "calc col should behave as auto in fixed layout: {}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 100.0),
            "second col should split evenly: {}",
            cell_at(&l, 0, 1).rect.w
        );
    }

    /// A `calc()` that is purely a percentage (`calc(50% + 0px)`)
    /// behaves as that percentage (csswg#94), not as `auto`.
    #[test]
    fn wpt_pure_percentage_calc_on_col_is_percentage() {
        let l = layout_table_html(
            "<table><col style=\"width:calc(50% + 0px)\"><tr><td>x</td><td>y</td></tr></table>",
            "table { width: 200px; border-collapse: collapse; } td { padding: 0; font-size: 10px; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 100.0),
            "calc(50% + 0px) col should be 50%: {}",
            cell_at(&l, 0, 0).rect.w
        );
    }

    /// A `calc()` whose percentage folded to zero (`calc(0% + 80px)`)
    /// syntactically contained a percentage leaf (`had_pct`), so it is
    /// cyclic and behaves as `auto` (CSS Values 4 §8.1.4) — not as a
    /// definite 80px. The column sizes from its content instead of
    /// pinning to 80px.
    #[test]
    fn zero_percent_calc_on_col_is_auto() {
        let l = layout_table_html(
            "<table><col style=\"width:calc(0% + 80px)\"><tr><td>x</td><td>y</td></tr></table>",
            "table { width: 200px; border-collapse: collapse; } td { padding: 0; font-size: 10px; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 100.0),
            "calc(0% + 80px) col should behave as auto: {}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 100.0),
            "second col should split evenly: {}",
            cell_at(&l, 0, 1).rect.w
        );
    }

    /// Regression: `min(50px, 20%)` on a `<col>` defers as the frozen
    /// base-0 form (`MathDeferred { pct: 0.0, px: 0.0, had_pct: true,
    /// affine: false }`), which is value-identical to the affine
    /// pure-percentage fold `calc(0% + 0px)`. It must behave as `auto`,
    /// not as a 0% participant: under fixed layout the two columns
    /// split the 600px table 300/300, not 0/600.
    #[test]
    fn wpt_min_max_math_on_col_is_auto_not_zero_pct() {
        let l = layout_table_html(
            "<table><col style=\"width:min(50px, 20%)\"><tr><td></td><td></td></tr></table>",
            "table { width: 600px; border-collapse: collapse; table-layout: fixed; } td { padding: 0; }",
        );
        assert!(
            approx(cell_at(&l, 0, 0).rect.w, 300.0),
            "min() col should behave as auto in fixed layout: {}",
            cell_at(&l, 0, 0).rect.w
        );
        assert!(
            approx(cell_at(&l, 0, 1).rect.w, 300.0),
            "second col should split evenly: {}",
            cell_at(&l, 0, 1).rect.w
        );
    }

    #[test]
    fn intrinsic_width_on_td_and_col_behaves_as_auto() {
        // F2: `width: min-content` / `max-content` / `fit-content` on
        // `td` and `col` are not specified widths for column sizing
        // (css-sizing-3 §5): they must not read the provisional fill
        // as the column minimum.
        //
        // `max-content`/`fit-content` behave as `auto` here. `min-content`
        // is different and deliberately so: per css-sizing-3 §5.2 a
        // definite preferred size caps the max-content contribution, so
        // `width: min-content` contributes its min-content size (F7) —
        // narrower than `auto`'s max-content contribution for wrappable
        // text — and must not be asserted identical to `auto`.
        for kw in ["max-content", "fit-content"] {
            let html = "<table><col><col><tr><td>aa</td><td>bb cc dd</td></tr></table>";
            let l = layout_table_html(
                html,
                &format!("table {{ width: 500px; }} td, col {{ font-size: 10px; width: {kw}; }}"),
            );
            let a = layout_table_html(html, "table { width: 500px; } td, col { font-size: 10px; }");
            let widths = |l: &crate::LayoutBox| (cell_at(l, 0, 0).rect.w, cell_at(l, 0, 1).rect.w);
            let (w0, w1) = widths(&l);
            let (a0, a1) = widths(&a);
            assert!(approx(w0, a0), "{kw}: col0 width {w0} != auto {a0}");
            assert!(approx(w1, a1), "{kw}: col1 width {w1} != auto {a1}");
            assert!(
                w0 < 500.0,
                "{kw}: col0 minimum must not blow to the table width, got {w0}"
            );
        }
        // `min-content`: definite preferred size caps the contribution
        // (F7) — columns are driven by min-content, never by the fill.
        // Both cells' min-content is ~2ch ("aa" vs "dd"), so both columns
        // have equal min/max contributions and the definite 500px table
        // width splits evenly (css-tables-3 §5.4.2: remaining space in
        // proportion to max-content contributions).
        {
            let html = "<table><col><col><tr><td>aa</td><td>bb cc dd</td></tr></table>";
            let l = layout_table_html(
                html,
                "table { width: 500px; } td, col { font-size: 10px; width: min-content; }",
            );
            let widths = |l: &crate::LayoutBox| (cell_at(l, 0, 0).rect.w, cell_at(l, 0, 1).rect.w);
            let (w0, w1) = widths(&l);
            assert!(
                approx(w0, 250.0) && approx(w1, 250.0),
                "min-content: equal min-content contributions must split the 500px table evenly, got ({w0}, {w1})"
            );
        }
    }

    #[test]
    fn table_col_spec_gates_on_had_pct_not_folded_coefficient() {
        // `calc(80px + 0%)` folds to `Math { pct: 0.0, px: 80.0 }` but
        // syntactically contained a percentage leaf (`had_pct`), so it
        // is cyclic and behaves as `auto` — not as a definite `80px`
        // (CSS Values 4 §8.1.4, symmetric with every other intrinsic
        // pass).
        let folded_zero = SpecLen::Math {
            pct: 0.0,
            px: 80.0,
            had_pct: true,
            affine: true,
        };
        assert!(
            matches!(table_col_spec(folded_zero), SpecLen::Auto),
            "calc(80px + 0%) must be auto"
        );
        // The csswg#94 pure-percentage nuance survives: `calc(50% + 0px)`
        // behaves as that percentage.
        let pure_pct = SpecLen::Math {
            pct: 50.0,
            px: 0.0,
            had_pct: true,
            affine: true,
        };
        assert!(matches!(table_col_spec(pure_pct), SpecLen::Pct(_)));
        // Mixed percentage+length is `auto` (csswg#94).
        let mixed = SpecLen::Math {
            pct: 20.0,
            px: 80.0,
            had_pct: true,
            affine: true,
        };
        assert!(matches!(table_col_spec(mixed), SpecLen::Auto));
        // No percentage leaf at all: a definite length.
        let plain = SpecLen::Math {
            pct: 0.0,
            px: 80.0,
            had_pct: false,
            affine: true,
        };
        assert!(matches!(table_col_spec(plain), SpecLen::Px(_)));
        // Frozen base-0 form of non-affine percentage math
        // (`min(50px, 20%)`): value-identical to `calc(0% + 0px)` but
        // `affine: false`, so it is `auto`, not a 0% participant.
        let frozen = SpecLen::Math {
            pct: 0.0,
            px: 0.0,
            had_pct: true,
            affine: false,
        };
        assert!(
            matches!(table_col_spec(frozen), SpecLen::Auto),
            "frozen non-affine math must be auto"
        );
    }

    /// WPT `width-distribution/computing-table-width-0.html` (text
    /// caption variant): CAPMIN bounds the table's border-box width —
    /// a 50px table with 2px borders and a 180px caption ends up 180px
    /// wide, not 184px.
    ///
    /// (The literal WPT test uses an inline-block caption child, which
    /// needs `min_content_width` to honor specified widths on
    /// inline-blocks — a `layout.rs` gap, out of scope here.)
    #[test]
    fn wpt_caption_min_width_bounds_border_box() {
        let l = layout_table_html(
            &format!(
                "<table><caption>{}</caption><tr><td>hi</td></tr></table>",
                "a".repeat(30)
            ),
            "table { width: 50px; border: 2px solid black; } td, caption { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 180.0),
            "CAPMIN should bound the border box: {}",
            table.rect.w
        );
    }

    /// CSS 2.1 §9.5.1: a table float's margin-box (outer) top may not sit
    /// above the table's content top. Regression: the resume pass seeded
    /// `fy = float_y_start - mt`, treating the margin-box start as a
    /// border-box top, so a `float:left;margin-top:20px` table child ended
    /// with its margin-box top 20px *above* the content top. Block flow
    /// (`resume_block`) instead treats the start as the margin-box top
    /// (`pending_float_y = cursor_y`, clamped by `fy.max(fr.content_top)`).
    #[test]
    fn table_float_margin_box_top_never_above_content_top() {
        let (doc, _) = bowser_dom::Document::parse_html(
            "<div class=\"t\"><div class=\"f\">x</div></div>".as_bytes(),
        );
        let ss = bowser_css::parse_stylesheet(
            ".t { display: table; } .f { float: left; margin-top: 20px; width: 50px; }",
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
        crate::layout::layout_forest(&mut forest, root, &FixedFontMetrics);
        let table = (0..forest.nodes.len())
            .find(|&i| forest.nodes[i].box_type == crate::BoxType::Table)
            .expect("table box");
        let table_top = forest.nodes[table].rect.y;
        let f = forest.nodes[table]
            .children
            .iter()
            .copied()
            .find(|&c| !matches!(forest.nodes[c].data.float, crate::style::Float::None))
            .expect("float child of table");
        // No borders, padding, or captions: content top == table border top.
        let margin_box_top = forest.nodes[f].rect.y - forest.nodes[f].metrics.mt;
        assert!(
            approx(margin_box_top, table_top),
            "float margin-box top {} must sit at the table content top {}",
            margin_box_top,
            table_top
        );
        assert!(
            approx(forest.nodes[f].rect.y, table_top + 20.0),
            "border box sits margin-top below content top: {}",
            forest.nodes[f].rect.y
        );
    }

    #[test]
    fn table_float_clear_offsets_by_top_margin() {
        // CSS 2.1 §9.5.2: clearance puts the clearing element's top
        // *border* edge at the earlier float's bottom margin edge, so
        // the margin-box top is the float bottom *minus the clearing
        // float's own top margin* — `cy - mt`, mirroring `resume_block`.
        // Before the fix the table path used `cy`, landing the float
        // 20px too low.
        let (doc, _) = bowser_dom::Document::parse_html(
            "<div class=\"t\"><div class=\"a\">a</div><div class=\"b\">b</div></div>".as_bytes(),
        );
        let ss = bowser_css::parse_stylesheet(
            ".t { display: table; } \
             .a { float: left; width: 50px; height: 30px; } \
             .b { float: left; clear: left; margin-top: 20px; width: 50px; height: 10px; }",
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
        crate::layout::layout_forest(&mut forest, root, &FixedFontMetrics);
        let table = (0..forest.nodes.len())
            .find(|&i| forest.nodes[i].box_type == crate::BoxType::Table)
            .expect("table box");
        let table_top = forest.nodes[table].rect.y;
        let mut floats: Vec<usize> = forest.nodes[table]
            .children
            .iter()
            .copied()
            .filter(|&c| !matches!(forest.nodes[c].data.float, crate::style::Float::None))
            .collect();
        floats.sort_by(|&a, &b| {
            forest.nodes[a]
                .rect
                .y
                .partial_cmp(&forest.nodes[b].rect.y)
                .unwrap()
        });
        assert_eq!(floats.len(), 2);
        let (a, b) = (floats[0], floats[1]);
        let a_bottom = forest.nodes[a].rect.y + forest.nodes[a].rect.h + forest.nodes[a].metrics.mb;
        assert!(
            approx(a_bottom, table_top + 30.0),
            "first float bottom margin edge {} (table top {})",
            a_bottom,
            table_top
        );
        let b_margin_top = forest.nodes[b].rect.y - forest.nodes[b].metrics.mt;
        assert!(
            approx(b_margin_top, a_bottom),
            "clearing float margin-box top {} must equal float bottom {} (not 20px below)",
            b_margin_top,
            a_bottom
        );
    }

    #[test]
    fn floated_table_shrink_to_fit_covers_float_child() {
        // CSS 2.1 §9.4.1 (a BFC contains its floats) plus shrink-to-fit
        // sizing per CSS 2.1 §10.3.5/§10.3.7: a shrink-to-fit (floated)
        // table with a float child wider than its columns sizes to the
        // float's margin box, not just the grid. Before the fix the
        // preferred width ignored floats, so the 200px float overflowed
        // a 6px table.
        let l = layout_table_html(
            "<div class=\"t\"><div class=\"r\"><div class=\"c\">x</div></div>\
             <div class=\"f\">y</div></div>",
            ".t { display: table; float: left; } .r { display: table-row; } \
             .c { display: table-cell; font-size: 10px; } \
             .f { float: left; width: 200px; height: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        assert!(
            approx(table.rect.w, 200.0),
            "table should cover its float child: {}",
            table.rect.w
        );
    }

    #[test]
    fn floated_table_shrink_to_fit_covers_auto_width_float_child() {
        // Same with an auto-width float: its shrink-to-fit (preferred)
        // width is the floor, not just the grid.
        let l = layout_table_html(
            "<div class=\"t\"><div class=\"r\"><div class=\"c\">x</div></div>\
             <div class=\"f\">longer float text</div></div>",
            ".t { display: table; float: left; } .r { display: table-row; } \
             .c { display: table-cell; font-size: 10px; } \
             .f { float: left; font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        // "longer float text" = 17 chars × 6px = 102px > 6px column.
        assert!(
            approx(table.rect.w, 102.0),
            "table should cover its auto-width float child: {}",
            table.rect.w
        );
    }

    #[test]
    fn float_only_table_shrink_to_fit_covers_float() {
        // A shrink-to-fit table whose only child is a float has no grid
        // columns; the float's margin box still floors the preferred
        // width. Before the fix the `ncols == 0` early return skipped the
        // float floor and the table sized to 0.
        let l = layout_table_html(
            "<div class=\"t\"><div class=\"f\">y</div></div>",
            ".t { display: table; float: left; } \
             .f { float: left; width: 200px; height: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        assert!(
            approx(table.rect.w, 200.0),
            "table should cover its float child: {}",
            table.rect.w
        );
    }

    // ---- table geometry measurement API (`offsetWidth` and friends) ----

    /// Navigate a laid-out table and return the ancestor chain plus the
    /// first cell: root → body → table → row group → row, and the cell.
    fn cell_with_ancestors(
        l: &crate::LayoutBox,
    ) -> (
        Vec<&crate::LayoutBox>,
        &crate::LayoutBox,
        &crate::LayoutBox,
        &crate::LayoutBox,
        &crate::LayoutBox,
        &crate::LayoutBox,
    ) {
        let body = &l.children[0];
        let table = &body.children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        let group = table
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::TableRowGroup)
            .expect("row group");
        let row = &group.children[0];
        let cell = &row.children[0];
        (
            vec![l, body, table, group, row],
            body,
            table,
            group,
            row,
            cell,
        )
    }

    #[test]
    fn table_geometry_cell_boxes() {
        // offsetWidth/offsetHeight read the border box; clientWidth/
        // clientHeight read the padding box (border box minus the 2px
        // used borders; the 5px padding stays inside).
        let (doc, l) = layout_table_doc(
            "<table><tr><td>hi</td></tr></table>",
            "td { font-size: 10px; border: 2px solid black; padding: 5px; }",
        );
        let (is_body, is_table_part) = geom_preds(&doc);
        let (ancestors, _, _, _, _, cell) = cell_with_ancestors(&l);
        let g = table_box_geometry(cell, &ancestors, is_body, is_table_part);
        assert!(approx(g.offset_width, cell.rect.w));
        assert!(approx(g.offset_height, cell.rect.h));
        assert!(approx(g.client_width, cell.rect.w - 4.0));
        assert!(approx(g.client_height, cell.rect.h - 4.0));
    }

    #[test]
    fn table_geometry_collapsed_borders_use_half_winners() {
        // Under `border-collapse: collapse` the used border on each edge
        // is half the winning width (winners center on the grid lines):
        // a 5px winning border contributes 2.5px per side.
        let (doc, l) = layout_table_doc(
            "<table><tr><td>x</td></tr></table>",
            "table { border-collapse: collapse; } td { font-size: 10px; border: 5px solid red; }",
        );
        let (is_body, is_table_part) = geom_preds(&doc);
        let (ancestors, _, _, _, _, cell) = cell_with_ancestors(&l);
        assert!(cell.collapsed_borders.is_some());
        let g = table_box_geometry(cell, &ancestors, is_body, is_table_part);
        assert!(approx(g.offset_width, cell.rect.w));
        assert!(approx(g.client_width, cell.rect.w - 5.0));
        assert!(approx(g.client_height, cell.rect.h - 5.0));
    }

    #[test]
    fn table_geometry_row_has_no_used_borders() {
        // Layout never resolves borders onto row rects, so a row's client
        // box equals its border box even with a specified row border.
        let (doc, l) = layout_table_doc(
            "<table><tr><td>x</td></tr></table>",
            "tr { border: 7px solid black; } td { font-size: 10px; }",
        );
        let (is_body, is_table_part) = geom_preds(&doc);
        let (ancestors, _, _, _, row, _) = cell_with_ancestors(&l);
        let g = table_box_geometry(row, &ancestors[..4], is_body, is_table_part);
        assert!(approx(g.offset_width, row.rect.w));
        assert!(approx(g.client_width, row.rect.w));
        assert!(approx(g.client_height, row.rect.h));
    }

    #[test]
    fn table_offset_parent_table_part_beats_positioned_ancestor() {
        // The relatively-positioned wrapper (3px border) is the *table's*
        // offset parent: the table's offsetLeft measures from the table's
        // border edge to the wrapper's padding edge. But the static `td`
        // cell gets the static `table` itself per the CSSOM View
        // td/th/table rule — the positioned wrapper further up does not
        // win over the nearer table part.
        let (doc, l) = layout_table_doc(
            "<div class=\"wrap\"><table><tr><td>hi</td></tr></table></div>",
            ".wrap { position: relative; border: 3px solid black; } td { font-size: 10px; }",
        );
        let body = &l.children[0];
        let wrap = &body.children[0];
        let table = &wrap.children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        let group = table
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::TableRowGroup)
            .expect("row group");
        let row = &group.children[0];
        let cell = &row.children[0];
        let ancestors = vec![&l, body, wrap, table, group, row];
        let (is_body, is_table_part) = geom_preds(&doc);
        let op =
            table_offset_parent(cell, &ancestors, is_body, is_table_part).expect("offset parent");
        assert!(std::ptr::eq(op, table));
        let (is_body, is_table_part) = geom_preds(&doc);
        let g = table_box_geometry(cell, &ancestors, is_body, is_table_part);
        assert!(approx(g.offset_left, cell.rect.x - table.rect.x));
        assert!(approx(g.offset_top, cell.rect.y - table.rect.y));
        // The table itself has no nearer table part: the positioned
        // wrapper wins for it.
        let (is_body, is_table_part) = geom_preds(&doc);
        let top = table_offset_parent(table, &ancestors[..3], is_body, is_table_part)
            .expect("offset parent");
        assert!(std::ptr::eq(top, wrap));
        let (is_body, is_table_part) = geom_preds(&doc);
        let gt = table_box_geometry(table, &ancestors[..3], is_body, is_table_part);
        assert!(approx(gt.offset_left, table.rect.x - (wrap.rect.x + 3.0)));
        assert!(approx(gt.offset_top, table.rect.y - (wrap.rect.y + 3.0)));
    }

    #[test]
    fn table_offset_parent_static_falls_back_to_table_then_body() {
        // CSSOM View: a static element with no positioned ancestor gets
        // the nearest `td`/`th`/`table` ancestor (here the `table`
        // itself), and an element with no table part at all gets
        // `document.body` — never null. The body's own offset parent is
        // null.
        let (doc, l) = layout_table_doc(
            "<table><tr><td>hi</td></tr></table>",
            "td { font-size: 10px; }",
        );
        let (ancestors, body, table, _, _, cell) = cell_with_ancestors(&l);
        let (is_body, is_table_part) = geom_preds(&doc);
        let op =
            table_offset_parent(cell, &ancestors, is_body, is_table_part).expect("offset parent");
        assert!(std::ptr::eq(op, table));
        let (is_body, is_table_part) = geom_preds(&doc);
        let g = table_box_geometry(cell, &ancestors, is_body, is_table_part);
        assert!(approx(g.offset_left, cell.rect.x - table.rect.x));
        assert!(approx(g.offset_top, cell.rect.y - table.rect.y));
        // The table itself has no table-part ancestor: its offset parent
        // is the body box.
        let (is_body, is_table_part) = geom_preds(&doc);
        let top = table_offset_parent(table, &ancestors[..2], is_body, is_table_part)
            .expect("offset parent");
        assert!(std::ptr::eq(top, body));
        let (is_body, is_table_part) = geom_preds(&doc);
        assert!(table_offset_parent(body, &ancestors[..1], is_body, is_table_part).is_none());
    }

    #[test]
    fn table_offset_parent_body_null_with_positioned_html() {
        // CSSOM View step 1: `document.body.offsetParent` is null even
        // when `<html>` is positioned — and a static element under the
        // positioned `<html>` still finds `body` first, not `<html>`.
        let (doc, l) = layout_table_doc(
            "<table><tr><td>hi</td></tr></table>",
            "html { position: relative; } td { font-size: 10px; }",
        );
        let (is_body, is_table_part) = geom_preds(&doc);
        let (ancestors, body, table, _, _, _) = cell_with_ancestors(&l);
        assert!(table_offset_parent(body, &ancestors[..1], is_body, is_table_part).is_none());
        let (is_body, is_table_part) = geom_preds(&doc);
        let top = table_offset_parent(table, &ancestors[..2], is_body, is_table_part)
            .expect("offset parent");
        assert!(std::ptr::eq(top, body));
    }

    #[test]
    fn table_offset_parent_static_div_in_td_gets_td() {
        // A static `div` inside a static `td`: the `td` is the offset
        // parent (CSSOM View step 2.2). The rule reads the HTML element,
        // not the box type: a `div { display: table-cell }` does *not*
        // qualify, so the inner div there falls through to the `table`.
        let (doc, l) = layout_table_doc(
            "<table><tr><td><div class=\"in\">hi</div></td></tr></table>",
            "td { font-size: 10px; } .in { width: 50px; height: 20px; }",
        );
        let (is_body, is_table_part) = geom_preds(&doc);
        let (ancestors, _, _, _, _, cell) = cell_with_ancestors(&l);
        let inner = cell
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::Block)
            .expect("inner div box");
        let mut inner_ancestors = ancestors.clone();
        inner_ancestors.push(cell);
        let op = table_offset_parent(inner, &inner_ancestors, is_body, is_table_part)
            .expect("offset parent");
        assert!(std::ptr::eq(op, cell));
        let (is_body, is_table_part) = geom_preds(&doc);
        let g = table_box_geometry(inner, &inner_ancestors, is_body, is_table_part);
        assert!(approx(g.offset_left, inner.rect.x - cell.rect.x));
        assert!(approx(g.offset_top, inner.rect.y - cell.rect.y));

        // Same shape, but the cell box comes from a `div` with
        // `display: table-cell`: the parser foster-parents it out of the
        // `tr`, so it is a `TableCell` box but not an HTML `td` — and the
        // inner div's offset parent is `document.body`, not the fake cell.
        let (doc2, l2) = layout_table_doc(
            "<table><tr><div class=\"fake\"><div class=\"in\">hi</div></div></tr></table>",
            ".fake { display: table-cell; font-size: 10px; } .in { width: 50px; height: 20px; }",
        );
        let (is_body2, is_table_part2) = geom_preds(&doc2);
        let body2 = &l2.children[0];
        let fake = &body2.children[0];
        assert_eq!(fake.box_type, crate::BoxType::TableCell);
        let inner2 = fake
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::Block)
            .expect("inner div box");
        let ancestors2 = vec![&l2, body2, fake];
        let op2 = table_offset_parent(inner2, &ancestors2, is_body2, is_table_part2)
            .expect("offset parent");
        assert!(std::ptr::eq(op2, body2));
    }

    #[test]
    fn table_geometry_op_borders_use_op_font_context() {
        // The offset parent's used borders resolve relative units against
        // the *offset parent's own* ancestor chain, not the node's: the
        // positioned wrapper has `font-size: 2em` (20px over the 10px
        // body) so its `0.5em` border is 10px — not 20px as the node's
        // full chain would compute (the node's 40px font size leaking
        // into the wrapper's `em`).
        let (doc, l) = layout_table_doc(
            "<div class=\"wrap\"><table><tr><td>hi</td></tr></table></div>",
            "body { font-size: 10px; } \
             .wrap { position: relative; font-size: 2em; border: 0.5em solid black; } \
             table { font-size: 40px; } td { font-size: 10px; }",
        );
        let (is_body, is_table_part) = geom_preds(&doc);
        let body = &l.children[0];
        let wrap = &body.children[0];
        let table = &wrap.children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        let ancestors = vec![&l, body, wrap];
        let op =
            table_offset_parent(table, &ancestors, is_body, is_table_part).expect("offset parent");
        assert!(std::ptr::eq(op, wrap));
        let (is_body, is_table_part) = geom_preds(&doc);
        let g = table_box_geometry(table, &ancestors, is_body, is_table_part);
        assert!(approx(g.offset_left, table.rect.x - (wrap.rect.x + 10.0)));
        assert!(approx(g.offset_top, table.rect.y - (wrap.rect.y + 10.0)));
    }

    #[test]
    fn table_offset_parent_none_when_fixed() {
        // `position: fixed` forces a null offsetParent (CSSOM View), with
        // ICB-relative offsets.
        let (doc, l) = layout_table_doc(
            "<table><tr><td>hi</td></tr></table>",
            "table { position: fixed; left: 15px; top: 25px; } td { font-size: 10px; }",
        );
        let (is_body, is_table_part) = geom_preds(&doc);
        let body = &l.children[0];
        let table = &body.children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        let ancestors = vec![&l, body];
        assert!(table_offset_parent(table, &ancestors, is_body, is_table_part).is_none());
        let (is_body, is_table_part) = geom_preds(&doc);
        let g = table_box_geometry(table, &ancestors, is_body, is_table_part);
        assert!(approx(g.offset_left, table.rect.x));
        assert!(approx(g.offset_top, table.rect.y));
    }

    /// Navigate a laid-out `.wrap > table > row-group > row > cell` tree
    /// down to a fixed-position div inside the cell, returning the fixed
    /// box plus its ancestor chain (for [`table_offset_parent`]).
    fn fixed_div_in_cell_with_ancestors(
        l: &crate::LayoutBox,
    ) -> (Vec<&crate::LayoutBox>, &crate::LayoutBox) {
        let body = &l.children[0];
        let wrap = &body.children[0];
        let table = &wrap.children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        let group = table
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::TableRowGroup)
            .expect("row group");
        let row = &group.children[0];
        let cell = &row.children[0];
        let fix = cell
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::Block)
            .expect("fixed div box");
        (vec![l, body, wrap, table, group, row, cell], fix)
    }

    #[test]
    fn table_offset_parent_fixed_inside_transformed_ancestor() {
        // CSS Position 3 §2.1 / CSS Transforms 1 §3: a non-`none`
        // `transform` on an ancestor makes it the containing block for
        // fixed-position descendants, and `offsetParent` follows the
        // containing block — the fixed div inside the `td` reports the
        // transformed wrapper, not the cell or the body.
        let (doc, l) = layout_table_doc(
            "<div class=\"wrap\"><table><tr><td><div class=\"fix\">hi</div></td></tr></table></div>",
            ".wrap { transform: translateX(0); } td { font-size: 10px; } \
             .fix { position: fixed; width: 20px; height: 10px; }",
        );
        let (is_body, is_table_part) = geom_preds(&doc);
        let (ancestors, fix) = fixed_div_in_cell_with_ancestors(&l);
        let wrap = ancestors[2];
        let op =
            table_offset_parent(fix, &ancestors, is_body, is_table_part).expect("offset parent");
        assert!(std::ptr::eq(op, wrap));
        // Geometry measures against the transformed wrapper's padding
        // box, matching layout's fixed-containing-block placement.
        let (is_body, is_table_part) = geom_preds(&doc);
        let g = table_box_geometry(fix, &ancestors, is_body, is_table_part);
        let op_b = wrap.rect;
        assert!(approx(g.offset_left, fix.rect.x - op_b.x));
        assert!(approx(g.offset_top, fix.rect.y - op_b.y));
    }

    #[test]
    fn table_offset_parent_fixed_inside_filtered_ancestor() {
        // A non-`none` `filter` likewise establishes a fixed containing
        // block (Filter Effects 1 §8): the fixed div's `offsetParent`
        // is the filtered wrapper. This covers the CSS-side branch of
        // the shared fixed-CB predicate (the transform test covers the
        // layout-side `transform` branch).
        let (doc, l) = layout_table_doc(
            "<div class=\"wrap\"><table><tr><td><div class=\"fix\">hi</div></td></tr></table></div>",
            ".wrap { filter: blur(5px); } td { font-size: 10px; } \
             .fix { position: fixed; width: 20px; height: 10px; }",
        );
        let (is_body, is_table_part) = geom_preds(&doc);
        let (ancestors, fix) = fixed_div_in_cell_with_ancestors(&l);
        let wrap = ancestors[2];
        let op =
            table_offset_parent(fix, &ancestors, is_body, is_table_part).expect("offset parent");
        assert!(std::ptr::eq(op, wrap));
    }

    #[test]
    fn table_offset_parent_fixed_control_positioned_only() {
        // Control: with no fixed-CB trigger on any ancestor, the fixed
        // div's `offsetParent` stays null (CSSOM View step 3) — a merely
        // positioned wrapper does not establish a fixed containing
        // block, and neither does the `td` (which would only qualify
        // for a *static* descendant).
        let (doc, l) = layout_table_doc(
            "<div class=\"wrap\"><table><tr><td><div class=\"fix\">hi</div></td></tr></table></div>",
            ".wrap { position: relative; } td { font-size: 10px; } \
             .fix { position: fixed; width: 20px; height: 10px; }",
        );
        let (is_body, is_table_part) = geom_preds(&doc);
        let (ancestors, fix) = fixed_div_in_cell_with_ancestors(&l);
        assert!(table_offset_parent(fix, &ancestors, is_body, is_table_part).is_none());
        // Geometry falls back to ICB-relative offsets.
        let (is_body, is_table_part) = geom_preds(&doc);
        let g = table_box_geometry(fix, &ancestors, is_body, is_table_part);
        assert!(approx(g.offset_left, fix.rect.x));
        assert!(approx(g.offset_top, fix.rect.y));
    }

    #[test]
    fn table_geometry_display_none_measures_zero() {
        // A box with no associated CSS layout box (`display: none`)
        // measures all zeros with a null offset parent.
        let (doc, _) = bowser_dom::Document::parse_html(b"<div></div>");
        let root = doc.document_element().expect("document element");
        let div = doc
            .children(root)
            .into_iter()
            .flat_map(|c| doc.children(c))
            .find(|&c| {
                matches!(&doc.kind(c), bowser_dom::NodeKind::Element(d) if d.tag_name == "div")
            })
            .expect("div");
        let ss = bowser_css::parse_stylesheet("div { display: none; }");
        let ua = bowser_css::default_ua_stylesheet();
        let style = bowser_css::compute_style(
            &doc,
            div,
            &[&ss],
            &ua,
            &bowser_css::Viewport { w: 800.0, h: 600.0 },
        );
        let b = crate::LayoutBox {
            rect: crate::Rect {
                x: 10.0,
                y: 20.0,
                w: 100.0,
                h: 50.0,
            },
            children: Vec::new(),
            box_type: crate::BoxType::TableCell,
            style: Some(std::rc::Rc::new(style)),
            text: None,
            node: None,
            fragments: Vec::new(),
            z_index: None,
            collapsed_borders: None,
            phantom_w: false,
            phantom_h: false,
        };
        let g = table_box_geometry(&b, &[], |_| false, |_| false);
        assert_eq!(
            g,
            TableBoxGeometry {
                offset_width: 0.0,
                offset_height: 0.0,
                client_width: 0.0,
                client_height: 0.0,
                offset_left: 0.0,
                offset_top: 0.0,
            }
        );
        assert!(table_offset_parent(&b, &[], |_| false, |_| false).is_none());
    }

    // ---- intrinsic-measurement allocation pooling (wave 56 P1) ----

    /// Thread-filtered allocation counter: a `#[global_allocator]`
    /// wrapper around `System` that counts `alloc`/`realloc` calls only
    /// while armed on the arming thread (a `thread_local!` flag and a
    /// `thread_local!` count, so the test harness's parallel test threads
    /// never interfere with each other), letting the per-cell intrinsic
    /// measurement loops below prove they no longer allocate per call.
    struct CountingAlloc;
    thread_local! {
        static COUNTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        static COUNT_ALLOCS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    #[inline(always)]
    fn note_alloc() {
        COUNTING.with(|c| {
            if c.get() {
                COUNT_ALLOCS.with(|n| n.set(n.get().wrapping_add(1)));
            }
        });
    }

    unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
            note_alloc();
            unsafe { std::alloc::System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(
            &self,
            ptr: *mut u8,
            layout: std::alloc::Layout,
            new_size: usize,
        ) -> *mut u8 {
            note_alloc();
            unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static COUNTING_GLOBAL_ALLOC: CountingAlloc = CountingAlloc;

    /// Disarms `COUNTING` on drop, so a panic inside the measured closure
    /// can't leave the armed flag set on a reused harness thread.
    struct DisarmOnDrop;
    impl Drop for DisarmOnDrop {
        fn drop(&mut self) {
            COUNTING.with(|c| c.set(false));
        }
    }

    /// Runs `f` with allocation counting armed on this thread; returns
    /// the number of allocating calls it made.
    fn count_allocations(f: impl FnOnce()) -> u64 {
        COUNT_ALLOCS.with(|n| n.set(0));
        COUNTING.with(|c| c.set(true));
        let guard = DisarmOnDrop;
        f();
        drop(guard);
        COUNT_ALLOCS.with(|n| n.get())
    }

    /// A box-tree forest containing one table with `rows * cols`
    /// single-column text cells; returns the forest and the cell node
    /// indices.
    fn intrinsic_test_table(rows: usize, cols: usize) -> (crate::boxes::Forest, Vec<usize>) {
        let mut html = String::from("<table>");
        for r in 0..rows {
            html.push_str("<tr>");
            for c in 0..cols {
                html.push_str(&format!("<td>cell {r} {c} with some words</td>"));
            }
            html.push_str("</tr>");
        }
        html.push_str("</table>");
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet("");
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
        let mut stack = vec![root];
        let mut table = None;
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == crate::BoxType::Table {
                table = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        let grid = collect_table_grid(&forest, table.expect("table box"));
        let cells: Vec<usize> = grid.cells.iter().map(|c| c.node).collect();
        assert_eq!(cells.len(), rows * cols);
        (forest, cells)
    }

    #[test]
    fn pooled_preferred_width_allocates_no_more_than_fresh() {
        use super::super::WalkBufs;
        use super::super::preferred_width_mcw;
        let (mut forest, cells) = intrinsic_test_table(4, 16);
        let fonts = FixedFontMetrics;

        // Correctness first: pooled and fresh must agree exactly.
        let mut pooled_widths = Vec::with_capacity(cells.len());
        {
            let mut cache = PrefWidthCache::new(forest.nodes.len());
            let mut mcw = Vec::new();
            let mut bufs = WalkBufs::new();
            let mut inner = WalkBufs::new();
            for &cell in &cells {
                pooled_widths.push(preferred_width_mcw(
                    &mut forest,
                    cell,
                    &fonts,
                    &mut cache,
                    &mut mcw,
                    &mut bufs,
                    &mut inner,
                ));
            }
        }
        let mut fresh_widths = Vec::with_capacity(cells.len());
        {
            let mut cache = PrefWidthCache::new(forest.nodes.len());
            for &cell in &cells {
                fresh_widths.push(super::super::preferred_width(
                    &mut forest,
                    cell,
                    &fonts,
                    &mut cache,
                ));
            }
        }
        assert_eq!(pooled_widths, fresh_widths);

        // Hermetic measurement: the counting window is global to the
        // arming thread, so any lazy one-time init (TLS first-touch, std
        // internals) that an earlier harness test left untriggered on
        // this thread would pollute the counts and flip the strict `<`.
        // Measure on a fresh thread instead (`Forest` is not `Send`, so
        // the forest is rebuilt there), and dry-run both closures
        // uncounted before arming to flush this thread's lazy state out
        // of the window. The `#[global_allocator]` machinery is shared;
        // only this test's measurement moves.
        let (pooled, fresh) = std::thread::spawn(|| {
            let (mut forest, cells) = intrinsic_test_table(4, 16);
            let fonts = FixedFontMetrics;
            // Warm the pooled buffers, then count a full pass with a fresh
            // cache (no cache hits: every cell walks its subtree). The
            // warm pass doubles as the pooled dry-run on this thread.
            let mut mcw = Vec::new();
            let mut bufs = WalkBufs::new();
            let mut inner = WalkBufs::new();
            {
                let mut cache = PrefWidthCache::new(forest.nodes.len());
                for &cell in &cells {
                    preferred_width_mcw(
                        &mut forest,
                        cell,
                        &fonts,
                        &mut cache,
                        &mut mcw,
                        &mut bufs,
                        &mut inner,
                    );
                }
            }
            // Dry-run the fresh closure uncounted.
            {
                let mut cache = PrefWidthCache::new(forest.nodes.len());
                for &cell in &cells {
                    super::super::preferred_width(&mut forest, cell, &fonts, &mut cache);
                }
            }
            let pooled = count_allocations(|| {
                let mut cache = PrefWidthCache::new(forest.nodes.len());
                for &cell in &cells {
                    preferred_width_mcw(
                        &mut forest,
                        cell,
                        &fonts,
                        &mut cache,
                        &mut mcw,
                        &mut bufs,
                        &mut inner,
                    );
                }
            });
            let fresh = count_allocations(|| {
                let mut cache = PrefWidthCache::new(forest.nodes.len());
                for &cell in &cells {
                    super::super::preferred_width(&mut forest, cell, &fonts, &mut cache);
                }
            });
            (pooled, fresh)
        })
        .join()
        .expect("allocation-measurement thread panicked");
        // The wrapper now pools via a thread-local `PrefWidthScratch`
        // (wave 66), so it must allocate no more than the hand-pooled
        // call above — the single counted alloc on both sides is the
        // per-pass `PrefWidthCache`.
        assert!(
            fresh <= pooled,
            "pooled preferred_width ({fresh} allocs) should allocate no more \
             than hand-pooled preferred_width_mcw ({pooled} allocs) over {} cells",
            cells.len()
        );
    }

    #[test]
    fn pooled_min_content_width_allocates_less_than_fresh() {
        use super::super::WalkBufs;
        use super::super::min_content_width_pooled;
        let (mut forest, cells) = intrinsic_test_table(4, 16);
        let fonts = FixedFontMetrics;

        // Correctness first: pooled and fresh must agree exactly. Both
        // share the `mcw` scratch (the established pattern); only the
        // walk buffers differ.
        let mut mcw = Vec::new();
        let mut pooled_widths = Vec::with_capacity(cells.len());
        {
            let mut bufs = WalkBufs::new();
            for &cell in &cells {
                pooled_widths.push(min_content_width_pooled(
                    &mut forest,
                    cell,
                    &fonts,
                    &mut mcw,
                    &mut bufs,
                ));
            }
        }
        let mut fresh_widths = Vec::with_capacity(cells.len());
        for &cell in &cells {
            fresh_widths.push(super::super::min_content_width(
                &mut forest,
                cell,
                &fonts,
                &mut mcw,
            ));
        }
        assert_eq!(pooled_widths, fresh_widths);

        // Hermetic measurement: the counting window is global to the
        // arming thread, so any lazy one-time init (TLS first-touch, std
        // internals) that an earlier harness test left untriggered on
        // this thread would pollute the counts and flip the strict `<`.
        // Measure on a fresh thread instead (`Forest` is not `Send`, so
        // the forest is rebuilt there), and dry-run both closures
        // uncounted before arming to flush this thread's lazy state out
        // of the window. The `#[global_allocator]` machinery is shared;
        // only this test's measurement moves.
        let (pooled, fresh) = std::thread::spawn(|| {
            let (mut forest, cells) = intrinsic_test_table(4, 16);
            let fonts = FixedFontMetrics;
            let mut mcw = Vec::new();
            // Warm the pooled buffers; doubles as the pooled dry-run on
            // this thread.
            let mut bufs = WalkBufs::new();
            for &cell in &cells {
                min_content_width_pooled(&mut forest, cell, &fonts, &mut mcw, &mut bufs);
            }
            // Dry-run the fresh closure uncounted.
            for &cell in &cells {
                super::super::min_content_width(&mut forest, cell, &fonts, &mut mcw);
            }
            let pooled = count_allocations(|| {
                for &cell in &cells {
                    min_content_width_pooled(&mut forest, cell, &fonts, &mut mcw, &mut bufs);
                }
            });
            let fresh = count_allocations(|| {
                for &cell in &cells {
                    super::super::min_content_width(&mut forest, cell, &fonts, &mut mcw);
                }
            });
            (pooled, fresh)
        })
        .join()
        .expect("allocation-measurement thread panicked");
        assert!(
            pooled < fresh,
            "pooled min_content_width_pooled ({pooled} allocs) should allocate strictly fewer \
             than fresh min_content_width ({fresh} allocs) over {} cells",
            cells.len()
        );
    }

    #[test]
    fn pooled_collect_table_grid_allocates_less_than_fresh() {
        // A table exercising every collection path: captions, colgroup
        // structure, a sectioned row group, and bare cells (anonymous
        // row).
        let html = "<table><caption>cap</caption><colgroup><col><col></colgroup>\
                    <thead><tr><th>h1</th><th>h2</th></tr></thead>\
                    <tbody><tr><td>a1</td><td>a2</td></tr>\
                    <tr><td>b1</td><td>b2</td></tr></tbody>bare1</table>";
        // Correctness first: pooled and fresh must agree exactly.
        let (forest, table) = forest_table(html, "");
        let mut pool = TableGridPool::new();
        let mut grid = pool.take_grid();
        collect_table_grid_into(&forest, table, &mut pool, &mut grid);
        let fresh_grid = collect_table_grid(&forest, table);
        assert_eq!(grid.nrows, fresh_grid.nrows);
        assert_eq!(grid.ncols, fresh_grid.ncols);
        assert_eq!(grid.fixed, fresh_grid.fixed);
        assert_eq!(grid.collapse, fresh_grid.collapse);
        assert_eq!(grid.cells.len(), fresh_grid.cells.len());
        for (a, b) in grid.cells.iter().zip(fresh_grid.cells.iter()) {
            assert_eq!(
                (a.node, a.row, a.col, a.colspan, a.rowspan),
                (b.node, b.row, b.col, b.colspan, b.rowspan)
            );
        }
        assert_eq!(grid.captions, fresh_grid.captions);
        assert_eq!(grid.collapsed_rows, fresh_grid.collapsed_rows);
        assert_eq!(grid.collapsed_cols, fresh_grid.collapsed_cols);
        pool.put_grid(grid);

        // Hermetic measurement on a fresh thread (same discipline as
        // the other pooled-alloc tests): dry-run both closures
        // uncounted, then compare one warm pooled collection against
        // one fresh collection.
        let (pooled, fresh) = std::thread::spawn(move || {
            let (forest, table) = forest_table(html, "");
            let mut pool = TableGridPool::new();
            // Warm the pool; doubles as the pooled dry-run.
            let mut grid = pool.take_grid();
            collect_table_grid_into(&forest, table, &mut pool, &mut grid);
            pool.put_grid(grid);
            // Dry-run the fresh closure uncounted.
            let _ = collect_table_grid(&forest, table);
            let pooled = count_allocations(|| {
                let mut grid = pool.take_grid();
                collect_table_grid_into(&forest, table, &mut pool, &mut grid);
                pool.put_grid(grid);
            });
            let fresh = count_allocations(|| {
                let _ = collect_table_grid(&forest, table);
            });
            (pooled, fresh)
        })
        .join()
        .expect("allocation-measurement thread panicked");
        eprintln!("pooled collect: {pooled} allocs, fresh collect: {fresh} allocs");
        assert!(
            pooled < fresh,
            "pooled collect_table_grid_into ({pooled} allocs) should allocate strictly fewer \
             than fresh collect_table_grid ({fresh} allocs)"
        );
    }

    #[test]
    fn fixup_ws_inside_anon_cell() {
        // CSS Tables 3 §2.2.1 rule 2.3 + §2.1.1 "consecutive": whitespace
        // around stray content in a row belongs to the same anonymous
        // cell (WPT `table-model-fixup` 1.4), not a separate cell.
        // (Uses divs with display: table-row to bypass HTML table
        // foster-parenting of non-cell content.)
        let (mut forest, table) = forest_table(
            "<div class=\"t\"><div class=\"r\"> <span>x</span> </div></div>",
            ".t { display: table; border-spacing: 0; } \
             .r { display: table-row; } \
             span { font-size: 10px; padding: 0; }",
        );
        wrap_table_stray_content(&mut forest, table);
        let grid = collect_table_grid(&forest, table);
        // One row, one cell: the ws + span collapsed into a single
        // anonymous cell.
        assert_eq!(grid.nrows, 1, "one row");
        assert_eq!(grid.ncols, 1, "one column: ws inside the cell");
        assert_eq!(grid.cells.len(), 1, "one cell");
    }

    #[test]
    fn fixup_row_stray_sequence_shares_anonymous_cell() {
        // CSS Tables 3 §2.2.1 rule 2.3 (WPT `table-model-fixup` 2.2):
        // one anonymous cell wraps each *sequence* of consecutive
        // non-cell children of a row, so a trailing span and a stray
        // caption share a cell instead of forming two columns. The
        // misparented column still breaks the run (and is dropped).
        let (mut forest, table) = forest_table(
            "<div class=\"t\"><div class=\"r\"><span>a</span><div class=\"col\"></div><div class=\"c\">2</div><span>b</span><div class=\"cap\">x</div></div></div>",
            ".t { display: table; border-spacing: 0; } \
             .r { display: table-row; } \
             .col { display: table-column; } \
             .c { display: table-cell; font-size: 10px; padding: 0; } \
             .cap { display: table-caption; font-size: 10px; padding: 0; } \
             span { font-size: 10px; }",
        );
        wrap_table_stray_content(&mut forest, table);
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.nrows, 1, "one row");
        assert_eq!(
            grid.ncols, 3,
            "three columns: span cell, real cell, span+caption cell"
        );
        assert_eq!(grid.cells.len(), 3, "three cells");
    }

    #[test]
    fn fixup_row_stray_inlines_share_anonymous_cell() {
        // CSS Tables 3 §2.2.1 rule 2.3: a sequence of consecutive
        // non-cell children of a row shares one anonymous cell —
        // three spans form one column, not three. (The
        // `html-to-css-mapping-1` five-column case is real
        // `table-cell` boxes, which stay separate as real cells;
        // it never pinned per-span wrapping.) (Uses divs with
        // display: table-row to bypass HTML table foster-parenting
        // of non-cell content.)
        let (mut forest, table) = forest_table(
            "<div class=\"t\"><div class=\"r\"><span>a</span><span>b</span><span>c</span></div></div>",
            ".t { display: table; border-spacing: 2px; } \
             .r { display: table-row; } \
             span { font-size: 10px; }",
        );
        wrap_table_stray_content(&mut forest, table);
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.nrows, 1, "one row");
        assert_eq!(grid.ncols, 1, "one column: one cell per stray sequence");
        assert_eq!(grid.cells.len(), 1, "one cell");
    }

    #[test]
    fn collapse_content_box_height_uses_half_border_no_padding() {
        // WPT `td-box-sizing-002` tables 6/13 (height axis): with
        // `box-sizing: content-box` + `border-collapse: collapse`, the
        // table has no padding and its border box extends only half the
        // specified border beyond the content on each side. A 100px
        // specified height with 10px padding and 20/40px borders gives
        // 100 + 10 + 20 = 130, not 100 + 20 + 60 = 180.
        let l = layout_table_html(
            "<table class=\"t\"><tbody><tr><td></td></tr></tbody></table>",
            "body { margin: 0; } \
             .t { display: table; border-top-width: 20px; border-bottom-width: 40px; \
                  border-left-width: 20px; border-right-width: 40px; border-style: solid; \
                  padding: 10px; border-spacing: 0; width: 100px; height: 100px; \
                  box-sizing: content-box; border-collapse: collapse; } \
             td { padding: 0; }",
        );
        let table = &l.children[0].children[0];
        assert_eq!(table.rect.h, 130.0, "collapse content-box height");
    }

    #[test]
    fn collapse_content_box_width_uses_half_border_no_padding() {
        // WPT `td-box-sizing-002` tables 6/13 (width axis): with
        // `box-sizing: content-box` + `border-collapse: collapse`, the
        // table has no padding and its border box extends only half the
        // specified border beyond the content on each side. A 100px
        // specified width with 10px padding and 20/40px borders gives
        // 100 + 10 + 20 = 130, not 100 + 20 + 60 = 180.
        let l = layout_table_html(
            "<table class=\"t\"><tbody><tr><td></td></tr></tbody></table>",
            "body { margin: 0; } \
             .t { display: table; border-top-width: 20px; border-bottom-width: 40px; \
                  border-left-width: 20px; border-right-width: 40px; border-style: solid; \
                  padding: 10px; border-spacing: 0; width: 100px; height: 100px; \
                  box-sizing: content-box; border-collapse: collapse; } \
             td { padding: 0; }",
        );
        let table = &l.children[0].children[0];
        assert_eq!(table.rect.w, 130.0, "collapse content-box width");
    }

    #[test]
    fn collapse_border_box_width_keeps_specified() {
        // Control for the width-axis collapse correction: with
        // `box-sizing: border-box` + `border-collapse: collapse`, a
        // 100px specified width stays 100 (WPT `td-box-sizing-002`
        // tables 4/5).
        let l = layout_table_html(
            "<table class=\"t\"><tbody><tr><td></td></tr></tbody></table>",
            "body { margin: 0; } \
             .t { display: table; border-top-width: 20px; border-bottom-width: 40px; \
                  border-left-width: 20px; border-right-width: 40px; border-style: solid; \
                  padding: 10px; border-spacing: 0; width: 100px; height: 100px; \
                  box-sizing: border-box; border-collapse: collapse; } \
             td { padding: 0; }",
        );
        let table = &l.children[0].children[0];
        assert_eq!(table.rect.w, 100.0, "collapse border-box width");
    }

    #[test]
    fn collapse_border_box_height_keeps_specified() {
        // Control for the height-axis collapse correction: with
        // `box-sizing: border-box` + `border-collapse: collapse`, a
        // 100px specified height stays 100 (WPT `td-box-sizing-002`
        // tables 4/5/12). The half-border correction is content-box
        // only; applying it here shrank the table to 50.
        let l = layout_table_html(
            "<table class=\"t\"><tbody><tr><td></td></tr></tbody></table>",
            "body { margin: 0; } \
             .t { display: table; border-top-width: 20px; border-bottom-width: 40px; \
                  border-left-width: 20px; border-right-width: 40px; border-style: solid; \
                  padding: 10px; border-spacing: 0; width: 100px; height: 100px; \
                  box-sizing: border-box; border-collapse: collapse; } \
             td { padding: 0; }",
        );
        let table = &l.children[0].children[0];
        assert_eq!(table.rect.h, 100.0, "collapse border-box height");
    }

    #[test]
    fn orphan_tbody_gets_anonymous_table() {
        // WPT `html-to-css-mapping-1`: a `<tbody>` with no `<table>`
        // ancestor — script moved it out from under its table, as the
        // test's inline script does with `replaceChild` — gets an
        // anonymous table wrapper per CSS 2.1 §17.2.1. With no table
        // element, border-spacing is 0, so the five cells contribute
        // only their 1px-each-side padding: the tbody measures 10px.
        let (mut doc, _) = bowser_dom::Document::parse_html(
            "<div id=\"d\"><table id=\"t\"><tbody><tr><td></td><td></td><td></td><td></td><td></td></tr></tbody></table></div>"
                .as_bytes(),
        );
        let d = doc.get_element_by_id("d").expect("div");
        let t = doc.get_element_by_id("t").expect("table");
        let tbody = doc
            .children(t)
            .into_iter()
            .find(|&n| {
                matches!(&doc.kind(n), bowser_dom::NodeKind::Element(e) if e.tag_name == "tbody")
            })
            .expect("tbody");
        doc.replace_child(d, tbody, t);
        let ss = bowser_css::parse_stylesheet(
            &[
                TABLE_TEST_RESET_CSS,
                "body { margin: 0; } td { padding: 1px; }",
            ]
            .concat(),
        );
        let l = layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics);
        // body > div > anonymous table > tbody.
        let div = &l.children[0].children[0];
        let anon_table = &div.children[0];
        assert_eq!(anon_table.box_type, crate::BoxType::Table);
        assert!(anon_table.node.is_none(), "wrapper is anonymous");
        let tbody_box = &anon_table.children[0];
        assert!(box_has_tag(&doc, tbody_box, "tbody"));
        assert_eq!(tbody_box.rect.w, 10.0, "orphan tbody width");
    }

    #[test]
    fn percent_padding_intrinsic_is_zero() {
        // WPT `td-box-sizing-003` table 8: "intrinsic size of % padding
        // is 0. final size of % padding is computed against table's
        // width." The 100px specified cell width + 20px borders give
        // 120px columns (240px table); the 30% padding resolves against
        // the 240px table width in final layout but never sizes the
        // columns.
        let l = layout_table_html(
            "<table class=\"t\"><tbody><tr><td><div style=\"width:50px;height:50px\"></div></td><td><div style=\"width:50px;height:50px\"></div></td></tr></tbody></table>",
            "body { margin: 0; } \
             .t { display: table; border-spacing: 0; } \
             .t td { width: 100px; border: 10px solid black; padding: 30%; }",
        );
        let table = &l.children[0].children[0];
        assert_eq!(table.rect.w, 240.0, "table width");
        let group = table
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::TableRowGroup)
            .expect("row group");
        let row = &group.children[0];
        assert_eq!(row.children[0].rect.w, 120.0, "td width");
        assert_eq!(row.children[1].rect.w, 120.0, "td width");
    }

    #[test]
    fn fixup_bare_cell_joins_stray_row() {
        // CSS Tables 3 §2.2.1 rule 2.1: a bare cell directly under the
        // table is not a proper table child, so it wraps into the same
        // anonymous row as the surrounding strays (WPT
        // `table-model-fixup` 2.3: 5b and 5c share a row).
        let (mut forest, table) = forest_table(
            "<div class=\"t\"><div class=\"c\">a</div><span>b</span></div>",
            ".t { display: table; border-spacing: 0; } \
             .c { display: table-cell; font-size: 10px; padding: 0; } \
             span { font-size: 10px; }",
        );
        wrap_table_stray_content(&mut forest, table);
        let grid = collect_table_grid(&forest, table);
        // One row with two cells: the bare cell and the anonymous cell
        // wrapping the span.
        assert_eq!(grid.nrows, 1, "bare cell and stray share one row");
        assert_eq!(grid.ncols, 2, "two columns in the row");
        assert_eq!(grid.cells.len(), 2, "two cells");
    }

    #[test]
    fn fixup_ws_only_run_dropped() {
        // CSS Tables 3 §2.2.1 stage 1 rules 3-4: whitespace between two
        // cells is dropped, not wrapped in an anonymous row.
        let (mut forest, table) = forest_table(
            "<div class=\"t\"><div class=\"c\">a</div> <div class=\"c\">b</div></div>",
            ".t { display: table; border-spacing: 0; } \
             .c { display: table-cell; font-size: 10px; padding: 0; }",
        );
        wrap_table_stray_content(&mut forest, table);
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.nrows, 1, "one row");
        assert_eq!(grid.ncols, 2, "two columns, ws dropped");
        assert_eq!(grid.cells.len(), 2, "two cells");
    }

    // ---- colspan redistribution regressions (CSS Tables 3 §3.9.3.2) ----

    /// WPT `colspan-redistribution` table 25: an unconstrained (auto-width)
    /// spanning cell does not grow a truly pinned column's maximum. The two
    /// 50px `<col>` columns pin their maxima at 50; the 200px cell's excess
    /// is dropped, not forced onto them.
    #[test]
    fn spanning_unconstrained_cell_skips_pinned_maxima() {
        let (mut forest, table) = forest_table(
            "<table><col style=\"width:50px\"><col style=\"width:50px\">\
             <tr><td>50</td><td>50</td></tr>\
             <tr><td colspan=\"2\"><div style=\"width:200px\">x</div></td></tr></table>",
            "table { border-collapse: collapse; } td { font-size: 10px; padding: 0; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.ncols, 2);
        let col_spec = column_specified_widths(&mut forest, &grid, 0.0, 0.0);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut mcw_scratch = Vec::new();
        let mut int_bufs = WalkBufs::new();
        let mut int_inner = WalkBufs::new();
        let (col_max, _) = auto_column_maxima(
            &mut forest,
            &grid,
            &col_spec,
            0.0,
            &FixedFontMetrics,
            &mut cache,
            &mut mcw_scratch,
            &mut int_bufs,
            &mut int_inner,
        );
        // Maxima stay pinned at the 50px col widths; the 200px spanning
        // cell contributes nothing.
        assert!(approx(col_max[0], 50.0), "col0 max {}", col_max[0]);
        assert!(approx(col_max[1], 50.0), "col1 max {}", col_max[1]);
    }

    /// WPT `colspan-redistribution` table 13: a column whose max-content
    /// exceeds its specified width is content-driven, not pinned — the
    /// spanning cell's excess still raises it. The 20px-specified column
    /// with 100px content takes the 300px cell's excess.
    #[test]
    fn spanning_cell_raises_content_driven_column() {
        let (mut forest, table) = forest_table(
            "<table><tr>\
             <td style=\"width:20px\"><div style=\"width:100px\">x</div></td>\
             <td style=\"width:40px\"><div style=\"width:100px\">x</div></td>\
             <td style=\"width:20px\">x</td></tr>\
             <tr><td colspan=\"2\"><div style=\"width:300px\">x</div></td></tr></table>",
            "table { border-collapse: collapse; } td { font-size: 10px; padding: 0; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.ncols, 3);
        let col_spec = column_specified_widths(&mut forest, &grid, 0.0, 0.0);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut mcw_scratch = Vec::new();
        let mut int_bufs = WalkBufs::new();
        let mut int_inner = WalkBufs::new();
        let (col_max, _) = auto_column_maxima(
            &mut forest,
            &grid,
            &col_spec,
            0.0,
            &FixedFontMetrics,
            &mut cache,
            &mut mcw_scratch,
            &mut int_bufs,
            &mut int_inner,
        );
        // The 300px cell over 100/100px maxima: 100px excess distributes,
        // raising both columns past their 100px content maxima.
        assert!(col_max[0] > 100.0, "col0 max {} > 100", col_max[0]);
        assert!(col_max[1] > 100.0, "col1 max {} > 100", col_max[1]);
    }

    /// Pinned-column epsilon: a column whose max-content exceeds its
    /// specified width by less than the old 0.5px fudge is
    /// content-driven, not pinned — an unconstrained spanning cell's
    /// excess must still reach it. The 50px `<col>` column with 50.4px
    /// of content takes the 200px spanning cell's 99.6px excess
    /// (growing to 150px) while its exactly-pinned 50px sibling stays
    /// out; the preferred table width is 200, not 100.4.
    #[test]
    fn spanning_cell_reaches_column_within_half_px_of_specified() {
        let (mut forest, table) = forest_table(
            "<table><col style=\"width:50px\"><col style=\"width:50px\">\
             <tr><td><div style=\"width:50.4px\">x</div></td><td></td></tr>\
             <tr><td colspan=\"2\"><div style=\"width:200px\">x</div></td></tr></table>",
            "table { border-collapse: collapse; } td { font-size: 10px; padding: 0; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.ncols, 2);
        let col_spec = column_specified_widths(&mut forest, &grid, 0.0, 0.0);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut mcw_scratch = Vec::new();
        let mut int_bufs = WalkBufs::new();
        let mut int_inner = WalkBufs::new();
        let (col_max, _) = auto_column_maxima(
            &mut forest,
            &grid,
            &col_spec,
            0.0,
            &FixedFontMetrics,
            &mut cache,
            &mut mcw_scratch,
            &mut int_bufs,
            &mut int_inner,
        );
        // The excess is distributed, not dropped: the content-driven
        // column absorbs it all (the exactly-pinned sibling is an
        // ineligible recipient).
        assert!(approx(col_max[0], 150.0), "col0 max {}", col_max[0]);
        assert!(approx(col_max[1], 50.0), "col1 max {}", col_max[1]);
        let mut pool = TableGridPool::new();
        let pref = table_pref_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut cache,
            &mut pool,
            true,
        );
        assert!(approx(pref, 200.0), "preferred {pref}");
    }

    /// WPT `colspan-redistribution` table 4: a spanning cell whose
    /// percentage width (19%) does not exceed the spanned columns'
    /// intrinsic percentages (20%) contributes no min/max-content at
    /// all — its percentage failed to redistribute.
    #[test]
    fn spanning_failed_percentage_sizes_nothing() {
        let (mut forest, table) = forest_table(
            "<table><tr>\
             <td style=\"width:20%\"><div style=\"width:20px\">x</div></td>\
             <td style=\"width:80px\"><div style=\"width:80px\">x</div></td></tr>\
             <tr><td colspan=\"2\" style=\"width:19%\"><div style=\"width:208px\">x</div></td></tr></table>",
            "table { border-collapse: collapse; } td { font-size: 10px; padding: 0; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.ncols, 2);
        let col_spec = column_specified_widths(&mut forest, &grid, 0.0, 0.0);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut mcw_scratch = Vec::new();
        let mut int_bufs = WalkBufs::new();
        let mut int_inner = WalkBufs::new();
        let (col_max, col_max_nonspanning) = auto_column_maxima(
            &mut forest,
            &grid,
            &col_spec,
            0.0,
            &FixedFontMetrics,
            &mut cache,
            &mut mcw_scratch,
            &mut int_bufs,
            &mut int_inner,
        );
        let mins = auto_column_minima(
            &mut forest,
            &grid,
            &col_spec,
            0.0,
            0.0,
            &FixedFontMetrics,
            &col_max,
            &col_max_nonspanning,
        );
        // Maxima stay at the content widths (20/80); the 208px cell
        // adds nothing. Minima likewise.
        assert!(approx(col_max[0], 20.0), "col0 max {}", col_max[0]);
        assert!(approx(col_max[1], 80.0), "col1 max {}", col_max[1]);
        assert!(approx(mins[0], 20.0), "col0 min {}", mins[0]);
        assert!(approx(mins[1], 80.0), "col1 min {}", mins[1]);
    }

    /// WPT `colspan-redistribution` tables 29-31: spanning distribution
    /// is cumulative in (span, DOM) order — a later cell distributes
    /// over the current measures, seeing earlier cells' contributions.
    #[test]
    fn spanning_distribution_is_cumulative() {
        // Table 29: the 208px cell (span 2) runs before the 408px cell
        // (span 2, later in DOM). Cumulatively, the 408px cell sees the
        // 208px cell's contributions; the result is 100/267/133.
        let (mut forest, table) = forest_table(
            "<table><tr><td colspan=\"2\"><div style=\"width:208px\">x</div></td>\
             <td><div style=\"width:50px\">x</div></td></tr>\
             <tr><td><div style=\"width:50px\">x</div></td>\
             <td colspan=\"2\"><div style=\"width:408px\">x</div></td></tr>\
             <tr><td>x</td><td><div style=\"width:50px\">x</div></td><td>x</td></tr></table>",
            "table { border-collapse: collapse; } td { font-size: 10px; padding: 0; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.ncols, 3);
        let col_spec = column_specified_widths(&mut forest, &grid, 0.0, 0.0);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut mcw_scratch = Vec::new();
        let mut int_bufs = WalkBufs::new();
        let mut int_inner = WalkBufs::new();
        let (col_max, _) = auto_column_maxima(
            &mut forest,
            &grid,
            &col_spec,
            0.0,
            &FixedFontMetrics,
            &mut cache,
            &mut mcw_scratch,
            &mut int_bufs,
            &mut int_inner,
        );
        // The middle column takes the largest share (267): it is spanned
        // by both cells, and the second cell's distribution weights by
        // the maxima the first cell established.
        assert!(
            col_max[1] > col_max[0],
            "col1 {} > col0 {}",
            col_max[1],
            col_max[0]
        );
        assert!(
            col_max[1] > col_max[2],
            "col1 {} > col2 {}",
            col_max[1],
            col_max[2]
        );
    }

    #[test]
    fn auto_mode_merge_uses_same_spanning_set_rule() {
        // WPT column-track-merging table 1 (td_auto): a colspan=10 cell
        // over 10 tracks with a 1-track cell beside it merges to 2 tracks.
        // Per CSS Tables 3 §3.3.2, a track merges only when it and the
        // previous track are spanned by the exact same set of cells:
        // tracks 1..=9 are spanned by exactly the colspan=10 cell (the
        // same set as track 0), so they merge; track 10 is spanned by
        // both the colspan cell (which ends there) and the 1-track cell
        // (which originates there), so it is preserved.
        let l = layout_table_html(
            "<table><tr><td colspan=\"10\"></td><td></td></tr>\
             <tr><td colspan=\"10\"></td><td></td></tr></table>",
            "table { border-spacing: 0; } td { width: 50px; padding: 0; }",
        );
        let table = &l.children[0].children[0];
        // 2 columns × 50px = 100 (no border, no spacing).
        assert_eq!(table.rect.w, 100.0);
    }

    #[test]
    fn auto_mode_merge_keeps_track_where_span_sets_differ() {
        // CSS Tables 3 §3.3.2 regression:
        // `<tr><td colspan="2">X</td></tr><tr><td colspan="3">Y</td></tr>`
        // Track 2 is spanned by a different set of cells than track 1
        // (the colspan=2 cell ends at track 2), so it must survive the
        // merge even though no cell originates there. The old
        // starts-only rule collapsed the grid to a single column.
        // (Asserted at the merge-function level: rendered widths cannot
        // discriminate, since the surviving empty track legitimately
        // resolves to zero width under auto column sizing.)
        // placed tuples: (node, row, col, colspan, rowspan, _).
        let mut placed = vec![
            (0usize, 0usize, 0usize, 2usize, 1usize, 0usize),
            (1usize, 1usize, 0usize, 3usize, 1usize, 0usize),
        ];
        let mut col_ranges: Vec<(usize, usize, usize)> = Vec::new();
        let mut group_ranges: Vec<(usize, usize, usize)> = Vec::new();
        let ncols = merge_redundant_columns_inner(
            &mut placed,
            &mut col_ranges,
            &mut group_ranges,
            &[false; 3],
            3,
            true,
        );
        assert_eq!(ncols, 2);
        // The colspan=2 cell lost the merged track; the colspan=3 cell
        // still spans both survivors.
        assert_eq!(placed[0].3, 1);
        assert_eq!(placed[1].3, 2);
    }

    /// WPT `colspan-redistribution` table 14: an auto-width spanning
    /// cell over all-percentage columns distributes its min-content
    /// proportionally to the percentages (not to headroom). The 300px
    /// cell over two 25% columns yields 146/146 minima.
    #[test]
    fn spanning_all_pct_distributes_proportionally() {
        let (mut forest, table) = forest_table(
            "<table><tr>\
             <td style=\"width:25%\"><div style=\"width:50px\">x</div></td>\
             <td style=\"width:25%\"><div style=\"width:30px\">x</div></td>\
             <td style=\"width:20px\">x</td></tr>\
             <tr><td colspan=\"2\"><div style=\"width:300px\">x</div></td></tr></table>",
            "table { border-spacing: 8px; } td { font-size: 10px; padding: 0; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert_eq!(grid.ncols, 3);
        let col_spec = column_specified_widths(&mut forest, &grid, 0.0, 0.0);
        let (_, spacing_h, _) = table_border_model(&forest, table);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut mcw_scratch = Vec::new();
        let mut int_bufs = WalkBufs::new();
        let mut int_inner = WalkBufs::new();
        let (col_max, col_max_nonspanning) = auto_column_maxima(
            &mut forest,
            &grid,
            &col_spec,
            spacing_h,
            &FixedFontMetrics,
            &mut cache,
            &mut mcw_scratch,
            &mut int_bufs,
            &mut int_inner,
        );
        let mins = auto_column_minima(
            &mut forest,
            &grid,
            &col_spec,
            spacing_h,
            0.0,
            &FixedFontMetrics,
            &col_max,
            &col_max_nonspanning,
        );
        // 300 - 8 spacing = 292 distributable; each 25% column takes
        // half of it: 146/146.
        assert!(approx(mins[0], 146.0), "col0 min {}", mins[0]);
        assert!(approx(mins[1], 146.0), "col1 min {}", mins[1]);
    }

    /// WPT `colspan-redistribution` table 14, preferred-width floor
    /// (rule #1): the per-column floor is `max(col_min, col_pref) / p`.
    /// The spanning-min redistribution raises the 25% columns' minima
    /// to 146/146 (see `spanning_all_pct_distributes_proportionally`),
    /// above their 50/30 max-contents, so the floor is 146/0.25 = 584,
    /// plus 32px border-spacing = 616.
    #[test]
    fn pct_floor_uses_redistributed_min() {
        let (mut forest, table) = forest_table(
            "<table><tr>\
             <td style=\"width:25%\"><div style=\"width:50px\">x</div></td>\
             <td style=\"width:25%\"><div style=\"width:30px\">x</div></td>\
             <td style=\"width:20px\">x</td></tr>\
             <tr><td colspan=\"2\"><div style=\"width:300px\">x</div></td></tr></table>",
            "table { border-spacing: 8px; } td { font-size: 10px; padding: 0; }",
        );
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut pool = TableGridPool::new();
        let pref = table_pref_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut cache,
            &mut pool,
            true,
        );
        assert!(approx(pref, 616.0), "preferred {pref}");
    }

    /// Guard for the `max(col_min, col_pref)` floor: the max with
    /// `col_pref` preserves the pure preferred-width floor. WPT
    /// `table-width-redistribution` table 5 (`width: fit-content`): a
    /// 20% column with max-content 75 and min-content 50 floors at
    /// 75/0.2 = 375, not 50/0.2 = 250 — plus 32px border-spacing = 407.
    #[test]
    fn pct_floor_pref_dominant_guard() {
        let (mut forest, table) = forest_table(
            "<table><tr>\
             <td style=\"width:20%; min-width:50px\"><div style=\"width:75px\">x</div></td>\
             <td>x</td><td>x</td></tr></table>",
            "table { width: fit-content; border-spacing: 8px; } \
             td { font-size: 10px; padding: 0; }",
        );
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut pool = TableGridPool::new();
        let pref = table_pref_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut cache,
            &mut pool,
            true,
        );
        assert!(approx(pref, 407.0), "preferred {pref}");
    }

    /// Vertical writing mode: the intrinsic percentage reads the
    /// inline-axis spec (`height`), never the physical `width` (the
    /// block axis there). A `height:25%` cell contributes 0.25; a
    /// `width:25%` cell contributes nothing.
    #[test]
    fn vwm_intrinsic_pcts_read_height_not_width() {
        let (forest, table) = forest_table(
            "<table><tr>\
             <td style=\"height:25%\"><div style=\"height:40px\">x</div></td>\
             <td style=\"width:25%\"><div style=\"height:30px\">x</div></td></tr></table>",
            "table { writing-mode: vertical-lr; border-spacing: 0; } \
             td { font-size: 10px; padding: 0; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.vertical, "vertical-lr table must set grid.vertical");
        assert_eq!(grid.ncols, 2);
        let (pcts, _) = intrinsic_column_pcts(&forest, &grid, &[0.0, 0.0]);
        assert!(approx(pcts[0], 0.25), "pcts {pcts:?}");
        assert!(approx(pcts[1], 0.0), "pcts {pcts:?}");
    }

    /// Vertical writing mode: a `<col style="width:100px">` is a
    /// block-axis width and must not floor the column's inline size;
    /// a `<col style="height:120px">` is the inline spec and pins it.
    #[test]
    fn vwm_col_spec_reads_height_not_width() {
        let (mut forest, table) = forest_table(
            "<table><col style=\"width:100px\"><col style=\"height:120px\"><tr>\
             <td><div style=\"height:10px\">x</div></td>\
             <td><div style=\"height:10px\">x</div></td></tr></table>",
            "table { writing-mode: vertical-lr; border-spacing: 0; } \
             td { font-size: 10px; padding: 0; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.vertical, "vertical-lr table must set grid.vertical");
        assert_eq!(grid.ncols, 2);
        let col_spec = column_specified_widths(&mut forest, &grid, 0.0, 0.0);
        assert!(
            col_spec.0[0].is_none(),
            "block-axis col width must not specify the inline size: {:?}",
            col_spec.0
        );
        assert!(
            col_spec.0[1].is_some_and(|w| approx(w, 120.0)),
            "inline-axis col height must specify the column: {:?}",
            col_spec.0
        );
    }

    /// Vertical writing mode, fixed layout: a non-first-row cell with a
    /// specified `height` contributes its vertical content-based width,
    /// not the horizontal `generic_preferred_width`. The 80px-high
    /// second-row cell holds 60px of content: the preferred inline
    /// size is 60, not the ~6px horizontal text width. (The table needs
    /// a definite inline `height` to trigger fixed mode.)
    #[test]
    fn vwm_fixed_non_first_row_uses_vertical_content() {
        let (mut forest, table) = forest_table(
            "<table><tr><td><div style=\"height:10px\">x</div></td></tr>\
             <tr><td style=\"height:80px\"><div style=\"height:60px\">x</div></td></tr></table>",
            "table { writing-mode: vertical-lr; table-layout: fixed; height: 200px; \
             border-spacing: 0; } td { font-size: 10px; padding: 0; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.vertical, "vertical-lr table must set grid.vertical");
        assert!(grid.fixed, "table-layout: fixed must set grid.fixed");
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut pool = TableGridPool::new();
        let pref = table_pref_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut cache,
            &mut pool,
            true,
        );
        assert!(approx(pref, 60.0), "preferred {pref}");
    }

    #[test]
    fn nested_pct_table_contributes_column_max_sum_not_infinity() {
        // WPT `css/css-tables/tentative/table-minmax.html` (table-layout
        // case): an inner table with a `width:100%` column beside a
        // 100px-max column contributes 30 + 100 + 30 = 160px to its
        // outer column — not its infinite percentage-floor preferred
        // width — so the outer table sizes to 290 and the inner table
        // to its 160 column, instead of the outer column collapsing to
        // the inner minima (110) and the inner table to 110.
        let (doc, l) = layout_table_doc(
            "<table class=\"o\"><tr><td><table class=\"i\"><tr>\
             <td style=\"width:100%\"><div style=\"width:30px\"></div></td>\
             <td><div style=\"width:50px\"></div><div style=\"width:50px\"></div></td>\
             </tr></table></td>\
             <td><div style=\"width:100px\"></div></td></tr></table>",
            "table { border-spacing: 10px 10px; } td { padding: 0; } \
             td > div { display: inline-block; }",
        );
        let mut tables: Vec<f32> = Vec::new();
        let mut stack: Vec<&crate::LayoutBox> = vec![&l];
        while let Some(b) = stack.pop() {
            if box_has_tag(&doc, b, "table") {
                tables.push(b.rect.w);
            }
            stack.extend(b.children.iter().rev());
        }
        assert_eq!(tables.len(), 2, "outer + inner tables, got {tables:?}");
        assert!(approx(tables[0], 290.0), "outer w {}, want 290", tables[0]);
        assert!(approx(tables[1], 160.0), "inner w {}, want 160", tables[1]);
    }
}
// w65: `has_pct_padding` is vertical-writing-mode aware.
// ---------------------------------------------------------------------------

/// Regression tests for the percentage-padding gate in
/// [`table_pref_width`]'s fixpoint preamble: for a vertical table the
/// inline axis is physical vertical, so percentage top/bottom padding
/// must open the second-pass loop just like left/right padding does for
/// horizontal tables.
#[cfg(test)]
mod w65_d_pctpad_tests {
    use super::{
        PrefWidthCache, TableGridPool, collect_table_grid, has_pct_padding, table_pref_width,
    };
    use crate::{FixedFontMetrics, TABLE_TEST_RESET_CSS};
    use std::collections::HashMap;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// Build a box forest for `html` + `css` and return it with the table
    /// box index.
    fn forest_table(html: &str, css: &str) -> (crate::boxes::Forest, usize) {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
        let ua = bowser_css::default_ua_stylesheet();
        let Some((forest, root)) = crate::boxes::build_box_tree_with_fonts(
            &doc,
            &[&ss],
            &ua,
            800.0,
            600.0,
            &HashMap::new(),
            &crate::FixedFontMetrics,
        ) else {
            panic!("no root box");
        };
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == crate::BoxType::Table {
                return (forest, i);
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        panic!("no table box");
    }

    fn pref_width(html: &str, css: &str) -> f32 {
        let (mut forest, table) = forest_table(html, css);
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let mut pool = TableGridPool::new();
        table_pref_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut cache,
            &mut pool,
            true,
        )
    }

    #[test]
    fn vertical_pct_top_padding_opens_gate() {
        let (forest, table) = forest_table(
            "<table><tr><td>x</td></tr></table>",
            "table { writing-mode: vertical-rl; } td { padding-top: 10%; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.vertical, "vertical-rl table must set grid.vertical");
        assert!(
            has_pct_padding(&forest, &grid),
            "percentage top padding on a vertical-table cell must open the second-pass gate"
        );
    }

    #[test]
    fn horizontal_pct_top_padding_keeps_gate_closed() {
        let (forest, table) = forest_table(
            "<table><tr><td>x</td></tr></table>",
            "td { padding-top: 10%; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert!(
            !grid.vertical,
            "table without writing-mode must be horizontal"
        );
        assert!(
            !has_pct_padding(&forest, &grid),
            "horizontal tables only consider physical left/right padding"
        );
    }

    #[test]
    fn horizontal_pct_side_padding_still_opens_gate() {
        let (forest, table) = forest_table(
            "<table><tr><td>x</td></tr></table>",
            "td { padding-left: 10%; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert!(
            has_pct_padding(&forest, &grid),
            "percentage left padding must keep opening the gate"
        );
    }

    /// End-to-end: percentage padding contributes 0 to the intrinsic
    /// width, even in vertical writing mode. Two columns: the first
    /// holds text (content width C), the second is empty with
    /// `padding-top: 10%`. Per WPT `td-box-sizing-003` ("intrinsic size
    /// of % padding is 0"), the preferred width must stay C; the
    /// percentage resolves only in final layout against the table's
    /// used width, never during intrinsic sizing.
    #[test]
    fn vertical_pct_top_padding_resolves_in_second_pass() {
        let html = "<table><tr><td>x</td><td class=\"pad\"></td></tr></table>";
        let base_css = "table { writing-mode: vertical-rl; } td { font-size: 10px; }";
        let pref_zero = pref_width(html, &format!("{base_css} .pad {{ padding-top: 0; }}"));
        let pref_pct = pref_width(html, &format!("{base_css} .pad {{ padding-top: 10%; }}"));
        assert!(pref_zero > 0.0, "content column must have positive width");
        assert!(
            approx(pref_pct, pref_zero),
            "percentage padding must not grow the intrinsic width: \
             pref_pct={pref_pct} pref_zero={pref_zero}"
        );
    }
}

/// Wave 65 (b-spanmin): WPT `colspan-redistribution` table 26 regression
/// tests. A percentage `<col>` width must resolve against the assignable
/// width (content width minus border-spacing), never against the
/// spacing-inclusive content width (CSS Tables 3 §3.9.2.1 rule 3,
/// mirroring the fixed layout's rebasing): the 50% `<col>` floors its
/// column at 142px of the 284px assignable width, not 154px of the 308px
/// content width, so the used-width excess lands on the constrained
/// 50px column instead of stranding the percentage column at 154.
#[cfg(test)]
mod w65_b_spanmin_tests {
    use crate::{BoxType, FixedFontMetrics, LayoutBox, Size, TABLE_TEST_RESET_CSS, layout};

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// Mode split (see `standards_html` in `tests.rs`): no doctype
    /// is injected, so doctype-less input parses as full quirks
    /// mode. Pass an explicit doctype when the fixture's mode
    /// matters.
    fn layout_html(html: &str, css: &str) -> LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
        layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics)
    }

    /// Navigate a laid-out table: body > table > row-group > row.
    fn row_at(l: &LayoutBox, row: usize) -> &LayoutBox {
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, BoxType::Table);
        let group = table
            .children
            .iter()
            .find(|c| c.box_type == BoxType::TableRowGroup)
            .expect("row group");
        &group.children[row]
    }

    /// Navigate a laid-out table: body > table > row-group > row > cell.
    fn cell_at(l: &LayoutBox, row: usize, col: usize) -> &LayoutBox {
        &row_at(l, row).children[col]
    }

    /// WPT `colspan-redistribution` table 26: `<col style="width:50px">`
    /// plus `<col style="width:50%">`; the second row's `colspan=2`
    /// auto-width cell holds four 50px inline-block divs (200px
    /// max-content, 8px border-spacing). The 50% `<col>` must floor its
    /// column at 142px (50% of the 284px assignable width), not 154px
    /// (50% of the 308px spacing-inclusive content width). The rule #1
    /// floor (142 / 0.5 + 3 * 8 = 308) sizes the table, and the 92px
    /// used-width excess lands on the constrained no-percentage C0:
    /// table 308, both columns 142, colspan cell 292.
    #[test]
    fn spanning_min_fill_admits_pct_column_with_headroom() {
        let l = layout_html(
            "<table><col style=\"width:50px\"><col style=\"width:50%\">\
             <tr><td>50</td><td>50</td></tr>\
             <tr><td colspan=\"2\"><div style=\"width:50px\">50</div>\
             <div style=\"width:50px\">50</div><div style=\"width:50px\">50</div>\
             <div style=\"width:50px\">50</div></td></tr></table>",
            "table { border-spacing: 8px; } td { font-size: 10px; } \
             td > div { display: inline-block; }",
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 308.0),
            "table width {}, want 308",
            table.rect.w
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c0.rect.w, 142.0), "col0 {}, want 142", c0.rect.w);
        assert!(approx(c1.rect.w, 142.0), "col1 {}, want 142", c1.rect.w);
        let span = cell_at(&l, 1, 0);
        assert!(
            approx(span.rect.w, 292.0),
            "spanning cell {}, want 292",
            span.rect.w
        );
    }

    /// The percentage `<col>` rebasing must not disturb tables without
    /// percentage column specs: WPT `colspan-redistribution` table 23's
    /// 40% *cell* keeps its 80px step-1 share while the fixed column
    /// takes the rest of the 208px spanning cell (table 252: 80/120/20).
    #[test]
    fn spanning_min_fill_still_skips_pct_column_past_max() {
        let l = layout_html(
            "<table style=\"width:1px\">\
             <tr><td style=\"width:40%\"><div style=\"width:20px\">40%</div></td>\
             <td style=\"width:50px\"><div style=\"width:50px\">50</div></td>\
             <td style=\"width:100px\"><div style=\"width:20px\">20</div></td></tr>\
             <tr><td colspan=\"2\"><div style=\"width:208px\">208px min</div></td></tr>\
             </table>",
            "table { border-spacing: 8px; } td { font-size: 10px; } \
             td > div { display: inline-block; }",
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 252.0),
            "table width {}, want 252",
            table.rect.w
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        let c2 = cell_at(&l, 0, 2);
        assert!(approx(c0.rect.w, 80.0), "pct col {}, want 80", c0.rect.w);
        assert!(
            approx(c1.rect.w, 120.0),
            "fixed col {}, want 120",
            c1.rect.w
        );
        assert!(approx(c2.rect.w, 20.0), "third col {}, want 20", c2.rect.w);
    }

    /// A percentage `<colgroup>` width flowing through an `auto`-width
    /// `<col>` must be rebased exactly like a direct percentage
    /// `<col>`: `column_specified_widths` falls through to the group's
    /// width when the col's own normalized spec is `Auto`, so
    /// `col_spec_pct_fractions` must do the same. Before that fix the
    /// rebase loop saw fraction 0.0, skipped the column, and left the
    /// group's 50% resolved against the spacing-inclusive content
    /// width (154 of 308) instead of the 284 assignable width (142).
    /// Geometry mirrors the table-26 test above with
    /// `<col style="width:50%">` replaced by a 50% `<colgroup>`
    /// containing an `auto` `<col>`.
    #[test]
    fn spanning_min_fill_rebases_group_pct_through_auto_col() {
        let l = layout_html(
            "<table><col style=\"width:50px\"><colgroup style=\"width:50%\">\
             <col style=\"width:auto\"></colgroup>\
             <tr><td>50</td><td>50</td></tr>\
             <tr><td colspan=\"2\"><div style=\"width:50px\">50</div>\
             <div style=\"width:50px\">50</div><div style=\"width:50px\">50</div>\
             <div style=\"width:50px\">50</div></td></tr></table>",
            "table { border-spacing: 8px; } td { font-size: 10px; } \
             td > div { display: inline-block; }",
        );
        let table = &l.children[0].children[0];
        assert!(
            approx(table.rect.w, 308.0),
            "table width {}, want 308",
            table.rect.w
        );
        let c0 = cell_at(&l, 0, 0);
        let c1 = cell_at(&l, 0, 1);
        assert!(approx(c0.rect.w, 142.0), "col0 {}, want 142", c0.rect.w);
        assert!(approx(c1.rect.w, 142.0), "col1 {}, want 142", c1.rect.w);
        let span = cell_at(&l, 1, 0);
        assert!(
            approx(span.rect.w, 292.0),
            "spanning cell {}, want 292",
            span.rect.w
        );
    }
}

/// Wave 65 (e-capvwm): vertical-writing-mode caption intrinsic widths.
/// For a vertical table the algorithm width is the inline (physical
/// vertical) size, so the caption floors must be measured on the
/// vertical axis — not the horizontal min/max-content measure used for
/// horizontal tables.
#[cfg(test)]
mod w65_e_capvwm_tests {
    use super::{
        PrefWidthCache, TableGridPool, collect_table_grid, table_caption_min_width,
        table_caption_pref_width, table_pref_width,
    };
    use crate::{FixedFontMetrics, TABLE_TEST_RESET_CSS};

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// Box tree plus the table node, for forest-level (pre-layout)
    /// intrinsic measurements.
    fn forest_table(html: &str, css: &str) -> (crate::boxes::Forest, usize) {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
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
        let mut stack = vec![root];
        let mut table = None;
        while let Some(i) = stack.pop() {
            if forest.nodes[i].box_type == crate::BoxType::Table {
                table = Some(i);
                break;
            }
            stack.extend(forest.nodes[i].children.iter().copied());
        }
        (forest, table.expect("table box"))
    }

    const VWM_CSS: &str = "table { writing-mode: vertical-lr; border-spacing: 0; } \
         td, caption { font-size: 10px; padding: 0; }";

    #[test]
    fn vertical_caption_min_width_is_vertical_extent() {
        // The caption holds a 60px-tall block; its horizontal
        // min-content is just the "x" (~6px). CAPMIN must be the
        // vertical extent (60), which is what the floor sites compare
        // against the inline (vertical) algorithm width.
        let (mut forest, table) = forest_table(
            "<table><caption><div style=\"height:60px\">x</div></caption>\
             <tr><td><div style=\"height:10px\">x</div></td></tr></table>",
            VWM_CSS,
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.vertical, "vertical-lr table must set grid.vertical");
        let mut pool = TableGridPool::new();
        let mut scratch = Vec::new();
        let capmin = table_caption_min_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut scratch,
            &mut pool,
        );
        assert!(approx(capmin, 60.0), "capmin {capmin}");
    }

    #[test]
    fn vertical_caption_pref_width_is_vertical_extent() {
        let (mut forest, table) = forest_table(
            "<table><caption><div style=\"height:60px\">x</div></caption>\
             <tr><td><div style=\"height:10px\">x</div></td></tr></table>",
            VWM_CSS,
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.vertical, "vertical-lr table must set grid.vertical");
        let mut pool = TableGridPool::new();
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let cap_pref =
            table_caption_pref_width(&mut forest, table, &FixedFontMetrics, &mut cache, &mut pool);
        assert!(approx(cap_pref, 60.0), "cap_pref {cap_pref}");
    }

    #[test]
    fn vertical_caption_vertical_extras_counted() {
        // Vertical extras ride along with the vertical measure: 60px
        // content + 8px top padding + 4px bottom padding + 2px top
        // margin + 3px bottom margin + 1px top/bottom borders = 79.
        let (mut forest, table) = forest_table(
            "<table><caption style=\"padding-top:8px; padding-bottom:4px; \
             margin-top:2px; margin-bottom:3px; border-top:1px solid black; \
             border-bottom:1px solid black;\"><div style=\"height:60px\">x</div>\
             </caption><tr><td><div style=\"height:10px\">x</div></td></tr></table>",
            VWM_CSS,
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.vertical, "vertical-lr table must set grid.vertical");
        let mut pool = TableGridPool::new();
        let mut scratch = Vec::new();
        let capmin = table_caption_min_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut scratch,
            &mut pool,
        );
        assert!(approx(capmin, 79.0), "capmin {capmin}");
    }

    #[test]
    fn vertical_caption_floors_table_pref_width() {
        // End-to-end: the caption's 60px vertical extent floors the
        // table's intrinsic (preferred) width even though the columns
        // only measure 10px on the vertical axis. The table is floated
        // so the caption max-content floor applies: for a block-level
        // table the caption max-content must NOT floor the preferred
        // width (only the caption min-content floors the used width via
        // CAPMIN — the WPT `caption.html` table 1 rule, transposed to
        // the vertical axis).
        let (mut forest, table) = forest_table(
            "<table><caption><div style=\"height:60px\">x</div></caption>\
             <tr><td><div style=\"height:10px\">x</div></td></tr></table>",
            "table { writing-mode: vertical-lr; border-spacing: 0; float: left; } \
             td, caption { font-size: 10px; padding: 0; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.vertical, "vertical-lr table must set grid.vertical");
        let mut pool = TableGridPool::new();
        let mut cache = PrefWidthCache::new(forest.nodes.len());
        let pref = table_pref_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut cache,
            &mut pool,
            true,
        );
        assert!(approx(pref, 60.0), "preferred {pref}");
    }

    #[test]
    fn horizontal_caption_axis_unchanged() {
        // Sanity: a horizontal table still measures the caption on the
        // horizontal axis (30 a's at 10px = 180).
        let (mut forest, table) = forest_table(
            &format!(
                "<table><caption>{}</caption><tr><td>hi</td></tr></table>",
                "a".repeat(30)
            ),
            "table { border-spacing: 0; } td, caption { font-size: 10px; padding: 0; }",
        );
        let grid = collect_table_grid(&forest, table);
        assert!(!grid.vertical, "plain table must not set grid.vertical");
        let mut pool = TableGridPool::new();
        let mut scratch = Vec::new();
        let capmin = table_caption_min_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut scratch,
            &mut pool,
        );
        assert!(approx(capmin, 180.0), "capmin {capmin}");
    }

    #[test]
    fn vertical_text_caption_falls_back_to_horizontal_measure() {
        // Known limitation: vertical text line stacking is not modeled,
        // so a text-only caption falls back to the horizontal
        // min-content measure ("hello" at 10px = 30) instead of a true
        // vertical extent. It must not panic and must stay non-negative.
        let (mut forest, table) = forest_table(
            "<table><caption>hello</caption>\
             <tr><td><div style=\"height:10px\">x</div></td></tr></table>",
            VWM_CSS,
        );
        let grid = collect_table_grid(&forest, table);
        assert!(grid.vertical, "vertical-lr table must set grid.vertical");
        let mut pool = TableGridPool::new();
        let mut scratch = Vec::new();
        let capmin = table_caption_min_width(
            &mut forest,
            table,
            &FixedFontMetrics,
            &mut scratch,
            &mut pool,
        );
        assert!(capmin >= 0.0, "capmin {capmin}");
        assert!(approx(capmin, 30.0), "capmin {capmin}");
    }
}

/// Wave 65 intrinsic-width floor assembly tests: the three
/// `table_pref_width_collected` floor fixes (percentage floor at
/// >=100% total, `max-content` gating, caption floor gating).
#[cfg(test)]
mod w65_a_floor_tests {
    use crate::{BoxType, FixedFontMetrics, Size, TABLE_TEST_RESET_CSS, layout};

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// Mode split (see `standards_html` in `tests.rs`): no doctype
    /// is injected, so doctype-less input parses as full quirks
    /// mode. Pass an explicit doctype when the fixture's mode
    /// matters.
    fn layout_html(html: &str, css: &str) -> crate::LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
        layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics)
    }

    /// body > div.wrap > table, then row-group > row > cell.
    fn wrapped_table(l: &crate::LayoutBox) -> &crate::LayoutBox {
        let table = &l.children[0].children[0].children[0];
        assert_eq!(table.box_type, BoxType::Table);
        table
    }

    fn wrapped_cell_at(l: &crate::LayoutBox, row: usize, col: usize) -> &crate::LayoutBox {
        let table = wrapped_table(l);
        let group = table
            .children
            .iter()
            .find(|c| c.box_type == BoxType::TableRowGroup)
            .expect("row group");
        &group.children[row].children[col]
    }

    #[test]
    fn pct_total_100_forces_infinite_max_table_width() {
        // WPT css/css-tables/tentative/colspan-redistribution.html
        // table 22: two 50% columns + one 100px column, table width
        // auto, inside a 500px div. Total intrinsic column percentage =
        // 100%: rule #2 forces the maximum table grid width to
        // infinity, so the block-level auto table fills the containing
        // block and the §3.9.3 used-width distribution yields
        // C0=C1=184, C2=100 (border-spacing: 8px like the WPT file:
        // four gaps = 32px).
        let l = layout_html(
            "<div class=\"wrap\"><table><tr>\
             <td class=\"p\"><div class=\"w100\">50%</div></td>\
             <td class=\"p\"><div class=\"w100\">50%</div></td>\
             <td class=\"f\"><div class=\"w100\">100px</div></td>\
             </tr></table></div>",
            ".wrap { width: 500px; } \
             table { border-spacing: 8px; } \
             td { padding: 0; font-size: 10px; } \
             .p { width: 50%; } .f { width: 100px; } .w100 { width: 100px; }",
        );
        let table = wrapped_table(&l);
        assert!(
            approx(table.rect.w, 500.0),
            "auto table must fill the 500px div, got {}",
            table.rect.w
        );
        let c0 = wrapped_cell_at(&l, 0, 0).rect.w;
        let c1 = wrapped_cell_at(&l, 0, 1).rect.w;
        let c2 = wrapped_cell_at(&l, 0, 2).rect.w;
        assert!(approx(c0, 184.0), "C0 expected 184, got {c0}");
        assert!(approx(c1, 184.0), "C1 expected 184, got {c1}");
        assert!(approx(c2, 100.0), "C2 expected 100, got {c2}");
    }

    #[test]
    fn floated_pct_total_100_saturates_to_available() {
        // Companion to the block-level table-22 test: the infinite
        // floor must saturate (not explode) through shrink-to-fit — a
        // floated 50%/50% table fills the available width and splits it
        // evenly (border-spacing: 8px, three gaps = 24px).
        let l = layout_html(
            "<div class=\"wrap\"><table class=\"fl\"><tr>\
             <td class=\"p\"><div class=\"w100\">x</div></td>\
             <td class=\"p\"><div class=\"w100\">y</div></td>\
             </tr></table></div>",
            ".wrap { width: 400px; } \
             table.fl { float: left; border-spacing: 8px; } \
             td { padding: 0; font-size: 10px; } \
             .p { width: 50%; } .w100 { width: 100px; }",
        );
        let table = wrapped_table(&l);
        assert!(
            table.rect.w.is_finite() && approx(table.rect.w, 400.0),
            "floated 50/50 table saturates to the 400px available width, got {}",
            table.rect.w
        );
        let c0 = wrapped_cell_at(&l, 0, 0).rect.w;
        let c1 = wrapped_cell_at(&l, 0, 1).rect.w;
        assert!(approx(c0, 188.0), "C0 expected 188, got {c0}");
        assert!(approx(c1, 188.0), "C1 expected 188, got {c1}");
    }

    #[test]
    fn max_content_table_skips_inverse_percentage_floor() {
        // WPT
        // css/css-tables/tentative/table-width-redistribution.html
        // table 4: width:max-content with a 20% column. "max-content
        // does not allow for assignable size to be influenced by
        // inverse percent": the table is the pure column max-content
        // sum + spacing (100 + 100 + 75 + 32 = 307), not the
        // inverse-percentage floor (75/0.2 + 32 = 407).
        let l = layout_html(
            "<div class=\"wrap\"><table class=\"mc\"><tr>\
             <td><div class=\"w50\">a</div><div class=\"w50\">b</div></td>\
             <td class=\"f\"><div class=\"w50\">c</div><div class=\"w25\">d</div></td>\
             <td class=\"p\"><div class=\"w50\">e</div><div class=\"w25\">f</div></td>\
             </tr></table></div>",
            ".wrap { width: 800px; } \
             table.mc { width: max-content; border-spacing: 8px; } \
             td { padding: 0; font-size: 10px; } \
             td > div { display: inline-block; } \
             .f { width: 100px; } .p { width: 20%; } \
             .w50 { width: 50px; } .w25 { width: 25px; }",
        );
        let table = wrapped_table(&l);
        assert!(
            approx(table.rect.w, 307.0),
            "max-content table must be the column sum + spacing (307), got {}",
            table.rect.w
        );
        // Exact WPT column widths: the percentage column is not
        // inflated by the inverse floor.
        let c0 = wrapped_cell_at(&l, 0, 0).rect.w;
        let c1 = wrapped_cell_at(&l, 0, 1).rect.w;
        let c2 = wrapped_cell_at(&l, 0, 2).rect.w;
        assert!(approx(c0, 120.0), "C0 expected 120, got {c0}");
        assert!(approx(c1, 100.0), "C1 expected 100, got {c1}");
        assert!(approx(c2, 55.0), "C2 expected 55, got {c2}");
    }

    #[test]
    fn fit_content_table_keeps_inverse_percentage_floor() {
        // Load-bearing companion to the max-content gate: WPT
        // `table-width-redistribution` table 5, width:fit-content with
        // a 20% column, expects the inverse-percentage floor to apply
        // (75/0.2 + 32 = 407).
        let l = layout_html(
            "<div class=\"wrap\"><table class=\"fc\"><tr>\
             <td><div class=\"w50\">a</div><div class=\"w50\">b</div></td>\
             <td class=\"f\"><div class=\"w50\">c</div><div class=\"w25\">d</div></td>\
             <td class=\"p\"><div class=\"w50\">e</div><div class=\"w25\">f</div></td>\
             </tr></table></div>",
            ".wrap { width: 800px; } \
             table.fc { width: fit-content; border-spacing: 8px; } \
             td { padding: 0; font-size: 10px; } \
             td > div { display: inline-block; } \
             .f { width: 100px; } .p { width: 20%; } \
             .w50 { width: 50px; } .w25 { width: 25px; }",
        );
        let table = wrapped_table(&l);
        assert!(
            approx(table.rect.w, 407.0),
            "fit-content table keeps the inverse-percentage floor (407), got {}",
            table.rect.w
        );
    }

    #[test]
    fn block_auto_table_ignores_caption_max_content() {
        // WPT css/css-tables/tentative/caption.html table 1: a
        // block-level auto table's caption max-content (80px) must not
        // grow the table — only the caption min (50px) floors the grid
        // min via CAPMIN.
        let l = layout_html(
            "<table><caption><div class=\"d50\"></div><div class=\"d30\"></div></caption>\
             <tr><td>auto</td></tr></table>",
            "caption > div { display: inline-block; } \
             .d50 { width: 50px; } .d30 { width: 30px; } \
             td { font-size: 10px; }",
        );
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, BoxType::Table);
        assert!(
            approx(table.rect.w, 50.0),
            "block auto table sizes to the caption min (50), not its max-content (80): got {}",
            table.rect.w
        );
    }

    #[test]
    fn floated_table_still_floors_to_caption_max_content() {
        // Companion to the caption gate: a floated (shrink-to-fit)
        // table with a caption wider than its columns still sizes to
        // the caption's preferred width.
        let caption: String = std::iter::repeat_n("w", 40).collect::<Vec<_>>().join(" ");
        let html = format!(
            "<table style=\"float:left\"><caption>{caption}</caption>\
             <tr><td>x</td></tr></table>"
        );
        let l = layout_html(&html, "td { font-size: 10px; }");
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, BoxType::Table);
        let cap_pref = caption.chars().count() as f32 * 0.6 * 16.0;
        assert!(
            table.rect.w >= cap_pref,
            "floated table width {} < caption preferred width {cap_pref}",
            table.rect.w
        );
    }
}

/// w65 regression tests: caption.html tables 6/7 and the
/// column-track-merging caption-adjacent subtests.
#[cfg(test)]
mod w65_c_caption_tests {
    use crate::{FixedFontMetrics, Size, TABLE_TEST_RESET_CSS, layout};

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// Mode split (see `standards_html` in `tests.rs`): no doctype
    /// is injected, so doctype-less input parses as full quirks
    /// mode. Pass an explicit doctype when the fixture's mode
    /// matters.
    fn layout_html(html: &str, css: &str) -> crate::LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
        layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics)
    }

    fn table_box(l: &crate::LayoutBox) -> &crate::LayoutBox {
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        table
    }

    fn caption_box(table: &crate::LayoutBox) -> &crate::LayoutBox {
        table
            .children
            .iter()
            .find(|c| c.box_type == crate::BoxType::Block)
            .expect("caption box")
    }

    #[test]
    fn empty_table_with_borders_honors_borders() {
        // WPT `css/css-tables/tentative/caption.html` table 6: "Caption
        // and size of empty table — larger of table borders, and caption
        // min size." An empty table with 50px borders and a caption
        // (min-content 50, preferred 100) stays at its border-box width
        // of 100: the caption's preferred width must not become the
        // table's content width, and CAPMIN (an outer width) must floor
        // the border box, not the content box.
        let l = layout_html(
            "<table class=\"t\"><caption><div class=\"d\">c</div>\
             <div class=\"d\">c</div></caption></table>",
            ".t { border: 50px solid green; border-spacing: 10px; } \
             caption .d { display: inline-block; width: 50px; }",
        );
        let table = table_box(&l);
        assert!(
            approx(table.rect.w, 100.0),
            "empty bordered table should honor its borders: {}",
            table.rect.w
        );
    }

    #[test]
    fn caption_percentage_height_does_not_resolve() {
        // WPT `css/css-tables/tentative/caption.html` table 7 (csswg
        // issue 4676): percentage heights on captions never resolve.
        // Table 200x100, caption `height: 80%`, 30px-tall div content:
        // the caption sizes to its content (30), not 80% of anything.
        // NOTE: the WPT original puts the "c" in via
        // `div:after { content: "c" }`; this engine has no generated
        // content, so the text goes directly in the div. With a truly
        // empty div the caption's line-box strut adds 3.84px below the
        // empty inline-block's bottom-edge baseline (correct per CSS
        // 2.1 §10.8.1); the browser's 30 comes from the ::after text
        // baseline.
        let l = layout_html(
            "<table class=\"t\"><caption class=\"c\">\
             <div class=\"d\">c</div></caption>\
             <tr><td>x</td></tr></table>",
            ".t { width: 200px; height: 100px; } \
             .c { height: 80%; } \
             .d { display: inline-block; height: 30px; } \
             td { font-size: 10px; }",
        );
        let table = table_box(&l);
        let cap = caption_box(table);
        assert!(
            approx(cap.rect.h, 30.0),
            "caption with unresolvable percentage height should size to content: {}",
            cap.rect.h
        );
        assert!(
            approx(table.rect.h, 100.0),
            "table keeps its specified height: {}",
            table.rect.h
        );
    }

    #[test]
    fn col_span_empty_tracks_merge_no_extra_spacing() {
        // WPT `css/css-tables/column-track-merging.html` "col_auto":
        // `<col span=10>` with two single-cell tds. The 8 empty middle
        // tracks merge away (an empty track merges even when a cell ends
        // at it), leaving 2 tracks: 20px borders + 3×20px spacing +
        // 2×50px columns = 180. Before the fix the surviving empty
        // track kept its spacing and the table measured 200.
        let l = layout_html(
            "<table class=\"t\"><caption>auto</caption><col span=\"10\">\
             <tr><td></td><td></td></tr>\
             <tr><td></td><td></td></tr></table>",
            ".t { border: 10px solid gray; border-spacing: 20px; } \
             td { width: 50px; height: 50px; padding: 0; }",
        );
        let table = table_box(&l);
        assert!(
            approx(table.rect.w, 180.0),
            "merged col-span tracks should not draw extra spacing: {}",
            table.rect.w
        );
    }

    #[test]
    fn col_span_zero_width_tracks_merge() {
        // WPT `column-track-merging.html` "col_auto_0px": a `col` with
        // `width: 0px` does not define its tracks ("everyone merges"),
        // so this lays out exactly like `col_auto` above: 180.
        let l = layout_html(
            "<table class=\"t\"><caption>auto col 0px</caption>\
             <col span=\"10\" style=\"width:0px\">\
             <tr><td></td><td></td></tr>\
             <tr><td></td><td></td></tr></table>",
            ".t { border: 10px solid gray; border-spacing: 20px; } \
             td { width: 50px; height: 50px; padding: 0; }",
        );
        let table = table_box(&l);
        assert!(
            approx(table.rect.w, 180.0),
            "zero-width col-span tracks should merge: {}",
            table.rect.w
        );
    }
}

/// Wave 66: vertical-writing-mode caption inline-size resolution.
/// A caption in a vertical writing mode consumes its inline size from
/// the physical `height` property (css-writing-modes-3 §7), so
/// `height: min-content | max-content` resolves against the vertical
/// intrinsic size — not the horizontal line stack the
/// writing-mode-unaware block engine otherwise produces (WPT
/// `css/css-tables/tentative/caption.html` table 13).
#[cfg(test)]
mod w66_vwm_caption_tests {
    use crate::{BoxType, FixedFontMetrics, Size, TABLE_TEST_RESET_CSS, layout};

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.01
    }

    /// Mode split (see `standards_html` in `tests.rs`): no doctype
    /// is injected, so doctype-less input parses as full quirks
    /// mode. Pass an explicit doctype when the fixture's mode
    /// matters.
    fn layout_html(html: &str, css: &str) -> crate::LayoutBox {
        let (doc, _) = bowser_dom::Document::parse_html(html.as_bytes());
        let ss = bowser_css::parse_stylesheet(&[TABLE_TEST_RESET_CSS, css].concat());
        layout(&doc, &[&ss], Size { w: 800.0, h: 600.0 }, &FixedFontMetrics)
    }

    fn table_box(l: &crate::LayoutBox) -> &crate::LayoutBox {
        let table = &l.children[0].children[0];
        assert_eq!(table.box_type, crate::BoxType::Table);
        table
    }

    fn caption_box(table: &crate::LayoutBox) -> &crate::LayoutBox {
        table
            .children
            .iter()
            .find(|c| c.box_type == BoxType::Block)
            .expect("caption box")
    }

    fn tbody_box(table: &crate::LayoutBox) -> &crate::LayoutBox {
        table
            .children
            .iter()
            .find(|c| c.box_type == BoxType::TableRowGroup)
            .expect("tbody box")
    }

    /// WPT `caption.html` table 13 ("Vertical writing mode caption"):
    /// the caption's `height: min-content` is its inline size in
    /// `vertical-rl`, so the 100px-tall inline-block sets the caption's
    /// physical height to 100 — not the 124px the caption text wraps to
    /// at the table's narrow content width. The tbody starts below the
    /// caption at y=100.
    #[test]
    fn vertical_rl_caption_min_content_height() {
        let l = layout_html(
            "<table><caption style=\"writing-mode:vertical-rl;height:min-content;\
             font-size:10px\"><div style=\"height:100px;display:inline-block\">\
             vertical</div> caption bigger than table</caption>\
             <tbody><tr><td>vertical!</td></tr></tbody></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(caption.rect.h, 100.0),
            "vertical caption height {}, want 100",
            caption.rect.h
        );
        let tbody = tbody_box(table);
        assert!(
            approx(tbody.rect.y, 100.0),
            "tbody y {}, want 100",
            tbody.rect.y
        );
    }

    /// The vertical override must not leak into horizontal captions:
    /// the same markup without `writing-mode` keeps the horizontal
    /// line-stack height (text wraps at the narrow table width).
    #[test]
    fn horizontal_caption_keeps_line_stack_height() {
        let l = layout_html(
            "<table><caption style=\"height:min-content;font-size:10px\">\
             <div style=\"height:100px;display:inline-block\">vertical</div>\
             caption bigger than table</caption>\
             <tbody><tr><td>vertical!</td></tr></tbody></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(caption.rect.h, 124.0),
            "horizontal caption height {}, want 124",
            caption.rect.h
        );
    }

    /// A definite `height` on a vertical caption already resolves
    /// through `specified_height`; the override path must not change it.
    #[test]
    fn vertical_caption_definite_height_untouched() {
        let l = layout_html(
            "<table><caption style=\"writing-mode:vertical-rl;height:100px;\
             font-size:10px\"><div style=\"height:100px;display:inline-block\">\
             vertical</div> caption bigger than table</caption>\
             <tbody><tr><td>vertical!</td></tr></tbody></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(caption.rect.h, 100.0),
            "vertical caption height {}, want 100",
            caption.rect.h
        );
    }

    /// `height: max-content` on a vertical caption resolves against the
    /// vertical max-content: the inline-block's 100px plus the unwrapped
    /// text line's vertical extent (26 chars at 0.6em = 156).
    #[test]
    fn vertical_rl_caption_max_content_height() {
        let l = layout_html(
            "<table><caption style=\"writing-mode:vertical-rl;height:max-content;\
             font-size:10px\"><div style=\"height:100px;display:inline-block\">\
             vertical</div> caption bigger than table</caption>\
             <tbody><tr><td>vertical!</td></tr></tbody></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(caption.rect.h, 256.0),
            "vertical caption height {}, want 256",
            caption.rect.h
        );
    }

    /// Wave 75 (B2 definite-spec fast path): WPT
    /// `css/css-tables/caption-writing-mode-001.html` probes.
    ///
    /// The two literal WPT-mirror probes were run fail-before and
    /// are NOT landed as passing tests, because the definite-spec
    /// `cb_w` fast path alone does not close them (measured, table
    /// vertical-lr 300x200, horizontal caption 120x50):
    /// - `margin-left: 20%`: the WPT 40px expectation is 20% of the
    ///   table's INLINE size (200). w75f3 splits the bases in
    ///   `resolve_metrics` (inline sizing on the 300 block size,
    ///   % margins/padding on the 200 inline size), so the used
    ///   margin is now 40; but the caption's final x offset is
    ///   still twice the used margin (80), because the caption's own
    ///   `resolve_metrics` already places it at `cb_x + ml` and
    ///   `place_table_rows` adds the block-start margin again for
    ///   vertical-lr (block-start is the physical left). That
    ///   double-count is follow-up work, not fixed here.
    /// - `margin: auto 0`: caption y offset 0 before and after.
    ///   Auto block margins do not distribute in bowser's block
    ///   engine; the WPT 75px centering needs orthogonal-caption auto
    ///   block-margin distribution against the table's inline size.
    ///
    /// Both remaining pieces are follow-up work alongside the general
    /// auto-block-size reorder (see `orthogonal_caption_cb_w`); the
    /// metadata entry for `caption-writing-mode-001.html` is left
    /// untouched.
    ///
    /// What the fast path DOES fix, and what this test pins: a
    /// horizontal caption's inline sizing in a vertical table. With
    /// `width: auto` the caption must fill the table's block content
    /// width (300), not the inline `content_w` (200) it used before.
    #[test]
    fn w75_horizontal_caption_in_vertical_table_fills_block_width() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px\">caption</caption></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(table.rect.w, 300.0),
            "table w {}, want 300",
            table.rect.w
        );
        assert!(
            approx(table.rect.h, 200.0),
            "table h {}, want 200",
            table.rect.h
        );
        assert!(
            approx(caption.rect.w, 300.0),
            "caption w {}, want 300 (block content width, not inline 200)",
            caption.rect.w
        );
        assert!(
            approx(caption.rect.h, 50.0),
            "caption h {}, want 50",
            caption.rect.h
        );
    }

    /// Wave 75 fixer F3 (item 1): the orthogonal caption's percentage
    /// margins must resolve against the table's INLINE size (200),
    /// while its inline sizing still uses the block size (300).
    /// `margin-left: 20%` is therefore 40, and the `width: auto`
    /// caption fills 300 - 40 = 260, not 300 - 60 = 240 (the
    /// pre-split base) and not 200 - 40 = 160 (the pre-33ee3b0 base).
    /// Fail-before: 240.
    #[test]
    fn w75f3_orthogonal_caption_pct_margin_uses_inline_base() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin-left:20%\">caption</caption></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(caption.rect.w, 260.0),
            "caption w {}, want 260 (300 block base minus 40 = 20% of 200 inline)",
            caption.rect.w
        );
    }

    /// Wave 75 fixer F3 (item 2): the definite-spec fast path must
    /// clamp `hspec` by the table's block-axis min/max specs, so the
    /// caption's `cb_w` matches the table's final block size.
    /// `min-width: 400px` on a 300px block spec grows both to 400.
    /// Fail-before: caption 300.
    #[test]
    fn w75f3_orthogonal_caption_cb_w_clamped_by_block_min() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px;min-width:400px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px\">caption</caption></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(table.rect.w, 400.0),
            "table w {}, want 400",
            table.rect.w
        );
        assert!(
            approx(caption.rect.w, 400.0),
            "caption w {}, want 400 (min-width-clamped block size)",
            caption.rect.w
        );
    }

    /// Same clamp, max side: `max-width: 250px` shrinks both to 250.
    /// Fail-before: caption 300.
    #[test]
    fn w75f3_orthogonal_caption_cb_w_clamped_by_block_max() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px;max-width:250px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px\">caption</caption></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(table.rect.w, 250.0),
            "table w {}, want 250",
            table.rect.w
        );
        assert!(
            approx(caption.rect.w, 250.0),
            "caption w {}, want 250 (max-width-clamped block size)",
            caption.rect.w
        );
    }

    /// Wave 75 fixer F3 (item 8): 33ee3b0 only tested vertical-lr.
    /// The same fast path must fill the block width under
    /// vertical-rl (the transpose mirrors, it does not resize).
    #[test]
    fn w75f3_horizontal_caption_in_vertical_rl_table_fills_block_width() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-rl;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px\">caption</caption></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(table.rect.w, 300.0),
            "table w {}, want 300",
            table.rect.w
        );
        assert!(
            approx(caption.rect.w, 300.0),
            "caption w {}, want 300 (block content width, not inline 200)",
            caption.rect.w
        );
        assert!(
            approx(caption.rect.h, 50.0),
            "caption h {}, want 50",
            caption.rect.h
        );
    }

    /// Wave 76 captions (task 1, fail-before probe from w75): an
    /// orthogonal caption's block-start margin must be counted once.
    /// Table vertical-lr, caption horizontal, `margin-left: 20%`
    /// resolves against the table's inline size (200) to 40, so the
    /// caption's border box starts at x=40 — not 80, the double count
    /// from adding the margin again in `place_table_rows` on top of
    /// the provisional position that already contains it.
    #[test]
    fn w76_orthogonal_caption_pct_margin_counted_once() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin-left:20%\">caption</caption></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(caption.rect.x, 40.0),
            "caption x {}, want 40 (margin 40 counted once, not 80)",
            caption.rect.x
        );
        assert!(
            approx(caption.rect.w, 260.0),
            "caption w {}, want 260",
            caption.rect.w
        );
        let _ = table;
    }

    /// Wave 76 captions (task 1, vertical-rl variant): the block-start
    /// margin is the right margin under vertical-rl, and the caption
    /// also carries a nonzero inline-end (left) margin. The gap from
    /// the table's block-start (right) edge must be exactly the right
    /// margin (40), not right + left (70).
    #[test]
    fn w76_orthogonal_caption_rl_block_start_margin_counted_once() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-rl;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin:0 40px 0 30px\">caption</caption></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(caption.rect.w, 230.0),
            "caption w {}, want 230 (300 - 40 - 30)",
            caption.rect.w
        );
        let gap = table.rect.w - (caption.rect.x + caption.rect.w);
        assert!(
            approx(gap, 40.0),
            "block-start gap {}, want 40 (right margin only, not 40 + 30)",
            gap
        );
    }

    /// Wave 76 captions (task 1, bottom-caption variant): the bottom
    /// loop in `place_table_rows` must use the same delta
    /// formulation as the top loop, or a bottom orthogonal caption
    /// double-counts its block-start margin too.
    #[test]
    fn w76_orthogonal_caption_bottom_pct_margin_counted_once() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin-left:20%;caption-side:bottom\">caption</caption></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(caption.rect.x, 40.0),
            "bottom caption x {}, want 40 (margin 40 counted once, not 80)",
            caption.rect.x
        );
    }

    /// Wave 76 captions (task 1, multi-caption variant): the delta
    /// formulation must hold past the first caption, where the cursor
    /// has advanced past the first caption's extent. Along the table's
    /// block axis an orthogonal caption occupies its inline extent
    /// (algorithm `rect.h` = physical width, 290 for the first caption
    /// with its 10px margin), so the cursor at the second caption is
    /// 300 and its border box starts at x=340 (300 + margin 40) — not
    /// 380 (provisional 40 + shift 340).
    #[test]
    fn w76_orthogonal_caption_second_caption_margin_counted_once() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin-left:10px\">one</caption>\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin-left:40px\">two</caption></table>",
            "",
        );
        let table = table_box(&l);
        let caps: Vec<&crate::LayoutBox> = table
            .children
            .iter()
            .filter(|c| c.box_type == BoxType::Block)
            .collect();
        assert_eq!(caps.len(), 2, "want 2 caption boxes, got {}", caps.len());
        assert!(
            approx(caps[0].rect.x, 10.0),
            "first caption x {}, want 10",
            caps[0].rect.x
        );
        assert!(
            approx(caps[1].rect.x, 340.0),
            "second caption x {}, want 340 (cursor 300 + margin 40, not provisional 40 + shift 340)",
            caps[1].rect.x
        );
    }

    /// Wave 76 captions (task 2): an orthogonal caption's auto block
    /// margins center it along the table's inline axis. Table
    /// vertical-lr, inline content 200, caption block extent 50:
    /// `margin: auto 0` distributes 150 as 75/75, so the caption's
    /// physical top (its block-start edge in table-inline terms) is
    /// at y=75.
    #[test]
    fn w76_orthogonal_caption_auto_block_margins_center() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin:auto 0\">caption</caption></table>",
            "",
        );
        let caption = caption_box(table_box(&l));
        assert!(
            approx(caption.rect.y, 75.0),
            "caption y {}, want 75 (auto block margins center along table inline axis)",
            caption.rect.y
        );
        // Bottom placement centers along the inline axis the same way.
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin:auto 0;caption-side:bottom\">caption</caption></table>",
            "",
        );
        let caption = caption_box(table_box(&l));
        assert!(
            approx(caption.rect.y, 75.0),
            "bottom caption y {}, want 75",
            caption.rect.y
        );
    }

    /// Wave 76 captions (task 2, one-auto variants): a single auto
    /// block margin takes the free space left after the fixed side.
    /// `margin-top: auto; margin-bottom: 10px` on a 50px caption in
    /// a 200px inline content box leaves 140 for the top margin.
    #[test]
    fn w76_orthogonal_caption_single_auto_block_margin() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin-top:auto;margin-bottom:10px\">caption</caption></table>",
            "",
        );
        let caption = caption_box(table_box(&l));
        assert!(
            approx(caption.rect.y, 140.0),
            "caption y {}, want 140 (200 - 50 - fixed 10)",
            caption.rect.y
        );
        // Fixed top margin still offsets along the inline axis.
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin-top:10px\">caption</caption></table>",
            "",
        );
        let caption = caption_box(table_box(&l));
        assert!(
            approx(caption.rect.y, 10.0),
            "caption y {}, want 10 (fixed block-start margin along table inline axis)",
            caption.rect.y
        );
    }

    /// Wave 76 captions (task 3): WPT `caption.html` table 8
    /// ("Multiple captions") replicated exactly: relative table,
    /// border-spacing 0, line-height 30px, two top captions, a bare
    /// td, and two bottom captions. The assert message prints all
    /// five rects so a failure discriminates a 60px td (row-height /
    /// wrapping bug) from swapped bottom captions from a driver-side
    /// offsetTop error.
    #[test]
    fn w76_caption_html_table8_multiple_captions() {
        let l = layout_html(
            "<table style=\"position:relative\">\
             <caption style=\"line-height:30px\">over1</caption>\
             <caption style=\"line-height:30px\">over2</caption>\
             <caption style=\"line-height:30px;caption-side:bottom\">under1</caption>\
             <caption style=\"line-height:30px;caption-side:bottom\">under2</caption>\
             <td style=\"line-height:30px\">2 above me, 2 below me?</td></table>",
            "",
        );
        let table = table_box(&l);
        let caps: Vec<&crate::LayoutBox> = table
            .children
            .iter()
            .filter(|c| c.box_type == BoxType::Block)
            .collect();
        assert_eq!(caps.len(), 4, "want 4 caption boxes, got {}", caps.len());
        fn find_cell(b: &crate::LayoutBox) -> Option<&crate::LayoutBox> {
            if b.box_type == BoxType::TableCell {
                return Some(b);
            }
            b.children.iter().find_map(find_cell)
        }
        let cell = find_cell(table).expect("td cell box");
        let (over1, over2, under1, under2) = (caps[0], caps[1], caps[2], caps[3]);
        let msg = format!(
            "over1 y={} h={} w={} | over2 y={} h={} w={} | td y={} h={} w={} | under1 y={} h={} w={} | under2 y={} h={} w={}",
            over1.rect.y,
            over1.rect.h,
            over1.rect.w,
            over2.rect.y,
            over2.rect.h,
            over2.rect.w,
            cell.rect.y,
            cell.rect.h,
            cell.rect.w,
            under1.rect.y,
            under1.rect.h,
            under1.rect.w,
            under2.rect.y,
            under2.rect.h,
            under2.rect.w
        );
        assert!(
            approx(over1.rect.y, 0.0)
                && approx(over2.rect.y, 30.0)
                && approx(cell.rect.y, 60.0)
                && approx(cell.rect.h, 30.0)
                && approx(under1.rect.y, 90.0)
                && approx(under2.rect.y, 120.0)
                // w76fix finding 13(b): widths, previously unasserted
                // (a width-only regression passed this test). All five
                // boxes fill the table width, itself the td text's
                // max-content (23 chars at 16px = 220.8).
                && approx(table.rect.w, 220.8)
                && approx(over1.rect.w, 220.8)
                && approx(over2.rect.w, 220.8)
                && approx(cell.rect.w, 220.8)
                && approx(under1.rect.w, 220.8)
                && approx(under2.rect.w, 220.8),
            "table 8 rects wrong: {msg}"
        );
    }

    /// w76fix finding 5 (vertical-lr): the caption cursor anchors at
    /// the block-start content edge (`bl + pl` = 15), not the physical
    /// content top (`bt + pt` = 5). The caption's block-start margin
    /// (left, 20) puts its border at 35 (pre-fix 25); rows stack after
    /// the caption in block order, so the cell starts at the caption's
    /// right edge, 300 (pre-fix 290). The caption's inline position
    /// (y = 5, the physical content top) is unchanged. Not asserted
    /// here: the caption width (265) and grown table width (338.4)
    /// come from the pre-existing caption-measurement pipeline
    /// (caption sized before the table's block size is known), not
    /// from this anchor.
    #[test]
    fn w76fix_bordered_vertical_lr_caption_anchors_at_block_start() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px;\
             border-left:10px solid black;padding-left:5px;\
             border-top:2px solid black;padding-top:3px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin-left:20px\">caption</caption>\
             <tbody><tr><td>cell</td></tr></tbody></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(caption.rect.x, 35.0),
            "caption x {}, want 35 (10 border + 5 padding + 20 margin)",
            caption.rect.x
        );
        assert!(
            approx(caption.rect.y, 5.0),
            "caption y {}, want 5 (physical content top, unchanged)",
            caption.rect.y
        );
        let mut stack: Vec<&crate::LayoutBox> = vec![table];
        let mut cell = None;
        while let Some(n) = stack.pop() {
            if n.box_type == BoxType::TableCell {
                cell = Some(n);
                break;
            }
            stack.extend(n.children.iter());
        }
        let cell = cell.expect("cell box");
        assert!(
            approx(cell.rect.x, caption.rect.x + caption.rect.w),
            "cell x {}, want caption right edge {} (row stacks after caption)",
            cell.rect.x,
            caption.rect.x + caption.rect.w
        );
    }

    /// w76fix finding 5 (vertical-lr, no caption): the same anchor
    /// moves rows, not just captions — a bordered table with no
    /// caption starts its first cell at the block-start content edge
    /// (15), not the physical content top (5).
    #[test]
    fn w76fix_bordered_vertical_lr_row_anchors_at_block_start() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px;\
             border-left:10px solid black;padding-left:5px;\
             border-top:2px solid black;padding-top:3px\">\
             <tbody><tr><td>cell</td></tr></tbody></table>",
            "",
        );
        let table = table_box(&l);
        let mut stack: Vec<&crate::LayoutBox> = vec![table];
        let mut cell = None;
        while let Some(n) = stack.pop() {
            if n.box_type == BoxType::TableCell {
                cell = Some(n);
                break;
            }
            stack.extend(n.children.iter());
        }
        let cell = cell.expect("cell box");
        assert!(
            approx(cell.rect.x, 15.0),
            "cell x {}, want 15 (block-start content edge, not 5)",
            cell.rect.x
        );
    }

    /// w76fix finding 5 (vertical-rl mirror): the block-start edge is
    /// the right side, so the caption's right gap from the table's
    /// border-box right edge is `br + pr + mr` = 35 (pre-fix 25, from
    /// the physical content top). Rows stack after the caption toward
    /// the block end (left), so the cell's right edge meets the
    /// caption's left edge.
    #[test]
    fn w76fix_bordered_vertical_rl_caption_anchors_at_block_start() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-rl;width:300px;height:200px;\
             border-right:10px solid black;padding-right:5px;\
             border-top:2px solid black;padding-top:3px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin-right:20px\">caption</caption>\
             <tbody><tr><td>cell</td></tr></tbody></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(table.rect.w - (caption.rect.x + caption.rect.w), 35.0),
            "caption right gap {}, want 35 (10 border + 5 padding + 20 margin)",
            table.rect.w - (caption.rect.x + caption.rect.w)
        );
        let mut stack: Vec<&crate::LayoutBox> = vec![table];
        let mut cell = None;
        while let Some(n) = stack.pop() {
            if n.box_type == BoxType::TableCell {
                cell = Some(n);
                break;
            }
            stack.extend(n.children.iter());
        }
        let cell = cell.expect("cell box");
        assert!(
            approx(cell.rect.x + cell.rect.w, caption.rect.x),
            "cell right {}, want caption left edge {} (row stacks after caption)",
            cell.rect.x + cell.rect.w,
            caption.rect.x
        );
    }

    /// w76fix finding 12 (orthogonal auto-margin coverage): the
    /// vertical-lr cases are covered by
    /// `w76_orthogonal_caption_auto_block_margins_center` (both-auto)
    /// and `w76_orthogonal_caption_single_auto_block_margin`
    /// (one-auto top); this pins the remaining shapes, all along the
    /// table's inline axis (physical y) via the caption's own
    /// top/bottom margins.
    #[test]
    fn w76fix_orthogonal_caption_auto_margins_rl_negative_pct_bottom() {
        // vertical-rl both-auto centers exactly like vertical-lr:
        // the inline axis is physical-vertical in both modes.
        let l = layout_html(
            "<table style=\"writing-mode:vertical-rl;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin:auto 0\">caption</caption></table>",
            "",
        );
        let caption = caption_box(table_box(&l));
        assert!(
            approx(caption.rect.y, 75.0),
            "rl both-auto caption y {}, want 75",
            caption.rect.y
        );
        // Bottom placement with one auto margin: the auto side takes
        // the free space left after the fixed 10px side.
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin-top:auto;margin-bottom:10px;caption-side:bottom\">caption</caption></table>",
            "",
        );
        let caption = caption_box(table_box(&l));
        assert!(
            approx(caption.rect.y, 140.0),
            "bottom one-auto caption y {}, want 140 (200 - 50 - fixed 10)",
            caption.rect.y
        );
        // Negative free space clamps to zero: a 250px caption in a
        // 200px inline box sits at 0, not at a negative offset.
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:250px;margin:auto 0\">caption</caption></table>",
            "",
        );
        let caption = caption_box(table_box(&l));
        assert!(
            approx(caption.rect.y, 0.0),
            "negative-free caption y {}, want 0",
            caption.rect.y
        );
        // Percentage fixed side resolves against the inline size
        // (200): 10% = 20, and the auto side takes the rest, so the
        // caption sits at the fixed 20.
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"writing-mode:horizontal-tb;height:50px;margin-top:10%;margin-bottom:auto\">caption</caption></table>",
            "",
        );
        let caption = caption_box(table_box(&l));
        assert!(
            approx(caption.rect.y, 20.0),
            "pct-fixed caption y {}, want 20 (10% of 200 inline)",
            caption.rect.y
        );
    }

    /// w77fix finding 2 (fail-before probe): a definite-spec
    /// vertical table with a same-writing-mode caption and
    /// horizontal margins must keep the caption's margin box
    /// constrained to the definite block size — border box =
    /// definite block minus margins (300 - 20 - 20 = 260), with
    /// the block-start margin counted once (x = 20). Before the
    /// fix the caption measured 200 (the table's inline content
    /// width, via the auto-regime override) with the gate alone,
    /// or 160 (inline minus margins) with the override gated but
    /// no same-mode fast path; the fix adds the same-mode twin of
    /// the definite fast path so the caption fills the definite
    /// block width minus its margins. The no-row twin pins the
    /// table total: caption margin box 300 = definite block size.
    /// (With a row present the table legitimately grows past the
    /// definite size — a specified table block size is a minimum,
    //  CSS 2.1 §17.5.3 — so the row case pins caption geometry
    /// only, not the table total.) The auto-table twin (no
    /// definite block size) keeps the override: there the caption's
    /// border box is the inline content width.
    #[test]
    fn w77fix_definite_same_mode_caption_margins_constrained() {
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"height:50px;margin-left:20px;margin-right:20px\">caption</caption>\
             <tbody><tr><td>cell</td></tr></tbody></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(caption.rect.w, 260.0),
            "definite same-mode caption w {}, want 260 (300 - 20 - 20)",
            caption.rect.w
        );
        assert!(
            approx(caption.rect.x, 20.0),
            "definite same-mode caption x {}, want 20 (block-start margin once)",
            caption.rect.x
        );
        // No-row twin: table total is exactly the definite size.
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;width:300px;height:200px\">\
             <caption style=\"height:50px;margin-left:20px;margin-right:20px\">caption</caption></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(caption.rect.w, 260.0),
            "no-row definite caption w {}, want 260",
            caption.rect.w
        );
        assert!(
            approx(table.rect.w, 300.0),
            "no-row definite table w {}, want 300 (caption margin box = block size)",
            table.rect.w
        );
        // Auto twin: no definite block size, so the override still
        // applies and the caption fills the inline content width.
        let l = layout_html(
            "<table style=\"writing-mode:vertical-lr;height:200px\">\
             <caption style=\"height:50px;margin-left:20px;margin-right:20px\">caption</caption>\
             <tbody><tr><td>cell</td></tr></tbody></table>",
            "",
        );
        let table = table_box(&l);
        let caption = caption_box(table);
        assert!(
            approx(caption.rect.w, 200.0),
            "auto same-mode caption w {}, want 200 (inline content width)",
            caption.rect.w
        );
    }
}
