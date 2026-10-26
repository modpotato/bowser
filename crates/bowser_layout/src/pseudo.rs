//! `::before` / `::after` generated-content boxes (CSS Generated Content 3).
//!
//! When an element matches a `::before` / `::after` rule whose `content`
//! computes to a renderable string, box construction synthesizes a
//! pseudo box around that text: an [`BoxType::Inline`] wrapper carrying
//! the pseudo-element's computed style, containing one [`BoxType::Text`]
//! child — the same shape [`marker::attach_inside`](super::marker) uses
//! for inside markers. `::before` prepends, `::after` appends.
//!
//! ## Perf
//!
//! Box construction gates the per-element query on
//! [`Styler::has_generated_content_rules`], a stylesheet-level flag
//! computed once per `Styler`, so documents without `::before` /
//! `::after` rules pay ~zero in the hot loop — mirroring how the marker
//! path gates on `display: list-item`.
//!
//! ## Tables
//!
//! Pseudo boxes are ordinary forest nodes built before `layout_table`
//! runs, so `collect_table_grid_into`, `wrap_table_stray_content`, and
//! `is_table_caption` handle them with no table-layout changes. A
//! pseudo with `display: table-caption` maps to [`BoxType::Block`] (like
//! any element: `display_of` folds `table-caption` to
//! [`Display::Block`]) and its stored style keeps the raw
//! `table-caption` string, so the existing caption path collects it.

use bowser_css::{BEFORE_AFTER_ATOMS, ComputedStyle, CssValue, Styler, counter_adjustments, props};
use bowser_dom::{Document, NodeId, NodeKind};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use super::BoxType;
use super::FontMetrics;
use super::boxes::{Forest, direction_rtl, shared_text};
use super::style::{
    Display, StyleData, display_of, font_family_name, resolve_font_size, resolve_line_height,
    text_transform_of, white_space_of,
};

/// Which generated-content pseudo-element to query / attach.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PseudoKind {
    Before,
    After,
}

impl PseudoKind {
    /// The pseudo-element name as it appears in selectors (and in
    /// [`Styler::pseudo_style_for`]'s contract).
    pub(crate) fn name(self) -> &'static str {
        match self {
            PseudoKind::Before => "before",
            PseudoKind::After => "after",
        }
    }
}

/// A renderable `::before` / `::after`: the pseudo-element's computed
/// style (shared by reference count via
/// [`Styler::pseudo_style_for_shared`], so identical pseudos share one
/// allocation) plus the `content` text to synthesize.
pub(crate) struct PseudoContent {
    pub style: Rc<ComputedStyle>,
    pub text: String,
}

/// One counter scope frame: counter name -> value. Frames nest
/// root-first; see [`CounterScopes`].
pub(crate) type CounterFrame = HashMap<String, i64>;

/// Counter scope stack threaded through box construction (see
/// `build_box_tree_rooted`).
///
/// CSS counters are stateful across siblings: an `li`'s
/// `counter-increment` must be visible to the *next* `li`'s
/// `::before`. A per-query ancestor fold cannot see sibling
/// increments, so box construction maintains this stack instead:
/// each entered element pops the entries of subtrees already left
/// (truncating the frames they pushed) and applies its own
/// `counter-reset` (pushing one new frame) / `counter-increment`
/// (mutating the innermost scope) in spec order.
///
/// Frames are shared (`Rc<RefCell<..>>`) so an increment to a parent
/// frame is visible to later siblings; each entry records the frame
/// count before its element's reset, so leaving the subtree truncates
/// exactly the frames it pushed. `display: contents` elements enter
/// (they still scope counters) but push no box; entries are keyed by
/// DOM node so their subtree exit is detected the same way.
pub(crate) struct CounterScopes {
    /// Shared scope frames, root-first.
    frames: Vec<Rc<RefCell<CounterFrame>>>,
    /// Per-entered-element: the DOM node and the frame count before
    /// its `counter-reset` pushed (for truncating on subtree exit).
    entries: Vec<(NodeId, usize)>,
}

impl CounterScopes {
    pub(crate) fn new() -> Self {
        CounterScopes {
            frames: vec![Rc::new(RefCell::new(CounterFrame::new()))],
            entries: Vec::new(),
        }
    }

    /// Enter `node`'s counter scope: pop the entries of subtrees
    /// already left, apply `node`'s own adjustments to the shared
    /// frames, and hand back a snapshot of the current scopes for
    /// `::before` / `::after` queries. The snapshot clones the small
    /// frame maps; resolution itself is read-only.
    pub(crate) fn enter(
        &mut self,
        doc: &Document,
        node: NodeId,
        style: &ComputedStyle,
    ) -> Vec<CounterFrame> {
        while let Some(&(scope_dom, base_len)) = self.entries.last() {
            if is_dom_ancestor_or_self(doc, scope_dom, node) {
                break;
            }
            self.entries.pop();
            self.frames.truncate(base_len);
        }
        let base_len = self.frames.len();
        enter_counter_scope(style, &mut self.frames);
        self.entries.push((node, base_len));
        self.frames.iter().map(|f| f.borrow().clone()).collect()
    }

    /// Snapshot the scopes as `node`'s `::after` observes them: pop the
    /// entries of `node`'s already-processed descendants (truncating the
    /// `counter-reset` frames they pushed — a descendant's reset scope
    /// is not in scope for the parent's `::after`), keeping `node`'s own
    /// entry: its `counter-reset` frame stays, with every in-scope
    /// increment applied, exactly the post-children state CSS Lists 3
    /// gives the last child. The entry itself is left for the regular
    /// pop-on-next-`enter`; increments that targeted ancestor frames
    /// persist there either way. Callers invoke this once per element
    /// (when its deferred-`::after` marker pops in box construction),
    /// so `node` is always the entries top's ancestor-or-self here.
    pub(crate) fn snapshot_after(&mut self, node: NodeId) -> Vec<CounterFrame> {
        while let Some(&(scope_dom, base_len)) = self.entries.last() {
            if scope_dom == node {
                break;
            }
            self.entries.pop();
            self.frames.truncate(base_len);
        }
        debug_assert!(self.entries.last().is_some_and(|&(d, _)| d == node));
        self.frames.iter().map(|f| f.borrow().clone()).collect()
    }
}

/// Apply `style`'s computed `counter-reset` (pushing one new scope
/// frame) and `counter-increment` (mutating the innermost scope) to
/// `frames`, in spec order: an element's own reset applies before its
/// increment, so the element's `::before` sees its own increments.
/// Increments target the innermost frame holding the counter, creating
/// it there at 0 when absent.
fn enter_counter_scope(style: &ComputedStyle, frames: &mut Vec<Rc<RefCell<CounterFrame>>>) {
    let resets = counter_adjustments(style.get_by_index(props::COUNTER_RESET), 0);
    if !resets.is_empty() {
        let mut frame = CounterFrame::new();
        for (name, value) in resets {
            frame.insert(name, value);
        }
        frames.push(Rc::new(RefCell::new(frame)));
    }
    for (name, delta) in counter_adjustments(style.get_by_index(props::COUNTER_INCREMENT), 1) {
        let target = frames
            .iter()
            .rev()
            .find(|f| f.borrow().contains_key(&name))
            .or_else(|| frames.last());
        if let Some(frame) = target {
            let mut map = frame.borrow_mut();
            match map.get_mut(&name) {
                Some(v) => *v = v.saturating_add(delta),
                // Created at 0, then incremented: the net is `delta`.
                None => {
                    map.insert(name, delta);
                }
            }
        }
    }
}

/// Whether `anc` is `node` itself or a DOM ancestor of `node`.
fn is_dom_ancestor_or_self(doc: &Document, anc: NodeId, node: NodeId) -> bool {
    let mut cursor = Some(node);
    while let Some(n) = cursor {
        if n == anc {
            return true;
        }
        cursor = doc.parent(n);
    }
    false
}

/// Element-scoped resolution of a pseudo-element's computed `content`
/// to renderable text, or `None` when `content` is `normal` / `none` /
/// absent (no box).
///
/// `<string>` items resolve verbatim; `attr()` reads the originating
/// element's attribute (ASCII case-insensitive for HTML elements, via
/// [`Document::get_attribute`]), falling back to the parsed fallback
/// or the empty string when the attribute is absent; `counter()` /
/// `counters()` resolve against `scopes` — the counter-scope snapshot
/// box construction threaded for the originating element (see
/// [`CounterScopes`]) — rendering in the requested counter style
/// (`decimal` when omitted or unknown). When no scope stack was
/// threaded (no stylesheet counter rule), an ancestor fold covers
/// inline `style` adjustments for this element's subtree (see
/// [`counter_scopes_ancestor_fold`]).
/// `var()` inside `content` already resolved during the pseudo cascade
/// ([`Styler::pseudo_winners`]), so no extra substitution happens
/// here.
pub(crate) fn resolve_content_text(
    styler: &Styler,
    doc: &Document,
    node: NodeId,
    style: &ComputedStyle,
    scopes: &[CounterFrame],
) -> Option<String> {
    let value = style.get_by_index(props::CONTENT)?;
    // Fallback: no scope stack was threaded, but the content uses a
    // counter — fold the ancestor chain so inline `style` adjustments
    // still resolve (the stylesheet gate found no counter rules, so
    // only inline adjustments can exist).
    let owned;
    let scopes: &[CounterFrame] = if scopes.is_empty() && content_uses_counters(value) {
        owned = counter_scopes_ancestor_fold(styler, doc, node);
        &owned
    } else {
        scopes
    };
    match value {
        CssValue::Keyword(k) if matches!(k.as_str(), "normal" | "none") => None,
        CssValue::String(s) => Some(s.clone()),
        CssValue::List(items) if !items.is_empty() => {
            let mut out = String::new();
            for item in items {
                out.push_str(&resolve_content_item(item, doc, node, Some(scopes)));
            }
            Some(out)
        }
        item @ (CssValue::Attr { .. } | CssValue::Counter { .. }) => {
            Some(resolve_content_item(item, doc, node, Some(scopes)))
        }
        _ => None,
    }
}

/// Resolve one `<content-list>` item to text. Shapes outside the
/// `content` grammar cannot reach a computed style; they degrade to
/// the empty string rather than panicking.
fn resolve_content_item(
    item: &CssValue,
    doc: &Document,
    node: NodeId,
    scopes: Option<&[CounterFrame]>,
) -> String {
    match item {
        CssValue::String(s) => s.clone(),
        CssValue::Attr(attr) => match doc.get_attribute(node, &attr.name) {
            Some(v) => v.to_owned(),
            None => match &attr.fallback {
                Some(fb) => resolve_content_item(fb, doc, node, scopes),
                None => String::new(),
            },
        },
        CssValue::Counter(counter) => {
            let values = counter_values(&counter.name, scopes);
            match &counter.separator {
                // `counters()`: every in-scope value, outermost first.
                Some(sep) => values
                    .iter()
                    .map(|&v| render_counter_value(v, counter.style.as_deref()))
                    .collect::<Vec<_>>()
                    .join(sep),
                // `counter()`: the innermost in-scope value.
                None => values
                    .last()
                    .map(|&v| render_counter_value(v, counter.style.as_deref()))
                    .unwrap_or_default(),
            }
        }
        // A nested list (only reachable inside an `attr()` fallback,
        // whose items never went through `fold_content_value`):
        // resolve each item and concatenate.
        CssValue::List(items) => {
            let mut out = String::new();
            for item in items {
                out.push_str(&resolve_content_item(item, doc, node, scopes));
            }
            out
        }
        _ => String::new(),
    }
}

/// Whether a computed `content` value holds a `counter()` /
/// `counters()` reference (directly or inside an `attr()` fallback).
fn content_uses_counters(value: &CssValue) -> bool {
    match value {
        CssValue::Counter { .. } => true,
        CssValue::Attr(attr) => attr.fallback.as_deref().is_some_and(content_uses_counters),
        CssValue::List(items) => items.iter().any(content_uses_counters),
        _ => false,
    }
}

/// In-scope values for counter `name`, outermost scope first; a
/// missing counter reads as 0 (matching browser behavior for
/// `counter()` on a name no `counter-reset` ever created).
fn counter_values(name: &str, scopes: Option<&[CounterFrame]>) -> Vec<i64> {
    let mut out: Vec<i64> = scopes
        .map(|frames| frames.iter().filter_map(|f| f.get(name).copied()).collect())
        .unwrap_or_default();
    if out.is_empty() {
        out.push(0);
    }
    out
}

/// Ancestor-fold fallback for counter scopes: one frame per
/// `counter-reset` on the ancestor chain (root-first, `node`
/// included), each mapping counter names to values, with
/// `counter-increment`s applied in document order.
///
/// Only used when box construction threaded no scope stack (no
/// stylesheet counter rule): it sees the ancestor chain's inline
/// adjustments but *not* preceding siblings' increments, so it is
/// exact only when increments come from stylesheets (which imply the
/// stack). The stack path ([`CounterScopes`]) is the primary
/// resolution; this keeps inline-`style` adjustments working when
/// they are the only counter rules in the document.
fn counter_scopes_ancestor_fold(
    styler: &Styler,
    doc: &Document,
    node: NodeId,
) -> Vec<CounterFrame> {
    // Ancestor chain, root-first.
    let mut chain = Vec::new();
    let mut cursor = Some(node);
    while let Some(n) = cursor {
        chain.push(n);
        cursor = doc.parent(n);
    }
    let mut scopes: Vec<CounterFrame> = vec![HashMap::new()];
    for &anc in chain.iter().rev() {
        if !matches!(doc.kind(anc), NodeKind::Element(_)) {
            continue;
        }
        let style = styler.style_for_shared(anc);
        // Resets first: one new scope frame per element carrying
        // `counter-reset` adjustments (`none` adjusts nothing and
        // creates no scope).
        let resets = counter_adjustments(style.get_by_index(props::COUNTER_RESET), 0);
        if !resets.is_empty() {
            let mut frame = HashMap::new();
            for (name, value) in resets {
                frame.insert(name, value);
            }
            scopes.push(frame);
        }
        // Then increments, innermost scope first.
        for (name, delta) in counter_adjustments(style.get_by_index(props::COUNTER_INCREMENT), 1) {
            match scopes.iter_mut().rev().find(|f| f.contains_key(&name)) {
                Some(frame) => {
                    let v = frame.get_mut(&name).expect("key just found");
                    *v = v.saturating_add(delta);
                }
                None => {
                    scopes.last_mut().expect("root scope").insert(name, delta);
                }
            }
        }
    }
    scopes
}

/// Render a counter value in a counter style (CSS Lists 3 §7 /
/// css-counter-styles-3): `decimal` when omitted; unknown or
/// non-numeric styles degrade to `decimal`.
///
/// The numeral renderers are local copies of `marker`'s: `marker.rs`
/// is outside this worker's owned files, so the pseudo-content path
/// cannot reuse them.
fn render_counter_value(value: i64, style: Option<&str>) -> String {
    let s = style.unwrap_or("decimal");
    if s.eq_ignore_ascii_case("decimal") {
        value.to_string()
    } else if s.eq_ignore_ascii_case("decimal-leading-zero") {
        format!("{value:02}")
    } else if s.eq_ignore_ascii_case("lower-roman") {
        to_roman(value.max(1)).to_lowercase()
    } else if s.eq_ignore_ascii_case("upper-roman") {
        to_roman(value.max(1))
    } else if s.eq_ignore_ascii_case("lower-latin") || s.eq_ignore_ascii_case("lower-alpha") {
        to_alpha(value.max(1)).to_lowercase()
    } else if s.eq_ignore_ascii_case("upper-latin") || s.eq_ignore_ascii_case("upper-alpha") {
        to_alpha(value.max(1))
    } else if s.eq_ignore_ascii_case("lower-greek") {
        to_greek(value.max(1))
    } else {
        value.to_string()
    }
}

/// Uppercase Roman numeral for 1..=3999; anything else degrades to the
/// decimal rendering (also bounds the output).
fn to_roman(n: i64) -> String {
    if !(1..=3999).contains(&n) {
        return n.to_string();
    }
    const TABLE: &[(i64, &str)] = &[
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut n = n;
    let mut out = String::new();
    for &(v, s) in TABLE {
        while n >= v {
            out.push_str(s);
            n -= v;
        }
    }
    out
}

/// Uppercase alphabetic counter: 1 -> A, 26 -> Z, 27 -> AA, ….
fn to_alpha(n: i64) -> String {
    let mut n = n;
    let mut out = Vec::new();
    while n > 0 {
        n -= 1;
        out.push((b'A' + (n % 26) as u8) as char);
        n /= 26;
    }
    out.iter().rev().collect()
}

/// Lowercase Greek counter: 1 -> α, …, 24 -> ω, 25 -> αα, ….
fn to_greek(n: i64) -> String {
    const GREEK: &[char] = &[
        'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ',
        'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
    ];
    let mut n = n;
    let mut out = Vec::new();
    while n > 0 {
        n -= 1;
        out.push(GREEK[(n % 24) as usize]);
        n /= 24;
    }
    out.iter().rev().collect()
}

/// Query `node`'s `::before` / `::after`: the pseudo-element's computed
/// style ([`Styler::pseudo_style_for_shared`] over [`BEFORE_AFTER_ATOMS`],
/// modeled on `marker_style_for_shared`) plus its renderable `content`
/// text (element-scoped resolution — `attr()` reads `doc`, `counter()`
/// reads `scopes`, the [`CounterScopes`] snapshot box construction
/// threaded for `node`; see [`resolve_content_text`]), or `None` when
/// nothing renders. Callers gate on
/// [`Styler::has_generated_content_rules`] first.
pub(crate) fn query_pseudo_content(
    styler: &Styler,
    doc: &Document,
    node: NodeId,
    kind: PseudoKind,
    scopes: &[CounterFrame],
) -> Option<PseudoContent> {
    let style = styler.pseudo_style_for_shared(node, kind.name(), BEFORE_AFTER_ATOMS)?;
    let text = resolve_content_text(styler, doc, node, &style, scopes)?;
    Some(PseudoContent { style, text })
}

/// `node`'s `::after` computed style, or `None` when no rule targets
/// it. Split out of [`query_pseudo_content`] so box construction can
/// resolve the style at element entry (counter-independent and
/// memoized, via [`Styler::pseudo_style_for_shared`]) and the `content`
/// text later, against the post-children snapshot from
/// [`CounterScopes::snapshot_after`] — `::after` is the last child, so
/// its `counter()` values must reflect the whole subtree.
pub(crate) fn query_after_style(styler: &Styler, node: NodeId) -> Option<Rc<ComputedStyle>> {
    styler.pseudo_style_for_shared(node, PseudoKind::After.name(), BEFORE_AFTER_ATOMS)
}

/// Synthesize the pseudo box for `content` as a child of the element box
/// `idx`: an inline wrapper carrying the pseudo style, holding one text
/// child. `::before` prepends, `::after` appends. Mirrors
/// `marker::attach_inside`.
///
/// `origin_style` is the originating element's computed style, used
/// only to keep the pseudo's own `display` distinct from the
/// originating element's: the merged pseudo style starts from the
/// element's style (see `marker_style_for`) and `display` is not
/// inherited, so a raw-value difference means the pseudo cascade set
/// it. (Comparing the mapped [`Display`] would conflate e.g.
/// `table-caption` on the pseudo with `block` on a `div` origin.)
/// Note the sibling's `pseudo_style_for` currently resets `display` to
/// its initial `inline` (`display` is in no pseudo's allowed set), so
/// in practice the pseudo display is `inline` and the differing branch
/// just re-derives that; the mapping stays correct if the allowed set
/// ever grows.
/// `display: none` on the pseudo suppresses the box entirely.
///
/// The wrapper inherits the origin's box data, then takes the pseudo's
/// own winners for the inherited text properties — `font-size` and
/// `line-height` resolved against the forest's root size / viewport
/// (carried by the [`Forest`]), plus `white-space`, `text-transform`,
/// and `direction` read straight from the pseudo style
/// (`pseudo_style_for` seeds inheritance from the origin, so unset
/// props read as inherited here). The text child runs through
/// [`shared_text`] with the pseudo's own `white-space` /
/// `text-transform`, exactly like DOM text in pass 1: whitespace-only
/// collapses drop the text child (the wrapper stays, preserving the
/// spec's empty-inline-box for `content: ""`).
pub(crate) fn attach(
    forest: &mut Forest,
    idx: usize,
    kind: PseudoKind,
    content: PseudoContent,
    origin_style: &ComputedStyle,
    fonts: &dyn FontMetrics,
) {
    let pseudo_display = if content.style.get_str_by_index(props::DISPLAY)
        != origin_style.get_str_by_index(props::DISPLAY)
    {
        display_of(&content.style)
    } else {
        // Unset by the pseudo rule: the initial value, `inline`.
        Display::Inline
    };
    if pseudo_display == Display::None {
        return;
    }
    // `display_of` already folds `table-caption` / `flow-root` to
    // `Block`; every other pseudo display renders as an inline-level
    // box (the pseudo initial value).
    let box_type = match pseudo_display {
        Display::Block => BoxType::Block,
        _ => BoxType::Inline,
    };
    // Inherited properties come from the originating element's box
    // data (correct per spec); everything else takes initial values —
    // the same base `marker::attach_inside` uses. The pseudo's own
    // winners for the inherited text properties apply on top, like
    // `marker::attach_inside` resolves the marker's `font-size` /
    // `line-height`.
    let origin_data = &forest.nodes[idx].data;
    let root_fs = forest.root_font_size;
    let (vp_w, vp_h) = (forest.vp_w, forest.vp_h);
    let font_size = resolve_font_size(
        content.style.get_by_index(props::FONT_SIZE),
        origin_data.font_size,
        root_fs,
        vp_w,
        vp_h,
    );
    let mut data = StyleData::inherit(origin_data);
    data.display = pseudo_display;
    data.font_size = font_size;
    data.line_height = resolve_line_height(
        content.style.get_by_index(props::LINE_HEIGHT),
        font_size,
        root_fs,
        vp_w,
        vp_h,
        fonts,
        font_family_name(&content.style),
    );
    data.white_space = white_space_of(&content.style);
    data.text_transform = text_transform_of(&content.style);
    data.rtl = direction_rtl(&content.style, origin_data.rtl);
    let (ws, tt) = (data.white_space, data.text_transform);
    // The pseudo style is already reference-counted (see
    // `query_pseudo_content`): share it, don't clone the style.
    let pseudo_idx = forest.add_node(
        box_type,
        Some(Rc::clone(&content.style)),
        data,
        None,
        Some(idx),
    );
    // Whitespace collapse + `text-transform` with the pseudo's own
    // `white-space` / `text-transform` (mirrors pass 1's `shared_text`
    // for DOM text). Empty output drops the text child only: the
    // childless Inline wrapper stays, preserving the spec's
    // empty-inline-box for `content: ""`.
    let source: Arc<str> = Arc::from(content.text);
    if let Some(text) = shared_text(&source, ws, tt) {
        let text_data = StyleData::inherit(&forest.nodes[pseudo_idx].data);
        forest.add_node(BoxType::Text, None, text_data, Some(text), Some(pseudo_idx));
    }
    if kind == PseudoKind::Before {
        // Prepend: `add_node` appends, so rotate the pseudo node to the
        // front (same trick as `marker::attach_inside`).
        let children = &mut forest.nodes[idx].children;
        if let Some(pos) = children.iter().rposition(|&c| c == pseudo_idx) {
            children[..=pos].rotate_right(1);
        }
    }
}
